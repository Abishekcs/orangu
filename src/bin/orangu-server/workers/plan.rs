// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Dividing a node's layers between itself and its workers.
//!
//! A pipeline gains nothing in speed from being split — every layer still
//! runs once per token, one after another — so the point of spreading a
//! model is room: each node holds the share of the layers its memory is
//! for. Shares are therefore sized by memory (the node's, or for a worker
//! its whole subtree's), and cut by the layers' own bytes rather than
//! their count — the same proportional idea `engine::placement` applies to
//! the GPUs of one machine.

use super::protocol::Capacity;
use crate::config::LocalLayers;
use std::ops::Range;

/// The share of a device's memory weights may take: the rest is the KV
/// cache, scratch, and the driver's own.
const DEVICE_SHARE: f64 = 0.9;

/// The share of RAM weights may take: the rest is the KV cache, the
/// process, and everything else on the machine.
const HOST_SHARE: f64 = 0.8;

/// The bytes of weights a node with `device_bytes` of device memory (`0`
/// for none) and `host_bytes` of RAM can hold: what its layers live in —
/// the device when it has one, the host otherwise — less room for the rest.
pub fn budget(device_bytes: u64, host_bytes: u64) -> u64 {
    if device_bytes > 0 {
        (device_bytes as f64 * DEVICE_SHARE) as u64
    } else {
        (host_bytes as f64 * HOST_SHARE) as u64
    }
}

/// A worker's weight in a plan: its whole subtree's budget.
pub fn subtree_budget(capacity: &Capacity) -> u64 {
    capacity.subtree_budget_bytes
}

/// Splits `range` into consecutive pieces, one per weight and in order, so
/// that each piece's share of the layers' bytes (`layer_bytes`, indexed by
/// layer) is as close as the layer boundaries allow to its share of the
/// weights. A piece with a positive weight gets at least one layer while
/// there are layers to go round; a zero weight gets none.
///
/// A cut is only ever made before a layer `allowed` accepts (the model's
/// `split_allowed`): the nearest allowed one to the target, which may leave
/// a piece empty, or give one piece every layer that cannot be separated.
pub fn split_range(
    range: Range<usize>,
    layer_bytes: &[u64],
    weights: &[f64],
    allowed: &dyn Fn(usize) -> bool,
) -> Vec<Range<usize>> {
    let n = range.len();
    // cumulative[j]: the bytes of the first `j` layers of the range.
    let mut cumulative = vec![0u64; n + 1];
    for j in 0..n {
        cumulative[j + 1] = cumulative[j] + layer_bytes.get(range.start + j).copied().unwrap_or(0);
    }
    let total_bytes = cumulative[n] as f64;
    let total_weight: f64 = weights.iter().filter(|w| **w > 0.0).sum();
    let positive: Vec<usize> = (0..weights.len()).filter(|i| weights[*i] > 0.0).collect();
    let mut pieces = Vec::with_capacity(weights.len());
    let mut start = 0usize;
    let mut weight_so_far = 0.0;
    for (i, &weight) in weights.iter().enumerate() {
        if weight <= 0.0 || start >= n {
            pieces.push(range.start + start..range.start + start);
            continue;
        }
        weight_so_far += weight;
        let after = positive.iter().filter(|&&p| p > i).count();
        let end = if after == 0 {
            n
        } else {
            let target = weight_so_far / total_weight * total_bytes;
            // One layer for this piece, and one each for as many of the
            // pieces after it as there are layers left for: the earlier
            // pieces are served first when there are too few to go round.
            let reserved = after.min(n - start - 1);
            let mut latest = n - reserved;
            // And a cut the model allows left for each of the pieces after it,
            // where there are enough: a model that may only be cut early (Gemma
            // 4's layers that share an earlier layer's KV cache) gave the rest
            // to one piece and left the next with nothing.
            // (The last piece ends at the end, always allowed: the pieces after
            // this one need one cut fewer than there are of them.)
            let later: Vec<usize> = (start + 1..n)
                .filter(|j| allowed(range.start + *j))
                .collect();
            if after > 0 && later.len() >= after {
                latest = latest.min(later[later.len() - after]);
            }
            let earliest = (start + 1).min(latest);
            let cut = |j: &usize| *j == n || allowed(range.start + *j);
            let closest = |candidates: &mut dyn Iterator<Item = usize>| {
                candidates.min_by(|a, b| {
                    let da = (cumulative[*a] as f64 - target).abs();
                    let db = (cumulative[*b] as f64 - target).abs();
                    da.total_cmp(&db)
                })
            };
            // An allowed cut within the room this piece has, or — when none
            // is — the closest allowed one anywhere after it. Latest first:
            // of two cuts equally close, the later one keeps the layers
            // nearer the top of the tree, a hop fewer from where every
            // forward starts and ends.
            closest(&mut (earliest..=latest).rev().filter(cut))
                .or_else(|| closest(&mut (start..=n).rev().filter(cut)))
                .unwrap_or(n)
        };
        pieces.push(range.start + start..range.start + end);
        start = end;
    }
    pieces
}

/// The share of `range` a node keeps and each of its workers gets, in that
/// order: the node's by `local_layers` (`auto` sizes it by `own_budget`
/// like any worker's), the workers' by their subtrees' budgets.
pub fn plan_shares(
    range: Range<usize>,
    layer_bytes: &[u64],
    local_layers: LocalLayers,
    own_budget: u64,
    worker_budgets: &[u64],
    allowed: &dyn Fn(usize) -> bool,
) -> (Range<usize>, Vec<Range<usize>>) {
    if worker_budgets.is_empty() {
        return (range, Vec::new());
    }
    let workers: Vec<f64> = worker_budgets.iter().map(|b| *b as f64).collect();
    match local_layers {
        LocalLayers::Count(count) => {
            // The asked-for count, or the nearest allowed cut to it.
            let want = range.start + count.min(range.len());
            let end = (range.start..=range.end)
                .filter(|j| *j == range.end || *j == range.start || allowed(*j))
                .min_by_key(|j| j.abs_diff(want))
                .unwrap_or(range.end);
            let local = range.start..end;
            let rest = split_range(local.end..range.end, layer_bytes, &workers, allowed);
            (local, rest)
        }
        LocalLayers::Auto => {
            let mut weights = vec![own_budget as f64];
            weights.extend(workers);
            let mut pieces = split_range(range, layer_bytes, &weights, allowed);
            let local = pieces.remove(0);
            (local, pieces)
        }
    }
}

/// Shares by speed: the bytes each node — or worker subtree — should
/// take for every part of a pipeline to take about as long, its `rate`
/// (bytes of weights a second, `super::speed`) over the sum, but never more
/// than its `budget` holds; what a budget turns away goes to the others the
/// same way. Weights for [`plan_shares`], in the same order. `None` — shares
/// by memory — when a rate is not known, or when the budgets together
/// cannot hold `total_bytes`, and every node is full anyway.
pub fn by_speed(budgets: &[u64], rates: &[f64], total_bytes: u64) -> Option<Vec<u64>> {
    let n = budgets.len();
    if n != rates.len() || rates.iter().any(|r| r.is_nan() || *r <= 0.0) {
        return None;
    }
    if budgets.iter().sum::<u64>() < total_bytes {
        return None;
    }
    let mut target = vec![0.0f64; n];
    let mut open: Vec<usize> = (0..n).collect();
    let mut left = total_bytes as f64;
    while !open.is_empty() {
        let sum: f64 = open.iter().map(|i| rates[*i]).sum();
        let (over, under): (Vec<usize>, Vec<usize>) = open
            .iter()
            .partition(|i| left * rates[**i] / sum > budgets[**i] as f64);
        if over.is_empty() {
            for i in under {
                target[i] = left * rates[i] / sum;
            }
            break;
        }
        for i in over {
            target[i] = budgets[i] as f64;
            left -= budgets[i] as f64;
        }
        open = under;
    }
    Some(target.iter().map(|t| (t.round() as u64).max(1)).collect())
}

/// The order of the workers with `rates` whose plan `cost`s least — a
/// plan's cost being, say, the time of a decode step through it. Order
/// matters where a model cannot be cut anywhere: what cannot be divided
/// goes to whichever share starts at the last cut. Every order of up to
/// six workers is tried, of more the given one and both by speed; of
/// equal costs the given order wins.
pub fn best_order(rates: &[f64], cost: &dyn Fn(&[usize]) -> f64) -> Vec<usize> {
    let n = rates.len();
    let mut orders: Vec<Vec<usize>> = vec![(0..n).collect()];
    if n <= 6 {
        orders.extend(permutations(n));
    } else {
        let mut by_speed: Vec<usize> = (0..n).collect();
        by_speed.sort_by(|a, b| rates[*a].total_cmp(&rates[*b]));
        orders.push(by_speed.clone());
        by_speed.reverse();
        orders.push(by_speed);
    }
    let mut best = (f64::INFINITY, (0..n).collect());
    for order in orders {
        let c = cost(&order);
        if c < best.0 {
            best = (c, order);
        }
    }
    best.1
}

/// Every order of `0..n`.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for shorter in permutations(n - 1) {
        for at in 0..=shorter.len() {
            let mut order = shorter.clone();
            order.insert(at, n - 1);
            out.push(order);
        }
    }
    out
}

/// The bytes of weights in `layers`.
pub fn bytes_of(layers: &Range<usize>, layer_bytes: &[u64]) -> u64 {
    layers.clone().filter_map(|il| layer_bytes.get(il)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens(pieces: &[Range<usize>]) -> Vec<usize> {
        pieces.iter().map(|p| p.len()).collect()
    }

    const EVEN: [u64; 64] = [1; 64];

    fn any(_: usize) -> bool {
        true
    }
    const ANY: fn(usize) -> bool = any;

    /// Cuts land only where the model allows them: here, like a Gemma 4
    /// file whose last layers share the KV of layers 5 and 6, nowhere from
    /// 6 to 11.
    #[test]
    fn cuts_land_only_where_allowed() {
        let allowed = |at: usize| at <= 5;
        let pieces = split_range(0..12, &EVEN, &[1.0, 1.0], &allowed);
        assert_eq!(pieces, [0..5, 5..12]);
        let pieces = split_range(0..12, &EVEN, &[1.0, 1.0, 1.0], &allowed);
        assert_eq!(pieces.last().unwrap().end, 12);
        assert!(
            pieces
                .iter()
                .all(|p| p.start == 0 || p.start <= 5 || p.is_empty())
        );
        // Between an empty share and all of an inseparable block, equally
        // far from the target, the node keeps the block itself: a worker
        // that would only relay it is a hop for nothing.
        assert_eq!(
            split_range(5..12, &EVEN, &[1.0, 1.0], &allowed),
            [5..12, 12..12]
        );
        // A share that would have to cut where it cannot takes nothing or
        // everything, whichever the target is nearer.
        let (local, rest) = plan_shares(0..12, &EVEN, LocalLayers::Count(9), 1, &[1], &allowed);
        assert_eq!(local, 0..12);
        assert!(rest.len() == 1 && rest[0].is_empty());
    }

    #[test]
    fn pieces_are_consecutive_and_proportional() {
        assert_eq!(
            split_range(10..50, &EVEN, &[1.0, 3.0], &ANY),
            [10..20, 20..50]
        );
        let pieces = split_range(0..10, &EVEN, &[1.0, 1.0, 1.0], &ANY);
        assert_eq!(lens(&pieces).iter().sum::<usize>(), 10);
        assert!(lens(&pieces).iter().all(|l| (3..=4).contains(l)));
        assert_eq!(pieces.last().unwrap().end, 10);
    }

    /// Shares follow bytes, not layer counts: a heavy layer counts for what
    /// it weighs.
    #[test]
    fn heavy_layers_count_for_their_bytes() {
        // Layer 0 weighs as much as the other three together.
        let bytes = [30, 10, 10, 10];
        assert_eq!(split_range(0..4, &bytes, &[1.0, 1.0], &ANY), [0..1, 1..4]);
    }

    #[test]
    fn a_small_weight_still_gets_a_layer_and_zero_gets_none() {
        assert_eq!(lens(&split_range(0..8, &EVEN, &[100.0, 1.0], &ANY)), [7, 1]);
        assert_eq!(
            lens(&split_range(0..8, &EVEN, &[1.0, 0.0, 1.0], &ANY)),
            [4, 0, 4]
        );
        // More pieces than layers: the extras get nothing.
        assert_eq!(
            lens(&split_range(0..2, &EVEN, &[1.0, 1.0, 1.0], &ANY)),
            [1, 1, 0]
        );
        assert_eq!(lens(&split_range(0..0, &EVEN, &[1.0], &ANY)), [0]);
        assert_eq!(lens(&split_range(0..5, &EVEN, &[0.0, 0.0], &ANY)), [0, 0]);
    }

    #[test]
    fn the_node_keeps_its_share_first() {
        let (local, workers) = plan_shares(0..12, &EVEN, LocalLayers::Auto, 1, &[1, 1], &ANY);
        assert_eq!((local, workers), (0..4, vec![4..8, 8..12]));
        let (local, workers) = plan_shares(0..12, &EVEN, LocalLayers::Count(0), 9, &[1, 3], &ANY);
        assert_eq!((local, workers), (0..0, vec![0..3, 3..12]));
        let (local, workers) = plan_shares(4..8, &EVEN, LocalLayers::Count(10), 1, &[1], &ANY);
        assert_eq!(local, 4..8);
        assert_eq!(workers.len(), 1);
        assert!(workers[0].is_empty());
        // No workers: everything stays.
        assert_eq!(
            plan_shares(0..5, &EVEN, LocalLayers::Count(0), 1, &[], &ANY),
            (0..5, vec![])
        );
    }

    /// Twelve even layers that may only be cut up to layer 4: the rest goes
    /// to one worker whatever the rates, and the best order gives it to the
    /// fastest. With every cut allowed, the given order stands.
    #[test]
    fn what_cannot_be_divided_goes_to_the_fastest() {
        let rates = [1.0, 4.0];
        let plan = |allowed: &dyn Fn(usize) -> bool, order: &[usize]| {
            let weights: Vec<u64> = order.iter().map(|i| (rates[*i] * 10.0) as u64).collect();
            plan_shares(0..12, &EVEN, LocalLayers::Auto, 10, &weights, allowed)
        };
        let cost = |allowed: &dyn Fn(usize) -> bool, order: &[usize]| {
            let (local, shares) = plan(allowed, order);
            local.len() as f64
                + shares
                    .iter()
                    .zip(order)
                    .map(|(s, i)| s.len() as f64 / rates[*i])
                    .sum::<f64>()
        };
        let head = |at: usize| at <= 4;
        let order = best_order(&rates, &|o| cost(&head, o));
        let (_, shares) = plan(&head, &order);
        let tail = order[shares.iter().position(|s| s.end == 12).unwrap()];
        assert_eq!(tail, 1, "{order:?} {shares:?}");
        assert_eq!(best_order(&rates, &|o| cost(&ANY, o)), vec![0, 1]);
        assert_eq!(permutations(3).len(), 6);
    }

    /// A model that may only be cut up to layer 4 still gives every piece
    /// a layer, however the weights lean: the cuts it allows are kept for
    /// the pieces after.
    #[test]
    fn early_cuts_still_give_every_piece_a_layer() {
        let early = |at: usize| at <= 4;
        let pieces = split_range(0..12, &EVEN, &[10.0, 1.0, 1.0], &early);
        assert!(pieces.iter().all(|p| !p.is_empty()), "{pieces:?}");
        assert_eq!(pieces[2].end, 12);
        assert!(pieces[1].end <= 4, "{pieces:?}");
    }

    #[test]
    fn a_faster_node_takes_more_up_to_its_memory() {
        // Twice as fast, twice the bytes.
        assert_eq!(by_speed(&[100, 100], &[2.0, 1.0], 90), Some(vec![60, 30]));
        // Four times as fast but room for 50: the rest to the others by
        // their speeds.
        assert_eq!(
            by_speed(&[50, 100, 100], &[4.0, 1.0, 1.0], 90),
            Some(vec![50, 20, 20])
        );
        // An unmeasured node, or too little room: by memory.
        assert_eq!(by_speed(&[100, 100], &[2.0, 0.0], 90), None);
        assert_eq!(by_speed(&[40, 40], &[2.0, 1.0], 90), None);
        // As weights, cut by bytes: 12 even layers, a node three times as
        // fast as each of two workers.
        let w = by_speed(&[100, 100, 100], &[3.0, 1.0, 1.0], 120).unwrap();
        let (local, workers) = plan_shares(0..12, &EVEN, LocalLayers::Auto, w[0], &w[1..], &ANY);
        assert_eq!((local, workers), (0..7, vec![7..10, 10..12]));
    }

    #[test]
    fn the_device_outranks_the_host() {
        assert_eq!(budget(10, 100), 9);
        assert_eq!(budget(0, 100), 80);
        assert_eq!(bytes_of(&(1..3), &[5, 6, 7, 8]), 13);
    }
}
