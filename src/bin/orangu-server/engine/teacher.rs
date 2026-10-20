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

//! Teacher-forced next-token distributions, for comparing two builds or
//! two configurations of one model (`POST /logits`).
//!
//! A fixed token sequence is run through the model and, after every prefix
//! of it, the most likely next tokens are kept with their log-probabilities.
//! Nothing is sampled, so two runs see exactly the same inputs at every
//! position and differ only in the arithmetic: a change of KV storage, of a
//! kernel's reduction order, of a weight layout. `orangu-bench --kl`
//! compares two such runs as a KL divergence and a top-1 agreement.
//!
//! `rows` picks the path: 1 runs every position as a decode step
//! ([`ModelForward::forward`], what generation runs); more runs the sequence
//! in `rows`-token passes through [`ModelForward::forward_verify`], what
//! speculative verification runs. The two compared tell whether a verified
//! token is the one a decode step would have chosen.

use anyhow::Result;

use super::arch::ModelForward;

/// The widest multi-position pass a request may ask for.
pub const MAX_ROWS: usize = 64;
/// The most candidates a position may keep.
pub const MAX_TOP: usize = 256;

/// One position's most likely next tokens as `(id, log-probability)`,
/// most likely first.
pub type TopLogprobs = Vec<(u32, f32)>;

/// The `top` most likely next tokens after every prefix of `tokens`, run
/// `rows` positions a pass in a fresh cache.
pub fn teacher_forced(
    model: &dyn ModelForward,
    tokens: &[u32],
    rows: usize,
    top: usize,
    slot_id: usize,
) -> Result<Vec<TopLogprobs>> {
    let rows = rows.clamp(1, MAX_ROWS);
    let mut cache = model.new_kv_cache(tokens.len().max(1));
    let mut out = Vec::with_capacity(tokens.len());
    for (i, part) in tokens.chunks(rows).enumerate() {
        let pos = i * rows;
        if rows == 1 {
            out.push(top_logprobs(
                &model.forward(&mut cache, part, pos, slot_id)?,
                top,
            ));
        } else {
            for row in model.forward_verify(&mut cache, part, pos, slot_id)? {
                out.push(top_logprobs(&row, top));
            }
        }
    }
    Ok(out)
}

/// The `top` largest entries of `logits` as log-probabilities over the
/// whole row, largest first.
pub fn top_logprobs(logits: &[f32], top: usize) -> TopLogprobs {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_z = max as f64
        + logits
            .iter()
            .map(|&l| ((l - max) as f64).exp())
            .sum::<f64>()
            .ln();
    let top = top.clamp(1, MAX_TOP).min(logits.len());
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    if top < ids.len() {
        ids.select_nth_unstable_by(top - 1, by_logit);
        ids.truncate(top);
    }
    ids.sort_by(by_logit);
    ids.into_iter()
        .map(|id| (id, (logits[id as usize] as f64 - log_z) as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_top_entries_are_normalised_over_the_whole_row() {
        let logits = [0.0f32, 2.0, 1.0, -1.0];
        let top = top_logprobs(&logits, 2);
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), [1, 2]);
        let z: f64 = logits.iter().map(|&l| (l as f64).exp()).sum();
        assert!((top[0].1 as f64 - (2.0 - z.ln())).abs() < 1e-6);
        // Asking for more than the row holds keeps the whole row.
        let all = top_logprobs(&logits, 10);
        assert_eq!(all.len(), 4);
        let mass: f64 = all.iter().map(|t| (t.1 as f64).exp()).sum();
        assert!((mass - 1.0).abs() < 1e-6);
    }
}
