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

//! `--kl`: how far a server's next-token distributions sit from a
//! reference's, teacher-forced over the same tokens.
//!
//! The first run against a path that does not exist yet saves the server's
//! distributions there as the reference. Every later run sends the
//! reference's exact tokens to `/logits` and compares position by position:
//! the KL divergence from the reference (mean, 99th percentile, worst) and
//! how often the most likely token is the same. A precision change — a KV
//! storage format, a kernel's reduction order, a weight layout — is judged
//! by these, not by whether a few generated replies look alright.
//!
//! Each side keeps only its top candidates, so the divergence is taken over
//! the reference's candidates plus one bucket for everything else. A
//! candidate the other side did not keep is given that side's smallest kept
//! probability, an upper bound on what it held.

use std::path::Path;

use crate::{Args, history};

/// A saved reference, or one server's answer.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Distributions {
    pub tokens: Vec<u32>,
    pub rows: usize,
    pub positions: Vec<Vec<(u32, f32)>>,
    pub seconds: f64,
}

/// One position compared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub kl: f64,
    pub same_top: bool,
}

/// KL(reference ‖ other) over the reference's candidates and the rest.
pub fn compare(reference: &[(u32, f32)], other: &[(u32, f32)]) -> Position {
    let floor = other
        .iter()
        .map(|&(_, lp)| f64::from(lp).exp())
        .fold(f64::INFINITY, f64::min);
    let floor = if floor.is_finite() { floor } else { 1e-12 };
    let mut kl = 0.0;
    let (mut p_kept, mut q_kept) = (0.0, 0.0);
    for &(id, lp) in reference {
        let p = f64::from(lp).exp();
        let q = other
            .iter()
            .find(|c| c.0 == id)
            .map_or(floor, |c| f64::from(c.1).exp());
        p_kept += p;
        q_kept += q;
        if p > 0.0 {
            kl += p * (p / q.max(1e-12)).ln();
        }
    }
    let p_rest = (1.0 - p_kept).max(0.0);
    let q_rest = (1.0 - q_kept).max(1e-12);
    if p_rest > 1e-9 {
        kl += p_rest * (p_rest / q_rest).ln();
    }
    Position {
        kl: kl.max(0.0),
        same_top: reference.first().map(|c| c.0) == other.first().map(|c| c.0),
    }
}

/// The text a reference is taken over: the chat fixture's system prompt
/// and turns, which are mostly code and tool definitions.
fn text(args: &Args) -> anyhow::Result<String> {
    if let Some(path) = &args.kl_text {
        return std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()));
    }
    let fixture = crate::chat::Fixture::load(args.chat_fixture.as_deref())?;
    let mut text = fixture.system;
    for turn in fixture.turns {
        text.push_str("\n\n");
        text.push_str(&turn);
    }
    Ok(text)
}

fn fetch(
    client: &reqwest::blocking::Client,
    args: &Args,
    tokens: Vec<u32>,
) -> anyhow::Result<Distributions> {
    let body = serde_json::json!({
        "tokens": tokens,
        "rows": args.kl_rows,
        "top": args.kl_top,
    });
    let resp = crate::post_with_one_retry(client, &format!("{}/logits", args.url), &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    Ok(resp.json()?)
}

fn tokens_of(
    client: &reqwest::blocking::Client,
    args: &Args,
    text: &str,
) -> anyhow::Result<Vec<u32>> {
    let body = serde_json::json!({ "content": text, "add_special": true });
    let resp = crate::post_with_one_retry(client, &format!("{}/tokenize", args.url), &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    let v: serde_json::Value = resp.json()?;
    let mut tokens: Vec<u32> = serde_json::from_value(v["tokens"].clone())?;
    tokens.truncate(args.kl_tokens as usize);
    Ok(tokens)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

pub fn run_kl(
    client: &reqwest::blocking::Client,
    args: &Args,
    path: &Path,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    if !path.exists() {
        let tokens = tokens_of(client, args, &text(args)?)?;
        let reference = fetch(client, args, tokens)?;
        std::fs::write(path, serde_json::to_string(&reference)?)?;
        println!(
            "reference: {} positions, {} rows a pass, {:.2} ms a position — saved to {}",
            reference.positions.len(),
            reference.rows,
            reference.seconds * 1000.0 / reference.positions.len().max(1) as f64,
            path.display()
        );
        return Ok(Vec::new());
    }
    let reference: Distributions = serde_json::from_str(&std::fs::read_to_string(path)?)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let other = fetch(client, args, reference.tokens.clone())?;
    let compared: Vec<Position> = reference
        .positions
        .iter()
        .zip(&other.positions)
        .map(|(r, o)| compare(r, o))
        .collect();
    let mut kls: Vec<f64> = compared.iter().map(|p| p.kl).collect();
    kls.sort_by(f64::total_cmp);
    let mean = kls.iter().sum::<f64>() / kls.len().max(1) as f64;
    let agree = 100.0 * compared.iter().filter(|p| p.same_top).count() as f64
        / compared.len().max(1) as f64;
    let ms_ref = reference.seconds * 1000.0 / reference.positions.len().max(1) as f64;
    let ms = other.seconds * 1000.0 / other.positions.len().max(1) as f64;
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "positions": compared.len(),
                "rows": other.rows,
                "reference_rows": reference.rows,
                "kl_mean": mean,
                "kl_p99": percentile(&kls, 0.99),
                "kl_max": kls.last().copied().unwrap_or(0.0),
                "top1_agree_pct": agree,
                "ms_per_position": ms,
                "reference_ms_per_position": ms_ref,
            })
        );
    } else {
        println!(
            "positions | rows (ref) |   KL mean |    KL p99 |    KL max | top-1 same | ms/pos (ref)"
        );
        println!(
            "{:>9} | {:>4} ({:>3}) | {:>9.6} | {:>9.6} | {:>9.6} | {:>9.2}% | {:>6.2} ({:.2})",
            compared.len(),
            other.rows,
            reference.rows,
            mean,
            percentile(&kls, 0.99),
            kls.last().copied().unwrap_or(0.0),
            agree,
            ms,
            ms_ref
        );
    }
    let record = |mode: &str, value: f64| history::Record {
        date: history::today(),
        label: label.to_string(),
        mode: mode.to_string(),
        n: other.rows as u32,
        best: value,
        mean: value,
        sd: 0.0,
        sd_sample: None,
        device: None,
    };
    Ok(vec![record("kl", mean), record("top1", agree)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(p: &[(u32, f64)]) -> Vec<(u32, f32)> {
        p.iter().map(|&(id, p)| (id, p.ln() as f32)).collect()
    }

    #[test]
    fn identical_distributions_are_zero_apart() {
        let r = dist(&[(3, 0.6), (1, 0.3), (2, 0.05)]);
        let c = compare(&r, &r);
        assert!(c.kl < 1e-7, "{}", c.kl);
        assert!(c.same_top);
    }

    #[test]
    fn a_moved_winner_is_counted_and_costs_divergence() {
        let r = dist(&[(3, 0.6), (1, 0.3), (2, 0.05)]);
        let o = dist(&[(1, 0.6), (3, 0.3), (2, 0.05)]);
        let c = compare(&r, &o);
        assert!(!c.same_top);
        // 0.6 ln 2 + 0.3 ln 0.5
        assert!((c.kl - 0.3 * 2f64.ln()).abs() < 1e-6, "{}", c.kl);
    }

    #[test]
    fn a_candidate_the_other_side_dropped_takes_its_smallest_kept_probability() {
        let r = dist(&[(3, 0.5), (9, 0.4)]);
        let o = dist(&[(3, 0.5), (1, 0.1)]);
        let c = compare(&r, &o);
        let rest = 0.1f64;
        let q_rest = 1.0 - 0.5 - 0.1;
        let want = 0.4 * (0.4f64 / 0.1).ln() + rest * (rest / q_rest).ln();
        assert!((c.kl - want).abs() < 1e-6, "{} vs {want}", c.kl);
    }
}
