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

//! `--interference`: what one request's prefill does to another's decode.
//!
//! Two agents on one server are not two independent streams: they share one
//! device, and a decode step queued behind a prefill submission waits for it.
//! This mode starts one stream decoding, waits for its first tokens, then
//! sends a second request with a long prompt, and records the gap between
//! every pair of the first stream's tokens. The worst gap and the 99th
//! percentile are what the decoding agent's user sees as a stall.
//!
//! Needs a server with two or more slots; on one slot the second request
//! simply queues and the gaps show nothing.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use crate::{Args, Stats, history};

/// Streams a completion and returns the arrival time of every token chunk.
fn token_times(
    client: &reqwest::blocking::Client,
    args: &Args,
    prompt: &str,
    n_gen: u32,
    started: &std::sync::Barrier,
) -> anyhow::Result<Vec<Instant>> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": n_gen,
        "n_predict": n_gen,
        "temperature": args.temperature,
        "stream": true,
        "cache_prompt": false,
        "ignore_eos": true,
    });
    if let Some(m) = &args.model {
        body["model"] = serde_json::Value::String(m.clone());
    }
    let endpoint = format!("{}/v1/completions", args.url);
    let resp = crate::post_with_one_retry(client, &endpoint, &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    let mut times = Vec::new();
    let mut signalled = false;
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Some(payload) = line.trim_start().strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if v["choices"][0]["text"]
            .as_str()
            .is_some_and(|t| !t.is_empty())
        {
            times.push(Instant::now());
            // A few tokens in, so the decode is steady before the intruder.
            if !signalled && times.len() == 8 {
                started.wait();
                signalled = true;
            }
        }
    }
    if !signalled {
        started.wait();
    }
    Ok(times)
}

/// One run: the decoder's inter-token gaps, and how long the intruder took.
struct Run {
    gaps_ms: Vec<f64>,
    intruder_s: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

/// `--interference`: one row per intruding prompt length.
pub fn run_interference(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    if !args.json {
        println!(
            "{:>8} | {:>9} | {:>9} | {:>9} | {:>10} | {:>16}",
            "intruder", "gap p50", "gap p99", "gap max", "intruder s", "max gap mean ± sd"
        );
        println!("{}", "-".repeat(78));
    }
    let mut records = Vec::new();
    for (point, &n) in args.interference.iter().enumerate() {
        if point > 0 {
            args.settle();
        }
        let mut runs = Vec::new();
        for rep in 0..args.reps.max(1) {
            let barrier = std::sync::Barrier::new(2);
            let decoder_prompt =
                format!("Run {rep}-{n}: tell a long, continuous story, do not stop:");
            // A prompt nothing has cached, so all of it is prefilled.
            let intruder_prompt = format!("Intruder {rep}. {}", crate::build_prompt(n));
            let (times, intruder_s) = std::thread::scope(|scope| {
                let decoder = scope
                    .spawn(|| token_times(client, args, &decoder_prompt, args.n_gen, &barrier));
                barrier.wait();
                let t0 = Instant::now();
                let never = std::sync::Barrier::new(1);
                let intruder = token_times(client, args, &intruder_prompt, 1, &never);
                let intruder_s = t0.elapsed().as_secs_f64();
                let times = decoder.join().expect("decoder thread");
                (times, intruder.map(|_| intruder_s))
            });
            let times = times?;
            let intruder_s = intruder_s?;
            let gaps_ms = times
                .windows(2)
                .map(|w| (w[1] - w[0]).as_secs_f64() * 1000.0)
                .collect();
            runs.push(Run {
                gaps_ms,
                intruder_s,
            });
            std::thread::sleep(Duration::from_millis(200));
        }
        let maxes: Vec<f64> = runs
            .iter()
            .map(|r| r.gaps_ms.iter().copied().fold(0.0, f64::max))
            .collect();
        let stats = Stats::of(&maxes, true);
        let mut all: Vec<f64> = runs
            .iter()
            .flat_map(|r| r.gaps_ms.iter().copied())
            .collect();
        all.sort_by(f64::total_cmp);
        let intruder_s = runs.iter().map(|r| r.intruder_s).sum::<f64>() / runs.len() as f64;
        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "intruder_tokens": n,
                    "gap_p50_ms": percentile(&all, 0.5),
                    "gap_p99_ms": percentile(&all, 0.99),
                    "gap_max_ms_best": stats.best,
                    "gap_max_ms_mean": stats.mean,
                    "gap_max_ms_sd": stats.sd,
                    "intruder_s": intruder_s,
                })
            );
        } else {
            println!(
                "{:>8} | {:>9.1} | {:>9.1} | {:>9.1} | {:>10.2} | {:>8.1} ± {}",
                n,
                percentile(&all, 0.5),
                percentile(&all, 0.99),
                all.last().copied().unwrap_or(0.0),
                intruder_s,
                stats.mean,
                stats.plus_minus(5, 1)
            );
        }
        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "itl".to_string(),
            n,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    #[test]
    fn percentiles_index_the_sorted_gaps() {
        let v = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(super::percentile(&v, 0.5), 3.0);
        assert_eq!(super::percentile(&v, 0.99), 100.0);
        assert_eq!(super::percentile(&[], 0.5), 0.0);
    }
}
