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

//! `--chat`: a coding session, turn by turn, through the chat endpoint.
//!
//! Every other text mode sends a repeated sentence to `/v1/completions`,
//! which skips the chat template, the tool declarations and the prefix cache
//! a conversation leans on. This mode sends what a coding agent sends: a long
//! system prompt, a tool array, and a conversation that grows by one user
//! message and the model's own reply per turn, to `/v1/chat/completions`.
//!
//! What it reports per turn is what such a session waits on: time to first
//! token, how much of the prompt the cache supplied, how long the server
//! spent prefilling, and the remainder — everything between the request
//! arriving and prefill starting (template, tokenizer, queue) plus the
//! network — as `other`. The recorded rate is the prompt over the time to
//! first token, so a turn served from cache and a turn rendered faster both
//! show as a higher number.
//!
//! The conversation is a fixture: [`BUILTIN`] unless `--chat-fixture` names
//! another file of the same shape (`system`, `tools`, `turns`). Each
//! repetition opens with a session line of its own, so one repetition's
//! cache never serves the next one's first turn.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use crate::{Args, Stats, history};

/// The built-in session: a coding agent's system prompt, four tools and
/// eight turns that carry source code.
const BUILTIN: &str = include_str!("fixtures/coding-session.json");

/// A conversation to replay.
#[derive(serde::Deserialize)]
pub struct Fixture {
    pub system: String,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    pub turns: Vec<String>,
}

impl Fixture {
    pub fn load(path: Option<&Path>) -> anyhow::Result<Fixture> {
        let text = match path {
            Some(path) => std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?,
            None => BUILTIN.to_string(),
        };
        let fixture: Fixture =
            serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("chat fixture: {e}"))?;
        if fixture.turns.is_empty() {
            anyhow::bail!("chat fixture has no turns");
        }
        Ok(fixture)
    }
}

/// One turn, timed.
pub struct TurnSample {
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    /// The server's prefill time, milliseconds.
    pub prompt_ms: f64,
    /// Send to first streamed content, reasoning or tool call.
    pub ttft_ms: f64,
    pub gen_tokens: u32,
    /// The server's decode rate.
    pub decode_tok_s: f64,
    pub content: String,
    pub tool_calls: Vec<serde_json::Value>,
}

impl TurnSample {
    /// The prompt over the time to first token.
    pub fn prompt_tok_s(&self) -> f64 {
        if self.ttft_ms > 0.0 {
            f64::from(self.prompt_tokens) * 1000.0 / self.ttft_ms
        } else {
            0.0
        }
    }

    /// Time to first token not spent in prefill.
    pub fn other_ms(&self) -> f64 {
        (self.ttft_ms - self.prompt_ms).max(0.0)
    }
}

/// Sends one chat request and reads its stream to the end.
pub fn send_turn(
    client: &reqwest::blocking::Client,
    args: &Args,
    messages: &[serde_json::Value],
    tools: Option<&serde_json::Value>,
) -> anyhow::Result<TurnSample> {
    let mut body = serde_json::json!({
        "messages": messages,
        "max_tokens": args.n_gen,
        "temperature": args.temperature,
        "stream": true,
        "cache_prompt": true,
    });
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
    }
    if let Some(m) = &args.model {
        body["model"] = serde_json::Value::String(m.clone());
    }
    let endpoint = format!("{}/v1/chat/completions", args.url);
    let t0 = Instant::now();
    let resp = crate::post_with_one_retry(client, &endpoint, &body)?;
    if !resp.status().is_success() {
        return Err(crate::status_error(resp));
    }
    let mut reader = BufReader::new(resp);
    let mut line = String::new();
    let mut first: Option<Instant> = None;
    let mut sample = TurnSample {
        prompt_tokens: 0,
        cached_tokens: 0,
        prompt_ms: 0.0,
        ttft_ms: 0.0,
        gen_tokens: 0,
        decode_tok_s: 0.0,
        content: String::new(),
        tool_calls: Vec::new(),
    };
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
        if let Some(err) = v.get("error") {
            anyhow::bail!("server error: {err}");
        }
        let delta = &v["choices"][0]["delta"];
        let content = delta["content"].as_str().unwrap_or("");
        let reasoning = delta["reasoning_content"].as_str().unwrap_or("");
        let calls = delta["tool_calls"].as_array();
        if !content.is_empty() || !reasoning.is_empty() || calls.is_some() {
            first.get_or_insert_with(Instant::now);
        }
        sample.content.push_str(content);
        if let Some(calls) = calls {
            sample.tool_calls.extend(calls.iter().cloned());
        }
        if let Some(t) = v.get("timings") {
            let u = |k: &str| t[k].as_u64().map(|n| n as u32);
            sample.prompt_tokens = u("prompt_n").unwrap_or(sample.prompt_tokens);
            sample.prompt_ms = t["prompt_ms"].as_f64().unwrap_or(sample.prompt_ms);
            sample.gen_tokens = u("predicted_n").unwrap_or(sample.gen_tokens);
            sample.decode_tok_s = t["predicted_per_second"]
                .as_f64()
                .unwrap_or(sample.decode_tok_s);
        }
        if let Some(n) = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64() {
            sample.cached_tokens = n as u32;
        }
    }
    sample.ttft_ms = first.map_or(0.0, |f| (f - t0).as_secs_f64() * 1000.0);
    Ok(sample)
}

/// The assistant message a client sends back for `sample`, and the tool
/// answers its calls need before the next user message.
fn reply_messages(sample: &TurnSample) -> Vec<serde_json::Value> {
    if sample.tool_calls.is_empty() {
        return vec![serde_json::json!({"role": "assistant", "content": sample.content})];
    }
    let mut out = vec![serde_json::json!({
        "role": "assistant",
        "content": sample.content,
        "tool_calls": sample.tool_calls,
    })];
    for (i, call) in sample.tool_calls.iter().enumerate() {
        let id = call["id"]
            .as_str()
            .map_or_else(|| format!("call_{i}"), str::to_string);
        out.push(serde_json::json!({
            "role": "tool",
            "tool_call_id": id,
            "content": "(no output: this is a benchmark replay)",
        }));
    }
    out
}

/// `--chat`: replays the fixture's first `turns` turns, `--reps` times.
pub fn run_chat(
    client: &reqwest::blocking::Client,
    args: &Args,
    label: &str,
) -> anyhow::Result<Vec<history::Record>> {
    let fixture = Fixture::load(args.chat_fixture.as_deref())?;
    let turns = (args.chat as usize).min(fixture.turns.len());
    if turns < args.chat as usize && !args.json {
        eprintln!(
            "note: the fixture has {} turns; replaying all of them",
            fixture.turns.len()
        );
    }
    let mut per_turn: Vec<Vec<TurnSample>> = (0..turns).map(|_| Vec::new()).collect();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    for rep in 0..args.reps.max(1) {
        if rep > 0 {
            args.settle();
        }
        let mut messages = vec![serde_json::json!({
            "role": "system",
            "content": format!("Session {stamp}-{rep}.\n\n{}", fixture.system),
        })];
        for (turn, text) in fixture.turns.iter().take(turns).enumerate() {
            messages.push(serde_json::json!({"role": "user", "content": text}));
            let sample = send_turn(client, args, &messages, fixture.tools.as_ref())?;
            messages.extend(reply_messages(&sample));
            per_turn[turn].push(sample);
        }
    }

    if !args.json {
        println!(
            "{:>4} | {:>6} | {:>6} | {:>4} | {:>8} | {:>8} | {:>8} | {:>7} | {:>16}",
            "turn",
            "prompt",
            "cached",
            "hit",
            "ttft_ms",
            "pp_ms",
            "other_ms",
            "tg t/s",
            "pp/ttft mean ± sd"
        );
        println!("{}", "-".repeat(98));
    }
    let mut records = Vec::new();
    for (turn, samples) in per_turn.iter().enumerate() {
        let rates: Vec<f64> = samples.iter().map(TurnSample::prompt_tok_s).collect();
        let stats = Stats::of(&rates, false);
        // The fastest repetition's row, the same one `best` names.
        let s = samples
            .iter()
            .max_by(|a, b| a.prompt_tok_s().total_cmp(&b.prompt_tok_s()))
            .expect("at least one rep ran");
        let hit = if s.prompt_tokens > 0 {
            100.0 * f64::from(s.cached_tokens) / f64::from(s.prompt_tokens)
        } else {
            0.0
        };
        if args.json {
            println!(
                "{}",
                serde_json::json!({
                    "turn": turn + 1,
                    "prompt_tokens": s.prompt_tokens,
                    "cached_tokens": s.cached_tokens,
                    "ttft_ms": s.ttft_ms,
                    "prompt_ms": s.prompt_ms,
                    "other_ms": s.other_ms(),
                    "gen_tokens": s.gen_tokens,
                    "decode_tok_per_s": s.decode_tok_s,
                    "tool_calls": s.tool_calls.len(),
                    "tok_per_s_best": stats.best,
                    "tok_per_s_mean": stats.mean,
                    "tok_per_s_sd": stats.sd,
                    "tok_per_s_sd_sample": stats.sd_sample,
                })
            );
        } else {
            println!(
                "{:>4} | {:>6} | {:>6} | {:>3.0}% | {:>8.1} | {:>8.1} | {:>8.1} | {:>7.2} | {:>8.1} ± {}",
                turn + 1,
                s.prompt_tokens,
                s.cached_tokens,
                hit,
                s.ttft_ms,
                s.prompt_ms,
                s.other_ms(),
                s.decode_tok_s,
                stats.mean,
                stats.plus_minus(5, 1)
            );
        }
        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "chat".to_string(),
            n: (turn + 1) as u32,
            best: stats.best,
            mean: stats.mean,
            sd: stats.sd,
            sd_sample: stats.sd_sample,
            device: None,
        });
        // The reply's decode rate as its own series: what a speculative or
        // decode-side change moves, and a prompt-side one should not.
        let decode: Vec<f64> = samples.iter().map(|s| s.decode_tok_s).collect();
        let decode = Stats::of(&decode, false);
        records.push(history::Record {
            date: history::today(),
            label: label.to_string(),
            mode: "chat-tg".to_string(),
            n: (turn + 1) as u32,
            best: decode.best,
            mean: decode.mean,
            sd: decode.sd,
            sd_sample: decode.sd_sample,
            device: None,
        });
    }
    if !args.json {
        let all: Vec<&TurnSample> = per_turn.iter().flatten().collect();
        let prompt: u64 = all.iter().map(|s| u64::from(s.prompt_tokens)).sum();
        let cached: u64 = all.iter().map(|s| u64::from(s.cached_tokens)).sum();
        let calls: usize = all.iter().map(|s| s.tool_calls.len()).sum();
        println!(
            "cache: {cached} of {prompt} prompt tokens ({:.1}%) · {calls} tool call(s)",
            if prompt > 0 {
                100.0 * cached as f64 / prompt as f64
            } else {
                0.0
            }
        );
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builtin_fixture_parses_and_declares_tools() {
        let f = Fixture::load(None).unwrap();
        assert!(f.turns.len() >= 8);
        assert!(
            f.tools
                .as_ref()
                .and_then(|t| t.as_array())
                .is_some_and(|t| !t.is_empty())
        );
    }

    #[test]
    fn every_tool_call_gets_an_answer_before_the_next_turn() {
        let sample = TurnSample {
            prompt_tokens: 0,
            cached_tokens: 0,
            prompt_ms: 0.0,
            ttft_ms: 0.0,
            gen_tokens: 0,
            decode_tok_s: 0.0,
            content: String::new(),
            tool_calls: vec![serde_json::json!({"id": "a"}), serde_json::json!({})],
        };
        let out = reply_messages(&sample);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1]["tool_call_id"], "a");
        assert_eq!(out[2]["tool_call_id"], "call_1");
    }
}
