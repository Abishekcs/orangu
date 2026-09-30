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

//! How fast a node runs its layers (W-86).
//!
//! The nodes of a tree may be anything — a GPU, cores of one kind or
//! another, prompts on an NPU — so a node times the model's first layers
//! itself, at start, through the path its layers take in a tree
//! (`ModelForward::forward_layers` on its own backend): one decode step and
//! one 128-token prompt chunk, over two ranges so that what a forward costs
//! whatever its layers cancels out. Each is reported as the gigabytes of layer
//! weights it gets through a second, which a parent divides a range by
//! (`super::plan::by_speed`), since a layer's cost follows its bytes.

use super::plan;
use crate::engine::arch::ModelForward;
use anyhow::Result;
use std::ops::Range;
use std::time::{Duration, Instant};

/// The prompt chunk timed: the parts a tree's prompts travel in.
pub const PROMPT_TOKENS: usize = 128;

/// Decode steps timed, after two that are not.
const DECODE_STEPS: usize = 8;

/// A node's measured speed: gigabytes of layer weights a second.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Speed {
    pub decode: f32,
    pub prompt: f32,
}

/// The most weights the longer of the two timed ranges takes: what a node
/// reads in at start to time itself.
const MEASURE_BYTES: u64 = 1 << 30;

/// Times the model's first layers through a prompt chunk and decode steps,
/// on two ranges from layer 0 — the first cut the model allows from the
/// second layer on, and a longer one (up to a quarter of the layers and
/// [`MEASURE_BYTES`]) — and rates the layers between the two cuts by the
/// difference. A forward's fixed costs, whatever its range (Gemma 4's
/// per-layer inputs for every layer, a device's submission and readback),
/// cancel out: timing a short range alone made them the rate, and those
/// costs differ from backend to backend (B-7 in `doc/BUGS.md`). A model too
/// small for two cuts is rated over the one. Each range runs once untimed
/// first: the first run reads the weights in, uploads them and builds a
/// device's pipelines. Returns the speed, the layers rated and how long it
/// all took.
pub fn measure(
    model: &dyn ModelForward,
    layer_bytes: &[u64],
) -> Result<(Speed, Range<usize>, Duration)> {
    let started = Instant::now();
    let n = model.config().n_layer;
    let cut = |at: usize| at == n || model.split_allowed(at);
    let short = (2.min(n)..=n).find(|at| cut(*at)).unwrap_or(n);
    let long = (short + 1..=(n / 4).max(short + 1).min(n))
        .rev()
        .find(|at| cut(*at) && plan::bytes_of(&(0..*at), layer_bytes) <= MEASURE_BYTES);
    let (short_prompt, short_decode) = time_range(model, 0..short)?;
    let (layers, prompt, decode) = match long {
        Some(long) => {
            let (long_prompt, long_decode) = time_range(model, 0..long)?;
            match (
                long_prompt.checked_sub(short_prompt),
                long_decode.checked_sub(short_decode),
            ) {
                (Some(p), Some(d)) if !p.is_zero() && !d.is_zero() => (short..long, p, d),
                _ => (0..long, long_prompt, long_decode),
            }
        }
        None => (0..short, short_prompt, short_decode),
    };
    let bytes = plan::bytes_of(&layers, layer_bytes) as f64;
    let rate = |d: Duration| (bytes / d.as_secs_f64().max(1e-9) / 1e9) as f32;
    let speed = Speed {
        decode: rate(decode),
        prompt: rate(prompt),
    };
    Ok((speed, layers, started.elapsed()))
}

/// One 128-token prompt chunk's time and one decode step's (the mean of
/// [`DECODE_STEPS`]) through `layers`, each after an untimed run.
fn time_range(model: &dyn ModelForward, layers: Range<usize>) -> Result<(Duration, Duration)> {
    let mut cache =
        model.new_kv_cache_for_layers(layers.clone(), 2 * PROMPT_TOKENS + 2 + DECODE_STEPS);
    let vocab = model.config().n_vocab.clamp(2, 1000) as u32;
    let tokens: Vec<u32> = (0..PROMPT_TOKENS as u32)
        .map(|i| 1 + i * 7 % (vocab - 1))
        .collect();
    let mut run = |tokens: &[u32], pos: usize| -> Result<Duration> {
        let hidden = model.embed(tokens)?;
        let at = Instant::now();
        model.forward_layers(&mut cache, hidden, tokens, layers.clone(), pos)?;
        Ok(at.elapsed())
    };
    run(&tokens, 0)?;
    let prompt = run(&tokens, PROMPT_TOKENS)?;
    let mut pos = 2 * PROMPT_TOKENS;
    for _ in 0..2 {
        run(&tokens[..1], pos)?;
        pos += 1;
    }
    let mut decode = Duration::ZERO;
    for _ in 0..DECODE_STEPS {
        decode += run(&tokens[..1], pos)?;
        pos += 1;
    }
    Ok((prompt, decode / DECODE_STEPS as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::pipeline::fixture::{self, Variant};

    #[test]
    fn a_node_measures_its_first_layers() {
        let model = fixture::model();
        let loaded = fixture::loaded(Variant::default());
        let n = model.config().n_layer;
        let mut layer_bytes = vec![0u64; n];
        for (name, bytes) in loaded.tensor_sizes() {
            if let Some(il) = crate::engine::loader::block_index(name).filter(|il| *il < n) {
                layer_bytes[il] += bytes;
            }
        }
        let (speed, layers, _) = measure(model.as_ref(), &layer_bytes).unwrap();
        // The layers between the first cut (2) and the longer range (3).
        assert_eq!(layers, 2..3);
        assert!(speed.decode > 0.0 && speed.prompt > 0.0, "{speed:?}");
    }
}
