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

//! Qwen3-Next (`general.architecture = "qwen3next"`), e.g.
//! `unsloth/Qwen3-Coder-Next-GGUF`.
//!
//! The same hybrid full-attention / gated-DeltaNet trunk and the same
//! routed + shared-expert MoE FFN as `qwen35moe`, both of which live in
//! [`engine::arch::qwen_hybrid`](super::qwen_hybrid). The only difference is
//! how a recurrent layer's projections are named and packed — `ssm_ba.
//! weight` carrying beta and alpha interleaved rather than the split
//! `ssm_beta`/`ssm_alpha`, and older checkpoints storing recurrent QKV and
//! `z` in one `ssm_in.weight` instead of the split `attn_qkv.weight` and
//! `attn_gate.weight`. Both variants are read by
//! `qwen_hybrid::RecurrentWeights::load`, which is why this module carries
//! no forward pass of its own.

use anyhow::Result;
use std::sync::Arc;

use super::ModelForward;
use super::qwen_hybrid::{MoeFfn, Trunk};
use crate::engine::backend::Backend;
use crate::engine::kv_cache::KvCache;
use crate::engine::loader::{LoadedModel, ModelConfig};

pub struct Qwen3NextModel {
    trunk: Trunk<MoeFfn>,
}

impl Qwen3NextModel {
    pub fn load_with_backend(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        Ok(Self {
            trunk: Trunk::load_moe(loaded, backend)?,
        })
    }
}

impl ModelForward for Qwen3NextModel {
    fn vulkan_backend(&self) -> Option<&crate::engine::backend::vulkan::VulkanBackend> {
        self.trunk.backend.as_wgpu()
    }

    fn config(&self) -> &ModelConfig {
        &self.trunk.config
    }

    fn n_trunk_layer(&self) -> usize {
        self.trunk.layer_count()
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        self.trunk.new_kv_cache(capacity)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
        self.trunk.forward(cache, tokens, start_pos)
    }

    fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::bail!("embeddings are not yet supported for Qwen3-Next models")
    }

    /// The same cut as `qwen35`'s, through the same trunk: each layer keeps its
    /// own state — a full-attention layer its KV rows, a gated-DeltaNet layer
    /// its recurrent state — and passes on only the residual stream, the
    /// experts included, so any cut is exact.
    fn supports_layer_split(&self) -> bool {
        true
    }

    fn new_kv_cache_for_layers(&self, layers: std::ops::Range<usize>, capacity: usize) -> KvCache {
        self.trunk.new_kv_cache_for_layers(layers, capacity)
    }

    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.trunk.embed(tokens)
    }

    /// The host path over `layers`.
    fn forward_layers(
        &self,
        cache: &mut KvCache,
        hidden: Vec<f32>,
        tokens: &[u32],
        layers: std::ops::Range<usize>,
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        self.trunk
            .forward_range(cache, hidden, tokens.len(), start_pos, layers)
    }

    fn head(&self, hidden: &[f32], n_rows: usize) -> Result<Vec<Vec<f32>>> {
        self.trunk.head_rows(hidden, n_rows)
    }
}
