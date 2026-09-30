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

//! A worker that downloads only what it runs (`[workers].download =
//! range`).
//!
//! A worker runs a range of layers, and building the model reads only its
//! vectors — norms, biases — while its matrices are read when a layer runs.
//! So a worker with no copy of the model fetches, at start, every tensor
//! outside the layer blocks (the embedding and the head, which a Gemma 4
//! worker reads for its per-layer inputs, are among them) and every block's
//! vectors; the blocks themselves when a parent assigns them, before the
//! model check reads them. The file is sparse: what was not fetched is a
//! hole, never read.
//!
//! Such a node cannot serve on its own, so its own API stays off.

use anyhow::Result;
use orangu::model_download::PartialModel;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

static PARTIAL: OnceLock<Arc<PartialModel>> = OnceLock::new();

/// Makes `partial` the model this process runs.
pub fn set(partial: Arc<PartialModel>) {
    let _ = PARTIAL.set(partial);
}

/// The partial model this process runs, if it runs one.
pub fn partial() -> Option<&'static Arc<PartialModel>> {
    PARTIAL.get()
}

/// Whether this process runs a model it has not fetched whole.
pub fn incomplete() -> bool {
    partial().is_some_and(|p| !p.complete())
}

/// What building the model reads: every tensor outside the layer blocks,
/// and every block's vectors.
pub fn needed_to_build(name: &str, dims: &[u64]) -> bool {
    crate::engine::loader::block_index(name).is_none() || dims.len() <= 1
}

/// Opens `spec` in part and fetches what building it reads.
pub fn open(models_dir: &std::path::Path, spec: &str) -> Result<Arc<PartialModel>> {
    let started = Instant::now();
    let partial = Arc::new(PartialModel::open(models_dir, spec)?);
    let bytes = partial.fetch(needed_to_build)?;
    log::info!(
        "orangu-server: [workers] download = range: fetched {} of {spec} in {:.1} s — what \
         building the model reads; each layer range comes when a parent assigns it",
        orangu::format::format_bytes(bytes),
        started.elapsed().as_secs_f64()
    );
    Ok(partial)
}

/// Fetches `layers`' tensors, if this process runs a partial model.
pub fn layers(layers: &Range<usize>) -> Result<()> {
    let Some(partial) = partial() else {
        return Ok(());
    };
    let started = Instant::now();
    let bytes = partial.fetch(|name, _| {
        crate::engine::loader::block_index(name).is_some_and(|il| layers.contains(&il))
    })?;
    if bytes > 0 {
        log::info!(
            "orangu-server: fetched layers {}..{} ({}) in {:.1} s",
            layers.start,
            layers.end,
            orangu::format::format_bytes(bytes),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
