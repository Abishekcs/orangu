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

//! One `orangu-server` process's place in a tree of workers.
//!
//! A [`Node`] is both halves at once, and which one it is right now follows
//! from what happens to it (`doc/WORKERS.md`, design decision 9):
//!
//! - **Top-level.** With workers of its own and no parent, it dials them,
//!   plans the model's layers over itself and them, and serves its own API
//!   through a [`DelegatingModel`] on that plan.
//! - **Worker.** When a parent connects and assigns it a range of layers,
//!   it checks that it has the parent's model, plans that range over itself
//!   and its own workers the same way, and answers the parent's forwards
//!   from a [`SessionStore`]. Its own API is off meanwhile. When the parent
//!   goes away it becomes top-level again.
//!
//! A node listens for a parent whenever its config has a `[workers]`
//! section, and hands layers down whenever its `workers` list is not empty;
//! a middle node does both.

use super::auth;
use super::guard::{GuardedStage, TakeStandby};
use super::identity;
use super::link::{ChildInfo, ChildLink};
use super::metrics::{LinkSnapshot, NodeMetrics};
use super::pipeline::{DelegatingModel, LayerPipeline, PipelineSource, Stage};
use super::plan;
use super::protocol::{
    ActivationFormat, Assign, Capacity, ErrorCode, FEATURES, HelloAck, MAX_DEPTH, Message,
    NodeSetup, PROTOCOL_VERSION, PlanEntry, WorkerError, check_path, read_frame, read_message,
    write_message,
};
use super::session::SessionStore;
use super::stage::{EncodedStage, MessageStage};
use super::transport::{self, Tls};
use crate::config::{DecodeOn, HeadOn, LocalLayers, Shares};
use crate::engine::arch::ModelForward;
use crate::engine::loader::LoadedModel;
use crate::engine::prompt_weights::{self, Decision};
use anyhow::{Context, Result};
use std::io::{BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

/// How long a session may sit unused on a worker before it is reclaimed: a
/// parent that went away without releasing it leaves it behind.
const SESSION_IDLE: Duration = Duration::from_secs(600);

/// How long a worker may take to answer an assignment: its layers may have
/// to be downloaded first (`[workers].download = range`).
const ASSIGN_TIMEOUT: Duration = Duration::from_secs(3600);

/// How long a node goes without a forward before it counts as between
/// requests, and may take a returning worker back. Sessions a parent keeps
/// for a slot's next request live far longer and do not count: a plan made
/// under them costs that request a rebuild.
const QUIET: Duration = Duration::from_secs(2);

pub struct NodeSettings {
    /// How this node is named in plans and error paths: its `[workers]`
    /// address.
    pub name: String,
    /// The address to listen on for a parent; `None` listens nowhere.
    pub listen: Option<String>,
    pub workers: Vec<String>,
    /// Spare workers: never planned into a share, one of them takes a lost
    /// worker's layers (`super::guard`).
    pub standby: Vec<String>,
    pub secret: Option<String>,
    pub activations: ActivationFormat,
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub local_layers: LocalLayers,
    /// Positions a sequence may hold, as a top-level node.
    pub n_ctx: usize,
    /// Sequences at once, as a top-level node.
    pub slots: usize,
    /// The served model as `list` names it, and its quantization — for the
    /// model check's messages.
    pub label: String,
    pub quant: String,
    /// This node's own capacity; the `subtree_*` fields are filled in.
    pub capacity: Capacity,
    /// `[workers].shares`: what shares are sized by.
    pub shares: Shares,
    /// `[workers].decode`: where sequences decode once their prompt is
    /// through the tree (W-60).
    pub decode: DecodeOn,
    /// `[workers].head`: which node applies the output head (W-61).
    pub head: HeadOn,
    /// How often idle workers are pinged.
    pub maintenance: Duration,
    /// How often a top-level node missing a worker tries to take it back,
    /// between requests.
    pub readmit: Duration,
    /// TLS for the listener and for dialing workers, when configured.
    pub tls: Option<Tls>,
    /// `prompt_weights`: how a top-level node decides the copies its tree
    /// runs prompts through (`engine::prompt_weights::decide`).
    pub prompt_weights: prompt_weights::PromptWeights,
}

/// A plan of a range over this node and its workers.
struct Planned {
    pipeline: Arc<LayerPipeline>,
    entries: Vec<PlanEntry>,
    /// The workers it hands layers to.
    used: Vec<String>,
    /// Standbys among them, and whom each fills in for.
    standing: Vec<(String, String)>,
}

/// What a parent's `Assign` asks of the nodes below.
#[derive(Clone, Copy)]
struct Terms {
    n_ctx: usize,
    slots: usize,
    activations: ActivationFormat,
    /// The top-level node's prompt-weight copies, for every node of the tree.
    weights: Decision,
}

struct Assignment {
    store: Arc<SessionStore>,
    layers: Range<usize>,
    /// What the parent asked for, and the path down to this node: what
    /// planning this range again needs.
    terms: Terms,
    path: Vec<String>,
}

struct Parent {
    connection: u64,
    node: String,
    stream: TcpStream,
}

pub struct Node {
    id: String,
    settings: NodeSettings,
    model: Arc<dyn ModelForward>,
    loaded: LoadedModel,
    links: Vec<Arc<ChildLink>>,
    /// `[workers].standby`: never planned into a share.
    standby_links: Vec<Arc<ChildLink>>,
    /// Standbys that took over a lost worker's layers since the last plan:
    /// `(lost worker, standby)`.
    standing_in: Mutex<Vec<(String, String)>>,
    /// This node, for the stages that call back into it for a standby.
    this: OnceLock<Weak<Node>>,
    /// A top-level node's prompt-weight decision, measured at its first plan.
    decision: OnceLock<Decision>,
    /// Switched to serve alone through `POST /props` (W-84): its workers
    /// let go, no plan made until switched back.
    alone: AtomicBool,
    /// Whether the current plan's sequences decode on this node alone once
    /// their prompt is through the tree (W-60).
    decode_alone: AtomicBool,
    /// Whether the node running the final layer applies the output head
    /// (W-61).
    head_on_last: AtomicBool,
    /// What the last plan's shares followed: a speed, or memory.
    shares_by: Mutex<&'static str>,
    /// The layers and decision this node's prompt-weight copies were built
    /// for.
    copied: Mutex<Option<(Range<usize>, Decision)>>,
    /// The top-level pipeline and its generation.
    top: RwLock<(Arc<LayerPipeline>, u64)>,
    assignment: Mutex<Option<Assignment>>,
    parent: Mutex<Option<Parent>>,
    plan: Mutex<Vec<PlanEntry>>,
    /// The addresses of the workers the current plan hands layers to.
    used: Mutex<Vec<String>>,
    /// Held by whoever is planning; the time of the last plan.
    planning: Mutex<Instant>,
    delegating: OnceLock<Weak<DelegatingModel>>,
    /// Each layer's weights, in bytes — what a plan sizes shares by.
    layer_bytes: Vec<u64>,
    /// The weights outside every layer: the embedding, the final norm and
    /// the output head, which only the top-level node runs.
    non_layer_bytes: u64,
    local_addr: Option<SocketAddr>,
    metrics: NodeMetrics,
    next_connection: AtomicU64,
    stop: AtomicBool,
}

static NODE: OnceLock<Arc<Node>> = OnceLock::new();

/// Makes `node` the one this process's API asks about.
pub fn set_global(node: Arc<Node>) {
    let _ = NODE.set(node);
}

/// Whether this process's API is off: working for a parent, or running a
/// model it has fetched only in part (`super::fetch`).
pub fn api_paused() -> bool {
    NODE.get().is_some_and(|node| node.is_assigned()) || super::fetch::incomplete()
}

/// Switches a top-level node between its tree and serving alone (W-84; the
/// engine's side is `engine::generate::ServingSwitch`). Refused on a node
/// that is not top-level with workers, and on one that has only part of the
/// model.
pub fn set_alone(alone: bool) -> Result<(), String> {
    match NODE.get() {
        Some(node) => node.set_alone(alone),
        None => Err("this server has no [workers] section".to_string()),
    }
}

/// The `workers` object of `GET /props`: whether the tree serves, and its
/// plan. `None` without a `[workers]` section.
pub fn props_json() -> Option<serde_json::Value> {
    let node = NODE.get()?;
    Some(serde_json::json!({
        "enabled": node.delegates() && !node.alone.load(Ordering::Acquire),
        "switchable": node.delegates() && !node.is_assigned() && !super::fetch::incomplete(),
        "plan": node.plan().iter().map(|e| serde_json::json!({
            "node": e.node,
            "layers": [e.layer_start, e.layer_end],
        })).collect::<Vec<_>>(),
    }))
}

/// Why this process should not be routed to, if a worker tree says so.
pub fn readiness() -> Option<&'static str> {
    NODE.get().and_then(|node| node.unready())
}

/// The `orangu_server_worker*` metric families, when this process has a
/// `[workers]` section; empty otherwise.
pub fn render_metrics() -> String {
    NODE.get()
        .map(|node| node.render_metrics())
        .unwrap_or_default()
}

/// Nodes' processors and measured speeds, for `/v1/workers`.
fn setups_json(setups: &[NodeSetup]) -> serde_json::Value {
    setups
        .iter()
        .map(|s| {
            serde_json::json!({
                "name": s.name,
                "decode_gb_per_s": (s.decode_rate > 0.0).then_some(s.decode_rate),
                "prompt_gb_per_s": (s.prompt_rate > 0.0).then_some(s.prompt_rate),
                "devices": s.devices.iter().map(|d| serde_json::json!({
                    "kind": d.kind,
                    "name": d.name,
                    "cores": (d.cores > 0).then_some(d.cores),
                    "memory_bytes": (d.memory_bytes > 0).then_some(d.memory_bytes),
                    "in_use": d.in_use,
                })).collect::<Vec<_>>(),
            })
        })
        .collect()
}

/// This process's place in its tree, for `GET /v1/workers`.
pub fn status_json() -> serde_json::Value {
    match NODE.get() {
        Some(node) => node.status(),
        None => serde_json::json!({ "role": "none" }),
    }
}

impl Node {
    /// Starts the node: binds its listener, plans a top-level tree when it
    /// has workers and the model can be split, and starts the thread that
    /// keeps an eye on its workers.
    pub fn start(
        mut settings: NodeSettings,
        model: Arc<dyn ModelForward>,
        loaded: LoadedModel,
    ) -> Result<Arc<Self>> {
        let n_layer = model.config().n_layer;
        let link_to = |addr: &String| {
            Arc::new(ChildLink::with_tls(
                addr.clone(),
                settings.timeout,
                settings.connect_timeout,
                settings.tls.clone(),
            ))
        };
        let links = settings.workers.iter().map(link_to).collect();
        let standby_links = settings.standby.iter().map(link_to).collect();
        let listener = match &settings.listen {
            Some(addr) => Some(
                TcpListener::bind(addr)
                    .with_context(|| format!("failed to bind the [workers] listener on {addr}"))?,
            ),
            None => None,
        };
        let local_addr = listener.as_ref().and_then(|l| l.local_addr().ok());
        let mut layer_bytes = vec![0u64; n_layer];
        let mut non_layer_bytes = 0u64;
        for (name, bytes) in loaded.tensor_sizes() {
            match crate::engine::loader::block_index(name).filter(|il| *il < n_layer) {
                Some(il) => layer_bytes[il] += bytes,
                None => non_layer_bytes += bytes,
            }
        }
        if settings.capacity.setups.is_empty() {
            settings.capacity.setups.push(NodeSetup {
                name: settings.name.clone(),
                ..NodeSetup::default()
            });
        }
        // Measured before a parent can ask (W-86): not for a node sharing by
        // memory, nor for one that has only part of the model on disk and
        // does not know yet which layers it will have.
        if settings.shares != Shares::Memory
            && settings.capacity.setups[0].decode_rate == 0.0
            && model.supports_layer_split()
            && !super::fetch::incomplete()
        {
            match super::speed::measure(model.as_ref(), &layer_bytes) {
                Ok((speed, layers, took)) => {
                    log::info!(
                        "orangu-server: [workers] this node runs {:.2} GB of weights a second \
                         in a decode step, {:.1} in a {}-token prompt chunk (layers {}..{}, \
                         measured in {:.1} s)",
                        speed.decode,
                        speed.prompt,
                        super::speed::PROMPT_TOKENS,
                        layers.start,
                        layers.end,
                        took.as_secs_f64()
                    );
                    settings.capacity.setups[0].decode_rate = speed.decode;
                    settings.capacity.setups[0].prompt_rate = speed.prompt;
                }
                Err(e) => log::warn!(
                    "orangu-server: [workers] could not time this node's layers, its share \
                     follows its memory: {e:#}"
                ),
            }
        }
        let local = if model.supports_layer_split() {
            LayerPipeline::new(model.clone(), 0..n_layer, Vec::new())?
        } else {
            // Never run: a model that cannot be split is never delegated,
            // and refuses every assignment.
            LayerPipeline::unchecked(model.clone(), 0..n_layer)
        };
        let node = Arc::new(Self {
            id: format!("{:016x}", rand::random::<u64>()),
            settings,
            model,
            loaded,
            links,
            standby_links,
            standing_in: Mutex::new(Vec::new()),
            this: OnceLock::new(),
            decision: OnceLock::new(),
            shares_by: Mutex::new("memory"),
            alone: AtomicBool::new(false),
            decode_alone: AtomicBool::new(false),
            head_on_last: AtomicBool::new(false),
            copied: Mutex::new(None),
            top: RwLock::new((Arc::new(local), 0)),
            assignment: Mutex::new(None),
            parent: Mutex::new(None),
            plan: Mutex::new(Vec::new()),
            used: Mutex::new(Vec::new()),
            planning: Mutex::new(Instant::now()),
            delegating: OnceLock::new(),
            layer_bytes,
            non_layer_bytes,
            local_addr,
            metrics: NodeMetrics::default(),
            next_connection: AtomicU64::new(1),
            stop: AtomicBool::new(false),
        });
        let _ = node.this.set(Arc::downgrade(&node));
        if let Some(listener) = listener {
            let node = node.clone();
            std::thread::Builder::new()
                .name("orangu-workers-listen".to_string())
                .spawn(move || node.accept(listener))?;
        }
        if node.delegates() {
            node.plan_top();
        } else if !node.links.is_empty() {
            log::warn!(
                "orangu-server: [workers] lists workers, but the {} architecture cannot be split \
                 across them yet; serving it here alone",
                node.model.config().architecture
            );
        }
        {
            let node = node.clone();
            std::thread::Builder::new()
                .name("orangu-workers-watch".to_string())
                .spawn(move || node.watch())?;
        }
        Ok(node)
    }

    /// Where the listener is bound, when there is one.
    #[cfg(test)]
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    #[cfg(test)]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Whether this node hands layers to workers when it is top-level.
    fn delegates(&self) -> bool {
        !self.links.is_empty() && self.model.supports_layer_split()
    }

    /// The model this node's own API serves: a [`DelegatingModel`] over
    /// its current plan when it has workers to delegate to, `None` when it
    /// runs the model alone.
    pub fn delegating_model(self: &Arc<Self>) -> Option<Arc<DelegatingModel>> {
        if !self.delegates() {
            return None;
        }
        let source: Arc<dyn PipelineSource> = self.clone();
        let model = Arc::new(DelegatingModel::with_source(self.model.clone(), source));
        let _ = self.delegating.set(Arc::downgrade(&model));
        Some(model)
    }

    pub fn is_assigned(&self) -> bool {
        self.assignment.lock().unwrap().is_some()
    }

    fn unready(&self) -> Option<&'static str> {
        if self.is_assigned() {
            return Some("serving a parent orangu-server");
        }
        if super::fetch::incomplete() {
            return Some("the model is only partly downloaded; this node serves only as a worker");
        }
        let used = self.used.lock().unwrap();
        let lost = self
            .links
            .iter()
            .any(|link| !link.is_connected() && used.contains(&link.addr));
        lost.then_some("a worker was lost")
    }

    /// The plan as it stands, one line per node.
    pub fn plan(&self) -> Vec<PlanEntry> {
        self.plan.lock().unwrap().clone()
    }

    fn render_metrics(&self) -> String {
        let links: Vec<LinkSnapshot<'_>> = self
            .links
            .iter()
            .map(|link| LinkSnapshot {
                addr: &link.addr,
                up: link.is_connected(),
                metrics: &link.metrics,
            })
            .collect();
        super::metrics::render(&self.metrics, self.is_assigned(), &links)
    }

    fn status(&self) -> serde_json::Value {
        let assignment = self.assignment.lock().unwrap();
        let plan: Vec<_> = self
            .plan()
            .into_iter()
            .map(|e| {
                serde_json::json!({
                    "node": e.node,
                    "layers": [e.layer_start, e.layer_end],
                })
            })
            .collect();
        let workers: Vec<_> = self
            .links
            .iter()
            .map(|link| {
                let info = link.info();
                serde_json::json!({
                    "address": link.addr,
                    "connected": info.is_some(),
                    "node": info.as_ref().map(|i| i.node.clone()),
                    "subtree_nodes": info.as_ref().map(|i| i.capacity.subtree_nodes),
                    "setups": info.as_ref().map(|i| setups_json(&i.capacity.setups)),
                })
            })
            .collect();
        serde_json::json!({
            "node": self.id,
            "name": self.settings.name,
            "role": if assignment.is_some() { "worker" } else { "top" },
            "alone": self.alone.load(Ordering::Acquire),
            "layers": assignment.as_ref().map(|a| [a.layers.start, a.layers.end]),
            "listen": self.local_addr.map(|a| a.to_string()),
            "model": self.settings.label,
            "quant": self.settings.quant,
            "plan": plan,
            "workers": workers,
            // This node's processors and speed (W-86), and what the last
            // plan's shares followed.
            "setup": setups_json(&self.settings.capacity.setups[..1])[0],
            "shares": *self.shares_by.lock().unwrap(),
            // Where sequences decode once their prompt is through the tree.
            "decode": if self.decode_alone.load(Ordering::Acquire) { "here alone" } else { "through the tree" },
            "head": if self.head_on_last.load(Ordering::Acquire) { "on the node with the final layer" } else { "here" },
            // Spare workers, and whom each stands in for since the last
            // plan, if anyone.
            "standby": self
                .standby_links
                .iter()
                .map(|link| {
                    let standing = self.standing_in.lock().unwrap();
                    serde_json::json!({
                        "address": link.addr,
                        "connected": link.is_connected(),
                        "standing_in_for": standing
                            .iter()
                            .find(|(_, standby)| *standby == link.addr)
                            .map(|(lost, _)| lost.clone()),
                    })
                })
                .collect::<Vec<_>>(),
            // Workers the current plan hands layers to that are not
            // connected: what a repair — the next request, or the worker
            // coming back — is waiting for.
            "lost": self
                .used
                .lock()
                .unwrap()
                .iter()
                .filter(|addr| {
                    self.links
                        .iter()
                        .any(|link| &link.addr == *addr && !link.is_connected())
                })
                .cloned()
                .collect::<Vec<_>>(),
        })
    }

    /// This node and every node below it that it is connected to.
    fn subtree_ids(&self) -> Vec<String> {
        let mut ids = vec![self.id.clone()];
        for link in &self.links {
            if let Some(info) = link.info() {
                ids.extend(info.subtree);
            }
        }
        ids
    }

    fn subtree_capacity(&self) -> Capacity {
        let mut capacity = self.settings.capacity.clone();
        capacity.subtree_nodes = 1;
        capacity.subtree_device_bytes = capacity.device_bytes;
        capacity.subtree_host_bytes = capacity.host_bytes;
        capacity.subtree_budget_bytes = capacity.budget_bytes;
        for info in self.links.iter().filter_map(|link| link.info()) {
            capacity.subtree_nodes += info.capacity.subtree_nodes;
            capacity.subtree_device_bytes += info.capacity.subtree_device_bytes;
            capacity.subtree_host_bytes += info.capacity.subtree_host_bytes;
            capacity.subtree_budget_bytes += info.capacity.subtree_budget_bytes;
            capacity.setups.extend(info.capacity.setups);
        }
        capacity
    }

    /// The speed a share over `setups` — a node, or a worker and every node
    /// below it — runs at, by `[workers].shares`: their rates added up, as
    /// each takes a part in proportion. `0` when one of them did not measure.
    fn rate(&self, setups: &[NodeSetup]) -> f64 {
        let rate = |s: &NodeSetup| match self.settings.shares {
            Shares::Decode => s.decode_rate as f64,
            Shares::Prompt => s.prompt_rate as f64,
            Shares::Memory => 0.0,
        };
        if setups.is_empty() || setups.iter().any(|s| rate(s) <= 0.0) {
            return 0.0;
        }
        setups.iter().map(rate).sum()
    }

    /// Plans `range` over this node and its workers, reached with `path` (the
    /// ids from the root down to this node), assigning each its share. A
    /// worker that cannot be reached, refuses, or does not have the model is
    /// left out, and the rest planned again without it.
    fn plan_subtree(&self, range: Range<usize>, path: &[String], terms: Terms) -> Result<Planned> {
        let secret = self.settings.secret.as_deref();
        let mut candidates: Vec<(Arc<ChildLink>, ChildInfo)> = Vec::new();
        // Standbys filling in for workers that do not answer:
        // `(missing worker, standby)`.
        let mut standing: Vec<(String, String)> = Vec::new();
        if path.len() < MAX_DEPTH {
            let mut missing = Vec::new();
            for link in &self.links {
                match link.ensure_connected(secret, path) {
                    Ok(info) => candidates.push((link.clone(), info)),
                    // Said once per outage: a worker that stays away is
                    // tried again every `readmit`, and the log should not
                    // fill with it.
                    Err(error) if link.note_unreachable() => {
                        log::warn!("orangu-server: worker left out: {error}");
                        missing.push(link.addr.clone());
                    }
                    Err(error) => {
                        log::debug!("orangu-server: worker still out: {error}");
                        missing.push(link.addr.clone());
                    }
                }
            }
            let mut spares = self.standby_links.iter();
            for lost in missing {
                for link in spares.by_ref() {
                    match link.ensure_connected(secret, path) {
                        Ok(info) => {
                            candidates.push((link.clone(), info));
                            standing.push((lost, link.addr.clone()));
                            break;
                        }
                        Err(error) => log::warn!("orangu-server: standby left out: {error}"),
                    }
                }
            }
        }
        // The top-level node also holds the embedding and the output head,
        // which no layer range counts: its budget for layers is what they
        // leave.
        let top = path.len() == 1;
        let own_budget = if top {
            self.settings
                .capacity
                .budget_bytes
                .saturating_sub(self.non_layer_bytes)
        } else {
            self.settings.capacity.budget_bytes
        };
        loop {
            let budgets: Vec<u64> = candidates
                .iter()
                .map(|(_, info)| plan::subtree_budget(&info.capacity))
                .collect();
            // By speed when every node below has measured it (W-86), each
            // share up to what its memory holds; by memory otherwise.
            let mut weights = vec![own_budget];
            weights.extend(&budgets);
            let mut rates = vec![self.rate(&self.settings.capacity.setups[..1])];
            rates.extend(
                candidates
                    .iter()
                    .map(|(_, info)| self.rate(&info.capacity.setups)),
            );
            let by_speed = if self.settings.shares == Shares::Memory {
                None
            } else {
                plan::by_speed(&weights, &rates, plan::bytes_of(&range, &self.layer_bytes))
            };
            *self.shares_by.lock().unwrap() = match (&by_speed, self.settings.shares) {
                (Some(_), Shares::Prompt) => "prompt speed",
                (Some(_), _) => "decode speed",
                (None, _) => "memory",
            };
            if let Some(by_speed) = by_speed {
                weights = by_speed;
            }
            let (local, shares) = plan::plan_shares(
                range.clone(),
                &self.layer_bytes,
                self.settings.local_layers,
                // A budget of nothing still takes a share when every budget
                // is nothing — a tree of nodes that did not say.
                weights[0].max(1),
                &weights[1..].iter().map(|b| (*b).max(1)).collect::<Vec<_>>(),
                &|at| self.model.split_allowed(at),
            );
            let mut entries = vec![PlanEntry {
                node: self.settings.name.clone(),
                layer_start: local.start as u32,
                layer_end: local.end as u32,
            }];
            let mut stages: Vec<Box<dyn Stage>> = Vec::new();
            let mut used = Vec::new();
            let mut failed = None;
            for (i, ((link, _), share)) in candidates.iter().zip(&shares).enumerate() {
                if share.is_empty() {
                    continue;
                }
                match self.assign(link, share.clone(), terms) {
                    Ok(plan) => {
                        entries.extend(plan);
                        used.push(link.addr.clone());
                        stages.push(self.stage_for(link, share.clone(), path, terms));
                    }
                    Err(error) => {
                        log::warn!("orangu-server: worker left out: {error}");
                        failed = Some(i);
                        break;
                    }
                }
            }
            match failed {
                Some(i) => {
                    let (link, _) = candidates.remove(i);
                    link.disconnect();
                }
                None => {
                    self.report_fit(&local, own_budget, &candidates, &shares, &budgets);
                    let pipeline = LayerPipeline::new(self.model.clone(), local, stages)?
                        .named(&self.settings.name);
                    standing.retain(|(_, standby)| used.contains(standby));
                    if !standing.is_empty() {
                        log::warn!(
                            "orangu-server: standing in for workers that do not answer: {}",
                            standing
                                .iter()
                                .map(|(lost, standby)| format!("{standby} for {lost}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                    return Ok(Planned {
                        pipeline: Arc::new(pipeline),
                        entries,
                        used,
                        standing,
                    });
                }
            }
        }
    }

    /// The stage that sends `layers` to the worker behind `link`: guarded
    /// by a standby when this node has any (`super::guard`).
    fn stage_for(
        &self,
        link: &Arc<ChildLink>,
        layers: Range<usize>,
        path: &[String],
        terms: Terms,
    ) -> Box<dyn Stage> {
        let stage = MessageStage::over(link.clone(), layers.clone(), terms.activations);
        if self.standby_links.is_empty() {
            return Box::new(stage);
        }
        let node = self.this.get().cloned().unwrap_or_default();
        let path = path.to_vec();
        let take: TakeStandby = Box::new(move |layers, lost| {
            let node = node
                .upgrade()
                .ok_or_else(|| anyhow::anyhow!("the node has stopped"))?;
            node.take_standby(layers, &path, terms, lost)
        });
        let width = self.model.config().n_embd;
        match GuardedStage::new(Arc::new(stage), width, take) {
            Some(guarded) => Box::new(guarded),
            None => {
                log::warn!(
                    "orangu-server: [workers].standby needs {}-wide rows to be whole q8_0 \
                     blocks; {} is not guarded",
                    width,
                    link.addr
                );
                Box::new(MessageStage::over(link.clone(), layers, terms.activations))
            }
        }
    }

    /// Assigns `layers` to the first free standby that answers and holds
    /// the same model, in place of the lost worker `lost`, and records it in
    /// the plan. The stage returned sends to it.
    fn take_standby(
        &self,
        layers: Range<usize>,
        path: &[String],
        terms: Terms,
        lost: &str,
    ) -> Result<Arc<dyn EncodedStage>> {
        let secret = self.settings.secret.as_deref();
        let mut standing = self.standing_in.lock().unwrap();
        for link in &self.standby_links {
            if standing.iter().any(|(_, standby)| *standby == link.addr) {
                continue;
            }
            if let Err(error) = link.ensure_connected(secret, path) {
                log::warn!("orangu-server: standby left out: {error}");
                continue;
            }
            let plan = match self.assign(link, layers.clone(), terms) {
                Ok(plan) => plan,
                Err(error) => {
                    log::warn!("orangu-server: standby left out: {error}");
                    link.disconnect();
                    continue;
                }
            };
            standing.push((lost.to_string(), link.addr.clone()));
            self.metrics.takeovers.fetch_add(1, Ordering::Relaxed);
            for used in self.used.lock().unwrap().iter_mut() {
                if used == lost {
                    *used = link.addr.clone();
                }
            }
            let mut entries = self.plan.lock().unwrap();
            entries.retain(|e| {
                e.node == self.settings.name
                    || (e.layer_end as usize) <= layers.start
                    || (e.layer_start as usize) >= layers.end
            });
            entries.extend(plan);
            entries.sort_by_key(|e| e.layer_start);
            log::info!("orangu-server: workers plan: {}", describe(&entries));
            return Ok(Arc::new(MessageStage::over(
                link.clone(),
                layers,
                terms.activations,
            )));
        }
        anyhow::bail!(
            "no standby is free to take layers {}..{}",
            layers.start,
            layers.end
        )
    }

    /// Records the standbys a new plan uses; any other that stood in is
    /// let go, and serves its own API again.
    fn set_standing(&self, standing: Vec<(String, String)>) {
        let old = std::mem::replace(&mut *self.standing_in.lock().unwrap(), standing.clone());
        for (_, standby) in old {
            if standing.iter().any(|(_, kept)| *kept == standby) {
                continue;
            }
            if let Some(link) = self.standby_links.iter().find(|l| l.addr == standby) {
                link.disconnect();
            }
        }
    }

    /// Warns about every share of a plan that is more than its node (or
    /// worker subtree) said it can hold. The plan stands — a share too
    /// large for memory still runs, paged in and out — but the operator
    /// should know that the tree is short of room, and where.
    fn report_fit(
        &self,
        local: &Range<usize>,
        own_budget: u64,
        candidates: &[(Arc<ChildLink>, ChildInfo)],
        shares: &[Range<usize>],
        budgets: &[u64],
    ) {
        let mut over = Vec::new();
        let local_bytes = plan::bytes_of(local, &self.layer_bytes);
        if local_bytes > own_budget {
            over.push(format!(
                "{} (this node) gets {} for {}",
                self.settings.name,
                orangu::format::format_bytes(local_bytes),
                orangu::format::format_bytes(own_budget)
            ));
        }
        for (((link, _), share), budget) in candidates.iter().zip(shares).zip(budgets) {
            let bytes = plan::bytes_of(share, &self.layer_bytes);
            if bytes > *budget {
                over.push(format!(
                    "{} gets {} for {}",
                    link.addr,
                    orangu::format::format_bytes(bytes),
                    orangu::format::format_bytes(*budget)
                ));
            }
        }
        if !over.is_empty() {
            log::warn!(
                "orangu-server: the workers are short of memory for this model — {}; the \
                 layers over budget will be paged in from disk as they run",
                over.join(", ")
            );
        }
    }

    /// Assigns `layers` to the worker behind `link`, and checks that what
    /// comes back is the same model.
    fn assign(
        &self,
        link: &ChildLink,
        layers: Range<usize>,
        terms: Terms,
    ) -> Result<Vec<PlanEntry>, WorkerError> {
        let expected = identity::model_identity(
            &self.loaded,
            &self.settings.label,
            &self.settings.quant,
            layers.clone(),
        );
        // An assignment may fetch its layers first (`super::fetch`): gigabytes,
        // at whatever the worker's link to the Hub gives.
        let reply = link.request_within(
            &Message::Assign(Assign {
                model: self.settings.label.clone(),
                identity: expected.clone(),
                layer_start: layers.start as u32,
                layer_end: layers.end as u32,
                n_ctx: terms.n_ctx as u32,
                kv_type: String::new(),
                slots: terms.slots as u32,
                activations: terms.activations,
                prompt_weights: Some(terms.weights.bits()),
            }),
            ASSIGN_TIMEOUT.max(self.settings.timeout),
        )?;
        let with_path = |mut error: WorkerError| {
            if error.path.first() != Some(&link.addr) {
                error.path.insert(0, link.addr.clone());
            }
            error
        };
        match reply {
            Message::AssignAck { identity, plan } => {
                // Checked here as well as there: a parent does not take a
                // worker's word for which model it has.
                identity::check(&expected, &identity, &layers).map_err(with_path)?;
                Ok(plan)
            }
            Message::Error(error) => Err(with_path(error)),
            other => Err(with_path(WorkerError::new(
                ErrorCode::Internal,
                format!("unexpected answer to an assignment: {other:?}"),
            ))),
        }
    }

    /// Plans this node as top-level: every layer, over itself and its
    /// workers.
    pub fn plan_top(&self) {
        let mut last = self.planning.lock().unwrap();
        self.plan_top_locked();
        *last = Instant::now();
    }

    fn plan_top_locked(&self) {
        if !self.delegates() || self.is_assigned() || self.alone.load(Ordering::Acquire) {
            return;
        }
        let n_layer = self.model.config().n_layer;
        let terms = Terms {
            n_ctx: self.settings.n_ctx,
            slots: self.settings.slots,
            activations: self.settings.activations,
            weights: *self
                .decision
                .get_or_init(|| prompt_weights::decide(&self.loaded, self.settings.prompt_weights)),
        };
        self.metrics.plans.fetch_add(1, Ordering::Relaxed);
        let planned = self.plan_subtree(0..n_layer, std::slice::from_ref(&self.id), terms);
        let (pipeline, entries, used, standing) = match planned {
            Ok(p) => (p.pipeline, p.entries, p.used, p.standing),
            Err(error) => {
                log::warn!("orangu-server: planning the workers failed, serving alone: {error:#}");
                let local = LayerPipeline::new(self.model.clone(), 0..n_layer, Vec::new())
                    .expect("a splittable model runs every layer alone");
                let entry = PlanEntry {
                    node: self.settings.name.clone(),
                    layer_start: 0,
                    layer_end: n_layer as u32,
                };
                (Arc::new(local), vec![entry], Vec::new(), Vec::new())
            }
        };
        if *self.plan.lock().unwrap() != entries {
            log::info!("orangu-server: workers plan: {}", describe(&entries));
        }
        let decode_alone = self.decides_to_decode_alone(&entries, &used);
        if decode_alone != self.decode_alone.swap(decode_alone, Ordering::AcqRel)
            || *self.plan.lock().unwrap() != entries
        {
            log::info!(
                "orangu-server: workers: sequences decode {} once their prompt is through the tree",
                if decode_alone {
                    "on this node alone"
                } else {
                    "through the tree"
                }
            );
        }
        let head_on_last = self.decides_head_on_last(&entries, &used);
        if head_on_last != self.head_on_last.swap(head_on_last, Ordering::AcqRel) {
            log::info!(
                "orangu-server: workers: the output head runs on {}",
                if head_on_last {
                    "the node with the final layer"
                } else {
                    "this node"
                }
            );
        }
        // Decoding here alone keeps every layer's weights at hand.
        self.release_unused(
            &if decode_alone {
                0..n_layer
            } else {
                pipeline.local_layers()
            },
            true,
        );
        self.copy_prompt_weights(&pipeline.local_layers(), terms.weights);
        let mut top = self.top.write().unwrap();
        *top = (pipeline, top.1 + 1);
        drop(top);
        self.set_standing(standing);
        *self.plan.lock().unwrap() = entries;
        *self.used.lock().unwrap() = used;
    }

    /// Whether sequences decode on this node alone once their prompt is
    /// through the tree (W-60), by `[workers].decode`: only when the plan
    /// uses workers, every one of them can send rows back, and this node's
    /// budget holds the whole model — and, with `auto`, when its measured
    /// decode speed takes a token through every layer sooner than the plan's
    /// nodes take it through theirs, each at its own speed.
    fn decides_to_decode_alone(&self, entries: &[PlanEntry], used: &[String]) -> bool {
        let working: Vec<&PlanEntry> = entries
            .iter()
            .filter(|e| e.layer_start < e.layer_end)
            .collect();
        if self.settings.decode == DecodeOn::Tree || working.len() < 2 {
            return false;
        }
        let rows = self.links.iter().chain(&self.standby_links).all(|link| {
            link.info()
                .is_none_or(|info| info.features & super::protocol::FEATURE_ROWS != 0)
                || !used.contains(&link.addr)
        });
        let total: u64 = self.layer_bytes.iter().sum();
        if !rows || self.settings.capacity.budget_bytes < total + self.non_layer_bytes {
            return false;
        }
        if self.settings.decode == DecodeOn::Top {
            return true;
        }
        let mut setups = self.settings.capacity.setups.clone();
        for link in &self.links {
            if let Some(info) = link.info() {
                setups.extend(info.capacity.setups);
            }
        }
        let rate = |name: &str| {
            setups
                .iter()
                .find(|s| s.name == name)
                .map(|s| s.decode_rate as f64)
                .filter(|r| *r > 0.0)
        };
        let Some(own) = rate(&self.settings.name) else {
            return false;
        };
        let mut tree = 0.0;
        for entry in working {
            let Some(r) = rate(&entry.node) else {
                return false;
            };
            let layers = entry.layer_start as usize..entry.layer_end as usize;
            tree += plan::bytes_of(&layers, &self.layer_bytes) as f64 / r;
        }
        total as f64 / own < tree
    }

    /// Whether the node running the model's final layer applies the output
    /// head (W-61), by `[workers].head`: only when that node is a worker of
    /// this plan and every worker used can send logits back — and, with
    /// `auto`, when its measured decode speed beats this node's: the head is
    /// a matrix as wide as the vocabulary, read once a token.
    fn decides_head_on_last(&self, entries: &[PlanEntry], used: &[String]) -> bool {
        let n_layer = self.model.config().n_layer as u32;
        let Some(last) = entries
            .iter()
            .find(|e| e.layer_start < e.layer_end && e.layer_end == n_layer)
        else {
            return false;
        };
        if self.settings.head == HeadOn::Top || last.node == self.settings.name {
            return false;
        }
        let head = self.links.iter().chain(&self.standby_links).all(|link| {
            !used.contains(&link.addr)
                || link
                    .info()
                    .is_some_and(|info| info.features & super::protocol::FEATURE_HEAD != 0)
        });
        if !head {
            return false;
        }
        if self.settings.head == HeadOn::Last {
            return true;
        }
        let mut setups = self.settings.capacity.setups.clone();
        for link in &self.links {
            if let Some(info) = link.info() {
                setups.extend(info.capacity.setups);
            }
        }
        let rate = |name: &str| {
            setups
                .iter()
                .find(|s| s.name == name)
                .map(|s| s.decode_rate)
                .filter(|r| *r > 0.0)
        };
        matches!((rate(&last.node), rate(&self.settings.name)), (Some(there), Some(here)) if there > here)
    }

    fn set_alone(&self, alone: bool) -> Result<(), String> {
        if self.is_assigned() {
            return Err("this node is working for a parent".to_string());
        }
        if !self.delegates() {
            return Err(
                "this node has no workers, or its model cannot be split across them".to_string(),
            );
        }
        if super::fetch::incomplete() {
            return Err("this node has only part of the model".to_string());
        }
        if !alone {
            self.alone.store(false, Ordering::Release);
            self.plan_top();
            log::info!(
                "orangu-server: [workers] serving through the tree again: {}",
                describe(&self.plan())
            );
            return Ok(());
        }
        let mut last = self.planning.lock().unwrap();
        self.alone.store(true, Ordering::Release);
        let n_layer = self.model.config().n_layer;
        let local = LayerPipeline::new(self.model.clone(), 0..n_layer, Vec::new())
            .expect("a splittable model runs every layer alone");
        let mut top = self.top.write().unwrap();
        *top = (Arc::new(local), top.1 + 1);
        drop(top);
        // Let go: each worker drops its sessions and serves its own API.
        for link in self.links.iter().chain(&self.standby_links) {
            link.disconnect();
        }
        self.set_standing(Vec::new());
        *self.plan.lock().unwrap() = vec![PlanEntry {
            node: self.settings.name.clone(),
            layer_start: 0,
            layer_end: n_layer as u32,
        }];
        self.used.lock().unwrap().clear();
        let weights = *self
            .decision
            .get_or_init(|| prompt_weights::decide(&self.loaded, self.settings.prompt_weights));
        self.copy_prompt_weights(&(0..n_layer), weights);
        *last = Instant::now();
        log::info!("orangu-server: [workers] serving alone; the workers are let go");
        Ok(())
    }

    /// Lets go of the weights this node's plan no longer runs: every layer
    /// outside `local`, and — unless `keep_head`: the top-level node, which
    /// embeds and samples, or the node with the final layer (W-61) — the
    /// embedding and output head. The kernel faults them
    /// back if a later plan wants them.
    fn release_unused(&self, local: &Range<usize>, keep_head: bool) {
        let n_layer = self.model.config().n_layer;
        let released =
            self.loaded.release_tensors(|name| {
                match crate::engine::loader::block_index(name).filter(|il| *il < n_layer) {
                    Some(il) => local.contains(&il),
                    None => keep_head,
                }
            });
        log::debug!(
            "orangu-server: workers: released {} of weights outside layers {}..{}",
            orangu::format::format_bytes(released),
            local.start,
            local.end
        );
    }

    /// Builds the prompt-weight copies of the layers this node runs, by the
    /// top-level node's decision (W-85): a node copies only its own layers,
    /// and every node rounds as the top does. Kept while a plan leaves the
    /// layers and the decision as they were.
    fn copy_prompt_weights(&self, local: &Range<usize>, weights: Decision) {
        let mut copied = self.copied.lock().unwrap();
        if copied.as_ref() == Some(&(local.clone(), weights)) {
            return;
        }
        prompt_weights::build_for_layers(
            &self.loaded,
            weights,
            local.clone(),
            self.model.config().n_layer,
            "as the top-level node decided",
        );
        *copied = Some((local.clone(), weights));
    }

    /// Answers a parent's `Assign`.
    fn handle_assign(&self, assign: Assign, parent_path: &[String]) -> Message {
        let n_layer = self.model.config().n_layer;
        let layers = assign.layer_start as usize..assign.layer_end as usize;
        if !self.model.supports_layer_split() {
            return error(
                ErrorCode::Unsupported,
                format!(
                    "the {} architecture cannot be split across workers yet",
                    self.model.config().architecture
                ),
            );
        }
        if layers.is_empty() || layers.end > n_layer {
            return error(
                ErrorCode::Unsupported,
                format!("layers {layers:?} of a model with {n_layer}"),
            );
        }
        for at in [layers.start, layers.end] {
            if at != 0 && at != n_layer && !self.model.split_allowed(at) {
                return error(
                    ErrorCode::Unsupported,
                    format!(
                        "layers {layers:?}: this model cannot be cut before layer {at}, where a \
                         later layer reads an earlier one's KV cache"
                    ),
                );
            }
        }
        // A partly downloaded model fetches the layers first: the check
        // below reads them.
        if let Err(e) = super::fetch::layers(&layers) {
            log::warn!("orangu-server: could not fetch layers {layers:?}: {e:#}");
            return error(
                ErrorCode::Internal,
                format!("fetching layers {}..{}: {e:#}", layers.start, layers.end),
            );
        }
        let own = identity::model_identity(
            &self.loaded,
            &self.settings.label,
            &self.settings.quant,
            layers.clone(),
        );
        if let Err(error) = identity::check(&assign.identity, &own, &layers) {
            log::warn!("orangu-server: refused an assignment: {error}");
            return Message::Error(error);
        }
        let mut last = self.planning.lock().unwrap();
        let mut path = parent_path.to_vec();
        path.push(self.id.clone());
        let terms = Terms {
            n_ctx: assign.n_ctx as usize,
            slots: assign.slots.max(1) as usize,
            activations: assign.activations,
            // A parent that does not say: the file's weights, rather than a
            // measurement of layers this node may not have fetched.
            weights: assign
                .prompt_weights
                .map(Decision::from_bits)
                .unwrap_or_default(),
        };
        self.metrics.plans.fetch_add(1, Ordering::Relaxed);
        let (pipeline, entries, used, standing) =
            match self.plan_subtree(layers.clone(), &path, terms) {
                Ok(p) => (p.pipeline, p.entries, p.used, p.standing),
                Err(e) => return error(ErrorCode::Internal, format!("{e:#}")),
            };
        self.set_standing(standing);
        // A node with the final layer keeps the output head, which it may be
        // asked to apply (W-61).
        let local = pipeline.local_layers();
        self.release_unused(&local, !local.is_empty() && local.end == n_layer);
        self.copy_prompt_weights(&pipeline.local_layers(), terms.weights);
        let store = Arc::new(
            SessionStore::new(pipeline, terms.n_ctx, terms.slots * 2).named(&self.settings.name),
        );
        *self.assignment.lock().unwrap() = Some(Assignment {
            store,
            layers: layers.clone(),
            terms,
            path: path.clone(),
        });
        // The top-level plan's sessions are gone with it; bumped so any
        // that come back rebuild on a fresh one.
        let mut top = self.top.write().unwrap();
        top.1 += 1;
        drop(top);
        log::info!(
            "orangu-server: working for a parent on layers {}..{} (this API is off until it \
             lets go): {}",
            layers.start,
            layers.end,
            describe(&entries)
        );
        *self.plan.lock().unwrap() = entries.clone();
        *self.used.lock().unwrap() = used;
        *last = Instant::now();
        Message::AssignAck {
            identity: own,
            plan: entries,
        }
    }

    /// The parent went away: drop its sessions, and be top-level again.
    fn unassign(self: &Arc<Self>) {
        let released = self.assignment.lock().unwrap().take();
        if released.is_none() {
            return;
        }
        drop(released);
        log::info!("orangu-server: the parent let go; serving this node's own API again");
        self.plan.lock().unwrap().clear();
        self.used.lock().unwrap().clear();
        if self.delegates() {
            let node = self.clone();
            let _ = std::thread::Builder::new()
                .name("orangu-workers-plan".to_string())
                .spawn(move || node.plan_top());
        }
    }

    fn accept(self: Arc<Self>, listener: TcpListener) {
        for stream in listener.incoming() {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            let Ok(stream) = stream else { continue };
            let node = self.clone();
            let _ = std::thread::Builder::new()
                .name("orangu-workers-parent".to_string())
                .spawn(move || node.serve_parent(stream));
        }
    }

    /// One connection from a would-be parent: the handshake, then its
    /// requests until it goes away.
    fn serve_parent(self: Arc<Self>, stream: TcpStream) {
        let connection = self.next_connection.fetch_add(1, Ordering::Relaxed);
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(self.settings.timeout));
        let Ok(parent_handle) = stream.try_clone() else {
            return;
        };
        let split = match transport::server(self.settings.tls.as_ref(), stream) {
            Ok(split) => split,
            Err(error) => {
                log::warn!("orangu-server: refused a parent: TLS: {error:#}");
                return;
            }
        };
        let mut reader = BufReader::new(split.reader);
        let mut writer = split.writer;
        let Some(hello) = self.handshake(&mut reader, &mut *writer, connection, parent_handle)
        else {
            return;
        };
        // A parent may be idle for as long as it likes.
        let _ = split.tcp.set_read_timeout(None);
        let writer = Arc::new(Mutex::new(writer));
        let answer = |writer: &Mutex<Box<dyn Write + Send>>, id: u64, reply: &Message| {
            let mut writer = writer.lock().unwrap();
            write_message(&mut **writer, id, reply).is_ok()
        };
        while let Ok((id, message, _)) = read_frame(&mut reader) {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            let reply = match message {
                // Planning is one thing at a time, and so are pings — which
                // is what keeps them answered while forwards run.
                Message::Assign(assign) => self.handle_assign(assign, &hello),
                Message::Ping { nonce } => Message::Pong { nonce },
                message => {
                    let store = self
                        .assignment
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|a| a.store.clone());
                    let Some(store) = store else {
                        if !answer(
                            &writer,
                            id,
                            &error(ErrorCode::UnknownSession, "no layers assigned here"),
                        ) {
                            break;
                        }
                        continue;
                    };
                    // Every forward on a thread of its own, answered when it
                    // is done: sequences overlap here as they do at the
                    // parent. One sequence's forwards never overlap each
                    // other — its parent waits for each before the next.
                    let writer = writer.clone();
                    let spawned = std::thread::Builder::new()
                        .name("orangu-workers-forward".to_string())
                        .spawn(move || {
                            super::clock::hold();
                            // In the request pool, as a request at the top
                            // runs: parallel regions split from a pool
                            // worker's own deque rather than being handed
                            // in from outside (`engine::cpu_pools`).
                            let reply =
                                crate::engine::cpu_pools::for_request(|| store.handle(message));
                            super::clock::hold();
                            let mut writer = writer.lock().unwrap();
                            let _ = write_message(&mut **writer, id, &reply);
                        });
                    if spawned.is_err() {
                        break;
                    }
                    continue;
                }
            };
            if !answer(&writer, id, &reply) {
                break;
            }
        }
        let _ = split.tcp.shutdown(std::net::Shutdown::Both);
        let mut parent = self.parent.lock().unwrap();
        if parent.as_ref().is_some_and(|p| p.connection == connection) {
            *parent = None;
            drop(parent);
            self.unassign();
        }
    }

    /// The worker's half of the handshake. Returns the parent's path on
    /// success, having claimed this node for it.
    fn handshake(
        &self,
        reader: &mut BufReader<Box<dyn Read + Send>>,
        writer: &mut dyn Write,
        connection: u64,
        tcp: TcpStream,
    ) -> Option<Vec<String>> {
        let refuse = |writer: &mut dyn Write, error: WorkerError| {
            log::warn!("orangu-server: refused a parent: {error}");
            let _ = write_message(writer, 0, &Message::Error(error));
            None
        };
        let hello = match read_message(reader) {
            Ok(Message::Hello(hello)) => hello,
            _ => return None,
        };
        if hello.version != PROTOCOL_VERSION {
            return refuse(
                writer,
                WorkerError::new(
                    ErrorCode::Unsupported,
                    format!(
                        "this node speaks worker protocol {PROTOCOL_VERSION}, the parent {}: run \
                         the same orangu-server release on every node",
                        hello.version
                    ),
                ),
            );
        }
        let parent_node = hello.path.last().cloned().unwrap_or_default();
        if let Some(parent) = self.parent.lock().unwrap().as_ref()
            && parent.node != parent_node
        {
            return refuse(
                writer,
                WorkerError::new(ErrorCode::Busy, "already working for another parent"),
            );
        }
        if let Err(error) = check_path(&hello.path, &self.subtree_ids()) {
            return refuse(writer, error);
        }
        let secret = self.settings.secret.as_deref();
        let nonce = auth::nonce();
        let ack = Message::HelloAck(HelloAck {
            version: PROTOCOL_VERSION,
            features: FEATURES,
            nonce,
            proof: auth::worker_proof(secret, &hello.nonce, &nonce),
            node: self.id.clone(),
            subtree: self.subtree_ids(),
            capacity: self.subtree_capacity(),
        });
        write_message(writer, 0, &ack).ok()?;
        let proof = match read_message(reader) {
            Ok(Message::Auth { proof }) => proof,
            _ => return None,
        };
        if !auth::worker_accepts(secret, &hello.nonce, &nonce, &proof) {
            return refuse(
                writer,
                WorkerError::new(
                    ErrorCode::Unauthorized,
                    "the parent could not prove [workers].secret",
                ),
            );
        }
        {
            let mut parent = self.parent.lock().unwrap();
            // The same parent coming back — its old connection is dead —
            // takes over; anyone else is turned away.
            match parent.as_ref() {
                Some(p) if p.node != parent_node => {
                    drop(parent);
                    return refuse(
                        writer,
                        WorkerError::new(ErrorCode::Busy, "already working for another parent"),
                    );
                }
                Some(p) => {
                    let _ = p.stream.shutdown(std::net::Shutdown::Both);
                }
                None => {}
            }
            *parent = Some(Parent {
                connection,
                node: parent_node,
                stream: tcp,
            });
        }
        write_message(writer, 0, &Message::Ready).ok()?;
        Some(hello.path)
    }

    /// Pings idle workers, reclaims forgotten sessions, and — as top-level,
    /// between requests — takes back workers that were lost.
    fn watch(self: Arc<Self>) {
        while !self.stop.load(Ordering::Relaxed) {
            std::thread::sleep(self.settings.maintenance);
            for link in &self.links {
                link.ping_if_idle();
            }
            let store = self
                .assignment
                .lock()
                .unwrap()
                .as_ref()
                .map(|a| a.store.clone());
            if let Some(store) = store {
                store.evict_idle(SESSION_IDLE);
                if self.delegates() && store.quiet(QUIET) && self.missing_and_due() {
                    self.replan_assignment();
                }
                continue;
            }
            if !self.delegates() || self.alone.load(Ordering::Acquire) {
                continue;
            }
            let idle = self
                .delegating
                .get()
                .and_then(Weak::upgrade)
                .is_none_or(|model| model.quiet(QUIET));
            if idle && self.missing_and_due() {
                self.plan_top();
            }
        }
    }

    /// Whether a configured worker is not connected, and it is time to try
    /// taking it back.
    ///
    /// While a standby stands in for a lost worker, only when that worker
    /// answers again: a plan without it would drop the standby too, and
    /// shrink the tree to what is left.
    fn missing_and_due(&self) -> bool {
        if self.planning.lock().unwrap().elapsed() < self.settings.readmit {
            return false;
        }
        let standing = !self.standing_in.lock().unwrap().is_empty();
        let path = self
            .assignment
            .lock()
            .unwrap()
            .as_ref()
            .map(|a| a.path.clone())
            .unwrap_or_else(|| vec![self.id.clone()]);
        let secret = self.settings.secret.as_deref();
        self.links.iter().any(|link| {
            !link.is_connected() && (!standing || link.ensure_connected(secret, &path).is_ok())
        })
    }

    /// Plans the assigned range again over the workers there are now — the
    /// same range, so the parent is none the wiser — once the node is
    /// quiet ([`QUIET`]). A worker lost below this node, and back, is taken
    /// back here: the parent only sees its own workers. Sessions still held
    /// go with the old plan; the parent rebuilds any it takes up again.
    fn replan_assignment(&self) {
        let mut last = self.planning.lock().unwrap();
        let current = self.assignment.lock().unwrap().as_ref().map(|a| {
            (
                a.layers.clone(),
                a.terms,
                a.path.clone(),
                a.store.quiet(QUIET),
            )
        });
        let Some((layers, terms, path, true)) = current else {
            return;
        };
        self.metrics.plans.fetch_add(1, Ordering::Relaxed);
        match self.plan_subtree(layers.clone(), &path, terms) {
            Ok(Planned {
                pipeline,
                entries,
                used,
                standing,
            }) => {
                let local = pipeline.local_layers();
                let n_layer = self.model.config().n_layer;
                self.release_unused(&local, !local.is_empty() && local.end == n_layer);
                self.copy_prompt_weights(&pipeline.local_layers(), terms.weights);
                let store = Arc::new(
                    SessionStore::new(pipeline, terms.n_ctx, terms.slots * 2)
                        .named(&self.settings.name),
                );
                if let Some(assignment) = self.assignment.lock().unwrap().as_mut() {
                    assignment.store = store;
                }
                if *self.plan.lock().unwrap() != entries {
                    log::info!(
                        "orangu-server: layers {}..{} planned again: {}",
                        layers.start,
                        layers.end,
                        describe(&entries)
                    );
                }
                *self.plan.lock().unwrap() = entries;
                *self.used.lock().unwrap() = used;
                self.set_standing(standing);
            }
            Err(error) => log::warn!("orangu-server: planning the workers failed: {error:#}"),
        }
        *last = Instant::now();
    }

    /// Stops listening, lets go of the parent and the workers. For tests,
    /// which run several nodes in one process.
    #[cfg(test)]
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(addr) = self.local_addr {
            let _ = TcpStream::connect_timeout(&addr, Duration::from_secs(1));
        }
        if let Some(parent) = self.parent.lock().unwrap().take() {
            let _ = parent.stream.shutdown(std::net::Shutdown::Both);
        }
        for link in &self.links {
            link.disconnect();
        }
    }
}

impl PipelineSource for Node {
    fn current(&self) -> (Arc<LayerPipeline>, u64) {
        self.top.read().unwrap().clone()
    }

    fn recover(&self, generation: u64) -> bool {
        let mut last = self.planning.lock().unwrap();
        if self.top.read().unwrap().1 != generation {
            return true;
        }
        if self.is_assigned() {
            return false;
        }
        self.metrics.recoveries.fetch_add(1, Ordering::Relaxed);
        self.plan_top_locked();
        *last = Instant::now();
        true
    }

    fn decode_alone(&self) -> bool {
        self.decode_alone.load(Ordering::Acquire)
    }

    fn head_on_last(&self) -> bool {
        self.head_on_last.load(Ordering::Acquire)
    }

    fn handover_failed(&self) {
        if self.decode_alone.swap(false, Ordering::AcqRel) {
            log::warn!(
                "orangu-server: workers: sequences decode through the tree until the next plan"
            );
        }
    }
}

fn error(code: ErrorCode, message: impl Into<String>) -> Message {
    Message::Error(WorkerError::new(code, message))
}

/// `a 0..8, b 8..20, c 20..32`.
pub fn describe(entries: &[PlanEntry]) -> String {
    entries
        .iter()
        .filter(|e| e.layer_start < e.layer_end)
        .map(|e| format!("{} {}..{}", e.node, e.layer_start, e.layer_end))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::pipeline::fixture::{self, N_CTX, N_LAYER, Variant};

    fn settings(name: &str, workers: Vec<String>, secret: Option<&str>) -> NodeSettings {
        NodeSettings {
            name: name.to_string(),
            listen: Some("127.0.0.1:0".to_string()),
            workers,
            standby: Vec::new(),
            secret: secret.map(str::to_string),
            activations: ActivationFormat::F32,
            timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(2),
            local_layers: LocalLayers::Auto,
            n_ctx: N_CTX,
            slots: 2,
            label: "fixture".to_string(),
            quant: "F32".to_string(),
            capacity: Capacity {
                backend: "cpu".to_string(),
                host_bytes: 1 << 30,
                budget_bytes: plan::budget(0, 1 << 30),
                ..Capacity::default()
            },
            shares: Shares::Memory,
            decode: DecodeOn::Tree,
            head: HeadOn::Top,
            maintenance: Duration::from_millis(50),
            readmit: Duration::from_millis(200),
            tls: None,
            prompt_weights: prompt_weights::PromptWeights::File,
        }
    }

    fn node(name: &str, workers: &[&Arc<Node>], variant: Variant) -> Arc<Node> {
        node_with(
            settings(
                name,
                workers
                    .iter()
                    .map(|w| w.local_addr().unwrap().to_string())
                    .collect(),
                None,
            ),
            variant,
        )
    }

    fn node_with(settings: NodeSettings, variant: Variant) -> Arc<Node> {
        let loaded = fixture::loaded(variant);
        let model: Arc<dyn ModelForward> = Arc::new(
            crate::engine::arch::llama::LlamaModel::load_with_backend(
                &loaded,
                Arc::new(crate::engine::backend::CpuBackend),
            )
            .unwrap(),
        );
        Node::start(settings, model, fixture::loaded(variant)).unwrap()
    }

    fn run(model: &dyn ModelForward, tokens: &[u32], prompt: usize) -> Vec<Vec<f32>> {
        let mut cache = model.new_kv_cache(N_CTX);
        let mut out = vec![model.forward(&mut cache, &tokens[..prompt], 0, 0).unwrap()];
        for (pos, token) in tokens.iter().enumerate().skip(prompt) {
            out.push(model.forward(&mut cache, &[*token], pos, 0).unwrap());
        }
        out
    }

    fn covered(plan: &[PlanEntry]) -> usize {
        plan.iter()
            .map(|e| (e.layer_end - e.layer_start) as usize)
            .sum()
    }

    /// A pyramid over TCP — a top-level node, a middle node, and a leaf
    /// below it — answers bit for bit what the model does alone, and the
    /// nodes below turn their own APIs off.
    #[test]
    fn a_pyramid_over_tcp_is_exact() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let plan = top.plan();
        assert_eq!(covered(&plan), N_LAYER, "{}", describe(&plan));
        assert!(
            plan.iter()
                .any(|e| e.node == "leaf" && e.layer_start < e.layer_end)
        );
        assert!(middle.is_assigned() && leaf.is_assigned() && !top.is_assigned());

        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        assert_eq!(run(split.as_ref(), &tokens, 5), whole);
        assert_eq!(split.active_sessions(), 0);
        for n in [&top, &middle, &leaf] {
            n.stop();
        }
    }

    /// Shares follow memory: a worker with three times the top's budget
    /// gets three times the layers.
    #[test]
    fn shares_follow_memory() {
        let mut big = settings("big", vec![], None);
        big.capacity.budget_bytes = 3 << 30;
        let big = node_with(big, Variant::default());
        let mut top = settings("top", vec![big.local_addr().unwrap().to_string()], None);
        top.capacity.budget_bytes = 1 << 30;
        let top = node_with(top, Variant::default());
        assert_eq!(describe(&top.plan()), format!("top 0..1, big 1..{N_LAYER}"));
        big.stop();
        top.stop();
    }

    /// Several sequences at once through one pyramid — their forwards in
    /// flight together on every link, answered in whatever order they
    /// finish — each come out exactly as they would alone.
    #[test]
    fn concurrent_sequences_through_one_tree_are_each_exact() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        // Room for all six: a worker holds what its parent's slots need.
        let mut settings = settings("top", vec![middle.local_addr().unwrap().to_string()], None);
        settings.slots = 6;
        let top = node_with(settings, Variant::default());
        let split = top.delegating_model().unwrap();
        let model = fixture::model();
        let sequences: Vec<Vec<u32>> = (0..6u32)
            .map(|i| (0..8u32).map(|t| (i * 7 + t * 3) % 60 + 1).collect())
            .collect();
        let want: Vec<_> = sequences
            .iter()
            .map(|s| run(model.as_ref(), s, 4))
            .collect();
        let got: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = sequences
                .iter()
                .map(|s| {
                    let split = split.clone();
                    scope.spawn(move || run(split.as_ref(), s, 4))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(got, want);
        for n in [&top, &middle, &leaf] {
            n.stop();
        }
    }

    /// A worker with another quantization is left out, and the top serves
    /// without it.
    #[test]
    fn a_worker_with_another_model_is_left_out() {
        let other = node(
            "other",
            &[],
            Variant {
                f16_ffn_down: true,
                ..Variant::default()
            },
        );
        let top = node("top", &[&other], Variant::default());
        assert!(!other.is_assigned());
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        other.stop();
        top.stop();
    }

    #[test]
    fn a_worker_with_another_secret_is_left_out() {
        let worker = node_with(settings("worker", vec![], Some("one")), Variant::default());
        let top = node_with(
            settings(
                "top",
                vec![worker.local_addr().unwrap().to_string()],
                Some("two"),
            ),
            Variant::default(),
        );
        assert!(!worker.is_assigned());
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        let with_secret = node_with(
            settings(
                "top2",
                vec![worker.local_addr().unwrap().to_string()],
                Some("one"),
            ),
            Variant::default(),
        );
        assert!(worker.is_assigned());
        assert_eq!(covered(&with_secret.plan()), N_LAYER);
        for n in [&worker, &top, &with_secret] {
            n.stop();
        }
    }

    fn tls(
        pair: Option<&(std::path::PathBuf, std::path::PathBuf)>,
        ca: Option<&std::path::Path>,
    ) -> Option<Tls> {
        Tls::from_paths(pair.map(|(c, k)| (c.as_path(), k.as_path())), ca).unwrap()
    }

    /// A tree sharing one certificate runs over TLS with nothing else set,
    /// and answers exactly as it does in the clear.
    #[test]
    fn a_tree_runs_over_tls() {
        let dir = tempfile::tempdir().unwrap();
        let shared = transport::fixture::certificate(dir.path(), "shared");
        let with_tls = |name: &str, workers: Vec<String>| {
            let mut s = settings(name, workers, Some("s"));
            s.tls = tls(Some(&shared), None);
            node_with(s, Variant::default())
        };
        let leaf = with_tls("leaf", vec![]);
        let middle = with_tls("middle", vec![leaf.local_addr().unwrap().to_string()]);
        let top = with_tls("top", vec![middle.local_addr().unwrap().to_string()]);
        assert_eq!(covered(&top.plan()), N_LAYER);
        assert!(leaf.is_assigned() && middle.is_assigned());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        assert_eq!(
            run(top.delegating_model().unwrap().as_ref(), &tokens, 5),
            whole
        );
        for n in [&top, &middle, &leaf] {
            n.stop();
        }
    }

    /// A worker behind TLS is left out by a parent that dials in the clear,
    /// or that trusts another certificate.
    #[test]
    fn a_tls_mismatch_leaves_the_worker_out() {
        let dir = tempfile::tempdir().unwrap();
        let worker_cert = transport::fixture::certificate(dir.path(), "worker");
        let other = transport::fixture::certificate(dir.path(), "other");
        let mut s = settings("worker", vec![], None);
        s.tls = tls(Some(&worker_cert), None);
        let worker = node_with(s, Variant::default());
        let addr = worker.local_addr().unwrap().to_string();

        let clear = node_with(
            settings("clear", vec![addr.clone()], None),
            Variant::default(),
        );
        assert_eq!(describe(&clear.plan()), format!("clear 0..{N_LAYER}"));
        let mut s = settings("wrong", vec![addr.clone()], None);
        s.tls = tls(None, Some(&other.0));
        let wrong = node_with(s, Variant::default());
        assert_eq!(describe(&wrong.plan()), format!("wrong 0..{N_LAYER}"));
        assert!(!worker.is_assigned());

        let mut s = settings("right", vec![addr], None);
        s.tls = tls(None, Some(&worker_cert.0));
        let right = node_with(s, Variant::default());
        assert!(worker.is_assigned());
        assert_eq!(covered(&right.plan()), N_LAYER);
        for n in [&worker, &clear, &wrong, &right] {
            n.stop();
        }
    }

    /// A second parent is turned away while the first holds the worker.
    #[test]
    fn a_worker_serves_one_parent() {
        let worker = node("worker", &[], Variant::default());
        let first = node("first", &[&worker], Variant::default());
        let second = node("second", &[&worker], Variant::default());
        assert!(first.plan().iter().any(|e| e.node == "worker"));
        assert_eq!(describe(&second.plan()), format!("second 0..{N_LAYER}"));
        for n in [&worker, &first, &second] {
            n.stop();
        }
    }

    /// Losing the leaf in the middle of a sequence costs a re-prefill, not
    /// the request: the next forward plans again without it, rebuilds the
    /// sequence, and continues with the same answers.
    #[test]
    fn a_lost_worker_is_recovered_mid_sequence() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();

        let mut cache = split.new_kv_cache(N_CTX);
        let first = split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        assert_eq!(first, whole[0]);
        leaf.stop();
        let mut worst = 0f32;
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            for (a, b) in got.iter().zip(&whole[pos - 4]) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "{worst}");
        let plan = top.plan();
        assert_eq!(covered(&plan), N_LAYER);
        assert!(
            !plan
                .iter()
                .any(|e| e.node == "leaf" && e.layer_start < e.layer_end)
        );
        drop(cache);
        top.stop();
        middle.stop();
    }

    /// A worker lost mid-sequence, with a standby configured: the standby
    /// takes its layers and the sequence is rebuilt there, with no new plan
    /// and nothing reaching the request. When the worker comes back it is
    /// taken back, and the standby let go.
    #[test]
    fn a_standby_stands_in_for_a_lost_worker() {
        let worker = node("worker", &[], Variant::default());
        let addr = worker.local_addr().unwrap();
        let standby = node("standby", &[], Variant::default());
        let mut top_settings = settings("top", vec![addr.to_string()], None);
        top_settings.standby = vec![standby.local_addr().unwrap().to_string()];
        let top = node_with(top_settings, Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        assert!(!standby.is_assigned(), "a standby waits");

        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        worker.stop();
        let mut worst = 0f32;
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            for (a, b) in got.iter().zip(&whole[pos - 4]) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "{worst}");
        assert!(standby.is_assigned());
        assert_eq!(top.metrics.takeovers.load(Ordering::Relaxed), 1);
        assert_eq!(top.metrics.recoveries.load(Ordering::Relaxed), 0);
        let status = top.status();
        assert_eq!(
            status["standby"][0]["standing_in_for"],
            addr.to_string(),
            "{status}"
        );
        assert!(top.plan().iter().any(|e| e.node == "standby"));
        assert!(top.unready().is_none(), "repaired, not lost");
        drop(cache);

        // Back: taken back once the node is quiet, and the standby let go.
        let mut again = settings("worker", vec![], None);
        again.listen = Some(addr.to_string());
        let back = node_with(again, Variant::default());
        let deadline = Instant::now() + Duration::from_secs(20);
        while !top.plan().iter().any(|e| e.node == "worker") {
            assert!(Instant::now() < deadline, "{}", describe(&top.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        while standby.is_assigned() {
            assert!(Instant::now() < deadline, "the standby was not let go");
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(top.status()["standby"][0]["standing_in_for"].is_null());
        top.stop();
        back.stop();
        standby.stop();
    }

    /// A worker that does not answer when the node plans is filled in for
    /// by a standby, which takes the share the worker would have had.
    /// Switched alone (W-84), a node lets its workers go and plans nothing
    /// while it stays so, even with them there to take back; switched back,
    /// it plans them in again.
    #[test]
    fn a_node_switched_alone_lets_its_workers_go_until_switched_back() {
        let worker = node("worker", &[], Variant::default());
        let top = node("top", &[&worker], Variant::default());
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        top.set_alone(true).unwrap();
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        assert_eq!(top.status()["alone"], true);
        let deadline = Instant::now() + Duration::from_secs(10);
        while worker.is_assigned() {
            assert!(Instant::now() < deadline, "the worker was not let go");
            std::thread::sleep(Duration::from_millis(50));
        }
        // Past `readmit` (200 ms here): still alone.
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        assert!(!worker.is_assigned());
        top.set_alone(false).unwrap();
        assert!(
            top.plan().iter().any(|e| e.node == "worker"),
            "{}",
            describe(&top.plan())
        );
        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5)
        );
        // A worker cannot switch: it serves its parent's tree.
        assert!(worker.set_alone(true).is_err());
        top.stop();
        worker.stop();
    }

    /// Shares follow the nodes' speeds when they differ (W-86), and each
    /// node's setup reaches `/v1/workers`.
    #[test]
    fn a_faster_node_takes_more_layers() {
        let with_rate = |name: &str, workers: Vec<String>, rate: f32| {
            let mut s = settings(name, workers, None);
            s.shares = Shares::Decode;
            s.capacity.setups = vec![NodeSetup {
                name: name.to_string(),
                decode_rate: rate,
                prompt_rate: rate * 10.0,
                ..NodeSetup::default()
            }];
            node_with(s, Variant::default())
        };
        let a = with_rate("a", vec![], 1.0);
        let b = with_rate("b", vec![], 1.0);
        let addrs = [&a, &b]
            .map(|n| n.local_addr().unwrap().to_string())
            .to_vec();
        let top = with_rate("top", addrs, 2.0);
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert_eq!(describe(&top.plan()), "top 0..2, a 2..3, b 3..4");
        let status = top.status();
        assert_eq!(status["shares"], "decode speed");
        assert_eq!(status["setup"]["decode_gb_per_s"], 2.0);
        assert_eq!(status["workers"][1]["setups"][0]["name"], "b");
        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5)
        );
        // One that does not say how fast it is measures itself.
        let mut s = settings("measured", vec![], None);
        s.shares = Shares::Decode;
        let measured = node_with(s, Variant::default());
        assert!(measured.settings.capacity.setups[0].decode_rate > 0.0);
        for n in [&top, &a, &b, &measured] {
            n.stop();
        }
    }

    #[test]
    fn a_standby_fills_in_at_plan_time() {
        let worker = node("worker", &[], Variant::default());
        let addr = worker.local_addr().unwrap().to_string();
        worker.stop();
        let standby = node("standby", &[], Variant::default());
        let mut top_settings = settings("top", vec![addr.clone()], None);
        top_settings.standby = vec![standby.local_addr().unwrap().to_string()];
        let top = node_with(top_settings, Variant::default());
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert!(
            top.plan().iter().any(|e| e.node == "standby"),
            "{}",
            describe(&top.plan())
        );
        assert_eq!(top.status()["standby"][0]["standing_in_for"], addr);
        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5)
        );
        top.stop();
        standby.stop();
    }

    /// Every node of a tree copies the prompt weights of its own layers only,
    /// by the top-level node's decision (B-6, W-85).
    #[test]
    fn each_node_copies_its_own_layers_by_the_top_s_decision() {
        let a = node("a", &[], Variant::default());
        let b = node("b", &[], Variant::default());
        let top = node("top", &[&a, &b], Variant::default());
        let _model = top.delegating_model().unwrap();
        let decided = *top.decision.get().unwrap();
        let plan = top.plan();
        assert_eq!(plan.len(), 3, "{}", describe(&plan));
        for (name, n) in [("top", &top), ("a", &a), ("b", &b)] {
            let entry = plan.iter().find(|e| e.node == name).unwrap();
            let range = entry.layer_start as usize..entry.layer_end as usize;
            assert_eq!(
                *n.copied.lock().unwrap(),
                Some((range, decided)),
                "{name} in {}",
                describe(&plan)
            );
        }
        top.stop();
        a.stop();
        b.stop();
    }

    /// A worker that goes away between requests is taken back once it
    /// returns — at the next plan, never in the middle of a sequence.
    #[test]
    fn a_returning_worker_is_taken_back() {
        let worker = node("worker", &[], Variant::default());
        let addr = worker.local_addr().unwrap();
        let top = node("top", &[&worker], Variant::default());
        let _model = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        worker.stop();
        // Gone: the next ping notices, and the plan falls back to the top.
        let deadline = Instant::now() + Duration::from_secs(10);
        while !describe(&top.plan()).eq(&format!("top 0..{N_LAYER}")) {
            assert!(Instant::now() < deadline, "{}", describe(&top.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        let mut again = settings("worker", vec![], None);
        again.listen = Some(addr.to_string());
        let back = node_with(again, Variant::default());
        while !top.plan().iter().any(|e| e.node == "worker") {
            assert!(Instant::now() < deadline, "{}", describe(&top.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(back.is_assigned());
        top.stop();
        back.stop();
    }

    /// A worker lost below the top — the leaf under a middle node — is
    /// taken back by the node above it, once that node holds no session,
    /// without the top having to plan again.
    #[test]
    fn a_returning_worker_is_taken_back_below_the_top() {
        let leaf = node("leaf", &[], Variant::default());
        let addr = leaf.local_addr().unwrap();
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let split = top.delegating_model().unwrap();
        leaf.stop();
        // The next sequence recovers without the leaf.
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let got = run(split.as_ref(), &tokens, 5);
        assert_eq!(got.len(), whole.len());
        assert!(!middle.plan().iter().any(|e| e.node == "leaf"));

        let mut again = settings("leaf", vec![], None);
        again.listen = Some(addr.to_string());
        let back = node_with(again, Variant::default());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !middle.plan().iter().any(|e| e.node == "leaf") {
            assert!(Instant::now() < deadline, "{}", describe(&middle.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(back.is_assigned());
        assert_eq!(run(split.as_ref(), &tokens, 5), whole);
        for n in [&top, &middle, &back] {
            n.stop();
        }
    }

    /// A batched decode step that loses a worker declines, with every row
    /// rolled back, and the rows then recover one by one.
    #[test]
    fn a_batched_step_that_loses_a_worker_falls_back() {
        use crate::engine::arch::DecodeRow;
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let split = top.delegating_model().unwrap();
        let prompts: [&[u32]; 2] = [&[1, 2, 3], &[9, 8]];
        let mut caches: Vec<_> = prompts
            .iter()
            .map(|p| {
                let mut cache = split.new_kv_cache(N_CTX);
                split.forward(&mut cache, p, 0, 0).unwrap();
                cache
            })
            .collect();
        leaf.stop();
        let mut rows: Vec<DecodeRow<'_>> = caches
            .iter_mut()
            .zip(&prompts)
            .map(|(cache, p)| DecodeRow {
                cache,
                pos: p.len(),
                slot: 0,
            })
            .collect();
        assert!(
            split
                .forward_decode_batch(&mut rows, &[5, 6])
                .unwrap()
                .is_none()
        );
        drop(rows);
        let model = fixture::model();
        for ((cache, prompt), token) in caches.iter_mut().zip(&prompts).zip([5u32, 6]) {
            assert_eq!(cache.committed_len(), prompt.len());
            let got = split.forward(cache, &[token], prompt.len(), 0).unwrap();
            let mut alone = model.new_kv_cache(N_CTX);
            model.forward(&mut alone, prompt, 0, 0).unwrap();
            let want = model
                .forward(&mut alone, &[token], prompt.len(), 0)
                .unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-3, "{worst}");
        }
        drop(caches);
        top.stop();
        middle.stop();
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// Losing the middle node takes its whole subtree with it; the top runs
    /// every layer itself from then on, and the sequence carries on.
    #[test]
    fn a_lost_subtree_is_recovered_mid_sequence() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        assert_eq!(
            split.forward(&mut cache, &tokens[..5], 0, 0).unwrap(),
            whole[0]
        );
        middle.stop();
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            assert!(worst(&got, &whole[pos - 4]) < 1e-3);
        }
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        // The leaf, whose parent went away, serves its own API again.
        let deadline = Instant::now() + Duration::from_secs(10);
        while leaf.is_assigned() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        }
        drop(cache);
        top.stop();
        leaf.stop();
    }

    /// A worker lost between two chunks of a prompt: the second chunk
    /// rebuilds the first on the new plan, and the prompt's answer is the
    /// same.
    #[test]
    fn a_worker_lost_mid_prefill_is_recovered() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let model = fixture::model();
        let mut alone = model.new_kv_cache(N_CTX);
        model
            .forward_no_logits(&mut alone, &tokens[..3], 0, 0)
            .unwrap();
        let want = model.forward(&mut alone, &tokens[3..6], 3, 0).unwrap();

        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split
            .forward_no_logits(&mut cache, &tokens[..3], 0, 0)
            .unwrap();
        leaf.stop();
        let got = split.forward(&mut cache, &tokens[3..6], 3, 0).unwrap();
        assert!(worst(&got, &want) < 1e-3);
        assert_eq!(cache.committed_len(), 6);
        drop(cache);
        top.stop();
        middle.stop();
    }

    /// W-81: a two-level tree of a real model over TCP — top, middle, leaf,
    /// each node with its own copy of the model (any architecture that
    /// splits; the prompt is Llama token ids, nonsense to another
    /// vocabulary, where `f16` and `q8_0` then drift apart on a chaotic
    /// continuation; `ORANGU_TEST_N_CTX` sets the context a worker is
    /// assigned) —
    /// against the model alone, greedy. With `f32` on the wire every token
    /// and every logit is the same; `f16` and `q8_0` keep the tokens and
    /// stay within a small distance of the logits. Run with
    /// `ORANGU_TEST_LLAMA_MODEL=/path/to/model.gguf cargo test --release
    /// --bin orangu-server a_real_tree_is_equivalent -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn a_real_tree_is_equivalent() {
        let path = std::path::PathBuf::from(
            std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL"),
        );
        // Each node its own model and mapping, as separate processes have:
        // any architecture that splits.
        let build = |settings: NodeSettings| {
            let loaded = LoadedModel::open(&path).unwrap();
            let backend: Arc<dyn crate::engine::backend::Backend> =
                Arc::new(crate::engine::backend::CpuBackend);
            let model = crate::build_model(&loaded, &backend).unwrap();
            (Node::start(settings, model.clone(), loaded).unwrap(), model)
        };
        let greedy = |m: &dyn ModelForward| {
            let mut cache = m.new_kv_cache(N_CTX);
            let prompt = [1u32, 785, 6722, 315, 9625, 374, 12095, 13];
            let mut logits = m.forward(&mut cache, &prompt, 0, 0).unwrap();
            let (mut tokens, mut steps) = (Vec::new(), vec![logits.clone()]);
            for pos in prompt.len()..prompt.len() + 24 {
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32;
                tokens.push(next);
                logits = m.forward(&mut cache, &[next], pos, 0).unwrap();
                steps.push(logits.clone());
            }
            (tokens, steps)
        };
        let mut want = None;
        for (format, tolerance) in [
            (ActivationFormat::F32, 0.0),
            (ActivationFormat::F16, 0.5),
            (ActivationFormat::Q8_0, 1.5),
        ] {
            let with = |name: &str, workers: Vec<String>| {
                let mut s = settings(name, workers, Some("s"));
                s.activations = format;
                if let Some(n_ctx) = std::env::var("ORANGU_TEST_N_CTX")
                    .ok()
                    .and_then(|v| v.parse().ok())
                {
                    s.n_ctx = n_ctx;
                }
                s.label = "model".to_string();
                s.quant = "any".to_string();
                s
            };
            let (leaf, _) = build(with("leaf", vec![]));
            let (middle, _) = build(with("middle", vec![leaf.local_addr().unwrap().to_string()]));
            let (top, model) = build(with("top", vec![middle.local_addr().unwrap().to_string()]));
            let (want_tokens, want_steps) =
                want.get_or_insert_with(|| greedy(model.as_ref())).clone();
            let (tokens, steps) = greedy(top.delegating_model().unwrap().as_ref());
            let worst = steps
                .iter()
                .zip(&want_steps)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0f32, f32::max);
            println!(
                "{format:?}: plan {}; worst logit difference {worst}",
                describe(&top.plan())
            );
            assert_eq!(tokens, want_tokens, "{format:?}");
            assert!(worst <= tolerance, "{format:?}: {worst} > {tolerance}");
            for n in [&top, &middle, &leaf] {
                n.stop();
            }
        }
    }

    /// Two nodes that list each other: whichever plans second finds itself
    /// below itself and refuses, so neither ends up its own ancestor.
    #[test]
    fn nodes_that_list_each_other_do_not_loop() {
        let a = node_with(settings("a", vec![], None), Variant::default());
        let mut b_settings = settings("b", vec![a.local_addr().unwrap().to_string()], None);
        b_settings.listen = Some("127.0.0.1:0".to_string());
        let b = node_with(b_settings, Variant::default());
        assert!(a.is_assigned());
        // `a` now tries to take `b`, which holds `a` below it.
        let link = ChildLink::new(
            b.local_addr().unwrap().to_string(),
            Duration::from_secs(5),
            Duration::from_secs(2),
        );
        let err = link.connect(None, &[a.id().to_string()]).unwrap_err();
        assert!(err.message.contains("loop"), "{err}");
        a.stop();
        b.stop();
    }
}
