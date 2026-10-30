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

//! Managed Agents' sessions: `/v1/sessions` (create, list, retrieve,
//! update, delete, archive) and their events — `GET` and `POST
//! /v1/sessions/{id}/events`, and the live `…/events/stream`.
//!
//! A session lives in `<api_data>/sessions/<id>/`: `session.json`,
//! `events.jsonl` (every event, in order) and `conversation.json` (the
//! Messages API conversation the agent loop runs on). Its tools run in a
//! code execution container of its own, made when the session is.
//!
//! The agent loop sends the conversation through the path a non-streamed
//! `/v1/messages` request takes, with the agent's tools declared as client
//! tools, and runs each call the model makes itself: the built-in tools
//! in the container, the web tools and MCP tools as the Messages API's
//! server tools run them. A call its permission policy asks about, and a
//! call to a custom tool, leave the session idle until the client answers.

use axum::{
    Json,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, sse::Event},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, VecDeque};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify, broadcast};

use super::advisor;
use super::agents::{
    ListQuery, Reply, agent_at, bad, body, cursor_page, environment, fail, host_allowed, metadata,
    missing, override_agent, respond,
};
use super::containers::Container;
use super::files::{FileStore, new_id, now_nanos, valid_id};
use super::memory;
use super::server_tools::{ServerTools, conversation_text};
use super::toolset::{self, Policy, Runner, Tool};
use super::{MessagesRequest, Piece, error_of, rfc3339, run_one, run_one_watched, unix_now};
use crate::engine::tool_calls::ParsedToolCall;
use crate::http::AppState;

mod workflows;

/// The most model requests one run of the agent loop makes before it
/// stops and waits for the client.
const MAX_STEPS: usize = 50;

/// How long a request the queue turned away waits before it is tried again.
const OVERLOADED_RETRY: Duration = Duration::from_secs(2);

/// How many times a request the queue turned away is tried.
const OVERLOADED_TRIES: usize = 60;

/// What the client sends events in.
const ACCEPTED_EVENTS: &str = "user.message, user.interrupt, user.custom_tool_result, user.tool_confirmation and \
     system.message";

/// A session while the server holds it: the stream its events go out on,
/// and its agent loop's state.
struct Live {
    events: broadcast::Sender<Value>,
    /// Held across every change to the conversation, and by the loop when
    /// it decides to stop, so a message cannot slip in unseen between.
    lock: Mutex<()>,
    running: AtomicBool,
    interrupted: AtomicBool,
    interrupt: Notify,
    /// Whether the session's stream last said it is idle — what a
    /// `session.status_running` follows.
    idle_reported: AtomicBool,
}

static LIVE: LazyLock<StdMutex<HashMap<String, Arc<Live>>>> = LazyLock::new(Default::default);

fn live(id: &str) -> Arc<Live> {
    let mut all = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    all.entry(id.to_string())
        .or_insert_with(|| {
            Arc::new(Live {
                events: broadcast::channel(1024).0,
                lock: Mutex::new(()),
                running: AtomicBool::new(false),
                interrupted: AtomicBool::new(false),
                interrupt: Notify::new(),
                idle_reported: AtomicBool::new(true),
            })
        })
        .clone()
}

impl Live {
    /// Resolves once the client interrupts.
    async fn interrupted(&self) {
        let notified = self.interrupt.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if self.interrupted.load(Ordering::SeqCst) {
            return;
        }
        notified.await;
    }
}

/// Where a tool call stands while the session waits on the client.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Call {
    /// The model's id for the call, which its `tool_result` names.
    tool_use_id: String,
    /// The `agent.tool_use`, `agent.mcp_tool_use` or
    /// `agent.custom_tool_use` event the client was shown.
    event_id: String,
    name: String,
    input: Value,
    kind: Kind,
    /// Whether the client must answer before it runs (or, for a custom
    /// tool, before its result is known).
    waiting: bool,
    /// The client's answer to a confirmation: allow, or the deny message.
    decision: Option<Result<(), Option<String>>>,
    /// The `tool_result` block, once there is one.
    result: Option<Value>,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Builtin,
    Mcp,
    Custom,
}

/// The conversation the agent loop runs on.
#[derive(Serialize, Deserialize, Default)]
struct Conversation {
    /// Messages API messages.
    messages: Vec<Value>,
    /// What `system.message` events added to the system prompt.
    #[serde(default)]
    system: Vec<String>,
    /// User content not yet sent to the model.
    #[serde(default)]
    inbox: Vec<Value>,
    /// The calls of the last assistant turn, while any waits on the client.
    #[serde(default)]
    pending: Vec<Call>,
    /// The outcome being worked toward, from `user.define_outcome`.
    #[serde(default)]
    outcome: Option<Outcome>,
}

/// An outcome: the task, the rubric its work is graded against, and how
/// many gradings it has had.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Outcome {
    outcome_id: String,
    description: String,
    rubric: String,
    max_iterations: usize,
    iteration: usize,
}

impl Conversation {
    fn waiting(&self) -> Vec<&Call> {
        self.pending
            .iter()
            .filter(|c| c.waiting && c.result.is_none() && c.decision.is_none())
            .collect()
    }

    /// Whether the loop has something to do: new user content, or every
    /// waiting call answered.
    fn has_work(&self) -> bool {
        !self.inbox.is_empty() || (!self.pending.is_empty() && self.waiting().is_empty())
    }
}

fn dir(state: &AppState, id: &str) -> PathBuf {
    state.api_data.join("sessions").join(id)
}

/// A thread's key splits into its session's id and, for a child thread,
/// its own: the primary thread's key is the session id, a child's
/// `sesn_…/threads/sthr_…` — which is also where under the session it is
/// stored.
fn split_key(key: &str) -> (&str, Option<&str>) {
    match key.split_once("/threads/") {
        Some((session, thread)) => (session, Some(thread)),
        None => (key, None),
    }
}

fn thread_key(session: &str, thread: &str) -> String {
    format!("{session}/threads/{thread}")
}

/// The id of a session's primary thread.
fn primary_thread(session: &str) -> String {
    format!("sthr_{}", session.trim_start_matches("sesn_"))
}

fn load(state: &AppState, id: &str) -> Option<Value> {
    let (session, thread) = split_key(id);
    if !valid_id(session, "sesn_") || thread.is_some_and(|t| !valid_id(t, "sthr_")) {
        return None;
    }
    let text = std::fs::read_to_string(dir(state, id).join("session.json")).ok()?;
    serde_json::from_str(&text).ok()
}

fn lookup(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    load(state, id).ok_or_else(|| missing("session", id))
}

fn write_atomic(path: PathBuf, bytes: Vec<u8>) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)
}

fn save(state: &AppState, session: &Value) {
    let id = session["_key"]
        .as_str()
        .or(session["id"].as_str())
        .unwrap_or_default();
    let root = dir(state, id);
    if root.is_dir() {
        let _ = write_atomic(
            root.join("session.json"),
            serde_json::to_vec(session).unwrap_or_default(),
        );
    }
}

fn load_conversation(state: &AppState, id: &str) -> Conversation {
    std::fs::read_to_string(dir(state, id).join("conversation.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save_conversation(state: &AppState, id: &str, conversation: &Conversation) {
    let root = dir(state, id);
    if root.is_dir() {
        let _ = write_atomic(
            root.join("conversation.json"),
            serde_json::to_vec(conversation).unwrap_or_default(),
        );
    }
}

/// Changes a session's record under the lock its loop and handlers share.
fn update(state: &AppState, id: &str, change: impl FnOnce(&mut Value)) {
    static LOCK: StdMutex<()> = StdMutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(mut session) = load(state, id) {
        change(&mut session);
        save(state, &session);
    }
}

/// Records an event and sends it to every open stream: its id and
/// `processed_at` given.
fn emit(state: &AppState, id: &str, mut event: Value) -> Value {
    if event.get("id").is_none_or(Value::is_null) {
        event["id"] = json!(new_id("sevt_"));
    }
    event["processed_at"] = json!(rfc3339(unix_now()));
    let path = dir(state, id).join("events.jsonl");
    if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
        let mut line = serde_json::to_vec(&event).unwrap_or_default();
        line.push(b'\n');
        let _ = file.write_all(&line);
    }
    let _ = live(id).events.send(event.clone());
    if split_key(id).1.is_none()
        && let Some(kind) = webhook_type(event["type"].as_str().unwrap_or_default())
    {
        super::webhooks::emit(state, kind, id);
    }
    event
}

/// The webhook event a session's own stream event is announced as.
fn webhook_type(event: &str) -> Option<&'static str> {
    Some(match event {
        "session.status_running" => "session.status_run_started",
        "session.status_idle" => "session.status_idled",
        "session.status_terminated" => "session.status_terminated",
        "session.updated" => "session.updated",
        "session.thread_created" => "session.thread_created",
        // On the session's stream these are its child threads'.
        "session.thread_status_idle" => "session.thread_idled",
        "session.thread_status_terminated" => "session.thread_terminated",
        "span.outcome_evaluation_end" => "session.outcome_evaluation_ended",
        _ => return None,
    })
}

fn events(state: &AppState, id: &str) -> Vec<Value> {
    std::fs::read_to_string(dir(state, id).join("events.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// A session's record as the API shows it.
fn session_json(session: &Value) -> Value {
    let mut out = session.clone();
    if let Some(map) = out.as_object_mut() {
        map.retain(|key, _| !key.starts_with('_'));
    }
    let id = session["id"].as_str().unwrap_or_default();
    let children = format!("{id}/threads/");
    let running = live(id).running.load(Ordering::SeqCst)
        || LIVE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|(key, l)| key.starts_with(&children) && l.running.load(Ordering::SeqCst))
        || workflows::any_open(id);
    if running {
        out["status"] = json!("running");
    } else if out["status"] == "running" {
        // A loop the server stopped with it.
        out["status"] = json!("idle");
    }
    let created = session["_created"].as_u64().unwrap_or_else(unix_now);
    let active = session["usage"]["active_seconds"].as_f64().unwrap_or(0.0);
    out["stats"] = json!({
        "duration_seconds": unix_now().saturating_sub(created) as f64,
        "active_seconds": active,
    });
    out
}

/// The agent a session runs, as the session shows it.
fn session_agent(agent: &Value) -> Value {
    let mut out = agent.clone();
    if let Some(map) = out.as_object_mut() {
        for key in ["created_at", "updated_at", "archived_at", "metadata"] {
            map.remove(key);
        }
    }
    out
}

/// The agent `agent` names — an id, `{type: "agent", id, version}`, or
/// `{type: "agent_with_overrides", …}` — as this session runs it.
fn resolve_agent(state: &AppState, given: &Value) -> Result<Value, Box<Response>> {
    let (id, version, overrides) = match given {
        Value::String(id) => (id.as_str(), None, None),
        Value::Object(o) => {
            let id = o
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("`agent.id` is missing"))?;
            let version = o.get("version").and_then(Value::as_u64);
            match o.get("type").and_then(Value::as_str) {
                Some("agent") | None => (id, version, None),
                Some("agent_with_overrides") => (id, version, Some(o)),
                Some(other) => {
                    return Err(bad(format!("'{other}' is not an agent reference type")));
                }
            }
        }
        _ => return Err(bad("`agent` is an agent id or an agent reference")),
    };
    let agent = agent_at(state, id, version).ok_or_else(|| match version {
        Some(v) => missing("agent version", &format!("{id} v{v}")),
        None => missing("agent", id),
    })?;
    if !agent["archived_at"].is_null() {
        return Err(bad(format!(
            "agent '{id}' is archived, and a new session cannot use it"
        )));
    }
    let agent = match overrides {
        Some(o) => override_agent(state, &agent, o).map_err(bad)?,
        None => agent,
    };
    Ok(session_agent(&agent))
}

/// The agents a coordinator may delegate to, each with the configuration
/// its threads run — `self` the coordinator's own, delegating no further —
/// and whether it may define agents of its own. Null for an agent that
/// does not delegate.
fn resolve_roster(state: &AppState, agent: &Value) -> Result<Value, Box<Response>> {
    let block = &agent["multiagent"];
    let (entries, inline) = match block["type"].as_str() {
        Some("coordinator") => (block["agents"].clone(), false),
        Some("multiagent_20261001") if block["subagents"]["type"] == "enabled" => (
            block["subagents"]["predefined_agents"].clone(),
            block["subagents"]["inline_agents"]["type"] == "enabled",
        ),
        _ => return Ok(Value::Null),
    };
    let agents = resolve_entries(state, agent, &entries)?;
    // A coordinator whose only entry is its advisor delegates to no one.
    if agents.is_empty() && !inline {
        return Ok(Value::Null);
    }
    Ok(json!({"agents": agents, "inline": inline}))
}

/// The agents a workflow run's plan may run, and whether it may define
/// its own: null when the agent has no dynamic workflows.
fn resolve_workflows(state: &AppState, agent: &Value) -> Result<Value, Box<Response>> {
    let block = &agent["multiagent"];
    if block["type"] != "multiagent_20261001" || block["workflows"]["type"] != "enabled" {
        return Ok(Value::Null);
    }
    let workflows = &block["workflows"];
    Ok(json!({
        "agents": resolve_entries(state, agent, &workflows["predefined_agents"])?,
        "inline": workflows["inline_agents"]["type"] == "enabled",
    }))
}

/// A list of agents' entries, each with the configuration its threads run
/// — `self` the session agent's own, delegating no further.
fn resolve_entries(
    state: &AppState,
    agent: &Value,
    entries: &Value,
) -> Result<Vec<Value>, Box<Response>> {
    let mut agents = Vec::new();
    for entry in entries.as_array().into_iter().flatten() {
        if entry["type"] == "advisor" {
            continue;
        }
        let mut config = if entry["type"] == "self" {
            agent.clone()
        } else {
            let id = entry["id"].as_str().unwrap_or_default();
            let found = agent_at(state, id, entry["version"].as_u64())
                .ok_or_else(|| missing("agent", id))?;
            session_agent(&found)
        };
        config["multiagent"] = Value::Null;
        agents.push(
            json!({"name": config["name"], "description": config["description"],
                           "agent": config}),
        );
    }
    Ok(agents)
}

/// The agent toolset's enabled tools.
fn builtin_tools(agent: &Value) -> Vec<Tool> {
    agent["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|t| t["type"] == toolset::TOOLSET)
        .map(toolset::enabled)
        .unwrap_or_default()
}

fn custom_tools(agent: &Value) -> Vec<Value> {
    agent["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["type"] == "custom")
        .map(|t| {
            json!({
                "name": t["name"], "description": t["description"],
                "input_schema": t["input_schema"],
            })
        })
        .collect()
}

fn mcp_toolset<'a>(agent: &'a Value, server: &str) -> Option<&'a Value> {
    agent["tools"]
        .as_array()?
        .iter()
        .find(|t| t["type"] == "mcp_toolset" && t["mcp_server_name"] == server)
}

/// Whether an agent can run where it is asked to: the sandbox its
/// container tools need, and the hosts a `limited` environment lets out.
fn check_runnable(state: &AppState, agent: &Value, environment: &Value) -> Result<(), String> {
    let tools = builtin_tools(agent);
    if state.code_execution == "off" && tools.iter().any(|t| matches!(t.name, "bash" | "grep")) {
        return Err("this agent's bash and grep tools run in a sandbox: set \
                    [orangu-server].code_execution = bwrap, or turn those tools off"
            .into());
    }
    let networking = &environment["config"]["networking"];
    if networking["type"] == "limited" && networking["allow_mcp_servers"] != true {
        let allowed = networking["allowed_hosts"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let blocked: Vec<String> = agent["mcp_servers"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s["url"].as_str())
            .filter_map(|url| reqwest::Url::parse(url).ok())
            .filter_map(|url| url.host_str().map(str::to_string))
            .filter(|host| !host_allowed(&allowed, host))
            .collect();
        if !blocked.is_empty() {
            return Err(format!(
                "the environment's limited networking blocks the agent's MCP servers ({}): set \
                 allow_mcp_servers or list them in allowed_hosts",
                blocked.join(", ")
            ));
        }
    }
    Ok(())
}

/// Copies a directory tree.
fn copy_tree(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// A `file` resource's place under `/mnt/session/uploads`.
fn upload_path(file_id: &str, mount_path: Option<&str>) -> Result<String, String> {
    let rel = mount_path.unwrap_or(file_id).trim_start_matches('/');
    let clean = std::path::Path::new(rel)
        .components()
        .all(|c| matches!(c, std::path::Component::Normal(_)));
    if rel.is_empty() || !clean {
        return Err(format!("the mount_path '{rel}' is not a plain path"));
    }
    Ok(rel.to_string())
}

/// Where the session's uploads and outputs are in its container's
/// directory: `/mnt/session/uploads` and `/mnt/session/outputs`.
fn session_dir(container_root: &std::path::Path) -> PathBuf {
    container_root.join("session")
}

/// Where an attached memory store's directory is in the container's
/// directory, by its mount's name; and what [`memory::sync_back`] last saw
/// there.
fn memory_dir(container_root: &std::path::Path, slug: &str) -> PathBuf {
    container_root.join("memory").join(slug)
}

fn memory_baseline(container_root: &std::path::Path, store: &str) -> PathBuf {
    container_root
        .join("memory-seen")
        .join(format!("{store}.json"))
}

/// The session's container: the agent's skills under `/workspace/skills`,
/// its `file` resources under `/mnt/session/uploads`, an empty
/// `/mnt/session/outputs`, and each `memory_store` resource's memories
/// under `/mnt/memory/<name>`.
async fn make_container(
    state: &AppState,
    agent: &Value,
    resources: &[Value],
) -> Result<(String, Vec<Value>), Box<Response>> {
    let skills = super::skills::resolve(
        &state.api_data,
        agent["skills"].as_array().map(Vec::as_slice).unwrap_or(&[]),
    )
    .map_err(bad)?;
    let store = FileStore::at(&state.api_data);
    let mut uploads = Vec::new();
    let mut stores: Vec<(Value, String, &Value)> = Vec::new();
    for resource in resources {
        match resource["type"].as_str() {
            Some("file") => {
                let file_id = resource["file_id"]
                    .as_str()
                    .ok_or_else(|| bad("a file resource needs a `file_id`"))?;
                if store.get(file_id).is_none() {
                    return Err(missing("file", file_id));
                }
                let rel = upload_path(file_id, resource["mount_path"].as_str()).map_err(bad)?;
                uploads.push((file_id.to_string(), rel));
            }
            Some("memory_store") => {
                let id = resource["memory_store_id"]
                    .as_str()
                    .ok_or_else(|| bad("a memory_store resource needs a `memory_store_id`"))?;
                let found = memory::attachable(state, id)?;
                if !matches!(
                    resource["access"].as_str(),
                    None | Some("read_write" | "read_only")
                ) {
                    return Err(bad("a memory store's `access` is read_write or read_only"));
                }
                if resource["instructions"]
                    .as_str()
                    .is_some_and(|i| i.chars().count() > 4096)
                {
                    return Err(bad(
                        "a memory store's `instructions` are at most 4096 characters",
                    ));
                }
                let base = memory::slug(found["name"].as_str().unwrap_or_default());
                let mut slug = base.clone();
                let mut n = 2;
                while stores.iter().any(|(_, s, _)| *s == slug) {
                    slug = format!("{base}-{n}");
                    n += 1;
                }
                stores.push((found, slug, resource));
            }
            Some(other) => {
                return Err(bad(format!(
                    "'{other}' resources are not supported here: attach `file` and \
                     `memory_store` resources"
                )));
            }
            None => return Err(bad("a `resources` entry needs a `type`")),
        }
    }
    if stores.len() > memory::MAX_SESSION_STORES {
        return Err(bad(format!(
            "a session attaches at most {} memory stores",
            memory::MAX_SESSION_STORES
        )));
    }
    let container = Container::open(&state.api_data, None)
        .await
        .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let workspace = container.workspace();
    let root = container.root().to_path_buf();
    let now = rfc3339(unix_now());
    let mut shown = Vec::new();
    let mut copy = || -> std::io::Result<()> {
        for skill in &skills {
            copy_tree(&skill.dir, &workspace.join("skills").join(&skill.name))?;
        }
        let session = session_dir(&root);
        std::fs::create_dir_all(session.join("uploads"))?;
        std::fs::create_dir_all(session.join("outputs"))?;
        for (file_id, rel) in &uploads {
            let target = session.join("uploads").join(rel);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(store.data_path(file_id), &target)?;
            shown.push(json!({
                "id": new_id("sesrsc_"), "type": "file", "file_id": file_id,
                "mount_path": format!("/mnt/session/uploads/{rel}"),
                "created_at": now, "updated_at": now,
            }));
        }
        for (found, slug, resource) in &stores {
            let id = found["id"].as_str().unwrap_or_default();
            let seen = memory::export(state, id, &memory_dir(&root, slug))?;
            let baseline = memory_baseline(&root, id);
            if let Some(parent) = baseline.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(baseline, serde_json::to_vec(&seen).unwrap_or_default())?;
            shown.push(json!({
                "id": new_id("sesrsc_"), "type": "memory_store", "memory_store_id": id,
                "name": found["name"], "description": found["description"],
                "access": resource["access"].as_str().unwrap_or("read_write"),
                "instructions": resource["instructions"],
                "mount_path": format!("/mnt/memory/{slug}"),
                "created_at": now, "updated_at": now,
            }));
        }
        Ok(())
    };
    copy().map_err(|err| {
        fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("setting up the session's container: {err}"),
        )
    })?;
    Ok((container.id.clone(), shown))
}

/// Shows the session's uploads, outputs and memory stores to its
/// container's commands and file tools, where its resources say they are.
fn mount_session(container: &mut Container, session: &Value) {
    let root = container.root().to_path_buf();
    let dir = session_dir(&root);
    if dir.is_dir() {
        container.mount(dir.join("uploads"), "/mnt/session/uploads", false);
        container.mount(dir.join("outputs"), "/mnt/session/outputs", true);
    }
    for resource in memory_resources(session) {
        let Some(at) = resource["mount_path"].as_str() else {
            continue;
        };
        let slug = at.trim_start_matches("/mnt/memory/");
        container.mount(
            memory_dir(&root, slug),
            at,
            resource["access"] != "read_only",
        );
    }
}

fn memory_resources(session: &Value) -> impl Iterator<Item = &Value> {
    session["resources"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| r["type"] == "memory_store")
}

/// Records in each writable memory store what the agent changed in it.
fn sync_memory(state: &AppState, session: &Value, container_root: &std::path::Path) {
    let id = session["id"].as_str().unwrap_or_default();
    for resource in memory_resources(session).filter(|r| r["access"] != "read_only") {
        let (Some(store), Some(at)) = (
            resource["memory_store_id"].as_str(),
            resource["mount_path"].as_str(),
        ) else {
            continue;
        };
        let path = memory_baseline(container_root, store);
        let mut seen: HashMap<String, String> = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let dir = memory_dir(container_root, at.trim_start_matches("/mnt/memory/"));
        memory::sync_back(state, store, &dir, &mut seen, id);
        let _ = write_atomic(path, serde_json::to_vec(&seen).unwrap_or_default());
    }
}

/// Stores in the Files API, under the session's scope, each file of
/// `/mnt/session/outputs` that is new or changed since it was last
/// stored; a changed file's earlier copy is deleted.
async fn capture_outputs(state: &AppState, session: &Value) {
    let (Some(id), Some(container)) = (session["id"].as_str(), session["_container"].as_str())
    else {
        return;
    };
    let root = state.api_data.join("containers").join(container);
    let outputs = session_dir(&root).join("outputs");
    if !outputs.is_dir() {
        return;
    }
    let record = session_dir(&root).join("captured.json");
    let mut captured: HashMap<String, (u64, u128, String)> = std::fs::read(&record)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let store = FileStore::at(&state.api_data);
    let now = super::containers::snapshot(outputs.clone()).await;
    for (rel, (size, mtime)) in now {
        let key = rel.to_string_lossy().into_owned();
        let nanos = mtime
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        if captured
            .get(&key)
            .is_some_and(|(s, m, _)| *s == size && *m == nanos)
        {
            continue;
        }
        let name = rel
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Ok(meta) = store
            .save_scoped(&name, &outputs.join(&rel), Some(id))
            .await
            && let Some((_, _, earlier)) = captured.insert(key, (size, nanos, meta.id))
        {
            store.delete(&earlier);
        }
    }
    let _ = write_atomic(record, serde_json::to_vec(&captured).unwrap_or_default());
}

/// The system prompt the agent runs under.
fn system_prompt(state: &AppState, agent: &Value, session: &Value, extra: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(system) = agent["system"].as_str().filter(|s| !s.is_empty()) {
        parts.push(system.to_string());
    }
    if !session["_thread"]["workflow_run_id"].is_null() {
        parts.push(
            "You are one of the agents of a workflow run: you work on the task you are given on \
             your own. Nothing sees your work but your final message, which is your result: \
             when you are done, end your turn with a final message that gives it in full, with \
             the paths of any files you wrote."
                .into(),
        );
    } else if session["_thread"].is_object() {
        parts.push(
            "You are working on a task another agent, your coordinator, gave you. It does not \
             see your work, only your report: when you are done, end your turn with a final \
             message that reports what you found or did, with the paths of any files you wrote."
                .into(),
        );
    } else if session["_roster"].is_object() {
        parts.push(
            "You can delegate. `list_agents` shows the agents you can give tasks to and the \
             threads already working; `send_to_agent` gives one a task, or sends a thread a \
             follow-up. Each works in parallel in its own thread, in the same container, and \
             sees none of this conversation, so every task must carry what the agent needs: \
             the question, the paths, the constraints, and what to report. Their reports reach \
             you as messages in a later turn: after delegating, end your turn instead of \
             waiting, and combine the reports when they come."
                .into(),
        );
    }
    if !builtin_tools(agent).is_empty() {
        parts.push(
            "Your tools work in a Linux container. Your working directory is /workspace, and \
             the files you write there stay for the rest of the session."
                .into(),
        );
    }
    let skills = super::skills::resolve(
        &state.api_data,
        agent["skills"].as_array().map(Vec::as_slice).unwrap_or(&[]),
    )
    .unwrap_or_default();
    if !skills.is_empty() {
        let mut text = "These skills are in the container, each a folder under \
                        /workspace/skills. A skill is not a tool and cannot be called: to use \
                        one, first read its instructions in /workspace/skills/<name>/SKILL.md, \
                        then follow them."
            .to_string();
        for skill in &skills {
            text.push_str(&format!("\n- {}: {}", skill.name, skill.description));
        }
        parts.push(text);
    }
    let uploads: Vec<&str> = session["resources"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| r["type"] == "file")
        .filter_map(|r| r["mount_path"].as_str())
        .collect();
    if !uploads.is_empty() {
        parts.push(format!(
            "These files are in the container, read-only: {}.",
            uploads.join(", ")
        ));
    }
    if !builtin_tools(agent).is_empty() {
        parts.push(
            "Files you write under /mnt/session/outputs are handed to the user when your turn \
             ends."
                .into(),
        );
    }
    let stores: Vec<String> = memory_resources(session)
        .map(|r| {
            let mut line = format!(
                "- {} ({}): {}",
                r["mount_path"].as_str().unwrap_or_default(),
                if r["access"] == "read_only" {
                    "read-only"
                } else {
                    "read-write"
                },
                r["name"].as_str().unwrap_or_default(),
            );
            for key in ["description", "instructions"] {
                if let Some(text) = r[key].as_str().filter(|t| !t.is_empty()) {
                    line.push_str(&format!(". {text}"));
                }
            }
            line
        })
        .collect();
    if !stores.is_empty() {
        parts.push(format!(
            "Memory stores are mounted in the container: directories of small text files that \
             outlive this session. Read them with your file tools; in a read-write one, what you \
             write, edit or delete is kept for later sessions.\n{}",
            stores.join("\n")
        ));
    }
    parts.extend(extra.iter().cloned());
    parts.join("\n\n")
}

/// The Messages API server tool declarations the agent's web tools and MCP
/// toolsets run as, under the environment's networking.
fn server_declarations(searches: bool, agent: &Value, environment: &Value) -> Vec<Value> {
    let networking = &environment["config"]["networking"];
    let limited = networking["type"] == "limited";
    let hosts: Vec<Value> = networking["allowed_hosts"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for tool in builtin_tools(agent) {
        let kind = match tool.name {
            "web_search" if searches => "web_search_20250305",
            "web_fetch" => "web_fetch_20250910",
            _ => continue,
        };
        let mut declaration = json!({"type": kind, "name": tool.name});
        for key in [
            "allowed_domains",
            "blocked_domains",
            "max_uses",
            "max_content_tokens",
        ] {
            if let Some(value) = tool.config.get(key).filter(|v| !v.is_null()) {
                declaration[key] = value.clone();
            }
        }
        if limited {
            if hosts.is_empty() {
                continue;
            }
            if declaration.get("allowed_domains").is_none() {
                declaration["allowed_domains"] = json!(hosts);
            }
        }
        out.push(declaration);
    }
    for server in agent["mcp_servers"].as_array().into_iter().flatten() {
        let name = server["name"].as_str().unwrap_or_default();
        let Some(toolset) = mcp_toolset(agent, name) else {
            continue;
        };
        // The Messages API's toolset names its tools' configs by key.
        let configs: Map<String, Value> = toolset["configs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| {
                Some((
                    c["name"].as_str()?.to_string(),
                    json!({"enabled": c["enabled"]}),
                ))
            })
            .collect();
        out.push(json!({
            "type": "mcp_toolset", "mcp_server_name": name,
            "default_config": {"enabled": toolset["default_config"]["enabled"]},
            "configs": configs,
        }));
    }
    out
}

/// The MCP servers the agent has a toolset for.
fn mcp_servers(agent: &Value) -> Vec<Value> {
    agent["mcp_servers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|s| mcp_toolset(agent, s["name"].as_str().unwrap_or_default()).is_some())
        .cloned()
        .collect()
}

/// A Messages API tool declaration from the OpenAI shape the server tools
/// give theirs in.
fn client_shape(function: &Value) -> Value {
    let f = &function["function"];
    json!({
        "name": f["name"],
        "description": f.get("description").cloned().unwrap_or(json!("")),
        "input_schema": f["parameters"],
    })
}

/// Appends user content to the conversation, into the last message when
/// that is the user's.
fn push_user(messages: &mut Vec<Value>, content: Vec<Value>) {
    if content.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut()
        && last["role"] == "user"
        && let Some(blocks) = last["content"].as_array_mut()
    {
        blocks.extend(content);
        return;
    }
    messages.push(json!({"role": "user", "content": content}));
}

fn tool_result(tool_use_id: &str, text: &str, is_error: bool) -> Value {
    json!({
        "type": "tool_result", "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}], "is_error": is_error,
    })
}

/// Whether a server tool's result block reports a failure.
fn failed(block: &Value) -> bool {
    block["is_error"] == true
        || block["content"]["type"]
            .as_str()
            .is_some_and(|t| t.ends_with("_error"))
}

/// What a run of the agent loop ended on.
enum Ended {
    Turn,
    Waiting(Vec<String>),
    Failed(String),
}

/// The live previews of one model request: each run of text announced
/// with an `event_start` and sent on as `event_delta`s, and thinking
/// announced, on the session's live streams only. The buffered
/// `agent.message` and `agent.thinking` events that follow take the ids
/// announced, in order.
#[derive(Default)]
struct Previews {
    messages: VecDeque<String>,
    thinking: VecDeque<String>,
    /// What the last piece was: text (and its block index), thinking, or
    /// anything else.
    in_text: Option<usize>,
    in_thinking: bool,
    /// The text before was cited: more text is the next block.
    cited: bool,
}

impl Previews {
    fn watch(&mut self, piece: &Piece, events: &broadcast::Sender<Value>) {
        match piece {
            Piece::Text(text) if !text.is_empty() => {
                self.in_thinking = false;
                let index = match self.in_text {
                    Some(index) if self.cited => index + 1,
                    Some(index) => index,
                    None => {
                        let id = new_id("sevt_");
                        let _ = events.send(json!({"type": "event_start",
                                                   "event": {"type": "agent.message", "id": id}}));
                        self.messages.push_back(id);
                        0
                    }
                };
                self.in_text = Some(index);
                self.cited = false;
                if let Some(id) = self.messages.back() {
                    let _ = events.send(json!({
                        "type": "event_delta", "event_id": id,
                        "delta": {"type": "content_delta", "index": index,
                                  "content": {"type": "text", "text": text}},
                    }));
                }
            }
            Piece::Text(_) => {}
            Piece::Citation(_) => self.cited = self.in_text.is_some(),
            Piece::Thinking(_) => {
                self.in_text = None;
                if !self.in_thinking {
                    self.in_thinking = true;
                    let id = new_id("sevt_");
                    let _ = events.send(json!({"type": "event_start",
                                               "event": {"type": "agent.thinking", "id": id}}));
                    self.thinking.push_back(id);
                }
            }
            _ => {
                self.in_text = None;
                self.in_thinking = false;
            }
        }
    }

    fn next_message(&mut self) -> Option<String> {
        self.messages.pop_front()
    }

    fn next_thinking(&mut self) -> Option<String> {
        self.thinking.pop_front()
    }
}

/// One session's agent loop while it runs.
struct Agent<'a> {
    state: &'a Arc<AppState>,
    id: &'a str,
    live: &'a Live,
    agent: Value,
    session: Value,
    builtin: Vec<Tool>,
    custom: Vec<String>,
    runner: Runner,
    server: ServerTools,
    tools: Vec<Value>,
    /// The advisor's model, on a primary thread whose agent has one.
    advisor: Option<String>,
}

/// How a consultation of the advisor ended.
enum Consulted {
    Advice(Result<advisor::Advice, &'static str>),
    /// Interrupted on the advisor's own thread: the agent carries on.
    Abandoned,
    /// Interrupted with the whole session.
    Halted,
}

impl Agent<'_> {
    fn emit(&self, event: Value) -> Value {
        emit(self.state, self.id, event)
    }

    /// Runs one call, unless the client interrupts first: its
    /// `tool_result` block, and the event that reports it.
    async fn execute(&mut self, call: &Call) -> Option<Value> {
        let input: Map<String, Value> = call.input.as_object().cloned().unwrap_or_default();
        if self.advises(&call.name) {
            let text = self.consult().await?;
            return Some(tool_result(&call.tool_use_id, &text, false));
        }
        if self.runs_workflows(&call.name) {
            let (text, is_error) =
                workflows::call(self.state, &self.session, &call.name, &input).await;
            return Some(tool_result(&call.tool_use_id, &text, is_error));
        }
        if self.delegates() && DELEGATION_TOOLS.contains(&call.name.as_str()) {
            let (text, is_error) = match self.delegate(&call.name, &input).await {
                Ok(text) => (text, false),
                Err(text) => (format!("Error: {text}"), true),
            };
            self.emit(json!({
                "type": "agent.tool_result", "tool_use_id": call.event_id,
                "content": [{"type": "text", "text": text}], "is_error": is_error,
            }));
            return Some(tool_result(&call.tool_use_id, &text, is_error));
        }
        let live = self.live;
        let contained = self.is_builtin_container(&call.name);
        // `read` of a picture, for a model that can see it: the picture
        // itself is the result.
        if contained
            && call.name == "read"
            && self.state.engine.vision.is_some()
            && let Some(read) = self.runner.picture(&input).transpose()
        {
            use base64::Engine as _;
            let (content, is_error) = match read {
                Ok((mime, bytes)) => (
                    json!([{"type": "image", "source": {
                        "type": "base64", "media_type": mime,
                        "data": base64::engine::general_purpose::STANDARD.encode(bytes),
                    }}]),
                    false,
                ),
                Err(message) => (
                    json!([{"type": "text", "text": format!("Error: {message}")}]),
                    true,
                ),
            };
            self.emit(json!({
                "type": "agent.tool_result", "tool_use_id": call.event_id,
                "content": content, "is_error": is_error,
            }));
            return Some(json!({
                "type": "tool_result", "tool_use_id": call.tool_use_id,
                "content": content, "is_error": is_error,
            }));
        }
        let ran = async {
            if contained {
                let ran = self.runner.run(&call.name, &input).await;
                if matches!(call.name.as_str(), "bash" | "write" | "edit") {
                    sync_memory(self.state, &self.session, self.runner.container.root());
                }
                return ran;
            }
            match self.server.kind(&call.name) {
                Some(kind) => {
                    let parsed = ParsedToolCall {
                        name: call.name.clone(),
                        arguments: call.input.to_string(),
                    };
                    let ran = self.server.run(&kind, &call.tool_use_id, &parsed).await;
                    (ran.text, failed(&ran.block))
                }
                None => (
                    format!("Error: there is no tool named `{}`", call.name),
                    true,
                ),
            }
        };
        let (text, is_error) = tokio::select! {
            out = ran => out,
            _ = live.interrupted() => return None,
        };
        let content = json!([{"type": "text", "text": text}]);
        match call.kind {
            Kind::Mcp => self.emit(json!({
                "type": "agent.mcp_tool_result", "mcp_tool_use_id": call.event_id,
                "content": content, "is_error": is_error,
            })),
            _ => self.emit(json!({
                "type": "agent.tool_result", "tool_use_id": call.event_id,
                "content": content, "is_error": is_error,
            })),
        };
        Some(tool_result(&call.tool_use_id, &text, is_error))
    }

    /// Whether `name` is a call that starts or stops a workflow run.
    fn runs_workflows(&self, name: &str) -> bool {
        split_key(self.id).1.is_none()
            && self.session["_workflows"].is_object()
            && workflows::TOOLS.contains(&name)
    }

    /// Whether `name` is a call of the advisor.
    fn advises(&self, name: &str) -> bool {
        self.advisor.is_some() && name == advisor::TOOL
    }

    /// Consults the advisor in a thread of its own: what the agent is told,
    /// the advice or a notice that there is none. `None` when the session
    /// is interrupted.
    async fn consult(&self) -> Option<String> {
        let model = self.advisor.clone()?;
        let config = json!({"type": "advisor", "model": model});
        let Ok(key) = create_thread(self.state, &self.session, config, advisor::NAME, None) else {
            return Some(advisor::failure_text("unavailable"));
        };
        let thread = split_key(&key).1.unwrap_or_default().to_string();
        let own = live(&key);
        own.running.store(true, Ordering::SeqCst);
        announce_running(self.state, &key);
        let conversation = load_conversation(self.state, self.id);
        let system = system_prompt(self.state, &self.agent, &self.session, &conversation.system);
        let transcript = advisor::transcript_of_messages(&system, &conversation.messages);
        let consulted = tokio::select! {
            advice = advisor::consult(self.state, &model, transcript, None) => {
                Consulted::Advice(advice)
            }
            _ = own.interrupted() => Consulted::Abandoned,
            _ = self.live.interrupted() => Consulted::Halted,
        };
        // An interrupt of the whole session reaches both.
        let consulted = match consulted {
            Consulted::Abandoned if self.live.interrupted.load(Ordering::SeqCst) => {
                Consulted::Halted
            }
            other => other,
        };
        let text = match &consulted {
            Consulted::Advice(Ok(advice)) => {
                let count = |key: &str| advice.usage[key].as_u64().unwrap_or(0);
                let add = |s: &mut Value| {
                    for key in ["input_tokens", "output_tokens", "cache_read_input_tokens"] {
                        let sum = s["usage"][key].as_u64().unwrap_or(0) + count(key);
                        s["usage"][key] = json!(sum);
                    }
                };
                update(self.state, &key, add);
                update(self.state, split_key(self.id).0, add);
                let content = json!([{"type": "text", "text": advice.text}]);
                emit(
                    self.state,
                    &key,
                    json!({"type": "agent.message", "content": content}),
                );
                self.emit(json!({
                    "type": "agent.thread_message_received", "from_session_thread_id": thread,
                    "from_agent_name": advisor::NAME, "content": content,
                }));
                advice.for_agent()
            }
            Consulted::Advice(Err(code)) => advisor::failure_text(code),
            Consulted::Abandoned | Consulted::Halted => advisor::failure_text("interrupted"),
        };
        own.running.store(false, Ordering::SeqCst);
        own.interrupted.store(false, Ordering::SeqCst);
        let now = rfc3339(unix_now());
        update(self.state, &key, |r| {
            r["archived_at"] = json!(now);
            r["updated_at"] = json!(now);
        });
        for event in [
            json!({"type": "session.thread_status_idle", "session_thread_id": thread,
                   "agent_name": advisor::NAME, "stop_reason": {"type": "end_turn"}}),
            json!({"type": "session.thread_status_terminated", "session_thread_id": thread,
                   "agent_name": advisor::NAME}),
        ] {
            emit(self.state, &key, event.clone());
            self.emit(event);
        }
        match consulted {
            Consulted::Halted => None,
            _ => Some(text),
        }
    }

    /// Whether this is a coordinator's thread: it has agents to delegate to.
    fn delegates(&self) -> bool {
        split_key(self.id).1.is_none() && self.session["_roster"].is_object()
    }

    /// Runs `list_agents` or `send_to_agent`: what the coordinator is told.
    async fn delegate(&self, name: &str, input: &Map<String, Value>) -> Result<String, String> {
        let roster = &self.session["_roster"];
        let members = roster["agents"].as_array().cloned().unwrap_or_default();
        let inline = roster["inline"] == true;
        let session = self.id;
        if name == "list_agents" {
            let agents: Vec<Value> = members
                .iter()
                .map(|m| json!({"agent_name": m["name"], "description": m["description"]}))
                .collect();
            let threads: Vec<Value> = delegated_threads(self.state, session)
                .iter()
                .filter(|t| t["archived_at"].is_null())
                .map(|t| {
                    json!({"session_thread_id": t["_thread"]["id"],
                           "agent_name": t["_thread"]["agent_name"],
                           "status": thread_status(t)})
                })
                .collect();
            return Ok(json!({"agents": agents, "threads": threads,
                             "inline_agents": if inline { "enabled" } else { "disabled" }})
            .to_string());
        }
        let message = input
            .get("message")
            .and_then(Value::as_str)
            .filter(|m| !m.trim().is_empty())
            .ok_or("`message` is required: the task or message, with everything the agent needs")?;
        if let Some(thread) = input.get("session_thread_id").and_then(Value::as_str) {
            let key = thread_key(session, thread);
            let record = load(self.state, &key)
                .filter(|r| r["archived_at"].is_null())
                .ok_or_else(|| format!("there is no open thread '{thread}'"))?;
            send_message(self.state, session, &key, message).await;
            return Ok(format!(
                "Sent to {} (session_thread_id {thread}). Its reply comes back to you as a \
                 message in a later turn.",
                record["_thread"]["agent_name"].as_str().unwrap_or_default()
            ));
        }
        let open = delegated_threads(self.state, session)
            .iter()
            .filter(|t| t["archived_at"].is_null())
            .count();
        if open >= MAX_CHILD_THREADS {
            return Err(format!(
                "this session already has {MAX_CHILD_THREADS} threads: send follow-ups to one of \
                 them with `session_thread_id`"
            ));
        }
        let config = if let Some(defined) = input.get("inline_agent").filter(|d| d.is_object()) {
            if !inline {
                return Err("defining agents is turned off: use one of list_agents' agents".into());
            }
            let mut config = inline_agent(&self.agent);
            for key in ["name", "description", "system"] {
                if let Some(value) = defined.get(key).and_then(Value::as_str) {
                    config[key] = json!(value);
                }
            }
            if defined.get("name").and_then(Value::as_str).is_none() {
                return Err("`inline_agent.name` is required".into());
            }
            config
        } else {
            let wanted = input.get("agent_name").and_then(Value::as_str).ok_or(
                "name the agent: `agent_name` from list_agents, `session_thread_id` of a \
                        thread, or define one with `inline_agent`",
            )?;
            members
                .iter()
                .find(|m| {
                    m["name"]
                        .as_str()
                        .is_some_and(|n| n.eq_ignore_ascii_case(wanted))
                        || m["agent"]["id"] == wanted
                })
                .map(|m| m["agent"].clone())
                .ok_or_else(|| format!("no agent is named '{wanted}': list_agents shows them"))?
        };
        let name = config["name"].as_str().unwrap_or("agent").to_string();
        let key = create_thread(self.state, &self.session, config, &name, None)?;
        let (_, Some(thread)) = split_key(&key) else {
            return Err("the thread could not be made".into());
        };
        let thread = thread.to_string();
        let agent_name = load(self.state, &key)
            .map(|r| r["_thread"]["agent_name"].clone())
            .unwrap_or_default();
        send_message(self.state, session, &key, message).await;
        Ok(format!(
            "Sent to {} (session_thread_id {thread}). It works on its own, in parallel, and does \
             not see this conversation; its report comes back to you as a message in a later \
             turn. End your turn when you have nothing else to do.",
            agent_name.as_str().unwrap_or_default()
        ))
    }

    fn is_builtin_container(&self, name: &str) -> bool {
        self.builtin
            .iter()
            .any(|t| t.name == name && !matches!(t.name, "web_fetch" | "web_search"))
    }

    /// The calls of a pending turn the client has answered: allowed ones
    /// run, denied ones refused. `None` when interrupted.
    async fn settle(&mut self, mut calls: Vec<Call>) -> Option<Vec<Value>> {
        let mut results = Vec::new();
        for call in calls.iter_mut() {
            if call.result.is_none() {
                call.result = Some(match &call.decision {
                    Some(Ok(())) => self.execute(call).await?,
                    Some(Err(message)) => {
                        let text = message.clone().unwrap_or_else(|| {
                            format!("Permission to use {} has been denied.", call.name)
                        });
                        let content = json!([{"type": "text", "text": text}]);
                        if call.kind == Kind::Mcp {
                            self.emit(json!({
                                "type": "agent.mcp_tool_result",
                                "mcp_tool_use_id": call.event_id,
                                "content": content, "is_error": true,
                            }));
                        } else {
                            self.emit(json!({
                                "type": "agent.tool_result", "tool_use_id": call.event_id,
                                "content": content, "is_error": true,
                            }));
                        }
                        tool_result(&call.tool_use_id, &text, true)
                    }
                    None => tool_result(&call.tool_use_id, "No result was given.", true),
                });
            }
            results.push(call.result.clone().unwrap_or_default());
        }
        Some(results)
    }

    /// One model request: the reply, and the ids its previews announced;
    /// or why there is none. `Ok(None)` when interrupted.
    async fn ask_model(
        &self,
        conversation: &Conversation,
    ) -> Result<Option<(Value, Previews)>, String> {
        let mut request = json!({
            "model": self.agent["model"]["id"],
            "messages": conversation.messages,
            "system": system_prompt(self.state, &self.agent, &self.session, &conversation.system),
        });
        if !self.tools.is_empty() {
            request["tools"] = json!(self.tools);
        }
        let headers = HeaderMap::new();
        for _ in 0..OVERLOADED_TRIES {
            let req: MessagesRequest =
                serde_json::from_value(request.clone()).map_err(|e| e.to_string())?;
            let mut previews = Previews::default();
            let events = &self.live.events;
            let reply = {
                let mut watch = |piece: &Piece| previews.watch(piece, events);
                tokio::select! {
                    reply = run_one_watched(self.state, &headers, req, &mut watch) => reply,
                    _ = self.live.interrupted() => return Ok(None),
                }
            };
            match reply {
                Ok(message) => return Ok(Some((message, previews))),
                Err(response) if response.status() == StatusCode::SERVICE_UNAVAILABLE => {
                    tokio::time::sleep(OVERLOADED_RETRY).await;
                }
                Err(response) => {
                    let error = error_of(*response).await;
                    return Err(error["error"]["message"]
                        .as_str()
                        .unwrap_or("the model request failed")
                        .to_string());
                }
            }
        }
        Err("the server stayed too busy to take the model request".into())
    }

    /// Runs the loop until the turn ends, the client must answer, or it is
    /// interrupted.
    async fn run(&mut self) -> Ended {
        for _ in 0..MAX_STEPS {
            if self.live.interrupted.load(Ordering::SeqCst) {
                return Ended::Turn;
            }
            // Answered calls first, then whatever the client has sent since.
            let pending = {
                let _turn = self.live.lock.lock().await;
                std::mem::take(&mut load_conversation(self.state, self.id).pending)
            };
            let settled = if pending.is_empty() {
                Vec::new()
            } else {
                match self.settle(pending).await {
                    Some(results) => results,
                    None => return Ended::Turn,
                }
            };
            let conversation = {
                let _turn = self.live.lock.lock().await;
                let mut conversation = load_conversation(self.state, self.id);
                conversation.pending.clear();
                push_user(&mut conversation.messages, settled);
                let inbox = std::mem::take(&mut conversation.inbox);
                push_user(&mut conversation.messages, inbox);
                save_conversation(self.state, self.id, &conversation);
                conversation
            };
            if conversation
                .messages
                .last()
                .is_none_or(|m| m["role"] != "user")
            {
                return Ended::Turn;
            }

            let start = self.emit(json!({"type": "span.model_request_start"}));
            let reply = self.ask_model(&conversation).await;
            let usage = reply
                .as_ref()
                .ok()
                .and_then(|r| r.as_ref())
                .map(|(m, _)| m["usage"].clone())
                .unwrap_or_default();
            let count = |key: &str| usage[key].as_u64().unwrap_or(0);
            self.emit(json!({
                "type": "span.model_request_end",
                "model_request_start_id": start["id"],
                "is_error": reply.is_err(),
                "model_usage": {
                    "input_tokens": count("input_tokens"),
                    "output_tokens": count("output_tokens"),
                    "cache_read_input_tokens": count("cache_read_input_tokens"),
                    "cache_creation_input_tokens": count("cache_creation_input_tokens"),
                },
            }));
            let (message, mut previews) = match reply {
                Ok(Some(reply)) => reply,
                Ok(None) => return Ended::Turn,
                Err(error) => return Ended::Failed(error),
            };
            let add = |s: &mut Value| {
                for key in ["input_tokens", "output_tokens", "cache_read_input_tokens"] {
                    let sum = s["usage"][key].as_u64().unwrap_or(0) + count(key);
                    s["usage"][key] = json!(sum);
                }
            };
            update(self.state, self.id, add);
            // A child thread's tokens are the session's too.
            if let (session, Some(_)) = split_key(self.id) {
                update(self.state, session, add);
            }

            let mut kept = Vec::new();
            let mut text = Vec::new();
            let mut calls = Vec::new();
            for block in message["content"].as_array().into_iter().flatten() {
                match block["type"].as_str() {
                    Some("text") => {
                        text.push(json!({"type": "text", "text": block["text"]}));
                        kept.push(block.clone());
                    }
                    Some("thinking") => {
                        self.flush_text(&mut text, &mut previews);
                        let mut event = json!({"type": "agent.thinking"});
                        if let Some(id) = previews.next_thinking() {
                            event["id"] = json!(id);
                        }
                        self.emit(event);
                    }
                    Some("tool_use") => {
                        self.flush_text(&mut text, &mut previews);
                        kept.push(block.clone());
                        calls.push(self.announce(block));
                    }
                    _ => {}
                }
            }
            self.flush_text(&mut text, &mut previews);
            if !kept.is_empty() {
                let _turn = self.live.lock.lock().await;
                let mut conversation = load_conversation(self.state, self.id);
                conversation
                    .messages
                    .push(json!({"role": "assistant", "content": kept}));
                save_conversation(self.state, self.id, &conversation);
            }
            if calls.is_empty() {
                let _turn = self.live.lock.lock().await;
                if load_conversation(self.state, self.id).inbox.is_empty() {
                    return Ended::Turn;
                }
                continue;
            }
            for call in calls.iter_mut() {
                if !call.waiting {
                    match self.execute(call).await {
                        Some(result) => call.result = Some(result),
                        None => {
                            self.store_pending(calls.clone()).await;
                            return Ended::Turn;
                        }
                    }
                }
            }
            let waiting: Vec<String> = calls
                .iter()
                .filter(|c| c.waiting)
                .map(|c| c.event_id.clone())
                .collect();
            self.store_pending(calls).await;
            if !waiting.is_empty() {
                return Ended::Waiting(waiting);
            }
        }
        Ended::Turn
    }

    async fn store_pending(&self, calls: Vec<Call>) {
        let _turn = self.live.lock.lock().await;
        let mut conversation = load_conversation(self.state, self.id);
        conversation.pending = calls;
        save_conversation(self.state, self.id, &conversation);
    }

    fn flush_text(&self, text: &mut Vec<Value>, previews: &mut Previews) {
        if !text.is_empty() {
            let mut event = json!({"type": "agent.message", "content": std::mem::take(text)});
            if let Some(id) = previews.next_message() {
                event["id"] = json!(id);
            }
            self.emit(event);
        }
    }

    /// Reports a call the model made, and says whether it may run now.
    fn announce(&self, block: &Value) -> Call {
        let name = block["name"].as_str().unwrap_or_default().to_string();
        let input = block["input"].clone();
        let tool_use_id = block["id"].as_str().unwrap_or_default().to_string();
        // A consultation shows as its thread's events, not as a call, and
        // a workflow run as its own events.
        if self.advises(&name) || self.runs_workflows(&name) {
            return Call {
                tool_use_id,
                event_id: new_id("sevt_"),
                name,
                input,
                kind: Kind::Builtin,
                waiting: false,
                decision: None,
                result: None,
            };
        }
        let builtin = self.builtin.iter().find(|t| t.name == name);
        let (kind, event) = if self.delegates() && DELEGATION_TOOLS.contains(&name.as_str()) {
            (
                Kind::Builtin,
                json!({
                    "type": "agent.tool_use", "name": name, "input": input,
                    "evaluated_permission": "allow",
                    "evaluation": {"type": "always_allow"},
                }),
            )
        } else if let Some(tool) = builtin {
            let (permission, evaluation) = tool.evaluate();
            (
                Kind::Builtin,
                json!({
                    "type": "agent.tool_use", "name": name, "input": input,
                    "evaluated_permission": permission, "evaluation": evaluation,
                }),
            )
        } else if self.custom.contains(&name) {
            (
                Kind::Custom,
                json!({"type": "agent.custom_tool_use", "name": name, "input": input}),
            )
        } else {
            let server = self.mcp_server_of(&name);
            let policy = mcp_toolset(&self.agent, &server)
                .and_then(|t| toolset::resolve_config(t, &name, Policy::AlwaysAsk).ok())
                .map_or(Policy::AlwaysAsk, |(_, p)| p);
            let (permission, evaluation) = policy.evaluate(false);
            (
                Kind::Mcp,
                json!({
                    "type": "agent.mcp_tool_use", "name": name, "input": input,
                    "mcp_server_name": server,
                    "evaluated_permission": permission, "evaluation": evaluation,
                }),
            )
        };
        let waiting = kind == Kind::Custom || event["evaluated_permission"] == "ask";
        let event = self.emit(event);
        // A child thread's call that waits on the client is shown on the
        // session's own stream, where the client answers it.
        if waiting && let (session, Some(thread)) = split_key(self.id) {
            let mut posted = event.clone();
            posted["session_thread_id"] = json!(thread);
            emit(self.state, session, posted);
        }
        Call {
            tool_use_id,
            event_id: event["id"].as_str().unwrap_or_default().to_string(),
            name,
            input,
            kind,
            waiting,
            decision: None,
            result: None,
        }
    }

    fn mcp_server_of(&self, tool: &str) -> String {
        let parsed = ParsedToolCall {
            name: tool.to_string(),
            arguments: "{}".into(),
        };
        self.server
            .kind(tool)
            .map(|kind| self.server.use_block(&kind, "", &parsed))
            .and_then(|b| b["server_name"].as_str().map(str::to_string))
            .unwrap_or_default()
    }
}

/// Starts the agent loop unless it already runs: it takes whatever the
/// conversation has waiting.
fn kick(state: &Arc<AppState>, id: &str) {
    let live = live(id);
    if live.running.swap(true, Ordering::SeqCst) {
        return;
    }
    let state = state.clone();
    let id = id.to_string();
    tokio::spawn(async move {
        loop {
            let started = Instant::now();
            update(&state, &id, |s| s["status"] = json!("running"));
            announce_running(&state, &id);
            let ended = run_loop(&state, &id, &live).await;
            if let Some(session) = load(&state, &id) {
                capture_outputs(&state, &session).await;
            }
            // A turn that ends with an outcome active is graded; work to
            // revise goes on without the session going idle.
            if matches!(ended, Ended::Turn)
                && !live.interrupted.load(Ordering::SeqCst)
                && grade(&state, &id, &live).await
            {
                continue;
            }
            update(&state, &id, |s| {
                let active = s["usage"]["active_seconds"].as_f64().unwrap_or(0.0)
                    + started.elapsed().as_secs_f64();
                s["usage"]["active_seconds"] = json!((active * 1000.0).round() / 1000.0);
                s["status"] = json!("idle");
                s["updated_at"] = json!(rfc3339(unix_now()));
            });
            let _turn = live.lock.lock().await;
            let stop_reason = match ended {
                Ended::Turn => json!({"type": "end_turn"}),
                Ended::Waiting(ids) => json!({"type": "requires_action", "event_ids": ids}),
                Ended::Failed(message) => {
                    emit(
                        &state,
                        &id,
                        json!({
                            "type": "session.error",
                            "error": {
                                "type": "model_request_failed_error", "message": message,
                                "retry_status": {"type": "terminal"},
                            },
                        }),
                    );
                    json!({"type": "retries_exhausted"})
                }
            };
            let interrupted = live.interrupted.swap(false, Ordering::SeqCst);
            if interrupted {
                let mut conversation = load_conversation(&state, &id);
                if conversation.outcome.is_some() {
                    end_outcome(
                        &state,
                        &id,
                        &mut conversation,
                        "",
                        "interrupted",
                        "Interrupted by the user.",
                    );
                    save_conversation(&state, &id, &conversation);
                }
                // Calls the interrupt cut short are answered as such, so
                // the conversation stays one the model can be sent.
                let mut conversation = load_conversation(&state, &id);
                if !conversation.pending.is_empty() && stop_reason["type"] != "requires_action" {
                    let results = std::mem::take(&mut conversation.pending)
                        .into_iter()
                        .map(|c| {
                            c.result.unwrap_or_else(|| {
                                tool_result(&c.tool_use_id, "The user interrupted this call.", true)
                            })
                        })
                        .collect();
                    push_user(&mut conversation.messages, results);
                    save_conversation(&state, &id, &conversation);
                }
            }
            let more =
                stop_reason["type"] == "end_turn" && load_conversation(&state, &id).has_work();
            match split_key(&id) {
                (session, None) => {
                    // The session is idle only when none of its threads runs.
                    if !session_busy(&state, session, &id) {
                        let reason = if stop_reason["type"] == "end_turn" {
                            idle_reason(&state, session)
                        } else {
                            stop_reason.clone()
                        };
                        report_idle(&state, session, reason);
                    }
                }
                (session, Some(thread)) => {
                    let event = json!({
                        "type": "session.thread_status_idle", "session_thread_id": thread,
                        "agent_name": thread_name(&state, &id), "stop_reason": stop_reason,
                    });
                    emit(&state, &id, event.clone());
                    emit(&state, session, event);
                }
            }
            if !more {
                live.running.store(false, Ordering::SeqCst);
                drop(_turn);
                if let (session, Some(_)) = split_key(&id) {
                    if workflows::run_of(&state, &id).is_some() {
                        // A run's thread reports to the run's plan.
                        let report = match stop_reason["type"].as_str() {
                            Some("end_turn") if !interrupted => {
                                Some(Ok(report_of(&load_conversation(&state, &id))))
                            }
                            Some("end_turn") => Some(Err("the agent was interrupted".into())),
                            Some("retries_exhausted") => {
                                Some(Err("the agent's model request failed".into()))
                            }
                            _ => None,
                        };
                        if let Some(report) = report {
                            workflows::thread_done(&id, report);
                        }
                    } else {
                        // A child's turn ends with its report to the coordinator.
                        let report = match stop_reason["type"].as_str() {
                            Some("end_turn") if !interrupted => {
                                Some(report_of(&load_conversation(&state, &id)))
                            }
                            Some("retries_exhausted") => {
                                Some("(The agent stopped: its model request failed.)".to_string())
                            }
                            _ => None,
                        };
                        if let Some(report) = report {
                            send_message(&state, &id, session, &report).await;
                        }
                    }
                    if !session_busy(&state, session, "") {
                        report_idle(&state, session, idle_reason(&state, session));
                    }
                }
                return;
            }
        }
    });
}

/// The thread of `session` whose waiting call is the event `call`: the
/// primary thread when no child's is.
fn owner_of(state: &AppState, session: &str, call: &str) -> String {
    let waits = |key: &str| {
        load_conversation(state, key)
            .pending
            .iter()
            .any(|c| c.event_id == call)
    };
    if waits(session) {
        return session.to_string();
    }
    child_threads(state, session)
        .iter()
        .filter_map(|t| t["_key"].as_str())
        .find(|key| waits(key))
        .unwrap_or(session)
        .to_string()
}

/// A thread as the API shows it.
fn thread_json(state: &AppState, session: &Value, record: Option<&Value>) -> Value {
    let session_id = session["id"].as_str().unwrap_or_default();
    match record {
        None => json!({
            "id": primary_thread(session_id), "type": "session_thread",
            "session_id": session_id, "parent_thread_id": null, "workflow_run_id": null,
            "agent": session["agent"],
            "status": if live(session_id).running.load(Ordering::SeqCst) { "running" } else { "idle" },
            "created_at": session["created_at"], "updated_at": session["updated_at"],
            "archived_at": session["archived_at"], "usage": session["usage"],
        }),
        Some(record) => {
            let _ = state;
            json!({
                "id": record["_thread"]["id"], "type": "session_thread",
                "session_id": session_id, "parent_thread_id": record["_thread"]["parent"],
                "workflow_run_id": record["_thread"]["workflow_run_id"], "agent": record["agent"],
                "status": thread_status(record),
                "created_at": record["created_at"], "updated_at": record["updated_at"],
                "archived_at": record["archived_at"], "usage": record["usage"],
            })
        }
    }
}

/// The key of thread `thread` of session `session`.
fn key_of_thread(state: &AppState, session: &str, thread: &str) -> Result<String, Box<Response>> {
    if thread == primary_thread(session) {
        return Ok(session.to_string());
    }
    let key = thread_key(session, thread);
    load(state, &key).ok_or_else(|| missing("session thread", thread))?;
    Ok(key)
}

pub async fn list_threads(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    respond((|| {
        let session = lookup(&state, &id)?;
        let mut items = vec![thread_json(&state, &session, None)];
        items.extend(
            child_threads(&state, &id)
                .iter()
                .map(|t| thread_json(&state, &session, Some(t))),
        );
        cursor_page(items, &query, 100)
    })())
}

pub async fn retrieve_thread(
    State(state): State<Arc<AppState>>,
    Path((id, thread)): Path<(String, String)>,
) -> Response {
    respond((|| {
        let session = lookup(&state, &id)?;
        let key = key_of_thread(&state, &id, &thread)?;
        Ok(match split_key(&key).1 {
            None => thread_json(&state, &session, None),
            Some(_) => thread_json(&state, &session, load(&state, &key).as_ref()),
        })
    })())
}

pub async fn archive_thread(
    State(state): State<Arc<AppState>>,
    Path((id, thread)): Path<(String, String)>,
) -> Response {
    respond((|| {
        let session = lookup(&state, &id)?;
        let key = key_of_thread(&state, &id, &thread)?;
        if split_key(&key).1.is_none() {
            return Err(bad("the primary thread is archived with its session"));
        }
        if workflows::run_of(&state, &key).is_some_and(|run| workflows::is_open(&run)) {
            return Err(workflows::open_error(format!(
                "thread '{thread}' belongs to a workflow run that is open: the run archives it"
            )));
        }
        if live(&key).running.load(Ordering::SeqCst) {
            return Err(bad(format!(
                "thread '{thread}' is running: interrupt it before archiving it"
            )));
        }
        update(&state, &key, |r| {
            if r["archived_at"].is_null() {
                r["archived_at"] = json!(rfc3339(unix_now()));
            }
        });
        let event = json!({"type": "session.thread_status_terminated",
                           "session_thread_id": thread, "agent_name": thread_name(&state, &key)});
        emit(&state, &key, event.clone());
        emit(&state, &id, event);
        Ok(thread_json(&state, &session, load(&state, &key).as_ref()))
    })())
}

pub async fn list_thread_events(
    State(state): State<Arc<AppState>>,
    Path((id, thread)): Path<(String, String)>,
    query: Query<Vec<(String, String)>>,
) -> Response {
    match key_of_thread(&state, &id, &thread) {
        Ok(key) => list_events(State(state), Path(key), query).await,
        Err(response) => *response,
    }
}

pub async fn stream_thread_events(
    State(state): State<Arc<AppState>>,
    Path((id, thread)): Path<(String, String)>,
    query: Query<Vec<(String, String)>>,
) -> Response {
    match key_of_thread(&state, &id, &thread) {
        Ok(key) => stream_events(State(state), Path(key), query).await,
        Err(response) => *response,
    }
}

/// A child thread's agent's name.
fn thread_name(state: &AppState, key: &str) -> String {
    load(state, key)
        .and_then(|r| r["_thread"]["agent_name"].as_str().map(str::to_string))
        .unwrap_or_default()
}

/// The events of a thread starting to run: a child's own, shown on the
/// session's stream too, and the session's when it was idle until now.
fn announce_running(state: &AppState, key: &str) {
    match split_key(key) {
        (session, None) => report_running(state, session),
        (session, Some(thread)) => {
            let event = json!({"type": "session.thread_status_running",
                               "session_thread_id": thread,
                               "agent_name": thread_name(state, key)});
            emit(state, key, event.clone());
            report_running(state, session);
            emit(state, session, event);
        }
    }
}

/// `session.status_running`, unless the stream already says so.
fn report_running(state: &AppState, session: &str) {
    if live(session).idle_reported.swap(false, Ordering::SeqCst) {
        emit(state, session, json!({"type": "session.status_running"}));
    }
}

/// `session.status_idle` for why the session stopped — but not the end of
/// a turn while a workflow run is open: the session runs on with it.
fn report_idle(state: &AppState, session: &str, reason: Value) {
    if reason["type"] == "end_turn" && workflows::any_open(session) {
        return;
    }
    live(session).idle_reported.store(true, Ordering::SeqCst);
    emit(
        state,
        session,
        json!({"type": "session.status_idle", "stop_reason": reason}),
    );
}

/// The coordinator's delegation tools.
const DELEGATION_TOOLS: [&str; 2] = ["list_agents", "send_to_agent"];

/// The most child threads one session holds, archived ones aside.
const MAX_CHILD_THREADS: usize = 25;

/// The delegation tools as the model is given them.
fn delegation_definitions(roster: &Value) -> Vec<Value> {
    let names: Vec<&str> = roster["agents"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["name"].as_str())
        .collect();
    let mut send = json!({
        "name": "send_to_agent",
        "description": format!(
            "Give an agent a task in a thread of its own, or send an existing thread a \
             follow-up. The agents are: {}. Returns at once; the agent's report arrives as a \
             message in a later turn.",
            if names.is_empty() { "none listed".to_string() } else { names.join(", ") }
        ),
        "input_schema": {"type": "object", "required": ["message"], "properties": {
            "message": {"type": "string",
                        "description": "The task or follow-up, complete in itself."},
            "agent_name": {"type": "string",
                           "description": "The agent to start a new thread with."},
            "session_thread_id": {"type": "string",
                                  "description": "An existing thread to send a follow-up to."},
        }},
    });
    if roster["inline"] == true {
        send["input_schema"]["properties"]["inline_agent"] = json!({
            "type": "object", "required": ["name", "system"],
            "description": "An agent you define for this task, instead of a listed one: it has \
                            your tools and model.",
            "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                           "system": {"type": "string"}},
        });
    }
    vec![
        json!({"name": "list_agents",
               "description": "List the agents you can delegate to and your threads with their status.",
               "input_schema": {"type": "object", "properties": {}}}),
        send,
    ]
}

/// An agent the coordinator or a workflow run's plan defines: the session
/// agent's model, tools, MCP servers and skills, its name and prompts its
/// own.
fn inline_agent(agent: &Value) -> Value {
    let mut config = json!({"type": "inline", "name": "agent", "description": null,
                            "system": null});
    for key in [
        "model",
        "tools",
        "mcp_servers",
        "skills",
        "execution_identity",
    ] {
        config[key] = agent[key].clone();
    }
    config
}

/// A session's child threads' records, oldest first.
fn child_threads(state: &AppState, session: &str) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(dir(state, session).join("threads"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            load(
                state,
                &thread_key(session, &e.file_name().to_string_lossy()),
            )
        })
        .collect();
    out.sort_by(|a, b| a["_order"].as_str().cmp(&b["_order"].as_str()));
    out
}

/// A session's child threads but the advisor's and its workflow runs':
/// the ones its coordinator delegated to.
fn delegated_threads(state: &AppState, session: &str) -> Vec<Value> {
    child_threads(state, session)
        .into_iter()
        .filter(|t| t["agent"]["type"] != "advisor" && t["_thread"]["workflow_run_id"].is_null())
        .collect()
}

/// A thread's status: `terminated` once archived, else running or idle.
fn thread_status(record: &Value) -> &'static str {
    let key = record["_key"]
        .as_str()
        .or(record["id"].as_str())
        .unwrap_or_default();
    if !record["archived_at"].is_null() {
        "terminated"
    } else if live(key).running.load(Ordering::SeqCst) {
        "running"
    } else {
        "idle"
    }
}

/// Makes a child thread of the session `session` that runs `agent` under
/// `name` — for the workflow run `run`, when one makes it: its key.
fn create_thread(
    state: &AppState,
    session: &Value,
    mut agent: Value,
    name: &str,
    run: Option<&str>,
) -> Result<String, String> {
    let session_id = session["id"].as_str().unwrap_or_default();
    let thread = new_id("sthr_");
    let key = thread_key(session_id, &thread);
    match agent["type"].as_str() {
        Some("advisor") => {}
        Some("inline") => {
            if let Some(map) = agent.as_object_mut() {
                for key in ["id", "version", "multiagent"] {
                    map.remove(key);
                }
            }
        }
        _ => agent["multiagent"] = Value::Null,
    }
    let now = rfc3339(unix_now());
    let mut record = session.clone();
    if let Some(map) = record.as_object_mut() {
        map.remove("_roster");
        map.remove("_workflows");
    }
    record["_key"] = json!(key);
    record["_thread"] = json!({"id": thread, "agent_name": name,
                               "parent": primary_thread(session_id),
                               "workflow_run_id": run});
    record["_order"] = json!(format!("{:039}", now_nanos()));
    record["agent"] = agent;
    record["status"] = json!("idle");
    record["created_at"] = json!(now);
    record["updated_at"] = json!(now);
    record["archived_at"] = Value::Null;
    record["usage"] = json!({"input_tokens": 0, "output_tokens": 0,
                             "cache_read_input_tokens": 0, "active_seconds": 0.0});
    let root = dir(state, &key);
    std::fs::create_dir_all(&root)
        .and_then(|()| std::fs::write(root.join("events.jsonl"), b""))
        .map_err(|e| format!("making the thread: {e}"))?;
    save(state, &record);
    save_conversation(state, &key, &Conversation::default());
    emit(
        state,
        session_id,
        json!({"type": "session.thread_created", "session_thread_id": thread,
               "agent_name": name, "workflow_run_id": run}),
    );
    Ok(key)
}

/// Sends `text` from one thread to another — the coordinator's task or
/// follow-up, or a child's report — as each side's event, into the
/// receiver's conversation, and starts the receiver.
async fn send_message(state: &Arc<AppState>, from: &str, to: &str, text: &str) {
    let name_of = |key: &str| -> (String, String) {
        let record = load(state, key).unwrap_or_default();
        match split_key(key) {
            (_, Some(thread)) => (
                thread.to_string(),
                record["_thread"]["agent_name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            ),
            (session, None) => (
                primary_thread(session),
                record["agent"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            ),
        }
    };
    let ((from_id, from_name), (to_id, to_name)) = (name_of(from), name_of(to));
    let content = json!([{"type": "text", "text": text}]);
    emit(
        state,
        from,
        json!({"type": "agent.thread_message_sent", "to_session_thread_id": to_id,
               "to_agent_name": to_name, "content": content}),
    );
    emit(
        state,
        to,
        json!({"type": "agent.thread_message_received", "from_session_thread_id": from_id,
               "from_agent_name": from_name, "content": content}),
    );
    let receiver = live(to);
    {
        let _turn = receiver.lock.lock().await;
        let mut conversation = load_conversation(state, to);
        let heading = if split_key(to).1.is_none() {
            format!("Report from {from_name} (session_thread_id {from_id}):\n\n")
        } else {
            String::new()
        };
        conversation
            .inbox
            .push(json!({"type": "text", "text": format!("{heading}{text}")}));
        save_conversation(state, to, &conversation);
    }
    kick(state, to);
}

/// What a child thread's turn ended with, as its report.
fn report_of(conversation: &Conversation) -> String {
    let text: Vec<&str> = conversation
        .messages
        .iter()
        .rev()
        .find(|m| m["role"] == "assistant")
        .and_then(|m| m["content"].as_array())
        .into_iter()
        .flatten()
        .filter_map(|b| b["text"].as_str())
        .collect();
    let text = text.join("\n").trim().to_string();
    if text.is_empty() {
        "(The agent ended its turn without a report.)".into()
    } else {
        text
    }
}

/// Whether any of a session's threads but `except` is running.
fn session_busy(state: &AppState, session: &str, except: &str) -> bool {
    let primary = session != except && live(session).running.load(Ordering::SeqCst);
    primary
        || child_threads(state, session).iter().any(|t| {
            let key = t["_key"].as_str().unwrap_or_default();
            key != except && live(key).running.load(Ordering::SeqCst)
        })
}

/// Why the whole session is idle: the calls any of its threads waits on,
/// or the end of its turn.
fn idle_reason(state: &AppState, session: &str) -> Value {
    let mut keys = vec![session.to_string()];
    keys.extend(
        child_threads(state, session)
            .iter()
            .filter_map(|t| t["_key"].as_str().map(str::to_string)),
    );
    let waiting: Vec<String> = keys
        .iter()
        .flat_map(|key| {
            load_conversation(state, key)
                .waiting()
                .into_iter()
                .map(|c| c.event_id.clone())
                .collect::<Vec<_>>()
        })
        .collect();
    if waiting.is_empty() {
        json!({"type": "end_turn"})
    } else {
        json!({"type": "requires_action", "event_ids": waiting})
    }
}

/// Gives the sandbox the vaults' environment variables as placeholders, and
/// — when it has the network — routes it through an egress proxy that puts
/// the secrets in their place: the proxy, which runs while it is held.
async fn vault_secrets(
    state: &AppState,
    vault_ids: &[String],
    container: &mut Container,
) -> Result<Option<super::egress::Proxy>, String> {
    let secrets = super::vaults::env_secrets(state, vault_ids);
    if secrets.is_empty() {
        return Ok(None);
    }
    for secret in &secrets {
        container.set_env(&secret.name, &secret.placeholder);
    }
    if !container.has_network() {
        return Ok(None);
    }
    let bundle = super::egress::bundle(&state.api_data)?;
    let proxy = super::egress::start(secrets).await?;
    let address = format!("http://127.0.0.1:{}", proxy.port);
    for name in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"] {
        container.set_env(name, &address);
    }
    container.set_env("NO_PROXY", "localhost,127.0.0.1,::1");
    container.set_env("no_proxy", "localhost,127.0.0.1,::1");
    for name in [
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "GIT_SSL_CAINFO",
        "PIP_CERT",
    ] {
        container.set_env(name, super::egress::BUNDLE_IN_SANDBOX);
    }
    container.bind_file(bundle, super::egress::BUNDLE_IN_SANDBOX);
    Ok(Some(proxy))
}

async fn run_loop(state: &Arc<AppState>, id: &str, live: &Live) -> Ended {
    let Some(session) = load(state, id) else {
        return Ended::Turn;
    };
    let agent = session["agent"].clone();
    let environment = session["_environment"].clone();
    let mut container = match Container::open(&state.api_data, session["_container"].as_str()).await
    {
        Ok(container) => container,
        Err(err) => return Ended::Failed(err),
    };
    if environment["config"]["networking"]["type"] == "unrestricted" {
        container.share_network();
    }
    mount_session(&mut container, &session);
    let vault_ids: Vec<String> = session["vault_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    // MCP servers authenticate with the vaults' tokens for their URLs.
    let mut servers = mcp_servers(&agent);
    for server in &mut servers {
        if let Some(url) = server["url"].as_str().map(str::to_string)
            && let Some(token) = super::vaults::mcp_token(state, &vault_ids, &url).await
        {
            server["authorization_token"] = json!(token);
        }
    }
    // Environment-variable credentials: placeholders in the sandbox, the
    // secrets put in on the way out. Held for as long as the loop runs.
    let _egress = match vault_secrets(state, &vault_ids, &mut container).await {
        Ok(proxy) => proxy,
        Err(err) => return Ended::Failed(err),
    };
    let declarations = server_declarations(state.web_search != "off", &agent, &environment);
    let conversation = load_conversation(state, id);
    let mut text = conversation_text_of(&conversation);
    text.push_str(agent["system"].as_str().unwrap_or_default());
    let server = match ServerTools::from_request(
        &declarations,
        &servers,
        &text,
        &state.web_search,
        &state.code_execution,
    )
    .await
    {
        Ok(server) => server,
        Err(err) => return Ended::Failed(err),
    };
    let builtin = builtin_tools(&agent);
    let mut tools: Vec<Value> = builtin
        .iter()
        .filter_map(|t| toolset::definition(t.name))
        .collect();
    for function in server.definitions() {
        let declared = client_shape(&function);
        if !tools.iter().any(|t| t["name"] == declared["name"]) {
            tools.push(declared);
        }
    }
    if split_key(id).1.is_none() && session["_roster"].is_object() {
        tools.extend(delegation_definitions(&session["_roster"]));
    }
    let advisor = split_key(id)
        .1
        .is_none()
        .then(|| super::agents::advisor_model(&agent["multiagent"]).map(str::to_string))
        .flatten();
    if advisor.is_some() {
        tools.push(client_shape(&advisor::definition(advisor::TOOL)));
    }
    if split_key(id).1.is_none() && session["_workflows"].is_object() {
        tools.extend(workflows::definitions(&session["_workflows"]));
    }
    let custom = custom_tools(&agent);
    let custom_names = custom
        .iter()
        .filter_map(|t| t["name"].as_str().map(str::to_string))
        .collect();
    tools.extend(custom);
    let mut run = Agent {
        state,
        id,
        live,
        agent,
        session,
        builtin,
        custom: custom_names,
        runner: Runner::new(container),
        server,
        tools,
        advisor,
    };
    run.run().await
}

/// The text of everything the client sent, for the URLs `web_fetch` may
/// fetch.
fn conversation_text_of(conversation: &Conversation) -> String {
    let messages: Vec<super::AnthropicMessage> = conversation
        .messages
        .iter()
        .chain(conversation.inbox.iter())
        .filter_map(|m| serde_json::from_value(m.clone()).ok())
        .collect();
    let mut text = conversation_text(&messages, None);
    for block in &conversation.inbox {
        text.push('\n');
        text.push_str(&block.to_string());
    }
    text
}

/// A user event's content blocks: a string becomes one text block.
fn content_of(event: &Value) -> Result<Vec<Value>, String> {
    match &event["content"] {
        Value::String(text) => Ok(vec![json!({"type": "text", "text": text})]),
        Value::Array(blocks) if !blocks.is_empty() => {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text" | "image" | "document") => {}
                    Some(other) => {
                        return Err(format!("a '{other}' block cannot go in a user message"));
                    }
                    None => return Err("a content block needs a `type`".into()),
                }
            }
            Ok(blocks.clone())
        }
        _ => Err("`content` must be a non-empty array of content blocks".into()),
    }
}

/// Applies the events a client sent to the conversation, all or none:
/// what each one is recorded as, and whether one interrupts.
fn apply(
    conversation: &mut Conversation,
    sent: &[Value],
    only_messages: bool,
) -> Result<(Vec<Value>, bool), String> {
    let mut echoed = Vec::new();
    let mut interrupt = false;
    for event in sent {
        let kind = event["type"].as_str().unwrap_or_default();
        if only_messages && !matches!(kind, "user.message" | "user.define_outcome") {
            return Err(format!(
                "initial_events take only user.message and user.define_outcome events, not \
                 '{kind}'"
            ));
        }
        let mut echo = json!({"type": kind});
        match kind {
            "user.message" => {
                let waiting = conversation.waiting();
                if !waiting.is_empty() {
                    return Err(format!(
                        "the session is waiting on {}: answer with user.custom_tool_result or \
                         user.tool_confirmation first",
                        waiting
                            .iter()
                            .map(|c| c.event_id.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                let content = content_of(event)?;
                echo["content"] = json!(content);
                conversation.inbox.extend(content);
            }
            "user.interrupt" => interrupt = true,
            "system.message" => {
                let content = event["content"]
                    .as_array()
                    .filter(|c| !c.is_empty())
                    .ok_or("a system.message needs `content`: text blocks")?;
                let mut text = Vec::new();
                for block in content {
                    text.push(
                        block["text"]
                            .as_str()
                            .filter(|_| block["type"] == "text")
                            .ok_or("a system.message's content is text blocks")?
                            .to_string(),
                    );
                }
                echo["content"] = json!(content);
                conversation.system.push(text.join("\n"));
            }
            "user.custom_tool_result" => {
                let id = event["custom_tool_use_id"]
                    .as_str()
                    .ok_or("a user.custom_tool_result needs `custom_tool_use_id`")?;
                let call = conversation
                    .pending
                    .iter_mut()
                    .find(|c| c.event_id == id && c.kind == Kind::Custom && c.result.is_none())
                    .ok_or_else(|| format!("no custom tool call '{id}' is waiting for a result"))?;
                let is_error = event["is_error"].as_bool().unwrap_or(false);
                let content = event["content"].clone();
                let content = if content.is_null() {
                    json!([])
                } else {
                    content
                };
                call.result = Some(json!({
                    "type": "tool_result", "tool_use_id": call.tool_use_id,
                    "content": content, "is_error": is_error,
                }));
                echo["custom_tool_use_id"] = json!(id);
                echo["content"] = content;
                echo["is_error"] = json!(is_error);
            }
            "user.tool_confirmation" => {
                let id = event["tool_use_id"]
                    .as_str()
                    .ok_or("a user.tool_confirmation needs `tool_use_id`")?;
                let call = conversation
                    .pending
                    .iter_mut()
                    .find(|c| {
                        c.event_id == id
                            && c.kind != Kind::Custom
                            && c.waiting
                            && c.decision.is_none()
                            && c.result.is_none()
                    })
                    .ok_or_else(|| format!("no tool call '{id}' is waiting for a confirmation"))?;
                let deny_message = event["deny_message"].as_str().map(str::to_string);
                call.decision = Some(match event["result"].as_str() {
                    Some("allow") => Ok(()),
                    Some("deny") => Err(deny_message.clone()),
                    _ => return Err("a user.tool_confirmation's `result` is allow or deny".into()),
                });
                echo["tool_use_id"] = json!(id);
                echo["result"] = event["result"].clone();
                if let Some(message) = deny_message {
                    echo["deny_message"] = json!(message);
                }
            }
            "user.define_outcome" => {
                if let Some(active) = &conversation.outcome {
                    return Err(format!(
                        "outcome '{}' is still being worked on: wait for its \
                         span.outcome_evaluation_end, or interrupt it",
                        active.outcome_id
                    ));
                }
                if !conversation.waiting().is_empty() {
                    return Err("the session is waiting on a tool call: answer it first".into());
                }
                let description = event["description"]
                    .as_str()
                    .filter(|d| !d.trim().is_empty())
                    .ok_or("a user.define_outcome needs a `description`")?;
                let rubric = match event["rubric"]["type"].as_str() {
                    Some("text") => event["rubric"]["content"].as_str(),
                    Some("file") => event["_rubric_text"].as_str(),
                    _ => None,
                }
                .filter(|r| !r.trim().is_empty())
                .ok_or("a user.define_outcome needs a `rubric`: text content or a file")?;
                let max_iterations = match &event["max_iterations"] {
                    Value::Null => 3,
                    value => value
                        .as_u64()
                        .filter(|n| (1..=20).contains(n))
                        .ok_or("`max_iterations` is from 1 to 20")?
                        as usize,
                };
                let outcome_id = new_id("outc_");
                echo["outcome_id"] = json!(outcome_id);
                echo["description"] = json!(description);
                echo["rubric"] = event["rubric"].clone();
                echo["max_iterations"] = json!(max_iterations);
                conversation
                    .inbox
                    .push(json!({"type": "text", "text": format!(
                        "Your task: {description}\n\nWhen you finish, your work is graded against \
                         this rubric, criterion by criterion:\n\n{rubric}\n\nWork until it meets \
                         every criterion. Put files you deliver in /mnt/session/outputs."
                    )}));
                conversation.outcome = Some(Outcome {
                    outcome_id,
                    description: description.to_string(),
                    rubric: rubric.to_string(),
                    max_iterations,
                    iteration: 0,
                });
            }
            other => {
                return Err(format!(
                    "'{other}' is not an event a client sends: the events are {ACCEPTED_EVENTS}"
                ));
            }
        }
        echoed.push(echo);
    }
    Ok((echoed, interrupt))
}

/// `sent`, with the text of each `user.define_outcome`'s file rubric read
/// in from the Files API.
fn with_rubrics(state: &AppState, sent: &[Value]) -> Result<Vec<Value>, Box<Response>> {
    let store = FileStore::at(&state.api_data);
    let mut out = sent.to_vec();
    for event in &mut out {
        if event["type"] != "user.define_outcome" || event["rubric"]["type"] != "file" {
            continue;
        }
        let file_id = event["rubric"]["file_id"].as_str().unwrap_or_default();
        let (_, bytes) = store
            .read(file_id)
            .ok_or_else(|| missing("file", file_id))?;
        let text = String::from_utf8(bytes)
            .map_err(|_| bad(format!("the rubric file '{file_id}' is not text")))?;
        event["_rubric_text"] = json!(text);
    }
    Ok(out)
}

/// Ends the active outcome with `result`: its `span.outcome_evaluation_end`
/// (after the start event `start`, empty when grading never began), and
/// its entry in the session's `outcome_evaluations`.
fn end_outcome(
    state: &AppState,
    id: &str,
    conversation: &mut Conversation,
    start: &str,
    result: &str,
    explanation: &str,
) {
    let Some(outcome) = conversation.outcome.take() else {
        return;
    };
    record_evaluation(state, id, &outcome, start, result, explanation, json!({}));
}

/// Emits a grading's `span.outcome_evaluation_end` and records it in the
/// session's `outcome_evaluations`.
fn record_evaluation(
    state: &AppState,
    id: &str,
    outcome: &Outcome,
    start: &str,
    result: &str,
    explanation: &str,
    usage: Value,
) {
    let count = |key: &str| usage[key].as_u64().unwrap_or(0);
    emit(
        state,
        id,
        json!({
            "type": "span.outcome_evaluation_end",
            "outcome_evaluation_start_id": start,
            "outcome_id": outcome.outcome_id,
            "iteration": outcome.iteration,
            "result": result,
            "explanation": explanation,
            "usage": {
                "input_tokens": count("input_tokens"),
                "output_tokens": count("output_tokens"),
                "cache_read_input_tokens": count("cache_read_input_tokens"),
                "cache_creation_input_tokens": count("cache_creation_input_tokens"),
            },
        }),
    );
    let entry = json!({
        "type": "outcome_evaluation", "outcome_id": outcome.outcome_id,
        "description": outcome.description, "iteration": outcome.iteration,
        "result": result, "explanation": explanation,
        "completed_at": rfc3339(unix_now()),
    });
    update(state, id, |s| {
        let list = s["outcome_evaluations"].as_array_mut();
        match list {
            Some(list) => {
                list.retain(|e| e["outcome_id"] != outcome.outcome_id.as_str());
                list.push(entry);
            }
            None => s["outcome_evaluations"] = json!([entry]),
        }
    });
}

/// The grader's instructions: it sees the work, not the agent's reasoning.
const GRADER: &str = "You grade one piece of work against a rubric. Judge every criterion \
on its own, strictly, from the evidence you are given: the task, the rubric, the agent's final \
messages, and the files it delivered. You cannot run anything, and a claim in the agent's \
messages is not evidence that a file says it. `satisfied` means every criterion is met; \
`needs_revision` means some are not and the work can be fixed; `failed` means the rubric cannot \
be met for this task at all, as when the rubric and the task contradict each other. In \
`explanation`, name each unmet criterion and what is missing, or say briefly why all are met.";

/// The most of one delivered file the grader is shown, and of them all.
const EVIDENCE_FILE_CHARS: usize = 20_000;
const EVIDENCE_CHARS: usize = 80_000;

/// The files under `dir`, as paths relative to it, at most `limit`.
fn files_in(dir: &std::path::Path, limit: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(rel) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(dir.join(&rel)) else {
            continue;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            let path = rel.join(&name);
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => pending.push(path),
                Ok(kind) if kind.is_file() => out.push(path),
                _ => {}
            }
            if out.len() >= limit {
                return out;
            }
        }
    }
    out
}

/// What the grader judges the work by.
fn evidence(
    state: &AppState,
    session: &Value,
    conversation: &Conversation,
    outcome: &Outcome,
) -> String {
    let mut text = format!(
        "# Task\n\n{}\n\n# Rubric\n\n{}\n\n# The agent's final messages\n\n",
        outcome.description, outcome.rubric
    );
    let said: Vec<String> = conversation
        .messages
        .iter()
        .rev()
        .filter(|m| m["role"] == "assistant")
        .flat_map(|m| m["content"].as_array().cloned().unwrap_or_default())
        .filter_map(|b| b["text"].as_str().map(str::to_string))
        .filter(|t| !t.trim().is_empty())
        .take(3)
        .collect();
    for message in said.iter().rev() {
        text.push_str(message);
        text.push_str("\n\n");
    }
    let root = state
        .api_data
        .join("containers")
        .join(session["_container"].as_str().unwrap_or_default());
    let outputs = session_dir(&root).join("outputs");
    text.push_str("# Files delivered in /mnt/session/outputs\n\n");
    let delivered = files_in(&outputs, 40);
    if delivered.is_empty() {
        text.push_str("(none)\n\n");
    }
    let mut budget = EVIDENCE_CHARS;
    for rel in delivered {
        let path = outputs.join(&rel);
        let shown = format!("/mnt/session/outputs/{}", rel.display());
        match std::fs::read(&path).map(String::from_utf8) {
            Ok(Ok(body)) if budget > 0 => {
                let cut: String = body.chars().take(EVIDENCE_FILE_CHARS.min(budget)).collect();
                budget = budget.saturating_sub(cut.len());
                let more = if cut.len() < body.len() {
                    "\n(… cut short)"
                } else {
                    ""
                };
                text.push_str(&format!("## {shown}\n\n```\n{cut}\n```{more}\n\n"));
            }
            Ok(Ok(body)) => text.push_str(&format!(
                "## {shown}\n\n({} bytes, not shown)\n\n",
                body.len()
            )),
            _ => {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                text.push_str(&format!("## {shown}\n\n(a binary file, {size} bytes)\n\n"));
            }
        }
    }
    let workspace = root.join("workspace");
    let listed = files_in(&workspace, 200);
    if !listed.is_empty() {
        text.push_str("# Files in /workspace\n\n");
        for rel in listed {
            let size = std::fs::metadata(workspace.join(&rel))
                .map(|m| m.len())
                .unwrap_or(0);
            text.push_str(&format!("- /workspace/{} ({size} bytes)\n", rel.display()));
        }
    }
    text
}

/// Grades the active outcome's work, when one is active: the evaluation
/// is recorded, and on `needs_revision` (or `max_iterations_reached`, for
/// one final revision) the agent is handed the gaps. Whether it goes on.
async fn grade(state: &Arc<AppState>, id: &str, live: &Live) -> bool {
    let (outcome, prompt, model) = {
        let _turn = live.lock.lock().await;
        let conversation = load_conversation(state, id);
        let (Some(outcome), Some(session)) = (conversation.outcome.clone(), load(state, id)) else {
            return false;
        };
        let prompt = evidence(state, &session, &conversation, &outcome);
        (outcome, prompt, session["agent"]["model"]["id"].clone())
    };
    let start = emit(
        state,
        id,
        json!({"type": "span.outcome_evaluation_start", "outcome_id": outcome.outcome_id,
               "iteration": outcome.iteration}),
    );
    let start_id = start["id"].as_str().unwrap_or_default().to_string();
    emit(
        state,
        id,
        json!({"type": "span.outcome_evaluation_ongoing", "outcome_id": outcome.outcome_id,
               "iteration": outcome.iteration}),
    );
    let request = json!({
        "model": model, "max_tokens": 2000, "system": GRADER,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": {
            "type": "object", "required": ["result", "explanation"],
            "additionalProperties": false,
            "properties": {
                "result": {"type": "string", "enum": ["satisfied", "needs_revision", "failed"]},
                "explanation": {"type": "string"},
            },
        }}},
    });
    let headers = HeaderMap::new();
    let mut verdict = Err("the server stayed too busy to grade the work".to_string());
    for _ in 0..OVERLOADED_TRIES {
        let Ok(req) = serde_json::from_value::<MessagesRequest>(request.clone()) else {
            break;
        };
        let reply = tokio::select! {
            reply = run_one(state, &headers, req) => reply,
            _ = live.interrupted() => {
                let _turn = live.lock.lock().await;
                let mut conversation = load_conversation(state, id);
                end_outcome(state, id, &mut conversation, &start_id, "interrupted",
                            "Interrupted by the user while the work was graded.");
                save_conversation(state, id, &conversation);
                return false;
            }
        };
        match reply {
            Ok(message) => {
                verdict = Ok(message);
                break;
            }
            Err(response) if response.status() == StatusCode::SERVICE_UNAVAILABLE => {
                tokio::time::sleep(OVERLOADED_RETRY).await;
            }
            Err(response) => {
                let error = error_of(*response).await;
                verdict = Err(error["error"]["message"]
                    .as_str()
                    .unwrap_or("the grader's request failed")
                    .to_string());
                break;
            }
        }
    }
    let (mut result, explanation, usage) = match verdict {
        Ok(message) => {
            let said = message["content"][0]["text"].as_str().unwrap_or_default();
            let parsed: Value = serde_json::from_str(said).unwrap_or_default();
            (
                parsed["result"].as_str().unwrap_or("failed").to_string(),
                parsed["explanation"]
                    .as_str()
                    .unwrap_or("The grader gave no explanation.")
                    .to_string(),
                message["usage"].clone(),
            )
        }
        Err(why) => (
            "failed".to_string(),
            format!("The work could not be graded: {why}."),
            json!({}),
        ),
    };
    let _turn = live.lock.lock().await;
    let mut conversation = load_conversation(state, id);
    let Some(mut active) = conversation.outcome.take() else {
        return false;
    };
    if result == "needs_revision" && active.iteration + 1 >= active.max_iterations {
        result = "max_iterations_reached".into();
    }
    record_evaluation(state, id, &active, &start_id, &result, &explanation, usage);
    let go_on = match result.as_str() {
        "needs_revision" => {
            conversation
                .inbox
                .push(json!({"type": "text", "text": format!(
                    "The grader checked your work against the rubric and it needs revision:\n\n\
                     {explanation}\n\nRevise the work so that it meets every criterion."
                )}));
            active.iteration += 1;
            conversation.outcome = Some(active);
            true
        }
        "max_iterations_reached" => {
            conversation
                .inbox
                .push(json!({"type": "text", "text": format!(
                    "The grader checked your work against the rubric for the last time:\n\n\
                     {explanation}\n\nMake one final revision; it will not be graded again."
                )}));
            true
        }
        _ => false,
    };
    save_conversation(state, id, &conversation);
    go_on
}

/// Applies the events, records them, and starts the loop when there is
/// work for it.
async fn deliver(
    state: &Arc<AppState>,
    id: &str,
    sent: &[Value],
    only_messages: bool,
) -> Result<Vec<Value>, Box<Response>> {
    if sent.is_empty() {
        return Err(bad("`events` must list at least one event"));
    }
    let sent = with_rubrics(state, sent)?;
    let live = live(id);
    let _turn = live.lock.lock().await;
    let mut conversation = load_conversation(state, id);
    let (echoed, interrupt) = apply(&mut conversation, &sent, only_messages).map_err(bad)?;
    // An interrupt while nothing runs ends the outcome there and then.
    if interrupt && !live.running.load(Ordering::SeqCst) {
        end_outcome(
            state,
            id,
            &mut conversation,
            "",
            "interrupted",
            "Interrupted by the user.",
        );
    }
    save_conversation(state, id, &conversation);
    let echoed: Vec<Value> = echoed.into_iter().map(|e| emit(state, id, e)).collect();
    if interrupt && live.running.load(Ordering::SeqCst) {
        live.interrupted.store(true, Ordering::SeqCst);
        live.interrupt.notify_waiters();
    }
    if conversation.has_work() {
        kick(state, id);
    }
    Ok(echoed)
}

fn writable(session: &Value) -> Result<(), Box<Response>> {
    if session["archived_at"].is_null() {
        Ok(())
    } else {
        Err(fail(
            StatusCode::CONFLICT,
            format!(
                "session '{}' is archived, and an archived session cannot change",
                session["id"].as_str().unwrap_or_default()
            ),
        ))
    }
}

pub async fn create_session(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    let reply: Reply = async {
        let body = body(payload)?;
        open_session(&state, &body, false).await
    }
    .await;
    respond(reply)
}

/// Makes a session from a create request's `body`, as `POST /v1/sessions`
/// and a deployment's runs do — a deployment's `initial_events` may hold
/// `system.message` events too.
pub(super) async fn open_session(
    state: &Arc<AppState>,
    body: &Map<String, Value>,
    from_deployment: bool,
) -> Reply {
    let state = state.clone();
    {
        let agent = resolve_agent(
            &state,
            body.get("agent")
                .ok_or_else(|| bad("`agent` is required"))?,
        )?;
        let roster = resolve_roster(&state, &agent)?;
        let workflows = resolve_workflows(&state, &agent)?;
        let environment_id = body
            .get("environment_id")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("`environment_id` is required"))?;
        let environment = environment(&state, environment_id)
            .ok_or_else(|| missing("environment", environment_id))?;
        if !environment["archived_at"].is_null() {
            return Err(bad(format!(
                "environment '{environment_id}' is archived, and a new session cannot use it"
            )));
        }
        let vault_ids: Vec<String> = match body.get("vault_ids") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(ids)) => ids
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| bad("`vault_ids` lists vault ids"))?,
            Some(_) => return Err(bad("`vault_ids` lists vault ids")),
        };
        for vault in &vault_ids {
            super::vaults::usable(&state, vault)?;
        }
        if body.get("budget").is_some_and(|b| !b.is_null()) {
            return Err(bad(
                "budgets are not supported on this server: it has no list prices to count",
            ));
        }
        check_runnable(&state, &agent, &environment).map_err(bad)?;
        let metadata = metadata(body.get("metadata"), None).map_err(bad)?;
        let initial: Vec<Value> = body
            .get("initial_events")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if initial.len() > 50 {
            return Err(bad("initial_events holds at most 50 events"));
        }
        // A deployment's system messages apply before its first message.
        let (notes, initial): (Vec<Value>, Vec<Value>) = initial
            .into_iter()
            .partition(|e| from_deployment && e["type"] == "system.message");
        // Checked before anything is made, so a bad one makes nothing.
        let initial = with_rubrics(&state, &initial)?;
        apply(&mut Conversation::default(), &notes, false).map_err(bad)?;
        apply(&mut Conversation::default(), &initial, true).map_err(bad)?;
        let resources: Vec<Value> = body
            .get("resources")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let (container, resources) = make_container(&state, &agent, &resources).await?;

        let id = new_id("sesn_");
        let now = rfc3339(unix_now());
        let session = json!({
            "id": id,
            "type": "session",
            "agent": agent,
            "environment_id": environment_id,
            "title": body.get("title").cloned().unwrap_or(Value::Null),
            "metadata": metadata,
            "resources": resources,
            "status": "idle",
            "archived_at": null,
            "budget": null,
            "vault_ids": vault_ids,
            "outcome_evaluations": [],
            "created_at": now,
            "updated_at": now,
            "usage": {
                "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                "active_seconds": 0.0,
            },
            "_order": format!("{:039}", now_nanos()),
            "_created": unix_now(),
            "_container": container,
            "_roster": roster,
            "_workflows": workflows,
            "_environment": environment,
        });
        let root = dir(&state, &id);
        std::fs::create_dir_all(&root)
            .and_then(|()| std::fs::write(root.join("events.jsonl"), b""))
            .map_err(|err| {
                fail(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("saving the session: {err}"),
                )
            })?;
        save(&state, &session);
        save_conversation(&state, &id, &Conversation::default());
        super::webhooks::emit(&state, "session.status_scheduled", &id);
        if !notes.is_empty() {
            deliver(&state, &id, &notes, false).await?;
        }
        if !initial.is_empty() {
            deliver(&state, &id, &initial, true).await?;
        }
        Ok(session_json(&load(&state, &id).unwrap_or(session)))
    }
}

fn all_sessions(state: &AppState) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(state.api_data.join("sessions"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| load(state, &e.file_name().to_string_lossy()))
        .collect();
    let order = |v: &Value| v["_order"].as_str().unwrap_or("").to_string();
    out.sort_by_key(|v| std::cmp::Reverse(order(v)));
    out
}

/// Ends the workflow runs a stopped server left open.
pub fn resume_workflows(state: Arc<AppState>) {
    tokio::spawn(workflows::resume(state));
}

pub async fn list_sessions(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    let archived = query.flag("include_archived");
    let agent_id = query.get("agent_id");
    let agent_version = query
        .get("agent_version")
        .and_then(|v| v.parse::<u64>().ok());
    let statuses = query.all("statuses");
    let mut items: Vec<Value> = all_sessions(&state)
        .iter()
        .filter(|s| archived || s["archived_at"].is_null())
        .filter(|s| agent_id.is_none_or(|a| s["agent"]["id"] == a))
        .filter(|s| agent_version.is_none_or(|v| s["agent"]["version"] == v))
        .map(session_json)
        .filter(|s| statuses.is_empty() || statuses.iter().any(|st| s["status"] == *st))
        .collect();
    if query.get("order") == Some("asc") {
        items.reverse();
    }
    respond(cursor_page(items, &query, 20))
}

pub async fn retrieve_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(lookup(&state, &id).map(|s| session_json(&s)))
}

pub async fn update_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    let reply: Reply = async {
        let body = body(payload)?;
        let session = lookup(&state, &id)?;
        writable(&session)?;
        if body.contains_key("vault_ids") {
            return Err(bad(
                "`vault_ids` is set when a session is created, not after",
            ));
        }
        if body.get("budget").is_some_and(|b| !b.is_null()) {
            return Err(bad("budgets are not supported on this server"));
        }
        let mut changed = Map::new();
        if let Some(title) = body.get("title") {
            changed.insert("title".into(), title.clone());
        }
        if body.contains_key("metadata") {
            changed.insert(
                "metadata".into(),
                metadata(body.get("metadata"), Some(&session["metadata"])).map_err(bad)?,
            );
        }
        if let Some(agent) = body.get("agent").and_then(Value::as_object) {
            if workflows::any_open(&id) {
                return Err(workflows::open_error(
                    "the session's agent cannot change while a workflow run is open",
                ));
            }
            if live(&id).running.load(Ordering::SeqCst) {
                return Err(fail(
                    StatusCode::CONFLICT,
                    "the session's agent changes only while the session is idle",
                ));
            }
            let mut overrides = Map::new();
            for key in ["tools", "mcp_servers"] {
                if let Some(value) = agent.get(key) {
                    overrides.insert(key.into(), value.clone());
                }
            }
            if overrides.len() != agent.len() {
                return Err(bad(
                    "a session update changes only the agent's tools and mcp_servers",
                ));
            }
            let updated = override_agent(&state, &session["agent"], &overrides).map_err(bad)?;
            let updated = session_agent(&updated);
            check_runnable(&state, &updated, &session["_environment"]).map_err(bad)?;
            changed.insert("agent".into(), updated);
        }
        let now = rfc3339(unix_now());
        update(&state, &id, |s| {
            for (key, value) in &changed {
                s[key] = value.clone();
            }
            s["updated_at"] = json!(now);
        });
        if !changed.is_empty() {
            let mut event = json!({"type": "session.updated"});
            for (key, value) in &changed {
                event[key] = value.clone();
            }
            emit(&state, &id, event);
        }
        lookup(&state, &id).map(|s| session_json(&s))
    }
    .await;
    respond(reply)
}

/// Interrupts every thread of session `id` that runs — a workflow run's
/// threads only with `runs`.
fn stop(state: &AppState, id: &str, runs: bool) {
    let children = format!("{id}/threads/");
    let lives: Vec<Arc<Live>> = LIVE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|(key, _)| *key == id || key.starts_with(&children))
        .map(|(key, l)| (key.clone(), l.clone()))
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|(key, _)| runs || workflows::run_of(state, key).is_none())
        .map(|(_, l)| l)
        .collect();
    for live in lives {
        if live.running.load(Ordering::SeqCst) {
            live.interrupted.store(true, Ordering::SeqCst);
            live.interrupt.notify_waiters();
        }
    }
}

pub async fn delete_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let reply: Reply = async {
        let session = lookup(&state, &id)?;
        workflows::end_all(&state, &id, false).await;
        stop(&state, &id, true);
        let _ = std::fs::remove_dir_all(dir(&state, &id));
        if let Some(container) = session["_container"].as_str()
            && valid_id(container, "container_")
        {
            let _ = std::fs::remove_dir_all(state.api_data.join("containers").join(container));
        }
        LIVE.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
        super::webhooks::emit(&state, "session.deleted", &id);
        Ok(json!({"id": id, "type": "session_deleted"}))
    }
    .await;
    respond(reply)
}

pub async fn archive_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let reply: Reply = async {
        let session = lookup(&state, &id)?;
        if session["archived_at"].is_null() {
            workflows::end_all(&state, &id, true).await;
            stop(&state, &id, true);
            let now = rfc3339(unix_now());
            update(&state, &id, |s| {
                s["archived_at"] = json!(now);
                s["updated_at"] = json!(now);
            });
            emit(&state, &id, json!({"type": "session.status_terminated"}));
        }
        lookup(&state, &id).map(|s| session_json(&s))
    }
    .await;
    respond(reply)
}

pub async fn list_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    let reply: Reply = (|| {
        lookup(&state, &id)?;
        let types = query.all("types");
        let mut items: Vec<Value> = events(&state, &id)
            .into_iter()
            .filter(|e| types.is_empty() || types.iter().any(|t| e["type"] == *t))
            .collect();
        if query.get("order") == Some("desc") {
            items.reverse();
        }
        cursor_page(items, &query, 1000)
    })();
    respond(reply)
}

pub async fn send_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    let reply: Reply = async {
        let body = body(payload)?;
        let session = lookup(&state, &id)?;
        writable(&session)?;
        let sent = body
            .get("events")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| bad("`events` must be an array of events"))?;
        // Each event goes to the thread it is for: an answer to the thread
        // whose call it answers, an interrupt to the thread it names.
        let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
        let mut interrupt_all = false;
        for event in sent {
            let answered = match event["type"].as_str() {
                Some("user.tool_confirmation") => event["tool_use_id"].as_str(),
                Some("user.custom_tool_result") => event["custom_tool_use_id"].as_str(),
                _ => None,
            };
            let mut target = match answered {
                Some(call) => owner_of(&state, &id, call),
                None => id.clone(),
            };
            if event["type"] == "user.interrupt" {
                match event["session_thread_id"].as_str() {
                    Some(thread) if thread == primary_thread(&id) => {}
                    Some(thread) => {
                        let key = thread_key(&id, thread);
                        let record =
                            load(&state, &key).ok_or_else(|| missing("session thread", thread))?;
                        if !record["archived_at"].is_null() {
                            return Err(bad(format!("thread '{thread}' is terminated")));
                        }
                        // A workflow run's threads stop with their run, not
                        // on an interrupt.
                        if !record["_thread"]["workflow_run_id"].is_null() {
                            continue;
                        }
                        target = key;
                    }
                    None => interrupt_all = true,
                }
            }
            match groups.last_mut() {
                Some((key, events)) if *key == target => events.push(event),
                _ => groups.push((target, vec![event])),
            }
        }
        let mut echoed = Vec::new();
        for (key, events) in groups {
            echoed.extend(deliver(&state, &key, &events, false).await?);
        }
        if interrupt_all {
            stop(&state, &id, false);
        }
        Ok(json!({"data": echoed}))
    }
    .await;
    respond(reply)
}

pub async fn stream_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    if let Err(response) = lookup(&state, &id) {
        return *response;
    }
    let deltas = query.all("event_deltas");
    if deltas.len() > 100
        || deltas
            .iter()
            .any(|d| !matches!(*d, "agent.message" | "agent.thinking"))
    {
        return *bad("`event_deltas` takes agent.message and agent.thinking");
    }
    let wants = |kind: &str| deltas.contains(&kind);
    let (messages, thinking) = (wants("agent.message"), wants("agent.thinking"));
    let mut rx = live(&id).events.subscribe();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    // Previews go only to a connection that asked for them.
                    let wanted = match event["type"].as_str() {
                        Some("event_start") => match event["event"]["type"].as_str() {
                            Some("agent.message") => messages,
                            Some("agent.thinking") => thinking,
                            _ => false,
                        },
                        Some("event_delta") => messages,
                        _ => true,
                    };
                    if !wanted {
                        continue;
                    }
                    let name = event["type"].as_str().unwrap_or("event").to_string();
                    yield Ok::<_, std::convert::Infallible>(
                        Event::default().event(name).data(event.to_string()),
                    );
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    axum::response::sse::Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(event_id: &str, kind: Kind, waiting: bool) -> Call {
        Call {
            tool_use_id: format!("toolu_{event_id}"),
            event_id: event_id.into(),
            name: "t".into(),
            input: json!({}),
            kind,
            waiting,
            decision: None,
            result: None,
        }
    }

    #[test]
    fn a_message_is_refused_while_a_call_waits_and_answers_settle_it() {
        let mut c = Conversation {
            pending: vec![
                call("sevt_a", Kind::Custom, true),
                call("sevt_b", Kind::Builtin, true),
            ],
            ..Default::default()
        };
        let message = json!({"type": "user.message", "content": "hi"});
        assert!(apply(&mut c, std::slice::from_ref(&message), false).is_err());
        assert!(!c.has_work());
        let answers = [
            json!({"type": "user.custom_tool_result", "custom_tool_use_id": "sevt_a",
                   "content": [{"type": "text", "text": "42"}]}),
            json!({"type": "user.tool_confirmation", "tool_use_id": "sevt_b", "result": "deny",
                   "deny_message": "not that"}),
        ];
        let (echoed, interrupt) = apply(&mut c, &answers, false).unwrap();
        assert_eq!(echoed.len(), 2);
        assert!(!interrupt);
        assert!(c.has_work());
        assert_eq!(c.pending[1].decision, Some(Err(Some("not that".into()))));
    }

    #[test]
    fn a_bad_event_changes_nothing_it_is_sent_with() {
        let mut c = Conversation::default();
        let sent = [
            json!({"type": "user.message", "content": "hi"}),
            json!({"type": "user.define_outcome", "description": "x"}),
        ];
        let mut copy = Conversation::default();
        assert!(apply(&mut copy, &sent, false).is_err());
        assert!(c.inbox.is_empty());
        assert!(apply(&mut c, &sent[..1], false).is_ok());
        assert_eq!(c.inbox, vec![json!({"type": "text", "text": "hi"})]);
    }

    /// A child thread's key is its session's with its own, and the primary
    /// thread's id is the session's own.
    #[test]
    fn thread_keys_split_into_session_and_thread() {
        let key = thread_key("sesn_ab12", "sthr_cd34");
        assert_eq!(split_key(&key), ("sesn_ab12", Some("sthr_cd34")));
        assert_eq!(split_key("sesn_ab12"), ("sesn_ab12", None));
        assert_eq!(primary_thread("sesn_ab12"), "sthr_ab12");
    }

    /// A child's report is its last assistant turn's text.
    #[test]
    fn a_report_is_the_last_reply() {
        let mut c = Conversation::default();
        assert!(report_of(&c).contains("without a report"));
        c.messages = vec![
            json!({"role": "assistant", "content": [{"type": "text", "text": "early"}]}),
            json!({"role": "user", "content": [{"type": "tool_result"}]}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "391"}]}),
        ];
        assert_eq!(report_of(&c), "391");
    }

    /// The coordinator is told its agents, and offered to define its own
    /// only when the roster allows it.
    #[test]
    fn delegation_tools_name_the_roster() {
        let roster = json!({"agents": [{"name": "math-helper"}], "inline": false});
        let tools = delegation_definitions(&roster);
        assert_eq!(tools[0]["name"], "list_agents");
        assert!(
            tools[1]["description"]
                .as_str()
                .unwrap()
                .contains("math-helper")
        );
        assert!(
            tools[1]["input_schema"]["properties"]
                .get("inline_agent")
                .is_none()
        );
        let open = delegation_definitions(&json!({"agents": [], "inline": true}));
        assert!(
            open[1]["input_schema"]["properties"]
                .get("inline_agent")
                .is_some()
        );
    }

    #[test]
    fn initial_events_take_only_messages_and_an_outcome() {
        let sent = [json!({"type": "user.interrupt"})];
        assert!(apply(&mut Conversation::default(), &sent, true).is_err());
        let outcome = json!({"type": "user.define_outcome", "description": "Do it",
                             "rubric": {"type": "text", "content": "- it is done"}});
        assert!(
            apply(
                &mut Conversation::default(),
                std::slice::from_ref(&outcome),
                true
            )
            .is_ok()
        );
        assert!(
            apply(
                &mut Conversation::default(),
                &[outcome.clone(), outcome],
                true
            )
            .is_err()
        );
    }

    /// An outcome becomes the agent's task, with its rubric, and holds
    /// until it ends; its fields are checked.
    #[test]
    fn an_outcome_is_the_agents_task_until_it_ends() {
        let mut c = Conversation::default();
        let outcome = json!({"type": "user.define_outcome", "description": "Write a haiku",
                             "rubric": {"type": "text", "content": "- three lines"},
                             "max_iterations": 5});
        let (echoed, _) = apply(&mut c, std::slice::from_ref(&outcome), false).unwrap();
        assert!(
            echoed[0]["outcome_id"]
                .as_str()
                .unwrap()
                .starts_with("outc_")
        );
        assert_eq!(echoed[0]["max_iterations"], 5);
        let task = c.inbox[0]["text"].as_str().unwrap();
        assert!(
            task.contains("Write a haiku") && task.contains("- three lines"),
            "{task}"
        );
        assert!(c.has_work());
        assert!(apply(&mut c, &[outcome], false).is_err());
        let bad = |field: Value| {
            let mut e = json!({"type": "user.define_outcome", "description": "x",
                               "rubric": {"type": "text", "content": "y"}});
            e.as_object_mut()
                .unwrap()
                .extend(field.as_object().unwrap().clone());
            apply(&mut Conversation::default(), &[e], false).is_err()
        };
        assert!(bad(json!({"rubric": null})));
        assert!(bad(json!({"max_iterations": 21})));
        assert!(bad(json!({"description": " "})));
    }

    #[test]
    fn user_content_joins_the_last_user_message() {
        let mut messages = vec![json!({"role": "assistant", "content": [{"type": "text"}]})];
        push_user(&mut messages, vec![json!({"type": "tool_result"})]);
        push_user(&mut messages, vec![json!({"type": "text", "text": "and"})]);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn an_upload_path_stays_under_uploads() {
        assert_eq!(upload_path("file_1", None).unwrap(), "file_1");
        assert_eq!(
            upload_path("file_1", Some("/data/a.csv")).unwrap(),
            "data/a.csv"
        );
        assert!(upload_path("file_1", Some("../x")).is_err());
    }

    #[test]
    fn web_tools_under_limited_networking_reach_only_its_hosts() {
        let agent =
            json!({"tools": [toolset::normalize(&json!({"type": toolset::TOOLSET})).unwrap()]});
        let env = |networking: Value| json!({"config": {"networking": networking}});
        let none = env(json!({"type": "limited", "allowed_hosts": []}));
        let some = env(json!({"type": "limited", "allowed_hosts": ["docs.rs"]}));
        let open = env(json!({"type": "unrestricted"}));
        assert!(server_declarations(true, &agent, &none).is_empty());
        let limited = server_declarations(true, &agent, &some);
        assert_eq!(limited.len(), 2);
        assert!(
            limited
                .iter()
                .all(|d| d["allowed_domains"] == json!(["docs.rs"]))
        );
        let unlimited = server_declarations(false, &agent, &open);
        assert_eq!(
            unlimited,
            vec![json!({"type": "web_fetch_20250910", "name": "web_fetch"})]
        );
    }
}
