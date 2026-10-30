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

//! Memory stores (`/v1/memory_stores`): named collections of small text
//! documents that outlive a session. A store is a record under
//! `<api_data>/memory_stores/`, its memories and their versions records
//! under `<api_data>/memory_stores/<id>/`. Every change to a memory —
//! through the API or by an agent — adds an immutable version.
//!
//! A session that attaches a store sees it as a directory, its memories as
//! files under their paths: [`export`] writes them there when the session
//! is created, and [`sync_back`] records what the agent changed after each
//! of its file tool calls and turns.

use axum::{
    Json,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use super::agents::{
    ListQuery, Records, Reply, bad, body, cursor_page, fail, metadata, missing, respond,
};
use super::files::{new_id, now_nanos};
use super::{rfc3339, unix_now};
use crate::http::AppState;

/// The largest memory, in bytes.
pub(super) const MAX_MEMORY_BYTES: usize = 100 * 1024;

/// The most stores one session attaches.
pub(super) const MAX_SESSION_STORES: usize = 8;

/// Serialises every change to memories and their versions.
fn lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn stores(state: &AppState) -> Records {
    Records::new(state, "memory_stores")
}

fn memories(state: &AppState, store: &str) -> Records {
    Records::new(state, &format!("memory_stores/{store}/memories"))
}

fn versions(state: &AppState, store: &str) -> Records {
    Records::new(state, &format!("memory_stores/{store}/versions"))
}

fn order_stamp() -> String {
    format!("{:039}", now_nanos())
}

fn sha256(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// A `409` with one of the memory error types, and what else it names.
fn conflict(kind: &str, message: String, extra: Value) -> Box<Response> {
    let mut error = json!({"type": kind, "message": message});
    if let (Some(error), Value::Object(extra)) = (error.as_object_mut(), extra) {
        error.extend(extra);
    }
    Box::new(
        (
            StatusCode::CONFLICT,
            Json(json!({"type": "error", "error": error})),
        )
            .into_response(),
    )
}

/// Who made a change: a session, or a client of the API.
pub(super) fn session_actor(session: &str) -> Value {
    json!({"type": "session_actor", "session_id": session})
}

fn api_actor() -> Value {
    json!({"type": "api_actor", "api_key_id": "local"})
}

/// A memory path checked: absolute, segments plain, no trailing slash.
fn checked_path(path: &str) -> Result<String, String> {
    let fine = path.starts_with('/')
        && path.len() <= 1024
        && !path.ends_with('/')
        && !path.contains('\0')
        && path[1..]
            .split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..");
    if fine {
        Ok(path.to_string())
    } else {
        Err(format!(
            "'{path}' is not a memory path: an absolute path of plain names, such as /notes/today.md"
        ))
    }
}

fn checked_content(content: &str) -> Result<(), String> {
    if content.len() > MAX_MEMORY_BYTES {
        return Err(format!(
            "a memory holds at most {} KB",
            MAX_MEMORY_BYTES / 1024
        ));
    }
    Ok(())
}

fn public(record: &Value) -> Value {
    let mut out = record.clone();
    if let Some(map) = out.as_object_mut() {
        map.retain(|k, _| !k.starts_with('_'));
    }
    out
}

/// A memory as the API shows it, its content only in the `full` view.
fn memory_json(record: &Value, full: bool) -> Value {
    let mut out = public(record);
    if !full {
        out["content"] = Value::Null;
    }
    out
}

/// A store, unless it does not exist.
fn store_lookup(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    stores(state)
        .get(id, "memstore_")
        .ok_or_else(|| missing("memory store", id))
}

/// A store that may still change.
fn writable_store(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    let store = store_lookup(state, id)?;
    if !store["archived_at"].is_null() {
        return Err(bad(format!(
            "memory store '{id}' is archived: its memories can be read and deleted, not changed"
        )));
    }
    Ok(store)
}

fn memory_lookup(state: &AppState, store: &str, id: &str) -> Result<Value, Box<Response>> {
    memories(state, store)
        .get(id, "mem_")
        .ok_or_else(|| missing("memory", id))
}

fn at_path(state: &AppState, store: &str, path: &str) -> Option<Value> {
    memories(state, store)
        .all()
        .into_iter()
        .find(|m| m["path"] == path)
}

/// Records a version of `memory` after `operation`.
fn add_version(
    state: &AppState,
    memory: &Value,
    operation: &str,
    actor: Value,
) -> Result<String, Box<Response>> {
    let store = memory["memory_store_id"].as_str().unwrap_or_default();
    let id = new_id("memver_");
    let deleted = operation == "deleted";
    let version = json!({
        "id": id, "type": "memory_version",
        "memory_id": memory["id"], "memory_store_id": store,
        "operation": operation,
        "path": memory["path"],
        "content": if deleted { Value::Null } else { memory["content"].clone() },
        "content_sha256": if deleted { Value::Null } else { memory["content_sha256"].clone() },
        "content_size_bytes": if deleted { Value::Null } else { memory["content_size_bytes"].clone() },
        "created_at": rfc3339(unix_now()),
        "created_by": actor,
        "redacted_at": Value::Null, "redacted_by": Value::Null,
        "_order": order_stamp(),
    });
    versions(state, store).put(&id, &version)?;
    Ok(id)
}

/// Creates the memory at `path`; a `409` when one is there.
fn create(
    state: &AppState,
    store: &str,
    path: &str,
    content: &str,
    actor: Value,
) -> Result<Value, Box<Response>> {
    let path = checked_path(path).map_err(bad)?;
    checked_content(content).map_err(bad)?;
    if let Some(existing) = at_path(state, store, &path) {
        return Err(conflict(
            "memory_path_conflict_error",
            format!("a memory is already at {path}"),
            json!({"conflicting_memory_id": existing["id"]}),
        ));
    }
    let id = new_id("mem_");
    let now = rfc3339(unix_now());
    let mut memory = json!({
        "id": id, "type": "memory", "memory_store_id": store,
        "path": path, "content": content,
        "content_sha256": sha256(content), "content_size_bytes": content.len(),
        "created_at": now, "updated_at": now,
        "_order": order_stamp(),
    });
    memory["memory_version_id"] = json!(add_version(state, &memory, "created", actor)?);
    memories(state, store).put(&id, &memory)?;
    Ok(memory)
}

/// Changes `memory`'s path, content or both.
fn modify(
    state: &AppState,
    mut memory: Value,
    path: Option<&str>,
    content: Option<&str>,
    actor: Value,
) -> Result<Value, Box<Response>> {
    let store = memory["memory_store_id"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    if let Some(path) = path {
        let path = checked_path(path).map_err(bad)?;
        if let Some(other) = at_path(state, &store, &path).filter(|o| o["id"] != memory["id"]) {
            return Err(conflict(
                "memory_path_conflict_error",
                format!("a memory is already at {path}"),
                json!({"conflicting_memory_id": other["id"]}),
            ));
        }
        memory["path"] = json!(path);
    }
    if let Some(content) = content {
        checked_content(content).map_err(bad)?;
        memory["content"] = json!(content);
        memory["content_sha256"] = json!(sha256(content));
        memory["content_size_bytes"] = json!(content.len());
    }
    memory["updated_at"] = json!(rfc3339(unix_now()));
    memory["memory_version_id"] = json!(add_version(state, &memory, "modified", actor)?);
    let id = memory["id"].as_str().unwrap_or_default().to_string();
    memories(state, &store).put(&id, &memory)?;
    Ok(memory)
}

fn remove(state: &AppState, memory: &Value, actor: Value) -> Result<(), Box<Response>> {
    let store = memory["memory_store_id"].as_str().unwrap_or_default();
    add_version(state, memory, "deleted", actor)?;
    memories(state, store).remove(memory["id"].as_str().unwrap_or_default());
    Ok(())
}

// --- Sessions ---------------------------------------------------------------

/// The directory name a store is mounted under in `/mnt/memory/`.
pub(super) fn slug(name: &str) -> String {
    let slug: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug = slug.trim_matches(|c| c == '-' || c == '.').to_string();
    if slug.is_empty() {
        "store".into()
    } else {
        slug
    }
}

/// The store a session's `memory_store` resource names, for attaching: it
/// must exist and not be archived.
pub(super) fn attachable(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    let store = store_lookup(state, id)?;
    if !store["archived_at"].is_null() {
        return Err(bad(format!(
            "memory store '{id}' is archived and cannot be attached to a new session"
        )));
    }
    Ok(store)
}

/// Writes every memory of `store` under `dir` at its path: what was
/// written, as path → SHA-256, for [`sync_back`] to compare against.
pub(super) fn export(
    state: &AppState,
    store: &str,
    dir: &std::path::Path,
) -> std::io::Result<HashMap<String, String>> {
    let _guard = lock();
    std::fs::create_dir_all(dir)?;
    let mut written = HashMap::new();
    for memory in memories(state, store).all() {
        let (Some(path), Some(content)) = (memory["path"].as_str(), memory["content"].as_str())
        else {
            continue;
        };
        let target = dir.join(path.trim_start_matches('/'));
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, content)?;
        written.insert(path.to_string(), sha256(content));
    }
    Ok(written)
}

/// The text files under `dir` as memory path → content: hidden names, and
/// files too large or not text, left out.
fn files_under(dir: &std::path::Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut pending = vec![std::path::PathBuf::new()];
    while let Some(rel) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(dir.join(&rel)) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            let path = rel.join(&name);
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file()
                && entry
                    .metadata()
                    .is_ok_and(|m| m.len() as usize <= MAX_MEMORY_BYTES)
                && let Ok(content) = std::fs::read_to_string(entry.path())
            {
                out.insert(format!("/{}", path.to_string_lossy()), content);
            }
            if out.len() >= 10_000 {
                return out;
            }
        }
    }
    out
}

/// Records in `store` what the session changed under `dir` since
/// `baseline` (path → SHA-256 of what was there), and moves `baseline` on:
/// a new file creates a memory, a changed one modifies it, a removed one
/// deletes it — each a version by the session.
pub(super) fn sync_back(
    state: &AppState,
    store: &str,
    dir: &std::path::Path,
    baseline: &mut HashMap<String, String>,
    session: &str,
) {
    let _guard = lock();
    let now = files_under(dir);
    let mut paths: BTreeSet<&String> = now.keys().collect();
    paths.extend(baseline.keys());
    let mut next = HashMap::new();
    for path in paths {
        let before = baseline.get(path);
        let after = now.get(path).map(|content| (content, sha256(content)));
        if after.as_ref().map(|(_, sha)| sha) == before {
            if let Some(sha) = before {
                next.insert(path.clone(), sha.clone());
            }
            continue;
        }
        let existing = at_path(state, store, path);
        let done = match (&after, existing) {
            (Some((content, _)), Some(memory)) => {
                modify(state, memory, None, Some(content), session_actor(session)).is_ok()
            }
            (Some((content, _)), None) => {
                create(state, store, path, content, session_actor(session)).is_ok()
            }
            (None, Some(memory)) => remove(state, &memory, session_actor(session)).is_ok(),
            (None, None) => true,
        };
        if let Some((_, sha)) = after.filter(|_| done) {
            next.insert(path.clone(), sha);
        }
    }
    *baseline = next;
}

// --- Stores -----------------------------------------------------------------

fn string_field(
    body: &Map<String, Value>,
    key: &str,
    max: usize,
) -> Result<Option<String>, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.chars().count() <= max => Ok(Some(s.clone())),
        Some(Value::String(_)) => Err(format!("`{key}` is over {max} characters")),
        Some(_) => Err(format!("`{key}` must be a string")),
    }
}

pub async fn create_store(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let name = string_field(&body, "name", 255)
            .map_err(bad)?
            .filter(|n| !n.trim().is_empty())
            .ok_or_else(|| bad("a memory store needs a `name`"))?;
        let description = string_field(&body, "description", 1024)
            .map_err(bad)?
            .unwrap_or_default();
        let metadata = metadata(body.get("metadata"), None).map_err(bad)?;
        let id = new_id("memstore_");
        let now = rfc3339(unix_now());
        let store = json!({
            "id": id, "type": "memory_store", "name": name, "description": description,
            "metadata": metadata, "created_at": now, "updated_at": now,
            "archived_at": Value::Null, "_order": order_stamp(),
        });
        let _guard = lock();
        stores(&state).put(&id, &store)?;
        super::webhooks::emit(&state, "memory_store.created", &id);
        Ok(public(&store))
    })())
}

pub async fn list_stores(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    let archived = query.flag("include_archived");
    let (gte, lte) = (query.get("created_at_gte"), query.get("created_at_lte"));
    let items = stores(&state)
        .all()
        .iter()
        .filter(|s| archived || s["archived_at"].is_null())
        .filter(|s| {
            let at = s["created_at"].as_str().unwrap_or_default();
            gte.is_none_or(|g| at >= g) && lte.is_none_or(|l| at <= l)
        })
        .map(public)
        .collect();
    respond(cursor_page(items, &query, 20))
}

pub async fn retrieve_store(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(store_lookup(&state, &id).map(|s| public(&s)))
}

pub async fn update_store(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = lock();
        let mut store = writable_store(&state, &id)?;
        if let Some(name) = string_field(&body, "name", 255).map_err(bad)? {
            if name.trim().is_empty() {
                return Err(bad("a memory store's `name` cannot be empty"));
            }
            store["name"] = json!(name);
        }
        if body.contains_key("description") {
            store["description"] = json!(
                string_field(&body, "description", 1024)
                    .map_err(bad)?
                    .unwrap_or_default()
            );
        }
        if body.contains_key("metadata") {
            store["metadata"] =
                metadata(body.get("metadata"), Some(&store["metadata"])).map_err(bad)?;
        }
        store["updated_at"] = json!(rfc3339(unix_now()));
        stores(&state).put(&id, &store)?;
        Ok(public(&store))
    })())
}

pub async fn delete_store(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    respond((|| {
        let _guard = lock();
        store_lookup(&state, &id)?;
        stores(&state).remove(&id);
        let _ = std::fs::remove_dir_all(state.api_data.join("memory_stores").join(&id));
        super::webhooks::emit(&state, "memory_store.deleted", &id);
        Ok(json!({"id": id, "type": "memory_store_deleted"}))
    })())
}

pub async fn archive_store(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    respond((|| {
        let _guard = lock();
        let mut store = store_lookup(&state, &id)?;
        if store["archived_at"].is_null() {
            let now = rfc3339(unix_now());
            store["archived_at"] = json!(now);
            store["updated_at"] = json!(now);
            stores(&state).put(&id, &store)?;
            super::webhooks::emit(&state, "memory_store.archived", &id);
        }
        Ok(public(&store))
    })())
}

// --- Memories ---------------------------------------------------------------

fn full_view(query: &ListQuery, default: bool) -> Result<bool, Box<Response>> {
    match query.get("view") {
        None => Ok(default),
        Some("full") => Ok(true),
        Some("basic") => Ok(false),
        Some(other) => Err(bad(format!("`view` is basic or full, not '{other}'"))),
    }
}

/// The entries a listing shows: memories under `prefix`, those deeper than
/// `depth` levels below it folded into one `memory_prefix` per directory.
fn listing(all: Vec<Value>, prefix: &str, depth: Option<usize>, full: bool) -> Vec<Value> {
    let mut all: Vec<Value> = all
        .into_iter()
        .filter(|m| m["path"].as_str().is_some_and(|p| p.starts_with(prefix)))
        .collect();
    all.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    let mut out = Vec::new();
    let mut folded = BTreeSet::new();
    for memory in all {
        let path = memory["path"].as_str().unwrap_or_default();
        let rest = path[prefix.len()..].trim_start_matches('/');
        let segments: Vec<&str> = rest.split('/').collect();
        match depth {
            Some(depth) if segments.len() > depth.max(1) => {
                let base = prefix.trim_end_matches('/');
                let folder = format!("{base}/{}/", segments[..depth.max(1)].join("/"));
                if folded.insert(folder.clone()) {
                    out.push(json!({"type": "memory_prefix", "path": folder}));
                }
            }
            _ => out.push(memory_json(&memory, full)),
        }
    }
    out
}

pub async fn list_memories(
    State(state): State<Arc<AppState>>,
    Path(store): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    respond((|| {
        let query = ListQuery::new(pairs);
        store_lookup(&state, &store)?;
        let full = full_view(&query, false)?;
        let depth = match query.get("depth") {
            Some(d) => Some(
                d.parse::<usize>()
                    .map_err(|_| bad("`depth` is a number of levels"))?,
            ),
            None => None,
        };
        let prefix = query.get("path_prefix").unwrap_or("/");
        let items = listing(memories(&state, &store).all(), prefix, depth, full);
        cursor_page(items, &query, 20)
    })())
}

pub async fn create_memory(
    State(state): State<Arc<AppState>>,
    Path(store): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let path = body
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("a memory needs a `path`"))?;
        let content = body
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("a memory needs its `content`"))?;
        let _guard = lock();
        writable_store(&state, &store)?;
        create(&state, &store, path, content, api_actor()).map(|m| memory_json(&m, true))
    })())
}

pub async fn retrieve_memory(
    State(state): State<Arc<AppState>>,
    Path((store, id)): Path<(String, String)>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    respond((|| {
        let full = full_view(&ListQuery::new(pairs), true)?;
        store_lookup(&state, &store)?;
        memory_lookup(&state, &store, &id).map(|m| memory_json(&m, full))
    })())
}

pub async fn update_memory(
    State(state): State<Arc<AppState>>,
    Path((store, id)): Path<(String, String)>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = lock();
        writable_store(&state, &store)?;
        let memory = memory_lookup(&state, &store, &id)?;
        if let Some(precondition) = body.get("precondition").filter(|p| !p.is_null()) {
            if precondition["type"] != "content_sha256" {
                return Err(bad("the only `precondition` type is content_sha256"));
            }
            if precondition["content_sha256"] != memory["content_sha256"] {
                return Err(conflict(
                    "memory_precondition_failed_error",
                    format!("memory '{id}' has changed since it was read"),
                    json!({}),
                ));
            }
        }
        let path = body.get("path").and_then(Value::as_str);
        let content = body.get("content").and_then(Value::as_str);
        if path.is_none() && content.is_none() {
            return Err(bad("an update changes `content`, `path` or both"));
        }
        modify(&state, memory, path, content, api_actor()).map(|m| memory_json(&m, true))
    })())
}

pub async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Path((store, id)): Path<(String, String)>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    respond((|| {
        let query = ListQuery::new(pairs);
        let _guard = lock();
        store_lookup(&state, &store)?;
        let memory = memory_lookup(&state, &store, &id)?;
        if let Some(expected) = query.get("expected_content_sha256")
            && memory["content_sha256"] != expected
        {
            return Err(conflict(
                "memory_precondition_failed_error",
                format!("memory '{id}' has changed since it was read"),
                json!({}),
            ));
        }
        remove(&state, &memory, api_actor())?;
        Ok(json!({"id": id, "type": "memory_deleted"}))
    })())
}

// --- Versions ---------------------------------------------------------------

fn version_json(record: &Value, full: bool) -> Value {
    let mut out = public(record);
    if !full {
        out["content"] = Value::Null;
    }
    out
}

pub async fn list_versions(
    State(state): State<Arc<AppState>>,
    Path(store): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    respond((|| {
        let query = ListQuery::new(pairs);
        store_lookup(&state, &store)?;
        let full = full_view(&query, false)?;
        let wanted = |key: &str, value: &Value| query.get(key).is_none_or(|w| value == w);
        let (gte, lte) = (query.get("created_at_gte"), query.get("created_at_lte"));
        let items = versions(&state, &store)
            .all()
            .iter()
            .filter(|v| wanted("memory_id", &v["memory_id"]))
            .filter(|v| wanted("operation", &v["operation"]))
            .filter(|v| wanted("session_id", &v["created_by"]["session_id"]))
            .filter(|v| wanted("api_key_id", &v["created_by"]["api_key_id"]))
            .filter(|v| {
                let at = v["created_at"].as_str().unwrap_or_default();
                gte.is_none_or(|g| at >= g) && lte.is_none_or(|l| at <= l)
            })
            .map(|v| version_json(v, full))
            .collect();
        cursor_page(items, &query, 20)
    })())
}

pub async fn retrieve_version(
    State(state): State<Arc<AppState>>,
    Path((store, id)): Path<(String, String)>,
) -> Response {
    respond((|| -> Reply {
        store_lookup(&state, &store)?;
        versions(&state, &store)
            .get(&id, "memver_")
            .map(|v| version_json(&v, true))
            .ok_or_else(|| missing("memory version", &id))
    })())
}

pub async fn redact_version(
    State(state): State<Arc<AppState>>,
    Path((store, id)): Path<(String, String)>,
) -> Response {
    respond((|| {
        let _guard = lock();
        store_lookup(&state, &store)?;
        let records = versions(&state, &store);
        let mut version = records
            .get(&id, "memver_")
            .ok_or_else(|| missing("memory version", &id))?;
        if let Some(memory) =
            memories(&state, &store).get(version["memory_id"].as_str().unwrap_or_default(), "mem_")
            && memory["memory_version_id"] == id
        {
            return Err(fail(
                StatusCode::CONFLICT,
                "this is the memory's current version: change or delete the memory first",
            ));
        }
        for key in ["content", "content_sha256", "content_size_bytes", "path"] {
            version[key] = Value::Null;
        }
        version["redacted_at"] = json!(rfc3339(unix_now()));
        version["redacted_by"] = api_actor();
        records.put(&id, &version)?;
        Ok(version_json(&version, true))
    })())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_paths_are_absolute_and_plain() {
        assert!(checked_path("/notes/today.md").is_ok());
        for bad in ["notes.md", "/a/../b", "/a//b", "/a/", "/", "/a/./b"] {
            assert!(checked_path(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_store_name_becomes_a_plain_directory_name() {
        assert_eq!(slug("User Preferences"), "user-preferences");
        assert_eq!(slug("../etc"), "etc");
        assert_eq!(slug("  "), "store");
    }

    /// Deeper memories fold into one prefix per directory at `depth`.
    #[test]
    fn a_listing_folds_what_is_deeper_than_its_depth() {
        let memory = |path: &str| json!({"type": "memory", "path": path, "content": "x"});
        let all = vec![
            memory("/a.md"),
            memory("/notes/b.md"),
            memory("/notes/deep/c.md"),
            memory("/other/d.md"),
        ];
        let paths = |items: Vec<Value>| -> Vec<String> {
            items
                .iter()
                .map(|i| {
                    format!(
                        "{}:{}",
                        i["type"].as_str().unwrap(),
                        i["path"].as_str().unwrap()
                    )
                })
                .collect()
        };
        assert_eq!(
            paths(listing(all.clone(), "/", Some(1), false)),
            [
                "memory:/a.md",
                "memory_prefix:/notes/",
                "memory_prefix:/other/"
            ]
        );
        assert_eq!(
            paths(listing(all.clone(), "/notes/", Some(1), true)),
            ["memory:/notes/b.md", "memory_prefix:/notes/deep/"]
        );
        assert_eq!(listing(all, "/", None, false).len(), 4);
    }
}
