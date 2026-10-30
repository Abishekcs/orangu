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

//! Managed Agents' agents and environments: `/v1/agents` (create, list,
//! retrieve, update, archive, versions) and `/v1/environments` (create,
//! list, retrieve, update, delete, archive).
//!
//! Each is one JSON record under `<api_data>/agents/` or
//! `<api_data>/environments/`. An agent's record holds every version of
//! it; an update that changes anything adds a version.

use axum::{
    Json,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Map, Value, json};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::files::{new_id, now_nanos, valid_id};
use super::{error_response, json_rejection, rfc3339, toolset, unix_now};
use crate::http::AppState;

/// What a handler answers: the JSON, or the error response.
pub(super) type Reply = Result<Value, Box<Response>>;

pub(super) fn fail(status: StatusCode, message: impl Into<String>) -> Box<Response> {
    Box::new(error_response(status, message))
}

pub(super) fn bad(message: impl Into<String>) -> Box<Response> {
    fail(StatusCode::BAD_REQUEST, message)
}

pub(super) fn missing(what: &str, id: &str) -> Box<Response> {
    fail(
        StatusCode::NOT_FOUND,
        format!("{what} '{id}' does not exist"),
    )
}

pub(super) fn respond(reply: Reply) -> Response {
    match reply {
        Ok(value) => Json(value).into_response(),
        Err(response) => *response,
    }
}

/// A JSON body, or the `400` it is rejected with.
pub(super) fn body(
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Map<String, Value>, Box<Response>> {
    match payload {
        Ok(Json(Value::Object(map))) => Ok(map),
        Ok(_) => Err(bad("the request body must be a JSON object")),
        Err(rejection) => Err(Box::new(json_rejection(rejection))),
    }
}

/// A list endpoint's query, every pair kept: `statuses[]` and `types[]`
/// repeat.
pub(super) struct ListQuery {
    pairs: Vec<(String, String)>,
}

impl ListQuery {
    pub(super) fn new(pairs: Vec<(String, String)>) -> Self {
        Self { pairs }
    }

    pub(super) fn get(&self, key: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Every value of `key`, as `key` or `key[]`.
    pub(super) fn all(&self, key: &str) -> Vec<&str> {
        let bracketed = format!("{key}[]");
        self.pairs
            .iter()
            .filter(|(k, _)| *k == key || *k == bracketed)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    pub(super) fn flag(&self, key: &str) -> bool {
        self.get(key) == Some("true")
    }
}

/// One page of `items`, in the `page` / `next_page` cursor scheme: the
/// cursor is the offset of the page's first item.
pub(super) fn cursor_page(items: Vec<Value>, query: &ListQuery, default_limit: usize) -> Reply {
    let limit = match query.get("limit") {
        Some(v) => v
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=1000).contains(n))
            .ok_or_else(|| bad("`limit` is a number from 1 to 1000"))?,
        None => default_limit,
    };
    let start = match query.get("page").filter(|p| !p.is_empty()) {
        Some(p) => p
            .strip_prefix("page_")
            .and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| bad(format!("'{p}' is not a page this server issued")))?,
        None => 0,
    };
    let end = (start + limit).min(items.len());
    let has_more = end < items.len();
    let data: Vec<Value> = items
        .into_iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect();
    Ok(json!({
        "data": data,
        "next_page": has_more.then(|| format!("page_{end}")),
    }))
}

/// A directory of JSON records, one file per id, each carrying an `_order`
/// field (its creation, in Unix nanoseconds) that lists newest first.
pub(super) struct Records {
    dir: PathBuf,
}

/// Serialises the read-modify-write of a record.
pub(super) fn records_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

impl Records {
    pub(super) fn new(state: &AppState, kind: &str) -> Self {
        Self {
            dir: state.api_data.join(kind),
        }
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    pub(super) fn get(&self, id: &str, prefix: &str) -> Option<Value> {
        if !valid_id(id, prefix) {
            return None;
        }
        let text = std::fs::read_to_string(self.path(id)).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub(super) fn put(&self, id: &str, record: &Value) -> Result<(), Box<Response>> {
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(&self.dir)?;
            let tmp = self.dir.join(format!(".{id}.tmp"));
            std::fs::write(&tmp, serde_json::to_vec(record).unwrap_or_default())?;
            std::fs::rename(tmp, self.path(id))
        };
        write().map_err(|err| {
            fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("saving '{id}': {err}"),
            )
        })
    }

    pub(super) fn remove(&self, id: &str) {
        let _ = std::fs::remove_file(self.path(id));
    }

    /// Every record, newest first.
    pub(super) fn all(&self) -> Vec<Value> {
        let mut out: Vec<Value> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.ends_with(".json") && !name.starts_with('.')
            })
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .filter_map(|text| serde_json::from_str(&text).ok())
            .collect();
        let order = |v: &Value| {
            v.get("_order")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        out.sort_by_key(|v| std::cmp::Reverse(order(v)));
        out
    }
}

/// A sortable creation stamp.
fn order_stamp() -> String {
    format!("{:039}", now_nanos())
}

/// `metadata` as given, checked: string values, at most 16 pairs. With
/// `base`, it updates those pairs: a `null` value removes one.
pub(super) fn metadata(given: Option<&Value>, base: Option<&Value>) -> Result<Value, String> {
    let mut out = base.and_then(Value::as_object).cloned().unwrap_or_default();
    match given {
        None | Some(Value::Null) => {}
        Some(Value::Object(pairs)) => {
            for (key, value) in pairs {
                if key.len() > 64 {
                    return Err(format!("the metadata key '{key}' is over 64 characters"));
                }
                match value {
                    Value::Null => {
                        out.remove(key);
                    }
                    Value::String(s) if s.len() <= 512 => {
                        out.insert(key.clone(), value.clone());
                    }
                    Value::String(_) => {
                        return Err(format!(
                            "the metadata value of '{key}' is over 512 characters"
                        ));
                    }
                    _ => return Err(format!("the metadata value of '{key}' must be a string")),
                }
            }
        }
        Some(_) => return Err("`metadata` must be an object of strings".into()),
    }
    if out.len() > 16 {
        return Err("`metadata` holds at most 16 pairs".into());
    }
    Ok(Value::Object(out))
}

fn optional_string(
    body: &Map<String, Value>,
    key: &str,
    max: usize,
) -> Result<Option<Value>, String> {
    match body.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(Value::Null)),
        Some(Value::String(s)) if s.chars().count() <= max => Ok(Some(json!(s))),
        Some(Value::String(_)) => Err(format!("`{key}` is over {max} characters")),
        Some(_) => Err(format!("`{key}` must be a string")),
    }
}

// --- Agents -----------------------------------------------------------------

const EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

/// `model` as an agent stores it: `{id, speed, effort?, inference_geo?}`.
pub(super) fn model(given: &Value) -> Result<Value, String> {
    let object = match given {
        Value::String(id) => return Ok(json!({"id": id, "speed": "standard"})),
        Value::Object(o) => o,
        _ => return Err("`model` is a model id, or an object with an `id`".into()),
    };
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .ok_or("`model.id` is missing")?;
    let speed = object
        .get("speed")
        .and_then(Value::as_str)
        .unwrap_or("standard");
    if !matches!(speed, "standard" | "fast") {
        return Err("`model.speed` is standard or fast".into());
    }
    let mut out = json!({"id": id, "speed": speed});
    if let Some(effort) = object.get("effort").filter(|e| !e.is_null()) {
        let level = effort
            .as_str()
            .or_else(|| effort.get("type").and_then(Value::as_str))
            .filter(|l| EFFORTS.contains(l))
            .ok_or_else(|| format!("`model.effort` is one of {}", EFFORTS.join(", ")))?;
        out["effort"] = json!({"type": level});
    }
    if let Some(geo) = object.get("inference_geo").and_then(Value::as_str) {
        out["inference_geo"] = json!(geo);
    }
    Ok(out)
}

/// `mcp_servers` checked: `{type: "url", name, url}`, names unique.
pub(super) fn mcp_servers(given: &Value) -> Result<Vec<Value>, String> {
    let list = given.as_array().ok_or("`mcp_servers` must be an array")?;
    if list.len() > 20 {
        return Err("an agent has at most 20 mcp_servers".into());
    }
    let mut out: Vec<Value> = Vec::new();
    for server in list {
        let name = server
            .get("name")
            .and_then(Value::as_str)
            .ok_or("an mcp_servers entry needs a `name`")?;
        let url = server
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("MCP server '{name}' needs a `url`"))?;
        if reqwest::Url::parse(url).is_err() {
            return Err(format!("MCP server '{name}' has no valid `url`"));
        }
        if out.iter().any(|s| s["name"] == name) {
            return Err(format!("two mcp_servers are named '{name}'"));
        }
        out.push(json!({"type": "url", "name": name, "url": url}));
    }
    Ok(out)
}

/// `tools` checked and filled in, against the agent's MCP servers.
pub(super) fn tools(given: &Value, servers: &[Value]) -> Result<Vec<Value>, String> {
    let list = given.as_array().ok_or("`tools` must be an array")?;
    if list.len() > 256 {
        return Err("an agent has at most 256 tools".into());
    }
    let mut out: Vec<Value> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for tool in list {
        match tool.get("type").and_then(Value::as_str) {
            Some(toolset::TOOLSET) => {
                if out.iter().any(|t| t["type"] == toolset::TOOLSET) {
                    return Err(format!("`tools` lists {} twice", toolset::TOOLSET));
                }
                out.push(toolset::normalize(tool)?);
            }
            Some("mcp_toolset") => {
                let normalized = toolset::normalize_mcp(tool, servers)?;
                if out.iter().any(|t| {
                    t["type"] == "mcp_toolset"
                        && t["mcp_server_name"] == normalized["mcp_server_name"]
                }) {
                    return Err(format!(
                        "two mcp_toolsets name '{}'",
                        normalized["mcp_server_name"].as_str().unwrap_or_default()
                    ));
                }
                out.push(normalized);
            }
            Some("custom") => {
                let name = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("a custom tool needs a `name`")?;
                if name.is_empty()
                    || name.len() > 64
                    || !name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                {
                    return Err(format!(
                        "the custom tool name '{name}' must be 1-64 letters, digits, _ or -"
                    ));
                }
                if name.starts_with("ant__") {
                    return Err(format!(
                        "the custom tool name '{name}' starts with `ant__`, which the server's \
                         own tools use"
                    ));
                }
                if toolset::TOOL_NAMES.contains(&name) || names.iter().any(|n| n == name) {
                    return Err(format!("the tool name '{name}' is already taken"));
                }
                names.push(name.to_string());
                let schema = tool
                    .get("input_schema")
                    .filter(|s| s.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
                out.push(json!({
                    "type": "custom",
                    "name": name,
                    "description": tool.get("description").and_then(Value::as_str).unwrap_or(""),
                    "input_schema": schema,
                }));
            }
            Some(other) => {
                return Err(format!(
                    "'{other}' is not a tool type: the types are {}, mcp_toolset and custom",
                    toolset::TOOLSET
                ));
            }
            None => return Err("a `tools` entry needs a `type`".into()),
        }
    }
    Ok(out)
}

/// `skills` checked against the Skills API: each a `custom` skill that
/// exists, at `latest` or the version it names.
pub(super) fn skills(state: &AppState, given: &Value) -> Result<Vec<Value>, String> {
    let list = given.as_array().ok_or("`skills` must be an array")?;
    let mut out = Vec::new();
    for entry in list {
        super::skills::resolve(&state.api_data, std::slice::from_ref(entry))?;
        let version = match entry.get("version") {
            Some(Value::String(v)) => v.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => "latest".to_string(),
        };
        out.push(json!({
            "type": "custom",
            "skill_id": entry.get("skill_id").and_then(Value::as_str).unwrap_or_default(),
            "version": version,
        }));
    }
    super::skills::resolve(&state.api_data, &out)?;
    Ok(out)
}

/// The configuration fields of an agent: those `body` sets, over `base`
/// (the current version) on an update.
fn configure(
    state: &AppState,
    body: &Map<String, Value>,
    base: Option<&Value>,
) -> Result<Map<String, Value>, String> {
    let mut out = base.and_then(Value::as_object).cloned().unwrap_or_default();
    for key in [
        "id",
        "type",
        "version",
        "created_at",
        "updated_at",
        "archived_at",
    ] {
        out.remove(key);
    }
    match body.get("name") {
        Some(Value::String(name)) if (1..=256).contains(&name.chars().count()) => {
            out.insert("name".into(), json!(name));
        }
        Some(_) => return Err("`name` is a string of 1 to 256 characters".into()),
        None if base.is_none() => return Err("`name` is required".into()),
        None => {}
    }
    match body.get("model") {
        Some(given) => {
            out.insert("model".into(), model(given)?);
        }
        None if base.is_none() => return Err("`model` is required".into()),
        None => {}
    }
    for (key, max) in [("description", 2048), ("system", 100_000)] {
        if let Some(value) = optional_string(body, key, max)? {
            out.insert(key.into(), value);
        } else if base.is_none() {
            out.insert(key.into(), Value::Null);
        }
    }
    let multiagent_block = match body.get("multiagent") {
        Some(Value::Null) => Value::Null,
        Some(given) => multiagent(
            state,
            given,
            out.get("multiagent"),
            base.and_then(|b| b["id"].as_str()),
        )?,
        None => out.get("multiagent").cloned().unwrap_or(Value::Null),
    };
    let servers = match body.get("mcp_servers") {
        Some(Value::Null) => Vec::new(),
        Some(given) => mcp_servers(given)?,
        None => out
            .get("mcp_servers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    };
    let tool_list = match body.get("tools") {
        Some(Value::Null) => Vec::new(),
        Some(given) => tools(given, &servers)?,
        None => {
            let current = out.get("tools").cloned().unwrap_or_else(|| json!([]));
            tools(&current, &servers)?
        }
    };
    let skill_list = match body.get("skills") {
        Some(Value::Null) => Vec::new(),
        Some(given) => skills(state, given)?,
        None => out
            .get("skills")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    };
    out.insert("mcp_servers".into(), json!(servers));
    out.insert("tools".into(), json!(tool_list));
    out.insert("skills".into(), json!(skill_list));
    out.insert(
        "metadata".into(),
        metadata(body.get("metadata"), out.get("metadata"))?,
    );
    out.insert("multiagent".into(), multiagent_block);
    out.insert(
        "execution_identity".into(),
        json!({"type": "service_account"}),
    );
    Ok(out)
}

/// The most agents a roster lists.
const MAX_ROSTER: usize = 20;

/// A roster's entries resolved: a stored agent at a version — the latest
/// when none is named — or `self`, which the agent's own id, `own`, names
/// too; no entry may itself delegate. With `advisor`, one
/// `{"type": "advisor", "model": …}` entry may be among them.
fn roster(
    state: &AppState,
    given: Option<&Value>,
    advisor: bool,
    own: Option<&str>,
) -> Result<Vec<Value>, String> {
    let entries = match given {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(entries)) => entries,
        Some(_) => return Err("`predefined_agents` is a list of agents".into()),
    };
    if entries.len() > MAX_ROSTER {
        return Err(format!("a roster lists at most {MAX_ROSTER} agents"));
    }
    let mut out = Vec::new();
    for entry in entries {
        let (id, version) = match entry {
            Value::String(id) => (id.as_str(), None),
            Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("self") => {
                out.push(json!({"type": "self"}));
                continue;
            }
            Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("advisor") => {
                if !advisor {
                    return Err(
                        "the advisor is the multiagent block's `advisor` member, not a \
                                roster entry"
                            .into(),
                    );
                }
                if out.iter().any(|e| e["type"] == "advisor") {
                    return Err("a coordinator lists at most one advisor".into());
                }
                let model = o
                    .get("model")
                    .and_then(Value::as_str)
                    .ok_or("an advisor entry names its `model`")?;
                out.push(json!({"type": "advisor", "model": model}));
                continue;
            }
            Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("agent") => (
                o.get("id")
                    .and_then(Value::as_str)
                    .ok_or("a roster entry {\"type\": \"agent\"} needs its `id`")?,
                o.get("version").and_then(Value::as_u64),
            ),
            _ => {
                return Err(
                    "a roster entry is an agent id, {\"type\": \"agent\", \"id\": …}, \
                            or {\"type\": \"self\"}"
                        .into(),
                );
            }
        };
        if own == Some(id) {
            out.push(json!({"type": "self"}));
            continue;
        }
        let agent = agent_at(state, id, version)
            .ok_or_else(|| format!("the roster's agent '{id}' does not exist"))?;
        if !agent["archived_at"].is_null() {
            return Err(format!("the roster's agent '{id}' is archived"));
        }
        if !agent["multiagent"].is_null() {
            return Err(format!(
                "the roster's agent '{id}' delegates to agents of its own; only one level of \
                 delegation is allowed"
            ));
        }
        out.push(json!({"type": "agent", "id": id, "version": agent["version"]}));
    }
    Ok(out)
}

/// A member of the multiagent block that is enabled or disabled, with its
/// defaults when left out.
fn member(given: Option<&Value>, default_enabled: bool) -> Result<Value, String> {
    match given {
        None | Some(Value::Null) => {
            Ok(json!({"type": if default_enabled { "enabled" } else { "disabled" }}))
        }
        Some(v) if matches!(v["type"].as_str(), Some("enabled" | "disabled")) => Ok(v.clone()),
        Some(_) => Err("a multiagent member's `type` is enabled or disabled".into()),
    }
}

/// An update's `multiagent_20261001` block merged into the stored one,
/// level by level: a key left out keeps its stored value, a `null` one
/// takes its default, and a member whose `type` changes is replaced whole.
fn merged(given: &Value, stored: &Value) -> Value {
    let mut out = stored.clone();
    for member in ["subagents", "workflows", "advisor"] {
        match given.get(member) {
            None => {}
            Some(Value::Null) => out[member] = Value::Null,
            Some(sent) if sent["type"] != stored[member]["type"] => out[member] = sent.clone(),
            Some(sent) => {
                for (key, value) in sent.as_object().into_iter().flatten() {
                    out[member][key] = value.clone();
                }
            }
        }
    }
    out
}

/// `multiagent` checked: `multiagent_20261001` with its `subagents`,
/// `workflows` and `advisor`, or the older `coordinator` with `agents`.
/// `previous` is the agent's block before an update: an update of a
/// `multiagent_20261001` block is merged into it.
fn multiagent(
    state: &AppState,
    given: &Value,
    previous: Option<&Value>,
    own: Option<&str>,
) -> Result<Value, String> {
    let merged_block;
    let given = match previous {
        Some(stored)
            if given["type"] == "multiagent_20261001" && stored["type"] == given["type"] =>
        {
            merged_block = merged(given, stored);
            &merged_block
        }
        _ => given,
    };
    let block = match given["type"].as_str() {
        Some("coordinator") => {
            let agents = roster(state, given.get("agents"), true, own)?;
            if agents.is_empty() {
                return Err("a coordinator lists 1 to 20 `agents`".into());
            }
            json!({"type": "coordinator", "agents": agents})
        }
        Some("multiagent_20261001") => {
            let mut subagents = member(given.get("subagents"), true)?;
            if subagents["type"] == "enabled" {
                let inline = member(subagents.get("inline_agents"), true)?;
                let listed = roster(state, subagents.get("predefined_agents"), false, own)?;
                if inline["type"] == "disabled" && listed.is_empty() {
                    return Err("subagents with inline_agents disabled need at least one \
                                predefined agent"
                        .into());
                }
                subagents = json!({"type": "enabled", "inline_agents": inline,
                                   "predefined_agents": listed});
            }
            let mut workflows = member(given.get("workflows"), true)?;
            if workflows["type"] == "enabled" {
                let inline = member(workflows.get("inline_agents"), true)?;
                let listed = roster(state, workflows.get("predefined_agents"), false, own)?;
                if inline["type"] == "disabled" && listed.is_empty() {
                    return Err("workflows with inline_agents disabled need at least one \
                                predefined agent"
                        .into());
                }
                workflows = json!({"type": "enabled", "inline_agents": inline,
                                   "predefined_agents": listed});
            }
            let advisor = member(given.get("advisor"), false)?;
            let advisor = match advisor["type"].as_str() {
                Some("enabled") => {
                    let model = advisor["model"]
                        .as_str()
                        .ok_or("an enabled advisor names its `model`")?;
                    json!({"type": "enabled", "model": model})
                }
                _ => json!({"type": "disabled"}),
            };
            json!({"type": "multiagent_20261001", "subagents": subagents,
                   "workflows": workflows, "advisor": advisor})
        }
        _ => return Err("`multiagent.type` is multiagent_20261001 or coordinator".into()),
    };
    if advisor_model(&block).is_some() {
        let listed = block["agents"]
            .as_array()
            .into_iter()
            .chain(block["subagents"]["predefined_agents"].as_array())
            .flatten();
        for entry in listed {
            let named = entry["id"]
                .as_str()
                .and_then(|id| agent_at(state, id, entry["version"].as_u64()))
                .is_some_and(|a| a["name"] == super::advisor::NAME);
            if named {
                return Err(format!(
                    "with an advisor, `{}` is the advisor's name: no listed agent may have it",
                    super::advisor::NAME
                ));
            }
        }
    }
    Ok(block)
}

/// The model an agent's `multiagent` block gives it as its advisor.
pub(super) fn advisor_model(block: &Value) -> Option<&str> {
    match block["type"].as_str() {
        Some("multiagent_20261001") if block["advisor"]["type"] == "enabled" => {
            block["advisor"]["model"].as_str()
        }
        Some("coordinator") => block["agents"]
            .as_array()?
            .iter()
            .find(|e| e["type"] == "advisor")?["model"]
            .as_str(),
        _ => None,
    }
}

fn agent_lookup(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    Records::new(state, "agents")
        .get(id, "agent_")
        .ok_or_else(|| missing("agent", id))
}

/// One version of an agent's record, as the API shows it.
fn agent_version(record: &Value, version: Option<u64>) -> Option<Value> {
    let versions = record.get("versions")?.as_array()?;
    let mut agent = match version {
        Some(v) => versions
            .iter()
            .find(|a| a.get("version").and_then(Value::as_u64) == Some(v))?
            .clone(),
        None => versions.last()?.clone(),
    };
    agent["archived_at"] = record.get("archived_at").cloned().unwrap_or(Value::Null);
    Some(agent)
}

/// An agent as a response shows it: a `self` entry of its multiagent block
/// as a reference to its own id and version.
fn shown(mut agent: Value) -> Value {
    let own = json!({"type": "agent", "id": agent["id"], "version": agent["version"]});
    for path in [
        "/multiagent/agents",
        "/multiagent/subagents/predefined_agents",
        "/multiagent/workflows/predefined_agents",
    ] {
        let entries = agent.pointer_mut(path).and_then(Value::as_array_mut);
        for entry in entries.into_iter().flatten() {
            if entry["type"] == "self" {
                *entry = own.clone();
            }
        }
    }
    agent
}

/// The agent a session names, at the version it names: `None` when
/// either does not exist.
pub(super) fn agent_at(state: &AppState, id: &str, version: Option<u64>) -> Option<Value> {
    let record = Records::new(state, "agents").get(id, "agent_")?;
    agent_version(&record, version)
}

/// The configuration fields a session may override on its agent, checked.
pub(super) fn override_agent(
    state: &AppState,
    agent: &Value,
    overrides: &Map<String, Value>,
) -> Result<Value, String> {
    let mut body = Map::new();
    for key in ["model", "system", "tools", "mcp_servers", "skills"] {
        if let Some(value) = overrides.get(key) {
            body.insert(key.into(), value.clone());
        }
    }
    let mut out = configure(state, &body, Some(agent))?;
    for key in ["id", "version"] {
        out.insert(key.into(), agent[key].clone());
    }
    out.insert("type".into(), json!("agent"));
    Ok(Value::Object(out))
}

pub async fn create_agent(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let config = configure(&state, &body, None).map_err(bad)?;
        let id = new_id("agent_");
        let now = rfc3339(unix_now());
        let mut agent = Value::Object(config);
        agent["id"] = json!(id);
        agent["type"] = json!("agent");
        agent["version"] = json!(1);
        agent["created_at"] = json!(now);
        agent["updated_at"] = json!(now);
        let record = json!({"_order": order_stamp(), "archived_at": null, "versions": [agent]});
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        Records::new(&state, "agents").put(&id, &record)?;
        super::webhooks::emit(&state, "agent.created", &id);
        Ok(shown(agent_version(&record, None).unwrap_or_default()))
    })())
}

pub async fn list_agents(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    let archived = query.flag("include_archived");
    let items = Records::new(&state, "agents")
        .all()
        .into_iter()
        .filter(|r| archived || r["archived_at"].is_null())
        .filter_map(|r| agent_version(&r, None))
        .map(shown)
        .collect();
    respond(cursor_page(items, &query, 20))
}

pub async fn retrieve_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    respond((|| {
        let record = agent_lookup(&state, &id)?;
        let version = match query.get("version") {
            Some(v) => Some(
                v.parse::<u64>()
                    .map_err(|_| bad("`version` is a whole number"))?,
            ),
            None => None,
        };
        agent_version(&record, version)
            .map(shown)
            .ok_or_else(|| missing("agent version", &format!("{id} v{}", version.unwrap_or(0))))
    })())
}

pub async fn update_agent(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut record = agent_lookup(&state, &id)?;
        if !record["archived_at"].is_null() {
            return Err(fail(
                StatusCode::CONFLICT,
                format!("agent '{id}' is archived, and an archived agent cannot change"),
            ));
        }
        let current = agent_version(&record, None).unwrap_or_default();
        let latest = current["version"].as_u64().unwrap_or(1);
        if let Some(expected) = body.get("version").filter(|v| !v.is_null())
            && expected.as_u64() != Some(latest)
        {
            return Err(fail(
                StatusCode::CONFLICT,
                format!("agent '{id}' is at version {latest}, not {expected}"),
            ));
        }
        let config = configure(&state, &body, Some(&current)).map_err(bad)?;
        let unchanged = config.iter().all(|(k, v)| current.get(k) == Some(v));
        if unchanged {
            return Ok(shown(current));
        }
        let mut agent = Value::Object(config);
        agent["id"] = json!(id);
        agent["type"] = json!("agent");
        agent["version"] = json!(latest + 1);
        agent["created_at"] = current["created_at"].clone();
        agent["updated_at"] = json!(rfc3339(unix_now()));
        if let Some(versions) = record["versions"].as_array_mut() {
            versions.push(agent);
        }
        Records::new(&state, "agents").put(&id, &record)?;
        super::webhooks::emit(&state, "agent.updated", &id);
        Ok(shown(agent_version(&record, None).unwrap_or_default()))
    })())
}

pub async fn archive_agent(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    respond((|| {
        let (record, archived) = {
            let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
            let mut record = agent_lookup(&state, &id)?;
            let archiving = record["archived_at"].is_null();
            if archiving {
                record["archived_at"] = json!(rfc3339(unix_now()));
                Records::new(&state, "agents").put(&id, &record)?;
            }
            (record, archiving)
        };
        if archived {
            super::webhooks::emit(&state, "agent.archived", &id);
            // Its deployments can no longer run.
            super::deployments::agent_archived(&state, &id);
        }
        Ok(shown(agent_version(&record, None).unwrap_or_default()))
    })())
}

pub async fn list_agent_versions(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    respond((|| {
        let record = agent_lookup(&state, &id)?;
        let mut items: Vec<Value> = record["versions"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|a| agent_version(&record, a["version"].as_u64()))
            .map(shown)
            .collect();
        items.reverse();
        cursor_page(items, &query, 20)
    })())
}

// --- Environments -----------------------------------------------------------

/// An environment's `config`: a `cloud` container, its `networking` and
/// `packages` checked. With no `networking` it is `limited`, which gives
/// the sandbox no network at all.
fn environment_config(given: Option<&Value>) -> Result<Value, String> {
    let given = given.filter(|v| !v.is_null());
    match given
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("cloud")
    {
        "cloud" => {}
        "self_hosted" => {
            return Err(
                "self-hosted environments are not supported here: sessions run their \
                        tools in this server's own sandbox"
                    .into(),
            );
        }
        other => return Err(format!("'{other}' is not an environment type: use cloud")),
    }
    let networking = given
        .and_then(|c| c.get("networking"))
        .filter(|v| !v.is_null());
    let networking = match networking
        .and_then(|n| n.get("type"))
        .and_then(Value::as_str)
    {
        Some("unrestricted") => json!({"type": "unrestricted"}),
        None | Some("limited") => {
            let hosts: Vec<String> = networking
                .and_then(|n| n.get("allowed_hosts"))
                .and_then(Value::as_array)
                .map(|hosts| {
                    hosts
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_ascii_lowercase)
                        .collect()
                })
                .unwrap_or_default();
            let flag = |key: &str| {
                networking
                    .and_then(|n| n.get(key))
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            };
            json!({
                "type": "limited",
                "allowed_hosts": hosts,
                "allow_mcp_servers": flag("allow_mcp_servers"),
                "allow_package_managers": flag("allow_package_managers"),
            })
        }
        Some(other) => {
            return Err(format!(
                "'{other}' is not a networking type: use unrestricted or limited"
            ));
        }
    };
    let mut packages = json!({"type": "packages"});
    for manager in ["apt", "cargo", "gem", "go", "npm", "pip"] {
        let listed = given
            .and_then(|c| c.get("packages"))
            .and_then(|p| p.get(manager))
            .and_then(Value::as_array)
            .is_some_and(|l| !l.is_empty());
        if listed {
            return Err(format!(
                "packages are not installed here: the sandbox runs the server's own programs, \
                 so install {manager} packages on the server instead"
            ));
        }
        packages[manager] = json!([]);
    }
    Ok(json!({"type": "cloud", "networking": networking, "packages": packages}))
}

/// Whether `host` is one a `limited` environment's `allowed_hosts` lets
/// through: a bare name matches itself, `*.name` every subdomain of it.
pub(super) fn host_allowed(allowed: &[Value], host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    allowed
        .iter()
        .filter_map(Value::as_str)
        .any(|pattern| match pattern.strip_prefix("*.") {
            Some(domain) => host.ends_with(&format!(".{domain}")),
            None => host == pattern,
        })
}

fn environment_lookup(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    Records::new(state, "environments")
        .get(id, "env_")
        .ok_or_else(|| missing("environment", id))
}

/// An environment's record as the API shows it.
fn environment_json(record: &Value) -> Value {
    let mut out = record.clone();
    if let Some(map) = out.as_object_mut() {
        map.remove("_order");
    }
    out
}

/// The environment a session names: `None` when it does not exist.
pub(super) fn environment(state: &AppState, id: &str) -> Option<Value> {
    Records::new(state, "environments")
        .get(id, "env_")
        .map(|r| environment_json(&r))
}

fn configure_environment(body: &Map<String, Value>, base: Option<&Value>) -> Result<Value, String> {
    let mut out = base.cloned().unwrap_or_else(|| json!({}));
    match body.get("name") {
        Some(Value::String(name)) if !name.is_empty() => out["name"] = json!(name),
        Some(_) => return Err("`name` must be a non-empty string".into()),
        None if base.is_none() => return Err("`name` is required".into()),
        None => {}
    }
    if let Some(value) = optional_string(body, "description", 2048)? {
        out["description"] = value;
    } else if base.is_none() {
        out["description"] = Value::Null;
    }
    if body.contains_key("config") || base.is_none() {
        out["config"] = environment_config(body.get("config"))?;
    }
    out["metadata"] = metadata(body.get("metadata"), base.and_then(|b| b.get("metadata")))?;
    if base.is_none() {
        out["scope"] = body.get("scope").cloned().unwrap_or(json!("organization"));
    }
    Ok(out)
}

pub async fn create_environment(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let mut record = configure_environment(&body, None).map_err(bad)?;
        let id = new_id("env_");
        let now = rfc3339(unix_now());
        record["id"] = json!(id);
        record["type"] = json!("environment");
        record["created_at"] = json!(now);
        record["updated_at"] = json!(now);
        record["archived_at"] = Value::Null;
        record["_order"] = json!(order_stamp());
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        Records::new(&state, "environments").put(&id, &record)?;
        super::webhooks::emit(&state, "environment.created", &id);
        Ok(environment_json(&record))
    })())
}

pub async fn list_environments(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    let archived = query.flag("include_archived");
    let items = Records::new(&state, "environments")
        .all()
        .iter()
        .filter(|r| archived || r["archived_at"].is_null())
        .map(environment_json)
        .collect();
    respond(cursor_page(items, &query, 20))
}

pub async fn retrieve_environment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(environment_lookup(&state, &id).map(|r| environment_json(&r)))
}

pub async fn update_environment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let current = environment_lookup(&state, &id)?;
        if !current["archived_at"].is_null() {
            return Err(fail(
                StatusCode::CONFLICT,
                format!(
                    "environment '{id}' is archived, and an archived environment cannot change"
                ),
            ));
        }
        let mut record = configure_environment(&body, Some(&current)).map_err(bad)?;
        if record == current {
            return Ok(environment_json(&record));
        }
        record["updated_at"] = json!(rfc3339(unix_now()));
        Records::new(&state, "environments").put(&id, &record)?;
        super::webhooks::emit(&state, "environment.updated", &id);
        Ok(environment_json(&record))
    })())
}

pub async fn delete_environment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        environment_lookup(&state, &id)?;
        Records::new(&state, "environments").remove(&id);
        super::webhooks::emit(&state, "environment.deleted", &id);
        Ok(json!({"id": id, "type": "environment_deleted"}))
    })())
}

pub async fn archive_environment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut record = environment_lookup(&state, &id)?;
        if record["archived_at"].is_null() {
            record["archived_at"] = json!(rfc3339(unix_now()));
            Records::new(&state, "environments").put(&id, &record)?;
            super::webhooks::emit(&state, "environment.archived", &id);
        }
        Ok(environment_json(&record))
    })())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_model_id_gets_standard_speed_and_effort_is_an_object() {
        assert_eq!(
            model(&json!("claude-opus-5-5")).unwrap(),
            json!({"id": "claude-opus-5-5", "speed": "standard"})
        );
        assert_eq!(
            model(&json!({"id": "m", "effort": "high"})).unwrap()["effort"],
            json!({"type": "high"})
        );
        assert!(model(&json!({"id": "m", "effort": "huge"})).is_err());
    }

    #[test]
    fn custom_tools_may_not_take_a_toolset_name_or_repeat() {
        let servers = [];
        let custom = |name: &str| json!({"type": "custom", "name": name, "description": "d"});
        assert!(tools(&json!([custom("bash")]), &servers).is_err());
        assert!(tools(&json!([custom("a"), custom("a")]), &servers).is_err());
        assert!(tools(&json!([custom("get_weather")]), &servers).is_ok());
        assert!(tools(&json!([custom("ant__mine")]), &servers).is_err());
    }

    /// An update keeps what it leaves out, resets what it sends as null,
    /// and replaces a member whose type it changes.
    #[test]
    fn a_multiagent_update_merges_level_by_level() {
        let stored = json!({"type": "multiagent_20261001",
            "subagents": {"type": "enabled", "inline_agents": {"type": "disabled"},
                          "predefined_agents": [{"type": "self"}]},
            "workflows": {"type": "enabled", "inline_agents": {"type": "enabled"},
                          "predefined_agents": []},
            "advisor": {"type": "enabled", "model": "m"}});
        let out = merged(
            &json!({"type": "multiagent_20261001",
                    "subagents": {"type": "enabled", "inline_agents": null},
                    "workflows": {"type": "disabled"}}),
            &stored,
        );
        assert_eq!(out["subagents"]["inline_agents"], Value::Null);
        assert_eq!(
            out["subagents"]["predefined_agents"],
            json!([{"type": "self"}])
        );
        assert_eq!(out["workflows"], json!({"type": "disabled"}));
        assert_eq!(out["advisor"], stored["advisor"]);
        assert_eq!(member(out.get("advisor"), false).unwrap()["model"], "m");
        assert_eq!(
            member(out["subagents"].get("inline_agents"), true).unwrap()["type"],
            "enabled"
        );
    }

    #[test]
    fn an_environment_without_networking_is_limited_and_packages_are_refused() {
        let config = environment_config(None).unwrap();
        assert_eq!(config["networking"]["type"], "limited");
        assert_eq!(config["packages"]["pip"], json!([]));
        assert!(environment_config(Some(&json!({"packages": {"pip": ["pandas"]}}))).is_err());
        assert!(environment_config(Some(&json!({"type": "self_hosted"}))).is_err());
    }

    #[test]
    fn allowed_hosts_match_a_name_or_its_subdomains() {
        let allowed = [json!("example.com"), json!("*.docs.io")];
        assert!(host_allowed(&allowed, "example.com"));
        assert!(!host_allowed(&allowed, "www.example.com"));
        assert!(host_allowed(&allowed, "api.docs.io"));
        assert!(!host_allowed(&allowed, "docs.io"));
    }

    #[test]
    fn metadata_updates_merge_and_null_removes() {
        let base = json!({"a": "1", "b": "2"});
        let out = metadata(Some(&json!({"b": null, "c": "3"})), Some(&base)).unwrap();
        assert_eq!(out, json!({"a": "1", "c": "3"}));
    }

    #[test]
    fn pages_follow_their_cursor() {
        let items: Vec<Value> = (0..5).map(|n| json!(n)).collect();
        let q = ListQuery::new(vec![("limit".into(), "2".into())]);
        let first = cursor_page(items.clone(), &q, 20).unwrap();
        assert_eq!(first["data"], json!([0, 1]));
        assert_eq!(first["next_page"], "page_2");
        let q = ListQuery::new(vec![
            ("limit".into(), "2".into()),
            ("page".into(), "page_4".into()),
        ]);
        let last = cursor_page(items, &q, 20).unwrap();
        assert_eq!(last["data"], json!([4]));
        assert!(last["next_page"].is_null());
    }
}
