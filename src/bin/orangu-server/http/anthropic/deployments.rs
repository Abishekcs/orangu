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

//! Scheduled deployments (`/v1/deployments`): an agent, its environment
//! and the events that start its work, run as a new session on a cron
//! schedule. Every attempt — scheduled or manual, successful or not — is a
//! deployment run (`/v1/deployment_runs`), the session it made or the
//! error that stopped it.
//!
//! A deployment is a record under `<api_data>/deployments/`, a run one
//! under `<api_data>/deployment_runs/`. [`start_scheduler`] checks the
//! active deployments at every minute and fires those that are due.

use axum::{
    Json,
    extract::{Path, Query, State, rejection::JsonRejection},
    response::Response,
};
use chrono::{Datelike, TimeZone, Timelike};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::Duration;

use super::agents::{
    ListQuery, Records, Reply, agent_at, bad, body, cursor_page, environment, metadata, missing,
    records_lock, respond,
};
use super::files::{new_id, now_nanos};
use super::webhooks;
use super::{rfc3339, unix_now};
use crate::http::AppState;

/// The most deployments the server holds, archived ones aside.
const MAX_DEPLOYMENTS: usize = 1000;

/// How many fire times `schedule.upcoming_runs_at` lists.
const UPCOMING: usize = 5;

/// How far ahead a schedule is searched for its next fire time: one that
/// never fires in this long is refused.
const SEARCH_MINUTES: i64 = 5 * 366 * 24 * 60;

/// A five-field POSIX cron expression: minute, hour, day of month, month
/// and day of week.
#[derive(Debug, PartialEq)]
pub(super) struct Cron {
    minutes: u64,
    hours: u32,
    days: u32,
    months: u16,
    weekdays: u8,
    /// Whether the day of month, and the day of week, are `*`: when both
    /// are restricted, a day matching either fires.
    any_day: bool,
    any_weekday: bool,
}

const MONTHS: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];
const WEEKDAYS: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];

/// One field's values as a bit set: `*`, `n`, `a-b`, any of them with
/// `/step`, and lists of these joined by commas.
fn field(text: &str, low: u32, high: u32, names: &[&str]) -> Result<u64, String> {
    let value = |part: &str| -> Result<u32, String> {
        let lower = part.to_ascii_lowercase();
        if let Some(at) = names.iter().position(|n| *n == lower) {
            return Ok(low + at as u32);
        }
        part.parse::<u32>()
            .map_err(|_| format!("'{part}' is not a number"))
    };
    let mut bits = 0u64;
    for item in text.split(',') {
        let (range, step) = match item.split_once('/') {
            Some((range, step)) => (
                range,
                step.parse::<u32>()
                    .ok()
                    .filter(|s| *s > 0)
                    .ok_or_else(|| format!("'{step}' is not a step"))?,
            ),
            None => (item, 1),
        };
        let (from, to) = if range == "*" {
            (low, high)
        } else if let Some((a, b)) = range.split_once('-') {
            (value(a)?, value(b)?)
        } else {
            let at = value(range)?;
            // `n/step` runs from n to the field's end.
            (at, if item.contains('/') { high } else { at })
        };
        if from < low || to > high || from > to {
            return Err(format!("'{item}' is outside {low}-{high}"));
        }
        let mut at = from;
        while at <= to {
            bits |= 1 << at;
            at += step;
        }
    }
    Ok(bits)
}

impl Cron {
    pub(super) fn parse(expression: &str) -> Result<Self, String> {
        let fields: Vec<&str> = expression.split_whitespace().collect();
        let [minute, hour, day, month, weekday] = fields[..] else {
            return Err(
                "a cron expression has five fields: minute hour day-of-month month \
                        day-of-week"
                    .into(),
            );
        };
        let named = |what: &str, r: Result<u64, String>| r.map_err(|e| format!("{what}: {e}"));
        // Day of week 7 is Sunday too.
        let mut weekdays = named("day of week", field(weekday, 0, 7, &WEEKDAYS))?;
        if weekdays & (1 << 7) != 0 {
            weekdays = (weekdays & 0x7f) | 1;
        }
        Ok(Self {
            minutes: named("minute", field(minute, 0, 59, &[]))?,
            hours: named("hour", field(hour, 0, 23, &[]))? as u32,
            days: named("day of month", field(day, 1, 31, &[]))? as u32,
            months: named("month", field(month, 1, 12, &MONTHS))? as u16,
            weekdays: weekdays as u8,
            any_day: day == "*",
            any_weekday: weekday == "*",
        })
    }

    /// Whether the wall-clock minute `at` fires.
    fn matches(&self, at: &impl Datelike, hour: u32, minute: u32) -> bool {
        let day = self.days & (1 << at.day()) != 0;
        let weekday = self.weekdays & (1 << at.weekday().num_days_from_sunday()) != 0;
        let day = match (self.any_day, self.any_weekday) {
            (false, false) => day || weekday,
            _ => day && weekday,
        };
        self.minutes & (1 << minute) != 0
            && self.hours & (1 << hour) != 0
            && self.months & (1 << at.month()) != 0
            && day
    }

    /// The fire times after Unix second `after`, in Unix seconds, at most
    /// `count` of them: each UTC minute whose wall-clock time in `zone`
    /// matches. A wall-clock time a change of offset skips never fires; one
    /// it repeats fires each time it occurs.
    pub(super) fn upcoming(&self, zone: chrono_tz::Tz, after: i64, count: usize) -> Vec<i64> {
        let mut out = Vec::new();
        let mut minute = after.div_euclid(60) + 1;
        let end = minute + SEARCH_MINUTES;
        while out.len() < count && minute < end {
            let at = minute * 60;
            if let Some(utc) = chrono::DateTime::from_timestamp(at, 0) {
                let local = zone.from_utc_datetime(&utc.naive_utc());
                if self.matches(&local, local.hour(), local.minute()) {
                    out.push(at);
                }
            }
            minute += 1;
        }
        out
    }
}

/// `schedule` checked: `{type: "cron", expression, timezone}`, firing at
/// least once in the time searched.
fn schedule(given: Option<&Value>) -> Result<Value, String> {
    let given = given
        .filter(|s| s.is_object())
        .ok_or("`schedule` is {\"type\": \"cron\", \"expression\": …, \"timezone\": …}")?;
    if given["type"] != "cron" {
        return Err("`schedule.type` is cron".into());
    }
    let expression = given["expression"]
        .as_str()
        .ok_or("`schedule.expression` is a cron expression")?;
    let cron = Cron::parse(expression).map_err(|e| format!("`schedule.expression`: {e}"))?;
    let zone_name = given["timezone"].as_str().unwrap_or("UTC");
    let zone: chrono_tz::Tz = zone_name
        .parse()
        .map_err(|_| format!("'{zone_name}' is not an IANA time zone"))?;
    if cron.upcoming(zone, unix_now() as i64, 1).is_empty() {
        return Err(format!("the schedule '{expression}' never fires"));
    }
    Ok(json!({"type": "cron", "expression": expression, "timezone": zone_name}))
}

/// A schedule's next fire times after `after`.
fn upcoming(schedule: &Value, after: i64, count: usize) -> Vec<i64> {
    let zone = schedule["timezone"]
        .as_str()
        .and_then(|z| z.parse::<chrono_tz::Tz>().ok());
    match (schedule["expression"].as_str().map(Cron::parse), zone) {
        (Some(Ok(cron)), Some(zone)) => cron.upcoming(zone, after, count),
        _ => Vec::new(),
    }
}

fn deployments(state: &AppState) -> Records {
    Records::new(state, "deployments")
}

fn runs(state: &AppState) -> Records {
    Records::new(state, "deployment_runs")
}

/// A deployment as the API shows it, its next fire times listed — what
/// the schedule would do if it were not paused; none once archived.
fn deployment_json(record: &Value) -> Value {
    let mut out = record.clone();
    if let Some(map) = out.as_object_mut() {
        map.retain(|key, _| !key.starts_with('_'));
    }
    let listed = if record["archived_at"].is_null() {
        upcoming(&record["schedule"], unix_now() as i64, UPCOMING)
    } else {
        Vec::new()
    };
    out["schedule"]["upcoming_runs_at"] = json!(
        listed
            .into_iter()
            .map(|at| rfc3339(at as u64))
            .collect::<Vec<_>>()
    );
    out
}

/// The agent a deployment fires with, pinned: `{type: "agent", id,
/// version}` — the latest version when an id alone is given — its agent
/// existing and not archived.
fn agent_reference(state: &AppState, given: Option<&Value>) -> Result<Value, String> {
    let (id, version) = match given {
        Some(Value::String(id)) => (id.as_str(), None),
        Some(Value::Object(o)) if o.get("type").and_then(Value::as_str) != Some("self") => (
            o.get("id")
                .and_then(Value::as_str)
                .ok_or("`agent.id` is missing")?,
            o.get("version").and_then(Value::as_u64),
        ),
        _ => return Err("`agent` is an agent id or {\"type\": \"agent\", \"id\": …}".into()),
    };
    let agent =
        agent_at(state, id, version).ok_or_else(|| format!("agent '{id}' does not exist"))?;
    if !agent["archived_at"].is_null() {
        return Err(format!("agent '{id}' is archived"));
    }
    Ok(json!({"type": "agent", "id": id, "version": agent["version"]}))
}

/// `initial_events` checked: at least one `user.message` or
/// `user.define_outcome`; `system.message` events may go with them.
fn initial_events(given: Option<&Value>) -> Result<Vec<Value>, String> {
    let events = given
        .and_then(Value::as_array)
        .ok_or("`initial_events` lists the events each run starts with")?;
    if events.len() > 50 {
        return Err("initial_events holds at most 50 events".into());
    }
    for event in events {
        let kind = event["type"].as_str().unwrap_or_default();
        if !matches!(
            kind,
            "user.message" | "user.define_outcome" | "system.message"
        ) {
            return Err(format!(
                "a deployment's initial_events take user.message, user.define_outcome and \
                 system.message events, not '{kind}'"
            ));
        }
    }
    if !events.iter().any(|e| {
        matches!(
            e["type"].as_str(),
            Some("user.message" | "user.define_outcome")
        )
    }) {
        return Err("initial_events needs a user.message or a user.define_outcome".into());
    }
    Ok(events.clone())
}

/// The configuration fields of a create or update `body`, checked into
/// `record`.
fn configure(
    state: &AppState,
    body: &Map<String, Value>,
    record: &mut Value,
) -> Result<(), String> {
    let creating = record["id"].is_null();
    match body.get("name").and_then(Value::as_str) {
        Some(name) if !name.trim().is_empty() => record["name"] = json!(name),
        Some(_) => return Err("`name` must be a non-empty string".into()),
        None if creating => return Err("`name` is required".into()),
        None => {}
    }
    if let Some(description) = body.get("description") {
        record["description"] = description.clone();
    } else if creating {
        record["description"] = Value::Null;
    }
    if body.contains_key("agent") || creating {
        record["agent"] = agent_reference(state, body.get("agent"))?;
    }
    if body.contains_key("environment_id") || creating {
        let id = body
            .get("environment_id")
            .and_then(Value::as_str)
            .ok_or("`environment_id` is required")?;
        let found =
            environment(state, id).ok_or_else(|| format!("environment '{id}' does not exist"))?;
        if !found["archived_at"].is_null() {
            return Err(format!("environment '{id}' is archived"));
        }
        record["environment_id"] = json!(id);
    }
    if body.contains_key("initial_events") || creating {
        record["initial_events"] = json!(initial_events(body.get("initial_events"))?);
    }
    if body.contains_key("schedule") || creating {
        record["schedule"] = schedule(body.get("schedule"))?;
    }
    for key in ["vault_ids", "resources"] {
        match body.get(key) {
            Some(Value::Array(list)) => record[key] = json!(list),
            Some(Value::Null) => record[key] = json!([]),
            Some(_) => return Err(format!("`{key}` is a list")),
            None if creating => record[key] = json!([]),
            None => {}
        }
    }
    if body.get("budget").is_some_and(|b| !b.is_null()) {
        return Err(
            "budgets are not supported on this server: it has no list prices to count".into(),
        );
    }
    if body.contains_key("metadata") || creating {
        record["metadata"] = metadata(body.get("metadata"), Some(&record["metadata"]))?;
    }
    Ok(())
}

fn lookup(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    deployments(state)
        .get(id, "depl_")
        .ok_or_else(|| missing("deployment", id))
}

/// The time searched from for a deployment's next scheduled run: now, so
/// a schedule that changes or resumes never fires for times gone by.
fn set_next(record: &mut Value) {
    let next = upcoming(&record["schedule"], unix_now() as i64, 1)
        .first()
        .copied();
    record["_next_run_at"] = json!(next);
}

pub async fn create_deployment(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let mut record = json!({});
        configure(&state, &body, &mut record).map_err(bad)?;
        let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
        let held = deployments(&state)
            .all()
            .iter()
            .filter(|d| d["archived_at"].is_null())
            .count();
        if held >= MAX_DEPLOYMENTS {
            return Err(bad(format!(
                "the server holds at most {MAX_DEPLOYMENTS} deployments"
            )));
        }
        let id = new_id("depl_");
        let now = rfc3339(unix_now());
        record["id"] = json!(id);
        record["type"] = json!("deployment");
        record["status"] = json!("active");
        record["paused_reason"] = Value::Null;
        record["budget"] = Value::Null;
        record["schedule"]["last_run_at"] = Value::Null;
        record["created_at"] = json!(now);
        record["updated_at"] = json!(now);
        record["archived_at"] = Value::Null;
        record["_order"] = json!(format!("{:039}", now_nanos()));
        set_next(&mut record);
        deployments(&state).put(&id, &record)?;
        webhooks::emit(&state, "deployment.created", &id);
        Ok(deployment_json(&record))
    })())
}

pub async fn list_deployments(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    respond({
        let statuses = query.all("status");
        let archived = query.flag("include_archived");
        let agent = query.get("agent_id");
        let items = deployments(&state)
            .all()
            .iter()
            .filter(|d| archived || d["archived_at"].is_null())
            .filter(|d| agent.is_none_or(|a| d["agent"]["id"] == a))
            .filter(|d| statuses.is_empty() || statuses.iter().any(|s| d["status"] == *s))
            .map(deployment_json)
            .collect();
        cursor_page(items, &query, 20)
    })
}

pub async fn retrieve_deployment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(lookup(&state, &id).map(|r| deployment_json(&r)))
}

pub async fn update_deployment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
        let mut record = lookup(&state, &id)?;
        if !record["archived_at"].is_null() {
            return Err(bad(format!(
                "deployment '{id}' is archived and cannot be changed"
            )));
        }
        let before = record.clone();
        let last_run = record["schedule"]["last_run_at"].clone();
        configure(&state, &body, &mut record).map_err(bad)?;
        record["schedule"]["last_run_at"] = last_run;
        if record != before {
            record["updated_at"] = json!(rfc3339(unix_now()));
            if record["schedule"] != before["schedule"] {
                set_next(&mut record);
            }
            deployments(&state).put(&id, &record)?;
            webhooks::emit(&state, "deployment.updated", &id);
        }
        Ok(deployment_json(&record))
    })())
}

/// Changes a deployment's state: `pause`, `unpause` or `archive`.
fn transition(state: &AppState, id: &str, to: &str) -> Reply {
    let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut record = lookup(state, id)?;
    if !record["archived_at"].is_null() {
        return Err(bad(format!("deployment '{id}' is archived")));
    }
    let now = rfc3339(unix_now());
    let event = match to {
        "pause" if record["status"] == "active" => {
            record["status"] = json!("paused");
            record["paused_reason"] = json!({"type": "manual"});
            Some("deployment.paused")
        }
        "unpause" if record["status"] == "paused" => {
            record["status"] = json!("active");
            record["paused_reason"] = Value::Null;
            // Missed times are not made up for.
            set_next(&mut record);
            Some("deployment.unpaused")
        }
        "archive" => {
            record["status"] = json!("paused");
            record["paused_reason"] = Value::Null;
            record["archived_at"] = json!(now);
            Some("deployment.archived")
        }
        _ => None,
    };
    if let Some(event) = event {
        record["updated_at"] = json!(now);
        deployments(state).put(id, &record)?;
        webhooks::emit(state, event, id);
    }
    Ok(deployment_json(&record))
}

pub async fn pause_deployment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(transition(&state, &id, "pause"))
}

pub async fn unpause_deployment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(transition(&state, &id, "unpause"))
}

pub async fn archive_deployment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(transition(&state, &id, "archive"))
}

pub async fn delete_deployment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
        lookup(&state, &id)?;
        deployments(&state).remove(&id);
        webhooks::emit(&state, "deployment.deleted", &id);
        Ok(json!({"id": id, "type": "deployment_deleted"}))
    })())
}

/// Runs a deployment now — while it is paused too: the run.
pub async fn run_deployment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let reply: Reply = async {
        let record = lookup(&state, &id)?;
        if !record["archived_at"].is_null() {
            return Err(bad(format!("deployment '{id}' is archived")));
        }
        Ok(fire(&state, &record, json!({"type": "manual"})).await)
    }
    .await;
    respond(reply)
}

pub async fn list_runs(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    respond({
        let deployment = query.get("deployment_id");
        let errors = query.get("has_error").map(|v| v == "true");
        let items = runs(&state)
            .all()
            .into_iter()
            .filter(|r| deployment.is_none_or(|d| r["deployment_id"] == d))
            .filter(|r| errors.is_none_or(|e| r["error"].is_null() != e))
            .map(|mut r| {
                if let Some(map) = r.as_object_mut() {
                    map.retain(|key, _| !key.starts_with('_'));
                }
                r
            })
            .collect();
        cursor_page(items, &query, 20)
    })
}

pub async fn retrieve_run(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    respond(
        runs(&state)
            .get(&id, "drun_")
            .map(|mut r| {
                if let Some(map) = r.as_object_mut() {
                    map.retain(|key, _| !key.starts_with('_'));
                }
                r
            })
            .ok_or_else(|| missing("deployment run", &id)),
    )
}

/// Archives every deployment that runs agent `agent`, now that it is
/// archived.
pub(super) fn agent_archived(state: &AppState, agent: &str) {
    let ids: Vec<String> = deployments(state)
        .all()
        .into_iter()
        .filter(|d| d["archived_at"].is_null())
        .filter(|d| d["agent"]["id"] == agent)
        .filter_map(|d| d["id"].as_str().map(str::to_string))
        .collect();
    for id in ids {
        let _ = transition(state, &id, "archive");
    }
}

/// Why a run of `deployment` cannot make its session, before it tries: the
/// run's `error`, and whether it is one that pauses the deployment.
fn precheck(state: &AppState, deployment: &Value, agent: Option<&Value>) -> Option<(Value, bool)> {
    let error = |kind: &str, message: String| json!({"type": kind, "message": message});
    let agent_id = deployment["agent"]["id"].as_str().unwrap_or_default();
    if agent.is_none_or(|a| !a["archived_at"].is_null()) {
        return Some((
            error(
                "agent_archived_error",
                format!("agent `{agent_id}` is archived"),
            ),
            false,
        ));
    }
    let environment_id = deployment["environment_id"].as_str().unwrap_or_default();
    match environment(state, environment_id) {
        None => {
            return Some((
                error(
                    "environment_not_found_error",
                    format!("environment `{environment_id}` does not exist"),
                ),
                true,
            ));
        }
        Some(e) if !e["archived_at"].is_null() => {
            return Some((
                error(
                    "environment_archived_error",
                    format!("environment `{environment_id}` is archived"),
                ),
                true,
            ));
        }
        _ => {}
    }
    let vaults = Records::new(state, "vaults");
    for vault in deployment["vault_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        match vaults.get(vault, "vlt_") {
            None => {
                return Some((
                    error(
                        "vault_not_found_error",
                        format!("vault `{vault}` does not exist"),
                    ),
                    true,
                ));
            }
            Some(v) if !v["archived_at"].is_null() => {
                return Some((
                    error(
                        "vault_archived_error",
                        format!("vault `{vault}` is archived"),
                    ),
                    true,
                ));
            }
            _ => {}
        }
    }
    None
}

/// Makes the session of one run of `deployment`: the run record, which
/// names the session or the error that stopped it. A scheduled run that
/// fails for a reason only a change to the deployment's resources fixes
/// pauses the deployment.
async fn fire(state: &Arc<AppState>, deployment: &Value, trigger: Value) -> Value {
    let id = deployment["id"].as_str().unwrap_or_default().to_string();
    let scheduled = trigger["type"] == "schedule";
    let run_id = new_id("drun_");
    let agent_id = deployment["agent"]["id"].as_str().unwrap_or_default();
    let agent = agent_at(state, agent_id, deployment["agent"]["version"].as_u64());
    let mut run = json!({
        "type": "deployment_run", "id": run_id, "deployment_id": id,
        "trigger_context": trigger, "session_id": null, "error": null,
        "agent": deployment["agent"],
        "created_at": rfc3339(unix_now()),
        "_order": format!("{:039}", now_nanos()),
    });
    if scheduled {
        webhooks::emit(state, "deployment_run.started", &run_id);
    }
    let mut pause = None;
    match precheck(state, deployment, agent.as_ref()) {
        Some((error, pauses)) => {
            if pauses {
                pause = Some(json!({"type": error["type"]}));
            }
            run["error"] = error;
        }
        None => {
            let mut body = Map::new();
            for key in [
                "agent",
                "environment_id",
                "vault_ids",
                "resources",
                "initial_events",
            ] {
                body.insert(key.into(), deployment[key].clone());
            }
            body.insert("title".into(), deployment["name"].clone());
            match super::sessions::open_session(state, &body, true).await {
                Ok(session) => run["session_id"] = session["id"].clone(),
                Err(response) => {
                    let message = super::error_of(*response).await["error"]["message"]
                        .as_str()
                        .unwrap_or("the session could not be made")
                        .to_string();
                    run["error"] =
                        json!({"type": "session_creation_rejected_error", "message": message});
                }
            }
        }
    }
    let _ = runs(state).put(&run_id, &run);
    if scheduled {
        let outcome = if run["error"].is_null() {
            "deployment_run.succeeded"
        } else {
            "deployment_run.failed"
        };
        webhooks::emit(state, outcome, &run_id);
        if let Some(error) = pause {
            let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
            if let Some(mut record) = deployments(state).get(&id, "depl_")
                && record["status"] == "active"
                && record["archived_at"].is_null()
            {
                record["status"] = json!("paused");
                record["paused_reason"] = json!({"type": "error", "error": error});
                record["updated_at"] = json!(rfc3339(unix_now()));
                if deployments(state).put(&id, &record).is_ok() {
                    webhooks::emit(state, "deployment.paused", &id);
                }
            }
        }
    }
    if let Some(map) = run.as_object_mut() {
        map.retain(|key, _| !key.starts_with('_'));
    }
    run
}

/// Fires the active deployments that are due at `now`: each once, for the
/// time it was due, its next time then found from `now` on.
async fn tick(state: &Arc<AppState>, now: i64) {
    let due: Vec<(Value, i64)> = {
        let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
        let mut due = Vec::new();
        for mut record in deployments(state).all() {
            if record["status"] != "active" || !record["archived_at"].is_null() {
                continue;
            }
            let Some(at) = record["_next_run_at"].as_i64() else {
                continue;
            };
            if at > now {
                continue;
            }
            record["_next_run_at"] = json!(upcoming(&record["schedule"], now, 1).first());
            let id = record["id"].as_str().unwrap_or_default().to_string();
            // A deployment whose agent is archived or gone is archived, and
            // makes no run.
            let agent = agent_at(
                state,
                record["agent"]["id"].as_str().unwrap_or_default(),
                None,
            );
            if agent.is_none_or(|a| !a["archived_at"].is_null()) {
                let stamp = rfc3339(unix_now());
                record["status"] = json!("paused");
                record["paused_reason"] = Value::Null;
                record["archived_at"] = json!(stamp);
                record["updated_at"] = json!(stamp);
                if deployments(state).put(&id, &record).is_ok() {
                    webhooks::emit(state, "deployment.archived", &id);
                }
                continue;
            }
            record["schedule"]["last_run_at"] = json!(rfc3339(unix_now()));
            if deployments(state).put(&id, &record).is_ok() {
                due.push((record, at));
            }
        }
        due
    };
    for (record, at) in due {
        let trigger = json!({"type": "schedule", "scheduled_at": rfc3339(at as u64)});
        fire(state, &record, trigger).await;
    }
}

/// Starts the task that fires scheduled deployments: at the start of every
/// minute, those that are due.
pub fn start_scheduler(state: Arc<AppState>) {
    tokio::spawn(async move {
        loop {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let into_minute = now.as_millis() % 60_000;
            tokio::time::sleep(Duration::from_millis((60_000 - into_minute) as u64 + 50)).await;
            tick(&state, unix_now() as i64).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> i64 {
        chrono::DateTime::parse_from_rfc3339(text)
            .unwrap()
            .timestamp()
    }

    #[test]
    fn cron_fields_take_ranges_steps_lists_and_names() {
        let cron = Cron::parse("*/15 9-17 * JAN,jul mon-fri").unwrap();
        assert_eq!(cron.minutes, 1 | 1 << 15 | 1 << 30 | 1 << 45);
        assert_eq!(cron.hours, (9..=17).map(|h| 1 << h).sum::<u32>());
        assert_eq!(cron.months, 1 << 1 | 1 << 7);
        assert_eq!(cron.weekdays, 0b0111110);
        assert_eq!(Cron::parse("0 0 * * 7").unwrap().weekdays, 1);
        assert_eq!(
            Cron::parse("5/20 * * * *").unwrap().minutes,
            1 << 5 | 1 << 25 | 1 << 45
        );
        assert!(Cron::parse("60 * * * *").is_err());
        assert!(Cron::parse("* * * *").is_err());
        assert!(Cron::parse("*/0 * * * *").is_err());
    }

    #[test]
    fn a_weekly_schedule_fires_at_local_wall_clock_time() {
        // Fridays at 20:00 in New York: 00:00 UTC on Saturday in summer.
        let cron = Cron::parse("0 20 * * 5").unwrap();
        let zone: chrono_tz::Tz = "America/New_York".parse().unwrap();
        let next = cron.upcoming(zone, at("2026-05-04T00:00:00Z"), 3);
        assert_eq!(
            next,
            vec![
                at("2026-05-09T00:00:00Z"),
                at("2026-05-16T00:00:00Z"),
                at("2026-05-23T00:00:00Z")
            ]
        );
    }

    #[test]
    fn a_skipped_hour_never_fires_and_a_repeated_one_fires_twice() {
        let cron = Cron::parse("30 2 * * *").unwrap();
        let zone: chrono_tz::Tz = "America/New_York".parse().unwrap();
        // 2026-03-08: 02:00 becomes 03:00, so 02:30 does not exist.
        let next = cron.upcoming(zone, at("2026-03-07T00:00:00Z"), 2);
        assert_eq!(
            next,
            vec![at("2026-03-07T07:30:00Z"), at("2026-03-09T06:30:00Z")]
        );
        let cron = Cron::parse("30 1 * * *").unwrap();
        // 2026-11-01: 01:00-02:00 happens twice.
        let next = cron.upcoming(zone, at("2026-11-01T00:00:00Z"), 2);
        assert_eq!(
            next,
            vec![at("2026-11-01T05:30:00Z"), at("2026-11-01T06:30:00Z")]
        );
    }

    #[test]
    fn day_of_month_and_day_of_week_either_fire_when_both_are_set() {
        // The 13th, and every Friday.
        let cron = Cron::parse("0 12 13 * 5").unwrap();
        let next = cron.upcoming(chrono_tz::UTC, at("2026-11-01T00:00:00Z"), 3);
        assert_eq!(
            next,
            vec![
                at("2026-11-06T12:00:00Z"),
                at("2026-11-13T12:00:00Z"),
                at("2026-11-20T12:00:00Z")
            ]
        );
    }

    #[test]
    fn a_schedule_that_never_fires_is_refused() {
        let never = json!({"type": "cron", "expression": "0 0 30 2 *", "timezone": "UTC"});
        assert!(schedule(Some(&never)).is_err());
        let bad_zone = json!({"type": "cron", "expression": "0 0 * * *", "timezone": "Mars/Base"});
        assert!(schedule(Some(&bad_zone)).is_err());
    }

    #[test]
    fn initial_events_need_a_starting_event() {
        let note = json!({"type": "system.message", "content": [{"type": "text", "text": "x"}]});
        let message = json!({"type": "user.message", "content": "go"});
        assert!(initial_events(Some(&json!([note]))).is_err());
        assert!(initial_events(Some(&json!([note, message]))).is_ok());
        assert!(initial_events(Some(&json!([{"type": "user.interrupt"}]))).is_err());
    }
}
