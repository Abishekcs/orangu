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

//! Webhooks: the server POSTs a thin event — its type and the resource's
//! id — to every endpoint subscribed to that type when a Managed Agents
//! resource changes state. Deliveries are signed as Standard Webhooks
//! (`webhook-id`, `webhook-timestamp`, `webhook-signature`) with the
//! endpoint's `whsec_` secret, so the SDKs' `webhooks.unwrap()` verifies
//! them.
//!
//! Endpoints (`/v1/webhook_endpoints`) are records under
//! `<api_data>/webhook_endpoints/`; each one's secret is a file of its own
//! that only the server's user can read, returned once, when it is made or
//! rotated.

use axum::{
    Json,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::StatusCode,
    response::Response,
};
use base64::Engine as _;
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use super::agents::{
    ListQuery, Records, Reply, bad, body, cursor_page, fail, missing, records_lock, respond,
};
use super::files::{new_id, now_nanos};
use super::{rfc3339, unix_now};
use crate::http::AppState;

/// Every event type an endpoint may subscribe to.
pub(super) const EVENT_TYPES: &[&str] = &[
    "session.status_scheduled",
    "session.status_run_started",
    "session.status_idled",
    "session.status_rescheduled",
    "session.status_terminated",
    "session.thread_created",
    "session.thread_idled",
    "session.thread_terminated",
    "session.outcome_evaluation_ended",
    "session.updated",
    "session.deleted",
    "vault.created",
    "vault.archived",
    "vault.deleted",
    "vault_credential.created",
    "vault_credential.archived",
    "vault_credential.deleted",
    "vault_credential.refresh_failed",
    "agent.created",
    "agent.updated",
    "agent.archived",
    "agent.deleted",
    "deployment.created",
    "deployment.updated",
    "deployment.paused",
    "deployment.unpaused",
    "deployment.archived",
    "deployment.deleted",
    "deployment_run.started",
    "deployment_run.succeeded",
    "deployment_run.failed",
    "environment.created",
    "environment.updated",
    "environment.archived",
    "environment.deleted",
    "memory_store.created",
    "memory_store.archived",
    "memory_store.deleted",
];

/// The organization and workspace every event names: this server is one
/// of each.
const ORGANIZATION_ID: &str = "local";
const WORKSPACE_ID: &str = "default";

/// How many times one event is attempted per endpoint.
const ATTEMPTS: u32 = 3;

/// The wait before the first retry; each later one waits twice as long,
/// up to [`MAX_BACKOFF`], with up to as much again of jitter.
const FIRST_BACKOFF: Duration = Duration::from_secs(5);
const MAX_BACKOFF: Duration = Duration::from_secs(120);

/// How long one delivery attempt may take.
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(15);

/// How long an endpoint may fail every delivery before it is disabled.
const SUSTAINED_FAILURE: Duration = Duration::from_secs(24 * 60 * 60);

/// The most endpoints the server holds.
const MAX_ENDPOINTS: usize = 100;

/// When each failing endpoint's run of failures began, in Unix seconds.
static FAILING_SINCE: LazyLock<Mutex<HashMap<String, u64>>> = LazyLock::new(Default::default);

fn endpoints(state: &AppState) -> Records {
    Records::new(state, "webhook_endpoints")
}

fn secret_path(state: &AppState, id: &str) -> std::path::PathBuf {
    state
        .api_data
        .join("webhook_endpoints")
        .join("secrets")
        .join(id)
}

/// A new signing secret: `whsec_` and 32 random bytes in base64.
fn new_secret() -> String {
    let bytes: [u8; 32] = rand::random();
    format!(
        "whsec_{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// Writes an endpoint's secret where only this user can read it.
fn write_secret(state: &AppState, id: &str, secret: &str) -> Result<(), Box<Response>> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt as _;
    let path = secret_path(state, id);
    let write = || -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(secret.as_bytes())?;
        std::fs::rename(tmp, &path)
    };
    write().map_err(|err| {
        fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("saving the endpoint's secret: {err}"),
        )
    })
}

/// The Standard Webhooks signature of one delivery: `v1,` and the base64
/// HMAC-SHA256 of `{id}.{timestamp}.{body}`, keyed with the secret's bytes.
pub(super) fn signature(secret: &str, id: &str, timestamp: u64, body: &str) -> Option<String> {
    let key = base64::engine::general_purpose::STANDARD
        .decode(secret.strip_prefix("whsec_").unwrap_or(secret))
        .ok()?;
    let mut mac = Hmac::<sha2::Sha256>::new_from_slice(&key).ok()?;
    mac.update(format!("{id}.{timestamp}.{body}").as_bytes());
    Some(format!(
        "v1,{}",
        base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
    ))
}

/// Announces that resource `id` had event `kind`: every enabled endpoint
/// subscribed to it is sent the event, in the background.
pub(super) fn emit(state: &AppState, kind: &str, id: &str) {
    let subscribed: Vec<Value> = endpoints(state)
        .all()
        .into_iter()
        .filter(|e| e["disabled_reason"].is_null())
        .filter(|e| {
            e["event_types"]
                .as_array()
                .is_some_and(|types| types.iter().any(|t| t == kind))
        })
        .collect();
    if subscribed.is_empty() {
        return;
    }
    let event_id = new_id("whe_");
    let body = json!({
        "type": "event",
        "id": event_id,
        "created_at": rfc3339(unix_now()),
        "data": {"type": kind, "id": id, "organization_id": ORGANIZATION_ID,
                 "workspace_id": WORKSPACE_ID},
    })
    .to_string();
    for endpoint in subscribed {
        let Some(endpoint_id) = endpoint["id"].as_str().map(str::to_string) else {
            continue;
        };
        let Ok(secret) = std::fs::read_to_string(secret_path(state, &endpoint_id)) else {
            continue;
        };
        let url = endpoint["url"].as_str().unwrap_or_default().to_string();
        let records = endpoints(state);
        let (event_id, body) = (event_id.clone(), body.clone());
        tokio::spawn(async move {
            deliver(&records, &endpoint_id, &url, &secret, &event_id, &body).await;
        });
    }
}

/// Sends one event to one endpoint: up to [`ATTEMPTS`] times, apart by a
/// growing, jittered wait. A redirect disables the endpoint at once, as
/// does failing every delivery for [`SUSTAINED_FAILURE`].
async fn deliver(
    records: &Records,
    endpoint: &str,
    url: &str,
    secret: &str,
    event_id: &str,
    body: &str,
) {
    let Ok(client) = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(ATTEMPT_TIMEOUT)
        .user_agent(concat!("orangu-server/", env!("CARGO_PKG_VERSION")))
        .build()
    else {
        return;
    };
    let mut backoff = FIRST_BACKOFF;
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            let jitter = rand::random::<f64>() * backoff.as_secs_f64();
            tokio::time::sleep(backoff + Duration::from_secs_f64(jitter)).await;
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
        // Each attempt is stamped anew, so a retry is fresh.
        let timestamp = unix_now();
        let Some(signed) = signature(secret, event_id, timestamp, body) else {
            return;
        };
        let sent = client
            .post(url)
            .header("content-type", "application/json")
            .header("webhook-id", event_id)
            .header("webhook-timestamp", timestamp.to_string())
            .header("webhook-signature", signed)
            .body(body.to_string())
            .send()
            .await;
        let status = sent.as_ref().map(|r| r.status()).ok();
        if status.is_some_and(|s| s.is_success()) {
            FAILING_SINCE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(endpoint);
            return;
        }
        if status.is_some_and(|s| s.is_redirection()) {
            disable(
                records,
                endpoint,
                "auto-disabled: endpoint URL returned a redirect (3xx)",
            );
            return;
        }
        let since = *FAILING_SINCE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(endpoint.to_string())
            .or_insert_with(unix_now);
        if unix_now().saturating_sub(since) >= SUSTAINED_FAILURE.as_secs() {
            disable(
                records,
                endpoint,
                "auto-disabled after sustained delivery failures",
            );
            return;
        }
    }
}

fn disable(records: &Records, endpoint: &str, reason: &str) {
    let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(mut record) = records.get(endpoint, "whep_")
        && record["disabled_reason"].is_null()
    {
        record["disabled_reason"] = json!(reason);
        record["updated_at"] = json!(rfc3339(unix_now()));
        let _ = records.put(endpoint, &record);
    }
    FAILING_SINCE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(endpoint);
}

/// An endpoint as the API shows it.
fn endpoint_json(record: &Value) -> Value {
    let mut out = record.clone();
    if let Some(map) = out.as_object_mut() {
        map.retain(|key, _| !key.starts_with('_'));
    }
    out
}

/// `url` checked: an absolute http(s) URL.
fn endpoint_url(given: Option<&Value>) -> Result<String, String> {
    let url = given
        .and_then(Value::as_str)
        .ok_or("`url` is the endpoint's http(s) URL")?;
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("`url` is not a URL: {e}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err("`url` is an http or https URL".into());
    }
    Ok(url.to_string())
}

/// `event_types` checked: one or more of [`EVENT_TYPES`].
fn event_types(given: Option<&Value>) -> Result<Vec<String>, String> {
    let list = given
        .and_then(Value::as_array)
        .filter(|l| !l.is_empty())
        .ok_or("`event_types` lists the event types the endpoint receives")?;
    let mut out = Vec::new();
    for kind in list {
        let kind = kind
            .as_str()
            .filter(|k| EVENT_TYPES.contains(k))
            .ok_or_else(|| format!("'{kind}' is not a webhook event type"))?;
        if !out.iter().any(|k| k == kind) {
            out.push(kind.to_string());
        }
    }
    Ok(out)
}

fn lookup(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    endpoints(state)
        .get(id, "whep_")
        .ok_or_else(|| missing("webhook endpoint", id))
}

pub async fn create_endpoint(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let url = endpoint_url(body.get("url")).map_err(bad)?;
        let types = event_types(body.get("event_types")).map_err(bad)?;
        let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
        if endpoints(&state).all().len() >= MAX_ENDPOINTS {
            return Err(bad(format!(
                "the server holds at most {MAX_ENDPOINTS} webhook endpoints"
            )));
        }
        let id = new_id("whep_");
        let now = rfc3339(unix_now());
        let record = json!({
            "type": "webhook_endpoint", "id": id, "url": url, "event_types": types,
            "description": body.get("description").and_then(Value::as_str),
            "disabled_reason": null, "created_at": now, "updated_at": now,
            "_order": format!("{:039}", now_nanos()),
        });
        let secret = new_secret();
        write_secret(&state, &id, &secret)?;
        endpoints(&state).put(&id, &record)?;
        let mut out = endpoint_json(&record);
        out["secret"] = json!(secret);
        Ok(out)
    })())
}

pub async fn list_endpoints(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    respond({
        let items = endpoints(&state).all().iter().map(endpoint_json).collect();
        cursor_page(items, &query, 100)
    })
}

pub async fn retrieve_endpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(lookup(&state, &id).map(|r| endpoint_json(&r)))
}

/// Changes `url`, `event_types` or `description`; `enabled: true` turns a
/// disabled endpoint back on, `enabled: false` turns it off.
pub async fn update_endpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body: Map<String, Value> = body(payload)?;
        let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
        let mut record = lookup(&state, &id)?;
        if body.contains_key("url") {
            record["url"] = json!(endpoint_url(body.get("url")).map_err(bad)?);
        }
        if body.contains_key("event_types") {
            record["event_types"] = json!(event_types(body.get("event_types")).map_err(bad)?);
        }
        if let Some(description) = body.get("description") {
            record["description"] = description.clone();
        }
        match body.get("enabled").and_then(Value::as_bool) {
            Some(true) => record["disabled_reason"] = Value::Null,
            Some(false) => record["disabled_reason"] = json!("disabled by request"),
            None => {}
        }
        record["updated_at"] = json!(rfc3339(unix_now()));
        endpoints(&state).put(&id, &record)?;
        FAILING_SINCE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
        Ok(endpoint_json(&record))
    })())
}

/// Replaces an endpoint's secret: the new one, shown this once.
pub async fn rotate_secret(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    respond((|| -> Reply {
        let record = lookup(&state, &id)?;
        let secret = new_secret();
        write_secret(&state, &id, &secret)?;
        let mut out = endpoint_json(&record);
        out["secret"] = json!(secret);
        Ok(out)
    })())
}

pub async fn delete_endpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|e| e.into_inner());
        lookup(&state, &id)?;
        endpoints(&state).remove(&id);
        let _ = std::fs::remove_file(secret_path(&state, &id));
        Ok(json!({"id": id, "type": "webhook_endpoint_deleted"}))
    })())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_follow_standard_webhooks() {
        // The Standard Webhooks specification's own test vector.
        let signed = signature(
            "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw",
            "msg_p5jXN8AQM9LWM0D4loKWxJek",
            1614265330,
            r#"{"test": 2432232314}"#,
        );
        assert_eq!(
            signed.as_deref(),
            Some("v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=")
        );
    }

    #[test]
    fn event_types_are_checked_and_deduplicated() {
        let given = json!(["agent.created", "agent.created", "session.status_idled"]);
        assert_eq!(
            event_types(Some(&given)).unwrap(),
            vec!["agent.created", "session.status_idled"]
        );
        assert!(event_types(Some(&json!(["session.status_idle"]))).is_err());
        assert!(event_types(Some(&json!([]))).is_err());
    }

    #[test]
    fn a_secret_is_32_bytes() {
        let secret = new_secret();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(secret.strip_prefix("whsec_").unwrap())
            .unwrap();
        assert_eq!(bytes.len(), 32);
    }
}
