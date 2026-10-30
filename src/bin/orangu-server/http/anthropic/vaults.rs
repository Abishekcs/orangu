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

//! Vaults (`/v1/vaults`): credentials a session uses without seeing them.
//! A vault is a record under `<api_data>/vaults/`, its credentials records
//! under `<api_data>/vaults/<id>/credentials/`, and each credential's
//! secret fields a file of their own under `<api_data>/vaults/<id>/secrets/`
//! that only the server's user can read, which no response ever includes
//! and archiving deletes.
//!
//! An MCP credential (`static_bearer`, `mcp_oauth`) is the bearer token of
//! the MCP server at its URL — an `mcp_oauth` one refreshed through its
//! token endpoint when it has run out. An `environment_variable` credential
//! is a variable the sandbox sees only a placeholder of; the session's
//! egress proxy puts the secret in its place on the way out (see
//! [`super::egress`]).

use axum::{
    Json,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::StatusCode,
    response::Response,
};
use serde_json::{Map, Value, json};
use std::sync::Arc;

use super::agents::{
    ListQuery, Records, Reply, bad, body, cursor_page, fail, metadata, missing, records_lock,
    respond,
};
use super::files::{new_id, now_nanos};
use super::{rfc3339, unix_now};
use crate::http::AppState;

/// The most active credentials one vault holds.
const MAX_CREDENTIALS: usize = 20;

/// How long before it runs out an OAuth access token is refreshed.
const REFRESH_MARGIN_SECS: i64 = 60;

fn vaults(state: &AppState) -> Records {
    Records::new(state, "vaults")
}

fn credentials(state: &AppState, vault: &str) -> Records {
    Records::new(state, &format!("vaults/{vault}/credentials"))
}

fn secret_path(state: &AppState, vault: &str, credential: &str) -> std::path::PathBuf {
    state
        .api_data
        .join("vaults")
        .join(vault)
        .join("secrets")
        .join(format!("{credential}.json"))
}

fn read_secret(state: &AppState, vault: &str, credential: &str) -> Map<String, Value> {
    std::fs::read(secret_path(state, vault, credential))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Writes a credential's secret fields where only this user can read them.
fn write_secret(
    state: &AppState,
    vault: &str,
    credential: &str,
    secret: &Map<String, Value>,
) -> Result<(), Box<Response>> {
    use std::io::Write as _;
    let path = secret_path(state, vault, credential);
    let write = || -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp)?;
        file.write_all(&serde_json::to_vec(secret).unwrap_or_default())?;
        std::fs::rename(tmp, &path)
    };
    write().map_err(|err| {
        fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("saving a credential's secret: {err}"),
        )
    })
}

fn order_stamp() -> String {
    format!("{:039}", now_nanos())
}

fn public(record: &Value) -> Value {
    let mut out = record.clone();
    if let Some(map) = out.as_object_mut() {
        map.retain(|k, _| !k.starts_with('_'));
    }
    out
}

/// An MCP server URL as credentials are matched on: scheme and host in
/// lower case, a default port and a trailing slash dropped.
pub(super) fn normalized(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url.trim()).ok()?;
    let scheme = parsed.scheme().to_ascii_lowercase();
    let host = parsed.host_str()?.to_ascii_lowercase();
    // `port()` is already `None` for the scheme's default port.
    let port = parsed.port();
    let path = parsed.path().trim_end_matches('/');
    let mut out = format!("{scheme}://{host}");
    if let Some(port) = port {
        out.push_str(&format!(":{port}"));
    }
    out.push_str(path);
    if let Some(query) = parsed.query() {
        out.push('?');
        out.push_str(query);
    }
    Some(out)
}

fn text<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a str, String> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("`auth.{key}` is required"))
}

/// A credential's `networking`, checked.
fn networking(given: &Value) -> Result<Value, String> {
    match given["type"].as_str() {
        Some("unrestricted") => Ok(json!({"type": "unrestricted"})),
        Some("limited") => {
            let hosts: Vec<String> = given["allowed_hosts"]
                .as_array()
                .ok_or("limited networking needs `allowed_hosts`")?
                .iter()
                .map(|h| h.as_str().map(|h| h.trim().to_ascii_lowercase()))
                .collect::<Option<_>>()
                .ok_or("`allowed_hosts` lists host names")?;
            if hosts.iter().any(|h| h.is_empty() || h.contains('/')) {
                return Err(
                    "`allowed_hosts` lists host names, such as api.example.com or \
                            *.example.com"
                        .into(),
                );
            }
            Ok(json!({"type": "limited", "allowed_hosts": hosts}))
        }
        _ => Err(
            "`auth.networking` is {\"type\": \"limited\", \"allowed_hosts\": [...]} or \
                  {\"type\": \"unrestricted\"}"
                .into(),
        ),
    }
}

/// `injection_location` merged onto `base`: on create (no base) an absent
/// object enables both and a given one defaults what it leaves out to off.
fn injection(given: Option<&Value>, base: Option<&Value>) -> Result<Value, String> {
    let mut out = match base {
        Some(base) => base.clone(),
        None if given.is_none() => json!({"header": true, "body": true}),
        None => json!({"header": false, "body": false}),
    };
    if let Some(given) = given {
        let given = given
            .as_object()
            .ok_or("`injection_location` is an object of header and body")?;
        for key in ["header", "body"] {
            match given.get(key) {
                None => {}
                Some(Value::Bool(on)) => out[key] = json!(on),
                Some(_) => return Err(format!("`injection_location.{key}` is true or false")),
            }
        }
    }
    if out["header"] != true && out["body"] != true {
        return Err("a credential is substituted in the header, the body, or both".into());
    }
    Ok(out)
}

/// The OAuth refresh settings as stored and shown, and their secrets.
fn refresh_settings(
    given: &Value,
    base: Option<&Value>,
) -> Result<(Value, Map<String, Value>), String> {
    let given = given.as_object().ok_or("`auth.refresh` is an object")?;
    let mut public = base.cloned().unwrap_or_else(|| json!({}));
    let mut secret = Map::new();
    if base.is_none() {
        for key in ["client_id", "token_endpoint"] {
            public[key] =
                json!(text(given, key).map_err(|_| format!("`auth.refresh.{key}` is required"))?);
        }
        reqwest::Url::parse(public["token_endpoint"].as_str().unwrap_or_default())
            .map_err(|_| "`auth.refresh.token_endpoint` is not a URL")?;
    }
    match given.get("refresh_token").and_then(Value::as_str) {
        Some(token) => {
            secret.insert("refresh_token".into(), json!(token));
        }
        None if base.is_none() => return Err("`auth.refresh.refresh_token` is required".into()),
        None => {}
    }
    for key in ["scope", "resource"] {
        if let Some(value) = given.get(key) {
            public[key] = value.clone();
        }
    }
    match given.get("token_endpoint_auth") {
        Some(auth) => {
            let kind = auth["type"].as_str().unwrap_or_default();
            match kind {
                "none" => {}
                "client_secret_basic" | "client_secret_post" => {
                    let client_secret = auth["client_secret"].as_str().ok_or(
                        "the token endpoint's client authentication needs its `client_secret`",
                    )?;
                    secret.insert("client_secret".into(), json!(client_secret));
                }
                _ => {
                    return Err(
                        "`token_endpoint_auth.type` is none, client_secret_basic or \
                                client_secret_post"
                            .into(),
                    );
                }
            }
            public["token_endpoint_auth"] = json!({"type": kind});
        }
        None if base.is_none() => {
            return Err("`auth.refresh.token_endpoint_auth` is required".into());
        }
        None => {}
    }
    Ok((public, secret))
}

/// A credential's `auth` as stored and shown, and its secret fields — made
/// from what was sent, over `base` (the stored auth) for an update.
fn auth(given: &Value, base: Option<&Value>) -> Result<(Value, Map<String, Value>), String> {
    let given = given.as_object().ok_or("`auth` is required: an object")?;
    let kind = given
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if let Some(base) = base
        && base["type"] != kind
    {
        return Err(format!(
            "this credential is {}, and an update cannot make it {kind}",
            base["type"].as_str().unwrap_or_default()
        ));
    }
    let mut secret = Map::new();
    let mut public = base.cloned().unwrap_or_else(|| json!({"type": kind}));
    let mcp_url = |public: &mut Value| -> Result<(), String> {
        if base.is_none() {
            let url = text(given, "mcp_server_url")?;
            normalized(url).ok_or("`auth.mcp_server_url` is not a URL")?;
            public["mcp_server_url"] = json!(url);
        }
        Ok(())
    };
    match kind {
        "static_bearer" => {
            mcp_url(&mut public)?;
            match given.get("token").and_then(Value::as_str) {
                Some(token) => {
                    secret.insert("token".into(), json!(token));
                }
                None if base.is_none() => return Err("`auth.token` is required".into()),
                None => {}
            }
        }
        "mcp_oauth" => {
            mcp_url(&mut public)?;
            match given.get("access_token").and_then(Value::as_str) {
                Some(token) => {
                    secret.insert("access_token".into(), json!(token));
                }
                None if base.is_none() => return Err("`auth.access_token` is required".into()),
                None => {}
            }
            if let Some(at) = given.get("expires_at") {
                if !at.is_null()
                    && at
                        .as_str()
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .is_none()
                {
                    return Err("`auth.expires_at` is an RFC 3339 time".into());
                }
                public["expires_at"] = at.clone();
            } else if base.is_none() {
                public["expires_at"] = Value::Null;
            }
            match given.get("refresh") {
                Some(Value::Null) => public["refresh"] = Value::Null,
                Some(refresh) => {
                    let current = base.map(|b| &b["refresh"]).filter(|r| !r.is_null());
                    let (shown, hidden) = refresh_settings(refresh, current)?;
                    public["refresh"] = shown;
                    secret.extend(hidden);
                }
                None if base.is_none() => public["refresh"] = Value::Null,
                None => {}
            }
        }
        "environment_variable" => {
            if base.is_none() {
                let name = text(given, "secret_name")?;
                let shell_name = name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                if !shell_name {
                    return Err(format!(
                        "`auth.secret_name` '{name}' is not an environment variable name"
                    ));
                }
                public["secret_name"] = json!(name);
            }
            match given.get("secret_value").and_then(Value::as_str) {
                Some(value) => {
                    secret.insert("secret_value".into(), json!(value));
                }
                None if base.is_none() => return Err("`auth.secret_value` is required".into()),
                None => {}
            }
            match given.get("networking") {
                Some(net) if !net.is_null() => public["networking"] = networking(net)?,
                _ if base.is_none() => return Err("`auth.networking` is required".into()),
                _ => {}
            }
            if given.get("injection_location").is_some_and(Value::is_null) {
                return Err("`injection_location` is left out rather than null".into());
            }
            public["injection_location"] = injection(
                given.get("injection_location"),
                base.map(|b| &b["injection_location"]),
            )?;
        }
        _ => {
            return Err("`auth.type` is mcp_oauth, static_bearer or environment_variable".into());
        }
    }
    Ok((public, secret))
}

/// What a credential is unique by in its vault.
fn key_of(auth: &Value) -> String {
    match auth["type"].as_str() {
        Some("environment_variable") => {
            format!("env:{}", auth["secret_name"].as_str().unwrap_or_default())
        }
        _ => format!(
            "mcp:{}",
            normalized(auth["mcp_server_url"].as_str().unwrap_or_default()).unwrap_or_default()
        ),
    }
}

fn vault_lookup(state: &AppState, id: &str) -> Result<Value, Box<Response>> {
    vaults(state)
        .get(id, "vlt_")
        .ok_or_else(|| missing("vault", id))
}

fn credential_lookup(state: &AppState, vault: &str, id: &str) -> Result<Value, Box<Response>> {
    credentials(state, vault)
        .get(id, "vcrd_")
        .ok_or_else(|| missing("credential", id))
}

/// A vault a new session can use: it exists and is not archived.
pub(super) fn usable(state: &AppState, id: &str) -> Result<(), Box<Response>> {
    let vault = vault_lookup(state, id)?;
    if !vault["archived_at"].is_null() {
        return Err(bad(format!("vault '{id}' is archived")));
    }
    Ok(())
}

fn active_credentials(state: &AppState, vault: &str) -> Vec<Value> {
    credentials(state, vault)
        .all()
        .into_iter()
        .filter(|c| c["archived_at"].is_null())
        .collect()
}

// --- Sessions ---------------------------------------------------------------

/// The bearer token for the MCP server at `url`, from the first of
/// `vault_ids` holding a credential for it: an `mcp_oauth` one refreshed
/// first when it has run out, or is about to.
pub(super) async fn mcp_token(state: &AppState, vault_ids: &[String], url: &str) -> Option<String> {
    let wanted = normalized(url)?;
    for vault in vault_ids {
        let found = active_credentials(state, vault).into_iter().find(|c| {
            c["auth"]["type"] != "environment_variable"
                && normalized(c["auth"]["mcp_server_url"].as_str().unwrap_or_default()).as_deref()
                    == Some(wanted.as_str())
        });
        let Some(credential) = found else {
            continue;
        };
        let id = credential["id"].as_str().unwrap_or_default();
        if credential["auth"]["type"] == "mcp_oauth" && runs_out_soon(&credential["auth"]) {
            let _ = refresh(state, vault, &credential).await;
        }
        let secret = read_secret(state, vault, id);
        return secret
            .get("token")
            .or_else(|| secret.get("access_token"))
            .and_then(Value::as_str)
            .map(str::to_string);
    }
    None
}

fn runs_out_soon(auth: &Value) -> bool {
    auth["expires_at"]
        .as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .is_some_and(|at| at.timestamp() - (unix_now() as i64) < REFRESH_MARGIN_SECS)
}

/// The `environment_variable` credentials of `vault_ids`, for the egress
/// proxy: a variable name earlier vaults hold wins.
pub(super) fn env_secrets(state: &AppState, vault_ids: &[String]) -> Vec<super::egress::Secret> {
    let mut out: Vec<super::egress::Secret> = Vec::new();
    for vault in vault_ids {
        for credential in active_credentials(state, vault) {
            let auth = &credential["auth"];
            if auth["type"] != "environment_variable" {
                continue;
            }
            let name = auth["secret_name"].as_str().unwrap_or_default().to_string();
            if out.iter().any(|s| s.name == name) {
                continue;
            }
            let secret = read_secret(state, vault, credential["id"].as_str().unwrap_or_default());
            let Some(value) = secret.get("secret_value").and_then(Value::as_str) else {
                continue;
            };
            let hosts = (auth["networking"]["type"] == "limited").then(|| {
                auth["networking"]["allowed_hosts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|h| h.as_str().map(str::to_string))
                    .collect()
            });
            out.push(super::egress::Secret::new(
                name,
                value.to_string(),
                hosts,
                auth["injection_location"]["header"] == true,
                auth["injection_location"]["body"] == true,
            ));
        }
    }
    out
}

/// The first `body_limit` characters of an HTTP response, as Anthropic
/// reports one.
async fn http_response(response: reqwest::Response) -> (u16, Value, Option<String>) {
    let status = response.status().as_u16();
    let kind = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let body = response.text().await.unwrap_or_default();
    let shown: String = body.chars().take(2000).collect();
    let truncated = shown.len() < body.len();
    (
        status,
        json!({"status_code": status, "content_type": kind, "body": shown,
               "body_truncated": truncated}),
        Some(body),
    )
}

/// Refreshes an `mcp_oauth` credential's access token through its token
/// endpoint, storing what comes back: the refresh object Anthropic reports.
/// A refresh the token endpoint refuses, or that cannot reach it, is
/// announced as `vault_credential.refresh_failed`.
async fn refresh(state: &AppState, vault: &str, credential: &Value) -> Value {
    let refreshed = try_refresh(state, vault, credential).await;
    if matches!(
        refreshed["status"].as_str(),
        Some("failed" | "connect_error")
    ) {
        let id = credential["id"].as_str().unwrap_or_default();
        super::webhooks::emit(state, "vault_credential.refresh_failed", id);
    }
    refreshed
}

async fn try_refresh(state: &AppState, vault: &str, credential: &Value) -> Value {
    let id = credential["id"].as_str().unwrap_or_default();
    let settings = &credential["auth"]["refresh"];
    let mut secret = read_secret(state, vault, id);
    let Some(refresh_token) = secret
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return json!({"status": "no_refresh_token", "http_response": null});
    };
    let endpoint = settings["token_endpoint"].as_str().unwrap_or_default();
    let client_id = settings["client_id"].as_str().unwrap_or_default();
    let mut form = vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token),
    ];
    for key in ["scope", "resource"] {
        if let Some(value) = settings[key].as_str() {
            form.push((key, value.to_string()));
        }
    }
    let client_secret = secret
        .get("client_secret")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let client = reqwest::Client::new();
    let mut request = client.post(endpoint);
    match settings["token_endpoint_auth"]["type"].as_str() {
        Some("client_secret_basic") => {
            request = request.basic_auth(client_id, Some(&client_secret));
        }
        Some("client_secret_post") => {
            form.push(("client_id", client_id.to_string()));
            form.push(("client_secret", client_secret));
        }
        _ => form.push(("client_id", client_id.to_string())),
    }
    let mut encoder = reqwest::Url::parse("http://form.invalid/").expect("a URL");
    encoder
        .query_pairs_mut()
        .extend_pairs(form.iter().map(|(k, v)| (*k, v.as_str())));
    let encoded = encoder.query().unwrap_or_default().to_string();
    let response = match request
        .timeout(std::time::Duration::from_secs(30))
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(encoded)
        .send()
        .await
    {
        Ok(response) => response,
        Err(_) => return json!({"status": "connect_error", "http_response": null}),
    };
    let (status, shown, body) = http_response(response).await;
    let granted: Value = body
        .as_deref()
        .and_then(|b| serde_json::from_str(b).ok())
        .unwrap_or_default();
    let Some(access) = granted["access_token"]
        .as_str()
        .filter(|_| (200..300).contains(&status))
    else {
        return json!({"status": "failed", "http_response": shown});
    };
    secret.insert("access_token".into(), json!(access));
    if let Some(next) = granted["refresh_token"].as_str() {
        secret.insert("refresh_token".into(), json!(next));
    }
    let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
    if write_secret(state, vault, id, &secret).is_ok() {
        let mut updated = credential.clone();
        updated["auth"]["expires_at"] = match granted["expires_in"].as_i64() {
            Some(seconds) => json!(rfc3339((unix_now() as i64 + seconds).max(0) as u64)),
            None => Value::Null,
        };
        updated["updated_at"] = json!(rfc3339(unix_now()));
        let _ = credentials(state, vault).put(id, &updated);
    }
    json!({"status": "succeeded", "http_response": shown})
}

// --- Vaults -----------------------------------------------------------------

fn display_name(
    body: &Map<String, Value>,
    required: bool,
) -> Result<Option<String>, Box<Response>> {
    match body.get("display_name") {
        Some(Value::String(name)) if !name.trim().is_empty() && name.chars().count() <= 255 => {
            Ok(Some(name.clone()))
        }
        None | Some(Value::Null) if !required => Ok(None),
        _ => Err(bad("`display_name` is a name of 1 to 255 characters")),
    }
}

pub async fn create_vault(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let name = display_name(&body, true)?.unwrap_or_default();
        let metadata = metadata(body.get("metadata"), None).map_err(bad)?;
        let id = new_id("vlt_");
        let now = rfc3339(unix_now());
        let vault = json!({
            "id": id, "type": "vault", "display_name": name, "metadata": metadata,
            "created_at": now, "updated_at": now, "archived_at": Value::Null,
            "_order": order_stamp(),
        });
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        vaults(&state).put(&id, &vault)?;
        super::webhooks::emit(&state, "vault.created", &id);
        Ok(public(&vault))
    })())
}

pub async fn list_vaults(
    State(state): State<Arc<AppState>>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    let query = ListQuery::new(pairs);
    let archived = query.flag("include_archived");
    let items = vaults(&state)
        .all()
        .iter()
        .filter(|v| archived || v["archived_at"].is_null())
        .map(public)
        .collect();
    respond(cursor_page(items, &query, 20))
}

pub async fn retrieve_vault(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    respond(vault_lookup(&state, &id).map(|v| public(&v)))
}

pub async fn update_vault(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut vault = vault_lookup(&state, &id)?;
        if !vault["archived_at"].is_null() {
            return Err(bad(format!("vault '{id}' is archived and cannot change")));
        }
        if let Some(name) = display_name(&body, false)? {
            vault["display_name"] = json!(name);
        }
        if body.contains_key("metadata") {
            vault["metadata"] =
                metadata(body.get("metadata"), Some(&vault["metadata"])).map_err(bad)?;
        }
        vault["updated_at"] = json!(rfc3339(unix_now()));
        vaults(&state).put(&id, &vault)?;
        Ok(public(&vault))
    })())
}

pub async fn delete_vault(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        vault_lookup(&state, &id)?;
        let held: Vec<String> = credentials(&state, &id)
            .all()
            .iter()
            .filter_map(|c| c["id"].as_str().map(str::to_string))
            .collect();
        vaults(&state).remove(&id);
        let _ = std::fs::remove_dir_all(state.api_data.join("vaults").join(&id));
        super::webhooks::emit(&state, "vault.deleted", &id);
        for credential in held {
            super::webhooks::emit(&state, "vault_credential.deleted", &credential);
        }
        Ok(json!({"id": id, "type": "vault_deleted"}))
    })())
}

pub async fn archive_vault(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut vault = vault_lookup(&state, &id)?;
        if vault["archived_at"].is_null() {
            let now = rfc3339(unix_now());
            vault["archived_at"] = json!(now);
            vault["updated_at"] = json!(now);
            vaults(&state).put(&id, &vault)?;
            super::webhooks::emit(&state, "vault.archived", &id);
            // An archived vault's credentials go with it, secrets and all.
            for credential in active_credentials(&state, &id) {
                archive_one(&state, &id, credential)?;
            }
        }
        Ok(public(&vault))
    })())
}

// --- Credentials ------------------------------------------------------------

pub async fn create_credential(
    State(state): State<Arc<AppState>>,
    Path(vault): Path<String>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let found = vault_lookup(&state, &vault)?;
        if !found["archived_at"].is_null() {
            return Err(bad(format!("vault '{vault}' is archived")));
        }
        let (auth, secret) = auth(body.get("auth").unwrap_or(&Value::Null), None).map_err(bad)?;
        let active = active_credentials(&state, &vault);
        if active.len() >= MAX_CREDENTIALS {
            return Err(bad(format!(
                "a vault holds at most {MAX_CREDENTIALS} credentials"
            )));
        }
        let key = key_of(&auth);
        if let Some(other) = active.iter().find(|c| key_of(&c["auth"]) == key) {
            return Err(fail(
                StatusCode::CONFLICT,
                format!(
                    "credential '{}' in this vault already holds that {}",
                    other["id"].as_str().unwrap_or_default(),
                    if auth["type"] == "environment_variable" {
                        "secret_name"
                    } else {
                        "mcp_server_url"
                    }
                ),
            ));
        }
        let id = new_id("vcrd_");
        let now = rfc3339(unix_now());
        let credential = json!({
            "id": id, "type": "vault_credential", "vault_id": vault, "auth": auth,
            "display_name": display_name(&body, false)?,
            "metadata": metadata(body.get("metadata"), None).map_err(bad)?,
            "created_at": now, "updated_at": now, "archived_at": Value::Null,
            "_order": order_stamp(),
        });
        write_secret(&state, &vault, &id, &secret)?;
        credentials(&state, &vault).put(&id, &credential)?;
        super::webhooks::emit(&state, "vault_credential.created", &id);
        Ok(public(&credential))
    })())
}

pub async fn list_credentials(
    State(state): State<Arc<AppState>>,
    Path(vault): Path<String>,
    Query(pairs): Query<Vec<(String, String)>>,
) -> Response {
    respond((|| {
        let query = ListQuery::new(pairs);
        vault_lookup(&state, &vault)?;
        let archived = query.flag("include_archived");
        let items = credentials(&state, &vault)
            .all()
            .iter()
            .filter(|c| archived || c["archived_at"].is_null())
            .map(public)
            .collect();
        cursor_page(items, &query, 20)
    })())
}

pub async fn retrieve_credential(
    State(state): State<Arc<AppState>>,
    Path((vault, id)): Path<(String, String)>,
) -> Response {
    respond(credential_lookup(&state, &vault, &id).map(|c| public(&c)))
}

pub async fn update_credential(
    State(state): State<Arc<AppState>>,
    Path((vault, id)): Path<(String, String)>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    respond((|| {
        let body = body(payload)?;
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut credential = credential_lookup(&state, &vault, &id)?;
        if !credential["archived_at"].is_null() {
            return Err(bad(format!(
                "credential '{id}' is archived and cannot change"
            )));
        }
        if let Some(given) = body.get("auth").filter(|a| !a.is_null()) {
            for key in ["mcp_server_url", "secret_name"] {
                if given.get(key).is_some() {
                    return Err(bad(format!(
                        "`auth.{key}` cannot change: archive this credential and create another"
                    )));
                }
            }
            if let Some(refresh) = given.get("refresh").filter(|r| r.is_object()) {
                for key in ["client_id", "token_endpoint"] {
                    if refresh.get(key).is_some() {
                        return Err(bad(format!(
                            "`auth.refresh.{key}` cannot change: archive this credential and \
                             create another"
                        )));
                    }
                }
            }
            let (auth, secret) = auth(given, Some(&credential["auth"])).map_err(bad)?;
            let mut stored = read_secret(&state, &vault, &id);
            stored.extend(secret);
            write_secret(&state, &vault, &id, &stored)?;
            credential["auth"] = auth;
        }
        if body.contains_key("display_name") {
            credential["display_name"] = json!(display_name(&body, false)?);
        }
        if body.contains_key("metadata") {
            credential["metadata"] =
                metadata(body.get("metadata"), Some(&credential["metadata"])).map_err(bad)?;
        }
        credential["updated_at"] = json!(rfc3339(unix_now()));
        credentials(&state, &vault).put(&id, &credential)?;
        Ok(public(&credential))
    })())
}

fn archive_one(state: &AppState, vault: &str, mut credential: Value) -> Reply {
    let id = credential["id"].as_str().unwrap_or_default().to_string();
    let now = rfc3339(unix_now());
    credential["archived_at"] = json!(now);
    credential["updated_at"] = json!(now);
    let _ = std::fs::remove_file(secret_path(state, vault, &id));
    credentials(state, vault).put(&id, &credential)?;
    super::webhooks::emit(state, "vault_credential.archived", &id);
    Ok(public(&credential))
}

pub async fn archive_credential(
    State(state): State<Arc<AppState>>,
    Path((vault, id)): Path<(String, String)>,
) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        let credential = credential_lookup(&state, &vault, &id)?;
        if !credential["archived_at"].is_null() {
            return Ok(public(&credential));
        }
        archive_one(&state, &vault, credential)
    })())
}

pub async fn delete_credential(
    State(state): State<Arc<AppState>>,
    Path((vault, id)): Path<(String, String)>,
) -> Response {
    respond((|| {
        let _guard = records_lock().lock().unwrap_or_else(|p| p.into_inner());
        credential_lookup(&state, &vault, &id)?;
        credentials(&state, &vault).remove(&id);
        let _ = std::fs::remove_file(secret_path(&state, &vault, &id));
        super::webhooks::emit(&state, "vault_credential.deleted", &id);
        Ok(json!({"id": id, "type": "vault_credential_deleted"}))
    })())
}

/// Checks an `mcp_oauth` credential: refreshes it when it can, then asks
/// its MCP server to `initialize` with the token.
pub async fn validate_credential(
    State(state): State<Arc<AppState>>,
    Path((vault, id)): Path<(String, String)>,
) -> Response {
    let reply: Reply = async {
        let credential = credential_lookup(&state, &vault, &id)?;
        if credential["auth"]["type"] != "mcp_oauth" {
            return Err(bad("only an mcp_oauth credential is validated"));
        }
        if !credential["archived_at"].is_null() {
            return Err(bad(format!("credential '{id}' is archived")));
        }
        let has_refresh = read_secret(&state, &vault, &id).contains_key("refresh_token");
        let refreshed = if has_refresh && !credential["auth"]["refresh"].is_null() {
            refresh(&state, &vault, &credential).await
        } else {
            json!({"status": "no_refresh_token", "http_response": null})
        };
        let token = read_secret(&state, &vault, &id)
            .get("access_token")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let url = credential["auth"]["mcp_server_url"]
            .as_str()
            .unwrap_or_default();
        let initialize = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "orangu-server", "version": env!("CARGO_PKG_VERSION")},
        }});
        let probe = reqwest::Client::new()
            .post(url)
            .bearer_auth(&token)
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .json(&initialize)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await;
        let (status, probe) = match probe {
            Ok(response) => {
                let (code, shown, _) = http_response(response).await;
                let status = match code {
                    200..=299 => "valid",
                    401 | 403 => "invalid",
                    _ => "unknown",
                };
                (
                    status,
                    json!({"method": "initialize", "http_response": shown}),
                )
            }
            Err(_) => (
                "unknown",
                json!({"method": "initialize", "http_response": null}),
            ),
        };
        Ok(json!({
            "type": "vault_credential_validation", "credential_id": id, "vault_id": vault,
            "status": status, "has_refresh_token": has_refresh,
            "refresh": refreshed, "mcp_probe": probe,
            "validated_at": rfc3339(unix_now()),
        }))
    }
    .await;
    respond(reply)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// URLs match whatever their host's case, default port or trailing
    /// slash; a different path or port does not.
    #[test]
    fn mcp_urls_match_when_normalized() {
        let n = |u: &str| normalized(u).unwrap();
        assert_eq!(
            n("HTTPS://MCP.Example.com:443/mcp/"),
            n("https://mcp.example.com/mcp")
        );
        assert_ne!(
            n("https://mcp.example.com/mcp"),
            n("https://mcp.example.com/other")
        );
        assert_ne!(
            n("https://mcp.example.com/mcp"),
            n("https://mcp.example.com:8443/mcp")
        );
    }

    /// Secrets never reach what is shown; an environment variable's
    /// locations default to both, and an object given defaults to neither.
    #[test]
    fn a_credentials_secrets_are_kept_apart_from_what_is_shown() {
        let (shown, secret) = auth(
            &json!({"type": "environment_variable", "secret_name": "API_KEY",
                    "secret_value": "s3cret", "networking": {"type": "unrestricted"}}),
            None,
        )
        .unwrap();
        assert!(!shown.to_string().contains("s3cret"));
        assert_eq!(secret["secret_value"], "s3cret");
        assert_eq!(
            shown["injection_location"],
            json!({"header": true, "body": true})
        );
        let (shown, _) = auth(
            &json!({"type": "environment_variable", "secret_name": "K", "secret_value": "v",
                    "networking": {"type": "limited", "allowed_hosts": ["api.example.com"]},
                    "injection_location": {"header": true}}),
            None,
        )
        .unwrap();
        assert_eq!(
            shown["injection_location"],
            json!({"header": true, "body": false})
        );
        let updated = injection(
            Some(&json!({"body": true})),
            Some(&shown["injection_location"]),
        );
        assert_eq!(updated.unwrap(), json!({"header": true, "body": true}));
        assert!(
            injection(
                Some(&json!({"header": false})),
                Some(&shown["injection_location"])
            )
            .is_err()
        );
        let (oauth, secret) = auth(
            &json!({"type": "mcp_oauth", "mcp_server_url": "https://m.example/mcp",
                    "access_token": "at", "refresh": {"refresh_token": "rt", "client_id": "c",
                    "token_endpoint": "https://m.example/token",
                    "token_endpoint_auth": {"type": "client_secret_post", "client_secret": "cs"}}}),
            None,
        )
        .unwrap();
        assert!(!oauth.to_string().contains("\"rt\"") && !oauth.to_string().contains("\"cs\""));
        assert_eq!(
            oauth["refresh"]["token_endpoint_auth"],
            json!({"type": "client_secret_post"})
        );
        assert_eq!(
            (
                secret["access_token"].as_str(),
                secret["client_secret"].as_str()
            ),
            (Some("at"), Some("cs"))
        );
        assert!(
            auth(
                &json!({"type": "environment_variable", "secret_name": "1BAD",
                             "secret_value": "v", "networking": {"type": "unrestricted"}}),
                None
            )
            .is_err()
        );
        assert!(auth(&json!({"type": "static_bearer", "token": "t"}), None).is_err());
    }
}
