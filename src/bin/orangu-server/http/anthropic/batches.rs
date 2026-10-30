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

//! Anthropic's Message Batches API: `POST /v1/messages/batches`, `GET` for
//! the list and one batch, `POST …/{id}/cancel`, `GET …/{id}/results` and
//! `DELETE …/{id}`.
//!
//! A batch lives in `<api_data>/batches/<id>/`: `batch.json` (status and
//! counts), `requests.jsonl` (as submitted) and `results.jsonl` (one line per
//! finished request, in the order they finished). A worker task runs each
//! batch's requests through the same path a non-streamed `/v1/messages`
//! request takes, as many at once as the server has slots, and a batch still
//! in progress when the server stops is picked up again when it starts.

use axum::{
    Json,
    extract::{Path, Query, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::files::{ListQuery, new_id, now_nanos, page, valid_id};
use super::{
    MessagesRequest, error_of, error_response, json_rejection, rfc3339, run_one, unix_now,
};
use crate::http::AppState;

/// The largest batch body accepted, as Anthropic's own limit.
pub const REQUEST_BODY_LIMIT: usize = 256 * 1024 * 1024;

/// The most requests one batch may hold, as Anthropic's own limit.
const MAX_REQUESTS: usize = 100_000;

/// How long a batch may run before its unfinished requests expire.
const EXPIRES_AFTER: u64 = 24 * 60 * 60;

/// How long a request the queue turned away waits before it is tried again.
const OVERLOADED_RETRY: Duration = Duration::from_secs(2);

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
enum Status {
    InProgress,
    Canceling,
    Ended,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug, PartialEq)]
struct Counts {
    processing: usize,
    succeeded: usize,
    errored: usize,
    canceled: usize,
    expired: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Batch {
    id: String,
    processing_status: Status,
    request_counts: Counts,
    created: u64,
    /// Orders the listing, as `FileMeta::created_nanos` does.
    #[serde(default)]
    created_nanos: u128,
    expires: u64,
    ended: Option<u64>,
    cancel_initiated: Option<u64>,
}

impl Batch {
    fn json(&self) -> Value {
        json!({
            "id": self.id,
            "type": "message_batch",
            "processing_status": self.processing_status,
            "request_counts": self.request_counts,
            "created_at": rfc3339(self.created),
            "expires_at": rfc3339(self.expires),
            "ended_at": self.ended.map(rfc3339),
            "cancel_initiated_at": self.cancel_initiated.map(rfc3339),
            "archived_at": null,
            "results_url": (self.processing_status == Status::Ended)
                .then(|| format!("/v1/messages/batches/{}/results", self.id)),
        })
    }
}

/// Serialises every read-modify-write of a `batch.json`, so a cancel and a
/// worker's count update cannot lose each other's change.
fn batch_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// The cancel flag of every batch a worker is running.
fn cancel_flags() -> &'static Mutex<HashMap<String, Arc<AtomicBool>>> {
    static FLAGS: std::sync::OnceLock<Mutex<HashMap<String, Arc<AtomicBool>>>> =
        std::sync::OnceLock::new();
    FLAGS.get_or_init(Default::default)
}

struct BatchStore {
    root: PathBuf,
}

impl BatchStore {
    fn new(state: &AppState) -> Self {
        Self {
            root: state.api_data.join("batches"),
        }
    }

    fn dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    fn get(&self, id: &str) -> Option<Batch> {
        if !valid_id(id, "msgbatch_") {
            return None;
        }
        let text = std::fs::read_to_string(self.dir(id).join("batch.json")).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn put(&self, batch: &Batch) -> std::io::Result<()> {
        let dir = self.dir(&batch.id);
        let partial = dir.join("batch.json.part");
        std::fs::write(&partial, serde_json::to_vec(batch)?)?;
        std::fs::rename(partial, dir.join("batch.json"))
    }

    /// Applies `change` to a stored batch under [`batch_lock`].
    fn update(&self, id: &str, change: impl FnOnce(&mut Batch)) -> Option<Batch> {
        let _guard = batch_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut batch = self.get(id)?;
        change(&mut batch);
        self.put(&batch).ok()?;
        Some(batch)
    }

    /// Every stored batch, newest first.
    fn list(&self) -> Vec<Batch> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut batches: Vec<Batch> = entries
            .flatten()
            .filter_map(|entry| self.get(entry.file_name().to_str()?))
            .collect();
        batches.sort_by(|a, b| (b.created_nanos, &b.id).cmp(&(a.created_nanos, &a.id)));
        batches
    }

    /// The custom ids `results.jsonl` already holds a line for.
    fn finished(&self, id: &str) -> HashSet<String> {
        let Ok(text) = std::fs::read_to_string(self.dir(id).join("results.jsonl")) else {
            return HashSet::new();
        };
        text.lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter_map(|line| line.get("custom_id")?.as_str().map(str::to_string))
            .collect()
    }

    fn requests(&self, id: &str) -> Vec<(String, Value)> {
        let Ok(text) = std::fs::read_to_string(self.dir(id).join("requests.jsonl")) else {
            return Vec::new();
        };
        text.lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter_map(|line| {
                let custom_id = line.get("custom_id")?.as_str()?.to_string();
                Some((custom_id, line.get("params")?.clone()))
            })
            .collect()
    }

    /// Appends one result and moves it from `processing` to its outcome's
    /// count, together under [`batch_lock`] so the two never disagree.
    fn record(&self, id: &str, custom_id: &str, result: Value) {
        let kind = result
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("errored")
            .to_string();
        let _guard = batch_lock().lock().unwrap_or_else(|p| p.into_inner());
        let line = json!({"custom_id": custom_id, "result": result});
        let written = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir(id).join("results.jsonl"))
            .and_then(|mut file| writeln!(file, "{line}"));
        if let Err(err) = written {
            log::warn!("orangu-server: batch {id}: writing a result: {err}");
            return;
        }
        let Some(mut batch) = self.get(id) else {
            return;
        };
        let counts = &mut batch.request_counts;
        counts.processing = counts.processing.saturating_sub(1);
        match kind.as_str() {
            "succeeded" => counts.succeeded += 1,
            "canceled" => counts.canceled += 1,
            "expired" => counts.expired += 1,
            _ => counts.errored += 1,
        }
        let _ = self.put(&batch);
    }
}

#[derive(Deserialize)]
pub struct CreateRequest {
    requests: Vec<BatchRequest>,
}

#[derive(Deserialize)]
struct BatchRequest {
    custom_id: String,
    params: Value,
}

/// What is wrong with a batch as submitted, if anything — checked before
/// any of it is stored, so a batch that is accepted can be run.
fn validate(requests: &[BatchRequest]) -> Result<(), String> {
    if requests.is_empty() {
        return Err("a batch needs at least one request".into());
    }
    if requests.len() > MAX_REQUESTS {
        return Err(format!("a batch holds at most {MAX_REQUESTS} requests"));
    }
    let mut seen = HashSet::new();
    for (index, request) in requests.iter().enumerate() {
        let id = &request.custom_id;
        if id.is_empty()
            || id.len() > 64
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(format!(
                "requests[{index}].custom_id must be 1 to 64 letters, digits, '_' or '-'"
            ));
        }
        if !seen.insert(id.as_str()) {
            return Err(format!("requests[{index}].custom_id '{id}' is used twice"));
        }
        let params: MessagesRequest = serde_json::from_value(request.params.clone())
            .map_err(|err| format!("requests[{index}].params: {err}"))?;
        if params.stream {
            return Err(format!(
                "requests[{index}].params: a batched request cannot be streamed"
            ));
        }
    }
    Ok(())
}

pub async fn create(
    State(state): State<Arc<AppState>>,
    req: Result<Json<CreateRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(req) => req,
        Err(rejection) => return json_rejection(rejection),
    };
    if let Err(message) = validate(&req.requests) {
        return error_response(StatusCode::BAD_REQUEST, message);
    }
    let store = BatchStore::new(&state);
    let created = unix_now();
    let batch = Batch {
        id: new_id("msgbatch_"),
        processing_status: Status::InProgress,
        request_counts: Counts {
            processing: req.requests.len(),
            ..Default::default()
        },
        created,
        created_nanos: now_nanos(),
        expires: created + EXPIRES_AFTER,
        ended: None,
        cancel_initiated: None,
    };
    let stored = (|| -> std::io::Result<()> {
        let dir = store.dir(&batch.id);
        std::fs::create_dir_all(&dir)?;
        let mut lines = String::new();
        for request in &req.requests {
            lines.push_str(
                &json!({"custom_id": request.custom_id, "params": request.params}).to_string(),
            );
            lines.push('\n');
        }
        std::fs::write(dir.join("requests.jsonl"), lines)?;
        std::fs::write(dir.join("results.jsonl"), "")?;
        store.put(&batch)
    })();
    if let Err(err) = stored {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("storing the batch: {err}"),
        );
    }
    spawn_worker(state.clone(), batch.id.clone());
    Json(batch.json()).into_response()
}

fn not_found(id: &str) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        format!("message batch '{id}' does not exist"),
    )
}

pub async fn list(State(state): State<Arc<AppState>>, Query(query): Query<ListQuery>) -> Response {
    let batches = BatchStore::new(&state).list();
    Json(page(batches, |b| b.id.as_str(), &query, Batch::json)).into_response()
}

pub async fn retrieve(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match BatchStore::new(&state).get(&id) {
        Some(batch) => Json(batch.json()).into_response(),
        None => not_found(&id),
    }
}

pub async fn cancel(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = BatchStore::new(&state);
    let Some(batch) = store.update(&id, |batch| {
        if batch.processing_status == Status::InProgress {
            batch.processing_status = Status::Canceling;
            batch.cancel_initiated = Some(unix_now());
        }
    }) else {
        return not_found(&id);
    };
    if let Some(flag) = cancel_flags()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&id)
    {
        flag.store(true, Ordering::Relaxed);
    }
    Json(batch.json()).into_response()
}

pub async fn results(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = BatchStore::new(&state);
    let Some(batch) = store.get(&id) else {
        return not_found(&id);
    };
    if batch.processing_status != Status::Ended {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("message batch '{id}' has not ended; its results are not ready"),
        );
    }
    let Ok(file) = tokio::fs::File::open(store.dir(&id).join("results.jsonl")).await else {
        return not_found(&id);
    };
    let mut response =
        axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(file)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-jsonl"),
    );
    response
}

pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = BatchStore::new(&state);
    let Some(batch) = store.get(&id) else {
        return not_found(&id);
    };
    if batch.processing_status != Status::Ended {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("message batch '{id}' is still running; cancel it first"),
        );
    }
    if let Err(err) = std::fs::remove_dir_all(store.dir(&id)) {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("deleting the batch: {err}"),
        );
    }
    Json(json!({"id": id, "type": "message_batch_deleted"})).into_response()
}

/// Starts a worker for every batch the store holds that has not ended —
/// called once at startup, so a restart resumes rather than strands them.
pub fn resume(state: Arc<AppState>) {
    for batch in BatchStore::new(&state).list() {
        if batch.processing_status != Status::Ended {
            log::info!("orangu-server: resuming message batch {}", batch.id);
            spawn_worker(state.clone(), batch.id);
        }
    }
}

fn spawn_worker(state: Arc<AppState>, id: String) {
    let flag = Arc::new(AtomicBool::new(false));
    cancel_flags()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(id.clone(), flag.clone());
    tokio::spawn(async move {
        run_batch(&state, &id, &flag).await;
        cancel_flags()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&id);
    });
}

/// Runs one request, waiting out a full queue rather than failing on it:
/// the whole point of a batch is that it can wait.
async fn run_request(state: &AppState, params: Value, flag: &AtomicBool, expires: u64) -> Value {
    let headers = HeaderMap::new();
    loop {
        if flag.load(Ordering::Relaxed) {
            return json!({"type": "canceled"});
        }
        if unix_now() >= expires {
            return json!({"type": "expired"});
        }
        // Validated at creation, so this parses.
        let req: MessagesRequest = match serde_json::from_value(params.clone()) {
            Ok(req) => req,
            Err(err) => {
                return json!({"type": "errored", "error": {"type": "error", "error": {
                    "type": "invalid_request_error", "message": err.to_string(),
                }}});
            }
        };
        match run_one(state, &headers, req).await {
            Ok(message) => return json!({"type": "succeeded", "message": message}),
            Err(response) if response.status() == StatusCode::SERVICE_UNAVAILABLE => {
                tokio::time::sleep(OVERLOADED_RETRY).await;
            }
            Err(response) => {
                return json!({"type": "errored", "error": error_of(*response).await});
            }
        }
    }
}

async fn run_batch(state: &Arc<AppState>, id: &str, flag: &Arc<AtomicBool>) {
    let store = BatchStore::new(state);
    let Some(batch) = store.get(id) else {
        return;
    };
    if batch.processing_status == Status::Canceling {
        flag.store(true, Ordering::Relaxed);
    }
    let finished = store.finished(id);
    let pending: Vec<(String, Value)> = store
        .requests(id)
        .into_iter()
        .filter(|(custom_id, _)| !finished.contains(custom_id))
        .collect();
    // As many at once as there are slots: each is one conversation, and a
    // request beyond that would only queue behind the others.
    let width = state.engine.slots.total().max(1);
    let gate = Arc::new(tokio::sync::Semaphore::new(width));
    let mut running = tokio::task::JoinSet::new();
    for (custom_id, params) in pending {
        let Ok(permit) = gate.clone().acquire_owned().await else {
            break;
        };
        let (state, flag, id) = (state.clone(), flag.clone(), id.to_string());
        let expires = batch.expires;
        running.spawn(async move {
            let result = run_request(&state, params, &flag, expires).await;
            BatchStore::new(&state).record(&id, &custom_id, result);
            drop(permit);
        });
    }
    while running.join_next().await.is_some() {}
    store.update(id, |batch| {
        batch.processing_status = Status::Ended;
        batch.ended = Some(unix_now());
    });
    log::info!("orangu-server: message batch {id} ended");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(custom_id: &str, params: Value) -> BatchRequest {
        BatchRequest {
            custom_id: custom_id.into(),
            params,
        }
    }

    #[test]
    fn a_batch_is_refused_before_it_is_stored() {
        let ok = json!({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]});
        assert!(validate(&[request("a", ok.clone()), request("b", ok.clone())]).is_ok());
        assert!(validate(&[]).is_err());
        assert!(validate(&[request("a", ok.clone()), request("a", ok.clone())]).is_err());
        assert!(validate(&[request("bad id!", ok.clone())]).is_err());
        let streamed = json!({"stream": true, "messages": []});
        assert!(validate(&[request("a", streamed)]).is_err());
        let malformed = json!({"messages": "nope"});
        assert!(validate(&[request("a", malformed)]).is_err());
    }

    #[test]
    fn the_listing_shape_names_results_only_once_ended() {
        let mut batch = Batch {
            id: "msgbatch_1".into(),
            processing_status: Status::InProgress,
            request_counts: Counts {
                processing: 2,
                ..Default::default()
            },
            created: 0,
            created_nanos: 0,
            expires: EXPIRES_AFTER,
            ended: None,
            cancel_initiated: None,
        };
        assert_eq!(batch.json()["processing_status"], "in_progress");
        assert!(batch.json()["results_url"].is_null());
        batch.processing_status = Status::Ended;
        assert_eq!(
            batch.json()["results_url"],
            "/v1/messages/batches/msgbatch_1/results"
        );
        assert_eq!(batch.json()["request_counts"]["processing"], 2);
    }

    #[test]
    fn recorded_results_move_out_of_processing() {
        let root = std::env::temp_dir().join(format!("orangu-batch-test-{}", std::process::id()));
        let store = BatchStore { root: root.clone() };
        let batch = Batch {
            id: "msgbatch_test".into(),
            processing_status: Status::InProgress,
            request_counts: Counts {
                processing: 3,
                ..Default::default()
            },
            created: 0,
            created_nanos: 0,
            expires: 1,
            ended: None,
            cancel_initiated: None,
        };
        std::fs::create_dir_all(store.dir(&batch.id)).unwrap();
        store.put(&batch).unwrap();
        store.record(&batch.id, "a", json!({"type": "succeeded", "message": {}}));
        store.record(&batch.id, "b", json!({"type": "errored", "error": {}}));
        store.record(&batch.id, "c", json!({"type": "canceled"}));
        let counts = store.get(&batch.id).unwrap().request_counts;
        assert_eq!(
            counts,
            Counts {
                processing: 0,
                succeeded: 1,
                errored: 1,
                canceled: 1,
                expired: 0
            }
        );
        assert_eq!(store.finished(&batch.id).len(), 3);
        let _ = std::fs::remove_dir_all(root);
    }
}
