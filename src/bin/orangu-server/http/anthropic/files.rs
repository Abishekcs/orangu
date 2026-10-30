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

//! Anthropic's Files API: `POST /v1/files`, `GET /v1/files`,
//! `GET /v1/files/{id}`, `GET /v1/files/{id}/content` and
//! `DELETE /v1/files/{id}`, over a directory — `<api_data>/files/`, each
//! file as `<id>.bin` beside its metadata in `<id>.json`.
//!
//! A `document` or `image` block whose source is `{"type": "file",
//! "file_id": …}` reads the stored file — see [`resolve_sources`].

use axum::{
    Json,
    extract::{Multipart, Path, Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

use super::{AnthropicMessage, error_response, rfc3339};
use crate::http::AppState;

/// The largest file `POST /v1/files` accepts, as Anthropic's own limit.
pub const UPLOAD_LIMIT: usize = 500 * 1024 * 1024;

#[derive(Serialize, Deserialize, Clone)]
pub struct FileMeta {
    pub id: String,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: u64,
    /// Unix seconds.
    pub created: u64,
    /// Unix nanoseconds, which is what orders the listing: newest first,
    /// even among files stored within the same second.
    #[serde(default)]
    pub created_nanos: u128,
    /// The session whose `/mnt/session/outputs` the file came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// Now, in Unix nanoseconds.
pub(super) fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

impl FileMeta {
    fn json(&self) -> Value {
        let mut file = json!({
            "id": self.id,
            "type": "file",
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "created_at": rfc3339(self.created),
            "downloadable": true,
        });
        if let Some(scope) = &self.scope {
            file["scope"] = json!({"id": scope, "type": "session"});
        }
        file
    }
}

/// Whether `id` is one this server could have issued — which is also what
/// keeps a request from naming a path outside the store.
pub(super) fn valid_id(id: &str, prefix: &str) -> bool {
    id.strip_prefix(prefix)
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// A fresh id: `prefix`, then a random 128-bit value in hex.
pub(super) fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub fn new(state: &AppState) -> Self {
        Self::at(&state.api_data)
    }

    /// The store under an `api_data` directory.
    pub fn at(api_data: &std::path::Path) -> Self {
        Self {
            root: api_data.join("files"),
        }
    }

    pub(super) fn data_path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.bin"))
    }

    fn meta_path(&self, id: &str) -> PathBuf {
        self.root.join(format!("{id}.json"))
    }

    pub fn get(&self, id: &str) -> Option<FileMeta> {
        if !valid_id(id, "file_") {
            return None;
        }
        let text = std::fs::read_to_string(self.meta_path(id)).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn read(&self, id: &str) -> Option<(FileMeta, Vec<u8>)> {
        let meta = self.get(id)?;
        let bytes = std::fs::read(self.data_path(id)).ok()?;
        Some((meta, bytes))
    }

    /// Every stored file, newest first.
    pub fn list(&self) -> Vec<FileMeta> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut files: Vec<FileMeta> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                self.get(name.strip_suffix(".json")?)
            })
            .collect();
        files.sort_by(|a, b| (b.created_nanos, &b.id).cmp(&(a.created_nanos, &a.id)));
        files
    }

    /// Stores a copy of the file at `src` under `filename`.
    pub(super) async fn save_copy(
        &self,
        filename: &str,
        src: &std::path::Path,
    ) -> std::io::Result<FileMeta> {
        self.save_scoped(filename, src, None).await
    }

    /// [`save_copy`](Self::save_copy), the file belonging to the session
    /// `scope` names.
    pub(super) async fn save_scoped(
        &self,
        filename: &str,
        src: &std::path::Path,
        scope: Option<&str>,
    ) -> std::io::Result<FileMeta> {
        tokio::fs::create_dir_all(&self.root).await?;
        let id = new_id("file_");
        let partial = self.root.join(format!("{id}.part"));
        let size = tokio::fs::copy(src, &partial).await?;
        let mut head = vec![0u8; 8192];
        let n = {
            use tokio::io::AsyncReadExt as _;
            let mut file = tokio::fs::File::open(&partial).await?;
            file.read(&mut head).await?
        };
        head.truncate(n);
        let meta = FileMeta {
            id: id.clone(),
            filename: filename.to_string(),
            mime_type: mime_of(&head, filename),
            size_bytes: size,
            created: super::unix_now(),
            created_nanos: now_nanos(),
            scope: scope.map(str::to_string),
        };
        tokio::fs::rename(&partial, self.data_path(&id)).await?;
        tokio::fs::write(
            self.meta_path(&id),
            serde_json::to_string(&meta).unwrap_or_default(),
        )
        .await?;
        Ok(meta)
    }

    pub fn delete(&self, id: &str) -> bool {
        if self.get(id).is_none() {
            return false;
        }
        // Metadata first: a file without it is invisible, one without its
        // bytes would be listed and fail to read.
        let _ = std::fs::remove_file(self.meta_path(id));
        let _ = std::fs::remove_file(self.data_path(id));
        true
    }
}

/// `limit`, `after_id` and `before_id`, as every Anthropic list endpoint
/// takes them.
#[derive(Deserialize)]
pub struct ListQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub after_id: Option<String>,
    #[serde(default)]
    pub before_id: Option<String>,
    /// Only the files of this session's `/mnt/session/outputs`.
    #[serde(default)]
    pub scope_id: Option<String>,
}

/// One page of `items` (already in listing order) in Anthropic's list
/// shape. `after_id` pages forward from an id, `before_id` back from one;
/// `limit` is 20 by default and at most 1000.
pub(super) fn page<T>(
    items: Vec<T>,
    id: impl Fn(&T) -> &str,
    query: &ListQuery,
    json: impl Fn(&T) -> Value,
) -> Value {
    let limit = query.limit.unwrap_or(20).clamp(1, 1000);
    let position = |wanted: &str| items.iter().position(|item| id(item) == wanted);
    let (start, end) = match (&query.after_id, &query.before_id) {
        (Some(after), _) => {
            let start = position(after).map_or(items.len(), |at| at + 1);
            (start, (start + limit).min(items.len()))
        }
        (None, Some(before)) => {
            let end = position(before).unwrap_or(0);
            (end.saturating_sub(limit), end)
        }
        (None, None) => (0, limit.min(items.len())),
    };
    let has_more = if query.before_id.is_some() && query.after_id.is_none() {
        start > 0
    } else {
        end < items.len()
    };
    let window = &items[start..end];
    json!({
        "data": window.iter().map(&json).collect::<Vec<_>>(),
        "has_more": has_more,
        "first_id": window.first().map(&id),
        "last_id": window.last().map(&id),
    })
}

pub async fn upload(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> Response {
    let store = FileStore::new(&state);
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "the upload carries no `file` field",
                );
            }
            Err(err) => return error_response(err.status(), err.body_text()),
        };
        if field.name() != Some("file") {
            continue;
        }
        let filename = field.file_name().unwrap_or("upload").to_string();
        let declared = field
            .content_type()
            .filter(|t| *t != "application/octet-stream")
            .map(str::to_string);
        return match store_field(&store, field, filename, declared).await {
            Ok(meta) => Json(meta.json()).into_response(),
            Err(rejection) => *rejection,
        };
    }
}

/// Streams one upload to disk, so a large file is never whole in memory, and
/// writes its metadata once the bytes are all there.
async fn store_field(
    store: &FileStore,
    mut field: axum::extract::multipart::Field<'_>,
    filename: String,
    declared: Option<String>,
) -> Result<FileMeta, Box<Response>> {
    let internal = |err: std::io::Error| {
        Box::new(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("storing the file: {err}"),
        ))
    };
    tokio::fs::create_dir_all(&store.root)
        .await
        .map_err(internal)?;
    let id = new_id("file_");
    let partial = store.root.join(format!("{id}.part"));
    let mut out = tokio::fs::File::create(&partial).await.map_err(internal)?;
    let mut size = 0u64;
    let mut head = Vec::new();
    loop {
        match field.chunk().await {
            Ok(Some(chunk)) => {
                if head.len() < 8192 {
                    head.extend_from_slice(&chunk[..chunk.len().min(8192 - head.len())]);
                }
                size += chunk.len() as u64;
                out.write_all(&chunk).await.map_err(internal)?;
            }
            Ok(None) => break,
            Err(err) => {
                drop(out);
                let _ = tokio::fs::remove_file(&partial).await;
                return Err(Box::new(error_response(err.status(), err.body_text())));
            }
        }
    }
    out.flush().await.map_err(internal)?;
    drop(out);
    let mime_type = declared.unwrap_or_else(|| mime_of(&head, &filename));
    let meta = FileMeta {
        id: id.clone(),
        filename,
        mime_type,
        size_bytes: size,
        created: super::unix_now(),
        created_nanos: now_nanos(),
        scope: None,
    };
    tokio::fs::rename(&partial, store.data_path(&id))
        .await
        .map_err(internal)?;
    let text = serde_json::to_string(&meta).unwrap_or_default();
    tokio::fs::write(store.meta_path(&id), text)
        .await
        .map_err(internal)?;
    Ok(meta)
}

/// A file's type from its first bytes, or failing that its name.
fn mime_of(head: &[u8], filename: &str) -> String {
    infer::get(head)
        .map(|kind| kind.mime_type().to_string())
        .or_else(|| {
            mime_guess::from_path(filename)
                .first()
                .map(|m| m.to_string())
        })
        .unwrap_or_else(|| "application/octet-stream".to_string())
}

fn not_found(id: &str) -> Response {
    error_response(StatusCode::NOT_FOUND, format!("file '{id}' does not exist"))
}

pub async fn list(State(state): State<Arc<AppState>>, Query(query): Query<ListQuery>) -> Response {
    let mut files = FileStore::new(&state).list();
    if let Some(scope) = &query.scope_id {
        files.retain(|f| f.scope.as_deref() == Some(scope.as_str()));
    }
    Json(page(files, |f| f.id.as_str(), &query, FileMeta::json)).into_response()
}

pub async fn retrieve(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match FileStore::new(&state).get(&id) {
        Some(meta) => Json(meta.json()).into_response(),
        None => not_found(&id),
    }
}

pub async fn content(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = FileStore::new(&state);
    let Some(meta) = store.get(&id) else {
        return not_found(&id);
    };
    let Ok(file) = tokio::fs::File::open(store.data_path(&id)).await else {
        return not_found(&id);
    };
    let body = axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(file));
    let mut response = body.into_response();
    let headers = response.headers_mut();
    if let Ok(value) = HeaderValue::from_str(&meta.mime_type) {
        headers.insert(header::CONTENT_TYPE, value);
    }
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(meta.size_bytes));
    let safe: String = meta
        .filename
        .chars()
        .filter(|c| c.is_ascii_graphic() && *c != '"' && *c != '\\' || *c == ' ')
        .collect();
    if let Ok(value) = HeaderValue::from_str(&format!("attachment; filename=\"{safe}\"")) {
        headers.insert(header::CONTENT_DISPOSITION, value);
    }
    response
}

pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if FileStore::new(&state).delete(&id) {
        Json(json!({"id": id, "type": "file_deleted"})).into_response()
    } else {
        not_found(&id)
    }
}

/// The source a stored file stands for in a `document` or `image` block: a
/// PDF's or a text file's text, or a picture's bytes.
/// A PDF as an inline `text` source: its text, and in `pdf_pages` each
/// page's, which a cited document is located by.
pub(super) fn pdf_source(bytes: &[u8]) -> anyhow::Result<Value> {
    let pages = orangu::pdf::extract_pages(bytes)?;
    let text = pages.concat().trim().to_string();
    Ok(json!({"type": "text", "media_type": "text/plain", "data": text, "pdf_pages": pages}))
}

fn inline_source(meta: &FileMeta, bytes: Vec<u8>) -> Result<Value, String> {
    use base64::Engine as _;
    let mime = meta.mime_type.as_str();
    if mime == "application/pdf" {
        return pdf_source(&bytes)
            .map_err(|err| format!("file '{}' is not a readable PDF: {err}", meta.id));
    }
    if mime.starts_with("image/") {
        return Ok(json!({
            "type": "base64", "media_type": mime,
            "data": base64::engine::general_purpose::STANDARD.encode(bytes),
        }));
    }
    match String::from_utf8(bytes) {
        Ok(text) => Ok(json!({"type": "text", "media_type": "text/plain", "data": text})),
        Err(_) => Err(format!(
            "file '{}' ({mime}) is neither text, a PDF nor a picture",
            meta.id
        )),
    }
}

/// Replaces every `{"type": "file", "file_id": …}` source in `messages` —
/// including inside `tool_result` content — with what the stored file holds,
/// so the rest of the request reads it like an inline source. Reading and
/// PDF extraction run off the async runtime.
pub(super) async fn resolve_sources(
    state: &AppState,
    messages: &mut [AnthropicMessage],
) -> Result<(), Box<Response>> {
    fn visit(blocks: &mut Value, wanted: &mut Vec<String>, apply: Option<&dyn Fn(&str) -> Value>) {
        let Some(blocks) = blocks.as_array_mut() else {
            return;
        };
        for block in blocks {
            if let Some(content) = block.get_mut("content") {
                visit(content, wanted, apply);
            }
            let Some(source) = block.get_mut("source") else {
                continue;
            };
            if source.get("type").and_then(Value::as_str) != Some("file") {
                continue;
            }
            let Some(id) = source.get("file_id").and_then(Value::as_str) else {
                continue;
            };
            match apply {
                None => wanted.push(id.to_string()),
                Some(apply) => *source = apply(id),
            }
        }
    }
    let mut wanted = Vec::new();
    for message in messages.iter_mut() {
        visit(&mut message.content, &mut wanted, None);
    }
    if wanted.is_empty() {
        return Ok(());
    }
    wanted.sort();
    wanted.dedup();
    let store = FileStore::new(state);
    let resolved = tokio::task::spawn_blocking(move || {
        wanted
            .into_iter()
            .map(|id| {
                let (meta, bytes) = store
                    .read(&id)
                    .ok_or_else(|| format!("file '{id}' does not exist"))?;
                Ok((id, inline_source(&meta, bytes)?))
            })
            .collect::<Result<std::collections::HashMap<String, Value>, String>>()
    })
    .await
    .unwrap_or_else(|err| Err(err.to_string()))
    .map_err(|message| Box::new(error_response(StatusCode::BAD_REQUEST, message)))?;
    let apply = |id: &str| resolved[id].clone();
    for message in messages.iter_mut() {
        visit(&mut message.content, &mut Vec::new(), Some(&apply));
    }
    Ok(())
}

/// The most bytes a `document` block's URL may return.
const MAX_DOCUMENT_BYTES: usize = 32 * 1024 * 1024;

/// What a fetched document holds, as an inline source: a PDF's text, or
/// text itself. `kind` is the response's Content-Type.
fn fetched_source(url: &str, kind: &str, bytes: Vec<u8>) -> Result<Value, String> {
    let pdf = kind.starts_with("application/pdf") || bytes.starts_with(b"%PDF-");
    if pdf {
        return pdf_source(&bytes).map_err(|err| format!("{url} is not a readable PDF: {err}"));
    }
    match String::from_utf8(bytes) {
        Ok(text) => Ok(json!({"type": "text", "media_type": "text/plain", "data": text})),
        Err(_) => Err(format!("{url} ({kind}) is neither text nor a PDF")),
    }
}

/// The document at `url`, fetched behind [`crate::http::public_net`]'s
/// guard, as an inline source.
async fn fetch_document(url: &str) -> Result<Value, String> {
    let parsed = reqwest::Url::parse(url).map_err(|err| format!("{url}: {err}"))?;
    crate::http::public_net::check(&parsed)?;
    let client = crate::http::public_net::guarded(reqwest::Client::builder())
        .timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!("orangu-server/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|err| err.to_string())?;
    let fail = |err: reqwest::Error| {
        let mut reason = err.to_string();
        let mut source = std::error::Error::source(&err);
        while let Some(cause) = source {
            reason = format!("{reason}: {cause}");
            source = cause.source();
        }
        format!("{url} could not be fetched: {reason}")
    };
    let mut response = client
        .get(parsed)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(fail)?;
    let kind = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(fail)? {
        bytes.extend_from_slice(&chunk);
        if bytes.len() > MAX_DOCUMENT_BYTES {
            return Err(format!(
                "{url} is larger than the {} MB this server takes",
                MAX_DOCUMENT_BYTES / (1024 * 1024)
            ));
        }
    }
    let url = url.to_string();
    tokio::task::spawn_blocking(move || fetched_source(&url, &kind, bytes))
        .await
        .unwrap_or_else(|err| Err(err.to_string()))
}

/// Replaces every `document` block's `{"type": "url", "url": …}` source in
/// `messages` — including inside `tool_result` content — with what the URL
/// holds. A picture's URL is left for the vision path to fetch.
pub(super) async fn resolve_document_urls(
    messages: &mut [AnthropicMessage],
) -> Result<(), Box<Response>> {
    fn visit(blocks: &mut Value, wanted: &mut Vec<String>, apply: Option<&dyn Fn(&str) -> Value>) {
        let Some(blocks) = blocks.as_array_mut() else {
            return;
        };
        for block in blocks {
            if let Some(content) = block.get_mut("content") {
                visit(content, wanted, apply);
            }
            if block.get("type").and_then(Value::as_str) != Some("document") {
                continue;
            }
            let Some(source) = block.get_mut("source") else {
                continue;
            };
            if source.get("type").and_then(Value::as_str) != Some("url") {
                continue;
            }
            let Some(url) = source.get("url").and_then(Value::as_str) else {
                continue;
            };
            match apply {
                None => wanted.push(url.to_string()),
                Some(apply) => *source = apply(url),
            }
        }
    }
    let mut wanted = Vec::new();
    for message in messages.iter_mut() {
        visit(&mut message.content, &mut wanted, None);
    }
    if wanted.is_empty() {
        return Ok(());
    }
    wanted.sort();
    wanted.dedup();
    let mut resolved = std::collections::HashMap::new();
    for url in wanted {
        let source = fetch_document(&url)
            .await
            .map_err(|message| Box::new(error_response(StatusCode::BAD_REQUEST, message)))?;
        resolved.insert(url, source);
    }
    let apply = |url: &str| resolved[url].clone();
    for message in messages.iter_mut() {
        visit(&mut message.content, &mut Vec::new(), Some(&apply));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fetched PDF is read as its text, fetched text as itself, and
    /// anything else is refused.
    #[test]
    fn a_fetched_document_is_read_by_what_it_is() {
        let text = fetched_source("u", "text/plain; charset=utf-8", b"hello".to_vec()).unwrap();
        assert_eq!(text["data"], "hello");
        assert!(fetched_source("u", "application/octet-stream", vec![0xff, 0xfe]).is_err());
        assert!(fetched_source("u", "application/pdf", b"%PDF-1.4 broken".to_vec()).is_err());
    }

    /// A document's URL on this machine is refused before anything is sent.
    #[tokio::test]
    async fn a_document_on_this_machine_is_not_fetched() {
        let mut messages = vec![AnthropicMessage {
            role: "user".into(),
            content: json!([{"type": "document",
                             "source": {"type": "url", "url": "http://127.0.0.1:9/a.pdf"}}]),
        }];
        assert!(resolve_document_urls(&mut messages).await.is_err());
    }

    #[test]
    fn ids_name_nothing_outside_the_store() {
        assert!(valid_id("file_0123abc", "file_"));
        assert!(!valid_id("file_", "file_"));
        assert!(!valid_id("file_../../etc/passwd", "file_"));
        assert!(!valid_id("msgbatch_abc", "file_"));
        assert!(valid_id(&new_id("file_"), "file_"));
    }

    fn query(limit: usize, after: Option<&str>, before: Option<&str>) -> ListQuery {
        ListQuery {
            limit: Some(limit),
            after_id: after.map(str::to_string),
            before_id: before.map(str::to_string),
            scope_id: None,
        }
    }

    #[test]
    fn pages_walk_forward_and_back() {
        let items: Vec<String> = ["a", "b", "c", "d", "e"].map(String::from).to_vec();
        let pages = |query: &ListQuery| page(items.clone(), |s| s.as_str(), query, |s| json!(s));
        let first = pages(&query(2, None, None));
        assert_eq!(first["data"], json!(["a", "b"]));
        assert_eq!(first["has_more"], true);
        let next = pages(&query(2, Some("b"), None));
        assert_eq!(next["data"], json!(["c", "d"]));
        let last = pages(&query(2, Some("d"), None));
        assert_eq!(
            (last["data"].clone(), last["has_more"].clone()),
            (json!(["e"]), json!(false))
        );
        let back = pages(&query(2, None, Some("d")));
        assert_eq!(back["data"], json!(["b", "c"]));
        assert_eq!(back["has_more"], true);
        assert_eq!(back["first_id"], "b");
    }

    #[test]
    fn stored_text_and_pdfs_become_text_sources() {
        let meta = |mime: &str| FileMeta {
            id: "file_x".into(),
            filename: "x".into(),
            mime_type: mime.into(),
            size_bytes: 0,
            created: 0,
            created_nanos: 0,
            scope: None,
        };
        let text = inline_source(&meta("text/plain"), b"hello".to_vec()).unwrap();
        assert_eq!(text["data"], "hello");
        let pdf = orangu::pdf::encode_text("t", "Inside the PDF.").unwrap();
        let from_pdf = inline_source(&meta("application/pdf"), pdf).unwrap();
        assert!(
            from_pdf["data"]
                .as_str()
                .unwrap()
                .contains("Inside the PDF.")
        );
        let image = inline_source(&meta("image/png"), vec![1, 2, 3]).unwrap();
        assert_eq!(image["type"], "base64");
        assert!(inline_source(&meta("application/zip"), vec![0xff, 0xfe]).is_err());
    }
}
