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

//! The Anthropic-compatible Skills API: `POST`/`GET /v1/skills`,
//! `GET`/`DELETE /v1/skills/{id}`, and a skill's versions under
//! `/v1/skills/{id}/versions`.
//!
//! A skill lives in `<api_data>/skills/<id>/`: `skill.json`, and each
//! version in `versions/<version>/` — `version.json` and `files/<name>/`,
//! the uploaded folder with its `SKILL.md`. A request that lists a skill in
//! `container.skills` has that folder mounted read-only at `/skills/<name>`
//! in its code execution container, and the skill's name and description
//! told to the model.

use axum::{
    Json,
    extract::{Multipart, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

use super::files::{new_id, now_nanos, valid_id};
use super::{error_response, rfc3339};
use crate::http::AppState;

/// The largest skill upload accepted.
pub const UPLOAD_LIMIT: usize = 30 * 1024 * 1024;

/// How many skills one request may mount, as Anthropic's own limit.
const MAX_MOUNTED: usize = 8;

#[derive(Serialize, Deserialize, Clone)]
struct SkillMeta {
    id: String,
    display_name: String,
    /// The kebab-case name every version's `SKILL.md` carries.
    name: String,
    /// Unix seconds.
    created: u64,
    updated: u64,
    created_nanos: u128,
}

#[derive(Serialize, Deserialize, Clone)]
struct VersionMeta {
    /// Unix microseconds as a string, as Anthropic names a version.
    id: String,
    skill_id: String,
    name: String,
    description: String,
    created: u64,
}

impl SkillMeta {
    fn json(&self, latest: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "type": "skill",
            "display_name": self.display_name,
            "latest_version_id": latest,
            "source": {"type": "custom"},
            "created_at": rfc3339(self.created),
            "updated_at": rfc3339(self.updated),
        })
    }
}

impl VersionMeta {
    fn json(&self) -> Value {
        json!({
            "id": self.id,
            "type": "skill_version",
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "created_at": rfc3339(self.created),
        })
    }
}

struct Store {
    root: PathBuf,
}

impl Store {
    fn new(api_data: &std::path::Path) -> Self {
        Self {
            root: api_data.join("skills"),
        }
    }

    fn skill_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    fn get(&self, id: &str) -> Option<SkillMeta> {
        if !valid_id(id, "skill_") {
            return None;
        }
        let text = std::fs::read_to_string(self.skill_dir(id).join("skill.json")).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Every version of a skill, newest first.
    fn versions(&self, id: &str) -> Vec<VersionMeta> {
        let Ok(entries) = std::fs::read_dir(self.skill_dir(id).join("versions")) else {
            return Vec::new();
        };
        let mut out: Vec<VersionMeta> = entries
            .flatten()
            .filter_map(|e| {
                let text = std::fs::read_to_string(e.path().join("version.json")).ok()?;
                serde_json::from_str(&text).ok()
            })
            .collect();
        out.sort_by_key(|v| std::cmp::Reverse(v.id.parse::<u128>().unwrap_or(0)));
        out
    }

    fn version_dir(&self, id: &str, version: &str) -> Option<PathBuf> {
        version.bytes().all(|b| b.is_ascii_digit()).then_some(())?;
        let dir = self.skill_dir(id).join("versions").join(version);
        dir.join("version.json").exists().then_some(dir)
    }

    fn json(&self, skill: &SkillMeta) -> Value {
        let latest = self.versions(&skill.id).first().map(|v| v.id.clone());
        skill.json(latest.as_deref())
    }

    /// Every skill, newest first.
    fn list(&self) -> Vec<SkillMeta> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut out: Vec<SkillMeta> = entries
            .flatten()
            .filter_map(|e| self.get(&e.file_name().into_string().ok()?))
            .collect();
        out.sort_by(|a, b| (b.created_nanos, &b.id).cmp(&(a.created_nanos, &a.id)));
        out
    }
}

/// `limit` and `page`, as the Skills list endpoints take them.
#[derive(Deserialize)]
pub struct PageQuery {
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    page: Option<String>,
    #[serde(default)]
    source: Option<String>,
}

/// One page of `items`: those after the id `page` names, with `next_page`
/// naming the last one shown when more follow.
fn page<T>(
    items: Vec<T>,
    id: impl Fn(&T) -> &str,
    query: &PageQuery,
    json: impl Fn(&T) -> Value,
) -> Value {
    let limit = query.limit.unwrap_or(20).clamp(1, 1000);
    let start = query
        .page
        .as_deref()
        .and_then(|after| items.iter().position(|item| id(item) == after))
        .map_or(0, |at| at + 1);
    let end = (start + limit).min(items.len());
    let window = &items[start.min(end)..end];
    let has_more = end < items.len();
    json!({
        "data": window.iter().map(json).collect::<Vec<_>>(),
        "has_more": has_more,
        "next_page": if has_more { window.last().map(&id) } else { None },
    })
}

fn not_found(what: &str) -> Response {
    error_response(StatusCode::NOT_FOUND, format!("{what} does not exist"))
}

fn bad_request(message: impl Into<String>) -> Response {
    error_response(StatusCode::BAD_REQUEST, message)
}

/// An upload's files: each one's path inside the skill folder, and its bytes.
struct Upload {
    display_name: Option<String>,
    files: Vec<(String, Vec<u8>)>,
}

/// Reads the multipart form: `display_name`, and the `files` (or
/// `files[]`) parts, each named by its path.
async fn read_upload(mut multipart: Multipart) -> Result<Upload, Box<Response>> {
    let mut upload = Upload {
        display_name: None,
        files: Vec::new(),
    };
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(err) => return Err(Box::new(error_response(err.status(), err.body_text()))),
        };
        match field.name() {
            Some("display_name") => {
                let text = field
                    .text()
                    .await
                    .map_err(|err| error_response(err.status(), err.body_text()))?;
                upload.display_name = Some(text.trim().to_string()).filter(|t| !t.is_empty());
            }
            Some("files" | "files[]") => {
                let path = field.file_name().unwrap_or_default().to_string();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|err| error_response(err.status(), err.body_text()))?;
                upload.files.push((path, bytes.to_vec()));
            }
            _ => {}
        }
    }
    if upload.files.is_empty() {
        return Err(Box::new(bad_request("the upload carries no `files`")));
    }
    Ok(upload)
}

/// `SKILL.md`'s front matter: its `name` and `description`.
fn front_matter(text: &str) -> (Option<String>, Option<String>) {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (None, None);
    }
    let (mut name, mut description) = (None, None);
    let mut folded: Option<String> = None;
    for line in lines {
        if line.trim() == "---" {
            break;
        }
        // A `description: >` block: the indented lines after it.
        if let Some(text) = folded.as_mut() {
            if line.starts_with([' ', '\t']) {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(line.trim());
                continue;
            }
            description = folded.take().filter(|t| !t.is_empty());
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches(['"', '\'']).to_string();
        match key.trim() {
            "name" => name = Some(value).filter(|v| !v.is_empty()),
            "description" if matches!(value.as_str(), ">" | "|" | ">-" | "|-") => {
                folded = Some(String::new());
            }
            "description" => description = Some(value).filter(|v| !v.is_empty()),
            _ => {}
        }
    }
    if let Some(text) = folded {
        description = Some(text).filter(|t| !t.is_empty());
    }
    (name, description)
}

/// What an upload makes: the skill's name and description, and its files by
/// their path inside the folder.
struct Checked {
    name: String,
    description: String,
    files: Vec<(PathBuf, Vec<u8>)>,
}

/// Checks an upload is one folder with a `SKILL.md` at its root, every path
/// a plain relative one, and the front matter naming the skill.
fn check(upload: Upload) -> Result<Checked, String> {
    let total: usize = upload.files.iter().map(|(_, b)| b.len()).sum();
    if total > UPLOAD_LIMIT {
        return Err(format!(
            "a skill's files may be at most {UPLOAD_LIMIT} bytes"
        ));
    }
    let mut split = Vec::new();
    for (path, _) in &upload.files {
        let parts: Vec<String> = path.split('/').map(str::to_string).collect();
        if path.is_empty()
            || path.contains('\\')
            || parts
                .iter()
                .any(|p| p.is_empty() || *p == "." || *p == "..")
        {
            return Err(format!("'{path}' is not a relative path inside the skill"));
        }
        split.push(parts);
    }
    // One top-level folder for every file, or none at all.
    let folder = match split[0].as_slice() {
        [first, _, ..] if split.iter().all(|p| p.len() > 1 && p[0] == *first) => {
            Some(first.clone())
        }
        _ if split.iter().all(|p| p.len() == 1) => None,
        _ => return Err("every file must be inside the same top-level folder".into()),
    };
    let skip = usize::from(folder.is_some());
    let files: Vec<(PathBuf, Vec<u8>)> = split
        .iter()
        .zip(upload.files)
        .map(|(parts, (_, bytes))| (parts[skip..].iter().collect::<PathBuf>(), bytes))
        .collect();
    let skill_md = files
        .iter()
        .find(|(path, _)| path == std::path::Path::new("SKILL.md"))
        .ok_or("the skill's folder must have a SKILL.md at its root")?;
    let (name, description) = front_matter(&String::from_utf8_lossy(&skill_md.1));
    let name = name
        .or(folder)
        .ok_or("SKILL.md's front matter must give the skill a `name`")?;
    if name.len() > 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err(format!(
            "skill name '{name}' must be at most 64 lowercase letters, digits and hyphens"
        ));
    }
    let description =
        description.ok_or("SKILL.md's front matter must give the skill a `description`")?;
    if description.len() > 1024 {
        return Err("a skill's description may be at most 1024 characters".into());
    }
    Ok(Checked {
        name,
        description,
        files,
    })
}

/// Writes a checked upload as a new version of `skill_id`.
async fn write_version(
    store: &Store,
    skill_id: &str,
    checked: Checked,
) -> Result<VersionMeta, Box<Response>> {
    let internal = |err: std::io::Error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("storing the skill: {err}"),
        )
    };
    let versions = store.skill_dir(skill_id).join("versions");
    let latest = store
        .versions(skill_id)
        .first()
        .and_then(|v| v.id.parse::<u128>().ok())
        .unwrap_or(0);
    let id = (now_nanos() / 1000).max(latest + 1).to_string();
    let partial = versions.join(format!("{id}.part"));
    let files = partial.join("files").join(&checked.name);
    for (path, bytes) in &checked.files {
        let target = files.join(path);
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(internal)?;
        }
        tokio::fs::write(&target, bytes).await.map_err(internal)?;
    }
    let meta = VersionMeta {
        id: id.clone(),
        skill_id: skill_id.to_string(),
        name: checked.name,
        description: checked.description,
        created: super::unix_now(),
    };
    tokio::fs::write(
        partial.join("version.json"),
        serde_json::to_string(&meta).unwrap_or_default(),
    )
    .await
    .map_err(internal)?;
    tokio::fs::rename(&partial, versions.join(&id))
        .await
        .map_err(internal)?;
    Ok(meta)
}

async fn save_skill(store: &Store, skill: &SkillMeta) -> Result<(), Box<Response>> {
    tokio::fs::create_dir_all(store.skill_dir(&skill.id))
        .await
        .and(
            tokio::fs::write(
                store.skill_dir(&skill.id).join("skill.json"),
                serde_json::to_string(skill).unwrap_or_default(),
            )
            .await,
        )
        .map_err(|err| {
            Box::new(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("storing the skill: {err}"),
            ))
        })
}

pub async fn create(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    let upload = match read_upload(multipart).await {
        Ok(upload) => upload,
        Err(rejection) => return *rejection,
    };
    let display_name = upload.display_name.clone();
    let checked = match check(upload) {
        Ok(checked) => checked,
        Err(message) => return bad_request(message),
    };
    let store = Store::new(&state.api_data);
    let now = super::unix_now();
    let skill = SkillMeta {
        id: new_id("skill_"),
        display_name: display_name.unwrap_or_else(|| checked.name.clone()),
        name: checked.name.clone(),
        created: now,
        updated: now,
        created_nanos: now_nanos(),
    };
    if let Err(rejection) = save_skill(&store, &skill).await {
        return *rejection;
    }
    if let Err(rejection) = write_version(&store, &skill.id, checked).await {
        let _ = tokio::fs::remove_dir_all(store.skill_dir(&skill.id)).await;
        return *rejection;
    }
    Json(store.json(&skill)).into_response()
}

pub async fn list(State(state): State<Arc<AppState>>, Query(query): Query<PageQuery>) -> Response {
    let store = Store::new(&state.api_data);
    // Every skill here is the user's own.
    let skills = match query.source.as_deref() {
        None | Some("custom") => store.list(),
        Some(_) => Vec::new(),
    };
    Json(page(skills, |s| s.id.as_str(), &query, |s| store.json(s))).into_response()
}

pub async fn retrieve(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = Store::new(&state.api_data);
    match store.get(&id) {
        Some(skill) => Json(store.json(&skill)).into_response(),
        None => not_found(&format!("skill '{id}'")),
    }
}

pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = Store::new(&state.api_data);
    if store.get(&id).is_none() {
        return not_found(&format!("skill '{id}'"));
    }
    if !store.versions(&id).is_empty() {
        return bad_request(format!(
            "skill '{id}' still has versions: delete them first"
        ));
    }
    let _ = tokio::fs::remove_dir_all(store.skill_dir(&id)).await;
    Json(json!({"id": id, "type": "skill_deleted"})).into_response()
}

pub async fn create_version(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    multipart: Multipart,
) -> Response {
    let store = Store::new(&state.api_data);
    let Some(mut skill) = store.get(&id) else {
        return not_found(&format!("skill '{id}'"));
    };
    let checked = match read_upload(multipart).await.map(check) {
        Ok(Ok(checked)) => checked,
        Ok(Err(message)) => return bad_request(message),
        Err(rejection) => return *rejection,
    };
    if checked.name != skill.name {
        return bad_request(format!(
            "this upload's SKILL.md names '{}', but the skill is '{}'",
            checked.name, skill.name
        ));
    }
    let version = match write_version(&store, &id, checked).await {
        Ok(version) => version,
        Err(rejection) => return *rejection,
    };
    skill.updated = super::unix_now();
    if let Err(rejection) = save_skill(&store, &skill).await {
        return *rejection;
    }
    Json(version.json()).into_response()
}

pub async fn list_versions(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(query): Query<PageQuery>,
) -> Response {
    let store = Store::new(&state.api_data);
    if store.get(&id).is_none() {
        return not_found(&format!("skill '{id}'"));
    }
    let versions = store.versions(&id);
    Json(page(versions, |v| v.id.as_str(), &query, VersionMeta::json)).into_response()
}

pub async fn retrieve_version(
    State(state): State<Arc<AppState>>,
    Path((id, version)): Path<(String, String)>,
) -> Response {
    let store = Store::new(&state.api_data);
    let found = store
        .get(&id)
        .and_then(|_| store.versions(&id).into_iter().find(|v| v.id == version));
    match found {
        Some(version) => Json(version.json()).into_response(),
        None => not_found(&format!("version '{version}' of skill '{id}'")),
    }
}

pub async fn delete_version(
    State(state): State<Arc<AppState>>,
    Path((id, version)): Path<(String, String)>,
) -> Response {
    let store = Store::new(&state.api_data);
    let Some(dir) = store
        .get(&id)
        .and_then(|_| store.version_dir(&id, &version))
    else {
        return not_found(&format!("version '{version}' of skill '{id}'"));
    };
    let _ = tokio::fs::remove_dir_all(dir).await;
    Json(json!({"id": version, "type": "skill_version_deleted"})).into_response()
}

/// A skill a request mounts: which version, and where its folder is.
pub(super) struct Mounted {
    pub skill_id: String,
    pub version: String,
    pub name: String,
    pub description: String,
    pub dir: PathBuf,
}

/// The skills `container.skills` names, each at the version it asks for
/// (`latest` when it names none).
pub(super) fn resolve(
    api_data: &std::path::Path,
    requested: &[Value],
) -> Result<Vec<Mounted>, String> {
    if requested.len() > MAX_MOUNTED {
        return Err(format!("a request may use at most {MAX_MOUNTED} skills"));
    }
    let store = Store::new(api_data);
    let mut out: Vec<Mounted> = Vec::new();
    for entry in requested {
        let skill_id = entry
            .get("skill_id")
            .and_then(Value::as_str)
            .ok_or("a container.skills entry needs a `skill_id`")?;
        match entry.get("type").and_then(Value::as_str) {
            Some("custom") => {}
            Some("anthropic") => {
                return Err(format!(
                    "the Anthropic-made skill '{skill_id}' is not on this server; upload a skill \
                     of your own with POST /v1/skills and use it as a `custom` skill"
                ));
            }
            _ => return Err("a container.skills entry's `type` must be \"custom\"".into()),
        }
        if store.get(skill_id).is_none() {
            return Err(format!("skill '{skill_id}' does not exist"));
        }
        let wanted = entry
            .get("version")
            .and_then(Value::as_str)
            .unwrap_or("latest");
        let versions = store.versions(skill_id);
        let version = match wanted {
            "latest" => versions.into_iter().next(),
            v => versions.into_iter().find(|meta| meta.id == v),
        }
        .ok_or_else(|| format!("skill '{skill_id}' has no version '{wanted}'"))?;
        if out.iter().any(|m| m.name == version.name) {
            return Err(format!("two skills named '{}' are listed", version.name));
        }
        let dir = store
            .skill_dir(skill_id)
            .join("versions")
            .join(&version.id)
            .join("files")
            .join(&version.name);
        out.push(Mounted {
            skill_id: skill_id.to_string(),
            version: version.id,
            name: version.name,
            description: version.description,
            dir,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upload(files: &[(&str, &str)]) -> Upload {
        Upload {
            display_name: None,
            files: files
                .iter()
                .map(|(p, t)| (p.to_string(), t.as_bytes().to_vec()))
                .collect(),
        }
    }

    const SKILL: &str =
        "---\nname: csv-report\ndescription: >\n  Turns a CSV\n  into a report.\n---\n# How\n";

    #[test]
    fn an_upload_is_one_folder_with_its_skill_md() {
        let checked = check(upload(&[
            ("csv-report/SKILL.md", SKILL),
            ("csv-report/scripts/run.py", "print(1)"),
        ]))
        .ok()
        .unwrap();
        assert_eq!(checked.name, "csv-report");
        assert_eq!(checked.description, "Turns a CSV into a report.");
        assert_eq!(checked.files[1].0, PathBuf::from("scripts/run.py"));
        // Files with no folder are the folder's contents.
        assert!(check(upload(&[("SKILL.md", SKILL)])).is_ok());
    }

    #[test]
    fn an_upload_that_is_not_one_clean_folder_is_refused() {
        for files in [
            vec![("a/SKILL.md", SKILL), ("b/x.py", "")],
            vec![("a/../SKILL.md", SKILL)],
            vec![("/a/SKILL.md", SKILL)],
            vec![("a/README.md", "no skill")],
            vec![("a/SKILL.md", "---\nname: Bad Name\ndescription: x\n---\n")],
            vec![("a/SKILL.md", "---\nname: ok\n---\n")],
        ] {
            assert!(check(upload(&files)).is_err(), "{files:?}");
        }
    }

    #[test]
    fn versions_resolve_to_latest_or_the_one_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path());
        let skill = SkillMeta {
            id: "skill_abc".into(),
            display_name: "x".into(),
            name: "csv-report".into(),
            created: 0,
            updated: 0,
            created_nanos: 0,
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (v1, v2) = rt.block_on(async {
            save_skill(&store, &skill).await.ok().unwrap();
            let v1 = write_version(
                &store,
                "skill_abc",
                check(upload(&[("SKILL.md", SKILL)])).ok().unwrap(),
            )
            .await
            .ok()
            .unwrap();
            let v2 = write_version(
                &store,
                "skill_abc",
                check(upload(&[("SKILL.md", SKILL)])).ok().unwrap(),
            )
            .await
            .ok()
            .unwrap();
            (v1, v2)
        });
        let latest = resolve(
            dir.path(),
            &[json!({"type": "custom", "skill_id": "skill_abc"})],
        )
        .ok()
        .unwrap();
        assert_eq!(latest[0].version, v2.id);
        assert!(latest[0].dir.join("SKILL.md").exists());
        let pinned = resolve(
            dir.path(),
            &[json!({"type": "custom", "skill_id": "skill_abc", "version": v1.id})],
        )
        .ok()
        .unwrap();
        assert_eq!(pinned[0].version, v1.id);
        assert!(
            resolve(
                dir.path(),
                &[json!({"type": "anthropic", "skill_id": "pptx"})]
            )
            .is_err()
        );
    }
}
