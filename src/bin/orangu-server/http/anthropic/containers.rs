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

//! The containers the `code_execution` server tool runs code in.
//!
//! A container is a directory, `<api_data>/containers/<id>/`: `meta.json`,
//! and `workspace/`, which a command sees as `/workspace` — the only place
//! it can write that outlives it. Each command runs in a fresh bubblewrap
//! sandbox: its own user, PID, network, IPC and UTS namespaces, so no
//! network beyond its own loopback; `/usr` and a few `/etc` files read-only;
//! a private `/tmp`; the request's skills read-only under `/skills`; and
//! the container's further mounts — a Managed Agents session's uploads,
//! outputs and memory stores — read-only or not as each says. The home
//! directory, and everything else on the host, is not there.
//!
//! The text editor's commands run here rather than in the sandbox, on the
//! same paths, each one resolved and checked to stay inside the workspace
//! (or a skill) before it is read or written.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

use super::files::{FileStore, new_id, valid_id};

/// How long a container outlives its last use, as Anthropic keeps one.
pub(super) const LIFETIME_SECS: u64 = 30 * 24 * 3600;

/// How long one command may run.
const RUN_TIMEOUT: Duration = Duration::from_secs(120);

/// What a command's stdout and stderr are each cut to, in bytes.
const OUTPUT_LIMIT: usize = 100_000;

/// A command's address-space limit, as Anthropic's container's memory.
const MEMORY_LIMIT: u64 = 5 << 30;

/// The largest file a command may write.
const FILE_SIZE_LIMIT: u64 = 1 << 30;

/// How many files one command's outputs report, and the largest reported.
const OUTPUT_FILES: usize = 20;
const OUTPUT_FILE_LIMIT: u64 = 100 << 20;

/// The largest file the text editor reads.
const EDIT_LIMIT: u64 = 10 << 20;

/// The `/etc` entries a sandbox sees: what the dynamic loader, user lookup
/// and fonts read, and nothing that configures this server.
const ETC_ENTRIES: &[&str] = &[
    "ld.so.cache",
    "ld.so.conf",
    "ld.so.conf.d",
    "alternatives",
    "localtime",
    "passwd",
    "group",
    "nsswitch.conf",
    "mime.types",
    "fonts",
];

/// The `/etc` entries a container with the network also sees: name
/// resolution and the certificates TLS checks against.
const NETWORK_ETC_ENTRIES: &[&str] = &[
    "resolv.conf",
    "hosts",
    "host.conf",
    "gai.conf",
    "ssl",
    "pki",
    "ca-certificates",
    "crypto-policies",
];

#[derive(Serialize, Deserialize, Clone)]
struct Meta {
    id: String,
    /// Unix seconds.
    created: u64,
    last_used: u64,
    /// The Files API files copied in, by id: where in the workspace each is.
    #[serde(default)]
    uploads: HashMap<String, String>,
}

/// One lock per container, so two requests on the same container take
/// their turns at it.
static LOCKS: LazyLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(Default::default);

fn lock_for(id: &str) -> Arc<tokio::sync::Mutex<()>> {
    let mut locks = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    locks.entry(id.to_string()).or_default().clone()
}

/// What a command printed and how it ended.
pub(super) struct Output {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
    pub timed_out: bool,
}

/// A text editor command that failed: Anthropic's error code, and what the
/// model is told.
pub(super) struct EditError {
    pub code: &'static str,
    pub message: String,
}

fn edit_error(code: &'static str, message: impl Into<String>) -> EditError {
    EditError {
        code,
        message: message.into(),
    }
}

/// A file's size and modification time, which is what says it changed.
pub(super) type Stamp = (u64, SystemTime);

/// A host directory a container's commands see at `at`.
pub(super) struct Mount {
    pub host: PathBuf,
    pub at: String,
    pub writable: bool,
}

pub(super) struct Container {
    pub id: String,
    root: PathBuf,
    api_data: PathBuf,
    meta: Meta,
    /// The skills this request mounts, by name: the directory each one's
    /// files are in.
    skills: Vec<(String, PathBuf)>,
    /// Whether commands share the host's network instead of having none.
    network: bool,
    /// Further directories commands see, beside `/workspace`.
    mounts: Vec<Mount>,
    /// Files commands see read-only, and the file tools do not.
    files: Vec<(PathBuf, String)>,
    /// Variables commands run with, beside the sandbox's own.
    env: Vec<(String, String)>,
}

fn now() -> u64 {
    super::unix_now()
}

/// Removes every container past its lifetime.
async fn sweep(store: &Path) {
    let Ok(mut entries) = tokio::fs::read_dir(store).await else {
        return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let Ok(text) = tokio::fs::read_to_string(entry.path().join("meta.json")).await else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<Meta>(&text) else {
            continue;
        };
        if meta.last_used + LIFETIME_SECS < now() {
            let _ = tokio::fs::remove_dir_all(entry.path()).await;
        }
    }
}

impl Container {
    /// The container a request names, or a new one when it names none.
    pub(super) async fn open(api_data: &Path, requested: Option<&str>) -> Result<Self, String> {
        let store = api_data.join("containers");
        sweep(&store).await;
        let (root, mut meta) = match requested {
            Some(id) => {
                let gone = || format!("container '{id}' does not exist or has expired");
                if !valid_id(id, "container_") {
                    return Err(gone());
                }
                let root = store.join(id);
                let text = tokio::fs::read_to_string(root.join("meta.json"))
                    .await
                    .map_err(|_| gone())?;
                let meta: Meta = serde_json::from_str(&text).map_err(|_| gone())?;
                (root, meta)
            }
            None => {
                let id = new_id("container_");
                let root = store.join(&id);
                tokio::fs::create_dir_all(root.join("workspace"))
                    .await
                    .map_err(|err| format!("creating the container: {err}"))?;
                let meta = Meta {
                    id,
                    created: now(),
                    last_used: now(),
                    uploads: HashMap::new(),
                };
                (root, meta)
            }
        };
        meta.last_used = now();
        let this = Self {
            id: meta.id.clone(),
            root,
            api_data: api_data.to_path_buf(),
            meta,
            skills: Vec::new(),
            network: false,
            mounts: Vec::new(),
            files: Vec::new(),
            env: Vec::new(),
        };
        this.save().await?;
        Ok(this)
    }

    /// Writes the container's record whole, by a rename, so that a thread
    /// opening it at the same time never reads it half written.
    async fn save(&self) -> Result<(), String> {
        let text = serde_json::to_string(&self.meta).unwrap_or_default();
        let tmp = self.root.join(format!("meta.json.{}", new_id("tmp_")));
        let saved = async {
            tokio::fs::write(&tmp, text).await?;
            tokio::fs::rename(&tmp, self.root.join("meta.json")).await
        }
        .await;
        if saved.is_err() {
            let _ = tokio::fs::remove_file(&tmp).await;
        }
        saved.map_err(|err| format!("saving the container: {err}"))
    }

    /// Where `/workspace` is on the host.
    pub(super) fn workspace(&self) -> PathBuf {
        self.root.join("workspace")
    }

    /// The container's own directory on the host, which `/workspace` is in.
    pub(super) fn root(&self) -> &Path {
        &self.root
    }

    /// Shows `host` to this container's commands at `at`, read-only unless
    /// `writable`.
    pub(super) fn mount(&mut self, host: PathBuf, at: &str, writable: bool) {
        self.mounts.push(Mount {
            host,
            at: at.to_string(),
            writable,
        });
    }

    pub(super) fn mounts(&self) -> &[Mount] {
        &self.mounts
    }

    /// Shows the file `host` to commands, read-only, at `at`.
    pub(super) fn bind_file(&mut self, host: PathBuf, at: &str) {
        self.files.push((host, at.to_string()));
    }

    /// Runs commands with `name` set to `value`.
    pub(super) fn set_env(&mut self, name: &str, value: &str) {
        self.env.retain(|(n, _)| n != name);
        self.env.push((name.to_string(), value.to_string()));
    }

    /// Whether commands share the host's network.
    pub(super) fn has_network(&self) -> bool {
        self.network
    }

    /// Lets this container's commands use the host's network, and read the
    /// `/etc` files name resolution and TLS need.
    pub(super) fn share_network(&mut self) {
        self.network = true;
    }

    /// When the container expires, in Unix seconds.
    pub(super) fn expires_at(&self) -> u64 {
        self.meta.last_used + LIFETIME_SECS
    }

    /// Mounts `skills` — `(name, directory)` — under `/skills` for this
    /// request's commands.
    pub(super) fn mount_skills(&mut self, skills: Vec<(String, PathBuf)>) {
        self.skills = skills;
    }

    /// Copies a Files API file into the workspace, once: the path a command
    /// finds it at.
    pub(super) async fn upload(&mut self, file_id: &str) -> Result<String, String> {
        if let Some(path) = self.meta.uploads.get(file_id) {
            return Ok(format!("/workspace/{path}"));
        }
        let store = FileStore::at(&self.api_data);
        let meta = store
            .get(file_id)
            .ok_or_else(|| format!("file '{file_id}' does not exist"))?;
        let base: String = Path::new(&meta.filename)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
            .trim_start_matches('.')
            .to_string();
        let base = if base.is_empty() {
            "upload".to_string()
        } else {
            base
        };
        let workspace = self.workspace();
        let mut name = base.clone();
        let mut n = 1;
        while tokio::fs::symlink_metadata(workspace.join(&name))
            .await
            .is_ok()
        {
            let path = Path::new(&base);
            name = match (path.file_stem(), path.extension()) {
                (Some(stem), Some(ext)) => {
                    format!("{}-{n}.{}", stem.to_string_lossy(), ext.to_string_lossy())
                }
                _ => format!("{base}-{n}"),
            };
            n += 1;
        }
        tokio::fs::copy(store.data_path(file_id), workspace.join(&name))
            .await
            .map_err(|err| format!("copying '{file_id}' into the container: {err}"))?;
        self.meta.uploads.insert(file_id.to_string(), name.clone());
        self.save().await?;
        Ok(format!("/workspace/{name}"))
    }

    /// The sandbox around `program`.
    fn command(&self, program: &[&str]) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new("bwrap");
        cmd.args([
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
        ]);
        // `/bin`, `/lib` and the rest as the host has them: links into
        // `/usr`, or directories of their own.
        for top in ["bin", "sbin", "lib", "lib32", "lib64"] {
            let host = Path::new("/").join(top);
            match std::fs::symlink_metadata(&host) {
                Ok(m) if m.file_type().is_symlink() => {
                    if let Ok(target) = std::fs::read_link(&host) {
                        cmd.arg("--symlink").arg(target).arg(&host);
                    }
                }
                Ok(m) if m.is_dir() => {
                    cmd.arg("--ro-bind").arg(&host).arg(&host);
                }
                _ => {}
            }
        }
        if self.network {
            cmd.arg("--share-net");
        }
        let network_entries: &[&str] = if self.network {
            NETWORK_ETC_ENTRIES
        } else {
            &[]
        };
        for entry in ETC_ENTRIES.iter().chain(network_entries) {
            let host = Path::new("/etc").join(entry);
            if host.exists() {
                cmd.arg("--ro-bind").arg(&host).arg(&host);
            }
        }
        cmd.args(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]);
        cmd.arg("--bind").arg(self.workspace()).arg("/workspace");
        for mount in &self.mounts {
            let bind = if mount.writable {
                "--bind"
            } else {
                "--ro-bind"
            };
            cmd.arg(bind).arg(&mount.host).arg(&mount.at);
        }
        for (host, at) in &self.files {
            cmd.arg("--ro-bind").arg(host).arg(at);
        }
        for (name, dir) in &self.skills {
            cmd.arg("--ro-bind").arg(dir).arg(format!("/skills/{name}"));
        }
        cmd.args([
            "--chdir",
            "/workspace",
            "--hostname",
            "sandbox",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/local/bin:/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/workspace",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
        ]);
        for (name, value) in &self.env {
            cmd.arg("--setenv").arg(name).arg(value);
        }
        cmd.arg("--");
        cmd.args(program);
        // SAFETY: only async-signal-safe calls (setrlimit) run between fork
        // and exec.
        unsafe {
            cmd.pre_exec(|| {
                let limit = |resource, value: u64| {
                    let rlim = libc::rlimit {
                        rlim_cur: value as libc::rlim_t,
                        rlim_max: value as libc::rlim_t,
                    };
                    libc::setrlimit(resource, &rlim);
                };
                limit(libc::RLIMIT_AS, MEMORY_LIMIT);
                limit(libc::RLIMIT_FSIZE, FILE_SIZE_LIMIT);
                limit(libc::RLIMIT_CORE, 0);
                Ok(())
            });
        }
        cmd
    }

    /// Starts `program` in the sandbox with its stdin and stdout piped to
    /// the caller, who talks to it and decides when it ends: it is killed
    /// when the handle is dropped.
    pub(super) fn spawn(&self, program: &[&str]) -> Result<tokio::process::Child, String> {
        let mut cmd = self.command(program);
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        cmd.spawn().map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => "bwrap is not installed on the server".to_string(),
            _ => format!("starting the sandbox: {err}"),
        })
    }

    /// Runs `program` in the sandbox, `stdin` fed to it, and waits for it
    /// for at most [`RUN_TIMEOUT`].
    pub(super) async fn run(
        &self,
        program: &[&str],
        stdin: Option<&str>,
    ) -> Result<Output, String> {
        let lock = lock_for(&self.id);
        let _turn = lock.lock().await;
        let mut cmd = self.command(program);
        cmd.stdin(if stdin.is_some() {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => "bwrap is not installed on the server".to_string(),
            _ => format!("starting the sandbox: {err}"),
        })?;
        if let (Some(input), Some(mut pipe)) = (stdin, child.stdin.take()) {
            let input = input.to_string();
            tokio::spawn(async move {
                let _ = pipe.write_all(input.as_bytes()).await;
            });
        }
        let stdout = tokio::spawn(read_capped(child.stdout.take()));
        let stderr = tokio::spawn(read_capped(child.stderr.take()));
        let (code, timed_out) = match tokio::time::timeout(RUN_TIMEOUT, child.wait()).await {
            Ok(Ok(status)) => {
                use std::os::unix::process::ExitStatusExt as _;
                let code = status
                    .code()
                    .or_else(|| status.signal().map(|s| 128 + s))
                    .unwrap_or(-1);
                (code, false)
            }
            Ok(Err(err)) => return Err(format!("waiting for the sandbox: {err}")),
            Err(_) => {
                // The sandbox's own init dies with it, and every process in
                // its PID namespace with that.
                let _ = child.kill().await;
                (137, true)
            }
        };
        let collect = |handle: tokio::task::JoinHandle<String>| async move {
            tokio::time::timeout(Duration::from_secs(2), handle)
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or_default()
        };
        Ok(Output {
            stdout: collect(stdout).await,
            stderr: collect(stderr).await,
            code,
            timed_out,
        })
    }

    /// Every regular file in the workspace with its stamp — hidden entries
    /// and `__pycache__` left out — to tell afterwards what a command wrote.
    pub(super) async fn snapshot(&self) -> HashMap<PathBuf, Stamp> {
        snapshot(self.workspace()).await
    }

    /// The files a command wrote — those new or changed since `before` —
    /// stored in the Files API: each one's file id and container path.
    pub(super) async fn outputs(&self, before: &HashMap<PathBuf, Stamp>) -> Vec<(String, String)> {
        let after = self.snapshot().await;
        let mut changed: Vec<&PathBuf> = after
            .iter()
            .filter(|(path, stamp)| before.get(*path) != Some(stamp))
            .filter(|(_, (size, _))| *size <= OUTPUT_FILE_LIMIT)
            .map(|(path, _)| path)
            .collect();
        changed.sort();
        let store = FileStore::at(&self.api_data);
        let mut out = Vec::new();
        for rel in changed.into_iter().take(OUTPUT_FILES) {
            let name = rel
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let Ok(meta) = store.save_copy(&name, &self.workspace().join(rel)).await {
                out.push((meta.id, format!("/workspace/{}", rel.display())));
            }
        }
        out
    }

    /// Where a path the model gave lives on the host, and whether it may be
    /// written: under `/workspace` (or relative to it) it may, under a
    /// mounted skill it is read-only, anywhere else it is refused.
    async fn resolve(&self, path: &str, write: bool) -> Result<PathBuf, EditError> {
        let path = path.trim();
        let invalid = |m: String| edit_error("invalid_tool_input", m);
        let (base, rest) = if let Some(rest) = path
            .strip_prefix("/workspace")
            .filter(|r| r.is_empty() || r.starts_with('/'))
        {
            (self.workspace(), rest.to_string())
        } else if let Some(rest) = path.strip_prefix("/skills/") {
            if write {
                return Err(invalid("/skills is read-only".into()));
            }
            let (name, inner) = rest.split_once('/').unwrap_or((rest, ""));
            let Some((_, dir)) = self.skills.iter().find(|(n, _)| n == name) else {
                return Err(edit_error(
                    "file_not_found",
                    format!("{path} does not exist"),
                ));
            };
            (dir.clone(), inner.to_string())
        } else if path.starts_with('/') {
            return Err(invalid(format!(
                "{path} is outside the container's files: use a path under /workspace"
            )));
        } else {
            (self.workspace(), path.to_string())
        };
        let mut resolved = base.clone();
        for component in Path::new(&rest).components() {
            match component {
                Component::Normal(part) => resolved.push(part),
                // Only names were pushed, so above the base there is one to
                // take off.
                Component::ParentDir if resolved != base => {
                    resolved.pop();
                }
                Component::ParentDir => {
                    return Err(invalid(format!("{path} leaves the container's files")));
                }
                _ => {}
            }
        }
        // A link written by a command could point anywhere on the host;
        // what the path really names has to be inside too.
        let base_real = tokio::fs::canonicalize(&base)
            .await
            .map_err(|err| edit_error("unavailable", err.to_string()))?;
        let mut existing = resolved.clone();
        loop {
            if let Ok(real) = tokio::fs::canonicalize(&existing).await {
                if !real.starts_with(&base_real) {
                    return Err(invalid(format!("{path} leaves the container's files")));
                }
                break;
            }
            if !existing.pop() {
                break;
            }
        }
        Ok(resolved)
    }

    /// Runs one text editor command: its result block's `content`, and the
    /// text the model reads.
    pub(super) async fn edit(
        &self,
        input: &Map<String, Value>,
    ) -> Result<(Value, String), EditError> {
        let lock = lock_for(&self.id);
        let _turn = lock.lock().await;
        let string = |key: &str| input.get(key).and_then(Value::as_str);
        let invalid = |m: &str| edit_error("invalid_tool_input", m);
        let path = string("path").ok_or_else(|| invalid("`path` is required"))?;
        match string("command") {
            Some("view") => self.view(path, input.get("view_range")).await,
            Some("create") => {
                let text = string("file_text").ok_or_else(|| invalid("`file_text` is required"))?;
                let target = self.resolve(path, true).await?;
                let existed = tokio::fs::symlink_metadata(&target).await.is_ok();
                if let Some(parent) = target.parent() {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|err| edit_error("unavailable", err.to_string()))?;
                }
                tokio::fs::write(&target, text)
                    .await
                    .map_err(|err| edit_error("unavailable", err.to_string()))?;
                let said = if existed {
                    format!("{path} was overwritten.")
                } else {
                    format!("{path} was created.")
                };
                Ok((
                    json!({"type": "text_editor_code_execution_create_result",
                           "is_file_update": existed}),
                    said,
                ))
            }
            Some("str_replace") => {
                let old = string("old_str").ok_or_else(|| invalid("`old_str` is required"))?;
                let new = string("new_str").unwrap_or_default();
                let target = self.resolve(path, true).await?;
                let text = read_text(&target, path).await?;
                let found = text.matches(old).count();
                if old.is_empty() || found == 0 {
                    return Err(invalid(&format!("`old_str` does not occur in {path}")));
                }
                if found > 1 {
                    return Err(invalid(&format!(
                        "`old_str` occurs {found} times in {path}; it must occur once"
                    )));
                }
                let at = text.find(old).unwrap_or_default();
                let start = text[..at].matches('\n').count() + 1;
                tokio::fs::write(&target, text.replacen(old, new, 1))
                    .await
                    .map_err(|err| edit_error("unavailable", err.to_string()))?;
                let old_lines: Vec<&str> = old.lines().collect();
                let new_lines: Vec<&str> = new.lines().collect();
                let lines: Vec<String> = old_lines
                    .iter()
                    .map(|l| format!("-{l}"))
                    .chain(new_lines.iter().map(|l| format!("+{l}")))
                    .collect();
                let said = format!("{path} was edited at line {start}:\n{}", lines.join("\n"));
                Ok((
                    json!({"type": "text_editor_code_execution_str_replace_result",
                           "old_start": start, "old_lines": old_lines.len(),
                           "new_start": start, "new_lines": new_lines.len(),
                           "lines": lines}),
                    said,
                ))
            }
            Some(other) => Err(invalid(&format!(
                "unknown command '{other}': use view, create or str_replace"
            ))),
            None => Err(invalid("`command` is required")),
        }
    }

    async fn view(&self, path: &str, range: Option<&Value>) -> Result<(Value, String), EditError> {
        let target = self.resolve(path, false).await?;
        let meta = tokio::fs::metadata(&target)
            .await
            .map_err(|_| edit_error("file_not_found", format!("{path} does not exist")))?;
        if meta.is_dir() {
            let mut names = Vec::new();
            let mut entries = tokio::fs::read_dir(&target)
                .await
                .map_err(|err| edit_error("unavailable", err.to_string()))?;
            while let Ok(Some(entry)) = entries.next_entry().await {
                let mut name = entry.file_name().to_string_lossy().into_owned();
                if entry.file_type().await.is_ok_and(|t| t.is_dir()) {
                    name.push('/');
                }
                names.push(name);
            }
            names.sort();
            let listing = names.join("\n");
            return Ok((
                json!({"type": "text_editor_code_execution_view_result", "file_type": "text",
                       "content": listing, "num_lines": names.len(), "start_line": 1,
                       "total_lines": names.len()}),
                listing,
            ));
        }
        if target.extension().and_then(|e| e.to_str()) == Some("pdf") {
            let bytes = tokio::fs::read(&target)
                .await
                .map_err(|err| edit_error("unavailable", err.to_string()))?;
            let text = orangu::pdf::extract_text(&bytes)
                .map_err(|err| edit_error("invalid_tool_input", err.to_string()))?;
            return Ok((
                json!({"type": "text_editor_code_execution_view_result", "file_type": "pdf",
                       "content": text}),
                text,
            ));
        }
        let text = read_text(&target, path).await?;
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        let (first, last) = match range.and_then(Value::as_array).map(|r| {
            (
                r.first().and_then(Value::as_i64).unwrap_or(1),
                r.get(1).and_then(Value::as_i64).unwrap_or(-1),
            )
        }) {
            Some((first, last)) => {
                let first = (first.max(1) as usize).min(total.max(1));
                let last = if last < 0 {
                    total
                } else {
                    (last as usize).clamp(first, total.max(first))
                };
                (first, last)
            }
            None => (1, total),
        };
        let shown = if total == 0 {
            &lines[..]
        } else {
            &lines[first - 1..last.min(total)]
        };
        let content = shown.join("\n");
        let numbered = shown
            .iter()
            .enumerate()
            .map(|(n, line)| format!("{:>6}\t{line}", first + n))
            .collect::<Vec<_>>()
            .join("\n");
        Ok((
            json!({"type": "text_editor_code_execution_view_result", "file_type": "text",
                   "content": content, "num_lines": shown.len(), "start_line": first,
                   "total_lines": total}),
            numbered,
        ))
    }
}

/// A file the editor reads, as text: missing, too large or not UTF-8 is an
/// error the model is told about.
async fn read_text(target: &Path, path: &str) -> Result<String, EditError> {
    let meta = tokio::fs::metadata(target)
        .await
        .map_err(|_| edit_error("file_not_found", format!("{path} does not exist")))?;
    if !meta.is_file() {
        return Err(edit_error(
            "invalid_tool_input",
            format!("{path} is not a file"),
        ));
    }
    if meta.len() > EDIT_LIMIT {
        return Err(edit_error(
            "invalid_tool_input",
            format!("{path} is too large to open in the editor; use bash"),
        ));
    }
    let bytes = tokio::fs::read(target)
        .await
        .map_err(|err| edit_error("unavailable", err.to_string()))?;
    String::from_utf8(bytes).map_err(|_| {
        edit_error(
            "invalid_tool_input",
            format!("{path} is not a text file; use bash"),
        )
    })
}

/// Everything a pipe gives until it closes, the first [`OUTPUT_LIMIT`]
/// bytes kept and the rest counted.
async fn read_capped(pipe: Option<impl AsyncRead + Unpin>) -> String {
    let Some(mut pipe) = pipe else {
        return String::new();
    };
    let mut kept = Vec::new();
    let mut dropped = 0usize;
    let mut buf = [0u8; 8192];
    loop {
        match pipe.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let room = OUTPUT_LIMIT.saturating_sub(kept.len());
                kept.extend_from_slice(&buf[..n.min(room)]);
                dropped += n.saturating_sub(room);
            }
        }
    }
    let mut text = String::from_utf8_lossy(&kept).into_owned();
    if dropped > 0 {
        text.push_str(&format!("\n[… {dropped} more bytes not shown]"));
    }
    text
}

/// Every file under `root` — at most 10 000, hidden names and
/// `__pycache__` left out — with its stamp.
pub(super) async fn snapshot(root: PathBuf) -> HashMap<PathBuf, Stamp> {
    tokio::task::spawn_blocking(move || {
        let mut out = HashMap::new();
        let mut pending = vec![PathBuf::new()];
        while let Some(rel) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(root.join(&rel)) else {
                continue;
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with('.') || name == "__pycache__" {
                    continue;
                }
                let Ok(meta) = entry.metadata() else { continue };
                let path = rel.join(&*name);
                if meta.file_type().is_dir() {
                    pending.push(path);
                } else if meta.file_type().is_file() {
                    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    out.insert(path, (meta.len(), mtime));
                }
                if out.len() >= 10_000 {
                    return out;
                }
            }
        }
        out
    })
    .await
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn container(dir: &Path) -> Container {
        Container::open(dir, None).await.unwrap()
    }

    fn input(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[tokio::test]
    async fn the_editor_creates_views_and_replaces_in_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let c = container(dir.path()).await;
        let (block, _) = c
            .edit(&input(
                json!({"command": "create", "path": "/workspace/a/notes.txt",
                                 "file_text": "one\ntwo\nthree\n"}),
            ))
            .await
            .ok()
            .unwrap();
        assert_eq!(block["is_file_update"], false);
        let (block, text) = c
            .edit(&input(
                json!({"command": "view", "path": "a/notes.txt", "view_range": [2, 3]}),
            ))
            .await
            .ok()
            .unwrap();
        assert_eq!(block["content"], "two\nthree");
        assert_eq!(block["total_lines"], 3);
        assert!(text.contains("     2\ttwo"));
        let (block, _) = c
            .edit(&input(
                json!({"command": "str_replace", "path": "/workspace/a/notes.txt",
                                 "old_str": "two", "new_str": "2"}),
            ))
            .await
            .ok()
            .unwrap();
        assert_eq!(block["old_start"], 2);
        assert_eq!(block["lines"], json!(["-two", "+2"]));
        let written = std::fs::read_to_string(c.workspace().join("a/notes.txt")).unwrap();
        assert_eq!(written, "one\n2\nthree\n");
    }

    #[tokio::test]
    async fn the_editor_stays_inside_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let c = container(dir.path()).await;
        for path in [
            "/etc/passwd",
            "../meta.json",
            "/workspace/../meta.json",
            "a/../../x",
        ] {
            let err = c
                .edit(&input(json!({"command": "view", "path": path})))
                .await
                .err()
                .unwrap();
            assert_eq!(err.code, "invalid_tool_input", "{path}");
        }
        // A link a command left behind is followed no further than the
        // workspace.
        std::os::unix::fs::symlink("/etc", c.workspace().join("out")).unwrap();
        let err = c
            .edit(&input(json!({"command": "view", "path": "out/passwd"})))
            .await
            .err()
            .unwrap();
        assert_eq!(err.code, "invalid_tool_input");
    }

    #[tokio::test]
    async fn a_named_container_is_reopened_and_an_unknown_one_refused() {
        let dir = tempfile::tempdir().unwrap();
        let c = container(dir.path()).await;
        std::fs::write(c.workspace().join("kept.txt"), "x").unwrap();
        let again = Container::open(dir.path(), Some(&c.id)).await.unwrap();
        assert!(again.workspace().join("kept.txt").exists());
        assert!(
            Container::open(dir.path(), Some("container_nope"))
                .await
                .is_err()
        );
        assert!(Container::open(dir.path(), Some("../files")).await.is_err());
    }

    #[tokio::test]
    async fn outputs_are_the_files_a_command_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let c = container(dir.path()).await;
        std::fs::write(c.workspace().join("old.txt"), "old").unwrap();
        let before = c.snapshot().await;
        std::fs::write(c.workspace().join("new.csv"), "a,b\n").unwrap();
        std::fs::create_dir(c.workspace().join(".cache")).unwrap();
        std::fs::write(c.workspace().join(".cache/x"), "hidden").unwrap();
        let outputs = c.outputs(&before).await;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].1, "/workspace/new.csv");
        let stored = FileStore::at(dir.path()).get(&outputs[0].0).unwrap();
        assert_eq!(stored.filename, "new.csv");
        assert_eq!(stored.size_bytes, 4);
    }

    #[tokio::test]
    async fn a_command_sees_its_workspace_and_nothing_of_the_host() {
        let dir = tempfile::tempdir().unwrap();
        let c = container(dir.path()).await;
        let script = "echo hi > made.txt; pwd; test -e /home && echo home; \
                      test -e /etc/hostname && echo etc; echo $HOME; \
                      python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 80), 2)' \
                      2>/dev/null || echo offline";
        let out = match c.run(&["/bin/bash", "-c", script], None).await {
            Ok(out) if !out.stderr.starts_with("bwrap:") => out,
            // No bubblewrap, or no user namespaces, on this machine.
            _ => return,
        };
        assert_eq!(
            out.stdout, "/workspace\n/workspace\noffline\n",
            "{}",
            out.stderr
        );
        assert_eq!(out.code, 0);
        assert!(c.workspace().join("made.txt").exists());
    }

    #[tokio::test]
    async fn output_past_the_cap_is_counted_not_kept() {
        let data = vec![b'x'; OUTPUT_LIMIT + 10];
        let text = read_capped(Some(&data[..])).await;
        assert!(text.starts_with("xxx"));
        assert!(text.ends_with("[… 10 more bytes not shown]"));
    }
}
