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

//! Managed Agents' built-in tools, `agent_toolset_20260401`: `bash`,
//! `read`, `write`, `edit`, `glob`, `grep`, `web_fetch` and `web_search`.
//!
//! The file tools are the workspace tools `orangu` itself gives a model
//! (`show_file`, `create_file`, `modify_file`), confined to the session
//! container's workspace; `bash` and `grep` run in the container's sandbox.
//! The web tools are the Messages API's server tools, which a session runs
//! itself.

use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};

use super::containers::Container;
use super::server_tools::run_text;

pub(super) const TOOLSET: &str = "agent_toolset_20260401";

/// The toolset's tools, in the order a declaration lists them.
pub(super) const TOOL_NAMES: [&str; 8] = [
    "bash",
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "web_fetch",
    "web_search",
];

/// The tools that touch nothing but the session's container.
fn sandboxed(name: &str) -> bool {
    !matches!(name, "web_fetch" | "web_search")
}

/// How a tool's calls are let through.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Policy {
    AlwaysAllow,
    AlwaysAsk,
    Auto,
}

impl Policy {
    pub(super) fn parse(value: &Value) -> Result<Self, String> {
        match value.get("type").and_then(Value::as_str) {
            Some("always_allow") => Ok(Self::AlwaysAllow),
            Some("always_ask") => Ok(Self::AlwaysAsk),
            Some("auto") => Ok(Self::Auto),
            _ => {
                Err("a permission_policy's `type` is always_allow, always_ask or auto".to_string())
            }
        }
    }

    fn json(self) -> Value {
        json!({"type": match self {
            Self::AlwaysAllow => "always_allow",
            Self::AlwaysAsk => "always_ask",
            Self::Auto => "auto",
        }})
    }

    /// What a call under this policy gets — `allow` or `ask` — and the
    /// `evaluation` that says why. `auto` lets through a call that touches
    /// only the container and asks about one that reaches anything else.
    pub(super) fn evaluate(self, contained: bool) -> (&'static str, Value) {
        match self {
            Self::AlwaysAllow => ("allow", json!({"type": "always_allow"})),
            Self::AlwaysAsk => ("ask", json!({"type": "always_ask"})),
            Self::Auto if contained => (
                "allow",
                json!({"type": "auto", "evaluated_permission": {"type": "allow"}}),
            ),
            Self::Auto => (
                "ask",
                json!({"type": "auto", "evaluated_permission": {
                    "type": "ask", "reason_code": "indeterminate",
                }}),
            ),
        }
    }
}

/// A toolset's `default_config` and one `configs` entry over it: whether
/// the tool is on, and its policy. `fallback` is the policy when neither
/// names one.
pub(super) fn resolve_config(
    toolset: &Value,
    name: &str,
    fallback: Policy,
) -> Result<(bool, Policy), String> {
    let default = toolset.get("default_config");
    let entry = toolset
        .get("configs")
        .and_then(Value::as_array)
        .and_then(|configs| {
            configs
                .iter()
                .find(|c| c.get("name").and_then(Value::as_str) == Some(name))
        });
    let field = |key: &str| {
        entry
            .and_then(|e| e.get(key))
            .filter(|v| !v.is_null())
            .or_else(|| default.and_then(|d| d.get(key)).filter(|v| !v.is_null()))
    };
    let enabled = field("enabled").and_then(Value::as_bool).unwrap_or(true);
    let policy = field("permission_policy")
        .map(Policy::parse)
        .transpose()?
        .unwrap_or(fallback);
    Ok((enabled, policy))
}

/// An `agent_toolset_20260401` declaration as an agent stores and shows
/// it: `default_config` filled in, and one `configs` entry for every tool.
pub(super) fn normalize(toolset: &Value) -> Result<Value, String> {
    if let Some(configs) = toolset.get("configs").and_then(Value::as_array) {
        for config in configs {
            let name = config.get("name").and_then(Value::as_str).unwrap_or("");
            if !TOOL_NAMES.contains(&name) {
                return Err(format!(
                    "'{name}' is not a tool of {TOOLSET}: the tools are {}",
                    TOOL_NAMES.join(", ")
                ));
            }
            if config.get("allowed_domains").is_some() && config.get("blocked_domains").is_some() {
                return Err(format!(
                    "the '{name}' config sets both allowed_domains and blocked_domains"
                ));
            }
        }
    }
    let default = toolset.get("default_config");
    let default_enabled = default
        .and_then(|d| d.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let default_policy = default
        .and_then(|d| d.get("permission_policy"))
        .filter(|v| !v.is_null())
        .map(Policy::parse)
        .transpose()?
        .unwrap_or(Policy::AlwaysAllow);
    let mut configs = Vec::new();
    for name in TOOL_NAMES {
        let (enabled, policy) = resolve_config(toolset, name, Policy::AlwaysAllow)?;
        let mut config = json!({
            "name": name, "type": name, "enabled": enabled,
            "permission_policy": policy.json(),
        });
        let given = toolset
            .get("configs")
            .and_then(Value::as_array)
            .and_then(|c| {
                c.iter()
                    .find(|c| c.get("name").and_then(Value::as_str) == Some(name))
            });
        if let Some(given) = given.and_then(Value::as_object) {
            for (key, value) in given {
                if !matches!(
                    key.as_str(),
                    "name" | "type" | "enabled" | "permission_policy"
                ) {
                    config[key] = value.clone();
                }
            }
        }
        configs.push(config);
    }
    Ok(json!({
        "type": TOOLSET,
        "default_config": {"enabled": default_enabled, "permission_policy": default_policy.json()},
        "configs": configs,
    }))
}

/// An `mcp_toolset` declaration as an agent stores it, its server one the
/// agent lists. An MCP tool asks before it runs unless a policy says
/// otherwise.
pub(super) fn normalize_mcp(toolset: &Value, servers: &[Value]) -> Result<Value, String> {
    let server = toolset
        .get("mcp_server_name")
        .and_then(Value::as_str)
        .ok_or("an mcp_toolset needs an `mcp_server_name`")?;
    if !servers
        .iter()
        .any(|s| s.get("name").and_then(Value::as_str) == Some(server))
    {
        return Err(format!(
            "the mcp_toolset names '{server}', which is not one of the agent's mcp_servers"
        ));
    }
    let default = toolset.get("default_config");
    let enabled = default
        .and_then(|d| d.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let policy = default
        .and_then(|d| d.get("permission_policy"))
        .filter(|v| !v.is_null())
        .map(Policy::parse)
        .transpose()?
        .unwrap_or(Policy::AlwaysAsk);
    let mut configs = Vec::new();
    for config in toolset
        .get("configs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = config
            .get("name")
            .and_then(Value::as_str)
            .ok_or("an mcp_toolset config needs a `name`")?;
        let (on, chosen) = resolve_config(toolset, name, Policy::AlwaysAsk)?;
        configs.push(json!({
            "name": name, "enabled": on, "permission_policy": chosen.json(),
        }));
    }
    Ok(json!({
        "type": "mcp_toolset",
        "mcp_server_name": server,
        "default_config": {"enabled": enabled, "permission_policy": policy.json()},
        "configs": configs,
    }))
}

/// One built-in tool as a session offers it: whether it is on and how its
/// calls are let through.
#[derive(Clone, Debug)]
pub(super) struct Tool {
    pub name: &'static str,
    pub policy: Policy,
    /// The tool's `configs` entry, for the web tools' domain lists.
    pub config: Value,
}

impl Tool {
    /// What a call gets under this tool's policy.
    pub(super) fn evaluate(&self) -> (&'static str, Value) {
        self.policy.evaluate(sandboxed(self.name))
    }
}

/// The enabled tools of a normalized toolset.
pub(super) fn enabled(toolset: &Value) -> Vec<Tool> {
    TOOL_NAMES
        .iter()
        .filter_map(|name| {
            let config = toolset
                .get("configs")
                .and_then(Value::as_array)?
                .iter()
                .find(|c| c.get("name").and_then(Value::as_str) == Some(name))?
                .clone();
            let (on, policy) = resolve_config(toolset, name, Policy::AlwaysAllow).ok()?;
            on.then_some(Tool {
                name,
                policy,
                config,
            })
        })
        .collect()
}

/// The declaration a container tool is offered to the model under, in the
/// Messages API's client tool shape. `None` for the web tools, which the
/// server tools declare.
pub(super) fn definition(name: &str) -> Option<Value> {
    let (description, schema) = match name {
        "bash" => (
            "Run a bash command in the container. The working directory is /workspace; each \
             command starts a fresh shell, and the files it writes there persist.",
            json!({"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout_ms": {"type": "integer"},
            }, "required": ["command"]}),
        ),
        "read" => (
            "Read a file under /workspace, each line numbered. `view_range` [start, end] reads \
             only those lines; an end of 0 or less reads to the end.",
            json!({"type": "object", "properties": {
                "file_path": {"type": "string"},
                "view_range": {"type": "array", "items": {"type": "integer"}},
            }, "required": ["file_path"]}),
        ),
        "write" => (
            "Write a whole file under /workspace, replacing it if it exists.",
            json!({"type": "object", "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            }, "required": ["file_path", "content"]}),
        ),
        "edit" => (
            "Replace `old_string` with `new_string` in a file under /workspace. `old_string` \
             must occur exactly once, unless `replace_all` replaces every occurrence.",
            json!({"type": "object", "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            }, "required": ["file_path", "old_string", "new_string"]}),
        ),
        "glob" => (
            "Find files under /workspace whose path matches a glob pattern such as `**/*.py`, \
             newest first. `path` is the directory to search from.",
            json!({"type": "object", "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            }, "required": ["pattern"]}),
        ),
        "grep" => (
            "Search the files under /workspace (or `path`) for an extended regular \
             expression; prints each matching line with its file and line number.",
            json!({"type": "object", "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            }, "required": ["pattern"]}),
        ),
        _ => return None,
    };
    Some(json!({"name": name, "description": description, "input_schema": schema}))
}

/// `path` relative to the directory the container shows at `at`, when it
/// is that directory or under it.
fn under(at: &str, path: &str) -> Option<String> {
    let rest = path
        .strip_prefix(at)
        .filter(|r| r.is_empty() || r.starts_with('/'))?;
    let rest = rest.trim_start_matches('/');
    Some(if rest.is_empty() {
        ".".into()
    } else {
        rest.into()
    })
}

/// Which of the directories at `areas` (the workspace first) a path the
/// model gave is in, and the path relative to it: a relative path is in
/// the workspace, and an absolute one outside them all is refused.
fn locate(areas: &[&str], path: &str) -> Result<(usize, String), String> {
    let path = path.trim();
    if !path.starts_with('/') {
        return Ok((
            0,
            if path.is_empty() {
                ".".into()
            } else {
                path.into()
            },
        ));
    }
    let mut best: Option<(usize, String)> = None;
    for (i, at) in areas.iter().enumerate() {
        if let Some(rel) = under(at, path)
            && best
                .as_ref()
                .is_none_or(|(b, _)| areas[*b].len() < at.len())
        {
            best = Some((i, rel));
        }
    }
    best.ok_or_else(|| {
        format!(
            "{path} is outside the workspace: every file is under {}",
            areas.join(", ")
        )
    })
}

/// Where a path relative to the directory shown at `at` shows in the
/// container.
fn shown(at: &str, rel: &Path) -> String {
    let rel = rel.to_string_lossy();
    let rel = rel.trim_start_matches("./");
    if rel.is_empty() || rel == "." {
        at.into()
    } else {
        format!("{at}/{rel}")
    }
}

fn string<'a>(input: &'a Map<String, Value>, key: &str) -> Result<&'a str, String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("the `{key}` input is missing"))
}

/// A directory of the container the file tools reach: where the
/// container shows it, where it is on the host, and whether the agent may
/// write in it.
struct Area {
    at: String,
    host: PathBuf,
    writable: bool,
    files: orangu::tools::ToolExecutor,
}

/// The container tools of one session, run against its container.
pub(super) struct Runner {
    pub container: Container,
    /// `/workspace` first, then the container's mounts.
    areas: Vec<Area>,
}

impl Runner {
    pub(super) fn new(container: Container) -> Self {
        let area = |at: &str, host: PathBuf, writable: bool| {
            let files = orangu::tools::ToolExecutor::with_config(&host, false, 0, 20, None);
            files.set_licence_choice(orangu::license::Choice::None);
            Area {
                at: at.to_string(),
                host,
                writable,
                files,
            }
        };
        let mut areas = vec![area("/workspace", container.workspace(), true)];
        for mount in container.mounts() {
            areas.push(area(&mount.at, mount.host.clone(), mount.writable));
        }
        Self { container, areas }
    }

    /// The area a path the model gave is in, and the path relative to it.
    fn locate(&self, path: &str) -> Result<(&Area, String), String> {
        let ats: Vec<&str> = self.areas.iter().map(|a| a.at.as_str()).collect();
        let (i, rel) = locate(&ats, path)?;
        Ok((&self.areas[i], rel))
    }

    /// [`locate`](Self::locate), for a path the agent writes.
    fn locate_writable(&self, path: &str) -> Result<(&Area, String), String> {
        let (area, rel) = self.locate(path)?;
        if !area.writable {
            return Err(format!("{} is read-only", shown(&area.at, Path::new(&rel))));
        }
        Ok((area, rel))
    }

    /// Runs one call: what the model is told, and whether it failed.
    pub(super) async fn run(&self, name: &str, input: &Map<String, Value>) -> (String, bool) {
        let result = match name {
            "bash" => self.bash(input).await,
            "read" => self.read(input).await,
            "write" => self.write(input).await,
            "edit" => self.edit(input).await,
            "glob" => self.glob(input).await,
            "grep" => self.grep(input).await,
            _ => Err(format!("there is no tool named `{name}`")),
        };
        match result {
            Ok(text) => (text, false),
            Err(message) => (format!("Error: {message}"), true),
        }
    }

    /// The picture `read` was asked for, as its media type and bytes;
    /// `None` when the file is not a picture.
    pub(super) fn picture(
        &self,
        input: &Map<String, Value>,
    ) -> Result<Option<(&'static str, Vec<u8>)>, String> {
        let (area, rel) = self.locate(string(input, "file_path")?)?;
        area.files
            .read_picture(&rel)
            .map_err(|err| format!("{}: {err}", shown(&area.at, Path::new(&rel))))
    }

    async fn bash(&self, input: &Map<String, Value>) -> Result<String, String> {
        let command = string(input, "command")?;
        let out = self
            .container
            .run(&["/bin/bash", "-c", command], None)
            .await?;
        let mut text = run_text(&out.stdout, &out.stderr, out.code);
        if out.timed_out {
            text.push_str("\nThe command ran out of time and was stopped.");
        }
        Ok(text)
    }

    async fn read(&self, input: &Map<String, Value>) -> Result<String, String> {
        let (area, rel) = self.locate(string(input, "file_path")?)?;
        let mut args = Map::new();
        args.insert("path".into(), json!(rel));
        if let Some(range) = input.get("view_range").and_then(Value::as_array) {
            let at = |i: usize| range.get(i).and_then(Value::as_i64);
            if let Some(start) = at(0).filter(|s| *s > 0) {
                args.insert("start_line".into(), json!(start));
            }
            if let Some(end) = at(1).filter(|e| *e > 0) {
                args.insert("end_line".into(), json!(end));
            }
        }
        let text = area
            .files
            .execute("show_file", &args)
            .await
            .map_err(|err| err.to_string())?;
        Ok(if text.is_empty() {
            "(the file is empty)".into()
        } else {
            text
        })
    }

    async fn write(&self, input: &Map<String, Value>) -> Result<String, String> {
        let (area, rel) = self.locate_writable(string(input, "file_path")?)?;
        let content = string(input, "content")?;
        let args = json!({
            "path": rel, "content": content, "overwrite": true, "parents": true, "git": false,
        });
        area.files
            .execute("create_file", args.as_object().expect("an object"))
            .await
            .map_err(|err| err.to_string())?;
        Ok(format!(
            "Wrote {} bytes to {}.",
            content.len(),
            shown(&area.at, Path::new(&rel))
        ))
    }

    async fn edit(&self, input: &Map<String, Value>) -> Result<String, String> {
        let (area, rel) = self.locate_writable(string(input, "file_path")?)?;
        let old = string(input, "old_string")?;
        if old.is_empty() {
            return Err("`old_string` is empty: write the file to replace it whole".into());
        }
        let args = json!({
            "path": rel, "old_text": old, "new_text": string(input, "new_string")?,
            "replace_all": input.get("replace_all").and_then(Value::as_bool).unwrap_or(false),
            "git": false,
        });
        let out = area
            .files
            .execute("modify_file", args.as_object().expect("an object"))
            .await
            .map_err(|err| format!("{}: {err}", shown(&area.at, Path::new(&rel))))?;
        let replaced = serde_json::from_str::<Value>(&out)
            .ok()
            .and_then(|v| v["replacements"].as_u64())
            .unwrap_or(1);
        Ok(format!(
            "Edited {}: {replaced} replacement{}.",
            shown(&area.at, Path::new(&rel)),
            if replaced == 1 { "" } else { "s" }
        ))
    }

    async fn glob(&self, input: &Map<String, Value>) -> Result<String, String> {
        let mut pattern = string(input, "pattern")?.to_string();
        let (mut area, mut base) =
            self.locate(input.get("path").and_then(Value::as_str).unwrap_or("."))?;
        if pattern.starts_with('/') {
            (area, pattern) = self.locate(&pattern)?;
            base = ".".into();
        }
        let args = json!({"path": base, "pattern": pattern});
        let listed = area
            .files
            .execute("list_directory", args.as_object().expect("an object"))
            .await
            .map_err(|err| err.to_string())?;
        // The workspace's paths as the container names them; the notes
        // around them as they are.
        Ok(listed
            .lines()
            .map(|line| match line {
                "No files match." => line.to_string(),
                _ if line.starts_with('(') && line.ends_with("more not shown)") => line.to_string(),
                _ => shown(&area.at, Path::new(line)),
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }

    async fn grep(&self, input: &Map<String, Value>) -> Result<String, String> {
        let pattern = string(input, "pattern")?;
        let (area, rel) = self.locate(input.get("path").and_then(Value::as_str).unwrap_or("."))?;
        orangu::tools::resolve_workspace_path(&area.host, &rel).map_err(|err| err.to_string())?;
        let target = shown(&area.at, Path::new(&rel));
        let out = self
            .container
            .run(&["grep", "-rnIE", "--", pattern, &target], None)
            .await?;
        match out.code {
            0 => Ok(out.stdout),
            1 if out.stderr.is_empty() => Ok("No matches.".into()),
            _ => Err(if out.stderr.is_empty() {
                format!("grep exited with {}", out.code)
            } else {
                out.stderr
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_toolset_enables_every_tool_and_allows_them() {
        let toolset = normalize(&json!({"type": TOOLSET})).unwrap();
        let tools = enabled(&toolset);
        assert_eq!(tools.len(), 8);
        assert!(tools.iter().all(|t| t.policy == Policy::AlwaysAllow));
    }

    #[test]
    fn a_config_entry_overrides_the_default() {
        let toolset = normalize(&json!({
            "type": TOOLSET,
            "default_config": {"permission_policy": {"type": "auto"}},
            "configs": [
                {"name": "bash", "permission_policy": {"type": "always_ask"}},
                {"name": "web_fetch", "enabled": false},
                {"name": "web_search", "enabled": false},
            ],
        }))
        .unwrap();
        let tools = enabled(&toolset);
        let names: Vec<_> = tools.iter().map(|t| t.name).collect();
        assert_eq!(names, ["bash", "read", "write", "edit", "glob", "grep"]);
        assert_eq!(tools[0].evaluate().0, "ask");
        assert_eq!(tools[1].evaluate().0, "allow");
        assert_eq!(
            tools[1].evaluate().1,
            json!({"type": "auto", "evaluated_permission": {"type": "allow"}})
        );
    }

    #[test]
    fn auto_asks_about_a_tool_that_leaves_the_container() {
        let (permission, evaluation) = Policy::Auto.evaluate(false);
        assert_eq!(permission, "ask");
        assert_eq!(
            evaluation["evaluated_permission"]["reason_code"],
            "indeterminate"
        );
    }

    #[test]
    fn an_unknown_tool_name_is_refused() {
        let err =
            normalize(&json!({"type": TOOLSET, "configs": [{"name": "browser"}]})).unwrap_err();
        assert!(err.contains("browser"), "{err}");
    }

    #[test]
    fn an_mcp_toolset_must_name_a_listed_server_and_asks_by_default() {
        let servers = [json!({"type": "url", "name": "docs", "url": "http://x"})];
        assert!(normalize_mcp(&json!({"mcp_server_name": "other"}), &servers).is_err());
        let toolset = normalize_mcp(&json!({"mcp_server_name": "docs"}), &servers).unwrap();
        assert_eq!(
            toolset["default_config"]["permission_policy"]["type"],
            "always_ask"
        );
    }

    #[test]
    fn paths_resolve_inside_the_workspace_and_the_mounts() {
        let areas = ["/workspace", "/mnt/session/uploads", "/mnt/session/outputs"];
        let at = |path: &str| locate(&areas, path);
        assert_eq!(at("/workspace/a/b.txt").unwrap(), (0, "a/b.txt".into()));
        assert_eq!(at("/workspace").unwrap(), (0, ".".into()));
        assert_eq!(at("notes.md").unwrap(), (0, "notes.md".into()));
        assert_eq!(
            at("/mnt/session/outputs/r.csv").unwrap(),
            (2, "r.csv".into())
        );
        assert!(at("/etc/passwd").is_err());
        assert!(at("/workspacefoo").is_err());
        assert!(at("/mnt/session").is_err());
    }

    async fn runner(dir: &Path) -> Runner {
        Runner::new(Container::open(dir, None).await.unwrap())
    }

    fn input(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[tokio::test]
    async fn the_file_tools_write_read_edit_and_find() {
        let dir = tempfile::tempdir().unwrap();
        let r = runner(dir.path()).await;
        let (text, err) = r
            .run(
                "write",
                &input(json!({"file_path": "/workspace/src/a.py", "content": "x = 1\ny = 1\n"})),
            )
            .await;
        assert!(!err, "{text}");
        let (text, _) = r
            .run("read", &input(json!({"file_path": "src/a.py"})))
            .await;
        assert_eq!(text, "1. x = 1\n2. y = 1");
        let (text, err) = r
            .run(
                "edit",
                &input(json!({"file_path": "src/a.py", "old_string": "= 1", "new_string": "= 2"})),
            )
            .await;
        assert!(err && text.contains("2 times"), "{text}");
        let (text, err) = r
            .run(
                "edit",
                &input(json!({
                    "file_path": "src/a.py", "old_string": "y = 1", "new_string": "y = 2",
                })),
            )
            .await;
        assert!(!err, "{text}");
        let (text, _) = r
            .run(
                "read",
                &input(json!({"file_path": "src/a.py", "view_range": [2, 0]})),
            )
            .await;
        assert_eq!(text, "2. y = 2");
        let (text, _) = r.run("glob", &input(json!({"pattern": "**/*.py"}))).await;
        assert_eq!(text, "/workspace/src/a.py");
        let (text, err) = r
            .run("read", &input(json!({"file_path": "/etc/passwd"})))
            .await;
        assert!(err && text.contains("outside the workspace"), "{text}");
        // A new file is written as given: no licence header on top.
        let written = std::fs::read_to_string(r.container.workspace().join("src/a.py")).unwrap();
        assert_eq!(written, "x = 1\ny = 2\n");
    }

    /// A mount is reached by its own path, and a read-only one refuses a
    /// write.
    #[tokio::test]
    async fn the_file_tools_reach_the_mounts_and_respect_read_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut container = Container::open(dir.path(), None).await.unwrap();
        let (uploads, outputs) = (dir.path().join("up"), dir.path().join("out"));
        std::fs::create_dir_all(&uploads).unwrap();
        std::fs::create_dir_all(&outputs).unwrap();
        std::fs::write(uploads.join("data.csv"), "a,b\n").unwrap();
        container.mount(uploads.clone(), "/mnt/session/uploads", false);
        container.mount(outputs.clone(), "/mnt/session/outputs", true);
        let r = Runner::new(container);
        let (text, _) = r
            .run(
                "read",
                &input(json!({"file_path": "/mnt/session/uploads/data.csv"})),
            )
            .await;
        assert_eq!(text, "1. a,b");
        let (text, err) = r
            .run(
                "write",
                &input(json!({"file_path": "/mnt/session/uploads/x", "content": "y"})),
            )
            .await;
        assert!(err && text.contains("read-only"), "{text}");
        let (text, err) = r
            .run(
                "write",
                &input(json!({"file_path": "/mnt/session/outputs/r.txt", "content": "done"})),
            )
            .await;
        assert!(!err, "{text}");
        assert_eq!(
            std::fs::read_to_string(outputs.join("r.txt")).unwrap(),
            "done"
        );
        let (text, _) = r
            .run(
                "glob",
                &input(json!({"pattern": "/mnt/session/outputs/**/*"})),
            )
            .await;
        assert!(text.contains("/mnt/session/outputs/r.txt"), "{text}");
    }

    #[tokio::test]
    async fn read_finds_a_picture_by_its_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let r = runner(dir.path()).await;
        let png = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        std::fs::write(r.container.workspace().join("plot.png"), png).unwrap();
        std::fs::write(r.container.workspace().join("a.txt"), "x").unwrap();
        let picture = r.picture(&input(json!({"file_path": "/workspace/plot.png"})));
        assert_eq!(picture.unwrap().unwrap().0, "image/png");
        assert!(
            r.picture(&input(json!({"file_path": "a.txt"})))
                .unwrap()
                .is_none()
        );
        assert!(
            r.picture(&input(json!({"file_path": "/etc/passwd"})))
                .is_err()
        );
    }

    #[tokio::test]
    async fn a_link_out_of_the_workspace_is_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let r = runner(dir.path()).await;
        std::os::unix::fs::symlink("/etc", r.container.workspace().join("etc")).unwrap();
        let (text, err) = r
            .run("read", &input(json!({"file_path": "etc/passwd"})))
            .await;
        assert!(err, "{text}");
    }
}
