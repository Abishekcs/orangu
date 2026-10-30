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

//! Programmatic tool calling: client tools whose `allowed_callers` name a
//! code execution version are async Python functions in the code the model
//! runs. The code runs in the container under a small harness that sends
//! each call to the server on its stdout and waits for the answer on its
//! stdin. When the code calls tools, the reply stops with those calls as
//! `tool_use` blocks naming the run as their `caller`, and the run waits,
//! paused, for the client's `tool_result`s on the next request; when the
//! code ends, its output is the `code_execution_tool_result` the model goes
//! on from.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};

use super::containers::Container;
use super::files::new_id;

/// The caller a programmatic call is tagged with.
pub(super) const CALLER: &str = "code_execution_20260120";

/// The code execution versions a tool's `allowed_callers` may name.
const CALLERS: [&str; 2] = ["code_execution_20260120", "code_execution_20260521"];

/// How long a paused run waits for the client's results.
const PAUSE_LIMIT: Duration = Duration::from_secs(270);

/// How long the code may run between tool calls.
const RUN_LIMIT: Duration = Duration::from_secs(120);

/// How long after one call the code has to make the next before the calls
/// so far are handed to the client — what lets `asyncio.gather` arrive as
/// one batch.
const BATCH_WINDOW: Duration = Duration::from_millis(200);

/// What marks a harness line on the code's stdout.
const MARK: &str = "\u{1e}PTC ";

/// Whether the code may call `tool`.
pub(super) fn callable(tool: &Value) -> bool {
    tool.get("allowed_callers")
        .and_then(Value::as_array)
        .is_some_and(|c| {
            c.iter()
                .any(|v| v.as_str().is_some_and(|v| CALLERS.contains(&v)))
        })
}

/// Whether the model may call `tool` itself.
pub(super) fn direct(tool: &Value) -> bool {
    match tool.get("allowed_callers").and_then(Value::as_array) {
        None => true,
        Some(callers) => callers.iter().any(|v| v == "direct"),
    }
}

/// Whether a `tool_use` block was made by running code.
pub(super) fn is_programmatic(block: &Value) -> bool {
    block["type"] == "tool_use"
        && block["caller"]["type"]
            .as_str()
            .is_some_and(|t| t.starts_with("code_execution_"))
}

/// The `code_execution` tool's description: the functions its code can
/// call, each with what it takes.
pub(super) fn description(tools: &[Value]) -> String {
    let example = tools
        .first()
        .and_then(|t| t["name"].as_str())
        .unwrap_or("tool");
    let mut text = format!(
        "Run Python code in the code execution container and return what it prints. The \
         functions below are already defined in the code — no import — as async functions \
         that call tools: each takes one dict of arguments and returns the tool's result as a \
         string (json.loads it when it is JSON). Use them with top-level await, for example \
         `result = await {example}({{...}})`, or several at once with \
         `await asyncio.gather(...)`, and print what you need to see: only the printed \
         output comes back to you."
    );
    for tool in tools {
        text.push_str(&format!(
            "\n- {}(args): {} Arguments: {}",
            tool["name"].as_str().unwrap_or_default(),
            tool["description"].as_str().unwrap_or_default(),
            tool.get("input_schema").cloned().unwrap_or(json!({})),
        ));
    }
    text
}

/// The harness the code runs under, with `names` as its functions.
fn harness(names: &[String]) -> String {
    let mut tools = String::new();
    for name in names {
        tools.push_str(&format!("{name} = _tool({})\n", json!(name)));
    }
    format!(
        r#"import ast, asyncio, contextlib, inspect, io, json, sys, threading, traceback
_mark = {mark}
_out = sys.__stdout__
_pending = {{}}
_count = 0
_lock = threading.Lock()
def _send(message):
    with _lock:
        _out.write(_mark + json.dumps(message) + "\n")
        _out.flush()
def _tool(name):
    async def call(args=None, **kwargs):
        global _count
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with _lock:
            _count += 1
            number = _count
            _pending[number] = (loop, future)
        _send({{"call": number, "name": name, "input": args if args is not None else kwargs}})
        return await future
    call.__name__ = name
    return call
{tools}def _settle(future, content):
    if not future.done():
        future.set_result(content)
def _answers():
    for line in sys.stdin:
        answer = json.loads(line)
        with _lock:
            waiting = _pending.pop(answer["call"], None)
        if waiting is not None:
            loop, future = waiting
            loop.call_soon_threadsafe(_settle, future, answer["content"])
def _main(source):
    threading.Thread(target=_answers, daemon=True).start()
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            compiled = compile(source, "<code>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            result = eval(compiled, globals())
            if inspect.iscoroutine(result):
                asyncio.run(result)
        except SystemExit as stop:
            code = stop.code if isinstance(stop.code, int) else 1
        except BaseException:
            traceback.print_exc()
            code = 1
    _send({{"done": True, "stdout": out.getvalue(), "stderr": err.getvalue(), "return_code": code}})
_main(json.loads(sys.stdin.readline())["code"])
"#,
        mark = json!(MARK),
    )
}

/// Where a run stands.
pub(super) enum Step {
    /// The code called these tools — `(call number, name, input)` — and
    /// waits for their results.
    Calls(Vec<(u64, String, Value)>),
    /// The code ended.
    Done {
        stdout: String,
        stderr: String,
        return_code: i64,
    },
    /// It could not go on: the error code, and why.
    Failed(&'static str, String),
}

/// A run of code that calls tools.
pub(super) struct Run {
    _child: Child,
    stdin: ChildStdin,
    lines: tokio::io::Lines<BufReader<ChildStdout>>,
    /// What the code printed that the harness did not capture (a
    /// subprocess's output).
    stray: String,
    /// The call number of each `tool_use` id handed to the client.
    calls: HashMap<String, u64>,
    paused_at: Instant,
}

impl Run {
    /// Starts `code` in `container`, its functions `names`, and runs it to
    /// its first stop.
    pub(super) async fn start(
        container: &Container,
        code: &str,
        names: &[String],
    ) -> Result<(Self, Step), String> {
        let source = harness(names);
        let mut child = container.spawn(&["python3", "-c", &source])?;
        let (Some(mut stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err("the sandbox gave the code no pipes".into());
        };
        let first = format!("{}\n", json!({"code": code}));
        stdin
            .write_all(first.as_bytes())
            .await
            .map_err(|err| format!("starting the code: {err}"))?;
        let mut run = Self {
            _child: child,
            stdin,
            lines: BufReader::new(stdout).lines(),
            stray: String::new(),
            calls: HashMap::new(),
            paused_at: Instant::now(),
        };
        let step = run.step().await;
        Ok((run, step))
    }

    /// Hands the code its results — `(tool_use id, text)` — and runs it to
    /// its next stop.
    pub(super) async fn resume(&mut self, results: Vec<(String, String)>) -> Step {
        for (id, content) in results {
            let Some(number) = self.calls.remove(&id) else {
                continue;
            };
            let line = format!("{}\n", json!({"call": number, "content": content}));
            if self.stdin.write_all(line.as_bytes()).await.is_err() {
                return Step::Failed("unavailable", "the code stopped running".into());
            }
        }
        self.step().await
    }

    /// The `tool_use` blocks for `calls`, made by the code execution call
    /// `caller`: each given an id the client answers it by.
    pub(super) fn blocks(&mut self, caller: &str, calls: Vec<(u64, String, Value)>) -> Vec<Value> {
        calls
            .into_iter()
            .map(|(number, name, input)| {
                let id = new_id("toolu_");
                self.calls.insert(id.clone(), number);
                json!({
                    "type": "tool_use", "id": id, "name": name, "input": input,
                    "caller": {"type": CALLER, "tool_id": caller},
                })
            })
            .collect()
    }

    /// Reads the code's output to its next stop: its calls, gathered while
    /// they keep coming, or its end.
    async fn step(&mut self) -> Step {
        let mut calls = Vec::new();
        loop {
            let wait = if calls.is_empty() {
                RUN_LIMIT
            } else {
                BATCH_WINDOW
            };
            let line = match tokio::time::timeout(wait, self.lines.next_line()).await {
                Err(_) if !calls.is_empty() => return Step::Calls(calls),
                Err(_) => {
                    return Step::Failed(
                        "execution_time_exceeded",
                        "the code ran past its time limit and was stopped".into(),
                    );
                }
                Ok(Ok(Some(line))) => line,
                Ok(_) => {
                    return Step::Failed(
                        "unavailable",
                        format!("the code stopped without finishing{}", self.stray_note()),
                    );
                }
            };
            let Some(message) = line
                .strip_prefix(MARK)
                .and_then(|m| serde_json::from_str::<Value>(m).ok())
            else {
                self.stray.push_str(&line);
                self.stray.push('\n');
                continue;
            };
            if message["done"] == true {
                let mut stdout = message["stdout"].as_str().unwrap_or_default().to_string();
                stdout.push_str(&std::mem::take(&mut self.stray));
                return Step::Done {
                    stdout,
                    stderr: message["stderr"].as_str().unwrap_or_default().to_string(),
                    return_code: message["return_code"].as_i64().unwrap_or(1),
                };
            }
            if let (Some(number), Some(name)) = (message["call"].as_u64(), message["name"].as_str())
            {
                calls.push((number, name.to_string(), message["input"].clone()));
            }
        }
    }

    fn stray_note(&self) -> String {
        if self.stray.is_empty() {
            String::new()
        } else {
            format!(": {}", self.stray.trim())
        }
    }
}

/// Runs paused for their results, by the id of the code execution call
/// that started each.
static PAUSED: LazyLock<Mutex<HashMap<String, Run>>> = LazyLock::new(Default::default);

fn paused() -> std::sync::MutexGuard<'static, HashMap<String, Run>> {
    let mut all = PAUSED.lock().unwrap_or_else(|p| p.into_inner());
    all.retain(|_, run| run.paused_at.elapsed() < PAUSE_LIMIT);
    all
}

/// Keeps `run`, which the code execution call `id` started, until its
/// results come.
pub(super) fn pause(id: &str, mut run: Run) {
    run.paused_at = Instant::now();
    paused().insert(id.to_string(), run);
}

/// The run the code execution call `id` started, if it is still waiting.
pub(super) fn take(id: &str) -> Option<Run> {
    paused().remove(id)
}

/// A code execution call's result block, and what the model is told.
pub(super) fn result_block(id: &str, step: &Step) -> (Value, String) {
    match step {
        Step::Done {
            stdout,
            stderr,
            return_code,
        } => {
            let mut text = format!("exit code: {return_code}");
            if stdout.is_empty() && stderr.is_empty() {
                text.push_str("\nThe code printed nothing.");
            }
            if !stdout.is_empty() {
                text.push_str(&format!("\nstdout:\n{stdout}"));
            }
            if !stderr.is_empty() {
                text.push_str(&format!("\nstderr:\n{stderr}"));
            }
            (
                json!({"type": "code_execution_tool_result", "tool_use_id": id, "content": {
                    "type": "code_execution_result", "stdout": stdout, "stderr": stderr,
                    "return_code": return_code, "content": [],
                }}),
                text,
            )
        }
        Step::Failed(code, why) => (
            json!({"type": "code_execution_tool_result", "tool_use_id": id, "content": {
                "type": "code_execution_tool_result_error", "error_code": code,
            }}),
            format!("Error: the code did not finish ({code}): {why}."),
        ),
        Step::Calls(_) => (Value::Null, String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_callers_decide_who_may_call() {
        let tool = |callers: Value| json!({"name": "q", "allowed_callers": callers});
        assert!(callable(&tool(json!(["code_execution_20260120"]))));
        assert!(!direct(&tool(json!(["code_execution_20260120"]))));
        assert!(direct(&tool(json!(["direct", "code_execution_20260521"]))));
        assert!(callable(&tool(json!([
            "direct",
            "code_execution_20260521"
        ]))));
        assert!(direct(&json!({"name": "q"})) && !callable(&json!({"name": "q"})));
    }

    /// The harness speaks its protocol: the code's gathered calls arrive as
    /// lines, the answers resume it, and its output ends the run — with
    /// top-level `await` and with the code's own `asyncio.run` alike.
    #[test]
    fn the_harness_pauses_for_calls_and_resumes_on_answers() {
        use std::io::{BufRead as _, Write as _};
        let source = harness(&["query".into()]);
        for code in [
            "import asyncio, json\nrows = await asyncio.gather(*(query({'n': i}) for i in range(3)))\nprint(sum(json.loads(r) for r in rows))",
            "import asyncio, json\nasync def main():\n    rows = await asyncio.gather(*(query({'n': i}) for i in range(3)))\n    print(sum(json.loads(r) for r in rows))\nasyncio.run(main())",
        ] {
            let Ok(mut child) = std::process::Command::new("python3")
                .args(["-I", "-c", &source])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
            else {
                eprintln!("python3 is not installed; skipping");
                return;
            };
            let mut stdin = child.stdin.take().unwrap();
            let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
            writeln!(stdin, "{}", json!({"code": code})).unwrap();
            let mut calls = Vec::new();
            while calls.len() < 3 {
                let line = lines.next().unwrap().unwrap();
                let message: Value =
                    serde_json::from_str(line.strip_prefix(MARK).unwrap()).unwrap();
                calls.push(message);
            }
            for call in &calls {
                let n = call["input"]["n"].as_u64().unwrap();
                writeln!(
                    stdin,
                    "{}",
                    json!({"call": call["call"], "content": (n * 10).to_string()})
                )
                .unwrap();
            }
            let done: Value =
                serde_json::from_str(lines.next().unwrap().unwrap().strip_prefix(MARK).unwrap())
                    .unwrap();
            assert_eq!(done["done"], true, "{done}");
            assert_eq!(done["stdout"], "30\n", "{done}");
            assert_eq!(done["return_code"], 0);
            let _ = child.wait();
        }
    }

    /// The harness is valid Python, whatever the tools are named.
    #[test]
    fn the_harness_compiles() {
        let source = harness(&["query_database".into(), "get_weather".into()]);
        let checked = std::process::Command::new("python3")
            .args(["-I", "-c", "import ast,sys; ast.parse(sys.stdin.read())"])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write as _;
                child.stdin.take().unwrap().write_all(source.as_bytes())?;
                child.wait()
            });
        match checked {
            Ok(status) => assert!(status.success(), "{source}"),
            Err(_) => eprintln!("python3 is not installed; skipping"),
        }
    }
}
