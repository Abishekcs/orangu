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

//! Dynamic workflows: a plan the session's agent writes, as Python, that
//! runs many agents in phases. The server runs the plan in the session's
//! container in the background — a workflow run — while the agent works on
//! or ends its turn. Each agent the plan calls works in a session thread of
//! its own, and its final message is what the plan's call returns; when
//! the plan ends, the agent is sent a notice with what it printed and takes
//! a turn on it. The client follows a run by its `workflow_run.*` events
//! on the primary thread's stream.

use super::*;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{Semaphore, oneshot};
use tokio::task::{JoinHandle, JoinSet};

/// The tool that starts a run.
pub(super) const START: &str = "ant__start_workflow_run";

/// The tool that stops one.
pub(super) const STOP: &str = "ant__stop_workflow_run";

pub(super) const TOOLS: [&str; 2] = [START, STOP];

/// The most runs a session has open at once.
const MAX_OPEN: usize = 10;

/// The most of a run's agents that work at once; the others wait.
const MAX_WORKING: usize = 64;

/// The most threads a run makes over its life.
const MAX_THREADS: usize = 1000;

/// A run's lifetime, unless its agent gives it a shorter one.
const LIFETIME_MINUTES: u64 = 24 * 60;

const NAME_CHARS: usize = 64;
const DESCRIPTION_CHARS: usize = 256;

/// The most of what a plan printed that its notice carries.
const OUTPUT_CHARS: usize = 20_000;

/// How long the plan's code has to compile before the start is refused.
const COMPILE_LIMIT: Duration = Duration::from_secs(30);

/// How long a stopped thread has to wind down before it is archived.
const WIND_DOWN: Duration = Duration::from_secs(30);

/// What marks a harness line on the plan's stdout.
const MARK: &str = "\u{1e}WFR ";

/// An open run.
pub(super) struct Run {
    id: String,
    session: String,
    name: String,
    /// Its plan's code.
    code: String,
    /// Wakes the run to end it early.
    stop: Notify,
    /// Why it is ended early: the agent stopped it (`stopped`), the session
    /// was archived (`archived`) or deleted (`gone`).
    stopping: StdMutex<Option<&'static str>>,
    /// The keys of the threads it made.
    threads: StdMutex<Vec<String>>,
    task: StdMutex<Option<JoinHandle<()>>>,
}

/// Open runs by id.
static RUNS: LazyLock<StdMutex<HashMap<String, Arc<Run>>>> = LazyLock::new(Default::default);

fn runs() -> std::sync::MutexGuard<'static, HashMap<String, Arc<Run>>> {
    RUNS.lock().unwrap_or_else(|e| e.into_inner())
}

/// What each of a run's threads hands its plan when its turn ends, by the
/// thread's key.
type Report = Result<String, String>;
static WAITERS: LazyLock<StdMutex<HashMap<String, oneshot::Sender<Report>>>> =
    LazyLock::new(Default::default);

fn open_runs(session: &str) -> Vec<Arc<Run>> {
    runs()
        .values()
        .filter(|r| r.session == session)
        .cloned()
        .collect()
}

/// Whether session `session` has a workflow run open.
pub(super) fn any_open(session: &str) -> bool {
    runs().values().any(|r| r.session == session)
}

/// Whether the run `id` is open.
pub(super) fn is_open(id: &str) -> bool {
    runs().contains_key(id)
}

/// The workflow run the thread `key` belongs to, if a run made it.
pub(super) fn run_of(state: &AppState, key: &str) -> Option<String> {
    split_key(key).1?;
    load(state, key)?["_thread"]["workflow_run_id"]
        .as_str()
        .map(str::to_string)
}

/// Hands a run's thread's report to the plan waiting on it.
pub(super) fn thread_done(key: &str, report: Report) {
    let waiter = WAITERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(key);
    if let Some(waiter) = waiter {
        let _ = waiter.send(report);
    }
}

/// The `400` of a change refused while a run is open.
pub(super) fn open_error(message: impl Into<String>) -> Box<Response> {
    Box::new(
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"type": "error", "error": {
                "type": "invalid_request_error", "message": message.into(),
                "details": {"error_code": "workflow_run_open"},
            }})),
        )
            .into_response(),
    )
}

/// The start and stop tools, for an agent whose session runs workflows
/// with `workflows`' agents.
pub(super) fn definitions(workflows: &Value) -> Vec<Value> {
    let inline = workflows["inline"] == true;
    let listed: Vec<String> = workflows["agents"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|a| {
            let name = a["name"].as_str().unwrap_or("agent");
            match a["description"].as_str().filter(|d| !d.is_empty()) {
                Some(description) => format!("`{name}` ({description})"),
                None => format!("`{name}`"),
            }
        })
        .collect();
    let mut agent = "- `await agent(prompt, name=None, system=None, use=None)` runs one agent on \
                     `prompt` — a task complete in itself — in a thread of its own, and returns \
                     its final message as a string. It raises `AgentFailed` when the agent \
                     fails."
        .to_string();
    if inline {
        agent.push_str(
            " An agent you define this way gets `system` as its system prompt, and your \
             model and tools; `name` names it.",
        );
    }
    if !listed.is_empty() {
        agent.push_str(&format!(
            " `use=` runs one of these agents instead, with its own prompt and tools: {}.",
            listed.join(", ")
        ));
        if !inline {
            agent.push_str(" Every agent of a run is one of these, named with `use=`.");
        }
    }
    let description = format!(
        "Start a workflow run: a plan, written as Python, that runs many agents in parallel \
         and in phases, in the background, and combines what they return. Use it for work \
         with many independent pieces — many files, documents, sources or items to handle the \
         same way — or work that finishes sooner in parallel; do small tasks yourself. It \
         returns at once with the run's id; when the run ends you are sent a notice with what \
         the plan printed. The plan's code runs with these defined, no import needed:\n\
         {agent}\n\
         - `with phase(name):` marks a stage of the plan; name one of the run's `phases`.\n\
         - Run agents at once with `results = await asyncio.gather(*(agent(...) for item in \
         items), return_exceptions=True)`, and check each with `isinstance(r, Exception)`, so \
         one failed agent does not fail the run.\n\
         - `print()` the combined result: what the plan prints is what you get back.\n\
         Use top-level `await`. Up to {MAX_WORKING} agents work at once (more wait their \
         turn), and a run has at most {MAX_THREADS}. The agents see neither this conversation \
         nor each other's work, so give every prompt all it needs: the task, the paths, what \
         to report. They share the container's files, which the plan can read too — to list \
         the items to work on, say. The time is `datetime.now(timezone.utc)`."
    );
    vec![
        json!({
            "name": START, "description": description,
            "input_schema": {"type": "object", "required": ["name", "code"], "properties": {
                "name": {"type": "string", "description": "A short name for the run."},
                "description": {"type": "string", "description": "What the run does."},
                "phases": {"type": "array", "description": "The plan's stages, in order.",
                           "items": {"type": "object", "required": ["name"], "properties": {
                               "name": {"type": "string"},
                               "description": {"type": "string"},
                           }}},
                "timeout_minutes": {"type": "integer",
                                    "description": "The run's lifetime: 1 to 1440 minutes; \
                                                    1440 when left out."},
                "code": {"type": "string", "description": "The plan: Python code."},
            }},
        }),
        json!({
            "name": STOP,
            "description": "Stop a workflow run you started: its agents stop, and it ends as \
                            stopped. You are sent its notice.",
            "input_schema": {"type": "object", "required": ["workflow_run_id"], "properties": {
                "workflow_run_id": {"type": "string"},
            }},
        }),
    ]
}

/// Runs the agent's call of the start or stop tool: what it is told, and
/// whether that is an error.
pub(super) async fn call(
    state: &Arc<AppState>,
    session: &Value,
    name: &str,
    input: &Map<String, Value>,
) -> (String, bool) {
    let outcome = if name == START {
        start(state, session, input).await
    } else {
        stop_run(session, input)
    };
    match outcome {
        Ok(text) => (text, false),
        Err(text) => (format!("Error: {text}"), true),
    }
}

fn clip(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
}

/// A phase a plan declares.
#[derive(Clone)]
struct Phase {
    id: String,
    name: String,
    description: Option<String>,
}

impl Phase {
    fn json(&self) -> Value {
        json!({"id": self.id, "name": self.name, "description": self.description})
    }
}

fn phases_of(given: Option<&Value>) -> Result<Vec<Phase>, String> {
    let entries = match given {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(entries)) => entries.clone(),
        // A list sent as its JSON text.
        Some(Value::String(text)) => match serde_json::from_str(text) {
            Ok(Value::Array(entries)) => entries,
            _ => return Err("`phases` is a JSON list of phases, not a string".into()),
        },
        Some(_) => return Err("`phases` is a list of phases".into()),
    };
    let mut out = Vec::new();
    for entry in &entries {
        let (name, description) = match entry {
            Value::String(name) => (name.as_str(), None),
            Value::Object(o) => (
                o.get("name")
                    .and_then(Value::as_str)
                    .ok_or("each of `phases` has a `name`")?,
                o.get("description").and_then(Value::as_str),
            ),
            _ => return Err("`phases` lists phases: {\"name\": …, \"description\": …}".into()),
        };
        out.push(Phase {
            id: new_id("wrph_"),
            name: clip(name, NAME_CHARS),
            description: description.map(|d| clip(d, DESCRIPTION_CHARS)),
        });
    }
    Ok(out)
}

/// The harness the plan runs under.
fn harness() -> String {
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
class AgentFailed(Exception):
    pass
async def agent(prompt, name=None, system=None, description=None, use=None):
    global _count
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent() takes the agent's task as its prompt")
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    with _lock:
        _count += 1
        number = _count
        _pending[number] = (loop, future)
    _send({{"agent": number, "prompt": prompt, "name": name, "system": system,
            "description": description, "use": use}})
    ok, text = await future
    if not ok:
        raise AgentFailed(text)
    return text
class phase:
    def __init__(self, name):
        self.name = str(name)
    def __enter__(self):
        _send({{"phase": "start", "name": self.name}})
        return self
    def __exit__(self, *exc):
        _send({{"phase": "end", "name": self.name}})
        return False
    async def __aenter__(self):
        return self.__enter__()
    async def __aexit__(self, *exc):
        return self.__exit__(*exc)
def _settle(future, value):
    if not future.done():
        future.set_result(value)
def _answers():
    for line in sys.stdin:
        answer = json.loads(line)
        with _lock:
            waiting = _pending.pop(answer["agent"], None)
        if waiting is not None:
            loop, future = waiting
            loop.call_soon_threadsafe(_settle, future, (answer["ok"], answer["text"]))
def _main(source):
    try:
        compiled = compile(source, "<plan>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    except (SyntaxError, ValueError):
        _send({{"compiled": False, "error": traceback.format_exc(limit=0)}})
        return
    _send({{"compiled": True}})
    threading.Thread(target=_answers, daemon=True).start()
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
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

/// The plan's process, its pipes, and what it printed past the harness.
struct Plan {
    _child: Child,
    stdin: ChildStdin,
    lines: tokio::io::Lines<BufReader<ChildStdout>>,
    stray: String,
}

impl Plan {
    /// The next harness message; `None` once the plan's output ends.
    async fn next(&mut self) -> Option<Value> {
        loop {
            let line = self.lines.next_line().await.ok()??;
            match line
                .strip_prefix(MARK)
                .and_then(|m| serde_json::from_str::<Value>(m).ok())
            {
                Some(message) => return Some(message),
                None => {
                    self.stray.push_str(&line);
                    self.stray.push('\n');
                }
            }
        }
    }

    async fn answer(&mut self, number: u64, report: Report) {
        let (ok, text) = match report {
            Ok(text) => (true, text),
            Err(text) => (false, text),
        };
        let line = format!("{}\n", json!({"agent": number, "ok": ok, "text": text}));
        let _ = self.stdin.write_all(line.as_bytes()).await;
    }
}

/// Starts the plan's code in the session's container and waits for it to
/// compile: the plan, or why it does not run.
async fn launch(state: &AppState, session: &Value, code: &str) -> Result<Plan, (bool, String)> {
    let mut container = Container::open(&state.api_data, session["_container"].as_str())
        .await
        .map_err(|e| (false, e))?;
    mount_session(&mut container, session);
    let source = harness();
    let mut child = container
        .spawn(&["python3", "-I", "-c", &source])
        .map_err(|e| (false, e))?;
    let (Some(mut stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        return Err((false, "the sandbox gave the plan no pipes".into()));
    };
    let first = format!("{}\n", json!({"code": code}));
    stdin
        .write_all(first.as_bytes())
        .await
        .map_err(|e| (false, format!("starting the plan: {e}")))?;
    let mut plan = Plan {
        _child: child,
        stdin,
        lines: BufReader::new(stdout).lines(),
        stray: String::new(),
    };
    match tokio::time::timeout(COMPILE_LIMIT, plan.next()).await {
        Ok(Some(message)) if message["compiled"] == true => Ok(plan),
        Ok(Some(message)) if message["compiled"] == false => Err((
            true,
            message["error"]
                .as_str()
                .unwrap_or_default()
                .trim()
                .to_string(),
        )),
        _ => Err((
            false,
            format!("the plan did not start{}", stray_note(&plan)),
        )),
    }
}

fn stray_note(plan: &Plan) -> String {
    match plan.stray.trim() {
        "" => String::new(),
        stray => format!(": {}", clip(stray, 2000)),
    }
}

fn error_json(kind: &str, message: &str) -> Value {
    json!({"type": kind, "message": message})
}

async fn start(
    state: &Arc<AppState>,
    session: &Value,
    input: &Map<String, Value>,
) -> Result<String, String> {
    let session_id = session["id"].as_str().unwrap_or_default().to_string();
    let code = input
        .get("code")
        .and_then(Value::as_str)
        .filter(|c| !c.trim().is_empty())
        .ok_or("`code` is required: the plan, as Python")?;
    let phases = phases_of(input.get("phases"))?;
    let minutes = match input.get("timeout_minutes") {
        None | Some(Value::Null) => LIFETIME_MINUTES,
        Some(v) => v
            .as_u64()
            .filter(|m| (1..=LIFETIME_MINUTES).contains(m))
            .ok_or("`timeout_minutes` is 1 to 1440")?,
    };
    // The same plan twice is the same call made again.
    if let Some(running) = open_runs(&session_id).iter().find(|r| r.code == code) {
        return Err(format!(
            "this plan already runs as workflow run {}: wait for its notice instead of \
             starting it again",
            running.id
        ));
    }
    if open_runs(&session_id).len() >= MAX_OPEN {
        emit(
            state,
            &session_id,
            json!({"type": "workflow_run.error", "workflow_run_id": null,
                   "error": error_json("max_workflow_runs_error",
                                       "The session is at its limit of open workflow runs.")}),
        );
        return Err(format!(
            "this session already has {MAX_OPEN} workflow runs open: wait for one to end, or \
             stop one"
        ));
    }
    let plan = match launch(state, session, code).await {
        Ok(plan) => plan,
        Err((compile, why)) => {
            let (kind, message) = if compile {
                ("program_error", "The plan's code does not compile.")
            } else {
                ("unknown_error", "The server could not start the plan.")
            };
            emit(
                state,
                &session_id,
                json!({"type": "workflow_run.error", "workflow_run_id": null,
                       "error": error_json(kind, message)}),
            );
            return Err(format!("the run was not started: {why}"));
        }
    };
    let id = new_id("wrun_");
    let name = input
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.trim().is_empty())
        .map(|n| clip(n, NAME_CHARS))
        .unwrap_or_else(|| format!("Workflow run {}", &id[5..13]));
    let description = input
        .get("description")
        .and_then(Value::as_str)
        .map(|d| clip(d, DESCRIPTION_CHARS));
    let run = Arc::new(Run {
        id: id.clone(),
        session: session_id.clone(),
        name: name.clone(),
        code: code.to_string(),
        stop: Notify::new(),
        stopping: StdMutex::new(None),
        threads: StdMutex::new(Vec::new()),
        task: StdMutex::new(None),
    });
    runs().insert(id.clone(), run.clone());
    record(state, &session_id, &id, |r| {
        *r = json!({"name": name, "open": true, "phases": {}});
    });
    emit(
        state,
        &session_id,
        json!({"type": "workflow_run.created", "workflow_run_id": id, "name": name,
               "description": description,
               "phases": phases.iter().map(Phase::json).collect::<Vec<_>>()}),
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(minutes * 60);
    let task = tokio::spawn(drive(
        state.clone(),
        session.clone(),
        run.clone(),
        plan,
        phases,
        deadline,
    ));
    *run.task.lock().unwrap_or_else(|e| e.into_inner()) = Some(task);
    Ok(format!(
        "Started workflow run {id} (\"{name}\"); it now runs in the background, so do not \
         start it again. When it ends you are sent a notice with what its plan printed. End \
         your turn now, unless you have other work to do meanwhile."
    ))
}

fn stop_run(session: &Value, input: &Map<String, Value>) -> Result<String, String> {
    let id = input
        .get("workflow_run_id")
        .and_then(Value::as_str)
        .ok_or("`workflow_run_id` is required")?;
    let run = runs()
        .get(id)
        .filter(|r| session["id"] == r.session.as_str())
        .cloned()
        .ok_or_else(|| format!("there is no open workflow run '{id}'"))?;
    end_early(&run, "stopped");
    Ok(format!(
        "Stopping workflow run {id} (\"{}\"): its agents stop, and you are sent its notice.",
        run.name
    ))
}

fn end_early(run: &Run, why: &'static str) {
    let mut stopping = run.stopping.lock().unwrap_or_else(|e| e.into_inner());
    if stopping.is_none_or(|w| w == "stopped" || why == "gone") {
        *stopping = Some(why);
    }
    run.stop.notify_one();
}

/// Ends session `session`'s open runs, and waits for them: as stopped
/// when it is archived (`announce`), silently when it is deleted.
pub(super) async fn end_all(state: &AppState, session: &str, announce: bool) {
    let _ = state;
    for run in open_runs(session) {
        end_early(&run, if announce { "archived" } else { "gone" });
        let task = run.task.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(task) = task {
            let _ = tokio::time::timeout(WIND_DOWN * 2, task).await;
        }
    }
}

/// How a run ended.
enum End {
    Completed {
        stdout: String,
        stderr: String,
    },
    Failed {
        kind: &'static str,
        message: &'static str,
        stdout: String,
        stderr: String,
    },
    /// Stopped, with a notice to the agent unless the session is archived.
    Stopped {
        notice: bool,
    },
    Gone,
}

/// One agent the plan calls: its configuration and thread name, or why it
/// cannot run.
fn agent_for(session: &Value, message: &Value, number: u64) -> Result<(Value, String), String> {
    let workflows = &session["_workflows"];
    let named = |key: &str| {
        message[key]
            .as_str()
            .filter(|v| !v.trim().is_empty())
            .map(str::to_string)
    };
    if let Some(wanted) = named("use") {
        let found = workflows["agents"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|a| {
                a["name"]
                    .as_str()
                    .is_some_and(|n| n.eq_ignore_ascii_case(&wanted))
                    || a["agent"]["id"] == wanted.as_str()
            })
            .ok_or_else(|| format!("no agent of this run is named '{wanted}'"))?;
        let config = found["agent"].clone();
        let name = named("name")
            .or_else(|| config["name"].as_str().map(str::to_string))
            .unwrap_or_else(|| format!("agent-{number}"));
        return Ok((config, name));
    }
    if workflows["inline"] != true {
        return Err("this run's agents are the listed ones: name one with `use=`".into());
    }
    let mut config = inline_agent(&session["agent"]);
    let name = named("name").unwrap_or_else(|| format!("agent-{number}"));
    config["name"] = json!(name);
    for key in ["description", "system"] {
        if let Some(value) = named(key) {
            config[key] = json!(value);
        }
    }
    Ok((config, name))
}

/// Runs one of the plan's agents in a thread of its own: its report.
async fn run_agent(
    state: Arc<AppState>,
    session: Value,
    run: Arc<Run>,
    config: Value,
    name: String,
    prompt: String,
) -> Report {
    let session_id = run.session.clone();
    let key = create_thread(&state, &session, config, &name, Some(&run.id))?;
    run.threads
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(key.clone());
    let (tx, rx) = oneshot::channel();
    WAITERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key.clone(), tx);
    // Its first input, on its own stream only.
    emit(
        &state,
        &key,
        json!({"type": "agent.thread_message_received",
               "from_session_thread_id": primary_thread(&session_id),
               "from_agent_name": session["agent"]["name"],
               "content": [{"type": "text", "text": prompt}]}),
    );
    {
        let live = live(&key);
        let _turn = live.lock.lock().await;
        let mut conversation = load_conversation(&state, &key);
        conversation
            .inbox
            .push(json!({"type": "text", "text": prompt}));
        save_conversation(&state, &key, &conversation);
    }
    kick(&state, &key);
    let report = rx.await.unwrap_or_else(|_| Err("the agent stopped".into()));
    retire(&state, &key).await;
    report
}

/// Archives one of a run's threads, stopping it first if it runs: a call
/// it still waits on is void.
async fn retire(state: &AppState, key: &str) {
    WAITERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(key);
    let Some(record) = load(state, key) else {
        return;
    };
    if !record["archived_at"].is_null() {
        return;
    }
    let live = live(key);
    if live.running.load(Ordering::SeqCst) {
        live.interrupted.store(true, Ordering::SeqCst);
        live.interrupt.notify_waiters();
        let since = Instant::now();
        while live.running.load(Ordering::SeqCst) && since.elapsed() < WIND_DOWN {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    {
        let _turn = live.lock.lock().await;
        let mut conversation = load_conversation(state, key);
        conversation.pending.clear();
        save_conversation(state, key, &conversation);
    }
    let now = rfc3339(unix_now());
    update(state, key, |r| {
        r["archived_at"] = json!(now);
        r["updated_at"] = json!(now);
    });
    let (session, Some(thread)) = split_key(key) else {
        return;
    };
    let event = json!({"type": "session.thread_status_terminated", "session_thread_id": thread,
                       "agent_name": thread_name(state, key)});
    emit(state, key, event.clone());
    emit(state, session, event);
}

/// Carries out a run's plan to its end, then ends the run.
async fn drive(
    state: Arc<AppState>,
    session: Value,
    run: Arc<Run>,
    mut plan: Plan,
    mut phases: Vec<Phase>,
    deadline: tokio::time::Instant,
) {
    let session_id = run.session.clone();
    emit(
        &state,
        &session_id,
        json!({"type": "workflow_run.status_running", "workflow_run_id": run.id}),
    );
    let working = Arc::new(Semaphore::new(MAX_WORKING));
    let mut made = 0;
    let mut agents: JoinSet<(u64, Report)> = JoinSet::new();
    // Open phases: the phase's id and the event that started it.
    let mut open: Vec<(String, String)> = Vec::new();
    let end = loop {
        tokio::select! {
            message = plan.next() => {
                let Some(message) = message else {
                    break End::Failed {
                        kind: "program_error", message: "The plan stopped without finishing.",
                        stdout: std::mem::take(&mut plan.stray), stderr: String::new(),
                    };
                };
                if let Some(number) = message["agent"].as_u64() {
                    if made >= MAX_THREADS {
                        break End::Failed {
                            kind: "thread_limit_error",
                            message: "The run went over its limit of threads.",
                            stdout: String::new(), stderr: String::new(),
                        };
                    }
                    match agent_for(&session, &message, number) {
                        Err(why) => plan.answer(number, Err(why)).await,
                        Ok((config, name)) => {
                            made += 1;
                            let prompt = message["prompt"].as_str().unwrap_or_default().to_string();
                            let (state, session, run) = (state.clone(), session.clone(), run.clone());
                            let working = working.clone();
                            agents.spawn(async move {
                                let _permit = working.acquire_owned().await;
                                (number, run_agent(state, session, run, config, name, prompt).await)
                            });
                        }
                    }
                } else if let Some(edge) = message["phase"].as_str() {
                    let named = message["name"].as_str().unwrap_or_default();
                    if edge == "start" {
                        let id = match phases.iter().find(|p| p.name == named || p.id == named) {
                            Some(phase) => phase.id.clone(),
                            None => {
                                let phase = Phase {
                                    id: new_id("wrph_"), name: clip(named, NAME_CHARS),
                                    description: None,
                                };
                                phases.push(phase.clone());
                                phase.id
                            }
                        };
                        let started = emit(&state, &session_id, json!({
                            "type": "workflow_run.phase_started", "workflow_run_id": run.id,
                            "workflow_run_phase_id": id,
                        }));
                        let started = started["id"].as_str().unwrap_or_default().to_string();
                        open.push((id, started));
                    } else if let Some(at) = open.iter().rposition(|(id, _)| {
                        phases.iter().any(|p| p.id == *id && (p.name == named || p.id == named))
                    }) {
                        let (id, started) = open.remove(at);
                        end_phase(&state, &run, &id, &started);
                    }
                    let held: Map<String, Value> =
                        open.iter().map(|(id, s)| (id.clone(), json!(s))).collect();
                    record(&state, &session_id, &run.id, |r| r["phases"] = json!(held));
                } else if message["done"] == true {
                    let text = |key: &str| message[key].as_str().unwrap_or_default().to_string();
                    let mut stdout = text("stdout");
                    stdout.push_str(&std::mem::take(&mut plan.stray));
                    break if message["return_code"].as_i64() == Some(0) {
                        End::Completed { stdout, stderr: text("stderr") }
                    } else {
                        End::Failed {
                            kind: "program_error", message: "The plan's code failed.",
                            stdout, stderr: text("stderr"),
                        }
                    };
                }
            }
            Some(joined) = agents.join_next(), if !agents.is_empty() => {
                if let Ok((number, report)) = joined {
                    plan.answer(number, report).await;
                }
            }
            _ = run.stop.notified() => {
                let why = *run.stopping.lock().unwrap_or_else(|e| e.into_inner());
                break match why {
                    Some("gone") => End::Gone,
                    why => End::Stopped { notice: why != Some("archived") },
                };
            }
            _ = tokio::time::sleep_until(deadline) => {
                break End::Failed {
                    kind: "timeout_error", message: "The run went past its lifetime.",
                    stdout: String::new(), stderr: String::new(),
                };
            }
        }
    };
    agents.abort_all();
    drop(plan);
    let threads = std::mem::take(&mut *run.threads.lock().unwrap_or_else(|e| e.into_inner()));
    let gone = matches!(end, End::Gone);
    if gone {
        for key in &threads {
            WAITERS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(key);
        }
        runs().remove(&run.id);
        return;
    }
    for key in &threads {
        retire(&state, key).await;
    }
    for (id, started) in open.drain(..) {
        end_phase(&state, &run, &id, &started);
    }
    let (result, notice) = match &end {
        End::Completed { stdout, stderr } => (
            json!({"type": "completed"}),
            format!("completed.{}", output(stdout, stderr)),
        ),
        End::Failed {
            kind,
            message,
            stdout,
            stderr,
        } => {
            let error = error_json(kind, message);
            emit(
                &state,
                &session_id,
                json!({"type": "workflow_run.error", "workflow_run_id": run.id, "error": error}),
            );
            (
                json!({"type": "error", "error": error}),
                format!("failed ({kind}): {message}{}", output(stdout, stderr)),
            )
        }
        End::Stopped { .. } | End::Gone => (
            json!({"type": "stopped"}),
            "stopped: its agents were stopped.".to_string(),
        ),
    };
    emit(
        &state,
        &session_id,
        json!({"type": "workflow_run.status_ended", "workflow_run_id": run.id,
               "result": result}),
    );
    record(&state, &session_id, &run.id, |r| {
        r["open"] = json!(false);
        r["phases"] = json!({});
    });
    runs().remove(&run.id);
    let archived = matches!(end, End::Stopped { notice: false })
        || load(&state, &session_id).is_none_or(|s| !s["archived_at"].is_null());
    if !archived {
        send_notice(&state, &session_id, &run.id, &run.name, &notice).await;
    }
}

fn end_phase(state: &AppState, run: &Run, id: &str, started: &str) {
    emit(
        state,
        &run.session,
        json!({"type": "workflow_run.phase_ended", "workflow_run_id": run.id,
               "workflow_run_phase_id": id, "phase_started_id": started}),
    );
}

/// What a plan printed, for its notice.
fn output(stdout: &str, stderr: &str) -> String {
    let mut text = String::new();
    let tail = |s: &str| {
        let chars = s.chars().count();
        if chars <= OUTPUT_CHARS {
            s.trim_end().to_string()
        } else {
            let kept: String = s.chars().skip(chars - OUTPUT_CHARS).collect();
            format!("[… the start is left out …]\n{}", kept.trim_end())
        }
    };
    if stdout.trim().is_empty() {
        text.push_str("\nThe plan printed nothing.");
    } else {
        text.push_str(&format!("\nWhat the plan printed:\n{}", tail(stdout)));
    }
    if !stderr.trim().is_empty() {
        text.push_str(&format!("\nIts error output:\n{}", tail(stderr)));
    }
    text
}

/// The notice of a run's end, to the session's agent, which takes a turn
/// on it.
async fn send_notice(state: &Arc<AppState>, session: &str, id: &str, name: &str, how: &str) {
    let text =
        format!("[Workflow run {id} (\"{name}\") {how}]\nGo on with your task from its result.");
    {
        let live = live(session);
        let _turn = live.lock.lock().await;
        let mut conversation = load_conversation(state, session);
        conversation
            .inbox
            .push(json!({"type": "text", "text": text}));
        save_conversation(state, session, &conversation);
    }
    kick(state, session);
}

/// Changes run `id`'s entry in its session's record of runs.
fn record(state: &AppState, session: &str, id: &str, change: impl FnOnce(&mut Value)) {
    static LOCK: StdMutex<()> = StdMutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = dir(state, session).join("workflow_runs.json");
    if !dir(state, session).is_dir() {
        return;
    }
    let mut all: Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({}));
    change(&mut all[id]);
    let _ = write_atomic(path, serde_json::to_vec(&all).unwrap_or_default());
}

/// Ends the runs a stopped server left open: the server cannot continue
/// them.
pub(super) async fn resume(state: Arc<AppState>) {
    for session in all_sessions(&state) {
        let session_id = session["id"].as_str().unwrap_or_default().to_string();
        let path = dir(&state, &session_id).join("workflow_runs.json");
        let Some(all) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        else {
            continue;
        };
        for (id, entry) in all.as_object().into_iter().flatten() {
            if entry["open"] != true {
                continue;
            }
            for thread in child_threads(&state, &session_id) {
                if thread["_thread"]["workflow_run_id"] == id.as_str()
                    && thread["archived_at"].is_null()
                {
                    retire(&state, thread["_key"].as_str().unwrap_or_default()).await;
                }
            }
            let run = Run {
                id: id.clone(),
                session: session_id.clone(),
                name: entry["name"].as_str().unwrap_or_default().to_string(),
                code: String::new(),
                stop: Notify::new(),
                stopping: StdMutex::new(None),
                threads: StdMutex::new(Vec::new()),
                task: StdMutex::new(None),
            };
            for (phase, started) in entry["phases"].as_object().into_iter().flatten() {
                end_phase(&state, &run, phase, started.as_str().unwrap_or_default());
            }
            let error = error_json("unknown_error", "The server could not continue the run.");
            emit(
                &state,
                &session_id,
                json!({"type": "workflow_run.error", "workflow_run_id": id, "error": error}),
            );
            emit(
                &state,
                &session_id,
                json!({"type": "workflow_run.status_ended", "workflow_run_id": id,
                       "result": {"type": "error", "error": error}}),
            );
            record(&state, &session_id, id, |r| {
                r["open"] = json!(false);
                r["phases"] = json!({});
            });
            if session["archived_at"].is_null() {
                send_notice(
                    &state,
                    &session_id,
                    id,
                    &run.name,
                    "failed (unknown_error): the server stopped while it ran.",
                )
                .await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the harness on `code`, answering each agent call with
    /// `answer(prompt)`: the harness messages it sent.
    fn drive_harness(code: &str, answer: impl Fn(&str) -> Report) -> Option<Vec<Value>> {
        use std::io::{BufRead as _, Write as _};
        let mut child = std::process::Command::new("python3")
            .args(["-I", "-c", &harness()])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take().unwrap();
        let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
        writeln!(stdin, "{}", json!({"code": code})).unwrap();
        let mut seen = Vec::new();
        while let Some(Ok(line)) = lines.next() {
            let Some(message) = line
                .strip_prefix(MARK)
                .and_then(|m| serde_json::from_str::<Value>(m).ok())
            else {
                continue;
            };
            seen.push(message.clone());
            if let Some(number) = message["agent"].as_u64() {
                let (ok, text) = match answer(message["prompt"].as_str().unwrap()) {
                    Ok(text) => (true, text),
                    Err(text) => (false, text),
                };
                writeln!(
                    stdin,
                    "{}",
                    json!({"agent": number, "ok": ok, "text": text})
                )
                .unwrap();
            }
            if message["done"] == true || message["compiled"] == false {
                break;
            }
        }
        let _ = child.wait();
        Some(seen)
    }

    /// A plan's agents run at once inside its phases, a failed one is an
    /// exception the plan can catch, and what it prints comes back.
    #[test]
    fn a_plan_runs_agents_in_phases() {
        let code = "import asyncio\n\
                    with phase('Read'):\n    \
                        results = await asyncio.gather(*(agent(f'read {i}') for i in range(3)), \
                    return_exceptions=True)\n\
                    with phase('Combine'):\n    \
                        ok = [r for r in results if not isinstance(r, Exception)]\n    \
                        failed = [str(r) for r in results if isinstance(r, AgentFailed)]\n\
                    print(sorted(ok), failed)";
        let Some(seen) = drive_harness(code, |prompt| match prompt {
            "read 1" => Err("it broke".into()),
            other => Ok(other.to_uppercase()),
        }) else {
            eprintln!("python3 is not installed; skipping");
            return;
        };
        assert_eq!(seen[0]["compiled"], true);
        let kinds: Vec<String> = seen
            .iter()
            .skip(1)
            .map(|m| {
                if m["agent"].is_u64() {
                    "agent".to_string()
                } else if let Some(edge) = m["phase"].as_str() {
                    format!("{edge} {}", m["name"].as_str().unwrap())
                } else {
                    "done".to_string()
                }
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "start Read",
                "agent",
                "agent",
                "agent",
                "end Read",
                "start Combine",
                "end Combine",
                "done"
            ]
        );
        let done = seen.last().unwrap();
        assert_eq!(done["return_code"], 0, "{done}");
        assert_eq!(done["stdout"], "['READ 0', 'READ 2'] ['it broke']\n");
    }

    /// Code that does not compile is refused before a run starts, and an
    /// agent failure the plan does not catch fails it.
    #[test]
    fn a_plan_that_breaks_fails() {
        let Some(seen) = drive_harness("with phase('x'):\nprint(1)", |_| Ok(String::new())) else {
            eprintln!("python3 is not installed; skipping");
            return;
        };
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0]["compiled"], false);
        assert!(
            seen[0]["error"]
                .as_str()
                .unwrap()
                .contains("IndentationError")
        );
        let seen = drive_harness("print(await agent('go'))", |_| Err("no".into())).unwrap();
        let done = seen.last().unwrap();
        assert_eq!(done["return_code"], 1);
        assert!(done["stderr"].as_str().unwrap().contains("AgentFailed: no"));
    }

    #[test]
    fn phases_take_names_and_clip_them() {
        let phases = phases_of(Some(&json!([
            "Read",
            {"name": "x".repeat(100), "description": "d"}
        ])))
        .unwrap();
        assert_eq!(phases[0].name, "Read");
        assert_eq!(phases[1].name.len(), NAME_CHARS);
        assert!(phases[0].id.starts_with("wrph_") && phases[0].id != phases[1].id);
        assert!(phases_of(Some(&json!([{"description": "no name"}]))).is_err());
        assert_eq!(
            phases_of(Some(&json!("[{\"name\": \"Read\"}]"))).unwrap()[0].name,
            "Read"
        );
        assert!(phases_of(Some(&json!("[{'name': 'Read'}]"))).is_err());
    }

    /// The tools say which agents a plan may run.
    #[test]
    fn the_start_tool_names_the_listed_agents() {
        let workflows = json!({"inline": false, "agents": [
            {"name": "reviewer", "description": "Reviews one contract",
             "agent": {"name": "reviewer"}},
        ]});
        let tools = definitions(&workflows);
        let text = tools[0]["description"].as_str().unwrap();
        assert!(text.contains("`reviewer` (Reviews one contract)"), "{text}");
        assert!(text.contains("Every agent of a run is one of these"));
        assert!(!text.contains("gets `system`"));
        assert_eq!(tools[1]["name"], STOP);
        let session = json!({"_workflows": workflows, "agent": {"name": "lead"}});
        let message = json!({"prompt": "p", "name": "own"});
        assert!(agent_for(&session, &message, 1).is_err());
        let message = json!({"prompt": "p", "use": "Reviewer"});
        let (_, name) = agent_for(&session, &message, 1).unwrap();
        assert_eq!(name, "reviewer");
    }
}
