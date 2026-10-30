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

//! The advisor: a second model request an agent consults mid-task. The
//! advisor reads the agent's conversation so far and answers with
//! guidance — a plan, a way past what is stuck, or a review before it
//! finishes. `/v1/messages` offers it as the `advisor_*` server tool, and a
//! session's primary thread as its agent's `advisor` member.

use axum::http::{HeaderMap, StatusCode};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use super::{MessagesRequest, error_of, run_one};
use crate::engine::chat_template::ChatMessage;
use crate::http::AppState;

/// The name an advisor's consultation threads carry, which no listed agent
/// may have while the advisor is enabled.
pub(super) const NAME: &str = "anthropic.advisor";

/// The name the advisor tool is called by, whose calls the advisor is not
/// shown: the one under way is the consultation itself.
pub(super) const TOOL: &str = "advisor";

/// What an advisor may write per consultation when the declaration sets no
/// `max_tokens`.
const DEFAULT_MAX_TOKENS: u64 = 2048;

/// How much of a conversation the advisor reads: the start (the task) and
/// the most recent part, about 12,000 tokens in all.
const TRANSCRIPT_HEAD: usize = 8_000;
const TRANSCRIPT_TAIL: usize = 40_000;

/// How often, and how far apart, a consultation is retried while the
/// server is too busy to take it.
const BUSY_TRIES: usize = 30;
const BUSY_RETRY: Duration = Duration::from_secs(2);

const SYSTEM: &str = "You advise another AI agent part-way through a task. You are shown its \
    conversation so far: its instructions, the task, what it has written, and the tools it called \
    with their results. Give it short, concrete, strategic guidance: the approach to take next, \
    what it is missing or getting wrong, how to get past what is stuck, or whether its work is \
    ready and what to check before it finishes. The agent has just consulted you: anything in the \
    conversation about calling or consulting an advisor refers to you, and is done. Do not do the \
    task yourself and do not call tools: answer with your advice only, addressed to the agent.";

/// What the agent is told when its advice was cut off at its length limit.
const CUT_OFF: &str = "\n\n[The advice was cut off at its length limit.]";

/// One consultation's answer.
pub(super) struct Advice {
    pub text: String,
    /// `end_turn`, or `max_tokens` when the advice was cut off.
    pub stop_reason: String,
    pub usage: Value,
}

impl Advice {
    /// The advice as the agent reads it.
    pub(super) fn for_agent(&self) -> String {
        if self.stop_reason == "max_tokens" {
            format!("{}{CUT_OFF}", self.text)
        } else {
            self.text.clone()
        }
    }
}

/// Asks `model` for advice on `transcript`: the advice, or the
/// `advisor_tool_result_error` code of why there is none.
pub(super) fn consult<'a>(
    state: &'a AppState,
    model: &'a str,
    transcript: String,
    max_tokens: Option<u64>,
) -> Pin<Box<dyn Future<Output = Result<Advice, &'static str>> + Send + 'a>> {
    Box::pin(async move {
        let request = json!({
            "model": model,
            "max_tokens": max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            "system": SYSTEM,
            "messages": [{"role": "user", "content": format!(
                "The agent's conversation so far:\n\n{transcript}\n\n## The agent consults you \
                 now\nIt has called you, its advisor, at this point. What is your advice on the \
                 task itself?"
            )}],
        });
        let headers = HeaderMap::new();
        for _ in 0..BUSY_TRIES {
            let req: MessagesRequest =
                serde_json::from_value(request.clone()).map_err(|_| "unavailable")?;
            match run_one(state, &headers, req).await {
                Ok(message) => {
                    let text: Vec<&str> = message["content"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|b| b["type"] == "text")
                        .filter_map(|b| b["text"].as_str())
                        .collect();
                    let text = text.join("\n").trim().to_string();
                    if text.is_empty() {
                        return Err("unavailable");
                    }
                    return Ok(Advice {
                        text,
                        stop_reason: message["stop_reason"]
                            .as_str()
                            .unwrap_or("end_turn")
                            .to_string(),
                        usage: message["usage"].clone(),
                    });
                }
                Err(response) if response.status() == StatusCode::SERVICE_UNAVAILABLE => {
                    tokio::time::sleep(BUSY_RETRY).await;
                }
                Err(response) => {
                    let error = error_of(*response).await;
                    let message = error["error"]["message"].as_str().unwrap_or_default();
                    return Err(if message.contains("context") {
                        "prompt_too_long"
                    } else {
                        "unavailable"
                    });
                }
            }
        }
        Err("overloaded")
    })
}

/// The `advisor_tool_result` block of a consultation, and what the agent
/// reads of it.
pub(super) fn result_block(tool_use_id: &str, advice: &Result<Advice, &str>) -> (Value, String) {
    match advice {
        Ok(advice) => (
            json!({"type": "advisor_tool_result", "tool_use_id": tool_use_id,
                   "content": {"type": "advisor_result", "text": advice.text,
                               "stop_reason": advice.stop_reason}}),
            advice.for_agent(),
        ),
        Err(code) => (
            json!({"type": "advisor_tool_result", "tool_use_id": tool_use_id,
                   "content": {"type": "advisor_tool_result_error", "error_code": code}}),
            failure_text(code),
        ),
    }
}

/// What the agent reads of a consultation that gave no advice.
pub(super) fn failure_text(code: &str) -> String {
    format!("The advisor could not be consulted ({code}). Carry on without its advice.")
}

/// The text an `advisor_tool_result` block stands for, sent back in a
/// later request.
pub(super) fn result_text(content: Option<&Value>) -> String {
    let field = |key: &str| {
        content
            .and_then(|c| c.get(key))
            .and_then(Value::as_str)
            .unwrap_or_default()
    };
    match content.and_then(|c| c.get("type")).and_then(Value::as_str) {
        Some("advisor_result") if field("stop_reason") == "max_tokens" => {
            format!("{}{CUT_OFF}", field("text"))
        }
        Some("advisor_result") => field("text").to_string(),
        Some("advisor_redacted_result") => "(The advisor's advice is not readable here.)".into(),
        _ => failure_text(field("error_code")),
    }
}

/// The advisor tool as the model is given it, in the OpenAI shape.
pub(super) fn definition(name: &str) -> Value {
    json!({"type": "function", "function": {
        "name": name,
        "description": "Consult your advisor: a second model that reads this whole conversation \
                        and gives strategic guidance. Use it before starting a hard task, when \
                        you are stuck, and to review your work before you finish. Takes no \
                        input; the advice comes back as the result.",
        "parameters": {"type": "object", "properties": {}},
    }})
}

/// `text` cut to what the advisor reads: its start and its end.
fn clip(text: String) -> String {
    let chars = text.chars().count();
    if chars <= TRANSCRIPT_HEAD + TRANSCRIPT_TAIL {
        return text;
    }
    let head: String = text.chars().take(TRANSCRIPT_HEAD).collect();
    let tail: String = text.chars().skip(chars - TRANSCRIPT_TAIL).collect();
    format!("{head}\n\n[… part of the conversation is left out …]\n\n{tail}")
}

/// A `/v1/messages` conversation as the advisor reads it: the chat so far,
/// and what the agent has written of its current turn.
pub(super) fn transcript_of_chat(chat: &[ChatMessage], current: &str) -> String {
    let mut out = String::new();
    for message in chat {
        let heading = match message.role.as_str() {
            "system" => "The agent's instructions".to_string(),
            "user" => "User".to_string(),
            "assistant" => "Agent".to_string(),
            "tool" => format!(
                "Result of {}",
                message.name.as_deref().unwrap_or("a tool call")
            ),
            other => other.to_string(),
        };
        out.push_str(&format!("## {heading}\n{}\n", message.content.trim()));
        for call in message
            .tool_calls
            .as_ref()
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|c| c.pointer("/function/name").and_then(Value::as_str) != Some(TOOL))
        {
            out.push_str(&format!(
                "[calls {} with {}]\n",
                call.pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or("a tool"),
                call.pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}"),
            ));
        }
        out.push('\n');
    }
    if !current.trim().is_empty() {
        out.push_str(&format!("## Agent (now)\n{}\n", current.trim()));
    }
    clip(out)
}

/// A session thread's conversation — Anthropic message objects — as the
/// advisor reads it.
pub(super) fn transcript_of_messages(system: &str, messages: &[Value]) -> String {
    let mut out = String::new();
    if !system.trim().is_empty() {
        out.push_str(&format!(
            "## The agent's instructions\n{}\n\n",
            system.trim()
        ));
    }
    for message in messages {
        let heading = if message["role"] == "assistant" {
            "Agent"
        } else {
            "User"
        };
        let blocks = match &message["content"] {
            Value::String(text) => vec![json!({"type": "text", "text": text})],
            Value::Array(blocks) => blocks.clone(),
            _ => Vec::new(),
        };
        let mut body = String::new();
        for block in &blocks {
            match block["type"].as_str() {
                Some("text") => {
                    body.push_str(block["text"].as_str().unwrap_or_default().trim());
                    body.push('\n');
                }
                Some("tool_use") if block["name"] == TOOL => {}
                Some("tool_use") => body.push_str(&format!(
                    "[calls {} with {}]\n",
                    block["name"].as_str().unwrap_or("a tool"),
                    block["input"]
                )),
                Some("tool_result") => {
                    let text = match &block["content"] {
                        Value::String(text) => text.clone(),
                        Value::Array(parts) => parts
                            .iter()
                            .filter_map(|p| p["text"].as_str())
                            .collect::<Vec<_>>()
                            .join("\n"),
                        _ => String::new(),
                    };
                    let error = if block["is_error"] == true {
                        " (an error)"
                    } else {
                        ""
                    };
                    body.push_str(&format!("[result{error}]\n{}\n", text.trim()));
                }
                _ => {}
            }
        }
        if !body.trim().is_empty() {
            out.push_str(&format!("## {heading}\n{body}\n"));
        }
    }
    clip(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_conversation_keeps_its_start_and_its_end() {
        let text = format!("TASK{}END", "x".repeat(TRANSCRIPT_HEAD + TRANSCRIPT_TAIL));
        let clipped = clip(text);
        assert!(clipped.starts_with("TASK"));
        assert!(clipped.ends_with("END"));
        assert!(clipped.contains("left out"));
    }

    #[test]
    fn advice_cut_off_says_so() {
        let advice = Advice {
            text: "Do A.".into(),
            stop_reason: "max_tokens".into(),
            usage: Value::Null,
        };
        assert!(advice.for_agent().ends_with(CUT_OFF));
        let (block, _) = result_block("srvtoolu_1", &Ok(advice));
        assert_eq!(block["content"]["type"], "advisor_result");
        assert!(result_text(Some(&block["content"])).ends_with(CUT_OFF));
        let (block, text) = result_block("srvtoolu_2", &Err("overloaded"));
        assert_eq!(block["content"]["error_code"], "overloaded");
        assert!(text.contains("overloaded"));
    }

    #[test]
    fn a_session_transcript_shows_calls_and_results() {
        let messages = vec![
            json!({"role": "user", "content": "Fix the build."}),
            json!({"role": "assistant", "content": [
                {"type": "text", "text": "Looking."},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "make"}},
            ]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": true,
                 "content": [{"type": "text", "text": "missing header"}]},
            ]}),
        ];
        let text = transcript_of_messages("Be careful.", &messages);
        assert!(text.contains("## The agent's instructions\nBe careful."));
        assert!(text.contains("[calls bash with {\"command\":\"make\"}]"));
        assert!(text.contains("[result (an error)]\nmissing header"));
    }
}
