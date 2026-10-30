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

//! Citations. A `document` with `citations.enabled` and every
//! `search_result` with it are cut into passages — a text or PDF document
//! into sentences, a `content` document and a search result into their
//! blocks — and each passage is shown to the model under a number, `[7]`.
//! The model is asked to write the numbers it draws on after its claims;
//! [`Citer`] takes them back out of the reply, each as the citation of the
//! text before it, located the way Anthropic locates one: by characters, by
//! page, or by block.

use serde_json::{Value, json};
use std::sync::Arc;

use super::{AnthropicMessage, Piece};

/// What the model is told when the conversation has numbered passages.
pub(super) const INSTRUCTION: &str = "Passages of the documents and search results in this \
conversation are numbered, like [7]. When a sentence of your answer uses what a numbered \
passage says, write the passage's number in brackets right after that claim, like [7], or \
[7][8] for several. Use only numbers shown in the documents.";

/// The longest number a marker carries.
const MAX_DIGITS: usize = 5;

/// Every numbered passage's citation, the number less one indexing it.
pub(super) struct Sources {
    passages: Vec<Value>,
}

impl Sources {
    fn passage(&self, number: usize) -> Option<&Value> {
        number.checked_sub(1).and_then(|n| self.passages.get(n))
    }
}

/// `text`'s sentences as byte ranges that together cover it: a sentence
/// ends after `.`, `!` or `?` (and any closing quote or bracket) followed by
/// whitespace, or at a blank line, and keeps the whitespace after it.
fn sentences(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut ended = false;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let next = chars.peek().map(|&(_, n)| n);
        if ended && !c.is_whitespace() {
            out.push((start, i));
            start = i;
            ended = false;
        }
        let terminal = matches!(c, '.' | '!' | '?')
            || (matches!(c, '"' | '\'' | ')' | ']' | '\u{201d}' | '\u{2019}')
                && text[..i].ends_with(['.', '!', '?']));
        if (terminal && next.is_none_or(char::is_whitespace)) || (c == '\n' && next == Some('\n')) {
            ended = true;
        }
    }
    if start < text.len() {
        out.push((start, text.len()));
    }
    out
}

/// A document's source as its text, and the text of each PDF page when it
/// is one.
fn document_source(source: &Value) -> Option<(String, Option<Vec<String>>)> {
    match source.get("type").and_then(Value::as_str)? {
        "text" => {
            let pages = source
                .get("pdf_pages")
                .and_then(Value::as_array)
                .map(|pages| {
                    pages
                        .iter()
                        .map(|p| p.as_str().unwrap_or_default().to_string())
                        .collect()
                });
            Some((source.get("data")?.as_str()?.to_string(), pages))
        }
        "base64" if source.get("media_type").and_then(Value::as_str) == Some("application/pdf") => {
            use base64::Engine as _;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(source.get("data")?.as_str()?)
                .ok()?;
            let pages = orangu::pdf::extract_pages(&bytes).ok()?;
            Some((pages.concat(), Some(pages)))
        }
        _ => None,
    }
}

/// The text blocks of a `content` array as (block index, text).
fn content_blocks(content: Option<&Value>) -> Vec<(usize, String)> {
    match content {
        Some(Value::String(text)) => vec![(0, text.clone())],
        Some(Value::Array(blocks)) => blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|(i, b)| Some((i, b.get("text")?.as_str()?.to_string())))
            .collect(),
        _ => Vec::new(),
    }
}

fn enabled(block: &Value) -> bool {
    block
        .pointer("/citations/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Numbers passages as they are met and writes the text the model sees.
#[derive(Default)]
struct Numbering {
    passages: Vec<Value>,
    documents: usize,
    results: usize,
}

impl Numbering {
    fn number(&mut self, citation: Value) -> String {
        self.passages.push(citation);
        format!("[{}] ", self.passages.len())
    }

    /// A cited text or PDF document's numbered text, or `None` when its
    /// source has no text to number.
    fn document(&mut self, block: &Value, index: usize) -> Option<String> {
        let title = block.get("title").cloned().unwrap_or(Value::Null);
        let source = block.get("source")?;
        let location = |kind: &str, cited: &str| {
            json!({"type": kind, "cited_text": cited, "document_index": index,
                   "document_title": title})
        };
        if source.get("type").and_then(Value::as_str) == Some("content") {
            let mut out = String::new();
            for (at, text) in content_blocks(source.get("content")) {
                let mut citation = location("content_block_location", &text);
                citation["start_block_index"] = json!(at);
                citation["end_block_index"] = json!(at + 1);
                out.push_str(&self.number(citation));
                out.push_str(&text);
                out.push('\n');
            }
            return Some(out);
        }
        let (text, pages) = document_source(source)?;
        let mut out = String::new();
        match pages {
            Some(pages) => {
                for (page, page_text) in pages.iter().enumerate() {
                    for (from, to) in sentences(page_text) {
                        let cited = &page_text[from..to];
                        if cited.trim().is_empty() {
                            out.push_str(cited);
                            continue;
                        }
                        let mut citation = location("page_location", cited);
                        citation["start_page_number"] = json!(page + 1);
                        citation["end_page_number"] = json!(page + 2);
                        out.push_str(&self.number(citation));
                        out.push_str(cited);
                    }
                }
            }
            None => {
                let mut chars = 0;
                for (from, to) in sentences(&text) {
                    let cited = &text[from..to];
                    let length = cited.chars().count();
                    if !cited.trim().is_empty() {
                        let mut citation = location("char_location", cited);
                        citation["start_char_index"] = json!(chars);
                        citation["end_char_index"] = json!(chars + length);
                        out.push_str(&self.number(citation));
                    }
                    out.push_str(cited);
                    chars += length;
                }
            }
        }
        Some(out)
    }

    /// A search result as the text the model sees: its title, source and
    /// content, each block numbered when its citations are on.
    fn search_result(&mut self, block: &Value) -> String {
        let index = self.results;
        self.results += 1;
        let title = block.get("title").cloned().unwrap_or(Value::Null);
        let source = block.get("source").cloned().unwrap_or(Value::Null);
        let mut out = format!(
            "Search result: {} ({})\n",
            title.as_str().unwrap_or_default(),
            source.as_str().unwrap_or_default()
        );
        let cited = enabled(block);
        for (at, text) in content_blocks(block.get("content")) {
            if cited {
                out.push_str(&self.number(json!({
                    "type": "search_result_location", "cited_text": text,
                    "source": source, "title": title, "search_result_index": index,
                    "start_block_index": at, "end_block_index": at + 1,
                })));
            }
            out.push_str(&text);
            out.push('\n');
        }
        out
    }

    fn visit(&mut self, blocks: &mut Value) {
        let Some(blocks) = blocks.as_array_mut() else {
            return;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_result")
                && let Some(content) = block.get_mut("content")
            {
                self.visit(content);
            }
            match block.get("type").and_then(Value::as_str) {
                Some("document") => {
                    let index = self.documents;
                    self.documents += 1;
                    if !enabled(block) {
                        continue;
                    }
                    if let Some(numbered) = self.document(block, index) {
                        let mut rewritten = json!({"type": "document",
                                                   "source": {"type": "text", "data": numbered}});
                        if let Some(title) = block.get("title") {
                            rewritten["title"] = title.clone();
                        }
                        *block = rewritten;
                    }
                }
                Some("search_result") => {
                    *block = json!({"type": "text", "text": self.search_result(block)});
                }
                _ => {}
            }
        }
    }
}

/// Numbers the passages of `messages`' cited documents and search results
/// in place, and turns every search result into the text the model reads.
/// `None` when nothing is cited.
pub(super) fn number(messages: &mut [AnthropicMessage]) -> Option<Arc<Sources>> {
    let mut numbering = Numbering::default();
    for message in messages {
        numbering.visit(&mut message.content);
    }
    (!numbering.passages.is_empty()).then(|| {
        Arc::new(Sources {
            passages: numbering.passages,
        })
    })
}

/// What a `[` starts in the reply.
enum Marker {
    /// `[n]`, this many bytes long.
    Whole(usize, usize),
    /// What has arrived could still become one.
    Partial,
    Not,
}

fn marker(text: &str) -> Marker {
    let digits = text[1..].bytes().take_while(u8::is_ascii_digit).count();
    let after = 1 + digits;
    if digits > MAX_DIGITS {
        return Marker::Not;
    }
    match text.as_bytes().get(after) {
        None => Marker::Partial,
        Some(b']') if digits > 0 => Marker::Whole(text[1..after].parse().unwrap_or(0), after + 1),
        _ => Marker::Not,
    }
}

/// Takes the passage numbers out of the reply's text as it arrives: the
/// text between them as [`Piece::Text`], each number as a
/// [`Piece::Citation`] of the text before it. A number that names no
/// passage stays text.
pub(super) struct Citer {
    sources: Arc<Sources>,
    held: String,
}

impl Citer {
    pub(super) fn new(sources: Arc<Sources>) -> Self {
        Self {
            sources,
            held: String::new(),
        }
    }

    fn send_text(text: &str, sink: &mut dyn FnMut(Piece)) {
        if !text.is_empty() {
            sink(Piece::Text(text.to_string()));
        }
    }

    pub(super) fn push(&mut self, text: &str, sink: &mut dyn FnMut(Piece)) {
        self.held.push_str(text);
        loop {
            let Some(at) = self.held.find('[') else {
                // Whitespace waits: it is dropped if a number follows.
                let keep = self.held.len() - self.held.trim_end().len();
                let ready: String = self.held.drain(..self.held.len() - keep).collect();
                Self::send_text(&ready, sink);
                return;
            };
            match marker(&self.held[at..]) {
                Marker::Whole(number, length) => {
                    let before: String = self.held.drain(..at).collect();
                    match self.sources.passage(number) {
                        Some(citation) => {
                            Self::send_text(before.trim_end(), sink);
                            sink(Piece::Citation(citation.clone()));
                            self.held.drain(..length);
                        }
                        None => {
                            let marker: String = self.held.drain(..length).collect();
                            Self::send_text(&(before + &marker), sink);
                        }
                    }
                }
                Marker::Partial => {
                    let before = &self.held[..at];
                    let ready = before.trim_end().len();
                    let ready: String = self.held.drain(..ready).collect();
                    Self::send_text(&ready, sink);
                    return;
                }
                Marker::Not => {
                    let ready: String = self.held.drain(..=at).collect();
                    Self::send_text(&ready, sink);
                }
            }
        }
    }

    /// Whatever is held, as text: the reply has moved on or ended.
    pub(super) fn flush(&mut self, sink: &mut dyn FnMut(Piece)) {
        let held = std::mem::take(&mut self.held);
        Self::send_text(&held, sink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joined(text: &str) -> Vec<&str> {
        sentences(text)
            .into_iter()
            .map(|(a, b)| &text[a..b])
            .collect()
    }

    #[test]
    fn text_is_cut_into_sentences_that_cover_it() {
        let text = "The grass is green. The sky is blue! Is it?\n\nNew \"para.\" e.g. this";
        let parts = joined(text);
        assert_eq!(
            parts,
            [
                "The grass is green. ",
                "The sky is blue! ",
                "Is it?\n\n",
                "New \"para.\" ",
                "e.g. ",
                "this"
            ]
        );
        assert_eq!(parts.concat(), text);
        assert_eq!(joined("3.14 is pi"), ["3.14 is pi"]);
    }

    fn messages(content: Value) -> Vec<AnthropicMessage> {
        vec![AnthropicMessage {
            role: "user".into(),
            content,
        }]
    }

    /// A cited text document is numbered by sentence and located by
    /// characters; a document without citations keeps its index.
    #[test]
    fn a_cited_document_is_numbered_by_sentence() {
        let mut m = messages(json!([
            {"type": "document", "source": {"type": "text", "data": "Plain."}},
            {"type": "document", "title": "Facts", "citations": {"enabled": true},
             "source": {"type": "text", "media_type": "text/plain",
                        "data": "The grass is green. The sky is blue."}},
            {"type": "text", "text": "Colours?"},
        ]));
        let sources = number(&mut m).unwrap();
        assert_eq!(
            m[0].content[1]["source"]["data"],
            "[1] The grass is green. [2] The sky is blue."
        );
        assert_eq!(m[0].content[1]["title"], "Facts");
        assert_eq!(m[0].content[0]["source"]["data"], "Plain.");
        let second = sources.passage(2).unwrap();
        assert_eq!(second["type"], "char_location");
        assert_eq!(second["cited_text"], "The sky is blue.");
        assert_eq!(second["document_index"], 1);
        assert_eq!(second["document_title"], "Facts");
        assert_eq!(
            (
                second["start_char_index"].as_u64(),
                second["end_char_index"].as_u64()
            ),
            (Some(20), Some(36))
        );
    }

    /// A PDF's sentences are located by page, a content document's and a
    /// search result's blocks by block.
    #[test]
    fn pages_and_blocks_locate_their_passages() {
        let mut m = messages(json!([
            {"type": "document", "citations": {"enabled": true},
             "source": {"type": "text", "data": "One. Two.", "pdf_pages": ["One. ", "Two."]}},
            {"type": "document", "citations": {"enabled": true},
             "source": {"type": "content", "content": [
                 {"type": "text", "text": "First block."}, {"type": "text", "text": "Second."}]}},
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "search_result", "source": "https://a.example", "title": "A",
                 "citations": {"enabled": true},
                 "content": [{"type": "text", "text": "Found it."}]}]},
        ]));
        let sources = number(&mut m).unwrap();
        let page = sources.passage(2).unwrap();
        assert_eq!(
            (page["type"].as_str(), page["start_page_number"].as_u64()),
            (Some("page_location"), Some(2))
        );
        let block = sources.passage(4).unwrap();
        assert_eq!(block["type"], "content_block_location");
        assert_eq!(block["document_index"], 1);
        assert_eq!(block["start_block_index"], 1);
        let found = sources.passage(5).unwrap();
        assert_eq!(found["type"], "search_result_location");
        assert_eq!(found["source"], "https://a.example");
        assert_eq!(found["search_result_index"], 0);
        assert_eq!(
            m[0].content[2]["content"][0]["text"],
            "Search result: A (https://a.example)\n[5] Found it.\n"
        );
    }

    #[derive(Debug, PartialEq)]
    enum Out {
        Text(String),
        Cite(String),
    }

    fn cite(pieces: &[&str]) -> Vec<Out> {
        let mut m = messages(json!([{"type": "document", "citations": {"enabled": true},
            "source": {"type": "text", "data": "Alpha. Beta."}}]));
        let mut citer = Citer::new(number(&mut m).unwrap());
        let mut out = Vec::new();
        let mut sink = |piece: Piece| match piece {
            Piece::Text(t) => match out.last_mut() {
                Some(Out::Text(last)) => last.push_str(&t),
                _ => out.push(Out::Text(t)),
            },
            Piece::Citation(c) => out.push(Out::Cite(c["cited_text"].as_str().unwrap().into())),
            _ => unreachable!(),
        };
        for piece in pieces {
            citer.push(piece, &mut sink);
        }
        citer.flush(&mut sink);
        out
    }

    /// Markers become citations however the reply is cut, the space before
    /// one goes, and brackets that are not a passage's number stay text.
    #[test]
    fn markers_become_citations_of_the_text_before_them() {
        let want = vec![
            Out::Text("It says alpha".into()),
            Out::Cite("Alpha. ".into()),
            Out::Cite("Beta.".into()),
            Out::Text(". See [9] and [x] and a[".into()),
        ];
        assert_eq!(
            cite(&["It says alpha [1][2]. See [9] and [x] and a["]),
            want
        );
        assert_eq!(
            cite(&[
                "It says alpha ",
                "[",
                "1",
                "]",
                "[2",
                "]",
                ". See [9",
                "] and [x] and a["
            ]),
            want
        );
    }
}
