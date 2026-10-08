//! Tool calls in model output: the format the loaded chat template renders them in, and a
//! streaming parser that splits decoded text into plain text and complete calls.

use serde_json::{Map, Value};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const MISTRAL: &str = "[TOOL_CALLS]";
const PYTHON_TAG: &str = "<|python_tag|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// Hermes / Qwen2.5: `<tool_call>{"name": .., "arguments": {..}}</tool_call>`.
    Hermes,
    /// Qwen3-Coder / Qwen3.5: `<tool_call><function=NAME><parameter=P>V</parameter></function></tool_call>`.
    QwenXml,
    /// Llama 3.x: the whole reply is `{"name": .., "parameters": {..}}` (optionally after
    /// `<|python_tag|>`, several separated by `;`).
    Llama,
    /// Mistral: `[TOOL_CALLS][{"name": .., "arguments": {..}}]`, or v11+ `[TOOL_CALLS]NAME[ARGS]{..}`.
    Mistral,
}

impl ToolFormat {
    /// The tool-call format a chat template renders; `None` when it has no tool support.
    pub fn detect(template: &str) -> Option<Self> {
        if template.contains("<function=") {
            Some(Self::QwenXml)
        } else if template.contains(OPEN) {
            Some(Self::Hermes)
        } else if template.contains(MISTRAL) {
            Some(Self::Mistral)
        } else if template.contains("<|start_header_id|>") && template.contains("parameters") {
            Some(Self::Llama)
        } else {
            None
        }
    }

    /// Assistant-turn text that commits the model to a tool call (`name`: to that tool). Mistral
    /// names are checked after generation instead (the call syntax depends on the tokenizer version).
    pub fn forced_prefix(self, name: Option<&str>) -> String {
        let quoted = |n: &str| Value::String(n.to_string()).to_string();
        match (self, name) {
            (Self::Hermes | Self::QwenXml, None) => format!("{OPEN}\n"),
            (Self::Hermes, Some(n)) => {
                format!("{OPEN}\n{{\"name\": {}, \"arguments\": ", quoted(n))
            }
            (Self::QwenXml, Some(n)) => format!("{OPEN}\n<function={n}>\n"),
            (Self::Llama, None) => "{\"name\": \"".to_string(),
            (Self::Llama, Some(n)) => format!("{{\"name\": {}, \"parameters\": ", quoted(n)),
            (Self::Mistral, _) => MISTRAL.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Unique per call; each schema adds its own prefix (`call_`, `toolu_`, `fc_`).
    pub id: String,
    pub name: String,
    /// Arguments as JSON object text.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Text(String),
    Call(ToolCall),
}

/// Incremental splitter: text that cannot start a call streams out at once; a possible call
/// opener is held until it resolves. Malformed calls come back as the text the model wrote.
pub struct ToolParser {
    format: ToolFormat,
    /// The request's tools (OpenAI shape): QwenXml parameter values are typed by their schema.
    tools: Vec<Value>,
    buf: String,
    in_call: bool,
    /// Llama: a call must open the reply, so the decision falls at its first non-blank text.
    decided: bool,
}

impl ToolParser {
    pub fn new(format: ToolFormat, tools: &[Value]) -> Self {
        Self {
            format,
            tools: tools.to_vec(),
            buf: String::new(),
            in_call: false,
            decided: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                // Llama and Mistral calls run to the end of the reply.
                let Some(end) = self.buf.find(CLOSE).filter(|_| self.tag_delimited()) else {
                    break;
                };
                let body: String = self.buf.drain(..end + CLOSE.len()).collect();
                self.in_call = false;
                out.push(self.block(&body[..end], true));
                continue;
            }
            let open = match self.format {
                ToolFormat::Hermes | ToolFormat::QwenXml => OPEN,
                ToolFormat::Mistral => MISTRAL,
                ToolFormat::Llama => {
                    if !self.decided {
                        let t = self.buf.trim_start();
                        if t.is_empty() {
                            break;
                        }
                        self.decided = true;
                        self.in_call = t.starts_with('{') || t.starts_with(PYTHON_TAG);
                        if self.in_call {
                            break;
                        }
                    }
                    out.push(Piece::Text(std::mem::take(&mut self.buf)));
                    break;
                }
            };
            if let Some(i) = self.buf.find(open) {
                if i > 0 {
                    out.push(Piece::Text(self.buf[..i].to_string()));
                }
                self.buf.drain(..i + open.len());
                self.in_call = true;
                continue;
            }
            let emit = self.buf.len() - partial_suffix(&self.buf, open);
            if emit > 0 {
                out.push(Piece::Text(self.buf.drain(..emit).collect()));
            }
            break;
        }
        out.retain(|p| !matches!(p, Piece::Text(t) if t.is_empty()));
        out
    }

    /// End of the reply: resolves a held opener or an unterminated call.
    pub fn finish(&mut self) -> Vec<Piece> {
        let rest = std::mem::take(&mut self.buf);
        let piece = if !self.in_call {
            Piece::Text(rest)
        } else if self.tag_delimited() {
            // The model stopped before `</tool_call>`: a complete body still counts.
            self.block(&rest, false)
        } else {
            let calls = match self.format {
                ToolFormat::Llama => parse_llama(&rest, &self.tools),
                _ => parse_mistral(&rest, &self.tools),
            };
            self.in_call = false;
            return match calls {
                Some(calls) => calls.into_iter().map(Piece::Call).collect(),
                None if self.format == ToolFormat::Mistral => {
                    vec![Piece::Text(MISTRAL.to_string() + &rest)]
                }
                None => vec![Piece::Text(rest)],
            };
        };
        self.in_call = false;
        match piece {
            Piece::Text(t) if t.is_empty() => Vec::new(),
            p => vec![p],
        }
    }

    fn tag_delimited(&self) -> bool {
        matches!(self.format, ToolFormat::Hermes | ToolFormat::QwenXml)
    }

    /// One `<tool_call>` body as a call, or the original text when it does not parse.
    fn block(&self, body: &str, closed: bool) -> Piece {
        let call = match self.format {
            ToolFormat::QwenXml => self.parse_xml(body),
            _ => serde_json::from_str::<Value>(body.trim())
                .ok()
                .and_then(|v| object_call(&v)),
        };
        call.map(Piece::Call).unwrap_or_else(|| {
            Piece::Text(format!("{OPEN}{body}{}", if closed { CLOSE } else { "" }))
        })
    }

    fn parse_xml(&self, body: &str) -> Option<ToolCall> {
        let rest = body.trim().strip_prefix("<function=")?;
        let (name, mut rest) = rest.split_once('>')?;
        let name = name.trim();
        let mut args = Map::new();
        while let Some(i) = rest.find("<parameter=") {
            let (key, r) = rest[i + "<parameter=".len()..].split_once('>')?;
            let end = ["</parameter>", "<parameter=", "</function>"]
                .iter()
                .filter_map(|m| r.find(m))
                .min()
                .unwrap_or(r.len());
            let raw = &r[..end];
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            args.insert(key.trim().to_string(), self.typed(name, key.trim(), raw));
            rest = &r[end..];
        }
        (!name.is_empty()).then(|| call(name, Value::Object(args).to_string()))
    }

    /// Like vLLM's qwen3_coder parser: string (or unknown) parameters stay text; other schema
    /// types parse as JSON when they can.
    fn typed(&self, tool: &str, param: &str, raw: &str) -> Value {
        let ty = self
            .tools
            .iter()
            .find(|t| t["function"]["name"] == tool)
            .and_then(|t| t["function"]["parameters"]["properties"][param]["type"].as_str());
        match ty {
            Some(t) if t != "string" => {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
            }
            _ => Value::String(raw.to_string()),
        }
    }
}

/// Length of the longest proper prefix of `marker` that `text` ends with (held back: the next
/// chunk may complete it).
pub fn partial_suffix(text: &str, marker: &str) -> usize {
    (1..marker.len().min(text.len() + 1))
        .rev()
        .find(|&k| {
            text.is_char_boundary(text.len() - k) && marker.starts_with(&text[text.len() - k..])
        })
        .unwrap_or(0)
}

fn call(name: &str, arguments: String) -> ToolCall {
    ToolCall {
        id: super::core::new_id(),
        name: name.to_string(),
        arguments,
    }
}

/// `{"name": .., "arguments"|"parameters": {..} or "{..}"}` as a call.
fn object_call(v: &Value) -> Option<ToolCall> {
    let name = v["name"].as_str().filter(|n| !n.is_empty())?;
    let args = match v.get("arguments").or_else(|| v.get("parameters")) {
        None | Some(Value::Null) => "{}".to_string(),
        Some(o @ Value::Object(_)) => o.to_string(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(o @ Value::Object(_)) => o.to_string(),
            _ => return None,
        },
        Some(_) => return None,
    };
    Some(call(name, args))
}

/// Whether `name` is one of the request's tools: untagged formats only count calls to them.
fn known(tools: &[Value], name: &Value) -> bool {
    tools.iter().any(|t| t["function"]["name"] == *name)
}

/// Bare JSON has no delimiter vouching for it: an object counts as a call only when it names
/// one of the request's tools and carries `parameters`/`arguments`, so JSON answers stay text.
fn parse_llama(text: &str, tools: &[Value]) -> Option<Vec<ToolCall>> {
    let t = text.trim_start();
    let mut rest = t.strip_prefix(PYTHON_TAG).unwrap_or(t);
    let mut calls = Vec::new();
    loop {
        rest = rest.trim_start().trim_start_matches(';').trim_start();
        if rest.is_empty() {
            break;
        }
        let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let v = values.next()?.ok()?;
        if !known(tools, &v["name"])
            || (v.get("parameters").is_none() && v.get("arguments").is_none())
        {
            return None;
        }
        calls.push(object_call(&v)?);
        rest = &rest[values.byte_offset()..];
    }
    (!calls.is_empty()).then_some(calls)
}

fn parse_mistral(text: &str, tools: &[Value]) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    for seg in text.split(MISTRAL).map(str::trim).filter(|s| !s.is_empty()) {
        if seg.starts_with('[') {
            for v in serde_json::from_str::<Vec<Value>>(seg).ok()? {
                calls.push(object_call(&v).filter(|_| known(tools, &v["name"]))?);
            }
        } else {
            let (head, args) = seg.split_once("[ARGS]")?;
            let name = head.split("[CALL_ID]").next()?.trim();
            let args: Value = serde_json::from_str(args.trim()).ok()?;
            let v = serde_json::json!({ "name": name, "arguments": args });
            calls.push(object_call(&v).filter(|_| known(tools, &v["name"]))?);
        }
    }
    (!calls.is_empty()).then_some(calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Feeds `chunks` one by one, then finishes; merges adjacent text and drops call ids.
    fn run(format: ToolFormat, chunks: &[&str]) -> Vec<Piece> {
        let tools = [
            json!({"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}}}}}),
        ];
        let mut p = ToolParser::new(format, &tools);
        let mut out: Vec<Piece> = Vec::new();
        let mut pieces: Vec<Piece> = chunks.iter().flat_map(|c| p.push(c)).collect();
        pieces.extend(p.finish());
        for piece in pieces {
            match (out.last_mut(), piece) {
                (Some(Piece::Text(a)), Piece::Text(b)) => a.push_str(&b),
                (_, Piece::Call(c)) => out.push(Piece::Call(ToolCall {
                    id: String::new(),
                    ..c
                })),
                (_, t) => out.push(t),
            }
        }
        out
    }

    fn text(s: &str) -> Piece {
        Piece::Text(s.to_string())
    }

    fn weather(args: Value) -> Piece {
        Piece::Call(ToolCall {
            id: String::new(),
            name: "get_weather".into(),
            arguments: args.to_string(),
        })
    }

    /// Every split point of `s` into two chunks parses like the whole string.
    fn same_at_every_split(format: ToolFormat, s: &str) -> Vec<Piece> {
        let whole = run(format, &[s]);
        for i in (1..s.len()).filter(|&i| s.is_char_boundary(i)) {
            assert_eq!(
                run(format, &[&s[..i], &s[i..]]),
                whole,
                "split at {i}: {s:?}"
            );
        }
        whole
    }

    #[test]
    fn detects_format_from_template() {
        let f = ToolFormat::detect;
        assert_eq!(
            f("{{'<tool_call>\\n<function=' + name}}"),
            Some(ToolFormat::QwenXml)
        );
        assert_eq!(
            f("<tool_call>\\n{\"name\": ...</tool_call>"),
            Some(ToolFormat::Hermes)
        );
        assert_eq!(f("{{ '[TOOL_CALLS]' }}"), Some(ToolFormat::Mistral));
        assert_eq!(
            f("<|start_header_id|>{\"name\": function name, \"parameters\": ...}"),
            Some(ToolFormat::Llama)
        );
        assert_eq!(f("<|im_start|>{{ message.content }}"), None);
    }

    #[test]
    fn hermes_calls_with_surrounding_text_at_every_chunk_boundary() {
        let s = "Checking.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n<tool_call>{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"Oslo\\\"}\"}</tool_call> done";
        assert_eq!(
            same_at_every_split(ToolFormat::Hermes, s),
            vec![
                text("Checking."),
                weather(json!({"city": "Paris"})),
                text("\n"),
                weather(json!({"city": "Oslo"})),
                text(" done"),
            ]
        );
    }

    #[test]
    fn hermes_malformed_and_unterminated() {
        let bad = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": </tool_call>";
        assert_eq!(run(ToolFormat::Hermes, &[bad]), vec![text(bad)]);
        // EOS right after a complete body still yields the call.
        let eos = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}";
        assert_eq!(run(ToolFormat::Hermes, &[eos]), vec![weather(json!({}))]);
        // A held partial opener is plain text when the reply ends.
        assert_eq!(
            run(ToolFormat::Hermes, &["a <tool_"]),
            vec![text("a <tool_")]
        );
        assert_eq!(run(ToolFormat::Hermes, &["x < y"]), vec![text("x < y")]);
    }

    #[test]
    fn qwen_xml_types_parameters_by_schema() {
        let s = "Sure.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            same_at_every_split(ToolFormat::QwenXml, s),
            vec![
                text("Sure.\n"),
                weather(json!({"city": "New York", "days": 3}))
            ]
        );
        let no_fn = "<tool_call>\n<parameter=city>x</parameter></tool_call>";
        assert_eq!(run(ToolFormat::QwenXml, &[no_fn]), vec![text(no_fn)]);
    }

    #[test]
    fn llama_json_reply_and_plain_text() {
        let s = " {\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}; {\"name\": \"get_weather\", \"parameters\": {\"city\": \"Bern\"}}";
        assert_eq!(
            same_at_every_split(ToolFormat::Llama, s),
            vec![
                weather(json!({"city": "Rome"})),
                weather(json!({"city": "Bern"}))
            ]
        );
        let tagged = "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {}}";
        assert_eq!(run(ToolFormat::Llama, &[tagged]), vec![weather(json!({}))]);
        // JSON that is not a call, and calls not opening the reply, stay text.
        // A JSON answer with a "name" field is not a call to a tool named after its value.
        let bob = "{\"name\": \"Bob\", \"age\": 3}";
        assert_eq!(run(ToolFormat::Llama, &[bob]), vec![text(bob)]);
        let no_args = "{\"name\": \"get_weather\"}";
        assert_eq!(run(ToolFormat::Llama, &[no_args]), vec![text(no_args)]);
        let unknown = "[TOOL_CALLS][{\"name\": \"rm_rf\", \"arguments\": {}}]";
        assert_eq!(run(ToolFormat::Mistral, &[unknown]), vec![text(unknown)]);
        assert_eq!(
            run(ToolFormat::Llama, &["{\"a\": 1}"]),
            vec![text("{\"a\": 1}")]
        );
        let late = "Use {\"name\": \"get_weather\", \"parameters\": {}}";
        assert_eq!(run(ToolFormat::Llama, &[late]), vec![text(late)]);
    }

    #[test]
    fn mistral_array_and_v11_syntax() {
        let v3 = "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Lyon\"}}]";
        assert_eq!(
            same_at_every_split(ToolFormat::Mistral, v3),
            vec![weather(json!({"city": "Lyon"}))]
        );
        let v11 = "Let me check.[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Nice\"}[TOOL_CALLS]get_weather[CALL_ID]abc123def[ARGS]{}";
        assert_eq!(
            run(ToolFormat::Mistral, &[v11]),
            vec![
                text("Let me check."),
                weather(json!({"city": "Nice"})),
                weather(json!({}))
            ]
        );
        let bad = "[TOOL_CALLS][{\"name\": \"get_weather\"";
        assert_eq!(run(ToolFormat::Mistral, &[bad]), vec![text(bad)]);
    }

    #[test]
    fn forced_prefix_parses_with_the_generated_rest() {
        for (format, rest) in [
            (ToolFormat::Hermes, "{\"city\": \"Paris\"}}\n</tool_call>"),
            (
                ToolFormat::QwenXml,
                "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            ),
            (ToolFormat::Llama, "{\"city\": \"Paris\"}}"),
        ] {
            let prefix = format.forced_prefix(Some("get_weather"));
            assert_eq!(
                run(format, &[&prefix, rest]),
                vec![weather(json!({"city": "Paris"}))],
                "{format:?}"
            );
        }
    }

    #[test]
    fn partial_suffix_respects_char_boundaries() {
        assert_eq!(partial_suffix("abc<tool", OPEN), 5);
        assert_eq!(partial_suffix("abc", OPEN), 0);
        assert_eq!(partial_suffix("é<", "<x"), 1);
        assert_eq!(partial_suffix("<tool_call>", OPEN), 0); // whole markers are found, not held
    }
}
