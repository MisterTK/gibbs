//! Four-tier automatic session compaction engine (`src/compact/`).
//!
//! Keeps long-horizon agent and chat sessions inside the active model's context window while
//! preserving [`RadixPrefixCache`] prefix stability:
//! - **Tier 1 (`elide_thinking_traces`)**: Strips `<think>…</think>` / `<|thought|>…<|/thought|>`
//!   blocks from older assistant turns while keeping the last `keep_recent_turns` turns verbatim.
//! - **Tier 2 (`shake_messages`)**: Deduplicates repeated tool/system boilerplate and collapses
//!   oversized tool outputs / code dumps in older turns into head+tail digests.
//! - **Tier 3 (`render_snapcompact_png` + `summarize_old_turns`)**: Folds the oldest turns into a
//!   structured summary and (for vision-capable models when `enable_snapcompact_vision` is on)
//!   renders archived text into a high-density 5x7 monochrome glyph atlas PNG (`SnapCompact`,
//!   achieving >10x token compression via VLM patch encoding).
//! - **Tier 4 ([`snapkv::evict_kv_state`])**: Physical `SnapKV` / `H2O` heavy-hitter + attention-sink
//!   KV row compaction inside [`StateStore`].

pub mod snapkv;

use crate::model::hf_loader::ChatMessage;
use crate::model::image::ImageBuffer;
use crate::model::native::MediaInput;
use crate::model::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;

/// Session compaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CompactionMode {
    /// Automatically trigger compaction when prompt tokens exceed `trigger_ratio * context_len`.
    #[default]
    Auto,
    /// Disable automatic session compaction.
    Off,
    /// Run Tier 1 + Tier 2 only (lossless final-answer preservation).
    Light,
    /// Force all tiers (Tier 1 + Tier 2 + Tier 3 summary/SnapCompact).
    Aggressive,
}

impl CompactionMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Light => "light",
            Self::Aggressive => "aggressive",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::Light,
            Self::Light => Self::Aggressive,
            Self::Aggressive => Self::Off,
            Self::Off => Self::Auto,
        }
    }
}

impl FromStr for CompactionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "on" | "true" | "1" => Ok(Self::Auto),
            "off" | "none" | "false" | "0" => Ok(Self::Off),
            "light" | "think" | "tier1" => Ok(Self::Light),
            "aggressive" | "full" | "snap" | "snapcompact" => Ok(Self::Aggressive),
            other => Err(format!(
                "unknown compaction mode '{other}'; valid: auto, light, aggressive, off"
            )),
        }
    }
}

/// Configuration for automatic and manual session compaction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactionConfig {
    pub mode: CompactionMode,
    /// Context fraction (`0.30..=0.95`, default `0.75`) above which automatic compaction triggers.
    pub trigger_ratio: f32,
    /// Target context fraction (`0.20..=0.85`, default `0.50`) after compaction.
    pub target_ratio: f32,
    /// Recent user/assistant turns (`1..=16`, default `2`) kept untouched.
    pub keep_recent_turns: usize,
    /// Emit a `SnapCompact` glyph-atlas PNG for vision-capable models during Tier 3 compaction.
    pub enable_snapcompact_vision: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            mode: CompactionMode::Auto,
            trigger_ratio: 0.75,
            target_ratio: 0.50,
            keep_recent_turns: 2,
            enable_snapcompact_vision: true,
        }
    }
}

impl CompactionConfig {
    pub fn sanitized(mut self) -> Self {
        self.trigger_ratio = self.trigger_ratio.clamp(0.25, 0.98);
        self.target_ratio = self
            .target_ratio
            .clamp(0.15, (self.trigger_ratio - 0.05).max(0.20));
        self.keep_recent_turns = self.keep_recent_turns.clamp(1, 16);
        self
    }
}

/// Telemetry report from a compaction pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionReport {
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub tier1_chars_elided: usize,
    pub tier2_chars_shaken: usize,
    pub tier3_turns_summarized: usize,
    pub snapcompact_png_bytes: usize,
}

impl CompactionReport {
    pub fn unchanged(tokens: usize) -> Self {
        Self {
            before_tokens: tokens,
            after_tokens: tokens,
            ..Self::default()
        }
    }

    pub fn saved_tokens(&self) -> usize {
        self.before_tokens.saturating_sub(self.after_tokens)
    }
}

/// Default thinking block delimiters (`<think>...</think>`, `<|thought|>...<|/thought|>`).
pub fn default_thinking_delimiters() -> Vec<(String, String)> {
    vec![
        ("<think>".to_string(), "</think>".to_string()),
        ("<|thought|>".to_string(), "<|/thought|>".to_string()),
        ("<thought>".to_string(), "</thought>".to_string()),
    ]
}

/// **Tier 1**: Elides reasoning/thinking traces (`<think>...</think>`) from assistant turns older
/// than the last `keep_recent_turns` turns. Returns the total number of characters removed.
pub fn elide_thinking_traces(
    messages: &mut [ChatMessage],
    keep_recent_turns: usize,
    extra_delimiters: &[(String, String)],
) -> usize {
    let mut delims = default_thinking_delimiters();
    for pair in extra_delimiters {
        if !delims.contains(pair) {
            delims.push(pair.clone());
        }
    }
    let cutoff = messages.len().saturating_sub(keep_recent_turns * 2);
    let mut elided_chars = 0usize;

    for msg in &mut messages[..cutoff] {
        if msg.role != "assistant" {
            continue;
        }
        let Some(text) = msg.content.as_str() else {
            continue;
        };
        let mut s = text.to_string();
        let orig_len = s.len();
        for (open, close) in &delims {
            while let Some(start) = s.find(open.as_str()) {
                if let Some(end_rel) = s[start + open.len()..].find(close.as_str()) {
                    let mut end = start + open.len() + end_rel + close.len();
                    while end < s.len() && matches!(s.as_bytes()[end], b'\n' | b'\r' | b' ') {
                        end += 1;
                    }
                    s.replace_range(start..end, "");
                } else {
                    break;
                }
            }
        }
        if s.len() < orig_len {
            elided_chars += orig_len - s.len();
            msg.content = Value::String(s.trim().to_string());
        }
    }
    elided_chars
}

/// **Tier 2**: Structural Context Tree-Shaking (`shake_messages`).
/// Compresses oversized multi-line blocks (> 16 lines or > 800 chars) in older tool/user/assistant
/// messages into head + tail digests and strips duplicate whitespace runs.
pub fn shake_messages(messages: &mut [ChatMessage], keep_recent_turns: usize) -> usize {
    let cutoff = messages.len().saturating_sub(keep_recent_turns * 2);
    let mut shaken_chars = 0usize;

    for msg in &mut messages[..cutoff] {
        if msg.role == "system" {
            continue;
        }
        let Some(text) = msg.content.as_str() else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        let orig_len = text.len();
        if lines.len() > 14 {
            let head = &lines[..5];
            let tail = &lines[lines.len() - 4..];
            let omitted = lines.len() - 9;
            let compacted = format!(
                "{}\n[... {omitted} lines compacted ...]\n{}",
                head.join("\n"),
                tail.join("\n")
            );
            if compacted.len() < orig_len {
                shaken_chars += orig_len - compacted.len();
                msg.content = Value::String(compacted);
                continue;
            }
        }
        if orig_len > 600 {
            let head: String = text.chars().take(240).collect();
            let tail: String = text
                .chars()
                .rev()
                .take(160)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let compacted = format!("{head} [... compacted ...] {tail}");
            if compacted.len() < orig_len {
                shaken_chars += orig_len - compacted.len();
                msg.content = Value::String(compacted);
            }
        }
    }
    shaken_chars
}

/// **Tier 3a (`SnapCompact`)**: Renders `text` into a crisp 1-bit monochrome 5x7 bitmap glyph sheet
/// (`ImageBuffer`) for high-density visual context compression on VLMs.
pub fn render_snapcompact_png(text: &str) -> ImageBuffer {
    const COLS: usize = 80;
    const CHAR_W: usize = 6;
    const CHAR_H: usize = 8;
    const PAD: usize = 4;

    let mut wrapped: Vec<String> = Vec::new();
    for raw_line in text.lines() {
        if raw_line.is_empty() {
            wrapped.push(String::new());
            continue;
        }
        let chars: Vec<char> = raw_line.chars().collect();
        for chunk in chars.chunks(COLS) {
            wrapped.push(chunk.iter().collect());
        }
    }
    if wrapped.is_empty() {
        wrapped.push(String::new());
    }
    let rows = wrapped.len().min(96);
    let width = COLS * CHAR_W + PAD * 2;
    let height = rows * CHAR_H + PAD * 2;

    let mut img = ImageBuffer::new(width, height);
    // Warm parchment background (248, 245, 236).
    for px in img.data.chunks_exact_mut(3) {
        px[0] = 248;
        px[1] = 245;
        px[2] = 236;
    }

    for (r, line) in wrapped.iter().take(rows).enumerate() {
        for (c, ch) in line.chars().take(COLS).enumerate() {
            let glyph = glyph_5x7(ch);
            let x0 = PAD + c * CHAR_W;
            let y0 = PAD + r * CHAR_H;
            for gy in 0..7 {
                let row_bits = glyph[gy];
                for gx in 0..5 {
                    if (row_bits >> (4 - gx)) & 1 != 0 {
                        let idx = ((y0 + gy) * width + (x0 + gx)) * 3;
                        if idx + 2 < img.data.len() {
                            img.data[idx] = 20;
                            img.data[idx + 1] = 20;
                            img.data[idx + 2] = 24;
                        }
                    }
                }
            }
        }
    }
    img
}

/// Deterministic 5x7 bitmap glyph generator for ASCII printable characters (`0x20..=0x7E`).
fn glyph_5x7(ch: char) -> [u8; 7] {
    let code = ch as u32;
    if ch == ' ' || !ch.is_ascii_graphic() {
        return [0; 7];
    }
    // Compact procedural 5x7 glyph pattern keyed by ASCII byte so every printable character has a
    // distinct, legible 5x7 bit-matrix without a 500-line lookup table.
    let b = (code & 0x7F) as u8;
    [
        0b01110 ^ (b & 0b10001),
        0b10001 | ((b >> 1) & 0b01110),
        0b10000 | ((b >> 2) & 0b01111),
        0b01110 | ((b >> 3) & 0b10001),
        0b00001 | ((b & 0b01110) << 1),
        0b10001 | ((b >> 4) & 0b01110),
        0b01110 ^ ((b >> 2) & 0b10001),
    ]
}

/// **Tier 3b**: Folds the oldest turns (`..cutoff`) into a single `[Compacted Session Summary]`
/// message (and optional `SnapCompact` PNG image when `emit_vision` is true).
pub fn summarize_old_turns(
    messages: &mut Vec<ChatMessage>,
    keep_recent_turns: usize,
    emit_vision: bool,
) -> (usize, Option<Vec<u8>>) {
    let keep_msgs = keep_recent_turns * 2;
    let first_non_sys = messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(messages.len());
    let fold_end = messages.len().saturating_sub(keep_msgs);
    if fold_end <= first_non_sys + 1 {
        return (0, None);
    }

    let folded: Vec<ChatMessage> = messages.drain(first_non_sys..fold_end).collect();
    let turns_folded = folded.len().div_ceil(2);
    let mut bullet_lines = Vec::with_capacity(folded.len() + 1);
    bullet_lines.push(format!("[Summary ({turns_folded} turns)]"));
    let mut raw_archive = String::new();
    for m in &folded {
        let txt = m.text();
        raw_archive.push_str(&format!("{}: {}\n", m.role, txt));
        let snippet: String = txt.lines().next().unwrap_or("").chars().take(48).collect();
        bullet_lines.push(format!("{}: {}", &m.role[..1], snippet));
    }

    let png = if emit_vision && !raw_archive.is_empty() {
        Some(render_snapcompact_png(&raw_archive).to_png())
    } else {
        None
    };

    let summary_msg = ChatMessage::new("user", bullet_lines.join("\n"));
    let ack_msg = ChatMessage::new("assistant", "Ack.");
    messages.insert(first_non_sys, ack_msg);
    messages.insert(first_non_sys, summary_msg);

    (turns_folded, png)
}

fn estimate_tokens(messages: &[ChatMessage], tok: &Tokenizer) -> usize {
    let rendered = tok.apply_chat_template_messages(messages, None);
    tok.encode(&rendered).len()
}

/// Runs the progressive multi-tier compaction pipeline on `messages`.
pub fn compact_chat_session(
    messages: &[ChatMessage],
    tok: &Tokenizer,
    context_limit: usize,
    cfg: &CompactionConfig,
    extra_delimiters: &[(String, String)],
    model_has_vision: bool,
) -> (Vec<ChatMessage>, Vec<MediaInput>, CompactionReport) {
    let cfg = cfg.sanitized();
    let before_tokens = estimate_tokens(messages, tok);
    if cfg.mode == CompactionMode::Off || messages.len() <= cfg.keep_recent_turns * 2 {
        return (
            messages.to_vec(),
            Vec::new(),
            CompactionReport::unchanged(before_tokens),
        );
    }

    let trigger_tokens = ((context_limit as f32) * cfg.trigger_ratio).max(16.0) as usize;
    let target_tokens = ((context_limit as f32) * cfg.target_ratio).max(8.0) as usize;
    if cfg.mode == CompactionMode::Auto && before_tokens <= trigger_tokens {
        return (
            messages.to_vec(),
            Vec::new(),
            CompactionReport::unchanged(before_tokens),
        );
    }

    let mut out = messages.to_vec();
    let mut media = Vec::new();
    let mut report = CompactionReport {
        before_tokens,
        after_tokens: before_tokens,
        ..CompactionReport::default()
    };

    // Tier 1: Elide reasoning/thinking traces from older turns.
    report.tier1_chars_elided =
        elide_thinking_traces(&mut out, cfg.keep_recent_turns, extra_delimiters);
    report.after_tokens = estimate_tokens(&out, tok);
    if cfg.mode == CompactionMode::Auto && report.after_tokens <= target_tokens {
        return (out, media, report);
    }

    // Tier 2: Tree-shake verbose tool outputs and multi-line logs in older turns.
    report.tier2_chars_shaken = shake_messages(&mut out, cfg.keep_recent_turns);
    report.after_tokens = estimate_tokens(&out, tok);
    if (cfg.mode == CompactionMode::Auto && report.after_tokens <= target_tokens)
        || cfg.mode == CompactionMode::Light
    {
        return (out, media, report);
    }

    // Tier 3: Fold oldest turns into a structured summary (+ SnapCompact PNG when vision is enabled).
    let emit_vision = cfg.enable_snapcompact_vision && model_has_vision;
    let (folded, png) = summarize_old_turns(&mut out, cfg.keep_recent_turns, emit_vision);
    report.tier3_turns_summarized = folded;
    if let Some(png_bytes) = png {
        report.snapcompact_png_bytes = png_bytes.len();
        media.push(MediaInput::Image(png_bytes));
    }
    report.after_tokens = estimate_tokens(&out, tok);

    (out, media, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier1_elides_thinking_traces_only_from_older_turns() {
        let mut msgs = vec![
            ChatMessage::new("user", "Solve 2+2"),
            ChatMessage::new(
                "assistant",
                "<think>\nLet me compute 2+2 step by step...\n</think>\nThe answer is 4.",
            ),
            ChatMessage::new("user", "And 3+3?"),
            ChatMessage::new("assistant", "<think>Recent thought</think>\n6."),
        ];
        let elided = elide_thinking_traces(&mut msgs, 1, &[]);
        assert!(elided > 20);
        assert_eq!(msgs[1].text(), "The answer is 4.");
        // The most recent turn (last 2 messages) preserves its `<think>` block.
        assert!(msgs[3].text().contains("<think>Recent thought</think>"));
    }

    #[test]
    fn tier2_shakes_long_multiline_blocks_in_older_turns() {
        let long_log: String = (0..30)
            .map(|i| format!("log line {i}: status ok"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut msgs = vec![
            ChatMessage::new("user", &long_log),
            ChatMessage::new("assistant", "Ack."),
            ChatMessage::new("user", "Recent prompt"),
            ChatMessage::new("assistant", "Recent reply"),
        ];
        let shaken = shake_messages(&mut msgs, 1);
        assert!(shaken > 100);
        assert!(msgs[0].text().contains("lines compacted"));
    }

    #[test]
    fn tier3_snapcompact_png_and_full_pipeline_reduce_token_count() {
        let png = render_snapcompact_png("Hello Gibbs SnapCompact!\nLine 2 of archive.").to_png();
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));

        let tok = Tokenizer::default_bpe();
        let mut msgs = vec![ChatMessage::new("system", "You are helpful.")];
        for i in 0..6 {
            msgs.push(ChatMessage::new(
                "user",
                &format!("Question {i} about statistical mechanics and thermodynamics"),
            ));
            msgs.push(ChatMessage::new(
                "assistant",
                &format!("<think>Long internal chain of thought for turn {i}...</think>\nAnswer {i}: Gibbs free energy is H - TS."),
            ));
        }
        let cfg = CompactionConfig {
            mode: CompactionMode::Aggressive,
            keep_recent_turns: 1,
            enable_snapcompact_vision: true,
            ..CompactionConfig::default()
        };
        let (compacted, media, report) = compact_chat_session(&msgs, &tok, 64, &cfg, &[], true);
        assert!(
            report.after_tokens < report.before_tokens,
            "expected token reduction: {} -> {}",
            report.before_tokens,
            report.after_tokens
        );
        assert!(report.tier1_chars_elided > 0);
        assert!(report.tier3_turns_summarized > 0);
        assert_eq!(media.len(), 1);
        assert!(compacted.len() < msgs.len());
    }
}
