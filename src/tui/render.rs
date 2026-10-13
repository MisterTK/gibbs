//! Drawing. Every panel shows measured or configured values only.

use super::layout::{self, Tier};
use super::theme::{theme, Theme, Token};
use super::widgets::{chip, instrument};
use super::{Entry, Focus, Kind, TuiApp};
use crate::scheduler::SequenceState;
use crate::telemetry::LogLevel;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Row, Table, Wrap},
    Frame,
};

/// Console text indent under the `HH:MM:SS ` timestamp.
const INDENT: &str = "         ";

/// `12345678` → `12,345,678`.
pub(super) fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub(super) fn secs(d: std::time::Duration) -> String {
    let s = d.as_secs_f64();
    if s < 1.0 {
        format!("{:.0} ms", s * 1e3)
    } else {
        format!("{s:.1} s")
    }
}

/// Splits `text` into lines of at most `width` chars, breaking after the last space that fits
/// (mid-word only when a word is longer than the line); newlines are kept.
// ponytail: counts chars, not display cells; wide CJK/emoji output overflows its line.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for line in text.split('\n') {
        let chars: Vec<char> = line.chars().collect();
        let mut start = 0;
        while chars.len() - start > width {
            let cut = match chars[start..start + width].iter().rposition(|&c| c == ' ') {
                Some(i) if i > 0 => i + 1,
                _ => width,
            };
            let piece: String = chars[start..start + cut].iter().collect();
            out.push(piece.trim_end().to_string());
            start += cut;
        }
        out.push(chars[start..].iter().collect());
    }
    out
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let (w, h) = (w.min(area.width), h.min(area.height));
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

impl TuiApp {
    pub(super) fn render(&self, f: &mut Frame) {
        let area = f.area();
        let th = theme();
        if th.truecolor {
            f.render_widget(Block::default().style(th.bg(Token::Void)), area);
        }
        let r = layout::regions(area);
        let v = self.vitals();
        self.kernels_visible.set(false);
        self.render_status(f, r.status, &v, &th);

        let compact = r.tier == Tier::Compact;
        let loading = self.load.is_some();
        let mut main = r.main;
        let mut show_requests = true;
        let mut hidden = 0;

        if compact {
            if let Some(dock) = r.dock {
                // The dock takes the rows its content needs; the rest returns to the main column.
                let need = self
                    .dock_lines(&v, &th, dock.width.saturating_sub(2) as usize)
                    .len() as u16
                    + 2;
                let used = need.min(dock.height);
                main.height += dock.height - used;
                let d = Rect::new(dock.x, dock.y + dock.height - used, dock.width, used);
                self.render_dock(f, d, &v, &th);
            }
            if let Some(c) = self.detail {
                let dh = (main.height * 2 / 3).min(24);
                if dh >= 6 {
                    let top = Rect::new(main.x, main.y, main.width, dh);
                    main = Rect::new(main.x, main.y + dh, main.width, main.height - dh);
                    self.render_comp(f, c, top, &v, &th);
                    show_requests = false;
                }
            }
        } else if let Some(side) = r.sidebar_area() {
            let cw = r.sidebar.first().map_or(0, |c| c.width);
            let specs = layout::default_specs(cw, area.height, loading, self.detail);
            let plan = layout::flow(&specs, side, r.sidebar.len() as u16);
            hidden = plan.hidden;
            for (c, rect) in plan.placed {
                self.render_comp(f, c, rect, &v, &th);
            }
        }
        self.render_keys(f, r.keys, &th, hidden);

        let log_h = if area.height >= 40 { 5 } else { 3 };
        let req_h = if !show_requests {
            0
        } else if loading && compact {
            7
        } else if area.height >= 40 {
            8
        } else {
            5
        };
        let [req, console, log] = Layout::vertical([
            Constraint::Length(req_h),
            Constraint::Min(3),
            Constraint::Length(log_h),
        ])
        .areas(main);
        match &self.load {
            Some(l) if compact && show_requests => self.render_load(f, req, l, &th),
            _ if show_requests => self.render_requests(f, req, &th),
            _ => {}
        }
        self.render_console(f, console, &th);
        self.render_log(f, log, &th);

        if self.focus == Focus::Picker {
            let loading = self.load.as_ref().map(|l| l.id.as_str());
            self.picker.render(f, area, &self.model_id(), loading);
        }
        if self.show_template_modal {
            self.render_template_modal(f, area, &th);
        }
        if self.show_settings_modal {
            self.render_settings_modal(f, area, &th);
        }
    }

    fn render_requests(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let dim = th.fg(Token::Dim);
        let Some(sched) = &self.scheduler else {
            f.render_widget(
                Paragraph::new(Span::styled("no scheduler", dim))
                    .block(instrument("Requests", None, false, th)),
                area,
            );
            return;
        };
        let v = sched.view();
        let active = v
            .seqs
            .iter()
            .filter(|s| matches!(s.state, SequenceState::Prefilling | SequenceState::Decoding))
            .count();
        let footer = format!(
            " waiting {} · prefill {} tok/step · step {} ",
            v.waiting,
            v.prefill_budget,
            if v.busy_step.is_zero() {
                "-".to_string()
            } else {
                secs(v.busy_step)
            }
        );
        let block = instrument(
            "Requests",
            Some(Line::from(format!("{active} active"))),
            false,
            th,
        )
        .title_bottom(Span::styled(footer, dim));
        if v.seqs.is_empty() {
            f.render_widget(Paragraph::new(Span::styled("idle", dim)).block(block), area);
            return;
        }
        let rows = v.seqs.iter().map(|s| {
            let (state, tok, prompt) = match s.state {
                SequenceState::Prefilling => (
                    "prefill",
                    Token::Gold,
                    format!("{}/{}", s.prefilled, s.prompt),
                ),
                SequenceState::Decoding => ("decode", Token::Ember, s.prompt.to_string()),
                SequenceState::Waiting => ("waiting", Token::Dim, s.prompt.to_string()),
                SequenceState::Finished => ("done", Token::Dim, s.prompt.to_string()),
            };
            let ink = th.fg(Token::Parchment);
            Row::new(vec![
                Span::styled(format!("#{}", s.id), ink),
                Span::styled(state, th.fg(tok)),
                Span::styled(prompt, ink),
                Span::styled(format!("{}/{}", s.generated, s.max_new), ink),
                Span::styled(secs(s.submitted.elapsed()), dim),
                Span::styled(
                    crate::server::core::request_label(s.id).unwrap_or_else(|| "tui".into()),
                    dim,
                ),
            ])
        });
        let header = Row::new(["id", "state", "prompt", "output", "age", "client"]).style(dim);
        f.render_widget(
            Table::new(
                rows,
                [
                    Constraint::Length(7),
                    Constraint::Length(8),
                    Constraint::Length(11),
                    Constraint::Length(11),
                    Constraint::Length(8),
                    Constraint::Min(10),
                ],
            )
            .header(header)
            .block(block),
            area,
        );
    }

    fn render_console(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let dim = th.fg(Token::Dim);
        let hint = match self.focus {
            Focus::Prompt => {
                " enter run · esc cancel chat / back · ctrl-x cancel · pgup/pgdn scroll "
            }
            _ => " /chat /image /decide <q> | a, b /embed /classify <t> [|| pair] /connect /agent ",
        };
        let block = instrument("Console", None, self.focus == Focus::Prompt, th)
            .title_bottom(Span::styled(hint, dim));
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 2 {
            return;
        }
        let [hist, input] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
        let width = hist.width as usize;
        let height = hist.height as usize;

        // Newest entries first, until the window (plus scroll) is covered.
        let mut blocks: Vec<Vec<Line>> = Vec::new();
        let mut total = 0;
        for e in self.history.iter().rev() {
            let lines = entry_lines(e, width, th);
            total += lines.len();
            blocks.push(lines);
            if total >= height + self.scroll {
                break;
            }
        }
        let lines: Vec<Line> = blocks.into_iter().rev().flatten().collect();
        let scroll = self.scroll.min(lines.len().saturating_sub(height));
        let end = lines.len() - scroll;
        let start = end.saturating_sub(height);
        let mut shown: Vec<Line> = lines[start..end].to_vec();
        if self.history.is_empty() {
            shown.push(Line::from(Span::styled(
                "Prompts and model loads appear here.",
                dim,
            )));
        }
        f.render_widget(Paragraph::new(shown), hist);

        let line = if self.focus == Focus::Prompt {
            let room = width.saturating_sub(3);
            let skip = self.input.chars().count().saturating_sub(room);
            let visible: String = self.input.chars().skip(skip).collect();
            Line::from(vec![
                Span::styled("› ", th.fg(Token::Brass)),
                Span::styled(visible, th.fg(Token::Parchment)),
                Span::styled("▏", th.fg(Token::Brass)),
            ])
        } else {
            Line::from(Span::styled("› press i to type a prompt", dim))
        };
        f.render_widget(Paragraph::new(line), input);
    }

    fn render_log(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let block = instrument("Log", None, false, th);
        let rows = block.inner(area).height as usize;
        let lines: Vec<Line> = self
            .telemetry
            .get_recent_logs(rows)
            .into_iter()
            .map(|e| {
                let level = match e.level {
                    LogLevel::Info => th.fg(Token::Jade),
                    LogLevel::Warn => th.fg(Token::Gold).add_modifier(Modifier::BOLD),
                    LogLevel::Debug => th.fg(Token::Dim),
                };
                Line::from(vec![
                    Span::styled(format!("{} ", e.timestamp), th.fg(Token::Dim)),
                    Span::styled(format!("{:<5} ", e.level.as_str()), level),
                    Span::styled(e.message, th.fg(Token::Parchment)),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_keys(&self, f: &mut Frame, area: Rect, th: &Theme, hidden: usize) {
        let keys: &[(&str, &str)] = match self.focus {
            Focus::Dashboard => &[
                ("i", "prompt"),
                ("m", "models"),
                ("F2", "templates"),
                ("^R", "roofline"),
                ("^P", "logprobs"),
                ("^T", "tokens"),
                ("^L", "latency"),
                ("^K", "kernels"),
                ("↑↓", "scroll"),
                ("q", "quit"),
            ],
            Focus::Prompt => &[
                ("enter", "run"),
                ("^R", "roofline"),
                ("^P", "logprobs"),
                ("^T", "tokens"),
                ("^L", "latency"),
                ("^K", "kernels"),
                ("esc/ctrl-x", "cancel chat"),
                ("ctrl-c", "quit"),
            ],
            Focus::Picker => &[
                ("enter", "load"),
                ("tab", "sort"),
                ("ctrl-d", "delete cached"),
                ("esc", "close"),
                ("ctrl-c", "quit"),
            ],
        };
        let mut spans = Vec::new();
        for (k, what) in keys {
            if !spans.is_empty() {
                spans.push(Span::raw("  "));
            }
            spans.extend(chip(k, what, th));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
        if hidden > 0 {
            let note = format!("+{hidden} hidden (resize)");
            let w = (note.chars().count() as u16).min(area.width);
            let right = Rect::new(area.x + area.width - w, area.y, w, area.height);
            f.render_widget(Paragraph::new(Span::styled(note, th.fg(Token::Dim))), right);
        }
    }

    fn render_template_modal(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let popup = centered(
            area,
            (area.width * 60 / 100).max(45),
            (area.height * 50 / 100).max(12),
        );
        f.render_widget(Clear, popup);
        let block = instrument("Chat Template & System Prompt (F2)", None, true, th);
        let templates = [
            (
                "0",
                "Model Default",
                "Declared in tokenizer_config.json or GGUF metadata",
            ),
            (
                "1",
                "ChatML",
                "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>",
            ),
            (
                "2",
                "Llama-3",
                "<|start_header_id|>system<|end_header_id|>\\n{system}<|eot_id|>",
            ),
            (
                "3",
                "Gemma",
                "<start_of_turn>user\\n{prompt}<end_of_turn>\\n<start_of_turn>model\\n",
            ),
            ("4", "Raw / Passthrough", "{prompt}"),
        ];
        let bold = th.fg(Token::Parchment).add_modifier(Modifier::BOLD);
        let mut lines = vec![
            Line::from(Span::styled(
                " Select Template (1-4 / ↑↓, Enter/Esc to confirm):",
                bold,
            )),
            Line::default(),
        ];
        for (i, (num, name, preview)) in templates.iter().enumerate() {
            let active = i == self.active_template;
            let marker = if active { " [*] " } else { " [ ] " };
            let style = if active {
                th.fg(Token::Brass).add_modifier(Modifier::BOLD)
            } else {
                th.fg(Token::Dim)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{marker}{num}. {name}: "), style),
                Span::styled(*preview, th.fg(Token::Dim)),
            ]));
        }
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled(" System Prompt: ", bold),
            Span::styled(&self.system_prompt, th.fg(Token::Gold)),
        ]));
        f.render_widget(Paragraph::new(lines).block(block), popup);
    }

    fn render_settings_modal(&self, f: &mut Frame, area: Rect, th: &Theme) {
        let popup = centered(
            area,
            (area.width * 60 / 100).max(50),
            (area.height * 50 / 100).max(14),
        );
        f.render_widget(Clear, popup);
        let block = instrument("Policy Settings (F4)", None, true, th);
        let inner = block.inner(popup);
        f.render_widget(block, popup);

        let (label, value, dim) = (
            th.fg(Token::Brass),
            th.fg(Token::Gold).add_modifier(Modifier::BOLD),
            th.fg(Token::Dim),
        );
        let hud = if self.show_roofline_hud {
            Span::styled("ENABLED", th.fg(Token::Jade))
        } else {
            Span::styled("DISABLED", th.fg(Token::Crimson))
        };
        let lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled(" [1] Re-encode Policy:   ", label),
                Span::styled(format!("{:?}", self.reencode_policy), value),
                Span::styled("  (1: toggle max-speed / low-bit / bit-exact)", dim),
            ]),
            Line::from(vec![
                Span::styled(" [2] Variant Policy:     ", label),
                Span::styled(format!("{:?}", self.variant_policy), value),
                Span::styled("  (2: toggle fastest / fastest-4bit / quality)", dim),
            ]),
            Line::from(vec![
                Span::styled(" [3] Roofline HUD:       ", label),
                hud,
                Span::styled("  (3: toggle live HUD overlay)", dim),
            ]),
            Line::from(vec![
                Span::styled(" [4] Context Limit:      ", label),
                Span::styled(
                    format!("{} tokens", self.ctx_limit),
                    th.fg(Token::Parchment),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                " Esc / F4: Close modal | Enter: Save settings",
                dim,
            )),
        ];
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }
}

/// One console entry as wrapped lines, followed by a blank separator line.
fn entry_lines(e: &Entry, width: usize, th: &Theme) -> Vec<Line<'static>> {
    let body_w = width.saturating_sub(INDENT.len()).max(1);
    let mut out = Vec::new();
    for (i, chunk) in wrap(&e.prompt, body_w).into_iter().enumerate() {
        out.push(Line::from(vec![
            if i == 0 {
                Span::styled(format!("{} ", e.time), th.fg(Token::Dim))
            } else {
                Span::raw(INDENT)
            },
            Span::styled(chunk, th.fg(Token::Brass).add_modifier(Modifier::BOLD)),
        ]));
    }
    let tok = match e.kind {
        Kind::Running | Kind::Text => Token::Parchment,
        Kind::Json => Token::Gold,
        Kind::Image => Token::Cyanotype,
        Kind::Error => Token::Crimson,
    };
    if !e.output.is_empty() {
        for chunk in wrap(&e.output, body_w) {
            out.push(Line::from(vec![
                Span::raw(INDENT),
                Span::styled(chunk, th.fg(tok)),
            ]));
        }
    }
    let meta = match e.took {
        None => Span::styled(
            format!("running {}…", secs(e.started.elapsed())),
            th.fg(Token::Gold),
        ),
        Some(t) => {
            let mut parts: Vec<String> = Vec::new();
            if !e.stats.is_empty() {
                parts.push(e.stats.clone());
            }
            if !t.is_zero() {
                parts.push(secs(t));
            }
            Span::styled(parts.join(" · "), th.fg(Token::Dim))
        }
    };
    if !meta.content.is_empty() {
        out.push(Line::from(vec![Span::raw(INDENT), meta]));
    }
    out.push(Line::default());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_and_thousands() {
        assert_eq!(wrap("abcdef\n\nxy", 4), ["abcd", "ef", "", "xy"]);
        assert_eq!(wrap("the quick brown", 10), ["the quick", "brown"]);
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(999), "999");
    }
}
