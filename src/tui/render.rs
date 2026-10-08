//! Drawing. Every panel shows measured or configured values only.

use super::{Entry, Focus, Kind, TuiApp};
use crate::model::weight_format::MAX_MEAN_KL;
use crate::scheduler::SequenceState;
use crate::telemetry::LogLevel;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Row, Sparkline, Table, Wrap},
    Frame,
};

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;
/// Console text indent under the `HH:MM:SS ` timestamp.
const INDENT: &str = "         ";

fn dim() -> Style {
    Style::default().fg(DIM)
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(dim())
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))
}

/// `12345678` → `12,345,678`.
fn thousands(n: u64) -> String {
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

fn secs(d: std::time::Duration) -> String {
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

impl TuiApp {
    pub(super) fn render(&self, f: &mut Frame) {
        let area = f.area();
        let [status, top, mid, log, keys] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(9),
            Constraint::Min(8),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .areas(area);
        self.render_status(f, status);
        let [throughput, requests] =
            Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(top);
        self.render_throughput(f, throughput);
        self.render_requests(f, requests);
        if mid.width >= 110 {
            let [console, side] =
                Layout::horizontal([Constraint::Min(50), Constraint::Length(56)]).areas(mid);
            self.render_console(f, console);
            let host_h = self.host_lines.len() as u16 + 2 + u16::from(self.http_addr.is_some());
            let [weights, host] =
                Layout::vertical([Constraint::Min(5), Constraint::Length(host_h)]).areas(side);
            self.render_weights(f, weights);
            self.render_host(f, host);
        } else {
            self.render_console(f, mid);
        }
        self.render_log(f, log);
        self.render_keys(f, keys);
        if self.focus == Focus::Picker {
            let loading = self.load.as_ref().map(|l| l.id.as_str());
            self.picker.render(f, area, &self.model_id(), loading);
        }
    }

    fn render_status(&self, f: &mut Frame, area: Rect) {
        let model = self.model_id();
        let mut left = vec![
            Span::styled(
                " gibbs ",
                Style::default()
                    .fg(Color::Black)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            if model.is_empty() {
                Span::styled("no model", dim())
            } else {
                Span::styled(model, Style::default().add_modifier(Modifier::BOLD))
            },
        ];
        // While loading, the right side needs the room; backend and format come back after.
        if self.load.is_none() {
            if let Some(c) = self.classifier.read().as_ref() {
                left.push(Span::styled(" · ", dim()));
                left.push(Span::raw(format!("text classifier: {}", c.summary())));
            } else {
                if let Some(s) = &self.scheduler {
                    left.push(Span::styled(" · ", dim()));
                    left.push(Span::raw(s.backend_label()));
                }
                if let Some(w) = &self.weights {
                    left.push(Span::styled(" · ", dim()));
                    left.push(Span::raw(format!(
                        "{:?} {:.1} bpw",
                        w.mode,
                        w.mode.effective_bits_per_weight()
                    )));
                }
            }
        }
        if let Some(addr) = self.http_addr {
            left.push(Span::styled(
                format!(" · http :{} · /connect agents", addr.port()),
                dim(),
            ));
        }

        let right = match &self.load {
            Some(l) => Span::styled(
                format!(
                    " · loading {} · {} · {} ",
                    l.id,
                    l.progress_text(),
                    secs(l.started.elapsed())
                ),
                Style::default().fg(Color::Yellow),
            ),
            None if self.running > 0 => Span::styled(
                format!(" · {} running ", self.running),
                Style::default().fg(Color::Yellow),
            ),
            None => Span::raw(""),
        };
        let [l, r] =
            Layout::horizontal([Constraint::Min(0), Constraint::Length(right.width() as u16)])
                .areas(area);
        f.render_widget(Paragraph::new(Line::from(left)), l);
        f.render_widget(Paragraph::new(Line::from(right)), r);
    }

    fn render_throughput(&self, f: &mut Frame, area: Rect) {
        let block = panel("Throughput");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let s = self.telemetry.get_summary();
        let spec = self.scheduler.as_ref().and_then(|sc| sc.view().spec);
        let [head, spark, lat, kv] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(2 + spec.is_some() as u16),
            Constraint::Length(1),
        ])
        .areas(inner);
        let peak = self.spark.iter().copied().max().unwrap_or(0);
        let roofline_str = if s.roofline_pct > 0.0 && s.effective_bandwidth_gbps > 0.0 {
            format!(
                " · {:.1} GB/s ({:.0}% roofline)",
                s.effective_bandwidth_gbps, s.roofline_pct
            )
        } else {
            String::new()
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("{:.1} tok/s", s.current_tps),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(
                        "  {} tokens · {} requests · peak {peak}/s{roofline_str}",
                        thousands(s.total_tokens),
                        thousands(s.completed_requests)
                    ),
                    dim(),
                ),
            ])),
            head,
        );
        // Newest samples on the right; only as many as fit.
        let n = (spark.width as usize).min(self.spark.len());
        f.render_widget(
            Sparkline::default()
                .data(&self.spark[self.spark.len() - n..])
                .style(Style::default().fg(Color::Green)),
            spark,
        );
        let ms = |v: f32| {
            if v == 0.0 {
                "-".to_string()
            } else if v < 10.0 {
                format!("{v:.1} ms")
            } else {
                format!("{v:.0} ms")
            }
        };
        f.render_widget(
            Paragraph::new(
                [
                    Line::from(vec![
                        Span::styled("TTFT ", dim()),
                        Span::raw(format!(
                            "p50 {} · p90 {}",
                            ms(s.ttft_p50_ms),
                            ms(s.ttft_p90_ms)
                        )),
                    ]),
                    Line::from(vec![
                        Span::styled("ITL  ", dim()),
                        Span::raw(format!(
                            "p50 {} · p90 {}",
                            ms(s.itl_p50_ms),
                            ms(s.itl_p90_ms)
                        )),
                    ]),
                ]
                .into_iter()
                // Speculative decoding: drafter, tuned k, acceptance and tokens per forward.
                .chain(spec.map(|v| {
                    Line::from(vec![
                        Span::styled("Spec ", dim()),
                        Span::raw(format!(
                            "{} {} · accept {:.0}% · {:.2} tok/fwd",
                            v.label,
                            match v.k {
                                None => "k -".to_string(),
                                Some(0) => "off (slower)".to_string(),
                                Some(k) => format!("k {k}"),
                            },
                            s.spec_accept_rate * 100.0,
                            s.spec_tokens_per_forward
                        )),
                    ])
                }))
                .collect::<Vec<_>>(),
            ),
            lat,
        );
        let total = self.kv_pool.total_blocks;
        let used = (self.kv_pool.utilization() * total as f32).round() as usize;
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("KV blocks ", dim()),
                Span::raw(format!("{used}/{total}")),
                Span::styled(" (admission)", dim()),
            ])),
            kv,
        );
    }

    fn render_requests(&self, f: &mut Frame, area: Rect) {
        let Some(sched) = &self.scheduler else {
            f.render_widget(
                Paragraph::new(Span::styled("no scheduler", dim())).block(panel("Requests")),
                area,
            );
            return;
        };
        let v = sched.view();
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
        let block = panel("Requests").title_bottom(Span::styled(footer, dim()));
        if v.seqs.is_empty() {
            f.render_widget(
                Paragraph::new(Span::styled("idle", dim())).block(block),
                area,
            );
            return;
        }
        let rows = v.seqs.iter().map(|s| {
            let (state, color, prompt) = match s.state {
                SequenceState::Prefilling => (
                    "prefill",
                    Color::Yellow,
                    format!("{}/{}", s.prefilled, s.prompt),
                ),
                SequenceState::Decoding => ("decode", Color::Green, s.prompt.to_string()),
                SequenceState::Waiting => ("waiting", DIM, s.prompt.to_string()),
                SequenceState::Finished => ("done", DIM, s.prompt.to_string()),
            };
            Row::new(vec![
                Span::raw(format!("#{}", s.id)),
                Span::styled(state, Style::default().fg(color)),
                Span::raw(prompt),
                Span::raw(format!("{}/{}", s.generated, s.max_new)),
                Span::styled(secs(s.submitted.elapsed()), dim()),
                Span::styled(
                    crate::server::core::request_label(s.id).unwrap_or_else(|| "tui".into()),
                    dim(),
                ),
            ])
        });
        let header = Row::new(["id", "state", "prompt", "output", "age", "client"]).style(dim());
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

    fn render_console(&self, f: &mut Frame, area: Rect) {
        let hint = match self.focus {
            Focus::Prompt => {
                " enter run · esc cancel chat / back · ctrl-x cancel · pgup/pgdn scroll "
            }
            _ => " /chat /image /decide <q> | a, b /embed /classify <t> [|| pair] /connect /agent ",
        };
        let block = panel("Console").title_bottom(Span::styled(hint, dim()));
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
            let lines = entry_lines(e, width);
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
                dim(),
            )));
        }
        f.render_widget(Paragraph::new(shown), hist);

        let line = if self.focus == Focus::Prompt {
            let room = width.saturating_sub(3);
            let skip = self.input.chars().count().saturating_sub(room);
            let visible: String = self.input.chars().skip(skip).collect();
            Line::from(vec![
                Span::styled("› ", Style::default().fg(ACCENT)),
                Span::raw(visible),
                Span::styled("▏", Style::default().fg(ACCENT)),
            ])
        } else {
            Line::from(Span::styled("› press i to type a prompt", dim()))
        };
        f.render_widget(Paragraph::new(line), input);
    }

    fn render_weights(&self, f: &mut Frame, area: Rect) {
        let block = panel("Weights");
        let Some(w) = &self.weights else {
            // Native models and classifiers keep their stored weights; only GraphLm loads choose.
            let serving = self.classifier.read().is_some()
                || self.scheduler.as_ref().is_some_and(|s| s.has_model());
            let text = if serving {
                "weights run as stored (no format pick)"
            } else {
                "no LLM loaded"
            };
            f.render_widget(Paragraph::new(Span::styled(text, dim())).block(block), area);
            return;
        };
        let mut lines = vec![Line::from(vec![
            Span::styled(
                format!("{:?} {:.2} bpw", w.mode, w.mode.effective_bits_per_weight()),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" · {}", w.origin()), dim()),
        ])];
        match w.trials.as_slice() {
            [] => {}
            [only] => lines.push(Line::from(Span::styled(only.verdict.clone(), dim()))),
            trials => {
                let base = trials[0].layer_gemv_us.max(1e-9);
                lines.push(Line::from(Span::styled(
                    format!(
                        "  {:<15}{:>8}{:>8}{:>11}{:>9}",
                        "format", "GEMV µs", "vs f32", "probe/tok", "KL"
                    ),
                    dim(),
                )));
                for t in trials {
                    let chosen = t.mode == w.mode;
                    let style = if chosen {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default()
                    };
                    let probe = t
                        .ms_per_token
                        .map_or("-".to_string(), |ms| format!("{ms:.2} ms"));
                    let kl = t.mean_kl.map_or("-".to_string(), |k| {
                        if k < 10.0 {
                            format!("{k:.4}")
                        } else {
                            format!("{k:.1}")
                        }
                    });
                    let kl_style = match t.mean_kl {
                        Some(k) if !(k <= MAX_MEAN_KL) => Style::default().fg(Color::Red),
                        _ => style,
                    };
                    lines.push(Line::from(vec![
                        Span::styled(if chosen { "● " } else { "  " }, style),
                        Span::styled(
                            format!(
                                "{:<15}{:>8.0}{:>7.2}x{:>11}",
                                format!("{:?}", t.mode),
                                t.layer_gemv_us,
                                t.layer_gemv_us / base,
                                probe
                            ),
                            style,
                        ),
                        Span::styled(format!("{kl:>9}"), kl_style),
                    ]));
                }
            }
        }
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_host(&self, f: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = self
            .host_lines
            .iter()
            .map(|l| Line::from(l.as_str()))
            .collect();
        if let Some(addr) = self.http_addr {
            lines.push(Line::from(format!("HTTP:   http://{addr}")));
        }
        f.render_widget(
            Paragraph::new(lines)
                .block(panel("Host"))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn render_log(&self, f: &mut Frame, area: Rect) {
        let block = panel("Log");
        let rows = block.inner(area).height as usize;
        let lines: Vec<Line> = self
            .telemetry
            .get_recent_logs(rows)
            .into_iter()
            .map(|e| {
                let level = match e.level {
                    LogLevel::Info => Style::default().fg(Color::Green),
                    LogLevel::Warn => Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                    LogLevel::Debug => dim(),
                };
                Line::from(vec![
                    Span::styled(format!("{} ", e.timestamp), dim()),
                    Span::styled(format!("{:<5} ", e.level.as_str()), level),
                    Span::raw(e.message),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_keys(&self, f: &mut Frame, area: Rect) {
        let keys: &[(&str, &str)] = match self.focus {
            Focus::Dashboard => &[
                ("i", "prompt"),
                ("m", "models"),
                ("↑↓ pgup pgdn", "scroll"),
                ("q", "quit"),
            ],
            Focus::Prompt => &[
                ("enter", "run"),
                ("esc/ctrl-x", "cancel chat"),
                ("esc", "back"),
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
            spans.push(Span::styled(format!(" {k} "), Style::default().fg(ACCENT)));
            spans.push(Span::styled(format!("{what} "), dim()));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }
}

/// One console entry as wrapped lines, followed by a blank separator line.
fn entry_lines(e: &Entry, width: usize) -> Vec<Line<'static>> {
    let body_w = width.saturating_sub(INDENT.len()).max(1);
    let mut out = Vec::new();
    for (i, chunk) in wrap(&e.prompt, body_w).into_iter().enumerate() {
        out.push(Line::from(vec![
            if i == 0 {
                Span::styled(format!("{} ", e.time), dim())
            } else {
                Span::raw(INDENT)
            },
            Span::styled(
                chunk,
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
        ]));
    }
    let color = match e.kind {
        Kind::Running | Kind::Text => Color::Reset,
        Kind::Json => Color::Yellow,
        Kind::Image => Color::Magenta,
        Kind::Error => Color::Red,
    };
    if !e.output.is_empty() {
        for chunk in wrap(&e.output, body_w) {
            out.push(Line::from(vec![
                Span::raw(INDENT),
                Span::styled(chunk, Style::default().fg(color)),
            ]));
        }
    }
    let meta = match e.took {
        None => Span::styled(
            format!("running {}…", secs(e.started.elapsed())),
            Style::default().fg(Color::Yellow),
        ),
        Some(t) => {
            let mut parts: Vec<String> = Vec::new();
            if !e.stats.is_empty() {
                parts.push(e.stats.clone());
            }
            if !t.is_zero() {
                parts.push(secs(t));
            }
            Span::styled(parts.join(" · "), dim())
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
