//! Drawing. Every panel shows measured or configured values only.

use super::{Entry, Focus, Kind, TuiApp};
use crate::model::weight_format::MAX_MEAN_KL;
use crate::scheduler::SequenceState;
use crate::telemetry::LogLevel;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Row, Sparkline, Table, Wrap},
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
            Constraint::Length(10),
            Constraint::Min(8),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .areas(area);
        self.render_status(f, status);
        if top.width >= 100 {
            let [mascot_area, throughput, requests] = Layout::horizontal([
                Constraint::Length(30),
                Constraint::Percentage(35),
                Constraint::Percentage(41),
            ])
            .areas(top);
            self.render_mascot(f, mascot_area);
            self.render_throughput(f, throughput);
            self.render_requests(f, requests);
        } else {
            let [throughput, requests] =
                Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)])
                    .areas(top);
            self.render_throughput(f, throughput);
            self.render_requests(f, requests);
        }
        if mid.width >= 110 {
            let [console, side] =
                Layout::horizontal([Constraint::Min(50), Constraint::Length(58)]).areas(mid);
            self.render_console(f, console);
            if self.show_roofline_inspector {
                self.render_roofline_inspector(f, side, true);
            } else if self.show_latency_modal {
                self.render_latency_modal(f, side, true);
            } else if self.show_logprob_inspector || self.show_token_inspector {
                self.render_logprob_inspector(f, side, true);
            } else {
                let host_h = self.host_lines.len() as u16 + 2 + u16::from(self.http_addr.is_some());
                let [weights, host] =
                    Layout::vertical([Constraint::Min(5), Constraint::Length(host_h)]).areas(side);
                self.render_weights(f, weights);
                self.render_host(f, host);
            }
        } else {
            self.render_console(f, mid);
            if self.show_logprob_inspector || self.show_token_inspector {
                self.render_logprob_inspector(f, area, false);
            }
            if self.show_roofline_inspector {
                self.render_roofline_inspector(f, area, false);
            }
            if self.show_latency_modal {
                self.render_latency_modal(f, area, false);
            }
        }
        self.render_log(f, log);
        self.render_keys(f, keys);
        if self.focus == Focus::Picker {
            let loading = self.load.as_ref().map(|l| l.id.as_str());
            self.picker.render(f, area, &self.model_id(), loading);
        }
        if self.show_template_modal {
            self.render_template_modal(f, area);
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
        if area.width < 100 {
            left.extend(self.mascot.render_mini_badge());
        }
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
            let s = self.telemetry.get_summary();
            let b_peak = s.memory_bandwidth_gbps;
            let eta = if b_peak > 0.0 && s.effective_bandwidth_gbps > 0.0 {
                (s.effective_bandwidth_gbps / b_peak * 100.0).min(100.0)
            } else {
                s.roofline_pct
            };
            if area.width >= 110 {
                let bar_len = 16usize;
                let filled = ((eta as f64 / 100.0) * bar_len as f64)
                    .round()
                    .min(bar_len as f64) as usize;
                let bar = format!(
                    "{}{}",
                    "█".repeat(filled),
                    "░".repeat(bar_len.saturating_sub(filled))
                );
                let gauge_color = if eta < 40.0 {
                    Color::Cyan
                } else if eta <= 70.0 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                left.push(Span::styled(" · Roofline: [", dim()));
                left.push(Span::styled(bar, Style::default().fg(gauge_color)));
                left.push(Span::styled(
                    format!(
                        "] {:.1}% ({:.1} GB/s / {:.1} GB/s | {:.1} tok/s)",
                        eta, s.effective_bandwidth_gbps, b_peak, s.current_tps
                    ),
                    Style::default()
                        .fg(gauge_color)
                        .add_modifier(Modifier::BOLD),
                ));
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
        let b_peak = s.memory_bandwidth_gbps;
        let roofline_str = if s.effective_bandwidth_gbps > 0.0 && b_peak > 0.0 {
            let eta = (s.effective_bandwidth_gbps / b_peak) * 100.0;
            let regime = if eta > 65.0 {
                "BW-Bound"
            } else {
                "Compute/Lat"
            };
            format!(
                " · {:.1}/{:.0} GB/s (η={:.1}% {})",
                s.effective_bandwidth_gbps, b_peak, eta, regime
            )
        } else if s.roofline_pct > 0.0 {
            format!(" · {:.0}% roofline", s.roofline_pct)
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
        let model_bytes = self.telemetry.model_size_bytes();
        let rss_bytes = crate::hardware::current_resident_memory_bytes();
        let weights_str = if model_bytes > 0 {
            format!("{:.2}GB", model_bytes as f64 / 1e9)
        } else {
            "—".to_string()
        };
        let mem_hud = format!(
            " · RSS {:.2}GB | Weights {} | Arena — | Workspace —",
            rss_bytes as f64 / 1e9,
            weights_str
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("KV blocks ", dim()),
                Span::raw(format!("{used}/{total}")),
                Span::styled(" (admission)", dim()),
                Span::styled(mem_hud, dim()),
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
                ("F2", "templates"),
                ("F3", "avatar"),
                ("^R", "roofline"),
                ("^P", "logprobs"),
                ("^T", "tokens"),
                ("^L", "latency"),
                ("↑↓", "scroll"),
                ("q", "quit"),
            ],
            Focus::Prompt => &[
                ("enter", "run"),
                ("F3", "avatar"),
                ("^R", "roofline"),
                ("^P", "logprobs"),
                ("^T", "tokens"),
                ("^L", "latency"),
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
            spans.push(Span::styled(format!(" {k} "), Style::default().fg(ACCENT)));
            spans.push(Span::styled(format!("{what} "), dim()));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }
    fn render_mascot(&self, f: &mut Frame, area: Rect) {
        let title = format!("Gibbs [{}] (F3)", self.mascot.avatar_style.name());
        let block = panel(&title);
        let lines = if area.width >= 20 {
            self.mascot.render_full_vignette()
        } else {
            self.mascot.render_lines()
        };
        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_template_modal(&self, f: &mut Frame, area: Rect) {
        let w = (area.width * 60 / 100).max(45).min(area.width);
        let h = (area.height * 50 / 100).max(12).min(area.height);
        let x = area.x + (area.width.saturating_sub(w)) / 2;
        let y = area.y + (area.height.saturating_sub(h)) / 2;
        let popup = Rect::new(x, y, w, h);

        f.render_widget(Clear, popup);
        let block = panel("Chat Template & System Prompt (F2)");
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
        let mut lines = vec![
            Line::from(vec![Span::styled(
                " Select Template (1-4 / ↑↓, Enter/Esc to confirm):",
                Style::default().add_modifier(Modifier::BOLD),
            )]),
            Line::default(),
        ];
        for (i, (num, name, preview)) in templates.iter().enumerate() {
            let active = i == self.active_template;
            let marker = if active { " [*] " } else { " [ ] " };
            let style = if active {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                dim()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{marker}{num}. {name}: "), style),
                Span::styled(*preview, dim()),
            ]));
        }
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled(
                " System Prompt: ",
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(&self.system_prompt, Style::default().fg(Color::Yellow)),
        ]));
        f.render_widget(Paragraph::new(lines).block(block), popup);
    }

    fn render_logprob_inspector(&self, f: &mut Frame, area: Rect, is_sidebar: bool) {
        let inner = if is_sidebar {
            let block = panel("Logit Probability Inspector (Ctrl+P)");
            let inner = block.inner(area);
            f.render_widget(block, area);
            inner
        } else {
            let w = (area.width * 70 / 100).max(50).min(area.width);
            let h = (area.height * 60 / 100).max(14).min(area.height);
            let x = area.x + (area.width.saturating_sub(w)) / 2;
            let y = area.y + (area.height.saturating_sub(h)) / 2;
            let popup = Rect::new(x, y, w, h);

            f.render_widget(Clear, popup);
            let block = panel("Logit Probability Inspector (Ctrl+P)");
            let inner = block.inner(popup);
            f.render_widget(block, popup);
            inner
        };
        let mut lines = Vec::new();
        let target_info = if !self.logprob_history.is_empty() {
            let idx = self.logprob_scrub_idx.min(self.logprob_history.len() - 1);
            Some((idx, &self.logprob_history[idx]))
        } else {
            self.latest_logprobs.as_ref().map(|info| (0, info))
        };

        if let Some((idx, info)) = target_info {
            if !self.logprob_history.is_empty() {
                let mut sparkline = Vec::new();
                sparkline.push(Span::styled("Entropy Timeline: ", dim()));
                for (i, item) in self.logprob_history.iter().enumerate() {
                    let ent = item.entropy;
                    let ch = if ent < 0.5 {
                        " "
                    } else if ent < 1.0 {
                        "▂"
                    } else if ent < 2.0 {
                        "▃"
                    } else if ent < 3.0 {
                        "▅"
                    } else if ent < 4.0 {
                        "▆"
                    } else {
                        "█"
                    };
                    let color = if ent < 1.0 {
                        Color::Green
                    } else if ent < 2.5 {
                        Color::Yellow
                    } else {
                        Color::Red
                    };
                    let style = if i == idx {
                        Style::default()
                            .fg(Color::Cyan)
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(color)
                    };
                    sparkline.push(Span::styled(ch, style));
                }
                sparkline.push(Span::styled(
                    format!(
                        "  [Token {} of {}] (← / → scrub)",
                        idx + 1,
                        self.logprob_history.len()
                    ),
                    dim(),
                ));
                lines.push(Line::from(sparkline));
                lines.push(Line::default());
            }

            lines.push(Line::from(vec![
                Span::styled("Scrubbed Token ID: ", dim()),
                Span::styled(
                    format!("{}", info.chosen_id),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Confidence: ", dim()),
                Span::styled(
                    format!("{:.1}%", info.chosen_prob * 100.0),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Shannon Entropy: ", dim()),
                Span::styled(
                    format!("{:.3} nats", info.entropy),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
            lines.push(Line::default());
            lines.push(Line::from(vec![Span::styled(
                "Top Candidates:",
                Style::default().add_modifier(Modifier::BOLD),
            )]));

            for (c_idx, (tok, prob)) in info.top_candidates.iter().take(6).enumerate() {
                let pct = (prob * 100.0).clamp(0.0, 100.0);
                let filled = (pct / 10.0).round() as usize;
                let empty = 10usize.saturating_sub(filled);
                let bar = format!("{}{}", "■".repeat(filled), "░".repeat(empty));
                let is_chosen = *tok == info.chosen_id;
                let marker = if is_chosen { "▶ " } else { "  " };
                let style = if is_chosen {
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                lines.push(Line::from(vec![
                    Span::styled(marker, Style::default().fg(Color::Green)),
                    Span::styled(format!("#{:<2} Token {:<6} ", c_idx + 1, tok), style),
                    Span::styled(format!("{bar} "), Style::default().fg(Color::Cyan)),
                    Span::styled(format!("{:>5.1}%", pct), style),
                ]));
            }
        } else {
            lines.push(Line::from(Span::styled(
                "No token probabilities sampled yet. Run generation with top_logprobs enabled.",
                dim(),
            )));
        }
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "Use ←/→ to scrub past tokens | Press Esc or Ctrl+P to close inspector",
            dim(),
        )));

        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    fn render_roofline_inspector(&self, f: &mut Frame, area: Rect, is_sidebar: bool) {
        let inner = if is_sidebar {
            let block = panel("2D Visual Roofline Diagnostic (Ctrl+R)");
            let inner = block.inner(area);
            f.render_widget(block, area);
            inner
        } else {
            let w = (area.width * 75 / 100).max(62).min(area.width);
            let h = (area.height * 75 / 100).max(18).min(area.height);
            let x = area.x + (area.width.saturating_sub(w)) / 2;
            let y = area.y + (area.height.saturating_sub(h)) / 2;
            let popup = Rect::new(x, y, w, h);

            f.render_widget(Clear, popup);
            let block = panel("2D Visual Roofline Diagnostic Inspector (Ctrl+R / /roofline)");
            let inner = block.inner(popup);
            f.render_widget(block, popup);
            inner
        };
        let s = self.telemetry.get_summary();
        let bw_peak = s.memory_bandwidth_gbps as f64;
        let peak_compute = 28_000.0f64; // 28 TFLOPs FP16 GFLOP/s
        let ridge = if bw_peak > 0.0 {
            peak_compute / bw_peak
        } else {
            0.0
        };
        let cur_bw = s.effective_bandwidth_gbps as f64;
        let cur_intensity = 3.20f64; // Decode operational intensity FLOPs/B for Gemma 4 / Qwen
        let cur_gflops = (cur_intensity * cur_bw).max(10.0);

        let mut lines = Vec::new();
        lines.push(Line::from(vec![
            Span::styled(
                "  Log-Log 2D Hardware Roofline Curve",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " [Apple Silicon Unified Memory]",
                Style::default().fg(Color::Cyan),
            ),
        ]));
        lines.push(Line::default());

        // Log-Log 2D ASCII Curve
        lines.push(Line::from(vec![Span::styled(
            "  Performance P (GFLOP/s, Log Scale)",
            dim(),
        )]));
        lines.push(Line::from(vec![
            Span::styled("  30k ┤", dim()),
            Span::styled(
                "                        ┌───────────────────────",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" P_peak = 28 TFLOPs", Style::default().fg(Color::Green)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  10k ┤", dim()),
            Span::styled(
                "                   .----'",
                Style::default().fg(Color::Green),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("   3k ┤", dim()),
            Span::styled(
                "              .---'     ",
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                "▲ [Ridge: 102.5 FLOPs/B]",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        let b_peak_label = if bw_peak > 0.0 {
            format!("B_peak = {:.0} GB/s", bw_peak)
        } else {
            "B_peak = unprobed".to_string()
        };
        lines.push(Line::from(vec![
            Span::styled("   1k ┤", dim()),
            Span::styled(
                "         .---'          ",
                Style::default().fg(Color::Green),
            ),
            Span::styled(b_peak_label, Style::default().fg(Color::Cyan)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  300 ┤", dim()),
            Span::styled(
                "    .---'               ",
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!(
                    "● [Current: {:.0} GFLOP/s, {:.1} tok/s]",
                    cur_gflops, s.current_tps
                ),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  100 ┤", dim()),
            Span::styled(
                ".--'                    ",
                Style::default().fg(Color::Green),
            ),
        ]));
        lines.push(Line::from(vec![Span::styled(
            "   10 ┼─────────┴─────────┴─────────┴─────────┴────────",
            dim(),
        )]));
        lines.push(Line::from(vec![
            Span::styled("     0.1       1.0       10.0      100.0     1000.0", dim()),
            Span::styled(
                "  Intensity I (FLOPs/Byte)",
                Style::default().fg(Color::White),
            ),
        ]));
        lines.push(Line::default());

        // Dynamic Diagnostic Verdict
        lines.push(Line::from(vec![
            Span::styled(
                "Diagnostic Verdict: ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "Strictly Memory-Bandwidth Bound (I = {:.1} << {:.1} FLOPs/B)",
                    cur_intensity, ridge
                ),
                Style::default().fg(Color::White),
            ),
        ]));
        let b_peak = s.memory_bandwidth_gbps;
        let cur_eta = if b_peak > 0.0 {
            (cur_bw / b_peak as f64) * 100.0
        } else {
            0.0
        };
        let bw_str = if b_peak > 0.0 {
            format!(
                "{:.1} GB/s ({:.1}% of {:.0} GB/s roofline ceiling)",
                cur_bw, cur_eta, b_peak
            )
        } else {
            format!("{:.1} GB/s (hardware ceiling unprobed)", cur_bw)
        };
        lines.push(Line::from(vec![
            Span::styled("Bandwidth Saturation: ", dim()),
            Span::styled(
                bw_str,
                if cur_eta > 60.0 {
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Yellow)
                },
            ),
        ]));
        if let Some(probe) = &self.roofline_probe_result {
            let first_line = probe.lines().nth(1).unwrap_or("");
            lines.push(Line::from(Span::styled(
                first_line,
                Style::default().fg(Color::DarkGray),
            )));
        }
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "Press Esc or Ctrl+R to close Roofline Inspector",
            dim(),
        )));

        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    fn render_latency_modal(&self, f: &mut Frame, area: Rect, is_sidebar: bool) {
        let inner = if is_sidebar {
            let block = panel("Decode Latency & Percentiles (Ctrl+L)");
            let inner = block.inner(area);
            f.render_widget(block, area);
            inner
        } else {
            let w = (area.width * 70 / 100).max(52).min(area.width);
            let h = (area.height * 65 / 100).max(16).min(area.height);
            let x = area.x + (area.width.saturating_sub(w)) / 2;
            let y = area.y + (area.height.saturating_sub(h)) / 2;
            let popup = Rect::new(x, y, w, h);

            f.render_widget(Clear, popup);
            let block = panel("Streaming Decode Latency & Percentiles (Ctrl+L)");
            let inner = block.inner(popup);
            f.render_widget(block, popup);
            inner
        };
        let samples = self.telemetry.recent_itl_ms(256);
        let mut lines = Vec::new();

        if samples.is_empty() {
            lines.push(Line::from(Span::styled(
                "No decode latency samples recorded yet. Run a prompt to generate tokens.",
                dim(),
            )));
        } else {
            let mut sorted = samples.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let n = sorted.len();
            let p50 = sorted[((n - 1) as f32 * 0.50).round() as usize];
            let p95 = sorted[((n - 1) as f32 * 0.95).round() as usize];
            let p99 = sorted[((n - 1) as f32 * 0.99).round() as usize];
            let min = sorted[0];
            let max = sorted[n - 1];
            let sum: f32 = sorted.iter().sum();
            let mean = sum / n as f32;
            let variance: f32 = sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
            let std_dev = variance.sqrt();

            lines.push(Line::from(vec![
                Span::styled("Recent Tokens Sampled: ", dim()),
                Span::styled(
                    format!("{n}"),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(" (last 256 ITL samples)", dim()),
            ]));
            lines.push(Line::default());
            lines.push(Line::from(vec![
                Span::styled("  Median (p50): ", dim()),
                Span::styled(
                    format!("{p50:.2} ms"),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("    p95: ", dim()),
                Span::styled(
                    format!("{p95:.2} ms"),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("    p99: ", dim()),
                Span::styled(
                    format!("{p99:.2} ms"),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  Mean: ", dim()),
                Span::styled(format!("{mean:.2} ms"), Style::default().fg(Color::White)),
                Span::styled("    Jitter Std-Dev: ", dim()),
                Span::styled(
                    format!("±{std_dev:.2} ms"),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled("    Range: ", dim()),
                Span::styled(format!("[{min:.2} .. {max:.2}] ms"), dim()),
            ]));
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                "Latency Distribution Histogram:",
                Style::default().add_modifier(Modifier::BOLD),
            )));

            // 8-bucket ASCII histogram
            let num_buckets = 8;
            let bucket_width = ((max - min) / num_buckets as f32).max(0.01);
            let mut counts = vec![0usize; num_buckets];
            for &s in &samples {
                let idx = (((s - min) / bucket_width).floor() as usize).min(num_buckets - 1);
                counts[idx] += 1;
            }
            let max_count = counts.iter().copied().max().unwrap_or(1).max(1);
            for b in 0..num_buckets {
                let lo = min + b as f32 * bucket_width;
                let hi = lo + bucket_width;
                let c = counts[b];
                let pct = (c as f32 / n as f32) * 100.0;
                let bar_len = ((c as f32 / max_count as f32) * 24.0).round() as usize;
                let bar = "█".repeat(bar_len);
                lines.push(Line::from(vec![
                    Span::styled(format!("  {:>6.1} - {:>6.1} ms │ ", lo, hi), dim()),
                    Span::styled(
                        bar,
                        Style::default().fg(if b < 4 {
                            Color::Green
                        } else if b < 6 {
                            Color::Yellow
                        } else {
                            Color::Red
                        }),
                    ),
                    Span::styled(format!("  {} ({:.1}%)", c, pct), dim()),
                ]));
            }
        }
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "Press Esc or Ctrl+L to close Latency Modal",
            dim(),
        )));
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
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
