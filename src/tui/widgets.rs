//! Pure glyph widgets: gauges, sparklines, bars, chips and the instrument panel frame.

use super::theme::{Theme, Token};
use ratatui::{
    layout::Alignment,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders},
};

const PARTIAL: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const ORBIT: [&str; 6] = ["◜", "◠", "◝", "◞", "◡", "◟"];

/// Display width of one char in terminal cells.
pub fn cell_width(c: char) -> usize {
    Span::raw(String::from(c)).width()
}

pub fn text_width(s: &str) -> usize {
    Span::raw(s).width()
}

/// Eighth-block gauge of exactly `cells` display cells on a `░` track.
pub fn gauge(frac: f32, cells: usize) -> String {
    let frac = if frac.is_finite() {
        frac.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eighths = (frac * cells as f32 * 8.0).round() as usize;
    let (full, rem) = (eighths / 8, eighths % 8);
    let mut s = "█".repeat(full.min(cells));
    if rem > 0 && full < cells {
        s.push(PARTIAL[rem]);
    }
    let used = text_width(&s);
    s.push_str(&"░".repeat(cells.saturating_sub(used)));
    s
}

/// Sparkline of exactly `cells` glyphs: values are resampled by bucket maximum; fewer values than
/// cells are right-aligned (newest on the right).
pub fn spark(values: &[f32], cells: usize) -> String {
    if values.is_empty() || cells == 0 {
        return " ".repeat(cells);
    }
    let n = values.len().min(cells);
    let buckets: Vec<f32> = (0..n)
        .map(|i| {
            let (lo, hi) = (
                i * values.len() / n,
                ((i + 1) * values.len() / n).max(i * values.len() / n + 1),
            );
            values[lo..hi.min(values.len())]
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max)
        })
        .collect();
    let max = buckets.iter().copied().fold(0.0f32, f32::max).max(1e-9);
    let mut s = " ".repeat(cells - n);
    s.extend(
        buckets
            .iter()
            .map(|v| SPARK[((v / max) * 7.0).round().clamp(0.0, 7.0) as usize]),
    );
    s
}

/// One coloured run per part, widths by largest remainder so the total is exactly `cells`.
pub fn stacked_bar(parts: &[(Token, f32)], cells: usize, theme: &Theme) -> Vec<Span<'static>> {
    let total: f32 = parts.iter().map(|p| p.1.max(0.0)).sum();
    if total <= 0.0 || cells == 0 {
        return vec![Span::styled("░".repeat(cells), theme.fg(Token::Hairline))];
    }
    let exact: Vec<f32> = parts
        .iter()
        .map(|p| p.1.max(0.0) / total * cells as f32)
        .collect();
    let mut w: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
    let mut order: Vec<usize> = (0..parts.len()).collect();
    order.sort_by(|&a, &b| (exact[b] - w[b] as f32).total_cmp(&(exact[a] - w[a] as f32)));
    let short = cells - w.iter().sum::<usize>();
    for &i in order.iter().take(short) {
        w[i] += 1;
    }
    parts
        .iter()
        .zip(w)
        .filter(|(_, n)| *n > 0)
        .map(|((t, _), n)| Span::styled("█".repeat(n), theme.fg(*t)))
        .collect()
}

/// `s` cut to at most `width` display cells, ending in `…` when cut.
pub fn trunc(s: &str, width: usize) -> String {
    if text_width(s) <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in s.chars() {
        let w = cell_width(c);
        if used + w + 1 > width {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');
    out
}

pub fn orbit(tick: usize) -> &'static str {
    ORBIT[tick % ORBIT.len()]
}

/// `⟦^R⟧ label`
pub fn chip(key: &str, label: &str, theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            format!("⟦{key}⟧"),
            theme.fg(Token::Brass).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {label}"), theme.fg(Token::Dim)),
    ]
}

/// Double-ruled panel `╔═╡ TITLE ╞════ metric ═╗`; BRASS border when focused.
pub fn instrument(
    title: &str,
    metric: Option<Line<'static>>,
    focused: bool,
    theme: &Theme,
) -> Block<'static> {
    let border = theme.fg(if focused {
        Token::Brass
    } else {
        Token::Hairline
    });
    let mut b = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(border)
        .style(if theme.truecolor {
            theme.bg(Token::Plinth)
        } else {
            Style::default()
        })
        .title(Line::from(vec![
            Span::styled("═╡ ", border),
            Span::styled(
                title.to_uppercase(),
                theme.fg(Token::Brass).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ╞", border),
        ]));
    if let Some(m) = metric {
        let mut spans = vec![Span::raw(" ")];
        spans.extend(m.spans.into_iter().map(|s| {
            if s.style == Style::default() {
                Span::styled(s.content, theme.fg(Token::Parchment))
            } else {
                s
            }
        }));
        spans.push(Span::styled(" ═", border));
        b = b.title(Line::from(spans).alignment(Alignment::Right));
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauge_cells_and_values() {
        assert_eq!(gauge(0.5625, 8), "████▌░░░");
        assert_eq!(gauge(0.0, 8), "░░░░░░░░");
        assert_eq!(gauge(1.0, 8), "████████");
        assert_eq!(gauge(7.0, 4), "████");
        assert_eq!(gauge(f32::NAN, 3), "░░░");
        for cells in 0..20 {
            for i in 0..=40 {
                assert_eq!(text_width(&gauge(i as f32 / 40.0, cells)), cells);
            }
        }
    }

    #[test]
    fn spark_resamples_and_pads() {
        assert_eq!(spark(&[0.0, 1.0], 2), "▁█");
        assert_eq!(spark(&[], 4), "    ");
        assert_eq!(spark(&[1.0], 3), "  █");
        assert_eq!(spark(&[0.0, 0.0, 1.0, 0.5], 2), "▁█");
        for cells in 0..12 {
            let v: Vec<f32> = (0..37).map(|i| i as f32).collect();
            assert_eq!(text_width(&spark(&v, cells)), cells);
        }
    }

    #[test]
    fn stacked_bar_total_width_is_exact() {
        let th = Theme {
            truecolor: true,
            mono: false,
        };
        for cells in 0..40 {
            let bar = stacked_bar(
                &[
                    (Token::Brass, 61.0),
                    (Token::Cyanotype, 23.0),
                    (Token::Jade, 16.0),
                ],
                cells,
                &th,
            );
            assert_eq!(bar.iter().map(|s| s.width()).sum::<usize>(), cells);
        }
        let empty = stacked_bar(&[], 5, &th);
        assert_eq!(empty.iter().map(|s| s.width()).sum::<usize>(), 5);
    }

    #[test]
    fn trunc_respects_width() {
        assert_eq!(trunc("abcdef", 6), "abcdef");
        assert_eq!(trunc("abcdef", 4), "abc…");
        assert_eq!(trunc("abc", 0), "");
        for w in 0..10 {
            assert!(text_width(&trunc("日本語のモデル名", w)) <= w);
        }
    }

    #[test]
    fn orbit_cycles() {
        assert_eq!(orbit(0), "◜");
        assert_eq!(orbit(6), "◜");
        assert_eq!(orbit(5), "◟");
    }

    #[test]
    fn chip_and_instrument_styles() {
        let th = Theme {
            truecolor: true,
            mono: false,
        };
        let c = chip("^R", "roofline", &th);
        assert_eq!(c[0].content, "⟦^R⟧");
        let area = ratatui::layout::Rect::new(0, 0, 30, 3);
        let metric = Some(Line::from("91.0%"));
        let render = |focused| {
            let mut buf = ratatui::buffer::Buffer::empty(area);
            ratatui::widgets::Widget::render(
                instrument("roofline", metric.clone(), focused, &th),
                area,
                &mut buf,
            );
            buf
        };
        let (idle, focused) = (render(false), render(true));
        let top: String = (0..30).map(|x| idle[(x, 0)].symbol()).collect();
        assert!(top.starts_with("╔═╡ ROOFLINE ╞"), "{top}");
        assert!(top.ends_with(" 91.0% ═╗"), "{top}");
        assert_eq!(idle[(0, 0)].fg, th.color(Token::Hairline));
        assert_eq!(focused[(0, 0)].fg, th.color(Token::Brass));
        assert_eq!(idle[(0, 1)].symbol(), "║");
    }
}
