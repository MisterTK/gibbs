//! J. Willard Gibbs cameo: the c.1895 photograph, ordered-dithered by `examples/gibbs_art.rs`
//! into `assets/art/gibbs-sprite.txt` (48×60, levels 0–7 plus pipe `p` / ember `e` overlay and
//! six smoke frames), drawn as half-block cells.

use super::theme::{rgb_to_ansi256, theme, Theme, Token};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::sync::LazyLock;

const SPRITE: &str = include_str!("../../assets/art/gibbs-sprite.txt");
const ASCII: &str = include_str!("../../assets/art/gibbs-ascii.txt");

const SHADOW: (u8, u8, u8) = (0x1A, 0x13, 0x0C);
const HIGHLIGHT: (u8, u8, u8) = (0xF5, 0xEE, 0xDB);
const EMBER: [(u8, u8, u8); 3] = [(0xFF, 0x8C, 0x00), (0xFF, 0x55, 0x00), (0xFF, 0xC0, 0x40)];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MascotState {
    #[default]
    Idle,
    Thinking,
    Generating,
    Error,
    Equilibrium,
}

impl MascotState {
    fn mid(self) -> (u8, u8, u8) {
        match self {
            Self::Idle => (0x8C, 0x6D, 0x46),
            Self::Thinking => (0xD4, 0xAF, 0x37),
            Self::Generating => (0xFF, 0x8C, 0x00),
            Self::Error => (0xC8, 0x46, 0x46),
            Self::Equilibrium => (0x6B, 0x9E, 0x78),
        }
    }
}

/// Parsed sprite: level plane (`0..=7`, overlay chars read as 0), overlay pixels, smoke frames.
struct Sprite {
    w: usize,
    h: usize,
    levels: Vec<u8>,
    pipe: Vec<(usize, usize)>,
    ember: Vec<(usize, usize)>,
    smoke: Vec<Vec<(usize, usize)>>,
}

static PARSED: LazyLock<Sprite> = LazyLock::new(|| parse(SPRITE));

fn parse(src: &str) -> Sprite {
    let mut lines = src.lines().enumerate().map(|(i, l)| (i + 1, l));
    let mut next = |what: &str| {
        lines
            .next()
            .unwrap_or_else(|| panic!("gibbs-sprite.txt: missing {what}"))
    };
    let (n, l) = next("header");
    assert_eq!(
        l.trim(),
        "GIBBS-SPRITE 1",
        "gibbs-sprite.txt:{n}: bad header {l:?}"
    );
    let (n, l) = next("size");
    let dims: Vec<usize> = l
        .strip_prefix("size ")
        .unwrap_or_else(|| panic!("gibbs-sprite.txt:{n}: expected `size W H`, got {l:?}"))
        .split_whitespace()
        .map(|t| {
            t.parse()
                .unwrap_or_else(|_| panic!("gibbs-sprite.txt:{n}: bad number {t:?}"))
        })
        .collect();
    let [w, h] = dims[..] else {
        panic!("gibbs-sprite.txt:{n}: expected `size W H`, got {l:?}");
    };
    let (n, l) = next("rows");
    assert_eq!(
        l.trim(),
        "rows",
        "gibbs-sprite.txt:{n}: expected `rows`, got {l:?}"
    );
    let mut s = Sprite {
        w,
        h,
        levels: vec![0; w * h],
        pipe: vec![],
        ember: vec![],
        smoke: vec![],
    };
    for y in 0..h {
        let (n, l) = next("pixel row");
        assert_eq!(
            l.chars().count(),
            w,
            "gibbs-sprite.txt:{n}: row is not {w} chars"
        );
        for (x, c) in l.chars().enumerate() {
            match c {
                '0'..='7' => s.levels[y * w + x] = c as u8 - b'0',
                'p' => s.pipe.push((x, y)),
                'e' => s.ember.push((x, y)),
                _ => panic!("gibbs-sprite.txt:{n}: bad pixel {c:?}"),
            }
        }
    }
    for f in 0..6 {
        let (n, l) = next("smoke frame");
        let mut it = l.split_whitespace();
        assert_eq!(
            it.next(),
            Some("smoke"),
            "gibbs-sprite.txt:{n}: expected `smoke {f} …`"
        );
        assert_eq!(
            it.next().and_then(|t| t.parse().ok()),
            Some(f),
            "gibbs-sprite.txt:{n}: frame index"
        );
        let px = it
            .map(|t| {
                let (x, y) = t
                    .split_once(',')
                    .unwrap_or_else(|| panic!("gibbs-sprite.txt:{n}: bad coordinate {t:?}"));
                let p = |v: &str| -> usize {
                    v.parse()
                        .unwrap_or_else(|_| panic!("gibbs-sprite.txt:{n}: bad coordinate {t:?}"))
                };
                let (x, y) = (p(x), p(y));
                assert!(
                    x < w && y < h,
                    "gibbs-sprite.txt:{n}: {t:?} outside {w}x{h}"
                );
                (x, y)
            })
            .collect();
        s.smoke.push(px);
    }
    s
}

fn conv(th: &Theme, (r, g, b): (u8, u8, u8)) -> Color {
    if th.mono {
        Color::Reset
    } else if th.truecolor {
        Color::Rgb(r, g, b)
    } else {
        Color::Indexed(rgb_to_ansi256(r, g, b))
    }
}

fn lerp(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// Level 0 is VOID; levels 1..=7 run SHADOW → per-state MID → HIGHLIGHT (stops at t = 0, ½, 1).
pub fn ramp(state: MascotState, th: &Theme) -> [Color; 8] {
    let mid = state.mid();
    let mut out = [th.color(Token::Void); 8];
    for (level, slot) in out.iter_mut().enumerate().skip(1) {
        let t = (level - 1) as f32 / 6.0;
        let rgb = if t <= 0.5 {
            lerp(SHADOW, mid, t * 2.0)
        } else {
            lerp(mid, HIGHLIGHT, (t - 0.5) * 2.0)
        };
        *slot = conv(th, rgb);
    }
    out
}

pub struct MascotEngine {
    pub state: MascotState,
    pub frame_tick: usize,
}

impl MascotEngine {
    pub fn new() -> Self {
        Self {
            state: MascotState::Idle,
            frame_tick: 0,
        }
    }

    pub fn set_state(&mut self, state: MascotState) {
        self.state = state;
    }

    pub fn tick(&mut self) {
        self.frame_tick = self.frame_tick.wrapping_add(1);
    }

    pub fn state_name(&self) -> &'static str {
        match self.state {
            MascotState::Idle => "Idle",
            MascotState::Thinking => "Thinking",
            MascotState::Generating => "Generating",
            MascotState::Error => "Error",
            MascotState::Equilibrium => "Equilibrium",
        }
    }

    /// The one ticker line under the cameo.
    pub fn ticker(&self) -> String {
        match self.state {
            MascotState::Idle => "J. Willard Gibbs 1839–1903".into(),
            MascotState::Thinking => match self.frame_tick % 6 {
                0 => "dU = TdS - PdV",
                1 => "∇ × E = -∂B/∂t",
                2 => "u × v = -v × u",
                3 => "F = C - P + 2",
                4 => "∮ (TdS - PdV)",
                _ => "∇ · B = 0",
            }
            .into(),
            MascotState::Generating => "Flow ⚡".into(),
            MascotState::Error => "▲ Spinodal Curve".into(),
            MascotState::Equilibrium => "◆ Triple Point".into(),
        }
    }

    /// Compact badge for rows too small for the cameo.
    pub fn render_mini_badge(&self) -> Vec<Span<'static>> {
        let th = theme();
        let (icon, tok) = match self.state {
            MascotState::Idle => ("●", Token::Brass),
            MascotState::Thinking => ("*", Token::Gold),
            MascotState::Generating => ("⚡", Token::Ember),
            MascotState::Error => ("!", Token::Crimson),
            MascotState::Equilibrium => ("◆", Token::Jade),
        };
        vec![
            Span::styled(format!(" {icon} "), th.fg(tok)),
            Span::styled(
                "Gibbs ",
                th.fg(Token::Parchment).add_modifier(Modifier::BOLD),
            ),
        ]
    }

    /// Ember colour of the pipe at the current tick (cycles every 4 ticks, 2 when generating).
    pub fn ember_color(&self, th: &Theme) -> Color {
        let step = if self.state == MascotState::Generating {
            2
        } else {
            4
        };
        conv(th, EMBER[(self.frame_tick / step) % EMBER.len()])
    }

    /// The sprite scaled to fit `area`, as `▀` cells; `None` when it would be under 12 px wide
    /// (or under `NO_COLOR`, where a half-block image carries no information).
    pub fn render_sprite(&self, area: Rect) -> Option<Vec<Line<'static>>> {
        let th = theme();
        if th.mono {
            return None;
        }
        let s = &*PARSED;
        let scale = (area.width as f32 / s.w as f32)
            .min(2.0 * area.height as f32 / s.h as f32)
            .min(1.0);
        let tw = (s.w as f32 * scale).round() as usize;
        let mut ph = (s.h as f32 * scale).round() as usize;
        ph += ph % 2;
        if tw < 12 || ph < 2 || ph / 2 > area.height as usize {
            return None;
        }
        let colors = ramp(self.state, &th);
        // Box filter of the level plane.
        let mut grid: Vec<Option<Color>> = vec![None; tw * ph];
        for ty in 0..ph {
            let (y0, y1) = (ty * s.h / ph, ((ty + 1) * s.h / ph).max(ty * s.h / ph + 1));
            for tx in 0..tw {
                let (x0, x1) = (tx * s.w / tw, ((tx + 1) * s.w / tw).max(tx * s.w / tw + 1));
                let mut sum = 0u32;
                for y in y0..y1.min(s.h) {
                    for x in x0..x1.min(s.w) {
                        sum += s.levels[y * s.w + x] as u32;
                    }
                }
                let n = ((y1.min(s.h) - y0) * (x1.min(s.w) - x0)).max(1) as f32;
                let level = (sum as f32 / n).round().clamp(0.0, 7.0) as usize;
                if level > 0 {
                    grid[ty * tw + tx] = Some(colors[level]);
                }
            }
        }
        let mut put = |(x, y): (usize, usize), c: Color| {
            grid[(y * ph / s.h).min(ph - 1) * tw + (x * tw / s.w).min(tw - 1)] = Some(c);
        };
        let pipe = conv(&th, SHADOW);
        s.pipe.iter().for_each(|&p| put(p, pipe));
        let div = if self.state == MascotState::Generating {
            2
        } else {
            4
        };
        s.smoke[(self.frame_tick / div) % s.smoke.len()]
            .iter()
            .for_each(|&p| put(p, colors[4]));
        let ember = self.ember_color(&th);
        s.ember.iter().for_each(|&p| put(p, ember));

        let void = th.color(Token::Void);
        let pad_x = (area.width as usize - tw) / 2;
        let rows = ph / 2;
        let pad_y = (area.height as usize - rows) / 2;
        let mut lines: Vec<Line<'static>> = vec![Line::default(); pad_y];
        for r in 0..rows {
            let mut spans = vec![Span::raw(" ".repeat(pad_x))];
            for x in 0..tw {
                let (top, bot) = (grid[2 * r * tw + x], grid[(2 * r + 1) * tw + x]);
                spans.push(match (top, bot) {
                    (None, None) => Span::raw(" "),
                    _ => Span::styled(
                        "▀",
                        Style::default()
                            .fg(top.unwrap_or(void))
                            .bg(bot.unwrap_or(void)),
                    ),
                });
            }
            lines.push(Line::from(spans));
        }
        Some(lines)
    }
}

/// `/gibbs`: the ASCII portrait generated from the sprite, then the biography.
pub fn gibbs_full_portrait() -> String {
    format!(
        "{}\n  JOSIAH WILLARD GIBBS (1839–1903)\n  Professor of Mathematical Physics, Yale University\n  Father of Thermodynamics, Statistical Mechanics & Vector Analysis\n  \"Mathematics is a language.\"\n",
        ASCII.trim_end_matches('\n')
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::widgets::text_width;

    fn truecolor() -> Theme {
        Theme {
            truecolor: true,
            mono: false,
        }
    }

    #[test]
    fn sprite_parses_with_declared_size_and_six_smoke_frames() {
        let s = &*PARSED;
        assert_eq!((s.w, s.h), (48, 60));
        assert_eq!(s.levels.len(), 48 * 60);
        assert_eq!(s.smoke.len(), 6);
        assert!(!s.pipe.is_empty() && !s.ember.is_empty());
        for line in SPRITE.lines().skip(3).take(60) {
            assert_eq!(line.chars().count(), 48);
            assert!(line.chars().all(|c| "01234567pe".contains(c)));
        }
    }

    #[test]
    #[should_panic(expected = "gibbs-sprite.txt:5")]
    fn malformed_sprite_names_the_line() {
        parse("GIBBS-SPRITE 1\nsize 2 2\nrows\n01\n0x\n");
    }

    #[test]
    fn render_sprite_fits_its_area() {
        let mut m = MascotEngine::new();
        let _theme = crate::tui::theme::test_guard(truecolor());
        for tick in [0, 5] {
            m.frame_tick = tick;
            // 12×6 scales to 10 px wide, under the 12 px floor, so the cameo yields to the badge.
            for (w, h, must) in [(48u16, 30u16, true), (24, 15, true), (12, 6, false)] {
                let lines = m.render_sprite(Rect::new(0, 0, w, h));
                assert!(!must || lines.is_some(), "{w}x{h}");
                if let Some(lines) = lines {
                    assert!(lines.len() <= h as usize, "{w}x{h}: {} rows", lines.len());
                    for l in &lines {
                        assert!(l.width() <= w as usize, "{w}x{h}: {}", l.width());
                    }
                }
            }
        }
        assert!(m.render_sprite(Rect::new(0, 0, 10, 4)).is_none());
        assert_eq!(text_width("▀"), 1);
    }

    #[test]
    fn state_ramps_differ_and_ember_cycles() {
        let th = truecolor();
        let states = [
            MascotState::Idle,
            MascotState::Thinking,
            MascotState::Generating,
            MascotState::Error,
            MascotState::Equilibrium,
        ];
        let l4: Vec<Color> = states.iter().map(|&s| ramp(s, &th)[4]).collect();
        for i in 0..l4.len() {
            for j in i + 1..l4.len() {
                assert_ne!(l4[i], l4[j], "{:?} vs {:?}", states[i], states[j]);
            }
        }
        assert_eq!(ramp(MascotState::Idle, &th)[0], th.color(Token::Void));
        let mut m = MascotEngine::new();
        let c0 = m.ember_color(&th);
        m.frame_tick = 4;
        assert_ne!(c0, m.ember_color(&th));
    }

    #[test]
    fn full_portrait_keeps_biography() {
        let p = gibbs_full_portrait();
        assert!(p.contains("JOSIAH WILLARD GIBBS") && p.contains('█'));
    }
}
