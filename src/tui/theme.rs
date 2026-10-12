//! The Gibbs palette (DESIGN.md §2): eleven tokens, one detection of terminal colour depth.
//! Renderers call [`theme()`] once per frame and pass the value down.

use ratatui::style::{Color, Modifier, Style};
use std::sync::{LazyLock, RwLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Void,
    Plinth,
    Hairline,
    Dim,
    Parchment,
    Brass,
    Gold,
    Ember,
    Jade,
    Crimson,
    Cyanotype,
}

/// (token, sRGB, ANSI-256 index). Contrast ratios against VOID/PLINTH are enforced by the tests.
pub const PALETTE: [(Token, (u8, u8, u8), u8); 11] = [
    (Token::Void, (0x14, 0x10, 0x0D), 233),
    (Token::Plinth, (0x1B, 0x15, 0x10), 234),
    (Token::Hairline, (0x7A, 0x60, 0x40), 95),
    (Token::Dim, (0xA8, 0x89, 0x5F), 137),
    (Token::Parchment, (0xF5, 0xEE, 0xDB), 230),
    (Token::Brass, (0xD4, 0xAF, 0x37), 178),
    (Token::Gold, (0xE5, 0xC1, 0x58), 179),
    (Token::Ember, (0xFF, 0x8C, 0x00), 208),
    (Token::Jade, (0x6B, 0x9E, 0x78), 72),
    (Token::Crimson, (0xE2, 0x60, 0x5A), 167),
    (Token::Cyanotype, (0x4A, 0x86, 0xC5), 68),
];

fn entry(t: Token) -> ((u8, u8, u8), u8) {
    let (_, rgb, idx) = PALETTE[t as usize];
    (rgb, idx)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Theme {
    pub truecolor: bool,
    /// `NO_COLOR`: no colour at all; hierarchy is carried by bold/dim/reverse.
    pub mono: bool,
}

impl Theme {
    pub fn detect() -> Self {
        Self::from_env(|k| std::env::var(k).ok())
    }

    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Self {
        let mono = get("NO_COLOR").is_some_and(|v| !v.is_empty());
        let truecolor = get("COLORTERM").is_some_and(|v| v == "truecolor" || v == "24bit");
        Self { truecolor, mono }
    }

    pub fn color(&self, t: Token) -> Color {
        let (rgb, idx) = entry(t);
        if self.mono {
            Color::Reset
        } else if self.truecolor {
            Color::Rgb(rgb.0, rgb.1, rgb.2)
        } else {
            Color::Indexed(idx)
        }
    }

    pub fn fg(&self, t: Token) -> Style {
        let s = Style::default().fg(self.color(t));
        match (self.mono, t) {
            (true, Token::Dim | Token::Hairline) => s.add_modifier(Modifier::DIM),
            _ => s,
        }
    }

    pub fn bg(&self, t: Token) -> Style {
        if self.mono {
            Style::default()
        } else {
            Style::default().bg(self.color(t))
        }
    }

    /// Efficiency tier: JADE ≥ 90, GOLD 60–90, EMBER 30–60, CRIMSON < 30.
    pub fn tier(pct: f32) -> Token {
        if pct >= 90.0 {
            Token::Jade
        } else if pct >= 60.0 {
            Token::Gold
        } else if pct >= 30.0 {
            Token::Ember
        } else {
            Token::Crimson
        }
    }

    /// Foreground for a colour `Token` or the matching reverse-video in mono (selection rows).
    pub fn selected(&self) -> Style {
        if self.mono {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            self.fg(Token::Parchment).patch(self.bg(Token::Hairline))
        }
    }
}

/// Nearest xterm-256 index: 6×6×6 cube vs the 24-step grayscale ramp, by squared distance.
pub fn rgb_to_ansi256(r: u8, g: u8, b: u8) -> u8 {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let d2 = |a: (u8, u8, u8), b: (u8, u8, u8)| -> i32 {
        let f = |x: u8, y: u8| (x as i32 - y as i32).pow(2);
        f(a.0, b.0) + f(a.1, b.1) + f(a.2, b.2)
    };
    let near = |v: u8| {
        (0..6)
            .min_by_key(|&i| (LEVELS[i] as i32 - v as i32).abs())
            .unwrap_or(0)
    };
    let (ci, cj, ck) = (near(r), near(g), near(b));
    let cube = (LEVELS[ci], LEVELS[cj], LEVELS[ck]);
    let cube_idx = 16 + 36 * ci + 6 * cj + ck;
    let avg = (r as i32 + g as i32 + b as i32) / 3;
    let step = ((avg - 8 + 5) / 10).clamp(0, 23);
    let gv = (8 + 10 * step) as u8;
    if d2((r, g, b), (gv, gv, gv)) < d2((r, g, b), cube) {
        (232 + step) as u8
    } else {
        cube_idx as u8
    }
}

/// WCAG 2.x contrast ratio of two sRGB colours.
pub fn contrast(a: (u8, u8, u8), b: (u8, u8, u8)) -> f32 {
    fn lum((r, g, b): (u8, u8, u8)) -> f32 {
        let lin = |c: u8| {
            let c = c as f32 / 255.0;
            if c <= 0.03928 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    }
    let (la, lb) = (lum(a), lum(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

fn cell() -> &'static RwLock<Theme> {
    static T: LazyLock<RwLock<Theme>> = LazyLock::new(|| RwLock::new(Theme::detect()));
    &T
}

pub fn set(t: Theme) {
    *cell().write().unwrap_or_else(|e| e.into_inner()) = t;
}

pub fn theme() -> Theme {
    *cell().read().unwrap_or_else(|e| e.into_inner())
}

/// Tests share the process-global theme; holding this guard serializes them and restores it.
#[cfg(test)]
pub(crate) struct TestGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev: Theme,
}

#[cfg(test)]
pub(crate) fn test_guard(t: Theme) -> TestGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = theme();
    set(t);
    TestGuard { _lock: lock, prev }
}

#[cfg(test)]
impl Drop for TestGuard {
    fn drop(&mut self) {
        set(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(t: Token) -> (u8, u8, u8) {
        entry(t).0
    }

    #[test]
    fn palette_indexed_by_discriminant() {
        for (i, (t, _, _)) in PALETTE.iter().enumerate() {
            assert_eq!(*t as usize, i);
        }
    }

    #[test]
    fn text_tokens_meet_aa_on_void_and_plinth() {
        for (t, c, _) in PALETTE {
            match t {
                Token::Void | Token::Plinth => {}
                Token::Hairline => {
                    assert!(contrast(c, rgb(Token::Void)) >= 3.0);
                    assert!(contrast(c, rgb(Token::Plinth)) >= 3.0);
                }
                _ => {
                    for bg in [Token::Void, Token::Plinth] {
                        let r = contrast(c, rgb(bg));
                        assert!(r >= 4.5, "{t:?} on {bg:?}: {r:.2}");
                    }
                }
            }
        }
    }

    #[test]
    fn ansi256_matches_table_within_one_cube_step() {
        let coords = |i: u8| -> Option<(i32, i32, i32)> {
            (16..232).contains(&i).then(|| {
                let n = (i - 16) as i32;
                (n / 36, (n / 6) % 6, n % 6)
            })
        };
        for (t, (r, g, b), want) in PALETTE {
            let got = rgb_to_ansi256(r, g, b);
            let ok = got == want
                || match (coords(got), coords(want)) {
                    (Some(a), Some(w)) => {
                        (a.0 - w.0).abs() <= 1 && (a.1 - w.1).abs() <= 1 && (a.2 - w.2).abs() <= 1
                    }
                    _ => (got as i32 - want as i32).abs() <= 1,
                };
            assert!(ok, "{t:?}: got {got}, table {want}");
        }
    }

    #[test]
    fn detect_honours_no_color_and_colorterm() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert!(Theme::from_env(env(&[("NO_COLOR", "1")])).mono);
        assert!(!Theme::from_env(env(&[("NO_COLOR", "")])).mono);
        assert!(Theme::from_env(env(&[("COLORTERM", "truecolor")])).truecolor);
        assert!(Theme::from_env(env(&[("COLORTERM", "24bit")])).truecolor);
        assert!(!Theme::from_env(env(&[])).truecolor);
    }

    #[test]
    fn tiers_follow_thresholds() {
        assert_eq!(Theme::tier(95.0), Token::Jade);
        assert_eq!(Theme::tier(90.0), Token::Jade);
        assert_eq!(Theme::tier(75.0), Token::Gold);
        assert_eq!(Theme::tier(45.0), Token::Ember);
        assert_eq!(Theme::tier(10.0), Token::Crimson);
    }

    #[test]
    fn mono_theme_has_no_colour() {
        let t = Theme {
            truecolor: true,
            mono: true,
        };
        for (tok, _, _) in PALETTE {
            assert_eq!(t.color(tok), Color::Reset);
        }
    }
}
