//! Josiah Willard Gibbs high-resolution pixel art portrait & reactive state machine.
//!
//! Honoring Josiah Willard Gibbs (1839–1903), Yale Professor of Mathematical Physics,
//! father of thermodynamics and vector analysis.
//!
//! Features:
/// Features:
/// - 24x20 high-resolution Unicode half-block sprite (`▄`, `▀`, `█`).
/// - Maxwell-Gibbs 1873 Geometric palette: polished brass, mahogany coat, gold watch chain.
/// - Animated briar pipe with oscillating amber embers and logarithmic vector smoke.
/// - Polished brass spectacles with decode glint.
/// - 5 reactive states: Idle, Thinking, Generating, Error, Equilibrium.
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MascotState {
    #[default]
    Idle,
    Thinking,
    Generating,
    Error,
    Equilibrium,
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

    pub fn render_lines(&self) -> Vec<Line<'static>> {
        self.render_classic(false)
    }

    pub fn render_full_vignette(&self) -> Vec<Line<'static>> {
        self.render_classic(true)
    }

    pub fn render_mini_badge(&self) -> Vec<Span<'static>> {
        let (icon, color) = match self.state {
            MascotState::Idle => ("●", Color::Rgb(212, 175, 55)), // Polished Brass
            MascotState::Thinking => ("*", Color::Rgb(229, 193, 88)), // Antique Gold
            MascotState::Generating => ("⚡", Color::Rgb(255, 140, 0)), // Briar Amber
            MascotState::Error => ("!", Color::Rgb(200, 70, 70)), // Spinodal Crimson
            MascotState::Equilibrium => ("◆", Color::Rgb(107, 158, 120)), // Stable Jade
        };
        vec![
            Span::styled(format!(" {icon} "), Style::default().fg(color)),
            Span::styled(
                "Gibbs ",
                Style::default()
                    .fg(Color::Rgb(245, 238, 219))
                    .add_modifier(Modifier::BOLD),
            ),
        ]
    }

    fn render_classic(&self, _full: bool) -> Vec<Line<'static>> {
        let tick = self.frame_tick;

        // Maxwell-Gibbs 1873 Geometric palette
        let c_hair = Style::default().fg(Color::Rgb(165, 140, 110));
        let c_face = Style::default().fg(Color::Rgb(240, 210, 180));
        let c_beard = Style::default().fg(Color::Rgb(195, 175, 145));
        let c_coat = Style::default().fg(Color::Rgb(50, 36, 28)); // Mahogany dark brown
        let c_waistcoat = Style::default().fg(Color::Rgb(212, 175, 55)); // Polished brass / gold
        let c_spec = if self.state == MascotState::Generating {
            Style::default()
                .fg(Color::Rgb(255, 235, 160)) // Active decode glint
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(229, 193, 88)) // Antique gold rims
        };

        let frame = tick % 4;
        let ember_color = if frame % 2 == 0 {
            Color::Rgb(255, 140, 0)
        } else {
            Color::Rgb(255, 85, 0)
        };

        let smoke_str = match frame {
            0 => "  ~ °   ",
            1 => " ° o ~  ",
            2 => "o  O    ",
            _ => "O  ~ °  ",
        };

        let blink = (tick % 16) == 0;
        let eye_str = if blink { " -   - " } else { " ⚯   ⚯ " };

        let thinking_eq = match tick % 6 {
            0 => " dU = TdS - PdV ",
            1 => " ∇ × E = -∂B/∂t ",
            2 => " u × v = -v × u ",
            3 => " F = C - P + 2  ",
            4 => " ∮ (TdS - PdV)  ",
            _ => " ∇ · B = 0      ",
        };

        let (badge, c_badge) = match self.state {
            MascotState::Idle => (
                " J. Willard Gibbs 1873 ",
                Style::default().fg(Color::Rgb(212, 175, 55)),
            ),
            MascotState::Thinking => (
                thinking_eq,
                Style::default()
                    .fg(Color::Rgb(229, 193, 88))
                    .add_modifier(Modifier::BOLD),
            ),
            MascotState::Generating => (
                " ⚡ Maxwell Surface ",
                Style::default()
                    .fg(Color::Rgb(255, 140, 0))
                    .add_modifier(Modifier::BOLD),
            ),
            MascotState::Error => (
                " ⚠️ Spinodal Curve ",
                Style::default()
                    .fg(Color::Rgb(200, 70, 70))
                    .add_modifier(Modifier::BOLD),
            ),
            MascotState::Equilibrium => (
                " ◆ Triple Point ",
                Style::default()
                    .fg(Color::Rgb(107, 158, 120))
                    .add_modifier(Modifier::BOLD),
            ),
        };

        vec![
            Line::from(vec![
                Span::styled("    ▄▄████████▄▄    ", c_hair),
                Span::styled(smoke_str, Style::default().fg(Color::Rgb(140, 109, 70))),
            ]),
            Line::from(vec![Span::styled("   ██████████████   ", c_hair)]),
            Line::from(vec![
                Span::styled("  ███", c_hair),
                Span::styled(eye_str, c_spec),
                Span::styled("███  ", c_hair),
            ]),
            Line::from(vec![
                Span::styled("  ███", c_hair),
                Span::styled("   ▾   ", c_face),
                Span::styled("███  ", c_hair),
            ]),
            Line::from(vec![
                Span::styled("   ██", c_hair),
                Span::styled(" ▀▀█▀▀ ", c_beard),
                Span::styled("██   ", c_hair),
            ]),
            Line::from(vec![
                Span::styled("    █", c_hair),
                Span::styled("███████", c_beard),
                Span::styled("█    ", c_hair),
                Span::styled(" ═╤╝", Style::default().fg(ember_color)),
            ]),
            Line::from(vec![
                Span::styled("   ▄██", c_coat),
                Span::styled(" ──◇── ", c_waistcoat),
                Span::styled("██▄  ", c_coat),
            ]),
            Line::from(vec![
                Span::styled("  ████", c_coat),
                Span::styled(" watch ", c_waistcoat),
                Span::styled("████ ", c_coat),
            ]),
            Line::from(vec![Span::styled("  ████       ████ ", c_coat)]),
            Line::from(vec![Span::styled(format!(" {badge} "), c_badge)]),
        ]
    }
}

pub fn gibbs_full_portrait() -> String {
    r#"
       .---.
      /     \
     | () () |   JOSIAH WILLARD GIBBS (1839–1903)
      \  ^  /    Professor of Mathematical Physics, Yale University
       |||||     Father of Thermodynamics, Statistical Mechanics & Vector Analysis
       |||||     "Mathematics is a language."
    "#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classic_mascot_rendering() {
        let mut mascot = MascotEngine::new();
        assert_eq!(mascot.state, MascotState::Idle);

        let lines = mascot.render_lines();
        assert!(!lines.is_empty());

        mascot.state = MascotState::Generating;
        let gen_lines = mascot.render_lines();
        assert!(gen_lines
            .iter()
            .any(|l| l.to_string().contains("Maxwell Surface")));
    }
}
