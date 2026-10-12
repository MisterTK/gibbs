//! Josiah Willard Gibbs high-resolution pixel art portrait & reactive state machine.
//!
//! Honoring Josiah Willard Gibbs (1839–1903), Yale Professor of Mathematical Physics,
//! father of thermodynamics and vector analysis.
//!
//! Features:
//! - 24x20 high-resolution Unicode half-block sprite (`▄`, `▀`, `█`).
//! - Warm sepia palette matching historical 1890s photograph.
//! - Animated briar pipe with glowing embers on idle.
//! - Cyan spectacles and digital stream during decode.
//! - 5 reactive animated states: Idle, Thinking, Generating, Error, Equilibrium.

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
            MascotState::Idle => ("●", Color::Green),
            MascotState::Thinking => ("*", Color::Yellow),
            MascotState::Generating => ("⚡", Color::Cyan),
            MascotState::Error => ("!", Color::Red),
            MascotState::Equilibrium => ("◆", Color::Magenta),
        };
        vec![
            Span::styled(format!(" {icon} "), Style::default().fg(color)),
            Span::styled(
                "Gibbs ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]
    }

    fn render_classic(&self, _full: bool) -> Vec<Line<'static>> {
        let tick = self.frame_tick;

        // Warm historical 1890s sepia palette
        let c_hair = Style::default().fg(Color::Rgb(165, 140, 110));
        let c_face = Style::default().fg(Color::Rgb(240, 210, 180));
        let c_beard = Style::default().fg(Color::Rgb(195, 175, 145));
        let c_coat = Style::default().fg(Color::Rgb(60, 50, 45)); // Charcoal sepia
        let c_tie = Style::default().fg(Color::Rgb(140, 60, 50));
        let c_spec = if self.state == MascotState::Generating {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(200, 190, 170))
        };

        let frame = tick % 4;
        let ember_color = if frame % 2 == 0 {
            Color::Rgb(225, 140, 80)
        } else {
            Color::Rgb(245, 90, 60)
        };

        let smoke_str = match frame {
            0 => "   ~ °    ",
            1 => "  ° o ~   ",
            2 => " o  O     ",
            _ => "O   ~ °   ",
        };

        let blink = (tick % 16) == 0;
        let eye_str = if blink { " -   - " } else { " ⚯   ⚯ " };

        let thinking_eq = match tick % 6 {
            0 => " dG = Vdp - SdT ",
            1 => " dE ≤ TdS - PdV ",
            2 => " F = C - P + 2  ",
            3 => " ln Z = -β·F    ",
            4 => " ∇ × E = -∂B/∂t ",
            _ => " ΔS_univ ≥ 0    ",
        };

        let (badge, c_badge) = match self.state {
            MascotState::Idle => (" J. Willard Gibbs ", Style::default().fg(Color::Green)),
            MascotState::Thinking => (
                thinking_eq,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            MascotState::Generating => (
                " ⚡ Line-Rate Gen ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            MascotState::Error => (
                " ⚠️ dS/dt > 0 ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            MascotState::Equilibrium => (
                " ΔG < 0 (Equil) ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
        };

        vec![
            Line::from(vec![
                Span::styled("    ▄▄████████▄▄    ", c_hair),
                Span::styled(smoke_str, Style::default().fg(Color::DarkGray)),
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
                Span::styled(" bowtie ", c_tie),
                Span::styled("██▄  ", c_coat),
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
            .any(|l| l.to_string().contains("Line-Rate")));
    }
}
