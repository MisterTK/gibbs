//! Josiah Willard Gibbs pixel art mascot engine & reactive state machine.
//!
//! Honoring Josiah Willard Gibbs (1839–1903), Yale Professor of Mathematical Physics,
//! father of thermodynamics and vector analysis.
//!
//! Provides 5 reactive animated states:
//! - State 1: `Idle & Briar Pipe` (4-frame cycle, glowing ember & drifting smoke rings)
//! - State 2: `Compiling & Calculus` (6-frame thermodynamic fundamental equations)
//! - State 3: `Line-Rate Generating` (4-frame cycle, cyan spectacles & digital stream)
//! - State 4: `Entropy Warning` (2-frame toggle, entropy / memory alert)
//! - State 5: `Equilibrium Success` (thermodynamic equilibrium & crystal badge)

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AvatarStyle {
    #[default]
    Classic,
    Chalkboard,
    Thermo,
    Lattice,
    Cyber,
    Ensemble,
    Vector,
    Surface,
    Triangle,
    Entropy,
    Galvanic,
    Optics,
    Tetrahedron,
    Partition,
    Yale,
}

impl AvatarStyle {
    pub fn cycle(&self) -> Self {
        match self {
            Self::Classic => Self::Chalkboard,
            Self::Chalkboard => Self::Thermo,
            Self::Thermo => Self::Lattice,
            Self::Lattice => Self::Cyber,
            Self::Cyber => Self::Ensemble,
            Self::Ensemble => Self::Vector,
            Self::Vector => Self::Surface,
            Self::Surface => Self::Triangle,
            Self::Triangle => Self::Entropy,
            Self::Entropy => Self::Galvanic,
            Self::Galvanic => Self::Optics,
            Self::Optics => Self::Tetrahedron,
            Self::Tetrahedron => Self::Partition,
            Self::Partition => Self::Yale,
            Self::Yale => Self::Classic,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Chalkboard => "chalkboard",
            Self::Thermo => "thermo",
            Self::Lattice => "lattice",
            Self::Cyber => "cyber",
            Self::Ensemble => "ensemble",
            Self::Vector => "vector",
            Self::Surface => "surface",
            Self::Triangle => "triangle",
            Self::Entropy => "entropy",
            Self::Galvanic => "galvanic",
            Self::Optics => "optics",
            Self::Tetrahedron => "tetrahedron",
            Self::Partition => "partition",
            Self::Yale => "yale",
        }
    }

    pub fn from_str_style(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "classic" => Some(Self::Classic),
            "chalkboard" | "chalk" | "lecture" => Some(Self::Chalkboard),
            "thermo" | "phase" | "diagram" => Some(Self::Thermo),
            "lattice" | "ising" | "spin" => Some(Self::Lattice),
            "cyber" | "neon" | "engine" => Some(Self::Cyber),
            "ensemble" | "statmech" | "ensembles" => Some(Self::Ensemble),
            "vector" | "vortex" | "calc" => Some(Self::Vector),
            "surface" | "energy" | "free" => Some(Self::Surface),
            "triangle" | "phase_rule" | "ternary" => Some(Self::Triangle),
            "entropy" | "wave" | "flux" => Some(Self::Entropy),
            "galvanic" | "emf" | "battery" | "electro" => Some(Self::Galvanic),
            "optics" | "light" | "prism" | "dispersion" => Some(Self::Optics),
            "tetrahedron" | "simplex" | "quaternary" => Some(Self::Tetrahedron),
            "partition" | "ladder" | "states" => Some(Self::Partition),
            "yale" | "sloane" | "tower" | "campus" => Some(Self::Yale),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MascotState {
    #[default]
    Idle,
    Thinking,
    Generating,
    Error,
    Equilibrium,
}

#[derive(Debug, Clone)]
pub struct MascotEngine {
    pub state: MascotState,
    pub avatar_style: AvatarStyle,
    pub frame_tick: usize,
}

impl Default for MascotEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MascotEngine {
    pub fn new() -> Self {
        Self {
            state: MascotState::Idle,
            avatar_style: AvatarStyle::Classic,
            frame_tick: 0,
        }
    }

    pub fn tick(&mut self) {
        self.frame_tick = self.frame_tick.wrapping_add(1);
    }

    pub fn set_state(&mut self, state: MascotState) {
        self.state = state;
    }

    /// Renders a compact miniature badge for narrow display headers (<100 cols).
    pub fn render_mini_badge(&self) -> Vec<Span<'static>> {
        let (icon, label, color) = match self.state {
            MascotState::Idle => ("⚯", "Gibbs: Idle", Color::Green),
            MascotState::Thinking => ("∫", "Gibbs: Calc", Color::Yellow),
            MascotState::Generating => ("⚡", "Gibbs: Flow", Color::Cyan),
            MascotState::Error => ("⚠️", "Gibbs: dS>0", Color::Red),
            MascotState::Equilibrium => ("◆", "Gibbs: ΔG<0", Color::Magenta),
        };
        vec![
            Span::styled(
                format!(" {icon} "),
                Style::default()
                    .fg(Color::Black)
                    .bg(color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {label} "), Style::default().fg(color)),
        ]
    }

    /// Renders the 8-line pixel mascot card for full-width displays.
    pub fn render_lines(&self) -> Vec<Line<'static>> {
        match self.avatar_style {
            AvatarStyle::Classic => self.render_scene_classic(),
            AvatarStyle::Chalkboard => self.render_scene_chalkboard(),
            AvatarStyle::Thermo => self.render_scene_thermo(),
            AvatarStyle::Lattice => self.render_scene_lattice(),
            AvatarStyle::Cyber => self.render_scene_cyber(),
            AvatarStyle::Ensemble => self.render_scene_ensemble(),
            AvatarStyle::Vector => self.render_scene_vector(),
            AvatarStyle::Surface => self.render_scene_surface(),
            AvatarStyle::Triangle => self.render_scene_triangle(),
            AvatarStyle::Entropy => self.render_scene_entropy(),
            AvatarStyle::Galvanic => self.render_scene_galvanic(),
            AvatarStyle::Optics => self.render_scene_optics(),
            AvatarStyle::Tetrahedron => self.render_scene_tetrahedron(),
            AvatarStyle::Partition => self.render_scene_partition(),
            AvatarStyle::Yale => self.render_scene_yale(),
        }
    }

    /// Scene 1: Classic Study & Briar Pipe (Default / Contemplative Idle)
    fn render_scene_classic(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_hair = Style::default().fg(Color::Rgb(160, 140, 110));
        let c_face = Style::default().fg(Color::Rgb(240, 210, 180));
        let c_spec = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let c_coat = Style::default().fg(Color::Rgb(60, 70, 90));

        let frame = tick % 4;
        let ember_color = if frame % 2 == 0 {
            Color::Rgb(209, 154, 102) // Amber #D19A66
        } else {
            Color::Rgb(224, 108, 117) // Ruby #E06C75
        };
        let smoke_str = match frame {
            0 => "   ~ °",
            1 => "  ° o ",
            2 => " o  O ",
            _ => "O   ~ ",
        };
        let blink = (tick % 16) == 0;
        let eye_str = if blink { " -   - " } else { " ⚯   ⚯ " };

        let thinking_eq = match tick % 6 {
            0 => " dG = Vdp-SdT ",
            1 => " dE ≤ TdS-PdV ",
            2 => " F = C-P+2   ",
            3 => " ln Z = -βF  ",
            4 => "  ∇ · B = 0   ",
            _ => " ΔS_univ ≥ 0  ",
        };
        let (badge, c_stat, stream_tok) = match self.state {
            MascotState::Idle => (" J.W. Gibbs ", Style::default().fg(Color::Green), " "),
            MascotState::Thinking => (
                thinking_eq,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
                "*",
            ),
            MascotState::Generating => (
                " Flow Rate >> ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
                "1",
            ),
            MascotState::Error => (
                " ⚠️ dS/dt > 0 ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                "!",
            ),
            MascotState::Equilibrium => (
                " ΔG < 0 (Spont)",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
                "◆",
            ),
        };

        vec![
            Line::from(vec![
                Span::styled("   ╭─────╮   ", c_hair),
                Span::styled(smoke_str, Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("  ╭┤", c_hair),
                Span::styled(eye_str, c_spec),
                Span::styled("├╮  ", c_hair),
                Span::styled(format!(" {stream_tok}"), Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("  ││", c_hair),
                Span::styled("   ▾   ", c_face),
                Span::styled("││  ", c_hair),
                Span::styled(
                    format!(" {stream_tok}"),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
            Line::from(vec![
                Span::styled("  ╰┴─", c_hair),
                Span::styled("▞▀▀▀▚", c_face),
                Span::styled("─┴╯  ", c_hair),
            ]),
            Line::from(vec![
                Span::styled("   ▞▀▀▀▀▀▚", c_coat),
                Span::styled("═╤╝", Style::default().fg(ember_color)),
            ]),
            Line::from(vec![Span::styled("  ▐       ▌  ", c_coat)]),
            Line::from(vec![Span::styled(badge, c_stat)]),
        ]
    }

    /// Scene 2: The Yale Chalkboard Lecture (Compiling & Graph Lowering)
    fn render_scene_chalkboard(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_board = Style::default().fg(Color::White).bg(Color::Rgb(30, 58, 47)); // Slate green #1E3A2F
        let c_chalk = Style::default()
            .fg(Color::Yellow)
            .bg(Color::Rgb(30, 58, 47))
            .add_modifier(Modifier::BOLD);
        let c_part = Style::default().fg(Color::DarkGray);

        let frame = tick % 6;
        let (eq1, eq2) = match frame {
            0 => (" ∇ · B = 0     ", " No Monopoles  "),
            1 => (" ∇ × E = -∂B/∂t", " Faraday's Law "),
            2 => (" dG = Vdp - SdT", " Gibbs Free En "),
            3 => (" dE ≤ TdS - PdV", " 1st & 2nd Law "),
            4 => (" F = C - P + 2 ", " Phase Rule    "),
            _ => (" ln Z = -βF    ", " Partition Fnc "),
        };
        let particle = match tick % 3 {
            0 => " ·  ",
            1 => "  * ",
            _ => " +  ",
        };

        vec![
            Line::from(vec![
                Span::styled(" ┌─[Yale Slate]─┐", c_board),
                Span::styled(particle, c_part),
            ]),
            Line::from(vec![Span::styled(format!(" │{eq1}│"), c_chalk)]),
            Line::from(vec![Span::styled(format!(" │{eq2}│"), c_board)]),
            Line::from(vec![Span::styled(" └──────────────┘", c_board)]),
            Line::from(vec![Span::styled(
                "   ⚯ ✎ Chalk    ",
                Style::default().fg(Color::White),
            )]),
            Line::from(vec![Span::styled(
                "  [J.W. Gibbs]  ",
                Style::default().fg(Color::Green),
            )]),
            Line::from(vec![Span::styled(
                "  Math is a Lang",
                Style::default().fg(Color::Yellow),
            )]),
        ]
    }

    /// Scene 3: Thermodynamic Phase Rule Diagram (Steady-State Line-Rate Decode)
    fn render_scene_thermo(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_solid = Style::default().fg(Color::Cyan);
        let _c_liquid = Style::default().fg(Color::Blue);
        let c_vapor = Style::default().fg(Color::Magenta);
        let c_cursor = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let c_flow = Style::default().fg(Color::Green);

        let phase_step = tick % 4;
        let (cursor_pos, phase_label) = match phase_step {
            0 => (" [S] ●──── [L] ", " Solid-Liquid  "),
            1 => ("  │   ●    │   ", " Triple-Point  "),
            2 => ("  │     ●  │   ", " Liquid-Vapor  "),
            _ => (" [S] ────● [V] ", " Sublimation   "),
        };

        vec![
            Line::from(vec![Span::styled(
                "  ▲ Gibbs Phase ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![Span::styled(
                " /  Ternary  \\  ",
                Style::default().fg(Color::DarkGray),
            )]),
            Line::from(vec![Span::styled(cursor_pos, c_cursor)]),
            Line::from(vec![
                Span::styled("[S]", c_solid),
                Span::styled(" ────── ", Style::default().fg(Color::DarkGray)),
                Span::styled("[V]", c_vapor),
                Span::styled(" ↓", c_flow),
            ]),
            Line::from(vec![Span::styled(
                phase_label,
                Style::default().fg(Color::Cyan),
            )]),
            Line::from(vec![Span::styled(
                " F = C - P + 2  ",
                Style::default().fg(Color::Yellow),
            )]),
            Line::from(vec![Span::styled(" ΔS_irrev = 0   ", c_flow)]),
        ]
    }

    /// Scene 4: Statistical Mechanics Spin Lattice (High Memory Pressure / Paging Alert)
    fn render_scene_lattice(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_spin = Style::default().fg(Color::Cyan);
        let c_hot = Style::default().fg(Color::Red).add_modifier(Modifier::BOLD);

        let spin_chars = ["⠋ ⠙ ⠹", "⠸ ⠼ ⠴", "⠦ ⠧ ⠇", "⠏ ⠋ ⠙"];
        let spins = spin_chars[tick % spin_chars.len()];
        let is_alert = self.state == MascotState::Error;

        let (temp_str, state_str) = if is_alert {
            (" T >> Tc (Critical) ", "⚠️ ENTROPY dS > 0")
        } else {
            (" β = 1/kT (Ordered) ", " Spins: Balanced ")
        };

        vec![
            Line::from(vec![Span::styled(
                " ┌[Ising Lattice]┐",
                Style::default().fg(Color::White),
            )]),
            Line::from(vec![Span::styled(
                format!(" │  {spins}  │"),
                if is_alert { c_hot } else { c_spin },
            )]),
            Line::from(vec![Span::styled(
                format!(" │  {spins}  │"),
                if is_alert { c_hot } else { c_spin },
            )]),
            Line::from(vec![Span::styled(
                " └───────────────┘",
                Style::default().fg(Color::White),
            )]),
            Line::from(vec![Span::styled(
                temp_str,
                Style::default().fg(Color::Yellow),
            )]),
            Line::from(vec![Span::styled(
                state_str,
                if is_alert {
                    c_hot
                } else {
                    Style::default().fg(Color::Green)
                },
            )]),
            Line::from(vec![Span::styled(
                " H = -J ∑ s_i s_j",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 5: Cyber-Gibbs Line-Rate Engine (Peak Throughput / Speculative Drafting Active)
    fn render_scene_cyber(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_neon = Style::default()
            .fg(Color::Rgb(0, 255, 255))
            .add_modifier(Modifier::BOLD);
        let c_tach = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let c_spark = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);

        let sparks = match tick % 4 {
            0 => "⚡ ✦ ",
            1 => "✦ ★ ",
            2 => "★ ✨ ",
            _ => "✨ ⚡ ",
        };

        vec![
            Line::from(vec![Span::styled(" ⚡ Cyber-Gibbs ⚡", c_neon)]),
            Line::from(vec![
                Span::styled(" ╭┤", Style::default().fg(Color::DarkGray)),
                Span::styled(" [●]   [●] ", c_neon),
                Span::styled("├╮ ", Style::default().fg(Color::DarkGray)),
                Span::styled(sparks, c_spark),
            ]),
            Line::from(vec![Span::styled(
                " ││   ▀▾▀   ││ ",
                Style::default().fg(Color::White),
            )]),
            Line::from(vec![Span::styled(
                " ╰┴──▞▀▀▀▚──┴╯ ",
                Style::default().fg(Color::DarkGray),
            )]),
            Line::from(vec![Span::styled(" [████████░░ 80%]", c_tach)]),
            Line::from(vec![Span::styled(" Line-Rate Stream", c_neon)]),
            Line::from(vec![Span::styled(
                " ΔG < 0 (Spont!) ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )]),
        ]
    }

    /// Scene 6: Thermodynamic Ensembles (1902 Elementary Principles in Statistical Mechanics)
    fn render_scene_ensemble(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let c_ring = Style::default().fg(Color::Rgb(100, 180, 240));
        let c_math = Style::default().fg(Color::Yellow);
        let c_dim = Style::default().fg(Color::DarkGray);

        let phase_dot = match tick % 4 {
            0 => " [NVE]  (● ·  )",
            1 => " [NVT]  ( · ● )",
            2 => " [μVT]  (  · ●)",
            _ => " [NPT]  (●  · )",
        };
        let ring_art = match tick % 3 {
            0 => "  ╭─( Z )─╮   ",
            1 => "  ╭─[ Ω ]─╮   ",
            _ => "  ╭─{ Ξ }─╮   ",
        };

        vec![
            Line::from(vec![Span::styled(" ⚯ Stat-Mech Ensembles", c_title)]),
            Line::from(vec![Span::styled(ring_art, c_ring)]),
            Line::from(vec![
                Span::styled(" │", c_ring),
                Span::styled(phase_dot, c_math),
                Span::styled("│", c_ring),
            ]),
            Line::from(vec![Span::styled("  ╰─────────╯   ", c_ring)]),
            Line::from(vec![Span::styled(" Z = ∑ exp(-β E_i) ", c_math)]),
            Line::from(vec![Span::styled(
                " Canonical Ensemble",
                Style::default().fg(Color::Green),
            )]),
            Line::from(vec![Span::styled(" ln Z = -β F (Spont)", c_dim)]),
        ]
    }

    /// Scene 7: Vector Analysis Vortex (1881 Elements of Vector Analysis)
    fn render_scene_vector(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Rgb(255, 140, 0))
            .add_modifier(Modifier::BOLD);
        let c_field = Style::default().fg(Color::Rgb(0, 220, 220));
        let c_quill = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let c_eq = Style::default().fg(Color::White);

        let vortex = match tick % 4 {
            0 => "  ⤹ ↗  ↘ ⤸  ",
            1 => "  ⤿ ↘  ↗ ⤾  ",
            2 => "  ↺ ↗  ↘ ↻  ",
            _ => "  ⤡ ↘  ↗ ⤢  ",
        };

        vec![
            Line::from(vec![Span::styled(" ∇ Vector Analysis ∇", c_title)]),
            Line::from(vec![Span::styled(vortex, c_field)]),
            Line::from(vec![
                Span::styled("   curl v = ∇×v ", c_field),
                Span::styled("✎", c_quill),
            ]),
            Line::from(vec![Span::styled("   div v  = ∇·v ", c_field)]),
            Line::from(vec![Span::styled(" ∇×(∇×A) = ∇(∇·A)-∇²A", c_eq)]),
            Line::from(vec![Span::styled(
                " Solenoidal & Lamellar",
                Style::default().fg(Color::Green),
            )]),
            Line::from(vec![Span::styled(
                " Yale Physics (1881)",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 8: Gibbs Free Energy Surface (1873 Graphical Methods in Thermodynamics)
    fn render_scene_surface(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD);
        let c_surf = Style::default().fg(Color::Rgb(180, 100, 255));
        let c_math = Style::default().fg(Color::Yellow);

        let surf_line1 = match tick % 3 {
            0 => "   /\\___/\\   ",
            1 => "  /  \\_/  \\  ",
            _ => "   \\_/ \\_/   ",
        };
        let surf_line2 = match tick % 3 {
            0 => "  /  ╳ ╳  \\  ",
            1 => "   ╲ ╳ ╳ ╱   ",
            _ => "  ╌───╳───╌  ",
        };

        vec![
            Line::from(vec![Span::styled(" ▞ Gibbs Energy Surface", c_title)]),
            Line::from(vec![Span::styled(surf_line1, c_surf)]),
            Line::from(vec![Span::styled(surf_line2, c_surf)]),
            Line::from(vec![Span::styled("  ╰───────────╯  ", c_surf)]),
            Line::from(vec![Span::styled(" G = H - T·S (Saddle)", c_math)]),
            Line::from(vec![Span::styled(
                " dG = V dp - S dT ",
                Style::default().fg(Color::Green),
            )]),
            Line::from(vec![Span::styled(
                " Maxwell Surface 1873",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 9: Heterogeneous Equilibrium Phase Triangle (1876 Gibbs Phase Rule)
    fn render_scene_triangle(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let c_tri = Style::default().fg(Color::Rgb(80, 220, 140));
        let c_math = Style::default().fg(Color::Yellow);

        let tie_line = match tick % 3 {
            0 => "   /   ●───\\   ",
            1 => "   / ──●──  \\   ",
            _ => "   / ───●   \\   ",
        };

        vec![
            Line::from(vec![Span::styled(" ▲ Ternary Phase Rule", c_title)]),
            Line::from(vec![Span::styled("        ▲        ", c_tri)]),
            Line::from(vec![Span::styled(tie_line, c_tri)]),
            Line::from(vec![Span::styled("     ▲─────▲     ", c_tri)]),
            Line::from(vec![Span::styled("   F = C - P + 2 ", c_math)]),
            Line::from(vec![Span::styled(
                " Invariant Eutectic",
                Style::default().fg(Color::Cyan),
            )]),
            Line::from(vec![Span::styled(
                " Equilibrium (1876)",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 10: Entropy Flux Wave (Statistical Entropy & Boltzmann-Gibbs S = -k ∑ p ln p)
    fn render_scene_entropy(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let c_wave = Style::default().fg(Color::Rgb(255, 100, 100));
        let c_math = Style::default().fg(Color::Cyan);

        let wave = match tick % 4 {
            0 => " ⣀⣄⣤⣦⣶⣷⣿⣷⣶⣦ ",
            1 => " ⣶⣦⣤⣄⣀⣀⣄⣤⣦⣶ ",
            2 => " ⣿⣷⣶⣦⣤⣄⣀⣄⣤⣦ ",
            _ => " ⣤⣦⣶⣷⣿⣷⣶⣦⣤⣄ ",
        };

        vec![
            Line::from(vec![Span::styled(" ≋ Entropy Flux Wave ≋", c_title)]),
            Line::from(vec![Span::styled(wave, c_wave)]),
            Line::from(vec![Span::styled(
                " ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ (Flux)",
                Style::default().fg(Color::Magenta),
            )]),
            Line::from(vec![Span::styled(" S = -k_B ∑ p_i ln p_i", c_math)]),
            Line::from(vec![Span::styled(
                " dS_total / dt ≥ 0",
                Style::default().fg(Color::Green),
            )]),
            Line::from(vec![Span::styled(
                " Statistical Mechanics",
                Style::default().fg(Color::White),
            )]),
            Line::from(vec![Span::styled(
                " Irreversible Flow",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 11: Electrochemical Cell & EMF (1878 Gibbs Free Energy of Galvanic Action)
    fn render_scene_galvanic(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let c_cell = Style::default().fg(Color::Rgb(100, 200, 255));
        let c_ion = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let c_math = Style::default().fg(Color::Green);

        let ions = match tick % 4 {
            0 => " [Zn] ──► e⁻ ──► [Cu] ",
            1 => " [Zn] ───► e⁻ ── [Cu] ",
            2 => " [Zn] ────► e⁻ ─ [Cu] ",
            _ => " [Zn] ─────► e⁻  [Cu] ",
        };

        vec![
            Line::from(vec![Span::styled(" ⚡ Galvanic Cell EMF ⚡", c_title)]),
            Line::from(vec![Span::styled(ions, c_ion)]),
            Line::from(vec![Span::styled("  │░░░│ ~~~~~ │▓▓▓│  ", c_cell)]),
            Line::from(vec![Span::styled("  ╰───╯ SaltB ╰───╯  ", c_cell)]),
            Line::from(vec![Span::styled("  E = -ΔG / (n·F)    ", c_math)]),
            Line::from(vec![Span::styled(
                " Reversible Cell Work",
                Style::default().fg(Color::White),
            )]),
            Line::from(vec![Span::styled(
                " Gibbs Electrochemistry",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 12: Electromagnetic Theory of Light & Wave Dispersion (Gibbs 1888–1889)
    fn render_scene_optics(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD);
        let c_prism = Style::default().fg(Color::White);
        let c_disp = Style::default().fg(Color::Rgb(255, 140, 60));
        let c_math = Style::default().fg(Color::Cyan);

        let rays = match tick % 3 {
            0 => "  ══► /\\ ═══► (Red)  ",
            1 => "  ══► /\\ ───► (Green)",
            _ => "  ══► /\\ ┄┄┄► (Blue) ",
        };

        vec![
            Line::from(vec![Span::styled(" ◈ Wave Optics & Prism ◈", c_title)]),
            Line::from(vec![Span::styled(rays, c_disp)]),
            Line::from(vec![Span::styled("     /  \\ Dispersion ", c_prism)]),
            Line::from(vec![Span::styled("    /____\\ Spectrum  ", c_prism)]),
            Line::from(vec![Span::styled(" ∇×E = -∂B/∂t (EM)   ", c_math)]),
            Line::from(vec![Span::styled(
                " Double Refraction   ",
                Style::default().fg(Color::Green),
            )]),
            Line::from(vec![Span::styled(
                " Yale Optics (1889)  ",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 13: Quaternary Phase Simplex (3D Composition Tetrahedron, Gibbs 1878)
    fn render_scene_tetrahedron(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Rgb(120, 220, 160))
            .add_modifier(Modifier::BOLD);
        let c_tet = Style::default().fg(Color::Rgb(80, 200, 240));
        let c_math = Style::default().fg(Color::Yellow);

        let rot = match tick % 4 {
            0 => "        ▲ (A)        ",
            1 => "       /▲\\ (A)       ",
            2 => "      //▲\\\\ (A)      ",
            _ => "       \\▲/ (A)       ",
        };

        vec![
            Line::from(vec![Span::styled(" ❖ Phase Tetrahedron ❖", c_title)]),
            Line::from(vec![Span::styled(rot, c_tet)]),
            Line::from(vec![Span::styled("   (B) /_|_\\ (C)     ", c_tet)]),
            Line::from(vec![Span::styled("     \\  |  /         ", c_tet)]),
            Line::from(vec![Span::styled("       \\v/ (D)       ", c_tet)]),
            Line::from(vec![Span::styled("   ∑ x_i = 1 (4-Phase)", c_math)]),
            Line::from(vec![Span::styled(
                " Spatial Equilibrium",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 14: Canonical Partition Function Ladder (Statistical Energy Microstates)
    fn render_scene_partition(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Rgb(255, 180, 50))
            .add_modifier(Modifier::BOLD);
        let c_ladder = Style::default().fg(Color::Rgb(100, 180, 255));
        let c_occ = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let c_math = Style::default().fg(Color::Yellow);

        let ladder = match tick % 3 {
            0 => " ───●─── E3 (n=1)    ",
            1 => " ────●── E3 (n=2)    ",
            _ => " ──●──── E3 (n=0)    ",
        };

        vec![
            Line::from(vec![Span::styled(" ≡ Microstate Partition", c_title)]),
            Line::from(vec![Span::styled(ladder, c_occ)]),
            Line::from(vec![Span::styled(" ──●─●── E2 (n=2)    ", c_ladder)]),
            Line::from(vec![Span::styled(" ─●─●─●─ E1 (Ground) ", c_ladder)]),
            Line::from(vec![Span::styled(" Z = ∑ exp(-β·E_i)   ", c_math)]),
            Line::from(vec![Span::styled(
                " P_i = e^(-βE_i) / Z ",
                Style::default().fg(Color::Cyan),
            )]),
            Line::from(vec![Span::styled(
                " StatMech Canonical  ",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Scene 15: Sloane Physics Laboratory (Yale University 1871–1903)
    fn render_scene_yale(&self) -> Vec<Line<'static>> {
        let tick = self.frame_tick;
        let c_title = Style::default()
            .fg(Color::Rgb(80, 160, 255))
            .add_modifier(Modifier::BOLD);
        let c_tower = Style::default().fg(Color::Rgb(170, 160, 140));
        let c_clock = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let c_lamp = Style::default().fg(Color::Rgb(255, 220, 100));

        let chime = match tick % 4 {
            0 => "     |  [12:00]  | ♫ ",
            1 => "     |  [ 3:15]  | ♪ ",
            2 => "     |  [ 6:30]  | ♬ ",
            _ => "     |  [ 9:45]  | ♩ ",
        };

        vec![
            Line::from(vec![Span::styled(" 🏛 Yale Sloane Physics", c_title)]),
            Line::from(vec![Span::styled("         /\\          ", c_tower)]),
            Line::from(vec![Span::styled(chime, c_clock)]),
            Line::from(vec![Span::styled("     |  |====|   |   ", c_tower)]),
            Line::from(vec![Span::styled("    /============\\   ", c_tower)]),
            Line::from(vec![Span::styled("   Desk Lamp: 💡 On  ", c_lamp)]),
            Line::from(vec![Span::styled(
                " Prof. J.W. Gibbs    ",
                Style::default().fg(Color::DarkGray),
            )]),
        ]
    }

    /// Renders a full 12-line vignette card for wide displays (>=120 cols).
    pub fn render_full_vignette(&self) -> Vec<Line<'static>> {
        let mut lines = self.render_lines();
        lines.push(Line::from(vec![
            Span::styled("Style: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("[{}] (F3)", self.avatar_style.name()),
                Style::default().fg(Color::Cyan),
            ),
        ]));
        lines
    }
}

/// Full 24x36 ASCII/half-block historical portrait of Josiah Willard Gibbs for `/gibbs`.
pub fn gibbs_full_portrait() -> String {
    r#"
  ═════════════════════════════════════════════════════════════════
                JOSIAH WILLARD GIBBS (1839 – 1903)
          Professor of Mathematical Physics, Yale University
  ═════════════════════════════════════════════════════════════════

                 ,-----------------.
                /   .-----------.   \
               /   /   ⚯     ⚯   \   \
              |   |       ▾       |   |
               \   \    ▞▀▀▀▚    /   /
                `---'─▞▀▀▀▀▀▀▀▚─'---'
                     ▐         ▌
                    /           \
                   |  dG=Vdp-SdT |
                   |  ΔS_univ ≥ 0|
                    \___________/

  "Mathematics is a language."
     — J. Willard Gibbs, Speech at Yale University Faculty Meeting, 1901

  "The whole is simpler than the sum of its parts."
     — J. Willard Gibbs, Elementary Principles in Statistical Mechanics (1902)

  Fundamental Gibbs Formulations in Engine Core:
  · Gibbs Free Energy:     dG = dH - T dS = V dp - S dT
  · Phase Equilibrium:     F = C - P + 2
  · Fundamental Vector:    ∇ × (∇ × A) = ∇(∇ · A) - ∇²A
  · Maximum Entropy:       dS/dt ≥ 0  (System 2 Equilibrium)
  ═════════════════════════════════════════════════════════════════
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mascot_all_five_states_and_animation_frames() {
        let mut mascot = MascotEngine::new();
        assert_eq!(mascot.state, MascotState::Idle);

        // State 1: Idle with briar pipe
        let l_idle0 = mascot.render_lines();
        assert_eq!(l_idle0.len(), 7);
        assert!(l_idle0[4].to_string().contains("═╤╝"));

        mascot.tick();
        let l_idle1 = mascot.render_lines();
        assert_eq!(l_idle1.len(), 7);

        // State 2: Thinking / Calculus (6 equations)
        mascot.set_state(MascotState::Thinking);
        let mut seen_eqs = Vec::new();
        for _ in 0..6 {
            let l = mascot.render_lines();
            seen_eqs.push(l[6].to_string());
            mascot.tick();
        }
        assert!(seen_eqs.iter().any(|s| s.contains("dG")));
        assert!(seen_eqs.iter().any(|s| s.contains("dE")));
        assert!(seen_eqs.iter().any(|s| s.contains("ΔS")));

        // State 3: Generating
        mascot.set_state(MascotState::Generating);
        let l_gen = mascot.render_lines();
        assert!(l_gen[6].to_string().contains("Flow"));

        // State 4: Error / Entropy warning
        mascot.set_state(MascotState::Error);
        let l_err = mascot.render_lines();
        assert!(l_err[6].to_string().contains("dS/dt") || l_err[6].to_string().contains("ENTROPY"));

        // State 5: Equilibrium
        mascot.set_state(MascotState::Equilibrium);
        let l_eq = mascot.render_lines();
        assert!(l_eq[6].to_string().contains("ΔG < 0"));

        // Mini badge for narrow displays
        let badge = mascot.render_mini_badge();
        assert!(!badge.is_empty());

        // Full portrait text
        let portrait = gibbs_full_portrait();
        assert!(portrait.contains("JOSIAH WILLARD GIBBS"));
        assert!(portrait.contains("Mathematics is a language"));
    }

    #[test]
    fn test_mascot_all_ten_avatar_styles_and_cycling() {
        let mut mascot = MascotEngine::new();
        assert_eq!(mascot.avatar_style, AvatarStyle::Classic);

        // 1. Classic
        let l_classic = mascot.render_lines();
        assert_eq!(l_classic.len(), 7);
        assert!(l_classic[4].to_string().contains("═╤╝"));

        // 2. Chalkboard
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Chalkboard);
        let l_chalk = mascot.render_lines();
        assert_eq!(l_chalk.len(), 7);
        assert!(l_chalk[0].to_string().contains("Yale Slate"));

        // 3. Thermo
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Thermo);
        let l_thermo = mascot.render_lines();
        assert_eq!(l_thermo.len(), 7);
        assert!(l_thermo[0].to_string().contains("Gibbs Phase"));

        // 4. Lattice
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Lattice);
        let l_lat = mascot.render_lines();
        assert_eq!(l_lat.len(), 7);
        assert!(l_lat[0].to_string().contains("Ising Lattice"));

        // 5. Cyber
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Cyber);
        let l_cyber = mascot.render_lines();
        assert_eq!(l_cyber.len(), 7);
        assert!(l_cyber[0].to_string().contains("Cyber-Gibbs"));

        // 6. Ensemble
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Ensemble);
        let l_ens = mascot.render_lines();
        assert_eq!(l_ens.len(), 7);
        assert!(l_ens[0].to_string().contains("Stat-Mech"));

        // 7. Vector
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Vector);
        let l_vec = mascot.render_lines();
        assert_eq!(l_vec.len(), 7);
        assert!(l_vec[0].to_string().contains("Vector Analysis"));

        // 8. Surface
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Surface);
        let l_surf = mascot.render_lines();
        assert_eq!(l_surf.len(), 7);
        assert!(l_surf[0].to_string().contains("Gibbs Energy"));

        // 9. Triangle
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Triangle);
        let l_tri = mascot.render_lines();
        assert_eq!(l_tri.len(), 7);
        assert!(l_tri[0].to_string().contains("Phase Rule"));

        // 10. Entropy
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Entropy);
        let l_ent = mascot.render_lines();
        assert_eq!(l_ent.len(), 7);
        assert!(l_ent[0].to_string().contains("Entropy Flux"));

        // 11. Galvanic
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Galvanic);
        let l_galv = mascot.render_lines();
        assert_eq!(l_galv.len(), 7);
        assert!(l_galv[0].to_string().contains("Galvanic Cell"));

        // 12. Optics
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Optics);
        let l_opt = mascot.render_lines();
        assert_eq!(l_opt.len(), 7);
        assert!(l_opt[0].to_string().contains("Wave Optics"));

        // 13. Tetrahedron
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Tetrahedron);
        let l_tet = mascot.render_lines();
        assert_eq!(l_tet.len(), 7);
        assert!(l_tet[0].to_string().contains("Phase Tetrahedron"));

        // 14. Partition
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Partition);
        let l_part = mascot.render_lines();
        assert_eq!(l_part.len(), 7);
        assert!(l_part[0].to_string().contains("Microstate Partition"));

        // 15. Yale
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Yale);
        let l_yale = mascot.render_lines();
        assert_eq!(l_yale.len(), 7);
        assert!(l_yale[0].to_string().contains("Yale Sloane Physics"));

        // Cycle back to Classic
        mascot.avatar_style = mascot.avatar_style.cycle();
        assert_eq!(mascot.avatar_style, AvatarStyle::Classic);
        // String parsing
        assert_eq!(
            AvatarStyle::from_str_style("classic"),
            Some(AvatarStyle::Classic)
        );
        assert_eq!(
            AvatarStyle::from_str_style("chalkboard"),
            Some(AvatarStyle::Chalkboard)
        );
        assert_eq!(
            AvatarStyle::from_str_style("thermo"),
            Some(AvatarStyle::Thermo)
        );
        assert_eq!(
            AvatarStyle::from_str_style("lattice"),
            Some(AvatarStyle::Lattice)
        );
        assert_eq!(
            AvatarStyle::from_str_style("cyber"),
            Some(AvatarStyle::Cyber)
        );
        assert_eq!(
            AvatarStyle::from_str_style("ensemble"),
            Some(AvatarStyle::Ensemble)
        );
        assert_eq!(
            AvatarStyle::from_str_style("vector"),
            Some(AvatarStyle::Vector)
        );
        assert_eq!(
            AvatarStyle::from_str_style("surface"),
            Some(AvatarStyle::Surface)
        );
        assert_eq!(
            AvatarStyle::from_str_style("triangle"),
            Some(AvatarStyle::Triangle)
        );
        assert_eq!(
            AvatarStyle::from_str_style("entropy"),
            Some(AvatarStyle::Entropy)
        );

        // Full vignette
        let vig = mascot.render_full_vignette();
        assert_eq!(vig.len(), 8);
        assert!(vig[7].to_string().contains("(F3)"));
    }
}
