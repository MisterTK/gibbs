//! Single source of every Gibbs image: the TUI sprite, the README ASCII block and the SVG art.
//! `cargo run --release --example gibbs_art` (deterministic; run from the repo root).
use gibbs_engine::model::image::ImageBuffer;
use std::fmt::Write as _;
use std::fs;

// ---- fixed constants (DESIGN.md §4) ----
const CROP: (f64, f64, f64, f64) = (0.06, 0.04, 0.98, 0.87);
const VIG_C: (f64, f64) = (0.49, 0.42);
const VIG_R: (f64, f64) = (0.60, 0.62);
const MOUTH: (f64, f64) = (0.250, 0.430);
const STEM_END: (f64, f64) = (0.140, 0.470);
const BOWL: (f64, f64) = (0.120, 0.485);
const BOWL_R: f64 = 0.030; // fraction of source width
const EMBER: (f64, f64) = (0.120, 0.455);
const TUI: (usize, usize) = (48, 60);
const HI: (usize, usize) = (96, 120);

const SHADOW: (u8, u8, u8) = (0x1A, 0x13, 0x0C);
const HIGHLIGHT: (u8, u8, u8) = (0xF5, 0xEE, 0xDB);
const IDLE_MID: (u8, u8, u8) = (0x8C, 0x6D, 0x46);
const BAYER_N: usize = 8;

fn bayer(x: usize, y: usize) -> usize {
    let (x, y) = (x & 7, y & 7);
    let mut v = 0;
    for j in 0..3 {
        let k = 2 - j;
        let (xb, yb) = ((x >> k) & 1, (y >> k) & 1);
        v |= (((xb ^ yb) << 1) | yb) << (2 * j);
    }
    v
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn ramp(level: usize) -> (u8, u8, u8) {
    let t = (level as f64 - 1.0) / 6.0;
    let lerp = |a: (u8, u8, u8), b: (u8, u8, u8), u: f64| {
        let f = |p: u8, q: u8| (p as f64 + (q as f64 - p as f64) * u).round() as u8;
        (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
    };
    if t <= 0.5 {
        lerp(SHADOW, IDLE_MID, t / 0.5)
    } else {
        lerp(IDLE_MID, HIGHLIGHT, (t - 0.5) / 0.5)
    }
}

fn hex(c: (u8, u8, u8)) -> String {
    format!("#{:02X}{:02X}{:02X}", c.0, c.1, c.2)
}

/// Cropped, levelled, vignetted luminance plane.
fn crop_lum(img: &ImageBuffer) -> (Vec<f64>, usize, usize) {
    let (sw, sh) = (img.width as f64, img.height as f64);
    let (x0, y0) = ((CROP.0 * sw) as usize, (CROP.1 * sh) as usize);
    let (x1, y1) = ((CROP.2 * sw) as usize, (CROP.3 * sh) as usize);
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut lum = Vec::with_capacity(cw * ch);
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * img.width + x) * img.channels;
            let d = &img.data;
            lum.push(if img.channels >= 3 {
                0.299 * d[i] as f64 + 0.587 * d[i + 1] as f64 + 0.114 * d[i + 2] as f64
            } else {
                d[i] as f64
            });
        }
    }
    let mut sorted = lum.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = (sorted[sorted.len() * 2 / 100], sorted[sorted.len() * 98 / 100]);
    for (i, v) in lum.iter_mut().enumerate() {
        let (x, y) = ((i % cw) as f64 / cw as f64, (i / cw) as f64 / ch as f64);
        let r = (((x - VIG_C.0) / VIG_R.0).powi(2) + ((y - VIG_C.1) / VIG_R.1).powi(2)).sqrt();
        let lv = ((*v - lo) / (hi - lo) * 255.0).clamp(0.0, 255.0);
        *v = lv * (1.0 - smoothstep(0.65, 1.10, r));
    }
    (lum, cw, ch)
}

struct Art {
    w: usize,
    h: usize,
    grid: Vec<u8>, // '0'..'7', 'p', 'e'
    smoke: Vec<Vec<(usize, usize, u8)>>, // per frame: x, y, level
}

fn build(lum: &[f64], cw: usize, ch: usize, (w, h): (usize, usize)) -> Art {
    let mut grid = vec![b'0'; w * h];
    for y in 0..h {
        for x in 0..w {
            let (xa, xb) = (x * cw / w, ((x + 1) * cw / w).max(x * cw / w + 1));
            let (ya, yb) = (y * ch / h, ((y + 1) * ch / h).max(y * ch / h + 1));
            let mut s = 0.0;
            for yy in ya..yb {
                for xx in xa..xb {
                    s += lum[yy * cw + xx];
                }
            }
            let avg = s / ((xb - xa) * (yb - ya)) as f64;
            let v = avg / 255.0 * 7.0;
            let base = v.floor();
            let thr = (bayer(x, y) as f64 + 0.5) / (BAYER_N * BAYER_N) as f64;
            let level = (base as i32 + (v - base > thr) as i32).clamp(0, 7);
            grid[y * w + x] = b'0' + level as u8;
        }
    }
    // overlay, source-normalized -> sprite px
    let s = w as f64 / TUI.0 as f64;
    let px = |p: (f64, f64)| ((p.0 - CROP.0) / (CROP.2 - CROP.0) * w as f64, (p.1 - CROP.1) / (CROP.3 - CROP.1) * h as f64);
    let (a, b, bc) = (px(MOUTH), px(STEM_END), px(BOWL));
    let br = BOWL_R / (CROP.2 - CROP.0) * w as f64;
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    for y in 0..h {
        for x in 0..w {
            let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
            let t = (((cx - a.0) * dx + (cy - a.1) * dy) / len2).clamp(0.0, 1.0);
            let d_stem = ((cx - a.0 - t * dx).powi(2) + (cy - a.1 - t * dy).powi(2)).sqrt();
            let d_bowl = ((cx - bc.0).powi(2) + (cy - bc.1).powi(2)).sqrt();
            if d_stem <= 0.6 * s || d_bowl <= br {
                grid[y * w + x] = b'p';
            }
        }
    }
    let em = px(EMBER);
    let (ex, ey) = (em.0.floor() as usize / s as usize * s as usize, em.1.floor() as usize / s as usize * s as usize);
    let sz = s as usize;
    for dy in 0..sz {
        for dx in 0..sz {
            grid[(ey + dy) * w + ex + dx] = b'e';
        }
    }
    // smoke: 6 frames, 5 puffs
    let (ew, eh) = (ex as f64, ey as f64);
    let mut smoke = vec![vec![]; 6];
    for (f, frame) in smoke.iter_mut().enumerate() {
        for k in 0..5 {
            let t = (f as f64 + 1.2 * k as f64) % 6.0;
            let y = eh - t * 0.055 * h as f64;
            let x = ew + 0.012 * w as f64 * (0.9 * t + 1.7 * k as f64).sin();
            let (xi, yi) = (x.round() as i64, y.round() as i64);
            let size = if t >= 3.0 { 2 * sz } else { sz } as i64;
            let level = 5 - (t / 2.0).floor() as u8;
            for dy in 0..size {
                for dx in 0..size {
                    let (xx, yy) = (xi + dx, yi + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h && !frame.iter().any(|&(a, b, _)| a == xx as usize && b == yy as usize) {
                        frame.push((xx as usize, yy as usize, level));
                    }
                }
            }
        }
    }
    Art { w, h, grid, smoke }
}

fn sprite_txt(a: &Art) -> String {
    let mut o = format!("GIBBS-SPRITE 1\nsize {} {}\nrows\n", a.w, a.h);
    for r in a.grid.chunks(a.w) {
        o.push_str(std::str::from_utf8(r).unwrap());
        o.push('\n');
    }
    for (f, fr) in a.smoke.iter().enumerate() {
        let _ = write!(o, "smoke {f}");
        for &(x, y, _) in fr {
            let _ = write!(o, " {x},{y}");
        }
        o.push('\n');
    }
    o
}

fn ascii_lines(a: &Art) -> Vec<String> {
    let ink = |x: usize, y: usize| match a.grid[y * a.w + x] {
        b'p' | b'e' => true,
        c => c >= b'4' && c <= b'7',
    };
    (0..a.h / 2)
        .map(|r| {
            (0..a.w)
                .map(|x| match (ink(x, 2 * r), ink(x, 2 * r + 1)) {
                    (true, true) => '█',
                    (true, false) => '▀',
                    (false, true) => '▄',
                    _ => ' ',
                })
                .collect()
        })
        .collect()
}

fn runs_path(w: usize, cells: impl Fn(usize, usize) -> bool, h: usize) -> String {
    let mut d = String::new();
    for y in 0..h {
        let mut x = 0;
        while x < w {
            if cells(x, y) {
                let s = x;
                while x < w && cells(x, y) {
                    x += 1;
                }
                let _ = write!(d, "M{s} {y}h{}v1h-{}z", x - s, x - s);
            } else {
                x += 1;
            }
        }
    }
    d
}

const CSS: &str = "\
@keyframes show6{0%{opacity:1}16.667%{opacity:0}100%{opacity:0}}\
@keyframes show3{0%{opacity:1}33.333%{opacity:0}100%{opacity:0}}\
@keyframes scan{from{transform:translateY(0)}to{transform:translateY(360px)}}\
@keyframes dash{to{stroke-dashoffset:-28}}\
.em{animation:show3 2.4s steps(1,end) infinite;animation-delay:calc(var(--i,0)*0.8s)}\
.sm{animation:show6 4.8s steps(1,end) infinite;animation-delay:calc(var(--i,0)*0.8s)}\
.scan{animation:scan 6s linear infinite}\
.dash{animation:dash 3s linear infinite}\
@media (prefers-reduced-motion: reduce){.anim{animation:none!important}}";

/// Portrait body, pipe, ember, smoke and scanline in 96x120 portrait units.
fn portrait_group(a: &Art) -> String {
    let mut g = String::new();
    for l in 1..=7u8 {
        let d = runs_path(a.w, |x, y| a.grid[y * a.w + x] == b'0' + l, a.h);
        if !d.is_empty() {
            let _ = writeln!(g, r#"<path fill="{}" d="{d}"/>"#, hex(ramp(l as usize)));
        }
    }
    let d = runs_path(a.w, |x, y| a.grid[y * a.w + x] == b'p', a.h);
    let _ = writeln!(g, r##"<path fill="#1A130C" d="{d}"/>"##);
    for (f, fr) in a.smoke.iter().enumerate() {
        let _ = write!(g, r#"<g class="anim sm" style="--i:{f}" opacity="{}">"#, (f == 0) as u8);
        for l in 1..=7u8 {
            let d = runs_path(a.w, |x, y| fr.iter().any(|&(sx, sy, sl)| sx == x && sy == y && sl == l), a.h);
            if !d.is_empty() {
                let _ = write!(g, r#"<path fill="{}" d="{d}"/>"#, hex(ramp(l as usize)));
            }
        }
        g.push_str("</g>\n");
    }
    let de = runs_path(a.w, |x, y| a.grid[y * a.w + x] == b'e', a.h);
    for (i, c) in ["#FF8C00", "#FF5500", "#FFC040"].iter().enumerate() {
        let _ = writeln!(g, r#"<path class="anim em" style="--i:{i}" opacity="{}" fill="{c}" d="{de}"/>"#, (i == 0) as u8);
    }
    g
}

fn portrait_svg(a: &Art) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="288" height="360" shape-rendering="crispEdges" role="img" aria-labelledby="gp-title">
<title id="gp-title">Dithered c.1895 photograph of J. Willard Gibbs</title>
<style>{CSS}</style>
<rect width="{w}" height="{h}" fill="#14100D"/>
{g}</svg>
"##,
        w = a.w,
        h = a.h,
        g = portrait_group(a)
    )
}

fn wireframe() -> String {
    let h = |i: f64, j: f64| 0.6 * (0.35 * i).sin() * (0.3 * j).cos() + 0.012 * (i - 12.0).powi(2);
    let pt = |i: f64, j: f64| format!("{:.1},{:.1}", (i - j) * 14.0, (i + j) * 7.0 - h(i, j) * 40.0);
    let mut s = String::new();
    for a in 0..24 {
        let row: Vec<_> = (0..24).map(|b| pt(a as f64, b as f64)).collect();
        let col: Vec<_> = (0..24).map(|b| pt(b as f64, a as f64)).collect();
        let _ = writeln!(s, r#"<polyline points="{}"/>"#, row.join(" "));
        let _ = writeln!(s, r#"<polyline points="{}"/>"#, col.join(" "));
    }
    s
}

fn hero_svg(a: &Art) -> String {
    let mono = "'SF Mono','Menlo','DejaVu Sans Mono',monospace";
    let serif = "'Bodoni 72','Didot','Playfair Display','Bodoni MT',Georgia,'Times New Roman',serif";
    let eq = [
        ("FUNDAMENTAL FORM", "dU = T dS − P dV + Σ μᵢ dnᵢ"),
        ("ANTISYMMETRY", "u × v = −v × u"),
        ("SOLENOIDAL", "∇ · (∇ × A) = 0"),
        ("TARGET", "GRAPH IR → METAL · GPU-ONLY BY DEFAULT"),
    ];
    let mut lines = String::new();
    for (i, (l, v)) in eq.iter().enumerate() {
        let y = 270 + 30 * i;
        let _ = writeln!(lines, r##"<text x="380" y="{y}" font-size="12" fill="#A8895F" letter-spacing="0.08em">{l}</text><text x="560" y="{y}" font-size="14" fill="#D4AF37">{v}</text>"##);
    }
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 440" width="1200" height="440" role="img" aria-labelledby="gibbs-hero-title gibbs-hero-desc">
<title id="gibbs-hero-title">Gibbs: thermodynamic inference engine</title>
<desc id="gibbs-hero-desc">Ordered-dithered 1895 photograph of J. Willard Gibbs in brass tones with a pipe whose ember glows and smoke rises, the GIBBS wordmark, four governing equations, a faint blue roofline curve and an illustrative thermodynamic surface wireframe (illustrative, not data).</desc>
<defs>
<linearGradient id="wm" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#D4AF37"/><stop offset="1" stop-color="#F5EEDB"/></linearGradient>
<clipPath id="pc"><rect width="96" height="120"/></clipPath>
</defs>
<style>{CSS}text{{font-family:{mono}}}.wm{{font-family:{serif}}}</style>
<rect width="1200" height="440" rx="14" fill="#14100D"/>
<rect x="1" y="1" width="1198" height="438" rx="13" fill="none" stroke="#7A6040" stroke-width="2"/>
<rect x="6" y="6" width="1188" height="428" rx="9" fill="none" stroke="#7A6040" stroke-width="1"/>
<g fill="#D4AF37"><rect x="10" y="10" width="4" height="4"/><rect x="1186" y="10" width="4" height="4"/><rect x="10" y="426" width="4" height="4"/><rect x="1186" y="426" width="4" height="4"/></g>
<g opacity="0.08" fill="none" stroke="#F5EEDB" stroke-width="1.8" transform="translate(1010,60) scale(0.55)"><desc>Illustrative surface h(i,j)=0.6 sin(0.35i) cos(0.3j)+0.012(i-12)^2 on a 24x24 grid; not measured data.</desc>
{wire}</g>
<polyline class="anim dash" fill="none" stroke="#4A86C5" stroke-width="1" opacity="0.6" stroke-dasharray="8 6" points="380,424 720,372 1160,372"/>
<g transform="translate(40,40) scale(3)" shape-rendering="crispEdges">
<g clip-path="url(#pc)">
{g}</g></g>
<g transform="translate(40,40)"><rect class="anim scan" width="288" height="2" fill="#D4AF37" opacity="0.18"/></g>
<text x="40" y="415" font-size="11" fill="#A8895F">J. WILLARD GIBBS · 1839–1903</text>
<text x="40" y="429" font-size="11" fill="#A8895F">PHOTOGRAPH c.1895 · PUBLIC DOMAIN</text>
<text class="wm" x="380" y="170" font-size="120" letter-spacing="0.12em" fill="url(#wm)">GIBBS</text>
<text x="384" y="208" font-size="18" letter-spacing="0.3em" fill="#F5EEDB">THERMODYNAMIC INFERENCE ENGINE</text>
<rect x="380" y="232" width="780" height="1" fill="#7A6040"/>
{lines}<text x="1160" y="418" font-size="12" text-anchor="end" fill="#A8895F">→ QUICKSTART  → BACKENDS  → MODELS  → TUI  → API</text>
</svg>
"##,
        wire = wireframe(),
        g = portrait_group(a)
    )
}

fn readme_block(art: &[String]) -> String {
    let w = art[0].chars().count();
    let mut s = String::from("<!-- gibbs-ascii:begin -->\n```\n");
    let _ = writeln!(s, "╔{}╗", "═".repeat(w + 2));
    for l in art {
        let _ = writeln!(s, "║ {l} ║");
    }
    let _ = writeln!(s, "╚{}╝", "═".repeat(w + 2));
    s.push_str("```\n<!-- gibbs-ascii:end -->");
    s
}

fn write_if(path: &str, body: &str) {
    fs::write(path, body).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn main() {
    let bytes = fs::read("assets/art/source/gibbs-1895.jpg").expect("assets/art/source/gibbs-1895.jpg (see PROVENANCE.md)");
    assert!(bytes.starts_with(&[0xFF, 0xD8]), "source is not a JPEG");
    let img = ImageBuffer::decode(&bytes).expect("decode source");
    let (lum, cw, ch) = crop_lum(&img);
    let tui = build(&lum, cw, ch, TUI);
    let hi = build(&lum, cw, ch, HI);

    write_if("assets/art/gibbs-sprite.txt", &sprite_txt(&tui));
    let ascii = ascii_lines(&tui);
    write_if("assets/art/gibbs-ascii.txt", &(ascii.join("\n") + "\n"));
    write_if("assets/gibbs-portrait.svg", &portrait_svg(&hi));
    write_if("assets/gibbs-hero.svg", &hero_svg(&hi));
    for f in ["assets/gibbs-1873-animated.svg", "assets/maxwell-surface-banner.svg"] {
        let _ = fs::remove_file(f);
    }

    let readme = fs::read_to_string("README.md").expect("README.md");
    let (b, e) = ("<!-- gibbs-ascii:begin -->", "<!-- gibbs-ascii:end -->");
    let (i, j) = (readme.find(b).expect("begin marker"), readme.find(e).expect("end marker") + e.len());
    write_if("README.md", &format!("{}{}{}", &readme[..i], readme_block(&ascii), &readme[j..]));
}
