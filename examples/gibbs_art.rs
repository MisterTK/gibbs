//! Single source of every Gibbs image: the TUI sprite and ASCII art, the README hero, architecture and roofline SVGs, and the README bench table.
//! `cargo run --release --example gibbs_art` (deterministic; run from the repo root).
use gibbs_engine::model::image::ImageBuffer;
use serde_json::Value;
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
const SUITE: &str = "bench/loop.json";
const BENCH: &str = "autoresearch/bench-readme.jsonl";
const META: &str = "autoresearch/bench-readme.meta.json";

const VOID: &str = "#14100D";
const PLINTH: &str = "#1B1510";
const HAIRLINE: &str = "#7A6040";
const DIM: &str = "#A8895F";
const PARCHMENT: &str = "#F5EEDB";
const BRASS: &str = "#D4AF37";
const GOLD: &str = "#E5C158";
const EMBER_HEX: &str = "#FF8C00";
const JADE: &str = "#6B9E78";
const CRIMSON: &str = "#E2605A";
const CYANOTYPE: &str = "#4A86C5";
const MONO: &str = "'SF Mono','Menlo','DejaVu Sans Mono',monospace";
const SERIF: &str =
    "'Bodoni 72','Didot','Playfair Display','Bodoni MT',Georgia,'Times New Roman',serif";

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
    let (lo, hi) = (
        sorted[sorted.len() * 2 / 100],
        sorted[sorted.len() * 98 / 100],
    );
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
    grid: Vec<u8>,                       // '0'..'7', 'p', 'e'
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
    let px = |p: (f64, f64)| {
        (
            (p.0 - CROP.0) / (CROP.2 - CROP.0) * w as f64,
            (p.1 - CROP.1) / (CROP.3 - CROP.1) * h as f64,
        )
    };
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
    let (ex, ey) = (
        em.0.floor() as usize / s as usize * s as usize,
        em.1.floor() as usize / s as usize * s as usize,
    );
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
                    if xx >= 0
                        && yy >= 0
                        && (xx as usize) < w
                        && (yy as usize) < h
                        && !frame
                            .iter()
                            .any(|&(a, b, _)| a == xx as usize && b == yy as usize)
                    {
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
.em{animation:show3 2.4s steps(1,end) infinite;animation-delay:calc(var(--i,0)*0.8s)}\
.sm{animation:show6 4.8s steps(1,end) infinite;animation-delay:calc(var(--i,0)*0.8s)}\
.scan{animation:scan 6s linear infinite}\
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
        let _ = write!(
            g,
            r#"<g class="anim sm" style="--i:{f}" opacity="{}">"#,
            (f == 0) as u8
        );
        for l in 1..=7u8 {
            let d = runs_path(
                a.w,
                |x, y| fr.iter().any(|&(sx, sy, sl)| sx == x && sy == y && sl == l),
                a.h,
            );
            if !d.is_empty() {
                let _ = write!(g, r#"<path fill="{}" d="{d}"/>"#, hex(ramp(l as usize)));
            }
        }
        g.push_str("</g>\n");
    }
    let de = runs_path(a.w, |x, y| a.grid[y * a.w + x] == b'e', a.h);
    for (i, c) in ["#FF8C00", "#FF5500", "#FFC040"].iter().enumerate() {
        let _ = writeln!(
            g,
            r#"<path class="anim em" style="--i:{i}" opacity="{}" fill="{c}" d="{de}"/>"#,
            (i == 0) as u8
        );
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

const E2B_QAT_ID: &str = "unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf";

#[derive(Clone)]
struct Row {
    key: String,
    label: String,
    tok_s: f64,
    pct: f64,
    roofline_tok_s: f64,
    gpu_ms: Option<f64>,
    host_ms: Option<f64>,
    dispatches: Option<f64>,
    ceiling: f64,
    error: Option<String>,
}

struct Bench {
    rows: Vec<Row>,
    ok: Vec<Row>,
    geomean: f64,
    ceiling: f64,
    e2b: Row,
    peak: Row,
    k: usize,
    decode_tokens: u64,
    runs: u64,
    reencode: String,
    commit: String,
    date: String,
    device: String,
    note: Option<String>,
}

fn tier(p: f64) -> &'static str {
    if p >= 90.0 {
        JADE
    } else if p >= 60.0 {
        GOLD
    } else if p >= 30.0 {
        EMBER_HEX
    } else {
        CRIMSON
    }
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            c => o.push(c),
        }
    }
    o
}

fn trunc_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut o: String = s.chars().take(max.saturating_sub(1)).collect();
        o.push('…');
        o
    }
}

fn label_for(key: &str, id: &str) -> String {
    match key {
        "unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf" => {
            "Gemma 4 E2B QAT · UD-Q2_K_XL".into()
        }
        "unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf" => {
            "Gemma 4 E2B · Q4_K_M".into()
        }
        "unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf" => {
            "Qwen3.5 2B · UD-Q4_K_XL".into()
        }
        "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M.gguf" => {
            "Qwen3.5 9B (DavidAU MTP) · IQ2_M".into()
        }
        "unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf" => {
            "Llama 3.2 1B · Q4_0".into()
        }
        "HuggingFaceTB/SmolLM2-135M-Instruct" => "SmolLM2 135M · safetensors".into(),
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" => "Llama 3.2 1B · bnb 4-bit".into(),
        "unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf" => {
            "Gemma 4 26B-A4B · UD-IQ2_XXS".into()
        }
        "e2b-qat-ctx4k" => "Gemma 4 E2B QAT · 4k context".into(),
        _ => {
            let seg = id.rsplit('/').next().unwrap_or(id);
            let stem = seg.strip_suffix(".gguf").unwrap_or(seg);
            trunc_chars(stem, 34)
        }
    }
}

fn bench() -> Bench {
    let suite: Value =
        serde_json::from_str(&fs::read_to_string(SUITE).expect(SUITE)).expect("parse suite");
    let models = suite["models"].as_array().expect("suite models");
    let defs = &suite["defaults"];
    let decode_tokens = defs["decode_tokens"].as_u64().expect("decode_tokens");
    let runs = defs["runs"].as_u64().expect("runs");
    let reencode = defs["reencode"].as_str().expect("reencode").to_string();

    let raw = fs::read_to_string(BENCH).expect(BENCH);
    let json_lines: Vec<Value> = raw
        .lines()
        .filter(|l| l.starts_with('{'))
        .map(|l| serde_json::from_str(l).expect("parse bench json line"))
        .collect();
    assert_eq!(
        json_lines.len(),
        models.len(),
        "bench row count ({}) != suite model count ({})",
        json_lines.len(),
        models.len()
    );

    let rows: Vec<Row> = models
        .iter()
        .zip(json_lines.iter())
        .map(|(m, r)| {
            let id = m["id"].as_str().expect("model id");
            let key = m["name"].as_str().unwrap_or(id).to_string();
            let label = label_for(&key, id);
            if let Some(err) = r.get("error").and_then(Value::as_str) {
                Row {
                    key,
                    label,
                    tok_s: 0.0,
                    pct: 0.0,
                    roofline_tok_s: 0.0,
                    gpu_ms: None,
                    host_ms: None,
                    dispatches: None,
                    ceiling: 0.0,
                    error: Some(err.to_string()),
                }
            } else {
                Row {
                    key,
                    label,
                    tok_s: r["decode_tok_s"].as_f64().expect("decode_tok_s"),
                    pct: r["roofline_pct"].as_f64().expect("roofline_pct"),
                    roofline_tok_s: r["roofline_tok_s"].as_f64().expect("roofline_tok_s"),
                    gpu_ms: r.get("gpu_ms").and_then(Value::as_f64),
                    host_ms: r.get("host_ms").and_then(Value::as_f64),
                    dispatches: r.get("dispatches").and_then(Value::as_f64),
                    ceiling: r["ceiling_gbps"].as_f64().expect("ceiling_gbps"),
                    error: None,
                }
            }
        })
        .collect();

    let ok: Vec<Row> = rows.iter().filter(|r| r.error.is_none()).cloned().collect();
    assert!(!ok.is_empty(), "no successful rows in {BENCH}");
    let geomean = (ok.iter().map(|r| r.pct.max(0.1).ln()).sum::<f64>() / ok.len() as f64).exp();
    let ceiling = ok[0].ceiling;
    let e2b = ok
        .iter()
        .find(|r| r.key == E2B_QAT_ID)
        .expect("E2B QAT row missing")
        .clone();
    let peak = ok
        .iter()
        .max_by(|a, b| a.tok_s.total_cmp(&b.tok_s))
        .expect("peak row")
        .clone();
    let k = ok.iter().filter(|r| r.pct >= 90.0).count();

    let meta: Value =
        serde_json::from_str(&fs::read_to_string(META).expect(META)).expect("parse meta");
    let commit = meta["commit"].as_str().expect("meta.commit").to_string();
    let date = meta["date"].as_str().expect("meta.date").to_string();
    let device = meta["device"].as_str().expect("meta.device").to_string();
    let note = meta.get("note").and_then(Value::as_str).map(String::from);

    Bench {
        rows,
        ok,
        geomean,
        ceiling,
        e2b,
        peak,
        k,
        decode_tokens,
        runs,
        reencode,
        commit,
        date,
        device,
        note,
    }
}

fn hero_svg(a: &Art, b: &Bench) -> String {
    let n = b.ok.len();
    let g_col = tier(b.geomean);
    let peak_lbl = esc(&trunc_chars(&b.peak.label.to_uppercase(), 32));
    let mut prov = format!(
        "{} · METAL · BUILD {} · {}",
        b.device.to_uppercase(),
        b.commit.to_uppercase(),
        b.date.to_uppercase()
    );
    if let Some(note) = &b.note {
        let _ = write!(prov, " · {}", note.to_uppercase());
    }
    let prov = esc(&prov);
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 440" width="1200" height="440" role="img" aria-labelledby="gibbs-hero-title gibbs-hero-desc">
<title id="gibbs-hero-title">Gibbs: Metal-native inference engine</title>
<desc id="gibbs-hero-desc">Ordered-dithered 1895 photograph of J. Willard Gibbs in brass tones with a pipe whose ember glows and smoke rises, the GIBBS wordmark, and four measured figures from the README benchmark: roofline geomean {geomean:.1}%, Gemma 4 E2B QAT {e2b:.0} tok/s, streaming ceiling {ceil:.0} GB/s, peak {peak:.0} tok/s.</desc>
<defs>
<linearGradient id="wm" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{BRASS}"/><stop offset="1" stop-color="{PARCHMENT}"/></linearGradient>
<clipPath id="pc"><rect width="96" height="120"/></clipPath>
</defs>
<style>{CSS}text{{font-family:{MONO}}}.wm{{font-family:{SERIF}}}</style>
<rect width="1200" height="440" rx="14" fill="{VOID}"/>
<rect x="1" y="1" width="1198" height="438" rx="13" fill="none" stroke="{HAIRLINE}" stroke-width="2"/>
<rect x="6" y="6" width="1188" height="428" rx="9" fill="none" stroke="{HAIRLINE}" stroke-width="1"/>
<g fill="{BRASS}"><rect x="10" y="10" width="4" height="4"/><rect x="1186" y="10" width="4" height="4"/><rect x="10" y="426" width="4" height="4"/><rect x="1186" y="426" width="4" height="4"/></g>
<g transform="translate(40,40) scale(3)" shape-rendering="crispEdges">
<g clip-path="url(#pc)">
{g}</g></g>
<g transform="translate(40,40)"><rect class="anim scan" width="288" height="2" fill="{BRASS}" opacity="0.18"/></g>
<text x="40" y="415" font-size="11" fill="{DIM}">J. WILLARD GIBBS · 1839–1903</text>
<text x="40" y="429" font-size="11" fill="{DIM}">PHOTOGRAPH c.1895 · PUBLIC DOMAIN</text>
<text class="wm" x="380" y="156" font-size="120" letter-spacing="0.12em" fill="url(#wm)">GIBBS</text>
<text x="384" y="196" font-size="18" letter-spacing="0.3em" fill="{PARCHMENT}">METAL-NATIVE INFERENCE ENGINE</text>
<rect x="380" y="216" width="780" height="1" fill="{HAIRLINE}"/>
<text x="380" y="248" font-size="12" fill="{DIM}" letter-spacing="0.08em">ROOFLINE GEOMEAN · {n} MODELS</text>
<text x="380" y="288" font-size="36" fill="{g_col}">{geomean:.1}%</text>
<text x="780" y="248" font-size="12" fill="{DIM}" letter-spacing="0.08em">GEMMA 4 E2B QAT · DECODE</text>
<text x="780" y="288" font-size="36" fill="{BRASS}">{e2b:.0} tok/s</text>
<text x="380" y="328" font-size="12" fill="{DIM}" letter-spacing="0.08em">STREAMING CEILING</text>
<text x="380" y="368" font-size="36" fill="{CYANOTYPE}">{ceil:.0} GB/s</text>
<text x="780" y="328" font-size="12" fill="{DIM}" letter-spacing="0.08em">PEAK · {peak_lbl}</text>
<text x="780" y="368" font-size="36" fill="{BRASS}">{peak:.0} tok/s</text>
<text x="380" y="412" font-size="11" fill="{DIM}" letter-spacing="0.08em">{prov}</text>
</svg>
"##,
        geomean = b.geomean,
        e2b = b.e2b.tok_s,
        ceil = b.ceiling,
        peak = b.peak.tok_s,
        g = portrait_group(a)
    )
}

fn roofline_svg(b: &Bench) -> String {
    let n = b.ok.len();
    let h = 160 + 40 * n + 80;
    let mut sub = format!(
        "{} · METAL · CEILING {:.0} GB/S · BUILD {} · {} · {} TOKENS × {} RUNS, MEDIAN",
        b.device.to_uppercase(),
        b.ceiling,
        b.commit.to_uppercase(),
        b.date.to_uppercase(),
        b.decode_tokens,
        b.runs
    );
    if let Some(note) = &b.note {
        let _ = write!(sub, " · {}", note.to_uppercase());
    }
    let sub = esc(&sub);
    let desc_rows: Vec<String> =
        b.ok.iter()
            .map(|r| format!("{} {:.0} tok/s {:.1}%", esc(&r.label), r.tok_s, r.pct))
            .collect();
    let desc = format!(
        "Bar chart of median decode speed as a percent of the measured memory-bandwidth roofline on {}, build {}: {}; geomean {:.1}%.",
        esc(&b.device),
        esc(&b.commit),
        desc_rows.join("; "),
        b.geomean
    );

    let mut rows_svg = String::new();
    for (i, r) in b.ok.iter().enumerate() {
        let y = 160 + 40 * i;
        let bw = (6.0 * r.pct.min(100.0)).round() as usize;
        let col = tier(r.pct);
        let lbl = esc(&r.label);
        let _ = writeln!(
            rows_svg,
            r#"<text x="48" y="{ty}" font-size="14" fill="{PARCHMENT}">{lbl}</text><rect x="360" y="{by}" width="600" height="16" fill="{PLINTH}" stroke="{HAIRLINE}" stroke-width="1"/><rect x="360" y="{by}" width="{bw}" height="16" fill="{col}"/><text x="984" y="{ty}" font-size="14" fill="{PARCHMENT}">{tok:.0} tok/s</text><text x="1152" y="{ty}" font-size="14" text-anchor="end" fill="{col}">{pct:.1}%</text>"#,
            ty = y + 24,
            by = y + 10,
            tok = r.tok_s,
            pct = r.pct
        );
    }
    let ty2 = 160 + 40 * n - 4;
    let fy = 160 + 40 * n + 12;
    let ly = 160 + 40 * n + 44;
    let sy = ly - 10;
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" width="1200" height="{h}" role="img" aria-labelledby="gibbs-roofline-title gibbs-roofline-desc">
<title id="gibbs-roofline-title">Gibbs decode speed vs memory-bandwidth roofline</title>
<desc id="gibbs-roofline-desc">{desc}</desc>
<style>text{{font-family:{MONO}}}.serif{{font-family:{SERIF}}}</style>
<rect width="1200" height="{h}" rx="14" fill="{VOID}"/>
<rect x="1" y="1" width="1198" height="{h_2}" rx="13" fill="none" stroke="{HAIRLINE}" stroke-width="2"/>
<rect x="6" y="6" width="1188" height="{h_12}" rx="9" fill="none" stroke="{HAIRLINE}" stroke-width="1"/>
<g fill="{BRASS}"><rect x="10" y="10" width="4" height="4"/><rect x="1186" y="10" width="4" height="4"/><rect x="10" y="{h_14}" width="4" height="4"/><rect x="1186" y="{h_14}" width="4" height="4"/></g>
<text x="48" y="56" font-size="12" fill="{DIM}" letter-spacing="0.12em">DECODE SPEED · PERCENT OF MEMORY-BANDWIDTH ROOFLINE</text>
<text class="serif" x="48" y="96" font-size="28" fill="{PARCHMENT}">{geomean:.1}% geomean across {n} models</text>
<text x="48" y="124" font-size="12" fill="{DIM}">{sub}</text>
{rows_svg}<line x1="900" y1="148" x2="900" y2="{ty2}" stroke="{CYANOTYPE}" stroke-width="1" stroke-dasharray="4 4"/>
<text x="900" y="144" font-size="11" fill="{CYANOTYPE}" text-anchor="middle">90% TARGET</text>
<rect x="48" y="{fy}" width="1104" height="1" fill="{HAIRLINE}"/>
<rect x="48" y="{sy}" width="12" height="12" fill="{JADE}"/><text x="68" y="{ly}" font-size="12" fill="{DIM}">≥ 90%</text>
<rect x="184" y="{sy}" width="12" height="12" fill="{GOLD}"/><text x="204" y="{ly}" font-size="12" fill="{DIM}">60–90%</text>
<rect x="320" y="{sy}" width="12" height="12" fill="{EMBER_HEX}"/><text x="340" y="{ly}" font-size="12" fill="{DIM}">30–60%</text>
<rect x="456" y="{sy}" width="12" height="12" fill="{CRIMSON}"/><text x="476" y="{ly}" font-size="12" fill="{DIM}">&lt; 30%</text>
<text x="1152" y="{ly}" font-size="11" fill="{DIM}" text-anchor="end">ROOFLINE = CEILING ÷ BYTES STREAMED PER TOKEN</text>
</svg>
"##,
        h_2 = h - 2,
        h_12 = h - 12,
        h_14 = h - 14,
        geomean = b.geomean
    )
}

fn architecture_svg() -> String {
    #[derive(Clone, Copy)]
    enum Kind {
        Normal,
        Focal,
        Optional,
    }
    let nodes = [
        (
            64,
            128,
            "Weights",
            "GGUF · safetensors",
            "ONNX · pickle · MLX",
            Kind::Normal,
        ),
        (
            288,
            128,
            "Loaders",
            "hub · family builders",
            "re-encode + KL gate",
            Kind::Normal,
        ),
        (
            512,
            128,
            "Graph IR",
            "SSA · weights by name",
            "src/graph",
            Kind::Normal,
        ),
        (
            736,
            128,
            "Passes",
            "decompose · dtype-adapt",
            "fusion · megakernel",
            Kind::Normal,
        ),
        (
            960,
            128,
            "Metal executor",
            "generated MSL GEMV",
            "tuned · arena-planned",
            Kind::Focal,
        ),
        (
            512,
            272,
            "Core ML → ANE",
            "System 1 encoder",
            "--backend ane",
            Kind::Optional,
        ),
        (
            736,
            272,
            "CPU executor",
            "parity oracle",
            "--backend cpu",
            Kind::Optional,
        ),
        (
            288,
            416,
            "Surfaces",
            "OpenAI · Anthropic",
            "Ollama · TUI · CLI",
            Kind::Normal,
        ),
        (
            512,
            416,
            "Scheduler",
            "KV admission",
            "prefill budget",
            Kind::Normal,
        ),
    ];
    let mut boxes = String::new();
    for (x, y, name, s1, s2, kind) in nodes {
        let (stroke, sw, dash, nfill) = match kind {
            Kind::Normal => (HAIRLINE, "1", "", PARCHMENT),
            Kind::Focal => (BRASS, "1.5", "", BRASS),
            Kind::Optional => (HAIRLINE, "1", r#" stroke-dasharray="4 3""#, PARCHMENT),
        };
        let _ = writeln!(
            boxes,
            r#"<rect x="{x}" y="{y}" width="176" height="80" rx="6" fill="{VOID}"/><rect x="{x}" y="{y}" width="176" height="80" rx="6" fill="{PLINTH}" stroke="{stroke}" stroke-width="{sw}"{dash}/><text x="{tx}" y="{ny}" font-size="14" fill="{nfill}">{name}</text><text x="{tx}" y="{s1y}" font-size="11" fill="{DIM}">{s1}</text><text x="{tx}" y="{s2y}" font-size="11" fill="{DIM}">{s2}</text>"#,
            tx = x + 12,
            ny = y + 28,
            s1y = y + 52,
            s2y = y + 68,
            name = esc(name),
            s1 = esc(s1),
            s2 = esc(s2)
        );
    }
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 560" width="1200" height="560" role="img" aria-labelledby="gibbs-arch-title gibbs-arch-desc">
<title id="gibbs-arch-title">Gibbs architecture</title>
<desc id="gibbs-arch-desc">Weight files are loaded and re-encoded into one graph IR, optimization passes fuse it, and the Metal executor runs it by default; the CPU executor (parity oracle) and Core ML on the Neural Engine (System 1) are opt-in placements; HTTP, TUI and CLI requests reach the Metal executor through the scheduler.</desc>
<defs>
<marker id="ar" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{DIM}"/></marker>
<marker id="ar-brass" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{BRASS}"/></marker>
</defs>
<style>text{{font-family:{MONO}}}.serif{{font-family:{SERIF}}}</style>
<rect width="1200" height="560" rx="14" fill="{VOID}"/>
<rect x="1" y="1" width="1198" height="558" rx="13" fill="none" stroke="{HAIRLINE}" stroke-width="2"/>
<rect x="6" y="6" width="1188" height="548" rx="9" fill="none" stroke="{HAIRLINE}" stroke-width="1"/>
<g fill="{BRASS}"><rect x="10" y="10" width="4" height="4"/><rect x="1186" y="10" width="4" height="4"/><rect x="10" y="546" width="4" height="4"/><rect x="1186" y="546" width="4" height="4"/></g>
<text x="64" y="56" font-size="12" fill="{DIM}" letter-spacing="0.12em">ARCHITECTURE</text>
<text class="serif" x="64" y="92" font-size="28" fill="{PARCHMENT}">One graph IR, lowered to Metal</text>
<g fill="none" stroke="{DIM}" stroke-width="1.2" marker-end="url(#ar)">
<path d="M240 168 H288"/>
<path d="M464 168 H512"/>
<path d="M688 168 H736"/>
<path d="M852 208 V272"/>
<path d="M796 208 V224 Q796 232 788 232 H608 Q600 232 600 240 V272"/>
<path d="M464 456 H512"/>
<path d="M688 456 H1040 Q1048 456 1048 448 V208"/>
</g>
<path d="M912 168 H960" fill="none" stroke="{BRASS}" stroke-width="1.5" marker-end="url(#ar-brass)"/>
<rect x="704" y="432" width="112" height="14" fill="{VOID}"/>
<text x="760" y="444" font-size="11" fill="{DIM}" letter-spacing="0.08em" text-anchor="middle">RUNS PROGRAMS</text>
{boxes}<rect x="64" y="508" width="1072" height="1" fill="{HAIRLINE}"/>
<rect x="64" y="522" width="24" height="12" rx="2" fill="{PLINTH}" stroke="{BRASS}" stroke-width="1.5"/>
<text x="96" y="532" font-size="11" fill="{DIM}">DEFAULT PATH</text>
<rect x="280" y="522" width="24" height="12" rx="2" fill="{PLINTH}" stroke="{HAIRLINE}" stroke-width="1" stroke-dasharray="4 3"/>
<text x="312" y="532" font-size="11" fill="{DIM}">OPT-IN PLACEMENT</text>
<path d="M520 528 H552" fill="none" stroke="{DIM}" stroke-width="1.2" marker-end="url(#ar)"/>
<text x="564" y="532" font-size="11" fill="{DIM}">DATA / CONTROL FLOW</text>
</svg>
"##
    )
}

fn bench_block(b: &Bench) -> String {
    let mut s = String::from(
        "<!-- gibbs-bench:begin -->\n\
         | Model | Decode tok/s | % of roofline | Roofline tok/s | GPU ms/token | Host ms/token | Dispatches/token |\n\
         | :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for r in &b.ok {
        let gpu = r.gpu_ms.map_or_else(|| "—".into(), |v| format!("{v:.2}"));
        let host = r.host_ms.map_or_else(|| "—".into(), |v| format!("{v:.2}"));
        let disp = r
            .dispatches
            .map_or_else(|| "—".into(), |v| format!("{v:.0}"));
        let _ = writeln!(
            s,
            "| {} | {:.1} | {:.1}% | {:.0} | {gpu} | {host} | {disp} |",
            r.label, r.tok_s, r.pct, r.roofline_tok_s
        );
    }
    let _ = writeln!(
        s,
        "| **Geomean** | | **{:.1}%** | | | | |\n\n\
         {}, Metal, measured streaming ceiling {:.1} GB/s. Binary built from `{}` ({}); `gibbs --bench bench/loop.json`: {} decode tokens, median of {} runs, `--reencode {}`; raw rows in [`autoresearch/bench-readme.jsonl`](autoresearch/bench-readme.jsonl). Roofline = ceiling ÷ bytes one decode step streams (`bytes_basis` in each row). {} of {} models reach the 90% roofline target; Gemma 4 E2B QAT decodes at {:.1} tok/s against its 150 tok/s target.",
        b.geomean,
        b.device,
        b.ceiling,
        b.commit,
        b.date,
        b.decode_tokens,
        b.runs,
        b.reencode,
        b.k,
        b.ok.len(),
        b.e2b.tok_s
    );
    for r in b.rows.iter().filter(|r| r.error.is_some()) {
        let _ = writeln!(
            s,
            "\nNot measured: {} ({})",
            r.label,
            r.error.as_deref().unwrap_or("")
        );
    }
    if let Some(note) = &b.note {
        let _ = writeln!(s, "\n> {note}");
    }
    s.push_str("<!-- gibbs-bench:end -->");
    s
}

fn write_if(path: &str, body: &str) {
    fs::write(path, body).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn main() {
    let bytes = fs::read("assets/art/source/gibbs-1895.jpg")
        .expect("assets/art/source/gibbs-1895.jpg (see PROVENANCE.md)");
    assert!(bytes.starts_with(&[0xFF, 0xD8]), "source is not a JPEG");
    let img = ImageBuffer::decode(&bytes).expect("decode source");
    let (lum, cw, ch) = crop_lum(&img);
    let tui = build(&lum, cw, ch, TUI);
    let hi = build(&lum, cw, ch, HI);
    let b = bench();

    write_if("assets/art/gibbs-sprite.txt", &sprite_txt(&tui));
    let ascii = ascii_lines(&tui);
    write_if("assets/art/gibbs-ascii.txt", &(ascii.join("\n") + "\n"));
    write_if("assets/gibbs-portrait.svg", &portrait_svg(&hi));
    write_if("assets/gibbs-hero.svg", &hero_svg(&hi, &b));
    write_if("assets/gibbs-architecture.svg", &architecture_svg());
    write_if("assets/gibbs-roofline.svg", &roofline_svg(&b));
    for f in [
        "assets/gibbs-1873-animated.svg",
        "assets/maxwell-surface-banner.svg",
    ] {
        let _ = fs::remove_file(f);
    }

    let readme = fs::read_to_string("README.md").expect("README.md");
    let (begin, end) = ("<!-- gibbs-bench:begin -->", "<!-- gibbs-bench:end -->");
    let i = readme.find(begin).expect("gibbs-bench begin marker");
    let j = readme.find(end).expect("gibbs-bench end marker") + end.len();
    write_if(
        "README.md",
        &format!("{}{}{}", &readme[..i], bench_block(&b), &readme[j..]),
    );
}
