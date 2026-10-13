# Gibbs Design System v2
*Warm brass on obsidian, with a cold cyanotype data channel. The colour is the measurement.*

Scope: the terminal UI (`src/tui/`), the README/web art (`assets/`), and the generator that produces all of it (`examples/gibbs_art.rs`). Palette values below are the single source for `src/tui/theme.rs` (`PALETTE`) and the generator ramps.

---

## 0. Reference study

Method: measured 2026-10-12, headless Chromium `getComputedStyle`, 1440×900.

| | Nous Research (`nousresearch.com`) | Extropic (`extropic.ai`) |
|:---|:---|:---|
| Ground | page frame `#0000F2`, inner canvas `#0D0D24`, text `#F2F2F2`, accent `#8080FF` | `#1C0101` (theme-color), text `#E5E7EB`/`#FFFFFF` |
| Accents | electric blue only | amber `#D97706`, gold `#F5D54C`, rust `#903001` |
| Type | ultra-condensed grotesque display (`Rules Gothic Compressed`) at hero scale, uppercase; `Aeonik Fono` body; uppercase micro-nav | `kessler` display (high-contrast serif + extended sans pairing), `Suisse Mono` for all UI at 16px, tracked uppercase wordmark `E X T R O P I C` |
| Imagery | blue halftone/dither art, engraved portrait cameo as logo, keycap chips (`⌘ K`) | macro hardware photography, warm bokeh, left-gold-rule announcement chips |
| Voice | `NODES → …` navigation, hairline frames | indexed technical nav, whitepaper density |

Re-verified this sprint with the same method: Nous `body` background `rgb(0,0,242)` and text `rgb(242,242,242)`, display face `Rules Gothic Compressed`; Extropic `body` background `rgb(28,1,1)`, `theme-color` `#1C0101`.

**Positioning (decided).** Neither cobalt-dither nor gold-on-oxblood. Gibbs is **warm brass on obsidian, with a cold cyanotype data channel** (Herschel 1842 blueprint blue: a 19th-century process that reads as both Victorian and space-age blueprint). Three differentiators:

1. Every panel title carries its live headline metric.
2. The portrait is the real c.1895 public-domain Gibbs photograph, ordered-dithered to a brass ramp (engraving/halftone register, authentic where competitors use illustration).
3. Warm = compute, cold = memory bandwidth, jade = at roofline, so the colour itself is a roofline reading.

**Booklet observations** (Nous brand booklet, first edition, 5 Feb 2024; rendered page by page in the browser tool; the PDF is image-only):
- Pages read are strictly black on white; imagery is 1-bit dither/halftone (sun disc with stippled edge, patch art, pixel-OS window frames for the *Nous-Hermes* and *Nous Forge* UIs).
- Type: ultra-condensed display wordmark `NOUS`; body in `Courier New` or the in-house `Courier Nous`; `Helvetica` for supporting text; `Mondwest` as a serif specimen; specimens shown in three-column text grids with regular, italic, bold and underlined variants.
- Marks: a `NOUS` wordmark inside a rounded-rectangle frame with a portrait ("the Nous girl"); a "Nous + sun" logo as the standard format; mission patches.
- Page furniture: tiny right-aligned footer caption `TOWARDS SELF-IMPROVEMENT`; uppercase typewriter captions.
- One page is a grid of blue swatches without readable labels or values; no palette names or hex values were legible, so none are recorded.

---

## 1. Principles

1. **Instrument, not app.** Dense, labelled, calibrated; no decorative chrome.
2. **Every row carries a live number.** An empty row is a bug.
3. **Warm = compute, cold = bandwidth, jade = at roofline.** Colour is a reading, not a theme.
4. **Engraved, not glossy.** Double rules, no gradients in the TUI.
5. **Motion encodes state, never decoration.** Spinners, ember and smoke change only when the engine's state changes; reduced-motion shows a static frame.

---

## 2. Palette

Contrast ratios computed by WCAG relative luminance against VOID `#14100D` (and PLINTH `#1B1510` where noted); `src/tui/theme.rs` tests enforce them.

| Token | Role | Hex | ANSI-256 | vs VOID | vs PLINTH | Min |
|:---|:---|:---|:---|:---|:---|:---|
| VOID | frame background (truecolor only) | `#14100D` | 233 | — | — | — |
| PLINTH | panel background | `#1B1510` | 234 | — | — | — |
| HAIRLINE | borders, rules | `#7A6040` | 95 | 3.22 | 3.08 | 3.0 (UI) |
| DIM | secondary text, timestamps | `#A8895F` | 137 | 5.78 | 5.52 | 4.5 |
| PARCHMENT | primary text | `#F5EEDB` | 230 | 16.34 | 15.62 | 4.5 |
| BRASS | primary accent, focus border, titles | `#D4AF37` | 178 | 9.00 | 8.60 | 4.5 |
| GOLD | secondary accent, prefill | `#E5C158` | 179 | 10.90 | 10.42 | 4.5 |
| EMBER | decode activity, 30–60% efficiency | `#FF8C00` | 208 | 8.11 | 7.75 | 4.5 |
| JADE | at roofline (≥ 90%), cache hit | `#6B9E78` | 72 | 6.12 | 5.84 | 4.5 |
| CRIMSON | error, < 30% efficiency | `#E2605A` | 167 | 5.46 | 5.21 | 4.5 |
| CYANOTYPE | memory bandwidth, cold data | `#4A86C5` | 68 | 4.96 | 4.74 | 4.5 |

v1 values changed where AA failed as text: `MAHOGANY_DIM #8C6D46` (3.96:1) was split into HAIRLINE (borders only) and DIM; `SPINODAL_CRIMSON #C84646` (3.99:1) became CRIMSON `#E2605A`. `#8C6D46` survives only as the Idle sprite ramp midpoint, and `#C84646` as the Error ramp midpoint (art, not text).

**Semantic map.**
- Efficiency tier: JADE ≥ 90, GOLD 60–90, EMBER 30–60, CRIMSON < 30 (percent of the bandwidth roofline).
- State pills: Equilibrium JADE, Decode EMBER, Prefill GOLD, Loading CYANOTYPE, Error CRIMSON.
- Series: compute BRASS, bandwidth CYANOTYPE.
- Colour modes: truecolor uses the hex; 256-colour uses the ANSI column; `NO_COLOR` set to a non-empty value renders without any colour (mono).

---

## 3. TUI grammar

**Tiers by terminal size.**

| Tier | Width | Layout |
|:---|:---|:---|
| Compact | < 100, or height < 20 | no sidebar; a bottom dock (`min(8, h/3)` rows) shows each component's one-line form |
| Standard | 100–159 | one sidebar flow column, width `clamp(w·36/100, 40, 64)` |
| Wide | ≥ 160 | two flow columns, total `clamp(w·40/100, 84, 120)`, each `(total−1)/2` |

Floor: 80×24 renders without panic; below 20 rows only the dock/status remain.

**Panel anatomy.** `╔═╡ TITLE ╞══════ metric ═╗`: Double border in HAIRLINE (BRASS when focused), PLINTH background under truecolor, title BRASS bold uppercase, the live headline metric right-aligned in PARCHMENT.

**Glyph set.**
- Gauges: eighth-block `▏▎▍▌▋▊▉█` on a `░` track.
- Sparklines: `▁▂▃▄▅▆▇█`; curves: braille.
- Key chips: `⟦^R⟧` (BRASS bold) with a DIM label.
- Orbit spinner: `◜◠◝◞◡◟`.
- State glyphs: `● ◇ ⚡ ▲`.

**Density rules.** No empty row; numerics right-aligned with fixed units; truncate with `…`; drop fields right-to-left when narrow.

**Greek-key rule** `╔╝╚╗` repeated: only in the README hero and the load screen.

**Focus.** `Ctrl+R/P/T/L` focus Roofline/Logprobs/Tokens/Latency and expand them to a `detail` form; `Esc` unfocuses; `Left/Right` scrubs logprobs when focused. There are no modal inspectors.

---

## 4. Art pipeline

Generator: `cargo run --release --example gibbs_art` (deterministic; no randomness; output is byte-identical between runs).

- **Source**: `assets/art/source/gibbs-1895.jpg`, Wikimedia Commons `File:JWGibbs.jpg`, J. Willard Gibbs c. 1895, public domain (published in the US before 1931; subject died 1903). Provenance in `assets/art/source/PROVENANCE.md`.
- **Crop**: normalized rect `(x0,y0,x1,y1) = (0.06, 0.04, 0.98, 0.87)`.
- **Tone**: luminance `0.299R+0.587G+0.114B`; auto-levels map the 2nd..98th percentile to 0..255.
- **Vignette** (crop-normalized): centre `(0.49, 0.42)`, radii `(0.60, 0.62)`, `r = sqrt(((x−cx)/rx)²+((y−cy)/ry)²)`, `lum *= 1 − smoothstep(0.65, 1.10, r)`.
- **Downscale**: box filter to `HI = 96×120` (SVG) and `TUI = 48×60`.
- **Dither**: 8×8 Bayer matrix `B`; `v = lum/255·7; base = floor(v); level = base + (v−base > (B[y&7][x&7]+0.5)/64)`, clamp 0..7. Level 0 is transparent/void.
- **Ramp per state**: levels 1..7 interpolate in sRGB with `t = (level−1)/6` through SHADOW `#1A130C` (t=0), the state MID (t=0.5), HIGHLIGHT `#F5EEDB` (t=1).

| State | MID |
|:---|:---|
| Idle | `#8C6D46` |
| Thinking | `#D4AF37` |
| Generating | `#FF8C00` |
| Error | `#C84646` |
| Equilibrium | `#6B9E78` |

- **Pipe overlay** (source-normalized): mouth/stem start `(0.250, 0.430)`, stem end `(0.140, 0.470)`, bowl centre `(0.120, 0.485)`, bowl radius `0.030·source_width`; ember pixel at `(0.120, 0.455)`. Stem and bowl are class `p`, ember class `e`. In the SVG the pipe is painted SHADOW `#1A130C` (the photograph background is light; DIM `#A8895F` disappears against it).
- **Smoke**: 6 frames, 5 puffs; puff `k` at frame `f`: `t=(f+1.2k) mod 6`, `y = ember_y − t·0.055·H`, `x = ember_x + 0.012·W·sin(0.9t + 1.7k)`, 1 px (2×2 when `t ≥ 3`), level `5 − floor(t/2)`. The ember is drawn over the smoke.
- **Timings**: ember 2.4 s, 3 frames (`#FF8C00`, `#FF5500`, `#FFC040`), `steps(1,end)`; smoke 4.8 s, 6 frames (one visible per 0.8 s); scanline 6 s linear; roofline dash 3 s. Reduced motion = static frame 0.
- **Outputs**: `assets/art/gibbs-sprite.txt` (TUI source of truth), `assets/art/gibbs-ascii.txt` (half-block text, ink when level ≥ 4), `assets/gibbs-portrait.svg`, `assets/gibbs-hero.svg`, and the README ASCII block between the `gibbs-ascii` markers. `assets/gibbs-hero-static.png` is produced separately (§6).

Sprite file: `GIBBS-SPRITE 1`, `size 48 60`, `rows`, 60 lines of 48 chars from `01234567pe`, then `smoke <f> x,y x,y …` for f = 0..5.

---

## 5. Components

Each component has `compact` (the rows below) and `detail` (adds rows from the former modal content: full roofline curve, logit histogram, latency histogram) used when focused by `Ctrl+R/P/T/L`; `Esc` unfocuses; `Left/Right` scrubs logprobs when focused. Inner width `W`; fields drop right-to-left when narrower; every metric colour comes from `Theme`.

|Component (title · right-title metric)|Rows (compact)|
|:---|:---|
|STATUS (1 row, no border)|`[ GIBBS ]` BRASS-on-VOID pill · model (PARCHMENT bold, ≤ 40% width, `trunc`) · `{gpu name} · Metal` · `{fmt} {bpw:.1} bpw` · `{tps:.1} tok/s` BRASS · `ROOF {pct:.0}%` + `gauge(pct/100, 8)` coloured by `tier` · `KV {pct}%` · `SEQ {running}/{max}` · `:{port}` · state pill right-aligned: `● EQUILIBRIUM` JADE, `⚡ DECODE` EMBER, `◇ PREFILL` GOLD, `{orbit} LOADING {pct}%` CYANOTYPE, `▲ SPINODAL` CRIMSON. Drop order when narrow: port, SEQ, KV, gauge, format, device.|
|ROOFLINE · `{pct:.1}%`|1 `{achieved:.1}/{roof:.1} tok/s` + `gauge(pct/100, W−24)`; 2 `BW {eff:.1}/{ceil:.1} GB/s` CYANOTYPE + `{bytes_per_token_gb:.3} GB/tok`; 3 `AI {ai:.1} F/B  ridge {ridge:.1}  ▸ {BANDWIDTH|COMPUTE}-BOUND`; 4–5 ratatui `Canvas` with `Marker::Braille`: log-log roofline (BRASS compute roof, CYANOTYPE bandwidth slope) and the operating point `●` in the tier colour; 6 (only with `/profile on`) top-3 kernels `{name} {µs} µs {pct:.0}%` else `profile off · /profile on`. `ai = flops_per_run / bytes_per_token`; other values from `telemetry.get_summary()`, `bytes_per_token()` and the measured `gemm_peak`.|
|TOKENS · `{tps:.1}/s`|1 `NOW {tps:.1}  AVG {avg:.1}  PEAK {peak:.1} tok/s`; 2 `spark(self.spark, W)` BRASS; 3 `Σ {total} tok · {req} req · TTFT {p50:.0}/{p90:.0} ms`; 4 (when speculative) `SPEC {accept:.0}% · {tok_per_fwd:.2} tok/fwd`.|
|LATENCY · `p99 {p99:.1} ms`|1 `ITL p50 {p50:.1}  p95 {p95:.1}  p99 {p99:.1} ms`; 2 `spark` of 16-bucket ITL histogram (`telemetry.recent_itl_ms(512)`) CYANOTYPE; 3 `σ {sd:.2} ms · n={n}`.|
|LOGPROBS · `H {entropy:.2} bits`|1 `'{tok}'  p {p:.2}`; 2 entropy strip over `logprob_history` (64): glyph height by entropy, colour JADE < 1 bit, GOLD < 3, EMBER < 5, CRIMSON ≥ 5; 3–5 top-3 candidates `gauge(p, 10)  {p:.2}  '{tok}'`.|
|WEIGHTS · `{fmt} {bpw:.2} bpw`|1 `stacked_bar` of resident bytes per codec; 2 legend `■ Q4_0 61% ■ TQ2_0 23% …` (colour per codec, ≤ 6 shown); 3 `reencode {policy} · variant {policy}`; 4 gate `KL {kl:.3} nats · top-1 {t:.0}%` or `bit-exact`; 5–7 (detail) per-trial GEMV table from `w.trials`.|
|MEMORY · `KV {pct:.0}%`|1 `KV gauge(pct, W−10)`; 2 `RSS {rss:.1} GB` (`getrusage`) · `blocks {used}/{total}`.|
|DEVICE · `{ceil:.0} GB/s`|1 `{gpu name}` (+ `· {cores}c` only if the host profile exposes the GPU core count); 2 `ceiling {ceil:.1} GB/s` + ` · codec-bench {done}/{n} {measuring…|age}` once the codec bench exists, else ` · codec-bench n/a`; 3 `http {addr}`.|
|LOAD · `{pct:.0}%`|1 `{id}` + `{rate} MB/s · ETA {eta}`; 2 track `RESOLVE ▸ DOWNLOAD ▸ MAP ▸ COMPILE` (done JADE `✓`, current BRASS bold `[ ]`, rest DIM); 3 `gauge(pct, W−14)` + `{done}/{total} MB`; 4 `reencode {done}/{total} tensors` while re-encoding.|
|AVATAR · state name|cameo from the sprite engine + one ticker line (state equation / `Flow ⚡`).|

**Flow layout.** Fixed specs (min/pref/max height, priority): Load 7/7/7/0 (only while loading), Roofline 6/8/12/1, Tokens 5/6/8/2, Latency 5/6/8/3, Logprobs 6/8/12/4, Weights 5/7/12/5, Avatar 12/16/30/6 (only when the column is ≥ 24 wide and the area is ≥ 44 rows), Memory 4/4/6/7, Device 3/3/4/8. Components are stable-sorted by priority and placed at `min_h`; what does not fit goes to the next column, then is hidden; spare rows go to `pref_h`, then `max_h`. A focused component takes priority 0 with `pref_h = min(area.h, 24)`.

**Main column** (not sidebar): STATUS / REQUESTS (min 5, pref 8) / CONSOLE (flex) / LOG (`h ≥ 40 ? 5 : 3`) / KEYS (chips). In the Compact tier the dock shows each component's one-line form horizontally, e.g. `ROOF 91% ▕████████▏ · TOK 187 · p99 7.1 ms · KV 12%`.

**Mascot.** Rendered from the sprite with `▀` half-block cells (foreground = top pixel, background = bottom pixel), scaled with a box filter to the available area; below 12 px wide it falls back to the mini badge. Pipe DIM; ember cycles `#FF8C00,#FF5500,#FFC040` every 4 ticks (every 2 while Generating); smoke frame `(tick/4) % 6` at level 4.

---

## 6. README and web assets

- **Hero**: `assets/gibbs-hero.svg` (viewBox `0 0 1200 440`): VOID rounded rect with a Double hairline frame (outer 2 px HAIRLINE, inner 1 px at 6 px inset), four 4×4 BRASS corner rivets; the dithered portrait (288×360 at `translate(40,40)`) with ember, smoke and a 288×2 BRASS scanline (opacity 0.18); caption in 11 px mono DIM; `GIBBS` wordmark in a high-contrast serif stack (`'Bodoni 72','Didot','Playfair Display','Bodoni MT',Georgia,'Times New Roman',serif`) with a BRASS→PARCHMENT gradient; subtitle `THERMODYNAMIC INFERENCE ENGINE` in tracked mono; four governing lines (labels DIM, values BRASS); a decorative 1 px CYANOTYPE roofline polyline; a faint (8% opacity) isometric wireframe `h(i,j)=0.6·sin(0.35i)·cos(0.3j)+0.012·(i−12)²` on a 24×24 grid, projected `x'=(i−j)·14, y'=(i+j)·7−h·40` (illustrative; labelled so in `<desc>`).
- **Accessibility of the SVG**: `role="img"`, `aria-labelledby="gibbs-hero-title gibbs-hero-desc"`; keyframes use opacity, transform and stroke-dashoffset only (no `filter`); `@media (prefers-reduced-motion: reduce){.anim{animation:none!important}}`; the un-animated default state shows frame 0 of every layer.
- **Static fallback**: `assets/gibbs-hero-static.png`, rasterized with `rsvg-convert -w 2400 assets/gibbs-hero.svg -o assets/gibbs-hero-static.png` (not a build dependency); the README `<picture>` serves it under `prefers-reduced-motion: reduce`.
- **Terminal ASCII block**: `assets/art/gibbs-ascii.txt` framed by a Double box (width `art_width + 4`) between `<!-- gibbs-ascii:begin -->` and `<!-- gibbs-ascii:end -->` in `README.md`.
- **TUI screenshot**: `assets/tui-snapshot.png`, from the ignored test `snapshot_tui` (`GIBBS_TUI_SNAPSHOT=1`).
- Wordmark font is a stack, not outlined glyphs; non-Apple viewers see Georgia/Times.

---

## 7. Accessibility

- Contrast: every text token meets 4.5:1 against VOID and PLINTH; HAIRLINE meets 3.0:1 (UI borders only). Table in §2.
- `NO_COLOR` set to a non-empty value: no colour at all; hierarchy by bold/underline/reverse and glyph shape.
- Reduced motion: README serves the static frame; the SVG disables all animation; the TUI mascot renders a fixed frame.
- State is never colour-only: every state pill has a glyph (`● ⚡ ◇ ▲`) and a word.
- Floor: 80×24 renders every essential field (status, console, keys, dock).
