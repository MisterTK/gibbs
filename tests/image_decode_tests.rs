//! PNG/JPEG decoder tests for `ImageBuffer::decode`.
//! PNGs are built in-test (fixed-Huffman deflate encoder below); dynamic-Huffman PNGs and all
//! JPEGs come from system tools at runtime (`sips`, `ffmpeg`, `cjpeg`, `jpegtran`, `djpeg`);
//! a missing tool prints `SKIP: <reason>` and the test returns.

use gibbs_engine::model::image::{adler32, base64_encode, crc32, ImageBuffer};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SIPS: &str = "/usr/bin/sips";
const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";
const CJPEG: &str = "/opt/homebrew/bin/cjpeg";
const JPEGTRAN: &str = "/opt/homebrew/bin/jpegtran";
const DJPEG: &str = "/opt/homebrew/bin/djpeg";

// ------------------------------------------------------------------ helpers

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("gibbs-imgdec-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn missing(tools: &[&str]) -> bool {
    if let Some(t) = tools.iter().find(|t| !Path::new(t).exists()) {
        println!("SKIP: {t} not installed");
        return true;
    }
    false
}

fn run(tool: &str, args: &[&str]) {
    let o = Command::new(tool).args(args).output().unwrap();
    assert!(
        o.status.success(),
        "{tool} {args:?} failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// Deterministic pseudo-random sequence (LCG).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// LSB-first deflate bit writer.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}
impl BitWriter {
    fn put(&mut self, v: u32, n: u32) {
        self.acc |= (v as u64) << self.n;
        self.n += n;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// Huffman codes are packed MSB-first.
    fn code(&mut self, c: u32, len: u32) {
        self.put(c.reverse_bits() >> (32 - len), len);
    }
    fn lit(&mut self, s: u32) {
        let (c, l) = match s {
            0..=143 => (0x30 + s, 8),
            144..=255 => (0x190 + s - 144, 9),
            256..=279 => (s - 256, 7),
            _ => (0xC0 + s - 280, 8),
        };
        self.code(c, l);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

const LEN_BASE: [u32; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// zlib stream with one fixed-Huffman (BTYPE=01) block: literals plus greedy back-references
/// (distances up to 300, so overlapping copies and multi-extra-bit codes are exercised).
fn zlib_fixed(data: &[u8]) -> Vec<u8> {
    let mut bw = BitWriter {
        out: vec![0x78, 0x01],
        acc: 0,
        n: 0,
    };
    bw.put(1, 1);
    bw.put(1, 2);
    let mut i = 0;
    while i < data.len() {
        let (mut best_len, mut best_dist) = (0, 0);
        for dist in 1..=i.min(300) {
            let len = (0..258.min(data.len() - i))
                .take_while(|&k| data[i + k] == data[i + k - dist])
                .count();
            if len > best_len {
                (best_len, best_dist) = (len, dist);
            }
        }
        if best_len < 3 {
            bw.lit(data[i] as u32);
            i += 1;
            continue;
        }
        let li = LEN_BASE
            .iter()
            .rposition(|&b| b <= best_len as u32)
            .unwrap();
        bw.lit(257 + li as u32);
        bw.put(best_len as u32 - LEN_BASE[li], LEN_EXTRA[li]);
        let di = DIST_BASE
            .iter()
            .rposition(|&b| b <= best_dist as u32)
            .unwrap();
        bw.code(di as u32, 5);
        bw.put(best_dist as u32 - DIST_BASE[di], DIST_EXTRA[di]);
        i += best_len;
    }
    bw.lit(256);
    let mut out = bw.finish();
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn chunk(png: &mut Vec<u8>, ty: &[u8], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(ty);
    png.extend_from_slice(data);
    let crc = crc32(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}

const ADAM7: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

fn channels(color: u8) -> usize {
    [1, 0, 3, 1, 2, 0, 4][color as usize]
}

/// Filtered, packed scanlines (all passes) for `samples` (`w*h*channels`, each < 2^depth).
/// `filter = None` cycles filter types 0..4 row by row.
fn png_raw(
    w: usize,
    h: usize,
    color: u8,
    depth: usize,
    interlace: bool,
    filter: Option<u8>,
    samples: &[u16],
) -> Vec<u8> {
    let ch = channels(color);
    let bpp = (ch * depth).div_ceil(8);
    let passes: &[_] = if interlace { &ADAM7 } else { &[(0, 0, 1, 1)] };
    let mut raw = Vec::new();
    let mut row_no = 0;
    for &(x0, y0, dx, dy) in passes {
        let prev_rows: Vec<Vec<u8>> = (y0..h)
            .step_by(dy)
            .map(|y| {
                let mut bits = Vec::new(); // MSB-first packing
                for x in (x0..w).step_by(dx) {
                    for c in 0..ch {
                        let v = samples[(y * w + x) * ch + c];
                        for b in (0..depth).rev() {
                            bits.push((v >> b) & 1);
                        }
                    }
                }
                bits.chunks(8)
                    .map(|b| {
                        b.iter()
                            .enumerate()
                            .fold(0u8, |a, (i, &bit)| a | (bit as u8) << (7 - i))
                    })
                    .collect()
            })
            .collect();
        if x0 >= w {
            continue;
        }
        for (r, line) in prev_rows.iter().enumerate() {
            let zero = vec![0u8; line.len()];
            let up = if r > 0 { &prev_rows[r - 1] } else { &zero };
            let ft = filter.unwrap_or((row_no % 5) as u8);
            row_no += 1;
            raw.push(ft);
            for i in 0..line.len() {
                let a = if i >= bpp { line[i - bpp] as i16 } else { 0 };
                let (b, c) = (up[i] as i16, if i >= bpp { up[i - bpp] as i16 } else { 0 });
                let pred = match ft {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => (a + b) / 2,
                    _ => {
                        let p = a + b - c;
                        let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                        if pa <= pb && pa <= pc {
                            a
                        } else if pb <= pc {
                            b
                        } else {
                            c
                        }
                    }
                };
                raw.push(line[i].wrapping_sub(pred as u8));
            }
        }
    }
    raw
}

fn png_file(
    w: u32,
    h: u32,
    color: u8,
    depth: u8,
    interlace: bool,
    plte: &[u8],
    zlib: &[u8],
    idat_split: usize,
) -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = [w.to_be_bytes(), h.to_be_bytes()].concat();
    ihdr.extend_from_slice(&[depth, color, 0, 0, interlace as u8]);
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"tEXt", b"Comment\0ancillary chunk");
    if !plte.is_empty() {
        chunk(&mut png, b"PLTE", plte);
        chunk(&mut png, b"tRNS", &[0, 128]);
    }
    for part in zlib.chunks(idat_split) {
        chunk(&mut png, b"IDAT", part);
    }
    chunk(&mut png, b"IEND", &[]);
    png
}

/// Expected RGB per PIL `convert("RGB")`: 16-bit -> high byte, low-bit gray scaled to 0..255,
/// palette lookup, alpha dropped.
fn expected_rgb(color: u8, depth: usize, samples: &[u16], plte: &[u8]) -> Vec<u8> {
    let to8 = |v: u16| match depth {
        16 => (v >> 8) as u8,
        8 => v as u8,
        d => (v as u32 * 255 / ((1 << d) - 1)) as u8,
    };
    samples
        .chunks(channels(color))
        .flat_map(|s| match color {
            0 | 4 => [to8(s[0]); 3],
            3 => [
                plte[s[0] as usize * 3],
                plte[s[0] as usize * 3 + 1],
                plte[s[0] as usize * 3 + 2],
            ],
            _ => [to8(s[0]), to8(s[1]), to8(s[2])],
        })
        .collect()
}

/// Smooth-ish samples (gradients + noise) so filters and back-references both matter.
fn png_samples(w: usize, h: usize, ch: usize, depth: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    let max = (1u32 << depth) - 1;
    (0..w * h * ch)
        .map(|i| {
            let (x, y, c) = ((i / ch) % w, (i / ch) / w, i % ch);
            let smooth = (x * 37 + y * 23 + c * 50) as u32 * max / 600;
            let v = if rng.next() % 4 == 0 {
                rng.next()
            } else {
                smooth
            };
            (v % (max + 1)) as u16
        })
        .collect()
}

fn small_png() -> (Vec<u8>, Vec<u8>) {
    let samples = png_samples(13, 11, 3, 8, 7);
    let raw = png_raw(13, 11, 2, 8, true, None, &samples);
    (
        png_file(13, 11, 2, 8, true, &[], &zlib_fixed(&raw), 64),
        expected_rgb(2, 8, &samples, &[]),
    )
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

/// Reads a binary PPM/PGM (djpeg output) as RGB.
fn read_pnm(path: &Path) -> (usize, usize, Vec<u8>) {
    let d = std::fs::read(path).unwrap();
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while d[i].is_ascii_whitespace() {
            i += 1;
        }
        let s = i;
        while !d[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&d[s..i]).to_string());
    }
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    let px = &d[i + 1..];
    let rgb = if fields[0] == "P5" {
        px.iter().flat_map(|&g| [g; 3]).collect()
    } else {
        px.to_vec()
    };
    (w, h, rgb)
}

/// Decodes `jpg` with libjpeg-turbo's djpeg (what PIL uses) and returns the max abs channel diff.
fn djpeg_max_diff(dir: &Path, jpg: &Path, ours: &ImageBuffer) -> Option<u8> {
    if !Path::new(DJPEG).exists() {
        return None;
    }
    let out = dir.join("djpeg.pnm");
    run(DJPEG, &["-pnm", "-outfile", p(&out), p(jpg)]);
    let (w, h, rgb) = read_pnm(&out);
    assert_eq!((w, h), (ours.width, ours.height));
    Some(
        rgb.iter()
            .zip(&ours.data)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap(),
    )
}

fn decode_file(path: &Path) -> ImageBuffer {
    ImageBuffer::decode(&std::fs::read(path).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// 97x61 smooth gradient with soft-edged shapes (odd size -> partial MCUs).
fn jpeg_source() -> ImageBuffer {
    let mut img = ImageBuffer::new(97, 61);
    for y in 0..61 {
        for x in 0..97 {
            let (fx, fy) = (x as f32, y as f32);
            let d = ((fx - 40.0).powi(2) + (fy - 30.0).powi(2)).sqrt();
            let disk = (1.0 - ((d - 15.0) / 3.0).clamp(0.0, 1.0)) * 90.0;
            let bar = if (70..85).contains(&x) { 60.0 } else { 0.0 };
            let r = (fx * 2.2 + disk).min(255.0);
            let g = (fy * 3.5 + bar).min(255.0);
            let b = (128.0 + 60.0 * (fx / 9.0).sin() * (fy / 7.0).cos()).clamp(0.0, 255.0);
            img.set_pixel(x, y, r as u8, g as u8, b as u8);
        }
    }
    img
}

fn luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks(3)
        .flat_map(|c| {
            [(0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32).round() as u8; 3]
        })
        .collect()
}

// ------------------------------------------------------------------ PNG

#[test]
fn png_all_color_types_depths_filters_interlace() {
    let combos: [(u8, &[usize]); 5] = [
        (0, &[1, 2, 4, 8, 16]),
        (2, &[8, 16]),
        (3, &[1, 2, 4, 8]),
        (4, &[8, 16]),
        (6, &[8, 16]),
    ];
    let mut cases = 0;
    for (color, depths) in combos {
        for &depth in depths {
            for (w, h) in [(13, 11), (1, 1), (5, 3), (9, 17)] {
                for filter in [Some(0), Some(1), Some(2), Some(3), Some(4), None] {
                    for interlace in [false, true] {
                        let seed = (color as u64) << 32 | (depth as u64) << 16 | (w * h) as u64;
                        let samples = png_samples(
                            w,
                            h,
                            channels(color),
                            if color == 3 { depth.min(4) } else { depth },
                            seed,
                        );
                        let plte: Vec<u8> = if color == 3 {
                            (0..(1usize << depth) * 3)
                                .map(|i| (i * 89 % 256) as u8)
                                .collect()
                        } else {
                            vec![]
                        };
                        let raw = png_raw(w, h, color, depth, interlace, filter, &samples);
                        let png = png_file(
                            w as u32,
                            h as u32,
                            color,
                            depth as u8,
                            interlace,
                            &plte,
                            &zlib_fixed(&raw),
                            5 + cases % 50,
                        );
                        let img = ImageBuffer::decode(&png)
                            .unwrap_or_else(|e| panic!("color {color} depth {depth} {w}x{h} filter {filter:?} interlace {interlace}: {e}"));
                        assert_eq!((img.width, img.height, img.channels), (w, h, 3));
                        assert_eq!(
                            img.data,
                            expected_rgb(color, depth, &samples, &plte),
                            "color {color} depth {depth} {w}x{h} filter {filter:?} interlace {interlace}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    println!("decoded {cases} PNG variants exactly");
}

#[test]
fn png_stored_roundtrip_and_base64() {
    let img = jpeg_source();
    assert_eq!(ImageBuffer::decode(&img.to_png()).unwrap(), img);
    assert_eq!(
        ImageBuffer::from_base64_or_path(&img.to_base64_png()).unwrap(),
        img
    );
}

#[test]
fn png_dynamic_huffman_from_sips() {
    if missing(&[SIPS]) {
        return;
    }
    let dir = tmp_dir("dyn");
    let mut src = ImageBuffer::new(64, 48);
    let mut rng = Lcg(3);
    for (i, v) in src.data.iter_mut().enumerate() {
        *v = if rng.next() % 3 == 0 {
            rng.next() as u8
        } else {
            (i % 192 + i / 192) as u8
        };
    }
    let (inp, out) = (dir.join("src.png"), dir.join("dyn.png"));
    std::fs::write(&inp, src.to_png()).unwrap();
    run(SIPS, &["-s", "format", "png", p(&inp), "--out", p(&out)]);
    let bytes = std::fs::read(&out).unwrap();
    let idat = bytes.windows(4).position(|w| w == b"IDAT").unwrap();
    assert_eq!(
        (bytes[idat + 6] >> 1) & 3,
        2,
        "sips output should start with a dynamic-Huffman block"
    );
    assert_eq!(decode_file(&out).data, src.data);
}

// ------------------------------------------------------------------ JPEG

#[test]
fn jpeg_variants_match_source_and_libjpeg() {
    if missing(&[SIPS, FFMPEG, CJPEG, JPEGTRAN]) {
        return;
    }
    let dir = tmp_dir("jpeg");
    let src = jpeg_source();
    let (png, ppm) = (dir.join("src.png"), dir.join("src.ppm"));
    std::fs::write(&png, src.to_png()).unwrap();
    std::fs::write(
        &ppm,
        [format!("P6\n97 61\n255\n").as_bytes(), &src.data].concat(),
    )
    .unwrap();
    let j = |name: &str| dir.join(name);

    run(
        SIPS,
        &[
            "-s",
            "format",
            "jpeg",
            "-s",
            "formatOptions",
            "90",
            p(&png),
            "--out",
            p(&j("base.jpg")),
        ],
    );
    for fmt in ["yuvj444p", "yuvj422p", "yuvj420p"] {
        run(
            FFMPEG,
            &[
                "-y",
                "-loglevel",
                "error",
                "-i",
                p(&png),
                "-pix_fmt",
                fmt,
                "-q:v",
                "2",
                p(&j(&format!("{fmt}.jpg"))),
            ],
        );
    }
    // ffmpeg's mjpeg encoder has no 4:1:1, 4:4:0 or gray output: cjpeg (libjpeg-turbo) covers those.
    for (name, args) in [
        ("cjpeg411.jpg", &["-sample", "4x1"][..]),
        ("cjpeg440.jpg", &["-sample", "1x2"]),
        ("gray.jpg", &["-grayscale"]),
        ("cjpeg420prog.jpg", &["-sample", "2x2", "-progressive"]),
    ] {
        run(
            CJPEG,
            &[
                &["-quality", "95"][..],
                args,
                &["-outfile", p(&j(name)), p(&ppm)],
            ]
            .concat(),
        );
    }
    run(
        JPEGTRAN,
        &[
            "-progressive",
            "-outfile",
            p(&j("prog.jpg")),
            p(&j("base.jpg")),
        ],
    );
    run(
        JPEGTRAN,
        &[
            "-restart",
            "1",
            "-outfile",
            p(&j("rst.jpg")),
            p(&j("base.jpg")),
        ],
    );
    run(
        JPEGTRAN,
        &[
            "-restart",
            "1B",
            "-outfile",
            p(&j("rst1b.jpg")),
            p(&j("base.jpg")),
        ],
    );

    let base = decode_file(&j("base.jpg"));
    for name in [
        "base.jpg",
        "yuvj444p.jpg",
        "yuvj422p.jpg",
        "yuvj420p.jpg",
        "cjpeg411.jpg",
        "cjpeg440.jpg",
        "gray.jpg",
        "cjpeg420prog.jpg",
        "prog.jpg",
        "rst.jpg",
        "rst1b.jpg",
    ] {
        let img = decode_file(&j(name));
        assert_eq!((img.width, img.height), (97, 61), "{name}");
        let reference = if name == "gray.jpg" {
            luma(&src.data)
        } else {
            src.data.clone()
        };
        let db = psnr(&img.data, &reference);
        let lib = djpeg_max_diff(&dir, &j(name), &img);
        println!("{name:18} PSNR {db:5.2} dB  max |ours - djpeg| = {lib:?}");
        assert!(db > 30.0, "{name}: PSNR {db:.2} dB");
        if let Some(diff) = lib {
            assert_eq!(
                diff, 0,
                "{name}: differs from libjpeg-turbo (PIL's decoder)"
            );
        }
        if matches!(name, "prog.jpg" | "rst.jpg" | "rst1b.jpg") {
            assert!(
                img.data == base.data,
                "{name}: lossless transcode must decode identically to baseline"
            );
        }
    }

    let b64 = base64_encode(&std::fs::read(j("base.jpg")).unwrap());
    assert!(b64.starts_with("/9j/"));
    assert_eq!(ImageBuffer::from_base64_or_path(&b64).unwrap(), base);
    assert_eq!(
        ImageBuffer::from_base64_or_path(&format!("data:image/jpeg;base64,{b64}")).unwrap(),
        base
    );
}

/// Photos in `GIBBS_VLM_IMAGES` (`cat`/`dog`/`eiffel`/`bananas` as `.jpg` plus `.png` conversions).
#[test]
fn jpeg_real_photos() {
    let photos = ["cat", "dog", "eiffel", "bananas"];
    let Some(root) = std::env::var_os("GIBBS_VLM_IMAGES").map(std::path::PathBuf::from) else {
        println!("SKIP: GIBBS_VLM_IMAGES not set");
        return;
    };
    let root = root.as_path();
    let dir = tmp_dir("real");
    for name in photos {
        let jpg = root.join(format!("{name}.jpg"));
        if !jpg.exists() || !root.join(format!("{name}.png")).exists() {
            println!("SKIP: {name}.jpg/.png not in {}", root.display());
            continue;
        }
        let bytes = std::fs::read(&jpg).unwrap();
        let t = Instant::now();
        let img = ImageBuffer::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let base_ms = t.elapsed().as_secs_f64() * 1e3;
        let lib = djpeg_max_diff(&dir, &jpg, &img);
        print!(
            "{name:8} {}x{} baseline {base_ms:7.1} ms  max |ours - djpeg| = {lib:?}",
            img.width, img.height
        );
        if let Some(diff) = lib {
            assert_eq!(
                diff, 0,
                "{name}: differs from libjpeg-turbo (PIL's decoder)"
            );
        }

        if Path::new(JPEGTRAN).exists() {
            let prog = dir.join(format!("{name}-prog.jpg"));
            run(JPEGTRAN, &["-progressive", "-outfile", p(&prog), p(&jpg)]);
            let pbytes = std::fs::read(&prog).unwrap();
            let t = Instant::now();
            let pimg =
                ImageBuffer::decode(&pbytes).unwrap_or_else(|e| panic!("{name} progressive: {e}"));
            print!("  progressive {:7.1} ms", t.elapsed().as_secs_f64() * 1e3);
            assert!(
                pimg == img,
                "{name}: progressive transcode must decode identically"
            );
        }
        println!();

        // The PNGs are resized conversions of the same photos: compare aspect ratio and mean colour.
        let png = decode_file(&root.join(format!("{name}.png")));
        let aspect = |i: &ImageBuffer| i.width as f64 / i.height as f64;
        assert!(
            (aspect(&png) / aspect(&img) - 1.0).abs() < 0.02,
            "{name}: aspect mismatch"
        );
        let mean = |i: &ImageBuffer, c: usize| {
            i.data
                .iter()
                .skip(c)
                .step_by(3)
                .map(|&v| v as f64)
                .sum::<f64>()
                / (i.width * i.height) as f64
        };
        for c in 0..3 {
            assert!(
                (mean(&png, c) - mean(&img, c)).abs() < 8.0,
                "{name}: channel {c} mean differs"
            );
        }
    }
}

// ------------------------------------------------------------------ malformed input

#[test]
fn png_malformed_inputs_are_errors() {
    let (png, rgb) = small_png();
    assert_eq!(ImageBuffer::decode(&png).unwrap().data, rgb);
    let idat = png.windows(4).position(|w| w == b"IDAT").unwrap();
    assert!(
        ImageBuffer::decode(&png[..idat + 20]).is_err(),
        "truncated mid-IDAT"
    );

    let samples = png_samples(13, 11, 3, 8, 7);
    let raw = png_raw(13, 11, 2, 8, false, Some(1), &samples);
    let build = |zlib: Vec<u8>| png_file(13, 11, 2, 8, false, &[], &zlib, 1 << 20);
    let err = |png: Vec<u8>| ImageBuffer::decode(&png).unwrap_err();
    let mut bad_adler = zlib_fixed(&raw);
    *bad_adler.last_mut().unwrap() ^= 1;
    assert!(err(build(bad_adler)).contains("Adler-32"));

    // Fixed-Huffman stream using reserved length symbol 286, distance code 30, or BTYPE=11.
    for case in 0..3 {
        let mut bw = BitWriter {
            out: vec![0x78, 0x01],
            acc: 0,
            n: 0,
        };
        bw.put(1, 1);
        bw.put(if case == 2 { 3 } else { 1 }, 2);
        bw.lit(65);
        if case == 0 {
            bw.lit(286);
        } else {
            bw.lit(257);
            bw.code(30, 5);
        }
        bw.lit(256);
        let e = err(build(bw.finish()));
        assert!(e.contains("deflate"), "case {case}: {e}");
    }

    let mut bad_filter = raw.clone();
    bad_filter[0] = 5;
    assert!(err(build(zlib_fixed(&bad_filter))).contains("filter"));
    assert!(
        ImageBuffer::decode(&build(zlib_fixed(&raw[..raw.len() - 10]))).is_err(),
        "short image data"
    );

    for (w, h) in [(0, 11), (13, 0), (100_000, 1), (16384, 16384)] {
        let png = png_file(w, h, 2, 8, false, &[], &zlib_fixed(&raw), 1 << 20);
        assert!(ImageBuffer::decode(&png).is_err(), "{w}x{h}");
    }
    let bad_depth = png_file(13, 11, 2, 4, false, &[], &zlib_fixed(&raw), 1 << 20);
    assert!(
        ImageBuffer::decode(&bad_depth).is_err(),
        "RGB at 4 bits is invalid"
    );
    let no_plte = png_file(13, 11, 3, 8, false, &[], &zlib_fixed(&raw), 1 << 20);
    assert!(
        ImageBuffer::decode(&no_plte).is_err(),
        "palette image without PLTE"
    );
    assert!(ImageBuffer::decode(b"GIF89a\x01\0\x01\0\0\0")
        .unwrap_err()
        .contains("GIF"));
    assert!(ImageBuffer::decode(b"RIFF\x10\0\0\0WEBPVP8 ")
        .unwrap_err()
        .contains("WebP"));
}

/// A tiny real JPEG (sips baseline) for malformed-input checks.
fn small_jpeg(dir: &Path) -> Option<Vec<u8>> {
    if missing(&[SIPS]) {
        return None;
    }
    let mut src = ImageBuffer::new(21, 13);
    for (i, v) in src.data.iter_mut().enumerate() {
        *v = (i * 7 % 256) as u8;
    }
    let (png, jpg) = (dir.join("small.png"), dir.join("small.jpg"));
    std::fs::write(&png, src.to_png()).unwrap();
    run(SIPS, &["-s", "format", "jpeg", p(&png), "--out", p(&jpg)]);
    Some(std::fs::read(jpg).unwrap())
}

#[test]
fn jpeg_malformed_inputs_are_errors() {
    let dir = tmp_dir("jpeg-bad");
    let Some(jpg) = small_jpeg(&dir) else { return };
    assert!(ImageBuffer::decode(&jpg).is_ok());
    let sos = jpg.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
    let sof = jpg.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
    let scan_len = jpg.len() - sos;
    assert!(
        ImageBuffer::decode(&jpg[..sos + scan_len / 2]).is_err(),
        "truncated mid-scan"
    );
    assert!(
        ImageBuffer::decode(&jpg[..jpg.len() - 2]).is_err(),
        "missing EOI"
    );
    assert!(
        ImageBuffer::decode(&[&jpg[..sos], &[0xFF, 0xD9][..]].concat()).is_err(),
        "SOF without SOS"
    );

    let with_dims = |h: u16, w: u16| {
        let mut b = jpg.clone();
        b[sof + 5..sof + 7].copy_from_slice(&h.to_be_bytes());
        b[sof + 7..sof + 9].copy_from_slice(&w.to_be_bytes());
        b
    };
    for (h, w) in [(0, 21), (13, 0), (65535, 65535), (8000, 8000)] {
        assert!(ImageBuffer::decode(&with_dims(h, w)).is_err(), "{w}x{h}");
    }
    let mut cmyk = jpg.clone();
    cmyk[sof + 9] = 4;
    assert!(ImageBuffer::decode(&cmyk).unwrap_err().contains("CMYK"));
    let mut lossless = jpg.clone();
    lossless[sof + 1] = 0xC3;
    assert!(ImageBuffer::decode(&lossless).unwrap_err().contains("SOF3"));
    let mut arithmetic = jpg.clone();
    arithmetic[sof + 1] = 0xC9;
    assert!(ImageBuffer::decode(&arithmetic)
        .unwrap_err()
        .contains("arithmetic"));
    let mut twelve = jpg.clone();
    twelve[sof + 4] = 12;
    assert!(ImageBuffer::decode(&twelve).unwrap_err().contains("12-bit"));
    // A component whose quantization table no DQT defined (libjpeg: "Quantization table not
    // defined") is an error, not a flat gray plane.
    let mut no_dqt = jpg.clone();
    no_dqt[sof + 12] = 3;
    assert!(ImageBuffer::decode(&no_dqt)
        .unwrap_err()
        .contains("quantization table 3"));
    // Scan work is bounded: EOB runs let a scan visit every block without consuming data, so
    // unbounded repeated scans would hang on a large frame.
    let eoi = jpg.len() - 2;
    let scan = &jpg[sos..eoi];
    let many = [&jpg[..eoi], &scan.repeat(1000), &jpg[eoi..]].concat();
    assert!(ImageBuffer::decode(&many)
        .unwrap_err()
        .contains("work limit"));
    if missing(&[CJPEG]) {
        return;
    }
    // The same on a 16 MP progressive frame whose last scan (an EOB-run refinement over every
    // block) repeats 1000 times in ~140 KB: Err within the limit's few passes, not minutes.
    let (w, h) = (4000, 4000);
    let pgm = dir.join("flat.pgm");
    let mut flat = format!("P5\n{w} {h}\n255\n").into_bytes();
    flat.resize(flat.len() + w * h, 128);
    std::fs::write(&pgm, flat).unwrap();
    let prog = dir.join("flat.jpg");
    run(
        CJPEG,
        &["-grayscale", "-progressive", "-outfile", p(&prog), p(&pgm)],
    );
    let jpg = std::fs::read(&prog).unwrap();
    assert!(ImageBuffer::decode(&jpg).is_ok());
    let (sos, eoi) = (
        jpg.windows(2).rposition(|w| w == [0xFF, 0xDA]).unwrap(),
        jpg.len() - 2,
    );
    let bomb = [&jpg[..eoi], &jpg[sos..eoi].repeat(1000), &jpg[eoi..]].concat();
    let started = Instant::now();
    assert!(ImageBuffer::decode(&bomb)
        .unwrap_err()
        .contains("work limit"));
    assert!(started.elapsed().as_secs() < 3, "{:?}", started.elapsed());
}

#[test]
fn fuzz_truncation_and_bit_flips_never_panic() {
    let dir = tmp_dir("fuzz");
    let (png, _) = small_png();
    let mut inputs = vec![png];
    inputs.extend(small_jpeg(&dir));
    if Path::new(JPEGTRAN).exists() && inputs.len() == 2 {
        let (base, prog) = (dir.join("small.jpg"), dir.join("small-prog.jpg"));
        run(
            JPEGTRAN,
            &[
                "-progressive",
                "-restart",
                "1B",
                "-outfile",
                p(&prog),
                p(&base),
            ],
        );
        inputs.push(std::fs::read(prog).unwrap());
    }
    let mut rng = Lcg(11);
    for data in &inputs {
        for len in 0..data.len() {
            let _ = ImageBuffer::decode(&data[..len]);
        }
        for _ in 0..400 {
            let mut d = data.clone();
            for _ in 0..1 + rng.next() % 3 {
                let i = rng.next() as usize % d.len();
                d[i] ^= 1 << (rng.next() % 8);
            }
            let _ = ImageBuffer::decode(&d);
        }
    }
    println!("fuzzed {} inputs", inputs.len());
}
