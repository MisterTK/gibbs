//! Image buffer with a dependency-free PNG encoder, PNG/JPEG decoders and base64 helpers.
//! Decoding matches PIL `Image.open(..).convert("RGB")`: PNG (all color types, bit depths,
//! Adam7; alpha dropped, 16-bit -> high byte) and JPEG (baseline/extended/progressive Huffman,
//! 1 or 3 components, libjpeg-turbo islow IDCT + fancy upsampling, bit-exact with `djpeg`).
//! Real diffusion lives in `crate::model::sd::SdPipeline`.

use std::fs;
use std::io::Write as _;
use std::path::Path;

/// Standard RFC 4648 Base64 character table
const B64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes raw byte slice into standard Base64 string without heap thrashing.
pub fn base64_encode(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(B64_TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_TABLE[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decodes Base64 string into raw bytes.
pub fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let clean: Vec<u8> = input
        .bytes()
        .filter(|&b| !b.is_ascii_whitespace() && b != b'=')
        .collect();

    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    let decode_val = |b: u8| -> Result<u32, String> {
        match b {
            b'A'..=b'Z' => Ok((b - b'A') as u32),
            b'a'..=b'z' => Ok((b - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((b - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("Invalid base64 character: {}", b as char)),
        }
    };

    for chunk in clean.chunks(4) {
        let mut accum = 0u32;
        for &b in chunk {
            accum = (accum << 6) | decode_val(b)?;
        }
        match chunk.len() {
            4 => {
                out.push(((accum >> 16) & 0xFF) as u8);
                out.push(((accum >> 8) & 0xFF) as u8);
                out.push((accum & 0xFF) as u8);
            }
            3 => {
                out.push(((accum >> 10) & 0xFF) as u8);
                out.push(((accum >> 2) & 0xFF) as u8);
            }
            2 => {
                out.push(((accum >> 4) & 0xFF) as u8);
            }
            _ => return Err("Invalid base64 chunk length".to_string()),
        }
    }
    Ok(out)
}

/// Computes Adler-32 checksum according to RFC 1950.
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for &b in data {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    (s2 << 16) | s1
}

/// Computes standard IEEE 802.3 CRC-32 checksum.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// In-memory 24-bit RGB Image Buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

impl ImageBuffer {
    /// Creates a black RGB image buffer of given width and height.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            channels: 3,
            data: vec![0u8; width * height * 3],
        }
    }

    /// Creates an RGB image buffer from raw RGB bytes.
    pub fn from_rgb(width: usize, height: usize, data: Vec<u8>) -> Result<Self, String> {
        if data.len() != width * height * 3 {
            return Err(format!(
                "Invalid RGB buffer size: expected {} ({}x{}x3), got {}",
                width * height * 3,
                width,
                height,
                data.len()
            ));
        }
        Ok(Self {
            width,
            height,
            channels: 3,
            data,
        })
    }

    /// Resizes to `width`×`height` like PIL `Image.resize(..., BICUBIC)` (what most HF image
    /// processors use): separable cubic (a = -0.5).
    pub fn resize(&self, width: usize, height: usize) -> Self {
        let cubic = |x: f64| {
            let (a, x) = (-0.5, x.abs());
            if x < 1.0 {
                ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            } else if x < 2.0 {
                (((x - 5.0) * x + 8.0) * x - 4.0) * a
            } else {
                0.0
            }
        };
        self.resample(width, height, &cubic, 2.0)
    }

    /// Resizes like PIL `Image.resize(..., LANCZOS)` (Idefics3/SmolVLM processors): 3-lobe
    /// windowed sinc.
    pub fn resize_lanczos(&self, width: usize, height: usize) -> Self {
        let sinc = |x: f64| {
            if x == 0.0 {
                1.0
            } else {
                (x * std::f64::consts::PI).sin() / (x * std::f64::consts::PI)
            }
        };
        self.resample(
            width,
            height,
            &|x| {
                if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            },
            3.0,
        )
    }

    /// PIL's separable resampling with `kernel` of radius `support`: support widened by the
    /// scale factor when shrinking (antialias), horizontal pass then vertical, each rounded to u8.
    fn resample(
        &self,
        width: usize,
        height: usize,
        kernel: &dyn Fn(f64) -> f64,
        support: f64,
    ) -> Self {
        // Per output index along one axis: (first source index, normalized weights).
        let taps = |src: usize, dst: usize| -> Vec<(usize, Vec<f32>)> {
            let scale = src as f64 / dst as f64;
            let fscale = scale.max(1.0);
            let support = support * fscale;
            (0..dst)
                .map(|i| {
                    let center = (i as f64 + 0.5) * scale;
                    let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
                    let hi = ((center + support + 0.5).floor() as usize).min(src);
                    let w: Vec<f64> = (lo..hi)
                        .map(|j| kernel((j as f64 - center + 0.5) / fscale))
                        .collect();
                    let sum: f64 = w.iter().sum();
                    (lo, w.iter().map(|v| (v / sum) as f32).collect())
                })
                .collect()
        };
        let px = |v: f32| v.round().clamp(0.0, 255.0) as u8;
        let (w0, h0) = (self.width, self.height);
        let mut mid = vec![0u8; width * h0 * 3];
        let tx = taps(w0, width);
        for y in 0..h0 {
            for (x, (lo, w)) in tx.iter().enumerate() {
                for c in 0..3 {
                    let s: f32 = w
                        .iter()
                        .enumerate()
                        .map(|(k, wk)| wk * self.data[(y * w0 + lo + k) * 3 + c] as f32)
                        .sum();
                    mid[(y * width + x) * 3 + c] = px(s);
                }
            }
        }
        let mut out = vec![0u8; width * height * 3];
        for (y, (lo, w)) in taps(h0, height).iter().enumerate() {
            for x in 0..width {
                for c in 0..3 {
                    let s: f32 = w
                        .iter()
                        .enumerate()
                        .map(|(k, wk)| wk * mid[((lo + k) * width + x) * 3 + c] as f32)
                        .sum();
                    out[(y * width + x) * 3 + c] = px(s);
                }
            }
        }
        Self {
            width,
            height,
            channels: 3,
            data: out,
        }
    }

    /// Sets pixel RGB value.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 3;
            self.data[idx] = r;
            self.data[idx + 1] = g;
            self.data[idx + 2] = b;
        }
    }

    /// Gets pixel RGB value.
    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> (u8, u8, u8) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 3;
            (self.data[idx], self.data[idx + 1], self.data[idx + 2])
        } else {
            (0, 0, 0)
        }
    }

    /// Encodes image to valid, standards-compliant PNG bytes.
    ///
    /// Uses uncompressed Deflate (BTYPE=00) IDAT chunks, avoiding external
    /// compression dependencies while generating native PNGs instantly readable
    /// by web browsers, macOS Preview, and ComfyUI.
    pub fn to_png(&self) -> Vec<u8> {
        let w = self.width as u32;
        let h = self.height as u32;

        // 1. Uncompressed scanlines: 1 byte filter (0x00: None) + row RGB bytes
        let row_bytes = self.width * 3;
        let scanline_size = 1 + row_bytes;
        let mut raw_scanlines = Vec::with_capacity(self.height * scanline_size);

        for y in 0..self.height {
            raw_scanlines.push(0x00); // Filter: None
            let row_start = y * row_bytes;
            raw_scanlines.extend_from_slice(&self.data[row_start..row_start + row_bytes]);
        }

        // 2. Build zlib Deflate stream
        let mut deflate_stream = Vec::new();
        deflate_stream.push(0x78); // CMF (Deflate, 32K window)
        deflate_stream.push(0x01); // FLG (check bits)

        // Split raw scanlines into non-compressed blocks (max 65535 bytes each)
        let max_block = 65535;
        let total_raw = raw_scanlines.len();
        let mut offset = 0;

        while offset < total_raw {
            let end = (offset + max_block).min(total_raw);
            let chunk = &raw_scanlines[offset..end];
            let is_last = end == total_raw;

            deflate_stream.push(if is_last { 0x01 } else { 0x00 });
            let len = chunk.len() as u16;
            let nlen = !len;

            deflate_stream.extend_from_slice(&len.to_le_bytes());
            deflate_stream.extend_from_slice(&nlen.to_le_bytes());
            deflate_stream.extend_from_slice(chunk);

            offset = end;
        }

        // Adler-32 of raw scanlines (RFC 1950)
        let adler = adler32(&raw_scanlines);
        deflate_stream.extend_from_slice(&adler.to_be_bytes());

        // 3. Assemble PNG Chunks
        let mut png = Vec::with_capacity(8 + 25 + deflate_stream.len() + 12 + 12);
        // Signature
        png.extend_from_slice(b"\x89PNG\r\n\x1a\n");

        // IHDR Chunk
        let mut ihdr_data = Vec::with_capacity(13);
        ihdr_data.extend_from_slice(&w.to_be_bytes());
        ihdr_data.extend_from_slice(&h.to_be_bytes());
        ihdr_data.push(8); // Bit depth: 8
        ihdr_data.push(2); // Color type: 2 (Truecolor RGB)
        ihdr_data.push(0); // Compression method: Deflate
        ihdr_data.push(0); // Filter method: Standard
        ihdr_data.push(0); // Interlace: None

        png.extend_from_slice(&(ihdr_data.len() as u32).to_be_bytes());
        let ihdr_crc_start = png.len();
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&ihdr_data);
        let ihdr_crc = crc32(&png[ihdr_crc_start..]);
        png.extend_from_slice(&ihdr_crc.to_be_bytes());

        // IDAT Chunk
        png.extend_from_slice(&(deflate_stream.len() as u32).to_be_bytes());
        let idat_crc_start = png.len();
        png.extend_from_slice(b"IDAT");
        png.extend_from_slice(&deflate_stream);
        let idat_crc = crc32(&png[idat_crc_start..]);
        png.extend_from_slice(&idat_crc.to_be_bytes());

        // IEND Chunk
        png.extend_from_slice(&0u32.to_be_bytes());
        let iend_crc_start = png.len();
        png.extend_from_slice(b"IEND");
        let iend_crc = crc32(&png[iend_crc_start..]);
        png.extend_from_slice(&iend_crc.to_be_bytes());

        png
    }

    /// Encodes image directly into a Base64-encoded PNG string.
    pub fn to_base64_png(&self) -> String {
        base64_encode(&self.to_png())
    }

    /// Decodes PNG or JPEG bytes (by signature) to RGB8, matching PIL `convert("RGB")`:
    /// alpha and tRNS are dropped, 16-bit samples keep their high byte.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            return decode_png(bytes);
        }
        if bytes.starts_with(&[0xFF, 0xD8]) {
            return decode_jpeg(bytes);
        }
        let format = if bytes.starts_with(b"GIF8") {
            "GIF"
        } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            "WebP"
        } else if bytes.starts_with(b"BM") {
            "BMP"
        } else if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
            "TIFF"
        } else {
            return Err("unrecognized image format (PNG and JPEG are read)".into());
        };
        Err(format!(
            "unsupported image format: {format} (PNG and JPEG are read)"
        ))
    }

    /// Parses an image from base64 string, data URL (`data:image/png;base64,...`), or local file path.
    pub fn from_base64_or_path(input: &str) -> Result<Self, String> {
        let trimmed = input.trim();

        // 1. Data URL
        if let Some(pos) = trimmed.find(";base64,") {
            let b64_part = &trimmed[pos + 8..];
            let bytes = base64_decode(b64_part)?;
            return Self::decode(&bytes);
        }

        // 2. Direct Base64 (PNG `iVBORw0KGgo`, JPEG `/9j/`)
        if trimmed.starts_with("iVBORw0KGgo")
            || trimmed.starts_with("/9j/")
            || (!trimmed.contains('/') && !trimmed.contains('.'))
        {
            if let Ok(bytes) = base64_decode(trimmed) {
                if let Ok(img) = Self::decode(&bytes) {
                    return Ok(img);
                }
            }
        }

        // 3. File path
        let path = Path::new(trimmed);
        if path.exists() {
            let bytes = fs::read(path).map_err(|e| e.to_string())?;
            return Self::decode(&bytes);
        }

        // Try raw base64 decode as final fallback
        let bytes = base64_decode(trimmed)?;
        Self::decode(&bytes)
    }
}

// ponytail: per-side and 100 MP caps bound every size computation and allocation on untrusted input.
fn check_dims(w: usize, h: usize) -> Result<(), String> {
    if w == 0 || h == 0 || w > 16384 || h > 16384 || w * h > 100_000_000 {
        return Err(format!("unsupported image dimensions {w}x{h}"));
    }
    Ok(())
}

// ---------------------------------------------------------------- zlib inflate (RFC 1950/1951)

/// Canonical Huffman decoder for deflate: LSB-first codes, full lookup table of
/// `1 << max_len` entries holding `symbol << 4 | len` (len 0 = unused code).
struct Inflate<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

struct Huff {
    table: Vec<u16>,
    mask: u64,
}

impl Huff {
    fn new(lens: &[u8]) -> Result<Self, String> {
        let max = lens.iter().copied().max().unwrap_or(0) as u32;
        let mut count = [0i32; 16];
        for &l in lens {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let (mut next, mut code, mut left) = ([0u32; 16], 0u32, 1i32);
        for l in 1..16 {
            left = (left << 1) - count[l];
            if left < 0 {
                return Err("invalid deflate Huffman lengths (over-subscribed)".into());
            }
            code = (code + count[l - 1] as u32) << 1;
            next[l] = code;
        }
        let mut table = vec![0u16; 1 << max];
        for (sym, &l) in lens.iter().enumerate().filter(|(_, &l)| l > 0) {
            let l = l as u32;
            let c = next[l as usize];
            next[l as usize] += 1;
            let rev = (c.reverse_bits() >> (32 - l)) as usize;
            for e in table[rev..].iter_mut().step_by(1 << l) {
                *e = (sym as u16) << 4 | l as u16;
            }
        }
        Ok(Huff {
            table,
            mask: (1u64 << max) - 1,
        })
    }
}

impl Inflate<'_> {
    fn fill(&mut self) {
        while self.n <= 56 {
            let Some(&b) = self.d.get(self.pos) else {
                break;
            };
            self.acc |= (b as u64) << self.n;
            self.pos += 1;
            self.n += 8;
        }
    }

    fn bits(&mut self, k: u32) -> Result<u32, String> {
        if self.n < k {
            self.fill();
            if self.n < k {
                return Err("truncated zlib stream".into());
            }
        }
        let v = (self.acc & ((1u64 << k) - 1)) as u32;
        self.acc >>= k;
        self.n -= k;
        Ok(v)
    }

    fn sym(&mut self, h: &Huff) -> Result<usize, String> {
        self.fill();
        let e = h.table[(self.acc & h.mask) as usize];
        let l = (e & 15) as u32;
        if l == 0 || l > self.n {
            return Err("invalid or truncated deflate Huffman code".into());
        }
        self.acc >>= l;
        self.n -= l;
        Ok((e >> 4) as usize)
    }

    /// Drops bits up to the byte boundary and un-reads whole buffered bytes.
    fn align(&mut self) {
        self.pos -= (self.n / 8) as usize;
        self.acc = 0;
        self.n = 0;
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Inflates a zlib stream (stored, fixed and dynamic blocks) and verifies Adler-32.
/// Output beyond `limit` bytes is an error, so allocation is bounded by the caller.
fn inflate(d: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    if d.len() < 2
        || d[0] & 0x0F != 8
        || d[1] & 0x20 != 0
        || (d[0] as u16 * 256 + d[1] as u16) % 31 != 0
    {
        return Err("invalid zlib header".into());
    }
    let mut r = Inflate {
        d,
        pos: 2,
        acc: 0,
        n: 0,
    };
    let mut out = Vec::with_capacity(limit.min(d.len().saturating_mul(1032)));
    let too_big = || "zlib data exceeds the image size".to_string();
    loop {
        let last = r.bits(1)?;
        let (lit, dist) = match r.bits(2)? {
            0 => {
                r.align();
                let hdr = d.get(r.pos..r.pos + 4).ok_or("truncated zlib stream")?;
                let len = u16::from_le_bytes([hdr[0], hdr[1]]) as usize;
                if len != !u16::from_le_bytes([hdr[2], hdr[3]]) as usize {
                    return Err("corrupt stored deflate block length".into());
                }
                let body = d
                    .get(r.pos + 4..r.pos + 4 + len)
                    .ok_or("truncated zlib stream")?;
                if out.len() + len > limit {
                    return Err(too_big());
                }
                out.extend_from_slice(body);
                r.pos += 4 + len;
                if last == 1 {
                    break;
                }
                continue;
            }
            1 => {
                let mut l = [8u8; 288];
                l[144..256].fill(9);
                l[256..280].fill(7);
                (Huff::new(&l)?, Huff::new(&[5; 30])?)
            }
            2 => {
                let hlit = r.bits(5)? as usize + 257;
                let hdist = r.bits(5)? as usize + 1;
                let hclen = r.bits(4)? as usize + 4;
                const ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut cl = [0u8; 19];
                for &i in &ORDER[..hclen] {
                    cl[i] = r.bits(3)? as u8;
                }
                let cl = Huff::new(&cl)?;
                let mut lens = Vec::with_capacity(hlit + hdist);
                while lens.len() < hlit + hdist {
                    let (v, n) = match r.sym(&cl)? {
                        s @ 0..=15 => (s as u8, 1),
                        16 => (
                            *lens
                                .last()
                                .ok_or("deflate repeat with no previous length")?,
                            3 + r.bits(2)?,
                        ),
                        17 => (0, 3 + r.bits(3)?),
                        _ => (0, 11 + r.bits(7)?),
                    };
                    lens.extend(std::iter::repeat(v).take(n as usize));
                }
                if lens.len() > hlit + hdist || lens[256] == 0 {
                    return Err("invalid deflate code lengths".into());
                }
                (Huff::new(&lens[..hlit])?, Huff::new(&lens[hlit..])?)
            }
            _ => return Err("invalid deflate block type".into()),
        };
        loop {
            let s = r.sym(&lit)?;
            if s < 256 {
                if out.len() >= limit {
                    return Err(too_big());
                }
                out.push(s as u8);
                continue;
            }
            if s == 256 {
                break;
            }
            let i = s - 257;
            let len = *LEN_BASE.get(i).ok_or("invalid deflate length code")? as usize
                + r.bits(LEN_EXTRA[i] as u32)? as usize;
            let ds = r.sym(&dist)?;
            let dist = *DIST_BASE.get(ds).ok_or("invalid deflate distance code")? as usize
                + r.bits(DIST_EXTRA[ds] as u32)? as usize;
            if dist > out.len() {
                return Err("deflate distance before start of data".into());
            }
            if out.len() + len > limit {
                return Err(too_big());
            }
            let start = out.len() - dist;
            if dist >= len {
                out.extend_from_within(start..start + len);
            } else {
                for i in start..start + len {
                    out.push(out[i]);
                }
            }
        }
        if last == 1 {
            break;
        }
    }
    r.align();
    let sum = d
        .get(r.pos..r.pos + 4)
        .ok_or("truncated zlib stream (missing Adler-32)")?;
    if u32::from_be_bytes([sum[0], sum[1], sum[2], sum[3]]) != adler32(&out) {
        return Err("zlib Adler-32 mismatch".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------- PNG

/// Adam7 passes as (x0, y0, dx, dy).
const ADAM7: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

fn decode_png(bytes: &[u8]) -> Result<ImageBuffer, String> {
    let be32 = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let (mut pos, mut ihdr, mut plte, mut idat) = (8, None, &[][..], Vec::new());
    while pos < bytes.len() {
        let hdr = bytes
            .get(pos..pos + 8)
            .ok_or("truncated PNG chunk header")?;
        let len = be32(hdr);
        let body = bytes
            .get(pos + 8..)
            .and_then(|b| b.get(..len))
            .ok_or("truncated PNG chunk")?;
        match &hdr[4..8] {
            b"IHDR" if len == 13 => ihdr = Some(body),
            b"PLTE" => plte = body,
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        pos += 12 + len;
    }
    let h = ihdr.ok_or("PNG missing IHDR")?;
    let (w, ht, depth, color) = (be32(h), be32(&h[4..]), h[8], h[9]);
    check_dims(w, ht)?;
    let channels = match (color, depth) {
        (0, 1 | 2 | 4 | 8 | 16) | (3, 1 | 2 | 4 | 8) => 1,
        (2, 8 | 16) => 3,
        (4, 8 | 16) => 2,
        (6, 8 | 16) => 4,
        _ => {
            return Err(format!(
                "invalid PNG color type {color} with bit depth {depth}"
            ))
        }
    };
    if h[10] != 0 || h[11] != 0 || h[12] > 1 {
        return Err("unsupported PNG compression, filter or interlace method".into());
    }
    if color == 3 && (plte.is_empty() || plte.len() % 3 != 0) {
        return Err("PNG palette missing or invalid".into());
    }
    let depth = depth as usize;
    let bpp = (channels * depth).div_ceil(8);
    let passes = if h[12] == 1 {
        &ADAM7[..]
    } else {
        &[(0, 0, 1, 1)][..]
    };
    let dims = |&(x0, y0, dx, dy): &(usize, usize, usize, usize)| {
        ((w + dx - 1 - x0) / dx, (ht + dy - 1 - y0) / dy)
    };
    let stride = |pw: usize| (pw * channels * depth).div_ceil(8);
    let raw_len: usize = passes
        .iter()
        .map(|p| match dims(p) {
            (0, _) => 0,
            (pw, ph) => ph * (1 + stride(pw)),
        })
        .sum();
    let mut raw = inflate(&idat, raw_len)?;
    if raw.len() != raw_len {
        return Err("PNG image data is truncated".into());
    }

    let sample = |line: &[u8], i: usize| -> u8 {
        match depth {
            16 => line[2 * i],
            8 => line[i],
            _ => (line[i * depth / 8] >> (8 - depth - i * depth % 8)) & ((1 << depth) - 1),
        }
    };
    let gray_scale = if depth < 8 {
        255 / ((1u8 << depth) - 1)
    } else {
        1
    };
    let mut out = vec![0u8; w * ht * 3];
    let mut off = 0;
    for p @ &(x0, y0, dx, dy) in passes {
        let (pw, ph) = dims(p);
        if pw == 0 || ph == 0 {
            continue;
        }
        let st = stride(pw);
        for r in 0..ph {
            let start = off + r * (st + 1);
            let (before, rest) = raw.split_at_mut(start + 1);
            let line = &mut rest[..st];
            unfilter(
                before[start],
                line,
                (r > 0).then(|| &before[start - st..start]),
                bpp,
            )?;
            for c in 0..pw {
                let px = &mut out[((y0 + r * dy) * w + x0 + c * dx) * 3..][..3];
                let s = c * channels;
                match color {
                    0 | 4 => px.fill(sample(line, s) * gray_scale),
                    3 => {
                        let i = sample(line, s) as usize * 3;
                        px.copy_from_slice(plte.get(i..i + 3).unwrap_or(&[0, 0, 0]));
                    }
                    _ => px.copy_from_slice(&[
                        sample(line, s),
                        sample(line, s + 1),
                        sample(line, s + 2),
                    ]),
                }
            }
        }
        off += ph * (st + 1);
    }
    ImageBuffer::from_rgb(w, ht, out)
}

fn unfilter(ft: u8, line: &mut [u8], prev: Option<&[u8]>, bpp: usize) -> Result<(), String> {
    let up = |i: usize| prev.map_or(0, |p| p[i]);
    for i in 0..line.len() {
        let a = if i >= bpp { line[i - bpp] } else { 0 };
        let pred = match ft {
            0 => return Ok(()),
            1 => a,
            2 => up(i),
            3 => ((a as u16 + up(i) as u16) / 2) as u8,
            4 => {
                let (b, c) = (up(i), if i >= bpp { up(i - bpp) } else { 0 });
                let p = a as i16 + b as i16 - c as i16;
                let (pa, pb, pc) = (
                    (p - a as i16).abs(),
                    (p - b as i16).abs(),
                    (p - c as i16).abs(),
                );
                if pa <= pb && pa <= pc {
                    a
                } else if pb <= pc {
                    b
                } else {
                    c
                }
            }
            _ => return Err(format!("invalid PNG filter type {ft}")),
        };
        line[i] = line[i].wrapping_add(pred);
    }
    Ok(())
}

// ---------------------------------------------------------------- JPEG (baseline, extended, progressive Huffman)

/// Zigzag scan index -> natural (row-major) coefficient index.
const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// JPEG Huffman table: 9-bit lookahead (`len << 8 | symbol`, 0 = longer code),
/// then the spec's maxcode/valptr walk for 10..16-bit codes.
struct JpegHuff {
    lut: [u16; 512],
    maxcode: [i32; 17],
    valoff: [i32; 17],
    vals: Vec<u8>,
}

impl JpegHuff {
    fn new(counts: &[u8], vals: &[u8]) -> Result<Self, String> {
        let mut h = JpegHuff {
            lut: [0; 512],
            maxcode: [-1; 17],
            valoff: [0; 17],
            vals: vals.to_vec(),
        };
        let (mut code, mut k) = (0i32, 0i32);
        for l in 1..=16 {
            let n = counts[l - 1] as i32;
            if code + n > 1 << l {
                return Err("invalid JPEG Huffman table".into());
            }
            h.valoff[l] = k - code;
            if l <= 9 {
                for c in code..code + n {
                    let entry = (l as u16) << 8 | vals[(c + h.valoff[l]) as usize] as u16;
                    h.lut[(c as usize) << (9 - l)..][..1 << (9 - l)].fill(entry);
                }
            }
            code += n;
            k += n;
            if n > 0 {
                h.maxcode[l] = code - 1;
            }
            code <<= 1;
        }
        Ok(h)
    }
}

/// MSB-first entropy reader. Stops at markers and pads with zero bits; `pad`
/// counts padded bits so consuming any of them is reported as truncation.
struct JpegBits<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
    pad: u32,
}

impl JpegBits<'_> {
    fn fill(&mut self) {
        while self.n <= 56 {
            let b = match (self.d.get(self.pos), self.d.get(self.pos + 1)) {
                (Some(0xFF), Some(0)) => {
                    self.pos += 2;
                    0xFF
                }
                (Some(&b), _) if b != 0xFF => {
                    self.pos += 1;
                    b
                }
                _ => {
                    self.pad += 8;
                    0
                }
            };
            self.acc = self.acc << 8 | b as u64;
            self.n += 8;
        }
    }

    fn truncated(&self) -> Result<(), String> {
        if self.n < self.pad {
            return Err("truncated or corrupt JPEG entropy data".into());
        }
        Ok(())
    }

    /// Reads `k <= 16` raw bits.
    fn bits(&mut self, k: u32) -> u32 {
        if self.n < k {
            self.fill();
        }
        self.n -= k;
        ((self.acc >> self.n) & ((1 << k) - 1)) as u32
    }

    /// RECEIVE + EXTEND (spec F.2.2.1): a signed `s`-bit magnitude.
    fn extend(&mut self, s: u32) -> Result<i32, String> {
        if s > 16 {
            return Err("invalid JPEG coefficient size".into());
        }
        let v = self.bits(s) as i32;
        Ok(if s > 0 && v < 1 << (s - 1) {
            v - (1 << s) + 1
        } else {
            v
        })
    }

    fn huff(&mut self, h: &JpegHuff) -> Result<u8, String> {
        if self.n < 16 {
            self.fill();
        }
        let e = h.lut[((self.acc >> (self.n - 9)) & 511) as usize];
        if e != 0 {
            self.n -= (e >> 8) as u32;
            return Ok(e as u8);
        }
        for l in 10..=16 {
            let code = ((self.acc >> (self.n - l)) & ((1 << l) - 1)) as i32;
            if code <= h.maxcode[l as usize] {
                self.n -= l;
                return h
                    .vals
                    .get((code + h.valoff[l as usize]) as usize)
                    .copied()
                    .ok_or_else(|| "bad JPEG Huffman code".into());
            }
        }
        Err("bad JPEG Huffman code".into())
    }

    /// Consumes an RSTn marker at a restart-interval boundary.
    fn restart(&mut self) -> Result<(), String> {
        self.truncated()?;
        (self.acc, self.n, self.pad) = (0, 0, 0);
        let start = self.pos;
        while self.d.get(self.pos) == Some(&0xFF) {
            self.pos += 1;
        }
        match self.d.get(self.pos) {
            Some(0xD0..=0xD7) if self.pos > start => {
                self.pos += 1;
                Ok(())
            }
            _ => Err("missing JPEG restart marker".into()),
        }
    }
}

struct JpegComp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// Blocks per row/column, padded to whole MCUs.
    bw: usize,
    bh: usize,
    /// Quantized coefficients, 64 per block in natural order.
    coefs: Vec<i16>,
    pred: i32,
    /// Scan work spent on each of this component's blocks so far, see [`MAX_SCAN_WORK`].
    work: usize,
}

struct JpegFrame {
    w: usize,
    h: usize,
    progressive: bool,
    hmax: usize,
    vmax: usize,
    comps: Vec<JpegComp>,
}

/// Scan work each component may spend per block: 16 full-band passes at (64 coefficients + 16
/// visit cost) each. EOB runs let a scan visit every block without consuming data, so this (not
/// the file size) bounds decode time on hostile input.
// ponytail: libjpeg's default progressive script spends ~290 per luma block; progressions refining
// every coefficient separately through many bit planes exceed the limit. Raise it if a real
// encoder's output ever does.
const MAX_SCAN_WORK: usize = 16 * (64 + 16);

fn be16(b: &[u8]) -> usize {
    u16::from_be_bytes([b[0], b[1]]) as usize
}

fn decode_jpeg(d: &[u8]) -> Result<ImageBuffer, String> {
    let (mut qt, mut qt_defined) = ([[0u16; 64]; 4], [false; 4]);
    let mut dc: [Option<JpegHuff>; 4] = Default::default();
    let mut ac: [Option<JpegHuff>; 4] = Default::default();
    let (mut frame, mut ri, mut adobe_rgb, mut scans) = (None, 0, false, 0);
    let mut pos = 2;
    loop {
        // Find the next marker, skipping fill bytes and stray entropy-coded data.
        while *d.get(pos).ok_or("truncated JPEG (missing EOI)")? != 0xFF {
            pos += 1;
        }
        while d.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let m = *d.get(pos).ok_or("truncated JPEG (missing EOI)")?;
        pos += 1;
        match m {
            0x00 | 0x01 | 0xD0..=0xD7 => continue,
            0xD9 => break,
            _ => {}
        }
        let len = d
            .get(pos..pos + 2)
            .map(be16)
            .ok_or("truncated JPEG segment")?;
        let seg = d
            .get(pos + 2..pos + len.max(2))
            .filter(|_| len >= 2)
            .ok_or("truncated JPEG segment")?;
        pos += len;
        match m {
            0xC0..=0xC2 if frame.is_some() => return Err("JPEG has more than one frame".into()),
            0xC0..=0xC2 => frame = Some(parse_sof(seg, m == 0xC2, d.len())?),
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(format!(
                    "unsupported JPEG process SOF{} (lossless, hierarchical or arithmetic-coded); baseline, extended and progressive Huffman are read",
                    m - 0xC0
                ))
            }
            0xC4 => {
                let mut s = seg;
                while let Some((&tc, rest)) = s.split_first() {
                    let (class, t) = (tc >> 4, (tc & 15) as usize);
                    let counts = rest.get(..16).filter(|_| class < 2 && t < 4).ok_or("invalid JPEG DHT")?;
                    let total = counts.iter().map(|&c| c as usize).sum::<usize>();
                    let vals = rest.get(16..16 + total).ok_or("invalid JPEG DHT")?;
                    let table = Some(JpegHuff::new(counts, vals)?);
                    if class == 0 { dc[t] = table } else { ac[t] = table }
                    s = &rest[16 + total..];
                }
            }
            0xDB => {
                let mut s = seg;
                while let Some((&pt, rest)) = s.split_first() {
                    let (wide, t) = (pt >> 4 == 1, (pt & 15) as usize);
                    let n = if wide { 128 } else { 64 };
                    if pt >> 4 > 1 || t > 3 || rest.len() < n {
                        return Err("invalid JPEG DQT".into());
                    }
                    for (i, &z) in ZIGZAG.iter().enumerate() {
                        qt[t][z as usize] = if wide { be16(&rest[2 * i..]) as u16 } else { rest[i] as u16 };
                    }
                    qt_defined[t] = true;
                    s = &rest[n..];
                }
            }
            0xDD => ri = seg.get(..2).map(be16).ok_or("invalid JPEG DRI")?,
            0xEE if seg.len() >= 12 && seg.starts_with(b"Adobe") => adobe_rgb = seg[11] == 0,
            0xDA => {
                let f = frame.as_mut().ok_or("JPEG scan before frame header")?;
                pos = decode_scan(d, pos, seg, f, &dc, &ac, ri)?;
                scans += 1;
            }
            _ => {}
        }
    }
    let f = frame.ok_or("JPEG has no frame header")?;
    if scans == 0 {
        return Err("JPEG has no scan data".into());
    }
    if let Some(c) = f.comps.iter().find(|c| !qt_defined[c.tq]) {
        return Err(format!("JPEG quantization table {} not defined", c.tq));
    }

    // IDCT each component, then upsample to full resolution (libjpeg "fancy" for h2v1/h1v2/h2v2).
    let (w, h) = (f.w, f.h);
    let planes: Vec<Vec<u8>> = f
        .comps
        .iter()
        .map(|c| {
            let stride = c.bw * 8;
            let mut p = vec![0u8; stride * c.bh * 8];
            for (i, blk) in c.coefs.chunks_exact(64).enumerate() {
                let (by, bx) = (i / c.bw, i % c.bw);
                idct_islow(blk, &qt[c.tq], &mut p[by * 8 * stride + bx * 8..], stride);
            }
            let (dw, dh) = ((w * c.h).div_ceil(f.hmax), (h * c.v).div_ceil(f.vmax));
            upsample(&p, stride, dw, dh, f.hmax / c.h, f.vmax / c.v, w, h)
        })
        .collect();
    let rgb_ids = f.comps.iter().map(|c| c.id).eq(*b"RGB");
    let mut out = vec![0u8; w * h * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        match &planes[..] {
            [y] => px.fill(y[i]),
            [r, g, b] if adobe_rgb || rgb_ids => px.copy_from_slice(&[r[i], g[i], b[i]]),
            [y, cb, cr] => {
                // JFIF YCbCr -> RGB with libjpeg's 16-bit fixed-point constants.
                let (y, cb, cr) = (y[i] as i32, cb[i] as i32 - 128, cr[i] as i32 - 128);
                let half = 1 << 15;
                let clamp = |v: i32| v.clamp(0, 255) as u8;
                px[0] = clamp(y + ((91881 * cr + half) >> 16));
                px[1] = clamp(y + ((-22554 * cb - 46802 * cr + half) >> 16));
                px[2] = clamp(y + ((116130 * cb + half) >> 16));
            }
            _ => unreachable!("component count is validated in parse_sof"),
        }
    }
    ImageBuffer::from_rgb(w, h, out)
}

fn parse_sof(seg: &[u8], progressive: bool, file_len: usize) -> Result<JpegFrame, String> {
    let hdr = seg.get(..6).ok_or("truncated JPEG SOF")?;
    if hdr[0] != 8 {
        return Err(format!(
            "unsupported JPEG precision {}-bit (8-bit is read)",
            hdr[0]
        ));
    }
    let (h, w, nf) = (be16(&hdr[1..]), be16(&hdr[3..]), hdr[5] as usize);
    match nf {
        1 | 3 => {}
        4 => return Err("unsupported JPEG: 4-component CMYK/YCCK".into()),
        _ => return Err(format!("unsupported JPEG component count {nf}")),
    }
    check_dims(w, h)?;
    // ponytail: every block costs >= 1 entropy bit, so a short file cannot claim huge dimensions
    // and force coefficient allocation (a progressive file omitting its DC scans could trip this).
    if w * h / 64 > file_len * 8 {
        return Err("JPEG data too short for its dimensions".into());
    }
    let spec = seg.get(6..6 + 3 * nf).ok_or("truncated JPEG SOF")?;
    let mut comps = Vec::with_capacity(nf);
    for c in spec.chunks_exact(3) {
        let (ch, cv, tq) = ((c[1] >> 4) as usize, (c[1] & 15) as usize, c[2] as usize);
        if !(1..=4).contains(&ch) || !(1..=4).contains(&cv) || tq > 3 {
            return Err("invalid JPEG component sampling or quantization table".into());
        }
        comps.push(JpegComp {
            id: c[0],
            h: ch,
            v: cv,
            tq,
            bw: 0,
            bh: 0,
            coefs: Vec::new(),
            pred: 0,
            work: 0,
        });
    }
    let hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
    let (mcux, mcuy) = (w.div_ceil(8 * hmax), h.div_ceil(8 * vmax));
    for c in &mut comps {
        if hmax % c.h != 0 || vmax % c.v != 0 {
            return Err("unsupported JPEG fractional sampling factors".into());
        }
        (c.bw, c.bh) = (mcux * c.h, mcuy * c.v);
        c.coefs = vec![0; c.bw * c.bh * 64];
    }
    Ok(JpegFrame {
        w,
        h,
        progressive,
        hmax,
        vmax,
        comps,
    })
}

/// Decodes one scan into the frame's coefficient buffers; returns the position after its entropy data.
fn decode_scan(
    d: &[u8],
    pos: usize,
    seg: &[u8],
    f: &mut JpegFrame,
    dc: &[Option<JpegHuff>; 4],
    ac: &[Option<JpegHuff>; 4],
    ri: usize,
) -> Result<usize, String> {
    let ns = *seg.first().ok_or("invalid JPEG SOS")? as usize;
    let params = seg
        .get(1 + 2 * ns..4 + 2 * ns)
        .filter(|_| (1..=f.comps.len()).contains(&ns));
    let params = params.ok_or("invalid JPEG SOS")?;
    let mut sc = Vec::with_capacity(ns);
    for c in seg[1..1 + 2 * ns].chunks_exact(2) {
        let ci = f
            .comps
            .iter()
            .position(|x| x.id == c[0])
            .ok_or("JPEG scan references unknown component")?;
        let (td, ta) = ((c[1] >> 4) as usize, (c[1] & 15) as usize);
        if td > 3 || ta > 3 {
            return Err("invalid JPEG Huffman table index".into());
        }
        sc.push((ci, td, ta));
    }
    let (ss, se, ah, al) = if f.progressive {
        let (ss, se, ah, al) = (
            params[0] as usize,
            params[1] as usize,
            (params[2] >> 4) as u32,
            (params[2] & 15) as u32,
        );
        if se > 63 || ss > se || (ss == 0) != (se == 0) || (ss > 0 && ns != 1) || ah > 13 || al > 13
        {
            return Err("invalid JPEG progressive scan parameters".into());
        }
        (ss, se, ah, al)
    } else {
        (0, 63, 0, 0)
    };
    fn table(t: &[Option<JpegHuff>; 4], i: usize) -> Result<&JpegHuff, String> {
        t[i].as_ref()
            .ok_or_else(|| "JPEG scan uses an undefined Huffman table".into())
    }
    // Non-interleaved scans cover only the component's real block extent, one block per "MCU".
    let (mx, my) = if ns == 1 {
        let c = &f.comps[sc[0].0];
        (
            (f.w * c.h).div_ceil(f.hmax).div_ceil(8),
            (f.h * c.v).div_ceil(f.vmax).div_ceil(8),
        )
    } else {
        (f.comps[0].bw / f.comps[0].h, f.comps[0].bh / f.comps[0].v)
    };
    for &(ci, ..) in &sc {
        let c = &mut f.comps[ci];
        c.pred = 0;
        // Per block: the band's coefficients plus a fixed visit cost.
        c.work += se - ss + 1 + 16;
        if c.work > MAX_SCAN_WORK {
            return Err(
                "JPEG scans exceed the decode work limit (16 full passes per component)".into(),
            );
        }
    }
    let mut r = JpegBits {
        d,
        pos,
        acc: 0,
        n: 0,
        pad: 0,
    };
    let mut eobrun = 0u32;
    for m in 0..mx * my {
        if ri > 0 && m > 0 && m % ri == 0 {
            r.restart()?;
            eobrun = 0;
            for &(ci, ..) in &sc {
                f.comps[ci].pred = 0;
            }
        }
        let (mxi, myi) = (m % mx, m / mx);
        for &(ci, td, ta) in &sc {
            let c = &mut f.comps[ci];
            let (bh, bv) = if ns == 1 { (1, 1) } else { (c.h, c.v) };
            for by in 0..bv {
                for bx in 0..bh {
                    let off = ((myi * bv + by) * c.bw + mxi * bh + bx) * 64;
                    let blk = &mut c.coefs[off..off + 64];
                    if !f.progressive {
                        decode_block_seq(&mut r, table(dc, td)?, table(ac, ta)?, blk, &mut c.pred)?;
                    } else if ss == 0 && ah == 0 {
                        let t = r.huff(table(dc, td)?)? as u32;
                        c.pred = c.pred.wrapping_add(r.extend(t)?);
                        blk[0] = (c.pred << al) as i16;
                    } else if ss == 0 {
                        blk[0] |= (r.bits(1) << al) as i16;
                    } else if ah == 0 {
                        decode_ac_first(&mut r, table(ac, ta)?, blk, ss, se, al, &mut eobrun)?;
                    } else {
                        decode_ac_refine(&mut r, table(ac, ta)?, blk, ss, se, al, &mut eobrun)?;
                    }
                }
            }
        }
        r.truncated()?;
    }
    Ok(r.pos)
}

fn decode_block_seq(
    r: &mut JpegBits,
    dc: &JpegHuff,
    ac: &JpegHuff,
    blk: &mut [i16],
    pred: &mut i32,
) -> Result<(), String> {
    let t = r.huff(dc)? as u32;
    *pred = pred.wrapping_add(r.extend(t)?);
    blk.fill(0);
    blk[0] = *pred as i16;
    let mut k = 1;
    while k < 64 {
        let rs = r.huff(ac)?;
        let (run, s) = ((rs >> 4) as usize, (rs & 15) as u32);
        if s == 0 {
            if run != 15 {
                break;
            }
            k += 16;
            continue;
        }
        k += run;
        if k > 63 {
            return Err("bad JPEG AC coefficient index".into());
        }
        blk[ZIGZAG[k] as usize] = r.extend(s)? as i16;
        k += 1;
    }
    Ok(())
}

fn decode_ac_first(
    r: &mut JpegBits,
    ac: &JpegHuff,
    blk: &mut [i16],
    ss: usize,
    se: usize,
    al: u32,
    eobrun: &mut u32,
) -> Result<(), String> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = ss;
    while k <= se {
        let rs = r.huff(ac)?;
        let (run, s) = ((rs >> 4) as u32, (rs & 15) as u32);
        if s == 0 {
            if run < 15 {
                *eobrun = (1 << run) - 1 + r.bits(run);
                break;
            }
            k += 16;
            continue;
        }
        k += run as usize;
        if k > se {
            return Err("bad JPEG AC coefficient index".into());
        }
        blk[ZIGZAG[k] as usize] = (r.extend(s)? << al) as i16;
        k += 1;
    }
    Ok(())
}

/// Successive-approximation AC refinement (libjpeg `decode_mcu_AC_refine`).
fn decode_ac_refine(
    r: &mut JpegBits,
    ac: &JpegHuff,
    blk: &mut [i16],
    ss: usize,
    se: usize,
    al: u32,
    eobrun: &mut u32,
) -> Result<(), String> {
    let (p1, m1) = (1i16 << al, -1i16 << al);
    let refine = |r: &mut JpegBits, c: &mut i16| {
        if r.bits(1) == 1 && *c & p1 == 0 {
            *c = c.wrapping_add(if *c >= 0 { p1 } else { m1 });
        }
    };
    let mut k = ss;
    if *eobrun == 0 {
        while k <= se {
            let rs = r.huff(ac)?;
            let (mut run, s) = (rs >> 4, rs & 15);
            let mut val = 0;
            if s != 0 {
                val = if r.bits(1) == 1 { p1 } else { m1 };
            } else if run != 15 {
                *eobrun = (1 << run) + r.bits(run as u32);
                break;
            }
            while k <= se {
                let c = &mut blk[ZIGZAG[k] as usize];
                if *c != 0 {
                    refine(r, c);
                } else if run == 0 {
                    break;
                } else {
                    run -= 1;
                }
                k += 1;
            }
            if val != 0 {
                if k > se {
                    return Err("bad JPEG AC coefficient index".into());
                }
                blk[ZIGZAG[k] as usize] = val;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        for &z in &ZIGZAG[k..=se] {
            if blk[z as usize] != 0 {
                refine(r, &mut blk[z as usize]);
            }
        }
        *eobrun -= 1;
    }
    Ok(())
}

/// libjpeg `jpeg_idct_islow`: dequantize + 8x8 integer IDCT (13-bit constants, 2 pass-1 bits).
fn idct_islow(coef: &[i16], q: &[u16; 64], out: &mut [u8], stride: usize) {
    fn pass(s: [i64; 8]) -> [i64; 8] {
        let z1 = (s[2] + s[6]) * 4433;
        let (t2, t3) = (z1 - s[6] * 15137, z1 + s[2] * 6270);
        let (t0, t1) = ((s[0] + s[4]) << 13, (s[0] - s[4]) << 13);
        let (t10, t13, t11, t12) = (t0 + t3, t0 - t3, t1 + t2, t1 - t2);
        let (o0, o1, o2, o3) = (s[7], s[5], s[3], s[1]);
        let (z1, z2, z3, z4) = (o0 + o3, o1 + o2, o0 + o2, o1 + o3);
        let z5 = (z3 + z4) * 9633;
        let (z1, z2, z3, z4) = (z1 * -7373, z2 * -20995, z3 * -16069 + z5, z4 * -3196 + z5);
        let (o0, o1, o2, o3) = (
            o0 * 2446 + z1 + z3,
            o1 * 16819 + z2 + z4,
            o2 * 25172 + z2 + z3,
            o3 * 12299 + z1 + z4,
        );
        [
            t10 + o3,
            t11 + o2,
            t12 + o1,
            t13 + o0,
            t13 - o0,
            t12 - o1,
            t11 - o2,
            t10 - o3,
        ]
    }
    let descale = |x: i64, n: u32| (x + (1 << (n - 1))) >> n;
    let mut ws = [0i64; 64];
    for c in 0..8 {
        let col = pass(std::array::from_fn(|r| {
            coef[r * 8 + c] as i64 * q[r * 8 + c] as i64
        }));
        for (r, v) in col.into_iter().enumerate() {
            ws[r * 8 + c] = descale(v, 11);
        }
    }
    for r in 0..8 {
        let row = pass(std::array::from_fn(|c| ws[r * 8 + c]));
        for (c, v) in row.into_iter().enumerate() {
            out[r * stride + c] = (descale(v, 18) + 128).clamp(0, 255) as u8;
        }
    }
}

/// Upsamples a component plane (`dw x dh` valid samples) by integer factors to `w x h`,
/// using libjpeg's triangle ("fancy") filters for h2v1, h1v2 and h2v2, replication otherwise.
#[allow(clippy::too_many_arguments)]
fn upsample(
    p: &[u8],
    stride: usize,
    dw: usize,
    dh: usize,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let row = |y: usize| &p[y.min(dh - 1) * stride..][..dw];
    let fancy = |cs: &[i32], o: &mut [u8], shift: u32, bias_l: i32, bias_r: i32| {
        for (x, v) in o.iter_mut().enumerate() {
            let i = x / 2;
            let (nb, bias) = if x % 2 == 0 {
                (cs[i.saturating_sub(1)], bias_l)
            } else {
                (cs[(i + 1).min(dw - 1)], bias_r)
            };
            *v = ((3 * cs[i] + nb + bias) >> shift) as u8;
        }
    };
    let mut out = vec![0u8; w * h];
    let mut cs = vec![0i32; dw];
    for (y, o) in out.chunks_exact_mut(w).enumerate() {
        // Vertical triangle neighbour: the row above for even output rows, below for odd.
        let (cur, near) = (
            row(y / 2),
            row(if y % 2 == 0 {
                (y / 2).saturating_sub(1)
            } else {
                y / 2 + 1
            }),
        );
        match (fx, fy) {
            (1, 1) => o.copy_from_slice(&row(y)[..w]),
            (2, 1) if dw > 2 => {
                for (s, &v) in cs.iter_mut().zip(row(y)) {
                    *s = v as i32;
                }
                fancy(&cs, o, 2, 1, 2);
            }
            (1, 2) => {
                for ((v, &a), &b) in o.iter_mut().zip(cur).zip(near) {
                    *v = ((3 * a as i32 + b as i32 + 1 + (y % 2) as i32) >> 2) as u8;
                }
            }
            (2, 2) if dw > 2 => {
                for ((s, &a), &b) in cs.iter_mut().zip(cur).zip(near) {
                    *s = 3 * a as i32 + b as i32;
                }
                fancy(&cs, o, 4, 8, 7);
            }
            _ => {
                let src = row(y / fy);
                for (x, v) in o.iter_mut().enumerate() {
                    *v = src[x / fx];
                }
            }
        }
    }
    out
}

/// Encodes `frames` (same size, even dims) to H.264 `path` by piping rgb24 into the `ffmpeg` binary.
pub fn write_mp4(frames: &[ImageBuffer], fps: u32, path: &Path) -> Result<(), String> {
    let first = frames.first().ok_or("write_mp4: no frames")?;
    let (w, h) = (first.width, first.height);
    if frames
        .iter()
        .any(|f| f.width != w || f.height != h || f.data.len() != w * h * 3)
    {
        return Err(format!("write_mp4: frames must all be RGB {w}x{h}"));
    }
    let mut child = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
        ])
        .arg(format!("{w}x{h}"))
        .arg("-r")
        .arg(fps.to_string())
        .args(["-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                "ffmpeg not found on PATH; install ffmpeg (with libx264) to write .mp4 video"
                    .to_string()
            }
            _ => format!("failed to start ffmpeg: {e}"),
        })?;
    let mut stdin = child.stdin.take().ok_or("ffmpeg stdin unavailable")?;
    // A write error means ffmpeg exited early; its stderr below says why.
    let write_err = frames
        .iter()
        .try_for_each(|f| stdin.write_all(&f.data))
        .err();
    drop(stdin);
    let out = child
        .wait_with_output()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !out.status.success() || write_err.is_some() {
        return Err(format!(
            "ffmpeg failed ({}): {}{}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim(),
            write_err
                .map(|e| format!(" (pipe: {e})"))
                .unwrap_or_default()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_png_encoder_and_roundtrip() {
        let mut img = ImageBuffer::new(8, 8);
        img.set_pixel(0, 0, 255, 0, 0); // Red
        img.set_pixel(7, 7, 0, 255, 0); // Green
        let png_bytes = img.to_png();

        assert!(png_bytes.len() > 16);
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");

        let decoded = ImageBuffer::decode(&png_bytes).expect("Failed to parse PNG");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.get_pixel(0, 0), (255, 0, 0));
        assert_eq!(decoded.get_pixel(7, 7), (0, 255, 0));
    }

    #[test]
    fn test_image_decode_rejects_unsupported_input() {
        assert!(ImageBuffer::decode(&[0u8; 12]).is_err());
        // Flipping the stored block to fixed Huffman (BTYPE=01) yields a corrupt stream, never a fake image.
        let mut png = ImageBuffer::new(4, 4).to_png();
        let idat = png.windows(4).position(|w| w == b"IDAT").unwrap();
        png[idat + 6] = 0x02;
        assert!(ImageBuffer::decode(&png).is_err());
        for (bytes, name) in [
            (&b"GIF89a\x01\0\x01\0"[..], "GIF"),
            (b"RIFF\0\0\0\0WEBPVP8 ", "WebP"),
            (b"BM\0\0\0\0\0\0", "BMP"),
            (b"II*\0\0\0\0\0", "TIFF"),
        ] {
            let err = ImageBuffer::decode(bytes).unwrap_err();
            assert!(err.contains(name) && err.contains("PNG and JPEG"), "{err}");
        }
    }

    #[test]
    fn test_base64_encode_decode() {
        let data = b"Foundry Engine Qwen-Image Diffusion Test";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).expect("Decode failed");
        assert_eq!(decoded, data);
    }
}
