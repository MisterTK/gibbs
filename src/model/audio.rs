//! Audio input I/O: WAV decoding (PCM 8/16/24/32-bit, IEEE float, downmixed to mono) and
//! band-limited resampling.

/// Mono samples in [-1, 1] and the sample rate of a RIFF/WAVE file; other containers or codecs
/// are an Err.
pub fn read_wav(b: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("audio is not a RIFF/WAVE file (only WAV input is supported)".into());
    }
    let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let (mut fmt, mut data) = (None, None);
    let mut o = 12;
    while o + 8 <= b.len() {
        let size = u32_at(o + 4) as usize;
        let body = &b[o + 8..(o + 8).saturating_add(size).min(b.len())];
        match &b[o..o + 4] {
            b"fmt " if body.len() >= 16 => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        o += 8 + size + (size & 1);
    }
    let fmt = fmt.ok_or("WAV has no fmt chunk")?;
    let data = data.ok_or("WAV has no data chunk")?;
    let f = |i: usize| u16::from_le_bytes([fmt[i], fmt[i + 1]]);
    let mut format = f(0);
    let channels = f(2) as usize;
    let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bits = f(14) as usize;
    if format == 0xFFFE && fmt.len() >= 26 {
        format = f(24); // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts with the format tag
    }
    let bytes = bits / 8;
    if channels == 0 || bytes == 0 || !bits.is_multiple_of(8) {
        return Err(format!(
            "WAV fmt is invalid ({channels} channels, {rate} Hz, {bits} bits)"
        ));
    }
    if !(1_000..=768_000).contains(&rate) {
        return Err(format!(
            "WAV sample rate {rate} Hz is outside 1 kHz..768 kHz"
        ));
    }
    let sample = |s: &[u8]| -> f32 {
        match (format, bits) {
            (1, 8) => (s[0] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
            (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
            (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
            (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            (3, 64) => f64::from_le_bytes(s[..8].try_into().unwrap()) as f32,
            _ => f32::NAN,
        }
    };
    if sample(&[0; 8]).is_nan() {
        return Err(format!(
            "WAV codec {format} at {bits} bits is not supported (PCM or IEEE float only)"
        ));
    }
    let frame = bytes * channels;
    let mono = data
        .chunks_exact(frame)
        .map(|fr| fr.chunks_exact(bytes).map(sample).sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, rate))
}

/// Samples [`resample`] returns for `len` input samples.
pub fn resampled_len(len: usize, from: u32, to: u32) -> usize {
    if from == to {
        return len;
    }
    (len as f64 / (from as f64 / to as f64)).round() as usize
}

/// Band-limited resampling (Hann-windowed sinc, 16 zero crossings).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    const ZEROS: f64 = 16.0;
    let ratio = from as f64 / to as f64;
    let cutoff = (1.0 / ratio).min(1.0);
    let half = ZEROS / cutoff;
    let n_out = resampled_len(x.len(), from, to);
    (0..n_out)
        .map(|i| {
            let t = i as f64 * ratio;
            let (lo, hi) = (
                (t - half).ceil().max(0.0) as usize,
                ((t + half).floor() as usize).min(x.len().saturating_sub(1)),
            );
            let mut acc = 0.0;
            for (k, &v) in x.iter().enumerate().take(hi + 1).skip(lo) {
                let d = t - k as f64;
                let a = std::f64::consts::PI * cutoff * d;
                let sinc = if d.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                let win = 0.5 + 0.5 * (std::f64::consts::PI * d / half).cos();
                acc += v as f64 * cutoff * sinc * win;
            }
            acc as f32
        })
        .collect()
}

/// A RIFF/WAVE file of `data` sample bytes (an unknown `LIST` chunk before `data` exercises
/// chunk skipping). Test fixture.
#[cfg(test)]
pub(crate) fn wav_bytes(format: u16, bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
    let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
    b.extend(16u32.to_le_bytes());
    b.extend(format.to_le_bytes());
    b.extend(channels.to_le_bytes());
    b.extend(rate.to_le_bytes());
    b.extend((rate * (bits / 8 * channels) as u32).to_le_bytes());
    b.extend((bits / 8 * channels).to_le_bytes());
    b.extend(bits.to_le_bytes());
    b.extend(b"LIST\x02\0\0\0xy"); // unknown chunk, skipped
    b.extend(b"data");
    b.extend((data.len() as u32).to_le_bytes());
    b.extend(data);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_decodes_pcm_float_and_downmixes() {
        let wav = wav_bytes;
        let pcm: Vec<u8> = [16384i16, -16384, 32767, 0]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let (x, rate) = read_wav(&wav(1, 16, 2, 22050, &pcm)).unwrap();
        assert_eq!(rate, 22050);
        assert_eq!(x.len(), 2);
        assert!(x[0].abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-3);
        let f: Vec<u8> = [0.25f32, -1.0]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        assert_eq!(
            read_wav(&wav(3, 32, 1, 16000, &f)).unwrap().0,
            vec![0.25, -1.0]
        );
        assert!(read_wav(&wav(2, 4, 1, 16000, &[0; 4])).is_err()); // ADPCM
        assert!(read_wav(b"OggS....").is_err());
        // Absurd declared rates are rejected before anything is sized from them.
        assert!(read_wav(&wav(1, 16, 1, 1, &pcm)).is_err());
        assert!(read_wav(&wav(1, 16, 1, 10_000_000, &pcm)).is_err());
    }

    #[test]
    fn resample_keeps_a_tone() {
        let tone: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        let y = resample(&tone, 48000, 16000);
        assert_eq!(y.len(), 16000);
        for (i, &v) in y.iter().enumerate().skip(100).take(15800) {
            let want = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
            assert!((v - want).abs() < 2e-2, "sample {i}: {v} vs {want}");
        }
    }
}
