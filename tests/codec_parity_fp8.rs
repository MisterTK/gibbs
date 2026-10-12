//! `FP8_E4M3_B32` GPU `decode8` must equal the CPU dequantizer (`repack::fp8`) bit-for-bit, on random
//! blocks with every non-NaN E4M3 byte and finite f32 scales (tiny and huge, overflow to inf included).
#![cfg(target_os = "macos")]

use gibbs_engine::graph::codec::{codec, msl_prelude};
use gibbs_engine::hardware::QuantFormat as Q;
use gibbs_engine::model::layers::gguf_block_codec;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

const BLOCKS: usize = 4096;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// A random finite f32 scale in `repack`'s domain: exponent field 10..=254 (|s| >= 2^-117), so every
/// `byte * scale` is a normal f32 or overflows to inf (identically on both sides). Smaller scales are
/// never emitted because Apple GPUs flush f32-subnormal results (CPU -6.35e-40 vs GPU -0.0).
fn rand_f32(r: &mut Lcg) -> f32 {
    let exp = 10 + (r.next() % 245) as u32;
    f32::from_bits(((r.next() as u32 & 1) << 31) | (exp << 23) | (r.next() as u32 & 0x7F_FFFF))
}

fn run_decode_all(device: &Device, msl: &str, block_elems: usize, block_bytes: usize, raw: &[u8]) -> Vec<f32> {
    let subs = block_elems / 8;
    let src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{msl}\n\
         kernel void decode_all(device const uchar* blocks [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
             uint b = gid / {subs}u, sub = gid % {subs}u;\n\
             float o[8];\n\
             decode8(blocks + (ulong)b * {block_bytes}ul, sub, o);\n\
             for (uint i = 0; i < 8; i++) out[gid * 8 + i] = o[i];\n\
         }}\n"
    );
    let opts = CompileOptions::new();
    // Same as production (`graph/metal.rs`): precise math.
    opts.set_fast_math_enabled(false);
    let lib = device.new_library_with_source(&src, &opts).unwrap_or_else(|e| panic!("MSL compile failed: {e}"));
    let pso = device
        .new_compute_pipeline_state_with_function(&lib.get_function("decode_all", None).unwrap())
        .unwrap();
    let threads = raw.len() / block_bytes * subs;
    let inb = device.new_buffer_with_data(raw.as_ptr().cast(), raw.len() as u64, MTLResourceOptions::StorageModeShared);
    let outb = device.new_buffer((threads * 8 * 4) as u64, MTLResourceOptions::StorageModeShared);
    let q = device.new_command_queue();
    let cb = q.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&inb), 0);
    enc.set_buffer(1, Some(&outb), 0);
    let tg = pso.max_total_threads_per_threadgroup().min(64);
    enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(tg, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    // SAFETY: shared-storage buffer of `threads * 8` f32, GPU work completed above.
    unsafe { std::slice::from_raw_parts(outb.contents() as *const f32, threads * 8) }.to_vec()
}

#[test]
fn fp8_e4m3_b32_decode_matches_cpu() {
    let device = Device::system_default().expect("no Metal device");
    let c = codec(Q::FP8_E4M3_B32).expect("FP8_E4M3_B32 codec is not registered");
    let (n, bytes, cpu) = gguf_block_codec(Q::FP8_E4M3_B32).expect("CPU reference");
    assert_eq!((c.block_elems as usize, c.block_bytes as usize, n, bytes), (32, 36, 32, 36));
    let mut r = Lcg(0x9E37_79B9);
    let mut raw = Vec::with_capacity(BLOCKS * bytes);
    for _ in 0..BLOCKS {
        // E4M3FN has one NaN code per sign (0x7F, 0xFF); repack never emits it.
        raw.extend((0..n).map(|_| loop {
            let b = (r.next() >> 3) as u8;
            if b & 0x7F != 0x7F {
                break b;
            }
        }));
        raw.extend_from_slice(&rand_f32(&mut r).to_le_bytes());
    }
    let msl = format!("{}{}", msl_prelude(), c.msl);
    let gpu = run_decode_all(&device, &msl, n, bytes, &raw);
    let mut want = vec![0.0f32; BLOCKS * n];
    for (b, y) in raw.chunks_exact(bytes).zip(want.chunks_exact_mut(n)) {
        cpu(b, y);
    }
    let bad = gpu.iter().zip(&want).position(|(g, w)| g.to_bits() != w.to_bits());
    assert!(bad.is_none(), "mismatch at {:?}: gpu {:?} cpu {:?}", bad, bad.map(|p| gpu[p]), bad.map(|p| want[p]));
}
