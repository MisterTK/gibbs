//! `codec_decode_matches_cpu`: every registered GPU codec's MSL `decode8`, run over all sub-blocks of
//! random blocks, must equal the CPU `dequantize_*_block` bit-for-bit. A codec that fails here must
//! not be registered in `graph::codec::codec`.
#![cfg(target_os = "macos")]

use gibbs_engine::graph::codec::{codec, msl_prelude};
use gibbs_engine::hardware::QuantFormat as Q;
use gibbs_engine::model::layers::gguf_block_codec;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

const BLOCKS: usize = 512;

/// Every GGUF block format with a CPU dequantizer (`gguf_block_codec`) plus its aliases.
const CANDIDATES: &[Q] = &[
    Q::GGUF_Q4_0,
    Q::GGUF_Q4_1,
    Q::GGUF_Q5_0,
    Q::GGUF_Q5_1,
    Q::GGUF_Q8_0,
    Q::GGUF_Q2_K,
    Q::GGUF_Q3_K,
    Q::GGUF_Q4_K_M,
    Q::GGUF_Q5_K_M,
    Q::GGUF_Q6_K,
    Q::GGUF_IQ4_NL,
    Q::GGUF_IQ4_XS,
    Q::GGUF_IQ3_XXS,
    Q::GGUF_IQ3_S,
    Q::GGUF_IQ2_XXS,
    Q::GGUF_IQ2_XS,
    Q::GGUF_IQ2_S,
    Q::GGUF_IQ2_M,
    Q::GGUF_IQ1_S,
    Q::GGUF_IQ1_M,
    Q::GGUF_TQ1_0,
    Q::GGUF_TQ2_0,
    Q::MXFP4,
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// A random finite f16 (no NaN/inf), including subnormals (1 in 8) and tiny/large exponents, so a
/// GPU that flushes or mis-converts them shows up as a mismatch.
fn rand_f16(r: &mut Lcg) -> u16 {
    let sign = (r.next() & 1) as u16;
    let exp = if r.next() % 8 == 0 { 0 } else { 1 + (r.next() % 25) as u16 };
    let mant = (r.next() & 0x3FF) as u16;
    (sign << 15) | (exp << 10) | mant
}

/// Byte offsets of the f16 fields a format's dequantizer reads (IQ1_M and MXFP4 are special-cased).
fn f16_offsets(f: Q) -> &'static [usize] {
    match f {
        Q::GGUF_Q4_1 | Q::GGUF_Q5_1 | Q::GGUF_Q4_K_M | Q::GGUF_Q5_K_M => &[0, 2],
        Q::GGUF_Q6_K => &[208],
        Q::GGUF_Q2_K => &[80, 82],
        Q::GGUF_Q3_K => &[108],
        Q::GGUF_TQ1_0 => &[52],
        Q::GGUF_TQ2_0 => &[64],
        Q::GGUF_IQ1_M | Q::MXFP4 => &[],
        _ => &[0],
    }
}

fn random_blocks(f: Q, block_bytes: usize, seed: u64) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut raw: Vec<u8> = (0..BLOCKS * block_bytes).map(|_| (r.next() >> 5) as u8).collect();
    for b in raw.chunks_exact_mut(block_bytes) {
        for &o in f16_offsets(f) {
            b[o..o + 2].copy_from_slice(&rand_f16(&mut r).to_le_bytes());
        }
        match f {
            // f16 `d` is spread over the top nibbles of the four u16 sub-scale words at 48..56.
            Q::GGUF_IQ1_M => {
                let h = rand_f16(&mut r);
                for k in 0..4 {
                    let at = 48 + 2 * k;
                    let w = u16::from_le_bytes([b[at], b[at + 1]]);
                    let w = (w & 0x0FFF) | (((h >> (4 * k)) & 0xF) << 12);
                    b[at..at + 2].copy_from_slice(&w.to_le_bytes());
                }
            }
            // E8M0 exponent; includes the subnormal-scale cases e < 2 now and then.
            Q::MXFP4 => b[0] = if r.next() % 16 == 0 { (r.next() % 2) as u8 } else { 100 + (r.next() % 50) as u8 },
            _ => {}
        }
    }
    raw
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
fn codec_decode_matches_cpu() {
    let device = Device::system_default().expect("no Metal device");
    let mut failures = Vec::new();
    for (i, &f) in CANDIDATES.iter().enumerate() {
        let c = codec(f).unwrap_or_else(|| panic!("codec {f:?} is not registered"));
        let (n, bytes, cpu) = gguf_block_codec(f).expect("CPU reference");
        assert_eq!((c.block_elems as usize, c.block_bytes as usize), (n, bytes), "{f:?} geometry");
        // Aliases (IQ2_S -> IQ2_M) share the codec; the CPU dequantizer is the same function.
        let raw = random_blocks(c.format, bytes, 0x9E37_79B9 + i as u64);
        let msl = format!("{}{}", msl_prelude(), c.msl); // guards make prelude + codec idempotent
        let gpu = run_decode_all(&device, &msl, n, bytes, &raw);
        let mut want = vec![0.0f32; BLOCKS * n];
        for (b, y) in raw.chunks_exact(bytes).zip(want.chunks_exact_mut(n)) {
            cpu(b, y);
        }
        let bad = gpu.iter().zip(&want).position(|(g, w)| g.to_bits() != w.to_bits());
        if let Some(p) = bad {
            failures.push(format!(
                "codec {f:?}: block {} elem {}: gpu {:e} ({:#010x}) != cpu {:e} ({:#010x})",
                p / n,
                p % n,
                gpu[p],
                gpu[p].to_bits(),
                want[p],
                want[p].to_bits()
            ));
            continue;
        }
        println!("codec {f:?}: {BLOCKS} blocks bit-exact");
    }
    assert!(failures.is_empty(), "unregister these codecs:\n{}", failures.join("\n"));
}

/// All tables in `msl_prelude()` compile together with the helpers.
#[test]
fn msl_prelude_compiles() {
    let device = Device::system_default().expect("no Metal device");
    let src = format!("#include <metal_stdlib>\nusing namespace metal;\n{}\nkernel void k() {{}}\n", msl_prelude());
    {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        device.new_library_with_source(&src, &opts)
    }.unwrap_or_else(|e| panic!("prelude: {e}"));
}
