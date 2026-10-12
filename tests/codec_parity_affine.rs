//! GPU `decode8` of the 4-bit affine formats (MLX `MLX_AFFINE4_G*`, compressed-tensors `INT4_G*`) must
//! equal the CPU dequantizer bit-for-bit on random blocks: random nibbles, finite random f16 scales and
//! biases / zero points (subnormals and large exponents included).
#![cfg(target_os = "macos")]

use gibbs_engine::graph::codec::{codec, msl_prelude};
use gibbs_engine::hardware::QuantFormat as Q;
use gibbs_engine::model::layers::gguf_block_codec;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

const BLOCKS: usize = 512;

const FORMATS: &[(Q, usize)] = &[
    (Q::MLX_AFFINE4_G32, 32),
    (Q::MLX_AFFINE4_G64, 64),
    (Q::MLX_AFFINE4_G128, 128),
    (Q::INT4_G32, 32),
    (Q::INT4_G64, 64),
    (Q::INT4_G128, 128),
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

fn random_blocks(group: usize, seed: u64) -> Vec<u8> {
    let block_bytes = group / 2 + 4;
    let mut r = Lcg(seed);
    let mut raw: Vec<u8> = (0..BLOCKS * block_bytes).map(|_| (r.next() >> 5) as u8).collect();
    for b in raw.chunks_exact_mut(block_bytes) {
        for o in [group / 2, group / 2 + 2] {
            b[o..o + 2].copy_from_slice(&rand_f16(&mut r).to_le_bytes());
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
fn affine_codec_decode_matches_cpu() {
    let device = Device::system_default().expect("no Metal device");
    for (i, &(f, g)) in FORMATS.iter().enumerate() {
        let c = codec(f).unwrap_or_else(|| panic!("codec {f:?} is not registered"));
        let (n, bytes, cpu) = gguf_block_codec(f).expect("CPU reference");
        assert_eq!((c.block_elems as usize, c.block_bytes as usize), (g, g / 2 + 4), "{f:?} geometry");
        assert_eq!((n, bytes), (g, g / 2 + 4), "{f:?} CPU geometry");
        let raw = random_blocks(g, 0x51ED_2700 + i as u64);
        let msl = format!("{}{}", msl_prelude(), c.msl);
        let gpu = run_decode_all(&device, &msl, n, bytes, &raw);
        let mut want = vec![0.0f32; BLOCKS * n];
        for (b, y) in raw.chunks_exact(bytes).zip(want.chunks_exact_mut(n)) {
            cpu(b, y);
        }
        if let Some(p) = gpu.iter().zip(&want).position(|(g, w)| g.to_bits() != w.to_bits()) {
            panic!(
                "codec {f:?}: block {} elem {}: gpu {:e} ({:#010x}) != cpu {:e} ({:#010x})",
                p / n,
                p % n,
                gpu[p],
                gpu[p].to_bits(),
                want[p],
                want[p].to_bits()
            );
        }
        println!("codec {f:?}: {BLOCKS} blocks bit-exact");
    }
}
