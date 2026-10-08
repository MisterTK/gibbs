//! Core ML / Apple Neural Engine (ANE) executor for Gibbs.
//!
//! Compiles Gibbs graphs to Core ML ML Program (MIL) format, using Apple's
//! rank-4 layout `[1, C, 1, S]` where linear projections become 1×1 2D convolutions,
//! weights are written to an external `weights/weight.bin` MILBlob file, and models
//! are compiled per sequence bucket and cached under `~/.cache/gibbs/coreml/`.
//!
//! Querying `MLComputePlan` at load reports the preferred compute device for every op,
//! validating the percentage of operations placed on the Apple Neural Engine (`MLNeuralEngineComputeDevice`).

#![allow(unexpected_cfgs)]

use crate::graph::Weights;
use crate::hardware::sysctl;
use metal::objc::runtime::{Class, Object};
use metal::objc::{class, msg_send, sel, sel_impl};
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

type Id = *mut Object;

#[link(name = "CoreML", kind = "framework")]
extern "C" {}
#[link(name = "Foundation", kind = "framework")]
extern "C" {
    static _NSConcreteGlobalBlock: c_void;
}

// ---------------------------------------------------------------------------
// Protobuf wire serialization helpers
// ---------------------------------------------------------------------------

fn varint(b: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        b.push((v as u8) | 0x80);
        v >>= 7;
    }
    b.push(v as u8);
}

fn int(b: &mut Vec<u8>, f: u32, v: u64) {
    varint(b, (f as u64) << 3);
    varint(b, v);
}

fn bytes(b: &mut Vec<u8>, f: u32, d: &[u8]) {
    varint(b, ((f as u64) << 3) | 2);
    varint(b, d.len() as u64);
    b.extend_from_slice(d);
}

fn msg(f: u32, build: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut m = vec![];
    build(&mut m);
    let mut o = vec![];
    bytes(&mut o, f, &m);
    o
}

fn map_entry(f: u32, key: &str, val: Vec<u8>) -> Vec<u8> {
    msg(f, |m| {
        bytes(m, 1, key.as_bytes());
        m.extend(val_as(2, val));
    })
}

fn val_as(f: u32, inner: Vec<u8>) -> Vec<u8> {
    let mut o = vec![];
    bytes(&mut o, f, &inner);
    o
}

// Core ML MIL DataType enum (MIL.proto)
const BOOL: u64 = 1;
const STRING: u64 = 2;
const FP16: u64 = 10;
#[allow(dead_code)]
const FP32: u64 = 11;
const INT32: u64 = 23;

fn tensor_type(dt: u64, dims: &[u64]) -> Vec<u8> {
    let mut t = vec![];
    int(&mut t, 1, dt);
    int(&mut t, 2, dims.len() as u64);
    for &d in dims {
        t.extend(msg(3, |m| m.extend(msg(1, |c| int(c, 1, d)))));
    }
    t
}

fn value_type(dt: u64, dims: &[u64]) -> Vec<u8> {
    val_as(1, tensor_type(dt, dims))
}

fn named(f: u32, name: &str, dt: u64, dims: &[u64]) -> Vec<u8> {
    msg(f, |m| {
        bytes(m, 1, name.as_bytes());
        m.extend(val_as(2, value_type(dt, dims)));
    })
}

fn str_value(s: &str) -> Vec<u8> {
    let mut v = val_as(2, value_type(STRING, &[]));
    v.extend(msg(3, |iv| {
        iv.extend(msg(1, |tv| {
            tv.extend(msg(4, |rs| bytes(rs, 1, s.as_bytes())))
        }))
    }));
    v
}

fn bool_value(val: bool) -> Vec<u8> {
    let mut v = val_as(2, value_type(BOOL, &[]));
    v.extend(msg(3, |iv| {
        iv.extend(msg(1, |tv| {
            tv.extend(msg(3, |rb| bytes(rb, 1, &[val as u8])));
        }))
    }));
    v
}

fn bool_const(name: &str, val: bool) -> Vec<u8> {
    msg(3, |op| {
        bytes(op, 1, b"const");
        op.extend(named(3, name, BOOL, &[]));
        op.extend(map_entry(5, "name", str_value(name)));
        op.extend(map_entry(5, "val", bool_value(val)));
    })
}

fn int32_const(name: &str, dims: &[u64], data: &[i32]) -> Vec<u8> {
    let mut packed = vec![];
    for &v in data {
        varint(&mut packed, v as i64 as u64);
    }
    let mut val = val_as(2, value_type(INT32, dims));
    val.extend(msg(3, |iv| {
        iv.extend(msg(1, |tv| tv.extend(msg(2, |ri| bytes(ri, 1, &packed)))))
    }));
    msg(3, |op| {
        bytes(op, 1, b"const");
        op.extend(named(3, name, INT32, dims));
        op.extend(map_entry(5, "name", str_value(name)));
        op.extend(map_entry(5, "val", val));
    })
}

fn fp16_const(name: &str, val: f32) -> Vec<u8> {
    let h = f32_to_f16(val);
    let mut val_m = val_as(2, value_type(FP16, &[]));
    val_m.extend(msg(3, |iv| {
        iv.extend(msg(1, |tv| {
            tv.extend(msg(7, |rb| bytes(rb, 1, &h.to_le_bytes())))
        }))
    }));
    msg(3, |op| {
        bytes(op, 1, b"const");
        op.extend(named(3, name, FP16, &[]));
        op.extend(map_entry(5, "name", str_value(name)));
        op.extend(map_entry(5, "val", val_m));
    })
}

fn str_const(name: &str, s: &str) -> Vec<u8> {
    msg(3, |op| {
        bytes(op, 1, b"const");
        op.extend(named(3, name, STRING, &[]));
        op.extend(map_entry(5, "name", str_value(name)));
        op.extend(map_entry(5, "val", str_value(s)));
    })
}

fn blob_file_value(file_name: &str, offset: u64, dims: &[u64], dt: u64) -> Vec<u8> {
    let mut val = val_as(2, value_type(dt, dims));
    val.extend(msg(5, |bv| {
        bytes(bv, 1, file_name.as_bytes());
        int(bv, 2, offset);
    }));
    val
}

fn blob_const_op(name: &str, file_name: &str, offset: u64, dims: &[u64], dt: u64) -> Vec<u8> {
    msg(3, |op| {
        bytes(op, 1, b"const");
        op.extend(named(3, name, dt, dims));
        op.extend(map_entry(5, "name", str_value(name)));
        op.extend(map_entry(
            5,
            "val",
            blob_file_value(file_name, offset, dims, dt),
        ));
    })
}

fn op(kind: &str, args: &[(&str, &str)], out: &str, dims: &[u64], dt: u64) -> Vec<u8> {
    msg(3, |o| {
        bytes(o, 1, kind.as_bytes());
        for (k, v) in args {
            o.extend(map_entry(2, k, msg(1, |b| bytes(b, 1, v.as_bytes()))));
        }
        o.extend(named(3, out, dt, dims));
        o.extend(map_entry(5, "name", str_value(out)));
    })
}

fn feature(f: u32, name: &str, shape: &[u64], dt_code: u64) -> Vec<u8> {
    msg(f, |fd| {
        bytes(fd, 1, name.as_bytes());
        fd.extend(msg(3, |ft| {
            ft.extend(msg(5, |at| {
                for &s in shape {
                    int(at, 1, s);
                }
                int(at, 2, dt_code);
            }))
        }));
    })
}

/// Exact IEEE-754 round-to-nearest-even FP32 -> FP16 conversion.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x007f_ffff;

    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - exp);
        let round = (m + 0x0000_1000) >> 13;
        sign | (round as u16)
    } else if exp >= 31 {
        if exp == 143 && mant != 0 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        }
    } else {
        let round_bit = (mant >> 12) & 1;
        let sticky = (mant & 0x0fff) != 0;
        let mut h_mant = (mant >> 13) as u32;
        if round_bit == 1 && (sticky || (h_mant & 1) != 0) {
            h_mant += 1;
        }
        let mut h_exp = exp as u32;
        if h_mant == 0x0400 {
            h_exp += 1;
            h_mant = 0;
        }
        sign | ((h_exp as u16) << 10) | (h_mant as u16)
    }
}

/// Exact IEEE-754 FP16 -> FP32 conversion including subnormals.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let mut m = mant;
            let mut e = 113 - 1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            f32::from_bits(sign | (e << 23) | (m << 13))
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits(sign | 0x7f80_0000)
        } else {
            f32::from_bits(sign | 0x7fc0_0000 | (mant << 13))
        }
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

// ---------------------------------------------------------------------------
// MILBlob external weight writer (`StorageFormat.hpp`)
// ---------------------------------------------------------------------------

pub struct MilBlobWriter {
    count: u32,
    bytes: Vec<u8>,
}

impl MilBlobWriter {
    pub fn new() -> Self {
        let mut b = Vec::with_capacity(64);
        b.extend_from_slice(&0u32.to_le_bytes()); // placeholder count
        b.extend_from_slice(&2u32.to_le_bytes()); // version = 2
        for _ in 0..7 {
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        MilBlobWriter { count: 0, bytes: b }
    }

    fn pad_64(&mut self) {
        let rem = self.bytes.len() % 64;
        if rem != 0 {
            self.bytes.resize(self.bytes.len() + (64 - rem), 0);
        }
    }

    /// Writes FP16 data and returns the metadata offset in `weight.bin`.
    pub fn write_fp16(&mut self, data: &[f32]) -> u64 {
        self.pad_64();
        let meta_offset = self.bytes.len() as u64;

        let raw_len = (data.len() * 2) as u64;
        let data_offset = meta_offset + 64;

        // Metadata (64 bytes)
        self.bytes.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // sentinel
        self.bytes.extend_from_slice(&1u32.to_le_bytes()); // dtype: 1 = fp16
        self.bytes.extend_from_slice(&raw_len.to_le_bytes());
        self.bytes.extend_from_slice(&data_offset.to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes()); // padding_bits
        for _ in 0..4 {
            self.bytes.extend_from_slice(&0u64.to_le_bytes());
        }

        // Data (aligned at +64)
        for &v in data {
            self.bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        self.pad_64();

        self.count += 1;
        meta_offset
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.bytes[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.bytes
    }
}

// ---------------------------------------------------------------------------
// Objective-C runtime bridge
// ---------------------------------------------------------------------------

unsafe fn ns(s: &str) -> Id {
    let c = CString::new(s).unwrap();
    msg_send![class!(NSString), stringWithUTF8String: c.as_ptr()]
}

unsafe fn rs(s: Id) -> String {
    if s.is_null() {
        return "<nil>".into();
    }
    let p: *const i8 = msg_send![s, UTF8String];
    CStr::from_ptr(p).to_string_lossy().into()
}

unsafe fn err_str(e: Id) -> String {
    if e.is_null() {
        "<no error>".into()
    } else {
        rs(msg_send![e, localizedDescription])
    }
}

unsafe fn class_name(o: Id) -> String {
    rs(msg_send![o, className])
}

unsafe fn nsnum_array(v: &[i64]) -> Id {
    let objs: Vec<Id> = v
        .iter()
        .map(|&x| msg_send![class!(NSNumber), numberWithLongLong: x])
        .collect();
    msg_send![class!(NSArray), arrayWithObjects: objs.as_ptr() count: objs.len()]
}

#[repr(C)]
struct BlockDesc {
    reserved: usize,
    size: usize,
}
#[repr(C)]
struct Block {
    isa: *const c_void,
    flags: i32,
    reserved: i32,
    invoke: unsafe extern "C" fn(*mut Block, Id, Id),
    desc: *const BlockDesc,
}
static DESC: BlockDesc = BlockDesc {
    reserved: 0,
    size: std::mem::size_of::<Block>(),
};
unsafe impl Sync for Block {}
unsafe impl Send for Block {}
static PLAN_LOCK: Mutex<()> = Mutex::new(());
static PLAN_STATE: Mutex<Option<(usize, usize)>> = Mutex::new(None);
static PLAN_BLOCK: std::sync::LazyLock<Block> = std::sync::LazyLock::new(|| Block {
    isa: unsafe { &_NSConcreteGlobalBlock },
    flags: 1 << 28,
    reserved: 0,
    invoke: on_plan,
    desc: &DESC,
});
unsafe extern "C" fn on_plan(_b: *mut Block, plan: Id, err: Id) {
    if !plan.is_null() {
        let _: Id = msg_send![plan, retain];
    }
    *PLAN_STATE.lock().unwrap() = Some((plan as usize, err as usize));
}

// ---------------------------------------------------------------------------
// Compute Plan placement inspection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct PlacementReport {
    pub total_ops: usize,
    pub ane_ops: usize,
    pub cpu_ops: usize,
    pub gpu_ops: usize,
    pub ane_cost: f64,
    pub cpu_cost: f64,
    pub gpu_cost: f64,
    pub op_devices: Vec<(String, String)>,
}

impl PlacementReport {
    pub fn ane_share(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            (self.ane_ops as f64 / self.total_ops as f64) * 100.0
        }
    }

    pub fn ane_cost_share(&self) -> f64 {
        let total = self.ane_cost + self.cpu_cost + self.gpu_cost;
        if total <= 1e-9 {
            0.0
        } else {
            (self.ane_cost / total) * 100.0
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "{}/{} ops on Neural Engine ({:.1}% ANE op share, {:.1}% cost share; {} CPU, {} GPU)",
            self.ane_ops,
            self.total_ops,
            self.ane_share(),
            self.ane_cost_share(),
            self.cpu_ops,
            self.gpu_ops,
        )
    }
}

unsafe fn inspect_compute_plan(compiled_url: Id, cfg: Id) -> Result<PlacementReport, String> {
    let _guard = PLAN_LOCK.lock().unwrap();
    *PLAN_STATE.lock().unwrap() = None;
    let cls: &Class = class!(MLComputePlan);
    let _: () = msg_send![cls, loadContentsOfURL: compiled_url configuration: cfg completionHandler: &*PLAN_BLOCK as *const Block];

    let (plan, perr) = loop {
        if let Some(p) = *PLAN_STATE.lock().unwrap() {
            break p;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    };

    let plan = plan as Id;
    if plan.is_null() {
        return Err(format!("MLComputePlan failed: {}", err_str(perr as Id)));
    }

    let mut rep = PlacementReport::default();
    let structure: Id = msg_send![plan, modelStructure];
    let program: Id = msg_send![structure, program];
    let funcs: Id = msg_send![program, functions];
    let main: Id = msg_send![funcs, objectForKey: ns("main")];
    if main.is_null() {
        return Ok(rep);
    }
    let blockp: Id = msg_send![main, block];
    let ops: Id = msg_send![blockp, operations];
    let n: usize = msg_send![ops, count];

    for i in 0..n {
        let o: Id = msg_send![ops, objectAtIndex: i];
        let op_name = rs(msg_send![o, operatorName]);
        if op_name == "const" {
            continue;
        }
        rep.total_ops += 1;
        let usage: Id = msg_send![plan, computeDeviceUsageForMLProgramOperation: o];
        let dev = if usage.is_null() {
            "Unknown".into()
        } else {
            class_name(msg_send![usage, preferredComputeDevice])
        };

        if dev.contains("NeuralEngine") {
            rep.ane_ops += 1;
        } else if dev.contains("CPU") {
            rep.cpu_ops += 1;
        } else if dev.contains("GPU") {
            rep.gpu_ops += 1;
        }
        let cost_obj: Id = msg_send![plan, estimatedCostOfMLProgramOperation: o];
        let w: f64 = if !cost_obj.is_null() {
            msg_send![cost_obj, weight]
        } else {
            0.0
        };
        if dev.contains("NeuralEngine") {
            rep.ane_cost += w;
        } else if dev.contains("CPU") {
            rep.cpu_cost += w;
        } else if dev.contains("GPU") {
            rep.gpu_cost += w;
        }
        rep.op_devices.push((op_name, dev));
    }

    Ok(rep)
}

// ---------------------------------------------------------------------------
// Core ML / ANE Session & Runtime
// ---------------------------------------------------------------------------

pub const ANE_CANDIDATE_BUCKETS: &[usize] = &[16, 64, 128, 256, 512];

pub struct CoreMlModel {
    pub bucket: usize,
    pub model: Id,
    pub report: PlacementReport,
    pub archive_hit: bool,
}
unsafe impl Send for CoreMlModel {}
unsafe impl Sync for CoreMlModel {}

impl Drop for CoreMlModel {
    fn drop(&mut self) {
        if !self.model.is_null() {
            unsafe {
                let _: () = msg_send![self.model, release];
            }
        }
    }
}
pub struct Session {
    cache_dir: PathBuf,
    weights_hash: u64,
    models: HashMap<usize, Arc<CoreMlModel>>,
    weights: Weights,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    pub active_buckets: Vec<usize>,
    pub bucket_placements: std::collections::BTreeMap<usize, PlacementReport>,
    pub report: String,
    pub archive_hit: bool,
}
unsafe impl Send for Session {}
unsafe impl Sync for Session {}
impl Session {
    pub fn new(
        cache_dir: &Path,
        weights: &Weights,
        hidden: usize,
        layers: usize,
        heads: usize,
    ) -> Result<Self, String> {
        let mut keys: Vec<&str> = weights.iter().map(|(k, _)| k).collect();
        keys.sort();
        let mut hasher_bytes = Vec::new();
        for &k in &keys {
            hasher_bytes.extend_from_slice(k.as_bytes());
            if let Ok(s) = weights.get(k).unwrap().as_f32() {
                hasher_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
                for &f in s.iter() {
                    hasher_bytes.extend_from_slice(&f.to_bits().to_le_bytes());
                }
            }
        }
        let weights_hash = fnv1a(&hasher_bytes);

        let mut s = Session {
            cache_dir: cache_dir.to_path_buf(),
            weights_hash,
            models: HashMap::new(),
            weights: weights.clone(),
            hidden_dim: hidden,
            num_layers: layers,
            num_heads: heads,
            active_buckets: Vec::new(),
            bucket_placements: std::collections::BTreeMap::new(),
            report: String::new(),
            archive_hit: false,
        };

        let mut active = Vec::new();
        let mut placements = std::collections::BTreeMap::new();
        let mut hit_count = 0;
        for &b in ANE_CANDIDATE_BUCKETS {
            let m = s.get_or_compile(b)?;
            if m.archive_hit {
                hit_count += 1;
            }
            if m.report.ane_share() >= 50.0 {
                active.push(b);
            }
            placements.insert(b, m.report.clone());
        }
        let per_bucket_summary: Vec<String> = placements
            .iter()
            .map(|(b, r)| format!("b{b}: {:.1}%", r.ane_share()))
            .collect();
        let per_bucket_str = per_bucket_summary.join(" · ");

        if active.is_empty() {
            return Err(format!(
                "Core ML: no sequence bucket qualified for Neural Engine (need >=50% ANE ops; got {per_bucket_str})"
            ));
        }

        s.active_buckets = active;
        let rep256 = placements.get(&256).cloned().unwrap_or_default();
        let hit_note = if hit_count == ANE_CANDIDATE_BUCKETS.len() {
            "cache hit"
        } else {
            "compiled"
        };
        s.report = format!(
            "Core ML ANE: {} ({hit_note}); active buckets {:?} ({per_bucket_str})",
            rep256.summary(),
            s.active_buckets,
        );
        s.archive_hit = hit_count > 0;
        s.bucket_placements = placements;
        Ok(s)
    }

    pub fn bucket_for(&self, seq_len: usize) -> Result<usize, String> {
        for &b in &self.active_buckets {
            if b >= seq_len {
                return Ok(b);
            }
        }
        let max = self.active_buckets.last().copied().unwrap_or(0);
        Err(format!(
            "sequence length {seq_len} exceeds largest ANE-qualified bucket {max}"
        ))
    }

    pub fn get_or_compile(&mut self, bucket: usize) -> Result<Arc<CoreMlModel>, String> {
        if let Some(m) = self.models.get(&bucket) {
            return Ok(Arc::clone(m));
        }

        let model = compile_or_load_bucket(
            &self.cache_dir,
            &self.weights,
            self.weights_hash,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            bucket,
        )?;
        let arc = Arc::new(model);
        self.models.insert(bucket, Arc::clone(&arc));
        Ok(arc)
    }

    /// Runs inference for an input tensor `[n, hidden]`.
    /// Padded to sequence bucket `bucket_for(n)`, with attention mask, then sliced back to `n`.
    pub fn run_encoder(&mut self, x: &[f32], n: usize) -> Result<Vec<f32>, String> {
        let h = self.hidden_dim;
        if x.len() != n * h {
            return Err(format!("Input length {} != n*h ({} * {})", x.len(), n, h));
        }
        let bucket = self.bucket_for(n)?;
        let m = self.get_or_compile(bucket)?;

        // ANE layout: [1, C, 1, S] where C = hidden, S = bucket
        // Input `x` is [n, hidden] in row-major order: x[i*h + c] -> p[c*bucket + i]
        let mut ane_input = vec![0.0f32; h * bucket];
        for i in 0..n {
            for c in 0..h {
                ane_input[c * bucket + i] = x[i * h + c];
            }
        }

        // Additive attention mask: 0.0 for active positions, -30000.0 for padded positions
        let mut ane_mask = vec![-30000.0f32; bucket];
        for i in 0..n {
            ane_mask[i] = 0.0;
        }

        // Run Core ML predict
        let ane_out = unsafe { predict_multi_array(m.model, &ane_input, &ane_mask, h, bucket)? };

        // Convert [1, C, 1, S] back to row-major [n, hidden]
        let mut out = vec![0.0f32; n * h];
        for i in 0..n {
            for c in 0..h {
                out[i * h + c] = ane_out[c * bucket + i];
            }
        }

        Ok(out)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

// ---------------------------------------------------------------------------
// Model emitter & compiler
// ---------------------------------------------------------------------------

fn compile_or_load_bucket(
    cache_root: &Path,
    weights: &Weights,
    weights_hash: u64,
    hidden: usize,
    layers: usize,
    heads: usize,
    bucket: usize,
) -> Result<CoreMlModel, String> {
    let os_build = sysctl::string(c"kern.osversion").unwrap_or_else(|| "unknown-os".into());
    let src_hash = fnv1a(include_str!("coreml.rs").as_bytes());
    let tag = format!("s1-{os_build}-b{bucket}-{src_hash:016x}-{weights_hash:016x}");
    let cache_dir = cache_root.join("coreml");
    std::fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;

    let compiled_path = cache_dir.join(format!("{tag}.mlmodelc"));

    unsafe {
        let cfg: Id = msg_send![class!(MLModelConfiguration), new];
        let _: () = msg_send![cfg, setComputeUnits: 3i64]; // cpuAndNeuralEngine

        if compiled_path.exists() {
            let url: Id =
                msg_send![class!(NSURL), fileURLWithPath: ns(&compiled_path.to_string_lossy())];
            let mut err: Id = std::ptr::null_mut();
            let model: Id = msg_send![class!(MLModel), modelWithContentsOfURL: url configuration: cfg error: &mut err];
            if !model.is_null() {
                let _: () = msg_send![model, retain];
                let report = inspect_compute_plan(url, cfg).unwrap_or_default();
                return Ok(CoreMlModel {
                    bucket,
                    model,
                    report,
                    archive_hit: true,
                });
            }
        }

        // Prune stale cache entries for this bucket
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            let prefix = format!("s1-{os_build}-b{bucket}-");
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&prefix) && !name.starts_with(&tag) {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }

        // Build .mlpackage
        let pkg_path = cache_dir.join(format!("{tag}.mlpackage"));
        let _ = std::fs::remove_dir_all(&pkg_path);

        build_mlpackage(
            &pkg_path,
            weights,
            hidden as u64,
            layers,
            heads as u64,
            bucket as u64,
        )?;

        let pkg_url: Id =
            msg_send![class!(NSURL), fileURLWithPath: ns(&pkg_path.to_string_lossy())];
        let mut err: Id = std::ptr::null_mut();
        let compiled_url: Id =
            msg_send![class!(MLModel), compileModelAtURL: pkg_url error: &mut err];
        if compiled_url.is_null() {
            return Err(format!("Core ML compilation failed: {}", err_str(err)));
        }

        // Move to permanent cache location
        let compiled_disk = rs(msg_send![compiled_url, path]);
        let _ = std::fs::remove_dir_all(&compiled_path);
        let _ = std::fs::rename(&compiled_disk, &compiled_path);
        let _ = std::fs::remove_dir_all(&pkg_path);

        let final_url: Id =
            msg_send![class!(NSURL), fileURLWithPath: ns(&compiled_path.to_string_lossy())];
        let model: Id = msg_send![class!(MLModel), modelWithContentsOfURL: final_url configuration: cfg error: &mut err];
        if model.is_null() {
            return Err(format!(
                "Failed to load compiled Core ML model: {}",
                err_str(err)
            ));
        }
        let _: () = msg_send![model, retain];
        let report = inspect_compute_plan(final_url, cfg).unwrap_or_default();
        Ok(CoreMlModel {
            bucket,
            model,
            report,
            archive_hit: false,
        })
    }
}

/// Builds the `.mlpackage` directory with `weights/weight.bin` and `model.mlmodel`.
fn build_mlpackage(
    pkg_path: &Path,
    weights: &Weights,
    c: u64,
    layers: usize,
    heads: u64,
    s: u64,
) -> Result<(), String> {
    let coreml_dir = pkg_path.join("Data").join("com.apple.CoreML");
    let weights_dir = coreml_dir.join("weights");
    std::fs::create_dir_all(&weights_dir).map_err(|e| e.to_string())?;

    let shape = [1, c, 1, s];
    let mask_shape = [1, 1, 1, s];
    let inter = c * 4;
    let d = c / heads; // head dimension (e.g. 32)

    let mut wb = MilBlobWriter::new();
    let weight_file = "@model_path/weights/weight.bin";

    let mut b = vec![];
    bytes(&mut b, 2, b"y");

    // Static constants
    b.extend(int32_const("strides", &[2], &[1, 1]));
    b.extend(int32_const("dil", &[2], &[1, 1]));
    b.extend(int32_const("pad", &[4], &[0, 0, 0, 0]));
    b.extend(int32_const("groups", &[], &[1]));
    b.extend(str_const("pad_type", "valid"));
    b.extend(str_const("exact", "EXACT"));
    b.extend(int32_const("ln_axes", &[1], &[1])); // channel axis
    b.extend(fp16_const("ln_eps", 1e-5)); // FP16 epsilon
    b.extend(fp16_const("attn_scale", 1.0 / (d as f32).sqrt()));
    b.extend(int32_const("softmax_axis", &[], &[-1]));
    b.extend(bool_const("fals", false));
    b.extend(bool_const("tru", true));

    // Reshape & transpose permutation vectors
    // Multi-head shapes: [1, H, S, D]
    b.extend(int32_const(
        "shape_head_sd",
        &[4],
        &[1, heads as i32, s as i32, d as i32],
    ));
    // Permutation to go [1, H, D, S] -> [1, H, S, D]
    b.extend(int32_const("perm_ds_to_sd", &[4], &[0, 1, 3, 2]));
    // Permutation to go [1, H, S, D] -> [1, H, D, S]
    b.extend(int32_const("perm_sd_to_ds", &[4], &[0, 1, 3, 2]));
    // Flatten back to [1, C, 1, S]
    b.extend(int32_const(
        "shape_c_1_s",
        &[4],
        &[1, c as i32, 1, s as i32],
    ));
    b.extend(int32_const(
        "shape_head_ds",
        &[4],
        &[1, heads as i32, d as i32, s as i32],
    ));

    let mut prev = "x".to_string();

    let get_weight = |name: &str| -> Result<&[f32], String> {
        let t = weights
            .get(name)
            .ok_or_else(|| format!("missing weight '{name}'"))?;
        t.as_f32().map_err(|e| format!("weight '{name}': {e}"))
    };

    for l in 0..layers {
        let p = format!("encoder.layer.{l}");

        // 1. Q, K, V 1x1 convs
        let qw_data = get_weight(&format!("{p}.attention.self.query.weight"))?;
        let qb_data = get_weight(&format!("{p}.attention.self.query.bias"))?;
        let kw_data = get_weight(&format!("{p}.attention.self.key.weight"))?;
        let kb_data = get_weight(&format!("{p}.attention.self.key.bias"))?;
        let vw_data = get_weight(&format!("{p}.attention.self.value.weight"))?;
        let vb_data = get_weight(&format!("{p}.attention.self.value.bias"))?;

        let q_off = wb.write_fp16(qw_data);
        let qb_off = wb.write_fp16(qb_data);
        let k_off = wb.write_fp16(kw_data);
        let kb_off = wb.write_fp16(kb_data);
        let v_off = wb.write_fp16(vw_data);
        let vb_off = wb.write_fp16(vb_data);

        let qw_node = format!("qw_{l}");
        let qb_node = format!("qb_{l}");
        let kw_node = format!("kw_{l}");
        let kb_node = format!("kb_{l}");
        let vw_node = format!("vw_{l}");
        let vb_node = format!("vb_{l}");

        b.extend(blob_const_op(
            &qw_node,
            weight_file,
            q_off,
            &[c, c, 1, 1],
            FP16,
        ));
        b.extend(blob_const_op(&qb_node, weight_file, qb_off, &[c], FP16));
        b.extend(blob_const_op(
            &kw_node,
            weight_file,
            k_off,
            &[c, c, 1, 1],
            FP16,
        ));
        b.extend(blob_const_op(&kb_node, weight_file, kb_off, &[c], FP16));
        b.extend(blob_const_op(
            &vw_node,
            weight_file,
            v_off,
            &[c, c, 1, 1],
            FP16,
        ));
        b.extend(blob_const_op(&vb_node, weight_file, vb_off, &[c], FP16));

        let q_node = format!("q_{l}");
        let k_node = format!("k_{l}");
        let v_node = format!("v_{l}");

        b.extend(op(
            "conv",
            &[
                ("x", &prev),
                ("weight", &qw_node),
                ("bias", &qb_node),
                ("strides", "strides"),
                ("pad_type", "pad_type"),
                ("pad", "pad"),
                ("dilations", "dil"),
                ("groups", "groups"),
            ],
            &q_node,
            &shape,
            FP16,
        ));

        b.extend(op(
            "conv",
            &[
                ("x", &prev),
                ("weight", &kw_node),
                ("bias", &kb_node),
                ("strides", "strides"),
                ("pad_type", "pad_type"),
                ("pad", "pad"),
                ("dilations", "dil"),
                ("groups", "groups"),
            ],
            &k_node,
            &shape,
            FP16,
        ));

        b.extend(op(
            "conv",
            &[
                ("x", &prev),
                ("weight", &vw_node),
                ("bias", &vb_node),
                ("strides", "strides"),
                ("pad_type", "pad_type"),
                ("pad", "pad"),
                ("dilations", "dil"),
                ("groups", "groups"),
            ],
            &v_node,
            &shape,
            FP16,
        ));

        // 2. Full Multi-Head Attention:
        // Reshape [1, C, 1, S] -> [1, H, D, S] -> transpose to [1, H, S, D]
        let q_ds = format!("q_ds_{l}");
        let k_ds = format!("k_ds_{l}");
        let v_ds = format!("v_ds_{l}");
        b.extend(op(
            "reshape",
            &[("x", &q_node), ("shape", "shape_head_ds")],
            &q_ds,
            &[1, heads, d, s],
            FP16,
        ));
        b.extend(op(
            "reshape",
            &[("x", &k_node), ("shape", "shape_head_ds")],
            &k_ds,
            &[1, heads, d, s],
            FP16,
        ));
        b.extend(op(
            "reshape",
            &[("x", &v_node), ("shape", "shape_head_ds")],
            &v_ds,
            &[1, heads, d, s],
            FP16,
        ));

        let q_sd = format!("q_sd_{l}");
        let k_sd = format!("k_sd_{l}");
        let v_sd = format!("v_sd_{l}");
        b.extend(op(
            "transpose",
            &[("x", &q_ds), ("perm", "perm_ds_to_sd")],
            &q_sd,
            &[1, heads, s, d],
            FP16,
        ));
        b.extend(op(
            "transpose",
            &[("x", &k_ds), ("perm", "perm_ds_to_sd")],
            &k_sd,
            &[1, heads, s, d],
            FP16,
        ));
        b.extend(op(
            "transpose",
            &[("x", &v_ds), ("perm", "perm_ds_to_sd")],
            &v_sd,
            &[1, heads, s, d],
            FP16,
        ));

        // q_scaled = q_sd * (1 / sqrt(d))
        let q_scaled = format!("q_scaled_{l}");
        b.extend(op(
            "mul",
            &[("x", &q_sd), ("y", "attn_scale")],
            &q_scaled,
            &[1, heads, s, d],
            FP16,
        ));

        // Scores = matmul(q_scaled, k_sd^T) -> [1, H, S, S]
        let scores = format!("scores_{l}");
        b.extend(op(
            "matmul",
            &[
                ("x", &q_scaled),
                ("y", &k_sd),
                ("transpose_x", "fals"),
                ("transpose_y", "tru"),
            ],
            &scores,
            &[1, heads, s, s],
            FP16,
        ));

        // Masked scores = scores + mask (mask [1, 1, 1, S] broadcasts across key dimension)
        let masked_scores = format!("masked_scores_{l}");
        b.extend(op(
            "add",
            &[("x", &scores), ("y", "mask")],
            &masked_scores,
            &[1, heads, s, s],
            FP16,
        ));

        // Probabilities = softmax(masked_scores, axis=-1)
        let attn_probs = format!("attn_probs_{l}");
        b.extend(op(
            "softmax",
            &[("x", &masked_scores), ("axis", "softmax_axis")],
            &attn_probs,
            &[1, heads, s, s],
            FP16,
        ));

        // Context = matmul(attn_probs, v_sd) -> [1, H, S, D]
        let ctx_sd = format!("ctx_sd_{l}");
        b.extend(op(
            "matmul",
            &[
                ("x", &attn_probs),
                ("y", &v_sd),
                ("transpose_x", "fals"),
                ("transpose_y", "fals"),
            ],
            &ctx_sd,
            &[1, heads, s, d],
            FP16,
        ));

        // Transpose back to [1, H, D, S] -> reshape to [1, C, 1, S]
        let ctx_ds = format!("ctx_ds_{l}");
        b.extend(op(
            "transpose",
            &[("x", &ctx_sd), ("perm", "perm_sd_to_ds")],
            &ctx_ds,
            &[1, heads, d, s],
            FP16,
        ));

        let ctx_flat = format!("ctx_flat_{l}");
        b.extend(op(
            "reshape",
            &[("x", &ctx_ds), ("shape", "shape_c_1_s")],
            &ctx_flat,
            &shape,
            FP16,
        ));

        // Attention output dense 1x1 conv:
        let out_w_data = get_weight(&format!("{p}.attention.output.dense.weight"))?;
        let out_b_data = get_weight(&format!("{p}.attention.output.dense.bias"))?;
        let out_w_off = wb.write_fp16(out_w_data);
        let out_b_off = wb.write_fp16(out_b_data);

        let out_w_node = format!("out_w_{l}");
        let out_b_node = format!("out_b_{l}");
        b.extend(blob_const_op(
            &out_w_node,
            weight_file,
            out_w_off,
            &[c, c, 1, 1],
            FP16,
        ));
        b.extend(blob_const_op(
            &out_b_node,
            weight_file,
            out_b_off,
            &[c],
            FP16,
        ));

        let attn_proj = format!("attn_proj_{l}");
        b.extend(op(
            "conv",
            &[
                ("x", &ctx_flat),
                ("weight", &out_w_node),
                ("bias", &out_b_node),
                ("strides", "strides"),
                ("pad_type", "pad_type"),
                ("pad", "pad"),
                ("dilations", "dil"),
                ("groups", "groups"),
            ],
            &attn_proj,
            &shape,
            FP16,
        ));

        // Residual + LayerNorm 1
        let res1 = format!("res1_{l}");
        b.extend(op(
            "add",
            &[("x", &prev), ("y", &attn_proj)],
            &res1,
            &shape,
            FP16,
        ));

        let ln1_w_data = get_weight(&format!("{p}.attention.output.LayerNorm.weight"))?;
        let ln1_b_data = get_weight(&format!("{p}.attention.output.LayerNorm.bias"))?;
        let ln1_w_off = wb.write_fp16(ln1_w_data);
        let ln1_b_off = wb.write_fp16(ln1_b_data);

        let ln1_w_node = format!("ln1_w_{l}");
        let ln1_b_node = format!("ln1_b_{l}");
        b.extend(blob_const_op(
            &ln1_w_node,
            weight_file,
            ln1_w_off,
            &[c],
            FP16,
        ));
        b.extend(blob_const_op(
            &ln1_b_node,
            weight_file,
            ln1_b_off,
            &[c],
            FP16,
        ));

        let ln1_out = format!("ln1_out_{l}");
        b.extend(op(
            "layer_norm",
            &[
                ("x", &res1),
                ("gamma", &ln1_w_node),
                ("beta", &ln1_b_node),
                ("axes", "ln_axes"),
                ("epsilon", "ln_eps"),
            ],
            &ln1_out,
            &shape,
            FP16,
        ));

        // MLP: intermediate dense [c -> inter]
        let ffn1_w_data = get_weight(&format!("{p}.intermediate.dense.weight"))?;
        let ffn1_b_data = get_weight(&format!("{p}.intermediate.dense.bias"))?;
        let ffn1_w_off = wb.write_fp16(ffn1_w_data);
        let ffn1_b_off = wb.write_fp16(ffn1_b_data);

        let ffn1_w_node = format!("ffn1_w_{l}");
        let ffn1_b_node = format!("ffn1_b_{l}");
        b.extend(blob_const_op(
            &ffn1_w_node,
            weight_file,
            ffn1_w_off,
            &[inter, c, 1, 1],
            FP16,
        ));
        b.extend(blob_const_op(
            &ffn1_b_node,
            weight_file,
            ffn1_b_off,
            &[inter],
            FP16,
        ));

        let ffn1_shape = [1, inter, 1, s];
        let ffn1_conv = format!("ffn1_conv_{l}");
        b.extend(op(
            "conv",
            &[
                ("x", &ln1_out),
                ("weight", &ffn1_w_node),
                ("bias", &ffn1_b_node),
                ("strides", "strides"),
                ("pad_type", "pad_type"),
                ("pad", "pad"),
                ("dilations", "dil"),
                ("groups", "groups"),
            ],
            &ffn1_conv,
            &ffn1_shape,
            FP16,
        ));

        // Exact GELU
        let gelu_out = format!("gelu_out_{l}");
        b.extend(op(
            "gelu",
            &[("x", &ffn1_conv), ("mode", "exact")],
            &gelu_out,
            &ffn1_shape,
            FP16,
        ));

        // MLP: output dense [inter -> c]
        let ffn2_w_data = get_weight(&format!("{p}.output.dense.weight"))?;
        let ffn2_b_data = get_weight(&format!("{p}.output.dense.bias"))?;
        let ffn2_w_off = wb.write_fp16(ffn2_w_data);
        let ffn2_b_off = wb.write_fp16(ffn2_b_data);

        let ffn2_w_node = format!("ffn2_w_{l}");
        let ffn2_b_node = format!("ffn2_b_{l}");
        b.extend(blob_const_op(
            &ffn2_w_node,
            weight_file,
            ffn2_w_off,
            &[c, inter, 1, 1],
            FP16,
        ));
        b.extend(blob_const_op(
            &ffn2_b_node,
            weight_file,
            ffn2_b_off,
            &[c],
            FP16,
        ));

        let ffn2_conv = format!("ffn2_conv_{l}");
        b.extend(op(
            "conv",
            &[
                ("x", &gelu_out),
                ("weight", &ffn2_w_node),
                ("bias", &ffn2_b_node),
                ("strides", "strides"),
                ("pad_type", "pad_type"),
                ("pad", "pad"),
                ("dilations", "dil"),
                ("groups", "groups"),
            ],
            &ffn2_conv,
            &shape,
            FP16,
        ));

        // Residual + LayerNorm 2
        let res2 = format!("res2_{l}");
        b.extend(op(
            "add",
            &[("x", &ln1_out), ("y", &ffn2_conv)],
            &res2,
            &shape,
            FP16,
        ));

        let ln2_w_data = get_weight(&format!("{p}.output.LayerNorm.weight"))?;
        let ln2_b_data = get_weight(&format!("{p}.output.LayerNorm.bias"))?;
        let ln2_w_off = wb.write_fp16(ln2_w_data);
        let ln2_b_off = wb.write_fp16(ln2_b_data);

        let ln2_w_node = format!("ln2_w_{l}");
        let ln2_b_node = format!("ln2_b_{l}");
        b.extend(blob_const_op(
            &ln2_w_node,
            weight_file,
            ln2_w_off,
            &[c],
            FP16,
        ));
        b.extend(blob_const_op(
            &ln2_b_node,
            weight_file,
            ln2_b_off,
            &[c],
            FP16,
        ));

        let ln2_out = if l + 1 == layers {
            "y".to_string()
        } else {
            format!("ln2_out_{l}")
        };

        b.extend(op(
            "layer_norm",
            &[
                ("x", &res2),
                ("gamma", &ln2_w_node),
                ("beta", &ln2_b_node),
                ("axes", "ln_axes"),
                ("epsilon", "ln_eps"),
            ],
            &ln2_out,
            &shape,
            FP16,
        ));

        prev = ln2_out;
    }

    let function = {
        let mut f = named(1, "x", FP16, &shape);
        f.extend(named(1, "mask", FP16, &mask_shape));
        bytes(&mut f, 2, b"CoreML7");
        f.extend(map_entry(3, "CoreML7", b));
        f
    };

    let program = {
        let mut p = vec![];
        int(&mut p, 1, 1);
        p.extend(map_entry(2, "main", function));
        p
    };

    let mut model = vec![];
    int(&mut model, 1, 8); // specificationVersion = 8 (iOS 17+ / macOS 14+)
    model.extend(msg(2, |d| {
        d.extend(feature(1, "x", &shape, 65552)); // FP16
        d.extend(feature(1, "mask", &mask_shape, 65552)); // FP16
        d.extend(feature(10, "y", &shape, 65552)); // FP16
    }));
    bytes(&mut model, 502, &program);

    // Write weight.bin
    let blob_bytes = wb.finish();
    std::fs::write(weights_dir.join("weight.bin"), blob_bytes).map_err(|e| e.to_string())?;

    // Write model.mlmodel
    std::fs::write(coreml_dir.join("model.mlmodel"), model).map_err(|e| e.to_string())?;

    // Write Manifest.json
    let manifest = r#"{"fileFormatVersion":"1.0.0","itemInfoEntries":{"m":{"author":"com.apple.CoreML","description":"CoreML Model Specification","name":"model.mlmodel","path":"com.apple.CoreML/model.mlmodel"},"w":{"author":"com.apple.CoreML","description":"CoreML Model Weights","name":"weights","path":"com.apple.CoreML/weights"}},"rootModelIdentifier":"m"}"#;
    std::fs::write(pkg_path.join("Manifest.json"), manifest).map_err(|e| e.to_string())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Inference prediction bridge
// ---------------------------------------------------------------------------

unsafe fn predict_multi_array(
    model: Id,
    x: &[f32],
    mask: &[f32],
    c: usize,
    s: usize,
) -> Result<Vec<f32>, String> {
    metal::objc::rc::autoreleasepool(|| {
        let mut err: Id = std::ptr::null_mut();

        let arr_x: Id = msg_send![class!(MLMultiArray), alloc];
        let arr_x: Id = msg_send![arr_x, initWithShape: nsnum_array(&[1, c as i64, 1, s as i64]) dataType: 65552i64 error: &mut err];
        if arr_x.is_null() {
            return Err(format!("MLMultiArray alloc for x failed: {}", err_str(err)));
        }
        let px: *mut u16 = msg_send![arr_x, dataPointer];
        for (i, &v) in x.iter().enumerate() {
            *px.add(i) = f32_to_f16(v);
        }

        let arr_m: Id = msg_send!(class!(MLMultiArray), alloc);
        let arr_m: Id = msg_send![arr_m, initWithShape: nsnum_array(&[1, 1, 1, s as i64]) dataType: 65552i64 error: &mut err];
        if arr_m.is_null() {
            let _: () = msg_send![arr_x, release];
            return Err(format!(
                "MLMultiArray alloc for mask failed: {}",
                err_str(err)
            ));
        }
        let pm: *mut u16 = msg_send![arr_m, dataPointer];
        for (i, &v) in mask.iter().enumerate() {
            *pm.add(i) = f32_to_f16(v);
        }

        let fv_x: Id = msg_send![class!(MLFeatureValue), featureValueWithMultiArray: arr_x];
        let fv_m: Id = msg_send![class!(MLFeatureValue), featureValueWithMultiArray: arr_m];
        let keys = [ns("x"), ns("mask")];
        let vals = [fv_x, fv_m];
        let dict: Id = msg_send![class!(NSDictionary), dictionaryWithObjects: vals.as_ptr() forKeys: keys.as_ptr() count: 2usize];
        let prov: Id = msg_send![class!(MLDictionaryFeatureProvider), alloc];
        let prov: Id = msg_send![prov, initWithDictionary: dict error: &mut err];
        if prov.is_null() {
            let _: () = msg_send![arr_x, release];
            let _: () = msg_send![arr_m, release];
            return Err(format!(
                "MLDictionaryFeatureProvider failed: {}",
                err_str(err)
            ));
        }

        let out: Id = msg_send![model, predictionFromFeatures: prov error: &mut err];
        let _: () = msg_send![prov, release];
        let _: () = msg_send![arr_x, release];
        let _: () = msg_send![arr_m, release];

        if out.is_null() {
            return Err(format!("Prediction failed: {}", err_str(err)));
        }

        let y: Id = msg_send![out, featureValueForName: ns("y")];
        if y.is_null() {
            return Err("Output feature 'y' not found".into());
        }
        let ya: Id = msg_send![y, multiArrayValue];
        let strides: Id = msg_send![ya, strides];
        let s1: Id = msg_send![strides, objectAtIndex: 1usize];
        let s3: Id = msg_send![strides, objectAtIndex: 3usize];
        let ch_stride: usize = msg_send![s1, longLongValue];
        let seq_stride: usize = msg_send![s3, longLongValue];

        let yp: *const u16 = msg_send![ya, dataPointer];

        let mut res = vec![0.0f32; c * s];
        for ch in 0..c {
            for seq in 0..s {
                let off = ch * ch_stride + seq * seq_stride;
                res[ch * s + seq] = f16_to_f32(*yp.add(off));
            }
        }

        Ok(res)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Tensor;
    #[test]
    fn test_mil_blob_writer_structure() {
        let mut wb = MilBlobWriter::new();
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![0.5f32; 16];
        let off1 = wb.write_fp16(&data1);
        let off2 = wb.write_fp16(&data2);

        assert_eq!(off1 % 64, 0);
        assert_eq!(off2 % 64, 0);
        assert!(off2 > off1);

        let bytes = wb.finish();
        assert_eq!(bytes.len() % 64, 0);
        // Header checks
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 2); // count = 2
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2); // version = 2

        // Metadata 1 sentinel
        let meta1 = off1 as usize;
        assert_eq!(
            u32::from_le_bytes(bytes[meta1..meta1 + 4].try_into().unwrap()),
            0xDEADBEEF
        );
        assert_eq!(
            u32::from_le_bytes(bytes[meta1 + 4..meta1 + 8].try_into().unwrap()),
            1
        ); // fp16

        // Metadata 2 sentinel
        let meta2 = off2 as usize;
        assert_eq!(
            u32::from_le_bytes(bytes[meta2..meta2 + 4].try_into().unwrap()),
            0xDEADBEEF
        );
        assert_eq!(
            u32::from_le_bytes(bytes[meta2 + 4..meta2 + 8].try_into().unwrap()),
            1
        ); // fp16
    }

    #[test]
    fn test_fp16_conversions() {
        for val in [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 1e-4, 0.12345] {
            let h = f32_to_f16(val);
            let back = f16_to_f32(h);
            let rel_err = (val - back).abs() / val.abs().max(1.0);
            assert!(
                rel_err < 1e-3,
                "val {val} -> back {back} (rel_err {rel_err})"
            );
        }
        // Subnormal test
        let sub = 1e-6f32;
        let h = f32_to_f16(sub);
        let back = f16_to_f32(h);
        assert!((sub - back).abs() < 1e-6);
    }

    #[test]
    fn second_load_restores_from_cache() {
        let dir = tempfile::tempdir().unwrap();
        let c = 384;
        let inter = c * 4;
        let heads = 12;
        let mut w = Weights::new();
        let p = "encoder.layer.0";
        for name in [
            format!("{p}.attention.self.query.weight"),
            format!("{p}.attention.self.key.weight"),
            format!("{p}.attention.self.value.weight"),
            format!("{p}.attention.output.dense.weight"),
        ] {
            w.insert(name, Tensor::F32(vec![0.01; c * c]));
        }
        for name in [
            format!("{p}.attention.self.query.bias"),
            format!("{p}.attention.self.key.bias"),
            format!("{p}.attention.self.value.bias"),
            format!("{p}.attention.output.dense.bias"),
            format!("{p}.attention.output.LayerNorm.weight"),
            format!("{p}.attention.output.LayerNorm.bias"),
            format!("{p}.output.dense.bias"),
            format!("{p}.output.LayerNorm.weight"),
            format!("{p}.output.LayerNorm.bias"),
        ] {
            w.insert(name, Tensor::F32(vec![0.0; c]));
        }
        w.insert(
            format!("{p}.intermediate.dense.weight"),
            Tensor::F32(vec![0.01; inter * c]),
        );
        w.insert(
            format!("{p}.intermediate.dense.bias"),
            Tensor::F32(vec![0.0; inter]),
        );
        w.insert(
            format!("{p}.output.dense.weight"),
            Tensor::F32(vec![0.01; c * inter]),
        );

        let s1 = Session::new(dir.path(), &w, c, 1, heads).unwrap();
        assert!(
            !s1.archive_hit,
            "first load should be compiled: {}",
            s1.report
        );

        let s2 = Session::new(dir.path(), &w, c, 1, heads).unwrap();
        assert!(
            s2.archive_hit,
            "second load should be cache hit: {}",
            s2.report
        );
        for n in [1, 16, 17, 128, 256, 300] {
            let b = s2.bucket_for(n).unwrap();
            assert!(b >= n && s2.active_buckets.contains(&b), "n={n} -> b={b}");
            assert!(
                s2.bucket_placements[&b].ane_share() >= 50.0,
                "bucket {b} must have >= 50% ANE share"
            );
        }
        assert!(s2.bucket_for(600).is_err());
    }

    #[test]
    fn missing_weights_return_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = Session::new(dir.path(), &Weights::new(), 64, 1, 2)
            .err()
            .expect("should return Err");
        assert!(err.contains("missing weight"), "{err}");
    }
}
