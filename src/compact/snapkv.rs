//! Tier 4 Physical KV Cache Compaction (`SnapKV` / `H2O` Heavy-Hitter + Attention Sink Eviction).
//!
//! Compacts `kv.{layer}.k` and `kv.{layer}.v` buffers inside a [`StateStore`] from `current_len`
//! positions down to `target_budget` positions without re-prefilling:
//! 1. Retains the first `sink_tokens` positions (`0..sinks`, StreamingLLM attention sinks).
//! 2. Retains the most recent `recent_window` positions (`current_len - recent..current_len`).
//! 3. Scores middle positions `sinks..(current_len - recent)` by pooled key/value L2 salience
//!    across all layers, selects the top `target_budget - sinks - recent` positions in
//!    chronological order, and packs their rows contiguously into `0..target_budget`.

use crate::graph::{StateStore, Tensor};

/// Default number of initial attention sink tokens preserved unconditionally.
pub const DEFAULT_SINK_TOKENS: usize = 4;

/// Compacts all `kv.*` buffers in `state` from `current_len` positions to `target_budget` positions.
/// Returns the new sequence length in `state` (`<= current_len`).
pub fn evict_kv_state(
    state: &mut StateStore,
    current_len: usize,
    target_budget: usize,
    sink_tokens: usize,
    recent_window: usize,
) -> usize {
    if current_len <= target_budget || target_budget == 0 {
        return current_len;
    }

    let sinks = sink_tokens.min(target_budget / 2).min(current_len);
    let recent = recent_window
        .min(target_budget.saturating_sub(sinks))
        .min(current_len.saturating_sub(sinks));
    let middle_budget = target_budget.saturating_sub(sinks + recent);
    let mid_start = sinks;
    let mid_end = current_len.saturating_sub(recent);

    // Compute per-position salience scores across all KV buffers for middle positions.
    let mid_len = mid_end.saturating_sub(mid_start);
    let mut raw_scores = vec![0.0f32; mid_len];

    for (name, tensor) in &state.buffers {
        if !name.starts_with("kv.") {
            continue;
        }
        if let Tensor::F32(buf) = tensor {
            let row_width = buf.len() / current_len.max(1);
            if row_width == 0 {
                continue;
            }
            for (idx, pos) in (mid_start..mid_end).enumerate() {
                let row = &buf[pos * row_width..(pos + 1) * row_width];
                let norm_sq: f32 = row.iter().map(|&x| x * x).sum();
                raw_scores[idx] += norm_sq.sqrt();
            }
        }
    }

    #[cfg(target_os = "macos")]
    for name in state.metal_buffers.keys() {
        if !name.starts_with("kv.") {
            continue;
        }
        if let Some(buf) = state.read_metal(name) {
            let row_width = buf.len() / current_len.max(1);
            if row_width == 0 {
                continue;
            }
            for (idx, pos) in (mid_start..mid_end).enumerate() {
                if (pos + 1) * row_width <= buf.len() {
                    let row = &buf[pos * row_width..(pos + 1) * row_width];
                    let norm_sq: f32 = row.iter().map(|&x| x * x).sum();
                    raw_scores[idx] += norm_sq.sqrt();
                }
            }
        }
    }

    // SnapKV 1D avg-pooling (kernel size 3) over middle scores to preserve local token clusters.
    let mut pooled = vec![0.0f32; mid_len];
    for i in 0..mid_len {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(mid_len.saturating_sub(1));
        let sum: f32 = raw_scores[lo..=hi].iter().sum();
        pooled[i] = sum / ((hi - lo + 1) as f32);
    }

    let mut ranked: Vec<usize> = (0..mid_len).collect();
    ranked.sort_by(|&a, &b| pooled[b].total_cmp(&pooled[a]).then_with(|| a.cmp(&b)));
    ranked.truncate(middle_budget);
    ranked.sort_unstable();

    // Build chronological index list of kept positions (length == target_budget).
    let mut kept_positions = Vec::with_capacity(target_budget);
    kept_positions.extend(0..sinks);
    kept_positions.extend(ranked.into_iter().map(|i| mid_start + i));
    kept_positions.extend(mid_end..current_len);
    let new_len = kept_positions.len();

    // Compact CPU KV buffers in-place.
    for (name, tensor) in state.buffers.iter_mut() {
        if !name.starts_with("kv.") {
            continue;
        }
        if let Tensor::F32(buf) = tensor {
            // Determine row width: a KV buffer may be sized to `current_len` or to the model's max context.
            // When sized to `current_len` (or divisible by `current_len`), compact rows `kept_positions` to `0..new_len`.
            if buf.len() >= current_len && buf.len() % current_len == 0 {
                let row_width = buf.len() / current_len;
                for (dst_idx, &src_pos) in kept_positions.iter().enumerate() {
                    if dst_idx != src_pos {
                        buf.copy_within(
                            src_pos * row_width..(src_pos + 1) * row_width,
                            dst_idx * row_width,
                        );
                    }
                }
                buf.truncate(new_len * row_width);
            }
        }
    }

    // Compact Metal KV buffers in-place when present.
    #[cfg(target_os = "macos")]
    {
        let kv_names: Vec<String> = state
            .metal_buffers
            .keys()
            .filter(|k| k.starts_with("kv."))
            .cloned()
            .collect();
        for name in kv_names {
            let f16 = state.metal_f16.contains(&name);
            if let Some((buf, total_elems)) = state.metal_buffers.get(&name) {
                if *total_elems >= current_len && *total_elems % current_len == 0 {
                    let row_width = *total_elems / current_len;
                    unsafe {
                        if f16 {
                            let slice = std::slice::from_raw_parts_mut(
                                buf.contents() as *mut u16,
                                *total_elems,
                            );
                            for (dst_idx, &src_pos) in kept_positions.iter().enumerate() {
                                if dst_idx != src_pos {
                                    slice.copy_within(
                                        src_pos * row_width..(src_pos + 1) * row_width,
                                        dst_idx * row_width,
                                    );
                                }
                            }
                        } else {
                            let slice = std::slice::from_raw_parts_mut(
                                buf.contents() as *mut f32,
                                *total_elems,
                            );
                            for (dst_idx, &src_pos) in kept_positions.iter().enumerate() {
                                if dst_idx != src_pos {
                                    slice.copy_within(
                                        src_pos * row_width..(src_pos + 1) * row_width,
                                        dst_idx * row_width,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    new_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapkv_eviction_preserves_sinks_recent_window_and_heavy_hitters() {
        let mut store = StateStore::new();
        let current_len = 12usize;
        let row_width = 4usize;
        let mut k_data = vec![0.1f32; current_len * row_width];
        // Make position 5 and position 6 heavy-hitter spikes in the middle (2..9).
        for c in 0..row_width {
            k_data[0 * row_width + c] = 1.0; // sink 0
            k_data[1 * row_width + c] = 2.0; // sink 1
            k_data[5 * row_width + c] = 50.0; // heavy hitter 5
            k_data[6 * row_width + c] = 40.0; // heavy hitter 6
            k_data[10 * row_width + c] = 10.0; // recent 10
            k_data[11 * row_width + c] = 11.0; // recent 11
        }
        store.insert("kv.0.k", Tensor::F32(k_data));

        // Target budget = 6: 2 sinks (0, 1) + 2 middle heavy hitters (5, 6) + 2 recent (10, 11).
        let new_len = evict_kv_state(&mut store, current_len, 6, 2, 2);
        assert_eq!(new_len, 6);
        let compacted = store.get("kv.0.k").unwrap().as_f32().unwrap();
        assert_eq!(compacted.len(), 6 * row_width);
        assert_eq!(compacted[0 * row_width], 1.0);
        assert_eq!(compacted[1 * row_width], 2.0);
        assert_eq!(compacted[2 * row_width], 50.0);
        assert_eq!(compacted[3 * row_width], 40.0);
        assert_eq!(compacted[4 * row_width], 10.0);
        assert_eq!(compacted[5 * row_width], 11.0);
    }
}
