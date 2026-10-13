//! Cost model for fusion decisions (plan F7; XLA analogue: `PriorityFusion` +
//! `GpuPerformanceModel::EstimateRunTimes`, `xla-catalog.md`): a fusion is applied only when its
//! modelled benefit is positive.
//!
//! `benefit = removed_dispatches * dispatch_us + removed_bytes / ceiling - extra_us`, where
//! `dispatch_us` is the device's measured cost of one dispatch boundary (`HostProfile::dispatch_us`,
//! an empty kernel back to back), `ceiling` the measured streaming rate, and `extra_us` the time the
//! fused kernel adds (recomputed work, a serialised tail).

/// Device constants of the model.
#[derive(Debug, Clone, Copy)]
pub struct FusionCost {
    pub dispatch_us: f64,
    pub ceiling_gbps: f64,
}

/// Single-threadgroup streaming rate assumed for work one threadgroup does alone (a fused merge tail).
/// [INFERENCE] ~10 GB/s for one threadgroup of 4 simdgroups; `F8` replaces it by a measurement.
pub const TG_ALONE_GBPS: f64 = 10.0;

impl FusionCost {
    pub fn benefit_us(&self, removed_dispatches: u32, removed_bytes: u64, extra_us: f64) -> f64 {
        removed_dispatches as f64 * self.dispatch_us
            + removed_bytes as f64 / (self.ceiling_gbps.max(1e-9) * 1e3)
            - extra_us
    }

    /// Time one threadgroup needs to read `bytes` alone.
    pub fn tg_alone_us(bytes: u64) -> f64 {
        bytes as f64 / (TG_ALONE_GBPS * 1e3)
    }

    /// Batching `n` independent same-kind steps into one dispatch.
    pub fn batch_wins(&self, n: usize) -> bool {
        n >= 2 && self.benefit_us(n as u32 - 1, 0, 0.0) > 0.0
    }

    /// Merging split-attention partials in the last threadgroup of a head tile instead of a second
    /// dispatch: the tile's `heads` heads each read `splits` partials of `head_dim + 2` floats serially.
    pub fn merge_in_kernel_wins(&self, heads: usize, splits: usize, head_dim: usize) -> bool {
        let tail = Self::tg_alone_us((heads * splits * (head_dim + 2) * 4) as u64);
        self.benefit_us(1, 0, tail) > 0.0
    }

    /// Storing the decode K/V row from the attention kernel instead of a separate append dispatch.
    pub fn append_in_kernel_wins(&self) -> bool {
        self.benefit_us(1, 0, 0.0) > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: FusionCost = FusionCost {
        dispatch_us: 1.2,
        ceiling_gbps: 300.0,
    };

    #[test]
    fn benefit_counts_dispatches_and_bytes_minus_extra_work() {
        // 2 dispatches = 2.4 us; 300 KB at 300 GB/s = 1 us; extra 0.5 us.
        let b = C.benefit_us(2, 300_000, 0.5);
        assert!((b - 2.9).abs() < 1e-9, "{b}");
    }

    #[test]
    fn batching_needs_two_steps_and_a_positive_dispatch_cost() {
        assert!(!C.batch_wins(1));
        assert!(C.batch_wins(2));
        assert!(!FusionCost {
            dispatch_us: 0.0,
            ..C
        }
        .batch_wins(3));
    }

    #[test]
    fn merge_tail_must_cost_less_than_a_dispatch() {
        // E2B sliding layer: 4 heads x 3 splits x (256 + 2) floats = 12.4 KB -> 1.24 us at 10 GB/s > 1.2 us.
        assert!(!C.merge_in_kernel_wins(4, 3, 256));
        // Llama-1B style: 4 heads x 2 splits x (64 + 2) floats = 2.1 KB -> 0.21 us.
        assert!(C.merge_in_kernel_wins(4, 2, 64));
    }
}
