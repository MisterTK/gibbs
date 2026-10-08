//! Intel OneAPI Level Zero stub executor.
//!
//! Level Zero is currently supported only as an explicit placement stub,
//! returning clear runtime errors and reporting unavailable on all platforms.

use super::{DType, Graph, Op, Tensor, Weights};

/// Returns whether Intel Level Zero hardware is available on this system.
pub fn is_available() -> bool {
    false
}

/// Returns whether the Level Zero backend implements `op` for `dtype`.
pub fn supports_op(_op: &Op, _dtype: DType) -> bool {
    false
}

/// Level Zero session stub.
pub struct LevelZeroSession;

impl LevelZeroSession {
    pub fn new(_weights: &Weights) -> Result<Self, String> {
        Err(
            "Intel Level Zero backend is a stub and is not implemented on this platform"
                .to_string(),
        )
    }

    pub fn run(&self, _graph: &Graph, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        Err("Intel Level Zero backend is not implemented".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_zero_is_stub() {
        assert!(!is_available());
        assert!(!supports_op(&Op::Linear, DType::F32));
        assert!(LevelZeroSession::new(&Weights::new()).is_err());
    }
}
