pub use crate as foundry_engine;

/// Apple Metal GPU backend (macOS).
pub mod backend;
pub use backend::*;

/// Model architectures as JSON5 specs (`arch/`): registry, expressions, loader, interpreter.
pub mod arch;

pub mod cluster;
pub mod graph;

/// Hardware acceleration targets, memory hierarchies, and device topologies.
pub mod hardware;
pub use hardware::*;

pub mod kv;

/// Architecture-agnostic model ingestion, dynamic fingerprinting, weights loading, and IR representations.
pub mod model;
pub use model::*;

pub mod bench;
pub mod quant;
pub mod scheduler;
pub mod server;
pub mod telemetry;
pub mod tui;
