pub mod metal;

#[cfg(target_os = "macos")]
pub use metal::MetalBackend;
