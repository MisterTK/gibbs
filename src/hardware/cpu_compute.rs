//! GPU-only tripwire: once the process resolves to a GPU backend, any CPU compute entry point that
//! runs without an [`allow`] guard panics instead of silently computing on the CPU.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static BAN: AtomicBool = AtomicBool::new(false);
static ALLOW: AtomicUsize = AtomicUsize::new(0);

/// Arms (or disarms) the tripwire.
pub fn ban(on: bool) {
    BAN.store(on, Ordering::SeqCst);
}

/// Whether a GPU backend was requested (the tripwire is armed).
pub fn banned() -> bool {
    BAN.load(Ordering::Relaxed)
}

/// Scope in which CPU compute is permitted (parity references, deliberate `--backend cpu` paths).
pub struct AllowGuard(());

impl Drop for AllowGuard {
    fn drop(&mut self) {
        ALLOW.fetch_sub(1, Ordering::SeqCst);
    }
}

pub fn allow() -> AllowGuard {
    ALLOW.fetch_add(1, Ordering::SeqCst);
    AllowGuard(())
}

/// Called at the top of every CPU compute entry point.
pub fn enter(site: &'static str) {
    if BAN.load(Ordering::Relaxed) && ALLOW.load(Ordering::SeqCst) == 0 {
        panic!("gpu-only violation: CPU compute entered at {site}; a code path bypassed the family gate");
    }
}
