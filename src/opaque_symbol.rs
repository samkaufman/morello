use std::sync::atomic::{self, AtomicU64};

static OPAQUE_SYMBOL_NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// An object equal only to itself (or its clones).
///
/// Internally, a unique integer is generated for each instance and used for equality comparison.
#[derive(Debug, PartialEq, Hash, Eq, Clone)]
pub struct OpaqueSymbol(u64);

impl OpaqueSymbol {
    pub fn new() -> Self {
        Self(OPAQUE_SYMBOL_NEXT_ID.fetch_add(1, atomic::Ordering::SeqCst))
    }
}
