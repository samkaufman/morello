pub mod action_seq;
pub mod codegen;
pub mod color;
pub mod common;
pub mod cost;
pub mod datadeps;
pub mod db;
pub mod expr;
pub mod grid;
pub mod imp;
pub mod layout;
pub mod memorylimits;
pub mod nameenv;
pub mod opaque_symbol;
pub mod pprint;
mod reconstruct;
mod rtree;
pub mod scheduling;
pub mod scheduling_sugar;
pub mod search;
pub mod spec;
pub mod target;
pub mod tensorspec;
pub mod tiling;
pub mod utils;
#[cfg(feature = "verification")]
pub mod verification;
pub mod views;

// Temporarily-export smallvec dependency
// TODO: Wrap the Shape type instead.
pub use smallvec;
