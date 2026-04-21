// lib.rs — shared module root for selph_core.
//
// The binary (main.rs) and the Python extension (pybridge.rs) both
// depend on these modules. Module declarations live here; main.rs
// uses `selph_core::*` and pybridge uses `crate::*`.

pub mod intern;
pub mod types;
pub mod parser;
pub mod arc;
pub mod types_v2;
pub mod eval_v2;
pub mod synth_v2;
pub mod meta_v2;
pub mod ast_tools;
pub mod kb;

#[cfg(feature = "python")]
pub mod pybridge;
