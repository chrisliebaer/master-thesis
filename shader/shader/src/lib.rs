#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]
#![cfg_attr(target_arch = "spirv", no_std)]
#[warn(clippy::needless_range_loop)]
pub mod thesis;
