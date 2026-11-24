//! VirtualGPU model core.

/// Git commit the binary was built from.
pub const BUILD_COMMIT: &str = env!("VGPU_BUILD_COMMIT");
/// Whether the code compiled into the binary had uncommitted changes. Results
/// from such a build are diagnostic only.
pub const BUILD_DIRTY: bool = matches!(env!("VGPU_BUILD_DIRTY").as_bytes(), b"true");

pub mod draws;
pub mod from_ir;
pub mod ir;
pub mod kernel;
pub mod l2;
pub mod occupancy;
pub mod profile;
pub mod sim;
pub mod verdict;

#[cfg(feature = "python")]
mod python;
