//! VirtualGPU model core.

pub mod draws;
pub mod from_ir;
pub mod ir;
pub mod kernel;
pub mod l2;
pub mod occupancy;
pub mod profile;
pub mod sim;

#[cfg(feature = "python")]
mod python;
