//! The kernel description the model core simulates.
//!
//! This is the core's own, already-resolved view of one kernel variant. The
//! KernelIR adapter fills it in; tests build it directly.

/// One global memory access stream of a phase, identified by the KernelIR op.
#[derive(Debug, Clone, PartialEq)]
pub struct Access {
    pub op_id: String,
    /// Unmasked bytes per CTA per step (per iteration, or per phase execution).
    pub nominal_bytes: f64,
}

/// Work done by one CTA in one pipeline iteration, or in one execution of the
/// prologue or epilogue.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Work {
    /// Tensor-core floating-point operations.
    pub mma_flops: f64,
    /// Non-MMA instructions, per warp.
    pub alu_insts_per_warp: f64,
    /// Shared memory bytes moved by the CTA.
    pub smem_bytes: f64,
    pub loads: Vec<Access>,
    pub stores: Vec<Access>,
}

/// What one CTA does for one output tile.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    pub prologue: Work,
    pub iterations: u32,
    pub per_iteration: Work,
    /// Iterations in flight in the software pipeline. Zero means synchronous
    /// loads: no overlap of loads and compute within the CTA. The prologue's
    /// own loads are listed explicitly in `prologue`.
    pub prefetch_distance: u32,
    pub epilogue: Work,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TripCounts {
    Uniform(u32),
    PerProgram(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    /// Number of programs (CTAs) launched.
    pub grid: u32,
    pub num_warps: u32,
    pub registers_per_thread: u32,
    /// Static plus dynamic shared memory per CTA.
    pub smem_per_cta_bytes: u64,
    /// Tiles processed by each program: one for ordinary kernels, more for
    /// persistent kernels.
    pub tiles_per_program: TripCounts,
    /// Profile key of the MMA instruction, `<op_class>:<a>:<b>:<acc>:<MxNxK>`.
    pub mma_key: String,
    pub tile: Tile,
}

impl Kernel {
    pub fn tiles_of(&self, program: u32) -> u32 {
        match &self.tiles_per_program {
            TripCounts::Uniform(n) => *n,
            TripCounts::PerProgram(v) => v[program as usize],
        }
    }
}
