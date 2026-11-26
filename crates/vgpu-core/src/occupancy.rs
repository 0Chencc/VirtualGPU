//! CTAs resident per SM, from the profile's allocation rules.

use crate::kernel::Kernel;
use crate::profile::Specs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limiter {
    Warps,
    Threads,
    Blocks,
    Registers,
    SharedMemory,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Occupancy {
    pub ctas_per_sm: u32,
    pub limiter: Limiter,
    /// Smallest carveout that holds the resident CTAs' shared memory.
    pub carveout_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OccupancyError {
    TooManyThreads {
        threads: u32,
        max: u32,
    },
    TooManyRegisters {
        per_thread: u32,
        max: u32,
    },
    TooMuchSharedMemory {
        bytes: u64,
        max: u64,
    },
    TooMuchOnChipBuffer {
        buffer: &'static str,
        bytes: u64,
        max: u64,
    },
    DoesNotFit,
}

fn round_up(x: u64, unit: u64) -> u64 {
    x.div_ceil(unit) * unit
}

pub fn occupancy(k: &Kernel, s: &Specs) -> Result<Occupancy, OccupancyError> {
    let threads = k.num_warps * s.warp_size;
    if threads > s.max_threads_per_block {
        return Err(OccupancyError::TooManyThreads {
            threads,
            max: s.max_threads_per_block,
        });
    }
    if k.registers_per_thread > s.max_registers_per_thread {
        return Err(OccupancyError::TooManyRegisters {
            per_thread: k.registers_per_thread,
            max: s.max_registers_per_thread,
        });
    }
    if k.smem_per_cta_bytes > s.max_shared_mem_per_block_bytes {
        return Err(OccupancyError::TooMuchSharedMemory {
            bytes: k.smem_per_cta_bytes,
            max: s.max_shared_mem_per_block_bytes,
        });
    }
    if let Some(asc) = &s.ascend {
        if k.ub_bytes > asc.ub_bytes {
            return Err(OccupancyError::TooMuchOnChipBuffer {
                buffer: "ub",
                bytes: k.ub_bytes,
                max: asc.ub_bytes,
            });
        }
        if k.l0c_bytes > asc.l0c_bytes {
            return Err(OccupancyError::TooMuchOnChipBuffer {
                buffer: "l0c",
                bytes: k.l0c_bytes,
                max: asc.l0c_bytes,
            });
        }
    }

    let regs_per_warp = round_up(
        u64::from(k.registers_per_thread.max(1)) * u64::from(s.warp_size),
        u64::from(s.register_alloc_unit_regs),
    );
    let warps_by_regs = u64::from(s.registers_per_sm) / regs_per_warp;

    let smem_per_cta = round_up(
        k.smem_per_cta_bytes + s.reserved_shared_mem_per_cta_bytes,
        s.shared_mem_alloc_granularity_bytes,
    );
    let smem_capacity = s
        .shared_mem_carveout_options_bytes
        .iter()
        .copied()
        .max()
        .unwrap_or(s.shared_mem_per_sm_bytes)
        .min(s.shared_mem_per_sm_bytes);
    let ctas_by_smem = smem_capacity
        .checked_div(smem_per_cta)
        .unwrap_or(u64::from(s.max_blocks_per_sm));

    let limits = [
        (Limiter::Warps, u64::from(s.max_warps_per_sm / k.num_warps)),
        (Limiter::Threads, u64::from(s.max_threads_per_sm / threads)),
        (Limiter::Blocks, u64::from(s.max_blocks_per_sm)),
        (Limiter::Registers, warps_by_regs / u64::from(k.num_warps)),
        (Limiter::SharedMemory, ctas_by_smem),
    ];
    let (limiter, ctas) = limits.iter().copied().min_by_key(|l| l.1).unwrap();
    if ctas == 0 {
        return Err(OccupancyError::DoesNotFit);
    }

    let needed = ctas * smem_per_cta;
    let carveout_bytes = s
        .shared_mem_carveout_options_bytes
        .iter()
        .copied()
        .filter(|&c| c >= needed)
        .min()
        .unwrap_or(smem_capacity);

    Ok(Occupancy {
        ctas_per_sm: ctas as u32,
        limiter,
        carveout_bytes,
    })
}
