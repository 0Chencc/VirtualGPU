//! KernelIR v0.1 to the core's kernel description, and the KernelIR-backed
//! L2 model.

use crate::ir::{self, CachePolicy, KernelIR, Op, WarpGroupRole};
use crate::kernel::{Access, Kernel, Tile, TripCounts, Work};
use crate::l2::{L2InitState, L2Model, L2Options, L2Profile};
use crate::profile::Profile;
use crate::sim::{ModelError, L2};
use std::collections::HashMap;

fn unsupported(what: impl Into<String>) -> ModelError {
    ModelError::Unsupported(what.into())
}

fn work(ops: &[Op], mma_key: &mut Option<String>) -> Result<Work, ModelError> {
    let mut w = Work::default();
    for op in ops {
        match op {
            Op::GmemLoad(g) => w.loads.push(Access {
                op_id: g.op_id.clone(),
                nominal_bytes: g.nominal_bytes_per_cta as f64,
            }),
            Op::GmemStore(g) => w.stores.push(Access {
                op_id: g.op_id.clone(),
                nominal_bytes: g.nominal_bytes_per_cta as f64,
            }),
            Op::SmemLoad(s) | Op::SmemStore(s) => w.smem_bytes += s.bytes_per_cta as f64,
            Op::Mma(m) => {
                let key = m.profile_key();
                match mma_key {
                    Some(k) if *k != key => {
                        return Err(unsupported(format!(
                            "mixed MMA instructions: {k} and {key}"
                        )))
                    }
                    _ => *mma_key = Some(key),
                }
                let [mi, ni, ki] = m.inst_mnk.map(f64::from);
                w.mma_flops += 2.0 * mi * ni * ki * m.count as f64;
            }
            Op::Alu(a) | Op::Sfu(a) => w.alu_insts_per_warp += a.count as f64,
            Op::Barrier(_) => {}
        }
    }
    Ok(w)
}

fn u32_of(x: u64, what: &str) -> Result<u32, ModelError> {
    u32::try_from(x).map_err(|_| unsupported(format!("{what} out of range: {x}")))
}

pub fn kernel_from_ir(ir: &KernelIR) -> Result<Kernel, ModelError> {
    if let Some(f) = ir.unsupported_features.first() {
        return Err(unsupported(f.clone()));
    }
    if ir
        .launch
        .warp_groups
        .iter()
        .any(|g| g.role != WarpGroupRole::Unified)
    {
        return Err(unsupported("warp specialization"));
    }
    if ir.launch.cluster_shape != [1, 1, 1] {
        return Err(unsupported("thread block clusters"));
    }
    if ir.body.epilogue.overlaps_next_tile {
        return Err(unsupported("epilogue overlapping the next tile"));
    }
    let iterations = match &ir.body.pipeline_loop.trip_counts {
        ir::TripCounts::Uniform(n) => u32_of(*n, "pipeline trip count")?,
        ir::TripCounts::PerItem(_) => return Err(unsupported("per-tile pipeline trip counts")),
    };
    let tiles_per_program = match &ir.body.outer_loop.trip_counts {
        ir::TripCounts::Uniform(n) => TripCounts::Uniform(u32_of(*n, "outer trip count")?),
        ir::TripCounts::PerItem(v) => TripCounts::PerProgram(
            v.iter()
                .map(|&n| u32_of(n, "outer trip count"))
                .collect::<Result<_, _>>()?,
        ),
    };

    let mut mma_key = None;
    let tile = Tile {
        prologue: work(&ir.body.prologue.ops, &mut mma_key)?,
        iterations,
        per_iteration: work(&ir.body.pipeline_loop.ops, &mut mma_key)?,
        prefetch_distance: ir.body.pipeline_loop.prefetch_distance,
        epilogue: work(&ir.body.epilogue.ops, &mut mma_key)?,
    };
    Ok(Kernel {
        grid: u32_of(ir.total_ctas(), "grid size")?,
        num_warps: ir.launch.num_warps,
        registers_per_thread: ir.resources.registers_per_thread,
        smem_per_cta_bytes: ir.total_smem_bytes(),
        tiles_per_program,
        mma_key: mma_key.ok_or_else(|| unsupported("kernel without MMA"))?,
        tile,
    })
}

/// The L2 reuse model behind the core's `L2` interface.
pub struct IrL2<'a> {
    model: L2Model<'a>,
    policy: HashMap<String, CachePolicy>,
}

impl<'a> IrL2<'a> {
    pub fn new(ir: &'a KernelIR, p: &Profile, initial: L2InitState, options: L2Options) -> Self {
        let tensor_policy: HashMap<&str, CachePolicy> = ir
            .tensors
            .iter()
            .map(|t| (t.id.as_str(), t.cache_policy))
            .collect();
        let b = &ir.body;
        let policy = b
            .prologue
            .ops
            .iter()
            .chain(&b.pipeline_loop.ops)
            .chain(&b.epilogue.ops)
            .filter_map(|op| match op {
                Op::GmemLoad(g) | Op::GmemStore(g) => Some((
                    g.op_id.clone(),
                    tensor_policy
                        .get(g.tensor_id.as_str())
                        .copied()
                        .unwrap_or(CachePolicy::Default),
                )),
                _ => None,
            })
            .collect();
        IrL2 {
            model: L2Model::new(ir, L2Profile::from(p), initial, options),
            policy,
        }
    }
}

impl L2 for IrL2<'_> {
    fn begin_tile(&mut self, sm: u32, program: u32, tile: u32, t_ns: f64) {
        self.model.begin_tile(sm, program, tile, t_ns);
    }

    fn end_tile(&mut self, sm: u32, program: u32, tile: u32, t_ns: f64) {
        self.model.end_tile(sm, program, tile, t_ns);
    }

    fn split(
        &mut self,
        sm: u32,
        program: u32,
        tile: u32,
        a: &Access,
        k0: u32,
        k1: u32,
        is_write: bool,
        t_ns: f64,
    ) -> (f64, f64) {
        let policy = self
            .policy
            .get(&a.op_id)
            .copied()
            .unwrap_or(CachePolicy::Default);
        let r = self.model.split(
            sm,
            program,
            tile,
            &a.op_id,
            k0.into(),
            k1.into(),
            is_write,
            policy,
            t_ns,
        );
        (r.l2_bytes, r.dram_bytes)
    }
}
