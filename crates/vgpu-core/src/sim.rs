//! Event-driven CTA scheduler with a fluid resource-sharing phase model.
//!
//! Events happen only at CTA phase boundaries (launch, prologue, steady-state
//! loop, epilogue). Between events every active phase progresses at a constant
//! rate, recomputed from the current set of active phases:
//!
//! - tensor throughput and non-MMA issue are shared among the CTAs of one SM;
//! - L2 and DRAM bandwidth are shared max-min fairly among all phases on the
//!   GPU, with the achievable bandwidth taken from the profile curve for the
//!   current number of active SMs.
//!
//! See `docs/proposals/model-core.md`, sections 3.1 to 3.5.

use crate::kernel::{Kernel, Work};
use crate::occupancy::{occupancy, Occupancy, OccupancyError};
use crate::profile::{interpolate, Profile};

/// The core's view of the L2 model: how many bytes of a load are L2 hits.
pub trait L2 {
    fn begin_tile(&mut self, _sm: u32, _program: u32, _tile: u32, _t_ns: f64) {}
    fn end_tile(&mut self, _sm: u32, _program: u32, _tile: u32, _t_ns: f64) {}
    /// Bytes served by L2 out of `bytes` loaded.
    fn hit_bytes(&mut self, sm: u32, program: u32, tile: u32, bytes: f64, t_ns: f64) -> f64;
}

/// Placeholder until the real L2 model lands: a fixed hit ratio for loads.
pub struct ConstantHitRatio(pub f64);

impl L2 for ConstantHitRatio {
    fn hit_bytes(&mut self, _: u32, _: u32, _: u32, bytes: f64, _: f64) -> f64 {
        bytes * self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Occupancy(OccupancyError),
    MissingMma(String),
    EmptyCurve(&'static str),
}

/// Profile values at one draw, in ns, bytes and FLOP units.
#[derive(Debug, Clone)]
pub struct Machine {
    pub sm_count: u32,
    /// (active SMs, bytes per ns).
    pub dram_bw: Vec<(f64, f64)>,
    pub l2_bw: Vec<(f64, f64)>,
    pub dram_latency_ns: f64,
    pub l2_latency_ns: f64,
    /// (MMA-issuing warps on the SM, FLOP per ns for the whole SM).
    pub mma: Vec<(f64, f64)>,
    /// Warp instructions per ns per SM.
    pub issue_rate: f64,
    pub launch_overhead_ns: f64,
    pub cta_launch_ns: f64,
}

fn curve<T>(
    points: &[T],
    f: impl Fn(&T) -> (f64, f64),
    name: &'static str,
) -> Result<Vec<(f64, f64)>, ModelError> {
    let mut v: Vec<(f64, f64)> = points.iter().map(f).collect();
    if v.is_empty() {
        return Err(ModelError::EmptyCurve(name));
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(v)
}

impl Machine {
    /// The unperturbed profile (draw 0).
    pub fn nominal(p: &Profile, mma_key: &str) -> Result<Machine, ModelError> {
        let m = &p.measured;
        let mma_points = m
            .compute
            .mma_tflops_per_sm_by_warps
            .get(mma_key)
            .ok_or_else(|| ModelError::MissingMma(mma_key.to_string()))?;
        Ok(Machine {
            sm_count: p.specs.sm_count,
            dram_bw: curve(
                &m.memory.dram_bw_by_active_sms,
                |q| (q.active_sms.into(), q.bw_gbps.value),
                "dram_bw_by_active_sms",
            )?,
            l2_bw: curve(
                &m.memory.l2_bw_by_active_sms,
                |q| (q.active_sms.into(), q.bw_gbps.value),
                "l2_bw_by_active_sms",
            )?,
            dram_latency_ns: m.memory.dram_latency_ns.value,
            l2_latency_ns: m.memory.l2_latency_ns.value,
            mma: curve(
                mma_points,
                |q| (q.mma_warps_per_sm.into(), q.tflops_per_sm.value * 1e3),
                "mma_tflops_per_sm_by_warps",
            )?,
            issue_rate: m.compute.non_mma_issue_rate_per_sm_gips.value,
            launch_overhead_ns: m.overhead.launch_overhead_us.value * 1e3,
            cta_launch_ns: m.overhead.per_cta_launch_latency_ns.value,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Launch overhead plus makespan.
    pub time_ns: f64,
    pub makespan_ns: f64,
    pub occupancy: Occupancy,
    /// Programs divided by CTA slots on the GPU.
    pub waves: f64,
    /// Busy SM time divided by SM count times makespan.
    pub tail_efficiency: f64,
    pub load_bytes: f64,
    pub l2_hit_bytes: f64,
    pub dram_bytes: f64,
    pub events: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Launch,
    Prologue,
    Main,
    Epilogue,
}

/// Total demand of one phase. Progress is a fraction from 1 down to 0.
#[derive(Debug, Clone, Default)]
struct Demand {
    flops: f64,
    warp_insts: f64,
    /// All bytes through L2 (loads and stores).
    bytes: f64,
    dram_bytes: f64,
    latency_ns: f64,
    /// No overlap of memory and compute within the CTA.
    serial: bool,
}

impl Demand {
    fn is_empty(&self) -> bool {
        self.flops == 0.0 && self.warp_insts == 0.0 && self.bytes == 0.0 && self.latency_ns == 0.0
    }
}

struct Cta {
    program: u32,
    sm: u32,
    tile: u32,
    tiles: u32,
    phase: Phase,
    remaining: f64,
    demand: Demand,
}

/// Max-min fair split of `capacity` among flows with the given caps.
fn waterfill(caps: &[f64], capacity: f64, order: &mut Vec<usize>, out: &mut [f64]) {
    order.clear();
    order.extend(0..caps.len());
    order.sort_by(|&a, &b| caps[a].total_cmp(&caps[b]));
    let mut left = capacity;
    for (done, &i) in order.iter().enumerate() {
        let fair = left / (caps.len() - done) as f64;
        out[i] = caps[i].min(fair);
        left -= out[i];
    }
}

struct Sim<'a, L: L2> {
    k: &'a Kernel,
    m: &'a Machine,
    l2: &'a mut L,
    t: f64,
    load_bytes: f64,
    l2_hit_bytes: f64,
    dram_bytes: f64,
}

impl<'a, L: L2> Sim<'a, L> {
    fn demand(&mut self, phase: Phase, sm: u32, program: u32, tile: u32) -> Demand {
        let tl = &self.k.tile;
        let warps = f64::from(self.k.num_warps);
        let d = tl.prefetch_distance.min(tl.iterations);
        let (work, load_iters, serial) = match phase {
            Phase::Launch => {
                return Demand {
                    latency_ns: self.m.cta_launch_ns,
                    ..Demand::default()
                }
            }
            Phase::Prologue => {
                let w = Work {
                    load_bytes: tl.prologue.load_bytes + tl.per_iteration.load_bytes * f64::from(d),
                    ..tl.prologue.clone()
                };
                (w, 1.0, false)
            }
            Phase::Main => {
                let n = f64::from(tl.iterations);
                let loads = f64::from(tl.iterations - d);
                let w = Work {
                    load_bytes: tl.per_iteration.load_bytes * loads,
                    ..tl.per_iteration.scaled(n)
                };
                (w, loads / f64::from(d.max(1)), d == 0)
            }
            Phase::Epilogue => (tl.epilogue.clone(), 0.0, false),
        };
        let hits = if work.load_bytes > 0.0 {
            self.l2
                .hit_bytes(sm, program, tile, work.load_bytes, self.t)
        } else {
            0.0
        };
        let misses = work.load_bytes - hits;
        self.load_bytes += work.load_bytes;
        self.l2_hit_bytes += hits;
        self.dram_bytes += misses + work.store_bytes;
        let latency = if work.load_bytes > 0.0 {
            (hits * self.m.l2_latency_ns + misses * self.m.dram_latency_ns) / work.load_bytes
        } else {
            0.0
        };
        Demand {
            flops: work.mma_flops,
            warp_insts: work.alu_insts_per_warp * warps,
            bytes: work.load_bytes + work.store_bytes,
            dram_bytes: misses + work.store_bytes,
            latency_ns: latency * load_iters,
            serial,
        }
    }

    fn next_phase(phase: Phase) -> Option<Phase> {
        match phase {
            Phase::Launch => Some(Phase::Prologue),
            Phase::Prologue => Some(Phase::Main),
            Phase::Main => Some(Phase::Epilogue),
            Phase::Epilogue => None,
        }
    }

    /// Move `c` to its next non-empty phase. Returns false when the CTA is done.
    fn advance(&mut self, c: &mut Cta) -> bool {
        loop {
            match Self::next_phase(c.phase) {
                Some(p) => {
                    if p == Phase::Prologue {
                        self.l2.begin_tile(c.sm, c.program, c.tile, self.t);
                    }
                    c.phase = p;
                }
                None => {
                    self.l2.end_tile(c.sm, c.program, c.tile, self.t);
                    c.tile += 1;
                    if c.tile == c.tiles {
                        return false;
                    }
                    self.l2.begin_tile(c.sm, c.program, c.tile, self.t);
                    c.phase = Phase::Prologue;
                }
            }
            c.demand = self.demand(c.phase, c.sm, c.program, c.tile);
            c.remaining = 1.0;
            if !c.demand.is_empty() {
                return true;
            }
        }
    }

    /// Returns `None` when the program has no work at all.
    fn launch(&mut self, program: u32, sm: u32) -> Option<Cta> {
        let tiles = self.k.tiles_of(program);
        if tiles == 0 {
            return None;
        }
        let mut c = Cta {
            program,
            sm,
            tile: 0,
            tiles,
            phase: Phase::Launch,
            remaining: 1.0,
            demand: Demand::default(),
        };
        c.demand = self.demand(Phase::Launch, sm, program, 0);
        if c.demand.is_empty() && !self.advance(&mut c) {
            return None;
        }
        Some(c)
    }
}

/// Reusable buffers for the rate computation.
#[derive(Default)]
struct Scratch {
    mma_users: Vec<u32>,
    issue_users: Vec<u32>,
    resident: Vec<u32>,
    t_compute: Vec<f64>,
    caps: Vec<f64>,
    l2_share: Vec<f64>,
    dram_share: Vec<f64>,
    order: Vec<usize>,
    rate: Vec<f64>,
}

fn rates(ctas: &[Cta], m: &Machine, warps: f64, s: &mut Scratch) {
    let n_sm = m.sm_count as usize;
    s.mma_users.clear();
    s.mma_users.resize(n_sm, 0);
    s.issue_users.clear();
    s.issue_users.resize(n_sm, 0);
    s.resident.clear();
    s.resident.resize(n_sm, 0);
    for c in ctas {
        let i = c.sm as usize;
        s.resident[i] += 1;
        s.mma_users[i] += u32::from(c.demand.flops > 0.0);
        s.issue_users[i] += u32::from(c.demand.warp_insts > 0.0);
    }
    let active_sms = s.resident.iter().filter(|&&r| r > 0).count() as f64;

    s.t_compute.clear();
    for c in ctas {
        let d = &c.demand;
        let i = c.sm as usize;
        let t_mma = if d.flops > 0.0 {
            let users = f64::from(s.mma_users[i]);
            d.flops * users / interpolate(&m.mma, users * warps)
        } else {
            0.0
        };
        let t_issue = if d.warp_insts > 0.0 {
            d.warp_insts * f64::from(s.issue_users[i]) / m.issue_rate
        } else {
            0.0
        };
        s.t_compute.push(if d.serial {
            d.latency_ns + t_mma + t_issue
        } else {
            d.latency_ns.max(t_mma).max(t_issue)
        });
    }

    let n = ctas.len();
    s.l2_share.resize(n, 0.0);
    s.dram_share.resize(n, 0.0);
    s.caps.clear();
    s.caps.extend(
        ctas.iter()
            .zip(&s.t_compute)
            .map(|(c, &t)| c.demand.bytes / t),
    );
    waterfill(
        &s.caps,
        interpolate(&m.l2_bw, active_sms),
        &mut s.order,
        &mut s.l2_share,
    );
    s.caps.clear();
    s.caps.extend(
        ctas.iter()
            .zip(&s.t_compute)
            .zip(&s.l2_share)
            .map(|((c, &t), &b)| {
                let d = &c.demand;
                if d.bytes == 0.0 {
                    0.0
                } else {
                    d.dram_bytes * (1.0 / t).min(b / d.bytes)
                }
            }),
    );
    waterfill(
        &s.caps,
        interpolate(&m.dram_bw, active_sms),
        &mut s.order,
        &mut s.dram_share,
    );

    s.rate.clear();
    for (i, c) in ctas.iter().enumerate() {
        let d = &c.demand;
        let t_l2 = if d.bytes > 0.0 {
            d.bytes / s.l2_share[i]
        } else {
            0.0
        };
        let t_dram = if d.dram_bytes > 0.0 {
            d.dram_bytes / s.dram_share[i]
        } else {
            0.0
        };
        let t_mem = t_l2.max(t_dram);
        let t = if d.serial {
            s.t_compute[i] + t_mem
        } else {
            s.t_compute[i].max(t_mem)
        };
        s.rate.push(1.0 / t);
    }
}

/// Simulate one kernel variant at one profile draw.
pub fn simulate<L: L2>(
    k: &Kernel,
    p: &Profile,
    m: &Machine,
    l2: &mut L,
) -> Result<Prediction, ModelError> {
    let occ = occupancy(k, &p.specs).map_err(ModelError::Occupancy)?;
    let n_sm = m.sm_count;
    let slots = occ.ctas_per_sm * n_sm;
    let mut sim = Sim {
        k,
        m,
        l2,
        t: 0.0,
        load_bytes: 0.0,
        l2_hit_bytes: 0.0,
        dram_bytes: 0.0,
    };

    // First wave: breadth-first round robin over SMs. Later: first SM to free a slot.
    let mut ctas: Vec<Cta> = Vec::with_capacity(slots as usize);
    let mut next = 0;
    while next < k.grid && (ctas.len() as u32) < slots {
        let sm = ctas.len() as u32 % n_sm;
        if let Some(c) = sim.launch(next, sm) {
            ctas.push(c);
        }
        next += 1;
    }

    let warps = f64::from(k.num_warps);
    let mut s = Scratch::default();
    let mut freed: Vec<u32> = Vec::new();
    let mut advanced: Vec<Cta> = Vec::new();
    let mut busy_sm_ns = 0.0;
    let mut events = 0u64;
    while !ctas.is_empty() {
        rates(&ctas, m, warps, &mut s);
        let dt = ctas
            .iter()
            .zip(&s.rate)
            .map(|(c, &r)| c.remaining / r)
            .fold(f64::INFINITY, f64::min);
        sim.t += dt;
        busy_sm_ns += dt * s.resident.iter().filter(|&&r| r > 0).count() as f64;
        events += 1;

        for (c, &r) in ctas.iter_mut().zip(&s.rate) {
            c.remaining -= r * dt;
        }
        freed.clear();
        let mut i = 0;
        while i < ctas.len() {
            if ctas[i].remaining > 1e-9 {
                i += 1;
                continue;
            }
            let mut c = ctas.swap_remove(i);
            if sim.advance(&mut c) {
                advanced.push(c);
            } else {
                freed.push(c.sm);
            }
        }
        ctas.append(&mut advanced);
        freed.sort_unstable();
        for &sm in &freed {
            while next < k.grid {
                let program = next;
                next += 1;
                if let Some(c) = sim.launch(program, sm) {
                    ctas.push(c);
                    break;
                }
            }
        }
    }

    let makespan = sim.t;
    Ok(Prediction {
        time_ns: m.launch_overhead_ns + makespan,
        makespan_ns: makespan,
        waves: f64::from(k.grid) / f64::from(slots),
        tail_efficiency: if makespan > 0.0 {
            busy_sm_ns / (f64::from(n_sm) * makespan)
        } else {
            1.0
        },
        occupancy: occ,
        load_bytes: sim.load_bytes,
        l2_hit_bytes: sim.l2_hit_bytes,
        dram_bytes: sim.dram_bytes,
        events,
    })
}
