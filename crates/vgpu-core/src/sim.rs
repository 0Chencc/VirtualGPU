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

use crate::kernel::{Access, Kernel, Spill, SpillScope};
use crate::occupancy::{occupancy, Occupancy, OccupancyError};
use crate::profile::{interpolate, Profile};

/// The core's view of the L2 model.
pub trait L2 {
    fn begin_tile(&mut self, _sm: u32, _program: u32, _tile: u32, _t_ns: f64) {}
    fn end_tile(&mut self, _sm: u32, _program: u32, _tile: u32, _t_ns: f64) {}
    /// Traffic of one access over steps `[k_begin, k_end)` of a phase, as
    /// `(bytes served by L2, bytes served by DRAM)`.
    #[allow(clippy::too_many_arguments)]
    fn split(
        &mut self,
        sm: u32,
        program: u32,
        tile: u32,
        access: &Access,
        k_begin: u32,
        k_end: u32,
        is_write: bool,
        t_ns: f64,
    ) -> (f64, f64);

    /// Spill traffic of all resident threads of one SM per execution of the
    /// spill scope, as `(L1 bytes, L2 bytes, DRAM bytes)`.
    fn spill(&mut self, _sm: u32, active_threads_in_sm: u32, s: &Spill) -> (f64, f64, f64) {
        (
            s.bytes_per_thread * f64::from(active_threads_in_sm),
            0.0,
            0.0,
        )
    }
}

/// Fixed hit ratio for loads, nominal bytes, stores go to DRAM. For tests.
pub struct ConstantHitRatio(pub f64);

impl L2 for ConstantHitRatio {
    fn split(
        &mut self,
        _: u32,
        _: u32,
        _: u32,
        a: &Access,
        k0: u32,
        k1: u32,
        is_write: bool,
        _: f64,
    ) -> (f64, f64) {
        let bytes = a.nominal_bytes * f64::from(k1 - k0);
        let hit = if is_write { 0.0 } else { self.0 };
        (bytes * hit, bytes * (1.0 - hit))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Occupancy(OccupancyError),
    MissingMma(String),
    EmptyCurve(&'static str),
    Unsupported(String),
}

/// Profile values at one draw, in ns, bytes and FLOP units.
#[derive(Debug, Clone)]
pub struct Machine {
    pub sm_count: u32,
    /// (active SMs, bytes per ns).
    pub dram_bw: Vec<(f64, f64)>,
    pub l2_bw: Vec<(f64, f64)>,
    /// Load latency at idle and at saturation of each level.
    pub dram_latency_ns: f64,
    pub dram_latency_loaded_ns: f64,
    pub l2_latency_ns: f64,
    pub l2_latency_loaded_ns: f64,
    /// (MMA-issuing warps on the SM, FLOP per ns for the whole SM).
    pub mma: Vec<(f64, f64)>,
    /// Warp instructions per ns per SM.
    pub issue_rate: f64,
    /// Shared memory bytes per ns per SM.
    pub smem_bw: f64,
    /// L1-resident local memory (spill) bytes per ns per SM.
    pub local_bw: f64,
    pub launch_overhead_ns: f64,
    pub cta_launch_ns: f64,
    /// Multiplier on per-iteration spill traffic whose loop placement is assumed.
    pub assumed_spill_scale: f64,
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
    /// Latencies when DRAM and L2 deliver the given fractions of their peak
    /// bandwidth: linear between the idle and the saturated measurement.
    fn latency(&self, dram_util: f64, l2_util: f64) -> Latency {
        let lerp = |a: f64, b: f64, u: f64| a + (b - a) * u.clamp(0.0, 1.0);
        Latency {
            dram: lerp(self.dram_latency_ns, self.dram_latency_loaded_ns, dram_util),
            l2: lerp(self.l2_latency_ns, self.l2_latency_loaded_ns, l2_util),
        }
    }

    fn peak(curve: &[(f64, f64)]) -> f64 {
        curve[curve.len() - 1].1
    }

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
            dram_latency_loaded_ns: m
                .memory
                .dram_latency_loaded_ns
                .as_ref()
                .map_or(m.memory.dram_latency_ns.value, |v| v.value),
            l2_latency_ns: m.memory.l2_latency_ns.value,
            l2_latency_loaded_ns: m
                .memory
                .l2_latency_loaded_ns
                .as_ref()
                .map_or(m.memory.l2_latency_ns.value, |v| v.value),
            mma: curve(
                mma_points,
                |q| (q.mma_warps_per_sm.into(), q.tflops_per_sm.value * 1e3),
                "mma_tflops_per_sm_by_warps",
            )?,
            issue_rate: m.compute.non_mma_issue_rate_per_sm_gips.value,
            smem_bw: m.memory.smem_bw_per_sm_gbps.value,
            local_bw: m
                .spill
                .get("local_mem_l1_hit_bw_per_sm_gbps")
                .and_then(Option::as_ref)
                .map_or(m.memory.smem_bw_per_sm_gbps.value, |v| v.value),
            launch_overhead_ns: m.overhead.launch_overhead_us.value * 1e3,
            cta_launch_ns: m.overhead.per_cta_launch_latency_ns.value,
            assumed_spill_scale: 1.0,
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
    /// Fraction of CTA-time during which each resource of `RESOURCES` was the
    /// bound of the CTA's current phase.
    pub bound_fraction: [f64; 7],
}

/// Resources a phase can be bound by, in `Prediction::bound_fraction` order.
pub const RESOURCES: [&str; 7] = ["latency", "mma", "issue", "smem", "local", "l2", "dram"];

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
    smem_bytes: f64,
    /// Spill bytes served by L1.
    local_bytes: f64,
    /// All bytes through L2 (loads and stores).
    bytes: f64,
    dram_bytes: f64,
    /// Latency that does not depend on memory load (CTA launch).
    fixed_latency_ns: f64,
    /// Memory round trips not hidden by the pipeline.
    round_trips: f64,
    /// Fraction of loaded bytes served by L2.
    hit_fraction: f64,
    /// No overlap of memory and compute within the CTA.
    serial: bool,
}

impl Demand {
    fn is_empty(&self) -> bool {
        self.flops == 0.0
            && self.warp_insts == 0.0
            && self.smem_bytes == 0.0
            && self.local_bytes == 0.0
            && self.bytes == 0.0
            && self.fixed_latency_ns == 0.0
            && self.round_trips == 0.0
    }

    /// Exposed latency given the current DRAM and L2 load latencies.
    fn latency(&self, lat: Latency) -> f64 {
        self.fixed_latency_ns
            + self.round_trips * (self.hit_fraction * lat.l2 + (1.0 - self.hit_fraction) * lat.dram)
    }
}

/// Load latencies at the current memory utilization, in ns.
#[derive(Debug, Clone, Copy)]
struct Latency {
    dram: f64,
    l2: f64,
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
///
/// Flows whose cap is below the fair share are frozen at their cap and the
/// rest is shared again; the fair share only grows, so one pass per freezing
/// round is enough, and rounds are few in practice.
fn waterfill(caps: &[f64], capacity: f64, out: &mut [f64]) {
    const OPEN: f64 = -1.0;
    out.fill(OPEN);
    let mut left = capacity;
    let mut open = caps.len();
    while open > 0 {
        let fair = left / open as f64;
        let before = open;
        for (o, &cap) in out.iter_mut().zip(caps) {
            if *o == OPEN && cap <= fair {
                *o = cap;
                left -= cap;
                open -= 1;
            }
        }
        if open == before {
            out.iter_mut()
                .filter(|o| **o == OPEN)
                .for_each(|o| *o = fair);
            return;
        }
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
    /// Spill traffic of one CTA per execution of the spill scope, as
    /// `(L1 bytes, L2 bytes, DRAM bytes)`.
    spill: (f64, f64, f64),
}

impl<'a, L: L2> Sim<'a, L> {
    /// Traffic of `accesses` over steps `[k0, k1)`, as (L2 hit bytes, DRAM bytes).
    #[allow(clippy::too_many_arguments)]
    fn traffic(
        &mut self,
        accesses: &[Access],
        k0: u32,
        k1: u32,
        is_write: bool,
        sm: u32,
        program: u32,
        tile: u32,
    ) -> (f64, f64) {
        let (mut hits, mut misses) = (0.0, 0.0);
        if k1 > k0 {
            for a in accesses {
                let (h, m) = self
                    .l2
                    .split(sm, program, tile, a, k0, k1, is_write, self.t);
                hits += h;
                misses += m;
            }
        }
        (hits, misses)
    }

    /// Phase demands under the KernelIR convention: the pipeline's first
    /// `prefetch_distance` loads are issued in the prologue (prefill), the
    /// steady state loads the rest, and the last `prefetch_distance`
    /// iterations only compute (drain). `prologue` holds only work outside
    /// the software pipeline.
    fn demand(&mut self, phase: Phase, sm: u32, program: u32, tile: u32) -> Demand {
        let k = self.k;
        let tl = &k.tile;
        let n = tl.iterations;
        let d = tl.prefetch_distance.min(n);
        let (compute, steps, loads, stores, round_trips, serial) = match phase {
            Phase::Launch => {
                return Demand {
                    fixed_latency_ns: self.m.cta_launch_ns,
                    ..Demand::default()
                }
            }
            Phase::Prologue => {
                let (h0, m0) = self.traffic(&tl.prologue.loads, 0, 1, false, sm, program, tile);
                let (h1, m1) =
                    self.traffic(&tl.per_iteration.loads, 0, d, false, sm, program, tile);
                let st = self.traffic(&tl.prologue.stores, 0, 1, true, sm, program, tile);
                (&tl.prologue, 1, (h0 + h1, m0 + m1), st, 1.0, false)
            }
            Phase::Main => {
                let ld = self.traffic(&tl.per_iteration.loads, d, n, false, sm, program, tile);
                let st = self.traffic(&tl.per_iteration.stores, 0, n, true, sm, program, tile);
                let trips = f64::from(n - d) / f64::from(d.max(1));
                (&tl.per_iteration, n, ld, st, trips, d == 0)
            }
            Phase::Epilogue => {
                let ld = self.traffic(&tl.epilogue.loads, 0, 1, false, sm, program, tile);
                let st = self.traffic(&tl.epilogue.stores, 0, 1, true, sm, program, tile);
                (&tl.epilogue, 1, ld, st, 1.0, false)
            }
        };
        let (hits, misses) = loads;
        let stores = stores.0 + stores.1;
        let load_bytes = hits + misses;
        self.load_bytes += load_bytes;
        self.l2_hit_bytes += hits;
        self.dram_bytes += misses + stores;
        let spill_executions = match (k.spill.scope, phase) {
            (SpillScope::Once, Phase::Prologue) => 1.0,
            (SpillScope::PerIteration { assumed: false }, Phase::Main) => f64::from(steps),
            (SpillScope::PerIteration { assumed: true }, Phase::Main) => {
                f64::from(steps) * self.m.assumed_spill_scale
            }
            _ => 0.0,
        };
        let (spill_l1, spill_l2, spill_dram) = self.spill;
        let (spill_l1, spill_l2, spill_dram) = (
            spill_l1 * spill_executions,
            spill_l2 * spill_executions,
            spill_dram * spill_executions,
        );
        self.dram_bytes += spill_dram;
        let steps = f64::from(steps);
        Demand {
            flops: compute.mma_flops * steps,
            warp_insts: compute.alu_insts_per_warp * f64::from(k.num_warps) * steps,
            smem_bytes: compute.smem_bytes * steps,
            local_bytes: spill_l1,
            bytes: load_bytes + stores + spill_l2 + spill_dram,
            dram_bytes: misses + stores + spill_dram,
            fixed_latency_ns: 0.0,
            round_trips: if load_bytes > 0.0 { round_trips } else { 0.0 },
            hit_fraction: if load_bytes > 0.0 {
                hits / load_bytes
            } else {
                0.0
            },
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
    smem_users: Vec<u32>,
    local_users: Vec<u32>,
    resident: Vec<u32>,
    t_compute: Vec<f64>,
    caps: Vec<f64>,
    l2_share: Vec<f64>,
    dram_share: Vec<f64>,
    rate: Vec<f64>,
    /// Per CTA, the time each resource alone would need, in `RESOURCES` order.
    terms: Vec<[f64; 7]>,
}

/// `mma_per_cta[u]` is the tensor throughput of each of `u` MMA-issuing CTAs on one SM.
fn rates(ctas: &[Cta], m: &Machine, mma_per_cta: &[f64], lat: Latency, s: &mut Scratch) {
    let n_sm = m.sm_count as usize;
    s.mma_users.clear();
    s.mma_users.resize(n_sm, 0);
    s.issue_users.clear();
    s.issue_users.resize(n_sm, 0);
    s.smem_users.clear();
    s.smem_users.resize(n_sm, 0);
    s.local_users.clear();
    s.local_users.resize(n_sm, 0);
    s.resident.clear();
    s.resident.resize(n_sm, 0);
    for c in ctas {
        let i = c.sm as usize;
        s.resident[i] += 1;
        s.mma_users[i] += u32::from(c.demand.flops > 0.0);
        s.issue_users[i] += u32::from(c.demand.warp_insts > 0.0);
        s.smem_users[i] += u32::from(c.demand.smem_bytes > 0.0);
        s.local_users[i] += u32::from(c.demand.local_bytes > 0.0);
    }
    let active_sms = s.resident.iter().filter(|&&r| r > 0).count() as f64;

    s.t_compute.clear();
    s.terms.clear();
    for c in ctas {
        let d = &c.demand;
        let i = c.sm as usize;
        let t_mma = if d.flops > 0.0 {
            d.flops / mma_per_cta[s.mma_users[i] as usize]
        } else {
            0.0
        };
        let t_issue = if d.warp_insts > 0.0 {
            d.warp_insts * f64::from(s.issue_users[i]) / m.issue_rate
        } else {
            0.0
        };
        let t_smem = if d.smem_bytes > 0.0 {
            d.smem_bytes * f64::from(s.smem_users[i]) / m.smem_bw
        } else {
            0.0
        };
        let t_local = if d.local_bytes > 0.0 {
            d.local_bytes * f64::from(s.local_users[i]) / m.local_bw
        } else {
            0.0
        };
        let latency = d.latency(lat);
        s.terms
            .push([latency, t_mma, t_issue, t_smem, t_local, 0.0, 0.0]);
        let t_sm = t_mma.max(t_issue).max(t_smem).max(t_local);
        // Time without memory bandwidth limits: caps the bandwidth demand.
        s.t_compute.push(if d.serial {
            latency + t_sm
        } else {
            latency.max(t_sm)
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
    waterfill(&s.caps, interpolate(&m.l2_bw, active_sms), &mut s.l2_share);
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
        s.terms[i][5] = t_l2;
        s.terms[i][6] = t_dram;
        let t_mem = t_l2.max(t_dram);
        let [latency, t_mma, t_issue, t_smem, t_local, ..] = s.terms[i];
        let t_sm = t_mma.max(t_issue).max(t_smem).max(t_local);
        // Synchronous loads: the CTA waits for its loads, then computes. The
        // wait is the longer of the round trip and the transfer.
        let t = if d.serial {
            latency.max(t_mem) + t_sm
        } else {
            latency.max(t_sm).max(t_mem)
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
    // Spill traffic at full residency, computed once for the steady state.
    let threads_in_sm = occ.ctas_per_sm * k.num_warps * p.specs.warp_size;
    let (l1, l2b, dram) = match k.spill.scope {
        SpillScope::None => (0.0, 0.0, 0.0),
        _ => l2.spill(0, threads_in_sm, &k.spill),
    };
    let per_cta = f64::from(occ.ctas_per_sm);
    let spill = (l1 / per_cta, l2b / per_cta, dram / per_cta);
    let mut sim = Sim {
        k,
        m,
        l2,
        t: 0.0,
        load_bytes: 0.0,
        l2_hit_bytes: 0.0,
        dram_bytes: 0.0,
        spill,
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
    let mma_per_cta: Vec<f64> = (0..=occ.ctas_per_sm)
        .map(|u| {
            let users = f64::from(u.max(1));
            interpolate(&m.mma, users * warps) / users
        })
        .collect();
    let mut s = Scratch::default();
    let mut freed: Vec<u32> = Vec::new();
    let mut advanced: Vec<Cta> = Vec::new();
    let mut busy_sm_ns = 0.0;
    let mut events = 0u64;
    let mut bound_ns = [0.0f64; 7];
    // Memory utilization carried between events; latency depends on it and
    // it depends on the rates, so each event takes two damped passes.
    let (dram_peak, l2_peak) = (Machine::peak(&m.dram_bw), Machine::peak(&m.l2_bw));
    let mut util = (0.0, 0.0);
    let achieved = |ctas: &[Cta], rate: &[f64]| {
        ctas.iter().zip(rate).fold((0.0, 0.0), |(d, l), (c, &r)| {
            (d + r * c.demand.dram_bytes, l + r * c.demand.bytes)
        })
    };
    while !ctas.is_empty() {
        for _ in 0..2 {
            rates(&ctas, m, &mma_per_cta, m.latency(util.0, util.1), &mut s);
            let (d, l) = achieved(&ctas, &s.rate);
            util = (0.5 * (util.0 + d / dram_peak), 0.5 * (util.1 + l / l2_peak));
        }
        let dt = ctas
            .iter()
            .zip(&s.rate)
            .map(|(c, &r)| c.remaining / r)
            .fold(f64::INFINITY, f64::min);
        sim.t += dt;
        busy_sm_ns += dt * s.resident.iter().filter(|&&r| r > 0).count() as f64;
        for t in &s.terms {
            let b = (0..t.len()).fold(0, |b, j| if t[j] > t[b] { j } else { b });
            bound_ns[b] += dt;
        }
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
        bound_fraction: {
            let total: f64 = bound_ns.iter().sum();
            bound_ns.map(|b| if total > 0.0 { b / total } else { 0.0 })
        },
    })
}
