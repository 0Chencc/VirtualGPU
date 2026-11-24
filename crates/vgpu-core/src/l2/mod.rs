//! Online, phase-segment-level L2 cache reuse model (v1).

use crate::ir::{AccessEntry, CachePolicy, KernelIR, Op, Strided2dEntry};
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2InitState {
    Cold,
    Warm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionMode {
    SoftLru,
    StrictLru,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Options {
    pub eviction_mode: EvictionMode,
    pub cross_wave_reuse: bool,
}

impl Default for L2Options {
    fn default() -> Self {
        Self {
            eviction_mode: EvictionMode::SoftLru,
            cross_wave_reuse: true,
        }
    }
}

/// Perturbable hardware L2/L1 parameters extracted from a GPU profile sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L2Profile {
    pub l2_size_bytes: u64,
    pub l2_effective_capacity_bytes: u64,
    pub l2_line_bytes: u64,
    pub l2_sector_bytes: u64,
    pub l1_size_bytes: u64,
    pub l1_to_l2_spill_threshold_bytes_per_sm: u64,
}

#[derive(Debug)]
pub struct L2ProfileError {
    pub message: String,
}

impl fmt::Display for L2ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L2Profile error: {}", self.message)
    }
}

impl std::error::Error for L2ProfileError {}

#[derive(Deserialize)]
struct RawGpuProfileForL2 {
    specs: RawL2Specs,
    measured: RawL2Measured,
}

#[derive(Deserialize)]
struct RawL2Specs {
    l2_size_bytes: u64,
    l2_line_bytes: u64,
    #[serde(default = "default_sector_bytes")]
    l2_sector_bytes: u64,
    l1_size_bytes: u64,
}

fn default_sector_bytes() -> u64 {
    32
}

#[derive(Deserialize)]
struct RawL2Measured {
    memory: RawL2MemoryMeasured,
    spill: RawL2SpillMeasured,
}

#[derive(Deserialize)]
struct RawL2MemoryMeasured {
    l2_effective_capacity_bytes: RawMeasuredScalar,
}

#[derive(Deserialize)]
struct RawL2SpillMeasured {
    l1_to_l2_spill_threshold_bytes_per_sm: RawMeasuredScalar,
}

#[derive(Deserialize)]
struct RawMeasuredScalar {
    value: f64,
}

impl L2Profile {
    /// Constructs an `L2Profile` directly from a GPU profile JSON document (schema v0.1).
    pub fn from_gpu_profile_json(json: &str) -> Result<Self, L2ProfileError> {
        let raw: RawGpuProfileForL2 =
            serde_json::from_str(json).map_err(|e| L2ProfileError {
                message: format!("failed to parse GPU profile JSON: {}", e),
            })?;
        let eff_val = raw.measured.memory.l2_effective_capacity_bytes.value;
        if !eff_val.is_finite() || eff_val <= 0.0 {
            return Err(L2ProfileError {
                message: format!("invalid l2_effective_capacity_bytes: {}", eff_val),
            });
        }
        let spill_val = raw.measured.spill.l1_to_l2_spill_threshold_bytes_per_sm.value;
        if !spill_val.is_finite() || spill_val < 0.0 {
            return Err(L2ProfileError {
                message: format!(
                    "invalid l1_to_l2_spill_threshold_bytes_per_sm: {}",
                    spill_val
                ),
            });
        }
        Ok(Self {
            l2_size_bytes: raw.specs.l2_size_bytes,
            l2_effective_capacity_bytes: eff_val.round() as u64,
            l2_line_bytes: raw.specs.l2_line_bytes,
            l2_sector_bytes: raw.specs.l2_sector_bytes,
            l1_size_bytes: raw.specs.l1_size_bytes,
            l1_to_l2_spill_threshold_bytes_per_sm: spill_val.round() as u64,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L2SplitResult {
    pub l2_bytes: f64,
    pub dram_bytes: f64,
    pub hit_ratio: f64,
    pub sector_aligned_bytes: u64,
    pub near_cliff: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpillTraffic {
    pub l1_bytes_per_iter: u64,
    pub l2_bytes_per_iter: u64,
    pub dram_bytes_per_iter: u64,
    pub overflow_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ChunkKey {
    tensor_idx: usize,
    base_byte_offset: u64,
    rows: u64,
    row_bytes: u64,
    row_stride_bytes: u64,
    inner_step_byte_stride: i64,
}

#[derive(Debug, Clone)]
struct OpBinding {
    tensor_idx: usize,
    access_ref: String,
    is_write: bool,
}

#[derive(Debug, Clone)]
struct ResidentChunk {
    key: ChunkKey,
    step_aligned_bytes: u64,
    full_stream_bytes: u64,
}

#[derive(Debug, Clone)]
struct ResidentTile {
    chunks: Vec<ResidentChunk>,
}

/// Stateful, instance-local L2 reuse model driven by Agent 4's event scheduler.
pub struct L2Model<'a> {
    ir: &'a KernelIR,
    profile: L2Profile,
    initial_state: L2InitState,
    options: L2Options,
    op_bindings: HashMap<String, OpBinding>,
    load_bindings: Vec<OpBinding>,
    total_read_footprint_bytes: u64,
    resident_tiles: HashMap<(u32, u32), ResidentTile>,
    active_chunks: HashMap<ChunkKey, (u32, u64)>,
    active_working_set_bytes: u64,
    recent_lru: VecDeque<(ChunkKey, u64)>,
    recent_set: HashMap<ChunkKey, u64>,
    recent_bytes: u64,
    spill_l2_pressure_bytes: u64,
}

impl<'a> L2Model<'a> {
    pub fn new(
        ir: &'a KernelIR,
        profile: L2Profile,
        initial_state: L2InitState,
        options: L2Options,
    ) -> Self {
        let tensor_index: HashMap<&str, usize> = ir
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.as_str(), i))
            .collect();

        let mut op_bindings = HashMap::new();
        let mut load_bindings = Vec::new();
        let mut seen_load_refs = HashMap::new();

        let all_ops = ir
            .body
            .prologue
            .ops
            .iter()
            .chain(ir.body.pipeline_loop.ops.iter())
            .chain(ir.body.epilogue.ops.iter());

        for op in all_ops {
            let (g, is_write) = match op {
                Op::GmemLoad(g) => (g, false),
                Op::GmemStore(g) => (g, true),
                _ => continue,
            };
            if let Some(&tensor_idx) = tensor_index.get(g.tensor_id.as_str()) {
                let binding = OpBinding {
                    tensor_idx,
                    access_ref: g.access_ref.clone(),
                    is_write,
                };
                if !is_write
                    && seen_load_refs
                        .insert((tensor_idx, g.access_ref.clone()), ())
                        .is_none()
                {
                    load_bindings.push(binding.clone());
                }
                op_bindings.insert(g.op_id.clone(), binding);
            }
        }

        let mut model = Self {
            ir,
            profile,
            initial_state,
            options,
            op_bindings,
            load_bindings,
            total_read_footprint_bytes: 0,
            resident_tiles: HashMap::new(),
            active_chunks: HashMap::new(),
            active_working_set_bytes: 0,
            recent_lru: VecDeque::new(),
            recent_set: HashMap::new(),
            recent_bytes: 0,
            spill_l2_pressure_bytes: 0,
        };
        model.total_read_footprint_bytes = model.compute_total_read_footprint();
        model
    }

    pub fn begin_tile(&mut self, _sm_id: u32, cta_id: u32, outer_idx: u32, _t_ns: f64) {
        let tile_key = (cta_id, outer_idx);
        if self.resident_tiles.contains_key(&tile_key) {
            return;
        }
        let inner_trips = self.ir.inner_trip_count(cta_id, outer_idx).unwrap_or(1).max(1);
        let mut chunks = Vec::with_capacity(self.load_bindings.len());
        for b in &self.load_bindings {
            if let Some(entry) = self.lookup_strided(&b.access_ref, cta_id, outer_idx) {
                let ckey = make_chunk_key(b.tensor_idx, entry);
                let step_aligned = self.sector_aligned_step_bytes(entry);
                if step_aligned == 0 {
                    continue;
                }
                let full_stream = full_chunk_stream_bytes(step_aligned, entry, inner_trips);
                let entry_ref = self.active_chunks.entry(ckey).or_insert((0, step_aligned));
                if entry_ref.0 == 0 {
                    self.active_working_set_bytes =
                        self.active_working_set_bytes.saturating_add(step_aligned);
                }
                entry_ref.0 += 1;
                chunks.push(ResidentChunk {
                    key: ckey,
                    step_aligned_bytes: step_aligned,
                    full_stream_bytes: full_stream,
                });
            }
        }
        self.resident_tiles.insert(tile_key, ResidentTile { chunks });
    }

    pub fn end_tile(&mut self, _sm_id: u32, cta_id: u32, outer_idx: u32, _t_ns: f64) {
        let Some(tile) = self.resident_tiles.remove(&(cta_id, outer_idx)) else {
            return;
        };
        for rc in tile.chunks {
            if let Some(slot) = self.active_chunks.get_mut(&rc.key) {
                slot.0 = slot.0.saturating_sub(1);
                if slot.0 == 0 {
                    self.active_chunks.remove(&rc.key);
                    self.active_working_set_bytes = self
                        .active_working_set_bytes
                        .saturating_sub(rc.step_aligned_bytes);
                    if self.options.cross_wave_reuse {
                        // Account for the full K-stream volume that flowed through L2 during this tile.
                        self.push_recent(rc.key, rc.full_stream_bytes);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &mut self,
        _sm_id: u32,
        cta_id: u32,
        outer_idx: u32,
        op_id: &str,
        k_begin: u64,
        k_end: u64,
        is_write: bool,
        cache_policy: CachePolicy,
        _t_ns: f64,
    ) -> L2SplitResult {
        let steps = k_end.saturating_sub(k_begin);
        if steps == 0 {
            return L2SplitResult {
                l2_bytes: 0.0,
                dram_bytes: 0.0,
                hit_ratio: 0.0,
                sector_aligned_bytes: 0,
                near_cliff: false,
            };
        }

        let Some(binding) = self.op_bindings.get(op_id) else {
            return L2SplitResult {
                l2_bytes: 0.0,
                dram_bytes: 0.0,
                hit_ratio: 0.0,
                sector_aligned_bytes: 0,
                near_cliff: false,
            };
        };

        let Some(entry) = self.lookup_strided(&binding.access_ref, cta_id, outer_idx) else {
            return L2SplitResult {
                l2_bytes: 0.0,
                dram_bytes: 0.0,
                hit_ratio: 0.0,
                sector_aligned_bytes: 0,
                near_cliff: false,
            };
        };

        let step_aligned = self.sector_aligned_step_bytes(entry);
        let total_aligned = step_aligned.saturating_mul(steps);
        let total_f = total_aligned as f64;

        if is_write || binding.is_write {
            // Intentional M1 simplification: treat all global stores as DRAM write-back traffic
            // without modeling write-allocate L2 hit credit.
            return L2SplitResult {
                l2_bytes: 0.0,
                dram_bytes: total_f,
                hit_ratio: 0.0,
                sector_aligned_bytes: total_aligned,
                near_cliff: false,
            };
        }

        if cache_policy == CachePolicy::Volatile {
            return L2SplitResult {
                l2_bytes: 0.0,
                dram_bytes: total_f,
                hit_ratio: 0.0,
                sector_aligned_bytes: total_aligned,
                near_cliff: false,
            };
        }

        let ckey = make_chunk_key(binding.tensor_idx, entry);
        let sharing_count = self
            .active_chunks
            .get(&ckey)
            .map(|(cnt, _)| (*cnt).max(1))
            .unwrap_or(1) as f64;

        // Intra-wave sharing: R concurrent CTAs sharing a chunk incur 1/R DRAM fetch.
        let mut raw_hit = (sharing_count - 1.0) / sharing_count;

        let eff_cap = self.profile.l2_effective_capacity_bytes.max(1) as f64;

        // Cross-wave hit: only hits if the full K-stream chunk has not been evicted from recent_lru.
        if self.options.cross_wave_reuse && self.recent_set.contains_key(&ckey) {
            raw_hit = raw_hit.max(1.0);
        }

        // Warm L2: only the fraction of total kernel read footprint that fits in L2 can be pre-warmed.
        if self.initial_state == L2InitState::Warm {
            let warm_hit = if self.total_read_footprint_bytes == 0 {
                1.0
            } else {
                (eff_cap / self.total_read_footprint_bytes as f64).clamp(0.0, 1.0)
            };
            raw_hit = raw_hit.max(warm_hit);
        }

        let working_set =
            (self.active_working_set_bytes.saturating_add(self.spill_l2_pressure_bytes)) as f64;

        let (cap_factor, near_cliff) = match self.options.eviction_mode {
            EvictionMode::SoftLru => {
                let lo = 0.85 * eff_cap;
                let hi = 1.15 * eff_cap;
                if working_set <= lo {
                    (1.0, false)
                } else if working_set < hi {
                    // Smooth linear transition across [0.85 * C_eff, 1.15 * C_eff].
                    let t = (working_set - lo) / (hi - lo);
                    let at_hi = (eff_cap / hi).min(1.0);
                    (1.0 - t * (1.0 - at_hi), true)
                } else {
                    ((eff_cap / working_set).min(1.0), false)
                }
            }
            EvictionMode::StrictLru => {
                if working_set <= eff_cap {
                    (1.0, false)
                } else {
                    ((eff_cap / working_set).min(1.0), false)
                }
            }
        };

        let hit_ratio = (raw_hit * cap_factor).clamp(0.0, 1.0);
        let l2_bytes = total_f * hit_ratio;
        let dram_bytes = total_f - l2_bytes;

        L2SplitResult {
            l2_bytes,
            dram_bytes,
            hit_ratio,
            sector_aligned_bytes: total_aligned,
            near_cliff,
        }
    }

    pub fn account_spill(
        &mut self,
        _sm_id: u32,
        active_threads_in_sm: u32,
        stack_frame_bytes: u64,
        spill_loads_bytes: u64,
        spill_stores_bytes: u64,
    ) -> SpillTraffic {
        let per_thread_traffic = spill_loads_bytes.saturating_add(spill_stores_bytes);
        let total_traffic = per_thread_traffic.saturating_mul(active_threads_in_sm as u64);
        if total_traffic == 0 {
            return SpillTraffic {
                l1_bytes_per_iter: 0,
                l2_bytes_per_iter: 0,
                dram_bytes_per_iter: 0,
                overflow_ratio: 0.0,
            };
        }

        let total_local_footprint = stack_frame_bytes.saturating_mul(active_threads_in_sm as u64);
        let thresh = if self.profile.l1_to_l2_spill_threshold_bytes_per_sm > 0 {
            self.profile.l1_to_l2_spill_threshold_bytes_per_sm
        } else {
            self.profile.l1_size_bytes.max(1)
        };

        let overflow_ratio = if total_local_footprint <= thresh {
            0.0
        } else {
            ((total_local_footprint - thresh) as f64 / total_local_footprint as f64).clamp(0.0, 1.0)
        };

        let overflow_footprint = (total_local_footprint as f64 * overflow_ratio) as u64;
        self.spill_l2_pressure_bytes = self.spill_l2_pressure_bytes.max(overflow_footprint);

        let overflow_bytes = (total_traffic as f64 * overflow_ratio) as u64;
        let l1_bytes = total_traffic.saturating_sub(overflow_bytes);

        let eff_cap = self.profile.l2_effective_capacity_bytes.max(1) as f64;
        let working_set =
            (self.active_working_set_bytes.saturating_add(self.spill_l2_pressure_bytes)) as f64;
        let l2_fit = if working_set <= eff_cap {
            1.0
        } else {
            (eff_cap / working_set).min(1.0)
        };
        let l2_bytes = (overflow_bytes as f64 * l2_fit) as u64;
        let dram_bytes = overflow_bytes.saturating_sub(l2_bytes);

        SpillTraffic {
            l1_bytes_per_iter: l1_bytes,
            l2_bytes_per_iter: l2_bytes,
            dram_bytes_per_iter: dram_bytes,
            overflow_ratio,
        }
    }

    pub fn active_working_set_bytes(&self) -> u64 {
        self.active_working_set_bytes
    }

    pub fn total_read_footprint_bytes(&self) -> u64 {
        self.total_read_footprint_bytes
    }

    fn compute_total_read_footprint(&self) -> u64 {
        let mut seen = HashSet::new();
        let mut total: u64 = 0;
        let total_ctas = self.ir.total_ctas() as u32;
        for cta_id in 0..total_ctas {
            let outer_trips = self.ir.outer_trip_count(cta_id).unwrap_or(1) as u32;
            for outer_idx in 0..outer_trips {
                let inner_trips = self.ir.inner_trip_count(cta_id, outer_idx).unwrap_or(1).max(1);
                for b in &self.load_bindings {
                    if let Some(entry) = self.lookup_strided(&b.access_ref, cta_id, outer_idx) {
                        let ckey = make_chunk_key(b.tensor_idx, entry);
                        if seen.insert(ckey) {
                            let step_aligned = self.sector_aligned_step_bytes(entry);
                            let full_stream =
                                full_chunk_stream_bytes(step_aligned, entry, inner_trips);
                            total = total.saturating_add(full_stream);
                        }
                    }
                }
            }
        }
        total
    }

    fn lookup_strided(&self, access_ref: &str, cta_id: u32, outer_idx: u32) -> Option<Strided2dEntry> {
        let idx = self.ir.dense_tile_index(cta_id, outer_idx)?;
        match self.ir.access_map.get(access_ref)? {
            AccessEntry::Strided2dPerOuterTile { entries, .. } => entries.get(idx).copied(),
            AccessEntry::ExplicitBlocks { .. } => None,
        }
    }

    fn sector_aligned_step_bytes(&self, entry: Strided2dEntry) -> u64 {
        let sector = self.profile.l2_sector_bytes.max(1);
        let row_bytes = entry.row_bytes();
        if row_bytes == 0 || entry.rows() == 0 {
            return 0;
        }
        let aligned_row = row_bytes.div_ceil(sector).saturating_mul(sector);
        entry.rows().saturating_mul(aligned_row)
    }

    fn push_recent(&mut self, key: ChunkKey, full_bytes: u64) {
        let cap = self.profile.l2_effective_capacity_bytes;
        if full_bytes == 0 || full_bytes > cap {
            return;
        }
        if let Some(prev_bytes) = self.recent_set.remove(&key) {
            self.recent_bytes = self.recent_bytes.saturating_sub(prev_bytes);
            if let Some(pos) = self.recent_lru.iter().position(|(k, _)| *k == key) {
                self.recent_lru.remove(pos);
            }
        }
        while self.recent_bytes.saturating_add(full_bytes) > cap {
            let Some((old_k, old_b)) = self.recent_lru.pop_front() else {
                break;
            };
            self.recent_set.remove(&old_k);
            self.recent_bytes = self.recent_bytes.saturating_sub(old_b);
        }
        self.recent_lru.push_back((key, full_bytes));
        self.recent_set.insert(key, full_bytes);
        self.recent_bytes = self.recent_bytes.saturating_add(full_bytes);
    }
}

fn full_chunk_stream_bytes(step_aligned: u64, entry: Strided2dEntry, inner_trips: u64) -> u64 {
    if entry.inner_step_byte_stride() == 0 {
        step_aligned
    } else {
        step_aligned.saturating_mul(inner_trips)
    }
}

fn make_chunk_key(tensor_idx: usize, e: Strided2dEntry) -> ChunkKey {
    ChunkKey {
        tensor_idx,
        base_byte_offset: e.base_byte_offset(),
        rows: e.rows(),
        row_bytes: e.row_bytes(),
        row_stride_bytes: e.row_stride_bytes(),
        inner_step_byte_stride: e.inner_step_byte_stride(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::TripCounts;

    fn rtx4070_l2_profile() -> L2Profile {
        L2Profile {
            l2_size_bytes: 36 * 1024 * 1024,
            l2_effective_capacity_bytes: 30 * 1024 * 1024,
            l2_line_bytes: 128,
            l2_sector_bytes: 32,
            l1_size_bytes: 128 * 1024,
            l1_to_l2_spill_threshold_bytes_per_sm: 64 * 1024,
        }
    }

    #[test]
    fn swizzled_resident_ctas_share_a_and_b_in_l2() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();
        let mut l2 = L2Model::new(&ir, rtx4070_l2_profile(), L2InitState::Cold, L2Options::default());

        // First 16 CTAs in GROUP_SIZE_M=8 cover pid_m in 0..8, pid_n in 0..2:
        // each A row-slice is shared by 2 CTAs; each B col-slice is shared by 8 CTAs.
        for cta in 0..16u32 {
            l2.begin_tile(cta % 46, cta, 0, 0.0);
        }

        let res_a = l2.split(0, 0, 0, "load_A", 0, 128, false, CachePolicy::Default, 0.0);
        let res_b = l2.split(0, 0, 0, "load_B", 0, 128, false, CachePolicy::Default, 0.0);

        assert!((res_a.hit_ratio - 0.5).abs() < 1e-6, "A shared by 2 CTAs -> 50% hit");
        assert!((res_b.hit_ratio - 0.875).abs() < 1e-6, "B shared by 8 CTAs -> 87.5% hit");
        assert_eq!(res_a.sector_aligned_bytes, 128 * 64 * 128);
        assert!((res_a.l2_bytes + res_a.dram_bytes - res_a.sector_aligned_bytes as f64).abs() < 1e-3);
        assert!(!res_a.near_cliff);

        // End wave 1 (8 A stripes + 2 B stripes = 10 MB <= 30 MB C_eff) -> fits in recent_lru.
        for cta in 0..16u32 {
            l2.end_tile(cta % 46, cta, 0, 1000.0);
        }
        l2.begin_tile(0, 16, 0, 1000.0);
        let res_a_next = l2.split(0, 16, 0, "load_A", 0, 128, false, CachePolicy::Default, 1000.0);
        assert!((res_a_next.hit_ratio - 1.0).abs() < 1e-6, "10 MB wave fits in 30 MB L2");
    }

    #[test]
    fn cross_wave_history_accounts_for_full_k_stream_and_evicts_on_4096_gemm() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();
        let prof = rtx4070_l2_profile(); // 36 MB nominal, 30 MB effective
        let mut l2 = L2Model::new(&ir, prof, L2InitState::Cold, L2Options::default());

        // 4096^3 f16 GEMM has 32 MB A + 32 MB B = 64 MB total read footprint.
        assert_eq!(l2.total_read_footprint_bytes(), 64 * 1024 * 1024);

        // Simulate 46-SM GPU with 2 CTAs/SM -> 92 CTAs per wave across all 1024 CTAs.
        // Group 0 (CTAs 0..256) streams 8 MB of A + 32 MB of B = 40 MB > 30 MB effective L2,
        // so earlier B stripes must be evicted before Group 1 (CTAs 256..512) reuses them.
        let wave_size = 92u32;
        let mut total_b_hit_wave2_plus = 0.0;
        let mut count_b_wave2_plus = 0u32;

        for wave_start in (0..1024u32).step_by(wave_size as usize) {
            let wave_end = (wave_start + wave_size).min(1024);
            for cta in wave_start..wave_end {
                l2.begin_tile(cta % 46, cta, 0, 0.0);
            }
            if wave_start >= 256 {
                // Measure solo-CTA cross-wave B hit by checking after intra-wave isolation,
                // or check directly on Group 1 start where B cols 0..N were streamed in Group 0.
                let res_b = l2.split(0, wave_start, 0, "load_B", 0, 128, false, CachePolicy::Default, 0.0);
                total_b_hit_wave2_plus += res_b.hit_ratio;
                count_b_wave2_plus += 1;
            }
            for cta in wave_start..wave_end {
                l2.end_tile(cta % 46, cta, 0, 1000.0);
            }
        }

        // Now dispatch a single solo CTA 256 (pid_m=8, pid_n=0) after the full 64 MB streamed:
        // its B column 0 (1 MB) was evicted long ago, so cross-wave hit must be 0.0 (not 1.0).
        l2.begin_tile(0, 256, 0, 2000.0);
        let solo_b = l2.split(0, 256, 0, "load_B", 0, 128, false, CachePolicy::Default, 2000.0);
        assert!(
            solo_b.hit_ratio < 0.01,
            "evicted B stripe after 64 MB stream must miss, got {}",
            solo_b.hit_ratio
        );
        // Average hit across later waves must also be strictly below 1.0 (not falsely 1.0).
        let avg_b_hit = total_b_hit_wave2_plus / count_b_wave2_plus as f64;
        assert!(
            avg_b_hit < 0.95,
            "cross-wave B hit on 64 MB GEMM in 30 MB L2 must be well below 1.0, got {}",
            avg_b_hit
        );
    }

    #[test]
    fn warm_state_scales_hit_ratio_by_l2_capacity_vs_footprint() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir_large = KernelIR::from_json_str(json).unwrap();
        let prof = rtx4070_l2_profile(); // 30 MB effective L2

        // Large 4096^3 GEMM: 64 MB footprint > 30 MB L2 -> warm hit must be < 1.0 (30/64 = 0.46875).
        let mut l2_large = L2Model::new(&ir_large, prof, L2InitState::Warm, L2Options::default());
        l2_large.begin_tile(0, 0, 0, 0.0);
        let large_res = l2_large.split(0, 0, 0, "load_A", 0, 128, false, CachePolicy::Default, 0.0);
        let expected_large = (30.0 * 1024.0 * 1024.0) / (64.0 * 1024.0 * 1024.0);
        assert!(
            (large_res.hit_ratio - expected_large).abs() < 1e-6,
            "expected warm hit {}, got {}",
            expected_large,
            large_res.hit_ratio
        );
        assert!(large_res.hit_ratio < 1.0);

        // Small footprint (reduce K iterations from 128 to 8 -> 4 MB total footprint <= 30 MB L2):
        // warm hit should be 1.0.
        let mut ir_small = ir_large.clone();
        ir_small.body.pipeline_loop.trip_counts = TripCounts::Uniform(8);
        let mut l2_small = L2Model::new(&ir_small, prof, L2InitState::Warm, L2Options::default());
        assert_eq!(l2_small.total_read_footprint_bytes(), 4 * 1024 * 1024);
        l2_small.begin_tile(0, 0, 0, 0.0);
        let small_res = l2_small.split(0, 0, 0, "load_A", 0, 8, false, CachePolicy::Default, 0.0);
        assert!(
            (small_res.hit_ratio - 1.0).abs() < 1e-6,
            "small 4 MB footprint in 30 MB warm L2 should hit ~1.0, got {}",
            small_res.hit_ratio
        );
    }

    #[test]
    fn soft_lru_flags_near_cliff_and_degrades_above_capacity() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();

        // 16 CTAs hold 8 unique A tiles (8*8KB = 64KB) + 2 unique B tiles (2*8KB = 16KB) = 80KB.
        let mut prof = rtx4070_l2_profile();
        prof.l2_effective_capacity_bytes = 80 * 1024; // right at the cliff (1.0 * C_eff)

        let mut l2 = L2Model::new(&ir, prof, L2InitState::Cold, L2Options::default());
        for cta in 0..16u32 {
            l2.begin_tile(cta, cta, 0, 0.0);
        }
        assert_eq!(l2.active_working_set_bytes(), 80 * 1024);

        let res = l2.split(0, 0, 0, "load_B", 0, 16, false, CachePolicy::Default, 0.0);
        assert!(res.near_cliff, "working set == C_eff should flag near_cliff");
        assert!(res.hit_ratio < 0.875 && res.hit_ratio > 0.7);
    }

    #[test]
    fn spill_overflow_transitions_from_l1_to_l2_dram() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();
        let mut l2 = L2Model::new(&ir, rtx4070_l2_profile(), L2InitState::Cold, L2Options::default());

        // 256 active threads * 128B stack = 32KB <= 64KB threshold -> fits in L1.
        let small = l2.account_spill(0, 256, 128, 32, 32);
        assert_eq!(small.overflow_ratio, 0.0);
        assert_eq!(small.l2_bytes_per_iter, 0);
        assert_eq!(small.dram_bytes_per_iter, 0);

        // 256 active threads * 512B stack = 128KB > 64KB -> 50% overflows to L2.
        let large = l2.account_spill(0, 256, 512, 64, 64);
        assert!((large.overflow_ratio - 0.5).abs() < 1e-6);
        assert!(large.l2_bytes_per_iter > 0);
    }

    #[test]
    fn construct_l2_profile_from_rtx4070_json() {
        let profile_json = include_str!("../../../../profiles/rtx4070.json");
        let prof = L2Profile::from_gpu_profile_json(profile_json)
            .expect("profiles/rtx4070.json should parse into L2Profile");

        // Measured values change whenever the profile is re-measured, so read them from the file.
        let raw: serde_json::Value = serde_json::from_str(profile_json).unwrap();
        let measured = |path: &str| raw.pointer(path).and_then(|v| v.as_f64()).unwrap() as u64;
        let capacity = measured("/measured/memory/l2_effective_capacity_bytes/value");

        assert_eq!(prof.l2_size_bytes, 37_748_736);
        assert_eq!(prof.l2_effective_capacity_bytes, capacity);
        assert_eq!(prof.l2_line_bytes, 128);
        assert_eq!(prof.l2_sector_bytes, 32);
        assert_eq!(prof.l1_size_bytes, 131_072);
        assert_eq!(
            prof.l1_to_l2_spill_threshold_bytes_per_sm,
            measured("/measured/spill/l1_to_l2_spill_threshold_bytes_per_sm/value")
        );

        // Verify L2Model actually consumes the loaded profile's l2_effective_capacity_bytes.
        let ir_json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(ir_json).unwrap();
        let mut l2 = L2Model::new(&ir, prof, L2InitState::Warm, L2Options::default());
        l2.begin_tile(0, 0, 0, 0.0);
        let res = l2.split(0, 0, 0, "load_A", 0, 128, false, CachePolicy::Default, 0.0);
        let expected = (capacity as f64 / (64.0 * 1024.0 * 1024.0)).min(1.0);
        assert!(
            (res.hit_ratio - expected).abs() < 1e-6,
            "expected warm hit from rtx4070.json ({}), got {}",
            expected,
            res.hit_ratio
        );
    }
}
