use vgpu_core::kernel::{Access, Kernel, Spill, SpillScope, Tile, TripCounts, Work};
use vgpu_core::occupancy::{occupancy, Limiter, OccupancyError};
use vgpu_core::profile::Profile;
use vgpu_core::sim::{simulate, ConstantHitRatio, Machine, Prediction, Traffic, L2};

const MMA: &str = "mma.sync:f16:f16:f32:16x8x16";

fn profile() -> Profile {
    Profile::from_json(include_str!("data/test_fixture_sm89.json")).unwrap()
}

fn access(op: &str, bytes: f64) -> Access {
    Access {
        op_id: op.to_string(),
        nominal_bytes: bytes,
    }
}

/// A 128x128x32 GEMM tile over `iterations` K steps.
fn gemm(grid: u32, iterations: u32, prefetch_distance: u32) -> Kernel {
    Kernel {
        grid,
        num_warps: 4,
        registers_per_thread: 128,
        smem_per_cta_bytes: 16384 * u64::from(prefetch_distance + 1),
        tiles_per_program: TripCounts::Uniform(1),
        mma_key: MMA.to_string(),
        spill: Spill::default(),
        async_loads: true,
        tile_iterations: None,
        tile: Tile {
            prologue: Work::default(),
            iterations,
            per_iteration: Work {
                mma_flops: 2.0 * 128.0 * 128.0 * 32.0,
                alu_insts_per_warp: 16.0,
                loads: vec![access("load_A", 8192.0), access("load_B", 8192.0)],
                ..Work::default()
            },
            prefetch_distance,
            epilogue: Work {
                stores: vec![access("store_C", 32768.0)],
                alu_insts_per_warp: 64.0,
                ..Work::default()
            },
        },
    }
}

/// Compute-only kernel: no memory traffic, no latency, one CTA per SM.
fn compute_only(grid: u32) -> Kernel {
    let mut k = gemm(grid, 64, 1);
    k.smem_per_cta_bytes = 90 * 1024;
    k.tile.per_iteration.loads.clear();
    k.tile.per_iteration.alu_insts_per_warp = 0.0;
    k.tile.epilogue = Work::default();
    k
}

fn run(k: &Kernel, mut m: Machine) -> Prediction {
    let p = profile();
    m.launch_overhead_ns = 0.0;
    simulate(k, &p, &m, &mut ConstantHitRatio(0.5)).unwrap()
}

fn machine(k: &Kernel) -> Machine {
    let mut m = Machine::nominal(&profile(), &k.mma_key).unwrap();
    m.cta_launch_ns = 0.0;
    m
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1.0)
}

// Occupancy

#[test]
fn occupancy_limited_by_registers() {
    let mut k = gemm(1, 1, 1);
    k.registers_per_thread = 255; // 8192 regs per warp after rounding -> 8 warps -> 2 CTAs
    k.smem_per_cta_bytes = 0;
    let o = occupancy(&k, &profile().specs).unwrap();
    assert_eq!((o.ctas_per_sm, o.limiter), (2, Limiter::Registers));
}

#[test]
fn occupancy_limited_by_shared_memory_with_reserve_and_carveout() {
    let mut k = gemm(1, 1, 1);
    k.registers_per_thread = 32;
    k.smem_per_cta_bytes = 32 * 1024; // plus 1 KiB reserve: 3 fit in 100 KiB, not 4
    let o = occupancy(&k, &profile().specs).unwrap();
    assert_eq!(
        (o.ctas_per_sm, o.limiter, o.carveout_bytes),
        (3, Limiter::SharedMemory, 102400)
    );
    k.smem_per_cta_bytes = 15 * 1024; // 16 KiB with reserve: 6 fit; 4 would need the 64 KiB carveout
    let o = occupancy(&k, &profile().specs).unwrap();
    assert_eq!(o.ctas_per_sm, 6);
}

#[test]
fn occupancy_limited_by_warps() {
    let mut k = gemm(1, 1, 1);
    k.num_warps = 16;
    k.registers_per_thread = 32;
    k.smem_per_cta_bytes = 0;
    let o = occupancy(&k, &profile().specs).unwrap();
    assert_eq!((o.ctas_per_sm, o.limiter), (3, Limiter::Warps));
}

#[test]
fn occupancy_rejects_oversized_cta() {
    let mut k = gemm(1, 1, 1);
    k.smem_per_cta_bytes = 200 * 1024;
    assert!(matches!(
        occupancy(&k, &profile().specs),
        Err(OccupancyError::TooMuchSharedMemory { .. })
    ));
}

// Waves and tails

#[test]
fn one_full_wave_is_fully_efficient() {
    let k = compute_only(46);
    let p = run(&k, machine(&k));
    assert_eq!(p.occupancy.ctas_per_sm, 1);
    assert!(close(p.waves, 1.0));
    assert!(close(p.tail_efficiency, 1.0));
}

#[test]
fn one_extra_cta_doubles_time_and_halves_efficiency() {
    let full = compute_only(46);
    let tail = compute_only(47);
    let a = run(&full, machine(&full));
    let b = run(&tail, machine(&tail));
    assert!(close(b.makespan_ns, 2.0 * a.makespan_ns));
    assert!(close(b.tail_efficiency, 47.0 / 92.0));
}

#[test]
fn compute_bound_time_matches_mma_throughput() {
    let k = compute_only(46);
    let m = machine(&k);
    let p = run(&k, m);
    // 4 MMA warps per SM -> 0.60 TFLOP/s per SM.
    let expected = 64.0 * 2.0 * 128.0 * 128.0 * 32.0 / (0.60 * 1e3);
    assert!(
        close(p.makespan_ns, expected),
        "{} vs {}",
        p.makespan_ns,
        expected
    );
}

#[test]
fn persistent_program_runs_its_tiles_in_sequence() {
    let one = compute_only(46);
    let mut two = compute_only(46);
    two.tiles_per_program = TripCounts::Uniform(2);
    let a = run(&one, machine(&one));
    let b = run(&two, machine(&two));
    assert!(close(b.makespan_ns, 2.0 * a.makespan_ns));
}

#[test]
fn memory_bound_time_matches_dram_bandwidth() {
    let mut k = compute_only(46 * 4);
    k.tile.per_iteration = Work {
        loads: vec![access("load", 1048576.0)],
        ..Work::default()
    };
    k.tile.iterations = 8;
    let mut m = machine(&k);
    m.dram_latency_ns = 0.0;
    m.l2_latency_ns = 0.0;
    let p = run(&k, m);
    // Half the loads hit L2; 2400 GB/s of L2 is not the bound, 450 GB/s of DRAM is.
    let expected = 46.0 * 4.0 * 8.0 * f64::from(1 << 20) * 0.5 / 450.0;
    assert!(
        close(p.makespan_ns, expected),
        "{} vs {}",
        p.makespan_ns,
        expected
    );
}

#[test]
fn pipelining_hides_latency() {
    // Same residency for both and a round trip longer than the compute of
    // an iteration, so only the pipeline differs and latency matters.
    let mut sync = gemm(46 * 8, 128, 0);
    let piped = gemm(46 * 8, 128, 2);
    sync.smem_per_cta_bytes = piped.smem_per_cta_bytes;
    let slow = |k: &Kernel| {
        let mut m = machine(k);
        m.dram_latency_ns = 20000.0;
        m.dram_latency_loaded_ns = 20000.0;
        m.l2_latency_ns = 20000.0;
        m.l2_latency_loaded_ns = 20000.0;
        m
    };
    let a = run(&sync, slow(&sync));
    let b = run(&piped, slow(&piped));
    assert!(
        b.makespan_ns < a.makespan_ns,
        "{} vs {}",
        b.makespan_ns,
        a.makespan_ns
    );
}

#[test]
fn every_byte_is_accounted_for() {
    let k = gemm(46 * 3 + 5, 16, 2);
    let p = run(&k, machine(&k));
    let loads = f64::from(k.grid) * 16.0 * 16384.0;
    assert!(close(p.load_bytes, loads));
    assert!(close(p.l2_hit_bytes, loads * 0.5));
    assert!(close(
        p.dram_bytes,
        loads * 0.5 + f64::from(k.grid) * 32768.0
    ));
}

#[test]
fn fully_prefetched_loop_loads_each_iteration_once() {
    let k = gemm(46, 2, 4); // prefetch distance beyond the trip count
    let p = run(&k, machine(&k));
    assert!(close(p.load_bytes, 46.0 * 2.0 * 16384.0));
}

#[test]
fn bound_fraction_names_the_bounding_resource() {
    let k = compute_only(46);
    let p = run(&k, machine(&k));
    assert!(close(p.bound_fraction.iter().sum(), 1.0));
    assert!(close(p.bound_fraction[0], 1.0), "{:?}", p.bound_fraction);
}

/// Memory-only kernel with plenty of bandwidth and a fixed load latency.
fn latency_bound(prefetch_distance: u32) -> (Kernel, Machine) {
    let mut k = compute_only(46);
    k.tile.per_iteration = Work {
        loads: vec![access("load", 4096.0)],
        ..Work::default()
    };
    k.tile.iterations = 64;
    k.tile.prefetch_distance = prefetch_distance;
    let mut m = machine(&k);
    m.dram_bw = vec![(1.0, 1e9)];
    m.l2_bw = vec![(1.0, 1e9)];
    m.dram_latency_ns = 1000.0;
    m.dram_latency_loaded_ns = 1000.0;
    m.l2_latency_ns = 1000.0;
    m.l2_latency_loaded_ns = 1000.0;
    (k, m)
}

#[test]
fn latency_bound_time_follows_littles_law() {
    // One round trip for the prefill, then d + 1 stage buffers share each
    // round trip of the remaining 64 - d iterations.
    for d in [1, 2, 4] {
        let (k, m) = latency_bound(d);
        let p = run(&k, m);
        let trips = 1.0 + f64::from(64 - d) / f64::from(d + 1);
        assert!(
            close(p.makespan_ns, trips * 1000.0),
            "d={d}: {}",
            p.makespan_ns
        );
    }
}

#[test]
fn synchronous_loads_pay_one_round_trip_per_iteration() {
    let (k, m) = latency_bound(0);
    let p = run(&k, m);
    assert!(close(p.makespan_ns, 64.0 * 1000.0), "{}", p.makespan_ns);
}

#[test]
fn load_raises_latency() {
    // One CTA per SM with one 16 KiB stage in flight: bandwidth bound at idle
    // latency, latency bound once latency grows with load.
    let mut k = compute_only(46);
    k.tile.per_iteration = Work {
        loads: vec![access("load", 16384.0)],
        ..Work::default()
    };
    k.tile.iterations = 16;
    k.tile.prefetch_distance = 1;
    let idle = run(&k, machine(&k));
    let mut m = machine(&k);
    m.dram_latency_loaded_ns = 10.0 * m.dram_latency_ns;
    m.l2_latency_loaded_ns = 10.0 * m.l2_latency_ns;
    let loaded = run(&k, m);
    assert!(
        loaded.makespan_ns > 1.2 * idle.makespan_ns,
        "{} vs {}",
        loaded.makespan_ns,
        idle.makespan_ns
    );
}

#[test]
fn spills_cost_local_memory_bandwidth_per_scope() {
    let base = compute_only(46);
    let spilled = |scope| {
        let mut k = compute_only(46);
        k.spill = Spill {
            bytes_per_thread: 4096.0,
            stack_frame_bytes: 64,
            scope,
        };
        k
    };
    let mut m = machine(&base);
    m.local_bw = 1.0; // slow enough that per-iteration spills dominate
    let t = |k: &Kernel| run(k, m.clone()).makespan_ns;
    let none = t(&base);
    let once = t(&spilled(SpillScope::Once));
    let per_iter = spilled(SpillScope::PerIteration { assumed: false });
    let every = t(&per_iter);
    assert!(none < once && once < every, "{none} {once} {every}");
    // 64 iterations of 4096 B for 128 threads at 1 B/ns, one CTA per SM.
    assert!(close(every, 64.0 * 4096.0 * 128.0), "{every}");
    assert!(run(&per_iter, m.clone()).bound_fraction[3] > 0.99);
}

#[test]
fn steep_load_latency_settles_at_the_littles_law_equilibrium() {
    // 46 CTAs with one 16 KiB stage in flight each, all from DRAM. With
    // latency L(X) = L0 + (L1 - L0) X / P, delivered bandwidth X solves
    // X L(X) = F, and the run takes one round trip per iteration.
    let mut k = compute_only(46);
    k.tile.per_iteration = Work {
        loads: vec![access("load", 16384.0)],
        ..Work::default()
    };
    k.tile.iterations = 256;
    k.tile.prefetch_distance = 1;
    let mut m = machine(&k);
    let (l0, l1, peak) = (500.0, 20000.0, 450.0);
    m.dram_bw = vec![(1.0, peak)];
    m.l2_bw = vec![(1.0, 1e9)];
    m.dram_latency_ns = l0;
    m.dram_latency_loaded_ns = l1;
    m.launch_overhead_ns = 0.0;
    // The prefill holds one stage per CTA in flight, the steady state two
    // (prefetch distance 1, two buffers), sharing each round trip.
    let f = 46.0 * 16384.0;
    let a = (l1 - l0) / peak;
    let latency = |in_flight: f64| {
        let x = (-l0 + (l0 * l0 + 4.0 * a * in_flight).sqrt()) / (2.0 * a);
        l0 + a * x
    };
    let expected = latency(f) + 255.0 / 2.0 * latency(2.0 * f);
    let p = simulate(&k, &profile(), &m, &mut ConstantHitRatio(0.0)).unwrap();
    assert!(
        (p.makespan_ns / expected - 1.0).abs() < 2e-3,
        "{} vs {}",
        p.makespan_ns,
        expected
    );
}

#[test]
fn mlp_curve_sets_latency_from_bytes_in_flight() {
    // 46 CTAs with two 16 KiB stage buffers each and no compute: the loads
    // in flight reach where the MLP curve has saturated at 368 B/ns, so the
    // loop delivers 368 B/ns.
    let mut k = compute_only(46);
    k.tile.per_iteration = Work {
        loads: vec![access("load", 16384.0)],
        ..Work::default()
    };
    k.tile.iterations = 64;
    k.tile.prefetch_distance = 1;
    let mut m = machine(&k);
    m.dram_bw = vec![(1.0, 1e9)];
    m.l2_bw = vec![(1.0, 1e9)];
    let f = 46.0 * 16384.0;
    let curve = vec![(4096.0, 4096.0 / 500.0), (f / 4.0, 368.0), (2.0 * f, 368.0)];
    m.dram_mlp = [curve.clone(), curve.clone()];
    m.l2_mlp = [curve.clone(), curve];
    m.launch_overhead_ns = 0.0;
    let p = simulate(&k, &profile(), &m, &mut ConstantHitRatio(0.0)).unwrap();
    // The equilibrium is found by bisection to 0.2%.
    let expected = 64.0 * f / 368.0;
    assert!(
        (p.makespan_ns / expected - 1.0).abs() < 2e-3,
        "{} vs {expected}",
        p.makespan_ns
    );
}

#[test]
fn other_ctas_compute_while_a_synchronous_cta_waits() {
    // Compute per iteration equals the round trip. One CTA per SM pays both
    // in turn; two CTAs per SM overlap one's wait with the other's compute.
    let mut k = compute_only(46);
    k.smem_per_cta_bytes = 40 * 1024;
    k.tile.prefetch_distance = 0;
    k.tile.per_iteration.loads = vec![access("load", 64.0)];
    let mut m = machine(&k);
    m.dram_bw = vec![(1.0, 1e9)];
    m.l2_bw = vec![(1.0, 1e9)];
    let compute = k.tile.per_iteration.mma_flops / (0.60 * 1e3); // 4 warps alone
    m.dram_latency_ns = compute;
    m.dram_latency_loaded_ns = compute;
    m.l2_latency_ns = compute;
    m.l2_latency_loaded_ns = compute;
    let two_per_sm = run(&k, m.clone());
    assert_eq!(two_per_sm.occupancy.ctas_per_sm, 2);
    k.smem_per_cta_bytes = 90 * 1024;
    let one_per_sm = run(&k, m);
    // Same work per SM: one CTA takes 2 * 64 * compute alone; two CTAs take
    // twice as long each but share the wait, so the SM finishes in the same
    // time instead of twice it.
    let per_sm_one = 2.0 * 64.0 * compute;
    assert!(
        close(one_per_sm.makespan_ns, per_sm_one),
        "{}",
        one_per_sm.makespan_ns
    );
    assert!(
        two_per_sm.makespan_ns < 1.1 * per_sm_one,
        "{}",
        two_per_sm.makespan_ns
    );
}

#[test]
fn stage_buffers_queue_for_the_sm() {
    // Closed loop with three buffers (prefetch distance 2), a fixed 1000 ns
    // round trip and 400 ns of compute alone per stage. The prefill pays one
    // round trip; then the three buffers cycle through the SM's pipes and the
    // round trip, and mean value analysis gives the stage rate.
    let mut k = compute_only(46);
    k.tile.per_iteration.loads = vec![access("load", 64.0)];
    k.tile.prefetch_distance = 2;
    let compute = k.tile.per_iteration.mma_flops / (0.60 * 1e3);
    k.tile.per_iteration.mma_flops *= 400.0 / compute;
    let mut m = machine(&k);
    m.dram_bw = vec![(1.0, 1e9)];
    m.l2_bw = vec![(1.0, 1e9)];
    m.dram_latency_ns = 1000.0;
    m.dram_latency_loaded_ns = 1000.0;
    m.l2_latency_ns = 1000.0;
    m.l2_latency_loaded_ns = 1000.0;
    let p = run(&k, m);
    let (service, delay) = (400.0, 1000.0);
    let x1 = 1.0 / (service + delay);
    let q1 = x1 * service;
    let r2 = service * (1.0 + q1);
    let x2 = 2.0 / (r2 + delay);
    let r3 = service * (1.0 + x2 * r2);
    let x3 = 3.0 / (r3 + delay);
    let expected = 1000.0 + 64.0 / x3;
    assert!(
        close(p.makespan_ns, expected),
        "{} vs {expected}",
        p.makespan_ns
    );
    // Between the deterministic bound and serial execution.
    assert!(p.makespan_ns > 1000.0 + 64.0 * (1400.0f64 / 3.0).max(400.0));
    assert!(p.makespan_ns < 1000.0 + 64.0 * 1400.0);
}

/// Loads split by fixed fractions between resident hits, merged requests,
/// and DRAM fetches.
struct Split(f64, f64);

impl L2 for Split {
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
    ) -> Traffic {
        let bytes = a.nominal_bytes * f64::from(k1 - k0);
        if is_write {
            return Traffic {
                resident: 0.0,
                merged: 0.0,
                dram: bytes,
            };
        }
        Traffic {
            resident: bytes * self.0,
            merged: bytes * self.1,
            dram: bytes * (1.0 - self.0 - self.1),
        }
    }
}

#[test]
fn merged_requests_wait_for_dram_but_use_no_dram_bandwidth() {
    // Latency bound, one stage in flight per CTA: L2 latency 100 ns, DRAM
    // 1000 ns. Half the bytes merged into fills of the other half.
    let (k, mut m) = latency_bound(1);
    m.l2_latency_ns = 100.0;
    m.l2_latency_loaded_ns = 100.0;
    let p = |l2: &mut Split| simulate(&k, &profile(), &m, l2).unwrap();
    let resident = p(&mut Split(0.5, 0.0));
    let merged = p(&mut Split(0.0, 0.5));
    let trips = 1.0 + 63.0 / 2.0;
    assert!(
        close(resident.makespan_ns, trips * 550.0),
        "{}",
        resident.makespan_ns
    );
    assert!(
        close(merged.makespan_ns, trips * 1000.0),
        "{}",
        merged.makespan_ns
    );
    assert!(close(merged.dram_bytes, resident.dram_bytes));
    assert!(close(merged.merged_bytes, merged.l2_hit_bytes));
}

#[test]
fn stage_synchronization_stalls_the_sm_for_every_cta() {
    // Compute-only steady state with two stage buffers and 100 ns of
    // synchronization per stage. One CTA per SM pays it once per iteration;
    // two CTAs per SM share the pipes and each stalls them.
    let mut k = compute_only(46);
    let mut m = machine(&k);
    m.fixed_sync = vec![(4, 2, 100.0), (4, 4, 1.0), (8, 2, 1.0)];
    let compute = 64.0 * k.tile.per_iteration.mma_flops / (0.60 * 1e3);
    let one = run(&k, m.clone());
    assert!(
        close(one.makespan_ns, compute + 64.0 * 100.0),
        "{}",
        one.makespan_ns
    );
    k.grid = 92;
    k.smem_per_cta_bytes = 40 * 1024;
    let two = run(&k, m);
    assert_eq!(two.occupancy.ctas_per_sm, 2);
    // Eight warps per SM reach 0.63 TFLOP/s on the fixture curve.
    let shared = 2.0 * 64.0 * k.tile.per_iteration.mma_flops / (0.63 * 1e3);
    assert!(
        close(two.makespan_ns, shared + 2.0 * 64.0 * 100.0),
        "{}",
        two.makespan_ns
    );
}

#[test]
fn bound_names_bandwidth_only_when_it_binds() {
    use vgpu_core::sim::RESOURCES;
    let at = |name: &str| RESOURCES.iter().position(|&r| r == name).unwrap();
    // Compute bound with light loads and synchronization: the bandwidth share
    // equals the demand, which must not read as bandwidth bound.
    let mut k = compute_only(46);
    k.tile.per_iteration.loads = vec![access("load", 64.0)];
    let mut m = machine(&k);
    m.fixed_sync = vec![(4, 2, 10.0)];
    let p = run(&k, m);
    assert!(p.bound_fraction[at("mma")] > 0.99, "{:?}", p.bound_fraction);
    // DRAM bound: the bandwidth share binds.
    let mut k = compute_only(46 * 4);
    k.tile.per_iteration = Work {
        loads: vec![access("load", 1048576.0)],
        ..Work::default()
    };
    k.tile.iterations = 8;
    let p = run(&k, machine(&k));
    assert!(
        p.bound_fraction[at("dram_bandwidth")] > 0.99,
        "{:?}",
        p.bound_fraction
    );
    // Latency bound with plenty of bandwidth.
    let (k, m) = latency_bound(1);
    let p = run(&k, m);
    assert!(
        p.bound_fraction[at("dram_latency")] + p.bound_fraction[at("l2_latency")] > 0.99,
        "{:?}",
        p.bound_fraction
    );
}

#[test]
fn per_tile_iterations_set_each_programs_length() {
    // Causal-like: program p runs p + 1 iterations, one CTA per SM, so the
    // last program alone sets the makespan.
    let mut k = compute_only(46);
    k.tile_iterations = Some((0..46).map(|p| vec![p + 1]).collect());
    let one = k.tile.per_iteration.mma_flops / (0.60 * 1e3);
    let p = run(&k, machine(&k));
    assert!(
        close(p.makespan_ns, 46.0 * one),
        "{} vs {}",
        p.makespan_ns,
        46.0 * one
    );
}

#[test]
fn special_function_and_shuffle_pipes_bound_their_instructions() {
    // Only SFU work: 64 iterations x 4 warps x 100 instructions at the SFU
    // rate; the same count of ALU instructions is bound by the faster issue
    // rate, and SFU instructions also take issue slots.
    let mut k = compute_only(46);
    k.tile.per_iteration = Work {
        sfu_insts_per_warp: 100.0,
        ..Work::default()
    };
    let mut m = machine(&k);
    // Four warps per SM read the 4-warp point of each curve.
    m.sfu = vec![(1.0, 0.25), (4.0, 1.0), (32.0, 4.0)];
    m.issue_rate = 10.0;
    m.shfl = vec![(1.0, 0.5), (4.0, 2.0), (32.0, 8.0)];
    let sfu = run(&k, m.clone());
    assert!(
        close(sfu.makespan_ns, 64.0 * 4.0 * 100.0 / 1.0),
        "{}",
        sfu.makespan_ns
    );
    assert!(sfu.bound_fraction[4] > 0.99, "{:?}", sfu.bound_fraction);
    k.tile.per_iteration = Work {
        shfl_insts_per_warp: 100.0,
        ..Work::default()
    };
    let shfl = run(&k, m.clone());
    assert!(
        close(shfl.makespan_ns, 64.0 * 4.0 * 100.0 / 2.0),
        "{}",
        shfl.makespan_ns
    );
    k.tile.per_iteration = Work {
        alu_insts_per_warp: 100.0,
        sfu_insts_per_warp: 1.0,
        ..Work::default()
    };
    let issue = run(&k, m);
    assert!(
        close(issue.makespan_ns, 64.0 * 4.0 * 101.0 / 10.0),
        "{}",
        issue.makespan_ns
    );
}
