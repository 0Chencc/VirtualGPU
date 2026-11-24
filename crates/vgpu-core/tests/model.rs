use vgpu_core::kernel::{Access, Kernel, Tile, TripCounts, Work};
use vgpu_core::occupancy::{occupancy, Limiter, OccupancyError};
use vgpu_core::profile::Profile;
use vgpu_core::sim::{simulate, ConstantHitRatio, Machine, Prediction};

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
    let sync = gemm(46 * 8, 128, 0);
    let piped = gemm(46 * 8, 128, 2);
    let a = run(&sync, machine(&sync));
    let b = run(&piped, machine(&piped));
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
    assert!(close(p.bound_fraction[1], 1.0), "{:?}", p.bound_fraction);
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
    // One round trip per `prefetch_distance` iterations, including the prefill.
    for d in [1, 2, 4] {
        let (k, m) = latency_bound(d);
        let p = run(&k, m);
        assert!(
            close(p.makespan_ns, 64.0 / f64::from(d) * 1000.0),
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
