//! The M1 chain: example KernelIR + measured RTX 4070 profile -> prediction.

use vgpu_core::from_ir::{kernel_from_ir, IrL2};
use vgpu_core::ir::KernelIR;
use vgpu_core::l2::{L2InitState, L2Options};
use vgpu_core::profile::Profile;
use vgpu_core::sim::{simulate, Machine};

const IR: &str = include_str!("../../../schemas/examples/gemm_sm89.json");
const PROFILES: [&str; 2] = [
    include_str!("../../../profiles/rtx4070.json"),
    include_str!("../../../schemas/examples/profile_example.json"),
];

#[test]
fn example_gemm_runs_on_every_profile() {
    let ir = KernelIR::from_json_str(IR).unwrap();
    ir.validate().unwrap();
    let k = kernel_from_ir(&ir).unwrap();
    for text in PROFILES {
        let p = Profile::from_json(text).unwrap();
        let m = Machine::nominal(&p, &k.mma_key).unwrap();
        let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
        let r = simulate(&k, &p, &m, &mut l2).unwrap();
        assert!(r.time_ns.is_finite() && r.time_ns > 0.0);
        assert!(r.dram_bytes > 0.0 && r.dram_bytes <= r.load_bytes + 4096.0 * 4096.0 * 2.0);
        eprintln!(
            "{}: {:.1} us, {:?}, waves {:.2}, tail {:.3}, L2 hit {:.2}",
            p.identity.sku,
            r.time_ns / 1e3,
            r.occupancy,
            r.waves,
            r.tail_efficiency,
            r.l2_hit_bytes / r.load_bytes
        );
    }
}

#[test]
fn causal_attention_example_runs_with_per_tile_iterations() {
    let ir = KernelIR::from_json_str(include_str!(
        "../../../schemas/examples/flash_attention_causal_gqa_sm89.json"
    ))
    .unwrap();
    ir.validate().unwrap();
    let k = kernel_from_ir(&ir).unwrap();
    let per_tile = k
        .tile_iterations
        .as_ref()
        .expect("causal attention has per-tile trip counts");
    assert!(per_tile.iter().flatten().any(|&n| n != per_tile[0][0]));
    assert!(
        k.tile.per_iteration.sfu_insts_per_warp > 0.0
            && k.tile.per_iteration.shfl_insts_per_warp > 0.0
    );
    let p = Profile::from_json(PROFILES[0]).unwrap();
    let m = Machine::nominal(&p, &k.mma_key).unwrap();
    let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
    let r = simulate(&k, &p, &m, &mut l2).unwrap();
    assert!(r.time_ns.is_finite() && r.time_ns > 0.0);
}

#[test]
fn ascend910b3_profile_simulates_cube_gemm_and_enforces_sram_limits() {
    use vgpu_core::occupancy::{occupancy, OccupancyError};

    let p = Profile::from_json(include_str!("../../../profiles/ascend910b3.json")).unwrap();
    assert_eq!(p.identity.vendor, "ascend");
    assert_eq!(p.identity.arch, "ascend910b3");
    assert_eq!(p.specs.sm_count, 20);
    let asc = p.specs.ascend.as_ref().expect("ascend specs present");
    assert_eq!(asc.l1_bytes, 524_288);
    assert_eq!(asc.l0c_bytes, 131_072);
    assert_eq!(asc.ub_bytes, 196_608);

    // 2048x2048x2048 FP16 GEMM, 128x128x64 tile, auto-blockified onto 20 AI Cores (12-13 tiles/core).
    let mut ir = KernelIR::from_json_str(IR).unwrap();
    ir.metadata.target_vendor = "ascend".into();
    ir.metadata.target_arch = "ascend910b3".into();
    ir.metadata.frontend = "triton_ascend".into();
    ir.launch.grid = [20, 1, 1];
    ir.launch.num_warps = 1;
    ir.launch.warp_size = 32;
    ir.launch.warp_groups[0].num_warps = 1;
    ir.resources.registers_per_thread = 1;
    ir.resources.static_smem_bytes = 65_536;
    ir.resources.ub_bytes = Some(0);
    ir.resources.l0c_bytes = Some(65_536);
    ir.resources.workspace_bytes = Some(0);
    ir.body.outer_loop.trip_count_mode = vgpu_core::ir::OuterTripCountMode::Uniform;
    ir.body.outer_loop.trip_counts = vgpu_core::ir::TripCounts::Uniform(12);
    ir.body.pipeline_loop.trip_counts = vgpu_core::ir::TripCounts::Uniform(32);
    ir.body.pipeline_loop.num_stages = 2;
    ir.body.pipeline_loop.prefetch_distance = 1;
    for op in &mut ir.body.pipeline_loop.ops {
        if let vgpu_core::ir::Op::Mma(m) = op {
            m.op_class = "cube.mmad".into();
            m.tile_mnk = [128, 128, 64];
            m.inst_mnk = [16, 16, 16];
            m.count = 8 * 8 * 4;
        }
    }
    for entry in ir.access_map.values_mut() {
        if let vgpu_core::ir::AccessEntry::Strided2dPerOuterTile { entries, .. } = entry {
            entries.truncate(240);
        }
    }
    ir.validate().unwrap();

    let k = kernel_from_ir(&ir).unwrap();
    let m = Machine::nominal(&p, &k.mma_key).unwrap();
    let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
    let r = simulate(&k, &p, &m, &mut l2).unwrap();
    assert!(r.time_ns.is_finite() && r.time_ns > 0.0);
    assert_eq!(r.occupancy.ctas_per_sm, 1);

    // UB overflow (> 192 KiB) must be rejected by occupancy().
    let mut bad_ub = k.clone();
    bad_ub.ub_bytes = asc.ub_bytes + 32;
    assert!(matches!(
        occupancy(&bad_ub, &p.specs),
        Err(OccupancyError::TooMuchSharedMemory { .. })
    ));

    // L0C overflow (> 128 KiB) must be rejected by occupancy().
    let mut bad_l0c = k;
    bad_l0c.l0c_bytes = asc.l0c_bytes + 32;
    assert!(matches!(
        occupancy(&bad_l0c, &p.specs),
        Err(OccupancyError::TooMuchSharedMemory { .. })
    ));
}
