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
