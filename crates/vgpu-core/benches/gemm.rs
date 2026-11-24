//! Performance budget from `docs/proposals/model-core.md` 3.1: a 4096-tile
//! GEMM should simulate in about 1 ms.
//!
//! Run with `cargo bench -p vgpu-core --bench gemm`.

use std::time::Instant;
use vgpu_core::kernel::{Kernel, Tile, TripCounts, Work};
use vgpu_core::profile::Profile;
use vgpu_core::sim::{simulate, ConstantHitRatio, Machine};

fn main() {
    let p = Profile::from_json(include_str!("../tests/data/test_fixture_sm89.json")).unwrap();
    // M = N = 8192 with 128x128 tiles, K = 4096 with BK = 32, num_stages = 3.
    let k = Kernel {
        grid: 64 * 64,
        num_warps: 4,
        registers_per_thread: 128,
        smem_per_cta_bytes: 3 * 16384,
        tiles_per_program: TripCounts::Uniform(1),
        mma_key: "mma.sync:f16:f16:f32:16x8x16".to_string(),
        tile: Tile {
            prologue: Work::default(),
            iterations: 128,
            per_iteration: Work {
                mma_flops: 2.0 * 128.0 * 128.0 * 32.0,
                alu_insts_per_warp: 16.0,
                load_bytes: 16384.0,
                store_bytes: 0.0,
            },
            prefetch_distance: 2,
            epilogue: Work {
                store_bytes: 32768.0,
                alu_insts_per_warp: 64.0,
                ..Work::default()
            },
        },
    };
    let m = Machine::nominal(&p, &k.mma_key).unwrap();

    let runs = 200;
    let mut last = None;
    let start = Instant::now();
    for _ in 0..runs {
        last = Some(simulate(&k, &p, &m, &mut ConstantHitRatio(0.5)).unwrap());
    }
    let per_run = start.elapsed().as_secs_f64() / f64::from(runs);
    let r = last.unwrap();
    println!(
        "tiles: {}, events: {}, predicted: {:.1} us",
        k.grid,
        r.events,
        r.time_ns / 1e3
    );
    println!("per simulation: {:.3} ms", per_run * 1e3);
}
