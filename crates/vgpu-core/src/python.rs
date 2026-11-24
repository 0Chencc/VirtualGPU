//! PyO3 bindings, exposed to Python as `virtualgpu._core`.

use crate::draws::{draws as draw_profiles, Draw};
use crate::from_ir::{kernel_from_ir, IrL2};
use crate::ir::KernelIR;
use crate::kernel::Kernel;
use crate::l2::L2InitState;
use crate::sim::{simulate, Machine, ModelError, RESOURCES};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{e:?}"))
}

/// Map `f` over the draws on all available cores, keeping the draw order.
fn parallel_times<F>(draws: &[Draw], f: F) -> Result<Vec<f64>, ModelError>
where
    F: Fn(&Draw) -> Result<f64, ModelError> + Sync,
{
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = draws.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = draws
            .chunks(chunk)
            .map(|part| scope.spawn(|| part.iter().map(&f).collect::<Result<Vec<_>, _>>()))
            .collect();
        let mut out = Vec::with_capacity(draws.len());
        for h in handles {
            out.extend(h.join().expect("draw thread panicked")?);
        }
        Ok(out)
    })
}

/// Predict one kernel variant on one GPU profile under perturbation draws.
///
/// Takes the KernelIR and profile as JSON strings, `l2_state` as "cold" or
/// "warm", the number of perturbed draws and their seed. Returns a dict with
/// `status`: "ok" with the draw 0 (unperturbed) prediction and diagnostics
/// plus `draws`, the time per draw ID including "0"; or "unsupported" with a
/// `reason` (for example a CTA that does not fit on an SM). Draws depend only
/// on the profile, `draws` and `seed`, so variants predicted with the same
/// arguments share them. Malformed input raises `ValueError`.
#[pyfunction]
#[pyo3(signature = (kernel_ir_json, profile_json, l2_state = "cold", draws = 32, seed = 17))]
fn predict<'py>(
    py: Python<'py>,
    kernel_ir_json: &str,
    profile_json: &str,
    l2_state: &str,
    draws: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let initial = match l2_state {
        "cold" => L2InitState::Cold,
        "warm" => L2InitState::Warm,
        other => {
            return Err(PyValueError::new_err(format!(
                "l2_state must be cold or warm, got {other}"
            )))
        }
    };
    let ir = KernelIR::from_json_str(kernel_ir_json).map_err(err)?;
    ir.validate().map_err(err)?;
    let all = draw_profiles(profile_json, draws, seed).map_err(err)?;
    let nominal = &all[0].profile;

    let d = PyDict::new(py);
    d.set_item("l2_state", l2_state)?;
    d.set_item("dispatch_policy", "round_robin")?;
    let one = |k: &Kernel, draw: &Draw| {
        let mut m = Machine::nominal(&draw.profile, &k.mma_key)?;
        m.assumed_spill_scale = draw.assumed_spill_scale;
        let mut l2 = IrL2::new(&ir, &draw.profile, initial, draw.l2);
        simulate(k, &draw.profile, &m, &mut l2)
    };
    let result = kernel_from_ir(&ir).and_then(|k| {
        let r = one(&k, &all[0])?;
        let rest = py.detach(|| parallel_times(&all[1..], |draw| one(&k, draw).map(|p| p.time_ns)));
        let mut times = vec![r.time_ns];
        times.extend(rest?);
        Ok((r, times))
    });
    let (r, times) = match result {
        Ok(x) => x,
        Err(ModelError::EmptyCurve(name)) => {
            return Err(PyValueError::new_err(format!(
                "profile curve {name} is empty"
            )))
        }
        Err(e) => {
            d.set_item("status", "unsupported")?;
            d.set_item("reason", format!("{e:?}"))?;
            return Ok(d);
        }
    };
    d.set_item("status", "ok")?;
    d.set_item("time_ns", r.time_ns)?;
    let by_draw = PyDict::new(py);
    for (draw, t) in all.iter().zip(times) {
        by_draw.set_item(draw.id.to_string(), t)?;
    }
    d.set_item("draws", by_draw)?;
    d.set_item("makespan_ns", r.makespan_ns)?;
    d.set_item("ctas_per_sm", r.occupancy.ctas_per_sm)?;
    d.set_item("occupancy_limiter", format!("{:?}", r.occupancy.limiter))?;
    d.set_item("waves", r.waves)?;
    d.set_item("tail_efficiency", r.tail_efficiency)?;
    d.set_item("load_bytes", r.load_bytes)?;
    d.set_item("l2_hit_bytes", r.l2_hit_bytes)?;
    d.set_item("dram_bytes", r.dram_bytes)?;
    let bound = PyDict::new(py);
    for (name, f) in RESOURCES.iter().zip(r.bound_fraction) {
        bound.set_item(name, f)?;
    }
    d.set_item("bound_fraction", bound)?;
    let mut flags: Vec<String> = Vec::new();
    if ir.resources.spill_loads_bytes + ir.resources.spill_stores_bytes > 0 {
        flags.push("spill".into());
    }
    if !ir.diagnostics.is_empty() {
        flags.push("unvalidated_feature".into());
    }
    let dispatch = &nominal.measured.cta_dispatch;
    if dispatch.candidate_policies.len() > 1 || dispatch.confidence.as_deref() != Some("high") {
        flags.push("dispatch_ambiguous".into());
    }
    d.set_item("flags", flags)?;
    d.set_item("diagnostics", ir.diagnostics.clone())?;
    Ok(d)
}

/// The JSON Schema contracts, embedded at build time so installed wheels have them.
#[pyfunction]
fn schema(name: &str) -> PyResult<&'static str> {
    match name {
        "kernel_ir" => Ok(include_str!("../../../schemas/kernel_ir.schema.json")),
        "gpu_profile" => Ok(include_str!("../../../schemas/gpu_profile.schema.json")),
        other => Err(PyValueError::new_err(format!(
            "unknown schema {other}; expected kernel_ir or gpu_profile"
        ))),
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    Ok(())
}
