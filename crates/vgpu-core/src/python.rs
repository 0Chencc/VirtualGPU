//! PyO3 bindings, exposed to Python as `virtualgpu._core`.

use crate::from_ir::{kernel_from_ir, IrL2};
use crate::ir::KernelIR;
use crate::l2::{L2InitState, L2Options};
use crate::profile::Profile;
use crate::sim::{simulate, Machine};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{e:?}"))
}

/// Predict one kernel variant on one GPU profile at the nominal draw.
///
/// Takes the KernelIR and profile as JSON strings and `l2_state` as "cold" or
/// "warm". Returns a dict with the predicted time and diagnostics.
#[pyfunction]
#[pyo3(signature = (kernel_ir_json, profile_json, l2_state = "cold"))]
fn predict<'py>(
    py: Python<'py>,
    kernel_ir_json: &str,
    profile_json: &str,
    l2_state: &str,
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
    let p = Profile::from_json(profile_json).map_err(err)?;
    let k = kernel_from_ir(&ir).map_err(err)?;
    let m = Machine::nominal(&p, &k.mma_key).map_err(err)?;
    let mut l2 = IrL2::new(&ir, &p, initial, L2Options::default());
    let r = simulate(&k, &p, &m, &mut l2).map_err(err)?;

    let d = PyDict::new(py);
    d.set_item("time_ns", r.time_ns)?;
    d.set_item("makespan_ns", r.makespan_ns)?;
    d.set_item("ctas_per_sm", r.occupancy.ctas_per_sm)?;
    d.set_item("occupancy_limiter", format!("{:?}", r.occupancy.limiter))?;
    d.set_item("waves", r.waves)?;
    d.set_item("tail_efficiency", r.tail_efficiency)?;
    d.set_item("load_bytes", r.load_bytes)?;
    d.set_item("l2_hit_bytes", r.l2_hit_bytes)?;
    d.set_item("dram_bytes", r.dram_bytes)?;
    d.set_item("l2_state", l2_state)?;
    d.set_item("dispatch_policy", "round_robin")?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    Ok(())
}
