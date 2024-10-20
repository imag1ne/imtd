mod algorithms;
mod cost;
mod distance;
mod graph;
mod mapping;
mod pm4py;

use pyo3::prelude::*;

use crate::algorithms::{filter_dfg, find_possible_partitions};
use crate::cost::{imbi::evaluate_cuts_imbi, imfc::evaluate_cuts_imfc};
use crate::distance::distance_matrix;
use crate::mapping::edge_case_id_mapping;

#[pyfunction]
fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[pymodule]
fn imtd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(find_possible_partitions, m)?)?;
    m.add_function(wrap_pyfunction!(filter_dfg, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_cuts_imfc, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_cuts_imbi, m)?)?;
    m.add_function(wrap_pyfunction!(distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(edge_case_id_mapping, m)?)?;

    Ok(())
}
