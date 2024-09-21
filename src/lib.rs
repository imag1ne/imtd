mod algorithms;
mod cost;
mod distance;
mod graph;
mod mapping;

use pyo3::prelude::*;

use crate::algorithms::find_possible_partitions;
use crate::cost::evaluate_cuts;
use crate::distance::distance_matrix;
use crate::graph::py_graph::PyGraph;
use crate::mapping::edge_case_id_mapping;

#[pyfunction]
fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[pymodule]
fn imtd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(find_possible_partitions, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_cuts, m)?)?;
    m.add_function(wrap_pyfunction!(distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(edge_case_id_mapping, m)?)?;

    Ok(())
}
