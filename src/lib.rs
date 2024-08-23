mod dfg;
mod algorithms;
mod distance;

use pyo3::prelude::*;

use crate::dfg::{DirectlyFollowsGraph, EdgeIdx, NodeIdx};
use crate::algorithms::find_possible_partitions;
use crate::distance::distance_matrix;

#[pyfunction]
fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[pymodule]
fn imtd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(find_possible_partitions, m)?)?;
    m.add_function(wrap_pyfunction!(distance_matrix, m)?)?;

    m.add_class::<DirectlyFollowsGraph>()?;
    m.add_class::<NodeIdx>()?;
    m.add_class::<EdgeIdx>()?;

    Ok(())
}
