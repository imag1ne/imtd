mod fit;
pub mod imbi;
pub mod imfc;
mod utils;

use crate::graph::py_graph::PyGraph;
use std::collections::HashSet;

pub type CutEvaluations<'a> = Vec<(
    (HashSet<&'a str>, HashSet<&'a str>),
    String,
    f64,
    f64,
    f64,
    f64,
)>;

pub trait CutCost {
    fn evaluate(
        &self,
        graph: &PyGraph,
        node_set_1: &HashSet<&str>,
        node_set_2: &HashSet<&str>,
    ) -> f64;
}
