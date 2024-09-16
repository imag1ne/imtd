use std::{collections::{HashMap, HashSet}, ops::Index};

use petgraph::graph::{DiGraph, NodeIndex};
use pyo3::{types::PyAnyMethods, Bound, FromPyObject, PyAny, PyResult};

#[derive(Debug)]
pub struct PyGraph<'a>{
    pub graph: DiGraph<&'a str, f64>,
    pub node_weight_index_map: HashMap<&'a str, NodeIndex>,
}

impl<'a, 'py: 'a> FromPyObject<'py> for PyGraph<'a> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let edges_obj = ob.getattr("edges")?;
        let edges_iter = edges_obj.call_method1("data", ("weight",))?.iter()?;

        let mut graph = DiGraph::new();
        let mut node_weight_index_map = HashMap::new();
        for item in edges_iter {
            let (source, target, weight) = item?.extract::<(&str, &str, f64)>()?;
            let source_idx = *node_weight_index_map.entry(source).or_insert_with(|| graph.add_node(source));
            let target_idx = *node_weight_index_map.entry(target).or_insert_with(|| graph.add_node(target));
            graph.update_edge(source_idx, target_idx, weight);
        }
        
        Ok(PyGraph {
            graph,
            node_weight_index_map,
        })
    }
}

impl PyGraph<'_> {
    pub fn out_degree(&self, node_weight: &str) -> f64 {
        let node_idx = self[node_weight];
        self.graph.edges(node_idx).map(|edge| edge.weight()).sum()
    }

    pub fn nodes_out_degree(&self, node_weights: &HashSet<&str>) -> f64 {
        node_weights.iter().map(|node_weight| self.out_degree(node_weight)).sum()
    }
}

impl Index<&str> for PyGraph<'_> {
    type Output = NodeIndex;

    fn index(&self, node_weight: &str) -> &Self::Output {
        &self.node_weight_index_map[node_weight]
    }
}