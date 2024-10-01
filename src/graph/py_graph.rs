use std::{
    collections::{BTreeSet, HashMap, HashSet},
    ops::{Deref, Index},
};

use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::{Dfs, EdgeRef, Reversed},
    Undirected,
};
use pyo3::{types::PyAnyMethods, Bound, FromPyObject, PyAny, PyResult};

#[derive(Debug)]
pub struct PyGraph<'a> {
    pub graph: DiGraph<&'a str, f64>,
    pub node_weight_index_map: HashMap<&'a str, NodeIndex>,
}

impl<'py> FromPyObject<'py> for PyGraph<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let edges_obj = ob.getattr("edges")?;
        let edges_iter = edges_obj.call_method1("data", ("weight",))?.iter()?;

        let mut graph = DiGraph::new();
        let mut node_weight_index_map = HashMap::new();
        for item in edges_iter {
            let (source, target, weight) = item?.extract::<(&str, &str, f64)>()?;
            let source_idx = *node_weight_index_map
                .entry(source)
                .or_insert_with(|| graph.add_node(source));
            let target_idx = *node_weight_index_map
                .entry(target)
                .or_insert_with(|| graph.add_node(target));
            graph.update_edge(source_idx, target_idx, weight);
        }

        Ok(PyGraph {
            graph,
            node_weight_index_map,
        })
    }
}

impl PyGraph<'_> {
    pub fn contains_node_weight(&self, node_weight: &str) -> bool {
        self.node_weight_index_map.contains_key(node_weight)
    }

    pub fn out_degree(&self, node_weight: &str) -> f64 {
        let node_idx = self[node_weight];
        self.graph.edges(node_idx).map(|edge| edge.weight()).sum()
    }

    pub fn nodes_out_degree(&self, node_weights: &HashSet<&str>) -> f64 {
        node_weights
            .iter()
            .map(|node_weight| self.out_degree(node_weight))
            .sum()
    }

    pub fn get_node_index(&self, node_weight: &str) -> Option<NodeIndex> {
        self.node_weight_index_map.get(node_weight).copied()
    }

    pub fn subgraph(&self, nodes: &BTreeSet<NodeIndex>) -> Self {
        let mut subgraph = DiGraph::with_capacity(nodes.len(), nodes.len());
        let mut node_map = HashMap::new();
        let mut node_weight_index_map = HashMap::new();

        for &node in nodes {
            let node_weight = self.graph[node];
            let node_idx = subgraph.add_node(node_weight);
            node_weight_index_map.insert(node_weight, node_idx);
            node_map.insert(node, node_idx);
        }

        for &source in nodes {
            for edge in self.edges(source) {
                let target = edge.target();
                if nodes.contains(&target) {
                    let edge_weight = *edge.weight();
                    let new_source = node_map[&source];
                    let new_target = node_map[&target];
                    subgraph.add_edge(new_source, new_target, edge_weight);
                }
            }
        }

        Self {
            graph: subgraph,
            node_weight_index_map,
        }
    }

    pub fn subgraph_by_weights(&self, weights: &BTreeSet<&str>) -> Self {
        let nodes = weights.iter().map(|w| self[w]).collect();
        self.subgraph(&nodes)
    }

    pub fn all_nodes_are_reachable_from(&self, start: NodeIndex) -> bool {
        let mut dfs = Dfs::new(&self.graph, start);
        let mut visited = HashSet::new();

        while let Some(node) = dfs.next(&self.graph) {
            visited.insert(node);
        }

        visited.len() == self.graph.node_count()
    }

    pub fn is_weakly_connected(&self) -> bool {
        let undirected_graph = self.graph.clone().into_edge_type::<Undirected>();
        let mut dfs = Dfs::new(&undirected_graph, self.graph.node_indices().next().unwrap());
        let mut visited = HashSet::new();

        while let Some(node) = dfs.next(&undirected_graph) {
            visited.insert(node);
        }

        visited.len() == self.graph.node_count()
    }
}

impl<'a> PyGraph<'a> {
    pub fn ancestors(&self, node: NodeIndex) -> BTreeSet<&'a str> {
        let reversed_graph = Reversed(&self.graph);
        let mut dfs = Dfs::new(&reversed_graph, node);

        let mut ancestors = BTreeSet::new();

        while let Some(ancestor) = dfs.next(&reversed_graph) {
            if ancestor != node {
                ancestors.insert(self.graph[ancestor]);
            }
        }

        ancestors
    }

    pub fn all_neighbors_weights(&self, weights: &BTreeSet<&str>) -> BTreeSet<&'a str> {
        weights
            .iter()
            .flat_map(|&node_weight| {
                let node_idx = self[node_weight];
                self.neighbors(node_idx)
                    .map(|neighbor| self.graph[neighbor])
                    .collect::<BTreeSet<&str>>()
            })
            .collect::<BTreeSet<&str>>()
    }
}

impl Index<&str> for PyGraph<'_> {
    type Output = NodeIndex;

    fn index(&self, node_weight: &str) -> &Self::Output {
        &self.node_weight_index_map[node_weight]
    }
}

impl<'a> Deref for PyGraph<'a> {
    type Target = DiGraph<&'a str, f64>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}
