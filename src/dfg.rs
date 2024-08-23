use std::collections::{BTreeSet, HashMap};
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};
use petgraph::Direction;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use pyo3::{pyclass, pymethods, Bound, PyResult};
use pyo3::types::{PyAnyMethods, PyDict};

#[pyclass]
#[derive(Debug)]
pub struct DirectlyFollowsGraph {
    inner: DiGraph<String, usize>,
    node_weight_index_map: HashMap<String, NodeIndex>,
    pub start_idx: NodeIndex,
    pub end_idx: NodeIndex,
    pub start_activities: BTreeSet<String>,
    pub end_activities: BTreeSet<String>,
}

#[pyclass]
#[derive(Clone)]
#[allow(dead_code)]
pub struct NodeIdx(NodeIndex);

#[pyclass]
#[allow(dead_code)]
pub struct EdgeIdx(EdgeIndex);

#[pymethods]
impl DirectlyFollowsGraph {
    #[staticmethod]
    pub fn try_from_dict(dict: &Bound<PyDict>) -> PyResult<Self> {
        let dict: HashMap<(String, String), usize> = dict.extract()?;
        let mut node_idx_map = HashMap::new();
        let mut graph = DiGraph::new();

        for ((a, b), weight) in dict {
            let a_idx = *node_idx_map.entry(a.clone()).or_insert_with(|| graph.add_node(a));
            let b_idx = *node_idx_map.entry(b.clone()).or_insert_with(|| graph.add_node(b));
            graph.update_edge(a_idx, b_idx, weight);
        }

        let start_idx = node_idx_map["start"];
        let end_idx = node_idx_map["end"];
        let start_activities = graph.neighbors(start_idx).map(|i| graph[i].clone()).collect();
        let end_activities = graph.neighbors_directed(end_idx, Direction::Incoming).map(|i| graph[i].clone()).collect();


        Ok(DirectlyFollowsGraph {
            inner: graph,
            node_weight_index_map: node_idx_map,
            start_idx,
            end_idx,
            start_activities,
            end_activities,
        })
    }

    pub fn display(&self) {
        println!("{:?}", self)
    }
}

impl From<DiGraph<String, usize>> for DirectlyFollowsGraph {
    fn from(graph: DiGraph<String, usize>) -> Self {
        let mut node_weight_index_map = HashMap::new();
        for n_idx in graph.node_indices() {
            node_weight_index_map.insert(graph[n_idx].clone(), n_idx);
        }

        let start_idx = node_weight_index_map["start"];
        let end_idx = node_weight_index_map["end"];
        let start_activities = graph.neighbors(start_idx).map(|i| graph[i].clone()).collect();
        let end_activities = graph.neighbors_directed(end_idx, Direction::Incoming).map(|i| graph[i].clone()).collect();

        Self {
            inner: graph,
            node_weight_index_map,
            start_idx,
            end_idx,
            start_activities,
            end_activities,
        }
    }
}

impl DirectlyFollowsGraph {
    pub fn get_node_index(&self, weight: &str) -> Option<NodeIndex> {
        self.node_weight_index_map.get(weight).copied()
    }

    pub fn add_start_and_end(&self, set: &mut BTreeSet<String>) {
        let has_start_activities = set.intersection(&self.start_activities).next().is_some();
        let has_end_activities = set.intersection(&self.end_activities).next().is_some();

        if has_start_activities {
            set.insert("start".to_string());
        }

        if has_end_activities {
            set.insert("end".to_string());
        }
    }

    pub fn subgraph(&self, nodes: &BTreeSet<NodeIndex>) -> DiGraph<String, usize> {
        let mut subgraph = DiGraph::with_capacity(nodes.len(), nodes.len());
        let mut node_map = HashMap::new();

        for &node in nodes {
            let node_weight = &self[node];
            let node_idx = subgraph.add_node(node_weight.clone());
            node_map.insert(node, node_idx);
        }

        for &node in nodes {
            for neighbor in self.neighbors(node) {
                if nodes.contains(&neighbor) {
                    let edge_weight = *self.edge_weight(self.find_edge(node, neighbor).unwrap()).unwrap();
                    let source = node_map[&node];
                    let target = node_map[&neighbor];
                    subgraph.add_edge(source, target, edge_weight);
                }
            }
        }

        subgraph
    }

    pub fn subgraph_by_weights(&self, weights: &BTreeSet<String>) -> DiGraph<String, usize> {
        let nodes = weights.iter().map(|w| self.get_node_index(w).unwrap()).collect();
        self.subgraph(&nodes)
    }

    pub fn all_neighbors_weights(&self, nodes: &BTreeSet<String>) -> BTreeSet<String> {
        let mut neighbors = BTreeSet::new();
        for node in nodes {
            let node_idx = self.get_node_index(node).unwrap();
            for neighbor in self.neighbors(node_idx) {
                neighbors.insert(self[neighbor].clone());
            }
        }
        neighbors
    }
}

impl Deref for DirectlyFollowsGraph {
    type Target = DiGraph<String, usize>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for DirectlyFollowsGraph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Display for DirectlyFollowsGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}
