use crate::graph::py_graph::PyGraph;
use petgraph::Direction;
use pyo3::pyfunction;
use std::collections::HashMap;
use std::{
    collections::{BTreeSet, HashSet},
    vec,
};

#[pyfunction]
pub fn find_possible_partitions(
    graph: PyGraph<'_>,
) -> Vec<(BTreeSet<&str>, BTreeSet<&str>, HashSet<&str>)> {
    let activities = graph
        .node_weights()
        .filter(|&w| *w != "start" && *w != "end")
        .copied()
        .collect::<BTreeSet<_>>();
    let start_activities = graph
        .neighbors(graph["start"])
        .map(|i| graph.graph[i])
        .collect::<BTreeSet<_>>();
    let end_activities = graph
        .neighbors_directed(graph["end"], Direction::Incoming)
        .map(|i| graph.graph[i])
        .collect::<BTreeSet<_>>();

    let mut queue = vec![(BTreeSet::new(), BTreeSet::from(["start"]))];
    let mut visited = HashSet::new();
    let mut valid = vec![];

    while let Some((current_state, current_adj)) = queue.pop() {
        for &x in &current_adj {
            let x_idx = graph[x];
            let mut new_state = &current_state | &BTreeSet::from([x]);
            add_start_and_end(&mut new_state, &start_activities, &end_activities);

            if !visited.contains(&new_state) {
                visited.insert(new_state.clone());
                let new_adj = &(&current_adj
                    | &graph.neighbors(x_idx).map(|i| graph.graph[i]).collect())
                    - &new_state;
                queue.push((new_state.clone(), new_adj.clone()));

                let mut remaining_activities = &activities - &new_state;
                if remaining_activities.is_empty() || remaining_activities.len() == activities.len()
                {
                    continue;
                }

                add_start_and_end(
                    &mut remaining_activities,
                    &start_activities,
                    &end_activities,
                );
                let graph_remaining_nodes = graph.subgraph_by_weights(&remaining_activities);

                if let Some(end) = graph_remaining_nodes.get_node_index("end") {
                    let mut disconnected_nodes_to_end_in_graph_part_b =
                        &graph_remaining_nodes.node_weights().copied().collect()
                            - &graph_remaining_nodes.ancestors(end);
                    disconnected_nodes_to_end_in_graph_part_b.remove("end");

                    if disconnected_nodes_to_end_in_graph_part_b.is_empty() {
                        let new_state_has_end = new_state.contains("end");
                        let remaining_activities_has_start = remaining_activities.contains("start");
                        let possible_cuts = possible_partitions_between_sets(
                            new_state_has_end,
                            remaining_activities_has_start,
                            visited.contains(&remaining_activities),
                        );
                        valid.push((new_state.clone(), remaining_activities, possible_cuts));
                    } else {
                        let part_a = &new_state | &disconnected_nodes_to_end_in_graph_part_b;
                        if !visited.contains(&part_a) {
                            visited.insert(part_a.clone());
                            let part_b =
                                &remaining_activities - &disconnected_nodes_to_end_in_graph_part_b;
                            let graph_part_a = graph.subgraph_by_weights(&part_a);
                            let reachable_from_start =
                                if let Some(start) = graph_part_a.get_node_index("start") {
                                    graph_part_a.all_nodes_are_reachable_from(start)
                                } else {
                                    graph_part_a.node_count() == 0
                                };

                            if !(&part_b - &BTreeSet::from(["start", "end"])).is_empty()
                                && reachable_from_start
                            {
                                let part_a_has_end = part_a.contains("end");
                                let part_b_has_start = part_b.contains("start");
                                let possible_cuts = possible_partitions_between_sets(
                                    part_a_has_end,
                                    part_b_has_start,
                                    visited.contains(&part_b),
                                );
                                valid.push((part_a.clone(), part_b, possible_cuts));
                            }
                            queue.push((
                                part_a.clone(),
                                &(&new_adj
                                    | &graph.all_neighbors_weights(
                                        &disconnected_nodes_to_end_in_graph_part_b,
                                    ))
                                    - &part_a,
                            ));
                        }
                    }
                } else if !remaining_activities.contains("start")
                    && graph_remaining_nodes.is_weakly_connected()
                {
                    valid.push((
                        new_state.clone(),
                        remaining_activities,
                        HashSet::from(["loop"]),
                    ));
                }
            }
        }
    }

    valid
}

fn possible_partitions_between_sets(
    set_a_has_end: bool,
    set_b_has_start: bool,
    set_b_in_visited: bool,
) -> HashSet<&'static str> {
    if set_a_has_end && set_b_has_start && !set_b_in_visited {
        HashSet::from(["seq", "exc", "par", "loop"])
    } else if set_a_has_end {
        HashSet::from(["loop", "seq"])
    } else {
        HashSet::from(["seq"])
    }
}

pub fn add_start_and_end(
    set: &mut BTreeSet<&str>,
    start_activities: &BTreeSet<&str>,
    end_activities: &BTreeSet<&str>,
) {
    let has_start_activities = set.intersection(start_activities).next().is_some();
    let has_end_activities = set.intersection(end_activities).next().is_some();

    if has_start_activities {
        set.insert("start");
    }

    if has_end_activities {
        set.insert("end");
    }
}

#[pyfunction]
pub fn filter_dfg<'a>(
    dfg: HashMap<(&'a str, &'a str), usize>,
    dfg_minus: HashMap<(&str, &str), usize>,
    theta: f64,
) -> HashMap<(&'a str, &'a str), usize> {
    assert!(
        theta >= 0.0 && theta <= 1.0,
        "Theta must be between 0 and 1"
    );

    if dfg_minus.is_empty() {
        return dfg.clone();
    }

    // Identify and keep the max outgoing edge for each node
    let mut max_outgoing_edges = HashMap::new();
    for (&(source, target), &weight) in &dfg {
        let (max_target, max_outgoing_weight) =
            max_outgoing_edges.entry(source).or_insert((target, weight));
        if weight > *max_outgoing_weight {
            *max_target = target;
            *max_outgoing_weight = weight;
        }
    }

    let max_outgoing_edges = max_outgoing_edges
        .into_iter()
        .map(|(source, (target, weight))| ((source, target), weight))
        .collect::<HashMap<_, _>>();

    // Prepare the list of edges that can be potentially removed
    let mut removable_edges = vec![];
    let mut total_volume = 0;
    for (&edge, &weight) in &dfg {
        let (source, target) = edge;
        if max_outgoing_edges.contains_key(&edge) || source == "start" || target == "end" {
            continue;
        }

        let volume = weight;
        let value = dfg_minus.get(&(source, target)).copied().unwrap_or(0);
        let remove_edge = RemoveEdge::new(source, target, volume, value);
        removable_edges.push(remove_edge);
        total_volume += volume;
    }

    // Edge case: If no edges can be removed
    if removable_edges.is_empty() || total_volume == 0 {
        return dfg.clone();
    }

    // Calculate the total capacity
    let capacity = (total_volume as f64 * theta) as usize;

    // Initialize DP table
    let n = removable_edges.len();
    let mut dp = vec![vec![0; capacity + 1]; n + 1];
    // Build the DP table
    for i in 1..=n {
        let item = &removable_edges[i - 1];
        let v = item.volume;
        let w = item.value;
        for c in 1..=capacity {
            if item.volume > c {
                dp[i][c] = dp[i - 1][c];
            } else {
                dp[i][c] = dp[i - 1][c].max(dp[i - 1][c - v] + w);
            }
        }
    }

    // Find the selected edges
    let mut c = capacity;
    let mut edges_to_remove = HashSet::new();
    for i in (1..=n).rev() {
        if dp[i][c] != dp[i - 1][c] {
            let edge = &removable_edges[i - 1];
            edges_to_remove.insert(edge.edge());
            c -= edge.volume;
        }
    }

    // Remove selected edges from the desirable DFG
    let filtered_dfg = dfg
        .into_iter()
        .filter(|(edge, _)| !edges_to_remove.contains(edge))
        .collect();

    filtered_dfg
}

#[derive(Debug)]
struct RemoveEdge<'a> {
    source: &'a str,
    target: &'a str,
    volume: usize,
    value: usize,
}

impl<'a> RemoveEdge<'a> {
    pub fn new(source: &'a str, target: &'a str, volume: usize, value: usize) -> Self {
        Self {
            source,
            target,
            volume,
            value,
        }
    }

    pub fn edge(&self) -> (&'a str, &'a str) {
        (self.source, self.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_dfg_basic_functionality() {
        let desirable_dfg = HashMap::from([
            (("A", "B"), 10),
            (("A", "C"), 5),
            (("B", "C"), 7),
            (("C", "D"), 3),
            (("B", "D"), 4),
        ]);
        let undesirable_dfg = HashMap::from([
            (("A", "C"), 6),
            (("B", "C"), 2),
            (("C", "D"), 5),
            (("B", "D"), 8),
        ]);
        let theta = 0.3;
        let filtered_dfg = filter_dfg(desirable_dfg, undesirable_dfg, theta);
        let expected_dfg = HashMap::from([
            (("A", "B"), 10),
            (("A", "C"), 5),
            (("B", "C"), 7),
            (("C", "D"), 3),
        ]);

        assert_eq!(filtered_dfg, expected_dfg);
    }

    #[test]
    fn test_filter_dfg_no_edges_to_remove() {
        let desirable_dfg = HashMap::from([(("A", "B"), 10), (("A", "C"), 5)]);
        let undesirable_dfg = HashMap::new();
        let theta = 0.5;
        let filtered_dfg = filter_dfg(desirable_dfg.clone(), undesirable_dfg, theta);

        assert_eq!(filtered_dfg, desirable_dfg);
    }

    #[test]
    fn test_filter_dfg_all_edges_removed() {
        let desirable_dfg = HashMap::from([(("A", "B"), 2), (("A", "C"), 3), (("B", "C"), 1)]);
        let undesirable_dfg = HashMap::from([(("A", "B"), 5), (("A", "C"), 6), (("B", "C"), 4)]);
        let theta = 1.0;
        let filtered_dfg = filter_dfg(desirable_dfg, undesirable_dfg, theta);
        let expected_dfg = HashMap::from([(("A", "C"), 3), (("B", "C"), 1)]);

        assert_eq!(filtered_dfg, expected_dfg);
    }

    #[test]
    fn test_filter_dfg_different_theta_values() {
        // Desirable DFG
        let desirable_dfg = HashMap::from([
            (("A", "B"), 4),
            (("A", "C"), 6),
            (("A", "D"), 3),
            (("B", "D"), 5),
            (("C", "D"), 2),
        ]);

        // Undesirable DFG
        let undesirable_dfg = HashMap::from([
            (("A", "B"), 3),
            (("A", "C"), 1),
            (("A", "D"), 5),
            (("C", "D"), 4),
        ]);

        // Test with theta = 0.2
        let theta_low = 0.2;
        let filtered_dfg_low =
            filter_dfg(desirable_dfg.clone(), undesirable_dfg.clone(), theta_low);

        let expected_dfg_low = HashMap::from([
            (("A", "B"), 4),
            (("A", "C"), 6),
            (("B", "D"), 5),
            (("C", "D"), 2),
        ]);

        assert_eq!(filtered_dfg_low, expected_dfg_low);

        // Test with theta = 0.5
        let theta_high = 0.5;
        let filtered_dfg_high = filter_dfg(desirable_dfg.clone(), undesirable_dfg, theta_high);

        let expected_dfg_high = HashMap::from([(("A", "C"), 6), (("B", "D"), 5), (("C", "D"), 2)]);

        assert_eq!(filtered_dfg_high, expected_dfg_high);
    }
}
