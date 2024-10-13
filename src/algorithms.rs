use crate::graph::py_graph::PyGraph;
use petgraph::Direction;
use pyo3::pyfunction;
use rayon::prelude::*;
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
        (0.0..=1.0).contains(&theta),
        "Theta must be between 0 and 1"
    );

    if dfg_minus.is_empty() || theta == 0.0 {
        return dfg;
    }

    // Identify and keep the max outgoing edge for each node
    let mut max_outgoing_edge_weights = HashMap::new();
    for (&(source, _), &weight) in &dfg {
        let max_weight = max_outgoing_edge_weights.entry(source).or_insert(weight);
        if weight > *max_weight {
            *max_weight = weight;
        }
    }

    // Prepare the list of edges that can be potentially removed
    let intersection_dfg = dfg
        .iter()
        .filter(|(edge, _)| dfg_minus.contains_key(edge))
        .map(|(edge, weight)| (*edge, *weight))
        .collect::<HashMap<_, _>>();
    let mut removable_edges = vec![];
    let mut total_weight = 0;
    for (&edge, &weight) in &intersection_dfg {
        let (source, target) = edge;
        let max_weight = max_outgoing_edge_weights[source];
        if weight == max_weight || source == "start" || target == "end" {
            continue;
        }

        let value = dfg_minus[&(source, target)];
        let remove_edge = RemovableEdge::new(source, target, weight, value);
        removable_edges.push(remove_edge);
        total_weight += weight;
    }

    // Edge case: If no edges can be removed
    if removable_edges.is_empty() || total_weight == 0 {
        return dfg;
    }

    // Calculate the total capacity
    let capacity = (total_weight as f64 * theta) as usize;

    // Initialize DP table
    let n = removable_edges.len();
    let mut dp = vec![vec![0; capacity + 1]; n + 1];
    // Build the DP table
    for i in 1..=n {
        let item = &removable_edges[i - 1];
        let weight = item.weight;
        let value = item.value;

        let (dp_before_i, dp_from_i) = dp.split_at_mut(i);
        let dp_prev = &dp_before_i[i - 1];
        let dp_curr = &mut dp_from_i[0];

        dp_curr
            .par_iter_mut()
            .enumerate()
            .for_each(|(c, dp_curr_c)| {
                if weight > c {
                    *dp_curr_c = dp_prev[c];
                } else {
                    *dp_curr_c = dp_prev[c].max(dp_prev[c - weight] + value);
                }
            });
        // for c in 1..=capacity {
        //     if item.weight > c {
        //         dp[i][c] = dp[i - 1][c];
        //     } else {
        //         dp[i][c] = dp[i - 1][c].max(dp[i - 1][c - weight] + value);
        //     }
        // }
    }

    // Find the selected edges
    let mut c = capacity;
    let mut edges_to_remove = HashSet::new();
    for i in (1..=n).rev() {
        if dp[i][c] != dp[i - 1][c] {
            let edge = &removable_edges[i - 1];
            edges_to_remove.insert(edge.edge());
            c -= edge.weight;
        }
    }

    // Remove selected edges from the desirable DFG
    dfg.into_iter()
        .filter(|(edge, _)| !edges_to_remove.contains(edge))
        .collect()
}

#[derive(Debug)]
struct RemovableEdge<'a> {
    source: &'a str,
    target: &'a str,
    weight: usize,
    value: usize,
}

impl<'a> RemovableEdge<'a> {
    pub fn new(source: &'a str, target: &'a str, weight: usize, value: usize) -> Self {
        Self {
            source,
            target,
            weight,
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
        let theta = 0.5;
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
    fn test_filter_dfg_only_filter_the_edges_in_dfg_minus() {
        let desirable_dfg = HashMap::from([(("A", "B"), 10), (("A", "C"), 9), (("A", "D"), 1)]);
        let undesirable_dfg = HashMap::from([(("A", "E"), 5)]);
        let theta = 0.4;
        let filtered_dfg = filter_dfg(desirable_dfg, undesirable_dfg, theta);
        let expected_dfg = HashMap::from([(("A", "B"), 10), (("A", "C"), 9), (("A", "D"), 1)]);

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

        // Test with theta = 0.5
        let theta_low = 0.5;
        let filtered_dfg_low =
            filter_dfg(desirable_dfg.clone(), undesirable_dfg.clone(), theta_low);

        let expected_dfg_low = HashMap::from([
            (("A", "B"), 4),
            (("A", "C"), 6),
            (("B", "D"), 5),
            (("C", "D"), 2),
        ]);

        assert_eq!(filtered_dfg_low, expected_dfg_low);

        // Test with theta = 1.0
        let theta_high = 1.0;
        let filtered_dfg_high = filter_dfg(desirable_dfg.clone(), undesirable_dfg, theta_high);

        let expected_dfg_high = HashMap::from([(("A", "C"), 6), (("B", "D"), 5), (("C", "D"), 2)]);

        assert_eq!(filtered_dfg_high, expected_dfg_high);
    }
}
