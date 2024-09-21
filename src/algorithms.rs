use std::{collections::{BTreeSet, HashSet}, vec};
use petgraph::Direction;
use pyo3::pyfunction;
use crate::graph::py_graph::PyGraph;

#[pyfunction]
pub fn find_possible_partitions(graph: PyGraph<'_>) -> Vec<(BTreeSet<&str>, BTreeSet<&str>, HashSet<&str>)> {
    let activities = graph.node_weights()
        .filter(|&w| *w != "start" && *w != "end")
        .copied()
        .collect::<BTreeSet<_>>();
    let start_activities = graph.neighbors(graph["start"]).map(|i| graph.graph[i]).collect::<BTreeSet<_>>();
    let end_activities = graph.neighbors_directed(graph["end"], Direction::Incoming).map(|i| graph.graph[i]).collect::<BTreeSet<_>>();

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
                let new_adj = &(&current_adj | &graph.neighbors(x_idx).map(|i| graph.graph[i]).collect()) - &new_state;
                queue.push((new_state.clone(), new_adj.clone()));

                let mut remaining_activities = &activities - &new_state;
                if remaining_activities.is_empty() || remaining_activities.len() == activities.len() {
                    continue;
                }

                add_start_and_end(&mut remaining_activities, &start_activities, &end_activities);
                let graph_remaining_nodes = graph.subgraph_by_weights(&remaining_activities);

                if let Some(end) = graph_remaining_nodes.get_node_index("end") {
                    let mut disconnected_nodes_to_end_in_graph_part_b = &graph_remaining_nodes.node_weights().copied().collect()  - &graph_remaining_nodes.ancestors(end);
                    disconnected_nodes_to_end_in_graph_part_b.remove("end");

                    if disconnected_nodes_to_end_in_graph_part_b.is_empty() {
                        let new_state_has_end = new_state.contains("end");
                        let remaining_activities_has_start = remaining_activities.contains("start");
                        let possible_cuts = possible_partitions_between_sets(new_state_has_end, remaining_activities_has_start, visited.contains(&remaining_activities));
                        valid.push((new_state.clone(), remaining_activities, possible_cuts));
                    } else {
                        let part_a = &new_state | &disconnected_nodes_to_end_in_graph_part_b;
                        if !visited.contains(&part_a) {
                            visited.insert(part_a.clone());
                            let part_b = &remaining_activities - &disconnected_nodes_to_end_in_graph_part_b;
                            let graph_part_a = graph.subgraph_by_weights(&part_a);
                            let reachable_from_start = if let Some(start) = graph_part_a.get_node_index("start") {
                                graph_part_a.all_nodes_are_reachable_from(start)
                            } else {
                                graph_part_a.node_count() == 0
                            };
                            
                            if !(&part_b - &BTreeSet::from(["start", "end"])).is_empty() &&  reachable_from_start {
                                let part_a_has_end = part_a.contains("end");
                                let part_b_has_start = part_b.contains("start");
                                let possible_cuts = possible_partitions_between_sets(part_a_has_end, part_b_has_start, visited.contains(&part_b));
                                valid.push((part_a.clone(), part_b, possible_cuts));
                            }
                            queue.push((part_a.clone(), &(&new_adj | &graph.all_neighbors_weights(&disconnected_nodes_to_end_in_graph_part_b)) - &part_a));
                        }
                    }
                } else if !remaining_activities.contains("start") && graph_remaining_nodes.is_weakly_connected() {
                    valid.push((new_state.clone(), remaining_activities, HashSet::from(["loop"])));
                }
            }
        }
    }

    valid
}

fn possible_partitions_between_sets(set_a_has_end: bool,set_b_has_start: bool, set_b_in_visited: bool) -> HashSet<&'static str> {
    if set_a_has_end && set_b_has_start && !set_b_in_visited {
        HashSet::from(["seq", "exc", "par", "loop"])
    } else if set_a_has_end {
        HashSet::from(["loop", "seq"])
    } else {
        HashSet::from(["seq"])
    }
}

pub fn add_start_and_end(set: &mut BTreeSet<&str>, start_activities: &BTreeSet<&str>, end_activities: &BTreeSet<&str>) {
    let has_start_activities = set.intersection(start_activities).next().is_some();
    let has_end_activities = set.intersection(end_activities).next().is_some();

    if has_start_activities {
        set.insert("start");
    }

    if has_end_activities {
        set.insert("end");
    }
}
