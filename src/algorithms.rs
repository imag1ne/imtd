use std::{collections::{BTreeSet, HashSet}, vec};
use petgraph::{graph::{DiGraph, NodeIndex}, visit::{Dfs, Reversed}, Undirected};
use pyo3::pyfunction;
use crate::DirectlyFollowsGraph;

#[pyfunction]
pub fn find_possible_partitions(graph: &DirectlyFollowsGraph) -> Vec<(BTreeSet<String>, BTreeSet<String>, HashSet<&'static str>)> {
    // let graph = &graph.inner;

    let activities = graph.node_weights()
        .filter(|&w| w != "start" && w != "end")
        .cloned()
        .collect::<BTreeSet<_>>();

    let mut queue = vec![(BTreeSet::new(), BTreeSet::from(["start".to_string()]))];
    let mut visited = HashSet::new();
    let mut valid = vec![];

    while let Some((current_state, current_adj)) = queue.pop() {
        for x in &current_adj {
            let x_idx = graph.get_node_index(x).unwrap();
            let mut new_state = current_state.union(&BTreeSet::from([x.to_string()])).map(|n| n.to_string()).collect();
            graph.add_start_and_end(&mut new_state);

            if !visited.contains(&new_state) {
                visited.insert(new_state.clone());
                let new_adj = &(&current_adj | &graph.neighbors(x_idx).map(|i| graph[i].clone()).collect()) - &new_state;
                queue.push((new_state.clone(), new_adj.clone()));

                let mut remaining_activities = &activities - &new_state;
                if remaining_activities.is_empty() || remaining_activities.len() == activities.len() {
                    continue;
                }

                graph.add_start_and_end(&mut remaining_activities);
                let graph_remaining_nodes = graph.subgraph_by_weights(&remaining_activities);

                if let Some(end) = get_end(&graph_remaining_nodes) {
                    let mut disconnected_nodes_to_end_in_graph_part_b = (&graph_remaining_nodes.node_weights().collect()  - &ancestors(&graph_remaining_nodes, end)).into_iter().map(String::to_string).collect::<BTreeSet<_>>();
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
                            let reachable_from_start = if let Some(start) = get_start(&graph_part_a) {
                                all_nodes_are_reachable_from(&graph_part_a, start)
                            } else {
                                graph_part_a.node_count() == 0
                            };
                            
                            if !(&part_b - &BTreeSet::from(["start".to_string(), "end".to_string()])).is_empty() &&  reachable_from_start {
                                let part_a_has_end = part_a.contains("end");
                                let part_b_has_start = part_b.contains("start");
                                let possible_cuts = possible_partitions_between_sets(part_a_has_end, part_b_has_start, visited.contains(&part_b));
                                valid.push((part_a.clone(), part_b, possible_cuts));
                            }
                            queue.push((part_a.clone(), &(&new_adj | &graph.all_neighbors_weights(&disconnected_nodes_to_end_in_graph_part_b)) - &part_a));
                        }
                    }
                } else if !remaining_activities.contains("start") && is_weakly_connected(&graph_remaining_nodes) {
                    let new_state_has_end = new_state.contains("end");
                    let remaining_activities_has_start = remaining_activities.contains("start");
                    let possible_cuts = possible_partitions_between_sets(new_state_has_end, remaining_activities_has_start, visited.contains(&remaining_activities));
                    valid.push((new_state.clone(), remaining_activities, possible_cuts)); 
                }


            }
        }
    }

    valid
}

fn get_end(graph: &DiGraph<String, usize>) -> Option<NodeIndex> {
    graph.node_indices().find(|&i| graph[i] == "end")
}

fn get_start(graph: &DiGraph<String, usize>) -> Option<NodeIndex> {
    graph.node_indices().find(|&i| graph[i] == "start")
}

fn ancestors(graph: &DiGraph<String, usize>, node: NodeIndex) -> BTreeSet<&String> {
    let reversed_graph = Reversed(graph);
    let mut dfs = Dfs::new(&reversed_graph, node);
    
    let mut ancestors = BTreeSet::new();
    
    while let Some(ancestor) = dfs.next(&reversed_graph) {
        if ancestor != node {
            ancestors.insert(&graph[ancestor]);
        }
    }

    ancestors
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

fn all_nodes_are_reachable_from(graph: &DiGraph<String, usize>, start: NodeIndex) -> bool {
    let mut dfs = Dfs::new(graph, start);
    let mut visited = HashSet::new();

    while let Some(node) = dfs.next(graph) {
        visited.insert(node);
    }

    visited.len() == graph.node_count()
}

fn is_weakly_connected(graph: &DiGraph<String, usize>) -> bool {
    let undirected_graph = graph.clone().into_edge_type::<Undirected>();
    let mut dfs = Dfs::new(&undirected_graph, graph.node_indices().next().unwrap());
    let mut visited = HashSet::new();

    while let Some(node) = dfs.next(&undirected_graph) {
        visited.insert(node);
    }

    visited.len() == graph.node_count()
}
