use crate::graph::py_graph::PyGraph;
use petgraph::{graph::EdgeReference, visit::EdgeRef, Direction};
use std::collections::{HashMap, HashSet};

pub(crate) type ActivitySetsParts<'a> = (
    HashSet<&'a str>,
    HashSet<&'a str>,
    HashSet<&'a str>,
    HashSet<&'a str>,
    HashSet<&'a str>,
);

pub(crate) fn get_activity_sets<'a>(
    dfg: &HashMap<(&'a str, &'a str), f64>,
    activitiy_set_1: &HashSet<&str>,
    activitiy_set_2: &HashSet<&str>,
) -> ActivitySetsParts<'a> {
    let start_activities_in_set_1 =
        get_start_activities_from_dfg_with_artificial_start(dfg, activitiy_set_1);
    let end_activities_in_set_1 =
        get_end_activities_from_dfg_with_artificial_end(dfg, activitiy_set_1);
    let start_activities_in_set_2 =
        get_start_activities_from_dfg_with_artificial_start(dfg, activitiy_set_2);
    let input_activities_in_set_2 = get_input_activities_from_dfg(dfg, activitiy_set_2);
    let output_activities_in_set_2 = get_output_activities_from_dfg(dfg, activitiy_set_2);

    (
        start_activities_in_set_1,
        end_activities_in_set_1,
        start_activities_in_set_2,
        input_activities_in_set_2,
        output_activities_in_set_2,
    )
}

fn get_start_activities_from_dfg_with_artificial_start<'a>(
    dfg: &HashMap<(&'a str, &'a str), f64>,
    activities: &HashSet<&str>,
) -> HashSet<&'a str> {
    dfg.keys()
        .filter(|(s, t)| *s == "start" && activities.contains(t))
        .map(|(_, t)| *t)
        .collect()
}

fn get_end_activities_from_dfg_with_artificial_end<'a>(
    dfg: &HashMap<(&'a str, &'a str), f64>,
    activities: &HashSet<&str>,
) -> HashSet<&'a str> {
    dfg.keys()
        .filter(|(s, t)| activities.contains(s) && *t == "end")
        .map(|(s, _)| *s)
        .collect()
}

fn get_input_activities_from_dfg<'a>(
    dfg: &HashMap<(&'a str, &'a str), f64>,
    activities: &HashSet<&str>,
) -> HashSet<&'a str> {
    dfg.keys()
        .filter(|(s, t)| !activities.contains(s) && activities.contains(t))
        .map(|(_, t)| *t)
        .collect()
}

fn get_output_activities_from_dfg<'a>(
    dfg: &HashMap<(&'a str, &'a str), f64>,
    activities: &HashSet<&str>,
) -> HashSet<&'a str> {
    dfg.keys()
        .filter(|(s, t)| activities.contains(s) && !activities.contains(t))
        .map(|(s, _)| *s)
        .collect()
}

pub(crate) fn edge_boundary_directed<'a>(
    graph: &'a PyGraph,
    node_set_1: &'a HashSet<&'a str>,
    node_set_2: &'a HashSet<&'a str>,
) -> impl Iterator<Item = EdgeReference<'a, f64>> {
    let di_graph = &graph.graph;

    di_graph.edge_references().filter(|edge| {
        let source = *di_graph.node_weight(edge.source()).unwrap();
        let target = *di_graph.node_weight(edge.target()).unwrap();

        node_set_1.contains(source) && node_set_2.contains(target)
    })
}

pub(crate) fn edge_boundary_directed_num(
    graph: &PyGraph,
    node_set_1: &HashSet<&str>,
    node_set_2: &HashSet<&str>,
) -> f64 {
    edge_boundary_directed(graph, node_set_1, node_set_2)
        .map(|edge| edge.weight())
        .sum()
}

pub(crate) fn average_edge_boundary_directed_num(
    graph: &PyGraph,
    node_set_1: &HashSet<&str>,
    node_set_2: &HashSet<&str>,
) -> f64 {
    let (count, total) = edge_boundary_directed(graph, node_set_1, node_set_2)
        .fold((0.0, 0.0), |(count, weights), edge| {
            (count + 1.0, weights + edge.weight())
        });

    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

pub(crate) fn node_to_nodes_num(
    graph: &PyGraph,
    node_weight: &str,
    node_set: &HashSet<&str>,
) -> f64 {
    let di_graph = &graph.graph;

    if !graph.contains_node_weight(node_weight) {
        return 0.0;
    }

    di_graph
        .edges(graph[node_weight])
        .filter(|edge| {
            let target = *di_graph.node_weight(edge.target()).unwrap();
            node_set.contains(target)
        })
        .map(|edge| edge.weight())
        .sum()
}

pub(crate) fn nodes_to_node_num(
    graph: &PyGraph,
    node_set: &HashSet<&str>,
    node_weight: &str,
) -> f64 {
    let di_graph = &graph.graph;

    if !graph.contains_node_weight(node_weight) {
        return 0.0;
    }

    di_graph
        .edges_directed(graph[node_weight], Direction::Incoming)
        .filter(|edge| {
            let source = *di_graph.node_weight(edge.source()).unwrap();
            node_set.contains(source)
        })
        .map(|edge| edge.weight())
        .sum()
}

pub(crate) fn node_to_node_num(graph: &PyGraph, node_weight_1: &str, node_weight_2: &str) -> f64 {
    let di_graph = &graph.graph;

    if !graph.contains_node_weight(node_weight_1) || !graph.contains_node_weight(node_weight_2) {
        return 0.0;
    }

    di_graph
        .edges_connecting(graph[node_weight_1], graph[node_weight_2])
        .map(|edge| edge.weight())
        .sum()
}
