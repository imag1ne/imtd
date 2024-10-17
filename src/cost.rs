use std::collections::{HashMap, HashSet};

use petgraph::graph::EdgeReference;
use petgraph::{visit::EdgeRef, Direction};
use pyo3::pyfunction;
use rayon::prelude::*;

use crate::PyGraph;

pub type CutEvaluations<'a> = Vec<(
    (HashSet<&'a str>, HashSet<&'a str>),
    String,
    f64,
    f64,
    f64,
    f64,
)>;

#[pyfunction]
pub fn evaluate_cuts<'a>(
    cuts: Vec<(HashSet<&'a str>, HashSet<&'a str>, HashSet<&str>)>,
    dfg: HashMap<(&str, &str), f64>,
    dfg_minus: HashMap<(&str, &str), f64>,
    nx_graph: PyGraph,
    nx_graph_minus: PyGraph,
    max_flow_graph: HashMap<(&str, &str), f64>,
    max_flow_graph_minus: HashMap<(&str, &str), f64>,
    activities_minus: HashSet<&str>,
    log_variants: HashMap<Vec<&str>, f64>,
    log_length: f64,
    log_minus_length: f64,
    case_id_trace_index_map: HashMap<&str, usize>,
    case_id_trace_index_map_m: HashMap<&str, usize>,
    original_edge_case_id_map: HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map: HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map_m: HashMap<(&str, &str), HashSet<&str>>,
    similarity_matrix: Vec<Vec<f64>>,
    feat_scores: HashMap<(&str, &str), f64>,
    feat_scores_toggle: HashMap<(&str, &str), f64>,
    sup: f64,
    ratio: f64,
    size_par: f64,
) -> CutEvaluations<'a> {
    cuts.par_iter()
        .flat_map(|(part_a, part_b, cut_types)| {
            let start_and_end_set = HashSet::from(["start", "end"]);
            let part_a = part_a - &start_and_end_set;
            let part_b = part_b - &start_and_end_set;
            let mut evaluations = vec![];

            let (start_part_a, end_part_a, start_part_b, input_part_b, output_part_b) =
                get_activity_sets(&dfg, &part_a, &part_b);
            let (
                start_part_a_minus,
                end_part_a_minus,
                start_part_b_minus,
                input_part_b_minus,
                output_part_b_minus,
            ) = get_activity_sets(&dfg_minus, &part_a, &part_b);
            let activities_minus_in_part_a = &part_a & &activities_minus;
            let activities_minus_in_part_b = &part_b & &activities_minus;
            let activities_minus_in_start_part_b_minus = &start_part_b_minus & &activities_minus;
            let activities_minus_in_end_part_a_minus = &end_part_a_minus & &activities_minus;

            let mut ratio = ratio;
            if activities_minus_in_part_a.is_empty() || activities_minus_in_part_b.is_empty() {
                ratio = 0.0;
            }

            if cut_types.contains("seq") {
                let fit_seq = fit_seq(&log_variants, &part_a, &part_b);
                if fit_seq > 0.0 {
                    let cost_seq_plus = cost_seq(
                        &nx_graph,
                        &part_a,
                        &part_b,
                        &start_part_b,
                        &end_part_a,
                        sup,
                        &max_flow_graph,
                        &feat_scores,
                    );
                    let cost_seq_minus = cost_seq_minus(
                        &nx_graph_minus,
                        &activities_minus_in_part_a,
                        &activities_minus_in_part_b,
                        &activities_minus_in_start_part_b_minus,
                        &activities_minus_in_end_part_a_minus,
                        sup,
                        &max_flow_graph_minus,
                        &feat_scores_toggle,
                        &case_id_trace_index_map,
                        &case_id_trace_index_map_m,
                        &original_edge_case_id_map,
                        &edge_case_id_map,
                        &edge_case_id_map_m,
                        &similarity_matrix,
                    );

                    let seq_evaluation = (
                        (part_a.clone(), part_b.clone()),
                        "seq".to_string(),
                        cost_seq_plus,
                        cost_seq_minus,
                        calculate_cost(cost_seq_plus, cost_seq_minus, ratio, size_par),
                        fit_seq,
                    );
                    evaluations.push(seq_evaluation);
                }
            }

            if cut_types.contains("exc") {
                let fit_exc = fit_exc(&log_variants, &part_a, &part_b);

                if fit_exc > 0.0 {
                    let cost_exc_plus = cost_exc(&nx_graph, &part_a, &part_b, sup);
                    let cost_exc_minus = cost_exc_minus(
                        &nx_graph_minus,
                        &activities_minus_in_part_a,
                        &activities_minus_in_part_b,
                        &case_id_trace_index_map,
                        &case_id_trace_index_map_m,
                        &original_edge_case_id_map,
                        &edge_case_id_map,
                        &edge_case_id_map_m,
                        &similarity_matrix,
                    );

                    let exc_evaluation = (
                        (part_a.clone(), part_b.clone()),
                        "exc".to_string(),
                        cost_exc_plus,
                        cost_exc_minus,
                        calculate_cost(cost_exc_plus, cost_exc_minus, ratio, size_par),
                        fit_exc,
                    );
                    evaluations.push(exc_evaluation);
                }
            }

            let start_to_end_num = node_to_node_num(&nx_graph, "start", "end");
            if start_to_end_num > 0.0 {
                let missing_exc_tau_plus = f64::max(0.0, sup * log_length - start_to_end_num);
                let missing_exc_tau_minus = f64::max(
                    0.0,
                    sup * log_minus_length - node_to_node_num(&nx_graph_minus, "start", "end"),
                );
                let part_a_union_part_b = &part_a | &part_b;
                let xor_tau_evaluation = (
                    (part_a_union_part_b, HashSet::new()),
                    "exc2".to_string(),
                    missing_exc_tau_plus,
                    missing_exc_tau_minus,
                    calculate_cost(missing_exc_tau_plus, missing_exc_tau_minus, ratio, size_par),
                    1.0,
                );
                evaluations.push(xor_tau_evaluation);
            }

            if cut_types.contains("par") {
                let cost_par_plus = cost_par(
                    &nx_graph,
                    &activities_minus_in_part_a,
                    &activities_minus_in_part_b,
                    sup,
                );
                let cost_par_minus = cost_par(
                    &nx_graph_minus,
                    &activities_minus_in_part_a,
                    &activities_minus_in_part_b,
                    sup,
                );

                let par_evaluation = (
                    (part_a.clone(), part_b.clone()),
                    "par".to_string(),
                    cost_par_plus,
                    cost_par_minus,
                    calculate_cost(cost_par_plus, cost_par_minus, ratio, size_par),
                    1.0,
                );
                evaluations.push(par_evaluation);
            }

            if cut_types.contains("loop") {
                let fit_loop =
                    fit_loop(&log_variants, &part_a, &part_b, &end_part_a, &start_part_a);
                if fit_loop > 0.0 {
                    if let Some(cost_loop_plus) = cost_loop(
                        &nx_graph,
                        &part_a,
                        &part_b,
                        &start_part_a,
                        &end_part_a,
                        &input_part_b,
                        &output_part_b,
                        sup,
                    ) {
                        let cost_loop_minus = cost_loop_minus(
                            &nx_graph_minus,
                            &part_a,
                            &part_b,
                            &start_part_a_minus,
                            &end_part_a_minus,
                            &input_part_b_minus,
                            &output_part_b_minus,
                            sup,
                            &case_id_trace_index_map,
                            &case_id_trace_index_map_m,
                            &original_edge_case_id_map,
                            &edge_case_id_map,
                            &edge_case_id_map_m,
                            &similarity_matrix,
                        )
                        .unwrap_or(0.0);

                        let loop_evaluation = (
                            (part_a.clone(), part_b.clone()),
                            "loop".to_string(),
                            cost_loop_plus,
                            cost_loop_minus,
                            calculate_cost(cost_loop_plus, cost_loop_minus, ratio, size_par),
                            fit_loop,
                        );
                        evaluations.push(loop_evaluation);
                    }
                }
            }

            evaluations
        })
        .collect::<Vec<_>>()
}

fn get_activity_sets<'a>(
    dfg: &HashMap<(&'a str, &'a str), f64>,
    activitiy_set_1: &HashSet<&str>,
    activitiy_set_2: &HashSet<&str>,
) -> (
    HashSet<&'a str>,
    HashSet<&'a str>,
    HashSet<&'a str>,
    HashSet<&'a str>,
    HashSet<&'a str>,
) {
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

fn fit_seq(
    log_variants: &HashMap<Vec<&str>, f64>,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
) -> f64 {
    let mut count = 0.0;

    for (trace, trace_num) in log_variants {
        for pair in trace.windows(2) {
            if part_b.contains(&pair[0]) && part_a.contains(&pair[1]) {
                count += trace_num;
                break;
            }
        }
    }

    let fit = 1.0 - (count / log_variants.values().sum::<f64>());
    fit
}

fn fit_exc(
    log_variants: &HashMap<Vec<&str>, f64>,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
) -> f64 {
    let mut count = 0.0;

    for (trace, trace_num) in log_variants {
        let activities = trace.iter().copied().collect::<HashSet<_>>();
        if activities.is_subset(part_a) || activities.is_subset(part_b) {
            count += trace_num;
        }
    }

    let fit = count / log_variants.values().sum::<f64>();
    fit
}

fn fit_loop(
    log_variants: &HashMap<Vec<&str>, f64>,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    end_part_a: &HashSet<&str>,
    start_part_a: &HashSet<&str>,
) -> f64 {
    let mut count = 0.0;

    for (trace, &trace_num) in log_variants {
        if trace.is_empty() {
            continue;
        }

        if part_b.contains(trace.first().unwrap()) || part_b.contains(trace.last().unwrap()) {
            count += trace_num;
            continue;
        }

        for pair in trace.windows(2) {
            let source = pair[0];
            let target = pair[1];
            if part_a.contains(source) && part_b.contains(target) {
                if !end_part_a.contains(source) {
                    count += trace_num;
                }
                break;
            }

            if part_a.contains(target) && part_b.contains(source) {
                if !start_part_a.contains(target) {
                    count += trace_num;
                }
                break;
            }
        }
    }

    let fit = 1.0 - (count / log_variants.values().sum::<f64>());
    fit
}

fn edge_boundary_directed<'a>(
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

fn edge_boundary_directed_num(
    graph: &PyGraph,
    node_set_1: &HashSet<&str>,
    node_set_2: &HashSet<&str>,
) -> f64 {
    edge_boundary_directed(graph, node_set_1, node_set_2)
        .map(|edge| edge.weight())
        .sum()
}

fn node_to_nodes_num(graph: &PyGraph, node_weight: &str, node_set: &HashSet<&str>) -> f64 {
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

fn nodes_to_node_num(graph: &PyGraph, node_set: &HashSet<&str>, node_weight: &str) -> f64 {
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

fn node_to_node_num(graph: &PyGraph, node_weight_1: &str, node_weight_2: &str) -> f64 {
    let di_graph = &graph.graph;

    if !graph.contains_node_weight(node_weight_1) || !graph.contains_node_weight(node_weight_2) {
        return 0.0;
    }

    di_graph
        .edges_connecting(graph[node_weight_1], graph[node_weight_2])
        .map(|edge| edge.weight())
        .sum()
}

fn deviating_edges_cost(
    graph: &PyGraph,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    case_id_trace_index_map: &HashMap<&str, usize>,
    case_id_trace_index_map_m: &HashMap<&str, usize>,
    original_edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    _edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map_m: &HashMap<(&str, &str), HashSet<&str>>,
    similarity_matrix: &[Vec<f64>],
) -> f64 {
    let mut cost = 0.0;

    for edge in edge_boundary_directed(graph, part_b, part_a) {
        let deviating_edge = (graph.graph[edge.source()], graph.graph[edge.target()]);
        // traces that contain the deviating edge in the original positive log
        let traces_p_indices = original_edge_case_id_map
            .get(&deviating_edge)
            .unwrap_or(&HashSet::new())
            .iter()
            .map(|case_id| case_id_trace_index_map[case_id])
            .collect::<HashSet<_>>();
        // traces that contain the deviating edge in the negative log
        let traces_m_indices = edge_case_id_map_m[&deviating_edge]
            .iter()
            .map(|case_id| case_id_trace_index_map_m[case_id]);
        for trace_m_index in traces_m_indices {
            let similarity = similarity_matrix
                .iter()
                .enumerate()
                .filter(|(i, _)| !traces_p_indices.contains(i))
                .map(|(_, v)| v[trace_m_index])
                .reduce(f64::max);

            if let Some(similarity) = similarity {
                cost += similarity;
            }
        }
    }

    cost
}

fn cost_seq(
    graph: &PyGraph,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    start_set: &HashSet<&str>,
    end_set: &HashSet<&str>,
    sup: f64,
    flow: &HashMap<(&str, &str), f64>,
    scores: &HashMap<(&str, &str), f64>,
) -> f64 {
    // deviating edges
    let cost_1 = edge_boundary_directed_num(graph, part_b, part_a);

    let mut total_flow = 0.0;
    let mut flow_count = 0;
    for &node_a in part_a {
        for &node_b in part_b {
            if let Some(flow_value) = flow.get(&(node_a, node_b)) {
                total_flow += flow_value;
                flow_count += 1;
            }
        }
    }
    let avg_flow = total_flow / flow_count as f64;

    let mut cost_2 = 0.0;
    let part_a_out_degree = graph.nodes_out_degree(part_a);
    let part_b_out_degree = graph.nodes_out_degree(part_b);
    let part_a_and_part_b_out_degree = part_a_out_degree * part_b_out_degree;
    for &node_a in part_a {
        for &node_b in part_b {
            let expected = sup * avg_flow * graph.out_degree(node_a) * graph.out_degree(node_b)
                / part_a_and_part_b_out_degree;
            let cost = f64::max(0.0, expected - flow[&(node_a, node_b)]);
            cost_2 += cost;
        }
    }

    // let mut cost_3 = 0.0;
    // let part_a_with_start = part_a | &HashSet::from(["start"]);
    // let part_b_with_end = part_b | &HashSet::from(["end"]);
    // let part_a_with_start_to_part_b_with_end_num =
    //     edge_boundary_directed_num(graph, &part_a_with_start, &part_b_with_end);
    // for &end_node in end_set {
    //     for &start_node in start_set {
    //         let num_end_node_to_part_b_with_end =
    //             node_to_nodes_num(graph, end_node, &part_b_with_end);
    //         let num_part_a_with_start_to_start_node =
    //             nodes_to_node_num(graph, &part_a_with_start, start_node);
    //         let cost = f64::max(
    //             0.0,
    //             scores.get(&(end_node, start_node)).unwrap_or(&1.0)
    //                 * num_end_node_to_part_b_with_end
    //                 * sup
    //                 * num_part_a_with_start_to_start_node
    //                 / part_a_with_start_to_part_b_with_end_num
    //                 - node_to_node_num(graph, end_node, start_node),
    //         );
    //         cost_3 += cost;
    //     }
    // }

    cost_1 + cost_2
}

fn cost_seq_minus(
    graph: &PyGraph,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    start_set: &HashSet<&str>,
    end_set: &HashSet<&str>,
    sup: f64,
    flow: &HashMap<(&str, &str), f64>,
    scores: &HashMap<(&str, &str), f64>,
    case_id_trace_index_map: &HashMap<&str, usize>,
    case_id_trace_index_map_m: &HashMap<&str, usize>,
    original_edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map_m: &HashMap<(&str, &str), HashSet<&str>>,
    similarity_matrix: &[Vec<f64>],
) -> f64 {
    // deviating edges
    let cost_1 = deviating_edges_cost(
        graph,
        part_a,
        part_b,
        case_id_trace_index_map,
        case_id_trace_index_map_m,
        original_edge_case_id_map,
        edge_case_id_map,
        edge_case_id_map_m,
        similarity_matrix,
    );

    let mut cost_2 = 0.0;
    let part_a_out_degree = graph.nodes_out_degree(part_a);
    let part_b_out_degree = graph.nodes_out_degree(part_b);
    let part_a_and_part_b_out_degree = part_a_out_degree + part_b_out_degree;
    for &node_a in part_a {
        for &node_b in part_b {
            let cost = f64::max(
                0.0,
                scores.get(&(node_a, node_b)).unwrap_or(&1.0)
                    * graph.out_degree(node_a)
                    * sup
                    * (graph.out_degree(node_b) / part_a_and_part_b_out_degree)
                    - flow[&(node_a, node_b)],
            );
            cost_2 += cost;
        }
    }

    let mut cost_3 = 0.0;
    let part_a_with_start = part_a | &HashSet::from(["start"]);
    let part_b_with_end = part_b | &HashSet::from(["end"]);
    let part_a_with_start_to_part_b_with_end_num =
        edge_boundary_directed_num(graph, &part_a_with_start, &part_b_with_end);
    for &end_node in end_set {
        for &start_node in start_set {
            let num_end_node_to_part_b_with_end =
                node_to_nodes_num(graph, end_node, &part_b_with_end);
            let num_part_a_with_start_to_start_node =
                nodes_to_node_num(graph, &part_a_with_start, start_node);
            let cost = f64::max(
                0.0,
                scores.get(&(end_node, start_node)).unwrap_or(&1.0)
                    * num_end_node_to_part_b_with_end
                    * sup
                    * num_part_a_with_start_to_start_node
                    / part_a_with_start_to_part_b_with_end_num
                    - node_to_node_num(graph, end_node, start_node),
            );
            cost_3 += cost;
        }
    }

    cost_1 + cost_2 + cost_3
}

fn cost_exc(graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>, sup: f64) -> f64 {
    let cost_1 = edge_boundary_directed_num(graph, part_a, part_b);
    let cost_2 = edge_boundary_directed_num(graph, part_b, part_a);
    cost_1 + cost_2
}

fn cost_exc_minus(
    graph: &PyGraph,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    case_id_trace_index_map: &HashMap<&str, usize>,
    case_id_trace_index_map_m: &HashMap<&str, usize>,
    original_edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map_m: &HashMap<(&str, &str), HashSet<&str>>,
    similarity_matrix: &[Vec<f64>],
) -> f64 {
    let cost_1 = deviating_edges_cost(
        graph,
        part_a,
        part_b,
        case_id_trace_index_map,
        case_id_trace_index_map_m,
        original_edge_case_id_map,
        edge_case_id_map,
        edge_case_id_map_m,
        similarity_matrix,
    );

    let cost_2 = deviating_edges_cost(
        graph,
        part_b,
        part_a,
        case_id_trace_index_map,
        case_id_trace_index_map_m,
        original_edge_case_id_map,
        edge_case_id_map,
        edge_case_id_map_m,
        similarity_matrix,
    );

    cost_1 + cost_2
}

fn cost_par(graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>, sup: f64) -> f64 {
    let mut cost_1 = 0.0;
    let mut cost_2 = 0.0;

    let (edges_count_a_to_b, total_weight_a_to_b) = edge_boundary_directed(graph, part_a, part_b)
        .fold((0.0, 0.0), |(count, weights), edge| {
            (count + 1.0, weights + edge.weight())
        });
    let (edges_count_b_to_a, total_weight_b_to_a) = edge_boundary_directed(graph, part_b, part_a)
        .fold((0.0, 0.0), |(count, weights), edge| {
            (count + 1.0, weights + edge.weight())
        });
    let avg_weight_a_to_b = total_weight_a_to_b / edges_count_a_to_b;
    let avg_weight_b_to_a = total_weight_b_to_a / edges_count_b_to_a;

    let part_a_out_degree = graph.nodes_out_degree(part_a);
    let part_b_out_degree = graph.nodes_out_degree(part_b);
    for &node_a in part_a {
        for &node_b in part_b {
            let expected = sup * graph.out_degree(node_a) * graph.out_degree(node_b)
                / (part_a_out_degree * part_b_out_degree);
            cost_1 += f64::max(
                0.0,
                expected * avg_weight_a_to_b - node_to_node_num(graph, node_a, node_b),
            );
            cost_2 += f64::max(
                0.0,
                expected * avg_weight_b_to_a - node_to_node_num(graph, node_b, node_a),
            );
        }
    }

    cost_1 + cost_2
}

fn cost_loop(
    graph: &PyGraph,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    start_part_a: &HashSet<&str>,
    end_part_a: &HashSet<&str>,
    input_part_b: &HashSet<&str>,
    output_part_b: &HashSet<&str>,
    sup: f64,
) -> Option<f64> {
    if edge_boundary_directed_num(graph, part_b, start_part_a) != 0.0 {
        if edge_boundary_directed_num(graph, end_part_a, part_b) == 0.0 {
            return None;
        }
    } else {
        return None;
    }

    let output_part_b_to_start_part_a_num =
        edge_boundary_directed_num(graph, output_part_b, start_part_a);
    let end_part_a_to_input_part_b_num =
        edge_boundary_directed_num(graph, end_part_a, input_part_b);
    let m_p = f64::max(
        output_part_b_to_start_part_a_num,
        end_part_a_to_input_part_b_num,
    );

    let (edges_count, total_weight) = edge_boundary_directed(graph, output_part_b, start_part_a)
        .fold((0.0, 0.0), |(count, weights), edge| {
            (count + 1.0, weights + edge.weight())
        });
    let avg_weight_output_part_b_to_start_part_a = total_weight / edges_count;
    let (edges_count, total_weight) = edge_boundary_directed(graph, end_part_a, input_part_b)
        .fold((0.0, 0.0), |(count, weights), edge| {
            (count + 1.0, weights + edge.weight())
        });
    let avg_weight_end_part_a_to_input_part_b = total_weight / edges_count;
    let max_avg_weight = f64::max(
        avg_weight_output_part_b_to_start_part_a,
        avg_weight_end_part_a_to_input_part_b,
    );

    let cost_1 =
        node_to_nodes_num(graph, "start", part_b) + nodes_to_node_num(graph, part_b, "end");
    let cost_2 = edge_boundary_directed_num(graph, &(part_a - end_part_a), part_b);
    let cost_3 = edge_boundary_directed_num(graph, part_b, &(part_a - start_part_a));

    let mut cost_4 = 0.0;
    let start_to_start_part_a_num = node_to_nodes_num(graph, "start", start_part_a);
    let output_part_b_to_start_part_a_num =
        edge_boundary_directed_num(graph, output_part_b, start_part_a);
    let start_part_a_out_degree = graph.nodes_out_degree(start_part_a);
    let output_part_b_out_degree = graph.nodes_out_degree(output_part_b);
    let start_part_a_and_output_part_b_out_degree =
        start_part_a_out_degree * output_part_b_out_degree;
    if !output_part_b.is_empty() {
        for &node_a in start_part_a {
            for &node_b in output_part_b {
                let expected =
                    sup * max_avg_weight * graph.out_degree(node_a) * graph.out_degree(node_b)
                        / start_part_a_and_output_part_b_out_degree;
                cost_4 += f64::max(0.0, expected - node_to_node_num(graph, node_b, node_a));
            }
        }
    }

    let mut cost_5 = 0.0;
    let end_part_a_to_end_num = nodes_to_node_num(graph, end_part_a, "end");
    let end_part_a_to_input_part_b_num =
        edge_boundary_directed_num(graph, end_part_a, input_part_b);
    let end_part_a_out_degree = graph.nodes_out_degree(end_part_a);
    let input_part_b_out_degree = graph.nodes_out_degree(input_part_b);
    let end_part_a_and_input_part_b_out_degree = end_part_a_out_degree * input_part_b_out_degree;
    if !input_part_b.is_empty() {
        for &node_a in end_part_a {
            for &node_b in input_part_b {
                let expected =
                    sup * max_avg_weight * graph.out_degree(node_a) * graph.out_degree(node_b)
                        / end_part_a_and_input_part_b_out_degree;
                cost_5 += f64::max(0.0, expected - node_to_node_num(graph, node_a, node_b));
            }
        }
    }

    if sup * m_p == 0.0 {
        return None;
    }

    if (cost_4 + cost_5) / (2.0 * sup * m_p) > 0.3 {
        return None;
    }

    Some(cost_1 + cost_2 + cost_3 + cost_4 + cost_5)
}

fn cost_loop_minus(
    graph: &PyGraph,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    start_part_a: &HashSet<&str>,
    end_part_a: &HashSet<&str>,
    input_part_b: &HashSet<&str>,
    output_part_b: &HashSet<&str>,
    sup: f64,
    case_id_trace_index_map: &HashMap<&str, usize>,
    case_id_trace_index_map_m: &HashMap<&str, usize>,
    original_edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map: &HashMap<(&str, &str), HashSet<&str>>,
    edge_case_id_map_m: &HashMap<(&str, &str), HashSet<&str>>,
    similarity_matrix: &[Vec<f64>],
) -> Option<f64> {
    if edge_boundary_directed_num(graph, part_b, start_part_a) != 0.0 {
        if edge_boundary_directed_num(graph, end_part_a, part_b) == 0.0 {
            return None;
        }
    } else {
        return None;
    }

    let output_part_b_to_start_part_a_num =
        edge_boundary_directed_num(graph, output_part_b, start_part_a);
    let end_part_a_to_input_part_b_num =
        edge_boundary_directed_num(graph, end_part_a, input_part_b);
    let m_p = f64::max(
        output_part_b_to_start_part_a_num,
        end_part_a_to_input_part_b_num,
    );

    let cost_1 = deviating_edges_cost(
        graph,
        part_b,
        &HashSet::from(["start"]),
        case_id_trace_index_map,
        case_id_trace_index_map_m,
        original_edge_case_id_map,
        edge_case_id_map,
        edge_case_id_map_m,
        similarity_matrix,
    ) + deviating_edges_cost(
        graph,
        &HashSet::from(["end"]),
        part_b,
        case_id_trace_index_map,
        case_id_trace_index_map_m,
        original_edge_case_id_map,
        edge_case_id_map,
        edge_case_id_map_m,
        similarity_matrix,
    );
    let cost_2 = deviating_edges_cost(
        graph,
        part_b,
        &(part_a - end_part_a),
        case_id_trace_index_map,
        case_id_trace_index_map_m,
        original_edge_case_id_map,
        edge_case_id_map,
        edge_case_id_map_m,
        similarity_matrix,
    );
    let cost_3 = deviating_edges_cost(
        graph,
        &(part_a - start_part_a),
        part_b,
        case_id_trace_index_map,
        case_id_trace_index_map_m,
        original_edge_case_id_map,
        edge_case_id_map,
        edge_case_id_map_m,
        similarity_matrix,
    );

    let mut cost_4 = 0.0;
    let start_to_start_part_a_num = node_to_nodes_num(graph, "start", start_part_a);
    let output_part_b_to_start_part_a_num =
        edge_boundary_directed_num(graph, output_part_b, start_part_a);
    if !output_part_b.is_empty() {
        for &node_a in start_part_a {
            for &node_b in output_part_b {
                let c = m_p
                    * sup
                    * (node_to_node_num(graph, "start", node_a) / start_to_start_part_a_num)
                    * (node_to_nodes_num(graph, node_b, start_part_a)
                        / output_part_b_to_start_part_a_num);
                cost_4 += f64::max(0.0, c - node_to_node_num(graph, node_b, node_a));
            }
        }
    }

    let mut cost_5 = 0.0;
    let end_part_a_to_end_num = nodes_to_node_num(graph, end_part_a, "end");
    let end_part_a_to_input_part_b_num =
        edge_boundary_directed_num(graph, end_part_a, input_part_b);
    if !input_part_b.is_empty() {
        for &node_a in end_part_a {
            for &node_b in input_part_b {
                let c = m_p
                    * sup
                    * (node_to_node_num(graph, node_a, "end") / end_part_a_to_end_num)
                    * (nodes_to_node_num(graph, end_part_a, node_b)
                        / end_part_a_to_input_part_b_num);
                cost_5 += f64::max(0.0, c - node_to_node_num(graph, node_a, node_b));
            }
        }
    }

    if sup * m_p == 0.0 {
        return None;
    }

    // if (cost_4 + cost_5) / (2.0 * sup * m_p) > 0.3 {
    //     return None;
    // }

    Some(cost_1 + cost_2 + cost_3 + cost_4 + cost_5)
}

fn calculate_cost(cost_plus: f64, cost_minus: f64, ratio: f64, size_par: f64) -> f64 {
    cost_plus - ratio * size_par * cost_minus
}

#[pyfunction]
pub fn evaluate_cuts_for_imbi<'a>(
    cuts: Vec<(HashSet<&'a str>, HashSet<&'a str>, HashSet<&str>)>,
    dfg: HashMap<(&str, &str), f64>,
    dfg_minus: HashMap<(&str, &str), f64>,
    nx_graph: PyGraph,
    nx_graph_minus: PyGraph,
    max_flow_graph: HashMap<(&str, &str), f64>,
    max_flow_graph_minus: HashMap<(&str, &str), f64>,
    activities_minus: HashSet<&str>,
    log_variants: HashMap<Vec<&str>, f64>,
    log_length: f64,
    log_minus_length: f64,
    feat_scores: HashMap<(&str, &str), f64>,
    feat_scores_toggle: HashMap<(&str, &str), f64>,
    sup: f64,
    ratio: f64,
    size_par: f64,
) -> CutEvaluations<'a> {
    cuts.par_iter()
        .flat_map(|(part_a, part_b, cut_types)| {
            let start_and_end_set = HashSet::from(["start", "end"]);
            let part_a = part_a - &start_and_end_set;
            let part_b = part_b - &start_and_end_set;
            let mut evaluations = vec![];

            let (start_part_a, end_part_a, start_part_b, input_part_b, output_part_b) =
                get_activity_sets(&dfg, &part_a, &part_b);
            let (
                start_part_a_minus,
                end_part_a_minus,
                start_part_b_minus,
                input_part_b_minus,
                output_part_b_minus,
            ) = get_activity_sets(&dfg_minus, &part_a, &part_b);
            let activities_minus_in_part_a = &part_a & &activities_minus;
            let activities_minus_in_part_b = &part_b & &activities_minus;
            let activities_minus_in_start_part_b_minus = &start_part_b_minus & &activities_minus;
            let activities_minus_in_end_part_a_minus = &end_part_a_minus & &activities_minus;

            let mut ratio = ratio;
            if activities_minus_in_part_a.is_empty() || activities_minus_in_part_b.is_empty() {
                ratio = 0.0;
            }

            if cut_types.contains("seq") {
                let fit_seq = fit_seq(&log_variants, &part_a, &part_b);
                if fit_seq > 0.0 {
                    let cost_seq_plus = cost_seq(
                        &nx_graph,
                        &part_a,
                        &part_b,
                        &start_part_b,
                        &end_part_a,
                        sup,
                        &max_flow_graph,
                        &feat_scores,
                    );
                    let cost_seq_minus = cost_seq(
                        &nx_graph_minus,
                        &activities_minus_in_part_a,
                        &activities_minus_in_part_b,
                        &activities_minus_in_start_part_b_minus,
                        &activities_minus_in_end_part_a_minus,
                        sup,
                        &max_flow_graph_minus,
                        &feat_scores_toggle,
                    );

                    let seq_evaluation = (
                        (part_a.clone(), part_b.clone()),
                        "seq".to_string(),
                        cost_seq_plus,
                        cost_seq_minus,
                        calculate_cost(cost_seq_plus, cost_seq_minus, ratio, size_par),
                        fit_seq,
                    );
                    evaluations.push(seq_evaluation);
                }
            }

            if cut_types.contains("exc") {
                let fit_exc = fit_exc(&log_variants, &part_a, &part_b);

                if fit_exc > 0.0 {
                    let cost_exc_plus = cost_exc(&nx_graph, &part_a, &part_b, sup);
                    let cost_exc_minus = cost_exc(
                        &nx_graph_minus,
                        &activities_minus_in_part_a,
                        &activities_minus_in_part_b,
                        sup,
                    );

                    let exc_evaluation = (
                        (part_a.clone(), part_b.clone()),
                        "exc".to_string(),
                        cost_exc_plus,
                        cost_exc_minus,
                        calculate_cost(cost_exc_plus, cost_exc_minus, ratio, size_par),
                        fit_exc,
                    );
                    evaluations.push(exc_evaluation);
                }
            }

            let start_to_end_num = node_to_node_num(&nx_graph, "start", "end");
            if start_to_end_num > 0.0 {
                let missing_exc_tau_plus = f64::max(0.0, sup * log_length - start_to_end_num);
                let missing_exc_tau_minus = f64::max(
                    0.0,
                    sup * log_minus_length - node_to_node_num(&nx_graph_minus, "start", "end"),
                );
                let part_a_union_part_b = &part_a | &part_b;
                let xor_tau_evaluation = (
                    (part_a_union_part_b, HashSet::new()),
                    "exc2".to_string(),
                    missing_exc_tau_plus,
                    missing_exc_tau_minus,
                    calculate_cost(missing_exc_tau_plus, missing_exc_tau_minus, ratio, size_par),
                    1.0,
                );
                evaluations.push(xor_tau_evaluation);
            }

            if cut_types.contains("par") {
                let cost_par_plus = cost_par(
                    &nx_graph,
                    &activities_minus_in_part_a,
                    &activities_minus_in_part_b,
                    sup,
                );
                let cost_par_minus = cost_par(
                    &nx_graph_minus,
                    &activities_minus_in_part_a,
                    &activities_minus_in_part_b,
                    sup,
                );

                let par_evaluation = (
                    (part_a.clone(), part_b.clone()),
                    "par".to_string(),
                    cost_par_plus,
                    cost_par_minus,
                    calculate_cost(cost_par_plus, cost_par_minus, ratio, size_par),
                    1.0,
                );
                evaluations.push(par_evaluation);
            }

            if cut_types.contains("loop") {
                let fit_loop =
                    fit_loop(&log_variants, &part_a, &part_b, &end_part_a, &start_part_a);
                if fit_loop > 0.0 {
                    if let Some(cost_loop_plus) = cost_loop(
                        &nx_graph,
                        &part_a,
                        &part_b,
                        &start_part_a,
                        &end_part_a,
                        &input_part_b,
                        &output_part_b,
                        sup,
                    ) {
                        let cost_loop_minus = cost_loop(
                            &nx_graph_minus,
                            &part_a,
                            &part_b,
                            &start_part_a_minus,
                            &end_part_a_minus,
                            &input_part_b_minus,
                            &output_part_b_minus,
                            sup,
                        )
                        .unwrap_or(0.0);

                        let loop_evaluation = (
                            (part_a.clone(), part_b.clone()),
                            "loop".to_string(),
                            cost_loop_plus,
                            cost_loop_minus,
                            calculate_cost(cost_loop_plus, cost_loop_minus, ratio, size_par),
                            fit_loop,
                        );
                        evaluations.push(loop_evaluation);
                    }
                }
            }

            evaluations
        })
        .collect::<Vec<_>>()
}
