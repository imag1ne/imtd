use crate::cost::fit::{fit_exc, fit_loop, fit_seq};
use crate::cost::utils::{
    edge_boundary_directed_num, get_activity_sets, node_to_node_num, node_to_nodes_num,
    nodes_to_node_num,
};
use crate::cost::{CutCost, CutEvaluations};
use crate::graph::py_graph::PyGraph;
use pyo3::types::PyAnyMethods;
use pyo3::{pyfunction, Bound, FromPyObject, PyAny, PyResult};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::{HashMap, HashSet};

pub struct ImbiEvaluatorParameters<'a> {
    dfg: HashMap<(&'a str, &'a str), f64>,
    dfg_minus: HashMap<(&'a str, &'a str), f64>,
    nx_graph: PyGraph<'a>,
    nx_graph_minus: PyGraph<'a>,
    max_flow_graph: HashMap<(&'a str, &'a str), f64>,
    max_flow_graph_minus: HashMap<(&'a str, &'a str), f64>,
    activities_minus: HashSet<&'a str>,
    log_variants: HashMap<Vec<&'a str>, f64>,
    log_length: f64,
    log_minus_length: f64,
    feat_scores: HashMap<(&'a str, &'a str), f64>,
    feat_scores_toggle: HashMap<(&'a str, &'a str), f64>,
    sup: f64,
    ratio: f64,
    size_par: f64,
}

#[pyfunction]
pub fn evaluate_cuts_imbi<'a>(
    cuts: Vec<(HashSet<&'a str>, HashSet<&'a str>, HashSet<&str>)>,
    params: ImbiEvaluatorParameters<'a>,
) -> CutEvaluations<'a> {
    let ImbiEvaluatorParameters {
        dfg,
        dfg_minus,
        nx_graph,
        nx_graph_minus,
        max_flow_graph,
        max_flow_graph_minus,
        activities_minus,
        log_variants,
        log_length,
        log_minus_length,
        feat_scores,
        feat_scores_toggle,
        sup,
        ratio,
        size_par,
    } = params;

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
                    let sequence_cut_cost_plus_evaluator = SequenceCutCost::new(
                        &start_part_b,
                        &end_part_a,
                        sup,
                        &max_flow_graph,
                        &feat_scores,
                    );
                    let cost_seq_plus =
                        sequence_cut_cost_plus_evaluator.evaluate(&nx_graph, &part_a, &part_b);
                    let sequence_cut_cost_minus_evaluator = SequenceCutCost::new(
                        &activities_minus_in_start_part_b_minus,
                        &activities_minus_in_end_part_a_minus,
                        sup,
                        &max_flow_graph_minus,
                        &feat_scores_toggle,
                    );
                    let cost_seq_minus = sequence_cut_cost_minus_evaluator.evaluate(
                        &nx_graph_minus,
                        &activities_minus_in_part_a,
                        &activities_minus_in_part_b,
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
                    let cost_exc_plus =
                        ExclusiveChoiceCutCost.evaluate(&nx_graph, &part_a, &part_b);
                    let cost_exc_minus = ExclusiveChoiceCutCost.evaluate(
                        &nx_graph_minus,
                        &activities_minus_in_part_a,
                        &activities_minus_in_part_b,
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
                let par_cost_evaluator = ParallelCutCost::new(sup);
                let cost_par_plus = par_cost_evaluator.evaluate(
                    &nx_graph,
                    &activities_minus_in_part_a,
                    &activities_minus_in_part_b,
                );
                let cost_par_minus = par_cost_evaluator.evaluate(
                    &nx_graph_minus,
                    &activities_minus_in_part_a,
                    &activities_minus_in_part_b,
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
                    let loop_cost_plus_evaluator = LoopCutCost::new(
                        &start_part_a,
                        &end_part_a,
                        &input_part_b,
                        &output_part_b,
                        sup,
                    );
                    let loop_cost_plus =
                        loop_cost_plus_evaluator.evaluate(&nx_graph, &part_a, &part_b);
                    if loop_cost_plus.is_finite() {
                        let loop_cost_minus_evaluator = LoopCutCost::new(
                            &start_part_a_minus,
                            &end_part_a_minus,
                            &input_part_b_minus,
                            &output_part_b_minus,
                            sup,
                        );
                        let loop_cost_minus =
                            loop_cost_minus_evaluator.evaluate(&nx_graph_minus, &part_a, &part_b);

                        let loop_evaluation = (
                            (part_a.clone(), part_b.clone()),
                            "loop".to_string(),
                            loop_cost_plus,
                            loop_cost_minus,
                            calculate_cost(loop_cost_plus, loop_cost_minus, ratio, size_par),
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

impl<'py> FromPyObject<'py> for ImbiEvaluatorParameters<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dfg: HashMap<(&str, &str), f64> = ob.get_item("dfg")?.extract()?;
        let dfg_minus: HashMap<(&str, &str), f64> = ob.get_item("dfg_minus")?.extract()?;
        let nx_graph: PyGraph = ob.get_item("nx_graph")?.extract()?;
        let nx_graph_minus: PyGraph = ob.get_item("nx_graph_minus")?.extract()?;
        let max_flow_graph: HashMap<(&str, &str), f64> =
            ob.get_item("max_flow_graph")?.extract()?;
        let max_flow_graph_minus: HashMap<(&str, &str), f64> =
            ob.get_item("max_flow_graph_minus")?.extract()?;
        let activities_minus: HashSet<&str> = ob.get_item("activities_minus")?.extract()?;
        let log_variants: HashMap<Vec<&str>, f64> = ob.get_item("log_variants")?.extract()?;
        let log_length: f64 = ob.get_item("log_length")?.extract()?;
        let log_minus_length: f64 = ob.get_item("log_minus_length")?.extract()?;
        let feat_scores: HashMap<(&str, &str), f64> = ob.get_item("feat_scores")?.extract()?;
        let feat_scores_toggle: HashMap<(&str, &str), f64> =
            ob.get_item("feat_scores_toggle")?.extract()?;
        let sup: f64 = ob.get_item("sup")?.extract()?;
        let ratio: f64 = ob.get_item("ratio")?.extract()?;
        let size_par: f64 = ob.get_item("size_par")?.extract()?;

        Ok(ImbiEvaluatorParameters {
            dfg,
            dfg_minus,
            nx_graph,
            nx_graph_minus,
            max_flow_graph,
            max_flow_graph_minus,
            activities_minus,
            log_variants,
            log_length,
            log_minus_length,
            feat_scores,
            feat_scores_toggle,
            sup,
            ratio,
            size_par,
        })
    }
}

fn calculate_cost(cost_plus: f64, cost_minus: f64, ratio: f64, size_par: f64) -> f64 {
    cost_plus - ratio * size_par * cost_minus
}

struct SequenceCutCost<'a> {
    start_nodes_in_part_b: &'a HashSet<&'a str>,
    end_nodes_in_part_a: &'a HashSet<&'a str>,
    sup: f64,
    flow: &'a HashMap<(&'a str, &'a str), f64>,
    scores: &'a HashMap<(&'a str, &'a str), f64>,
}

impl<'a> SequenceCutCost<'a> {
    fn new(
        start_nodes_in_part_b: &'a HashSet<&'a str>,
        end_nodes_in_part_a: &'a HashSet<&'a str>,
        sup: f64,
        flow: &'a HashMap<(&'a str, &'a str), f64>,
        scores: &'a HashMap<(&'a str, &'a str), f64>,
    ) -> Self {
        SequenceCutCost {
            start_nodes_in_part_b,
            end_nodes_in_part_a,
            sup,
            flow,
            scores,
        }
    }
}

impl CutCost for SequenceCutCost<'_> {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        // deviating edges
        let cost_1 = edge_boundary_directed_num(graph, part_b, part_a);

        let mut cost_2 = 0.0;
        let part_a_out_degree = graph.nodes_out_degree(part_a);
        let part_b_out_degree = graph.nodes_out_degree(part_b);
        let part_a_and_part_b_out_degree = part_a_out_degree + part_b_out_degree;
        for &node_a in part_a {
            for &node_b in part_b {
                let cost = f64::max(
                    0.0,
                    self.scores.get(&(node_a, node_b)).unwrap_or(&1.0)
                        * graph.out_degree(node_a)
                        * self.sup
                        * (graph.out_degree(node_b) / part_a_and_part_b_out_degree)
                        - self.flow[&(node_a, node_b)],
                );
                cost_2 += cost;
            }
        }

        let mut cost_3 = 0.0;
        let part_a_with_start = part_a | &HashSet::from(["start"]);
        let part_b_with_end = part_b | &HashSet::from(["end"]);
        let part_a_with_start_to_part_b_with_end_num =
            edge_boundary_directed_num(graph, &part_a_with_start, &part_b_with_end);
        for &end_node in self.end_nodes_in_part_a {
            for &start_node in self.start_nodes_in_part_b {
                let num_end_node_to_part_b_with_end =
                    node_to_nodes_num(graph, end_node, &part_b_with_end);
                let num_part_a_with_start_to_start_node =
                    nodes_to_node_num(graph, &part_a_with_start, start_node);
                let cost = f64::max(
                    0.0,
                    self.scores.get(&(end_node, start_node)).unwrap_or(&1.0)
                        * num_end_node_to_part_b_with_end
                        * self.sup
                        * num_part_a_with_start_to_start_node
                        / part_a_with_start_to_part_b_with_end_num
                        - node_to_node_num(graph, end_node, start_node),
                );
                cost_3 += cost;
            }
        }

        cost_1 + cost_2 + cost_3
    }
}

struct ExclusiveChoiceCutCost;

impl CutCost for ExclusiveChoiceCutCost {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        let cost_1 = edge_boundary_directed_num(graph, part_a, part_b);
        let cost_2 = edge_boundary_directed_num(graph, part_b, part_a);
        cost_1 + cost_2
    }
}

struct ParallelCutCost {
    sup: f64,
}

impl ParallelCutCost {
    fn new(sup: f64) -> Self {
        ParallelCutCost { sup }
    }
}

impl CutCost for ParallelCutCost {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        let mut cost_1 = 0.0;
        let mut cost_2 = 0.0;

        let part_a_out_degree = graph.nodes_out_degree(part_a);
        let part_b_out_degree = graph.nodes_out_degree(part_b);
        for &node_a in part_a {
            for &node_b in part_b {
                let c = self.sup * graph.out_degree(node_a) * graph.out_degree(node_b)
                    / (part_a_out_degree + part_b_out_degree);
                cost_1 += f64::max(0.0, c - node_to_node_num(graph, node_a, node_b));
                cost_2 += f64::max(0.0, c - node_to_node_num(graph, node_b, node_a));
            }
        }

        cost_1 + cost_2
    }
}

struct LoopCutCost<'a, 'b> {
    start_part_a: &'a HashSet<&'b str>,
    end_part_a: &'a HashSet<&'b str>,
    input_part_b: &'a HashSet<&'b str>,
    output_part_b: &'a HashSet<&'b str>,
    sup: f64,
}

impl<'a, 'b> LoopCutCost<'a, 'b> {
    fn new(
        start_part_a: &'a HashSet<&'b str>,
        end_part_a: &'a HashSet<&'b str>,
        input_part_b: &'a HashSet<&'b str>,
        output_part_b: &'a HashSet<&'b str>,
        sup: f64,
    ) -> Self {
        LoopCutCost {
            start_part_a,
            end_part_a,
            input_part_b,
            output_part_b,
            sup,
        }
    }
}

impl CutCost for LoopCutCost<'_, '_> {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        if edge_boundary_directed_num(graph, part_b, self.start_part_a) != 0.0 {
            if edge_boundary_directed_num(graph, self.end_part_a, part_b) == 0.0 {
                return f64::INFINITY;
            }
        } else {
            return f64::INFINITY;
        }

        let output_part_b_to_start_part_a_num =
            edge_boundary_directed_num(graph, self.output_part_b, self.start_part_a);
        let end_part_a_to_input_part_b_num =
            edge_boundary_directed_num(graph, self.end_part_a, self.input_part_b);
        let m_p = f64::max(
            output_part_b_to_start_part_a_num,
            end_part_a_to_input_part_b_num,
        );

        let cost_1 =
            node_to_nodes_num(graph, "start", part_b) + nodes_to_node_num(graph, part_b, "end");
        let cost_2 = edge_boundary_directed_num(graph, &(part_a - self.end_part_a), part_b);
        let cost_3 = edge_boundary_directed_num(graph, part_b, &(part_a - self.start_part_a));

        let mut cost_4 = 0.0;
        let start_to_start_part_a_num = node_to_nodes_num(graph, "start", self.start_part_a);
        let output_part_b_to_start_part_a_num =
            edge_boundary_directed_num(graph, self.output_part_b, self.start_part_a);
        if !self.output_part_b.is_empty() {
            for &node_a in self.start_part_a {
                for &node_b in self.output_part_b {
                    let c = m_p
                        * self.sup
                        * (node_to_node_num(graph, "start", node_a) / start_to_start_part_a_num)
                        * (node_to_nodes_num(graph, node_b, self.start_part_a)
                            / output_part_b_to_start_part_a_num);
                    cost_4 += f64::max(0.0, c - node_to_node_num(graph, node_b, node_a));
                }
            }
        }

        let mut cost_5 = 0.0;
        let end_part_a_to_end_num = nodes_to_node_num(graph, self.end_part_a, "end");
        let end_part_a_to_input_part_b_num =
            edge_boundary_directed_num(graph, self.end_part_a, self.input_part_b);
        if !self.input_part_b.is_empty() {
            for &node_a in self.end_part_a {
                for &node_b in self.input_part_b {
                    let c = m_p
                        * self.sup
                        * (node_to_node_num(graph, node_a, "end") / end_part_a_to_end_num)
                        * (nodes_to_node_num(graph, self.end_part_a, node_b)
                            / end_part_a_to_input_part_b_num);
                    cost_5 += f64::max(0.0, c - node_to_node_num(graph, node_a, node_b));
                }
            }
        }

        if self.sup * m_p == 0.0 {
            return f64::INFINITY;
        }

        if (cost_4 + cost_5) / (2.0 * self.sup * m_p) > 0.3 {
            return f64::INFINITY;
        }

        cost_1 + cost_2 + cost_3 + cost_4 + cost_5
    }
}
