mod balanced;
mod weighted;

use crate::cost::imfc::balanced::{
    BalancedExclusiveChoiceCutCost, BalancedLoopCutCost, BalancedParallelCutCost,
    BalancedSequenceCutCost,
};
use crate::cost::{
    fit::{fit_exc, fit_loop, fit_seq},
    utils::{get_activity_sets, node_to_node_num},
    CutCost, CutEvaluations,
};
use crate::graph::py_graph::PyGraph;
use pyo3::types::PyAnyMethods;
use pyo3::{pyfunction, Bound, FromPyObject, PyAny, PyResult};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::{HashMap, HashSet};

pub struct ImfcEvaluatorParameters<'a> {
    dfg: HashMap<(&'a str, &'a str), f64>,
    nx_graph: PyGraph<'a>,
    max_flow_graph: HashMap<(&'a str, &'a str), f64>,
    activities_minus: HashSet<&'a str>,
    log_variants: HashMap<Vec<&'a str>, f64>,
    log_length: f64,
    sup: f64,
}

#[pyfunction]
pub fn evaluate_cuts_imfc<'a>(
    cuts: Vec<(HashSet<&'a str>, HashSet<&'a str>, HashSet<&str>)>,
    parameters: ImfcEvaluatorParameters<'a>,
) -> CutEvaluations<'a> {
    let ImfcEvaluatorParameters {
        dfg,
        nx_graph,
        max_flow_graph,
        activities_minus,
        log_variants,
        log_length,
        sup,
    } = parameters;

    let start_to_end_num = node_to_node_num(&nx_graph, "start", "end");
    let missing_exc_tau_plus = f64::max(0.0, sup * log_length - start_to_end_num);
    cuts.par_iter()
        .flat_map(|(part_a, part_b, cut_types)| {
            let start_and_end_set = HashSet::from(["start", "end"]);
            let part_a = part_a - &start_and_end_set;
            let part_b = part_b - &start_and_end_set;
            let mut evaluations = vec![];

            let (start_part_a, end_part_a, start_part_b, input_part_b, output_part_b) =
                get_activity_sets(&dfg, &part_a, &part_b);
            let _activities_minus_in_part_a = &part_a & &activities_minus;
            let _activities_minus_in_part_b = &part_b & &activities_minus;

            if cut_types.contains("seq") {
                let fit_seq = fit_seq(&log_variants, &part_a, &part_b);
                if fit_seq > 0.0 {
                    let seq_cost_evaluator = BalancedSequenceCutCost::new(
                        &start_part_b,
                        &end_part_a,
                        sup,
                        &max_flow_graph,
                    );
                    let seq_cost = seq_cost_evaluator.evaluate(&nx_graph, &part_a, &part_b);

                    let seq_evaluation = (
                        (part_a.clone(), part_b.clone()),
                        "seq".to_string(),
                        seq_cost,
                        0.0,
                        seq_cost,
                        fit_seq,
                    );
                    evaluations.push(seq_evaluation);
                }
            }

            if cut_types.contains("exc") {
                let fit_exc = fit_exc(&log_variants, &part_a, &part_b);

                if fit_exc > 0.0 {
                    let exc_cost_evaluator = BalancedExclusiveChoiceCutCost::new(sup);
                    let cost_exc = exc_cost_evaluator.evaluate(&nx_graph, &part_a, &part_b);

                    let exc_evaluation = (
                        (part_a.clone(), part_b.clone()),
                        "exc".to_string(),
                        cost_exc,
                        0.0,
                        cost_exc,
                        fit_exc,
                    );
                    evaluations.push(exc_evaluation);
                }
            }

            if start_to_end_num > 0.0 {
                let part_a_union_part_b = &part_a | &part_b;
                let xor_tau_evaluation = (
                    (part_a_union_part_b, HashSet::new()),
                    "exc2".to_string(),
                    missing_exc_tau_plus,
                    0.0,
                    missing_exc_tau_plus,
                    1.0,
                );
                evaluations.push(xor_tau_evaluation);
            }

            if cut_types.contains("par") {
                let parallel_cost_evaluator = BalancedParallelCutCost::new(sup);
                let parallel_cost = parallel_cost_evaluator.evaluate(&nx_graph, &part_a, &part_b);

                let par_evaluation = (
                    (part_a.clone(), part_b.clone()),
                    "par".to_string(),
                    parallel_cost,
                    0.0,
                    parallel_cost,
                    1.0,
                );
                evaluations.push(par_evaluation);
            }

            if cut_types.contains("loop") {
                let fit_loop =
                    fit_loop(&log_variants, &part_a, &part_b, &end_part_a, &start_part_a);
                if fit_loop > 0.0 {
                    let loop_cost_evaluator = BalancedLoopCutCost::new(
                        &start_part_a,
                        &end_part_a,
                        &input_part_b,
                        &output_part_b,
                        sup,
                    );
                    let loop_cost = loop_cost_evaluator.evaluate(&nx_graph, &part_a, &part_b);
                    if loop_cost.is_finite() {
                        let loop_evaluation = (
                            (part_a.clone(), part_b.clone()),
                            "loop".to_string(),
                            loop_cost,
                            0.0,
                            loop_cost,
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

impl<'py> FromPyObject<'py> for ImfcEvaluatorParameters<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dfg: HashMap<(&str, &str), f64> = ob.get_item("dfg")?.extract()?;
        let nx_graph: PyGraph = ob.get_item("nx_graph")?.extract()?;
        let max_flow_graph: HashMap<(&str, &str), f64> =
            ob.get_item("max_flow_graph")?.extract()?;
        let activities_minus: HashSet<&str> = ob.get_item("activities_minus")?.extract()?;
        let log_variants: HashMap<Vec<&str>, f64> = ob.get_item("log_variants")?.extract()?;
        let log_length: f64 = ob.get_item("log_length")?.extract()?;
        let sup: f64 = ob.get_item("sup")?.extract()?;

        Ok(ImfcEvaluatorParameters {
            dfg,
            nx_graph,
            max_flow_graph,
            activities_minus,
            log_variants,
            log_length,
            sup,
        })
    }
}
