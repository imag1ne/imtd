#![allow(dead_code)]
use crate::cost::utils::{
    average_edge_boundary_directed_num, edge_boundary_directed_num, node_to_node_num,
    node_to_nodes_num, nodes_to_node_num,
};
use crate::cost::CutCost;
use crate::graph::py_graph::PyGraph;
use std::collections::{HashMap, HashSet};

pub(crate) struct WeightedSequenceCutCost<'a, 'b> {
    start_nodes_in_part_b: &'a HashSet<&'b str>,
    end_nodes_in_part_a: &'a HashSet<&'b str>,
    sup: f64,
    flow: &'a HashMap<(&'b str, &'b str), f64>,
}

impl<'a, 'b> WeightedSequenceCutCost<'a, 'b> {
    pub(crate) fn new(
        start_nodes_in_part_b: &'a HashSet<&'b str>,
        end_nodes_in_part_a: &'a HashSet<&'b str>,
        sup: f64,
        flow: &'a HashMap<(&'b str, &'b str), f64>,
    ) -> Self {
        WeightedSequenceCutCost {
            start_nodes_in_part_b,
            end_nodes_in_part_a,
            sup,
            flow,
        }
    }
}

impl CutCost for WeightedSequenceCutCost<'_, '_> {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        // deviating edges
        let cost_1 = edge_boundary_directed_num(graph, part_b, part_a);

        let mut total_flow = 0.0;
        let mut flow_count = 0;
        for &node_a in part_a {
            for &node_b in part_b {
                if let Some(flow_value) = self.flow.get(&(node_a, node_b)) {
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
        let part_b_excluding_start = part_b - self.start_nodes_in_part_b;
        for &node_a in part_a {
            for &node_b in &part_b_excluding_start {
                let expected =
                    self.sup * avg_flow * graph.out_degree(node_a) * graph.out_degree(node_b)
                        / part_a_and_part_b_out_degree;
                let cost = f64::max(0.0, expected - self.flow[&(node_a, node_b)]);
                cost_2 += cost;
            }
        }

        let mut cost_3 = 0.0;
        let avg_weight_a_end_to_b_start = average_edge_boundary_directed_num(
            graph,
            self.end_nodes_in_part_a,
            self.start_nodes_in_part_b,
        );
        let end_node_and_start_node_out_degree = graph.nodes_out_degree(self.end_nodes_in_part_a)
            * graph.nodes_out_degree(self.start_nodes_in_part_b);

        for &end_node in self.end_nodes_in_part_a {
            for &start_node in self.start_nodes_in_part_b {
                let expected = self.sup
                    * avg_weight_a_end_to_b_start
                    * graph.out_degree(end_node)
                    * graph.out_degree(start_node)
                    / end_node_and_start_node_out_degree;
                let cost = f64::max(
                    0.0,
                    expected - node_to_node_num(graph, end_node, start_node),
                );
                cost_3 += cost;
            }
        }

        cost_1 + cost_2 + cost_3
    }
}

pub(crate) struct WeightedExclusiveChoiceCutCost;

impl CutCost for WeightedExclusiveChoiceCutCost {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        let cost_1 = edge_boundary_directed_num(graph, part_a, part_b);
        let cost_2 = edge_boundary_directed_num(graph, part_b, part_a);
        cost_1 + cost_2
    }
}

pub(crate) struct WeightedParallelCutCost {
    sup: f64,
}

impl WeightedParallelCutCost {
    pub(crate) fn new(sup: f64) -> Self {
        WeightedParallelCutCost { sup }
    }
}

impl CutCost for WeightedParallelCutCost {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        let mut cost_1 = 0.0;
        let mut cost_2 = 0.0;

        let avg_weight_a_to_b = average_edge_boundary_directed_num(graph, part_a, part_b);
        let avg_weight_b_to_a = average_edge_boundary_directed_num(graph, part_b, part_a);

        let part_a_out_degree = graph.nodes_out_degree(part_a);
        let part_b_out_degree = graph.nodes_out_degree(part_b);
        let part_a_and_part_b_out_degree = part_a_out_degree * part_b_out_degree;
        for &node_a in part_a {
            for &node_b in part_b {
                let expected = self.sup * graph.out_degree(node_a) * graph.out_degree(node_b)
                    / part_a_and_part_b_out_degree;
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
}

pub(crate) struct WeightedLoopCutCost<'a, 'b> {
    start_part_a: &'a HashSet<&'b str>,
    end_part_a: &'a HashSet<&'b str>,
    input_part_b: &'a HashSet<&'b str>,
    output_part_b: &'a HashSet<&'b str>,
    sup: f64,
}

impl<'a, 'b> WeightedLoopCutCost<'a, 'b> {
    pub(crate) fn new(
        start_part_a: &'a HashSet<&'b str>,
        end_part_a: &'a HashSet<&'b str>,
        input_part_b: &'a HashSet<&'b str>,
        output_part_b: &'a HashSet<&'b str>,
        sup: f64,
    ) -> Self {
        WeightedLoopCutCost {
            start_part_a,
            end_part_a,
            input_part_b,
            output_part_b,
            sup,
        }
    }
}

impl CutCost for WeightedLoopCutCost<'_, '_> {
    fn evaluate(&self, graph: &PyGraph, part_a: &HashSet<&str>, part_b: &HashSet<&str>) -> f64 {
        let output_part_b_to_start_part_a_num =
            edge_boundary_directed_num(graph, self.output_part_b, self.start_part_a);
        let end_part_a_to_input_part_b_num =
            edge_boundary_directed_num(graph, self.end_part_a, self.input_part_b);
        let max_observed = f64::max(
            output_part_b_to_start_part_a_num,
            end_part_a_to_input_part_b_num,
        );

        if max_observed == 0.0 {
            return f64::INFINITY;
        }

        let avg_weight_output_part_b_to_start_part_a =
            average_edge_boundary_directed_num(graph, self.output_part_b, self.start_part_a);
        let avg_weight_end_part_a_to_input_part_b =
            average_edge_boundary_directed_num(graph, self.end_part_a, self.input_part_b);
        let max_avg_weight = f64::max(
            avg_weight_output_part_b_to_start_part_a,
            avg_weight_end_part_a_to_input_part_b,
        );

        let cost_1 =
            node_to_nodes_num(graph, "start", part_b) + nodes_to_node_num(graph, part_b, "end");
        let cost_2 = edge_boundary_directed_num(graph, &(part_a - self.end_part_a), part_b);
        let cost_3 = edge_boundary_directed_num(graph, part_b, &(part_a - self.start_part_a));

        let mut cost_4 = 0.0;
        let start_part_a_out_degree = graph.nodes_out_degree(self.start_part_a);
        let output_part_b_out_degree = graph.nodes_out_degree(self.output_part_b);
        let start_part_a_and_output_part_b_out_degree =
            start_part_a_out_degree * output_part_b_out_degree;
        if !self.output_part_b.is_empty() {
            for &node_a in self.start_part_a {
                for &node_b in self.output_part_b {
                    let expected = self.sup
                        * max_avg_weight
                        * graph.out_degree(node_a)
                        * graph.out_degree(node_b)
                        / start_part_a_and_output_part_b_out_degree;
                    cost_4 += f64::max(0.0, expected - node_to_node_num(graph, node_b, node_a));
                }
            }
        }

        let mut cost_5 = 0.0;
        let end_part_a_out_degree = graph.nodes_out_degree(self.end_part_a);
        let input_part_b_out_degree = graph.nodes_out_degree(self.input_part_b);
        let end_part_a_and_input_part_b_out_degree =
            end_part_a_out_degree * input_part_b_out_degree;
        if !self.input_part_b.is_empty() {
            for &node_a in self.end_part_a {
                for &node_b in self.input_part_b {
                    let expected = self.sup
                        * max_avg_weight
                        * graph.out_degree(node_a)
                        * graph.out_degree(node_b)
                        / end_part_a_and_input_part_b_out_degree;
                    cost_5 += f64::max(0.0, expected - node_to_node_num(graph, node_a, node_b));
                }
            }
        }

        cost_1 + cost_2 + cost_3 + cost_4 + cost_5
    }
}
