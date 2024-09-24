import collections
import copy
from typing import Any, Optional
from collections import Counter, defaultdict
from collections.abc import Mapping, Set
from dataclasses import dataclass, field

import numpy
from networkx import DiGraph
from numpy.typing import NDArray
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py import util as pm_util
from pm4py.util import exec_utils
from pm4py.util import constants
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.algo.discovery.dfg.utils.dfg_utils import get_activities_from_dfg
from pm4py.objects.log.obj import EventLog

from imtd.algo.analysis import dfg_functions
from imtd.algo.discovery.dfg import algorithm as dfg_discovery
from imtd.algo.discovery.inductive.util.petri_el_count import Counts
from imtd.algo.discovery.inductive.variants.im_bi.util import splitting as split
from imtd.algo.discovery.inductive.variants.im_td.util import log_utils
from imtd import evaluate_cuts, find_possible_partitions


def generate_nx_graph_from_dfg(dfg: dict[tuple[str, str], float]) -> DiGraph:
    """Generate a NetworkX directed graph from a Directly-Follows Graph (DFG).

    :param dfg: The Directly-Follows Graph (DFG) from which the NetworkX directed graph is generated.
    :return: The NetworkX directed graph generated from the Directly-Follows Graph (DFG).
    """
    graph = DiGraph()

    graph.add_weighted_edges_from([(u, v, w) for (u, v), w in dfg.items()])

    return graph


@dataclass
class SubtreePlain:
    # Event logs
    log: EventLog
    log_minus: EventLog
    log_art: EventLog = field(init=False)
    log_minus_art: EventLog = field(init=False)

    # DFGs
    master_dfg: Mapping[tuple[str, str], int]
    initial_dfg: Mapping[tuple[str, str], int]

    # Activities
    initial_start_activities: Set[str]
    initial_end_activities: Set[str]
    start_activities: Mapping[str, int] = field(init=False)
    start_activities_minus: Mapping[str, int] = field(init=False)
    end_activities: Mapping[str, int] = field(init=False)
    end_activities_minus: Mapping[str, int] = field(init=False)

    similarity_matrix: NDArray[numpy.float64]
    case_id_trace_index_map: defaultdict[tuple[str, str], list[int]]
    case_id_trace_index_map_minus: defaultdict[tuple[str, str], list[int]]

    counts: Counts
    recursion_depth: int
    noise_threshold: float = 0.0

    sup: float = 0.0
    ratio: float = 0.0
    size_par: float = 1.0

    parameters: Optional[Mapping[Any, Any]] = None

    dfg: Optional[Mapping[tuple[str, str], int]] = None
    activities: Optional[Set[str]] = None
    edge_case_id_map: Optional[Mapping[tuple[str, str], Set[str]]] = None
    edge_case_id_map_minus: Optional[Mapping[tuple[str, str], Set[str]]] = None
    detected_cut: Optional[str] = field(init=False, default=None)
    children: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.log_art = log_utils.artificial_start_end(copy.deepcopy(self.log))
        self.log_minus_art = log_utils.artificial_start_end(copy.deepcopy(self.log_minus))

        self.dfg = self.dfg or dfg_inst.apply(self.log, parameters=self.parameters)

        self.activities = self.activities or get_activities_from_dfg(self.dfg)
        self.start_activities = start_activities_filter.get_start_activities(self.log)
        self.start_activities_minus = start_activities_filter.get_start_activities(self.log_minus)
        self.end_activities = end_activities_filter.get_end_activities(self.log)
        self.end_activities_minus = end_activities_filter.get_end_activities(self.log_minus)
        self.edge_case_id_map = self.edge_case_id_map or dfg_functions.edge_case_id_mapping(self.log_art)
        self.edge_case_id_map_minus = self.edge_case_id_map_minus or dfg_functions.edge_case_id_mapping(
            self.log_minus_art)

        self.detect_cut()

    def detect_cut(self, _second_iteration=False):
        ratio = self.ratio
        sup = self.sup
        size_par = self.size_par

        parameters = self.parameters
        log_variants = Counter([tuple([x['concept:name'] for x in t]) for t in self.log])
        log_minus_variants = Counter([tuple([x['concept:name'] for x in t]) for t in self.log_minus])

        parameters = parameters or {}
        activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, parameters,
                                                  pm_util.xes_constants.DEFAULT_NAME_KEY)
        # check base cases:
        is_base, cut = dfg_functions.check_base_case(self, log_variants, log_minus_variants, sup, ratio, size_par)

        if not is_base:
            feat_scores, feat_scores_togg = initialize_feature_scores(self.log_art, self.log_minus_art)

            dfg_art = dfg_discovery.apply(self.log_art, variant=dfg_discovery.Variants.FREQUENCY)
            dfg_art_minus = dfg_discovery.apply(self.log_minus_art, variant=dfg_discovery.Variants.FREQUENCY)

            nx_graph = generate_nx_graph_from_dfg(dfg_art)
            nx_graph_minus = generate_nx_graph_from_dfg(dfg_art_minus)

            max_flow_graph = dfg_functions.max_flow_graph(nx_graph)
            max_flow_graph_minus = dfg_functions.max_flow_graph(nx_graph_minus)

            activities_minus = set(a for x in log_minus_variants.keys() for a in x)
            start_activities = frozenset(self.start_activities.keys())
            end_activities = frozenset(self.end_activities.keys())

            cut = self.evaluate_start_end_loop_tau(start_activities, end_activities, dfg_art, dfg_art_minus, sup, ratio,
                                                   size_par)

            possible_partitions = find_possible_partitions(nx_graph)

            cut += evaluate_cuts(possible_partitions, dfg_art, dfg_art_minus, nx_graph, nx_graph_minus,
                                 max_flow_graph, max_flow_graph_minus, activities_minus, log_variants,
                                 len(self.log), len(self.log_minus), feat_scores, feat_scores_togg, sup, ratio,
                                 size_par)

            sorted_cuts = sorted(cut, key=lambda x: (
                x[4], x[2], ['exc', 'exc2', 'seq', 'par', 'loop', 'loop_tau'].index(x[1]),
                -(len(x[0][0]) * len(x[0][1]) / (len(x[0][0]) + len(x[0][1])))))

            if len(sorted_cuts) != 0:
                cut = sorted_cuts[0]
            else:
                cut = ('none', 'none', 'none', 'none', 'none', 'none')

        if cut[1] == 'par':
            self.detected_cut = 'parallel'
            self.split_and_create_subtree('par', cut, activity_key, parameters, sup, ratio, size_par)
        elif cut[1] == 'seq':
            self.detected_cut = 'sequential'
            self.split_and_create_subtree('seq', cut, activity_key, parameters, sup, ratio, size_par)
        elif (cut[1] == 'exc') or (cut[1] == 'exc2'):
            self.detected_cut = 'concurrent'
            self.split_and_create_subtree('exc', cut, activity_key, parameters, sup, ratio, size_par)
        elif cut[1] == 'loop':
            self.detected_cut = 'loopCut'
            self.split_and_create_subtree('loop', cut, activity_key, parameters, sup, ratio, size_par)
        elif cut[1] == 'loop1':
            self.detected_cut = 'loopCut'
            self.split_and_create_subtree('loop1', cut, activity_key, parameters, sup, ratio, size_par)
        elif cut[1] == 'loop_tau':
            self.detected_cut = 'loopCut'
            self.split_and_create_subtree('loop_tau', cut, activity_key, parameters, sup, ratio, size_par)
        elif cut[1] == 'none':
            self.detected_cut = 'flower'

    def evaluate_start_end_loop_tau(self, start_acts_plus, end_acts_plus, dfg_plus, dfg_minus, sup, ratio, size_par):
        cut = []
        missing_loop_plus, missing_loop_minus, c_rec = 0, 0, 0
        rej_tau_loop = len(start_acts_plus.intersection(end_acts_plus)) != 0

        missing_loop_plus = calculate_missing_loop(len(self.log), self.start_activities, self.end_activities, dfg_plus,
                                                   sup)
        missing_loop_minus = calculate_missing_loop(len(self.log_minus), self.start_activities_minus,
                                                    self.end_activities_minus, dfg_minus, sup)

        for x in start_acts_plus:
            for y in end_acts_plus:
                c_rec += dfg_plus[(y, x)]

        if not rej_tau_loop and c_rec > 0:
            cut.append(((start_acts_plus, end_acts_plus), 'loop_tau', missing_loop_plus, missing_loop_minus,
                        missing_loop_plus - ratio * size_par * missing_loop_minus, 1))

        return cut

    def split_and_create_subtree(self, cut_type, cut, activity_key, parameters, sup, ratio, size_par):
        log_a, log_b = split.split(cut_type, [cut[0][0], cut[0][1]], self.log, activity_key)
        log_minus_a, log_minus_b = split.split(cut_type, [cut[0][0], cut[0][1]], self.log_minus, activity_key)
        new_logs = [[log_a, log_minus_a], [log_b, log_minus_b]]
        for new_log, new_log_minus in new_logs:
            self.children.append(
                SubtreePlain(new_log, new_log_minus, self.master_dfg, self.initial_dfg, self.initial_start_activities,
                             self.initial_end_activities, self.similarity_matrix, self.case_id_trace_index_map,
                             self.case_id_trace_index_map_minus,
                             self.counts,
                             self.recursion_depth + 1,
                             self.noise_threshold, sup, ratio, size_par,
                             parameters))


def make_tree(log, log_minus, master_dfg, initial_dfg, initial_start_activities, initial_end_activities,
              similarity_matrix,
              case_id_trace_index_map_plus, case_id_trace_index_map_minus,
              c, recursion_depth, noise_threshold, sup=None, ratio=None,
              size_par=None, parameters=None):
    tree = SubtreePlain(log, log_minus, master_dfg, initial_dfg, initial_start_activities,
                        initial_end_activities, similarity_matrix, case_id_trace_index_map_plus,
                        case_id_trace_index_map_minus,
                        c, recursion_depth, noise_threshold, sup, ratio, size_par,
                        parameters)

    return tree


def calculate_missing_loop(trace_num, start_activities, end_activities, dfg, sup):
    missing_loop = 0
    for x in start_activities:
        for y in end_activities:
            n = max(0, trace_num * sup * (start_activities[x] / sum(start_activities.values())) * (
                    end_activities[y] / sum(end_activities.values())) - dfg[(y, x)])
            missing_loop += n
    return missing_loop


def initialize_feature_scores(log_plus, log_minus):
    dfg_plus = dfg_discovery.apply(log_plus, variant=dfg_discovery.Variants.FREQUENCY)
    del dfg_plus[('start', 'end')]
    dfg_minus = dfg_discovery.apply(log_minus, variant=dfg_discovery.Variants.FREQUENCY)
    del dfg_minus[('start', 'end')]

    feat_scores_togg = collections.defaultdict(lambda: 1)
    feat_scores = collections.defaultdict(lambda: 1)

    for x in dfg_plus.keys():
        feat_scores[x] = 1
        feat_scores_togg[x] = 1

    for y in dfg_minus.keys():
        feat_scores[y] = 1
        feat_scores_togg[y] = 1

    return feat_scores, feat_scores_togg


def get_activity_sets(dfg, activity_set_1, activity_set_2):
    start_activities_in_set_1 = get_start_activities_from_dfg_with_artificial_start(dfg, activity_set_1)
    end_activities_in_set_1 = get_end_activities_from_dfg_with_artificial_end(dfg, activity_set_1)
    start_activities_in_set_2 = get_start_activities_from_dfg_with_artificial_start(dfg, activity_set_2)
    input_to_activity_set_2 = set(t for s, t in dfg if ((s not in activity_set_2) and (t in activity_set_2)))
    output_from_activity_set_2 = set(s for s, t in dfg if ((s in activity_set_2) and (t not in activity_set_2)))
    return start_activities_in_set_1, end_activities_in_set_1, start_activities_in_set_2, input_to_activity_set_2, output_from_activity_set_2


def get_start_activities_from_dfg_with_artificial_start(dfg, activities):
    return set(t for s, t in dfg if ((s == 'start') and (t in activities)))


def get_end_activities_from_dfg_with_artificial_end(dfg, activities):
    return set(s for s, t in dfg if (s in activities and (t == 'end')))