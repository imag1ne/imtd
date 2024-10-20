import collections
import copy
from typing import Any, Optional
from collections import Counter, defaultdict
from collections.abc import Mapping, Set
from dataclasses import dataclass, field
import numpy
from imtd.algo.discovery.inductive.util.petri_el_count import Counts
from networkx import DiGraph
from numpy.typing import NDArray
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py import util as pm_util
from imtd.algo.discovery.inductive.variants.im_bi.util import splitting as split
from pm4py.util import exec_utils
from pm4py.util import constants
from pm4py.util import xes_constants
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.algo.discovery.dfg.utils.dfg_utils import get_activities_from_dfg
from pm4py.objects.log.obj import EventLog, Event
from imtd.algo.analysis import dfg_functions
from imtd.algo.discovery.dfg import algorithm as dfg_discovery
from imtd import evaluate_cuts_imbi, find_possible_partitions


def artificial_start_end(event_log: EventLog) -> EventLog:
    """Add artificial start and end events to the traces in the event log.

    The start event is added at the beginning of each trace with the activity name 'start',
    and the end event is added at the end of each trace with the activity name 'end'.
    The start and end events are not copied, but are added as references to the same event objects.
    :param event_log: The event log to which the artificial start and end events are added.
    :return: The event log with the artificial start and end events added.
    """
    activity_key = xes_constants.DEFAULT_NAME_KEY
    start_event = Event({activity_key: 'start'})
    end_event = Event({activity_key: 'end'})

    for trace in event_log:
        trace.insert(0, start_event)
        trace.append(end_event)

    return event_log


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

    counts: Counts
    recursion_depth: int
    noise_threshold: float = 0.0

    sup: float = 0.0
    ratio: float = 0.0
    size_par: float = 1.0

    parameters: Optional[Mapping[Any, Any]] = None

    parallel: bool = False

    dfg: Optional[Mapping[tuple[str, str], int]] = None
    activities: Optional[Set[str]] = None
    detected_cut: Optional[str] = field(init=False, default=None)
    children: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.log_art = artificial_start_end(copy.deepcopy(self.log))
        self.log_minus_art = artificial_start_end(copy.deepcopy(self.log_minus))

        self.dfg = self.dfg or dfg_inst.apply(self.log, parameters=self.parameters)

        self.activities = self.activities or get_activities_from_dfg(self.dfg)
        self.start_activities = start_activities_filter.get_start_activities(self.log)
        self.start_activities_minus = start_activities_filter.get_start_activities(self.log_minus)
        self.end_activities = end_activities_filter.get_end_activities(self.log)
        self.end_activities_minus = end_activities_filter.get_end_activities(self.log_minus)

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
            ratio_backup = ratio

            possible_partitions = find_possible_partitions(nx_graph)

            if self.parallel:
                cut += evaluate_cuts(possible_partitions, dfg_art, dfg_art_minus, nx_graph, nx_graph_minus,
                                     max_flow_graph, max_flow_graph_minus, activities_minus, log_variants,
                                     len(self.log), len(self.log_minus), feat_scores, feat_scores_togg, sup, ratio,
                                     size_par)
            else:
                for part_a, part_b, cut_types in possible_partitions:
                    part_a, part_b = part_a - {'start', 'end'}, part_b - {'start', 'end'}
                    start_part_a, end_part_a, start_part_b, input_part_b, output_part_b = get_activity_sets(dfg_art,
                                                                                                            part_a,
                                                                                                            part_b)
                    start_part_a_minus, end_part_a_minus, start_part_b_minus, input_part_b_minus, output_part_b_minus = get_activity_sets(
                        dfg_art_minus, part_a, part_b)

                    if len(set(activities_minus).intersection(part_a)) == 0 or len(
                            set(activities_minus).intersection(part_b)) == 0:
                        ratio = 0
                    else:
                        ratio = ratio_backup

                        #####################################################################
                        # seq check
                    if "seq" in cut_types:
                        fit_seq = dfg_functions.fit_seq(log_variants, part_a, part_b)
                        if fit_seq > 0.0:
                            cost_seq_plus = dfg_functions.cost_seq(nx_graph, part_a, part_b, start_part_b, end_part_a,
                                                                   sup,
                                                                   max_flow_graph, feat_scores)
                            cost_seq_minus = dfg_functions.cost_seq(nx_graph_minus,
                                                                    part_a.intersection(activities_minus),
                                                                    part_b.intersection(activities_minus),
                                                                    start_part_b_minus.intersection(activities_minus),
                                                                    end_part_a_minus.intersection(activities_minus),
                                                                    sup,
                                                                    max_flow_graph_minus,
                                                                    feat_scores_togg)
                            cut.append(((part_a, part_b), 'seq', cost_seq_plus, cost_seq_minus,
                                        cost_seq_plus - ratio * size_par * cost_seq_minus,
                                        fit_seq))
                    #####################################################################

                    #####################################################################
                    # xor check
                    if "exc" in cut_types:
                        fit_exc = dfg_functions.fit_exc(log_variants, part_a, part_b)
                        if fit_exc > 0.0:
                            cost_exc_plus = dfg_functions.cost_exc(nx_graph, part_a, part_b, feat_scores)
                            cost_exc_minus = dfg_functions.cost_exc(nx_graph_minus,
                                                                    part_a.intersection(activities_minus),
                                                                    part_b.intersection(activities_minus), feat_scores)
                            cut.append(((part_a, part_b), 'exc', cost_exc_plus, cost_exc_minus,
                                        cost_exc_plus - ratio * size_par * cost_exc_minus,
                                        fit_exc))
                    #####################################################################

                    #####################################################################
                    # xor-tau check
                    if dfg_functions.n_edges(nx_graph, {'start'}, {'end'}) > 0:
                        missing_exc_tau_plus = 0
                        missing_exc_tau_plus += max(0,
                                                    sup * len(self.log) - dfg_functions.n_edges(nx_graph, {'start'},
                                                                                                {'end'}))

                        missing_exc_tau_minus = 0
                        missing_exc_tau_minus += max(0,
                                                     sup * len(self.log_minus) - dfg_functions.n_edges(nx_graph_minus,
                                                                                                       {'start'},
                                                                                                       {'end'}))

                        cost_exc_tau_plus = missing_exc_tau_plus
                        cost_exc_tau_minus = missing_exc_tau_minus
                        cut.append(((part_a.union(part_b), set()), 'exc2', cost_exc_tau_plus, cost_exc_tau_minus,
                                    cost_exc_tau_plus - ratio * size_par * cost_exc_tau_minus, 1))
                    #####################################################################

                    #####################################################################
                    # parallel check
                    if "par" in cut_types:
                        cost_par_plus = dfg_functions.cost_par(nx_graph, part_a.intersection(activities_minus),
                                                               part_b.intersection(activities_minus),
                                                               sup, feat_scores)
                        cost_par_minus = dfg_functions.cost_par(nx_graph_minus, part_a.intersection(activities_minus),
                                                                part_b.intersection(activities_minus),
                                                                sup, feat_scores)
                        cut.append(((part_a, part_b), 'par', cost_par_plus, cost_par_minus,
                                    cost_par_plus - ratio * size_par * cost_par_minus, 1))
                    #####################################################################

                    #####################################################################
                    # loop check
                    if "loop" in cut_types:
                        fit_loop = dfg_functions.fit_loop(log_variants, part_a, part_b, end_part_a, start_part_a)
                        if fit_loop > 0.0:
                            cost_loop_plus = dfg_functions.cost_loop(nx_graph, part_a, part_b, sup, start_part_a,
                                                                     end_part_a, input_part_b,
                                                                     output_part_b, feat_scores)
                            cost_loop_minus = dfg_functions.cost_loop(nx_graph_minus, part_a, part_b, sup,
                                                                      start_part_a_minus, end_part_a_minus,
                                                                      input_part_b_minus,
                                                                      output_part_b_minus, feat_scores)

                            if cost_loop_plus is not False:
                                cut.append(((part_a, part_b), 'loop', cost_loop_plus, cost_loop_minus,
                                            cost_loop_plus - ratio * size_par * cost_loop_minus, fit_loop))

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
                             self.initial_end_activities,
                             self.counts,
                             self.recursion_depth + 1,
                             self.noise_threshold, sup, ratio, size_par,
                             parameters, parallel=self.parallel))


def make_tree(log, log_minus, master_dfg, initial_dfg, initial_start_activities, initial_end_activities,
              c, recursion_depth, noise_threshold, sup=None, ratio=None,
              size_par=None, parameters=None, parallel=False):
    tree = SubtreePlain(log, log_minus, master_dfg, initial_dfg, initial_start_activities,
                        initial_end_activities,
                        c, recursion_depth, noise_threshold, sup, ratio, size_par,
                        parameters, parallel=parallel)

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


def evaluate_cuts(possible_partitions, dfg, dfg_minus, nx_graph, nx_graph_minus,
                  max_flow_graph, max_flow_graph_minus, activities_minus, log_variants,
                  log_length, log_minus_length, feat_scores, feat_scores_toggle, sup, ratio,
                  size_par):
    parameters = {
        "dfg": dfg,
        "dfg_minus": dfg_minus,
        "nx_graph": nx_graph,
        "nx_graph_minus": nx_graph_minus,
        "max_flow_graph": max_flow_graph,
        "max_flow_graph_minus": max_flow_graph_minus,
        "activities_minus": activities_minus,
        "log_variants": log_variants,
        "log_length": log_length,
        "log_minus_length": log_minus_length,
        "feat_scores": feat_scores,
        "feat_scores_toggle": feat_scores_toggle,
        "sup": sup,
        "ratio": ratio,
        "size_par": size_par
    }
    return evaluate_cuts_for_imbi(possible_partitions, parameters)
