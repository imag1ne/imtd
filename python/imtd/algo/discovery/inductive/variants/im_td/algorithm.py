'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import copy
from enum import Enum

import deprecation
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py.util import exec_utils, variants_util, xes_constants, constants
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.process_tree.utils import generic
from pm4py.objects.process_tree.utils.generic import tree_sort
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.objects.conversion.process_tree import converter as tree_to_petri

from imtd.algo.analysis.dfg_functions import edge_case_id_mapping
from imtd.algo.discovery.inductive.util import tree_consistency
from imtd.algo.discovery.inductive.util.petri_el_count import Counts
from imtd.algo.discovery.inductive.variants.im_td.data_structures import subtree_plain as subtree
from imtd.algo.discovery.inductive.variants.im_td.util import get_tree_repr_implain
from imtd.algo.discovery.inductive.variants.im_td.util.log_utils import artificial_start_end, \
    case_id_trace_index_mapping


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    NOISE_THRESHOLD = "noiseThreshold"
    EMPTY_TRACE_KEY = "empty_trace"
    ONCE_PER_TRACE_KEY = "once_per_trace"
    CONCURRENT_KEY = "concurrent"
    STRICT_TAU_LOOP_KEY = "strict_tau_loop"
    TAU_LOOP_KEY = "tau_loop"


def apply(logp, logm, similarity_matrix, parameters=None, sup=None, ratio=None, size_par=None):
    """
    Apply the IM algorithm to a log obtaining a Petri net along with an initial and final marking

    Parameters
    -----------
    log
        Log
    parameters
        Parameters of the algorithm, including:
            Parameters.ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    """
    if parameters is None:
        parameters = {}

    process_tree = apply_tree(logp, logm, similarity_matrix, parameters=parameters, sup=sup, ratio=ratio,
                              size_par=size_par)
    net, initial_marking, final_marking = tree_to_petri.apply(process_tree)
    return net, initial_marking, final_marking


def apply_variants(variants, parameters=None):
    """
    Apply the IM algorithm to a dictionary of variants, obtaining a Petri net along with an initial and final marking

    Parameters
    -----------
    variants
        Variants
    parameters
        Parameters of the algorithm, including:
            Parameters.ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    """
    net, im, fm = tree_to_petri.apply(apply_tree_variants(variants, parameters=parameters))
    return net, im, fm


@deprecation.deprecated('2.2.10', '3.0.0', details='use newer IM implementation (IM_CLEAN)')
def apply_tree(logp, logm, similarity_matrix, parameters=None, sup=None, ratio=None, size_par=None):
    """
    Apply the IM algorithm to a log obtaining a process tree

    Parameters
    ----------
    log
        Log
    parameters
        Parameters of the algorithm, including:
            Parameters.ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    ----------
    process_tree
        Process tree
    """
    if parameters is None:
        parameters = {}

    dfgp = [(k, v) for k, v in dfg_inst.apply(logp, parameters=parameters).items() if v > 0]

    c = Counts()
    start_activitiesp = list(start_activities_get.get_start_activities(logp, parameters=parameters).keys())
    end_activitiesp = list(end_activities_get.get_end_activities(logp, parameters=parameters).keys())
    contains_empty_traces = False
    for trace in logp:
        if len(trace) == 0:
            contains_empty_traces = True
            break

    recursion_depth = 0
    log_art = artificial_start_end(copy.deepcopy(logp))
    log_m_art = artificial_start_end(copy.deepcopy(logm))
    case_id_trace_index_map_plus = case_id_trace_index_mapping(log_art)
    case_id_trace_index_map_minus = case_id_trace_index_mapping(log_m_art)
    edge_case_id_map = edge_case_id_mapping(log_art)
    sub = subtree.make_tree(logp, logm, dfgp, dfgp, start_activitiesp,
                            end_activitiesp, similarity_matrix, case_id_trace_index_map_plus,
                            case_id_trace_index_map_minus, edge_case_id_map,
                            c, recursion_depth, 0.0, sup, ratio, size_par, parameters)

    process_tree = get_tree_repr_implain.get_repr(sub, 0, contains_empty_traces=contains_empty_traces)
    # Ensures consistency to the parent pointers in the process tree
    tree_consistency.fix_parent_pointers(process_tree)
    # Fixes a 1 child XOR that is added when single-activities flowers are found
    tree_consistency.fix_one_child_xor_flower(process_tree)
    # folds the process tree (to simplify it in case fallthroughs/filtering is applied)
    process_tree = generic.fold(process_tree)
    # sorts the process tree to ensure consistency in different executions of the algorithm
    tree_sort(process_tree)

    return process_tree


def apply_tree_variants(variants, parameters=None):
    """
    Apply the IM algorithm to a dictionary of variants obtaining a process tree

    Parameters
    ----------
    variants
        Variants
    parameters
        Parameters of the algorithm, including:
            Parameters.ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    ----------
    process_tree
        Process tree
    """
    log = EventLog()
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)

    for var in variants.keys():
        trace = Trace()
        activities = variants_util.get_activities_from_variant(var)
        for act in activities:
            trace.append(Event({activity_key: act}))
        log.append(trace)

    return apply_tree(log, parameters=parameters)
