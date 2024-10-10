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
__doc__ = """
The ``pm4py.discovery`` module contains the process discovery algorithms implemented in ``pm4py``
"""

from typing import Tuple, Union

import pandas as pd

from pm4py.objects.dfg.obj import DFG
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.obj import EventStream
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
from pm4py.utils import get_properties, __event_log_deprecation_warning
from pm4py.util import constants
from pm4py.algo.discovery.dfg.variants import native as dfg_inst


def discover_petri_net_inductive_bi(logp, logm, parameters=None, sup=None, ratio=None, size_par=None, parallel=False):
    from imtd.algo.discovery.inductive.variants.im_bi import algorithm as im_bi_algo
    return im_bi_algo.apply(logp, logm, parameters=parameters, sup=sup, ratio=ratio, size_par=size_par,
                            parallel=parallel)


def discover_petri_net_inductive_td(logp, logm, similarity_matrix, parameters=None, sup=None, ratio=None,
                                    size_par=None, weight=None):
    from imtd.algo.discovery.inductive.variants.im_td import algorithm as im_td_algo
    return im_td_algo.apply(logp, logm, similarity_matrix, parameters=parameters, sup=sup, ratio=ratio,
                            size_par=size_par, weight=weight)


def discover_petri_net_inductive(log: Union[EventLog, pd.DataFrame, DFG],
                                 undesirable_log: Union[EventLog, pd.DataFrame, DFG],
                                 filter_ratio: float = 0.0, noise_threshold: float = 0.0,
                                 multi_processing: bool = False,
                                 activity_key: str = "concept:name", timestamp_key: str = "time:timestamp",
                                 case_id_key: str = "case:concept:name"):
    """
    Discovers a Petri net using the inductive miner algorithm.

    The basic idea of Inductive Miner is about detecting a 'cut' in the log (e.g. sequential cut, parallel cut, concurrent cut and loop cut) and then recur on sublogs, which were found applying the cut, until a base case is found. The Directly-Follows variant avoids the recursion on the sublogs but uses the Directly Follows graph.

    Inductive miner models usually make extensive use of hidden transitions, especially for skipping/looping on a portion on the model. Furthermore, each visible transition has a unique label (there are no transitions in the model that share the same label).

    :param log: event log / Pandas dataframe / typed DFG
    :param undesirable_log: event log / Pandas dataframe / typed DFG
    :param filter_ratio: filter ratio (default: 0.0)
    :param noise_threshold: noise threshold (default: 0.0)
    :param multi_processing: boolean that enables/disables multiprocessing in inductive miner
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :rtype: ``Tuple[PetriNet, Marking, Marking]``

    .. code-block:: python3

        import pm4py

        net, im, fm = pm4py.discover_petri_net_inductive(dataframe, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream, DFG]:
        raise Exception(
            "the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(
            log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)

    pt = discover_process_tree_inductive(
        log, undesirable_log, filter_ratio, noise_threshold, multi_processing=multi_processing,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        case_id_key=case_id_key)
    from pm4py.convert import convert_to_petri_net
    return convert_to_petri_net(pt)


def discover_process_tree_inductive(log: Union[EventLog, pd.DataFrame, DFG],
                                    undesirable_log: Union[EventLog, pd.DataFrame, DFG],
                                    filter_ratio: float = 0.0, noise_threshold: float = 0.0,
                                    multi_processing: bool = constants.ENABLE_MULTIPROCESSING_DEFAULT,
                                    activity_key: str = "concept:name", timestamp_key: str = "time:timestamp",
                                    case_id_key: str = "case:concept:name") -> ProcessTree:
    """
    Discovers a process tree using the inductive miner algorithm

    The basic idea of Inductive Miner is about detecting a 'cut' in the log (e.g. sequential cut, parallel cut, concurrent cut and loop cut) and then recur on sublogs, which were found applying the cut, until a base case is found. The Directly-Follows variant avoids the recursion on the sublogs but uses the Directly Follows graph.

    Inductive miner models usually make extensive use of hidden transitions, especially for skipping/looping on a portion on the model. Furthermore, each visible transition has a unique label (there are no transitions in the model that share the same label).

    :param log: event log / Pandas dataframe / typed DFG
    :param undesirable_log: event log / Pandas dataframe / typed DFG
    :param filter_ratio: filter ratio (default: 0.0)
    :param noise_threshold: noise threshold (default: 0.0)
    :param activity_key: attribute to be used for the activity
    :param multi_processing: boolean that enables/disables multiprocessing in inductive miner
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :rtype: ``ProcessTree``

    .. code-block:: python3

        import pm4py

        process_tree = pm4py.discover_process_tree_inductive(dataframe, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream, DFG]:
        raise Exception(
            "the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(
            log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)

    from imtd.algo.discovery.inductive import algorithm as inductive_miner
    parameters = get_properties(
        log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)
    parameters["original_undesirable_log"] = undesirable_log
    parameters["noise_threshold"] = noise_threshold
    parameters["filter_ratio"] = filter_ratio
    parameters["multiprocessing"] = multi_processing

    undesirable_dfg = dfg_inst.apply(undesirable_log, parameters=parameters)
    parameters["original_undesirable_dfg"] = undesirable_dfg

    variant = inductive_miner.Variants.IMf

    if isinstance(log, DFG):
        variant = inductive_miner.Variants.IMd

    return inductive_miner.apply(log, variant=variant, parameters=parameters)
