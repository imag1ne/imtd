import argparse
import csv

import pm4py
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from pathlib import Path

from typing import List, Dict, Any, Union, Optional, Tuple, Set

from pm4py.objects.log.obj import EventLog, Trace, Event, EventStream
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.util import xes_constants, constants
from pm4py.utils import get_properties, __event_log_deprecation_warning
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
from pm4py.util import exec_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.algo.conformance.alignments.petri_net.algorithm import Parameters, DEFAULT_VARIANT, apply_trace
from pm4py.util.constants import CASE_CONCEPT_NAME
import pkgutil
from copy import copy
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(prog='evaluate', description='evaluate discovered model')
    parser.add_argument('-d', '--process-model', type=str, required=True)
    parser.add_argument('-p', '--desirable-log', type=str, required=True)
    parser.add_argument('-m', '--undesirable-log', type=str, required=True)
    parser.add_argument('-x', '--parallel', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-b', '--batch', type=int, required=False, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    process_model = args.process_model
    desirable_log = args.desirable_log
    undesirable_log = args.undesirable_log
    output = Path(args.process_model).parent
    model_filename = Path(args.process_model).stem

    # load the event logs
    log_p = pm4py.read_xes(desirable_log, return_legacy_log_object=True)
    log_m = pm4py.read_xes(undesirable_log, return_legacy_log_object=True)
    net, initial_marking, final_marking = pm4py.read_pnml(process_model)

    mes = apply_petri(log_p, log_m, net, initial_marking, final_marking, multi_processing=args.parallel,
                      batch=args.batch)
    save_measurements(mes, output, model_filename)

    print("\U00002705Completed evaluating the models.")


def save_measurements(measurements, output: Path, filename: str):
    result = [
        ('acc', measurements['acc']),
        ('F1', measurements['F1']),
        ('precision', measurements['precision']),
        ('fitP', measurements['fitP']),
        ('fitM', measurements['fitM']),
        ('acc_perf', measurements['acc_perf']),
        ('F1_perf', measurements['F1_perf']),
        ('acc_ML', measurements['acc_ML']),
        ('prc_ML', measurements['prc_ML']),
        ('rec_ML', measurements['rec_ML']),
    ]
    mes_filename = output.joinpath(filename + '.csv')
    with open(mes_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)


def apply_petri(LPlus, LMinus, net, i_m, i_f, multi_processing=False, batch=1):
    measures = {}

    alp = conformance_diagnostics_alignments(LPlus, net, i_m, i_f, multi_processing=multi_processing, batch=batch)
    fp_inf = replay_fitness.evaluate(alp, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fp = fp_inf['averageFitness']
    fp_pef = fp_inf['percentage_of_fitting_traces'] / 100
    # roc_data = [('p', x['fitness']) for x in alp]

    ################################################################################
    prec_Plus = pm4py.precision_alignments(LPlus, net, i_m, i_f, multi_processing=multi_processing)
    ################################################################################

    alm = conformance_diagnostics_alignments(LMinus, net, i_m, i_f, multi_processing=multi_processing, batch=batch)
    fm_inf = replay_fitness.evaluate(alm, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fm = fm_inf['averageFitness']
    fm_pef = fm_inf['percentage_of_fitting_traces'] / 100
    # roc_data += [('n', x['fitness']) for x in alm]

    measures['acc'] = round(fp - fm, 2)
    measures['F1'] = round(2 * ((fp * (1 - fm)) / (fp + (1 - fm))), 2)
    measures['precision'] = round(prec_Plus, 2)
    measures['fitP'] = round(fp, 2)
    measures['fitM'] = round(fm, 2)
    measures['acc_perf'] = round(fp_pef - fm_pef, 2)
    measures['F1_perf'] = round(2 * ((fp_pef * (1 - fm_pef)) / (fp_pef + (1 - fm_pef))), 2)

    TP = fp_pef * len(LPlus)
    FP = fm_pef * len(LMinus)
    FN = (1 - fp_pef) * len(LPlus)
    TN = (1 - fm_pef) * len(LMinus)
    measures['acc_ML'] = (TP + TN) / (TP + TN + FP + FN)
    if (TP + FP) != 0:
        measures['prc_ML'] = TP / (TP + FP)
    else:
        measures['prc_ML'] = 'ignore'

    if (TP + FN) != 0:
        measures['rec_ML'] = TP / (TP + FN)
    else:
        measures['rec_ML'] = 'ignore'
    # measures['roc_data'] = roc_data

    return measures


def conformance_diagnostics_token_based_replay(log: Union[EventLog, pd.DataFrame], petri_net: PetriNet,
                                               initial_marking: Marking,
                                               final_marking: Marking, activity_key: str = "concept:name",
                                               timestamp_key: str = "time:timestamp",
                                               case_id_key: str = "case:concept:name") -> List[Dict[str, Any]]:
    """
    Apply token-based replay for conformance checking analysis.
    The methods return the full token-based-replay diagnostics.

    Token-based replay matches a trace and a Petri net model, starting from the initial place, in order to discover which transitions are executed and in which places we have remaining or missing tokens for the given process instance. Token-based replay is useful for Conformance Checking: indeed, a trace is fitting according to the model if, during its execution, the transitions can be fired without the need to insert any missing token. If the reaching of the final marking is imposed, then a trace is fitting if it reaches the final marking without any missing or remaining tokens.

    In PM4Py there is an implementation of a token replayer that is able to go across hidden transitions (calculating shortest paths between places) and can be used with any Petri net model with unique visible transitions and hidden transitions. When a visible transition needs to be fired and not all places in the preset are provided with the correct number of tokens, starting from the current marking it is checked if for some place there is a sequence of hidden transitions that could be fired in order to enable the visible transition. The hidden transitions are then fired and a marking that permits to enable the visible transition is reached.
    The approach is described in:
    Berti, Alessandro, and Wil MP van der Aalst. "Reviving Token-based Replay: Increasing Speed While Improving Diagnostics." ATAED@ Petri Nets/ACSD. 2019.

    The output of the token-based replay, stored in the variable replayed_traces, contains for each trace of the log:

    - trace_is_fit: boolean value (True/False) that is true when the trace is according to the model.
    - activated_transitions: list of transitions activated in the model by the token-based replay.
    - reached_marking: marking reached at the end of the replay.
    - missing_tokens: number of missing tokens.
    - consumed_tokens: number of consumed tokens.
    - remaining_tokens: number of remaining tokens.
    - produced_tokens: number of produced tokens.

    :param log: event log
    :param petri_net: petri net
    :param initial_marking: initial marking
    :param final_marking: final marking
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :rtype: ``List[Dict[str, Any]]``

    .. code-block:: python3

        import pm4py

        net, im, fm = pm4py.discover_petri_net_inductive(dataframe, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
        tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(dataframe, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]: raise Exception(
        "the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(log, activity_key=activity_key, timestamp_key=timestamp_key,
                                       case_id_key=case_id_key)

    properties = get_properties(log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)

    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    return token_replay.apply(log, petri_net, initial_marking, final_marking, parameters=properties)


def conformance_diagnostics_alignments(log: Union[EventLog, pd.DataFrame], *args,
                                       multi_processing: bool = constants.ENABLE_MULTIPROCESSING_DEFAULT,
                                       activity_key: str = "concept:name", timestamp_key: str = "time:timestamp",
                                       case_id_key: str = "case:concept:name", batch=1) -> List[Dict[str, Any]]:
    """
    Apply the alignments algorithm between a log and a process model.
    The methods return the full alignment diagnostics.

    Alignment-based replay aims to find one of the best alignment between the trace and the model. For each trace, the output of an alignment is a list of couples where the first element is an event (of the trace) or » and the second element is a transition (of the model) or ». For each couple, the following classification could be provided:

    - Sync move: the classification of the event corresponds to the transition label; in this case, both the trace and the model advance in the same way during the replay.
    - Move on log: for couples where the second element is », it corresponds to a replay move in the trace that is not mimicked in the model. This kind of move is unfit and signal a deviation between the trace and the model.
    - Move on model: for couples where the first element is », it corresponds to a replay move in the model that is not mimicked in the trace. For moves on model, we can have the following distinction:
        * Moves on model involving hidden transitions: in this case, even if it is not a sync move, the move is fit.
        * Moves on model not involving hidden transitions: in this case, the move is unfit and signals a deviation between the trace and the model.

    With each trace, a dictionary containing among the others the following information is associated:

    alignment: contains the alignment (sync moves, moves on log, moves on model)
    cost: contains the cost of the alignment according to the provided cost function
    fitness: is equal to 1 if the trace is perfectly fitting.

    :param log: event log
    :param args: specification of the process model
    :param multi_processing: boolean value that enables the multiprocessing
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :rtype: ``List[Dict[str, Any]]``

    .. code-block:: python3

        import pm4py

        net, im, fm = pm4py.discover_petri_net_inductive(dataframe, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
        alignments_diagnostics = pm4py.conformance_diagnostics_alignments(dataframe, net, im, fm, activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]: raise Exception(
        "the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(log, activity_key=activity_key, timestamp_key=timestamp_key,
                                       case_id_key=case_id_key)

    properties = get_properties(log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)
    # properties['cores'] = 36

    if len(args) == 3:
        if type(args[0]) is PetriNet:
            # Petri net alignments
            from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
            if multi_processing:
                return apply_multiprocessing(log, args[0], args[1], args[2], parameters=properties, batch=batch)
            else:
                return alignments.apply(log, args[0], args[1], args[2], parameters=properties)
        elif isinstance(args[0], dict):
            # DFG alignments
            from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
            return dfg_alignment.apply(log, args[0], args[1], args[2], parameters=properties)
    elif len(args) == 1:
        if type(args[0]) is ProcessTree:
            # process tree alignments
            from pm4py.algo.conformance.alignments.process_tree.variants import search_graph_pt
            if multi_processing:
                return search_graph_pt.apply_multiprocessing(log, args[0], parameters=properties)
            else:
                return search_graph_pt.apply(log, args[0], parameters=properties)
    # try to convert to Petri net
    import pm4py
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
    net, im, fm = pm4py.convert_to_petri_net(*args)
    if multi_processing:
        return alignments.apply_multiprocessing(log, net, im, fm, parameters=properties)
    else:
        return alignments.apply(log, net, im, fm, parameters=properties)


def apply_multiprocessing(log, petri_net, initial_marking, final_marking, parameters=None, variant=DEFAULT_VARIANT,
                          batch=1):
    """
    Applies the alignments using a process pool (multiprocessing)

    Parameters
    ---------------
    log
        Event log
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm

    Returns
    ----------------
    aligned_traces
        Alignments
    """
    if parameters is None:
        parameters = {}

    import multiprocessing

    num_cores = exec_utils.get_param_value(Parameters.CORES, parameters, multiprocessing.cpu_count() - 2)

    best_worst_cost = __get_best_worst_cost(petri_net, initial_marking, final_marking, variant, parameters)
    variants_idxs, one_tr_per_var = __get_variants_structure(log, parameters)
    parameters[Parameters.BEST_WORST_COST_INTERNAL] = best_worst_cost

    all_alignments = []
    batch_num = batch
    batch_size = len(one_tr_per_var) // batch_num

    from concurrent.futures import ProcessPoolExecutor
    batch_n = 0
    for batch_cur in range(0, len(one_tr_per_var), batch_size):
        one_tr_per_var_batch = one_tr_per_var[batch_cur:batch_cur + batch_size]
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(apply_trace, trace, petri_net, initial_marking, final_marking, parameters) for
                       trace in one_tr_per_var_batch]
            progress = __get_progress_bar(batch_n, len(one_tr_per_var_batch), parameters)
            if progress is not None:
                alignments_ready = 0
                while alignments_ready != len(futures):
                    current = 0
                    for future in futures:
                        current = current + 1 if future.done() else current
                    if current > alignments_ready:
                        for i in range(0, current - alignments_ready):
                            progress.update()
                    alignments_ready = current
            for future in futures:
                all_alignments.append(future.result())
            __close_progress_bar(progress)
        batch_n += 1

    alignments = __form_alignments(variants_idxs, all_alignments)

    return alignments


def __get_best_worst_cost(petri_net, initial_marking, final_marking, variant, parameters):
    parameters_best_worst = copy(parameters)

    best_worst_cost = exec_utils.get_variant(variant).get_best_worst_cost(petri_net, initial_marking, final_marking,
                                                                          parameters=parameters_best_worst)

    return best_worst_cost


def __get_variants_structure(log, parameters):
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)

    variants_idxs = {}
    one_tr_per_var = []

    if type(log) is pd.DataFrame:
        case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME)
        traces = list(log.groupby(case_id_key)[activity_key].apply(tuple))
        for idx, trace in enumerate(traces):
            if trace not in variants_idxs:
                variants_idxs[trace] = [idx]
                case = Trace()
                for act in trace:
                    case.append(Event({activity_key: act}))
                one_tr_per_var.append(case)
            else:
                variants_idxs[trace].append(idx)
    else:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
        for idx, case in enumerate(log):
            trace = tuple(x[activity_key] for x in case)
            if trace not in variants_idxs:
                variants_idxs[trace] = [idx]
                one_tr_per_var.append(case)
            else:
                variants_idxs[trace].append(idx)

    return variants_idxs, one_tr_per_var


def __get_progress_bar(batch_n, num_variants, parameters):
    show_progress_bar = exec_utils.get_param_value(Parameters.SHOW_PROGRESS_BAR, parameters, True)
    progress = None
    if pkgutil.find_loader("tqdm") and show_progress_bar and num_variants > 1:
        from tqdm.auto import tqdm
        desc = "aligning log[batch {}], completed variants :: ".format(batch_n)
        progress = tqdm(total=num_variants, desc=desc)
    return progress


def __form_alignments(variants_idxs, all_alignments):
    al_idx = {}
    for index_variant, variant in enumerate(variants_idxs):
        for trace_idx in variants_idxs[variant]:
            al_idx[trace_idx] = all_alignments[index_variant]

    alignments = []
    for i in range(len(al_idx)):
        alignments.append(al_idx[i])

    return alignments


def __close_progress_bar(progress):
    if progress is not None:
        progress.close()
    del progress


def safe_apply_trace(trace, petri_net, initial_marking, final_marking, parameters):
    try:
        return apply_trace(trace, petri_net, initial_marking, final_marking, parameters)
    except Exception as e:
        return {'exception': str(e)}


if __name__ == "__main__":
    main()
