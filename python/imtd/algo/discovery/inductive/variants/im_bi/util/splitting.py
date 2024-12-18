"""
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
"""
from collections.abc import Iterable
from pm4py.objects.log import obj
from pm4py.util import xes_constants
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from pm4py.objects.log import obj as log_instance


def project(log, A, B):
    new_log = log_instance.EventLog()
    deep_log = log.__deepcopy__()
    for tr in range(0, len(deep_log)):
        trace = log_instance.Trace()
        event = log_instance.Event()
        event['concept:name'] = 'start'
        trace.append(event)
        for ev in deep_log[tr]:
            event = log_instance.Event()
            if ev['concept:name'] in A:
                event['concept:name'] = 'a'
            elif ev['concept:name'] in B:
                event['concept:name'] = 'b'
            else:
                event['concept:name'] = 'c'
            trace.append(event)
        event = log_instance.Event()
        event['concept:name'] = 'end'
        trace.append(event)
        new_log.append(trace)
    return new_log


def split(cut_type, cut, l, activity_key):
    case_id_key = xes_constants.DEFAULT_TRACEID_KEY
    LA = obj.EventLog()
    LB = obj.EventLog()

    if cut_type == 'seq':
        for trace in l:
            case_id = trace.attributes[case_id_key]
            cost = []
            for i in range(0, len(trace) + 1):
                cost.append(sum((x['concept:name'] in cut[1] for x in trace[0:i])) + sum(
                    (x['concept:name'] in cut[0] for x in trace[i:])))
            split_point = cost.index(min(cost))
            trace_A = new_trace(case_id, (x for x in trace[0:split_point] if x['concept:name'] in cut[0]))
            trace_B = new_trace(case_id, (x for x in trace[split_point:] if x['concept:name'] in cut[1]))
            LA.append(trace_A)
            LB.append(trace_B)

    if cut_type == 'exc':
        for tr in l:
            case_id = tr.attributes[case_id_key]
            if len(tr) == 0:
                T = new_trace(case_id)
                LB.append(T)
                continue
            A_count = 0
            B_count = 0
            for ev in tr:
                if ev[activity_key] in cut[0]:
                    A_count += 1
                elif ev[activity_key] in cut[1]:
                    B_count += 1
            if A_count >= B_count:
                T = new_trace(case_id, (ev for ev in tr if ev[activity_key] in cut[0]))
                LA.append(T)
            elif A_count < B_count:
                T = new_trace(case_id, (ev for ev in tr if ev[activity_key] in cut[1]))
                LB.append(T)

    if cut_type == 'par':
        for tr in l:
            case_id = tr.attributes[case_id_key]
            T1 = new_trace(case_id)
            T2 = new_trace(case_id)
            for ev in tr:
                if ev[activity_key] in cut[0]:
                    T1.append(ev)
                elif ev[activity_key] in cut[1]:
                    T2.append(ev)
            LA.append(T1)
            LB.append(T2)

    if cut_type == 'loop':
        counter = 0
        for tr in l:
            case_id = tr.attributes[case_id_key]
            flagA = False
            flagB = False
            if len(tr) == 0:
                T = new_trace(case_id)
                LA.append(T)
                continue
            if tr[0][activity_key] in cut[0]:
                flagA = True
            elif tr[0][activity_key] in cut[1]:
                flagB = True
                T = new_trace(case_id)
                LA.append(T)

            T = new_trace(case_id)
            for ind, ev in enumerate(tr):
                if flagA:
                    T.append(ev)
                    if ind != len(tr) - 1:
                        if tr[ind + 1][activity_key] in cut[1]:
                            flagA = False
                            flagB = True
                            # T.attributes[case_id_key] = counter
                            LA.append(T)
                            # counter += 1
                            T = new_trace(case_id)
                    elif ind == len(tr) - 1:
                        # T.attributes[case_id_key] = counter
                        LA.append(T)
                        counter += 1
                        T = new_trace(case_id)
                elif flagB:
                    T.append(ev)
                    if ind != len(tr) - 1:
                        if tr[ind + 1][activity_key] in cut[0]:
                            flagA = True
                            flagB = False
                            # T.attributes[case_id_key] = counter
                            LB.append(T)
                            counter += 1
                            T = new_trace(case_id)
                    elif ind == len(tr) - 1:
                        # T.attributes[case_id_key] = counter
                        LB.append(T)
                        # counter += 1
                        T = new_trace(case_id)
                        LA.append(T)

    if cut_type == 'loop1':
        for tr in l:
            case_id = tr.attributes[case_id_key]
            if len(tr) == 0:
                T = new_trace(case_id)
                LA.append(T)
            elif len(tr) == 1:
                T = new_trace(case_id)
                T.append(tr[0])
                LA.append(T)
            else:
                T = new_trace(case_id)
                T.append(tr[0])
                LA.append(T)
                for x in range(1, len(tr)):
                    T = new_trace(case_id)
                    LB.append(T)

    if cut_type == 'loop_tau':
        st_acts = cut[0]
        en_acts = cut[1]
        for tr in l:
            case_id = tr.attributes[case_id_key]
            if len(tr) == 0:
                T = new_trace(case_id)
                LA.append(T)
            else:
                T = new_trace(case_id)
                for i, ev in enumerate(tr):
                    T.append(ev)
                    if i < (len(tr) - 1):
                        if (tr[i][activity_key] in en_acts) and (tr[i + 1][activity_key] in st_acts):
                            LA.append(T)
                            T = new_trace(case_id)
                            LB.append(new_trace(case_id))
                    else:
                        LA.append(T)

    return LA, LB  # new_logs is a list that contains logs


def filter_trace_on_cut_partition(trace, partition, activity_key):
    filtered_trace = obj.Trace()
    for event in trace:
        if event[activity_key] in partition:
            filtered_trace.append(event)
    return filtered_trace


def find_split_point(trace, cut_partition, start, ignore, activity_key):
    possibly_best_before_first_activity = False
    least_cost = start
    position_with_least_cost = start
    cost = float(0)
    i = start
    while i < len(trace):
        if trace[i][activity_key] in cut_partition:
            cost = cost - 1
        elif trace[i][activity_key] not in ignore:
            # use bool variable for the case, that the best split is before the first activity
            if i == 0:
                possibly_best_before_first_activity = True
            cost = cost + 1
        if cost <= least_cost:
            least_cost = cost
            position_with_least_cost = i + 1
        i += 1
    if possibly_best_before_first_activity and position_with_least_cost == 1:
        position_with_least_cost = 0
    return position_with_least_cost


def cut_trace_between_two_points(trace, point_a, point_b):
    cutted_trace = obj.Trace()
    # we have to use <= although in the paper the intervall is [) because our index starts at 0
    while point_a < point_b:
        cutted_trace.append(trace[point_a])
        point_a += 1

    return cutted_trace


def split_xor_infrequent(cut, l, activity_key):
    # TODO think of empty logs
    # creating the empty L_1,...,L_n from the second code-line on page 205
    n = len(cut)
    new_logs = [obj.EventLog() for i in range(0, n)]
    for trace in l:  # for all traces
        number_of_events_in_trace = 0
        index_of_cut_partition = 0
        i = 0
        # use i as index here so that we can write in L_i
        for i in range(0, len(cut)):  # for all cut partitions
            temp_counter = 0
            for event in trace:  # for all events in current trace
                if event[activity_key] in cut[i]:  # count amount of events from trace in partition
                    temp_counter += 1
            if temp_counter > number_of_events_in_trace:
                number_of_events_in_trace = temp_counter
                index_of_cut_partition = i
        filtered_trace = filter_trace_on_cut_partition(trace, cut[index_of_cut_partition], activity_key)
        new_logs[index_of_cut_partition].append(filtered_trace)
    return new_logs


def split_sequence_infrequent(cut, l, activity_key):
    # write L_1,...,L_n like in second line of code on page 206
    n = len(cut)
    new_logs = [obj.EventLog() for j in range(0, n)]
    ignore = []
    split_points_list = [0] * len(l)
    for i in range(0, n):
        split_point = 0
        # write our ignore list with all elements from past cut partitions
        if i != 0:
            for element in cut[i - 1]:
                ignore.append(element)
        for j in range(len(l)):
            trace = l[j]
            new_split_point = find_split_point(trace, cut[i], split_points_list[j], ignore, activity_key)
            cutted_trace = cut_trace_between_two_points(trace, split_points_list[j], new_split_point)
            filtered_trace = filter_trace_on_cut_partition(cutted_trace, cut[i], activity_key)
            new_logs[i].append(filtered_trace)
            split_points_list[j] = new_split_point
    return new_logs


def split_loop_infrequent(cut, l, activity_key):
    n = len(cut)
    new_logs = [obj.EventLog() for i in range(0, n)]
    for trace in l:
        s = cut[0]
        st = obj.Trace()
        for act in trace:
            if act in s:
                st.insert(act)
            else:
                j = 0
                for j in range(0, len(cut)):
                    if cut[j] == s:
                        break
                new_logs[j].append(st)
                st = obj.Trace()
                for partition in cut:
                    if act[activity_key] in partition:
                        s.append(partition)
        # L_j <- L_j + [st] with sigma_j = s
        j = 0
        for j in range(0, len(cut)):
            if cut[j] == s:
                break
        new_logs[j].append(st)
        if s != cut[0]:
            new_logs[0].append(obj.EventLog())

    return new_logs


def split_parallel_infrequent(cut, l, activity_key):
    new_logs = []
    for c in cut:
        lo = obj.EventLog()
        for trace in l:
            new_trace = obj.Trace()
            for event in trace:
                if event[activity_key] in c:
                    new_trace.append(event)
            lo.append(new_trace)
        new_logs.append(lo)
    return new_logs


def new_trace(case_id: str, iterable: Iterable[obj.Event] = ()) -> obj.Trace:
    trace = obj.Trace(iterable)
    trace.attributes['concept:name'] = case_id

    return trace
