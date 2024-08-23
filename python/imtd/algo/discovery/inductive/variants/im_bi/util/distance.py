from collections import Counter
from multiprocessing import Pool

import numpy as np
from numpy import floating
from pm4py.objects.log.obj import Trace, EventLog
from pm4py.objects.log.util.xes import DEFAULT_NAME_KEY, DEFAULT_RESOURCE_KEY

def distance_matrix_par(event_log1: EventLog, event_log2: EventLog, activity_key=DEFAULT_NAME_KEY,
                    resource_key=DEFAULT_RESOURCE_KEY) -> np.ndarray:
    """Calculate the distance matrix between two event logs based on the frequency of activities, transitions and resources in each trace

    :param event_log1: One of the event logs to compare
    :param event_log2: Another event log to compare
    :param activity_key: the key to use to extract the activity name from the events
    :param resource_key: the key to use to extract the resource name from the events
    :return: the distance matrix between the two event logs
    """
    m = len(event_log1)
    n = len(event_log2)
    matrix = np.zeros((m, n))
    args = [(matrix, i, j, trace1, trace2, activity_key, resource_key) for i, trace1 in enumerate(event_log1) for j, trace2 in enumerate(event_log2)]
    with Pool(8) as p:
        p.map(trace_distance2, args)

    return matrix

def distance_matrix(event_log1: EventLog, event_log2: EventLog, activity_key=DEFAULT_NAME_KEY,
                    resource_key=DEFAULT_RESOURCE_KEY) -> np.ndarray:
    """Calculate the distance matrix between two event logs based on the frequency of activities, transitions and resources in each trace

    :param event_log1: One of the event logs to compare
    :param event_log2: Another event log to compare
    :param activity_key: the key to use to extract the activity name from the events
    :param resource_key: the key to use to extract the resource name from the events
    :return: the distance matrix between the two event logs
    """
    m = len(event_log1)
    n = len(event_log2)
    matrix = np.zeros((m, n))

    for i, trace1 in enumerate(event_log1):
        for j, trace2 in enumerate(event_log2):
            matrix[i, j] = trace_distance(trace1, trace2, activity_key, resource_key)

    return matrix

def trace_distance2(result, i, j, trace1: Trace, trace2: Trace, activity_key=DEFAULT_NAME_KEY,
                   resource_key=DEFAULT_RESOURCE_KEY) -> float | floating:
    """Calculate the distance between two traces based on the frequency of activities, transitions and resources in each trace
    :param result: the result of the distance calculation
    :param i: the index of the first trace
    :param j: the index of the second trace
    :param trace1: One of the traces to compare
    :param trace2: Another trace to compare
    :param activity_key: the key to use to extract the activity name from the events
    :param resource_key: the key to use to extract the resource name from the events
    :return: the distance between the two traces
    """
    activity_dist = activity_distance(trace1, trace2, activity_key)
    transition_dist = transition_distance(trace1, trace2, activity_key)
    resource_dist = resource_distance(trace1, trace2, activity_key, resource_key)

    distance = activity_dist + transition_dist + resource_dist

    result[i, j] = distance

def trace_distance(trace1: Trace, trace2: Trace, activity_key=DEFAULT_NAME_KEY,
                   resource_key=DEFAULT_RESOURCE_KEY) -> float | floating:
    """Calculate the distance between two traces based on the frequency of activities, transitions and resources in each trace
    :param trace1: One of the traces to compare
    :param trace2: Another trace to compare
    :param activity_key: the key to use to extract the activity name from the events
    :param resource_key: the key to use to extract the resource name from the events
    :return: the distance between the two traces
    """
    activity_dist = activity_distance(trace1, trace2, activity_key)
    transition_dist = transition_distance(trace1, trace2, activity_key)
    resource_dist = resource_distance(trace1, trace2, activity_key, resource_key)

    distance = activity_dist + transition_dist + resource_dist

    return distance


def activity_distance(trace1: Trace, trace2: Trace, activity_key=DEFAULT_NAME_KEY) -> float | floating:
    """
    Calculate the distance between two traces based on the frequency of activities in each trace
    :param trace1: One of the traces to compare
    :param trace2: Another trace to compare
    :param activity_key: the key to use to extract the activity name from the events
    :return: the distance between the two traces
    """
    activity_count1 = Counter([event[activity_key] for event in trace1])
    activity_count2 = Counter([event[activity_key] for event in trace2])

    distance = distance_from_profiles(activity_count1, activity_count2)

    return distance


def transition_distance(trace1: Trace, trace2: Trace, activity_key=DEFAULT_NAME_KEY) -> float | floating:
    """Calculate the distance between two traces based on the frequency of transitions in each trace
    :param trace1: One of the traces to compare
    :param trace2: Another trace to compare
    :param activity_key: the key to use to extract the activity name from the events
    :return: the distance between the two traces
    """
    transition_count1 = Counter([(trace1[i][activity_key], trace1[i + 1][activity_key])
                                 for i in range(len(trace1) - 1)])
    transition_count2 = Counter([(trace2[i][activity_key], trace2[i + 1][activity_key])
                                 for i in range(len(trace2) - 1)])

    distance = distance_from_profiles(transition_count1, transition_count2)

    return distance


def resource_distance(trace1: Trace, trace2: Trace, activity_key=DEFAULT_NAME_KEY,
                      resource_key=DEFAULT_RESOURCE_KEY) -> float | floating:
    """Calculate the distance between two traces based on the frequency of resources in each trace
    :param trace1: One of the traces to compare
    :param trace2: Another trace to compare
    :param activity_key: the key to use to extract the activity name from the events
    :param resource_key: the key to use to extract the resource name from the events
    :return: the distance between the two traces
    """
    resource_count1 = Counter([(event[activity_key], event[resource_key]) for event in trace1])
    resource_count2 = Counter([(event[activity_key], event[resource_key]) for event in trace2])

    distance = distance_from_profiles(resource_count1, resource_count2)

    return distance


def distance_from_profiles(profile1: Counter, profile2: Counter) -> float | floating:
    """Calculate the distance between two profiles based on the frequency of elements in each profile
    :param profile1: One of the profile to compare
    :param profile2: Another profile to compare
    :return: the distance between the two profiles
    """
    all_keys = unique_list_from_counters(profile1, profile2)

    vector1 = np.array([profile1.get(key, 0) for key in all_keys])
    vector2 = np.array([profile2.get(key, 0) for key in all_keys])

    distance = np.linalg.norm(vector1 - vector2)

    return distance


def unique_list_from_counters(counter1: Counter, counter2: Counter) -> list:
    """Return a list of unique elements from two counters
    :param counter1: One of the counter
    :param counter2: Another counter
    :return: a list of unique elements from the two counters
    """
    return list(set(counter1.keys()).union(set(counter2.keys())))
