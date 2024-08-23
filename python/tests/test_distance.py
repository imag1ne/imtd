import pytest
from pm4py.objects.log.obj import Event, Trace
from pm4py.util.xes_constants import DEFAULT_NAME_KEY, DEFAULT_RESOURCE_KEY

from imtd.algo.discovery.inductive.variants.im_bi.util.distance import activity_distance, transition_distance, resource_distance

def test_activity_distance():
    trace_1 = Trace()
    trace_2 = Trace()
    assert activity_distance(trace_1, trace_2) == 0

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['A', 'B', 'C']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_2])
    assert activity_distance(trace_1, trace_2) == 0

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['C', 'B', 'A']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_2])
    assert activity_distance(trace_1, trace_2) == 0

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['A', 'D', 'E', 'F']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_2])
    assert round(activity_distance(trace_1, trace_2), 2) == 2.24

def test_transition_distance():
    trace_1 = Trace()
    trace_2 = Trace()
    assert transition_distance(trace_1, trace_2) == 0

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['A', 'B', 'C']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_2])
    assert transition_distance(trace_1, trace_2) == 0

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['C', 'B', 'A']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_2])
    assert transition_distance(trace_1, trace_2) == 2

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['A', 'B', 'A', 'B', 'D']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_2])
    assert transition_distance(trace_1, trace_2) == 2

def test_resource_distance():
    trace_1 = Trace()
    trace_2 = Trace()
    assert resource_distance(trace_1, trace_2) == 0

    trace_1 = [('A', 'User_1'), ('B', 'User_2'), ('C', 'User_3')]
    trace_2 = [('A', 'User_1'), ('C', 'User_3'), ('B', 'User_2')]
    trace_1 = Trace(
        [Event({DEFAULT_NAME_KEY: activity, DEFAULT_RESOURCE_KEY: resource}) for activity, resource in trace_1])
    trace_2 = Trace(
        [Event({DEFAULT_NAME_KEY: activity, DEFAULT_RESOURCE_KEY: resource}) for activity, resource in trace_2])
    assert resource_distance(trace_1, trace_2) == 0

    trace_1 = [('A', 'User_1'), ('B', 'User_2'), ('C', 'User_3')]
    trace_2 = [('A', 'User_2'), ('C', 'User_3'), ('B', 'User_2')]
    trace_1 = Trace(
        [Event({DEFAULT_NAME_KEY: activity, DEFAULT_RESOURCE_KEY: resource}) for activity, resource in trace_1])
    trace_2 = Trace(
        [Event({DEFAULT_NAME_KEY: activity, DEFAULT_RESOURCE_KEY: resource}) for activity, resource in trace_2])
    assert round(resource_distance(trace_1, trace_2), 2) == 1.41

    trace_1 = [('A', 'User_1'), ('B', 'User_2'), ('C', 'User_3')]
    trace_2 = [('A', 'User_1'), ('B', 'User_2'), ('A', 'User_1'), ('B', 'User_2'), ('D', 'User_4')]
    trace_1 = Trace(
        [Event({DEFAULT_NAME_KEY: activity, DEFAULT_RESOURCE_KEY: resource}) for activity, resource in trace_1])
    trace_2 = Trace(
        [Event({DEFAULT_NAME_KEY: activity, DEFAULT_RESOURCE_KEY: resource}) for activity, resource in trace_2])
    assert resource_distance(trace_1, trace_2) == 2