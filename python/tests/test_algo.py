import pytest
from pm4py.objects.log.obj import Event, Trace, EventLog
from pm4py.util.xes_constants import DEFAULT_NAME_KEY

from imtd.algo.analysis.dfg_functions import edge_trace_mapping


def test_edge_trace_mapping():
    event_log = EventLog([])
    m = edge_trace_mapping(event_log)
    assert len(m) == 0

    trace = ['A', 'B', 'C']
    trace = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace])
    event_log = EventLog([trace])
    m = edge_trace_mapping(event_log)
    assert len(m) == 2
    assert m[('A', 'B')] == [0]
    assert m[('B', 'C')] == [0]

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['A', 'B', 'C']
    trace_3 = ['C', 'B', 'A']
    trace_4 = ['A', 'D', 'E', 'F']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_2])
    trace_3 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_3])
    trace_4 = Trace([Event({DEFAULT_NAME_KEY: activity}) for activity in trace_4])
    event_log = EventLog([trace_1, trace_2, trace_3, trace_4])
    m = edge_trace_mapping(event_log)
    assert len(m) == 7
    assert m[('A', 'B')] == [0, 1]
    assert m[('B', 'C')] == [0, 1]
    assert m[('C', 'B')] == [2]
    assert m[('B', 'A')] == [2]
    assert m[('A', 'D')] == [3]
    assert m[('D', 'E')] == [3]
    assert m[('E', 'F')] == [3]
