import pytest
from pm4py.objects.log.obj import Event, Trace, EventLog
from pm4py.util.xes_constants import DEFAULT_NAME_KEY

from imtd.algo.analysis.dfg_functions import edge_trace_mapping, case_id_trace_index_mapping, generate_nx_graph_from_event_log


CASE_ID_KEY = 'case:concept:name'

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

def test_case_id_trace_index_mapping():
    event_log = EventLog([])
    m = case_id_trace_index_mapping(event_log)
    assert len(m) == 0

    trace = ['A', 'B', 'C']
    trace = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '0'}) for activity in trace])
    event_log = EventLog([trace])
    m = case_id_trace_index_mapping(event_log)
    assert len(m) == 1
    assert m['0'] == 0

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['A', 'B', 'C']
    trace_3 = ['C', 'B', 'A']
    trace_4 = ['A', 'D', 'E', 'F']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '0'}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '1'}) for activity in trace_2])
    trace_3 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '3'}) for activity in trace_3])
    trace_4 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '2'}) for activity in trace_4])
    event_log = EventLog([trace_1, trace_2, trace_3, trace_4])
    m = case_id_trace_index_mapping(event_log)
    assert len(m) == 4
    assert m['0'] == 0
    assert m['1'] == 1
    assert m['2'] == 3
    assert m['3'] == 2


def test_generate_nx_graph_from_event_log():
    event_log = EventLog([])
    g = generate_nx_graph_from_event_log(event_log)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0

    trace = ['A', 'B', 'C']
    trace = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '0'}) for activity in trace])
    event_log = EventLog([trace])
    g = generate_nx_graph_from_event_log(event_log)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    assert g['A']['B']['case_id_set'] == {'0'}
    assert g['B']['C']['case_id_set'] == {'0'}

    trace_1 = ['A', 'B', 'C']
    trace_2 = ['A', 'B', 'C']
    trace_3 = ['C', 'B', 'A']
    trace_4 = ['A', 'D', 'E', 'F']
    trace_1 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '0'}) for activity in trace_1])
    trace_2 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '1'}) for activity in trace_2])
    trace_3 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '3'}) for activity in trace_3])
    trace_4 = Trace([Event({DEFAULT_NAME_KEY: activity, CASE_ID_KEY: '2'}) for activity in trace_4])
    event_log = EventLog([trace_1, trace_2, trace_3, trace_4])
    g = generate_nx_graph_from_event_log(event_log)
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 7
    assert g['A']['B']['case_id_set'] == {'0', '1'}
    assert g['B']['C']['case_id_set'] == {'0', '1'}
    assert g['C']['B']['case_id_set'] == {'3'}
    assert g['B']['A']['case_id_set'] == {'3'}
    assert g['A']['D']['case_id_set'] == {'2'}
    assert g['D']['E']['case_id_set'] == {'2'}
    assert g['E']['F']['case_id_set'] == {'2'}

