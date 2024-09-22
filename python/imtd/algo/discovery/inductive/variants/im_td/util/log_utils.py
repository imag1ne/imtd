from collections.abc import Mapping

from pm4py.objects.log.obj import EventLog, Event
from pm4py.util import xes_constants


def case_id_trace_index_mapping(event_log: EventLog) -> Mapping[str, int]:
    case_id_trace_index_map = {}
    for trace_idx, trace in enumerate(event_log):
        case_id = trace.attributes['concept:name']
        case_id_trace_index_map[case_id] = trace_idx

    return case_id_trace_index_map


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
