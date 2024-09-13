import pm4py
from pm4py.objects.log.obj import Event, Trace, EventLog
from pm4py.util.xes_constants import DEFAULT_NAME_KEY, DEFAULT_RESOURCE_KEY

WORKING_DIR = '../Dataset/Example Event Log/'
OUTPUT_FILENAME = 'event_log_example.xes'

def main():
    # event_log = gen_event_log_1()
    # pm4py.write_xes(event_log, WORKING_DIR + OUTPUT_FILENAME)

    # event_log = gen_event_log_1_un()
    # pm4py.write_xes(event_log, WORKING_DIR + 'event_log_example_un.xes')

    # event_log = gen_event_log_2()
    # pm4py.write_xes(event_log, WORKING_DIR + 'event_log_example_2.xes')

    event_log = gen_event_log_2_un()
    pm4py.write_xes(event_log, WORKING_DIR + 'event_log_example_2_un.xes')

def gen_event_log_1():
    trace_variant_1 = ['A', 'B', 'C', 'A', 'B', 'E', 'F']
    trace_variant_2 = ['A', 'B', 'F', 'E']
    trace_variant_3 = ['D', 'E', 'F']
    trace_variant_4 = ['D', 'F', 'E']
    trace_variant_5 = ['D', 'E', 'D', 'F']

    trace_variant_1 = gen_trace_from_activities(trace_variant_1)
    trace_variant_2 = gen_trace_from_activities(trace_variant_2)
    trace_variant_3 = gen_trace_from_activities(trace_variant_3)
    trace_variant_4 = gen_trace_from_activities(trace_variant_4)
    trace_variant_5 = gen_trace_from_activities(trace_variant_5)

    event_log = gen_event_log([(trace_variant_1, 50),
                               (trace_variant_2, 100),
                               (trace_variant_3, 100),
                               (trace_variant_4, 100),
                               (trace_variant_5, 50)])

    return event_log

def gen_event_log_1_un():
    trace_variant_1 = ['D', 'E', 'D', 'F', 'G']

    trace_variant_1 = gen_trace_from_activities(trace_variant_1)

    event_log = gen_event_log([(trace_variant_1, 50)])

    return event_log

def gen_event_log_2():
    trace_variant_1 = ['A', 'B', 'C', 'D']
    trace_variant_2 = ['A', 'C', 'B', 'D']

    trace_variant_1 = gen_trace_from_activities(trace_variant_1)
    trace_variant_2 = gen_trace_from_activities(trace_variant_2)

    event_log = gen_event_log([(trace_variant_1, 50),
                               (trace_variant_2, 30)])

    return event_log

def gen_event_log_2_un():
    trace_variant_1 = ['A', 'C', 'B', 'D', 'E']

    trace_variant_1 = gen_trace_from_activities(trace_variant_1)

    event_log = gen_event_log([(trace_variant_1, 30)])

    return event_log

def gen_trace_from_activities(activities: list[str]) -> Trace:
    return Trace([Event({DEFAULT_NAME_KEY: activity, DEFAULT_RESOURCE_KEY: 'User1'}) for activity in activities])

def gen_event_log(traces: list[tuple[Trace, int]]) -> EventLog:
    event_log = []
    for trace_variant, amount in traces:
        event_log += [trace_variant for _ in range(amount)]
    return EventLog(event_log)

if __name__ == '__main__':
    main()