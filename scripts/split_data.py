import pm4py
from pm4py.objects.log.util.xes import DEFAULT_NAME_KEY
from pm4py.utils import sample_cases

EVENT_LOG_PATH = "../../Dataset/BPI Challenge 2017_1_all/"
EVENT_LOG_NAME = "BPI Challenge 2017.xes"


def main():
    event_log = pm4py.read_xes(EVENT_LOG_PATH + EVENT_LOG_NAME)

    sizes = [100, 1000, 2000, 5000]
    samples = (sample_cases(event_log, size) for size in sizes)

    for sample, size in zip(samples, sizes):
        desirable_event_log, undesirable_event_log = split_event_log(sample)
        pm4py.write_xes(desirable_event_log, EVENT_LOG_PATH + "desirable_event_log_sample_" + str(size) + ".xes")
        pm4py.write_xes(undesirable_event_log, EVENT_LOG_PATH + "undesirable_event_log_sample_" + str(size) + ".xes")


def split_event_log(event_log):
    desirable_event_log = pm4py.filter_trace_attribute_values(event_log, DEFAULT_NAME_KEY, ['W_Call incomplete files'])
    undesirable_event_log = pm4py.filter_trace_attribute_values(event_log, DEFAULT_NAME_KEY,
                                                                ['W_Call incomplete files'], False)

    return (desirable_event_log, undesirable_event_log)


if __name__ == "__main__":
    main()
