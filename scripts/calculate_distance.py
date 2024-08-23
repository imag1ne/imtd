import time

import pm4py
from pm4py.objects.log.util.xes import DEFAULT_NAME_KEY, DEFAULT_RESOURCE_KEY

from imtd.algo.discovery.inductive.variants.im_bi.util.distance import distance_matrix
from imtd import distance_matrix as distance_matrix_par


def main():
    event_log_plus = '../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_1000.xes'
    event_log_minus = '../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_1000.xes'
    event_log_plus = pm4py.read_xes(event_log_plus, return_legacy_log_object=True)
    event_log_minus = pm4py.read_xes(event_log_minus, return_legacy_log_object=True)

    start = time.time()
    m1 = distance_matrix(event_log_plus, event_log_minus)
    end = time.time()
    print("Python implementation elapsed time: ", end - start)

    event_log_plus_extracted = []
    for trace in event_log_plus:
        new_trace = []
        for event in trace:
            new_trace.append((event[DEFAULT_NAME_KEY], event[DEFAULT_RESOURCE_KEY]))
        event_log_plus_extracted.append(new_trace)
    event_log_minus_extracted = []
    for trace in event_log_minus:
        new_trace = []
        for event in trace:
            new_trace.append((event[DEFAULT_NAME_KEY], event[DEFAULT_RESOURCE_KEY]))
        event_log_minus_extracted.append(new_trace)

    start = time.time()
    m2 = distance_matrix_par(event_log_plus_extracted, event_log_minus_extracted)
    end = time.time()
    print("Rust implementation elapsed time: ", end - start)

if __name__ == '__main__':
    main()