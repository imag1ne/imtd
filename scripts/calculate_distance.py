import time

import pm4py
from pm4py.objects.log.util.xes import DEFAULT_NAME_KEY, DEFAULT_RESOURCE_KEY

from imtd import distance_matrix as distance_matrix_par

# DESIRABLE_EVENT_LOG = '../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_1000.xes'
# UNDESIRABLE_EVENT_LOG = '../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_1000.xes'
DESIRABLE_EVENT_LOG = '../Dataset/BPI Challenge 2017_1_all/BPI Challenge 2017 desirable.xes'
UNDESIRABLE_EVENT_LOG = '../Dataset/BPI Challenge 2017_1_all/BPI Challenge 2017 undesirable.xes'

def main():
    event_log_plus = pm4py.read_xes(DESIRABLE_EVENT_LOG, return_legacy_log_object=True)
    event_log_minus = pm4py.read_xes(UNDESIRABLE_EVENT_LOG, return_legacy_log_object=True)

    start = time.time()
    m2 = distance_matrix_par(event_log_plus, event_log_minus)
    end = time.time()
    print("Rust implementation elapsed time: ", end - start)

def check_matrices(m1, m2):
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            assert m1[i][j] == m2[i][j]

if __name__ == '__main__':
    main()