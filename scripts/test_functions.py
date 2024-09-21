import time
from pprint import pprint

import pm4py

from imtd import edge_case_id_mapping as edge_case_id_mapping_1
from imtd import edge_case_id_mapping_2
from imtd.algo.analysis.dfg_functions import edge_case_id_mapping

WORKING_DIR = '../Dataset/BPI Challenge 2017_1_all/'
EVENT_LOG = 'BPI Challenge 2017.xes'
# EVENT_LOG = 'desirable_event_log_sample_100.xes'

def main():
    event_log = pm4py.read_xes(WORKING_DIR + EVENT_LOG, return_legacy_log_object=True)

    start = time.time()
    m1 = edge_case_id_mapping(event_log)
    end = time.time()
    print("Python implementation elapsed time: ", end - start)

    start = time.time()
    m2 = edge_case_id_mapping_1(event_log)
    end = time.time()
    print("Rust implementation (1) elapsed time: ", end - start)

    start = time.time()
    m3 = edge_case_id_mapping_2(event_log)
    end = time.time()
    print("Rust implementation (2) elapsed time: ", end - start)

    check_maps(m1, m2)
    check_maps(m1, m3)

def check_maps(m1, m2):
    m1_len = len(m1)
    m2_len = len(m2)
    assert m1_len == m2_len

    for k, v in m1.items():
        assert v == m2[k]


if __name__ == '__main__':
    main()