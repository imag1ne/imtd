import time
import csv

import pm4py
import numpy as np

from imtd import distance_matrix as distance_matrix_par

WORKING_DIR = '../Dataset/Example Event Log/'
DESIRABLE_EVENT_LOG = 'event_log_example_2.xes'
UNDESIRABLE_EVENT_LOG = 'event_log_example_2_un.xes'
# DESIRABLE_EVENT_LOG = 'BPI Challenge 2017 desirable.xes'
# UNDESIRABLE_EVENT_LOG = 'BPI Challenge 2017 undesirable.xes'
OUTPUT_DISTANCE_FILENAME = 'example_2_distance_matrix'
OUTPUT_SIMILARITY_FILENAME = 'example_2_similarity_matrix'

def main():
    event_log_plus = pm4py.read_xes(WORKING_DIR + DESIRABLE_EVENT_LOG, return_legacy_log_object=True)
    event_log_minus = pm4py.read_xes(WORKING_DIR + UNDESIRABLE_EVENT_LOG, return_legacy_log_object=True)

    start = time.time()
    dm = np.array(distance_matrix_par(event_log_plus, event_log_minus))
    end = time.time()
    print("Rust implementation elapsed time: ", end - start)

    dm_filename = OUTPUT_DISTANCE_FILENAME + '_100' + '.csv'
    print("Saving distance matrix to file {}...".format(dm_filename))
    with open(WORKING_DIR + dm_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dm)

    max_distance = np.max(dm)
    sm = 1 - dm / max_distance
    sm_filename = OUTPUT_SIMILARITY_FILENAME + '_100' + '.csv'
    print("Saving similarity matrix to file {}...".format(sm_filename))
    with open(WORKING_DIR + sm_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sm)

def check_matrices(m1, m2):
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            assert m1[i][j] == m2[i][j]

if __name__ == '__main__':
    main()