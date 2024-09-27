import argparse
import csv
from pathlib import Path

import pm4py
import numpy as np

from imtd import distance_matrix

WORKING_DIR = '../Dataset/Example Event Log/'
DESIRABLE_EVENT_LOG = 'event_log_example_2.xes'
UNDESIRABLE_EVENT_LOG = 'event_log_example_2_un.xes'
# DESIRABLE_EVENT_LOG = 'BPI Challenge 2017 desirable.xes'
# UNDESIRABLE_EVENT_LOG = 'BPI Challenge 2017 undesirable.xes'
OUTPUT_DISTANCE_FILENAME = 'example_2_distance_matrix'
OUTPUT_SIMILARITY_FILENAME = 'example_2_similarity_matrix'


def parse_args():
    parser = argparse.ArgumentParser(prog='Distance matrix', description='Distance matrix')

    parser.add_argument('-p', '--desirable-log', type=Path, required=True)
    parser.add_argument('-m', '--undesirable-log', type=Path, required=True)
    parser.add_argument('-a', '--activity', type=float, required=False, default=0)
    parser.add_argument('-t', '--transition', type=float, required=False, default=0)
    parser.add_argument('-r', '--resource', type=float, required=False, default=0)
    parser.add_argument('-o', '--output', type=Path, default='output')

    return parser.parse_args()


def main():
    args = parse_args()
    desirable_log_path = args.desirable_log
    undesirable_log_path = args.undesirable_log
    activity_weight = args.activity
    transition_weight = args.transition
    resource_weight = args.resource

    Path(args.output).mkdir(parents=True, exist_ok=True)

    event_log_plus = pm4py.read_xes(str(desirable_log_path), return_legacy_log_object=True)
    event_log_minus = pm4py.read_xes(str(undesirable_log_path), return_legacy_log_object=True)

    print("Calculating distance matrix...")
    dm = np.array(
        distance_matrix(event_log_plus, event_log_minus,
                        [('activity', activity_weight), ('transition', transition_weight),
                         ('resource', resource_weight)]))

    dm_filename = args.output.joinpath('distance_matrix.csv')
    print("Saving distance matrix to file {}...".format(dm_filename))
    with open(dm_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dm)

    print("Calculating similarity matrix...")
    max_distance = np.max(dm)
    sm = 1 - dm / max_distance
    sm_filename = args.output.joinpath('similarity_matrix.csv')
    print("Saving similarity matrix to file {}...".format(sm_filename))
    with open(sm_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sm)


if __name__ == '__main__':
    main()
