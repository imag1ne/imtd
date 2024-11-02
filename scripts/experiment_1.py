import argparse
import csv
import time

import pm4py
import numpy as np
from pathlib import Path

from imtd import discover_petri_net_inductive, discover_petri_net_inductive_bi, discover_petri_net_inductive_td, \
    Optimzation_Goals


def parse_args():
    parser = argparse.ArgumentParser(prog='experiment', description='experiment')
    parser.add_argument('-v', '--variant', type=str, required=True)
    parser.add_argument('-t', '--noise-threshold', type=str, required=False)
    parser.add_argument('-s', '--support', type=str, required=False)
    parser.add_argument('-r', '--ratio', type=str, required=False)
    parser.add_argument('-f', '--filter-ratio', type=str, required=False)
    parser.add_argument('-p', '--desirable-log', type=str, required=True)
    parser.add_argument('-m', '--undesirable-log', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='output')

    return parser.parse_args()


def main():
    args = parse_args()
    variant = args.variant
    desirable_log = args.desirable_log
    undesirable_log = args.undesirable_log
    output = Path(args.output)

    output.mkdir(parents=True, exist_ok=True)

    # load the event logs
    log_p = pm4py.read_xes(desirable_log, return_legacy_log_object=True)
    log_m = pm4py.read_xes(undesirable_log, return_legacy_log_object=True)

    print("\U0001F375Discovering and evaluating petri nets...")
    start = time.time()
    match variant:
        case 'imf':
            print("\U0001F9E9 Inductive Miner f (noise_threshold={})".format(args.noise_threshold))
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log_p,
                                                                                     noise_threshold=args.noise_threshold,
                                                                                     multi_processing=True)
            suffix = 't{}'.format(args.noise_threshold)
            model_filename = 'imtd_petri_{}'.format(suffix)

        case 'imfbi':
            print("\U0001F9E9 Inductive Miner fbi (threshold={}, filter_ratio={})".format(args.noise_threshold,
                                                                                          args.filter_ratio))
            net, initial_marking, final_marking = discover_petri_net_inductive(log_p, log_m,
                                                                               filter_ratio=args.filter_ratio,
                                                                               noise_threshold=args.noise_threshold,
                                                                               multi_processing=True)
            suffix = 't{}_f{}'.format(args.noise_threshold, args.filter_ratio)
            model_filename = 'imtd_petri_{}'.format(suffix)

        case 'imbi':
            print("\U0001F9E9 Inductive Miner bi (support={}, ratio={})".format(args.support, args.ratio))
            net, initial_marking, final_marking = discover_petri_net_inductive_bi(
                log_p,
                log_m,
                sup=args.support,
                ratio=args.ratio,
                size_par=len(log_p) / len(log_m),
                parallel=args.parallel)
            suffix = 's{}_r{}'.format(args.support, args.ratio)
            model_filename = 'imtd_petri_{}'.format(suffix)

        case 'imtd':
            similarity_matrix = np.zeros((len(log_p), len(log_m)))
            print("\U0001F9E9 Inductive Miner td (support={}, filter_ratio={})".format(args.support, args.filter_ratio))
            net, initial_marking, final_marking = discover_petri_net_inductive_td(
                log_p,
                log_m,
                similarity_matrix,
                sup=args.support,
                ratio=0.0,
                size_par=len(log_p) / len(log_m),
                weight=args.filter_ratio)
            suffix = 's{}_f{}'.format(args.support, args.filter_ratio)
            model_filename = 'imtd_petri_{}'.format(suffix)

        case _:
            print("\U00002699 Invalid variant")
            return

    end = time.time()
    elapsed_time = end - start
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Discover time: {} ({} s)".format(elapsed_time_str, elapsed_time))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # save the petri net
    save_petri_net(net, initial_marking, final_marking, output, model_filename)

    print("\U00002705 Completed discovering petri nets.")

    mes = Optimzation_Goals.apply_petri(log_p, log_m, net, initial_marking, final_marking)
    mes['time'] = elapsed_time
    save_measurements(mes, output, model_filename)

    print("\U00002705Completed evaluating the models.")


def save_petri_net(petri_net, initial_marking, final_marking, output: Path, filename: str):
    model_file_path = output.joinpath(filename)
    pm4py.write_pnml(petri_net, initial_marking, final_marking, str(model_file_path))
    graph_file_path = output.joinpath(filename + '.svg')
    pm4py.vis.save_vis_petri_net(petri_net, initial_marking, final_marking, str(graph_file_path))


def save_measurements(measurements, output: Path, filename: str):
    result = [
        ('acc', measurements['acc']),
        ('F1', measurements['F1']),
        ('precision', measurements['precision']),
        ('fitP', measurements['fitP']),
        ('fitM', measurements['fitM']),
        ('acc_perf', measurements['acc_perf']),
        ('F1_perf', measurements['F1_perf']),
        ('acc_ML', measurements['acc_ML']),
        ('prc_ML', measurements['prc_ML']),
        ('rec_ML', measurements['rec_ML']),
        ('time', measurements['time']),
    ]
    mes_filename = output.joinpath(filename + '.csv')
    with open(mes_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)


if __name__ == "__main__":
    main()
