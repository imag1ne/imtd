import argparse
import csv
import pm4py
import numpy as np
from pathlib import Path

from imtd import discover_petri_net_inductive, discover_petri_net_inductive_bi, discover_petri_net_inductive_td, \
    Optimzation_Goals


def parse_args():
    parser = argparse.ArgumentParser(prog='experiment', description='experiment')
    parser.add_argument('-v', '--variants', type=str, required=False, default='fbtk')
    parser.add_argument('-t', '--noise-threshold', type=str, required=False, default=None)
    parser.add_argument('-s', '--support', type=str, required=False)
    parser.add_argument('-r', '--ratio', type=str, required=False)
    parser.add_argument('-f', '--filter-ratio', type=str, required=False)
    parser.add_argument('-l', '--event-log', type=Path, required=True)
    parser.add_argument('-p', '--desirable-log', type=Path, required=True)
    parser.add_argument('-m', '--undesirable-log', type=Path, required=False)
    parser.add_argument('-d', '--similarity-matrix', type=Path, required=False)
    parser.add_argument('-o', '--output', type=Path, default=Path('output'))

    return parser.parse_args()


def main():
    args = parse_args()
    variants = args.variants
    event_log = str(args.event_log)
    desirable_log = str(args.desirable_log)
    undesirable_log = str(args.undesirable_log)
    similarity_matrix = args.similarity_matrix
    output = args.output

    output.mkdir(parents=True, exist_ok=True)

    # load the event logs
    log = pm4py.read_xes(event_log, return_legacy_log_object=True)
    log_p = pm4py.read_xes(desirable_log, return_legacy_log_object=True)
    if undesirable_log:
        log_m = pm4py.read_xes(undesirable_log, return_legacy_log_object=True)
    else:
        log_m = log_p

    print("\U0001F375Discovering and evaluating petri nets...")
    if 'f' in variants:
        noise_thresholds = parse_to_float_list(args.noise_threshold)
        n = len(noise_thresholds)
        for i, noise_threshold in enumerate(noise_thresholds, 1):
            print("\U0001F9E9[{}/{}] Inductive Miner (noise_threshold={})".format(i, n, noise_threshold))
            petri_net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log,
                                                                                           noise_threshold=noise_threshold,
                                                                                           multi_processing=True)
            suffix = 't{}'.format(noise_threshold)
            # save the petri net
            model_filename = 'imf_petri_{}'.format(suffix)
            save_petri_net(petri_net, initial_marking, final_marking, output, model_filename)
            mes = Optimzation_Goals.apply_petri(log_p, log_m, petri_net, initial_marking, final_marking)
            save_measurements(mes, output, model_filename)

    if 'b' in variants:
        supports = parse_to_float_list(args.support)
        ratios = parse_to_float_list(args.ratio)
        n = len(supports) * len(ratios)
        for i, support in enumerate(supports, 1):
            for j, ratio in enumerate(ratios, 1):
                cur_n = (i - 1) * len(ratios) + j
                print("\U0001F9E9[{}/{}]Inductive Miner bi (support={}, ratio={})".format(cur_n, n, support, ratio))
                petri_net, initial_marking, final_marking = discover_petri_net_inductive_bi(
                    log_p,
                    log_m,
                    sup=support,
                    ratio=ratio,
                    size_par=len(log_p) / len(log_m),
                    parallel=True)
                suffix = 's{}_r{}'.format(support, ratio)
                # save the petri net
                model_filename = 'imbi_petri_{}'.format(suffix)
                save_petri_net(petri_net, initial_marking, final_marking, output, model_filename)
                mes = Optimzation_Goals.apply_petri(log_p, log_m, petri_net, initial_marking, final_marking)
                save_measurements(mes, output, model_filename)

    if 't' in variants:
        if similarity_matrix is None:
            print("Similarity matrix is required for Inductive Miner td.")
        else:
            similarity_matrix = np.genfromtxt(similarity_matrix, delimiter=',')
            supports = parse_to_float_list(args.support)
            ratios = parse_to_float_list(args.ratio)
            filter_ratios = parse_to_float_list(args.filter_ratio)
            n = len(supports) * len(ratios) * len(filter_ratios)
            for i, support in enumerate(supports, 1):
                for j, ratio in enumerate(ratios, 1):
                    for k, filter_ratio in enumerate(filter_ratios, 1):
                        cur_n = (i - 1) * len(ratios) * len(filter_ratios) + (j - 1) * len(filter_ratios) + k
                        print(
                            "\U0001F9E9[{}/{}]Inductive Miner td (support={}, ratio={}, filter_ratio={})".format(cur_n,
                                                                                                                 n,
                                                                                                                 support,
                                                                                                                 ratio,
                                                                                                                 filter_ratio))
                        petri_net, initial_marking, final_marking = discover_petri_net_inductive_td(
                            log,
                            log_m,
                            similarity_matrix,
                            sup=support,
                            ratio=ratio,
                            size_par=len(log) / len(log_m),
                            weight=filter_ratio)
                        suffix = 's{}_r{}_f{}'.format(support, ratio, filter_ratio)
                        # save the petri net
                        model_filename = 'imtd_petri_{}'.format(suffix)
                        save_petri_net(petri_net, initial_marking, final_marking, output, model_filename)
                        mes = Optimzation_Goals.apply_petri(log_p, log_m, petri_net, initial_marking, final_marking)
                        save_measurements(mes, output, model_filename)

    if 'k' in variants:
        filter_ratios = parse_to_float_list(args.filter_ratio)
        noise_thresholds = parse_to_float_list(args.noise_threshold)
        n = len(filter_ratios) * len(noise_thresholds)
        for i, noise_threshold in enumerate(noise_thresholds):
            for j, filter_ratio in enumerate(filter_ratios):
                cur_n = (i - 1) * len(filter_ratios) + j
                print("\U0001F9E9[{}/{}]Inductive Miner fbi (threshold={}, filter_ratio={})".format(cur_n, n,
                                                                                                    noise_threshold,
                                                                                                    filter_ratio))
                petri_net, initial_marking, final_marking = discover_petri_net_inductive(log,
                                                                                         log_m,
                                                                                         filter_ratio, noise_threshold,
                                                                                         multi_processing=True)
                suffix = 't{}_f{}'.format(noise_threshold, filter_ratio)
                # save the petri net
                model_filename = 'imfbi_petri_{}'.format(suffix)
                save_petri_net(petri_net, initial_marking, final_marking, output, model_filename)
                mes = Optimzation_Goals.apply_petri(log_p, log_m, petri_net, initial_marking, final_marking)
                save_measurements(mes, output, model_filename)

    print("\U00002705Completed discovering petri nets and evaluating the models.")


def parse_to_float_list(param: str | None) -> list[float]:
    if param:
        return [float(x) for x in param.split(',')]
    else:
        return list(round(p, 1) for p in np.arange(0.0, 1.1, 0.1))


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
    ]
    mes_filename = output.joinpath(filename + '.csv')
    with open(mes_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)


if __name__ == "__main__":
    main()

# run the code
# python scripts/main.py -s 0.3 -r 0.3 -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -m "../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_100.xes" -d "../Dataset/BPI Challenge 2017_1_all/similarity_matrix_100.csv" -o "output/output_100_traces"
# poetry run experiment -s 0.3 -r 0.3 -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -m "../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_100.xes" -d "../Dataset/BPI Challenge 2017_1_all/similarity_matrix_100.csv" -o "output/output_100_traces"
