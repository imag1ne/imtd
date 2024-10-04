import argparse
import csv
import pm4py
import numpy as np
import time
from pathlib import Path

from imtd import discover_petri_net_inductive_bi, discover_petri_net_inductive_td, Optimzation_Goals


def parse_args():
    parser = argparse.ArgumentParser(prog='InductiveMiner', description='InductiveMiner')
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', help='additional help',
                                       dest='subcommand')

    im_parser = subparsers.add_parser('im', help='Inductive Miner')
    im_parser.add_argument('-p', '--desirable-log', type=str, required=True)
    im_parser.add_argument('-m', '--undesirable-log', type=str, required=False)
    im_parser.add_argument('-t', '--noise-threshold', type=float, required=False)
    im_parser.add_argument('-o', '--output', type=str, default='output')

    imbi_parser = subparsers.add_parser('imbi', help='Inductive Miner bi')
    imbi_parser.add_argument('-s', '--support', type=float, required=True)
    imbi_parser.add_argument('-r', '--ratio', type=float, required=True)
    imbi_parser.add_argument('-p', '--desirable-log', type=str, required=True)
    imbi_parser.add_argument('-m', '--undesirable-log', type=str, required=True)
    imbi_parser.add_argument('-o', '--output', type=str, default='output')
    imbi_parser.add_argument('-x', '--parallel', type=bool, default=False, action=argparse.BooleanOptionalAction)

    imtd_parser = subparsers.add_parser('imtd', help='Inductive Miner td')
    imtd_parser.add_argument('-s', '--support', type=float, required=True)
    imtd_parser.add_argument('-r', '--ratio', type=float, required=True)
    imtd_parser.add_argument('-w', '--weight', type=float, required=True)
    imtd_parser.add_argument('-p', '--desirable-log', type=str, required=True)
    imtd_parser.add_argument('-m', '--undesirable-log', type=str, required=True)
    imtd_parser.add_argument('-d', '--similarity-matrix', type=str, required=True)
    imtd_parser.add_argument('-o', '--output', type=str, default='output')

    return parser.parse_args()


def main():
    args = parse_args()

    log_p = None
    log_m = None

    # load the event logs
    match args.subcommand:
        case 'im':
            log_p = pm4py.read_xes(args.desirable_log, return_legacy_log_object=True)
            if args.undesirable_log:
                log_m = pm4py.read_xes(args.undesirable_log, return_legacy_log_object=True)
            else:
                log_m = log_p
        case _:
            log_p = pm4py.read_xes(args.desirable_log, return_legacy_log_object=True)
            log_m = pm4py.read_xes(args.undesirable_log, return_legacy_log_object=True)

    # discover the petri net
    print("Discovering the petri net...")
    net = None
    initial_marking = None
    final_marking = None
    start = time.time()
    match args.subcommand:
        case 'im':
            noise_threshold = args.noise_threshold or 0.0
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log_p,
                                                                                     noise_threshold=noise_threshold,
                                                                                     multi_processing=True)
        case 'imbi':
            net, initial_marking, final_marking = discover_petri_net_inductive_bi(
                log_p,
                log_m,
                sup=args.support,
                ratio=args.ratio,
                size_par=len(log_p) / len(log_m),
                parallel=args.parallel)
        case 'imtd':
            similarity_matrix = np.genfromtxt(args.similarity_matrix, delimiter=',')
            net, initial_marking, final_marking = discover_petri_net_inductive_td(
                log_p,
                log_m,
                similarity_matrix,
                sup=args.support,
                ratio=args.ratio,
                size_par=len(log_p) / len(log_m),
                weight=args.weight)

    end = time.time()
    elapsed_time = end - start
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Discover time: {} ({} s)".format(elapsed_time_str, elapsed_time))

    output = args.output
    Path(args.output).mkdir(parents=True, exist_ok=True)

    pnml_file_name = None
    pnsvg_file_name = None
    mes_filename = None
    match args.subcommand:
        case 'im':
            pnml_file_name = "/petri_im"
            pnsvg_file_name = "/petri_im.svg"
            mes_filename = "/mes_im.csv"
        case 'imbi':
            pnml_file_name = "/petri_imbi_r" + str(args.ratio) + "_s" + str(args.support)
            pnsvg_file_name = "/petri_imbi_r" + str(args.ratio) + "_s" + str(args.support) + ".svg"
            mes_filename = "/mes_imbi_r" + str(args.ratio) + "_s" + str(args.support) + ".csv"
        case 'imtd':
            pnml_file_name = "/petri_imtd_r" + str(args.ratio) + "_s" + str(args.support) + "_w" + str(args.weight)
            pnsvg_file_name = "/petri_imtd_r" + str(args.ratio) + "_s" + str(args.support) + "_w" + str(
                args.weight) + ".svg"
            mes_filename = "/mes_imtd_r" + str(args.ratio) + "_s" + str(args.support) + "_w" + str(args.weight) + ".csv"

    # save the petri net
    print("Saving the petri net...")
    pm4py.write_pnml(net, initial_marking, final_marking, output + pnml_file_name)

    # visualize the petri net
    pm4py.vis.save_vis_petri_net(net, initial_marking, final_marking, output + pnsvg_file_name)

    mes = Optimzation_Goals.apply_petri(log_p, log_m, net, initial_marking, final_marking)
    result = [
        ('acc', mes['acc']),
        ('F1', mes['F1']),
        ('precision', mes['precision']),
        ('fitP', mes['fitP']),
        ('fitM', mes['fitM']),
        ('acc_perf', mes['acc_perf']),
        ('F1_perf', mes['F1_perf']),
        ('acc_ML', mes['acc_ML']),
        ('prc_ML', mes['prc_ML']),
        ('rec_ML', mes['rec_ML']),
        ('time', "{} ({} s)".format(elapsed_time_str, elapsed_time))
    ]
    with open(output + mes_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)

    for m, v in result:
        print(m, ":", v)


if __name__ == "__main__":
    main()

# run the code
# python scripts/main.py imtd -s 0.3 -r 0.3 -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -m "../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_100.xes" -d "../Dataset/BPI Challenge 2017_1_all/similarity_matrix_100.csv" -o "output/output_100_traces"
# poetry run discover imtd -s 0.3 -r 0.3 -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -m "../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_100.xes" -d "../Dataset/BPI Challenge 2017_1_all/similarity_matrix_100.csv" -o "output/output_100_traces"
# poetry run discover imbi -s 0.3 -r 0.3 -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -m "../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_100.xes" -o "output/output_100_traces"
# poetry run discover im -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -o "output/output_100_traces"
