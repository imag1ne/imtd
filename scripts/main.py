import argparse
import csv
import pm4py
import time
from pathlib import Path

from imtd import discover_petri_net_inductive, discover_petri_net_inductive_td, \
    Optimzation_Goals


def parse_args():
    parser = argparse.ArgumentParser(prog='InductiveMiner', description='InductiveMiner')
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', help='additional help',
                                       dest='subcommand')

    imfbi_parser = subparsers.add_parser('imfbi', help='Inductive Miner fbi')
    imfbi_parser.add_argument('-p', '--desirable-log', type=str, required=True)
    imfbi_parser.add_argument('-m', '--undesirable-log', type=str, required=False)
    imfbi_parser.add_argument('-f', '--filter-ratio', type=float, required=False)
    imfbi_parser.add_argument('-t', '--noise-threshold', type=float, required=False)
    imfbi_parser.add_argument('-o', '--output', type=str, default='output')

    imtd_parser = subparsers.add_parser('imfc', help='Inductive Miner fc')
    imtd_parser.add_argument('-w', '--weight', type=float, required=False)
    imtd_parser.add_argument('-f', '--filter-ratio', type=float, required=False)
    imtd_parser.add_argument('-p', '--desirable-log', type=str, required=True)
    imtd_parser.add_argument('-m', '--undesirable-log', type=str, required=True)
    imtd_parser.add_argument('-o', '--output', type=str, default='output')

    return parser.parse_args()


def main():
    args = parse_args()

    # load the event logs
    log_p = pm4py.read_xes(args.desirable_log, return_legacy_log_object=True)
    log_m = pm4py.read_xes(args.undesirable_log, return_legacy_log_object=True)

    # discover the petri net
    print("Discovering the petri net...")
    net = None
    initial_marking = None
    final_marking = None
    pnml_file_name = ''
    pnsvg_file_name = ''
    mes_filename = ''
    start = time.time()
    match args.subcommand:
        case 'imfbi':
            filter_ratio = args.filter_ratio or 0.0
            noise_threshold = args.noise_threshold or 0.0
            net, initial_marking, final_marking = discover_petri_net_inductive(log_p, log_m, filter_ratio=filter_ratio,
                                                                               noise_threshold=noise_threshold,
                                                                               multi_processing=True)
            suffix = 'f{}_t{}'.format(filter_ratio, noise_threshold)
            pnml_file_name = 'imfbi_petri_{}'.format(suffix)
            pnsvg_file_name = 'imfbi_petri_{}.svg'.format(suffix)
            mes_filename = 'imfbi_petri_{}.csv'.format(suffix)

        case 'imfc':
            weight = args.weight or 0.5
            filter_ratio = args.filter_ratio or 0.0

            net, initial_marking, final_marking = discover_petri_net_inductive_td(
                log_p,
                log_m,
                weight=weight,
                filter_ratio=filter_ratio)
            suffix = 'w{}_f{}'.format(weight, filter_ratio)
            pnml_file_name = 'imtd_petri_{}'.format(suffix)
            pnsvg_file_name = 'imtd_petri_{}.svg'.format(suffix)
            mes_filename = 'imtd_petri_{}.csv'.format(suffix)

    end = time.time()
    elapsed_time = end - start
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Discover time: {} ({} s)".format(elapsed_time_str, elapsed_time))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pnml_file_path = output.joinpath(pnml_file_name)
    pnsvg_file_path = output.joinpath(pnsvg_file_name)
    mes_file_path = output.joinpath(mes_filename)

    # save the petri net
    print("Saving the petri net...")
    pm4py.write_pnml(net, initial_marking, final_marking, str(pnml_file_path))

    # visualize the petri net
    pm4py.vis.save_vis_petri_net(net, initial_marking, final_marking, str(pnsvg_file_path))

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
    with open(mes_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)

    for m, v in result:
        print(m, ":", v)


if __name__ == "__main__":
    main()

# run the code
# python scripts/main.py imfc -w 0.3 -f 0.3 -p "../Dataset/BPIC_2017/desirable_event_log_sample_100.xes" -m "../Dataset/BPIC_2017/undesirable_event_log_sample_100.xes" -o "output/output_100_traces"
# poetry run discover imfc -w 0.3 -f 0.3 -p "../Dataset/BPIC_2017/desirable_event_log_sample_100.xes" -m "../Dataset/BPIC_2017/undesirable_event_log_sample_100.xes" -o "output/output_100_traces"
