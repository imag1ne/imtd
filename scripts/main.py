import argparse
import csv
import pm4py
import numpy as np
import time
from pathlib import Path

from imtd import inductive_miner, Optimzation_Goals

def parse_args():
    parser = argparse.ArgumentParser(prog='InductiveMiner_bi', description='InductiveMiner_bi')
    parser.add_argument('-s', '--support', type=float, required=True)
    parser.add_argument('-r', '--ratio', type=float, required=True)
    parser.add_argument('-p', '--desirable-log', type=str, required=True)
    parser.add_argument('-m', '--undesirable-log', type=str, required=True)
    parser.add_argument('-d', '--similarity-matrix', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('-x', '--parallel', type=bool, default=False, action=argparse.BooleanOptionalAction)

    return parser.parse_args()

def main():
    args = parse_args()
    support = args.support
    ratio = args.ratio
    LPlus_LogFile = args.desirable_log
    LMinus_LogFile = args.undesirable_log
    similarity_matrix = args.similarity_matrix
    output = args.output
    parallel = args.parallel

    # load the event logs
    print("Loading event logs...")
    logP = pm4py.read_xes(LPlus_LogFile, return_legacy_log_object=True)
    logM = pm4py.read_xes(LMinus_LogFile, return_legacy_log_object=True)
    print("Desirable log: ", len(logP), " traces")
    print("Undesirable log: ", len(logM), " traces")

    # load the similarity matrix
    print("Loading similarity matrix...")
    similarity_matrix = np.genfromtxt(similarity_matrix, delimiter=',')

    # discover the petri net
    print("Discovering the petri net...")
    start = time.time()
    net, initial_marking, final_marking = inductive_miner.apply_bi(
        logP,
        logM,
        similarity_matrix,
        variant=inductive_miner.Variants.IMbi,
        sup=support,
        ratio=ratio,
        size_par=len(logP)/len(logM),
        parallel=parallel)
    end = time.time()
    elapsed_time = end - start
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("Elapsed time: {} ({} s)".format(elapsed_time_str, elapsed_time))

    Path(output).mkdir(parents=True, exist_ok=True)

    # save the petri net
    print("Saving the petri net...")
    pnml_file_name = output + "/petri_r"+str(ratio)+"_s"+str(support)
    pm4py.write_pnml(net, initial_marking, final_marking, pnml_file_name)

    # visualize the petri net
    pnsvg_file_name = output + "/petri_r"+str(ratio)+"_s"+str(support)+".svg"
    pm4py.vis.save_vis_petri_net(net, initial_marking, final_marking, pnsvg_file_name)

    mes = Optimzation_Goals.apply_petri(logP, logM, net, initial_marking, final_marking)
    mes_filename = output + "/mes_r" + str(ratio) + "_s" + str(support) + ".csv"
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
        ('elapsed', "{} ({} s)".format(elapsed_time_str, elapsed_time))
    ]
    with open(mes_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)

    for m, v in result:
        print(m, ":", v)

if __name__ == "__main__":
    main()

# run the code
# python scripts/main.py -s 0.3 -r 0.3 -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -m "../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_100.xes" -d "../Dataset/BPI Challenge 2017_1_all/similarity_matrix_100.csv" -o "output/output_100_traces"
# poetry run entrypoint -s 0.3 -r 0.3 -p "../Dataset/BPI Challenge 2017_1_all/desirable_event_log_sample_100.xes" -m "../Dataset/BPI Challenge 2017_1_all/undesirable_event_log_sample_100.xes" -d "../Dataset/BPI Challenge 2017_1_all/similarity_matrix_100.csv" -o "output/output_100_traces"