import argparse
import csv

import pm4py
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(prog='evaluate', description='evaluate discovered model')
    parser.add_argument('-d', '--process-model', type=str, required=True)
    parser.add_argument('-p', '--desirable-log', type=str, required=True)
    parser.add_argument('-m', '--undesirable-log', type=str, required=True)
    parser.add_argument('-x', '--parallel', type=bool, default=False, action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def main():
    args = parse_args()
    process_model = args.process_model
    desirable_log = args.desirable_log
    undesirable_log = args.undesirable_log
    output = Path(args.process_model).parent
    model_filename = Path(args.process_model).stem

    # load the event logs
    log_p = pm4py.read_xes(desirable_log, return_legacy_log_object=True)
    log_m = pm4py.read_xes(undesirable_log, return_legacy_log_object=True)
    net, initial_marking, final_marking = pm4py.read_pnml(process_model)

    mes = apply_petri(log_p, log_m, net, initial_marking, final_marking, multi_processing=args.parallel)
    save_measurements(mes, output, model_filename)

    print("\U00002705Completed evaluating the models.")


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


def apply_petri(LPlus, LMinus, net, i_m, i_f, multi_processing=False):
    measures = {}

    alp = pm4py.conformance_diagnostics_alignments(LPlus, net, i_m, i_f, multi_processing=multi_processing)
    fp_inf = replay_fitness.evaluate(alp, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fp = fp_inf['averageFitness']
    fp_pef = fp_inf['percentage_of_fitting_traces'] / 100
    # roc_data = [('p', x['fitness']) for x in alp]

    ################################################################################
    prec_Plus = pm4py.precision_alignments(LPlus, net, i_m, i_f, multi_processing=multi_processing)
    ################################################################################

    alm = pm4py.conformance_diagnostics_alignments(LMinus, net, i_m, i_f, multi_processing=multi_processing)
    fm_inf = replay_fitness.evaluate(alm, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fm = fm_inf['averageFitness']
    fm_pef = fm_inf['percentage_of_fitting_traces'] / 100
    # roc_data += [('n', x['fitness']) for x in alm]

    measures['acc'] = round(fp - fm, 2)
    measures['F1'] = round(2 * ((fp * (1 - fm)) / (fp + (1 - fm))), 2)
    measures['precision'] = round(prec_Plus, 2)
    measures['fitP'] = round(fp, 2)
    measures['fitM'] = round(fm, 2)
    measures['acc_perf'] = round(fp_pef - fm_pef, 2)
    measures['F1_perf'] = round(2 * ((fp_pef * (1 - fm_pef)) / (fp_pef + (1 - fm_pef))), 2)

    TP = fp_pef * len(LPlus)
    FP = fm_pef * len(LMinus)
    FN = (1 - fp_pef) * len(LPlus)
    TN = (1 - fm_pef) * len(LMinus)
    measures['acc_ML'] = (TP + TN) / (TP + TN + FP + FN)
    if (TP + FP) != 0:
        measures['prc_ML'] = TP / (TP + FP)
    else:
        measures['prc_ML'] = 'ignore'

    if (TP + FN) != 0:
        measures['rec_ML'] = TP / (TP + FN)
    else:
        measures['rec_ML'] = 'ignore'
    # measures['roc_data'] = roc_data

    return measures


if __name__ == "__main__":
    main()
