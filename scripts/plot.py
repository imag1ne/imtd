import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(prog='Plot', description='Plot')
    parser.add_argument('-d', '--data-path', type=Path, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    measurement_keys = {'acc', 'F1', 'precision', 'fitP', 'fitM', 'acc_perf', 'F1_perf', 'acc_ML', 'prc_ML', 'rec_ML'}
    imf_results = {key: [] for key in measurement_keys}
    imf_results['threshold'] = []
    imbi_results = {key: [] for key in measurement_keys}
    imbi_results['support'] = []
    imbi_results['ratio'] = []
    imtd_results = {key: [] for key in measurement_keys}
    imtd_results['support'] = []
    imtd_results['ratio'] = []
    imtd_results['weight'] = []
    for dirpath, dirnames, filenames in data_path.walk():
        for filename in filenames:
            filepath = dirpath.joinpath(filename)
            parsed = parse_data_filename(filename)
            if not parsed:
                continue

            variant, params = parsed
            match variant:
                case 'imf':
                    threshold = params[0]
                    imf_results['threshold'].append(threshold)
                    load_data_to_dict(filepath, imf_results, measurement_keys)

                case 'imbi':
                    support = params[0]
                    imbi_results['support'].append(support)
                    ratio = params[1]
                    imbi_results['ratio'].append(ratio)
                    load_data_to_dict(filepath, imbi_results, measurement_keys)

                case 'imtd':
                    support = params[0]
                    imtd_results['support'].append(support)
                    ratio = params[1]
                    imtd_results['ratio'].append(ratio)
                    weight = params[2]
                    imtd_results['weight'].append(weight)
                    load_data_to_dict(filepath, imtd_results, measurement_keys)

    imf_df = pd.DataFrame(imf_results)
    imbi_df = pd.DataFrame(imbi_results)
    imtd_df = pd.DataFrame(imtd_results)

    plot_imf_results(imf_df, data_path)
    plot_imbi_results(imbi_df, data_path)
    plot_imtd_results(imtd_df, data_path)


def as_float(n: str) -> float:
    try:
        return float(n)
    except ValueError:
        return np.nan


def load_data_to_dict(filepath: Path, obj: dict, keys: set[str]) -> None:
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key, value = row[0], as_float(row[1])
            if key in keys:
                obj[key].append(value)


def parse_data_filename(filename: str) -> tuple[str, list[float]] | None:
    filename_parts = filename.split('_')
    if len(filename_parts) < 3 or not filename.endswith('.csv'):
        return None

    variant = filename_parts[0]
    match variant:
        case 'imf':
            if not filename_parts[2].startswith('t'):
                return None

            threshold = float(filename_parts[2][1:4])
            return variant, [threshold]
        case 'imbi':
            if len(filename_parts) < 4 or not filename_parts[2].startswith('s') or not filename_parts[3].startswith(
                    'r'):
                return None

            support = float(filename_parts[2][1:])
            ratio = float(filename_parts[3][1:].rstrip('.csv'))
            return variant, [support, ratio]
        case 'imtd':
            if len(filename_parts) < 4 or not filename_parts[2].startswith('s') or not filename_parts[3].startswith(
                    'r') or not filename_parts[4].startswith('w'):
                return None

            support = float(filename_parts[2][1:])
            ratio = float(filename_parts[3][1:])
            weight = float(filename_parts[4][1:].rstrip('.csv'))
            return variant, [support, ratio, weight]
        case _:
            return None


def plot_results(df, title, xlabel, xkey, savepath):
    df = df.sort_values(by=xkey)
    plt.plot(df[xkey], df['precision'], label='Precision', color='red', marker='s', linestyle='-', linewidth=2)
    plt.plot(df[xkey], df['fitP'], label='Fitness', color='blue', marker='s', linestyle='-', linewidth=2)
    plt.plot(df[xkey], df['fitM'], label='Fitness_M', color='green', marker='s', linestyle='-', linewidth=2)
    plt.xticks(df[xkey])
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('evaluation metrics')
    plt.grid(True)
    plt.legend()
    plt.savefig(savepath, dpi=300)
    plt.clf()


def plot_imf_results(imf_df, savepath: Path):
    plot_results(imf_df, 'IMF-BPIC17', 'f', 'threshold', savepath.joinpath('imf_fig.png'))


def plot_imbi_results(imbi_df, savepath: Path):
    imbi_df_support_group = imbi_df.groupby('support')
    for support, df in imbi_df_support_group:
        plot_results(df, 'IMbi-BPIC17 (support={})'.format(support), 'ratio', 'ratio',
                     savepath.joinpath('imbi_fig_s{}.png'.format(support)))

    imbi_df_ratio_group = imbi_df.groupby('ratio')
    for ratio, df in imbi_df_ratio_group:
        plot_results(df, 'IMbi-BPIC17 (ratio={})'.format(ratio), 'support', 'support',
                     savepath.joinpath('imbi_fig_r{}.png'.format(ratio)))


def plot_imtd_results(imtd_df, savepath: Path):
    # imtd_df_support_group = imtd_df.groupby('support')
    # for support, df in imtd_df_support_group:
    #     plot_results(df, 'IMtd-BPIC17 (support={})'.format(support), 'ratio', 'ratio',
    #                  savepath.joinpath('imtd_fig_s{}.png'.format(support)))
    #
    # imtd_df_ratio_group = imtd_df.groupby('ratio')
    # for ratio, df in imtd_df_ratio_group:
    #     plot_results(df, 'IMtd-BPIC17 (ratio={})'.format(ratio), 'support', 'support',
    #                  savepath.joinpath('imtd_fig_r{}.png'.format(ratio)))

    imtd_df_support_group = imtd_df.groupby('support')
    for support, df in imtd_df_support_group:
        df_ratio_group = df.groupby('ratio')
        for ratio, df in df_ratio_group:
            plot_results(df, 'IMtd-BPIC17 (support={}, ratio={})'.format(support, ratio), 'weight', 'weight',
                         savepath.joinpath('imtd_fig_s{}_r{}.png'.format(support, ratio)))


if __name__ == "__main__":
    main()
