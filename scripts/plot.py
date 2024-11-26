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

    imfbi_results = {key: [] for key in measurement_keys}
    imfbi_results['threshold'] = []
    imfbi_results['filter_ratio'] = []

    imbi_results = {key: [] for key in measurement_keys}
    imbi_results['support'] = []
    imbi_results['ratio'] = []

    imtd_results = {key: [] for key in measurement_keys}
    imtd_results['support'] = []
    imtd_results['ratio'] = []
    imtd_results['filter_ratio'] = []

    dirpath, _, filenames = next(data_path.walk())
    dataset_name = dirpath.stem.split('_')[0]
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

            case 'imfbi':
                threshold = params[0]
                imfbi_results['threshold'].append(threshold)
                filter_ratio = params[1]
                imfbi_results['filter_ratio'].append(filter_ratio)
                load_data_to_dict(filepath, imfbi_results, measurement_keys)

            case 'imbi':
                support = params[0]
                imbi_results['support'].append(support)
                ratio = params[1]
                imbi_results['ratio'].append(ratio)
                load_data_to_dict(filepath, imbi_results, measurement_keys)

            case 'imtd':
                support = params[0]
                imtd_results['support'].append(support)
                filter_ratio = params[1]
                imtd_results['filter_ratio'].append(filter_ratio)
                load_data_to_dict(filepath, imtd_results, measurement_keys)

    imf_df = pd.DataFrame(imf_results)
    imfbi_df = pd.DataFrame(imfbi_results)
    imbi_df = pd.DataFrame(imbi_results)
    imtd_df = pd.DataFrame(imtd_results)

    plot_imf_results(imf_df, dataset_name, data_path)
    plot_imfbi_results(imfbi_df, dataset_name, data_path)
    plot_imbi_results(imbi_df, dataset_name, data_path)
    plot_imtd_results(imtd_df, dataset_name, data_path)


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

        case 'imfbi':
            if len(filename_parts) < 4 or not filename_parts[2].startswith('t') or not filename_parts[3].startswith(
                    'f'):
                return None

            threshold = float(filename_parts[2][1:])
            filter_ratio = float(filename_parts[3][1:].rstrip('.csv'))
            return variant, [threshold, filter_ratio]

        case 'imbi':
            if len(filename_parts) < 4 or not filename_parts[2].startswith('s') or not filename_parts[3].startswith(
                    'r'):
                return None

            support = float(filename_parts[2][1:])
            ratio = float(filename_parts[3][1:].rstrip('.csv'))
            return variant, [support, ratio]

        case 'imtd':
            if len(filename_parts) < 3 or not filename_parts[2].startswith('s') or not filename_parts[3].startswith(
                    'f'):
                return None

            support = float(filename_parts[2][1:])
            filter_ratio = float(filename_parts[3][1:].rstrip('.csv'))
            return variant, [support, filter_ratio]

        case _:
            return None


def plot_results(df, title, xlabel, xkey, savepath):
    if len(df[xkey]) <= 1:
        return

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


def plot_nested_group_results(df, group_names, title, savepath):
    file_prefix = savepath.stem
    savepath = savepath.parent

    for xlabel, xlabel_short in group_names:
        xkey = xlabel
        g_names = [group_name for group_name, _ in group_names if group_name != xlabel]
        g_short_names = [group_name_short for _, group_name_short in group_names if group_name_short != xlabel_short]
        df_group = df.groupby(g_names)
        for group_key, grouped_df in df_group:
            params_info = ', '.join('{}={}'.format(n, v) for n, v in zip(g_names, group_key))
            title_with_param_info = '{} ({})'.format(title, params_info)
            file_suffix = '_'.join(g_short_names[i] + str(group_key[i]) for i in range(len(group_key)))
            filename = '{}_{}.png'.format(file_prefix, file_suffix)
            plot_results(grouped_df, title_with_param_info, xlabel, xkey,
                         savepath.joinpath(filename))


def plot_imf_results(imf_df, dataset_name, savepath: Path):
    title = 'IMf-{}'.format(dataset_name)
    plot_results(imf_df, title, 'f', 'threshold', savepath.joinpath('imf_fig.png'))


def plot_imfbi_results(imfbi_df, dataset_name, savepath: Path):
    title = 'IMfbi-{}'.format(dataset_name)
    plot_nested_group_results(imfbi_df, [('threshold', 't'), ('filter_ratio', 'f')], title,
                              savepath.joinpath('imfbi_fig'))


def plot_imbi_results(imbi_df, dataset_name, savepath: Path):
    title = 'IMbi-{}'.format(dataset_name)
    plot_nested_group_results(imbi_df, [('support', 's'), ('ratio', 'r')], title, savepath.joinpath('imbi_fig'))


def plot_imtd_results(imtd_df, dataset_name, savepath: Path):
    title = 'IMtd-{}'.format(dataset_name)
    plot_nested_group_results(imtd_df, [('support', 's'), ('filter_ratio', 'f')], title,
                              savepath.joinpath('imtd_fig'))


if __name__ == "__main__":
    main()
