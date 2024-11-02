#!/bin/bash

# Usage: ./run_discovery.sh -v [variant] -t [threshold] -s [support] -r [ratio] -f [filter_ratio] -d [dataset] -e [event_log_type] -a [sample_amount] -o [output_dir]
# Example: ./run_discovery.sh -v imfbi -t 0.0 -d Sepsis_Cases -e complete -o output/sepsis_imfbi_t0.0
#          ./run_discovery.sh -v imfbi -t 0.0 -d BPIC_2017 -e sampled -a 2000 -o output/bpic_2017_imfbi_t0.0

while getopts v:t:s:r:f:d:e:a:o: flag
do
    case "${flag}" in
        v) variant=${OPTARG};;
        t) threshold=${OPTARG};;
        s) support=${OPTARG};;
        r) ratio=${OPTARG};;
        f) filter_ratio=${OPTARG};;
        d) dataset=${OPTARG};;
        e) event_log_type=${OPTARG};;
        a) sample_amount=${OPTARG};;
        o) output_dir=${OPTARG};;
    esac
done

# Set dataset path based on the dataset and event log type selection
case $dataset in
    "Sepsis_Cases")
        dataset_path="../Dataset/Sepsis_Cases";;
    "BPIC_2017")
        dataset_path="../Dataset/BPIC_2017";;
    "CCC19")
        dataset_path="../Dataset/CCC19";;
    *)
        echo -e "\U274C Invalid dataset selection '${dataset}'. Use 'Sepsis_Cases', 'CCC19' or 'BPIC_2017'."
        exit 1;;
esac

case $event_log_type in
    "complete")
        desirable_event_log_suffix="desirable_event_log.xes"
        undesirable_event_log_suffix="undesirable_event_log.xes";;
    "sampled")
        if [ -z "$sample_amount" ]; then
            sample_amount=1000
        fi
        desirable_event_log_suffix="desirable_event_log_sample_${sample_amount}.xes"
        undesirable_event_log_suffix="undesirable_event_log_sample_${sample_amount}.xes";;
    *)
        echo -e "\U274C Invalid event log type selection '${event_log_type}'. Use 'complete' or 'sampled'."
        exit 1;;
esac

# Define the event log paths
desirable_event_log="$dataset_path/$desirable_event_log_suffix"
undesirable_event_log="$dataset_path/$undesirable_event_log_suffix"

# Construct the discovery command
discover_command="poetry run experiment_1 -v $variant -p $desirable_event_log -m $undesirable_event_log -o $output_dir"

# Default ranges for parameters if not provided
default_range() {
  seq 0.0 0.1 1.0
}

# Assign parameter values if provided, otherwise use default range
threshold_values=${threshold:-$(default_range)}
support_values=${support:-$(default_range)}
ratio_values=${ratio:-$(default_range)}
filter_ratio_values=${filter_ratio:-$(default_range)}

# Add specific parameters based on the selected variant
case $variant in
    "imf")
        for threshold in $threshold_values; do
            discover_cmd="$discover_command -t $threshold"
            echo -e "\U2699 Running discovery with command: $discover_cmd"
            $discover_cmd
        done;;
    "imbi")
        for support in $support_values; do
            for ratio in $ratio_values; do
                discover_cmd="$discover_command -s $support -r $ratio"
                echo -e "\U2699 Running discovery with command: $discover_cmd"
                $discover_cmd
            done
        done;;
    "imtd")
        for support in $support_values; do
            for filter_ratio in $filter_ratio_values; do
                discover_cmd="$discover_command -s $support -f $filter_ratio"
                echo -e "\U2699 Running discovery with command: $discover_cmd"
                $discover_cmd
            done
        done;;
    "imfbi")
        for threshold in $threshold_values; do
            for filter_ratio in $filter_ratio_values; do
                discover_cmd="$discover_command -t $threshold -f $filter_ratio"
                echo -e "\U2699 Running discovery with command: $discover_cmd"
                $discover_cmd
            done
        done;;
    *)
        echo -e "\U274C Invalid variant selection '${variant}'. Use 'imf', 'imbi', 'imtd', or 'imfbi'."
        exit 1;;
esac
