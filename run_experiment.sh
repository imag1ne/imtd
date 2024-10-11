#!/bin/bash

# Usage: ./run_experiment.sh -v [variant] -t [threshold] -s [support] -r [ratio] -f [filter_ratio] -d [dataset] -e [event_log_type] -a [sample_amount] -o [output_dir]
# Example: ./run_experiment.sh -v k -t 0.0 -d Sepsis_Cases -e complete -o output/sepsis_imfbi_t0.0
#          ./run_experiment.sh -v k -t 0.0 -d BPIC_2017 -e sampled -a 2000 -o output/sepsis_imfbi_t0.0

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
    *)
        echo "\U274C Invalid dataset selection. Use 'Sepsis_Cases' or 'BPIC_2017'."
        exit 1;;
esac

case $event_log_type in
    "complete")
        event_log_suffix="event_log.xes"
        desirable_event_log_suffix="desirable_event_log.xes"
        undesirable_event_log_suffix="undesirable_event_log.xes";;
    "sampled")
        if [ -z "$sample_amount" ]; then
            sample_amount=1000
        fi
        event_log_suffix="event_log_sample_${sample_amount}.xes"
        desirable_event_log_suffix="desirable_event_log_sample_${sample_amount}.xes"
        undesirable_event_log_suffix="undesirable_event_log_sample_${sample_amount}.xes";;
    *)
        echo "\U274C Invalid event log type selection. Use 'complete' or 'sampled'."
        exit 1;;
esac

# Define the event log paths
event_log="$dataset_path/$event_log_suffix"
desirable_event_log="$dataset_path/$desirable_event_log_suffix"
undesirable_event_log="$dataset_path/$undesirable_event_log_suffix"

# Construct the experiment command
experiment_command="poetry run experiment -v $variant -l $event_log -p $desirable_event_log -m $undesirable_event_log -o $output_dir"

# Add specific parameters based on the selected variant
case $variant in
    "f")
        if [ ! -z "$threshold" ]; then
            experiment_command+=" -t $threshold"
        fi;;
    "b"|"t")
        if [ ! -z "$support" ]; then
            experiment_command+=" -s $support"
        fi
        if [ ! -z "$ratio" ]; then
            experiment_command+=" -r $ratio"
        fi
        if [ "$variant" = "t" ] && [ ! -z "$filter_ratio" ]; then
            experiment_command+=" -f $filter_ratio"
        fi;;
    "k")
        if [ ! -z "$threshold" ]; then
            experiment_command+=" -t $threshold"
        fi
        if [ ! -z "$filter_ratio" ]; then
            experiment_command+=" -f $filter_ratio"
        fi;;
    *)
        echo "\U274C Invalid variant selection. Use 'f', 'b', 't', or 'k'."
        exit 1;;
esac

# Run the experiment
echo "\U2699 Running experiment with command: $experiment_command"
$experiment_command

# Run the plot command
plot_command="poetry run plot -d $output_dir"
echo "\U2699 Running plot with command: $plot_command"
$plot_command
