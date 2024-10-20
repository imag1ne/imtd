use std::collections::{HashMap, HashSet};

pub(crate) fn fit_seq(
    log_variants: &HashMap<Vec<&str>, f64>,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
) -> f64 {
    let mut count = 0.0;

    for (trace, trace_num) in log_variants {
        for pair in trace.windows(2) {
            if part_b.contains(&pair[0]) && part_a.contains(&pair[1]) {
                count += trace_num;
                break;
            }
        }
    }

    let fit = 1.0 - (count / log_variants.values().sum::<f64>());
    fit
}

pub(crate) fn fit_exc(
    log_variants: &HashMap<Vec<&str>, f64>,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
) -> f64 {
    let mut count = 0.0;

    for (trace, trace_num) in log_variants {
        let activities = trace.iter().copied().collect::<HashSet<_>>();
        if activities.is_subset(part_a) || activities.is_subset(part_b) {
            count += trace_num;
        }
    }

    let fit = count / log_variants.values().sum::<f64>();
    fit
}

pub(crate) fn fit_loop(
    log_variants: &HashMap<Vec<&str>, f64>,
    part_a: &HashSet<&str>,
    part_b: &HashSet<&str>,
    end_part_a: &HashSet<&str>,
    start_part_a: &HashSet<&str>,
) -> f64 {
    let mut count = 0.0;

    for (trace, &trace_num) in log_variants {
        if trace.is_empty() {
            continue;
        }

        if part_b.contains(trace.first().unwrap()) || part_b.contains(trace.last().unwrap()) {
            count += trace_num;
            continue;
        }

        for pair in trace.windows(2) {
            let source = pair[0];
            let target = pair[1];
            if part_a.contains(source) && part_b.contains(target) {
                if !end_part_a.contains(source) {
                    count += trace_num;
                }
                break;
            }

            if part_a.contains(target) && part_b.contains(source) {
                if !start_part_a.contains(target) {
                    count += trace_num;
                }
                break;
            }
        }
    }

    let fit = 1.0 - (count / log_variants.values().sum::<f64>());
    fit
}
