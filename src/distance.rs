use std::collections::HashSet;
use std::hash::Hash;

use crate::pm4py::obj::Event;
use counter::Counter;
use pyo3::{pyfunction, PyResult};
use rayon::prelude::*;

const DEFAULT_NAME_KEY: &str = "concept:name";
const DEFAULT_RESOURCE_KEY: &str = "org:resource";

#[pyfunction]
#[pyo3(signature = (desirable_log, undesirable_log, activity_key=None, resource_key=None))]
pub fn distance_matrix<'py>(
    desirable_log: Vec<Vec<Event<'py>>>,
    undesirable_log: Vec<Vec<Event<'py>>>,
    activity_key: Option<&str>,
    resource_key: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let activity_key = activity_key.unwrap_or(DEFAULT_NAME_KEY);
    let resource_key = resource_key.unwrap_or(DEFAULT_RESOURCE_KEY);

    let desirable_log = preprocess_event_log(&desirable_log, activity_key, resource_key)?;
    let undesirable_log = preprocess_event_log(&undesirable_log, activity_key, resource_key)?;

    let matrix = desirable_log
        .par_iter()
        .map(|trace_p| {
            undesirable_log
                .iter()
                .map(|trace_m| {
                    let trace_p_len = trace_p.len();
                    let trace_m_len = trace_m.len();
                    if trace_p_len > trace_m_len {
                        trace_p
                            .windows(trace_m_len)
                            .map(|trace_p_slice| {
                                let activity_distance = activity_distance(trace_p_slice, trace_m);
                                let transition_distance =
                                    transition_distance(trace_p_slice, trace_m);
                                let resource_distance = resource_distance(trace_p_slice, trace_m);
                                activity_distance + transition_distance + resource_distance
                            })
                            .reduce(f64::min)
                            .unwrap()
                    } else {
                        let activity_distance = activity_distance(trace_p, trace_m);
                        let transition_distance = transition_distance(trace_p, trace_m);
                        let resource_distance = resource_distance(trace_p, trace_m);
                        activity_distance + transition_distance + resource_distance
                    }
                })
                .collect()
        })
        .collect();

    Ok(matrix)
}

fn preprocess_event_log<'py>(
    event_log: &[Vec<Event<'py>>],
    activity_key: &str,
    resource_key: &str,
) -> PyResult<Vec<Vec<ExtractedEvent<'py>>>> {
    let processed_event_log = event_log
        .iter()
        .map(|trace| {
            let processed_trace = trace
                .iter()
                .map(|event| {
                    let activity = event.get_str(activity_key)?;
                    let resource = event.get_str(resource_key)?;

                    let extracted_event = ExtractedEvent { activity, resource };

                    Ok(extracted_event)
                })
                .collect::<PyResult<Vec<ExtractedEvent<'_>>>>()?;

            Ok(processed_trace)
        })
        .collect::<PyResult<Vec<Vec<ExtractedEvent<'_>>>>>()?;

    Ok(processed_event_log)
}

fn activity_distance(trace1: &[ExtractedEvent], trace2: &[ExtractedEvent]) -> f64 {
    let activity_count1 = trace1.iter().map(|event| event.activity).collect();
    let activity_count2 = trace2.iter().map(|event| event.activity).collect();

    profile_distance(&activity_count1, &activity_count2)
}

fn transition_distance(trace1: &[ExtractedEvent], trace2: &[ExtractedEvent]) -> f64 {
    let transition_count1 = transition_counter_from_trace(trace1);
    let transition_count2 = transition_counter_from_trace(trace2);

    profile_distance(&transition_count1, &transition_count2)
}

fn resource_distance(trace1: &[ExtractedEvent], trace2: &[ExtractedEvent]) -> f64 {
    let resource_count1 = resource_counter_from_trace(trace1);
    let resource_count2 = resource_counter_from_trace(trace2);

    profile_distance(&resource_count1, &resource_count2)
}

fn transition_counter_from_trace<'a>(trace: &[ExtractedEvent<'a>]) -> Counter<(&'a str, &'a str)> {
    trace
        .windows(2)
        .map(|events| (events[0].activity, events[1].activity))
        .collect()
}

fn resource_counter_from_trace<'a>(trace: &[ExtractedEvent<'a>]) -> Counter<(&'a str, &'a str)> {
    trace
        .iter()
        .map(|event| (event.activity, event.resource))
        .collect()
}

fn profile_distance<T>(profile1: &Counter<T>, profile2: &Counter<T>) -> f64
where
    T: Hash + Eq + Copy,
{
    let profile1_keys: HashSet<T> = profile1.keys().copied().collect();
    let profile2_keys: HashSet<T> = profile2.keys().copied().collect();
    let all_keys_iter = profile1_keys.union(&profile2_keys);

    let sum = all_keys_iter
        .map(|key| {
            let x = *profile1.get(key).unwrap_or(&0) as isize;
            let y = *profile2.get(key).unwrap_or(&0) as isize;
            (x - y).pow(2)
        })
        .sum::<isize>() as f64;

    f64::sqrt(sum)
}

struct ExtractedEvent<'a> {
    activity: &'a str,
    resource: &'a str,
}
