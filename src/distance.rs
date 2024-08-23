use std::collections::HashSet;
use std::hash::Hash;

use rayon::prelude::*;
use counter::Counter;
use pyo3::{pyfunction, types::{PyAnyMethods, PyList}, Bound, PyResult};

#[pyfunction]
pub fn distance_matrix(event_log_1: &Bound<'_, PyList>, event_log_2: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f64>>> {
    let event_log_1 = event_log_1.extract::<Vec<Vec<(String, String)>>>()?;
    let event_log_2 = event_log_2.extract::<Vec<Vec<(String, String)>>>()?;

    let matrix = event_log_1.par_iter().map(|trace_1| {
        event_log_2.par_iter().map(|trace_2| {
            let activity_distance = activity_distance(trace_1, trace_2);
            let transition_distance = transition_distance(trace_1, trace_2);
            let resource_distance = resource_distance(trace_1, trace_2);
            activity_distance + transition_distance + resource_distance
        }).collect()
    }).collect();

    Ok(matrix)
}

pub fn activity_distance(trace1: &[(String,String)], trace2: &[(String, String)]) -> f64 {
    let activity_count1 = trace1.iter().map(|(a, _)| a).collect();
    let activity_count2 = trace2.iter().map(|(a, _)| a).collect();

    profile_distance(&activity_count1, &activity_count2)
}

pub fn transition_distance(trace1: &[(String,String)], trace2: &[(String,String)]) -> f64 {
    let transition_count1 = transition_counter_from_trace(trace1);
    let transition_count2 = transition_counter_from_trace(trace2);

    profile_distance(&transition_count1, &transition_count2)
}

pub fn resource_distance(trace1: &[(String, String)], trace2: &[(String, String)]) -> f64 {
    let resource_count1 = trace1.iter().collect();
    let resource_count2 = trace2.iter().collect();

    profile_distance(&resource_count1, &resource_count2)
}

fn transition_counter_from_trace(trace: &[(String,String)]) -> Counter<(&String, &String)> {
    trace.windows(2).map(|events| (&events[0].0, &events[1].0)).collect()
}

fn profile_distance<T>(profile1: &Counter<T>, profile2: &Counter<T>) -> f64  where T: Hash + Eq + Copy{
    let profile1_keys: HashSet<T> = profile1.keys().copied().collect();
    let profile2_keys: HashSet<T> = profile2.keys().copied().collect();
    let all_keys_iter = profile1_keys.union(&profile2_keys);

    let sum = all_keys_iter.map(|key| {
        let x = *profile1.get(key).unwrap_or(&0) as isize;
        let y = *profile2.get(key).unwrap_or(&0) as isize;
        (x - y).pow(2)
     }).sum::<isize>() as f64;

    f64::sqrt(sum)
}
