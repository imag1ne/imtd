use std::collections::HashSet;
use std::hash::Hash;

use crate::pm4py::obj::Event;
use counter::Counter;
use pyo3::types::PyAnyMethods;
use pyo3::{pyfunction, Bound, FromPyObject, PyAny, PyResult};
use rayon::prelude::*;

const DEFAULT_NAME_KEY: &str = "concept:name";
const DEFAULT_RESOURCE_KEY: &str = "org:resource";

#[derive(Default)]
pub struct Perspectives(Vec<Perspective>);

impl FromPyObject<'_> for Perspectives {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let perspectives = ob.extract::<Vec<(&str, f64)>>()?;
        let perspectives_length = perspectives.len() as f64;
        let weight_sum = perspectives.iter().map(|(_, w)| w).sum::<f64>();
        let perspectives = perspectives
            .into_iter()
            .map(|(p, w)| {
                let weight = if weight_sum == 0.0 {
                    1.0 / perspectives_length
                } else {
                    w / weight_sum
                };
                let perspective = match p {
                    "activity" => Perspective::Activity { weight },
                    "transition" => Perspective::Transition { weight },
                    "resource" => Perspective::Resource { weight },
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Invalid perspective",
                        ))
                    }
                };
                Ok(perspective)
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Perspectives(perspectives))
    }
}

impl Perspectives {
    fn distance(&self, trace1: &[ExtractedEvent], trace2: &[ExtractedEvent]) -> f64 {
        self.0
            .iter()
            .map(|perspective| perspective.distance(trace1, trace2))
            .sum()
    }
}

enum Perspective {
    Activity { weight: f64 },
    Transition { weight: f64 },
    Resource { weight: f64 },
}

impl Perspective {
    fn distance(&self, trace1: &[ExtractedEvent], trace2: &[ExtractedEvent]) -> f64 {
        match self {
            Perspective::Activity { weight } => {
                if *weight == 0.0 {
                    return 0.0;
                }

                let activity_distance = activity_distance(trace1, trace2);
                activity_distance * weight
            }
            Perspective::Transition { weight } => {
                if *weight == 0.0 {
                    return 0.0;
                }

                let transition_distance = transition_distance(trace1, trace2);
                transition_distance * weight
            }
            Perspective::Resource { weight } => {
                if *weight == 0.0 {
                    return 0.0;
                }

                let resource_distance = resource_distance(trace1, trace2);
                resource_distance * weight
            }
        }
    }
}

impl FromPyObject<'_> for Perspective {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (p, w) = ob.extract::<(&str, f64)>()?;
        let perspective = match p {
            "activity" => Perspective::Activity { weight: w },
            "transition" => Perspective::Transition { weight: w },
            "resource" => Perspective::Resource { weight: w },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid perspective",
                ))
            }
        };

        Ok(perspective)
    }
}

#[pyfunction]
#[pyo3(signature = (desirable_log, undesirable_log, weights=None, activity_key=None, resource_key=None))]
pub fn distance_matrix<'py>(
    desirable_log: Vec<Vec<Event<'py>>>,
    undesirable_log: Vec<Vec<Event<'py>>>,
    weights: Option<Perspectives>,
    activity_key: Option<&str>,
    resource_key: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let weights = weights.unwrap_or_default();
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
                            .map(|trace_p_slice| weights.distance(trace_p_slice, trace_m))
                            .reduce(f64::min)
                            .unwrap()
                    } else {
                        weights.distance(trace_p, trace_m)
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
