use std::collections::{HashMap, HashSet};

use pyo3::{
    pyfunction,
    types::{PyAny, PyAnyMethods},
    Bound, PyResult,
};

#[pyfunction]
#[pyo3(signature = (event_log, case_id_key=None, activity_key=None))]
pub fn edge_case_id_mapping<'a>(
    event_log: Vec<Bound<'a, PyAny>>,
    case_id_key: Option<&str>,
    activity_key: Option<&str>,
) -> PyResult<HashMap<(&'a str, &'a str), HashSet<&'a str>>> {
    let case_id_key = case_id_key.unwrap_or("concept:name");
    let activity_key = activity_key.unwrap_or("concept:name");

    let mut edge_case_id_map: HashMap<(&str, &str), HashSet<&str>> = HashMap::new();
    for trace in event_log {
        let case_id = trace
            .getattr("attributes")?
            .get_item(case_id_key)?
            .extract::<&str>()?;
        let trace = trace.extract::<Vec<Bound<'a, PyAny>>>()?;
        for event_pair in trace.windows(2) {
            let source_activity = event_pair[0].get_item(activity_key)?.extract::<&str>()?;
            let target_activity = event_pair[1].get_item(activity_key)?.extract::<&str>()?;
            let edge = (source_activity, target_activity);
            edge_case_id_map.entry(edge).or_default().insert(case_id);
        }
    }

    Ok(edge_case_id_map)
}
