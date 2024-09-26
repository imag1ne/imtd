use pyo3::types::PyAnyMethods;
use pyo3::{Bound, FromPyObject, PyAny, PyResult};

pub struct Event<'py>(Bound<'py, PyAny>);

impl<'py> Event<'py> {
    pub fn get_str(&self, key: &str) -> PyResult<&'py str> {
        self.0.get_item(key).and_then(|item| item.extract::<&str>())
    }
}

impl<'py> FromPyObject<'py> for Event<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        ob.extract().map(Event)
    }
}
