#![allow(uncommon_codepoints)]
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod physics;

#[cfg(feature = "python")]
#[pymodule]
fn symbolica_community(m: &Bound<'_, PyModule>) -> PyResult<()> {
    symbolica::api::python::create_symbolica_module(m)?;

    // register new components
    physics::initialize(m)?;

    Ok(())
}
