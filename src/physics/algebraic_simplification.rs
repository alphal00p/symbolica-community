use color::{ColorError, ColorSimplifier};
use gamma::GammaSimplifier;
use metric::MetricSimplifier;
use pyo3::{
    exceptions::PyRuntimeWarning,
    pyfunction,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};
use symbolica::api::python::PythonExpression;

pub mod color;
pub mod gamma;
pub mod metric;
pub mod rep_symbols;
pub mod representations;

#[pyfunction]
pub fn simplify_gamma(expr: &PythonExpression) -> PythonExpression {
    expr.expr.simplify_gamma().into()
}

#[pyfunction]
pub fn to_dots(expr: &PythonExpression) -> PythonExpression {
    expr.expr.to_dots().into()
}

#[pyfunction]
pub fn simplify_metrics(expr: &PythonExpression) -> PythonExpression {
    expr.expr.simplify_metrics().into()
}

#[pyfunction]
pub fn simplify_color(expr: &PythonExpression) -> PyResult<PythonExpression> {
    expr.expr.simplify_color().map(|a| a.into()).map_err(|a| {
        PyRuntimeWarning::new_err(match a {
            ColorError::NotFully(a) => format!("Not fully simplified: {}", a),
        })
    })
}

pub(crate) fn initialize_alg_simp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(m.py(), "algebraic_simplification")?;

    m.getattr("Expression")?
        .setattr("simplify_gamma", wrap_pyfunction!(simplify_gamma, m)?)?;
    m.getattr("Expression")?
        .setattr("to_dots", wrap_pyfunction!(to_dots, m)?)?;
    m.getattr("Expression")?
        .setattr("simplify_metrics", wrap_pyfunction!(simplify_metrics, m)?)?;
    m.getattr("Expression")?
        .setattr("simplify_color", wrap_pyfunction!(simplify_color, m)?)?;

    m.add_submodule(&child_module)?;

    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("symbolica_community.algebraic_simplification", child_module)
}
