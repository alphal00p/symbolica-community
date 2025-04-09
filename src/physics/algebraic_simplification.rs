use color::{ColorError, ColorSimplifier};
use gamma::GammaSimplifier;
use metric::MetricSimplifier;
use pyo3::{
    exceptions::PyRuntimeWarning,
    ffi::c_str,
    pyclass, pyfunction, pymethods,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, IntoPyObject, PyAny, PyRef, PyResult, PyTypeInfo, Python,
};
use symbolica::api::python::PythonExpression;

pub mod color;
pub mod gamma;
pub mod metric;
pub mod rep_symbols;
pub mod representations;
#[pyfunction]
pub fn simplify_gamma(self_: &PythonExpression) -> PythonExpression {
    self_.expr.simplify_gamma().into()
}

#[pyfunction]
pub fn to_dots(self_: &PythonExpression) -> PythonExpression {
    self_.expr.to_dots().into()
}

#[pyfunction]
pub fn simplify_metrics(self_: &PythonExpression) -> PythonExpression {
    self_.expr.simplify_metrics().into()
}

#[pyfunction]
pub fn simplify_color(self_: &PythonExpression) -> PyResult<PythonExpression> {
    self_.expr.simplify_color().map(|a| a.into()).map_err(|a| {
        PyRuntimeWarning::new_err(match a {
            ColorError::NotFully(a) => format!("Not fully simplified: {}", a),
        })
    })
}
/// Shorthand notation for :func:`Expression.parse`.
// #[pyfunction(name = "EA", signature = (expr,default_namespace="python"))]
// fn expression_shorthand(
//     expr: &str,
//     default_namespace: &str,
//     py: Python,
// ) -> PyResult<(AlgebraicSimplification, PythonExpression)> {
//     let expr =
//         PythonExpression::parse(&PythonExpression::type_object(py), expr, default_namespace)?;
//     Ok((AlgebraicSimplification::default(), expr))
// }

// #[pyclass(extends=PythonExpression, subclass)]
// #[derive(Clone, Copy, IntoPyObject, Default)]
// struct AlgebraicSimplification {
//     _private: (),
// }

// #[pymethods]
// impl AlgebraicSimplification {
//     #[new]
//     pub fn new() -> (Self, PythonExpression) {
//         (Self::default(), PythonExpression::__new__())
//     }

//     pub fn simplify_gamma(self_: PyRef<'_, Self>) -> (Self, PythonExpression) {
//         (
//             AlgebraicSimplification::default(),
//             self_.as_super().expr.simplify_gamma().into(),
//         )
//     }

//     pub fn to_dots(self_: PyRef<'_, Self>) -> (Self, PythonExpression) {
//         (
//             AlgebraicSimplification::default(),
//             self_.as_super().expr.to_dots().into(),
//         )
//     }

//     pub fn simplify_metrics(self_: PyRef<'_, Self>) -> (Self, PythonExpression) {
//         (
//             AlgebraicSimplification::default(),
//             self_.as_super().expr.simplify_metrics().into(),
//         )
//     }

//     pub fn simplify_color(self_: PyRef<'_, Self>) -> PyResult<(Self, PythonExpression)> {
//         self_
//             .as_super()
//             .expr
//             .simplify_color()
//             .map(|a| (Self::default(), a.into()))
//             .map_err(|a| {
//                 PyRuntimeWarning::new_err(match a {
//                     ColorError::NotFully(a) => format!("Not fully simplified: {}", a),
//                 })
//             })
//     }
// }

// pub(crate) fn initialize_alg_simp(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     let child_module = PyModule::new(m.py(), "algebraic_simplification")?;
//     child_module.add_class::<AlgebraicSimplification>()?;
//     child_module.add_function(wrap_pyfunction!(expression_shorthand, m)?)?;
//     m.add_submodule(&child_module)?;

//     m.py()
//         .import("sys")?
//         .getattr("modules")?
//         .set_item("symbolica_community.algebraic_simplification", child_module)
// }
pub(crate) fn initialize_alg_simp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(m.py(), "algebraic_simplification")?;

    let expression_class = m.getattr("Expression")?;

    // Set all the methods on the retrieved class
    expression_class.setattr("simplify_gamma", wrap_pyfunction!(simplify_gamma, m)?)?;
    expression_class.setattr("to_dots", wrap_pyfunction!(to_dots, m)?)?;
    expression_class.setattr("simplify_metrics", wrap_pyfunction!(simplify_metrics, m)?)?;
    expression_class.setattr("simplify_color", wrap_pyfunction!(simplify_color, m)?)?;
    m.add_submodule(&child_module)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("symbolica_community.algebraic_simplification", child_module)?;

    Ok(())
}
