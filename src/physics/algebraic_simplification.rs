use color::{color_conj_impl, ColorError, ColorSimplifier};
use gamma::{factor_conj_impl, gamma_conj_impl, pol_conj_impl, GammaSimplifier};
use metric::{
    cook_function_view, cook_indices_impl, list_dangling_impl, wrap_dummies_impl,
    wrap_indices_impl, CookingError, MetricSimplifier,
};
use pyo3::{
    exceptions::{PyRuntimeWarning, PyTypeError},
    pyfunction,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};
use representations::initialize;
use symbolica::{
    api::python::PythonExpression,
    atom::{Atom, AtomView, Symbol},
};

pub mod color;
pub mod gamma;
pub mod metric;
pub mod rep_symbols;
pub mod representations;

pub trait IndexTooling {
    fn wrap_indices(&self, header: Symbol) -> Atom;
    fn wrap_dummies(&self, header: Symbol) -> Atom;
    fn cook_indices(&self) -> Atom;
    fn cook_function(&self) -> Result<Atom, CookingError>;
    fn conj(&self) -> Atom;
    fn list_dangling(&self) -> Vec<Atom>;
}

impl IndexTooling for Atom {
    fn wrap_indices(&self, header: Symbol) -> Atom {
        self.as_view().wrap_indices(header)
    }
    fn wrap_dummies(&self, header: Symbol) -> Atom {
        self.as_view().wrap_dummies(header)
    }
    fn cook_indices(&self) -> Atom {
        self.as_view().cook_indices()
    }

    fn cook_function(&self) -> Result<Atom, CookingError> {
        self.as_view().cook_function()
    }
    fn conj(&self) -> Atom {
        self.as_view().conj()
    }
    fn list_dangling(&self) -> Vec<Atom> {
        self.as_view().list_dangling()
    }
}

impl<'a> IndexTooling for AtomView<'a> {
    fn conj(&self) -> Atom {
        factor_conj_impl(
            pol_conj_impl(gamma_conj_impl(color_conj_impl(*self).as_view()).as_view()).as_view(),
        )
    }

    fn cook_function(&self) -> Result<Atom, CookingError> {
        cook_function_view(*self)
    }
    fn wrap_indices(&self, header: Symbol) -> Atom {
        wrap_indices_impl(*self, header)
    }
    fn cook_indices(&self) -> Atom {
        cook_indices_impl(*self)
    }
    fn wrap_dummies(&self, header: Symbol) -> Atom {
        wrap_dummies_impl(*self, header)
    }
    fn list_dangling(&self) -> Vec<Atom> {
        list_dangling_impl(*self)
    }
}

#[pyfunction]
pub fn conj(self_: &PythonExpression) -> PythonExpression {
    self_.expr.conj().into()
}

#[pyfunction]
pub fn wrap_indices(self_: &PythonExpression, header: Symbol) -> PythonExpression {
    self_.expr.wrap_indices(header).into()
}

#[pyfunction]
pub fn cook_indices(self_: &PythonExpression) -> PythonExpression {
    self_.expr.cook_indices().into()
}

#[pyfunction]
pub fn cook_function(self_: &PythonExpression) -> PyResult<PythonExpression> {
    self_
        .expr
        .cook_function()
        .map_err(|a| PyTypeError::new_err(format!("cannot cook: {a:?}")))
        .map(|a| a.into())
}

#[pyfunction]
pub fn wrap_dummies(self_: &PythonExpression, header: Symbol) -> PythonExpression {
    self_.expr.wrap_dummies(header).into()
}

#[pyfunction]
pub fn list_dangling(self_: &PythonExpression) -> Vec<PythonExpression> {
    self_
        .expr
        .list_dangling()
        .into_iter()
        .map(|a| a.into())
        .collect()
}

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

pub(crate) fn initialize_alg_simp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    initialize();
    let child_module = PyModule::new(m.py(), "algebraic_simplification")?;

    // let expression_class = m.getattr("Expression")?;

    // Set all the methods on the retrieved class
    child_module.add_function(wrap_pyfunction!(simplify_gamma, m)?)?;
    child_module.add_function(wrap_pyfunction!(to_dots, m)?)?;
    child_module.add_function(wrap_pyfunction!(simplify_metrics, m)?)?;
    child_module.add_function(wrap_pyfunction!(simplify_color, m)?)?;
    child_module.add_function(wrap_pyfunction!(wrap_indices, m)?)?;
    child_module.add_function(wrap_pyfunction!(cook_indices, m)?)?;
    child_module.add_function(wrap_pyfunction!(cook_function, m)?)?;
    child_module.add_function(wrap_pyfunction!(wrap_dummies, m)?)?;
    child_module.add_function(wrap_pyfunction!(list_dangling, m)?)?;
    child_module.add_function(wrap_pyfunction!(conj, m)?)?;

    m.add_submodule(&child_module)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("symbolica_community.algebraic_simplification", child_module)?;

    Ok(())
}
