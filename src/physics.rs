#[cfg(feature = "vakint")]
pub mod vakint;
#[cfg(feature = "vakint")]
use vakint::{
    NumericalEvaluationResultWrapper, VakintEvaluationMethodWrapper, VakintExpressionWrapper,
    VakintWrapper,
};
#[cfg(feature = "algebraic_simplification")]
pub mod algebraic_simplification;
#[cfg(feature = "spenso")]
pub mod tensors;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult,
};

pub(crate) fn initialize(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "spenso")]
    {
        tensors::initialize_spenso(m)?;
    }

    #[cfg(feature = "algebraic_simplification")]
    {
        algebraic_simplification::initialize_alg_simp(m)?;
    }

    #[cfg(feature = "vakint")]
    {
        m.add_class::<VakintWrapper>()?;
        m.add_class::<NumericalEvaluationResultWrapper>()?;
        m.add_class::<VakintExpressionWrapper>()?;
        m.add_class::<VakintEvaluationMethodWrapper>()?;
    }
    Ok(())
}
