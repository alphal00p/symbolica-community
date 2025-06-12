use std::{ops::Neg, sync::LazyLock};

#[cfg(feature = "python")]
use pyo3::{exceptions::PyTypeError, pyclass, pymethods, PyResult};

#[cfg(feature = "python")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use spenso::{
    network::library::symbolic::{ExplicitKey, TensorLibrary},
    structure::HasStructure,
    tensors::parametric::MixedTensor,
};

use super::Spensor;

#[cfg(feature = "python")]
use super::ModuleInit;

#[cfg_attr(
    feature = "python",
    gen_stub_pyclass(module = "symbolica_community.tensors"),
    pyclass(name = "TensorLibrary", module = "symbolica_community.tensors")
)]
// #[derive(Clone)]
pub struct SpensorLibrary {
    pub(crate) library: TensorLibrary<MixedTensor<f64, ExplicitKey>>,
}

#[cfg(feature = "python")]
impl ModuleInit for SpensorLibrary {}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[pymethods]
impl SpensorLibrary {
    #[new]
    pub fn new() -> Self {
        let mut a = Self {
            library: TensorLibrary::new(),
        };
        a.library.update_ids();
        a
    }

    pub fn register(&mut self, tensor: Spensor) -> PyResult<()> {
        self.library.insert_explicit(
            tensor
                .tensor
                .clone()
                .map_structure_result(ExplicitKey::try_from)
                .map_err(|s| PyTypeError::new_err(s.to_string()))?,
        );
        Ok(())
    }

    #[staticmethod]
    pub fn weyl() -> Self {
        Self {
            library: weyl::weyl(1., 0.),
        }
    }
}

#[cfg_attr(
    feature = "python",
    gen_stub_pyclass_enum(module = "symbolica_community.tensors"),
    pyclass(eq, eq_int)
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorNamespace {
    Weyl,
    Algebra,
}
