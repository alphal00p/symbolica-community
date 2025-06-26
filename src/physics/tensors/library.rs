use std::{ops::Neg, sync::LazyLock};

use anyhow::anyhow;
use pyo3::{exceptions, pybacked::PyBackedStr, types::PyAnyMethods, Bound, FromPyObject};
#[cfg(feature = "python")]
use pyo3::{exceptions::PyTypeError, pyclass, pymethods, PyResult};

#[cfg(feature = "python")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use pyo3_stub_gen::PyStubType;
use spenso::{
    network::library::symbolic::{ExplicitKey, TensorLibrary},
    structure::{HasStructure, PermutedStructure},
    tensors::parametric::MixedTensor,
};
use spenso_hep_lib::hep_lib;
use symbolica::{
    api::python::{ConvertibleToExpression, PythonExpression},
    atom::{AtomView, Symbol},
    symbol, try_symbol,
};

use super::{library_tensor::LibrarySpensor, structure::SpensoStructure, Spensor};

#[cfg(feature = "python")]
use super::ModuleInit;

#[cfg_attr(
    feature = "python",
    gen_stub_pyclass(module = "symbolica_community.tensors"),
    pyclass(name = "TensorLibrary", module = "symbolica_community.tensors")
)]
pub struct SpensorLibrary {
    pub(crate) library: TensorLibrary<MixedTensor<f64, ExplicitKey>>,
}

#[cfg(feature = "python")]
impl ModuleInit for SpensorLibrary {}

pub enum ConvertibleToSymbol {
    Name(String),
    Symbol(PythonExpression),
}

impl ConvertibleToSymbol {
    fn symbol(&self) -> anyhow::Result<Symbol> {
        match self {
            ConvertibleToSymbol::Name(name) => Ok(try_symbol!(name).map_err(|e| anyhow!(e))?),
            ConvertibleToSymbol::Symbol(symbol) => {
                if let AtomView::Var(a) = symbol.as_view() {
                    Ok(a.get_symbol())
                } else {
                    Err(anyhow::anyhow!("Symbol is not a variable"))
                }
            }
        }
    }
}

#[cfg(feature = "python")]
impl<'a> FromPyObject<'a> for ConvertibleToSymbol {
    fn extract_bound(ob: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<String>() {
            Ok(ConvertibleToSymbol::Name(a))
        } else if let Ok(num) = ob.extract::<PythonExpression>() {
            Ok(ConvertibleToSymbol::Symbol(num))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        }
    }
}

#[cfg(feature = "python")]
impl PyStubType for ConvertibleToSymbol {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        ConvertibleToExpression::type_output() | String::type_input()
    }
}

pub struct ConvertibleToLibraryTensor(LibrarySpensor);

#[cfg(feature = "python")]
impl<'a> FromPyObject<'a> for ConvertibleToLibraryTensor {
    fn extract_bound(ob: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<LibrarySpensor>() {
            Ok(ConvertibleToLibraryTensor(a))
        } else if let Ok(num) = ob.extract::<Spensor>() {
            Ok(ConvertibleToLibraryTensor(LibrarySpensor {
                tensor: num.tensor.map_structure(|a| a.map_structure(Into::into)),
            }))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to library tensor",
            ))
        }
    }
}

#[cfg(feature = "python")]
impl PyStubType for ConvertibleToLibraryTensor {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        Spensor::type_output() | LibrarySpensor::type_input()
    }
}

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

    pub fn register(&mut self, tensor: ConvertibleToLibraryTensor) -> PyResult<()> {
        self.library.insert_explicit(tensor.0.tensor);
        Ok(())
    }

    pub fn __getitem__(&self, key: ConvertibleToSymbol) -> anyhow::Result<SpensoStructure> {
        let symbol = key.symbol()?;
        let key = self.library.get_key_from_name(symbol)?;

        Ok(SpensoStructure {
            structure: PermutedStructure::identity(key),
        })
    }

    #[staticmethod]
    pub fn hep_lib() -> Self {
        Self {
            library: spenso_hep_lib::hep_lib(1., 0.),
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
