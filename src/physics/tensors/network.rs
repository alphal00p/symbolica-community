use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
};

use spenso::{
    network::TensorNetwork,
    parametric::MixedTensor,
    tensor_library::{ExplicitKey, ShadowedStructure, TensorLibrary},
};
use symbolica::{api::python::PythonExpression, atom::Atom};

use super::{structure::PossiblyIndexed, ModuleInit, Spensor};
use pyo3_stub_gen::derive::*;

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorNetwork", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A tensor network.
///
/// This class is a wrapper around the `TensorNetwork` class from the `spenso` crate.
/// Such a network is a graph representing the arithmetic operations between tensors.
/// In the most basic case, edges represent the contraction of indices.
pub struct SpensoNet {
    pub network: TensorNetwork<MixedTensor<f64, ShadowedStructure>, Atom>,
}

impl ModuleInit for SpensoNet {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SpensoNet>()
    }

    fn append_to_symbolica(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.getattr("Expression")?
            .setattr("to_net", wrap_pyfunction!(python_to_tensor_network, m)?)
    }
}

#[pyfunction(name = "to_net")]
pub fn python_to_tensor_network(
    a: &Bound<'_, PythonExpression>,
    library: &SpensorLibrary,
) -> anyhow::Result<SpensoNet> {
    SpensoNet::from_expression(a, library)
}

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorLibrary", module = "symbolica_community.tensors")]
// #[derive(Clone)]
pub struct SpensorLibrary {
    library: TensorLibrary<MixedTensor<f64, ExplicitKey>>,
}

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
                .map_structure_fallible(ExplicitKey::try_from)
                .map_err(|s| PyTypeError::new_err(s.to_string()))?,
        );
        Ok(())
    }
}

pub type ParsingNet = TensorNetwork<MixedTensor<f64, ShadowedStructure>, Atom>;

#[pymethods]
impl SpensoNet {
    #[new]
    pub fn from_expression(
        expr: &Bound<'_, PythonExpression>,
        library: &SpensorLibrary,
    ) -> anyhow::Result<SpensoNet> {
        Ok(SpensoNet {
            network: ParsingNet::try_from_view(expr.borrow().expr.as_view(), &library.library)?,
        })
    }

    fn contract(&mut self) -> PyResult<()> {
        self.network.contract();
        Ok(())
    }

    fn result(&self) -> PyResult<Spensor> {
        Ok(Spensor {
            tensor: self
                .network
                .result_tensor_smart()
                .map_err(|s| PyRuntimeError::new_err(s.to_string()))?
                .map_structure(PossiblyIndexed::from),
        })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.network.rich_graph().dot())
    }
}
