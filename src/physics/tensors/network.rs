use pyo3::{exceptions::PyRuntimeError, prelude::*};

use spenso::{
    network::{
        library::symbolic::ExplicitKey, parsing::ShadowedStructure, store::NetworkStore, Network,
        Sequential, SmallestDegree,
    },
    parametric::MixedTensor,
    structure::HasStructure,
};
use symbolica::{api::python::PythonExpression, atom::Atom};

use super::{library::SpensorLibrary, structure::PossiblyIndexed, ModuleInit, Spensor};
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
    pub network: Network<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, ExplicitKey>,
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

#[gen_stub_pyfunction]
#[pyfunction(name = "to_net")]
pub fn python_to_tensor_network(
    a: &Bound<'_, PythonExpression>,
    library: &SpensorLibrary,
) -> anyhow::Result<SpensoNet> {
    SpensoNet::from_expression(a, library)
}

pub type ParsingNet = Network<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, ExplicitKey>;

#[gen_stub_pymethods]
#[pymethods]
impl SpensoNet {
    #[new]
    /// Parses an expression into a
    pub fn from_expression(
        expr: &Bound<'_, PythonExpression>,
        library: &SpensorLibrary,
    ) -> anyhow::Result<SpensoNet> {
        Ok(SpensoNet {
            network: ParsingNet::try_from_view(expr.borrow().expr.as_view(), &library.library)?,
        })
    }

    fn execute(&mut self, library: &SpensorLibrary) -> PyResult<()> {
        self.network
            .execute::<Sequential, SmallestDegree, _>(&library.library)
            .map_err(|a| PyRuntimeError::new_err(a.to_string()))
    }

    fn result_tensor(&self, library: &SpensorLibrary) -> PyResult<Spensor> {
        Ok(Spensor {
            tensor: self
                .network
                .result_tensor(&library.library)
                .map_err(|s| PyRuntimeError::new_err(s.to_string()))?
                .into_owned()
                .map_structure(PossiblyIndexed::from),
        })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.network.dot_display_impl(
            |a| a.to_plain_string(),
            |l| Some(l.global_name?.to_string()),
            |t| t.to_string(),
        ))
    }
}
