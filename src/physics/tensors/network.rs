use std::ops::Deref;

#[cfg(feature = "python")]
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
};

use spenso::{
    network::{
        library::symbolic::ExplicitKey, parsing::ShadowedStructure, store::NetworkStore,
        ExecutionResult, Network, Sequential, SingleSmallestDegree, SmallestDegree, Steps,
    },
    structure::HasName,
    tensors::parametric::MixedTensor,
};
use spenso_hep_lib::HEP_LIB;
use symbolica::atom::Atom;

#[cfg(feature = "python")]
use symbolica::api::python::ConvertibleToExpression;

use super::{library::SpensorLibrary, Spensor};

#[cfg(feature = "python")]
use super::ModuleInit;
#[cfg(feature = "python")]
use pyo3_stub_gen::{derive::*, PyStubType};

#[cfg_attr(
    feature = "python",
    gen_stub_pyclass(module = "symbolica_community.tensors"),
    pyclass(name = "TensorNetwork", module = "symbolica_community.tensors")
)]
#[derive(Clone)]
/// A tensor network.
///
/// This class is a wrapper around the `TensorNetwork` class from the `spenso` crate.
/// Such a network is a graph representing the arithmetic operations between tensors.
/// In the most basic case, edges represent the contraction of indices.
pub struct SpensoNet {
    pub network: Network<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, ExplicitKey>,
}

#[cfg(feature = "python")]
impl ModuleInit for SpensoNet {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SpensoNet>()
    }

    fn append_to_symbolica(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.getattr("Expression")?
            .setattr("to_net", wrap_pyfunction!(python_to_tensor_network, m)?)
    }
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction]
#[pyfunction(name = "to_net")]
pub fn python_to_tensor_network(
    a: ConvertibleToExpression,
    library: Option<&SpensorLibrary>,
) -> anyhow::Result<SpensoNet> {
    SpensoNet::from_expression(a, library)
}

pub type ParsingNet = Network<NetworkStore<MixedTensor<f64, ShadowedStructure>, Atom>, ExplicitKey>;

impl From<ParsingNet> for SpensoNet {
    fn from(network: ParsingNet) -> Self {
        SpensoNet { network }
    }
}

pub struct ConvertibleToSpensoNet(SpensoNet);

impl ConvertibleToSpensoNet {
    pub fn to_net(self) -> SpensoNet {
        self.0
    }
}

#[cfg(feature = "python")]
impl<'a> FromPyObject<'a> for ConvertibleToSpensoNet {
    fn extract_bound(ob: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<SpensoNet>() {
            Ok(ConvertibleToSpensoNet(a))
        } else if let Ok(num) = ob.extract::<Spensor>() {
            Ok(ConvertibleToSpensoNet(SpensoNet {
                network: Network::from_tensor(num.tensor.structure),
            }))
        } else if let Ok(a) = ob.extract::<ConvertibleToExpression>() {
            Ok(ConvertibleToSpensoNet(SpensoNet {
                network: ParsingNet::try_from_view(
                    a.to_expression().expr.as_view(),
                    &SpensorLibrary::new().library,
                )
                .map_err(|a| PyRuntimeError::new_err(a.to_string()))?,
            }))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        }
    }
}

#[cfg(feature = "python")]
impl PyStubType for ConvertibleToSpensoNet {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        ConvertibleToExpression::type_output() | SpensoNet::type_output() | Spensor::type_output()
    }
}

// #[gen_stub_pymethods]

#[cfg(feature = "python")]
#[pymethods]
impl SpensoNet {
    #[new]
    /// Parses an expression into a network
    #[pyo3(signature = (expr, library=None))]
    pub fn from_expression(
        expr: ConvertibleToExpression,
        library: Option<&SpensorLibrary>,
    ) -> anyhow::Result<SpensoNet> {
        let lib = library.map(|l| &l.library).unwrap_or(HEP_LIB.deref());

        Ok(SpensoNet {
            network: ParsingNet::try_from_view(expr.to_expression().as_view(), lib)?,
        })
    }

    #[staticmethod]
    pub fn one() -> SpensoNet {
        SpensoNet {
            network: Network::one(),
        }
    }

    #[staticmethod]
    pub fn zero() -> SpensoNet {
        SpensoNet {
            network: Network::zero(),
        }
    }

    #[pyo3(signature = (library=None, n_steps=None, single_contract=false))]
    fn execute(
        &mut self,
        library: Option<&SpensorLibrary>,
        n_steps: Option<usize>,
        single_contract: bool,
    ) -> PyResult<()> {
        let lib = library.map(|l| &l.library).unwrap_or(HEP_LIB.deref());

        if let Some(n) = n_steps {
            for _ in 0..n {
                if single_contract {
                    self.network
                        .execute::<Steps<1>, SingleSmallestDegree<false>, _, _>(lib)
                        .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                } else {
                    self.network
                        .execute::<Steps<1>, SmallestDegree, _, _>(lib)
                        .map_err(|a| PyRuntimeError::new_err(a.to_string()))?;
                }
            }
            Ok(())
        } else {
            if single_contract {
                self.network
                    .execute::<Sequential, SingleSmallestDegree<false>, _, _>(lib)
                    .map_err(|a| PyRuntimeError::new_err(a.to_string()))
            } else {
                self.network
                    .execute::<Sequential, SmallestDegree, _, _>(lib)
                    .map_err(|a| PyRuntimeError::new_err(a.to_string()))
            }
        }
    }
    #[pyo3(signature = (library=None))]
    fn result_tensor(&self, library: Option<&SpensorLibrary>) -> PyResult<Spensor> {
        let lib = library.map(|l| &l.library).unwrap_or(HEP_LIB.deref());

        Ok(
            match self
                .network
                .result_tensor(lib)
                .map_err(|s| PyRuntimeError::new_err(s.to_string()))?
            {
                ExecutionResult::One => Spensor::one(),
                ExecutionResult::Zero => Spensor::zero(),
                ExecutionResult::Val(v) => v.into_owned().into(),
            },
        )
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.network.dot_display_impl(
            |a| a.to_plain_string(),
            |l| Some(l.global_name?.to_string()),
            |t| {
                t.name()
                    .map(|a| a.to_string())
                    .unwrap_or("unnamed".to_owned())
            },
        ))
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((self.network.clone() + rhs.network).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((self.network.clone() - rhs.network).into())
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((rhs.network - self.network.clone()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((rhs.network * self.network.clone()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToSpensoNet) -> PyResult<SpensoNet> {
        let rhs = rhs.to_net();
        Ok((rhs.network * self.network.clone()).into())
    }

    // pub fn __pow__(&self, rhs: usize, number: Option<i64>) -> PyResult<PythonExpression> {
    //     if number.is_some() {
    //         return Err(exceptions::PyValueError::new_err(
    //             "Optional number argument not supported",
    //         ));
    //     }

    //     // let rhs = rhs.to_net();
    //     Ok(self.network.pow(&rhs).into())
    // }
}
