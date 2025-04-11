use delegate::delegate;
use itertools::Itertools;
use pyo3::{
    exceptions::{self, PyIndexError, PyRuntimeError, PyTypeError},
    prelude::*,
    pybacked::PyBackedStr,
    types::{PyDict, PyTuple},
};
use spenso::{
    structure::{
        abstract_index::{AbstractIndex, AbstractIndexError},
        dimension::Dimension,
        representation::{ExtendibleReps, LibraryRep, Minkowski, RepName, Representation},
        slot::{IsAbstractSlot, Slot},
        HasName, IndexLess, NamedStructure, StructureContract, TensorStructure, ToSymbolic,
        VecStructure,
    },
    tensor_library::{ExplicitKey, ShadowedStructure, ETS},
};
use symbolica::{
    api::python::{ConvertibleToExpression, PythonExpression},
    atom::{Atom, AtomView, FunctionBuilder, NamespacedSymbol, Symbol},
    symbol,
};
use thiserror::Error;

use crate::physics::algebraic_simplification::{
    gamma::AGS, representations::Bispinor, IndexTooling,
};

use super::{
    library::{TensorNamespace, WEYL},
    ModuleInit, SliceOrIntOrExpanded,
};
use auto_enums::auto_enum;
use pyo3_stub_gen::derive::*;

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorIndices", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A structure that can be used to represent the "shape" of a tensor, along with a list of abstract indices.
/// This has an optional name, and accompanying symbolica expressions that are considered as additional non-indexed arguments.
/// The structure is essentially a list of `Slots` that are used to define the structure of the tensor.
pub struct SpensoIndices {
    pub structure: NamedStructure<Symbol, Vec<Atom>, LibraryRep>,
}

impl From<ShadowedStructure> for SpensoIndices {
    fn from(value: ShadowedStructure) -> Self {
        SpensoIndices { structure: value }
    }
}

impl ModuleInit for SpensoIndices {
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SpensoIndices>()?;
        m.add_class::<SpensoSlot>()?;
        m.add_class::<SpensoStucture>()?;
        m.add_class::<SpensoRepresentation>()?;
        Ok(())
    }
}

pub enum ArithmeticStructure {
    Convertible(ConvertibleToExpression),
    Structure(SpensoIndices),
    Expression(PythonExpression),
}

impl ArithmeticStructure {
    pub fn to_expression(self) -> PyResult<PythonExpression> {
        match self {
            ArithmeticStructure::Convertible(expr) => Ok(expr.to_expression()),
            ArithmeticStructure::Structure(indices) => indices.to_expression(),
            ArithmeticStructure::Expression(expr) => Ok(expr),
        }
    }
}

impl<'a> FromPyObject<'a> for ArithmeticStructure {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        if let Ok(ob) = ob.extract::<ConvertibleToExpression>() {
            Ok(ArithmeticStructure::Convertible(ob))
        } else if let Ok(ob) = ob.extract::<SpensoIndices>() {
            Ok(ArithmeticStructure::Structure(ob))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Only convertible expressions and spenso indices can be used",
            ))
        }
    }
}

#[pymethods]
impl SpensoIndices {
    #[new]
    #[pyo3(signature =
           (name,
           *additional_args))]
    pub fn from_list(
        name: PythonExpression,
        additional_args: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        let mut args = Vec::new();
        let mut slots = Vec::new();
        for a in additional_args {
            if let Ok(s) = a.extract::<SpensoSlot>() {
                slots.push(s.slot);
            } else if let Ok(arg) = a.extract::<PythonExpression>() {
                args.push(arg.expr.into());
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only slots and expressions can be used",
                ));
            }
        }

        let args = if args.is_empty() { None } else { Some(args) };

        let id = match name.expr.as_view() {
            AtomView::Var(v) => v.get_symbol(),
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only symbols can be used as names",
                ))
            }
        };

        Ok(SpensoIndices {
            structure: ShadowedStructure::from_iter(slots, id.into(), args),
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.structure)
    }

    fn __str__(&self) -> String {
        if let Some(atom) = self.structure.to_symbolic() {
            format!("{}", atom)
        } else {
            let args = self
                .structure
                .external_structure_iter()
                .map(|r| r.to_atom())
                .join(",");

            format!("({})", args.trim_end())
        }
    }

    fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(self
            .structure
            .to_symbolic()
            .ok_or(PyRuntimeError::new_err("No name"))?
            .into())
    }

    fn __len__(&self) -> usize {
        self.structure.size().unwrap()
    }

    fn __getitem__(&self, item: SliceOrIntOrExpanded) -> PyResult<Py<PyAny>> {
        match item {
            SliceOrIntOrExpanded::Int(i) => {
                let out: Vec<_> = self
                    .structure
                    .expanded_index(i.into())
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Expanded(idxs) => {
                let out: usize = self
                    .structure
                    .flat_index(&idxs)
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Slice(s) => {
                let r = s.indices(self.structure.size().unwrap() as isize)?;

                let start = if r.start < 0 {
                    (r.slicelength as isize + r.start) as usize
                } else {
                    r.start as usize
                };

                let end = if r.stop < 0 {
                    (r.slicelength as isize + r.stop) as usize
                } else {
                    r.stop as usize
                };

                let (range, step) = if r.step < 0 {
                    (end..start, -r.step as usize)
                } else {
                    (start..end, r.step as usize)
                };

                let slice: Result<Vec<Vec<usize>>, _> = range
                    .step_by(step)
                    .map(|i| {
                        self.structure
                            .expanded_index(i.into())
                            .map(Vec::<usize>::from)
                    })
                    .collect();

                match slice {
                    Ok(slice) => {
                        Ok(
                            Python::with_gil(|py| slice.into_pyobject(py).map(|a| a.unbind()))?
                                .into_any(),
                        )
                    }
                    Err(e) => Err(PyIndexError::new_err(e.to_string())),
                }
            }
        }
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression()?;
        Ok((self.to_expression()?.expr.as_ref() + rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression()?.__neg__()?;
        self.__add__(ArithmeticStructure::Expression(rhs))
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let s = self.to_expression()?.__neg__()?.expr;

        let r = rhs.to_expression()?.expr;
        Ok((r + s).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression()?;
        Ok((self.to_expression()?.expr.as_ref() * rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ArithmeticStructure) -> PyResult<PythonExpression> {
        self.__mul__(rhs)
    }

    /// Take `self` to power `exp`, returning the result.
    pub fn __pow__(
        &self,
        rhs: ArithmeticStructure,
        number: Option<i64>,
    ) -> PyResult<PythonExpression> {
        if number.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        let rhs = rhs.to_expression()?;
        Ok(self.to_expression()?.exp().pow(&rhs.expr).into())
    }
}

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "TensorStructure", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A structure that can be used to represent the "shape" of a tensor.
/// This has an optional name, and accompanying symbolica expressions that are considered as additional non-indexed arguments.
/// The structure is essentially a list of `Representation` that are used to define the structure of the tensor.
pub struct SpensoStucture {
    pub structure: ExplicitKey,
}

impl From<ExplicitKey> for SpensoStucture {
    fn from(value: ExplicitKey) -> Self {
        SpensoStucture { structure: value }
    }
}

#[derive(Clone)]
pub enum PossiblyIndexed {
    Unindexed(SpensoStucture),
    Indexed(SpensoIndices),
}

impl<'py> FromPyObject<'py> for PossiblyIndexed {
    fn extract_bound(structure: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(structure) = structure.extract::<SpensoIndices>() {
            Ok(PossiblyIndexed::from(structure))
        } else if let Ok(structure) = structure.extract::<SpensoStucture>() {
            Ok(PossiblyIndexed::from(structure))
        } else if let Ok(s) = structure.extract::<Vec<SpensoSlot>>() {
            Ok(PossiblyIndexed::Indexed(SpensoIndices {
                structure: VecStructure::from_iter(s.into_iter().map(|s| s.slot)).into(),
            }))
        } else if let Ok(s) = structure.extract::<Vec<SpensoRepresentation>>() {
            Ok(PossiblyIndexed::Unindexed(SpensoStucture {
                structure: IndexLess::from_iter(s.into_iter().map(|s| s.representation)).into(),
            }))
        } else if let Ok(s) = structure.extract::<Vec<usize>>() {
            Ok(PossiblyIndexed::Unindexed(SpensoStucture {
                structure: IndexLess::from_iter(
                    s.into_iter().map(|s| ExtendibleReps::EUCLIDEAN.rep(s)),
                )
                .into(),
            }))
        } else {
            Err(PyTypeError::new_err("Internal tensor structure can only be build from TensorIndices, TensorStructure, lists of Representations or of Slots"))
        }
    }
}

impl From<SpensoIndices> for PossiblyIndexed {
    fn from(s: SpensoIndices) -> Self {
        PossiblyIndexed::Indexed(s)
    }
}

impl From<SpensoStucture> for PossiblyIndexed {
    fn from(s: SpensoStucture) -> Self {
        PossiblyIndexed::Unindexed(s)
    }
}
impl From<ExplicitKey> for PossiblyIndexed {
    fn from(s: ExplicitKey) -> Self {
        PossiblyIndexed::Unindexed(s.into())
    }
}

impl From<ShadowedStructure> for PossiblyIndexed {
    fn from(s: ShadowedStructure) -> Self {
        PossiblyIndexed::Indexed(s.into())
    }
}

impl TensorStructure for PossiblyIndexed {
    type Slot = Slot<LibraryRep>;

    fn dual(self) -> Self {
        match self {
            Self::Indexed(i) => Self::Indexed(SpensoIndices {
                structure: i.structure.dual(),
            }),
            Self::Unindexed(i) => Self::Unindexed(SpensoStucture {
                structure: i.structure.dual(),
            }),
        }
    }
    delegate! {
        to match self {
            PossiblyIndexed::Indexed(u) => u.structure,
            PossiblyIndexed::Unindexed(u) => u.structure,
        }{
            #[auto_enum(Iterator)]
            fn external_structure_iter(&self) -> impl Iterator<Item =  Slot<LibraryRep>>;
            #[auto_enum(Iterator)]
            fn external_dims_iter(&self) -> impl Iterator<Item = Dimension>;
            #[auto_enum(Iterator)]
            fn external_reps_iter(
                &self,
            ) -> impl Iterator<Item = Representation<LibraryRep>>;
            #[auto_enum(Iterator)]
            fn external_indices_iter(&self) -> impl Iterator<Item = AbstractIndex>;
            fn get_aind(&self, i: usize) -> Option<AbstractIndex>;
            fn get_rep(&self, i: usize) -> Option<Representation<LibraryRep>>;
            fn get_dim(&self, i: usize) -> Option<Dimension>;
            fn get_slot(&self, i: usize) -> Option<Slot<LibraryRep>>;

            fn order(&self) -> usize;
        }
    }
}

impl HasName for PossiblyIndexed {
    type Name = Symbol;
    type Args = Vec<Atom>;

    delegate! {
        to match self {
            PossiblyIndexed::Indexed(i) => i.structure,
            PossiblyIndexed::Unindexed(u) => u.structure,
        }{
            fn name(&self) -> Option<Self::Name>;
            fn args(&self) -> Option<Self::Args>;
            fn set_name(&mut self, name: Self::Name);
        }
    }
}

impl StructureContract for PossiblyIndexed {
    fn concat(&mut self, other: &Self) {
        match self {
            PossiblyIndexed::Indexed(i) => {
                if let PossiblyIndexed::Indexed(j) = other {
                    i.structure.concat(&j.structure).into()
                } else {
                    panic!("Cannot merge indexed and unindexed structures")
                }
            }
            PossiblyIndexed::Unindexed(_) => {
                panic!("Cannot concat indexed and unindexed structures")
            }
        }
    }
    #[must_use]
    fn merge_at(&self, other: &Self, positions: (usize, usize)) -> Self {
        match self {
            PossiblyIndexed::Indexed(i) => {
                if let PossiblyIndexed::Indexed(j) = other {
                    i.structure.merge_at(&j.structure, positions).into()
                } else {
                    panic!("Cannot merge indexed and unindexed structures")
                }
            }
            PossiblyIndexed::Unindexed(_) => {
                panic!("Cannot merge indexed and unindexed structures")
            }
        }
    }

    fn merge(&mut self, other: &Self) -> Option<usize> {
        match self {
            PossiblyIndexed::Indexed(i) => {
                if let PossiblyIndexed::Indexed(j) = other {
                    i.structure.merge(&j.structure)
                } else {
                    panic!("Cannot merge indexed and unindexed structures")
                }
            }
            PossiblyIndexed::Unindexed(_) => {
                panic!("Cannot merge indexed and unindexed structures")
            }
        }
    }

    fn trace(&mut self, i: usize, j: usize) {
        match self {
            PossiblyIndexed::Indexed(s) => s.structure.trace(i, j),
            PossiblyIndexed::Unindexed(_) => panic!("cannot trace unindexed"),
        }
    }

    fn trace_out(&mut self) {
        match self {
            PossiblyIndexed::Indexed(s) => s.structure.trace_out(),
            PossiblyIndexed::Unindexed(_) => panic!("cannot trace uninidexed"),
        }
    }
}

#[derive(Error, Debug)]
pub enum SpensoError {
    #[error("Must have a name to register")]
    NoName,
}

impl TryFrom<PossiblyIndexed> for ExplicitKey {
    type Error = SpensoError;
    fn try_from(s: PossiblyIndexed) -> Result<Self, Self::Error> {
        match s {
            PossiblyIndexed::Indexed(i) => Ok(ExplicitKey::from_iter(
                i.structure.external_reps_iter(),
                i.structure.name().ok_or(SpensoError::NoName)?.into(),
                i.structure
                    .args()
                    .map(|a| a.into_iter().map(|a| a.into()).collect()),
            )),
            PossiblyIndexed::Unindexed(i) => Ok(ExplicitKey::from_iter(
                i.structure.external_reps_iter(),
                i.structure.name().ok_or(SpensoError::NoName)?.into(),
                i.structure
                    .args()
                    .map(|a| a.into_iter().map(|a| a.into()).collect()),
            )),
        }
    }
}

#[pymethods]
impl SpensoStucture {
    #[new]
    #[pyo3(signature =
           (
           *additional_args,name=None))]
    pub fn from_list(
        additional_args: &Bound<'_, PyTuple>,
        name: Option<PythonExpression>,
    ) -> PyResult<Self> {
        let mut args = Vec::new();
        let mut slots = Vec::new();
        for a in additional_args {
            if let Ok(s) = a.extract::<SpensoRepresentation>() {
                slots.push(s.representation);
            } else if let Ok(arg) = a.extract::<PythonExpression>() {
                args.push(arg.expr.into());
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only slots and expressions can be used",
                ));
            }
        }

        let args = if args.is_empty() { None } else { Some(args) };

        let mut a: ExplicitKey = IndexLess::from_iter(slots).into();
        if let Some(name) = name {
            match name.expr.as_view() {
                AtomView::Var(v) => a.set_name(v.get_symbol().into()),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can used as names",
                    ))
                }
            }
        };
        a.additional_args = args;

        Ok(SpensoStucture { structure: a })
    }

    fn __repr__(&self) -> String {
        format!("{}", self.structure.to_symbolic().unwrap())
    }

    fn __str__(&self) -> String {
        let slot = self
            .structure
            .external_reps()
            .into_iter()
            .map(|r| r.to_symbolic([]))
            .join(",");

        match (self.structure.name(), self.structure.args()) {
            (Some(name), Some(args)) => {
                let args = args.iter().join(",");
                format!("{}({})[{}]", name, args, slot)
            }
            (Some(name), None) => {
                format!("{}[{}]", name, slot)
            }
            (None, Some(args)) => {
                let args = args.iter().join(",");
                format!("({})[{}]", args, slot)
            }
            (None, None) => {
                format!("[{}]", slot)
            }
        }
    }

    fn __len__(&self) -> usize {
        self.structure.size().unwrap()
    }

    fn __getitem__(&self, item: SliceOrIntOrExpanded) -> PyResult<Py<PyAny>> {
        match item {
            SliceOrIntOrExpanded::Int(i) => {
                let out: Vec<_> = self
                    .structure
                    .expanded_index(i.into())
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Expanded(idxs) => {
                let out: usize = self
                    .structure
                    .flat_index(&idxs)
                    .map_err(|s| PyIndexError::new_err(s.to_string()))?
                    .into();

                Ok(Python::with_gil(|py| out.into_pyobject(py).map(|a| a.unbind()))?.into_any())
            }
            SliceOrIntOrExpanded::Slice(s) => {
                let r = s.indices(self.structure.size().unwrap() as isize)?;

                let start = if r.start < 0 {
                    (r.slicelength as isize + r.start) as usize
                } else {
                    r.start as usize
                };

                let end = if r.stop < 0 {
                    (r.slicelength as isize + r.stop) as usize
                } else {
                    r.stop as usize
                };

                let (range, step) = if r.step < 0 {
                    (end..start, -r.step as usize)
                } else {
                    (start..end, r.step as usize)
                };

                let slice: Result<Vec<Vec<usize>>, _> = range
                    .step_by(step)
                    .map(|i| {
                        self.structure
                            .expanded_index(i.into())
                            .map(Vec::<usize>::from)
                    })
                    .collect();

                match slice {
                    Ok(slice) => {
                        Ok(
                            Python::with_gil(|py| slice.into_pyobject(py).map(|a| a.unbind()))?
                                .into_any(),
                        )
                    }
                    Err(e) => Err(PyIndexError::new_err(e.to_string())),
                }
            }
        }
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let mut cook_indices = false;
        let mut to_atom = false;
        if let Some(kwargs) = kwargs {
            if let Some(cook) = kwargs.get_item("cook_indices")? {
                cook_indices = cook.extract::<bool>()?;
            }

            if let Some(to_atom_val) = kwargs.get_item("to_atom")? {
                to_atom = to_atom_val.extract::<bool>()?;
            }
        }

        if to_atom {
            let mut aind = Vec::new();
            for a in args {
                if let Ok(s) = a.extract::<isize>() {
                    aind.push(Atom::new_num(s as i64));
                } else if let Ok(arg) = a.extract::<PythonExpression>() {
                    aind.push(if cook_indices {
                        arg.expr.cook_indices()
                    } else {
                        arg.expr
                    });
                } else {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only integers and symbols can be used",
                    ));
                }
            }

            let slots = self
                .structure
                .external_reps_iter()
                .zip(aind)
                .map(|(rep, ind)| rep.to_symbolic([ind]))
                .collect::<Vec<_>>();

            let mut value_builder = FunctionBuilder::new(
                self.structure
                    .name()
                    .ok_or(PyRuntimeError::new_err("structure has no name"))?,
            );

            if let Some(args) = self.structure.args() {
                value_builder = value_builder.add_args(&args);
            }
            let expr = PythonExpression::from(value_builder.add_args(&slots).finish());
            return Ok(expr.into_pyobject(py).map(|a| a.unbind())?.into_any());
        }

        let mut aind: Vec<AbstractIndex> = Vec::new();
        for a in args {
            if let Ok(s) = a.extract::<isize>() {
                aind.push(s.into());
            } else if let Ok(arg) = a.extract::<PythonExpression>() {
                let ind = if cook_indices {
                    arg.expr.cook_indices().as_view().try_into()
                } else {
                    arg.expr.as_view().try_into()
                };

                aind.push(
                    ind.map_err(|a: AbstractIndexError| PyTypeError::new_err(a.to_string()))?,
                );
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only integers and symbols can be used",
                ));
            }
        }
        Ok(SpensoIndices {
            structure: self.structure.clone().to_indexed(&aind),
        }
        .into_pyobject(py)
        .map(|a| a.unbind())?
        .into_any())
    }

    #[staticmethod]
    pub fn id(rep: SpensoRepresentation) -> Self {
        ExplicitKey::from_iter(
            [rep.representation, rep.representation.dual()],
            ETS.id,
            None,
        )
        .into()
    }

    #[staticmethod]
    pub fn metric(rep: SpensoRepresentation) -> Self {
        ExplicitKey::from_iter([rep.representation, rep.representation], ETS.metric, None).into()
    }

    #[allow(non_snake_case)]
    #[staticmethod]
    //make it optional
    pub fn gamma4D(namespace: TensorNamespace) -> Self {
        let name = match namespace {
            TensorNamespace::Weyl => WEYL.gamma,
            TensorNamespace::Algebra => AGS.gamma,
        };

        ExplicitKey::from_iter(
            [
                LibraryRep::from(Minkowski {}).rep(4),
                Bispinor {}.rep(4).cast(),
                Bispinor {}.rep(4).cast(),
            ],
            name,
            None,
        )
        .into()
    }

    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn gammadD(dim: PythonExpression) -> PyResult<Self> {
        if let AtomView::Var(v) = dim.expr.as_view() {
            Ok(ExplicitKey::from_iter(
                [
                    LibraryRep::from(Minkowski {}).rep(v.get_symbol()),
                    Bispinor {}.rep(4).cast(),
                    Bispinor {}.rep(4).cast(),
                ],
                AGS.gamma,
                None,
            )
            .into())
        } else {
            return Err(exceptions::PyTypeError::new_err(
                "Only symbols can used as dims",
            ));
        }
    }

    #[staticmethod]
    pub fn gamma5(namespace: TensorNamespace) -> Self {
        let name = match namespace {
            TensorNamespace::Weyl => WEYL.gamma5,
            TensorNamespace::Algebra => AGS.gamma5,
        };

        ExplicitKey::from_iter([Bispinor {}.rep(4), Bispinor {}.rep(4)], name, None).into()
    }

    #[staticmethod]
    pub fn projm(namespace: TensorNamespace) -> Self {
        let name = match namespace {
            TensorNamespace::Algebra => AGS.projm,
            TensorNamespace::Weyl => WEYL.projm,
        };

        ExplicitKey::from_iter([Bispinor {}.rep(4), Bispinor {}.rep(4)], name, None).into()
    }

    #[staticmethod]
    pub fn projp(namespace: TensorNamespace) -> Self {
        let name = match namespace {
            TensorNamespace::Algebra => AGS.projp,
            TensorNamespace::Weyl => WEYL.projp,
        };

        ExplicitKey::from_iter([Bispinor {}.rep(4), Bispinor {}.rep(4)], name, None).into()
    }
}

#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "Representation", module = "symbolica_community.tensors")]
#[derive(Clone)]
/// A representation class in the sense of representation theory. This class is used to represent the representation of a tensor. It is essentially a pair of a name and a dimension.
/// New representations are registered when constructing.
/// Some representations are dualizable, meaning that they have a dual representation.
/// Indices will only ever match across dual representations.
/// There are some already registered representations, such as:
///  EUCLIDEAN: Rep = Rep::SelfDual(0);
///  BISPINOR: Rep = Rep::SelfDual(1);
///  COLORADJ: Rep = Rep::SelfDual(2);
///  MINKOWSKI: Rep = Rep::SelfDual(3);
///
///  LORENTZ_UP: Rep = Rep::Dualizable(1);
///  LORENTZ_DOWN: Rep = Rep::Dualizable(-1);
///  SPINFUND: Rep = Rep::Dualizable(2);
///  SPINANTIFUND: Rep = Rep::Dualizable(-2);
///  COLORFUND: Rep = Rep::Dualizable(3);
///  COLORANTIFUND: Rep = Rep::Dualizable(-3);
///  COLORSEXT: Rep = Rep::Dualizable(4);
///  COLORANTISEXT: Rep = Rep::Dualizable(-4);
///
pub struct SpensoRepresentation {
    pub representation: Representation<LibraryRep>,
}

// #[gen_stub_pymethods]
#[pymethods]
impl SpensoRepresentation {
    #[new]
    #[pyo3(signature =
           (
           name,dimension,is_self_dual=false))]
    /// Register a new representation with the given name and dimension. If dual is true, the representation will be dualizable, else it will be self-dual.
    pub fn register_new(
        name: Bound<'_, PyAny>,
        dimension: Bound<'_, PyAny>,
        is_self_dual: bool,
    ) -> PyResult<Self> {
        let name = name.extract::<PyBackedStr>()?;

        let dim = if let Ok(i) = dimension.extract::<usize>() {
            Dimension::from(i)
        } else if let Ok(expr) = dimension.extract::<PythonExpression>() {
            let id = match expr.expr.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can be abstract indices",
                    ))
                }
            };

            Dimension::from(id)
        } else if let Ok(s) = dimension.extract::<PyBackedStr>() {
            let ns = "spenso_python";
            let id = Symbol::new(NamespacedSymbol {
                symbol: format!("{}::{}", ns, s).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            })
            .build()
            .unwrap();

            Dimension::from(id)
        } else {
            return Err(PyTypeError::new_err(
                "dimension must be an non-zero integer or a symbol",
            ));
        };

        let rep = if is_self_dual {
            LibraryRep::new_self_dual(&name).unwrap().rep(dim)
        } else {
            LibraryRep::new_dual(&name).unwrap().rep(dim)
        };
        Ok(SpensoRepresentation {
            representation: rep,
        })
    }

    /// Generate a new slot with the given index, from this representation
    fn __call__(&self, aind: Bound<'_, PyAny>) -> PyResult<SpensoSlot> {
        if let Ok(i) = aind.extract::<isize>() {
            Ok(SpensoSlot {
                slot: self.representation.slot(i),
            })
        } else if let Ok(expr) = aind.extract::<PythonExpression>() {
            let id = match expr.expr.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can be abstract indices",
                    ))
                }
            };

            let aind = AbstractIndex::Symbol(id.into());
            Ok(SpensoSlot {
                slot: self.representation.slot(aind),
            })
        } else if let Ok(s) = aind.extract::<PyBackedStr>() {
            let id = symbol!(&s);

            Ok(SpensoSlot {
                slot: self.representation.slot(AbstractIndex::Symbol(id.into())),
            })
        } else {
            Err(PyTypeError::new_err("aind must be an integer or a symbol"))
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.representation)
    }

    fn __str__(&self) -> String {
        format!("{}", self.representation.to_symbolic([]))
    }

    fn to_expression(&self) -> PythonExpression {
        PythonExpression::from(self.representation.to_symbolic([]))
    }
}

/// An abstract index slot for a tensor.
/// This is essentially a tuple of a `Representation` and an abstract index id.
///
/// The abstract index id can be either an integer or a symbol.
/// This is the building block for creating tensor structures that can be contracted.
#[gen_stub_pyclass(module = "symbolica_community.tensors")]
#[pyclass(name = "Slot", module = "symbolica_community.tensors")]
#[derive(Clone)]
pub struct SpensoSlot {
    pub slot: Slot<LibraryRep>,
}

// #[gen_stub_pymethods]
#[pymethods]
impl SpensoSlot {
    fn __repr__(&self) -> String {
        format!("{:?}", self.slot)
    }

    fn __str__(&self) -> String {
        format!("{}", self.slot.to_atom())
    }

    #[new]
    #[pyo3(signature =
           (
           name,dimension,aind,dual=false))]
    /// Create a new slot from a name of a representation, a dimension and an abstract index.
    ///  If dual is true, the representation will be dualizable, else it will be self-dual.
    pub fn register_new(
        name: Bound<'_, PyAny>,
        dimension: usize,
        aind: Bound<'_, PyAny>,
        dual: bool,
    ) -> PyResult<Self> {
        let name = name.extract::<PyBackedStr>()?;
        let rep = if dual {
            LibraryRep::new_dual(&name).unwrap().rep(dimension)
        } else {
            LibraryRep::new_self_dual(&name).unwrap().rep(dimension)
        };
        if let Ok(i) = aind.extract::<isize>() {
            Ok(SpensoSlot { slot: rep.slot(i) })
        } else if let Ok(expr) = aind.extract::<PythonExpression>() {
            let id = match expr.expr.as_view() {
                AtomView::Var(v) => v.get_symbol(),
                _ => {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only symbols can be abstract indices",
                    ))
                }
            };

            let aind = AbstractIndex::Symbol(id.into());
            Ok(SpensoSlot {
                slot: rep.slot(aind),
            })
        } else if let Ok(s) = aind.extract::<PyBackedStr>() {
            let id = symbol!(&s);

            Ok(SpensoSlot {
                slot: rep.slot(AbstractIndex::Symbol(id.into())),
            })
        } else {
            Err(PyTypeError::new_err("aind must be an integer or a symbol"))
        }
    }

    fn to_expression(&self) -> PythonExpression {
        PythonExpression::from(self.slot.to_atom())
    }
}
