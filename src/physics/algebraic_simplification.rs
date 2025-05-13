use color::{color_conj_impl, ColorError, ColorSimplifier, SelectiveExpand};
use gamma::{factor_conj_impl, gamma_conj_impl, pol_conj_impl, GammaSimplifier};
use metric::{
    cook_function_view, cook_indices_impl, list_dangling_impl, wrap_dummies_impl,
    wrap_indices_impl, CookingError, MetricSimplifier,
};

#[cfg(feature = "python")]
use pyo3::{
    exceptions::{PyRuntimeWarning, PyTypeError},
    pyfunction,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};

#[cfg(feature = "python")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use representations::initialize;
use symbolica::atom::{Atom, AtomView, Symbol};

#[cfg(feature = "python")]
use symbolica::api::python::PythonExpression;

pub mod color;
pub mod gamma;
pub mod metric;
pub mod rep_symbols;
pub mod representations;

/// Defines operations related to manipulating abstract indices within symbolic expressions,
/// particularly relevant for physics calculations involving tensor structures and diagrams.
///
/// This trait provides methods for conjugating expressions, wrapping indices (both all and
/// only dummy/contracted ones), simplifying indices ("cooking"), and identifying external
/// ("dangling") indices.
pub trait IndexTooling {
    /// Wraps all abstract indices within the expression using a specified header symbol.
    ///
    /// This transforms indices like `mink(dim,idx)` into `mink(dim,header(idx))`. Useful for distinguishing
    /// between different copies of an expression, e.g., an amplitude and its complex conjugate.
    ///
    /// # Arguments
    /// * `header` - The [`Symbol`] to use as the wrapping function name.
    ///
    /// # Returns
    /// A new [`Atom`] with all indices wrapped.
    fn wrap_indices(&self, header: Symbol) -> Atom;

    /// Wraps only the dummy (contracted) abstract indices within the expression using a header symbol.
    ///
    /// Identifies indices that appear contracted (e.g., one covariant, one contravariant)
    /// and wraps only those, leaving external indices unchanged. Transforms `idx -> header(idx)`
    /// for dummy indices `idx`.
    ///
    /// # Arguments
    /// * `header` - The [`Symbol`] to use as the wrapping function name for dummy indices.
    ///
    /// # Returns
    /// A new [`Atom`] with only dummy indices wrapped.
    fn wrap_dummies(&self, header: Symbol) -> Atom;

    /// Simplifies structured indices within function arguments into flattened variable symbols.
    ///
    /// Replaces indices like `mink(4, mu)` inside function arguments (not top-level) with
    /// unique variable symbols like `var_mink_4_mu`. This can aid pattern matching.
    ///
    /// # Returns
    /// A new [`Atom`] with "cooked" indices.
    fn cook_indices(&self) -> Atom;

    /// Converts a single function [`Atom`] into a flattened variable symbol based on its name and arguments.
    ///
    /// Expects the input `Atom` to be a function. Returns a variable `Atom` whose symbol name
    /// encodes the original function and its arguments (e.g., `f(a, b)` might become `var_f_a_b`).
    /// Fails if the input is not a function or if arguments are not convertible (e.g., polynomials).
    ///
    /// # Returns
    /// `Ok(Atom)` containing the new variable symbol on success.
    /// `Err(CookingError)` if the input cannot be cooked.
    fn cook_function(&self) -> Result<Atom, CookingError>;

    /// Computes the physics-aware conjugate of the expression.
    ///
    /// Applies conjugation rules specific to physics objects like spinors, gamma matrices,
    /// color representations, and the imaginary unit `i`. See implementation details for
    /// specific rules applied.
    ///
    /// # Returns
    /// A new [`Atom`] representing the conjugated expression.
    fn conj(&self) -> Atom;

    /// Identifies and returns a list of dangling (external, uncontracted) indices.
    ///
    /// Analyzes the expression to find indices that are not summed over. Returns them
    /// as `Atom`s. Note that dual indices might be represented wrapped in a `dind` function.
    ///
    /// # Returns
    /// A `Vec<Atom>` where each `Atom` represents a dangling index.
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

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Calculates the physics-aware conjugate of the expression.
///
/// This considers the conjugation rules for various physics objects:
/// - Complex numbers: `i -> -i`
/// - Polarization vectors: `eps(p) <-> epsbar(p)`
/// - Spinors: `u(p) <-> ubar(p)`, `v(p) <-> vbar(p)`
/// - Gamma matrices: `gamma(mu, a, b) -> -gamma(mu, b, a)` (note the index swap and sign)
/// - Gamma5: `gamma5(a, b) -> gamma5(b, a)`
/// - Color generators: `t(i, a, b) -> t(i, b, a)` (for fundamental `a`, antifundamental `b`)
/// - Color representations: Switches fundamental and anti-fundamental representations.
///
/// # Args:
///     self_ (Expression): The expression to conjugate.
///
/// # Returns:
///     Expression: The conjugated expression.
pub fn conj(self_: &PythonExpression) -> PythonExpression {
    self_.expr.conj().into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
pub fn expand_mink(self_: &PythonExpression) -> PythonExpression {
    self_.expr.expand_mink().into()
}
#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
pub fn expand_bis(self_: &PythonExpression) -> PythonExpression {
    self_.expr.expand_bis().into()
}
#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
pub fn expand_mink_bis(self_: &PythonExpression) -> PythonExpression {
    self_.expr.expand_mink_bis().into()
}
#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
pub fn expand_color(self_: &PythonExpression) -> PythonExpression {
    self_.expr.expand_color().into()
}
#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
pub fn expand_metrics(self_: &PythonExpression) -> PythonExpression {
    self_.expr.expand_metrics().into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Wraps all abstract indices within the expression using a header symbol.
///
/// This is often used to distinguish indices belonging to different parts
/// of a calculation, e.g., the amplitude and its conjugate in an amplitude squared.
/// For an index `idx`, the transformation is `idx -> header(idx)`.
///
/// # Example:
///     `wrap_indices(expr, symbol("left"))` might turn `p(mink(4, mu))`
///     into `p(mink(4, left(mu)))`.
///
/// # Args:
///     self_ (Expression): The input expression.
///     header (Symbol): The symbol to use as the wrapper function name.
///
/// # Returns:
///     Expression: A new expression with all indices wrapped.
pub fn wrap_indices(self_: &PythonExpression, header: Symbol) -> PythonExpression {
    self_.expr.wrap_indices(header).into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// "Cooks" indices within function arguments into simplified, unique symbols.
///
/// This process takes structured indices like `mink(4, f(g(mu)))` appearing as
/// arguments to functions (but not the top-level function arguments themselves)
/// and replaces them with flattened symbols like `mink(4,f_g_mu)`.
///
/// # Args:
///     self_ (Expression): The expression containing indices to be cooked.
///
/// # Returns:
///     Expression: A new expression with cooked indices inside function arguments.
pub fn cook_indices(self_: &PythonExpression) -> PythonExpression {
    self_.expr.cook_indices().into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Converts a single function atom into a flattened variable symbol.
///
/// Takes an expression that must be a single function call (e.g., `f(a, b)`)
/// and converts it into a variable symbol whose name encodes the original
/// function and its arguments (e.g., `f_a_b`).
///
/// # Args:
///     self_ (Expression): The expression representing the function atom to cook.
///         Must not be a sum, product, power, variable, or number.
///
/// # Returns:
///     Expression: An expression representing the new variable symbol.
///
/// # Raises:
///     TypeError: If the input expression is not a single function atom or
///         if arguments contain types that cannot be cooked (e.g., polynomials).
pub fn cook_function(self_: &PythonExpression) -> PyResult<PythonExpression> {
    self_
        .expr
        .cook_function()
        .map_err(|a| PyTypeError::new_err(format!("cannot cook: {a:?}")))
        .map(|a| a.into())
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Wraps only the dummy (contracted) indices within the expression using a header symbol.
///
/// Similar to `wrap_indices`, but it identifies contracted indices (those appearing
/// once upstairs and once downstairs, or twice in a self-dual representation)
/// and only wraps those, leaving external (dangling) indices untouched.
///
/// # Example:
///     `wrap_dummies(term1 * term2, symbol("internal"))` where `term1` and `term2`
///     share a contracted index `mu`, might wrap `mu` resulting in
///     `internal(mu)` where it appears, but leave other external indices as they are.
///
/// # Args:
///     self_ (Expression): The input expression.
///     header (Symbol): The symbol to use as the wrapper function name for dummy indices.
///
/// # Returns:
///     Expression: A new expression with only dummy indices wrapped.
pub fn wrap_dummies(self_: &PythonExpression, header: Symbol) -> PythonExpression {
    self_.expr.wrap_dummies(header).into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Lists the dangling (external, uncontracted) indices present in the expression.
///
/// Identifies indices that are not summed over (i.e., not dummy indices).
/// For dualizable representations, downstairs indices are represented wrapped
/// in `dind(...)`.
///
/// # Args:
///     self_ (Expression): The expression to analyze.
///
/// # Returns:
///     list[Expression]: A list of expressions, each representing a dangling index.
///
pub fn list_dangling(self_: &PythonExpression) -> Vec<PythonExpression> {
    self_
        .expr
        .list_dangling()
        .into_iter()
        .map(|a| a.into())
        .collect()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Applies Clifford algebra rules and trace identities to simplify gamma matrices.
///
/// Performs simplifications based on the anticommutation relations:
/// `{gamma(mu), gamma(nu)} = 2 * g(mu, nu)`
/// and evaluates traces of gamma matrix chains. Assumes gamma matrices
/// are represented by `alg::gamma(...)` and the metric by `spenso::g(...)`.
/// Uses internal helper symbols like `alg::gamma_chain` and `alg::gamma_trace`.
///
/// # Args:
///     self_ (Expression): The expression containing gamma matrices.
///
/// # Returns:
///     Expression: The simplified expression.
pub fn simplify_gamma(self_: &PythonExpression) -> PythonExpression {
    self_.expr.simplify_gamma().into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Converts contracted Lorentz/Minkowski indices into dot product notation.
///
/// Looks for patterns like `p(mink(D, mu)) * q(mink(D, mu))` and replaces
/// them with `dot(p, q)`. Assumes vectors are represented by functions
/// taking a single Minkowski index.
///
/// # Args:
///     self_ (Expression): The expression with contracted indices.
///
/// # Returns:
///     Expression: The expression with contractions replaced by `dot(...)` calls.
pub fn to_dots(self_: &PythonExpression) -> PythonExpression {
    self_.expr.to_dots().into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Simplifies contractions involving metric tensors and identity tensors.
///
/// Applies rules like:
/// - `g(mu, nu) * p(nu) -> p(mu)`
/// - `g(mu, nu) * g(nu, rho) -> g(mu, rho)` (or `id(mu, rho)`)
/// - `g(mu, mu) -> D` (dimension)
/// - `id(mu, nu) * p(nu) -> p(mu)`
/// Assumes the metric tensor is `g(...)` or `metric(...)` and the identity
/// is `id(...)` or `𝟙(...)`.
///
/// # Args:
///     self_ (Expression): The expression with metric/identity tensors.
///
/// # Returns:
///     Expression: The simplified expression.
pub fn simplify_metrics(self_: &PythonExpression) -> PythonExpression {
    self_.expr.simplify_metrics().into()
}

#[cfg(feature = "python")]
#[gen_stub_pyfunction(module = "symbolica_community.algebraic_simplification")]
#[pyfunction]
/// Applies SU(N) color algebra rules to simplify color structures.
///
/// Performs simplifications involving:
/// - Structure constants `alg::f(a, b, c)`
/// - Generators `alg::t(a, i, j)`
/// - Traces `alg::TR`
/// - Number of colors `alg::Nc`
/// - Casimir invariants, Fierz identities, etc.
///
/// Note: Simplification might not be complete for all possible color structures.
/// If the result still contains explicit color indices (like `cof(...)`, `coad(...)`),
/// simplification was not fully successful.
///
/// # Args:
///     self_ (Expression): The expression with color factors.
///
/// # Returns:
///     Expression: The simplified expression, potentially containing only color-scalar factors
///                 like `Nc`, `TR`, `CF`, `CA`.
///
/// Raises:
///     RuntimeWarning: If the simplification could not fully eliminate all explicit
///                     color indices, indicating an incomplete simplification. The
///                     partially simplified expression is still returned.
pub fn simplify_color(self_: &PythonExpression) -> PyResult<PythonExpression> {
    self_.expr.simplify_color().map(|a| a.into()).map_err(|a| {
        PyRuntimeWarning::new_err(match a {
            ColorError::NotFully(a) => format!("Not fully simplified: {}", a),
        })
    })
}

#[cfg(feature = "python")]
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
    child_module.add_function(wrap_pyfunction!(expand_bis, m)?)?;
    child_module.add_function(wrap_pyfunction!(expand_mink_bis, m)?)?;
    child_module.add_function(wrap_pyfunction!(expand_mink, m)?)?;
    child_module.add_function(wrap_pyfunction!(expand_metrics, m)?)?;
    child_module.add_function(wrap_pyfunction!(expand_color, m)?)?;

    m.add_submodule(&child_module)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("symbolica_community.algebraic_simplification", child_module)?;

    Ok(())
}
