import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="One-loop gluon propagator and UV expansion",
)


@app.cell
def _():
    import marimo as mo
    from copy import copy
    from symbolica import S, E, Expression
    from symbolica.community import idenso
    from symbolica.community.spenso import (
        Representation, TensorExpression, TensorName, TensorPattern, dot,
    )
    from symbolica.community.hep import Model, SnailFilterOptions, FeynmanDiagram
    from symbolica.community.hep.oneloop import A0

    return (
        A0,
        E,
        Expression,
        FeynmanDiagram,
        Model,
        Representation,
        S,
        SnailFilterOptions,
        TensorExpression,
        TensorName,
        TensorPattern,
        copy,
        dot,
        idenso,
        mo,
    )


@app.cell
def _():
    return


@app.cell
def _(Representation, S, TensorName, dot):
    D, eps = S('hep_gluon::D', 'hep_gluon::eps')
    lorentz = Representation.mink(D)
    adjoint = Representation.coad(8)
    P = TensorName.vector('hep_gluon::P')
    K = TensorName.vector('hep_gluon::K')
    p, k = P(lorentz), K(lorentz)
    UV_P = TensorName.vector('hep_gluon::uv_P')
    uv_p = UV_P(lorentz)
    index_ = S('hep_gluon::index_')
    MOMENTUM = S('gammalooprs::Q')
    t = S("hep_gluon::t", is_scalar=True) # a scale that can move out of dot products
    mUV = S("hep_gluon::mUV", is_scalar=True)
    uv_probe = S("hep_gluon::uv_probe", is_scalar=True)
    prop = S("hep_gluon::prop", is_scalar=True)
    q_, mass2_ = S("hep_gluon::q_", "hep_gluon::mass2_")
    # Rewrite patterns contain wildcards rather than concrete rank-one tensors.
    dot_pattern = S('spenso::dot')
    # Python aliases for actual dot expressions, not substitute scalar symbols.
    k2, kp, p2 = dot(k, k), dot(k, p), dot(p, p)
    return (
        D,
        K,
        MOMENTUM,
        P,
        UV_P,
        adjoint,
        dot_pattern,
        eps,
        index_,
        k,
        k2,
        kp,
        lorentz,
        mUV,
        mass2_,
        p2,
        prop,
        q_,
        t,
        uv_p,
        uv_probe,
    )


@app.cell
def _(A0, mUV):
    A0(mUV, 1.)
    return


@app.cell
def _(TensorExpression, idenso):
    color_settings = idenso.ColorSimplifySettings(
        substitute_cof_dimension_invariants=True,
    )

    def contract_indices(expression: TensorExpression) -> TensorExpression:
        """Close internal Dirac, color, and Lorentz indices, retaining external ports."""
        # Idenso iterates internally, but gamma-chain collection precedes its loop.
        # The bottom numerator still needs a second call to close newly exposed chains.
        for _ in range(12):
            # Scalar Casimirs are currently mistaken for unresolved tensor ports.
            # Resolve them in Idenso before the typed result re-infers its ports.
            simplified = (expression.expand().simplify_gamma()
                          .simplify_color(color_settings).simplify_metrics().expand())
            if simplified == expression:
                return simplified
            expression = simplified
        raise RuntimeError('Tensor contractions did not reach a fixed point')

    return (contract_indices,)


@app.cell
def _(D, FeynmanDiagram, P, TensorExpression, adjoint, lorentz, p2):
    def external_projector(diagram: FeynmanDiagram, *, longitudinal=False) -> TensorExpression:
        """Color-average the transverse trace (or the longitudinal Ward check).

        P_T = delta_ab (g_mu_nu - p_mu p_nu/p^2) / [8 (D-1)].
        Thus P_T . [delta_ab (p^2 g_mu_nu - p_mu p_nu) Pi] = p^2 Pi.
        """
        slots = diagram.numerator_expression().with_lorentz_dimension(D).interface
        mu, nu = [slot for slot in slots if slot.to_expression().get_name() == 'spenso::mink']
        a, b = [slot for slot in slots if slot.to_expression().get_name() == 'spenso::coad']
        color_metric = TensorExpression.g(adjoint)(a, b)
        momentum_pair = P(mu).outer(P(nu)) / p2
        if longitudinal:
            return color_metric.outer(momentum_pair) / 8
        lorentz_metric = TensorExpression.g(lorentz)(mu, nu)
        return color_metric.outer(lorentz_metric - momentum_pair) / (8 * (D - 1))

    def apply_projector(numerator: TensorExpression, projector: TensorExpression) -> TensorExpression:
        """Contract every named external slot with its matching projector slot."""
        if set(numerator.list_dangling()) != set(projector.list_dangling()):
            raise ValueError('Numerator and projector must have matching external slots')
        # Multiple compatible ports make implicit multiplication ambiguous.
        # Selecting one pair also contracts the remaining matching Einstein indices.
        right = projector.interface.index(numerator.interface[0])
        return numerator.contract(projector, left=0, right=right).schoonschip_net(
            expand_contracted_sums=True,
        )

    return apply_projector, external_projector


@app.cell
def _(FeynmanDiagram, K, P, S, TensorExpression, TensorPattern, index_):
    def route_momenta(diagram: FeynmanDiagram, expression: TensorExpression) -> TensorExpression:
        """Route edge momenta through the graph, then name the basis vectors k and p."""
        if diagram.loop_count != 1 or len(diagram.external_edges) != 2:
            raise ValueError('This example requires a one-loop two-point graph')
        routed = diagram.loop_momentum_basis.route_expression(expression.to_expression())
        # The wildcard stands for a whole slot or compact representation.
        routed = routed.replace(S('gammalooprs::K')(0, index_), TensorPattern(K, ports=[index_]))
        routed = routed.replace(S('gammalooprs::P')(0, index_), TensorPattern(P, ports=[index_]))
        return TensorExpression(routed.expand())

    return (route_momenta,)


@app.cell
def _(Expression, FeynmanDiagram, Model, S, index_):
    def diagram_weight(diagram: FeynmanDiagram, model: Model) -> Expression:
        """Evaluate provenance factors and supply the closed-ghost-loop sign.

        The current closed_fermion_loop_count only counts
        is_fermion() particles (UFO spin 2), excluding spin -1 FP ghosts.
        This example therefore includes their Grassmann minus explicitly.
        """
        factor = diagram.overall_factor_expression() * diagram.numerator_prefactor_expression()
        for name in ('AutG', 'CouplingsMultiplicity', 'ExternalFermionOrderingSign',
                     'InternalFermionLoopSign', 'AntiFermionSpinSumSign'):
            factor = factor.replace(S('feynkit_generator_factor::' + name)(index_), index_)
        if all(model.particle(edge.particle_name).spin == -1 for edge in diagram.internal_edges):
            factor = -factor
        return factor

    return (diagram_weight,)


@app.cell
def _(
    E,
    FeynmanDiagram,
    MOMENTUM,
    Model,
    S,
    TensorExpression,
    lorentz,
    prop,
    route_momenta,
):
    def graph_propagators(diagram: FeynmanDiagram, model: Model) -> TensorExpression:
        """Build quadratic UV propagators with the model masses and graph routing."""
        if len(diagram.internal_edges) != 2:
            raise ValueError('This example supports two-propagator bubbles only')
        product = E('1')
        for edge in diagram.internal_edges:
            mass = model.particle(edge.particle_name).mass_parameter
            mass_squared = E('0') if mass == 'ZERO' else S('UFO::' + mass)**2
            momentum = route_momenta(diagram, TensorExpression(MOMENTUM(edge.id, lorentz.to_expression())))
            product *= prop(momentum, mass_squared)
        return TensorExpression(product)

    return (graph_propagators,)


@app.cell
def _(Expression, TensorExpression, dot_pattern, k, mUV, mass2_, prop, q_, t):
    def evaluate_propagators(expression: Expression) -> TensorExpression:
        """Interpret prop(q, M2) as 1/(q.q-M2), retaining the model's mass term."""
        return TensorExpression(expression.replace(prop(q_, mass2_), 1/(dot_pattern(q_, q_)-mass2_)))

    def uv_deform(expression: Expression) -> TensorExpression:
        """Scale the full integrand and shift normalized propagator masses."""
        scaled = expression.replace(k, k/t)
        # Factoring t^2 from a rescaled propagator rescales its mass to t^2*m^2.
        # Apply prop(q,M2) -> prop(q,M2+(1-t^2)*mUV^2) in that normalization.
        return TensorExpression(scaled.replace(
            prop(q_, mass2_),
            t**2 * prop(t*q_, t**2*mass2_ + (1-t**2)*mUV**2),
        ))

    return evaluate_propagators, uv_deform


@app.cell
def _(
    E,
    Expression,
    FeynmanDiagram,
    MOMENTUM,
    S,
    TensorExpression,
    TensorPattern,
    UV_P,
    dot,
    index_,
    k2,
    kp,
    lorentz,
    p2,
    uv_p,
):
    def uv_tensor_input(expression: Expression, diagram: FeynmanDiagram) -> TensorExpression:
        """Convert mixed dot powers to the indexed input FeynKit accepts.

        This only changes notation; no angular averaging is performed here.
        Each contraction receives independent dummy indices.
        Radial dot(K,K) factors remain scalar coefficients.
        """
        loop_edge, = diagram.loop_momentum_basis.loop_edges
        indexed = E('0')
        for power, coefficient in expression.expand().coefficient_list(kp):
            rank = 0 if power == E('1') else power.to_polynomial().degree(kp)
            tensor = E('1')
            for n in range(rank):
                mu = S(f'hep_gluon::uv_mu{n}')
                # Keep distinct explicit contractions: undo_dots currently reuses
                # dummy indices in powers, and expand_dots requires concrete D.
                tensor *= MOMENTUM(loop_edge, lorentz(mu).to_expression()) * UV_P(lorentz(mu)).to_expression()
            indexed += coefficient * tensor
        return TensorExpression(indexed)

    def uv_scalar_invariants(expression: Expression, diagram: FeynmanDiagram) -> TensorExpression:
        """Restore the original loop and external vectors inside scalar dots."""
        loop_edge, = diagram.loop_momentum_basis.loop_edges
        loop_vector = TensorExpression(MOMENTUM(loop_edge, lorentz.to_expression()))
        reduced = expression.replace(dot(loop_vector, loop_vector), k2)
        reduced = reduced.replace(dot(uv_p, uv_p), p2).expand()
        if reduced.replace(MOMENTUM(loop_edge, index_), 0).replace(TensorPattern(UV_P, ports=[index_]), 0) != reduced:
            raise ValueError('Tensor reduction left an indexed momentum')
        return TensorExpression(reduced)

    return uv_scalar_invariants, uv_tensor_input


@app.cell
def _(D, TensorExpression, k, k2, uv_probe):
    def uv_pole_residue(expression: TensorExpression) -> TensorExpression:
        """Extract the logarithmic radial tail: its one-loop pole is i/(16*pi^2*eps)."""
        tail = expression.replace(k, k/uv_probe)
        # Include the one-loop measure and keep its scale-independent term.
        logarithmic = (tail/uv_probe**4).series(uv_probe, 0, 0)[0]
        return TensorExpression((logarithmic*k2**2).expand().cancel().replace(D, 4).expand())

    return (uv_pole_residue,)


@app.cell
def _(
    D,
    FeynmanDiagram,
    TensorExpression,
    copy,
    evaluate_propagators,
    t,
    uv_deform,
    uv_pole_residue,
    uv_scalar_invariants,
    uv_tensor_input,
):
    def uv_expansion_data(expression: TensorExpression, diagram: FeynmanDiagram):
        """Expand an expression copy, with numerator and massive UV denominators together."""
        uv_copy = copy(expression)
        deformed = uv_deform(uv_copy)
        measure_factor = t**(-4*diagram.loop_count)
        with_measure = evaluate_propagators(deformed) * measure_factor
        # Retain every UV-divergent power, then remove the measure bookkeeping.
        series = TensorExpression((with_measure.series(t, 0, 0).to_expression()/measure_factor).expand())
        reduced = uv_scalar_invariants(
            diagram.tensor_reduce(D, expression=uv_tensor_input(series, diagram)), diagram
        )
        counterterm = TensorExpression(-reduced.replace(t, 1).expand())
        return {'copy': uv_copy, 'deformed': deformed, 'series': series,
                'reduced': reduced, 'counterterm': counterterm,
                'residue': -uv_pole_residue(counterterm)}

    return (uv_expansion_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The gluon propagator at one loop

    Generate the QCD contributions to the gluon two-point function,
    $g^* \to g^*$, and inspect a diagram and its symbolic expressions.
    The external momentum is off shell.
    """)
    return


@app.cell
def _(Model, mo):
    model = Model(mo.notebook_location() / "hep_sm.json")
    g = model.particle("g")
    g
    return g, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate the diagrams

    Keep exactly two powers of the QCD coupling and no QED vertices.
    The current selection keeps the gluon, ghost, and bottom-quark bubbles.
    The zero-snail filter removes the massless tadpole.
    """)
    return


@app.cell
def _(SnailFilterOptions, g, model):
    loops = 2
    diagrams = model.generate_diagrams(
        [g],
        [g],
        loops=loops,
        coupling_orders={"QCD": 2 * loops, "QED": 0},
        particle_veto=["c", "t", "s", "u", "d"],
        zero_snails=SnailFilterOptions(),
    )
    return (diagrams,)


@app.cell(hide_code=True)
def _(diagrams, mo):
    mo.md(f"""
    **{len(diagrams)} diagrams** with the filters above.
    The bottom mass is kept symbolic. Remove the particle veto to include
    the other quark flavors.
    """)
    return


@app.cell(hide_code=True)
def _(diagrams, mo):
    _options = {
        f"{i}: {', '.join(e.particle_name for e in d.internal_edges)}": i
        for i, d in enumerate(diagrams)
    }
    _default = next(
        (label for label, index in _options.items()
         if all(edge.particle_name == "g" for edge in diagrams[index].internal_edges)),
        next(iter(_options)),
    )
    diagram_index = mo.ui.dropdown(
        options=_options,
        # Select the gluon bubble without assuming its generated diagram index.
        value=_default,
        label="Particles in the loop",
    )
    diagram_index
    return (diagram_index,)


@app.cell
def _(diagram_index, diagrams):
    diagram = diagrams[diagram_index.value]
    diagram
    return (diagram,)


@app.cell(hide_code=True)
def _(diagram, mo):
    _options = {
        "None": ("none", None),
        "Internal lines": ("internal", None),
        "External lines": ("external", None),
    }
    _options.update(
        {
            f"{name} lines": ("particle", name)
            for name in sorted({edge.particle_name for edge in diagram.edges})
        }
    )
    highlight_selection = mo.ui.dropdown(
        options=_options,
        value="Internal lines",
        label="Highlight subgraph",
    )
    highlight_selection
    return (highlight_selection,)


@app.cell
def _(diagram, highlight_selection):
    import linnet as _linnet

    _graph = diagram.to_linnet()
    _graph.render_config = _graph.render_config.overlay(
        _linnet.RenderConfig(
            selectors=_linnet.DrawingSelectors(
                # Capture the factory for rendering after Marimo clears cell-local names.
                edge=lambda edge, drawing=_linnet.EdgeDrawing: drawing(
                    label=edge.data.particle_name
                )
            )
        )
    )
    _kind, _particle = highlight_selection.value
    if _kind == "internal":
        highlighted_subgraph = _graph.filter(
            edge=lambda edge: not edge.data.is_external
        )
    elif _kind == "external":
        highlighted_subgraph = _graph.filter(edge=lambda edge: edge.data.is_external)
    elif _kind == "particle":
        highlighted_subgraph = _graph.filter(
            edge=lambda edge: edge.data.particle_name == _particle
        )
    else:
        highlighted_subgraph = None
    highlighted_subgraph
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Inspect the numerator

    The Feynman rules return a `TensorExpression`. Below we contract internal
    Dirac, color, and Lorentz indices, keeping the two external gluon indices.
    Lorentz slots are continued to symbolic $D$ before contraction.
    """)
    return


@app.cell
def _(diagram):
    diagram.numerator_expression(in_lmb=True).to_expression()
    return


@app.cell
def _(diagram):
    r = diagram.numerator_expression(in_lmb=True)
    r
    return


@app.cell
def _(diagram):
    diagram.to_linnet()
    return


@app.cell
def _(diagram):
    diagram.subgraph(nodes=[2, 3])
    return


@app.cell
def _(diagram, mUV):
    diagram.uv_counterterm(mUV, subgraph=diagram.subgraph(nodes=[2, 3]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Collect bispinor chains

    Choose a diagram containing a bottom-quark line in **Particles in the loop**
    to see slash notation: $\not{q}=\gamma^\mu q_\mu$.
    This view collects the bispinor factors into ordered chains and traces,
    leaving the Dirac traces unevaluated.
    """)
    return


@app.cell
def _(D, diagram):
    bispinor_numerator = (
        diagram.numerator_expression()
        .with_lorentz_dimension(D)
        .schoonschip()
        .collect_gamma_chains()
        # Collection exposes identity factors in the massive propagator terms.
        .simplify_metrics()
        .collect_gamma_chains()
    )
    bispinor_numerator
    return


@app.cell
def _(D, contract_indices, diagram):
    # Continue Lorentz slots to D, retaining the external tensor interface.
    contracted_numerator = contract_indices(
        diagram.numerator_expression().with_lorentz_dimension(D)
    )
    contracted_numerator
    return (contracted_numerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project the external indices

    For off-shell $p^2\ne0$, use the transverse, color-averaged projector
    $$\mathcal P^{ab}_{\mu\nu}=
    \frac{\delta^{ab}}{8(D-1)}
    \left(g_{\mu\nu}-\frac{p_\mu p_\nu}{p^2}\right).$$
    It extracts $p^2\Pi_T$ from
    $\delta^{ab}(p^2g^{\mu\nu}-p^\mu p^\nu)\Pi_T$.
    This replaces the default external polarization vectors; it is not an
    additional polarization sum. The helper reads the actual external slots.
    """)
    return


@app.cell
def _(diagram, external_projector):
    projector = external_projector(diagram)
    projector
    return (projector,)


@app.cell
def _(diagram):
    diagram.loop_momentum_basis
    return


@app.cell
def _(
    apply_projector,
    contract_indices,
    contracted_numerator,
    diagram,
    projector,
    route_momenta,
):
    routed_numerator = route_momenta(diagram, contracted_numerator)
    projected_numerator = contract_indices(apply_projector(routed_numerator, projector)).to_dots().expand()
    projected_numerator
    return (projected_numerator,)


@app.cell
def _(TensorExpression, diagram, diagram_weight, model, projected_numerator):
    weighted_projected_numerator = TensorExpression(model.expand_couplings(
        projected_numerator * diagram_weight(diagram, model)
    )).expand()
    weighted_projected_numerator
    return (weighted_projected_numerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The last expression includes the symmetry factor and closed-loop signs,
    with the model couplings expressed through `UFO::G`.
    The current generator's loop-sign counter excludes
    spin −1 ghosts: `diagram_weight` supplies their Grassmann minus explicitly.
    All algebra functions are defined in the notebook cells above.

    The cells below expand an expression copy, apply FeynKit's tensor reduction,
    and compute the integrated UV counterterm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the energy denominators

    `build_cff()` constructs the Cross-Free Family representation of this
    diagram. Its output is a normal Symbolica expression. This step constructs
    the denominators; it does not evaluate the loop integral.
    """)
    return


@app.cell
def _(diagram):
    cff = diagram.build_cff().to_expression()
    cff
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UV expansion on a copy of the expression

    Read each internal edge's mass from the model and use
    the graph's momentum routing. Here `prop(q, M2)` denotes $1/(q^2-M^2)$.
    Copy the full projected integrand and scale $k\to k/t$ in numerator and
    propagators together. After extracting $t^2$ from each propagator, apply
    $$\operatorname{prop}(k+tp,t^2m^2)\longrightarrow
    \operatorname{prop}(k+tp,t^2m^2+(1-t^2)m_{\rm UV}^2).$$
    At $t=1$ this recovers the physical integrand; at $t=0$ the denominator is
    $k^2-m_{\rm UV}^2$. Include the loop-measure factor $t^{-4L}$ for
    four-dimensional power counting and expand through $t^0$. For these
    quadratically divergent bubbles, the series starts at $t^{-2}$.
    Remove the measure factor before tensor reduction to recover the
    counterterm integrand. This bookkeeping does not change the symbolic
    dimension $D$ used in the contractions.
    All momentum squares remain dot products, including $p\cdot p$.
    """)
    return


@app.cell
def _(diagram, graph_propagators, model, weighted_projected_numerator):
    graph_propagator_product = graph_propagators(diagram, model)
    if not weighted_projected_numerator.is_scalar:
        raise ValueError('Contract all external indices before taking the UV series')
    projected_integrand = weighted_projected_numerator.expand() * graph_propagator_product
    projected_integrand
    return (projected_integrand,)


@app.cell
def _(D, diagram, route_momenta):
    graph_denominator = route_momenta(diagram, diagram.denominator_expression(dimension=D))
    graph_denominator
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same massive expansion is available directly on a diagram. It retains
    every UV-divergent order automatically and returns the additive local
    counterterm in edge momenta with tagged `denom` propagators. Diagram-wide
    weights and projectors remain separate. Pass `numerator=` for a prepared
    numerator and `subgraph=diagram.filter(...)` to expand a selected region.
    This is a single UV limit, before forest subtraction and pole integration.
    """)
    return


@app.cell
def _(diagram, mUV):
    diagram_local_uv_counterterm = diagram.uv_counterterm(mUV)
    diagram_local_uv_counterterm
    return


@app.cell
def _(copy, projected_integrand, uv_deform):
    uv_expression_copy = copy(projected_integrand)
    uv_deformed_expression = uv_deform(uv_expression_copy)
    uv_deformed_expression
    return (uv_deformed_expression,)


@app.cell
def _(diagram, evaluate_propagators, t, uv_deformed_expression):
    uv_scaled_expression = evaluate_propagators(uv_deformed_expression)
    uv_measure_factor = t**(-4*diagram.loop_count)
    uv_with_measure = uv_scaled_expression * uv_measure_factor
    uv_series = uv_with_measure.series(t, 0, 0)
    uv_series
    return uv_measure_factor, uv_series


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tensor-reduce the UV expansion

    Each expanded denominator depends only on $k^2$.
    `diagram.tensor_reduce(D, expression=uv_indexed_series)` identifies the
    internal momentum from the graph and keeps $p$ external.
    `uv_tensor_input` only restores explicit
    indices for the mixed dots $(k\cdot p)^n$. FeynKit computes the
    tensor reduction, including the vanishing odd-rank terms.
    """)
    return


@app.cell
def _(diagram, uv_measure_factor, uv_series, uv_tensor_input):
    uv_indexed_series = uv_tensor_input(uv_series.to_expression()/uv_measure_factor, diagram)
    return (uv_indexed_series,)


@app.cell
def _(D, diagram, uv_indexed_series):
    uv_tensor_reduced = diagram.tensor_reduce(D, expression=uv_indexed_series)
    uv_tensor_reduced
    return (uv_tensor_reduced,)


@app.cell
def _(TensorExpression, diagram, t, uv_scalar_invariants, uv_tensor_reduced):
    uv_reduced_series = uv_scalar_invariants(uv_tensor_reduced, diagram)
    uv_reduced_expression = TensorExpression(uv_reduced_series.replace(t, 1).expand())
    uv_reduced_expression
    return (uv_reduced_series,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UV counterterm

    The UV expansion already has massive denominators
    $(k^2-m_{\rm UV}^2)^n$. Tensor-reduce the series, set $t=1$, and negate it
    to obtain the angular-averaged counterterm. No mass is inserted afterward.

    The integrated pole is determined by the coefficient of $1/(k^2)^2$
    in its large-$k$ radial expansion. With $D=4-2\epsilon$ and measure
    $d^Dk/(2\pi)^D$, multiply that coefficient by $i/(16\pi^2\epsilon)$.
    The auxiliary $m_{\rm UV}$ cancels from this coefficient. Finite parts
    are not evaluated.
    """)
    return


@app.cell
def _(TensorExpression, t, uv_reduced_series):
    uv_counterterm_integrand = TensorExpression(-uv_reduced_series.replace(t, 1).expand())
    uv_counterterm_integrand
    return (uv_counterterm_integrand,)


@app.cell
def _(E, eps, uv_counterterm_integrand, uv_pole_residue):
    uv_residue = -uv_pole_residue(uv_counterterm_integrand)
    uv_counterterm = -E('1i') * uv_residue / (16 * E('pi')**2 * eps)
    uv_counterterm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the sum

    For the selected gluon, ghost, and bottom loops, the transverse pole coefficient
    is $13G^2p^2/3$. The longitudinal pole cancels. This is a gluon-field counterterm;
    it is not the QCD beta-function coefficient. Keep $p^2\ne0$.
    """)
    return


@app.cell
def _(
    D,
    E,
    TensorExpression,
    apply_projector,
    contract_indices,
    diagram_weight,
    diagrams,
    external_projector,
    graph_propagators,
    mo,
    model,
    route_momenta,
    uv_expansion_data,
):
    _uv_rows = []
    for _d in diagrams:
        _propagators = graph_propagators(_d, model)
        _weight = diagram_weight(_d, model)
        _routed = route_momenta(
            _d, contract_indices(_d.numerator_expression().with_lorentz_dimension(D))
        )
        _transverse = contract_indices(apply_projector(
            _routed, external_projector(_d)
        )).to_dots().expand()
        _longitudinal = contract_indices(apply_projector(
            _routed, external_projector(_d, longitudinal=True)
        )).to_dots().expand()
        _transverse = TensorExpression(model.expand_couplings(_transverse * _weight)).expand()
        _longitudinal = TensorExpression(model.expand_couplings(_longitudinal * _weight)).expand()
        if not _transverse.is_scalar or not _longitudinal.is_scalar:
            raise ValueError('Contract all external indices before taking the UV series')
        _uv_rows.append({
            'loop': ', '.join(_e.particle_name for _e in _d.internal_edges),
            'transverse': uv_expansion_data(_transverse.expand() * _propagators, _d)['residue'],
            'longitudinal': uv_expansion_data(_longitudinal.expand() * _propagators, _d)['residue'],
        })
    total_uv_residue = TensorExpression(sum((_row['transverse'].to_expression() for _row in _uv_rows), E('0')).expand())
    longitudinal_uv_residue = TensorExpression(sum((_row['longitudinal'].to_expression() for _row in _uv_rows), E('0')).expand())
    assert longitudinal_uv_residue == E('0')
    mo.ui.table([
        {'Loop': _row['loop'],
         'Transverse residue': _row['transverse'].format_tensor(),
         'Longitudinal residue': _row['longitudinal'].format_tensor()}
        for _row in _uv_rows
    ], selection=None)
    return (total_uv_residue,)


@app.cell
def _(E, eps, total_uv_residue):
    total_uv_counterterm = -E('1i') * total_uv_residue / (16 * E('pi')**2 * eps)
    total_uv_counterterm
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Generated QED cross section

    Generate $e^-e^+\to\mu^-\mu^+$, sum and average the incoming spins,
    and contract the generated numerator with Idenso. The massive squared
    matrix element is retained; its massless limit is integrated over two-body
    phase space below. This reproduces the unpolarized result in the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-MuAmu).
    """)
    return


@app.cell
def _(E, Model, S, idenso, mo):
    from symbolica.community import hep as qed_hep

    qed_model = Model(mo.notebook_location() / "hep_sm.json")
    _process = qed_hep.Process.cross_section([11, -11], [13, -13]).with_loop_count(1, 1)
    _result = qed_hep.Generator(qed_model).generate(
        _process,
        max_vertices=4,
        maximum_bridges=None,
        vertex_allow=["V_98", "V_99"],
        numerator_grouping=None,
        progress=None,
    )
    assert len(_result.diagrams) == 1
    qed_diagram = _result.diagrams[0]
    _cut = qed_diagram.cuts[0]
    _Q, _P, _K = S("gammalooprs::Q", "gammalooprs::P", "gammalooprs::K")
    qed_p1, qed_p2, qed_k1, qed_k2 = _P(0), _P(1), _K(0), S("qed_example::k2")
    qed_s = S("qed_example::s", is_positive=True)
    qed_t, qed_u = S("qed_example::t", "qed_example::u")
    qed_me, qed_mm, qed_e = S("UFO::Me", "UFO::MM", "UFO::ee")
    _kin = qed_hep.Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [qed_me**2, qed_me**2, qed_mm**2, qed_mm**2],
        [qed_s, qed_t, qed_u],
    )
    _projector = qed_diagram.projector_expression()
    for _edge in qed_diagram.external_edges:
        _projector = qed_model.particle_by_pdg(_edge.particle_pdg).sum_spins(
            _projector,
            _Q(_edge.id),
            edge=_edge.id,
            average=True,
        )
    _numerator = qed_model.expand_couplings(
        qed_diagram.numerator_expression().to_expression() * _projector
    )
    _contracted = idenso.to_dots(idenso.simplify_gamma(_numerator.expand()).expand())
    _contracted = _kin.apply(
        qed_diagram.loop_momentum_basis.route_expression(_contracted)
    )
    _denominator = qed_diagram.denominator_expression(
        edge_powers={_edge.id: 0 for _edge in _cut.edges},
        dimension=4,
        in_lmb=True,
    ).to_expression()
    _a, _b, _c, _inverse = S(
        "qed_example::a_", "qed_example::b_", "qed_example::c_", "qed_example::inverse_"
    )
    _denominator = _kin.apply(
        _denominator.replace(S("gammalooprs::denom")(_a, _b, _c, _inverse), _inverse)
    )
    qed_squared = (
        qed_diagram.overall_factor_expression(evaluate=True)
        * qed_diagram.numerator_prefactor_expression()
        * _contracted
        / _denominator
    ).together()
    _expected = (
        2
        * qed_e**4
        / qed_s**2
        * (
            2 * qed_me**2 * (2 * qed_mm**2 + qed_s - qed_t - qed_u)
            + 2 * qed_me**4
            + 2 * qed_mm**4
            + 2 * qed_mm**2 * (qed_s - qed_t - qed_u)
            + qed_t**2
            + qed_u**2
        )
    )
    assert (qed_squared - _expected).replace(
        qed_u, 2 * qed_me**2 + 2 * qed_mm**2 - qed_s - qed_t
    ).together() == E("0")
    mo.vstack(
        [
            mo.md("**Generated graph**"),
            qed_diagram,
            mo.md("**Massive, spin-averaged squared matrix element**"),
            qed_squared,
        ]
    )
    return (
        qed_e,
        qed_hep,
        qed_k1,
        qed_k2,
        qed_me,
        qed_mm,
        qed_model,
        qed_p1,
        qed_p2,
        qed_s,
        qed_squared,
        qed_t,
        qed_u,
    )


@app.cell
def _(
    E,
    Expression,
    S,
    mo,
    qed_e,
    qed_hep,
    qed_k1,
    qed_k2,
    qed_me,
    qed_mm,
    qed_p1,
    qed_p2,
    qed_s,
    qed_squared,
    qed_t,
    qed_u,
):
    qed_cos_theta, qed_alpha = S("qed_example::cos_theta", "qed_example::alpha")
    _pi = Expression.PI
    _kin = qed_hep.Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [E("0")] * 4,
        [qed_s, qed_t, qed_u],
    )
    _angular = (
        qed_squared.replace(qed_me, E("0"))
        .replace(qed_mm, E("0"))
        .replace(qed_t, -qed_s * (1 - qed_cos_theta) / 2)
        .replace(qed_u, -qed_s * (1 + qed_cos_theta) / 2)
        .replace(qed_e**4, (4 * _pi * qed_alpha) ** 2)
    )
    qed_differential = (
        _angular * _kin.two_body_phase_space(qed_k1, qed_k2) / _kin.flux(qed_p1, qed_p2)
    ).expand()
    _primitive = (
        qed_differential.to_polynomial().integrate(qed_cos_theta).to_expression()
    )
    qed_total = (
        2
        * _pi
        * (
            _primitive.replace(qed_cos_theta, E("1"))
            - _primitive.replace(qed_cos_theta, E("-1"))
        )
    ).together()
    assert (
        qed_differential - qed_alpha**2 * (1 + qed_cos_theta**2) / (4 * qed_s)
    ).together() == E("0")
    assert (qed_total - 4 * _pi * qed_alpha**2 / (3 * qed_s)).together() == E("0")
    mo.vstack(
        [
            mo.md(r"**Massless angular distribution** $d\sigma/d\Omega$"),
            qed_differential,
            mo.md(r"**Integrated cross section** $\sigma=4\pi\alpha^2/(3s)$"),
            qed_total,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Polarized QED currents

    Spenso constructs the chiral projectors and Idenso evaluates their traces.
    The currents below select right-chiral fermion wavefunctions; in the
    massless limit these describe right-handed particles and left-handed
    antiparticles. No initial spin average is taken for specified states.
    This calculation assembles the physical currents explicitly.
    """)
    return


@app.cell
def _(
    E,
    S,
    idenso,
    mo,
    qed_e,
    qed_hep,
    qed_k1,
    qed_k2,
    qed_me,
    qed_mm,
    qed_model,
    qed_p1,
    qed_p2,
    qed_s,
    qed_t,
    qed_u,
):
    from symbolica.community.spenso import Representation as _Rep
    from symbolica.community.spenso import TensorExpression as _Tensor
    from symbolica.community.spenso import chain as _chain

    _i0, _i1, _i2, _i3, _j0, _j1, _j2, _j3, _mu, _nu = S(
        "chiral::i0",
        "chiral::i1",
        "chiral::i2",
        "chiral::i3",
        "chiral::j0",
        "chiral::j1",
        "chiral::j2",
        "chiral::j3",
        "chiral::mu",
        "chiral::nu",
    )
    _incoming = (
        qed_model.particle_by_pdg(-11).spin_sum(qed_p2, _i0, _i1)
        * _Tensor.gamma(4)(_i1, _i2, _mu).to_expression()
        * qed_model.particle_by_pdg(11).spin_sum(qed_p1, _i2, _i3)
        * _Tensor.gamma(4)(_i3, _i0, _nu).to_expression()
    )
    _outgoing = (
        qed_model.particle_by_pdg(13).spin_sum(qed_k1, _j0, _j1)
        * _Tensor.gamma(4)(_j1, _j2, _mu).to_expression()
        * qed_model.particle_by_pdg(-13).spin_sum(qed_k2, _j2, _j3)
        * _Tensor.gamma(4)(_j3, _j0, _nu).to_expression()
    )
    _a, _b, _ell = S("chiral::a_", "chiral::b_", "chiral::ell_")
    _pattern = _Tensor.gamma(4)(_a, _b, _ell).to_expression()
    _right_current = _chain(
        _Rep.bis(4)(_a),
        _Rep.bis(4)(_b),
        _Tensor.gamma(4)(_a, "middle", _ell),
        _Tensor.projp(4)("middle", _b),
    ).to_expression()
    _projected = (_incoming * _outgoing).replace(_pattern, _right_current)
    _traces = idenso.simplify_gamma(_projected.expand()).expand()
    _contracted = idenso.to_dots(idenso.simplify_epsilon(_traces).expand())
    _kin = qed_hep.Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [qed_me**2, qed_me**2, qed_mm**2, qed_mm**2],
        [qed_s, qed_t, qed_u],
    )
    qed_polarized = (_kin.apply(_contracted) * qed_e**4 / qed_s**2).together()
    assert (
        qed_polarized - 4 * qed_e**4 * (qed_me**2 + qed_mm**2 - qed_u) ** 2 / qed_s**2
    ).together() == E("0")
    qed_polarized_massless = qed_polarized.replace(qed_me, E("0")).replace(
        qed_mm, E("0")
    )
    assert (
        qed_polarized_massless - 4 * qed_e**4 * qed_u**2 / qed_s**2
    ).together() == E("0")
    mo.vstack(
        [
            mo.md("**Massive chiral-projected squared matrix element**"),
            qed_polarized,
            mo.md("**Massless polarized squared matrix element**"),
            qed_polarized_massless,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
