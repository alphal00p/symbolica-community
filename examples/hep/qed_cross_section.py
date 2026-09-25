import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="QED annihilation: unpolarized and polarized",
)


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.hep import Model
    from symbolica.community.spenso import Representation, TensorExpression, chain

    _set_namespace("qed_example")

    return E, Model, Representation, S, Symbol, TensorExpression, chain, hep, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/)
    """)
    return


@app.cell(hide_code=True)
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
def _(E, Model, S, hep, mo):
    qed_model = Model.standard_model()
    _process = qed_model.process(
        ["e-", "e+"], ["mu-", "mu+"], vertex_allow=["V_98", "V_99"]
    )
    _result = _process.generate_cross_section(
        loops=1,
        max_vertices=4,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(_result.diagrams) == 1
    qed_diagram = _result.diagrams[0]
    _cut = qed_diagram.cuts[0]
    _Q, _P, _K = S("gammalooprs::Q", "gammalooprs::P", "gammalooprs::K")
    qed_p1, qed_p2 = _P(0), _P(1)
    qed_k1, qed_k2 = S("k1", "k2")
    _loop_edge = qed_diagram.loop_momentum_basis.loop_edges[0]
    _loop_particle = next(
        _particle.name
        for _edge, _particle in zip(_cut.edges, _cut.particles)
        if _edge.id == _loop_edge
    )
    _physical_momentum = qed_k1 if _loop_particle == "mu-" else qed_k2
    _index = S("index_")
    qed_loop_pattern = _K(0, _index)
    qed_loop_replacement = _cut.orientations[_loop_edge] * _physical_momentum(_index)
    qed_s = S("s", is_positive=True)
    qed_t, qed_u = S("t", "u")
    qed_me, qed_mm, qed_e = S("UFO::Me", "UFO::MM", "UFO::ee")
    _kin = hep.Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [qed_me**2, qed_me**2, qed_mm**2, qed_mm**2],
        [qed_s, qed_t, qed_u],
    )
    _projector = qed_diagram.projector_expression()
    for _edge in qed_diagram.external_edges:
        _projector = qed_model.particle(_edge.particle_name).sum_spins(
            _projector,
            _Q(_edge.id),
            edge=_edge.id,
            average=True,
        )
    _numerator = qed_model.expand_couplings(
        (qed_diagram.numerator_expression() * _projector).schoonschip_net(
            expand_contracted_sums=True,
        )
    )
    _contracted = _numerator.expand().simplify_gamma().expand().to_dots()
    _contracted = _kin.apply(
        qed_diagram.loop_momentum_basis.route_expression(_contracted).replace(
            qed_loop_pattern, qed_loop_replacement
        )
    )
    _denominator = qed_diagram.denominator_expression(
        edge_powers={_edge.id: 0 for _edge in _cut.edges},
        dimension=4,
        in_lmb=True,
    )
    _a, _b, _c, _inverse = S("a_", "b_", "c_", "inverse_")
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
        qed_diagram,
        qed_e,
        qed_k1,
        qed_k2,
        qed_loop_pattern,
        qed_loop_replacement,
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
    S,
    Symbol,
    hep,
    mo,
    qed_e,
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
    qed_cos_theta, qed_alpha = S("cos_theta", "alpha")
    _pi = Symbol.PI
    _kin = hep.Kinematics.mandelstam(
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
    # The Jacobian d(cos(theta))/dt = 2/s gives the second gallery coordinate.
    qed_differential_t = (
        2 * qed_differential.replace(qed_cos_theta, 1 + 2 * qed_t / qed_s) / qed_s
    ).expand()
    _expected_t = (
        qed_alpha**2 * (qed_s**2 + 2 * qed_s * qed_t + 2 * qed_t**2) / qed_s**4
    )
    assert (qed_differential_t - _expected_t).together() == 0
    _primitive_t = qed_differential_t.to_polynomial().integrate(qed_t).to_expression()
    _total_t = (
        2 * _pi * (_primitive_t.replace(qed_t, 0) - _primitive_t.replace(qed_t, -qed_s))
    )
    assert (_total_t - qed_total).together() == 0
    mo.vstack(
        [
            mo.md(r"**Massless angular distribution** $d\sigma/d\Omega$"),
            qed_differential,
            mo.md(r"**Mandelstam distribution** $d\sigma/(dt\,d\varphi)$"),
            qed_differential_t,
            mo.md(
                r"**Both coordinates give the integrated cross section** $\sigma=4\pi\alpha^2/(3s)$"
            ),
            qed_total,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated polarized QED cross section

    Spenso constructs the chiral projectors and Idenso evaluates their traces.
    The currents below select right-chiral fermion wavefunctions; in the
    massless limit these describe right-handed particles and left-handed
    antiparticles. No initial spin average is taken for specified states.
    Apply the projectors to the generated interaction vertices. Cut particle
    identities and orientations fix the physical outgoing momenta.
    """)
    return


@app.cell
def _(
    E,
    Representation,
    S,
    TensorExpression,
    chain,
    hep,
    mo,
    qed_diagram,
    qed_e,
    qed_k1,
    qed_k2,
    qed_loop_pattern,
    qed_loop_replacement,
    qed_me,
    qed_mm,
    qed_model,
    qed_p1,
    qed_p2,
    qed_s,
    qed_t,
    qed_u,
):
    _a, _b, _ell = S("chiral::a_", "chiral::b_", "chiral::ell_")
    _pattern = TensorExpression.gamma(4)(_a, _b, _ell)
    _right_current = chain(
        Representation.bis(4)(_a),
        Representation.bis(4)(_b),
        TensorExpression.gamma(4)(_a, "middle", _ell),
        TensorExpression.projp(4)("middle", _b),
    )
    _numerator = TensorExpression(1)
    for _vertex in qed_diagram.vertices:
        _numerator *= _vertex.numerator_expression().replace(_pattern, _right_current)
    for _edge in qed_diagram.internal_edges:
        _numerator *= _edge.numerator_expression()
    _Q = S("gammalooprs::Q")
    _projector = qed_diagram.projector_expression()
    for _edge in qed_diagram.external_edges:
        _projector = qed_model.particle(_edge.particle_name).sum_spins(
            _projector, _Q(_edge.id), edge=_edge.id
        )
    _numerator = qed_model.expand_couplings(
        (_numerator * _projector).schoonschip_net(
            expand_contracted_sums=True,
        )
    )
    _contracted = (
        _numerator.expand()
        .simplify_gamma()
        .expand()
        .simplify_epsilon()
        .expand()
        .to_dots()
    )
    _kin = hep.Kinematics.mandelstam(
        [qed_p1, qed_p2, qed_k1, qed_k2],
        [qed_me**2, qed_me**2, qed_mm**2, qed_mm**2],
        [qed_s, qed_t, qed_u],
    )
    _contracted = _kin.apply(
        qed_diagram.loop_momentum_basis.route_expression(_contracted).replace(
            qed_loop_pattern, qed_loop_replacement
        )
    )
    _cut = qed_diagram.cuts[0]
    _denominator = qed_diagram.denominator_expression(
        edge_powers={_edge.id: 0 for _edge in _cut.edges}, dimension=4, in_lmb=True
    )
    _q, _m, _power, _inverse = S(
        "chiral::q_", "chiral::m_", "chiral::power_", "chiral::inverse_"
    )
    _denominator = _kin.apply(
        _denominator.replace(
            S("gammalooprs::denom")(_q, _m, _power, _inverse), _inverse
        )
    )
    qed_polarized = (
        (
            qed_diagram.overall_factor_expression(evaluate=True)
            * qed_diagram.numerator_prefactor_expression()
            * _contracted
            / _denominator
        )
        .replace(qed_t, 2 * qed_me**2 + 2 * qed_mm**2 - qed_s - qed_u)
        .factor()
    )
    assert (
        qed_polarized - 4 * qed_e**4 * (qed_me**2 + qed_mm**2 - qed_u) ** 2 / qed_s**2
    ).replace(qed_u, 2 * qed_me**2 + 2 * qed_mm**2 - qed_s - qed_t).together() == E("0")
    qed_polarized_massless = qed_polarized.replace(qed_me, E("0")).replace(
        qed_mm, E("0")
    )
    assert (qed_polarized_massless - 4 * qed_e**4 * qed_u**2 / qed_s**2).replace(
        qed_u, -qed_s - qed_t
    ).together() == E("0")
    mo.vstack(
        [
            mo.md("**Generated massive chiral-projected squared matrix element**"),
            qed_polarized,
            mo.md("**Generated massless polarized squared matrix element**"),
            qed_polarized_massless,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
