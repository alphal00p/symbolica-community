import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive Compton scattering")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.hep import Model
    from symbolica.community.spenso import CookSettings, TensorExpression

    _set_namespace("compton")

    return CookSettings, E, Model, S, TensorExpression, hep, mo


@app.cell
def _(Model):
    qed_model = Model.standard_model()
    return (qed_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated Compton scattering

    Generate the two $e^-\gamma \to e^-\gamma$ amplitudes, form their Dirac
    adjoint with Idenso, and contract the physical spin sums shared with
    GammaLoop. Both diagrams and their interference are retained. This agrees
    with the full massive [FeynCalc result](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElGa-ElGa).

    External ports come from the generated wavefunctions. The adjoint gets its
    own index names; the mass, charge, external momenta and invariants are real.
    Incoming spins are averaged and outgoing spins are summed.
    """)
    return


@app.cell
def _(CookSettings, E, S, TensorExpression, hep, mo, qed_model):
    _result = qed_model.process(
        ["e-", "a"], ["e-", "a"], vertex_allow=["V_98"]
    ).generate_diagrams(
        loops=0,
        max_vertices=2,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    _P = S("gammalooprs::P")
    _s, _t, _u, _me, _e = S("s", "t", "u", "UFO::Me", "UFO::ee")
    _kin = hep.Kinematics.mandelstam(
        [_P(0), _P(1), _P(2), _P(3)], [_me**2, E("0"), _me**2, E("0")], [_s, _t, _u]
    )
    _a, _b, _c, _inv = S("a_", "b_", "c_", "inv_")
    _wave, _rep, _index = S("wave_", "rep_", "index_")
    _ports = S("external_0", "external_1", "external_2", "external_3")
    _amplitude = E("0")
    for _diagram in _result.diagrams:
        _numerator = qed_model.expand_couplings(
            _diagram.numerator_expression(in_lmb=True)
        )
        for _edge in _diagram.external_edges:
            _matches = list(
                _diagram.projector_expression().match(
                    _wave(_edge.id, _rep(4, _index)), max_level=0
                )
            )
            assert len(_matches) == 1
            _match = dict(_matches[0])
            _numerator = _numerator.rename_indices(
                {_match[_index]: _ports[_edge.external_index]}
            )
        _denominator = _diagram.denominator_expression(dimension=4, in_lmb=True)
        _denominator = _kin.apply(
            _denominator.replace(S("gammalooprs::denom")(_a, _b, _c, _inv), _inv)
        )
        _amplitude += (
            _numerator
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
            / _denominator
        )
    _operator = _amplitude.expand()
    _adjoint = _operator.dirac_adjoint().expand().simplify_gamma0()
    _conj = S("spenso::conj")
    _wrapper = S("conjugate_index")
    _real_adjoint = _adjoint.replace(_conj(_P(_a, _b)), _P(_a, _b))
    for _real in (_me, _e, _s, _t, _u):
        _real_adjoint = _real_adjoint.replace(_conj(_real), _real)
    _wrapped = _real_adjoint.wrap_indices(_wrapper)
    _spins = E("1")
    for _name, _momentum, _left, _right, _average in [
        ("e-", _P(0), _ports[0], _wrapper(_ports[2]), True),
        ("e-", _P(2), _wrapper(_ports[0]), _ports[2], False),
        ("a", _P(1), _ports[1], _wrapper(_ports[1]), True),
        ("a", _P(3), _ports[3], _wrapper(_ports[3]), False),
    ]:
        _spins *= qed_model.particle(_name).spin_sum(
            _momentum, _left, _right, average=_average
        )
    _product = _operator.to_expression() * _wrapped * _spins
    compton_squared = (
        _kin.apply(
            TensorExpression(_product.expand(), cook_indices=CookSettings.indices())
            .simplify_gamma()
            .expand()
            .to_dots()
        )
        .replace(_t, 2 * _me**2 - _s - _u)
        .together()
    )
    _expected = (
        2
        * _e**4
        * (
            -(_me**4) * (3 * _s**2 + 14 * _s * _u + 3 * _u**2)
            + _me**2 * (_s**3 + 7 * _s**2 * _u + 7 * _s * _u**2 + _u**3)
            + 6 * _me**8
            - _s * _u * (_s**2 + _u**2)
        )
        / ((_s - _me**2) ** 2 * (_u - _me**2) ** 2)
    )
    assert (compton_squared - _expected).together() == E("0")
    mo.vstack(
        [
            mo.hstack(_result.diagrams),
            mo.md("**Massive, spin-averaged squared matrix element**"),
            compton_squared,
            mo.md("**Massless limit**"),
            compton_squared.replace(_me, E("0")).together(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
