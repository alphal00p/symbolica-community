import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Quark annihilation into gluons")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.hep import Model
    from symbolica.community.spenso import CookSettings, TensorExpression

    _set_namespace("qcd_gluons")

    return CookSettings, E, Model, Replacement, S, TensorExpression, hep, mo


@app.cell
def _(Model):
    qcd_model = Model.standard_model()
    return (qcd_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quark annihilation into gluons

    [Browse all notebooks](/) · [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py)

    Generate $b\bar b\to gg$ with the bottom mass retained. The three diagrams
    include both quark-exchange channels and the three-gluon vertex. Summing
    amplitudes before forming their Dirac adjoint includes all interference.

    `Particle.color_sum` averages incoming colors and sums outgoing colors.
    `Particle.spin_sum` supplies fermion completeness and physical gluon
    polarizations. Idenso performs the color and Dirac algebra using Spenso
    representations, with $T_R=1/2$ and four-dimensional external spins.

    Two physical polarization references give the same massive $SU(N_c)$ result:
    the opposite final gluon and the incoming massive quark. The checks below
    compare against [FeynCalc](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QQbar-GlGl)
    and verify the massless $SU(3)$ limit and exchange of the final gluons.
    The symbolic calculation takes a few minutes.
    """)
    return


@app.cell
def _(CookSettings, E, Replacement, S, TensorExpression, hep, mo, qcd_model):
    _P = S("gammalooprs::P")
    _s, _t, _u, _mass, _gs = S("s", "t", "u", "UFO::MB", "UFO::G")
    _Nc, _dA, _cof, _coad = S("spenso::Nc", "dA", "spenso::cof", "spenso::coad")
    _kinematics = hep.Kinematics.mandelstam(
        [_P(0), _P(1), _P(2), _P(3)], [_mass**2, _mass**2, E("0"), E("0")], [_s, _t, _u]
    )
    gluon_generated = (
        hep.Process(qcd_model, ["b", "b~"], ["g", "g"])
        .with_loop_count(0, 0)
        .generate_diagrams(
            max_vertices=2,
            maximum_bridges=None,
            vertex_allow=["V_76", "V_36"],
            numerator_grouping=None,
            progress=None,
        )
    )
    assert len(gluon_generated.diagrams) == 3
    _ports = S("external_0", "external_1", "external_2", "external_3")
    _a, _b, _c, _inverse, _wave, _rep, _index = S(
        "a_", "b_", "c_", "inverse_", "wave_", "rep_", "index_"
    )
    _conjugate, _adjoint_index = S("spenso::conj", "adjoint_index")
    _amplitude = E("0")
    _denominators = []
    for _diagram in gluon_generated.diagrams:
        _numerator = qcd_model.expand_couplings(
            _diagram.numerator_expression(in_lmb=True).to_expression()
        )
        # Align both spin and color ports using each generated external wavefunction.
        # This preserves interference without depending on internal half-edge numbering.
        for _edge in _diagram.external_edges:
            _match = dict(
                next(
                    _diagram.projector_expression().match(
                        _wave(_edge.id, _rep(4, _index)), max_level=0
                    )
                )
            )
            _numerator = _numerator.replace(
                _match[_index], _ports[_edge.external_index]
            )
        _denominator = _kinematics.apply(
            _diagram.denominator_expression(dimension=4, in_lmb=True)
            .to_expression()
            .replace(S("gammalooprs::denom")(_a, _b, _c, _inverse), _inverse)
        ).expand()
        _denominators.append(_denominator)
        _amplitude += (
            _numerator
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
            / _denominator
        )
    assert all(_d in _denominators for _d in (_s, _t - _mass**2, _u - _mass**2))
    _operator = TensorExpression(_amplitude.expand())
    assert len(_operator.structure.slots) == 8
    _adjoint = _operator.dirac_adjoint().expand().simplify_gamma0().to_expression()
    _adjoint = _adjoint.replace(_conjugate(_P(_a, _b)), _P(_a, _b))
    for _real in (_mass, _gs, _s, _t, _u):
        _adjoint = _adjoint.replace(_conjugate(_real), _real)
    _adjoint = TensorExpression(_adjoint).wrap_indices(_adjoint_index)

    # Match the shared completeness tensor against the dual of each external slot.
    # Matching selects color slots and determines the quark/antiquark orientation.
    _left, _right, _metric = S("left_", "right_", "spenso::g")
    gluon_color_projector = E("1")
    _initial_colors = 1
    for _position, _name in enumerate(("b", "b~", "g", "g")):
        _particle = qcd_model.particle(_name)
        if _position < 2:
            _initial_colors *= abs(_particle.color)
        _matches = []
        for _slot in _operator.structure.slots:
            _original = _slot.to_expression()
            if _original.replace(_ports[_position], E("0")) == _original:
                continue
            _closure = _metric(
                _slot.dual().to_expression(),
                _original.replace(_ports[_position], _adjoint_index(_ports[_position])),
            )
            _match = next(
                _closure.match(_particle.color_sum(_left, _right), max_level=0), None
            )
            if _match is not None:
                _matches.append(_match)
        assert len(_matches) == 1
        _indices = dict(_matches[0])
        gluon_color_projector *= _particle.color_sum(
            _indices[_left], _indices[_right], average=_position < 2
        )

    # Spenso dimensions are integers or symbols. Keep dA symbolic until color
    # contraction, then impose the SU(N) relation and convert scalar Casimirs.
    _generic = (
        (
            _operator.to_expression()
            * _adjoint
            * gluon_color_projector
            * _initial_colors
            / _Nc**2
        )
        .replace(_cof(3, _index), _cof(_Nc, _index))
        .replace(_coad(8, _index), _coad(_dA, _index))
    )
    _colored = (
        TensorExpression(_generic, cook_indices=CookSettings.indices())
        .simplify_color()
        .to_expression()
        .replace(_dA, _Nc**2 - 1)
    )
    _colored = TensorExpression(_colored).to_cof_dimension_invariants()

    # Dirac adjunction exchanges the two endpoints of the open fermion chain.
    # Reduce the fermion trace before introducing the physical gluon projectors.
    _spin_projector = qcd_model.particle("b").spin_sum(
        _P(0), _ports[0], _adjoint_index(_ports[1]), average=True
    ) * qcd_model.particle("b~").spin_sum(
        _P(1), _adjoint_index(_ports[0]), _ports[1], average=True
    )
    _spin_summed = (
        TensorExpression(
            _colored.to_expression() * _spin_projector,
            cook_indices=CookSettings.indices(),
        )
        .expand()
        .simplify_gamma()
        .expand()
        .simplify_gamma()
    )
    _expected = (
        (_Nc**2 - 1)
        * _gs**4
        * (
            _mass**4 * (3 * _t**2 + 14 * _t * _u + 3 * _u**2)
            - _mass**2 * (_t**3 + 7 * _t**2 * _u + 7 * _t * _u**2 + _u**3)
            - 6 * _mass**8
            + _t * _u * (_t**2 + _u**2)
        )
        * (
            -2 * _Nc**2 * _mass**2 * (_t + _u)
            + 2 * _Nc**2 * _mass**4
            + _Nc**2 * (_t**2 + _u**2)
            - _s**2
        )
        / (2 * _Nc**3 * _s**2 * (_u - _mass**2) ** 2 * (_t - _mass**2) ** 2)
    )
    gluon_results = []
    for _references in ((_P(3), _P(2)), (_P(0), _P(0))):
        _polarizations = E("1")
        for _position, _reference in zip((2, 3), _references):
            _polarizations *= qcd_model.particle("g").spin_sum(
                _P(_position),
                _ports[_position],
                _adjoint_index(_ports[_position]),
                reference=_reference,
            )
        _scalar = (
            TensorExpression(
                _spin_summed.to_expression() * _polarizations,
                cook_indices=CookSettings.indices(),
            )
            .expand()
            .simplify_metrics()
            .to_dots()
        )
        assert _scalar.is_scalar
        _squared = (
            _kinematics.apply(_scalar.to_expression())
            .replace(_s, 2 * _mass**2 - _t - _u)
            .together()
        )
        assert (
            _squared - _expected.replace(_s, 2 * _mass**2 - _t - _u)
        ).together() == E("0")
        gluon_results.append(_squared)
        print(
            f"Generated massive SU(N) quark annihilation into gluons: references={_references} passed"
        )
    assert gluon_results[0] == gluon_results[1]

    gluon_massless = gluon_results[0].replace(_mass, E("0")).replace(_Nc, E("3"))
    _expected_massless = (
        E("32/27") * _gs**4 * (_t**2 + _u**2) / (_t * _u)
        - E("8/3") * _gs**4 * (_t**2 + _u**2) / _s**2
    )
    assert (gluon_massless - _expected_massless.replace(_s, -_t - _u)).together() == E(
        "0"
    )
    # Bose exchange acts on the labeled squared amplitude. A cross section integrated
    # over both identical-gluon labels would additionally require a factor of 1/2!.
    assert (
        gluon_results[0]
        - gluon_results[0].replace_multiple([Replacement(_t, _u), Replacement(_u, _t)])
    ).together() == E("0")
    print(
        "Massive gauge-reference independence, Bose symmetry and massless SU(3) reference passed"
    )

    mo.vstack(
        [
            mo.md("**The three generated amplitudes**"),
            mo.hstack(list(gluon_generated.diagrams)),
            mo.md("**Color completeness, including the initial average**"),
            gluon_color_projector,
            mo.md("**Massive spin- and color-averaged squared amplitude**"),
            gluon_results[0].factor(),
            mo.md("**Massless SU(3) limit**"),
            gluon_massless.factor(),
            mo.md(
                "Both polarization references and the Bose-exchange check agree. This is a labeled squared amplitude; integrating over both identical final-gluon labels requires the usual factor of 1/2!."
            ),
        ]
    )
    return gluon_generated, gluon_color_projector, gluon_results, gluon_massless


if __name__ == "__main__":
    app.run()
