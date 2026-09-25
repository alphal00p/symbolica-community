import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Gluon ghost subtraction")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.hep import Model
    from symbolica.community.spenso import CookSettings, TensorExpression

    _set_namespace("qcd_ghosts")

    return CookSettings, E, Model, Replacement, S, TensorExpression, hep, mo


@app.cell
def _(Model):
    ghost_model = Model.standard_model()
    return (ghost_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gluon ghost subtraction

    [Browse all notebooks](/) · [Physical polarization calculation](/?file=hep/qcd_gluons.py)

    Generate $b\bar b\to gg$ with the bottom mass retained and use covariant
    gluon polarization sums. These include unphysical states. Generate both
    ghost–antighost orderings separately and subtract their squared amplitudes
    to recover the physical result.

    All three gluon diagrams and their interference are included. Shared
    `Particle.color_sum` and `Particle.spin_sum` provide the initial averages
    and completeness tensors; Idenso and Spenso perform the algebra.
    Ghosts carry color but have no external polarization wavefunctions.

    The checks compare each ghost contribution and the final massive $SU(N_c)$
    result with [FeynCalc](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QQbar-GlGl-2),
    then verify the massless $SU(3)$ limit and exchange of the two final gluons.
    External spins are four-dimensional and $T_R=1/2$.
    The symbolic calculation takes a few minutes.
    """)
    return


@app.cell
def _(CookSettings, E, Replacement, S, TensorExpression, ghost_model, hep, mo):
    _P = S("gammalooprs::P")
    _s, _t, _u, _mass, _gs = S("s", "t", "u", "UFO::MB", "UFO::G")
    _Nc, _dA, _cof, _coad = S("spenso::Nc", "dA", "spenso::cof", "spenso::coad")
    _kinematics = hep.Kinematics.mandelstam(
        [_P(0), _P(1), _P(2), _P(3)], [_mass**2, _mass**2, E("0"), E("0")], [_s, _t, _u]
    )
    ghost_results = []
    ghost_generated = []
    for _outgoing, _vertices, _count in (
        (["ghG", "ghG~"], ["V_76", "V_35"], 1),
        (["ghG~", "ghG"], ["V_76", "V_35"], 1),
        (["g", "g"], ["V_76", "V_36"], 3),
    ):
        _generated = ghost_model.process(
            ["b", "b~"], _outgoing, vertex_allow=_vertices
        ).generate_diagrams(
            loops=0,
            max_vertices=2,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == _count
        ghost_generated.append(_generated)
        _ports = S("external_0", "external_1", "external_2", "external_3")
        _a, _b, _c, _inverse, _index = S("a_", "b_", "c_", "inverse_", "index_")
        _conjugate, _adjoint_index = S("spenso::conj", "adjoint_index")
        _amplitude = E("0")
        _denominators = []
        for _diagram in _generated.diagrams:
            _numerator = ghost_model.expand_couplings(
                _diagram.numerator_expression(in_lmb=True)
            )
            # Align external spin and color ports using the graph's native half-edge IDs.
            # Ghosts have no polarization wavefunctions; these IDs cover them as well.
            for _half in _diagram.half_edges:
                _edge = _half.edge.data
                if _edge.is_external:
                    _numerator = _numerator.rename_indices(
                        {
                            S("gammalooprs::hedge")(_half.data, 1): _ports[
                                _edge.external_index
                            ]
                        }
                    )
            _denominator = _kinematics.apply(
                _diagram.denominator_expression(dimension=4, in_lmb=True).replace(
                    S("gammalooprs::denom")(_a, _b, _c, _inverse), _inverse
                )
            ).expand()
            _denominators.append(_denominator)
            _amplitude += (
                _numerator
                * _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
                / _denominator
            )
        assert all(
            _d in _denominators
            for _d in ((_s, _t - _mass**2, _u - _mass**2) if _count == 3 else (_s,))
        )
        _operator = _amplitude.expand()
        assert len(_operator.structure.slots) == (8 if _count == 3 else 6)
        _adjoint = _operator.dirac_adjoint().expand().simplify_gamma0()
        _adjoint = _adjoint.replace(_conjugate(_P(_a, _b)), _P(_a, _b))
        for _real in (_mass, _gs, _s, _t, _u):
            _adjoint = _adjoint.replace(_conjugate(_real), _real)
        _adjoint = _adjoint.wrap_indices(_adjoint_index)

        # Match the shared completeness tensor against the dual of each external slot.
        # Matching selects color slots and determines the quark/antiquark orientation.
        _left, _right, _metric = S("left_", "right_", "spenso::g")
        _color_projector = E("1")
        _initial_colors = 1
        for _position, _name in enumerate(("b", "b~", *_outgoing)):
            _particle = ghost_model.particle(_name)
            if _position < 2:
                _initial_colors *= abs(_particle.color)
            _matches = []
            for _slot in _operator.structure.slots:
                _original = _slot.to_expression()
                if _original.replace(_ports[_position], E("0")) == _original:
                    continue
                _closure = _metric(
                    _slot.dual().to_expression(),
                    _original.replace(
                        _ports[_position], _adjoint_index(_ports[_position])
                    ),
                )
                _match = next(
                    _closure.match(_particle.color_sum(_left, _right), max_level=0),
                    None,
                )
                if _match is not None:
                    _matches.append(_match)
            assert len(_matches) == 1
            _indices = dict(_matches[0])
            _color_projector *= _particle.color_sum(
                _indices[_left], _indices[_right], average=_position < 2
            )

        # Spenso dimensions are integers or symbols. Keep dA symbolic until color
        # contraction, then impose the SU(N) relation and convert scalar Casimirs.
        _generic = (
            (
                _operator.to_expression()
                * _adjoint
                * _color_projector
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
        _spin_projector = ghost_model.particle("b").spin_sum(
            _P(0), _ports[0], _adjoint_index(_ports[1]), average=True
        ) * ghost_model.particle("b~").spin_sum(
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
        _polarizations = E("1")
        if _count == 3:
            for _position in (2, 3):
                _polarizations *= ghost_model.particle("g").spin_sum(
                    _P(_position),
                    _ports[_position],
                    _adjoint_index(_ports[_position]),
                    covariant=True,
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
            _kinematics.apply(_scalar).replace(_s, 2 * _mass**2 - _t - _u).together()
        )
        ghost_results.append(_squared)
        print("Finished channel", _outgoing, flush=True)

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
    _ghost_reference = (
        (_Nc**2 - 1) * _gs**4 * (_u - _mass**2) * (_t - _mass**2) / (4 * _Nc * _s**2)
    )
    for _result in ghost_results[:2]:
        assert (
            _result - _ghost_reference.replace(_s, 2 * _mass**2 - _t - _u)
        ).together() == E("0")
    assert ghost_results[0] == ghost_results[1]
    ghost_physical = (ghost_results[2] - ghost_results[0] - ghost_results[1]).together()
    assert (
        ghost_physical - _expected.replace(_s, 2 * _mass**2 - _t - _u)
    ).together() == E("0")
    print(
        "Both ghost contributions and massive SU(N) ghost-subtracted gluons passed",
        flush=True,
    )

    # The same massive physical reference was independently checked using two axial
    # polarization choices in installed_feyncalc_qcd_gluons.py.
    assert (ghost_results[2] - ghost_physical).together() != E("0")
    ghost_massless = ghost_physical.replace(_mass, E("0")).replace(_Nc, E("3"))
    _expected_massless = (
        E("32/27") * _gs**4 * (_t**2 + _u**2) / (_t * _u)
        - E("8/3") * _gs**4 * (_t**2 + _u**2) / _s**2
    )
    assert (ghost_massless - _expected_massless.replace(_s, -_t - _u)).together() == E(
        "0"
    )
    # A labeled squared amplitude has no identical-particle phase-space factor.
    assert (
        ghost_physical
        - ghost_physical.replace_multiple([Replacement(_t, _u), Replacement(_u, _t)])
    ).together() == E("0")
    print(
        "Massless SU(3), Bose symmetry and nonzero ghost correction passed", flush=True
    )

    mo.vstack(
        [
            mo.md("**Covariant gluon amplitudes**"),
            mo.hstack(list(ghost_generated[2].diagrams)),
            mo.md("**Both generated ghost orderings**"),
            mo.hstack([ghost_generated[0].diagrams[0], ghost_generated[1].diagrams[0]]),
            mo.md(
                "**Each ghost contribution (including initial spin and color averages)**"
            ),
            ghost_results[0].factor(),
            mo.md("**Covariant result before subtraction**"),
            ghost_results[2].factor(),
            mo.md(
                "**Physical result: covariant − ghost ordering 1 − ghost ordering 2**"
            ),
            ghost_physical.factor(),
            mo.md("**Massless SU(3) limit**"),
            ghost_massless.factor(),
            mo.md(
                "All exact checks passed. These are labeled squared amplitudes; integrating over both identical final-gluon labels requires the usual factor of 1/2!."
            ),
        ]
    )
    return ghost_generated, ghost_results, ghost_physical, ghost_massless


if __name__ == "__main__":
    app.run()
