import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Sewing-aware QED spin sums")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep as fk
    from symbolica.community.spenso import TensorExpression

    _set_namespace("sewn_qed")

    return E, S, TensorExpression, fk, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sewing-aware QED spin sums

    [Browse notebooks](/) · [Amplitude-based Compton calculation](/?file=hep/compton.py) ·
    [Sewn diphoton annihilation](/?file=hep/sewn_diphoton.py)

    Generate squared tree amplitudes as sewn forward graphs, retaining the fermion
    masses. Compare electron and positron Compton scattering, electron–positron
    annihilation into muons, and electron–muon scattering with the
    [FeynCalc tree examples](https://feyncalc.github.io/examples).

    The graph and generator preserve particle flow at the sewing boundary.
    `Particle.sum_spins` supplies the same physical completeness tensors used by
    ordinary amplitudes. Native graph factors include the relative permutation
    sign when open fermion chains close into cycles. The cut's physical particles
    and orientations determine the positive-energy final-state momenta.

    For Compton scattering, covariant and timelike axial photon projectors are
    checked separately against the complete massive result. The two choices
    must agree. These are squared matrix elements; phase-space integration is
    separate.
    """)


@app.cell(hide_code=True)
def _(E, S, TensorExpression, fk):
    _model = fk.Model.standard_model()
    sewn_results = {}
    _Q, _P, _K = S("gammalooprs::Q", "gammalooprs::P", "gammalooprs::K")
    _s, _t, _u, _me, _mm, _e = S("s", "t", "u", "UFO::Me", "UFO::MM", "UFO::ee")
    _k1, _k2, _index = S("k1", "k2", "index_")
    _a, _b, _c, _inverse = S("a_", "b_", "c_", "inverse_")
    _electron, _muon, _photon = (_model.particle(_name) for _name in ("e-", "mu-", "a"))
    _vertices = [
        v
        for v in _model.vertex_rules
        if sorted(v.particles)
        in (
            sorted([_electron.antiname, _electron.name, _photon.name]),
            sorted([_muon.antiname, _muon.name, _photon.name]),
        )
    ]
    assert len(_vertices) == 2

    for _name, _incoming, _outgoing in [
        ("electron Compton", ["e-", "a"], ["e-", "a"]),
        ("positron Compton", ["e+", "a"], ["e+", "a"]),
        ("annihilation", ["e-", "e+"], ["mu-", "mu+"]),
        ("electron-muon scattering", ["e-", "mu-"], ["e-", "mu-"]),
    ]:
        _compton = _name.endswith("Compton")
        _masses = (
            [_me**2, E("0"), _me**2, E("0")]
            if _compton
            else [_me**2, _me**2, _mm**2, _mm**2]
            if _name == "annihilation"
            else [_me**2, _mm**2, _me**2, _mm**2]
        )
        _kin = fk.Kinematics.mandelstam([_P(0), _P(1), _k1, _k2], _masses, [_s, _t, _u])
        _result = _model.process(
            _incoming, _outgoing, vertex_allow=_vertices
        ).generate_cross_section(
            loops=1,
            max_vertices=4,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_result.diagrams) == (4 if _compton else 1)
        if _compton:
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
        elif _name == "annihilation":
            _expected = (
                2
                * _e**4
                / _s**2
                * (
                    2 * _me**2 * (2 * _mm**2 + _s - _t - _u)
                    + 2 * _me**4
                    + 2 * _mm**4
                    + 2 * _mm**2 * (_s - _t - _u)
                    + _t**2
                    + _u**2
                )
            )
        else:
            _expected = (
                2
                * _e**4
                / _t**2
                * (
                    -2 * _me**2 * (-2 * _mm**2 + _s - _t + _u)
                    + 2 * _me**4
                    + 2 * _mm**4
                    - 2 * _mm**2 * (_s - _t + _u)
                    + _s**2
                    + _u**2
                )
            )
        for _reference in [None, _P(0)] if _compton else [None]:
            _squared = E("0")
            for _diagram in _result.diagrams:
                assert len(_diagram.cuts) == 1
                _cut = _diagram.cuts[0]
                assert sorted(p.name for p in _cut.particles) == sorted(_outgoing)
                _coordinate = next(
                    _edge.id
                    for _edge, particle in zip(_cut.edges, _cut.particles)
                    if particle.name == _outgoing[0]
                )
                _diagram = _diagram.with_loop_momentum_edges([_coordinate])
                _cut = _diagram.cuts[0]
                assert all(side.loop_count == 0 for side in (_cut.left, _cut.right))
                _orientation = _cut.orientations[_coordinate]
                _projector = _diagram.projector_expression()
                for _edge in _diagram.external_edges:
                    _projector = _model.particle(_edge.particle_name).sum_spins(
                        _projector,
                        _Q(_edge.id),
                        edge=_edge.id,
                        average=True,
                        reference=_reference,
                        covariant=_reference is None,
                    )
                _numerator = _model.expand_couplings(
                    _diagram.numerator_expression().to_expression() * _projector
                )
                _numerator = (
                    TensorExpression(_numerator.expand())
                    .simplify_gamma()
                    .expand()
                    .to_dots()
                    .to_expression()
                )
                _numerator = _kin.apply(
                    _diagram.loop_momentum_basis.route_expression(_numerator).replace(
                        _K(0, _index), _orientation * _k1(_index)
                    )
                )
                _denominator = _diagram.denominator_expression(
                    edge_powers={_edge.id: 0 for _edge in _cut.edges},
                    dimension=4,
                    in_lmb=True,
                ).to_expression()
                _denominator = _kin.apply(
                    _denominator.replace(
                        S("gammalooprs::denom")(_a, _b, _c, _inverse), _inverse
                    ).replace(_K(0, _index), _orientation * _k1(_index))
                )
                _squared += (
                    _diagram.overall_factor_expression(evaluate=True)
                    * _diagram.numerator_prefactor_expression()
                    * _numerator
                    / _denominator
                )
            _difference = (
                (_squared - _expected).replace(_t, sum(_masses) - _s - _u).together()
            )
            assert _difference == 0, (_name, _reference, _difference)
            if _compton:
                _massless = _squared.replace(_t, 2 * _me**2 - _s - _u).replace(
                    _me, E("0")
                )
                assert (_massless + 2 * _e**4 * (_s / _u + _u / _s)).together() == 0
            _label = _name + (" · covariant" if _reference is None else " · axial n=p₁")
            sewn_results[_label] = {
                "result": _squared.replace(_t, sum(_masses) - _s - _u).together(),
                "reference": _expected.replace(_t, sum(_masses) - _s - _u).together(),
                "residual": _difference,
                "diagrams": _result.diagrams,
                "incoming": _incoming,
                "outgoing": _outgoing,
            }
            print(
                f"Sewn {_name}, reference={_reference}: exact massive result passed",
                flush=True,
            )

    return (sewn_results,)


@app.cell(hide_code=True)
def _(mo, sewn_results):
    selected_case = mo.ui.dropdown(
        list(sewn_results),
        value=next(iter(sewn_results)),
        label="Process and incoming-photon projector",
    )
    mo.vstack([selected_case])
    return (selected_case,)


@app.cell(hide_code=True)
def _(mo, selected_case, sewn_results):
    selected = sewn_results[selected_case.value]
    assert selected["residual"] == 0
    mo.vstack(
        [
            mo.md("## Generated sewn diagrams"),
            mo.hstack(selected["diagrams"]),
            mo.md(
                r"Each cut separates two tree amplitudes. Incoming spins are averaged; final spins are summed. Typed particles and vertex rules select the QED interactions."
            ),
            mo.md("**Massive squared matrix element**"),
            selected["result"],
            mo.md("**Independent reference**"),
            selected["reference"],
            mo.md("**Exact difference after the Mandelstam relation**"),
            selected["residual"],
        ]
    )


if __name__ == "__main__":
    app.run()
