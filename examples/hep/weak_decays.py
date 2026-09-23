import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Charged-current W and top decays")


@app.cell
def _():
    import math

    import marimo as mo
    from symbolica import E, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.spenso import CookSettings, TensorExpression

    _set_namespace("weak_decay")

    return (
        CookSettings,
        E,
        Replacement,
        S,
        Symbol,
        TensorExpression,
        hep,
        math,
        mo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Charged-current W and top decays
    [Browse notebooks](/) · [W-pair production](/?file=hep/ww_production.py) · [Muon decay](/?file=hep/muon_decay.py) · [Z decays](/?file=hep/z_decay.py) · [Higgs decays](/?file=hep/higgs_decay.py)

    Generate eight channels from `Model.standard_model()`: charged W decays to leptons, unequal-mass quarks or massless light quarks, together with top and antitop decay. Keep the chiral vertex and complex CKM factor, sum physical final polarizations and average the initial spin. The full massive-vector spin sum includes the longitudinal polarization.

    The symbolic comparisons reproduce the [leptonic W](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/W-ElAnel), [quark W](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/W-QiQjbar) and [top](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/Qt-QbW) references. Massive W quark channels use the model's charm/bottom fields to retain two nonzero, unequal masses. Their formula is independent of the chosen generations.

    Write $H$ for the parent mass, $x,y$ for the daughter masses and $V=V_R+iV_I$. All widths use the shared two-body phase-space and decay-flux methods, with $e^2=4\sqrt2 G_Fm_W^2\sin^2\theta_W$. The physical domain is $H>x+y$; the numerical checks also include the threshold limit.
    """)


@app.cell
def _(CookSettings, E, Replacement, S, Symbol, TensorExpression, hep):
    _model = hep.Model.standard_model()
    _P = S("gammalooprs::P")
    _charge, _sw, _mw = S("UFO::ee", "UFO::sw", "UFO::MW")
    _me, _mc, _mb, _mt = S("UFO::Me", "UFO::MC", "UFO::MB", "UFO::MT")
    _conjugate, _wrapped = S("spenso::conj", "adjoint_index")
    _ports = S("parent", "first", "second")
    _wave, _rep, _index, _left, _right = S(
        "wave_",
        "rep_",
        "index_",
        "left_",
        "right_",
    )
    _metric = S("spenso::g")
    Vr, Vi, GF = S("Vr", "Vi", "GF")
    H, x, y = (S(_name, is_positive=True) for _name in ("H", "x", "y"))
    zero, _one, _pi = (E("0"), E("1"), Symbol.PI)
    results, diagrams = ({}, {})
    for _label, _names, _masses, _ckm_name in [
        ("W- leptons", ("W-", "e-", "ve~"), (_mw, _me, zero), None),
        ("W+ leptons", ("W+", "e+", "ve"), (_mw, _me, zero), None),
        ("W- quarks", ("W-", "c~", "b"), (_mw, _mc, _mb), "CKM2x3"),
        ("W+ quarks", ("W+", "c", "b~"), (_mw, _mc, _mb), "CKM2x3"),
        ("W- light quarks", ("W-", "u~", "d"), (_mw, zero, zero), "CKM1x1"),
        ("W+ light quarks", ("W+", "u", "d~"), (_mw, zero, zero), "CKM1x1"),
        ("Top", ("t", "b", "W+"), (_mt, _mb, _mw), "CKM3x3"),
        ("Antitop", ("t~", "b~", "W-"), (_mt, _mb, _mw), "CKM3x3"),
    ]:
        _generated = _model.generate_diagrams(
            [_names[0]],
            list(_names[1:]),
            max_vertices=1,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == 1
        _diagram = _generated.diagrams[0]
        diagrams[_label] = _diagram
        assert _diagram.denominator_expression(dimension=4).to_expression() == _one
        _particles = [_model.particle(_name) for _name in _names]
        _numerator = _model.expand_couplings(
            _diagram.numerator_expression().to_expression()
        )
        _mixing = _one
        if _ckm_name is not None:
            _ckm = S("UFO::" + _ckm_name)
            _numerator = _numerator.replace(
                S("UFO::complexconjugate")(_ckm), Vr - Symbol.I * Vi
            ).replace(_ckm, Vr + Symbol.I * Vi)
            _mixing = Vr**2 + Vi**2
        for _edge in _diagram.external_edges:
            _matches = list(
                _diagram.projector_expression().match(
                    _wave(_edge.id, _rep(4, _index)), max_level=0
                )
            )
            assert len(_matches) == 1
            _numerator = _numerator.replace(
                dict(_matches[0])[_index], _ports[_edge.external_index]
            )
        _operator = TensorExpression(
            (
                _numerator
                * _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
            ).expand()
        )
        _adjoint = _operator.dirac_adjoint().expand().simplify_gamma0().to_expression()
        for _real in (_charge, _sw, _mw, _me, _mc, _mb, _mt, Vr, Vi):
            _adjoint = _adjoint.replace(_conjugate(_real), _real)
        _adjoint = TensorExpression(_adjoint).wrap_indices(_wrapped)
        _is_top = _names[0] in ("t", "t~")
        _vector_position = 2 if _is_top else 0
        if _is_top:
            _left_position, _right_position = (
                (1, 0) if not _particles[0].is_antiparticle else (0, 1)
            )
        else:
            _left_position = next(
                _i for _i in (1, 2) if not _particles[_i].is_antiparticle
            )
            _right_position = 3 - _left_position
        _spin_projector = (
            _particles[_left_position].spin_sum(
                _P(_left_position),
                _wrapped(_ports[_right_position]),
                _ports[_left_position],
                average=_left_position == 0,
            )
            * _particles[_right_position].spin_sum(
                _P(_right_position),
                _ports[_right_position],
                _wrapped(_ports[_left_position]),
                average=_right_position == 0,
            )
            * _particles[_vector_position].spin_sum(
                _P(_vector_position),
                _ports[_vector_position],
                _wrapped(_ports[_vector_position]),
                average=_vector_position == 0,
            )
        )
        _parent, _first, _second = _masses
        _kin = (
            hep.Kinematics()
            .with_scalar_product(_P(0), _P(0), _parent**2)
            .with_scalar_product(_P(1), _P(1), _first**2)
            .with_scalar_product(_P(2), _P(2), _second**2)
            .with_scalar_product(
                _P(1), _P(2), (_parent**2 - _first**2 - _second**2) / 2
            )
            .with_scalar_product(
                _P(0), _P(1), (_parent**2 + _first**2 - _second**2) / 2
            )
            .with_scalar_product(
                _P(0), _P(2), (_parent**2 - _first**2 + _second**2) / 2
            )
        )
        _physical_first = x if _first != zero else zero
        _physical_second = y if _second != zero else zero
        _physical_masses = [Replacement(_parent, H)]
        if _first != zero:
            _physical_masses.append(Replacement(_first, x))
        if _second != zero:
            _physical_masses.append(Replacement(_second, y))
        _phase_kin = (
            hep.Kinematics()
            .with_scalar_product(_P(0), _P(0), H**2)
            .with_scalar_product(_P(1), _P(1), _physical_first**2)
            .with_scalar_product(_P(2), _P(2), _physical_second**2)
            .with_scalar_product(
                _P(1), _P(2), (H**2 - _physical_first**2 - _physical_second**2) / 2
            )
        )
        _measure = _phase_kin.two_body_phase_space(_P(1), _P(2))
        _radicand = S("radicand_")
        for _match in list(_measure.match(_radicand.sqrt())):
            _value = dict(_match)[_radicand]
            _measure = _measure.replace(_value.sqrt(), _value.expand().sqrt())
        _flux = _phase_kin.flux(_P(0))
        assert _flux == 2 * H
        _kallen = (
            H**4
            + _physical_first**4
            + _physical_second**4
            - 2 * H**2 * _physical_first**2
            - 2 * H**2 * _physical_second**2
            - 2 * _physical_first**2 * _physical_second**2
        ).expand()
        _reference_measure = _kallen.sqrt() / (32 * _pi**2 * H**2)
        assert (_measure - _reference_measure).together() == zero
        for _convention in ("Physical", "Gallery color sum"):
            _color_projector = _one
            for _position, _particle in enumerate(_particles):
                if _particle.color == 1:
                    continue
                _matches = []
                for _slot in _operator.interface:
                    _original = _slot.to_expression()
                    if _original.replace(_ports[_position], zero) == _original:
                        continue
                    _closure = _metric(
                        _slot.dual().to_expression(),
                        _original.replace(
                            _ports[_position], _wrapped(_ports[_position])
                        ),
                    )
                    _match = next(
                        _closure.match(_particle.color_sum(_left, _right), max_level=0),
                        None,
                    )
                    if _match is not None:
                        _matches.append(dict(_match))
                assert len(_matches) == 1
                _color_projector *= _particle.color_sum(
                    _matches[0][_left],
                    _matches[0][_right],
                    average=_position == 0 and _convention == "Physical",
                )
            _contracted = (
                TensorExpression(
                    _operator.to_expression()
                    * _adjoint
                    * _spin_projector
                    * _color_projector,
                    cook_indices=CookSettings.indices(),
                )
                .expand()
                .simplify_gamma()
                .expand()
                .simplify_gamma()
                .simplify_color()
                .simplify_metrics()
                .to_dots()
            )
            assert _contracted.is_scalar
            _squared = (
                _kin.apply(_contracted.to_expression())
                .replace(_charge**2, 4 * E("2").sqrt() * GF * _mw**2 * _sw**2)
                .replace_multiple(_physical_masses)
                .together()
            )
            _color_factor = (
                (3 if _convention == "Gallery color sum" else 1)
                if _is_top
                else 3
                if _ckm_name is not None
                else 1
            )
            if _is_top:
                _polynomial = (H**2 - x**2) ** 2 + y**2 * (H**2 + x**2) - 2 * y**4
                _expected_squared = (
                    E("2").sqrt() * _color_factor * GF * _mixing * _polynomial
                )
            else:
                _a, _b = (_physical_first**2, _physical_second**2)
                _polynomial = 2 * H**4 - H**2 * (_a + _b) - (_a - _b) ** 2
                _expected_squared = (
                    2 * E("2").sqrt() * _color_factor * GF * _mixing * _polynomial / 3
                )
            assert (_squared - _expected_squared).together() == zero, (
                _label,
                _convention,
                _squared,
            )
            _width = (4 * _pi * _measure * _squared / _flux).together()
            _expected_width = (
                _color_factor
                * GF
                * _mixing
                * _kallen.sqrt()
                * _polynomial
                * E("2").sqrt()
                / (_pi * H**3 * (16 if _is_top else 24))
            )
            assert (_width - _expected_width).together() == zero
            results[_label, _convention] = {
                "squared": _squared,
                "width": _width,
                "mixing": _mixing,
                "first": _physical_first,
                "second": _physical_second,
            }
        _ratio = 3 if _is_top else 1
        assert (
            results[_label, "Gallery color sum"]["width"]
            - _ratio * results[_label, "Physical"]["width"]
        ).together() == zero
    for _negative, _positive in [
        ("W- leptons", "W+ leptons"),
        ("W- quarks", "W+ quarks"),
        ("W- light quarks", "W+ light quarks"),
        ("Top", "Antitop"),
    ]:
        for _convention in ("Physical", "Gallery color sum"):
            assert (
                results[_negative, _convention]["width"]
                - results[_positive, _convention]["width"]
            ).together() == zero
    return GF, H, Vi, Vr, diagrams, results, x, y, zero


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial color convention
    The physical top width averages its initial color, so the final bottom-color sum gives an overall factor one. The gallery sums initial colors and therefore has an extra factor three. Both choices use `Particle.color_sum`; the calculation proves their ratio exactly. W decays have no initial color, so both choices coincide.

    The squared amplitudes and widths of each charge-conjugate pair agree for generic $V_R,V_I$, with dependence only on $|V|^2=V_R^2+V_I^2$.
    """)


@app.cell
def _(E, GF, H, Replacement, Vi, Vr, math, results, x, y, zero):
    numeric_checks = []
    for (_label, _convention), _result in results.items():
        _is_top = _label in ("Top", "Antitop")
        _ratios = (
            [(0.0, 0.46), (0.03, 0.46), (0.3, 0.69), (0.3, 0.7)]
            if _is_top
            else [(0.0, 0.0), (0.01, 0.02), (0.2, 0.3), (0.45, 0.54)]
        )
        for _r1, _r2 in _ratios:
            _r1 = _r1 if _result["first"] != zero else 0.0
            _r2 = _r2 if _result["second"] != zero else 0.0
            _substitutions = [
                Replacement(H, E("100")),
                Replacement(x, E(str(100 * _r1))),
                Replacement(y, E(str(100 * _r2))),
                Replacement(GF, E("0.00001")),
                Replacement(Vr, E("0.8")),
                Replacement(Vi, E("0.3")),
            ]
            _actual = complex(
                _result["width"].replace_multiple(_substitutions).evaluate({})
            )
            _phase = math.sqrt(
                max(0.0, (1 - (_r1 + _r2) ** 2) * (1 - (_r1 - _r2) ** 2))
            )
            _mixing = 1.0 if "leptons" in _label else 0.8**2 + 0.3**2
            _colors = (
                (3 if _convention == "Gallery color sum" else 1)
                if _is_top
                else 1
                if "leptons" in _label
                else 3
            )
            _shape = (
                (1 - _r1**2) ** 2 + _r2**2 * (1 + _r1**2) - 2 * _r2**4
                if _is_top
                else 1 - (_r1**2 + _r2**2) / 2 - (_r1**2 - _r2**2) ** 2 / 2
            )
            _expected = (
                _colors
                * 1e-05
                * 100**3
                * _mixing
                * _phase
                * _shape
                / ((8 if _is_top else 6) * math.sqrt(2) * math.pi)
            )
            assert abs(_actual - _expected) < 2e-09, (
                _label,
                _convention,
                _r1,
                _r2,
                _actual,
                _expected,
            )
            numeric_checks.append((_label, _convention, _r1, _r2, _actual, _expected))
    return (numeric_checks,)


@app.cell
def _(mo):
    channel = mo.ui.dropdown(
        [
            "W- leptons",
            "W+ leptons",
            "W- quarks",
            "W+ quarks",
            "W- light quarks",
            "W+ light quarks",
            "Top",
            "Antitop",
        ],
        value="Top",
        label="Decay channel",
    )
    color_convention = mo.ui.dropdown(
        ["Physical", "Gallery color sum"],
        value="Physical",
        label="Initial color convention",
    )
    kinematic_point = mo.ui.dropdown(
        {
            "Light daughter limit": 0,
            "Separated masses": 1,
            "Heavy daughters": 2,
            "Closest to threshold": 3,
        },
        value="Separated masses",
        label="Kinematics",
    )
    mo.vstack([channel, color_convention, kinematic_point])
    return channel, color_convention, kinematic_point


@app.cell
def _(
    channel,
    color_convention,
    diagrams,
    kinematic_point,
    mo,
    numeric_checks,
    results,
):
    selected_result = results[channel.value, color_convention.value]
    _checks = [
        _row
        for _row in numeric_checks
        if _row[:2] == (channel.value, color_convention.value)
    ]
    selected_point = _checks[kinematic_point.value]
    _, _, _r1, _r2, selected_width, reference_width = selected_point
    assert abs(selected_width - reference_width) < 2e-09
    assert selected_width.real >= -1e-12 and abs(selected_width.imag) < 2e-09
    mo.vstack(
        [
            diagrams[channel.value],
            mo.md("## Spin-averaged squared amplitude"),
            selected_result["squared"],
            mo.md("## Total width"),
            selected_result["width"],
            mo.md("## Numerical comparison"),
            mo.ui.table(
                [
                    {
                        "Parent H (GeV)": 100,
                        "First daughter (GeV)": 100 * _r1,
                        "Second daughter (GeV)": 100 * _r2,
                        "GF (GeV⁻²)": 1e-05,
                        "CKM V": ".8 + .3i" if "leptons" not in channel.value else "1",
                    }
                ],
                selection=None,
            ),
            mo.md(f"**Generated width:** `{selected_width.real:.12g} GeV`"),
            mo.md(f"**Reference width:** `{reference_width:.12g} GeV`"),
            mo.md(
                f"**Absolute difference:** `{abs(selected_width - reference_width):.3g} GeV`"
            ),
            mo.md(
                "These are illustrative kinematic inputs; the symbolic formula above retains arbitrary masses, GF and CKM element. All eight channels and both color conventions pass the exact and numerical comparisons."
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
