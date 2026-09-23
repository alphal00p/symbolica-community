import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive electron self-energy")


@app.cell
def _():
    import json
    import math

    import marimo as mo
    import numpy as np
    from symbolica import E, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.hep import oneloop
    from symbolica.community.spenso import TensorExpression

    _set_namespace("qed_full")

    return (
        E,
        Replacement,
        S,
        Symbol,
        TensorExpression,
        hep,
        json,
        math,
        mo,
        np,
        oneloop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massive electron self-energy

    [Browse notebooks](/) · [QED UV renormalization](/?file=hep/qed_renormalization.py) ·
    [Electron g−2](/?file=hep/gminus2.py) ·
    [Two-loop electron](/?file=hep/electron_two_loop.py)

    Generate the one-loop electron self-energy in a symbolic covariant gauge,
    project its two Dirac structures, and reduce the integrals with native IBP.
    The longitudinal photon creates a **squared photon denominator**. Shared
    OneLOop evaluates the remaining masters, including finite terms and the
    physical branch cut. A separate Feynman-parameter integral checks the answer.

    Use $D=4-2\epsilon$, $s=p^2$, $m>0$ and $a_4=e^2/(16\pi^2)$.
    The photon numerator convention is $-i[g^{\mu\nu}-(1-\xi)k^\mu k^\nu/k^2]$;
    $\xi=1$ is Feynman gauge and $\xi=0$ is Landau gauge.
    Write the inverse-propagator insertion as
    $$\Gamma_2^{(1)}=-i\Sigma=i a_4[V\,\not p+S\,m].$$
    The finite coefficient uses the OneLOop normalization and is **unrenormalized**.
    No on-shell equation is used to project the off-shell numerator.
    """)


@app.cell(hide_code=True)
def _(E, S, Symbol, TensorExpression, hep, json):
    model = hep.Model.standard_model()
    D, eps, xi, s = S("D", "eps", "xi", "s")
    specification = json.loads(model.to_json())
    for _propagator in specification["propagators"]:
        if _propagator["particle"] == "a":
            _propagator["numerator"] = (
                "-1𝑖*(UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
                "-(1-qed_full::xi)*UFO::P(UFO::idx(1,1))*UFO::P(UFO::idx(1,2))"
                "/spenso::dot(UFO::P(spenso::mink(4)),UFO::P(spenso::mink(4))))"
            )
    model = hep.Model.from_json(json.dumps(specification))
    _electron, _photon = (model.particle(_name) for _name in ("e-", "a"))
    assert _electron.mass_parameter == "Me"
    _vertices = [
        v
        for v in model.vertex_rules
        if sorted(v.particles)
        == sorted([_electron.name, _electron.antiname, _photon.name])
    ]
    assert len(_vertices) == 1
    result = model.generate_diagrams(
        [_electron],
        [_electron],
        loops=1,
        max_vertices=2,
        maximum_bridges=0,
        vertex_allow=_vertices,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(result.diagrams) == 1
    diagram = result.diagrams[0]
    K, P, mass, charge = S("gammalooprs::K", "gammalooprs::P", "UFO::Me", "UFO::ee")
    mink, bis, gamma, metric = S(
        "spenso::mink", "spenso::bis", "spenso::gamma", "spenso::g"
    )
    _index, _wave, _mu, args = S("index_", "wave_", "mu", "args___")
    _ordering, _value = S(
        "feynkit_generator_factor::ExternalFermionOrderingSign", "value_"
    )
    zero, one = E("0"), E("1")
    kinematics = hep.Kinematics(D, momenta=[K(0), P(0)]).with_scalar_product(
        P(0), P(0), s
    )
    family = diagram.integral_family(kinematics=kinematics)
    coordinates = S("d0", "d1")
    _q2 = kinematics.scalar_product(K(0), K(0))
    _pq = kinematics.scalar_product(P(0), K(0))
    assert (family.denominators[0] - _q2 + mass**2).expand() == zero
    assert (family.denominators[1] - s - _q2 + 2 * _pq).expand() == zero
    assert [e.particle_name for e in diagram.internal_edges] == [
        _electron.name,
        _photon.name,
    ]

    # The UFO vertex is -i e gamma; electron and photon propagator numerators
    # carry +i and -i. Their product gives the unweighted kernel -e^2 N.
    # Both generated e->e and e->gamma e amplitudes have an additional external
    # Wick-ordering factor -1. Establish that sign independently at tree level.
    _tree_result = model.generate_diagrams(
        [_electron],
        [_photon, _electron],
        max_vertices=1,
        vertex_allow=_vertices,
        numerator_grouping=None,
        progress=None,
    )
    assert len(_tree_result.diagrams) == 1
    tree_diagram = _tree_result.diagrams[0]
    _tree_ports = {}
    for _edge in tree_diagram.external_edges:
        _representation = mink if _edge.external_index == 1 else bis
        _tree_ports[_edge.external_index] = dict(
            next(
                tree_diagram.projector_expression().match(
                    _wave(_edge.id, _representation(4, _index)), max_level=0
                )
            )
        )[_index]
    _tree_num = model.expand_couplings(
        tree_diagram.numerator_expression().to_expression()
    ).replace(mink(4, _index), mink(D, _index))
    _tree_probe = gamma(
        bis(4, _tree_ports[0]), bis(4, _tree_ports[2]), mink(D, _tree_ports[1])
    ) / (4 * D)
    tree_coupling = (
        TensorExpression((_tree_num * _tree_probe).expand())
        .simplify_gamma()
        .expand()
        .to_expression()
    )
    assert (tree_coupling + Symbol.I * charge).expand() == zero
    raw_factor = diagram.overall_factor_expression()
    external_ordering = (
        raw_factor / raw_factor.replace(_ordering(_value), one)
    ).replace(_ordering(_value), _value)
    assert external_ordering == -one
    assert diagram.overall_factor_expression(evaluate=True) == external_ordering
    assert tree_diagram.overall_factor_expression(evaluate=True) == external_ordering
    assert (
        diagram.numerator_prefactor_expression()
        == tree_diagram.numerator_prefactor_expression()
        == one
    )
    assert (tree_coupling * external_ordering - Symbol.I * charge).expand() == zero

    _ports = {
        _edge.external_index: dict(
            next(
                diagram.projector_expression().match(
                    _wave(_edge.id, bis(4, _index)), max_level=0
                )
            )
        )[_index]
        for _edge in diagram.external_edges
    }
    numerator = (
        model.expand_couplings(
            diagram.numerator_expression(in_lmb=True).to_expression()
        )
        .replace(mink(4, _index), mink(D, _index))
        .replace(mink(4), mink(D))
    )
    probes = [
        gamma(bis(4, _ports[0]), bis(4, _ports[1]), mink(D, _mu))
        * P(0, mink(D, _mu))
        / (4 * s),
        metric(bis(4, _ports[0]), bis(4, _ports[1])) / (4 * mass),
    ]
    native_factor = (
        diagram.overall_factor_expression(evaluate=True)
        * diagram.numerator_prefactor_expression()
    )
    # Independent D-dimensional gamma contraction, k=p-q and q=K. The scalar
    # coefficients multiply pslash and m; no external on-shell equation is used.
    _photon_square = s + _q2 - 2 * _pq
    independent_numerators = [
        (3 - D - xi)
        + (D - 1 - xi) * (s - _pq) / s
        - 2 * (1 - xi) * (s - _pq) ** 2 / (s * _photon_square),
        D - 1 + xi,
    ]

    return (
        D,
        K,
        P,
        args,
        charge,
        coordinates,
        diagram,
        eps,
        external_ordering,
        family,
        independent_numerators,
        kinematics,
        mass,
        native_factor,
        numerator,
        one,
        probes,
        raw_factor,
        s,
        tree_coupling,
        tree_diagram,
        xi,
        zero,
    )


@app.cell(hide_code=True)
def _(diagram, family, mo, raw_factor, tree_coupling, tree_diagram):
    mo.vstack(
        [
            mo.md("**Generated tree vertex and electron self-energy**"),
            mo.hstack([tree_diagram, diagram]),
            mo.hstack([mo.md("Tree Dirac coefficient:"), tree_coupling]),
            mo.hstack([mo.md("Native factor:"), raw_factor]),
            mo.md(r"""
        The tree Dirac kernel is $-ie\gamma^\mu$. Both generated amplitudes carry
        the named external Wick-ordering sign $-1$, giving $+ie\gamma^\mu$ for
        the ordered tree amplitude. We preserve that factor throughout the loop
        calculation. Dividing by this same external-state convention at the end
        yields the amputated $\Gamma_2=-i\Sigma$ defined above.

        With $q$ the electron momentum, the generated family is
        $\Delta_0=q^2-m^2$ and $\Delta_1=(q-p)^2$.
        """),
            family,
        ]
    )


@app.cell
def _(
    D,
    K,
    P,
    Replacement,
    S,
    TensorExpression,
    args,
    charge,
    coordinates,
    external_ordering,
    family,
    hep,
    independent_numerators,
    kinematics,
    mass,
    native_factor,
    numerator,
    probes,
    s,
    xi,
    zero,
):
    terms_by_structure, targets, traced_coefficients = [], set(), []
    for _projector, _independent in zip(probes, independent_numerators, strict=True):
        _trace = (
            TensorExpression((numerator * _projector).expand())
            .simplify_gamma()
            .expand()
            .to_dots()
            .to_expression()
        )
        _scalar = (kinematics.apply(_trace) * native_factor / charge**2).together()
        assert (_scalar - _independent).together() == zero
        traced_coefficients.append(_scalar)
        _rational = family.rewrite_numerator(_scalar, coordinates).together().expand()
        _terms = []
        # The graph supplies one power of each denominator. Its longitudinal
        # numerator has another inverse photon square; retain that raised power.
        for _monomial, _coefficient in _rational.coefficient_list(*coordinates):
            _powers = tuple(
                1 - int((_monomial.derivative(c) * c / _monomial).together())
                for c in coordinates
            )
            assert (
                _monomial
                - coordinates[0] ** (1 - _powers[0])
                * coordinates[1] ** (1 - _powers[1])
            ).together() == zero
            assert not _coefficient.matches(K(args))
            assert not _coefficient.matches(P(args))
            assert all(_coefficient.derivative(c) == zero for c in coordinates)
            targets.add(_powers)
            _terms.append((_powers, _coefficient))
        terms_by_structure.append(_terms)
    assert (1, 2) in targets
    solution = hep.IBPFamily(family, name="qed_full").reduce_laporta(
        [list(p) for p in sorted(targets)], max_depth=2
    )
    assert {tuple(p) for p in solution.residuals} == {(1, 0), (1, 1)}
    I, A, B = S("I", "A", "B")
    _x0, _x1 = S("x0", "x1")
    _U, _F = family.symanzik([_x0, _x1])
    assert (_U - _x0 - _x1).expand() == zero
    assert (_F - mass**2 * _x0 * _U + s * _x0 * _x1).expand() == zero
    # Hence I(1,0)=A0(m²), I(1,1)=B0(s; m²,0). Massless pinches are scaleless
    # and are removed by the shared IBP solver, not assigned a numerical master.
    _master_rules = [Replacement(I(1, 0), A), Replacement(I(1, 1), B)]
    native_coefficients = [
        sum(
            (
                _coefficient * solution.reduce(list(_powers), integral=I)
                for _powers, _coefficient in _terms
            ),
            zero,
        )
        .replace_multiple(_master_rules)
        .together()
        for _terms in terms_by_structure
    ]
    # Native ordered matrix element = i*a4*[V_native pslash + S_native m],
    # a4=e²/(16π²). The amputated inverse-propagator insertion Γ₂=-iΣ is the
    # ordered result divided by its explicitly established external Wick sign.
    # Thus conventional Σ/a4 has the native coefficients, while Γ₂/(i*a4)
    # has their opposites. No internal graph factor is changed or discarded.
    amputated_coefficients = [
        (_coefficient / external_ordering).together()
        for _coefficient in native_coefficients
    ]
    exact_reference = [
        xi * (D - 2) * ((s + mass**2) * B - A) / (2 * s),
        -(D - 1 + xi) * B,
    ]
    for _native, _amputated, _reference in zip(
        native_coefficients, amputated_coefficients, exact_reference, strict=True
    ):
        assert (_amputated - _reference).together() == zero
        assert (_native + _reference).together() == zero
        for _master in (A, B):
            _coefficient = _amputated.expand().coefficient(_master).together()
            assert _coefficient.series(D, 4, 0).to_expression().replace(
                D, 4
            ) == _coefficient.replace(D, 4)
            assert not _coefficient.matches(I(args))
            assert not _coefficient.matches(K(args))
            assert all(_coefficient.derivative(c) == zero for c in coordinates)
    assert amputated_coefficients[0].replace(xi, 0).together() == zero

    return (
        A,
        B,
        I,
        amputated_coefficients,
        native_coefficients,
        solution,
        targets,
        traced_coefficients,
    )


@app.cell(hide_code=True)
def _(
    I,
    amputated_coefficients,
    mo,
    native_coefficients,
    solution,
    targets,
    traced_coefficients,
):
    mo.vstack(
        [
            mo.md(r"**Projected scalar integrands and native IBP**"),
            *traced_coefficients,
            mo.ui.table(
                [
                    {
                        "electron power": _powers[0],
                        "photon power": _powers[1],
                        "reduction": str(solution.reduce(list(_powers), integral=I)),
                    }
                    for _powers in sorted(targets)
                ],
                selection=None,
            ),
            mo.ui.table([solution.stats], selection=None),
            mo.md(r"""
        The projectors are $\operatorname{tr}(\not p\,N)/(4s)$ and
        $\operatorname{tr}(N)/(4m)$, with $\operatorname{tr}1=4$.
        Pinches containing only the massless photon are scaleless and vanish in
        the shared solver. The two masters are $A=A_0(m^2;\mu^2)$ and
        $B=B_0(s;m^2,0;\mu^2)$; their identification is checked by the family's
        Symanzik polynomials.

        **Native ordered coefficients** (divided by $i a_4$)
        """),
            *native_coefficients,
            mo.md(r"**Amputated coefficients** $V$ and $S$ (divided by $i a_4$)"),
            *amputated_coefficients,
            mo.md(r"""
        $$V=\frac{\xi(D-2)}{2s}[(s+m^2)B-A],\qquad
          S=-(D-1+\xi)B.$$
        The vector coefficient vanishes identically in Landau gauge.
        """),
        ]
    )


@app.cell
def _(
    A,
    B,
    D,
    E,
    S,
    amputated_coefficients,
    eps,
    mass,
    one,
    oneloop,
    s,
    xi,
    zero,
):
    Af, Bf, scale2, L = S("Af", "Bf", "scale2", "L")
    laurent = [
        _coefficient.replace(A, mass**2 / eps + Af)
        .replace(B, 1 / eps + Bf)
        .replace(D, 4 - 2 * eps)
        .series(eps, 0, 0)
        .to_expression()
        .expand()
        for _coefficient in amputated_coefficients
    ]
    _expected_poles = [xi, -(3 + xi)]
    _expected_finite = [xi * ((s + mass**2) * Bf - Af) / s - xi, -(3 + xi) * Bf + 2]
    finite = []
    for _expression, _pole, _reference in zip(
        laurent, _expected_poles, _expected_finite, strict=True
    ):
        assert _expression.coefficient(eps**-2) == zero
        assert (_expression.coefficient(eps**-1) - _pole).together() == zero
        _constant = dict(_expression.coefficient_list(eps))[one]
        assert (_constant - _reference).together() == zero
        finite.append(_constant)
    # Exact s->0 limit: Bf=1-L+s/(2m²)+... . No evaluation of the projector's
    # removable 1/s singularity at s=0. The on-shell value is finite but its
    # derivative is infrared singular, so this is not an on-shell Z2 calculation.
    zero_invariant = [
        _expression.replace(Af, mass**2 * (1 - L))
        .replace(Bf, 1 - L + s / (2 * mass**2))
        .series(s, 0, 0)
        .to_expression()
        .expand()
        for _expression in finite
    ]
    assert (zero_invariant[0] - xi * (E("1/2") - L)).expand() == zero
    assert (zero_invariant[1] + (3 + xi) * (1 - L) - 2).expand() == zero
    on_shell = [
        _expression.replace(s, mass**2)
        .replace(Af, mass**2 * (1 - L))
        .replace(Bf, 2 - L)
        .expand()
        for _expression in finite
    ]
    assert (sum(on_shell, zero) + 4 - 3 * L).expand() == zero
    assert (sum(_expected_poles, zero) + 3).expand() == zero
    assert sum(on_shell, zero).derivative(xi) == zero

    # Shared OneLOop callbacks. Preserve the D-dependent coefficients through the
    # Laurent expansion: the finite rational terms are -xi and +2 respectively.
    a_coefficients = oneloop.master_coefficients(
        oneloop.A0(mass**2, scale2)
    )
    b_coefficients = oneloop.master_coefficients(
        oneloop.B0(s, mass**2, 0, scale2)
    )
    finite_oneloop = [
        _expression.replace(Af, a_coefficients[0]).replace(Bf, b_coefficients[0])
        for _expression in finite
    ]

    return L, finite_oneloop, laurent, scale2, zero_invariant


@app.cell(hide_code=True)
def _(laurent, mo, zero_invariant):
    mo.vstack(
        [
            mo.md(r"**Laurent expansion, before evaluating the masters**"),
            *laurent,
            mo.md(r"""
        Put $A=m^2/\epsilon+A_f$ and $B=1/\epsilon+B_f$. Keeping $D$ until
        this step gives
        $$V_f=\xi\left[\frac{(s+m^2)B_f-A_f}{s}-1\right],\qquad
          S_f=-(3+\xi)B_f+2.$$
        The finite rational terms $-\xi$ and $+2$ come from $D$ multiplying
        the master poles. Setting $D=4$ before the Laurent expansion loses them.

        At $s=m^2$, $V_f+S_f=-4+3\log(m^2/\mu^2)$ is independent of gauge.
        This agrees with the conventional on-shell mass shift
        $\Sigma/m=a_4[3/\epsilon+4-3\log(m^2/\mu^2)]$.
        The value is finite after pole removal; its momentum derivative has an
        infrared singularity. This notebook does not compute on-shell $Z_2$.
        """),
            mo.md(r"**Continuous $s\to0$ limits**, with $L=\log(m^2/\mu^2)$"),
            *zero_invariant,
        ]
    )


@app.cell
def _(mo):
    mass_control = mo.ui.number(
        start=0.05, stop=500.0, step=0.05, value=1.0, label="Mass m"
    )
    scale_control = mo.ui.number(
        start=0.01, stop=250000.0, step=0.1, value=1.0, label="Scale μ²"
    )
    ratio_control = mo.ui.number(
        start=-10.0, stop=10.0, step=0.1, value=-1.0, label="p² / m²"
    )
    gauge_control = mo.ui.number(
        start=-2.0, stop=5.0, step=0.1, value=1.0, label="Gauge ξ"
    )
    mo.vstack(
        [
            mo.md("**Explore the finite amplitude**"),
            mo.hstack([mass_control, scale_control]),
            mo.hstack([ratio_control, gauge_control]),
        ]
    )
    return gauge_control, mass_control, ratio_control, scale_control


@app.cell(hide_code=True)
def _(np):
    parameter_rules = {}
    # All rules avoid endpoints. t^4 resolves the logarithmic endpoint while
    # keeping a polynomial Jacobian; 120/160 agreement checks convergence.
    for _count in (120, 160):
        _nodes, _weights = np.polynomial.legendre.leggauss(_count)
        parameter_rules[_count] = ((_nodes + 1) / 2, _weights / 2)

    return (parameter_rules,)


@app.cell
def _(
    L,
    finite_oneloop,
    gauge_control,
    mass,
    mass_control,
    math,
    mo,
    np,
    parameter_rules,
    ratio_control,
    s,
    scale2,
    scale_control,
    xi,
    zero_invariant,
):
    _ratio = ratio_control.value
    _mass_value = mass_control.value
    _scale_value = scale_control.value
    _gauge = gauge_control.value
    _invariant = _ratio * _mass_value**2
    _mass_squared = _mass_value**2
    quadrature_values = []
    for _count in (120, 160):
        _t, _weights = parameter_rules[_count]
        if _ratio == 0:
            _L0 = math.log(_mass_squared / _scale_value) - 1
            _L1 = math.log(_mass_squared / _scale_value) / 2 - 3 / 4
        elif _ratio == 1:
            _L0 = math.log(_mass_squared / _scale_value) - 2
            _L1 = math.log(_mass_squared / _scale_value) / 2 - 3 / 2
        else:
            # Peel math.log(x) analytically: integral math.log(x)=-1 and
            # integral (1-x)math.log(x)=-3/4. Only math.log(m²-s+s*x) remains.
            _L0, _L1 = -1 + 0j, -3 / 4 + 0j
            if _ratio > 1:
                _root = 1 - 1 / _ratio
                _segments = [
                    (_root, -1.0, _root, -1j * math.pi),
                    (_root, 1.0, 1 - _root, 0j),
                ]
                for _origin, _direction, _length, _phase in _segments:
                    _x = _origin + _direction * _length * _t**4
                    _jacobian = 4 * _length * _t**3
                    # The linear factor is exactly +/-s*length*t^4. Do not
                    # reconstruct it by cancellation near its internal root.
                    _logarithm = (
                        math.log(_invariant * _length / _scale_value)
                        + 4 * np.log(_t)
                        + _phase
                    )
                    _L0 += np.sum(_weights * _jacobian * _logarithm)
                    _L1 += np.sum(_weights * _jacobian * (1 - _x) * _logarithm)
            else:
                _x = _t**4 if _ratio > 0 else 1 - _t**4
                _jacobian = 4 * _t**3
                _linear_factor = (
                    _mass_squared * (1 - _ratio) + _invariant * _t**4
                    if _ratio > 0
                    else _mass_squared - _invariant * _t**4
                )
                _logarithm = np.log(_linear_factor / _scale_value)
                _L0 += np.sum(_weights * _jacobian * _logarithm)
                _L1 += np.sum(_weights * _jacobian * (1 - _x) * _logarithm)
        quadrature_values.append(
            np.array([-_gauge * (1 + 2 * _L1), 2 + (3 + _gauge) * _L0], dtype=complex)
        )
    convergence_error = float(
        np.max(np.abs(quadrature_values[0] - quadrature_values[1]))
    )
    assert convergence_error < 2e-11
    _point = {s: _invariant, mass: _mass_value, scale2: _scale_value, xi: _gauge}
    if _ratio == 0:
        # Evaluate the exact symbolic limit instead of a 0/0 master combination.
        values = [
            complex(
                _expr.evaluate({xi: _gauge, L: math.log(_mass_squared / _scale_value)})
            )
            for _expr in zero_invariant
        ]
    else:
        values = [complex(_expr.evaluate(_point)) for _expr in finite_oneloop]
    parameter_error = float(np.max(np.abs(np.array(values) - quadrature_values[1])))
    _conditioning = (
        max(1.0, abs(_gauge) * (1 + 1 / abs(_ratio)), 3 + abs(_gauge))
        if _ratio
        else max(1.0, 3 + abs(_gauge))
    )
    tolerance = 2e-11 * _conditioning
    assert parameter_error < tolerance, (parameter_error, tolerance)

    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "Structure": label,
                        "OneLOop / exact zero-momentum limit": str(value),
                        "Parameter integral": str(complex(reference)),
                        "Absolute difference": float(abs(value - reference)),
                    }
                    for label, value, reference in zip(
                        ("V finite", "S finite"),
                        values,
                        quadrature_values[1],
                        strict=True,
                    )
                ],
                selection=None,
            ),
            mo.md(
                f"Parameter check: **passed**. Absolute error {parameter_error:.2g}; "
                f"120/160-point quadrature difference {convergence_error:.2g}."
            ),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Independent parameter check.** Let
    $\Delta(x)=xm^2-x(1-x)s-i0$ and
    $$L_0=\int_0^1\log\frac{\Delta(x)}{\mu^2}\,dx,\qquad
    L_1=\int_0^1(1-x)\log\frac{\Delta(x)}{\mu^2}\,dx.$$
    Direct parameter integration gives $V_f=-\xi(1+2L_1)$ and
    $S_f=2+(3+\xi)L_0$. The quadrature splits at the internal root for
    $s>m^2$, retaining $\log(-a-i0)=\log a-i\pi$; the bubble then has positive
    imaginary part. At $s=0$ and $s=m^2$, exact parameter limits avoid a
    removable zero or an endpoint singularity.

    Near $s=0$, separate double-precision evaluation of $A_0$ and $B_0$ loses
    accuracy in their difference divided by $s$. The comparison tolerance
    accounts for this conditioning, and the exact symbolic limit is used at zero.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The exact-$D$ check uses the Abelian limit of
    [Davydychev, Osland and Saks](https://arxiv.org/abs/hep-ph/0008171),
    Eqs. (2.19)–(2.21), with their gauge parameter $\xi_{\rm paper}=1-\xi$.
    Appendix C.6 provides a separate analytic check of the scalar bubble.
    The [FeynCalc QED example](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/Renormalization2)
    displays the massive electron self-energy and then extracts its UV part;
    the notebook here retains its finite coefficient as well.

    Generation, tensor algebra, integral families, IBP and scalar master evaluation
    all use the shared components. The parameter integral above is an independent
    check, not a second master-integral evaluator in the Feynkit API.
    """)


if __name__ == "__main__":
    app.run()
