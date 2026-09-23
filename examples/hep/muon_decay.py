import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Muon decay and three-body phase space",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolica import E, Replacement, S, Symbol
    from symbolica.community import hep
    from symbolica.community.spenso import CookSettings, TensorExpression

    return (
        CookSettings,
        E,
        Replacement,
        S,
        Symbol,
        TensorExpression,
        hep,
        mo,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Muon decay and three-body phase space
    [Browse notebooks](/) · [W and top decays](/?file=hep/weak_decays.py)

    Generate $\mu^-\to e^-\bar\nu_e\nu_\mu$ with the embedded Standard Model charged-current vertices. The existing W propagator supplies its complete unitary-gauge numerator, including the longitudinal term. Shared particle spin sums average the muon spin and sum all final spins.

    First compare the full squared amplitude with the [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/Mu-ElAnelNmu), then derive the Fermi limit. Two calls to `Kinematics.two_body_phase_space` give the recursive three-body measure. Symbolica integrates the resulting spectrum and finite-mass correction.

    Write $M=m_\mu$, $m=m_e$, $s=(p_e+p_{\bar\nu_e})^2$ and $t=(p_e+p_{\nu_\mu})^2$. Both neutrinos are massless. The coupling convention is $e^4=32G_F^2m_W^4\sin^4\theta_W$.
    """)


@app.cell
def _(CookSettings, E, Replacement, S, Symbol, TensorExpression, hep, mo):
    _model = hep.Model.standard_model()
    _muon, _electron = (_model.particle_by_pdg(13), _model.particle_by_pdg(11))
    _vertices = [
        _v
        for _v in _model.vertex_rules
        if any(_p in ("W-", "W+") for _p in _v.particles)
        and any(_p in ("e-", "e+", "mu-", "mu+") for _p in _v.particles)
    ]
    assert len(_vertices) == 4
    _generated = _model.generate_diagrams(
        [13],
        [11, -12, 14],
        max_vertices=2,
        maximum_bridges=None,
        vertex_allow=_vertices,
        numerator_grouping=None,
        progress=None,
    )
    assert len(_generated.diagrams) == 1
    diagram = _generated.diagrams[0]
    assert len(diagram.internal_edges) == 1
    assert abs(diagram.internal_edges[0].particle_pdg) == 24
    P = S("gammalooprs::P")
    _charge, _sw, _mw, _mm, _me = S(
        "UFO::ee", "UFO::sw", "UFO::MW", "UFO::MM", "UFO::Me"
    )
    GF, s, t, z = S(
        "muon_decay::GF", "muon_decay::s", "muon_decay::t", "muon_decay::inverse_W2"
    )
    M = S("muon_decay::M", is_positive=True)
    mass = S("muon_decay::m", is_positive=True)
    _conjugate, _wrapped = S("spenso::conj", "muon_decay::adjoint")
    _ports = S(
        "muon_decay::mu", "muon_decay::e", "muon_decay::antinue", "muon_decay::numu"
    )
    _wave, _rep, _index = S(
        "muon_decay::wave_", "muon_decay::rep_", "muon_decay::index_"
    )
    zero, one, pi = (E("0"), E("1"), Symbol.PI)
    _numerator = _model.expand_couplings(
        diagram.numerator_expression(in_lmb=True).to_expression()
    )
    for _edge in diagram.external_edges:
        _matches = list(
            diagram.projector_expression().match(
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
            * diagram.overall_factor_expression(evaluate=True)
            * diagram.numerator_prefactor_expression()
        ).expand()
    )
    _adjoint = _operator.dirac_adjoint().expand().simplify_gamma0().to_expression()
    _momentum_index = S("muon_decay::momentum_index_")
    _adjoint = _adjoint.replace(
        _conjugate(P(_momentum_index, _index)), P(_momentum_index, _index)
    )
    for _real in (_charge, _sw, _mw, _mm, _me):
        _adjoint = _adjoint.replace(_conjugate(_real), _real)
    _adjoint = TensorExpression(_adjoint).wrap_indices(_wrapped)
    _projector = (
        _muon.spin_sum(P(0), _ports[0], _wrapped(_ports[3]), average=True)
        * _electron.spin_sum(P(1), _wrapped(_ports[2]), _ports[1])
        * _model.particle_by_pdg(-12).spin_sum(P(2), _ports[2], _wrapped(_ports[1]))
        * _model.particle_by_pdg(14).spin_sum(P(3), _wrapped(_ports[0]), _ports[3])
    )
    _contracted = (
        TensorExpression(
            _operator.to_expression() * _adjoint * _projector,
            cook_indices=CookSettings.indices(),
        )
        .expand()
        .simplify_gamma()
        .expand()
        .simplify_gamma()
        .simplify_epsilon()
        .expand()
        .simplify_metrics()
        .to_dots()
    )
    assert _contracted.is_scalar
    _kin = hep.Kinematics()
    for _i, _mass_squared in enumerate((_mm**2, _me**2, zero, zero)):
        _kin = _kin.with_scalar_product(P(_i), P(_i), _mass_squared)
    for _i, _j, _value in [
        (1, 2, (s - _me**2) / 2),
        (1, 3, (t - _me**2) / 2),
        (2, 3, (_mm**2 + _me**2 - s - t) / 2),
        (0, 1, (s + t) / 2),
        (0, 2, (_mm**2 - t) / 2),
        (0, 3, (_mm**2 - s) / 2),
    ]:
        _kin = _kin.with_scalar_product(P(_i), P(_j), _value)
    _den = S("gammalooprs::denom")
    _edge_, _momentum_, _mass_, _quad_ = S(
        "muon_decay::edge_",
        "muon_decay::momentum_",
        "muon_decay::mass_",
        "muon_decay::quad_",
    )
    _denominator = _kin.apply(
        diagram.denominator_expression(dimension=4, in_lmb=True)
        .to_expression()
        .replace(_den(_edge_, _momentum_, _mass_, _quad_), _quad_)
    )
    assert (_denominator - s + _mw**2).expand() == zero
    squared = (_kin.apply(_contracted.to_expression()) / _denominator**2).together()
    squared = (
        squared.replace(_charge**4, 32 * GF**2 * _mw**4 * _sw**4)
        .replace_multiple([Replacement(_mm, M), Replacement(_me, mass)])
        .together()
    )
    _pk, _pq1, _pq2 = ((s + t) / 2, (M**2 - t) / 2, (M**2 - s) / 2)
    _kq1, _kq2, _q1q2 = (
        (s - mass**2) / 2,
        (t - mass**2) / 2,
        (M**2 + mass**2 - s - t) / 2,
    )
    _reference_full = (
        16
        * GF**2
        / (s - _mw**2) ** 2
        * (
            -2 * mass**2 * _pq2 * _kq1**2
            - 2 * mass**2 * _mw**2 * _kq2 * _pq1
            + 2 * mass**2 * _mw**2 * _kq1 * _pq2
            - 2 * mass**2 * _mw**2 * _pk * _q1q2
            - mass**4 * _kq1 * _pq2
            + 2 * mass**2 * _pk * _kq1 * _kq2
            + 2 * mass**2 * _kq1 * _kq2 * _pq1
            + 2 * mass**2 * _pk * _kq1 * _q1q2
            + 2 * mass**2 * _kq1 * _pq1 * _q1q2
            - 4 * mass**2 * _mw**2 * _pq1 * _q1q2
            + 4 * _mw**4 * _kq2 * _pq1
        )
    )
    assert (squared - _reference_full).together() == zero
    fermi = squared.replace(_mw, 1 / z.sqrt()).series(z, 0, 0).to_expression().expand()
    assert (fermi - 16 * GF**2 * (t - mass**2) * (M**2 - t)).together() == zero
    finite_W = (
        squared.replace(mass, zero)
        .replace(_mw, 1 / z.sqrt())
        .series(z, 0, 1)
        .to_expression()
        .expand()
    )
    assert (finite_W - fermi.replace(mass, zero) * (1 + 2 * s * z)).together() == zero
    mo.vstack(
        [
            diagram,
            mo.accordion(
                {"Full unitary-gauge squared amplitude": mo.vstack([squared])}
            ),
            mo.md("**Derived low-energy squared amplitude**"),
            fermi,
        ]
    )
    return GF, M, P, fermi, finite_W, mass, one, pi, s, t, z, zero


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derive the Dalitz measure
    Factor the decay into $P\to Q+\bar\nu_e$ and $Q\to e+\nu_\mu$, with $Q^2=t$. Integrate the global orientation and the inner azimuth, and include $dt/(2\pi)$. The pair-rest-frame energies determine $s(\cos\theta)$ and its Jacobian.

    On the physical domain $m^2<t<M^2$, the two positive square roots are $M^2-t$ and $t-m^2$. The derived boundaries are $m^2M^2/t\leq s\leq M^2+m^2-t$. All final particles are distinct.
    """)


@app.cell
def _(M, P, S, fermi, hep, mass, mo, one, pi, t, zero):
    _Q = S("muon_decay::Q")
    _outer = (
        hep.Kinematics()
        .with_scalar_product(_Q, _Q, t)
        .with_scalar_product(P(2), P(2), zero)
        .with_scalar_product(_Q, P(2), (M**2 - t) / 2)
    )
    _inner = (
        hep.Kinematics()
        .with_scalar_product(P(1), P(1), mass**2)
        .with_scalar_product(P(3), P(3), zero)
        .with_scalar_product(P(1), P(3), (t - mass**2) / 2)
    )
    _outer_measure = _outer.two_body_phase_space(_Q, P(2))
    _inner_measure = _inner.two_body_phase_space(P(1), P(3))
    _outer_measure = _outer_measure.replace(
        (4 * ((M**2 - t) / 2).expand() ** 2).sqrt(), M**2 - t
    ).together()
    _inner_measure = _inner_measure.replace(
        (4 * ((t - mass**2) / 2).expand() ** 2).sqrt(), t - mass**2
    ).together()
    assert (_outer_measure - (M**2 - t) / (32 * pi**2 * M**2)).together() == zero
    assert (_inner_measure - (t - mass**2) / (32 * pi**2 * t)).together() == zero
    _cosine = S("muon_decay::cosine")
    _electron_energy = (t + mass**2) / (2 * t.sqrt())
    _electron_momentum = (t - mass**2) / (2 * t.sqrt())
    _spectator_energy = (M**2 - t) / (2 * t.sqrt())
    _s_of_cosine = mass**2 + 2 * _spectator_energy * (
        _electron_energy - _electron_momentum * _cosine
    )
    s_min = _s_of_cosine.replace(_cosine, one).together()
    s_max = _s_of_cosine.replace(_cosine, -one).together()
    assert (s_min - mass**2 * M**2 / t).together() == zero
    assert (s_max - M**2 - mass**2 + t).together() == zero
    _jacobian = -_s_of_cosine.derivative(_cosine)
    dalitz_measure = (
        _outer_measure * _inner_measure * (4 * pi) * (2 * pi) / (2 * pi * _jacobian)
    ).together()
    assert (dalitz_measure - 1 / (128 * pi**3 * M**2)).together() == zero
    flux = hep.Kinematics().with_scalar_product(P(0), P(0), M**2).flux(P(0))
    assert flux == 2 * M
    density = (fermi * dalitz_measure / flux).together()
    mo.vstack(
        [
            mo.md("**Outer and inner two-body measures per solid angle**"),
            mo.hstack([_outer_measure, _inner_measure]),
            mo.md("**Derived s bounds**"),
            mo.hstack([s_min, s_max]),
            mo.md("**dΦ₃ / (ds dt)**"),
            dalitz_measure,
            mo.md("**dΓ / (ds dt) in the Fermi limit**"),
            density,
        ]
    )
    return dalitz_measure, density, flux, s_max, s_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exact integrations
    Let $r=m^2/M^2$ and $\Gamma_0=G_F^2M^5/(192\pi^3)$. Integrating the massive Dalitz distribution derives $\Gamma/\Gamma_0=f(r)$, including the logarithmic term.

    For a massless electron, $x_E=2E_e/M$ runs from zero to one. Integrating at fixed $x_E$ derives the normalized Michel spectrum. Expanding the full generated W amplitude also gives its first propagator correction. The finite-mass result uses the Fermi limit; the finite-W result below uses a massless electron.
    """)


@app.cell
def _(
    GF,
    M,
    S,
    dalitz_measure,
    density,
    finite_W,
    flux,
    mass,
    mo,
    one,
    pi,
    s,
    s_max,
    s_min,
    t,
    z,
    zero,
):
    _tau = S("muon_decay::tau", is_positive=True)
    r = S("muon_decay::r", is_positive=True)
    normalization = GF**2 * M**5 / (192 * pi**3)
    _integrand = (
        (density * (s_max - s_min) * M**2 / normalization)
        .replace(t, M**2 * _tau)
        .replace(mass, M * r.sqrt())
        .together()
    )
    assert (
        _integrand - 12 * (1 - _tau) ** 2 * (_tau - r) ** 2 / _tau
    ).together() == zero
    _primitive = _integrand.integrate(_tau)
    mass_correction = (
        _primitive.replace(_tau, one) - _primitive.replace(_tau, r)
    ).expand()
    _reference_correction = 1 - 8 * r + 8 * r**3 - r**4 - 12 * r**2 * r.log()
    assert (mass_correction - _reference_correction).together() == zero
    assert mass_correction.replace(r, one) == zero
    _total_width = normalization * mass_correction
    xE = S("muon_decay::xE", is_positive=True)
    _massless_density = density.replace(mass, zero)
    _dimensionless_density = (
        (_massless_density * M**4 / normalization).replace(t, M**2 * _tau).together()
    )
    _michel_primitive = _dimensionless_density.integrate(_tau)
    michel_spectrum = (
        _michel_primitive.replace(_tau, xE) - _michel_primitive.replace(_tau, zero)
    ).expand()
    assert (michel_spectrum - 2 * xE**2 * (3 - 2 * xE)).together() == zero
    _michel_integral = michel_spectrum.integrate(xE)
    assert (
        _michel_integral.replace(xE, one) - _michel_integral.replace(xE, zero) - 1
    ).together() == zero
    _correction_density = (finite_W.coefficient(z) * dalitz_measure / flux).replace(
        mass, zero
    )
    _correction_primitive = _correction_density.integrate(s)
    _correction_t = (
        _correction_primitive.replace(s, M**2 - t)
        - _correction_primitive.replace(s, zero)
    ).together()
    _correction_primitive = _correction_t.integrate(t)
    W_coefficient = (
        _correction_primitive.replace(t, M**2) - _correction_primitive.replace(t, zero)
    ).together()
    assert (W_coefficient / normalization - 3 * M**2 / 5).together() == zero
    W_relative_coefficient = (W_coefficient / (normalization * M**2)).together()
    mo.vstack(
        [
            mo.md("**Mass correction f(r)**"),
            mass_correction,
            mo.md("**Normalized massless Michel spectrum**"),
            michel_spectrum,
            mo.md("**Massless width through order M²/mW²**"),
            normalization + W_coefficient / S("UFO::MW") ** 2,
        ]
    )
    return W_relative_coefficient, mass_correction, michel_spectrum, r, xE


@app.cell
def _(E, mass_correction, np, r):
    _nodes, _weights = np.polynomial.legendre.leggauss(160)
    numeric_checks = []
    for _rv in [0.0, (0.511 / 105.658) ** 2, 0.01, 0.1, 0.3, 0.7]:
        _value = (
            1.0
            if _rv == 0
            else complex(mass_correction.replace(r, E(str(_rv))).evaluate({}))
        )
        assert abs(complex(_value).imag) < 1e-14
        _actual = complex(_value).real
        _integral_value = sum(
            (
                float(_w)
                * (1 - _rv)
                / 2
                * 12
                * (1 - (_rv + (1 - _rv) * (float(_n) + 1) / 2)) ** 2
                * (_rv + (1 - _rv) * (float(_n) + 1) / 2 - _rv) ** 2
                / (_rv + (1 - _rv) * (float(_n) + 1) / 2)
                for _n, _w in zip(_nodes, _weights, strict=True)
            )
        )
        assert abs(_actual - _integral_value) < 2e-08, (_rv, _actual, _integral_value)
        assert 0 < _actual <= 1
        numeric_checks.append((_rv, _actual, _integral_value))
    return (numeric_checks,)


@app.cell
def _(mo):
    daughter_mass = mo.ui.dropdown(
        {
            "Massless electron": 0,
            "Illustrative electron/muon ratio": 1,
            "r=0.01": 2,
            "r=0.1": 3,
            "r=0.3": 4,
            "r=0.7": 5,
        },
        value="Illustrative electron/muon ratio",
        label="Daughter mass ratio r",
    )
    W_scale = mo.ui.dropdown(
        {"Fermi limit": 0.0, "M²/mW²=0.000002": 2e-06, "M²/mW²=0.01": 0.01},
        value="Fermi limit",
        label="Finite-W expansion",
    )
    mo.vstack([daughter_mass, W_scale])
    return W_scale, daughter_mass


@app.cell
def _(
    E,
    W_relative_coefficient,
    W_scale,
    daughter_mass,
    michel_spectrum,
    mo,
    np,
    numeric_checks,
    xE,
):
    selected_r, selected_correction, quadrature = numeric_checks[daughter_mass.value]
    assert abs(selected_correction - quadrature) < 2e-08
    relative_W_width = (
        1 + complex(W_relative_coefficient.evaluate({})).real * W_scale.value
    )
    _spectrum_samples = [
        {
            "xE": float(_v),
            "(dΓ/dxE)/Γ₀": complex(
                michel_spectrum.replace(xE, E(str(_v))).evaluate({})
            ).real,
        }
        for _v in np.linspace(0, 1, 11)
    ]
    mo.vstack(
        [
            mo.md("## Mass-corrected Fermi width"),
            mo.md(f"`r = {selected_r:.9g}` · **Γ/Γ₀ = {selected_correction:.12g}**"),
            mo.md(
                f"Independent Dalitz quadrature: `{quadrature:.12g}`; difference `{abs(selected_correction - quadrature):.3g}`."
            ),
            mo.md("## Finite-W correction, massless electron"),
            mo.md(f"**Γ/Γ₀ = {relative_W_width:.12g}** through first order in M²/mW²."),
            mo.md("## Massless Michel spectrum samples"),
            mo.ui.table(_spectrum_samples, selection=None),
            mo.md(
                "The spectrum integrates to one. Both mass and W corrections above are normalized to Γ₀; they are evaluated in their stated limits."
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
