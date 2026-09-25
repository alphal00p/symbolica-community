import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Bhabha and Møller scattering")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, Expression, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep as fk
    from symbolica.community.spenso import CookSettings, TensorExpression

    _set_namespace("lepton_scattering")

    return CookSettings, E, Expression, S, TensorExpression, fk, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bhabha and Møller scattering

    [Browse all notebooks](/) · [QED production](/?file=hep/qed_cross_section.py) ·
    [Diphoton annihilation](/?file=hep/diphoton.py)

    Generate the two tree diagrams for either **Bhabha scattering**,
    $e^-e^+\to e^-e^+$, or **Møller scattering**, $e^-e^-\to e^-e^-$.
    The calculation retains the electron mass and the interference between
    channels. It averages both incoming spins and sums both outgoing spins,
    with $s+t+u=4m_e^2$.

    The default generated amplitudes retain the labeled external particles
    and their relative fermion signs. In particular, Møller scattering
    requires both exchange diagrams before squaring the amplitude.

    Reference calculations: [FeynCalc Bhabha scattering](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-ElAel)
    and [FeynCalc Møller scattering](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElEl-ElEl).
    """)
    return


@app.cell
def _(fk, mo):
    lepton_model = fk.Model.standard_model()
    reaction = mo.ui.dropdown(
        ["Bhabha", "Møller"], value="Bhabha", label="Scattering process"
    )
    mo.vstack([reaction])
    return lepton_model, reaction


@app.cell
def _(
    CookSettings,
    E,
    S,
    TensorExpression,
    fk,
    lepton_model,
    mo,
    reaction,
):
    P = S("gammalooprs::P")
    s = S("s", is_positive=True)
    t, u, mass, charge = S("t", "u", "UFO::Me", "UFO::ee")
    _names = ["e-", "e+", "e-", "e+"] if reaction.value == "Bhabha" else ["e-"] * 4
    _kinematics = fk.Kinematics.mandelstam(
        [P(_position) for _position in range(4)], [mass**2] * 4, [s, t, u]
    )
    lepton_generated = lepton_model.process(
        _names[:2], _names[2:], vertex_allow=["V_98"]
    ).generate_diagrams(
        loops=0,
        max_vertices=2,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(lepton_generated.diagrams) == 2
    _ports = S(
        "external_0",
        "external_1",
        "external_2",
        "external_3",
    )
    _a, _b, _c, _inverse, _wave, _rep, _index = S(
        "a_", "b_", "c_", "inverse_", "wave_", "rep_", "index_"
    )
    _conjugate, _wrapped = S("spenso::conj", "adjoint_index")
    _operators, _denominators = [], []
    for _diagram in lepton_generated.diagrams:
        _numerator = lepton_model.expand_couplings(
            _diagram.numerator_expression(in_lmb=True).to_expression()
        )
        # Align ports through the generated wavefunctions, independent of each
        # channel's half-edge numbering.
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
        _denominator = _kinematics.apply(
            _diagram.denominator_expression(dimension=4, in_lmb=True)
            .to_expression()
            .replace(S("gammalooprs::denom")(_a, _b, _c, _inverse), _inverse)
        ).expand()
        _denominators.append(_denominator)
        _operator = TensorExpression(
            (
                _numerator
                * _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
                / _denominator
            ).expand()
        )
        _operators.append(_operator.to_expression())
    assert set(_denominators) == ({s, t} if reaction.value == "Bhabha" else {t, u})

    _spins = E("1")
    for _position, _name in enumerate(_names):
        _particle = lepton_model.particle(_name)
        _column = (_position < 2) == (not _particle.is_antiparticle)
        _left, _right = (
            (_ports[_position], _wrapped(_ports[_position]))
            if _column
            else (_wrapped(_ports[_position]), _ports[_position])
        )
        _spins *= _particle.spin_sum(P(_position), _left, _right, average=_position < 2)
    _squares = []
    # Adjoint the full amplitude in the first pass; isolate the channel squares
    # in the second pass so their difference exposes the interference.
    for _terms in ([sum(_operators, E("0"))], _operators):
        _raw = E("0")
        for _term in _terms:
            # Preserve physical external-leg labels under adjunction so different
            # channels share the same completeness tensors in interference terms.
            _adjoint = (
                TensorExpression(_term)
                .dirac_adjoint(preserve_indices=True)
                .to_expression()
            )
            for _real in (s, t, u, mass, charge):
                _adjoint = _adjoint.replace(_conjugate(_real), _real)
            _raw += _term * TensorExpression(_adjoint).wrap_indices(_wrapped)
        _scalar = (
            TensorExpression(_raw * _spins, cook_indices=CookSettings.indices())
            .expand()
            .simplify_gamma()
            .expand()
            .simplify_gamma()
            .simplify_metrics()
            .to_dots()
        )
        assert _scalar.is_scalar
        _squares.append(
            _kinematics.apply(_scalar.to_expression())
            .replace(u, 4 * mass**2 - s - t)
            .together()
        )
    lepton_squared = _squares[0]
    lepton_interference = (lepton_squared - _squares[1]).together()
    _massless_interference = (
        4 * charge**4 * u**2 / (s * t)
        if reaction.value == "Bhabha"
        else 4 * charge**4 * s**2 / (t * u)
    )
    assert (
        lepton_interference.replace(mass, E("0"))
        - _massless_interference.replace(u, -s - t)
    ).together() == E("0")
    if reaction.value == "Møller":
        assert (
            lepton_squared - lepton_squared.replace(t, 4 * mass**2 - s - t)
        ).together() == E("0")
    mo.vstack(
        [
            mo.hstack(lepton_generated.diagrams),
            mo.md("**Massive, spin-averaged squared amplitude**"),
            lepton_squared,
            mo.md("**Interference between the two channels**"),
            lepton_interference,
        ]
    )
    return (
        P,
        charge,
        lepton_generated,
        lepton_interference,
        lepton_squared,
        mass,
        s,
        t,
        u,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Angular distribution

    In the center-of-mass frame let $x=\cos\theta$ and
    $\beta=\sqrt{1-4m_e^2/s}$. Then
    $t=-s\beta^2(1-x)/2$ and $u=-s\beta^2(1+x)/2$.
    The shared flux and phase-space methods give
    $d\Phi_2/(d\Omega\,\mathrm{flux})=1/(64\pi^2s)$.

    For Bhabha scattering, $\theta$ labels the outgoing electron.
    For Møller scattering either outgoing electron can carry that label:
    the displayed **event density includes $1/2!$ on the full sphere**.
    Equivalently, integrate the labeled density over one hemisphere.
    This factor belongs to phase-space counting; both interfering exchange
    amplitudes remain in the squared matrix element.
    """)
    return


@app.cell
def _(
    E,
    Expression,
    P,
    S,
    charge,
    fk,
    lepton_interference,
    lepton_squared,
    mass,
    mo,
    reaction,
    s,
    t,
    u,
):
    beta = S("beta", is_positive=True)
    cos_theta, alpha = S("cos_theta", "alpha")
    _mass_squared = s * (1 - beta**2) / 4
    _kinematics = fk.Kinematics.mandelstam(
        [P(_position) for _position in range(4)], [_mass_squared] * 4, [s, t, u]
    )
    # Choose the physical branch s > 0, beta > 0 after the shared calculation.
    _flux = (
        _kinematics.flux(P(0), P(1)).expand().replace((s**2 * beta**2).sqrt(), s * beta)
    )
    _measure = (
        _kinematics.two_body_phase_space(P(2), P(3))
        .expand()
        .replace((s**2 * beta**2).sqrt(), s * beta)
    )
    assert (_measure / _flux - 1 / (64 * Expression.PI**2 * s)).together() == E("0")
    _counting = E("1") if reaction.value == "Bhabha" else E("1/2")
    _densities = []
    for _square in (lepton_squared, lepton_interference):
        _densities.append(
            (
                _square.replace(t, -s * beta**2 * (1 - cos_theta) / 2)
                .replace(mass, _mass_squared.sqrt())
                .replace(charge**4, (4 * Expression.PI * alpha) ** 2)
                * _measure
                / _flux
                * _counting
            ).together()
        )
    angular_density, angular_interference = _densities
    normalized_density = (angular_density * s / alpha**2).together()
    massless_kernel = (
        (3 + cos_theta**2) ** 2 / (4 * (1 - cos_theta) ** 2)
        if reaction.value == "Bhabha"
        else (3 + cos_theta**2) ** 2 / (2 * (1 - cos_theta**2) ** 2)
    )
    assert (normalized_density.replace(beta, E("1")) - massless_kernel).together() == E(
        "0"
    )
    mo.vstack(
        [
            mo.md(r"**Event angular density $d\sigma/d\Omega$**"),
            angular_density,
            mo.md(r"**Massless limit, in units of $\alpha^2/s$**"),
            massless_kernel,
        ]
    )
    return (
        alpha,
        angular_density,
        angular_interference,
        beta,
        cos_theta,
        massless_kernel,
        normalized_density,
    )


@app.cell
def _(mo):
    cm_speed = mo.ui.slider(
        0.1, 1.0, step=0.05, value=1.0, label="CM speed beta (1 = massless limit)"
    )
    scattering_cosine = mo.ui.slider(
        -0.95, 0.95, step=0.05, value=0.0, label="Scattering angle cos(theta)"
    )
    mo.vstack([cm_speed, scattering_cosine])
    return cm_speed, scattering_cosine


@app.cell
def _(
    E,
    alpha,
    angular_interference,
    beta,
    cm_speed,
    cos_theta,
    mo,
    normalized_density,
    s,
    scattering_cosine,
):
    _interference = (angular_interference * s / alpha**2).together()
    angular_value, interference_value = (
        _quantity.replace(beta, E(str(cm_speed.value)))
        .replace(cos_theta, E(str(scattering_cosine.value)))
        .to_float(16)
        for _quantity in (normalized_density, _interference)
    )
    assert float(angular_value) > 0
    mo.md(
        rf"Event density: **{angular_value}** × $\alpha^2/s$. "
        rf"The interference contributes **{interference_value}** × $\alpha^2/s$."
    )
    return angular_value, interference_value


if __name__ == "__main__":
    app.run()
