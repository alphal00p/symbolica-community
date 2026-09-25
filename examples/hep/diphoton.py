import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Electron–positron annihilation into photons",
)


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep as fk

    _set_namespace("diphoton")
    return E, S, Symbol, fk, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Electron–positron annihilation into two photons

    [Browse all notebooks](/) · [Sewn diphoton calculation](/?file=hep/sewn_diphoton.py) ·
    [Compton scattering](/?file=hep/compton.py)

    Generate both diagrams for $e^-e^+\to\gamma\gamma$, retaining the electron
    mass and their interference. Average the two incoming spins and sum the
    photon polarizations using shared Feynkit, Spenso and Idenso operations.
    The result agrees with the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-GaGa).
    The kinematics obey $s+t+u=2m_e^2$.
    """)
    return


@app.cell
def _(fk):
    model = fk.Model.standard_model()
    return (model,)


@app.cell
def _(E, S, fk, mo, model):
    P = S("gammalooprs::P")
    s = S("s", is_positive=True)
    t, u, mass, charge = S("t", "u", "UFO::Me", "UFO::ee")
    kin = fk.Kinematics.mandelstam(
        [P(0), P(1), P(2), P(3)], [mass**2, mass**2, E("0"), E("0")], [s, t, u]
    )

    process = fk.Process(model, ["e-", "e+"], ["a", "a"])
    amplitude = process.generate_amplitude(max_vertices=2, vertex_allow=["V_98"])
    assert len(amplitude.diagrams) == 2
    assert amplitude.expression().rank == 4
    spin_summed = amplitude.squared().sum_spins([0, 1], average_initial=True)

    mo.vstack([mo.hstack(amplitude.diagrams), amplitude])
    return P, charge, kin, mass, s, spin_summed, t, u


@app.cell
def _(fk, mo, model):
    generated2 = fk.Process(model, ["e-", "e+"], ["a", "a"]).generate_diagrams(
        maximum_bridges=None
    )
    mo.vstack([mo.hstack(generated2.diagrams)])
    return


@app.cell
def _(mo):
    gauge = mo.ui.dropdown(
        ["Covariant", "Other photon", "Incoming electron"],
        value="Covariant",
        label="Photon polarization reference",
    )
    mo.vstack([gauge])
    return (gauge,)


@app.cell
def _(E, P, charge, gauge, kin, mass, mo, s, spin_summed, t, u):
    _references = {
        "Covariant": (None, None),
        "Other photon": (P(3), P(2)),
        "Incoming electron": (P(0), P(0)),
    }[gauge.value]
    _scalar = (
        spin_summed.sum_spins(
            [2, 3],
            references={
                position: reference
                for position, reference in zip((2, 3), _references)
                if reference is not None
            },
        )
        .expression()
        .expand()
        .simplify_gamma()
        .expand()
        .simplify_gamma()
        .simplify_metrics()
        .to_dots()
    )
    assert _scalar.is_scalar
    diphoton_squared = (
        kin.apply(_scalar.to_expression()).replace(s, 2 * mass**2 - t - u).together()
    )
    _x, _y = t - mass**2, u - mass**2
    _expected = (
        2
        * charge**4
        * (
            _x / _y
            + _y / _x
            + 4 * mass**2 * s / (_x * _y)
            - 4 * mass**4 * s**2 / (_x**2 * _y**2)
        )
    )
    assert (
        diphoton_squared - _expected.replace(s, 2 * mass**2 - t - u)
    ).together() == E("0")
    mo.vstack(
        [
            mo.md(
                "**Massive, spin-averaged squared matrix element** — all three polarization choices give the same result."
            ),
            diphoton_squared,
        ]
    )
    return (diphoton_squared,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identical photons and angular acceptance

    Let $c=\cos\theta$ and accept events with $|c|<C<1$. The labeled massless
    angular density is $\alpha^2(1+c^2)/[s(1-c^2)]$. Integrating it over the
    full sphere counts both photon assignments; the event rate includes $1/2!$.
    Equivalently, select the forward photon and integrate over $0<c<C$.

    `Kinematics.flux` and `two_body_phase_space` supply the flux and measure.
    Symbolica integrates the rational angular distribution. The massless result,
    $\sigma=2\pi\alpha^2[2\operatorname{atanh}C-C]/s$, also matches
    [DELPHI's Born formula, Eq. (2)](https://arxiv.org/pdf/hep-ex/0409058).
    For a massive electron, $\beta=\sqrt{1-4m_e^2/s}$ is the incoming CM speed.
    """)
    return


@app.cell
def _(E, P, S, Symbol, charge, diphoton_squared, fk, kin, mass, mo, s, t, u):
    massless = diphoton_squared.replace(mass, E("0"))
    assert (massless - 2 * charge**4 * (t / u + u / t)).together() == E("0")
    cos_theta, alpha, cutoff = S("cos_theta", "alpha", "cutoff")
    kin_massless = fk.Kinematics.mandelstam(
        [P(0), P(1), P(2), P(3)], [E("0")] * 4, [s, t, u]
    )
    angular = (
        massless.replace(t, -s * (1 - cos_theta) / 2)
        .replace(u, -s * (1 + cos_theta) / 2)
        .replace(charge**4, (4 * Symbol.PI * alpha) ** 2)
    )
    labeled = (
        angular
        * kin_massless.two_body_phase_space(P(2), P(3))
        / kin_massless.flux(P(0), P(1))
    ).together()
    assert (
        labeled - alpha**2 / s * (1 + cos_theta**2) / (1 - cos_theta**2)
    ).together() == E("0")
    # The phase-space API excludes identical-particle factors. Count each event
    # once by selecting its forward photon, or equivalently use 1/2! on the full sphere.
    primitive = (labeled * s / alpha**2).integrate(cos_theta).together()
    assert (primitive.derivative(cos_theta) - labeled * s / alpha**2).together() == E(
        "0"
    )
    assert primitive.replace(cos_theta, E("0")) == E("0")
    event_cross_section = (
        2 * Symbol.PI * alpha**2 / s * primitive.replace(cos_theta, cutoff)
    ).together()
    assert (
        event_cross_section
        - 2 * Symbol.PI * alpha**2 / s * (2 * cutoff.atanh() - cutoff)
    ).together() == E("0")
    # Retain the electron mass: beta is the incoming CM speed, 0 < beta < 1.
    beta = S("beta", is_positive=True)
    cm_squared = (
        diphoton_squared.replace(t, mass**2 - s * (1 - beta * cos_theta) / 2)
        .replace(u, mass**2 - s * (1 + beta * cos_theta) / 2)
        .replace(mass, (s * (1 - beta**2) / 4).sqrt())
    ).together()
    kernel = (cm_squared / (4 * charge**4)).together()
    expected_kernel = (1 + beta**2 * cos_theta**2) / (
        1 - beta**2 * cos_theta**2
    ) + 2 * beta**2 * (1 - beta**2) * (1 - cos_theta**2) / (
        1 - beta**2 * cos_theta**2
    ) ** 2
    assert (kernel - expected_kernel).together() == E("0")
    assert kernel.replace(beta, E("0")).together() == E("1")
    massive_primitive = kernel.integrate(cos_theta).together()
    assert (massive_primitive.derivative(cos_theta) - kernel).together() == E("0")
    assert massive_primitive.replace(cos_theta, E("0")).together() == E("0")
    expected_primitive = (
        (3 - beta**4) / beta * (beta * cos_theta).atanh()
        - cos_theta
        - (1 - beta**2) ** 2 * cos_theta / (1 - beta**2 * cos_theta**2)
    )
    assert (massive_primitive - expected_primitive).together() == E("0")
    # The massive flux contributes 1/beta; the final photons have unit speed.
    # Select its physical positive branch, s > 0 and beta > 0. The shared
    # phase-space measure excludes the identical-particle factor.
    massive_flux = (
        kin.flux(P(0), P(1)).replace(mass, (s * (1 - beta**2) / 4).sqrt()).expand()
    )
    assert (massive_flux**2 - 4 * s**2 * beta**2).together() == E("0")
    massive_flux = massive_flux.replace((s**2 * beta**2).sqrt(), s * beta)
    massive_density_factor = (
        (4 * charge**4 * kin.two_body_phase_space(P(2), P(3)) / massive_flux)
        .replace(charge**4, (4 * Symbol.PI * alpha) ** 2)
        .together()
    )
    assert (massive_density_factor - alpha**2 / (s * beta)).together() == E("0")
    massive_cross_section = (
        2
        * Symbol.PI
        * massive_density_factor
        * massive_primitive.replace(cos_theta, cutoff)
    )
    assert (
        massive_cross_section.replace(beta, E("1")) - event_cross_section
    ).together() == E("0")

    mo.vstack(
        [
            mo.md("**Massless labeled angular density**"),
            labeled,
            mo.md("**Massless event cross section with angular cut**"),
            event_cross_section,
            mo.md("**Massive event cross section with angular cut**"),
            massive_cross_section,
        ]
    )
    return alpha, beta, cutoff, massive_cross_section


@app.cell
def _(mo):
    angular_cut = mo.ui.slider(0.1, 0.95, step=0.05, value=0.8, label="Acceptance C")
    electron_speed = mo.ui.slider(
        0.05, 1.0, step=0.05, value=1.0, label="Incoming speed beta"
    )
    mo.hstack([angular_cut, electron_speed])
    return angular_cut, electron_speed


@app.cell
def _(
    E,
    Symbol,
    alpha,
    angular_cut,
    beta,
    cutoff,
    electron_speed,
    massive_cross_section,
    mo,
    s,
):
    _ratio = (massive_cross_section * s / (2 * Symbol.PI * alpha**2)).together()
    accepted_rate = (
        _ratio.replace(cutoff, E(str(angular_cut.value)))
        .replace(beta, E(str(electron_speed.value)))
        .to_float(16)
    )
    mo.md(rf"Accepted event rate: **{accepted_rate}** × $2\pi\alpha^2/s$.")
    return


if __name__ == "__main__":
    app.run()
