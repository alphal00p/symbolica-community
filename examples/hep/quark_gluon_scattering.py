import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive quark–gluon scattering")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolica import E, S, Symbol
    from symbolica.community import hep
    from symbolica.community.spenso import CookSettings, TensorExpression

    return CookSettings, E, S, Symbol, TensorExpression, hep, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massive quark–gluon scattering

    [Browse notebooks](/) · [Light-cone soft radiation](/?file=hep/soft_function.py) · [Quark annihilation to gluons](/?file=hep/qcd_gluons.py) ·
    [Gluon scattering](/?file=hep/gluon_scattering.py)

    Generate the three $qg\to qg$ diagrams and every interference term, retaining
    the quark mass and symbolic SU($N_c$) color algebra. The model's bottom quark
    supplies the generic mass; particles and vertices are selected by name.

    Compare two independent routes from the FeynCalc examples:
    [physical gluon polarizations](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QGl-QGl)
    and [covariant sums with ghost subtraction](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QGl-QGl-2).
    Shared particle spin/color sums, Spenso and Idenso perform the contractions.
    Both null and timelike gauge references give the same massive result.

    The initial symbolic calculation runs once. The controls reuse its results.
    """)
    return


@app.cell
def _(CookSettings, E, S, TensorExpression, hep):
    def calculate_scattering():
        model = hep.Model.standard_model()
        quark = model.particle("b")
        allowed = [
            v
            for v in model.vertex_rules
            if sorted(v.particles)
            in [
                sorted(names)
                for names in [("b", "b~", "g"), ("g", "g", "g"), ("ghG", "ghG~", "g")]
            ]
        ]
        assert len(allowed) == 3
        P = S("gammalooprs::P")
        s = S("s", is_positive=True)
        t, u, mass, gs = S("t", "u", "UFO::MB", "UFO::G")
        Nc, dA, cof, coad = S("spenso::Nc", "dA", "spenso::cof", "spenso::coad")
        kinematics = hep.Kinematics.mandelstam(
            [P(0), P(1), P(2), P(3)], [mass**2, E("0"), mass**2, E("0")], [s, t, u]
        )
        results = {}
        generated_channels = {}
        for boson, count in (("ghG", 1), ("ghG~", 1), ("g", 3)):
            generated = model.generate_diagrams(
                ["b", boson],
                ["b", boson],
                max_vertices=2,
                maximum_bridges=None,
                vertex_allow=allowed,
                numerator_grouping=None,
                progress=None,
            )
            assert len(generated.diagrams) == count
            generated_channels[boson] = generated
            ports = S("external_0", "external_1", "external_2", "external_3")
            a, b, c, inverse, index = S("a_", "b_", "c_", "inverse_", "index_")
            conjugate, adjoint_index = S("spenso::conj", "adjoint_index")
            amplitude = E("0")
            denominators = []
            for diagram in generated.diagrams:
                numerator = model.expand_couplings(
                    diagram.numerator_expression(in_lmb=True).to_expression()
                )
                # Align external spin and color ports using the graph's native half-edge IDs.
                # Ghosts have no polarization wavefunctions; these IDs cover them as well.
                for half in diagram.half_edges:
                    edge = half.edge.data
                    if edge.is_external:
                        numerator = numerator.replace(
                            S("gammalooprs::hedge")(half.data, 1),
                            ports[edge.external_index],
                        )
                denominator = kinematics.apply(
                    diagram.denominator_expression(dimension=4, in_lmb=True)
                    .to_expression()
                    .replace(S("gammalooprs::denom")(a, b, c, inverse), inverse)
                ).expand()
                denominators.append(denominator)
                amplitude += (
                    numerator
                    * diagram.overall_factor_expression(evaluate=True)
                    * diagram.numerator_prefactor_expression()
                    / denominator
                )
            assert all(
                d in denominators
                for d in ((t, s - mass**2, u - mass**2) if count == 3 else (t,))
            )
            operator = TensorExpression(amplitude.expand())
            assert len(operator.interface) == (8 if count == 3 else 6)
            adjoint = (
                operator.dirac_adjoint().expand().simplify_gamma0().to_expression()
            )
            adjoint = adjoint.replace(conjugate(P(a, b)), P(a, b))
            for real in (mass, gs, s, t, u):
                adjoint = adjoint.replace(conjugate(real), real)
            adjoint = TensorExpression(adjoint).wrap_indices(adjoint_index)

            # Match the shared completeness tensor against the dual of each external slot.
            # Matching selects color slots and determines the quark/antiquark orientation.
            left, right, metric = S("left_", "right_", "spenso::g")
            color_projector = E("1")
            initial_colors = 1
            for position, name in enumerate(("b", boson, "b", boson)):
                particle = model.particle(name)
                if position < 2:
                    initial_colors *= abs(particle.color)
                matches = []
                for slot in operator.interface:
                    original = slot.to_expression()
                    if original.replace(ports[position], E("0")) == original:
                        continue
                    closure = metric(
                        slot.dual().to_expression(),
                        original.replace(
                            ports[position], adjoint_index(ports[position])
                        ),
                    )
                    match = next(
                        closure.match(particle.color_sum(left, right), max_level=0),
                        None,
                    )
                    if match is not None:
                        matches.append(match)
                assert len(matches) == 1
                indices = dict(matches[0])
                color_projector *= particle.color_sum(
                    indices[left], indices[right], average=position < 2
                )

            # Spenso dimensions are integers or symbols. Keep dA symbolic until color
            # contraction, then impose the SU(N) relation and convert scalar Casimirs.
            generic = (
                (
                    operator.to_expression()
                    * adjoint
                    * color_projector
                    * initial_colors
                    / (Nc * dA)
                )
                .replace(cof(3, index), cof(Nc, index))
                .replace(coad(8, index), coad(dA, index))
            )
            print("color", boson, flush=True)
            colored = (
                TensorExpression(generic, cook_indices=CookSettings.indices())
                .simplify_color()
                .to_expression()
                .replace(dA, Nc**2 - 1)
            )
            colored = TensorExpression(colored).to_cof_dimension_invariants()

            # Dirac adjunction exchanges the two endpoints of the open fermion chain.
            # Reduce the fermion trace before introducing the physical gluon projectors.
            spin_projector = quark.spin_sum(
                P(0), ports[0], adjoint_index(ports[2]), average=True
            ) * quark.spin_sum(P(2), adjoint_index(ports[0]), ports[2])
            print("spin", boson, flush=True)
            spin_summed = (
                TensorExpression(
                    colored.to_expression() * spin_projector,
                    cook_indices=CookSettings.indices(),
                )
                .expand()
                .simplify_gamma()
                .expand()
                .simplify_gamma()
            )
            reduced = kinematics.apply(
                spin_summed.expand().simplify_metrics().to_dots()
            )
            pieces = [
                (structure, coefficient.together())
                for structure, coefficient in reduced.expand_mink()
            ]
            print("Lorentz structures", boson, len(pieces), flush=True)
            for mode in ("covariant", "null", "timelike") if count == 3 else ("ghost",):
                polarizations = E("1")
                if count == 3:
                    for position in (1, 3):
                        polarizations *= model.particle("g").spin_sum(
                            P(position),
                            ports[position],
                            adjoint_index(ports[position]),
                            reference=None
                            if mode == "covariant"
                            else P(4 - position)
                            if mode == "null"
                            else P(0),
                            covariant=mode == "covariant",
                        )
                # Keep scalar coefficients factored while contracting the Lorentz parts.
                # This avoids repeatedly importing expanded scalar polynomials as tensors.
                contracted = []
                for structure, coefficient in pieces:
                    scalar = (
                        TensorExpression(
                            (structure * polarizations).expand(),
                            cook_indices=CookSettings.indices(),
                        )
                        .simplify_metrics()
                        .to_dots()
                    )
                    assert scalar.is_scalar
                    contracted.append(
                        coefficient * kinematics.apply(scalar.to_expression())
                    )
                squared = (
                    sum(contracted, E("0")).replace(t, 2 * mass**2 - s - u).together()
                )
                results[boson, mode] = squared
                print("Finished", boson, mode, flush=True)

        expected = (
            gs**4
            * (
                -(mass**4) * (3 * s**2 + 14 * s * u + 3 * u**2)
                + mass**2 * (s**3 + 7 * s**2 * u + 7 * s * u**2 + u**3)
                + 6 * mass**8
                - s * u * (s**2 + u**2)
            )
            * (
                -2 * Nc**2 * mass**2 * (s + u)
                + 2 * Nc**2 * mass**4
                + Nc**2 * s**2
                + Nc**2 * u**2
                - t**2
            )
            / (2 * Nc**2 * t**2 * (u - mass**2) ** 2 * (s - mass**2) ** 2)
        )
        ghost_reference = gs**4 * (mass**2 - u) * (s - mass**2) / (2 * t**2)
        for boson in ("ghG", "ghG~"):
            delta = (
                results[boson, "ghost"]
                - ghost_reference.replace(t, 2 * mass**2 - s - u)
            ).together()
            assert delta == 0
        # Average the two incoming gluon states after subtracting unphysical modes.
        physical = (
            (
                results["g", "covariant"]
                - results["ghG", "ghost"]
                - results["ghG~", "ghost"]
            )
            / 2
        ).together()
        assert results["ghG", "ghost"] == results["ghG~", "ghost"]
        assert (results["g", "covariant"] / 2 - physical).together() != 0
        for mode in ("null", "timelike"):
            delta = (
                results["g", mode] / 2 - expected.replace(t, 2 * mass**2 - s - u)
            ).together()
            assert delta == 0
        assert (physical - expected.replace(t, 2 * mass**2 - s - u)).together() == 0
        massless = physical.replace(mass, 0).replace(Nc, 3)
        reference3 = gs**4 * ((s**2 + u**2) / t**2 - E("4/9") * (s**2 + u**2) / (s * u))
        assert (massless - reference3.replace(t, -s - u)).together() == 0
        print(
            "Massive SU(N) quark-gluon scattering, physical gauges, ghost subtraction and massless reference passed",
            flush=True,
        )

        return (
            results,
            generated_channels,
            physical,
            expected,
            kinematics,
            P,
            s,
            t,
            u,
            mass,
            gs,
            Nc,
        )

    (
        results,
        generated_channels,
        physical,
        _expected,
        kinematics,
        P,
        s,
        t,
        u,
        mass,
        gs,
        Nc,
    ) = calculate_scattering()
    return (
        Nc,
        P,
        generated_channels,
        gs,
        kinematics,
        mass,
        physical,
        results,
        s,
        t,
        u,
    )


@app.cell
def _(generated_channels, mo):
    mo.vstack(
        [
            mo.md(r"**Quark–gluon scattering: three diagrams**"),
            mo.hstack(generated_channels["g"].diagrams, justify="space-around"),
            mo.md(r"**Ghost and antighost channels: one diagram each**"),
            mo.hstack(
                [
                    *generated_channels["ghG"].diagrams,
                    *generated_channels["ghG~"].diagrams,
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Initial averages and the ghost subtraction**

    Each result includes the incoming quark spin average and the initial color
    factor $1/[N_c(N_c^2-1)]$. The covariant calculation includes unphysical
    gluon modes. Subtract both generated ghost channels before averaging the
    two physical incoming gluon polarizations:
    $$\overline{|\mathcal M|^2}=\tfrac12
    (|\mathcal M_{\rm cov}|^2-|\mathcal M_{\rm gh}|^2-|\mathcal M_{\rm anti\,gh}|^2).$$
    Each ghost contribution before that last factor is
    $g_s^4(m^2-u)(s-m^2)/(2t^2)$. It is nonzero; a covariant sum alone does not
    reproduce the physical rate.

    At $m=0$, $N_c=3$, the physical result is
    $$g_s^4(s^2+u^2)\left(\frac1{t^2}-\frac4{9su}\right).$$
    """)
    return


@app.cell
def _(Nc, P, S, Symbol, gs, kinematics, mass, np, physical, s, t, u):
    def integrate_rates():
        # Restore u so the CM substitution uses t as the angular variable.
        physical_st = physical.replace(u, 2 * mass**2 - s - t).together()
        phase = kinematics.two_body_phase_space(P(2), P(3)) / kinematics.flux(
            P(0), P(1)
        )
        assert (phase - 1 / (64 * Symbol.PI**2 * s)).together() == 0
        z, rho, cutoff, alpha = S("qg::z", "qg::rho", "qg::cutoff", "qg::alpha_s")
        angle_t = -((s - mass**2) ** 2) * (1 - z) / (2 * s)
        density = (
            (physical_st.replace(t, angle_t) * phase * 2 * Symbol.PI * s / alpha**2)
            .replace(gs**4, (4 * Symbol.PI * alpha) ** 2)
            .replace(mass, (rho * s).sqrt())
            .together()
        )
        assert density.derivative(s).together() == 0
        primitive = density.integrate(z)
        assert (primitive.derivative(z) - density).together() == 0
        cut_rate = primitive.replace(z, cutoff) - primitive.replace(z, -cutoff)
        assert cut_rate.replace(cutoff, 0).together() == 0
        assert (
            cut_rate.derivative(cutoff)
            - density.replace(z, cutoff)
            - density.replace(z, -cutoff)
        ).together() == 0
        rate_checks = []
        nodes, weights = np.polynomial.legendre.leggauss(128)
        for nc in (2, 3, 5):
            for fraction in (0.0, 0.25, 2 / 3):
                for cut in (0.2, 0.5, 0.8):
                    exact = cut_rate.evaluate({Nc: nc, rho: fraction, cutoff: cut})
                    quadrature = cut * sum(
                        w * density.evaluate({Nc: nc, rho: fraction, z: cut * x})
                        for x, w in zip(nodes, weights, strict=True)
                    )
                    assert exact.real > 0 and abs(exact.imag) < 1e-10, (
                        nc,
                        fraction,
                        cut,
                        exact,
                    )
                    assert abs(exact - quadrature) < 2e-11 * max(1, abs(exact)), (
                        nc,
                        fraction,
                        cut,
                        exact,
                        quadrature,
                    )
                    rate_checks.append((nc, fraction, cut, abs(exact - quadrature)))
        print("All 27 cut rates passed", flush=True)

        return density, cut_rate, z, rho, cutoff, rate_checks, angle_t

    density, cut_rate, z, rho, cutoff, rate_checks, angle_t = integrate_rates()
    return angle_t, cut_rate, cutoff, density, rate_checks, rho, z


@app.cell(hide_code=True)
def _(mo, rate_checks):
    assert len(rate_checks) == 27
    mo.md(r"""
    **Massive angular-cut rates**

    Write $\rho=m^2/s$, with $0\leq\rho<1$, and $z=\cos\theta$ for the outgoing
    quark angle. Then $t=-s(1-\rho)^2(1-z)/2$ and $u=2m^2-s-t$.
    Native flux and two-body phase space give
    $$\frac{d\sigma}{dz}=\frac{\overline{|\mathcal M|^2}}{32\pi s}.$$
    The quark and gluon are distinct final particles. The cut $|z|<c<1$ excludes
    the forward pole and the additional backward pole in the massless limit.

    Symbolica integrates the angular density exactly. Differentiation, cut
    boundary checks, and 27 Gaussian quadratures validate the rates below.
    Displayed rates use $s\sigma/\alpha_s^2$, where $g_s^2=4\pi\alpha_s$.
    """)
    return


@app.cell
def _(mo):
    colors = mo.ui.dropdown(
        {"SU(2)": 2, "SU(3)": 3, "SU(5)": 5}, value="SU(3)", label="Color group"
    )
    mass_ratio = mo.ui.dropdown(
        {"Massless": "0", "m²/s = 1/4": "1/4", "m²/s = 2/3": "2/3"},
        value="m²/s = 1/4",
        label="Quark mass",
    )
    method = mo.ui.dropdown(
        {
            "Physical · null reference": "null",
            "Physical · timelike reference": "timelike",
            "Covariant minus ghosts": "subtracted",
        },
        value="Covariant minus ghosts",
        label="Polarization sum",
    )
    cut = mo.ui.dropdown(
        {"|cosθ| < 0.2": 0.2, "|cosθ| < 0.5": 0.5, "|cosθ| < 0.8": 0.8},
        value="|cosθ| < 0.8",
        label="Angular acceptance",
    )
    mo.vstack([mo.hstack([colors, mass_ratio]), mo.hstack([method, cut])])
    return colors, cut, mass_ratio, method


@app.cell
def _(
    E,
    Nc,
    Symbol,
    angle_t,
    colors,
    density,
    gs,
    mass,
    mass_ratio,
    method,
    mo,
    physical,
    results,
    rho,
    s,
    t,
    u,
):
    _selected = (
        physical if method.value == "subtracted" else results["g", method.value] / 2
    )
    _fraction = E(mass_ratio.value)
    shown_squared = (
        (_selected / gs**4)
        .replace(u, 2 * mass**2 - s - t)
        .replace(t, angle_t)
        .replace(mass, (rho * s).sqrt())
        .replace(rho, _fraction)
        .replace(Nc, colors.value)
        .together()
    )
    _reference = (
        (2 * density / Symbol.PI).replace(rho, _fraction).replace(Nc, colors.value)
    )
    assert (shown_squared - _reference).together() == 0
    mo.vstack(
        [
            mo.md(
                r"**Squared amplitude divided by $g_s^4$ in center-of-mass variables**"
            ),
            shown_squared.factor(),
        ]
    )
    return


@app.cell
def _(
    E,
    Nc,
    colors,
    cut,
    cut_rate,
    cutoff,
    density,
    mass_ratio,
    mo,
    np,
    rho,
    z,
):
    _fraction = E(mass_ratio.value)
    selected_rate = cut_rate.replace(rho, _fraction).evaluate(
        {Nc: colors.value, cutoff: cut.value}
    )
    assert selected_rate.real > 0 and abs(selected_rate.imag) < 1e-10
    angular_rows = [
        {
            "cos(theta)": float(angle),
            "s/alpha_s² × dσ/dcos(theta)": density.replace(rho, _fraction)
            .evaluate({Nc: colors.value, z: float(angle)})
            .real,
        }
        for angle in np.linspace(-cut.value, cut.value, 11)
    ]
    mo.vstack(
        [
            mo.md(f"Accepted rate: **{selected_rate.real:.8g}** × αs²/s."),
            mo.ui.table(angular_rows, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
