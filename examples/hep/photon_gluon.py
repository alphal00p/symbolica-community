import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Photon–gluon currents and crossing")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolica import E, S, Symbol
    from symbolica.community import hep
    from symbolica.community.spenso import CookSettings, TensorExpression

    return mo, np, E, S, Symbol, hep, CookSettings, TensorExpression


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Photon–gluon currents and crossing

    [Browse notebooks](/) · [Real QCD radiation](/?file=hep/photon_radiation.py) ·
    [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py)

    Generate both diagrams for each process:
    $\gamma^*g\to q\bar q$, $q\gamma^*\to gq$ and
    $q\bar q\to\gamma^*g$. Keep the quark mass, photon virtuality and
    generic SU($N_c$) color. Model particles and vertices are selected by name.

    These calculations reproduce FeynCalc's
    [photon–gluon fusion](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/GaGl-QQbar),
    [QCD Compton](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QGa-GlQ), and
    [annihilation](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QQbar-GaGl)
    examples. Each channel is generated independently before checking crossing.

    The off-shell photon current is contracted with $-g_{\mu\nu}$ without
    a photon spin average, matching the references. Incoming quark/gluon
    spins and colors are averaged. This current contraction is not a physical
    virtual-photon cross section; a lepton tensor is needed for an observable
    involving a virtual photon.
    """)
    return


@app.cell
def _(E, S, Symbol, hep, CookSettings, TensorExpression, np):
    def calculate_channels():
        model = hep.Model.standard_model()
        P = S("gammalooprs::P")
        mass, ee, gs = S("UFO::MB", "UFO::ee", "UFO::G")
        s, t, u, virtuality = S(
            "photon_gluon::s", "photon_gluon::t", "photon_gluon::u", "photon_gluon::q2"
        )
        Nc, dA, cof, coad = S(
            "photon_gluon::Nc", "photon_gluon::dA", "spenso::cof", "spenso::coad"
        )

        def calculate(names, photon_position, gluon_position, fermion_ports):
            masses = [
                mass**2
                if name in ("b", "b~")
                else virtuality
                if name == "a"
                else E("0")
                for name in names
            ]
            kin = hep.Kinematics.mandelstam([P(i) for i in range(4)], masses, [s, t, u])
            allowed = [
                v
                for v in model.vertex_rules
                if sorted(v.particles)
                in [
                    sorted(["b", "b~", "a"]),
                    sorted(["b", "b~", "g"]),
                ]
            ]
            generated = model.generate_diagrams(
                names[:2],
                names[2:],
                vertex_allow=allowed,
                max_vertices=2,
                maximum_bridges=None,
                numerator_grouping=None,
                progress=None,
            )
            assert len(generated.diagrams) == 2
            ports = S(
                "photon_gluon::i0",
                "photon_gluon::i1",
                "photon_gluon::i2",
                "photon_gluon::i3",
            )
            a, b, c, inv, index, left, right = S(
                "a_", "b_", "c_", "inv_", "index_", "left_", "right_"
            )
            conj, wrap, metric = S("spenso::conj", "photon_gluon::adjoint", "spenso::g")
            terms = []
            for diagram in generated.diagrams:
                numerator = model.expand_couplings(
                    diagram.numerator_expression(in_lmb=True).to_expression()
                )
                for half in diagram.half_edges:
                    edge = half.edge.data
                    if edge.is_external:
                        numerator = numerator.replace(
                            S("gammalooprs::hedge")(half.data, 1),
                            ports[edge.external_index],
                        )
                denominator = kin.apply(
                    diagram.denominator_expression(dimension=4, in_lmb=True)
                    .to_expression()
                    .replace(S("gammalooprs::denom")(a, b, c, inv), inv)
                ).expand()
                assert denominator in (s - mass**2, t - mass**2, u - mass**2), (
                    denominator
                )
                terms.append(
                    numerator
                    * diagram.overall_factor_expression(evaluate=True)
                    * diagram.numerator_prefactor_expression()
                    / denominator
                )
            operator = TensorExpression(sum(terms, E("0")).expand())
            adjoint = (
                operator.dirac_adjoint()
                .expand()
                .simplify_gamma0()
                .to_expression()
                .replace(conj(P(a, b)), P(a, b))
            )
            for real in (mass, ee, gs, s, t, u, virtuality):
                adjoint = adjoint.replace(conj(real), real)
            adjoint = TensorExpression(adjoint).wrap_indices(wrap)
            color_projector = E("1")
            initial_colors = 1
            generic_initial_colors = E("1")
            for position, name in enumerate(names):
                particle = model.particle(name)
                if particle.color == 1:
                    continue
                if position < 2:
                    initial_colors *= abs(particle.color)
                    generic_initial_colors *= dA if name == "g" else Nc
                matches = []
                for slot in operator.interface:
                    original = slot.to_expression()
                    if original.replace(ports[position], E("0")) == original:
                        continue
                    closure = metric(
                        slot.dual().to_expression(),
                        original.replace(ports[position], wrap(ports[position])),
                    )
                    match = next(
                        closure.match(particle.color_sum(left, right), max_level=0),
                        None,
                    )
                    if match is not None:
                        matches.append(dict(match))
                assert len(matches) == 1
                color_projector *= particle.color_sum(
                    matches[0][left], matches[0][right], average=position < 2
                )
            generic = (
                (
                    operator.to_expression()
                    * adjoint
                    * color_projector
                    * initial_colors
                    / generic_initial_colors
                )
                .replace(cof(3, index), cof(Nc, index))
                .replace(coad(8, index), coad(dA, index))
            )
            colored = (
                TensorExpression(generic, cook_indices=CookSettings.indices())
                .simplify_color()
                .to_expression()
                .replace(dA, Nc**2 - 1)
            )
            colored = TensorExpression(colored).to_cof_dimension_invariants()
            # Adjoint reverses the single open fermion chain. The explicit endpoint
            # pairing distinguishes pair creation, annihilation and Compton scattering.
            spins = E("1")
            for position, left_position, right_position, wrap_left in fermion_ports:
                left_port, right_port = ports[left_position], ports[right_position]
                if wrap_left:
                    left_port = wrap(left_port)
                else:
                    right_port = wrap(right_port)
                spins *= model.particle(names[position]).spin_sum(
                    P(position), left_port, right_port, average=position < 2
                )
            traced = (
                TensorExpression(
                    colored.to_expression() * spins,
                    cook_indices=CookSettings.indices(),
                )
                .expand()
                .simplify_gamma()
                .expand()
                .simplify_gamma()
            )
            reduced = kin.apply(traced.expand().simplify_metrics().to_dots())
            pieces = [
                (tensor, coeff.together()) for tensor, coeff in reduced.expand_mink()
            ]
            results = {}
            modes = (
                "covariant",
                "photon reference",
                "quark reference",
                "gluon Ward",
                "photon Ward",
            )
            quark_position = next(i for i, name in enumerate(names) if name == "b")
            for mode in modes:
                photon = model.particle("a").spin_sum(
                    P(photon_position),
                    ports[photon_position],
                    wrap(ports[photon_position]),
                    covariant=True,
                )
                reference = (
                    P(photon_position)
                    if mode == "photon reference"
                    else P(quark_position)
                    if mode == "quark reference"
                    else None
                )
                gluon = model.particle("g").spin_sum(
                    P(gluon_position),
                    ports[gluon_position],
                    wrap(ports[gluon_position]),
                    reference=reference,
                    covariant=reference is None,
                    average=gluon_position < 2,
                )
                if mode == "gluon Ward":
                    gluon = P(
                        gluon_position, S("spenso::mink")(4, ports[gluon_position])
                    ) * P(
                        gluon_position,
                        S("spenso::mink")(4, wrap(ports[gluon_position])),
                    )
                if mode == "photon Ward":
                    photon = P(
                        photon_position, S("spenso::mink")(4, ports[photon_position])
                    ) * P(
                        photon_position,
                        S("spenso::mink")(4, wrap(ports[photon_position])),
                    )
                contractions = []
                for tensor, coeff in pieces:
                    contracted = (
                        TensorExpression(
                            (tensor * photon * gluon).expand(),
                            cook_indices=CookSettings.indices(),
                        )
                        .simplify_metrics()
                        .to_dots()
                    )
                    assert contracted.is_scalar
                    contractions.append(coeff * kin.apply(contracted.to_expression()))
                results[mode] = (
                    sum(contractions, E("0"))
                    .replace(u, 2 * mass**2 + virtuality - s - t)
                    .together()
                )
            for mode in [name for name in modes if name.endswith("Ward")]:
                assert results[mode] == E("0")
            for mode in [name for name in modes if name.endswith("reference")]:
                assert (results[mode] - results["covariant"]).together() == E("0")

            return generated, results, kin

        # Reference polynomial shared by the crossed channels, independently stated
        # in invariant variables. No crossing rule is inserted into the amplitudes.
        def reference_polynomial(x, y):
            return (
                -(mass**4)
                * (
                    2 * virtuality**2
                    - 2 * virtuality * (x + y)
                    + 3 * x**2
                    + 14 * x * y
                    + 3 * y**2
                )
                + mass**2
                * (
                    2 * virtuality**2 * (x + y)
                    - 8 * virtuality * x * y
                    + x**3
                    + 7 * x**2 * y
                    + 7 * x * y**2
                    + y**3
                )
                + 6 * mass**8
                - x * y * (2 * virtuality**2 - 2 * virtuality * (x + y) + x**2 + y**2)
            ) / ((x - mass**2) ** 2 * (y - mass**2) ** 2)

        channels = {}
        for channel, names, photon, gluon, spin_ports, prefactor, pair in (
            (
                "fusion",
                ["a", "g", "b", "b~"],
                0,
                1,
                [(2, 3, 2, True), (3, 3, 2, False)],
                -2 * ee**2 * gs**2 / 9,
                (t, u),
            ),
            (
                "compton",
                ["b", "a", "g", "b"],
                1,
                2,
                [(0, 0, 3, False), (3, 0, 3, True)],
                2 * ee**2 * gs**2 * (Nc**2 - 1) / (9 * Nc),
                (s, t),
            ),
            (
                "annihilation",
                ["b", "b~", "a", "g"],
                2,
                3,
                [(0, 0, 1, False), (1, 0, 1, True)],
                -(ee**2) * gs**2 * (Nc**2 - 1) / (9 * Nc**2),
                (t, u),
            ),
        ):
            generated, results, kin = calculate(names, photon, gluon, spin_ports)
            target = (
                (prefactor * reference_polynomial(*pair))
                .replace(u, 2 * mass**2 + virtuality - s - t)
                .together()
            )
            delta = (results["covariant"] - target).together()
            assert delta == E("0"), channel
            x, y = pair
            massless_target = (
                -prefactor
                * (2 * virtuality**2 - 2 * virtuality * (x + y) + x**2 + y**2)
                / (x * y)
            )
            assert (
                results["covariant"].replace(mass, 0)
                - massless_target.replace(u, virtuality - s - t)
            ).together() == E("0")
            channels[channel] = (generated, results, kin)

        fusion = channels["fusion"][1]["covariant"]
        compton = channels["compton"][1]["covariant"]
        annihilation = channels["annihilation"][1]["covariant"]
        # Crossing changes both the fermion-trace sign and the incoming color average.
        crossed_fusion = fusion.replace(s, 2 * mass**2 + virtuality - s - t)
        assert (compton + (Nc**2 - 1) / Nc * crossed_fusion).together() == E("0")
        assert (annihilation - (Nc**2 - 1) / (2 * Nc**2) * fusion).together() == E("0")
        assert (
            fusion - fusion.replace(t, 2 * mass**2 + virtuality - s - t)
        ).together() == E("0")
        # Check the reference spacelike continuation q^2=-Q^2 explicitly at zero mass.
        Q2 = S("photon_gluon::Q2", is_positive=True)
        for channel, x, y, prefactor in (
            ("fusion", t, u, 2 * ee**2 * gs**2 / 9),
            ("compton", s, t, -16 * ee**2 * gs**2 / 27),
        ):
            reference = prefactor * (x / y + y / x + 2 * Q2 * (x + y + Q2) / (x * y))
            result = (
                channels[channel][1]["covariant"]
                .replace(mass, 0)
                .replace(virtuality, -Q2)
                .replace(Nc, 3)
            )
            assert (result - reference.replace(u, -Q2 - s - t)).together() == E("0")

        # Only for a real photon do we form ordinary two-body cross sections.
        # Incoming photons have two physical states, so include their missing 1/2;
        # the off-shell current convention used above intentionally has no such average.
        z, rho, beta = S("photon_gluon::z", "photon_gluon::rho", "photon_gluon::beta")
        alpha, alpha_s = S("photon_gluon::alpha", "photon_gluon::alpha_s")
        angular = {}
        for channel, (_, sums, kin) in channels.items():
            phase = (
                (kin.two_body_phase_space(P(2), P(3)) / kin.flux(P(0), P(1)))
                .replace(virtuality, 0)
                .together()
            )
            # Squaring the measure ratio checks its invariant normalization without
            # imposing a branch identity outside the physical region.
            velocity_squared = 1 - 4 * mass**2 / s
            factor_squared = (
                velocity_squared
                if channel == "fusion"
                else 1 / velocity_squared
                if channel == "annihilation"
                else E("1")
            )
            assert (
                phase**2 - factor_squared / (64 * Symbol.PI**2 * s) ** 2
            ).together() == E("0")
            angle_t = (
                mass**2 - s * (1 - beta * z) / 2
                if channel != "compton"
                else mass**2
                - (s**2 - mass**4) / (2 * s)
                + (s - mass**2) ** 2 * z / (2 * s)
            )
            # Normalize out alpha*alpha_s/s. Final particles are distinct in all channels.
            density = (
                sums["covariant"].replace(virtuality, 0).replace(t, angle_t)
                * phase
                * 2
                * Symbol.PI
                * s
                / (alpha * alpha_s)
            )
            density = density.replace(ee**2, 4 * Symbol.PI * alpha).replace(
                gs**2, 4 * Symbol.PI * alpha_s
            )
            if channel != "annihilation":
                density /= 2
            density = density.replace(mass, (rho * s).sqrt()).together()
            angular[channel] = density
            for nc in (2, 3, 5):
                for r in (0.0, 0.02, 0.1):
                    for cosine in (-0.7, 0.0, 0.7):
                        values = {
                            s: 100.0,
                            Nc: nc,
                            rho: r,
                            beta: np.sqrt(1 - 4 * r),
                            z: cosine,
                        }
                        exact = complex(density.evaluate(values))
                        scaled = complex(density.evaluate({**values, s: 400.0}))
                        assert abs(exact.imag) < 1e-12 and exact.real > 0
                        assert abs(exact - scaled) < 1e-10

        # Off-shell currents at explicit center-of-mass invariants, with incoming
        # photons spacelike and the outgoing photon timelike.
        for channel, (_, sums, _) in channels.items():
            for r in (0.0, 0.02, 0.1):
                for v in (0.05, 0.2):
                    v = v if channel == "annihilation" else -v
                    for cosine in (-0.7, 0.0, 0.7):
                        kallen = 1 + r * r + v * v - 2 * r - 2 * v - 2 * r * v
                        angle_t = (
                            r - (1 - v) * (1 - np.sqrt(1 - 4 * r) * cosine) / 2
                            if channel != "compton"
                            else r
                            - (1 + r - v) * (1 - r) / 2
                            + (1 - r) * np.sqrt(kallen) * cosine / 2
                        )
                        point = {
                            s: 1,
                            t: angle_t,
                            mass: np.sqrt(r),
                            virtuality: v,
                            Nc: 3,
                            ee: 1,
                            gs: 1,
                        }
                        values = [
                            complex(sums[mode].evaluate(point))
                            for mode in (
                                "covariant",
                                "photon reference",
                                "quark reference",
                            )
                        ]
                        assert max(abs(value.imag) for value in values) < 1e-10
                        assert max(abs(value - values[0]) for value in values) < 1e-8
                        assert all(np.isfinite(value.real) for value in values)
        return {
            "channels": channels,
            "angular": angular,
            "mass": mass,
            "ee": ee,
            "gs": gs,
            "s": s,
            "t": t,
            "u": u,
            "virtuality": virtuality,
            "Nc": Nc,
            "z": z,
            "rho": rho,
            "beta": beta,
        }

    calculation = calculate_channels()
    return (calculation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checks before choosing a numerical point

    - Both diagrams and their interference reproduce the complete massive references.
    - Photon and gluon Ward contractions vanish in every channel.
    - Gluon references along the quark or virtual photon give the same current contraction.
    - Exact crossing includes the fermion sign and changed incoming color averages.
    - The massless spacelike-photon limits agree with the reference formulas.
    - Shared flux and two-body phase space supply the real-photon angular rates.

    Write $F(s,t,u)$ for the fusion current contraction. With the same invariant
    arguments, annihilation gives $(N_c^2-1)F/(2N_c^2)$; crossing $s\leftrightarrow u$
    gives the Compton current with factor $-(N_c^2-1)/N_c$.
    """)
    return


@app.cell
def _(mo):
    channel = mo.ui.dropdown(
        {
            "Photon–gluon fusion": "fusion",
            "QCD Compton": "compton",
            "Quark annihilation": "annihilation",
        },
        value="Photon–gluon fusion",
        label="Process",
    )
    colors = mo.ui.dropdown(
        {f"SU({n})": n for n in (2, 3, 5)}, value="SU(3)", label="Color group"
    )
    mass_ratio = mo.ui.slider(
        0, 0.2, step=0.01, value=0.02, label="Quark mass squared / s"
    )
    virtuality_ratio = mo.ui.slider(
        0, 0.3, step=0.01, value=0.2, label="Photon |q²| / s"
    )
    angle = mo.ui.slider(-0.8, 0.8, step=0.1, value=0.0, label="cos θ")
    mo.vstack(
        [mo.hstack([channel, colors]), mo.hstack([mass_ratio, virtuality_ratio, angle])]
    )
    return channel, colors, mass_ratio, virtuality_ratio, angle


@app.cell
def _(calculation, channel, mo):
    _diagrams = calculation["channels"][channel.value][0].diagrams
    mo.vstack(
        [
            mo.md("## Generated diagrams"),
            mo.hstack(_diagrams),
            mo.accordion(
                {
                    "Exact massive current contraction": calculation["channels"][
                        channel.value
                    ][1]["covariant"]
                }
            ),
        ]
    )
    return


@app.cell
def _(calculation, channel, colors, mass_ratio, virtuality_ratio, angle, np, mo):
    _r = mass_ratio.value
    signed_virtuality = virtuality_ratio.value * (
        1 if channel.value == "annihilation" else -1
    )
    _v = signed_virtuality
    _beta = np.sqrt(1 - 4 * _r)
    _kallen = 1 + _r**2 + _v**2 - 2 * _r - 2 * _v - 2 * _r * _v
    _shape = (
        9
        * calculation["channels"][channel.value][1]["covariant"]
        / (calculation["ee"] ** 2 * calculation["gs"] ** 2)
    )
    _shape = _shape.together()
    _rate = calculation["angular"][channel.value]
    angular_rows = []
    selected_current = None
    selected_density = None
    for _z in sorted(
        set(
            [round(float(v), 1) for v in np.linspace(-0.8, 0.8, 17)]
            + [round(angle.value, 1)]
        )
    ):
        if channel.value == "compton":
            _t = (
                _r - (1 + _r - _v) * (1 - _r) / 2 + (1 - _r) * np.sqrt(_kallen) * _z / 2
            )
        else:
            _t = _r - (1 - _v) * (1 - _beta * _z) / 2
        _point = {
            calculation["s"]: 1,
            calculation["t"]: _t,
            calculation["mass"]: np.sqrt(_r),
            calculation["virtuality"]: _v,
            calculation["Nc"]: colors.value,
        }
        _current = complex(_shape.evaluate(_point))
        _density = complex(
            _rate.evaluate(
                {
                    calculation["s"]: 1,
                    calculation["rho"]: _r,
                    calculation["beta"]: _beta,
                    calculation["Nc"]: colors.value,
                    calculation["z"]: _z,
                }
            )
        )
        assert (
            abs(_current.imag) < 1e-8
            and abs(_density.imag) < 1e-10
            and _density.real > 0
        )
        angular_rows.append(
            {
                "cos θ": _z,
                "Off-shell current / (e² gs² Qq²)": _current.real,
                "Real-photon s/(α αs) dσ/dcosθ": _density.real,
            }
        )
        if abs(_z - angle.value) < 1e-8:
            selected_current, selected_density = _current.real, _density.real
    assert selected_current is not None and selected_density is not None
    # Gaussian quadrature convergence checks the additional angular-cut rate.
    _integrals = []
    for _order in (48, 96):
        _nodes, _weights = np.polynomial.legendre.leggauss(_order)
        _values = [
            complex(
                _rate.evaluate(
                    {
                        calculation["s"]: 1,
                        calculation["rho"]: _r,
                        calculation["beta"]: _beta,
                        calculation["Nc"]: colors.value,
                        calculation["z"]: 0.8 * _node,
                    }
                )
            ).real
            for _node in _nodes
        ]
        _integrals.append(0.8 * float(np.dot(_weights, _values)))
    assert abs(_integrals[0] - _integrals[1]) < 1e-9
    cut_rate = _integrals[1]
    mo.vstack(
        [
            mo.md("## Off-shell current and real-photon rate"),
            mo.md(
                f"**Photon q²/s = {signed_virtuality:.2f}.** Incoming photons are spacelike; outgoing photons are timelike."
            ),
            mo.md(f"**Selected current / (e² gs² Qq²):** {selected_current:.8g}"),
            mo.md(f"**Real-photon s/(α αs) dσ/dcosθ:** {selected_density:.8g}"),
            mo.md(
                f"**Real-photon angular-cut rate s σ/(α αs), |cosθ| < 0.8:** {cut_rate:.8g}"
            ),
            mo.md(r"""
        The angle is between the first incoming and first outgoing particle.
        The off-shell column uses the displayed virtuality. The rate column sets
        $q^2=0$ and averages an incoming photon's two physical polarizations.
        All final particles are distinct. To obtain a rate, multiply by
        $\alpha\alpha_s/s$; the model's bottom-quark charge is $Q_q=-1/3$.
        The angular cut keeps the massless collinear endpoints outside the integral.
        """),
            mo.ui.table(angular_rows, selection=None),
        ]
    )
    return angular_rows, selected_current, selected_density, cut_rate, signed_virtuality


if __name__ == "__main__":
    app.run()
