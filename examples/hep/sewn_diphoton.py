import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Diphoton annihilation through sewing",
)


@app.cell
def _():
    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.spenso import CookSettings, TensorExpression

    _set_namespace("sewn_diphoton")

    return CookSettings, E, Replacement, S, TensorExpression, hep, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diphoton annihilation through sewing

    [Browse notebooks](/) · [Ordinary diphoton amplitudes](/?file=hep/diphoton.py) ·
    [Other sewn QED processes](/?file=hep/sewn_qed.py)

    Generate $e^-e^+ \to \gamma\gamma$ as forward graphs cut through the two
    final photons. Keep the electron mass, all interference terms and graph
    factors, then compare with the
    [FeynCalc result](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-GaGa).

    A stored cut-edge momentum has a direction. `cut.orientations` converts it
    into a positive-energy outgoing momentum before imposing the Mandelstam
    relations. Both choices of cut photon as the loop coordinate must give the
    same result. Particle lookup uses model names throughout.

    Incoming spins use the shared `Particle.sum_spins`. The outgoing photons
    use covariant sums or physical axial projectors from `Particle.spin_sum`.
    Null and timelike gauge references agree exactly; replacing either photon
    polarization by its momentum gives zero.
    """)
    return


@app.cell
def _(CookSettings, E, Replacement, S, TensorExpression, hep):
    def calculate_sewn_diphotons():
        model = hep.Model.standard_model()
        electron, photon = (model.particle(name) for name in ("e-", "a"))
        vertices = [
            vertex
            for vertex in model.vertex_rules
            if sorted(vertex.particles)
            == sorted([electron.name, electron.antiname, photon.name])
        ]
        assert len(vertices) == 1
        result = model.generate_diagrams(
            ["e-", "e+"],
            ["a", "a"],
            kind="cross_section",
            loops=1,
            max_vertices=4,
            maximum_bridges=None,
            vertex_allow=vertices,
            numerator_grouping=None,
            progress=None,
        )
        assert len(result.diagrams) == 2
        Q, P, K = S("gammalooprs::Q", "gammalooprs::P", "gammalooprs::K")
        s, t, u, mass, charge = S("s", "t", "u", "UFO::Me", "UFO::ee")
        k1, k2, index = S("k1", "k2", "index_")
        a, b, c, inverse = S("a_", "b_", "c_", "inverse_")
        metric, mink = S("spenso::g", "spenso::mink")
        kin = hep.Kinematics.mandelstam(
            [P(0), P(1), k1, k2], [mass**2, mass**2, E("0"), E("0")], [s, t, u]
        )
        x, y = t - mass**2, u - mass**2
        expected = (
            2
            * charge**4
            * (
                x / y
                + y / x
                + 4 * mass**2 * s / (x * y)
                - 4 * mass**4 * s**2 / (x**2 * y**2)
            )
        )
        expected = expected.replace(s, 2 * mass**2 - t - u).together()
        results = {}
        routing_rows = []
        for coordinate_index in (0, 1):
            for mode in ("covariant", "null", "timelike", "Ward 0", "Ward 1"):
                total = E("0")
                for original in result.diagrams:
                    coordinate = original.cuts[0].edges[coordinate_index].id
                    diagram = original.with_loop_momentum_edges([coordinate])
                    cut = diagram.cuts[0]
                    assert [particle.name for particle in cut.particles] == ["a", "a"]
                    assert all(side.loop_count == 0 for side in (cut.left, cut.right))
                    orientation = cut.orientations[coordinate]
                    if mode == "covariant":
                        routing_rows.append(
                            {
                                "Coordinate": coordinate_index + 1,
                                "Diagram": diagram.name,
                                "Cut edge": coordinate,
                                "K(0) / k1": orientation,
                            }
                        )
                    projector = diagram.projector_expression()
                    for edge in diagram.external_edges:
                        projector = model.particle(edge.particle_name).sum_spins(
                            projector, Q(edge.id), edge=edge.id, average=True
                        )
                    numerator = model.expand_couplings(
                        diagram.numerator_expression().to_expression() * projector
                    )
                    for position, edge in enumerate(cut.edges):
                        if mode == "covariant" or (
                            mode.startswith("Ward") and mode != f"Ward {position}"
                        ):
                            continue
                        # The photon propagator numerator is -i*g. Replacing g by the
                        # negative physical density retains its original propagator phase.
                        edge_metric = (
                            E("1𝑖") * edge.numerator_expression().to_expression()
                        )
                        match = dict(
                            next(
                                edge_metric.match(
                                    metric(mink(4, a), mink(4, b)), max_level=0
                                )
                            )
                        )
                        if mode.startswith("Ward"):
                            physical = Q(edge.id, mink(4, match[a])) * Q(
                                edge.id, mink(4, match[b])
                            )
                        else:
                            reference = (
                                P(0)
                                if mode == "timelike"
                                else Q(cut.edges[1 - position].id)
                            )
                            physical = photon.spin_sum(
                                Q(edge.id), match[a], match[b], reference=reference
                            )
                        numerator = numerator.replace(edge_metric, -physical)
                    scalar = (
                        TensorExpression(
                            numerator.expand(), cook_indices=CookSettings.indices()
                        )
                        .simplify_gamma()
                        .expand()
                        .simplify_gamma()
                        .expand()
                        .simplify_metrics()
                        .to_dots()
                    )
                    assert scalar.is_scalar
                    # A cut coordinate is a stored directed edge momentum. Convert it
                    # to the positive-energy outgoing photon before imposing kinematics.
                    contracted = kin.apply(
                        diagram.loop_momentum_basis.route_expression(
                            scalar.to_expression()
                        ).replace(K(0, index), orientation * k1(index))
                    )
                    denominator = kin.apply(
                        diagram.denominator_expression(
                            edge_powers={edge.id: 0 for edge in cut.edges},
                            dimension=4,
                            in_lmb=True,
                        )
                        .to_expression()
                        .replace(S("gammalooprs::denom")(a, b, c, inverse), inverse)
                        .replace(K(0, index), orientation * k1(index))
                    )
                    total += (
                        diagram.overall_factor_expression(evaluate=True)
                        * diagram.numerator_prefactor_expression()
                        * contracted
                        / denominator
                    ).replace(s, 2 * mass**2 - t - u)
                # The forward graphs identify the two identical cut photons. Restore
                # both labeled final-state assignments for the labeled squared amplitude.
                squared = (
                    total
                    + total.replace_multiple([Replacement(t, u), Replacement(u, t)])
                ).together()
                target = E("0") if mode.startswith("Ward") else expected
                assert (squared - target).together() == 0, (coordinate_index, mode)
                if not mode.startswith("Ward"):
                    assert (
                        squared.replace(mass, 0) - 2 * charge**4 * (t / u + u / t)
                    ).together() == 0
                    assert squared.replace(mass, 1).replace(charge, 1).replace(
                        t, -1
                    ).replace(u, -7).together() == E("83/8")
                results[coordinate_index, mode] = squared
                print("PASS", coordinate_index, mode, flush=True)
        assert results[0, "covariant"] == results[1, "covariant"]
        print(
            "Both cut coordinates, physical photon sums, Ward identities and massive diphoton reference passed",
            flush=True,
        )

        return results, result.diagrams, routing_rows, expected, mass, charge, t, u

    results, diagrams, routing_rows, expected, mass, charge, t, u = (
        calculate_sewn_diphotons()
    )
    return charge, diagrams, expected, mass, results, routing_rows, t, u


@app.cell
def _(diagrams, mo):
    mo.hstack(diagrams, justify="space-around")
    return


@app.cell(hide_code=True)
def _(mo, routing_rows):
    mo.vstack(
        [
            mo.md(r"""
        **Physical cut momenta and identical photons**

        The table gives the sign in $K(0)=\eta k_1$. Cut propagator denominators
        belong to the phase-space measure and are omitted from the squared
        amplitude. The two forward topologies identify the identical cut photons;
        adding the $t\leftrightarrow u$ assignment restores the fully labeled result.

        This labeled squared amplitude has no final-state $1/2!$ factor. The
        [ordinary-amplitude notebook](/?file=hep/diphoton.py) includes that factor
        when integrating an event rate over both photon labels.
        """),
            mo.ui.table(routing_rows, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    coordinate = mo.ui.dropdown(
        {"First cut photon": 0, "Second cut photon": 1},
        value="First cut photon",
        label="Loop coordinate",
    )
    polarization = mo.ui.dropdown(
        {
            "Covariant": "covariant",
            "Physical · null reference": "null",
            "Physical · timelike reference": "timelike",
            "Ward · first photon": "Ward 0",
            "Ward · second photon": "Ward 1",
        },
        value="Physical · timelike reference",
        label="Photon projectors",
    )
    massless = mo.ui.switch(value=False, label="Massless electron limit")
    mo.hstack([coordinate, polarization, massless])
    return coordinate, massless, polarization


@app.cell
def _(
    E,
    charge,
    coordinate,
    expected,
    mass,
    massless,
    mo,
    polarization,
    results,
    t,
    u,
):
    shown = results[coordinate.value, polarization.value]
    reference = E("0") if polarization.value.startswith("Ward") else expected
    if massless.value:
        shown = shown.replace(mass, 0)
        reference = reference.replace(mass, 0)
    residual = (shown - reference).together()
    assert residual == 0
    sample = (
        shown.replace(mass, 1)
        .replace(charge, 1)
        .replace(t, -1)
        .replace(u, -7)
        .together()
    )
    mo.vstack(
        [
            mo.md(r"**Spin-averaged labeled squared amplitude**"),
            shown.factor(),
            mo.md("**Exact difference from the reference**"),
            residual,
            mo.md(r"""
        With $e=m_e=1$, $t=-1$, $u=-7$ (hence $s=10$), the massive physical
        result is $83/8$. Ward projections vanish. In the massless limit,
        $\overline{|\mathcal M|^2}=2e^4(t/u+u/t)$ and $s=8$ at the same $t,u$.
        """),
            sample,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
