import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Light-cone soft radiation")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.spenso import (
        GammaChainOrdering,
        GammaSimplifySettings,
        Representation,
        SchoonschipSettings,
        TensorExpression,
        TensorName,
    )

    _set_namespace("soft")

    return (
        E,
        S,
        Symbol,
        hep,
        GammaChainOrdering,
        GammaSimplifySettings,
        Representation,
        SchoonschipSettings,
        TensorExpression,
        TensorName,
        mo,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Light-cone soft radiation

    [Browse all notebooks](/) · [Full QCD radiation](/?file=hep/photon_radiation.py) · [Quark–gluon scattering](/?file=hep/quark_gluon_scattering.py) ·
    [Polarization sums](/?file=hep/polarization_sums.py)

    Generate $\gamma^*\to q\bar q$ and both diagrams for
    $\gamma^*\to q\bar qg$, then take the ultrasoft limit $k\to\lambda^2 k$.
    The calculation follows
    [FeynCalc's SCET soft-function example](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/Ga-QQbar-SoftFunction).

    The quark and antiquark move along null directions $n,\bar n$, with
    $n\cdot\bar n=2$. Their independent large momentum components are $Q,\bar Q$.
    The gluon is massless. Lorentz dimension $D$ stays symbolic; the spinor
    representation has dimension four.

    The generated amplitudes retain open spinor and photon indices. Their
    leading terms factor onto the same Born current before any spin sum.
    Only gluon polarizations and final-state colors are summed.
    """)
    return


@app.cell
def _(
    E,
    S,
    Symbol,
    hep,
    GammaChainOrdering,
    GammaSimplifySettings,
    Representation,
    SchoonschipSettings,
    TensorExpression,
    TensorName,
):
    def calculate_soft_function():
        model = hep.Model.standard_model()
        vertices = [
            vertex
            for vertex in model.vertex_rules
            if sorted(vertex.particles)
            in [sorted(["b", "b~", "a"]), sorted(["b", "b~", "g"])]
        ]
        assert len(vertices) == 2
        P = S("gammalooprs::P")
        D, Q, Qbar, kp, km, lam = S("D", "Q", "Qbar", "kp", "km", "lam")
        # Rank-one declarations let Spenso recognize contracted vectors and slashes.
        n, nb, k, ref = [
            TensorName.vector(name).to_expression() for name in ("n", "nb", "k", "r")
        ]
        mink, bis = S("spenso::mink", "spenso::bis")
        zero, one = E("0"), E("1")
        kin = hep.Kinematics(D, momenta=[n, nb, k])
        for vector in (n, nb, k):
            kin = kin.with_scalar_product(vector, vector, zero)
        kin = (
            kin.with_scalar_product(n, nb, E("2"))
            .with_scalar_product(n, k, kp)
            .with_scalar_product(nb, k, km)
        )
        ports = S("i0", "i1", "i2", "i3")
        a, b, c, inv, idx = S("a_", "b_", "c_", "inv_", "idx_")
        gamma, metric = S("spenso::gamma", "spenso::g")
        si, sj, sx, sy, sm, sn = S("si", "sj", "sx", "sy", "sm", "sn")
        ordering = GammaSimplifySettings(chain_ordering=GammaChainOrdering.Canonical)

        def simplify(expression):
            return (
                kin.apply(
                    TensorExpression(expression)
                    .expand()
                    .schoonschip(
                        SchoonschipSettings(simplify_chain_like_functions=True)
                    )
                    .simplify_gamma(ordering)
                    .expand()
                    .simplify_metrics()
                    .to_dots()
                )
                .to_expression()
                .expand()
            )

        def projector(left, right, middle):
            # P_minus = nbar_slash n_slash / 4; all spinor dimensions remain four.
            return (
                gamma(bis(4, left), bis(4, middle), nb(mink(D)))
                * gamma(bis(4, middle), bis(4, right), n(mink(D)))
                / 4
            )

        def project(expression):
            # The collinear bra and anticollinear ket both select P_minus.
            expression = expression.replace(bis(4, ports[1]), bis(4, sx)).replace(
                bis(4, ports[2]), bis(4, sy)
            )
            return (
                projector(ports[1], sx, sm) * expression * projector(sy, ports[2], sn)
            )

        # Check the large-component projector before applying it to generated amplitudes.
        pm = projector(si, sj, sm)
        identity = metric(bis(4, si), bis(4, sj))
        pp = identity - pm
        assert (
            simplify(pm.replace(bis(4, sj), bis(4, sx)) * projector(sx, sj, sn) - pm)
            == zero
        )
        assert (
            simplify(
                pm.replace(bis(4, sj), bis(4, sx))
                * pp.replace(bis(4, si), bis(4, sx)).replace(bis(4, sm), bis(4, sn))
            )
            == zero
        )
        assert simplify(pm.replace(bis(4, sj), bis(4, si))) == 2
        assert (
            simplify(
                pm.replace(bis(4, sj), bis(4, sx))
                * gamma(bis(4, sx), bis(4, sj), n(mink(D)))
            )
            == zero
        )
        assert (
            simplify(
                gamma(bis(4, si), bis(4, sx), nb(mink(D)))
                * pm.replace(bis(4, si), bis(4, sx))
            )
            == zero
        )

        # The transverse metric projects onto the D-2 directions orthogonal to n,nbar.
        mu, nu, rho = S("mu", "nu", "rho")
        transverse = (
            metric(mink(D, mu), mink(D, nu))
            - (n(mink(D, mu)) * nb(mink(D, nu)) + nb(mink(D, mu)) * n(mink(D, nu))) / 2
        )
        assert simplify(transverse * n(mink(D, nu))) == zero
        assert simplify(transverse * nb(mink(D, nu))) == zero
        assert simplify(transverse * metric(mink(D, mu), mink(D, nu))) == D - 2
        assert (
            simplify(
                transverse * transverse.replace(mu, rho) - transverse.replace(nu, rho)
            )
            == zero
        )

        channels, denominators, generated_channels = [], [], []
        for outgoing in (["b", "b~"], ["b", "b~", "g"]):
            result = hep.Process(model, ["a"], outgoing).generate_diagrams(
                vertex_allow=vertices,
                max_vertices=len(outgoing) - 1,
                maximum_bridges=None,
                numerator_grouping=None,
                progress=None,
            )
            assert len(result.diagrams) == len(outgoing) - 1
            generated_channels.append(result)
            for diagram in result.diagrams:
                numerator = model.expand_couplings(
                    diagram.numerator_expression(in_lmb=True).to_expression()
                ).replace(S("UFO::MB"), zero)
                for half in diagram.half_edges:
                    edge = half.edge.data
                    if edge.is_external:
                        numerator = numerator.replace(
                            S("gammalooprs::hedge")(half.data, 1),
                            ports[edge.external_index],
                        )
                numerator = (
                    TensorExpression(numerator)
                    .with_lorentz_dimension(D)
                    .to_expression()
                )
                denominator = (
                    diagram.denominator_expression(dimension=D, in_lmb=True)
                    .to_expression()
                    .replace(S("gammalooprs::denom")(a, b, c, inv), inv)
                    .replace(S("UFO::MB"), zero)
                )
                # Expand the dot products before substituting linear vector combinations.
                denominator = TensorExpression(denominator).undo_dots().to_expression()
                for position, momentum in (
                    (0, (Q * n(idx) + Qbar * nb(idx)) / 2 + lam**2 * k(idx)),
                    (1, Q * n(idx) / 2),
                    (2, Qbar * nb(idx) / 2),
                    (3, lam**2 * k(idx)),
                ):
                    numerator = numerator.replace(P(position, idx), momentum)
                    denominator = denominator.replace(P(position, idx), momentum)
                denominator = (
                    kin.apply(
                        TensorExpression(denominator)
                        .expand()
                        .simplify_metrics()
                        .to_dots()
                    )
                    .to_expression()
                    .expand()
                )
                denominators.append(denominator)
                amplitude = (
                    numerator
                    * diagram.overall_factor_expression(evaluate=True)
                    * diagram.numerator_prefactor_expression()
                    / denominator
                )
                leading = amplitude.series(
                    lam, 0, -2 if len(outgoing) == 3 else 0
                ).to_expression()
                channels.append(simplify(project(leading.replace(lam, one))))
        assert set(denominators) == {one, Q * kp * lam**2, Qbar * km * lam**2}
        born = channels[0]
        assert simplify(born * n(mink(D, ports[0]))) == zero
        assert simplify(born * nb(mink(D, ports[0]))) == zero

        # Keep the photon and both spinor ports open; divide out only the Born color delta.
        color_identity = TensorExpression.g(
            Representation.cof(3), Representation.cof(3).dual()
        )(ports[2], ports[1]).to_expression()
        current = (born / color_identity).together()
        generator = TensorExpression.t(8, 3)(
            ports[3], ports[2], ports[1]
        ).to_expression()
        gs = S("UFO::G")
        # The generated convention gives a common minus relative to the reference's
        # emission current. Its relative quark/antiquark sign and square agree.
        for emitted, vector, product, sign in zip(
            channels[1:], (n, nb), (kp, km), (-1, 1)
        ):
            expected = (
                sign * gs * vector(mink(D, ports[3])) / product * current * generator
            )
            assert simplify(emitted - expected).together() == zero
        eikonal = ((channels[1] + channels[2]) / (current * generator * gs)).together()
        assert (
            eikonal + n(mink(D, ports[3])) / kp - nb(mink(D, ports[3])) / km
        ).expand() == zero
        assert simplify(eikonal * k(mink(D, ports[3]))) == zero
        assert eikonal.derivative(Q) == zero and eikonal.derivative(Qbar) == zero

        # Close color indices without summing or averaging the external spin states.
        Nc, dA = S("Nc", "dA")
        fund = Representation.cof(Nc)
        color_born = TensorExpression.g(fund, fund.dual())(ports[2], ports[1])
        color_real = TensorExpression.t(dA, Nc)(ports[3], ports[2], ports[1])
        colors = []
        for tensor in (color_born, color_real):
            norm = (
                TensorExpression(
                    tensor.to_expression() * tensor.spenso_conjugate().to_expression()
                )
                .simplify_color()
                .to_expression()
            )
            colors.append(
                TensorExpression(norm.replace(dA, Nc**2 - 1))
                .to_cof_dimension_invariants()
                .to_expression()
            )
        assert colors == [Nc, (Nc**2 - 1) / 2]
        cf = (colors[1] / colors[0]).together()

        # An arbitrary reference includes a nonzero r^2; no gauge term may survive.
        rn, rnb, rk, r2 = S("rn", "rnb", "rk", "r2")
        kin = (
            kin.with_scalar_product(ref, n, rn)
            .with_scalar_product(ref, nb, rnb)
            .with_scalar_product(ref, k, rk)
            .with_scalar_product(ref, ref, r2)
        )
        mu2 = S("mu2")
        polarization_results = {}
        for label, reference in (
            ("Covariant", None),
            ("n", n),
            ("nbar", nb),
            ("Arbitrary reference", ref),
        ):
            density = model.particle("g").spin_sum(
                k,
                ports[3],
                mu2,
                dimension=D,
                reference=reference,
                covariant=reference is None,
            )
            squared = simplify(
                eikonal * eikonal.replace(ports[3], mu2) * density
            ).together()
            assert (squared - 4 / (kp * km)).together() == zero
            polarization_results[label] = squared
        alpha = S("alpha_s")
        soft_function = (
            squared * cf * 4 * Symbol.PI * alpha / (32 * Symbol.PI**2)
        ).together()
        assert (
            soft_function - cf * alpha / (2 * Symbol.PI * kp * km)
        ).together() == zero
        eta = S("eta")
        assert (
            soft_function.replace(kp, eta * kp).replace(km, km / eta) - soft_function
        ).together() == zero
        energy, z = S("energy", "z")
        angular_soft = (
            soft_function.replace(kp, energy * (1 - z))
            .replace(km, energy * (1 + z))
            .together()
        )
        assert (
            angular_soft.replace(energy, 2 * energy) - angular_soft / 4
        ).together() == zero
        return {
            "generated_channels": generated_channels,
            "D": D,
            "n": n,
            "nb": nb,
            "k": k,
            "kp": kp,
            "km": km,
            "Q": Q,
            "Qbar": Qbar,
            "pm": pm,
            "transverse": transverse,
            "born": born,
            "current": current,
            "eikonal": eikonal,
            "colors": colors,
            "cf": cf,
            "Nc": Nc,
            "alpha": alpha,
            "energy": energy,
            "z": z,
            "soft_function": soft_function,
            "angular_soft": angular_soft,
            "polarization_results": polarization_results,
        }

    soft = calculate_soft_function()
    return (soft,)


@app.cell
def _(mo, soft):
    mo.vstack(
        [
            mo.md("## Generated amplitudes"),
            mo.md("One Born diagram and two real-emission diagrams:"),
            mo.hstack(soft["generated_channels"][0].diagrams),
            mo.hstack(soft["generated_channels"][1].diagrams),
        ]
    )
    return


@app.cell
def _(TensorExpression, mo, soft):
    mo.vstack(
        [
            mo.md(r"""
        ## Collinear projection

        The transverse metric is
        $g_\perp^{\mu\nu}=g^{\mu\nu}-
        (n^\mu\bar n^\nu+\bar n^\mu n^\nu)/2$.
        It is idempotent, annihilates both light-cone vectors and has trace $D-2$.

        Both the collinear bra and anticollinear ket select
        $P_-=\not{\bar n}\not n/4$.
        The calculation checks $P_-^2=P_-$, $P_-(1-P_-)=0$,
        $\mathrm{tr}P_-=2$ and the corresponding Dirac annihilation identities.
        """),
            TensorExpression(soft["transverse"]),
            TensorExpression(soft["pm"]),
            mo.md("**Projected Born amplitude:**"),
            TensorExpression(soft["born"]),
        ]
    )
    return


@app.cell
def _(TensorExpression, mo, soft):
    mo.vstack(
        [
            mo.md(r"""
        ## Factorization and gauge independence

        After a Laurent expansion through $\lambda^{-2}$, each generated
        emission diagram equals the Born current times its color generator
        and an eikonal factor. The independent hard scales $Q,\bar Q$ cancel.

        The current below is extracted from their sum. Its common sign depends
        on the emission convention and cancels from the square; the relative
        quark/antiquark sign is fixed by the generated diagrams.
        Contracting it with $k$ gives zero.
        """),
            TensorExpression(soft["eikonal"]),
            mo.md(r"""
        The Born and emitted color norms are $N_c$ and $(N_c^2-1)/2$,
        so their ratio is $C_F=(N_c^2-1)/(2N_c)$. The common open Born current
        cancels. All four polarization choices below agree exactly, including
        a generic reference vector with arbitrary nonzero norm.
        """),
            mo.ui.table(
                [
                    {"Polarization sum": label, "Squared eikonal current": str(value)}
                    for label, value in soft["polarization_results"].items()
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo, soft):
    colors = mo.ui.dropdown(
        {f"SU({n})": n for n in (2, 3, 5)}, value="SU(3)", label="Color group"
    )
    polarization = mo.ui.dropdown(
        list(soft["polarization_results"]),
        value="Arbitrary reference",
        label="Gluon polarization sum",
    )
    gluon_energy = mo.ui.slider(0.5, 5.0, step=0.5, value=2.0, label="Gluon energy")
    angle = mo.ui.slider(-0.9, 0.9, step=0.1, value=0.0, label="cos θ")
    coupling = mo.ui.slider(0.05, 0.2, step=0.001, value=0.118, label="αs")
    mo.vstack(
        [mo.hstack([colors, polarization]), mo.hstack([gluon_energy, angle, coupling])]
    )
    return colors, polarization, gluon_energy, angle, coupling


@app.cell
def _(Symbol, angle, colors, coupling, gluon_energy, mo, np, polarization, soft):
    selected_soft = (
        soft["polarization_results"][polarization.value]
        * soft["cf"]
        * 4
        * Symbol.PI
        * soft["alpha"]
        / (32 * Symbol.PI**2)
    ).together()
    assert (selected_soft - soft["soft_function"]).together() == 0
    _parameters = {
        soft["Nc"]: colors.value,
        soft["alpha"]: coupling.value,
        soft["energy"]: gluon_energy.value,
        soft["z"]: angle.value,
    }
    selected_value = complex(soft["angular_soft"].evaluate(_parameters)).real
    assert selected_value > 0
    angular_rows = []
    for _z in np.linspace(-0.9, 0.9, 19):
        _point = {**_parameters, soft["z"]: float(_z)}
        _value = complex(soft["angular_soft"].evaluate(_point)).real
        _double_energy = complex(
            soft["angular_soft"].evaluate(
                {**_point, soft["energy"]: 2 * gluon_energy.value}
            )
        ).real
        assert abs(_double_energy - _value / 4) < 1e-12
        angular_rows.append(
            {
                "cos θ": round(float(_z), 2),
                "Soft function": _value,
                "At twice the energy": _double_energy,
            }
        )
    mo.vstack(
        [
            mo.md(r"""
        ## Leading soft function

        With $g_s^2=4\pi\alpha_s$ and the reference's explicit
        prefactor $1/(32\pi^2)$,
        $$S^{(0)}(k)=\frac{C_F\alpha_s}{2\pi k^+k^-},\qquad
        k^+=n\cdot k,\quad k^-=\bar n\cdot k.$$

        In the back-to-back frame $k^\pm=E_g(1\mp\cos\theta)$.
        The table shows both the collinear enhancement and the $E_g^{-2}$
        soft scaling. Energy values use any consistent units; $S^{(0)}$ has
        inverse-energy-squared units.

        This is the unintegrated leading soft kernel with the stated prefactor.
        No phase-space integration or infrared regulator is included.
        """),
            selected_soft,
            mo.md(f"**Selected value: {selected_value:.8g}**"),
            mo.ui.table(angular_rows, selection=None),
        ]
    )
    return angular_rows, selected_soft, selected_value


if __name__ == "__main__":
    app.run()
