import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="One-loop QED renormalization")


@app.cell
def _():
    import json

    import marimo as mo
    from symbolica import E, Matrix, S, Symbol
    from symbolica.community import hep
    from symbolica.community.hep import oneloop
    from symbolica.community.spenso import TensorExpression

    return E, Matrix, S, Symbol, TensorExpression, hep, json, mo, oneloop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop QED renormalization

    [Browse all notebooks](/) · [Electron g−2](/?file=hep/gminus2.py) ·
    [Two-loop photon](/?file=hep/photon_two_loop.py) ·
    [Two-loop electron](/?file=hep/electron_two_loop.py)

    Generate the massive electron self-energy, photon self-energy and
    electron-photon vertex with a **symbolic covariant gauge parameter** $\xi$.
    Their ultraviolet poles determine all six constants in the
    [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/Renormalization).

    The model's photon numerator is
    $-i[g^{\mu\nu}-(1-\xi)q^\mu q^\nu/q^2]$.
    Feynkit expands each graph at large loop momentum, keeping the mass
    corrections through its degree of divergence. Shared Dirac traces and
    vacuum tensor reduction feed the native IBP solver.

    We use $D=4-2\epsilon$, $a_4=e^2/(16\pi^2)$ and
    $M=m_{\rm UV}^2>0$. The supplied analytic input is the tadpole pole
    $I(1)=M/\epsilon+O(1)$ in the $i/(16\pi^2)$ loop measure.
    The auxiliary mass cancels from every pole. This calculation extracts UV
    poles; it does not give the finite off-shell amplitudes.
    """)


@app.cell
def _(E, Matrix, S, Symbol, TensorExpression, hep, json, mo, oneloop):
    _location = mo.notebook_location()
    if _location is None:
        raise RuntimeError("Cannot locate hep_sm.json")
    model = hep.Model(_location.parent / "hep_sm.json")
    D, epsilon, M, mUV, coordinate, integral, xi, s, Nf = S(
        "qed_ren::D",
        "qed_ren::eps",
        "qed_ren::M",
        "qed_ren::mUV",
        "qed_ren::d0",
        "qed_ren::I",
        "qed_ren::xi",
        "qed_ren::s",
        "qed_ren::Nf",
    )
    specification = json.loads(model.to_json())
    for propagator in specification["propagators"]:
        if propagator["particle"] == "a":
            propagator["numerator"] = (
                "-1𝑖*(UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))-(1-qed_ren::xi)*UFO::P(UFO::idx(1,1))*UFO::P(UFO::idx(1,2))/spenso::dot(UFO::P(spenso::mink(4)),UFO::P(spenso::mink(4))))"
            )
    model = hep.Model.from_json(json.dumps(specification))
    K, P, mass, charge = S("gammalooprs::K", "gammalooprs::P", "UFO::Me", "UFO::ee")
    mink, bis, gamma, metric = S(
        "spenso::mink", "spenso::bis", "spenso::gamma", "spenso::g"
    )
    index, dim, wave, mu, nu = S(
        "qed_ren::index_",
        "qed_ren::dim_",
        "qed_ren::wave_",
        "qed_ren::mu",
        "qed_ren::nu",
    )
    den, edge_, mom_, mass_, quad_ = S(
        "gammalooprs::denom",
        "qed_ren::edge_",
        "qed_ren::mom_",
        "qed_ren::mass_",
        "qed_ren::quad_",
    )
    ordering, value = S(
        "feynkit_generator_factor::ExternalFermionOrderingSign", "qed_ren::value_"
    )
    kinematics = hep.Kinematics(D, momenta=[K(0), P(0)]).with_scalar_product(
        P(0), P(0), s
    )
    vacuum = hep.Kinematics(D, momenta=[K(0)])
    family = hep.IntegralFamily(
        [K(0)], [], [vacuum.scalar_product(K(0), K(0)) - M], kinematics=vacuum
    )
    reducer = hep.TensorReducer(D).with_integrated_vector(K(0, mink(D)))
    gmunu = metric(mink(D, mu), mink(D, nu))
    ppmunu = P(0, mink(D, mu)) * P(0, mink(D, nu))
    zero, one = E("0"), E("1")
    parts, diagrams = {}, {}
    for kind, incoming, outgoing, loops in [
        ("tree", [11], [22, 11], 0),
        ("electron", [11], [11], 1),
        ("photon", [22], [22], 1),
        ("vertex", [11], [22, 11], 1),
    ]:
        generated = hep.Generator(model).generate(
            hep.Process.amplitude(incoming, outgoing).with_loop_count(loops, loops),
            max_vertices=len(incoming) + len(outgoing) - 2 + 2 * loops,
            maximum_bridges=0,
            vertex_allow=["V_98"],
            self_energy=None,
            tadpoles=None,
            zero_snails=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(generated.diagrams) == 1
        diagram = generated.diagrams[0]
        diagrams[kind] = diagram
        ports = {}
        for edge in diagram.external_edges:
            rep = (
                mink
                if kind == "photon"
                or (kind in ("tree", "vertex") and edge.external_index == 1)
                else bis
            )
            ports[edge.external_index] = dict(
                next(
                    diagram.projector_expression().match(
                        wave(edge.id, rep(4, index)), max_level=0
                    )
                )
            )[index]
        numerator = model.expand_couplings(
            diagram.numerator_expression().to_expression()
        )
        if loops:
            numerator = diagram.momentum_basis().route_expression(
                diagram.uv_expansion(mUV, numerator=numerator).to_expression()
            )
        numerator = (
            numerator.replace(mink(dim, index), mink(D, index))
            .replace(mink(dim), mink(D))
            .replace(mUV**2, M)
        )
        if kind == "photon":
            numerator = numerator.replace(mink(D, ports[0]), mink(D, mu)).replace(
                mink(D, ports[1]), mink(D, nu)
            )
        pattern = den(edge_, mom_, mass_, quad_)
        for match in list(numerator.match(pattern)):
            values = dict(match)
            formal = family.rewrite_numerator(values[quad_], [coordinate])
            assert formal == coordinate
            numerator = numerator.replace(
                den(values[edge_], values[mom_], values[mass_], values[quad_]), formal
            )
        factor = (
            diagram.overall_factor_expression(evaluate=True)
            * diagram.numerator_prefactor_expression()
        )
        if kind == "electron":
            raw = diagram.overall_factor_expression()
            removed = (raw / raw.replace(ordering(value), one)).replace(
                ordering(value), value
            )
            assert removed == -one
            factor /= removed
            probes = [
                (
                    "electron_p",
                    gamma(bis(4, ports[0]), bis(4, ports[1]), mink(D, mu))
                    * P(0, mink(D, mu))
                    / (4 * s),
                ),
                ("electron_m", metric(bis(4, ports[0]), bis(4, ports[1])) / (4 * mass)),
            ]
        elif kind in ("vertex", "tree"):
            probes = [
                (
                    kind,
                    gamma(bis(4, ports[0]), bis(4, ports[2]), mink(D, ports[1]))
                    / (4 * D),
                )
            ]
        else:
            probes = [(kind, one)]
        for label, projector in probes:
            trace = (
                TensorExpression((numerator * projector).expand())
                .simplify_gamma()
                .expand()
                .to_dots()
                .to_expression()
            )
            scalar = family.rewrite_numerator(
                kinematics.apply(reducer.reduce(trace)), [coordinate]
            )
            scalar = (scalar * factor).together().expand()
            terms = []
            for monomial, coefficient in scalar.coefficient_list(coordinate):
                power = -int(
                    (monomial.derivative(coordinate) * coordinate / monomial).together()
                )
                assert monomial == coordinate ** (-power)
                assert not coefficient.matches(K(S("qed_ren::args___")))
                terms.append(([power], coefficient))
            parts[label] = terms

    tree = parts.pop("tree")[0][1]
    assert tree == Symbol.I * charge
    solution = hep.IBPFamily(family, name="qed_one_loop").reduce_laporta(
        sorted({tuple(p) for terms in parts.values() for p, c in terms}), max_depth=2
    )
    assert solution.residuals == [[1]]
    assert abs(complex(oneloop.A0(1.0, 1.0)[1]) - 1) < 1e-12
    reduced, uv_poles = {}, {}
    for label, terms in parts.items():
        expression = sum(
            (c * solution.reduce(p, integral=integral) for p, c in terms), zero
        ).together()
        reduced[label] = expression
        normalized = (
            expression * (Symbol.I / tree if label == "vertex" else one) / charge**2
        )
        if label == "photon":
            normalized *= Nf
        pole = (
            normalized.replace(integral(1), M / epsilon)
            .replace(D, 4 - 2 * epsilon)
            .series(epsilon, 0, -1)
            .to_expression()
            .expand()
        )
        pole = pole.replace(mink(4, index), mink(D, index))
        uv_poles[label] = pole
        assert pole.derivative(M).expand() == zero
        assert pole.derivative(mass).expand() == zero
        assert pole.coefficient(epsilon**-2) == zero
    # The four references follow from independent UV power counting and the
    # published result, after extracting the stated tree/kernel normalization.
    expected = [
        xi / epsilon,
        -(xi + 3) / epsilon,
        -4 * Nf * (s * gmunu - ppmunu) / (3 * epsilon),
        xi / epsilon,
    ]
    for label, reference in zip(
        ("electron_p", "electron_m", "photon", "vertex"), expected, strict=True
    ):
        assert (uv_poles[label] - reference).together() == zero

    # Counterterm structures follow the local kinetic, mass and vertex operators.
    # M=mUV²; deltaZAm multiplies the additive auxiliary mass operator M*A²/2.
    # Solve for coefficients of Z=1+a4*deltaZ. No counterterm vertices are generated.
    photon_g = uv_poles["photon"].coefficient(gmunu)
    photon_pp = uv_poles["photon"].coefficient(ppmunu)
    # Unknown order: deltaZpsi, deltaZm, deltaZA, deltaZxi, deltaZe, deltaZAm.
    ct_matrix = Matrix.from_linear(
        6,
        6,
        [
            1,
            0,
            0,
            0,
            0,
            0,
            -1,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            -1,
            0,
            0,
            0,
            0,
            0,
            xi - 1,
            1,
            0,
            0,
            1,
            0,
            E("1/2"),
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
        ],
    )
    rhs = Matrix.vec(
        [
            -uv_poles["electron_p"],
            -uv_poles["electron_m"],
            -photon_g.coefficient(s),
            -xi * photon_pp,
            -uv_poles["vertex"],
            -photon_g.replace(s, zero) / M,
        ]
    )
    deltas = ct_matrix.solve(rhs)
    for row, reference in enumerate(
        [
            -xi / epsilon,
            -3 / epsilon,
            -4 * Nf / (3 * epsilon),
            -4 * Nf / (3 * epsilon),
            2 * Nf / (3 * epsilon),
            zero,
        ]
    ):
        assert (deltas[row, 0].to_expression() - reference).together() == zero
    assert (deltas[0, 0].to_expression() + uv_poles["vertex"]).together() == zero
    assert (
        deltas[4, 0].to_expression() + deltas[2, 0].to_expression() / 2
    ).together() == zero
    residual = ct_matrix * deltas - rhs
    assert all(residual[row, 0].to_expression().together() == zero for row in range(6))
    counterterm_names = ["Zψ", "Zm", "ZA", "Zξ", "Ze", "ZAm"]
    residues = [
        (epsilon * deltas[row, 0].to_expression()).together() for row in range(6)
    ]
    return (
        Nf,
        counterterm_names,
        ct_matrix,
        deltas,
        diagrams,
        epsilon,
        family,
        reduced,
        residues,
        solution,
        uv_poles,
        xi,
    )


@app.cell
def _(diagrams, family, mo, solution, uv_poles):
    mo.vstack(
        [
            mo.md("## Generated diagrams and UV poles"),
            mo.hstack([diagrams[_name] for _name in ("electron", "photon", "vertex")]),
            mo.md(
                r"The electron and photon poles below multiply $i a_4$; the vertex entry multiplies its generated tree vertex times $a_4$. Electron entries are the coefficients of $\not p$ and $m$, respectively."
            ),
            mo.vstack(
                [
                    mo.hstack([mo.md("**" + _name + "**"), _pole])
                    for _name, _pole in uv_poles.items()
                ]
            ),
            mo.md("## Shared vacuum family and IBP reduction"),
            family,
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "The powers I(1) through I(4) reduce to the tadpole I(1). Its analytic value is a supplied input; a finite-depth residual is not a general proof of a minimal master basis."
            ),
        ]
    )


@app.cell(hide_code=True)
def _(ct_matrix, deltas, mo):
    mo.vstack(
        [
            mo.md(r"""
        ## Counterterms from local operators

        Write $Z=1+a_4\delta Z$. The field and parameter conventions are
        $\psi_0=\sqrt{Z_\psi}\psi$, $m_0=Z_m m$,
        $A_0=\sqrt{Z_A}A$, $\xi_0=Z_\xi\xi$ and $e_0=Z_e e$.
        The six rows cancel the electron's kinetic and mass poles, the photon
        transverse and longitudinal poles, the vertex pole, and the additive
        auxiliary-mass operator $(Z_{Am}-1)MA^2/2$.

        These counterterm structures are specified from the Lagrangian;
        counterterm diagrams and subtraction forests are not generated here.
        Symbolica solves the displayed system in the column order
        $(\delta Z_\psi,\delta Z_m,\delta Z_A,\delta Z_\xi,\delta Z_e,\delta Z_{Am})$.
        """),
            ct_matrix,
            deltas,
            mo.md(r"""
        The resulting Ward identities are $Z_\xi=Z_A$ and
        $\delta Z_e+\delta Z_A/2=0$. The vertex pole cancels with
        $\delta Z_\psi$, even as $\xi$ changes. The mass and charge
        renormalizations are gauge independent.

        The photon-mass counterterm vanishes because the UV expansion retains
        the mass correction terms. This differs from a rearrangement that
        drops those terms and compensates with an auxiliary mass counterterm.
        Landau gauge is obtained by taking $\xi\to0$ after the symbolic solve.
        """),
        ]
    )


@app.cell
def _(mo):
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    flavor_count = mo.ui.slider(
        1, 6, step=1, value=1, label="Identical charged flavors Nf"
    )
    mo.hstack([gauge_parameter, flavor_count])
    return flavor_count, gauge_parameter


@app.cell
def _(Nf, counterterm_names, flavor_count, gauge_parameter, mo, residues, xi):
    selected_residues = [
        complex(
            _value.evaluate({xi: gauge_parameter.value, Nf: flavor_count.value})
        ).real
        for _value in residues
    ]
    ward_error = abs(selected_residues[4] + selected_residues[2] / 2)
    assert ward_error < 1e-12
    assert selected_residues[2] == selected_residues[3]
    mo.vstack(
        [
            mo.md(
                r"**Coefficients of $a_4/\epsilon$** — vary the gauge and flavor count."
            ),
            mo.ui.table(
                [
                    {"Constant": _name, "Pole coefficient": _value}
                    for _name, _value in zip(
                        counterterm_names, selected_residues, strict=True
                    )
                ],
                selection=None,
            ),
            mo.md(f"Ward identity residual: **{ward_error:.1e}**"),
        ]
    )
    return selected_residues, ward_error


if __name__ == "__main__":
    app.run()
