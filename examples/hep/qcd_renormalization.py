import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="One-loop QCD renormalization")


@app.cell
def _():
    import json

    import marimo as mo
    from symbolica import E, Matrix, S, Symbol
    from symbolica.community import hep
    from symbolica.community.hep import oneloop
    from symbolica.community.spenso import (
        ColorCasimirSettings,
        Representation,
        TensorExpression,
    )

    return (
        ColorCasimirSettings,
        E,
        Matrix,
        Representation,
        S,
        Symbol,
        TensorExpression,
        hep,
        json,
        mo,
        oneloop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop QCD renormalization

    [Browse all notebooks](/) · [QED renormalization](/?file=hep/qed_renormalization.py) ·
    [Two-loop electron](/?file=hep/electron_two_loop.py)

    Generate the quark, ghost and gluon self-energies and the quark–gluon
    vertex in a **symbolic covariant gauge**. Shared color contractions,
    Dirac traces, UV expansion and native IBP give the six physical
    renormalization constants in the
    [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/Renormalization).
    The quark mass remains symbolic throughout.

    We use $D=4-2\epsilon$, $a_4=g_s^2/(16\pi^2)$,
    $T_R=1/2$ and the gluon numerator
    $-i[g^{\mu\nu}-(1-\xi)q^\mu q^\nu/q^2]$.
    The auxiliary squared mass is $M=m_{\rm UV}^2>0$.
    The analytic tadpole pole $I(1)=M/\epsilon+O(1)$ is a supplied input,
    checked with OneLOop. IBP determines its rational coefficients.

    The full UV expansion retains the mass correction terms. They cancel
    auxiliary mass poles, so our two auxiliary mass counterterms vanish.
    This differs from the reference's selective infrared rearrangement,
    which needs a nonzero gluon auxiliary mass counterterm. The six physical
    constants agree. Finite amplitudes and automatic counterterm diagrams
    are outside this calculation.
    """)


@app.cell(hide_code=True)
def _(
    ColorCasimirSettings,
    E,
    Matrix,
    Representation,
    S,
    Symbol,
    TensorExpression,
    hep,
    json,
    mo,
    oneloop,
):
    _location = mo.notebook_location()
    if _location is None:
        raise RuntimeError("Cannot locate hep_sm.json")
    model = hep.Model(_location.parent / "hep_sm.json")
    D, epsilon, M, mUV, coordinate, integral, xi, s, Nf = S(
        "qcd_ren::D",
        "qcd_ren::eps",
        "qcd_ren::M",
        "qcd_ren::mUV",
        "qcd_ren::d0",
        "qcd_ren::I",
        "qcd_ren::xi",
        "qcd_ren::s",
        "qcd_ren::Nf",
    )
    Nc, dA, CF, CA = S("qcd_ren::Nc", "qcd_ren::dA", "qcd_ren::CF", "qcd_ren::CA")
    specification = json.loads(model.to_json())
    for propagator in specification["propagators"]:
        if propagator["particle"] == "g":
            propagator["numerator"] = (
                "-1𝑖*(UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
                "-(1-qcd_ren::xi)*UFO::P(UFO::idx(1,1))*UFO::P(UFO::idx(1,2))"
                "/spenso::dot(UFO::P(spenso::mink(4)),UFO::P(spenso::mink(4))))"
            )
    model = hep.Model.from_json(json.dumps(specification))
    K, P, mass, gs = S("gammalooprs::K", "gammalooprs::P", "UFO::MB", "UFO::G")
    mink, bis, gamma, metric = S(
        "spenso::mink", "spenso::bis", "spenso::gamma", "spenso::g"
    )
    cof, coad, cas, trace_index = S(
        "spenso::cof", "spenso::coad", "spenso::cas", "spenso::idx"
    )
    index, dim, wave, mu, nu, left, right = S(
        "qcd_ren::index_",
        "qcd_ren::dim_",
        "qcd_ren::wave_",
        "qcd_ren::mu",
        "qcd_ren::nu",
        "qcd_ren::left_",
        "qcd_ren::right_",
    )
    den, edge_, mom_, mass_, quad_ = S(
        "gammalooprs::denom",
        "qcd_ren::edge_",
        "qcd_ren::mom_",
        "qcd_ren::mass_",
        "qcd_ren::quad_",
    )
    ordering, value, arguments = S(
        "feynkit_generator_factor::ExternalFermionOrderingSign",
        "qcd_ren::value_",
        "qcd_ren::arguments___",
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
    for kind, incoming, outgoing, loops, vertices, count in [
        ("tree", [5], [21, 5], 0, ["V_76"], 1),
        ("quark", [5], [5], 1, ["V_76"], 1),
        ("ghost", [9000005], [9000005], 1, ["V_35"], 1),
        ("gluon_loop", [21], [21], 1, ["V_36"], 1),
        ("ghost_loop", [21], [21], 1, ["V_35"], 1),
        ("quark_loop", [21], [21], 1, ["V_76"], 1),
        ("tadpole", [21], [21], 1, ["V_37"], 1),
        ("vertex", [5], [21, 5], 1, ["V_76", "V_36"], 2),
    ]:
        generated = hep.Generator(model).generate(
            hep.Process.amplitude(incoming, outgoing).with_loop_count(loops, loops),
            max_vertices=len(incoming) + len(outgoing) - 2 + 2 * loops,
            maximum_bridges=0,
            vertex_allow=vertices,
            self_energy=None,
            tadpoles=None,
            zero_snails=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(generated.diagrams) == count
        diagrams[kind] = generated.diagrams
        for number, diagram in enumerate(generated.diagrams):
            ports = {}
            if kind != "ghost":
                for edge in diagram.external_edges:
                    is_gluon = incoming == [21] or (
                        kind in ("tree", "vertex") and edge.external_index == 1
                    )
                    rep = mink if is_gluon else bis
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
            if kind in ("tree", "vertex"):
                # Conjugate the tree's color tensor; the generated tree fixes the norm.
                color_projector = (
                    TensorExpression.t(dA, Nc)(ports[1], ports[0], ports[2])
                    .spenso_conjugate()
                    .to_expression()
                )
                numerator = (
                    numerator.replace(cof(3, index), cof(Nc, index)).replace(
                        coad(8, index), coad(dA, index)
                    )
                    * color_projector
                )
                settings = ColorCasimirSettings(rewrite_fundamental_dimension=False)
            else:
                particle = model.particle_by_pdg(incoming[0])
                rep, numeric_dim, symbolic_dim = (
                    (cof, 3, Nc) if kind == "quark" else (coad, 8, dA)
                )
                color_slots = [
                    slot.dual().to_expression()
                    for slot in TensorExpression(numerator).interface
                    if slot.to_expression().matches(rep(numeric_dim, index))
                ]
                assert len(color_slots) == 2
                color_indices = dict(
                    next(
                        metric(*color_slots).match(
                            particle.color_sum(left, right), max_level=0
                        )
                    )
                )
                color_projector = particle.color_sum(
                    color_indices[left], color_indices[right]
                )
                numerator = (
                    (numerator * color_projector / symbolic_dim)
                    .replace(cof(3, index), cof(Nc, index))
                    .replace(coad(8, index), coad(dA, index))
                )
                settings = ColorCasimirSettings()
            numerator = (
                TensorExpression(numerator)
                .simplify_color()
                .to_color_casimir(
                    fundamental=Representation.cof(Nc),
                    adjoint=Representation.coad(dA),
                    settings=settings,
                )
                .to_expression()
            )
            # CF and CA are display names for shared representation-aware invariants;
            # the quark-loop trace uses the conventional fundamental index TR=1/2.
            numerator = (
                numerator.replace(cas(2, cof(Nc)), CF)
                .replace(cas(2, coad(dA)), CA)
                .replace(trace_index(2, cof(Nc)), one / 2)
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
            if incoming == [21]:
                numerator = numerator.replace(mink(D, ports[0]), mink(D, mu)).replace(
                    mink(D, ports[1]), mink(D, nu)
                )
            pattern = den(edge_, mom_, mass_, quad_)
            for match in list(numerator.match(pattern)):
                values = dict(match)
                formal = family.rewrite_numerator(values[quad_], [coordinate])
                assert formal == coordinate
                numerator = numerator.replace(
                    den(values[edge_], values[mom_], values[mass_], values[quad_]),
                    formal,
                )
            factor = (
                diagram.overall_factor_expression(evaluate=True)
                * diagram.numerator_prefactor_expression()
            )
            if kind == "quark":
                # An amputated two-point kernel omits only external fermion ordering.
                raw = diagram.overall_factor_expression()
                removed = (raw / raw.replace(ordering(value), one)).replace(
                    ordering(value), value
                )
                assert removed == -one
                factor /= removed
                probes = [
                    (
                        "quark_p",
                        gamma(bis(4, ports[0]), bis(4, ports[1]), mink(D, mu))
                        * P(0, mink(D, mu))
                        / (4 * s),
                    ),
                    (
                        "quark_m",
                        metric(bis(4, ports[0]), bis(4, ports[1])) / (4 * mass),
                    ),
                ]
            elif kind in ("tree", "vertex"):
                probes = [
                    (
                        f"{kind}_{number}",
                        gamma(bis(4, ports[0]), bis(4, ports[2]), mink(D, ports[1]))
                        / (4 * D),
                    )
                ]
            else:
                probes = [(kind, one)]
            for label, projector in probes:
                traced = (
                    TensorExpression((numerator * projector).expand())
                    .simplify_gamma()
                    .expand()
                    .simplify_metrics()
                    .to_dots()
                    .to_expression()
                )
                scalar = family.rewrite_numerator(
                    kinematics.apply(reducer.reduce(traced)), [coordinate]
                )
                scalar = (scalar * factor).together().expand()
                terms = []
                for monomial, coefficient in scalar.coefficient_list(coordinate):
                    power = -int(
                        (
                            monomial.derivative(coordinate) * coordinate / monomial
                        ).together()
                    )
                    assert monomial == coordinate ** (-power)
                    assert not coefficient.matches(K(arguments))
                    terms.append(([power], coefficient))
                parts[label] = terms

    tree_terms = parts.pop("tree_0")
    assert len(tree_terms) == 1 and tree_terms[0][0] == [0]
    tree = tree_terms[0][1]
    assert (tree + Symbol.I * gs * Nc * CF).together() == zero
    solution = hep.IBPFamily(family, name="qcd_one_loop").reduce_laporta(
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
            expression
            * (Symbol.I / tree if label.startswith("vertex") else one)
            / gs**2
        )
        if label == "quark_loop":
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
    uv_poles["gluon"] = sum(
        (
            uv_poles[label]
            for label in ("gluon_loop", "ghost_loop", "quark_loop", "tadpole")
        ),
        zero,
    ).expand()
    uv_poles["vertex"] = (uv_poles["vertex_0"] + uv_poles["vertex_1"]).expand()
    expected = [
        CF * xi / epsilon,
        -CF * (xi + 3) / epsilon,
        CA * (xi - 3) * s / (4 * epsilon),
        ((13 - 3 * xi) * CA - 4 * Nf) * (s * gmunu - ppmunu) / (6 * epsilon),
        (CF * xi + CA * (xi + 3) / 4) / epsilon,
    ]
    for label, reference in zip(
        ("quark_p", "quark_m", "ghost", "gluon", "vertex"), expected, strict=True
    ):
        assert (uv_poles[label] - reference).together() == zero

    # Local counterterm operators, with Z=1+a4*deltaZ, a4=gs²/(16*pi²).
    # Unknowns: Zq, Zm, ZA, Zxi, Zc, Zg, ZAm, Zcm. Auxiliary mass operators are
    # (ZAm-1)*M*A²/2 and (Zcm-1)*M*cbar*c. No CT diagrams are generated here.
    gluon_g = uv_poles["gluon"].coefficient(gmunu)
    gluon_pp = uv_poles["gluon"].coefficient(ppmunu)
    ct_matrix = Matrix.from_linear(
        8,
        8,
        [
            1,
            0,
            0,
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
            0,
            0,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            xi - 1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            E("1/2"),
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
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
            -uv_poles["quark_p"],
            -uv_poles["quark_m"],
            -gluon_g.coefficient(s),
            -xi * gluon_pp,
            -uv_poles["ghost"].coefficient(s),
            -uv_poles["vertex"],
            -gluon_g.replace(s, zero) / M,
            -uv_poles["ghost"].replace(s, zero) / M,
        ]
    )
    deltas = ct_matrix.solve(rhs)
    references = [
        -CF * xi / epsilon,
        -3 * CF / epsilon,
        ((13 - 3 * xi) * CA - 4 * Nf) / (6 * epsilon),
        ((13 - 3 * xi) * CA - 4 * Nf) / (6 * epsilon),
        CA * (3 - xi) / (4 * epsilon),
        -(11 * CA - 2 * Nf) / (6 * epsilon),
        zero,
        zero,
    ]
    for row, reference in enumerate(references):
        assert (deltas[row, 0].to_expression() - reference).together() == zero
    residual = ct_matrix * deltas - rhs
    assert all(residual[row, 0].to_expression().together() == zero for row in range(8))
    assert deltas[1, 0].to_expression().derivative(xi).expand() == zero
    assert deltas[5, 0].to_expression().derivative(xi).expand() == zero
    counterterm_names = ["Zq", "Zm", "ZA", "Zξ", "Zc", "Zg", "ZAm", "Zcm"]
    residues = [
        (epsilon * deltas[row, 0].to_expression()).together() for row in range(8)
    ]
    vertex_residue = (epsilon * uv_poles["vertex"]).together()
    return (
        CA,
        CF,
        Nf,
        counterterm_names,
        ct_matrix,
        deltas,
        diagrams,
        family,
        integral,
        reduced,
        residues,
        solution,
        uv_poles,
        vertex_residue,
        xi,
    )


@app.cell
def _(diagrams, mo, uv_poles):
    mo.vstack(
        [
            mo.md("## Generated diagrams and UV poles"),
            mo.ui.tabs(
                {
                    "Quark": mo.vstack(
                        [
                            mo.hstack(diagrams["quark"]),
                            uv_poles["quark_p"],
                            uv_poles["quark_m"],
                        ]
                    ),
                    "Ghost": mo.vstack(
                        [mo.hstack(diagrams["ghost"]), uv_poles["ghost"]]
                    ),
                    "Gluon": mo.vstack(
                        [
                            mo.hstack(diagrams["gluon_loop"] + diagrams["ghost_loop"]),
                            mo.hstack(diagrams["quark_loop"] + diagrams["tadpole"]),
                            uv_poles["gluon"],
                        ]
                    ),
                    "Quark–gluon vertex": mo.vstack(
                        [mo.hstack(diagrams["vertex"]), uv_poles["vertex"]]
                    ),
                }
            ),
            mo.md(
                r"The self-energies multiply $i a_4$ and the external color identity. Quark entries are coefficients of $\not p$ and $m$. The vertex entry multiplies its generated tree value times $a_4$. Both vertex diagrams contribute, and the gluon pole becomes transverse only after summing its loop classes."
            ),
        ]
    )


@app.cell
def _(family, integral, mo, solution):
    mo.vstack(
        [
            mo.md("## One shared vacuum family"),
            family,
            mo.md(
                r"The longitudinal gluon denominators cancel after tensor projection. All scalar terms belong to this family; the calculation rejects hidden loop momenta in their coefficients. Native IBP reduces $I(1),\ldots,I(4)$ to $I(1)$:"
            ),
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.md(f"**I({_power})**"),
                            solution.reduce([_power], integral=integral),
                        ]
                    )
                    for _power in range(1, 5)
                ]
            ),
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "The tadpole pole is an analytic input. A finite-depth IBP residual alone does not certify a minimal master basis."
            ),
        ]
    )


@app.cell(hide_code=True)
def _(ct_matrix, deltas, mo):
    mo.vstack(
        [
            mo.md(r"""
        ## Match the local counterterms

        Write $Z=1+a_4\delta Z$, with
        $q_0=\sqrt{Z_q}q$, $m_0=Z_m m$, $A_0=\sqrt{Z_A}A$,
        $\xi_0=Z_\xi\xi$, $c_0=\sqrt{Z_c}c$ and $g_{s,0}=Z_g g_s$.
        The quark–gluon counterterm is
        $\delta Z_q+\delta Z_g+\delta Z_A/2$ times the tree vertex.

        Symbolica solves eight equations. The coefficient matrix and solution
        below use the column order
        $(\delta Z_q,\delta Z_m,\delta Z_A,\delta Z_\xi,
        \delta Z_c,\delta Z_g,\delta Z_{Am},\delta Z_{cm})$.
        The last two constants multiply the additive auxiliary operators
        $a_4 M A^2/2$ and $a_4 M\bar c c$.
        These local operators are stated inputs, not generated counterterm diagrams.
        """),
            ct_matrix,
            deltas,
            mo.md(r"""
        The result obeys $Z_\xi=Z_A$. Both mass and coupling renormalization
        are gauge independent. The coupling pole determines
        $\beta_0=(11 C_A-2N_f)/3$ in
        $\mu\,d g_s/d\mu=-\beta_0 g_s^3/(16\pi^2)+O(g_s^5)$.
        The displayed vertex cancellation is checked using its calculated pole.
        """),
        ]
    )


@app.cell
def _(mo):
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    flavor_count = mo.ui.slider(0, 20, step=1, value=5, label="Quark flavors Nf")
    color_count = mo.ui.slider(2, 5, step=1, value=3, label="Colors Nc")
    mo.hstack([gauge_parameter, flavor_count, color_count])
    return color_count, flavor_count, gauge_parameter


@app.cell
def _(
    CA,
    CF,
    Nf,
    color_count,
    counterterm_names,
    flavor_count,
    gauge_parameter,
    mo,
    residues,
    vertex_residue,
    xi,
):
    _n = color_count.value
    _parameters = {
        CA: _n,
        CF: (_n**2 - 1) / (2 * _n),
        Nf: flavor_count.value,
        xi: gauge_parameter.value,
    }
    selected_residues = [
        complex(_value.evaluate(_parameters)).real for _value in residues
    ]
    _vertex = complex(vertex_residue.evaluate(_parameters)).real
    vertex_error = abs(
        selected_residues[0] + selected_residues[5] + selected_residues[2] / 2 + _vertex
    )
    beta0 = -2 * selected_residues[5]
    assert vertex_error < 1e-12
    assert selected_residues[2] == selected_residues[3]
    assert abs(beta0 - (11 * _n - 2 * flavor_count.value) / 3) < 1e-12
    mo.vstack(
        [
            mo.md(
                r"**Coefficients of $a_4/\epsilon$** — vary the gauge, flavor count and SU(Nc) color group."
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
            mo.md(
                f"Quark–gluon vertex cancellation residual: **{vertex_error:.1e}**.  "
                + rf"One-loop $\beta_0$: **{beta0:.6g}**."
            ),
        ]
    )
    return beta0, selected_residues, vertex_error


if __name__ == "__main__":
    app.run()
