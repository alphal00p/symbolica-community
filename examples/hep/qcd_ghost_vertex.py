import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Ghost–gluon vertex and IBP")


@app.cell
def _():
    import json

    import marimo as mo
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.spenso import TensorExpression

    _set_namespace("ghost_uv")

    return E, S, Symbol, TensorExpression, hep, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ghost–gluon vertex and IBP

    [Browse notebooks](/) · [QCD renormalization](/?file=hep/qcd_renormalization.py) ·
    [Ghost subtraction at tree level](/?file=hep/qcd_ghosts.py)

    Generate both one-loop ghost–gluon vertex diagrams and the ghost self-energy,
    retaining a symbolic covariant-gauge parameter $\xi$. Repeat with an
    incoming antighost and check crossing with all external color and Lorentz
    indices open. This reproduces the UV result of the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/GhGl-Gh).

    Shared UV expansion, color algebra and tensor reduction produce raised
    powers of the single vacuum denominator $k^2-M$, where $M=m_{\rm UV}^2$.
    Native IBP reduces them to the tadpole. Its pole $I(1)=M/\epsilon+O(1)$
    is an explicit analytic input, in the $i/(16\pi^2)$ loop measure.
    We use $D=4-2\epsilon$ and $a_4=g_s^2/(16\pi^2)$.

    The model keeps the upstream UFO rule $P_2+P_3=-P_1$ with all vertex
    momenta incoming. The loop result is normalized against the generated
    tree tensor, including its sign and the ghost momentum convention.
    The two diagrams give $C_A\xi/(8\epsilon)$ and
    $3C_A\xi/(8\epsilon)$ times the tree and $a_4$.
    """)


@app.cell(hide_code=True)
def _(E, S, Symbol, TensorExpression, hep, json):
    model = hep.Model.standard_model()
    spec = json.loads(model.to_json())
    for propagator in spec["propagators"]:
        if propagator["particle"] == "g":
            propagator["numerator"] = (
                "-1𝑖*(UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
                "-(1-ghost_uv::xi)*UFO::P(UFO::idx(1,1))*UFO::P(UFO::idx(1,2))"
                "/spenso::dot(UFO::P(spenso::mink(4)),UFO::P(spenso::mink(4))))"
            )
    model = hep.Model.from_json(json.dumps(spec))
    D, eps, M, mUV, x, integral, xi, s, dA, CA = S(
        "D",
        "eps",
        "M",
        "mUV",
        "x",
        "I",
        "xi",
        "s",
        "dA",
        "CA",
    )
    K, P, gs = S("gammalooprs::K", "gammalooprs::P", "UFO::G")
    mink, coad, metric, hedge, f, cas = S(
        "spenso::mink",
        "spenso::coad",
        "spenso::g",
        "gammalooprs::hedge",
        "spenso::f",
        "spenso::cas",
    )
    idx, dim, a, b, c, arguments = S(
        "idx_",
        "dim_",
        "a",
        "b",
        "c",
        "arguments___",
    )
    den, ed, mom, ms, quad = S(
        "gammalooprs::denom",
        "ed_",
        "mom_",
        "ms_",
        "quad_",
    )
    zero, one = E("0"), E("1")
    kinematics = hep.Kinematics(D, momenta=[K(0), P(0), P(1)]).with_scalar_product(
        P(0), P(0), s
    )
    vacuum = hep.Kinematics(D, momenta=[K(0)])
    family = hep.IntegralFamily(
        [K(0)], [], [vacuum.scalar_product(K(0), K(0)) - M], kinematics=vacuum
    )
    reducer = hep.TensorReducer(D).with_integrated_vector(K(0, mink(D)))
    color = f(coad(dA, a), coad(dA, b), coad(dA, c))
    terms_by_diagram, trees, generated_diagrams = {}, {}, {}
    for particle_name in ("ghG", "ghG~"):
        for kind, outgoing, loops, count in (
            ("tree", ["g", particle_name], 0, 1),
            ("vertex", ["g", particle_name], 1, 2),
            ("self", [particle_name], 1, 1),
        ):
            diagrams = (
                hep.Generator(model)
                .generate(
                    hep.Process.amplitude([particle_name], outgoing).with_loop_count(
                        loops, loops
                    ),
                    max_vertices=len(outgoing) - 1 + 2 * loops,
                    maximum_bridges=0,
                    vertex_allow=["V_35", "V_36"],
                    self_energy=None,
                    tadpoles=None,
                    zero_snails=None,
                    numerator_grouping=None,
                    progress=None,
                )
                .diagrams
            )
            assert len(diagrams) == count, (particle_name, kind, len(diagrams))
            generated_diagrams[particle_name, kind] = diagrams
            for diagram in diagrams:
                topology = tuple(
                    sorted(vertex.interaction for vertex in diagram.vertices)
                )
                weight = (
                    diagram.overall_factor_expression(evaluate=True)
                    * diagram.numerator_prefactor_expression()
                )
                # Compare amputated ghost kernels in canonical field order.
                # Preserve all internal-loop and graph symmetry factors.
                _ordering, _value = S(
                    "feynkit_generator_factor::ExternalFermionOrderingSign",
                    "ordering_value_",
                )
                _raw = diagram.overall_factor_expression()
                _external_ordering = (
                    _raw / _raw.replace(_ordering(_value), one)
                ).replace(_ordering(_value), _value)
                weight /= _external_ordering
                assert weight == one
                # Keep every external Lorentz/color slot open; no polarization or
                # contraction with external momentum can hide an unwanted tensor.
                numerator = model.expand_couplings(
                    diagram.numerator_expression().to_expression()
                )
                for half in diagram.half_edges:
                    if half.edge.data.is_external:
                        numerator = numerator.replace(
                            hedge(half.data, 1),
                            [a, b, c][half.edge.data.external_index],
                        )
                numerator = numerator.replace(coad(8, idx), coad(dA, idx))
                if loops:
                    numerator = diagram.uv_expansion(
                        mUV, numerator=numerator
                    ).to_expression()
                numerator = diagram.momentum_basis().route_expression(numerator)
                numerator = (
                    numerator.replace(mink(dim, idx), mink(D, idx))
                    .replace(mink(dim), mink(D))
                    .replace(mUV**2, M)
                )
                for match in list(numerator.match(den(ed, mom, ms, quad))):
                    values = dict(match)
                    formal = family.rewrite_numerator(values[quad], [x])
                    assert formal == x
                    numerator = numerator.replace(
                        den(values[ed], values[mom], values[ms], values[quad]), formal
                    )
                tensor = (
                    TensorExpression(numerator.expand())
                    .simplify_color()
                    .simplify_metrics()
                    .to_dots()
                    .to_expression()
                    .replace(cas(2, coad(dA)), CA)
                )
                scalar = family.rewrite_numerator(
                    kinematics.apply(reducer.reduce(tensor)), [x]
                )
                scalar = (weight * scalar).together().expand()
                assert not scalar.matches(K(arguments))
                assert not scalar.matches(hedge(arguments))
                if not loops:
                    expected_momentum = P(0, mink(D, b))
                    if particle_name == "ghG~":
                        expected_momentum -= P(1, mink(D, b))
                    assert (scalar + gs * expected_momentum * color).expand() == zero
                    trees[particle_name] = scalar
                    continue
                terms = []
                for monomial, coefficient in scalar.coefficient_list(x):
                    power = -int((monomial.derivative(x) * x / monomial).together())
                    assert monomial == x**-power
                    terms.append(([power], coefficient))
                key = (particle_name, kind, topology)
                assert key not in terms_by_diagram
                terms_by_diagram[key] = terms

    # Crossing exchanges ghost color slots and sends its incoming momentum to
    # minus the outgoing antighost momentum; both sides retain the open gluon slot.
    crossed_tree = (
        trees["ghG"]
        .replace(P(0, mink(D, b)), -P(0, mink(D, b)) + P(1, mink(D, b)))
        .replace(color, f(coad(dA, c), coad(dA, b), coad(dA, a)))
    )
    assert (crossed_tree - trees["ghG~"]).expand() == zero

    targets = sorted(
        {tuple(p) for terms in terms_by_diagram.values() for p, _ in terms}
    )
    solution = hep.IBPFamily(family, name="qcd_ghost_vertex").reduce_laporta(
        [list(target) for target in targets], max_depth=2
    )
    assert solution.residuals == [[1]]
    vertex_poles = {"ghG": zero, "ghG~": zero}
    self_poles = {}
    for (particle_name, kind, topology), terms in terms_by_diagram.items():
        reduced = sum(
            (
                coefficient * solution.reduce(powers, integral=integral)
                for powers, coefficient in terms
            ),
            zero,
        ).together()
        pole = (
            (reduced / gs**2)
            .replace(integral(1), M / eps)
            .replace(D, 4 - 2 * eps)
            .series(eps, 0, -1)
            .to_expression()
            .expand()
        )
        assert pole.derivative(M).expand() == zero
        assert pole.derivative(mUV).expand() == zero
        assert pole.coefficient(eps**-2) == zero
        assert not pole.matches(integral(arguments))
        if kind == "self":
            self_poles[particle_name] = pole
            expected = CA * (xi - 3) * s * metric(coad(dA, a), coad(dA, b)) / (4 * eps)
            assert (pole - expected).expand() == zero
            assert pole.replace(s, 0) == zero  # No auxiliary ghost mass pole.
        else:
            assert topology in (("V_35", "V_35", "V_35"), ("V_35", "V_35", "V_36"))
            multiplicity = 3 if "V_36" in topology else 1
            expected_ratio = multiplicity * CA * xi / (8 * eps)
            tree = trees[particle_name].replace(D, 4)
            physical_pole = (Symbol.I * pole).expand()
            assert (physical_pole - tree * expected_ratio).expand() == zero
            ratio = (physical_pole / tree).together()
            assert ratio == expected_ratio
            vertex_poles[particle_name] += physical_pole
    for particle_name, pole in vertex_poles.items():
        assert (
            pole - trees[particle_name].replace(D, 4) * CA * xi / (2 * eps)
        ).expand() == zero

    # The gluon field counterterm is an explicit reference input from the separate
    # generated QCD renormalization calculation. The ghost kinetic pole is computed
    # here; combining it with the vertex supplies an independent coupling check.
    Nf = S("Nf")
    vertex_ratios = {
        particle_name: (pole / trees[particle_name].replace(D, 4)).together()
        for particle_name, pole in vertex_poles.items()
    }
    ghost_ct = {
        particle_name: -pole.coefficient(metric(coad(dA, a), coad(dA, b))).coefficient(
            s
        )
        for particle_name, pole in self_poles.items()
    }
    gluon_ct = ((13 - 3 * xi) * CA - 4 * Nf) / (6 * eps)
    coupling_ct = {
        particle_name: (
            -vertex_ratios[particle_name] - ghost_ct[particle_name] - gluon_ct / 2
        ).together()
        for particle_name in trees
    }
    for counterterm in coupling_ct.values():
        assert (counterterm + (11 * CA - 2 * Nf) / (6 * eps)).together() == zero
        assert counterterm.derivative(xi).expand() == zero
    return (
        CA,
        Nf,
        coupling_ct,
        eps,
        family,
        generated_diagrams,
        ghost_ct,
        gluon_ct,
        integral,
        self_poles,
        solution,
        trees,
        vertex_poles,
        vertex_ratios,
        xi,
    )


@app.cell
def _(mo):
    antighost = mo.ui.checkbox(value=False, label="Incoming antighost")
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    color_count = mo.ui.slider(2, 5, step=1, value=3, label="Colors Nc")
    flavor_count = mo.ui.slider(0, 20, step=1, value=5, label="Quark flavors Nf")
    mo.vstack(
        [
            mo.hstack([antighost, gauge_parameter]),
            mo.hstack([color_count, flavor_count]),
        ]
    )
    return antighost, color_count, flavor_count, gauge_parameter


@app.cell
def _(antighost, generated_diagrams, mo, trees, vertex_poles):
    selected_particle = "ghG~" if antighost.value else "ghG"
    mo.vstack(
        [
            mo.md("## Generated tree and loop tensors"),
            mo.hstack(generated_diagrams[selected_particle, "tree"]),
            trees[selected_particle],
            mo.hstack(generated_diagrams[selected_particle, "vertex"]),
            vertex_poles[selected_particle],
            mo.md(
                r"The displayed loop pole multiplies $a_4$. Its entire tensor is proportional to the tree: no contraction with a polarization or momentum is used to hide extra structures."
            ),
        ]
    )
    return (selected_particle,)


@app.cell
def _(family, integral, mo, solution):
    mo.vstack(
        [
            mo.md("## Native vacuum IBP"),
            family,
            mo.hstack([mo.md("**I(2)**"), solution.reduce([2], integral=integral)]),
            mo.hstack([mo.md("**I(3)**"), solution.reduce([3], integral=integral)]),
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "Both powers reduce to I(1). The calculation checks that the UV poles have no auxiliary mass dependence, residual loop momentum, or double poles."
            ),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A coupling counterterm from the ghost vertex

    Write $Z=1+a_4\delta Z$. The generated self-energy determines
    $\delta Z_c=C_A(3-\xi)/(4\epsilon)$.
    The ghost–gluon counterterm is
    $\delta Z_c+\delta Z_g+\delta Z_A/2$ times its tree tensor.

    Here the gluon field counterterm
    $\delta Z_A=[C_A(13-3\xi)-4N_f]/(6\epsilon)$ is supplied from the
    separate QCD renormalization example. Combining it with the calculated
    ghost and vertex poles gives
    $\delta Z_g=-(11C_A-2N_f)/(6\epsilon)$, independently of $\xi$.
    No counterterm diagrams are generated here.

    In Landau gauge ($\xi=0$), the vertex pole vanishes. Vary the gauge to
    check its cancellation against the three counterterms.
    """)


@app.cell
def _(
    CA,
    Nf,
    color_count,
    coupling_ct,
    eps,
    flavor_count,
    gauge_parameter,
    ghost_ct,
    gluon_ct,
    mo,
    selected_particle,
    vertex_ratios,
    xi,
):
    _parameters = {
        CA: color_count.value,
        Nf: flavor_count.value,
        xi: gauge_parameter.value,
    }
    _expressions = [
        vertex_ratios[selected_particle],
        ghost_ct[selected_particle],
        gluon_ct,
        coupling_ct[selected_particle],
    ]
    selected_residues = [
        complex((eps * _expression).together().evaluate(_parameters)).real
        for _expression in _expressions
    ]
    cancellation_error = abs(
        selected_residues[0]
        + selected_residues[1]
        + selected_residues[2] / 2
        + selected_residues[3]
    )
    assert cancellation_error < 1e-12
    mo.vstack(
        [
            mo.md(r"**Coefficients of $a_4/\epsilon$**"),
            mo.ui.table(
                [
                    {"Quantity": _name, "Pole coefficient": _value}
                    for _name, _value in zip(
                        ["Vertex / tree", "Zc", "ZA (input)", "Zg"],
                        selected_residues,
                        strict=True,
                    )
                ],
                selection=None,
            ),
            mo.md(
                f"Ghost–gluon counterterm cancellation residual: **{cancellation_error:.1e}**"
            ),
        ]
    )
    return cancellation_error, selected_residues


if __name__ == "__main__":
    app.run()
