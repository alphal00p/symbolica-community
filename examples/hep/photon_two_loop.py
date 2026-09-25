import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Two-loop photon renormalization and IBP",
)


@app.cell
def _():
    import json
    import math

    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import (
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        TensorReducer,
    )
    from symbolica.community.spenso import TensorExpression

    _set_namespace("photon_2l")
    return (
        E,
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        Replacement,
        S,
        TensorExpression,
        TensorReducer,
        json,
        math,
        mo,
    )


@app.cell
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-loop photon renormalization and IBP

    [Browse notebooks](/) · [One-loop QED](/?file=hep/qed_renormalization.py) ·
    [Two-loop electron](/?file=hep/electron_two_loop.py)

    Generate all three massless QED photon self-energy diagrams in a
    **symbolic covariant gauge**, retain both open Lorentz indices, and
    reduce their UV expansion with native IBP. Calculate the four one-loop
    counterterm insertions on a generated photon bubble, then determine
    the two-loop field and auxiliary-mass counterterms.

    This reproduces the complete tensor poles and renormalization constants
    of the [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QED/TwoLoops/Renormalization-GaGa).
    We use $D=4-2\epsilon$, $a_4=e^2/(16\pi^2)$ and
    $T^{\mu\nu}=p^2g^{\mu\nu}-p^\mu p^\nu$.
    Both tensor coefficients are checked independently. Adjust the gauge,
    flavor count and auxiliary mass below to see the cancellation.
    """)
    return


@app.cell(hide_code=True)
def _(
    Symbols,
    E,
    IBPFamily,
    IntegralFamily,
    Kinematics,
    Model,
    Replacement,
    S,
    TensorExpression,
    TensorReducer,
    json,
):
    model = Model.standard_model()
    d, s, mass_squared, uv_mass, eps, log_mass, log_4pi, flavors = S(
        "D",
        "s",
        "M",
        "mUV",
        "eps",
        "Lm",
        "L4pi",
        "Nf",
    )
    k, p = Symbols.loop_momentum, Symbols.external_momentum
    mass = model.particle("e-").mass_expression
    charge = model.parameter("ee").symbol
    mink, wave = (Symbols.lorentz, S("wave_"))
    index, dimension, mu = S("index_", "dimension_", "mu")
    closed_loop, arguments = S(
        "feynkit_generator_factor::InternalFermionLoopSign",
        "arguments___",
    )
    denominator, edge_pattern, momentum_pattern, mass_pattern, quadratic_pattern = (
        Symbols.denominator,
        S("edge_"),
        S("momentum_"),
        S("mass_"),
        S("quadratic_"),
    )
    coordinates = list(S("d0", "d1", "d2"))
    tadpole_squared, sunset, integral = S("T2", "V", "I")
    zero, one, imaginary = E("0"), E("1"), E("1i")
    kinematics = Kinematics(d, momenta=[k(0), k(1), p(0)]).with_scalar_product(
        p(0), p(0), s
    )
    vacuum_kinematics = Kinematics(d, momenta=[k(0), k(1)])
    family = IntegralFamily(
        [k(0), k(1)],
        [],
        [
            vacuum_kinematics.scalar_product(q, q) - mass_squared
            for q in (k(0), k(1), k(0) - k(1))
        ],
        kinematics=vacuum_kinematics,
    )
    reducer = (
        TensorReducer(d)
        .with_integrated_vector(k(0, mink(d)))
        .with_integrated_vector(k(1, mink(d)))
    )

    xi, nu = S("xi", "nu")
    metric = Symbols.metric
    specification = json.loads(model.to_json())
    for particle in specification["particles"]:
        if particle["name"] in ("e-", "e+"):
            particle["mass"] = "ZERO"
    for propagator in specification["propagators"]:
        if propagator["particle"] == "a":
            propagator["numerator"] = (
                E("-1𝑖")
                * (
                    Symbols.ufo_metric(Symbols.ufo_index(1, 1), Symbols.ufo_index(1, 2))
                    - (1 - S("xi"))
                    * Symbols.ufo_momentum(Symbols.ufo_index(1, 1))
                    * Symbols.ufo_momentum(Symbols.ufo_index(1, 2))
                    / Symbols.dot(
                        Symbols.ufo_momentum(Symbols.lorentz(4)),
                        Symbols.ufo_momentum(Symbols.lorentz(4)),
                    )
                )
            ).format_plain()
    model = Model.from_json(json.dumps(specification))
    gmunu = metric(mink(d, mu), mink(d, nu))
    ppmunu = p(0, mink(d, mu)) * p(0, mink(d, nu))
    Qg, Qpp = S("metric_basis", "momentum_basis")
    Q, dot = (Symbols.edge_momentum, Symbols.dot)
    result = model.process(["a"], ["a"], vertex_allow=["V_98"]).generate_diagrams(
        loops=2,
        max_vertices=4,
        maximum_bridges=0,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(result.diagrams) == 3
    targets, diagram_integrals = set(), []
    for diagram in result.diagrams:
        raw_factor = diagram.overall_factor_expression()
        assert raw_factor.matches(closed_loop(-one))
        factor = diagram.overall_factor_expression(evaluate=True)
        numerator = model.expand_couplings(
            diagram.numerator_expression().to_expression()
        ).replace(mass, zero)
        photon = next(
            edge for edge in diagram.internal_edges if edge.particle_name == "a"
        )
        # Selective mass replacement acts on the two propagator terms separately.
        # Clearing q² across the Feynman term would instead introduce an extra
        # M/(q²-M)² contribution and change this infrared-rearrangement prescription.
        feynman = numerator.replace(xi, one)
        longitudinal = (
            ((numerator - feynman) * dot(Q(photon.id, mink(4)), Q(photon.id, mink(4))))
            .together()
            .expand()
        )
        uv = diagram.uv_expansion(uv_mass, numerator=feynman).to_expression()
        uv += diagram.uv_expansion(
            uv_mass, numerator=longitudinal, edge_powers={photon.id: 2}
        ).to_expression()
        uv = diagram.momentum_basis().route_expression(uv)
        uv = uv.replace(mink(dimension, index), mink(d, index)).replace(
            mink(dimension), mink(d)
        )
        ports = {
            edge.external_index: dict(
                next(
                    diagram.projector_expression().match(
                        wave(edge.id, mink(4, index)), max_level=0
                    )
                )
            )[index]
            for edge in diagram.external_edges
        }
        uv = uv.replace(mink(d, ports[0]), mink(d, mu)).replace(
            mink(d, ports[1]), mink(d, nu)
        )
        pattern = denominator(
            edge_pattern, momentum_pattern, mass_pattern, quadratic_pattern
        )
        quadratics = {
            dict(match)[quadratic_pattern].replace(uv_mass**2, mass_squared)
            for match in uv.match(pattern)
        }
        source_family = IntegralFamily(
            [k(0), k(1)], [], sorted(quadratics, key=str), kinematics=vacuum_kinematics
        )
        mapping = source_family.find_mapping(family)
        assert mapping is not None
        # Freeze the massive denominators before removing explicit mUV terms.
        # Selecting mUV^0 discards the compensation terms in the shared full UV
        # expansion while retaining M in the integral family.  This reproduces the
        # reference's selective auxiliary-mass replacement and momentum Taylor series;
        # it does not change the default full-UV prescription used by other notebooks.
        # Keep inverse denominators opaque during the polynomial tensor projection.
        for match in list(uv.match(pattern)):
            values = dict(match)
            formal = source_family.rewrite_numerator(
                values[quadratic_pattern].replace(uv_mass**2, mass_squared), coordinates
            )
            assert formal in coordinates
            uv = uv.replace(
                denominator(
                    values[edge_pattern],
                    values[momentum_pattern],
                    values[mass_pattern],
                    values[quadratic_pattern],
                ),
                formal,
            )
        assert not uv.matches(pattern)
        for monomial, _ in uv.expand().coefficient_list(uv_mass):
            power = int((monomial.derivative(uv_mass) * uv_mass / monomial).together())
            assert power >= 0 and monomial == uv_mass**power
        uv = uv.replace(uv_mass, zero)
        traced = (
            TensorExpression(uv.expand())
            .simplify_gamma()
            .expand()
            .to_dots()
            .to_expression()
        )
        scalar = source_family.rewrite_numerator(
            kinematics.apply(reducer.reduce(traced)), coordinates
        )
        scalar *= factor * diagram.numerator_prefactor_expression()
        # Scalar basis labels avoid differentiating symbolic-D tensor slots during
        # the epsilon expansion.  The complete open tensor is checked before this.
        cg = scalar.expand().coefficient(gmunu)
        cp = scalar.expand().coefficient(ppmunu)
        assert (scalar - cg * gmunu - cp * ppmunu).expand() == zero
        scalar = cg * Qg + cp * Qpp
        terms = []
        for monomial, coefficient in scalar.expand().coefficient_list(*coordinates):
            powers = [
                -int((monomial.derivative(den) * den / monomial).together())
                for den in coordinates
            ]
            assert (
                monomial
                == coordinates[0] ** -powers[0]
                * coordinates[1] ** -powers[1]
                * coordinates[2] ** -powers[2]
            )
            assert all(
                coefficient.derivative(x).expand() == zero for x in coordinates
            ), coefficient
            assert not coefficient.matches(k(arguments))
            assert not coefficient.matches(p(arguments))
            assert not coefficient.matches(Q(arguments))
            powers = mapping.map_powers(powers)
            targets.add(tuple(powers))
            terms.append((powers, coefficient))
        diagram_integrals.append(terms)
    solution = IBPFamily(family, name="photon_two_loop").reduce_laporta(
        [list(target) for target in sorted(targets)], max_depth=2
    )
    assert solution.stats["rows"] > 0
    assert {tuple(powers) for powers in solution.residuals} == {
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    }
    # Verified unit-Jacobian maps relate the three disconnected tadpole products.
    for images in ([k(1), k(0)], [k(0), k(0) - k(1)]):
        assert family.mapping_to(family, images) is not None
    basis = {
        (0, 1, 1): tadpole_squared,
        (1, 0, 1): tadpole_squared,
        (1, 1, 0): tadpole_squared,
        (1, 1, 1): sunset,
    }
    basis_rules = [
        Replacement(integral(*powers), master) for powers, master in basis.items()
    ]
    # Explicit analytic inputs, not values inferred or certified by the IBP solver.
    master_poles = [
        Replacement(
            tadpole_squared, mass_squared**2 * (1 / eps**2 + 2 * (1 - log_mass) / eps)
        ),
        Replacement(
            sunset, mass_squared * (E("3/2") / eps**2 + (E("9/2") - 3 * log_mass) / eps)
        ),
    ]
    bare_uv = zero
    diagram_poles = []
    for terms in diagram_integrals:
        reduced = sum(
            (
                coefficient * solution.reduce(powers, integral=integral)
                for powers, coefficient in terms
            ),
            zero,
        )
        reduced = reduced.replace_multiple(basis_rules).together()
        # Pole-only analytic masters suffice only if their exact coefficients have
        # no spurious pole at D=4.  The two independent master symbols stay intact.
        assert (
            reduced.replace(d, 4 - 2 * eps).series(eps, 0, -1).to_expression() == zero
        ), "Master coefficient singular at D=4"
        poles = reduced.replace_multiple(master_poles).replace(d, 4 - 2 * eps)
        # Each loop measure is i*(4pi)^(eps-2); divide by i*e^4/(16pi^2)^2.
        poles = (
            (-poles * (1 + 2 * eps * log_4pi) / (imaginary * charge**4))
            .series(eps, 0, -1)
            .to_expression()
        )
        poles = poles.expand()
        diagram_poles.append(poles)
        bare_uv += flavors * poles

    bare_uv = bare_uv.expand()

    transverse = s * Qg - Qpp
    expected_bare = flavors * (
        mass_squared * xi * Qg * (-2 / eps**2 + (4 * log_mass - 4 * log_4pi + 3) / eps)
        - 2 * (2 * xi + 3) * transverse / (3 * eps)
    )
    assert (bare_uv - expected_bare).together() == zero

    ct_coordinate, ct_integral = S("ctd", "ctI")
    # Derive the four one-loop counterterm insertions on one generated bubble.
    # QED.mod gives vertexCT/tree = sqrt(ZA)*Ze*Zpsi-1 and a kinetic insertion
    # i*(Zpsi-1)*slash(q).  The one-loop Ward relation gives deltaZ1=deltaZpsi;
    # hence two vertex factors +deltaZpsi and two kinetic factors -deltaZpsi*q_i².
    # Keeping the latter's squared denominator through IR rearrangement is essential.
    # Physical electron mass is zero, so its mass counterterm contributes nothing.
    # These explicit local insertions are not automatic CT/forest generation.
    one_loop_generated = model.process(
        ["a"], ["a"], vertex_allow=["V_98"]
    ).generate_diagrams(
        loops=1,
        max_vertices=2,
        maximum_bridges=0,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(one_loop_generated.diagrams) == 1
    one_loop_diagram = one_loop_generated.diagrams[0]
    factor = (
        one_loop_diagram.overall_factor_expression(evaluate=True)
        * one_loop_diagram.numerator_prefactor_expression()
    )
    assert factor == -one  # Preserve the native closed-fermion-loop sign.
    fermions = [edge for edge in one_loop_diagram.edges if not edge.is_external]
    assert len(fermions) == 2 and all(
        edge.particle_name in ("e-", "e+") for edge in fermions
    )
    ports = {
        edge.external_index: dict(
            next(
                one_loop_diagram.projector_expression().match(
                    wave(edge.id, mink(4, index)), max_level=0
                )
            )
        )[index]
        for edge in one_loop_diagram.external_edges
    }
    numerator = model.expand_couplings(
        one_loop_diagram.numerator_expression().to_expression()
    ).replace(mass, zero)
    ct_kinematics = Kinematics(d, momenta=[k(0), p(0)]).with_scalar_product(
        p(0), p(0), s
    )
    ct_vacuum = Kinematics(d, momenta=[k(0)])
    ct_family = IntegralFamily(
        [k(0)],
        [],
        [ct_vacuum.scalar_product(k(0), k(0)) - mass_squared],
        kinematics=ct_vacuum,
    )
    ct_reducer = TensorReducer(d).with_integrated_vector(k(0, mink(d)))
    # Supplied one-loop field counterterm; the bubble and insertion sum are computed.
    zpsi_one = -xi / eps
    insertions = [
        ("bare", numerator, {}, one),
        ("vertex_0", numerator, {}, zpsi_one),
        ("vertex_1", numerator, {}, zpsi_one),
    ]
    for number, edge in enumerate(fermions):
        qi = Q(edge.id, mink(4))
        insertions.append(
            (f"kinetic_{number}", numerator * dot(qi, qi), {edge.id: 2}, -zpsi_one)
        )
    ct_parts, ct_targets = ({}, set())
    for label, inserted, powers, weight in insertions:
        uv = one_loop_diagram.momentum_basis().route_expression(
            one_loop_diagram.uv_expansion(
                uv_mass, numerator=inserted, edge_powers=powers
            )
        )
        uv = (
            uv.with_lorentz_dimension(d)
            .replace(Symbols.dimension, d)
            .rename_indices({ports[0]: mu, ports[1]: nu})
        )
        pattern = denominator(
            edge_pattern, momentum_pattern, mass_pattern, quadratic_pattern
        )
        for match in list(uv.match(pattern)):
            values = dict(match)
            coordinate = ct_family.rewrite_numerator(
                values[quadratic_pattern].replace(uv_mass**2, mass_squared),
                [ct_coordinate],
            )
            assert coordinate == ct_coordinate
            uv = uv.replace(
                denominator(
                    values[edge_pattern],
                    values[momentum_pattern],
                    values[mass_pattern],
                    values[quadratic_pattern],
                ),
                coordinate,
            )
        assert not uv.matches(pattern)
        for monomial, _ in uv.expand().coefficient_list(uv_mass):
            power = int((monomial.derivative(uv_mass) * uv_mass / monomial).together())
            assert power >= 0 and monomial == uv_mass**power
        # As above, formal coordinates already retain the massive denominators.
        selective = uv.replace(uv_mass, zero)
        trace = (
            TensorExpression(selective.expand())
            .simplify_gamma()
            .expand()
            .to_dots()
            .to_expression()
        )
        scalar = ct_family.rewrite_numerator(
            ct_kinematics.apply(ct_reducer.reduce(trace)), [ct_coordinate]
        )
        tensor = (factor * scalar / charge**2).together().expand()
        scalar = tensor.replace(gmunu, Qg).replace(ppmunu, Qpp).expand()
        assert (
            scalar - scalar.coefficient(Qg) * Qg - scalar.coefficient(Qpp) * Qpp
        ).expand() == zero
        terms = []
        for monomial, coefficient in scalar.coefficient_list(ct_coordinate):
            power = -int(
                (
                    monomial.derivative(ct_coordinate) * ct_coordinate / monomial
                ).together()
            )
            assert monomial == ct_coordinate ** (-power)
            assert coefficient.derivative(ct_coordinate).expand() == zero
            assert not coefficient.matches(k(arguments))
            assert not coefficient.matches(p(arguments))
            assert not coefficient.matches(Q(arguments))
            ct_targets.add((power,))
            terms.append(([power], coefficient))
        ct_parts[label] = terms
    ct_solution = IBPFamily(ct_family, name="photon2_ct").reduce_laporta(
        [list(target) for target in sorted(ct_targets)], max_depth=2
    )
    assert ct_solution.residuals == [[1]]
    # The analytic A0 finite term is an explicit input: multiplication by deltaZpsi
    # promotes it to a single pole.  Retain the D dependence and loop-measure term.
    # ct_sum is divided by i*a4²*Nf; bare_uv already includes the flavor multiplicity.
    tadpole = mass_squared * (1 / eps + 1 - log_mass)
    ct_reduced, ct_integrated, ct_poles = ({}, {}, {})
    for label, _, _, weight in insertions:
        ct_reduced[label] = sum(
            (
                coefficient * ct_solution.reduce(power, integral=ct_integral)
                for power, coefficient in ct_parts[label]
            ),
            zero,
        ).together()
        assert (
            ct_reduced[label].replace(d, 4 - 2 * eps).series(eps, 0, -1).to_expression()
            == zero
        ), "CT master coefficient singular at D=4"
        ct_integrated[label] = ct_reduced[label].replace(
            ct_integral(1), tadpole
        ).replace(d, 4 - 2 * eps) * (1 + eps * log_4pi)
        ct_poles[label] = (
            (weight * ct_integrated[label]).series(eps, 0, -1).to_expression().expand()
        )
    bare_finite = ct_integrated["bare"].series(eps, 0, 0).to_expression().expand()
    ct_sum = sum(
        (ct_poles[label] for label in ct_poles if label != "bare"), zero
    ).expand()
    za_one = -flavors * ct_poles["bare"].coefficient(Qpp)
    zam_one = (
        -flavors
        * ct_poles["bare"].coefficient(Qg).replace(s, zero)
        / (2 * mass_squared)
    ).together()
    # These references are assertions after generation and integration, not inputs.
    T = s * Qg - Qpp
    assert (
        ct_poles["bare"] - (4 * mass_squared * Qg - 4 * T / 3) / eps
    ).together() == zero
    assert (za_one + 4 * flavors / (3 * eps)).together() == zero
    assert (zam_one + 2 * flavors / eps).together() == zero
    assert (
        flavors * ct_poles["bare"] - za_one * T + 2 * mass_squared * zam_one * Qg
    ).together() == zero
    expected_ct = xi * mass_squared * Qg * (
        4 / eps**2 + (-4 * log_mass + 4 * log_4pi - 4) / eps
    ) + 4 * xi * T / (3 * eps)
    assert (ct_sum - expected_ct).together() == zero
    assert (ct_poles["vertex_0"] - ct_poles["vertex_1"]).expand() == zero
    assert (ct_poles["kinetic_0"] - ct_poles["kinetic_1"]).expand() == zero
    assert ct_sum.coefficient(eps ** (-3)) == zero
    assert ct_sum.replace(xi, zero) == zero

    # The auxiliary photon operator in the reference model is i*M*(ZAm²-1)*g.
    # Its second-order coefficient includes the square of the derived one-loop term.
    loop_sum = (bare_uv + flavors * ct_sum).expand()
    za_two = -loop_sum.coefficient(Qpp)
    zam_two = (
        -(loop_sum.coefficient(Qg).replace(s, zero) / mass_squared + zam_one**2) / 2
    )
    za_two, zam_two = za_two.together(), zam_two.together()
    tree_two = -za_two * T + mass_squared * (2 * zam_two + zam_one**2) * Qg
    assert (loop_sum + tree_two).together() == zero
    assert (za_two + 2 * flavors / eps).together() == zero
    assert (
        zam_two - flavors * xi / (2 * eps) + flavors * (2 * flavors + xi) / eps**2
    ).together() == zero
    assert za_two.derivative(xi).expand() == zero
    assert za_two.coefficient(eps**-2) == zero
    for value in (za_one, zam_one, za_two, zam_two):
        assert value.derivative(log_mass).expand() == zero
        assert value.derivative(log_4pi).expand() == zero
        assert value.derivative(mass_squared).expand() == zero
        assert value.coefficient(eps**-3) == zero
    assert (
        loop_sum
        - flavors * (mass_squared * xi * Qg * (2 / eps**2 - 1 / eps) - 2 * T / eps)
    ).together() == zero
    return (
        Qg,
        Qpp,
        bare_finite,
        bare_uv,
        ct_family,
        ct_integral,
        ct_solution,
        ct_sum,
        eps,
        family,
        flavors,
        gmunu,
        integral,
        log_4pi,
        log_mass,
        mass_squared,
        one_loop_diagram,
        ppmunu,
        result,
        s,
        solution,
        tree_two,
        xi,
        za_one,
        za_two,
        zam_one,
        zam_two,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Propagator powers and the auxiliary mass

    A longitudinal photon term contains $q^\mu q^\nu/(q^2)^2$.
    Its numerator is polynomial and its edge gets `edge_powers={edge_id: 2}`.
    The Feynman-gauge term keeps power one. Both use the shared UV expansion.

    To match the reference's **selective infrared rearrangement**, freeze the
    massive denominators into family coordinates, then discard explicit
    powers of the auxiliary mass in the remaining numerator. The family
    still contains $k^2-M$. This omits the compensating mass terms that the
    complete UV expansion normally retains; the auxiliary counterterm below
    therefore differs from the complete-expansion one-loop example.

    Native IBP supplies rational coefficients. Analytic master integrals are
    explicit inputs: $A=M[1/\epsilon+1-\log M]+O(\epsilon)$ and
    $V=M[3/(2\epsilon^2)+(9/2-3\log M)/\epsilon]+O(1)$.
    The two-loop master coefficients are checked to be regular at $D=4$.
    The remaining loop-measure factor is $(4\pi)^{L\epsilon}$.
    """)
    return


@app.cell
def _(
    Qg,
    Qpp,
    bare_finite,
    ct_family,
    ct_integral,
    ct_solution,
    family,
    gmunu,
    integral,
    mo,
    ppmunu,
    solution,
):
    mo.ui.tabs(
        {
            "Two-loop reduction": mo.vstack(
                [
                    family,
                    mo.ui.table([solution.stats], selection=None),
                    mo.md("**Example raised powers: I(2, 2, 1)**"),
                    solution.reduce([2, 2, 1], integral=integral),
                    mo.md(
                        "The residuals are the sunset and three equivalent products of tadpoles; momentum maps identify the products."
                    ),
                ]
            ),
            "Counterterm insertions": mo.vstack(
                [
                    ct_family,
                    mo.ui.table([ct_solution.stats], selection=None),
                    mo.md("**Example raised power: I(5)**"),
                    ct_solution.reduce([5], integral=ct_integral),
                    mo.md(
                        "**Expanded one-loop bubble through finite order**, used inside the counterterm insertions:"
                    ),
                    bare_finite.replace(Qg, gmunu).replace(Qpp, ppmunu),
                ]
            ),
        }
    )
    return


@app.cell
def _(mo, one_loop_diagram, result):
    mo.vstack(
        [
            mo.md("## Generated diagrams"),
            mo.hstack(result.diagrams),
            mo.md("**One-loop bubble used for four local insertions**"),
            one_loop_diagram,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculate the counterterm insertions

    The one-loop result supplies $\delta Z_\psi=-\xi/\epsilon$ and the Ward
    relation $\delta Z_1=\delta Z_\psi$. Insert it once at each of the two
    vertices, and once on each of the two fermion lines. A kinetic insertion
    multiplies the numerator by $-\delta Z_\psi q_e^2$ and squares that
    fermion denominator. Shared UV expansion and one-loop IBP calculate all
    four contributions; the finite tadpole term is needed because each
    insertion already contains $1/\epsilon$.

    These are explicitly constructed local insertions on a generated graph.
    This example does not enumerate arbitrary counterterm models or forests.
    The one-loop bubble also determines $\delta Z_{A,1}$ and the auxiliary
    $\delta Z_{Am,1}$. The reference's auxiliary operator is
    $iM(Z_{Am}^2-1)g^{\mu\nu}$, so its two-loop term includes
    $2\delta Z_{Am,2}+\delta Z_{Am,1}^2$.
    """)
    return


@app.cell
def _(mo, za_one, za_two, zam_one, zam_two):
    mo.vstack(
        [
            mo.md("## Derived renormalization constants"),
            mo.md(
                r"Write $Z=1+a_4\delta Z_1+a_4^2\delta Z_2$. The photon result is independent of $\xi$; the auxiliary mass constant depends on the rearrangement."
            ),
            mo.vstack(
                [
                    mo.hstack([mo.md(_label), _value])
                    for _label, _value in (
                        ("**δZA,1**", za_one),
                        ("**δZA,2**", za_two),
                        ("**δZAm,1**", zam_one),
                        ("**δZAm,2**", zam_two),
                    )
                ]
            ),
            mo.md(
                r"The physical two-loop field counterterm is $\delta Z_{A,2}=-2N_f/\epsilon$; no finite two-loop amplitude is evaluated here."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    flavor_count = mo.ui.slider(0, 8, step=1, value=1, label="Lepton flavors Nf")
    auxiliary_mass_squared = mo.ui.slider(
        0.5, 3.0, step=0.5, value=1.0, label="Auxiliary squared mass M"
    )
    mo.vstack([mo.hstack([gauge_parameter, flavor_count]), auxiliary_mass_squared])
    return auxiliary_mass_squared, flavor_count, gauge_parameter


@app.cell
def _(
    Qg,
    Qpp,
    auxiliary_mass_squared,
    bare_uv,
    ct_sum,
    eps,
    flavor_count,
    flavors,
    gauge_parameter,
    log_4pi,
    log_mass,
    mass_squared,
    math,
    mo,
    s,
    tree_two,
    xi,
    za_one,
    za_two,
    zam_one,
    zam_two,
):
    _values = {
        xi: gauge_parameter.value,
        flavors: flavor_count.value,
        mass_squared: auxiliary_mass_squared.value,
        log_mass: math.log(auxiliary_mass_squared.value),
        log_4pi: math.log(4 * math.pi),
        s: 1,
    }
    _rows = []
    _entries = []
    for _name, _tensor in (
        ("Two-loop diagrams", bare_uv),
        ("Four one-loop insertions", flavors * ct_sum),
        ("Local two-loop counterterm", tree_two),
    ):
        _coefficients = [
            complex(
                _tensor.expand()
                .coefficient(_basis)
                .coefficient(eps**-_power)
                .evaluate(_values)
            ).real
            for _basis in (Qg, Qpp)
            for _power in (2, 1)
        ]
        _entries.append(_coefficients)
        _rows.append(
            dict(
                zip(
                    ["Contribution", "g / ε²", "g / ε", "pp / ε²", "pp / ε"],
                    [_name, *_coefficients],
                    strict=True,
                )
            )
        )
    cancellation_error = max(
        abs(sum(_column)) for _column in zip(*_entries, strict=True)
    )
    assert cancellation_error < 1e-10
    selected_constants = [
        complex(_constant.expand().coefficient(eps**-_power).evaluate(_values)).real
        for _constant, _power in (
            (za_one, 1),
            (za_two, 1),
            (zam_one, 1),
            (zam_two, 2),
            (zam_two, 1),
        )
    ]
    mo.vstack(
        [
            mo.md(
                r"**Two-loop pole cancellation**, in units of $ia_4^2$, with $p^2=1$. Each tensor coefficient is checked separately."
            ),
            mo.ui.table(_rows, selection=None),
            mo.md(f"Largest cancellation residual: **{cancellation_error:.1e}**"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
