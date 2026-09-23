import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Electron anomalous magnetic moment")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica.community import hep
    from symbolica.community.hep import oneloop
    from symbolica.community.spenso import TensorExpression

    return E, Matrix, Replacement, S, Symbol, TensorExpression, hep, mo, np, oneloop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Electron anomalous magnetic moment

    [Browse all notebooks](/) ·
    [Pauli form factor across threshold](/?file=hep/pauli_form_factor.py) ·
    [Massive electron self-energy](/?file=hep/electron_self_energy.py) ·
    [Unequal-mass bubble](/?file=hep/ibp_bubble.py) ·
    [Two-loop φ⁴](/?file=hep/ibp_phi4.py) · [Higgs to gluons](/?file=hep/higgs_gluons.py)

    Generate the QED vertex and extract its Pauli form factor, reproducing
    [FeynCalc's one-loop g−2 example](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/El-GaEl).
    Both electrons are on shell, $p^2=p'^2=m^2$, while the photon probes a
    spacelike momentum transfer $q=p-p'$, $t=q^2<0$.

    Between the external spinors the vertex has the form
    $(F_1+F_2)\gamma^\mu-F_2(p+p')^\mu/(2m)$.
    Shared spin sums close the Dirac traces; a two-by-two Symbolica system
    extracts the coefficient of $(p+p')^\mu/(2m)$.
    We project at generic $t$ and take $t\to0$ only after integration because
    those two projectors become degenerate at zero momentum transfer.

    All generator weights are retained. Normalizing to the generated tree
    vertex fixes the coupling and external-fermion phase convention. The
    common loop measure is $i/(16\pi^2)$ in the OneLOop normalization.
    """)


@app.cell
def _(E, Matrix, Replacement, S, Symbol, TensorExpression, hep, mo):
    model = hep.Model.standard_model()
    dimension, transfer, mass, charge, mu = S(
        "gminus2::D", "gminus2::t", "UFO::Me", "UFO::ee", "gminus2::mu"
    )
    K, P, mink, bis, metric, gamma = S(
        "gammalooprs::K",
        "gammalooprs::P",
        "spenso::mink",
        "spenso::bis",
        "spenso::g",
        "spenso::gamma",
    )
    a, b, c, d, index, wave, slot = S(
        "gminus2::a",
        "gminus2::b",
        "gminus2::c",
        "gminus2::d",
        "index_",
        "wave_",
        "slot_",
    )
    kinematics = (
        hep.Kinematics(dimension, momenta=[K(0), P(0), P(1)])
        .with_scalar_product(P(0), P(0), mass**2)
        .with_scalar_product(P(1), P(1), transfer)
        .with_scalar_product(P(0), P(1), transfer / 2)
    )
    operators, diagrams = [], []
    for loops in (0, 1):
        generated = hep.Generator(model).generate(
            hep.Process.amplitude([11], [22, 11]).with_loop_count(loops, loops),
            max_vertices=1 + 2 * loops,
            maximum_bridges=0,
            vertex_allow=["V_98"],
            numerator_grouping=None,
            progress=None,
        )
        assert len(generated.diagrams) == 1
        diagram = generated.diagrams[0]
        operator = (
            model.expand_couplings(
                diagram.numerator_expression(in_lmb=True).to_expression()
            )
            * diagram.overall_factor_expression(evaluate=True)
            * diagram.numerator_prefactor_expression()
        )
        for edge in diagram.external_edges:
            representation = mink if edge.external_index == 1 else bis
            matches = list(
                diagram.projector_expression().match(
                    wave(edge.id, representation(4, index)), max_level=0
                )
            )
            assert len(matches) == 1
            port = dict(matches[0])[index]
            operator = operator.replace(
                representation(4, port),
                representation(4, {0: b, 1: mu, 2: a}[edge.external_index]),
            )
        operators.append(operator.replace(mink(4, index), mink(dimension, index)))
        diagrams.append(diagram)

    # Normalize to the generated tree vertex, including its external fermion sign.
    # A scalar loop contributes i/(16*pi^2) in the OneLOop convention below.
    tree_coupling = (
        operators[0] / gamma(bis(4, a), bis(4, b), mink(dimension, mu))
    ).together()
    assert (tree_coupling - Symbol.I * charge).expand() == E("0")
    vertex = (Symbol.I * operators[1] / (tree_coupling * charge**2)).expand()
    particle = model.particle_by_pdg(11)
    incoming = particle.spin_sum(P(0), b, c)
    outgoing = particle.spin_sum(P(2), d, a).replace(
        P(2, slot), P(0, slot) - P(1, slot)
    )
    incoming = incoming.replace(mink(4, index), mink(dimension, index)).replace(
        mink(4), mink(dimension)
    )
    outgoing = outgoing.replace(mink(4, index), mink(dimension, index)).replace(
        mink(4), mink(dimension)
    )
    vector_sum = 2 * P(0, mink(dimension, mu)) - P(1, mink(dimension, mu))
    basis = [
        gamma(bis(4, a), bis(4, b), mink(dimension, mu)),
        vector_sum * metric(bis(4, a), bis(4, b)) / (2 * mass),
    ]
    probes = [
        gamma(bis(4, c), bis(4, d), mink(dimension, mu)),
        vector_sum * metric(bis(4, c), bis(4, d)) / (2 * mass),
    ]
    traces = []
    for operator in basis + [vertex]:
        for probe in probes:
            trace = (
                TensorExpression((operator * incoming * probe * outgoing).expand())
                .simplify_gamma()
                .expand()
                .to_dots()
                .to_expression()
            )
            traces.append(kinematics.apply(trace).expand())
    # These independently specified traces also fix the normalization of the
    # projector basis gamma^mu and (p_in+p_out)^mu/(2m).
    expected_gram = [
        2 * (dimension - 2) * transfer + 8 * mass**2,
        8 * mass**2 - 2 * transfer,
        8 * mass**2 - 2 * transfer,
        (4 * mass**2 - transfer) ** 2 / (2 * mass**2),
    ]
    assert all(
        (actual - expected).expand() == E("0")
        for actual, expected in zip(traces[:4], expected_gram, strict=True)
    )
    projectors = Matrix.from_linear(2, 2, traces[:4])
    pauli_numerator = projectors.solve(Matrix.vec(traces[4:]))[1, 0].to_expression()
    family = diagrams[1].integral_family(kinematics=kinematics)
    denominators = S("gminus2::d0", "gminus2::d1", "gminus2::d2")
    polynomial = family.rewrite_numerator(pauli_numerator, denominators).expand()
    terms = []
    for monomial, coefficient in polynomial.coefficient_list(*denominators):
        powers = [
            1 - monomial.to_polynomial(vars=denominators).degree(label)
            for label in denominators
        ]
        terms.append((powers, coefficient))
    solution = hep.IBPFamily(family, name="electron_vertex").reduce_laporta(
        [powers for powers, coefficient in terms], max_depth=2
    )
    assert {tuple(powers) for powers in solution.residuals} == {
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    }
    integral, tadpole, bubble = S("gminus2::I", "gminus2::A0", "gminus2::B0")
    reduction = sum(
        (
            coefficient * solution.reduce(powers, integral=integral)
            for powers, coefficient in terms
        ),
        E("0"),
    )
    # Both one-line pinches are massive tadpoles; their loop shifts have unit Jacobian.
    # The two-line pinch carries q^2=t and equal masses m^2.
    reduction = reduction.replace_multiple(
        [
            Replacement(integral(0, 1, 0), tadpole),
            Replacement(integral(1, 0, 0), tadpole),
            Replacement(integral(1, 1, 0), bubble),
        ]
    ).together()
    expected = (
        2
        * (dimension - 5)
        * (-(dimension - 2) * tadpole + 2 * mass**2 * (dimension - 3) * bubble)
        / ((dimension - 3) * (transfer - 4 * mass**2))
    )
    assert (reduction - expected).together() == E("0")
    epsilon, finite_a, finite_b, logarithm = S(
        "gminus2::eps", "gminus2::Af", "gminus2::Bf", "gminus2::L"
    )
    laurent = (
        reduction.replace(dimension, 4 - 2 * epsilon)
        .replace_multiple(
            [
                Replacement(tadpole, finite_a + mass**2 / epsilon),
                Replacement(bubble, finite_b + 1 / epsilon),
            ]
        )
        .series(epsilon, 0, 0)
        .to_expression()
        .expand()
    )
    assert laurent.coefficient(epsilon**-1).together() == E("0")
    finite = dict(laurent.coefficient_list(epsilon))[E("1")].together()
    assert (
        finite
        - 4 * (finite_a + mass**2 - mass**2 * finite_b) / (transfer - 4 * mass**2)
    ).together() == E("0")
    # Gordon decomposition gives F2=-b. Restoring e^2/(16*pi^2), the b=-2
    # limit gives F2(0)=e^2/(8*pi^2)=alpha/(2*pi), independently of m and mu.
    limit = (
        finite.replace(transfer, E("0"))
        .replace_multiple(
            [
                Replacement(finite_a, mass**2 * (1 - logarithm)),
                Replacement(finite_b, -logarithm),
            ]
        )
        .together()
    )
    assert limit == E("-2")

    alpha = S("gminus2::alpha")
    anomalous_moment = (-alpha * limit / (4 * Symbol.PI)).together()
    mo.vstack(
        [
            mo.md("**Generated tree and one-loop vertex**"),
            mo.hstack(diagrams),
            mo.md("**Trace-projector matrix**"),
            projectors,
            mo.md("**Scalar integral family**"),
            family,
            mo.md("**Native IBP reduction of the projected coefficient**"),
            reduction.factor(),
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "The finite-depth residuals are two shifted massive tadpoles and "
                "the equal-mass bubble. Their identification follows directly from "
                "the displayed propagators; the solver does not certify a minimal basis."
            ),
            mo.md(r"""
            **Finite result:** with $A_f$ and $B_f$ the finite OneLOop coefficients,
            $b=4[A_f+m^2-m^2B_f]/(t-4m^2)$ and $F_2=-\alpha b/(4\pi)$.
            The $1/\epsilon$ pole cancels. Keeping $D=4-2\epsilon$ until after
            reduction retains finite terms from coefficients multiplying UV poles.

            At $t=0$, $A_f=m^2[1-\log(m^2/\mu^2)]$ and
            $B_f=-\log(m^2/\mu^2)$, so $b=-2$ independently of mass and scale.
            Since $a_e=(g-2)/2=F_2(0)$, the generated result is:
            """),
            anomalous_moment,
        ]
    )
    return anomalous_moment, finite, finite_a, finite_b, mass, transfer


@app.cell
def _(mo):
    virtuality = mo.ui.slider(
        -3.0,
        2.0,
        step=0.25,
        value=0.0,
        label="log10(Q²/m²), where t = −Q²",
    )
    mo.vstack([virtuality])
    return (virtuality,)


@app.cell
def _(finite, finite_a, finite_b, mass, mo, np, oneloop, transfer, virtuality):
    _ratio = 10.0**virtuality.value
    _t = -_ratio
    normalized_form_factor = (
        -complex(
            finite.evaluate(
                {
                    transfer: _t,
                    mass: 1.0,
                    finite_a: complex(oneloop.A0(1.0, 1.0)[0]),
                    finite_b: complex(oneloop.B0(_t, 1.0, 1.0, 1.0)[0]),
                }
            )
        )
        / 2
    )
    # Independent Feynman-parameter integral, without IBP coefficients or masters.
    _nodes, _weights = np.polynomial.legendre.leggauss(96)
    _x, _weights = (_nodes + 1) / 2, _weights / 2
    parameter_reference = float(np.dot(_weights, 1 / (1 + _ratio * _x * (1 - _x))))
    _derivative = 1 + _t * complex(oneloop.dB0(_t, 1.0, 1.0, 1.0)[0])
    form_factor_error = abs(normalized_form_factor - parameter_reference)
    assert form_factor_error < 2e-12
    assert abs(normalized_form_factor - _derivative) < 2e-12
    mo.vstack(
        [
            mo.md(r"""
            **Spacelike form factor beyond the zero-transfer limit**

            The independent parameter formula is
            $$\frac{F_2(t)}{\alpha/(2\pi)}
              =\int_0^1\!dx\,\frac{m^2}{m^2-tx(1-x)}
              =1+t\,\frac{dB_0(t,m^2,m^2)}{dt}.$$
            Move the slider to compare the generated reduction with quadrature
            and the shared OneLOop derivative. The normalized result approaches
            one as $Q^2/m^2\to0$.
            """),
            mo.ui.table(
                [
                    {
                        "Q²/m²": _ratio,
                        "Generated F₂ / (α/2π)": normalized_form_factor.real,
                        "Parameter integral": parameter_reference,
                        "Absolute error": form_factor_error,
                    }
                ],
                selection=None,
            ),
        ]
    )
    return form_factor_error, normalized_form_factor, parameter_reference


if __name__ == "__main__":
    app.run()
