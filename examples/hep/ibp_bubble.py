import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="IBP: unequal-mass bubble")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolica import E, Replacement, S
    from symbolica.community import hep
    from symbolica.community.hep import oneloop

    return E, Replacement, S, hep, mo, np, oneloop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # IBP reduction of an unequal-mass bubble

    [Browse all notebooks](/) · [Integral families](/?file=hep/integral_families.py) ·
    [Two-loop φ⁴](/?file=hep/ibp_phi4.py)

    Raised propagator powers occur in mass derivatives, counterterm insertions
    and Taylor expansions. Reduce them with the native RustRed solver, keeping
    the dimension and kinematics symbolic. The family uses
    $D_1=k^2-a$, $D_2=(k-p)^2-b$, $p^2=s$ and
    $I(n_1,n_2)=\int_k D_1^{-n_1}D_2^{-n_2}$, with positive squared masses
    $a,b$ and the common OneLOop normalization.

    The solver computes exact reduction coefficients. OneLOop evaluates the
    remaining tadpoles and ordinary bubble. An independent Feynman-parameter
    quadrature checks each finite result below the production threshold.
    """)


@app.cell
def _(S, hep, mo):
    dimension, loop, external, invariant, mass_a, mass_b, epsilon = S(
        "ibp_bubble::D",
        "ibp_bubble::k",
        "ibp_bubble::p",
        "ibp_bubble::s",
        "ibp_bubble::a",
        "ibp_bubble::b",
        "ibp_bubble::eps",
    )
    integral = S("ibp_bubble::I")
    kinematics = hep.Kinematics(
        dimension, momenta=[loop, external]
    ).with_scalar_product(external, external, invariant)
    bubble_family = hep.IntegralFamily(
        [loop],
        [external],
        [
            kinematics.scalar_product(loop, loop) - mass_a,
            kinematics.scalar_product(loop - external, loop - external) - mass_b,
        ],
        kinematics=kinematics,
    )
    bubble_ibp = hep.IBPFamily(bubble_family, name="bubble")
    discriminant = (
        invariant**2
        + mass_a**2
        + mass_b**2
        - 2 * invariant * mass_a
        - 2 * invariant * mass_b
        - 2 * mass_a * mass_b
    )
    mo.vstack([bubble_family, mo.md("**Källén polynomial λ(s,a,b)**"), discriminant])
    return (
        bubble_ibp,
        dimension,
        discriminant,
        epsilon,
        integral,
        invariant,
        mass_a,
        mass_b,
    )


@app.cell
def _(bubble_ibp, integral, mo):
    targets = [[2, 1], [1, 2], [2, 2]]
    bubble_solution = bubble_ibp.reduce_laporta(targets, max_depth=2)
    assert {tuple(powers) for powers in bubble_solution.residuals} == {
        (1, 0),
        (0, 1),
        (1, 1),
    }
    reductions = {
        tuple(target): bubble_solution.reduce(target, integral=integral).together()
        for target in targets
    }
    mo.vstack(
        [
            mo.md("**Exact Laporta reductions**"),
            *[
                mo.hstack([integral(*target), mo.md(r"$\longrightarrow$"), value])
                for target, value in reductions.items()
            ],
            mo.md(
                "The unresolved basis at this search depth is "
                "`I(1,0), I(0,1), I(1,1)`. These are the two tadpoles and the "
                "ordinary bubble; `residuals` does not certify a minimal master basis."
            ),
            mo.ui.table([bubble_solution.stats], selection=None),
        ]
    )
    return bubble_solution, reductions


@app.cell
def _(E, Replacement, S, dimension, epsilon, integral, mass_a, mass_b, mo, reductions):
    finite_a, finite_b, finite_bubble = S(
        "ibp_bubble::A_finite", "ibp_bubble::B_finite", "ibp_bubble::bubble_finite"
    )
    master_expansions = [
        Replacement(integral(1, 0), finite_a + mass_a / epsilon),
        Replacement(integral(0, 1), finite_b + mass_b / epsilon),
        Replacement(integral(1, 1), finite_bubble + 1 / epsilon),
    ]
    finite_reductions = {}
    for _target, _reduced in reductions.items():
        _laurent = (
            _reduced.replace(dimension, 4 - 2 * epsilon)
            .replace_multiple(master_expansions)
            .series(epsilon, 0, 0)
            .to_expression()
            .expand()
        )
        assert _laurent.coefficient(epsilon**-1).together() == 0
        finite_reductions[_target] = dict(_laurent.coefficient_list(epsilon))[
            E("1")
        ].together()
    mo.md(r"""
    **Dimensional expansion:** all three $1/\epsilon$ coefficients cancel.
    Expand at $D=4-2\epsilon$ **before** setting $\epsilon=0$:
    the $O(\epsilon)$ reduction coefficients multiply UV poles of the
    individual master integrals and contribute finite rational terms.
    """)
    return finite_a, finite_b, finite_bubble, finite_reductions


@app.cell
def _(mo):
    integral_choice = mo.ui.dropdown(
        ["I(2,1)", "I(1,2)", "I(2,2)"], value="I(2,1)", label="Raised integral"
    )
    point_choice = mo.ui.dropdown(
        ["Spacelike", "Timelike", "Unequal masses", "Near threshold"],
        value="Spacelike",
        label="Kinematic point",
    )
    mo.hstack([integral_choice, point_choice])
    return integral_choice, point_choice


@app.cell
def _(
    discriminant,
    finite_a,
    finite_b,
    finite_bubble,
    finite_reductions,
    integral_choice,
    invariant,
    mass_a,
    mass_b,
    mo,
    np,
    oneloop,
    point_choice,
):
    _target = {"I(2,1)": (2, 1), "I(1,2)": (1, 2), "I(2,2)": (2, 2)}[
        integral_choice.value
    ]
    _s, _a, _b = {
        "Spacelike": (-1.0, 2.0, 3.0),
        "Timelike": (2.0, 2.0, 3.0),
        "Unequal masses": (5.0, 1.0, 4.0),
        "Near threshold": (8.0, 1.0, 4.0),
    }[point_choice.value]
    _parameters = {invariant: _s, mass_a: _a, mass_b: _b}
    assert _a > 0 and _b > 0 and _s < (np.sqrt(_a) + np.sqrt(_b)) ** 2
    assert abs(complex(discriminant.evaluate(_parameters))) > 1e-10
    _parameters.update(
        {
            finite_a: complex(oneloop.A0(_a, 1.0)[0]),
            finite_b: complex(oneloop.A0(_b, 1.0)[0]),
            finite_bubble: complex(oneloop.B0(_s, _a, _b, 1.0)[0]),
        }
    )
    reduced_value = complex(finite_reductions[_target].evaluate(_parameters))

    # This quadrature does not use the IBP coefficients or OneLOop masters.
    _nodes, _weights = np.polynomial.legendre.leggauss(96)
    _x, _weights = (_nodes + 1) / 2, _weights / 2
    _delta = _x * _a + (1 - _x) * _b - _s * _x * (1 - _x)
    _integrand = {
        (2, 1): -_x / _delta,
        (1, 2): -(1 - _x) / _delta,
        (2, 2): _x * (1 - _x) / _delta**2,
    }[_target]
    parameter_value = float(np.dot(_weights, _integrand))
    absolute_error = abs(reduced_value - parameter_value)
    assert absolute_error < 2e-11
    mo.vstack(
        [
            mo.md(r"""
            **Independent finite integral:** with
            $\Delta(x)=ax+b(1-x)-sx(1-x)$, the integrands are
            $-x/\Delta$, $-(1-x)/\Delta$ and $x(1-x)/\Delta^2$ for
            $I(2,1)$, $I(1,2)$ and $I(2,2)$ respectively.
            The integration range is $0\leq x\leq1$; the scale is $\mu^2=1$.
            """),
            mo.ui.table(
                [
                    {
                        "s": _s,
                        "a": _a,
                        "b": _b,
                        "IBP + OneLOop": str(reduced_value),
                        "Parameter integral": parameter_value,
                        "Absolute error": absolute_error,
                    }
                ],
                selection=None,
            ),
        ]
    )
    return absolute_error, parameter_value, reduced_value


@app.cell
def _(E, bubble_ibp, integral, mo):
    recurrence = bubble_ibp.solve_parametric([True, True], fixed=[None, 1], max_depth=1)
    rule = recurrence.rules[0]
    mo.vstack(
        [
            mo.md("**Reusable symbolic-index recurrence**"),
            mo.hstack(
                [
                    integral(*rule.target),
                    mo.md(r"$\longrightarrow$"),
                    sum(
                        (
                            coefficient * integral(*powers)
                            for powers, coefficient in rule.terms
                        ),
                        E("0"),
                    ).together(),
                ]
            ),
            mo.md("Each of these polynomials must remain nonzero:"),
            *[condition.factor() for condition in rule.nonzero_conditions],
            mo.md(r"""
            `rule.apply([2, 1])` specializes the indices for one recurrence step.
            `solution.reduce(...)` from Laporta already includes back-substitution.

            The generic formulas above require $a\ne0$, $b\ne0$ and
            $\lambda(s,a,b)\ne0$. At a vanishing mass or Källén polynomial,
            construct a family with those kinematics and solve that case separately.
            A symbolic rule's remaining kinematic conditions also apply when its
            coefficients are evaluated numerically. The presets here avoid those loci.
            """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
