import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="IBP: bubble differential equations",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolica import E, Matrix, Replacement
    from symbolica.community.hep import oneloop

    return E, Matrix, Replacement, mo, np, oneloop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Differential equations from IBP
    [Browse all notebooks](/) · [Bubble reductions](/?file=hep/ibp_bubble.py) ·
    [Massless triangle](/?file=hep/ibp_triangle.py)

    Differentiating a loop integral produces raised propagator powers. IBP
    reduces these derivatives to the original basis, giving differential
    equations for the master integrals. This notebook composes the
    unequal-mass bubble notebook, reusing its family and reductions.

    For $D_1=k^2-a$, $D_2=(k-p)^2-b$, $p^2=s$, define the vector
    $\mathbf J=(A(a),A(b),B(s,a,b))^T=(I(1,0),I(0,1),I(1,1))^T$.
    The three equations are $\partial_v\mathbf J=M_v\mathbf J$,
    for $v=s,a,b$. Symbolica checks their compatibility exactly; then a
    numerical solution transports one boundary value to another invariant.
    """)


@app.cell
def _():
    from ibp_bubble import app as bubble_app

    return (bubble_app,)


@app.cell
async def _(bubble_app):
    _shared = await bubble_app.embed()
    bubble_ibp = _shared.defs["bubble_ibp"]
    dimension = _shared.defs["dimension"]
    epsilon = _shared.defs["epsilon"]
    integral = _shared.defs["integral"]
    invariant = _shared.defs["invariant"]
    mass_a = _shared.defs["mass_a"]
    mass_b = _shared.defs["mass_b"]
    finite_a = _shared.defs["finite_a"]
    finite_b = _shared.defs["finite_b"]
    finite_bubble = _shared.defs["finite_bubble"]
    reductions = _shared.defs["reductions"]
    return (
        bubble_ibp,
        dimension,
        epsilon,
        finite_a,
        finite_b,
        finite_bubble,
        integral,
        invariant,
        mass_a,
        mass_b,
        reductions,
    )


@app.cell
def _(
    E,
    Matrix,
    bubble_ibp,
    dimension,
    integral,
    invariant,
    mass_a,
    mass_b,
    mo,
    reductions,
):
    masters = [integral(1, 0), integral(0, 1), integral(1, 1)]
    zero = E("0")
    # The invariant derivative follows from p·∂p/(2s) at fixed masses.
    _tadpole_solution = bubble_ibp.reduce_laporta([[0, 2]], max_depth=1)
    tadpole_derivative = _tadpole_solution.reduce([0, 2], integral=integral)
    derivative_s = (
        tadpole_derivative
        - integral(1, 1)
        + (mass_a - mass_b - invariant) * reductions[1, 2]
    ) / (2 * invariant)
    derivatives = {
        invariant: [zero, zero, derivative_s],
        mass_a: [(dimension - 2) / (2 * mass_a) * masters[0], zero, reductions[2, 1]],
        mass_b: [zero, tadpole_derivative, reductions[1, 2]],
    }
    connections = {
        variable: Matrix.from_linear(
            3,
            3,
            [
                expression.expand().coefficient(master).together()
                for expression in expressions
                for master in masters
            ],
        )
        for variable, expressions in derivatives.items()
    }
    # Homogeneity follows independently from the mass dimension D-4.
    scaling_residual = (
        invariant * derivative_s
        + mass_a * reductions[2, 1]
        + mass_b * reductions[1, 2]
        - (dimension / 2 - 2) * masters[2]
    ).together()
    assert scaling_residual == zero
    curvatures = {}
    for _v, _w in [(invariant, mass_a), (invariant, mass_b), (mass_a, mass_b)]:
        _Mv, _Mw = connections[_v], connections[_w]
        _curvature = (
            _Mw.map(lambda entry, variable=_v: entry.derivative(variable))
            + _Mw * _Mv
            - _Mv.map(lambda entry, variable=_w: entry.derivative(variable))
            - _Mv * _Mw
        )
        assert _curvature == Matrix(3, 3)
        curvatures[_v, _w] = _curvature
    mo.vstack(
        [
            mo.md(r"""
        **Differentiate before reduction:**
        $$\partial_a B=I(2,1),\quad \partial_b B=I(1,2),\quad
        \partial_s B=\frac{I(0,2)-I(1,1)+(a-b-s)I(1,2)}{2s}.$$
        **Checks:** $s\partial_sB+a\partial_aB+b\partial_bB=(D/2-2)B$,
        and all three compatibility matrices
        $\partial_vM_w-\partial_wM_v+M_wM_v-M_vM_w$ vanish exactly.
        """),
        ]
    )
    return connections, derivatives, masters, zero


@app.cell
def _(mo):
    derivative_choice = mo.ui.dropdown(
        ["s", "a", "b"],
        value="s",
        label="Differentiate with respect to",
    )
    point_choice = mo.ui.dropdown(
        {
            "Spacelike": (-5.0, 2.0, 3.0, -1.0),
            "Timelike": (2.0, 2.0, 3.0, 1.0),
            "Unequal masses": (5.0, 1.0, 4.0, 2.0),
            "Near threshold": (8.0, 1.0, 4.0, 2.0),
        },
        value="Spacelike",
        label="Endpoint and masses",
    )
    mo.hstack([derivative_choice, point_choice])
    return derivative_choice, point_choice


@app.cell
def _(
    E,
    Replacement,
    derivatives,
    dimension,
    epsilon,
    finite_a,
    finite_b,
    finite_bubble,
    mass_a,
    mass_b,
    masters,
    zero,
):
    _master_series = [
        Replacement(masters[0], mass_a / epsilon + finite_a),
        Replacement(masters[1], mass_b / epsilon + finite_b),
        Replacement(masters[2], 1 / epsilon + finite_bubble),
    ]
    finite_derivatives = {}
    for _variable, _rows in derivatives.items():
        _expanded = (
            _rows[2]
            .replace(dimension, 4 - 2 * epsilon)
            .replace_multiple(_master_series)
            .series(epsilon, 0, 0)
            .to_expression()
            .expand()
        )
        assert _expanded.coefficient(epsilon**-1).together() == zero
        finite_derivatives[_variable] = (
            dict(_expanded.coefficient_list(epsilon)).get(E("1"), zero).together()
        )
    return (finite_derivatives,)


@app.cell
def _(
    connections,
    derivative_choice,
    finite_a,
    finite_b,
    finite_bubble,
    finite_derivatives,
    invariant,
    mass_a,
    mass_b,
    mo,
    np,
    oneloop,
    point_choice,
):
    _variable = {"s": invariant, "a": mass_a, "b": mass_b}[derivative_choice.value]
    _sv, _av, _bv, _ = point_choice.value
    _values = {
        invariant: _sv,
        mass_a: _av,
        mass_b: _bv,
        finite_a: complex(oneloop.A0(_av, 1.0)[0]),
        finite_b: complex(oneloop.A0(_bv, 1.0)[0]),
        finite_bubble: complex(oneloop.B0(_sv, _av, _bv, 1.0)[0]),
    }
    selected_derivative = complex(finite_derivatives[_variable].evaluate(_values))
    _nodes, _weights = np.polynomial.legendre.leggauss(128)
    _x, _weights = (_nodes + 1) / 2, _weights / 2
    _delta = _av * _x + _bv * (1 - _x) - _sv * _x * (1 - _x)
    assert min(_delta) > 0
    _numerator = {"s": _x * (1 - _x), "a": -_x, "b": -(1 - _x)}[derivative_choice.value]
    parameter_derivative = float(np.dot(_weights, _numerator / _delta))
    derivative_error = abs(selected_derivative - parameter_derivative)
    assert derivative_error < 2e-11
    mo.vstack(
        [
            mo.md(f"**Connection matrix M_{derivative_choice.value}**"),
            connections[_variable],
            mo.md("**Finite derivative after expanding at D=4−2ε**"),
            finite_derivatives[_variable],
            mo.md(r"""
        Independent parameter integral:
        $B_{\rm finite}=-\int_0^1\log[\Delta(x)/\mu^2]\,dx$,
        $\Delta=ax+b(1-x)-sx(1-x)$. Its $s,a,b$ derivatives have numerators
        $x(1-x),-x,-(1-x)$ over $\Delta$. The presets stay below threshold.
        """),
            mo.ui.table(
                [
                    {
                        "IBP derivative": str(selected_derivative),
                        "Parameter derivative": parameter_derivative,
                        "Absolute error": derivative_error,
                    }
                ],
                selection=None,
            ),
        ]
    )


@app.cell
def _(
    finite_a,
    finite_b,
    finite_bubble,
    finite_derivatives,
    invariant,
    mass_a,
    mass_b,
    mo,
    oneloop,
    point_choice,
):
    _endpoint, _av, _bv, _start = point_choice.value
    _rhs = finite_derivatives[invariant]
    _parameters = {
        mass_a: _av,
        mass_b: _bv,
        finite_a: complex(oneloop.A0(_av, 1.0)[0]),
        finite_b: complex(oneloop.A0(_bv, 1.0)[0]),
    }
    _boundary = complex(oneloop.B0(_start, _av, _bv, 1.0)[0])
    _reference = complex(oneloop.B0(_endpoint, _av, _bv, 1.0)[0])

    # A local RK4 demonstration; all physics coefficients come from the IBP result.
    def transport_rhs(s_value, bubble_value):
        return complex(
            _rhs.evaluate(
                {**_parameters, invariant: s_value, finite_bubble: bubble_value}
            )
        )

    transport_checks = []
    for _steps in (128, 256):
        _h = (_endpoint - _start) / _steps
        _value = _boundary
        for _step in range(_steps):
            _s = _start + _step * _h
            _k1 = transport_rhs(_s, _value)
            _k2 = transport_rhs(_s + _h / 2, _value + _h * _k1 / 2)
            _k3 = transport_rhs(_s + _h / 2, _value + _h * _k2 / 2)
            _k4 = transport_rhs(_s + _h, _value + _h * _k3)
            _value += _h * (_k1 + 2 * _k2 + 2 * _k3 + _k4) / 6
        transport_checks.append(
            {
                "Steps": _steps,
                "Transported B finite": str(_value),
                "OneLOop B finite": str(_reference),
                "Error": abs(_value - _reference),
            }
        )
    transport_error = transport_checks[-1]["Error"]
    assert transport_error < 2e-9
    mo.vstack(
        [
            mo.md(
                f"**Transport the finite bubble from s={_start:g} to s={_endpoint:g}**"
            ),
            mo.ui.table(transport_checks, selection=None),
            mo.md(r"""
        OneLOop supplies one boundary value and the two constant tadpoles.
        A fourth-order Runge–Kutta integration then uses only the derived
        differential equation. A separate OneLOop evaluation checks the endpoint.
        Doubling the step count provides a convergence check.

        The connection has singular coefficients at $s=0$ and at the Källén
        zeros $\lambda(s,a,b)=0$. These paths avoid those points. Transport
        through a singular point or across a physical cut requires appropriate
        boundary data and analytic continuation; it is not implemented here.
        """),
        ]
    )


if __name__ == "__main__":
    app.run()
