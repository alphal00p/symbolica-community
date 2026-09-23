import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Color Fierz contractions")


@app.cell
def _():
    import marimo as mo
    from symbolica import E
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.spenso import (
        ColorSimplifySettings,
        Representation,
        TensorExpression,
    )

    _set_namespace("color_fierz")

    return ColorSimplifySettings, E, Representation, TensorExpression, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Color Fierz contractions

    [Browse all notebooks](/) · [Color conjugation](/?file=hep/color_algebra.py)

    A shared adjoint index can connect an open color chain to a closed trace,
    or two traces to each other. The fundamental Fierz identity contracts it:

    $$\sum_a (T^a)_{ij}\operatorname{Tr}(T^bT^aT^c)
      =\frac12(T^cT^b)_{ij}-\frac{1}{4N}\delta_{ij}\delta^{bc}.$$

    The order $T^cT^b$ follows by cutting the cyclic trace at $T^a$.
    `simplify_color()` applies this identity in shared Idenso algebra.
    We use $N=3$ and $\operatorname{Tr}(T^aT^b)=\delta^{ab}/2$ here.
    [FeynCalc reference](https://feyncalc.github.io/FeynCalcBook/SUNSimplify.html).
    """)
    return


@app.cell
def _(Representation, TensorExpression, mo):
    generator = TensorExpression.t(8, 3)
    fundamental = Representation.cof(3)
    _adjoint = Representation.coad(8)
    _loop = (
        generator("b", "k", "l") * generator("a", "l", "m") * generator("c", "m", "k")
    )
    color_trace = _loop.collect_chains(fundamental)
    mixed_color = TensorExpression(
        generator("a", "i", "j").to_expression() * color_trace.to_expression()
    )
    fierz_result = mixed_color.simplify_color().to_cof_dimension_invariants()
    # An identity pairs a fundamental slot with a dual slot.
    _identity = TensorExpression.g(fundamental, fundamental.dual())("i", "j")
    _expected = TensorExpression(
        (generator("c", "i", "k") * generator("b", "k", "j")).to_expression() / 2
        - _identity.to_expression() * _adjoint.g("b", "c").to_expression() / 12
    )
    assert fierz_result.to_expression().expand() == (
        _expected.simplify_color().to_expression().expand()
    )
    _difference = (mixed_color - _expected).undo_trace().undo_chain().to_network()
    _difference.execute()
    _components = _difference.result_tensor()[:]
    assert len(_components) == 576
    numeric_fierz_error = max(abs(complex(_value)) for _value in _components)
    assert numeric_fierz_error < 1e-12
    mo.vstack(
        [
            mo.md("**Generator contracted with a trace**"),
            mixed_color,
            mo.md("**Reduced open chain and identities**"),
            fierz_result,
            mo.md(
                f"Independent SU(3) matrix check: maximum component error {numeric_fierz_error:.2e}."
            ),
        ]
    )
    return color_trace, fierz_result, mixed_color, numeric_fierz_error


@app.cell
def _(ColorSimplifySettings, fierz_result, mixed_color, mo):
    _settings = ColorSimplifySettings(
        evaluate_traces=False, expand_cross_chain_fierz=False
    )
    preserved_color = mixed_color.simplify_color(_settings)
    assert preserved_color.to_expression() != fierz_result.to_expression()
    assert (
        preserved_color.simplify_color().to_cof_dimension_invariants().to_expression()
        == fierz_result.to_expression()
    )
    mo.vstack(
        [
            mo.md("**Keeping the trace and separate color lines**"),
            preserved_color,
            mo.md(
                "Disable both trace evaluation and Fierz expansion to retain this form. "
                "Re-enabling simplification gives the same reduced result."
            ),
        ]
    )
    return


@app.cell
def _(E, TensorExpression, color_trace, mo):
    _closed = TensorExpression(
        color_trace.to_expression() * color_trace.spenso_conjugate().to_expression()
    )
    trace_norm = (
        _closed.simplify_color().to_cof_dimension_invariants().simplify_metrics()
    )
    assert trace_norm.is_scalar
    assert trace_norm.to_expression() == E("7/3")
    _network = _closed.undo_trace().undo_chain().to_network()
    _network.execute()
    numeric_trace_norm = _network.result_scalar()
    assert abs(complex(numeric_trace_norm) - 7 / 3) < 1e-12
    mo.vstack(
        [
            mo.md(r"""
            **Two contracted traces**

            $$\sum_{a,b,c}|\operatorname{Tr}(T^aT^bT^c)|^2
              =\frac{(N^2-1)(N^2-2)}{8N}=\frac73\quad(N=3).$$

            The exact symbolic result agrees with an independent numerical
            contraction of explicit SU(3) matrices from Spenso's tensor library.
            """),
            trace_norm,
            numeric_trace_norm,
        ]
    )
    return (trace_norm,)


if __name__ == "__main__":
    app.run()
