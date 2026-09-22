import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Color chains and conjugation")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica.community.spenso import Representation, TensorExpression

    return E, S, Representation, TensorExpression, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Color chains and conjugation

    [Browse all notebooks](/) · [Quark annihilation into gluons](/?file=hep/qcd_gluons.py)

    Build SU(3) color generators using Spenso's existing tensor constructors.
    Contracted fundamental indices can be collected into compact chains or traces.
    Conjugation gives the same result before and after this collection.

    For Hermitian generators, conjugating $(T^a T^b)_{ij}$ gives
    $(T^b T^a)_{ji}$. The order and endpoints both reverse. The trace of three
    generators can be complex, so its order must also reverse under conjugation.
    These rules belong to shared Idenso algebra and also apply through
    `dirac_adjoint` when a larger expression contains fermion chains.
    """)
    return


@app.cell
def _(Representation, TensorExpression, mo):
    generator = TensorExpression.t(8, 3)
    fundamental = Representation.cof(3)
    explicit_word = generator("a", "i", "k") * generator("b", "k", "j")
    color_word = explicit_word.collect_chains(fundamental)
    color_conjugate = color_word.spenso_conjugate()
    _expected = (generator("b", "j", "k") * generator("a", "k", "i")).collect_chains(
        fundamental
    )
    assert color_conjugate.to_expression() == _expected.to_expression()
    assert (
        color_conjugate.spenso_conjugate().to_expression() == color_word.to_expression()
    )
    assert color_conjugate.simplify_color().to_expression() == (
        explicit_word.spenso_conjugate().simplify_color().to_expression()
    )
    assert color_word.dirac_adjoint().to_expression() == color_conjugate.to_expression()
    mo.vstack(
        [
            mo.md("**Explicit indexed network**"),
            explicit_word,
            mo.md("**Collected chain**"),
            color_word,
            mo.md("**Complex conjugate**"),
            color_conjugate,
        ]
    )
    return color_conjugate, color_word, fundamental, generator


@app.cell
def _(E, color_conjugate, color_word, mo):
    color_norm = (
        (color_word * color_conjugate)
        .simplify_color()
        .to_cof_dimension_invariants()
        .simplify_metrics()
    )
    assert color_norm.is_scalar
    assert color_norm.to_expression() == E("16/3")
    mo.vstack(
        [
            mo.md(
                r"**Summed norm:** $\sum_{a,b,i,j}|(T^aT^b)_{ij}|^2=16/3$ for $T_R=1/2$."
            ),
            color_norm,
        ]
    )
    return (color_norm,)


@app.cell
def _(fundamental, generator, mo):
    _explicit_loop = (
        generator("a", "i", "j") * generator("b", "j", "k") * generator("c", "k", "i")
    )
    color_trace = _explicit_loop.collect_chains(fundamental)
    color_trace_conjugate = color_trace.spenso_conjugate()
    assert color_trace_conjugate.to_expression() != color_trace.to_expression()
    assert (
        color_trace_conjugate.spenso_conjugate().to_expression()
        == color_trace.to_expression()
    )
    assert color_trace_conjugate.simplify_color().to_expression() == (
        _explicit_loop.spenso_conjugate().simplify_color().to_expression()
    )
    mo.vstack(
        [
            mo.md("**Three-generator trace**"),
            color_trace,
            mo.md("**Conjugate trace**"),
            color_trace_conjugate,
            mo.md(
                "Compact and expanded forms agree, and conjugating twice restores each original expression."
            ),
        ]
    )
    return color_trace, color_trace_conjugate


@app.cell
def _(E, S, TensorExpression, color_trace, color_trace_conjugate, mo):
    _cyclic, _sym, _antisym, _factors = S(
        "spenso::cyclic", "spenso::sym", "spenso::antisym", "factors__"
    )
    symmetric_color_trace = TensorExpression(
        color_trace.to_expression().replace(_cyclic(_factors), _sym(_factors))
    )
    antisymmetric_color_trace = TensorExpression(
        color_trace.to_expression().replace(_cyclic(_factors), _antisym(_factors))
    )
    assert (
        symmetric_color_trace.spenso_conjugate().to_expression()
        == symmetric_color_trace.to_expression()
    )
    assert (
        antisymmetric_color_trace.spenso_conjugate() + antisymmetric_color_trace
    ).simplify_color().to_expression() == E("0")
    assert (
        symmetric_color_trace.simplify_color().to_expression()
        == ((color_trace + color_trace_conjugate) / 2).simplify_color().to_expression()
    )
    assert (
        antisymmetric_color_trace.simplify_color().to_expression()
        == ((color_trace - color_trace_conjugate) / 2).simplify_color().to_expression()
    )
    mo.vstack(
        [
            mo.md(r"""
            **Symmetric and antisymmetric generator groups**

            Cyclic invariance reduces the six permutations of three generators
            to two orientations. Their half-sum is the symmetric, real trace;
            their half-difference is the antisymmetric, purely imaginary trace.
            Conjugation preserves the compact projectors, including their signs.
            """),
            symmetric_color_trace,
            antisymmetric_color_trace,
            mo.md("**Conjugate antisymmetric trace**"),
            antisymmetric_color_trace.spenso_conjugate(),
        ]
    )
    return antisymmetric_color_trace, symmetric_color_trace


if __name__ == "__main__":
    app.run()
