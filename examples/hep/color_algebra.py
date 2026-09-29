import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Color chains and conjugation")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.spenso import (
        AUTO,
        ColorSimplifySettings,
        Representation,
        TensorExpression,
        chain,
    )

    _set_namespace("color_algebra")

    return (
        AUTO,
        ColorSimplifySettings,
        E,
        Representation,
        S,
        TensorExpression,
        chain,
        mo,
    )


@app.cell
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


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
def _(ColorSimplifySettings, Representation, TensorExpression):
    generator = TensorExpression.t(8, 3)
    fundamental = Representation.cof(3)
    explicit_word = generator("a", "i", "k") * generator("b", "k", "j")
    explicit_conjugate = explicit_word.dirac_adjoint()
    _expected_explicit = generator("a", "k", "i") * generator("b", "j", "k")
    assert explicit_word.rank == explicit_conjugate.rank == 4
    assert explicit_conjugate.to_expression() == _expected_explicit.to_expression()
    assert (
        explicit_conjugate.dirac_adjoint().to_expression()
        == explicit_word.to_expression()
    )
    compact_color_settings = ColorSimplifySettings(
        evaluate_traces=False, expand_cross_chain_fierz=False
    )
    color_settings = ColorSimplifySettings(substitute_cof_dimension_invariants=True)
    color_word = explicit_word.simplify_color(compact_color_settings).to_expression()
    color_conjugate = color_word.dirac_adjoint()
    _expected = _expected_explicit.simplify_color(
        compact_color_settings
    ).to_expression()
    assert color_conjugate.to_expression() == _expected.to_expression()
    assert color_conjugate.dirac_adjoint().to_expression() == color_word.to_expression()
    assert color_conjugate.simplify_color().to_expression().to_expression() == (
        explicit_conjugate.simplify_color().to_expression().to_expression()
    )

    return (
        color_conjugate,
        color_settings,
        color_word,
        compact_color_settings,
        explicit_conjugate,
        explicit_word,
        fundamental,
        generator,
    )


@app.cell
def _(color_conjugate, color_word, explicit_conjugate, explicit_word, mo):
    mo.vstack(
        [
            mo.md("**Explicit indexed network**"),
            explicit_word,
            mo.md("**Explicit conjugate**"),
            explicit_conjugate,
            mo.md("**Compact chain**"),
            color_word,
            mo.md("**Complex conjugate**"),
            color_conjugate,
        ]
    )
    return


@app.cell
def _(
    E,
    S,
    color_conjugate,
    color_settings,
    color_word,
    explicit_conjugate,
    explicit_word,
    mo,
):
    ket_word = explicit_word.wrap_indices(S("ket"), dummies_only=True)
    bra_word = explicit_conjugate.wrap_indices(S("bra"), dummies_only=True)
    assert ket_word.structure.slots == explicit_word.structure.slots
    assert bra_word.structure.slots == explicit_conjugate.structure.slots
    scoped_norm = ket_word * bra_word
    color_norm = scoped_norm.simplify_color(color_settings).to_expression()
    assert color_norm.is_scalar
    assert color_norm.to_expression() == E("16/3")
    assert (color_word * color_conjugate).simplify_color(
        color_settings
    ).to_expression().to_expression() == E("16/3")
    _network = scoped_norm.to_network()
    _network.execute()
    numeric_norm = _network.result_scalar()
    assert abs(complex(numeric_norm) - 16 / 3) < 1e-12
    mo.vstack(
        [
            mo.md(r"""
            **Independent dummy indices in a squared word**

            Each explicit word sums over its own internal `k`. Multiplying the raw
            expressions would put `k` on four tensor ports. Give the two copies
            distinct dummy scopes before multiplying them. `dummies_only=True`
            keeps the external `a`, `b`, `i`, and `j` labels shared, so those ports
            contract in the norm. Compact chains already hide their internal sums.

            **Summed norm:** $\sum_{a,b,i,j}|(T^aT^b)_{ij}|^2=16/3$
            for $T_R=1/2$. The explicit SU(3) component network checks the same value.
            """),
            ket_word,
            bra_word,
            color_norm,
            numeric_norm,
        ]
    )
    return bra_word, color_norm, ket_word, numeric_norm, scoped_norm


@app.cell
def _(compact_color_settings, generator, mo):
    _explicit_loop = (
        generator("a", "i", "j") * generator("b", "j", "k") * generator("c", "k", "i")
    )
    color_trace = _explicit_loop.simplify_color(compact_color_settings).to_expression()
    color_trace_conjugate = color_trace.dirac_adjoint()
    assert color_trace_conjugate.to_expression() != color_trace.to_expression()
    assert (
        color_trace_conjugate.dirac_adjoint().to_expression()
        == color_trace.to_expression()
    )
    assert color_trace_conjugate.simplify_color().to_expression().to_expression() == (
        _explicit_loop.dirac_adjoint().simplify_color().to_expression().to_expression()
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
def _(Symbols, E, S, TensorExpression, color_trace, color_trace_conjugate, mo):
    _cyclic, _sym, _antisym, _factors = (
        Symbols.cyclic,
        Symbols.symmetric,
        Symbols.antisymmetric,
        S("factors__"),
    )
    symmetric_color_trace = TensorExpression(
        color_trace.to_expression().replace(_cyclic(_factors), _sym(_factors))
    )
    antisymmetric_color_trace = TensorExpression(
        color_trace.to_expression().replace(_cyclic(_factors), _antisym(_factors))
    )
    assert (
        symmetric_color_trace.dirac_adjoint().to_expression()
        == symmetric_color_trace.to_expression()
    )
    assert (
        antisymmetric_color_trace.dirac_adjoint() + antisymmetric_color_trace
    ).simplify_color().to_expression().to_expression() == E("0")
    assert (
        symmetric_color_trace.simplify_color().to_expression().to_expression()
        == ((color_trace + color_trace_conjugate) / 2)
        .simplify_color()
        .to_expression()
        .to_expression()
    )
    assert (
        antisymmetric_color_trace.simplify_color().expand().to_expression()
        == ((color_trace - color_trace_conjugate) / 2)
        .simplify_color()
        .expand()
        .to_expression()
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
            antisymmetric_color_trace.dirac_adjoint(),
        ]
    )
    return


@app.cell
def _(AUTO, E, S, TensorExpression, chain, fundamental, generator, mo):
    _z = S("color_weight_z")
    weighted_color_word = chain(
        fundamental("i"),
        fundamental.dual()("j"),
        _z**2 * generator("a", AUTO, AUTO),
        generator("b", AUTO, AUTO),
    )
    weighted_color_conjugate = weighted_color_word.dirac_adjoint()
    assert (
        weighted_color_conjugate.dirac_adjoint().to_expression()
        == weighted_color_word.to_expression()
    )
    _norm = weighted_color_word * weighted_color_conjugate
    _numeric_norm = TensorExpression(_norm.to_expression().replace(_z, E("1+2𝑖")))
    _network = _numeric_norm.to_network()
    _network.execute()
    weighted_numeric_norm = _network.result_scalar()
    assert abs(complex(weighted_numeric_norm) - 400 / 3) < 1e-12
    mo.vstack(
        [
            mo.md(r"""
            **Scalar weights inside an ordered color chain**

            `chain(...)` accepts weighted generators with unresolved fundamental
            ports. Powers, inverse factors and scalar functions conjugate using
            Symbolica's existing rules; the generator order and endpoints reverse.
            A scalar symbol is not assumed real.

            Here the word is $z^2(T^aT^b)_{ij}$. At $z=1+2i$, its summed norm is
            $|z^2|^2\,16/3=400/3$, checked independently with explicit SU(3) matrices.
            """),
            weighted_color_word,
            weighted_color_conjugate,
            weighted_numeric_norm,
        ]
    )
    return (weighted_numeric_norm,)


if __name__ == "__main__":
    app.run()
