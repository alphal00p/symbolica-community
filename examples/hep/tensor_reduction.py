import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Compact scalar products and tensor reduction",
)


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import TensorReducer

    _set_namespace("angular_demo")
    return (E, S, TensorReducer, mo)


@app.cell
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/)
    """)
    return


@app.cell
def _(Symbols, E, S, TensorReducer, mo):
    _D, _k, _p, _mink, _dot = (S("D"), S("k"), S("p"), Symbols.lorentz, Symbols.dot)
    _kc, _pc = _k(_mink(_D)), _p(_mink(_D))
    _reducer = TensorReducer(_D).with_integrated_vector(_kc)
    compact_dot_average = _reducer.reduce(_dot(_kc, _pc) ** 2)
    assert _reducer.reduce(_dot(_kc, _pc)) == E("0")
    assert (compact_dot_average - _dot(_kc, _kc) * _dot(_pc, _pc) / _D).together() == E(
        "0"
    )
    mo.vstack(
        [
            mo.md(r"""
        ## Compact scalar products in tensor reduction

        A vacuum denominator has no preferred direction. Pass $(k\cdot p)^2$
        directly to `TensorReducer.reduce`: the angular average is
        $k^2p^2/D$, while the odd moment vanishes. Compact dots and explicit
        indexed tensors use the same projector.
        """),
            compact_dot_average,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
