import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Integral families and Feynman parameters",
)


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S

    return E, S, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feynman-parameter polynomials

    For the two-loop denominators $(k+l)^2-m^2$ and $(k-l)\cdot p+\delta$,
    with $p^2=s$, the quadratic loop matrix is singular. Feynkit still returns
    its algebraic polynomials $U=0$ and $F=sxy^2$.
    These polynomials alone do not define an integration formula or establish
    that the integral vanishes.
    """)
    return


@app.cell
def _(E, S, mo):
    from symbolica.community import hep as _hep

    _k, _l, _p, _s, _m2, _delta, _x, _y = S(
        "parameters::k",
        "parameters::l",
        "parameters::p",
        "parameters::s",
        "parameters::m2",
        "parameters::delta",
        "parameters::x",
        "parameters::y",
    )
    _kin = _hep.Kinematics(momenta=[_k, _l, _p]).with_scalar_product(_p, _p, _s)
    parameter_family = _hep.IntegralFamily(
        [_k, _l],
        [_p],
        [
            _kin.scalar_product(_k + _l, _k + _l) - _m2,
            _kin.scalar_product(_k - _l, _p) + _delta,
        ],
        kinematics=_kin,
    )
    parameter_u, parameter_f = parameter_family.symanzik([_x, _y])
    assert parameter_u == E("0")
    assert (parameter_f - _s * _x * _y**2).expand() == E("0")
    mo.vstack(
        [
            mo.md("**Integral family**"),
            parameter_family,
            mo.md("**First polynomial, U**"),
            parameter_u,
            mo.md("**Second polynomial, F**"),
            parameter_f,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
