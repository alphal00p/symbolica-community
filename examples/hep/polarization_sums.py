import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Polarization sums in D dimensions")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.spenso import TensorExpression

    _set_namespace("polarization")

    return E, S, TensorExpression, hep, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polarization sums in $D$ dimensions

    [Browse all notebooks](/) · [Light-cone soft radiation](/?file=hep/soft_function.py) · [Gluon scattering](/?file=hep/gluon_scattering.py) ·
    [Massive Dirac spin states](/?file=hep/polarized_spin.py) ·
    [Gluon scattering from quarks](/?file=hep/qcd_gluons.py)

    Query a particle by name and call `particle.spin_sum(p, i, j, dimension=D)`.
    Massless vectors have $D-2$ physical states; massive vectors have $D-1$.
    `average=True` divides by this count. Four dimensions remain the default.

    The massless projector uses an axial reference $n$ with $p\cdot n=\chi\ne0$.
    Its $n^2=\zeta$ term is retained, so the reference need not be null.
    The massive projector includes the longitudinal Proca polarization.

    A fixed four-dimensional averaging factor is also useful when comparing
    dimensional continuations. It changes only the normalization: the projector
    still uses $D$-dimensional Lorentz slots. For example, the
    [four-gluon FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/GlGl-GlGl)
    divides each incoming $D$-dimensional gluon sum by two.
    """)


@app.cell
def _(mo):
    species = mo.ui.dropdown(
        {"Gluon": "g", "Photon": "a", "Massive Z": "Z"}, value="Gluon", label="Particle"
    )
    dimension_choice = mo.ui.dropdown(
        ["Symbolic D", "4", "6"], value="Symbolic D", label="Lorentz dimension"
    )
    normalization = mo.ui.dropdown(
        [
            "Sum over states",
            "Average in D dimensions",
            "Average over four-dimensional states",
        ],
        value="Average in D dimensions",
        label="Normalization",
    )
    mo.hstack([species, dimension_choice, normalization])
    return dimension_choice, normalization, species


@app.cell
def _(E, S, TensorExpression, dimension_choice, hep, normalization, species):
    model = hep.Model.standard_model()
    particle = model.particle(species.value)
    D = S("D") if dimension_choice.value == "Symbolic D" else E(dimension_choice.value)
    p, n, i, j, chi, zeta = S(
        "p",
        "n",
        "i",
        "j",
        "chi",
        "zeta",
    )
    mink, metric = S("spenso::mink", "spenso::g")
    mass_squared = (
        E("0") if particle.is_massless else S("UFO::" + particle.mass_parameter) ** 2
    )
    kinematics = (
        hep.Kinematics(D)
        .with_scalar_product(p, p, mass_squared)
        .with_scalar_product(p, n, chi)
        .with_scalar_product(n, n, zeta)
    )
    options = {
        "dimension": D,
        "reference": n if particle.is_massless else None,
        "average": normalization.value == "Average in D dimensions",
    }
    scale = E("1")
    if normalization.value == "Average over four-dimensional states":
        scale /= 2 if particle.is_massless else 3
    projector = scale * particle.spin_sum(p, i, j, **options)
    state_count = D - (2 if particle.is_massless else 1)
    density = TensorExpression(projector)
    trace = kinematics.apply(
        TensorExpression((projector * metric(mink(D, i), mink(D, j))).expand())
        .simplify_metrics()
        .to_dots()
    ).together()
    expected_trace = -scale if options["average"] else -scale * state_count
    assert (trace - expected_trace).together() == E("0")
    transverse = kinematics.apply(
        TensorExpression((projector * p(mink(D, i))).expand())
        .simplify_metrics()
        .to_dots()
    ).together()
    assert transverse == E("0")
    if particle.is_massless:
        axial = kinematics.apply(
            TensorExpression((projector * n(mink(D, i))).expand())
            .simplify_metrics()
            .to_dots()
        ).together()
        assert axial == E("0")
    return (
        D,
        density,
        i,
        j,
        kinematics,
        mink,
        options,
        p,
        particle,
        projector,
        scale,
        state_count,
        trace,
        transverse,
    )


@app.cell
def _(density, mo, particle, state_count, trace, transverse):
    mo.vstack(
        [
            mo.md(f"**{particle.name}: projector and physical state count**"),
            density,
            mo.tree(
                {
                    "Physical states": state_count,
                    "Lorentz trace": trace,
                    "Momentum contraction": transverse,
                }
            ),
            mo.md(
                r"With the $(+,-,\ldots,-)$ metric, the projector trace is the negative state count. Both the momentum contraction and, for massless vectors, the reference contraction vanish exactly."
            ),
        ]
    )


@app.cell
def _(D, E, S, i, j, mink, mo, options, p, particle, projector, scale):
    _ket, _bra = S("gammalooprs::ϵ", "gammalooprs::ϵbar")
    _pair = _ket(7, mink(D, i)) * _bra(7, mink(D, j))
    _spectator = _ket(8, mink(D, i))
    sewn = scale * particle.sum_spins(_pair + _spectator, p, edge=7, **options)
    assert (sewn - projector - scale * _spectator).expand() == E("0")
    mo.vstack(
        [
            mo.md(
                "**Sewing-aware completeness**\n\n`sum_spins(..., edge=7, dimension=D)` replaces the paired wavefunctions on edge 7. The unpaired wavefunction on edge 8 remains present. This is the same replacement used by GammaLoop."
            ),
            mo.tree({"Before": _pair + _spectator, "After": sewn}),
        ]
    )
    return (sewn,)


if __name__ == "__main__":
    app.run()
