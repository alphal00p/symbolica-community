import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Symmetric UFO color tensor")


@app.cell
def _():
    import json

    import marimo as mo
    from symbolica import E, S
    from symbolica.community import hep as fk
    from symbolica.community.spenso import TensorExpression

    return E, S, TensorExpression, fk, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Symmetric UFO color tensor

    [Browse all notebooks](/) · [Color algebra](/?file=hep/color_algebra.py)

    Generate a cubic interaction of adjoint scalar fields with color tensor
    $d^{abc}$. The model below reuses the stored scalar mass and coupling
    records, restricts the interactions to one cubic vertex, and gives the
    scalar an adjoint color index. The label `H` refers here to this toy field.

    The UFO color rule `d(1,2,3)` lowers to Idenso's normalized symmetric
    generator trace with the factor required by
    $d^{abc}=2\operatorname{Tr}(\{T^a,T^b\}T^c)$.
    [FeynCalc's reference](https://feyncalc.github.io/FeynCalcBook/SUND.html)
    gives $d^{abc}d^{abc}=40/3$ for SU(3), with $T_R=1/2$.
    The norm displayed below has the scalar coupling removed.
    """)
    return


@app.cell
def _(fk, json, mo):
    _directory = mo.notebook_location()
    if _directory is None:
        raise RuntimeError("Cannot locate hep_sm.json")
    _source = json.loads((_directory.parent / "hep_sm.json").read_text())
    _source["name"] = "adjoint_scalar_example"
    for _particle in _source["particles"]:
        if _particle["pdg_code"] == 25:
            _particle["color"] = 8
    _source["vertex_rules"] = [
        _vertex for _vertex in _source["vertex_rules"] if _vertex["name"] == "V_9"
    ]
    _source["vertex_rules"][0]["color_structures"] = ["d(1,2,3)"]
    adjoint_model = fk.Model.from_json(json.dumps(_source))
    return (adjoint_model,)


@app.cell
def _(adjoint_model, fk, mo):
    symmetric_generated = fk.Generator(adjoint_model).generate(
        fk.Process.amplitude([25, 25], [25]).with_loop_count(0, 0),
        max_vertices=1,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(symmetric_generated.diagrams) == 1
    mo.vstack(
        [
            mo.md("**Generated cubic vertex**"),
            symmetric_generated.diagrams[0],
        ]
    )
    return (symmetric_generated,)


@app.cell
def _(E, S, TensorExpression, adjoint_model, mo, symmetric_generated):
    symmetric_color = TensorExpression(
        symmetric_generated.diagrams[0]
        .numerator_expression()
        .to_expression()
        .replace(S("UFO::GC_69"), E("1"))
    )
    assert len(symmetric_color.interface) == 3
    assert (
        symmetric_color.spenso_conjugate().to_expression()
        == symmetric_color.to_expression()
    )
    # The Symbolica product contracts all three matching explicit adjoint indices.
    symmetric_color_norm = (
        TensorExpression(
            symmetric_color.to_expression()
            * symmetric_color.spenso_conjugate().to_expression()
        )
        .simplify_color()
        .to_cof_dimension_invariants()
    )
    assert symmetric_color_norm.is_scalar
    assert symmetric_color_norm.to_expression() == E("40/3")
    # Initial-state color averaging is independently supplied by the particle API.
    _particle = adjoint_model.particle_by_pdg(25)
    assert TensorExpression(
        _particle.color_sum(S("x"), S("x"), average=True)
    ).simplify_metrics().to_expression() == E("1")
    mo.vstack(
        [
            mo.md("**Shared representation of the generated color tensor**"),
            symmetric_color,
            mo.md("**Summed color norm**"),
            symmetric_color_norm,
            mo.md(
                "Reality, three adjoint ports, and the exact SU(3) normalization all pass."
            ),
        ]
    )
    return symmetric_color, symmetric_color_norm


if __name__ == "__main__":
    app.run()
