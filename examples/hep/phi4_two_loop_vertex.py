import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Two-loop phi4 coupling renormalization")


@app.cell
def _():
    from math import prod

    import marimo as mo
    from ibp_phi4 import app as self_energy_app
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica.community import hep
    from symbolica.community.spenso import TensorExpression

    return (
        E,
        Matrix,
        Replacement,
        S,
        Symbol,
        TensorExpression,
        hep,
        mo,
        prod,
        self_energy_app,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-loop $\phi^4$ coupling renormalization
    [Browse all notebooks](/) · [Two-loop self-energy](/?file=hep/ibp_phi4.py) · [One-loop renormalization](/?file=hep/phi4_renormalization.py)

    Reproduce the [FeynCalc two-loop four-point example](https://feyncalc.github.io/FeynCalcExamples/Phi4/TwoLoops/Renormalization-SSSS) with 12 generated bare diagrams, 12 one-loop counterterm diagrams and one local second-order counterterm.

    Marimo composition supplies the model, vacuum integral family, verified momentum mappings, analytic master inputs, counterterm rules and computed two-loop field constant from the self-energy notebook. The vertex calculation uses the same Feynkit and native IBP components.

    Take the zeroth Taylor coefficient in external momenta with a nonzero scalar mass. This preserves the local UV divergence without introducing an infrared singularity. Amplitudes below are in units of $ig^3/(16\pi^2)^2$.
    """)


@app.cell
async def _(self_energy_app):
    _shared_calculation = await self_energy_app.embed()
    K = _shared_calculation.defs["K"]
    P = _shared_calculation.defs["P"]
    coordinates = _shared_calculation.defs["coordinates"]
    coupling = _shared_calculation.defs["coupling"]
    ct_coordinate = _shared_calculation.defs["ct_coordinate"]
    ct_integral = _shared_calculation.defs["ct_integral"]
    ct_model = _shared_calculation.defs["ct_model"]
    d = _shared_calculation.defs["d"]
    den = _shared_calculation.defs["den"]
    dim = _shared_calculation.defs["dim"]
    dimension = _shared_calculation.defs["dimension"]
    edge_ = _shared_calculation.defs["edge_"]
    eps = _shared_calculation.defs["eps"]
    family = _shared_calculation.defs["family"]
    field1 = _shared_calculation.defs["field1"]
    field2 = _shared_calculation.defs["field2"]
    h = _shared_calculation.defs["h"]
    ibp = _shared_calculation.defs["ibp"]
    index = _shared_calculation.defs["index"]
    integral = _shared_calculation.defs["integral"]
    k1 = _shared_calculation.defs["k1"]
    k2 = _shared_calculation.defs["k2"]
    log_4pi = _shared_calculation.defs["log_4pi"]
    log_mass = _shared_calculation.defs["log_mass"]
    loop_coupling = _shared_calculation.defs["loop_coupling"]
    mass1 = _shared_calculation.defs["mass1"]
    mass_ = _shared_calculation.defs["mass_"]
    mass_squared = _shared_calculation.defs["mass_squared"]
    mink = _shared_calculation.defs["mink"]
    model = _shared_calculation.defs["model"]
    model_coupling = _shared_calculation.defs["model_coupling"]
    model_mass = _shared_calculation.defs["model_mass"]
    mom_ = _shared_calculation.defs["mom_"]
    parametric = _shared_calculation.defs["parametric"]
    particle = _shared_calculation.defs["particle"]
    pi = _shared_calculation.defs["pi"]
    quad_ = _shared_calculation.defs["quad_"]
    reference_basis = _shared_calculation.defs["reference_basis"]
    symmetries = _shared_calculation.defs["symmetries"]
    tadpole_coefficient = _shared_calculation.defs["tadpole_coefficient"]
    tadpole_family = _shared_calculation.defs["tadpole_family"]
    tadpole_poles = _shared_calculation.defs["tadpole_poles"]
    tadpole_squared = _shared_calculation.defs["tadpole_squared"]
    tadpole_squared_poles = _shared_calculation.defs["tadpole_squared_poles"]
    vacuum_integral = _shared_calculation.defs["vacuum_integral"]
    vacuum_poles = _shared_calculation.defs["vacuum_poles"]
    vertex1 = _shared_calculation.defs["vertex1"]
    vertex2 = _shared_calculation.defs["vertex2"]
    zero = _shared_calculation.defs["zero"]
    zg1 = _shared_calculation.defs["zg1"]
    zm1 = _shared_calculation.defs["zm1"]
    zphi2 = _shared_calculation.defs["zphi2"]
    return (
        K,
        P,
        coordinates,
        coupling,
        ct_coordinate,
        ct_integral,
        ct_model,
        d,
        den,
        dim,
        dimension,
        edge_,
        eps,
        family,
        field1,
        field2,
        h,
        ibp,
        index,
        integral,
        k1,
        k2,
        log_4pi,
        log_mass,
        loop_coupling,
        mass1,
        mass_,
        mass_squared,
        mink,
        model,
        model_coupling,
        model_mass,
        mom_,
        parametric,
        particle,
        pi,
        quad_,
        reference_basis,
        symmetries,
        tadpole_coefficient,
        tadpole_family,
        tadpole_poles,
        tadpole_squared,
        tadpole_squared_poles,
        vacuum_integral,
        vacuum_poles,
        vertex1,
        vertex2,
        zero,
        zg1,
        zm1,
        zphi2,
    )


@app.cell
def _(
    E,
    K,
    Replacement,
    Symbol,
    coordinates,
    d,
    den,
    dim,
    edge_,
    eps,
    family,
    ibp,
    index,
    integral,
    k1,
    k2,
    log_4pi,
    log_mass,
    mass_,
    mass_squared,
    mo,
    model,
    model_coupling,
    model_mass,
    mom_,
    particle,
    prod,
    quad_,
    reference_basis,
    symmetries,
    tadpole_squared,
    tadpole_squared_poles,
    vacuum_integral,
    vacuum_poles,
    zero,
):
    vertex_diagrams = model.generate_diagrams(
        [particle] * 2,
        [particle] * 2,
        loops=2,
        max_vertices=3,
        maximum_bridges=0,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
    ).diagrams
    assert len(vertex_diagrams) == 12
    vertex_input, vertex_terms = (zero, [])
    for _diagram in vertex_diagrams:
        _numerator = model.expand_couplings(
            _diagram.numerator_expression().to_expression()
        )
        _expanded = _diagram.momentum_basis().route_expression(
            _diagram.uv_expansion(model_mass, numerator=_numerator).to_expression()
        )
        _expanded = (
            _expanded.replace(dim, d)
            .replace(K(0, index), k1(index))
            .replace(K(1, index), -k2(index))
            .replace(model_mass**2, mass_squared)
        )
        for _match in list(_expanded.match(den(edge_, mom_, mass_, quad_))):
            _values = dict(_match)
            _expanded = _expanded.replace(
                den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                family.rewrite_numerator(_values[quad_], coordinates),
            )
        _scalar = (
            (
                _expanded
                * _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
                / (Symbol.I * model_coupling**3)
            )
            .together()
            .expand()
        )
        for _monomial, _coefficient in _scalar.coefficient_list(*coordinates):
            _powers = [
                -int((_monomial.derivative(_x) * _x / _monomial).together())
                for _x in coordinates
            ]
            assert _monomial == prod(
                (
                    _x ** (-_power)
                    for _x, _power in zip(coordinates, _powers, strict=True)
                )
            )
            vertex_terms.append((_powers, _coefficient))
            vertex_input += _coefficient * integral(*_powers)
    assert (
        vertex_input
        - 3 * integral(3, 1, 0) / 2
        - 3 * integral(2, 2, 0) / 4
        - 3 * integral(2, 1, 1)
    ).together() == zero
    vertex_targets = sorted({tuple(_powers) for _powers, _coefficient in vertex_terms})
    vertex_laporta = ibp.reduce_laporta(vertex_targets, max_depth=2)
    _vertex_canonical = {
        tuple(_powers): min(
            tuple(_mapping.map_powers(_powers)) for _mapping in symmetries
        )
        for _powers in vertex_laporta.residuals
    }
    vertex_reductions = {}
    for _target in vertex_targets:
        _terms = vertex_laporta.reduce(_target)
        assert all(
            (
                _vertex_canonical[tuple(_powers)] in reference_basis
                for _powers, _coefficient in _terms
            )
        )
        vertex_reductions[_target] = sum(
            (
                _coefficient * reference_basis[_vertex_canonical[tuple(_powers)]]
                for _powers, _coefficient in _terms
            ),
            zero,
        ).together()
    _vertex_reduced = vertex_input.replace_multiple(
        [
            Replacement(integral(*_target), _value)
            for _target, _value in vertex_reductions.items()
        ]
    )
    vertex_bare_uv = (
        (
            -(1 + 2 * eps * log_4pi)
            * _vertex_reduced.replace(d, 4 - 2 * eps)
            .replace(tadpole_squared, tadpole_squared_poles)
            .replace(vacuum_integral, vacuum_poles)
        )
        .series(eps, 0, -1)
        .to_expression()
        .expand()
    )
    assert (
        vertex_bare_uv
        + E("9/4") / eps**2
        - (E("9/2") * (log_mass - log_4pi) - E("3/4")) / eps
    ).together() == zero
    mo.vstack(
        [
            mo.md("## Bare four-point UV reduction"),
            vertex_input,
            mo.md("**Computed native IBP reductions**"),
            *[
                mo.hstack([integral(*_target), mo.md("$\\longrightarrow$"), _value])
                for _target, _value in vertex_reductions.items()
            ],
            mo.md("**Bare two-loop UV poles**"),
            vertex_bare_uv,
        ]
    )
    return (
        vertex_diagrams,
        vertex_input,
        vertex_terms,
        vertex_targets,
        vertex_laporta,
        vertex_reductions,
        vertex_bare_uv,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Counterterms retain UV-finite integrals
    A mass insertion produces a UV-finite one-loop integral, but its coefficient contains a renormalization pole. Dropping that integral would lose part of the two-loop counterterm. Keep the entire zeroth-order Taylor coefficient and reduce both doubled and tripled tadpoles with the parametric IBP recurrence.

    The scalar mass and logarithms cancel between bare graphs and counterterm insertions. The generated local vertex contains both the coupling and field renormalization constants; the previously computed field constant is required to determine $Z_g$.
    """)


@app.cell
def _(
    E,
    K,
    Matrix,
    P,
    S,
    Symbol,
    TensorExpression,
    coupling,
    ct_coordinate,
    ct_integral,
    ct_model,
    d,
    den,
    dimension,
    edge_,
    eps,
    field1,
    field2,
    h,
    index,
    k1,
    log_4pi,
    log_mass,
    loop_coupling,
    mass1,
    mass_,
    mass_squared,
    mink,
    mo,
    model_coupling,
    model_mass,
    mom_,
    parametric,
    particle,
    pi,
    quad_,
    tadpole_coefficient,
    tadpole_family,
    tadpole_poles,
    vertex1,
    vertex2,
    vertex_bare_uv,
    zero,
    zg1,
    zm1,
    zphi2,
):
    vertex_ct_diagrams = {}
    vertex_ct_input, _vertex_tree_ct = (zero, zero)
    _external_index = S("ibp_phi4::external_")
    for _loops, _order in [(1, 1), (0, 2)]:
        _result = ct_model.generate_diagrams(
            [particle.name] * 2,
            [particle.name] * 2,
            loops=_loops,
            max_vertices=3 if _loops else 1,
            coupling_orders={"CT": _order},
            maximum_bridges=0,
            self_energy=None,
            tadpoles=None,
            zero_snails=None,
            numerator_grouping=None,
            progress=None,
        )
        vertex_ct_diagrams[_loops] = _result.diagrams
        assert len(_result.diagrams) == (12 if _loops else 1)
        for _diagram in _result.diagrams:
            _numerator = ct_model.expand_couplings(
                _diagram.numerator_expression().to_expression()
            )
            _numerator = TensorExpression(_numerator.expand()).to_dots().to_expression()
            _numerator = _diagram.momentum_basis().route_expression(_numerator)
            if _loops:
                _numerator /= _diagram.denominator_expression(
                    dimension=d, in_lmb=True
                ).to_expression()
            _numerator = (
                _numerator.replace(mink(dimension), mink(d))
                .replace(P(_external_index, index), zero)
                .replace(K(0, index), k1(index))
                .replace(model_mass**2, mass_squared)
            )
            for _match in list(_numerator.match(den(edge_, mom_, mass_, quad_))):
                _values = dict(_match)
                _numerator = _numerator.replace(
                    den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                    tadpole_family.rewrite_numerator(_values[quad_], [ct_coordinate]),
                )
            _numerator *= (
                _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
            )
            if _loops:
                _scalar = (
                    (
                        tadpole_family.rewrite_numerator(_numerator, [ct_coordinate])
                        / (model_coupling**2 * h)
                    )
                    .together()
                    .expand()
                )
                for _monomial, _coefficient in _scalar.coefficient_list(ct_coordinate):
                    _power = -int(
                        (
                            _monomial.derivative(ct_coordinate)
                            * ct_coordinate
                            / _monomial
                        ).together()
                    )
                    assert _monomial == ct_coordinate ** (-_power)
                    vertex_ct_input += _coefficient * ct_integral(_power)
            else:
                _vertex_tree_ct += _numerator / (Symbol.I * model_coupling * h**2)
    assert (
        vertex_ct_input
        - 3 * (vertex1 + field1) * ct_integral(2)
        - 3 * mass_squared * mass1 * ct_integral(3)
    ).together() == zero
    assert (
        _vertex_tree_ct + vertex2 + 2 * field2 + 2 * vertex1 * field1 + field1**2
    ).together() == zero
    _tripled_tadpole = parametric.reduce([3], integral=ct_integral).replace(
        ct_integral(2), tadpole_coefficient * ct_integral(1)
    )
    _vertex_ct_reduced = vertex_ct_input.replace(
        ct_integral(3), _tripled_tadpole
    ).replace(ct_integral(2), tadpole_coefficient * ct_integral(1))
    _vertex_ct_reduced = (
        _vertex_ct_reduced.replace(field1, zero)
        .replace(vertex1, zg1)
        .replace(mass1, zm1)
    )
    vertex_ct_uv = (
        (
            (1 + eps * log_4pi)
            * _vertex_ct_reduced.replace(d, 4 - 2 * eps).replace(
                ct_integral(1), tadpole_poles
            )
        )
        .series(eps, 0, -1)
        .to_expression()
        .expand()
    )
    assert (
        vertex_ct_uv
        - E("9/2") / eps**2
        - (-E("3/4") + E("9/2") * (log_4pi - log_mass)) / eps
    ).together() == zero
    mass_insertion_pole = (
        (vertex_ct_input.expand().coefficient(mass1) * zm1)
        .replace(ct_integral(3), _tripled_tadpole)
        .replace(d, 4 - 2 * eps)
        .replace(ct_integral(1), tadpole_poles)
        .series(eps, 0, -1)
        .to_expression()
    )
    assert (mass_insertion_pole + E("3/4") / eps).together() == zero
    vertex_loop_uv = (vertex_bare_uv + vertex_ct_uv).expand()
    _vertex_tree_ct = (
        _vertex_tree_ct.replace(field1, zero).replace(field2, zphi2).expand()
    )
    vertex_matrix = Matrix.from_linear(1, 1, [_vertex_tree_ct.coefficient(vertex2)])
    _vertex_rhs = Matrix.vec([-vertex_loop_uv - _vertex_tree_ct.replace(vertex2, zero)])
    _vertex_solution = vertex_matrix.solve(_vertex_rhs)
    zg2 = _vertex_solution[0, 0].to_expression().expand()
    assert (zg2 - E("9/4") / eps**2 + E("17/12") / eps).together() == zero
    assert (vertex_loop_uv + _vertex_tree_ct.replace(vertex2, zg2)).together() == zero
    zg = (1 + loop_coupling * zg1 + loop_coupling**2 * zg2).expand()
    assert (
        zg
        - 1
        - 3 * coupling / (32 * pi**2 * eps)
        - 9 * coupling**2 / (1024 * pi**4 * eps**2)
        + 17 * coupling**2 / (3072 * pi**4 * eps)
    ).together() == zero
    mo.vstack(
        [
            mo.md("**Generated one-loop counterterm integral combination**"),
            vertex_ct_input,
            mo.md("**Tripled-tadpole recurrence**"),
            _tripled_tadpole,
            mo.md("**One-loop counterterm poles**"),
            vertex_ct_uv,
            mo.md("**Pole from the UV-finite mass insertion**"),
            mass_insertion_pole,
            mo.md("**Sum before local subtraction**"),
            vertex_loop_uv,
            mo.md("**Generated local counterterm matrix**"),
            vertex_matrix,
            mo.md("**Coupling renormalization**"),
            zg,
            mo.md(
                "All poles cancel exactly after inserting the solved coupling constant."
            ),
        ]
    )
    return (
        vertex_ct_diagrams,
        vertex_ct_input,
        vertex_ct_uv,
        mass_insertion_pole,
        vertex_loop_uv,
        vertex_matrix,
        zg,
        zg2,
    )


@app.cell
def _(E, S, eps, mo, zero, zg1, zg2):
    a = S("ibp_phi4::a")
    _Zg = 1 + a * zg1 + a**2 * zg2
    beta = (
        (-2 * eps * a * _Zg / (_Zg + a * _Zg.derivative(a)))
        .series(a, 0, 3)
        .to_expression()
        .series(eps, 0, 0)
        .to_expression()
        .expand()
    )
    assert (beta - 3 * a**2 + E("17/3") * a**3).together() == zero
    mo.vstack(
        [
            mo.md("## Beta function from scale independence"),
            mo.md(
                "For $a=g/(16\\pi^2)$, require the bare coupling $\\mu^{2\\epsilon}a Z_g$ to be scale independent. Expanding $-2\\epsilon aZ_g/(Z_g+a\\partial_a Z_g)$ derives the four-dimensional beta function:"
            ),
            beta,
        ]
    )
    return a, beta


@app.cell
def _(mo):
    diagram_group = mo.ui.dropdown(
        ["Bare two-loop", "One-loop counterterms", "Local two-loop counterterm"],
        value="Bare two-loop",
        label="Generated diagrams",
    )
    mo.vstack([diagram_group])
    return diagram_group


@app.cell
def _(diagram_group, mo, vertex_ct_diagrams, vertex_diagrams):
    selected_diagrams = {
        "Bare two-loop": vertex_diagrams,
        "One-loop counterterms": vertex_ct_diagrams[1],
        "Local two-loop counterterm": vertex_ct_diagrams[0],
    }[diagram_group.value]
    mo.vstack(
        [
            mo.md(f"## {diagram_group.value}: {len(selected_diagrams)} diagrams"),
            *[
                mo.hstack(selected_diagrams[_i : _i + 3])
                for _i in range(0, len(selected_diagrams), 3)
            ],
        ]
    )
    return selected_diagrams


if __name__ == "__main__":
    app.run()
