import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="One-loop QCD renormalization")


@app.cell
def _():
    import json

    import marimo as mo
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica.community import hep
    from symbolica.community.hep import oneloop
    from symbolica.community.spenso import (
        ColorCasimirSettings,
        Representation,
        TensorExpression,
    )

    return (
        ColorCasimirSettings,
        E,
        Matrix,
        Replacement,
        Representation,
        S,
        Symbol,
        TensorExpression,
        hep,
        json,
        mo,
        oneloop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop QCD renormalization

    [Browse all notebooks](/) · [QED renormalization](/?file=hep/qed_renormalization.py) ·
    [Ghost–gluon vertex](/?file=hep/qcd_ghost_vertex.py) ·
    [Two-loop electron](/?file=hep/electron_two_loop.py)

    Generate the quark, ghost and gluon self-energies and the quark–gluon
    vertex in a **symbolic covariant gauge**. Shared color contractions,
    Dirac traces, UV expansion and native IBP give the six physical
    renormalization constants in the
    [FeynCalc MS/MSbar reference](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/Renormalization2).
    The quark mass remains symbolic throughout.

    We use $D=4-2\epsilon$, $a_4=g_s^2/(16\pi^2)$,
    $T_R=1/2$ and the gluon numerator
    $-i[g^{\mu\nu}-(1-\xi)q^\mu q^\nu/q^2]$.
    The auxiliary squared mass is $M=m_{\rm UV}^2>0$.
    The analytic tadpole pole $I(1)=M/\epsilon+O(1)$ is a supplied input,
    checked with OneLOop. IBP determines its rational coefficients.

    The full UV expansion retains the mass correction terms. They cancel
    auxiliary mass poles, so our two auxiliary mass counterterms vanish.
    A second calculation follows the [selective infrared rearrangement reference](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/Renormalization)
    directly, retaining the quark mass and adding an auxiliary mass only to
    massless propagators. It reproduces the nonzero gluon auxiliary counterterm;
    the six physical constants agree between prescriptions. Here **eight generated counterterm diagrams** supply the
    matching equations, and the controls compare MS and MSbar subtraction.
    The finite column contains subtraction constants, not finite loop amplitudes.
    """)


@app.cell(hide_code=True)
def _(
    ColorCasimirSettings,
    E,
    Representation,
    S,
    Symbol,
    TensorExpression,
    hep,
    json,
    mo,
    oneloop,
):
    _location = mo.notebook_location()
    if _location is None:
        raise RuntimeError("Cannot locate hep_sm.json")
    model = hep.Model(_location.parent / "hep_sm.json")
    D, epsilon, M, _mUV, coordinate, integral, xi, s, Nf = S(
        "qcd_ren::D",
        "qcd_ren::eps",
        "qcd_ren::M",
        "qcd_ren::mUV",
        "qcd_ren::d0",
        "qcd_ren::I",
        "qcd_ren::xi",
        "qcd_ren::s",
        "qcd_ren::Nf",
    )
    Nc, dA, CF, CA = S("qcd_ren::Nc", "qcd_ren::dA", "qcd_ren::CF", "qcd_ren::CA")
    _specification = json.loads(model.to_json())
    for _propagator in _specification["propagators"]:
        if _propagator["particle"] == "g":
            _propagator["numerator"] = (
                "-1𝑖*(UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
                "-(1-qcd_ren::xi)*UFO::P(UFO::idx(1,1))*UFO::P(UFO::idx(1,2))"
                "/spenso::dot(UFO::P(spenso::mink(4)),UFO::P(spenso::mink(4))))"
            )
    model = hep.Model.from_json(json.dumps(_specification))
    K, P, mass, gs = S("gammalooprs::K", "gammalooprs::P", "UFO::MB", "UFO::G")
    mink, bis, gamma, metric = S(
        "spenso::mink", "spenso::bis", "spenso::gamma", "spenso::g"
    )
    cof, coad, cas, trace_index = S(
        "spenso::cof", "spenso::coad", "spenso::cas", "spenso::idx"
    )
    index, dim, wave, mu, nu, left, right = S(
        "qcd_ren::index_",
        "qcd_ren::dim_",
        "qcd_ren::wave_",
        "qcd_ren::mu",
        "qcd_ren::nu",
        "qcd_ren::left_",
        "qcd_ren::right_",
    )
    den, edge_, mom_, mass_, quad_ = S(
        "gammalooprs::denom",
        "qcd_ren::edge_",
        "qcd_ren::mom_",
        "qcd_ren::mass_",
        "qcd_ren::quad_",
    )
    _ordering, _value, _arguments = S(
        "feynkit_generator_factor::ExternalFermionOrderingSign",
        "qcd_ren::value_",
        "qcd_ren::arguments___",
    )
    kinematics = hep.Kinematics(D, momenta=[K(0), P(0)]).with_scalar_product(
        P(0), P(0), s
    )
    vacuum = hep.Kinematics(D, momenta=[K(0)])
    family = hep.IntegralFamily(
        [K(0)], [], [vacuum.scalar_product(K(0), K(0)) - M], kinematics=vacuum
    )
    reducer = hep.TensorReducer(D).with_integrated_vector(K(0, mink(D)))
    gmunu = metric(mink(D, mu), mink(D, nu))
    ppmunu = P(0, mink(D, mu)) * P(0, mink(D, nu))
    zero, one = E("0"), E("1")
    quark, gluon, ghost = (model.particle_by_pdg(pdg) for pdg in (5, 21, 9000005))
    interaction_rules = {}
    for _label, _particles in [
        ("qqg", [quark.antiname, quark.name, gluon.name]),
        ("ccg", [ghost.antiname, ghost.name, gluon.name]),
        ("ggg", [gluon.name] * 3),
        ("gggg", [gluon.name] * 4),
    ]:
        interaction_rules[_label] = [
            vertex
            for vertex in model.vertex_rules
            if sorted(vertex.particles) == sorted(_particles)
        ]
        assert len(interaction_rules[_label]) == 1
    _parts, diagrams, irr_inputs = {}, {}, {}
    for _kind, _incoming, _outgoing, _loops, _vertices, _count in [
        ("tree", [quark], [gluon, quark], 0, interaction_rules["qqg"], 1),
        ("quark", [quark], [quark], 1, interaction_rules["qqg"], 1),
        ("ghost", [ghost], [ghost], 1, interaction_rules["ccg"], 1),
        ("gluon_loop", [gluon], [gluon], 1, interaction_rules["ggg"], 1),
        ("ghost_loop", [gluon], [gluon], 1, interaction_rules["ccg"], 1),
        ("quark_loop", [gluon], [gluon], 1, interaction_rules["qqg"], 1),
        ("tadpole", [gluon], [gluon], 1, interaction_rules["gggg"], 1),
        (
            "vertex",
            [quark],
            [gluon, quark],
            1,
            interaction_rules["qqg"] + interaction_rules["ggg"],
            2,
        ),
    ]:
        _generated = model.generate_diagrams(
            _incoming,
            _outgoing,
            loops=_loops,
            max_vertices=len(_incoming) + len(_outgoing) - 2 + 2 * _loops,
            maximum_bridges=0,
            vertex_allow=_vertices,
            self_energy=None,
            tadpoles=None,
            zero_snails=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == _count
        diagrams[_kind] = _generated.diagrams
        for _number, _diagram in enumerate(_generated.diagrams):
            _ports = {}
            if _kind != "ghost":
                for _edge in _diagram.external_edges:
                    _is_gluon = _incoming == [gluon] or (
                        _kind in ("tree", "vertex") and _edge.external_index == 1
                    )
                    _rep = mink if _is_gluon else bis
                    _ports[_edge.external_index] = dict(
                        next(
                            _diagram.projector_expression().match(
                                wave(_edge.id, _rep(4, index)), max_level=0
                            )
                        )
                    )[index]
            _numerator = model.expand_couplings(
                _diagram.numerator_expression().to_expression()
            )
            if _kind in ("tree", "vertex"):
                # Conjugate the tree's color tensor; the generated tree fixes the norm.
                _color_projector = (
                    TensorExpression.t(dA, Nc)(_ports[1], _ports[0], _ports[2])
                    .spenso_conjugate()
                    .to_expression()
                )
                _numerator = (
                    _numerator.replace(cof(3, index), cof(Nc, index)).replace(
                        coad(8, index), coad(dA, index)
                    )
                    * _color_projector
                )
                _settings = ColorCasimirSettings(rewrite_fundamental_dimension=False)
            else:
                _particle = _incoming[0]
                _rep, _numeric_dim, _symbolic_dim = (
                    (cof, 3, Nc) if _kind == "quark" else (coad, 8, dA)
                )
                _color_slots = [
                    slot.dual().to_expression()
                    for slot in TensorExpression(_numerator).interface
                    if slot.to_expression().matches(_rep(_numeric_dim, index))
                ]
                assert len(_color_slots) == 2
                _color_indices = dict(
                    next(
                        metric(*_color_slots).match(
                            _particle.color_sum(left, right), max_level=0
                        )
                    )
                )
                _color_projector = _particle.color_sum(
                    _color_indices[left], _color_indices[right]
                )
                _numerator = (
                    (_numerator * _color_projector / _symbolic_dim)
                    .replace(cof(3, index), cof(Nc, index))
                    .replace(coad(8, index), coad(dA, index))
                )
                _settings = ColorCasimirSettings()
            _numerator = (
                TensorExpression(_numerator)
                .simplify_color()
                .to_color_casimir(
                    fundamental=Representation.cof(Nc),
                    adjoint=Representation.coad(dA),
                    settings=_settings,
                )
                .to_expression()
            )
            # CF and CA are display names for shared representation-aware invariants;
            # the quark-loop trace uses the conventional fundamental index TR=1/2.
            _numerator = (
                _numerator.replace(cas(2, cof(Nc)), CF)
                .replace(cas(2, coad(dA)), CA)
                .replace(trace_index(2, cof(Nc)), one / 2)
            )
            # Keep the unexpanded graph numerator for the independent direct IRR path.
            _raw_numerator = _numerator
            if _loops:
                _numerator = _diagram.momentum_basis().route_expression(
                    _diagram.uv_expansion(_mUV, numerator=_numerator).to_expression()
                )
            _numerator = (
                _numerator.replace(mink(dim, index), mink(D, index))
                .replace(mink(dim), mink(D))
                .replace(_mUV**2, M)
            )
            if _incoming == [gluon]:
                _numerator = _numerator.replace(
                    mink(D, _ports[0]), mink(D, mu)
                ).replace(mink(D, _ports[1]), mink(D, nu))
            _pattern = den(edge_, mom_, mass_, quad_)
            for _match in list(_numerator.match(_pattern)):
                _values = dict(_match)
                _formal = family.rewrite_numerator(_values[quad_], [coordinate])
                assert _formal == coordinate
                _numerator = _numerator.replace(
                    den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                    _formal,
                )
            _factor = (
                _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
            )
            if _kind == "quark":
                # An amputated two-point kernel omits only external fermion ordering.
                _raw = _diagram.overall_factor_expression()
                _removed = (_raw / _raw.replace(_ordering(_value), one)).replace(
                    _ordering(_value), _value
                )
                assert _removed == -one
                _factor /= _removed
                _probes = [
                    (
                        "quark_p",
                        gamma(bis(4, _ports[0]), bis(4, _ports[1]), mink(D, mu))
                        * P(0, mink(D, mu))
                        / (4 * s),
                    ),
                    (
                        "quark_m",
                        metric(bis(4, _ports[0]), bis(4, _ports[1])) / (4 * mass),
                    ),
                ]
            elif _kind in ("tree", "vertex"):
                _probes = [
                    (
                        f"{_kind}_{_number}",
                        gamma(bis(4, _ports[0]), bis(4, _ports[2]), mink(D, _ports[1]))
                        / (4 * D),
                    )
                ]
            else:
                _probes = [(_kind, one)]
            if _loops:
                irr_inputs[_kind, _number] = (
                    _diagram,
                    _raw_numerator,
                    _ports.copy(),
                    _probes,
                    _factor,
                )
            for _label, _projector in _probes:
                _traced = (
                    TensorExpression((_numerator * _projector).expand())
                    .simplify_gamma()
                    .expand()
                    .simplify_metrics()
                    .to_dots()
                    .to_expression()
                )
                _scalar = family.rewrite_numerator(
                    kinematics.apply(reducer.reduce(_traced)), [coordinate]
                )
                _scalar = (_scalar * _factor).together().expand()
                _terms = []
                for _monomial, _coefficient in _scalar.coefficient_list(coordinate):
                    _power = -int(
                        (
                            _monomial.derivative(coordinate) * coordinate / _monomial
                        ).together()
                    )
                    assert _monomial == coordinate ** (-_power)
                    assert not _coefficient.matches(K(_arguments))
                    _terms.append(([_power], _coefficient))
                _parts[_label] = _terms

    _tree_terms = _parts.pop("tree_0")
    assert len(_tree_terms) == 1 and _tree_terms[0][0] == [0]
    tree = _tree_terms[0][1]
    assert (tree + Symbol.I * gs * Nc * CF).together() == zero
    solution = hep.IBPFamily(family, name="qcd_one_loop").reduce_laporta(
        sorted({tuple(p) for _terms in _parts.values() for p, c in _terms}), max_depth=2
    )
    assert solution.residuals == [[1]]
    assert abs(complex(oneloop.A0(1.0, 1.0)[1]) - 1) < 1e-12
    _reduced, uv_poles = {}, {}
    for _label, _terms in _parts.items():
        _expression = sum(
            (c * solution.reduce(p, integral=integral) for p, c in _terms), zero
        ).together()
        _reduced[_label] = _expression
        _normalized = (
            _expression
            * (Symbol.I / tree if _label.startswith("vertex") else one)
            / gs**2
        )
        if _label == "quark_loop":
            _normalized *= Nf
        _pole = (
            _normalized.replace(integral(1), M / epsilon)
            .replace(D, 4 - 2 * epsilon)
            .series(epsilon, 0, -1)
            .to_expression()
            .expand()
        )
        _pole = _pole.replace(mink(4, index), mink(D, index))
        uv_poles[_label] = _pole
        assert _pole.derivative(M).expand() == zero
        assert _pole.derivative(mass).expand() == zero
        assert _pole.coefficient(epsilon**-2) == zero
    uv_poles["gluon"] = sum(
        (
            uv_poles[_label]
            for _label in ("gluon_loop", "ghost_loop", "quark_loop", "tadpole")
        ),
        zero,
    ).expand()
    uv_poles["vertex"] = (uv_poles["vertex_0"] + uv_poles["vertex_1"]).expand()
    _expected = [
        CF * xi / epsilon,
        -CF * (xi + 3) / epsilon,
        CA * (xi - 3) * s / (4 * epsilon),
        ((13 - 3 * xi) * CA - 4 * Nf) * (s * gmunu - ppmunu) / (6 * epsilon),
        (CF * xi + CA * (xi + 3) / 4) / epsilon,
    ]
    for _label, _reference in zip(
        ("quark_p", "quark_m", "ghost", "gluon", "vertex"), _expected, strict=True
    ):
        assert (uv_poles[_label] - _reference).together() == zero

    return (
        CA,
        CF,
        D,
        K,
        M,
        Nc,
        Nf,
        P,
        bis,
        cas,
        coad,
        cof,
        coordinate,
        dA,
        den,
        diagrams,
        dim,
        edge_,
        epsilon,
        family,
        gamma,
        ghost,
        gluon,
        gmunu,
        gs,
        index,
        integral,
        interaction_rules,
        irr_inputs,
        kinematics,
        left,
        mass,
        mass_,
        metric,
        mink,
        model,
        mom_,
        mu,
        nu,
        one,
        ppmunu,
        quad_,
        quark,
        reducer,
        right,
        s,
        solution,
        trace_index,
        tree,
        uv_poles,
        vacuum,
        wave,
        xi,
        zero,
    )


@app.cell
def _(diagrams, mo, uv_poles):
    mo.vstack(
        [
            mo.md("## Generated diagrams and UV poles"),
            mo.ui.tabs(
                {
                    "Quark": mo.vstack(
                        [
                            mo.hstack(diagrams["quark"]),
                            uv_poles["quark_p"],
                            uv_poles["quark_m"],
                        ]
                    ),
                    "Ghost": mo.vstack(
                        [mo.hstack(diagrams["ghost"]), uv_poles["ghost"]]
                    ),
                    "Gluon": mo.vstack(
                        [
                            mo.hstack(diagrams["gluon_loop"] + diagrams["ghost_loop"]),
                            mo.hstack(diagrams["quark_loop"] + diagrams["tadpole"]),
                            uv_poles["gluon"],
                        ]
                    ),
                    "Quark–gluon vertex": mo.vstack(
                        [mo.hstack(diagrams["vertex"]), uv_poles["vertex"]]
                    ),
                }
            ),
            mo.md(
                r"The self-energies multiply $i a_4$ and the external color identity. Quark entries are coefficients of $\not p$ and $m$. The vertex entry multiplies its generated tree value times $a_4$. Both vertex diagrams contribute, and the gluon pole becomes transverse only after summing its loop classes."
            ),
        ]
    )


@app.cell
def _(family, integral, mo, solution):
    mo.vstack(
        [
            mo.md("## One shared vacuum family"),
            family,
            mo.md(
                r"The longitudinal gluon denominators cancel after tensor projection. All scalar terms belong to this family; the calculation rejects hidden loop momenta in their coefficients. Native IBP reduces $I(1),\ldots,I(4)$ to $I(1)$:"
            ),
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.md(f"**I({_power})**"),
                            solution.reduce([_power], integral=integral),
                        ]
                    )
                    for _power in range(1, 5)
                ]
            ),
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "The tadpole pole is an analytic input. A finite-depth IBP residual alone does not certify a minimal master basis."
            ),
        ]
    )


@app.cell
def _(
    CA,
    CF,
    ColorCasimirSettings,
    D,
    M,
    Matrix,
    Nc,
    Nf,
    P,
    Representation,
    S,
    Symbol,
    TensorExpression,
    bis,
    cas,
    coad,
    cof,
    dA,
    diagrams,
    dim,
    epsilon,
    gamma,
    ghost,
    gluon,
    gmunu,
    hep,
    index,
    interaction_rules,
    json,
    kinematics,
    left,
    mass,
    metric,
    mink,
    model,
    mu,
    nu,
    one,
    ppmunu,
    quark,
    right,
    s,
    solution,
    trace_index,
    tree,
    uv_poles,
    wave,
    xi,
    zero,
):
    # Local counterterm operators, with Z=1+a4*deltaZ, a4=gs²/(16*pi²).
    # Unknowns: Zq, Zm, ZA, Zxi, Zc, Zg, ZAm, Zcm. The local operator basis
    # is an explicit model input; all matching entries come from generated graphs.
    _qqg = interaction_rules["qqg"]
    _spec = json.loads(model.to_json())
    _vertex_definition = next(
        v for v in _spec["vertex_rules"] if v["name"] == _qqg[0].name
    )
    # Auxiliary operators follow the reference model:
    # M*(ZAm²-1)*A²/2 and M*(Zcm²-1)*cbar*c/2. Identical gluons supply
    # the factor two in their local rule; the distinct ghost fields do not.
    _a4 = S("qcd_ct::a4")
    unknowns = list(
        S(
            "qcd_ct::deltaZq",
            "qcd_ct::deltaZm",
            "qcd_ct::deltaZA",
            "qcd_ct::deltaZxi",
            "qcd_ct::deltaZc",
            "qcd_ct::deltaZg",
            "qcd_ct::deltaZAm",
            "qcd_ct::deltaZcm",
        )
    )
    _Zq, _Zm, _ZA, _Zxi, _Zc, _Zg, _ZAm, _Zcm = (
        one + _a4 * _delta for _delta in unknowns
    )
    _spec["orders"].append({"name": "CT", "expansion_order": 1, "hierarchy": 1})
    for _label, _particles, _spins, _color, _lorentz, _coupling in [
        (
            "qq_kinetic",
            [quark.antiname, quark.name],
            [2, 2],
            "Identity(1,2)",
            "Gamma(dummy(1),idx(1,1),idx(1,2))*P(dummy(1),2)",
            Symbol.I * (_Zq - one),
        ),
        (
            "qq_mass",
            [quark.antiname, quark.name],
            [2, 2],
            "Identity(1,2)",
            "Identity(idx(1,1),idx(1,2))",
            -Symbol.I * mass * (_Zq * _Zm - one),
        ),
        (
            "gg_kinetic",
            [gluon.name] * 2,
            [3, 3],
            "Identity(1,2)",
            "Metric(idx(1,1),idx(1,2))*P(dummy(1),1)*P(dummy(1),1)-P(idx(1,1),1)*P(idx(1,2),1)",
            -Symbol.I * (_ZA - one),
        ),
        (
            "gg_gauge",
            [gluon.name] * 2,
            [3, 3],
            "Identity(1,2)",
            "P(idx(1,1),1)*P(idx(1,2),1)",
            -Symbol.I * (_ZA / _Zxi - one) / xi,
        ),
        (
            "gg_auxmass",
            [gluon.name] * 2,
            [3, 3],
            "Identity(1,2)",
            "Metric(idx(1,1),idx(1,2))",
            Symbol.I * M * (_ZAm**2 - one),
        ),
        (
            "ghost_kinetic",
            [ghost.antiname, ghost.name],
            [-1, -1],
            "Identity(1,2)",
            "P(dummy(1),2)*P(dummy(1),2)",
            Symbol.I * (_Zc - one),
        ),
        (
            "ghost_auxmass",
            [ghost.antiname, ghost.name],
            [-1, -1],
            "Identity(1,2)",
            "1",
            Symbol.I * M * (_Zcm**2 - one) / 2,
        ),
    ]:
        _spec["lorentz_structures"].append(
            {"name": "CT_L_" + _label, "spins": _spins, "structure": _lorentz}
        )
        _spec["couplings"].append(
            {
                "name": "CT_GC_" + _label,
                "expression": repr(_coupling.series(_a4, 0, 1).to_expression()),
                "orders": [["QCD", 2], ["CT", 1]],
                "value": None,
            }
        )
        _spec["vertex_rules"].append(
            {
                "name": "CT_" + _label,
                "particles": _particles,
                "color_structures": [_color],
                "lorentz_structures": ["CT_L_" + _label],
                "couplings": [["CT_GC_" + _label]],
            }
        )
    # Copy every color/Lorentz slot of the actual model rule and dress its coupling.
    _ct_vertex = json.loads(json.dumps(_vertex_definition))
    _ct_vertex["name"] = "CT_qqg"
    for _row, _couplings in enumerate(_ct_vertex["couplings"]):
        for _col, _coupling in enumerate(_couplings):
            if _coupling is None:
                continue
            _dressed = model.expand_couplings(S("UFO::" + _coupling)) * (
                _Zq * _Zg * _ZA.sqrt() - one
            )
            _name = f"CT_qqg_{_row}_{_col}"
            _spec["couplings"].append(
                {
                    "name": _name,
                    "expression": repr(_dressed.series(_a4, 0, 1).to_expression()),
                    "orders": [["QCD", 3], ["CT", 1]],
                    "value": None,
                }
            )
            _couplings[_col] = _name
    _spec["vertex_rules"].append(_ct_vertex)
    _ct_model = hep.Model.from_json(json.dumps(_spec))
    _ct_quark, _ct_gluon, _ct_ghost = (
        _ct_model.particle_by_pdg(pdg) for pdg in (5, 21, 9000005)
    )
    _ct_vertices = [v for v in _ct_model.vertex_rules if v.name.startswith("CT_")]
    ct_diagrams, ct_coefficients = {}, {}
    _external_ordering = diagrams["tree"][0].overall_factor_expression(evaluate=True)
    for _kind, _incoming, _outgoing, _count, _qcd_order in [
        ("quark", [_ct_quark], [_ct_quark], 2, 2),
        ("gluon", [_ct_gluon], [_ct_gluon], 3, 2),
        ("ghost", [_ct_ghost], [_ct_ghost], 2, 2),
        ("vertex", [_ct_quark], [_ct_gluon, _ct_quark], 1, 3),
    ]:
        # CT order is perturbative bookkeeping; two-point insertions have no loops.
        # Bound vertices explicitly to exclude arbitrarily long insertion chains.
        _options = {
            "loops": 0,
            "max_vertices": 1,
            "maximum_bridges": None,
            "vertex_allow": _ct_vertices,
            "self_energy": None,
            "tadpoles": None,
            "zero_snails": None,
            "numerator_grouping": None,
            "progress": None,
        }
        _generated = _ct_model.generate_diagrams(
            _incoming,
            _outgoing,
            coupling_orders={"QCD": _qcd_order, "CT": 1},
            **_options,
        )
        assert len(_generated.diagrams) == _count
        assert not _ct_model.generate_diagrams(
            _incoming, _outgoing, coupling_orders={"CT": 0}, **_options
        ).diagrams
        ct_diagrams[_kind] = _generated.diagrams
        for _diagram in _generated.diagrams:
            assert len(_diagram.internal_edges) == 0
            assert _diagram.symmetry_factor == one
            _factor = (
                _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
            )
            assert _factor == (
                _external_ordering if _kind in ("quark", "vertex") else one
            )
            _ports = {}
            if _kind != "ghost":
                for _edge in _diagram.external_edges:
                    _rep = (
                        mink
                        if _kind == "gluon"
                        or (_kind == "vertex" and _edge.external_index == 1)
                        else bis
                    )
                    _ports[_edge.external_index] = dict(
                        next(
                            _diagram.projector_expression().match(
                                wave(_edge.id, _rep(4, index)), max_level=0
                            )
                        )
                    )[index]
            _numerator = _ct_model.expand_couplings(
                _diagram.numerator_expression(in_lmb=True).to_expression()
            )
            if _kind == "vertex":
                _color_projector = (
                    TensorExpression.t(dA, Nc)(_ports[1], _ports[0], _ports[2])
                    .spenso_conjugate()
                    .to_expression()
                )
                _numerator = (
                    _numerator.replace(cof(3, index), cof(Nc, index)).replace(
                        coad(8, index), coad(dA, index)
                    )
                    * _color_projector
                )
                _settings = ColorCasimirSettings(rewrite_fundamental_dimension=False)
            else:
                _particle = _incoming[0]
                _rep, _numeric_dim, _symbolic_dim = (
                    (cof, 3, Nc) if _kind == "quark" else (coad, 8, dA)
                )
                _color_slots = [
                    slot.dual().to_expression()
                    for slot in TensorExpression(_numerator).interface
                    if slot.to_expression().matches(_rep(_numeric_dim, index))
                ]
                assert len(_color_slots) == 2, (_kind, _numerator)
                _color_indices = dict(
                    next(
                        metric(*_color_slots).match(
                            _particle.color_sum(left, right), max_level=0
                        )
                    )
                )
                _color_projector = _particle.color_sum(
                    _color_indices[left], _color_indices[right]
                )
                _numerator = (
                    (_numerator * _color_projector / _symbolic_dim)
                    .replace(cof(3, index), cof(Nc, index))
                    .replace(coad(8, index), coad(dA, index))
                )
                _settings = ColorCasimirSettings()
            _numerator = (
                TensorExpression(_numerator)
                .simplify_color()
                .to_color_casimir(
                    fundamental=Representation.cof(Nc),
                    adjoint=Representation.coad(dA),
                    settings=_settings,
                )
                .to_expression()
            )
            _numerator = (
                _numerator.replace(cas(2, cof(Nc)), CF)
                .replace(cas(2, coad(dA)), CA)
                .replace(trace_index(2, cof(Nc)), one / 2)
                .replace(mink(dim, index), mink(D, index))
                .replace(mink(dim), mink(D))
            )
            if _kind == "quark":
                _probes = [
                    (
                        "quark_p",
                        gamma(bis(4, _ports[0]), bis(4, _ports[1]), mink(D, mu))
                        * P(0, mink(D, mu))
                        / (4 * s),
                    ),
                    (
                        "quark_m",
                        metric(bis(4, _ports[0]), bis(4, _ports[1])) / (4 * mass),
                    ),
                ]
                _normalization = Symbol.I * _a4 * _external_ordering
            elif _kind == "vertex":
                _probes = [
                    (
                        _kind,
                        gamma(bis(4, _ports[0]), bis(4, _ports[2]), mink(D, _ports[1]))
                        / (4 * D),
                    )
                ]
                _normalization = tree * _a4
            else:
                if _kind == "gluon":
                    _numerator = _numerator.replace(
                        mink(D, _ports[0]), mink(D, mu)
                    ).replace(mink(D, _ports[1]), mink(D, nu))
                _probes = [(_kind, one)]
                _normalization = Symbol.I * _a4
            for _label, _projector in _probes:
                _trace = (
                    TensorExpression((_numerator * _projector).expand())
                    .simplify_gamma()
                    .expand()
                    .simplify_metrics()
                    .to_dots()
                    .to_expression()
                )
                _coefficient = (
                    (kinematics.apply(_trace) * _factor / _normalization)
                    .together()
                    .expand()
                )
                ct_coefficients[_label] = (
                    ct_coefficients.get(_label, zero) + _coefficient
                )
    _ct_g = ct_coefficients["gluon"].coefficient(gmunu)
    _ct_pp = ct_coefficients["gluon"].coefficient(ppmunu)
    _ct_rows = [
        ct_coefficients["quark_p"],
        ct_coefficients["quark_m"],
        _ct_g.coefficient(s),
        xi * _ct_pp,
        ct_coefficients["ghost"].coefficient(s),
        ct_coefficients["vertex"],
        _ct_g.replace(s, zero) / M,
        ct_coefficients["ghost"].replace(s, zero) / M,
    ]
    _entries = [r.derivative(z).together() for r in _ct_rows for z in unknowns]
    assert all(e.derivative(z) == zero for e in _entries for z in unknowns)
    for _i, _row in enumerate(_ct_rows):
        assert (
            _row - sum((_entries[8 * _i + j] * z for j, z in enumerate(unknowns)), zero)
        ).together() == zero
    ct_matrix = Matrix.from_linear(8, 8, _entries)
    assert ct_matrix[6, 6].to_expression() == 2
    assert ct_matrix[7, 7].to_expression() == 1
    assert (
        ct_coefficients["gluon"] - _ct_g * gmunu - _ct_pp * ppmunu
    ).together() == zero
    assert _ct_g.derivative(s).derivative(s) == zero
    assert ct_coefficients["ghost"].derivative(s).derivative(s) == zero

    _gluon_g = uv_poles["gluon"].coefficient(gmunu)
    _gluon_pp = uv_poles["gluon"].coefficient(ppmunu)
    _rhs = Matrix.vec(
        [
            -uv_poles["quark_p"],
            -uv_poles["quark_m"],
            -_gluon_g.coefficient(s),
            -xi * _gluon_pp,
            -uv_poles["ghost"].coefficient(s),
            -uv_poles["vertex"],
            -_gluon_g.replace(s, zero) / M,
            -uv_poles["ghost"].replace(s, zero) / M,
        ]
    )
    deltas = ct_matrix.solve(_rhs)
    _references = [
        -CF * xi / epsilon,
        -3 * CF / epsilon,
        ((13 - 3 * xi) * CA - 4 * Nf) / (6 * epsilon),
        ((13 - 3 * xi) * CA - 4 * Nf) / (6 * epsilon),
        CA * (3 - xi) / (4 * epsilon),
        -(11 * CA - 2 * Nf) / (6 * epsilon),
        zero,
        zero,
    ]
    for _row, _reference in enumerate(_references):
        assert (deltas[_row, 0].to_expression() - _reference).together() == zero
    _residual = ct_matrix * deltas - _rhs
    assert all(
        _residual[_row, 0].to_expression().together() == zero for _row in range(8)
    )
    assert deltas[1, 0].to_expression().derivative(xi).expand() == zero
    assert deltas[5, 0].to_expression().derivative(xi).expand() == zero
    print(
        "Generated QCD one-loop: symbolic gauge, four IBP targets, eight generated counterterms passed",
        solution.stats,
    )

    # The completed MS/MSbar reference only needs the local UV poles. Use the
    # OneLOop measure conversion (4*pi)^eps*rGamma=1+cDelta*eps+O(eps²), where
    # cDelta=log(4*pi)-EulerGamma, with D=4-2eps. Keep it symbolic in exact checks.
    _cDelta = S("qcd_ct::cDelta")
    assert (-2 / (D - 4)).replace(D, 4 - 2 * epsilon) == 1 / epsilon
    counterterms_by_scheme = {
        "MS": [deltas[_row, 0].to_expression().together() for _row in range(8)],
        "MSbar": [
            (
                epsilon * deltas[_row, 0].to_expression() * (1 / epsilon + _cDelta)
            ).together()
            for _row in range(8)
        ],
    }
    for _scheme, _constants in counterterms_by_scheme.items():
        for _label, _generated_ct in ct_coefficients.items():
            for _delta, _constant in zip(unknowns, _constants, strict=True):
                _generated_ct = _generated_ct.replace(_delta, _constant)
            _subtraction_part = uv_poles[_label] * (
                one + epsilon * _cDelta if _scheme == "MSbar" else one
            )
            assert (_generated_ct + _subtraction_part).together() == zero
            # In one common conventional measure MS retains the finite cDelta*P.
            _remainder = (
                uv_poles[_label] * (one + epsilon * _cDelta) + _generated_ct
            ).together()
            _expected_remainder = (
                epsilon * _cDelta * uv_poles[_label] if _scheme == "MS" else zero
            )
            assert (_remainder - _expected_remainder).together() == zero
    print("Generated QCD CT amplitudes: complete MS/MSbar tensor cancellation passed")

    counterterm_names = ["Zq", "Zm", "ZA", "Zξ", "Zc", "Zg", "ZAm", "Zcm"]
    residues = [
        (epsilon * deltas[_row, 0].to_expression()).together() for _row in range(8)
    ]
    vertex_residue = (epsilon * uv_poles["vertex"]).together()

    return (
        counterterm_names,
        counterterms_by_scheme,
        ct_coefficients,
        ct_diagrams,
        ct_matrix,
        deltas,
        residues,
        unknowns,
        vertex_residue,
    )


@app.cell(hide_code=True)
def _(ct_coefficients, ct_diagrams, ct_matrix, deltas, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Generated local counterterms

    Write $Z=1+a_4\delta Z$, with $q_0=\sqrt{Z_q}q$, $m_0=Z_m m$,
    $A_0=\sqrt{Z_A}A$, $\xi_0=Z_\xi\xi$, $c_0=\sqrt{Z_c}c$ and
    $g_{s,0}=Z_g g_s$. The local kinetic, mass and gauge-fixing operators are model
    inputs. Symbolica expands their bare factors to obtain the CT couplings.
    The quark–gluon rule copies the original model's ordered particle, color and
    Lorentz data and multiplies its actual coupling by $Z_qZ_g\sqrt{Z_A}-1$.

    These are tree topologies with **CT order one**, separate from loop count.
    An explicit vertex bound prevents arbitrarily long two-point insertion chains.
    The generator produces two quark, three gluon, two ghost and one vertex CT;
    setting CT order zero excludes them.
    """),
            mo.ui.tabs(
                {
                    _kind: mo.hstack(_diagrams)
                    for _kind, _diagrams in ct_diagrams.items()
                }
            ),
            mo.md(r"**Coefficients extracted from generated numerators**"),
            *[
                mo.hstack([mo.md("**" + _kind + "**"), _coefficient])
                for _kind, _coefficient in ct_coefficients.items()
            ],
            mo.md(r"""
    Shared particle color sums close the two-point color indices, and the vertex
    uses the conjugate tree color tensor. Native graph weights are retained.
    Only the named external-fermion convention is converted for the amputated
    quark two-point kernel; ghosts retain their own native sign.

    The columns below are $(\delta Z_q,\delta Z_m,\delta Z_A,\delta Z_\xi,
    \delta Z_c,\delta Z_g,\delta Z_{Am},\delta Z_{cm})$.
    Every matrix entry is a derivative of a generated coefficient, with linearity
    checked before solving. The complete tensor amplitudes cancel separately.

    The [reference model](https://raw.githubusercontent.com/FeynCalc/feyncalc/master/FeynCalc/Examples/FeynRules/QCD/QCD.fr)
    uses $M(Z_{Am}^2-1)A^2/2$ and $M(Z_{cm}^2-1)\bar c c/2$.
    Identical gluons supply a factor two; the distinct ghost fields do not.
    Their rows therefore contain $2\delta Z_{Am}$ and $\delta Z_{cm}$.
    """),
            ct_matrix,
            deltas,
            mo.md(r"""
    The result obeys $Z_\xi=Z_A$. Mass and coupling renormalization are gauge
    independent. The coupling pole gives $\beta_0=(11C_A-2N_f)/3$ in
    $\mu\,d g_s/d\mu=-\beta_0g_s^3/(16\pi^2)+O(g_s^5)$.
    """),
        ]
    )


@app.cell(hide_code=True)
def _(counterterm_names, counterterms_by_scheme, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## MS and MSbar at the same scale

    Use $\Delta=1/\epsilon+c_\Delta$ with
    $c_\Delta=\log(4\pi)-\gamma_E$ and $D=4-2\epsilon$.
    [OneLOop](https://arxiv.org/abs/1007.4716) divides its master integrals by
    $r_\Gamma=\Gamma(1-\epsilon)^2\Gamma(1+\epsilon)/\Gamma(1-2\epsilon)$.
    The conventional loop measure restores $(4\pi)^\epsilon r_\Gamma=
    1+c_\Delta\epsilon+O(\epsilon^2)$, with $i/(16\pi^2)$ stripped off.

    If a normalized loop coefficient is $P/\epsilon+F$, this measure gives
    $P/\epsilon+F+c_\Delta P$. MS subtracts its pole; MSbar also subtracts
    $c_\Delta P$. The actual generated CT amplitudes cancel the corresponding
    terms in every sector. The finite scheme shift left in MS is checked too.

    The reference extracts UV terms after tensor reduction. Shared UV expansion
    provides those same local coefficients here; no full finite vertex is needed
    for this comparison. The direct selective-IRR calculation below derives the original reference's
    nonzero auxiliary gluon mass separately. Its physical pole constants agree.
    """),
            mo.ui.table(
                [
                    {"Constant": _name, "MS": str(_ms), "MSbar": str(_msbar)}
                    for _name, _ms, _msbar in zip(
                        counterterm_names,
                        counterterms_by_scheme["MS"],
                        counterterms_by_scheme["MSbar"],
                        strict=True,
                    )
                ],
                selection=None,
            ),
        ]
    )


@app.cell
def _(
    CA,
    D,
    K,
    M,
    Matrix,
    Nf,
    P,
    Replacement,
    S,
    Symbol,
    TensorExpression,
    coordinate,
    ct_coefficients,
    ct_matrix,
    deltas,
    den,
    dim,
    edge_,
    epsilon,
    family,
    gluon,
    gmunu,
    gs,
    hep,
    index,
    integral,
    irr_inputs,
    mass,
    mass_,
    mink,
    mom_,
    mu,
    nu,
    one,
    ppmunu,
    quad_,
    reducer,
    s,
    tree,
    unknowns,
    uv_poles,
    vacuum,
    xi,
    zero,
):
    # Direct n=0 infrared rearrangement: massify only massless propagators, then
    # Taylor-expand external momenta to the superficial divergence degree. This
    # follows the original massive reference independently of graph.uv_expansion.
    _Q, _dot, _x, _t, _a, _b = S(
        "gammalooprs::Q",
        "spenso::dot",
        "qcd_irr::x",
        "qcd_irr::t",
        "qcd_irr::a_",
        "qcd_irr::b_",
    )
    _irr_kin = hep.Kinematics(D, momenta=[K(0), P(0), P(1)]).with_scalar_product(
        P(0), P(0), s
    )
    _irr_scalars = {}
    for (_kind, _number), (
        _diagram,
        _raw,
        _ports,
        _probes,
        _factor,
    ) in irr_inputs.items():
        # Separate every primitive longitudinal 1/q² before changing propagators.
        # A common squared denominator would spuriously massify the Feynman term.
        _tags = {
            edge.id: S(f"qcd_irr::gluon_{edge.id}")
            for edge in _diagram.internal_edges
            if edge.particle_name == gluon.name
        }
        _tagged = _raw
        for _edge_id, _tag in _tags.items():
            _tagged = _tagged.replace(
                _dot(_Q(_edge_id, mink(4)), _Q(_edge_id, mink(4))), _tag
            )
        _components = (
            _tagged.expand().coefficient_list(*_tags.values())
            if _tags
            else [(one, _tagged)]
        )
        _reconstructed = zero
        for _monomial, _numerator in _components:
            _powers = {}
            for _edge_id, _tag in _tags.items():
                _exponent = int(
                    (_monomial.derivative(_tag) * _tag / _monomial).together()
                )
                assert _exponent in (0, -1), (_kind, _exponent, _monomial)
                _powers[_edge_id] = 1 - _exponent
                _monomial = _monomial.replace(
                    _tag, _dot(_Q(_edge_id, mink(4)), _Q(_edge_id, mink(4)))
                )
            _reconstructed += _monomial * _numerator
            if any(_power == 2 for _power in _powers.values()):
                assert _numerator.replace(xi, one).expand() == zero
            _denominator = (
                _diagram.denominator_expression(in_lmb=True, edge_powers=_powers)
                .to_expression()
                .replace(mink(dim, index), mink(D, index))
                .replace(mink(dim), mink(D))
            )
            # Keep massive quark denominators intact; only massless ones get M.
            for _match in list(_denominator.match(den(edge_, mom_, mass_, quad_))):
                _vals = dict(_match)
                _denominator = _denominator.replace(
                    den(_vals[edge_], _vals[mom_], _vals[mass_], _vals[quad_]),
                    _vals[quad_] - (M if _vals[mass_] == zero else zero),
                )
            _numerator = (
                _diagram.momentum_basis()
                .route_expression(_numerator)
                .replace(mink(dim, index), mink(D, index))
                .replace(mink(dim), mink(D))
            )
            if _kind in ("gluon_loop", "ghost_loop", "quark_loop", "tadpole"):
                _numerator = _numerator.replace(
                    mink(D, _ports[0]), mink(D, mu)
                ).replace(mink(D, _ports[1]), mink(D, nu))
            for _label, _projector in _probes:
                _traced = (
                    TensorExpression((_numerator * _projector).expand())
                    .simplify_gamma()
                    .expand()
                    .simplify_metrics()
                    .to_dots()
                    .to_expression()
                )
                _direct = _irr_kin.apply(_traced / _denominator).replace(
                    vacuum.scalar_product(K(0), K(0)), _x
                )
                _direct = _direct.replace_multiple(
                    [
                        Replacement(
                            _dot(K(0, mink(D)), P(_a, mink(D))),
                            _t * _dot(K(0, mink(D)), P(_a, mink(D))),
                        ),
                        Replacement(
                            _dot(P(_a, mink(D)), K(0, mink(D))),
                            _t * _dot(P(_a, mink(D)), K(0, mink(D))),
                        ),
                        Replacement(
                            _dot(P(_a, mink(D)), P(_b, mink(D))),
                            _t**2 * _dot(P(_a, mink(D)), P(_b, mink(D))),
                        ),
                        Replacement(P(_a, mink(D, index)), _t * P(_a, mink(D, index))),
                        Replacement(s, _t**2 * s),
                    ]
                )
                # The pslash projector lowers the quark external degree by one.
                # Two-point gluon/ghost tensors need degree two; the vertex needs zero.
                _direct = (
                    _direct.series(
                        _t,
                        0,
                        2
                        if _kind
                        in (
                            "gluon_loop",
                            "ghost_loop",
                            "quark_loop",
                            "tadpole",
                            "ghost",
                        )
                        else 0,
                    )
                    .to_expression()
                    .replace(_t, one)
                )
                _scalar = _irr_kin.apply(reducer.reduce(_direct)).replace(
                    vacuum.scalar_product(K(0), K(0)), _x
                )
                _scalar = (
                    (
                        _scalar
                        * _factor
                        / gs**2
                        * (Symbol.I / tree if _kind == "vertex" else one)
                        * (Nf if _kind == "quark_loop" else one)
                    )
                    .together()
                    .expand()
                )
                _irr_scalars[_label] = (
                    _irr_scalars.get(_label, zero) + _scalar
                ).together()
        assert (_reconstructed - _raw).together() == zero
    _irr_terms, _irr_targets = {}, set()
    for _label, _scalar in _irr_scalars.items():
        # Symbolica partial fractions separate the physical and auxiliary tadpoles.
        # Their principal parts must reconstruct the complete projected integrand.
        _apart = _scalar.apart(_x)
        assert (_apart - _scalar).together() == zero
        _reconstructed = zero
        _terms = []
        for _squared_mass in (M, mass**2):
            _principal = (
                _apart.replace(_x, coordinate + _squared_mass)
                .series(coordinate, 0, -1)
                .to_expression()
                .expand()
            )
            _reconstructed += _principal.replace(coordinate, _x - _squared_mass)
            for _monomial, _coefficient in _principal.coefficient_list(coordinate):
                if _coefficient == zero:
                    continue
                _power = -int(
                    (
                        _monomial.derivative(coordinate) * coordinate / _monomial
                    ).together()
                )
                assert _monomial == coordinate ** (-_power) and _power > 0
                _terms.append((_squared_mass, _power, _coefficient))
                _irr_targets.add((_power,))
        assert (_scalar - _reconstructed).together() == zero, _label
        _irr_terms[_label] = _terms
    irr_solution = hep.IBPFamily(family, name="qcd_direct_irr").reduce_laporta(
        [list(p) for p in sorted(_irr_targets)], max_depth=2
    )
    assert irr_solution.residuals == [[1]]
    _irr_poles = {}
    for _label, _terms in _irr_terms.items():
        _expression = sum(
            (
                _coefficient
                * irr_solution.reduce([_power], integral=integral)
                .replace(M, _squared_mass)
                .replace(integral(1), _squared_mass / epsilon)
                for _squared_mass, _power, _coefficient in _terms
            ),
            zero,
        ).together()
        _pole = (
            _expression.replace(D, 4 - 2 * epsilon)
            .series(epsilon, 0, -1)
            .to_expression()
            .expand()
            .replace(mink(4, index), mink(D, index))
        )
        _irr_poles[_label] = _pole.together().expand()
        assert _pole.coefficient(epsilon**-2) == zero
        assert _pole.derivative(mass).together() == zero
    _irr_poles["gluon"] = sum(
        (_irr_poles[k] for k in ("gluon_loop", "ghost_loop", "quark_loop", "tadpole")),
        zero,
    ).expand()
    _irr_poles["vertex"] = (_irr_poles["vertex_0"] + _irr_poles["vertex_1"]).expand()

    # Physical UV constants agree, but the auxiliary mass does depend on the
    # rearrangement prescription. The reference's quark loop retains its true mass.
    for _label in ("quark_p", "quark_m", "ghost", "vertex"):
        assert (_irr_poles[_label] - uv_poles[_label]).together() == zero
    _irr_auxiliary_pole = CA * (1 + 3 * xi) * M * gmunu / (4 * epsilon)
    assert (
        _irr_poles["gluon"] - uv_poles["gluon"] - _irr_auxiliary_pole
    ).together() == zero
    assert _irr_poles["quark_loop"].derivative(M) == zero
    _irr_gluon_g = _irr_poles["gluon"].coefficient(gmunu)
    _irr_gluon_pp = _irr_poles["gluon"].coefficient(ppmunu)
    _irr_rhs = Matrix.vec(
        [
            -_irr_poles["quark_p"],
            -_irr_poles["quark_m"],
            -_irr_gluon_g.coefficient(s),
            -xi * _irr_gluon_pp,
            -_irr_poles["ghost"].coefficient(s),
            -_irr_poles["vertex"],
            -_irr_gluon_g.replace(s, zero) / M,
            -_irr_poles["ghost"].replace(s, zero) / M,
        ]
    )
    irr_deltas = ct_matrix.solve(_irr_rhs)
    for _row in range(8):
        _reference = (
            -CA * (1 + 3 * xi) / (8 * epsilon)
            if _row == 6
            else deltas[_row, 0].to_expression()
        )
        assert (irr_deltas[_row, 0].to_expression() - _reference).together() == zero
    _irr_ct_rules = [
        Replacement(delta, irr_deltas[_row, 0].to_expression())
        for _row, delta in enumerate(unknowns)
    ]
    for _label, _generated_ct in ct_coefficients.items():
        assert (
            _irr_poles[_label] + _generated_ct.replace_multiple(_irr_ct_rules)
        ).together() == zero
    print(
        "Direct massive QCD IRR: two tadpole masses, five IBP targets, eight generated CT equations passed",
        irr_solution.stats,
    )

    irr_residues = [
        (epsilon * irr_deltas[_row, 0].to_expression()).together() for _row in range(8)
    ]

    return irr_deltas, irr_residues, irr_solution


@app.cell
def _(irr_deltas, irr_solution, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Direct infrared rearrangement with a massive quark

    The original reference replaces each **massless** propagator $q^2$ by
    $q^2-M$, then Taylor-expands the external momenta to the superficial degree
    of divergence. The massive quark denominators $q^2-m_q^2$ remain intact.
    This is calculated independently of the full UV expansion above.

    Each internal gluon's longitudinal numerator contains an additional $1/q^2$.
    The calculation isolates these factors before massification, including both
    internal gluons of the non-Abelian vertex and gluon loop. Recombining the
    pieces reproduces the original generated numerator exactly. Their powers
    are passed to the graph's existing denominator API.

    After tensor reduction, Symbolica partial fractions separate tadpoles at
    $M$ and $m_q^2$. Reconstructing the rational integrand checks the decomposition
    before IBP. Five powers reduce to one tadpole per mass, with the shared
    OneLOop pole $A_0(m^2)=m^2/\epsilon$.
    """),
            irr_solution,
            mo.md(r"""
    The physical poles agree with the full UV expansion. The extra local gluon
    pole is $C_A(1+3\xi)M g^{\mu\nu}/(4\epsilon)$. Solving the same matrix
    from the eight generated CT diagrams therefore gives
    $$\delta Z_{Am}=-\frac{C_A(1+3\xi)}{8\epsilon},\qquad \delta Z_{cm}=0.$$
    The complete projected tensor amplitudes cancel against those CT insertions.
    The quark loop contributes no auxiliary-mass term because its physical mass
    was retained. This result is for massive quarks; massifying an initially
    massless quark loop is a different prescription.
    """),
            irr_deltas,
        ]
    )


@app.cell
def _(mo):
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    flavor_count = mo.ui.slider(0, 20, step=1, value=5, label="Quark flavors Nf")
    color_count = mo.ui.slider(2, 5, step=1, value=3, label="Colors Nc")
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    rearrangement = mo.ui.dropdown(
        ["Full UV expansion", "Direct massive IRR"],
        value="Full UV expansion",
        label="Rearrangement",
    )
    mo.vstack(
        [
            mo.hstack([gauge_parameter, flavor_count, color_count]),
            mo.hstack([subtraction_scheme, rearrangement]),
        ]
    )
    return (
        color_count,
        flavor_count,
        gauge_parameter,
        rearrangement,
        subtraction_scheme,
    )


@app.cell
def _(
    CA,
    CF,
    Nf,
    Symbol,
    color_count,
    counterterm_names,
    flavor_count,
    gauge_parameter,
    irr_residues,
    mo,
    rearrangement,
    residues,
    subtraction_scheme,
    vertex_residue,
    xi,
):
    _n = color_count.value
    _parameters = {
        CA: _n,
        CF: (_n**2 - 1) / (2 * _n),
        Nf: flavor_count.value,
        xi: gauge_parameter.value,
    }
    _c_delta = complex(((4 * Symbol.PI).log() - Symbol.EULER_GAMMA).evaluate({})).real
    _finite_factor = _c_delta if subtraction_scheme.value == "MSbar" else 0.0
    _active_residues = (
        irr_residues if rearrangement.value == "Direct massive IRR" else residues
    )
    selected_residues = [
        complex(_value.evaluate(_parameters)).real for _value in _active_residues
    ]
    _vertex = complex(vertex_residue.evaluate(_parameters)).real
    vertex_error = abs(
        selected_residues[0] + selected_residues[5] + selected_residues[2] / 2 + _vertex
    )
    beta0 = -2 * selected_residues[5]
    assert vertex_error < 1e-12
    assert selected_residues[2] == selected_residues[3]
    assert abs(beta0 - (11 * _n - 2 * flavor_count.value) / 3) < 1e-12
    mo.vstack(
        [
            mo.md(
                r"**Counterterms** — vary gauge, flavor count, SU(Nc) color group and subtraction scheme."
            ),
            mo.ui.table(
                [
                    {
                        "Constant": _name,
                        "1/ε coefficient": _value,
                        "Finite subtraction coefficient": _finite_factor * _value,
                    }
                    for _name, _value in zip(
                        counterterm_names, selected_residues, strict=True
                    )
                ],
                selection=None,
            ),
            mo.md(
                f"Quark–gluon vertex cancellation residual: **{vertex_error:.1e}**.  "
                + rf"One-loop $\beta_0$: **{beta0:.6g}**."
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
