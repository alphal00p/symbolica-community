import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium", app_title="QCD annihilation: color sums and cross sections"
)


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep
    from symbolica.community.hep import Model
    from symbolica.community.spenso import ColorSimplifySettings, TensorExpression

    _set_namespace("qcd_annihilation")

    return ColorSimplifySettings, E, Model, S, Symbol, TensorExpression, hep, mo


@app.cell
def _(Model):
    qcd_model = Model.standard_model()
    return (qcd_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Different-flavor QCD annihilation

    [Browse all notebooks](/) · [Elastic quark scattering](/?file=hep/quark_scattering.py)

    Generate $b\bar b\to t\bar t$ through gluon exchange, retaining both
    quark masses. Initial spins and colors are averaged; final spins and colors
    are summed. `Particle.color_sum` supplies the shared completeness tensor,
    and Idenso contracts the color and Dirac algebra.

    The symbolic $SU(N_c)$ result uses $T_R=1/2$ and agrees with the
    [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QiQibar-QjQjbar).
    The massless $SU(3)$ limit is carried through two-body phase space below.
    """)


@app.cell
def _(ColorSimplifySettings, E, S, Symbol, TensorExpression, hep, mo, qcd_model):
    _result = hep.Generator(qcd_model).generate(
        hep.Process.cross_section(["b", "b~"], ["t", "t~"]).with_loop_count(1, 1),
        max_vertices=4,
        maximum_bridges=None,
        vertex_allow=["V_76", "V_137"],
        numerator_grouping=None,
        progress=None,
    )
    assert len(_result.diagrams) == 1
    qcd_diagram = _result.diagrams[0]
    assert len(qcd_diagram.cuts) == 1
    _cut = qcd_diagram.cuts[0]
    assert sorted(_p.name for _p in _cut.particles) == ["t", "t~"]
    assert all(_side.loop_count == 0 for _side in (_cut.left, _cut.right))
    _Q, _P, _K = S("gammalooprs::Q", "gammalooprs::P", "gammalooprs::K")
    _s, _t, _u, _mb, _mt, _gs, _k1, _k2 = S(
        "s", "t", "u", "UFO::MB", "UFO::MT", "UFO::G", "k1", "k2"
    )
    _kinematics = hep.Kinematics.mandelstam(
        [_P(0), _P(1), _k1, _k2], [_mb**2, _mb**2, _mt**2, _mt**2], [_s, _t, _u]
    )
    _projector = qcd_diagram.projector_expression()
    for _edge in qcd_diagram.external_edges:
        _projector = qcd_model.particle(_edge.particle_name).sum_spins(
            _projector, _Q(_edge.id), edge=_edge.id, average=True
        )
    _numerator = qcd_model.expand_couplings(
        qcd_diagram.numerator_expression().to_expression() * _projector
    )

    # Associate open color slots with the generated external wavefunction indices.
    # Slot duals determine the closure orientation, including the antiquark; no
    # fixed half-edge numbering or hand-written SU(3) averaging factor is needed.
    _bis, _wave, _index, _metric, _left, _right = S(
        "spenso::bis", "wave_", "index_", "spenso::g", "left_", "right_"
    )
    _slots = TensorExpression(_numerator).structure.slots
    _color_projector = E("1")
    _initial_color_states = 1
    for _edge in qcd_diagram.external_edges:
        _particle = qcd_model.particle(_edge.particle_name)
        _initial_color_states *= abs(_particle.color)
        _indices = [
            dict(_match)[_index]
            for _match in qcd_diagram.projector_expression().match(
                _wave(_edge.id, _bis(4, _index)), max_level=0
            )
        ]
        _closure_slots = [
            _slot.dual().to_expression()
            for _slot in _slots
            if any(
                _slot.to_expression().replace(_i, E("0")) != _slot.to_expression()
                for _i in _indices
            )
        ]
        assert len(_closure_slots) == 2
        _closure = _metric(*_closure_slots)
        _color_indices = dict(
            next(_closure.match(_particle.color_sum(_left, _right), max_level=0))
        )
        _color_projector *= _particle.color_sum(
            _color_indices[_left], _color_indices[_right], average=True
        )

    # Check singlet and adjoint sums through the same installed public interface.
    _i, _j = S("color_i", "color_j")
    assert qcd_model.particle("e-").color_sum(_i, _j, average=True) == E("1")
    for _name in ("b", "b~", "g"):
        _identity = qcd_model.particle(_name).color_sum(_i, _i, average=True)
        assert TensorExpression(_identity).simplify_metrics().to_expression() == E("1")

    # A named adjoint dimension is required by Spenso. Impose dA=Nc²-1 after
    # contraction, when the dimensions are ordinary scalar expressions.
    _Nc, _dA, _cof, _coad = S("Nc", "dA", "spenso::cof", "spenso::coad")
    _generic = (
        (_numerator * _color_projector * _initial_color_states / _Nc**2)
        .replace(_cof(3, _index), _cof(_Nc, _index))
        .replace(_coad(8, _index), _coad(_dA, _index))
    )
    _contracted = (
        TensorExpression(_generic.expand())
        .simplify_color(ColorSimplifySettings(substitute_cof_dimension_invariants=True))
        .simplify_gamma()
        .expand()
        .to_dots()
    )
    assert _contracted.is_scalar
    _loop_edge = qcd_diagram.loop_momentum_basis.loop_edges[0]
    _loop_particle = next(
        _p.name
        for _edge, _p in zip(_cut.edges, _cut.particles)
        if _edge.id == _loop_edge
    )
    _physical_momentum = _k1 if _loop_particle == "t" else _k2
    _routed = qcd_diagram.loop_momentum_basis.route_expression(
        _contracted.to_expression()
    ).replace(_K(0, _index), _cut.orientations[_loop_edge] * _physical_momentum(_index))
    _a, _b, _c, _inverse, _denom = S("a_", "b_", "c_", "inverse_", "gammalooprs::denom")
    _denominator = _kinematics.apply(
        qcd_diagram.denominator_expression(
            edge_powers={_edge.id: 0 for _edge in _cut.edges}, dimension=4, in_lmb=True
        )
        .to_expression()
        .replace(_denom(_a, _b, _c, _inverse), _inverse)
    )
    assert (_denominator - _s**2).expand() == E("0")
    qcd_squared = (
        qcd_diagram.overall_factor_expression(evaluate=True)
        * qcd_diagram.numerator_prefactor_expression()
        * _kinematics.apply(_routed)
        / _denominator
    ).replace(_dA, _Nc**2 - 1)
    _expected = (
        (_Nc**2 - 1)
        * _gs**4
        / (2 * _Nc**2 * _s**2)
        * (
            2 * _mb**2 * (2 * _mt**2 + _s - _t - _u)
            + 2 * _mb**4
            + 2 * _mt**4
            + 2 * _mt**2 * (_s - _t - _u)
            + _t**2
            + _u**2
        )
    )
    assert (qcd_squared - _expected).replace(
        _u, 2 * _mb**2 + 2 * _mt**2 - _s - _t
    ).together() == E("0")
    qcd_massless = qcd_squared.replace(_mb, E("0")).replace(_mt, E("0"))
    assert (
        qcd_massless - (_Nc**2 - 1) * _gs**4 * (_t**2 + _u**2) / (2 * _Nc**2 * _s**2)
    ).replace(_u, -_s - _t).together() == E("0")
    _su3 = qcd_massless.replace(_Nc, E("3"))
    assert (_su3 - 4 * _gs**4 * (_t**2 + _u**2) / (9 * _s**2)).replace(
        _u, -_s - _t
    ).together() == E("0")

    # Carry the generated massless SU(3) result through physical two-body phase space.
    _s_physical = S("s_physical", is_positive=True)
    _costheta, _alpha_s = S("costheta", "alpha_s")
    _pi = Symbol.PI
    _massless_kinematics = hep.Kinematics.mandelstam(
        [_P(0), _P(1), _k1, _k2], [E("0")] * 4, [_s_physical, _t, _u]
    )
    _angular = (
        _su3.replace(_s, _s_physical)
        .replace(_t, -_s_physical * (1 - _costheta) / 2)
        .replace(_u, -_s_physical * (1 + _costheta) / 2)
        .replace(_gs**4, (4 * _pi * _alpha_s) ** 2)
    )
    qcd_differential = (
        _angular
        * _massless_kinematics.two_body_phase_space(_k1, _k2)
        / _massless_kinematics.flux(_P(0), _P(1))
    ).expand()
    assert (
        qcd_differential - _alpha_s**2 * (1 + _costheta**2) / (18 * _s_physical)
    ).together() == E("0")
    _primitive = qcd_differential.to_polynomial().integrate(_costheta).to_expression()
    qcd_total = (
        2
        * _pi
        * (
            _primitive.replace(_costheta, E("1"))
            - _primitive.replace(_costheta, E("-1"))
        )
    )
    assert (qcd_total - 8 * _pi * _alpha_s**2 / (27 * _s_physical)).together() == E("0")

    mo.vstack(
        [
            qcd_diagram,
            mo.md("**Initial color projector, including the color average**"),
            _color_projector,
            mo.md("**Massive, spin- and color-averaged squared matrix element**"),
            qcd_squared.replace(_u, 2 * _mb**2 + 2 * _mt**2 - _s - _t).together(),
            mo.md("**Massless SU(3) result**"),
            qcd_massless.replace(_Nc, E("3")).together(),
            mo.md("**Massless angular distribution**"),
            qcd_differential,
            mo.md("**Integrated massless cross section**"),
            qcd_total.together(),
        ]
    )
    return qcd_diagram, qcd_squared, qcd_massless, qcd_differential, qcd_total


if __name__ == "__main__":
    app.run()
