"""Exercise color-first contraction without running the interactive notebook."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from symbolica import E, S, Expression
from symbolica.community.hep import Model, SnailFilterOptions
from symbolica.community.spenso import ColorSimplifySettings, TensorExpression


@pytest.fixture(scope="module")
def notebook():
    pytest.importorskip("marimo", minversion="0.24.0")
    from examples.hep_showcase import app

    definitions = {}
    cells = list(app._cell_manager.cells())
    for name in (
        "D",
        "project_color",
        "external_projector",
        "route_momenta",
        "diagram_weight",
        "graph_propagators",
        "uv_expansion_data",
    ):
        cell = next(cell for cell in cells if cell is not None and name in cell.defs)
        _, values = cell.run()
        definitions.update(values)
    model = Model(str(Path(__file__).parents[1] / "examples/hep_sm.json"))
    diagrams = model.generate_diagrams(
        ["g"],
        ["g"],
        loops=1,
        coupling_orders={"QCD": 2, "QED": 0},
        particle_veto=["c", "t", "s", "u", "d"],
        zero_snails=SnailFilterOptions(),
    )
    return SimpleNamespace(**definitions, model=model, diagrams=diagrams)


def test_color_projection_preserves_only_the_lorentz_interface(notebook):
    n = notebook
    for diagram in n.diagrams:
        original = diagram.to_json()
        color, stripped = n.project_color(diagram)
        assert isinstance(color, Expression)
        assert TensorExpression(color).is_scalar
        assert color != E('0')
        assert isinstance(stripped, TensorExpression)
        assert stripped.rank == 2
        assert all(slot.to_expression().get_name() == 'spenso::mink'
                   for slot in stripped.interface)
        assert all(structure == E('1') for structure, _ in stripped.expand_color())
        for longitudinal in (False, True):
            projector = n.external_projector(stripped, longitudinal=longitudinal)
            assert projector.rank == 2
            assert set(projector.list_dangling()) == set(stripped.list_dangling())
        assert diagram.to_json() == original


def test_color_projection_leaves_dirac_contractions_for_later(notebook):
    n = notebook
    bottom = next(d for d in n.diagrams if d.internal_edges[0].particle_name == "b")
    _, stripped = n.project_color(bottom)
    gamma = S("spenso::gamma")
    assert gamma in stripped.to_expression().get_all_symbols()
    contracted = stripped.simplify_gamma().schoonschip_net(expand_contracted_sums=True)
    assert gamma not in contracted.to_expression().get_all_symbols()
    assert set(contracted.list_dangling()) == set(stripped.list_dangling())


def test_distinct_color_structures_keep_their_relative_coefficients(notebook):
    n = notebook
    a, b, c, d = (n.adjoint(name) for name in ('a', 'b', 'c', 'd'))
    mu, nu = n.lorentz('mu'), n.lorentz('nu')
    metric = TensorExpression.g(n.lorentz)(mu, nu)
    momentum_pair = n.P(mu).outer(n.P(nu)) / n.p2
    color_metric = TensorExpression.g(n.adjoint)(a, b)
    f = TensorExpression.f(8)
    color_pair = f(a, c, d).to_expression() * f(b, c, d).to_expression()
    # Preserve the explicit Einstein indices rather than composing tensor channels.
    numerator = TensorExpression(
        color_metric.to_expression() * metric.to_expression()
        + color_pair * momentum_pair.to_expression()
    )
    diagram = SimpleNamespace(numerator_expression=lambda: numerator)

    color, stripped = n.project_color(diagram)

    # The normalized adjoint trace is 1 for delta_ab and C_A = 3 for f_acd f_bcd.
    expected = metric + 3 * momentum_pair
    assert (color * stripped - expected).expand() == E('0')
    assert stripped.rank == 2
    assert all(structure == E('1') for structure, _ in stripped.expand_color())


def test_zero_color_projection_can_pass_through_lorentz_projection(notebook):
    n = notebook
    a, b, c, d, e = (n.adjoint(name) for name in ('a', 'b', 'c', 'd', 'e'))
    mu, nu = n.lorentz('mu'), n.lorentz('nu')
    f, g = TensorExpression.f(8), TensorExpression.g(n.adjoint)
    # Both the internal trace and the external color average kill this structure.
    color_tensor = (f(a, b, c).to_expression() * f(c, d, e).to_expression()
                    * g(d, e).to_expression())
    numerator = TensorExpression(
        color_tensor * TensorExpression.g(n.lorentz)(mu, nu).to_expression()
    )
    diagram = SimpleNamespace(numerator_expression=lambda: numerator)

    color, stripped = n.project_color(diagram)

    assert color == E('0')
    assert stripped == E('0')
    assert n.apply_projector(stripped, n.external_projector(stripped)) == E('0')


def test_global_color_factor_preserves_full_projection_and_uv_residues(notebook):
    n = notebook
    color_settings = ColorSimplifySettings(substitute_cof_dimension_invariants=True)

    def contract_full(expression):
        """Reference pipeline: retain color during Dirac and Lorentz contraction."""
        for _ in range(12):
            simplified = (
                expression.expand()
                .simplify_gamma()
                .simplify_color(color_settings)
                .simplify_metrics()
                .expand()
            )
            if simplified == expression:
                return simplified
            expression = simplified
        raise AssertionError("Reference contractions did not reach a fixed point")

    assert {d.internal_edges[0].particle_name for d in n.diagrams} == {"g", "ghG", "b"}
    for diagram in n.diagrams:
        color, stripped = n.project_color(diagram)
        routed = n.route_momenta(
            diagram,
            stripped.simplify_gamma().schoonschip_net(expand_contracted_sums=True),
        )
        full = diagram.numerator_expression().with_lorentz_dimension(n.D)
        a, b = [
            slot
            for slot in full.interface
            if slot.to_expression().get_name() == "spenso::coad"
        ]
        color_projector = TensorExpression.g(n.adjoint)(a, b) / n.adjoint.dimension
        routed_full = n.route_momenta(diagram, contract_full(full))
        propagators = n.graph_propagators(diagram, n.model)
        weight = n.diagram_weight(diagram)
        for longitudinal in (False, True):
            projector = n.external_projector(stripped, longitudinal=longitudinal)
            projected = n.apply_projector(routed, projector).to_dots().expand()
            weighted = TensorExpression(
                n.model.expand_couplings(
                    projected * weight,
                )
            ).expand()
            assert weighted.is_scalar

            full_projector = TensorExpression(
                color_projector.to_expression() * projector.to_expression()
            )
            full_projected = (
                contract_full(
                    n.apply_projector(
                        routed_full,
                        full_projector,
                    )
                )
                .to_dots()
                .expand()
            )
            full_weighted = TensorExpression(
                n.model.expand_couplings(
                    full_projected * weight,
                )
            ).expand()
            assert (color * weighted - full_weighted).expand().cancel() == E("0")

            residue = (
                color * n.uv_expansion_data(weighted * propagators, diagram)["residue"]
            )
            full_residue = n.uv_expansion_data(full_weighted * propagators, diagram)[
                "residue"
            ]
            assert (residue - full_residue).expand().cancel() == E("0")


@pytest.mark.parametrize("particle", ["g", "ghG", "b"])
def test_contraction_preserves_unrelated_scalar_factors(notebook, particle):
    n = notebook
    diagram = next(
        d for d in n.diagrams if d.internal_edges[0].particle_name == particle
    )
    x, y, z, w = S(
        "factorized_contraction::x",
        "factorized_contraction::y",
        "factorized_contraction::z",
        "factorized_contraction::w",
        is_scalar=True,
    )
    powers = [(x + y) ** 12, (z + w) ** 12]
    spectator = powers[0] * powers[1]
    numerator = diagram.numerator_expression()
    weighted_diagram = SimpleNamespace(
        numerator_expression=lambda: TensorExpression(
            spectator * numerator.to_expression()
        )
    )
    color, stripped = n.project_color(weighted_diagram)
    contracted = stripped.simplify_gamma().schoonschip_net(expand_contracted_sums=True)
    routed = n.route_momenta(diagram, contracted)
    projected = n.apply_projector(routed, n.external_projector(routed)).to_dots()
    for stage in (stripped, contracted, routed, projected):
        assert stage != E("0")
        for power in powers:
            assert stage.replace(power, 0) == E("0")
    assert set(contracted.list_dangling()) == set(stripped.list_dangling())
    assert S("spenso::gamma") not in contracted.to_expression().get_all_symbols()
    assert projected.is_scalar
    # Remove the deliberately large coefficients before the small reference check.
    unweighted = projected.replace(powers[0], 1).replace(powers[1], 1)
    plain_color, plain = n.project_color(diagram)
    reference = n.route_momenta(diagram, plain.simplify_gamma())
    reference = n.apply_projector(reference, n.external_projector(reference)).to_dots()
    difference = (color * unweighted - plain_color * reference).expand()
    assert difference.simplify_metrics().to_dots().expand() == E("0")


@pytest.fixture(scope="module")
def uv_reference():
    from examples.hep_showcase_uv import app

    definitions = {}
    cells = list(app._cell_manager.cells())
    for name in ("bubble_uv_data", "external_projector", "numerator_in_d"):
        cell = next(cell for cell in cells if cell is not None and name in cell.defs)
        _, values = cell.run()
        definitions.update(values)
    return SimpleNamespace(**definitions)


@pytest.mark.parametrize(
    ("particle", "transverse", "longitudinal"),
    [("ghG", "1/4", "3/4"), ("g", "19/4", "-3/4"), ("b", "-2/3", "0")],
)
def test_factorized_uv_reference(
    notebook, uv_reference, particle, transverse, longitudinal
):
    diagram = next(
        d for d in notebook.diagrams if d.internal_edges[0].particle_name == particle
    )
    scale = S("hep_gluon::p2") * S("UFO::G") ** 2
    for is_longitudinal, coefficient in ((False, transverse), (True, longitudinal)):
        result = uv_reference.bubble_uv_data(
            diagram,
            notebook.model,
            longitudinal=is_longitudinal,
        )
        assert (result["residue"] - E(coefficient) * scale).expand() == E("0")

    x, y = S("uv_spectator::x", "uv_spectator::y", is_scalar=True)
    spectator = (x + y) ** 12
    numerator = uv_reference.numerator_in_d(diagram)
    projected = uv_reference.apply_projector(
        TensorExpression(spectator * numerator.to_expression()),
        uv_reference.external_projector(diagram),
    )
    assert projected.is_scalar
    assert projected != E("0")
    assert projected.replace(spectator, 0) == E("0")
