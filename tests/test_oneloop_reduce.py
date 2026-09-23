"""One-loop reduction and master evaluation in the host's Symbolica kernel."""

import importlib
import math
import signal
import subprocess
import sys

import pytest

from symbolica import E, Expression, S
from symbolica.community import hep
from symbolica.community.hep import oneloop


def evaluate_coefficients(reduction, parameters, values, mu_squared=None):
    coefficients = oneloop.reduction_coefficients(reduction, mu_squared=mu_squared)
    assert len(coefficients) == 3
    assert all(isinstance(coefficient, Expression) for coefficient in coefficients)
    evaluator = oneloop.compile_native(coefficients, parameters)
    return [complex(value) for value in evaluator.evaluate_complex(values).reshape(-1)]


def test_namespace_and_dot_attributes():
    assert importlib.import_module("symbolica.community.hep.oneloop") is oneloop
    for name in ("Propagator", "IntegralFamily", "Reduction", "MasterIntegral", "Evaluator"):
        assert getattr(oneloop, name).__module__ == "symbolica.community.hep.oneloop"
    assert oneloop.EXPRESSION_INTEROP

    # The import must initialize these attributes before user code can create
    # the symbol with incompatible defaults in the shared symbol table.
    dot = S("oneloopreduce::dot", is_symmetric=True, is_linear=True)
    k, q1, q2 = S("oneloopreduce::k", "oneloopreduce::q1", "oneloopreduce::q2")
    assert dot(q1, k) == dot(k, q1)
    assert dot(k, q1 + q2) == dot(k, q1) + dot(k, q2)
    with pytest.raises(TypeError):
        S("oneloopreduce::dot", is_symmetric=False)


def test_symbolic_reduction_keeps_numerical_initialization_lazy():
    script = """
from symbolica import E, S
from symbolica.community.hep import oneloop
assert not oneloop.is_initialized()
dot = S("oneloopreduce::dot", is_symmetric=True, is_linear=True)
family = oneloop.IntegralFamily([oneloop.Propagator(E("0"))] * 2, [E("s")])
reduction = family.reduce().simplify()
assert reduction.to_oneloopmaster() == E("oneloopmaster::B0(s,0,0,1)")
assert not oneloop.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    if result.returncode == -signal.SIGABRT and (
        "Cannot start another restricted Symbolica thread while this user's thread allowance is in use."
        in output
    ):
        pytest.skip("The parent pytest process holds the restricted Symbolica thread allowance")
    assert result.returncode == 0, output


@pytest.mark.parametrize(
    "kind,head,masses,invariants,arguments",
    [
        ("tadpole", "A0", ["m0"], [], ["m0"]),
        ("bubble", "B0", ["m0", "m1"], ["p01"], ["p01", "m0", "m1"]),
        (
            "triangle", "C0", ["m0", "m1", "m2"], ["p01", "p02", "p12"],
            ["p01", "p12", "p02", "m0", "m1", "m2"],
        ),
        (
            "box", "D0", ["m0", "m1", "m2", "m3"],
            ["p01", "p02", "p03", "p12", "p13", "p23"],
            ["p01", "p12", "p23", "p03", "p02", "p13", "m0", "m1", "m2", "m3"],
        ),
    ],
)
def test_scalar_master_mapping(kind, head, masses, invariants, arguments):
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E(mass)) for mass in masses], list(map(E, invariants)),
    )
    reduction = family.reduce().simplify()
    ((coefficient, master),) = reduction.terms
    assert isinstance(reduction, oneloop.Reduction)
    assert isinstance(master, oneloop.MasterIntegral)
    assert coefficient == E("1")
    assert (master.kind, master.head) == (kind, head)
    assert master.arguments == list(map(E, arguments))
    assert family.exponents == [1] * len(masses)
    assert [propagator.mass_sq for propagator in family.propagators] == list(map(E, masses))

    symbolic_master = S(f"oneloopreduce::{head}")(*map(E, arguments))
    assert reduction.to_expression() == master.to_expression() == symbolic_master
    for scale in (None, E("renormalization_scale_squared")):
        expected = S(f"oneloopmaster::{head}")(
            *map(E, arguments), E("1") if scale is None else scale,
        )
        assert master.to_oneloopmaster(mu_squared=scale) == expected
        assert reduction.to_oneloopmaster(mu_squared=scale) == expected


def test_rank_one_triangle_matches_the_scalar_master_combination():
    dot, k, q1 = S("oneloopreduce::dot", "oneloopreduce::k", "oneloopreduce::q1")
    mass, p1, p2, s, scale = S(
        "oneloop_test::mass", "oneloop_test::p1", "oneloop_test::p2",
        "oneloop_test::s", "oneloop_test::scale",
    )
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(mass)] * 3, [p1, s, p2], numerator=dot(k, q1),
    )
    reduction = family.reduce().simplify()
    assert sorted(master.kind for _, master in reduction.terms) == ["bubble", "bubble", "triangle"]
    bubble, triangle = S("oneloopreduce::B0", "oneloopreduce::C0")
    expected = (bubble(s, mass, mass) - bubble(p2, mass, mass) - p1 * triangle(p1, p2, s, mass, mass, mass)) / 2
    assert (reduction.to_expression() - expected).expand() == E("0")

    actual = evaluate_coefficients(
        reduction, [mass, p1, p2, s, scale], [2 + 0j, -1 + 0j, -2 + 0j, -3 + 0j, 5 + 0j],
        mu_squared=scale,
    )
    bubble_s = oneloop.B0(-3, 2, 2, 5)
    bubble_p2 = oneloop.B0(-2, 2, 2, 5)
    scalar_triangle = oneloop.C0(-1, -2, -3, 2, 2, 2, 5)
    expected_values = [(a - b + c) / 2 for a, b, c in zip(bubble_s, bubble_p2, scalar_triangle)]
    assert actual == pytest.approx(expected_values, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("scale", [None, E("5")])
def test_raised_tadpole_retains_the_dimension_dependent_finite_part(scale):
    mass = S("oneloop_test::tadpole_mass_squared")
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(mass)], [], exponents=[2],
    ).reduce().simplify()
    ((coefficient, master),) = reduction.terms
    dimension = S("oneloopreduce::d")
    assert (coefficient - (dimension - 2) / (2 * mass)).expand() == E("0")
    assert master.head == "A0"
    actual = evaluate_coefficients(reduction, [mass], [2 + 0j], mu_squared=scale)
    scale_value = 1 if scale is None else 5
    # Differentiating A0 with respect to mass squared removes its finite +1.
    assert actual == pytest.approx([-math.log(2 / scale_value), 1, 0], rel=1e-12, abs=1e-12)


def test_raised_massless_bubble_mixes_pole_into_the_finite_part():
    s = S("oneloop_test::bubble_s")
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(E("0"))] * 2, [s], exponents=[2, 1],
    ).reduce().simplify()
    ((coefficient, master),) = reduction.terms
    dimension = S("oneloopreduce::d")
    assert (coefficient + (dimension - 3) / s).expand() == E("0")
    assert master.head == "B0"
    actual = evaluate_coefficients(reduction, [s], [-3 + 0j], mu_squared=E("2"))
    finite, pole, double_pole = oneloop.B0(-3, 0, 0, 2)
    expected = [(-finite + 2 * pole) / -3, (-pole + 2 * double_pole) / -3, -double_pole / -3]
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert actual == pytest.approx([-math.log(3 / 2) / 3, 1 / 3, 0], rel=1e-12, abs=1e-12)


def test_quadratic_dimension_coefficient_mixes_double_pole_into_finite_part():
    dimension, s = S("oneloopreduce::d", "oneloop_test::triangle_s")
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(E("0"))] * 3, [E("0"), s, E("0")],
        numerator=(dimension - 4) ** 2,
    ).reduce().simplify()
    coefficients = oneloop.reduction_coefficients(reduction)
    assert coefficients[1:] == [E("0"), E("0")]
    actual = evaluate_coefficients(reduction, [s], [-3 + 0j])
    double_pole = oneloop.C0(0, 0, -3, 0, 0, 0)[2]
    assert abs(double_pole) > 0
    # (d-4)^2 = 4*eps^2, so only the master's double pole contributes.
    assert actual == pytest.approx([4 * double_pole, 0, 0], rel=1e-12, abs=1e-12)


def test_coefficient_pole_requires_unavailable_higher_order_masters():
    dimension = S("oneloopreduce::d")
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(E("2"))], [], numerator=1 / (dimension - 4),
    ).reduce().simplify()
    # Exact symbolic reduction remains useful even when finite/pole evaluation
    # would require the master's positive powers of epsilon.
    assert reduction.to_expression() == S("oneloopreduce::A0")(2) / (dimension - 4)
    assert reduction.to_oneloopmaster() == S("oneloopmaster::A0")(2, 1) / (dimension - 4)
    with pytest.raises(ValueError, match="pole at d=4"):
        oneloop.reduction_coefficients(reduction)


def test_fractional_dimension_coefficient_is_rejected():
    dimension = S("oneloopreduce::d")
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(E("2"))], [], numerator=(dimension - 4) ** E("1/2"),
    ).reduce()
    with pytest.raises(ValueError, match="integer-power Taylor expansion"):
        oneloop.reduction_coefficients(reduction)


@pytest.mark.parametrize("dependent_argument", ["scale", "mass", "invariant"])
def test_dimension_dependent_master_arguments_are_rejected(dependent_argument):
    dimension = S("oneloopreduce::d")
    mass = dimension if dependent_argument == "mass" else E("2")
    invariant = dimension if dependent_argument == "invariant" else E("-3")
    scale = dimension if dependent_argument == "scale" else None
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(mass)] * 2, [invariant],
    ).reduce().simplify()
    assert isinstance(reduction.to_oneloopmaster(mu_squared=scale), Expression)
    with pytest.raises(ValueError, match="independent of oneloopreduce::d"):
        oneloop.reduction_coefficients(reduction, mu_squared=scale)


@pytest.mark.parametrize("zero_kind", ["scaleless", "zero_numerator"])
def test_zero_reduction_has_three_zero_expression_coefficients(zero_kind):
    mass = E("0") if zero_kind == "scaleless" else E("2")
    numerator = E("0") if zero_kind == "zero_numerator" else E("1")
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(mass)], [], numerator=numerator,
    ).reduce().simplify()
    assert reduction.to_expression() == E("0")
    assert reduction.to_oneloopmaster() == E("0")
    coefficients = oneloop.reduction_coefficients(reduction)
    assert all(isinstance(value, Expression) for value in coefficients)
    assert coefficients == [E("0"), E("0"), E("0")]


def test_reduction_coefficients_are_shared_with_feynkit():
    mass = S("oneloop_test::shared_mass_squared")
    reduction = oneloop.IntegralFamily([oneloop.Propagator(mass)], []).reduce()
    coefficients = oneloop.reduction_coefficients(reduction)
    assert all(isinstance(value, Expression) for value in coefficients)
    expression = hep.TensorReducer.feynkit(E("4")).reduce(coefficients[0] + coefficients[1])
    evaluator = oneloop.compile_native([expression], [mass])
    value = complex(evaluator.evaluate_complex([2 + 0j]).reshape(-1)[0])
    assert value == pytest.approx(2 * (2 - math.log(2)), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "count,invariants,exponents",
    [(0, [], None), (2, [], None), (2, ["s", "t"], None), (2, ["s"], [1])],
)
def test_malformed_family_raises_value_error(count, invariants, exponents):
    with pytest.raises(ValueError):
        oneloop.IntegralFamily(
            [oneloop.Propagator(E("0"))] * count, list(map(E, invariants)), exponents=exponents,
        )


@pytest.mark.parametrize("exponents", [[40, 1], [-1, 1]])
def test_unsupported_indices_raise_value_error(exponents):
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E("0"))] * 2, [E("s")], exponents=exponents,
    )
    with pytest.raises(ValueError):
        family.reduce()
