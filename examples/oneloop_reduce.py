"""Reduce a raised tadpole and evaluate its finite and pole coefficients.

Run with the community Symbolica build and NumPy installed.
"""

from math import isclose, log

from symbolica import E, S
from symbolica.community.hep import oneloop


def main():
    mass_squared = S("oneloop_example::mass_squared")
    family = oneloop.IntegralFamily(
        propagators=[oneloop.Propagator(mass_squared)],
        invariants=[],
        exponents=[2],
    )
    reduction = family.reduce().simplify()
    print("Reduction:", reduction.to_expression())
    print("OneLoopMaster expression:", reduction.to_oneloopmaster())

    # Keep the O(eps) part of (d-2)/(2*m²): it multiplies the A0 pole
    # and contributes -1 to the finite part.
    coefficients = oneloop.reduction_coefficients(reduction, mu_squared=E("1"))
    evaluator = oneloop.compile_native(coefficients, [mass_squared])
    values = evaluator.evaluate_complex([2 + 0j]).reshape(-1)
    print("At m²=2 (finite, 1/eps, 1/eps²):", values)
    assert isclose(values[0].real, -log(2), rel_tol=1e-12)
    assert abs(values[0].imag) < 1e-12
    assert abs(values[1] - 1) < 1e-12
    assert abs(values[2]) < 1e-12


if __name__ == "__main__":
    main()
