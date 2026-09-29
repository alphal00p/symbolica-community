"""Execute the color notebook with independent dummy scopes and component oracles."""

import pytest
from symbolica import E
from symbolica.community.spenso import TensorExpression


def test_color_algebra_notebook_preserves_explicit_and_compact_norms():
    pytest.importorskip("marimo", minversion="0.24.0")
    from examples.hep.color_algebra import app

    _, values = app.run()
    explicit = values["explicit_word"]
    conjugate = values["explicit_conjugate"]
    assert explicit.rank == conjugate.rank == 4
    assert explicit.dirac_adjoint().to_expression() == conjugate.to_expression()
    assert values["ket_word"].structure.slots == explicit.structure.slots
    assert values["bra_word"].structure.slots == conjugate.structure.slots
    assert values["scoped_norm"].is_scalar
    assert values["color_norm"].to_expression() == E("16/3")
    assert abs(complex(values["numeric_norm"]) - 16 / 3) < 1e-12
    assert abs(complex(values["weighted_numeric_norm"]) - 400 / 3) < 1e-12
    # Equal printed names do not turn two independent sums into one four-way edge.
    with pytest.raises(ValueError, match="more than two compatible tensor ports"):
        TensorExpression(explicit.to_expression() * conjugate.to_expression())
