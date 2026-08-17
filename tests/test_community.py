"""Tests for symbolica-community extensions."""
import pytest


def test_spenso_import():
    """Test that spenso module can be imported."""
    from symbolica.community.spenso import Tensor, TensorIndices, Representation
    assert Tensor is not None
    assert TensorIndices is not None
    assert Representation is not None


def test_idenso_import():
    """Test that the Idenso module is registered by the native extension."""
    from symbolica.community.idenso import simplify_metrics

    assert simplify_metrics is not None


def test_vakint_import():
    """Test that the Vakint module is registered by the native extension."""
    from symbolica.community.vakint import (
        Vakint,
        VakintEvaluationMethod,
        VakintExpression,
    )

    assert Vakint is not None
    assert VakintEvaluationMethod is not None
    assert VakintExpression is not None
