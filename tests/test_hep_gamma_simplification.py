"""Run the gamma notebook's default identity and exact component checks."""

from pathlib import Path

import pytest


def test_gamma_notebook_preserves_identity_and_component_oracles(monkeypatch):
    pytest.importorskip("marimo", minversion="0.24.0")
    notebook_directory = Path(__file__).resolve().parents[1] / "examples/hep"
    monkeypatch.syspath_prepend(str(notebook_directory))
    from examples.hep.gamma_simplification import app

    # The cells retain their explicit Clifford, trace and HEP component oracles.
    # Optional benchmark buttons stay off; the dedicated harness tests those routes.
    app.run()
