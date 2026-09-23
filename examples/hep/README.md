# HEP bindings

The GammaLoop `feynkit` branch is bundled through `feynkit-py`. Its public API
is flat under `symbolica.community.hep`, sharing the same Symbolica kernel as
Spenso and Idenso:

```python
from symbolica.community.hep import FeynmanDiagram, Model, Generator, TensorReducer
```

| Component | Examples of public classes |
| --- | --- |
| `feynkit-graph` | `FeynmanDiagram`, `DiagramEdge`, `DiagramVertex`, `LoopMomentumBasis` |
| `feynkit-generator` | `Generator`, `Process`, `GenerationOptions`, `GenerationResult` |
| `feynkit-cff` | `CffGenerator`, `CffResult`, `CffSurface`, `CffOrientation` |
| `feynkit-tensor` | `TensorReducer`; reduction methods also live on `FeynmanDiagram` |
| `feynkit-py` | All of the above, plus models, UFO loading, kinematics, and jet clustering |

Class identities and the exception hierarchy are preserved; their Python
`__module__` and the bundled stubs name `symbolica.community.hep`.

The Marimo directory server (`marimo edit examples/` from the repository root)
provides separate worked IBP notebooks:

- [Unequal-mass bubble](ibp_bubble.py): raised powers and numerical master evaluation.
- [Bubble differential equations](ibp_differential_equations.py): compatible equations
  in the invariant and masses, with numerical transport from a boundary value.
- [Massless form-factor triangle](ibp_triangle.py): raised powers and infrared poles.
- [Massive two-loop 2- and 3-point functions](ibp_two_loop_masses.py): symbolic-index
  recurrences with independent masses, application to raised powers and linear
  combinations, and exact checks of the defining IBP identities.
- [Two-loop phi4 self-energy](ibp_phi4.py) and [vertex](phi4_two_loop_vertex.py):
  generated diagrams, reductions and counterterms.

They use the shared `IntegralFamily` and `IBPFamily` API, Symbolica algebra,
and native RustRed reductions. Links inside each notebook stay on the same server.

After installing a community wheel built from this checkout:

```sh
python examples/hep/diagrams.py
```

The small `scalar_phi3.json` model is a one-particle, cubic-interaction subset
of GammaLoop's `scalars_2p_3p.json` test fixture (MIT / Apache-2.0). It requires
no external model downloads or integral backend.

The bindings are included in both native and Pyodide builds. Native-only
dependencies such as Vakint remain excluded from WASM. Diagram SVG/HTML
rendering may additionally require the upstream optional `typst-py` renderer;
generation, DOT/JSON export, and CFF algebra do not require it.

Regenerate only the HEP stubs, without rewriting the core or other modules:

```sh
cargo run --bin stub_gen --no-default-features --features python_stubgen -- --hep-only
```
