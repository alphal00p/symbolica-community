import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Photon vacuum polarization")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, S
    from symbolica.community.hep import Model
    from symbolica.community.spenso import TensorExpression

    return E, Model, S, TensorExpression, mo


@app.cell
def _(Model, mo):
    _notebook_dir = mo.notebook_location()
    if _notebook_dir is None:
        raise RuntimeError("Cannot locate the notebook directory to load hep_sm.json")
    model_path = _notebook_dir.parent / "hep_sm.json"
    qed_model = Model(model_path)
    return (qed_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [Browse all notebooks](/)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Photon vacuum polarization

    Generate the electron loop and reduce its numerator in $D=4-2\epsilon$.
    The result is transverse: $(p^2g^{\mu\nu}-p^\mu p^\nu)F(p^2)$.
    Keeping $D$ symbolic until after reduction retains the finite rational term.

    The scalar masters use the shared OneLOop evaluator. The expressions below
    omit the common loop factor $i/(16\pi^2)$; the UV coefficient is
    $-4e^2/3$. Numerical values use $e=m^2=\mu^2=1$ and show the physical
    imaginary part above the pair-production threshold $p^2=4m^2$.
    """)
    return


@app.cell
def _(E, S, TensorExpression, mo, qed_model):
    from symbolica.community import hep as _fk
    from symbolica.community.hep import oneloop as _olo

    _generated = _fk.Generator(qed_model).generate(
        _fk.Process.amplitude([22], [22]).with_loop_count(1, 1),
        max_vertices=2,
        maximum_bridges=None,
        vertex_allow=["V_98"],
        numerator_grouping=None,
        progress=None,
    )
    assert len(_generated.diagrams) == 1
    _diagram = _generated.diagrams[0]
    _D, _s, _mass, _charge, _mu, _nu = S("D", "s", "UFO::Me", "UFO::ee", "mu", "nu")
    _K, _P, _mink, _metric = S(
        "gammalooprs::K", "gammalooprs::P", "spenso::mink", "spenso::g"
    )
    _index, _wave = S("index_", "wave_")
    _numerator = qed_model.expand_couplings(
        _diagram.numerator_expression(in_lmb=True).to_expression()
    )
    for _edge in _diagram.external_edges:
        _matches = list(
            _diagram.projector_expression().match(
                _wave(_edge.id, _mink(4, _index)), max_level=0
            )
        )
        assert len(_matches) == 1
        _port = dict(_matches[0])[_index]
        _numerator = _numerator.replace(
            _mink(4, _port), _mink(_D, (_mu, _nu)[_edge.external_index])
        )
    _numerator = _numerator.replace(_mink(4, _index), _mink(_D, _index))
    _trace = (
        TensorExpression(_numerator.expand())
        .simplify_gamma()
        .expand()
        .to_dots()
        .to_expression()
    )
    _kin = _fk.Kinematics(_D, momenta=[_K(0), _P(0)]).with_scalar_product(
        _P(0), _P(0), _s
    )
    photon_family = _diagram.integral_family(kinematics=_kin)
    _reducer = (
        _fk.TensorReducer(_D)
        .with_integrated_vector(_K(0, _mink(_D)))
        .with_external_vector(_P(0, _mink(_D)))
    )
    _reduced = _kin.apply(_reducer.reduce(_trace))
    _d0, _d1, _a0, _b0 = S("d0", "d1", "A0", "B0")
    _scalar = photon_family.rewrite_numerator(_reduced, [_d0, _d1]) / (_d0 * _d1)
    _scalar *= (
        _diagram.overall_factor_expression(evaluate=True)
        * _diagram.numerator_prefactor_expression()
    )
    # Equal-mass tadpoles are related by a loop-momentum shift. Polynomial
    # moments are scaleless; the bubble's remaining scalar master is B0.
    _weights = {
        E("1"): E("0"),
        1 / _d0: _a0,
        1 / _d1: _a0,
        _d0 / _d1: _s * _a0,
        _d1 / _d0: _s * _a0,
        1 / (_d0 * _d1): _b0,
    }
    _parts = _scalar.expand().coefficient_list(_d0, _d1)
    _integrated = sum(
        _coefficient * _weights[_monomial] for _monomial, _coefficient in _parts
    )
    _metric_tensor = _metric(_mink(_D, _mu), _mink(_D, _nu))
    photon_form_factor = (
        _integrated.expand().coefficient(_metric_tensor) / _s
    ).together()
    _transverse = _s * _metric_tensor - _P(0, _mink(_D, _mu)) * _P(0, _mink(_D, _nu))
    assert (_integrated - photon_form_factor * _transverse).together() == 0
    _eps, _af, _bf = S("eps", "A0_finite", "B0_finite")
    _laurent = (
        photon_form_factor.replace(_D, 4 - 2 * _eps)
        .replace(_a0, _af + _mass**2 / _eps)
        .replace(_b0, _bf + 1 / _eps)
        .series(_eps, 0, 0)
        .to_expression()
    )
    photon_uv = _laurent.coefficient(_eps**-1)
    assert (photon_uv + 4 * _charge**2 / 3).together() == 0
    _finite = dict(_laurent.expand().coefficient_list(_eps))[E("1")]
    _a_coeffs = _olo.master_coefficients(S("oneloopmaster::A0")(_mass**2, E("1")))
    _b_coeffs = _olo.master_coefficients(
        S("oneloopmaster::B0")(_s, _mass**2, _mass**2, E("1"))
    )
    _finite = _finite.replace(_af, _a_coeffs[0]).replace(_bf, _b_coeffs[0])
    photon_finite_values = [
        {
            "p²": _point,
            "Finite coefficient": str(
                _finite.evaluate({_s: _point, _mass: 1.0, _charge: 1.0})
            ),
        }
        for _point in (-1.0, 3.0, 5.0)
    ]
    mo.vstack(
        [
            _diagram,
            photon_family,
            mo.md("**Transverse form factor in terms of A₀ and B₀**"),
            photon_form_factor,
            mo.md("**Coefficient of 1/ε**"),
            photon_uv,
            mo.ui.table(photon_finite_values, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
