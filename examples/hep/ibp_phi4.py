import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Two-loop phi4 IBP reduction")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, Replacement, S, Symbol
    from symbolica.community import hep

    return E, Replacement, S, Symbol, hep, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-loop $\phi^4$: IBP reduction and UV poles

    [Browse all notebooks](/) · [Integral families](/?file=hep/integral_families.py) ·
    [Unequal-mass bubble](/?file=hep/ibp_bubble.py)

    Reproduce the [FeynCalc two-loop self-energy example](https://feyncalc.github.io/FeynCalcExamples/Phi4/TwoLoops/Renormalization-SS)
    using `IntegralFamily`, the shared tensor reducer and the native IBP bridge.
    Both Laporta elimination and a parametric recurrence run in this notebook.

    The input is the reference's two scalar diagram integrands, with $ig^2$
    suppressed: $1/(4D_1^2D_2)$ and
    $1/[6D_1D_2((k_1+k_2+p)^2-M)]$, where $M=m^2>0$.
    Define $D_1=k_1^2-M$, $D_2=k_2^2-M$, $D_3=(k_1+k_2)^2-M$;
    $I(a,b,c)$ integrates $D_1^{-a}D_2^{-b}D_3^{-c}$ and $p_2=p^2$.
    Taylor expansion through $p^2$ preserves their UV poles without introducing
    an infrared singularity. The tensor reducer supplies
    $\langle(q\cdot p)^2\rangle=q^2p^2/d$.

    Analytic integral values enter later as reference inputs. The IBP search
    computes reduction coefficients; its residuals are **not certified masters**.
    """)
    return


@app.cell
def _(E, Replacement, S, Symbol, hep, mo):
    d, k1, k2, mass_squared, p_squared, eps, coupling = S(
        "ibp_phi4::d",
        "ibp_phi4::k1",
        "ibp_phi4::k2",
        "ibp_phi4::M",
        "ibp_phi4::p2",
        "ibp_phi4::eps",
        "ibp_phi4::g",
    )
    tadpole_squared, vacuum_integral, log_mass, log_4pi = S(
        "ibp_phi4::T2",
        "ibp_phi4::V",
        "ibp_phi4::Lm",
        "ibp_phi4::L4pi",
    )
    integral = S("ibp_phi4::I")
    zero = E("0")
    pi = Symbol.PI

    # Taylor-expand the shifted line, then project its rank-two vacuum numerator.
    q, p, scaling, denominator = S(
        "ibp_phi4::q", "ibp_phi4::p", "ibp_phi4::scaling", "ibp_phi4::D3"
    )
    mink, dot = S("spenso::mink", "spenso::dot")
    qc, pc = q(mink(d)), p(mink(d))
    shifted_line = 1 / (
        denominator + 2 * scaling * dot(qc, pc) + scaling**2 * dot(pc, pc)
    )
    taylor = shifted_line.series(scaling, 0, 2).to_expression().replace(scaling, E("1"))
    taylor = hep.TensorReducer(d).with_integrated_vector(qc).reduce(taylor)
    taylor = (
        taylor.replace(dot(qc, qc), denominator + mass_squared)
        .replace(dot(pc, pc), p_squared)
        .expand()
    )
    expected_taylor = (
        1 / denominator
        + p_squared * (4 / d - 1) / denominator**2
        + 4 * mass_squared * p_squared / (d * denominator**3)
    )
    assert (taylor - expected_taylor).together() == zero
    bare_input = (
        integral(2, 1, 0) / 4
        + taylor.replace_multiple(
            [
                Replacement(denominator**-1, integral(1, 1, 1)),
                Replacement(denominator**-2, integral(1, 1, 2)),
                Replacement(denominator**-3, integral(1, 1, 3)),
            ]
        )
        / 6
    )

    mo.vstack([mo.md("**Taylor-expanded integral input**"), bare_input])
    return (
        bare_input,
        coupling,
        d,
        eps,
        integral,
        k1,
        k2,
        log_4pi,
        log_mass,
        mass_squared,
        p_squared,
        pi,
        tadpole_squared,
        vacuum_integral,
        zero,
    )


@app.cell
def _(d, hep, k1, k2, mass_squared, mo):
    kinematics = hep.Kinematics(d, momenta=[k1, k2])
    family = hep.IntegralFamily(
        [k1, k2],
        [],
        [kinematics.scalar_product(k, k) - mass_squared for k in (k1, k2, k1 + k2)],
        kinematics=kinematics,
    )
    assert family.is_complete and family.is_independent
    ibp = hep.IBPFamily(family, name="phi4_vacuum")
    identities = ibp.ibp_identities()
    assert len(identities) == 4
    symmetries = [
        family.mapping_to(family, images)
        for images in (
            [k1, k2],
            [k2, k1],
            [-k1 - k2, k2],
            [k1, -k1 - k2],
            [k2, -k1 - k2],
            [-k1 - k2, k1],
        )
    ]
    assert all(mapping is not None for mapping in symmetries)
    assert len({tuple(mapping.denominator_map) for mapping in symmetries}) == 6

    mo.vstack(
        [
            mo.md("**Equal-mass vacuum family**"),
            family,
            mo.md(
                "Six unit-Jacobian loop-momentum maps verify all denominator permutations."
            ),
        ]
    )
    return ibp, symmetries


@app.cell
def _(
    Replacement,
    bare_input,
    d,
    ibp,
    integral,
    mass_squared,
    mo,
    symmetries,
    tadpole_squared,
    vacuum_integral,
    zero,
):
    targets = [[2, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3]]
    laporta = ibp.reduce_laporta(targets, max_depth=2)
    assert laporta.stats["rows"] > 0
    # These are the solver's unresolved integrals at finite search depth. Their
    # subsequent identification uses independently verified loop-momentum maps.
    canonical = {
        tuple(powers): min(tuple(mapping.map_powers(powers)) for mapping in symmetries)
        for powers in laporta.residuals
    }
    reference_basis = {(0, 1, 1): tadpole_squared, (1, 1, 1): vacuum_integral}
    raw_reductions = {tuple(target): laporta.reduce(target) for target in targets}
    reduced = {}
    for target, terms in raw_reductions.items():
        assert all(tuple(powers) in canonical for powers, coefficient in terms)
        assert all(
            canonical[tuple(powers)] in reference_basis for powers, coefficient in terms
        )
        reduced[target] = sum(
            (
                coefficient * reference_basis[canonical[tuple(powers)]]
                for powers, coefficient in terms
            ),
            zero,
        ).together()
    assert (
        reduced[2, 1, 0] - (d - 2) * tadpole_squared / (2 * mass_squared)
    ).together() == zero
    assert (
        reduced[1, 1, 2] - (d - 3) * vacuum_integral / (3 * mass_squared)
    ).together() == zero
    assert (
        reduced[1, 1, 3]
        - (d - 8) * (d - 3) * vacuum_integral / (18 * mass_squared**2)
        - (d - 2) ** 2 * tadpole_squared / (12 * mass_squared**3)
    ).together() == zero
    bare_reduced = bare_input.replace_multiple(
        [Replacement(integral(*target), value) for target, value in reduced.items()]
    ).together()

    _raw = [
        mo.hstack(
            [
                integral(*_target),
                mo.md(r"$\longrightarrow$"),
                laporta.reduce(list(_target), integral=integral),
            ]
        )
        for _target in raw_reductions
    ]
    mo.vstack(
        [
            mo.md("**Computed Laporta reductions**"),
            mo.md(
                f"The finite search used {laporta.stats['rows']} equations from {laporta.stats['seeds']} seeds."
            ),
            mo.accordion({"Raw reductions before momentum mappings": mo.vstack(_raw)}),
            mo.md("**Unresolved integral basis at this search depth**"),
            mo.hstack([integral(*_powers) for _powers in laporta.residuals]),
            mo.md(
                r"The verified mappings identify the three disconnected routings as $T^2=I_{110}$; write $V=I_{111}$."
            ),
            *[
                mo.hstack([integral(*_target), mo.md(r"$\longrightarrow$"), _value])
                for _target, _value in reduced.items()
            ],
            mo.md("**Reduced self-energy integral combination**"),
            bare_reduced,
        ]
    )
    return (bare_reduced,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analytic reference inputs

    Use $d=4-2\epsilon$, $L_m=\log M$ and $L_{4\pi}=\log(4\pi)$.
    The [one-loop tadpole](https://raw.githubusercontent.com/FeynCalc/feyncalc/master/FeynCalc/Examples/MasterIntegrals/Tadpoles/tad1LxFx1x1xxEp999x.m)
    and [equal-mass two-loop vacuum](https://raw.githubusercontent.com/FeynCalc/feyncalc/master/FeynCalc/Examples/MasterIntegrals/Tadpoles/tad2LxFx111x111xxEp1x.m)
    give the needed coefficients:
    $$T=M[1/\epsilon+1-L_m+O(\epsilon)],\qquad
      V=M[3/(2\epsilon^2)+(9/2-3L_m)/\epsilon+O(1)].$$

    These analytic values are supplied, not evaluated by the reducer. Below,
    amplitudes are divided by $ig^2/(16\pi^2)^2$. Restoring the two-loop measure
    contributes $-(1+2\epsilon L_{4\pi})$ through the required order.
    """)
    return


@app.cell
def _(
    E,
    bare_reduced,
    d,
    eps,
    log_4pi,
    log_mass,
    mass_squared,
    mo,
    p_squared,
    tadpole_squared,
    vacuum_integral,
    zero,
):
    # Analytic master coefficients are reference inputs, not computed by IBP.
    tadpole_poles = mass_squared * (1 / eps + 1 - log_mass)
    tadpole_squared_poles = mass_squared**2 * (1 / eps**2 + 2 * (1 - log_mass) / eps)
    vacuum_poles = mass_squared * (E("3/2") / eps**2 + (E("9/2") - 3 * log_mass) / eps)
    bare_uv = (
        (
            -(1 + 2 * eps * log_4pi)
            * bare_reduced.replace(d, 4 - 2 * eps)
            .replace(tadpole_squared, tadpole_squared_poles)
            .replace(vacuum_integral, vacuum_poles)
        )
        .series(eps, 0, -1)
        .to_expression()
        .expand()
    )
    expected_bare = (
        -mass_squared / (2 * eps**2)
        + (mass_squared * (log_mass - 1 - log_4pi) + p_squared / 24) / eps
    )
    assert (bare_uv - expected_bare).together() == zero

    mo.vstack([mo.md("**Bare two-loop UV poles**"), bare_uv])
    return bare_uv, tadpole_poles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Counterterm recurrence and pole cancellation

    The one-loop counterterm uses $T(2)$. Parametric IBP derives
    $T(a+1)=(d-2a)T(a)/(2aM)$, with its applicability conditions retained.
    Insert the one-loop constants $z_g^{(1)}=3/(2\epsilon)$ and
    $z_m^{(1)}=1/(2\epsilon)$ in units of $g/(16\pi^2)$.

    With $a=g/(16\pi^2)$ the final reference result is
    $$Z_\phi=1-\frac{a^2}{24\epsilon},\qquad
      Z_m=1+\frac{a}{2\epsilon}+a^2\left[\frac1{2\epsilon^2}-\frac5{24\epsilon}\right].$$
    Here $Z_m$ renormalizes the mass squared. The assertions check both constants
    and cancellation of all momentum, mass and logarithmic poles.
    """)
    return


@app.cell
def _(
    E,
    bare_uv,
    coupling,
    d,
    eps,
    hep,
    k1,
    log_4pi,
    log_mass,
    mass_squared,
    mo,
    p_squared,
    pi,
    tadpole_poles,
    zero,
):
    tadpole_kin = hep.Kinematics(d, momenta=[k1])
    tadpole_family = hep.IntegralFamily(
        [k1],
        [],
        [tadpole_kin.scalar_product(k1, k1) - mass_squared],
        kinematics=tadpole_kin,
    )
    parametric = hep.IBPFamily(tadpole_family, name="phi4_tadpole").solve_parametric(
        [True], max_depth=1
    )
    assert parametric.rules
    tadpole_terms = parametric.reduce([2])
    assert len(tadpole_terms) == 1 and tadpole_terms[0][0] == [1]
    tadpole_coefficient = tadpole_terms[0][1]
    assert (tadpole_coefficient - (d - 2) / (2 * mass_squared)).together() == zero
    zg1, zm1 = 3 / (2 * eps), 1 / (2 * eps)
    counterterm_uv = (
        (
            (1 + eps * log_4pi)
            * (zg1 + mass_squared * zm1 * tadpole_coefficient.replace(d, 4 - 2 * eps))
            * tadpole_poles
            / 2
        )
        .series(eps, 0, -1)
        .to_expression()
        .expand()
    )
    expected_counterterm = (
        mass_squared / eps**2 + mass_squared * (E("3/4") - log_mass + log_4pi) / eps
    )
    assert (counterterm_uv - expected_counterterm).together() == zero
    loop_uv = (bare_uv + counterterm_uv).expand()
    assert (
        loop_uv
        - mass_squared / (2 * eps**2)
        + mass_squared / (4 * eps)
        - p_squared / (24 * eps)
    ).together() == zero
    zphi2 = -loop_uv.coefficient(p_squared).expand()
    zm2 = (loop_uv.replace(p_squared, zero) / mass_squared - zphi2).expand()
    assert (
        loop_uv + p_squared * zphi2 - mass_squared * (zm2 + zphi2)
    ).together() == zero
    loop_coupling = coupling / (16 * pi**2)
    zphi = (1 + loop_coupling**2 * zphi2).expand()
    zm = (1 + loop_coupling * zm1 + loop_coupling**2 * zm2).expand()
    assert (zphi - 1 + coupling**2 / (6144 * pi**4 * eps)).together() == zero
    assert (
        zm
        - 1
        - coupling / (32 * pi**2 * eps)
        - coupling**2 / (512 * pi**4 * eps**2)
        + 5 * coupling**2 / (6144 * pi**4 * eps)
    ).together() == zero
    mo.vstack(
        [
            mo.md("**Parametric reduction specialized to the doubled tadpole**"),
            tadpole_coefficient,
            mo.accordion(
                {
                    "Recurrence conditions": mo.vstack(
                        [
                            mo.md(
                                f"Rule {_i + 1}: nonzero conditions {_rule.nonzero_conditions}; exceptional index loci {_rule.exceptions}"
                            )
                            for _i, _rule in enumerate(parametric.rules)
                        ]
                    )
                }
            ),
            mo.md("**One-loop counterterm UV poles**"),
            counterterm_uv,
            mo.md("**Combined loop poles before the tree counterterm**"),
            loop_uv,
            mo.md(r"**Renormalization constants $Z_\phi$ and $Z_m$**"),
            zphi,
            zm,
            mo.callout(
                "The computed IBP reductions and supplied analytic integral values reproduce the reference UV poles and renormalization constants.",
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
