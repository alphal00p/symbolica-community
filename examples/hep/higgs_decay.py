import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive Higgs decays")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, Expression, S
    from symbolica.community import hep as fk
    from symbolica.community.spenso import CookSettings, TensorExpression

    return CookSettings, E, Expression, S, TensorExpression, fk, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massive Higgs decays

    [Browse all notebooks](/) · [Chiral Z decays](/?file=hep/z_decay.py)

    Generate $H\to f\bar f$, $H\to W^-W^+$ or $H\to ZZ$ from the Standard
    Model, retaining the final-particle masses. The calculation sums final spins
    and colors, including all three physical polarizations of each massive vector.
    A scalar Higgs needs no initial spin average.

    These are **on-shell two-body decays above threshold**, $m_H>2m_f$ or
    $m_H>2m_V$. The $WW$ and $ZZ$ results describe a Higgs above those thresholds;
    they do not describe the off-shell $WW^*$ and $ZZ^*$ decays at 125 GeV.

    The reference results are FeynCalc's
    [fermion](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/H-FFbar),
    [WW](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/H-WW), and
    [ZZ](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/H-ZZ) examples.
    """)


@app.cell
def _(fk, mo):
    _location = mo.notebook_location()
    if _location is None:
        raise RuntimeError("Cannot locate hep_sm.json beside the notebook directory")
    model = fk.Model(_location.parent / "hep_sm.json")
    channel = mo.ui.dropdown(
        ["Electrons", "Charm quarks", "Bottom quarks", "W bosons", "Z bosons"],
        value="Bottom quarks",
        label="Decay channel",
    )
    mo.vstack([channel])
    return channel, model


@app.cell
def _(CookSettings, E, S, TensorExpression, channel, fk, mo, model):
    P = S("gammalooprs::P")
    charge, sw, cw, mh, mw, mz, vev = S(
        "UFO::ee", "UFO::sw", "UFO::cw", "UFO::MH", "UFO::MW", "UFO::MZ", "UFO::vev"
    )
    conjugate, wrapped = S("spenso::conj", "higgs_decay::adjoint_index")
    ports = S("higgs_decay::h", "higgs_decay::first", "higgs_decay::second")
    wave, rep, index, left, right = S("wave_", "rep_", "index_", "left_", "right_")
    metric = S("spenso::g")
    pdgs, mass, yukawa, yukawa_mass = {
        "Electrons": ((11, -11), S("UFO::Me"), "ye", "yme"),
        "Charm quarks": ((4, -4), S("UFO::MC"), "yc", "ymc"),
        "Bottom quarks": ((5, -5), S("UFO::MB"), "yb", "ymb"),
        "W bosons": ((-24, 24), mw, None, None),
        "Z bosons": ((23, 23), mz, None, None),
    }[channel.value]
    generated = fk.Generator(model).generate(
        fk.Process.amplitude([25], list(pdgs)),
        max_vertices=1,
        numerator_grouping=None,
        progress=None,
    )
    assert len(generated.diagrams) == 1
    diagram = generated.diagrams[0]
    assert diagram.denominator_expression(dimension=4).to_expression() == E("1")
    assert diagram.overall_factor_expression(evaluate=True) == E("1")
    numerator = model.expand_couplings(diagram.numerator_expression().to_expression())
    if yukawa is not None:
        numerator = numerator.replace(
            S(f"UFO::{yukawa}"), model.parameter(yukawa).expression
        )
        # UFO Yukawa masses and pole masses are distinct inputs. The gallery
        # comparison imposes their tree-level equality explicitly.
        numerator = numerator.replace(S(f"UFO::{yukawa_mass}"), mass)
    numerator = numerator.replace(vev, model.parameter("vev").expression)
    for edge in diagram.external_edges:
        if edge.external_index == 0:
            continue  # A scalar Higgs has no open spin or color index.
        matches = list(
            diagram.projector_expression().match(
                wave(edge.id, rep(4, index)), max_level=0
            )
        )
        assert len(matches) == 1
        numerator = numerator.replace(
            dict(matches[0])[index], ports[edge.external_index]
        )
    operator = TensorExpression(
        (
            numerator
            * diagram.overall_factor_expression(evaluate=True)
            * diagram.numerator_prefactor_expression()
        ).expand()
    )
    adjoint = operator.dirac_adjoint().expand().simplify_gamma0().to_expression()
    for real in (charge, sw, cw, mh, mw, mz, mass):
        adjoint = adjoint.replace(conjugate(real), real)
    adjoint = TensorExpression(adjoint).wrap_indices(wrapped)

    color_projector = E("1")
    for position, particle_pdg in enumerate(pdgs, start=1):
        particle = model.particle_by_pdg(particle_pdg)
        if particle.color == 1:
            continue
        matches = []
        for slot in operator.interface:
            original = slot.to_expression()
            if original.replace(ports[position], E("0")) == original:
                continue
            closure = metric(
                slot.dual().to_expression(),
                original.replace(ports[position], wrapped(ports[position])),
            )
            match = next(
                closure.match(particle.color_sum(left, right), max_level=0), None
            )
            if match is not None:
                matches.append(dict(match))
        assert len(matches) == 1
        color_projector *= particle.color_sum(matches[0][left], matches[0][right])

    if yukawa is not None:
        # Dirac adjunction exchanges the open fermion-chain endpoints.
        spin_projector = model.particle_by_pdg(pdgs[0]).spin_sum(
            P(1), wrapped(ports[2]), ports[1]
        ) * model.particle_by_pdg(pdgs[1]).spin_sum(P(2), ports[2], wrapped(ports[1]))
    else:
        # Sum all three physical polarizations of each final vector, including
        # its longitudinal Proca term. There is no initial spin average.
        spin_projector = model.particle_by_pdg(pdgs[0]).spin_sum(
            P(1), ports[1], wrapped(ports[1])
        ) * model.particle_by_pdg(pdgs[1]).spin_sum(P(2), ports[2], wrapped(ports[2]))
    scalar = (
        TensorExpression(
            operator.to_expression() * adjoint * spin_projector * color_projector,
            cook_indices=CookSettings.indices(),
        )
        .expand()
        .simplify_gamma()
        .expand()
        .simplify_gamma()
        .simplify_color()
        .simplify_metrics()
        .to_dots()
    )
    assert scalar.is_scalar
    kin = (
        fk.Kinematics()
        .with_scalar_product(P(0), P(0), mh**2)
        .with_scalar_product(P(1), P(1), mass**2)
        .with_scalar_product(P(2), P(2), mass**2)
        .with_scalar_product(P(1), P(2), (mh**2 - 2 * mass**2) / 2)
    )
    squared = kin.apply(scalar.to_expression())
    colors = abs(model.particle_by_pdg(pdgs[0]).color)
    if yukawa is not None:
        expected = (
            colors * charge**2 * mass**2 * (mh**2 - 4 * mass**2) / (2 * mw**2 * sw**2)
        )
        assert squared.replace(mass, E("0")) == E("0")
    else:
        expected = (
            charge**2
            * (mh**4 - 4 * mh**2 * mass**2 + 12 * mass**4)
            / (4 * mw**2 * sw**2)
        )
        if pdgs == (23, 23):
            # This is the labeled amplitude: the identical-Z factor belongs to
            # phase space below, not to the amplitude or polarization sum.
            expected *= mw**4 / (mz**4 * cw**4)
    residual = (squared - expected).expand().replace(cw, (1 - sw**2).sqrt()).together()
    assert residual == E("0"), (pdgs, residual.format_plain())
    mo.vstack(
        [
            mo.md("**Generated decay diagram and open amplitude**"),
            diagram,
            operator,
            mo.md("**Squared matrix element, summed over final spins and colors**"),
            squared.replace(cw, (1 - sw**2).sqrt()).together(),
        ]
    )
    return P, charge, colors, cw, mass, mh, mw, mz, pdgs, squared, sw, yukawa


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decay width

    Write $M=m_H$, $r=m^2/M^2$ and $\beta=\sqrt{1-4r}$. The shared two-body
    phase-space measure is $d\Phi_2/d\Omega=\beta/(32\pi^2)$ and the rest-frame
    decay flux is $2M$. Integrating over solid angle gives
    $\Gamma=S\beta\sum|\mathcal M|^2/(16\pi M)$.

    The identical-particle factor is **$S=1/2!$ for $ZZ$** and $S=1$ for the
    distinct $W^-W^+$ and $f\bar f$ final states. It is applied once, to the
    phase-space integral. For the $ZZ$ comparison we use $m_W=m_Z\cos\theta_W$;
    for fermions we identify the Yukawa mass with the kinematic mass at tree level.

    The gallery widths, with $\alpha=e^2/(4\pi)$, are
    $$\Gamma_{f\bar f}=\frac{N_c\alpha M m_f^2}{8m_W^2\sin^2\theta_W}\beta^3,$$
    $$\Gamma_{WW}=\frac{\alpha M^3}{16m_W^2\sin^2\theta_W}
      \beta(1-4r+12r^2),\qquad
      \Gamma_{ZZ}=\frac{\alpha M^3}{32m_W^2\sin^2\theta_W}
      \beta(1-4r+12r^2).$$
    Here $N_c=1$ for leptons and $3$ for quarks; use the selected final-particle
    mass in $r$ and $\beta$.
    """)


@app.cell
def _(
    E,
    Expression,
    P,
    S,
    charge,
    colors,
    cw,
    fk,
    mass,
    mh,
    mo,
    mw,
    mz,
    pdgs,
    squared,
    sw,
    yukawa,
):
    pi = Expression.PI
    alpha = S("higgs_decay::alpha")
    physical_mass = S("higgs_decay::M", is_positive=True)
    beta = S("higgs_decay::beta", is_positive=True)
    mass_squared = physical_mass**2 * (1 - beta**2) / 4

    # M>0 and 0<beta<1 select the physical, above-threshold two-body branch.
    physical_kin = (
        fk.Kinematics()
        .with_scalar_product(P(0), P(0), physical_mass**2)
        .with_scalar_product(P(1), P(1), mass_squared)
        .with_scalar_product(P(2), P(2), mass_squared)
        .with_scalar_product(P(1), P(2), (physical_mass**2 - 2 * mass_squared) / 2)
    )
    measure = physical_kin.two_body_phase_space(P(1), P(2)).expand()
    # Symbolica retains products under radicals; choose the positive root explicitly.
    measure = measure.replace(
        (physical_mass**4 * beta**2).sqrt(), physical_mass**2 * beta
    )
    flux = physical_kin.flux(P(0))
    assert (measure - beta / (32 * pi**2)).together() == E("0")
    assert flux == 2 * physical_mass
    # Apply the explicit 1/2! only for the identical ZZ final state.
    symmetry = E("1/2") if pdgs == (23, 23) else E("1")
    width = symmetry * 4 * pi * measure * squared / flux
    width = width.replace(charge**2, 4 * pi * alpha)
    if yukawa is not None:
        expected_width = (
            colors * alpha * physical_mass * mass**2 * beta**3 / (8 * mw**2 * sw**2)
        )
    else:
        expected_width = (
            alpha
            * physical_mass**3
            * beta
            / ((32 if pdgs == (23, 23) else 16) * mw**2 * sw**2)
        )
        expected_width *= (
            1
            - 4 * mass_squared / physical_mass**2
            + 12 * mass_squared**2 / physical_mass**4
        )
    if pdgs == (23, 23):
        # The gallery width uses the tree-level electroweak mass relation.
        width = width.replace(mw, cw * mz)
        expected_width = expected_width.replace(mw, cw * mz)
    width = width.replace(mh, physical_mass).replace(
        mass, physical_mass * (1 - beta**2).sqrt() / 2
    )
    expected_width = expected_width.replace(
        mass, physical_mass * (1 - beta**2).sqrt() / 2
    )
    _residual = (
        (width - expected_width).expand().replace(cw, (1 - sw**2).sqrt()).together()
    )
    assert _residual == E("0"), (pdgs, _residual.format_plain())
    if yukawa is not None:
        assert width.replace(beta, E("1")).together() == E("0")
    higgs_decay_width = width.replace(cw, (1 - sw**2).sqrt()).together()
    mo.vstack(
        [
            mo.md(
                r"**Total decay width** in terms of $M=m_H$ and $\beta$, with $m^2=M^2(1-\beta^2)/4$."
            ),
            higgs_decay_width,
            mo.callout(
                "The generated massive squared amplitude and integrated decay width agree with the FeynCalc reference.",
                kind="success",
            ),
        ]
    )
    return (higgs_decay_width,)


if __name__ == "__main__":
    app.run()
