import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Chiral Z decays")


@app.cell
def _():
    import marimo as mo
    from symbolica import E, Expression, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep as fk
    from symbolica.community.spenso import CookSettings, TensorExpression

    _set_namespace("z_decay")
    return CookSettings, E, Expression, S, TensorExpression, fk, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chiral Z decays

    [Browse all notebooks](/) · [W and top decays](/?file=hep/weak_decays.py) · [Two-photon annihilation](/?file=hep/diphoton.py)

    Generate $Z\to f\bar f$ from the Standard Model and retain the fermion mass,
    left- and right-chiral couplings, and the Z's longitudinal polarization.
    Choose a neutrino, charged lepton, up-type quark or down-type quark to reproduce
    the four classes in the
    [FeynCalc example](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/Z-FFbar).

    The three initial Z polarizations are averaged; final spins and colors are
    summed. Fermion and antifermion are distinct, so no identical-particle factor
    is needed.
    """)
    return


@app.cell
def _(fk, mo):
    model = fk.Model.standard_model()
    channel = mo.ui.dropdown(
        ["Electron neutrinos", "Electrons", "Charm quarks", "Bottom quarks"],
        value="Electrons",
        label="Decay channel",
    )
    mo.vstack([channel])
    return channel, model


@app.cell
def _(CookSettings, E, S, TensorExpression, channel, fk, mo, model):
    P = S("gammalooprs::P")
    charge, sw, cw, mz = S("UFO::ee", "UFO::sw", "UFO::cw", "UFO::MZ")
    conjugate, wrapped = S("spenso::conj", "adjoint_index")
    ports = S("z", "fermion", "antifermion")
    wave, rep, index, left, right = S("wave_", "rep_", "index_", "left_", "right_")
    metric = S("spenso::g")
    particle_name = {
        "Electron neutrinos": "ve",
        "Electrons": "e-",
        "Charm quarks": "c",
        "Bottom quarks": "b",
    }[channel.value]
    fermion = model.particle(particle_name)
    mass = fermion.mass_expression
    weak_isospin = fermion.weak_isospin
    electric_charge = fermion.charge
    assert weak_isospin is not None
    antifermion = fermion.antiparticle
    generated = model.process(["Z"], [fermion, antifermion]).generate_diagrams(
        max_vertices=1, numerator_grouping=None, progress=None
    )
    assert len(generated.diagrams) == 1
    diagram = generated.diagrams[0]
    assert diagram.denominator_expression(dimension=4) == E("1")
    numerator = model.expand_couplings(diagram.numerator_expression())
    for edge in diagram.external_edges:
        matches = list(
            diagram.projector_expression().match(
                wave(edge.id, rep(4, index)), max_level=0
            )
        )
        assert len(matches) == 1
        # Both spin and color ports carry the external half-edge identifier.
        numerator = numerator.rename_indices(
            {dict(matches[0])[index]: ports[edge.external_index]}
        )
    operator = (
        numerator
        * diagram.overall_factor_expression(evaluate=True)
        * diagram.numerator_prefactor_expression()
    ).expand()
    adjoint = operator.dirac_adjoint().expand().simplify_gamma0()
    for real in (charge, sw, cw, mz, mass):
        adjoint = adjoint.replace(conjugate(real), real)
    adjoint = adjoint.wrap_indices(wrapped)

    color_projector = E("1")
    for position, particle in enumerate((fermion, antifermion), start=1):
        if particle.color == 1:
            continue
        matches = []
        for slot in operator.structure.slots:
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

    # The final fermions are summed, while the three initial Z polarizations
    # are averaged with the full Proca projector, including its longitudinal term.
    spin_projector = (
        fermion.spin_sum(P(1), wrapped(ports[2]), ports[1])
        * antifermion.spin_sum(P(2), ports[2], wrapped(ports[1]))
        * model.particle("Z").spin_sum(P(0), ports[0], wrapped(ports[0]), average=True)
    )
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
        .with_scalar_product(P(0), P(0), mz**2)
        .with_scalar_product(P(1), P(1), mass**2)
        .with_scalar_product(P(2), P(2), mass**2)
        .with_scalar_product(P(1), P(2), (mz**2 - 2 * mass**2) / 2)
        .with_scalar_product(P(0), P(1), mz**2 / 2)
        .with_scalar_product(P(0), P(2), mz**2 / 2)
    )
    squared = kin.apply(scalar)
    colors = abs(fermion.color)
    cv, ca = weak_isospin - 2 * electric_charge * sw**2, weak_isospin
    expected = (
        colors
        * charge**2
        / (3 * sw**2 * cw**2)
        * (cv**2 * (mz**2 + 2 * mass**2) + ca**2 * (mz**2 - 4 * mass**2))
    )
    residual = (
        ((squared - expected) * sw**2 * cw**2 / charge**2)
        .expand()
        .replace(cw, (1 - sw**2).sqrt())
        .together()
    )
    assert residual == E("0"), (particle_name, residual.format_plain())
    mo.vstack(
        [
            mo.md("**Generated decay diagram and chiral current**"),
            diagram,
            operator,
            operator.to_expression(),
            mo.md("**Spin-averaged squared matrix element**"),
            squared.together(),
        ]
    )
    return P, ca, charge, colors, cv, cw, mass, mz, squared, sw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Physical phase space

    With $r=m_f^2/m_Z^2$ and $\beta=\sqrt{1-4r}$, the decay width is
    $$\Gamma=\frac{N_c G_F m_Z^3\beta}{6\pi\sqrt2}
    \left[c_V^2(1+2r)+c_A^2(1-4r)\right],$$
    where $c_V=T_3-2Q_f\sin^2\theta_W$ and $c_A=T_3$.
    Here $N_c=1$ for leptons and $3$ for quarks. The expression applies above
    threshold, $m_Z>2m_f$.

    Feynkit supplies the two-body measure and rest-frame decay flux.
    Integrating the isotropic spin-averaged density over solid angle gives the
    total width. In the massless neutrino channel it reduces to
    $G_Fm_Z^3/(12\pi\sqrt2)$.
    """)
    return


@app.cell
def _(
    E,
    Expression,
    P,
    S,
    ca,
    charge,
    colors,
    cv,
    cw,
    fk,
    mass,
    mo,
    mz,
    squared,
    sw,
):
    pi = Expression.PI
    # Express the physical branch with positive M and beta. The on-shell
    # relation m^2=M^2(1-beta^2)/4 describes 0<beta<=1 above threshold.
    physical_mass = S("M", is_positive=True)
    beta = E("1") if mass == E("0") else S("beta", is_positive=True)
    mass_squared = physical_mass**2 * (1 - beta**2) / 4
    physical_kin = (
        fk.Kinematics()
        .with_scalar_product(P(0), P(0), physical_mass**2)
        .with_scalar_product(P(1), P(1), mass_squared)
        .with_scalar_product(P(2), P(2), mass_squared)
        .with_scalar_product(P(1), P(2), (physical_mass**2 - 2 * mass_squared) / 2)
    )
    measure = physical_kin.two_body_phase_space(P(1), P(2)).expand()
    # Symbolica keeps products under radicals intact; select the positive
    # root explicitly using M>0 and beta>0.
    measure = measure.replace(
        (physical_mass**4 * beta**2).sqrt(), physical_mass**2 * beta
    )
    flux = physical_kin.flux(P(0))
    assert (measure - beta / (32 * pi**2)).together() == E("0")
    assert flux == 2 * physical_mass
    physical_squared = squared.replace(mz, physical_mass)
    if mass != E("0"):
        physical_squared = physical_squared.replace(
            mass, physical_mass * (1 - beta**2).sqrt() / 2
        )
    # f and fbar are distinct final particles, so no factorial is present.
    width = 4 * pi * measure * physical_squared / flux
    gf = S("G_F")
    width = width.replace(
        charge**2, 4 * E("2").sqrt() * gf * physical_mass**2 * sw**2 * cw**2
    )
    expected_width = (
        colors
        * gf
        * physical_mass**3
        * beta
        * E("2").sqrt()
        / (12 * pi)
        * (
            cv**2 * (1 + 2 * mass_squared / physical_mass**2)
            + ca**2 * (1 - 4 * mass_squared / physical_mass**2)
        )
    )
    _residual = (
        (width - expected_width).expand().replace(cw, (1 - sw**2).sqrt()).together()
    )
    assert _residual == E("0"), _residual.format_plain()
    if mass != E("0"):
        massless_width = width.replace(beta, E("1"))
        expected_massless = (
            colors * gf * physical_mass**3 * (cv**2 + ca**2) * E("2").sqrt() / (12 * pi)
        )
        assert (massless_width - expected_massless).expand().replace(
            cw, (1 - sw**2).sqrt()
        ).together() == E("0")
    else:
        assert (
            width - gf * physical_mass**3 * E("2").sqrt() / (24 * pi)
        ).expand().replace(cw, (1 - sw**2).sqrt()).together() == E("0")

    z_decay_width = width.replace(cw, (1 - sw**2).sqrt()).together()
    mo.vstack(
        [
            mo.md(
                r"**Total decay width** in terms of $M=m_Z$ and the final-particle speed $\beta$. The massless neutrino channel has $\beta=1$."
            ),
            z_decay_width,
            mo.callout(
                "The generated squared amplitude, phase-space normalization and massless limit agree with the FeynCalc reference.",
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
