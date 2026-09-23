import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Massive polarized spin states")


@app.cell
def _():
    import math

    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep as fk
    from symbolica.community.spenso import GammaSimplifySettings, TensorExpression

    _set_namespace("polarized_spin")

    return E, GammaSimplifySettings, S, TensorExpression, fk, math, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massive polarized spin states

    [Browse all notebooks](/) · [QED cross section](/?file=hep/qed_cross_section.py) ·
    [Polarization sums in D dimensions](/?file=hep/polarization_sums.py)

    `Particle.spin_sum(p, i, j, spin_vector=s)` selects one physical spin state
    of a massive Dirac particle. Omitting `spin_vector` sums both states.
    The selected state's dimensionless spin vector obeys
    $p\cdot s=0$ and $s^2=-1$ in the $(+,-,-,-)$ metric.

    The shared density matrix is
    $\rho_s=(\not p\pm m)(1+\gamma^5\not s)/2$.
    The mass sign distinguishes particle and antiparticle; the spin-projector
    sign is the same for both. Do not reverse $s$ merely because the particle
    is an antiparticle. Selecting a state already supplies the projector's
    factor $1/2$, so `average=True` is rejected with `spin_vector`.

    These are massive spin states with a chosen rest-frame direction; no
    helicity convention is assumed. This example uses the tau, whose mass
    is nonzero in the stored model. `Particle.sum_spins(..., spin_vector=s)`
    uses the same density when replacing paired external wavefunctions.
    """)
    return


@app.cell
def _(fk, mo):
    spin_model = fk.Model.standard_model()
    species = mo.ui.dropdown(["Tau", "Antitau"], value="Tau", label="Physical particle")
    mo.vstack([species])
    return species, spin_model


@app.cell
def _(E, GammaSimplifySettings, S, TensorExpression, fk, mo, species, spin_model):
    _particle = spin_model.particle("ta-" if species.value == "Tau" else "ta+")
    _sign = -1 if _particle.is_antiparticle else 1
    spin_momentum, spin_vector, spin_mass = S("p", "s", "UFO::MTA")
    _i, _j, _k, _slot = S(
        "i",
        "j",
        "k",
        "slot_",
    )
    _settings = GammaSimplifySettings.canonical()
    density = _particle.spin_sum(spin_momentum, _i, _j, spin_vector=spin_vector)
    _ordinary = _particle.spin_sum(spin_momentum, _i, _j)
    _opposite = density.replace(spin_vector(_slot), -spin_vector(_slot))
    assert TensorExpression((density + _opposite - _ordinary).expand()).simplify_gamma(
        _settings
    ).expand().to_expression() == E("0")
    density_trace = (
        TensorExpression(density.replace(_j, _i))
        .simplify_gamma(_settings)
        .expand()
        .to_expression()
    )
    assert density_trace == _sign * 2 * spin_mass
    _kinematics = (
        fk.Kinematics()
        .with_scalar_product(spin_momentum, spin_momentum, spin_mass**2)
        .with_scalar_product(spin_momentum, spin_vector, E("0"))
        .with_scalar_product(spin_vector, spin_vector, E("-1"))
    )
    _squared = _particle.spin_sum(
        spin_momentum, _i, _k, spin_vector=spin_vector
    ) * _particle.spin_sum(spin_momentum, _k, _j, spin_vector=spin_vector)
    _purity_residual = (
        TensorExpression((_squared - _sign * 2 * spin_mass * density).expand())
        .simplify_gamma(_settings)
        .expand()
        .to_dots()
        .to_expression()
    )
    assert _kinematics.apply(_purity_residual).expand() == E("0")
    mo.vstack(
        [
            mo.md("**Selected spin density**"),
            TensorExpression(density).simplify_gamma(_settings),
            mo.md("**Spinor trace**"),
            density_trace,
            mo.md(
                r"Verified: $\rho_s+\rho_{-s}=\not p\pm m$, "
                r"$\operatorname{tr}\rho_s=\pm2m$, and "
                r"$\rho_s^2=(\pm2m)\rho_s$. The last identity expresses "
                r"the rank-one character of a pure spin state."
            ),
        ]
    )
    return density, density_trace, spin_mass, spin_momentum, spin_vector


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boost a rest-frame spin direction

    Start with $p/m=(1,0,0,0)$ and $s=(0,\sin\theta,0,\cos\theta)$.
    The same `Boost` transforms both four-vectors. The controls show their
    components after a boost along $z$ and check the invariant constraints.
    Momenta below are measured in units of the particle mass.
    """)
    return


@app.cell
def _(mo):
    spin_speed = mo.ui.slider(
        start=-0.9, stop=0.9, step=0.1, value=0.5, label="Boost velocity along z"
    )
    spin_angle = mo.ui.slider(
        start=0, stop=180, step=5, value=45, label="Rest-frame spin angle (degrees)"
    )
    mo.vstack([spin_speed, spin_angle])
    return spin_angle, spin_speed


@app.cell
def _(fk, math, mo, spin_angle, spin_speed):
    _angle = math.radians(spin_angle.value)
    _boost = fk.Boost(fk.ThreeMomentum(0.0, 0.0, spin_speed.value))
    boosted_momentum = _boost.apply(fk.FourMomentum(1.0, 0.0, 0.0, 0.0))
    boosted_spin = _boost.apply(
        fk.FourMomentum(0.0, math.sin(_angle), 0.0, math.cos(_angle))
    )
    spin_constraints = (
        boosted_momentum.dot(boosted_momentum),
        boosted_momentum.dot(boosted_spin),
        boosted_spin.dot(boosted_spin),
    )
    assert all(
        abs(_actual - _expected) < 1e-12
        for _actual, _expected in zip(spin_constraints, (1.0, 0.0, -1.0), strict=True)
    )
    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "Vector": _label,
                        **dict(
                            zip(
                                ("time", "x", "y", "z"),
                                _vector.components(),
                                strict=True,
                            )
                        ),
                    }
                    for _label, _vector in (
                        ("p / m", boosted_momentum),
                        ("s", boosted_spin),
                    )
                ],
                selection=None,
            ),
            mo.md(r"The checks pass: $(p/m)^2=1$, $(p/m)\cdot s=0$, $s^2=-1$."),
        ]
    )
    return boosted_momentum, boosted_spin, spin_constraints


if __name__ == "__main__":
    app.run()
