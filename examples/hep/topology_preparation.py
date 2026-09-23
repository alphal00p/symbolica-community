import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="B → ηc: two-loop topology preparation")


@app.cell
def _():
    import json
    from functools import reduce
    from pathlib import Path

    import marimo as mo
    from symbolica import E, S
    from symbolica.community import hep

    return mo, json, reduce, Path, E, S, hep


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # B → ηc: two-loop topology preparation

    [Browse notebooks](/) · [Integral families](/?file=hep/integral_families.py) ·
    [Two-loop IBP](/?file=hep/ibp_two_loop_masses.py)

    Run the full input from FeynCalc's
    [topology-minimization example](https://feyncalc.github.io/FeynCalcExamples/TopologyIdentification/TwoLoops/B-EtaC):
    **251 integrals in 248 distinct families**, built from 89 propagators.
    The denominators mix quadratic and eikonal terms, with light-cone vectors
    $n^2=\bar n^2=0$, $n\cdot\bar n=2$.

    Shared `IntegralFamily` operations decompose dependent denominators,
    check the resulting independent sectors, and complete each basis with
    propagators drawn from the full input. Every partial-fraction identity and
    every reconstructed scalar product is checked exactly.

    **Remaining work:** global topology minimization and classification of
    singular quadratic forms. Those sectors stay visible. This notebook does
    not calculate a decay amplitude or perform an IBP reduction.
    """)
    return


@app.cell
def _(Path, json):
    fixture = json.loads(
        (Path(__file__).parent / "data/feyncalc_etac_topologies.json").read_text()
    )
    return (fixture,)


@app.cell
def _(E, S, hep, fixture, reduce):
    def prepare_topologies():
        D, k1, k2, n, nb, gkin, meta, u0b = S(
            "etac::D",
            "etac::k1",
            "etac::k2",
            "etac::n",
            "etac::nb",
            "etac::gkin",
            "etac::meta",
            "etac::u0b",
            is_real=True,
        )
        kin = (
            hep.Kinematics(D, momenta=[k1, k2, n, nb])
            .with_scalar_product(n, n, E("0"))
            .with_scalar_product(nb, nb, E("0"))
            .with_scalar_product(n, nb, E("2"))
        )
        momenta = dict(zip(("k1", "k2", "n", "nb"), (k1, k2, n, nb)))
        labels = S(*["etac::" + label for label in fixture["labels"]], is_real=True)
        products = [
            kin.scalar_product(momenta[a], momenta[b])
            for a, b in fixture["scalar_products"]
        ]
        pool = []
        for text in fixture["denominators"]:
            denominator = E(text)
            for label, product in zip(labels, products):
                denominator = denominator.replace(label, product)
            pool.append(denominator.expand())
        assert len(pool) == len(set(pool)) == 89
        assert len(fixture["integrals"]) == 251
        assert (
            len({tuple(sorted(row["propagators"])) for row in fixture["integrals"]})
            == 248
        )
        # Two direct source checks fix the SFAD mass-term sign and the mixed linear form.
        assert pool[0] == kin.scalar_product(k1, k1)
        assert (
            pool[2]
            - kin.scalar_product(k1 + k2, k1 + k2)
            - 2 * gkin * meta * u0b * kin.scalar_product(k1 + k2, n)
            + meta * u0b * kin.scalar_product(k1 + k2, nb)
            + 2 * gkin * meta**2 * u0b**2
        ).expand() == E("0")

        sectors = {}
        input_rows = []
        for number, row in enumerate(fixture["integrals"], 1):
            denominators = [pool[i] for i in row["propagators"]]
            family = hep.IntegralFamily([k1, k2], [n, nb], denominators, kinematics=kin)
            fractions = family.partial_fraction(row["powers"])
            original = reduce(
                lambda a, b: a * b,
                (d ** (-p) for d, p in zip(denominators, row["powers"])),
                E("1"),
            )
            reconstructed = E("0")
            for coefficient, powers in fractions:
                reconstructed += coefficient * reduce(
                    lambda a, b: a * b,
                    (d ** (-p) for d, p in zip(denominators, powers)),
                    E("1"),
                )
                sector = family.sector(powers)
                assert sector.is_independent
                key = tuple(
                    sorted(d.to_canonical_string() for d in sector.denominators)
                )
                sectors.setdefault(key, sector)
            assert (original - reconstructed).together() == E("0"), number
            input_rows.append((family, fractions))
        assert len(sectors) == 677

        statistics = {
            "scaleless certificate": 0,
            "not detected": 0,
            "singular quadratic form": 0,
        }
        completed_rows = []
        for family in sectors.values():
            parameters = [S(f"etac::x{i}") for i in range(len(family.denominators))]
            U, F = family.symanzik(parameters)
            if U == E("0"):
                # Degenerate U/F do not prove scalelessness; never discard these sectors.
                status = "singular quadratic form"
            else:
                weights = family.scaleless_scaling(parameters)
                status = "not detected" if weights is None else "scaleless certificate"
                if weights is not None:
                    G = U + F
                    assert (
                        sum(
                            (
                                w * x * G.derivative(x)
                                for w, x in zip(weights, parameters)
                            ),
                            E("0"),
                        )
                        - G
                    ).expand() == E("0")
            statistics[status] += 1
            completed = family.complete(candidates=pool)
            assert completed.is_complete and completed.is_independent
            assert (
                completed.denominators[: len(family.denominators)]
                == family.denominators
            )
            assert all(d in pool for d in completed.denominators)
            assert (
                completed.complete(candidates=pool).denominators
                == completed.denominators
            )
            # Reconstruct every independent loop scalar product from the selected basis.
            names = [S(f"etac::d{i}") for i in range(len(completed.denominators))]
            rules = completed.scalar_product_rules(names)
            for scalar_product, expression in rules:
                for name, denominator in zip(names, completed.denominators):
                    expression = expression.replace(name, denominator)
                assert (expression - scalar_product).together() == E("0")
            completed_rows.append((family, completed, status))
        assert statistics == {
            "scaleless certificate": 112,
            "not detected": 434,
            "singular quadratic form": 131,
        }
        return input_rows, completed_rows, statistics, pool

    input_rows, completed_rows, statistics, pool = prepare_topologies()
    return input_rows, completed_rows, statistics, pool


@app.cell(hide_code=True)
def _(fixture, mo, statistics):
    mo.vstack(
        [
            mo.md("## Verified preparation"),
            mo.ui.table(
                [
                    {"Classification": name, "Independent sectors": count}
                    for name, count in statistics.items()
                ],
                selection=None,
            ),
            mo.md(r"""
        This partial-fraction ordering produces **677 distinct sectors**.
        All have independent denominators, and all can be completed to the
        seven-dimensional loop scalar-product basis using the supplied pool.
        The 112 scaleless classifications have explicit scaling certificates.
        A singular quadratic form is not a certificate of scalelessness.
        Different decomposition orders can produce different intermediate counts.

        Use `family.complete(candidates=pool)` to prefer existing propagators;
        `family.complete()` uses bare scalar products. Original propagators
        retain their positions, dependent candidates are skipped, and bare scalar
        products fill any missing directions. Completion does not add physical
        propagators to an integral: auxiliary powers initially remain zero.
        """),
            mo.md(
                f"[Pinned mathematical input]({fixture['source']}) · Source SHA-256: `{fixture['source_sha256']}`"
            ),
            mo.md(
                "The fixture omits the common +iη prescription for these algebraic checks. No contour equivalence is inferred."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    input_index = mo.ui.slider(1, 251, value=1, step=1, label="Input integral")
    sector_index = mo.ui.slider(1, 677, value=1, step=1, label="Independent sector")
    mo.hstack([input_index, sector_index])
    return input_index, sector_index


@app.cell
def _(fixture, input_rows, input_index, mo):
    selected_family, selected_fractions = input_rows[input_index.value - 1]
    _input = fixture["integrals"][input_index.value - 1]
    mo.vstack(
        [
            mo.md("## Input and partial fractions"),
            selected_family,
            mo.md(f"**Original powers:** {_input['powers']}"),
            mo.ui.table(
                [
                    {
                        "Coefficient": str(coefficient),
                        "Powers in original order": str(powers),
                    }
                    for coefficient, powers in selected_fractions
                ],
                selection=None,
            ),
            mo.md(
                "The exact rational difference between the input and this sum is zero."
            ),
        ]
    )
    return selected_family, selected_fractions


@app.cell
def _(completed_rows, sector_index, mo):
    original_sector, completed_sector, sector_status = completed_rows[
        sector_index.value - 1
    ]
    added_count = len(completed_sector.denominators) - len(original_sector.denominators)
    mo.vstack(
        [
            mo.md("## Complete the selected sector"),
            mo.md(
                f"**Classification:** {sector_status}. **Auxiliary propagators added:** {added_count}."
            ),
            original_sector,
            completed_sector,
            mo.md(
                "All seven loop scalar products reconstruct exactly from the completed basis. Every auxiliary entry comes from the 89-propagator pool."
            ),
        ]
    )
    return original_sector, completed_sector, sector_status, added_count


if __name__ == "__main__":
    app.run()
