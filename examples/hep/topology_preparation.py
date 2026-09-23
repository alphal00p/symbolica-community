import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="B → ηc: two-loop topology preparation")


@app.cell
def _():
    import json
    from functools import reduce
    from pathlib import Path

    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community import hep

    _set_namespace("etac")

    return mo, json, reduce, Path, E, S, Replacement, hep


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

    The surviving sectors are grouped with verified loop-momentum maps. Every
    input integral is expressed in completed representative families, retaining
    its coefficients and propagator powers. The grouping keeps external momenta
    fixed and treats different propagator counts separately. Cross-size
    subtopology minimization remains to be integrated into this workflow.
    """)


@app.cell
def _(Path, json):
    fixture = json.loads(
        (Path(__file__).parent / "data/feyncalc_etac_topologies.json").read_text()
    )
    return (fixture,)


@app.cell
def _(E, S, Replacement, hep, fixture, reduce):
    def prepare_topologies():
        D, k1, k2, n, nb, gkin, meta, u0b = S(
            "D",
            "k1",
            "k2",
            "n",
            "nb",
            "gkin",
            "meta",
            "u0b",
            is_real=True,
        )
        kin = (
            hep.Kinematics(D, momenta=[k1, k2, n, nb])
            .with_scalar_product(n, n, E("0"))
            .with_scalar_product(nb, nb, E("0"))
            .with_scalar_product(n, nb, E("2"))
        )
        momenta = dict(zip(("k1", "k2", "n", "nb"), (k1, k2, n, nb)))
        labels = S(*fixture["labels"], is_real=True)
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
            "transverse certificate": 0,
        }
        completed_rows = []
        for family in sectors.values():
            parameters = [S(f"x{i}") for i in range(len(family.denominators))]
            U, F = family.symanzik(parameters)
            if U == E("0"):
                direction = family.scaleless_transverse_direction()
                assert direction is not None and any(w != E("0") for w in direction)
                assert all(w.is_real() is True for w in direction)
                # Independently shift each loop by w_i*r_perp. External products stay
                # fixed; z_i=r_perp.k_i and z2=r_perp^2 are independent formal symbols.
                loops = [k1, k2]
                z = S("etac_shift::z1", "etac_shift::z2")
                z_squared = S("etac_shift::squared")
                shifts = [
                    Replacement(
                        kin.scalar_product(ki, kj),
                        kin.scalar_product(ki, kj)
                        + direction[i] * z[j]
                        + direction[j] * z[i]
                        + direction[i] * direction[j] * z_squared,
                    )
                    for i, ki in enumerate(loops)
                    for j, kj in enumerate(loops[i:], i)
                ]
                for denominator in family.denominators:
                    assert (
                        denominator.replace_multiple(shifts) - denominator
                    ).together() == E("0")
                status = "transverse certificate"
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
            if status == "not detected":
                # All surviving families must pass automatic self-mapping before any
                # global minimization can rely on the shift search. Fractional
                # light-cone offsets previously broke candidate reconstruction.
                mapping = family.find_mapping(family)
                assert mapping is not None
                for source, target in enumerate(mapping.denominator_map):
                    assert (
                        mapping.apply(family.denominators[source])
                        - family.denominators[target]
                    ).together() == E("0")
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
            names = [S(f"d{i}") for i in range(len(completed.denominators))]
            rules = completed.scalar_product_rules(names)
            for scalar_product, expression in rules:
                for name, denominator in zip(names, completed.denominators):
                    expression = expression.replace(name, denominator)
                assert (expression - scalar_product).together() == E("0")
            completed_rows.append((family, completed, status))
        assert statistics == {
            "scaleless certificate": 112,
            "not detected": 434,
            "transverse certificate": 131,
        }

        # Group physical sectors before adding auxiliary denominators. Complete only
        # the retained representatives when constructing final integral coordinates.
        surviving_rows = [row for row in completed_rows if row[2] == "not detected"]
        surviving = [row[0] for row in surviving_rows]
        mappings = hep.IntegralFamily.find_mappings(surviving)
        representatives = sorted({target for target, _ in mappings})
        for source, (target, mapping) in enumerate(mappings):
            assert mappings[target][0] == target
            for i, j in enumerate(mapping.denominator_map):
                assert (
                    mapping.apply(surviving[source].denominators[i])
                    - surviving[target].denominators[j]
                ).together() == E("0")
            # Independently check the two-loop Jacobian in the returned coordinates.
            images = [image for _, image in mapping.momentum_rules]
            coefficients = [dict(image.coefficient_list(k1, k2)) for image in images]
            determinant = (
                coefficients[0].get(k1, E("0")) * coefficients[1].get(k2, E("0"))
                - coefficients[0].get(k2, E("0")) * coefficients[1].get(k1, E("0"))
            ).together()
            assert determinant in (E("1"), E("-1"))
            assert (
                mapping.apply(kin.scalar_product(k1, k2)) - kin.scalar_product(*images)
            ).together() == E("0")
        assert [
            target
            for target, _ in hep.IntegralFamily.find_mappings(
                [surviving[i] for i in representatives]
            )
        ] == list(range(len(representatives)))

        # Map every nonzero partial-fraction term from every original input to the
        # completed representative. Preserve coefficients and propagator powers.
        survivor_indices = {
            tuple(sorted(d.to_canonical_string() for d in family.denominators)): i
            for i, family in enumerate(surviving)
        }
        final_integrals = []
        for family, fractions in input_rows:
            terms = {}
            for coefficient, powers in fractions:
                assert all(power >= 0 for power in powers)
                sector = family.sector(powers)
                key = tuple(
                    sorted(d.to_canonical_string() for d in sector.denominators)
                )
                if key not in survivor_indices:
                    continue  # Removed only by one of the independently checked certificates.
                source = survivor_indices[key]
                alignment = sector.mapping_to(surviving[source], [k1, k2])
                assert alignment is not None
                source_powers = alignment.map_powers(
                    [power for power in powers if power > 0]
                )
                target, mapping = mappings[source]
                mapped_powers = mapping.map_powers(source_powers)
                completed = surviving_rows[target][1]
                mapped_powers += [0] * (
                    len(completed.denominators) - len(mapped_powers)
                )
                term = (target, tuple(mapped_powers))
                terms[term] = terms.get(term, E("0")) + coefficient
            final_integrals.append(
                {
                    term: coefficient.together()
                    for term, coefficient in terms.items()
                    if coefficient.together() != E("0")
                }
            )
        assert len(final_integrals) == 251
        assert len(representatives) == 223
        assert sum(len(terms) for terms in final_integrals) == 551
        assert sum(not terms for terms in final_integrals) == 59
        return (
            input_rows,
            completed_rows,
            statistics,
            pool,
            surviving_rows,
            mappings,
            representatives,
            final_integrals,
        )

    (
        input_rows,
        completed_rows,
        statistics,
        pool,
        surviving_rows,
        mappings,
        representatives,
        final_integrals,
    ) = prepare_topologies()
    return (
        input_rows,
        completed_rows,
        statistics,
        pool,
        surviving_rows,
        mappings,
        representatives,
        final_integrals,
    )


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
        There are **112 parametric scaling certificates** and **131 transverse
        certificates**. For the latter, an explicit loop shift leaves every
        denominator unchanged, isolating a polynomial transverse integral that
        vanishes in dimensional regularization. The other 434 sectors remain
        unclassified. A singular quadratic form alone does not prove zero.
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


@app.cell
def _(mo):
    input_index = mo.ui.slider(1, 251, value=1, step=1, label="Input integral")
    sector_index = mo.ui.slider(1, 677, value=1, step=1, label="Independent sector")
    mo.hstack([input_index, sector_index])
    return input_index, sector_index


@app.cell
def _(fixture, input_rows, input_index, final_integrals, topology_names, mo):
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
            mo.md("### In completed representative families"),
            mo.ui.table(
                [
                    {
                        "Topology": topology_names[target],
                        "Coefficient": str(coefficient),
                        "Powers": str(powers),
                    }
                    for (target, powers), coefficient in final_integrals[
                        input_index.value - 1
                    ].items()
                ],
                selection=None,
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


@app.cell
def _(original_sector, mo):
    transverse_direction = original_sector.scaleless_transverse_direction()
    if transverse_direction is not None:
        mo.output.append(
            mo.md(r"""
        ### Transverse certificate
        The entries below define $k_i \mapsto k_i+w_i r_\perp$, with
        $r_\perp\cdot n=r_\perp\cdot\bar n=0$. Every denominator is
        unchanged. The unconstrained transverse integral vanishes for any
        polynomial numerator in dimensional regularization.
        """)
        )
        mo.output.append(
            mo.ui.table(
                [
                    {"Loop": str(loop), "Weight": str(weight)}
                    for loop, weight in zip(
                        original_sector.loop_momenta, transverse_direction
                    )
                ],
                selection=None,
            )
        )
    return (transverse_direction,)


@app.cell
def _(representatives, mappings, final_integrals, mo):
    topology_names = {target: f"T{i + 1}" for i, target in enumerate(representatives)}
    mo.md(f"""
    ## Verified topology mappings

    **{len(mappings)} surviving families → {len(representatives)} representatives.**
    Every map preserves the loop measure and matches every denominator exactly.
    All representatives have complete seven-propagator bases; auxiliary powers
    remain zero. {sum(not terms for terms in final_integrals)} of the 251 input
    integrals vanish after certified scaleless terms are removed and equal
    mapped terms are combined.

    `IntegralFamily.find_mappings(families)` returns `(target_index, mapping)`
    for each input, with indices into the original list. Earlier families are
    preferred, and every map goes directly to a representative.
    """)
    return (topology_names,)


@app.cell
def _(mappings, mo):
    mapping_index = mo.ui.slider(
        1, len(mappings), value=1, step=1, label="Surviving family"
    )
    mo.output.append(mapping_index)
    return (mapping_index,)


@app.cell
def _(mapping_index, mappings, surviving_rows, topology_names, mo):
    mapped_target, selected_mapping = mappings[mapping_index.value - 1]
    mapped_source = surviving_rows[mapping_index.value - 1][0]
    mapped_family = surviving_rows[mapped_target][1]
    mo.vstack(
        [
            mo.md(
                f"### Family {mapping_index.value} → {topology_names[mapped_target]}"
            ),
            mo.ui.table(
                [
                    {"Source momentum": str(momentum), "Target coordinates": str(image)}
                    for momentum, image in selected_mapping.momentum_rules
                ],
                selection=None,
            ),
            mo.md(
                f"**Propagator map (zero-based):** {selected_mapping.denominator_map}"
            ),
            mapped_source,
            mapped_family,
        ]
    )
    return mapped_target, selected_mapping, mapped_source, mapped_family


if __name__ == "__main__":
    app.run()
