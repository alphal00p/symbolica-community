use std::{collections::HashMap, sync::LazyLock};

use spenso::{
    structure::{
        abstract_index::AIND_SYMBOLS,
        representation::{LibraryRep, RepName},
    },
    tensor_library::ETS,
};
use symbolica::{
    atom::{Atom, AtomCore, AtomType, AtomView, Symbol},
    function,
    id::{Condition, MatchSettings, PatternRestriction, Replacement, WildcardRestriction},
    symbol,
};

use super::rep_symbols::RS;

pub struct MetricSymbols {
    pub dim: Symbol,
    pub dot: Symbol,
    pub dummy: Symbol,
}

pub static MS: LazyLock<MetricSymbols> = LazyLock::new(|| MetricSymbols {
    dim: symbol!("dim"),
    dot: symbol!("dot"),
    dummy: symbol!("custom::dummy"),
});

pub fn wrap_indices_impl(view: AtomView, header: Symbol) -> Atom {
    let mut expr = view.expand();
    let dim = RS.d_;
    let dima = Atom::new_var(dim);

    let mut reps = vec![];
    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        reps.push(
            Replacement::new(
                function!(RS.g_, RS.a___, i.to_symbolic([dim, RS.a_]), RS.b___).to_pattern(),
                function!(
                    RS.g_,
                    RS.a___,
                    i.to_symbolic([dima.clone(), function!(header, Atom::new_var(RS.a_))]),
                    RS.b___
                ),
            )
            .with_conditions(num_or_var(RS.a_)),
        );
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();
        reps.push(
            Replacement::new(
                function!(RS.g_, RS.a___, i.to_symbolic([dim, RS.a_]), RS.b___).to_pattern(),
                function!(
                    RS.g_,
                    RS.a___,
                    i.to_symbolic([dima.clone(), function!(header, Atom::new_var(RS.a_))]),
                    RS.b___
                ),
            )
            .with_conditions(num_or_var(RS.a_)),
        );
        reps.push(
            Replacement::new(
                function!(RS.g_, RS.a___, di.to_symbolic([dim, RS.a_]), RS.b___).to_pattern(),
                function!(
                    RS.g_,
                    RS.a___,
                    di.to_symbolic([dima.clone(), function!(header, Atom::new_var(RS.a_))]),
                    RS.b___
                ),
            )
            .with_conditions(num_or_var(RS.a_)),
        );
    }
    let mut atom = Atom::new();
    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
    }
    expr
}

pub fn list_dangling_impl(view: AtomView) -> Vec<Atom> {
    let a = view.expand();
    let settings = MatchSettings {
        level_range: (0, Some(1)),
        ..Default::default()
    };
    let mut dangling = HashMap::new();
    if let AtomView::Add(a) = a.as_view() {
        if let Some(first_term) = a.iter().next() {
            // println!("First term: {}", first_term);
            for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
                let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
                for p in first_term.pattern_match(&ipat, None, &settings) {
                    *dangling.entry(ipat.replace_wildcards(&p)).or_insert(0) += 1;
                }
            }
            for i in LibraryRep::all_dualizables() {
                let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
                let ipat_dual = i.dual().to_symbolic([RS.d_, RS.a_]).to_pattern();

                for p in first_term.pattern_match(&ipat, None, &settings) {
                    *dangling.entry(ipat.replace_wildcards(&p)).or_insert(0) += 1;
                }
                for p in first_term.pattern_match(&ipat_dual, None, &settings) {
                    *dangling.entry(ipat.replace_wildcards(&p)).or_insert(0) -= 1;
                }
            }
        }
    }

    dangling
        .into_iter()
        .filter_map(|(k, v)| {
            // println!("Dangling: {}, Value: {}", k, v);
            match v {
                1 => Some(k),
                -1 => Some(function!(AIND_SYMBOLS.dind, k)),
                _ => None,
            }
        })
        .collect()
}

pub fn wrap_dummies_impl(view: AtomView, header: Symbol) -> Atom {
    let a = view.expand();
    let settings = MatchSettings {
        level_range: (0, Some(1)),
        ..Default::default()
    };
    let mut dangling = HashMap::new();
    if let AtomView::Add(a) = a.as_view() {
        if let Some(first_term) = a.iter().next() {
            for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
                let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
                for p in first_term.pattern_match(&ipat, None, &settings) {
                    let entry = dangling.entry(ipat.replace_wildcards(&p));
                    entry
                        .or_insert((
                            0,
                            i.to_symbolic([
                                p[&RS.d_].clone(),
                                function!(header, p[&RS.a_].clone()),
                            ]),
                        ))
                        .0 += 1;
                }
            }
            for i in LibraryRep::all_dualizables() {
                let ipat = i.to_symbolic([RS.d_, RS.a_]).to_pattern();
                let ipat_dual = i.dual().to_symbolic([RS.d_, RS.a_]).to_pattern();

                for p in first_term.pattern_match(&ipat, None, &settings) {
                    let entry = dangling.entry(ipat.replace_wildcards(&p));
                    entry
                        .or_insert((
                            0,
                            i.to_symbolic([
                                p[&RS.d_].clone(),
                                function!(header, p[&RS.a_].clone()),
                            ]),
                        ))
                        .0 += 1;
                }
                for p in first_term.pattern_match(&ipat_dual, None, &settings) {
                    let entry = dangling.entry(ipat.replace_wildcards(&p));
                    entry
                        .or_insert((
                            0,
                            i.to_symbolic([
                                p[&RS.d_].clone(),
                                function!(header, p[&RS.a_].clone()),
                            ]),
                        ))
                        .0 += 1;
                }
            }
        }
    }

    let mut expr = view.to_owned();
    for (k, v) in dangling {
        match v.0 {
            1 | -1 => {}
            _ => expr = expr.replace(k.to_pattern()).repeat().with(v.1.to_pattern()),
        }
    }

    expr
}

pub fn simplify_metrics_impl(view: AtomView) -> Atom {
    let mut expr = view.expand();

    let mut reps = vec![];
    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        reps.extend(
            [
                (
                    function!(ETS.id, i.to_symbolic([RS.a__]), i.to_symbolic([RS.i__]))
                        * function!(RS.f_, RS.a___, i.to_symbolic([RS.i__]), RS.b___),
                    function!(RS.f_, RS.a___, i.to_symbolic([RS.a__]), RS.b___),
                ),
                (
                    function!(ETS.metric, i.to_symbolic([RS.a__]), i.to_symbolic([RS.i__]))
                        * function!(RS.f_, RS.a___, i.to_symbolic([RS.i__]), RS.b___),
                    function!(RS.f_, RS.a___, i.to_symbolic([RS.a__]), RS.b___),
                ),
                (
                    function!(
                        ETS.metric,
                        i.to_symbolic([RS.d_, RS.i_]),
                        i.to_symbolic([RS.d_, RS.a_])
                    )
                    .pow(Atom::new_num(2)),
                    Atom::new_var(RS.d_),
                ),
                (
                    function!(
                        ETS.metric,
                        i.to_symbolic([RS.d_, RS.i_]),
                        i.to_symbolic([RS.d_, RS.i_])
                    ),
                    Atom::new_var(RS.d_),
                ),
                (
                    function!(
                        ETS.id,
                        i.to_symbolic([RS.d_, RS.i_]),
                        i.to_symbolic([RS.d_, RS.i_])
                    )
                    .pow(Atom::new_num(2)),
                    Atom::new_var(RS.d_),
                ),
            ]
            .into_iter()
            .map(|(p, r)| Replacement::new(p.to_pattern(), r)),
        );
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();

        reps.extend(
            [
                (
                    function!(ETS.id, i.to_symbolic([RS.a__]), di.to_symbolic([RS.i__]))
                        * function!(RS.f_, RS.a___, i.to_symbolic([RS.i__]), RS.b___),
                    function!(RS.f_, RS.a___, i.to_symbolic([RS.a__]), RS.b___),
                ),
                (
                    function!(
                        ETS.id,
                        i.to_symbolic([RS.d_, RS.i_]),
                        di.to_symbolic([RS.d_, RS.a_])
                    )
                    .pow(Atom::new_num(2)),
                    Atom::new_var(RS.d_),
                ),
                (
                    function!(
                        ETS.metric,
                        i.to_symbolic([RS.d_, RS.i_]),
                        di.to_symbolic([RS.d_, RS.i_])
                    ),
                    Atom::new_var(RS.d_),
                ),
                (
                    function!(ETS.id, di.to_symbolic([RS.a__]), i.to_symbolic([RS.i__]))
                        * function!(RS.f_, RS.a___, di.to_symbolic([RS.i__]), RS.b___),
                    function!(RS.f_, RS.a___, di.to_symbolic([RS.a__]), RS.b___),
                ),
                (
                    function!(ETS.metric, i.to_symbolic([RS.a__]), i.to_symbolic([RS.i__]))
                        * function!(RS.f_, RS.a___, di.to_symbolic([RS.i__]), RS.b___),
                    function!(RS.f_, RS.a___, i.to_symbolic([RS.a__]), RS.b___),
                ),
                (
                    function!(
                        ETS.metric,
                        di.to_symbolic([RS.a__]),
                        di.to_symbolic([RS.i__])
                    ) * function!(RS.f_, RS.a___, i.to_symbolic([RS.i__]), RS.b___),
                    function!(RS.f_, RS.a___, di.to_symbolic([RS.a__]), RS.b___),
                ),
                (
                    function!(
                        ETS.metric,
                        di.to_symbolic([RS.a__]),
                        di.to_symbolic([RS.i__])
                    ) * function!(ETS.metric, i.to_symbolic([RS.i__]), i.to_symbolic([RS.b__])),
                    function!(ETS.id, di.to_symbolic([RS.a__]), i.to_symbolic([RS.b__])),
                ),
            ]
            .into_iter()
            .map(|(p, r)| Replacement::new(p.to_pattern(), r)),
        );
    }

    let mut atom = Atom::new();

    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
        expr = expr.expand();
    }

    expr
}

pub fn num_or_var(sym: Symbol) -> Condition<PatternRestriction> {
    sym.restrict(WildcardRestriction::IsAtomType(AtomType::Var))
        | sym.restrict(WildcardRestriction::IsAtomType(AtomType::Num))
}

pub fn to_dots_impl(expr: AtomView) -> Atom {
    let mut reps = vec![];

    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        reps.push(Replacement::new(
            (function!(RS.f_, i.to_symbolic([RS.i__])) * function!(RS.g_, i.to_symbolic([RS.i__])))
                .to_pattern(),
            function!(MS.dot, RS.f_, RS.g_),
        ));

        reps.push(
            Replacement::new(
                (function!(RS.f_, i.to_symbolic([RS.i__])).pow(Atom::new_num(2))).to_pattern(),
                function!(MS.dot, RS.f_, RS.f_),
            )
            .with_conditions(num_or_var(RS.x_)),
        );

        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x_, i.to_symbolic([RS.i__]))
                    * function!(RS.g_, RS.y_, i.to_symbolic([RS.i__])))
                .to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x_), function!(RS.g_, RS.y_)),
            )
            .with_conditions(num_or_var(RS.x_) & num_or_var(RS.y_)),
        );

        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x_, i.to_symbolic([RS.i__])).pow(Atom::new_num(2)))
                    .to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x_), function!(RS.f_, RS.x_)),
            )
            .with_conditions(num_or_var(RS.x_)),
        );
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();
        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x_, i.to_symbolic([RS.i__]))
                    * function!(RS.g_, RS.y_, di.to_symbolic([RS.i__])))
                .to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x_), function!(RS.g_, RS.y_)),
            )
            .with_conditions(num_or_var(RS.x_) & num_or_var(RS.y_)),
        );
    }

    let mut atom = Atom::new();
    let mut expr = expr.expand();
    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
    }

    expr
}

pub trait MetricSimplifier {
    fn simplify_metrics(&self) -> Atom;
    fn to_dots(&self) -> Atom;
}

impl MetricSimplifier for Atom {
    fn to_dots(&self) -> Atom {
        to_dots_impl(self.as_view())
    }
    fn simplify_metrics(&self) -> Atom {
        simplify_metrics_impl(self.as_view())
    }
}

impl<'a> MetricSimplifier for AtomView<'a> {
    fn to_dots(&self) -> Atom {
        to_dots_impl(*self)
    }
    fn simplify_metrics(&self) -> Atom {
        simplify_metrics_impl(*self)
    }
}
#[cfg(test)]
mod test {

    use crate::physics::algebraic_simplification::representations::initialize;

    use super::*;

    use symbolica::parse_lit;

    #[test]
    fn metric_contract() {
        initialize();
        let expr =
            parse_lit!(spenso::g(spenso::mink(4, 0), spenso::mink(4, 1)) * p(spenso::mink(4, 1)))
                .unwrap()
                .simplify_metrics();

        assert_eq!(
            expr,
            parse_lit!(p(spenso::mink(4, 0))).unwrap(),
            "got {:#}",
            expr
        );
    }
}
