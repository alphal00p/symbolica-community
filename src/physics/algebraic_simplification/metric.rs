use std::sync::LazyLock;

use spenso::{
    structure::representation::{LibraryRep, RepName},
    tensor_library::ETS,
};
use symbolica::{
    atom::{Atom, AtomCore, AtomType, AtomView, Symbol},
    function,
    id::{Condition, Replacement, WildcardRestriction},
    symbol,
};

use super::rep_symbols::RS;

pub struct MetricSymbols {
    pub dim: Symbol,
    pub dot: Symbol,
}

pub static MS: LazyLock<MetricSymbols> = LazyLock::new(|| MetricSymbols {
    dim: symbol!("dim"),
    dot: symbol!("dot"),
});

pub fn simplify_metrics_impl(view: AtomView) -> Atom {
    let mut expr = view.expand();

    let mut reps = vec![];
    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        reps.push(Replacement::new(
            (function!(ETS.id, i.to_pattern(RS.a__), i.to_pattern(RS.i__))
                * function!(RS.f_, RS.a___, i.to_pattern(RS.i__), RS.b___))
            .to_pattern(),
            function!(RS.f_, RS.a___, i.to_pattern(RS.a__), RS.b___),
        ));
        reps.push(Replacement::new(
            (function!(ETS.metric, i.to_pattern(RS.a__), i.to_pattern(RS.i__))
                * function!(RS.f_, RS.a___, i.to_pattern(RS.i__), RS.b___))
            .to_pattern(),
            function!(RS.f_, RS.a___, i.to_pattern(RS.a__), RS.b___),
        ));
        reps.push(Replacement::new(
            (function!(
                ETS.metric,
                i.to_symbolic([RS.d_, RS.i_]),
                i.to_symbolic([RS.d_, RS.a_])
            )
            .pow(Atom::new_num(2)))
            .to_pattern(),
            Atom::new_var(RS.d_),
        ));
        reps.push(Replacement::new(
            function!(
                ETS.metric,
                i.to_symbolic([RS.d_, RS.i_]),
                i.to_symbolic([RS.d_, RS.i_])
            )
            .to_pattern(),
            Atom::new_var(RS.d_),
        ));
        reps.push(Replacement::new(
            (function!(
                ETS.id,
                i.to_symbolic([RS.d_, RS.i_]),
                i.to_symbolic([RS.d_, RS.i_])
            )
            .pow(Atom::new_num(2)))
            .to_pattern(),
            Atom::new_var(RS.d_),
        ));
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();
        reps.push(Replacement::new(
            (function!(ETS.id, i.to_pattern(RS.a__), di.to_pattern(RS.i__))
                * function!(RS.f_, RS.a___, i.to_pattern(RS.i__), RS.b___))
            .to_pattern(),
            function!(RS.f_, RS.a___, i.to_pattern(RS.a__), RS.b___),
        ));
        reps.push(Replacement::new(
            (function!(
                ETS.id,
                i.to_symbolic([RS.d_, RS.i_]),
                di.to_symbolic([RS.d_, RS.a_])
            )
            .pow(Atom::new_num(2)))
            .to_pattern(),
            Atom::new_var(RS.d_),
        ));
        reps.push(Replacement::new(
            function!(
                ETS.metric,
                i.to_symbolic([RS.d_, RS.i_]),
                di.to_symbolic([RS.d_, RS.i_])
            )
            .to_pattern(),
            Atom::new_var(RS.d_),
        ));

        reps.push(Replacement::new(
            (function!(ETS.id, di.to_pattern(RS.a__), i.to_pattern(RS.i__))
                * function!(RS.f_, RS.a___, di.to_pattern(RS.i__), RS.b___))
            .to_pattern(),
            function!(RS.f_, RS.a___, di.to_pattern(RS.a__), RS.b___),
        ));
        reps.push(Replacement::new(
            (function!(ETS.metric, i.to_pattern(RS.a__), i.to_pattern(RS.i__))
                * function!(RS.f_, RS.a___, di.to_pattern(RS.i__), RS.b___))
            .to_pattern(),
            function!(RS.f_, RS.a___, i.to_pattern(RS.a__), RS.b___),
        ));

        reps.push(Replacement::new(
            (function!(ETS.metric, di.to_pattern(RS.a__), di.to_pattern(RS.i__))
                * function!(RS.f_, RS.a___, i.to_pattern(RS.i__), RS.b___))
            .to_pattern(),
            function!(RS.f_, RS.a___, di.to_pattern(RS.a__), RS.b___),
        ));

        reps.push(Replacement::new(
            (function!(ETS.metric, di.to_pattern(RS.a__), di.to_pattern(RS.i__))
                * function!(ETS.metric, i.to_pattern(RS.i__), i.to_pattern(RS.b__)))
            .to_pattern(),
            function!(ETS.id, di.to_pattern(RS.a__), i.to_pattern(RS.b__)),
        ));
    }

    let mut atom = Atom::new();

    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
        expr = expr.expand();
    }

    expr
}

pub fn to_dots_impl(expr: AtomView) -> Atom {
    let mut reps = vec![];
    for i in LibraryRep::all_self_duals().chain(LibraryRep::all_inline_metrics()) {
        reps.push(Replacement::new(
            (function!(RS.f_, i.to_pattern(RS.i__)) * function!(RS.g_, i.to_pattern(RS.i__)))
                .to_pattern(),
            function!(MS.dot, RS.f_, RS.g_),
        ));

        reps.push(
            Replacement::new(
                (function!(RS.f_, i.to_pattern(RS.i__)).pow(Atom::new_num(2))).to_pattern(),
                function!(MS.dot, RS.f_, RS.f_),
            )
            .with_conditions(
                RS.x_
                    .restrict(WildcardRestriction::IsAtomType(AtomType::Var)),
            ),
        );

        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x_, i.to_pattern(RS.i__))
                    * function!(RS.g_, RS.y_, i.to_pattern(RS.i__)))
                .to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x_), function!(RS.g_, RS.y_)),
            )
            .with_conditions(Condition::And(Box::new((
                RS.x_
                    .restrict(WildcardRestriction::IsAtomType(AtomType::Var)),
                RS.y_
                    .restrict(WildcardRestriction::IsAtomType(AtomType::Var)),
            )))),
        );

        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x_, i.to_pattern(RS.i__)).pow(Atom::new_num(2))).to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x_), function!(RS.f_, RS.x_)),
            )
            .with_conditions(
                RS.x_
                    .restrict(WildcardRestriction::IsAtomType(AtomType::Var)),
            ),
        );
    }

    for i in LibraryRep::all_dualizables() {
        let di = i.dual();
        reps.push(
            Replacement::new(
                (function!(RS.f_, RS.x_, i.to_pattern(RS.i__))
                    * function!(RS.g_, RS.y_, di.to_pattern(RS.i__)))
                .to_pattern(),
                function!(MS.dot, function!(RS.f_, RS.x_), function!(RS.g_, RS.y_)),
            )
            .with_conditions(Condition::And(Box::new((
                RS.x_
                    .restrict(WildcardRestriction::IsAtomType(AtomType::Var)),
                RS.y_
                    .restrict(WildcardRestriction::IsAtomType(AtomType::Var)),
            )))),
        );
    }

    let mut atom = Atom::new();
    let mut expr = expr.expand();
    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
        // expr = expr.expand();
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
