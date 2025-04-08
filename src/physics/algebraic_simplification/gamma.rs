use std::sync::LazyLock;

use spenso::{
    structure::representation::{Minkowski, RepName},
    tensor_library::ETS,
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, FunctionBuilder, Symbol},
    function,
    id::{Context, Replacement},
    symbol,
};

use crate::physics::algebraic_simplification::{metric::MetricSimplifier, rep_symbols::RS};

pub struct GammaSymbols {
    pub gamma: Symbol,
    pub projp: Symbol,
    pub projm: Symbol,
    pub gamma5: Symbol,
    pub dim: Symbol,
    pub gamma_chain: Symbol,
    pub gamma_trace: Symbol,
    pub dot: Symbol,
}

pub static GS: LazyLock<GammaSymbols> = LazyLock::new(|| GammaSymbols {
    gamma: symbol!("gamma"),
    projp: symbol!("projp"),
    projm: symbol!("projm"),
    gamma5: symbol!("gamma5"),
    dim: symbol!("dim"),
    gamma_chain: symbol!("gamma_chain"),
    gamma_trace: symbol!("gamma_trace"),
    dot: symbol!("dot"),
});

pub fn gamma_simplify_impl(expr: AtomView) -> Atom {
    let mink = Minkowski {};

    let mut expr = expr.expand();

    let reps: Vec<_> = [
        (
            function!(GS.projp, RS.a_, RS.b_),
            (function!(ETS.id, RS.a_, RS.b_) - function!(GS.gamma5, RS.a_, RS.b_)) / 2,
        ),
        (
            function!(GS.projm, RS.a_, RS.b_),
            (function!(ETS.id, RS.a_, RS.b_) + function!(GS.gamma5, RS.a_, RS.b_)) / 2,
        ),
        (
            function!(GS.gamma, RS.a_, RS.b_, RS.c_) * function!(GS.gamma, RS.d_, RS.c_, RS.e_),
            function!(GS.gamma_chain, RS.a_, RS.d_, RS.b_, RS.e_),
        ),
        (function!(GS.gamma, RS.a_, RS.b_, RS.b_), Atom::Zero),
        (
            function!(GS.gamma_chain, RS.a__, RS.a_, RS.b_)
                * function!(GS.gamma_chain, RS.b__, RS.b_, RS.c_),
            function!(GS.gamma_chain, RS.a__, RS.b__, RS.a_, RS.c_),
        ),
        (
            function!(GS.gamma_chain, RS.a__, RS.a_, RS.b_)
                * function!(GS.gamma, RS.y_, RS.b_, RS.c_),
            function!(GS.gamma_chain, RS.a__, RS.y_, RS.a_, RS.c_),
        ),
        (
            function!(GS.gamma, RS.a_, RS.a_, RS.b_)
                * function!(GS.gamma_chain, RS.y__, RS.b_, RS.c_),
            function!(GS.gamma_chain, RS.a_, RS.y__, RS.a_, RS.c_),
        ),
    ]
    .iter()
    .map(|(a, b)| {
        // println!("{}->{}", a, b);

        Replacement::new(a.to_pattern(), b.to_pattern())
    })
    .collect();

    let mut atom = Atom::new();

    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
        expr = expr.expand();
        expr = expr.simplify_metrics();
    }

    let reps: Vec<_> = [
        (
            function!(
                GS.gamma_chain,
                RS.a___,
                mink.to_symbolic([RS.d_, RS.a_]),
                mink.to_symbolic([RS.d_, RS.a_]),
                RS.b__
            ),
            function!(GS.gamma_chain, RS.a___, RS.b__) * RS.d_,
        ),
        (
            function!(GS.gamma_chain, RS.a_, RS.b_),
            function!(ETS.id, RS.a_, RS.b_),
        ),
        (
            function!(
                GS.gamma_chain,
                RS.a___,
                RS.a_,
                RS.b___,
                RS.b_,
                RS.a_,
                RS.a__
            ),
            function!(GS.gamma_chain, RS.a___, RS.b_, RS.b___, RS.a__) * 2
                - function!(
                    GS.gamma_chain,
                    RS.a___,
                    RS.a_,
                    RS.b___,
                    RS.a_,
                    RS.b_,
                    RS.a__
                ),
        ),
    ]
    .iter()
    .map(|(a, b)| {
        Replacement::new(a.to_pattern(), b.to_pattern())
        // .with_conditions(symbolica::id::Condition::Yield(()))
    })
    .collect();

    while expr.replace_multiple_into(&reps, &mut atom) {
        std::mem::swap(&mut expr, &mut atom);
    }

    fn gamma_chain_accumulator(arg: AtomView, _context: &Context, out: &mut Atom) -> bool {
        if let AtomView::Fun(f) = arg {
            if f.get_symbol() == GS.gamma_chain {
                let mut args = f.iter().collect::<Vec<_>>();
                if args.len() >= 4 {
                    for i in 0..args.len().saturating_sub(3) {
                        // println!("{}", args[i]);
                        // println!("{}?{}", args[i], args[i + 1]);
                        if args[i] > args[i + 1] {
                            // println!("{}>{}", args[i], args[i + 1]);
                            args.swap(i, i + 1);
                            let swapped = FunctionBuilder::new(GS.gamma_chain)
                                .add_args(&args)
                                .finish();
                            let mu = args.remove(i);
                            let nu = args.remove(i);
                            let metric = function!(ETS.metric, mu, nu)
                                * 2
                                * FunctionBuilder::new(GS.gamma_chain)
                                    .add_args(&args)
                                    .finish();
                            *out = metric - swapped;
                            // println!("{}->{}", a, c);
                            return true;
                        }
                    }
                    return false;
                } else {
                    return false;
                }
            }
            false
        } else {
            false
        }
    }

    loop {
        let new = expr
            .replace_map(&gamma_chain_accumulator)
            .replace_multiple(&reps);
        if new == expr {
            break;
        } else {
            expr = new;
        }
    }

    expr = expr
        .replace(function!(GS.gamma_chain, RS.a__, RS.x_, RS.x_).to_pattern())
        .repeat()
        .with(function!(GS.gamma_trace, RS.a__).to_pattern());

    // //Chisholm identity:
    // expr.replace_all_repeat_mut(
    //     &(function!(GS.gamma, RS.a_, RS.x_, RS.y_) * function!(gamma_trace, RS.a_, RS.a__)).to_pattern(),
    //     (function!(gamma_chain, RS.a__)).to_pattern(),
    //     None,
    //     None,
    // );
    //
    fn gamma_tracer(arg: AtomView, _context: &Context, out: &mut Atom) -> bool {
        let gamma_trace = symbol!("gamma_trace");

        let mut found = false;
        if let AtomView::Fun(f) = arg {
            if f.get_symbol() == gamma_trace {
                // println!("{arg}");
                found = true;
                let mut sum = Atom::Zero;

                if f.get_nargs() == 1 {
                    *out = Atom::Zero;
                }
                let args = f.iter().collect::<Vec<_>>();

                for i in 1..args.len() {
                    let sign = if i % 2 == 0 { -1 } else { 1 };

                    let mut gcn = FunctionBuilder::new(gamma_trace);
                    #[allow(clippy::needless_range_loop)]
                    for j in 1..args.len() {
                        if i != j {
                            gcn = gcn.add_arg(args[j]);
                        }
                    }

                    let metric = if args[0] == args[i] {
                        if let AtomView::Fun(f) = args[0].as_atom_view() {
                            f.iter().next().unwrap().to_owned()
                        } else {
                            panic!("aaaa")
                        }
                        // Atom::new_num(4)
                    } else {
                        function!(ETS.metric, args[0], args[i])
                    };
                    if args.len() == 2 {
                        sum = sum + metric * sign * Atom::new_num(4);
                    } else {
                        sum = sum + metric * gcn.finish() * sign;
                    }
                }
                *out = sum;

                // println!("{}->{}", arg, out);
            }
        }

        found
    }

    loop {
        let new = expr.replace_map(&gamma_tracer);
        if new == expr {
            break;
        } else {
            expr = new;
        }
    }

    expr = expr
        .replace(
            function!(GS.gamma, mink.to_symbolic([RS.d_, RS.b_]), RS.a__)
                .pow(Atom::new_num(2))
                .to_pattern(),
        )
        .repeat()
        .with(Atom::new_var(RS.d_) * 4)
        .simplify_metrics();

    expr
}

pub trait GammaSimplifier {
    fn simplify_gamma(&self) -> Atom;
}

impl GammaSimplifier for Atom {
    fn simplify_gamma(&self) -> Atom {
        gamma_simplify_impl(self.as_atom_view())
    }
}

impl<'a> GammaSimplifier for AtomView<'a> {
    fn simplify_gamma(&self) -> Atom {
        gamma_simplify_impl(self.as_atom_view())
    }
}

pub fn id_atom(i: impl Into<Atom>, j: impl Into<Atom>) -> Atom {
    function!(ETS.id, i.into(), j.into())
}

#[macro_export]
macro_rules! id {
    ($i: expr, $j: expr) => {{
        let i = symbolica::parse_lit!($i).unwrap();
        let j = symbolica::parse_lit!($j).unwrap();
        id_atom(i, j)
    }};
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::id;
    use symbolica::{
        atom::{Atom, AtomCore},
        parse_lit,
    };

    use crate::physics::algebraic_simplification::representations::initialize;

    #[test]
    fn gamma_alg() {
        initialize();
        let expr = parse_lit!(
            symbolica_community::gamma_chain(mink(4, 0), mink(4, 0), b(1), b(2)),
            "spenso"
        )
        .unwrap()
        .simplify_gamma();

        assert_eq!(expr, id!(spenso::b(1), spenso::b(2)) * 4, "got {:#}", expr);

        let expr = parse_lit!(
            p(mink(4, nu1))
                * (p(mink(4, nu3)) + q(mink(4, nu3)))
                * symbolica_community::gamma_chain(
                    mink(4, nu1),
                    mink(4, mu),
                    mink(4, nu3),
                    mink(4, nu),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .unwrap()
        .simplify_gamma()
        .expand();
        assert_eq!(
            expr,
            parse_lit!(
                -4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1))
                    ^ 2 + 8 * p(mink(4, mu)) * p(mink(4, nu))
                        + 4 * p(mink(4, mu)) * q(mink(4, nu))
                        + 4 * p(mink(4, nu)) * q(mink(4, mu))
                        - 4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1)) * q(mink(4, nu1)),
                "spenso"
            )
            .unwrap(),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            g(mink(dim, 5), mink(dim, 6))
                * (g(mink(dim, 1), mink(dim, 2))
                    * g(mink(dim, 3), mink(dim, 4))
                    * g(mink(dim, 5), mink(dim, 6))
                    - g(mink(dim, 1), mink(dim, 3))
                        * g(mink(dim, 2), mink(dim, 6))
                        * g(mink(dim, 5), mink(dim, 4)))
                * (g(mink(dim, 1), mink(dim, 2)) * g(mink(dim, 3), mink(dim, 4))
                    - g(mink(dim, 1), mink(dim, 3)) * g(mink(dim, 2), mink(dim, 4))),
            "spenso"
        )
        .unwrap()
        .simplify_gamma();
        assert_eq!(
            expr,
            parse_lit!(-dim + dim ^ 3, "spenso").unwrap(),
            "got {}",
            expr
        );

        let expr = parse_lit!(
            p(mink(4, nu1))
                * (p(mink(4, nu3)) + q(mink(4, nu3)))
                * symbolica_community::gamma_chain(
                    mink(4, nu1),
                    mink(4, mu),
                    mink(4, nu),
                    mink(4, nu3),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .unwrap()
        .simplify_gamma();
        assert_eq!(
            expr,
            parse_lit!(
                4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1))
                    ^ 2 + 4 * p(mink(4, mu)) * q(mink(4, nu)) - 4 * q(mink(4, mu)) * p(mink(4, nu))
                        + 4 * g(mink(4, mu), mink(4, nu)) * p(mink(4, nu1)) * q(mink(4, nu1)),
                "spenso"
            )
            .unwrap(),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            p(mink(dim, nu1))
                * (p(mink(dim, nu3)) + q(mink(dim, nu3)))
                * symbolica_community::gamma_chain(
                    mink(dim, nu1),
                    mink(dim, nu),
                    mink(dim, nu),
                    mink(dim, nu3),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .unwrap()
        .simplify_gamma();
        assert_eq!(
            expr,
            parse_lit!(
                4 * dim * p(mink(dim, nu1)) ^ 2 + 4 * dim * p(mink(dim, nu1)) * q(mink(dim, nu1)),
                "spenso"
            )
            .unwrap(),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            p(mink(dim, nu1))
                * (p(mink(dim, nu3)) + q(mink(dim, nu3)))
                * symbolica_community::gamma_chain(
                    mink(dim, nu1),
                    mink(dim, nu),
                    mink(dim, nu3),
                    mink(dim, nu),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .unwrap()
        .simplify_gamma();
        assert_eq!(
            expr,
            parse_lit!(
                8 * p(mink(dim, nu1))
                    ^ 2 - 4 * dim * p(mink(dim, nu1))
                    ^ 2 + 8 * p(mink(dim, nu1)) * q(mink(dim, nu1))
                        - 4 * dim * p(mink(dim, nu1)) * q(mink(dim, nu1)),
                "spenso"
            )
            .unwrap(),
            "got {:#}",
            expr
        );

        let expr = parse_lit!(
            symbolica_community::p(mink(dim, nu1))
                * symbolica_community::q(mink(dim, nu2))
                * (symbolica_community::p(mink(dim, nu3)) + symbolica_community::q(mink(dim, nu3)))
                * symbolica_community::q(mink(dim, nu4))
                * symbolica_community::gamma_chain(
                    mink(dim, nu1),
                    mink(dim, nu4),
                    mink(dim, nu3),
                    mink(dim, nu2),
                    b(1),
                    b(1)
                ),
            "spenso"
        )
        .unwrap()
        .simplify_gamma()
        .to_dots();
        assert_eq!(
            expr,
            parse_lit!(8 * dot(p, q) ^ 2 - 4 * dot(p, p) * dot(q, q) + 4 * dot(p, q) * dot(q, q))
                .unwrap(),
            "got {}",
            expr
        );

        let expr = parse_lit!(
            symbolica_community::gamma_chain(
                mink(dim, mu),
                mink(dim, nu),
                mink(dim, mu),
                mink(dim, nu),
                b(1),
                b(2)
            ),
            "spenso"
        )
        .unwrap()
        .simplify_gamma()
        .to_dots();

        let dim = Atom::new_var(symbol!("spenso::dim"));
        assert_eq!(
            expr,
            &dim * id!(spenso::b(1), spenso::b(2)) * 2
                - dim.pow(Atom::new_num(2)) * id!(spenso::b(1), spenso::b(2)),
            "got {}",
            expr
        );
    }
}
