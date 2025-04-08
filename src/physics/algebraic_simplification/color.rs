use std::sync::LazyLock;

use spenso::{
    structure::representation::{LibraryRep, RepName},
    tensor_library::ETS,
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    function,
    id::{MatchSettings, Replacement},
    symbol,
};

use super::representations::{ColorAdjoint, ColorFundamental};
use super::{rep_symbols::RS, representations::ColorSextet};

#[derive(Debug)]
pub enum ColorError {
    NotFully(Atom),
}

pub struct ColorSymbols {
    pub t: Symbol,
    pub f: Symbol,
    pub tr: Symbol,
    pub nc: Symbol,
}

pub static CS: LazyLock<ColorSymbols> = LazyLock::new(|| ColorSymbols {
    t: symbol!("t"),
    f: symbol!("f"),
    tr: symbol!("TR"),
    nc: symbol!("Nc"),
});

pub fn color_simplify_impl(expression: AtomView) -> Result<Atom, ColorError> {
    let cof = ColorFundamental {};
    let coaf = ColorFundamental {}.dual();
    let coad = ColorAdjoint {};
    let tr = Atom::new_var(CS.tr);
    let nc = Atom::new_var(CS.nc);
    let reps = vec![
        (
            function!(RS.f_, RS.a___, cof.to_pattern(RS.b_), RS.c___)
                * function!(ETS.id, coaf.to_pattern(RS.b_), cof.to_pattern(RS.c_)),
            function!(RS.f_, RS.a___, cof.to_pattern(RS.c_), RS.c___),
        ),
        (
            function!(RS.f_, RS.a___, coaf.to_pattern(RS.b_), RS.c___)
                * function!(ETS.id, cof.to_pattern(RS.b_), coaf.to_pattern(RS.c_)),
            function!(RS.f_, RS.a___, coaf.to_pattern(RS.c_), RS.c___),
        ),
        (
            function!(RS.f_, RS.a___, coad.to_pattern(RS.b_), RS.c___)
                * function!(ETS.id, coad.to_pattern(RS.b_), coad.to_pattern(RS.a_)),
            function!(RS.f_, RS.a___, coad.to_pattern(RS.a_), RS.c___),
        ),
        (
            function!(RS.f_, RS.a___, coad.to_pattern(RS.a_), RS.c___)
                * function!(ETS.id, coad.to_pattern(RS.b_), coad.to_pattern(RS.a_)),
            function!(RS.f_, RS.a___, coad.to_pattern(RS.b_), RS.c___),
        ),
        (
            function!(ETS.id, coaf.to_pattern(RS.a_), cof.to_pattern(RS.a_)),
            nc.clone(),
        ),
        (
            function!(ETS.id, cof.to_pattern(RS.a_), coaf.to_pattern(RS.a_)),
            nc.clone(),
        ),
        (
            function!(ETS.id, coad.to_pattern(RS.a_), coad.to_pattern(RS.a_)),
            (&nc * &nc) - 1,
        ),
        (
            function!(CS.t, RS.a_, cof.to_pattern(RS.b_), coaf.to_pattern(RS.b_)),
            Atom::new_num(0),
        ),
        (
            function!(CS.t, RS.a_, cof.to_pattern(RS.c_), coaf.to_pattern(RS.e_))
                * function!(CS.t, RS.b_, cof.to_pattern(RS.e_), coaf.to_pattern(RS.c_)),
            &tr * function!(ETS.id, RS.a_, RS.b_),
        ),
        (
            function!(CS.t, RS.a_, cof.to_pattern(RS.c_), coaf.to_pattern(RS.e_))
                .pow(Atom::new_num(2)),
            &tr * function!(ETS.id, RS.a_, RS.a_),
        ),
        (
            function!(CS.t, RS.e_, RS.a_, RS.b_) * function!(CS.t, RS.e_, RS.c_, RS.d_),
            &tr * (function!(ETS.id, RS.a_, RS.d_) * function!(ETS.id, RS.c_, RS.b_)
                - (function!(ETS.id, RS.a_, RS.b_) * function!(ETS.id, RS.c_, RS.d_) / &nc)),
        ),
        (
            function!(CS.t, RS.i_, RS.a_, coaf.to_pattern(RS.b_))
                * function!(CS.t, RS.e_, cof.to_pattern(RS.b_), coaf.to_pattern(RS.c_))
                * function!(CS.t, RS.i_, cof.to_pattern(RS.c_), RS.d_),
            -(&tr / &nc) * function!(CS.t, RS.e_, RS.a_, RS.d_),
        ),
        (
            function!(
                CS.f,
                coad.to_pattern(RS.a_),
                coad.to_pattern(RS.b_),
                coad.to_pattern(RS.c_)
            )
            .pow(Atom::new_num(2)),
            &nc * (&nc * &nc - 1),
        ),
    ];

    let i = symbol!("i");
    let j = symbol!("j");
    let k = symbol!("k");

    let frep = [Replacement::new(
        function!(
            CS.f,
            coad.to_pattern(RS.a_),
            coad.to_pattern(RS.b_),
            coad.to_pattern(RS.c_)
        )
        .to_pattern(),
        (((function!(
            CS.t,
            coad.to_pattern(RS.a_),
            cof.to_symbolic([function!(i, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([function!(j, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_pattern(RS.b_),
            cof.to_symbolic([function!(j, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([function!(k, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_pattern(RS.c_),
            cof.to_symbolic([function!(k, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([function!(i, RS.a_, RS.b_, RS.c_)])
        ) - function!(
            CS.t,
            coad.to_pattern(RS.a_),
            cof.to_symbolic([function!(i, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([function!(j, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_pattern(RS.c_),
            cof.to_symbolic([function!(j, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([function!(k, RS.a_, RS.b_, RS.c_)])
        ) * function!(
            CS.t,
            coad.to_pattern(RS.b_),
            cof.to_symbolic([function!(k, RS.a_, RS.b_, RS.c_)]),
            coaf.to_symbolic([function!(i, RS.a_, RS.b_, RS.c_)])
        )) / &tr)
            * -Atom::new_var(Atom::I))
        .to_pattern(),
    )];

    let settings = MatchSettings {
        rhs_cache_size: 0,
        ..Default::default()
    };
    let replacements: Vec<Replacement> = reps
        .into_iter()
        .map(|(a, b)| {
            Replacement::new(a.to_pattern(), b.to_pattern()).with_settings(settings.clone())
        })
        .collect();

    let mut atom = Atom::new_num(0);
    // for r in &replacements {
    //     println!("{r}")
    // }

    let mut expression = expression.to_owned();
    let mut first = true;
    while first || expression.replace_multiple_into(&replacements, &mut atom) {
        if !first {
            std::mem::swap(&mut expression, &mut atom)
        };
        first = false;
        expression = expression.replace_multiple(&frep);
        expression = expression.expand();
    }

    let pats: Vec<LibraryRep> = vec![ColorAdjoint {}.into()];
    let dualizablepats: Vec<LibraryRep> = vec![ColorFundamental {}.into(), ColorSextet {}.into()];

    let mut fully_simplified = true;
    for p in pats.iter().chain(&dualizablepats) {
        if expression
            .pattern_match(&p.to_pattern(RS.a_).to_pattern(), None, None)
            .next()
            .is_some()
        {
            fully_simplified = false;
        }
    }

    if fully_simplified {
        Ok(expression)
    } else {
        Err(ColorError::NotFully(expression))
    }
}
