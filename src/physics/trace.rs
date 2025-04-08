use std::sync::LazyLock;
use symbolica::{
    atom::{Atom, AtomView, Symbol},
    symbol,
};

pub fn trace(a: AtomView) -> Atom {
    // TODO
    a.to_owned()
}
