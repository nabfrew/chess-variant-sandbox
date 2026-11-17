//! R-L5c2 (`docs/reference/common-path-plan.md` §2): the common path compiled twice.
//!
//! A rare rule's test is a *gate*: `rare!(P::SWEEP, t.has_sweep)` is `P::SWEEP && t.has_sweep`,
//! so a copy whose `P` says the family is off (`Lean`) has the test and the arm behind it
//! deleted by the compiler, and the copy whose `P` says it is on (`Full`) is today's code. One
//! text, two compilations: a game picks its copy once, when its tables are built
//! (`Tables::lean`, from `Tables::rare_in_use`), and each public entry (`generate_legal`,
//! `generate_pseudo_legal`, `restrict`) asks that one flag and calls its copy, inside which
//! nothing dispatches again.
//!
//! What keeps the two copies one game (§2.5): a lean game reads none of the fields a gate
//! names (`rare_in_use` reads every one of them; `tests/test_lean_path.py` reads this crate's
//! gates and fails on a field `rare_in_use` does not read), a checking build asserts every
//! gate on the lean copy (the `debug_assert!` in `rare!` and `rare_opt`), and the forced-full
//! switch (`force_full_path`, tests only) plays a lean game on the full copy for the equality
//! tests. The `Full` copy is correct for every game; `Lean` only for a lean one.
//!
//! A rule stage that adds a test to a common-path function writes it behind its family's gate
//! and adds the field to `rare_in_use` (skill `new-move-primitive`), so the lean copy, which
//! chess plays, is not recompiled by it.

use std::sync::atomic::{AtomicBool, Ordering};

/// One constant per rare family (§2.3). A family is rare because chess does not use it. Each
/// defaults to `ON`, so a tier (R-L5c5) is one more `impl` overriding the families it keeps.
pub(crate) trait Path {
    /// Every family's default: false on the lean copy, true on the full ones.
    const ON: bool;
    /// Schema 22: the copy walks folds (`generate_folded`'s). Only the full copy's folded
    /// monomorph sets it; the full copy itself keeps the runtime dispatch to it.
    const FOLDS: bool;
    /// `drops` (placement, gates, drop mate), and a drop's royal entry.
    const HAND: bool = Self::ON;
    /// `strike` (the `choose` fan, the shortcut's strike test, an own contact link) and
    /// `explosion` in the shortcut's premise.
    const CAPTURE: bool = Self::ON;
    /// `has_capture_matrix`.
    const MATRIX: bool = Self::ON;
    /// `has_sweep`.
    const SWEEP: bool = Self::ON;
    /// `push` and a landing's `MODE_PUSH`.
    const PUSH: bool = Self::ON;
    /// `hops`, `hop_runs`, `has_hop_attacks`.
    const HOPS: bool = Self::ON;
    /// `portals`, folds.
    const PORTALS: bool = Self::ON;
    /// `sight`.
    const SIGHT: bool = Self::ON;
    /// `majority`: nobody's pieces a player moves.
    const HELD: bool = Self::ON;
    /// `has_npcs`.
    const WILD: bool = Self::ON;
    /// `duck`.
    const DUCK: bool = Self::ON;
    /// `purse`.
    const PURSE: bool = Self::ON;
    /// `rest`, `parts`, `passes`.
    const TURN: bool = Self::ON;
    /// `has_facing`.
    const FACING: bool = Self::ON;
    /// `multi_royal`, `royal_facing_forbidden`.
    const ROYALS: bool = Self::ON;
    /// `forbid_check`, `forbid_immobility`, `must_capture`, `region_play`.
    const RESTRICT: bool = Self::ON;
    /// `entry_only_lost` (SR8a's count of the board before a move, R-L5c3).
    const ENTRY: bool = Self::ON;
}

/// Every family off: the copy a game with no rare rule runs.
pub(crate) struct Lean;
impl Path for Lean {
    const ON: bool = false;
    const FOLDS: bool = false;
}

/// Every family on: today's code, correct for every game.
pub(crate) struct Full;
impl Path for Full {
    const ON: bool = true;
    const FOLDS: bool = false;
}

/// R-L5c3: the full copy of the legality filter for a game with no wild piece, so `WILD`
/// off and every other family on. The full copy of the filter keeps R-L5d's runtime
/// dispatch on `has_npcs` (as `FOLDS` keeps its own): a wild game's filter is `Full`'s, every
/// other full game's this one's, and neither pays for the other's arm. Only the filter's
/// entries choose it (`leaves_royal_attacked_p`, `gives_check_p`), after reading `has_npcs`.
pub(crate) struct Tame;
impl Path for Tame {
    const ON: bool = true;
    const FOLDS: bool = false;
    const WILD: bool = false;
}

/// The full copy that walks folds (schema 22's `FOLDS` monomorph).
pub(crate) struct Folded;
impl Path for Folded {
    const ON: bool = true;
    const FOLDS: bool = true;
}

/// A rare rule's test behind its family's constant: `$family && $test`. A checking build
/// asserts that a copy with the family off never meets the rule in use, which is what
/// `Tables::rare_in_use` promises.
macro_rules! rare {
    ($family:expr, $test:expr) => {{
        debug_assert!($family || !($test), "a lean copy met a rare rule: {}", stringify!($test));
        $family && $test
    }};
}
pub(crate) use rare;

/// `rare!` for an `Option` read: the value on a copy with the family on, `None` otherwise.
#[inline(always)]
pub(crate) fn rare_opt<T>(family: bool, value: Option<T>) -> Option<T> {
    debug_assert!(family || value.is_none(), "a lean copy met a rare rule");
    if family {
        value
    } else {
        None
    }
}

/// The forced-full switch (§2.5 item 4): while set, tables built from then on choose the full
/// copy whatever they hold, so a test can play a lean game on both copies. Never set outside
/// tests (`_core.force_full_path`).
static FORCE_FULL: AtomicBool = AtomicBool::new(false);

pub(crate) fn force_full(on: bool) {
    FORCE_FULL.store(on, Ordering::SeqCst);
}

pub(crate) fn forced_full() -> bool {
    FORCE_FULL.load(Ordering::SeqCst)
}
