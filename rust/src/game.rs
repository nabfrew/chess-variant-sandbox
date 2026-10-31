//! The rule interpreter: move generation, application, and terminal detection.
//!
//! This is the engine: `chess_variant_sandbox/engine/engine.py` is a facade over it
//! (there is no second implementation), and it walks the lookup tables that
//! `engine/compile.py` builds from the spec. Where a faster formulation would have changed
//! which moves come out in which order, the established order won, because a game is a
//! pure function of `(ruleset_hash, seed, agent_configs)` and replays must not move.
//!
//! The one structural departure is make/unmake. Python's `apply` returns a new
//! immutable state because that is what makes states safe to hand between processes;
//! here a search runs inside one thread on one position, so a move is applied in place
//! and taken back, and the repetition window is a pair of indices into a single history
//! rather than a tuple copied per node.

use std::cell::Cell;

use crate::tables::*;

// Schema 26 (Q918a): rounds, a child of this module so it walks the same private state.
// A file beside this one rather than under `game/`, because the build fingerprint
// (`build.rs`, `measurement.py`) hashes `rust/src/*.rs` and nothing deeper.
#[path = "rounds.rs"]
mod rounds;
// Schema 27 (Q935a): the spy's phases, a child for the same reason.
#[path = "spy.rs"]
mod spy;
pub use spy::Phase;
// Schema 29 (Q983a): rest as a piece-type property, one rest for turns and rounds.
#[path = "rest.rs"]
mod rest;
pub use rest::{rest_lists_from, RestLists, NO_REST};
// Schema 30 (Q999a): lines of sight with passive effects, one query for movegen and attack.
#[path = "sight.rs"]
mod sight;
// TC1 (`docs/plan-turncoat.md` §3.3): the one source of foreign moves, a piece the mover moves
// but does not own (today a commanded one).
#[path = "foreign.rs"]
mod foreign;
// Generation 36 (N1, `docs/archive/plan-npc.md` §4.1): wild pieces that come back, step 3 of the phase.
#[path = "respawn.rs"]
mod respawn;
pub use respawn::respawn_clocks_from;
// R6 (CS10a, `docs/plan-smells-rust.md`): the wild phase's walk, once, generic over its sink.
#[path = "phase.rs"]
mod phase;
use phase::{Coming, Filter, Live};
// Generation 38 (N5, `docs/archive/plan-npc.md` §4.7): wild pieces that turn back, step 2 on a line.
#[path = "turning.rs"]
mod turning;
pub use turning::heading_from;
// Generation 39 (N3b, `docs/archive/plan-npc.md` §4.3): wild pieces that pounce, step 1 of the phase.
#[path = "pounce.rs"]
mod pounce;
// Generation 42 (N4b, `docs/archive/plan-npc.md` §4.6): the duck, the second half of every turn.
#[path = "duck.rs"]
mod duck;
// Q1074b2 (`docs/plan-push.md`): the push rule, a move that moves the piece it lands on.
#[path = "push.rs"]
mod push;
use push::{PUSH_ARM_HOP, PUSH_ARM_LEAP, PUSH_ARM_SLIDE};

/// What one entry of the phase log is (`Game::npc_log`'s fourth field): a wild piece's step
/// along its route, `(from, to, eaten)`.
pub const PHASE_STEP: u8 = 0;
/// A wild piece coming back onto a start (generation 36), `(start, start, eaten)`.
pub const PHASE_RETURN: u8 = 1;
/// A turning-back piece that found its way blocked and turned round (generation 38):
/// `(from, to, eaten)` for its step the other way, `(square, square, 0)` where it waits.
pub const PHASE_TURN: u8 = 2;
/// A pouncer taking its prey (generation 39, N3b): `(from, to, eaten)`, the pouncer's square,
/// the victim's and the victim's byte. `unmake` puts it back as it puts back a step.
pub const PHASE_POUNCE: u8 = 3;
/// A splitting pouncer growing a copy of itself on an attacker's square (generation 41, N3d,
/// `docs/archive/plan-npc.md` §4.5): `(parent, square, eaten)`, the parent's square, the copy's and the
/// attacker's byte. `unmake` puts back only the eaten byte, as for a return.
pub const PHASE_COPY: u8 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    RoyalCaptured,
    RoyalDestroyed,
    Annihilated,
    Repetition,
    /// Q1069a: the material left is one of FIDE's dead sets. A draw.
    InsufficientMaterial,
    NoProgress,
    PlyLimit,
    Checkmate,
    Stalemate,
    StalemateLoss,
    StalemateWin,
    Reached,
    Extinction,
    BareRoyal,
    CheckLimit,
    /// Enough of the board was coloured.
    Coloured,
    /// The clock ran out and a scoring rule decided it. Distinct from `PlyLimit` and
    /// `NoProgress` on purpose: those two are draws and this one is a result, and a
    /// records file that called them the same thing would make every long game look
    /// drawn in the metrics.
    Scored,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::RoyalCaptured => "royal_captured",
            Reason::RoyalDestroyed => "royal_destroyed",
            Reason::Annihilated => "annihilated",
            Reason::Repetition => "repetition",
            Reason::InsufficientMaterial => "insufficient_material",
            Reason::NoProgress => "no_progress",
            Reason::PlyLimit => "ply_limit",
            Reason::Checkmate => "checkmate",
            Reason::Stalemate => "stalemate",
            Reason::StalemateLoss => "stalemate_loss",
            Reason::StalemateWin => "stalemate_win",
            Reason::Reached => "reached",
            Reason::Extinction => "extinction",
            Reason::BareRoyal => "bare_royal",
            Reason::CheckLimit => "check_limit",
            Reason::Coloured => "coloured",
            Reason::Scored => "scored",
        }
    }
}

/// `winner` is `-1` for a draw, matching `Outcome.winner is None` on the Python side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Outcome {
    pub winner: i8,
    pub reason: Reason,
}

impl Outcome {
    #[inline]
    pub fn score_for(&self, player: usize) -> f64 {
        if self.winner < 0 {
            0.5
        } else if self.winner as usize == player {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
struct Undo {
    origin: usize,
    target: usize,
    kind: u32,
    moving: u8,
    captured: u8,
    captured_square: usize,
    target_before: u8,
    rook: Option<(usize, usize, u8)>,
    unmoved: SquareSet,
    ep_target: i16,
    ep_victim: i16,
    halfmove: u32,
    key: u64,
    window: usize,
    checks: [u32; 2],
    /// `(player, piece type, delta)` where this move changed a hand, and `None` where it
    /// did not -- which is every move of every ruleset without drops.
    ///
    /// A whole clone of both hands used to live here, which is two heap allocations per
    /// move on the hottest path in the engine for crazyhouse, shogi and dobutsu. **At
    /// most one entry changes, by one**: a drop spends one piece from the mover's hand
    /// and a capture adds one to it, and the two cannot both happen, because a drop's
    /// target square is empty by construction. An explosion is not an exception -- the
    /// blast puts nothing in a hand.
    hand: Option<(u8, u8, i8)>,
    /// Squares cleared by an explosion or by a sweep's path (schema 13), to be restored
    /// on unmake. The two rules are refused together, so at most one writes it.
    cleared: Vec<(usize, u8)>,
    /// Moves left in the turn, and how many turns have been taken, before this move.
    remaining: u32,
    turns_taken: u32,
    /// `(square, colour before)` when this move changed a square's colour. One square
    /// at most: a move ends on one square, and that is the only one it can colour.
    coloured: Option<(usize, u8)>,
    /// Where this move's phase steps begin in `Game::npc_log` (schema 9).
    ///
    /// An index rather than a `Vec` of its own, and measured rather than tidied. C6 asked
    /// for the `cleared` shape -- a `Vec` per `Undo`, empty and allocation-free
    /// where the rule is off -- and that shape costs 24 bytes on a struct `make` writes at
    /// every node of every search, which chess paid for in perft while never reading it.
    /// One shared stack and one index is the `history`/`window` shape instead: four bytes
    /// here, and no allocation at all after the first phase of the first game.
    npc_log: u32,
    /// Schema 12: one more than the landing index of a survivor this move placed under
    /// `position: "choose"`, and 0 where it placed none. The fourth square a strike can
    /// touch, which `unmake` clears before the generic restore. A `u8` sits in the
    /// struct's one byte of padding, so `Undo` does not grow.
    placed: u8,
}

/// The squares a legality filter has temporarily overwritten, and what stood on them.
///
/// A filter plays a move on the board, asks one question and puts it back, so the whole
/// of its bookkeeping is a handful of `(square, value)` pairs -- and it ran once per
/// candidate move at every node, allocating a `Vec` each time.
///
/// **Five inline is not a guess.** Castling touches four squares; the deepest other path
/// is a capture that strikes, which is the origin and the target plus the three
/// `strike_writes` returns. `position: "choose"` (schema 12) is still three writes, its
/// third square the survivor's. An explosion's blast is bounded only by the board, so it
/// spills, and so does an en passant strike, whose victim's square is a sixth entry: rare,
/// correct, and not worth sixteen bytes on every filter call chess makes.
struct Saved {
    inline: [(usize, u8); Saved::INLINE],
    len: usize,
    spill: Vec<(usize, u8)>,
}

/// What a contact leaves behind: the squares to write, and the two things only `make`
/// needs to know about afterwards.
///
/// The legality filters want the writes and nothing else. `make` wants both of the rest,
/// and neither can be read back off the board: whether a struck piece is standing
/// somewhere (its `unmoved` bit has to be cleared, and the no-progress clock compares its
/// value against the one it replaced) and whether the *attacker* changed into something
/// of a different worth, which since schema 10 is a second way for the board's declared
/// material to move without a piece leaving it.
struct StrikeOutcome {
    /// The squares to write, in order; the first `count`. Three for every position but
    /// schema 17's `home`, whose reborn piece is a fourth square beside the three an
    /// advancing capture writes.
    writes: [(usize, u8); 4],
    count: usize,
    survivor: Option<(usize, u8)>,
    mover_material_changed: bool,
}

impl StrikeOutcome {
    #[inline]
    fn writes(&self) -> &[(usize, u8)] {
        &self.writes[..self.count]
    }
}

impl Saved {
    const INLINE: usize = 5;

    fn new() -> Self {
        Self { inline: [(0, 0); Self::INLINE], len: 0, spill: Vec::new() }
    }

    #[inline]
    fn push(&mut self, entry: (usize, u8)) {
        if self.len < Self::INLINE {
            self.inline[self.len] = entry;
            self.len += 1;
        } else {
            self.spill.push(entry);
        }
    }

    /// Put the board back, newest square first -- which is what the `Vec` this replaced
    /// did, and it matters wherever one square was written twice.
    #[inline]
    fn restore(&self, board: &mut [u8; MAX_SQUARES]) {
        for &(square, value) in self.spill.iter().rev() {
            board[square] = value;
        }
        for &(square, value) in self.inline[..self.len].iter().rev() {
            board[square] = value;
        }
    }
}

/// What a royal piece of each player's looks like on the board, or `0` where asking is
/// meaningless -- no royalty at all, or more than one royal per side, which is the case
/// `Game::royal_hint` is not sound for.
fn royal_values(t: &Tables) -> [u8; 2] {
    match t.royal_types.first() {
        Some(piece_type) if !t.multi_royal => {
            // A royal may not declare a facing -- the validator refuses it by name --
            // so a royal type is one board value per player and this hint stays sound.
            [t.entering_value(0, *piece_type), t.entering_value(1, *piece_type)]
        }
        _ => [0, 0],
    }
}

/// Tally a colouring. The slow definition `colour_counts` is maintained against, and
/// the one every entry point into a position that was not reached by a move uses.
pub fn count_colours(colour: &[u8]) -> [u32; 2] {
    let mut counts = [0u32; 2];
    for square in colour {
        if *square != 0 {
            counts[*square as usize - 1] += 1;
        }
    }
    counts
}

/// Which squares each owner's pieces stand on: the slow definition `Game::occupied`
/// is maintained against, and what a position that was not reached by a move starts
/// from.
///
/// **Three sets, not two, since schema 9.** The third is the wild pieces', and it is
/// there because `write` and `put` index this by `owner_of` -- a board that can hold a
/// third owner's byte needs a set for it or the two writers panic. Nothing that means
/// *the two armies* reads it: `all_occupied` is still the players' union, which is what
/// keeps every material term blind to a piece nobody owns.
pub fn scan_occupied(t: &Tables, board: &[u8; MAX_SQUARES]) -> [SquareSet; 3] {
    let mut occupied = [SquareSet::EMPTY; 3];
    for (square, value) in board.iter().enumerate().take(t.n_squares) {
        if *value != 0 {
            occupied[t.owner_of(*value)].set(square);
        }
    }
    occupied
}

/// A position plus the history a repetition rule needs, mutated in place.
pub struct Game<'t> {
    pub t: &'t Tables,
    pub board: [u8; MAX_SQUARES],
    pub side: usize,
    pub unmoved: SquareSet,
    pub ep_target: i16,
    pub ep_victim: i16,
    pub halfmove: u32,
    pub ply: u32,
    pub key: u64,
    /// Every position hash of the game so far. `window` is the index of the first one
    /// that survived the last irreversible move, so `history[window..]` is exactly the
    /// `repetition` tuple the Python state carries.
    pub history: Vec<u64>,
    pub window: usize,
    /// Checks delivered so far by each player (schema 2: check_limit).
    pub checks: [u32; 2],
    /// Pieces held off the board, per player, counted by piece type (schema 2: drops).
    pub hands: [Vec<u32>; 2],
    /// Moves left in the current turn. 1 for every ruleset without a turn rule, and
    /// the side flips whenever it reaches zero -- which is what `make` used to do
    /// unconditionally.
    pub remaining: u32,
    /// How many turns have been taken, so the opening ramp knows where it is.
    pub turns_taken: u32,
    /// Who owns each square, 0 for nobody (schema 4: territory). Empty when the rule is
    /// off, so a ruleset without it pays one `is_empty` and no memory.
    pub colour: Vec<u8>,
    /// How many squares each side has claimed, maintained alongside `colour`.
    ///
    /// Counted rather than scanned because three things now ask per node -- the win
    /// condition, the evaluator's goal term, and the search's table key -- and a scan
    /// is one pass over the board for each. A move claims at most one square, so the
    /// delta is two increments and `unmake` reverses them.
    colour_counts: [u32; 2],
    /// Which squares each player's pieces stand on, maintained alongside `board`.
    ///
    /// Move generation used to find the mover's pieces by reading every square -- 256
    /// reads on the largest board for perhaps twenty pieces -- and the evaluator, the
    /// terminal counts and the drop generator each walked the board again for the same
    /// answer. A set per player is one word per 64 squares, and walking it in ascending
    /// order is exactly the order the scan visited, which is the one thing that must
    /// not change: piece order is move order is tie-break order.
    ///
    /// **State, not a hint**, and so it has to be right everywhere the board changes.
    /// `write` and `put` are the two places, one for `make` and one for `unmake`, and
    /// nothing else writes a square except the legality filters -- which play a move,
    /// ask about attacks, and put the board back without ever generating a move or
    /// counting anything. `occupied()` refuses to answer while one of those has the
    /// board mid-experiment, so a later caller that asks from inside a filter fails a
    /// debug build rather than reading a set that describes the board a moment ago.
    occupied: [SquareSet; 3],
    /// What each player has been paid for taking wild pieces (schema 9).
    ///
    /// `scoring.count: "points"` compares the two at the clocks. **No Zobrist term**: a
    /// bounty is awarded only when a wild piece is taken, which is a material change,
    /// which is irreversible, which clears the repetition window -- so two positions the
    /// repetition rule ever compares always share the tally. Exactly the `hands`
    /// argument and the permanent-`colour` argument, with the same transposition-table
    /// caveat all three carry.
    pub points: [f64; 2],
    /// Schema 18 (Q786, `reach.on: "leave"`): the side whose last move took the named
    /// piece off one of its goal squares, or -1. Nothing on the board says what the last
    /// move was, so the ending cannot read it there; `make` sets it, `unmake` restores it
    /// from `Undo`, and it rides on `Snapshot` and across the boundary.
    ///
    /// **Not in the key.** A position where it is set is terminal (`schema_two_ending`
    /// reads it in reach's slot), so no search and no repetition rule ever compares it
    /// with another position. -1 for every ruleset without the rule, where `make` never
    /// writes it.
    pub goal_left_by: i8,
    /// `goal_left_by` before each move still on the undo stack, oldest first: written by
    /// `make` and read back by `unmake` only under the rule (a terminal position can
    /// still be made from, since perft walks through one).
    ///
    /// **A stack of its own rather than a byte on `Undo`**, which `make` writes at every
    /// node of every search for games that would never read it. This is `npc_log`'s
    /// shape: empty and never touched where the rule is off, so those games pay one flag
    /// test in `make` and one in `unmake`. (Q786 benched both; they were alike, and both
    /// within what a build that changes nothing on the hot path moves, see the record.)
    goal_left_log: Vec<i8>,
    /// Schema 20 (Q807, `no_straight_back`): the portal end standing on `portal_bar[0]` was
    /// moved there from `portal_bar[1]` by the last move, and may not go straight back on
    /// this one; `[-1, -1]` for no bar. Nothing on the board says what the last move was,
    /// so it is state: `make` sets it on such an end's own move and clears it on every
    /// other, `unmake` restores it from `portal_bar_log`, and it rides on `Snapshot` and
    /// across the boundary. Q883: set only where it can bar the next move -- the compiler's
    /// `no_back` (false for a bar no move can ever meet) and the next mover among the end's
    /// `movers` (`PortalTables::sets_bar`) -- so a bar that can bar nothing is never keyed.
    ///
    /// **In the key** (`PortalTables::bar_key`), as the en passant file is: two positions
    /// alike but for a bar have different legal moves, so the repetition rule and a
    /// transposition table must tell them apart. `[-1, -1]` for every game without the
    /// rule, where `make` never writes it.
    pub portal_bar: [i16; 2],
    /// `portal_bar` before each move still on the undo stack, oldest first: `goal_left_log`'s
    /// shape, empty and never touched without the rule.
    portal_bar_log: Vec<[i16; 2]>,
    /// Schema 22 (Q843c): where each fold end stands, by end (`PortalTables`' end index),
    /// and -1 for a momentum end and in every game without folds. A fold stands on no
    /// square of the board byte -- a piece may stand in it -- so this is state of its own:
    /// `make` moves one on its `MOVE_FOLD`, `unmake` moves it back from the move's origin,
    /// it rides on `Snapshot` and across the boundary (`GameState.folds`), and it is **in
    /// the key** (`FoldTables::key`).
    pub fold_at: [i16; MAX_PORTAL_ENDS],
    /// Where each player's royal stands: a *hint*, not state.
    ///
    /// A hint because it is validated rather than invalidated -- `royal_square` checks
    /// that the remembered square still holds that player's royal and rescans when it
    /// does not. That is what makes it safe inside the two filters that play a move on
    /// the board, ask a question and put it back: `leaves_royal_attacked` and
    /// `restrict` would each have to maintain a field that had to be trusted, and
    /// getting one of them wrong is the kind of thing no perft can see.
    ///
    /// Sound only while a player has at most one royal, which is exactly what
    /// `multi_royal` is false for -- the compiler's own note on that field says the
    /// single-royal case can answer "where is it?" with one `find`. With two, a square
    /// holding a royal need not be the *first* one, and the scan's order is observable.
    /// `-1` is "not looked up yet" rather than "no royal": a missing royal is a real
    /// answer and is remembered like any other.
    royal_hint: Cell<[i32; 2]>,
    /// What a royal of each player's looks like on the board, or `0` where the hint
    /// does not apply. Resolved once rather than per call: `t.royal_types.first()` is a
    /// pointer into the tables and a bounds check, and validating the hint is on the
    /// movegen path, where that is the same order of cost as the scan it replaces.
    royal_value: [u8; 2],
    /// "Is this side in check", remembered for the position now on the board. `-1`
    /// unknown, `0` no, `1` yes.
    ///
    /// Pure memoisation of a function of the position, so it is invalidated wherever
    /// the position moves -- `make`, `unmake` and `restore` -- and is not consulted at
    /// all while `scratch` says the board is mid-experiment.
    check_memo: Cell<[i8; 2]>,
    /// How many legality filters currently have the board temporarily mutated.
    ///
    /// The completeness half of the memo above. Nothing in those filters asks
    /// `in_check` today, so the memo would be correct without this -- but "correct
    /// because no caller happens to ask" is an invariant a later edit breaks silently,
    /// and what it breaks is a cached answer about a position no game was ever in.
    scratch: u32,
    undo: Vec<Undo>,
    /// Every phase step of every move still on the undo stack, oldest first (schema 9).
    ///
    /// One stack for the whole game rather than one `Vec` per move, for the reason
    /// `history` is one and the repetition window is a pair of indices into it: a move's
    /// own steps are everything above the index its `Undo` recorded, and the two are the
    /// same LIFO. Empty and never touched where nothing is wild.
    npc_log: Vec<(usize, usize, u8, u8)>,
    /// Generation 36 (N1): one clock per respawning half (`Tables::respawn`), 0 idle, else
    /// the phases until a return is tried. Empty without the rule. In the key.
    pub respawn: Vec<u8>,
    /// `respawn` before each move still on the undo stack, oldest first, flat: written by
    /// `make` only under the rule, `goal_left_log`'s shape.
    respawn_log: Vec<u8>,
    /// Generation 38 (N5): the squares of the wild pieces walking their line back. Empty
    /// without the rule. In the key.
    pub heading_back: SquareSet,
    /// `heading_back` before each move still on the undo stack, oldest first: written by
    /// `make` only under the rule.
    heading_log: Vec<SquareSet>,
    /// TC4 (`docs/plan-turncoat.md` §5.2): each player's coins, 0 to `PURSE_MAX`. `[0, 0]`
    /// in every game without a purse. In the key (`PurseTables::key`).
    pub purse: [u16; 2],
    /// `purse` before each move still on the undo stack, oldest first: written by `make`
    /// only under the rule, `rest_log`'s shape, so `Undo` does not grow. A log and not a
    /// read-back, because the cap makes a payment lossy: 98 and a bounty of 3 is 99, and
    /// 99 less 3 is not 98.
    purse_log: Vec<[u16; 2]>,
    /// Q1074b6 (K5, `rules.push.no_push_back`, `docs/plan-push.md` §3.12): the piece standing
    /// on `push_bar[0]` was pushed there from `push_bar[1]` by the last move, and no push on
    /// this one may carry it back onto `push_bar[1]`, as the mover or in the line (Q1078);
    /// `NO_BAR` for none. Set by `make` on a push that moved its target's piece and did not
    /// remove it, whoever owns that piece (`push::push_bar_after`), cleared by every other move
    /// of `make`'s (a duck half leaves it, being no piece move). Restored from
    /// `push_bar_log` only under the rule; `portal_bar`'s shape.
    ///
    /// **In the key** (`PushTables::bar_key`): two positions alike but for a bar have
    /// different legal moves, so the repetition rule and a transposition table tell them
    /// apart. `NO_BAR` in every game without the rule, where `make` never writes it.
    pub push_bar: [i16; 2],
    /// `push_bar` before each move still on the undo stack, oldest first: empty and never
    /// touched without the rule.
    push_bar_log: Vec<[i16; 2]>,
    /// Guard against infinite recursion in drop-mate checking.
    checking_drop_mate: bool,
    /// The key the move in progress is building. Scratch, not state: seeded from `key`
    /// at the top of every `make` and installed at the bottom. Separate from `key`
    /// because `Undo` stores the key as it was *before* the move and is pushed last.
    working_key: u64,
    /// Schema 26 (Q918a): who holds the initiative (§3, E), or -1 where the tie is not
    /// the initiative. In the key while player 1 holds it (`SimTables::zobrist_initiative`).
    pub initiative: i8,
    /// Schema 26: the void run (§2.7), the pair `[player 0's square, player 1's square]`
    /// whose one void round was the last round, or `NO_BAR`. In the key.
    pub void_run: [i16; 2],
    /// Schema 26: the barred pairs (§2.7), ascending; empty in almost every position. In
    /// the key.
    pub bars: Vec<[i16; 2]>,
    /// Schema 29 (Q983a, `pieces[].rest`, `docs/plan-rest.md` §4.2): each player's resting
    /// pieces, `(square, turns left, the code that rested there)`, written by `make`,
    /// `make_fold` and `make_round` at the end of each player's turn (`rest_after_turn`),
    /// read by both generators (`rest_filter`). `NO_REST` in every game without the rule. In
    /// the key (`RestTables::key`).
    pub resting: RestLists,
    /// `resting` before each move still on the undo stack, oldest first: `goal_left_log`'s
    /// shape, empty and never touched without the rule, so `Undo` does not grow.
    rest_log: Vec<RestLists>,
    /// Q1074b4 (`docs/plan-push.md` K2): the direction of each `MOVE_PUSH` still on the undo
    /// stack, oldest first, under `chain: "row"` or `"weight"` only (a line of one needs
    /// none), for `unmake` to walk a row's middle back (`push_unshift`). Empty and never
    /// touched otherwise, so `Undo` does not grow.
    push_log: Vec<u8>,
    /// Schema 26: one `RoundUndo` per round still on the stack.
    round_undo: Vec<rounds::RoundUndo>,
    /// Schema 26: what the last round did, `(event, square, player)` (`rounds::EVENT_*`),
    /// for the boundary to hand the page. Scratch, like `npc_log`.
    pub round_log: Vec<(u8, usize, i8)>,
    /// Schema 27 (Q935a): the initiative before each `spy_turn` still on the stack, for
    /// `unmake_spy_turn` to put back what the deal (E10) wrote. Empty without a spy.
    spy_deals: Vec<i8>,
}

/// Everything needed to put a `Game` back where it was, for the tree search that
/// cannot use the undo stack because it revisits nodes out of order.
#[derive(Clone)]
pub struct Snapshot {
    pub board: [u8; MAX_SQUARES],
    pub side: usize,
    pub unmoved: SquareSet,
    pub ep_target: i16,
    pub ep_victim: i16,
    pub halfmove: u32,
    pub ply: u32,
    pub key: u64,
    pub repetition: Vec<u64>,
    pub checks: [u32; 2],
    pub hands: [Vec<u32>; 2],
    pub colour: Vec<u8>,
    pub remaining: u32,
    pub turns_taken: u32,
    /// The two maintained tallies, carried so that `restore` copies them rather than
    /// recounting the board -- which it did on every descent step of a tree search.
    /// A snapshot taken from a `Game` copies its tallies; one built from a Python
    /// state computes them once, at the boundary.
    pub occupied: [SquareSet; 3],
    pub colour_counts: [u32; 2],
    /// The bounty tally, carried like every other maintained quantity.
    pub points: [f64; 2],
    /// Schema 18 (Q786): the side that has just left a goal square, or -1.
    pub goal_left_by: i8,
    /// Schema 20 (Q807): the straight-back bar, `[-1, -1]` for none.
    pub portal_bar: [i16; 2],
    /// Schema 22 (Q843c): where each fold end stands, -1 for none.
    pub fold_at: [i16; MAX_PORTAL_ENDS],
    /// Schema 26 (Q918a): the initiative, the void run and the bars.
    pub initiative: i8,
    pub void_run: [i16; 2],
    pub bars: Vec<[i16; 2]>,
    /// Schema 29 (Q983a): the resting pieces.
    pub resting: RestLists,
    /// Generation 36 (N1): the respawn clocks.
    pub respawn: Vec<u8>,
    /// Generation 38 (N5): the squares of the wild pieces walking their line back.
    pub heading_back: SquareSet,
    /// TC4: each player's coins.
    pub purse: [u16; 2],
    /// Q1074b6 (K5): the push-back bar, `NO_BAR` for none.
    pub push_bar: [i16; 2],
}

/// Schema 26: who holds the initiative at the start, or -1 where the tie is not it.
fn initiative_start(t: &Tables) -> i8 {
    t.simultaneous.as_deref().filter(|sim| sim.tie == 1).map_or(-1, |sim| sim.initiative_start)
}

/// Schema 22: where the folds start, from the tables; all -1 without folds.
fn fold_starts(t: &Tables) -> [i16; MAX_PORTAL_ENDS] {
    t.portals
        .as_deref()
        .and_then(|portals| portals.folds.as_ref())
        .map_or([-1; MAX_PORTAL_ENDS], |folds| folds.starts)
}

impl<'t> Game<'t> {
    pub fn initial(t: &'t Tables) -> Self {
        // Schema 10 (Q51): a hand may start with something in it. `stock` is empty for
        // every ruleset written before that, so both hands start at zero exactly as they
        // did -- and both players get the same stock, because a supply one side has and
        // the other does not is a first-seat effect rather than a rule.
        let hands = match t.drops.as_ref() {
            Some(rule) if !rule.stock.is_empty() => [rule.stock.clone(), rule.stock.clone()],
            Some(_) => [vec![0u32; t.n_types], vec![0u32; t.n_types]],
            None => [vec![], vec![]],
        };
        let mut game = Self {
            t,
            board: t.initial_board,
            side: 0,
            unmoved: t.initial_unmoved,
            ep_target: NO_SQUARE,
            ep_victim: NO_SQUARE,
            halfmove: 0,
            ply: 0,
            key: 0,
            history: Vec::with_capacity(64),
            window: 0,
            checks: [0, 0],
            hands,
            remaining: t.turn.as_ref().map_or(1, |rule| rule.length_of(0)),
            turns_taken: 0,
            colour: if t.territory.is_some() {
                vec![0u8; t.n_squares]
            } else {
                Vec::new()
            },
            colour_counts: [0, 0],
            occupied: scan_occupied(t, &t.initial_board),
            points: [0.0, 0.0],
            goal_left_by: -1,
            portal_bar: NO_BAR,
            portal_bar_log: Vec::new(),
            fold_at: fold_starts(t),
            royal_hint: Cell::new([-1, -1]),
            royal_value: royal_values(t),
            check_memo: Cell::new([-1, -1]),
            scratch: 0,
            undo: Vec::with_capacity(64),
            npc_log: Vec::new(),
            respawn: t.respawn.as_deref().map_or(Vec::new(), |rule| vec![0; rule.halves.len()]),
            respawn_log: Vec::new(),
            heading_back: SquareSet::EMPTY,
            heading_log: Vec::new(),
            purse: t.purse.as_deref().map_or([0, 0], |purse| [purse.start, purse.start]),
            purse_log: Vec::new(),
            push_bar: NO_BAR,
            push_bar_log: Vec::new(),
            goal_left_log: Vec::new(),
            checking_drop_mate: false,
            working_key: 0,
            initiative: initiative_start(t),
            void_run: NO_BAR,
            bars: Vec::new(),
            resting: NO_REST,
            rest_log: Vec::new(),
            push_log: Vec::new(),
            round_undo: Vec::new(),
            round_log: Vec::new(),
            spy_deals: Vec::new(),
        };
        game.key = game.compute_hash();
        game.history.push(game.key);
        game
    }

    pub fn from_snapshot(t: &'t Tables, snapshot: &Snapshot) -> Self {
        debug_assert_eq!(snapshot.occupied, scan_occupied(t, &snapshot.board));
        debug_assert_eq!(snapshot.colour_counts, count_colours(&snapshot.colour));
        Self {
            t,
            board: snapshot.board,
            side: snapshot.side,
            unmoved: snapshot.unmoved,
            ep_target: snapshot.ep_target,
            ep_victim: snapshot.ep_victim,
            halfmove: snapshot.halfmove,
            ply: snapshot.ply,
            key: snapshot.key,
            history: snapshot.repetition.clone(),
            window: 0,
            checks: snapshot.checks,
            hands: snapshot.hands.clone(),
            remaining: snapshot.remaining.max(1),
            turns_taken: snapshot.turns_taken,
            // Sized here rather than trusted from the caller. A position built by hand
            // in Python may carry no colouring at all, and every read of this is a
            // per-square index -- "nothing has been claimed" is the well-defined
            // reading of an absent array, and a panic is not.
            colour: if t.territory.is_some() {
                let mut colour = snapshot.colour.clone();
                colour.resize(t.n_squares, 0);
                colour
            } else {
                Vec::new()
            },
            colour_counts: snapshot.colour_counts,
            occupied: snapshot.occupied,
            points: snapshot.points,
            goal_left_by: snapshot.goal_left_by,
            portal_bar: snapshot.portal_bar,
            portal_bar_log: Vec::new(),
            fold_at: snapshot.fold_at,
            royal_hint: Cell::new([-1, -1]),
            royal_value: royal_values(t),
            check_memo: Cell::new([-1, -1]),
            scratch: 0,
            undo: Vec::with_capacity(64),
            npc_log: Vec::new(),
            respawn: snapshot.respawn.clone(),
            respawn_log: Vec::new(),
            heading_back: snapshot.heading_back,
            heading_log: Vec::new(),
            purse: snapshot.purse,
            purse_log: Vec::new(),
            push_bar: snapshot.push_bar,
            push_bar_log: Vec::new(),
            goal_left_log: Vec::new(),
            checking_drop_mate: false,
            working_key: 0,
            initiative: snapshot.initiative,
            void_run: snapshot.void_run,
            bars: snapshot.bars.clone(),
            resting: snapshot.resting,
            rest_log: Vec::new(),
            push_log: Vec::new(),
            round_undo: Vec::new(),
            round_log: Vec::new(),
            spy_deals: Vec::new(),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            board: self.board,
            side: self.side,
            unmoved: self.unmoved,
            ep_target: self.ep_target,
            ep_victim: self.ep_victim,
            halfmove: self.halfmove,
            ply: self.ply,
            key: self.key,
            repetition: self.history[self.window..].to_vec(),
            checks: self.checks,
            hands: self.hands.clone(),
            colour: self.colour.clone(),
            remaining: self.remaining,
            turns_taken: self.turns_taken,
            occupied: self.occupied,
            colour_counts: self.colour_counts,
            points: self.points,
            goal_left_by: self.goal_left_by,
            portal_bar: self.portal_bar,
            fold_at: self.fold_at,
            initiative: self.initiative,
            void_run: self.void_run,
            bars: self.bars.clone(),
            resting: self.resting,
            respawn: self.respawn.clone(),
            heading_back: self.heading_back,
            purse: self.purse,
            push_bar: self.push_bar,
        }
    }

    /// Reset this game to a snapshot without reallocating. The undo stack is dropped:
    /// a caller that restores has abandoned whatever it was in the middle of.
    ///
    /// Without allocating either: the tree search calls this once per descent step,
    /// and it used to clone both hands -- two heap allocations -- and recount the
    /// colouring with a pass over the board. `clone_from` reuses the vectors already
    /// here, and the two tallies ride on the snapshot.
    pub fn restore(&mut self, snapshot: &Snapshot) {
        self.board = snapshot.board;
        self.side = snapshot.side;
        self.unmoved = snapshot.unmoved;
        self.ep_target = snapshot.ep_target;
        self.ep_victim = snapshot.ep_victim;
        self.halfmove = snapshot.halfmove;
        self.ply = snapshot.ply;
        self.key = snapshot.key;
        self.history.clear();
        self.history.extend_from_slice(&snapshot.repetition);
        self.window = 0;
        self.checks = snapshot.checks;
        self.hands[0].clone_from(&snapshot.hands[0]);
        self.hands[1].clone_from(&snapshot.hands[1]);
        self.remaining = snapshot.remaining.max(1);
        self.turns_taken = snapshot.turns_taken;
        self.colour.clear();
        self.colour.extend_from_slice(&snapshot.colour);
        if self.t.territory.is_some() {
            self.colour.resize(self.t.n_squares, 0);
        }
        debug_assert_eq!(snapshot.colour_counts, count_colours(&snapshot.colour));
        debug_assert_eq!(snapshot.occupied, scan_occupied(self.t, &snapshot.board));
        self.colour_counts = snapshot.colour_counts;
        self.occupied = snapshot.occupied;
        self.points = snapshot.points;
        self.goal_left_by = snapshot.goal_left_by;
        self.portal_bar = snapshot.portal_bar;
        self.fold_at = snapshot.fold_at;
        self.initiative = snapshot.initiative;
        self.void_run = snapshot.void_run;
        self.bars.clone_from(&snapshot.bars);
        self.resting = snapshot.resting;
        self.rest_log.clear();
        self.push_log.clear();
        self.respawn.clone_from(&snapshot.respawn);
        self.respawn_log.clear();
        self.heading_back = snapshot.heading_back;
        self.heading_log.clear();
        self.purse = snapshot.purse;
        self.purse_log.clear();
        self.push_bar = snapshot.push_bar;
        self.push_bar_log.clear();
        self.round_undo.clear();
        self.round_log.clear();
        // The check bits describe the position that just went away. The royal hint is
        // not cleared here and does not need to be: it is checked against the board
        // before it is believed.
        self.check_memo.set([-1, -1]);
        self.undo.clear();
        // The phase log is scratch belonging to the undo stack, and the stack is gone.
        self.npc_log.clear();
        self.goal_left_log.clear();
        self.portal_bar_log.clear();
    }

    /// Which squares each player's pieces stand on, for a board that is settled.
    ///
    /// Settled means no legality filter has it mid-experiment: the filters write the
    /// board directly and put it back, and the sets are not told, so an answer read
    /// from inside one would describe the board a moment ago. Nothing asks from there
    /// today -- the filters only ask about attacks -- and this is what keeps that a
    /// checked fact rather than a happy accident. A debug build also re-derives the
    /// sets from the board on every read and compares.
    #[inline(always)]
    pub fn occupied(&self) -> &[SquareSet; 3] {
        debug_assert_eq!(self.scratch, 0, "occupancy read while a filter has the board");
        debug_assert_eq!(self.occupied, scan_occupied(self.t, &self.board));
        &self.occupied
    }

    /// Both players' pieces, as one set. Walked in ascending order this is board
    /// order over the occupied squares, which is the order every material sum adds
    /// its terms in.
    ///
    /// **The two armies, and deliberately not the board.** A wild piece is nobody's, so
    /// it is not here -- every caller is a material sum or a feature over the pieces
    /// somebody owns, and each decodes an owner out of the board byte by dividing by
    /// `side_stride`, which has no row for a third owner. `all_pieces` is the one that
    /// means "every square with something on it".
    #[inline(always)]
    pub fn all_occupied(&self) -> SquareSet {
        let occupied = self.occupied();
        occupied[0].union(&occupied[1])
    }

    /// Every square with a piece on it, whoever owns it -- the wild ones included.
    ///
    /// What "empty" means to a rule that puts a piece somewhere: a drop may not land on
    /// a boar. Identical to `all_occupied` for every ruleset before schema 9, because
    /// the third set is empty there.
    #[inline(always)]
    pub fn all_pieces(&self) -> SquareSet {
        let occupied = self.occupied();
        occupied[0].union(&occupied[1]).union(&occupied[WILD])
    }

    pub fn compute_hash(&self) -> u64 {
        let t = self.t;
        let mut digest = 0u64;
        for square in 0..t.n_squares {
            let value = self.board[square];
            if value != 0 {
                digest ^= t.zobrist(value, square);
            }
        }
        if self.side != 0 {
            digest ^= t.zobrist_side;
        }
        if self.ep_target != NO_SQUARE {
            digest ^= t.zobrist_ep_file(self.ep_target);
        }
        // Only a *repainting* colouring enters the key. A permanent one is monotone, so
        // laying one down is irreversible, so it clears the repetition window, so two
        // positions the repetition rule ever compares always share the same colouring --
        // exactly the argument that keeps `hands` out of the key.
        if t.territory.as_ref().is_some_and(|rule| rule.repaints) {
            for square in 0..t.n_squares {
                digest ^= t.zobrist_colour(self.colour[square], square);
            }
        }
        // Two positions identical but for how many moves are left in the turn are not
        // the same position: one side is about to move again and the other is not.
        digest ^= t.zobrist_remaining(self.remaining);
        // Schema 20: the straight-back bar, zero without one and without the rule.
        if let Some(portals) = t.portals.as_deref() {
            digest ^= portals.bar_key(t.n_squares, self.portal_bar);
            // Schema 22: where each fold stands.
            if let Some(folds) = portals.folds.as_ref() {
                for (end, square) in self.fold_at.iter().enumerate().take(portals.n_ends) {
                    if *square >= 0 {
                        digest ^= folds.key(t.n_squares, end, *square as usize);
                    }
                }
            }
        }
        // Schema 26: the initiative, the void run and the bars; zero in an alternating game.
        digest ^= self.standoff_key();
        // Schema 29: the resting pieces; zero in a game without a rest.
        digest ^= self.rest_key();
        // Generation 36: the respawn clocks; zero in a game without the rule.
        digest ^= self.respawn_key();
        // Generation 38: the headings; zero in a game without the rule.
        digest ^= self.heading_key();
        // TC4: the purse; zero in a game without one.
        if let Some(purse) = t.purse.as_deref() {
            digest ^= purse.key(0, self.purse[0]) ^ purse.key(1, self.purse[1]);
        }
        // Q1074b6 (K5): the push-back bar; zero without one and without the rule.
        if let Some(push) = t.push.as_deref() {
            digest ^= push.bar_key(t.n_squares, self.push_bar);
        }
        digest
    }

    /// Where `player`'s royal stands, or `-1`.
    ///
    /// A linear scan over the board, remembered. It is asked from `in_check`, from
    /// `terminal_prefix`, from the check filter once per *suspect move*, from the
    /// `forbid_check` filter and from the drop-mate test, so the scan was several
    /// passes over every square at every node of every checkmate ruleset -- and it
    /// grows with the board, which is where the searches are already shallowest.
    ///
    /// The hint is validated against the board rather than maintained in `make`, which
    /// is what makes it safe inside the filters that mutate the board and put it back.
    /// See `royal_hint` for why it is only sound while `multi_royal` is false.
    #[inline]
    pub fn royal_square(&self, player: usize) -> i32 {
        let t = self.t;
        if t.multi_royal {
            return self.scan_royal_square(player);
        }
        let hint = self.royal_hint.get()[player];
        if hint >= 0 && self.holds_royal(player, hint as usize) {
            debug_assert_eq!(hint, self.scan_royal_square(player));
            return hint;
        }
        let found = self.scan_royal_square(player);
        // Not remembered from inside a filter's experiment: that answer is about a
        // board on its way back to what it was, so storing it would trade a scan saved
        // for a scan wasted on the next call.
        if self.scratch == 0 {
            let mut hints = self.royal_hint.get();
            hints[player] = found;
            self.royal_hint.set(hints);
        }
        found
    }

    /// Whether `square` holds a royal piece of `player`'s. One load and one compare:
    /// `multi_royal` being false means there is at most one royal *type*, and the value
    /// it encodes to was resolved when the game was built.
    #[inline]
    fn holds_royal(&self, player: usize, square: usize) -> bool {
        let value = self.royal_value[player];
        value != 0 && self.board[square] == value
    }

    /// The scan the hint above is a memo of, and the definition it is checked against.
    fn scan_royal_square(&self, player: usize) -> i32 {
        let t = self.t;
        for piece_type in &t.royal_types {
            let value = t.entering_value(player, *piece_type);
            for square in 0..t.n_squares {
                if self.board[square] == value {
                    return square as i32;
                }
            }
        }
        -1
    }

    /// How many royals `player` still has. One board read on the single-royal path,
    /// through the validated hint; a scan where a side can have several.
    #[inline]
    fn royal_count(&self, player: usize) -> usize {
        if self.t.multi_royal {
            self.royal_squares(player).len()
        } else {
            usize::from(self.royal_square(player) >= 0)
        }
    }

    /// Whether `player` still holds a royal in hand (schema 18, Q784).
    ///
    /// Under `drops.placement` a royal may start in the stock, so a royal off the board is
    /// either *not yet* played or *gone*, and only the second loses. `royal_stocked` is
    /// asked first, so a ruleset whose stock holds no royal -- every one before schema
    /// 18 -- never reads a hand here. Every caller has already found the board without
    /// a royal on it, which is the rare branch.
    #[inline]
    pub fn royal_in_hand(&self, player: usize) -> bool {
        let t = self.t;
        match t.drops.as_ref() {
            Some(rule) if rule.royal_stocked => t
                .royal_types
                .iter()
                .any(|piece_type| self.hands[player].get(*piece_type).is_some_and(|n| *n > 0)),
            _ => false,
        }
    }

    /// Whether `player`'s royal is off the board for good: not standing, and not waiting
    /// in hand. The one reading of a missing royal that `royal_loss_ends_it` pairs with.
    #[inline]
    fn royal_gone(&self, player: usize) -> bool {
        self.royal_square(player) < 0 && !self.royal_in_hand(player)
    }

    /// Every royal square `player` still holds. Only called when there can be several.
    pub fn royal_squares(&self, player: usize) -> Vec<usize> {
        let t = self.t;
        let mut squares = Vec::new();
        for piece_type in &t.royal_types {
            let value = t.entering_value(player, *piece_type);
            for square in 0..t.n_squares {
                if self.board[square] == value {
                    squares.push(square);
                }
            }
        }
        squares
    }

    fn has_any_piece(&self, player: usize) -> bool {
        !self.occupied()[player].is_empty()
    }

    /// Whether the two royal pieces see each other down an otherwise empty file.
    /// Xiangqi's flying general.
    ///
    /// **Reads a table; computes nothing.** This function used to walk a file by
    /// `+= width`, which made it the third and last place in the core that worked a
    /// square out by arithmetic -- blind to a seam, because the line can run the other
    /// way round, and blind to a wall, because a wall is not a piece standing anywhere.
    /// It was refused beside both rather than fixed. `Tables::facing_lines` answers the
    /// question the compiler was always able to answer: for this square, which squares
    /// are at the far end of a file line, and what has to be empty in between.
    ///
    /// Joined ranks put the same far end in the list twice, one entry each way round,
    /// so the loop is over *lines* rather than over pairs and the first open one wins.
    fn royals_face(&self) -> bool {
        let first = self.royal_square(0);
        let second = self.royal_square(1);
        if first < 0 || second < 0 {
            return false;
        }
        let second = second as u16;
        let t = self.t;
        for &(other, start, length) in t.facing_from(first as usize) {
            if other != second {
                continue;
            }
            if t.facing_path(start, length)
                .iter()
                .all(|&square| self.board[square as usize] == 0)
            {
                return true;
            }
        }
        false
    }

    /// Whether `player` attacks `square` with any capture-capable move.
    ///
    /// Reads the inverted table: a flat list of the ways this square can be reached,
    /// each rejected on one byte comparison before any path work happens.
    /// Whether `player` could capture whatever stands on `square`.
    ///
    /// An empty square has no victim to consult the capture matrix about, so it is
    /// "attacked" if anything reaches it at all. Every caller that means a *specific*
    /// piece -- castling's transit squares are the only ones -- calls
    /// `attacks_square_as` with that piece's type instead.
    pub fn attacks_square(&self, player: usize, square: usize) -> bool {
        // The victim's type is only worth reading when something consults it. This is
        // the hottest query in the engine -- the check filter asks it once per candidate
        // move -- and the division is not free.
        if !self.t.has_capture_matrix {
            return self.attacks_square_as(player, square, usize::MAX);
        }
        let occupant = self.board[square];
        let victim = if occupant == 0 {
            usize::MAX
        } else {
            self.t.type_of(occupant)
        };
        self.attacks_square_as(player, square, victim)
    }

    /// Whether `player` could capture a piece of `victim` type standing on `square`.
    ///
    /// `usize::MAX` means "any piece", which skips the capture-matrix filter.
    pub fn attacks_square_as(&self, player: usize, square: usize, victim: usize) -> bool {
        let t = self.t;
        for source in t.attackers_of(player, square) {
            if self.board[source.origin as usize] != source.piece_value {
                continue;
            }
            if source.requires_unmoved && !self.unmoved.get(source.origin as usize) {
                continue;
            }
            // Schema 3: an attacker that is not allowed to take this victim is not
            // attacking it. The attacker's type is already here, encoded in the value
            // the source matched against.
            if t.has_capture_matrix && victim != usize::MAX {
                let attacker = t.type_of(source.piece_value);
                // Q1074b2 (R1): in a push game a source attacks by capture, by push or both
                // (its `dirs` byte), each under its own list. A push game always crosses the
                // capture matrix, so this is the branch it takes.
                let reaches = match t.push.as_deref() {
                    None => t.may_capture(attacker, victim),
                    Some(push) => self.push_game_attacks(push, source.dirs, attacker, victim),
                };
                if !reaches {
                    continue;
                }
            }
            if source.sweep != 0 {
                if self.sweep_attacks(player, source) {
                    return true;
                }
                continue;
            }
            if self.source_path_clear(player, source) {
                return true;
            }
        }
        false
    }

    /// Whether `source`'s path lets it reach its square: the prefix empty and the window
    /// holding exactly the screens it expects, of the right side. Asked of a source whose
    /// piece already stands on its origin. One statement for `attacks_square_as` and for
    /// an attack through a portal (schema 19), which asks it of the entry square.
    #[inline(always)]
    fn source_path_clear(&self, player: usize, source: &AttackSource) -> bool {
        self.path_clear_on(&self.board, player, source)
    }

    /// `source_path_clear` on a given board (generation 40, N3c): the pass test asks it of
    /// its copy, after the pounces before it (`pounce.rs`). One statement of a player's path,
    /// so the hot path above is this, inlined on `self.board`.
    #[inline(always)]
    pub(super) fn path_clear_on(&self, board: &[u8; MAX_SQUARES], player: usize, source: &AttackSource) -> bool {
        let t = self.t;
        {
            let path = t.attack_path(source);
            // The prefix must be empty and the window must hold exactly the screens the
            // source expects. A slide is all prefix and no screens; a cannon is all
            // window and one screen; a bounded hop is both, which is the case that used
            // to be inexpressible here.
            let (prefix, window) = path.split_at((source.prefix_len as usize).min(path.len()));
            // The window holds exactly the screens the source expects, and -- since
            // schema 3 -- they have to be the right side's pieces. A piece of the wrong
            // ownership still occupies the square, so it is counted as an obstruction
            // rather than skipped: `occupied` is the blocking test, `screens` the
            // permission one, and a hop that wants a friendly screen is not attacking
            // through an enemy one.
            let occupied = window.iter().filter(|s| board[**s as usize] != 0).count();
            if occupied != source.screens as usize {
                return false;
            }
            let usable = if source.screen_owner == SCREEN_ANY {
                occupied
            } else {
                let low = t.value_low(player);
                let high = low + t.side_stride as u16;
                window
                    .iter()
                    .filter(|s| {
                        let occupant = board[**s as usize] as u16;
                        occupant != 0
                            && screen_matches(source.screen_owner, occupant, low, high)
                    })
                    .count()
            };
            usable == source.screens as usize
                && prefix.iter().all(|s| board[*s as usize] == 0)
        }
    }

    /// Schema 13: whether a sweep source attacks its square. A sweep over the enemy
    /// royal is check.
    ///
    /// The path is the whole ray and `prefix_len` is the attacked square's index on it.
    /// Under a capture matrix a piece the sweeper may not take blocks, before the square
    /// or after it. The square is attacked when it is a capture landing itself, or when a
    /// landing further along would sweep it: an empty square with move mode, or the other
    /// side's piece on a square with capture mode. The occupant of the square itself was
    /// already judged by the caller's matrix test.
    ///
    /// Q1063: from the sweep's length on (`SWEEP_LENGTH_SHIFT`), the path is a ride: any
    /// piece there before the square blocks, and a landing past such a square never counts,
    /// since it would have to pass over the piece on it (the compiler gives a ride square no
    /// `SWEEP_BEYOND`, and the walk beyond stops at the first piece on the ride).
    ///
    /// Its own walk over the attack path, not `Tables::sweep_victims`: that one answers
    /// what a played sweep removes, this one whether a sweep could reach the square. A
    /// change to what a sweep clears or what stops it changes both.
    #[inline(never)]
    fn sweep_attacks(&self, player: usize, source: &AttackSource) -> bool {
        let t = self.t;
        let path = t.attack_path(source);
        let index = source.prefix_len as usize;
        let attacker = t.type_of(source.piece_value);
        let blocks = |occupant: u8| {
            occupant != 0 && t.has_capture_matrix && !t.may_capture(attacker, t.type_of(occupant))
        };
        if path[..index].iter().any(|square| blocks(self.board[*square as usize])) {
            return false;
        }
        let swept = (source.sweep >> SWEEP_LENGTH_SHIFT) as usize;
        let rides_from = if swept == 0 { usize::MAX } else { swept };
        if swept != 0
            && swept < index
            && path[swept..index].iter().any(|square| self.board[*square as usize] != 0)
        {
            return false;
        }
        if source.sweep & SWEEP_LANDS != 0 {
            return true;
        }
        if source.sweep & SWEEP_BEYOND == 0 {
            return false;
        }
        let modes = t.attack_path_modes(source);
        let low = t.value_low(player);
        let high = low + t.side_stride as u16;
        for beyond in index + 1..path.len() {
            let occupant = self.board[path[beyond] as usize];
            let mode = modes[beyond];
            if occupant == 0 {
                if mode & 1 != 0 {
                    return true;
                }
            } else if blocks(occupant) {
                return false;
            } else if !(low <= occupant as u16 && (occupant as u16) < high) && mode & 2 != 0 {
                return true;
            } else if beyond >= rides_from {
                return false;
            }
        }
        false
    }

    /// Whether `square` is attacked by `by`, **or by a wild piece** (schema 9).
    ///
    /// One definition of "attacked", and every place the core asks whether a royal is
    /// safe goes through it: the legality filter, the illegal-position guard, checkmate,
    /// `forbid_check`, castling's transit squares and `reach.safe`. A wild piece's
    /// threat is the square it will step to next, which is one attack source per route
    /// square in a third owner's block, so this is the same slice walk twice.
    ///
    /// **`check_limit` is the one place that does not ask.** It counts checks a *player*
    /// delivers, and a boar's threat is nobody's check.
    ///
    /// Generation 37 (N2): a shielded victim is not threatened by a wild piece, which never
    /// takes one. Asked only where the wild source answers yes, and only beside a shield.
    #[inline(always)]
    fn threatened(&self, by: usize, square: usize) -> bool {
        self.attacks_any(by, square)
            || (self.t.has_npcs
                && self.attacks_square(WILD, square)
                && (!self.t.npc_shield || !self.shielded_from_wild(square)))
            || (self.t.majority.as_deref().is_some_and(|m| m.threatens_royals)
                && self.held_attacks(by, square, None))
    }

    /// `threatened`, for the callers that mean a specific victim -- castling's transit
    /// squares, which are empty and would otherwise skip the capture matrix.
    #[inline]
    fn threatened_as(&self, by: usize, square: usize, victim: usize) -> bool {
        (match self.t.portals.as_deref() {
            None => match self.t.sight.as_deref() {
                None => self.attacks_square_as(by, square, victim),
                Some(sight) => self.attacks_in_sight_game(sight, by, square, Some(victim)),
            },
            Some(portals) => self.attacks_in_portal_game(portals, by, square, Some(victim)),
        }) || (self.t.has_npcs
            && self.attacks_square_as(WILD, square, victim)
            && !(self.t.npc_shield && self.shielded_from_wild(square)))
            || (self.t.majority.as_deref().is_some_and(|m| m.threatens_royals)
                && self.held_attacks(by, square, Some(victim)))
    }

    /// Whether `player` attacks `square` by any way the game has: `attacks_square`, and in
    /// a portal game (schema 19) through a portal too. A check through a portal is a check,
    /// so every place a player's attack on a royal is asked -- `threatened`, the check
    /// count, castling out of check, the drop-mate test -- asks this. One null test where
    /// there are no portals, which is the whole of what chess pays for portals and folds
    /// here: the fold game's answer is a statement of its own (`attacks_in_portal_game`),
    /// so `attacks_square`, the hottest query, is the one it always was.
    #[inline(always)]
    fn attacks_any(&self, player: usize, square: usize) -> bool {
        match self.t.portals.as_deref() {
            // Schema 30 (Q999a): a game with a sight reads attack under its effects (a frozen
            // or disarmed piece attacks nothing, a shielded one is attacked by nothing), the
            // third arm; portals are refused beside a sight, so the arms never meet.
            None => match self.t.sight.as_deref() {
                None => self.attacks_square(player, square),
                Some(sight) => self.attacks_in_sight_game(sight, player, square, None),
            },
            Some(portals) => self.attacks_in_portal_game(portals, player, square, None),
        }
    }

    /// `attacks_any` in a portal game: directly (a fold is floor to a source that crosses
    /// it, schema 24), through a fold onto a far end (`fold_attacks`), or through a
    /// momentum portal (`portal_attacks`). `victim` is the type a caller means, or `None`
    /// for whatever stands on the square.
    #[cold]
    #[inline(never)]
    fn attacks_in_portal_game(
        &self, portals: &PortalTables, player: usize, square: usize, victim: Option<usize>,
    ) -> bool {
        let victim = victim.unwrap_or_else(|| {
            let occupant = self.board[square];
            if occupant == 0 { usize::MAX } else { self.t.type_of(occupant) }
        });
        self.attacks_square_as(player, square, victim)
            || (portals.folds.is_some() && self.fold_attacks(portals, player, square, victim))
            || self.portal_attacks(portals, player, square, victim)
    }

    /// Where each portal end stands: end `k` (type `first_type + k`) at index `k`, and
    /// `u16::MAX` for an end not found.
    ///
    /// The third owner's occupancy set is read as a *hint* and every square it names is
    /// checked against the board, which is what makes this sound inside a legality filter,
    /// where the sets describe the board before the move under test. The only move that
    /// can carry a portal off a square is that portal's own, so an end the hint missed has
    /// moved, and one scan of the board finds it.
    #[inline(never)]
    fn portal_squares(&self, portals: &PortalTables) -> [u16; MAX_PORTAL_ENDS] {
        let t = self.t;
        let mut at = [u16::MAX; MAX_PORTAL_ENDS];
        let mut found = 0;
        for square in self.occupied[PORTAL_OWNER].iter() {
            found += self.note_portal(portals, &mut at, square);
        }
        if found < portals.n_momentum {
            for square in 0..t.n_squares {
                self.note_portal(portals, &mut at, square);
            }
        }
        at
    }

    /// `portal_squares`' one step: record the end standing on `square`, if one does and it
    /// is not recorded yet, and say how many were added.
    #[inline(always)]
    fn note_portal(&self, portals: &PortalTables, at: &mut [u16; MAX_PORTAL_ENDS], square: usize) -> usize {
        let value = self.board[square];
        if !self.t.is_portal_end(value) {
            return 0;
        }
        let end = self.t.type_of(value) - portals.first_type;
        if at[end] != u16::MAX {
            return 0;
        }
        at[end] = square as u16;
        1
    }

    /// Schema 19: whether `player` could take a piece of `victim` type (`usize::MAX` for
    /// any) on `square` through a portal: some end that admits the source's type for
    /// `player` (schema 21's `admits`) is reached by a capture-capable source of
    /// `player`'s, travelling in a direction that brings it out of the end this one leads
    /// to (`dest`) onto `square`. The source tables are the ordinary ones read at the entry
    /// square; the directions are the momentum the compiler gave each source.
    #[inline(never)]
    fn portal_attacks(
        &self, portals: &PortalTables, player: usize, square: usize, victim: usize,
    ) -> bool {
        let t = self.t;
        let at = self.portal_squares(portals);
        for end in 0..portals.n_ends {
            let entry = at[end];
            let entry_type = portals.first_type + end;
            let admits = portals.admits[entry_type][player];
            let to = portals.dest[entry_type];
            if entry == u16::MAX || admits == 0 || to < 0 {
                continue;
            }
            let far = at[to as usize - portals.first_type];
            if far == u16::MAX {
                continue;
            }
            let mut wanted = 0u8;
            for direction in 0..EXIT_DIRECTIONS {
                if portals.exit(t.n_squares, player, far as usize, direction) == square as i16 {
                    wanted |= 1 << direction;
                }
            }
            if wanted == 0 {
                continue;
            }
            for source in t.attackers_of(player, entry as usize) {
                if source.dirs & wanted == 0
                    || source.origin as usize == square
                    || self.board[source.origin as usize] != source.piece_value
                    || (source.requires_unmoved && !self.unmoved.get(source.origin as usize))
                    || admits & (1 << t.type_of(source.piece_value)) == 0
                {
                    continue;
                }
                if victim != usize::MAX && !t.may_capture(t.type_of(source.piece_value), victim) {
                    continue;
                }
                if self.source_path_clear(player, source) {
                    return true;
                }
            }
        }
        false
    }

    /// Schema 22: whether a fold stands on `square`. Eight compares, asked only in a game
    /// with folds; every other game's `fold_at` is all -1.
    #[inline(always)]
    pub fn fold_on(&self, square: usize) -> bool {
        self.fold_at.contains(&(square as i16))
    }

    /// Schema 22: the fold end standing on `square`, if one does.
    #[inline(always)]
    fn fold_end_at(&self, portals: &PortalTables, square: usize) -> Option<usize> {
        (0..portals.n_ends).find(|&end| self.fold_at[end] == square as i16)
    }

    /// Schema 24: whether the fold on `square` lets `player`'s piece of `piece_type` go
    /// through (`docs/archive/plan-portals-2.md` §3R.1 rule 2): it admits it and leads somewhere.
    /// The end and the square of the end it leads to, or `None`. Says nothing of either
    /// square's occupant: `square` is empty where a landing asks, and what stands on the
    /// far square is the landing's to judge.
    #[inline]
    fn fold_through(
        &self, portals: &PortalTables, square: usize, piece_type: usize, player: usize,
    ) -> Option<(usize, usize)> {
        let end = self.fold_end_at(portals, square)?;
        let entry = portals.first_type + end;
        if portals.admits[entry][player] & (1 << piece_type) == 0 {
            return None;
        }
        let to = portals.dest[entry];
        if to < 0 {
            return None;
        }
        let far = self.fold_at[to as usize - portals.first_type];
        (far >= 0).then_some((end, far as usize))
    }

    /// Schema 24: whether `player` could take a piece of `victim` type (`usize::MAX` for
    /// any) on `square` by going through a fold (`docs/archive/plan-portals-2.md` §3R.1 rule 8): a
    /// fold end stands on `square`, and some end leading to it stands on an empty square
    /// that a capture step of `player`'s reaches -- an attack source onto that square whose
    /// piece stands on its origin (unmoved where it must be), is not the piece on `square`,
    /// is admitted by the end, may take `victim`, and whose path is clear. No board walk
    /// and no ray: a source onto the near end *is* a capture step onto it. A piece standing
    /// in a fold reaches nothing through it, so an occupied near end is skipped.
    #[inline(never)]
    fn fold_attacks(
        &self, portals: &PortalTables, player: usize, square: usize, victim: usize,
    ) -> bool {
        let t = self.t;
        let Some(arrival) = self.fold_end_at(portals, square) else {
            return false;
        };
        let arrival_type = (portals.first_type + arrival) as i8;
        for end in 0..portals.n_ends {
            let entry = portals.first_type + end;
            let near = self.fold_at[end];
            if portals.dest[entry] != arrival_type || near < 0 || self.board[near as usize] != 0 {
                continue;
            }
            let admits = portals.admits[entry][player];
            if admits == 0 {
                continue;
            }
            for source in t.attackers_of(player, near as usize) {
                let attacker = t.type_of(source.piece_value);
                if source.origin as usize == square
                    || source.sweep != 0
                    || self.board[source.origin as usize] != source.piece_value
                    || (source.requires_unmoved && !self.unmoved.get(source.origin as usize))
                    || admits & (1 << attacker) == 0
                {
                    continue;
                }
                if t.has_capture_matrix && victim != usize::MAX && !t.may_capture(attacker, victim)
                {
                    continue;
                }
                if self.source_path_clear(player, source) {
                    return true;
                }
            }
        }
        false
    }

    /// Whether `player`'s royal is attacked, remembered for this position.
    ///
    /// Three callers ask it about one node -- the illegal-position guard in
    /// `terminal_prefix`, the check filter in `generate_legal`, and the evaluator's
    /// royal-exposure term, which asks for *both* sides at every leaf -- and each was
    /// paying for its own attack scan.
    pub fn in_check(&self, player: usize) -> bool {
        if self.t.royal_types.is_empty() {
            return false;
        }
        self.remembered_check(player, || self.compute_in_check(player))
    }

    /// `in_check`, for the one caller that has already located the royal.
    ///
    /// The check filter looks the royal up to decide whether it needs the slow path at
    /// all, so asking `in_check` there would locate it a second time. Equivalent only on
    /// the single-royal path, which is the only place it is called from.
    #[inline]
    fn in_check_at(&self, player: usize, royal: usize) -> bool {
        debug_assert!(!self.t.multi_royal);
        self.remembered_check(player, || self.threatened(1 - player, royal))
    }

    /// Read the memo, or answer and remember. **The only place the memo is touched**,
    /// which is what keeps its one rule in one place: while `scratch` says a filter has
    /// the board mid-experiment, the answer is about a position no game is in, so it is
    /// neither read from the memo nor written to it.
    #[inline]
    fn remembered_check(&self, player: usize, answer: impl Fn() -> bool) -> bool {
        if self.scratch > 0 {
            return answer();
        }
        let memo = self.check_memo.get();
        if memo[player] >= 0 {
            debug_assert_eq!(memo[player] != 0, self.compute_in_check(player));
            return memo[player] != 0;
        }
        let found = answer();
        debug_assert_eq!(found, self.compute_in_check(player));
        let mut memo = self.check_memo.get();
        memo[player] = i8::from(found);
        self.check_memo.set(memo);
        found
    }

    /// The definition `in_check` memoises, and what every `debug_assert` above compares
    /// a remembered answer against.
    fn compute_in_check(&self, player: usize) -> bool {
        // Generation 39 (N3b, `docs/archive/plan-npc.md` §4.3): the side to move in a pounce game is in
        // check by the pass test.
        if self.t.pounce.is_some() && player == self.side {
            return self.in_check_by_pass(player);
        }
        if self.t.multi_royal {
            return self.royal_squares(player).iter().any(|&royal| {
                self.threatened(1 - player, royal)
            });
        }
        let royal = self.royal_square(player);
        royal >= 0 && self.threatened(1 - player, royal as usize)
    }
}

impl<'t> Game<'t> {
    /// Locate the rook a castling move moves, and where it lands.
    ///
    /// For step-based castling: the first rook outward from the king in the direction
    /// of travel, landing on the square the king crossed.
    /// For file-based castling: the rook file is encoded in the move's promotion field,
    /// and destination is looked up from the castle_files table.
    pub fn castling_partner(&self, origin: usize, target: usize) -> Option<(usize, usize)> {
        let t = self.t;
        let (_, rook_type, _) = t.castling?;
        let rank_base = (origin / t.width) * t.width;

        if let Some((_, _, rook_qs, rook_ks)) = t.castle_files {
            // File-based: can't determine rook_file from origin/target alone
            // This path is used by lib.rs boundary check -- for file-based we need the
            // rook_file from the move encoding, so just check if any rook exists.
            let king_file = origin % t.width;
            // Try both directions
            for &rook_file_dest in &[rook_qs, rook_ks] {
                if rook_file_dest < 0 { continue; }
                // Find a rook on the right side
                let step: i32 = if (target % t.width) >= king_file { 1 } else { -1 };
                let mut file = king_file as i32 + step;
                while file >= 0 && file < t.width as i32 {
                    let square = rank_base + file as usize;
                    if t.is_piece(self.board[square], self.side, rook_type) {
                        return Some((square, rank_base + rook_file_dest as usize));
                    }
                    file += step;
                }
            }
            None
        } else {
            let step: i32 = if target > origin { 1 } else { -1 };
            let mut file = (origin % t.width) as i32 + step;
            while file >= 0 && file < t.width as i32 {
                let square = rank_base + file as usize;
                if t.is_piece(self.board[square], self.side, rook_type) {
                    return Some((square, (target as i32 - step) as usize));
                }
                file += step;
            }
            None
        }
    }

    /// Internal castling rook lookup. `rook_file_hint` is the promotion field of the
    /// move; for file-based castling it names the rook's file, for step-based it is 0.
    fn castling_rook_squares(&self, origin: usize, target: usize, rook_file_hint: i32) -> (usize, usize) {
        let t = self.t;
        let (_, rook_type, _) = t.castling.expect("castling without a rule");
        let rank_base = (origin / t.width) * t.width;

        if let Some((_, _, rook_qs, rook_ks)) = t.castle_files {
            // File-based castling: rook_file is encoded in the move
            let side = if rook_file_hint as usize > origin % t.width { 1 } else { 0 };
            let rook_dest = if side == 1 { rook_ks } else { rook_qs };
            (rank_base + rook_file_hint as usize, rank_base + rook_dest as usize)
        } else {
            // Step-based: first rook outward from king
            let step: i32 = if target > origin { 1 } else { -1 };
            let mut file = (origin % t.width) as i32 + step;
            while file >= 0 && file < t.width as i32 {
                let square = rank_base + file as usize;
                if t.is_piece(self.board[square], self.side, rook_type) {
                    return (square, (target as i32 - step) as usize);
                }
                file += step;
            }
            panic!("castling move applied to a position with no partner rook");
        }
    }

    /// Write a square, and put the change into the key as it happens.
    ///
    /// **The one place a move's effects touch the board.** Promotion, en passant,
    /// castling, all three strike position rules and an explosion all go through here,
    /// so the key is right for every one of them by construction rather than by five
    /// branches each computing the same quantity a different way. The next effect gets
    /// its Zobrist handling for nothing instead of getting it wrong -- which is the
    /// failure this replaces: the atomic explosion bug was a square written by an effect
    /// the delta did not know about, and no perft can see one, because none of it
    /// changes a move count.
    ///
    /// **A square written twice cancels itself**, which is why nothing has to be
    /// deduplicated and no log has to be stored. A target square is written up to three
    /// times in one move -- the mover arrives, promotion replaces it, a strike or a
    /// blast overwrites it again -- and `a -> b -> c` XORs out `a` and in `b`, then out
    /// `b` and in `c`. What survives is exactly `a` out and `c` in. Writing a square the
    /// value it already holds cancels to nothing at all, which is the early return.
    #[inline]
    fn write(&mut self, square: usize, value: u8) {
        let before = self.board[square];
        if before == value {
            return;
        }
        if before != 0 {
            self.working_key ^= self.t.zobrist(before, square);
            self.occupied[self.t.owner_of(before)].clear(square);
        }
        if value != 0 {
            self.working_key ^= self.t.zobrist(value, square);
            self.occupied[self.t.owner_of(value)].set(square);
        }
        self.board[square] = value;
    }

    /// Write a square on the way *back*: `write` without the key, for `unmake`.
    ///
    /// `unmake` restores the key wholesale from `Undo`, so it never wanted the Zobrist
    /// half -- but it does want the occupancy half, and this is the one place it gets
    /// it. Every square `unmake` touches goes through here, in the order it always
    /// did, so the sets track the board write for write and are right whatever
    /// intermediate value a square passes through on the way to its final one.
    #[inline]
    fn put(&mut self, square: usize, value: u8) {
        let before = self.board[square];
        if before == value {
            return;
        }
        if before != 0 {
            self.occupied[self.t.owner_of(before)].clear(square);
        }
        if value != 0 {
            self.occupied[self.t.owner_of(value)].set(square);
        }
        self.board[square] = value;
    }

    /// Push a move that ends on `target`: one word per promotion target, then the plain
    /// arrival where promotion is optional or absent. `fan` says the target is occupied
    /// under `position: "choose"`, so each word fans out over its landing squares.
    /// `leaving` says the move starts in the mover's zone under schema 15's
    /// `on_leaving`, so it promotes wherever it lands.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn push_arrival(
        &self, out: &mut Vec<u32>, origin: usize, target: usize, piece_type: usize,
        player: usize, promoting: bool, leaving: bool, promo_optional: bool, fan: bool,
    ) {
        let t = self.t;
        if promoting && (leaving || t.promotes_on(player, piece_type, target)) {
            for &promoted in t.promotion_targets(player, piece_type) {
                if !self.promotion_offered(player, piece_type, promoted) {
                    continue;
                }
                self.push_capture(
                    out, encode_move(origin, target, MOVE_PROMOTION, promoted), piece_type,
                    player, target, fan,
                );
            }
            if promo_optional {
                self.push_capture(
                    out, encode_move(origin, target, MOVE_NORMAL, 0), piece_type, player,
                    target, fan,
                );
            }
        } else {
            self.push_capture(
                out, encode_move(origin, target, MOVE_NORMAL, 0), piece_type, player, target,
                fan,
            );
        }
    }

    /// Push `m`, or under `fan` one word per legal landing square of the piece struck on
    /// `captured_square` (schema 12).
    ///
    /// A capture whose target does not survive is one word with landing index 0. One that
    /// survives fans out over the static landing list in ascending order, and a candidate
    /// is legal when it is empty once the attacker has left: empty now, or the origin, and
    /// never the target the attacker lands on. The index is into the static list, so a
    /// stored move names the same square in any position. No legal candidate is no move.
    #[inline(always)]
    fn push_capture(
        &self, out: &mut Vec<u32>, m: u32, mover_type: usize, player: usize,
        captured_square: usize, fan: bool,
    ) {
        if fan {
            self.fan_out_landing(out, m, mover_type, player, captured_square);
        } else {
            out.push(m);
        }
    }

    fn fan_out_landing(
        &self, out: &mut Vec<u32>, m: u32, mover_type: usize, player: usize,
        captured_square: usize,
    ) {
        let t = self.t;
        let rule = t.strike.as_ref().expect("a landing fan-out without a strike rule");
        let victim = self.board[captured_square];
        let cell = mover_type * rule.n_types + t.type_of(victim);
        let becomes = if t.owner_of(victim) == player { rule.own_to[cell] } else { rule.to[cell] };
        if becomes < 0 {
            out.push(m);
            return;
        }
        let origin = move_from(m);
        let target = move_to(m);
        for (index, &square) in rule.landing(player, captured_square).iter().enumerate() {
            let square = square as usize;
            if square != target && (self.board[square] == 0 || square == origin) {
                out.push(with_landing(m, index));
            }
        }
    }

    /// Where a contact leaves the two pieces, as squares to write in order, or `None`
    /// when this capture is an ordinary one after all.
    ///
    /// Keyed by the *pair* since schema 10 (`rules.contacts`): what a meeting leaves
    /// behind can depend on who arrived as well as on who was standing there, and it can
    /// change the attacker as well as the piece it took. A `strike` fills the attacker's
    /// column with -1 and gives every attacker the same row, so schema 6 is this function
    /// with one column empty rather than a second function beside it.
    ///
    /// **One statement of the rule, three callers.** `make` plays it, and the legality
    /// filter has to mirror it twice -- a strike can turn its victim into a piece that
    /// gives check, and a filter that did not know would let a king be captured. Those
    /// two mirrors were written by hand and were the fifth and sixth copies of the same
    /// paragraph; a rule stated in more than one place is a rule that can be stated
    /// differently in one of them.
    ///
    /// A survivor can be in exactly two places and there is no third: where it was
    /// (`Stay`, the attacker never left its square) or where its attacker was (`Swap`,
    /// the one square a strike is guaranteed to have freed). `Advance` is what capture
    /// has always done and is refused by the validator beside a chain that keeps
    /// anything alive, so this returns `None` for it.
    ///
    /// `Choose` (schema 12) is `Advance` for the attacker and places the survivor on the
    /// landing square `landing_index` names, which movegen offered only while it was empty
    /// or the attacker's origin. The origin is emptied first so a survivor placed there
    /// stays.
    fn strike_writes(
        &self, mover_type: usize, captured: u8, origin: usize, target: usize,
        captured_square: usize, landing_index: usize,
    ) -> Option<StrikeOutcome> {
        let strike = self.t.strike.as_ref()?;
        if captured == 0 {
            return None;
        }
        // Read after the promotion branch, so an attacker that promoted on the strike
        // carries its new type wherever it ends up -- and *overridden* by a contact,
        // because a contact is stated about this particular meeting while a rank
        // promotion is stated about the square. `mover_type` is the attacker as it stood
        // before the move, which is the reading `explosion.fires` also takes: "a capture
        // by P" has to mean the piece that set out, or it is not readable.
        let attacker = self.board[target];
        let mover_owner = self.t.owner_of(attacker);
        let owner = self.t.owner_of(captured);
        let cell = mover_type * strike.n_types + self.t.type_of(captured);
        // Schema 11: a friendly target reads its own pair of tables. Movegen offers such a
        // move only where `own_link` is set, so the tables are never empty here.
        let (becomes, mover_becomes) = if owner == mover_owner {
            (strike.own_to[cell], strike.own_mover_to[cell])
        } else {
            (strike.to[cell], strike.mover_to[cell])
        };
        // Schema 17 (Q540): the attacker changes sides (Andernach) -- only on a meeting
        // with the other side, since `switches_sides` is a strike's and a strike never
        // meets its own.
        let flips = owner != mover_owner && strike.flips(cell);
        if becomes == -1 && mover_becomes < 0 && !flips {
            return None;
        }
        // The survivor is a piece that has just become something else, which is the
        // same event as a promotion and a drop and takes the same answer: it enters
        // play as its new type's initial orientation. There is no other reading -- the
        // facing it had belonged to the type it no longer is. The attacker's half of a
        // contact is the same event read from the other side and gets the same answer.
        let landing = match strike.position {
            StrikePosition::Stay => origin,
            _ => target,
        };
        // A side switch is the attacker changing sides as whatever it now is (a pawn that
        // promoted on the capture switches as the promoted piece, at that type's initial
        // facing). TC1: the one change of allegiance on the board is `rehome_value`'s; TC2
        // (`docs/plan-turncoat.md` §6): it keeps the heading it has, so `attacker` is the
        // mover's byte as it stands, turned by nothing.
        let attacker = if flips {
            let now = if mover_becomes >= 0 {
                self.t.entering_value(mover_owner, mover_becomes as usize)
            } else {
                attacker
            };
            self.t.rehome_value(now, 1 - mover_owner, landing)
        } else if mover_becomes >= 0 {
            self.t.arriving_value(mover_owner, mover_becomes as usize, landing)
        } else {
            attacker
        };
        let mover_material_changed = mover_becomes >= 0
            && self.t.piece_values[mover_type] != self.t.piece_values[mover_becomes as usize];
        // Where the survivor lands is known in every arm before anything is written, and
        // schema 9's `promote_on_entry` needs it: a piece left standing inside its own
        // promotion zone by a strike has entered play there, exactly as a dropped one has.
        //
        // `Advance` keeps nothing standing -- the validator refuses to pair it with a
        // table that does -- so it reaches here only for the attacker's half, which is
        // one write where schema 6 had none and is why this returns `None` for it only
        // when there is nothing to say.
        //
        // Schema 17 (Q540) adds three readings. `REMOVED_IN_PLACE` under `Stay` removes the
        // victim and keeps the attacker on its origin (rifle), which is the `Stay` arm
        // with nothing left standing. `Home` advances the attacker and puts the survivor
        // on its home square, read off the rebirth table *after* the attacker has landed:
        // the origin is empty by then and may take it, the target is not and may not, and
        // a home square anything stands on removes it (circe). And `flips` (above) only
        // changes what `attacker` is, so every arm carries it.
        let changes_attacker = mover_becomes >= 0 || flips;
        let (writes, count, survivor) = match (strike.position, becomes >= 0) {
            (StrikePosition::Stay, true) => {
                let survivor = self.t.arriving_value(owner, becomes as usize, captured_square);
                (
                    [(target, 0), (origin, attacker), (captured_square, survivor), (0, 0)],
                    3,
                    Some((captured_square, survivor)),
                )
            }
            (StrikePosition::Swap, true) => {
                let survivor = self.t.arriving_value(owner, becomes as usize, origin);
                (
                    [(captured_square, 0), (target, attacker), (origin, survivor), (0, 0)],
                    3,
                    Some((origin, survivor)),
                )
            }
            (StrikePosition::Choose, true) => {
                let chosen = strike.landing(mover_owner, captured_square)[landing_index] as usize;
                let survivor = self.t.arriving_value(owner, becomes as usize, chosen);
                (
                    [(origin, 0), (target, attacker), (chosen, survivor), (0, 0)],
                    3,
                    Some((chosen, survivor)),
                )
            }
            (StrikePosition::Home, true) => {
                let home = strike.rebirth_square(owner, becomes as usize, captured_square);
                let advance = [(captured_square, 0), (origin, 0), (target, attacker)];
                if home >= 0 && home as usize != target && self.board[home as usize] == 0 {
                    let home = home as usize;
                    let survivor = self.t.arriving_value(owner, becomes as usize, home);
                    (
                        [advance[0], advance[1], advance[2], (home, survivor)],
                        4,
                        Some((home, survivor)),
                    )
                } else {
                    if !changes_attacker {
                        return None;
                    }
                    ([advance[0], advance[1], advance[2], (0, 0)], 3, None)
                }
            }
            (StrikePosition::Stay, false) => {
                if !changes_attacker && becomes != REMOVED_IN_PLACE {
                    return None;
                }
                ([(target, 0), (captured_square, 0), (origin, attacker), (0, 0)], 3, None)
            }
            (StrikePosition::Swap, false)
            | (StrikePosition::Choose, false)
            | (StrikePosition::Home, false)
            | (StrikePosition::Advance, _) => {
                if !changes_attacker {
                    return None;
                }
                ([(captured_square, 0), (origin, 0), (target, attacker), (0, 0)], 3, None)
            }
        };
        Some(StrikeOutcome { writes, count, survivor, mover_material_changed })
    }

    /// The squares wild pieces stand on right now, ascending (schema 9).
    ///
    /// **Listed before the phase moves anything, and that is load-bearing**: a single
    /// ascending scan that stepped a piece onto a higher square would meet it again in
    /// the same pass and step it twice.
    ///
    /// Read off `npc_squares` -- every square any route visits -- rather than off the
    /// board or off `occupied[WILD]`. The board is up to 256 squares for what is usually
    /// two pieces, and the occupancy sets are stale inside a legality filter, which
    /// writes the board directly and does not tell them; the route list is right in both
    /// places because it is a fact about the rules rather than about the position.
    #[inline]
    fn npc_here(&self) -> SquareSet {
        let mut here = SquareSet::EMPTY;
        for &square in &self.t.npc_squares {
            // Generation 42: the duck may stand on a route square and never walks it.
            if self.t.is_wild_piece(self.board[square as usize]) {
                here.set(square as usize);
            }
        }
        here
    }

    /// What the wild piece standing on `square` does: `Some(next)` to step there, taking
    /// whatever it may, and `None` to wait where it is.
    ///
    /// **The rule, stated once**, for the two callers that play it -- `make`, which
    /// writes through `write` so the key and the occupancy follow, and the legality
    /// filter, which writes the raw board and puts it back. Both read the board as the
    /// wild pieces earlier in the same phase left it, which is what makes the phase a
    /// sequence rather than a set.
    ///
    /// Three ways to wait, and each fails silently the other way: the next square is the
    /// square itself, which is the one-square route's sentinel for a wild piece that
    /// never moves; it holds another wild piece, which one never takes; or it holds a
    /// piece this one's *type* may not capture, read off schema 3's matrix exactly as a
    /// player's piece reads it.
    ///
    /// The middle one is reachable since schema 23's `also_at` put several wild pieces on
    /// one route: the piece behind waits for the one in front. Whether the one in front
    /// has already gone this phase is `npc_step_order`'s business, not this function's --
    /// it reads the board as the earlier steps left it.
    ///
    /// Generation 37 (N2, `docs/archive/plan-npc.md` §4.2): a fourth, **a wild piece never takes a
    /// shielded piece**. `shields` is the phase's snapshot of the shielded squares
    /// (`wild_shields`), read once at the phase's start, so a step earlier in the phase that
    /// moves a shield (a wild piece carrying one, or one eating the piece that gave it) bares
    /// what it shielded from the next phase, not for the next step (§3.1). Empty in every
    /// game without a shield beside wild pieces.
    /// `SHIELDS` is `Tables::npc_shield` as a constant, so every game without a shield beside
    /// wild pieces walks the step it walked before generation 37 (the bench: *wolves* lost 11%
    /// to the one set test when it was a run-time branch).
    #[inline]
    fn npc_step_of<const SHIELDS: bool>(&self, square: usize, shields: &SquareSet) -> Option<usize> {
        let t = self.t;
        let value = self.board[square];
        debug_assert!(value != 0 && t.owner_of(value) == WILD);
        let next = t.npc_next[square];
        if next < 0 || next as usize == square {
            return None;
        }
        let next = next as usize;
        let standing = self.board[next];
        if standing == 0 {
            return Some(next);
        }
        if t.owner_of(standing) == WILD || (SHIELDS && shields.get(next)) {
            return None;
        }
        if t.may_capture(t.type_of(value), t.type_of(standing)) {
            Some(next)
        } else {
            None
        }
    }

    /// Every wild piece standing now, in the order the phase steps them (schema 23).
    ///
    /// **Front first.** With several wild pieces on one route, a piece whose next square
    /// holds a wild piece that has yet to step this phase goes after it, so a queue moves
    /// as one when its front moves and waits as one when its front waits. Ascending order
    /// alone would make that depend on board numbering: the piece behind would follow on
    /// a turn where the one in front had the lower index and wait on a turn where it had
    /// the higher. A route full from end to end has no front, and every piece on it waits
    /// (each one's next square holds a piece that is not leaving). `NpcOrder` has the
    /// passes; the order reads only this snapshot and `npc_next`, never what a step did,
    /// so the real phase, the legality filter and `coming_phases` step in one order.
    #[inline]
    fn npc_step_order(&self) -> NpcOrder<'t> {
        NpcOrder::new(self.t, self.npc_here())
    }

    /// The phase, played for real: every wild piece walks, and every square written goes
    /// through `write` so the Zobrist key and the occupancy sets follow it.
    ///
    /// Returns whether anything was eaten, which is the material change the no-progress
    /// clock and the repetition window read. A quiet step is neither -- the route is a
    /// cycle, so the position will come round again, and that is what lets the repetition
    /// rule catch a game in which only the boar moves.
    ///
    /// **Cold and never inlined (Q857).** Schema 23's two monomorphs, inlined into `make`
    /// behind `has_npcs`, cost chess 3% in code it never runs. Out of line, chess got it
    /// back and the wild-piece games got faster too (*wolves patrol* perft(5) 10%):
    /// `docs/reference/replay-2026-10-12.md`, *Q857*.
    #[cold]
    #[inline(never)]
    fn npc_phase(&mut self) -> bool {
        // Generation 39 (N3b): a game with a pounce plays its own phase (`pounce.rs`).
        if self.t.pounce.is_some() {
            return self.pounce_phase();
        }
        // Generation 38 (N5): a game with a line plays its own phase (`turning.rs`).
        if self.t.npc_lines.is_some() {
            return self.npc_phase_lines();
        }
        if self.t.npc_shield {
            return self.npc_phase_shielded();
        }
        // The steps, then (generation 36, N1) the returns: one walk (`phase.rs`).
        let mut live = Live::new();
        self.wild_phase::<_, false>(&mut live);
        live.ate
    }

    /// `npc_phase` in a game with a shield beside wild pieces (generation 37): the shields
    /// read once, at the phase's start (`wild_shields`). Its own function, so the phase of
    /// every other game is the one it was.
    #[cold]
    #[inline(never)]
    fn npc_phase_shielded(&mut self) -> bool {
        let mut live = Live::new();
        self.wild_phase::<_, true>(&mut live);
        live.ate
    }

    /// `make`'s phase: the respawn clocks logged first (generation 36, only under the rule),
    /// then the phase. Cold and out of line for `npc_phase`'s reason: chess pays one flag
    /// test in `make` for the whole of it.
    #[cold]
    #[inline(never)]
    fn npc_phase_of_make(&mut self) -> bool {
        // Generation 42 (N4b): a duck game's piece half runs no phase; the duck half does.
        if self.t.duck.is_some() && !self.duck_owed() {
            return self.duck_piece_half();
        }
        if self.t.respawn.is_some() {
            self.respawn_save();
        }
        if self.t.npc_lines.is_some() {
            self.heading_log.push(self.heading_back);
        }
        self.npc_phase()
    }

    /// `unmake`'s half of the phase: every step above `base` walked back, newest first,
    /// and the respawn clocks put back. Behind `has_npcs`, cold, for `npc_phase_of_make`'s
    /// reason.
    #[cold]
    #[inline(never)]
    fn unmake_phase(&mut self, base: usize) {
        while self.npc_log.len() > base {
            let (from, to, eaten, kind) =
                self.npc_log.pop().expect("npc log shorter than its base");
            let value = self.board[to];
            self.put(to, eaten);
            // A return (generation 36) and a copy (generation 41) came from nowhere, so
            // nothing goes back. A turn (generation 38) is a step, and a turned wait puts its
            // piece back where it is.
            if kind != PHASE_RETURN && kind != PHASE_COPY {
                self.put(from, value);
            }
        }
        if self.t.respawn.is_some() {
            self.respawn_unsave();
        }
        if self.t.npc_lines.is_some() {
            self.heading_back =
                self.heading_log.pop().expect("heading log shorter than the undo stack");
        }
    }

    /// What the phase did, oldest step first: `(from, to, eaten)` per wild piece that
    /// walked, `eaten` being the board byte the destination held at the instant the wild
    /// piece wrote over it.
    ///
    /// **That byte, and not the piece the ply started with.** A pawn that promoted and
    /// was then eaten on the square it promoted onto is the piece it *became*, and a
    /// castling rook eaten in transit is a rook on its new square. A caller diffing two
    /// positions has to guess at both; this is the core not making it guess.
    ///
    /// The log is scratch belonging to the undo stack, so it holds one move's steps only
    /// where one move was made on a fresh `Game` -- which is what the boundary's `apply`
    /// does, and the only caller this is for.
    pub fn phase_steps(&self) -> &[(usize, usize, u8, u8)] {
        &self.npc_log
    }

    /// What the phase would do if it ran on the board as it stands: `(from, to, eaten)`
    /// per wild piece that would walk, in the order the real phase walks them (Q660).
    ///
    /// The question the board asks to draw where each wild piece is going next. The real
    /// phase runs after the *coming* move, which can step into a wild piece's way, out of
    /// it, or take the piece, so this is the phase as the position now has it -- every
    /// step decided by `npc_step_of`, the one statement of the rule, on the board as the
    /// earlier steps of the same phase leave it. The board is put back before returning.
    ///
    /// `count` phases, one after another, each played on the board the one before it left
    /// (Q764): a wild piece walks twice before the player to move moves again -- after
    /// that player's move and after the reply -- so the board asks for two. Every phase
    /// after the first is still the board *without* the moves between them, which is the
    /// one thing this cannot know. Exactly `count` entries, an empty one for a phase in
    /// which nothing would walk, and every entry empty for a ruleset without wild pieces.
    pub fn coming_phases(&mut self, count: usize) -> Vec<Vec<(usize, usize, u8, u8)>> {
        let mut phases = vec![Vec::new(); count];
        if !self.t.has_npcs {
            return phases;
        }
        if self.t.pounce.is_some() {
            return self.pounce_coming_phases(count);
        }
        let mut saved = Saved::new();
        // Generation 36: the clocks move on a phase at a time, on a copy.
        let mut clocks = self.respawn.clone();
        // Generation 38: the headings move on a phase at a time, on a copy.
        let mut heading = self.heading_back;
        for steps in phases.iter_mut() {
            // Generation 37: each phase's shields, read on the board it starts from.
            let shields = self.wild_shields();
            let mut sink =
                Coming { saved: &mut saved, steps, clocks: &mut clocks, heading: &mut heading };
            if self.t.npc_lines.is_some() {
                self.line_steps(&mut sink, &shields);
            } else {
                // Once per drawn position, never in a search, so the one order (and the
                // shielded step) serves every game.
                let order = self.npc_step_order();
                self.wild_steps::<_, _, true>(&mut sink, order, &shields);
            }
            self.respawn_walk(&mut sink, &shields);
        }
        saved.restore(&mut self.board);
        phases
    }

    /// The same phase, played on the raw board for a legality filter, with every square
    /// written pushed to `saved` so it can be put back.
    ///
    /// The filter has to see the position the move *leaves behind*, and that is the
    /// post-phase position: a move that lets a boar eat your royal in your own phase is
    /// illegal, and so is one that leaves your royal standing where a boar steps next.
    /// Two squares per wild piece is more than `Saved`'s five inline entries, so this is
    /// the one heap allocation the rule pays, and only on the rulesets that have it.
    ///
    /// Cold and never inlined into `play_for_filter`, for `npc_phase`'s reason (Q857).
    #[cold]
    #[inline(never)]
    fn npc_phase_for_filter(&mut self, saved: &mut Saved) {
        if self.t.pounce.is_some() {
            return self.pounce_phase_for_filter(saved);
        }
        if self.t.npc_lines.is_some() {
            return self.npc_phase_for_filter_lines(saved);
        }
        if self.t.npc_shield {
            return self.npc_phase_for_filter_shielded(saved);
        }
        // The steps, then (generation 36, N1) the returns: `npc_phase`'s walk (`phase.rs`).
        self.wild_phase::<_, false>(&mut Filter { saved });
    }

    /// `npc_phase_for_filter` beside a shield (generation 37), as `npc_phase_shielded`.
    #[cold]
    #[inline(never)]
    fn npc_phase_for_filter_shielded(&mut self, saved: &mut Saved) {
        self.wild_phase::<_, true>(&mut Filter { saved });
    }

    /// Close the en passant window if the phase wrote either of its two squares.
    ///
    /// An en passant capture is generated onto `ep_target` and removes the piece on
    /// `ep_victim`; if a boar has since stepped onto either, the capture would write over
    /// the boar without counting it, or remove a piece that is no longer there. The
    /// window is restored wholesale from `Undo`, so there is nothing to log.
    #[inline]
    fn close_ep_window_over(&mut self, first: usize, second: usize) {
        if self.ep_target == NO_SQUARE && self.ep_victim == NO_SQUARE {
            return;
        }
        let touched = |square: i16| square == first as i16 || square == second as i16;
        if touched(self.ep_target) || touched(self.ep_victim) {
            self.ep_target = NO_SQUARE;
            self.ep_victim = NO_SQUARE;
        }
    }

    /// Play `m`: the board, then the rights and the window, the payments, the wild phase,
    /// the hand, the counters and the key's tail, each a phase of its own (R5, CS8). Every
    /// phase is `#[inline(always)]`, so the body the compiler sees is the one it saw when
    /// this was one function; what the phases share travels as one `Copy` struct, `Played`.
    pub fn make(&mut self, m: u32) {
        let t = self.t;
        let kind = move_kind(m);
        // Schema 22: a fold's own move touches no square of the board byte. Generation 42:
        // the duck's half is its own function too. One comparison on the hot path for both
        // (TC5: a range, since a buy, kind 10, is played here as a drop is).
        if (MOVE_FOLD..=MOVE_DUCK).contains(&kind) {
            if kind == MOVE_FOLD {
                self.make_fold(m);
            } else {
                self.make_duck(m);
            }
            return;
        }
        let player = self.side;

        let previous_unmoved = self.unmoved;
        let previous_ep = self.ep_target;
        let previous_ep_victim = self.ep_victim;
        let previous_checks = self.checks;
        // Taken before anything moves, because `Undo` is pushed further down and the
        // decrement below has to happen before the key is finished.
        let previous_remaining = self.remaining;
        let previous_turns = self.turns_taken;
        // Seeded with the side flipped, which is right exactly when the turn ends; the
        // flip is undone further down when the same player moves again.
        self.working_key = self.key ^ t.zobrist_side;

        let mut cleared: Vec<(usize, u8)> = Vec::new();
        let mut played = self.write_board(m, player, &mut cleared);
        self.write_gate(m, &mut played);
        self.write_unmoved(&played);
        self.write_ep_window(m, &played);
        self.write_payments(&played);

        // The phase (schema 9): every wild piece walks, after the player's move is
        // complete and the en passant window has been set, and before the hand, the
        // check count, the material clock and the irreversibility test below. It runs
        // before the key's en passant terms because it may close that window.
        let npc_log = self.npc_log.len() as u32;
        // Generation 42 (N4b): under the duck the phase belongs to the turn's end, after the
        // duck half (`make_duck`), never to the piece move: `npc_phase_of_make` asks.
        let phase_ate = t.has_npcs && self.npc_phase_of_make();

        // Read after the phase, which clears the `unmoved` bit of every square it writes:
        // the irreversibility test is asked of the position the whole ply leaves behind.
        let rights_changed = self.unmoved != previous_unmoved;

        // The squares are already in the key: `write` put each change there as it
        // happened, which is the whole of this item. What is left is the part of the key
        // that is not about a square.
        //
        // This used to be five branches computing the same quantity five ways --
        // explosion, strike, drop, castle, normal -- one of which rescanned the whole
        // board on every capture in atomic *because* an explosion clears squares the
        // delta did not know about. The delta knows now. Nothing else should move, and
        // the ply-by-ply hash walk over every committed and archived ruleset is what
        // says so: no perft can see this class of bug, because none of it changes a
        // move count.
        let mut key = self.working_key;
        if previous_ep != NO_SQUARE {
            key ^= t.zobrist_ep_file(previous_ep);
        }
        if self.ep_target != NO_SQUARE {
            key ^= t.zobrist_ep_file(self.ep_target);
        }

        self.deposit_capture(&mut played);
        self.count_check(player);
        let coloured = self.write_territory(&played, &mut key);
        let (irreversible, progress) = progress_of(
            t, m, &played, phase_ate, !cleared.is_empty(), coloured.is_some(), rights_changed,
        );
        self.write_goal_left(&played);
        self.write_portal_bar(&played, &mut key);
        self.write_push_bar(m, &played, &mut key);
        self.write_rest(m, &played, &mut key);

        self.undo.push(Undo {
            origin: played.origin,
            target: played.target,
            kind,
            moving: played.moving,
            captured: played.captured,
            captured_square: played.captured_square,
            target_before: played.target_before,
            rook: played.rook,
            unmoved: previous_unmoved,
            ep_target: previous_ep,
            ep_victim: previous_ep_victim,
            halfmove: self.halfmove,
            key: self.key,
            window: self.window,
            checks: previous_checks,
            hand: played.hand,
            cleared,
            coloured,
            remaining: previous_remaining,
            turns_taken: previous_turns,
            npc_log,
            placed: played.placed,
        });
        self.end_ply(player, key, previous_remaining, irreversible, progress);
    }

    /// `make`'s board phase: every square the player's own move writes, by kind (a drop or
    /// a buy, a hire, a turn, a castling, a step with what it takes), and what the rest of
    /// `make` reads off them. The gate is `write_gate`'s; the wild phase comes later.
    #[inline(always)]
    fn write_board(&mut self, m: u32, player: usize, cleared: &mut Vec<(usize, u8)>) -> Played {
        let t = self.t;
        let origin = move_from(m);
        let target = move_to(m);
        let kind = move_kind(m);
        let mut hand: Option<(u8, u8, i8)> = None;

        // Handle MOVE_DROP separately -- it has no origin piece on the board. TC5: nor has a
        // buy (`MOVE_BUY`), which places what arrives the same way and spends coins, not a
        // hand slot (the purse is written below, beside the bounty, under its log).
        let (moving, piece_type) = if enters_play(kind) {
            self.write_entry(m, player, &mut hand)
        } else {
            let mv = self.board[origin];
            (mv, t.type_of(mv))
        };
        let target_before = if enters_play(kind) { 0 } else { self.board[target] };
        // Two kinds end on an occupied square without taking what stands there, and
        // saying so here is what keeps the hand, the explosion, the material clock and
        // the strike rule out of both. A rotation's target *is* its origin, so the square
        // holds the piece that is turning; a file-based castling can land the king on its
        // own rook, which the castling branch below moves rather than removes.
        // TC6: a hire's or a bribe's target is the piece bought where it stands, so it takes
        // nothing either.
        let captured = if kind == MOVE_ROTATE || kind == MOVE_CASTLE || kind == MOVE_HIRE {
            0
        } else {
            target_before
        };
        let mut played = Played {
            origin,
            target,
            kind,
            player,
            moving,
            piece_type,
            target_before,
            captured,
            captured_square: target,
            rook: None,
            struck: None,
            placed: 0,
            contact_changed_mover: false,
            // Schema 18 (Q786): whether the mover kept its origin. A strike placed `Stay`
            // leaves the attacker where it was, so its capture takes nothing off a goal.
            stayed: false,
            // TC6: what a hire or a bribe spends, read off whose the piece was before it
            // changes sides (written into the purse below, beside a buy's price and the
            // bounty).
            spent: 0,
            hand,
            gated: false,
        };

        if kind == MOVE_HIRE {
            // TC6 (`docs/plan-turncoat.md` §5.5): the whole of the move. The piece stays where
            // it stands and becomes the mover's through the one change of side
            // (`rehome_value`, which keeps a facing piece's heading, §6); `write` keys it.
            let purse = t.purse.as_deref().expect("a hire without a purse");
            played.spent = purse.cost(piece_type, t.owner_of(moving), player);
            self.write(origin, t.rehome_value(moving, player, origin));
        } else if kind == MOVE_ROTATE {
            // The whole of the move. `write` puts the change into the key as it happens,
            // and a piece value that carries a facing has its own Zobrist row -- so two
            // positions that differ only in which way a piece points are different
            // positions, and the repetition test can see that they are.
            // Q1007c: the turning piece's owner, which a commanded turn is not the mover.
            self.write(
                origin,
                t.piece_value(t.owner_of(moving), move_promotion(m) as usize, piece_type),
            );
        } else if kind == MOVE_CASTLE {
            self.write_castle(m, &mut played);
        } else if !enters_play(kind) {
            self.write_step(m, &mut played, cleared);
            self.write_strike(m, &mut played);
            self.write_blast(&played, cleared);
        }
        played
    }

    /// A drop's or a buy's square, and the hand slot a drop spends: what stands on the
    /// target afterwards and the type that entered.
    #[inline(always)]
    fn write_entry(
        &mut self,
        m: u32,
        player: usize,
        hand: &mut Option<(u8, u8, i8)>,
    ) -> (u8, usize) {
        let t = self.t;
        let target = move_to(m);
        let pt = move_promotion(m) as usize;
        // What stands on the square afterwards is the table's business (schema 9's
        // `promote_on_entry`): a pawn dropped on the far rank is a queen there, or a
        // tokin in shogi. The hand still gives up `pt`, the dropped type, and
        // `Undo.hand` records `pt` rather than the board value, so `unmake` needs
        // nothing -- and neither does the move word, whose promotion field carries
        // the dropped type.
        let value = t.arriving_value(player, pt, target);
        self.write(target, value);
        if move_kind(m) == MOVE_DROP {
            // The slot the drop spends: `pt` itself, or under schema 18's `either_face`
            // (Q785) the base whose other face `pt` is. `Undo.hand` records the slot, so
            // `unmake` gives back what was spent.
            let slot = t.drops.as_ref().map_or(pt, |rule| rule.slot_of(pt));
            // Remove from hand -- unless the slot is a supply rather than a count
            // (schema 10, Q51). An unlimited slot is never spent, so it records no undo
            // delta either, which is the whole of what `unmake` needs to know about it.
            if !t.drops.as_ref().is_some_and(|rule| rule.is_unlimited(slot)) {
                self.hands[player][slot] -= 1;
                *hand = Some((player as u8, slot as u8, -1));
            }
        }
        (value, pt)
    }

    /// A castling's four writes and the two rooks' `unmoved` bits.
    #[inline(always)]
    fn write_castle(&mut self, m: u32, played: &mut Played) {
        let (origin, target) = (played.origin, played.target);
        // Both pieces leave before either arrives, which is the order
        // `play_move_for_filter` already plays a castling in. Under file-based
        // castling the two squares can overlap -- the king's destination is the
        // rook's own square, the rook's the king's -- and any other order writes one
        // piece over the other. The rook's value is read before the king moves for
        // the same reason.
        let (rook_from, rook_to) =
            self.castling_rook_squares(origin, target, move_promotion(m) as i32);
        let rook_value = self.board[rook_from];
        self.write(origin, 0);
        self.write(rook_from, 0);
        self.write(target, played.moving);
        self.write(rook_to, rook_value);
        played.rook = Some((rook_from, rook_to, rook_value));
        self.unmoved.clear(rook_from);
        self.unmoved.clear(rook_to);
    }

    /// A piece's step from its origin to its target, with what the kind adds there: a
    /// promotion's new type, en passant's victim, a sweep's path, a promoting transit.
    #[inline(always)]
    fn write_step(&mut self, m: u32, played: &mut Played, cleared: &mut Vec<(usize, u8)>) {
        let t = self.t;
        let (origin, target, kind) = (played.origin, played.target, played.kind);
        self.write(origin, 0);
        self.write(target, played.moving);

        if kind == MOVE_PROMOTION {
            self.write(target, t.entering_value(played.player, move_promotion(m) as usize));
        } else if kind == MOVE_EN_PASSANT {
            played.captured_square = self.ep_victim as usize;
            played.captured = self.board[played.captured_square];
            self.write(played.captured_square, 0);
        } else if kind == MOVE_SWEEP {
            // Schema 13: every piece on the path goes, in ray order, onto the same
            // log an explosion writes (`Tables::sweep_victims`). `write` keys each
            // clear as it happens.
            let mut victims = t.sweep_victims(played.moving, origin, target);
            while let Some((square, value)) = victims.next_on(&self.board) {
                cleared.push((square, value));
                self.write(square, 0);
            }
        } else if kind == MOVE_PORTAL && move_promotion(m) != 0 {
            // Schema 19: a transit that promotes at its exit carries the type plus one.
            self.write(target, t.entering_value(played.player, move_promotion(m) as usize - 1));
        } else if kind == MOVE_PUSH {
            self.write_push(m, played);
        }
    }

    /// Q1074b2 (`docs/plan-push.md` K1, K2): a push's own writes, after the pusher has
    /// landed. The pusher promotes by its landing (P1, the type plus one); the line it landed
    /// on shifts as `push_shift_line` says (Q1074b4: one piece under `chain: "one"`, the whole
    /// line under `"row"` and `"weight"`). The piece that falls off is `captured`, from the
    /// square it fell from; when none falls it is 0, and `captured_square` names the sink, so
    /// `unmake`'s generic restore empties it (`push.rs`). Every pushed piece loses its
    /// `unmoved` bit (E1, E3): it stands on a square the move wrote. Under the longer chains
    /// the direction goes on `push_log`, for `unmake` to walk the line's middle back.
    #[inline(never)]
    fn write_push(&mut self, m: u32, played: &mut Played) {
        let t = self.t;
        if move_promotion(m) != 0 {
            self.write(
                played.target, t.entering_value(played.player, move_promotion(m) as usize - 1),
            );
        }
        let (captured, captured_square) =
            self.push_shift_line(m, played.target_before, |game, square, value| {
                game.write(square, value);
                game.unmoved.clear(square);
            });
        played.captured = captured;
        played.captured_square = captured_square;
        if t.push.as_deref().is_some_and(|push| push.chain != PUSH_CHAIN_ONE) {
            self.push_log.push(move_push(m) as u8);
        }
    }

    /// Strike (schema 6): a taken piece may be transformed rather than removed.
    /// Movegen never got here -- the move was generated, blocked and permitted
    /// exactly as a capture -- and the only question left is where the two pieces
    /// stand afterwards, which `strike_writes` answers for all three callers.
    #[inline(always)]
    fn write_strike(&mut self, m: u32, played: &mut Played) {
        let t = self.t;
        let landing_index = move_landing(m);
        if let Some(outcome) = self.strike_writes(
            played.piece_type,
            played.captured,
            played.origin,
            played.target,
            played.captured_square,
            landing_index,
        ) {
            for &(square, value) in outcome.writes() {
                self.write(square, value);
            }
            played.struck = outcome.survivor;
            played.contact_changed_mover = outcome.mover_material_changed;
            played.stayed =
                t.strike.as_ref().is_some_and(|rule| rule.position == StrikePosition::Stay);
            if played.struck.is_some() {
                // Both positions that place a survivor off the three squares the
                // generic restore knows mark it here: `choose` with its landing
                // index, schema 17's `home` with 1 (`unmake` reads the square back off
                // the rebirth table, which `Undo` already has the keys to).
                match t.strike.as_ref().map(|rule| rule.position) {
                    Some(StrikePosition::Choose) => played.placed = landing_index as u8 + 1,
                    Some(StrikePosition::Home) => played.placed = 1,
                    _ => {}
                }
            }
        }
    }

    /// Explosion: captures destroy neighbourhood.
    ///
    /// Schema 10 (Q18) asks `fires` first: `by` names the capturer types whose
    /// captures explode and `of` the victim types whose being taken does, and both
    /// empty is atomic. `piece_type` is the mover as it stood *before* the move,
    /// so a pawn that captures and promotes is a pawn's capture -- which is what
    /// "a capture by P" has to mean if it is to be readable.
    #[inline(always)]
    fn write_blast(&mut self, played: &Played, cleared: &mut Vec<(usize, u8)>) {
        let t = self.t;
        let target = played.target;
        if let Some(explosion) = t.explosion.as_ref() {
            if played.captured != 0 && explosion.fires(played.piece_type, t.type_of(played.captured))
            {
                // Capturing piece goes up
                cleared.push((target, self.board[target]));
                self.write(target, 0);
                // Neighbourhood
                for &sq in explosion.reach(target) {
                    let value = self.board[sq];
                    if explosion.clears(t, value) {
                        cleared.push((sq, value));
                        self.write(sq, 0);
                    }
                }
            }
        }
    }

    /// Q1059a (`drops.gating`): the held piece enters on the square the move vacated, the
    /// origin or a castle's rook origin, spent from the hand as a drop spends it. After
    /// the move, so it stands on an emptied square; `unmake`'s restore of that square
    /// writes over it. A gating hand never fills from captures (the validator), so this
    /// is the move's one hand entry. One mask test on every move of every game.
    /// The tag is a fold end in a fold game, so the rule is asked as well as the bits.
    #[inline(always)]
    fn write_gate(&mut self, m: u32, played: &mut Played) {
        let t = self.t;
        played.gated =
            move_gate(m) != 0 && t.drops.as_ref().is_some_and(|rule| rule.gates.is_some());
        if played.gated {
            let player = played.player;
            let pt = move_gate(m) - 1;
            let square = if move_gate_rook(m) {
                played.rook.expect("a rook-square gate on a move that is not a castle").0
            } else {
                played.origin
            };
            self.write(square, t.arriving_value(player, pt, square));
            let rule = t.drops.as_ref().expect("checked above");
            let slot = rule.slot_of(pt);
            if !rule.is_unlimited(slot) {
                self.hands[player][slot] -= 1;
                debug_assert!(played.hand.is_none(), "a move changed two hand entries");
                played.hand = Some((player as u8, slot as u8, -1));
            }
        }
    }

    /// The `unmoved` bits the move spends (and the one a reborn piece gets back).
    #[inline(always)]
    fn write_unmoved(&mut self, played: &Played) {
        let t = self.t;
        self.unmoved.clear(played.origin);
        self.unmoved.clear(played.target);
        // A struck piece that survived has been through something, so it is no longer
        // unmoved. Origin and target are already cleared above and cover `Stay` and
        // `Swap` on an ordinary capture; this catches the one square they do not, which
        // is an en passant victim standing off both of them.
        //
        // Schema 17 (Q540): a piece reborn on its home square is the opposite case. It is
        // back where the setup stood it, as if it had never moved, so it gets its unmoved
        // bit -- a pawn may make its first move again, and a rook may castle (orthodox
        // circe). The key does not carry `unmoved`; `rights_changed` in `make` sees the
        // bit go up and clears the repetition window, as it does when one goes down.
        if let Some((square, _)) = played.struck {
            if played.placed != 0
                && t.strike.as_ref().is_some_and(|rule| rule.position == StrikePosition::Home)
            {
                self.unmoved.set(square);
            } else {
                self.unmoved.clear(square);
            }
        }
    }

    /// The en passant window: closed by every move, opened by a double step.
    #[inline(always)]
    fn write_ep_window(&mut self, m: u32, played: &Played) {
        let t = self.t;
        let kind = played.kind;
        self.ep_target = NO_SQUARE;
        self.ep_victim = NO_SQUARE;
        if !enters_play(kind)
            && t.ep_type >= 0
            && played.piece_type == t.ep_type as usize
            && played.captured == 0
            && kind != MOVE_PROMOTION
            && kind != MOVE_PORTAL
            && kind != MOVE_HIRE
            // Q1074b2: a push opens no window, the pusher's (it took the square in front) nor
            // the pushed piece's (it made no double step, E1).
            && kind != MOVE_PUSH
            // Schema 22: a move that went through a fold opens no window (rule 9). The
            // extension is empty on every other move that gets here: a strike's landing
            // index rides only on a capture. Q1007c: a commanded move carries bit 30
            // (`MOVE_FOREIGN`), so it opens none either (`docs/archive/plan-schema-31.md` §3.2).
            // Q1059a: a gate rides in the extension too, and a gating double step opens the
            // window as the plain one does.
            && !move_extended(if played.gated { without_gate(m) } else { m })
        {
            // The square stepped over is a table fact, not arithmetic on the delta:
            // where the board's opposite edges are joined, a move's raw delta is not
            // the distance it travelled, and the midpoint of the wrong delta is a
            // square the piece never crossed.
            if let Some(passed) =
                t.ep_passed(t.frame_of_value(played.moving), played.origin, played.target)
            {
                self.ep_target = passed as i16;
                self.ep_victim = played.target as i16;
            }
        }
    }

    /// The payments: the bounty into `points`, and the purse (a buy's price, a hire's or a
    /// bribe's cost, the bounty again) under its log and its key.
    ///
    /// The tally (schema 9), before the phase and after `captured` is known: a player
    /// whose move takes a wild piece is paid the bounty of the route it walked, and
    /// (TC4) one whose move takes any piece but their own is paid its type's bounty
    /// (`Tables::bounty_paid`). A phase step never awards anything -- a boar eating a
    /// boar is refused by the rule, and a boar eating a player's piece is nobody's gain.
    ///
    /// **Not logged.** `unmake` reads the award back off `captured`, `captured_square`
    /// and the side that moved, which `Undo` already carries -- one statement of what
    /// a bounty is, asked twice, rather than a second copy of the number stored beside
    /// the first. Measured, not tidied: an `Option<(usize, f64)>` here cost 24 bytes
    /// on a struct `make` writes at every node of every search, and chess -- which has
    /// no wild piece and never reads it -- paid for it in perft.
    ///
    /// TC4: under a purse the same payment also fills the payee's purse, capped at
    /// `PURSE_MAX`. That one *is* logged (`purse_log`, only under the rule), because
    /// the cap loses what passes it and a read-back could not find it again.
    #[inline(always)]
    fn write_payments(&mut self, played: &Played) {
        let t = self.t;
        let player = played.player;
        let paid = if t.pays && played.captured != 0 {
            t.bounty_paid(played.captured, played.captured_square, player)
        } else {
            0.0
        };
        // TC9c (§5.10): the payee is the taker, or under `bounty_to: "loser"` the taken
        // piece's owner (`WILD`: nobody's piece, no one is paid). Asked only for a payment.
        let payee = if paid != 0.0 { t.bounty_payee(played.captured, player) } else { player };
        if paid != 0.0 && payee < 2 {
            self.points[payee] += paid;
        }
        if let Some(purse) = t.purse.as_deref() {
            self.purse_log.push(self.purse);
            let before = self.purse[player];
            let mut after = before;
            // TC5: a buy spends its type's price; movegen offered it only when the purse held
            // it, and a buy takes nothing, so nothing is paid on the same move.
            if played.kind == MOVE_BUY {
                after -= purse.price[played.piece_type];
            }
            // TC6: a hire or a bribe spends what `PurseTables::cost` said, read above.
            after -= played.spent;
            if after != before {
                self.working_key ^= purse.key(player, before) ^ purse.key(player, after);
                self.purse[player] = after;
            }
            if paid > 0.0 && payee < 2 {
                let before = self.purse[payee];
                let after = (before + paid as u16).min(PURSE_MAX);
                if after != before {
                    self.working_key ^= purse.key(payee, before) ^ purse.key(payee, after);
                    self.purse[payee] = after;
                }
            }
        }
    }

    /// Drops: captured pieces go to captor's hand.
    #[inline(always)]
    fn deposit_capture(&mut self, played: &mut Played) {
        let t = self.t;
        if let Some(ref drops_rule) = t.drops {
            if played.kind == MOVE_DROP {
                // Already handled above (hand decremented)
            } else if played.captured != 0
                && drops_rule.from_captures
                // Q1074b3 (C2): a piece the mover pushed off its own side reaches no hand,
                // as a swept friendly piece does not. Every other capture takes an enemy.
                && !(played.kind == MOVE_PUSH && t.owner_of(played.captured) == played.player)
            {
                let demoted = t.demotions[t.type_of(played.captured)];
                // An unlimited slot takes no deposit either (schema 10, Q51). Taking a
                // stone does not give you a stone you already always have, and leaving
                // the slot constant is what makes "the hand is out of the Zobrist key"
                // still true of a supply rather than merely still safe.
                if !drops_rule.is_unlimited(demoted) {
                    self.hands[played.player][demoted] += 1;
                    debug_assert!(played.hand.is_none(), "a move changed two hand entries");
                    played.hand = Some((played.player as u8, demoted as u8, 1));
                }
            }
        }
    }

    /// Check limit tracking: one more check to the mover's count if the move gave one.
    #[inline(always)]
    fn count_check(&mut self, player: usize) {
        let t = self.t;
        if t.check_limit > 0 {
            let opponent = 1 - player;
            let delivered = if t.multi_royal {
                self.royal_squares(opponent).iter().any(|&royal| self.attacks_any(player, royal))
            } else {
                let royal = self.royal_square(opponent);
                royal >= 0 && self.attacks_any(player, royal as usize)
            };
            if delivered {
                self.checks[player] += 1;
            }
        }
    }

    /// Territory: the moving side claims the square it lands on. A repainting rule
    /// takes it whatever was there; a permanent one only claims an unclaimed square,
    /// which is what makes the colouring monotone and the Zobrist term unnecessary.
    /// Returns `Undo.coloured`.
    #[inline(always)]
    fn write_territory(&mut self, played: &Played, key: &mut u64) -> Option<(usize, u8)> {
        let t = self.t;
        let (player, target) = (played.player, played.target);
        let mut coloured = None;
        // TC6: a hire or a bribe moves no piece onto anything, so it claims no square.
        if let Some(rule) = t.territory.as_ref().filter(|_| played.kind != MOVE_HIRE) {
            let want = (player + 1) as u8;
            let before = self.colour[target];
            if before != want && (rule.repaints || before == 0) {
                coloured = Some((target, before));
                self.colour[target] = want;
                self.colour_counts[player] += 1;
                if before != 0 {
                    self.colour_counts[before as usize - 1] -= 1;
                }
                if rule.repaints {
                    *key ^= t.zobrist_colour(before, target);
                    *key ^= t.zobrist_colour(want, target);
                }
            }
        }
        coloured
    }

    /// Schema 18 (Q786): win by leaving. Read off the move's origin and the mover as it
    /// set out, never off the board, which the phase may have changed since; the ending
    /// reads the field in reach's slot. Every ruleset without the rule skips this on
    /// one flag, and never writes the field or its log.
    #[inline(always)]
    fn write_goal_left(&mut self, played: &Played) {
        let t = self.t;
        if t.reach_leaves {
            let (player, origin, kind) = (played.player, played.origin, played.kind);
            self.goal_left_log.push(self.goal_left_by);
            // A drop or a buy (TC5) leaves nothing: neither has a piece on its origin.
            // TC6: nor does a hire or a bribe, which moves no piece.
            let set_out = !(played.stayed || enters_play(kind) || kind == MOVE_HIRE);
            let left = set_out
                && kind != MOVE_ROTATE
                && !(kind == MOVE_CASTLE && played.target == origin)
                && t.reach.as_ref().is_some_and(|rule| {
                    rule.squares[player][origin]
                        && rule.admits(played.piece_type)
                });
            self.goal_left_by = if left { player as i8 } else { -1 };
        }
    }

    /// Schema 20 (Q807): the straight-back bar. Set by a barring end's own move to
    /// (where it stands now, where it came from), cleared by every other move, and
    /// keyed. Every game without the rule skips this on one flag. Q883: set only when
    /// the player to move next may move that end (`sets_bar`); `remaining` is not yet
    /// spent, so the same player moves again exactly when more than one is left.
    #[inline(always)]
    fn write_portal_bar(&mut self, played: &Played, key: &mut u64) {
        let t = self.t;
        if let Some(portals) = t.portals.as_deref().filter(|portals| portals.has_no_back) {
            let player = played.player;
            self.portal_bar_log.push(self.portal_bar);
            let next = if t.turn.is_some() && self.remaining > 1 { player } else { 1 - player };
            let bar = if played.kind == MOVE_NORMAL && portals.sets_bar(played.piece_type, next) {
                [played.target as i16, played.origin as i16]
            } else {
                NO_BAR
            };
            *key ^= portals.bar_key(t.n_squares, self.portal_bar)
                ^ portals.bar_key(t.n_squares, bar);
            self.portal_bar = bar;
        }
    }

    /// Schema 29 (Q983a, `docs/plan-rest.md` §3.2-§3.4): the end of the player's turn (a
    /// long turn is refused beside rest). The piece that made the move stands where the
    /// move ended: its origin for a turn in place or a strike that stays, else the
    /// target; one blown up or eaten in the phase stands nowhere, which the board says.
    #[inline(always)]
    fn write_rest(&mut self, m: u32, played: &Played, key: &mut u64) {
        if self.t.rest.is_some() {
            self.rest_log.push(self.resting);
            let kind = played.kind;
            // Q1074b8 (N5): every piece the push moved rests no more.
            if kind == MOVE_PUSH {
                *key ^= self.rest_unrest_line(m, played.captured_square);
            }
            let end = if kind == MOVE_ROTATE || played.stayed { played.origin } else { played.target };
            // TC6: a hire or a bribe moves no piece, so nothing enters a rest; a bribed piece's
            // own rest ends because its square no longer holds the code that rested there.
            *key ^= self.rest_after_turn(played.player, (kind != MOVE_HIRE).then_some(end));
        }
    }

    /// The ply's end, after `Undo` is pushed: the turn's move spent, the key set, the
    /// no-progress clock, the side and the turn counter, the history and its window.
    #[inline(always)]
    fn end_ply(
        &mut self,
        player: usize,
        key: u64,
        previous_remaining: u32,
        irreversible: bool,
        progress: bool,
    ) {
        let t = self.t;
        // One move of the turn is spent. A ruleset without a turn rule spends the whole
        // turn, which is what makes the flip below unconditional for every game so far.
        self.remaining = if t.turn.is_some() { self.remaining - 1 } else { 0 };

        self.key = key;
        self.halfmove = if progress { 0 } else { self.halfmove + 1 };
        // The side flips when the turn runs out of moves, which for every ruleset
        // without a turn rule is every move -- `remaining` was set to zero above.
        if self.remaining == 0 {
            self.side = 1 - player;
            self.turns_taken += 1;
            self.remaining = t.turn.as_ref().map_or(1, |rule| rule.length_of(self.turns_taken));
        }
        // The key was computed with the side flipped unconditionally, which is right
        // exactly when it did flip. Undo it when the same player moves again, and swap
        // in the counter's own term either way.
        if self.side == player {
            self.key ^= t.zobrist_side;
        }
        self.key ^= t.zobrist_remaining(previous_remaining);
        self.key ^= t.zobrist_remaining(self.remaining);

        self.ply += 1;
        self.history.push(self.key);
        if irreversible {
            self.window = self.history.len() - 1;
        }
        // A different position, so the check bits are about one that is no longer on
        // the board. The royal hint survives on purpose: it is validated before it is
        // believed, and a royal usually stands where it stood.
        self.check_memo.set([-1, -1]);
    }

    pub fn unmake(&mut self) {
        let undo = self.undo.pop().expect("unmake without a matching make");
        self.history.pop();
        if (MOVE_FOLD..=MOVE_DUCK).contains(&undo.kind) {
            if undo.kind == MOVE_FOLD {
                self.unmake_fold(&undo);
            } else {
                self.unmake_duck(&undo);
            }
            return;
        }

        // The phase, walked back before the player's move (schema 9). Newest step first,
        // and the wild piece's own value is read off the square it stands on now --
        // there is nowhere else it could be.
        if self.t.has_npcs {
            self.unmake_phase(undo.npc_log as usize);
        }

        // Restore explosion-cleared squares first (in reverse)
        for &(sq, value) in undo.cleared.iter().rev() {
            self.put(sq, value);
        }

        if let Some((rook_from, rook_to, rook_value)) = undo.rook {
            // `make`'s order run backwards: both destinations are emptied before either
            // origin is written back, so an overlap under file-based castling restores
            // rather than erases. A castling takes nothing, so the generic restore's
            // `target_before` and `captured` have no part in it.
            self.put(undo.target, 0);
            self.put(rook_to, 0);
            self.put(undo.origin, undo.moving);
            self.put(rook_from, rook_value);
        } else {
            // A survivor placed under `choose` stands on a square the generic restore does
            // not know. Cleared first, so one placed on the origin is overwritten by the
            // mover coming back.
            if undo.placed != 0 {
                let t = self.t;
                let rule = t.strike.as_ref().expect("a placed survivor without a strike rule");
                let placed_on = if rule.position == StrikePosition::Home {
                    // The square `strike_writes` read, asked again of the same keys: the
                    // victim as it stood, and what the strike table turned it into.
                    let cell = t.type_of(undo.moving) * rule.n_types + t.type_of(undo.captured);
                    let becomes = rule.to[cell] as usize;
                    rule.rebirth_square(t.owner_of(undo.captured), becomes, undo.captured_square)
                        as usize
                } else {
                    rule.landing(t.owner_of(undo.moving), undo.captured_square)
                        [undo.placed as usize - 1] as usize
                };
                self.put(placed_on, 0);
            }
            // Q1074b4 (K2): a row's middle, walked back before the generic restore writes
            // its two ends.
            if undo.kind == MOVE_PUSH
                && self.t.push.as_deref().is_some_and(|push| push.chain != PUSH_CHAIN_ONE)
            {
                let direction =
                    self.push_log.pop().expect("push log shorter than the undo stack");
                self.push_unshift(direction as usize, undo.target, undo.captured_square);
            }
            self.put(undo.target, undo.target_before);
            self.put(undo.captured_square, undo.captured);
            if !enters_play(undo.kind) {
                self.put(undo.origin, undo.moving);
            }
        }

        self.unmoved = undo.unmoved;
        self.ep_target = undo.ep_target;
        self.ep_victim = undo.ep_victim;
        self.halfmove = undo.halfmove;
        self.key = undo.key;
        self.window = undo.window;
        self.checks = undo.checks;
        if let Some((player, piece_type, delta)) = undo.hand {
            let slot = &mut self.hands[player as usize][piece_type as usize];
            *slot = slot.wrapping_add_signed(-i32::from(delta));
        }
        if let Some((square, before)) = undo.coloured {
            let want = self.colour[square];
            self.colour[square] = before;
            self.colour_counts[want as usize - 1] -= 1;
            if before != 0 {
                self.colour_counts[before as usize - 1] += 1;
            }
        }
        // `turns_taken` increments exactly when the side flipped, which is the only
        // reliable witness once a turn can be longer than one move: `remaining` changes
        // on every move whether the side flipped or not.
        if self.turns_taken != undo.turns_taken {
            self.side = 1 - self.side;
        }
        self.remaining = undo.remaining;
        self.turns_taken = undo.turns_taken;
        // The bounty, taken back (schema 9). `self.side` is the side that made this move
        // again by here -- the flip above is undone from `turns_taken`, which is the only
        // reliable witness -- and what the move was paid is the same question `make`
        // asked of the same two fields.
        if self.t.pays && undo.captured != 0 {
            let paid = self.t.bounty_paid(undo.captured, undo.captured_square, self.side);
            if paid != 0.0 {
                // TC9c: the same payee `make` paid.
                let payee = self.t.bounty_payee(undo.captured, self.side);
                if payee < 2 {
                    self.points[payee] -= paid;
                }
            }
        }
        // TC4: the purse, from its log (the key came back with `undo.key`).
        if self.t.purse.is_some() {
            self.purse = self.purse_log.pop().expect("purse log shorter than the undo stack");
        }
        if self.t.push.as_deref().is_some_and(PushTables::no_back) {
            self.push_bar = self.push_bar_log.pop().expect("push bar log shorter than the undo stack");
        }
        if self.t.reach_leaves {
            self.goal_left_by =
                self.goal_left_log.pop().expect("goal log shorter than the undo stack");
        }
        if self.t.portals.as_deref().is_some_and(|portals| portals.has_no_back) {
            self.portal_bar =
                self.portal_bar_log.pop().expect("bar log shorter than the undo stack");
        }
        if self.t.rest.is_some() {
            self.resting = self.rest_log.pop().expect("rest log shorter than the undo stack");
        }
        self.ply -= 1;
        self.check_memo.set([-1, -1]);
    }
}

impl<'t> Game<'t> {
    /// Schema 22: stand fold end `end` on `square` (or nowhere, -1), keeping the set beside
    /// it. Not the key: `make_fold` keys the move.
    #[inline]
    fn set_fold(&mut self, end: usize, square: i16) {
        self.fold_at[end] = square;
    }

    /// Schema 22: play a fold's own move (`MOVE_FOLD`). The fold moves; nothing on the
    /// board byte does, so no piece is taken or moved and no `unmoved` bit changes. It is
    /// reversible (it neither clears the repetition window nor resets the no-progress
    /// clock), as a momentum end's move is, and it is keyed. The rest is `make`'s
    /// bookkeeping for a move that writes no square: the en passant window closes, a check
    /// it uncovers is counted, the straight-back bar is set or cleared, the turn passes.
    #[cold]
    #[inline(never)]
    fn make_fold(&mut self, m: u32) {
        let t = self.t;
        let origin = move_from(m);
        let target = move_to(m);
        let end = move_fold(m);
        let player = self.side;
        let portals = t.portals.as_deref().expect("a fold move without portals");
        let folds = portals.folds.as_ref().expect("a fold move without folds");
        let previous_ep = self.ep_target;
        let previous_ep_victim = self.ep_victim;
        let previous_checks = self.checks;
        let previous_remaining = self.remaining;
        let previous_turns = self.turns_taken;
        let mut key = self.key ^ t.zobrist_side;
        key ^= folds.key(t.n_squares, end, origin) ^ folds.key(t.n_squares, end, target);
        self.set_fold(end, target as i16);
        if previous_ep != NO_SQUARE {
            key ^= t.zobrist_ep_file(previous_ep);
        }
        self.ep_target = NO_SQUARE;
        self.ep_victim = NO_SQUARE;
        if t.check_limit > 0 {
            let opponent = 1 - player;
            let delivered = if t.multi_royal {
                self.royal_squares(opponent).iter().any(|&royal| self.attacks_any(player, royal))
            } else {
                let royal = self.royal_square(opponent);
                royal >= 0 && self.attacks_any(player, royal as usize)
            };
            if delivered {
                self.checks[player] += 1;
            }
        }
        if t.reach_leaves {
            self.goal_left_log.push(self.goal_left_by);
            self.goal_left_by = -1;
        }
        if portals.has_no_back {
            self.portal_bar_log.push(self.portal_bar);
            let next = if t.turn.is_some() && self.remaining > 1 { player } else { 1 - player };
            let bar = if portals.sets_bar(portals.first_type + end, next) {
                [target as i16, origin as i16]
            } else {
                NO_BAR
            };
            key ^= portals.bar_key(t.n_squares, self.portal_bar) ^ portals.bar_key(t.n_squares, bar);
            self.portal_bar = bar;
        }
        // Schema 29: a fold's own move is the player's turn, so its rests count down; no
        // piece made it (a portal end never rests), so nothing enters.
        if t.rest.is_some() {
            self.rest_log.push(self.resting);
            key ^= self.rest_after_turn(player, None);
        }
        self.undo.push(Undo {
            origin,
            target,
            kind: MOVE_FOLD,
            moving: 0,
            captured: 0,
            captured_square: target,
            target_before: self.board[target],
            rook: None,
            unmoved: self.unmoved,
            ep_target: previous_ep,
            ep_victim: previous_ep_victim,
            halfmove: self.halfmove,
            key: self.key,
            window: self.window,
            checks: previous_checks,
            hand: None,
            cleared: Vec::new(),
            coloured: None,
            remaining: previous_remaining,
            turns_taken: previous_turns,
            npc_log: self.npc_log.len() as u32,
            placed: 0,
        });
        self.remaining = if t.turn.is_some() { self.remaining - 1 } else { 0 };
        self.key = key;
        self.halfmove += 1;
        if self.remaining == 0 {
            self.side = 1 - player;
            self.turns_taken += 1;
            self.remaining = t.turn.as_ref().map_or(1, |rule| rule.length_of(self.turns_taken));
        }
        if self.side == player {
            self.key ^= t.zobrist_side;
        }
        self.key ^= t.zobrist_remaining(previous_remaining);
        self.key ^= t.zobrist_remaining(self.remaining);
        self.ply += 1;
        self.history.push(self.key);
        self.check_memo.set([-1, -1]);
    }

    /// Schema 22: take back a fold's own move: the fold that stands on the move's target
    /// goes back to its origin, and every field `make_fold` wrote is restored.
    #[cold]
    #[inline(never)]
    fn unmake_fold(&mut self, undo: &Undo) {
        let end = (0..MAX_PORTAL_ENDS)
            .find(|&end| self.fold_at[end] == undo.target as i16)
            .expect("no fold stands where the fold move put it");
        self.set_fold(end, undo.origin as i16);
        self.ep_target = undo.ep_target;
        self.ep_victim = undo.ep_victim;
        self.halfmove = undo.halfmove;
        self.key = undo.key;
        self.window = undo.window;
        self.checks = undo.checks;
        if self.turns_taken != undo.turns_taken {
            self.side = 1 - self.side;
        }
        self.remaining = undo.remaining;
        self.turns_taken = undo.turns_taken;
        if self.t.reach_leaves {
            self.goal_left_by =
                self.goal_left_log.pop().expect("goal log shorter than the undo stack");
        }
        if self.t.portals.as_deref().is_some_and(|portals| portals.has_no_back) {
            self.portal_bar =
                self.portal_bar_log.pop().expect("bar log shorter than the undo stack");
        }
        if self.t.rest.is_some() {
            self.resting = self.rest_log.pop().expect("rest log shorter than the undo stack");
        }
        self.ply -= 1;
        self.check_memo.set([-1, -1]);
    }
}

/// Schema 22: `dedup_segment` for a fold game (`docs/archive/plan-portals-2.md` §3.3). Two words
/// alike but for the fold end they went through (`move_fold`) reach one position, so they
/// are one move: the first occurrence keeps its place and takes the lowest tag, which is
/// the direct move's 0 where there is one, else the lowest fold end's.
fn dedup_fold_segment(out: &mut Vec<u32>, start: usize) {
    const WORD: u32 = MOVE_BASE_MASK;
    let mut write = start;
    for index in start..out.len() {
        let m = out[index];
        if let Some(kept) = out[start..write].iter_mut().find(|kept| **kept & WORD == m & WORD) {
            if move_fold(m) < move_fold(*kept) {
                *kept = m;
            }
            continue;
        }
        out[write] = m;
        write += 1;
    }
    out.truncate(write);
}

/// Drop every repeat of a move already in `out[start..]`, keeping the first occurrence of
/// each in the order it was generated.
///
/// Runs only where the compiler's `dedup` flag says a piece on a square can still reach
/// some target two ways after static resolution -- a rook on a wrapped board, whose two
/// rays round the join meet -- so on most rulesets it never runs at all. **First
/// occurrence order is the observable behaviour**: the surviving list is what every
/// tie-break downstream reads, so nothing here may sort or hash the moves into another
/// order.
///
/// Allocation-free. The moves kept so far *are* `out[start..write]`, so the membership
/// test reads that prefix rather than a second list holding the same words. Two equal
/// moves land on the same square, so a 256-bit mask of the targets kept so far settles
/// the common case -- a target not reached before -- without a scan; the scan runs once
/// per repeated target, which is once per duplicate the flag exists for. The mask is
/// only ever a negative filter: a set bit says "scan", never "drop".
fn dedup_segment(out: &mut Vec<u32>, start: usize) {
    let mut targets = SquareSet::EMPTY;
    let mut write = start;
    for index in start..out.len() {
        let m = out[index];
        let to = move_to(m);
        if targets.get(to) && out[start..write].contains(&m) {
            continue;
        }
        targets.set(to);
        out[write] = m;
        write += 1;
    }
    out.truncate(write);
}

impl<'t> Game<'t> {
    /// Moves that satisfy the piece rules, without the self-check filter.
    ///
    /// Appended to `out` rather than returned: a search reuses one buffer for every
    /// node on the path, which is the difference between one allocation per ply and
    /// one per node.
    pub fn generate_pseudo_legal(&self, out: &mut Vec<u32>) {
        // Generation 42 (N4b): a duck half lists the duck's moves alone.
        if self.duck_owed() {
            return self.push_duck_moves(out);
        }
        // Schema 22: one dispatch per call to a copy monomorphised on whether the game has
        // folds, so chess and every game without them run the walks they always ran
        // (`docs/archive/plan-portals-2.md` §3.4).
        if self.t.portals.as_deref().is_some_and(|portals| portals.folds.is_some()) {
            return self.generate_folded(out);
        }
        // Schema 30 (Q999a, `docs/archive/plan-passive.md` §3.4): the sight's filter binds every
        // caller of the generator; one null test in a game without one. Portals (and so
        // folds) are refused beside a sight, so the fold copy above never needs it.
        let base = out.len();
        self.generate_pseudo_legal_as::<false>(out);
        if let Some(sight) = self.t.sight.as_deref() {
            // Q1007c (`docs/archive/plan-schema-31.md` §3.7): the opponent's commanded pieces are a
            // second source of moves, filtered with the rest; TC3's held neutrals a third.
            if sight.has_command || self.t.majority.is_some() {
                self.push_foreign(out);
            }
            self.sight_filter(sight, out, base);
        } else if self.t.majority.is_some() {
            // TC3 (`docs/plan-turncoat.md` §4.3): the neutrals the side to move holds.
            self.push_foreign(out);
        }
    }

    /// Schema 22: `generate_pseudo_legal` for a game with folds, out of line so the copy
    /// every other game runs is the one inlined.
    #[cold]
    #[inline(never)]
    fn generate_folded(&self, out: &mut Vec<u32>) {
        self.generate_pseudo_legal_as::<true>(out)
    }

    #[inline(always)]
    fn generate_pseudo_legal_as<const FOLDS: bool>(&self, out: &mut Vec<u32>) {
        let t = self.t;
        let player = self.side;
        let folded = if FOLDS { t.portals.as_deref() } else { None };
        // Q1059a: where this call's moves begin, for gating to read back (`push_gates`).
        let base = out.len();
        // Schema 18 (Q784): the placement phase. While the side to move holds anything in
        // hand, a drop is its only move (Fairy-Stockfish's `mustDrop`). One `Option` test
        // for a ruleset with no hand, which is the price every other game pays.
        if let Some(rule) = t.drops.as_ref() {
            if rule.placement && self.hands[player].iter().any(|held| *held > 0) {
                self.generate_drops(out);
                return;
            }
        }
        let low = t.value_low(player);
        let high = low + t.side_stride as u16;
        let ep_target = self.ep_target;
        let ep_victim = self.ep_victim;
        // Schema 3's capture matrix binds the en passant arm exactly as it binds every
        // other capture, and this is the one arm that cannot ask the square it moves to:
        // an en passant capture lands on an **empty** square and takes the piece standing
        // on `ep_victim`, so the three `may_take` tests below -- each reading the occupant
        // of the *target* -- see an empty square, take the `occupant == 0` branch, and let
        // it through. Without this line `captures: []` on every piece still takes a piece
        // in passing, which made `spec.schema::_capture_is_possible` false about the game:
        // it reads the same matrix and is what refuses `must_capture`,
        // `explode_on_capture` and `drops` as declaring nothing.
        //
        // Asked once for the position rather than once per piece: `capturable_ep` below
        // already requires the mover to *be* `ep_type`, so the attacker is the same type
        // at every origin and the answer cannot differ between them. The two squares are
        // written, cleared and restored as a pair everywhere in this file; the second test
        // is for a position handed in from outside that set only one of them. All three
        // arms share this one test, so the hop arm's branch (Q121) is bound by the matrix
        // on arrival rather than being the fourth arm this comment warns about.
        //
        // Q531: and the victim must not be the mover's own. In one-move turns the side
        // that opened the window has always handed the move over, so this never bound;
        // in a long turn the window a double step opens is still open on the same
        // player's next move, and without it a pawn took its own pawn in passing.
        let ep_occupant = if ep_victim != NO_SQUARE {
            self.board[ep_victim as usize] as u16
        } else {
            0
        };
        let ep_permitted = ep_target != NO_SQUARE
            && ep_victim != NO_SQUARE
            && t.ep_type >= 0
            && !(low <= ep_occupant && ep_occupant < high)
            && may_take(t, t.ep_type as usize, ep_occupant);
        // Schema 12: a capture under `position: "choose"` fans out over its landing squares.
        let choosing =
            t.strike.as_ref().is_some_and(|rule| rule.position == StrikePosition::Choose);

        // The mover's pieces, in ascending square order -- the order the scan this
        // replaced visited them in, and therefore the order their moves come out in.
        for origin in self.occupied()[player].iter() {
            let raw = self.board[origin];
            debug_assert!(low <= raw as u16 && (raw as u16) < high);
            // A *frame*, not a player. What a piece may do depends on which way is
            // forward and, since schema 8, on which way the piece itself is pointing --
            // so the answer comes off the piece's own board value and not off whose
            // turn it is.
            let (piece_type, frame) = t.decode_owned(raw, low, player);
            let origin_unmoved = self.unmoved.get(origin);
            let promoting = t.is_promoting(piece_type);
            let capturable_ep = ep_permitted && piece_type == t.ep_type as usize;
            let slot = t.slot(frame, piece_type, origin);
            let ambiguous = t.dedup[slot];
            let segment_start = out.len();
            let promo_optional = promoting && t.promotion_optional(piece_type);
            // Schema 15: under `on_leaving` a move that starts in the zone promotes
            // wherever it lands, so the origin is asked once here and every target of
            // this piece reads the answer. False for every ruleset before it.
            let leaving = promoting
                && t.promotes_on_leaving(piece_type)
                && t.promotes_on(player, piece_type, origin);

            for entry in t.leaps.entries(slot) {
                if entry.requires_unmoved && !origin_unmoved {
                    continue;
                }
                let squares = t.leaps.squares(entry);
                let modes = t.leaps.modes(entry);
                for index in 0..squares.len() {
                    let target = squares[index] as usize;
                    let occupant = self.board[target] as u16;
                    let mode = modes[index];
                    if occupant == 0 {
                        if capturable_ep && target as i16 == ep_target && mode & 2 != 0 {
                            self.push_capture(
                                out,
                                encode_move(origin, target, MOVE_EN_PASSANT, 0),
                                piece_type,
                                player,
                                ep_victim as usize,
                                choosing,
                            );
                        }
                        // Schema 24: a landing on an empty fold may also go through.
                        if FOLDS && mode != 0 && self.fold_on(target) {
                            self.push_through(out, origin, target, mode, piece_type, leaving);
                        }
                        if mode & 1 == 0 {
                            continue;
                        }
                    } else {
                        // Q1074b2: a landing that may push offers the push beside (or, its
                        // capture bit off, instead of) the capture. One bit test here for
                        // every game without a push.
                        if mode & MODE_PUSH != 0 {
                            self.push_offered(
                                out, origin, target, piece_type, player, mode, PUSH_ARM_LEAP,
                                entry.start as usize + index, promoting, leaving, promo_optional,
                            );
                        }
                        if !may_land(t, mode, piece_type, occupant, low, high) {
                            continue;
                        }
                    }
                    self.push_arrival(
                        out, origin, target, piece_type, player, promoting, leaving,
                        promo_optional, choosing && occupant != 0,
                    );
                }
            }

            for entry in t.slides.entries(slot) {
                if entry.requires_unmoved && !origin_unmoved {
                    continue;
                }
                let squares = t.slides.squares(entry);
                let modes = t.slides.modes(entry);
                for index in 0..squares.len() {
                    let target = squares[index] as usize;
                    let occupant = self.board[target] as u16;
                    let mode = modes[index];
                    if occupant == 0 {
                        if capturable_ep && target as i16 == ep_target && mode & 2 != 0 {
                            self.push_capture(
                                out,
                                encode_move(origin, target, MOVE_EN_PASSANT, 0),
                                piece_type,
                                player,
                                ep_victim as usize,
                                choosing,
                            );
                        }
                        // Schema 24: a fold is floor to a slide that runs on past it;
                        // one that may end on it may also go through.
                        if FOLDS && mode != 0 && self.fold_on(target) {
                            self.push_through(out, origin, target, mode, piece_type, leaving);
                        }
                        if mode & 1 == 0 {
                            continue;
                        }
                    } else {
                        // Q1074b2: a slider pushes only at its first occupied square, where
                        // a capture would have been (§1.3).
                        if mode & MODE_PUSH != 0 {
                            self.push_offered(
                                out, origin, target, piece_type, player, mode, PUSH_ARM_SLIDE,
                                entry.start as usize + index, promoting, leaving, promo_optional,
                            );
                        }
                        if !may_land(t, mode, piece_type, occupant, low, high) {
                            // An occupied square ends the ray whether or not it can be
                            // taken: a piece this one may not capture blocks it exactly as
                            // a friendly piece with no own contact link does.
                            break;
                        }
                    }
                    self.push_arrival(
                        out, origin, target, piece_type, player, promoting, leaving,
                        promo_optional, choosing && occupant != 0,
                    );
                    if occupant != 0 {
                        break;
                    }
                }
            }

            let piece = PieceAt {
                origin, slot, piece_type, player, low, high, origin_unmoved, promoting, leaving,
                promo_optional, capturable_ep, choosing,
            };
            self.push_hops::<FOLDS>(out, &piece);
            // Schema 13: one test for every game without a sweep.
            if t.has_sweep {
                self.push_sweeps(out, &piece);
            }

            // Schema 24: a go-through to a square also reached directly, or through two
            // folds, is one move (`dedup_fold_segment`); only a segment holding a tagged
            // word can have one.
            if FOLDS && out[segment_start..].iter().any(|&m| move_fold(m) != 0) {
                dedup_fold_segment(out, segment_start);
            } else if ambiguous {
                dedup_segment(out, segment_start);
            }

            // Rotations, after the uniqueness pass rather than inside it: a rotation
            // carries its own move kind, so it can never duplicate a move that goes
            // somewhere. A piece turns to any of its other orientations in one move --
            // three of them for a piece with four axes, one for a piece with two -- and
            // the compiler refuses a facing whose rotation changes nothing, so none of
            // these is a pass. A facing no move turns (schema 25) has one orientation
            // to turn through, so it gets none.
            let axes = if t.has_facing { t.turns_of(piece_type) } else { 1 };
            if axes > 1 {
                let step = t.n_facings / axes;
                let facing = frame % t.n_facings;
                for turn in 1..axes {
                    let to = (facing + turn * step) % t.n_facings;
                    out.push(encode_move(origin, origin, MOVE_ROTATE, to as u32));
                }
            }
        }

        // Schema 19: every move through a portal and every portal's own move, after the
        // ordinary moves of every piece. One null test here is what a game without
        // portals pays: inside the loop, the same test per piece cost chess perft(6) 5%.
        if let Some(portals) = t.portals.as_deref() {
            self.push_portal_moves(out, portals);
        }
        // Schema 22: every fold the side may move, after the momentum ends' moves.
        if let (true, Some(portals)) = (FOLDS, folded) {
            self.push_fold_own_moves(out, portals);
        }

        if t.castling.is_some() {
            self.generate_castling(out);
        }
        if let Some(rule) = t.drops.as_ref() {
            // Q1059a: a gating hand enters only on a square a move vacates, never by a drop.
            if let Some(gates) = rule.gates.as_ref() {
                self.push_gates(out, base, gates);
            } else {
                self.generate_drops(out);
            }
        }
        // TC5 (`docs/plan-turncoat.md` §5.4): buys from the market. One null test for every
        // game without a purse.
        // TC6 (§5.5): hires and bribes, behind the same test.
        if let Some(purse) = t.purse.as_deref() {
            self.generate_buys(out);
            if purse.contested() {
                self.generate_hires(out, purse);
            }
        }
    }

    /// The hop arm of `generate_pseudo_legal_as`: every hop of `piece`, in table order.
    /// A hop vaults the first occupied square on its line (the screen, if its owner
    /// matches) and lands on the empty squares past it up to `screen_range`, or on the
    /// first piece past it that it may take. Its own function so a task that edits hops
    /// edits one body (R20); `#[inline(always)]` keeps it in the movegen loop.
    #[inline(always)]
    fn push_hops<const FOLDS: bool>(&self, out: &mut Vec<u32>, piece: &PieceAt) {
        let t = self.t;
        let PieceAt {
            origin, slot, piece_type, player, low, high, origin_unmoved, promoting, leaving,
            promo_optional, capturable_ep, choosing,
        } = *piece;
        let ep_target = self.ep_target;
        let ep_victim = self.ep_victim;
        for entry in t.hops.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            let squares = t.hops.squares(entry);
            let modes = t.hops.modes(entry);
            let length = squares.len();
            let mut index = 0;
            while index < length && self.board[squares[index] as usize] == 0 {
                index += 1;
            }
            if index >= length {
                continue; // nothing to hop over
            }
            // Schema 3: whose piece the screen is. A hop still stops at the *first*
            // occupied square whatever the answer -- a piece it may not vault blocks
            // the line rather than being skipped over, exactly as a piece it may not
            // capture blocks a slide.
            if !screen_matches(
                entry.screen_owner, self.board[squares[index] as usize] as u16, low, high
            ) {
                continue;
            }
            index += 1;
            // A cannon may land anywhere beyond its screen; a grasshopper only just
            // past it. `screen_range` of zero means the rest of the ray.
            let stop = if entry.screen_range == 0 {
                length
            } else {
                length.min(index + entry.screen_range as usize)
            };
            while index < stop {
                let target = squares[index] as usize;
                let occupant = self.board[target] as u16;
                let mode = modes[index];
                if occupant == 0 {
                    // The third arm, and it read differently from the other two until
                    // Q121: a hop that may capture on the empty square a piece just
                    // crossed may take it there, exactly as a leap or a slide may.
                    // The window this consumes is never one a hop *opened* -- a hop's
                    // passed square is its own screen, so `_ep_passes` builds none --
                    // but a slide's double step opens one a hop can reach.
                    if capturable_ep && target as i16 == ep_target && mode & 2 != 0 {
                        self.push_capture(
                            out,
                            encode_move(origin, target, MOVE_EN_PASSANT, 0),
                            piece_type,
                            player,
                            ep_victim as usize,
                            choosing,
                        );
                    }
                    if FOLDS && mode != 0 && self.fold_on(target) {
                        self.push_through(out, origin, target, mode, piece_type, leaving);
                    }
                    if mode & 1 != 0 {
                        if promoting
                            && (leaving || t.promotes_on(player, piece_type, target))
                        {
                            for &promoted in t.promotion_targets(player, piece_type) {
                                if self.promotion_offered(player, piece_type, promoted) {
                                    out.push(encode_move(
                                        origin, target, MOVE_PROMOTION, promoted,
                                    ));
                                }
                            }
                            if promo_optional {
                                out.push(encode_move(origin, target, MOVE_NORMAL, 0));
                            }
                        } else {
                            out.push(encode_move(origin, target, MOVE_NORMAL, 0));
                        }
                    }
                    index += 1;
                    continue;
                }
                // Q1074b2: a hop pushes the first piece past its screen (D5).
                if mode & MODE_PUSH != 0 {
                    self.push_offered(
                        out, origin, target, piece_type, player, mode, PUSH_ARM_HOP,
                        entry.start as usize + index, promoting, leaving, promo_optional,
                    );
                }
                if may_land(t, mode, piece_type, occupant, low, high) {
                    self.push_arrival(
                        out, origin, target, piece_type, player, promoting, leaving,
                        promo_optional, choosing,
                    );
                }
                break;
            }
        }
    }

    /// The sweep arm of `generate_pseudo_legal_as` (schema 13): every sweep of `piece`,
    /// in table order. The ray does not break at a piece; every piece before a landing is
    /// swept, friendly included, and one the sweeper may not capture ends the ray. A
    /// landing over an empty path is `MOVE_NORMAL`, so the uniqueness pass folds it into
    /// a slide or leap to the same square. What a played sweep removes is
    /// `Tables::sweep_victims`'. Its own function for the same reason as `push_hops`.
    #[inline(always)]
    fn push_sweeps(&self, out: &mut Vec<u32>, piece: &PieceAt) {
        let t = self.t;
        let PieceAt { origin, slot, piece_type, low, high, origin_unmoved, promoting, .. } = *piece;
        for entry in t.sweeps.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            debug_assert!(!promoting, "a sweeper may not be a promoting piece");
            let squares = t.sweeps.squares(entry);
            let modes = t.sweeps.modes(entry);
            // Q1063: from this index on the ray is a slide's (`usize::MAX`: never).
            let rides_from = entry.sweep_length();
            let mut path_holds_piece = false;
            for index in 0..squares.len() {
                let target = squares[index] as usize;
                let occupant = self.board[target] as u16;
                if occupant != 0 && !may_take(t, piece_type, occupant) {
                    break;
                }
                let mode = modes[index];
                let lands = if occupant == 0 {
                    mode & 1 != 0
                } else {
                    may_land(t, mode, piece_type, occupant, low, high)
                };
                if lands {
                    let kind = if path_holds_piece { MOVE_SWEEP } else { MOVE_NORMAL };
                    out.push(encode_move(origin, target, kind, 0));
                }
                if occupant != 0 {
                    if index >= rides_from {
                        break;
                    }
                    path_holds_piece = true;
                }
            }
        }
    }
}

/// What `make`'s board phase leaves for its later phases to read (CS8): the move's
/// squares and kind, the piece as it set out, what it took and where, and the marks the
/// later phases and `Undo` need. Every phase is inlined, so it never reaches memory.
#[derive(Clone, Copy)]
struct Played {
    origin: usize,
    target: usize,
    kind: u32,
    player: usize,
    moving: u8,
    piece_type: usize,
    target_before: u8,
    captured: u8,
    captured_square: usize,
    rook: Option<(usize, usize, u8)>,
    struck: Option<(usize, u8)>,
    placed: u8,
    contact_changed_mover: bool,
    stayed: bool,
    spent: u16,
    hand: Option<(u8, u8, i8)>,
    gated: bool,
}

/// `make`'s two flags about progress: `(irreversible, progress)`, whether the repetition
/// window clears and whether the no-progress clock resets.
#[inline(always)]
fn progress_of(
    t: &Tables,
    m: u32,
    played: &Played,
    phase_ate: bool,
    cleared_any: bool,
    coloured: bool,
    rights_changed: bool,
) -> (bool, bool) {
    let kind = played.kind;
    let piece_type = played.piece_type;
    let captured = played.captured;
    // What the no-progress clock is actually about is *progress*, and until schema 6
    // "a piece was taken" and "the board holds less material than it did" were the
    // same sentence. A strike separates them: a demotion moves material down and is
    // progress, a promotion moves it up, and a pure swap moves nothing at all -- a
    // swap-and-promote game could shuffle for ever if the old test were kept.
    // Resolved as **the board's total declared material changed**, which is
    // objective, is the same quantity `scoring.count: "material"` already reads, and
    // is true of every plain capture -- so nothing before this schema moves.
    //
    // A wild piece that ate in the phase is a material change on the same reading --
    // the board holds less than it did -- so `|| phase_ate` is the whole of schema
    // 9's claim on this test, and on the no-progress clock below it.
    // Schema 10 adds the second half of the same reading: a contact can change what
    // the *attacker* is worth, and a pawn that takes a rook and becomes a knight has
    // moved the board's declared material even though nothing left the board and the
    // struck piece's own worth may not have moved at all.
    // Schema 13: a sweep that took only path pieces changed material too.
    let material_changed = phase_ate
        || played.contact_changed_mover
        || cleared_any
        || match played.struck {
            Some((_, survivor)) => {
                t.piece_values[t.type_of(captured)] != t.piece_values[t.type_of(survivor)]
            }
            None => captured != 0,
        };
    // A permanent colour laid down is a one-way step and has to reset the
    // window, or the repetition test could compare two positions with different
    // colourings and the key -- which deliberately does not carry colour --
    // would call them the same position.
    let laid_colour = coloured && !t.territory.as_ref().is_some_and(|r| r.repaints);
    // A rotation is **reversible**: the piece can turn back, and a game that turns
    // back and forth has to be caught by the repetition rule rather than have its
    // window cleared under it every ply. So it is not on this list -- and it does
    // not reset the no-progress clock below either, for the same reason. What it
    // *does* do is clear the turning piece's `unmoved` bit, which is a right that
    // cannot come back, so the first turn of any piece is irreversible via
    // `rights_changed` and every turn after it is not.
    // A one-way piece can never return to a square it left, so its move is progress
    // however quiet it looks. **This clause used to read `piece_type == ep_type`**,
    // and `ep_type` is set only when `en_passant` is declared -- so *a pawn move is
    // progress*, a fact about chess, was switched on by a spelling of an unrelated
    // capturing rule. Q113 separated them: `one_way` is derived from the movement
    // tables by the compiler, which is where the geometry already is, and this is a
    // flag lookup rather than a rule.
    // Schema 19: a transit is a move like any other and can be walked back through the
    // other end, so it is reversible unless it promoted.
    // Q1074b2 (K4): so is a push that removes nothing and promotes nothing, a step of two
    // pieces either of which may step back; one that removes is a material change.
    let promoted = kind == MOVE_PROMOTION
        || ((kind == MOVE_PORTAL || kind == MOVE_PUSH) && move_promotion(m) != 0);
    let irreversible = material_changed
        || (kind != MOVE_NORMAL
            && kind != MOVE_ROTATE
            && ((kind != MOVE_PORTAL && kind != MOVE_PUSH) || promoted))
        || t.is_one_way(piece_type)
        || rights_changed
        || laid_colour;
    // The no-progress clock asks the same question the window above asks, so it reads
    // the same derived flag. **This clause used to read `t.is_promoting(piece_type)`**
    // -- so *declaring a promotion rule* was what made a piece's every move count as
    // progress, including the quiet ones and including a rotation, which the comment
    // above calls reversible. Q135 separated them (Q113's separation, one key on).
    //
    // `kind == MOVE_PROMOTION` is what the old term carried and keeps: a piece that
    // changes type at the far end has spent itself, and that is progress whether or
    // not the piece could have walked back. It is a fact about the move rather than
    // about the type's rule sheet, which is the whole difference.
    // TC5: a buy adds material to the board (`docs/plan-turncoat.md` §5.4), so it is
    // progress as a capture is; `material_changed` asks only what left the board.
    // TC6: so does a hire or a bribe, which moves material from one army to another.
    // Q1059a: a gating move is not progress by itself. It adds a piece, but it is the
    // move it rides on that the clock reads, as Seirawan's own fifty-move rule (and
    // Fairy-Stockfish's) has it; the window is cleared by the gate's spent right.
    let progress = material_changed
        || kind == MOVE_BUY
        || kind == MOVE_HIRE
        || t.is_one_way(piece_type)
        || promoted
        || laid_colour;
    (irreversible, progress)
}

/// What movegen's loop knows about the piece it is generating for, handed whole to the
/// arms that are their own functions (`push_hops`, `push_sweeps`) rather than as a dozen
/// arguments. Built per piece; both arms are inlined, so it never reaches memory.
#[derive(Clone, Copy)]
struct PieceAt {
    origin: usize,
    slot: usize,
    piece_type: usize,
    player: usize,
    low: u16,
    high: u16,
    origin_unmoved: bool,
    promoting: bool,
    leaving: bool,
    promo_optional: bool,
    capturable_ep: bool,
    choosing: bool,
}

impl<'t> Game<'t> {
    /// Schema 19: every move through a portal of the piece on `origin`, pushed as
    /// `MOVE_PORTAL` words (`docs/archive/plan-portals.md` §2).
    ///
    /// A second walk of the piece's own leaps, slides and hops. The first walk met each
    /// portal end as an uncapturable piece -- the compiler takes every end out of the
    /// capture matrix -- so a slide stopped there and a leap did not land; this walk asks
    /// what the piece would have done had it gone in. An end is entered where the walk
    /// first meets it with a landing mode, and only if it admits this piece's type for the
    /// side to move (schema 21's `admits`); each entry ORs in the landing's momentum
    /// (`compile.momentum_bits`), so a piece that reaches one end along two lines has two
    /// directions to come out in.
    ///
    /// Each direction's exit is the neighbour of the end this one leads to (`dest`) under
    /// the mover's wrap permission, read from the exit table; an end that leads nowhere (a
    /// way out only) admits nobody. It is a move when the mode the piece arrived
    /// with allows what stands there -- empty under a move-mode landing, a piece it may
    /// take under a capture-mode one -- and it is not the mover's own origin (a pass) or
    /// another portal (a chain, refused by the user's answer). Distinct exits are pushed in
    /// ascending order of exit square, then of entry square, each with the promotions
    /// `push_arrival` would give a move ending there.
    ///
    /// Out of line: it runs only in a portal game, and the movegen loop pays one null
    /// test per piece for it.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn push_transits(
        &self, out: &mut Vec<u32>, portals: &PortalTables, at: &[u16; MAX_PORTAL_ENDS], origin: usize,
        slot: usize, piece_type: usize, leaving: bool,
    ) {
        let t = self.t;
        let player = self.side;
        let low = t.value_low(player);
        let high = low + t.side_stride as u16;
        let origin_unmoved = self.unmoved.get(origin);
        // (entry square, momentum) for each end this piece reaches.
        let mut entered = [(0u16, 0u16); MAX_PORTAL_ENDS];
        let mut n_entered = 0;
        let mut enter = |square: usize, bits: u16| {
            let value = self.board[square];
            if bits == 0 || !t.is_portal_end(value) {
                return;
            }
            if portals.admits[t.type_of(value)][player] & (1 << piece_type) == 0 {
                return;
            }
            for item in entered.iter_mut().take(n_entered) {
                if item.0 == square as u16 {
                    item.1 |= bits;
                    return;
                }
            }
            entered[n_entered] = (square as u16, bits);
            n_entered += 1;
        };
        for entry in t.leaps.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            for (index, &square) in t.leaps.squares(entry).iter().enumerate() {
                if self.board[square as usize] != 0 {
                    enter(square as usize, portals.leap_dirs[entry.start as usize + index]);
                }
            }
        }
        // Schema 24: a fold is floor to a ray that crosses it, so a slide or hop may reach
        // a momentum end beyond one (`docs/archive/plan-portals-2.md` §3R.1 rule 7).
        for entry in t.slides.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            for (index, &square) in t.slides.squares(entry).iter().enumerate() {
                if self.board[square as usize] != 0 {
                    enter(square as usize, portals.slide_dirs[entry.start as usize + index]);
                    break;
                }
            }
        }
        for entry in t.hops.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            let squares = t.hops.squares(entry);
            let length = squares.len();
            let mut index = 0;
            while index < length && self.board[squares[index] as usize] == 0 {
                index += 1;
            }
            if index >= length
                || !screen_matches(entry.screen_owner, self.board[squares[index] as usize] as u16, low, high)
            {
                continue;
            }
            index += 1;
            let stop = if entry.screen_range == 0 {
                length
            } else {
                length.min(index + entry.screen_range as usize)
            };
            while index < stop {
                let square = squares[index] as usize;
                if self.board[square] != 0 {
                    enter(square, portals.hop_dirs[entry.start as usize + index]);
                    break;
                }
                index += 1;
            }
        }
        if n_entered == 0 {
            return;
        }

        // Every (exit, entry square, entry type) the momentum allows, deduplicated.
        let mut exits = [(0u16, 0u16, 0u8); MAX_PORTAL_ENDS * EXIT_DIRECTIONS];
        let mut n_exits = 0;
        for &(entry, bits) in &entered[..n_entered] {
            let entry_type = t.type_of(self.board[entry as usize]);
            let to = portals.dest[entry_type];
            if to < 0 {
                continue;
            }
            let far = at[to as usize - portals.first_type];
            if far == u16::MAX {
                continue;
            }
            for direction in 0..EXIT_DIRECTIONS {
                let quiet = bits & (1 << direction) != 0;
                let taking = bits & (1 << (EXIT_DIRECTIONS + direction)) != 0;
                if !quiet && !taking {
                    continue;
                }
                let exit = portals.exit(t.n_squares, player, far as usize, direction);
                if exit < 0 || exit as usize == origin {
                    continue;
                }
                let occupant = self.board[exit as usize] as u16;
                let allowed = if occupant == 0 {
                    quiet
                } else {
                    taking && may_land(t, 2, piece_type, occupant, low, high)
                };
                let item = (exit as u16, entry, entry_type as u8);
                if allowed && !exits[..n_exits].contains(&item) {
                    exits[n_exits] = item;
                    n_exits += 1;
                }
            }
        }
        let exits = &mut exits[..n_exits];
        exits.sort_unstable();

        let promoting = t.is_promoting(piece_type);
        let promo_optional = promoting && t.promotion_optional(piece_type);
        for &(exit, _, entry_type) in exits.iter() {
            let exit = exit as usize;
            let word = with_portal(encode_move(origin, exit, MOVE_PORTAL, 0), entry_type as usize);
            if promoting && (leaving || t.promotes_on(player, piece_type, exit)) {
                for &promoted in t.promotion_targets(player, piece_type) {
                    if self.promotion_offered(player, piece_type, promoted) {
                        out.push(word | ((promoted + 1) << 20));
                    }
                }
                if promo_optional {
                    out.push(word);
                }
            } else {
                out.push(word);
            }
        }
    }

    /// Schema 19: every move of the side to move that a portal adds, after all its
    /// ordinary moves. First the transits, piece by piece in ascending square order
    /// (`push_transits`); then the moves of every portal end the side may move, in
    /// ascending order of the end's square -- its type's own tables, read in the mover's
    /// frame (a portal's moves read the same for both players, which the validator
    /// checks), onto empty squares only: every move of a portal type is `mode: "move"`.
    #[inline(never)]
    fn push_portal_moves(&self, out: &mut Vec<u32>, portals: &PortalTables) {
        let t = self.t;
        let player = self.side;
        let low = t.value_low(player);
        let high = low + t.side_stride as u16;
        let at = &self.portal_squares(portals);
        // The transits first, piece by piece in ascending square order, as the ordinary
        // moves were generated: the piece's frame and promotion state read as the loop
        // above reads them.
        for origin in self.occupied()[player].iter() {
            let (piece_type, frame) = t.decode_owned(self.board[origin], low, player);
            let slot = t.slot(frame, piece_type, origin);
            let leaving = t.is_promoting(piece_type)
                && t.promotes_on_leaving(piece_type)
                && t.promotes_on(player, piece_type, origin);
            self.push_transits(out, portals, at, origin, slot, piece_type, leaving);
        }
        let mut ends = [(u16::MAX, 0usize); MAX_PORTAL_ENDS];
        for (end, item) in ends.iter_mut().enumerate().take(portals.n_ends) {
            *item = (at[end], end);
        }
        let ends = &mut ends[..portals.n_ends];
        ends.sort_unstable();
        for &(square, end) in ends.iter() {
            let piece_type = portals.first_type + end;
            if square == u16::MAX || portals.movers[piece_type] & (1 << player) == 0 {
                continue;
            }
            let origin = square as usize;
            let slot = t.slot(player * t.n_facings, piece_type, origin);
            let origin_unmoved = self.unmoved.get(origin);
            let segment_start = out.len();
            for entry in t.leaps.entries(slot) {
                if entry.requires_unmoved && !origin_unmoved {
                    continue;
                }
                let modes = t.leaps.modes(entry);
                for (index, &target) in t.leaps.squares(entry).iter().enumerate() {
                    if modes[index] & 1 != 0
                        && self.board[target as usize] == 0
                        && !self.fold_on(target as usize)
                    {
                        out.push(encode_move(origin, target as usize, MOVE_NORMAL, 0));
                    }
                }
            }
            for entry in t.slides.entries(slot) {
                if entry.requires_unmoved && !origin_unmoved {
                    continue;
                }
                let modes = t.slides.modes(entry);
                for (index, &target) in t.slides.squares(entry).iter().enumerate() {
                    if self.board[target as usize] != 0 {
                        break;
                    }
                    if modes[index] & 1 != 0 && !self.fold_on(target as usize) {
                        out.push(encode_move(origin, target as usize, MOVE_NORMAL, 0));
                    }
                }
            }
            for entry in t.hops.entries(slot) {
                if entry.requires_unmoved && !origin_unmoved {
                    continue;
                }
                let squares = t.hops.squares(entry);
                let modes = t.hops.modes(entry);
                let length = squares.len();
                let mut index = 0;
                while index < length && self.board[squares[index] as usize] == 0 {
                    index += 1;
                }
                if index >= length
                    || !screen_matches(entry.screen_owner, self.board[squares[index] as usize] as u16, low, high)
                {
                    continue;
                }
                index += 1;
                let stop = if entry.screen_range == 0 {
                    length
                } else {
                    length.min(index + entry.screen_range as usize)
                };
                while index < stop && self.board[squares[index] as usize] == 0 {
                    if modes[index] & 1 != 0 && !self.fold_on(squares[index] as usize) {
                        out.push(encode_move(origin, squares[index] as usize, MOVE_NORMAL, 0));
                    }
                    index += 1;
                }
            }
            if t.dedup[slot] {
                dedup_segment(out, segment_start);
            }
            // Schema 20 (Q807): the end the last move carried here may not go straight
            // back. `portal_bar` is `[-1, -1]` without the rule, so this never matches.
            if self.portal_bar[0] == square as i16 {
                let barred = encode_move(origin, self.portal_bar[1] as usize, MOVE_NORMAL, 0);
                if let Some(index) = out[segment_start..].iter().position(|m| *m == barred) {
                    out.remove(segment_start + index);
                }
            }
        }
    }
}

impl<'t> Game<'t> {
    /// Schema 24 (`docs/archive/plan-portals-2.md` §3R.1 rule 2): a landing step of the piece on
    /// `origin` onto the empty fold on `square`, with the step's `mode`, also **goes
    /// through**: the same step ends on the square of the end this one leads to, judged as
    /// a landing there -- onto an empty square under a *move* mode, onto a piece it may
    /// take under a *capture* mode (`may_land`: never the mover's own), never onto the
    /// mover's own square (a pass). Only where the fold admits this piece for the side to
    /// move and leads somewhere. Pushed as `push_arrival` would push a move ending there,
    /// so promotion is the far square's, every word tagged with the end gone in by plus
    /// one in the tag field (`with_fold`, which `make` reads to open no window). The caller has
    /// already asked `fold_on`; a move goes through one fold at most, since it ends there.
    #[cold]
    #[inline(never)]
    fn push_through(
        &self, out: &mut Vec<u32>, origin: usize, square: usize, mode: u8, piece_type: usize,
        leaving: bool,
    ) {
        let t = self.t;
        let Some(portals) = t.portals.as_deref() else {
            return;
        };
        let player = self.side;
        let Some((end, far)) = self.fold_through(portals, square, piece_type, player) else {
            return;
        };
        if far == origin {
            return;
        }
        let occupant = self.board[far] as u16;
        let lands = if occupant == 0 {
            mode & 1 != 0
        } else {
            let low = t.value_low(player);
            may_land(t, mode, piece_type, occupant, low, low + t.side_stride as u16)
        };
        if !lands {
            return;
        }
        let promoting = t.is_promoting(piece_type);
        let promo_optional = promoting && t.promotion_optional(piece_type);
        let start = out.len();
        self.push_arrival(
            out, origin, far, piece_type, player, promoting, leaving, promo_optional, false,
        );
        for word in &mut out[start..] {
            *word = with_fold(*word, end + 1);
        }
    }

    /// Schema 22: every move of a fold end the side to move may move (rule 6), as
    /// `MOVE_FOLD`, in ascending order of the fold's square: its group's piece's moves,
    /// read in the mover's frame, landing on an empty square or on a piece of either side
    /// (which then stands in it), never on another portal end of either kind. A fold's
    /// slide stops at the first piece, and may land on it. The straight-back bar holds as
    /// for a momentum end.
    #[inline(never)]
    fn push_fold_own_moves(&self, out: &mut Vec<u32>, portals: &PortalTables) {
        let t = self.t;
        let player = self.side;
        let low = t.value_low(player);
        let high = low + t.side_stride as u16;
        let mut ends = [(u16::MAX, 0usize); MAX_PORTAL_ENDS];
        let mut n_ends = 0;
        for end in 0..portals.n_ends {
            let square = self.fold_at[end];
            if square >= 0 && portals.movers[portals.first_type + end] & (1 << player) != 0 {
                ends[n_ends] = (square as u16, end);
                n_ends += 1;
            }
        }
        let ends = &mut ends[..n_ends];
        ends.sort_unstable();
        let lands = |target: usize| {
            let value = self.board[target];
            !self.fold_on(target) && !t.is_portal_end(value)
        };
        for &(square, end) in ends.iter() {
            let origin = square as usize;
            let piece_type = portals.first_type + end;
            let slot = t.slot(player * t.n_facings, piece_type, origin);
            let word = |target: usize| {
                with_fold(encode_move(origin, target, MOVE_FOLD, 0), end)
            };
            let segment_start = out.len();
            for entry in t.leaps.entries(slot) {
                let modes = t.leaps.modes(entry);
                for (index, &target) in t.leaps.squares(entry).iter().enumerate() {
                    if modes[index] & 1 != 0 && lands(target as usize) {
                        out.push(word(target as usize));
                    }
                }
            }
            for entry in t.slides.entries(slot) {
                let modes = t.slides.modes(entry);
                for (index, &target) in t.slides.squares(entry).iter().enumerate() {
                    if modes[index] & 1 != 0 && lands(target as usize) {
                        out.push(word(target as usize));
                    }
                    if self.board[target as usize] != 0 {
                        break;
                    }
                }
            }
            for entry in t.hops.entries(slot) {
                let squares = t.hops.squares(entry);
                let modes = t.hops.modes(entry);
                let length = squares.len();
                let mut index = 0;
                while index < length && self.board[squares[index] as usize] == 0 {
                    index += 1;
                }
                if index >= length
                    || !screen_matches(entry.screen_owner, self.board[squares[index] as usize] as u16, low, high)
                {
                    continue;
                }
                index += 1;
                let stop = if entry.screen_range == 0 {
                    length
                } else {
                    length.min(index + entry.screen_range as usize)
                };
                while index < stop {
                    let target = squares[index] as usize;
                    if modes[index] & 1 != 0 && lands(target) {
                        out.push(word(target));
                    }
                    if self.board[target] != 0 {
                        break;
                    }
                    index += 1;
                }
            }
            dedup_segment(out, segment_start);
            // Schema 20 (Q807): the fold the last move carried here may not go straight back.
            if self.portal_bar[0] == square as i16 {
                let barred = word(self.portal_bar[1] as usize);
                if let Some(index) = out[segment_start..].iter().position(|m| *m == barred) {
                    out.remove(segment_start + index);
                }
            }
        }
    }

    /// Drop moves: pieces from hand onto the board.
    ///
    /// Never inlined (Q785). Schema 18e's second face made the body a nested loop, and
    /// inlined into `generate_pseudo_legal` it cost chess, which never drops, 7% of
    /// perft(6) (3.83 s to 4.12 s, quiet, `--threads 1`); out of line it costs nothing.
    #[inline(never)]
    fn generate_drops(&self, out: &mut Vec<u32>) {
        let t = self.t;
        let rule = t.drops.as_ref().expect("generate_drops without drops");
        let player = self.side;
        let hand = &self.hands[player];
        if hand.iter().all(|&count| count == 0) {
            return;
        }

        let width = t.width;
        // The empty squares are the complement of both occupancies, masked to the
        // board; walked in ascending order they are the squares the scan this replaced
        // found, in the order it found them. The file mask was a heap allocation *per
        // piece type* once; it is on the stack at the size the core is built for, and
        // is not cleared beyond the part it uses.
        // Every piece, not just the two armies: a square a wild piece stands on is not
        // an empty square, and a drop onto it would overwrite a boar (schema 9).
        let empty = t.all_squares.difference(&self.all_pieces());
        let mut blocked_files = [false; MAX_SQUARES];

        for (slot, &held) in hand.iter().enumerate().take(t.n_types) {
            if held == 0 {
                continue;
            }
            // Schema 18 (Q785): under `either_face` a held piece may also be dropped as
            // its paired face, spent from the same slot. The region is the slot's (where
            // a piece from that slot may be returned, as Fairy-Stockfish's `dropPromoted`
            // reads it); `no_doubled` is the face's, being about the piece that stands.
            let paired = if rule.faces.is_empty() { -1 } else { rule.faces[slot] };
            let region = &rule.regions[player][slot];
            for face in std::iter::once(slot).chain((paired >= 0).then_some(paired as usize)) {
                if rule.no_doubled[face] {
                    blocked_files[..width].fill(false);
                    // Any orientation counts: a doubled pawn is doubled whichever way the
                    // one already there happens to be pointing.
                    for sq in self.occupied()[player].iter() {
                        if t.is_piece(self.board[sq], player, face) {
                            blocked_files[sq % width] = true;
                        }
                    }
                }
                for square in empty.iter() {
                    if let Some(ref region_mask) = region {
                        if !region_mask[square] {
                            continue;
                        }
                    }
                    if rule.no_doubled[face] && blocked_files[square % width] {
                        continue;
                    }
                    // The move word's promotion field carries the face, what stands on
                    // the square afterwards; `make` spends `slot_of(face)` from the hand.
                    out.push(encode_move(square, square, MOVE_DROP, face as u32));
                }
            }
        }
    }

    /// Q1059a (`drops.gating`, Seirawan's gating): for every move in `out[base..]` that takes
    /// a piece off one of the mover's gate squares for the first time (its `unmoved` bit up),
    /// the same move once more per type held, placing that piece on the vacated square. A
    /// castle vacates two, the king's origin and the rook's (`with_gate`'s bit 31 says which).
    /// The plain move stays: entering is a choice. Appended after every other move, in move
    /// order then hand order, so the moves of a game without gating are the ones it had.
    ///
    /// Only the kinds that leave a piece's origin: a drop, a turn or a buy vacates nothing,
    /// and the rules that make the other kinds are refused beside gating.
    #[cold]
    #[inline(never)]
    fn push_gates(&self, out: &mut Vec<u32>, base: usize, gates: &[SquareSet; 2]) {
        let t = self.t;
        let player = self.side;
        let gates = &gates[player];
        let hand = &self.hands[player];
        if hand.iter().all(|&count| count == 0) {
            return;
        }
        let end = out.len();
        for index in base..end {
            let m = out[index];
            let kind = move_kind(m);
            let origin = move_from(m);
            let (first, second) = match kind {
                MOVE_NORMAL | MOVE_EN_PASSANT | MOVE_PROMOTION => (origin, usize::MAX),
                MOVE_CASTLE => {
                    // A square is vacated only if neither piece lands back on it: a castle
                    // that leaves the king in place, or file-based castling that puts one
                    // piece where the other stood, vacates fewer than two.
                    let target = move_to(m);
                    let (rook_from, rook_to) =
                        self.castling_rook_squares(origin, target, move_promotion(m) as i32);
                    let vacated = |square: usize| square != target && square != rook_to;
                    (
                        if vacated(origin) { origin } else { usize::MAX },
                        if vacated(rook_from) { rook_from } else { usize::MAX },
                    )
                }
                _ => continue,
            };
            for (square, at_rook) in [(first, false), (second, true)] {
                if square == usize::MAX || !gates.get(square) || !self.unmoved.get(square) {
                    continue;
                }
                for (slot, &held) in hand.iter().enumerate().take(t.n_types) {
                    if held > 0 {
                        out.push(with_gate(m, slot, at_rook));
                    }
                }
            }
        }
    }

    /// TC5 (`docs/plan-turncoat.md` §5.4): every buy the side to move can afford, onto each
    /// empty square of its region a market type may stand on, in market order then square
    /// order. Supply is unlimited; the coins are the only scarcity. Under `drops.placement`
    /// a held hand returns before this is reached, so a drop is then the only move.
    #[cold]
    #[inline(never)]
    fn generate_buys(&self, out: &mut Vec<u32>) {
        let t = self.t;
        let purse = t.purse.as_deref().expect("generate_buys without a purse");
        let player = self.side;
        let coins = self.purse[player];
        for (entry, &(piece_type, price)) in purse.market.iter().enumerate() {
            if price > coins {
                continue;
            }
            for &square in purse.squares(player, entry) {
                let square = square as usize;
                if self.board[square] == 0 {
                    out.push(encode_move(square, square, MOVE_BUY, u32::from(piece_type)));
                }
            }
        }
    }

    /// TC6 (`docs/plan-turncoat.md` §5.5): every hire and bribe the side to move can afford:
    /// a nobody's piece of a type for hire (`PurseTables::hire`), or under `rules.purse.bribe`
    /// an opponent's piece of a priced type (`PurseTables::bribe`, twice the price), standing
    /// next to (`PurseTables::near`) one of the mover's pieces. In square order. The board is
    /// read rather than the occupancy sets, since a filter may ask for these moves.
    #[cold]
    #[inline(never)]
    fn generate_hires(&self, out: &mut Vec<u32>, purse: &PurseTables) {
        let t = self.t;
        let player = self.side;
        let coins = self.purse[player];
        for square in 0..t.n_squares {
            let value = self.board[square];
            if value == 0 {
                continue;
            }
            let piece_type = t.type_of(value);
            let cost = purse.cost(piece_type, t.owner_of(value), player);
            if cost == 0 || cost > coins {
                continue;
            }
            let beside = purse.near(t.n_squares, piece_type, square).iter().any(|&near| {
                let held = self.board[near as usize];
                held != 0 && t.owner_of(held) == player
            });
            if beside {
                out.push(encode_move(square, square, MOVE_HIRE, 0));
            }
        }
    }

    /// King-and-rook moves, generated separately because they move two pieces.
    fn generate_castling(&self, out: &mut Vec<u32>) {
        let t = self.t;
        let (king_type, rook_type, king_steps) =
            t.castling.expect("generate_castling without a rule");
        let player = self.side;
        let width = t.width;
        // One value per orientation the king can be in, and that is one value unless the
        // castling king is *not* royal -- a royal may not face. Written as a loop over
        // values rather than as a decode per square because this ran at every node of
        // every castling ruleset, and with one value it is the single byte comparison
        // it has always been. The candidates are the mover's unmoved pieces, walked in
        // ascending order: the first of them holding the king's value is the square
        // the scan over every square used to stop on.
        let candidates = self.occupied()[player].intersection(&self.unmoved);
        let mut king_square = usize::MAX;
        'find: for facing in 0..t.facings_of(king_type) {
            let king_value = t.piece_value(player, facing, king_type);
            for square in candidates.iter() {
                if self.board[square] == king_value {
                    king_square = square;
                    break 'find;
                }
            }
        }
        if king_square == usize::MAX {
            return;
        }

        let needs_safety = t.needs_check_filter();
        if needs_safety
            && self.attacks_any(1 - player, king_square)
        {
            return;
        }

        let rank = king_square / width;
        let king_file = (king_square % width) as i32;
        // Schema 24 (`docs/archive/plan-portals-2.md` §3R.1 rule 9): a fold on the span is floor.
        let blocked_at = |square: usize| self.board[square] != 0;

        for rook_file in 0..width as i32 {
            if rook_file == king_file {
                continue;
            }
            let rook_square = rank * width + rook_file as usize;
            if !t.is_piece(self.board[rook_square], player, rook_type)
                || !self.unmoved.get(rook_square)
            {
                continue;
            }

            if let Some((king_qs, king_ks, rook_qs, rook_ks)) = t.castle_files {
                // File-based castling (schema 2)
                let side = if rook_file > king_file { 1 } else { 0 };
                let king_to = if side == 1 { king_ks } else { king_qs };
                let rook_to = if side == 1 { rook_ks } else { rook_qs };
                if king_to < 0 || king_to >= width as i32 || rook_to < 0 || rook_to >= width as i32 {
                    continue;
                }

                // Check that the span is clear (excluding the castling pieces themselves)
                let span_lo = king_file.min(king_to).min(rook_file).min(rook_to);
                let span_hi = king_file.max(king_to).max(rook_file).max(rook_to);
                let mut blocked = false;
                for file in span_lo..=span_hi {
                    if file == king_file || file == rook_file {
                        continue;
                    }
                    if blocked_at(rank * width + file as usize) {
                        blocked = true;
                        break;
                    }
                }
                if blocked {
                    continue;
                }

                if needs_safety {
                    let low = king_file.min(king_to);
                    let high = king_file.max(king_to);
                    let mut crossed_attack = false;
                    for file in low..=high {
                        // The victim is the king, not whatever happens to stand there:
                        // a transit square is empty by the check above, so
                        // `attacks_square` would skip the capture matrix and call a
                        // square attacked by a piece that could not take a king.
                        if self.threatened_as(
                            1 - player, rank * width + file as usize, king_type,
                        ) {
                            crossed_attack = true;
                            break;
                        }
                    }
                    if crossed_attack {
                        continue;
                    }
                }

                out.push(encode_move(
                    king_square,
                    rank * width + king_to as usize,
                    MOVE_CASTLE,
                    rook_file as u32,
                ));
            } else {
                // Step-based castling (schema 1)
                let step: i32 = if rook_file > king_file { 1 } else { -1 };

                let mut file = king_file + step;
                let mut blocked = false;
                while file != rook_file {
                    if blocked_at(rank * width + file as usize) {
                        blocked = true;
                        break;
                    }
                    file += step;
                }
                if blocked {
                    continue;
                }

                let target_file = king_file + step * king_steps;
                if target_file < 0 || target_file >= width as i32 {
                    continue;
                }
                if (target_file - king_file).abs() >= (rook_file - king_file).abs() {
                    continue;
                }

                if needs_safety {
                    let mut file = king_file + step;
                    let mut crossed_attack = false;
                    loop {
                        if self.threatened_as(
                            1 - player, rank * width + file as usize, king_type,
                        ) {
                            crossed_attack = true;
                            break;
                        }
                        if file == target_file {
                            break;
                        }
                        file += step;
                    }
                    if crossed_attack {
                        continue;
                    }
                }

                out.push(encode_move(
                    king_square,
                    rank * width + target_file as usize,
                    MOVE_CASTLE,
                    0,
                ));
            }
        }
    }

    /// Squares holding the sole blocker of an enemy line onto `royal`.
    fn pinned_squares(&self, attacker: usize, royal: usize) -> SquareSet {
        let t = self.t;
        let mut pinned = SquareSet::EMPTY;
        for source in t.attackers_of(attacker, royal) {
            if source.screens > 0 || self.board[source.origin as usize] != source.piece_value {
                continue;
            }
            let mut blocker: i32 = -1;
            for square in t.attack_path(source) {
                if self.board[*square as usize] != 0 {
                    if blocker >= 0 {
                        blocker = -1;
                        break;
                    }
                    blocker = *square as i32;
                }
            }
            if blocker >= 0 {
                pinned.set(blocker as usize);
            }
        }
        pinned
    }

    /// Play a move on the board for a legality filter to ask one question about, saving
    /// every square it writes.
    ///
    /// **One statement of what a filter's board looks like, for the two filters that ask.**
    /// `leaves_royal_attacked` and the `forbid_check` branch of `restrict` held two copies
    /// of this, and schema 8 had to repair both: a rotation's origin *is* its target, so
    /// "clear the origin, write the target" cancels and the filter would have asked its
    /// question about a board the move had not been played on.
    ///
    /// Returns the value that stood on the target square, and `None` for the three kinds
    /// that are not an ordinary move -- which is what the explosion branch in
    /// `leaves_royal_attacked` keys on. A rotation's "target" holds the piece that is
    /// turning, so a flag saying *this was an ordinary move* is the difference between an
    /// explosion firing where a capture happened and one firing wherever a piece turns.
    fn play_for_filter(&mut self, m: u32, saved: &mut Saved) -> Option<u8> {
        let played = self.play_move_for_filter(m, saved);
        // The phase belongs to the ply, so a filter asking about the position a move
        // leaves behind has to play it too (schema 9). Here rather than in the two
        // callers, which is what makes "legality is asked of the post-phase position"
        // one statement -- `must_capture` asks about the move alone and does not come
        // through this function at all.
        if self.t.has_npcs {
            self.npc_phase_for_filter(saved);
        }
        played
    }

    /// The move alone, without the phase: the half `play_for_filter` shared before
    /// schema 9 gave it a second half.
    fn play_move_for_filter(&mut self, m: u32, saved: &mut Saved) -> Option<u8> {
        let t = self.t;
        let player = self.side;
        let origin = move_from(m);
        let target = move_to(m);
        let kind = move_kind(m);

        // Schema 22: a fold's own move writes no square; its caller moves the fold
        // (`shift_fold_for_filter`).
        if kind == MOVE_FOLD {
            return None;
        }
        if enters_play(kind) {
            saved.push((target, self.board[target]));
            // What arrives, not what was dropped (schema 9): a struck or dropped piece
            // that promotes on arrival can give check, and a filter that did not know
            // would let a royal be captured -- the `strike_writes`-in-the-filter argument
            // applied to the new table.
            self.board[target] = t.arriving_value(player, move_promotion(m) as usize, target);
            return None;
        }
        if kind == MOVE_HIRE {
            // TC6: the piece changes sides where it stands, and so what it attacks; nothing
            // moves.
            let bought = self.board[origin];
            saved.push((origin, bought));
            self.board[origin] = t.rehome_value(bought, player, origin);
            return None;
        }
        if kind == MOVE_ROTATE {
            // A turn changes what the piece attacks, so it can put a royal in check --
            // and the pin shortcut in `generate_legal` is right to treat it like any
            // other move from that square. What it must not do is nothing, which is
            // what the ordinary "clear the origin, write the target" path amounts to
            // when the two squares are the same.
            let turning = self.board[origin];
            saved.push((origin, turning));
            // Q1007c: the turning piece's owner (a commanded turn turns the opponent's).
            self.board[origin] = t.piece_value(
                t.owner_of(turning), move_promotion(m) as usize, t.type_of(turning),
            );
            return None;
        }
        if kind == MOVE_CASTLE {
            let moving = self.board[origin];
            let (rook_from, rook_to) =
                self.castling_rook_squares(origin, target, move_promotion(m) as i32);
            let rook_value = self.board[rook_from];
            saved.push((origin, self.board[origin]));
            saved.push((target, self.board[target]));
            saved.push((rook_from, self.board[rook_from]));
            saved.push((rook_to, self.board[rook_to]));
            self.board[origin] = 0;
            self.board[rook_from] = 0;
            self.board[target] = moving;
            self.board[rook_to] = rook_value;
            self.gate_for_filter(m, origin, rook_from, saved);
            return None;
        }

        let moving = self.board[origin];
        let target_before = self.board[target];
        saved.push((origin, moving));
        saved.push((target, target_before));
        self.board[origin] = 0;
        self.board[target] = moving;

        let mut captured = target_before;
        let mut captured_square = target;
        if kind == MOVE_PROMOTION {
            self.board[target] = t.entering_value(player, move_promotion(m) as usize);
        } else if kind == MOVE_EN_PASSANT {
            let ep_sq = self.ep_victim as usize;
            captured = self.board[ep_sq];
            captured_square = ep_sq;
            saved.push((ep_sq, captured));
            self.board[ep_sq] = 0;
        } else if kind == MOVE_SWEEP {
            // Schema 13: the path is cleared as `make` clears it. Three victims fit inline
            // beside the origin and the target; a fourth spills, correctly and rarely.
            // `Tables::sweep_victims` names them.
            let mut victims = 0;
            let mut path = t.sweep_victims(moving, origin, target);
            while let Some((square, value)) = path.next_on(&self.board) {
                saved.push((square, value));
                self.board[square] = 0;
                victims += 1;
            }
            debug_assert_eq!(saved.spill.is_empty(), victims + 2 <= Saved::INLINE);
        } else if kind == MOVE_PORTAL && move_promotion(m) != 0 {
            self.board[target] = t.entering_value(player, move_promotion(m) as usize - 1);
        } else if kind == MOVE_PUSH {
            // Q1074b2: the push as `write_push` plays it, `push_shift_line` the one reading
            // (Q1074b4: the whole line under the longer chains, every square saved).
            if move_promotion(m) != 0 {
                self.board[target] = t.entering_value(player, move_promotion(m) as usize - 1);
            }
            (captured, captured_square) =
                self.push_shift_line(m, target_before, |game, square, value| {
                    saved.push((square, game.board[square]));
                    game.board[square] = value;
                });
        }
        // Q1059a: a gating move's piece stands on the origin it vacated, and may block a line.
        self.gate_for_filter(m, origin, origin, saved);

        // Strike (schema 6): the surviving piece has to be on the board before the
        // attack query, or a victim that a strike turns into a checking piece is
        // invisible to the filter and a king can be captured. `strike_writes` is the
        // same statement `make` plays, rather than a copy of it, and it is handed the
        // piece `make` hands it -- en passant's victim stands off the target.
        if let Some(outcome) = self.strike_writes(
            t.type_of(moving), captured, origin, target, captured_square, move_landing(m),
        ) {
            for &(square, value) in outcome.writes() {
                saved.push((square, self.board[square]));
                self.board[square] = value;
            }
            // A reborn piece (schema 17's `home`) is a fourth write and spills, as an en
            // passant strike does: rare, correct, and not worth a sixth inline slot.
            debug_assert!(
                kind == MOVE_EN_PASSANT || outcome.count > 3 || saved.spill.is_empty(),
                "a strike spilled out of Saved's inline slots"
            );
        }

        Some(target_before)
    }

    /// Q1059a: a gating move's held piece, written onto the square it enters (the origin, or
    /// `rook_from` under bit 31) for a legality filter, saved for the restore. A gated piece
    /// fills a square the move emptied, so it can only block: a pinned piece may leave its
    /// square when one enters behind it, which is why the filter must see it. One mask test
    /// on every other move.
    #[inline(always)]
    fn gate_for_filter(&mut self, m: u32, origin: usize, rook_from: usize, saved: &mut Saved) {
        if move_gate(m) == 0 {
            return;
        }
        let t = self.t;
        if t.drops.as_ref().is_none_or(|rule| rule.gates.is_none()) {
            return;
        }
        let square = if move_gate_rook(m) { rook_from } else { origin };
        saved.push((square, self.board[square]));
        self.board[square] = t.arriving_value(self.side, move_gate(m) - 1, square);
    }

    /// Legality test that skips building a successor state: only the board and the
    /// unmoved mask affect the answer, so the move is played, tested and taken back.
    fn leaves_royal_attacked(&mut self, m: u32) -> bool {
        let t = self.t;
        let player = self.side;
        let origin = move_from(m);
        let target = move_to(m);
        let kind = move_kind(m);

        let saved_unmoved = self.unmoved;
        let mut saved = Saved::new();
        // How many royals this player had *before* the move. The phase can eat one --
        // a king that steps onto a boar's next square is taken in the mover's own phase
        // -- and that is the one way a move can remove the royal it is being tested for.
        // A royal that has been eaten is not an unattacked royal, and without this the
        // post-phase question ("is it attacked?") answers no because there is nothing
        // there to attack. Zero where nothing is wild, and the count is then never read.
        let royals_before = if t.has_npcs { self.royal_count(player) } else { 0 };
        self.scratch += 1;

        // Schema 10 (Q18): who is capturing, read before the move is played because
        // `play_for_filter` empties the origin. Behind `explosion.is_some()` so a ruleset
        // without the rule pays nothing, and `make`'s reading exactly -- the mover's type
        // as it stood before the move.
        let capturer = if t.explosion.is_some() && !enters_play(kind) {
            t.type_of(self.board[origin])
        } else {
            0
        };

        // Apply the move to the board, saving every square we touch.
        let target_before = self.play_for_filter(m, &mut saved);
        let fold_moved = self.shift_fold_for_filter(m);

        // Explosion. Only after an ordinary move, and only where something was taken:
        // the target square of a rotation holds the piece that is turning, and reading
        // it as a victim would detonate every turn.
        if let (Some(target_before), Some(explosion)) = (target_before, t.explosion.as_ref()) {
            if target_before != 0 && explosion.fires(capturer, t.type_of(target_before)) {
                // The captor blows up where it now stands. Its square is already saved
                // holding `target_before`, which is what the restore has to put back.
                self.board[target] = 0;
                for &sq in explosion.reach(target) {
                    let value = self.board[sq];
                    if explosion.clears(t, value) {
                        saved.push((sq, value));
                        self.board[sq] = 0;
                    }
                }
            }
        }

        if fold_moved.is_none() {
            self.unmoved.clear(origin);
            self.unmoved.clear(target);
        }

        // Whether this move could have emptied a royal square at all (Q133). Only two
        // things take a piece off a square the move did not move it from: something being
        // taken on the target -- which is also the only thing that sets off a blast -- and
        // the wild phase, which walks after every move. Everything else is quiet, and a
        // quiet move pays one `bool` rather than a royal lookup it cannot need. The
        // lookup is worth avoiding: inside a filter `royal_hint` is read but never
        // written, so a miss is a board scan that is paid again on the next move tested.
        let took_something = matches!(target_before, Some(value) if value != 0);
        let could_empty_a_royal_square = took_something || t.has_npcs || kind == MOVE_SWEEP;

        // Where losing the royal is losing, a move that removes the **opponent's** wins on
        // the spot -- `terminal_prefix` says so before anyone moves again -- so whatever it
        // leaves the mover's own royal exposed to cannot matter: the game is over before
        // the exposure could be collected. That is the whole content of the branch this
        // replaces, which was gated on `explode_on_capture` and carried a second copy of
        // the three questions below to get back to the ordinary answer.
        let attacked = if could_empty_a_royal_square
            && t.royal_loss_ends_it(1 - player)
            && self.royal_gone(1 - player)
        {
            false
        } else if t.royal_facing_forbidden && self.royals_face() {
            true
        } else if t.multi_royal {
            let mine = self.royal_squares(player);
            // The mover's own royalty gone, and this side had royalty to lose: the move
            // loses outright, and a move that loses outright is refused rather than played
            // and then lost. `mine.len() < royals_before` is the narrower reading beside
            // it -- losing *one* of several, which only the wild phase can do.
            (mine.is_empty() && t.royal_loss_ends_it(player) && !self.royal_in_hand(player))
                || mine.len() < royals_before
                || mine.iter().any(|&r| self.threatened(1 - player, r))
        } else {
            match self.royal_square(player) {
                // Gone, and this side had one to lose. This read `royals_before > 0`,
                // which is the same answer wherever the question can be reached -- the
                // filter is only entered with the royal standing -- and `royals_before` is
                // zero unless the ruleset has a wild phase, so the old form also silently
                // permitted a move that removed the mover's own royal any other way. One
                // reading now, and it is the one the explosion arm already took.
                //
                // Schema 18 (Q784): or not yet played. A royal still in hand under
                // `drops.placement` is not lost, and this move does not lose it.
                r if r < 0 => t.royal_loss_ends_it(player) && !self.royal_in_hand(player),
                r => self.threatened(1 - player, r as usize),
            }
        };

        saved.restore(&mut self.board);
        if let Some((end, square)) = fold_moved {
            self.set_fold(end, square);
        }
        self.unmoved = saved_unmoved;
        self.scratch -= 1;
        attacked
    }

    /// Schema 22: for a legality filter, stand the fold a `MOVE_FOLD` moves on its target,
    /// and say which end and where it stood, for the filter to put back. `None` for every
    /// other move.
    #[inline(always)]
    fn shift_fold_for_filter(&mut self, m: u32) -> Option<(usize, i16)> {
        if move_kind(m) != MOVE_FOLD {
            return None;
        }
        let end = move_fold(m);
        let before = self.fold_at[end];
        self.set_fold(end, move_to(m) as i16);
        Some((end, before))
    }
}

impl<'t> Game<'t> {
    /// Every move the side to move may play, appended to `out`.
    ///
    /// Schema 29 (Q983a, `docs/plan-rest.md` §3.7): the rest filters after the check filter
    /// and before `restrict`, lifted when it would leave nothing; and when `restrict` then
    /// empties a list the rest shortened, the list is made again without it, so the rest
    /// alone never leaves a player without a move. One test in every game without a rest.
    pub fn generate_legal(&mut self, out: &mut Vec<u32>) {
        // Generation 42 (N4b): a duck half lists the duck's moves alone, unfiltered: a duck
        // game has no check, the duck never rests, and portals are refused beside it.
        if self.duck_owed() {
            return self.push_duck_moves(out);
        }
        let base = out.len();
        self.generate_checked(out, base);
        if self.t.rest.is_some() {
            self.rest_then_restrict(out, base);
        } else {
            self.restrict(out, base);
        }
    }

    /// `generate_legal`'s tail under a rest, out of line so a game without one carries
    /// neither it nor a second copy of the generator (Q983a's benches).
    #[inline(never)]
    fn rest_then_restrict(&mut self, out: &mut Vec<u32>, base: usize) {
        let rested = self.rest_filter(self.side, out, base);
        self.restrict(out, base);
        if rested && out.len() == base {
            self.regenerate_checked(out, base);
        }
    }

    /// The list made again without the rest (§3.7): rare, so out of line.
    #[cold]
    #[inline(never)]
    fn regenerate_checked(&mut self, out: &mut Vec<u32>, base: usize) {
        self.generate_checked(out, base);
        self.restrict(out, base);
    }

    /// The side to move's pseudo-legal moves less those that leave its royal attacked,
    /// appended to `out` from `base`.
    #[inline(always)]
    fn generate_checked(&mut self, out: &mut Vec<u32>, base: usize) {
        let t = self.t;
        self.generate_pseudo_legal(out);

        if t.needs_check_filter() {
            let player = self.side;

            if t.multi_royal {
                // Multi-royal: verify every candidate
                let mut write = base;
                for index in base..out.len() {
                    let m = out[index];
                    if !self.leaves_royal_attacked(m) {
                        out[write] = m;
                        write += 1;
                    }
                }
                out.truncate(write);
            } else {
                let royal = self.royal_square(player);
                if royal >= 0 {
                    let royal = royal as usize;

                    // Verify every candidate. Four reasons, and the last one is not like
                    // the others: three are about the *ruleset*, and say the shortcut
                    // below has no premise to stand on; the fourth is about the
                    // *position*, and says the premise holds but admits nothing, because
                    // out of check any move at all may be the one that answers it.
                    //
                    // **The shortcut's premise, stated once.** A move changes whether the
                    // mover's own royal is attacked only by touching one of the royal's
                    // lines -- the royal's own square, the sole blocker of an enemy line
                    // onto it, a hop line, an en passant or a drop -- so every other move
                    // may be admitted unasked. Three rules make that false, and each was
                    // found by a played game rather than by a perft, because a filter
                    // that errs in the permissive direction still hands back a move list
                    // that looks like a move list:
                    //
                    // * `explosion` takes pieces off squares the move never touched.
                    // * `royals_face` makes a move illegal by a geometry no attack table
                    //   carries.
                    // * `npcs` (schema 9) walks a piece nobody owns after **every** move,
                    //   so the board the question is asked of is not the board the move
                    //   left behind. A wild piece stepping *off* a square unblocks
                    //   whatever line ran through it, and one stepping *onto* a square
                    //   carries its threat -- the square it steps to next -- along with
                    //   it. Neither has anything to do with the move being tested, which
                    //   is what makes the shortcut unsound here and not merely
                    //   incomplete.
                    //
                    // Spelled out rather than precomputed into a `Tables` field, and that
                    // is a measurement and not a preference: the field costs chess -- none
                    // of the three -- 4.3% of a random playout, because one more field
                    // moves a struct the movegen reads at every ply, while these three
                    // loads cost it nothing. The order is by price: two discriminants and
                    // a bool before the attack query, all of them short-circuiting.
                    // * `has_sweep` (schema 13) takes pieces off squares the move never
                    //   touched, and a quiet move can open a landing beyond a royal.
                    // * `portals` (schema 19) move, and a move off any line through one
                    //   can open a check that comes out of the other end.
                    //
                    // A sight (schema 30) that bears on check widens the premise rather than
                    // dropping it: a move can also thaw an attacker or bare the royal, but
                    // only by touching a source of an effect that binds now, or its ray
                    // (`effect_lines`, `docs/archive/plan-passive.md` §3.5's second lever), so those
                    // squares join the hop lines below.
                    // * Q1077: a held neutral that `takes_royal` attacks for whoever holds
                    //   it, and any move next to one can change who that is.
                    if t.explosion.is_some()
                        || t.royal_facing_forbidden
                        || t.has_npcs
                        || t.has_sweep
                        || t.portals.is_some()
                        || t.majority.as_deref().is_some_and(|m| m.threatens_royals)
                        || self.in_check_at(player, royal)
                    {
                        let mut write = base;
                        for index in base..out.len() {
                            let m = out[index];
                            if !self.leaves_royal_attacked(m) {
                                out[write] = m;
                                write += 1;
                            }
                        }
                        out.truncate(write);
                    } else {
                        // Out of check: pin shortcut
                        let pinned = self.pinned_squares(1 - player, royal);
                        let mut hop_lines = if t.has_hop_attacks {
                            t.hop_line(1 - player, royal)
                        } else {
                            SquareSet::EMPTY
                        };
                        if let Some(sight) = t.sight.as_deref() {
                            if sight.bears_on_check {
                                hop_lines = hop_lines.union(&self.effect_lines(sight, player, royal));
                            }
                            // Q1007c (`docs/archive/plan-schema-31.md` §3.4): a commanded move starts on
                            // one of the opponent's squares, so every move from or onto one is
                            // asked, and every commanded move is verified. Read here, outside the
                            // loop, so a game without a command pays nothing per move.
                            if sight.has_command {
                                hop_lines = hop_lines.union(&self.occupied()[1 - player]);
                            }
                        }
                        // TC3: a held neutral starts on nobody's square, which may be the sole
                        // blocker of a line onto the royal, so every move from one is asked.
                        if t.majority.is_some() {
                            hop_lines = hop_lines.union(&self.occupied()[WILD]);
                        }
                        let hop_relevant = !hop_lines.is_empty();

                        let has_strike = t.strike.is_some();
                        // Q1074b2 (R5): a push moves a second piece, which may have been the
                        // sole blocker of a line onto the royal, so every push is asked.
                        let has_push = t.push.is_some();
                        let mut write = base;
                        for index in base..out.len() {
                            let m = out[index];
                            let origin = move_from(m);
                            let suspect = origin == royal
                                || pinned.get(origin)
                                || move_kind(m) == MOVE_EN_PASSANT
                                || enters_play(move_kind(m))
                                || (has_push && move_kind(m) == MOVE_PUSH)
                                || (hop_relevant && (hop_lines.get(origin) || hop_lines.get(move_to(m))))
                                || (has_strike && self.board[move_to(m)] != 0);
                            if suspect && self.leaves_royal_attacked(m) {
                                continue;
                            }
                            out[write] = m;
                            write += 1;
                        }
                        out.truncate(write);
                    }
                } else if self.royal_in_hand(player) {
                    // Schema 18 (Q784): the royal is still in hand, so nothing on the
                    // board can be left attacking it, and the one move that can expose
                    // it is its own drop. Each royal drop is asked; every other move
                    // stands. A royal gone for good is `terminal_prefix`'s, not this.
                    let mut write = base;
                    for index in base..out.len() {
                        let m = out[index];
                        let royal_drop = move_kind(m) == MOVE_DROP
                            && t.royal_types.contains(&(move_promotion(m) as usize));
                        if royal_drop && self.leaves_royal_attacked(m) {
                            continue;
                        }
                        out[write] = m;
                        write += 1;
                    }
                    out.truncate(write);
                }
            }
        }
    }

    /// Apply the restriction rules that narrow an already-legal move list.
    fn restrict(&mut self, out: &mut Vec<u32>, base: usize) {
        let t = self.t;

        // forbid_check: racing kings — giving check is illegal
        if t.forbid_check && !t.royal_types.is_empty() {
            let player = self.side;
            let opponent = 1 - player;
            let mut write = base;
            self.scratch += 1;
            for index in base..out.len() {
                let m = out[index];
                // Apply move temporarily to check if it gives check. No explosion arm:
                // this filter never had one, and `play_for_filter` is deliberately the
                // half the two callers share rather than the union of them.
                let origin = move_from(m);
                let target = move_to(m);
                let mut saved = Saved::new();
                self.play_for_filter(m, &mut saved);
                let fold_moved = self.shift_fold_for_filter(m);

                let saved_unmoved = self.unmoved;
                if fold_moved.is_none() {
                    self.unmoved.clear(origin);
                    self.unmoved.clear(target);
                }

                // "Gives check" reads schema 9's definition too: a move that leaves
                // the other royal standing where a boar steps next has left it in check,
                // whoever is threatening it, and this filter is about the position the
                // move leaves behind.
                let gives_check = if t.multi_royal {
                    self.royal_squares(opponent).iter().any(|&r| self.threatened(player, r))
                } else {
                    let r = self.royal_square(opponent);
                    r >= 0 && self.threatened(player, r as usize)
                };

                saved.restore(&mut self.board);
                if let Some((end, square)) = fold_moved {
                    self.set_fold(end, square);
                }
                self.unmoved = saved_unmoved;

                if !gives_check {
                    out[write] = m;
                    write += 1;
                }
            }
            self.scratch -= 1;
            out.truncate(write);
        }

        // forbid_immobility: shogi — no move may strand a piece
        if t.forbid_immobility {
            let player = self.side;
            let mut write = base;
            for index in base..out.len() {
                let m = out[index];
                let target = move_to(m);
                let kind = move_kind(m);
                // Schema 22: a fold's own move moves no piece, so it strands none.
                if kind == MOVE_FOLD {
                    out[write] = m;
                    write += 1;
                    continue;
                }
                // The frame the piece will be reading *after* the move, which for a
                // rotation is the whole content of the move: a piece may not turn itself
                // into a position it could never leave, exactly as it may not step into
                // one.
                let moving = self.board[move_from(m)];
                let (piece_type, frame) = match kind {
                    // A drop asks of the piece **as it stands after arrival** (schema 9):
                    // shogi with `promote_on_entry` must not refuse a pawn drop on the far
                    // rank as stranding a pawn, when what stands there is a tokin. The
                    // `MOVE_PROMOTION` arm is the moved piece's own promotion, chosen by
                    // the move, and reads the same table it always did.
                    MOVE_DROP | MOVE_BUY => {
                        let arrived = t.arriving_value(player, move_promotion(m) as usize, target);
                        (t.type_of(arrived), t.frame_of_value(arrived))
                    }
                    MOVE_PROMOTION => (
                        move_promotion(m) as usize,
                        t.frame_of_value(t.entering_value(player, move_promotion(m) as usize)),
                    ),
                    // Schema 19: a transit that promotes carries the type plus one.
                    // Q1074b2: and so does a push that promotes the pusher.
                    MOVE_PORTAL | MOVE_PUSH if move_promotion(m) != 0 => (
                        move_promotion(m) as usize - 1,
                        t.frame_of_value(t.entering_value(player, move_promotion(m) as usize - 1)),
                    ),
                    // Q1007c: the owner's frame, which a commanded turn's mover is not.
                    MOVE_ROTATE => (
                        t.type_of(moving),
                        t.owner_of(moving) * t.n_facings + move_promotion(m) as usize,
                    ),
                    // TC6: the bought piece as it stands after it changes sides.
                    MOVE_HIRE => {
                        let rehomed = t.rehome_value(moving, player, target);
                        (t.type_of(rehomed), t.frame_of_value(rehomed))
                    }
                    _ => (t.type_of(moving), t.frame_of_value(moving)),
                };
                if !t.is_immobile_square(frame, piece_type, target) {
                    out[write] = m;
                    write += 1;
                }
            }
            out.truncate(write);
        }

        // Drop-mate check: shogi forbids finishing by dropping a pawn on the king
        if let Some(rule) = t.drops.as_ref().filter(|_| !self.checking_drop_mate) {
            if rule.no_mate.iter().any(|&v| v) {
                let mut write = base;
                for index in base..out.len() {
                    let m = out[index];
                    if move_kind(m) == MOVE_DROP {
                        let pt = move_promotion(m) as usize;
                        if rule.no_mate[pt] && self.is_forbidden_drop_mate(m) {
                            continue;
                        }
                    }
                    out[write] = m;
                    write += 1;
                }
                out.truncate(write);
            }
        }

        // must_capture: antichess — a capture on offer is the only move
        if t.must_capture {
            // A capture takes an enemy piece off the board, and three move kinds can.
            // A leap, a slide and a hop all refuse a friendly occupant, so an occupied
            // target under `MOVE_NORMAL` or `MOVE_PROMOTION` holds an enemy piece; an
            // en passant takes the piece on `ep_victim` and lands on an empty square.
            // The kinds left out end on an occupied square and take nothing -- a
            // rotation on the piece that is turning, a file-based castling on the
            // king's own rook -- so reading the target square alone would make castling
            // compulsory wherever it is available, which is a rule nobody wrote.
            let takes = |m: u32, board: &[u8; MAX_SQUARES]| {
                let kind = move_kind(m);
                // Schema 13: a sweep takes its path even onto an empty square.
                kind == MOVE_EN_PASSANT
                    || kind == MOVE_SWEEP
                    || ((kind == MOVE_NORMAL || kind == MOVE_PROMOTION || kind == MOVE_PORTAL)
                        && board[move_to(m)] != 0)
                    // Q1074b3 (C6): a push that throws an enemy off the board takes it; one
                    // that moves its piece, or throws the mover's own off, does not.
                    || (kind == MOVE_PUSH && t.push_takes(board, m))
            };
            let has_capture = (base..out.len()).any(|index| takes(out[index], &self.board));
            if has_capture {
                let mut write = base;
                for index in base..out.len() {
                    let m = out[index];
                    if takes(m, &self.board) {
                        out[write] = m;
                        write += 1;
                    }
                }
                out.truncate(write);
            }
        }
    }

    /// Whether a drop move delivers checkmate (forbidden by shogi for pawns).
    fn is_forbidden_drop_mate(&mut self, m: u32) -> bool {
        let t = self.t;
        let player = self.side;
        let target = move_to(m);
        let pt = move_promotion(m) as usize;
        // The value placed is what arrives (schema 9): a pawn dropped on the far rank
        // under `promote_on_entry` gives a queen's check, not a pawn's. `no_mate` is still
        // read off the *dropped* type by the caller, which is right -- shogi's rule is
        // about what was in the hand.
        let value = t.arriving_value(player, pt, target);
        let opponent = 1 - player;

        // Place the piece
        self.scratch += 1;
        self.board[target] = value;

        // Does it give check?
        let gives_check = if t.multi_royal {
            self.royal_squares(opponent)
                .iter()
                .any(|&r| self.attacks_any(player, r))
        } else {
            let r = self.royal_square(opponent);
            r >= 0 && self.attacks_any(player, r as usize)
        };

        if !gives_check {
            self.board[target] = 0;
            self.scratch -= 1;
            return false;
        }

        // Check if the opponent is mated (with this rule suppressed)
        self.checking_drop_mate = true;
        // Temporarily apply the move
        self.board[target] = 0; // undo placement
        self.scratch -= 1;
        self.make(m);
        let is_mate = self.legal_moves().is_empty();
        self.unmake();
        self.checking_drop_mate = false;

        is_mate
    }

    pub fn legal_moves(&mut self) -> Vec<u32> {
        let mut moves = Vec::new();
        if self.t.spy.is_some() && !self.phase().is_round() {
            // Schema 27 (Q935a): a spy game's turn, the alternating game's moves.
            self.generate_legal(&mut moves);
        } else if self.t.simultaneous.is_some() {
            // Schema 26: the side to move's own moves in a round, bars applied.
            self.generate_legal_for(self.side, &mut moves);
        } else {
            self.generate_legal(&mut moves);
        }
        moves
    }

    #[inline]
    fn repetition_count(&self) -> u32 {
        let key = self.key;
        self.history[self.window..]
            .iter()
            .filter(|entry| **entry == key)
            .count() as u32
    }

    /// The half of the verdict that does not need the caller's move list.
    ///
    /// Complete except for what an empty move list means *before* a clock runs out:
    /// once `None`, the game is decided by `terminal_from_moves` alone. On the ply a
    /// clock runs out it generates the moves itself, and only when the side to move is
    /// in check, because a mate on that ply stands (Q412, `clock_or_mate`).
    pub fn terminal_prefix(&mut self) -> Option<Outcome> {
        if let Some(outcome) = self.ending_before_the_clocks() {
            return Some(outcome);
        }
        let reason = self.clock_run_out()?;
        if self.could_be_mated() {
            let moves = self.legal_moves();
            return Some(self.clock_or_mate(reason, &moves));
        }
        Some(self.on_the_clock(reason))
    }

    /// Everything asked before the clocks: the endings a position declares, then
    /// repetition.
    fn ending_before_the_clocks(&self) -> Option<Outcome> {
        let t = self.t;

        // A side whose royal is off the board has lost, wherever losing it is what loses
        // (Q133). `Tables::royal_loss_ends_it` is the whole condition and it asks the
        // *win condition* and the *setup*, never a rule key.
        //
        // This used to be gated on `t.explosion.is_some()`, so **declaring a blast** was
        // what made a missing royal end a game -- a second rule under a key that does not
        // mention it, and one `Win::RoyalCapture` already spelled out for itself further
        // down. Both spellings are this loop now.
        //
        // The reason follows the win condition rather than whatever emptied the square: a
        // game won *by capturing* the royal reports a capture, and one where the royal was
        // never supposed to be takeable at all reports its destruction. That is the split
        // every explosion ruleset already had, restated as a consequence.
        if !t.royal_types.is_empty() {
            for side in 0..2usize {
                // Schema 18 (Q784): a royal still in hand is not yet played, not lost.
                if t.royal_loss_ends_it(side) && self.royal_gone(side) {
                    return Some(Outcome {
                        winner: (1 - side) as i8,
                        reason: match t.win {
                            Win::RoyalCapture => Reason::RoyalCaptured,
                            _ => Reason::RoyalDestroyed,
                        },
                    });
                }
            }
        }

        // Illegal position: the opponent's royal is under attack on the side to
        // move's turn, but the side to move is not itself in check.  This can only
        // happen in a hand-built or generated starting position -- normal play never
        // leaves a king in check after the move that gave it.  Treating it as an
        // immediate win for the side to move prevents the king from being captured
        // on the next ply without the game ending.
        // Only for Checkmate games, where leaving your king in check is illegal;
        // RoyalCapture games allow check and only end on actual capture.
        // When both sides are in check (e.g. on a torus), the normal checkmate /
        // stalemate path handles it via the move list.
        if !t.royal_types.is_empty() && t.win == Win::Checkmate {
            let player = self.side;
            let opponent = 1 - player;
            if self.in_check(opponent) && !self.in_check(player) {
                return Some(Outcome {
                    winner: player as i8,
                    reason: Reason::RoyalCaptured,
                });
            }
        }

        // A `match` on `t.win` until Q133, when `Win::RoyalCapture`'s arm -- its own copy
        // of the missing-royal loop -- moved into the block above, which is now both
        // spellings of that rule. `Win::Annihilation` is what was left.
        if t.win == Win::Annihilation {
            for side in 0..2usize {
                if !self.has_any_piece(side) {
                    return Some(Outcome {
                        winner: (1 - side) as i8,
                        reason: Reason::Annihilated,
                    });
                }
            }
        }

        // Schema-2 endings: asked before the draw clocks
        if t.has_v2_endings() {
            if let Some(outcome) = self.schema_two_ending() {
                return Some(outcome);
            }
        }

        // Q1069a: the draw by insufficient material, after every ending a position declares
        // and before repetition. A dead set admits no mate, so nothing above can disagree.
        if let Some(ref rule) = t.insufficient_material {
            if self.material_is_dead(rule) {
                return Some(Outcome {
                    winner: -1,
                    reason: Reason::InsufficientMaterial,
                });
            }
        }

        // Generation 42 (N4b): repetition is asked only at a turn's start, never with a duck
        // move owed.
        if !self.history.is_empty()
            && !self.duck_owed()
            && self.repetition_count() >= t.repetition_draw
        {
            return Some(Outcome {
                winner: -1,
                reason: Reason::Repetition,
            });
        }
        None
    }

    /// Q1069a: whether the material on the board is one of FIDE's dead sets, read off the
    /// compiled roles (`compile.InsufficientMaterialRule`): besides the royals, nothing, one
    /// lone leaper or rider, or only riders all of one colour class. Walks each player's
    /// pieces and stops at the first that keeps the game alive, which in an opening is the
    /// first square; never a search.
    fn material_is_dead(&self, rule: &InsufficientMaterial) -> bool {
        let t = self.t;
        let occupied = self.occupied();
        if !occupied[2].is_empty() {
            return false;
        }
        let mut minors = 0u32;
        let mut leapers = 0u32;
        let mut classes = 0u8;
        for mine in &occupied[..2] {
            for square in mine.iter() {
                match rule.roles[t.type_of(self.board[square])] {
                    INSUFFICIENT_ROYAL => {}
                    INSUFFICIENT_LEAPER => {
                        minors += 1;
                        leapers += 1;
                    }
                    INSUFFICIENT_RIDER => {
                        minors += 1;
                        classes |= 1 << (rule.colour[square] & 7);
                    }
                    _ => return false,
                }
                if minors > 1 && leapers > 0 {
                    return false;
                }
            }
        }
        minors <= 1 || classes.count_ones() == 1
    }

    /// Which clock has run out, the no-progress clock before the ply limit.
    #[inline]
    fn clock_run_out(&self) -> Option<Reason> {
        let t = self.t;
        // Generation 42 (N4b): the clocks are asked only at a turn's start (`max_plies` is
        // even beside the duck, so the ply limit falls there anyway).
        if self.duck_owed() {
            None
        } else if self.halfmove >= t.no_progress_draw {
            Some(Reason::NoProgress)
        } else if self.ply >= t.max_plies {
            Some(Reason::PlyLimit)
        } else {
            None
        }
    }

    /// Whether an empty move list here would be a checkmate rather than a stalemate.
    #[inline]
    fn could_be_mated(&self) -> bool {
        let t = self.t;
        t.win == Win::Checkmate && !t.royal_types.is_empty() && self.in_check(self.side)
    }

    /// The checkmate `moves` makes of this position, if it makes one.
    fn mate_in(&self, moves: &[u32]) -> Option<Outcome> {
        if !moves.is_empty() || !self.could_be_mated() {
            return None;
        }
        let player = self.side;
        Some(match self.t.checkmate_result {
            CheckmateResult::Loss => Outcome {
                winner: (1 - player) as i8,
                reason: Reason::Checkmate,
            },
            CheckmateResult::Win => Outcome {
                winner: player as i8,
                reason: Reason::Checkmate,
            },
        })
    }

    /// A clock that has run out, unless the move that ran it out gave mate (Q412).
    ///
    /// Until Q412 the clocks were asked before the move list, so a mate given on the ply
    /// a clock ran out was a draw here, or whatever a scoring rule made of it. FIDE's
    /// 75-move rule and Fairy-Stockfish both let such a mate stand, and the user decided
    /// on 2026-09-24 that it wins (*"mate wins"*). Only a mate: a stalemate on that ply
    /// is still the clock's, as it is in both.
    fn clock_or_mate(&self, reason: Reason, moves: &[u32]) -> Outcome {
        self.mate_in(moves)
            .unwrap_or_else(|| self.on_the_clock(reason))
    }

    /// What a clock produces: a draw, unless a scoring rule says otherwise.
    ///
    /// Deliberately *not* part of `schema_two_ending`, which runs before the clocks. A
    /// score compared at every ply would end the game at ply zero, awarding it to
    /// whoever the opening position happens to favour; the whole content of this rule
    /// is that it fires when nothing else could resolve the game.
    ///
    /// Repetition is left alone. A game that has repeated three times is drawn by
    /// agreement rather than by exhaustion, and handing it to whoever is ahead would
    /// make repetition a *tool* for converting a lead rather than a way to stop.
    fn on_the_clock(&self, reason: Reason) -> Outcome {
        let Some(ref scoring) = self.t.scoring else {
            return Outcome { winner: -1, reason };
        };
        let first = self.score_for(0, scoring);
        let second = self.score_for(1, scoring);
        if first == second {
            return Outcome { winner: -1, reason };
        }
        Outcome {
            winner: if first > second { 0 } else { 1 },
            reason: Reason::Scored,
        }
    }

    /// How many squares this side has coloured.
    #[inline]
    pub fn coloured_by(&self, player: usize) -> usize {
        self.colour_counts[player] as usize
    }

    /// One side's score under a scoring rule. Everything it counts is already on the
    /// board, which is what makes this cheap enough to sit on a terminal check.
    ///
    /// Public because `eval.rs` has a term for it: a rule that decides the game at the
    /// clock is invisible to a player that cannot read the quantity being compared, and
    /// two rulesets differing only in that rule played out identically until it could.
    pub fn score_for(&self, player: usize, scoring: &ScoringRule) -> f64 {
        // Territory is counted off the colouring rather than off the pieces, so it does
        // not belong in the loop below at all -- "hold the most ground when the board
        // fills" is about squares nobody is standing on as much as squares somebody is.
        if scoring.count == 3 {
            return self.coloured_by(player) as f64;
        }
        // Points (schema 9): what this player has been paid for taking wild pieces. Off
        // the tally rather than off the board, because what it counts is gone.
        if scoring.count == 4 {
            return self.points[player];
        }
        let t = self.t;
        let mut total = 0.0;
        // The player's pieces in ascending square order: the same terms added in the
        // same order as the scan over every square this replaced, since a square
        // holding nothing of this player's added nothing.
        for square in self.occupied()[player].iter() {
            let raw = self.board[square];
            total += match scoring.count {
                // Declared values, in board order. Board order because floating point
                // addition is not associative and a different traversal is a different
                // sum -- the same reason `eval.rs` accumulates material the way it does.
                0 => t.piece_values[t.type_of(raw)],
                2 => match scoring.squares {
                    Some(ref squares) if squares[player][square] => 1.0,
                    _ => 0.0,
                },
                _ => 1.0,
            };
        }
        total
    }

    /// Schema 18 (Q783), `extinction.per_type`: a side has run out when any one counted
    /// type is below `min_count`, each type counted on its own.
    ///
    /// The side that made the last move is read first, so a move that runs out a type of
    /// its own loses even where it also runs out one of the opponent's: promoting the last
    /// pawn while taking the last knight. That is Fairy-Stockfish's order for
    /// `extinctionPieceTypes` (the side not to move, then the side to move), and with one
    /// move a turn it is the same thing. The summed count keeps the board order it always
    /// had, so no earlier game reads differently.
    ///
    /// Schema 27 (Q936): each side is read against its own list, `side_types[side]`, which
    /// is `piece_types` for both unless the rule says `"own"` (`ExtinctionRule::side_types`).
    fn extinct_per_type(&self, extinction: &ExtinctionRule) -> Option<Outcome> {
        let mover = self.last_mover();
        for side in [mover, 1 - mover] {
            if self.short_of(side, &extinction.side_types[side], extinction.min_count) {
                return Some(decide(side, extinction.result, Reason::Extinction));
            }
        }
        None
    }

    /// Whether `side` has fewer than `min` pieces on the board of any one type in `types`,
    /// each type counted on its own (schema 27, Q936). The one count of a side's kinds
    /// against a per-side list the compiler resolved: extinction's `per_type` reads it in
    /// alternating play and in rounds. The spy does not (Q935a): *has a spy* is summed over
    /// the spy types (`Game::has_spy`), where this is short at any one type below `min`. An
    /// empty list is never short. Wild pieces are nobody's and are not in `occupied()[side]`.
    pub(crate) fn short_of(&self, side: usize, types: &[usize], min: usize) -> bool {
        if types.is_empty() {
            return false;
        }
        let t = self.t;
        let mut counts = [0usize; MAX_PIECE_TYPES];
        for square in self.occupied()[side].iter() {
            counts[t.type_of(self.board[square])] += 1;
        }
        types.iter().any(|&piece_type| counts[piece_type] < min)
    }

    /// The side whose move made this position: the side to move while its turn still
    /// has moves left in it (schema 4's long turns), the other side once it has passed.
    /// At ply zero, where nobody has moved, the second player, by the same arithmetic.
    fn last_mover(&self) -> usize {
        match self.t.turn.as_ref() {
            Some(rule) if self.remaining < rule.length_of(self.turns_taken) => self.side,
            _ => 1 - self.side,
        }
    }

    /// Schema 18 (Q783), `promotion.only_lost`: whether `promoted` may be offered to
    /// `player` promoting a `piece_type` now. Always where the type's rule does not say
    /// `only_lost`, which is one flag load for every ruleset before it; otherwise while
    /// the player has fewer of `promoted` on the board than it started with. The count is
    /// over the board as it stands before the move, which is the board after it for
    /// every piece but the one promoting, and that one is never its own target.
    #[inline]
    fn promotion_offered(&self, player: usize, piece_type: usize, promoted: u32) -> bool {
        let t = self.t;
        if !t.promotes_only_lost(piece_type) {
            return true;
        }
        let promoted = promoted as usize;
        let standing = self.occupied()[player]
            .iter()
            .filter(|&square| t.type_of(self.board[square]) == promoted)
            .count() as u32;
        standing < t.opening_count(player, promoted)
    }

    /// Schema-2 endings: reach, extinction, bare_royal, check_limit.
    fn schema_two_ending(&self) -> Option<Outcome> {
        let t = self.t;

        // Reach: win by arrival. Each side's own pieces in ascending order are the
        // same candidate squares the scan over every square found, in the same order,
        // so the first arrival it reports is the one it always reported.
        //
        // Schema 18 (Q786): or by leaving, read off the field `make` set from the last
        // move's origin, in the same slot of the ending order. A piece standing on a goal
        // then decides nothing.
        if let Some(ref reach) = t.reach {
            if reach.leaves {
                if self.goal_left_by >= 0 {
                    return Some(decide(self.goal_left_by as usize, reach.result, Reason::Reached));
                }
            } else {
                for (side, mine) in self.occupied()[..2].iter().enumerate() {
                    for square in mine.iter() {
                        if !reach.squares[side][square] {
                            continue;
                        }
                        let piece_type = t.type_of(self.board[square]);
                        if !reach.admits(piece_type) {
                            continue;
                        }
                        if reach.safe && self.threatened(1 - side, square) {
                            continue;
                        }
                        return Some(decide(side, reach.result, Reason::Reached));
                    }
                }
            }
        }

        // Extinction: loss by running out
        if let Some(ref extinction) = t.extinction {
            if extinction.per_type {
                if let Some(outcome) = self.extinct_per_type(extinction) {
                    return Some(outcome);
                }
            } else {
                for (side, mine) in self.occupied()[..2].iter().enumerate() {
                    let count = if extinction.any_piece {
                        mine.len()
                    } else {
                        mine.iter()
                            .filter(|&sq| extinction.counts(t.type_of(self.board[sq])))
                            .count()
                    };
                    if count < extinction.min_count {
                        return Some(decide(side, extinction.result, Reason::Extinction));
                    }
                }
            }
        }

        // Bare royal: the side has a royal and nothing else. Shatranj's loss.
        //
        // `!mine.is_empty()` is the whole of Q134. The second test alone reads *no
        // non-royal piece of mine*, which is **vacuously true for a side with nothing
        // left**, so the key also declared a wiped-out player an ending -- a second rule
        // its name does not mention. Unlike Q113's and Q133's, this one is not rehoused:
        // a spec that means *a side with nothing left has lost* says so itself, with
        // `win: "annihilation"` or with `extinction`, and both are read before this arm.
        if let Some(result) = t.bare_royal {
            for (side, mine) in self.occupied()[..2].iter().enumerate() {
                let bared = !mine.is_empty()
                    && !mine.iter().any(|sq| !t.is_royal(t.type_of(self.board[sq])));
                if bared {
                    return Some(decide(side, result, Reason::BareRoyal));
                }
            }
        }

        // Territory: enough of the board coloured
        if let Some(ref rule) = t.territory {
            if rule.win_at > 0 {
                for side in 0..2usize {
                    if self.coloured_by(side) >= rule.win_at as usize {
                        return Some(Outcome {
                            winner: side as i8,
                            reason: Reason::Coloured,
                        });
                    }
                }
            }
        }

        // Check limit
        if t.check_limit > 0 {
            for side in 0..2usize {
                if self.checks[side] >= t.check_limit {
                    return Some(Outcome {
                        winner: side as i8,
                        reason: Reason::CheckLimit,
                    });
                }
            }
        }

        None
    }

    /// Return the outcome if the game is over, else `None`, given an already generated
    /// legal move list.
    ///
    /// The same verdict as `terminal`, in the same order, but read off the list handed
    /// in, so it never generates one of its own.
    pub fn terminal_with(&self, moves: &[u32]) -> Option<Outcome> {
        if let Some(outcome) = self.ending_before_the_clocks() {
            return Some(outcome);
        }
        if let Some(reason) = self.clock_run_out() {
            return Some(self.clock_or_mate(reason, moves));
        }
        self.terminal_from_moves(moves)
    }

    /// The other half of the verdict: what an empty move list means.
    ///
    /// Split out because a search wants to ask the cheap half *before* it consults its
    /// transposition table -- a cached value is a value for a position that is still
    /// being played, and returning one for a position that has already ended is how a
    /// territorial win got looked up as a quiet middlegame. Generating moves to find
    /// that out would undo the table's whole purpose, and this half is the only one
    /// that needs them. (The one exception is the ply a clock runs out on with the side
    /// to move in check, where `terminal_prefix` generates them to see whether it is
    /// mate: rare, and cheaper than splitting the clocks across the two halves.)
    ///
    /// The whole verdict only once `terminal_prefix` has said `None`.
    pub fn terminal_from_moves(&self, moves: &[u32]) -> Option<Outcome> {
        if !moves.is_empty() {
            return None;
        }
        if let Some(mate) = self.mate_in(moves) {
            return Some(mate);
        }

        let player = self.side;
        Some(match self.t.stalemate {
            Stalemate::Draw => Outcome {
                winner: -1,
                reason: Reason::Stalemate,
            },
            Stalemate::Loss => Outcome {
                winner: (1 - player) as i8,
                reason: Reason::StalemateLoss,
            },
            Stalemate::Win => Outcome {
                winner: player as i8,
                reason: Reason::StalemateWin,
            },
        })
    }

    pub fn terminal(&mut self) -> Option<Outcome> {
        if let Some(outcome) = self.terminal_prefix() {
            return Some(outcome);
        }
        let moves = self.legal_moves();
        self.terminal_with(&moves)
    }

    /// Count leaf nodes at `depth`, with optional `stop_at_terminal` for schema 2.
    pub fn perft_stop(&mut self, arena: &mut Vec<u32>, depth: u32, stop_at_terminal: bool) -> u64 {
        if depth == 0 {
            return 1;
        }
        if stop_at_terminal && self.terminal_prefix().is_some() {
            return 0;
        }
        let base = arena.len();
        self.generate_legal(arena);
        let end = arena.len();
        if stop_at_terminal {
            // Check for terminal with moves (stalemate/checkmate detection)
            if end == base {
                arena.truncate(base);
                return 0; // terminal position
            }
        }
        if depth == 1 {
            arena.truncate(base);
            return (end - base) as u64;
        }
        let mut total = 0u64;
        for index in base..end {
            let m = arena[index];
            self.make(m);
            total += self.perft_stop(arena, depth - 1, stop_at_terminal);
            self.unmake();
        }
        arena.truncate(base);
        total
    }

    /// Count leaf nodes at `depth`. The engine's correctness workhorse.
    pub fn perft(&mut self, arena: &mut Vec<u32>, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let base = arena.len();
        self.generate_legal(arena);
        let end = arena.len();
        if depth == 1 {
            arena.truncate(base);
            return (end - base) as u64;
        }
        let mut total = 0u64;
        for index in base..end {
            let m = arena[index];
            self.make(m);
            total += self.perft(arena, depth - 1);
            self.unmake();
        }
        arena.truncate(base);
        total
    }
}

/// Whether a piece of `attacker` type may capture the piece encoded in `occupant`.
///
/// Free when no ruleset in play declares a capture matrix, which `may_capture`
/// short-circuits. Defined once rather than inlined at each of the three capture tests,
/// because a slide, a leap and a hop must not be able to drift apart on what a legal
/// capture is -- perft would count all three and notice none of it.
#[inline(always)]
fn may_take(t: &Tables, attacker: usize, occupant: u16) -> bool {
    if !t.has_capture_matrix {
        return true;
    }
    t.may_capture(attacker, t.type_of(occupant as u8))
}

/// Whether a move may end on an occupied square, for all three movement arms.
///
/// An enemy occupant is a capture, bound by the capture matrix. A friendly occupant is a
/// target only through an own contact link (schema 11). Both need the capture mode bit,
/// so an own contact travels the lines a capture travels. Without an own link a friendly
/// piece blocks, as it always has.
#[inline(always)]
fn may_land(t: &Tables, mode: u8, attacker: usize, occupant: u16, low: u16, high: u16) -> bool {
    if mode & 2 == 0 {
        return false;
    }
    if low <= occupant && occupant < high {
        return t.strike.as_ref().is_some_and(|rule| {
            !rule.own_link.is_empty()
                && rule.own_link[attacker * rule.n_types + t.type_of(occupant as u8)]
        });
    }
    may_take(t, attacker, occupant)
}

/// Map a stated ending result to an `Outcome`. `actor` is the side the ending
/// describes -- the one who arrives, runs out, or is stripped to royalty -- and
/// `result` is -1 loss, 0 draw, 1 win for them.
///
/// The zero case is why this takes an integer rather than a boolean. The validator
/// accepts `"draw"` for all three of these rules; a boolean has nowhere to put it, so
/// every drawn ending was silently executed as one of the other two.
fn decide(actor: usize, result: i8, reason: Reason) -> Outcome {
    let winner = match result {
        1 => actor as i8,
        -1 => (1 - actor) as i8,
        _ => -1,
    };
    Outcome { winner, reason }
}

/// The wild pieces of one phase in stepping order (schema 23): `Game::npc_step_order`.
///
/// Passes over a pending set rather than a sort: each pass visits, ascending, every pending
/// piece whose next square holds no pending piece. With one wild piece per route that is
/// the first pass alone, ascending, the order the phase always had -- but the phase and the
/// filter walk this only under `Tables::npc_queues` and otherwise walk `npc_here` directly
/// (`phase.rs::wild_phase` is monomorphised for both): walking this everywhere cost wolves' perft
/// 10%. Holds the tables and a snapshot, never the game, so the caller writes the board
/// while it walks.
pub struct NpcOrder<'t> {
    t: &'t Tables,
    pending: SquareSet,
    pass: SquareIter,
    deferred: bool,
    stepped: bool,
}

impl<'t> NpcOrder<'t> {
    #[inline]
    fn new(t: &'t Tables, here: SquareSet) -> Self {
        Self { t, pending: here, pass: here.iter(), deferred: false, stepped: false }
    }
}

impl Iterator for NpcOrder<'_> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        loop {
            if let Some(square) = self.pass.next() {
                let next = self.t.npc_next[square];
                if next >= 0 && next as usize != square && self.pending.get(next as usize) {
                    self.deferred = true;
                    continue;
                }
                self.pending.clear(square);
                self.stepped = true;
                return Some(square);
            }
            // Nothing deferred: done. Nothing stepped while something was deferred: a full
            // route, and every piece left waits.
            if !self.deferred || !self.stepped {
                return None;
            }
            self.pass = self.pending.iter();
            self.deferred = false;
            self.stepped = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::dedup_segment;
    use crate::tables::{encode_move, MOVE_EN_PASSANT, MOVE_NORMAL, MOVE_PROMOTION};

    /// The pass `dedup_segment` replaced, kept as the oracle: a second list of the moves
    /// kept so far, consulted before every push. First occurrence wins, in order.
    fn reference(out: &[u32], start: usize) -> Vec<u32> {
        let mut kept: Vec<u32> = out[..start].to_vec();
        let mut seen: Vec<u32> = Vec::new();
        for &m in &out[start..] {
            if !seen.contains(&m) {
                seen.push(m);
                kept.push(m);
            }
        }
        kept
    }

    #[test]
    fn dedup_keeps_the_first_occurrence_in_generated_order() {
        let a = encode_move(12, 20, MOVE_NORMAL, 0);
        let b = encode_move(12, 28, MOVE_NORMAL, 0);
        // The same square as `a`, another kind: a different move, and it stays.
        let c = encode_move(12, 20, MOVE_EN_PASSANT, 0);
        let d = encode_move(12, 60, MOVE_PROMOTION, 3);
        // The same square as `d`, another piece to become: also stays.
        let e = encode_move(12, 60, MOVE_PROMOTION, 4);
        // An earlier piece's move to `a`'s square, before the segment: not "seen".
        let before = encode_move(5, 20, MOVE_NORMAL, 0);
        let mut out = vec![before, a, b, a, c, d, b, e, d, a, c, e];
        let expected = reference(&out, 1);
        dedup_segment(&mut out, 1);
        assert_eq!(out, expected);
        assert_eq!(out, vec![before, a, b, c, d, e]);
    }

    #[test]
    fn dedup_leaves_the_prefix_alone_and_an_empty_segment_empty() {
        let m = encode_move(3, 9, MOVE_NORMAL, 0);
        let mut out = vec![m, m, m];
        dedup_segment(&mut out, 3);
        assert_eq!(out, vec![m, m, m]);
        // Whatever stands before `start` is another piece's business.
        dedup_segment(&mut out, 2);
        assert_eq!(out, vec![m, m, m]);
        dedup_segment(&mut out, 1);
        assert_eq!(out, vec![m, m]);
    }

    /// Many segments drawn over a small alphabet of targets, kinds and promotions, so
    /// repeats are dense, the target mask says "scan" often, and moves that share a
    /// square while differing elsewhere are common. Targets are spread over all four
    /// words of the mask.
    #[test]
    fn dedup_matches_the_list_it_replaced_on_dense_repeats() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _trial in 0..2000 {
            let start = (next() % 4) as usize;
            let length = (next() % 48) as usize;
            let origin = (next() % 256) as usize;
            let mut out: Vec<u32> = (0..start + length)
                .map(|_| {
                    let r = next();
                    let target = ((r % 7) as usize * 41) % 256;
                    let kind = ((r >> 8) % 3) as u32;
                    let promotion = ((r >> 16) % 2) as u32;
                    encode_move(origin, target, kind, promotion)
                })
                .collect();
            let expected = reference(&out, start);
            dedup_segment(&mut out, start);
            assert_eq!(out, expected);
        }
    }
}
