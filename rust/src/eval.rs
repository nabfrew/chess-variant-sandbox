//! Position evaluation, mirroring `chess_variant_sandbox/agents/evaluation.py`.
//!
//! The piece values are not computed here. Their *reach* half is measured on the
//! Python side and handed over with the tables, so there is exactly one definition of
//! how far a piece gets. What this file adds to them is the part that cannot be
//! precomputed, because it is a property of the position rather than of the rules:
//!
//! **The prey share is live.** A piece is worth what it may take, and what it may take
//! is whatever the enemy still has on the board. Taking the share over the *opening*
//! army instead -- which is what `_measured_values` used to fold into the handed-over
//! values -- makes a rock the joint most valuable piece in a game whose scissors have
//! all been gone for two hundred plies. Gated on `has_capture_matrix`, so a ruleset
//! that never restricts what a piece may capture runs the same instructions it always
//! did.
//!
//! **There is a term for the goal.** Material and mobility read a game won by capture
//! and nothing else, so an agent playing a game won by claiming ground, or by a score
//! compared at the clock, was playing the opening of a different game. Every goal term
//! lands in `[-1, 1]` as "share of the way to winning" and they share one scale,
//! stated in units of the mean non-royal piece value so that "ten percent closer"
//! means the same fraction of a piece on a board of any size. The race, the one term
//! that is a *distance* rather than a count, takes `RACE_SHARE` of that scale.
//!
//! **A piece in hand is material.** Under `drops` the hand holds pieces a player owns,
//! and a material term that read only the board paid a player for dropping a piece it
//! already had. The hand is counted at its full value (`v11` A6, measured by Q228's
//! duel); see `Evaluator::material`.

use crate::game::Game;
use crate::tables::{
    DropRule, EntryTable, ExtinctionRule, Tables, GOAL_UNREACHABLE, MAX_PIECE_TYPES, WILD,
};

pub const WIN_SCORE: f64 = 1_000_000.0;

/// Q986b (`docs/reference/round-hardest-weakness.md` §10): what a finished round is worth in
/// the round search, in mean non-royal pieces (`round_win` multiplies it out, as
/// `goal_weight` multiplies out `goal_pieces`). Not `WIN_SCORE`: regret matching+ at
/// `REGRET_ITERATIONS` resolves a matrix only to a fraction of its range, so beside a ±10⁶
/// cell every evaluation was noise and the round players spread over every move that did not
/// lose this round (§3). Not a fixed number of units either (Q986, §9): a unit is a piece on
/// the watering hole and a fifth of one on Gardner. Mirrors `ROUND_WIN_PIECES` in
/// `chess_variant_sandbox/agents/evaluation.py`; a test holds the two together.
pub const ROUND_WIN_PIECES: f64 = 100.0;

/// Schema 27 (Q942b, `docs/archive/plan-round-chase.md` §3 R1): what a rested royal that a standing
/// enemy piece reaches costs its side in a round's evaluation, in mean non-royal pieces. Half
/// a win: the strike lands next round unless the side answers it by taking the other royal in
/// the same round, so it is nearly a loss, and it stays below every finished round
/// (`round_win - ply / 1000`), so a royal actually taken is still the worse cell. Half of
/// `WIN_SCORE` until Q986b. Mirrors `RESTED_ROYAL_PIECES` in
/// `chess_variant_sandbox/agents/evaluation.py`; a test holds the two together.
pub const RESTED_ROYAL_PIECES: f64 = ROUND_WIN_PIECES / 2.0;

/// Q986b: a finished round's worth in the units of an evaluator whose piece values are
/// `values`: `ROUND_WIN_PIECES` mean non-royal pieces. `agents/evaluation.py::round_win` is
/// the oracle.
pub fn round_win(t: &Tables, values: &[f64]) -> f64 {
    ROUND_WIN_PIECES * mean_non_royal(values, &t.royal_types, t.drops.as_ref())
}

/// Schema 27 (Q935h, `docs/plan-spy.md` §6.1): what holding the only spy is worth, in mean
/// non-royal pieces, mine minus theirs. A side that has a spy while the other has none
/// answers every round with the other's move in view; the spy's `value` prices the piece
/// (it blocks, it can be taken), and this prices the rule it holds, so a holder gives its
/// spy up only for more than both. Zero while both spies stand and once both are gone, so
/// it moves nothing in a game's alternating or blind phase except at the capture that ends
/// one, and no game without a spy reads it. One mean piece, fixed by its unit as
/// `IMMUNE_PIECES` is (a holder's moving pieces are near-immune to an announced capture,
/// §1.6). Mirrors `SPY_PIECES` in `chess_variant_sandbox/agents/evaluation.py`; a test holds
/// the two together.
pub const SPY_PIECES: f64 = 1.0;

/// Whether `side`'s royal stands resting on its square with a standing enemy piece that
/// reaches it (schema 27, Q942b; over schema 29's resting list since Q983a): one of `side`'s
/// resting pieces is `side`'s royal, its only one (none other on the board, none in hand) and
/// one whose loss ends the game; `side` has no move from that square in `lists[side]` (so the
/// rest was not lifted); and the other side has a move onto it in `lists[1 - side]` (whose
/// resting pieces' moves are already gone). With every type resting 1, at most one piece
/// rests a player and this is schema 27's term to the bit; `docs/plan-rest.md` §5.4 (Q983b)
/// generalises it. `agents/evaluation.py::rested_royal_falls` is the same test.
pub fn rested_royal_falls(game: &Game<'_>, side: usize, lists: [&[u32]; 2]) -> bool {
    game.resting_of(side)
        .iter()
        .any(|&(square, _)| royal_resting_falls(game, side, square as usize, lists))
}

fn royal_resting_falls(game: &Game<'_>, side: usize, square: usize, lists: [&[u32]; 2]) -> bool {
    use crate::tables::{
        move_from, move_kind, move_to, MOVE_DROP, MOVE_NORMAL, MOVE_PORTAL, MOVE_PROMOTION,
        MOVE_SWEEP,
    };
    let t = game.t;
    let value = game.board[square];
    if value == 0 || t.owner_of(value) != side || !t.is_royal(t.type_of(value)) {
        return false;
    }
    if !t.royal_loss_ends_it(side) {
        return false;
    }
    for other in 0..t.n_squares {
        let held = game.board[other];
        if other != square && held != 0 && t.owner_of(held) == side && t.is_royal(t.type_of(held))
        {
            return false;
        }
    }
    // A royal in hand (a hand is empty in a game without drops, so no rule is asked).
    let hand = &game.hands[side];
    if t.royal_types.iter().any(|piece_type| hand.get(*piece_type).is_some_and(|n| *n > 0)) {
        return false;
    }
    // The lift's own test (`rounds.rs::generate_legal_for`): any move off the square.
    if lists[side].iter().any(|&m| move_kind(m) != MOVE_DROP && move_from(m) == square) {
        return false;
    }
    // A move that ends on an occupied square and takes what stands there (the kinds of
    // "what counts as a capture" that land on their target: a drop never lands on a piece, a
    // fold's own move takes nothing, a rotation and a castling end on the mover's squares).
    lists[1 - side].iter().any(|&m| {
        move_to(m) == square
            && matches!(move_kind(m), MOVE_NORMAL | MOVE_PROMOTION | MOVE_PORTAL | MOVE_SWEEP)
    })
}

/// Whether `side`'s royal stands where a strictly faster enemy piece takes it next round
/// (schema 28, Q943b, `docs/plan-speed.md` §3.2, §4.4): `speed` is the compiled `[type]`
/// table (non-empty). The royal is `side`'s only one (none other on the board, none in
/// hand) and one whose loss ends the game; an enemy move in `lists[1 - side]` lands on it
/// and takes from a piece of a strictly greater speed, so the royal cannot run (the
/// attacker's move is over first, §2.5 A) and cannot strike back first. It is answered
/// only by a move of `side`'s in `lists[side]` that takes the attacker where it stands from
/// a piece strictly faster than the attacker; a block is not counted (§4.4), and two such
/// attackers are two threats that one move cannot answer, so the royal falls. The same
/// landing captures as `rested_royal_falls`. `agents/evaluation.py::caught_royal_falls` is
/// the same test.
pub fn caught_royal_falls(game: &Game<'_>, side: usize, lists: [&[u32]; 2], speed: &[u8]) -> bool {
    use crate::tables::{move_from, move_kind, move_to, MOVE_NORMAL, MOVE_PORTAL, MOVE_PROMOTION, MOVE_SWEEP};
    let t = game.t;
    if !t.royal_loss_ends_it(side) {
        return false;
    }
    let mut royal = None;
    for square in 0..t.n_squares {
        let held = game.board[square];
        if held != 0 && t.owner_of(held) == side && t.is_royal(t.type_of(held)) {
            if royal.is_some() {
                return false;
            }
            royal = Some(square);
        }
    }
    let Some(square) = royal else {
        return false;
    };
    let hand = &game.hands[side];
    if t.royal_types.iter().any(|piece_type| hand.get(*piece_type).is_some_and(|n| *n > 0)) {
        return false;
    }
    let takes = |m: u32| matches!(move_kind(m), MOVE_NORMAL | MOVE_PROMOTION | MOVE_PORTAL | MOVE_SWEEP);
    let speed_at = |at: usize| speed[t.type_of(game.board[at])];
    let royal_speed = speed_at(square);
    // The first faster attacker's square; a second one is a threat no move answers.
    let mut attacker: Option<usize> = None;
    for &m in lists[1 - side] {
        if move_to(m) != square || !takes(m) {
            continue;
        }
        let origin = move_from(m);
        if speed_at(origin) <= royal_speed {
            continue;
        }
        match attacker {
            None => attacker = Some(origin),
            Some(first) if first != origin => return true,
            Some(_) => {}
        }
    }
    let Some(at) = attacker else {
        return false;
    };
    let attacker_speed = speed_at(at);
    !lists[side].iter().any(|&m| {
        move_to(m) == at && takes(m) && speed_at(move_from(m)) > attacker_speed
    })
}

/// What a piece that may capture nothing is worth, as a share of full reach. Not zero:
/// it still takes up a square and blocks a line. Mirrors `NO_PREY_FLOOR` in
/// `chess_variant_sandbox/agents/evaluation.py`, which is the definition the two sides share.
pub const NO_PREY_FLOOR: f64 = 0.1;

/// The share of the goal weight a *distance* term gets: the race to a `reach` square.
/// Mirrors `RACE_SHARE` in `chess_variant_sandbox/agents/evaluation.py`, which states
/// the reasoning; the module exports this one so a test holds the two together.
///
/// One eighth, so two mean pieces for a whole race at the default goal weight of 16:
/// stage 17 measured a distance term wanting a constant an order of magnitude below a
/// counting term's, and Q462 measured the same search at 2 beating itself at 16 0.742
/// (`weak-player-intransitive.md` §9). A power of two on purpose: at the default it is
/// bit-for-bit the weight `goal=2` gave the race before the share existed.
pub const RACE_SHARE: f64 = 0.125;

/// The immunity term's weight (Q466, F2), in mean non-royal pieces an immune piece: a
/// piece no enemy type present may ever capture counts one mean piece more, and the
/// nearest such racer adds its proximity to the goal on the same scale. Mirrors
/// `IMMUNE_PIECES` in `chess_variant_sandbox/agents/evaluation.py`, which states the
/// reasoning; the module exports this one so a test holds the two together.
///
/// The default of every greedy and minimax player on the built-in objective since Q471
/// (the user's decision C, 2026-09-25), and the weight the `decision_criticality` probe
/// reads it at. Python sends it explicitly on every `AgentSpec` and every probe, so the
/// core applies no default of its own; `immune=0` is the player without the term.
/// Zero by construction wherever `Immunity::of` finds the term undefined.
pub const IMMUNE_PIECES: f64 = 1.0;

/// The two value tables an objective needs, so neither can be passed as the other.
///
/// Mirrors `agents/evaluation.PieceValues`, and exists for the same reason: they are the
/// same shape and mean different things. `derived` is what the rules imply and it
/// normalises material, orders captures and fixes the goal weight; `objective` is what
/// *this player* thinks a piece is worth and exactly three terms read it. Two bare
/// `&[f64]` at a call site would swap silently and the symptom would be a slightly
/// different player rather than a failure.
#[derive(Clone)]
pub struct PieceValues {
    derived: Vec<f64>,
    objective: Vec<f64>,
}

impl PieceValues {
    /// No override: both halves are the table the rules imply.
    pub fn derived_only(derived: &[f64]) -> Self {
        Self {
            derived: derived.to_vec(),
            objective: derived.to_vec(),
        }
    }

    /// The pair for one player, with an empty `override_values` meaning "no opinion".
    ///
    /// `Err` where the override is not a table for *this* ruleset. The length and the
    /// royal slots are checked here rather than on the Python `Heuristic`, because this
    /// is the first place a board is in hand -- see `AgentSpec::check_against`, which is
    /// what turns this into a Python exception at the boundary.
    pub fn of(derived: &[f64], override_values: &[f64], t: &Tables) -> Result<Self, String> {
        if override_values.is_empty() {
            return Ok(Self::derived_only(derived));
        }
        Self::check(override_values, t)?;
        // The scale rule, mirroring `evaluation.rescaled`: the mean of `|value|` over
        // non-royal types equals the derived table's. `_material` divides by
        // `total_material`, which is derived and stays derived, so an override on any
        // other scale would silently rescale every other term in the vocabulary
        // relative to material. Exactly `1.0` when the override *is* the derived table,
        // which is what makes an inert override play a byte-identical game.
        let scale = mean_abs_non_royal(derived, &t.royal_types)
            / mean_abs_non_royal(override_values, &t.royal_types);
        Ok(Self {
            derived: derived.to_vec(),
            objective: override_values.iter().map(|v| v * scale).collect(),
        })
    }

    /// Whether a non-empty override is a table for *this* ruleset, without building one.
    ///
    /// Split out so the boundary can refuse before a `Player` exists, which is what
    /// makes the pairing inside `Player::new` unreachable rather than merely unlikely.
    pub fn check(override_values: &[f64], t: &Tables) -> Result<(), String> {
        if override_values.len() != t.n_types {
            return Err(format!(
                "a piece-value override for this ruleset has {} entries, got {}; that                  table was written for a different game",
                t.n_types,
                override_values.len()
            ));
        }
        for piece_type in &t.royal_types {
            if override_values[*piece_type] != 0.0 {
                return Err(format!(
                    "piece type {} is royal and its value is {}; losing a royal is                      terminal rather than material, so the evaluator has no term for it",
                    piece_type, override_values[*piece_type]
                ));
            }
        }
        if mean_abs_non_royal(override_values, &t.royal_types) == 0.0 {
            return Err(
                "every non-royal piece is worth nothing, which is not an opinion about                  the pieces but the absence of one"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// What the rules imply: capture ordering, and the normaliser.
    #[inline]
    pub fn derived(&self) -> &[f64] {
        &self.derived
    }

    /// What this player thinks: material, the prey correction, and the hand.
    #[inline]
    pub fn objective(&self) -> &[f64] {
        &self.objective
    }
}

#[derive(Clone)]
pub struct Evaluator {
    pub values: Vec<f64>,
    pub mobility_weight: f64,
    /// A5's constant, already multiplied out into material units: the goal terms are
    /// shares of the way to a win, and this is what a whole win is worth beside a
    /// piece. One number for the counting terms, never one per rule.
    goal_weight: f64,
    /// The race's weight: `goal_weight * RACE_SHARE`. One per *shape* of quantity --
    /// a distance, not a count -- and never one per rule or per ruleset.
    race_weight: f64,
    /// Whether having *more* material is what this ruleset rewards. See `material`.
    material_sign: f64,
    n_types: usize,
    /// How many board values one player owns, so this can tell whose piece a square
    /// holds without a second statement of the encoding. `n_types` where nothing turns.
    side_stride: usize,
    /// Schema 3's capture matrix, copied so `live_values` needs no tables in hand.
    capture_allowed: Vec<bool>,
    /// Which types can be won at all: the denominator of the prey share, and the pool
    /// `features::prey_weight` shares. `prey_types` is where the argument lives.
    pub prey: [bool; MAX_PIECE_TYPES],
    /// Schema 27 (Q935b): `[owner]`, a bit per spy type (`SpyTables::mask`), zero in every
    /// game without a spy. A spy is priced at its value whatever it may capture.
    spy_mask: [u32; 2],
    /// Schema 27 (Q935h): the spy term's weight in material units, `SPY_PIECES` times the
    /// mean non-royal piece; zero in every game without a spy, which then pays one
    /// comparison per evaluated node.
    spy_weight: f64,
    /// Q986b: what a rested or caught royal costs its side in a round, in material units:
    /// `RESTED_ROYAL_PIECES` times the mean non-royal piece, as `spy_weight` is.
    rested_royal: f64,
    has_capture_matrix: bool,
    /// How many pieces each side started with, of the types an `extinction` rule counts.
    /// The denominator of the shortfall term, and constant for the whole game.
    initial_counts: [usize; 2],
    /// Whether the ruleset has hands at all. Gates the hand half of `material`, so a
    /// ruleset with no `drops` runs exactly the instructions it always did.
    has_drops: bool,
    /// Q466 (F2): the immunity term's weight in material units, `immune_pieces` times
    /// the mean non-royal piece -- the unit `goal_weight` is in. `IMMUNE_PIECES` for a
    /// player that does not spell `immune=` (Q471); zero is off, and the term then costs
    /// one comparison per node. Zero as well wherever `Immunity::of` finds the term
    /// undefined, which is every ruleset without a capture matrix.
    immune_weight: f64,
    immunity: Option<Box<Immunity>>,
    /// Q708 (F1): which goal squares are held by a piece the racing side can never
    /// capture. `None` wherever that is not a fact about the capture relation alone, and
    /// then the race reads the compiled goal table as it always did.
    seal: Option<Box<Seal>>,
}

/// Q466 (F2): which pieces the opponent can never capture, read off the capture relation.
///
/// Schema 3's capture matrix makes a relation between *types*, and the relation says
/// what a person counts in *Intransitive*: once the opponent has no scissors left,
/// nothing on their side can ever take a paper, however the pieces stand. The material
/// term prices that paper at its prey share and the race term prices only the single
/// nearest piece, so the decisive exchange -- taking the last predator of a type -- read
/// as one piece of 0.3 (`docs/reference/immunity-term.md`).
///
/// `threat[player][attacker]` is the mask of victim types a piece of `attacker` owned by
/// `player` could capture, now or after any chain of promotions: a type that hunts at
/// all (the frame-0 `modes & 2` walk `prey_types` reads) and is permitted the victim.
/// A piece in hand is counted as present, since it can be dropped. Nothing here names
/// a piece; it is the compiled tables, read once.
#[derive(Clone)]
struct Immunity {
    threat: [[u32; MAX_PIECE_TYPES]; 2],
    /// Types a piece of which is material: not royal, not an unlimited drop. Only these
    /// are counted, for the reason `mean_non_royal` skips the other two.
    counted: u32,
    /// The mean non-royal piece: immune material is counted in these units.
    mean: f64,
    n_types: usize,
    side_stride: usize,
}

impl Immunity {
    /// `None` where immunity is not a fact about the capture relation alone, so the term
    /// reads zero rather than something wrong:
    ///
    /// - no capture matrix: every hunting type may take every type, so a piece is immune
    ///   only when the enemy has nothing that captures at all -- and in chess the enemy
    ///   king always does. Gating here makes chess run the instructions it always ran;
    /// - an explosion, a strike or wild pieces: a piece can leave the board, or change
    ///   type, by a route that is not an enemy piece capturing it.
    fn of(t: &Tables) -> Option<Self> {
        if !t.has_capture_matrix || t.explosion.is_some() || t.strike.is_some() || t.has_npcs {
            return None;
        }
        let n_types = t.n_types;
        let threat = capture_threats(t);
        let counted = (0..n_types)
            .filter(|piece_type| {
                !t.royal_types.contains(piece_type)
                    && !t.drops.as_ref().is_some_and(|rule| rule.is_unlimited(*piece_type))
            })
            .fold(0u32, |mask, piece_type| mask | (1 << piece_type));
        Some(Self {
            threat,
            counted,
            mean: 1.0,
            n_types,
            side_stride: t.side_stride,
        })
    }
}

/// `threat[player][attacker]`: the mask of victim types a piece of `attacker` owned by
/// `player` could capture, now or after any chain of promotions -- a type that hunts at
/// all (the frame-0 `modes & 2` walk `prey_types` reads) and is permitted the victim
/// (`may_capture`, so without a capture matrix every hunter takes every type). Read by
/// the immunity term (Q466) and the seal (Q708); `evaluation.capture_threats` is the
/// mirror.
fn capture_threats(t: &Tables) -> [[u32; MAX_PIECE_TYPES]; 2] {
    let n_types = t.n_types;
    let hunters = hunting_types(t);
    let direct: Vec<u32> = (0..n_types)
        .map(|attacker| {
            if !hunters[attacker] {
                return 0;
            }
            (0..n_types)
                .filter(|victim| t.may_capture(attacker, *victim))
                .fold(0u32, |mask, victim| mask | (1 << victim))
        })
        .collect();
    let mut threat = [[0u32; MAX_PIECE_TYPES]; 2];
    for (player, row) in threat.iter_mut().enumerate() {
        for (attacker, slot) in row[..n_types].iter_mut().enumerate() {
            // Every type this one can become, by promotion, transitively.
            let mut reach = 1u32 << attacker;
            loop {
                let mut grown = reach;
                for from in 0..n_types {
                    if reach & (1 << from) != 0 {
                        for target in t.promo_targets[player][from].iter() {
                            grown |= 1 << *target;
                        }
                    }
                }
                if grown == reach {
                    break;
                }
                reach = grown;
            }
            *slot = (0..n_types)
                .filter(|b| reach & (1 << b) != 0)
                .fold(0u32, |mask, b| mask | direct[b]);
        }
    }
    threat
}

/// Q708 (F1): a `reach` square held by a piece its racing side can never capture.
///
/// *Intransitive*'s quiet endgame (`docs/reference/intransitive-quiet-endgame.md`): with
/// two rocks each, a rock on its own home corner is a goal the other side can never
/// reach while it stays, because no rock may take a rock. The compiled goal table cannot
/// see that, so a blocker leaving its corner cost nothing and mobility pulled it out.
/// A goal square is *sealed* against `side` when it holds a piece of `side`'s opponent
/// whose type no type `side` has on the board may capture, now or after any promotion.
/// Sealed goals are no goals: the race and the immunity term's proximity read the
/// nearest unsealed one, and a side with every goal sealed is as far away as it can be.
#[derive(Clone)]
struct Seal {
    threat: [[u32; MAX_PIECE_TYPES]; 2],
    n_types: usize,
    side_stride: usize,
}

impl Seal {
    /// `None`, and the race reads the goal table unchanged, where the seal is not a fact
    /// about the capture relation alone or the race is not priced:
    ///
    /// - no priced `reach` rule (`reach_is_priceable`);
    /// - an explosion, a strike or wild pieces, which remove a piece by another route;
    /// - drops, since capturing a piece can put a new captor in the hand;
    /// - more than 64 goal squares a side, past the mask a position's seal is held in.
    fn of(t: &Tables) -> Option<Self> {
        let rule = t.reach.as_ref().filter(|rule| t.reach_is_priceable(rule))?;
        if t.explosion.is_some() || t.strike.is_some() || t.has_npcs || t.drops.is_some() {
            return None;
        }
        if rule.goals.iter().any(|goals| goals.len() > 64) {
            return None;
        }
        Some(Self {
            threat: capture_threats(t),
            n_types: t.n_types,
            side_stride: t.side_stride,
        })
    }
}

impl Evaluator {
    /// `values` are the blended piece values from `agents/evaluation.py`, and
    /// `goal_pieces` is what one whole win is worth in mean non-royal pieces. Both
    /// arrive from Python rather than being derived here, for the same reason: a
    /// second derivation is a second thing that can disagree.
    pub fn new(t: &Tables, values: Vec<f64>, mobility_weight: f64, goal_pieces: f64) -> Self {
        let goal_weight = goal_pieces * mean_non_royal(&values, &t.royal_types, t.drops.as_ref());
        let race_weight = goal_weight * RACE_SHARE;
        let material_sign = match &t.extinction {
            Some(rule) if rule.any_piece && rule.result > 0 => -1.0,
            _ => 1.0,
        };
        let spy_weight = if t.spy.is_some() {
            SPY_PIECES * mean_non_royal(&values, &t.royal_types, t.drops.as_ref())
        } else {
            0.0
        };
        let rested_royal =
            RESTED_ROYAL_PIECES * mean_non_royal(&values, &t.royal_types, t.drops.as_ref());
        let initial_counts = match &t.extinction {
            Some(rule) => [0, 1].map(|player| {
                let low = t.value_low(player) as usize;
                let high = low + t.side_stride;
                (0..t.n_squares)
                    .filter(|square| {
                        let raw = t.initial_board[*square];
                        let value = raw as usize;
                        value >= low
                            && value < high
                            && (rule.any_piece || rule.counts(t.type_of(raw)))
                    })
                    .count()
            }),
            None => [0, 0],
        };
        Self {
            values,
            mobility_weight,
            goal_weight,
            race_weight,
            material_sign,
            n_types: t.n_types,
            side_stride: t.side_stride,
            capture_allowed: t.capture_allowed.clone(),
            prey: prey_types(t),
            spy_mask: t.spy.as_ref().map_or([0, 0], |spy| spy.mask),
            spy_weight,
            rested_royal,
            has_capture_matrix: t.has_capture_matrix,
            initial_counts,
            has_drops: t.drops.is_some(),
            immune_weight: 0.0,
            immunity: None,
            seal: Seal::of(t).map(Box::new),
        }
    }

    /// Q708 (F1): per racing side, the mask of its goal squares (by index into
    /// `ReachRule::goals`) held by a piece of its opponent's that it can never capture.
    /// `[0, 0]` wherever `Seal::of` found the seal undefined; and a side's types are
    /// gathered only once an enemy piece stands on one of its goals.
    pub fn sealed_goals(&self, game: &Game<'_>) -> [u64; 2] {
        let Some(seal) = self.seal.as_deref() else {
            return [0, 0];
        };
        let Some(rule) = game.t.reach.as_ref() else {
            return [0, 0];
        };
        let board = &game.board;
        let mut sealed = [0u64; 2];
        for (side, mask) in sealed.iter_mut().enumerate() {
            let mut exposed: Option<u32> = None;
            for (index, square) in rule.goals[side].iter().enumerate() {
                let value = board[*square] as usize;
                if value == 0 || (value - 1) / seal.side_stride != 1 - side {
                    continue;
                }
                let exposed = *exposed.get_or_insert_with(|| {
                    let mut types = 0u32;
                    for held in game.occupied()[side].iter() {
                        let piece_type = (board[held] as usize - 1) % seal.n_types;
                        types |= seal.threat[side][piece_type];
                    }
                    types
                });
                if exposed & (1 << ((value - 1) % seal.n_types)) == 0 {
                    *mask |= 1 << index;
                }
            }
        }
        sealed
    }

    /// The fewest moves any of `side`'s pieces of the types in `types` needs to reach a
    /// goal square not in `sealed`, or `GOAL_UNREACHABLE`. With nothing sealed it is the
    /// compiled union table; with some it is the per-goal table over the open goals.
    fn goal_distance_of(&self, game: &Game<'_>, side: usize, types: u32, sealed: u64) -> u8 {
        let t = game.t;
        let n_types = self.n_types;
        let mut best = GOAL_UNREACHABLE;
        if sealed == 0 {
            for square in game.occupied()[side].iter() {
                let piece_type = (game.board[square] as usize - 1) % n_types;
                if types & (1 << piece_type) == 0 {
                    continue;
                }
                let distance = t.goal_distance[t.goal_slot(side, piece_type, square)];
                if distance < best {
                    best = distance;
                }
            }
            return best;
        }
        let Some(rule) = t.reach.as_ref() else {
            return best;
        };
        let n_goals = rule.goals[side].len();
        if sealed.count_ones() as usize == n_goals {
            return best;
        }
        for square in game.occupied()[side].iter() {
            let piece_type = (game.board[square] as usize - 1) % n_types;
            if types & (1 << piece_type) == 0 {
                continue;
            }
            for goal in 0..n_goals {
                if sealed & (1 << goal) != 0 {
                    continue;
                }
                let slot = t.goal_each_slot(rule, side, goal, piece_type, square);
                let distance = t.goal_distance_each[slot];
                if distance < best {
                    best = distance;
                }
            }
        }
        best
    }

    /// The same evaluator with the immunity term (Q466) at `immune_pieces` mean
    /// non-royal pieces an immune piece. Zero, or a ruleset where `Immunity::of` finds
    /// the term undefined, returns the evaluator unchanged.
    pub fn with_immunity(mut self, t: &Tables, immune_pieces: f64) -> Self {
        if immune_pieces == 0.0 {
            return self;
        }
        if let Some(mut immunity) = Immunity::of(t) {
            let mean = mean_non_royal(&self.values, &t.royal_types, t.drops.as_ref());
            self.immune_weight = immune_pieces * mean;
            immunity.mean = mean;
            self.immunity = Some(Box::new(immunity));
        }
        self
    }

    /// Q466 (F2): mine minus theirs, of the pieces nothing the opponent holds can ever
    /// capture, plus -- where a priced `reach` rule is a race -- how far the nearest such
    /// racer is along it.
    ///
    /// Per side, `worth + proximity`: `worth` is that side's immune material in mean
    /// non-royal pieces -- each immune piece at its derived value over the mean, so a
    /// piece the table prices at nothing is worth nothing more for being safe;
    /// `proximity` is `(D - d) / D` for its nearest immune piece eligible to race, at
    /// `d` moves on the compiled goal table over the board's goal diameter `D`, zero
    /// with none, and signed by the rule's result like the race term. It composes with
    /// the race term rather than replacing it: that term still reads the nearest piece
    /// of either kind, and this one adds that an immune runner is the one that arrives.
    /// Zero wherever the term is undefined, and never reached with a zero weight.
    pub fn immunity(&self, game: &Game<'_>, player: usize) -> f64 {
        let Some(imm) = self.immunity.as_deref() else {
            return 0.0;
        };
        let n_types = imm.n_types;
        let board = &game.board;
        let mut present = [0u32; 2];
        for square in game.all_occupied().iter() {
            let value = board[square] as usize - 1;
            present[value / imm.side_stride] |= 1 << (value % n_types);
        }
        if self.has_drops {
            for (side, hand) in game.hands.iter().enumerate() {
                for (piece_type, count) in hand.iter().enumerate().take(n_types) {
                    if *count > 0 {
                        present[side] |= 1 << piece_type;
                    }
                }
            }
        }
        // `safe[side]`: the types of `side`'s that nothing its opponent holds can take.
        let mut safe = [0u32; 2];
        for (side, slot) in safe.iter_mut().enumerate() {
            let enemy = 1 - side;
            let mut exposed = 0u32;
            for attacker in 0..n_types {
                if present[enemy] & (1 << attacker) != 0 {
                    exposed |= imm.threat[enemy][attacker];
                }
            }
            *slot = !exposed & imm.counted & present[side];
        }
        if safe == [0, 0] {
            return 0.0;
        }
        // Immune material, summed in board order on both sides of the boundary.
        let mut worth = [0.0f64; 2];
        for square in game.all_occupied().iter() {
            let value = board[square] as usize - 1;
            let owner = value / imm.side_stride;
            let piece_type = value % n_types;
            if safe[owner] & (1 << piece_type) != 0 {
                worth[owner] += self.values[piece_type];
            }
        }
        let t = game.t;
        let mut race = [0.0f64; 2];
        if let Some(rule) = t.reach.as_ref().filter(|rule| t.reach_is_priceable(rule)) {
            let sealed = self.sealed_goals(game);
            for (side, slot) in race.iter_mut().enumerate() {
                let mut eligible = safe[side];
                if rule.piece_type >= 0 {
                    eligible &= 1 << rule.piece_type;
                }
                if eligible == 0 {
                    continue;
                }
                let best = self.goal_distance_of(game, side, eligible, sealed[side]);
                if best != GOAL_UNREACHABLE {
                    let share = (t.goal_diameter - f64::from(best)) / t.goal_diameter;
                    *slot = clamp(share).max(0.0) * f64::from(rule.result);
                }
            }
        }
        let mine = worth[player] / imm.mean + race[player];
        let theirs = worth[1 - player] / imm.mean + race[1 - player];
        mine - theirs
    }

    /// Signed material balance from `player`'s point of view: the board, then the hands.
    ///
    /// **A piece in hand is material at its full value** (`v11` A6, Q228). Under
    /// `drops.from_captures` a capture changes a piece's owner rather than destroying
    /// it, so the hand holds pieces the player owns. Counting them at `h` times their
    /// value splits the 2v a capture and its re-drop are worth into (1 + h)v and
    /// (1 - h)v; `h = 1` is conservation -- the capture pays 2v and the drop, which
    /// moves a piece its owner already had, pays nothing. The board-only term this
    /// replaced was `h = 0`, and Q228's paired duel measured it: at 6,400 nodes the
    /// objective with the hand counted at `h = 1` took 0.93 to 0.99 of the points on all
    /// four `from_captures` games and lost nowhere (`docs/reference/hand-value-duel.md`).
    ///
    /// The sign inverts the whole balance, hand included: where running out is the win,
    /// holding a piece is holding material. Mirrors `agents/evaluation.Evaluator.material`,
    /// in the same order.
    #[inline]
    pub fn material(&self, game: &Game<'_>, player: usize) -> f64 {
        let board = self.board_material(game, player);
        if !self.has_drops {
            return board;
        }
        let hand = self.hand_value(game, player);
        if self.material_sign < 0.0 {
            board - hand
        } else {
            board + hand
        }
    }

    /// The hands, valued off this player's table: mine minus theirs.
    ///
    /// Off the table and not off the live prey share, because a piece in hand threatens
    /// nothing yet -- and because this is the quantity Q228 measured, the vocabulary's
    /// `hand_material` column (`features::hand_material`) times `total_material`, which
    /// is what lets `features.builtin_vector` spell this objective on every ruleset. An
    /// `"unlimited"` slot is priced at `INEXHAUSTIBLE_VALUE`, zero, and never empties, so
    /// it adds nothing here. Summed per side, then signed, in the order
    /// `agents/evaluation.Evaluator.hand_value` sums it.
    pub fn hand_value(&self, game: &Game<'_>, player: usize) -> f64 {
        let mut total = 0.0;
        for side in 0..2 {
            let mut worth = 0.0;
            for (piece_type, count) in game.hands[side].iter().enumerate() {
                worth += self.values[piece_type] * *count as f64;
            }
            if side == player {
                total += worth;
            } else {
                total -= worth;
            }
        }
        total
    }

    /// Signed material on the board alone, from `player`'s point of view.
    ///
    /// What `material` was before the hand was counted, and still the vocabulary's
    /// `material` column: the hand has a column of its own (`hand_material`), so a
    /// weight vector drawn before A6 means what it meant.
    ///
    /// Summed in board order, which is not incidental: the Python evaluator sums the
    /// same terms in the same order, and floating point addition is not associative.
    /// The walk is over the occupied squares in ascending order rather than over every
    /// square, which is the same terms in the same order -- an empty square added
    /// nothing -- read off the sets the game maintains instead of found by scanning.
    #[inline]
    pub fn board_material(&self, game: &Game<'_>, player: usize) -> f64 {
        // `-1.0` where an extinction rule says running out of anything is a *win*. A
        // goal term for that rule is added to material and therefore merely argues with
        // it; the sign is the only thing that settles the argument. Mirrors
        // `agents/evaluation.Evaluator.material_sign`, which carries the reasoning.
        if self.material_sign < 0.0 {
            return -self.plain_material(game, player);
        }
        if self.has_capture_matrix {
            return self.relational_material(game, player);
        }
        self.plain_material(game, player)
    }

    /// Material without the relational factor, summed in board order.
    ///
    /// **A wild piece is nobody's material** (schema 9), and the walk is what says so:
    /// `all_occupied` is the two players' union and a boar is in neither set. This sum
    /// is a *difference* between the players and its defining property is antisymmetry,
    /// `f(board, 0) == -f(board, 1)`. The test below is `owner == player`, which is
    /// false for a wild piece whichever seat is asking, so a boar reaching this loop
    /// would be subtracted from both players at once -- a legal, plausible, wrong
    /// number with nothing to point at. Hence the assertion.
    fn plain_material(&self, game: &Game<'_>, player: usize) -> f64 {
        let board = &game.board;
        let mut total = 0.0;
        for square in game.all_occupied().iter() {
            let value = board[square] as usize;
            let owner = (value - 1) / self.side_stride;
            debug_assert!(owner != WILD, "a wild piece reached plain_material");
            let piece_type = (value - 1) % self.n_types;
            if owner == player {
                total += self.values[piece_type];
            } else {
                total -= self.values[piece_type];
            }
        }
        total
    }

    /// The same sum, with each piece's value scaled by what it may still take.
    ///
    /// Two passes rather than one: the live army has to be counted before any piece can
    /// be valued, because my rock's worth is a function of their scissors. Symmetric by
    /// construction, and reached only when some piece declared `captures`.
    ///
    /// A wild piece is in neither army, by the same walk `plain_material` uses, and
    /// `scaled` has two rows -- so one arriving here would be a bounds panic rather than
    /// a wrong number.
    fn relational_material(&self, game: &Game<'_>, player: usize) -> f64 {
        let scaled = self.live_values(game);
        let board = &game.board;
        let mut total = 0.0;
        for square in game.all_occupied().iter() {
            let value = board[square] as usize;
            let owner = (value - 1) / self.side_stride;
            let piece_type = (value - 1) % self.n_types;
            if owner == player {
                total += scaled[owner][piece_type];
            } else {
                total -= scaled[owner][piece_type];
            }
        }
        total
    }

    /// `[owner][piece type]` -> that piece's value scaled by the share of the enemy
    /// **prey** it may legally capture, floored so a piece with no prey left is still
    /// worth its square.
    ///
    /// Fixed-size rather than a `Vec`: this runs once per evaluated node, and two heap
    /// allocations there cost more than the two passes over the board that are the
    /// point of the function. `MAX_PIECE_TYPES` is the validator's own cap and
    /// `Tables::from_py` refuses anything wider, so the tail is unused rather than
    /// unchecked.
    ///
    /// The share is taken over the enemy pieces that **can be won**, not over the enemy
    /// army: of the enemy material anybody could ever capture, what fraction may this
    /// piece capture. Two kinds of piece are in neither half of that fraction. A wild
    /// piece, because it is nobody's material -- the walk is the players' union, and
    /// `counts` has a row per player and none for a third owner. And a piece **no
    /// attacker may take**, because it can never be won by anyone, so a valuation
    /// derived from capture has nothing to say about it; counting one paid its placer a
    /// share of the opponent's whole army for a move that captured nothing.
    /// `agents/evaluation.py::_live_values` carries the measurement and the argument,
    /// and `features::prey_weight` says the same thing about the same quantity.
    fn live_values(&self, game: &Game<'_>) -> [[f64; MAX_PIECE_TYPES]; 2] {
        let n_types = self.n_types;
        let board = &game.board;
        let mut counts = [[0u32; MAX_PIECE_TYPES]; 2];
        for square in game.all_occupied().iter() {
            let value = board[square] as usize;
            counts[(value - 1) / self.side_stride][(value - 1) % n_types] += 1;
        }

        let mut scaled = [[0.0f64; MAX_PIECE_TYPES]; 2];
        for (owner, row) in scaled.iter_mut().enumerate() {
            let enemy = &counts[1 - owner][..n_types];
            let army: u32 = enemy
                .iter()
                .enumerate()
                .filter(|(victim, _)| self.prey[*victim])
                .map(|(_, count)| *count)
                .sum();
            let spies = self.spy_mask[owner];
            for (attacker, slot) in row[..n_types].iter_mut().enumerate() {
                // Schema 27 (Q935b, `plan-spy.md` §3.5): a spy's worth is the turn rule it
                // holds, not what it may take, so the share does not read it; a spy that
                // captures nothing would otherwise be worth a tenth of its value and be
                // given away for nothing, whatever the designer declared.
                if spies != 0 && spies & (1 << attacker) != 0 {
                    *slot = self.values[attacker];
                    continue;
                }
                // An enemy with nothing to win leaves the share undefined rather than
                // zero: a valuation derived from capture has no correction to make and
                // the derived table stands. 1.0 matches what the static share did with
                // an empty opening army, which is the case a spec with no pieces reaches.
                let share = if army == 0 {
                    1.0
                } else {
                    let mut allowed = 0u32;
                    for (victim, count) in enemy.iter().enumerate() {
                        // `prey` as well as the permission: a type outside the pool is
                        // out of the numerator too, or an attacker permitted to take one
                        // -- which the stone in `93b02b591c87` is, of its own kind --
                        // would read a share above 1.
                        if self.prey[victim] && self.capture_allowed[attacker * n_types + victim]
                        {
                            allowed += *count;
                        }
                    }
                    (allowed as f64 / army as f64).max(NO_PREY_FLOOR)
                };
                *slot = self.values[attacker] * share;
            }
        }
        scaled
    }

    /// Score the position for `player`. Higher is better. `move_count` is the mover's
    /// legal move count, which every caller already has in hand.
    #[inline]
    pub fn evaluate(&self, game: &Game<'_>, player: usize, move_count: usize) -> f64 {
        let mobility = if game.side == player {
            move_count as f64
        } else {
            -(move_count as f64)
        };
        self.score(game, player, mobility)
    }

    /// `evaluate` in a simultaneous game (schema 26, Q918b): there is no side to move and
    /// both players' moves are known, so mobility is mine minus theirs, both counted
    /// (`docs/archive/plan-simultaneous.md` §6.2). `counts` is each player's number of legal moves,
    /// in seat order. `agents/evaluation.py::Evaluator.evaluate_round` is its oracle.
    ///
    /// `lists` are both players' legal moves on this board, in seat order. Their lengths
    /// are the mobility; their contents are read only under `rest` (schema 27, Q942b), by
    /// `rested_royal_falls`: a side whose rested royal a standing enemy piece reaches is
    /// `RESTED_ROYAL_PIECES` mean pieces from a loss, mine minus theirs.
    ///
    /// Under speeds (schema 28, Q943b) the same price, once, for a royal that a strictly
    /// faster enemy piece takes next round unanswered (`caught_royal_falls`): a side whose
    /// royal falls either way is one near-loss, not two. A game without a speed reads an
    /// empty table and runs the rule-of-rest arm alone, as before.
    pub fn evaluate_round(&self, game: &Game<'_>, player: usize, lists: [&[u32]; 2]) -> f64 {
        let mobility = lists[player].len() as f64 - lists[1 - player].len() as f64;
        let mut score = self.score(game, player, mobility);
        // The resting lists are empty without a rest, and `speed` is empty without a speed,
        // so every other game pays one test and one emptiness test.
        let rested = game.t.rest.is_some()
            && game.resting.iter().flatten().any(|entry| entry.live());
        let speed = game.t.simultaneous.as_deref().map_or(&[][..], |sim| &sim.speed[..]);
        if rested || !speed.is_empty() {
            let falls = |side: usize| {
                (rested && rested_royal_falls(game, side, lists))
                    || (!speed.is_empty() && caught_royal_falls(game, side, lists, speed))
            };
            let theirs = falls(1 - player);
            let mine = falls(player);
            if theirs != mine {
                score += if theirs { self.rested_royal } else { -self.rested_royal };
            }
        }
        score
    }

    /// The objective with mobility already signed: the one body both callers share, so
    /// the alternating game's arithmetic is the same sums in the same order as before.
    #[inline]
    fn score(&self, game: &Game<'_>, player: usize, mobility: f64) -> f64 {
        let mut score = self.material(game, player);
        if self.mobility_weight != 0.0 {
            score += self.mobility_weight * mobility;
        }
        if self.goal_weight != 0.0 && game.t.has_goal_terms() {
            score += self.goal_weight * self.goal_progress(game, player);
            score += self.race_weight * self.race_progress(game, player);
        }
        if self.immune_weight != 0.0 {
            score += self.immune_weight * self.immunity(game, player);
        }
        if self.spy_weight != 0.0 {
            let edge = spy_edge(game, player);
            if edge != 0 {
                score += self.spy_weight * f64::from(edge);
            }
        }
        score
    }

    /// How far along `player` is towards the endings that are not capture and are
    /// *counted*, as a signed share of the way there. Each rule contributes its own term
    /// in `[-1, 1]`; a ruleset declaring none of them never reaches this function. The
    /// race is a distance and has its own weight: `race_progress`.
    fn goal_progress(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        let mut total = 0.0;

        // Territory that is itself a win condition. A monotone goal, so progress is
        // simply the share of the way there -- deliberately linear, see the measurement
        // in `docs/reference/stage-17-measurements.md`.
        if let Some(rule) = &t.territory {
            if rule.win_at > 0 {
                let target = rule.win_at as f64;
                let mine = game.coloured_by(player) as f64 / target;
                let theirs = game.coloured_by(1 - player) as f64 / target;
                total += clamp(mine - theirs);
            }
        }

        // A score compared when the clock runs out. It only ever fires at the clock, so
        // it is worth nothing at ply zero and everything at `max_plies`: without the
        // ramp a scoring rule would be a constant pull from the opening move, which is
        // not what the rule says and is not when it decides anything.
        if let Some(scoring) = &t.scoring {
            let mine = game.score_for(player, scoring);
            let theirs = game.score_for(1 - player, scoring);
            let spread = mine + theirs;
            if spread > 0.0 {
                let clock = (game.ply as f64 / t.max_plies as f64).min(1.0);
                total += clamp((mine - theirs) / spread) * clock;
            }
        }

        // A counter that ends the game. Both of these are already on the state, so the
        // term is a subtraction and a divide.
        if t.check_limit > 0 {
            let limit = f64::from(t.check_limit);
            let mine = f64::from(game.checks[player]) / limit;
            let theirs = f64::from(game.checks[1 - player]) / limit;
            total += clamp(mine - theirs);
        }

        if let Some(rule) = &t.extinction {
            if rule.result != 0 {
                let mine = self.shortfall(game, player, rule);
                let theirs = self.shortfall(game, 1 - player, rule);
                // Whose shortfall is good news depends on the rule. Running out is a
                // loss for the horde and a *win* for antichess, and the difference is
                // the whole reason a material evaluator reads one and not the other.
                total += clamp((theirs - mine) * -f64::from(rule.result));
            }
        }

        total
    }

    /// How far ahead `player` is in the race to a `reach` square, as a signed share of
    /// the board's goal diameter in `[-1, 1]`, or zero where no race is priced.
    ///
    /// The distance comes from the rules -- a breadth-first search over the compiled
    /// tables, done once at compile time -- because a leaper's distance to a rank is not
    /// the number of ranks and a forward-only piece may not be able to come back at all.
    ///
    /// Only where the *journey* is priced -- see `Tables::reach_is_priceable`, which
    /// rules out a `safe` arrival and a royal racing piece, both on measurements rather
    /// than on reasoning. Neither exclusion is a special case for a ruleset: each reads a
    /// rule the schema declares.
    fn race_progress(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        match t.reach.as_ref().filter(|rule| t.reach_is_priceable(rule)) {
            Some(rule) => {
                let sealed = self.sealed_goals(game);
                let mine = self.nearest_to_goal(game, player, rule.piece_type, sealed[player]);
                let theirs =
                    self.nearest_to_goal(game, 1 - player, rule.piece_type, sealed[1 - player]);
                let progress = (theirs - mine) / t.goal_diameter;
                // Stated for the side that arrives, so a rule whose result is a *loss*
                // for the arriving side -- which the schema allows -- is a race to run
                // away from.
                clamp(progress) * f64::from(rule.result)
            }
            None => 0.0,
        }
    }

    /// The fewest moves any of `player`'s eligible pieces needs to reach a goal square
    /// that is not `sealed` (Q708).
    ///
    /// `goal_diameter` where nothing can get there, so a side with no eligible piece
    /// left, or with every goal sealed, scores as far away as it is possible to be
    /// rather than as a special case. Capped there too: one open goal of several can be
    /// farther than the union table's diameter.
    fn nearest_to_goal(&self, game: &Game<'_>, player: usize, wanted: i32, sealed: u64) -> f64 {
        let t = game.t;
        let types = if wanted >= 0 { 1u32 << wanted } else { u32::MAX };
        let best = self.goal_distance_of(game, player, types, sealed);
        if best == GOAL_UNREACHABLE {
            t.goal_diameter
        } else {
            f64::from(best).min(t.goal_diameter)
        }
    }

    /// How far `player` is towards running out, as a share of the way there.
    ///
    /// Zero at the opening army and one at the threshold. Counted against the *opening*
    /// army rather than against the threshold alone, so "half your pieces are gone" says
    /// the same thing in a game with forty of them and a game with four.
    fn shortfall(&self, game: &Game<'_>, player: usize, rule: &ExtinctionRule) -> f64 {
        let n_types = self.n_types;
        let mine = &game.occupied()[player];
        let now = if rule.any_piece {
            mine.len()
        } else {
            mine.iter()
                .filter(|&square| rule.counts((game.board[square] as usize - 1) % n_types))
                .count()
        };
        let start = self.initial_counts[player];
        let room = start.saturating_sub(rule.min_count);
        if room == 0 {
            return 0.0;
        }
        (start.saturating_sub(now) as f64 / room as f64).min(1.0)
    }
}

/// Schema 27 (Q935h, `plan-spy.md` §6.1): `1` when `player` holds the only spy, `-1` when the
/// other side does, `0` while both or neither have one (`Game::has_spy`, the summed mask the
/// phase reads). Asked only in a spy game. `agents/evaluation.py::Evaluator.spy_edge` is the
/// same reading.
pub fn spy_edge(game: &Game<'_>, player: usize) -> i32 {
    i32::from(game.has_spy(player)) - i32::from(game.has_spy(1 - player))
}

#[inline]
fn clamp(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

/// Per type: can a piece of it ever be **won** -- the pool the prey share is over.
///
/// `capture_allowed` says who is *permitted* to take what, and a permission is not a
/// capture. Both halves are asked here: is there a piece that can capture at all, and
/// is it allowed to take this. The second alone is the trap, and it is not theoretical
/// -- `archive/stones` `93b02b591c87` leaves its stone's own `captures` list absent,
/// which means everything, so `capture_allowed[stone][stone]` holds and a test for a
/// column with no true cell would find the stone winnable, by a stone that has no moves.
///
/// A fact about the rules and not about the position, deliberately: restricting the
/// hunters to types a player currently holds would make the pool answer to the thing it
/// divides, which is how the defect got in.
///
/// Mirrors `agents/evaluation.prey_types`. Both read frame 0's `modes & 2` -- the same
/// walk `_measured_values` prices a piece by -- off the same compiled tables, so this is
/// one table read twice rather than a second derivation, exactly the position
/// `mean_non_royal`'s `drops.unlimited` is already in. `tests/test_backends.py` holds
/// the two together.
fn prey_types(t: &Tables) -> [bool; MAX_PIECE_TYPES] {
    let n_types = t.n_types;
    let hunting = hunting_types(t);
    let hunters: Vec<usize> = (0..n_types).filter(|a| hunting[*a]).collect();
    let mut prey = [false; MAX_PIECE_TYPES];
    for (victim, slot) in prey[..n_types].iter_mut().enumerate() {
        *slot = hunters
            .iter()
            .any(|&attacker| t.capture_allowed[attacker * n_types + victim]);
    }
    prey
}

/// Per type: does it have a capture move anywhere on an empty board -- frame 0's
/// `modes & 2`, the walk `prey_types` and the immunity term (Q466) both read.
fn hunting_types(t: &Tables) -> [bool; MAX_PIECE_TYPES] {
    let n_types = t.n_types;
    let hunts = |piece_type: usize| -> bool {
        (0..t.n_squares).any(|square| {
            let slot = t.slot(0, piece_type, square);
            // The sweep table is empty and unindexable unless the ruleset has one.
            let tables: &[&EntryTable] = if t.has_sweep {
                &[&t.leaps, &t.slides, &t.hops, &t.sweeps]
            } else {
                &[&t.leaps, &t.slides, &t.hops]
            };
            tables.iter().any(|table| {
                table
                    .entries(slot)
                    .iter()
                    .any(|entry| table.modes(entry).iter().any(|mode| mode & 2 != 0))
            })
        })
    };
    let mut hunting = [false; MAX_PIECE_TYPES];
    for (piece_type, slot) in hunting[..n_types].iter_mut().enumerate() {
        *slot = hunts(piece_type);
    }
    hunting
}

/// The mean *magnitude* of a piece that is not royal: the scale an override is on.
///
/// Magnitude rather than value, because a fitted table may price a piece negatively --
/// antichess and its like -- and a mean that cancelled would rescale such a table by an
/// arbitrary amount. Accumulated in the same order as `evaluation.mean_abs_non_royal`,
/// for the reason every other sum in this file is.
fn mean_abs_non_royal(values: &[f64], royal_types: &[usize]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for (piece_type, value) in values.iter().enumerate() {
        if royal_types.contains(&piece_type) {
            continue;
        }
        total += value.abs();
        count += 1;
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}

/// The mean value of a piece that is *material*, which is the unit the goal weight is
/// stated in. Royal pieces are valued at zero and would drag the mean towards nothing
/// on a game with several of them; an inexhaustible type is out for the reason
/// `agents/evaluation.INEXHAUSTIBLE_VALUE` gives, which is the same sentence: a piece
/// that is not material cannot be the unit material is counted in. Leaving it in
/// understated the goal weight of any game with both an endless supply and a goal rule.
///
/// `drops.unlimited` is the flattened array `agents/evaluation.inexhaustible_types`
/// reads on the other side, so this is the same table read twice and not a second
/// derivation of it -- exactly the position `royal_types` is already in. Mirrors
/// `evaluation._mean_non_royal`, skipping the same two sets in the same order.
fn mean_non_royal(values: &[f64], royal_types: &[usize], drops: Option<&DropRule>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for (piece_type, value) in values.iter().enumerate() {
        if royal_types.contains(&piece_type) {
            continue;
        }
        if drops.is_some_and(|rule| rule.is_unlimited(piece_type)) {
            continue;
        }
        total += *value;
        count += 1;
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}
