//! The players, ported from `chess_variant_sandbox/agents/`.
//!
//! Two conventions from the Python side are load-bearing and are reproduced exactly.
//! Budget is nodes, never depth, because depth is not comparable across rulesets with
//! different branching factors. And every source of randomness arrives through the
//! seat's `PyRandom`, because a game has to stay a pure function of
//! `(ruleset_hash, seed, agent_configs)`.
//!
//! Where a faster search was available but would have changed which move comes out --
//! a bounded transposition table, a different tie-break, an unstable sort -- the
//! Python behaviour won. These agents are not meant to be strong; they are meant to be
//! a *ladder*, and the index reads the rungs.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use crate::eval::{PieceValues, WIN_SCORE};
use crate::features::{Objective, VOCABULARY_LEN};
use crate::game::{Game, Outcome, Snapshot};
use crate::net::{Net, NetEvaluator};
use crate::rng::PyRandom;
use crate::tables::{
    move_kind, move_from, move_promotion, move_to, Tables, MOVE_CASTLE, MOVE_DROP,
    MOVE_DUCK, MOVE_EN_PASSANT, MOVE_FOLD, MOVE_PORTAL, MOVE_PROMOTION, MOVE_ROTATE, MOVE_SWEEP,
};

// Schema 26 (Q918b): the players for rounds, a child of this module so they share the
// player's private state (its objective, budget and counters).
#[path = "matrix.rs"]
mod matrix;
pub use matrix::{one_round_teacher, solve, REGRET_ITERATIONS};

// Schema 27 (Q935b): the searching players across a spy game's phases
// (`docs/plan-spy.md` §3), a child of this module so they reuse the search's private parts.
#[path = "spy_players.rs"]
mod spy_players;

/// How deep a quiescence search may go past the horizon before it settles for the
/// standing evaluation. Bounded because a game where every move is a capture --
/// antichess, under `must_capture` -- would otherwise let it run to the node budget on
/// every leaf, which is a full-width search wearing a different name.
const QUIESCENCE_DEPTH: i32 = 4;

/// What a stored value means. A search inside a window only ever proves a bound for
/// every move except the best, so a table that records the number without recording
/// which kind of number it is hands back a bound as though it were a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bound {
    /// The search saw every move at a full window: this is the value.
    Exact,
    /// A beta cutoff. The true value is at least this.
    Lower,
    /// Nothing beat alpha. The true value is at most this.
    Upper,
}

/// A hasher for a key that is already a hash.
///
/// The transposition table is filed under a Zobrist digest, which is a well-mixed
/// 64-bit number by construction. The default `HashMap` runs SipHash-1-3 over it at
/// every probe and every store -- mixing what is already mixed, on one of the hotter
/// instruction streams in the search.
///
/// **Invisible**, and for a stronger reason than "the numbers came out the same": a
/// `HashMap` answers on exact key equality, which no hasher can change, so every entry
/// is found exactly when it was found before. The map is only ever `get`, `insert` and
/// `clear` -- it is never iterated -- so not even an ordering is observable. This is
/// the shape of optimisation `agents.rs`'s own preamble asks for: the two
/// implementations cannot observe each other here.
#[derive(Default)]
struct ZobristHasher(u64);

impl Hasher for ZobristHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    /// The only method reached in practice: `Hash for u64` calls it directly.
    ///
    /// One multiply and one rotate, because `hashbrown` reads both ends of the value --
    /// the low bits pick the bucket and the top seven are the control byte -- and a
    /// multiply alone never carries entropy downward.
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(32);
    }

    /// Unreached while the key is a `u64`. Written correctly rather than as a panic:
    /// a key type that changes underneath this should hash *worse*, not abort.
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
}

/// Depth, value, what kind of number the value is, and the search that stored it (Q502i:
/// the generation `KEEP_TABLE` ages entries by), filed under the position's key.
type Transpositions = HashMap<u64, (i32, f64, Bound, u16), BuildHasherDefault<ZobristHasher>>;

/// Q502i: the generic search repairs (`general-player.md` §2 F), one bit each on
/// `AgentSpec::repairs`. Zero is today's search exactly, which is what every player
/// that does not spell one sends.
///
/// **Keep the table between moves.** Entries stored by this player's previous search
/// survive into the next one; anything older is dropped, so the table holds at most two
/// searches' worth. Every stored value is on the absolute-ply scale (`terminal_value`
/// discounts by `game.ply`, not by distance from the root), so an entry means the same
/// thing from any root.
pub const KEEP_TABLE: u32 = 1;
/// **Threat-aware quiescence.** Past the horizon a quiet move is chased too when, made,
/// the engine's own terminal check says its mover has won, or -- on the first quiescence
/// ply only -- it leaves the other side's royal attacked. The second is how a quiet mate,
/// a drop mate or a counted check is reached: the child quiescence node asks the
/// terminal questions before it may stand pat.
pub const THREATS: u32 = 2;
/// **Loud drops.** On the first quiescence ply a drop is chased as a capture is. Inert
/// in a game without a hand.
pub const LOUD_DROPS: u32 = 4;
/// Every bit a spec may carry. A bit outside it is a player nobody has defined.
pub const REPAIRS: u32 = KEEP_TABLE | THREATS | LOUD_DROPS;
/// The repairs a minimax player that spells none of them searches with: the ones that won
/// their pre-registered duel (`general-player.md` §4.3 E9, §6.7). `THREATS` won (0.542
/// over twelve cells); `KEEP_TABLE` and `LOUD_DROPS` lost and stay off. Python sends the
/// bits explicitly and holds its copy to this one by a test, so the core keeps no default
/// of its own -- this constant is here so that moving it moves the build fingerprint.
pub const DEFAULT_REPAIRS: u32 = THREATS;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Random,
    Greedy,
    Minimax,
    Mcts,
    /// Schema 26 (Q918b): `matrix:nodes=N`, iterative deepening on the matrix games of a
    /// simultaneous game (`matrix.rs`). Rounds only.
    Matrix,
}

#[derive(Clone, Debug)]
pub struct AgentSpec {
    pub kind: Kind,
    pub nodes: u64,
    pub mobility: f64,
    pub max_depth: u32,
    pub rollout_depth: u32,
    pub exploration: f64,
    /// What a whole win by a non-capture ending is worth, in mean non-royal pieces.
    /// An option rather than a constant because the sweep that fixed it had to be able
    /// to turn it off, and because `goal=0` is the only honest control for "did the
    /// term do anything here".
    pub goal: f64,
    /// A heuristic's weight vector over `features::VOCABULARY`, or empty for the
    /// built-in objective. Empty is what keeps every committed number where it is.
    pub weights: Vec<f64>,
    /// What this player thinks the pieces are worth, or empty for the table the rules
    /// imply. Ruleset-shaped rather than vocabulary-shaped, which is why it is a second
    /// vector and not more entries in `VOCABULARY`.
    ///
    /// It rides here *beside* the weights and does not replace the `values` argument
    /// `play_game` and friends already take: that one is the derived table, which
    /// normalises material and orders captures for every player on the board whatever
    /// its opinion of the pieces.
    pub values: Vec<f64>,
    /// Q466 (F2): the immunity term's weight, in mean non-royal pieces an immune
    /// piece; `eval::Evaluator::with_immunity`. Python sends `eval::IMMUNE_PIECES` for
    /// a player that does not spell `immune=` (Q471); zero is off.
    pub immune: f64,
    /// Q502e: the learned net this player evaluates with, parsed once from the store's
    /// bytes and shared by every game the spec plays. `None` -- no bytes, which is what
    /// an absent or refused store resolves to -- is the built-in objective, exactly.
    pub net: Option<Arc<Net>>,
    /// Q502i: which of `KEEP_TABLE`, `THREATS` and `LOUD_DROPS` this player searches
    /// with. Only minimax reads it; zero is today's search.
    pub repairs: u32,
}

impl AgentSpec {
    /// The spec these parts name, or the sentence naming the check that refused them.
    ///
    /// Two checks, and the error says which one failed. It used to be an `Option`, and
    /// `lib.rs::agent_spec` turned every `None` into "has no native implementation" --
    /// so a weight vector fitted against the previous vocabulary blamed `minimax` for
    /// its own length, and a person reading it went looking for a missing agent (Q142).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        kind: &str,
        nodes: u64,
        mobility: f64,
        max_depth: u32,
        rollout_depth: u32,
        exploration: f64,
        goal: f64,
        weights: Vec<f64>,
        values: Vec<f64>,
        immune: f64,
        net: &[u8],
        repairs: u32,
    ) -> Result<Self, String> {
        let name = kind;
        let kind = match name {
            "random" => Kind::Random,
            "greedy" => Kind::Greedy,
            "minimax" => Kind::Minimax,
            "mcts" => Kind::Mcts,
            "matrix" => Kind::Matrix,
            _ => {
                return Err(format!(
                    "{name:?} has no native implementation; the Python agent loop handles it"
                ))
            }
        };
        // A vector of the wrong length is a heuristic written against a different
        // vocabulary, and reinterpreting it would be a differently-playing agent under
        // an unchanged name. Refused here rather than truncated or padded, with both
        // lengths, because the length is the whole diagnosis: the store that supplied
        // it predates a vocabulary bump and wants refitting.
        if !weights.is_empty() && weights.len() != VOCABULARY_LEN {
            return Err(format!(
                "the weight vector for {name:?} has the wrong length (weights.len() != \
                 VOCABULARY_LEN): expected {VOCABULARY_LEN}, got {}. It was written \
                 against another feature vocabulary and is refused rather than truncated \
                 or padded; refit or re-search the store it came from",
                weights.len()
            ));
        }
        if repairs & !REPAIRS != 0 {
            return Err(format!(
                "search repairs {repairs:#x} name a bit outside {REPAIRS:#x}; a player \
                 spelling a repair this core does not define would play today's search \
                 under another name"
            ));
        }
        let net = if net.is_empty() {
            None
        } else {
            if !matches!(kind, Kind::Greedy | Kind::Minimax | Kind::Matrix) {
                return Err(format!("{name:?} has no evaluator for a net to replace"));
            }
            if !weights.is_empty() || !values.is_empty() {
                return Err(
                    "a net was given beside a weight vector or piece values; the net is the whole objective, and a player naming two is one nobody measured"
                        .to_string(),
                );
            }
            Some(Arc::new(Net::parse(net)?))
        };
        Ok(Self {
            kind,
            nodes: nodes.max(1),
            mobility,
            max_depth,
            rollout_depth,
            exploration,
            goal,
            weights,
            values,
            immune,
            net,
            repairs,
        })
    }

    /// Whether this spec is one *this ruleset* can be played with.
    ///
    /// Separate from `from_parts` because it needs a board and `from_parts` does not:
    /// the vector's exact length, and which of its slots are royal, are facts about the
    /// tables. It lives here so the rule sits next to `Player`, and it is called from
    /// `lib.rs` where a failure can still be an ordinary Python exception naming the
    /// cause -- every path into `Player::new` comes through that boundary.
    pub fn check_against(&self, t: &Tables) -> Result<(), String> {
        // Schema 26: a round is a matrix game (`docs/archive/plan-simultaneous.md` §6.2). `random`,
        // `greedy` (its matrix arm) and `matrix:nodes=N` play one; `minimax` and `mcts` are
        // refused, so one name never means two algorithms on one game. The round players
        // read the built-in objective or a learned net trained on rounds (Q918g): a
        // heuristic, fitted or searched one was made for one mover at a time. `matrix`
        // plays rounds and nothing else.
        // Schema 27 (Q935b): a spy game alternates, then plays announced rounds, then blind
        // ones (`plan-spy.md` §3.3). `random`, `greedy` and `minimax:nodes=N` play every
        // phase (`spy_players.rs`; minimax plays a blind round as `matrix:` would), so one
        // name per budget: `matrix:` and `mcts:` are refused, and so is every stored
        // objective, below with the round games'.
        if t.spy.is_some() && matches!(self.kind, Kind::Mcts | Kind::Matrix) {
            return Err(concat!(
                "a spy game (pieces[].spy) is played by random, greedy and minimax:nodes=N, ",
                "which play every phase (minimax plays the blind rounds as matrix would); ",
                "matrix and mcts are refused",
            )
            .to_string());
        }
        if t.simultaneous.is_some() {
            if matches!(self.kind, Kind::Minimax | Kind::Mcts) && t.spy.is_none() {
                return Err(concat!(
                    "a simultaneous game (rules.simultaneous) is played by random, greedy and ",
                    "matrix:nodes=N; minimax and mcts search one mover at a time",
                )
                .to_string());
            }
            // Q918g (`docs/archive/plan-simultaneous.md` §6.4): a learned net plays a round game, read
            // from seat 0 and negated for seat 1 (`net.rs`); it learns the round's own state
            // from its round segments. Not in a spy game, whose turns and rounds no net has
            // been trained across.
            if !self.weights.is_empty() || !self.values.is_empty() {
                return Err(concat!(
                    "a simultaneous game (rules.simultaneous) is played on the built-in ",
                    "objective only, or a learned net trained on rounds; a heuristic, fitted ",
                    "or searched objective was made for alternating play",
                )
                .to_string());
            }
            if self.net.is_some() && t.spy.is_some() {
                return Err(concat!(
                    "a spy game (pieces[].spy) is played on the built-in objective only; no ",
                    "learned net has been trained across its turns and rounds",
                )
                .to_string());
            }
        } else if self.kind == Kind::Matrix {
            return Err(concat!(
                "matrix plays a simultaneous game (rules.simultaneous) only; on a game that ",
                "alternates, the searching player is minimax",
            )
            .to_string());
        }
        if let Some(net) = &self.net {
            net.check_against(t)?;
        }
        if self.values.is_empty() {
            return Ok(());
        }
        if self.weights.is_empty() {
            return Err(
                "piece values were given with no weight vector, which names the \
                 built-in objective and would then quietly not use them; an override is \
                 half of a heuristic and needs the other half"
                    .to_string(),
            );
        }
        PieceValues::check(&self.values, t)
    }
}

#[derive(Clone, Copy, Default)]
pub struct Stats {
    pub nodes: u64,
    pub moves: u64,
    /// The deepest iteration of deepening the last search completed. Zero for the
    /// agents that do not deepen, and for a position with only one legal move.
    pub depth: u32,
    /// The same quantity summed over every search this player ran, with `searches`
    /// counting them. `depth` is overwritten on every move and is what a caller asking
    /// about *one* position wants; a game record wants the mean, and a mean needs both
    /// halves carried out of the game.
    ///
    /// A position with a single legal move contributes to neither: no search happened,
    /// so counting it as a zero-ply one would report the position rather than the
    /// player. The agents that never deepen leave both at zero, which is how a reader
    /// tells "searched nothing" from "searched and got nowhere".
    pub depth_sum: u64,
    pub searches: u64,
}

struct Exhausted;

struct MctsNode {
    snapshot: Snapshot,
    moves: Vec<u32>,
    children: Vec<usize>,
    visits: u64,
    wins: f64,
    parent: i32,
    edge: u32,
    outcome: Option<Outcome>,
}

pub struct Player {
    spec: AgentSpec,
    eval: Objective,
    pub stats: Stats,
    table: Transpositions,
    nodes: u64,
    arena: Vec<u32>,
    /// Two quiet moves per ply that caused a cutoff there before. Cheap, and the only
    /// ordering signal available in a game where captures are rare -- which is every
    /// ruleset whose pieces mostly cannot take each other.
    killers: Vec<[u32; 2]>,
    /// `[from * n_squares + to]`, raised on every cutoff by the square of the depth it
    /// happened at. A cutoff near the root is worth more than one at a leaf because it
    /// pruned more, and squaring the depth is the usual way of saying so.
    history: Vec<u64>,
    /// Which entries of `history` are non-zero, so a new search clears those and only
    /// those. The table is `n_squares` squared -- 32 KB on an 8x8 board, 128 KB on the
    /// largest committed one -- and it was zeroed whole before every search, which on
    /// the bottom rung of the ladder is a memset the size of the search itself. A
    /// search raises at most one entry per cutoff, so this is short, and an entry
    /// that is not on it reads exactly as the zero the memset left.
    history_touched: Vec<u32>,
    /// One `(capture gain, killer rank, history, move)` per move being ordered, reused
    /// between nodes. `order` used to rebuild that triple *inside* the comparator, for
    /// both sides of every comparison -- `2 n log n` board reads, moduli and table
    /// lookups per node where `n` is enough.
    order_keys: Vec<(f64, u8, u64, u32)>,
    n_squares: usize,
    /// Q502i: which search this is, for `KEEP_TABLE`'s aging. Wraps; only equality with
    /// the current and the previous generation is ever asked.
    generation: u16,
    /// Schema 26 (Q918b): the mixed strategy the last round decision drew from, over the
    /// player's own moves in generation order; empty for `random`. Read by the probe
    /// `Core.round_strategy`, never by play.
    pub strategy: Vec<f64>,
    /// Q986b (`docs/reference/round-hardest-weakness.md` §10): what a finished round is worth
    /// in the round search, in this player's evaluator units: `eval::ROUND_WIN_PIECES` mean
    /// non-royal pieces of the values its objective reads. Read only by a round search.
    round_win: f64,
}

/// How many plies from the root the killer table has room for. `MAX_DEPTH` is 64 and a
/// quiescence search runs past it, so this is the sum with a little room.
const MAX_PLY: usize = 96;

impl Player {
    /// `values` are the blended piece values from `agents/evaluation.py`. They arrive
    /// per call rather than living on the tables because they are an agent's opinion
    /// about the pieces, not a fact about the rules -- and because the engine has no
    /// business knowing that agents exist.
    pub fn new(t: &Tables, spec: AgentSpec, values: &[f64]) -> Self {
        let (mobility, goal, immune) = match spec.kind {
            Kind::Greedy | Kind::Minimax | Kind::Matrix => (spec.mobility, spec.goal, spec.immune),
            _ => (0.0, 0.0, 0.0),
        };
        // `values` is the derived table -- the same one for both seats, for the whole
        // game. The player's own opinion, if it has one, is on its spec, and pairing
        // them here is the only place the two meet. Refused at the boundary rather than
        // here, where there is nothing to return an error to; the rule itself is
        // `AgentSpec::check_against`, one function above.
        let pair = PieceValues::of(values, &spec.values, t)
            .expect("AgentSpec::check_against refuses this at the boundary");
        let eval = match &spec.net {
            Some(net) if matches!(spec.kind, Kind::Greedy | Kind::Minimax | Kind::Matrix) => Objective::Net(
                Box::new(NetEvaluator::new(t, Arc::clone(net), pair.derived().to_vec())),
            ),
            _ => Objective::new(t, &spec.weights, &pair, mobility, goal, immune),
        };
        let round_win = crate::eval::round_win(t, eval.values());
        Self {
            round_win,
            eval,
            spec,
            stats: Stats::default(),
            table: Transpositions::default(),
            nodes: 0,
            arena: Vec::new(),
            killers: vec![[u32::MAX; 2]; MAX_PLY],
            history: vec![0; t.n_squares * t.n_squares],
            history_touched: Vec::new(),
            order_keys: Vec::new(),
            n_squares: t.n_squares,
            generation: 0,
            strategy: Vec::new(),
        }
    }

    pub fn select_move(&mut self, game: &mut Game<'_>, moves: &[u32], rng: &mut PyRandom) -> u32 {
        match self.spec.kind {
            Kind::Random => {
                self.stats.moves += 1;
                self.stats.nodes += moves.len() as u64;
                rng.choice(moves)
            }
            // Schema 27 (Q935b): a turn of a spy game is played by `spy_turn` and its tree may
            // cross into rounds, so the spy game has its own instance of each search.
            Kind::Greedy if game.t.spy.is_some() => self.select_greedy::<true>(game, moves, rng),
            Kind::Minimax if game.t.spy.is_some() => self.select_minimax::<true>(game, moves, rng),
            Kind::Greedy => self.select_greedy::<false>(game, moves, rng),
            Kind::Minimax => self.select_minimax::<false>(game, moves, rng),
            Kind::Mcts => self.select_mcts(game, moves, rng),
            Kind::Matrix => unreachable!("check_against refuses matrix on a game that alternates"),
        }
    }
}

impl Player {
    /// One ply of lookahead: take the move with the best immediate evaluation.
    ///
    /// Ties are broken randomly rather than by move order. Without that, a greedy
    /// player is fully deterministic and every greedy-vs-greedy game is the same game,
    /// which would make the repetitiveness metric measure the agent instead of the
    /// rules.
    fn select_greedy<const SPY: bool>(
        &mut self,
        game: &mut Game<'_>,
        moves: &[u32],
        rng: &mut PyRandom,
    ) -> u32 {
        let player = game.side;
        self.stats.moves += 1;
        self.stats.nodes += moves.len() as u64;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_moves: Vec<u32> = Vec::new();
        let mut successor_moves: Vec<u32> = Vec::new();

        for m in moves {
            step::<SPY>(game, *m);
            // Schema 27 (Q935b): a spy game's turn may leave a round, valued as a round is
            // (`plan-spy.md` §3.5): its verdict from both lists, else `evaluate_round`.
            let score = if SPY && game.phase().is_round() {
                self.greedy_round_child(game, player)
            } else {
                match game.terminal_prefix() {
                    Some(outcome) => terminal_score(outcome, player),
                    None => {
                        successor_moves.clear();
                        game.generate_legal(&mut successor_moves);
                        match game.terminal_with(&successor_moves) {
                            Some(outcome) => terminal_score(outcome, player),
                            None => self.eval.evaluate(game, player, successor_moves.len()),
                        }
                    }
                }
            };
            unstep::<SPY>(game);

            if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(*m);
            } else if score == best_score {
                best_moves.push(*m);
            }
        }
        rng.choice(&best_moves)
    }

    /// Negamax with alpha-beta pruning, a transposition table, and capture ordering,
    /// under iterative deepening.
    ///
    /// Iterative deepening is not an optimisation here, it is what makes a *node*
    /// budget usable at all: the search always has a completed answer in hand when the
    /// budget runs out.
    fn select_minimax<const SPY: bool>(
        &mut self,
        game: &mut Game<'_>,
        moves: &[u32],
        rng: &mut PyRandom,
    ) -> u32 {
        self.stats.depth = 0;
        if moves.len() == 1 {
            return moves[0];
        }
        self.deepen::<SPY>(game, moves, rng).0
    }

    /// Q665 (archive/plan-teacher §8.2, T2v): the player's own search, asked for its value.
    ///
    /// `(value, move, depth)`: the root value of the last iteration of deepening this
    /// player completed, for the side to move, in its objective's units, with the move
    /// it would play and that iteration's depth. The value is exact, not a bound:
    /// `search_root`'s window is `(alpha, +inf)`, so the best move's score is searched
    /// to a value. `None` when not even depth 1 finished inside the budget.
    ///
    /// The same search as `select_move` (one body, `deepen`), with one difference: a
    /// position with a single legal move is searched too, because a label needs a value
    /// where a player needs only the move. Nothing that plays calls this, so no game,
    /// move or node count depends on it. Minimax only; the caller refuses other kinds.
    pub fn search_value(
        &mut self,
        game: &mut Game<'_>,
        moves: &[u32],
        rng: &mut PyRandom,
    ) -> Option<(f64, u32, u32)> {
        self.stats.depth = 0;
        if moves.is_empty() {
            return None;
        }
        let (best_move, value) = if game.t.spy.is_some() {
            self.deepen::<true>(game, moves, rng)
        } else {
            self.deepen::<false>(game, moves, rng)
        };
        let depth = self.stats.depth;
        if depth == 0 {
            return None;
        }
        Some((value, best_move, depth))
    }

    /// Iterative deepening at the root: the best move and its value at the last depth
    /// that finished (`NEG_INFINITY` if none did; `stats.depth` is then 0).
    fn deepen<const SPY: bool>(
        &mut self,
        game: &mut Game<'_>,
        moves: &[u32],
        rng: &mut PyRandom,
    ) -> (u32, f64) {
        self.begin_search();

        // Shuffle once so that equal-scoring moves are not resolved by generation
        // order; otherwise every game between two equal-budget minimax agents is
        // identical and the repetitiveness metric ends up measuring the search.
        let mut ordered = moves.to_vec();
        rng.shuffle(&mut ordered);
        let mut best_move = ordered[0];
        let mut best_value = f64::NEG_INFINITY;

        for depth in 1..=self.spec.max_depth as i32 {
            match self.search_root::<SPY>(game, &ordered, depth) {
                Err(Exhausted) => break,
                Ok((candidate, value)) => {
                    best_move = candidate;
                    best_value = value;
                    self.stats.depth = depth as u32;
                    // Put the previous best first: alpha-beta prunes far harder with a
                    // good first guess, so the next iteration is cheaper than this one.
                    if let Some(position) = ordered.iter().position(|m| *m == best_move) {
                        ordered.remove(position);
                    }
                    ordered.insert(0, best_move);
                    if self.nodes >= self.spec.nodes {
                        break;
                    }
                }
            }
        }

        self.end_search();
        (best_move, best_value)
    }

    /// What every search does before its first iteration: the counters, the table (kept
    /// or cleared by the repairs), the killers and the history.
    fn begin_search(&mut self) {
        self.nodes = 0;
        self.generation = self.generation.wrapping_add(1);
        if self.spec.repairs & KEEP_TABLE != 0 {
            // `retain` visits entries in bucket order, but the predicate reads only the
            // entry, so which entries survive does not depend on that order.
            let (now, previous) = (self.generation, self.generation.wrapping_sub(1));
            self.table.retain(|_, entry| entry.3 == now || entry.3 == previous);
        } else {
            self.table.clear();
        }
        for slot in &mut self.killers {
            *slot = [u32::MAX; 2];
        }
        for index in self.history_touched.drain(..) {
            self.history[index as usize] = 0;
        }
        self.stats.moves += 1;
    }

    /// What every search does after its last: the totals the record carries.
    fn end_search(&mut self) {
        self.stats.nodes += self.nodes;
        self.stats.depth_sum += self.stats.depth as u64;
        self.stats.searches += 1;
    }
}

impl Player {
    /// The best move at `depth` and its score (`alpha` when the loop ends).
    fn search_root<const SPY: bool>(
        &mut self,
        game: &mut Game<'_>,
        moves: &[u32],
        depth: i32,
    ) -> Result<(u32, f64), Exhausted> {
        let mut alpha = f64::NEG_INFINITY;
        let mut best_move = moves[0];
        let mut arena = std::mem::take(&mut self.arena);
        arena.clear();

        let mut exhausted = false;
        for m in moves {
            // See `negamax` below: the negation and the window flip are only right when
            // the side actually changed, which a turn longer than one move makes
            // conditional.
            let mover = game.side;
            step::<SPY>(game, *m);
            let flipped = game.side != mover;
            let child = if flipped {
                self.negamax::<SPY>(game, &mut arena, depth - 1, 1, f64::NEG_INFINITY, -alpha)
            } else {
                self.negamax::<SPY>(game, &mut arena, depth - 1, 1, alpha, f64::INFINITY)
            };
            unstep::<SPY>(game);
            match child {
                Err(Exhausted) => {
                    exhausted = true;
                    break;
                }
                Ok(value) => {
                    let score = if flipped { -value } else { value };
                    if score > alpha {
                        alpha = score;
                        best_move = *m;
                    }
                }
            }
        }

        self.arena = arena;
        if exhausted {
            Err(Exhausted)
        } else {
            Ok((best_move, alpha))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax<const SPY: bool>(
        &mut self,
        game: &mut Game<'_>,
        arena: &mut Vec<u32>,
        depth: i32,
        ply: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<f64, Exhausted> {
        self.nodes += 1;
        if self.nodes >= self.spec.nodes {
            return Err(Exhausted);
        }
        // Schema 27 (Q935b): a spy game's tree crosses into rounds once a spy falls in it;
        // a round node is searched by its phase (`spy_players.rs`).
        if SPY {
            let phase = game.phase();
            if phase.is_round() {
                return self.round_node(game, arena, phase, depth, ply, alpha, beta);
            }
        }

        // Asked before the table, not after, and no longer only on the rulesets whose
        // endings the evaluator reads. A cached value is a value for a position still
        // being played; a position that has already ended has an outcome instead. Two
        // positions sharing a key can differ in ply, in the no-progress clock and in how
        // often the position has repeated -- none of which the key carries -- so this is
        // not a territorial problem and never was. Ungating it moves fifteen committed
        // rulesets, which is why it waited for this phase and its campaign.
        if let Some(outcome) = game.terminal_prefix() {
            return Ok(terminal_value(outcome, game));
        }

        let key = self.table_key(game);
        if let Some((stored_depth, value, bound, _)) = self.table.get(&key) {
            // A stored number is only usable as an answer when the search that produced
            // it proved the right kind of thing. This used to return every entry as
            // though it were exact, which hands back a beta cutoff's lower bound as a
            // value -- and a lower bound is exactly what a cutoff refuses to establish.
            if *stored_depth >= depth {
                let usable = match bound {
                    Bound::Exact => true,
                    Bound::Lower => *value >= beta,
                    Bound::Upper => *value <= alpha,
                };
                if usable {
                    return Ok(*value);
                }
            }
        }

        let base = arena.len();
        game.generate_legal(arena);
        let end = arena.len();

        if let Some(outcome) = game.terminal_from_moves(&arena[base..end]) {
            arena.truncate(base);
            return Ok(terminal_value(outcome, game));
        }

        if depth <= 0 {
            // Not the standing evaluation. Iterative deepening stops on whichever depth
            // the budget allowed, which is as often odd as even, so a leaf is as likely
            // as not to be in the middle of an exchange -- and material scored there is
            // a number about a position nobody would ever stop in.
            //
            // Entered with the list this node already built (perf survey R1). Asking
            // `quiesce` would generate it again on the same position and ask the same
            // two terminal questions, both already answered "not over" above. The node
            // it counts and the budget test are kept where `quiesce` puts them, so node
            // counts and the point of exhaustion do not move.
            self.nodes += 1;
            if self.nodes >= self.spec.nodes {
                arena.truncate(base);
                return Err(Exhausted);
            }
            return self.quiesce_moves::<SPY>(game, arena, base, QUIESCENCE_DEPTH, ply, alpha, beta);
        }

        self.order(game, arena, base, end, ply);

        let opened_at = alpha;
        let mut best = f64::NEG_INFINITY;
        let mut alpha = alpha;
        for index in base..end {
            let m = arena[index];
            // Negamax negates the child and flips the window because the side is
            // assumed to alternate. When a turn is longer than one move it does not
            // always, and when the same player moves again the child is already from
            // this player's point of view -- so value and window both pass through.
            //
            // Asked as "did the side change" rather than "is there a turn rule",
            // because that is the only form that cannot drift out of step with `make`:
            // a ramp means the answer differs between one ply and the next.
            let mover = game.side;
            let quiet = !self.is_loud(game, m);
            step::<SPY>(game, m);
            let flipped = game.side != mover;
            let child = if flipped {
                self.negamax::<SPY>(game, arena, depth - 1, ply + 1, -beta, -alpha)
            } else {
                self.negamax::<SPY>(game, arena, depth - 1, ply + 1, alpha, beta)
            };
            unstep::<SPY>(game);
            let score = match child {
                Err(Exhausted) => {
                    arena.truncate(base);
                    return Err(Exhausted);
                }
                Ok(value) => if flipped { -value } else { value },
            };
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                self.remember(m, quiet, depth, ply);
                break;
            }
        }

        arena.truncate(base);
        let bound = if best <= opened_at {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.table.insert(key, (depth, best, bound, self.generation));
        Ok(best)
    }

    /// Search on past the horizon until nothing is hanging.
    ///
    /// The standing evaluation is a lower bound on what the side to move can get, since
    /// it can decline to do anything loud -- that is the "stand pat", and it is what
    /// makes this a search over a *subset* of moves rather than a broken one.
    ///
    /// Only loud moves are considered, which in a game where captures are rare is a very
    /// short list and costs almost nothing; in a game where every move is a capture it
    /// is the whole list, which is what `QUIESCENCE_DEPTH` is for.
    fn quiesce<const SPY: bool>(
        &mut self,
        game: &mut Game<'_>,
        arena: &mut Vec<u32>,
        depth: i32,
        ply: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<f64, Exhausted> {
        self.nodes += 1;
        if self.nodes >= self.spec.nodes {
            return Err(Exhausted);
        }
        // Schema 27 (Q935b): quiescence stops at a round; a capture in a round can be
        // dodged, so a round is a leaf here (`plan-spy.md` §3.3).
        if SPY && game.phase().is_round() {
            return self.round_leaf(game, arena);
        }
        if let Some(outcome) = game.terminal_prefix() {
            return Ok(terminal_value(outcome, game));
        }

        let base = arena.len();
        game.generate_legal(arena);
        let end = arena.len();
        if let Some(outcome) = game.terminal_from_moves(&arena[base..end]) {
            arena.truncate(base);
            return Ok(terminal_value(outcome, game));
        }
        self.quiesce_moves::<SPY>(game, arena, base, depth, ply, alpha, beta)
    }

    /// `quiesce` past its terminal questions: `arena[base..]` is this position's legal
    /// move list, the node is already counted, and the position is known not to be over.
    /// Truncates the arena back to `base` on every return.
    #[allow(clippy::too_many_arguments)]
    fn quiesce_moves<const SPY: bool>(
        &mut self,
        game: &mut Game<'_>,
        arena: &mut Vec<u32>,
        base: usize,
        depth: i32,
        ply: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<f64, Exhausted> {
        let end = arena.len();
        let stand_pat = self.eval.evaluate(game, game.side, end - base);
        if depth <= 0 || stand_pat >= beta {
            arena.truncate(base);
            return Ok(stand_pat);
        }

        // Keep the loud moves, in place, preserving generation order among equals so the
        // two implementations settle ties the same way.
        let horizon = depth == QUIESCENCE_DEPTH;
        let mut write = base;
        for index in base..end {
            let m = arena[index];
            if self.is_loud(game, m) || self.chases::<SPY>(game, m, horizon) {
                arena[write] = m;
                write += 1;
            }
        }
        arena.truncate(write);
        let end = write;
        self.order(game, arena, base, end, ply);

        let mut best = stand_pat;
        let mut alpha = alpha.max(stand_pat);
        for index in base..end {
            let m = arena[index];
            let mover = game.side;
            step::<SPY>(game, m);
            let flipped = game.side != mover;
            let child = if flipped {
                self.quiesce::<SPY>(game, arena, depth - 1, ply + 1, -beta, -alpha)
            } else {
                self.quiesce::<SPY>(game, arena, depth - 1, ply + 1, alpha, beta)
            };
            unstep::<SPY>(game);
            let score = match child {
                Err(Exhausted) => {
                    arena.truncate(base);
                    return Err(Exhausted);
                }
                Ok(value) => if flipped { -value } else { value },
            };
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }

        arena.truncate(base);
        Ok(best)
    }

    /// Whether a move changes material, which is what quiescence chases and what the
    /// ordering puts first.
    ///
    /// Read off the encoding and the board rather than off any piece's name: a move onto
    /// an occupied square, an en-passant capture, which lands on an empty one, and a
    /// promotion, which adds material without taking any.
    #[inline]
    fn is_loud(&self, game: &Game<'_>, m: u32) -> bool {
        let kind = move_kind(m);
        // Two kinds end on an occupied square and take nothing. A rotation's "target"
        // is the square the turning piece stands on, and a file-based castling can send
        // the king to its own rook's square -- a quiescence search that chased either
        // would never come to rest.
        // Schema 22: a fold moved onto a piece takes nothing either. Generation 42: nor does
        // the duck, whose pass names its own square.
        if kind != MOVE_ROTATE
            && kind != MOVE_CASTLE
            && kind != MOVE_FOLD
            && kind != MOVE_DUCK
            && game.board[move_to(m)] != 0
        {
            return true;
        }
        // Schema 19: a transit that promotes at its exit adds material as a promotion does.
        kind == MOVE_EN_PASSANT
            || kind == MOVE_PROMOTION
            || kind == MOVE_SWEEP
            || (kind == MOVE_PORTAL && move_promotion(m) != 0)
    }

    /// Q502i: whether quiescence chases this quiet move under the player's repairs.
    ///
    /// `horizon` is the first quiescence ply. Asked of the engine, never of a rule: the
    /// terminal check says whether the move won, `in_check` whether it attacks the other
    /// side's royal. Mirrored by `agents/minimax.py::_chases`.
    fn chases<const SPY: bool>(&self, game: &mut Game<'_>, m: u32, horizon: bool) -> bool {
        let repairs = self.spec.repairs;
        if repairs & LOUD_DROPS != 0 && horizon && move_kind(m) == MOVE_DROP {
            return true;
        }
        if repairs & THREATS == 0 {
            return false;
        }
        let mover = game.side;
        step::<SPY>(game, m);
        // Schema 27: a spy game's verdict is its phase's (a turn may leave a round).
        let verdict = if SPY { game.spy_terminal_prefix() } else { game.terminal_prefix() };
        let won = matches!(verdict, Some(outcome) if outcome.winner == mover as i8);
        let checks = !won && horizon && game.in_check(1 - mover);
        unstep::<SPY>(game);
        won || checks
    }

    /// Order a move list: captures by what they take, then killers, then history.
    ///
    /// Captures-only ordering is close to no ordering at all in a game where captures
    /// are rare, and that is most of the corpus -- it is why a branching factor of 41
    /// bought only depth 2. The two additions are the cheap standard ones and neither
    /// needs anything the rules do not already say.
    /// Decorate, sort, undecorate. The key is computed **once per move** instead of once
    /// per side of every comparison, and the comparator is the same three fields in the
    /// same order over the same values, on the same stable sort -- so the move that comes
    /// out first is the move that came out first before, including on a tie, which is the
    /// one thing this must not change.
    fn order(&mut self, game: &Game<'_>, arena: &mut [u32], base: usize, end: usize, ply: usize) {
        let mut keys = std::mem::take(&mut self.order_keys);
        keys.clear();
        {
            let board = &game.board;
            let t = game.t;
            let values = self.eval.values();
            let killers = self.killers[ply.min(MAX_PLY - 1)];
            let history = &self.history;
            let n_squares = self.n_squares;
            for &m in &arena[base..end] {
                let gain = capture_gain(t, board, m, values);
                let killer: u8 = if m == killers[0] {
                    2
                } else if m == killers[1] {
                    1
                } else {
                    0
                };
                keys.push((gain, killer, history[move_from(m) * n_squares + move_to(m)], m));
            }
        }
        // Stable, and compared field by field rather than folded into one number: a
        // single blended score would have to pick scale factors, and two orderings that
        // disagree about a tie are two different games.
        keys.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(right.1.cmp(&left.1))
                .then(right.2.cmp(&left.2))
        });
        for (slot, entry) in arena[base..end].iter_mut().zip(keys.iter()) {
            *slot = entry.3;
        }
        self.order_keys = keys;
    }

    /// Record that this move caused a cutoff, so the next search tries it sooner.
    fn remember(&mut self, m: u32, quiet: bool, depth: i32, ply: usize) {
        let slot = ply.min(MAX_PLY - 1);
        if quiet && self.killers[slot][0] != m {
            self.killers[slot][1] = self.killers[slot][0];
            self.killers[slot][0] = m;
        }
        let weight = (depth.max(0) as u64) * (depth.max(0) as u64);
        let index = move_from(m) * self.n_squares + move_to(m);
        if self.history[index] == 0 {
            self.history_touched.push(index as u32);
        }
        self.history[index] += weight;
    }

    /// The key this search's transposition table is filed under.
    ///
    /// Not `game.key`. That key deliberately omits a *permanent* colouring, and the
    /// argument for the omission is sound for the rule it was made for: a permanent
    /// claim is irreversible, so it resets the no-progress clock, so it clears the
    /// repetition window, so two positions the repetition rule ever compares always
    /// share a colouring. A search tree is not a repetition window. Two branches
    /// routinely reach the same placement having claimed different squares, and once
    /// the evaluator reads territory those are two positions at different distances
    /// from a win. Fixing `game.key` instead would make a permanent colouring enter the
    /// hash and contradict the engine's own argument; the repetition rule is right and
    /// the agent's cache is what was wrong.
    #[inline]
    fn table_key(&self, game: &Game<'_>) -> u64 {
        if game.t.territory.is_none() {
            return game.key;
        }
        game.key
            ^ (game.coloured_by(0) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (game.coloured_by(1) as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
    }
}

/// One turn: `make`, or in a spy game `spy_turn` (the initiative's deal where rounds
/// begin, `plan-spy.md` E10). A constant, so the alternating search compiles to itself.
#[inline(always)]
fn step<const SPY: bool>(game: &mut Game<'_>, m: u32) {
    if SPY {
        game.spy_turn(m);
    } else {
        game.make(m);
    }
}

/// Take back `step`.
#[inline(always)]
fn unstep<const SPY: bool>(game: &mut Game<'_>) {
    if SPY {
        game.unmake_spy_turn();
    } else {
        game.unmake();
    }
}

/// A finished position's value, from the point of view of the side to move in it.
///
/// Discounted by ply so a forced win found sooner is preferred and a loss is delayed.
/// Without it the search is indifferent between mate now and mate in six, and games
/// drift instead of finishing.
fn terminal_value(outcome: Outcome, game: &Game<'_>) -> f64 {
    if outcome.winner < 0 {
        return 0.0;
    }
    let signed = WIN_SCORE - game.ply as f64;
    if outcome.winner as usize == game.side {
        signed
    } else {
        -signed
    }
}

fn terminal_score(outcome: Outcome, player: usize) -> f64 {
    if outcome.winner < 0 {
        0.0
    } else if outcome.winner as usize == player {
        1e9
    } else {
        -1e9
    }
}

/// Order captures first, most valuable victim first. Cheap and generic.
fn capture_gain(t: &Tables, board: &[u8; 256], m: u32, values: &[f64]) -> f64 {
    // Same trap as `is_loud`, and the same two kinds: a rotation's target is its own
    // square and a file-based castling's can be its own rook's, so reading a victim off
    // the target would order both ahead of every quiet move by the value of the mover's
    // own piece.
    let kind = move_kind(m);
    if kind == MOVE_ROTATE || kind == MOVE_CASTLE || kind == MOVE_FOLD || kind == MOVE_DUCK {
        return 0.0;
    }
    let victim = board[move_to(m)];
    let mut gain = if victim == 0 { 0.0 } else { values[t.type_of(victim)] };
    // Schema 13: a sweep also takes every piece on its path, and they are summed onto
    // the target's. Mirrored by `agents/minimax.py::_capture_gain`.
    if kind == MOVE_SWEEP {
        let origin = move_from(m);
        let mut victims = t.sweep_victims(board[origin], origin, move_to(m));
        while let Some((_, swept)) = victims.next_on(board) {
            gain += values[t.type_of(swept)];
        }
    }
    gain
}

impl Player {
    /// UCT search with random rollouts; the node budget is a simulation count.
    fn select_mcts(&mut self, game: &mut Game<'_>, moves: &[u32], rng: &mut PyRandom) -> u32 {
        if moves.len() == 1 {
            return moves[0];
        }
        let root = game.snapshot();
        let mut root_moves = moves.to_vec();
        rng.shuffle(&mut root_moves);

        let mut nodes = vec![MctsNode {
            snapshot: root.clone(),
            moves: root_moves,
            children: Vec::new(),
            visits: 0,
            wins: 0.0,
            parent: -1,
            edge: u32::MAX,
            outcome: None,
        }];
        self.stats.moves += 1;

        for _ in 0..self.spec.nodes {
            let leaf = self.descend(&mut nodes, game, rng);
            let reward = self.rollout(&nodes[leaf], game, rng);
            backpropagate(&mut nodes, leaf, reward);
        }
        self.stats.nodes += self.spec.nodes;

        // `max` in Python returns the first maximal element; matching that keeps the
        // choice identical when two children have the same visit count.
        let mut best = nodes[0].children[0];
        let mut best_visits = 0u64;
        let mut first = true;
        for child in &nodes[0].children {
            let visits = nodes[*child].visits;
            if first || visits > best_visits {
                best_visits = visits;
                best = *child;
                first = false;
            }
        }
        let chosen = nodes[best].edge;
        game.restore(&root);
        chosen
    }

    fn descend(
        &self,
        nodes: &mut Vec<MctsNode>,
        game: &mut Game<'_>,
        rng: &mut PyRandom,
    ) -> usize {
        let mut current = 0usize;
        loop {
            if nodes[current].outcome.is_some() {
                return current;
            }
            if nodes[current].children.len() < nodes[current].moves.len() {
                let m = nodes[current].moves[nodes[current].children.len()];
                game.restore(&nodes[current].snapshot);
                game.make(m);
                let mut child_moves = Vec::new();
                game.generate_legal(&mut child_moves);
                let outcome = game.terminal_with(&child_moves);
                rng.shuffle(&mut child_moves);
                nodes.push(MctsNode {
                    snapshot: game.snapshot(),
                    moves: child_moves,
                    children: Vec::new(),
                    visits: 0,
                    wins: 0.0,
                    parent: current as i32,
                    edge: m,
                    outcome,
                });
                let index = nodes.len() - 1;
                nodes[current].children.push(index);
                return index;
            }
            if nodes[current].children.is_empty() {
                return current;
            }
            current = self.best_child(nodes, current);
        }
    }

    /// UCT selection.
    ///
    /// A child's `wins` are banked from the perspective of the player to move *at that
    /// child*, which is usually the opponent of the player choosing here. The
    /// exploitation term is therefore `1 - winrate`: picking the child with the highest
    /// stored score would be picking the move that most helps the opponent.
    ///
    /// "Usually" is the schema-4 caveat. When a turn is longer than one move the child
    /// can belong to the *same* player, and its winrate is then already the right way
    /// round. Read off the snapshots rather than off a turn rule, for the same reason
    /// the search does: a ramp means the answer differs from one ply to the next.
    fn best_child(&self, nodes: &[MctsNode], parent: usize) -> usize {
        let log_visits = ((nodes[parent].visits + 1) as f64).ln();
        let exploration = self.spec.exploration;
        let side = nodes[parent].snapshot.side;
        let mut best = nodes[parent].children[0];
        let mut best_key = f64::NEG_INFINITY;
        let mut first = true;
        for index in &nodes[parent].children {
            let child = &nodes[*index];
            let key = if child.visits > 0 {
                let visits = child.visits as f64;
                let exploit = if child.snapshot.side == side {
                    child.wins / visits
                } else {
                    1.0 - child.wins / visits
                };
                exploit + exploration * (log_visits / visits).sqrt()
            } else {
                f64::INFINITY
            };
            if first || key > best_key {
                best_key = key;
                best = *index;
                first = false;
            }
        }
        best
    }

    /// Play randomly to a verdict, scored for the player to move at `node`.
    ///
    /// A rollout that hits the depth cap without finishing is scored as a draw rather
    /// than by evaluating the position. Reaching for the evaluator here would fold
    /// alpha-beta's blind spots straight back into MCTS and destroy the independence
    /// that makes running both agents worthwhile.
    fn rollout(&self, node: &MctsNode, game: &mut Game<'_>, rng: &mut PyRandom) -> f64 {
        let perspective = node.snapshot.side;
        if let Some(outcome) = node.outcome {
            return outcome.score_for(perspective);
        }

        game.restore(&node.snapshot);
        let mut moves = node.moves.clone();
        for _ in 0..self.spec.rollout_depth {
            if let Some(outcome) = game.terminal_with(&moves) {
                return outcome.score_for(perspective);
            }
            if moves.is_empty() {
                return 0.5;
            }
            let m = rng.choice(&moves);
            game.make(m);
            moves.clear();
            game.generate_legal(&mut moves);
        }
        0.5
    }
}

/// `reward` is from the perspective of the player to move at the leaf, so it flips on
/// every step back towards the root at which the player actually changes -- which with
/// one move per turn is every step, and is what this used to assume outright.
fn backpropagate(nodes: &mut [MctsNode], leaf: usize, reward: f64) {
    let mut current = leaf as i32;
    let mut reward = reward;
    while current >= 0 {
        let index = current as usize;
        let side = nodes[index].snapshot.side;
        let parent = nodes[index].parent;
        nodes[index].visits += 1;
        nodes[index].wins += reward;
        if parent >= 0 && nodes[parent as usize].snapshot.side != side {
            reward = 1.0 - reward;
        }
        current = parent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(kind: &str, weights: Vec<f64>) -> Result<AgentSpec, String> {
        AgentSpec::from_parts(kind, 400, 0.05, 0, 0, 1.4, 0.0, weights, Vec::new(), 0.0, &[], 0)
    }

    /// Q142: each refusal names its own check. One blanket sentence for both once sent a
    /// person looking for a missing agent kind when the fault was a stored vector one
    /// term short of the vocabulary.
    #[test]
    fn a_vector_from_another_vocabulary_is_refused_by_length_not_by_kind() {
        let short = parts("minimax", vec![0.1; VOCABULARY_LEN - 1]).unwrap_err();
        assert!(short.contains("weights.len() != VOCABULARY_LEN"), "{short}");
        let lengths = format!("expected {VOCABULARY_LEN}, got {}", VOCABULARY_LEN - 1);
        assert!(short.contains(&lengths), "{short}");
        assert!(!short.contains("native implementation"), "{short}");
        let long = parts("minimax", vec![0.1; VOCABULARY_LEN + 1]).unwrap_err();
        let too_many = format!("got {}", VOCABULARY_LEN + 1);
        assert!(long.contains(&too_many), "{long}");
    }

    #[test]
    fn an_unknown_kind_is_the_one_refusal_that_blames_the_kind() {
        let error = parts("fairystockfish", Vec::new()).unwrap_err();
        assert!(
            error.contains("\"fairystockfish\" has no native implementation"),
            "{error}"
        );
    }

    #[test]
    fn a_repair_bit_nobody_defined_is_refused() {
        let spell = |repairs| {
            AgentSpec::from_parts(
                "minimax", 400, 0.05, 0, 0, 1.4, 0.0, Vec::new(), Vec::new(), 0.0, &[], repairs,
            )
        };
        assert_eq!(spell(REPAIRS).unwrap().repairs, REPAIRS);
        let error = spell(REPAIRS + 1).unwrap_err();
        assert!(error.contains("outside"), "{error}");
    }

    /// F10b (Q752b): the transposition table's hasher folds bytes in as FNV-1a does, so a
    /// key that is not a `u64` still hashes, and well: from FNV's offset basis, the
    /// published 64-bit FNV-1a values of "a" and "foobar".
    #[test]
    fn the_zobrist_hasher_reads_bytes_as_fnv_1a() {
        use std::hash::Hasher;
        let fnv = |bytes: &[u8]| {
            let mut hasher = ZobristHasher(0xcbf2_9ce4_8422_2325);
            hasher.write(bytes);
            hasher.finish()
        };
        assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn the_empty_vector_and_a_full_one_are_both_accepted() {
        assert!(parts("minimax", Vec::new()).is_ok());
        assert!(parts("minimax", vec![0.1; VOCABULARY_LEN]).is_ok());
        for kind in ["random", "greedy", "mcts"] {
            assert!(parts(kind, Vec::new()).is_ok(), "{kind}");
        }
    }
}
