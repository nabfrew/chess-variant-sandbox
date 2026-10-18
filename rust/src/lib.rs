//! The Python boundary.
//!
//! `chess_variant_sandbox._core.Core` is one compiled ruleset, ready to play. Python compiles the
//! rule spec, hands the tables over once, and from then on drives whole units of work
//! across this boundary -- a game, a schedule, a perft -- rather than a move at a time.
//! The per-move methods exist for tests, the visualiser and the CLI, where the
//! conversion cost is irrelevant and being able to hold a position in Python is worth
//! a great deal.

mod agents;
mod eval;
mod features;
mod game;
mod net;
mod parallel;
mod play;
mod probe;
mod rng;
mod tables;

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::agents::{AgentSpec, Player};
use crate::eval::Evaluator;
use crate::game::{count_colours, scan_occupied, Game, Snapshot};
use crate::rng::PyRandom;
use crate::tables::{SquareSet, Tables, MAX_SQUARES};

/// The full state crossing the boundary. Twelve fields: the original nine, plus checks
/// (two u32), hands (two Vec<u32>), and everything schema 4 added grouped into one --
/// the square colouring (bytes, empty when there is no territory rule), the moves left
/// in this turn and the turns taken.
///
/// Grouped rather than flat because PyO3 implements the conversion traits for tuples up
/// to twelve elements and a thirteenth simply does not compile. That is a real ceiling
/// worth knowing about: the next field added here has to join a group.
///
/// **Schema 9 is the first field to meet that ceiling.** The bounty tally joins slot 11
/// as a fourth element rather than taking a slot of its own, which is the shape the
/// docstring promised. Schema 18's side that has just left a goal square (Q786) joins
/// it as a fifth, and schema 20's straight-back bar (Q807) as a sixth. Schema 22's folds
/// (Q843c) are the seventh: the square of each fold end, in end order, empty without folds.
type StateIn = (
    Vec<u8>, u8, Vec<u8>, i16, i16, u32, u32, u64, Vec<u64>,
    (u32, u32),
    (Vec<u32>, Vec<u32>),
    (Vec<u8>, u32, u32, (f64, f64), i8, (i16, i16), Vec<i16>, Rounds),
);
/// Schema 26 (Q918a): the initiative, the void run and the bars, slot 11's eighth element;
/// schema 27 (Q942a) adds the rested squares.
type Rounds = (i8, (i16, i16), Vec<(i16, i16)>, (i16, i16));
type StateOut<'py> = (
    Bound<'py, PyBytes>,
    u8,
    Bound<'py, PyBytes>,
    i16,
    i16,
    u32,
    u32,
    u64,
    Vec<u64>,
    (u32, u32),
    (Vec<u32>, Vec<u32>),
    (Bound<'py, PyBytes>, u32, u32, (f64, f64), i8, (i16, i16), Vec<i16>, Rounds),
);

/// What the wild-piece phase did on the way to a position (schema 9): one
/// `(from, to, eaten)` per wild piece that walked, oldest step first.
///
/// `eaten` is the board byte the destination held at the instant the wild piece wrote
/// over it, which is the half a caller cannot recover from two positions. It travels
/// beside the position rather than inside it because the position is already at PyO3's
/// twelve-slot ceiling -- and because it describes the *transition*, not the state.
type PhaseOut = Vec<(u32, u32, u8)>;

/// Schema 26 (Q918a): what a round's meetings did, `(event, square, player)` per
/// `game::rounds::EVENT_*`, beside the position as the phase is.
type RoundOut = Vec<(u8, u32, i8)>;

/// `(kind, nodes, mobility, max_depth, rollout_depth, exploration, goal, weights,
/// values)`, resolved by each agent's `register_native` resolver from the registry's own
/// defaults.
///
/// The seventh slot is stage 17A's: what a whole win by a non-capture ending is worth,
/// in mean non-royal pieces. It travels rather than living as a constant here for the
/// same reason the piece values do -- a second definition is a second thing that can
/// disagree with the first.
///
/// The ninth is stage 30's: what this player thinks the pieces are worth. It is
/// *beside* the `values` argument each method below takes, not a replacement for it --
/// that one is the derived table, one per game, which normalises material and orders
/// captures for every player whatever its opinion. Nine of PyO3's twelve.
///
/// The tenth is Q466's: the immunity term's weight, in mean non-royal pieces an immune
/// piece, zero for every player that does not ask for it.
///
/// The eleventh is Q502e's: a learned net's store bytes (`net.rs`), empty for every
/// player without one -- which is also what an absent or refused store resolves to, so
/// that player is the built-in objective exactly.
///
/// The twelfth is Q502i's: the search repairs (`agents::KEEP_TABLE`, `THREATS`,
/// `LOUD_DROPS`) as bits, zero for today's search. Twelve is PyO3's ceiling, which is
/// why three flags share one slot: the next field needs a struct.
type AgentIn = (String, u64, f64, u32, u32, f64, f64, Vec<f64>, Vec<f64>, f64, Vec<u8>, u32);

/// Everything a `GameRecord` needs, minus the fields Python already knows.
///
/// Ten of the twelve slots PyO3 will convert. The last three are the search's cost:
/// nodes visited, plies completed, and searches run, each per seat.
type RecordOut = (
    Vec<u32>,
    Option<i8>,
    &'static str,
    Vec<u32>,
    Vec<f64>,
    Vec<u8>,
    (u64, u64),
    (u64, u64),
    (u64, u64),
    f64,
);

/// A position as Python holds it, ready for the core. The two tallies the game
/// maintains -- who stands where, how much is coloured -- are counted here, once, at
/// the boundary; inside the core they travel with the snapshot.
fn snapshot_from(t: &Tables, state: &StateIn) -> Snapshot {
    let mut board = [0u8; MAX_SQUARES];
    let len = state.0.len().min(MAX_SQUARES);
    board[..len].copy_from_slice(&state.0[..len]);
    Snapshot {
        occupied: scan_occupied(t, &board),
        colour_counts: count_colours(&state.11.0),
        board,
        side: state.1 as usize,
        unmoved: SquareSet::from_bytes(&state.2),
        ep_target: state.3,
        ep_victim: state.4,
        halfmove: state.5,
        ply: state.6,
        key: state.7,
        repetition: state.8.clone(),
        checks: [state.9.0, state.9.1],
        hands: [state.10.0.clone(), state.10.1.clone()],
        colour: state.11.0.clone(),
        remaining: state.11.1,
        turns_taken: state.11.2,
        points: [state.11.3.0, state.11.3.1],
        goal_left_by: state.11.4,
        portal_bar: [state.11.5.0, state.11.5.1],
        fold_at: folds_in(t, &state.11.6),
        initiative: state.11.7.0,
        void_run: [state.11.7.1.0, state.11.7.1.1],
        bars: state.11.7.2.iter().map(|&(a, b)| [a, b]).collect(),
        rest: [state.11.7.3.0, state.11.7.3.1],
    }
}

/// Schema 22: the folds as Python holds them -- one square per fold end, in end order --
/// as `Game::fold_at`, indexed by end. An empty tuple is where the folds start, which is
/// what a position built by hand without saying where they stand means.
fn folds_in(t: &Tables, folds: &[i16]) -> [i16; 8] {
    let Some(starts) = t.portals.as_deref().and_then(|p| p.folds.as_ref()).map(|f| f.starts)
    else {
        return [-1; 8];
    };
    if folds.is_empty() {
        return starts;
    }
    let mut at = [-1i16; 8];
    let mut given = folds.iter();
    for end in 0..8 {
        if starts[end] >= 0 {
            at[end] = given.next().copied().unwrap_or(starts[end]);
        }
    }
    at
}

/// Schema 22: `Game::fold_at` as Python holds it, the fold ends' squares in end order.
fn folds_out(game: &Game<'_>) -> Vec<i16> {
    let Some(starts) = game.t.portals.as_deref().and_then(|p| p.folds.as_ref()).map(|f| f.starts)
    else {
        return Vec::new();
    };
    (0..8).filter(|end| starts[*end] >= 0).map(|end| game.fold_at[end]).collect()
}

fn state_out<'py>(py: Python<'py>, game: &Game<'_>) -> StateOut<'py> {
    (
        PyBytes::new(py, &game.board[..game.t.n_squares]),
        game.side as u8,
        PyBytes::new(py, &game.unmoved.to_bytes()),
        game.ep_target,
        game.ep_victim,
        game.halfmove,
        game.ply,
        game.key,
        game.history[game.window..].to_vec(),
        (game.checks[0], game.checks[1]),
        (game.hands[0].clone(), game.hands[1].clone()),
        (
            PyBytes::new(py, &game.colour),
            game.remaining,
            game.turns_taken,
            (game.points[0], game.points[1]),
            game.goal_left_by,
            (game.portal_bar[0], game.portal_bar[1]),
            folds_out(game),
            (
                game.initiative,
                (game.void_run[0], game.void_run[1]),
                game.bars.iter().map(|bar| (bar[0], bar[1])).collect(),
                (game.rest[0], game.rest[1]),
            ),
        ),
    )
}

/// One agent, checked against the ruleset it is about to play.
///
/// The check is here rather than inside `Player::new` because this is the last point at
/// which a refusal can be an ordinary Python exception naming its cause: below this,
/// a player is built inside `py.detach` on a rayon pool. Every path to a `Player` comes
/// through here, which is what makes the pairing in `Player::new` unreachable.
///
/// Every refusal carries `from_parts`'s or `check_against`'s own sentence, never one
/// written here: a blanket message at this line once blamed the agent kind for a weight
/// vector of the wrong length (Q142).
fn agent_spec(t: &Tables, agent: &AgentIn) -> PyResult<AgentSpec> {
    let spec = AgentSpec::from_parts(
        &agent.0,
        agent.1,
        agent.2,
        agent.3,
        agent.4,
        agent.5,
        agent.6,
        agent.7.clone(),
        agent.8.clone(),
        agent.9,
        &agent.10,
        agent.11,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    spec.check_against(t)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(spec)
}

fn record_out(output: play::GameOutput) -> RecordOut {
    (
        output.moves,
        if output.winner < 0 {
            None
        } else {
            Some(output.winner)
        },
        output.reason,
        output.branching,
        output.material,
        output.captures,
        (output.nodes[0], output.nodes[1]),
        (output.depth_sum[0], output.depth_sum[1]),
        (output.searches[0], output.searches[1]),
        output.duration_ms,
    )
}

/// One compiled ruleset. Construct once per ruleset and reuse.
#[pyclass(module = "chess_variant_sandbox._core", frozen)]
pub struct Core {
    tables: Arc<Tables>,
}

/// Schema 27 (Q935b): a spy game's round is chosen with `select_round`, never one move at
/// a time.
fn refuse_a_round(tables: &Tables, snapshot: &Snapshot) -> PyResult<()> {
    if tables.spy.is_some() {
        let phase = Game::from_snapshot(tables, snapshot).phase();
        if phase.is_round() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "this spy game is in its {} phase, played in rounds; ask select_round",
                phase.name()
            )));
        }
    }
    Ok(())
}

#[pymethods]
impl Core {
    #[new]
    fn new(tables: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            tables: Arc::new(Tables::from_py(tables)?),
        })
    }

    fn initial_state<'py>(&self, py: Python<'py>) -> StateOut<'py> {
        let game = Game::initial(&self.tables);
        state_out(py, &game)
    }

    fn legal_moves(&self, state: StateIn) -> Vec<u32> {
        let snapshot = snapshot_from(&self.tables, &state);
        Game::from_snapshot(&self.tables, &snapshot).legal_moves()
    }

    fn pseudo_legal_moves(&self, state: StateIn) -> Vec<u32> {
        let snapshot = snapshot_from(&self.tables, &state);
        let game = Game::from_snapshot(&self.tables, &snapshot);
        let mut moves = Vec::new();
        game.generate_pseudo_legal(&mut moves);
        moves
    }

    /// The position after `m`, and what the wild-piece phase did to reach it.
    ///
    /// The phase is the half of a ply no diff of two positions recovers: it runs after
    /// the mover's move inside the same `make`, so one move makes two board changes and
    /// a caller with only the end position has to work the second one out. It gets the
    /// steps themselves instead.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: StateIn,
        m: u32,
    ) -> PyResult<(StateOut<'py>, PhaseOut)> {
        // Schema 27 (Q935a): a spy game's turn, in an alternating or reply phase only, played
        // by `spy_turn` so the initiative's deal (E10) is made where rounds begin.
        if self.tables.spy.is_some() {
            let snapshot = snapshot_from(&self.tables, &state);
            let mut game = Game::from_snapshot(&self.tables, &snapshot);
            let phase = game.phase();
            if phase.is_round() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "this spy game is in its {} phase, played in rounds; apply both players'                      moves together with apply_round",
                    phase.name()
                )));
            }
            game.spy_turn(m);
            let steps = game
                .phase_steps()
                .iter()
                .map(|&(from, to, eaten)| (from as u32, to as u32, eaten))
                .collect();
            return Ok((state_out(py, &game), steps));
        }
        // Schema 26 (Q918a): one move alone is not a round, and `make` would play the game
        // as if it alternated. Refused here, where it is still an ordinary exception.
        if self.tables.simultaneous.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                concat!(
                    "this game is played in rounds (rules.simultaneous); apply both players' ",
                    "moves together with apply_round",
                ),
            ));
        }
        let snapshot = snapshot_from(&self.tables, &state);
        let mut game = Game::from_snapshot(&self.tables, &snapshot);
        // A castling move names only the king's squares, so applying one to a position
        // with no partner rook is the one way a caller holding a hand-built move can
        // reach an unrecoverable state inside the core. Reject it here, where it can
        // still be an ordinary Python exception.
        if tables::move_kind(m) == tables::MOVE_CASTLE
            && game
                .castling_partner(tables::move_from(m), tables::move_to(m))
                .is_none()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "castling move applied to a position with no partner rook",
            ));
        }
        game.make(m);
        // The game was built from the snapshot a moment ago, so its phase log holds
        // this move's steps and nothing else.
        let phase = game
            .phase_steps()
            .iter()
            .map(|&(from, to, eaten)| (from as u32, to as u32, eaten))
            .collect();
        Ok((state_out(py, &game), phase))
    }

    /// Schema 26 (Q918a): every move `player` may choose in a round on this board.
    fn legal_moves_for(&self, state: StateIn, player: u8) -> PyResult<Vec<u32>> {
        if self.tables.simultaneous.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "legal_moves_for is for a simultaneous game; this one alternates",
            ));
        }
        let snapshot = snapshot_from(&self.tables, &state);
        let mut game = Game::from_snapshot(&self.tables, &snapshot);
        let mut moves = Vec::new();
        game.generate_legal_for(player as usize, &mut moves);
        Ok(moves)
    }

    /// Schema 26 (Q918a): the position after one round, `first` by player 0 and `second` by
    /// player 1, with what the wild-piece phase did and what the round's meetings did
    /// (`(event, square, player)`, `game::EVENT_*`) for the page to draw.
    fn apply_round<'py>(
        &self,
        py: Python<'py>,
        state: StateIn,
        first: u32,
        second: u32,
    ) -> PyResult<(StateOut<'py>, PhaseOut, RoundOut)> {
        if self.tables.simultaneous.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "apply_round is for a simultaneous game; this one alternates",
            ));
        }
        let snapshot = snapshot_from(&self.tables, &state);
        let mut game = Game::from_snapshot(&self.tables, &snapshot);
        // Schema 27 (Q935a): a spy game plays rounds only once a spy is lost.
        if self.tables.spy.is_some() {
            let phase = game.phase();
            if !phase.is_round() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "this spy game is in its {} phase, played one move at a time; apply the                      move with apply",
                    phase.name()
                )));
            }
        }
        game.make_round([first, second]);
        let phase = game
            .phase_steps()
            .iter()
            .map(|&(from, to, eaten)| (from as u32, to as u32, eaten))
            .collect();
        let events = game
            .round_log
            .iter()
            .map(|&(event, square, player)| (event, square as u32, player))
            .collect();
        Ok((state_out(py, &game), phase, events))
    }

    /// What the next `count` wild-piece phases would do on the board as it stands (Q660,
    /// Q764), each in `apply`'s phase shape and each played on the board the one before
    /// it left: where each wild piece is going next and after that, for the board to
    /// draw. Every phase empty for a ruleset without wild pieces.
    fn coming_phases(&self, state: StateIn, count: usize) -> Vec<PhaseOut> {
        let snapshot = snapshot_from(&self.tables, &state);
        let mut game = Game::from_snapshot(&self.tables, &snapshot);
        game.coming_phases(count)
            .into_iter()
            .map(|phase| {
                phase
                    .into_iter()
                    .map(|(from, to, eaten)| (from as u32, to as u32, eaten))
                    .collect()
            })
            .collect()
    }

    #[pyo3(signature = (state, moves = None))]
    fn terminal(
        &self,
        state: StateIn,
        moves: Option<Vec<u32>>,
    ) -> Option<(Option<i8>, &'static str)> {
        let snapshot = snapshot_from(&self.tables, &state);
        let mut game = Game::from_snapshot(&self.tables, &snapshot);
        let outcome = match moves {
            // Schema 27 (Q935a): a spy game's verdict is its phase's; a turn's from the
            // caller's list where there is one.
            Some(moves) if self.tables.spy.is_some() && !game.phase().is_round() => {
                game.terminal_with(&moves)
            }
            _ if self.tables.spy.is_some() => game.spy_terminal(),
            // Schema 26: a round's verdict needs both players' lists, so it makes its own.
            _ if self.tables.simultaneous.is_some() => game.round_terminal(),
            Some(moves) => game.terminal_with(&moves),
            None => game.terminal(),
        };
        outcome.map(|outcome| {
            (
                if outcome.winner < 0 {
                    None
                } else {
                    Some(outcome.winner)
                },
                outcome.reason.as_str(),
            )
        })
    }

    /// The half of the verdict that needs no move list.
    ///
    /// Exposed for one caller: `agents/minimax.py`, the declared oracle for
    /// `agents.rs`, has to ask the same question in the same place. Its search consults
    /// its transposition table after this and before generating moves, and an oracle
    /// that instead asked the *whole* verdict first would be a different search --
    /// silently, and only in the positions where a cached value and an ended game meet.
    fn terminal_prefix(&self, state: StateIn) -> Option<(Option<i8>, &'static str)> {
        let snapshot = snapshot_from(&self.tables, &state);
        let mut game = Game::from_snapshot(&self.tables, &snapshot);
        let outcome = if self.tables.spy.is_some() {
            game.spy_terminal_prefix()
        } else if self.tables.simultaneous.is_some() {
            game.round_terminal_prefix()
        } else {
            game.terminal_prefix()
        };
        outcome.map(|outcome| {
            (
                if outcome.winner < 0 {
                    None
                } else {
                    Some(outcome.winner)
                },
                outcome.reason.as_str(),
            )
        })
    }

    /// Schema 27 (Q935a, `docs/plan-spy.md` §2.5): a spy game's phase here, `alternating`,
    /// `reply`, `announced` or `blind`, and the announcer (-1 outside an announced round).
    /// The one place the phase is worked out: Python asks, and never answers.
    fn phase(&self, state: StateIn) -> PyResult<(&'static str, i8)> {
        if self.tables.spy.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "phase is for a spy game (pieces[].spy); this one has no spy",
            ));
        }
        let snapshot = snapshot_from(&self.tables, &state);
        let phase = Game::from_snapshot(&self.tables, &snapshot).phase();
        Ok((phase.name(), phase.announcer()))
    }

    fn in_check(&self, state: StateIn, player: u8) -> bool {
        let snapshot = snapshot_from(&self.tables, &state);
        Game::from_snapshot(&self.tables, &snapshot).in_check(player as usize)
    }

    /// The slow definition of the Zobrist hash: a full scan. `make` maintains the key
    /// incrementally, and a test holds the two to agreeing.
    fn compute_hash(&self, state: StateIn) -> u64 {
        let snapshot = snapshot_from(&self.tables, &state);
        Game::from_snapshot(&self.tables, &snapshot).compute_hash()
    }

    #[pyo3(signature = (state, depth, stop_at_terminal = false, threads = 1))]
    fn perft(&self, py: Python<'_>, state: StateIn, depth: u32, stop_at_terminal: bool, threads: usize) -> u64 {
        let snapshot = snapshot_from(&self.tables, &state);
        py.detach(|| parallel::perft(&self.tables, &snapshot, depth, threads, stop_at_terminal))
    }

    #[pyo3(signature = (agents, values, seed, collect_series = true))]
    fn play_game(
        &self,
        py: Python<'_>,
        agents: (AgentIn, AgentIn),
        values: Vec<f64>,
        seed: u64,
        collect_series: bool,
    ) -> PyResult<RecordOut> {
        let specs = [
            agent_spec(&self.tables, &agents.0)?,
            agent_spec(&self.tables, &agents.1)?,
        ];
        let output =
            py.detach(|| play::play_game(&self.tables, &specs, &values, seed, collect_series));
        Ok(record_out(output))
    }

    /// One position, one decision: what would this agent play *here*?
    ///
    /// The boundary otherwise carries only whole units of work -- a game, a schedule, a
    /// perft -- which left every caller that needs a single move (an interactive
    /// opponent, a position probe) driving the Python agents instead: a different
    /// implementation from the one whose rating the index reads. This closes that,
    /// taking the same `AgentIn` the tournament passes to `play_game` so there is no
    /// second way to describe an agent.
    ///
    /// The seat stream is fresh, derived from `seed` exactly as seat 0 of a game is.
    /// A caller that wants a stream continued across moves should be playing a game --
    /// `play_from` is that call.
    ///
    /// `None` where the position has no legal move, which is the caller's cue to ask
    /// `terminal` rather than to treat a missing move as an error.
    #[pyo3(signature = (state, agent, values, seed = 0))]
    fn select_move(
        &self,
        py: Python<'_>,
        state: StateIn,
        agent: AgentIn,
        values: Vec<f64>,
        seed: u64,
    ) -> PyResult<Option<u32>> {
        let spec = agent_spec(&self.tables, &agent)?;
        let snapshot = snapshot_from(&self.tables, &state);
        refuse_a_round(&self.tables, &snapshot)?;
        Ok(py.detach(|| {
            let mut game = Game::from_snapshot(&self.tables, &snapshot);
            let legal = game.legal_moves();
            if legal.is_empty() {
                return None;
            }
            let mut player = Player::new(&self.tables, spec, &values);
            let mut rng = PyRandom::for_seat(seed, 0);
            Some(player.select_move(&mut game, &legal, &mut rng))
        }))
    }

    /// Schema 26 (Q918b): one seat's choice in a round, and what it was drawn from.
    ///
    /// `(move, strategy, nodes, depth)`: the move the agent plays for `seat` on this board,
    /// the mixed strategy over that seat's legal moves (`legal_moves_for`'s order) it drew
    /// it from -- empty for `random` -- the nodes it spent and its last finished pass. The
    /// seat's stream is fresh, derived from `seed` and `seat` exactly as that seat's is at
    /// the start of a game. `None` for the move where either player has none (the round
    /// is over before anyone chooses). The probe the oracle's tests and the play page ask.
    ///
    /// Schema 27 (Q935b, `docs/plan-spy.md` §3.2): in a spy game's announced round the seat
    /// says which choice is asked. The announcer's is asked with `announced` absent; the
    /// answerer's needs it, the move the announcer showed. Both are pure, so the strategy is
    /// the chosen move alone. A blind round is asked as any round; a turn is `select_move`'s.
    #[pyo3(signature = (state, seat, agent, values, seed = 0, announced = None))]
    #[allow(clippy::too_many_arguments)]
    fn select_round(
        &self,
        py: Python<'_>,
        state: StateIn,
        seat: u8,
        agent: AgentIn,
        values: Vec<f64>,
        seed: u64,
        announced: Option<u32>,
    ) -> PyResult<(Option<u32>, Vec<f64>, u64, u32)> {
        if self.tables.simultaneous.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "select_round is for a simultaneous game; this one alternates",
            ));
        }
        let seat = seat as usize;
        if seat > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err("a seat is 0 or 1"));
        }
        let spec = agent_spec(&self.tables, &agent)?;
        let snapshot = snapshot_from(&self.tables, &state);
        let phase = if self.tables.spy.is_some() {
            Some(Game::from_snapshot(&self.tables, &snapshot).phase())
        } else {
            None
        };
        let refuse = |words: String| Err(pyo3::exceptions::PyValueError::new_err(words));
        match (phase, announced) {
            (Some(phase), _) if !phase.is_round() => {
                return refuse(format!(
                    "this spy game is in its {} phase, played one move at a time; ask \
                     select_move",
                    phase.name()
                ));
            }
            (Some(game::Phase::Announced { announcer }), None) if seat != announcer => {
                return refuse(
                    "the answerer's choice in an announced round needs the announced move"
                        .to_string(),
                );
            }
            (Some(game::Phase::Announced { announcer }), Some(_)) if seat == announcer => {
                return refuse(
                    "the announcer chooses before any move is announced; leave announced out"
                        .to_string(),
                );
            }
            (Some(game::Phase::Blind), Some(_)) | (None, Some(_)) => {
                return refuse("only an announced round of a spy game has an announced move".to_string());
            }
            _ => {}
        }
        Ok(py.detach(|| {
            let mut game = Game::from_snapshot(&self.tables, &snapshot);
            let mut first = Vec::new();
            let mut second = Vec::new();
            game.generate_legal_for(0, &mut first);
            game.generate_legal_for(1, &mut second);
            if first.is_empty() || second.is_empty() {
                return (None, Vec::new(), 0, 0);
            }
            let mut player = Player::new(&self.tables, spec, &values);
            let mut rng = PyRandom::for_seat(seed, seat);
            let lists = [&first[..], &second[..]];
            let chosen = match (phase, announced) {
                (Some(game::Phase::Announced { .. }), Some(shown)) => {
                    player.select_answer(&mut game, seat, lists, shown, &mut rng)
                }
                (Some(game::Phase::Announced { .. }), None) => {
                    player.select_announcement(&mut game, seat, lists, &mut rng)
                }
                _ => player.select_round(&mut game, seat, lists, &mut rng),
            };
            (Some(chosen), player.strategy.clone(), player.stats.nodes, player.stats.depth)
        }))
    }

    /// `select_move`, plus how deep the search actually got.
    ///
    /// A search's depth is a property of the position and the budget together, and no
    /// caller can compute it from the outside. The heuristic pass needs it because a
    /// "deep" arm that reaches two plies on a wide board makes every game look shallow
    /// for a reason that has nothing to do with the rules -- so the control has to be
    /// measured rather than assumed.
    #[pyo3(signature = (state, agent, values, seed = 0))]
    fn select_move_depth(
        &self,
        py: Python<'_>,
        state: StateIn,
        agent: AgentIn,
        values: Vec<f64>,
        seed: u64,
    ) -> PyResult<(Option<u32>, u32)> {
        let spec = agent_spec(&self.tables, &agent)?;
        let snapshot = snapshot_from(&self.tables, &state);
        refuse_a_round(&self.tables, &snapshot)?;
        Ok(py.detach(|| {
            let mut game = Game::from_snapshot(&self.tables, &snapshot);
            let legal = game.legal_moves();
            if legal.is_empty() {
                return (None, 0);
            }
            let mut player = Player::new(&self.tables, spec, &values);
            let mut rng = PyRandom::for_seat(seed, 0);
            let chosen = player.select_move(&mut game, &legal, &mut rng);
            (Some(chosen), player.stats.depth)
        }))
    }

    /// Q502e: a net's raw output along a line of moves, for the Python oracle to check.
    ///
    /// One `(raw, inputs, fresh)` per position -- the start and after each move: `raw`
    /// from one evaluator whose accumulator is carried along the line (the incremental
    /// path the search uses), `inputs` the active input indices, and `fresh` from an
    /// accumulator summed from nothing. The moves must be legal in turn.
    ///
    /// Q918g: in a round game the moves are a record's flat list, each round's two in seat
    /// order, and a position is the start and after each round (`make_round`); an odd
    /// count is refused. A spy game, whose steps mix turns and rounds, is refused.
    fn net_line(
        &self,
        state: StateIn,
        net: Vec<u8>,
        moves: Vec<u32>,
    ) -> PyResult<Vec<(i64, Vec<u32>, i64)>> {
        if self.tables.spy.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "net_line walks turns or rounds; a spy game mixes the two",
            ));
        }
        let parsed = net::Net::parse(&net).map_err(pyo3::exceptions::PyValueError::new_err)?;
        parsed
            .check_against(&self.tables)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let parsed = std::sync::Arc::new(parsed);
        let snapshot = snapshot_from(&self.tables, &state);
        let mut game = Game::from_snapshot(&self.tables, &snapshot);
        let mut evaluator = net::NetEvaluator::new(&self.tables, parsed.clone(), Vec::new());
        let mut out = Vec::with_capacity(moves.len() + 1);
        let mut step = |game: &Game<'_>| {
            let raw = evaluator.raw(game);
            let inputs = evaluator.active().to_vec();
            let fresh = parsed.forward(&parsed.refresh(&inputs), game.side);
            debug_assert_eq!(evaluator.accumulator(), parsed.refresh(&inputs).as_slice());
            (raw, inputs, fresh)
        };
        out.push(step(&game));
        if self.tables.simultaneous.is_some() {
            if !moves.len().is_multiple_of(2) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "a round game's line is its rounds' moves in pairs; this one has an odd count",
                ));
            }
            for pair in moves.chunks_exact(2) {
                game.make_round([pair[0], pair[1]]);
                out.push(step(&game));
            }
            return Ok(out);
        }
        for m in moves {
            game.make(m);
            out.push(step(&game));
        }
        Ok(out)
    }

    /// Q502e: how many inputs this ruleset's net layout has (`net::Layout`).
    fn net_inputs(&self) -> usize {
        net::Layout::of(&self.tables).n_inputs
    }

    /// Score every legal move at a full window: the analysis probe, not a player.
    ///
    /// Returns one value per legal move, in the order `legal_moves` produced them.
    /// See `probe.rs` for why this cannot reuse the agents' search.
    #[pyo3(signature = (state, values, nodes, max_depth, goal = 0.0, immune = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn root_scores(
        &self,
        py: Python<'_>,
        state: StateIn,
        values: Vec<f64>,
        nodes: u64,
        max_depth: u32,
        goal: f64,
        immune: f64,
    ) -> Vec<f64> {
        let snapshot = snapshot_from(&self.tables, &state);
        py.detach(|| {
            let mut game = Game::from_snapshot(&self.tables, &snapshot);
            let legal = game.legal_moves();
            if legal.len() < 2 {
                return vec![0.0; legal.len()];
            }
            // Mobility off: this measures how far apart the *positions* are, and a
            // mobility term would mix the size of the move list back into the answer.
            // The goal weight stays on, because a probe that cannot price the win
            // condition measures the evaluator rather than the game -- which is exactly
            // what `decision_criticality` was doing on every ruleset won by anything
            // other than capture. The immunity term (Q471) is on for the same reason, at
            // whatever weight the analysis pass's evaluator carries.
            let eval =
                Evaluator::new(&self.tables, values, 0.0, goal).with_immunity(&self.tables, immune);
            probe::root_scores(&mut game, &eval, &legal, nodes, max_depth)
        })
    }

    /// Q665 (plan-teacher §8.2, T2v): the playing search's own value of each position.
    ///
    /// One `(value, move, depth)` per state, in order: the root value of the last
    /// iteration of deepening `agent` completed there, for the side to move, in its
    /// objective's units (built-in units for the built-in objective), with its move and
    /// that iteration's depth. `None` where the position has no move or has ended, or
    /// where no iteration finished. `seeds[i]` names state `i`'s stream exactly as
    /// `select_move`'s `seed` does, so the move is `select_move`'s wherever the position
    /// has more than one legal move. A training label's teacher, never a player: nothing
    /// that plays calls it. Minimax only. Spread over `threads`; the answer does not
    /// depend on how many (`parallel::search_values`).
    #[pyo3(signature = (states, agent, values, seeds, threads = 1))]
    fn search_values(
        &self,
        py: Python<'_>,
        states: Vec<StateIn>,
        agent: AgentIn,
        values: Vec<f64>,
        seeds: Vec<u64>,
        threads: usize,
    ) -> PyResult<Vec<Option<(f64, u32, u32)>>> {
        let spec = agent_spec(&self.tables, &agent)?;
        if spec.kind != agents::Kind::Minimax {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "search_values asks a minimax player for its value; {} is not one",
                agent.0
            )));
        }
        if seeds.len() != states.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "search_values needs one seed per state: {} states, {} seeds",
                states.len(),
                seeds.len()
            )));
        }
        let snapshots: Vec<Snapshot> = states
            .iter()
            .map(|state| snapshot_from(&self.tables, state))
            .collect();
        Ok(py.detach(|| {
            parallel::search_values(&self.tables, &snapshots, &spec, &values, &seeds, threads)
        }))
    }

    /// Q918g (`plan-simultaneous.md` §6.4): a round player's own value of each round, for
    /// player 0, the round games' player teacher (`search_values` is minimax's).
    ///
    /// One `(value, depth)` per state, in order: the value, in the agent's objective's
    /// units, of the matrix game the player would draw from there (greedy's one-round
    /// matrix, or `matrix:nodes=N`'s last solve) at the strategies it would draw with, and
    /// the last pass that finished. `None` where the round is over before anyone chooses.
    /// No stream is read. Spread over `threads`; the answer does not depend on how many
    /// (`parallel::round_values`). A training label's teacher, never a player.
    ///
    /// `score_scale` (Q918i, `plan-simultaneous.md` §7, *Q918h, the re-plan*, part 3) asks
    /// for the value on **expected score**: every cell of the one-round matrix is player 0's
    /// `sigmoid(units / score_scale)`, or 1, ½ or 0 where the round ends the game, and the
    /// value (already a score) is that matrix game's. Only the one-round teacher has it
    /// (`greedy`, `matrix:nodes=1`); a deeper one is refused by name.
    #[pyo3(signature = (states, agent, values, threads = 1, score_scale = None))]
    fn round_values(
        &self,
        py: Python<'_>,
        states: Vec<StateIn>,
        agent: AgentIn,
        values: Vec<f64>,
        threads: usize,
        score_scale: Option<f64>,
    ) -> PyResult<Vec<Option<(f64, u32)>>> {
        if self.tables.simultaneous.is_none() || self.tables.spy.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "round_values is for a game played in rounds (rules.simultaneous, no spy)",
            ));
        }
        let spec = agent_spec(&self.tables, &agent)?;
        if !matches!(spec.kind, agents::Kind::Greedy | agents::Kind::Matrix) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "round_values asks a round player that searches (greedy, matrix) for its                  value; {} is not one",
                agent.0
            )));
        }
        if let Some(scale) = score_scale {
            if !(scale > 0.0 && scale.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "round_values' score_scale is a positive number of units",
                ));
            }
            if !agents::one_round_teacher(&spec) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "a score payoff is the one-round teacher's (greedy or matrix:nodes=1); {}                      searches deeper, and its deeper cells are sub-games solved in units",
                    agent.0
                )));
            }
        }
        let snapshots: Vec<Snapshot> = states
            .iter()
            .map(|state| snapshot_from(&self.tables, state))
            .collect();
        Ok(py.detach(|| {
            parallel::round_values(&self.tables, &snapshots, &spec, &values, threads, score_scale)
        }))
    }

    /// Finish a game from a position that is already in progress.
    ///
    /// Same players, same seat-stream discipline, same record shape as `play_game`;
    /// only the starting position differs. The blunder trial is what wants it: force
    /// one arbitrary move into a recorded game and ask whether it is still a game.
    #[pyo3(signature = (state, agents, values, seed, collect_series = false))]
    fn play_from(
        &self,
        py: Python<'_>,
        state: StateIn,
        agents: (AgentIn, AgentIn),
        values: Vec<f64>,
        seed: u64,
        collect_series: bool,
    ) -> PyResult<RecordOut> {
        let specs = [
            agent_spec(&self.tables, &agents.0)?,
            agent_spec(&self.tables, &agents.1)?,
        ];
        let snapshot = snapshot_from(&self.tables, &state);
        let output = py.detach(|| {
            let game = Game::from_snapshot(&self.tables, &snapshot);
            play::play_from(&self.tables, game, &specs, &values, seed, collect_series)
        });
        Ok(record_out(output))
    }

    /// Play a whole schedule across `threads` threads. `tasks` is
    /// `(seed, first agent index, second agent index)` into `agents`.
    #[pyo3(signature = (agents, values, tasks, collect_series = true, threads = 1))]
    fn run_tournament(
        &self,
        py: Python<'_>,
        agents: Vec<AgentIn>,
        values: Vec<f64>,
        tasks: Vec<(u64, usize, usize)>,
        collect_series: bool,
        threads: usize,
    ) -> PyResult<Vec<RecordOut>> {
        let specs = agents
            .iter()
            .map(|agent| agent_spec(&self.tables, agent))
            .collect::<PyResult<Vec<AgentSpec>>>()?;
        let outputs = py.detach(|| {
            parallel::run_tournament(&self.tables, &tasks, &specs, &values, collect_series, threads)
        });
        Ok(outputs.into_iter().map(record_out).collect())
    }

    /// Random playouts for the cheap gate, one independent stream per game.
    #[pyo3(signature = (games, seed = 0, time_budget = 20.0, threads = 1))]
    fn random_playouts(
        &self,
        py: Python<'_>,
        games: usize,
        seed: u64,
        time_budget: f64,
        threads: usize,
    ) -> Vec<(u32, Option<i8>, &'static str, u64, u64)> {
        let results = py.detach(|| {
            parallel::random_playouts(&self.tables, games, seed, time_budget, threads)
        });
        results
            .into_iter()
            .map(|playout| {
                (
                    playout.plies,
                    if playout.winner < 0 {
                        None
                    } else {
                        Some(playout.winner)
                    },
                    playout.reason,
                    playout.branching_total,
                    playout.branching_samples,
                )
            })
            .collect()
    }
}

/// The agent kinds this core can run. Anything else falls back to the Python loop,
/// which is what keeps `agents.register` a real extension point.
#[pyfunction]
fn native_agent_kinds() -> Vec<&'static str> {
    vec!["random", "greedy", "minimax", "mcts", "matrix"]
}

/// Schema 26 (Q918b): the round players' solver on one matrix (row-major, the row
/// player's payoff), for the oracle's test to hold `agents/matrix.py::solve` to it bit for
/// bit. `iterations` defaults to the players' own count.
#[pyfunction]
#[pyo3(signature = (values, rows, cols, iterations = agents::REGRET_ITERATIONS))]
fn regret_solve(
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    iterations: u32,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if rows == 0 || cols == 0 || values.len() != rows * cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "a {rows} by {cols} matrix has {} entries, not {}",
            rows * cols,
            values.len()
        )));
    }
    Ok(agents::solve(&values, rows, cols, iterations))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Core>()?;
    m.add_function(wrap_pyfunction!(native_agent_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(regret_solve, m)?)?;
    // Which sources this extension was built from, hashed by `build.rs`. Python compares
    // it against the same hash taken from `rust/src/` on disk, which is the only way to
    // tell a current build from one that predates an edit to the core.
    m.add("BUILD_FINGERPRINT", env!("CHESS_VARIANT_SANDBOX_BUILD_FINGERPRINT"))?;
    // Every table key `tables.rs` reads, straight from the registry its readers are
    // spelled out of. `engine/bridge.py` compares it against the keys `build_tables`
    // writes, which is the only way to catch a core built before a table was added: a
    // missing key raises on the way in, an unread one is silent.
    m.add("TABLE_KEYS", PyTuple::new(m.py(), crate::tables::keys::ALL)?)?;
    // Whether this build re-derives what it memoises and compares. `game.rs` keeps two
    // memos -- where a royal stands and whether a side is in check -- and each carries a
    // `debug_assert_eq!` against the definition it stands in for, which is the only
    // mechanical check that a cache and its source have not drifted apart. Exposed
    // because the flag is invisible from Python otherwise, and a verification run that
    // silently used the ordinary build proves nothing at all.
    m.add("DEBUG_ASSERTIONS", cfg!(debug_assertions))?;
    // The race term's share of the goal weight. Exported so the Python oracle's copy is
    // held to it by a test rather than by hand: a share out of step on one side is a
    // legal, plausible, *different* player on every ruleset with a priced race.
    m.add("RACE_SHARE", crate::eval::RACE_SHARE)?;
    // The immunity term's default weight, for the same reason.
    m.add("NET_INPUTS_VERSION", net::INPUTS_VERSION)?;
    m.add("IMMUNE_PIECES", crate::eval::IMMUNE_PIECES)?;
    // Q502i: the search repairs a minimax player takes unless it spells them, for the
    // same reason again.
    m.add("DEFAULT_REPAIRS", crate::agents::DEFAULT_REPAIRS)?;
    // Q918b: the round players' solver iterations, for the same reason again.
    m.add("REGRET_ITERATIONS", crate::agents::REGRET_ITERATIONS)?;
    // Q942b: what a rested royal under a standing attacker costs, for the same reason again.
    m.add("RESTED_ROYAL", crate::eval::RESTED_ROYAL)?;
    // Q935h: what holding the only spy is worth, for the same reason again.
    m.add("SPY_PIECES", crate::eval::SPY_PIECES)?;
    m.add("__doc__", "Move generation, search and tournament execution.")?;
    Ok(())
}
