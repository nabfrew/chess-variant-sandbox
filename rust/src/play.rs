//! Playing one game to a verdict, reproducibly.
//!
//! The reproducibility contract lives here, as it does in `tournament/play.py`: a game
//! is a pure function of `(ruleset, seed, agent configs)`. Each seat gets its own
//! stream derived from the seed and the seat, so how much randomness one agent happens
//! to consume cannot shift what the other draws.

use std::time::Instant;

use crate::agents::{AgentSpec, Player};
use crate::eval::Evaluator;
use crate::game::Game;
use crate::rng::PyRandom;
use crate::tables::Tables;

pub struct GameOutput {
    pub moves: Vec<u32>,
    /// `-1` for a draw, matching `Outcome.winner is None`.
    pub winner: i8,
    pub reason: &'static str,
    pub branching: Vec<u32>,
    pub material: Vec<f64>,
    pub captures: Vec<u8>,
    pub nodes: [u64; 2],
    /// Plies completed, summed over every search each seat ran, and how many searches
    /// that was. Two numbers rather than a mean because a mean over a game cannot be
    /// pooled over a corpus without them.
    pub depth_sum: [u64; 2],
    pub searches: [u64; 2],
    pub duration_ms: f64,
}

/// Play one game and return everything a `GameRecord` needs.
///
/// `collect_series` controls whether per-ply branching and material are recorded. The
/// smoke gate turns it off: it only needs to know whether a ruleset produces finite,
/// legal games, and the series would be pure overhead across hundreds of throwaway
/// games.
pub fn play_game(
    t: &Tables,
    specs: &[AgentSpec; 2],
    values: &[f64],
    seed: u64,
    collect_series: bool,
) -> GameOutput {
    play_from(t, Game::initial(t), specs, values, seed, collect_series)
}

/// Finish a game that is already in progress.
///
/// The blunder trial in `tournament/analysis.py` is the reason this exists: force one
/// random move into a recorded game and play it out with the same players, to ask
/// whether the game is still a game afterwards. Fresh per-seat streams, as the Python
/// pass uses -- the resumed game is a new game with its own randomness, not a
/// continuation of the original's stream, and the two must agree on that or the trials
/// stop reproducing.
///
/// `moves` in the output covers only what was played from `game` onward. The ply
/// counter continues from the position handed in, which is what the terminal rules
/// (`max_plies`, `no_progress_draw`) have to see.
pub fn play_from(
    t: &Tables,
    game: Game<'_>,
    specs: &[AgentSpec; 2],
    values: &[f64],
    seed: u64,
    collect_series: bool,
) -> GameOutput {
    let mut players = [
        Player::new(t, specs[0].clone(), values),
        Player::new(t, specs[1].clone(), values),
    ];
    let mut rngs = [PyRandom::for_seat(seed, 0), PyRandom::for_seat(seed, 1)];
    let series_eval = if collect_series {
        Some(Evaluator::new(t, values.to_vec(), 0.0, 0.0))
    } else {
        None
    };

    let mut game = game;
    let mut moves = Vec::new();
    let mut branching = Vec::new();
    let mut material = Vec::new();
    let mut captures = Vec::new();
    let mut legal = Vec::new();

    let started = Instant::now();
    if t.spy.is_some() {
        // Schema 27 (Q935a, `docs/plan-spy.md` §2.5): a spy game, each step played by its
        // phase. A turn is one move in the record and one entry in every series; a round is
        // two, in seat order, as in a round game. A replay recomputes the phase from the
        // position, so the record needs no field for it.
        let mut second = Vec::new();
        let outcome = loop {
            let phase = game.phase();
            if !phase.is_round() {
                legal.clear();
                game.generate_legal(&mut legal);
                if let Some(outcome) = game.terminal_with(&legal) {
                    break outcome;
                }
                if let Some(evaluator) = &series_eval {
                    branching.push(legal.len() as u32);
                    material.push(evaluator.material(&game, 0));
                }
                let mover = game.side;
                let m = players[mover].select_move(&mut game, &legal, &mut rngs[mover]);
                game.spy_turn(m);
                if collect_series {
                    captures.push(u8::from(game.halfmove == 0));
                }
                moves.push(m);
                continue;
            }
            legal.clear();
            second.clear();
            game.generate_legal_for(0, &mut legal);
            game.generate_legal_for(1, &mut second);
            if let Some(outcome) = game.round_terminal_with([&legal, &second]) {
                break outcome;
            }
            if let Some(evaluator) = &series_eval {
                branching.push(legal.len() as u32);
                branching.push(second.len() as u32);
                let value = evaluator.material(&game, 0);
                material.push(value);
                material.push(value);
            }
            let chosen = match phase {
                crate::game::Phase::Announced { announcer } => {
                    // The announcer chooses first; its move is shown to the answerer, who
                    // chooses knowing it (§1.4). Each from its own stream.
                    let answerer = 1 - announcer;
                    let lists = [&legal[..], &second[..]];
                    let announced = players[announcer].select_announcement(
                        &mut game, announcer, lists, &mut rngs[announcer],
                    );
                    let answer = players[answerer].select_answer(
                        &mut game, answerer, lists, announced, &mut rngs[answerer],
                    );
                    let mut pair = [0u32; 2];
                    pair[announcer] = announced;
                    pair[answerer] = answer;
                    pair
                }
                _ => {
                    let first = players[0].select_round(&mut game, 0, [&legal, &second], &mut rngs[0]);
                    let other = players[1].select_round(&mut game, 1, [&legal, &second], &mut rngs[1]);
                    [first, other]
                }
            };
            game.make_round(chosen);
            if collect_series {
                let quiet = u8::from(game.halfmove == 0);
                captures.push(quiet);
                captures.push(quiet);
            }
            moves.push(chosen[0]);
            moves.push(chosen[1]);
        };
        return finish(outcome, moves, branching, material, captures, &players, started);
    }
    if t.simultaneous.is_some() {
        // Schema 26 (Q918a): rounds. Both seats are asked on the same board, each from its
        // own stream, and neither sees the other's choice; the record keeps both moves of a
        // round in seat order, and every per-ply series twice, so a round is two plies.
        let mut second = Vec::new();
        let outcome = loop {
            legal.clear();
            second.clear();
            game.generate_legal_for(0, &mut legal);
            game.generate_legal_for(1, &mut second);
            if let Some(outcome) = game.round_terminal_with([&legal, &second]) {
                break outcome;
            }
            if let Some(evaluator) = &series_eval {
                branching.push(legal.len() as u32);
                branching.push(second.len() as u32);
                let value = evaluator.material(&game, 0);
                material.push(value);
                material.push(value);
            }
            // Each seat sees both lists and the board, never the other's choice (Q918b).
            let first = players[0].select_round(&mut game, 0, [&legal, &second], &mut rngs[0]);
            let other = players[1].select_round(&mut game, 1, [&legal, &second], &mut rngs[1]);
            game.make_round([first, other]);
            if collect_series {
                let quiet = u8::from(game.halfmove == 0);
                captures.push(quiet);
                captures.push(quiet);
            }
            moves.push(first);
            moves.push(other);
        };
        return finish(outcome, moves, branching, material, captures, &players, started);
    }
    let outcome = loop {
        legal.clear();
        game.generate_legal(&mut legal);
        if let Some(outcome) = game.terminal_with(&legal) {
            break outcome;
        }
        if let Some(evaluator) = &series_eval {
            branching.push(legal.len() as u32);
            material.push(evaluator.material(&game, 0));
        }

        let mover = game.side;
        let m = players[mover].select_move(&mut game, &legal, &mut rngs[mover]);
        game.make(m);
        if collect_series {
            captures.push(u8::from(game.halfmove == 0));
        }
        moves.push(m);
    };
    finish(outcome, moves, branching, material, captures, &players, started)
}

fn finish(
    outcome: crate::game::Outcome,
    moves: Vec<u32>,
    branching: Vec<u32>,
    material: Vec<f64>,
    captures: Vec<u8>,
    players: &[Player; 2],
    started: Instant,
) -> GameOutput {
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;

    GameOutput {
        winner: outcome.winner,
        reason: outcome.reason.as_str(),
        moves,
        branching,
        material,
        captures,
        nodes: [players[0].stats.nodes, players[1].stats.nodes],
        depth_sum: [players[0].stats.depth_sum, players[1].stats.depth_sum],
        searches: [players[0].stats.searches, players[1].stats.searches],
        duration_ms,
    }
}

/// One random playout, as the cheap gate plays them: no agents, no series, one stream.
pub fn random_playout(t: &Tables, rng: &mut PyRandom) -> (u32, i8, &'static str, u64, u64) {
    let mut game = Game::initial(t);
    let mut legal = Vec::new();
    let mut branching_total = 0u64;
    let mut branching_samples = 0u64;

    if t.spy.is_some() {
        // Schema 27 (Q935a): one stream, a turn's one draw, an announced round's
        // announcer before its answerer, a blind round's player 0 before player 1.
        let mut second = Vec::new();
        let outcome = loop {
            let phase = game.phase();
            if !phase.is_round() {
                legal.clear();
                game.generate_legal(&mut legal);
                if let Some(outcome) = game.terminal_with(&legal) {
                    break outcome;
                }
                branching_total += legal.len() as u64;
                branching_samples += 1;
                let m = rng.choice(&legal);
                game.spy_turn(m);
                continue;
            }
            legal.clear();
            second.clear();
            game.generate_legal_for(0, &mut legal);
            game.generate_legal_for(1, &mut second);
            if let Some(outcome) = game.round_terminal_with([&legal, &second]) {
                break outcome;
            }
            branching_total += (legal.len() + second.len()) as u64;
            branching_samples += 2;
            let pair = if let crate::game::Phase::Announced { announcer } = phase {
                let lists = [&legal, &second];
                let announced = rng.choice(lists[announcer]);
                let answer = rng.choice(lists[1 - announcer]);
                let mut pair = [0u32; 2];
                pair[announcer] = announced;
                pair[1 - announcer] = answer;
                pair
            } else {
                let first = rng.choice(&legal);
                let other = rng.choice(&second);
                [first, other]
            };
            game.make_round(pair);
        };
        return (game.ply, outcome.winner, outcome.reason.as_str(), branching_total, branching_samples);
    }
    if t.simultaneous.is_some() {
        // Schema 26: one stream, player 0's draw before player 1's each round.
        let mut second = Vec::new();
        let outcome = loop {
            legal.clear();
            second.clear();
            game.generate_legal_for(0, &mut legal);
            game.generate_legal_for(1, &mut second);
            if let Some(outcome) = game.round_terminal_with([&legal, &second]) {
                break outcome;
            }
            branching_total += (legal.len() + second.len()) as u64;
            branching_samples += 2;
            let first = rng.choice(&legal);
            let other = rng.choice(&second);
            game.make_round([first, other]);
        };
        return (game.ply, outcome.winner, outcome.reason.as_str(), branching_total, branching_samples);
    }

    let outcome = loop {
        legal.clear();
        game.generate_legal(&mut legal);
        if let Some(outcome) = game.terminal_with(&legal) {
            break outcome;
        }
        branching_total += legal.len() as u64;
        branching_samples += 1;
        let m = rng.choice(&legal);
        game.make(m);
    };

    (
        game.ply,
        outcome.winner,
        outcome.reason.as_str(),
        branching_total,
        branching_samples,
    )
}
