//! Everything that runs more than one game, and the only place threads appear.
//!
//! Parallelism here is safe to take for granted precisely because a game is a pure
//! function of its inputs: the schedule fixes a seed per game before any thread starts,
//! so the corpus a tournament produces is identical whether it ran on one core or
//! sixteen. Nothing is shared but the immutable tables.

use std::time::Instant;

use rayon::prelude::*;

use crate::agents::{AgentSpec, Player};
use crate::game::{Game, Snapshot};
use crate::play::{play_game, random_playout, GameOutput};
use crate::rng::PyRandom;
use crate::tables::Tables;

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .expect("failed to build a thread pool")
}

/// Count leaf nodes at `depth`, splitting the root moves across threads.
///
/// The split must answer what `Game::perft_stop` answers on one thread, root included:
/// under `stop_at_terminal` a root that is already decided has no leaves. Until Q945 the
/// split went straight to the root's moves, so a root decided by an ending some move
/// undoes (an arrival taken off its square, a clock a capture resets) counted its live
/// children on several threads and nothing on one.
pub fn perft(t: &Tables, snapshot: &Snapshot, depth: u32, threads: usize, stop_at_terminal: bool) -> u64 {
    let mut game = Game::from_snapshot(t, snapshot);
    // Schema 26 (Q918a): a joint perft, every pair of legal moves a round, serially.
    if t.simultaneous.is_some() {
        return game.perft_round(depth, stop_at_terminal);
    }
    if depth <= 1 || threads <= 1 {
        let mut arena = Vec::new();
        if stop_at_terminal {
            return game.perft_stop(&mut arena, depth, true);
        }
        return game.perft(&mut arena, depth);
    }
    // The root's half of `perft_stop`; an empty move list sums to 0 below, as it does there.
    if stop_at_terminal && game.terminal_prefix().is_some() {
        return 0;
    }

    let mut roots = Vec::new();
    game.generate_legal(&mut roots);
    pool(threads).install(|| {
        roots
            .par_iter()
            .map(|m| {
                let mut child = Game::from_snapshot(t, snapshot);
                child.make(*m);
                let mut arena = Vec::new();
                if stop_at_terminal {
                    child.perft_stop(&mut arena, depth - 1, true)
                } else {
                    child.perft(&mut arena, depth - 1)
                }
            })
            .sum()
    })
}

/// Play a whole schedule. `tasks` is `(seed, first agent index, second agent index)`,
/// already ordered by the Python scheduler, and the results come back in that order.
pub fn run_tournament(
    t: &Tables,
    tasks: &[(u64, usize, usize)],
    specs: &[AgentSpec],
    values: &[f64],
    collect_series: bool,
    threads: usize,
) -> Vec<GameOutput> {
    let play = |task: &(u64, usize, usize)| {
        let pair = [specs[task.1].clone(), specs[task.2].clone()];
        play_game(t, &pair, values, task.0, collect_series)
    };

    if threads <= 1 {
        return tasks.iter().map(play).collect();
    }
    pool(threads).install(|| tasks.par_iter().map(play).collect())
}

/// Q665 (plan-teacher §8.2, T2v): the player's own root value on each position.
///
/// One fresh `Player` per position, its stream `PyRandom::for_seat(seeds[i], 0)` as
/// `Core.select_move`'s is, so a position's answer is a pure function of the position,
/// the agent and its seed: the caller fixes every seed before a thread starts, nothing
/// is carried from one position to the next, and the results come back in input order.
/// The thread count therefore never changes an answer. `None` where the position has no
/// legal move or has ended, or where not even depth 1 finished.
pub fn search_values(
    t: &Tables,
    snapshots: &[Snapshot],
    spec: &AgentSpec,
    values: &[f64],
    seeds: &[u64],
    threads: usize,
) -> Vec<Option<(f64, u32, u32)>> {
    let one = |index: usize| {
        let mut game = Game::from_snapshot(t, &snapshots[index]);
        let legal = game.legal_moves();
        if legal.is_empty() || game.terminal_with(&legal).is_some() {
            return None;
        }
        let mut player = Player::new(t, spec.clone(), values);
        let mut rng = PyRandom::for_seat(seeds[index], 0);
        player.search_value(&mut game, &legal, &mut rng)
    };
    if threads <= 1 {
        return (0..snapshots.len()).map(one).collect();
    }
    pool(threads).install(|| (0..snapshots.len()).into_par_iter().map(one).collect())
}

pub struct Playout {
    pub plies: u32,
    pub winner: i8,
    pub reason: &'static str,
    pub branching_total: u64,
    pub branching_samples: u64,
}

/// Random playouts for the cheap gate.
///
/// Each game gets its own stream keyed by its index, rather than drawing from one
/// shared generator as the serial Python version did. That is what makes the result
/// independent of how the games were spread across threads -- the same seed produces
/// the same set of games at any worker count.
///
/// `time_budget` is wall-clock, and games that have not started when it expires are
/// abandoned, so fewer than `games` playouts may come back. Nothing here decides what
/// that means: `loop/gates.smoke_test` refuses a ruleset whose playouts the clock cut
/// short and never grades the remainder (Q451), so a slow machine cannot pass or fail
/// a design on a sample a fast one would have graded whole.
pub fn random_playouts(
    t: &Tables,
    games: usize,
    seed: u64,
    time_budget: f64,
    threads: usize,
) -> Vec<Playout> {
    let started = Instant::now();
    let one = |index: usize| -> Option<Playout> {
        if started.elapsed().as_secs_f64() > time_budget {
            return None;
        }
        let mut rng = PyRandom::for_seat(seed, index);
        let (plies, winner, reason, branching_total, branching_samples) =
            random_playout(t, &mut rng);
        Some(Playout {
            plies,
            winner,
            reason,
            branching_total,
            branching_samples,
        })
    };

    if threads <= 1 {
        return (0..games).filter_map(one).collect();
    }
    pool(threads).install(|| (0..games).into_par_iter().filter_map(one).collect())
}
