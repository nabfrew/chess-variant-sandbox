//! The analysis probe, ported from `tournament/analysis.py`.
//!
//! This is not a player. It asks one question about a position -- *how far apart were
//! the options here?* -- and the answer feeds `decision_criticality`, which is the
//! thing `mean_branching` was standing in for: a game with forty legal moves of which
//! thirty-eight are equivalent is a narrow game with a wide move list.
//!
//! Two properties are deliberate and both are load-bearing.
//!
//! **A full window, not an alpha-beta one.** A playing search evaluates root moves
//! inside a window, so everything except the best move comes back as a bound rather
//! than a value: fine for choosing a move, useless for measuring how close the
//! alternatives were. Every root move is searched to a value here, which is why this
//! could not simply reuse `agents::Player`.
//!
//! **It reproduces the Python probe exactly, including its ragged edge.** When the node
//! budget runs out part-way through a depth iteration, the score vector is left holding
//! the new depth's values for the moves already visited and the previous depth's for
//! the rest. That mixed vector is what the committed `decision_criticality` numbers
//! were measured from, so it is reproduced rather than tidied: a cleaner rule here
//! would silently re-scale a published component.

use crate::eval::{Evaluator, WIN_SCORE};
use crate::game::Game;

/// Raised in Python, returned here. Same job: abandon the iteration mid-flight.
struct Exhausted;

struct Counter {
    used: u64,
    budget: u64,
}

/// Evaluate every legal move at a full window, under a node budget.
///
/// `legal` must be the move list the caller already generated, in generation order, so
/// that a score can be read back against the move it belongs to.
pub fn root_scores(
    game: &mut Game<'_>,
    eval: &Evaluator,
    legal: &[u32],
    nodes: u64,
    max_depth: u32,
) -> Vec<f64> {
    // Budget is nodes rather than depth, matching the rule that governs agents: a fixed
    // depth is wildly different amounts of work in a game with eight legal moves and
    // one with sixty, and this number has to be comparable across rulesets.
    let budget = (legal.len() as u64 * 2).max(nodes);
    let mut counter = Counter { used: 0, budget };
    let mut scores = vec![0.0f64; legal.len()];
    // One buffer for the whole probe, in the shape `agents::negamax` already threads:
    // each frame generates onto the end of it and truncates back to where it started.
    // This allocated a `Vec` at every node, and the analysis pass is the dominant cost
    // of scoring a ruleset.
    let mut arena: Vec<u32> = Vec::new();

    for depth in 1..=max_depth as i32 {
        let mut exhausted = false;
        for (index, m) in legal.iter().enumerate() {
            let mover = game.side;
            game.make(*m);
            let flipped = game.side != mover;
            let value = probe(game, eval, depth - 1, &mut counter, &mut arena);
            game.unmake();
            match value {
                Ok(value) => scores[index] = if flipped { -value } else { value },
                Err(Exhausted) => {
                    exhausted = true;
                    break;
                }
            }
        }
        if exhausted || counter.used >= budget {
            break;
        }
    }
    scores
}

// Negamax assumes the side alternates on every ply, and negates the child's value and
// flips the window to say so. That assumption is false the moment a turn is longer than
// one move: when the same player moves again the child is already from this player's
// point of view, so the value passes through and the window passes through with it.
//
// Written as "did the side actually change" rather than "is there a turn rule", because
// it is the only form that cannot drift out of step with `make` -- a ramp means the
// answer differs between one ply and the next of the same game.
fn probe(
    game: &mut Game<'_>,
    eval: &Evaluator,
    depth: i32,
    counter: &mut Counter,
    arena: &mut Vec<u32>,
) -> Result<f64, Exhausted> {
    counter.used += 1;
    if counter.used >= counter.budget {
        return Err(Exhausted);
    }

    let base = arena.len();
    game.generate_legal(arena);
    let end = arena.len();
    if let Some(outcome) = game.terminal_with(&arena[base..end]) {
        arena.truncate(base);
        if outcome.winner < 0 {
            return Ok(0.0);
        }
        // Sooner is better: a mate in three beats a mate in nine, and subtracting the
        // ply is what makes a search prefer the shorter one.
        let signed = WIN_SCORE - game.ply as f64;
        return Ok(if outcome.winner as usize == game.side {
            signed
        } else {
            -signed
        });
    }
    if depth <= 0 {
        let move_count = end - base;
        arena.truncate(base);
        return Ok(eval.evaluate(game, game.side, move_count));
    }

    let mut best = f64::NEG_INFINITY;
    for index in base..end {
        let m = arena[index];
        let mover = game.side;
        game.make(m);
        let flipped = game.side != mover;
        let value = probe(game, eval, depth - 1, counter, arena);
        game.unmake();
        let raw = match value {
            Ok(raw) => raw,
            Err(exhausted) => {
                arena.truncate(base);
                return Err(exhausted);
            }
        };
        let candidate = if flipped { -raw } else { raw };
        // Strictly greater, so an equal score leaves the earlier move in place. Python's
        // `max(best, x)` returns its first argument on a tie and this has to agree.
        if candidate > best {
            best = candidate;
        }
    }
    arena.truncate(base);
    Ok(best)
}
