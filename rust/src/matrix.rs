//! The computer players for a simultaneous game (schema 26, Q918b).
//!
//! `docs/plan-simultaneous.md` §6.2. A round is a **matrix game**: rows are my moves,
//! columns the opponent's, each cell the position after resolving that pair. A good player
//! in such a game mixes, so every player above `random` chooses a probability over its
//! moves and draws one with its own seat's stream -- which keeps a game a pure function of
//! `(ruleset_hash, seed, agent_configs)`: the strategy is computed without randomness and
//! the one draw is the seat's.
//!
//! * `greedy` builds the whole one-round matrix (every pair of legal moves, resolved, scored
//!   by the evaluator from its own seat), solves it and draws. Nodes are cells resolved.
//! * `matrix:nodes=N` deepens. Pass 1 is greedy's matrix; each further pass re-values the
//!   cells by the value of the matrix game one round further on, in order of the previous
//!   pass's joint probability. One node is one resolved round, counted as minimax counts a
//!   `make`. A cell whose re-valuation the budget cuts off keeps its previous value, the
//!   cells re-valued before it keep their new ones, and the depth reported is the last pass
//!   that finished.
//!
//! `chess_variant_sandbox/agents/matrix.py` is the declared oracle: the same solver summed
//! in the same order, the same node counts, the same draw.

use super::*;

/// Iterations of regret matching+ per matrix game. Chosen by the pre-registered test
/// against exact equilibria of small matrices (`plan-simultaneous.md`, *Q918b,
/// pre-registered*), never by a game's score. `agents/matrix.py` holds the same number,
/// and a test holds the two together: a change is a new player under every round name.
pub const REGRET_ITERATIONS: u32 = 100;

/// The average strategies of regret matching+ on `values` (row-major, the row player's
/// payoff; the column player's is its negation).
///
/// Alternating updates, as CFR+ makes them: the row player answers the column player's
/// current strategy, then the column player answers the row player's new one; each regret
/// is clamped at zero, and each player's average weights the strategy it took at iteration
/// `t` by `t`. Every sum is a plain left-to-right loop, the oracle's order, so the two
/// agree to the bit.
pub fn solve(values: &[f64], rows: usize, cols: usize, iterations: u32) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(values.len(), rows * cols);
    let mut regret_r = vec![0.0; rows];
    let mut regret_c = vec![0.0; cols];
    let mut sum_r = vec![0.0; rows];
    let mut sum_c = vec![0.0; cols];
    let mut p = vec![1.0 / rows as f64; rows];
    let mut q = vec![1.0 / cols as f64; cols];
    let mut utility_r = vec![0.0; rows];
    let mut utility_c = vec![0.0; cols];
    for t in 1..=iterations {
        let weight = f64::from(t);
        for i in 0..rows {
            let row = &values[i * cols..(i + 1) * cols];
            let mut total = 0.0;
            for j in 0..cols {
                total += row[j] * q[j];
            }
            utility_r[i] = total;
        }
        let mut expected = 0.0;
        for i in 0..rows {
            expected += p[i] * utility_r[i];
        }
        for i in 0..rows {
            let updated = regret_r[i] + utility_r[i] - expected;
            regret_r[i] = if updated > 0.0 { updated } else { 0.0 };
        }
        normalise_into(&regret_r, &mut p);
        for i in 0..rows {
            sum_r[i] += weight * p[i];
        }
        for j in 0..cols {
            let mut total = 0.0;
            for i in 0..rows {
                total += values[i * cols + j] * p[i];
            }
            utility_c[j] = -total;
        }
        let mut expected = 0.0;
        for j in 0..cols {
            expected += q[j] * utility_c[j];
        }
        for j in 0..cols {
            let updated = regret_c[j] + utility_c[j] - expected;
            regret_c[j] = if updated > 0.0 { updated } else { 0.0 };
        }
        normalise_into(&regret_c, &mut q);
        for j in 0..cols {
            sum_c[j] += weight * q[j];
        }
    }
    let mut row = vec![0.0; rows];
    let mut col = vec![0.0; cols];
    normalise_into(&sum_r, &mut row);
    normalise_into(&sum_c, &mut col);
    (row, col)
}

/// `weights` as a distribution, or uniform when they sum to nothing.
fn normalise_into(weights: &[f64], out: &mut [f64]) {
    let mut total = 0.0;
    for weight in weights {
        total += weight;
    }
    if total > 0.0 {
        for (slot, weight) in out.iter_mut().zip(weights) {
            *slot = weight / total;
        }
    } else {
        let uniform = 1.0 / weights.len() as f64;
        for slot in out.iter_mut() {
            *slot = uniform;
        }
    }
}

/// The row player's expected payoff under `p` and `q`, rows outer.
pub fn game_value(values: &[f64], p: &[f64], q: &[f64]) -> f64 {
    let cols = q.len();
    let mut total = 0.0;
    for (i, row_weight) in p.iter().enumerate() {
        for (j, col_weight) in q.iter().enumerate() {
            total += row_weight * col_weight * values[i * cols + j];
        }
    }
    total
}

/// One move from `strategy`: one uniform draw over the cumulative sums in generation
/// order, the last move if rounding leaves the draw past the total.
fn draw(moves: &[u32], strategy: &[f64], rng: &mut PyRandom) -> u32 {
    let point = rng.random();
    let mut running = 0.0;
    for (m, weight) in moves.iter().zip(strategy) {
        running += weight;
        if point < running {
            return *m;
        }
    }
    *moves.last().expect("a player with no move is not asked")
}

/// A finished round's value for `seat`: a win sooner is worth more, as minimax's is.
fn round_terminal_value(outcome: Outcome, game: &Game<'_>, seat: usize) -> f64 {
    if outcome.winner < 0 {
        return 0.0;
    }
    let signed = WIN_SCORE - game.ply as f64;
    if outcome.winner as usize == seat {
        signed
    } else {
        -signed
    }
}

impl Player {
    /// One round's choice for `seat`: `lists` are both players' legal moves on this board,
    /// in seat order, and the opponent's choice is not known. The position is the same
    /// on return.
    pub fn select_round(
        &mut self,
        game: &mut Game<'_>,
        seat: usize,
        lists: [&[u32]; 2],
        rng: &mut PyRandom,
    ) -> u32 {
        let (mine, theirs) = (lists[seat], lists[1 - seat]);
        match self.spec.kind {
            Kind::Random => {
                self.stats.moves += 1;
                self.stats.nodes += mine.len() as u64;
                self.strategy.clear();
                rng.choice(mine)
            }
            Kind::Greedy => self.greedy_round(game, seat, mine, theirs, rng),
            Kind::Matrix => self.matrix_round(game, seat, mine, theirs, rng),
            Kind::Minimax | Kind::Mcts => {
                unreachable!("check_against refuses minimax and mcts on a simultaneous game")
            }
        }
    }

    /// Schema 27 (Q935a, `docs/plan-spy.md` §3.2): the move an announcer shows first in an
    /// announced round of a spy game. Only `random` plays one yet (`check_against`): uniform
    /// over its own moves, from its own stream.
    pub fn select_announcement(
        &mut self,
        _game: &mut Game<'_>,
        seat: usize,
        lists: [&[u32]; 2],
        rng: &mut PyRandom,
    ) -> u32 {
        match self.spec.kind {
            Kind::Random => {
                let mine = lists[seat];
                self.stats.moves += 1;
                self.stats.nodes += mine.len() as u64;
                self.strategy.clear();
                rng.choice(mine)
            }
            _ => unreachable!("check_against refuses every player but random on a spy game"),
        }
    }

    /// Schema 27 (Q935a, §3.2): the answerer's move, chosen knowing `announced`. `random`
    /// ignores it.
    pub fn select_answer(
        &mut self,
        _game: &mut Game<'_>,
        seat: usize,
        lists: [&[u32]; 2],
        _announced: u32,
        rng: &mut PyRandom,
    ) -> u32 {
        match self.spec.kind {
            Kind::Random => {
                let mine = lists[seat];
                self.stats.moves += 1;
                self.stats.nodes += mine.len() as u64;
                self.strategy.clear();
                rng.choice(mine)
            }
            _ => unreachable!("check_against refuses every player but random on a spy game"),
        }
    }

    /// Greedy's round arm: the one-round matrix, solved, drawn from.
    fn greedy_round(
        &mut self,
        game: &mut Game<'_>,
        seat: usize,
        mine: &[u32],
        theirs: &[u32],
        rng: &mut PyRandom,
    ) -> u32 {
        self.stats.moves += 1;
        if mine.len() == 1 {
            self.strategy = vec![1.0];
            return mine[0];
        }
        self.nodes = 0;
        let mut arena = std::mem::take(&mut self.arena);
        arena.clear();
        let values = match self.round_cells(game, seat, mine, theirs, 1, &mut arena, false) {
            Ok(values) => values,
            Err(Exhausted) => unreachable!("an unbounded pass never runs out"),
        };
        self.arena = arena;
        self.stats.nodes += self.nodes;
        let (p, _) = solve(&values, mine.len(), theirs.len(), REGRET_ITERATIONS);
        let chosen = draw(mine, &p, rng);
        self.strategy = p;
        chosen
    }

    /// `matrix:nodes=N`: iterative deepening on matrix games (§6.2).
    fn matrix_round(
        &mut self,
        game: &mut Game<'_>,
        seat: usize,
        mine: &[u32],
        theirs: &[u32],
        rng: &mut PyRandom,
    ) -> u32 {
        self.stats.depth = 0;
        self.stats.moves += 1;
        if mine.len() == 1 {
            self.strategy = vec![1.0];
            return mine[0];
        }
        self.nodes = 0;
        let mut arena = std::mem::take(&mut self.arena);
        arena.clear();
        let (rows, cols) = (mine.len(), theirs.len());
        let mut values = match self.round_cells(game, seat, mine, theirs, 1, &mut arena, false) {
            Ok(values) => values,
            Err(Exhausted) => unreachable!("an unbounded pass never runs out"),
        };
        let (mut p, mut q) = solve(&values, rows, cols, REGRET_ITERATIONS);
        let mut done = 1u32;
        let budget = self.spec.nodes;
        let mut order: Vec<usize> = Vec::with_capacity(rows * cols);
        let mut joint: Vec<f64> = Vec::with_capacity(rows * cols);
        while self.nodes < budget && done < self.spec.max_depth {
            let depth = done + 1;
            joint.clear();
            joint.extend((0..rows * cols).map(|k| p[k / cols] * q[k % cols]));
            order.clear();
            order.extend(0..rows * cols);
            // Stable, descending: equal probabilities keep generation order, as the
            // oracle's `sorted(..., reverse=True)` keeps them.
            order.sort_by(|&x, &y| {
                joint[y].partial_cmp(&joint[x]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut complete = true;
            for &k in &order {
                if self.nodes >= budget {
                    complete = false;
                    break;
                }
                match self.round_cell(game, seat, mine[k / cols], theirs[k % cols], depth, &mut arena, true) {
                    Ok(value) => values[k] = value,
                    Err(Exhausted) => {
                        complete = false;
                        break;
                    }
                }
            }
            (p, q) = solve(&values, rows, cols, REGRET_ITERATIONS);
            if !complete {
                break;
            }
            done = depth;
        }
        self.arena = arena;
        self.stats.depth = done;
        self.stats.nodes += self.nodes;
        self.stats.depth_sum += u64::from(done);
        self.stats.searches += 1;
        let chosen = draw(mine, &p, rng);
        self.strategy = p;
        chosen
    }

    /// Every cell's value for `seat` at `depth` rounds, rows (my moves) outer.
    #[allow(clippy::too_many_arguments)]
    fn round_cells(
        &mut self,
        game: &mut Game<'_>,
        seat: usize,
        mine: &[u32],
        theirs: &[u32],
        depth: u32,
        arena: &mut Vec<u32>,
        bounded: bool,
    ) -> Result<Vec<f64>, Exhausted> {
        let mut values = Vec::with_capacity(mine.len() * theirs.len());
        for &a in mine {
            for &b in theirs {
                values.push(self.round_cell(game, seat, a, b, depth, arena, bounded)?);
            }
        }
        Ok(values)
    }

    /// One cell: resolve my `a` against their `b` (one node), value the round after it.
    #[allow(clippy::too_many_arguments)]
    fn round_cell(
        &mut self,
        game: &mut Game<'_>,
        seat: usize,
        a: u32,
        b: u32,
        depth: u32,
        arena: &mut Vec<u32>,
        bounded: bool,
    ) -> Result<f64, Exhausted> {
        self.nodes += 1;
        if bounded && self.nodes >= self.spec.nodes {
            return Err(Exhausted);
        }
        game.make_round(if seat == 0 { [a, b] } else { [b, a] });
        let value = self.round_value(game, seat, depth - 1, arena, bounded);
        game.unmake_round();
        value
    }

    /// The value of this position for `seat`, `depth` rounds deep: the verdict, the
    /// evaluator at depth 0, else the solved matrix game's value.
    fn round_value(
        &mut self,
        game: &mut Game<'_>,
        seat: usize,
        depth: u32,
        arena: &mut Vec<u32>,
        bounded: bool,
    ) -> Result<f64, Exhausted> {
        let base = arena.len();
        game.generate_legal_for(0, arena);
        let mid = arena.len();
        game.generate_legal_for(1, arena);
        let end = arena.len();
        if let Some(outcome) = game.round_terminal_with([&arena[base..mid], &arena[mid..end]]) {
            arena.truncate(base);
            return Ok(round_terminal_value(outcome, game, seat));
        }
        if depth == 0 {
            arena.truncate(base);
            return Ok(self.eval.evaluate_round(game, seat, [mid - base, end - mid]));
        }
        let lists = [arena[base..mid].to_vec(), arena[mid..end].to_vec()];
        arena.truncate(base);
        let (mine, theirs) = (&lists[seat], &lists[1 - seat]);
        let values = self.round_cells(game, seat, mine, theirs, depth, arena, bounded)?;
        let (p, q) = solve(&values, mine.len(), theirs.len(), REGRET_ITERATIONS);
        Ok(game_value(&values, &p, &q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_pennies_is_half_and_half() {
        let (p, q) = solve(&[1.0, -1.0, -1.0, 1.0], 2, 2, REGRET_ITERATIONS);
        for value in p.iter().chain(&q) {
            assert!((value - 0.5).abs() <= 0.01, "{p:?} {q:?}");
        }
    }

    #[test]
    fn a_dominant_row_takes_almost_everything() {
        let (p, _) = solve(&[2.0, 3.0, 0.0, 1.0], 2, 2, REGRET_ITERATIONS);
        assert!(p[0] > 0.999, "{p:?}");
    }

    #[test]
    fn the_draw_follows_the_cumulative_sums() {
        let mut rng = PyRandom::new(1);
        let moves = [10u32, 20, 30];
        assert_eq!(draw(&moves, &[0.0, 1.0, 0.0], &mut rng), 20);
        assert_eq!(draw(&moves, &[0.0, 0.0, 0.0], &mut rng), 30);
    }
}
