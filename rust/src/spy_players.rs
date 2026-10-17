//! The searching players across a spy game's phases (schema 27, Q935b).
//!
//! `docs/plan-spy.md` §3. A spy game alternates while both spies stand, plays **announced
//! rounds** once one falls (the player without a spy chooses first and in the open, the
//! other answers knowing it) and **blind rounds** once both are gone. An announced round is
//! a perfect-information exchange, so it is searched as two plies of the ordinary tree: the
//! announcer's ply writes nothing to the board, the answerer's resolves the round. Neither
//! choice mixes (§3.2: a mix is shown before the answer, so it is worth no more than its
//! best move), and neither draws from the seat's stream; ties go by the order the moves are
//! tried in.
//!
//! * `greedy` is one round deep in every phase: the announcer takes the one-round maximin
//!   over the whole matrix (a node a cell), the answerer the best cell of the announced row
//!   (a node a cell), a turn's child that is a round is valued as a round.
//! * `minimax:nodes=N` is one budget for every decision. Its alternating search (`negamax`,
//!   instantiated for a spy game) meets **round nodes** inside the tree: an announced one is
//!   searched by `round_node`, two plies, one node a `make_round` and one step of depth a
//!   round; a blind one is a leaf, valued by `evaluate_round` (a search through blind rounds
//!   is a matrix search, which this tree cannot hold); quiescence stops at a round. The
//!   announcer's root is the maximin by iterative deepening, the answerer's the announced
//!   row alone, and a blind root is `matrix:nodes=N`'s search at the same budget.
//!
//! The values are from the side to move's point of view, as every value in `negamax` is.
//! In a round `side` is the spy holder, which in an announced round is the answerer.
//!
//! `chess_variant_sandbox/agents/minimax.py` and `agents/simple.py` are the declared oracles:
//! the same tree in the same order, the same node counts.

use super::*;
use crate::game::Phase;

impl Player {
    /// A round position's value as a leaf, for the side to move: its verdict from both
    /// lists, else `evaluate_round`. The node is already counted.
    pub(super) fn round_leaf(
        &mut self,
        game: &mut Game<'_>,
        arena: &mut Vec<u32>,
    ) -> Result<f64, Exhausted> {
        if let Some(outcome) = game.round_terminal_prefix() {
            return Ok(terminal_value(outcome, game));
        }
        Ok(self.round_leaf_lists(game, arena))
    }

    /// `round_leaf` past its list-free half.
    fn round_leaf_lists(&mut self, game: &mut Game<'_>, arena: &mut Vec<u32>) -> f64 {
        let base = arena.len();
        game.generate_legal_for(0, arena);
        let mid = arena.len();
        game.generate_legal_for(1, arena);
        let end = arena.len();
        let lists = [&arena[base..mid], &arena[mid..end]];
        let value = match game.round_terminal_with(lists) {
            Some(outcome) => terminal_value(outcome, game),
            None => self.eval.evaluate_round(game, game.side, lists),
        };
        arena.truncate(base);
        value
    }

    /// A round inside the alternating search's tree (`plan-spy.md` §3.3), for the side to
    /// move, in the window `(alpha, beta)`. The node is already counted.
    ///
    /// A blind round is a leaf. An announced one at depth is the announcer's ply, a max over
    /// its moves from its own seat (negated, as a child is), over the answerer's ply, a max
    /// over the side to move's moves with the announced one fixed (`answer_row`). Stored in
    /// the table under the round's board, as any node is.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn round_node(
        &mut self,
        game: &mut Game<'_>,
        arena: &mut Vec<u32>,
        phase: Phase,
        depth: i32,
        ply: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<f64, Exhausted> {
        if let Some(outcome) = game.round_terminal_prefix() {
            return Ok(terminal_value(outcome, game));
        }
        let Phase::Announced { announcer } = phase else {
            return Ok(self.round_leaf_lists(game, arena));
        };
        debug_assert_eq!(game.side, 1 - announcer, "an announced round's side is its answerer");

        let key = self.table_key(game);
        if let Some((stored_depth, value, bound, _)) = self.table.get(&key) {
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
        game.generate_legal_for(0, arena);
        let mid = arena.len();
        game.generate_legal_for(1, arena);
        let end = arena.len();
        if let Some(outcome) = game.round_terminal_with([&arena[base..mid], &arena[mid..end]]) {
            arena.truncate(base);
            return Ok(terminal_value(outcome, game));
        }
        if depth <= 0 {
            // No quiescence below a round: a capture in a round can be dodged.
            let value =
                self.eval.evaluate_round(game, game.side, [&arena[base..mid], &arena[mid..end]]);
            arena.truncate(base);
            return Ok(value);
        }
        let (announced, answers) = if announcer == 0 {
            ((base, mid), (mid, end))
        } else {
            ((mid, end), (base, mid))
        };
        self.order(game, arena, announced.0, announced.1, ply);

        // The announcer's ply, from its seat: the window negated.
        let opened_at = alpha;
        let announcer_beta = -alpha;
        let mut announcer_alpha = -beta;
        let mut announcer_best = f64::NEG_INFINITY;
        for index in announced.0..announced.1 {
            let a = arena[index];
            let quiet = !self.is_loud(game, a);
            let reply = self.answer_row(
                game,
                arena,
                announcer,
                a,
                answers,
                depth,
                ply + 1,
                -announcer_beta,
                -announcer_alpha,
            );
            let score = match reply {
                Err(Exhausted) => {
                    arena.truncate(base);
                    return Err(Exhausted);
                }
                Ok(value) => -value,
            };
            if score > announcer_best {
                announcer_best = score;
            }
            if announcer_best > announcer_alpha {
                announcer_alpha = announcer_best;
            }
            if announcer_alpha >= announcer_beta {
                self.remember(a, quiet, depth, ply);
                break;
            }
        }
        arena.truncate(base);

        let best = -announcer_best;
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

    /// The answerer's ply of an announced round: the best of the side to move's moves
    /// (`arena[answers.0..answers.1]`, copied and ordered at `ply`) with the announcer's `a`
    /// fixed, each child one `make_round` searched `depth - 1` rounds on. The side does not
    /// change across a round, so a child's value passes through.
    #[allow(clippy::too_many_arguments)]
    fn answer_row(
        &mut self,
        game: &mut Game<'_>,
        arena: &mut Vec<u32>,
        announcer: usize,
        a: u32,
        answers: (usize, usize),
        depth: i32,
        ply: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<f64, Exhausted> {
        let base = arena.len();
        arena.extend_from_within(answers.0..answers.1);
        let end = arena.len();
        self.order(game, arena, base, end, ply);
        let mut best = f64::NEG_INFINITY;
        let mut alpha = alpha;
        for index in base..end {
            let b = arena[index];
            let quiet = !self.is_loud(game, b);
            game.make_round(seat_pair(announcer, a, b));
            let child = self.negamax::<true>(game, arena, depth - 1, ply + 1, alpha, beta);
            game.unmake_round();
            let score = match child {
                Err(Exhausted) => {
                    arena.truncate(base);
                    return Err(Exhausted);
                }
                Ok(value) => value,
            };
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                self.remember(b, quiet, depth, ply);
                break;
            }
        }
        arena.truncate(base);
        Ok(best)
    }

    /// `minimax:nodes=N` as the announcer (§3.2): the move whose best answer is worst for
    /// the answerer, by iterative deepening from the moves in generation order, the last
    /// finished iteration's best first. No draw from the stream.
    pub(super) fn announce_minimax(
        &mut self,
        game: &mut Game<'_>,
        announcer: usize,
        lists: [&[u32]; 2],
    ) -> u32 {
        self.stats.depth = 0;
        let mine = lists[announcer];
        if mine.len() == 1 {
            return mine[0];
        }
        self.begin_search();
        let mut ordered = mine.to_vec();
        let answers = lists[1 - announcer];
        let mut best_move = ordered[0];
        for depth in 1..=self.spec.max_depth as i32 {
            match self.announce_root(game, announcer, &ordered, answers, depth) {
                Err(Exhausted) => break,
                Ok(candidate) => {
                    best_move = candidate;
                    self.stats.depth = depth as u32;
                    promote(&mut ordered, best_move);
                    if self.nodes >= self.spec.nodes {
                        break;
                    }
                }
            }
        }
        self.end_search();
        best_move
    }

    fn announce_root(
        &mut self,
        game: &mut Game<'_>,
        announcer: usize,
        moves: &[u32],
        answers: &[u32],
        depth: i32,
    ) -> Result<u32, Exhausted> {
        let mut arena = std::mem::take(&mut self.arena);
        arena.clear();
        arena.extend_from_slice(answers);
        let row = (0, arena.len());
        let mut alpha = f64::NEG_INFINITY;
        let mut best_move = moves[0];
        let mut outcome = Ok(());
        for &a in moves {
            match self.answer_row(game, &mut arena, announcer, a, row, depth, 1, f64::NEG_INFINITY, -alpha) {
                Err(Exhausted) => {
                    outcome = Err(Exhausted);
                    break;
                }
                Ok(value) => {
                    if -value > alpha {
                        alpha = -value;
                        best_move = a;
                    }
                }
            }
        }
        arena.clear();
        self.arena = arena;
        outcome.map(|()| best_move)
    }

    /// `minimax:nodes=N` as the answerer (§3.2): the best reply to `announced`, the announced
    /// row searched alone by iterative deepening, from the moves in generation order. No
    /// draw from the stream.
    pub(super) fn answer_minimax(
        &mut self,
        game: &mut Game<'_>,
        answerer: usize,
        lists: [&[u32]; 2],
        announced: u32,
    ) -> u32 {
        self.stats.depth = 0;
        let mine = lists[answerer];
        if mine.len() == 1 {
            return mine[0];
        }
        self.begin_search();
        let mut ordered = mine.to_vec();
        let mut best_move = ordered[0];
        for depth in 1..=self.spec.max_depth as i32 {
            match self.answer_root(game, 1 - answerer, announced, &ordered, depth) {
                Err(Exhausted) => break,
                Ok(candidate) => {
                    best_move = candidate;
                    self.stats.depth = depth as u32;
                    promote(&mut ordered, best_move);
                    if self.nodes >= self.spec.nodes {
                        break;
                    }
                }
            }
        }
        self.end_search();
        best_move
    }

    fn answer_root(
        &mut self,
        game: &mut Game<'_>,
        announcer: usize,
        announced: u32,
        moves: &[u32],
        depth: i32,
    ) -> Result<u32, Exhausted> {
        let mut arena = std::mem::take(&mut self.arena);
        arena.clear();
        let mut alpha = f64::NEG_INFINITY;
        let mut best_move = moves[0];
        let mut outcome = Ok(());
        for &b in moves {
            game.make_round(seat_pair(announcer, announced, b));
            let child = self.negamax::<true>(game, &mut arena, depth - 1, 1, alpha, f64::INFINITY);
            game.unmake_round();
            match child {
                Err(Exhausted) => {
                    outcome = Err(Exhausted);
                    break;
                }
                Ok(score) => {
                    if score > alpha {
                        alpha = score;
                        best_move = b;
                    }
                }
            }
        }
        self.arena = arena;
        outcome.map(|()| best_move)
    }

    /// `greedy` as the announcer (§3.3): the one-round maximin over the whole matrix, every
    /// cell resolved and valued from the announcer's seat (a node each). The first move of
    /// the best worst case, in generation order.
    pub(super) fn announce_greedy(
        &mut self,
        game: &mut Game<'_>,
        announcer: usize,
        lists: [&[u32]; 2],
    ) -> u32 {
        let (mine, theirs) = (lists[announcer], lists[1 - announcer]);
        self.stats.moves += 1;
        if mine.len() == 1 {
            return mine[0];
        }
        self.nodes = 0;
        let mut arena = std::mem::take(&mut self.arena);
        arena.clear();
        let values = match self.round_cells(game, announcer, mine, theirs, 1, &mut arena, false) {
            Ok(values) => values,
            Err(Exhausted) => unreachable!("an unbounded pass never runs out"),
        };
        self.arena = arena;
        self.stats.nodes += self.nodes;
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (row, cells) in values.chunks(theirs.len()).enumerate() {
            let mut worst = f64::INFINITY;
            for value in cells {
                if *value < worst {
                    worst = *value;
                }
            }
            if worst > best_value {
                best_value = worst;
                best = row;
            }
        }
        mine[best]
    }

    /// `greedy` as the answerer (§3.3): the announced row only, a cell each of its own moves
    /// against `announced`, valued from its seat. The first best, in generation order.
    pub(super) fn answer_greedy(
        &mut self,
        game: &mut Game<'_>,
        answerer: usize,
        lists: [&[u32]; 2],
        announced: u32,
    ) -> u32 {
        let mine = lists[answerer];
        self.stats.moves += 1;
        if mine.len() == 1 {
            return mine[0];
        }
        self.nodes = 0;
        let mut arena = std::mem::take(&mut self.arena);
        arena.clear();
        let mut best = mine[0];
        let mut best_value = f64::NEG_INFINITY;
        for &b in mine {
            let value = match self.round_cell(game, answerer, b, announced, 1, &mut arena, false) {
                Ok(value) => value,
                Err(Exhausted) => unreachable!("an unbounded cell never runs out"),
            };
            if value > best_value {
                best_value = value;
                best = b;
            }
        }
        self.arena = arena;
        self.stats.nodes += self.nodes;
        best
    }

    /// Greedy's value of a turn's child that is a round, for `player`: its verdict (scored
    /// as greedy scores a turn's), else `evaluate_round`.
    pub(super) fn greedy_round_child(&mut self, game: &mut Game<'_>, player: usize) -> f64 {
        if let Some(outcome) = game.round_terminal_prefix() {
            return terminal_score(outcome, player);
        }
        let mut first = Vec::new();
        let mut second = Vec::new();
        game.generate_legal_for(0, &mut first);
        game.generate_legal_for(1, &mut second);
        match game.round_terminal_with([&first, &second]) {
            Some(outcome) => terminal_score(outcome, player),
            None => self.eval.evaluate_round(game, player, [&first, &second]),
        }
    }
}

/// A round's two moves in seat order, `a` the announcer's.
fn seat_pair(announcer: usize, a: u32, b: u32) -> [u32; 2] {
    let mut pair = [0u32; 2];
    pair[announcer] = a;
    pair[1 - announcer] = b;
    pair
}

/// Put `best` first, the rest in their order: the next iteration tries it first.
fn promote(ordered: &mut Vec<u32>, best: u32) {
    if let Some(position) = ordered.iter().position(|m| *m == best) {
        ordered.remove(position);
    }
    ordered.insert(0, best);
}
