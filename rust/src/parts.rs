//! Turn parts (TS3, `docs/plan-tessabi.md` §3, `rules.turn.parts`, `rules.turn.order`).
//!
//! **The turn.** The compiler makes a parts game's turn as many moves as it has parts
//! (`TurnRule { moves: parts }`), so `rules.turn`'s machinery carries it unchanged: `remaining`
//! counts the unfilled parts, the side flips when the last is filled, `turns_taken` counts
//! turns, the searches keep the sign while the side does not change. This file adds which
//! piece a part admits, the keep, and the completion filter.
//!
//! **The mask** (`Game::parts_filled`): a bit per part filled this turn, 0 at every turn's
//! start, in the key (`PartsRule::key`), logged on `parts_log` by `make` under the rule only.
//! Under order `"fixed"` the filled parts are always the first ones, so the next is the
//! count; under `"any"` a move fills the one unfilled part that admits its piece (the
//! validator refuses parts that overlap there, so it is one).
//!
//! **Which piece** a move is of is read off its origin's byte before it moves: the side to
//! move's own pieces by their type (`PartsRule::own`), nobody's pieces the mover may move,
//! held or shared, by theirs (`PartsRule::nobody`). A drop, a buy, a fold's own move and a
//! command are refused beside parts, so every move offered here has a piece on its origin.
//!
//! **The keep** (`MOVE_KEEP`): fills an optional part and moves nothing; under `"fixed"` the
//! next part, under `"any"` the first unfilled optional one, whose index rides in the
//! promotion field. It is a ply (the no-progress clock counts it, the repetition window
//! stays open) and, as a turn's first move, closes the en passant window, as a turn's first
//! move of a piece does.
//!
//! **The completion filter** (`parts_filter`): a move is offered only if the turn can still be
//! finished after it, every mandatory part filled: no mandatory part left unfilled by it, or
//! a legal move after it (which is this same filter, one part shorter, so the search is at
//! most the turn's parts deep), or a position the game ends in. A turn with no completion
//! at its start lists nothing, which is `rules.stalemate`'s.
//!
//! **Endings** are read after every move, as a long turn reads them; repetition and the
//! clocks only at a turn's start (`Game::mid_turn`), as beside the duck.

use super::*;

impl<'t> Game<'t> {
    /// Whether a turn is in progress that repetition and the clocks must not read: an owed
    /// duck half, or a parts game with a part filled (`parts_filled` is 0 in every other
    /// game).
    #[inline(always)]
    pub fn mid_turn(&self) -> bool {
        self.duck_owed() || self.parts_filled != 0
    }

    /// The part `m` fills from the parts filled now.
    pub(super) fn part_of(&self, m: u32) -> usize {
        let t = self.t;
        let parts = t.parts.as_deref().expect("a part without turn parts");
        if move_kind(m) == MOVE_KEEP {
            return move_promotion(m) as usize;
        }
        let value = self.board[move_from(m)];
        debug_assert!(value != 0, "a parts game's move with no piece on its origin");
        let found = if value == 0 {
            None
        } else {
            parts.part_for(self.parts_filled, t.owner_of(value), self.side, t.type_of(value))
        };
        // A move the filter never offered (a hand-built word) fills the first unfilled part.
        found.unwrap_or_else(|| (0..parts.len()).find(|&p| self.parts_filled & (1 << p) == 0).unwrap_or(0))
    }

    /// `part` filled: the mask logged, moved on and keyed into `key`.
    pub(super) fn write_parts(&mut self, part: usize, key: &mut u64) {
        let parts = self.t.parts.as_deref().expect("parts written without turn parts");
        self.parts_log.push(self.parts_filled);
        let next = parts.after(self.parts_filled, part);
        *key ^= parts.key(self.parts_filled) ^ parts.key(next);
        self.parts_filled = next;
    }

    /// The turn's parts over a generated list from `base`: the moves no unfilled part admits
    /// dropped, the keep added where one is offered, then every move after which the turn
    /// cannot be finished dropped.
    #[cold]
    #[inline(never)]
    pub(super) fn parts_filter(&mut self, out: &mut Vec<u32>, base: usize) {
        let t = self.t;
        let parts = t.parts.as_deref().expect("the parts filter without turn parts");
        let filled = self.parts_filled;
        let side = self.side;
        let mut write = base;
        for index in base..out.len() {
            let m = out[index];
            let value = self.board[move_from(m)];
            if value != 0 && parts.part_for(filled, t.owner_of(value), side, t.type_of(value)).is_some() {
                out[write] = m;
                write += 1;
            }
        }
        out.truncate(write);
        if let Some(part) = parts.keep_part(filled) {
            out.push(encode_move(0, 0, MOVE_KEEP, part as u32));
        }
        let mut scratch: Vec<u32> = Vec::new();
        let mut write = base;
        for index in base..out.len() {
            let m = out[index];
            let part = self.part_of(m);
            let finished = parts.mandatory & !(filled | (1 << part)) == 0;
            if finished || self.completes(m, &mut scratch) {
                out[write] = m;
                write += 1;
            }
        }
        out.truncate(write);
    }

    /// Whether the turn can be finished after `m`: the game ends there, or a legal move
    /// follows (the filter again, one part shorter).
    fn completes(&mut self, m: u32, scratch: &mut Vec<u32>) -> bool {
        self.make(m);
        let answer = self.ending_before_the_clocks().is_some() || {
            let base = scratch.len();
            self.generate_legal(scratch);
            let any = scratch.len() > base;
            scratch.truncate(base);
            any
        };
        self.unmake();
        answer
    }

    /// Play a keep: the part filled, nothing moved, the ply spent. Cold and out of line
    /// (`make_duck`'s pattern).
    #[cold]
    #[inline(never)]
    pub(super) fn make_keep(&mut self, m: u32) {
        let t = self.t;
        let part = move_promotion(m) as usize;
        let player = self.side;
        let previous_remaining = self.remaining;
        self.undo.push(Undo {
            origin: 0,
            target: 0,
            kind: MOVE_KEEP,
            moving: 0,
            captured: 0,
            captured_square: 0,
            target_before: 0,
            rook: None,
            unmoved: self.unmoved,
            ep_target: self.ep_target,
            ep_victim: self.ep_victim,
            halfmove: self.halfmove,
            key: self.key,
            window: self.window,
            checks: self.checks,
            hand: None,
            cleared: Vec::new(),
            coloured: None,
            remaining: previous_remaining,
            turns_taken: self.turns_taken,
            npc_log: self.npc_log.len() as u32,
            placed: 0,
        });
        // Seeded with the side flipped, as `make` seeds it; `end_ply` undoes the flip when
        // the turn goes on.
        let mut key = self.key ^ t.zobrist_side;
        // The window is the side to move's for its turn's first move only: a keep that is
        // that move closes it, as a piece's first move does (`write_ep_window`).
        if self.parts_filled == 0 && self.ep_target != NO_SQUARE {
            key ^= t.zobrist_ep_file(self.ep_target);
            self.ep_target = NO_SQUARE;
            self.ep_victim = NO_SQUARE;
        }
        self.write_parts(part, &mut key);
        self.end_ply(player, key, previous_remaining, false, false);
    }

    /// Take back a keep: every field `make_keep` changed, from its `Undo` and the parts log.
    #[cold]
    #[inline(never)]
    pub(super) fn unmake_keep(&mut self, undo: &Undo) {
        self.ep_target = undo.ep_target;
        self.ep_victim = undo.ep_victim;
        self.halfmove = undo.halfmove;
        self.key = undo.key;
        self.window = undo.window;
        if self.turns_taken != undo.turns_taken {
            self.side = 1 - self.side;
        }
        self.remaining = undo.remaining;
        self.turns_taken = undo.turns_taken;
        self.parts_filled = self.parts_log.pop().expect("parts log shorter than the undo stack");
        self.ply -= 1;
        self.check_memo.set([-1, -1]);
    }
}
