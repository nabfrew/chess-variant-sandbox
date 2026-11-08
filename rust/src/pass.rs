//! The pass (H3, `docs/plan-hunger.md` §4 and §R3, `rules.pass`).
//!
//! **The move** (`MOVE_PASS`): the side to move makes no move. Origin, target and promotion 0;
//! nothing on the board changes, no piece is taken, entered or turned, and nothing is paid (D4:
//! a move that places nothing pays nothing).
//!
//! **Where it is offered** (`push_pass`): once, at the very end of `generate_legal`, after every
//! filter, so it never changes which other moves are offered (a rest is lifted, or not, over the
//! piece moves alone). It is the null move **filtered as every move is** (D1, as Q1130 re-cut
//! it): the board after it is the board before it, so the check filter drops it exactly when the
//! side to move is in check, and `forbid_check` drops it exactly when the opponent's royal stands
//! attacked by the mover now. In a game without check (royal capture, annihilation, none) nothing
//! drops it. So under `rules.pass` a player out of check always has a move, and stalemate cannot
//! happen (the validator refuses a `rules.stalemate` other than the default beside it).
//!
//! **What it does** (`make_pass`): a ply of the player's turn, as a fold's own move is: the en
//! passant window closes (it was the mover's for this move only), a check the position already
//! shows is counted under `check_limit` as any move's is, a goal left last move is no longer the
//! last move's, the player's rests count down (a pass is their turn), and `end_ply` spends the
//! turn. It is reversible: it keeps the repetition window open and counts toward the no-progress
//! clock. No new state: everything it changes is on `Undo` or a log that already exists.
//!
//! **Refused beside it** (the validator's `pass.P*`; `Tables::from_py` holds all but forced capture and the placement phase, which it never reads): rounds, the duck, any
//! turn longer than one move (turn parts too), wild pieces (whether a pass sets the wild phase
//! going is unsettled), portals and a push (each keeps a bar for the very next move), forced
//! capture and a placement phase.

use super::*;

impl<'t> Game<'t> {
    /// Append the pass to `out` where the null move survives the legality filter: never in check
    /// (a game with check), never with the opponent's royal attacked under `forbid_check`.
    #[cold]
    #[inline(never)]
    pub(super) fn push_pass(&mut self, out: &mut Vec<u32>) {
        let t = self.t;
        let player = self.side;
        if t.needs_check_filter() && self.in_check(player) {
            return;
        }
        if t.forbid_check && !t.royal_types.is_empty() {
            let opponent = 1 - player;
            let gives = if t.multi_royal {
                self.royal_squares(opponent).iter().any(|&royal| self.threatened(player, royal))
            } else {
                let royal = self.royal_square(opponent);
                royal >= 0 && self.threatened(player, royal as usize)
            };
            if gives {
                return;
            }
        }
        out.push(encode_move(0, 0, MOVE_PASS, 0));
    }

    /// Play a pass: nothing moved, the turn spent. Cold and out of line (`make_keep`'s pattern).
    #[cold]
    #[inline(never)]
    pub(super) fn make_pass(&mut self) {
        let t = self.t;
        let player = self.side;
        let previous_remaining = self.remaining;
        self.undo.push(Undo {
            origin: 0,
            target: 0,
            kind: MOVE_PASS,
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
        // Seeded with the side flipped, as `make` seeds it.
        let mut key = self.key ^ t.zobrist_side;
        if self.ep_target != NO_SQUARE {
            key ^= t.zobrist_ep_file(self.ep_target);
            self.ep_target = NO_SQUARE;
            self.ep_victim = NO_SQUARE;
        }
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
        if t.goal_event {
            self.goal_left_log.push(self.goal_left_by);
            self.goal_left_by = -1;
        }
        // A pass is the player's turn, so their rests count down; no piece moved, so none enters.
        if t.rest.is_some() {
            self.rest_log.push(self.resting);
            let ends_turn = self.rest_turn_ends();
            key ^= self.rest_after_turn(player, None, player, ends_turn);
        }
        self.end_ply(player, key, previous_remaining, false, false);
    }

    /// Take back a pass: every field `make_pass` changed, from its `Undo` and the logs.
    #[cold]
    #[inline(never)]
    pub(super) fn unmake_pass(&mut self, undo: &Undo) {
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
        if self.t.goal_event {
            self.goal_left_by = self.goal_left_log.pop().expect("goal log shorter than the undo stack");
        }
        if self.t.rest.is_some() {
            self.resting = self.rest_log.pop().expect("rest log shorter than the undo stack");
        }
        self.ply -= 1;
        self.check_memo.set([-1, -1]);
    }
}
