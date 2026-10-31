//! The push rule (Q1074b2, `docs/plan-push.md` §1 to §3): a move that lands on a piece it may
//! push moves that piece one step on along the push direction, or off the board.
//!
//! The compiler resolved everything a push means into `PushTables` (`Tables::push`): which
//! landings push and whose pieces (the landing's mode bits, `MODE_PUSH_ENEMY` 4 and
//! `MODE_PUSH_OWN` 8), the push direction at each landing (parallel to the movement tables'
//! landings), where a pushed piece goes from each square (`next`, the edge and walls
//! resolved), who may push whom (`allowed`) and where a pushed piece may stand (`stand`).
//! Nothing here computes a square. Two statements are the whole of it:
//!
//! * `push_offered`, from the three movement arms of the generator when a landing with a push
//!   bit holds a piece: the `MOVE_PUSH` words that landing offers, where `push_line_moves`
//!   says the line may move (`"one"`: the piece behind must be empty, or the edge under
//!   `edge: "removes"`; `"row"`, `"weight"`: the whole line to its first empty square).
//! * `push_shift_line`, the one reading of where the line's pieces go, asked by `make`'s
//!   `write_step` and by the legality filter's play, so the two can never disagree.
//!
//! Under `chain: "row"` and `"weight"` (Q1074b4) the line is the target's piece and every
//! piece behind it up to the first empty square; `push_line_moves` decides whether it may
//! move. The piece that leaves the board (the line's last) is `make`'s `captured`, with the
//! square it fell from; when nothing falls, `captured` is 0 and `Undo.captured_square` names
//! the sink, so the generic restore (`target_before` on the target, then `captured` on that
//! square, then the mover on its origin) takes a line of one back with no field of its own.
//! A longer line's middle comes back by `push_unshift`, which needs the push direction:
//! `Game::push_log` holds it under the longer chains only, so `Undo` does not grow.
//!
//! Every game without a push has `Tables::push` `None`, and pays one mode-bit test on an
//! occupied landing. `tests/push_oracle.py` is the declared oracle.

use super::*;

/// `push_offered`'s `arm`: which movement table a landing index reads.
pub(super) const PUSH_ARM_LEAP: usize = 0;
pub(super) const PUSH_ARM_SLIDE: usize = 1;
pub(super) const PUSH_ARM_HOP: usize = 2;

impl<'t> Game<'t> {
    /// The `MOVE_PUSH` words a landing on the occupied `target` offers to `player`'s piece of
    /// `piece_type` on `origin`. `landing` is the landing's index in its movement table, `arm`
    /// which table (`PUSH_ARM_LEAP`, `_SLIDE`, `_HOP`), for its push direction; `mode` is the
    /// landing's mode, which holds a push bit.
    ///
    /// The line landed on must be one that may move (`push_line_moves`: whose, the lists, the
    /// regions, the edge, the chain and the weights). The words carry the pusher's promotion as `push_arrival` would give a
    /// move ending there, the type plus one, as `MOVE_PORTAL`'s.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push_offered(
        &self, out: &mut Vec<u32>, origin: usize, target: usize, piece_type: usize,
        player: usize, mode: u8, arm: usize, landing: usize, promoting: bool, leaving: bool,
        promo_optional: bool,
    ) {
        let t = self.t;
        let Some(push) = t.push.as_deref() else {
            return;
        };
        let direction = match arm {
            PUSH_ARM_LEAP => push.leap_dirs[landing],
            PUSH_ARM_SLIDE => push.slide_dirs[landing],
            _ => push.hop_dirs[landing],
        };
        if direction < 0 {
            return;
        }
        let direction = direction as usize;
        // K5 (Q1074b6): the piece pushed onto `origin` from `target` by the last move may not
        // push straight back onto it. `push_bar` is `NO_BAR` without the rule, so this never
        // matches there.
        if self.push_bar[0] == origin as i16 && self.push_bar[1] == target as i16 {
            return;
        }
        if !self.push_line_moves(push, origin, target, direction, player, piece_type, mode) {
            return;
        }
        let word = |promoted: u32| {
            with_push(encode_move(origin, target, MOVE_PUSH, promoted), direction)
        };
        if promoting && (leaving || t.promotes_on(player, piece_type, target)) {
            for &promoted in t.promotion_targets(player, piece_type) {
                if self.promotion_offered(player, piece_type, promoted) {
                    out.push(word(promoted + 1));
                }
            }
            if promo_optional {
                out.push(word(0));
            }
        } else {
            out.push(word(0));
        }
    }

    /// Whether the line a push by `player`'s piece of `piece_type` from `origin` onto the
    /// occupied `target` in `direction` may move (§2), under a landing of `mode`.
    ///
    /// The line is the target's piece and every piece behind it, walked along `next`, up to
    /// the first empty square (the pusher's origin counts as empty, since it leaves it in
    /// the same move), the edge, or a wall. Every piece in it must be one the landing pushes
    /// (the enemy's under `MODE_PUSH_ENEMY`, the mover's own under `MODE_PUSH_OWN`; nobody's,
    /// the duck and wild pieces never, N1-N3) and of a type the pusher's list names, and
    /// each must be able to stand where it goes (its region, W5). A wall behind the line
    /// blocks it (W1); so does the edge under `edge: "blocks"`, or where the piece that would
    /// fall off is the mover's own royal (R4). Under `chain: "one"` the line is one piece;
    /// under `"weight"` the pusher's strength must be at least the line's summed weight (the
    /// user, 2026-10-30: greater or equal; strength and weight are two numbers, Q1074b4s).
    /// A line that cannot move is no push: it never stops part-way.
    ///
    /// Q1074b7: under `walls: "crush"` a wall behind the line (`PUSH_CRUSH`) removes the
    /// piece pressed against it, as the edge removes one (C1), never the mover's own royal
    /// (R4). On a wrapped board (H4) the walk stops at a square it already walked: a line
    /// that comes back round to its target is a full ring, with no sink, and cannot move; one
    /// that reaches the pusher's origin ends there, the origin being empty.
    #[allow(clippy::too_many_arguments)]
    fn push_line_moves(
        &self, push: &PushTables, origin: usize, target: usize, direction: usize,
        player: usize, piece_type: usize, mode: u8,
    ) -> bool {
        let t = self.t;
        let mut square = target;
        let mut load = 0u32;
        for _ in 0..t.n_squares {
            let occupant = self.board[square];
            let owner = t.owner_of(occupant);
            let whose = if owner == player {
                MODE_PUSH_OWN
            } else if owner == 1 - player {
                MODE_PUSH_ENEMY
            } else {
                return false;
            };
            let victim = t.type_of(occupant);
            if mode & whose == 0 || !push.may_push(t.n_types, piece_type, victim) {
                return false;
            }
            if push.chain == PUSH_CHAIN_WEIGHT {
                load += u32::from(push.weight[victim]);
            }
            let behind = push.next_of(t.n_squares, direction, square);
            if behind == PUSH_STOP {
                return false;
            }
            if behind == PUSH_OFF || behind == PUSH_CRUSH {
                if (behind == PUSH_OFF && !push.edge_removes)
                    || (owner == player && t.is_royal(victim))
                {
                    return false;
                }
                return push_bears(push, piece_type, load);
            }
            let behind = behind as usize;
            if behind == target {
                return false;
            }
            if !push.may_stand(t.n_types, t.n_squares, owner, victim, behind) {
                return false;
            }
            if behind == origin || self.board[behind] == 0 {
                return push_bears(push, piece_type, load);
            }
            if push.chain == PUSH_CHAIN_ONE {
                return false;
            }
            square = behind;
        }
        false
    }

    /// Q1074b6 (K5, `rules.push.no_push_back`): `make`'s bar after the move `played` (the
    /// word `m`), keyed into `key`. A push that moved its target's piece from `x` (the
    /// target) to `y` and did not remove it bars that piece from pushing from `y` onto `x`
    /// on the next piece move, where its owner makes that move (`push_bar_after`); every
    /// other move of `make`'s clears the bar. Every game without the rule skips this on one
    /// test.
    #[inline(always)]
    pub(super) fn write_push_bar(&mut self, m: u32, played: &Played, key: &mut u64) {
        let t = self.t;
        if let Some(push) = t.push.as_deref().filter(|push| push.no_back()) {
            self.push_bar_log.push(self.push_bar);
            let bar = if played.kind == MOVE_PUSH {
                self.push_bar_after(m, played)
            } else {
                NO_BAR
            };
            *key ^= push.bar_key(t.n_squares, self.push_bar) ^ push.bar_key(t.n_squares, bar);
            self.push_bar = bar;
        }
    }

    /// The bar a push leaves (K5): `[y, x]`, or `NO_BAR` where the target's piece fell off
    /// (it is no longer on the board to push back), or where the piece now on `y` is not
    /// the next piece mover's (a bar only that player could meet, Q883's reading for the
    /// portals' bar: a bar nobody can meet is never set or keyed). The next piece mover is
    /// the same player while a long turn has moves left, and the opponent otherwise, a duck
    /// half between (the duck's move is no piece move, and leaves the bar standing).
    #[cold]
    #[inline(never)]
    fn push_bar_after(&self, m: u32, played: &Played) -> [i16; 2] {
        let t = self.t;
        let target = played.target;
        if played.captured != 0 && played.captured_square == target {
            return NO_BAR;
        }
        let Some(standing) = self.push_next(m, target) else {
            return NO_BAR;
        };
        let player = played.player;
        let next = if t.duck.is_none() && t.turn.is_some() && self.remaining > 1 {
            player
        } else {
            1 - player
        };
        if t.owner_of(self.board[standing]) != next {
            return NO_BAR;
        }
        [standing as i16, target as i16]
    }

    /// The square after `square` on the line of the push `m`, `None` off the board or
    /// against a crushing wall (Q1074b7). Asked only of a line the push was offered for, so
    /// never of a wall that blocks.
    #[inline]
    pub(super) fn push_next(&self, m: u32, square: usize) -> Option<usize> {
        let t = self.t;
        let push = t.push.as_deref().expect("a push word in a game without a push");
        let behind = push.next_of(t.n_squares, move_push(m), square);
        debug_assert!(behind != PUSH_STOP, "a push onto a wall was offered");
        (behind >= 0).then_some(behind as usize)
    }

    /// Q1074b4 (§2, K2): shift the line the `MOVE_PUSH` `m` landed on one step, the pusher
    /// already on the target (so its origin is empty) and `first` the piece that stood there.
    /// Walks forward carrying each piece onto the next square until that square was empty
    /// (the line's sink) or the line meets the edge or a crushing wall, and returns `(captured,
    /// captured_square)`: the piece that fell off and the square it fell from, or 0 and the
    /// sink. Under `chain: "one"` the line is one piece, so this is one write. `put(game,
    /// square, value)` writes a square: `make` keys it and clears its `unmoved` bit (E1, E3),
    /// the legality filter saves what stood there first. The one reading `make` and the
    /// filter share, so the two can never disagree.
    #[inline]
    pub(super) fn push_shift_line(
        &mut self, m: u32, first: u8, mut put: impl FnMut(&mut Self, usize, u8),
    ) -> (u8, usize) {
        let mut carry = first;
        let mut square = move_to(m);
        for _ in 0..self.t.n_squares {
            let Some(behind) = self.push_next(m, square) else {
                return (carry, square);
            };
            let there = self.board[behind];
            put(self, behind, carry);
            if there == 0 {
                return (0, behind);
            }
            carry = there;
            square = behind;
        }
        unreachable!("a push line with no end was offered")
    }

    /// Q1074b4 (K2): `unmake`'s half of `push_shift_line`, before the generic restore: every
    /// square of the line strictly between the target and `captured_square` gets back the
    /// piece one square further on, which is the piece that stood there. The generic restore
    /// then writes the target (`target_before`) and `captured_square` (the piece that fell
    /// off from it, or 0 on the sink). A line of one has nothing between; `direction` is
    /// `Game::push_log`'s.
    #[cold]
    #[inline(never)]
    pub(super) fn push_unshift(&mut self, direction: usize, target: usize, captured_square: usize) {
        if captured_square == target {
            return;
        }
        let t = self.t;
        let push = t.push.as_deref().expect("a push log in a game without a push");
        let step = |square: usize| push.next_of(t.n_squares, direction, square) as usize;
        let mut square = step(target);
        while square != captured_square {
            let behind = step(square);
            self.put(square, self.board[behind]);
            square = behind;
        }
    }

    /// Whether an attack source `source` of a piece of `attacker` type reaches a piece of
    /// `victim` type in a push game: by capture where its landing captures and the capture
    /// matrix lets it, by push where its landing pushes enemies and the lists let it (R1).
    /// `dirs` holds those two bits in a push game.
    #[inline]
    pub(super) fn push_game_attacks(
        &self, push: &PushTables, kinds: u8, attacker: usize, victim: usize,
    ) -> bool {
        let t = self.t;
        (kinds & ATTACK_BY_CAPTURE != 0 && t.may_capture(attacker, victim))
            || (kinds & ATTACK_BY_PUSH != 0 && push.may_push(t.n_types, attacker, victim))
    }
}

/// Whether the pusher bears a line of `load` (§2): always, unless the chain is `"weight"`,
/// where the pusher's strength (its weight unless stated, Q1074b4s) must be at least the
/// line's summed weight.
#[inline]
fn push_bears(push: &PushTables, piece_type: usize, load: u32) -> bool {
    push.chain != PUSH_CHAIN_WEIGHT || load <= u32::from(push.strength[piece_type])
}
