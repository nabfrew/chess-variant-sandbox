//! The duck (generation 42, N4b, `docs/archive/plan-npc.md` §4.6): one piece nobody owns, which the
//! mover moves as the second half of every turn.
//!
//! **The turn.** The compiler makes a duck game's turn two moves long (`TurnRule { moves: 2 }`),
//! so `rules.turn`'s machinery carries the halves unchanged: `remaining` 2 owes the piece move,
//! 1 the duck move; `zobrist_remaining` puts the half in the key; the side flips only when the
//! duck half is spent; `turns_taken` counts turns; the searches keep the sign when the side
//! does not flip. This file adds the duck's own half: which moves it has, and how one is made
//! and taken back.
//!
//! **Where the duck is** is read off the board: the square of the third owner's set whose
//! byte is the duck's (`duck_square`). No state, nothing in `Snapshot`, `Undo` or the key
//! beyond the board and the half.
//!
//! **Its moves** (`MOVE_DUCK`, the promotion field saying how): off the board or under
//! `goes: "anywhere"`, every empty square the duck may stand on, ascending; under `"by_moves"`
//! on the board, its type's frame-0 leap and slide entries from its square onto empty squares
//! with a clear path, in table order; and if that leaves nothing, the one pass word. So the
//! list is never empty, and a position owing a duck move is never mate or stalemate.
//!
//! **Its half** (`make_duck`): the duck written off its square and onto the target through
//! `write` (so the key and the occupancy follow), the half spent (the side flips), and then
//! the wild phase if the game has wild pieces, with the duck's mover as the mover: under the
//! duck the phase runs once a turn, after the duck half, and never after the piece move. The
//! duck move leaves the en passant window, the no-progress clock and the repetition window
//! alone; a phase that eats or clears a right moves them as it does after a piece move.

use super::*;

impl<'t> Game<'t> {
    /// Whether the side to move owes its duck move: a duck game, the piece half spent.
    #[inline(always)]
    pub fn duck_owed(&self) -> bool {
        self.t.duck.is_some() && self.remaining == 1
    }

    /// The duck's square, or `None` while it is still off the board (only the opening can be,
    /// since nothing removes it). Read off the third owner's set, checked against the board.
    pub fn duck_square(&self) -> Option<usize> {
        let duck = self.t.duck.as_ref()?;
        debug_assert!(self.t.kind_of(duck.value) == CONTROL_DUCK);
        self.occupied[WILD].iter().find(|&square| self.t.kind_of(self.board[square]) == CONTROL_DUCK)
    }

    /// The duck's moves (see the module's note), appended to `out`. Never none.
    #[cold]
    #[inline(never)]
    pub(super) fn push_duck_moves(&self, out: &mut Vec<u32>) {
        let t = self.t;
        let duck = t.duck.as_ref().expect("a duck move without the duck");
        let start = out.len();
        let at = self.duck_square();
        match at {
            Some(from) if duck.by_moves => {
                let slot = t.slot(0, duck.piece_type, from);
                for entry in t.leaps.entries(slot) {
                    let squares = t.leaps.squares(entry);
                    let modes = t.leaps.modes(entry);
                    for index in 0..squares.len() {
                        let target = squares[index] as usize;
                        if modes[index] & 1 != 0 && self.board[target] == 0 && duck.squares.get(target) {
                            out.push(encode_move(from, target, MOVE_DUCK, DUCK_STEP));
                        }
                    }
                }
                for entry in t.slides.entries(slot) {
                    let squares = t.slides.squares(entry);
                    let modes = t.slides.modes(entry);
                    for index in 0..squares.len() {
                        let target = squares[index] as usize;
                        if self.board[target] != 0 {
                            break;
                        }
                        if modes[index] & 1 != 0 && duck.squares.get(target) {
                            out.push(encode_move(from, target, MOVE_DUCK, DUCK_STEP));
                        }
                    }
                }
            }
            Some(from) => {
                // Its own square is occupied, which is the *may not stay*.
                let empty = duck.squares.difference(&self.all_pieces());
                for target in empty.iter() {
                    out.push(encode_move(from, target, MOVE_DUCK, DUCK_STEP));
                }
            }
            None => {
                let empty = duck.squares.difference(&self.all_pieces());
                for target in empty.iter() {
                    out.push(encode_move(target, target, MOVE_DUCK, DUCK_PLACE));
                }
            }
        }
        if out.len() == start {
            let square = at.unwrap_or(0);
            out.push(encode_move(square, square, MOVE_DUCK, DUCK_PASS));
        }
    }

    /// `make`'s stand-in for the phase on a duck game's piece half: no wild piece moves, but
    /// the phase's logs are written as `npc_phase_of_make` writes them, so `unmake`'s one
    /// `unmake_phase` reads them back alike, and a heading under a piece just taken is dropped
    /// at once (the key reads the headings off the board). Eats nothing.
    #[cold]
    #[inline(never)]
    pub(super) fn duck_piece_half(&mut self) -> bool {
        if self.t.respawn.is_some() {
            self.respawn_save();
        }
        if self.t.npc_lines.is_some() {
            self.heading_log.push(self.heading_back);
            self.heading_tidy();
        }
        false
    }

    /// Play a duck move: the duck moves (or is placed, or passes), the half is spent and the
    /// side flips, and the wild phase runs. Cold and out of line (`make_fold`'s pattern).
    #[cold]
    #[inline(never)]
    pub(super) fn make_duck(&mut self, m: u32) {
        let t = self.t;
        let duck = t.duck.as_ref().expect("a duck move without the duck");
        let origin = move_from(m);
        let target = move_to(m);
        let how = move_promotion(m);
        let player = self.side;
        let previous_unmoved = self.unmoved;
        let previous_ep = self.ep_target;
        let previous_ep_victim = self.ep_victim;
        let previous_remaining = self.remaining;
        let previous_turns = self.turns_taken;
        let previous_key = self.key;
        let previous_window = self.window;
        let previous_halfmove = self.halfmove;
        debug_assert_eq!(previous_remaining, 1, "a duck move with no duck move owed");
        // `write` keys every square it changes into `working_key`, the phase's too.
        self.working_key = previous_key;
        match how {
            DUCK_STEP => {
                self.write(origin, 0);
                self.write(target, duck.value);
            }
            DUCK_PLACE => self.write(target, duck.value),
            _ => {}
        }
        // The phase, once a turn, after the duck half, the duck's mover as the mover (`side`
        // has not flipped yet).
        let npc_log = self.npc_log.len() as u32;
        let phase_ate = t.has_npcs && self.npc_phase_of_make();
        // `write` keyed the squares; what is left is the window the phase may have closed.
        let mut key = self.working_key;
        if previous_ep != NO_SQUARE && self.ep_target == NO_SQUARE {
            key ^= t.zobrist_ep_file(previous_ep);
        }
        self.undo.push(Undo {
            origin,
            target,
            kind: MOVE_DUCK,
            moving: duck.value,
            captured: 0,
            captured_square: target,
            target_before: 0,
            rook: None,
            unmoved: previous_unmoved,
            ep_target: previous_ep,
            ep_victim: previous_ep_victim,
            halfmove: previous_halfmove,
            key: previous_key,
            window: previous_window,
            checks: self.checks,
            hand: None,
            cleared: Vec::new(),
            coloured: None,
            remaining: previous_remaining,
            turns_taken: previous_turns,
            npc_log,
            placed: how as u8,
        });
        // The half spent: the turn passes.
        self.side = 1 - player;
        self.turns_taken += 1;
        self.remaining = t.turn.as_ref().map_or(1, |rule| rule.length_of(self.turns_taken));
        key ^= t.zobrist_side;
        key ^= t.zobrist_remaining(previous_remaining);
        key ^= t.zobrist_remaining(self.remaining);
        self.key = key;
        // The duck move itself is no progress and reversible; a phase that ate is both, and
        // one that cleared a right is irreversible, as after a piece move.
        if phase_ate {
            self.halfmove = 0;
        }
        self.ply += 1;
        self.history.push(self.key);
        if phase_ate || self.unmoved != previous_unmoved {
            self.window = self.history.len() - 1;
        }
        self.check_memo.set([-1, -1]);
    }

    /// Take back a duck move: the phase walked back, the duck put back, every field restored.
    #[cold]
    #[inline(never)]
    pub(super) fn unmake_duck(&mut self, undo: &Undo) {
        if self.t.has_npcs {
            self.unmake_phase(undo.npc_log as usize);
        }
        match undo.placed as u32 {
            DUCK_STEP => {
                self.put(undo.target, 0);
                self.put(undo.origin, undo.moving);
            }
            DUCK_PLACE => self.put(undo.target, 0),
            _ => {}
        }
        self.unmoved = undo.unmoved;
        self.ep_target = undo.ep_target;
        self.ep_victim = undo.ep_victim;
        self.halfmove = undo.halfmove;
        self.key = undo.key;
        self.window = undo.window;
        self.side = 1 - self.side;
        self.remaining = undo.remaining;
        self.turns_taken = undo.turns_taken;
        self.ply -= 1;
        self.check_memo.set([-1, -1]);
    }
}
