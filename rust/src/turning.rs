//! Wild pieces that turn back (generation 38, N5, `docs/plan-npc.md` §4.7): step 2 of the
//! wild-piece phase for a piece on a **line**.
//!
//! **The rule, stated once** (`Game::turn_step`). A turning-back entry's route is a line,
//! walked first to last and back; it never wraps. Each piece on it has a heading, forward at
//! the start. In its step it looks at the next square of its heading: if that is empty or
//! holds a piece it may take (not wild, not shielded at the phase's start, of a type its own
//! may capture), it steps there. Otherwise it **turns round** and looks once the other way:
//! free, it steps there; not, it waits, turned. A piece that lands on an end of its line
//! turns there, so a piece standing on an end always faces into the line (the plan's *at an
//! end of the line the heading turns first*, kept as state that never says otherwise). At an
//! end there is nothing the other way, so a piece blocked there waits facing in, unturned.
//!
//! **The state.** `Game::heading_back`, the squares of the pieces walking their line back:
//! a square set that moves with the piece, as the unmoved bits do. In the key
//! (`LineTables::zobrist`, one key per line square, drawn last of all and only under the
//! rule), on `Snapshot`, across the boundary (slot 11's eleventh element), and logged by
//! `make` on `heading_log` only under the rule, so `Undo` does not grow; `make_round` keeps it
//! on `RoundUndo`. A taken piece's bit is cleared at the next phase's start (`heading_tidy`),
//! which is inside the same `make`, so every position a caller sees holds bits only under
//! turning-back pieces.
//!
//! **The log.** A step that turned round at a block is `PHASE_TURN`, `(from, to, eaten)`, and
//! a wait with the heading turned is `(square, square, 0)`; `unmake` reads both as a step. The
//! turn at an end is part of the walk and logs as a step.

use super::*;

/// What one turning-back piece does in a phase: where it ends (`to`, its own square when it
/// waits), which way it then faces, and whether it turned round because its way was blocked.
#[derive(Clone, Copy)]
pub(super) struct TurnStep {
    pub to: usize,
    pub back: bool,
    pub turned: bool,
}

impl<'t> Game<'t> {
    /// Whether the wild piece `value` may step onto `target`: empty, or a player's piece its
    /// type may capture that `shields` (the phase's snapshot) does not cover.
    #[inline]
    fn wild_may_enter(&self, value: u8, target: usize, shields: &SquareSet) -> bool {
        let t = self.t;
        let standing = self.board[target];
        if standing == 0 {
            return true;
        }
        if t.owner_of(standing) == WILD || shields.get(target) {
            return false;
        }
        t.may_capture(t.type_of(value), t.type_of(standing))
    }

    /// **The rule, stated once**, for the piece on line square `square` facing back (`back`)
    /// or forward, on the board as the earlier steps of the phase left it.
    pub(super) fn turn_step(&self, square: usize, back: bool, shields: &SquareSet) -> TurnStep {
        let t = self.t;
        let lines = t.npc_lines.as_deref().expect("turn_step without a line");
        let value = self.board[square];
        debug_assert!(value != 0 && t.owner_of(value) == WILD && lines.squares.get(square));
        let landed = |to: usize, back: bool, turned: bool| {
            // Landing on an end turns the piece into its line.
            let back = if lines.ahead(&t.npc_next, to, back).is_none() { !back } else { back };
            TurnStep { to, back, turned }
        };
        let Some(ahead) = lines.ahead(&t.npc_next, square, back) else {
            // Facing off an end (a position built by hand): the end turns it first.
            return match lines.ahead(&t.npc_next, square, !back) {
                Some(other) if self.wild_may_enter(value, other, shields) => {
                    landed(other, !back, false)
                }
                _ => TurnStep { to: square, back: !back, turned: false },
            };
        };
        if self.wild_may_enter(value, ahead, shields) {
            return landed(ahead, back, false);
        }
        match lines.ahead(&t.npc_next, square, !back) {
            // On an end, blocked: nothing the other way, so it waits facing in.
            None => TurnStep { to: square, back, turned: false },
            Some(other) if self.wild_may_enter(value, other, shields) => landed(other, !back, true),
            Some(_) => TurnStep { to: square, back: !back, turned: true },
        }
    }

    /// Set or clear `square`'s heading bit, the working key following.
    #[inline]
    fn set_heading(&mut self, square: usize, back: bool) {
        if self.heading_back.get(square) == back {
            return;
        }
        let lines = self.t.npc_lines.as_deref().expect("a heading without a line");
        self.working_key ^= lines.zobrist[square];
        if back {
            self.heading_back.set(square);
        } else {
            self.heading_back.clear(square);
        }
    }

    /// Clear every heading bit with no turning-back piece under it (one a player has just
    /// taken), the working key following. At the phase's start, and in a round that ended
    /// before its phase.
    pub(super) fn heading_tidy(&mut self) {
        let t = self.t;
        let standing = self.heading_back;
        for square in standing.iter() {
            let value = self.board[square];
            if value == 0 || t.owner_of(value) != WILD {
                self.set_heading(square, false);
            }
        }
    }

    /// The phase, played for real, in a game with a line (generation 38): the steps (each
    /// line piece by `turn_step`, every other by `npc_step_of`), then the returns. Every
    /// square through `write`, every heading change into the working key. Its own function,
    /// so the phase of every game without the rule is the one it was. Returns whether
    /// anything was eaten.
    #[cold]
    #[inline(never)]
    pub(super) fn npc_phase_lines(&mut self) -> bool {
        let t = self.t;
        let lines = t.npc_lines.as_deref().expect("npc_phase_lines without a line");
        let shields = self.wild_shields();
        self.heading_tidy();
        let mut ate = false;
        let order: Vec<usize> = self.npc_step_order().collect();
        for square in order {
            let value = self.board[square];
            let (next, kind) = if lines.squares.get(square) {
                let step = self.turn_step(square, self.heading_back.get(square), &shields);
                self.set_heading(square, false);
                self.set_heading(step.to, step.back);
                let kind = if step.turned { PHASE_TURN } else { PHASE_STEP };
                if step.to == square {
                    if step.turned {
                        self.npc_log.push((square, square, 0, PHASE_TURN));
                    }
                    continue;
                }
                (step.to, kind)
            } else {
                match self.npc_step_of::<true>(square, &shields) {
                    Some(next) => (next, PHASE_STEP),
                    None => continue,
                }
            };
            let eaten = self.board[next];
            self.write(square, 0);
            self.write(next, value);
            self.unmoved.clear(square);
            self.unmoved.clear(next);
            ate |= eaten != 0;
            self.close_ep_window_over(square, next);
            self.npc_log.push((square, next, eaten, kind));
        }
        if t.respawn.is_some() {
            self.respawn_phase(&shields) | ate
        } else {
            ate
        }
    }

    /// The same phase on the raw board for a legality filter, every square written pushed to
    /// `saved`. The headings are read, never written: each piece reads its own square's bit
    /// before anything moves it, and the filter asks about one position.
    #[cold]
    #[inline(never)]
    pub(super) fn npc_phase_for_filter_lines(&mut self, saved: &mut Saved) {
        let t = self.t;
        let lines = t.npc_lines.as_deref().expect("npc_phase_for_filter_lines without a line");
        let shields = self.wild_shields();
        let order: Vec<usize> = self.npc_step_order().collect();
        for square in order {
            let next = if lines.squares.get(square) {
                let to = self.turn_step(square, self.heading_back.get(square), &shields).to;
                if to == square {
                    continue;
                }
                to
            } else {
                match self.npc_step_of::<true>(square, &shields) {
                    Some(next) => next,
                    None => continue,
                }
            };
            let value = self.board[square];
            saved.push((square, value));
            saved.push((next, self.board[next]));
            self.board[square] = 0;
            self.board[next] = value;
        }
        if t.respawn.is_some() {
            self.respawn_phase_for_filter(saved, &shields);
        }
    }

    /// One phase's steps for `coming_phases` in a game with a line, on the raw board with the
    /// headings in `heading`, which it moves on; each step (and each turned wait) pushed to
    /// `steps`.
    pub(super) fn npc_coming_lines(
        &mut self,
        heading: &mut SquareSet,
        saved: &mut Saved,
        steps: &mut Vec<(usize, usize, u8, u8)>,
        shields: &SquareSet,
    ) {
        let lines = self.t.npc_lines.as_deref().expect("npc_coming_lines without a line");
        let order: Vec<usize> = self.npc_step_order().collect();
        for square in order {
            let (next, kind) = if lines.squares.get(square) {
                let step = self.turn_step(square, heading.get(square), shields);
                heading.clear(square);
                if step.back {
                    heading.set(step.to);
                }
                let kind = if step.turned { PHASE_TURN } else { PHASE_STEP };
                if step.to == square {
                    if step.turned {
                        steps.push((square, square, 0, PHASE_TURN));
                    }
                    continue;
                }
                (step.to, kind)
            } else {
                match self.npc_step_of::<true>(square, shields) {
                    Some(next) => (next, PHASE_STEP),
                    None => continue,
                }
            };
            let value = self.board[square];
            let eaten = self.board[next];
            saved.push((square, value));
            saved.push((next, eaten));
            self.board[square] = 0;
            self.board[next] = value;
            steps.push((square, next, eaten, kind));
        }
    }

    /// The headings' part of the key; 0 in a game without the rule or with every piece
    /// walking forward.
    pub(super) fn heading_key(&self) -> u64 {
        let Some(lines) = self.t.npc_lines.as_deref() else {
            return 0;
        };
        self.heading_back.iter().fold(0, |key, square| key ^ lines.zobrist[square])
    }
}

/// The headings as Python holds them (squares walking back): kept only where a wild piece
/// stands on a line, so a position built by hand is read the way the phase would leave it.
pub fn heading_from(t: &Tables, board: &[u8], given: &[u16]) -> SquareSet {
    let mut set = SquareSet::EMPTY;
    let Some(lines) = t.npc_lines.as_deref() else {
        return set;
    };
    for &square in given {
        let square = square as usize;
        if square < t.n_squares && lines.squares.get(square) {
            let value = board[square];
            if value != 0 && t.owner_of(value) == WILD {
                set.set(square);
            }
        }
    }
    set
}
