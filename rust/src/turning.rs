//! Wild pieces that turn back (generation 38, N5, `docs/archive/plan-npc.md` §4.7): step 2 of the
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
use super::sight::WildSight;
use super::phase::PhaseSink;

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
    /// type may capture that `shields` (the phase's snapshot) does not cover, unless it is
    /// `disarmed` (generation 84: it walks onto empty squares only).
    #[inline]
    fn wild_may_enter(
        &self, board: &[u8; MAX_SQUARES], value: u8, target: usize, shields: &SquareSet, disarmed: bool,
    ) -> bool {
        let t = self.t;
        let standing = board[target];
        if standing == 0 {
            return true;
        }
        if disarmed || t.halts_wild(standing) || shields.get(target) {
            return false;
        }
        t.may_capture(t.type_of(value), t.victim_of(standing))
    }

    /// **The rule, stated once**, for the piece on line square `square` facing back (`back`)
    /// or forward, on the board as the earlier steps of the phase left it. Generation 84
    /// (SR5f, V11): a piece the snapshot holds frozen waits as it faces, never turning; one it
    /// holds disarmed reads every piece in its way as a block.
    #[inline(always)]
    pub(super) fn turn_step(&self, square: usize, back: bool, sight: &WildSight) -> TurnStep {
        self.turn_step_on(&self.board, square, back, sight)
    }

    /// `turn_step` on a given board: the game's own, or `line_threatens`' copy (generation
    /// 125), so the threat and the phase read one statement of the rule.
    #[inline(always)]
    fn turn_step_on(
        &self, board: &[u8; MAX_SQUARES], square: usize, back: bool, sight: &WildSight,
    ) -> TurnStep {
        let t = self.t;
        let lines = t.npc_lines.as_deref().expect("turn_step without a line");
        let value = board[square];
        debug_assert!(value != 0 && t.owner_of(value) == WILD && lines.squares.get(square));
        if sight.frozen.get(square) {
            return TurnStep { to: square, back, turned: false };
        }
        let shields = &sight.shields;
        let disarmed = sight.disarmed.get(square);
        let landed = |to: usize, back: bool, turned: bool| {
            // Landing on an end turns the piece into its line.
            let back = if lines.ahead(&t.npc_next, to, back).is_none() { !back } else { back };
            TurnStep { to, back, turned }
        };
        let Some(ahead) = lines.ahead(&t.npc_next, square, back) else {
            // Facing off an end (a position built by hand): the end turns it first.
            return match lines.ahead(&t.npc_next, square, !back) {
                Some(other) if self.wild_may_enter(board, value, other, shields, disarmed) => {
                    landed(other, !back, false)
                }
                _ => TurnStep { to: square, back: !back, turned: false },
            };
        };
        if self.wild_may_enter(board, value, ahead, shields, disarmed) {
            return landed(ahead, back, false);
        }
        match lines.ahead(&t.npc_next, square, !back) {
            // On an end, blocked: nothing the other way, so it waits facing in.
            None => TurnStep { to: square, back, turned: false },
            Some(other) if self.wild_may_enter(board, value, other, shields, disarmed) => landed(other, !back, true),
            Some(_) => TurnStep { to: square, back: !back, turned: true },
        }
    }

    /// Set or clear `square`'s heading bit, the working key following.
    #[inline]
    pub(super) fn set_heading(&mut self, square: usize, back: bool) {
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
            // Generation 42: a duck put down where a taken piece stood carries no heading.
            if !t.is_wild_piece(self.board[square]) {
                self.set_heading(square, false);
            }
        }
    }

    /// Step 2 of the phase in a game with a line (generation 38), walked once (CS10b) into
    /// `sink`: in `LineOrder` (front first by each piece's heading, generation 123) where a
    /// line starts with two pieces, else in `npc_step_order`, each line
    /// piece by `turn_step` (its heading read and moved on by the sink), every other by
    /// `npc_step_of`, shielded, on the board as the earlier steps of the same phase leave it.
    /// A step that turned round at a block is a `PHASE_TURN`, and so is a turned wait
    /// (`PhaseSink::wait_turned`).
    #[inline(always)]
    pub(super) fn line_steps<S: PhaseSink>(&mut self, sink: &mut S, sight: &WildSight) {
        let t = self.t;
        let lines = t.npc_lines.as_deref().expect("line_steps without a line");
        if !lines.queues {
            let order = self.npc_step_order();
            return self.line_steps_in(sink, order, sight);
        }
        let here = self.npc_here();
        // Each pending piece's heading as the phase starts: a piece's bit is never written
        // before it steps (nothing lands on a square a wild piece still holds), so the order
        // reads the same headings whichever sink holds them.
        let mut back = SquareSet::EMPTY;
        for square in here.iter() {
            if lines.squares.get(square) && sink.heading(self, square) {
                back.set(square);
            }
        }
        self.line_steps_in(sink, LineOrder::new(t, lines, here, back), sight);
    }

    /// `line_steps` in a given order: `NpcOrder` where no line starts with two pieces
    /// (`LineTables::queues`), `LineOrder` where one does.
    #[inline(always)]
    fn line_steps_in<S: PhaseSink, I: Iterator<Item = usize>>(
        &mut self, sink: &mut S, order: I, sight: &WildSight,
    ) {
        let t = self.t;
        let lines = t.npc_lines.as_deref().expect("line_steps without a line");
        for square in order {
            let (next, kind) = if lines.squares.get(square) {
                let step = self.turn_step(square, sink.heading(self, square), sight);
                sink.turn(self, square, step);
                if step.to == square {
                    if step.turned {
                        sink.wait_turned(self, square);
                    }
                    continue;
                }
                (step.to, if step.turned { PHASE_TURN } else { PHASE_STEP })
            } else {
                match self.npc_step_of::<true>(square, sight) {
                    Some(next) => (next, PHASE_STEP),
                    None => continue,
                }
            };
            sink.step(self, square, next, kind);
        }
    }

    /// The whole phase of a game with a line: the shields read once, the steps, the returns.
    #[inline(always)]
    fn line_phase<S: PhaseSink>(&mut self, sink: &mut S) {
        let sight = self.wild_sight();
        self.line_steps(sink, &sight);
        self.respawn_walk(sink, &sight);
    }

    /// The phase, played for real, in a game with a line: a taken piece's heading dropped
    /// (`heading_tidy`), then `line_phase` into `Live`. Its own function, so the phase of
    /// every game without the rule is the one it was. Returns whether anything was eaten.
    #[cold]
    #[inline(never)]
    pub(super) fn npc_phase_lines(&mut self) -> bool {
        self.heading_tidy();
        let mut live = Live::new();
        self.line_phase(&mut live);
        live.ate
    }

    /// The same phase on the raw board for a legality filter, every square written pushed to
    /// `saved`. Generation 125 (SR8h, N5): the headings move with the pieces as the phase
    /// leaves them (`Filter::turn`, plain bits, no key), so the filter's threat question
    /// (`line_threatens`) reads the post-phase headings; the headings before are pushed on
    /// `heading_log` here and put back by `filter_headings_back`, which every caller of
    /// `play_for_filter` asks after its restore. A taken piece's bit is dropped first, as
    /// `heading_tidy` drops it, without the key.
    #[cold]
    #[inline(never)]
    pub(super) fn npc_phase_for_filter_lines(&mut self, saved: &mut Saved) {
        self.heading_log.push(self.heading_back);
        let standing = self.heading_back;
        for square in standing.iter() {
            if !self.t.is_wild_piece(self.board[square]) {
                self.heading_back.clear(square);
            }
        }
        self.line_phase(&mut Filter { saved });
    }

    /// The headings `npc_phase_for_filter_lines` pushed, put back: asked after a filter's
    /// restore wherever `play_for_filter` played the phase (`has_npcs`). Nothing unless the
    /// phase was a line game's (no pounce: `npc_phase_for_filter`'s routing).
    #[cold]
    #[inline(never)]
    pub(super) fn filter_headings_back(&mut self) {
        if self.t.npc_lines.is_some() && self.t.pounce.is_none() {
            self.heading_back = self.heading_log.pop().expect("filter headings not pushed");
        }
    }

    /// Generation 125 (SR8h, N5, `docs/plan-schema-next.md` §2.8, ruled 2026-10-27): whether a
    /// turning-back piece threatens `square` -- **the square it steps to, read after its turn**:
    /// the square ahead by its heading, or, where that holds a piece it may not take, the one
    /// behind after it turns round. The line's second attack source, beside the cyclic routes'
    /// table sources (`compile._build_attack_sources` files none for a line square).
    ///
    /// Read as the phase would play on this board now: every line piece stepped by
    /// `turn_step_on`, the one statement of the rule, in the phase's own order (`LineOrder`
    /// where a line starts with two pieces, else `npc_step_order`), on a copy of the board, so
    /// a queue ahead that walks away first is no block and a piece that turned at a block
    /// threatens behind it. A cyclic piece never walks a line square (routes are disjoint), so
    /// it is skipped. `victim` stands a piece of that type on `square` first (castling's
    /// transit squares, `threatened_as`). A frozen piece threatens nothing (it waits); a
    /// disarmed one nothing either (it eats nothing), as `wild_attacks_free` reads a cyclic one.
    /// Cold: only a game with a line asks (`threatened`, behind `has_npcs`).
    #[cold]
    #[inline(never)]
    pub(super) fn line_threatens(&self, square: usize, victim: Option<usize>) -> bool {
        let t = self.t;
        let Some(lines) = t.npc_lines.as_deref() else {
            return false;
        };
        let mut board = self.board;
        if let Some(victim) = victim {
            if !t.is_wild_piece(board[square]) {
                board[square] = t.piece_value(0, 0, victim);
            }
        }
        let sight = self.wild_sight();
        let here = self.npc_here();
        let mut heading = SquareSet::EMPTY;
        for from in here.iter() {
            if lines.squares.get(from) && self.heading_back.get(from) {
                heading.set(from);
            }
        }
        if lines.queues {
            let order = LineOrder::new(t, lines, here, heading);
            self.line_threat_walk(&mut board, &mut heading, order, &sight, square)
        } else {
            let order = self.npc_step_order();
            self.line_threat_walk(&mut board, &mut heading, order, &sight, square)
        }
    }

    /// `line_threatens`' walk in a given order, on its copy of the board and the headings.
    #[inline(always)]
    fn line_threat_walk<I: Iterator<Item = usize>>(
        &self, board: &mut [u8; MAX_SQUARES], heading: &mut SquareSet, order: I, sight: &WildSight,
        square: usize,
    ) -> bool {
        let lines = self.t.npc_lines.as_deref().expect("line_threatens without a line");
        for from in order {
            if !lines.squares.get(from) {
                continue;
            }
            let step = self.turn_step_on(board, from, heading.get(from), sight);
            heading.clear(from);
            if step.back {
                heading.set(step.to);
            } else {
                heading.clear(step.to);
            }
            if step.to != from {
                if step.to == square {
                    return !sight.disarmed.get(from);
                }
                board[step.to] = board[from];
                board[from] = 0;
            }
        }
        false
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

/// The wild pieces of one phase in a game with a line, in stepping order (generation 123,
/// SR8f, N4: `docs/plan-schema-next.md` §2.8, ruled 2026-10-27).
///
/// `NpcOrder`'s front first, read by heading: a piece whose square **ahead of it** (the next
/// square walking forward, the previous walking back; a cyclic route's next square) holds a
/// wild piece still to step goes after that piece, so a queue walking either way moves as one
/// when its front moves. **Head-on**, two pieces each ahead of the other, the one nearer the
/// route's first square steps first: on a line that is the one walking forward, whose square
/// ahead is the other's. With one wild piece per line (every game before generation 123) no
/// line piece ever waits on another, and a cyclic piece waits as `NpcOrder` has it, so the
/// order is `NpcOrder`'s. A line has no cycle but a head-on pair, so every pass that defers
/// also steps; a full cyclic route still waits whole.
pub(super) struct LineOrder<'t> {
    t: &'t Tables,
    lines: &'t LineTables,
    /// The pending pieces walking their line back, read once at the phase's start.
    back: SquareSet,
    pending: SquareSet,
    pass: SquareIter,
    deferred: bool,
    stepped: bool,
}

impl<'t> LineOrder<'t> {
    #[inline]
    pub(super) fn new(t: &'t Tables, lines: &'t LineTables, here: SquareSet, back: SquareSet) -> Self {
        Self { t, lines, back, pending: here, pass: here.iter(), deferred: false, stepped: false }
    }

    /// The square ahead of the piece on `square`, by its heading on a line, else its route's
    /// next; `None` at a line's end or on a one-square route.
    #[inline]
    fn ahead(&self, square: usize) -> Option<usize> {
        if self.lines.squares.get(square) {
            self.lines.ahead(&self.t.npc_next, square, self.back.get(square))
        } else {
            let next = self.t.npc_next[square];
            (next >= 0 && next as usize != square).then_some(next as usize)
        }
    }

    /// Whether the piece on `square` waits this pass for a pending piece ahead of it.
    #[inline]
    fn waits(&self, square: usize) -> bool {
        let Some(ahead) = self.ahead(square) else {
            return false;
        };
        if !self.pending.get(ahead) {
            return false;
        }
        // Head-on: the piece walking forward is the nearer the route's first square, and goes.
        let head_on = self.ahead(ahead) == Some(square);
        !(head_on && self.lines.squares.get(square) && !self.back.get(square))
    }
}

impl Iterator for LineOrder<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if let Some(square) = self.pass.next() {
                if !self.pending.get(square) {
                    continue;
                }
                if self.waits(square) {
                    self.deferred = true;
                    continue;
                }
                self.pending.clear(square);
                self.stepped = true;
                return Some(square);
            }
            if !self.deferred || !self.stepped {
                return None;
            }
            self.pass = self.pending.iter();
            self.deferred = false;
            self.stepped = false;
        }
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
            // Generation 42: the duck carries no heading.
            if t.is_wild_piece(board[square]) {
                set.set(square);
            }
        }
    }
    set
}
