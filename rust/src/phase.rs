//! The wild-piece phase walked once (R6, CS10a, `docs/plan-smells-rust.md`): the steps and the
//! returns, generic over where a step's writes go.
//!
//! The phase is played three ways, and before this each had its own copy of the walk: for
//! real by `make` (every square through `write`, logged for `unmake`), on the raw board by the
//! legality filter (every square pushed to `Saved`), and on the raw board by `coming_phases`
//! (pushed to `Saved` and listed for the board's drawing). They differ only in what a step
//! does to the state, so that is the `PhaseSink` trait, and the walk (`wild_steps`,
//! `respawn_walk`, `wild_phase`) is written once. Every method is `#[inline(always)]` and the
//! walk is generic over the sink, so each caller monomorphises to the body it ran before.
//!
//! The rules themselves stay where they were: a step is `npc_step_of`, a return
//! `respawn_due`, and the order `npc_step_order`. A game with a line (`turning.rs::line_steps`)
//! and a pounce game (`pounce.rs::pounce_walk`) walk their own steps, each once (CS10b), into
//! the same sinks: a turn's heading, a turned wait and a split's copy are sink methods too.

use super::*;
use super::turning::TurnStep;

/// Where one phase's writes go: `Live` (`make`), `Filter` (a legality filter) and `Coming`
/// (`coming_phases`).
pub(super) trait PhaseSink {
    /// Whether a pounce walk's working wild set follows every move, not only a split's: so
    /// for `Coming`, whose next phase starts from it, and for no other sink.
    const TRACKS: bool;
    /// A wild piece walks from `from` to `to`, eating what stands there: a `PHASE_STEP`, a
    /// `PHASE_TURN` (a line's step after turning round) or a `PHASE_POUNCE`.
    fn step(&mut self, game: &mut Game<'_>, from: usize, to: usize, kind: u8);
    /// A copy of the splitting pouncer on `parent` grows on `to`, eating what stands there:
    /// a `PHASE_COPY`.
    fn copy(&mut self, game: &mut Game<'_>, parent: usize, to: usize);
    /// Whether the line piece on `square` walks back, as this sink holds the headings.
    fn heading(&self, game: &Game<'_>, square: usize) -> bool;
    /// The line piece on `from` took `step`: its heading moves to `step.to` and becomes
    /// `step.back`. Before the step's own board write, if it moves.
    fn turn(&mut self, game: &mut Game<'_>, from: usize, step: TurnStep);
    /// The line piece on `square` turned round and waits there: a `PHASE_TURN` wait.
    fn wait_turned(&mut self, game: &mut Game<'_>, square: usize);
    /// Respawning half `index`'s clock before this phase.
    fn clock(&self, game: &Game<'_>, index: usize) -> u8;
    /// Half `index`'s clock moves from `before` to `after` over this phase.
    fn set_clock(
        &mut self, game: &mut Game<'_>, rule: &RespawnTables, index: usize, before: u8, after: u8,
    );
    /// A wild piece `value` comes back onto `start`, eating what stands there.
    fn put_back(&mut self, game: &mut Game<'_>, start: usize, value: u8);
}

/// The phase played for real: every square through `write` so the key and the occupancy
/// sets follow, every step on `npc_log` for `unmake`, every clock change into the working
/// key. `ate` is whether anything was eaten, the material change the clocks read.
pub(super) struct Live {
    pub(super) ate: bool,
}

impl Live {
    #[inline(always)]
    pub(super) fn new() -> Self {
        Self { ate: false }
    }
}

impl PhaseSink for Live {
    const TRACKS: bool = false;

    #[inline(always)]
    fn step(&mut self, game: &mut Game<'_>, from: usize, to: usize, kind: u8) {
        let value = game.board[from];
        let eaten = game.board[to];
        game.write(from, 0);
        game.write(to, value);
        // A square a wild piece has walked over is a square something has happened on.
        // Wild squares carry no `unmoved` bit at ply zero, so the only bit this ever clears
        // is a piece's -- and that piece has just been eaten, which is a material change and
        // irreversible on its own account.
        game.unmoved.clear(from);
        game.unmoved.clear(to);
        self.ate |= eaten != 0;
        game.close_ep_window_over(from, to);
        game.npc_log.push((from, to, eaten, kind));
    }

    #[inline(always)]
    fn copy(&mut self, game: &mut Game<'_>, parent: usize, to: usize) {
        let value = game.board[parent];
        let eaten = game.board[to];
        game.write(to, value);
        game.unmoved.clear(to);
        self.ate |= eaten != 0;
        game.close_ep_window_over(to, to);
        game.npc_log.push((parent, to, eaten, PHASE_COPY));
    }

    #[inline(always)]
    fn heading(&self, game: &Game<'_>, square: usize) -> bool {
        game.heading_back.get(square)
    }

    #[inline(always)]
    fn turn(&mut self, game: &mut Game<'_>, from: usize, step: TurnStep) {
        game.set_heading(from, false);
        game.set_heading(step.to, step.back);
    }

    #[inline(always)]
    fn wait_turned(&mut self, game: &mut Game<'_>, square: usize) {
        game.npc_log.push((square, square, 0, PHASE_TURN));
    }

    #[inline(always)]
    fn clock(&self, game: &Game<'_>, index: usize) -> u8 {
        game.respawn[index]
    }

    #[inline(always)]
    fn set_clock(
        &mut self, game: &mut Game<'_>, rule: &RespawnTables, index: usize, before: u8, after: u8,
    ) {
        if after != before {
            game.working_key ^= rule.key(index, before) ^ rule.key(index, after);
            game.respawn[index] = after;
        }
    }

    #[inline(always)]
    fn put_back(&mut self, game: &mut Game<'_>, start: usize, value: u8) {
        let eaten = game.board[start];
        game.write(start, value);
        game.unmoved.clear(start);
        self.ate |= eaten != 0;
        game.close_ep_window_over(start, start);
        game.npc_log.push((start, start, eaten, PHASE_RETURN));
    }
}

/// The phase on the raw board for a legality filter, every square written pushed to `saved`
/// so it can be put back. The clocks are read, never written: the filter asks about one
/// position.
pub(super) struct Filter<'a> {
    pub(super) saved: &'a mut Saved,
}

impl PhaseSink for Filter<'_> {
    const TRACKS: bool = false;

    #[inline(always)]
    fn step(&mut self, game: &mut Game<'_>, from: usize, to: usize, _kind: u8) {
        let value = game.board[from];
        self.saved.push((from, value));
        self.saved.push((to, game.board[to]));
        game.board[from] = 0;
        game.board[to] = value;
    }

    #[inline(always)]
    fn copy(&mut self, game: &mut Game<'_>, parent: usize, to: usize) {
        self.saved.push((to, game.board[to]));
        game.board[to] = game.board[parent];
    }

    /// The headings are read, never written: each piece reads its own square's bit before
    /// anything moves it, and the filter asks about one position.
    #[inline(always)]
    fn heading(&self, game: &Game<'_>, square: usize) -> bool {
        game.heading_back.get(square)
    }

    #[inline(always)]
    fn turn(&mut self, _game: &mut Game<'_>, _from: usize, _step: TurnStep) {}

    #[inline(always)]
    fn wait_turned(&mut self, _game: &mut Game<'_>, _square: usize) {}

    #[inline(always)]
    fn clock(&self, game: &Game<'_>, index: usize) -> u8 {
        game.respawn[index]
    }

    #[inline(always)]
    fn set_clock(
        &mut self, _game: &mut Game<'_>, _rule: &RespawnTables, _index: usize, _before: u8,
        _after: u8,
    ) {
    }

    #[inline(always)]
    fn put_back(&mut self, game: &mut Game<'_>, start: usize, value: u8) {
        self.saved.push((start, game.board[start]));
        game.board[start] = value;
    }
}

/// One phase for `coming_phases`: the raw board as the filter plays it, the clocks and the
/// headings moved on copies (`clocks`, `heading`), and every step, copy, turned wait and
/// return listed in `steps`.
pub(super) struct Coming<'a> {
    pub(super) saved: &'a mut Saved,
    pub(super) steps: &'a mut Vec<(usize, usize, u8, u8)>,
    pub(super) clocks: &'a mut [u8],
    pub(super) heading: &'a mut SquareSet,
}

impl PhaseSink for Coming<'_> {
    const TRACKS: bool = true;

    #[inline(always)]
    fn step(&mut self, game: &mut Game<'_>, from: usize, to: usize, kind: u8) {
        let value = game.board[from];
        let eaten = game.board[to];
        self.saved.push((from, value));
        self.saved.push((to, eaten));
        game.board[from] = 0;
        game.board[to] = value;
        self.steps.push((from, to, eaten, kind));
    }

    #[inline(always)]
    fn copy(&mut self, game: &mut Game<'_>, parent: usize, to: usize) {
        let eaten = game.board[to];
        self.steps.push((parent, to, eaten, PHASE_COPY));
        self.saved.push((to, eaten));
        game.board[to] = game.board[parent];
    }

    #[inline(always)]
    fn heading(&self, _game: &Game<'_>, square: usize) -> bool {
        self.heading.get(square)
    }

    #[inline(always)]
    fn turn(&mut self, _game: &mut Game<'_>, from: usize, step: TurnStep) {
        self.heading.clear(from);
        if step.back {
            self.heading.set(step.to);
        }
    }

    #[inline(always)]
    fn wait_turned(&mut self, _game: &mut Game<'_>, square: usize) {
        self.steps.push((square, square, 0, PHASE_TURN));
    }

    #[inline(always)]
    fn clock(&self, _game: &Game<'_>, index: usize) -> u8 {
        self.clocks[index]
    }

    #[inline(always)]
    fn set_clock(
        &mut self, _game: &mut Game<'_>, _rule: &RespawnTables, index: usize, _before: u8,
        after: u8,
    ) {
        self.clocks[index] = after;
    }

    #[inline(always)]
    fn put_back(&mut self, game: &mut Game<'_>, start: usize, value: u8) {
        let eaten = game.board[start];
        self.saved.push((start, eaten));
        game.board[start] = value;
        self.steps.push((start, start, eaten, PHASE_RETURN));
    }
}

impl<'t> Game<'t> {
    /// Step 2 of the phase: every wild piece in `order` that `npc_step_of` lets walk,
    /// walked into `sink`, on the board as the earlier steps of the same phase leave it.
    #[inline(always)]
    pub(super) fn wild_steps<S: PhaseSink, I: Iterator<Item = usize>, const SHIELDS: bool>(
        &mut self, sink: &mut S, order: I, shields: &SquareSet,
    ) {
        for square in order {
            let Some(next) = self.npc_step_of::<SHIELDS>(square, shields) else {
                continue;
            };
            sink.step(self, square, next, PHASE_STEP);
        }
    }

    /// Step 3 of the phase (generation 36, N1), after every step: each respawning half's
    /// clock moved on by `respawn_due`, and its piece put back where the rule says. Nothing
    /// in a game without the rule.
    #[inline(always)]
    pub(super) fn respawn_walk<S: PhaseSink>(&mut self, sink: &mut S, shields: &SquareSet) {
        let t = self.t;
        let Some(rule) = t.respawn.as_deref() else {
            return;
        };
        for index in 0..rule.halves.len() {
            let before = sink.clock(self, index);
            let (after, start) = self.respawn_due(index, before, shields);
            sink.set_clock(self, rule, index, before, after);
            if let Some(start) = start {
                sink.put_back(self, start, rule.halves[index].value);
            }
        }
    }

    /// The whole phase of a game with neither a line nor a pounce: the steps, then the
    /// returns. `SHIELDS` is `Tables::npc_shield` as a constant (the shields read once, at
    /// the phase's start); the order is `npc_step_order` only under `Tables::npc_queues`,
    /// and the ascending walk of `npc_here` otherwise (Q850: walking `NpcOrder` everywhere
    /// cost *wolves*' perft 10%), each its own monomorph.
    #[inline(always)]
    pub(super) fn wild_phase<S: PhaseSink, const SHIELDS: bool>(&mut self, sink: &mut S) {
        let shields = if SHIELDS { self.wild_shields() } else { SquareSet::EMPTY };
        if self.t.npc_queues {
            let order = self.npc_step_order();
            self.wild_steps::<S, _, SHIELDS>(sink, order, &shields);
        } else {
            let order = self.npc_here().iter();
            self.wild_steps::<S, _, SHIELDS>(sink, order, &shields);
        }
        self.respawn_walk(sink, &shields);
    }
}
