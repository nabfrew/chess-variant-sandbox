//! Wild pieces that pounce (generation 39, N3b, `docs/archive/plan-npc.md` §3.1 to §3.3, §4.3): step 1
//! of the wild-piece phase, and the pass test that is check in a pounce game.
//!
//! **The rule, stated once** (`Game::pounce_prey_on`). After a move, a `"prey"` pouncer looks
//! at the pieces **of the player who just moved** that it attacks by its own type's capture
//! moves from where it stands (blocked as they are blocked, the capture matrix applying, a
//! shielded piece never prey). If there is one, it moves onto the first in the **prey order**
//! and eats it, and that is its action for the phase: a royal first, then the higher declared
//! value, then the nearer (king's distance), then the mover's reading order (the square
//! nearest the mover's own first rank, then the one furthest to its left). The core folds the
//! four into one integer (`prey_key`) and takes the least; the last term is unique per
//! square, so no tie remains, and every term is the same for both seats or read from the
//! mover's, so mirrored positions choose mirrored victims.
//!
//! **The phase** (§3.1): the wild pieces standing are listed once, from the third owner's
//! square set (`occupied[WILD]`, read as a hint and checked against the board, since inside a
//! legality filter it is the board's before the move and a move can take a wild piece off but
//! never put one on). The pouncers pounce first, in the mover's reading order of their
//! squares; then every wild piece that did not pounce steps (`NpcOrder` over them, front
//! first), walking a route only where the square's route is its own type's (`PounceTables::
//! piece`, §3.3): a pouncer that left its loop stands until it pounces back onto it. The
//! shields are read once, at the phase's start. Respawn and turning back are refused beside a
//! pounce, so the phase has no step 3 and no line.
//!
//! **Pounce on attackers** (generation 40, N3c, §4.4; `Game::pounce_attackers_on`). An
//! `"attackers"` pouncer looks instead at the pieces of the player who just moved that attack
//! the square it stands on (their own capture moves, through the players' own inverse table,
//! the capture matrix both ways: an attacker that may not take it, or that it may not take,
//! is ignored and safe), less the shielded, frozen and disarmed ones of the phase's snapshot,
//! and takes the least `prey_key` among them, as a `"prey"` pouncer does. A piece the mover
//! commanded is its owner's, so never one.
//!
//! **Check** (§4.3, the pass test): the side to move is in check when a player's piece attacks
//! its royal, or the phase, played on the board as it stands with the side to move as the
//! mover, would take it (`pass_takes_royal`). Asked only of the side to move; the other player
//! and every legality filter read the tables as today, because no pounce ever takes a piece of
//! the player who did not just move.

use super::*;

/// The phase's effect snapshot (§3.1), read once at its start: the players' pieces under a
/// shield, which no wild piece takes, and the mover's frozen or disarmed pieces, which attack
/// nothing (generation 40: read only beside a restraint and an `"attackers"` pouncer).
pub(super) struct PhaseEffects {
    pub(super) shields: SquareSet,
    restrained: SquareSet,
}

/// The pouncers of one phase, in the mover's reading order: inline up to 64, spilling past.
struct Pouncers {
    inline: [u16; 64],
    len: usize,
    spill: Vec<u16>,
}

impl Pouncers {
    fn as_slice(&self) -> &[u16] {
        if self.spill.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }
}

/// A splitting pouncer's attackers in prey order, `(prey_key, square)`: inline up to 16,
/// spilling past.
struct Attackers {
    inline: [(u32, u16); 16],
    len: usize,
    spill: Vec<(u32, u16)>,
}

impl Attackers {
    fn push(&mut self, key: u32, square: u16) {
        if self.as_slice().iter().any(|&(_, kept)| kept == square) {
            return;
        }
        if self.spill.is_empty() && self.len < self.inline.len() {
            self.inline[self.len] = (key, square);
            self.len += 1;
        } else {
            if self.spill.is_empty() {
                self.spill.extend_from_slice(&self.inline);
            }
            self.spill.push((key, square));
        }
    }

    fn as_slice(&self) -> &[(u32, u16)] {
        if self.spill.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }

    /// Insertion sort by key; the key is unique per square, so the order is total.
    fn sort(&mut self) {
        let len = self.len;
        let squares: &mut [(u32, u16)] = if self.spill.is_empty() { &mut self.inline[..len] } else { &mut self.spill };
        for i in 1..squares.len() {
            let mut j = i;
            while j > 0 && squares[j - 1].0 > squares[j].0 {
                squares.swap(j - 1, j);
                j -= 1;
            }
        }
    }
}

/// What one splitting pouncer does (§4.5): copies onto the first `copies` of `attackers`, then
/// the original onto `lunge`, if any.
struct Split {
    attackers: Attackers,
    copies: usize,
    lunge: Option<usize>,
}

impl Split {
    fn copy_squares(&self) -> impl Iterator<Item = usize> + '_ {
        self.attackers.as_slice()[..self.copies].iter().map(|&(_, origin)| origin as usize)
    }

    fn acted(&self) -> bool {
        self.copies > 0 || self.lunge.is_some()
    }
}

impl<'t> Game<'t> {
    #[inline]
    fn pounce_tables(&self) -> &'t PounceTables {
        self.t.pounce.as_deref().expect("a pounce phase without the pounce tables")
    }

    /// The wild pieces standing at the phase's start: the third owner's set as a hint, each
    /// square checked against the board (sound inside a filter: a move takes a wild piece off
    /// a square, never puts one on).
    #[inline]
    fn wild_standing(&self) -> SquareSet {
        let t = self.t;
        let mut standing = self.occupied[WILD];
        for square in self.occupied[WILD].iter() {
            // Generation 42: the duck is in the third owner's set too, and never a wild piece.
            if !t.is_wild_piece(self.board[square]) {
                standing.clear(square);
            }
        }
        standing
    }

    /// The phase's effect snapshot (§3.1) for `mover`: every player's piece under a shield,
    /// which no wild piece takes, and, beside an `"attackers"` pouncer and a restraint, the
    /// mover's frozen or disarmed pieces, which attack nothing. Each empty, at one flag test,
    /// in a game without its rule.
    #[inline(never)]
    fn pounce_effects(&self, mover: usize) -> PhaseEffects {
        let t = self.t;
        let mut effects = PhaseEffects { shields: SquareSet::EMPTY, restrained: SquareSet::EMPTY };
        if t.npc_shield {
            for square in 0..t.n_squares {
                if self.shielded_from_wild(square) {
                    effects.shields.set(square);
                }
            }
        }
        if let Some(sight) = t.sight.as_deref() {
            if sight.has_restraint && self.pounce_tables().attackers {
                for square in 0..t.n_squares {
                    if self.effects_at(sight, mover, square) & (SIGHT_FREEZE | SIGHT_DISARM) != 0 {
                        effects.restrained.set(square);
                    }
                }
            }
        }
        effects
    }

    /// The pouncers among `standing`, sorted by `mover`'s reading order of their squares.
    fn pouncers(&self, standing: SquareSet, mover: usize) -> Pouncers {
        let t = self.t;
        let p = self.pounce_tables();
        let mut list = Pouncers { inline: [0; 64], len: 0, spill: Vec::new() };
        for square in standing.iter() {
            if !p.pounces(t.type_of(self.board[square])) {
                continue;
            }
            if list.len < list.inline.len() {
                list.inline[list.len] = square as u16;
                list.len += 1;
            } else {
                if list.spill.is_empty() {
                    list.spill.extend_from_slice(&list.inline);
                }
                list.spill.push(square as u16);
            }
        }
        let reading = &p.reading[mover * p.n_squares..(mover + 1) * p.n_squares];
        let sort = |squares: &mut [u16]| {
            for i in 1..squares.len() {
                let mut j = i;
                while j > 0 && reading[squares[j - 1] as usize] > reading[squares[j] as usize] {
                    squares.swap(j - 1, j);
                    j -= 1;
                }
            }
        };
        if list.spill.is_empty() {
            let len = list.len;
            sort(&mut list.inline[..len]);
        } else {
            sort(&mut list.spill);
        }
        list
    }

    /// Whether `entry`'s path is open on `board`: the prefix empty and the window holding
    /// exactly the screens it expects, of anybody's (a hop whose screen names an owner is
    /// refused beside a pounce).
    #[inline]
    fn path_open(board: &[u8; MAX_SQUARES], p: &PounceTables, entry: &ReachEntry) -> bool {
        let path = p.path(entry);
        let (prefix, window) = path.split_at(entry.prefix_len as usize);
        prefix.iter().all(|s| board[*s as usize] == 0)
            && window.iter().filter(|s| board[**s as usize] != 0).count() == entry.screens as usize
    }

    /// **The rule, stated once**: the square of the piece the `"prey"` pouncer on `square`
    /// takes on `board`, or `None`. Reads the board it is given, never `self.board`, so `make`,
    /// the filter, `coming_phases` and the pass test (on its copy) play the one rule.
    pub(super) fn pounce_prey_on(
        &self, board: &[u8; MAX_SQUARES], square: usize, mover: usize, shields: &SquareSet,
    ) -> Option<usize> {
        let t = self.t;
        let p = self.pounce_tables();
        let value = board[square];
        debug_assert!(value != 0 && t.owner_of(value) == WILD);
        let ty = t.type_of(value);
        let reading = &p.reading[mover * p.n_squares..(mover + 1) * p.n_squares];
        let mut best: Option<(u32, usize)> = None;
        for entry in p.targets_from(ty, square) {
            let target = entry.target as usize;
            let victim = board[target];
            if victim == 0 || t.owner_of(victim) != mover {
                continue;
            }
            let victim_type = t.type_of(victim);
            if !t.may_capture(ty, victim_type) || shields.get(target) || !Self::path_open(board, p, entry) {
                continue;
            }
            let key = Self::prey_key(t, p, reading, square, target, victim_type);
            if best.is_none_or(|(least, _)| key < least) {
                best = Some((key, target));
            }
        }
        best.map(|(_, target)| target)
    }

    /// The **prey order** folded into one integer, least first (§3.2): a royal, then the
    /// higher declared value, then the nearer to the pouncer on `square`, then the mover's
    /// reading order of `target`, which is unique per square, so no tie remains.
    #[inline]
    fn prey_key(
        t: &Tables, p: &PounceTables, reading: &[u16], square: usize, target: usize, victim_type: usize,
    ) -> u32 {
        (u32::from(!t.is_royal(victim_type)) << 30)
            | (u32::from(p.value_rank[victim_type]) << 20)
            | (u32::from(p.distance[square * p.n_squares + target]) << 10)
            | u32::from(reading[target])
    }

    /// **The rule, stated once** (generation 40, N3c, §4.4): the square of the piece the
    /// `"attackers"` pouncer on `square` takes on `board`, or `None`: the least `prey_key` of
    /// `attackers_kept_on`.
    #[inline(never)]
    pub(super) fn pounce_attackers_on(
        &self, board: &[u8; MAX_SQUARES], square: usize, mover: usize, effects: &PhaseEffects,
    ) -> Option<usize> {
        let mut best: Option<(u32, usize)> = None;
        self.attackers_kept_on(board, square, mover, effects, |key, origin| {
            if best.is_none_or(|(least, _)| key < least) {
                best = Some((key, origin));
            }
        });
        best.map(|(_, origin)| origin)
    }

    /// The attackers an `"attackers"` pouncer on `square` may take on `board`, each handed to
    /// `keep` with its `prey_key`. The mover's pieces that attack its square, through the
    /// players' own inverse table, each matched against the given board and its path
    /// (`path_clear_on`); kept when its type may capture the pouncer's (what `immune` spells),
    /// the pouncer's may capture it, and it is not in the snapshot's shields or restraints.
    /// Only the mover's own pieces are walked, so one the mover commanded (its owner's) is
    /// never one. A sweep and a portal are refused beside every wild piece, so neither is
    /// asked.
    #[inline(always)]
    fn attackers_kept_on(
        &self, board: &[u8; MAX_SQUARES], square: usize, mover: usize, effects: &PhaseEffects,
        mut keep: impl FnMut(u32, usize),
    ) {
        let t = self.t;
        let p = self.pounce_tables();
        let value = board[square];
        debug_assert!(value != 0 && t.owner_of(value) == WILD);
        let ty = t.type_of(value);
        let reading = &p.reading[mover * p.n_squares..(mover + 1) * p.n_squares];
        for source in t.attackers_of(mover, square) {
            let origin = source.origin as usize;
            if board[origin] != source.piece_value {
                continue;
            }
            debug_assert_eq!(source.sweep, 0, "a sweep beside a wild piece");
            if source.requires_unmoved && !self.unmoved.get(origin) {
                continue;
            }
            let attacker = t.type_of(source.piece_value);
            if !t.may_capture(attacker, ty)
                || !t.may_capture(ty, attacker)
                || effects.shields.get(origin)
                || effects.restrained.get(origin)
                || !self.path_clear_on(board, mover, source)
            {
                continue;
            }
            keep(Self::prey_key(t, p, reading, square, origin, attacker), origin);
        }
    }

    /// `attackers_kept_on` as a list in prey order (§4.4, for §4.5's split), each square once
    /// (two sources of one piece onto the square are one attacker).
    fn attacker_squares_on(
        &self, board: &[u8; MAX_SQUARES], square: usize, mover: usize, effects: &PhaseEffects,
    ) -> Attackers {
        let mut list = Attackers { inline: [(0, 0); 16], len: 0, spill: Vec::new() };
        self.attackers_kept_on(board, square, mover, effects, |key, origin| list.push(key, origin as u16));
        list.sort();
        list
    }

    /// **Split, stated once** (generation 41, N3d, §4.5): what the splitting pouncer on
    /// `square` does on `board` with `room` left under its type's cap. Its attackers in prey
    /// order: a copy onto each of the first `room`, then, if one remains, the original onto
    /// it; the rest are spared. Nothing when no attacker is there. The walks apply it, each
    /// in its own way of writing a square, so `make`, the filter, `coming_phases` and the pass
    /// test play the one rule.
    fn pounce_split_on(
        &self, board: &[u8; MAX_SQUARES], square: usize, mover: usize, effects: &PhaseEffects, room: usize,
    ) -> Split {
        let attackers = self.attacker_squares_on(board, square, mover, effects);
        let all = attackers.as_slice();
        let copies = room.min(all.len());
        let lunge = all.get(copies).map(|&(_, origin)| origin as usize);
        Split { attackers, copies, lunge }
    }

    /// The room left under the cap of the splitting type `ty`: the cap less the wild pieces
    /// of the type in `live`, the phase's working set (§4.5: counted live and by type, so two
    /// pouncers of one type share it, and copies made earlier in the phase count), never
    /// below 0.
    #[inline]
    fn split_room(&self, board: &[u8; MAX_SQUARES], live: SquareSet, ty: usize, cap: usize) -> usize {
        let t = self.t;
        let standing = live.iter().filter(|&square| t.type_of(board[square]) == ty).count();
        cap.saturating_sub(standing)
    }

    /// What the pouncer on `square` takes on `board`, by its type's kind: the one place the
    /// phase walks and the pass test ask.
    #[inline]
    fn pounce_on(
        &self, board: &[u8; MAX_SQUARES], square: usize, mover: usize, effects: &PhaseEffects,
    ) -> Option<usize> {
        let p = self.pounce_tables();
        match p.kind[self.t.type_of(board[square])] {
            POUNCE_PREY => self.pounce_prey_on(board, square, mover, &effects.shields),
            POUNCE_ATTACKERS => self.pounce_attackers_on(board, square, mover, effects),
            _ => None,
        }
    }

    /// The step of the wild piece on `square` in a pounce game, on `board`: `npc_step_of`'s
    /// rule with the route-type wait (§3.3), a piece standing on a square whose route is not
    /// its own type's waiting there (a pouncer off its loop stands).
    #[inline]
    fn pounce_step_on(&self, board: &[u8; MAX_SQUARES], square: usize, shields: &SquareSet) -> Option<usize> {
        let t = self.t;
        let p = self.pounce_tables();
        let value = board[square];
        let next = t.npc_next[square];
        if next < 0 || next as usize == square || p.piece[square] as isize != t.type_of(value) as isize {
            return None;
        }
        let next = next as usize;
        let standing = board[next];
        if standing == 0 {
            return Some(next);
        }
        if t.owner_of(standing) == WILD || shields.get(next) {
            return None;
        }
        t.may_capture(t.type_of(value), t.type_of(standing)).then_some(next)
    }

    /// The phase, played for real, in a pounce game: the pounces, then the steps, every square
    /// through `write`, each logged. The mover is the side that has just moved (`make` flips
    /// the side after the phase). Returns whether anything was eaten.
    #[cold]
    #[inline(never)]
    pub(super) fn pounce_phase(&mut self) -> bool {
        let mover = self.side;
        let effects = self.pounce_effects(mover);
        let standing = self.wild_standing();
        let mut steppers = standing;
        let mut live = standing;
        let splits = !self.pounce_tables().split.is_empty();
        let mut ate = false;
        for &square in self.pouncers(standing, mover).as_slice() {
            let square = square as usize;
            if let Some(split) = self.split_here_if(splits, &self.board, square, mover, &effects, live) {
                if !split.acted() {
                    continue;
                }
                steppers.clear(square);
                for copy in split.copy_squares() {
                    ate |= self.wild_copies(square, copy);
                    live.set(copy);
                }
                if let Some(victim) = split.lunge {
                    ate |= self.wild_moves(square, victim, PHASE_POUNCE);
                    live.clear(square);
                    live.set(victim);
                }
                continue;
            }
            let Some(victim) = self.pounce_on(&self.board, square, mover, &effects) else {
                continue;
            };
            steppers.clear(square);
            ate |= self.wild_moves(square, victim, PHASE_POUNCE);
            if splits {
                live.clear(square);
                live.set(victim);
            }
        }
        for square in NpcOrder::new(self.t, steppers) {
            if let Some(next) = self.pounce_step_on(&self.board, square, &effects.shields) {
                ate |= self.wild_moves(square, next, PHASE_STEP);
            }
        }
        ate
    }

    /// One wild piece moving from `square` onto `to` for real, eating what stands there:
    /// through `write`, the `unmoved` bits and the en passant window as a step's, logged.
    #[inline]
    fn wild_moves(&mut self, square: usize, to: usize, kind: u8) -> bool {
        let value = self.board[square];
        let eaten = self.board[to];
        self.write(square, 0);
        self.write(to, value);
        self.unmoved.clear(square);
        self.unmoved.clear(to);
        self.close_ep_window_over(square, to);
        self.npc_log.push((square, to, eaten, kind));
        eaten != 0
    }

    /// A copy of the splitting pouncer on `parent` growing on `to` for real, eating the
    /// attacker there (§4.5): through `write`, its `unmoved` bit cleared and an en passant
    /// window over it closed, as a step's, logged as `PHASE_COPY`.
    #[inline]
    fn wild_copies(&mut self, parent: usize, to: usize) -> bool {
        let value = self.board[parent];
        let eaten = self.board[to];
        self.write(to, value);
        self.unmoved.clear(to);
        self.close_ep_window_over(to, to);
        self.npc_log.push((parent, to, eaten, PHASE_COPY));
        eaten != 0
    }

    /// `split_here` behind the walk's own `splits` flag (whether any type splits), read once
    /// before its loop, so a pounce game without the rule pays one predictable test.
    #[inline(always)]
    fn split_here_if(
        &self, splits: bool, board: &[u8; MAX_SQUARES], square: usize, mover: usize, effects: &PhaseEffects,
        live: SquareSet,
    ) -> Option<Split> {
        if splits { self.split_here(board, square, mover, effects, live) } else { None }
    }

    /// The split the pouncer on `square` makes, or `None` for a type that does not split
    /// (which pounces as `pounce_on` says). `live` is the phase's working wild set.
    #[inline(never)]
    fn split_here(
        &self, board: &[u8; MAX_SQUARES], square: usize, mover: usize, effects: &PhaseEffects, live: SquareSet,
    ) -> Option<Split> {
        let p = self.pounce_tables();
        if p.split.is_empty() {
            return None;
        }
        let ty = self.t.type_of(board[square]);
        let cap = p.split_cap(ty);
        if cap == 0 {
            return None;
        }
        let room = self.split_room(board, live, ty, cap);
        Some(self.pounce_split_on(board, square, mover, effects, room))
    }

    /// The same phase on the raw board for a legality filter, every square written pushed to
    /// `saved`.
    #[cold]
    #[inline(never)]
    pub(super) fn pounce_phase_for_filter(&mut self, saved: &mut Saved) {
        let mover = self.side;
        let effects = self.pounce_effects(mover);
        let standing = self.wild_standing();
        let mut steppers = standing;
        let mut live = standing;
        let splits = !self.pounce_tables().split.is_empty();
        for &square in self.pouncers(standing, mover).as_slice() {
            let square = square as usize;
            if let Some(split) = self.split_here_if(splits, &self.board, square, mover, &effects, live) {
                if split.acted() {
                    steppers.clear(square);
                }
                for copy in split.copy_squares() {
                    Self::raw_copy(&mut self.board, saved, square, copy);
                    live.set(copy);
                }
                if let Some(victim) = split.lunge {
                    Self::raw_move(&mut self.board, saved, square, victim);
                    live.clear(square);
                    live.set(victim);
                }
                continue;
            }
            if let Some(victim) = self.pounce_on(&self.board, square, mover, &effects) {
                steppers.clear(square);
                Self::raw_move(&mut self.board, saved, square, victim);
                if splits {
                    live.clear(square);
                    live.set(victim);
                }
            }
        }
        for square in NpcOrder::new(self.t, steppers) {
            if let Some(next) = self.pounce_step_on(&self.board, square, &effects.shields) {
                Self::raw_move(&mut self.board, saved, square, next);
            }
        }
    }

    #[inline]
    fn raw_move(board: &mut [u8; MAX_SQUARES], saved: &mut Saved, square: usize, to: usize) {
        let value = board[square];
        saved.push((square, value));
        saved.push((to, board[to]));
        board[square] = 0;
        board[to] = value;
    }

    #[inline]
    fn raw_copy(board: &mut [u8; MAX_SQUARES], saved: &mut Saved, parent: usize, to: usize) {
        saved.push((to, board[to]));
        board[to] = board[parent];
    }

    /// `coming_phases` in a pounce game: `count` phases on the raw board, each with the wild
    /// set it starts from (a copy, moved with each square written) and its shields, the mover
    /// alternating from the side to move.
    pub(super) fn pounce_coming_phases(&mut self, count: usize) -> Vec<Vec<(usize, usize, u8, u8)>> {
        let mut phases = vec![Vec::new(); count];
        let mut saved = Saved::new();
        let mut standing = self.wild_standing();
        let mut mover = self.side;
        let splits = !self.pounce_tables().split.is_empty();
        for steps in phases.iter_mut() {
            let effects = self.pounce_effects(mover);
            let mut steppers = standing;
            for &square in self.pouncers(standing, mover).as_slice() {
                let square = square as usize;
                if let Some(split) = self.split_here_if(splits, &self.board, square, mover, &effects, standing) {
                    if split.acted() {
                        steppers.clear(square);
                    }
                    for copy in split.copy_squares() {
                        steps.push((square, copy, self.board[copy], PHASE_COPY));
                        Self::raw_copy(&mut self.board, &mut saved, square, copy);
                        standing.set(copy);
                    }
                    if let Some(victim) = split.lunge {
                        steps.push((square, victim, self.board[victim], PHASE_POUNCE));
                        Self::raw_move(&mut self.board, &mut saved, square, victim);
                        standing.clear(square);
                        standing.set(victim);
                    }
                    continue;
                }
                if let Some(victim) = self.pounce_on(&self.board, square, mover, &effects) {
                    steps.push((square, victim, self.board[victim], PHASE_POUNCE));
                    Self::raw_move(&mut self.board, &mut saved, square, victim);
                    steppers.clear(square);
                    standing.clear(square);
                    standing.set(victim);
                }
            }
            for square in NpcOrder::new(self.t, steppers) {
                if let Some(next) = self.pounce_step_on(&self.board, square, &effects.shields) {
                    steps.push((square, next, self.board[next], PHASE_STEP));
                    Self::raw_move(&mut self.board, &mut saved, square, next);
                    standing.clear(square);
                    standing.set(next);
                }
            }
            mover = 1 - mover;
        }
        saved.restore(&mut self.board);
        phases
    }

    /// The pass test (§4.3): whether the phase, played on the board as it stands with `mover`
    /// as the mover, would take one of `mover`'s royals. On a copy of the board: the pounces in
    /// order, true at the first that eats a royal; then any wild piece left whose step would
    /// eat one (step 2 cannot save a royal standing on a step's next square, whatever order it
    /// walks). There is no step 3 beside a pounce.
    pub(super) fn pass_takes_royal(&self, mover: usize) -> bool {
        let t = self.t;
        let royal_of_mover = |value: u8| value != 0 && t.owner_of(value) == mover && t.is_royal(t.type_of(value));
        let mut board = self.board;
        let effects = self.pounce_effects(mover);
        let standing = self.wild_standing();
        let mut steppers = standing;
        let mut live = standing;
        let splits = !self.pounce_tables().split.is_empty();
        for &square in self.pouncers(standing, mover).as_slice() {
            let square = square as usize;
            if let Some(split) = self.split_here_if(splits, &board, square, mover, &effects, live) {
                if split.acted() {
                    steppers.clear(square);
                }
                for copy in split.copy_squares() {
                    if royal_of_mover(board[copy]) {
                        return true;
                    }
                    board[copy] = board[square];
                    live.set(copy);
                }
                if let Some(victim) = split.lunge {
                    if royal_of_mover(board[victim]) {
                        return true;
                    }
                    board[victim] = board[square];
                    board[square] = 0;
                    live.clear(square);
                    live.set(victim);
                }
                continue;
            }
            if let Some(victim) = self.pounce_on(&board, square, mover, &effects) {
                if royal_of_mover(board[victim]) {
                    return true;
                }
                board[victim] = board[square];
                board[square] = 0;
                steppers.clear(square);
                if splits {
                    live.clear(square);
                    live.set(victim);
                }
            }
        }
        steppers.iter().any(|square| {
            self.pounce_step_on(&board, square, &effects.shields)
                .is_some_and(|next| royal_of_mover(board[next]))
        })
    }

    /// `compute_in_check` for the side to move in a pounce game: a player's piece attacks its
    /// royal, or the pass test takes it. The wild pieces' step sources are not asked here: the
    /// pass test plays their steps, after the pounces that come first.
    #[cold]
    #[inline(never)]
    pub(super) fn in_check_by_pass(&self, player: usize) -> bool {
        debug_assert_eq!(self.scratch, 0, "the pass test asked inside a filter");
        let attacked = if self.t.multi_royal {
            self.royal_squares(player).iter().any(|&royal| self.attacks_any(1 - player, royal))
        } else {
            let royal = self.royal_square(player);
            royal >= 0 && self.attacks_any(1 - player, royal as usize)
        };
        attacked || self.pass_takes_royal(player)
    }
}
