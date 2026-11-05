//! The one source of foreign moves (TC1, `docs/plan-turncoat.md` §3.3): every move the side to
//! move makes with a piece that is not its own, marked `MOVE_FOREIGN` (bit 30).
//!
//! A piece's **allegiance** (its owner block) and its **control** (who may move it now) are two
//! things; a foreign move is one where they differ for the mover. `push_foreign` is one walk
//! over candidate pieces with three parameters, and two candidate sets fill them:
//!
//! | candidates | frame | may land on |
//! |---|---|---|
//! | the opponent's pieces a command sight covers (Q1007c, `docs/archive/plan-schema-31.md` §3) | the owner's | empty squares; under `captures_own` without disarm, the owner's own non-royal pieces |
//! | nobody's `MAJORITY` pieces the mover holds (TC3, `docs/plan-turncoat.md` §4) | frame 0 | empty squares; the mover's opponent's non-royal pieces its `captures` allow |
//!
//! Both feed the same per-piece walk, `push_foreign_piece`. Every foreign move excludes
//! castling, en passant, drops and promotion, and every one is verified for the mover's royal
//! (`generate_checked`).
//!
//! Out of line and cold: a game with no foreign candidates never calls it, so the copy every
//! other game runs is untouched (`generate_pseudo_legal` tests two pointers).

use super::*;

/// What a foreign piece may land on besides an empty square: the half-open range of board
/// bytes it may take (one player's block), less royals and less what its capture list
/// forbids. `None` lands on empty squares only.
pub(super) type ForeignTakes = Option<(u16, u16)>;

/// What a foreign piece may land on, and whose zone it would promote in: the one statement
/// `push_foreign_piece` reads for its own landings and, SR3g (P10), for the exit of a
/// momentum end and the far end of a fold (`Game::push_transits`, `Game::push_through_as`),
/// so a held piece transits as its holder's piece would and lands only where it may land.
/// `owner` is also whose passage it is (`Game::passage_player`, SR5c2): a commanded piece
/// answers to its owner's `enter` and `carries`, a held one to its holder's.
#[derive(Clone, Copy)]
pub(super) struct ForeignLanding {
    pub takes: ForeignTakes,
    /// Q1077: a held neutral of a `takes_royal` type may take a royal in `takes`.
    pub royals: bool,
    /// The player in whose zone the piece promotes (the owner's frame of its moves).
    pub owner: usize,
}

impl ForeignLanding {
    /// Whether the piece of `piece_type` may end on the non-empty byte `occupant` arriving
    /// with `mode`: a capture, on a capture line, of a non-royal byte in `takes` (a royal
    /// too under `royals`) of a type its capture list allows.
    #[inline(always)]
    pub fn lands(&self, t: &Tables, piece_type: usize, occupant: u16, mode: u8) -> bool {
        match self.takes {
            Some((from, to)) => {
                mode & 2 != 0
                    && from <= occupant
                    && occupant < to
                    && (self.royals || !t.is_royal(t.type_of(occupant as u8)))
                    && may_take(t, piece_type, occupant)
            }
            None => false,
        }
    }

    /// Whether an arrival on `target` would have to promote, which a foreign move never
    /// does, so it is not offered. `leaving` is the origin's answer under `on_leaving`.
    #[inline(always)]
    pub fn must_promote(&self, t: &Tables, piece_type: usize, leaving: bool, target: usize) -> bool {
        t.is_promoting(piece_type)
            && !t.promotion_optional(piece_type)
            && (leaving || t.promotes_on(self.owner, piece_type, target))
    }
}

impl<'t> Game<'t> {
    /// Every foreign move the side to move may make, appended to `out` with `MOVE_FOREIGN`
    /// set. Today the commanded pieces alone (the table in the module's note): each of the
    /// opponent's pieces that a command binds and no freeze does, moved as its owner would
    /// move it (the owner's frame, first-move rights and screens), taking the owner's own
    /// non-royal pieces only under `SIGHT_COMMAND_OWN` without `SIGHT_DISARM`.
    ///
    /// Royals are never commanded (the compiler files no command source under a royal type).
    /// SR5c2 (V1): a commanded piece passes a portal end or a fold as its owner's piece
    /// would. SR6f (V8): a commanded sweeper sweeps as its owner's would, removing only what
    /// the command lets it take (`push_commanded_sweeps`). The shield on a victim is
    /// `sight_filter`'s, which runs over these moves as over every other, but for a sweep's,
    /// which ends the ray in the walk (`Game::commanded_sweeps`, as SR6g's `sweep_shielded`
    /// does for every sweep).
    #[cold]
    #[inline(never)]
    pub(super) fn push_foreign(&self, out: &mut Vec<u32>) {
        self.push_commanded(out);
        self.push_held(out);
    }

    /// Candidate set two (TC3, `docs/plan-turncoat.md` §4.3, §4.4): every piece nobody owns
    /// of a majority type that the side to move holds (`controller_of`), moved in frame 0,
    /// landing on empty squares and on the opponent's non-royal pieces its `captures` allow.
    /// Its byte stays nobody's: moving it does not change its allegiance.
    fn push_held(&self, out: &mut Vec<u32>) {
        let t = self.t;
        if t.majority.is_none() {
            return;
        }
        let player = self.side;
        if let Some(rule) = t.drops.as_ref() {
            if rule.placement && self.hands[player].iter().any(|held| *held > 0) {
                return;
            }
        }
        let low = t.value_low(1 - player);
        let high = low + t.side_stride as u16;
        let rule = t.majority.as_deref().expect("checked above");
        for origin in self.occupied()[WILD].iter() {
            let value = self.board[origin];
            if t.kind_of(value) != CONTROL_MAJORITY || self.controller_of(origin) != player as i8 {
                continue;
            }
            let piece_type = t.type_of(value);
            let royals = rule.takes_royal(piece_type);
            self.push_foreign_piece(origin, piece_type, 0, player, Some((low, high)), royals, out);
        }
    }

    /// Q1077 (`docs/plan-turncoat.md` §4.7): whether a held neutral of a `takes_royal` type
    /// that `by` holds now attacks `square` (as a capture of `victim`, or of whatever stands
    /// there). The compiler files those types' frame-0 attack sources under the third owner's
    /// block with the nobody's byte, so this is `attacks_square_as`'s walk over that block,
    /// each matching source then asked who holds it. Reads the board, never the occupancy
    /// sets, so a legality filter may ask it. Sweeps are refused on a majority type, and
    /// its hop screens name no owner, so the path test reads alike for either holder.
    #[cold]
    #[inline(never)]
    pub(super) fn held_attacks(&self, by: usize, square: usize, victim: Option<usize>) -> bool {
        let t = self.t;
        let victim = victim.unwrap_or_else(|| {
            let occupant = self.board[square];
            if occupant == 0 { usize::MAX } else { t.victim_of(occupant) }
        });
        for source in t.attackers_of(WILD, square) {
            if self.board[source.origin as usize] != source.piece_value {
                continue;
            }
            if t.has_capture_matrix
                && victim != usize::MAX
                && !t.may_capture(t.type_of(source.piece_value), victim)
            {
                continue;
            }
            if self.path_clear_on(&self.board, by, source)
                && self.controller_of(source.origin as usize) == by as i8
            {
                return true;
            }
        }
        false
    }

    /// TC3 (`docs/plan-turncoat.md` §4.3): who may move the piece on `square` as a held
    /// neutral: the player with strictly more of their own pieces on its neighbour squares
    /// (any piece, royals, frozen and resting ones included: presence, not ability), `-1` on
    /// a tie (none against none included), and `-1` for anything but a majority neutral. Read
    /// off the board as it stands, so it needs no state and no key.
    pub fn controller_of(&self, square: usize) -> i8 {
        let t = self.t;
        let value = self.board[square];
        let Some(majority) = t.majority.as_deref() else {
            return -1;
        };
        if t.kind_of(value) != CONTROL_MAJORITY {
            return -1;
        }
        let mut count = [0i32; 2];
        for &next in majority.neighbours(t.n_squares, t.type_of(value), square) {
            let occupant = self.board[next as usize];
            if occupant != 0 {
                let owner = t.owner_of(occupant);
                if owner < 2 {
                    count[owner] += 1;
                }
            }
        }
        match count[0].cmp(&count[1]) {
            std::cmp::Ordering::Greater => 0,
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Equal => -1,
        }
    }

    /// Candidate set one (Q1007c): the opponent's commanded pieces, in the owner's frame.
    fn push_commanded(&self, out: &mut Vec<u32>) {
        let t = self.t;
        let player = self.side;
        // The placement phase: a drop is the only move (as in `generate_pseudo_legal_as`).
        if let Some(rule) = t.drops.as_ref() {
            if rule.placement && self.hands[player].iter().any(|held| *held > 0) {
                return;
            }
        }
        if let Some(sight) = t.sight.as_deref().filter(|sight| sight.has_command) {
            let owner = 1 - player;
            let low = t.value_low(owner);
            let high = low + t.side_stride as u16;
            for origin in self.occupied()[owner].iter() {
                let (piece_type, frame) = t.decode_owned(self.board[origin], low, owner);
                let bits = self.effects_at(sight, owner, origin);
                if bits & SIGHT_COMMAND == 0 || bits & SIGHT_FREEZE != 0 {
                    continue;
                }
                debug_assert!(!t.is_royal(piece_type), "a royal under a command");
                let turncoat = bits & SIGHT_COMMAND_OWN != 0 && bits & SIGHT_DISARM == 0;
                let takes = if turncoat { Some((low, high)) } else { None };
                self.push_foreign_piece(origin, piece_type, frame, owner, takes, false, out);
            }
        }
    }

    /// One foreign piece's moves: the piece of `piece_type` on `origin`, read in `frame`,
    /// promoting, screening and holding first-move rights as `owner`'s piece, landing on empty
    /// squares and on the bytes `takes` admits (never a royal, only a type its capture list
    /// allows, only on a capture line). A turn in place where its type turns. Never a
    /// castling, an en passant, a drop or a promotion: an arrival that would have to promote
    /// is not offered, and under optional promotion the piece arrives as it is.
    #[inline(always)]
    ///
    /// `royals` (Q1077): a held neutral of a `takes_royal` type may take a royal in `takes`
    /// too. False for every other foreign piece.
    #[allow(clippy::too_many_arguments)]
    fn push_foreign_piece(
        &self, origin: usize, piece_type: usize, frame: usize, owner: usize, takes: ForeignTakes,
        royals: bool, out: &mut Vec<u32>,
    ) {
        let t = self.t;
        let low = t.value_low(owner);
        let high = low + t.side_stride as u16;
        let origin_unmoved = self.unmoved.get(origin);
        let promoting = t.is_promoting(piece_type);
        let leaving =
            promoting && t.promotes_on_leaving(piece_type) && t.promotes_on(owner, piece_type, origin);
        let landing = ForeignLanding { takes, royals, owner };
        let must_promote = |target: usize| landing.must_promote(t, piece_type, leaving, target);
        // A capture: a non-royal byte in the admitted range, on a capture line, of a type the
        // piece may take.
        let may_land_on = |occupant: u16, mode: u8| landing.lands(t, piece_type, occupant, mode);
        let word = |target: usize| with_foreign(encode_move(origin, target, MOVE_NORMAL, 0));
        let slot = t.slot(frame, piece_type, origin);
        let segment_start = out.len();
        // SR3g (P10): a landing on an empty fold may also go through, as the holder's piece's
        // would (`Game::push_through_as`); SR5c2 (V1) a commanded piece's as its owner's. One
        // test for every other game.
        let folds = t.portals.as_deref().is_some_and(|portals| portals.folds.is_some());
        // SR4k (O3): a held piece's capture under `position: "choose"` fans out over its
        // holder's landing squares, as its holder's piece's would. Only a held piece: a
        // commanded one never strikes (an empty landing, or SR4h's own-side removal).
        let choosing = owner == self.side
            && t.strike.as_ref().is_some_and(|rule| rule.position == StrikePosition::Choose);

        for entry in t.leaps.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            let squares = t.leaps.squares(entry);
            let modes = t.leaps.modes(entry);
            for index in 0..squares.len() {
                let target = squares[index] as usize;
                let occupant = self.board[target] as u16;
                let mode = modes[index];
                if folds && occupant == 0 && mode != 0 && self.fold_on(target) {
                    self.push_through_as(out, origin, target, mode, piece_type, leaving, Some(landing));
                }
                let lands = if occupant == 0 { mode & 1 != 0 } else { may_land_on(occupant, mode) };
                if lands && !must_promote(target) {
                    if choosing && occupant != 0 {
                        self.fan_out_landing(out, word(target), piece_type, owner, target);
                    } else {
                        out.push(word(target));
                    }
                }
            }
        }

        for entry in t.slides.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            let squares = t.slides.squares(entry);
            let modes = t.slides.modes(entry);
            for index in 0..squares.len() {
                let target = squares[index] as usize;
                let occupant = self.board[target] as u16;
                let mode = modes[index];
                if folds && occupant == 0 && mode != 0 && self.fold_on(target) {
                    self.push_through_as(out, origin, target, mode, piece_type, leaving, Some(landing));
                }
                let lands = if occupant == 0 { mode & 1 != 0 } else { may_land_on(occupant, mode) };
                if lands && !must_promote(target) {
                    if choosing && occupant != 0 {
                        self.fan_out_landing(out, word(target), piece_type, owner, target);
                    } else {
                        out.push(word(target));
                    }
                }
                if occupant != 0 {
                    break;
                }
            }
        }

        for entry in t.hops.entries(slot) {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            let squares = t.hops.squares(entry);
            let modes = t.hops.modes(entry);
            let length = squares.len();
            let mut index = 0;
            while index < length && self.board[squares[index] as usize] == 0 {
                index += 1;
            }
            // The screen is judged as the owner would judge it.
            if index >= length
                || !screen_matches(entry.screen_owner, self.board[squares[index] as usize] as u16, low, high, &t.screen_types)
            {
                continue;
            }
            index += 1;
            let stop = if entry.screen_range == 0 {
                length
            } else {
                length.min(index + entry.screen_range as usize)
            };
            while index < stop {
                let target = squares[index] as usize;
                let occupant = self.board[target] as u16;
                let mode = modes[index];
                if folds && occupant == 0 && mode != 0 && self.fold_on(target) {
                    self.push_through_as(out, origin, target, mode, piece_type, leaving, Some(landing));
                }
                let lands = if occupant == 0 { mode & 1 != 0 } else { may_land_on(occupant, mode) };
                if lands && !must_promote(target) {
                    if choosing && occupant != 0 {
                        self.fan_out_landing(out, word(target), piece_type, owner, target);
                    } else {
                        out.push(word(target));
                    }
                }
                if occupant != 0 {
                    break;
                }
                index += 1;
            }
        }

        // SR6f (V8): a commanded piece's sweeps (`push_commanded_sweeps`); a held piece's sweep
        // is refused (O16). Before the uniqueness pass, which folds a sweep over an empty line
        // into a slide's word.
        if t.has_sweep && owner != self.side {
            let piece = PieceAt {
                origin, slot, piece_type, player: owner, low, high, origin_unmoved,
                promoting: false, leaving: false, promo_optional: false, capturable_ep: false,
                choosing: false,
            };
            self.push_commanded_sweeps(out, &piece, &landing, folds);
        }

        // A go-through to a square also reached directly, or through two folds, is one move
        // (`dedup_fold_segment`, as for a player's piece).
        if folds && (choosing || out[segment_start..].iter().any(|&m| move_fold(m) != 0)) {
            dedup_fold_segment(out, segment_start, |m| choosing && self.word_fans_out(m, owner));
        } else if t.dedup[slot] {
            dedup_segment(out, segment_start);
        }

        // SR3g (P10): through a momentum end, as the holder's piece would go (SR5c2, V1: a
        // commanded piece as its owner's), landing at the exit only where this piece may land
        // (`Game::push_transits`).
        if let Some(portals) = t.portals.as_deref() {
            let at = self.portal_squares(portals);
            self.push_transits(out, portals, &at, origin, slot, piece_type, leaving, Some(landing));
        }

        // A turn in place, as the owner may turn it (`generate_pseudo_legal_as`).
        let axes = if t.has_facing { t.turns_of(piece_type) } else { 1 };
        if axes > 1 {
            let step = t.n_facings / axes;
            let facing = frame % t.n_facings;
            for turn in 1..axes {
                let to = (facing + turn * step) % t.n_facings;
                out.push(with_foreign(encode_move(origin, origin, MOVE_ROTATE, to as u32)));
            }
        }
    }

    /// SR6f (V8, `docs/plan-schema-next.md` §2.7, ruled 2026-10-27): the sweeps of a
    /// commanded piece, walked as its owner's sweep always is (its rays, modes and capture
    /// list, `rides_from` past its cleared part, a fold go-through as its owner's), with the
    /// sides it removes the command's: a piece on the ray that `commanded_sweeps` refuses
    /// (the commander's, a royal, nobody's, a shielded one, or any piece at all without
    /// `captures_own`) ends the ray and is never a landing. Without `captures_own` a
    /// commanded sweep is a move over an empty line onto an empty square, the slide's word.
    /// Its words name a ray and keep one sweep per swept set as the owner's would
    /// (`Game::keep_sweep`, over this piece's segment, which starts where its sweeps do).
    #[cold]
    #[inline(never)]
    fn push_commanded_sweeps(
        &self, out: &mut Vec<u32>, piece: &PieceAt, landing: &ForeignLanding, folds: bool,
    ) {
        let t = self.t;
        let PieceAt { origin, slot, piece_type, origin_unmoved, .. } = *piece;
        let through = folds && t.fold_sweeps;
        let named = t.sweep_rays || through;
        let start = out.len();
        for (ray, entry) in t.sweeps.entries(slot).iter().enumerate() {
            if entry.requires_unmoved && !origin_unmoved {
                continue;
            }
            let squares = t.sweeps.squares(entry);
            let modes = t.sweeps.modes(entry);
            let rides_from = entry.sweep_length();
            let mut path_holds_piece = false;
            for index in 0..squares.len() {
                let target = squares[index] as usize;
                let occupant = self.board[target];
                if occupant != 0 && !self.commanded_sweeps(landing, piece_type, target) {
                    break;
                }
                let mode = modes[index];
                let lands = if occupant == 0 { mode & 1 != 0 } else { mode & 2 != 0 };
                if lands {
                    let kind = if path_holds_piece { MOVE_SWEEP } else { MOVE_NORMAL };
                    let m = encode_move(origin, target, kind, 0);
                    if kind == MOVE_SWEEP && named {
                        let word = if t.sweep_rays { with_ray(m, ray) } else { m };
                        self.keep_sweep(out, start, with_foreign(word), self.board[origin]);
                    } else {
                        out.push(with_foreign(m));
                    }
                }
                if occupant == 0 {
                    if through && mode != 0 && self.fold_on(target) {
                        self.push_sweep_through(
                            out, start, piece, ray, index, path_holds_piece, Some(*landing),
                        );
                    }
                } else {
                    if index >= rides_from {
                        break;
                    }
                    path_holds_piece = true;
                }
            }
        }
    }

    /// SR6f (V8): whether a commanded sweep of the piece of `piece_type` may remove the piece
    /// on `square` -- sweep it, or take it on its landing. Only under `captures_own` without
    /// disarm (`ForeignLanding::takes`), only a piece of its owner's side, never a royal, only
    /// a type its capture list allows, and never a shielded piece (`Game::sweep_shielded`,
    /// SR6g: every sweep reads the shield on the squares it passes and lands on). Every other
    /// piece ends the commanded ray.
    pub(super) fn commanded_sweeps(
        &self, landing: &ForeignLanding, piece_type: usize, square: usize,
    ) -> bool {
        let t = self.t;
        let occupant = self.board[square] as u16;
        let Some((from, to)) = landing.takes else {
            return false;
        };
        if occupant < from || occupant >= to || t.is_royal(t.type_of(occupant as u8)) {
            return false;
        }
        may_take(t, piece_type, occupant) && !self.sweep_shielded(square)
    }
}
