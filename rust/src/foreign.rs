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
//! | nobody's `MAJORITY` pieces the mover holds (TC3, `docs/plan-turncoat.md` §4), and every shared one (TS2, `docs/plan-tessabi.md` §2) | frame 0 | empty squares; the mover's opponent's non-royal pieces its `captures` allow |
//!
//! Both feed the same per-piece walk, `push_foreign_piece`. Every foreign move excludes
//! castling, en passant and drops, a commanded one promotion too (a held piece promotes in its
//! holder's zone since SR8j, `push_held_promotions`), and every one is verified for the mover's
//! royal (`generate_checked`).
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

    /// Whether an arrival on `target` would have to promote, which a commanded move never
    /// does, so it is not offered. `leaving` is the origin's answer under `on_leaving`. A
    /// held piece's promotion (SR8j) is asked before this, by every site that offers one.
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
    /// the command lets it take (`push_foreign_sweeps`). The shield on a victim is
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
    /// of a majority type that the side to move holds (`controller_of`), landing on empty
    /// squares and on the opponent's non-royal pieces its `captures` allow. Its byte stays
    /// nobody's: moving it does not change its allegiance.
    ///
    /// SR8i (`docs/plan-schema-next.md` §2.11, O9-O19): it moves in its holder's frame, as
    /// the holder's own piece of the type pointing the same way would (`NeutralTables::
    /// held_frame`: forward, ranks, region and conditions the holder's; a facing its arrow;
    /// a first move its own mark), screened as the holder's (`own` the holder's pieces,
    /// `enemy` the other player's), and turned in nobody's frame. Where every held type
    /// reads alike for both players that frame is frame 0, as TC3 built it.
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
            if t.kind_of(value) != CONTROL_MAJORITY {
                continue;
            }
            let piece_type = t.type_of(value);
            // TS2: a shared type is the side to move's whoever stands next to it.
            if !rule.shared(piece_type) && self.controller_of(origin) != player as i8 {
                continue;
            }
            let royals = rule.takes_royal(piece_type);
            // SR8i (*the mover's reading*): its holder's frame, the holder's own piece of
            // the type pointing where it points; frame 0 where every held type reads alike.
            // SR8j: a type that does not face stands at facing 0, which is no arrow, so its
            // holder's frame is the holder's own forward (`player * n_facings`), the frame the
            // holder's own piece of it reads, whatever the other types' facings. SR8k: one
            // statement, which a held sweep's path reads too (`Tables::held_mover_frame`).
            let frame = t.held_mover_frame(value, player);
            self.push_foreign_piece(origin, piece_type, frame, player, Some((low, high)), royals, out);
        }
    }

    /// Q1077 (`docs/plan-turncoat.md` §4.7): whether a held neutral of a `takes_royal` type
    /// that `by` holds now attacks `square` (as a capture of `victim`, or of whatever stands
    /// there). The compiler files those types' frame-0 attack sources under the third owner's
    /// block with the nobody's byte, so this is `attacks_square_as`'s walk over that block,
    /// each matching source then asked who holds it. Reads the board, never the occupancy
    /// sets, so a legality filter may ask it. Its hop screens name no owner, so the path test
    /// reads alike for either holder. SR8k (O16): a sweep source attacks nothing here, since
    /// a royal ends a held sweep's ray (`commanded_sweeps`): a held sweep never takes one.
    #[cold]
    #[inline(never)]
    pub(super) fn held_attacks(&self, by: usize, square: usize, victim: Option<usize>) -> bool {
        let t = self.t;
        let victim = victim.unwrap_or_else(|| {
            let occupant = self.board[square];
            if occupant == 0 { usize::MAX } else { t.victim_of(occupant) }
        });
        if t.majority.as_deref().is_some_and(|rule| rule.in_holders_frame()) {
            return self.held_attacks_in_holders_frame(by, square, victim);
        }
        for source in t.attackers_of(WILD, square) {
            if source.sweep != 0 || self.board[source.origin as usize] != source.piece_value {
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

    /// SR8i (O17, with Q1077): `held_attacks` where a held piece moves in its holder's frame.
    /// A held neutral that `by` holds attacks as `by`'s own piece of its type pointing the
    /// same way would, so the sources are `by`'s own block's for that type and facing
    /// (`attackers_of(by, ..)`), each matched against the nobody's byte whose holder's frame
    /// it is (`NeutralTables::held_frame`), on the board, then asked who holds it. A screen
    /// named `enemy` is the other player's piece only (a nobody's piece screens under `any`
    /// alone). Reads the board, never the occupancy sets.
    #[cold]
    #[inline(never)]
    fn held_attacks_in_holders_frame(&self, by: usize, square: usize, victim: usize) -> bool {
        let t = self.t;
        let rule = t.majority.as_deref().expect("asked only with majority tables");
        let other_low = t.value_low(1 - by);
        let other_high = other_low + t.side_stride as u16;
        for source in t.attackers_of(by, square) {
            let piece_type = t.type_of(source.piece_value);
            // SR8k (O16): a held sweep never takes a royal, so its sources attack nothing.
            if !rule.takes_royal(piece_type) || source.sweep != 0 {
                continue;
            }
            let frame = by * t.n_facings + t.facing_of_value(source.piece_value);
            // SR8j: a type that does not face is nobody's at facing 0 and reads the holder's
            // forward (`push_held`); only a type that faces is matched by its arrow.
            let found = if t.facings_of(piece_type) > 1 {
                (0..t.n_facings).find(|&facing| rule.held_frame(t.n_facings, by, facing) == frame)
            } else {
                (frame == by * t.n_facings).then_some(0)
            };
            let Some(facing) = found else {
                continue;
            };
            let origin = source.origin as usize;
            if self.board[origin] != t.piece_value(WILD, facing, piece_type) {
                continue;
            }
            if source.requires_unmoved && !self.unmoved.get(origin) {
                continue;
            }
            if t.has_capture_matrix && victim != usize::MAX && !t.may_capture(piece_type, victim) {
                continue;
            }
            if !self.path_clear_on(&self.board, by, source) {
                continue;
            }
            if source.screen_owner == SCREEN_ENEMY
                && t.attack_path(source)[(source.prefix_len as usize).min(t.attack_path(source).len())..]
                    .iter()
                    .any(|&s| {
                        let occupant = self.board[s as usize] as u16;
                        occupant != 0 && !(other_low <= occupant && occupant < other_high)
                    })
            {
                continue;
            }
            if self.controller_of(origin) == by as i8 {
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
    /// castling, an en passant or a drop. A commanded piece never promotes: an arrival that
    /// would have to is not offered, and under optional promotion it arrives as it is. A held
    /// piece (SR8j, O15) promotes in its holder's zone (`push_held_promotions`).
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
        let held = owner == self.side;
        let choosing = held
            && t.strike.as_ref().is_some_and(|rule| rule.position == StrikePosition::Choose);
        // SR8i (O19): the other player's pieces, a held piece's `enemy`.
        let other_low = t.value_low(1 - owner);
        let held_enemy = |occupant: u16| {
            other_low <= occupant && occupant < other_low + t.side_stride as u16
        };
        // SR8j (O15): a held piece of a promoting type promotes in its holder's zone, to the
        // holder's targets (`push_held_promotions`); a commanded piece never promotes. False
        // in every game where no held type promotes, which was every game before SR8j.
        let held_promotes = |target: usize| {
            held && promoting && (leaving || t.promotes_on(owner, piece_type, target))
        };

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
                if lands && held_promotes(target) {
                    self.push_held_promotions(out, origin, target, piece_type, owner, choosing && occupant != 0);
                } else if lands && !must_promote(target) {
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
                if lands && held_promotes(target) {
                    self.push_held_promotions(out, origin, target, piece_type, owner, choosing && occupant != 0);
                } else if lands && !must_promote(target) {
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
            // The screen is judged as the owner would judge it. SR8i (O19): a held piece's
            // `enemy` screen is the other player's piece only; a nobody's piece is neither
            // side's, so it screens a held hop only under `any`.
            if index >= length {
                continue;
            }
            let screen = self.board[squares[index] as usize] as u16;
            let screened = if held && entry.screen_owner == SCREEN_ENEMY {
                held_enemy(screen)
            } else {
                screen_matches(entry.screen_owner, screen, low, high, &t.screen_types)
            };
            if !screened {
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
                if lands && held_promotes(target) {
                    self.push_held_promotions(out, origin, target, piece_type, owner, choosing && occupant != 0);
                } else if lands && !must_promote(target) {
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

        // SR6f (V8): a commanded piece's sweeps; SR8k (O16) a held piece's, removing what its
        // holder could capture with it (`push_foreign_sweeps`). Before the uniqueness pass,
        // which folds a sweep over an empty line into a slide's word.
        if t.has_sweep {
            let piece = PieceAt {
                origin, slot, piece_type, player: owner, low, high, origin_unmoved,
                promoting: false, leaving: false, promo_optional: false, capturable_ep: false,
                choosing: false,
            };
            self.push_foreign_sweeps(out, &piece, &landing, folds);
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

        // A turn in place, as the owner may turn it (`generate_pseudo_legal_as`). The word
        // names the facing its byte will carry: the owner's index for a commanded piece, and
        // for a held one (SR8i, O9) nobody's, which `make` writes into the third block.
        let axes = if t.has_facing { t.turns_of(piece_type) } else { 1 };
        if axes > 1 {
            let step = t.n_facings / axes;
            let facing = t.facing_of_value(self.board[origin]);
            for turn in 1..axes {
                let to = (facing + turn * step) % t.n_facings;
                out.push(with_foreign(encode_move(origin, origin, MOVE_ROTATE, to as u32)));
            }
        }
    }

    /// SR8j (`docs/plan-schema-next.md` §2.11 O15, *the mover's reading*, ruled 2026-10-28):
    /// the promotions a held piece of `piece_type` arriving on `target` in `holder`'s zone
    /// may make, each a `MOVE_PROMOTION` word marked foreign, then the plain arrival where
    /// the type's promotion is optional: the holder's own targets (`promotion_targets`,
    /// `only_lost` counted for the holder), less those a held piece never becomes
    /// (`NeutralTables::held_promoted`: a royal, a wild type, the duck). `make` writes the
    /// piece nobody's (`Tables::promoted_value`). Under `fan` (a capture under `position:
    /// "choose"`) each word fans out over the holder's landings, as `push_arrival`'s does.
    ///
    /// R-L6c (F2): a target is offered only where it may stand, read as the holder's piece of
    /// the type (`promotion_offered` on `target`), as the held piece's own moves read its
    /// holder's region (SR8i).
    #[cold]
    #[inline(never)]
    pub(super) fn push_held_promotions(
        &self, out: &mut Vec<u32>, origin: usize, target: usize, piece_type: usize,
        holder: usize, fan: bool,
    ) {
        let t = self.t;
        let mut emit = |m: u32| {
            if fan {
                self.fan_out_landing(out, m, piece_type, holder, target);
            } else {
                out.push(m);
            }
        };
        for promoted in self.held_promotion_targets(holder, piece_type, target) {
            emit(with_foreign(encode_move(origin, target, MOVE_PROMOTION, promoted)));
        }
        if t.promotion_optional(piece_type) {
            emit(with_foreign(encode_move(origin, target, MOVE_NORMAL, 0)));
        }
    }

    /// SR8j (O15): the targets a held piece of `piece_type` that `holder` promotes may
    /// become, in the holder's order: the holder's own promotion targets offered now
    /// (`promotion_offered`), less those `NeutralTables::held_promoted` never offers.
    pub(super) fn held_promotion_targets(
        &self, holder: usize, piece_type: usize, square: usize,
    ) -> impl Iterator<Item = u32> + '_ {
        let t = self.t;
        let rule = t.majority.as_deref();
        t.promotion_targets(holder, piece_type).iter().copied().filter(move |&promoted| {
            rule.is_some_and(|rule| rule.held_promoted(t, holder, promoted as usize).is_some())
                && self.promotion_offered(holder, piece_type, promoted, square)
        })
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
    ///
    /// SR8k (`docs/plan-schema-next.md` §2.11 O16, *the mover's reading*, ruled 2026-10-28):
    /// a held sweeper (`landing.owner` the side to move, `piece` read in its holder's frame)
    /// walks the same rays with the holder as the side: `commanded_sweeps` admits only the
    /// other player's non-royal, unshielded pieces its captures allow (`landing.takes`, the
    /// range `push_held` hands it), so the holder's pieces, a royal, another nobody's piece
    /// and a shielded piece end its ray. A held sweeper promotes in its holder's zone as any
    /// held piece does (SR8j), its word the promoting sweep's (`push_held_sweep_promotions`).
    #[cold]
    #[inline(never)]
    fn push_foreign_sweeps(
        &self, out: &mut Vec<u32>, piece: &PieceAt, landing: &ForeignLanding, folds: bool,
    ) {
        let t = self.t;
        let PieceAt { origin, slot, piece_type, origin_unmoved, .. } = *piece;
        let through = folds && t.fold_sweeps;
        let named = t.sweep_rays || through;
        // SR8b (W7): a commanded sweeper never promotes, so a landing that would have to is
        // not offered, and under an optional promotion it arrives as it is (as every
        // commanded arrival, `ForeignLanding::must_promote`). SR8k: a held one promotes.
        let leaving = t.is_promoting(piece_type)
            && t.promotes_on_leaving(piece_type)
            && t.promotes_on(landing.owner, piece_type, origin);
        let held_promoting = landing.owner == self.side && t.is_promoting(piece_type);
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
                if lands
                    && held_promoting
                    && (leaving || t.promotes_on(landing.owner, piece_type, target))
                {
                    self.push_held_sweep_promotions(
                        out, start, origin, target, piece_type, landing.owner, ray,
                        path_holds_piece, named,
                    );
                } else if lands && !landing.must_promote(t, piece_type, leaving, target) {
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

    /// SR8k (O16, with SR8j's O15): a held sweep's landing on `target` in its holder's zone:
    /// one word per target a held piece of the type may become for `holder`
    /// (`held_promotion_targets`), then the plain landing under an optional promotion. Over
    /// pieces a `MOVE_SWEEP` with the type + 1 in the promotion field, kept as any sweep
    /// (`keep_sweep`); over an empty line the slide's `MOVE_PROMOTION`. Every word foreign, so
    /// `make` writes the promoted piece nobody's (`Tables::promoted_value`). Never beside folds
    /// (a promoting sweeper is refused there), so the field is never a fold step.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn push_held_sweep_promotions(
        &self, out: &mut Vec<u32>, start: usize, origin: usize, target: usize, piece_type: usize,
        holder: usize, ray: usize, path_holds_piece: bool, named: bool,
    ) {
        let t = self.t;
        debug_assert!(!t.fold_sweeps, "a promoting sweeper beside folds");
        let moving = self.board[origin];
        let push = |game: &Self, out: &mut Vec<u32>, m: u32| {
            let m = with_foreign(m);
            if move_kind(m) == MOVE_SWEEP && named {
                let word = if t.sweep_rays { with_ray(m, ray) } else { m };
                game.keep_sweep(out, start, word, moving);
            } else {
                out.push(m);
            }
        };
        let targets: Vec<u32> = self.held_promotion_targets(holder, piece_type, target).collect();
        for promoted in targets {
            let m = if path_holds_piece {
                encode_move(origin, target, MOVE_SWEEP, promoted + 1)
            } else {
                encode_move(origin, target, MOVE_PROMOTION, promoted)
            };
            push(self, out, m);
        }
        if t.promotion_optional(piece_type) {
            let kind = if path_holds_piece { MOVE_SWEEP } else { MOVE_NORMAL };
            push(self, out, encode_move(origin, target, kind, 0));
        }
    }

    /// SR6f (V8): whether a commanded sweep of the piece of `piece_type` may remove the piece
    /// on `square` -- sweep it, or take it on its landing. Only under `captures_own` without
    /// disarm (`ForeignLanding::takes`), only a piece of its owner's side, never a royal, only
    /// a type its capture list allows, and never a shielded piece (`Game::sweep_shielded`,
    /// SR6g: every sweep reads the shield on the squares it passes and lands on). Every other
    /// piece ends the commanded ray. SR8k (O16): a held sweep reads it too, its `takes` the
    /// other player's block (`push_held`), so it removes only what its holder could capture
    /// with it, and the holder's pieces, a royal, a nobody's piece and a shielded one end it.
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
