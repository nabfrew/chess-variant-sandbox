//! Lines of sight with passive effects (schema 30, Q999a): the one effect query in the core,
//! read by the generator's filter and by the attack test.
//!
//! `docs/archive/plan-passive.md` is the rule (§2) and its build (§3). A piece type may declare up to
//! four sights; every player's piece standing in one is under its effect, read from the board
//! as it stands, with no memory. Three effects: **freeze** (the piece may not move, and
//! attacks nothing), **disarm** (it may not capture, and attacks nothing), **shield** (nothing
//! may capture it; a shielded royal is never in check). An effect is a function of the board,
//! so there is no state, no key and no undo. Three calls are the whole of it:
//!
//! * `effects_at`, the one query: the effect bits on a player's piece on a square, a walk of
//!   the compiler's reverse table (`SightTables`) as `attacks_square_as` walks its sources.
//! * `sight_filter`, at the end of `generate_pseudo_legal`: drop every move a frozen piece
//!   makes (and a castling whose partner is frozen), every landing-on-a-piece move of a
//!   disarmed piece, and every move that lands on, or takes in passing, a shielded piece.
//!   Every caller of the generator sees it without being told.
//! * `attacks_in_sight_game`, the attack test's third arm (`attacks_any`, `threatened_as`):
//!   a shielded victim is attacked by nothing, and a frozen or disarmed attacker attacks
//!   nothing. `attacks_square` itself, the hottest query, is the one it always was.
//!
//! * A fourth effect, **command** (Q1007c, `docs/archive/plan-schema-31.md` §3: the piece's
//!   opponent may also move it), is a second source of moves: the side to move's moves with
//!   the opponent's commanded pieces, walked before the filter by `foreign.rs::push_foreign`
//!   (TC1, the one source of foreign moves).
//!
//! Every game without a sight has `Tables::sight` `None` and pays one test at each call site.
//! `tests/sight_oracle.py` is the declared oracle.

use super::*;

/// Why the filter removed a move, for `Core.effects`' tallies (§3.4): 0 kept, else one of
/// these.
pub const BARRED_FREEZE: u8 = 1;
pub const BARRED_DISARM: u8 = 2;
pub const BARRED_SHIELD: u8 = 3;

/// The answer to `Core.effects`: each square's effect bits (on the piece standing there, for
/// its owner; 0 on an empty square or a wild piece), how many of the side to move's moves the
/// filter removed for each reason (freeze, disarm, shield), and whether the side to move's
/// royal stands under an attack the effects suppressed.
pub struct EffectsReport {
    pub bits: Vec<u8>,
    pub removed: [u32; 3],
    pub suppressed: bool,
    /// Q1007c: how many of the side to move's legal moves are commanded ones.
    pub commanded: u32,
    /// Generation 37 (N2): each square's effect bits from the wild pieces' sights alone (on
    /// a player's piece, for its owner), the tallies' third owner column. Empty in a game
    /// without wild pieces.
    pub wild_bits: Vec<u8>,
}

impl<'t> Game<'t> {
    /// The effect bits on `player`'s piece standing on `square`: zero for an empty square, a
    /// wild piece, or the other player's piece. Reads the board alone, so it is sound inside a
    /// legality filter, where the board is mid-experiment.
    #[inline]
    pub(super) fn effects_at(&self, sight: &SightTables, player: usize, square: usize) -> u8 {
        let t = self.t;
        let occupant = self.board[square];
        if occupant == 0 || t.owner_of(occupant) != player {
            return 0;
        }
        let mut bits = 0u8;
        for source in sight.sources_of(player, t.type_of(occupant), square) {
            // Skip a source whose bits are all set already. One bit per source but a
            // command that `captures_own` (Q1007c), whose two bits are a union too.
            if source.effect & !bits == 0 || self.board[source.origin as usize] != source.piece_value
            {
                continue;
            }
            if sight.path(source).iter().all(|s| self.board[*s as usize] == 0) {
                bits |= source.effect;
            }
        }
        bits
    }

    /// Generation 37 (N2, `docs/archive/plan-npc.md` §4.2): the shielded squares a wild phase reads,
    /// once, at its start: every route square holding a player's piece under a shield. A
    /// step or a return only ever lands on a route square, so these are all it can ask
    /// about. Empty, at one flag test, in every game without a shield beside wild pieces.
    #[inline]
    pub(super) fn wild_shields(&self) -> SquareSet {
        if !self.t.npc_shield {
            return SquareSet::EMPTY;
        }
        self.wild_shields_walk()
    }

    #[cold]
    #[inline(never)]
    fn wild_shields_walk(&self) -> SquareSet {
        let t = self.t;
        let mut shields = SquareSet::EMPTY;
        for &square in &t.npc_squares {
            if self.shielded_from_wild(square as usize) {
                shields.set(square as usize);
            }
        }
        shields
    }

    /// Whether the piece on `square` is a player's under a shield, which no wild piece takes
    /// (generation 37): the wild arm of `threatened` asks it, so a shielded royal on a wild
    /// piece's next square is not in check from it. Only where `Tables::npc_shield`.
    #[cold]
    #[inline(never)]
    pub(super) fn shielded_from_wild(&self, square: usize) -> bool {
        let t = self.t;
        let Some(sight) = t.sight.as_deref() else {
            return false;
        };
        let occupant = self.board[square];
        if occupant == 0 {
            return false;
        }
        let owner = t.owner_of(occupant);
        owner < 2 && self.effects_at(sight, owner, square) & SIGHT_SHIELD != 0
    }

    /// Generation 37 (N2): the effect bits on the player's piece on `square` from the wild
    /// pieces' sights alone, read as `effects_at` reads every source. For `Core.effects`.
    fn wild_effects_at(&self, sight: &SightTables, square: usize) -> u8 {
        let t = self.t;
        let occupant = self.board[square];
        if occupant == 0 || t.owner_of(occupant) >= 2 {
            return 0;
        }
        let mut bits = 0u8;
        for source in sight.sources_of(t.owner_of(occupant), t.type_of(occupant), square) {
            if t.owner_of(source.piece_value) == WILD
                && self.board[source.origin as usize] == source.piece_value
                && sight.path(source).iter().all(|s| self.board[*s as usize] == 0)
            {
                bits |= source.effect;
            }
        }
        bits
    }

    /// The generator's filter (§3.4): drop from `out[base..]` what the effects on this board
    /// forbid the side to move. Out of line, so a game without a sight carries none of it.
    #[cold]
    #[inline(never)]
    pub(super) fn sight_filter(&self, sight: &SightTables, out: &mut Vec<u32>, base: usize) {
        if out.len() == base {
            return;
        }
        let mut barring = SightBarring::new(self, sight);
        if !barring.binds() {
            return;
        }
        let mut write = base;
        for index in base..out.len() {
            let m = out[index];
            if barring.reason(self, m) == 0 {
                out[write] = m;
                write += 1;
            }
        }
        out.truncate(write);
    }

    /// Whether `player` could capture what stands on `square` (or a piece of `victim` type
    /// there, where a caller means one) under this board's effects (§2.4): a shielded victim
    /// is attacked by nothing, and a frozen or disarmed attacker attacks nothing. The attack
    /// tables give the geometry, the capture matrix and the effects the permission.
    #[cold]
    #[inline(never)]
    pub(super) fn attacks_in_sight_game(
        &self, sight: &SightTables, player: usize, square: usize, victim: Option<usize>,
    ) -> bool {
        let t = self.t;
        let occupant = self.board[square];
        if sight.has_shield && occupant != 0 {
            let owner = t.owner_of(occupant);
            if owner < 2 && self.effects_at(sight, owner, square) & SIGHT_SHIELD != 0 {
                return false;
            }
        }
        let victim = victim.unwrap_or_else(|| {
            if occupant == 0 || !t.has_capture_matrix { usize::MAX } else { t.type_of(occupant) }
        });
        for source in t.attackers_of(player, square) {
            if self.board[source.origin as usize] != source.piece_value {
                continue;
            }
            if source.requires_unmoved && !self.unmoved.get(source.origin as usize) {
                continue;
            }
            if t.has_capture_matrix
                && victim != usize::MAX
                && !t.may_capture(t.type_of(source.piece_value), victim)
            {
                continue;
            }
            let reaches = if source.sweep != 0 {
                self.sweep_attacks(player, source)
            } else {
                self.source_path_clear(player, source)
            };
            if reaches
                && (!sight.has_restraint
                    || self.effects_at(sight, player, source.origin as usize)
                        & (SIGHT_FREEZE | SIGHT_DISARM)
                        == 0)
            {
                return true;
            }
        }
        false
    }

    /// The squares a move of `player`'s must touch, as its origin or its target, to change a
    /// check on `player`'s royal on `royal` through an effect (§3.5's second lever): the
    /// origin and the ray of every source whose freeze or disarm binds one of the other
    /// player's pieces now, and of every shield that binds the royal now. A move off these
    /// squares adds effects at most (a new source can only restrain an attacker or shield the
    /// royal), so it cannot leave the royal attacked where it was not; a move that changes a
    /// piece's type in place (a strike or contact) is asked anyway by the shortcut's own test.
    #[inline(never)]
    pub(super) fn effect_lines(&self, sight: &SightTables, player: usize, royal: usize) -> SquareSet {
        let t = self.t;
        let mut lines = SquareSet::EMPTY;
        let note = |source: &SightSource, lines: &mut SquareSet| {
            if self.board[source.origin as usize] == source.piece_value
                && sight.path(source).iter().all(|s| self.board[*s as usize] == 0)
            {
                lines.set(source.origin as usize);
                for square in sight.path(source) {
                    lines.set(*square as usize);
                }
            }
        };
        if sight.has_restraint {
            let enemy = 1 - player;
            for square in self.occupied()[enemy].iter() {
                let piece_type = t.type_of(self.board[square]);
                for source in sight.sources_of(enemy, piece_type, square) {
                    if source.effect & (SIGHT_FREEZE | SIGHT_DISARM) != 0 {
                        note(source, &mut lines);
                    }
                }
            }
        }
        if sight.has_shield {
            let piece_type = t.type_of(self.board[royal]);
            for source in sight.sources_of(player, piece_type, royal) {
                if source.effect == SIGHT_SHIELD {
                    note(source, &mut lines);
                }
            }
        }
        lines
    }

    /// The movement tables alone: `generate_pseudo_legal` without the sight's filter, for the
    /// one caller that asks about geometry (rounds' `lands_if_empty`, §2.8).
    pub(super) fn generate_movement(&self, out: &mut Vec<u32>) {
        if self.t.portals.as_deref().is_some_and(|portals| portals.folds.is_some()) {
            return self.generate_folded(out);
        }
        self.generate_pseudo_legal_as::<false>(out)
    }

    /// `Core.effects` (§3.4): every square's effect bits, the moves the filter removed for the
    /// side to move by reason, and whether its royal stands under a suppressed attack. Zeros
    /// and false in a game without a sight.
    pub fn effects_report(&mut self) -> EffectsReport {
        let t = self.t;
        let mut report = EffectsReport {
            bits: vec![0; t.n_squares],
            removed: [0; 3],
            suppressed: false,
            commanded: 0,
            wild_bits: Vec::new(),
        };
        let Some(sight) = t.sight.as_deref() else {
            return report;
        };
        for (square, bits) in report.bits.iter_mut().enumerate() {
            let occupant = self.board[square];
            if occupant != 0 {
                let owner = t.owner_of(occupant);
                if owner < 2 {
                    *bits = self.effects_at(sight, owner, square);
                }
            }
        }
        if t.has_npcs {
            report.wild_bits = (0..t.n_squares).map(|square| self.wild_effects_at(sight, square)).collect();
        }
        let mut moves = Vec::new();
        self.generate_movement(&mut moves);
        let mut barring = SightBarring::new(self, sight);
        if barring.binds() {
            for &m in &moves {
                let reason = barring.reason(self, m);
                if reason != 0 {
                    report.removed[reason as usize - 1] += 1;
                }
            }
        }
        let player = self.side;
        report.suppressed = self.royal_squares(player).iter().any(|&royal| {
            self.attacks_square(1 - player, royal)
                && !self.attacks_in_sight_game(sight, 1 - player, royal, None)
        });
        if sight.has_command {
            report.commanded = self
                .legal_moves()
                .iter()
                .filter(|&&m| m & MOVE_FOREIGN != 0)
                .count() as u32;
        }
        report
    }

    /// `Core.sight_of` (§3.4): the squares the piece on `square` sees now, each with the
    /// effect bits of the sights that see it, in ascending square order. Empty for an empty
    /// square or a game without a sight. A ray's stopping piece is seen; nothing past it.
    pub fn sight_of(&self, square: usize) -> Vec<(usize, u8)> {
        let t = self.t;
        let Some(sight) = t.sight.as_deref() else {
            return Vec::new();
        };
        let value = self.board[square];
        if value == 0 {
            return Vec::new();
        }
        let mut seen = vec![0u8; t.n_squares];
        for player in 0..2 {
            for piece_type in 0..t.n_types {
                for (target, bits) in seen.iter_mut().enumerate() {
                    for source in sight.sources_of(player, piece_type, target) {
                        if source.origin as usize == square
                            && source.piece_value == value
                            && sight.path(source).iter().all(|s| self.board[*s as usize] == 0)
                        {
                            *bits |= source.effect;
                        }
                    }
                }
            }
        }
        seen.iter()
            .enumerate()
            .filter(|(_, bits)| **bits != 0)
            .map(|(target, bits)| (target, *bits))
            .collect()
    }
}

/// The side to move's frozen and disarmed pieces on one board, and the shielded pieces a move
/// lands on, asked as the moves need them. One per filter pass.
struct SightBarring<'a> {
    sight: &'a SightTables,
    frozen: SquareSet,
    disarmed: SquareSet,
    asked: SquareSet,
    shielded: SquareSet,
}

impl<'a> SightBarring<'a> {
    fn new(game: &Game<'_>, sight: &'a SightTables) -> Self {
        let player = game.side;
        let mut frozen = SquareSet::EMPTY;
        let mut disarmed = SquareSet::EMPTY;
        let restrained = if sight.has_restraint { game.occupied()[player] } else { SquareSet::EMPTY };
        for square in restrained.iter() {
            let bits = game.effects_at(sight, player, square);
            if bits & SIGHT_FREEZE != 0 {
                frozen.set(square);
            } else if bits & SIGHT_DISARM != 0 {
                disarmed.set(square);
            }
        }
        Self { sight, frozen, disarmed, asked: SquareSet::EMPTY, shielded: SquareSet::EMPTY }
    }

    /// Whether anything could be barred: a piece of the side to move under freeze or disarm,
    /// or a shield somewhere in the game.
    fn binds(&self) -> bool {
        self.sight.has_shield || !self.frozen.is_empty() || !self.disarmed.is_empty()
    }

    /// Whether `square`'s piece is shielded, asked once per square.
    fn shielded(&mut self, game: &Game<'_>, square: usize) -> bool {
        if !self.asked.get(square) {
            self.asked.set(square);
            let occupant = game.board[square];
            let owner = if occupant == 0 { WILD } else { game.t.owner_of(occupant) };
            if owner < 2 && game.effects_at(self.sight, owner, square) & SIGHT_SHIELD != 0 {
                self.shielded.set(square);
            }
        }
        self.shielded.get(square)
    }

    /// Why the effects forbid `m` (`BARRED_*`), or 0.
    fn reason(&mut self, game: &Game<'_>, m: u32) -> u8 {
        let kind = move_kind(m);
        // TC6: a hire or a bribe moves no piece and takes none, so no effect bars it.
        if enters_play(kind) || kind == MOVE_HIRE {
            return 0;
        }
        let origin = move_from(m);
        if self.frozen.get(origin) {
            return BARRED_FREEZE;
        }
        let target = move_to(m);
        match kind {
            MOVE_ROTATE => return 0,
            MOVE_CASTLE => {
                // The partner would move too, so a frozen partner bars the castling.
                let (rook_from, _) =
                    game.castling_rook_squares(origin, target, move_promotion(m) as i32);
                return if self.frozen.get(rook_from) { BARRED_FREEZE } else { 0 };
            }
            _ => {}
        }
        let victim = match kind {
            MOVE_EN_PASSANT => game.ep_victim as usize,
            // A sweep over a piece takes it (shields are refused beside sweeps).
            MOVE_SWEEP => {
                return if self.disarmed.get(origin) { BARRED_DISARM } else { 0 };
            }
            _ if game.board[target] != 0 => target,
            _ => return 0,
        };
        if self.disarmed.get(origin) {
            return BARRED_DISARM;
        }
        if self.sight.has_shield && self.shielded(game, victim) {
            return BARRED_SHIELD;
        }
        0
    }
}
