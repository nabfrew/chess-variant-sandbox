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
/// its owner; 0 on an empty square, and on a wild piece or the duck unless a sight names its
/// type, generation 84), how many of the side to move's moves the
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
    /// The effect bits on `player`'s piece standing on `square`: zero for an empty square or
    /// another owner's piece. `player` is `WILD` (a wild piece or the duck) only where
    /// `SightTables::tamed` (generation 84). Reads the board alone, so it is sound inside a
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
            if self.sight_path_open(sight, source) {
                bits |= source.effect;
            }
        }
        if sight.folds {
            bits |= self.fold_effects_at(sight, player, t.type_of(occupant), square);
        }
        bits
    }

    /// Generation 84 (SR5f, V11, V15): the effect bits on whatever stands on `square`: a
    /// player's piece for its owner, and, where a sight names a wild type
    /// (`SightTables::tamed`), a wild piece or the duck (no other nobody's byte has a source
    /// filed under `WILD`). Zero on an empty square. The one reading of *is this piece
    /// shielded* for a capture, a blast and an attack.
    #[inline]
    pub(super) fn piece_effects(&self, sight: &SightTables, square: usize) -> u8 {
        let occupant = self.board[square];
        if occupant == 0 {
            return 0;
        }
        let owner = self.t.owner_of(occupant);
        if owner < 2 || sight.tamed {
            self.effects_at(sight, owner, square)
        } else {
            0
        }
    }

    /// Generation 96 (SR6g, W10): whether the piece on `square` stands shielded, for a sweep:
    /// a shielded piece may not be captured, so it ends a sweep's ray, as a piece the capture
    /// matrix refuses does (the sweep is neither offered past it nor onto it, and reaches
    /// nothing past it for check). Read off the board by `piece_effects`, whoever owns the
    /// piece: a sweep clears friendly pieces too, and a shield bars every removal. False
    /// without a shield in the game, so a sweep game without one pays one test.
    #[inline]
    pub(super) fn sweep_shielded(&self, square: usize) -> bool {
        match self.t.sight.as_deref() {
            Some(sight) if sight.has_shield => {
                self.piece_effects(sight, square) & SIGHT_SHIELD != 0
            }
            _ => false,
        }
    }

    /// Whether `source`'s line is open on this board: every square of its path empty, or, for
    /// an aura's square beside momentum portals (`SIGHT_PATH_AURA`, SR5c), holding no portal
    /// end. The one reading of a sight's path.
    #[inline(always)]
    pub(super) fn sight_path_open(&self, sight: &SightTables, source: &SightSource) -> bool {
        sight.path(source).iter().all(|&s| {
            let occupant = self.board[(s & SIGHT_PATH_SQUARE) as usize];
            occupant == 0 || (s & SIGHT_PATH_AURA != 0 && !self.t.is_portal_end(occupant))
        })
    }

    /// SR5c (V1, ruled 2026-10-27: *a fold propagates a sight*): the effect bits on `player`'s
    /// piece of `piece_type` on `square` that come through a fold. A fold end stands on
    /// `square`, an end leading to it stands on an empty square, and a line reaches that near
    /// square (a source onto it, as if the piece stood there: its origin holding its piece,
    /// its path open) whose piece the near end lets in: the line also sees the far end's square
    /// and the piece on it, and nothing past it (one fold per line). A piece never sees itself
    /// through a fold, and a wild piece or the duck goes through none. Asked only where the
    /// game has folds.
    #[cold]
    #[inline(never)]
    fn fold_effects_at(
        &self, sight: &SightTables, player: usize, piece_type: usize, square: usize,
    ) -> u8 {
        let t = self.t;
        let Some(portals) = t.portals.as_deref() else {
            return 0;
        };
        let Some(arrival) = self.fold_end_at(portals, square) else {
            return 0;
        };
        let arrival_type = (portals.first_type + arrival) as i8;
        let mut bits = 0u8;
        for end in 0..portals.n_ends {
            let entry = portals.first_type + end;
            let near = self.fold_at[end];
            if portals.dest[entry] != arrival_type || near < 0 || self.board[near as usize] != 0 {
                continue;
            }
            for source in sight.sources_of(player, piece_type, near as usize) {
                let origin = source.origin as usize;
                if source.effect & !bits == 0
                    || origin == square
                    || self.board[origin] != source.piece_value
                {
                    continue;
                }
                let owner = t.owner_of(source.piece_value);
                if owner >= 2
                    || portals.admits[entry][owner] & (1 << t.type_of(source.piece_value)) == 0
                {
                    continue;
                }
                if self.sight_path_open(sight, source) {
                    bits |= source.effect;
                }
            }
        }
        bits
    }

    /// Generation 37 (N2, `docs/archive/plan-npc.md` §4.2): the sight snapshot a wild phase
    /// reads, once, at its start: every route square holding a player's piece under a shield
    /// (a step or a return only ever lands on a route square, so these are all it can ask
    /// about), and (generation 84, SR5f, V11) every wild piece standing frozen or disarmed. A
    /// frozen wild piece neither walks nor pounces; a disarmed one walks onto empty squares
    /// only and never pounces. Empty, at one flag test, in every game without a shield or a
    /// sight on a wild type beside wild pieces (`Tables::npc_sight`).
    #[inline]
    pub(super) fn wild_sight(&self) -> WildSight {
        if !self.t.npc_sight {
            return WildSight::NONE;
        }
        self.wild_sight_walk()
    }

    #[cold]
    #[inline(never)]
    fn wild_sight_walk(&self) -> WildSight {
        let t = self.t;
        let mut snapshot = WildSight::NONE;
        if t.npc_shield {
            for &square in &t.npc_squares {
                if self.shielded_from_wild(square as usize) {
                    snapshot.shields.set(square as usize);
                }
            }
        }
        if t.npc_tamed {
            let sight = t.sight.as_deref().expect("a tamed phase without the sight");
            for square in self.wild_standing().iter() {
                let bits = self.effects_at(sight, WILD, square);
                if bits & SIGHT_FREEZE != 0 {
                    snapshot.frozen.set(square);
                } else if bits & SIGHT_DISARM != 0 {
                    snapshot.disarmed.set(square);
                }
            }
        }
        snapshot
    }

    /// Generation 84 (SR5f, V11): whether a wild piece that may take what stands on `square`
    /// (a piece of `victim` type, `usize::MAX` for whatever stands there) steps onto it next
    /// and is free to: not frozen or disarmed on this board. `attacks_square_as(WILD, ..)`
    /// with the effects read, as `attacks_in_sight_game` reads a player's; asked only beside
    /// `Tables::npc_tamed`, after the plain test said yes.
    #[cold]
    #[inline(never)]
    pub(super) fn wild_attacks_free(&self, square: usize, victim: usize) -> bool {
        let t = self.t;
        let Some(sight) = t.sight.as_deref() else {
            return true;
        };
        let victim = if victim != usize::MAX || !t.has_capture_matrix {
            victim
        } else {
            let occupant = self.board[square];
            if occupant == 0 { usize::MAX } else { t.victim_of(occupant) }
        };
        t.attackers_of(WILD, square).iter().any(|source| {
            let origin = source.origin as usize;
            self.board[origin] == source.piece_value
                && (!t.has_capture_matrix
                    || victim == usize::MAX
                    || t.may_capture(t.type_of(source.piece_value), victim))
                && self.source_path_clear(WILD, source)
                && self.effects_at(sight, WILD, origin) & (SIGHT_FREEZE | SIGHT_DISARM) == 0
        })
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

    /// SR5b (V2, ruled 2026-10-27: *a shield also shields from a blast*): the squares of the
    /// blast centred on `target` whose piece a shield spares, as if its type were in
    /// `explode_on_capture.immune`. Read once, on the board the capture leaves (the capturer
    /// standing on `target`, nothing cleared yet), so a shield whose source the same blast
    /// clears still spares: the blast is one event. The capturer always goes, as an immune one
    /// does. Asked by `make` and the legality filter alike (a round asks `shields_among` of
    /// its whole blast); empty, at one test, in every game without a shield.
    #[inline(always)]
    pub(super) fn blast_shields(&self, explosion: &ExplosionRule, target: usize) -> SquareSet {
        if !self.t.sight.as_deref().is_some_and(|sight| sight.has_shield) {
            return SquareSet::EMPTY;
        }
        let mut reach = SquareSet::EMPTY;
        for &square in explosion.reach(target) {
            reach.set(square);
        }
        self.shields_among(&reach)
    }

    /// SR5b (V2): of `squares`, those holding a player's piece under a shield on this board.
    /// The one reading of a shield against a blast, for `blast_shields` and a round's
    /// `blasts`, which read every blast of the round at once.
    #[cold]
    #[inline(never)]
    pub(super) fn shields_among(&self, squares: &SquareSet) -> SquareSet {
        let t = self.t;
        let mut shields = SquareSet::EMPTY;
        let Some(sight) = t.sight.as_deref() else {
            return shields;
        };
        for square in squares.iter() {
            if self.piece_effects(sight, square) & SIGHT_SHIELD != 0 {
                shields.set(square);
            }
        }
        shields
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
                && self.sight_path_open(sight, source)
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
        if sight.has_shield && occupant != 0 && self.piece_effects(sight, square) & SIGHT_SHIELD != 0 {
            return false;
        }
        let victim = victim.unwrap_or_else(|| {
            if occupant == 0 || !t.has_capture_matrix { usize::MAX } else { t.victim_of(occupant) }
        });
        for source in t.attackers_of(player, square) {
            if self.board[source.origin as usize] != source.piece_value {
                continue;
            }
            if source.requires_unmoved && !self.unmoved.get(source.origin as usize) {
                continue;
            }
            if t.has_capture_matrix && victim != usize::MAX {
                // Q1074b8 (N4, R1): in a push game a source attacks by capture, by push or
                // both, each under its own list, as `attacks_square_as` reads it.
                let attacker = t.type_of(source.piece_value);
                let reaches = match t.push.as_deref() {
                    None => t.may_capture(attacker, victim),
                    Some(push) => self.push_game_attacks(push, source.dirs, attacker, victim),
                };
                if !reaches {
                    continue;
                }
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
                && self.sight_path_open(sight, source)
            {
                lines.set(source.origin as usize);
                for square in sight.path(source) {
                    lines.set((square & SIGHT_PATH_SQUARE) as usize);
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
            *bits = self.piece_effects(sight, square);
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
        report.suppressed = self.royal_squares(player).iter().any(|&royal| match t.portals.as_deref() {
            None => {
                self.attacks_square(1 - player, royal)
                    && !self.attacks_in_sight_game(sight, 1 - player, royal, None)
            }
            // SR5c (V1): an attack through a portal is an attack the effects may suppress.
            Some(portals) => {
                self.attacks_through_portals(portals, None, 1 - player, royal, None)
                    && !self.attacks_through_portals(portals, Some(sight), 1 - player, royal, None)
            }
        });
        if sight.has_command {
            report.commanded = self
                .legal_moves()
                .iter()
                .filter(|&&m| move_foreign(m))
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
        // Generation 84 (SR5f, V11): and the wild pieces and the duck a sight names.
        for player in 0..if sight.tamed { 3 } else { 2 } {
            for piece_type in 0..t.n_types {
                for (target, bits) in seen.iter_mut().enumerate() {
                    for source in sight.sources_of(player, piece_type, target) {
                        if source.origin as usize == square
                            && source.piece_value == value
                            && self.sight_path_open(sight, source)
                        {
                            *bits |= source.effect;
                        }
                    }
                }
            }
        }
        // SR5c (V1): a line that sees an empty fold that lets this piece in also sees the
        // square of the end that fold leads to, and nothing past it.
        if sight.folds {
            let portals = t.portals.as_deref().expect("folds without portals");
            let owner = t.owner_of(value);
            let direct = seen.clone(); // one fold per line: an exit is no fold's near square
            for end in 0..portals.n_ends {
                let near = self.fold_at[end];
                if near < 0 || self.board[near as usize] != 0 || direct[near as usize] == 0 {
                    continue;
                }
                let entry = portals.first_type + end;
                if owner >= 2 || portals.admits[entry][owner] & (1 << t.type_of(value)) == 0 {
                    continue;
                }
                let to = portals.dest[entry];
                if to < 0 {
                    continue;
                }
                let far = self.fold_at[to as usize - portals.first_type];
                if far >= 0 && far as usize != square {
                    seen[far as usize] |= direct[near as usize];
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
            if game.piece_effects(self.sight, square) & SIGHT_SHIELD != 0 {
                self.shielded.set(square);
            }
        }
        self.shielded.get(square)
    }

    /// Why the effects forbid `m` (`BARRED_*`), or 0.
    fn reason(&mut self, game: &Game<'_>, m: u32) -> u8 {
        let kind = move_kind(m);
        // TC6: a hire or a bribe moves no piece and takes none, so no effect bars it; nor a
        // fold's own move (SR5c), which may move under a piece and takes nothing.
        if enters_play(kind) || kind == MOVE_HIRE || kind == MOVE_FOLD {
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
            // A sweep over a piece takes it. A shielded piece ends the ray in movegen
            // (SR6g, `Game::sweep_shielded`), so a sweep never passes or lands on one.
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
        // Q1074b8 (N4, `docs/plan-push.md` §3.9): a shielded piece may not be pushed, so under
        // a row it blocks the line wherever it stands in it, not only where it is landed on.
        if kind == MOVE_PUSH && self.sight.has_shield && self.shields_line(game, m) {
            return BARRED_SHIELD;
        }
        0
    }

    /// Whether a piece behind the target of the push `m` (the line it moves, read off the
    /// board before it, up to its first empty square or the pusher's origin) is shielded.
    /// The target itself is the caller's. Under `chain: "one"` the square behind is empty.
    #[cold]
    #[inline(never)]
    fn shields_line(&mut self, game: &Game<'_>, m: u32) -> bool {
        let (origin, target) = (move_from(m), move_to(m));
        let mut square = target;
        for _ in 0..game.t.n_squares {
            let Some(behind) = game.push_next(m, square) else {
                return false;
            };
            if behind == origin || behind == target || game.board[behind] == 0 {
                return false;
            }
            if self.shielded(game, behind) {
                return true;
            }
            square = behind;
        }
        false
    }
}

/// Generation 84 (SR5f, V11): what a wild phase reads of the sights, once, at its start
/// (`Game::wild_sight`): the players' pieces under a shield, which no wild piece takes, and
/// the wild pieces standing frozen (which neither walk nor pounce) or disarmed (which walk
/// onto empty squares only and never pounce). By square, read on the board the phase starts
/// from; every wild piece acts once a phase, from its own square, so the snapshot is sound.
#[derive(Clone, Copy)]
pub(super) struct WildSight {
    pub(super) shields: SquareSet,
    pub(super) frozen: SquareSet,
    pub(super) disarmed: SquareSet,
}

impl WildSight {
    pub(super) const NONE: Self =
        Self { shields: SquareSet::EMPTY, frozen: SquareSet::EMPTY, disarmed: SquareSet::EMPTY };
}
