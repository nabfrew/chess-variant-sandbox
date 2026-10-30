//! Wild pieces that come back (generation 36, N1, `docs/archive/plan-npc.md` §4.1): step 3 of the
//! wild-piece phase, after the steps.
//!
//! **The key's state.** One clock per respawning half (`Tables::respawn`'s order: `rules.npcs`
//! order, a mirrored half after its original), `Game::respawn`: 0 idle, else the phases until
//! a return is tried, counting the coming one. It is state the board does not show, so it is
//! in the key (`RespawnTables::key`, drawn last of all), on `Snapshot`, across the boundary
//! (`GameState.respawn`, slot 11's tenth element), and logged by `make` on `respawn_log`
//! only under the rule, so `Undo` does not grow. The later wild-piece rules that keep state
//! (a heading, a cap) are meant to take this shape.
//!
//! **The clock reads the board.** A half's clock starts when fewer of its pieces stand on
//! its route than it has starts, whatever took one: the phase notices, so `make`, the
//! legality filter and `coming_phases` play the one rule (`respawn_due`, walked by
//! `phase.rs::respawn_walk` for all three) and a position
//! built by hand with a piece missing is well defined. Started at `every` by the capture,
//! it is counted down by the phase that follows it; at 0 the first start that is empty or
//! holds a player's non-royal piece the wild type may take gets the piece back (eating
//! that one), and otherwise the return is held and tried every phase. After a return the
//! clock starts again at `every` while the half still stands short.

use super::*;

impl<'t> Game<'t> {
    /// How many of `half`'s pieces stand: the wild pieces on its route.
    #[inline]
    fn respawn_standing(&self, half: &RespawnHalf) -> usize {
        let t = self.t;
        half.route
            .iter()
            .filter(|square| {
                // Generation 42: the duck on a route square is not one of its pieces.
                t.is_wild_piece(self.board[**square as usize])
            })
            .count()
    }

    /// The first of `half`'s starts a piece may come back onto: empty, or a player's
    /// non-royal piece the wild type may capture. A royal, a piece it may not take, a
    /// piece in `shields` (generation 37: the phase's snapshot of the shielded squares, a
    /// wild piece never takes a shielded piece) and another wild piece each hold the start.
    #[inline]
    fn respawn_start(&self, half: &RespawnHalf, shields: &SquareSet) -> Option<usize> {
        let t = self.t;
        let wild_type = t.type_of(half.value);
        half.starts.iter().map(|square| *square as usize).find(|square| {
            let standing = self.board[*square];
            if standing == 0 {
                return true;
            }
            if t.owner_of(standing) == WILD || shields.get(*square) {
                return false;
            }
            let victim = t.type_of(standing);
            !t.is_royal(victim) && t.may_capture(wild_type, victim)
        })
    }

    /// **The rule, stated once**: what `half` does this phase on the board as it stands
    /// (after the steps), its clock being `clock` before the phase. Returns the clock after
    /// the phase and the start a piece comes back onto, if one does. `shields` is the
    /// phase's snapshot of the shielded squares (`Game::wild_shields`), read at its start.
    pub(super) fn respawn_due(
        &self, index: usize, clock: u8, shields: &SquareSet,
    ) -> (u8, Option<usize>) {
        let rule = self.t.respawn.as_deref().expect("respawn_due without the rule");
        let half = &rule.halves[index];
        let standing = self.respawn_standing(half);
        let mut left = clock;
        if left == 0 {
            if standing >= half.starts.len() {
                return (0, None);
            }
            left = half.every;
        }
        left -= 1;
        if left > 0 {
            return (left, None);
        }
        match self.respawn_start(half, shields) {
            None => (1, None),
            Some(start) => {
                let next = if standing + 1 < half.starts.len() { half.every } else { 0 };
                (next, Some(start))
            }
        }
    }

    /// The clocks' part of the key; 0 in a game without the rule or with every clock idle.
    pub(super) fn respawn_key(&self) -> u64 {
        let Some(rule) = self.t.respawn.as_deref() else {
            return 0;
        };
        self.respawn
            .iter()
            .enumerate()
            .fold(0, |key, (index, clock)| key ^ rule.key(index, *clock))
    }

    /// Log the clocks before a move's phase (`make`), under the rule only.
    #[inline]
    pub(super) fn respawn_save(&mut self) {
        self.respawn_log.extend_from_slice(&self.respawn);
    }

    /// Put back the clocks `respawn_save` logged (`unmake`).
    #[inline]
    pub(super) fn respawn_unsave(&mut self) {
        let at = self.respawn_log.len() - self.respawn.len();
        self.respawn.copy_from_slice(&self.respawn_log[at..]);
        self.respawn_log.truncate(at);
    }
}

/// The clocks as Python holds them, sized to this game's halves: an empty tuple (a position
/// built by hand) is every clock idle, and a clock longer than its half's `every` is cut to
/// it.
pub fn respawn_clocks_from(t: &Tables, given: &[u8]) -> Vec<u8> {
    let Some(rule) = t.respawn.as_deref() else {
        return Vec::new();
    };
    rule.halves
        .iter()
        .enumerate()
        .map(|(index, half)| given.get(index).copied().unwrap_or(0).min(half.every))
        .collect()
}
