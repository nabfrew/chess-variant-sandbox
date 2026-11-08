//! A move's cost (H1, `docs/plan-hunger.md` §2 and §R1, `pieces[].upkeep`,
//! `rules.purse.upkeep`).
//!
//! **The table.** The compiler resolved the measure into `PurseTables::upkeep`, what a move costs
//! its player that ends with its piece of a type on a square; the core reads a number and never
//! knows which measure made it (`rows` in H1). Empty in every other game, which pays one test.
//!
//! **What pays** (`move_cost`, D4 re-cut by Q1130): a move pays for the square its piece stands
//! on after it, as the type it is after it. A step, en passant, a sweep and a displacement by
//! their target; a castle the castling piece's target only (its partner travels free, as the
//! pieces a displacement shifts do); a rotation and a capture a strike placed `stay` keeps on its
//! origin by the origin; a promotion both arrivals (`"both"`, D5) or the promoted type's alone
//! (`"new"`); a buy its price plus its arrival's cost (the bought piece as it arrives, so a
//! promotion on entry is the type it becomes); a hire, a keep, a fold's move and the duck's
//! nothing. Read off the word and the board **before** the move, so the filter and `make` ask the
//! one function. Every rule whose arrival is not read this way is refused beside the rule
//! (`bridge._upkeep_holds`, `Tables::from_py`): a foreign move, a mover a meeting changes or
//! turns, a transit, a drop, a push, rounds, a long turn.
//!
//! **The filter** (`upkeep_filter`) drops every move the side to move's purse cannot pay for,
//! after the check filter and before the rest (`generate_checked`'s tail), so a rest that would
//! leave no affordable move is lifted as one that would leave no legal move is. The booty a
//! capture brings is paid after (D2): the purse before the move must cover it. Attack is
//! geometry, never coins (D3): check detection reads no purse.
//!
//! **The payment** is `make`'s, under the purse's log and key with the buy's price and the hire's
//! cost (`Played::spent`), before the bounty; `unmake` reads the log. No new state.

use super::*;

impl<'t> Game<'t> {
    /// `move_cost` for a caller outside the module (`Core.move_cost`): 0 without the rule.
    pub fn upkeep_cost(&self, m: u32) -> u16 {
        match self.t.purse.as_deref() {
            Some(purse) if !purse.upkeep.is_empty() => self.move_cost(purse, m),
            _ => 0,
        }
    }

    /// What `m` costs the side to move under `rules.purse.upkeep`, read off the word and the
    /// board before it is played. A buy's price is not in it (`upkeep_filter` adds it; `make`
    /// already spends it).
    #[inline]
    pub(super) fn move_cost(&self, purse: &PurseTables, m: u32) -> u16 {
        let t = self.t;
        let kind = move_kind(m);
        let player = self.side;
        let origin = move_from(m);
        let target = move_to(m);
        let at = |piece_type: usize, square: usize| {
            purse.upkeep_at(t.n_types, t.n_squares, player, piece_type, square)
        };
        match kind {
            // H3 (D4): a pass places nothing, so it pays nothing.
            MOVE_HIRE | MOVE_KEEP | MOVE_DUCK | MOVE_FOLD | MOVE_PASS => 0,
            MOVE_BUY | MOVE_DROP => {
                let arrives = t.arriving_value(player, move_promotion(m) as usize, target);
                at(t.type_of(arrives), target)
            }
            MOVE_ROTATE => at(t.type_of(self.board[origin]), origin),
            MOVE_CASTLE => at(t.type_of(self.board[origin]), target),
            _ => {
                let mover = t.type_of(self.board[origin]);
                let square = if self.strike_stays(kind, mover, target) { origin } else { target };
                let promoted = match kind {
                    MOVE_PROMOTION => Some(move_promotion(m) as usize),
                    MOVE_DISPLACE | MOVE_PUSH | MOVE_PORTAL => (move_promotion(m) as usize).checked_sub(1),
                    _ => None,
                };
                match promoted {
                    None => at(mover, square),
                    Some(new) if purse.upkeep_new => at(new, square),
                    Some(new) => at(mover, square) + at(new, square),
                }
            }
        }
    }

    /// Whether the moving piece of `mover` stays on its origin: a capture a strike placed
    /// `stay` reaches (rifle), not one a blast takes. The meeting's tables as `strike_writes`
    /// reads them; a strike that changes or turns the mover is refused beside the rule, so a
    /// struck meeting is one whose victim's entry is not -1.
    #[inline]
    fn strike_stays(&self, kind: u32, mover: usize, target: usize) -> bool {
        let t = self.t;
        let Some(strike) = t.strike.as_ref() else {
            return false;
        };
        if strike.position != StrikePosition::Stay {
            return false;
        }
        let captured = if kind == MOVE_EN_PASSANT {
            self.board[self.ep_victim as usize]
        } else {
            self.board[target]
        };
        if captured == 0 {
            return false;
        }
        if t.explosion.is_some() && (kind == MOVE_SWEEP || self.blast_sets_off(mover, captured, self.side)) {
            return false;
        }
        let cell = mover * strike.n_types + t.type_of(captured);
        let table = if t.owner_of(captured) == self.side { &strike.own_to } else { &strike.to };
        table.get(cell).copied().unwrap_or(-1) != -1
    }

    /// The side to move's moves from `base` less those its purse cannot pay for (a buy its
    /// price as well). No lift: a move the purse cannot pay for is never offered back.
    #[inline(never)]
    pub(super) fn upkeep_filter(&self, purse: &PurseTables, out: &mut Vec<u32>, base: usize) {
        let coins = self.purse[self.side];
        let mut write = base;
        for index in base..out.len() {
            let m = out[index];
            let mut cost = self.move_cost(purse, m);
            if move_kind(m) == MOVE_BUY {
                cost += purse.price[move_promotion(m) as usize];
            }
            if cost <= coins {
                out[write] = m;
                write += 1;
            }
        }
        out.truncate(write);
    }
}
