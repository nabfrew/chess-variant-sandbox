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
                // SR8a (R-N5): the type it arrives as, `promote_on_entry` beside `only_lost`
                // included, counted on the board before the move as `make` counts it. Asked
                // outside `make`, so it counts the board itself rather than trust the scratch.
                let tally = if t.entry_only_lost { self.entry_counts() } else { [[0; MAX_PIECE_TYPES]; 2] };
                let arrives = self.arriving_in(&tally, player, move_promotion(m) as usize, target);
                at(t.type_of(arrives), target)
            }
            MOVE_ROTATE => at(t.type_of(self.board[origin]), origin),
            MOVE_CASTLE => at(t.type_of(self.board[origin]), target),
            _ => {
                let mover = t.type_of(self.board[origin]);
                let stays = match t.capture.as_deref().filter(|program| program.general) {
                    Some(program) => self.general_stays(program, m, mover),
                    None => self.strike_stays(kind, mover, target),
                };
                let square = if stays { origin } else { target };
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

    /// R-L4b stage 2 (§3.2, `upkeep.U9`, `upkeep.U10`): whether the moving piece of `mover`
    /// stands on its origin when a `general` capture list has run, read off the board before
    /// the move as the walk would play it (`walk_list`). The cost is the move's own: for the
    /// type the move made it (the caller's promotion), on the square it stands on after the
    /// list, so a link's change of type or a side switch after it is not charged (and the
    /// mover pays). Only a strike or contacts entry's mover half moves the taker (`stay`,
    /// back to an empty origin), and only while it stands: a blast that fires first takes it,
    /// and a piece the list takes away pays for the square it last stood on, its target. The
    /// first entry whose landing link names a change decides (`walk_strike`'s mover half,
    /// once per entry; a change without `stay` leaves it on the target).
    #[cold]
    #[inline(never)]
    fn general_stays(&self, program: &CaptureProgram, m: u32, mover: usize) -> bool {
        let t = self.t;
        let (kind, origin, target, player) = (move_kind(m), move_from(m), move_to(m), self.side);
        let moving = self.board[origin];
        let captured = match kind {
            MOVE_EN_PASSANT => self.board[self.ep_victim as usize],
            MOVE_NORMAL | MOVE_PROMOTION | MOVE_PORTAL | MOVE_SWEEP => self.board[target],
            _ => 0,
        };
        let mut swept: Vec<u8> = Vec::new();
        if kind == MOVE_SWEEP {
            let mut victims = t.sweep_victims(m, moving, player);
            while let Some((_, value)) = victims.next_on(&self.board) {
                swept.push(value);
            }
        }
        // A gate fills the origin in part one (§3.2), so a `stay` finds it full.
        let gated = m & MOVE_TAG_MASK != 0
            && t.drops.as_ref().is_some_and(|rule| rule.gates.is_some())
            && move_gate(m) != 0
            && !move_gate_rook(m);
        let mover_owner = t.owner_of(moving);
        for entry in &program.entries {
            match entry.kind {
                CaptureKind::Pay | CaptureKind::Hand => {}
                CaptureKind::Blast => {
                    let landing = entry.on & ON_LANDING != 0
                        && self.blast_sets_off(mover, captured, player);
                    let path = entry.reads_path(mover)
                        && swept.iter().any(|&value| self.blast_sets_off(mover, value, player));
                    if landing || path {
                        return false;
                    }
                }
                CaptureKind::Strike | CaptureKind::Contacts => {
                    if entry.on & ON_LANDING == 0 || captured == 0 {
                        continue;
                    }
                    let Some(strike) = t.strike.as_ref() else {
                        continue;
                    };
                    let rule = program.table(entry, strike);
                    let owner = t.owner_of(captured);
                    if owner == mover_owner && mover_owner != player {
                        continue;
                    }
                    let cell = mover * rule.n_types + t.type_of(captured);
                    let (to, mover_to, flips) = if owner == mover_owner {
                        (
                            rule.own_to.get(cell).copied().unwrap_or(-1),
                            rule.own_mover_to.get(cell).copied().unwrap_or(-1),
                            false,
                        )
                    } else {
                        (rule.to[cell], rule.mover_to[cell], rule.flips(cell))
                    };
                    let stay = (to != -1 || mover_to >= 0) && rule.position == StrikePosition::Stay;
                    if mover_to >= 0 || flips || stay {
                        return stay && !gated;
                    }
                }
            }
        }
        false
    }

    /// What the purse filter charges `m`: `move_cost`, and a buy's price as well. The one
    /// reading of a move's price that `upkeep_filter` and `short_of_dearest` share.
    #[inline(always)]
    fn purse_cost(&self, purse: &PurseTables, m: u32) -> u16 {
        let mut cost = self.move_cost(purse, m);
        if move_kind(m) == MOVE_BUY {
            cost += purse.price[move_promotion(m) as usize];
        }
        cost
    }

    /// T1d's second arm (`agents::AFFORD`, `docs/plan-player-weaknesses.md` §T1d): whether the
    /// side to move's purse is below the dearest cost, as the purse filter reads it
    /// (`purse_cost`), among its **pseudo-legal** moves (`generate_pseudo_legal`: before the
    /// check filter and the purse filter). `scratch` is appended to and truncated back to its
    /// length. False without `rules.income` or upkeep. Mirrored by
    /// `agents/minimax.py::_short_of_dearest`.
    #[inline(never)]
    pub fn short_of_dearest(&self, scratch: &mut Vec<u32>) -> bool {
        let t = self.t;
        let Some(purse) = t.purse.as_deref() else {
            return false;
        };
        if purse.upkeep.is_empty() || t.income.is_none() {
            return false;
        }
        let coins = self.purse[self.side];
        let base = scratch.len();
        self.generate_pseudo_legal(scratch);
        let short = scratch[base..].iter().any(|&m| self.purse_cost(purse, m) > coins);
        scratch.truncate(base);
        short
    }

    /// The side to move's moves from `base` less those its purse cannot pay for (a buy its
    /// price as well). No lift: a move the purse cannot pay for is never offered back.
    #[inline(never)]
    pub(super) fn upkeep_filter(&self, purse: &PurseTables, out: &mut Vec<u32>, base: usize) {
        let coins = self.purse[self.side];
        let mut write = base;
        for index in base..out.len() {
            let m = out[index];
            let cost = self.purse_cost(purse, m);
            if cost <= coins {
                out[write] = m;
                write += 1;
            }
        }
        out.truncate(write);
    }
}
