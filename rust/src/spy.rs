//! The spy (schema 27, Q935a): a game whose turn structure is decided by who still has a
//! spy on the board.
//!
//! `docs/plan-spy.md` is the rule; this is its §2.5. While both players have a spy, play
//! alternates exactly as it does now (`make`). When one is lost, the player who lost it
//! plays one ordinary **reply** turn if it is theirs to play; from then on play is in
//! **announced rounds**: the player without a spy chooses first, the move is shown to the
//! other, and both are resolved as a round (`make_round`). Once neither has a spy, play is in
//! plain blind rounds. The phase is a function of the board and the side to move, never
//! state, read afresh after every turn and every round.
//!
//! SR7h (`docs/plan-schema-next.md` §2.12, *The gained spy, said once*): a spy can be gained
//! (a promotion into one, Y3), so the phase moves both ways. A turn needs nothing new: `make`
//! flips the side as ever and the table below reads the board it leaves (a reply turn that
//! gains back leaves an alternating position, the other player to move; a second spy is a
//! buffer, since the count is summed). A round writes `side` (`after_round`, undone with the
//! round's snapshot): after it, both with a spy is turns again, **the player who did not gain
//! to move** (both gained: the first player); one with a spy is announced rounds, the holder
//! to move (so the other announces); neither is blind rounds, the side left where it stood.
//!
//! The one other write is the initiative's deal (E10): under `tie: "initiative"` the token goes
//! to the announcer each time announced rounds begin after a stretch without them, from a turn
//! (`spy_turn`; `make` is untouched, the deal sits beside it and is undone by
//! `unmake_spy_turn`) or from a blind round (`after_round`). A round that keeps the phase
//! announced, the spy changing hands, deals nothing: announced rounds did not begin there.
//!
//! `tests/spy_oracle.py` is the declared oracle.

use super::*;

/// Which kind of step a spy game's position is played by (§1.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Both spies stand: an ordinary alternating turn.
    Alternating,
    /// The side to move has just lost its spy and the other still has one: its one
    /// ordinary reply turn before rounds begin.
    Reply,
    /// Only the side to move has a spy: a round in which `announcer`, the player without
    /// one, chooses first and in the open.
    Announced { announcer: usize },
    /// Neither has a spy: a round chosen in secret.
    Blind,
}

impl Phase {
    /// The phase's name at the boundary (`Core.phase`).
    pub fn name(self) -> &'static str {
        match self {
            Phase::Alternating => "alternating",
            Phase::Reply => "reply",
            Phase::Announced { .. } => "announced",
            Phase::Blind => "blind",
        }
    }

    /// Whether the step is a round (both players move), not a turn.
    pub fn is_round(self) -> bool {
        matches!(self, Phase::Announced { .. } | Phase::Blind)
    }

    /// The announcer, or -1 outside an announced round.
    pub fn announcer(self) -> i8 {
        match self {
            Phase::Announced { announcer } => announcer as i8,
            _ => -1,
        }
    }
}

impl<'t> Game<'t> {
    /// Whether `side` has a spy: at least one piece of any of its spy types on the board,
    /// summed over the types (§1.2: a side given two spies has lost "the spy" when it has
    /// none left). Wild pieces are nobody's and are not in `occupied()[side]`.
    pub fn has_spy(&self, side: usize) -> bool {
        let spy = self.t.spy.as_ref().expect("a spy question in a game without a spy");
        let mask = spy.mask[side];
        mask != 0
            && self.occupied()[side]
                .iter()
                .any(|square| mask & (1u32 << self.t.type_of(self.board[square])) != 0)
    }

    /// This position's phase (§1.3's table), from who has a spy and whose turn it is.
    pub fn phase(&self) -> Phase {
        match [self.has_spy(0), self.has_spy(1)] {
            [true, true] => Phase::Alternating,
            [false, false] => Phase::Blind,
            [first, _] => {
                let holder = if first { 0 } else { 1 };
                if self.side == holder {
                    Phase::Announced { announcer: 1 - holder }
                } else {
                    Phase::Reply
                }
            }
        }
    }

    /// One alternating or reply turn of a spy game: `make`, then the initiative's deal if
    /// the position it leaves is the first announced round (E10). Every alternating step of
    /// a spy game goes through here, so the phase before it was alternating or reply and an
    /// announced phase after it is the first.
    pub fn spy_turn(&mut self, m: u32) {
        debug_assert!(!self.phase().is_round(), "a turn in a round phase");
        let before = self.initiative;
        self.make(m);
        let deal = self.t.spy.as_ref().is_some_and(|spy| spy.deal);
        if deal {
            if let Phase::Announced { announcer } = self.phase() {
                let to = announcer as i8;
                if to != before {
                    let sim = self.sim_tables();
                    if before == 1 {
                        self.key ^= sim.zobrist_initiative;
                    }
                    if to == 1 {
                        self.key ^= sim.zobrist_initiative;
                    }
                    self.initiative = to;
                    if let Some(last) = self.history.last_mut() {
                        *last = self.key;
                    }
                }
            }
        }
        self.spy_deals.push(before);
    }

    /// Take back the last `spy_turn`.
    pub fn unmake_spy_turn(&mut self) {
        let before = self.spy_deals.pop().expect("unmake_spy_turn without a matching spy_turn");
        self.initiative = before;
        self.unmake();
    }

    /// Who has a spy, `[player 0, player 1]`: what `make_round` reads before a round of a
    /// spy game, so that `after_round` can tell a gain (SR7h).
    pub(super) fn spies(&self) -> [bool; 2] {
        [self.has_spy(0), self.has_spy(1)]
    }

    /// SR7h (§2.12): the side to move a round of a spy game leaves, and the initiative's deal
    /// where the round begins announced rounds after blind ones; `had` is who had a spy before
    /// the round. Returns the key's change for the side (the deal is the standoff key's, which
    /// `make_round` reads after this). A round's undo is a snapshot, so nothing is logged.
    pub(super) fn after_round(&mut self, had: [bool; 2]) -> u64 {
        let has = self.spies();
        let side = match has {
            // Turns again: the player who did not gain moves first, answering the gain;
            // both gained, the first player, as at the start of a game.
            [true, true] => usize::from(!had[0] && had[1]),
            // Announced rounds: the holder answers, so the other announces.
            [true, false] => 0,
            [false, true] => 1,
            [false, false] => self.side,
        };
        let deal = self.t.spy.as_ref().is_some_and(|spy| spy.deal);
        if deal && had == [false, false] && has[0] != has[1] {
            // Announced rounds begin after blind ones: the token to the announcer (E10).
            self.initiative = (1 - side) as i8;
        }
        if side == self.side {
            return 0;
        }
        self.side = side;
        self.t.zobrist_side
    }

    fn sim_tables(&self) -> &'t SimTables {
        self.t.simultaneous.as_deref().expect("a spy game without rules.simultaneous")
    }

    /// The half of a spy position's verdict that needs no move lists: the alternating
    /// game's while it alternates, a round's once it is rounds (§1.5 E12).
    pub fn spy_terminal_prefix(&mut self) -> Option<Outcome> {
        if self.phase().is_round() {
            self.round_terminal_prefix()
        } else {
            self.terminal_prefix()
        }
    }

    /// The whole verdict, generating the lists the phase needs.
    pub fn spy_terminal(&mut self) -> Option<Outcome> {
        if self.phase().is_round() {
            self.round_terminal()
        } else {
            self.terminal()
        }
    }

    /// The whole verdict where a round's two lists are already in hand (SR7h, the matrix
    /// search's leaf): a round's from them, a turn's from its own list.
    pub fn spy_terminal_with(&mut self, lists: [&[u32]; 2]) -> Option<Outcome> {
        if self.phase().is_round() {
            self.round_terminal_with(lists)
        } else {
            self.terminal()
        }
    }

    /// Perft over a spy game (§2.5): a step is one ply while play alternates and one
    /// round (every pair of legal moves) once it is rounds, the step's kind read per node.
    pub fn perft_spy(&mut self, depth: u32, stop_at_terminal: bool) -> u64 {
        if depth == 0 {
            return 1;
        }
        if stop_at_terminal && self.spy_terminal_prefix().is_some() {
            return 0;
        }
        if self.phase().is_round() {
            let mut first = Vec::new();
            let mut second = Vec::new();
            self.generate_legal_for(0, &mut first);
            self.generate_legal_for(1, &mut second);
            if depth == 1 {
                return (first.len() * second.len()) as u64;
            }
            let mut total = 0u64;
            for &a in &first {
                for &b in &second {
                    self.make_round([a, b]);
                    total += self.perft_spy(depth - 1, stop_at_terminal);
                    self.unmake_round();
                }
            }
            return total;
        }
        let mut moves = Vec::new();
        self.generate_legal(&mut moves);
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut total = 0u64;
        for &m in &moves {
            self.spy_turn(m);
            total += self.perft_spy(depth - 1, stop_at_terminal);
            self.unmake_spy_turn();
        }
        total
    }
}
