//! Rest as a piece-type property (schema 29, Q983a): the one rest in the core, read by the
//! alternating generator and the round generator alike.
//!
//! `docs/plan-rest.md` is the rule (§3) and its build (§4). A piece that makes its player's
//! move rests for that player's next *n* turns, *n* its type's (`RestTables::turns`), where a
//! turn is an alternating turn of that player or a round. The state is each player's short
//! list of resting pieces, `(square, turns left, the piece code that rested there)`; it is in
//! the key, keyed per (turns left, player, square). Two calls are the whole of it:
//!
//! * `rest_filter`, from `generate_legal` (after the check filter) and `generate_legal_for`
//!   (after the bars), before `restrict`: a resting piece's moves are not offered, unless
//!   that would leave none (the lift). Its caller recomputes the list without the rest if
//!   `restrict` then empties a list the filter shortened, so the rest alone never leaves a
//!   player without a move (§3.7).
//! * `rest_after_turn`, from `make` (and `make_fold`) at the end of a player's turn and from
//!   `make_round` once per player: count that player's entries down, drop any entry whose
//!   square no longer holds the code that rested there (either player's: the opponent's move
//!   can end a rest, §3.4), then enter the piece standing where the move ended, as the type it
//!   now is (§3.3).
//!
//! Every game without a rest has `Tables::rest` `None` and pays one test at each call site.
//! `tests/rest_oracle.py` is the declared oracle.

use super::*;

/// One resting piece: where it stands, how many of its player's turns it still rests, and
/// the board value that stood there when the rest began. `square` is -1 in an unused slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Resting {
    pub square: i16,
    pub left: u8,
    pub code: u8,
}

impl Resting {
    pub const NONE: Resting = Resting { square: NO_SQUARE, left: 0, code: 0 };

    #[inline]
    pub fn live(&self) -> bool {
        self.square >= 0
    }
}

/// Each player's resting pieces, `REST_MAX` slots each, unused slots `Resting::NONE`.
pub type RestLists = [[Resting; REST_MAX]; 2];

pub const NO_REST: RestLists = [[Resting::NONE; REST_MAX]; 2];

impl<'t> Game<'t> {
    /// Drop from `out[base..]` every move of one of `player`'s resting pieces (never a drop,
    /// and never a fold's own move, which moves no piece), unless that would leave none: the
    /// lift (§3.7). Returns whether anything was dropped.
    pub(super) fn rest_filter(&self, player: usize, out: &mut Vec<u32>, base: usize) -> bool {
        // The live squares first, so each move pays a comparison or two rather than a walk
        // of every slot: under a rest of 1 a player has at most one entry, and breather's
        // rounds bench within 3% of schema 27's one square only so (Q983a's benches).
        let slots = self.t.rest.as_deref().map_or(REST_MAX, |rest| rest.slots);
        let mut squares = [usize::MAX; REST_MAX];
        let mut live = 0;
        for entry in self.resting[player][..slots].iter().filter(|e| e.live()) {
            squares[live] = entry.square as usize;
            live += 1;
        }
        match live {
            0 => false,
            1 => drop_resting(out, base, |from| from == squares[0]),
            _ => drop_resting(out, base, |from| squares[..live].contains(&from)),
        }
    }

    /// The end of `player`'s turn under the rest: count `player`'s entries down, drop every
    /// entry (either player's) whose square no longer holds the code that rested there, and
    /// enter the piece now standing on `end` (the square the move ended on, `None` where no
    /// piece made it or none is left) if it is `player`'s and its type rests. Returns the
    /// change to the key. Out of line, so `make` stays its size in a game without a rest.
    #[inline(never)]
    pub(super) fn rest_after_turn(&mut self, player: usize, end: Option<usize>) -> u64 {
        let t = self.t;
        let rest = t.rest.as_deref().expect("a rest update without a rest");
        let n = t.n_squares;
        // The piece that enters, decided first: the board does not change here.
        let entering = end.and_then(|square| {
            let value = self.board[square];
            if value == 0 || t.owner_of(value) != player {
                return None;
            }
            let turns = rest.turns[t.type_of(value)];
            (turns != 0).then_some((square, value, turns))
        });
        // One pass over both lists: `player`'s entries count down, and an entry leaves when
        // its count ends, when its square no longer holds the code that rested there, or
        // when it is `player`'s on the square a piece enters from (a turn in place, a contact
        // that stays, under the lift: it rests afresh, one entry a square).
        let entering_square = entering.map_or(usize::MAX, |(square, _, _)| square);
        let mut delta = 0u64;
        for owner in 0..2 {
            let own = owner == player;
            for entry in self.resting[owner][..rest.slots].iter_mut() {
                if !entry.live() {
                    continue;
                }
                let square = entry.square as usize;
                let left = entry.left - own as u8;
                if left == 0
                    || self.board[square] != entry.code
                    || (own && square == entering_square)
                {
                    delta ^= rest.key(n, owner, square, entry.left);
                    *entry = Resting::NONE;
                } else if own {
                    delta ^= rest.key(n, owner, square, entry.left) ^ rest.key(n, owner, square, left);
                    entry.left = left;
                }
            }
        }
        let Some((square, value, turns)) = entering else {
            return delta;
        };
        let slot = self.resting[player][..rest.slots]
            .iter_mut()
            .find(|e| !e.live())
            .expect("more resting pieces than the longest rest: a long turn beside rest");
        *slot = Resting { square: square as i16, left: turns, code: value };
        delta ^ rest.key(n, player, square, turns)
    }

    /// The part of the key the resting pieces contribute (`compute_hash`).
    pub(super) fn rest_key(&self) -> u64 {
        let Some(rest) = self.t.rest.as_deref() else {
            return 0;
        };
        let mut key = 0u64;
        for (player, entries) in self.resting.iter().enumerate() {
            for entry in entries.iter().filter(|e| e.live()) {
                key ^= rest.key(self.t.n_squares, player, entry.square as usize, entry.left);
            }
        }
        key
    }

    /// What the rest does to `player`'s choice on this board (Q983b, `docs/plan-rest.md` §5.4,
    /// §5.6, §7): `(binds, lifted)`. `binds` when `player`'s legal moves differ from those
    /// on the same board with nothing resting, so the rest took away at least one move;
    /// `lifted` when they do not, yet a resting piece's move (never a drop's, never a fold's
    /// own) is among them, which the rule allows only when the rest would leave no move
    /// (§3.7). Never both. The list is the one the game offers here: a round's for `player`
    /// in a game in rounds or a spy game's round, the side to move's otherwise (asked for
    /// `player`, as `player` would see it). Both false in a game without a rest. For the
    /// analysis pass's tallies and the page's sentence: Python asks, the core answers.
    pub fn rest_turn(&mut self, player: usize) -> (bool, bool) {
        let Some(slots) = self.t.rest.as_deref().map(|rest| rest.slots) else {
            return (false, false);
        };
        let round = if self.t.spy.is_some() {
            self.phase().is_round()
        } else {
            self.t.simultaneous.is_some()
        };
        let side = self.side;
        if !round {
            self.side = player;
        }
        let mut held = self.rest_turn_list(player, round);
        let resting = self.resting;
        self.resting = NO_REST;
        let mut free = self.rest_turn_list(player, round);
        self.resting = resting;
        self.side = side;
        held.sort_unstable();
        free.sort_unstable();
        if held != free {
            return (true, false);
        }
        let entries = &self.resting[player][..slots];
        let lifted = free.iter().any(|&m| {
            let kind = move_kind(m);
            kind != MOVE_DROP
                && kind != MOVE_FOLD
                && entries.iter().any(|e| e.live() && e.square as usize == move_from(m))
        });
        (false, lifted)
    }

    fn rest_turn_list(&mut self, player: usize, round: bool) -> Vec<u32> {
        let mut moves = Vec::new();
        if round {
            self.generate_legal_for(player, &mut moves);
        } else {
            self.generate_legal(&mut moves);
        }
        moves
    }

    /// `player`'s resting pieces as `(square, turns left)`, by square: the boundary's shape.
    pub fn resting_of(&self, player: usize) -> Vec<(i16, u8)> {
        let mut out: Vec<(i16, u8)> = self.resting[player]
            .iter()
            .filter(|e| e.live())
            .map(|e| (e.square, e.left))
            .collect();
        out.sort_unstable();
        out
    }
}

/// `rest_filter`'s one pass: keep in `out[base..]` the drops, the folds' own moves and every
/// move whose origin `resting_from` refuses. Moves are written back only when kept, so when
/// nothing is kept `out` is untouched and the lift is simply returning `false`.
#[inline(always)]
fn drop_resting(out: &mut Vec<u32>, base: usize, resting_from: impl Fn(usize) -> bool) -> bool {
    let before = out.len();
    let mut write = base;
    for index in base..before {
        let m = out[index];
        let kind = move_kind(m);
        if kind == MOVE_DROP || kind == MOVE_FOLD || !resting_from(move_from(m)) {
            out[write] = m;
            write += 1;
        }
    }
    if write == base {
        return false;
    }
    out.truncate(write);
    write != before
}

/// The resting lists a position from outside the core carries, `(square, turns left)` per
/// player, with each entry's code read off `board` (the piece standing there now is the one
/// resting). Empty in a game without a rest; an entry past the longest rest's count of
/// them (`RestTables::slots`), off the board, on an empty square or resting longer than any
/// type rests is dropped.
pub fn rest_lists_from(t: &Tables, board: &[u8; MAX_SQUARES], given: [&[(i16, u8)]; 2]) -> RestLists {
    let mut lists = NO_REST;
    let Some(rest) = t.rest.as_deref() else {
        return lists;
    };
    let longest = rest.slots as u8;
    for (player, entries) in given.iter().enumerate() {
        let mut slot = 0;
        for &(square, left) in entries.iter() {
            if slot == rest.slots
                || square < 0
                || square as usize >= t.n_squares
                || left == 0
                || left > longest
            {
                continue;
            }
            let code = board[square as usize];
            if code == 0 {
                continue;
            }
            lists[player][slot] = Resting { square, left, code };
            slot += 1;
        }
    }
    lists
}
