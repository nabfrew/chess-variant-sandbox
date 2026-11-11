//! The learned evaluator (Q502e): integer inference over a stored net, with an
//! accumulator.
//!
//! The design record is `docs/reference/learned-evaluator.md`; this file is its
//! arithmetic, and `chess_variant_sandbox/agents/nets.py` is the declared test oracle
//! that must agree with it bit for bit (`tests/test_nets.py`). Change one only with the
//! other, and bump `INPUTS_VERSION` or the store's format version when either moves.
//!
//! **What the net reads** is the position as the core holds it, laid out by `Layout`
//! from the tables' sizes alone -- no rule is interpreted here, no piece is named, and
//! nothing computes a square: a feature is "this board byte on this square index",
//! "this square's `unmoved` bit", "this many of that type in hand". A rule the DSL adds
//! later that adds *state* needs a segment (and an `INPUTS_VERSION` bump); a rule that
//! only changes what the existing state means needs nothing.
//!
//! **Exact on every machine.** Every step is integer: an `i16` accumulator with
//! wrapping arithmetic (a group, so the incremental sum always equals a fresh one),
//! clipped to `[0, 127]`; two `i8` layers summed in `i32` (exact by `DOT_BOUND`), biased
//! in `i64`, shifted and clipped; an `i8` output layer; then one division by a power of
//! two into the search's `f64`.
//!
//! **The accumulator** is lazy: `NetEvaluator` keeps the sorted list of active inputs it
//! last summed, and at the next evaluation subtracts the rows that went away and adds the
//! rows that arrived. Consecutive evaluations in a search are neighbouring leaves, so the
//! difference is a handful of rows. It needs no hook in `Game::make`/`unmake`, so a
//! player without a net pays nothing for this file.

use crate::game::Game;
use crate::tables::{Tables, PURSE_MAX};

/// The largest `|sum|` of one dense-layer dot product any accepted store can reach: at most
/// `MAX_ACCUMULATOR` terms (the first layer; the second has at most `MAX_HIDDEN`), each an
/// `i8` weight (`|w| <= 128`) times an activation in `[0, ACTIVATION_MAX]`.
/// 2,048 * 128 * 127 = 33,292,288, under 2^25, so `dot` may sum in `i32` and agree with an
/// `i64` sum bit for bit (Q961). Widening a layer past this is a compile error here.
pub const DOT_BOUND: i64 = MAX_ACCUMULATOR as i64 * 128 * ACTIVATION_MAX;
const _: () = assert!(DOT_BOUND <= i32::MAX as i64 && MAX_HIDDEN <= MAX_ACCUMULATOR);

/// One dense layer's dot product, `i8` weights against activations in `[0, ACTIVATION_MAX]`,
/// summed in `i32`: exact by `DOT_BOUND`. The arithmetic the sparse pass (`sparse_layer`)
/// must equal; kept for the tests that say so.
#[cfg(test)]
fn dot(weights: &[i8], activations: &[i16]) -> i32 {
    weights.iter().zip(activations).map(|(&w, &a)| w as i32 * a as i32).sum()
}

/// Output lanes the sparse pass sums at once, held in registers across every input (Q1120):
/// every rated net's hidden layers are 32 wide, one block. A layer's columns are padded
/// with zero weights to a multiple of it.
const BLOCK: usize = 32;
const _: () = assert!(MAX_HIDDEN.is_multiple_of(BLOCK));

/// The largest `|w * a|` of one product: an `i8` weight times an activation in
/// `[0, ACTIVATION_MAX]`, 128 * 127 = 16,256. Two of them sum to 32,512, inside `i16`, so the
/// sparse pass adds a pair of products in `i16` and widens once (Q1120).
pub const PAIR_BOUND: i64 = 2 * 128 * ACTIVATION_MAX;
const _: () = assert!(PAIR_BOUND <= i16::MAX as i64);

/// A dense layer's weights `[k * inputs + j]` (output `k`, input `j`) as columns, one
/// `BLOCK` of outputs at a time: `[(k / BLOCK * inputs + j) * BLOCK + k % BLOCK]`, widened to
/// `i16`, the padding past the last output zero (Q1120).
fn columns(weights: &[i8], outputs: usize, inputs: usize) -> Vec<i16> {
    let mut out = vec![0i16; outputs.div_ceil(BLOCK) * inputs * BLOCK];
    for k in 0..outputs {
        for j in 0..inputs {
            out[(k / BLOCK * inputs + j) * BLOCK + k % BLOCK] = weights[k * inputs + j] as i16;
        }
    }
    out
}

/// One dense layer over its non-zero inputs (Q1120): `sums[k] = sum_j w[k][j] * x[j]` for
/// every output `k` the columns hold (a whole number of blocks), from the columns `columns`
/// builds over `inputs` inputs and the list `on` of the inputs whose `x` is not zero. A zero
/// input adds nothing, so skipping it is exact; each pair's two products sum in `i16`
/// (`PAIR_BOUND`) and every lane in `i32` (`DOT_BOUND`), so the sums are the dense `dot`'s
/// bit for bit, in any order.
#[inline]
fn sparse_layer(columns: &[i16], inputs: usize, on: &[u16], x: &[i16], sums: &mut [i32]) {
    let (columns, _) = columns.as_chunks::<BLOCK>();
    let (outs, _) = sums.as_chunks_mut::<BLOCK>();
    for (block, out) in columns.chunks_exact(inputs).zip(outs) {
        let mut lanes = [0i32; BLOCK];
        let mut pairs = on.chunks_exact(2);
        for pair in &mut pairs {
            let (j0, j1) = (pair[0] as usize, pair[1] as usize);
            let (c0, c1, x0, x1) = (&block[j0], &block[j1], x[j0], x[j1]);
            for k in 0..BLOCK {
                lanes[k] += (c0[k] * x0 + c1[k] * x1) as i32;
            }
        }
        if let [j] = *pairs.remainder() {
            let (c, v) = (&block[j as usize], x[j as usize]);
            for k in 0..BLOCK {
                lanes[k] += (c[k] * v) as i32;
            }
        }
        *out = lanes;
    }
}

/// The list of `x`'s non-zero slots, ascending, written into the front of `on` (at least as
/// long as `x`); the count. Branch-free: every slot is written and the count moves only past
/// a non-zero one, since half the slots are zero in no pattern a predictor learns (Q1120).
#[inline]
fn list_on(x: &[i16], on: &mut [u16]) -> usize {
    let mut n = 0;
    for (j, &v) in x.iter().enumerate() {
        on[n] = j as u16;
        n += (v != 0) as usize;
    }
    n
}

/// The buffers an evaluation writes, kept by the caller so it allocates nothing (Q961,
/// Q1120): the clipped accumulator and the list of its non-zero slots.
#[derive(Clone, Debug, Default)]
pub struct Buffers {
    a0: Vec<i16>,
    on: Vec<u16>,
}

/// The store's first eight bytes.
pub const MAGIC: &[u8; 8] = b"CVSNET01";
/// The input layout's version. A new segment, or a changed encoding in one, is a bump,
/// and a store of another version is refused rather than reinterpreted.
///
/// Q918g added the round segments (`Layout::initiative` to `bars`) **without** a bump, on
/// purpose: they exist only on tables with `rules.simultaneous`, where no store could be
/// written before (the core refused a net in a round until then), so no stored byte is read
/// differently, and every alternating layout, store and procedure digest stays as it was
/// (`tests/test_round_nets.py` holds it). A later change to a segment a store exists for
/// is a bump.
///
/// **Version 2** (Q1172, W4, `docs/plan-player-weaknesses.md` §T4) appends three segments
/// after `bars`, each only on tables that carry its rule: the purse (`rules.purse`), the
/// bank (`rules.reach.removes`) and the filled-parts mask (`rules.turn.parts` under order
/// `"any"`). On tables with none of the three, layout 2 is layout 1 input for input, so a
/// store of version 1 is still read there (`OLDEST_INPUTS_VERSION`), and refused by
/// `Net::check_against` on tables that carry one: no stored byte is read differently.
/// Stores are written as 2, so a procedure trained from now on is a new name.
pub const INPUTS_VERSION: u32 = 2;
/// The oldest store version this core reads, where the tables carry none of the segments
/// added since (`Layout::since_v1`).
pub const OLDEST_INPUTS_VERSION: u32 = 1;

/// Thermometer lengths for the counted state.
pub const HAND_CAP: usize = 16;
pub const CHECK_CAP: usize = 8;
pub const POINTS_CAP: usize = 16;
pub const CLOCK_BUCKETS: usize = 8;
pub const TURN_CAP: usize = 4;
/// Layout 2's purse (Q1172): each player's coins as `PURSE_UNITS` unit inputs (one per coin
/// up to 16, where one coin is the margin a move costs) and then the scalar scaled by
/// `tables::PURSE_MAX` as `PURSE_BUCKETS` thermometer inputs, the clocks' rule (bucket `k`
/// on once `purse * PURSE_BUCKETS >= (k + 1) * PURSE_MAX`).
pub const PURSE_UNITS: usize = 16;
pub const PURSE_BUCKETS: usize = 8;
/// Layout 2's bank (Q1172): each player's banked pieces as a thermometer of `BANK_CAP`.
pub const BANK_CAP: usize = 16;

/// The hidden layer after the accumulator is `(sum >> WEIGHT_SHIFT)` clipped to
/// `[0, ACTIVATION_MAX]`.
pub const WEIGHT_SHIFT: u32 = 6;
pub const ACTIVATION_MAX: i64 = 127;
/// The output is clamped to `±OUTPUT_LIMIT` and divided by `OUTPUT_SCALE`, which puts it
/// in the built-in objective's units and far inside `eval::WIN_SCORE`.
pub const OUTPUT_LIMIT: i64 = 1 << 27;
pub const OUTPUT_SCALE: f64 = 1024.0;

/// Largest layer widths a store may declare.
pub const MAX_ACCUMULATOR: usize = 2048;
pub const MAX_HIDDEN: usize = 128;

/// Where each segment of the input vector starts, for one ruleset's tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub n_squares: usize,
    pub n_values: usize,
    pub n_types: usize,
    pub pieces: usize,
    pub unmoved: usize,
    pub ep: usize,
    pub colour: usize,
    pub hands: usize,
    pub checks: usize,
    pub points: usize,
    pub halfmove: usize,
    pub ply: usize,
    pub turn: usize,
    /// Q918g: a round game's own state (`docs/archive/plan-simultaneous.md` §6.4, *Q918g, scoped*),
    /// after `turn`. Each segment is there only on tables that carry its rule, so every
    /// alternating layout -- and every store and digest written for one -- is unchanged.
    /// `initiative`: two inputs, who holds the token, under `tie: "initiative"`.
    pub initiative: usize,
    /// `rest`: `2 * n_squares * R`, one-hot per (turns left, player, square), index
    /// `((left - 1) * 2 + player) * n_squares + square` (the key's own order), `R` the
    /// longest rest: in **every game that rests**, alternating or in rounds (schema 29's
    /// `pieces[].rest`, Q983b, `docs/plan-rest.md` §5.4). For `R = 1` in a round game it is
    /// schema 27's plane, `player * n_squares + square`, so a round net's layout is
    /// unchanged; in an alternating game it follows `turn` (no round segment sits between),
    /// a game that rests having no store before it. Every game without a rest is unchanged.
    pub rest: usize,
    /// `R`, the longest rest (`RestTables::longest`), 0 without a rest.
    pub rest_turns: usize,
    /// `void`: `2 * n_squares`, the void run's two squares (player 0's, then player 1's).
    pub void: usize,
    /// `bars`: `2 * n_squares`, every barred pair's squares, player 0's then player 1's.
    pub bars: usize,
    /// Layout 2 (Q1172), after `bars`, each only where its rule is on. `purse`:
    /// `2 * (PURSE_UNITS + PURSE_BUCKETS)` under `rules.purse`, player 0's units then
    /// buckets, then player 1's. `bank`: `2 * BANK_CAP` under `rules.reach.removes`.
    /// `parts`: one input per part, on while that part is filled, under `rules.turn.parts`
    /// with order `"any"` (under `"fixed"` the mask is the moves left, which `turn` carries).
    pub purse: usize,
    pub bank: usize,
    pub parts: usize,
    pub has_purse: bool,
    pub has_bank: bool,
    pub n_parts: usize,
    pub has_initiative: bool,
    pub has_rest: bool,
    pub rounds: bool,
    pub n_inputs: usize,
    pub no_progress: u32,
    pub max_plies: u32,
}

impl Layout {
    pub fn of(t: &Tables) -> Self {
        let n_squares = t.n_squares;
        let n_types = t.n_types;
        let n_values = t.n_owners * t.n_facings * n_types;
        let pieces = 0;
        let unmoved = pieces + n_squares * n_values;
        let ep = unmoved + n_squares;
        let colour = ep + n_squares;
        let hands = colour + 2 * n_squares;
        let checks = hands + 2 * n_types * HAND_CAP;
        let points = checks + 2 * CHECK_CAP;
        let halfmove = points + 2 * POINTS_CAP;
        let ply = halfmove + CLOCK_BUCKETS;
        let turn = ply + CLOCK_BUCKETS;
        // Q918g: the round segments, each only where its rule is on (a spy game alternates
        // as well, but plays no net: `AgentSpec::check_against` refuses one there).
        let sim = t.simultaneous.as_deref();
        let rounds = sim.is_some();
        let has_initiative = sim.is_some_and(|sim| sim.tie == 1);
        // Q983b: the rest plane in every game that rests, its width the longest rest's.
        let rest_turns = t.rest.as_deref().map_or(0, |rest| rest.longest);
        let has_rest = rest_turns > 0;
        let initiative = turn + TURN_CAP;
        let rest = initiative + if has_initiative { 2 } else { 0 };
        let void = rest + 2 * n_squares * rest_turns;
        let bars = void + if rounds { 2 * n_squares } else { 0 };
        // Q1172 (layout 2): the purse, the bank and the parts mask, each only where its
        // rule is on, so a layout with none of them is layout 1's.
        let has_purse = t.purse.is_some();
        let has_bank = t.reach_removes;
        let n_parts = t.parts.as_deref().map_or(0, |parts| if parts.any_order { parts.len() } else { 0 });
        let purse = bars + if rounds { 2 * n_squares } else { 0 };
        let bank = purse + if has_purse { 2 * (PURSE_UNITS + PURSE_BUCKETS) } else { 0 };
        let parts = bank + if has_bank { 2 * BANK_CAP } else { 0 };
        let n_inputs = parts + n_parts;
        Self {
            n_squares,
            n_values,
            n_types,
            pieces,
            unmoved,
            ep,
            colour,
            hands,
            checks,
            points,
            halfmove,
            ply,
            turn,
            initiative,
            rest,
            rest_turns,
            void,
            bars,
            purse,
            bank,
            parts,
            has_purse,
            has_bank,
            n_parts,
            has_initiative,
            has_rest,
            rounds,
            n_inputs,
            no_progress: t.no_progress_draw,
            max_plies: t.max_plies,
        }
    }

    /// Whether these tables carry a segment layout 1 did not have, where a store of
    /// version 1 is refused (`Net::check_against`).
    pub fn since_v1(&self) -> bool {
        self.has_purse || self.has_bank || self.n_parts > 0
    }

    /// The active inputs of `game`, ascending, appended to `out` (which is cleared).
    pub fn features(&self, game: &Game<'_>, out: &mut Vec<u32>) {
        out.clear();
        self.layout_one(game, out);
        // Q1172 (layout 2), after every segment of layout 1, so ascending.
        if self.has_purse {
            for player in 0..2 {
                let base = self.purse + player * (PURSE_UNITS + PURSE_BUCKETS);
                let coins = game.purse[player] as usize;
                for k in 0..coins.min(PURSE_UNITS) {
                    out.push((base + k) as u32);
                }
                push_scaled(out, base + PURSE_UNITS, PURSE_BUCKETS, coins as u64, PURSE_MAX as u64);
            }
        }
        if self.has_bank {
            for player in 0..2 {
                let base = self.bank + player * BANK_CAP;
                for k in 0..(game.banked[player] as usize).min(BANK_CAP) {
                    out.push((base + k) as u32);
                }
            }
        }
        for part in 0..self.n_parts {
            if game.parts_filled >> part & 1 == 1 {
                out.push((self.parts + part) as u32);
            }
        }
    }

    /// Layout 1's segments with the round and rest segments, ascending.
    fn layout_one(&self, game: &Game<'_>, out: &mut Vec<u32>) {
        let s = self.n_squares;
        for square in 0..s {
            let byte = game.board[square] as usize;
            if byte != 0 && byte <= self.n_values {
                out.push((self.pieces + square * self.n_values + byte - 1) as u32);
            }
        }
        for square in 0..s {
            if game.unmoved.get(square) {
                out.push((self.unmoved + square) as u32);
            }
        }
        if game.ep_target >= 0 && (game.ep_target as usize) < s {
            out.push((self.ep + game.ep_target as usize) as u32);
        }
        for (square, &c) in game.colour.iter().take(s).enumerate() {
            if c == 1 || c == 2 {
                out.push((self.colour + square * 2 + c as usize - 1) as u32);
            }
        }
        for player in 0..2 {
            for (piece_type, &count) in game.hands[player].iter().take(self.n_types).enumerate() {
                let base = self.hands + (player * self.n_types + piece_type) * HAND_CAP;
                for k in 0..(count as usize).min(HAND_CAP) {
                    out.push((base + k) as u32);
                }
            }
        }
        for player in 0..2 {
            let base = self.checks + player * CHECK_CAP;
            for k in 0..(game.checks[player] as usize).min(CHECK_CAP) {
                out.push((base + k) as u32);
            }
        }
        for player in 0..2 {
            let base = self.points + player * POINTS_CAP;
            let whole = game.points[player].max(0.0).floor();
            let whole = if whole >= POINTS_CAP as f64 { POINTS_CAP } else { whole as usize };
            for k in 0..whole {
                out.push((base + k) as u32);
            }
        }
        push_clock(out, self.halfmove, game.halfmove, self.no_progress);
        push_clock(out, self.ply, game.ply, self.max_plies);
        let extra = (game.remaining.saturating_sub(1) as usize).min(TURN_CAP);
        for k in 0..extra {
            out.push((self.turn + k) as u32);
        }
        if self.has_initiative && (game.initiative == 0 || game.initiative == 1) {
            out.push((self.initiative + game.initiative as usize) as u32);
        }
        if self.has_rest {
            // At most `REST_SLOTS` a player; each input on once, ascending.
            let start = out.len();
            for player in 0..2 {
                for (square, left) in game.resting_of(player) {
                    let left = left as usize;
                    if square >= 0 && (square as usize) < s && left >= 1 && left <= self.rest_turns {
                        out.push((self.rest + ((left - 1) * 2 + player) * s + square as usize) as u32);
                    }
                }
            }
            out[start..].sort_unstable();
        }
        if !self.rounds {
            return;
        }
        for player in 0..2 {
            let square = game.void_run[player];
            if square >= 0 && (square as usize) < s {
                out.push((self.void + player * s + square as usize) as u32);
            }
        }
        if !game.bars.is_empty() {
            // A square may be in two bars; each input is on once, ascending.
            let start = out.len();
            for player in 0..2 {
                for bar in &game.bars {
                    let square = bar[player];
                    if square >= 0 && (square as usize) < s {
                        out.push((self.bars + player * s + square as usize) as u32);
                    }
                }
            }
            out[start..].sort_unstable();
            let mut kept = start;
            for i in start..out.len() {
                if kept == start || out[i] != out[kept - 1] {
                    out[kept] = out[i];
                    kept += 1;
                }
            }
            out.truncate(kept);
        }
    }
}

/// `CLOCK_BUCKETS` thermometer inputs: bucket `k` is on once `value` has run
/// `(k + 1) / CLOCK_BUCKETS` of `limit`. Nothing when the clock is off.
fn push_clock(out: &mut Vec<u32>, base: usize, value: u32, limit: u32) {
    push_scaled(out, base, CLOCK_BUCKETS, value as u64, limit as u64);
}

/// `buckets` thermometer inputs from `base`: bucket `k` is on once `value` has run
/// `(k + 1) / buckets` of `limit`. Nothing when `limit` is 0.
fn push_scaled(out: &mut Vec<u32>, base: usize, buckets: usize, value: u64, limit: u64) {
    if limit == 0 {
        return;
    }
    for k in 0..buckets {
        if value * buckets as u64 >= (k as u64 + 1) * limit {
            out.push((base + k) as u32);
        }
    }
}

/// One side-to-move bucket of the layers after the accumulator.
#[derive(Clone, Debug)]
pub struct Head {
    /// `[k * accumulator + j]`, as stored. Since Q1120 the pass reads `c1`; this is kept
    /// for the tests' `i64` reference, which must not read the layout it checks.
    #[cfg_attr(not(test), allow(dead_code))]
    pub w1: Vec<i8>,
    pub b1: Vec<i32>,
    /// `[k * hidden1 + j]`, as stored; the pass reads `c2`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub w2: Vec<i8>,
    pub b2: Vec<i32>,
    pub w3: Vec<i8>,
    pub b3: i32,
    /// `w1` and `w2` as `columns` lays them out for `sparse_layer`, made once at parse
    /// (Q1120); the store and the fields above are unchanged.
    c1: Vec<i16>,
    c2: Vec<i16>,
}

/// A parsed store.
#[derive(Clone, Debug)]
pub struct Net {
    /// The store's input layout version, `OLDEST_INPUTS_VERSION` to `INPUTS_VERSION`.
    pub version: u32,
    pub n_inputs: usize,
    pub accumulator: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    /// `[input * accumulator + j]`: one input's row is contiguous.
    pub w0: Vec<i16>,
    pub b0: Vec<i16>,
    pub heads: [Head; 2],
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.bytes.len() - self.at < n {
            return Err(format!(
                "the net store is truncated: wanted {n} more bytes at offset {}, {} left",
                self.at,
                self.bytes.len() - self.at
            ));
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32s(&mut self, n: usize) -> Result<Vec<i32>, String> {
        let b = self.take(4 * n)?;
        Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    fn i16s(&mut self, n: usize) -> Result<Vec<i16>, String> {
        let b = self.take(2 * n)?;
        Ok(b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn i8s(&mut self, n: usize) -> Result<Vec<i8>, String> {
        Ok(self.take(n)?.iter().map(|&b| b as i8).collect())
    }
}

impl Net {
    /// Parse a store's bytes, or the sentence saying why they are refused.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(8)? != MAGIC {
            return Err("the net store does not start with the CVSNET01 magic".to_string());
        }
        let version = r.u32()?;
        if !(OLDEST_INPUTS_VERSION..=INPUTS_VERSION).contains(&version) {
            return Err(format!(
                "the net store was written for input layout {version} and this core reads \
                 layouts {OLDEST_INPUTS_VERSION} to {INPUTS_VERSION}; it is refused rather than \
                 reinterpreted"
            ));
        }
        let n_inputs = r.u32()? as usize;
        let accumulator = r.u32()? as usize;
        let hidden1 = r.u32()? as usize;
        let hidden2 = r.u32()? as usize;
        if accumulator == 0 || accumulator > MAX_ACCUMULATOR {
            return Err(format!("the net's accumulator width {accumulator} is outside 1..={MAX_ACCUMULATOR}"));
        }
        for (name, width) in [("first hidden", hidden1), ("second hidden", hidden2)] {
            if width == 0 || width > MAX_HIDDEN {
                return Err(format!("the net's {name} width {width} is outside 1..={MAX_HIDDEN}"));
            }
        }
        let w0 = r.i16s(n_inputs.checked_mul(accumulator).ok_or("the net is too large")?)?;
        let b0 = r.i16s(accumulator)?;
        let mut head = || -> Result<Head, String> {
            let w1 = r.i8s(hidden1 * accumulator)?;
            let b1 = r.i32s(hidden1)?;
            let w2 = r.i8s(hidden2 * hidden1)?;
            let (c1, c2) = (columns(&w1, hidden1, accumulator), columns(&w2, hidden2, hidden1));
            Ok(Head { w1, b1, w2, b2: r.i32s(hidden2)?, w3: r.i8s(hidden2)?, b3: r.i32s(1)?[0], c1, c2 })
        };
        let heads = [head()?, head()?];
        if r.at != bytes.len() {
            return Err(format!(
                "the net store has {} bytes after its last layer",
                bytes.len() - r.at
            ));
        }
        Ok(Self { version, n_inputs, accumulator, hidden1, hidden2, w0, b0, heads })
    }

    /// Whether this net was laid out for these tables.
    pub fn check_against(&self, t: &Tables) -> Result<(), String> {
        let layout = Layout::of(t);
        if self.version < INPUTS_VERSION && layout.since_v1() {
            return Err(format!(
                "the net was written for input layout {} and this ruleset carries a segment \
                 layout {INPUTS_VERSION} added (the purse, the bank or the parts mask); it is \
                 refused rather than reinterpreted",
                self.version
            ));
        }
        let want = layout.n_inputs;
        if self.n_inputs != want {
            return Err(format!(
                "the net reads {} inputs and this ruleset's layout has {want}; it was \
                 trained on another game and is refused",
                self.n_inputs
            ));
        }
        Ok(())
    }

    /// The raw output for the side to move, from an accumulator.
    pub fn forward(&self, acc: &[i16], side: usize) -> i64 {
        self.forward_with(acc, side, &mut Buffers::default())
    }

    /// `forward` with the clipped accumulator and its non-zero slots written into
    /// `buffers`, which the caller keeps (Q961, Q1120): they hold the net's own width, so
    /// an evaluation allocates nothing.
    ///
    /// The dense layers sum `i8 * [0, 127]` products in `i32` (Q961), which is exact:
    /// `DOT_BOUND` below is the largest such sum any store `parse` accepts, and it is far
    /// inside `i32`, so the sum equals the `i64` one bit for bit. Since Q1120 each layer
    /// visits only its non-zero inputs (`sparse_layer`), the same integer sum. The bias,
    /// the shift and the output layer stay in `i64`, as before.
    pub fn forward_with(&self, acc: &[i16], side: usize, buffers: &mut Buffers) -> i64 {
        let head = &self.heads[side & 1];
        let Buffers { a0, on } = buffers;
        a0.clear();
        a0.extend(acc.iter().take(self.accumulator).map(|&v| v.clamp(0, ACTIVATION_MAX as i16)));
        on.resize(a0.len(), 0);
        let n_on = list_on(a0, on);
        let mut sums = [0i32; MAX_HIDDEN];
        sparse_layer(&head.c1, self.accumulator, &on[..n_on], a0, &mut sums);
        let mut a1 = [0i16; MAX_HIDDEN];
        for ((slot, &b), &sum) in a1.iter_mut().zip(&head.b1).zip(&sums[..self.hidden1]) {
            *slot = ((b as i64 + sum as i64) >> WEIGHT_SHIFT).clamp(0, ACTIVATION_MAX) as i16;
        }
        let mut on1 = [0u16; MAX_HIDDEN];
        let n_on1 = list_on(&a1[..self.hidden1], &mut on1);
        sparse_layer(&head.c2, self.hidden1, &on1[..n_on1], &a1, &mut sums);
        let mut out = head.b3 as i64;
        for ((&b, &w), &sum) in head.b2.iter().zip(&head.w3).zip(&sums[..self.hidden2]) {
            let a2 = ((b as i64 + sum as i64) >> WEIGHT_SHIFT).clamp(0, ACTIVATION_MAX);
            out += w as i64 * a2;
        }
        out.clamp(-OUTPUT_LIMIT, OUTPUT_LIMIT)
    }

    /// A fresh accumulator over `features`.
    pub fn refresh(&self, features: &[u32]) -> Vec<i16> {
        let mut acc = self.b0.clone();
        for &i in features {
            self.add_row(&mut acc, i as usize);
        }
        acc
    }

    #[inline]
    fn add_row(&self, acc: &mut [i16], input: usize) {
        let row = &self.w0[input * self.accumulator..(input + 1) * self.accumulator];
        for (a, &w) in acc.iter_mut().zip(row) {
            *a = a.wrapping_add(w);
        }
    }

    #[inline]
    fn sub_row(&self, acc: &mut [i16], input: usize) {
        let row = &self.w0[input * self.accumulator..(input + 1) * self.accumulator];
        for (a, &w) in acc.iter_mut().zip(row) {
            *a = a.wrapping_sub(w);
        }
    }
}

/// A net with its accumulator: the objective of a `net=` player.
pub struct NetEvaluator {
    net: std::sync::Arc<Net>,
    layout: Layout,
    /// The derived table, for capture ordering only; the net is the objective.
    pub values: Vec<f64>,
    acc: Vec<i16>,
    active: Vec<u32>,
    scratch: Vec<u32>,
    /// The clipped accumulator and its non-zero slots, kept so an evaluation allocates
    /// nothing.
    buffers: Buffers,
}

impl NetEvaluator {
    pub fn new(t: &Tables, net: std::sync::Arc<Net>, values: Vec<f64>) -> Self {
        let acc = net.b0.clone();
        let buffers = Buffers {
            a0: Vec::with_capacity(net.accumulator),
            on: Vec::with_capacity(net.accumulator),
        };
        let (active, scratch) = (Vec::new(), Vec::new());
        Self { net, layout: Layout::of(t), values, acc, active, scratch, buffers }
    }

    /// The raw integer output for the side to move, bringing the accumulator up to
    /// `game` by the difference from the last position it summed.
    pub fn raw(&mut self, game: &Game<'_>) -> i64 {
        let mut next = std::mem::take(&mut self.scratch);
        self.layout.features(game, &mut next);
        // Merge the two ascending lists: rows only in `active` leave, rows only in
        // `next` arrive.
        let (mut i, mut j) = (0, 0);
        let (old, new) = (&self.active, &next);
        while i < old.len() || j < new.len() {
            if j == new.len() || (i < old.len() && old[i] < new[j]) {
                self.net.sub_row(&mut self.acc, old[i] as usize);
                i += 1;
            } else if i == old.len() || new[j] < old[i] {
                self.net.add_row(&mut self.acc, new[j] as usize);
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        self.scratch = std::mem::replace(&mut self.active, next);
        self.net.forward_with(&self.acc, game.side, &mut self.buffers)
    }

    /// The score for `player`: the net's output for the side to move, negated for the
    /// other side, in the built-in objective's units. In a round game `side` stays 0
    /// (`docs/archive/plan-simultaneous.md` §6.1), so the one head read there is seat 0's and seat 1's
    /// score is its negation (§6.4): the round's evaluator needs no arm of its own.
    pub fn evaluate(&mut self, game: &Game<'_>, player: usize) -> f64 {
        let score = self.raw(game) as f64 / OUTPUT_SCALE;
        if player == game.side {
            score
        } else {
            -score
        }
    }

    /// The accumulator as it stands, for the test that holds it to a fresh sum.
    pub fn accumulator(&self) -> &[i16] {
        &self.acc
    }

    pub fn active(&self) -> &[u32] {
        &self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store's bytes for the given widths, every weight `w` and bias `b`.
    fn store(n_inputs: u32, widths: (u32, u32, u32), w: i8, b: i32) -> Vec<u8> {
        let (acc, h1, h2) = widths;
        let mut out = MAGIC.to_vec();
        for v in [INPUTS_VERSION, n_inputs, acc, h1, h2] {
            out.extend(v.to_le_bytes());
        }
        for _ in 0..n_inputs * acc {
            out.extend((w as i16).to_le_bytes());
        }
        for _ in 0..acc {
            out.extend((b as i16).to_le_bytes());
        }
        for _ in 0..2 {
            out.extend(std::iter::repeat_n(w as u8, (h1 * acc) as usize));
            for _ in 0..h1 {
                out.extend(b.to_le_bytes());
            }
            out.extend(std::iter::repeat_n(w as u8, (h2 * h1) as usize));
            for _ in 0..h2 {
                out.extend(b.to_le_bytes());
            }
            out.extend(std::iter::repeat_n(w as u8, h2 as usize));
            out.extend(b.to_le_bytes());
        }
        out
    }

    #[test]
    fn a_well_formed_store_parses_and_a_damaged_one_says_why() {
        let bytes = store(5, (4, 3, 2), 1, 0);
        let net = Net::parse(&bytes).unwrap();
        assert_eq!((net.n_inputs, net.accumulator, net.hidden1, net.hidden2), (5, 4, 3, 2));
        let short = Net::parse(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(short.contains("truncated"), "{short}");
        let mut long = bytes.clone();
        long.push(0);
        assert!(Net::parse(&long).unwrap_err().contains("after its last layer"));
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(Net::parse(&magic).unwrap_err().contains("magic"));
        let mut version = bytes;
        version[8] = 99;
        assert!(Net::parse(&version).unwrap_err().contains("layout 99"));
    }

    /// The arithmetic by hand, for one small net: two inputs on, weights 1, biases 0.
    /// Accumulator 2 per slot; first hidden `(4 * 2) >> 6 = 0`, so the output is the
    /// output bias. With biases 64: accumulator 66 per slot; first hidden
    /// `(64 + 4 * 66) >> 6 = 5`; second `(64 + 3 * 5) >> 6 = 1`; output `64 + 2 * 1`.
    #[test]
    fn the_forward_pass_is_the_stated_integer_arithmetic() {
        let net = Net::parse(&store(5, (4, 3, 2), 1, 0)).unwrap();
        assert_eq!(net.forward(&net.refresh(&[0, 3]), 0), 0);
        let net = Net::parse(&store(5, (4, 3, 2), 1, 64)).unwrap();
        let acc = net.refresh(&[0, 3]);
        assert_eq!(acc, vec![66; 4]);
        assert_eq!(net.forward(&acc, 1), 66);
    }

    /// The forward pass as it was before Q961: every sum in `i64`, over a zeroed
    /// `MAX_ACCUMULATOR` buffer. The `i32` path must agree with it everywhere.
    fn forward_in_i64(net: &Net, acc: &[i16], side: usize) -> i64 {
        let head = &net.heads[side & 1];
        let mut a0 = [0i64; MAX_ACCUMULATOR];
        for (slot, &v) in a0.iter_mut().zip(acc) {
            *slot = (v as i64).clamp(0, ACTIVATION_MAX);
        }
        let a0 = &a0[..net.accumulator];
        let mut a1 = [0i64; MAX_HIDDEN];
        for (k, slot) in a1.iter_mut().enumerate().take(net.hidden1) {
            let row = &head.w1[k * net.accumulator..(k + 1) * net.accumulator];
            let sum: i64 = row.iter().zip(a0).map(|(&w, &a)| w as i64 * a).sum();
            *slot = ((head.b1[k] as i64 + sum) >> WEIGHT_SHIFT).clamp(0, ACTIVATION_MAX);
        }
        let a1 = &a1[..net.hidden1];
        let mut out = head.b3 as i64;
        for k in 0..net.hidden2 {
            let row = &head.w2[k * net.hidden1..(k + 1) * net.hidden1];
            let sum: i64 = row.iter().zip(a1).map(|(&w, &a)| w as i64 * a).sum();
            let a2 = ((head.b2[k] as i64 + sum) >> WEIGHT_SHIFT).clamp(0, ACTIVATION_MAX);
            out += head.w3[k] as i64 * a2;
        }
        out.clamp(-OUTPUT_LIMIT, OUTPUT_LIMIT)
    }

    /// Q961: at the widest store `parse` accepts, every weight at an `i8` extreme and
    /// every activation at 127, the `i32` dot product reaches `DOT_BOUND` exactly and the
    /// output equals the `i64` reference's.
    #[test]
    fn the_i32_sum_is_exact_at_the_widest_most_extreme_store() {
        for w in [i8::MIN, i8::MAX] {
            let widths = (MAX_ACCUMULATOR as u32, MAX_HIDDEN as u32, MAX_HIDDEN as u32);
            let net = Net::parse(&store(1, widths, w, 127)).unwrap();
            let acc = net.refresh(&[]);
            assert!(acc.iter().all(|&a| a == 127));
            let a0 = vec![127i16; MAX_ACCUMULATOR];
            let row = &net.heads[0].w1[..MAX_ACCUMULATOR];
            assert_eq!(dot(row, &a0) as i64, w as i64 * 127 * MAX_ACCUMULATOR as i64);
            for side in 0..2 {
                assert_eq!(net.forward(&acc, side), forward_in_i64(&net, &acc, side), "w {w}");
            }
        }
        assert_eq!(DOT_BOUND, 2048 * 128 * 127);
    }

    /// Q961: random stores of many widths and full-range weights, over random
    /// accumulators, give the `i64` reference's output, with one buffer carried across
    /// nets of different widths as the evaluator carries it.
    #[test]
    fn the_forward_pass_equals_the_i64_reference_on_random_nets() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut buffer = Buffers::default();
        for _ in 0..40 {
            let acc_w = 1 + (next() % 300) as usize;
            let (h1, h2) = (1 + (next() % 40) as usize, 1 + (next() % 40) as usize);
            let mut bytes = MAGIC.to_vec();
            for v in [INPUTS_VERSION, 1, acc_w as u32, h1 as u32, h2 as u32] {
                bytes.extend(v.to_le_bytes());
            }
            for _ in 0..2 * acc_w {
                bytes.extend((next() as i16 / 64).to_le_bytes());
            }
            for _ in 0..2 {
                bytes.extend((0..h1 * acc_w).map(|_| next() as u8));
                bytes.extend((0..h1).flat_map(|_| ((next() % 16_000) as i32 - 4000).to_le_bytes()));
                bytes.extend((0..h2 * h1).map(|_| next() as u8));
                bytes.extend((0..h2).flat_map(|_| ((next() % 8_000) as i32 - 2000).to_le_bytes()));
                bytes.extend((0..h2).map(|_| next() as u8));
                bytes.extend(((next() % 10_000) as i32 - 5000).to_le_bytes());
            }
            let net = Net::parse(&bytes).unwrap();
            for round in 0..8 {
                // Q1120: every other accumulator mostly clipped to zero, the sparse pass's
                // case (an odd count of non-zero slots, one, none).
                let acc: Vec<i16> = (0..acc_w)
                    .map(|_| match round % 2 {
                        0 => (next() % 400) as i16 - 120,
                        _ if next() % 5 == 0 => (next() % 140) as i16,
                        _ => -((next() % 50) as i16),
                    })
                    .collect();
                for side in 0..2 {
                    let want = forward_in_i64(&net, &acc, side);
                    assert_eq!(net.forward(&acc, side), want);
                    assert_eq!(net.forward_with(&acc, side, &mut buffer), want);
                }
            }
        }
    }

    #[test]
    fn the_accumulator_wraps_so_adding_and_removing_is_exact() {
        let net = Net::parse(&store(3, (2, 1, 1), 127, 32_000)).unwrap();
        let mut acc = net.refresh(&[0, 1, 2]);
        net.sub_row(&mut acc, 1);
        net.sub_row(&mut acc, 2);
        assert_eq!(acc, net.refresh(&[0]));
    }
}
