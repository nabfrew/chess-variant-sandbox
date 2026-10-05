//! The position-independent tables one ruleset compiles to.
//!
//! Nothing here interprets the rule DSL. `chess_variant_sandbox/engine/compile.py` does that once,
//! in Python, and hands the result over as flat integer arrays; this module only
//! rearranges them into the layout the generator wants. Keeping the interpretation on
//! one side of the boundary is what stops the two implementations from drifting apart
//! on what a rule *means* -- the only thing duplicated across the boundary is how a
//! table is walked, and perft is a direct test of that.

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const NO_SQUARE: i16 = -1;
/// The third owner: a wild piece, which neither player owns (schema 9).
///
/// Named rather than spelled `2` for the reason `SCREEN_ANY` is: the number appears in
/// an ownership test, a byte-encoding test and an attack-table index, and a reader who
/// meets a bare `2` in any of them cannot tell which of the three it is. Mirrors
/// `chess_variant_sandbox/engine/compile.py::WILD`.
pub const WILD: usize = 2;
/// What `goal_distance` holds for a square a piece can never get to. Mirrors
/// `compile.UNREACHABLE`, and is one more than any real distance can be so that
/// "unreachable" compares as worse than the worst reachable square.
pub const GOAL_UNREACHABLE: u8 = 255;
pub const MAX_SQUARES: usize = 256;
/// The most piece types a spec may declare. Not a limit invented here: it is
/// `chess_variant_sandbox/spec/schema.py::MAX_PIECE_TYPES`, which the validator already enforces, so
/// this is a promise being kept rather than a new rule. `from_py` refuses a wider
/// table, because the alternative is a per-type buffer sized at run time on a path that
/// runs once per evaluated node.
pub const MAX_PIECE_TYPES: usize = 24;

pub const MOVE_NORMAL: u32 = 0;
pub const MOVE_EN_PASSANT: u32 = 1;
pub const MOVE_CASTLE: u32 = 2;
pub const MOVE_PROMOTION: u32 = 3;
pub const MOVE_DROP: u32 = 4;
/// Schema 8: a piece turns where it stands, and the new facing travels in the promotion
/// field. Origin equals target, as for a drop -- so `move_kind` is the only honest test
/// for either, and anything that told a drop apart by `origin == target` is now wrong.
pub const MOVE_ROTATE: u32 = 5;
/// Schema 13: a sweep whose path held at least one piece. A sweep over an empty path is
/// `MOVE_NORMAL`, the same word a slide to that square is.
pub const MOVE_SWEEP: u32 = 6;

/// `AttackSource::sweep` bit 0: the attacked square is itself a capture landing.
pub const SWEEP_LANDS: u8 = 1;
/// `AttackSource::sweep` bit 1: the ray has a landing square past the attacked one.
pub const SWEEP_BEYOND: u8 = 2;

#[inline(always)]
pub fn encode_move(origin: usize, target: usize, kind: u32, promotion: u32) -> u32 {
    origin as u32 | ((target as u32) << 8) | (kind << 16) | (promotion << 20)
}

#[inline(always)]
pub fn move_from(m: u32) -> usize {
    (m & 0xFF) as usize
}

#[inline(always)]
pub fn move_to(m: u32) -> usize {
    ((m >> 8) & 0xFF) as usize
}

#[inline(always)]
pub fn move_kind(m: u32) -> u32 {
    (m >> 16) & 0xF
}

#[inline(always)]
pub fn move_promotion(m: u32) -> u32 {
    (m >> 20) & 0x1F
}

/// Schema 12: which entry of the strike's landing list a survivor is placed on, in bits
/// 25 to 31. Zero on every move that places nothing.
#[inline(always)]
pub fn move_landing(m: u32) -> usize {
    ((m >> 25) & 0x7F) as usize
}

#[inline(always)]
pub fn with_landing(m: u32, index: usize) -> u32 {
    debug_assert!(index < 128, "a landing index past the move word's seven bits");
    m | ((index as u32) << 25)
}

/// A 256-bit mask over squares: the Rust counterpart of the arbitrary-width Python
/// integer used for `unmoved`. 256 is not arbitrary -- it is `MAX_BOARD_DIM` squared.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SquareSet {
    pub words: [u64; 4],
}

impl SquareSet {
    pub const EMPTY: Self = Self { words: [0; 4] };

    #[inline(always)]
    pub fn get(&self, square: usize) -> bool {
        (self.words[square >> 6] >> (square & 63)) & 1 == 1
    }

    #[inline(always)]
    pub fn set(&mut self, square: usize) {
        self.words[square >> 6] |= 1u64 << (square & 63);
    }

    #[inline(always)]
    pub fn clear(&mut self, square: usize) {
        self.words[square >> 6] &= !(1u64 << (square & 63));
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.words == [0u64; 4]
    }

    /// How many squares are in the set.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    #[inline(always)]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            words: [
                self.words[0] | other.words[0],
                self.words[1] | other.words[1],
                self.words[2] | other.words[2],
                self.words[3] | other.words[3],
            ],
        }
    }

    #[inline(always)]
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            words: [
                self.words[0] & other.words[0],
                self.words[1] & other.words[1],
                self.words[2] & other.words[2],
                self.words[3] & other.words[3],
            ],
        }
    }

    /// The squares in `self` that are not in `other`.
    #[inline(always)]
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            words: [
                self.words[0] & !other.words[0],
                self.words[1] & !other.words[1],
                self.words[2] & !other.words[2],
                self.words[3] & !other.words[3],
            ],
        }
    }

    /// The set's squares in **ascending** order.
    ///
    /// Ascending is the whole contract. Move generation walks the mover's pieces
    /// through this, and the order pieces are visited in is the order their moves come
    /// out in, which is the order every tie-break downstream reads -- so a set that
    /// yielded its squares any other way would change which move a search picks while
    /// every perft in the suite still passed. Words low to high and, within a word, the
    /// lowest set bit first, is ascending by construction rather than by sorting.
    #[inline(always)]
    pub fn iter(&self) -> SquareIter {
        SquareIter { words: self.words, index: 0 }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut set = Self::EMPTY;
        for (index, byte) in bytes.iter().take(32).enumerate() {
            set.words[index / 8] |= (*byte as u64) << (8 * (index % 8));
        }
        set
    }

    pub fn to_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = ((self.words[index / 8] >> (8 * (index % 8))) & 0xFF) as u8;
        }
        out
    }
}

/// `SquareSet::iter`: the set's squares, lowest first.
pub struct SquareIter {
    words: [u64; 4],
    index: usize,
}

impl Iterator for SquareIter {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        while self.index < 4 {
            let word = self.words[self.index];
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                self.words[self.index] = word & (word - 1);
                return Some(self.index * 64 + bit);
            }
            self.index += 1;
        }
        None
    }
}

/// One movement descriptor for one piece on one square, after compilation: a run of
/// target squares with a per-target mode mask, plus whether it needs an unmoved piece.
#[derive(Clone, Copy)]
pub struct EntryRef {
    pub start: u32,
    pub len: u32,
    pub requires_unmoved: bool,
    /// Hops only: how far past its screen this entry may land, 0 for the rest of the
    /// ray. The xiangqi cannon is 0; a grasshopper is 1. Leaps and slides carry 0 and
    /// never read it.
    pub screen_range: u16,
    /// Hops only: whose piece the screen must be. `SCREEN_ANY` is the cannon and every
    /// hop written before schema 3; `SCREEN_OWN` is a piece that vaults its own army.
    pub screen_owner: u8,
}

pub const SCREEN_ANY: u8 = 0;
pub const SCREEN_OWN: u8 = 1;
pub const SCREEN_ENEMY: u8 = 2;

/// Whether a piece encoded in `occupant` may serve as a screen for `player`.
#[inline(always)]
pub fn screen_matches(owner: u8, occupant: u16, low: u16, high: u16) -> bool {
    match owner {
        SCREEN_OWN => low <= occupant && occupant < high,
        SCREEN_ENEMY => !(low <= occupant && occupant < high),
        _ => true,
    }
}

/// `[player][piece type][square] -> entries`, flattened.
///
/// The nesting is gone on purpose. Move generation reads one slot per piece per ply,
/// and three levels of pointer chasing per read is most of what a table lookup costs
/// once the interpretation has already been done elsewhere.
pub struct EntryTable {
    index: Vec<u32>,
    refs: Vec<EntryRef>,
    squares: Vec<u16>,
    modes: Vec<u8>,
}

impl EntryTable {
    /// How many `(frame, type, square)` slots this table was built for.
    #[inline(always)]
    pub fn slots(&self) -> usize {
        self.index.len()
    }

    #[inline(always)]
    pub fn entries(&self, slot: usize) -> &[EntryRef] {
        let lo = self.index[slot] as usize;
        let hi = self.index[slot + 1] as usize;
        &self.refs[lo..hi]
    }

    #[inline(always)]
    pub fn squares(&self, entry: &EntryRef) -> &[u16] {
        let lo = entry.start as usize;
        &self.squares[lo..lo + entry.len as usize]
    }

    #[inline(always)]
    pub fn modes(&self, entry: &EntryRef) -> &[u8] {
        let lo = entry.start as usize;
        &self.modes[lo..lo + entry.len as usize]
    }

    fn build(index: Vec<u32>, packed: Vec<u32>, squares: Vec<u16>, modes: Vec<u8>) -> Self {
        let refs = packed
            .chunks_exact(5)
            .map(|chunk| EntryRef {
                start: chunk[0],
                len: chunk[1],
                requires_unmoved: chunk[2] != 0,
                screen_range: chunk[3] as u16,
                screen_owner: chunk[4] as u8,
            })
            .collect();
        Self {
            index,
            refs,
            squares,
            modes,
        }
    }
}

/// One way a square can be attacked, inverted out of the move tables by the compiler.
#[derive(Clone, Copy)]
pub struct AttackSource {
    pub origin: u16,
    pub piece_value: u8,
    pub screens: u8,
    pub requires_unmoved: bool,
    pub path_start: u32,
    pub path_len: u32,
    /// How many leading squares of the path must be *empty*. The rest is the window
    /// that has to hold exactly `screens` pieces.
    ///
    /// One flat scan serves all three primitives because of this split. A slide or a
    /// leap puts everything in the prefix and wants no screens; a cannon puts everything
    /// in the window and wants exactly one; a grasshopper, which may only land a bounded
    /// distance past its screen, puts everything before that window in the prefix -- so
    /// "the screen is in the right place" becomes the same count test as "there is a
    /// screen at all". Without it the core could not tell a bounded screen range from an
    /// unbounded one, which is why it used to refuse the ruleset outright.
    pub prefix_len: u32,
    /// Whose piece the screens have to be. `SCREEN_ANY` for every slide, every leap and
    /// every hop written before schema 3.
    pub screen_owner: u8,
    /// Schema 13: zero, or `SWEEP_LANDS | SWEEP_BEYOND` bits on a sweep source. A sweep
    /// source's path is the whole ray and `prefix_len` is the attacked square's index on
    /// it. The byte sits in the struct's padding, which a `cargo test` pins.
    pub sweep: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Win {
    Checkmate,
    RoyalCapture,
    Annihilation,
    /// Schema 2: no built-in win condition; only the named endings.
    None,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stalemate {
    Draw,
    Loss,
    Win,
}

/// What checkmate means for the mated player.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CheckmateResult {
    Loss,
    Win,
}

/// Explosion rule: captures destroy the neighbourhood.
pub struct ExplosionRule {
    pub immune_types: Vec<bool>,
    /// `blast[square]` -> list of squares destroyed.
    pub blast: Vec<Vec<usize>>,
    /// Schema 10 (Q18): which capturer types set a blast off, and which victim types
    /// have to be taken for one. `immune_types` is a third question about the same event
    /// -- what the blast *reaches* -- and none of the three can be said with the others.
    ///
    /// **Empty means every type**, which is atomic and every ruleset written before
    /// schema 10. That is the boundary's inert default, so `canonical_tables` drops both
    /// fields and no existing ruleset's fingerprint moved when they were added. The
    /// compiler has already turned a list naming every type into an empty one, and a
    /// list naming nobody into no explosion rule at all, so neither degenerate spelling
    /// reaches here.
    pub by_types: Vec<bool>,
    pub of_types: Vec<bool>,
}

impl ExplosionRule {
    /// Whether a capture of `victim` by `capturer` sets the blast off.
    ///
    /// Two `is_empty` tests on the common path, both of which the branch predictor sees
    /// the same answer to for the life of a ruleset. The alternative -- a full-length
    /// vector of `true` -- would have cost the same test and moved every stored
    /// fingerprint in the corpus.
    #[inline]
    pub fn fires(&self, capturer: usize, victim: usize) -> bool {
        (self.by_types.is_empty() || self.by_types[capturer])
            && (self.of_types.is_empty() || self.of_types[victim])
    }
}

/// Contact rule (schema 6 as `strike`, schema 10 as `contacts`): what happens instead
/// of removal when two pieces meet.
///
/// Movement knows nothing about this. A strike is *generated* as a capture -- same mode
/// bit, same blocking, same `may_capture` test -- so a strike ruleset's perft is exactly
/// its capture ruleset's perft, and only `make` reads anything here.
///
/// **One shape, whichever spelling the spec used.** The boundary sends either the
/// schema-6 array, keyed by the victim alone, or the two contact tables keyed by the
/// pair; the loader widens the first into the second, so everything below this line asks
/// one question. That is what lets `strike` keep sending the bytes it always sent while
/// the core carries only the general rule.
pub struct StrikeRule {
    /// `to[mover * n_types + victim]` -> the type the struck piece becomes, or -1 for it
    /// leaving the board. A link the table never mentions and an immune type both arrive
    /// as -1, so this is one test rather than three.
    pub to: Vec<i32>,
    /// `mover_to[mover * n_types + victim]` -> the type the *attacker* becomes, or -1
    /// for it staying what it is. All -1 for a `strike`, which is what makes schema 6 a
    /// special case of this rather than a second rule beside it.
    pub mover_to: Vec<i32>,
    /// Schema 11: the same two tables for a meeting with a *friendly* piece. Empty when
    /// no link says `own`, and then movegen never offers such a move.
    pub own_to: Vec<i32>,
    pub own_mover_to: Vec<i32>,
    /// `own_link[mover * n_types + target]`: whether the mover may move onto a friendly
    /// piece of that type. Derived from the two tables above, and empty with them.
    pub own_link: Vec<bool>,
    /// The stride of both tables, kept here so the one hot lookup needs nothing else.
    pub n_types: usize,
    /// Where the two pieces end up. A survivor can be in exactly two places: where it
    /// was (`Stay`) or where its attacker was (`Swap`). `Advance` is what capture has
    /// always done and never leaves a survivor -- the validator refuses to pair it with
    /// a table that keeps anything alive, because there would be no square to use. It
    /// can still transform the attacker, which is the one write schema 6 never had.
    /// `Choose` (schema 12) advances the attacker and places the survivor on a square of
    /// the landing list that the move word names.
    pub position: StrikePosition,
    /// `landing_index[player * landing_stride + square]..[+ 1]` slices `landing_squares`:
    /// where a struck piece standing on `square` may be placed, ascending. Empty unless
    /// the position is `Choose`.
    pub landing_index: Vec<u32>,
    pub landing_squares: Vec<u16>,
    pub landing_stride: usize,
}

impl StrikeRule {
    /// The squares a piece struck on `square` by `player` may be placed on, ascending.
    #[inline]
    pub fn landing(&self, player: usize, square: usize) -> &[u16] {
        let slot = player * self.landing_stride + square;
        let start = self.landing_index[slot] as usize;
        let end = self.landing_index[slot + 1] as usize;
        &self.landing_squares[start..end]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrikePosition {
    Stay,
    Advance,
    Swap,
    Choose,
}

/// Drop rule: pieces played from hand.
pub struct DropRule {
    pub from_captures: bool,
    /// `regions[player][piece_type]` -> None (any empty) or Some(set of squares).
    pub regions: Vec<Vec<Option<Vec<bool>>>>,
    /// piece types that cannot be dropped on a file already holding one.
    pub no_doubled: Vec<bool>,
    /// piece types whose drop may not deliver checkmate.
    pub no_mate: Vec<bool>,
    /// Schema 10 (Q51): what each player's hand holds before anything has been captured.
    ///
    /// Empty when nothing is stocked, which is crazyhouse, shogi, dobutsu and every other
    /// ruleset with a hand up to now -- that is the boundary's inert default, so none of
    /// their fingerprints moved.
    pub stock: Vec<u32>,
    /// Which of those slots is never spent. An unlimited slot sits at 1 for the whole
    /// game: `make` skips its decrement and `unmake` therefore records no delta for it,
    /// which is why a stone needs no Zobrist term -- the hand is out of the key already
    /// (schema 2's argument) and this keeps it constant besides.
    pub unlimited: Vec<bool>,
}

impl DropRule {
    /// Whether this type's hand slot is a supply rather than a count.
    #[inline]
    pub fn is_unlimited(&self, piece_type: usize) -> bool {
        !self.unlimited.is_empty() && self.unlimited[piece_type]
    }
}

/// Reach rule: win by arrival.
pub struct ReachRule {
    pub piece_type: i32,
    /// `squares[player]` -> set of target squares.
    pub squares: [Vec<bool>; 2],
    /// `goals[player]`: the same squares, ascending. The order `goal_distance_each`
    /// lists them in (Q708).
    pub goals: [Vec<usize>; 2],
    /// Tri-state, from the arriving player's point of view: -1 loss, 0 draw, 1 win.
    /// A boolean here read "draw" as a loss for the player who got there first.
    pub result: i8,
    pub safe: bool,
}

/// Turn rule: how many moves a turn is, and how the opening ramps up to it.
pub struct TurnRule {
    /// Moves per turn once the ramp is over.
    pub moves: u32,
    /// Length of each opening turn, in order. Everything past the end is `moves`.
    pub ramp: Vec<u32>,
}

impl TurnRule {
    #[inline]
    pub fn length_of(&self, turn: u32) -> u32 {
        *self.ramp.get(turn as usize).unwrap_or(&self.moves)
    }
}

/// Territory rule: squares that remember who moved onto them.
pub struct TerritoryRule {
    /// Whether a colour can be taken back. `false` is the cheap half: a colour that
    /// never changes is monotone, so a ply that lays one down is irreversible and
    /// resets the no-progress clock, so colour is constant inside any window a
    /// repetition test compares across -- and it needs no Zobrist term at all, exactly
    /// as `hands` needs none.
    pub repaints: bool,
    /// Colouring this many squares wins outright. 0 leaves the colouring to whatever
    /// else reads it, which is `ScoringRule` with count 3.
    pub win_at: u32,
}

/// Scoring rule: what to compare when the clock runs out, instead of drawing.
pub struct ScoringRule {
    /// 0 material, 1 pieces, 2 own pieces standing in a named region.
    pub count: u8,
    /// `squares[player]` -> the region that player is trying to hold. Only for count 2.
    pub squares: Option<[Vec<bool>; 2]>,
}

/// Extinction rule: loss by running out of a piece type.
pub struct ExtinctionRule {
    pub piece_types: Vec<usize>,
    /// `piece_types` as a flag per type, for the count that runs at every node.
    pub counted: [bool; MAX_PIECE_TYPES],
    pub any_piece: bool,
    pub min_count: usize,
    /// Tri-state, from the point of view of the side that ran out: -1 loss, 0 draw,
    /// 1 win. Antichess is a win, the horde a loss, and "draw" is now itself.
    pub result: i8,
}

impl ExtinctionRule {
    /// Whether a piece of this type is one the rule counts. `any_piece` is the
    /// caller's branch, not this one's: a rule that counts everything never asks.
    #[inline(always)]
    pub fn counts(&self, piece_type: usize) -> bool {
        self.counted[piece_type]
    }
}

/// A per-type flag table, built once from a list of types.
///
/// The three lists this replaces on the hot paths -- royal types, promoting types and
/// an extinction rule's counted types -- are each a handful of entries, and `contains`
/// on a handful is cheap; it was still a loop with a bounds check per entry, asked once
/// per piece per node from the innermost loop of move generation. A flag is one load.
fn type_flags(types: &[usize]) -> [bool; MAX_PIECE_TYPES] {
    let mut flags = [false; MAX_PIECE_TYPES];
    for piece_type in types {
        flags[*piece_type] = true;
    }
    flags
}

pub struct Tables {
    pub width: usize,
    pub height: usize,
    pub n_squares: usize,
    pub n_types: usize,

    /// How many owners a board byte can belong to: 2, or 3 once anything is wild.
    ///
    /// The one number the byte encoding, the value tables and the attack table all read.
    /// **Not** a third set of movement frames -- `n_frames` stays `2 * n_facings`,
    /// because a wild piece walks a route and never reads a movement table.
    pub n_owners: usize,
    /// How many readings of the movement vectors the tables below hold.
    ///
    /// Two per facing, one per player, flattened **player-major**: frame
    /// `player * n_facings + facing`. With one facing that is the player index, which is
    /// why `[player][type][square]` and `[frame][type][square]` are the same table for
    /// every ruleset before schema 8.
    pub n_frames: usize,
    /// How many orientations this ruleset's pieces have between them. 1 where nothing
    /// turns, which is what makes `has_facing` false and every decode below the identity.
    pub n_facings: usize,
    /// How many board values one player owns: `n_facings * n_types`.
    ///
    /// The stride of every "is this piece mine?" range test in the core. That those
    /// tests survived schema 8 as range tests at all is the whole argument for the
    /// player-major encoding -- the other spelling interleaves the two armies' values,
    /// and every one of those comparisons becomes a silent bug.
    pub side_stride: usize,
    /// `facing_axes[type]` -> how many orientations that type has; 1 for one that does
    /// not turn. Empty when nothing in the ruleset turns at all.
    pub facing_axes: Vec<usize>,
    /// `facing_initial[type]` -> the facing a piece of that type takes when it *enters
    /// play*: dropped from hand, promoted into, or left behind by a strike. One rule,
    /// three callers. Empty when nothing turns.
    pub facing_initial: Vec<usize>,
    pub has_facing: bool,
    /// `[board value] -> owner`, `-> frame`, `-> piece type`.
    ///
    /// Tables rather than arithmetic, for the reason everything else here is one: the
    /// encoding is stated once, where it is built, and no hot path divides by `n_types`
    /// to rediscover it. Index 0 is the empty square and every caller tests that first.
    value_owner: [u8; 256],
    value_frame: [u8; 256],
    value_type: [u8; 256],

    pub royal_types: Vec<usize>,
    /// `royal_types` as a flag per type. See `type_flags`.
    royal_flags: [bool; MAX_PIECE_TYPES],
    /// Whether each player's *setup* holds a royal (Q133), so that "this side's royal is
    /// gone" can be told apart from "this side never had one".
    ///
    /// Read off `initial_board` at build time rather than off the position a `Game`
    /// happens to start on, because it is a fact about the **ruleset**: a game built on a
    /// hand-cut fragment is still playing the game whose setup this is. Nothing crosses
    /// the Python boundary for it -- the compiled setup is already here.
    ///
    /// `rulesets/horde.json` is why it exists. One army is thirty-six pawns and no king,
    /// `win` is `checkmate`, and `extinction` is how that side loses; "a side with no
    /// royal has lost" would end every game of horde at ply zero.
    royal_at_start: [bool; 2],
    pub leaps: EntryTable,
    pub slides: EntryTable,
    pub hops: EntryTable,
    /// Schema 13: the fourth move table, shaped as `slides`. Empty and never indexed
    /// unless `has_sweep`.
    pub sweeps: EntryTable,
    pub has_sweep: bool,

    /// Every square on this board, as a set. What an "empty squares" query masks the
    /// complement of an occupancy with, so the bits past `n_squares` never read as
    /// squares.
    pub all_squares: SquareSet,

    attack_index: Vec<u32>,
    attack_sources: Vec<AttackSource>,
    attack_paths: Vec<u16>,
    /// Schema 13: the mode of every `attack_paths` square, read only for a sweep source.
    /// Empty unless `has_sweep`.
    attack_path_modes: Vec<u8>,

    /// `[player][piece type][square] -> this square's moves still need a uniqueness
    /// check`. Normally all false; the compiler resolves every case it can prove away.
    pub dedup: Vec<bool>,
    pub has_hop_attacks: bool,
    /// Per (attacker, royal square): every square whose occupancy could switch a hop
    /// attack on that royal on or off. Pure geometry, so it is precomputed here rather
    /// than memoised on first use as the Python engine does.
    hop_lines: Vec<SquareSet>,

    pub initial_board: [u8; MAX_SQUARES],
    pub initial_unmoved: SquareSet,

    pub win: Win,
    pub stalemate: Stalemate,
    pub checkmate_result: CheckmateResult,
    pub max_plies: u32,
    pub repetition_draw: u32,
    pub no_progress_draw: u32,

    /// `one_way[piece_type]` -> the piece can never return to a square it left, so a
    /// move by it is progress: it clears the repetition window and resets the
    /// no-progress clock. Derived from the movement tables by
    /// `compile._one_way_types` -- the core walks no graph and knows of no pawn.
    ///
    /// **This used to be `ep_type`** (Q113). The irreversible test asked
    /// `piece_type == ep_type`, and `ep_type` is set only when `en_passant` is declared,
    /// so whether a game's pawn reset its own clock depended on whether the game
    /// happened to declare an unrelated capturing rule. A flag per type also says what
    /// one `i32` could not: shogi has three such pieces, and one of them is not the
    /// promoting piece at all.
    one_way: [bool; MAX_PIECE_TYPES],

    pub ep_type: i32,
    /// `[frame * n_squares + origin]` -> the destinations that open an en passant
    /// window from that square, each with the square the mover stepped over.
    ///
    /// By frame, not by player: a pawn that can turn steps over a different square
    /// depending on which way it points, and an origin alone cannot say which.
    ///
    /// A table rather than arithmetic. The core used to halve the move's delta to find
    /// the crossed square, which is right on a board whose edges are edges and wrong on
    /// one whose opposite edges are joined -- and wrong silently, because nothing about
    /// it changes a move count. The compiler knows the line the piece moved along, so
    /// it answers instead.
    ep_passes: Vec<Vec<(u16, u16)>>,

    // --- Promotion (schema 2: per-piece, per-player, optional, zones) ---
    // `promo_squares[player][piece_type * n_squares + square]` -> true if promoting
    pub promo_squares: Vec<Vec<bool>>,
    // `promo_targets[player][piece_type]` -> list of promotion target types
    pub promo_targets: Vec<Vec<Vec<u32>>>,
    // `promo_optional[piece_type]` -> true if promotion can be declined
    pub promo_optional: Vec<bool>,
    // Which piece types promote at all (have a promotion rule)
    pub promo_piece_types: Vec<usize>,
    /// `promo_piece_types` as a flag per type: `is_promoting` is asked once per piece
    /// per node from the movement generator's outer loop.
    promo_flags: [bool; MAX_PIECE_TYPES],
    /// Schema 15, `promotion.on_leaving`: the types that also promote on a move that
    /// *starts* in their zone, wherever it lands. Asked once per promoting piece per node,
    /// beside `is_promoting`; all false for every ruleset before it.
    promo_leaving_flags: [bool; MAX_PIECE_TYPES],

    pub castling: Option<(usize, usize, i32)>,
    /// Schema 2 castling by destination file: (king_qs, king_ks, rook_qs, rook_ks).
    pub castle_files: Option<(i32, i32, i32, i32)>,

    // --- Schema 2 runtime features ---
    /// `capture_allowed[attacker * n_types + victim]` -- schema 3's capture matrix.
    ///
    /// All true unless a piece declared `captures`, which is why the feature is inert
    /// on every ruleset written before it. It cannot be folded into the per-square
    /// tables the way a region or a lame leap is: what a move may do depends on the
    /// *victim's* type, which belongs to the position rather than to the geometry.
    pub capture_allowed: Vec<bool>,
    /// Whether any piece restricts what it may capture. False skips every lookup, which
    /// is every committed ruleset.
    pub has_capture_matrix: bool,

    pub multi_royal: bool,
    /// `demotions[piece_type]` -> what it reverts to when captured.
    pub demotions: Vec<usize>,
    pub explosion: Option<ExplosionRule>,
    /// Schema 6. `None` means a captured piece leaves the board, which is every game
    /// before this one and every strike ruleset whose chain keeps nothing alive.
    pub strike: Option<StrikeRule>,
    pub drops: Option<DropRule>,
    pub reach: Option<ReachRule>,
    /// `[player * n_types * n_squares + piece_type * n_squares + square]` -> how many
    /// moves that piece needs, on an empty board, to reach a square a `reach` rule wins
    /// on. `GOAL_UNREACHABLE` where it cannot. Empty when there is no reach rule.
    ///
    /// Compiled in Python beside the attack sources, because it is a fact about the
    /// rules rather than an agent's opinion: "how far is this piece from a winning
    /// square" answered with geometry is wrong about every leaper.
    pub goal_distance: Vec<u8>,
    /// The largest finite entry above -- this board's diameter *for these pieces*, and
    /// the only honest normaliser for a distance measured in moves.
    pub goal_diameter: f64,
    /// Q708 (F1): the same distance to each goal square alone,
    /// `[player][goal][piece_type][square]`, goals in `ReachRule::goals`' order. Empty
    /// unless a priced race gives a side more than one goal square; read through
    /// `goal_each_slot` only when some but not all of a side's goals are sealed.
    pub goal_distance_each: Vec<u8>,
    pub extinction: Option<ExtinctionRule>,
    /// Schema 3. `None` keeps the draw the clocks have always produced.
    pub scoring: Option<ScoringRule>,
    /// Schema 4. `None` means the board is just a board.
    pub territory: Option<TerritoryRule>,
    /// Schema 4. `None` means one move per turn, which is every game so far.
    pub turn: Option<TurnRule>,

    // --- Wild pieces (schema 9) ---
    /// Whether anything on this board is nobody's. False is every game before schema 9,
    /// and it is the branch that keeps the phase, the third owner's attack block and the
    /// tally free for them.
    pub has_npcs: bool,
    /// `npc_next[square]` -> the square a wild piece standing here steps to. `-1` off
    /// every route; the square itself for a one-square route, which is a wild piece that
    /// never moves. Empty when nothing is wild.
    pub npc_next: Vec<i16>,
    /// `npc_bounty[square]` -> what taking the wild piece that walks this square is
    /// worth. `0.0` off every route. Empty when nothing is wild.
    pub npc_bounty: Vec<f64>,
    /// Every route square, ascending: the only squares a wild piece can ever stand on.
    ///
    /// Derived here from `npc_next` rather than crossing the boundary, because it is the
    /// same fact stated shorter. The phase walks it to find the wild pieces instead of
    /// scanning the board or reading the occupancy sets -- the sets are stale inside a
    /// legality filter, which writes the board directly, and the board is up to 256
    /// squares for what is usually two pieces.
    pub npc_squares: Vec<u16>,
    /// Every bounty on the board at ply zero: what a `points` tally can add up to.
    /// The ceiling `features.rs` measures a points gap against, and 0.0 when nothing is
    /// wild.
    pub npc_bounty_total: f64,

    /// Schema 9's second key, `rules.promote_on_entry`: what a piece that *enters play*
    /// on a square becomes, as `[player][piece_type][square]` flattened in that order,
    /// or `-1` where nothing happens.
    ///
    /// **Empty when the rule is off**, which is every game before this one -- the inert
    /// default, so no earlier ruleset carries it. The compiler has already chosen the
    /// target and already decided that `mandatory` does not apply, so what is here is the
    /// answer and not the rule.
    entry_promotion: Vec<i16>,
    /// Declared piece values, indexed by type. Used by a material scoring rule and by
    /// nothing else -- the agents get their own values handed to them, because what a
    /// piece is *worth* is an opinion and what a rule *counts* must not be.
    pub piece_values: Vec<f64>,
    pub must_capture: bool,
    pub forbid_check: bool,
    pub forbid_immobility: bool,
    pub royal_facing_forbidden: bool,
    pub check_limit: u32,
    /// `None` when the rule is off; otherwise tri-state from the point of view of the
    /// side reduced to royalty alone: -1 loss (shatranj), 0 draw, 1 win. The off
    /// switch is the `Option` rather than a sentinel value, because "draw" is a
    /// result the validator accepts and once shared the sentinel with "off".
    pub bare_royal: Option<i8>,
    /// `immobile_squares[player * n_types * n_squares + piece_type * n_squares + square]`
    pub immobile_squares: Vec<bool>,

    zobrist_pieces: Vec<u64>,
    pub zobrist_side: u64,
    zobrist_ep: Vec<u64>,
    /// `[colour * n_squares + square]`, for a repainting territory rule. Colour 0 is in
    /// the table and never read: the entry exists so the index arithmetic needs no
    /// special case, and an uncoloured square contributes nothing to the key.
    zobrist_colour: Vec<u64>,
    /// One per possible moves-remaining count, for a ruleset whose turn is longer than
    /// one move. Without it two positions identical but for how many moves are left in
    /// the turn collide, and a repetition test calls them the same position.
    zobrist_turn: Vec<u64>,

    /// `facing_lines[square]` -> the far ends of the file lines from that square, each
    /// with the squares that have to be empty for the line to be open.
    ///
    /// The flying general, as a table, for exactly the reason `ep_passes` is one:
    /// `royals_face` used to walk a file by `+= width` asking what stood on each square,
    /// which is right only on a board whose edges are edges and whose empty squares are
    /// the only empty squares. It was refused beside a seam and refused again beside a
    /// wall; it is neither now. Joined ranks give a square two entries for the same far
    /// end, one each way round, and the rule is satisfied if either path is clear.
    ///
    /// Empty for every ruleset that does not declare the rule.
    facing_lines: Vec<Vec<(u16, u32, u32)>>,
    facing_paths: Vec<u16>,
}

impl Tables {
    /// The value a piece takes when it **enters play**: dropped from hand, promoted
    /// into, or left standing by a strike that changed what it is.
    ///
    /// One rule for all three, and it is the only rule the schema offers -- a piece that
    /// was not on the board a moment ago has no orientation to keep, so it takes its
    /// type's declared initial one. Named for that rather than `encode_piece`, because
    /// the old name read like "the value of this piece" and there is no longer one value
    /// per (player, type).
    #[inline(always)]
    pub fn entering_value(&self, player: usize, piece_type: usize) -> u8 {
        let facing = match self.facing_initial.get(piece_type) {
            Some(facing) => *facing,
            None => 0,
        };
        self.piece_value(player, facing, piece_type)
    }

    /// The value a piece takes when it enters play **on a particular square**: schema 9's
    /// `promote_on_entry`, which turns a piece appearing inside its own promotion zone
    /// into the first type that rule lists for its owner.
    ///
    /// **One method is the whole rule, and it has four callers** -- the drop branch of
    /// `make`, `strike_writes`, the immobility filter and the drop-mate test. Stated once
    /// because a rule stated in more than one place is a rule that can be stated
    /// differently in one of them; `entering_value` where the table is empty, which is
    /// every ruleset before this one.
    #[inline]
    pub fn arriving_value(&self, player: usize, piece_type: usize, square: usize) -> u8 {
        if self.entry_promotion.is_empty() {
            return self.entering_value(player, piece_type);
        }
        let slot = (player * self.n_types + piece_type) * self.n_squares + square;
        match self.entry_promotion[slot] {
            becomes if becomes >= 0 => self.entering_value(player, becomes as usize),
            _ => self.entering_value(player, piece_type),
        }
    }

    /// The board value of one piece at an explicit facing.
    #[inline(always)]
    pub fn piece_value(&self, player: usize, facing: usize, piece_type: usize) -> u8 {
        ((player * self.n_facings + facing) * self.n_types + piece_type + 1) as u8
    }

    /// The lowest board value `player` owns. Their values run from here for
    /// `side_stride`, which is what keeps every ownership test a range comparison.
    #[inline(always)]
    pub fn value_low(&self, player: usize) -> u16 {
        (player * self.side_stride + 1) as u16
    }

    #[inline(always)]
    pub fn owner_of(&self, value: u8) -> usize {
        self.value_owner[value as usize] as usize
    }

    #[inline(always)]
    pub fn type_of(&self, value: u8) -> usize {
        self.value_type[value as usize] as usize
    }

    /// Which frame a piece reads its movement out of, straight off its board value.
    ///
    /// **Never asked of a wild byte.** A wild piece reads no movement table, so schema 9
    /// gave it no frame: `value_frame` of its byte is `2 * n_facings`, one past the last
    /// real frame, so a table indexed with it panics rather than silently reading
    /// somebody else's moves. `decode_owned` asserts the value belongs to the player it
    /// was handed.
    ///
    /// `[player][type][square]` was a lie the tables told since the beginning: that
    /// first index has always meant *which way is forward*, and it coincided with the
    /// player only while nothing could turn. This is where the two stop coinciding, and
    /// the answer comes from the piece rather than from whose turn it is.
    #[inline(always)]
    pub fn frame_of_value(&self, value: u8) -> usize {
        self.value_frame[value as usize] as usize
    }

    /// `(piece type, frame)` for a value already known to belong to `player`, whose
    /// `value_low` the caller has in hand.
    ///
    /// The one decode on the movement generator's inner loop, and the only place the
    /// encoding is read twice. The `else` arm is what the loop did before schema 8 --
    /// one subtraction and no load -- and it is exactly right whenever there is one
    /// frame per player: `value - low` is then the type, and the frame *is* the player.
    /// The branch is on a constant of the tables, so it is predicted perfectly and the
    /// rulesets that do not turn pay nothing at all for the ones that do; without it
    /// chess perft is 3.5% slower for a feature it does not use.
    ///
    /// `debug_assert` is what keeps that from being a second, driftable statement of
    /// the encoding: in a debug build and in `cargo test` the fast arm is checked
    /// against the tables it is standing in for, every time it is taken.
    #[inline(always)]
    pub fn decode_owned(&self, value: u8, low: u16, player: usize) -> (usize, usize) {
        // Schema 9's guard: movegen walks `occupied()[side]`, so a wild byte can never
        // arrive here -- and if one ever did, the fast arm would hand back the player as
        // the frame and the slow arm a frame no table has. Asserted rather than handled,
        // because there is no right answer for a piece that reads no movement table.
        debug_assert_eq!(self.owner_of(value), player, "a wild byte reached decode_owned");
        if self.has_facing {
            return (self.type_of(value), self.frame_of_value(value));
        }
        let quick = ((value as u16 - low) as usize, player);
        debug_assert_eq!(quick, (self.type_of(value), self.frame_of_value(value)));
        quick
    }

    /// Whether a board value is a piece of this player's of this type, whichever way it
    /// happens to be pointing. What a scan for "where is the king" or "is there already
    /// a pawn on this file" means once a type is several values.
    #[inline(always)]
    pub fn is_piece(&self, value: u8, player: usize, piece_type: usize) -> bool {
        value != 0 && self.owner_of(value) == player && self.type_of(value) == piece_type
    }

    /// How many orientations a piece type has: 1 for one that does not turn.
    #[inline(always)]
    pub fn facings_of(&self, piece_type: usize) -> usize {
        match self.facing_axes.get(piece_type) {
            Some(axes) => *axes,
            None => 1,
        }
    }

    /// Where one `[frame][type][square]` table's entries for this slot begin.
    #[inline(always)]
    pub fn slot(&self, frame: usize, piece_type: usize, square: usize) -> usize {
        (frame * self.n_types + piece_type) * self.n_squares + square
    }

    /// Where `goal_distance`'s entry for this piece begins.
    ///
    /// By **player**, not by frame, and that is not an oversight: a piece that can turn
    /// is measured as though turning were free, so the distances of all its frames are
    /// unioned into one answer per player. The compiler's `_goal_distances` states why.
    /// Spelled as its own function rather than as `slot(player, ..)` so that a reader
    /// cannot mistake the table for a frame-indexed one and index it with a frame.
    #[inline(always)]
    pub fn goal_slot(&self, player: usize, piece_type: usize, square: usize) -> usize {
        (player * self.n_types + piece_type) * self.n_squares + square
    }

    /// Where `goal_distance_each`'s entry for this piece and the `goal`-th of `player`'s
    /// goal squares begins. The first player's goals come first, so the second player's
    /// block starts after however many the first has.
    #[inline(always)]
    pub fn goal_each_slot(
        &self,
        rule: &ReachRule,
        player: usize,
        goal: usize,
        piece_type: usize,
        square: usize,
    ) -> usize {
        let before = if player == 0 { 0 } else { rule.goals[0].len() };
        ((before + goal) * self.n_types + piece_type) * self.n_squares + square
    }

    /// The ways `owner` could attack `square`. `owner` is a player, or `WILD` once the
    /// ruleset has wild pieces -- the compiler built a third block from the routes and
    /// this is the same slice walk over it.
    #[inline(always)]
    pub fn attackers_of(&self, player: usize, square: usize) -> &[AttackSource] {
        debug_assert!(player < self.n_owners, "no attack block for owner {player}");
        let slot = player * self.n_squares + square;
        let lo = self.attack_index[slot] as usize;
        let hi = self.attack_index[slot + 1] as usize;
        &self.attack_sources[lo..hi]
    }

    #[inline(always)]
    pub fn attack_path(&self, source: &AttackSource) -> &[u16] {
        let lo = source.path_start as usize;
        &self.attack_paths[lo..lo + source.path_len as usize]
    }

    /// Schema 13: the modes of a sweep source's path squares, parallel to `attack_path`.
    #[inline(always)]
    pub fn attack_path_modes(&self, source: &AttackSource) -> &[u8] {
        let lo = source.path_start as usize;
        &self.attack_path_modes[lo..lo + source.path_len as usize]
    }

    /// Schema 13: the squares a sweep by `moving` from `origin` to `target` passes over,
    /// in ray order, or nothing when no sweep of that piece reaches `target`.
    ///
    /// A scan of the slot's few sweep entries rather than a per-pair table. One ray
    /// reaches one target from one origin, because `wrap` is refused beside a sweep.
    pub fn sweep_path(&self, moving: u8, origin: usize, target: usize) -> &[u16] {
        if !self.has_sweep || moving == 0 {
            return &[];
        }
        let slot = self.slot(self.frame_of_value(moving), self.type_of(moving), origin);
        for entry in self.sweeps.entries(slot) {
            let squares = self.sweeps.squares(entry);
            if let Some(index) = squares.iter().position(|square| *square as usize == target) {
                return &squares[..index];
            }
        }
        &[]
    }

    #[inline(always)]
    pub fn zobrist(&self, value: u8, square: usize) -> u64 {
        self.zobrist_pieces[value as usize * self.n_squares + square]
    }

    /// The key contribution of the moves left in this turn. Zero when the turn is one
    /// move long, which is when there is nothing to tell apart.
    #[inline(always)]
    pub fn zobrist_remaining(&self, remaining: u32) -> u64 {
        match self.zobrist_turn.get(remaining as usize) {
            Some(value) if self.turn.is_some() => *value,
            _ => 0,
        }
    }

    /// The key contribution of one square's colour. Zero for an uncoloured square, so
    /// a caller never has to ask whether there is a colour to XOR.
    #[inline(always)]
    pub fn zobrist_colour(&self, colour: u8, square: usize) -> u64 {
        if colour == 0 {
            0
        } else {
            self.zobrist_colour[colour as usize * self.n_squares + square]
        }
    }

    #[inline(always)]
    pub fn zobrist_ep_file(&self, square: i16) -> u64 {
        self.zobrist_ep[square as usize % self.width]
    }

    #[inline(always)]
    pub fn needs_check_filter(&self) -> bool {
        self.win == Win::Checkmate && !self.royal_types.is_empty()
    }

    /// Whether a royal off the board settles the game against `side` (Q133).
    ///
    /// Two facts, both of them already here, and neither of them a declaration.
    ///
    /// * **The victory condition is about the royal.** `Checkmate` says it must never be
    ///   takeable and `RoyalCapture` says taking it is the win; both say the same thing
    ///   about one that is gone. Under `Annihilation` or `None` a royal is a piece with a
    ///   rule attached -- something for `forbid_check` or `royal_facing` to name -- and
    ///   its loss is not the game's ending.
    /// * **The side had one to lose.** See `royal_at_start`: losing your royal is what
    ///   loses, and never having had one is a different game.
    ///
    /// Deliberately *not* keyed on `explode_on_capture`, which is what used to decide it.
    /// A blast is the commonest way to empty a royal square and is not the reason an
    /// empty one decides anything. See `docs/reference/derived-royal-loss.md`.
    #[inline(always)]
    pub fn royal_loss_ends_it(&self, side: usize) -> bool {
        self.royal_at_start[side] && matches!(self.win, Win::Checkmate | Win::RoyalCapture)
    }

    /// The square a piece stepped over to reach `target` from `origin`, if that move
    /// is one that opens an en passant window.
    #[inline]
    pub fn ep_passed(&self, frame: usize, origin: usize, target: usize) -> Option<usize> {
        self.ep_passes[frame * self.n_squares + origin]
            .iter()
            .find(|(destination, _)| *destination as usize == target)
            .map(|(_, passed)| *passed as usize)
    }

    /// The file lines out of `square`, each as `(far end, path start, path length)`.
    #[inline]
    pub fn facing_from(&self, square: usize) -> &[(u16, u32, u32)] {
        match self.facing_lines.get(square) {
            Some(lines) => lines,
            None => &[],
        }
    }

    /// The squares that have to be empty for one of `facing_from`'s lines to be open.
    #[inline]
    pub fn facing_path(&self, start: u32, length: u32) -> &[u16] {
        &self.facing_paths[start as usize..(start + length) as usize]
    }

    /// Whether a piece type is a promoting piece.
    #[inline(always)]
    pub fn is_promoting(&self, piece_type: usize) -> bool {
        self.promo_flags[piece_type]
    }

    /// Whether a piece type is royal. One load, for the callers that ask per piece per
    /// node; `royal_types` itself is still the list for anything that walks the royals.
    #[inline(always)]
    pub fn is_royal(&self, piece_type: usize) -> bool {
        self.royal_flags[piece_type]
    }

    /// Whether a move of this type that starts in its promotion zone promotes wherever
    /// it lands (schema 15). The zone is still `promotes_on`'s; this says only that the
    /// origin counts as well as the target.
    #[inline(always)]
    pub fn promotes_on_leaving(&self, piece_type: usize) -> bool {
        self.promo_leaving_flags[piece_type]
    }

    /// Whether this piece promotes on this square for this player.
    #[inline]
    pub fn promotes_on(&self, player: usize, piece_type: usize, square: usize) -> bool {
        self.promo_squares[player][piece_type * self.n_squares + square]
    }

    /// Promotion targets for a piece type for a player.
    #[inline]
    pub fn promotion_targets(&self, player: usize, piece_type: usize) -> &[u32] {
        &self.promo_targets[player][piece_type]
    }

    /// Whether this piece's promotion is optional.
    #[inline]
    pub fn promotion_optional(&self, piece_type: usize) -> bool {
        self.promo_optional[piece_type]
    }

    /// Whether a move by this piece can never be taken back. See `one_way`.
    #[inline(always)]
    pub fn is_one_way(&self, piece_type: usize) -> bool {
        self.one_way[piece_type]
    }

    /// Whether a square strands a piece of the given type reading the given frame.
    #[inline]
    pub fn is_immobile_square(&self, frame: usize, piece_type: usize, square: usize) -> bool {
        self.immobile_squares[self.slot(frame, piece_type, square)]
    }

    /// Whether a piece of `attacker` type may capture one of `victim` type.
    ///
    /// Hot: this sits on the capture test in move generation and on every consumer of
    /// `attackers_of`. The `has_capture_matrix` short-circuit is what keeps it free for
    /// the rulesets that do not use it, which is all twenty-two committed ones.
    #[inline(always)]
    pub fn may_capture(&self, attacker: usize, victim: usize) -> bool {
        !self.has_capture_matrix || self.capture_allowed[attacker * self.n_types + victim]
    }

    /// Whether the game has any schema-2 ending rules.
    #[inline]
    pub fn has_v2_endings(&self) -> bool {
        self.reach.is_some()
            || self.territory.as_ref().is_some_and(|t| t.win_at > 0)
            || self.extinction.is_some()
            || self.bare_royal.is_some()
            || self.check_limit > 0
    }

    /// Whether this ruleset declares any of the endings the evaluator has a term for.
    ///
    /// The gate that makes stage 17A free: a ruleset that states none of these is
    /// evaluated by exactly the arithmetic it always was, so every committed number
    /// stands. It is deliberately narrower than `has_v2_endings` -- reach, extinction
    /// and a check limit are endings the evaluator still cannot read, and claiming them
    /// here would cost a calibration campaign for a term that does not exist.
    #[inline]
    pub fn has_goal_terms(&self) -> bool {
        self.territory.as_ref().is_some_and(|t| t.win_at > 0)
            || self.scoring.is_some()
            || self.reach.as_ref().is_some_and(|rule| self.reach_is_priceable(rule))
            || self.check_limit > 0
            || self.extinction.as_ref().is_some_and(|rule| rule.result != 0)
    }

    /// Whether a `reach` rule is one the evaluator may put a term on.
    ///
    /// Two rules out, both measured rather than reasoned about, and both the same shape:
    /// a distance term is all reward and no risk unless the *journey* is priced.
    ///
    /// **`safe`.** The table counts moves to a square and knows nothing about who is
    /// looking at it, so under `safe` it overstates progress systematically -- the
    /// search is told to race towards squares it will be refused when it arrives.
    ///
    /// **A royal racing piece.** `ROYAL_VALUE` is zero, deliberately: losing a royal is
    /// terminal rather than material, so the evaluator has no term for it. A distance
    /// term on a royal therefore rewards marching it forward with *nothing* opposing it
    /// except the search's own horizon. King of the hill is what that looks like: with
    /// a term of any weight at all, mean game length falls from 54.7 plies to 20 and
    /// `quick_kill_rate` goes from 0.041 to over 0.54 -- past the gate, at the smallest
    /// weight tested. `piece_type < 0` means "any piece", which is not one-sided,
    /// because the nearest piece to the goal is usually not the royal one.
    pub fn reach_is_priceable(&self, rule: &ReachRule) -> bool {
        !rule.safe && (rule.piece_type < 0 || !self.is_royal(rule.piece_type as usize))
    }

    #[inline(always)]
    pub fn hop_line(&self, player: usize, square: usize) -> SquareSet {
        self.hop_lines[player * self.n_squares + square]
    }

    fn build_hop_lines(&self) -> Vec<SquareSet> {
        if !self.has_hop_attacks {
            return vec![SquareSet::EMPTY; 2 * self.n_squares];
        }
        let mut lines = vec![SquareSet::EMPTY; 2 * self.n_squares];
        for player in 0..2 {
            for square in 0..self.n_squares {
                let mut set = SquareSet::EMPTY;
                for source in self.attackers_of(player, square) {
                    if source.screens > 0 {
                        set.set(source.origin as usize);
                        for intervening in self.attack_path(source) {
                            set.set(*intervening as usize);
                        }
                    }
                }
                lines[player * self.n_squares + square] = set;
            }
        }
        lines
    }
}

impl Tables {
    pub fn from_py(source: &Bound<'_, PyDict>) -> PyResult<Self> {
        let width: usize = field(source, keys::WIDTH)?;
        let height: usize = field(source, keys::HEIGHT)?;
        let n_squares: usize = field(source, keys::N_SQUARES)?;
        let n_types: usize = field(source, keys::N_TYPES)?;
        if n_types > MAX_PIECE_TYPES {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{n_types} piece types; the core is built for at most {MAX_PIECE_TYPES},                  which is what the validator enforces"
            )));
        }

        let packed_sources: Vec<u32> = field(source, keys::ATTACK_SRCS)?;
        // Schema 13: one byte per source, or empty when nothing sweeps.
        let attack_sweep: Vec<u8> = field(source, keys::ATTACK_SWEEP)?;
        let attack_sources: Vec<AttackSource> = packed_sources
            .chunks_exact(8)
            .enumerate()
            .map(|(index, chunk)| AttackSource {
                origin: chunk[0] as u16,
                piece_value: chunk[1] as u8,
                screens: chunk[2] as u8,
                requires_unmoved: chunk[3] != 0,
                path_start: chunk[4],
                path_len: chunk[5],
                prefix_len: chunk[6],
                screen_owner: chunk[7] as u8,
                sweep: attack_sweep.get(index).copied().unwrap_or(0),
            })
            .collect();
        if !attack_sweep.is_empty() && attack_sweep.len() != attack_sources.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "attack_sweep holds {} bytes for {} attack sources",
                attack_sweep.len(), attack_sources.len(),
            )));
        }

        let ep_pass_flat: Vec<u32> = field(source, keys::EP_PASS_FLAT)?;
        let ep_pass_index: Vec<u32> = field(source, keys::EP_PASS_INDEX)?;
        let ep_passes: Vec<Vec<(u16, u16)>> = (0..ep_pass_index.len() - 1)
            .map(|slot| {
                let start = ep_pass_index[slot] as usize;
                let end = ep_pass_index[slot + 1] as usize;
                ep_pass_flat[start * 2..end * 2]
                    .chunks_exact(2)
                    .map(|pair| (pair[0] as u16, pair[1] as u16))
                    .collect()
            })
            .collect();

        let facing_flat: Vec<u32> = field(source, keys::FACING_ENTRIES)?;
        let facing_index: Vec<u32> = field(source, keys::FACING_INDEX)?;
        let facing_paths: Vec<u16> = field::<Vec<u32>>(source, keys::FACING_PATHS)?
            .into_iter()
            .map(|square| square as u16)
            .collect();
        let facing_lines: Vec<Vec<(u16, u32, u32)>> = (0..facing_index.len().saturating_sub(1))
            .map(|slot| {
                let start = facing_index[slot] as usize;
                let end = facing_index[slot + 1] as usize;
                facing_flat[start * 3..end * 3]
                    .chunks_exact(3)
                    .map(|entry| (entry[0] as u16, entry[1], entry[2]))
                    .collect()
            })
            .collect();

        let initial: Vec<u8> = field(source, keys::INITIAL_BOARD)?;
        let mut initial_board = [0u8; MAX_SQUARES];
        initial_board[..initial.len()].copy_from_slice(&initial);

        let win = match field::<String>(source, keys::WIN)?.as_str() {
            "checkmate" => Win::Checkmate,
            "royal_capture" => Win::RoyalCapture,
            "annihilation" => Win::Annihilation,
            "none" => Win::None,
            other => return Err(unknown("win condition", other)),
        };
        let stalemate = match field::<String>(source, keys::STALEMATE)?.as_str() {
            "draw" => Stalemate::Draw,
            "loss" => Stalemate::Loss,
            "win" => Stalemate::Win,
            other => return Err(unknown("stalemate rule", other)),
        };
        let checkmate_result = match field::<String>(source, keys::CHECKMATE_RESULT)?.as_str() {
            "loss" => CheckmateResult::Loss,
            "win" => CheckmateResult::Win,
            other => return Err(unknown("checkmate result", other)),
        };

        let castle_king: i32 = field(source, keys::CASTLE_KING)?;
        let castling = if castle_king < 0 {
            None
        } else {
            Some((
                castle_king as usize,
                field::<i32>(source, keys::CASTLE_ROOK)? as usize,
                field::<i32>(source, keys::CASTLE_STEPS)?,
            ))
        };

        // Castle by file (schema 2)
        let castle_king_files: Vec<i32> = field(source, keys::CASTLE_KING_FILES)?;
        let castle_files = if castle_king_files[0] < 0 {
            None
        } else {
            let castle_rook_files: Vec<i32> = field(source, keys::CASTLE_ROOK_FILES)?;
            Some((castle_king_files[0], castle_king_files[1],
                  castle_rook_files[0], castle_rook_files[1]))
        };

        // Promotion tables (schema 2: per-piece, per-player, optional, zones)
        let promo_squares_flat: Vec<u8> = field(source, keys::PROMO_SQUARES)?;
        let promo_targets_flat: Vec<i32> = field(source, keys::PROMO_TARGETS_FLAT)?;
        let promo_targets_index: Vec<u32> = field(source, keys::PROMO_TARGETS_INDEX)?;
        let promo_optional_raw: Vec<u8> = field(source, keys::PROMO_OPTIONAL)?;
        let promo_piece_types: Vec<usize> = field(source, keys::PROMO_PIECE_TYPES)?;
        let promo_on_leaving: Vec<usize> = field(source, keys::PROMO_ON_LEAVING)?;

        // Rebuild promo_squares as Vec<Vec<bool>> [player][piece_type * n_squares + square]
        let player_stride = n_types * n_squares;
        let promo_squares = vec![
            promo_squares_flat[..player_stride].iter().map(|v| *v != 0).collect(),
            promo_squares_flat[player_stride..].iter().map(|v| *v != 0).collect(),
        ];

        // Rebuild promo_targets as Vec<Vec<Vec<u32>>> [player][piece_type] -> Vec<u32>
        let mut promo_targets: Vec<Vec<Vec<u32>>> = Vec::new();
        for player in 0..2 {
            let mut per_type: Vec<Vec<u32>> = Vec::new();
            for pt in 0..n_types {
                let idx = player * n_types + pt;
                let start = promo_targets_index[idx] as usize;
                let end = promo_targets_index[idx + 1] as usize;
                per_type.push(promo_targets_flat[start..end].iter().map(|v| *v as u32).collect());
            }
            promo_targets.push(per_type);
        }

        let promo_optional: Vec<bool> = promo_optional_raw.iter().map(|v| *v != 0).collect();

        // Schema 2 runtime features
        let multi_royal: bool = field(source, keys::MULTI_ROYAL)?;
        let demotions: Vec<usize> = field(source, keys::DEMOTIONS)?;
        let must_capture: bool = field(source, keys::MUST_CAPTURE)?;
        let forbid_check: bool = field(source, keys::FORBID_CHECK)?;
        let forbid_immobility: bool = field(source, keys::FORBID_IMMOBILITY)?;
        let royal_facing_forbidden: bool = field(source, keys::ROYAL_FACING_FORBIDDEN)?;
        let check_limit: u32 = field(source, keys::CHECK_LIMIT)?;
        let has_bare_royal: bool = field(source, keys::HAS_BARE_ROYAL)?;
        let capture_allowed_raw: Vec<u8> = field(source, keys::CAPTURE_ALLOWED)?;
        let capture_allowed: Vec<bool> = capture_allowed_raw.iter().map(|v| *v != 0).collect();
        let has_capture_matrix: bool = field(source, keys::HAS_CAPTURE_MATRIX)?;

        let bare_royal: Option<i8> = if has_bare_royal {
            Some(field(source, keys::BARE_ROYAL)?)
        } else {
            None
        };

        let immobile_squares_raw: Vec<u8> = field(source, keys::IMMOBILE_SQUARES)?;
        let immobile_squares: Vec<bool> = immobile_squares_raw.iter().map(|v| *v != 0).collect();

        // Explosion
        let has_explosion: bool = field(source, keys::HAS_EXPLOSION)?;
        let explosion = if has_explosion {
            let immune_raw: Vec<u8> = field(source, keys::EXPLOSION_IMMUNE)?;
            let blast_flat: Vec<i32> = field(source, keys::EXPLOSION_BLAST)?;
            let blast_index: Vec<u32> = field(source, keys::EXPLOSION_BLAST_INDEX)?;
            let mut blast: Vec<Vec<usize>> = Vec::with_capacity(n_squares);
            for sq in 0..n_squares {
                let start = blast_index[sq] as usize;
                let end = blast_index[sq + 1] as usize;
                blast.push(blast_flat[start..end].iter().map(|v| *v as usize).collect());
            }
            let by_raw: Vec<u8> = field(source, keys::EXPLOSION_BY)?;
            let of_raw: Vec<u8> = field(source, keys::EXPLOSION_OF)?;
            Some(ExplosionRule {
                immune_types: immune_raw.iter().map(|v| *v != 0).collect(),
                blast,
                by_types: by_raw.iter().map(|v| *v != 0).collect(),
                of_types: of_raw.iter().map(|v| *v != 0).collect(),
            })
        } else {
            None
        };

        // Strike
        let has_strike: bool = field(source, keys::HAS_STRIKE)?;
        let strike = if has_strike {
            let position = match field::<u8>(source, keys::STRIKE_POSITION)? {
                0 => StrikePosition::Stay,
                1 => StrikePosition::Advance,
                2 => StrikePosition::Swap,
                3 => StrikePosition::Choose,
                other => return Err(unknown("strike position", &other.to_string())),
            };
            let landing_squares: Vec<u16> = field(source, keys::STRIKE_LANDING)?;
            let landing_index: Vec<u32> = field(source, keys::STRIKE_LANDING_INDEX)?;
            let expected = if position == StrikePosition::Choose { 2 * n_squares + 1 } else { 0 };
            if landing_index.len() != expected {
                return Err(unknown("strike landing index length", &landing_index.len().to_string()));
            }
            // Two encodings, one table. `contact_onto_to` is the general form and
            // crosses only when the rule genuinely needs it; otherwise the schema-6
            // array crosses and is widened here by giving every attacker the same row.
            // Widening in the loader rather than in the bridge is what keeps the
            // canonical tables of every ruleset written before schema 10 byte-identical
            // while the core below carries exactly one rule.
            let contact_onto: Vec<i32> = field(source, keys::CONTACT_ONTO_TO)?;
            let (to, mover_to) = if contact_onto.is_empty() {
                let compact: Vec<i32> = field(source, keys::STRIKE_TO)?;
                let mut to = Vec::with_capacity(n_types * n_types);
                for _ in 0..n_types {
                    to.extend_from_slice(&compact);
                }
                (to, vec![-1; n_types * n_types])
            } else {
                (contact_onto, field(source, keys::CONTACT_MOVER_TO)?)
            };
            let own_to: Vec<i32> = field(source, keys::CONTACT_OWN_ONTO_TO)?;
            let own_mover_to: Vec<i32> = field(source, keys::CONTACT_OWN_MOVER_TO)?;
            let own_link = own_to
                .iter()
                .zip(&own_mover_to)
                .map(|(&onto, &mover)| onto >= 0 || mover >= 0)
                .collect();
            Some(StrikeRule {
                to,
                mover_to,
                own_to,
                own_mover_to,
                own_link,
                n_types,
                position,
                landing_index,
                landing_squares,
                landing_stride: n_squares,
            })
        } else {
            None
        };

        // Drops
        let has_drops: bool = field(source, keys::HAS_DROPS)?;
        let drops = if has_drops {
            let from_captures: bool = field(source, keys::DROPS_FROM_CAPTURES)?;
            let regions_flat: Vec<i32> = field(source, keys::DROPS_REGIONS)?;
            // regions_flat: [player][piece_type][square] flattened; -1 means "any"
            // We represent as Vec<Vec<Option<Vec<bool>>>> [player][piece_type]
            let mut regions: Vec<Vec<Option<Vec<bool>>>> = Vec::new();
            for player in 0..2 {
                let mut per_type: Vec<Option<Vec<bool>>> = Vec::new();
                for pt in 0..n_types {
                    let base = (player * n_types + pt) * n_squares;
                    if regions_flat[base] < 0 {
                        // -1 sentinel: any empty square
                        per_type.push(None);
                    } else {
                        per_type.push(Some(
                            regions_flat[base..base + n_squares].iter().map(|v| *v != 0).collect()
                        ));
                    }
                }
                regions.push(per_type);
            }
            let no_doubled_raw: Vec<u8> = field(source, keys::DROPS_NO_DOUBLED)?;
            let no_mate_raw: Vec<u8> = field(source, keys::DROPS_NO_MATE)?;
            let stock: Vec<u32> = field(source, keys::DROPS_STOCK)?;
            let unlimited_raw: Vec<u8> = field(source, keys::DROPS_UNLIMITED)?;
            Some(DropRule {
                from_captures,
                regions,
                no_doubled: no_doubled_raw.iter().map(|v| *v != 0).collect(),
                no_mate: no_mate_raw.iter().map(|v| *v != 0).collect(),
                stock,
                unlimited: unlimited_raw.iter().map(|v| *v != 0).collect(),
            })
        } else {
            None
        };

        // Reach
        let goal_distance: Vec<u8> = field(source, keys::GOAL_DISTANCE)?;
        let goal_diameter: f64 = field::<u32>(source, keys::GOAL_DIAMETER)? as f64;
        let goal_distance_each: Vec<u8> = field(source, keys::GOAL_DISTANCE_EACH)?;
        let has_reach: bool = field(source, keys::HAS_REACH)?;
        let reach = if has_reach {
            let reach_piece: i32 = field(source, keys::REACH_PIECE_TYPE)?;
            let reach_squares_raw: Vec<u8> = field(source, keys::REACH_SQUARES)?;
            let reach_result: i8 = field(source, keys::REACH_RESULT)?;
            let reach_safe: bool = field(source, keys::REACH_SAFE)?;
            let squares: [Vec<bool>; 2] = [
                reach_squares_raw[..n_squares].iter().map(|v| *v != 0).collect(),
                reach_squares_raw[n_squares..].iter().map(|v| *v != 0).collect(),
            ];
            let goals = [0, 1].map(|player| {
                (0..n_squares)
                    .filter(|square| squares[player][*square])
                    .collect::<Vec<usize>>()
            });
            let per_goal = (goals[0].len() + goals[1].len()) * n_types * n_squares;
            if !goal_distance_each.is_empty() && goal_distance_each.len() != per_goal {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "goal_distance_each is not one table per goal square",
                ));
            }
            Some(ReachRule {
                piece_type: reach_piece,
                squares,
                goals,
                result: reach_result,
                safe: reach_safe,
            })
        } else {
            None
        };

        // Extinction
        let has_turn: bool = field(source, keys::HAS_TURN)?;
        let turn = if has_turn {
            Some(TurnRule {
                moves: field(source, keys::TURN_MOVES)?,
                ramp: field(source, keys::TURN_RAMP)?,
            })
        } else {
            None
        };

        let has_territory: bool = field(source, keys::HAS_TERRITORY)?;
        let territory = if has_territory {
            Some(TerritoryRule {
                repaints: field(source, keys::TERRITORY_REPAINTS)?,
                win_at: field(source, keys::TERRITORY_WIN_AT)?,
            })
        } else {
            None
        };

        let piece_values: Vec<f64> = field(source, keys::PIECE_VALUES)?;
        let has_scoring: bool = field(source, keys::HAS_SCORING)?;
        let scoring = if has_scoring {
            let count: u8 = field(source, keys::SCORING_COUNT)?;
            let raw: Vec<u8> = field(source, keys::SCORING_SQUARES)?;
            let squares = if raw.len() >= 2 * n_squares {
                Some([
                    raw[..n_squares].iter().map(|v| *v != 0).collect(),
                    raw[n_squares..2 * n_squares].iter().map(|v| *v != 0).collect(),
                ])
            } else {
                None
            };
            Some(ScoringRule { count, squares })
        } else {
            None
        };

        let has_extinction: bool = field(source, keys::HAS_EXTINCTION)?;
        let extinction = if has_extinction {
            let ext_types: Vec<usize> = field(source, keys::EXTINCTION_TYPES)?;
            let ext_any: bool = field(source, keys::EXTINCTION_ANY_PIECE)?;
            let ext_min: usize = field(source, keys::EXTINCTION_MIN_COUNT)?;
            let ext_result: i8 = field(source, keys::EXTINCTION_RESULT)?;
            Some(ExtinctionRule {
                counted: type_flags(&ext_types),
                piece_types: ext_types,
                any_piece: ext_any,
                min_count: ext_min,
                result: ext_result,
            })
        } else {
            None
        };

        let n_frames: usize = field(source, keys::N_FRAMES)?;
        let n_facings = n_frames / 2;
        let side_stride = n_facings * n_types;
        let facing_axes: Vec<usize> = field(source, keys::FACING_AXES)?;
        let facing_initial: Vec<usize> = field(source, keys::FACING_INITIAL)?;

        // --- Wild pieces (schema 9) ---
        // Three owners share the one byte once anything is wild, which is what the bound
        // below has to know about; `_check_facing_coherence` refuses the same thing on
        // the Python side, with the same arithmetic and a better message.
        let has_npcs: bool = field(source, keys::HAS_NPCS)?;
        let n_owners = if has_npcs { 3 } else { 2 };
        let npc_next: Vec<i16> = field::<Vec<i32>>(source, keys::NPC_NEXT)?
            .into_iter()
            .map(|square| square as i16)
            .collect();
        let npc_bounty: Vec<f64> = field(source, keys::NPC_BOUNTY)?;
        if has_npcs && (npc_next.len() < n_squares || npc_bounty.len() < n_squares) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "wild-piece tables hold {} and {} squares, not the {n_squares} this board                  has",
                npc_next.len(), npc_bounty.len(),
            )));
        }
        // Schema 9's second key. Empty when the flag is off, which is what makes
        // `arriving_value` `entering_value` for every ruleset before this one.
        let entry_promotion: Vec<i16> = field::<Vec<i32>>(source, keys::ENTRY_PROMOTION)?
            .into_iter()
            .map(|becomes| becomes as i16)
            .collect();
        if !entry_promotion.is_empty() && entry_promotion.len() != 2 * n_types * n_squares {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "the entry-promotion table holds {} slots, not the {} that 2 players x                  {n_types} types x {n_squares} squares needs",
                entry_promotion.len(), 2 * n_types * n_squares,
            )));
        }

        let npc_squares: Vec<u16> = (0..npc_next.len())
            .filter(|square| npc_next[*square] >= 0)
            .map(|square| square as u16)
            .collect();

        if n_owners * side_stride >= 256 {
            let who = if n_owners == 2 {
                "two players"
            } else {
                "two players and a wild third owner"
            };
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{n_types} piece types at {n_facings} facings for {who} need {} board                  values, and a square is one byte",
                n_owners * side_stride + 1
            )));
        }
        // The encoding, stated once. Everything downstream reads these three arrays
        // rather than dividing by `n_types`, which is what stops the board byte's layout
        // from being restated -- and possibly restated differently -- in the dozen hot
        // paths that need to know whose piece a square holds.
        let mut value_owner = [0u8; 256];
        let mut value_frame = [0u8; 256];
        let mut value_type = [0u8; 256];
        // Three owners once anything is wild. `value_frame` of a wild byte comes out as
        // `2 * n_facings`, one past the last real frame -- see `frame_of_value`.
        for value in 1..(n_owners * side_stride + 1) {
            value_owner[value] = ((value - 1) / side_stride) as u8;
            value_frame[value] = ((value - 1) / n_types) as u8;
            value_type[value] = ((value - 1) % n_types) as u8;
        }

        let royal_types: Vec<usize> = field(source, keys::ROYAL_TYPES)?;
        let mut all_squares = SquareSet::EMPTY;
        for square in 0..n_squares {
            all_squares.set(square);
        }

        // Which players start with a royal (Q133). One pass over the compiled setup,
        // reading the same three tables `scan_occupied` walks.
        let royal_flags = type_flags(&royal_types);
        let mut royal_at_start = [false; 2];
        for value in initial_board[..n_squares].iter().copied().filter(|v| *v != 0) {
            // Owner 2 is `WILD`, which has no royalty to lose: a wild royal is refused by
            // the validator, because check is about a *player's* royal.
            let owner = value_owner[value as usize] as usize;
            if owner < 2 && royal_flags[value_type[value as usize] as usize] {
                royal_at_start[owner] = true;
            }
        }

        let mut tables = Self {
            width,
            height,
            n_squares,
            n_types,
            n_owners,
            n_frames,
            n_facings,
            side_stride,
            has_facing: n_facings > 1,
            facing_axes,
            facing_initial,
            value_owner,
            value_frame,
            value_type,
            royal_flags,
            royal_at_start,
            royal_types,
            all_squares,
            leaps: EntryTable::build(
                field(source, keys::LEAP_INDEX)?,
                field(source, keys::LEAP_REFS)?,
                field(source, keys::LEAP_SQUARES)?,
                field(source, keys::LEAP_MODES)?,
            ),
            slides: EntryTable::build(
                field(source, keys::SLIDE_INDEX)?,
                field(source, keys::SLIDE_REFS)?,
                field(source, keys::SLIDE_SQUARES)?,
                field(source, keys::SLIDE_MODES)?,
            ),
            hops: EntryTable::build(
                field(source, keys::HOP_INDEX)?,
                field(source, keys::HOP_REFS)?,
                field(source, keys::HOP_SQUARES)?,
                field(source, keys::HOP_MODES)?,
            ),
            sweeps: EntryTable::build(
                field(source, keys::SWEEP_INDEX)?,
                field(source, keys::SWEEP_REFS)?,
                field(source, keys::SWEEP_SQUARES)?,
                field(source, keys::SWEEP_MODES)?,
            ),
            has_sweep: field(source, keys::HAS_SWEEP)?,
            attack_index: field(source, keys::ATTACK_INDEX)?,
            attack_sources,
            attack_paths: field(source, keys::ATTACK_PATHS)?,
            attack_path_modes: field(source, keys::ATTACK_PATH_MODES)?,
            dedup: field::<Vec<u8>>(source, keys::DEDUP)?
                .into_iter()
                .map(|flag| flag != 0)
                .collect(),
            has_hop_attacks: field(source, keys::HAS_HOP_ATTACKS)?,
            hop_lines: Vec::new(),
            initial_board,
            initial_unmoved: SquareSet::from_bytes(
                &field::<Vec<u8>>(source, keys::INITIAL_UNMOVED)?,
            ),
            win,
            stalemate,
            checkmate_result,
            max_plies: field(source, keys::MAX_PLIES)?,
            repetition_draw: field(source, keys::REPETITION_DRAW)?,
            no_progress_draw: field(source, keys::NO_PROGRESS_DRAW)?,
            one_way: type_flags(&field::<Vec<usize>>(source, keys::ONE_WAY_TYPES)?),
            ep_type: field(source, keys::EP_TYPE)?,
            ep_passes,
            facing_lines,
            facing_paths,
            promo_squares,
            promo_targets,
            promo_optional,
            promo_flags: type_flags(&promo_piece_types),
            promo_leaving_flags: type_flags(&promo_on_leaving),
            promo_piece_types,
            castling,
            castle_files,
            multi_royal,
            demotions,
            explosion,
            strike,
            drops,
            reach,
            goal_distance,
            goal_diameter,
            goal_distance_each,
            extinction,
            scoring,
            territory,
            turn,
            has_npcs,
            npc_next,
            npc_bounty,
            npc_squares,
            npc_bounty_total: 0.0,
            entry_promotion,
            piece_values,
            must_capture,
            forbid_check,
            forbid_immobility,
            royal_facing_forbidden,
            check_limit,
            capture_allowed,
            has_capture_matrix,
            bare_royal,
            immobile_squares,
            zobrist_pieces: field(source, keys::ZOBRIST_PIECES)?,
            zobrist_side: field(source, keys::ZOBRIST_SIDE)?,
            zobrist_ep: field(source, keys::ZOBRIST_EP)?,
            zobrist_colour: field(source, keys::ZOBRIST_COLOUR)?,
            zobrist_turn: field(source, keys::ZOBRIST_TURN)?,
        };

        // The two sides have to agree about how many frames there are, and the way they
        // could disagree is silent: a short table read at a high slot is whatever bytes
        // follow it. `_flatten_entries` writes one index per (frame, type, square) plus
        // a terminator, so the length is the shape, and checking it here is the only
        // place either side can be told it was wrong.
        let expected = tables.n_frames * tables.n_types * tables.n_squares + 1;
        for (name, table) in [
            ("leaps", &tables.leaps),
            ("slides", &tables.slides),
            ("hops", &tables.hops),
        ] {
            if table.slots() != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "compiled {name} table holds {} slots, not the {expected} that                      {} frames x {} types x {} squares needs",
                    table.slots(), tables.n_frames, tables.n_types, tables.n_squares,
                )));
            }
        }
        // Schema 13: the sweep table and the path modes are empty unless something sweeps.
        let sweep_slots = if tables.has_sweep { expected } else { 0 };
        let path_modes = if tables.has_sweep { tables.attack_paths.len() } else { 0 };
        if tables.sweeps.slots() != sweep_slots || tables.attack_path_modes.len() != path_modes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compiled sweep tables hold {} slots and {} path modes, not the {sweep_slots} \
                 and {path_modes} this ruleset needs",
                tables.sweeps.slots(), tables.attack_path_modes.len(),
            )));
        }
        if tables.dedup.len() != expected - 1 || tables.immobile_squares.len() != expected - 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "compiled dedup/immobile tables hold {} and {} entries, not the {} that                  {} frames x {} types x {} squares needs",
                tables.dedup.len(), tables.immobile_squares.len(), expected - 1,
                tables.n_frames, tables.n_types, tables.n_squares,
            )));
        }

        // The same shape check, for the table a third owner widened. The compiler adds a
        // block per owner and the core indexes it by owner, so a two-block table read at
        // `WILD` would run off the end of `attack_index` -- loudly here rather than as
        // whatever bytes follow it.
        let owners_expected = tables.n_owners * tables.n_squares + 1;
        if tables.attack_index.len() != owners_expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "the attack table holds {} slots, not the {owners_expected} that                  {} owners x {} squares needs",
                tables.attack_index.len(), tables.n_owners, tables.n_squares,
            )));
        }

        // What the bounties on the board come to at ply zero. Read off the opening
        // position rather than off `npc_bounty`, because a bounty is stated per *route*
        // and a route is several squares.
        if tables.has_npcs {
            tables.npc_bounty_total = (0..tables.n_squares)
                .filter(|square| {
                    let value = tables.initial_board[*square];
                    value != 0 && tables.owner_of(value) == WILD
                })
                .map(|square| tables.npc_bounty[square])
                .sum();
        }

        tables.hop_lines = tables.build_hop_lines();
        Ok(tables)
    }
}

fn unknown(what: &str, value: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("unknown {what} {value:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract move generation stands on: a set yields its squares lowest first,
    /// across word boundaries, which is the order a scan over every square found them.
    #[test]
    fn a_square_set_iterates_in_ascending_order() {
        let squares = [255usize, 0, 64, 63, 200, 1, 128, 65, 127];
        let mut set = SquareSet::EMPTY;
        for square in squares {
            set.set(square);
        }
        let mut expected = squares.to_vec();
        expected.sort_unstable();
        assert_eq!(set.iter().collect::<Vec<_>>(), expected);
        assert_eq!(set.len(), squares.len());
    }

    /// Schema 13's `sweep` byte sits in the padding. A ninth field in the packed chunk
    /// would have rewritten `attack_srcs` for every ruleset instead.
    #[test]
    fn an_attack_source_is_twenty_bytes() {
        assert_eq!(std::mem::size_of::<AttackSource>(), 20);
    }

    /// The macro puts every declared key in `ALL`; what it cannot do is stop the same key
    /// being declared twice, which would make the count the refusal quotes a lie.
    #[test]
    fn the_reported_key_vocabulary_names_each_key_once() {
        let mut sorted = keys::ALL.to_vec();
        let declared = sorted.len();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), declared);
    }

    #[test]
    fn set_algebra_matches_the_board_it_stands_for() {
        let mut mine = SquareSet::EMPTY;
        let mut theirs = SquareSet::EMPTY;
        let mut all = SquareSet::EMPTY;
        for square in 0..90 {
            all.set(square);
        }
        for square in [3usize, 17, 64, 89] {
            mine.set(square);
        }
        for square in [5usize, 17, 70] {
            theirs.set(square);
        }
        assert_eq!(mine.union(&theirs).iter().collect::<Vec<_>>(), vec![3, 5, 17, 64, 70, 89]);
        assert_eq!(mine.intersection(&theirs).iter().collect::<Vec<_>>(), vec![17]);
        let empty = all.difference(&mine.union(&theirs));
        assert_eq!(empty.len(), 90 - 6);
        assert!(empty.iter().all(|square| square < 90 && !mine.get(square) && !theirs.get(square)));
        assert!(SquareSet::EMPTY.iter().next().is_none());
    }
}

/// Every table key `from_py` reads, and the only way to name one.
///
/// The vocabulary has to be *one* list, or it is worth nothing. `TableKey`'s field is
/// private to this module and `declare_keys!` is the only thing that ever builds one, so
/// `field` below cannot be handed a key that is not declared here -- a new reader that
/// tries a bare string literal does not compile. Registering a key and reading it are
/// therefore the same act, and `ALL` cannot fall behind the readers.
///
/// `chess_variant_sandbox._core.TABLE_KEYS` is `ALL`, and `engine/bridge.py` compares it against the
/// keys `build_tables` writes. That comparison is the half of the stale-core question
/// `field` cannot answer: a key the tables are *missing* raises here, but one the core
/// has never heard of would otherwise cross unread, and the game played would be the one
/// the core knew rather than the one the spec asked for.
pub mod keys {
    /// A key the core reads. Opaque on purpose -- the field is private, so nothing
    /// outside this module can spell one out of a literal.
    #[derive(Clone, Copy)]
    pub struct TableKey(&'static str);

    impl TableKey {
        pub const fn name(self) -> &'static str {
            self.0
        }
    }

    macro_rules! declare_keys {
        ($($ident:ident => $key:literal),* $(,)?) => {
            $(pub const $ident: TableKey = TableKey($key);)*

            /// The vocabulary as plain strings, in the order declared, for the Python
            /// side to compare against what it sends.
            pub const ALL: &[&str] = &[$($key),*];
        };
    }

    declare_keys! {
        ATTACK_INDEX => "attack_index",
        ATTACK_PATH_MODES => "attack_path_modes",
        ATTACK_PATHS => "attack_paths",
        ATTACK_SRCS => "attack_srcs",
        ATTACK_SWEEP => "attack_sweep",
        BARE_ROYAL => "bare_royal",
        CAPTURE_ALLOWED => "capture_allowed",
        CASTLE_KING => "castle_king",
        CASTLE_KING_FILES => "castle_king_files",
        CASTLE_ROOK => "castle_rook",
        CASTLE_ROOK_FILES => "castle_rook_files",
        CASTLE_STEPS => "castle_steps",
        CHECK_LIMIT => "check_limit",
        CHECKMATE_RESULT => "checkmate_result",
        CONTACT_MOVER_TO => "contact_mover_to",
        CONTACT_ONTO_TO => "contact_onto_to",
        CONTACT_OWN_MOVER_TO => "contact_own_mover_to",
        CONTACT_OWN_ONTO_TO => "contact_own_onto_to",
        DEDUP => "dedup",
        DEMOTIONS => "demotions",
        DROPS_FROM_CAPTURES => "drops_from_captures",
        DROPS_NO_DOUBLED => "drops_no_doubled",
        DROPS_NO_MATE => "drops_no_mate",
        DROPS_REGIONS => "drops_regions",
        DROPS_STOCK => "drops_stock",
        DROPS_UNLIMITED => "drops_unlimited",
        ENTRY_PROMOTION => "entry_promotion",
        EP_PASS_FLAT => "ep_pass_flat",
        EP_PASS_INDEX => "ep_pass_index",
        EP_TYPE => "ep_type",
        EXPLOSION_BLAST => "explosion_blast",
        EXPLOSION_BLAST_INDEX => "explosion_blast_index",
        EXPLOSION_BY => "explosion_by",
        EXPLOSION_IMMUNE => "explosion_immune",
        EXPLOSION_OF => "explosion_of",
        EXTINCTION_ANY_PIECE => "extinction_any_piece",
        EXTINCTION_MIN_COUNT => "extinction_min_count",
        EXTINCTION_RESULT => "extinction_result",
        EXTINCTION_TYPES => "extinction_types",
        FACING_AXES => "facing_axes",
        FACING_ENTRIES => "facing_entries",
        FACING_INDEX => "facing_index",
        FACING_INITIAL => "facing_initial",
        FACING_PATHS => "facing_paths",
        FORBID_CHECK => "forbid_check",
        FORBID_IMMOBILITY => "forbid_immobility",
        GOAL_DIAMETER => "goal_diameter",
        GOAL_DISTANCE => "goal_distance",
        GOAL_DISTANCE_EACH => "goal_distance_each",
        HAS_BARE_ROYAL => "has_bare_royal",
        HAS_CAPTURE_MATRIX => "has_capture_matrix",
        HAS_DROPS => "has_drops",
        HAS_EXPLOSION => "has_explosion",
        HAS_EXTINCTION => "has_extinction",
        HAS_HOP_ATTACKS => "has_hop_attacks",
        HAS_NPCS => "has_npcs",
        HAS_REACH => "has_reach",
        HAS_SCORING => "has_scoring",
        HAS_STRIKE => "has_strike",
        HAS_SWEEP => "has_sweep",
        HAS_TERRITORY => "has_territory",
        HAS_TURN => "has_turn",
        HEIGHT => "height",
        HOP_INDEX => "hop_index",
        HOP_MODES => "hop_modes",
        HOP_REFS => "hop_refs",
        HOP_SQUARES => "hop_squares",
        IMMOBILE_SQUARES => "immobile_squares",
        INITIAL_BOARD => "initial_board",
        INITIAL_UNMOVED => "initial_unmoved",
        LEAP_INDEX => "leap_index",
        LEAP_MODES => "leap_modes",
        LEAP_REFS => "leap_refs",
        LEAP_SQUARES => "leap_squares",
        MAX_PLIES => "max_plies",
        MULTI_ROYAL => "multi_royal",
        MUST_CAPTURE => "must_capture",
        N_FRAMES => "n_frames",
        N_SQUARES => "n_squares",
        N_TYPES => "n_types",
        NO_PROGRESS_DRAW => "no_progress_draw",
        NPC_BOUNTY => "npc_bounty",
        NPC_NEXT => "npc_next",
        ONE_WAY_TYPES => "one_way_types",
        PIECE_VALUES => "piece_values",
        PROMO_ON_LEAVING => "promo_on_leaving",
        PROMO_OPTIONAL => "promo_optional",
        PROMO_PIECE_TYPES => "promo_piece_types",
        PROMO_SQUARES => "promo_squares",
        PROMO_TARGETS_FLAT => "promo_targets_flat",
        PROMO_TARGETS_INDEX => "promo_targets_index",
        REACH_PIECE_TYPE => "reach_piece_type",
        REACH_RESULT => "reach_result",
        REACH_SAFE => "reach_safe",
        REACH_SQUARES => "reach_squares",
        REPETITION_DRAW => "repetition_draw",
        ROYAL_FACING_FORBIDDEN => "royal_facing_forbidden",
        ROYAL_TYPES => "royal_types",
        SCORING_COUNT => "scoring_count",
        SCORING_SQUARES => "scoring_squares",
        SLIDE_INDEX => "slide_index",
        SLIDE_MODES => "slide_modes",
        SLIDE_REFS => "slide_refs",
        SLIDE_SQUARES => "slide_squares",
        STALEMATE => "stalemate",
        STRIKE_LANDING => "strike_landing",
        STRIKE_LANDING_INDEX => "strike_landing_index",
        STRIKE_POSITION => "strike_position",
        STRIKE_TO => "strike_to",
        SWEEP_INDEX => "sweep_index",
        SWEEP_MODES => "sweep_modes",
        SWEEP_REFS => "sweep_refs",
        SWEEP_SQUARES => "sweep_squares",
        TERRITORY_REPAINTS => "territory_repaints",
        TERRITORY_WIN_AT => "territory_win_at",
        TURN_MOVES => "turn_moves",
        TURN_RAMP => "turn_ramp",
        WIDTH => "width",
        WIN => "win",
        ZOBRIST_COLOUR => "zobrist_colour",
        ZOBRIST_EP => "zobrist_ep",
        ZOBRIST_PIECES => "zobrist_pieces",
        ZOBRIST_SIDE => "zobrist_side",
        ZOBRIST_TURN => "zobrist_turn",
    }
}

/// The one place a compiled table is read out of the dict Python sent. Taking a
/// `TableKey` rather than a `&str` is what keeps `keys::ALL` honest.
fn field<'py, T>(source: &Bound<'py, PyDict>, key: keys::TableKey) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py>,
{
    let name = key.name();
    let item = source
        .get_item(name)?
        .ok_or_else(|| PyKeyError::new_err(format!("compiled tables are missing {name:?}")))?;
    item.extract().map_err(Into::into)
}
