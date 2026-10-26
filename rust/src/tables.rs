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
/// Schema 20 (Q807): no straight-back bar (`Game::portal_bar`).
pub const NO_BAR: [i16; 2] = [NO_SQUARE, NO_SQUARE];
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
/// Schema 29 (Q983a, `docs/plan-rest.md` §3.1): the longest rest a type may declare, and so
/// how many of one player's pieces can rest at once (one enters a turn, none stays longer).
/// `spec/schema.py::REST_MAX` is the validator's copy.
pub const REST_MAX: usize = 4;

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
/// Schema 19: a transit. The piece moves onto a portal end and comes out one king's step
/// beyond the other end of its pair: the target is that exit square, where the move
/// ends, bits 25 to 31 hold the internal type of the end it went in by, and the
/// promotion field holds the promoted type plus one (0 for none), because one kind field
/// cannot say both "portal" and "promotion".
pub const MOVE_PORTAL: u32 = 7;
/// Schema 22: a fold's own move. Origin and target are the fold's squares, bits 25 to 31
/// the end (its index among the portal ends), and nothing on the board byte changes: a
/// fold stands on no square of it (`Game::fold_at`). An ordinary move that went through
/// a fold -- landed on it and came out on the end it leads to, its target (schema 24) --
/// carries the end it went in by plus one in the same bits.
pub const MOVE_FOLD: u32 = 8;
/// Generation 42 (N4b, `docs/plan-npc.md` §4.6): the duck's move, the second half of every
/// turn under `rules.duck`. The promotion field says how: `DUCK_STEP` the standing duck
/// moves from origin to target, `DUCK_PLACE` it is put on the target from off the board
/// (origin equals target, as a drop's), `DUCK_PASS` it has nowhere to go (origin equals
/// target, the duck's square or 0; never read). Takes nothing, on any reading.
pub const MOVE_DUCK: u32 = 9;
pub const DUCK_STEP: u32 = 0;
pub const DUCK_PLACE: u32 = 1;
pub const DUCK_PASS: u32 = 2;
/// The owner a portal end's board byte belongs to: the third block, nobody's, which a wild
/// piece uses too. The validator refuses the two rules together, so a game's third block
/// holds one or the other (`compile.PortalRule`).
pub const PORTAL_OWNER: usize = WILD;

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

    /// How many landings the table holds, over every entry: the length of `squares`.
    pub fn n_landings(&self) -> usize {
        self.squares.len()
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
    /// Schema 19: the directions this source captures in, one bit per
    /// `compile.PORTAL_DIRECTIONS` entry, for an attack through a portal. Zero in every
    /// game without portals. The struct's last padding byte.
    pub dirs: u8,
}

/// Schema 19: groups of portals, as the compiler resolved them (`compile.PortalRule`);
/// schema 21 made a pair one case of a map (`docs/archive/plan-portals-2.md` §2.2).
///
/// Boxed behind one `Option` on `Tables`, so a game without portals pays one pointer and
/// one null test per piece in move generation, and the struct the movegen reads at every
/// ply grows by eight bytes rather than by these arrays.
pub struct PortalTables {
    /// The first end's internal type; end `k` is type `first_type + k`.
    pub first_type: usize,
    /// How many ends, across every group: at least two, at most eight.
    pub n_ends: usize,
    /// `dest[type]` -> the type of the end a piece that goes in here comes out of, or -1
    /// for a way out only and for a type that is not a portal.
    pub dest: [i8; MAX_PIECE_TYPES],
    /// `admits[type][player]` -> a bit per declared piece type that may go in by this end:
    /// the end's `enter` and `carries`, compiled. Zero where that player may not go in.
    pub admits: [[u32; 2]; MAX_PIECE_TYPES],
    /// `movers[type]` -> a bit per player who may move this end.
    pub movers: [u8; MAX_PIECE_TYPES],
    /// `[(player * n_squares + square) * 8 + direction]` -> the square one king's step away
    /// under that player's wrap permission, or -1.
    pub exits: Vec<i16>,
    /// The momentum masks, parallel to each movement table's squares: bits 0-7 the
    /// directions a move-mode landing there was reached in, bits 8-15 a capture-mode one.
    pub leap_dirs: Vec<u16>,
    pub slide_dirs: Vec<u16>,
    pub hop_dirs: Vec<u16>,
    /// Schema 20 (Q807): `no_back[type]` -> once this end has moved, it may not move
    /// straight back onto the square it left on the very next move (`no_straight_back`).
    pub no_back: [bool; MAX_PIECE_TYPES],
    /// Whether any end bars going straight back: `Game::portal_bar` and its log are kept
    /// only then, so a portal game without the rule pays one flag test in `make`.
    pub has_no_back: bool,
    /// The bar's Zobrist keys: `[left]` for the square the end left, `[n_squares +
    /// standing]` for the one it stands on. Empty without the rule.
    pub zobrist_bar: Vec<u64>,
    /// Schema 22: how many ends are momentum ends, which stand on the board byte and are
    /// what `portal_squares` looks for. `n_ends` when no group is a fold.
    pub n_momentum: usize,
    /// Schema 22: the fold tables, or `None` where no group is a fold.
    pub folds: Option<FoldTables>,
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
    /// Schema 17 (Q540), `Home` only: `rebirth[(owner * n_types + type) * landing_stride +
    /// square]` is the home square a piece of `owner` and `type` taken on `square` is
    /// reborn on, or -1 where it has none. Empty under every other position.
    pub rebirth: Vec<i32>,
    /// Schema 17 (Q540): `mover_flips[mover * n_types + victim]` -- the attacker changes
    /// sides (Andernach). Empty when no type does, so every earlier rule reads nothing.
    pub mover_flips: Vec<bool>,
}

/// Schema 17 (Q540): a `to` entry that removes the victim and leaves the attacker on its
/// origin (rifle). Below -1, so "survives" (`>= 0`) still reads it as removal.
pub const REMOVED_IN_PLACE: i32 = -2;

impl StrikeRule {
    /// Where a piece of `owner` and `piece_type` taken on `square` is reborn, or -1.
    #[inline]
    pub fn rebirth_square(&self, owner: usize, piece_type: usize, square: usize) -> i32 {
        self.rebirth[(owner * self.n_types + piece_type) * self.landing_stride + square]
    }

    /// Whether a capture of `victim` by `mover` sends the attacker over to the other side.
    #[inline]
    pub fn flips(&self, cell: usize) -> bool {
        !self.mover_flips.is_empty() && self.mover_flips[cell]
    }

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
    /// Schema 17 (Q540): the attacker advances and the survivor is reborn on its home
    /// square, read off `rebirth`; an occupied home square removes it (circe).
    Home,
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
    /// Schema 18 (Q784): the placement phase. While a side holds anything in hand, its
    /// only legal moves are drops (`generate_pseudo_legal`'s gate). False for every
    /// ruleset before it, which is the boundary's inert default.
    pub placement: bool,
    /// Whether the stock holds a royal type, so a side's royal may be in its hand rather
    /// than off the board for good. Derived here from `stock` and `royal_types`, not sent:
    /// the readings of a missing royal (`Game::royal_gone`) ask it before they ask the
    /// hand, so a ruleset whose stock holds no royal never reads a hand for it.
    pub royal_stocked: bool,
    /// Schema 18 (Q785): drop either face. `faces[slot]` is the other face a piece held
    /// in that hand slot may be dropped as (the piece whose `promoted_from` it is), or -1.
    /// Empty for every ruleset before it, which is the boundary's inert default, so
    /// `generate_drops` asks `faces.is_empty()` once and walks the hand as it always did.
    pub faces: Vec<i32>,
    /// The inverse, derived here rather than sent: `slot_of[face]` is the hand slot a
    /// drop of `face` spends. The identity where `faces` is empty. The validator refuses
    /// every spelling under which a hand could hold a paired face itself (a stocked face,
    /// a chain of `promoted_from`), so one face names one slot.
    pub slot_of: Vec<usize>,
}

impl DropRule {
    /// Whether this type's hand slot is a supply rather than a count.
    #[inline]
    pub fn is_unlimited(&self, piece_type: usize) -> bool {
        !self.unlimited.is_empty() && self.unlimited[piece_type]
    }

    /// The hand slot a drop placing `face` spends: `face` itself, except under schema
    /// 18's `either_face`, where a paired face is spent from its base's slot.
    #[inline]
    pub fn slot_of(&self, face: usize) -> usize {
        if self.slot_of.is_empty() {
            face
        } else {
            self.slot_of[face]
        }
    }
}

/// Reach rule: win by arrival, or (schema 18) by leaving.
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
    /// Schema 18 (Q786, Jeson Mor): `reach.on: "leave"`. The game is decided by the move
    /// that takes the named piece off one of its side's goal squares, which `make` records
    /// in `Game::goal_left_by`; a piece standing on a goal decides nothing. False for
    /// every ruleset before it, which is the boundary's inert default.
    pub leaves: bool,
}

/// Generation 42 (N4b, `docs/plan-npc.md` §4.6): the duck. One piece nobody owns, in the
/// third owner's block at facing 0, which the mover moves every turn as the turn's second
/// half (the compiled turn is two moves; `remaining` 1 is the duck half). Its square is read
/// off the board (`Game::duck_square`), never kept.
pub struct DuckRule {
    /// The duck's board byte: `WILD * side_stride + piece_type + 1`.
    pub value: u8,
    /// Its declared type, whose frame-0 leap and slide entries a `by_moves` duck walks.
    pub piece_type: usize,
    /// `goes: "by_moves"`; false is `"anywhere"`.
    pub by_moves: bool,
    /// The squares a duck may stand on: every square of the board but a wall or an absent
    /// cell, which the compiler knows and the core does not.
    pub squares: SquareSet,
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
    /// Schema 18 (Q783): each counted type on its own, so a side has run out when any
    /// one of them is below `min_count`. False is the summed count, which is every
    /// ruleset before it. Never true beside `any_piece` (the validator refuses it).
    pub per_type: bool,
    /// Schema 27 (Q936): the types counted against each side, apart, under `per_type`.
    /// `"per_type": "own"` resolves them in the compiler, each side's being the listed
    /// types it starts with on the board (`compile._side_types`); every other spelling
    /// crosses the boundary empty and both entries are `piece_types`, so no earlier game
    /// reads a different list. The core walks the list it is handed and never asks who
    /// started with what.
    pub side_types: [Vec<usize>; 2],
    /// Whether `side_types` came across the boundary (`"per_type": "own"`), so the two
    /// lists may differ. Read by the evaluator's features, never by the ending.
    pub own: bool,
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

/// Simultaneous moves (schema 26, Q918a; `docs/reference/rounds.md` §5.3), as the
/// compiler resolved them (`compile.SimRule`). The core reads every number here and decides
/// none of them.
pub struct SimTables {
    /// 0 both eliminated (the default), 1 the initiative, 2 impossible (read as 0).
    pub tie: u8,
    /// Who holds the initiative at the start, under the initiative.
    pub initiative_start: i8,
    /// How many void rounds of one pair in a row bar it (`compile.VOID_RUN`, 2).
    pub void_run: u8,
    /// Whether a collision's loser goes through the strike rule (`rules.strike`, never
    /// `rules.contacts`).
    pub collision_strikes: bool,
    /// `[a * n_types + b]`: a type `a` eliminates a type `b` when they collide.
    pub collision_allowed: Vec<bool>,
    /// XORed in while player 1 holds the initiative (0 unless the tie is the initiative).
    pub zobrist_initiative: u64,
    /// `[square]` for a standoff entry's player-0 square, `[n_squares + square]` for its
    /// player-1 square.
    pub zobrist_squares: Vec<u64>,
    /// `[state - 1]`: 1 a run of one void round, 2 barred.
    pub zobrist_states: Vec<u64>,
    /// Schema 27 (R2): two moves each ending on the other's origin meet as a collision.
    pub swap_collides: bool,
    /// Schema 27 (X1): an ending that fires for both players in one round goes to the
    /// player ahead under `rules.scoring`.
    pub both_score: bool,
    /// Schema 28 (Q943a, `docs/archive/plan-speed.md` §4.2): `[type]`, each type's speed. Empty
    /// unless a piece declares one, so `make_round` never orders a round in any other game.
    pub speed: Vec<u8>,
}

impl SimTables {
    /// Whether a type `a` eliminates a type `b` in a collision.
    #[inline]
    pub fn beats(&self, n_types: usize, a: usize, b: usize) -> bool {
        self.collision_allowed[a * n_types + b]
    }

    /// The key of one standoff entry: a fixed mix (splitmix64's finaliser) of its two
    /// squares' draws and its state's, so the bars `{(a, b), (c, d)}` and `{(a, d), (c, b)}`
    /// hash apart while the tables stay the size of the board (§5.3).
    #[inline]
    pub fn standoff_key(&self, n_squares: usize, first: usize, second: usize, state: u8) -> u64 {
        let mut z = self.zobrist_squares[first]
            ^ self.zobrist_squares[n_squares + second]
            ^ self.zobrist_states[state as usize - 1];
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Schema 29 (Q983a, `docs/plan-rest.md` §3, §4): rest as a piece-type property, as the
/// compiler resolved it (`CompiledSpec.rest`). Boxed, as the rounds are, so a game without
/// it pays one pointer, and `None` is the one flag every rest branch in the core tests.
pub struct RestTables {
    /// `[type]`: how many of its player's turns a piece of that type rests after it makes
    /// its player's move, 0 for never. Read off the type that stands at the end of the move.
    pub turns: Vec<u8>,
    /// `[((left - 1) * 2 + player) * n_squares + square]`: XORed in for each resting piece.
    pub zobrist: Vec<u64>,
    /// The longest rest, and so how many of a player's `REST_MAX` slots can ever be live:
    /// a player enters at most one piece a turn and each rests at most this many of its
    /// turns, and an entry takes the first free slot. The core walks only these.
    pub slots: usize,
}

impl RestTables {
    /// The key of one resting piece: `player`'s, on `square`, with `left` turns to go.
    #[inline]
    pub fn key(&self, n_squares: usize, player: usize, square: usize, left: u8) -> u64 {
        self.zobrist[((left as usize - 1) * 2 + player) * n_squares + square]
    }
}

/// One route of wild pieces that come back (generation 36, N1, `docs/plan-npc.md` §4.1): an
/// `npcs` entry with `respawn`, or its mirrored half, each with a clock of its own.
pub struct RespawnHalf {
    /// The clock's length: a taken piece comes back this many phases after it was taken.
    pub every: u8,
    /// Where a piece comes back, first free first: the route's first square, then
    /// `also_at` in route order (`schema.npc_starts`).
    pub starts: Vec<u16>,
    /// Every square of the route, where the half's pieces stand: walkers never leave it, so
    /// the pieces standing are the wild pieces on these squares.
    pub route: Vec<u16>,
    /// The board byte of a piece of this half, read off the opening board.
    pub value: u8,
    /// Where this half's `every` keys begin in `RespawnTables::zobrist`.
    pub key_base: usize,
}

/// Generation 36 (N1): every respawning half, in `rules.npcs` order with a mirrored half after
/// its original (the order the phase's step 3 returns them in), and the clocks' keys. The
/// state is `Game::respawn`, one clock per half (`respawn.rs`).
pub struct RespawnTables {
    pub halves: Vec<RespawnHalf>,
    /// `[key_base + clock - 1]` for a running clock (1 to `every`); an idle clock (0) is keyed
    /// by nothing, so a game whose wild pieces all stand keys as it did before the rule.
    pub zobrist: Vec<u64>,
}

impl RespawnTables {
    /// The key of `half`'s clock at `clock`; 0 for idle.
    #[inline]
    pub fn key(&self, half: usize, clock: u8) -> u64 {
        if clock == 0 {
            0
        } else {
            self.zobrist[self.halves[half].key_base + clock as usize - 1]
        }
    }
}

/// Generation 38 (N5, `docs/plan-npc.md` §4.7): the routes walked as **lines**, first to last
/// and back, by an `npcs` entry with `turns_back`. `npc_next` holds a line's last square on
/// itself (it never wraps); `prev` is the way back. The state is `Game::heading_back`, the
/// squares of the pieces walking back (`turning.rs`).
pub struct LineTables {
    /// `prev[square]`: the square a piece walking its line back steps to; the square itself
    /// at a line's first square; `-1` off every line.
    pub prev: Vec<i16>,
    /// Every line square: where a turning-back piece can stand.
    pub squares: SquareSet,
    /// `[square]`: the key of a piece on a line walking it back; 0 off every line.
    pub zobrist: Vec<u64>,
}

impl LineTables {
    /// The square ahead of a piece on line square `square` walking back (`back`) or forward,
    /// or `None` at the line's end that way.
    #[inline]
    pub fn ahead(&self, next: &[i16], square: usize, back: bool) -> Option<usize> {
        let to = if back { self.prev[square] } else { next[square] };
        (to >= 0 && to as usize != square).then_some(to as usize)
    }
}

/// One capture a pouncing type could make from a square (generation 39, N3b): the target and
/// the path it needs, as an attack source's (`prefix_len` leading squares empty, the rest the
/// window holding exactly `screens` pieces, of anybody's).
#[derive(Clone, Copy)]
pub struct ReachEntry {
    pub target: u16,
    pub screens: u8,
    pub prefix_len: u16,
    pub path_len: u16,
    pub path_start: u32,
}

/// Generation 39 (N3b, `docs/plan-npc.md` §3.2, §4.3): wild pieces that pounce, as the
/// compiler resolved them (`compile.PounceRule`). `None` on `Tables` for every game in which
/// no `npcs` entry says `pounce`. The core sorts nothing geometric: the prey order is three
/// tables and the reach a forward list per type and square (`pounce.rs`).
pub struct PounceTables {
    /// `[type]`: 0 for a type that does not pounce, else `POUNCE_PREY` or `POUNCE_ATTACKERS`.
    pub kind: Vec<u8>,
    /// Whether any type pounces on its attackers (generation 40, N3c): only then does the
    /// phase's snapshot list the mover's frozen and disarmed pieces.
    pub attackers: bool,
    /// `[type]`: the type's place among the declared values, highest 0, equal values equal.
    pub value_rank: Vec<u8>,
    /// `[player * n_squares + square]`: the square's place in that player's reading order.
    pub reading: Vec<u16>,
    /// `[a * n_squares + b]`: king's distance (the lattice's own on a hexagon).
    pub distance: Vec<u8>,
    /// `[type * n_squares + origin]` .. `+ 1`: that type's captures from `origin` in `targets`.
    pub targets_index: Vec<u32>,
    pub targets: Vec<ReachEntry>,
    pub paths: Vec<u16>,
    /// `[square]`: the type whose route the square is on, -1 off every route (`NpcRule.piece`).
    /// A wild piece walks a route only where the route is its own type's (§3.3).
    pub piece: Vec<i8>,
    /// `[type]`: what taking a wild piece of the type pays, read off the routes' bounties at
    /// load (one per type, the validator says), 0.0 for a type no route carries.
    pub bounty: Vec<f64>,
    /// `[type]`: the most wild pieces of a splitting type standing at once (generation 41,
    /// N3d, `docs/plan-npc.md` §4.5), 0 for a type that does not split; empty where nothing
    /// splits.
    pub split: Vec<u8>,
    pub n_squares: usize,
}

/// `PounceTables::kind` of a type that takes the pieces it attacks (`"prey"`).
pub const POUNCE_PREY: u8 = 1;
/// `PounceTables::kind` of a type that takes the pieces attacking it (`"attackers"`,
/// generation 40, N3c).
pub const POUNCE_ATTACKERS: u8 = 2;

impl PounceTables {
    /// `ty`'s captures from `origin`.
    #[inline]
    pub fn targets_from(&self, ty: usize, origin: usize) -> &[ReachEntry] {
        let slot = ty * self.n_squares + origin;
        &self.targets[self.targets_index[slot] as usize..self.targets_index[slot + 1] as usize]
    }

    #[inline]
    pub fn path(&self, entry: &ReachEntry) -> &[u16] {
        let start = entry.path_start as usize;
        &self.paths[start..start + entry.path_len as usize]
    }

    /// Whether a wild piece of type `ty` pounces.
    #[inline]
    pub fn pounces(&self, ty: usize) -> bool {
        self.kind[ty] != 0
    }

    /// The split cap of type `ty` (generation 41, N3d), 0 for a type that does not split.
    #[inline]
    pub fn split_cap(&self, ty: usize) -> usize {
        self.split.get(ty).copied().unwrap_or(0) as usize
    }
}

/// The spy (schema 27, Q935a; `docs/plan-spy.md` §2.4), as the compiler resolved it
/// (`compile.SpyRule`). The core reads it and decides nothing: the phase is a function of
/// the board and the side to move (`Game::phase`).
pub struct SpyTables {
    /// `[player]`: a bit per type, set for each of that player's spy types (`1 << type`;
    /// `MAX_PIECE_TYPES` is 24, so a `u32` holds them).
    pub mask: [u32; 2],
    /// Whether the initiative is dealt to the announcer at the first announced round
    /// (`SimRule.deal_initiative`, E10): a spy game under `tie: "initiative"`.
    pub deal: bool,
}

/// Schema 30 (Q999a, `docs/archive/plan-passive.md` §3.2): one way a piece can be under an effect,
/// inverted out of the forward lines by the compiler (`compile.SightSource`) and filed under
/// the affected player, type and square. It binds when the source's piece stands on `origin`
/// and every square of its path is empty.
#[derive(Clone, Copy)]
pub struct SightSource {
    pub origin: u16,
    pub piece_value: u8,
    /// `SIGHT_FREEZE`, `SIGHT_DISARM`, `SIGHT_SHIELD`, or `SIGHT_COMMAND` alone or with
    /// `SIGHT_COMMAND_OWN` (one source of a command sight that says `captures_own`).
    pub effect: u8,
    pub path_start: u32,
    pub path_len: u32,
}

pub const SIGHT_FREEZE: u8 = 1;
pub const SIGHT_DISARM: u8 = 2;
pub const SIGHT_SHIELD: u8 = 4;
/// Q1007c (`docs/plan-schema-31.md` §3): the piece's opponent may also move it.
pub const SIGHT_COMMAND: u8 = 8;
/// Q1007c: a commanded move may also take the piece's own side's non-royal pieces
/// (`captures_own`); never set without `SIGHT_COMMAND` on the same source.
pub const SIGHT_COMMAND_OWN: u8 = 16;
/// Q1007c: bit 30 of a move word marks a commanded move -- the side to move moving one of
/// the opponent's pieces. Bits 25 to 31 are otherwise a strike's landing, a portal end or a
/// fold, each refused beside a sight, so in a command game this bit is the only one set
/// there. It also keeps a commanded move from opening an en passant window (`make` opens
/// one only where those bits are zero).
pub const MOVE_COMMANDED: u32 = 1 << 30;

/// Schema 30: lines of sight, as the compiler resolved them (`compile.SightRule`). Boxed,
/// as the rounds and the rest are, so a game without a sight pays one pointer, and `None` is
/// the one flag every sight branch in the core tests.
pub struct SightTables {
    /// `[(player * n_types + type) * n_squares + square]` .. `[+ 1]`: the sources that could
    /// bind that player's piece of that type on that square, in `sources`.
    index: Vec<u32>,
    sources: Vec<SightSource>,
    paths: Vec<u16>,
    /// Whether any sight shields: only then is a landing asked about its victim.
    pub has_shield: bool,
    /// Whether any sight can change a check (a freeze, a disarm, or a shield that can cover a
    /// royal other than its own source): the pin shortcut is off then.
    pub bears_on_check: bool,
    /// Whether any sight freezes or disarms: only then is a mover or an attacker asked.
    pub has_restraint: bool,
    /// Q1007c: whether any sight commands. Only then does the generator walk the opponent's
    /// pieces for commanded moves (`Game::push_commanded`).
    pub has_command: bool,
    /// `n_types * n_squares`, the stride of a player's block in `index`.
    block: usize,
    n_squares: usize,
}

impl SightTables {
    /// The sources that could put an effect on `player`'s piece of `piece_type` standing on
    /// `square`.
    #[inline(always)]
    pub fn sources_of(&self, player: usize, piece_type: usize, square: usize) -> &[SightSource] {
        let slot = player * self.block + piece_type * self.n_squares + square;
        let lo = self.index[slot] as usize;
        let hi = self.index[slot + 1] as usize;
        &self.sources[lo..hi]
    }

    #[inline(always)]
    pub fn path(&self, source: &SightSource) -> &[u16] {
        let lo = source.path_start as usize;
        &self.paths[lo..lo + source.path_len as usize]
    }
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
    /// Schema 25 (Q869): `facing_turns[type]` -> whether a move may turn that type.
    /// Empty where every facing piece turns, which is every ruleset before schema 25;
    /// read it through `turns_of`.
    pub facing_turns: Vec<bool>,
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
    /// Schema 18 (Q783), `promotion.only_lost`: the promoting types that offer a target
    /// only while the mover has fewer of it than it started with. All false for every
    /// ruleset before it, so `promotion_offered` is one load and a branch there.
    promo_only_lost_flags: [bool; MAX_PIECE_TYPES],
    /// `promo_opening[player][type]` -> how many of that type the player started with.
    /// Read only behind `promo_only_lost_flags`; zeros where nothing says `only_lost`.
    promo_opening: [[u32; MAX_PIECE_TYPES]; 2],

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
    /// `reach.leaves`, hoisted: the one flag `make` asks on every move, so a ruleset
    /// without the rule pays one load for it rather than an `Option` and a field.
    pub reach_leaves: bool,
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
    /// Whether any route carries more than one wild piece at ply zero (schema 23's
    /// `also_at`). Derived here from the opening board and `npc_next`, so nothing new
    /// crosses the boundary. Only then can one wild piece wait for another, and only then
    /// does the phase need `Game::npc_step_order`'s passes; every other game steps its
    /// wild pieces in one ascending pass, as before the bump, at no extra cost.
    pub npc_queues: bool,
    /// Generation 37 (N2, `docs/plan-npc.md` §4.2): whether a shield and wild pieces share
    /// the game. Only then does a wild step, a return or a wild threat ask whether the piece
    /// in its way is shielded (**a wild piece never takes a shielded piece**). Derived here
    /// from the sight's flags and `has_npcs`, so nothing new crosses the boundary.
    pub npc_shield: bool,

    /// Schema 19: the portal tables, or `None` for every game without portals.
    pub portals: Option<Box<PortalTables>>,
    /// Schema 26 (Q918a): simultaneous moves, `None` for every alternating game. Boxed, as
    /// the portals are, so an alternating game pays one pointer for it.
    pub simultaneous: Option<Box<SimTables>>,
    /// Schema 27 (Q935a): the spy, `None` for every game without one, boxed as the rounds
    /// are so a game without one pays one pointer. Never without
    /// `simultaneous` (the validator refuses it, and so does `from_py`).
    pub spy: Option<Box<SpyTables>>,
    /// Schema 29 (Q983a): the rest, `None` for every game in which no type rests, alternating
    /// or in rounds.
    pub rest: Option<Box<RestTables>>,
    /// Schema 30 (Q999a): lines of sight, `None` for every game in which no type sees.
    pub sight: Option<Box<SightTables>>,
    /// Generation 36 (N1, `docs/plan-npc.md` §4.1): wild pieces that come back, `None` for
    /// every game in which no `npcs` entry says `respawn`.
    pub respawn: Option<Box<RespawnTables>>,
    /// Generation 38 (N5, `docs/plan-npc.md` §4.7): wild pieces that turn back, `None` for
    /// every game in which no `npcs` entry says `turns_back`.
    pub npc_lines: Option<Box<LineTables>>,
    /// Generation 39 (N3b, `docs/plan-npc.md` §4.3): wild pieces that pounce, `None` for
    /// every game in which no `npcs` entry says `pounce`.
    pub pounce: Option<Box<PounceTables>>,
    /// Generation 42 (N4b, `docs/plan-npc.md` §4.6): the duck, `None` for every game without
    /// `rules.duck`.
    pub duck: Option<Box<DuckRule>>,
    /// The duck's byte, or 0 without a duck: what `is_wild_piece` tells a wild piece from the
    /// duck by, since both stand in the third owner's block.
    pub duck_byte: u8,

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

    /// Whether `value` is a wild piece (`rules.npcs`): the third owner's byte, and not the
    /// duck's (generation 42), which stands in the same block and never walks a route.
    #[inline(always)]
    pub fn is_wild_piece(&self, value: u8) -> bool {
        value != 0 && self.owner_of(value) == WILD && value != self.duck_byte
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

    /// How many orientations a move may turn this type through: its `facings_of`, or 1
    /// for a facing no move turns (schema 25), whose orientation is the setup's alone.
    #[inline(always)]
    pub fn turns_of(&self, piece_type: usize) -> usize {
        match self.facing_turns.get(piece_type) {
            Some(false) => 1,
            _ => self.facings_of(piece_type),
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

    /// What taking the wild piece `value` standing on `square` pays: the bounty of the route the
    /// square is on, or, in a pounce game (generation 39), of the piece's type, since a pouncer
    /// leaves its route (entries of one type agree on it, the validator says).
    #[inline]
    pub fn wild_bounty(&self, value: u8, square: usize) -> f64 {
        match self.pounce.as_deref() {
            None => self.npc_bounty[square],
            Some(pounce) => pounce.bounty[self.type_of(value)],
        }
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

    /// Schema 18 (Q783): whether this promoting type offers a target only while its side
    /// is short of its opening count of that type.
    #[inline(always)]
    pub fn promotes_only_lost(&self, piece_type: usize) -> bool {
        self.promo_only_lost_flags[piece_type]
    }

    /// Schema 18 (Q783): how many of `piece_type` the player started with.
    #[inline]
    pub fn opening_count(&self, player: usize, piece_type: usize) -> u32 {
        self.promo_opening[player][piece_type]
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
        // Schema 19: one byte per source, or empty when there are no portals.
        let attack_dirs: Vec<u8> = field(source, keys::ATTACK_DIRS)?;
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
                dirs: attack_dirs.get(index).copied().unwrap_or(0),
            })
            .collect();
        if !attack_dirs.is_empty() && attack_dirs.len() != attack_sources.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "attack_dirs holds {} bytes for {} attack sources",
                attack_dirs.len(), attack_sources.len(),
            )));
        }
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
        let promo_only_lost: Vec<usize> = field(source, keys::PROMO_ONLY_LOST)?;
        let promo_opening_flat: Vec<u32> = field(source, keys::PROMO_OPENING)?;
        let mut promo_opening = [[0u32; MAX_PIECE_TYPES]; 2];
        if !promo_opening_flat.is_empty() {
            for (player, row) in promo_opening.iter_mut().enumerate() {
                row[..n_types]
                    .copy_from_slice(&promo_opening_flat[player * n_types..(player + 1) * n_types]);
            }
        }

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
                4 => StrikePosition::Home,
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
            // Schema 17 (Q540): -2 removes the victim and leaves the attacker where it
            // stood, which only `stay` can say.
            if position != StrikePosition::Stay && to.contains(&REMOVED_IN_PLACE) {
                return Err(unknown("strike entry", "removed in place outside stay"));
            }
            let rebirth: Vec<i32> = field(source, keys::STRIKE_REBIRTH)?;
            let expected = if position == StrikePosition::Home { 2 * n_types * n_squares } else { 0 };
            if rebirth.len() != expected {
                return Err(unknown("strike rebirth length", &rebirth.len().to_string()));
            }
            let flips_raw: Vec<u8> = field(source, keys::CONTACT_MOVER_FLIPS)?;
            if !flips_raw.is_empty() && flips_raw.len() != n_types * n_types {
                return Err(unknown("contact mover flips length", &flips_raw.len().to_string()));
            }
            let mover_flips: Vec<bool> = flips_raw.iter().map(|v| *v != 0).collect();
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
                rebirth,
                mover_flips,
            })
        } else {
            None
        };

        // Drops
        let has_drops: bool = field(source, keys::HAS_DROPS)?;
        let mut drops = if has_drops {
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
            let placement: bool = field(source, keys::DROPS_PLACEMENT)?;
            let faces: Vec<i32> = field(source, keys::DROPS_FACES)?;
            let mut slot_of: Vec<usize> = Vec::new();
            if !faces.is_empty() {
                if faces.len() != n_types {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "drops_faces has {} entries for {} piece types",
                        faces.len(),
                        n_types
                    )));
                }
                slot_of = (0..n_types).collect();
                for (slot, &face) in faces.iter().enumerate() {
                    if face >= 0 {
                        let face = face as usize;
                        if face >= n_types || slot_of[face] != face {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "drops_faces pairs piece type {face} with more than one hand slot"
                            )));
                        }
                        slot_of[face] = slot;
                    }
                }
            }
            Some(DropRule {
                from_captures,
                regions,
                no_doubled: no_doubled_raw.iter().map(|v| *v != 0).collect(),
                no_mate: no_mate_raw.iter().map(|v| *v != 0).collect(),
                stock,
                unlimited: unlimited_raw.iter().map(|v| *v != 0).collect(),
                placement,
                // Settled below, once `royal_types` has been read.
                royal_stocked: false,
                faces,
                slot_of,
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
            let reach_leaves: bool = field(source, keys::REACH_LEAVES)?;
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
                leaves: reach_leaves,
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
            let ext_per_type: bool = field(source, keys::EXTINCTION_PER_TYPE)?;
            let ext_side: Vec<Vec<usize>> = field(source, keys::EXTINCTION_SIDE_TYPES)?;
            let ext_own = !ext_side.is_empty();
            let side_types = match <[Vec<usize>; 2]>::try_from(ext_side) {
                Ok(lists) => lists,
                Err(empty) if empty.is_empty() => [ext_types.clone(), ext_types.clone()],
                Err(other) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "extinction_side_types has {} lists; it is empty or one per player",
                        other.len()
                    )))
                }
            };
            Some(ExtinctionRule {
                counted: type_flags(&ext_types),
                piece_types: ext_types,
                any_piece: ext_any,
                min_count: ext_min,
                result: ext_result,
                per_type: ext_per_type,
                side_types,
                own: ext_own,
            })
        } else {
            None
        };

        let n_frames: usize = field(source, keys::N_FRAMES)?;
        let n_facings = n_frames / 2;
        let side_stride = n_facings * n_types;
        let facing_axes: Vec<usize> = field(source, keys::FACING_AXES)?;
        let facing_initial: Vec<usize> = field(source, keys::FACING_INITIAL)?;
        let facing_turns: Vec<bool> = field::<Vec<usize>>(source, keys::FACING_TURNS)?
            .into_iter()
            .map(|turns| turns != 0)
            .collect();

        // --- Wild pieces (schema 9) ---
        // Three owners share the one byte once anything is wild, which is what the bound
        // below has to know about; `_check_facing_coherence` refuses the same thing on
        // the Python side, with the same arithmetic and a better message.
        let has_npcs: bool = field(source, keys::HAS_NPCS)?;
        // Schema 19: portals are nobody's too, in the same third block.
        let has_portals: bool = field(source, keys::HAS_PORTALS)?;
        if has_npcs && has_portals {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "wild pieces and portals share the third owner's block; the validator                  refuses the two together",
            ));
        }
        // Generation 42 (N4b): the duck stands in the third block too. `[type, by_moves,
        // squares...]`, empty without a duck.
        let duck_field: Vec<u32> = field(source, keys::DUCK)?;
        if !duck_field.is_empty() && has_portals {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the duck and portals share the third owner's block; the validator refuses the two together",
            ));
        }
        let n_owners = if has_npcs || has_portals || !duck_field.is_empty() { 3 } else { 2 };
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
        // Schema 18 (Q784): a royal may start in the hand under `drops.placement`, and a
        // side whose royal is in its stock had one to lose as surely as one whose royal
        // was placed. Both players get the same stock (`_validate_stock`), so it is both
        // sides or neither.
        if let Some(rule) = drops.as_mut() {
            rule.royal_stocked = royal_types
                .iter()
                .any(|piece_type| rule.stock.get(*piece_type).is_some_and(|held| *held > 0));
            if rule.royal_stocked {
                royal_at_start = [true; 2];
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
            facing_turns,
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
            promo_only_lost_flags: type_flags(&promo_only_lost),
            promo_opening,
            promo_piece_types,
            castling,
            castle_files,
            multi_royal,
            demotions,
            explosion,
            strike,
            drops,
            reach_leaves: reach.as_ref().is_some_and(|rule| rule.leaves),
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
            npc_queues: false,
            npc_shield: false,
            portals: None,
            simultaneous: None,
            spy: None,
            rest: None,
            sight: None,
            respawn: None,
            npc_lines: None,
            pounce: None,
            duck: None,
            duck_byte: 0,
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

        // Generation 42 (N4b): the duck.
        if let [piece_type, by_moves, squares @ ..] = duck_field.as_slice() {
            let piece_type = *piece_type as usize;
            if piece_type >= tables.n_types || squares.iter().any(|&square| square as usize >= tables.n_squares) {
                return Err(pyo3::exceptions::PyValueError::new_err("the duck's tables do not fit this board"));
            }
            let mut set = SquareSet::EMPTY;
            for &square in squares {
                set.set(square as usize);
            }
            let value = (WILD * tables.side_stride + piece_type + 1) as u8;
            tables.duck_byte = value;
            tables.duck = Some(Box::new(DuckRule { value, piece_type, by_moves: *by_moves != 0, squares: set }));
        } else if !duck_field.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("a duck field too short to name its type"));
        }

        // What the bounties on the board come to at ply zero. Read off the opening
        // position rather than off `npc_bounty`, because a bounty is stated per *route*
        // and a route is several squares.
        if tables.has_npcs {
            tables.npc_bounty_total = (0..tables.n_squares)
                .filter(|square| {
                    tables.is_wild_piece(tables.initial_board[*square])
                })
                .map(|square| tables.npc_bounty[square])
                .sum();
            // One key per route, its lowest square; two wild pieces meeting in a key queue.
            let mut seen: Vec<usize> = Vec::new();
            for square in 0..tables.n_squares {
                let value = tables.initial_board[square];
                if !tables.is_wild_piece(value) || tables.npc_next[square] < 0 {
                    continue;
                }
                let mut lowest = square;
                let mut cursor = tables.npc_next[square] as usize;
                while cursor != square {
                    lowest = lowest.min(cursor);
                    let after = tables.npc_next[cursor] as usize;
                    // Generation 38: a line's last square holds itself, and a line starts
                    // where its piece does, so the walk to it has seen the whole line.
                    if after == cursor {
                        break;
                    }
                    cursor = after;
                }
                if seen.contains(&lowest) {
                    tables.npc_queues = true;
                }
                seen.push(lowest);
            }
        }

        // Generation 38 (N5): the lines' way back and the heading's keys, both empty unless
        // an entry turns back. Read before the respawning halves, which walk a line to its end.
        let npc_prev: Vec<i32> = field(source, keys::NPC_PREV)?;
        let zobrist_heading: Vec<u64> = field(source, keys::ZOBRIST_HEADING)?;
        if !npc_prev.is_empty() {
            tables.npc_lines = Some(Box::new(Self::line_tables(&tables, &npc_prev, zobrist_heading)?));
        } else if !zobrist_heading.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("heading keys without a line"));
        }

        // Generation 36 (N1): respawning halves, `[every, n_starts, starts...]` each, and
        // their clocks' keys. Both empty unless an entry says `respawn`.
        let npc_respawn: Vec<u32> = field(source, keys::NPC_RESPAWN)?;
        let zobrist_respawn: Vec<u64> = field(source, keys::ZOBRIST_RESPAWN)?;
        if !npc_respawn.is_empty() {
            tables.respawn = Some(Box::new(Self::respawn_tables(&tables, &npc_respawn, zobrist_respawn)?));
        } else if !zobrist_respawn.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("respawn keys without a respawn"));
        }

        // Generation 39 (N3b): the pounce tables, every one empty unless an entry pounces.
        let pounce_kind: Vec<u8> = field::<Vec<u32>>(source, keys::POUNCE_KIND)?
            .into_iter()
            .map(|kind| kind as u8)
            .collect();
        if !pounce_kind.is_empty() {
            tables.pounce = Some(Box::new(Self::pounce_tables(&tables, source, pounce_kind)?));
        }

        if has_portals {
            tables.portals = Some(Box::new(Self::portal_tables(source, &tables)?));
        }


        let sim_rule: Vec<u32> = field(source, keys::SIM_RULE)?;
        if !sim_rule.is_empty() {
            let collision: Vec<u8> = field(source, keys::SIM_COLLISION)?;
            let zobrist: Vec<u64> = field(source, keys::ZOBRIST_SIM)?;
            let n = tables.n_squares;
            let void_run = sim_rule[2] as usize;
            if sim_rule.len() != 4
                || collision.len() != tables.n_types * tables.n_types
                || zobrist.len() != 1 + 2 * n + void_run
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the simultaneous tables do not fit this ruleset's types and squares",
                ));
            }
            // Schema 27 (Q942a): empty unless a chase switch is on. The first slot was the
            // round's rest, which schema 29 moved to `piece_rest`; it is always 0.
            let chase: Vec<u8> = field(source, keys::SIM_CHASE)?;
            let switch = |index: usize| chase.get(index).is_some_and(|flag| *flag != 0);
            if !(chase.is_empty() || chase.len() == 3) || switch(0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the chase switches do not fit this core (the round's rest is pieces[].rest)",
                ));
            }
            // Schema 28 (Q943a): empty unless a piece declares a speed.
            let speed: Vec<u8> = field(source, keys::SIM_SPEED)?;
            if !(speed.is_empty() || speed.len() == tables.n_types) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the speeds do not fit this ruleset's types",
                ));
            }
            tables.simultaneous = Some(Box::new(SimTables {
                speed,
                swap_collides: switch(1),
                both_score: switch(2),
                tie: sim_rule[0] as u8,
                initiative_start: sim_rule[1] as i8,
                void_run: sim_rule[2] as u8,
                collision_strikes: sim_rule[3] != 0,
                collision_allowed: collision.iter().map(|flag| *flag != 0).collect(),
                zobrist_initiative: zobrist[0],
                zobrist_squares: zobrist[1..1 + 2 * n].to_vec(),
                zobrist_states: zobrist[1 + 2 * n..].to_vec(),
            }));
        }

        // Schema 29 (Q983a): the rest, both empty unless a type rests.
        let piece_rest: Vec<u8> = field(source, keys::PIECE_REST)?;
        let zobrist_rest: Vec<u64> = field(source, keys::ZOBRIST_REST)?;
        if !piece_rest.is_empty() {
            let max = piece_rest.iter().copied().max().unwrap_or(0) as usize;
            if piece_rest.len() != tables.n_types
                || max == 0
                || max > REST_MAX
                || zobrist_rest.len() != max * 2 * tables.n_squares
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the rest tables do not fit this ruleset's types and squares",
                ));
            }
            tables.rest =
                Some(Box::new(RestTables { turns: piece_rest, zobrist: zobrist_rest, slots: max }));
        } else if !zobrist_rest.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rest keys without a rest",
            ));
        }

        // Schema 30 (Q999a): lines of sight, all four empty unless a type sees.
        let sight_index: Vec<u32> = field(source, keys::SIGHT_INDEX)?;
        let sight_srcs: Vec<u32> = field(source, keys::SIGHT_SRCS)?;
        let sight_paths: Vec<u16> = field(source, keys::SIGHT_PATHS)?;
        let sight_flags: Vec<u8> = field(source, keys::SIGHT_FLAGS)?;
        let sight_command: bool = field(source, keys::SIGHT_COMMAND)?;
        if !sight_flags.is_empty() {
            let sources: Vec<SightSource> = sight_srcs
                .chunks_exact(5)
                .map(|chunk| SightSource {
                    origin: chunk[0] as u16,
                    piece_value: chunk[1] as u8,
                    effect: chunk[2] as u8,
                    path_start: chunk[3],
                    path_len: chunk[4],
                })
                .collect();
            let fits = sight_flags.len() == 3
                && sight_srcs.len().is_multiple_of(5)
                && sight_index.len() == 2 * tables.n_types * tables.n_squares + 1
                && sight_index.last().copied() == Some(sources.len() as u32)
                && sources.iter().all(|s| {
                    (s.origin as usize) < tables.n_squares
                        && (s.path_start + s.path_len) as usize <= sight_paths.len()
                        && (matches!(s.effect, SIGHT_FREEZE | SIGHT_DISARM | SIGHT_SHIELD)
                            || (sight_command
                                && (s.effect == SIGHT_COMMAND
                                    || s.effect == SIGHT_COMMAND | SIGHT_COMMAND_OWN)))
                })
                && sight_paths.iter().all(|square| (*square as usize) < tables.n_squares)
                && tables.portals.is_none();
            if !fits {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the sight tables do not fit this ruleset's squares (or sit beside portals)",
                ));
            }
            tables.sight = Some(Box::new(SightTables {
                index: sight_index,
                sources,
                paths: sight_paths,
                has_shield: sight_flags[0] != 0,
                bears_on_check: sight_flags[1] != 0,
                has_restraint: sight_flags[2] != 0,
                has_command: sight_command,
                block: tables.n_types * tables.n_squares,
                n_squares: tables.n_squares,
            }));
            tables.npc_shield = tables.has_npcs && sight_flags[0] != 0;
        } else if !sight_index.is_empty()
            || !sight_srcs.is_empty()
            || !sight_paths.is_empty()
            || sight_command
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sight sources without the sight's flags",
            ));
        }

        // Schema 27 (Q935a): the spy, empty unless a piece is one.
        let spy_types: Vec<Vec<usize>> = field(source, keys::SPY_TYPES)?;
        let spy_deal: bool = field(source, keys::SPY_DEAL)?;
        if !spy_types.is_empty() {
            let fits = spy_types.len() == 2
                && tables.simultaneous.is_some()
                && spy_types.iter().all(|types| types.iter().all(|&t| t < tables.n_types));
            if !fits {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the spy tables are two lists of this ruleset's types, beside rules.simultaneous",
                ));
            }
            let mask = |types: &[usize]| types.iter().fold(0u32, |mask, &t| mask | (1 << t));
            tables.spy = Some(Box::new(SpyTables {
                mask: [mask(&spy_types[0]), mask(&spy_types[1])],
                deal: spy_deal,
            }));
        } else if spy_deal {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "spy_deal without a spy",
            ));
        }

        tables.hop_lines = tables.build_hop_lines();
        Ok(tables)
    }
}

impl Tables {
    /// Generation 38 (N5): read the lines, each square checked against `npc_next`: a line
    /// square is on a route, and its way back and its way on are each other's inverse.
    fn line_tables(tables: &Tables, prev: &[i32], zobrist: Vec<u64>) -> PyResult<LineTables> {
        let bad = || pyo3::exceptions::PyValueError::new_err("the line tables do not fit this board's routes");
        let n = tables.n_squares;
        if !tables.has_npcs || prev.len() < n || zobrist.len() < n {
            return Err(bad());
        }
        let prev: Vec<i16> = prev.iter().map(|p| *p as i16).collect();
        let mut squares = SquareSet::EMPTY;
        for square in 0..n {
            let back = prev[square];
            if back < 0 {
                continue;
            }
            let on = tables.npc_next[square];
            if on < 0 || back as usize >= n {
                return Err(bad());
            }
            if back as usize != square && tables.npc_next[back as usize] as usize != square {
                return Err(bad());
            }
            if on as usize != square && prev[on as usize] as usize != square {
                return Err(bad());
            }
            squares.set(square);
        }
        Ok(LineTables { prev, squares, zobrist })
    }

    /// Generation 39 (N3b): read the pounce tables, each checked for its size. A pounce needs
    /// wild pieces, and is refused beside a line and a respawn (`schema._check_npc_pounce`),
    /// whose phases do not list a pouncer.
    fn pounce_tables(tables: &Tables, source: &Bound<'_, PyDict>, kind: Vec<u8>) -> PyResult<PounceTables> {
        let bad = |what: &str| {
            pyo3::exceptions::PyValueError::new_err(format!("the pounce tables do not fit this board: {what}"))
        };
        let n = tables.n_squares;
        let n_types = tables.n_types;
        if !tables.has_npcs || tables.npc_lines.is_some() || tables.respawn.is_some() {
            return Err(bad("a pounce beside no wild piece, a line or a respawn"));
        }
        let value_rank: Vec<u8> = field::<Vec<u32>>(source, keys::POUNCE_VALUE_RANK)?
            .into_iter()
            .map(|rank| rank.min(255) as u8)
            .collect();
        let reading: Vec<u16> = field::<Vec<u32>>(source, keys::POUNCE_READING)?
            .into_iter()
            .map(|rank| rank as u16)
            .collect();
        let distance: Vec<u8> = field::<Vec<u32>>(source, keys::POUNCE_DISTANCE)?
            .into_iter()
            .map(|steps| steps.min(255) as u8)
            .collect();
        let targets_index: Vec<u32> = field(source, keys::POUNCE_REACH_INDEX)?;
        let flat: Vec<u32> = field(source, keys::POUNCE_REACH)?;
        let paths: Vec<u16> = field::<Vec<u32>>(source, keys::POUNCE_PATHS)?
            .into_iter()
            .map(|square| square as u16)
            .collect();
        let piece: Vec<i8> = field::<Vec<i32>>(source, keys::NPC_PIECE)?
            .into_iter()
            .map(|ty| ty as i8)
            .collect();
        if kind.len() != n_types
            || value_rank.len() != n_types
            || reading.len() != 2 * n
            || distance.len() != n * n
            || targets_index.len() != n_types * n + 1
            || piece.len() != n
            || !flat.len().is_multiple_of(5)
        {
            return Err(bad("a table of the wrong size"));
        }
        let targets: Vec<ReachEntry> = flat
            .chunks_exact(5)
            .map(|chunk| ReachEntry {
                target: chunk[0] as u16,
                path_start: chunk[1],
                path_len: chunk[2] as u16,
                prefix_len: chunk[3] as u16,
                screens: chunk[4] as u8,
            })
            .collect();
        let in_bounds = targets.iter().all(|entry| {
            (entry.target as usize) < n
                && entry.path_start as usize + entry.path_len as usize <= paths.len()
                && entry.prefix_len <= entry.path_len
        }) && targets_index.windows(2).all(|pair| pair[0] <= pair[1])
            && *targets_index.last().expect("sized above") as usize == targets.len()
            && paths.iter().all(|square| (*square as usize) < n);
        if !in_bounds {
            return Err(bad("a reach entry off the board"));
        }
        let mut bounty = vec![0.0; n_types];
        for (square, ty) in piece.iter().enumerate() {
            if *ty >= 0 && (*ty as usize) < n_types {
                bounty[*ty as usize] = tables.npc_bounty[square];
            }
        }
        if kind.iter().any(|k| *k > POUNCE_ATTACKERS) {
            return Err(bad("a pounce of no kind the core knows"));
        }
        let attackers = kind.contains(&POUNCE_ATTACKERS);
        // Generation 41 (N3d): the split caps, only on a type that pounces on its attackers.
        let split: Vec<u8> = field::<Vec<u32>>(source, keys::POUNCE_SPLIT)?
            .into_iter()
            .map(|cap| cap.min(255) as u8)
            .collect();
        if !split.is_empty()
            && (split.len() != n_types
                || split.iter().zip(&kind).any(|(cap, k)| *cap != 0 && *k != POUNCE_ATTACKERS))
        {
            return Err(bad("a split on a type that does not pounce on its attackers"));
        }
        Ok(PounceTables {
            kind, attackers, value_rank, reading, distance, targets_index, targets, paths, piece, bounty, split,
            n_squares: n,
        })
    }

    /// Generation 36 (N1): read the respawning halves, each checked against the board. The
    /// route is walked off `npc_next` from the first start, and the piece's byte read off the
    /// opening board there, so neither crosses the boundary twice.
    fn respawn_tables(tables: &Tables, flat: &[u32], zobrist: Vec<u64>) -> PyResult<RespawnTables> {
        let bad = || pyo3::exceptions::PyValueError::new_err("the respawn tables do not fit this board's routes");
        let n = tables.n_squares;
        let mut halves = Vec::new();
        let mut at = 0;
        let mut key_base = 0;
        while at < flat.len() {
            let every = *flat.get(at).ok_or_else(bad)?;
            let count = *flat.get(at + 1).ok_or_else(bad)? as usize;
            let starts: Vec<u16> = flat.get(at + 2..at + 2 + count).ok_or_else(bad)?.iter().map(|s| *s as u16).collect();
            at += 2 + count;
            if !(1..=16).contains(&every) || starts.is_empty() || starts.iter().any(|s| *s as usize >= n) {
                return Err(bad());
            }
            let first = starts[0] as usize;
            let value = tables.initial_board[first];
            if !tables.has_npcs || value == 0 || tables.owner_of(value) != WILD || tables.npc_next[first] < 0 {
                return Err(bad());
            }
            let mut route = vec![first as u16];
            let mut cursor = tables.npc_next[first] as usize;
            while cursor != first {
                route.push(cursor as u16);
                let after = tables.npc_next[cursor] as usize;
                // Generation 38: a line ends on a square that holds itself.
                if after == cursor {
                    break;
                }
                cursor = after;
                if route.len() > n {
                    return Err(bad());
                }
            }
            if starts.iter().any(|s| !route.contains(s)) {
                return Err(bad());
            }
            halves.push(RespawnHalf { every: every as u8, starts, route, value, key_base });
            key_base += every as usize;
        }
        if zobrist.len() != key_base {
            return Err(bad());
        }
        Ok(RespawnTables { halves, zobrist })
    }

    /// Schema 19: read the portal tables, checking each against the shape it indexes.
    fn portal_tables(source: &Bound<'_, PyDict>, tables: &Tables) -> PyResult<PortalTables> {
        let fail = |what: String| Err(pyo3::exceptions::PyValueError::new_err(what));
        let n_types = tables.n_types;
        let dest_raw: Vec<i32> = field(source, keys::PORTAL_DEST)?;
        let group_raw: Vec<i32> = field(source, keys::PORTAL_GROUP)?;
        let admits_raw: Vec<u32> = field(source, keys::PORTAL_ADMITS)?;
        let movers_raw: Vec<u8> = field(source, keys::PORTAL_MOVERS)?;
        if dest_raw.len() != n_types || group_raw.len() != n_types
            || admits_raw.len() != 2 * n_types || movers_raw.len() != n_types
        {
            return fail(format!("portal tables are not one row per each of {n_types} types"));
        }
        let mut dest = [-1i8; MAX_PIECE_TYPES];
        let mut admits = [[0u32; 2]; MAX_PIECE_TYPES];
        let mut movers = [0u8; MAX_PIECE_TYPES];
        for piece_type in 0..n_types {
            dest[piece_type] = dest_raw[piece_type] as i8;
            admits[piece_type] = [admits_raw[2 * piece_type], admits_raw[2 * piece_type + 1]];
            movers[piece_type] = movers_raw[piece_type];
        }
        // The ends are the types with a group; every `dest` names another of them.
        let ends: Vec<usize> = (0..n_types).filter(|t| group_raw[*t] >= 0).collect();
        let first_type = ends.first().copied().unwrap_or(0);
        if ends.len() < 2 || ends.len() > 8
            || ends.iter().enumerate().any(|(k, t)| *t != first_type + k)
        {
            return fail("portal ends are not two to eight contiguous types".to_string());
        }
        let is_end = |t: usize| t >= first_type && t < first_type + ends.len();
        if (0..n_types).any(|t| {
            let to = dest[t];
            (to >= 0 && (!is_end(t) || !is_end(to as usize) || to as usize == t))
                || (!is_end(t) && admits[t] != [0, 0])
        }) {
            return fail("portal_dest or portal_admits names a type that is not an end".to_string());
        }
        let exits: Vec<i16> = field::<Vec<i32>>(source, keys::PORTAL_EXITS)?
            .into_iter()
            .map(|square| square as i16)
            .collect();
        if exits.len() != 2 * tables.n_squares * 8 {
            return fail(format!("portal_exits holds {} entries", exits.len()));
        }
        let leap_dirs: Vec<u16> = field(source, keys::LEAP_DIRS)?;
        let slide_dirs: Vec<u16> = field(source, keys::SLIDE_DIRS)?;
        let hop_dirs: Vec<u16> = field(source, keys::HOP_DIRS)?;
        // Schema 20: empty unless some pair sets `no_straight_back`.
        let no_back_raw: Vec<u8> = field(source, keys::PORTAL_NO_BACK)?;
        let zobrist_bar: Vec<u64> = field(source, keys::ZOBRIST_PORTAL_BAR)?;
        if !no_back_raw.is_empty() && no_back_raw.len() != n_types {
            return fail(format!(
                "portal_no_back holds {} rows for {n_types} types", no_back_raw.len(),
            ));
        }
        let has_no_back = no_back_raw.iter().any(|flag| *flag != 0);
        if zobrist_bar.len() != if has_no_back { 2 * tables.n_squares } else { 0 } {
            return fail(format!("zobrist_portal_bar holds {} keys", zobrist_bar.len()));
        }
        let mut no_back = [false; MAX_PIECE_TYPES];
        for (piece_type, flag) in no_back_raw.iter().enumerate() {
            no_back[piece_type] = *flag != 0;
        }
        for (name, dirs, table) in [
            ("leap", &leap_dirs, &tables.leaps),
            ("slide", &slide_dirs, &tables.slides),
            ("hop", &hop_dirs, &tables.hops),
        ] {
            if dirs.len() != table.n_landings() {
                return fail(format!(
                    "{name}_dirs holds {} masks for {} landings", dirs.len(), table.n_landings(),
                ));
            }
        }
        let folds = Self::fold_tables(source, tables, first_type, ends.len())?;
        let n_momentum = match folds.as_ref() {
            Some(found) => found.starts.iter().take(ends.len()).filter(|s| **s < 0).count(),
            None => ends.len(),
        };
        Ok(PortalTables {
            n_momentum,
            folds,
            first_type,
            n_ends: ends.len(),
            dest,
            admits,
            movers,
            exits,
            leap_dirs,
            slide_dirs,
            hop_dirs,
            no_back,
            has_no_back,
            zobrist_bar,
        })
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

/// Schema 22: fold portals, as the compiler resolved them (`compile.FoldRule`); schema 24
/// (Q867b) cut them to where each starts and the key (`docs/archive/plan-portals-2.md` §3R.3).
///
/// A fold end never stands on the board byte: where each stands is `Game::fold_at`, from
/// `starts`. A landing on an empty fold goes through to the square of the end it leads to
/// (`PortalTables::dest`), so the core needs no geometry of its own for a fold.
pub struct FoldTables {
    /// Where each portal end starts if it is a fold, by end; -1 for a momentum end.
    pub starts: [i16; 8],
    /// `[end * n_squares + square]`: a fold standing on `square`, in the key.
    pub zobrist: Vec<u64>,
}

impl FoldTables {
    /// The key of fold end `end` standing on `square`.
    #[inline(always)]
    pub fn key(&self, n_squares: usize, end: usize, square: usize) -> u64 {
        self.zobrist[end * n_squares + square]
    }
}

impl PortalTables {
    /// Q883: whether a move by an end of type `piece_type` sets the straight-back bar,
    /// given who moves next (`next`): the end bars going back (`no_back`, which the
    /// compiler leaves false where no move can ever meet the bar) and `next` may move it.
    /// A bar nobody can meet on the next move is no bar, so it is never set, and two
    /// positions alike in everything that affects play are not keyed apart by it.
    #[inline]
    pub fn sets_bar(&self, piece_type: usize, next: usize) -> bool {
        self.no_back[piece_type] && self.movers[piece_type] & (1 << next) != 0
    }

    /// The key term of a straight-back bar: the end standing on `bar[0]` may not move to
    /// `bar[1]`. Zero for no bar, and for every game without the rule.
    #[inline]
    pub fn bar_key(&self, n_squares: usize, bar: [i16; 2]) -> u64 {
        if bar[0] < 0 || self.zobrist_bar.is_empty() {
            0
        } else {
            self.zobrist_bar[bar[1] as usize] ^ self.zobrist_bar[n_squares + bar[0] as usize]
        }
    }

    /// The square one king's step from `square` in `direction`, for `player`, or -1.
    #[inline(always)]
    pub fn exit(&self, n_squares: usize, player: usize, square: usize, direction: usize) -> i16 {
        self.exits[(player * n_squares + square) * 8 + direction]
    }
}

impl Tables {
    /// Schema 22: read the fold tables, or `None` where no portal group is a fold. Cold and
    /// out of line: it runs once per ruleset, and folded into `portal_tables` it moved chess's
    /// perft by the code it shifted, though chess never reaches it.
    #[cold]
    #[inline(never)]
    fn fold_tables(
        source: &Bound<'_, PyDict>, tables: &Tables, first_type: usize, n_ends: usize,
    ) -> PyResult<Option<FoldTables>> {
        let fail = |what: String| Err(pyo3::exceptions::PyValueError::new_err(what));
        let kind: Vec<u8> = field(source, keys::PORTAL_KIND)?;
        if kind.is_empty() {
            return Ok(None);
        }
        let n_squares = tables.n_squares;
        let starts_raw: Vec<i32> = field(source, keys::FOLD_STARTS)?;
        if kind.len() != tables.n_types || starts_raw.len() != n_ends {
            return fail("portal_kind or fold_starts is not one row per type or end".to_string());
        }
        let mut starts = [-1i16; 8];
        for (end, square) in starts_raw.iter().enumerate() {
            let fold = kind[first_type + end] != 0;
            if fold != (*square >= 0) || *square >= n_squares as i32 {
                return fail(format!("fold_starts[{end}] does not match portal_kind"));
            }
            starts[end] = *square as i16;
        }
        let zobrist: Vec<u64> = field(source, keys::ZOBRIST_FOLD)?;
        if zobrist.len() != n_ends * n_squares {
            return fail("the fold tables do not fit the shapes they index".to_string());
        }
        Ok(Some(FoldTables { starts, zobrist }))
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
        ATTACK_DIRS => "attack_dirs",
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
        CONTACT_MOVER_FLIPS => "contact_mover_flips",
        CONTACT_MOVER_TO => "contact_mover_to",
        CONTACT_ONTO_TO => "contact_onto_to",
        CONTACT_OWN_MOVER_TO => "contact_own_mover_to",
        CONTACT_OWN_ONTO_TO => "contact_own_onto_to",
        DEDUP => "dedup",
        DEMOTIONS => "demotions",
        DROPS_FACES => "drops_faces",
        DROPS_FROM_CAPTURES => "drops_from_captures",
        DROPS_NO_DOUBLED => "drops_no_doubled",
        DROPS_NO_MATE => "drops_no_mate",
        DROPS_PLACEMENT => "drops_placement",
        DROPS_REGIONS => "drops_regions",
        DROPS_STOCK => "drops_stock",
        DROPS_UNLIMITED => "drops_unlimited",
        DUCK => "duck",
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
        EXTINCTION_PER_TYPE => "extinction_per_type",
        EXTINCTION_RESULT => "extinction_result",
        EXTINCTION_SIDE_TYPES => "extinction_side_types",
        EXTINCTION_TYPES => "extinction_types",
        FOLD_STARTS => "fold_starts",
        FACING_AXES => "facing_axes",
        FACING_ENTRIES => "facing_entries",
        FACING_INDEX => "facing_index",
        FACING_INITIAL => "facing_initial",
        FACING_TURNS => "facing_turns",
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
        HAS_PORTALS => "has_portals",
        HAS_REACH => "has_reach",
        HAS_SCORING => "has_scoring",
        HAS_STRIKE => "has_strike",
        HAS_SWEEP => "has_sweep",
        HAS_TERRITORY => "has_territory",
        HAS_TURN => "has_turn",
        HEIGHT => "height",
        HOP_DIRS => "hop_dirs",
        HOP_INDEX => "hop_index",
        HOP_MODES => "hop_modes",
        HOP_REFS => "hop_refs",
        HOP_SQUARES => "hop_squares",
        IMMOBILE_SQUARES => "immobile_squares",
        INITIAL_BOARD => "initial_board",
        INITIAL_UNMOVED => "initial_unmoved",
        LEAP_DIRS => "leap_dirs",
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
        PORTAL_ADMITS => "portal_admits",
        PORTAL_DEST => "portal_dest",
        PORTAL_EXITS => "portal_exits",
        PORTAL_GROUP => "portal_group",
        PORTAL_KIND => "portal_kind",
        PORTAL_MOVERS => "portal_movers",
        PORTAL_NO_BACK => "portal_no_back",
        PROMO_ON_LEAVING => "promo_on_leaving",
        PROMO_ONLY_LOST => "promo_only_lost",
        PROMO_OPENING => "promo_opening",
        PROMO_OPTIONAL => "promo_optional",
        PROMO_PIECE_TYPES => "promo_piece_types",
        PROMO_SQUARES => "promo_squares",
        PROMO_TARGETS_FLAT => "promo_targets_flat",
        PROMO_TARGETS_INDEX => "promo_targets_index",
        REACH_LEAVES => "reach_leaves",
        REACH_PIECE_TYPE => "reach_piece_type",
        REACH_RESULT => "reach_result",
        REACH_SAFE => "reach_safe",
        REACH_SQUARES => "reach_squares",
        REPETITION_DRAW => "repetition_draw",
        ROYAL_FACING_FORBIDDEN => "royal_facing_forbidden",
        ROYAL_TYPES => "royal_types",
        SCORING_COUNT => "scoring_count",
        SCORING_SQUARES => "scoring_squares",
        SLIDE_DIRS => "slide_dirs",
        SLIDE_INDEX => "slide_index",
        SLIDE_MODES => "slide_modes",
        SLIDE_REFS => "slide_refs",
        SLIDE_SQUARES => "slide_squares",
        STALEMATE => "stalemate",
        STRIKE_LANDING => "strike_landing",
        STRIKE_LANDING_INDEX => "strike_landing_index",
        STRIKE_POSITION => "strike_position",
        STRIKE_REBIRTH => "strike_rebirth",
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
        ZOBRIST_FOLD => "zobrist_fold",
        SIM_CHASE => "sim_chase",
        SIM_COLLISION => "sim_collision",
        SIM_RULE => "sim_rule",
        SIM_SPEED => "sim_speed",
        SIGHT_COMMAND => "sight_command",
        SIGHT_FLAGS => "sight_flags",
        SIGHT_INDEX => "sight_index",
        SIGHT_PATHS => "sight_paths",
        SIGHT_SRCS => "sight_srcs",
        SPY_DEAL => "spy_deal",
        SPY_TYPES => "spy_types",
        ZOBRIST_REST => "zobrist_rest",
        PIECE_REST => "piece_rest",
        NPC_RESPAWN => "npc_respawn",
        ZOBRIST_RESPAWN => "zobrist_respawn",
        NPC_PREV => "npc_prev",
        ZOBRIST_HEADING => "zobrist_heading",
        NPC_PIECE => "npc_piece",
        POUNCE_KIND => "pounce_kind",
        POUNCE_VALUE_RANK => "pounce_value_rank",
        POUNCE_READING => "pounce_reading",
        POUNCE_DISTANCE => "pounce_distance",
        POUNCE_REACH_INDEX => "pounce_reach_index",
        POUNCE_REACH => "pounce_reach",
        POUNCE_PATHS => "pounce_paths",
        POUNCE_SPLIT => "pounce_split",
        ZOBRIST_SIM => "zobrist_sim",
        ZOBRIST_PIECES => "zobrist_pieces",
        ZOBRIST_PORTAL_BAR => "zobrist_portal_bar",
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
