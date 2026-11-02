//! A forward endgame solver (N1, `docs/reference/no-forced-mate.md` §5).
//!
//! For a material set -- the pieces each side has, named by type -- this labels every position
//! of the set, and of every set it leads to, **won**, **lost** or **drawn** for the side to
//! move, with the distance to the end in plies. It answers the question "can this material
//! force a win in this game?" from the game's own rules and nothing else.
//!
//! **Forward only.** A position's moves come from `Game::generate_legal`, its children from
//! `make`, its ending from `terminal_with`: the only definition of a legal move and an ending
//! the core has (rules 3 and 4). Nothing here generates a move backwards, so nothing here is a
//! second reading of any primitive. The walk generates each position's moves once, stores the
//! children as edges, and the labels are then propagated over the stored edges in memory, in
//! sweeps by distance, with no movegen at all.
//!
//! **The closure.** The walk starts from every placement of the set and follows every move, so
//! whatever a move turns the material into -- a capture, a promotion, a circe rebirth, a side
//! switch -- is reached and solved in the same pass, and reported as a set of its own.
//!
//! **A position's identity** is the core's Zobrist key, extended by the state the key leaves out
//! on purpose because the repetition rule never compares across it (the first-move bits, the
//! check counts, the points, the hands, the goal just left, a permanent colouring). A table is not
//! a repetition window, so it needs them. Two distinct positions sharing a 64-bit identity would be
//! merged; at the cap that is a chance of about 10^-5, which is accepted, not checked.
//!
//! **The clocks are left out.** Every position is solved with its no-progress clock and ply count
//! at zero and no repetition history, so a draw here means neither side can force an ending, and
//! a win longer than the game's own no-progress clock is reported apart (`beyond_clock`), measured
//! as the distance to the end or to the next move that resets the clock, the winner taking the
//! shortest such line and the loser the longest (Syzygy's DTZ, without its fifty-move nesting).
//!
//! **What is refused, by name** (`refusal`): a capture hand or an endless stock (material never
//! shrinks), the purse (coins buy pieces), rounds and spies (a position's value is not won, lost or
//! drawn), and two kinds of nobody's piece whose squares are state beyond a placement: wild pieces
//! on routes and the duck.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::game::{scan_occupied, Game, Reason, Snapshot};
use crate::tables::{EntryRef, EntryTable, SquareSet, Tables, Win, CONTROL_DUCK, MAX_SQUARES, WILD};

/// The most positions one solve may hold, closure included. About 200 bytes a position at the
/// worst (the identity map, the edges, the labels, the pending boards), so about 4 GB on this
/// shared 15 GB machine. Chess at one piece besides the kings is 5 x 10^5 a set, two knights
/// against a king 1.5 x 10^7; two pieces on a 6x6 board 3.4 x 10^6.
pub const MAX_POSITIONS: u64 = 20_000_000;

/// The most stored edges one solve may hold, 4 bytes each: 1.6 GB.
pub const MAX_EDGES: u64 = 400_000_000;

/// Every position is solved as a turn of the steady state, past any ramp of shorter first turns
/// (`TurnRule::length_of`).
pub const STEADY_TURNS: u32 = 1 << 20;

const ZEROING: u32 = 1 << 31;
const ID_MASK: u32 = ZEROING - 1;

const UNKNOWN: u8 = 0;
const WON: u8 = 1;
const LOST: u8 = 2;
/// An ended position that is drawn (stalemate, dead material, ...).
const DRAWN_END: u8 = 3;
/// A position the game could never reach: the side not to move is in check in a checkmate game
/// (`terminal_prefix`'s illegal position). Starting placements of that kind are left out before
/// they are interned (`Solver::add_start`); this is the core's verdict on one met anyway.
const ILLEGAL: u8 = 4;

/// The pieces of one material set, by type index: player 0's, player 1's, and nobody's.
pub type Material = [Vec<usize>; 3];

/// One material set's verdicts, for the side to move.
#[derive(Clone, Debug, Default)]
pub struct SetStats {
    pub material: Material,
    pub won: u64,
    pub lost: u64,
    pub drawn: u64,
    /// Placements of the set no legal move leaves, not solved (`Solver::add_start`).
    pub illegal: u64,
    /// The longest forced win, in plies to the end (a mate in n moves is 2n - 1 plies).
    pub longest_win: u32,
    /// The longest forced loss, in plies.
    pub longest_loss: u32,
    /// Won positions whose distance to the end or to the next clock reset exceeds the game's
    /// no-progress clock: wins its own rules would draw.
    pub beyond_clock: u64,
    /// The longest distance to the end or the next clock reset over the won positions.
    pub longest_to_reset: u32,
    pub requested: bool,
}

/// One solved position of the requested set, for a caller to check: its value for the side to
/// move (1 won, -1 lost) and its distance in plies.
pub struct Sample {
    pub snapshot: Snapshot,
    pub value: i8,
    pub distance: u32,
}

pub struct Solve {
    pub sets: Vec<SetStats>,
    pub positions: u64,
    pub edges: u64,
    pub sweeps: u32,
    pub samples: Vec<Sample>,
}

/// Why a game cannot be solved here, by the rule's name; `None` when it can.
pub fn refusal(t: &Tables) -> Option<String> {
    if let Some(drops) = t.drops.as_ref() {
        if drops.from_captures {
            return Some(
                "a capture hand (rules.drops from captures): every capture puts a piece in hand, so material never shrinks"
                    .into(),
            );
        }
        if drops.unlimited.iter().any(|endless| *endless) {
            return Some("an endless stock (rules.drops.stock \"unlimited\"): the hand never empties".into());
        }
    }
    if t.purse.is_some() {
        return Some("the purse (rules.purse): coins are state and buy pieces, so material is not closed".into());
    }
    if t.spy.is_some() {
        return Some("spies (pieces[].spy): a position is not a state the player knows".into());
    }
    if t.simultaneous.is_some() {
        return Some(
            "rounds (rules.simultaneous): a position's value is a matrix game, not won, lost or drawn".into(),
        );
    }
    if t.has_npcs {
        return Some("wild pieces (rules.npcs): their squares, clocks and headings are not enumerated here".into());
    }
    if t.duck.is_some() {
        return Some("the duck (rules.duck): its square and half-turn are not enumerated here".into());
    }
    None
}

/// The hasher for the identity map: the identity is already a mixed 64-bit hash.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type IdMap = HashMap<u64, u32, BuildHasherDefault<PassThrough>>;

/// splitmix64's finaliser, the mixer for every hash below.
#[inline]
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

struct Digest(u64);

impl Digest {
    fn add(&mut self, value: u64) {
        self.0 = mix(self.0 ^ value);
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.add(bytes.len() as u64);
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }
}

/// The state the Zobrist key leaves out and a table needs: the first-move bits, the check
/// counts, the hands, the points, the goal just left and the colouring. Read from a position's
/// parts, so a child can be looked up before it is copied out of the game.
fn extras_of(unmoved: SquareSet, checks: [u32; 2], hands: &[Vec<u32>; 2], points: [f64; 2], goal_left_by: i8, colour: &[u8]) -> u64 {
    let mut d = Digest(0x5eed);
    d.bytes(&unmoved.to_bytes());
    d.add(u64::from(checks[0]) | (u64::from(checks[1]) << 32));
    for hand in hands {
        for count in hand {
            d.add(u64::from(*count));
        }
    }
    d.add(points[0].to_bits());
    d.add(points[1].to_bits());
    d.add(goal_left_by as u64);
    d.bytes(colour);
    d.0
}

fn extras(s: &Snapshot) -> u64 {
    extras_of(s.unmoved, s.checks, &s.hands, s.points, s.goal_left_by, &s.colour)
}

/// A position's identity in the table: its Zobrist key and the state the key leaves out.
fn identity(s: &Snapshot) -> u64 {
    s.key ^ extras(s)
}

fn identity_of_game(game: &Game<'_>) -> u64 {
    game.key ^ extras_of(game.unmoved, game.checks, &game.hands, game.points, game.goal_left_by, &game.colour)
}

/// Everything in a snapshot but the board and what follows from it (the occupancy, the key):
/// what a pending position shares with others, stored once.
fn residual(s: &Snapshot) -> u64 {
    let mut d = Digest(extras(s));
    d.add(s.side as u64);
    d.add(s.ep_target as u64 & 0xffff | ((s.ep_victim as u64 & 0xffff) << 16));
    d.add(u64::from(s.remaining) | (u64::from(s.turns_taken) << 32));
    d.add(s.portal_bar[0] as u64 & 0xffff | ((s.portal_bar[1] as u64 & 0xffff) << 16));
    for square in s.fold_at {
        d.add(square as u64);
    }
    d.add(s.initiative as u64);
    d.add(s.void_run[0] as u64 & 0xffff | ((s.void_run[1] as u64 & 0xffff) << 16));
    for bar in &s.bars {
        d.add(bar[0] as u64 & 0xffff | ((bar[1] as u64 & 0xffff) << 16));
    }
    for player in &s.resting {
        for entry in player {
            d.add(entry.square as u64 & 0xffff | (u64::from(entry.left) << 16) | (u64::from(entry.code) << 24));
        }
    }
    d.bytes(&s.respawn);
    d.bytes(&s.heading_back.to_bytes());
    d.add(u64::from(s.purse[0]) | (u64::from(s.purse[1]) << 16));
    d.add(s.push_bar[0] as u64 & 0xffff | ((s.push_bar[1] as u64 & 0xffff) << 16));
    d.0
}

/// The clocks taken out of a position: what every position is solved at.
fn normalise(s: &mut Snapshot) {
    s.halfmove = 0;
    s.ply = 0;
    s.repetition.clear();
    s.turns_taken = STEADY_TURNS;
}

/// The material on a board, by owner, each list ascending. Portal ends are the board's, not the
/// set's.
fn material_of(t: &Tables, board: &[u8; MAX_SQUARES]) -> Material {
    let mut material: Material = [Vec::new(), Vec::new(), Vec::new()];
    for value in board.iter().take(t.n_squares) {
        if *value != 0 && !t.is_portal_end(*value) {
            material[t.owner_of(*value).min(WILD)].push(t.type_of(*value));
        }
    }
    for list in &mut material {
        list.sort_unstable();
    }
    material
}

/// Builds a whole position from a board, the side to move and the first-move bits, through the
/// boundary's own reading (`lib.rs::snapshot_from`), its clocks at zero and its key computed.
pub type Build<'a> = dyn Fn(&[u8], usize, [u8; 32]) -> Snapshot + 'a;
/// The board value a piece of `owner`'s takes on entering play, nobody's at facing 0.
fn value_for(t: &Tables, owner: usize, piece_type: usize) -> u8 {
    if owner < 2 {
        t.entering_value(owner, piece_type)
    } else {
        t.piece_value(WILD, 0, piece_type)
    }
}

/// One table's entries at `slot`, none where the table has no such slot (a sweep table is empty
/// unless something sweeps; a nobody's frame past the players' has no table).
fn entries_at(table: &EntryTable, slot: usize) -> &[EntryRef] {
    if slot + 1 < table.slots() {
        table.entries(slot)
    } else {
        &[]
    }
}

/// Whether the piece `value` on `square` has a move only an unmoved piece may make there.
fn first_move_square(t: &Tables, value: u8, square: usize) -> bool {
    let slot = t.slot(t.frame_of_value(value), t.type_of(value), square);
    [&t.leaps, &t.slides, &t.hops, &t.sweeps]
        .iter()
        .any(|table| entries_at(table, slot).iter().any(|entry| entry.requires_unmoved))
}

/// The squares a piece may start on: every square one of its own moves lands on, and every square
/// the opening board puts it on, less a zone where it must promote and the board's fixed bytes.
/// Read off the movement tables, so walls, absent cells and regions are the compiler's.
fn standing_squares(t: &Tables, owner: usize, value: u8, fixed: &[u8; MAX_SQUARES]) -> Vec<usize> {
    let n = t.n_squares;
    let piece_type = t.type_of(value);
    let mut can = vec![false; n];
    for (square, can) in can.iter_mut().enumerate() {
        *can = t.initial_board[square] == value;
    }
    let frame = t.frame_of_value(value);
    for table in [&t.leaps, &t.slides, &t.hops, &t.sweeps] {
        for origin in 0..n {
            for entry in entries_at(table, t.slot(frame, piece_type, origin)) {
                for (landing, mode) in table.squares(entry).iter().zip(table.modes(entry)) {
                    if *mode != 0 && (*landing as usize) < n {
                        can[*landing as usize] = true;
                    }
                }
            }
        }
    }
    let must_promote = owner < 2 && t.is_promoting(piece_type) && !t.promotion_optional(piece_type);
    (0..n)
        .filter(|&square| can[square] && fixed[square] == 0)
        .filter(|&square| !(must_promote && t.promotes_on(owner, piece_type, square)))
        .collect()
}

struct Placement {
    value: u8,
    squares: Vec<usize>,
}

/// The set's starting positions: every placement of its pieces on the fixed board.
struct Walk<'a, 't> {
    t: &'t Tables,
    build: &'a Build<'a>,
    placements: Vec<Placement>,
    fixed: [u8; MAX_SQUARES],
}

type Visit<'v> = dyn FnMut(&[u8; MAX_SQUARES]) -> Result<(), String> + 'v;

impl Walk<'_, '_> {
    /// Every placement, identical pieces on ascending squares so each board comes once.
    fn each_board(&self, visit: &mut Visit<'_>) -> Result<(), String> {
        let mut board = self.fixed;
        self.place(0, 0, &mut board, visit)
    }

    fn place(&self, index: usize, floor: usize, board: &mut [u8; MAX_SQUARES], visit: &mut Visit<'_>) -> Result<(), String> {
        let Some(piece) = self.placements.get(index) else {
            return visit(board);
        };
        let same_next = self.placements.get(index + 1).is_some_and(|next| next.value == piece.value);
        for &square in &piece.squares {
            if square < floor || board[square] != 0 {
                continue;
            }
            board[square] = piece.value;
            self.place(index + 1, if same_next { square + 1 } else { 0 }, board, visit)?;
            board[square] = 0;
        }
        Ok(())
    }

    /// The set's starting position on `board` with `side` to move. A piece standing on its own
    /// opening square keeps its first-move bit where it has a first move there; nothing else is
    /// unmoved, so no castling right survives into an endgame (as in Syzygy).
    fn start(&self, board: &[u8; MAX_SQUARES], side: usize) -> Snapshot {
        let t = self.t;
        let mut unmoved = SquareSet::EMPTY;
        for (square, &value) in board.iter().enumerate().take(t.n_squares) {
            if value != 0
                && self.fixed[square] == 0
                && t.initial_board[square] == value
                && first_move_square(t, value, square)
            {
                unmoved.set(square);
            }
        }
        (self.build)(&board[..t.n_squares], side, unmoved.to_bytes())
    }
}

/// A position waiting to be expanded: its id, its key and its shared residual state. Its board
/// is the matching `n_squares` bytes at the end of `Solver::boards` (both stacks pop together).
struct Pending {
    id: u32,
    key: u64,
    residual: u32,
}

struct Solver<'t> {
    t: &'t Tables,
    cap: u64,
    ids: IdMap,
    side: Vec<u8>,
    group: Vec<u32>,
    value: Vec<u8>,
    dist: Vec<u32>,
    edge_lo: Vec<u64>,
    edge_len: Vec<u32>,
    edges: Vec<u32>,
    groups: Vec<Material>,
    group_of: HashMap<Material, u32>,
    residuals: Vec<Snapshot>,
    residual_of: HashMap<u64, u32>,
    pending: Vec<Pending>,
    boards: Vec<u8>,
    moves: Vec<u32>,
    /// The game every position is expanded on (`add_start`).
    game: Option<Game<'t>>,
    /// Starting placements left out as impossible (`add_start`).
    impossible: u64,
}

impl<'t> Solver<'t> {
    fn new(t: &'t Tables, cap: u64) -> Self {
        Self {
            t,
            cap,
            ids: IdMap::default(),
            side: Vec::new(),
            group: Vec::new(),
            value: Vec::new(),
            dist: Vec::new(),
            edge_lo: Vec::new(),
            edge_len: Vec::new(),
            edges: Vec::new(),
            groups: Vec::new(),
            group_of: HashMap::new(),
            residuals: Vec::new(),
            residual_of: HashMap::new(),
            pending: Vec::new(),
            boards: Vec::new(),
            moves: Vec::new(),
            game: None,
            impossible: 0,
        }
    }

    /// The id of the position `s`, interning it if it is new (and then queueing it).
    fn intern(&mut self, s: &Snapshot) -> Result<u32, String> {
        let identity = identity(s);
        if let Some(id) = self.ids.get(&identity) {
            return Ok(*id);
        }
        let id = self.side.len() as u64;
        if id >= self.cap {
            return Err(format!(
                "the solve passed its cap of {} positions (endgame::MAX_POSITIONS is {MAX_POSITIONS}); solve a smaller set",
                self.cap
            ));
        }
        let id = id as u32;
        self.ids.insert(identity, id);
        let material = material_of(self.t, &s.board);
        let next = self.groups.len() as u32;
        let group = *self.group_of.entry(material.clone()).or_insert(next);
        if group == next {
            self.groups.push(material);
        }
        self.side.push(s.side as u8);
        self.group.push(group);
        self.value.push(UNKNOWN);
        self.dist.push(0);
        self.edge_lo.push(0);
        self.edge_len.push(0);
        let hash = residual(s);
        let index = match self.residual_of.get(&hash) {
            Some(index) => *index,
            None => {
                let mut kept = s.clone();
                kept.board = [0; MAX_SQUARES];
                self.residuals.push(kept);
                let index = (self.residuals.len() - 1) as u32;
                self.residual_of.insert(hash, index);
                index
            }
        };
        self.boards.extend_from_slice(&s.board[..self.t.n_squares]);
        self.pending.push(Pending { id, key: s.key, residual: index });
        Ok(id)
    }

    /// The next queued position, rebuilt from its board and residual.
    fn pop(&mut self) -> Option<(u32, Snapshot)> {
        let next = self.pending.pop()?;
        let n = self.t.n_squares;
        let lo = self.boards.len() - n;
        let mut s = self.residuals[next.residual as usize].clone();
        s.board[..n].copy_from_slice(&self.boards[lo..]);
        self.boards.truncate(lo);
        s.occupied = scan_occupied(self.t, &s.board);
        s.key = next.key;
        Some((next.id, s))
    }

    /// Solve forward from one starting position: everything it reaches, expanded once. A
    /// placement no legal move leaves -- in a game where check binds, the player who has just
    /// moved standing in check (the kings side by side, say) -- is counted and left out.
    fn add_start(&mut self, start: &Snapshot) -> Result<(), String> {
        if self.ids.contains_key(&identity(start)) {
            return Ok(());
        }
        if self.t.win == Win::Checkmate && Game::from_snapshot(self.t, start).in_check(1 - start.side) {
            self.impossible += 1;
            return Ok(());
        }
        self.intern(start)?;
        // One game for the whole walk, put back to each position without reallocating.
        let mut game = self.game.take().unwrap_or_else(|| Game::from_snapshot(self.t, start));
        let mut walked = Ok(());
        while let Some((id, s)) = self.pop() {
            game.restore(&s);
            walked = self.expand(&mut game, id, s.side);
            if walked.is_err() {
                break;
            }
        }
        self.game = Some(game);
        walked
    }

    /// One position's ending, or its moves stored as edges to its children. `game` stands on it.
    fn expand(&mut self, game: &mut Game<'t>, id: u32, side: usize) -> Result<(), String> {
        let t = self.t;
        let mut moves = std::mem::take(&mut self.moves);
        moves.clear();
        game.generate_legal(&mut moves);
        if let Some(outcome) = game.terminal_with(&moves) {
            self.value[id as usize] = if t.win == Win::Checkmate && outcome.reason == Reason::RoyalCaptured {
                ILLEGAL
            } else if outcome.winner < 0 {
                DRAWN_END
            } else if outcome.winner as usize == side {
                WON
            } else {
                LOST
            };
            self.moves = moves;
            return Ok(());
        }
        if (self.edges.len() + moves.len()) as u64 > MAX_EDGES {
            return Err(format!("the solve passed its cap of {MAX_EDGES} edges (endgame::MAX_EDGES); solve a smaller set"));
        }
        let lo = self.edges.len();
        for &m in &moves {
            game.make(m);
            let zeroing = game.halfmove == 0;
            let child_id = match self.ids.get(&identity_of_game(game)) {
                Some(&known) => known,
                None => {
                    let mut child = game.snapshot();
                    normalise(&mut child);
                    self.intern(&child)?
                }
            };
            game.unmake();
            self.edges.push(child_id | if zeroing { ZEROING } else { 0 });
        }
        self.edge_lo[id as usize] = lo as u64;
        self.edge_len[id as usize] = moves.len() as u32;
        self.moves = moves;
        Ok(())
    }

    fn edges_of(&self, id: usize) -> &[u32] {
        let lo = self.edge_lo[id] as usize;
        &self.edges[lo..lo + self.edge_len[id] as usize]
    }

    /// A resolved child's value from the parent's side: `WON`, `LOST`, or `UNKNOWN`. An illegal
    /// position is the core's win for its side to move, and is read as one.
    fn seen_from(&self, parent: usize, child: usize) -> u8 {
        let value = match self.value[child] {
            ILLEGAL => WON,
            WON => WON,
            LOST => LOST,
            _ => return UNKNOWN,
        };
        if self.side[child] == self.side[parent] {
            value
        } else if value == WON {
            LOST
        } else {
            WON
        }
    }

    /// Distances to the end, in sweeps over the stored edges: at sweep k a position is won in k
    /// if a child resolved before this sweep loses for its mover (or wins, when the same side
    /// moves again), and lost in k when every child is resolved as a win for the other side.
    fn propagate(&mut self) -> u32 {
        let mut open: Vec<u32> = (0..self.value.len() as u32).filter(|id| self.value[*id as usize] == UNKNOWN).collect();
        let mut sweep = 0;
        loop {
            sweep += 1;
            let mut still = Vec::with_capacity(open.len());
            for &id in &open {
                let id = id as usize;
                let mut win = false;
                let mut all = true;
                let mut far = 0u32;
                for &edge in self.edges_of(id) {
                    let child = (edge & ID_MASK) as usize;
                    let seen = self.seen_from(id, child);
                    if seen == UNKNOWN || self.dist[child] >= sweep {
                        all = false;
                        continue;
                    }
                    if seen == WON {
                        win = true;
                        break;
                    }
                    far = far.max(self.dist[child]);
                }
                if win {
                    self.value[id] = WON;
                    self.dist[id] = sweep;
                } else if all {
                    self.value[id] = LOST;
                    self.dist[id] = far + 1;
                } else {
                    still.push(id as u32);
                }
            }
            let done = still.len() == open.len();
            open = still;
            if done {
                return sweep;
            }
        }
    }

    /// The distance to the end or to the next move that resets the no-progress clock, over the
    /// decided positions, the winner taking the shortest and the loser the longest: the same
    /// sweeps, a resetting move counting as one ply to a fresh clock.
    fn to_reset(&self) -> Vec<u32> {
        let unset = u32::MAX;
        let mut reset = vec![unset; self.value.len()];
        let mut open = Vec::new();
        for (id, reset) in reset.iter_mut().enumerate() {
            match self.value[id] {
                WON | LOST if self.dist[id] > 0 => open.push(id),
                WON | LOST | ILLEGAL => *reset = 0,
                _ => {}
            }
        }
        let mut sweep = 0;
        loop {
            sweep += 1;
            let mut still = Vec::with_capacity(open.len());
            for &id in &open {
                let mut settled = None;
                if self.value[id] == WON {
                    for &edge in self.edges_of(id) {
                        let child = (edge & ID_MASK) as usize;
                        if self.seen_from(id, child) == WON && (edge & ZEROING != 0 || reset[child] < sweep) {
                            settled = Some(sweep);
                            break;
                        }
                    }
                } else {
                    let mut far = 0;
                    let mut all = true;
                    for &edge in self.edges_of(id) {
                        let child = (edge & ID_MASK) as usize;
                        if edge & ZEROING != 0 {
                            continue;
                        }
                        if reset[child] < sweep {
                            far = far.max(reset[child]);
                        } else {
                            all = false;
                            break;
                        }
                    }
                    if all {
                        settled = Some(far + 1);
                    }
                }
                match settled {
                    Some(distance) => reset[id] = distance,
                    None => still.push(id),
                }
            }
            let done = still.len() == open.len();
            open = still;
            if done {
                return reset;
            }
        }
    }
}

/// Solve `requested` and its closure. `build` makes a position (`lib.rs`), `cap` bounds the
/// positions (at most `MAX_POSITIONS`), and every position of the requested set decided at
/// `sample_plies` or fewer comes back as a sample.
pub fn solve(t: &Tables, requested: &Material, build: &Build<'_>, cap: u64, sample_plies: u32) -> Result<Solve, String> {
    if let Some(why) = refusal(t) {
        return Err(format!("this game cannot be solved here: {why}"));
    }
    let cap = cap.min(MAX_POSITIONS);
    let n = t.n_squares;
    // The fixed board: the portal ends, which stand where the opening puts them.
    let mut fixed = [0u8; MAX_SQUARES];
    for (fixed, &value) in fixed.iter_mut().zip(&t.initial_board).take(n) {
        if t.is_portal_end(value) {
            *fixed = value;
        }
    }
    let mut placements = Vec::new();
    let mut wanted: Material = [Vec::new(), Vec::new(), Vec::new()];
    for (owner, types) in requested.iter().enumerate() {
        let mut types = types.clone();
        types.sort_unstable();
        for &piece_type in &types {
            if piece_type >= t.n_types {
                return Err(format!("piece type {piece_type} is not one of this game's {} types", t.n_types));
            }
            let value = value_for(t, owner, piece_type);
            if owner == WILD && (t.is_wild_piece(value) || t.kind_of(value) == CONTROL_DUCK || t.is_portal_end(value)) {
                return Err(format!("piece type {piece_type} is a wild piece, the duck or a portal end, never part of a set"));
            }
            let squares = standing_squares(t, owner, value, &fixed);
            if squares.is_empty() {
                return Err(format!("piece type {piece_type} of owner {owner} has no square it can stand on"));
            }
            placements.push(Placement { value, squares });
        }
        wanted[owner] = types;
    }
    if placements.is_empty() {
        return Err("a material set needs at least one piece".into());
    }
    // An upper bound on the starting positions: both sides to move, each run of identical pieces
    // choosing its squares as a set.
    let mut estimate = 2.0f64;
    let mut run = 0usize;
    for (index, piece) in placements.iter().enumerate() {
        run = if index > 0 && placements[index - 1].value == piece.value { run + 1 } else { 0 };
        estimate *= (piece.squares.len() - run.min(piece.squares.len())) as f64 / (run + 1) as f64;
    }
    if estimate > cap as f64 {
        return Err(format!(
            "this set has up to {estimate:.2e} starting positions, over the cap of {cap} (endgame::MAX_POSITIONS is {MAX_POSITIONS})"
        ));
    }
    let walk = Walk { t, build, placements, fixed };
    let mut solver = Solver::new(t, cap);
    walk.each_board(&mut |board| {
        for side in 0..2 {
            solver.add_start(&walk.start(board, side))?;
        }
        Ok(())
    })?;
    let sweeps = solver.propagate();
    let reset = solver.to_reset();

    let clock = t.no_progress_draw;
    let mut sets: Vec<SetStats> = solver
        .groups
        .iter()
        .map(|material| SetStats { material: material.clone(), requested: *material == wanted, ..SetStats::default() })
        .collect();
    if !sets.iter().any(|set| set.requested) {
        sets.insert(0, SetStats { material: wanted.clone(), requested: true, ..SetStats::default() });
    }
    if let Some(asked) = sets.iter_mut().find(|set| set.requested) {
        asked.illegal = solver.impossible;
    }
    let shift = usize::from(!solver.groups.contains(&wanted));
    for id in 0..solver.value.len() {
        let stats = &mut sets[solver.group[id] as usize + shift];
        match solver.value[id] {
            ILLEGAL => stats.illegal += 1,
            WON => {
                stats.won += 1;
                stats.longest_win = stats.longest_win.max(solver.dist[id]);
                stats.longest_to_reset = stats.longest_to_reset.max(reset[id]);
                if reset[id] > clock {
                    stats.beyond_clock += 1;
                }
            }
            LOST => {
                stats.lost += 1;
                stats.longest_loss = stats.longest_loss.max(solver.dist[id]);
            }
            _ => stats.drawn += 1,
        }
    }

    let mut samples = Vec::new();
    if sample_plies > 0 {
        walk.each_board(&mut |board| {
            for side in 0..2 {
                let start = walk.start(board, side);
                let Some(&id) = solver.ids.get(&identity(&start)) else {
                    continue;
                };
                let (value, distance) = (solver.value[id as usize], solver.dist[id as usize]);
                if (value == WON || value == LOST) && distance >= 1 && distance <= sample_plies {
                    samples.push(Sample { snapshot: start, value: if value == WON { 1 } else { -1 }, distance });
                }
            }
            Ok(())
        })?;
    }
    Ok(Solve {
        sets,
        positions: solver.value.len() as u64,
        edges: solver.edges.len() as u64,
        sweeps,
        samples,
    })
}
