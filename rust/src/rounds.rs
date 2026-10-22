//! Simultaneous moves (schema 26, Q918a): a round of two secret moves, resolved together.
//!
//! `docs/reference/rounds.md` is the rule (§2, §4); this is its build
//! (`docs/archive/plan-simultaneous.md` §6.1). A round is a new path beside
//! `make`, never inside it, so the alternating game's hot path does not move. `make_round`
//! reads both moves off the round's board, settles which moves are void from the two
//! choices alone (§2.4), then writes the round in the rule's order (§4): departures, each
//! move's own consequences, the meetings, what the captures cause, the explosions, the
//! claims, the wild pieces, the void run and its bars. `unmake_round` puts back a snapshot:
//! the simpler first form §6.1 names, an incremental undo being an optimisation for after
//! it is right.
//!
//! `tests/simultaneous_oracle.py` is the declared oracle, and resolves a round the same way
//! from the compiled tables and the two move words.

use super::*;

/// One round's undo: everything `make_round` may write, as it was before. A snapshot,
/// not a delta (§6.1).
pub(super) struct RoundUndo {
    board: [u8; MAX_SQUARES],
    occupied: [SquareSet; 3],
    unmoved: SquareSet,
    halfmove: u32,
    ply: u32,
    key: u64,
    window: usize,
    history_len: usize,
    hands: [Vec<u32>; 2],
    colour: Vec<u8>,
    colour_counts: [u32; 2],
    points: [f64; 2],
    goal_left_by: i8,
    initiative: i8,
    void_run: [i16; 2],
    bars: Vec<[i16; 2]>,
    resting: RestLists,
    npc_log: usize,
}

/// One player's move of a round, read off the round's board.
#[derive(Clone, Copy)]
struct Plan {
    m: u32,
    kind: u32,
    /// `None` for a drop, which has no origin.
    origin: Option<usize>,
    target: usize,
    /// The piece as it stands before the move, or a drop's arriving value.
    mover: u8,
    /// Its type before the move (the dropped type for a drop): "a capture by P" means
    /// the piece that set out, as `make` reads it.
    piece_type: usize,
}

/// A capture the round's meetings decided (§2.3, §4 step 4), for what captures cause.
#[derive(Clone, Copy)]
struct Taking {
    captor: usize,
    captor_type: usize,
    victim: u8,
    square: usize,
    origin: Option<usize>,
    /// Whether the strike rule (or an enemy contact link) speaks for this capture: always
    /// for a capture of a standing piece, and for a collision's only under `rules.strike`.
    strikes: bool,
    /// The captor was itself eliminated (a *yes/yes* tie under the default): it captured,
    /// but there is no captor left to transform (§2.4, last bullet).
    captor_gone: bool,
}

/// `Game::round_log` events, for the page to draw a round (§6.1, *The boundary*).
/// `(event, square, player)`:
pub const EVENT_COLLISION: u8 = 0; // square of the meeting, the winner or -1 for none
pub const EVENT_VOID: u8 = 1; // the void piece's square, its player
pub const EVENT_INITIATIVE: u8 = 2; // the meeting a tie the initiative decided, the holder
pub const EVENT_BARRED: u8 = 3; // the barred pair's player-0 square, -1
/// Schema 27 (Q942a, `swap: "collision"`): two pieces that swapped squares met. The square
/// is player 0's destination (the other's origin), the player the winner or -1 for none.
pub const EVENT_SWAP: u8 = 4;
/// Schema 28 (Q943a, `docs/plan-speed.md` §4.2): the two movers' speeds differed and the
/// round was two moves in order. The square is the faster move's destination, the player
/// the one who moved first.
pub const EVENT_ORDER: u8 = 5;
/// Schema 28: the slower move was lost, its piece taken, moved or changed by the faster
/// move. The square is the slower move's origin, the player its player.
pub const EVENT_LOST: u8 = 6;
/// Schema 28: the faster move ended the game, so the slower was never played. The square
/// is the slower move's origin (a drop's destination), the player its player. A slower
/// move made void by the board the faster left is an `EVENT_VOID` (its origin, or a
/// drop's destination).
pub const EVENT_UNPLAYED: u8 = 7;

/// What a round's steps wrote that its tail reads (§4 steps 8 to 10): one for the round,
/// filled by one step at equal speeds and by two in an ordered round (schema 28).
struct StepFacts {
    touched: SquareSet,
    material_changed: bool,
    promoted: bool,
    one_way: bool,
    irreversible_kind: bool,
    initiative_passed: bool,
    laid_permanent: bool,
    key_extra: u64,
}

impl StepFacts {
    fn new() -> Self {
        StepFacts {
            touched: SquareSet::EMPTY,
            material_changed: false,
            promoted: false,
            one_way: false,
            irreversible_kind: false,
            initiative_passed: false,
            laid_permanent: false,
            key_extra: 0,
        }
    }
}

/// Each player's move after one step: arrived, stood (a turn, a contact that stays), void.
struct Step {
    arriving: [bool; 2],
    standing: [bool; 2],
    void: [bool; 2],
}

impl<'t> Game<'t> {
    fn sim(&self) -> &'t SimTables {
        self.t.simultaneous.as_deref().expect("a round in an alternating game")
    }

    /// Every move `player` may make on this board: the alternating generator for that
    /// player, with no royal-safety filter (there is no check, §2.5), less the barred
    /// capture-only moves (§2.7) and, under a rest (schema 29, `pieces[].rest`), the resting
    /// pieces' moves unless they are all there is (the lift, `rest_filter`), before
    /// `restrict` so `must_capture` sees fewer; and made again without the rest when
    /// `restrict` empties a list the rest shortened (`docs/plan-rest.md` §3.7).
    pub fn generate_legal_for(&mut self, player: usize, out: &mut Vec<u32>) {
        debug_assert!(!self.t.needs_check_filter(), "a simultaneous game has no check");
        let saved = self.side;
        self.side = player;
        let base = out.len();
        self.generate_unbarred(player, out, base);
        let rested = self.t.rest.is_some() && self.rest_filter(player, out, base);
        self.restrict(out, base);
        if rested && out.len() == base {
            self.regenerate_unbarred(player, out, base);
        }
        self.side = saved;
    }

    /// The list made again without the rest (§3.7): rare, so out of line, which keeps a
    /// second copy of the generator out of `generate_legal_for` (Q983a's benches).
    #[cold]
    #[inline(never)]
    fn regenerate_unbarred(&mut self, player: usize, out: &mut Vec<u32>, base: usize) {
        self.generate_unbarred(player, out, base);
        self.restrict(out, base);
    }

    /// `player`'s pseudo-legal moves less the barred ones, appended to `out` from `base`;
    /// `side` is already `player`.
    fn generate_unbarred(&mut self, player: usize, out: &mut Vec<u32>, base: usize) {
        self.generate_pseudo_legal(out);
        if !self.bars.is_empty() {
            let mut write = base;
            for index in base..out.len() {
                let m = out[index];
                if !self.barred(player, m) {
                    out[write] = m;
                    write += 1;
                }
            }
            out.truncate(write);
        }
    }

    /// Whether `m` is one of a barred pair's capture-only moves for `player` (§2.7).
    fn barred(&mut self, player: usize, m: u32) -> bool {
        if move_kind(m) == MOVE_DROP {
            return false;
        }
        let origin = move_from(m) as i16;
        let target = move_to(m) as i16;
        let hit = self.bars.iter().any(|bar| bar[player] == origin && bar[1 - player] == target);
        hit && !self.lands_if_empty(player, m)
    }

    /// Whether `m`'s origin and destination would be offered with the destination empty
    /// (§5.3, `lands_if_empty`): the move could land on an empty square, so it survives
    /// its victim's flight. Asked of the movement tables by generating `player`'s moves with
    /// the one square emptied, which also finds a second pattern onto the same square (a
    /// capture-only leap beside a quiet slide); a bit per table entry would miss it. Rare:
    /// only a chase or a barred pair asks.
    fn lands_if_empty(&mut self, player: usize, m: u32) -> bool {
        let origin = move_from(m);
        let target = move_to(m);
        let standing = self.board[target];
        let saved = self.side;
        self.put(target, 0);
        self.side = player;
        let mut moves = Vec::new();
        // The movement tables alone (schema 30, `docs/plan-passive.md` §2.8): the move was
        // already offered under the round's effects, and emptying a square can change them.
        self.generate_movement(&mut moves);
        self.side = saved;
        self.put(target, standing);
        moves.iter().any(|&other| {
            let kind = move_kind(other);
            kind != MOVE_DROP && kind != MOVE_ROTATE && move_from(other) == origin
                && move_to(other) == target
        })
    }

    fn plan(&self, player: usize, m: u32) -> Plan {
        let t = self.t;
        let kind = move_kind(m);
        let target = move_to(m);
        if kind == MOVE_DROP {
            let piece_type = move_promotion(m) as usize;
            return Plan {
                m, kind, origin: None, target,
                mover: t.arriving_value(player, piece_type, target),
                piece_type,
            };
        }
        let origin = move_from(m);
        let mover = self.board[origin];
        Plan { m, kind, origin: Some(origin), target, mover, piece_type: t.type_of(mover) }
    }

    /// What a contact link (or the strike rule) says of `mover_type` meeting `victim`:
    /// `(the victim becomes, the mover becomes, the mover switches sides)`, `-1` for
    /// "leaves the board" and "stays itself".
    fn meeting(&self, mover_type: usize, victim: u8, mover_owner: usize) -> (i32, i32, bool) {
        let Some(strike) = self.t.strike.as_ref() else {
            return (-1, -1, false);
        };
        let owner = self.t.owner_of(victim);
        let cell = mover_type * strike.n_types + self.t.type_of(victim);
        if owner == mover_owner {
            (strike.own_to[cell], strike.own_mover_to[cell], false)
        } else {
            (strike.to[cell], strike.mover_to[cell], owner != WILD && strike.flips(cell))
        }
    }

    /// Whether a move is a contact that leaves its mover on its origin (`contacts.position:
    /// "stay"`) and fires on what stands on its target on the round's board.
    fn stays_put(&self, plan: &Plan, player: usize) -> bool {
        let Some(strike) = self.t.strike.as_ref() else {
            return false;
        };
        if strike.position != StrikePosition::Stay || plan.origin.is_none() {
            return false;
        }
        let victim = self.board[plan.target];
        if victim == 0 || plan.origin == Some(plan.target) {
            return false;
        }
        let (becomes, mover_becomes, flips) = self.meeting(plan.piece_type, victim, player);
        becomes != -1 || mover_becomes >= 0 || flips
    }

    /// The part of the key a round's standoff state contributes (§5.3, `zobrist_standoff`).
    pub(super) fn standoff_key(&self) -> u64 {
        let Some(sim) = self.t.simultaneous.as_deref() else {
            return 0;
        };
        let n = self.t.n_squares;
        let mut key = 0u64;
        if self.void_run != NO_BAR {
            key ^= sim.standoff_key(n, self.void_run[0] as usize, self.void_run[1] as usize, 1);
        }
        for bar in &self.bars {
            key ^= sim.standoff_key(n, bar[0] as usize, bar[1] as usize, 2);
        }
        if self.initiative == 1 {
            key ^= sim.zobrist_initiative;
        }
        key
    }

    /// Play one round: `moves[0]` by player 0 and `moves[1]` by player 1, each legal for
    /// its player on this board (`generate_legal_for`), resolved together (§2, §4).
    ///
    /// Schema 28 (Q943a, `docs/plan-speed.md` §2.4): when the two movers' speeds differ, the
    /// round is two moves in order instead. The faster is resolved alone, with every piece
    /// of the slower player standing; the endings are read; then the slower is resolved
    /// alone on the board the faster left, if its piece still stands unchanged on its origin
    /// (else it is *lost*) and its word is still offered by its generator (else it is
    /// *void*). Both steps are `resolve_step`, the one body every round goes through, so an
    /// equal-speed round (every round of a game without speeds) is the round it always was.
    pub fn make_round(&mut self, moves: [u32; 2]) {
        let t = self.t;
        let sim = self.sim();
        debug_assert_eq!(sim.void_run, 2, "the run is kept as one pair at count one");
        self.round_undo.push(RoundUndo {
            board: self.board,
            occupied: self.occupied,
            unmoved: self.unmoved,
            halfmove: self.halfmove,
            ply: self.ply,
            key: self.key,
            window: self.window,
            history_len: self.history.len(),
            hands: self.hands.clone(),
            colour: self.colour.clone(),
            colour_counts: self.colour_counts,
            points: self.points,
            goal_left_by: self.goal_left_by,
            initiative: self.initiative,
            void_run: self.void_run,
            bars: self.bars.clone(),
            resting: self.resting,
            npc_log: self.npc_log.len(),
        });
        self.round_log.clear();
        let standoff_before = self.standoff_key();
        self.working_key = self.key;
        let previous_unmoved = self.unmoved;
        let npc_mark = self.npc_log.len();

        let plans = [self.plan(0, moves[0]), self.plan(1, moves[1])];
        let mut facts = StepFacts::new();
        // Schema 28: one emptiness test in every game without speeds.
        let faster = if sim.speed.is_empty() {
            None
        } else {
            let speeds = [sim.speed[plans[0].piece_type], sim.speed[plans[1].piece_type]];
            (speeds[0] != speeds[1]).then(|| usize::from(speeds[1] > speeds[0]))
        };
        // Each player's move, as the tail reads it: arrived, stood (a turn or a contact that
        // stays), void, and -- in an ordered round only -- gone (lost or unplayed).
        let mut arriving = [false; 2];
        let mut standing = [false; 2];
        let mut void = [false; 2];
        let mut gone = [false; 2];
        let mut ended = false;
        match faster {
            None => {
                let step = self.resolve_step(&plans, [true, true], &mut facts);
                arriving = step.arriving;
                standing = step.standing;
                void = step.void;
            }
            Some(f) => {
                let s = 1 - f;
                self.round_log.push((EVENT_ORDER, plans[f].target, f as i8));
                let mut active = [false; 2];
                active[f] = true;
                let first = self.resolve_step(&plans, active, &mut facts);
                arriving[f] = first.arriving[f];
                standing[f] = first.standing[f];
                // §2.6: the endings, read after the faster move; one that fires ends the
                // round there and the slower move is never played.
                if t.reach_leaves {
                    self.goal_left_by = self.goal_left(&plans, arriving);
                }
                self.check_memo.set([-1, -1]);
                ended = self.round_goal_ending().is_some();
                let plan = plans[s];
                let square = plan.origin.unwrap_or(plan.target);
                if ended {
                    gone[s] = true;
                    self.round_log.push((EVENT_UNPLAYED, square, s as i8));
                } else if plan
                    .origin
                    .is_some_and(|o| facts.touched.get(o) || self.board[o] != plan.mover)
                {
                    // §2.5 F: the piece that chose is no longer there as itself.
                    gone[s] = true;
                    self.round_log.push((EVENT_LOST, square, s as i8));
                } else if !self.offered(s, plan.m) {
                    // §2.5 A to H: the word is not one the piece could make on this board.
                    // A single void move, never part of a void run (§2.8).
                    self.round_log.push((EVENT_VOID, square, s as i8));
                } else {
                    let mut active = [false; 2];
                    active[s] = true;
                    let second = self.resolve_step(&plans, active, &mut facts);
                    arriving[s] = second.arriving[s];
                    standing[s] = second.standing[s];
                }
            }
        }
        let StepFacts {
            mut touched,
            mut material_changed,
            promoted,
            one_way,
            irreversible_kind,
            initiative_passed,
            laid_permanent,
            key_extra,
        } = facts;

        // §4 step 8: the wild pieces, once a round -- not when the round ended after the
        // faster move (§2.6).
        if !ended && t.has_npcs && self.npc_phase() {
            material_changed = true;
        }
        for &(from, to, _) in &self.npc_log[npc_mark..] {
            touched.set(from);
            touched.set(to);
        }

        // Schema 29 (Q983a, `pieces[].rest`): the round is a turn of each player, so each
        // player's rests count down and the piece that made its move rests where it ended the
        // round (its origin if it stood, turned or was void), if that square still holds that
        // player's piece; a piece eliminated, eaten or turned over rests nowhere, and nor does
        // a slower move lost or unplayed (schema 28).
        let mut rest_key = 0u64;
        if t.rest.is_some() {
            for i in 0..2 {
                let end = if gone[i] {
                    None
                } else if arriving[i] {
                    Some(plans[i].target)
                } else {
                    plans[i].origin
                };
                rest_key ^= self.rest_after_turn(i, end);
            }
        }

        // §4 step 9: the void run and the bars (§2.7). Only an equal-speed round can be
        // void for both.
        let mut run = NO_BAR;
        if void[0] && void[1] {
            let pair = [
                plans[0].origin.expect("a void move has an origin") as i16,
                plans[1].origin.expect("a void move has an origin") as i16,
            ];
            if self.void_run == pair {
                if !self.bars.contains(&pair) {
                    self.bars.push(pair);
                    self.bars.sort_unstable();
                }
                self.round_log.push((EVENT_BARRED, pair[0] as usize, -1));
            } else {
                run = pair;
            }
        }
        if run != NO_BAR && (touched.get(run[0] as usize) || touched.get(run[1] as usize)) {
            run = NO_BAR;
        }
        self.void_run = run;
        self.bars.retain(|bar| !touched.get(bar[0] as usize) && !touched.get(bar[1] as usize));

        // §4 step 10's bookkeeping: rights, the leaving goal, the clocks, the key.
        for i in 0..2 {
            if arriving[i] || standing[i] {
                if let Some(origin) = plans[i].origin {
                    self.unmoved.clear(origin);
                }
                self.unmoved.clear(plans[i].target);
            }
        }
        if t.reach_leaves {
            self.goal_left_by = self.goal_left(&plans, arriving);
        }
        let rights_changed = self.unmoved != previous_unmoved;
        let progress = material_changed || promoted || one_way || laid_permanent;
        let irreversible =
            progress || irreversible_kind || rights_changed || initiative_passed;
        self.halfmove = if progress { 0 } else { self.halfmove + 2 };
        self.ply += 2;
        self.key =
            self.working_key ^ key_extra ^ standoff_before ^ self.standoff_key() ^ rest_key;
        self.history.push(self.key);
        if irreversible {
            self.window = self.history.len() - 1;
        }
        self.check_memo.set([-1, -1]);
    }

    /// Who left a goal this round (`reach.on: "leave"`): a move that arrived from a goal
    /// square of its player, by the counted type, whose piece still stands where it went.
    /// `goal_left_by`'s values: 2 both, 0 or 1 that player, -1 neither.
    fn goal_left(&self, plans: &[Plan; 2], arriving: [bool; 2]) -> i8 {
        let t = self.t;
        let reach = t.reach.as_ref().expect("reach_leaves without a reach rule");
        let mut left = [false; 2];
        for i in 0..2 {
            let Some(origin) = plans[i].origin else { continue };
            let value = self.board[plans[i].target];
            left[i] = arriving[i]
                && reach.squares[i][origin]
                && (reach.piece_type < 0 || plans[i].piece_type == reach.piece_type as usize)
                && value != 0
                && t.owner_of(value) == i;
        }
        match left {
            [true, true] => 2,
            [true, false] => 0,
            [false, true] => 1,
            _ => -1,
        }
    }

    /// Schema 28 (§2.5): whether `player`'s generator offers the word `m` on this board --
    /// the movement tables, blocks, screens, modes, `captures`, regions, conditions, drop
    /// regions and the promotion offer -- without the list-level filters (`must_capture`,
    /// the bars, the rest, `forbid_immobility`), which were asked on the round's board.
    fn offered(&mut self, player: usize, m: u32) -> bool {
        let saved = self.side;
        self.side = player;
        let mut moves = Vec::new();
        self.generate_pseudo_legal(&mut moves);
        self.side = saved;
        moves.contains(&m)
    }

    /// §2.2 to §4 step 7 for the players `active` names: both in an equal-speed round, one
    /// at a time in an ordered one (schema 28), where the other player's pieces all stand.
    /// Writes the board and accumulates what the round's tail needs into `facts`.
    #[inline(always)]
    fn resolve_step(&mut self, plans: &[Plan; 2], active: [bool; 2], facts: &mut StepFacts) -> Step {
        let t = self.t;
        let sim = self.sim();
        let touched = &mut facts.touched;

        // §2.2, §2.4: who moves, who stands, which moves are void -- from the two choices.
        let raw = [
            active[0] && plans[0].origin != Some(plans[0].target),
            active[1] && plans[1].origin != Some(plans[1].target),
        ];
        let stays = [
            active[0] && self.stays_put(&plans[0], 0),
            active[1] && self.stays_put(&plans[1], 1),
        ];
        let mut standing = [false; 2];
        for i in 0..2 {
            let j = 1 - i;
            let own_target = stays[i] && t.owner_of(self.board[plans[i].target]) == i;
            let vacated = raw[j] && !stays[j] && plans[j].origin == Some(plans[i].target);
            standing[i] =
                active[i] && (plans[i].kind == MOVE_ROTATE || (stays[i] && (own_target || !vacated)));
        }
        let moving = [raw[0] && !standing[0], raw[1] && !standing[1]];
        // Schema 27 (Q942a, `swap: "collision"`): two moves each ending on the other's
        // origin meet, whatever their modes, so neither is void (`docs/archive/plan-round-chase.md` R2).
        let swapped = sim.swap_collides
            && moving[0]
            && moving[1]
            && plans[0].origin.is_some()
            && plans[1].origin == Some(plans[0].target)
            && plans[0].origin == Some(plans[1].target);
        let mut void = [false; 2];
        for i in 0..2 {
            let j = 1 - i;
            void[i] = !swapped
                && moving[i]
                && moving[j]
                && plans[i].origin.is_some()
                && plans[j].origin == Some(plans[i].target)
                && !self.lands_if_empty(i, plans[i].m);
            if void[i] {
                self.round_log.push((EVENT_VOID, plans[i].origin.unwrap_or(0), i as i8));
            }
        }
        let arriving = [moving[0] && !void[0], moving[1] && !void[1]];

        for i in 0..2 {
            let made = arriving[i] || standing[i];
            if made {
                facts.one_way |= t.is_one_way(plans[i].piece_type);
                let kind = plans[i].kind;
                let plain_portal = kind == MOVE_PORTAL && move_promotion(plans[i].m) == 0;
                facts.irreversible_kind |= kind != MOVE_NORMAL && kind != MOVE_ROTATE && !plain_portal;
                facts.promoted |= kind == MOVE_PROMOTION
                    || (kind == MOVE_PORTAL && move_promotion(plans[i].m) != 0);
            }
        }

        // §4 step 2: departures.
        for i in 0..2 {
            if !arriving[i] {
                continue;
            }
            match plans[i].origin {
                Some(origin) => {
                    self.write(origin, 0);
                    touched.set(origin);
                }
                None => {
                    let rule = t.drops.as_ref().expect("a drop without a drop rule");
                    let slot = rule.slot_of(plans[i].piece_type);
                    if !rule.is_unlimited(slot) {
                        self.hands[i][slot] -= 1;
                    }
                }
            }
        }

        // §4 step 3: each move's own consequences, as if alone, up to the meeting.
        let mut arrival = [0u8; 2];
        for i in 0..2 {
            if !active[i] {
                continue;
            }
            let plan = plans[i];
            if plan.kind == MOVE_ROTATE {
                let origin = plan.origin.expect("a rotation has an origin");
                self.write(origin, t.piece_value(i, move_promotion(plan.m) as usize, plan.piece_type));
                touched.set(origin);
                continue;
            }
            if standing[i] {
                let there = self.board[plan.target];
                if there != 0 && t.owner_of(there) == i {
                    facts.material_changed |= self.contact_in_place(i, &plan, touched);
                }
                continue;
            }
            if !arriving[i] {
                continue;
            }
            let promotion = move_promotion(plan.m) as usize;
            let mut value = match plan.kind {
                MOVE_PROMOTION => t.entering_value(i, promotion),
                MOVE_PORTAL if promotion != 0 => t.entering_value(i, promotion - 1),
                _ => plan.mover,
            };
            let occupant = self.board[plan.target];
            if occupant != 0 && t.owner_of(occupant) == i {
                // An own merge under `advance` (§4, contacts `own`): the target leaves, the
                // mover may change, and the merged piece is the one that meets.
                let (_, mover_becomes, _) = self.meeting(plan.piece_type, occupant, i);
                if mover_becomes >= 0 {
                    value = t.arriving_value(i, mover_becomes as usize, plan.target);
                }
                self.write(plan.target, 0);
                touched.set(plan.target);
                facts.material_changed = true;
            }
            arrival[i] = value;
        }

        // §4 step 4: the meetings. A collision first, then each other arrival, then the
        // contacts whose movers stood still.
        let mut takings: Vec<Taking> = Vec::new();
        let collided = arriving[0] && arriving[1] && plans[0].target == plans[1].target;
        if collided {
            let square = plans[0].target;
            touched.set(square);
            facts.material_changed = true;
            let types = [t.type_of(arrival[0]), t.type_of(arrival[1])];
            let beats = [
                sim.beats(t.n_types, types[0], types[1]),
                sim.beats(t.n_types, types[1], types[0]),
            ];
            let winner = match beats {
                [true, false] => Some(0),
                [false, true] => Some(1),
                _ if sim.tie == 1 => {
                    let holder = self.initiative as usize;
                    self.initiative = (1 - holder) as i8;
                    facts.initiative_passed = true;
                    self.round_log.push((EVENT_INITIATIVE, square, holder as i8));
                    Some(holder)
                }
                _ => None,
            };
            // A standing piece on the meeting square can only be a wild one: the winner
            // takes it; with no winner it goes with the two.
            let standing_there = self.board[square];
            match winner {
                Some(w) => {
                    if standing_there != 0 {
                        takings.push(self.taking(w, &plans[w], standing_there, square, true));
                    }
                    self.write(square, arrival[w]);
                    let mut taken = self.taking(w, &plans[w], arrival[1 - w], square, false);
                    taken.strikes = sim.collision_strikes;
                    takings.push(taken);
                }
                None => {
                    self.write(square, 0);
                    if beats[0] && beats[1] {
                        for w in 0..2 {
                            let mut taken = self.taking(w, &plans[w], arrival[1 - w], square, false);
                            taken.strikes = sim.collision_strikes;
                            taken.captor_gone = true;
                            takings.push(taken);
                        }
                    }
                }
            }
            self.round_log.push((EVENT_COLLISION, square, winner.map_or(-1, |w| w as i8)));
        } else if swapped {
            // Both destinations were emptied by the departures; each piece is taken where it
            // would have landed, on the other's origin.
            facts.material_changed = true;
            touched.set(plans[0].target);
            touched.set(plans[1].target);
            let types = [t.type_of(arrival[0]), t.type_of(arrival[1])];
            let beats = [
                sim.beats(t.n_types, types[0], types[1]),
                sim.beats(t.n_types, types[1], types[0]),
            ];
            let winner = match beats {
                [true, false] => Some(0),
                [false, true] => Some(1),
                _ if sim.tie == 1 => {
                    let holder = self.initiative as usize;
                    self.initiative = (1 - holder) as i8;
                    facts.initiative_passed = true;
                    self.round_log.push((EVENT_INITIATIVE, plans[holder].target, holder as i8));
                    Some(holder)
                }
                _ => None,
            };
            match winner {
                Some(w) => {
                    let square = plans[w].target;
                    self.write(square, arrival[w]);
                    let mut taken = self.taking(w, &plans[w], arrival[1 - w], square, false);
                    taken.strikes = sim.collision_strikes;
                    takings.push(taken);
                }
                None if beats[0] && beats[1] => {
                    for w in 0..2 {
                        let square = plans[w].target;
                        let mut taken = self.taking(w, &plans[w], arrival[1 - w], square, false);
                        taken.strikes = sim.collision_strikes;
                        taken.captor_gone = true;
                        takings.push(taken);
                    }
                }
                None => {}
            }
            self.round_log.push((EVENT_SWAP, plans[0].target, winner.map_or(-1, |w| w as i8)));
        } else {
            for i in 0..2 {
                if !arriving[i] {
                    continue;
                }
                let square = plans[i].target;
                touched.set(square);
                let occupant = self.board[square];
                debug_assert!(occupant == 0 || t.owner_of(occupant) != i, "an arrival on its own piece");
                if occupant != 0 {
                    facts.material_changed = true;
                    takings.push(self.taking(i, &plans[i], occupant, square, true));
                }
                self.write(square, arrival[i]);
            }
        }
        for i in 0..2 {
            let plan = plans[i];
            if !standing[i] || plan.kind == MOVE_ROTATE {
                continue;
            }
            let victim = self.board[plan.target];
            let origin = plan.origin.expect("a contact that stays has an origin");
            if victim == 0 || t.owner_of(victim) == i || self.board[origin] != plan.mover {
                continue;
            }
            facts.material_changed |= self.contact_in_place(i, &plan, touched);
        }

        // §4 step 5: what the captures cause.
        for taken in &takings {
            if let Some(rule) = t.drops.as_ref() {
                if rule.from_captures && t.owner_of(taken.victim) != taken.captor {
                    let slot = t.demotions[t.type_of(taken.victim)];
                    if !rule.is_unlimited(slot) {
                        self.hands[taken.captor][slot] += 1;
                    }
                }
            }
            if t.has_npcs && t.owner_of(taken.victim) == WILD {
                self.points[taken.captor] += t.npc_bounty[taken.square];
            }
            if taken.strikes {
                self.strike_after(taken, touched);
            }
        }

        // §4 step 6: every explosion at once.
        if let Some(explosion) = t.explosion.as_ref() {
            let mut centres = SquareSet::EMPTY;
            let mut blast = SquareSet::EMPTY;
            for taken in &takings {
                if explosion.fires(taken.captor_type, t.type_of(taken.victim)) {
                    centres.set(taken.square);
                    for &square in &explosion.blast[taken.square] {
                        blast.set(square);
                    }
                }
            }
            for square in 0..t.n_squares {
                let value = self.board[square];
                if value == 0 {
                    continue;
                }
                let hit = centres.get(square)
                    || (blast.get(square) && !explosion.immune_types[t.type_of(value)]);
                if hit {
                    self.write(square, 0);
                    touched.set(square);
                    facts.material_changed = true;
                }
            }
        }

        // §4 step 7: claims, by the pieces that ended a move on a square and stand there.
        if let Some(rule) = t.territory.as_ref() {
            for i in 0..2 {
                let square = if active[i] && plans[i].kind == MOVE_ROTATE {
                    plans[i].origin.expect("a rotation has an origin")
                } else if arriving[i] {
                    plans[i].target
                } else {
                    continue;
                };
                let value = self.board[square];
                if value == 0 || t.owner_of(value) != i {
                    continue;
                }
                let want = (i + 1) as u8;
                let before = self.colour[square];
                if before != want && (rule.repaints || before == 0) {
                    self.colour[square] = want;
                    self.colour_counts[i] += 1;
                    if before != 0 {
                        self.colour_counts[before as usize - 1] -= 1;
                    }
                    if rule.repaints {
                        facts.key_extra ^= t.zobrist_colour(before, square) ^ t.zobrist_colour(want, square);
                    } else {
                        facts.laid_permanent = true;
                    }
                }
            }
        }

        Step { arriving, standing, void }
    }

    /// Take back the last `make_round`.
    pub fn unmake_round(&mut self) {
        let undo = self.round_undo.pop().expect("unmake_round without a matching make_round");
        self.board = undo.board;
        self.occupied = undo.occupied;
        self.unmoved = undo.unmoved;
        self.halfmove = undo.halfmove;
        self.ply = undo.ply;
        self.key = undo.key;
        self.window = undo.window;
        self.history.truncate(undo.history_len);
        self.hands = undo.hands;
        self.colour = undo.colour;
        self.colour_counts = undo.colour_counts;
        self.points = undo.points;
        self.goal_left_by = undo.goal_left_by;
        self.initiative = undo.initiative;
        self.void_run = undo.void_run;
        self.bars = undo.bars;
        self.resting = undo.resting;
        self.npc_log.truncate(undo.npc_log);
        self.check_memo.set([-1, -1]);
    }

    fn taking(&self, captor: usize, plan: &Plan, victim: u8, square: usize, strikes: bool) -> Taking {
        Taking {
            captor,
            captor_type: plan.piece_type,
            victim,
            square,
            origin: plan.origin,
            strikes: strikes && self.t.strike.is_some(),
            captor_gone: false,
        }
    }

    /// A contact whose mover stays on its origin (§4, contacts `stay`): the mover and its
    /// target both change where they stand. Returns whether material changed.
    fn contact_in_place(&mut self, player: usize, plan: &Plan, touched: &mut SquareSet) -> bool {
        let t = self.t;
        let origin = plan.origin.expect("a contact that stays has an origin");
        let victim = self.board[plan.target];
        let mover = self.board[origin];
        let (becomes, mover_becomes, flips) = self.meeting(plan.piece_type, victim, player);
        let as_type = if mover_becomes >= 0 { mover_becomes as usize } else { t.type_of(mover) };
        let attacker = if flips {
            t.arriving_value(1 - player, as_type, origin)
        } else if mover_becomes >= 0 {
            t.arriving_value(player, as_type, origin)
        } else {
            mover
        };
        let survivor =
            if becomes >= 0 { t.arriving_value(t.owner_of(victim), becomes as usize, plan.target) } else { 0 };
        self.write(origin, attacker);
        self.write(plan.target, survivor);
        touched.set(origin);
        touched.set(plan.target);
        let worth = |value: u8| if value == 0 { -1.0 } else { t.piece_values[t.type_of(value)] };
        worth(attacker) != worth(mover) || worth(survivor) != worth(victim)
    }

    /// §4 step 5 for one capture under the strike rule or an enemy contact link: the
    /// captor may change or switch sides where it stands, and a surviving victim goes home
    /// (`home`) or to the captor's origin (`swap`) if that square is empty after the
    /// meetings, else it is removed (circe's rule, §4).
    fn strike_after(&mut self, taken: &Taking, touched: &mut SquareSet) {
        let t = self.t;
        let strike = t.strike.as_ref().expect("a strike without a strike rule");
        let (becomes, mover_becomes, flips) = self.meeting(taken.captor_type, taken.victim, taken.captor);
        let standing = self.board[taken.square];
        if !taken.captor_gone
            && standing != 0
            && t.owner_of(standing) == taken.captor
            && (mover_becomes >= 0 || flips)
        {
            let as_type = if mover_becomes >= 0 { mover_becomes as usize } else { t.type_of(standing) };
            let owner = if flips { 1 - taken.captor } else { taken.captor };
            self.write(taken.square, t.arriving_value(owner, as_type, taken.square));
        }
        if becomes < 0 {
            return;
        }
        let owner = t.owner_of(taken.victim);
        let place = match strike.position {
            StrikePosition::Home => {
                let home = strike.rebirth_square(owner, becomes as usize, taken.square);
                (home >= 0).then_some(home as usize)
            }
            StrikePosition::Swap => taken.origin,
            _ => None,
        };
        if let Some(square) = place {
            if self.board[square] == 0 {
                self.write(square, t.arriving_value(owner, becomes as usize, square));
                touched.set(square);
                if strike.position == StrikePosition::Home {
                    self.unmoved.set(square);
                } else {
                    self.unmoved.clear(square);
                }
            }
        }
    }

    /// The endings a simultaneous position declares, before the clocks (§2.6): the
    /// alternating order, with no illegal-position test and every rule that fires for both
    /// players in one round a draw at its place -- or, under schema 27's `both: "score"`
    /// (Q942a, X1), a win for the player ahead under `rules.scoring`, with the ending's own
    /// reason, and a draw only on equal scores.
    pub fn round_ending_before_the_clocks(&self) -> Option<Outcome> {
        if let Some(outcome) = self.round_goal_ending() {
            return Some(outcome);
        }
        if !self.history.is_empty() && self.repetition_count() >= self.t.repetition_draw {
            return Some(Outcome { winner: -1, reason: Reason::Repetition });
        }
        None
    }

    /// The endings of `round_ending_before_the_clocks` that a board fulfils, without
    /// repetition: what an ordered round reads after its faster move (schema 28, §2.6).
    fn round_goal_ending(&self) -> Option<Outcome> {
        let t = self.t;
        let scored = t
            .scoring
            .as_ref()
            .filter(|_| t.simultaneous.as_deref().is_some_and(|sim| sim.both_score));
        let both_fired = |reason: Reason| -> Outcome {
            if let Some(scoring) = scored {
                let first = self.score_for(0, scoring);
                let second = self.score_for(1, scoring);
                if first != second {
                    return Outcome { winner: if first > second { 0 } else { 1 }, reason };
                }
            }
            Outcome { winner: -1, reason }
        };
        let both = |fired: [bool; 2], reason: Reason, result: i8| -> Option<Outcome> {
            match fired {
                [true, true] => Some(both_fired(reason)),
                [true, false] => Some(decide(0, result, reason)),
                [false, true] => Some(decide(1, result, reason)),
                _ => None,
            }
        };
        if !t.royal_types.is_empty() {
            let lost = [0, 1].map(|side| t.royal_loss_ends_it(side) && self.royal_gone(side));
            let reason = match t.win {
                Win::RoyalCapture => Reason::RoyalCaptured,
                _ => Reason::RoyalDestroyed,
            };
            if let Some(outcome) = both(lost, reason, -1) {
                return Some(outcome);
            }
        }
        if t.win == Win::Annihilation {
            let gone = [0, 1].map(|side| !self.has_any_piece(side));
            if let Some(outcome) = both(gone, Reason::Annihilated, -1) {
                return Some(outcome);
            }
        }
        if t.has_v2_endings() {
            if let Some(reach) = t.reach.as_ref() {
                let fired = if reach.leaves {
                    match self.goal_left_by {
                        2 => [true, true],
                        0 => [true, false],
                        1 => [false, true],
                        _ => [false, false],
                    }
                } else {
                    [0, 1].map(|side| {
                        self.occupied()[side].iter().any(|square| {
                            reach.squares[side][square]
                                && (reach.piece_type < 0
                                    || t.type_of(self.board[square]) == reach.piece_type as usize)
                                && !(reach.safe && self.threatened(1 - side, square))
                        })
                    })
                };
                if let Some(outcome) = both(fired, Reason::Reached, reach.result) {
                    return Some(outcome);
                }
            }
            if let Some(extinction) = t.extinction.as_ref() {
                let short = [0, 1].map(|side| {
                    let mine = self.occupied()[side];
                    if extinction.per_type {
                        self.short_of(side, &extinction.side_types[side], extinction.min_count)
                    } else if extinction.any_piece {
                        mine.len() < extinction.min_count
                    } else {
                        mine.iter().filter(|&sq| extinction.counts(t.type_of(self.board[sq]))).count()
                            < extinction.min_count
                    }
                });
                if let Some(outcome) = both(short, Reason::Extinction, extinction.result) {
                    return Some(outcome);
                }
            }
            if let Some(result) = t.bare_royal {
                let bared = [0, 1].map(|side| {
                    let mine = self.occupied()[side];
                    !mine.is_empty() && !mine.iter().any(|sq| !t.is_royal(t.type_of(self.board[sq])))
                });
                if let Some(outcome) = both(bared, Reason::BareRoyal, result) {
                    return Some(outcome);
                }
            }
            if let Some(rule) = t.territory.as_ref().filter(|rule| rule.win_at > 0) {
                let enough = [0, 1].map(|side| self.coloured_by(side) >= rule.win_at as usize);
                if let Some(outcome) = both(enough, Reason::Coloured, 1) {
                    return Some(outcome);
                }
            }
        }
        None
    }

    /// The half of a round's verdict that needs no move lists: the endings, then the
    /// clocks (there is no mate to outrank them).
    pub fn round_terminal_prefix(&self) -> Option<Outcome> {
        if let Some(outcome) = self.round_ending_before_the_clocks() {
            return Some(outcome);
        }
        self.clock_run_out().map(|reason| self.on_the_clock(reason))
    }

    /// The whole verdict from both players' move lists: a player with no legal move at the
    /// start of a round gets `rules.stalemate`'s result; both without one is a draw (§2.6).
    pub fn round_terminal_with(&self, lists: [&[u32]; 2]) -> Option<Outcome> {
        if let Some(outcome) = self.round_terminal_prefix() {
            return Some(outcome);
        }
        let stuck = [lists[0].is_empty(), lists[1].is_empty()];
        let player = match stuck {
            [true, true] => return Some(Outcome { winner: -1, reason: Reason::Stalemate }),
            [true, false] => 0,
            [false, true] => 1,
            _ => return None,
        };
        Some(match self.t.stalemate {
            Stalemate::Draw => Outcome { winner: -1, reason: Reason::Stalemate },
            Stalemate::Loss => Outcome { winner: (1 - player) as i8, reason: Reason::StalemateLoss },
            Stalemate::Win => Outcome { winner: player as i8, reason: Reason::StalemateWin },
        })
    }

    /// The whole verdict, generating both lists.
    pub fn round_terminal(&mut self) -> Option<Outcome> {
        if let Some(outcome) = self.round_terminal_prefix() {
            return Some(outcome);
        }
        let mut first = Vec::new();
        let mut second = Vec::new();
        self.generate_legal_for(0, &mut first);
        self.generate_legal_for(1, &mut second);
        self.round_terminal_with([&first, &second])
    }

    /// Joint perft (§6.1): positions after `depth` rounds, every pair of legal moves.
    pub fn perft_round(&mut self, depth: u32, stop_at_terminal: bool) -> u64 {
        if depth == 0 {
            return 1;
        }
        if stop_at_terminal && self.round_terminal_prefix().is_some() {
            return 0;
        }
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
                total += self.perft_round(depth - 1, stop_at_terminal);
                self.unmake_round();
            }
        }
        total
    }
}
