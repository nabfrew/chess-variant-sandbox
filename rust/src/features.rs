//! The feature vocabulary, mirroring `chess_variant_sandbox/agents/features.py`.
//!
//! `features.py` is the oracle and this is the port, exactly as `agents/minimax.py` is
//! the oracle for `agents.rs`. Every term here accumulates in the same order as the
//! Python one, because float addition is not associative and the two are held to
//! producing the same number.
//!
//! Nothing here derives a piece value. Those are blended on the Python side and handed
//! over with the tables, so there is still exactly one definition of what a piece is
//! worth -- and a heuristic's weights change what is being *maximised*, never what a
//! piece is.

use crate::eval::{Evaluator, PieceValues};
use crate::game::Game;
use crate::net::NetEvaluator;
use crate::tables::{Tables, GOAL_UNREACHABLE, MAX_PIECE_TYPES, WILD};

/// Bumped in step with `features.VOCABULARY_VERSION`.
pub const VOCABULARY_LEN: usize = 15;

// Positions in `features.VOCABULARY`. A term added on one side and not the other shows
// up as a length mismatch at construction rather than as a quietly different player.
const MATERIAL: usize = 0;
const PREY_WEIGHT: usize = 1;
const MOBILITY: usize = 2;
const ADVANCE: usize = 3;
const CENTRALITY: usize = 4;
const ROYAL_EXPOSURE: usize = 5;
const TERRITORY_MARGIN: usize = 6;
const SCORE_MARGIN: usize = 7;
const SCORE_URGENCY: usize = 8;
const CHECK_MARGIN: usize = 9;
const EXTINCTION_MARGIN: usize = 10;
const HAND_MATERIAL: usize = 11;
const PROMOTION_OCCUPANCY: usize = 12;
const REACH_PROGRESS: usize = 13;
const FACING_ALIGNMENT: usize = 14;

/// Mirrors `agents/evaluation.NO_PREY_FLOOR`.
const NO_PREY_FLOOR: f64 = 0.1;

/// How much of its own forward a piece at each quarter turn still points: the cosine of
/// the angle, exact at all four. Mirrors `features._ALIGNMENT`, and a table rather than
/// `f64::cos` because the three reachable values are exactly 1, 0 and -1 and a
/// transcendental function is not how two implementations agree to the bit.
const ALIGNMENT: [f64; 4] = [1.0, 0.0, -1.0, 0.0];

/// Per-ruleset constants, derived once. Everything here is a fact about the rules.
pub struct FeatureContext {
    total_material: f64,
    centre_x: f64,
    centre_y: f64,
    diameter: f64,
    territory_win_at: usize,
    has_territory: bool,
    has_scoring: bool,
    score_ceiling: f64,
    max_plies: u32,
    check_limit: u32,
    extinction_types: Vec<usize>,
    extinction_initial: Vec<usize>,
    /// Schema 27 (Q936), `"per_type": "own"`: each side's counted types with the count of
    /// each it started with. Empty for both sides under every other spelling.
    extinction_own: [Vec<(usize, usize)>; 2],
    has_drops: bool,
    promotable: Vec<usize>,
}

impl FeatureContext {
    /// The opening scan counts the **two armies** and no third owner.
    ///
    /// This is the one place in this file that walks `initial_board` rather than
    /// `Game::all_occupied`, so it is the one place a wild piece has to be excluded by
    /// hand -- every other walk here is already the players' union and never sees one.
    ///
    /// `total` is the denominator `material`, `prey_weight` and `hand_material` divide
    /// by, and those numerators are sums over the two armies; counting a boar here would
    /// price every army against a quantity larger than anything either army can hold.
    /// `counts` feeds `score_ceiling` under `material` and under the piece count, and
    /// `extinction_initial`, which is compared against `live_of` -- a count over
    /// `occupied()[side]`, which has always excluded a wild piece. A threshold and the
    /// thing it thresholds have to be the same quantity.
    ///
    /// Mirrors `features.feature_context`, which carries the same reasoning.
    pub fn new(t: &Tables, values: &[f64]) -> Self {
        let mut counts = vec![0usize; t.n_types];
        let mut total = 0.0;
        for square in 0..t.n_squares {
            let value = t.initial_board[square];
            if value != 0 && t.owner_of(value) != WILD {
                let piece_type = t.type_of(value);
                counts[piece_type] += 1;
                total += values[piece_type];
            }
        }

        let (extinction_types, extinction_initial) = match &t.extinction {
            Some(rule) if !rule.any_piece => {
                let types = rule.piece_types.clone();
                let initial = types.iter().map(|ty| counts[*ty]).collect();
                (types, initial)
            }
            _ => (Vec::new(), Vec::new()),
        };
        let extinction_own = match &t.extinction {
            Some(rule) if rule.own => [0, 1].map(|player| {
                rule.side_types[player]
                    .iter()
                    .map(|&piece_type| {
                        let started = (0..t.n_squares)
                            .filter(|&square| {
                                let value = t.initial_board[square];
                                value != 0
                                    && t.owner_of(value) == player
                                    && t.type_of(value) == piece_type
                            })
                            .count();
                        (piece_type, started)
                    })
                    .collect()
            }),
            _ => [Vec::new(), Vec::new()],
        };

        let score_ceiling = match &t.scoring {
            None => 1.0,
            Some(rule) => match rule.count {
                // 0 material, 1 pieces, 2 region, 3 territory, 4 points.
                0 => {
                    let mut sum = 0.0;
                    for (value, count) in t.piece_values.iter().zip(counts.iter()) {
                        sum += value * *count as f64;
                    }
                    if sum > 1.0 {
                        sum
                    } else {
                        1.0
                    }
                }
                3 => t.n_squares as f64,
                // Every bounty on the board at ply zero: what the two tallies can come
                // to between them, which is what a points gap is a share of. Falling
                // through to the piece count below would measure the gap against a
                // quantity the rule does not count.
                4 => t.npc_bounty_total.max(1.0),
                2 => match &rule.squares {
                    Some([first, second]) => {
                        let a = first.iter().filter(|s| **s).count();
                        let b = second.iter().filter(|s| **s).count();
                        std::cmp::max(std::cmp::max(a, b), 1) as f64
                    }
                    None => 1.0,
                },
                _ => std::cmp::max(counts.iter().sum::<usize>(), 1) as f64,
            },
        };

        Self {
            total_material: if total != 0.0 { total } else { 1.0 },
            centre_x: (t.width as f64 - 1.0) / 2.0,
            centre_y: (t.height as f64 - 1.0) / 2.0,
            diameter: std::cmp::max(t.width, t.height) as f64,
            territory_win_at: t.territory.as_ref().map(|r| r.win_at as usize).unwrap_or(0),
            has_territory: t.territory.is_some(),
            has_scoring: t.scoring.is_some(),
            score_ceiling,
            max_plies: t.max_plies,
            check_limit: t.check_limit,
            extinction_types,
            extinction_initial,
            extinction_own,
            has_drops: t.drops.is_some(),
            promotable: t.promo_piece_types.clone(),
        }
    }
}

/// `dot(weights, features(position))`, skipping the terms weighted at zero.
///
/// Skipping rather than multiplying by zero and summing is not an optimisation: a term
/// contributing `0.0` still perturbs a float sum, so two heuristics differing only in a
/// term neither uses would otherwise be different players.
pub struct FeatureEvaluator {
    context: FeatureContext,
    active: Vec<(usize, f64)>,
    /// The three valuation terms -- material, the prey correction and the hand -- read
    /// this, and it carries the player's own values where it has an opinion.
    objective: Evaluator,
    /// What the rules imply, kept for capture ordering. Ordering is not the objective;
    /// if it moved with an override, the node budget would buy different moves and an
    /// arm would differ in two things at once.
    derived: Vec<f64>,
}

impl FeatureEvaluator {
    pub fn new(t: &Tables, weights: &[f64], values: &PieceValues) -> Self {
        let active = weights
            .iter()
            .enumerate()
            .filter(|(_, w)| **w != 0.0)
            .map(|(i, w)| (i, *w))
            .collect();
        Self {
            // Derived, always. `total_material` is what puts every term on one scale, so
            // a player able to move it would be changing what its other thirteen
            // weights mean.
            context: FeatureContext::new(t, values.derived()),
            active,
            objective: Evaluator::new(t, values.objective().to_vec(), 0.0, 0.0),
            derived: values.derived().to_vec(),
        }
    }

    pub fn values(&self) -> &[f64] {
        &self.derived
    }

    pub fn evaluate(&self, game: &Game<'_>, player: usize, move_count: usize) -> f64 {
        let mut score = 0.0;
        for (index, weight) in &self.active {
            score += weight * self.term(*index, game, player, move_count);
        }
        score
    }

    fn term(&self, index: usize, game: &Game<'_>, player: usize, move_count: usize) -> f64 {
        let t = game.t;
        let c = &self.context;
        match index {
            MATERIAL => self.objective.board_material(game, player) / c.total_material,
            PREY_WEIGHT => self.prey_weight(game, player),
            MOBILITY => {
                let signed = if game.side == player {
                    move_count as f64
                } else {
                    -(move_count as f64)
                };
                signed / t.n_squares as f64
            }
            ADVANCE => self.advance(game, player),
            CENTRALITY => self.centrality(game, player),
            ROYAL_EXPOSURE => {
                if t.royal_types.is_empty() {
                    0.0
                } else {
                    f64::from(game.in_check(1 - player)) - f64::from(game.in_check(player))
                }
            }
            TERRITORY_MARGIN => self.territory_margin(game, player),
            SCORE_MARGIN => {
                if !c.has_scoring {
                    0.0
                } else {
                    clamp(self.score_gap(game, player) / c.score_ceiling)
                }
            }
            SCORE_URGENCY => {
                if !c.has_scoring || c.max_plies == 0 {
                    0.0
                } else {
                    let urgency = (game.ply as f64 / c.max_plies as f64).min(1.0);
                    clamp(self.score_gap(game, player) / c.score_ceiling) * urgency
                }
            }
            CHECK_MARGIN => {
                if c.check_limit == 0 {
                    0.0
                } else {
                    clamp(
                        (game.checks[player] as f64 - game.checks[1 - player] as f64)
                            / c.check_limit as f64,
                    )
                }
            }
            EXTINCTION_MARGIN => self.extinction_margin(game, player),
            HAND_MATERIAL => self.hand_material(game, player),
            PROMOTION_OCCUPANCY => self.promotion_occupancy(game, player),
            REACH_PROGRESS => self.reach_progress(game, player),
            FACING_ALIGNMENT => facing_alignment(game, player),
            _ => 0.0,
        }
    }

    fn prey_weight(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        if !t.has_capture_matrix {
            return 0.0;
        }
        let n_types = t.n_types;
        // Fixed size rather than four heap allocations at every evaluated node, which is
        // the shape `Evaluator::live_values` is already written in and for the same
        // reason. Entries past `n_types` are never read; the two sums below are sliced so
        // that they cannot be, and integer addition of the zeros would be exact anyway.
        // Every walk here is over the occupied squares in ascending order, which is
        // board order over the terms the scan over every square actually added.
        //
        // Both walks are over the *players'* union, so a wild piece is in neither army
        // and in neither the share nor the valuation: the quantity is "of the enemy
        // material anybody could ever win, what fraction may I win", and a boar is not
        // one of my opponent's pieces however takeable it is. Nor is a piece **no
        // attacker may take** in that quantity, which is what `prey` filters out of both
        // halves below: it can never be won by anyone, so counting it paid its placer a
        // share of the opponent's whole army for a move that captured nothing.
        // `Evaluator::live_values` says the same thing about the same quantity and
        // carries the argument; `prey` is that evaluator's own pool rather than a second
        // derivation. Both `live` and `share` have two rows, so a boar arriving here
        // would be a bounds panic rather than a wrong number.
        let prey = &self.objective.prey;
        let occupied = game.all_occupied();
        let mut live = [[0usize; MAX_PIECE_TYPES]; 2];
        for square in occupied.iter() {
            let value = game.board[square];
            live[t.owner_of(value)][t.type_of(value)] += 1;
        }
        let army = [
            live[0][..n_types]
                .iter()
                .enumerate()
                .filter(|(victim, _)| prey[*victim])
                .map(|(_, count)| *count)
                .sum::<usize>(),
            live[1][..n_types]
                .iter()
                .enumerate()
                .filter(|(victim, _)| prey[*victim])
                .map(|(_, count)| *count)
                .sum::<usize>(),
        ];

        let mut share = [[1.0f64; MAX_PIECE_TYPES]; 2];
        for side in 0..2 {
            let enemy = &live[1 - side];
            let total_enemy = army[1 - side];
            if total_enemy == 0 {
                continue;
            }
            for (attacker, entry) in share[side][..n_types].iter_mut().enumerate() {
                let mut allowed = 0usize;
                for (victim, count) in enemy[..n_types].iter().enumerate() {
                    if prey[victim] && t.capture_allowed[attacker * n_types + victim] {
                        allowed += count;
                    }
                }
                let value = allowed as f64 / total_enemy as f64;
                *entry = if value > NO_PREY_FLOOR { value } else { NO_PREY_FLOOR };
            }
        }

        let mut relational = 0.0;
        let mut static_value = 0.0;
        for square in occupied.iter() {
            let value = game.board[square];
            let owner = t.owner_of(value);
            let piece_type = t.type_of(value);
            // The override is what the live share scales, not the derived table: the
            // share is a correction *to a valuation*.
            let worth = self.objective.values[piece_type];
            let live_worth = worth * share[owner][piece_type];
            if owner == player {
                relational += live_worth;
                static_value += worth;
            } else {
                relational -= live_worth;
                static_value -= worth;
            }
        }
        (relational - static_value) / self.context.total_material
    }

    /// How far each army has walked towards the far side, mine minus theirs.
    ///
    /// **A wild piece is in neither the numerator nor the denominator**, and the walk is
    /// what excludes it: `all_occupied` is the players' union and a boar is in neither
    /// set. It has no forward -- it walks a fixed cycle stated at compile time -- so it
    /// makes no progress towards anybody's far side, and leaving it in `count` would
    /// divide a two-army difference by a count of three owners' pieces. The comparison
    /// below is `owner == player`, which is false for a wild piece whoever is asking, so
    /// a boar reaching this loop would be subtracted from *both* players and the term
    /// would stop being antisymmetric -- silently. Hence the assertion.
    fn advance(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        if t.height < 2 {
            return 0.0;
        }
        let span = (t.height - 1) as f64;
        let mut total = 0.0;
        let mut count = 0usize;
        for square in game.all_occupied().iter() {
            let value = game.board[square];
            let owner = t.owner_of(value);
            debug_assert!(owner != WILD, "a wild piece reached advance");
            let piece_type = t.type_of(value);
            if t.is_royal(piece_type) {
                continue;
            }
            let rank = (square / t.width) as f64;
            let progress = if owner == 0 {
                rank / span
            } else {
                (span - rank) / span
            };
            if owner == player {
                total += progress;
            } else {
                total -= progress;
            }
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    /// How near the middle each army stands, mine minus theirs.
    ///
    /// A wild piece stands in neither army, and is out of both halves of the subtraction
    /// and out of the average, for the reason `advance` states and by the same walk. A
    /// boar on the centre is in both players' way equally; what it costs each of them is
    /// that it blocks and it threatens, which `mobility` and `royal_exposure` read by
    /// asking the engine rather than by decoding a board byte here.
    fn centrality(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        let c = &self.context;
        let mut total = 0.0;
        let mut count = 0usize;
        for square in game.all_occupied().iter() {
            let owner = t.owner_of(game.board[square]);
            debug_assert!(owner != WILD, "a wild piece reached centrality");
            let rank = (square / t.width) as f64;
            let file = (square % t.width) as f64;
            let distance = (file - c.centre_x).abs().max((rank - c.centre_y).abs()) / c.diameter;
            let near = 1.0 - distance;
            if owner == player {
                total += near;
            } else {
                total -= near;
            }
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    fn territory_margin(&self, game: &Game<'_>, player: usize) -> f64 {
        let c = &self.context;
        if !c.has_territory || game.colour.is_empty() {
            return 0.0;
        }
        let mine = game.coloured_by(player) as f64;
        let theirs = game.coloured_by(1 - player) as f64;
        let denominator = if c.territory_win_at != 0 {
            c.territory_win_at as f64
        } else {
            game.t.n_squares as f64
        };
        clamp((mine - theirs) / denominator)
    }

    fn extinction_margin(&self, game: &Game<'_>, player: usize) -> f64 {
        let c = &self.context;
        if c.extinction_types.is_empty() {
            return 0.0;
        }
        if !c.extinction_own[0].is_empty() || !c.extinction_own[1].is_empty() {
            // Schema 27 (Q936): the armies count different kinds, so one side's count of a
            // type against the other's means nothing. Each side's worst own kind against
            // its own opening, and the margin is the difference.
            let worst = |side: usize| {
                c.extinction_own[side]
                    .iter()
                    .fold(1.0f64, |worst, &(piece_type, started)| {
                        let left = live_of(game, piece_type, side) as f64;
                        worst.min(left / std::cmp::max(started, 1) as f64)
                    })
            };
            return clamp(worst(player) - worst(1 - player));
        }
        let mut worst = 1.0f64;
        for (index, piece_type) in c.extinction_types.iter().enumerate() {
            let initial = std::cmp::max(c.extinction_initial[index], 1) as f64;
            let mine = live_of(game, *piece_type, player) as f64;
            let theirs = live_of(game, *piece_type, 1 - player) as f64;
            let value = (mine - theirs) / initial;
            if value < worst {
                worst = value;
            }
        }
        clamp(worst)
    }

    /// The hands over the opening material: `eval::Evaluator::hand_value`, the half of
    /// the built-in `material` that the board-only `material` column leaves out. One
    /// definition, so `builtin_vector`'s weight on this column is the built-in's hand.
    fn hand_material(&self, game: &Game<'_>, player: usize) -> f64 {
        if !self.context.has_drops {
            return 0.0;
        }
        self.objective.hand_value(game, player) / self.context.total_material
    }

    fn promotion_occupancy(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        if self.context.promotable.is_empty() {
            return 0.0;
        }
        let occupied = game.occupied();
        let mut share = [0.0f64; 2];
        for (side, entry) in share.iter_mut().enumerate() {
            let mut eligible = 0usize;
            let mut standing = 0usize;
            for square in occupied[side].iter() {
                let piece_type = t.type_of(game.board[square]);
                if !t.is_promoting(piece_type) {
                    continue;
                }
                eligible += 1;
                if t.promotes_on(side, piece_type, square) {
                    standing += 1;
                }
            }
            *entry = if eligible == 0 {
                0.0
            } else {
                standing as f64 / eligible as f64
            };
        }
        share[player] - share[1 - player]
    }

    fn reach_progress(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        let rule = match &t.reach {
            None => return 0.0,
            Some(rule) => rule,
        };
        if t.goal_distance.is_empty() {
            return 0.0;
        }
        let occupied = game.occupied();
        let mut near = [0.0f64; 2];
        for (side, entry) in near.iter_mut().enumerate() {
            if !rule.squares[side].iter().any(|s| *s) {
                continue;
            }
            let mut best = t.goal_diameter;
            for square in occupied[side].iter() {
                let piece_type = t.type_of(game.board[square]);
                // `REACH_ANY` is "any piece", as in `eval.rs::nearest_to_goal` and the
                // arrival check in `game.rs`. Until Q465 this cast the -1 to a `usize`
                // no type equals, so the term read zero on every such ruleset.
                if !rule.admits(piece_type) {
                    continue;
                }
                let slot = t.goal_slot(side, piece_type, square);
                let distance = t.goal_distance[slot];
                if distance != GOAL_UNREACHABLE && (distance as f64) < best {
                    best = distance as f64;
                }
            }
            *entry = 1.0 - (best / t.goal_diameter).min(1.0);
        }
        near[player] - near[1 - player]
    }

    fn score_gap(&self, game: &Game<'_>, player: usize) -> f64 {
        let t = game.t;
        let rule = match &t.scoring {
            None => return 0.0,
            Some(rule) => rule,
        };
        let mut counts = [0.0f64; 2];
        if rule.count == 3 {
            if game.colour.is_empty() {
                return 0.0;
            }
            counts[0] = game.coloured_by(0) as f64;
            counts[1] = game.coloured_by(1) as f64;
        } else if rule.count == 4 {
            // Points are a tally, not a count of anything on the board (schema 9).
            counts[0] = game.points[0];
            counts[1] = game.points[1];
        } else if rule.count == 2 {
            if let Some(squares) = &rule.squares {
                let occupied = game.occupied();
                for side in 0..2 {
                    let mut held = 0.0;
                    for square in occupied[side].iter() {
                        if squares[side][square] {
                            held += 1.0;
                        }
                    }
                    counts[side] = held;
                }
            }
        } else {
            // A wild piece scores for nobody: the walk is the players' union, so a boar
            // is in neither column. `counts` has two, so one arriving here would be a
            // bounds panic rather than a piece quietly credited to the opponent.
            let material = rule.count == 0;
            for square in game.all_occupied().iter() {
                let value = game.board[square];
                let owner = t.owner_of(value);
                let piece_type = t.type_of(value);
                counts[owner] += if material { t.piece_values[piece_type] } else { 1.0 };
            }
        }
        counts[player] - counts[1 - player]
    }
}

/// How far each turnable piece is turned from its own side's forward, mine minus theirs.
///
/// Mirrors `features.FeatureEvaluator._facing_alignment`, which carries the reasoning.
/// The only term in the vocabulary that reads a piece's **frame**, and therefore the
/// only one that can price `piece.facing` at all -- see `docs/reference/schema-8.md`.
///
/// Three things have to be decoded right and only one of them is checked by anything
/// else. `frame_of_value` is the **frame**, not the owner: the owner is `owner_of` and the
/// facing is `frame % n_facings`, and swapping the owner and the facing
/// reads prices the wrong army -- silently, and only on a ruleset that faces, which is
/// why `tests/test_backends.py`'s facing arms give the two players *different* facing
/// distributions. A type with one orientation has no frame to read and is in neither the
/// sum nor the count; a spec where nothing turns leaves before any arithmetic, which is
/// what keeps the term exactly `0.0` and free on the rest of the corpus.
///
/// A free function rather than a method because it reads nothing off `FeatureContext`:
/// every constant it needs is already on `Tables`, and copying them would be a second
/// place for `n_facings` to be wrong.
fn facing_alignment(game: &Game<'_>, player: usize) -> f64 {
    let t = game.t;
    if !t.has_facing {
        return 0.0;
    }
    let step = 4 / t.n_facings;
    let mut total = 0.0;
    let mut count = 0usize;
    for square in game.all_occupied().iter() {
        let value = game.board[square];
        let frame = t.frame_of_value(value);
        let piece_type = t.type_of(value);
        let owner = t.owner_of(value);
        debug_assert!(owner != WILD, "a wild piece reached facing_alignment");
        // A facing no move turns (schema 25) is out with the types that have none.
        if t.turns_of(piece_type) < 2 {
            continue;
        }
        let aligned = ALIGNMENT[(frame % t.n_facings) * step];
        if owner == player {
            total += aligned;
        } else {
            total -= aligned;
        }
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[inline]
fn clamp(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

fn live_of(game: &Game<'_>, piece_type: usize, side: usize) -> usize {
    // Every orientation of the type, because "how many of these are left" is a question
    // about the piece and not about which way it is pointing.
    let t = game.t;
    game.occupied()[side]
        .iter()
        .filter(|&square| t.is_piece(game.board[square], side, piece_type))
        .count()
}

/// Which objective a player maximises, mirroring `chess_variant_sandbox/agents/objective.py`.
///
/// An empty weight vector must reach `eval::Evaluator` and nothing else -- not a feature
/// evaluator configured to behave like it. Every committed number was produced by that
/// type, and float addition is not associative, so "equivalent arithmetic" is not the
/// same claim as "the same arithmetic".
pub enum Objective {
    Builtin(Evaluator),
    Heuristic(Box<FeatureEvaluator>),
    /// Q502e: a learned net (`net.rs`). Reached only when a `net=` player's store
    /// resolved to one; a refused or absent store sends no bytes and lands on `Builtin`.
    Net(Box<NetEvaluator>),
}

impl Objective {
    /// An empty weight vector reaches `eval::Evaluator` on the **derived** table, which
    /// is the built-in objective and the thing every committed number was produced by.
    /// A player with no weights therefore has no piece values either; the boundary
    /// refuses that combination rather than silently ignoring the override.
    pub fn new(
        t: &Tables,
        weights: &[f64],
        values: &PieceValues,
        mobility: f64,
        goal: f64,
        immune: f64,
    ) -> Self {
        if weights.is_empty() {
            Objective::Builtin(
                Evaluator::new(t, values.derived().to_vec(), mobility, goal)
                    .with_immunity(t, immune),
            )
        } else {
            Objective::Heuristic(Box::new(FeatureEvaluator::new(t, weights, values)))
        }
    }

    /// `&mut` because a net keeps its accumulator between evaluations.
    #[inline]
    pub fn evaluate(&mut self, game: &Game<'_>, player: usize, move_count: usize) -> f64 {
        match self {
            Objective::Builtin(e) => e.evaluate(game, player, move_count),
            Objective::Heuristic(e) => e.evaluate(game, player, move_count),
            Objective::Net(e) => e.evaluate(game, player),
        }
    }

    /// Schema 26 (Q918b): the objective in a round, mobility mine minus theirs. The
    /// built-in objective and, since Q918g, a learned net play rounds (`AgentSpec::
    /// check_against` refuses a heuristic on a simultaneous game). The net reads no move
    /// list: its one head is seat 0's (`game.side` stays 0 in a round) and seat 1's score
    /// is its negation (`NetEvaluator::evaluate`).
    #[inline]
    pub fn evaluate_round(&mut self, game: &Game<'_>, player: usize, lists: [&[u32]; 2]) -> f64 {
        match self {
            Objective::Builtin(e) => e.evaluate_round(game, player, lists),
            Objective::Net(e) => e.evaluate(game, player),
            Objective::Heuristic(_) => {
                unreachable!("check_against refuses a heuristic objective in a round")
            }
        }
    }

    /// The piece values capture ordering reads. Ordering is not the objective, so a
    /// heuristic never changes it.
    #[inline]
    pub fn values(&self) -> &[f64] {
        match self {
            Objective::Builtin(e) => &e.values,
            Objective::Heuristic(e) => e.values(),
            Objective::Net(e) => &e.values,
        }
    }
}
