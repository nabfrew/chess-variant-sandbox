"""The feature vocabulary a heuristic player is a weight vector over.

Every term reads a rule the schema declares or a fact about the compiled tables. None
names a piece, and none assumes a board -- ``tests/test_dsl_generality.py`` exists to
catch a chess assumption creeping in, and this file is the most tempting place for one.

Three properties are load-bearing and none of them is cosmetic.

**Fixed length.** The vector has one entry per term in ``VOCABULARY``, always, whatever
the ruleset declares. That is what makes a digest comparable, what lets the model's
response schema be a fixed-shape array with ``additionalProperties: false``, and what
makes ``heuristic_headroom`` on chess and on a territory game the same quantity: a term
whose rule the ruleset does not declare reads exactly ``0.0`` and is inert.

**Signed and normalised.** Every term is from ``player``'s point of view and lands in
roughly ``[-1, 1]``, expressed wherever possible as a share of the way to winning. That
is what lets one weight mean the same thing on a 3x4 board and a 12x8 one. It is also
`docs/plan.md` stage 17's A5 argument -- one derived goal weight, in units that survive
a change of board -- generalised from three terms to the whole vocabulary.

**No term that is a difference of two nearly-equal quantities.** A `dispersion` term
-- how spread out each army is, mine minus theirs -- was written, measured and removed.
It is the only term anyone proposed that reads *no declared rule at all*, which already
makes it the kind of thing `tests/test_dsl_generality.py` exists to catch; and because
the result is a small difference between two large means, a last-bit disagreement in
either operand is amplified by up to 962x on the measured corpus, which is enough to
change which move comes out. A term the two implementations cannot agree on to the bit
is a term that makes the Python oracle and the Rust port different players, and the
project's answer to that is not a tolerance.

**No per-piece-type weights.** A tempting extra entry, and excluded deliberately.
It would make the vector's length depend on the ruleset (three types on `rps-v1`, nine
on `courier`), which breaks both the fixed schema and the comparability of a digest; it
is precisely where chess knowledge would enter, since "the knight is worth three" is a
model's prior and not a fact about the rules; and the project already derives piece
values from the rules once, so letting a proposal overwrite them turns this into a
piece-value tuner, which is a different experiment with a different confound.

**This is not the evaluator the built-in agents use.** ``agents/evaluation.py`` is
untouched and an empty weight vector never reaches this file, which is what keeps every
committed number where it is. ``builtin_vector`` returns the weights that reproduce the
*material and mobility* half of it, which was the whole of it before stage 17A added the
goal terms -- see ``agents/seeds.anchor`` for why that half is still the right anchor and
why claiming more would be worse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from chess_variant_sandbox.agents.evaluation import NO_PREY_FLOOR, Evaluator, PieceValues
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import UNREACHABLE, WILD, CompiledSpec

# Bumped whenever a term is added, removed or redefined. A weight vector only means
# anything against the vocabulary it was drawn for, so this is the same kind of number
# as `SCHEMA_VERSION` and as a weights version: adding a term is a bump, never an edit,
# and a heuristic store whose version does not match is refused at load.
#
# **Schema 9's third owner did not bump it, and the argument was the inert set.** Six
# terms below learned that a wild piece is neither player's, which on a ruleset with
# `rules.npcs` changes what they read -- but every such ruleset previously raised an
# `IndexError` here or read a boar as the enemy, so there was no vector anywhere that had
# been drawn against the old answer. On the two-owner rulesets every stored heuristic set
# was drawn on, `owner == WILD` is never true and each edit was a no-op to the bit. A
# bump would have orphaned every store to buy a change to no number that existed.
#
# **`facing_alignment` bumped it to 2, and that argument does not transfer.** The
# question is not whether the new term *reads* zero on a ruleset with no facing -- it
# does, exactly, before any arithmetic runs -- but whether a stored vector still names a
# player. It does not: a vector is fixed-length by construction, so a fourteen-weight
# store is refused by its own length at `FeatureEvaluator.__init__` and by
# `Heuristic.__post_init__`, and the refusal in `from_dict` for an unbumped edit ("one
# of them was edited without a bump") is exactly the state an unbumped addition would
# leave the stores in. So the bump is owed by the *shape* of the vector rather than by
# the value of any term, and it is owed whether or not a ruleset faces.
VOCABULARY_VERSION = 2

VOCABULARY: tuple[str, ...] = (
    "material",
    "prey_weight",
    "mobility",
    "advance",
    "centrality",
    "royal_exposure",
    "territory_margin",
    "score_margin",
    "score_urgency",
    "check_margin",
    "extinction_margin",
    "hand_material",
    "promotion_occupancy",
    "reach_progress",
    "facing_alignment",
)

# How much of its own forward a piece at each quarter turn still points, which is the
# cosine of the angle and is exact at every one of the four. A table rather than
# `math.cos`, because the three values a spec can reach are exactly 1, 0 and -1 and a
# transcendental function is not the way to say so on two sides of a boundary.
_ALIGNMENT: tuple[float, ...] = (1.0, 0.0, -1.0, 0.0)

# What each term means, in English, for the model that writes a vector. These are shown
# to the proposer, so two rules govern them and both are tested:
#
#   * a gloss may NAME a quantity and may never state a preferred value or a band.
#     "How many moves the mover has" is a term definition; "the number of options in a
#     comfortable range" is a curve, and a curve is the part of an objective an
#     optimiser can game.
#   * no gloss may restate a scored measurement's operational sentence.
GLOSSES: dict[str, str] = {
    "material": (
        "Your pieces minus theirs, valued by how much of the board each piece can "
        "attack, as a share of everything on the board at the start."
    ),
    "prey_weight": (
        "Only meaningful where pieces may capture some kinds of piece and not others. "
        "Raise it to value a piece by what it can still take on the board right now "
        "rather than by what it could take at the start."
    ),
    "mobility": "How many moves the side to move has, as a share of the squares.",
    "advance": (
        "How far your pieces have travelled from your own side of the board towards "
        "the far side, minus the same for the opponent."
    ),
    "centrality": "How close your pieces are to the middle of the board, minus theirs.",
    "royal_exposure": "Whether the opponent's royal piece is in check, minus whether yours is.",
    "territory_margin": (
        "Only where a rule claims squares. The ground you hold minus the ground they "
        "hold, as a share of the amount that wins."
    ),
    "score_margin": (
        "Only where a rule decides the game on a count when the clock runs out. Your "
        "count minus theirs, as a share of the largest it could be."
    ),
    "score_urgency": (
        "The same count, but growing as the clock approaches, so it is worth nothing at "
        "the start and everything at the end."
    ),
    "check_margin": (
        "Only where a rule ends the game on a number of checks. Checks you have given "
        "minus checks they have, as a share of the number that wins."
    ),
    "extinction_margin": (
        "Only where a rule ends the game when a kind of piece runs out. How much of "
        "that kind you still have, minus how much they do."
    ),
    "hand_material": (
        "Only where captured pieces can be played back. What you hold off the board "
        "minus what they hold."
    ),
    "promotion_occupancy": (
        "Only where pieces promote. The share of your promotable pieces that are "
        "standing where they would promote."
    ),
    "reach_progress": (
        "Only where a rule is won by getting a piece to a particular square. How close "
        "your nearest eligible piece is to such a square, minus theirs."
    ),
    "facing_alignment": (
        "Only where a piece may turn to face another way. How much your turnable pieces "
        "point the way the rules call forward for your side rather than sideways or "
        "back, minus the same for the opponent."
    ),
}

# What a heuristic may not do to itself. A vector that is all zeros is a player that
# scores every move identically and therefore moves at random; that is a legitimate null
# control but never a proposal, so the validator rejects it.
MAX_ABS_WEIGHT = 10.0


@dataclass(frozen=True)
class FeatureContext:
    """Per-ruleset constants, derived once so a node does not re-derive them.

    Everything here is a fact about the rules. Nothing is fitted, and nothing depends on
    a game having been played.
    """

    base: Evaluator
    n_types: int
    # How many board values one player owns: `n_facings * n_types`, and `n_types` for
    # every ruleset in which nothing turns. `(value - 1) // n_types` is the *frame*, not
    # the owner, and reading it as the owner would price the wrong army -- silently, and
    # only on a ruleset that has a facing.
    side_stride: int
    # How many orientations the *spec* has, which is the frame stride inside one army:
    # `frame // n_facings` is the owner and `frame % n_facings` is the facing. One for
    # every ruleset in which nothing turns, where both halves collapse to what they were
    # before schema 8 and `has_facing` is false.
    n_facings: int
    has_facing: bool
    # How many orientations each *type* can turn through, which is 1 for a type that cannot
    # turn even in a spec where something else can -- one with no facing, or (schema 25) a
    # facing no move turns. Such a piece has no facing to choose, so it is in neither half
    # of `facing_alignment`; `features.rs` asks `Tables::turns_of` the same question.
    facing_axes: tuple[int, ...]
    n_squares: int
    width: int
    height: int
    total_material: float
    royal_types: frozenset[int]
    centre: tuple[float, float]
    diameter: float
    territory_win_at: int
    has_territory: bool
    has_scoring: bool
    score_ceiling: float
    max_plies: int
    check_limit: int
    extinction_types: tuple[int, ...]
    extinction_initial: tuple[int, ...]
    # Schema 27 (Q936), `"per_type": "own"`: each side's counted types, each with how many
    # of it that side started with. Empty for both sides under every other spelling.
    extinction_own: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    has_drops: bool
    promotable: frozenset[int]
    promotion_squares: tuple[frozenset[int], frozenset[int]]
    reach_types: frozenset[int]
    reach_squares: tuple[frozenset[int], frozenset[int]]
    goal_distance: tuple[tuple[tuple[int, ...], ...], ...]
    goal_diameter: float
    live_prey: tuple[tuple[bool, ...], ...]
    # Which types can be won at all: the denominator ``prey_weight``'s share is over.
    # Taken off the built-in evaluator rather than derived a second time here, because
    # the two terms are the same quantity and the docstrings say so.
    prey_pool: tuple[bool, ...]
    has_capture_matrix: bool
    initial_counts: tuple[int, ...]


def feature_context(spec: CompiledSpec) -> FeatureContext:
    """Derive the per-ruleset constants every term reads.

    The opening scan counts the **two armies** and no third owner. ``total_material`` is
    the denominator ``material`` and ``prey_weight`` divide by and those numerators no
    longer count a wild piece, so counting one here would price every army against a
    quantity larger than anything either army can hold; and ``initial_counts`` feeds
    ``extinction_initial``, which is compared against ``_live_of``, a range test that has
    always excluded a wild piece. A threshold and the thing it thresholds have to be the
    same quantity.
    """
    base = Evaluator.for_spec(spec, mobility_weight=0.0)

    stride = spec.side_stride
    counts = [0] * spec.n_types
    total = 0.0
    for square in range(spec.n_squares):
        value = spec.initial_board[square]
        if value and (value - 1) // stride != WILD:
            piece_type = (value - 1) % spec.n_types
            counts[piece_type] += 1
            total += base.values[piece_type]

    territory = spec.territory
    scoring = spec.scoring
    extinction = spec.extinction
    reach = spec.reach

    extinction_types: tuple[int, ...] = ()
    if extinction is not None and not extinction.any_piece:
        extinction_types = tuple(extinction.piece_types)

    promotable: set[int] = set()
    promotion_squares: tuple[set[int], set[int]] = (set(), set())
    for rule in spec.promotions:
        promotable.add(rule.piece_type)
        for player in (0, 1):
            promotion_squares[player].update(rule.squares[player])

    reach_types: set[int] = set()
    reach_squares: tuple[set[int], set[int]] = (set(), set())
    if reach is not None:
        # `piece_type == -1` is "any piece" (`compile.ReachRule`). Adding the -1 itself,
        # as this did until Q465, named a type no board byte decodes to, so the term read
        # exactly zero on every such ruleset -- `intransitive` and `rps-v3` -- while
        # reading correctly on a named racer. Every type is eligible, royals included,
        # which is what `Evaluator._nearest_to_goal` and the core's arrival check count.
        if reach.piece_type < 0:
            reach_types.update(range(spec.n_types))
        else:
            reach_types.add(reach.piece_type)
        for player in (0, 1):
            reach_squares[player].update(reach.squares[player])

    return FeatureContext(
        base=base,
        n_types=spec.n_types,
        side_stride=spec.n_facings * spec.n_types,
        n_facings=spec.n_facings,
        has_facing=spec.has_facing,
        facing_axes=tuple(
            spec.facing_axes[piece_type] if spec.turns(piece_type) else 1
            for piece_type in range(spec.n_types)
        ),
        n_squares=spec.n_squares,
        width=spec.width,
        height=spec.height,
        total_material=total or 1.0,
        royal_types=frozenset(spec.royal_types),
        centre=((spec.width - 1) / 2.0, (spec.height - 1) / 2.0),
        diameter=float(max(spec.width, spec.height)) or 1.0,
        territory_win_at=territory.win_at if territory else 0,
        has_territory=territory is not None,
        has_scoring=scoring is not None,
        score_ceiling=_score_ceiling(spec, base, counts),
        max_plies=spec.max_plies,
        check_limit=spec.check_limit,
        extinction_types=extinction_types,
        extinction_initial=tuple(counts[t] for t in extinction_types),
        extinction_own=_extinction_own(spec),
        has_drops=spec.drops is not None,
        promotable=frozenset(promotable),
        promotion_squares=(frozenset(promotion_squares[0]), frozenset(promotion_squares[1])),
        reach_types=frozenset(reach_types),
        reach_squares=(frozenset(reach_squares[0]), frozenset(reach_squares[1])),
        goal_distance=spec.goal_distance,
        goal_diameter=float(spec.goal_diameter or 1),
        live_prey=spec.capture_allowed,
        prey_pool=base.prey,
        has_capture_matrix=spec.has_capture_matrix,
        initial_counts=tuple(counts),
    )


def _score_ceiling(spec: CompiledSpec, base: Evaluator, counts: list[int]) -> float:
    """The largest a scoring rule's count could plausibly be, for normalisation.

    Not the true maximum -- that would need a search -- but a bound derived from the
    opening position, which is what makes it a fact about the rules rather than about a
    game. Floored at 1.0 so the division is always safe.
    """
    scoring = spec.scoring
    if scoring is None:
        return 1.0
    if scoring.count == "material":
        return max(sum(spec.piece_values[t] * counts[t] for t in range(spec.n_types)), 1.0)
    if scoring.count == "territory":
        return float(spec.n_squares)
    if scoring.count == "points":
        # Every bounty standing on the board at ply zero: what the two tallies can come
        # to between them, which is what a points gap is a share of. `counts` would be
        # the wrong denominator twice over -- it counts pieces where the rule counts
        # money, and it no longer counts the wild pieces that are the only thing paying.
        # Read off the opening board rather than off `npcs.bounty`, because a bounty is
        # stated per *route* and a route is several squares. `Tables::npc_bounty_total`
        # is the same sum in the same ascending order.
        assert spec.npcs is not None  # the validator refuses "points" without one
        return max(
            sum(
                spec.npcs.bounty[square]
                for square in range(spec.n_squares)
                if spec.initial_board[square]
                and (spec.initial_board[square] - 1) // spec.side_stride == WILD
            ),
            1.0,
        )
    if scoring.count == "region" and scoring.squares is not None:
        return float(max(len(scoring.squares[0]), len(scoring.squares[1]), 1))
    return float(max(sum(counts), 1))


def builtin_vector(context: FeatureContext) -> tuple[float, ...]:
    """The weights that reproduce material-and-mobility on this ruleset.

    Every set of heuristics carries this at index 0, and it is never proposed by
    anything. It anchors one point per ruleset that owes nothing to a sample, which is
    what keeps a corpus interpretable even if every proposed heuristic turns out to be
    junk.

    The weights are ruleset-dependent because the normalisation is: ``material`` is
    divided by the opening material and ``mobility`` by the square count, so multiplying
    back by exactly those constants recovers ``material + 0.05 * mobility``. Recovers
    its *decisions*, not its bits -- dividing and multiplying by the same float is not
    the identity -- which is why the built-in agents keep their own code path and this
    vector is an anchor rather than a reproduction.

    **Where the ruleset has hands, the hand is material too** (``v11`` A6): the
    built-in ``material`` counts a piece in hand at its full value, and this
    vocabulary's ``material`` column is the board alone, so the hand's own column
    carries the same constant, signed as the built-in signs material. Only where the
    ruleset declares ``drops``, so every other ruleset's vector is the one it always
    was, to the bit. Before A6 this vector was Q228's ``h = 0`` arm, and it is now its
    ``h = 1`` arm.

    Returned at unit length. The ratio between the weights is the whole content --
    a search's argmax is invariant under scaling every weight by a positive constant --
    and the raw constants run to tens on a large board, which would trip the sanity
    bound a *submitted* vector is held to. That bound is there to catch a proposal
    typing 1e9, not to say anything about scale.
    """
    weights = [0.0] * len(VOCABULARY)
    weights[VOCABULARY.index("material")] = context.total_material
    weights[VOCABULARY.index("mobility")] = 0.05 * context.n_squares
    if context.has_drops:
        weights[VOCABULARY.index("hand_material")] = (
            context.base.material_sign * context.total_material
        )
    # `math.fsum`, not `sum`: CPython 3.12 made float `sum` compensated, so on a hand
    # ruleset (three non-zero terms) 3.11 summed shogi's squares one ULP apart and wrote
    # an anchor that no longer matched its committed store (Q447, Q448). `fsum` is
    # correctly rounded on every version, and reproduces every committed anchor.
    norm = math.sqrt(math.fsum(w * w for w in weights)) or 1.0
    return tuple(w / norm for w in weights)


class FeatureEvaluator:
    """``dot(weights, features(position))``, skipping the terms weighted at zero.

    Skipping rather than multiplying by zero and summing is not an optimisation. Float
    addition is not associative, so a term contributing ``0.0`` still perturbs the sum;
    two heuristics that differ only in a term neither of them uses would then be
    different players. Skipping makes a zero weight mean *absent*, which is what the
    fixed-length vector promises.

    The interface matches ``agents.evaluation.Evaluator`` because the search reads both
    ``evaluate`` and ``values`` -- the latter for capture ordering, which is not the
    objective and therefore keeps the spec-derived values whatever the heuristic says.

    **A player may also say what the pieces are worth.** Three terms read that opinion --
    ``material``, ``prey_weight`` and ``hand_material``, the three whose subject *is* the
    valuation -- and everything else keeps the derived table, each for a stated reason:

    * ``ctx.total_material`` normalises ``material``, so an override that moved it would
      make doubling every value a no-op **and** silently rescale every other term in the
      vocabulary relative to material.
    * ``self.values`` orders captures. Ordering is not the objective; if it moved, the
      node budget would buy different moves and an arm would differ in two things at
      once.

    That split is what ``PieceValues`` exists to keep straight, and a type alone does not
    prove a call site chose right -- ``tests/test_heuristics.py`` asserts the two derived
    readers never see the override.
    """

    __slots__ = ("context", "weights", "values", "piece_values", "_objective", "_active")

    def __init__(
        self,
        context: FeatureContext,
        weights: tuple[float, ...],
        values: PieceValues | None = None,
    ) -> None:
        if len(weights) != len(VOCABULARY):
            raise ValueError(
                f"a heuristic has {len(VOCABULARY)} weights under vocabulary "
                f"{VOCABULARY_VERSION}, got {len(weights)}"
            )
        self.context = context
        self.weights = weights
        self.piece_values = values or PieceValues.derived_only(context.base.values)
        # Capture ordering, and it is the derived table by construction.
        self.values = self.piece_values.derived
        # The evaluator the three valuation terms read. The *same object* where there is
        # no override, so that case runs the arithmetic it always ran rather than an
        # equal copy of it -- ``material`` has three branches and float addition is not
        # associative.
        self._objective = (
            context.base
            if self.piece_values.objective is self.piece_values.derived
            else replace(context.base, values=self.piece_values.objective)
        )
        self._active = tuple(
            (index, weights[index]) for index in range(len(VOCABULARY)) if weights[index] != 0.0
        )

    @classmethod
    def for_engine(
        cls,
        engine: Engine,
        weights: tuple[float, ...],
        values: tuple[float, ...] = (),
    ) -> FeatureEvaluator:
        context = feature_context(engine.spec)
        return cls(
            context,
            weights,
            PieceValues.of(context.base.values, values, engine.spec.royal_types),
        )

    def material(self, state: GameState, player: int) -> float:
        """Kept for the callers that want raw material rather than the objective."""
        return self.context.base.material(state, player)

    def evaluate(
        self, engine: Engine, state: GameState, player: int, move_count: int | None = None
    ) -> float:
        context = self.context
        score = 0.0
        for index, weight in self._active:
            score += weight * _TERMS[index](self, engine, state, player, move_count, context)
        return score

    def features(
        self, engine: Engine, state: GameState, player: int, move_count: int | None = None
    ) -> tuple[float, ...]:
        """Every term, whatever this instance's weights say.

        Deliberately ignores ``_active``: a reader of single terms (the oracle tests in
        `tests/test_backends.py`) needs the whole vocabulary read off a position,
        including the terms the *current* weights zero out. Nothing in a search calls
        this -- ``evaluate`` is still the dot product with the zeros skipped, so a weight
        of zero still means absent where it is load-bearing.
        """
        context = self.context
        return tuple(
            term(self, engine, state, player, move_count, context) for term in _TERMS
        )

    # -- the terms ---------------------------------------------------------------
    #
    # Each takes the same arguments and returns a signed, normalised float. The
    # uniformity is what lets `_TERMS` be a table indexed by position in `VOCABULARY`,
    # so adding a term is one row here and one row there and cannot get out of step.

    def _material(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """The valuation this player holds, over the material the rules imply.

        The numerator moves with an override and the denominator does not, and that
        asymmetry is the point: ``total_material`` is what puts every term in the
        vocabulary on one scale, so a player able to move it would be changing what its
        other fourteen weights mean.
        """
        return self._objective.board_material(state.board, player) / ctx.total_material

    def _prey_weight(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """Material re-valued by what each piece may capture *on the board right now*.

        Zero on any ruleset without a capture matrix, so this is inert everywhere the
        question does not arise. Where it does, it is the correction the RPS designer's
        own complaint names: a rock in a game with no scissors left is not worth what a
        rock was worth at the start.

        Reported as a *difference* from the static valuation, so weighting it does not
        double-count material -- a heuristic that wants purely relational values sets
        ``material`` and ``prey_weight`` to the same number.

        A wild piece is in neither army, in both passes: it is nobody's material, and it
        is not part of the enemy army the share is a share *of*. Nor is a piece **nobody
        may capture**: the share is over the enemy's *prey*, not over its pieces, and a
        type no piece able to capture is permitted to take can never be won by anyone.
        Counting one made placing it pay the placer a share of the opponent's whole army
        -- +3.1954 on ``archive/stones``, measured -- and the pool is ``ctx.prey_pool``,
        which is the built-in evaluator's own. ``Evaluator._live_values`` says the same
        thing about the same quantity, and carries the argument.
        """
        if not ctx.has_capture_matrix:
            return 0.0
        n_types = ctx.n_types
        board = state.board[: ctx.n_squares]

        # One pass for the live army, by owner and type, so the valuation below is a
        # second pass and not a scan per piece.
        live = [[0] * n_types, [0] * n_types]
        for value in board:
            if value:
                owner, piece_type = (value - 1) // ctx.side_stride, (value - 1) % n_types
                if owner == WILD:
                    continue
                live[owner][piece_type] += 1
        pool = ctx.prey_pool
        army = [
            sum(n for v, n in enumerate(live[0]) if pool[v]),
            sum(n for v, n in enumerate(live[1]) if pool[v]),
        ]

        # The live share, per (attacker type, enemy side), derived once.
        share = [[1.0] * n_types, [1.0] * n_types]
        for side in (0, 1):
            enemy = live[1 - side]
            total_enemy = army[1 - side]
            if total_enemy == 0:
                continue
            for attacker in range(n_types):
                allowed = sum(
                    enemy[victim]
                    for victim in range(n_types)
                    if ctx.live_prey[attacker][victim] and pool[victim]
                )
                share[side][attacker] = max(allowed / total_enemy, NO_PREY_FLOOR)

        relational = 0.0
        static = 0.0
        for value in board:
            if value == 0:
                continue
            owner, piece_type = (value - 1) // ctx.side_stride, (value - 1) % n_types
            if owner == WILD:
                continue
            # The override is what the live share scales, not the derived table: the
            # share is a correction *to a valuation*, so it applies to whichever
            # valuation this player holds.
            worth = self._objective.values[piece_type]
            live_worth = worth * share[owner][piece_type]
            if owner == player:
                relational += live_worth
                static += worth
            else:
                relational -= live_worth
                static -= worth
        return (relational - static) / ctx.total_material

    def _mobility(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        if move_count is None:
            move_count = len(engine.legal_moves(state))
        signed = move_count if state.side_to_move == player else -move_count
        return signed / ctx.n_squares

    def _advance(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """Progress along the axis the spec itself calls forward.

        Not chess knowledge: `setup.orientation` and the `+y forward` convention that
        `symmetry: lateral` is defined against are declared by the schema, so "how far
        my army has travelled towards the far side" reads a rule. Royals are excluded
        because a royal advancing is a different idea from an army advancing, and a
        ruleset won by getting a royal somewhere has `reach_progress` for that.

        **A wild piece is excluded from the numerator and from the denominator both.**
        It has no forward: a boar walks a fixed cycle stated at compile time, so it makes
        no progress towards anybody's far side and never will. Leaving it in the average
        would have divided a two-army difference by a count of three owners' pieces,
        which shrinks the term by a constant that says nothing about the position. The
        old code did worse than that -- ``owner == player`` is false for a wild piece
        whoever is asking, so a boar was subtracted from *both* players and the term
        stopped being antisymmetric.
        """
        if ctx.height < 2:
            return 0.0
        n_types = ctx.n_types
        width = ctx.width
        span = ctx.height - 1
        total = 0.0
        count = 0
        for square, value in enumerate(state.board[: ctx.n_squares]):
            if value == 0:
                continue
            owner, piece_type = (value - 1) // ctx.side_stride, (value - 1) % n_types
            if owner == WILD or piece_type in ctx.royal_types:
                continue
            rank = square // width
            progress = rank / span if owner == 0 else (span - rank) / span
            total += progress if owner == player else -progress
            count += 1
        return total / count if count else 0.0

    def _centrality(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """How near the middle each army stands, mine minus theirs.

        A wild piece stands in neither army and is left out of both halves of the
        subtraction and out of the average. A boar in the middle is not *my* piece in
        the middle, and it is in the way of both players equally -- what it costs each
        of them is that it blocks and it threatens, which ``mobility`` and
        ``royal_exposure`` read by asking the engine rather than by reading the board
        here. Counted as the opponent's, which is what the two-row table did, a boar
        parked on the centre made both players look worse placed than they were.
        """
        width = ctx.width
        cx, cy = ctx.centre
        total = 0.0
        count = 0
        for square, value in enumerate(state.board[: ctx.n_squares]):
            if value == 0:
                continue
            owner = (value - 1) // ctx.side_stride
            if owner == WILD:
                continue
            rank, file = divmod(square, width)
            distance = max(abs(file - cx), abs(rank - cy)) / ctx.diameter
            near = 1.0 - distance
            total += near if owner == player else -near
            count += 1
        return total / count if count else 0.0

    def _royal_exposure(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        if not ctx.royal_types:
            return 0.0
        return float(engine.in_check(state, 1 - player)) - float(engine.in_check(state, player))

    def _territory_margin(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        if not ctx.has_territory or not state.colour:
            return 0.0
        mine = state.colour.count(player + 1)
        theirs = state.colour.count(2 - player)
        denominator = ctx.territory_win_at or ctx.n_squares
        return _clamp((mine - theirs) / denominator)

    def _score_margin(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        if not ctx.has_scoring:
            return 0.0
        return _clamp(_score_gap(state, ctx, player) / ctx.score_ceiling)

    def _score_urgency(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """The same gap, but worth nothing at the start and everything at the clock.

        A scoring rule only fires when the clock runs out, so a term that is constant
        from ply zero says a lead matters as much on move one as on the last move. It
        does not, and this is the term that makes a game decided on a count play
        differently from the same game decided some other way.
        """
        if not ctx.has_scoring or ctx.max_plies <= 0:
            return 0.0
        urgency = min(1.0, state.ply / ctx.max_plies)
        return _clamp(_score_gap(state, ctx, player) / ctx.score_ceiling) * urgency

    def _check_margin(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        if ctx.check_limit <= 0:
            return 0.0
        return _clamp((state.checks[player] - state.checks[1 - player]) / ctx.check_limit)

    def _extinction_margin(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """How much of the endangered kind each side has left, worst case first.

        The minimum across named types rather than the sum: a rule that ends the game
        when *any* named type runs out is lost by the scarcest one, and a sum would let
        a surplus of one kind hide the extinction of another.
        """
        if not ctx.extinction_types:
            return 0.0
        if ctx.extinction_own[0] or ctx.extinction_own[1]:
            # Schema 27 (Q936): the armies count different kinds, so one side's count of a
            # type against the other's means nothing. Each side's worst own kind against
            # its own opening, and the margin is the difference (`features.rs`).
            def own_worst(side: int) -> float:
                worst = 1.0
                for piece_type, started in ctx.extinction_own[side]:
                    left = _live_of(state.board, ctx, piece_type, side)
                    worst = min(worst, left / max(started, 1))
                return worst

            return _clamp(own_worst(player) - own_worst(1 - player))
        worst = 1.0
        for index, piece_type in enumerate(ctx.extinction_types):
            initial = ctx.extinction_initial[index] or 1
            mine = _live_of(state.board, ctx, piece_type, player)
            theirs = _live_of(state.board, ctx, piece_type, 1 - player)
            worst = min(worst, (mine - theirs) / initial)
        return _clamp(worst)

    def _hand_material(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """The hands over the opening material: the half of the built-in ``material``
        that this vocabulary's board-only ``material`` column leaves out.

        ``Evaluator.hand_value`` is the one definition, so ``builtin_vector``'s weight on
        this column is exactly the built-in's hand (``v11`` A6).
        """
        if not ctx.has_drops or not state.hands:
            return 0.0
        return self._objective.hand_value(state.hands, player) / ctx.total_material

    def _promotion_occupancy(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """What share of each side's promotable pieces stand where they would promote.

        A wild piece is excluded by the ``owner != side`` test below and needs nothing
        more: a boar walks its route and never promotes, whatever square it is standing
        on and whatever its type does in a player's hands.
        """
        if not ctx.promotable:
            return 0.0
        n_types = ctx.n_types
        share = [0.0, 0.0]
        for side in (0, 1):
            eligible = 0
            standing = 0
            zone = ctx.promotion_squares[side]
            for square, value in enumerate(state.board[: ctx.n_squares]):
                if value == 0:
                    continue
                owner, piece_type = (value - 1) // ctx.side_stride, (value - 1) % n_types
                if owner != side or piece_type not in ctx.promotable:
                    continue
                eligible += 1
                if square in zone:
                    standing += 1
            share[side] = standing / eligible if eligible else 0.0
        return share[player] - share[1 - player]

    def _reach_progress(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """How close each side's nearest eligible piece is to a square that wins.

        Distance **through the piece's own movement**, read out of the table
        `engine/compile.py` builds for exactly this: a breadth-first search backwards
        from the winning squares over the compiled move tables. It replaces the Chebyshev
        distance this term shipped with, which was right for a king-mover, optimistic for
        a slider and pessimistic for a leaper -- and wrong in a way no amount of care
        about the arithmetic could fix, because a knight's distance to a rank is not the
        number of ranks and a piece that only moves forward may not be able to come back
        at all.

        Normalised by the board's diameter *for these pieces* rather than by its
        geometry, which is the only honest denominator for a distance measured in moves.
        A piece that cannot reach a winning square at all reads as the full diameter,
        which is the same thing "as far away as anything gets" means here.

        A wild piece is excluded by the ``owner != side`` test below: it wins nothing by
        arriving anywhere, so it is not racing for either side.
        """
        if not ctx.reach_types or not ctx.goal_distance:
            return 0.0
        n_types = ctx.n_types
        near = [0.0, 0.0]
        for side in (0, 1):
            if not ctx.reach_squares[side]:
                continue
            best = ctx.goal_diameter
            for square, value in enumerate(state.board[: ctx.n_squares]):
                if value == 0:
                    continue
                owner, piece_type = (value - 1) // ctx.side_stride, (value - 1) % n_types
                if owner != side or piece_type not in ctx.reach_types:
                    continue
                distance = ctx.goal_distance[side][piece_type][square]
                if distance != UNREACHABLE and distance < best:
                    best = float(distance)
            near[side] = 1.0 - min(1.0, best / ctx.goal_diameter)
        return near[player] - near[1 - player]

    def _facing_alignment(self, engine, state, player, move_count, ctx) -> float:  # type: ignore[no-untyped-def]
        """How far each turnable piece is turned from its own side's forward, mine minus theirs.

        The first term in the vocabulary that varies with a piece's *frame* rather than
        with its square or its type, and the whole of `docs/bugs.md`'s facing entry: a
        pawn pointing away from the enemy was priced exactly as one pointing at them, so
        every other term counted material as though every piece were useful, and a
        rotation was a move that changed nothing an evaluator could see.

        **Not chess knowledge, and the argument is `advance`'s.** `piece.facing` declares
        how many orientations a type has, `FACING_NAMES` fixes their order as *forward,
        right, backward, left* clockwise, and `setup.orientation` declares which way
        forward is for each army -- so "how much of its own forward is this piece still
        pointing" reads three declared rules and no preference. Whether pointing forward
        is *good* is what the weight is for, and a stated vector is entitled to answer
        with a negative number.

        Exactly `+1.0`, `0.0` or `-1.0` per piece, read out of `_ALIGNMENT` by quarter
        turns, so the sum is an exact integer however many pieces there are and the term
        is one division. That is what keeps it byte-identical across the boundary without
        an argument about float order, and it is the property `dispersion` did not have.

        **Three exclusions, each a fact about the rules rather than about the position.**
        A ruleset where nothing turns returns `0.0` before any arithmetic -- the term is
        then constant, and a constant is not a feature. A *type* with one orientation is
        in neither the sum nor the count, for the same reason: it has no frame to read,
        and counting it would add a fixed `+1` per piece that varies with nothing. A
        facing no move turns (schema 25) is out on the same argument: its frame is the
        setup's, so it prices where a piece was placed rather than a choice anyone makes. And a
        wild piece is in neither, on `advance`'s argument -- it walks a fixed route, so
        it is turned towards nobody's far side, and `owner == player` is false for it
        whoever asks, which would cost both players the same piece and break antisymmetry.

        The decode is the one this file's `side_stride` comment warns about, spelled out
        because this is the term that would be wrong if it were taken for granted:
        `(value - 1) // n_types` is the **frame**, the owner is `frame // n_facings` and
        the facing is `frame % n_facings`. Reading the frame as the owner prices the
        wrong army, and only ever on a ruleset that faces.
        """
        if not ctx.has_facing:
            return 0.0
        n_types = ctx.n_types
        n_facings = ctx.n_facings
        step = 4 // n_facings
        total = 0.0
        count = 0
        for value in state.board[: ctx.n_squares]:
            if value == 0:
                continue
            frame, piece_type = (value - 1) // n_types, (value - 1) % n_types
            owner = frame // n_facings
            if owner == WILD or ctx.facing_axes[piece_type] < 2:
                continue
            aligned = _ALIGNMENT[(frame % n_facings) * step]
            total += aligned if owner == player else -aligned
            count += 1
        return total / count if count else 0.0


def _clamp(value: float) -> float:
    return -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)


def _extinction_own(
    spec: CompiledSpec,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Each side's ``"own"`` extinction types with its opening count of each (schema 27)."""
    rule = spec.extinction
    if rule is None or not rule.own:
        return (), ()
    stride = spec.side_stride
    lists = []
    for player in (0, 1):
        low = player * stride + 1
        lists.append(tuple(
            (piece_type, sum(
                1 for value in spec.initial_board
                if low <= value < low + stride and (value - 1) % spec.n_types == piece_type
            ))
            for piece_type in rule.side_types[player]
        ))
    return lists[0], lists[1]


def _live_of(board: bytes, ctx: FeatureContext, piece_type: int, side: int) -> int:
    # Every orientation of the type: "how many of these are left" is a question about
    # the piece, not about which way it points.
    low = side * ctx.side_stride + 1
    high = low + ctx.side_stride
    return sum(
        1
        for value in board[: ctx.n_squares]
        if low <= value < high and (value - 1) % ctx.n_types == piece_type
    )


# Indexed by position in `VOCABULARY`. A term added to one and not the other fails at
# import, which is the cheapest place for that mistake to surface.
_TERMS = (
    FeatureEvaluator._material,
    FeatureEvaluator._prey_weight,
    FeatureEvaluator._mobility,
    FeatureEvaluator._advance,
    FeatureEvaluator._centrality,
    FeatureEvaluator._royal_exposure,
    FeatureEvaluator._territory_margin,
    FeatureEvaluator._score_margin,
    FeatureEvaluator._score_urgency,
    FeatureEvaluator._check_margin,
    FeatureEvaluator._extinction_margin,
    FeatureEvaluator._hand_material,
    FeatureEvaluator._promotion_occupancy,
    FeatureEvaluator._reach_progress,
    FeatureEvaluator._facing_alignment,
)

assert len(_TERMS) == len(VOCABULARY), "every vocabulary term needs exactly one function"
assert set(GLOSSES) == set(VOCABULARY), "every vocabulary term needs exactly one gloss"


def _score_gap(state: GameState, ctx: FeatureContext, player: int) -> float:
    """One side's scoring count minus the other's.

    Deliberately reads the *declared* piece values rather than the evaluator's blended
    ones where the count is material: a scoring rule is an objective fact about who won,
    and two agents holding different opinions about a piece must not be able to disagree
    about it. That is the same argument `rules.scoring` itself is built on.

    **A wild piece scores for nobody, and under `points` nothing on the board scores at
    all.** The tally is what the rule counts, and what it counts -- the wild pieces a
    player has taken -- is gone from the board by the time the question is asked. Under
    every other count a boar is neither my piece nor my opponent's, so it goes into
    neither column; before schema 9 there were only two columns and it raised.
    """
    scoring = ctx.base.spec.scoring
    if scoring is None:
        return 0.0
    counts = [0.0, 0.0]
    if scoring.count == "territory":
        if not state.colour:
            return 0.0
        counts = [float(state.colour.count(1)), float(state.colour.count(2))]
    elif scoring.count == "points":
        # `features.rs::score_gap` reads `game.points` for the same rule.
        counts = [float(state.points[0]), float(state.points[1])]
    elif scoring.count == "region" and scoring.squares is not None:
        for side in (0, 1):
            counts[side] = float(
                sum(
                    1
                    for square in scoring.squares[side]
                    if state.board[square]
                    and (state.board[square] - 1) // ctx.side_stride == side
                )
            )
    else:
        material = scoring.count == "material"
        n_types = ctx.n_types
        for value in state.board[: ctx.n_squares]:
            if value == 0:
                continue
            owner, piece_type = (value - 1) // ctx.side_stride, (value - 1) % n_types
            if owner == WILD:
                continue
            counts[owner] += ctx.base.spec.piece_values[piece_type] if material else 1.0
    return counts[player] - counts[1 - player]
