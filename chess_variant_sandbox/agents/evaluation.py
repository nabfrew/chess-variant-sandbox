"""Position evaluation derived from the rules rather than from chess knowledge.

The interesting decision here is where piece values come from. A spec *declares* a
value per piece, but for an LLM-proposed ruleset that number is an unvalidated
opinion -- the model writes ``"value": 9`` because the piece feels queen-like, not
because anything measured it. Declared values are therefore used only as a weak prior;
the working value is derived from average mobility, which is a fact about the rules.

This matters beyond playing strength. Every agent in the tournament shares this
evaluator, so if it carried chess assumptions, "does deeper search win?" would really
be measuring "does this ruleset resemble chess?" -- and that question is already the
index's job, asked honestly elsewhere.

Two things here are *not* facts about the rules alone, and both arrived with stage 17A
after the static version was measured and found blind:

**What a piece may take depends on what is left.** The prey share is taken over the
live enemy *prey* rather than the opening army, so a rock in a game whose scissors are
all gone is worth its square and not a queen -- and a piece nobody may capture is in
neither half of that fraction, because it is not enemy material anyone can win. That
factor cannot be folded into the values handed across the boundary, because it changes
every capture -- so the *reach* half is derived once here and travels, and the share is
applied inside ``evaluate`` on both sides of the boundary. What is duplicated is one
multiplication; ``tests/test_backends.py`` is what holds the two to it.

**A game won by something other than capture needs a term for the thing it is won by.**
Material and mobility read checkmate and royal capture and nothing else. Every goal term
lands in ``[-1, 1]`` as a share of the way to winning, and they share one weight stated
in units of the mean non-royal piece value -- one number, not one per rule, and not a
number fitted per ruleset.

**A wild piece is nobody's material** (schema 9). Every sum below is a *difference*
between the two players, and its defining property is antisymmetry: ``f(board, 0)`` is
``-f(board, 1)``. A third owner counted as "not this player" breaks that -- it makes
both players poorer by the same boar -- so a wild piece enters neither side of the
subtraction and neither the enemy-army denominators the prey share is taken over. What
a boar is worth to a player is its **bounty**, which is a tally on the state and is read
by ``score_for`` under ``scoring.count: "points"``; what it costs a player is that it
blocks and it threatens, which the evaluator sees through mobility and through
``in_check`` -- both of which ask the engine rather than reading the board here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import UNREACHABLE, WILD, CompiledSpec, ReachRule

# Weight on the declared value when blending with the measured one. Low on purpose:
# design intent is worth something, but a number nobody checked is worth less.
DECLARED_WEIGHT = 0.25

# What a piece that may capture nothing is worth, as a share of full reach. Not zero:
# it still takes up a square and blocks a line, and an agent that valued it at nothing
# would give it away for free.
NO_PREY_FLOOR = 0.1

# What a piece drawn from a stock that is never spent is worth to the *material* term.
#
# Zero, and the argument is what material is. Material is a scarce quantity one player
# can be ahead on; "I have a stone and you do not" is worth nothing against an opponent
# who may place one whenever they like. So an inexhaustible type enters neither side of
# the subtraction, for the same reason a wild piece does not: counting it would make the
# difference say something the game does not.
#
# Any other number is a pump. Until `v11` A6 the material term read the *board* and not
# the hand, so a drop moved a piece from a place that was not counted to a place that
# was, and a type priced at v paid the mover v for the act of placing one. With an
# unlimited stock that was v per ply, for ever, and a searching agent that is paid every
# ply to do one thing does that thing: `archive/stones` (chess plus an unlimited supply
# of moveless stones) put 0.5198 of its moves into drops, ran 204 of 256 games into the
# ply limit, and indexed 0.3144 against chess's 0.7943 -- while `archive/three-stones`,
# the same game with `stock: 3`, indexed 0.8058. That comparison is about the price
# alone, and read as the general claim it first sounded like -- *the rule was never the
# problem* -- it does not survive Q131: an unlimited stock of a genuinely inert piece is
# a second, independent defect, a stalemate race whatever the piece is worth.
# `docs/reference/agents-and-eval-record.md`, "The fix stands; one sentence around it
# does not", has the account.
#
# Counting the hand (A6) closed the finite case -- a drop now takes from the hand what
# it adds to the board -- and leaves this one standing: an endless slot never empties,
# so its drop still adds v to the board and takes nothing from the hand. So the price
# stays zero, which is what this constant alone was ever going to fix; the denominator
# is the prey pool, not the enemy army, closed a third and separate defect in
# `_live_values` (Q137, see `prey_types`). It says nothing about what a moveless piece
# with a *finite* stock is worth, which is still the fallback described on
# `_measured_values`. See `docs/bugs.md`.
INEXHAUSTIBLE_VALUE = 0.0

# What one whole win by a non-capture ending is worth, in mean non-royal pieces.
#
# A5's single constant: one number for the whole goal family, so "ten percent closer to
# a territorial win" is worth a stated fraction of a piece and means the same thing on
# a board of any size. Deliberately *not* one weight per rule and *not* fitted per
# ruleset -- the corpus that exercises these rules is six games by one designer, and it
# validates rather than anchors.
#
# Sixteen, by a stated rule rather than an argmax: **the largest weight that is not
# significantly worse than the field on any ruleset, at any budget tested.** Thirty-two
# is better on average and loses significantly on `rps-v4`, whose colouring can be
# repainted -- which is exactly the ruleset that punishes paying material for a claim
# you cannot keep. Largest rather than safest because the quantity being weighted is
# progress towards *winning*, and a win is worth more than any amount of material; the
# only thing holding the number down is the trade it would talk an agent into. Swept at
# three budgets over the three rulesets that declare a goal rule; the table is in
# `docs/reference/stage-17-measurements.md`.
GOAL_PIECES = 16.0

# The share of that weight a *distance* term gets: the race to a ``reach`` square, the one
# goal term that measures how far away a win is rather than counting how much of it is
# done. A constant per *shape* of quantity -- the one stage 17 deferred for want of a
# second ruleset -- and still never one per rule or per ruleset.
#
# One eighth, so two mean pieces for a whole race at the default. Stage 17 measured a
# distance term wanting a constant an order of magnitude below a counting term's: on
# ``racingkings`` 2 was the best of 2, 4, 8 and 16 and 16 was fatal. That term was then
# gated off every committed ruleset, and ``intransitive`` arrived later with a race it
# prices at 16: one step of the lead piece worth six pieces. Q462 measured the same
# 200,000-node search at 2 beating itself at 16 0.742 (+184 Elo), and the user chose to
# fix the rated players rather than only the site's (Q469,
# ``docs/reference/weak-player-intransitive.md`` sections 6 and 9).
#
# A share of ``goal`` rather than a second option, so ``goal=0`` still turns every goal
# term off. A power of two on purpose: at the default it is bit-for-bit the weight
# ``goal=2`` gave the race before the share existed, which is how the change was checked
# move for move. ``rust/src/eval.rs::RACE_SHARE`` is the core's copy, exported as
# ``_core.RACE_SHARE`` and held to this one by ``tests/test_backends.py``.
RACE_SHARE = 0.125

# The immunity term's weight (Q466, F2): a piece no enemy type present may ever capture
# under the capture relation counts one mean non-royal piece more, and the nearest such
# racer adds its proximity to the goal on the same scale. Fixed by its unit, not by a
# sweep (``docs/reference/immunity-term.md`` section 1). The default of every player on
# the built-in objective since Q471 (the user's decision C, 2026-09-25): the measuring
# players and so the site's hardest. ``immune=0`` spells the player without it. Zero by
# construction where ``immunity_threats`` finds the term undefined, which is every
# ruleset without a capture matrix. ``rust/src/eval.rs::IMMUNE_PIECES`` is the core's
# copy, exported as ``_core.IMMUNE_PIECES`` and held to this one by
# ``tests/test_immunity.py``.
IMMUNE_PIECES = 1.0

ROYAL_VALUE = 0.0
WIN_SCORE = 1_000_000.0


@dataclass(frozen=True)
class PieceValues:
    """The two value tables an objective needs, so neither can be passed as the other.

    They are the same shape and mean different things, which is the whole problem this
    type exists for. ``derived`` is what the rules imply -- reach on an empty board,
    blended with the declared value -- and it normalises material, orders captures and
    fixes the goal weight. ``objective`` is what *this player* thinks a piece is worth,
    and it is read by exactly three terms. Two bare ``tuple[float, ...]`` at a call site
    would swap silently and the symptom would be a slightly different player rather than
    a failure, so nothing downstream takes a bare value table again: a caller has to
    name which one it wants.

    An override arrives on whatever scale its producer used and leaves on the derived
    table's, by the rule in ``rescaled``. That is not cosmetic either; see there.
    """

    derived: tuple[float, ...]
    objective: tuple[float, ...]

    @classmethod
    def derived_only(cls, derived: tuple[float, ...]) -> PieceValues:
        """No override: both halves are the table the rules imply.

        The same object in both slots, so the built-in path runs the arithmetic it
        always ran rather than an equal copy of it.
        """
        return cls(derived=derived, objective=derived)

    @classmethod
    def of(
        cls,
        derived: tuple[float, ...],
        override: tuple[float, ...],
        royal_types: tuple[int, ...] | frozenset[int],
    ) -> PieceValues:
        """The pair for one player, with ``override`` empty meaning "no opinion".

        This is where the length is checked against the ruleset, because this is the
        first place a spec is in hand: ``Heuristic`` holds a vector and no board, so it
        can only bound the length by ``MAX_PIECE_TYPES``. A vector of the wrong length
        here is a values table written for a different game, and reinterpreting it would
        be a differently-playing agent under an unchanged name.
        """
        if not override:
            return cls.derived_only(derived)
        if len(override) != len(derived):
            raise ValueError(
                f"a piece-value override for this ruleset has {len(derived)} entries, "
                f"got {len(override)}; that table was written for a different game"
            )
        for piece_type in royal_types:
            if override[piece_type] != 0.0:
                raise ValueError(
                    f"piece type {piece_type} is royal and its value is "
                    f"{override[piece_type]}; losing a royal is terminal rather than "
                    "material, so the evaluator has no term for it at all"
                )
        if mean_abs_non_royal(override, royal_types) == 0.0:
            raise ValueError(
                "every non-royal piece is worth nothing, which is not an opinion about "
                "the pieces but the absence of one; leave the override empty instead"
            )
        return cls(derived=derived, objective=rescaled(derived, override, royal_types))


def rescaled(
    derived: tuple[float, ...],
    override: tuple[float, ...],
    royal_types: tuple[int, ...] | frozenset[int],
) -> tuple[float, ...]:
    """``override`` moved onto the derived table's scale, magnitudes and all.

    **The rule: the mean of ``|value|`` over non-royal types equals the derived table's
    mean over non-royal types.** One rule, no branches, and never a division by zero
    because an all-zero override is refused before it gets here.

    The tempting convention -- normalise to mean 1, the unit ``GOAL_PIECES`` is stated
    in -- is wrong, and wrong in a way that would survive review. ``_material`` is
    ``material(board) / total_material``, and ``total_material`` is derived and stays
    derived: xiangqi's derived table runs from 1.25 to 74, so an override at mean 1
    would make ``_material`` about a thirtieth of its usual magnitude there. Saying what
    a ``material`` weight of 1 says today would then need a weight near 30 -- past
    ``MAX_ABS_WEIGHT``, and silently rescaling every other term in the vocabulary
    relative to it.

    It also makes the browser honest: an override is a *reallocation* of a fixed budget,
    which is what a person means when they say the elephant is worth more than the
    engine thinks. Setting every slider to five times is the same opponent as leaving
    them all at one.

    Exactly the identity when ``override`` is the derived table: both means are the same
    float, so the scale is ``1.0`` and multiplying by it changes no bit. That is what
    the inertness test in ``tests/test_backends.py`` holds this to.
    """
    scale = mean_abs_non_royal(derived, royal_types) / mean_abs_non_royal(
        override, royal_types
    )
    return tuple(value * scale for value in override)


def mean_abs_non_royal(
    values: tuple[float, ...] | list[float],
    royal_types: tuple[int, ...] | frozenset[int],
) -> float:
    """The mean magnitude of a piece that is not royal: the scale an override is on.

    Magnitude rather than value, because a fitted table may price a piece negatively --
    antichess and its like -- and a mean that cancelled would rescale such a table by an
    arbitrary amount.

    **Not the same set as ``_mean_non_royal``, and deliberately.** That one is the unit
    *material* is counted in and so drops an inexhaustible type; this one is the scale an
    *override* is stored on, and an override is a player's opinion about every type it
    was handed -- including one this evaluator prices at nothing. Narrowing it here would
    move `rescaled`, which `tests/test_backends.py` pins as the identity, for a game
    nobody has written; `docs/bugs.md` records it as the same shape of hole one level
    over.

    ``rust/src/eval.rs::mean_abs_non_royal`` accumulates in the same order for the same
    reason every other sum in this file does.
    """
    total = 0.0
    count = 0
    for piece_type, value in enumerate(values):
        if piece_type in royal_types:
            continue
        total += abs(value)
        count += 1
    return total / count if count else 1.0


@dataclass(frozen=True)
class Evaluator:
    """Material, mobility and goal scoring for one compiled ruleset."""

    spec: CompiledSpec
    values: tuple[float, ...]
    mobility_weight: float = 0.05
    goal_pieces: float = GOAL_PIECES
    # Q466 (F2): the immunity term's weight in mean non-royal pieces an immune piece.
    # ``IMMUNE_PIECES`` unless a player spells another (Q471); zero is off.
    immune_pieces: float = IMMUNE_PIECES
    # The denominator of the prey share, derived rather than passed. Not a constructor
    # argument on purpose: a value table and the pool it is scaled against are one fact
    # about one ruleset, and a caller able to hand in a mismatched pair would be a
    # differently-playing agent under an unchanged name. See ``prey_types``.
    prey: tuple[bool, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prey", prey_types(self.spec))

    @classmethod
    def for_engine(
        cls,
        engine: Engine,
        mobility_weight: float = 0.05,
        goal_pieces: float = GOAL_PIECES,
        immune_pieces: float = IMMUNE_PIECES,
    ) -> Evaluator:
        return cls.for_spec(
            engine.spec,
            mobility_weight=mobility_weight,
            goal_pieces=goal_pieces,
            immune_pieces=immune_pieces,
        )

    @classmethod
    def for_spec(
        cls,
        spec: CompiledSpec,
        mobility_weight: float = 0.05,
        goal_pieces: float = GOAL_PIECES,
        immune_pieces: float = IMMUNE_PIECES,
    ) -> Evaluator:
        """The same evaluator from the tables alone, with no engine in hand.

        The Rust agents need these values handed to them before a game starts, and the
        boundary takes a compiled spec rather than an engine. Deriving them twice --
        once here and once there -- would let an agent value pieces differently from
        the agent it is rated against, so there is one derivation and it is this one.
        """
        measured = _measured_values(spec)
        declared = spec.piece_values
        endless = inexhaustible_types(spec)

        # The types material is actually measured over. Royals are out because a royal
        # is priced by the game ending and not by a number; an inexhaustible type is out
        # for the reason on ``INEXHAUSTIBLE_VALUE``. Both are then also out of the two
        # *scales* below, which is the same statement made twice: a piece that is not
        # material cannot be the unit material is counted in.
        priced = [
            i
            for i in range(spec.n_types)
            if i not in spec.royal_types and i not in endless
        ]

        reaches = [measured[i] for i in priced]
        scale = min(reaches) if reaches and min(reaches) > 0 else 1.0

        declared_priced = [declared[i] for i in priced]
        declared_scale = (
            min(v for v in declared_priced if v > 0)
            if any(v > 0 for v in declared_priced)
            else 1.0
        )

        values = []
        for piece_type in range(spec.n_types):
            if piece_type in spec.royal_types:
                values.append(ROYAL_VALUE)
                continue
            if piece_type in endless:
                values.append(INEXHAUSTIBLE_VALUE)
                continue
            blended = (1.0 - DECLARED_WEIGHT) * (measured[piece_type] / scale)
            blended += DECLARED_WEIGHT * (declared[piece_type] / declared_scale)
            values.append(blended)

        return cls(
            spec=spec,
            values=tuple(values),
            mobility_weight=mobility_weight,
            goal_pieces=goal_pieces,
            immune_pieces=immune_pieces,
        )

    @property
    def goal_weight(self) -> float:
        """The goal family's weight in material units.

        ``goal_pieces`` is stated in mean non-royal pieces so that it means the same
        thing on a board of any size and with any army; this is that constant multiplied
        out. ``rust/src/eval.rs`` derives it the same way from the same three inputs.
        """
        return self.goal_pieces * _mean_non_royal(
            self.values, self.spec.royal_types, inexhaustible_types(self.spec)
        )

    @property
    def race_weight(self) -> float:
        """The race term's weight in material units: ``RACE_SHARE`` of ``goal_weight``."""
        return self.goal_weight * RACE_SHARE

    @property
    def material_sign(self) -> float:
        """Whether having *more* material is the thing this ruleset rewards.

        ``-1.0`` where an extinction rule says that running out of **anything** is a
        **win**, which is antichess and every game like it. Stage 17B gave the evaluator
        a goal term for that rule and it was not enough: a goal term is *added* to the
        material term, so the two simply fight, and material wins. Shedding a piece in
        antichess -- which is progress towards the win condition -- scored ``-2.05``
        against an opening of ``0.00``.

        Read off the spec rather than offered as an agent option, for the reason
        ``rules.scoring`` is: which side is ahead has to be an objective fact, and two
        agents holding different opinions about it could not be compared. The rule says
        in so many words that running out wins; the evaluator is entitled to believe it.

        Deliberately narrow. It fires only for ``any_piece``, because "run out of
        *everything* and you win" is the case where losing material is the whole
        objective. A rule naming particular types is a different shape of game -- there
        the goal term is the right instrument, and inverting all of material would be
        wrong about every piece the rule does not name.
        """
        rule = self.spec.extinction
        if rule is not None and rule.any_piece and rule.result == "win":
            return -1.0
        return 1.0

    def material(self, state: GameState, player: int) -> float:
        """Signed material balance from ``player``'s point of view: the board, then the hands.

        **A piece in hand is material at its full value** (``v11`` A6, Q228). Under
        ``drops.from_captures`` a capture changes a piece's owner rather than destroying
        it, so what a hand holds is owned. Counting it at *h* times its board value
        splits the 2v a capture and its re-drop are worth between them into (1 + h)v and
        (1 - h)v. ``h = 1`` is conservation: the capture pays 2v, and the drop, which
        only moves a piece its owner already had, pays nothing. The board-only term this
        replaced was ``h = 0``, which paid the whole second v as a bribe for the act of
        dropping. Q228's paired duel measured the choice rather than arguing it: at
        6,400 nodes the objective with the hand counted at ``h = 1`` took 0.93 to 0.99 of
        the points against ``h = 0`` on all four ``from_captures`` games, and lost
        nowhere, including ``archive/three-stones`` and 400 nodes
        (``docs/reference/hand-value-duel.md``).

        The sign inverts the whole balance, hand included: where running out is the win,
        a piece held is still a piece. ``rust/src/eval.rs::material`` is the same
        function, in the same order, and a ruleset with no ``drops`` skips the hand on
        both sides, so it runs exactly the arithmetic it always did.
        """
        board = self.board_material(state.board, player)
        if self.spec.drops is None:
            return board
        hand = self.hand_value(state.hands, player)
        return board - hand if self.material_sign < 0.0 else board + hand

    def hand_value(self, hands: tuple[tuple[int, ...], ...], player: int) -> float:
        """The hands, valued off this evaluator's table: mine minus theirs.

        Off the table and not off the live prey share, because a piece in hand threatens
        nothing yet -- and because this is the quantity Q228 measured: the vocabulary's
        ``hand_material`` column is this over ``total_material``, which is what lets
        ``features.builtin_vector`` spell the built-in objective on every ruleset. An
        ``"unlimited"`` slot is priced at ``INEXHAUSTIBLE_VALUE`` and never empties, so it
        adds nothing.

        A plain loop and not ``sum``: since Python 3.12 ``sum`` of floats compensates,
        and ``rust/src/eval.rs::hand_value`` adds one term at a time. Per side, then
        signed, in that order on both sides of the boundary.
        """
        total = 0.0
        for side in (0, 1):
            worth = 0.0
            for piece_type, count in enumerate(hands[side]):
                worth += self.values[piece_type] * count
            total += worth if side == player else -worth
        return total

    def board_material(self, board: bytes, player: int) -> float:
        """Signed material on the board alone, from ``player``'s point of view.

        What ``material`` was before the hand was counted, and still the vocabulary's
        ``material`` column: the hand has a column of its own, ``hand_material``, so a
        weight vector drawn before A6 means what it meant and no vocabulary bump is owed.
        """
        spec = self.spec
        n_types = spec.n_types
        side_stride = spec.n_facings * n_types
        if self.material_sign < 0.0:
            return -self._plain_material(board, player)
        if spec.has_capture_matrix:
            return self._relational_material(board, player)
        total = 0.0
        for value in board:
            if value == 0:
                continue
            owner, piece_type = (value - 1) // side_stride, (value - 1) % n_types
            if owner == WILD:  # nobody's material; see the module docstring
                continue
            total += self.values[piece_type] if owner == player else -self.values[piece_type]
        return total

    def _plain_material(self, board: bytes, player: int) -> float:
        """Material without the relational factor, in board order.

        Separated out because the inverted case needs it and no committed ruleset both
        inverts and declares a capture matrix -- combining the two is a question nobody
        has asked yet, and guessing at it here would be inventing a rule.
        """
        n_types = self.spec.n_types
        side_stride = self.spec.n_facings * n_types
        total = 0.0
        for value in board:
            if value == 0:
                continue
            owner, piece_type = (value - 1) // side_stride, (value - 1) % n_types
            if owner == WILD:
                continue
            total += self.values[piece_type] if owner == player else -self.values[piece_type]
        return total

    def _relational_material(self, board: bytes, player: int) -> float:
        """The same sum, with each piece scaled by what it may still take.

        Two passes rather than one: the live army has to be counted before any piece can
        be valued, because my rock's worth is a function of their scissors. Reached only
        when some piece declared ``captures``, which is the six RPS probes and nothing
        else in the committed corpus.
        """
        n_types = self.spec.n_types
        side_stride = self.spec.n_facings * n_types
        scaled = self._live_values(board)
        total = 0.0
        for value in board:
            if value == 0:
                continue
            owner, piece_type = (value - 1) // side_stride, (value - 1) % n_types
            if owner == WILD:
                continue
            row = scaled[owner]
            total += row[piece_type] if owner == player else -row[piece_type]
        return total

    def _live_values(self, board: bytes) -> tuple[list[float], list[float]]:
        """``[owner][piece type]`` scaled by the share of the *live* enemy prey it may take.

        **The quantity, stated: of the enemy material that can be won at all, what
        fraction may this piece win.** Not "what fraction of the pieces my opponent has"
        -- that was the old reading and it is the defect below. The denominator is the
        *prey pool*: the enemy pieces some piece able to capture is permitted to take.
        A piece nobody may take is in neither the numerator nor the denominator, because
        it can never be won by anybody and a valuation derived from capture has nothing
        to say about it.

        That makes three exclusions of one shape, and the shape is the rule: **a
        quantity may only be moved by the things it is a quantity about.** A wild piece
        is nobody's material, so it is out of both armies (ownership). An inexhaustible
        type can never be scarce, so it is out of both sides of the subtraction
        (``INEXHAUSTIBLE_VALUE``). A piece nobody may capture can never be won, so it is
        out of the prey share (this).

        **What counting it did, measured.** ``archive/stones`` ``93b02b591c87`` is chess
        plus an endless supply of stones that no piece may take. Under the old
        denominator, one white stone took black's army from 16 pieces to 17 in every
        black attacker's share -- pawn 1.0 -> 0.9412, queen 13.3929 -> 12.6050 -- and
        white's evaluation of the opening rose by **+3.1954** with mobility zeroed, which
        is one seventeenth of black's whole army and beats every piece move on the board
        by more than three pawns. Nothing capturable had changed hands. A material
        difference that moves for a non-capture is a bounty on the act that moved it,
        which is the same argument ``INEXHAUSTIBLE_VALUE`` won on the other side of the
        subtraction, thirteen times larger and invisible from there.

        Nothing about the stones is lost by this. They still block, which is *mobility*'s
        term to read and it reads it honestly; and they are still their owner's material,
        at whatever their own value is.

        **A permission to capture is worth nothing to a piece that cannot execute one,
        and that is why the pool is not simply "a column with no true cell".** In
        ``93b02b591c87`` the stone's own row is all true -- it declares no ``captures``
        list, and absent means everything -- so ``capture_allowed[stone][stone]`` holds
        and a column test would find the stone takeable, by a stone, and change nothing
        on the one spec that motivated the fix. The pool is therefore taken over
        attackers that have capture *reach*: the same ``modes & 2`` walk over the same
        frame-0 tables that ``_measured_values`` does to price them. See ``prey_types``.

        Symmetric by construction, and floored rather than zeroed for the same reason
        the static share was: a piece with no prey left still occupies a square and
        blocks a line, and an agent that valued it at nothing would hand it over. The
        floor is now reached only where prey *exists* and this attacker may take none of
        it; an enemy with no prey at all leaves the share undefined rather than floored,
        which is the ``army == 0`` branch and is the continuous answer -- one prey piece
        this attacker may take gives 1/1, and the piece before that gave 0/0.

        A wild piece is in neither army, so it neither dilutes nor enriches the share for
        a second reason: a boar is not one of my opponent's pieces however takeable it
        is, where a stone is my opponent's piece and is takeable by nobody.
        """
        spec = self.spec
        n_types = spec.n_types
        side_stride = spec.n_facings * n_types
        counts = [[0] * n_types, [0] * n_types]
        for value in board:
            if value == 0:
                continue
            owner, piece_type = (value - 1) // side_stride, (value - 1) % n_types
            if owner == WILD:
                continue
            counts[owner][piece_type] += 1

        prey = self.prey
        scaled: list[list[float]] = [[0.0] * n_types, [0.0] * n_types]
        for owner in (0, 1):
            enemy = counts[1 - owner]
            army = sum(
                count for victim, count in enumerate(enemy) if prey[victim]
            )
            for attacker in range(n_types):
                if army == 0:
                    # Undefined rather than zero: with nothing to win, a valuation
                    # derived from capture has no correction to make and the derived
                    # table stands. Matches what the static share did with an empty
                    # opening army, and is what the ruleset with no matrix at all does.
                    share = 1.0
                else:
                    allowed = sum(
                        count
                        for victim, count in enumerate(enemy)
                        if spec.capture_allowed[attacker][victim] and prey[victim]
                    )
                    share = max(allowed / army, NO_PREY_FLOOR)
                scaled[owner][attacker] = self.values[attacker] * share
        return scaled[0], scaled[1]

    def cheapest_piece(self, board: bytes) -> float:
        """The least this evaluator would put on a piece standing on ``board``.

        The unit its scores are in, which is what anything comparing two of them needs.
        ``analysis._epsilon_for`` is the caller: "these were alternatives" means "closer
        together than the cheapest piece", and that threshold and the numbers it
        thresholds have to be the same quantity.

        They came apart once. The prey share used to be folded into ``values`` and is
        applied per position now, so ``min(values)`` stopped being the smallest thing on
        the board -- the threshold stayed put while every score it judged shrank by the
        share, and `decision_criticality` fell on four rulesets for a reason that was
        entirely about units.
        """
        if self.spec.has_capture_matrix:
            candidates = [
                value for row in self._live_values(board) for value in row if value > 0
            ]
        else:
            candidates = [value for value in self.values if value > 0]
        return min(candidates) if candidates else 1.0

    def evaluate(
        self, engine: Engine, state: GameState, player: int, move_count: int | None = None
    ) -> float:
        """Score the position for ``player``. Higher is better.

        ``move_count`` is the mover's legal move count when the caller already has it.
        Mobility is a deliberately small term: it is the one positional idea that
        generalises across arbitrary rules, but leaning on it turns every game into
        "spread out", which is not a strategy anyone finds interesting.

        The goal terms are the answer to what "spread out" was standing in for on a game
        won by anything else. They are gated on the ruleset declaring the rule, so a
        game won by checkmate is scored by exactly the arithmetic it always was.
        """
        mobility = 0
        if self.mobility_weight:
            if move_count is None:
                move_count = len(engine.legal_moves(state))
            mobility = move_count if state.side_to_move == player else -move_count
        return self._score(state, player, mobility)

    def evaluate_round(
        self, engine: Engine, state: GameState, player: int, counts: tuple[int, int]
    ) -> float:
        """``evaluate`` in a simultaneous game (schema 26, Q918b): there is no side to
        move, both players' moves are known, and mobility is mine minus theirs, both
        counted (`docs/plan-simultaneous.md` §6.2). ``counts`` is each player's number of
        legal moves on this board, in seat order. ``eval.rs::evaluate_round`` is the
        same function."""
        return self._score(state, player, counts[player] - counts[1 - player])

    def _score(self, state: GameState, player: int, mobility: int) -> float:
        score = self.material(state, player)
        if self.mobility_weight:
            score += self.mobility_weight * mobility
        if self.goal_pieces and has_goal_terms(self.spec):
            score += self.goal_weight * self.goal_progress(state, player)
            score += self.race_weight * self.race_progress(state, player)
        if self.immune_pieces and immunity_threats(self.spec) is not None:
            weight = self.immune_pieces * _mean_non_royal(
                self.values, self.spec.royal_types, inexhaustible_types(self.spec)
            )
            score += weight * self.immunity(state, player)
        return score

    def immunity(self, state: GameState, player: int) -> float:
        """Q466 (F2): mine minus theirs, of the pieces the opponent can never capture.

        Per side, ``worth + proximity``: ``worth`` is that side's material, in mean
        non-royal pieces, of the types no piece its opponent holds (on the board or in
        hand, or after any chain of promotions) may capture under the capture relation; ``proximity`` is
        ``(D - d) / D`` for its nearest such piece eligible to race under a priced
        ``reach`` rule, ``d`` off the compiled goal table and ``D`` the goal diameter,
        zero with none, signed by the rule's result. Zero where ``immunity_threats``
        finds the term undefined. ``rust/src/eval.rs::Evaluator::immunity`` is the same
        function, in the same order.
        """
        spec = self.spec
        table = immunity_threats(spec)
        if table is None:
            return 0.0
        threat, counted = table
        n_types = spec.n_types
        stride = spec.n_facings * n_types
        # The two armies only, as the core walks ``all_occupied``: a portal end's byte is in
        # the third owner's block (``WILD``), nobody's piece (Q846).
        present = [0, 0]
        for value in state.board:
            if value and (value - 1) // stride != WILD:
                present[(value - 1) // stride] |= 1 << ((value - 1) % n_types)
        if spec.drops is not None:
            for side in (0, 1):
                for piece_type, held in enumerate(state.hands[side][:n_types]):
                    if held > 0:
                        present[side] |= 1 << piece_type
        safe = [0, 0]
        for side in (0, 1):
            enemy = 1 - side
            exposed = 0
            for attacker in range(n_types):
                if present[enemy] & (1 << attacker):
                    exposed |= threat[enemy][attacker]
            safe[side] = ~exposed & counted & present[side]
        if safe == [0, 0]:
            return 0.0
        # Immune material, summed in board order as the core sums it.
        worth = [0.0, 0.0]
        for value in state.board:
            if value:
                owner = (value - 1) // stride
                piece_type = (value - 1) % n_types
                if owner != WILD and safe[owner] & (1 << piece_type):
                    worth[owner] += self.values[piece_type]
        race = [0.0, 0.0]
        reach = spec.reach
        if reach is not None and _reach_is_priceable(spec, reach):
            direction = {"win": 1.0, "draw": 0.0, "loss": -1.0}[reach.result]
            sealed = self.sealed_goals(state)
            for side in (0, 1):
                eligible = safe[side]
                if reach.piece_type >= 0:
                    eligible &= 1 << reach.piece_type
                if not eligible:
                    continue
                best = self._goal_distance_of(state, side, eligible, sealed[side])
                if best != UNREACHABLE:
                    diameter = spec.goal_diameter
                    share = (diameter - float(best)) / diameter
                    race[side] = max(_clamp(share), 0.0) * direction
        mean = _mean_non_royal(self.values, spec.royal_types, inexhaustible_types(spec))
        mine = worth[player] / mean + race[player]
        theirs = worth[1 - player] / mean + race[1 - player]
        return mine - theirs

    def goal_progress(self, state: GameState, player: int) -> float:
        """How far along ``player`` is towards the endings that are not capture and are
        *counted*.

        A signed share of the way there, per rule, each in ``[-1, 1]``. The race is a
        distance and has its own weight: ``race_progress``.
        ``rust/src/eval.rs::goal_progress`` is the same function and the two are held to
        agreeing by ``tests/test_backends.py``.
        """
        spec = self.spec
        total = 0.0

        territory = spec.territory
        if territory is not None and territory.win_at > 0:
            target = float(territory.win_at)
            mine = state.colour.count(player + 1) / target
            theirs = state.colour.count((1 - player) + 1) / target
            total += _clamp(mine - theirs)


        if spec.check_limit:
            limit = float(spec.check_limit)
            total += _clamp(state.checks[player] / limit - state.checks[1 - player] / limit)

        extinction = spec.extinction
        if extinction is not None and extinction.result != "draw":
            mine_short = self._shortfall(state.board, player)
            theirs_short = self._shortfall(state.board, 1 - player)
            # Whose shortfall is good news depends on the rule. Running out is a loss for
            # the horde and a *win* for antichess, and the difference is the whole reason
            # a material evaluator reads one and not the other.
            direction = {"win": 1.0, "loss": -1.0}[extinction.result]
            total += _clamp((theirs_short - mine_short) * -direction)

        scoring = spec.scoring
        if scoring is not None:
            mine_score = self.score_for(state, player)
            theirs_score = self.score_for(state, 1 - player)
            spread = mine_score + theirs_score
            if spread > 0.0:
                # A scoring rule only ever fires at the clock, so it is worth nothing at
                # ply zero and everything at ``max_plies``. Without the ramp it would be
                # a constant pull from the opening move, which is neither what the rule
                # says nor when it decides anything.
                clock = min(state.ply / spec.max_plies, 1.0)
                total += _clamp((mine_score - theirs_score) / spread) * clock

        return total

    def race_progress(self, state: GameState, player: int) -> float:
        """How far ahead ``player`` is in the race to a ``reach`` square.

        A signed share of the board's goal diameter in ``[-1, 1]``, or zero where no race
        is priced. The distance comes from the rules -- a breadth-first search over the
        compiled tables, done once at compile time -- because a leaper's distance to a
        rank is not the number of ranks and a forward-only piece may not be able to come
        back at all. Only where the *journey* is priced; see ``_reach_is_priceable``.
        ``rust/src/eval.rs::race_progress`` is the same function.
        """
        spec = self.spec
        reach = spec.reach
        if reach is None or not _reach_is_priceable(spec, reach):
            return 0.0
        sealed = self.sealed_goals(state)
        mine = self._nearest_to_goal(state, player, reach.piece_type, sealed[player])
        theirs = self._nearest_to_goal(state, 1 - player, reach.piece_type, sealed[1 - player])
        # Stated for the side that arrives, so a rule whose result is a *loss* for the
        # arriving side -- which the schema allows -- is a race to run away from.
        direction = {"win": 1.0, "draw": 0.0, "loss": -1.0}[reach.result]
        return _clamp((theirs - mine) / spec.goal_diameter) * direction

    def _nearest_to_goal(
        self, state: GameState, player: int, wanted: int, sealed: int = 0
    ) -> float:
        """The fewest moves any of ``player``'s eligible pieces needs to reach a goal
        square that is not ``sealed`` (Q708).

        ``goal_diameter`` where nothing can get there, so a side with no eligible piece
        left, or with every goal sealed, scores as far away as it is possible to be
        rather than as a special case. Capped there too: one open goal of several can be
        farther than the union table's diameter.

        A wild piece is excluded by the owner test in ``_goal_distance_of`` and nothing
        more is needed: a boar walks a route and wins nothing by arriving anywhere, so it
        is not a piece either side is racing.
        """
        spec = self.spec
        types = (1 << wanted) if wanted >= 0 else -1
        best = self._goal_distance_of(state, player, types, sealed)
        if best == UNREACHABLE:
            return float(spec.goal_diameter)
        return float(min(best, spec.goal_diameter))

    def sealed_goals(self, state: GameState) -> tuple[int, int]:
        """Q708 (F1): per racing side, the mask of its goal squares (by index into
        ``seal_goals``' ascending lists) held by a piece of its opponent's that it can
        never capture. ``(0, 0)`` wherever ``seal_goals`` finds the seal undefined.
        ``rust/src/eval.rs::Evaluator::sealed_goals`` is the same function.
        """
        spec = self.spec
        goals = seal_goals(spec)
        if goals is None:
            return (0, 0)
        threat = capture_threats(spec)
        n_types = spec.n_types
        stride = spec.n_facings * n_types
        board = state.board
        sealed = [0, 0]
        for side in (0, 1):
            exposed: int | None = None
            for index, square in enumerate(goals[side]):
                value = board[square]
                if not value or (value - 1) // stride != 1 - side:
                    continue
                if exposed is None:
                    exposed = 0
                    for held in board:
                        if held and (held - 1) // stride == side:
                            exposed |= threat[side][(held - 1) % n_types]
                if not exposed & (1 << ((value - 1) % n_types)):
                    sealed[side] |= 1 << index
        return (sealed[0], sealed[1])

    def _goal_distance_of(self, state: GameState, side: int, types: int, sealed: int) -> int:
        """The fewest moves any of ``side``'s pieces of the types in ``types`` needs to
        reach a goal square not in ``sealed``, or ``UNREACHABLE``: the compiled union table
        with nothing sealed, the per-goal table over the open goals otherwise."""
        spec = self.spec
        n_types = spec.n_types
        stride = spec.n_facings * n_types
        best = UNREACHABLE
        pieces = [
            (square, (value - 1) % n_types)
            for square, value in enumerate(state.board[: spec.n_squares])
            if value and (value - 1) // stride == side and types & (1 << ((value - 1) % n_types))
        ]
        if not sealed:
            for square, piece_type in pieces:
                best = min(best, spec.goal_distance[side][piece_type][square])
            return best
        goals = seal_goals(spec)
        assert goals is not None
        n_goals = len(goals[side])
        if bin(sealed).count("1") == n_goals:
            return best
        for square, piece_type in pieces:
            for goal in range(n_goals):
                if sealed & (1 << goal):
                    continue
                best = min(best, spec.goal_distance_each[side][goal][piece_type][square])
        return best

    def _shortfall(self, board: bytes, player: int) -> float:
        """How far ``player`` is towards running out, as a share of the way there.

        Zero at the opening army and one at the threshold. Counted against the *opening*
        army rather than against the threshold alone, so "half your pieces are gone" says
        the same thing in a game with forty of them and a game with four.
        """
        rule = self.spec.extinction
        assert rule is not None
        start = self._extinction_counts[player]
        room = max(0, start - rule.min_count)
        if room == 0:
            return 0.0
        return min((start - self._count_extinction(board, player)) / room, 1.0)

    def _count_extinction(self, board: bytes, player: int) -> int:
        spec = self.spec
        rule = spec.extinction
        assert rule is not None
        stride = spec.n_facings * spec.n_types
        low = player * stride + 1
        high = low + stride
        return sum(
            1
            for value in board
            if low <= value < high
            and (rule.any_piece or ((value - 1) % spec.n_types) in rule.piece_types)
        )

    @property
    def _extinction_counts(self) -> tuple[int, int]:
        """How many pieces each side started with, of the types the rule counts.

        The denominator of the shortfall term, and constant for the whole game.
        """
        return (
            self._count_extinction(self.spec.initial_board, 0),
            self._count_extinction(self.spec.initial_board, 1),
        )

    def score_for(self, state: GameState, player: int) -> float:
        """One side's score under the ruleset's scoring rule.

        The oracle's copy of ``Game::score_for``. It is duplicated for the same reason
        the whole of this module is -- the Python agents are the declared oracle for the
        Rust ones -- and it counts in board order because floating point addition is not
        associative and the two sums have to agree to the bit.
        """
        spec = self.spec
        scoring = spec.scoring
        if scoring is None:
            return 0.0
        if scoring.count == "territory":
            return float(state.colour.count(player + 1))
        # Points (schema 9) are a tally on the state, not a count of anything standing
        # on the board: what they count -- the wild pieces this player took -- is gone
        # by the time the question is asked. `Game::score_for` reads `points[player]`
        # for the same rule, and falling through to the loop below would have counted
        # this player's own army instead, which is the `pieces` rule under another name.
        if scoring.count == "points":
            return state.points[player]

        stride = spec.n_facings * spec.n_types
        low = player * stride + 1
        high = low + stride
        total = 0.0
        for square in range(spec.n_squares):
            value = state.board[square]
            if value < low or value >= high:
                continue
            if scoring.count == "material":
                total += spec.piece_values[(value - 1) % spec.n_types]
            elif scoring.count == "region":
                assert scoring.squares is not None
                total += 1.0 if square in scoring.squares[player] else 0.0
            else:
                total += 1.0
        return total


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def has_goal_terms(spec: CompiledSpec) -> bool:
    """Whether this ruleset declares any ending the evaluator has a term for.

    The gate that makes stage 17A free: a ruleset stating none of these is evaluated by
    exactly the arithmetic it always was. Deliberately narrower than ``has_v2_endings``
    -- reach, extinction and a check limit are endings the evaluator still cannot read,
    and claiming them here would cost a calibration campaign for a term that does not
    exist. ``Tables::has_goal_terms`` is the same predicate on the other side.
    """
    return bool(
        (spec.territory is not None and spec.territory.win_at > 0)
        or spec.scoring is not None
        or (spec.reach is not None and _reach_is_priceable(spec, spec.reach))
        or spec.check_limit
        or (spec.extinction is not None and spec.extinction.result != "draw")
    )


def _reach_is_priceable(spec: CompiledSpec, reach: ReachRule) -> bool:
    """Whether a ``reach`` rule is one the evaluator may put a term on.

    Two rules out, both measured rather than reasoned about, and both the same shape: a
    distance term is all reward and no risk unless the *journey* is priced.

    **``safe``.** The compiled table counts moves to a square and knows nothing about who
    is looking at it, so under ``safe`` it overstates progress systematically -- the
    search is told to race towards squares it will be refused when it arrives. On
    ``dobutsu``, the only committed ruleset with it, a term costs two of five budget steps
    of ladder monotonicity and 300 points of rating spread.

    **A royal racing piece.** ``ROYAL_VALUE`` is zero, deliberately: losing a royal is
    terminal rather than material, so the evaluator has no term for it at all. A distance
    term on a royal therefore rewards marching it forward with *nothing* opposing it
    except the search's own horizon. King of the hill is what that looks like: with a term
    of any weight tested -- 2, 4, 8 or 16 -- mean game length falls from 54.7 plies to 20
    and ``quick_kill_rate`` goes from 0.041 to over 0.54, which is past the gate. Racing
    kings is the same shape and reaches ``quick_kill_rate`` 0.505 at the shared constant.

    ``piece_type`` of -1 means "any piece" and is left in: the nearest piece to the goal
    is usually not the royal one, so the term is not one-sided there.

    The cost of the second rule is that no committed ruleset exercises the term any more.
    That is the same position ``screen`` and ``rules.turn`` are in, and it is handled the
    same way -- constructed arms in ``tests/test_backends.py`` rather than a guess
    committed to ``rulesets/``.
    """
    if reach.safe:
        return False
    return reach.piece_type < 0 or reach.piece_type not in spec.royal_types


def _mean_non_royal(
    values: tuple[float, ...],
    royal_types: tuple[int, ...],
    endless: frozenset[int] = frozenset(),
) -> float:
    """The mean value of a piece that is *material*: the unit the goal weight is in.

    Royal pieces are valued at zero and would drag the mean towards nothing on a game
    that has several of them. **An inexhaustible type is out for the same reason it is
    out of the two scales in ``Evaluator.for_spec``**, which is the same sentence one
    line further on: a piece that is not material cannot be the unit material is counted
    in, and the goal weight is stated in exactly that unit. Leaving it in averaged
    ``INEXHAUSTIBLE_VALUE``'s zero together with every real piece, so a game with both an
    endless supply and a goal rule got a goal term understated by construction --
    Go-shaped; `docs/reference/bugs-investigations.md`, "The Go-shaped hole beside it
    is closed, because it had no unknown in it", has the record.

    Inert on the whole corpus, which is why it could be fixed before such a game exists:
    ``archive/stones`` is the only spec that declares an unlimited stock and it is won by
    checkmate, so it has no goal term for the weight to reach.
    ``rust/src/eval.rs::mean_non_royal`` skips the same two sets in the same order, read
    off ``DropRule::is_unlimited`` -- the same flattened ``drops.unlimited`` array
    ``inexhaustible_types`` reads here, not a second derivation of it.
    """
    total = 0.0
    count = 0
    for piece_type, value in enumerate(values):
        if piece_type in royal_types or piece_type in endless:
            continue
        total += value
        count += 1
    return total / count if count else 1.0


def inexhaustible_types(spec: CompiledSpec) -> frozenset[int]:
    """Piece types a player may put on the board at will, for ever.

    ``drops.unlimited`` is a fact about the game rather than a resource -- the core never
    decrements such a slot -- so the hand it names never empties and no sequence of moves
    can leave a player short of one. That is the property that makes the type worthless
    as material; see ``INEXHAUSTIBLE_VALUE``.

    Read off the compiled spec and not off the hand, deliberately. A hand is a fact about
    a *position* and the value table is derived once before the game, so the question this
    answers has to be one the rules settle: "can this run out", not "has it".
    """
    drops = spec.drops
    if drops is None:
        return frozenset()
    return frozenset(
        piece_type for piece_type, endless in enumerate(drops.unlimited) if endless
    )


def prey_types(spec: CompiledSpec) -> tuple[bool, ...]:
    """Per type: can a piece of it ever be **won** -- the pool the prey share is over.

    ``capture_allowed[attacker][victim]`` says who is *permitted* to take what, and a
    permission is not a capture. The pool asks the whole question: is there a piece that
    can capture at all, and is it allowed to take this. Both halves are needed and the
    second alone is the trap.

    **The trap, on the spec that motivated this.** ``archive/stones`` ``93b02b591c87``
    gives its six real pieces a ``captures`` list that omits the stone, and leaves the
    stone's own list absent -- which means *everything*, because that is what an absent
    list has meant since schema 3. So the stone's column is not empty:
    ``capture_allowed[stone][stone]`` is true, and "no attacker may take this type"
    reads false for the one type the whole repair is about. What is true of the stone is
    that it has no moves, so its permission can never be exercised, and that is what the
    ``reach`` test below reads.

    Deliberately a fact about the **rules** and not about the position. Restricting the
    hunters to types a player currently holds would make the pool move as pieces are
    placed and taken, and a denominator that answers to the thing it is dividing is how
    the defect got in. Over all declared types, once, before the game.

    The walk is the frame-0 ``modes & 2`` walk ``_measured_values`` does, and shares its
    reading of what a capture is: a piece that only moves quietly has no capture reach,
    and neither has a piece with no moves. It is the same table read for the same bit, so
    a type priced as capturing nothing cannot also enter this pool.
    ``rust/src/eval.rs::prey_types`` is the mirror and ``tests/test_backends.py`` holds
    the two together.
    """
    hunters = [
        attacker
        for attacker in range(spec.n_types)
        if has_capture_reach(spec, attacker)
    ]
    return tuple(
        any(spec.capture_allowed[attacker][victim] for attacker in hunters)
        for victim in range(spec.n_types)
    )


def immunity_threats(spec: CompiledSpec) -> tuple[tuple[tuple[int, ...], ...], int] | None:
    """Q466 (F2): ``threat[player][attacker]`` and the counted types, or ``None``.

    ``threat[player][attacker]`` is the bit mask of victim types a piece of ``attacker``
    owned by ``player`` could capture, now or after any chain of promotions: a type with
    capture reach (``has_capture_reach``) permitted the victim by the capture matrix.
    ``counted`` is the mask of material types: not royal, not an unlimited drop.

    ``None`` where immunity is not a fact about the capture relation alone: no capture
    matrix (every hunter takes everything, so in chess the enemy king always can), and an
    explosion, a strike or wild pieces, which remove or change a piece by another route.
    ``rust/src/eval.rs::Immunity::of`` is the mirror.
    """
    if (
        not spec.has_capture_matrix
        or spec.explosion is not None
        or spec.strike is not None
        or spec.npcs is not None
    ):
        return None
    threat = capture_threats(spec)
    endless = inexhaustible_types(spec)
    counted = 0
    for piece_type in range(spec.n_types):
        if piece_type not in spec.royal_types and piece_type not in endless:
            counted |= 1 << piece_type
    return threat, counted


def capture_threats(spec: CompiledSpec) -> tuple[tuple[int, ...], ...]:
    """``threat[player][attacker]``: the bit mask of victim types a piece of ``attacker``
    owned by ``player`` could capture, now or after any chain of promotions.

    A type with capture reach (``has_capture_reach``) permitted the victim by the capture
    matrix (without one ``capture_allowed`` is all true, so every hunter takes every
    type). Read by the immunity term (Q466) and the seal (Q708, ``sealed_goals``);
    ``rust/src/eval.rs::capture_threats`` is the mirror.
    """
    n_types = spec.n_types
    direct = []
    for attacker in range(n_types):
        mask = 0
        if has_capture_reach(spec, attacker):
            for victim in range(n_types):
                if spec.capture_allowed[attacker][victim]:
                    mask |= 1 << victim
        direct.append(mask)
    targets: list[list[set[int]]] = [[set() for _ in range(n_types)] for _ in (0, 1)]
    for rule in spec.promotions:
        for player in (0, 1):
            targets[player][rule.piece_type].update(rule.targets[player])
    threat = []
    for player in (0, 1):
        row = []
        for attacker in range(n_types):
            reach = 1 << attacker
            while True:
                grown = reach
                for source in range(n_types):
                    if reach & (1 << source):
                        for target in targets[player][source]:
                            grown |= 1 << target
                if grown == reach:
                    break
                reach = grown
            mask = 0
            for source in range(n_types):
                if reach & (1 << source):
                    mask |= direct[source]
            row.append(mask)
        threat.append(tuple(row))
    return tuple(threat)


def seal_goals(spec: CompiledSpec) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Q708 (F1): each side's goal squares, ascending, where a goal can be *sealed*; else
    ``None``.

    A ``reach`` square is sealed against a side when it holds a piece of that side's
    opponent that nothing the side has on the board may ever capture (``capture_threats``).
    Defined only where that is a fact about the capture relation alone and the race is
    priced: a priced ``reach`` rule, no explosion, strike, wild pieces or drops (a capture
    can put a new captor in the hand), and at most 64 goal squares a side.
    ``rust/src/eval.rs::Seal::of`` is the mirror.
    """
    reach = spec.reach
    if reach is None or not _reach_is_priceable(spec, reach):
        return None
    if (
        spec.explosion is not None
        or spec.strike is not None
        or spec.npcs is not None
        or spec.drops is not None
    ):
        return None
    goals = (
        tuple(sorted(square for square in reach.squares[0] if square < spec.n_squares)),
        tuple(sorted(square for square in reach.squares[1] if square < spec.n_squares)),
    )
    if max(len(goals[0]), len(goals[1])) > 64:
        return None
    return goals


def has_capture_reach(spec: CompiledSpec, piece_type: int) -> bool:
    """Whether this type has a capture move anywhere on an empty board.

    Exactly ``_measured_values(spec)[piece_type] > 0``, short-circuited: that function
    counts the squares a piece may capture on and this one only asks whether there is
    one. Kept as its own walk rather than a call, so that the pool costs nothing on the
    rulesets that never restrict a capture -- and so that the two readings of ``& 2``
    sit next to each other where a change to one is visibly a change to both.

    Public (Q138): ``spec/describe.py::capturing_types`` imports it, and importing a
    name spelled private was always a wrinkle rather than a rule.
    """
    for square in range(spec.n_squares):
        for entries in (
            spec.leaps[0][piece_type][square],
            spec.slides[0][piece_type][square],
            spec.hops[0][piece_type][square],
            spec.sweeps[0][piece_type][square],
        ):
            for _squares, modes, *_ in entries:
                if any(mode & 2 for mode in modes):
                    return True
    return False


def _measured_values(spec: CompiledSpec) -> tuple[float, ...]:
    """Average number of squares each piece can attack from an otherwise empty board.

    Overstates sliders slightly and ignores blocking entirely, but it is cheap,
    deterministic, and -- unlike anything tuned on game results -- cannot quietly encode
    a preference for the rulesets already seen.

    **What a piece may take is no longer folded in here.** It used to be: the share of
    the *opening* enemy army a piece was allowed to capture multiplied this reach, once,
    for the whole game. That made a rock worth 1.375 on the last ply of a game in which
    the opponent had had no scissors for two hundred plies, and the docstring said so and
    deferred the fix. The share is a property of the position, so it is applied in
    ``Evaluator.material`` on both sides of the boundary instead; what still travels is
    reach, which is a fact about the rules and is derived exactly once, here.

    **Zero is a real answer here, and "no opinion" is what it means.** What is counted is
    *capture* reach -- ``modes[index] & 2`` -- so a piece that moves but never takes, and
    a piece with no moves at all, both come back 0. The blend in ``Evaluator.for_spec``
    is then left with one informative input and prices such a piece at ``DECLARED_WEIGHT``
    times its declared value, which is the number the module docstring calls an
    unvalidated opinion never to be used alone. That fallback is **unmeasured** and is
    deliberately left standing rather than replaced by an invented constant: the one
    ruleset that reaches it with a finite stock is ``archive/three-stones``, whose stone
    is priced 0.25 against its pawn's 1.3984 and which measured 0.8058, viable. What it
    would cost to settle is in ``docs/bugs.md``. The *unbounded* case is a different
    defect, is not this fallback, and is fixed: see ``INEXHAUSTIBLE_VALUE``.
    """
    values = []
    for piece_type in range(spec.n_types):
        reachable = 0
        for square in range(spec.n_squares):
            targets: set[int] = set()
            for entries in (
                spec.leaps[0][piece_type][square],
                spec.slides[0][piece_type][square],
                spec.hops[0][piece_type][square],
                spec.sweeps[0][piece_type][square],
            ):
                # Hop entries carry a fourth field, the reach past their screen. Only
                # the first two matter here, so the tail is discarded rather than named
                # -- this loop should not have to know which table it is reading.
                for squares, modes, *_ in entries:
                    targets.update(
                        target for index, target in enumerate(squares) if modes[index] & 2
                    )
            reachable += len(targets)
        values.append(reachable / spec.n_squares)
    return tuple(values)
