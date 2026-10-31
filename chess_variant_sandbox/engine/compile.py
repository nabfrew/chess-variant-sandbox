"""Compile a validated rule spec into flat lookup tables.

All the interpretation happens here, once, so that move generation is a table walk
rather than a rule walk. Three things are resolved at compile time that would
otherwise cost something on every generated move:

* **Symmetry expansion.** Base vectors become concrete directions per player.
* **Rank conditions.** ``from_rank`` depends only on the origin square, which is a
  table index, so a descriptor that cannot apply at a square is simply absent there.
* **Overlapping descriptors.** Directions are merged into a single ray per direction
  carrying a per-distance mode mask, so the generator can never emit a duplicate move.
  This matters more than it looks: duplicate moves would silently inflate branching
  factor, and branching factor feeds the interestingness index.

What survives into the hot path is only what genuinely depends on the position:
occupancy, and the ``unmoved`` mask behind ``first_move``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import gcd
from typing import Any

from chess_variant_sandbox.engine.moves import (
    MOVE_PUSH,
    MOVE_SWEEP,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.moves import move_push as move_push_direction
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.legs import Legs, stated_legs
from chess_variant_sandbox.spec.schema import (
    DEFAULT_RANGE,
    MAX_MOVES_PER_TURN,
    POUNCE_KINDS,
    PURSE_MAX,
    REPETITION_NEVER,
    STOCK_UNLIMITED,
    WRAP_PAIRS,
    SpecError,
    absent_squares,
    beats_of,
    blast_directions,
    bribe_ids,
    control_geometry,
    duck_piece,
    has_rests,
    has_speeds,
    has_third_owner,
    hex_distance,
    hire_ids,
    insufficient_material_live,
    insufficient_material_roles,
    is_nobodys,
    majority_ids,
    move_push,
    neutral_ids,
    npc_immune_attackers,
    npc_pounce,
    npc_respawn,
    npc_routes,
    npc_split,
    npc_starts,
    npc_turns_back,
    obstacle_squares,
    parse_square,
    path_of,
    piece_bounties,
    piece_prices,
    piece_rests,
    piece_sights,
    piece_speeds,
    piece_strengths,
    piece_weights,
    portal_end_ids,
    portal_groups,
    portal_has_folds,
    portal_movement_of_groups,
    portal_movement_types,
    promotion_squares,
    promotion_targets,
    purse_market,
    purse_pays_loser,
    purse_rule,
    purse_squares,
    push_lists,
    push_rule,
    reach_pieces,
    reach_reflection,
    reflect_square,
    sight_affected_ids,
    simultaneous_switch,
    spy_ids,
    strike_is_live,
    sweep_length,
    takes_royal_ids,
    turning_order,
    turns_less_than_half,
    validate_or_raise,
    wrap_stated,
)
from chess_variant_sandbox.spec.schema import (
    facing_turns as facing_turns_of,
)

MODE_MOVE = 1
MODE_CAPTURE = 2
MODE_BOTH = 3
# Q1074b1 (`docs/plan-push.md` §1.3): a landing on a piece this entry may push, the piece
# the enemy's (4) or the mover's own (8). A ``push.captures: false`` entry keeps its capture
# bit off where it pushes enemies. Bits nobody's tables held before, so every stored spec's
# masks are unchanged.
MODE_PUSH_ENEMY = 4
MODE_PUSH_OWN = 8
MODE_PUSH = MODE_PUSH_ENEMY | MODE_PUSH_OWN
# `PushRule.next`: off the board (an edge or a hexagon's absent cell, W2), and a square nothing
# may enter (a wall, §3.6 W1's block).
PUSH_OFF = -1
PUSH_STOP = -2
# Q1074b7 (W1): a wall under `rules.push.walls: "crush"`, against which a pushed piece is
# removed, counted as one pushed off the board (C1). `Tables::PUSH_CRUSH`'s twin.
PUSH_CRUSH = -3
# `PushRule.leap_dirs` (and slides', hops'): a landing that pushes nothing.
PUSH_NO_DIRECTION = -1
PUSH_CHAIN = {"one": 0, "row": 1, "weight": 2}
# `PushRule.attack_kinds`: what a source attacks by, a capture (1), a push of an enemy (2).
ATTACK_BY_CAPTURE = 1
ATTACK_BY_PUSH = 2

# What ``rules.repetition_draw: "never"`` compiles to (schema 10). ``u32::MAX``, because
# the core's threshold is a ``u32`` and a repetition count can never reach it -- so the
# rule is removed by the *table* rather than by a branch, and no ruleset that states a
# number is affected. See ``_compile_repetition_draw``.
NO_REPETITION_DRAW = 0xFFFF_FFFF

# The orientations, in quarter turns clockwise from the piece's own forward. Index into
# this is the *quarter turn count*, which is not the same number as a facing index
# unless the spec has four of them -- see ``facing_quarter_turns``.
FACING_NAMES = ("forward", "right", "backward", "left")

# The owner index of a piece neither player owns (schema 9). Owners are 0 and 1 for the
# two players and this for a wild piece, so a wild byte sits in a third contiguous block
# above both armies and every "is this piece mine?" range test is untouched.
WILD = 2

# Schema 17 (Q540): a strike entry for a victim that is removed while its attacker stays
# where it stood (rifle chess). Below -1 so that every reader asking "does it survive?"
# (``>= 0``) reads it as removal; only the core's ``strike_writes`` tells it from -1.
REMOVED_IN_PLACE = -2


def frame_index(player: int, facing: int, n_facings: int) -> int:
    """Which frame a piece reads its movement out of.

    A **frame** is one way of reading the movement vectors. There was exactly one per
    player until schema 8, and the answer was the player index -- which is why the tables
    have looked like `[player][type][square]` since the beginning. That first index has
    always secretly meant *which way is forward*, and calling it the player is what this
    generalises.

    `player * n_facings + facing`, **player-major**, and that ordering is the whole of
    the encoding argument. The board byte is `player * n_facings * n_types + facing *
    n_types + type + 1`, so it runs frame-major too, and every "is this piece mine?" test
    in the core stays the range comparison it always was -- only the stride changes. The
    other spelling, `facing * 2 + player`, interleaves the two players' values and turns
    each of those range tests into a silent bug. With `n_facings` of 1 both spellings are
    the identity on the player, so the frames of every ruleset without a facing are the
    two they always were and the tables are byte-identical.
    """
    return player * n_facings + facing


def facing_quarter_turns(facing: int, n_facings: int) -> int:
    """How far a facing index turns the vectors, in quarter turns clockwise.

    A spec's facings are evenly spaced round the circle, so the step is `4 // n_facings`:
    with four of them a facing index *is* a quarter turn, and with two of them each
    index is a half turn. A piece with fewer axes than the spec simply steps by more
    than one index at a time -- see ``CompiledSpec.facing_step``.
    """
    return facing * (4 // n_facings)


def _coat_facing(headings: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """TC2 (`docs/plan-turncoat.md` §6): which facing index keeps a piece's heading when it
    changes sides on the board, ``[from_owner][facing] -> facing`` for the other owner.

    Facing indices are relative to the owner's forward (``frame_index``), so keeping the
    heading is a new index, read off the frames' own vectors -- ``facing_headings``, the
    rotation and then the player's reflection `_compile_square` puts every vector through.
    Under ``"rotate"`` forward becomes backward and right becomes left; under ``"mirror"``
    forward becomes backward and right stays right. The core looks the answer up and
    computes nothing (rule 4). Every lattice the language has has a matching index for
    every facing (a reflection or a half turn maps the four headings onto themselves), so
    a missing one is a compiler bug, not a spec to refuse.
    """
    coat: list[tuple[int, ...]] = []
    for owner in (0, 1):
        theirs = headings[1 - owner]
        row = []
        for heading in headings[owner]:
            if heading not in theirs:
                raise AssertionError(f"no facing of the other owner points at {heading} degrees")
            row.append(theirs.index(heading))
        coat.append(tuple(row))
    return tuple(coat)


def _rotate(dx: int, dy: int, quarters: int) -> tuple[int, int]:
    """One vector turned clockwise, a quarter turn at a time.

    Clockwise so that the four orientations read *forward, right, backward, left* in
    index order, which is the order ``FACING_NAMES`` states and the order the prose says
    out loud. Applied to the vector as the spec wrote it -- before the second player's
    reflection -- so that a facing means the same relative thing to both armies, exactly
    as a base vector does.
    """
    for _ in range(quarters % 4):
        dx, dy = dy, -dx
    return dx, dy

_MODE_BITS = {"move": MODE_MOVE, "capture": MODE_CAPTURE, "both": MODE_BOTH}

# (targets or ray, per-index mode mask, requires the origin piece to be unmoved)
Entry = tuple[tuple[int, ...], tuple[int, ...], bool]

# A hop also carries how far past its screen it may land -- 0 for the whole rest of the
# ray, which is the xiangqi cannon, or n for the grasshopper family -- and whose piece
# the screen has to be: SCREEN_ANY, SCREEN_OWN or SCREEN_ENEMY.
HopEntry = tuple[tuple[int, ...], tuple[int, ...], bool, int, int]

# Q1063: a sweep also carries its sweep length, how many of the ray's first squares it
# clears: 0 for the whole ray (the sweep with one length, which every sweep before Q1063
# is). Past it the ray is a slide's: any piece ends it, taken where it may be. It crosses in
# the slot a hop's screen range uses, which a sweep left at 0.
SweepEntry = tuple[tuple[int, ...], tuple[int, ...], bool, int]

SCREEN_ANY = 0
SCREEN_OWN = 1
SCREEN_ENEMY = 2
_SCREEN_BITS = {"any": SCREEN_ANY, "own": SCREEN_OWN, "enemy": SCREEN_ENEMY}

# One way a square can be attacked:
#   (origin, piece value, squares that must be empty, squares that hold the screens,
#    how many screens, whether the attacker must be unmoved, whose piece the screen is)
# Splitting the intervening squares into a must-be-empty prefix and a screen window is
# what lets one flat scan serve all three primitives. A slide or a leap puts everything
# in the prefix and wants no screens. A cannon puts everything in the window and wants
# exactly one. A grasshopper, which must land immediately past its screen, puts all but
# the last square in the prefix -- so "the screen is in the right place" is the same
# count test as "there is a screen at all".
#
# Schema 13 adds two fields: sweep bits and the path's modes. Both are 0 and () for every
# source but a sweep's. A sweep source's prefix is the ray before the attacked square and
# its window is the rest of the ray; bit 0 says the square is a capture landing, bit 1
# says some square past it may be one.
AttackSource = tuple[
    int, int, tuple[int, ...], tuple[int, ...], int, bool, int, int, tuple[int, ...]
]
SWEEP_LANDS = 1
SWEEP_BEYOND = 2
# Q1063: a sweep source's sweep length rides above the two bits, 0 for a sweep that clears
# its whole ray (so every source before Q1063 keeps its byte). Past that index the ray is
# a slide's, and any piece on it ends the ray.
SWEEP_LENGTH_SHIFT = 2


@dataclass(frozen=True)
class WrapRule:
    """Which edges each player may cross, and therefore what shape the board is.

    Two separate questions live here and conflating them is the mistake to avoid. A
    *permission* is directional and belongs to a player: the first player may leave by
    the left edge and the second by the right, and those are different rules. The
    *topology* is not directional and belongs to the board: ``left`` and ``right`` name
    the same seam, because file ``a`` is adjacent to file ``h`` whichever way you cross
    it. Movement reads the permission; an explosion's blast reads the topology.

    Stored as ``[player] -> (may cross the low edge, may cross the high edge)``, where
    low is file ``a`` and rank 1 -- absolute, from the standard diagram, so that two
    identical strings can never mean two different seams.

    **Q281a2 adds a hexagon's three pairs of sides** (unreachable until the gate,
    `docs/reference/hex-build-plan.md` *Q281a2*): ``n_s``, ``ne_sw`` and ``nw_se``, one
    field each, shaped exactly like ``files`` and ``ranks`` -- ``[player] -> (may leave
    by the low side, may leave by the high side)``, `schema.WRAP_PAIRS`' own low and
    high. ``files``, ``ranks``, ``files_join``, ``ranks_join`` and ``active_for`` keep
    their meaning for a grid unchanged, so nothing that reads them for a rectangle moves.
    """

    files: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False))
    ranks: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False))
    n_s: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False))
    ne_sw: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False))
    nw_se: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False))

    @property
    def files_join(self) -> bool:
        """Whether the two file edges are the same edge, for anyone at all."""
        return any(any(side) for side in self.files)

    @property
    def ranks_join(self) -> bool:
        return any(any(side) for side in self.ranks)

    @property
    def seams_join(self) -> bool:
        """Whether any of a hexagon's three pairs is joined, for anyone at all."""
        return any(
            any(any(side) for side in getattr(self, key))
            for key in ("n_s", "ne_sw", "nw_se")
        )

    def active_for(self, player: int) -> bool:
        """Whether this player may cross anything, which is what bounds a ray."""
        return (
            any(self.files[player]) or any(self.ranks[player])
            or any(any(getattr(self, key)[player]) for key in ("n_s", "ne_sw", "nw_se"))
        )

    def may_leave(self, side: str, player: int) -> bool:
        """Whether ``player`` may leave the board by this hexagon side (Q281a2).

        A move reads the mover's own permission, which is what makes ``{"0": "n", "1":
        "s"}`` a one-way seam rather than a wall one player happens to stand at.
        """
        for key, pair in WRAP_PAIRS.items():
            low, high = getattr(self, key)[player]
            if side == pair.low:
                return low
            if side == pair.high:
                return high
        return False

    def seam_open(self, side: str) -> bool:
        """Whether anyone at all may cross this hexagon side, either way (Q281a2).

        The topology question, for a blast or the flying general: a pair is a seam if
        anyone may cross it either way, exactly as ``files_join``/``ranks_join`` read a
        rectangle's two axes today.
        """
        for key, pair in WRAP_PAIRS.items():
            if side in (pair.low, pair.high):
                low, high = getattr(self, key)
                return any(low) or any(high)
        return False


def _wrap_axis(value: int, size: int, low: bool, high: bool) -> int | None:
    """One coordinate after a step, or ``None`` where the step leaves the board.

    The sign of the excursion names the edge: a coordinate below zero left by the low
    edge and one at or past ``size`` by the high edge, however far past it went. That is
    what lets a leap apply its whole offset in one go while a slide, which steps one
    square at a time, checks each crossing separately -- both with the same function.
    """
    if 0 <= value < size:
        return value
    if value < 0:
        return value % size if low else None
    return value % size if high else None


def _land(
    at_file: int,
    at_rank: int,
    width: int,
    height: int,
    low_file: bool,
    high_file: bool,
    low_rank: bool,
    high_rank: bool,
    radius: int = 0,
    permitted: Callable[[str], bool] | None = None,
) -> int | None:
    """The one place that works out where a step lands (Q281a2, unreachable until the
    gate -- `docs/reference/hex-build-plan.md` *Q281a2*, *What the compiler resolves it
    into*).

    On a rectangle (``radius == 0``) this is exactly today's two-axis wrap: ``low_file``/
    ``high_file`` and ``low_rank``/``high_rank`` are read through ``_wrap_axis`` as they
    always were, so every existing spec's tables are untouched byte for byte.

    On a hexagon (``radius`` the board's R), ``(at_file, at_rank)`` is the raw target in
    the frame and ``c = (at_file - R, at_rank - R)``. The direct cell is used if it is on
    the board; otherwise the target lies in exactly one of the board's six neighbouring
    copies (`schema.WRAP_PAIRS`, the tiling `tests/hex_geometry.py` proves), found by
    subtracting that side's copy vector and testing the same hex-distance bound. That
    side is then the one it takes ``permitted`` to open -- a move reads the mover's own
    permission (``WrapRule.may_leave``); a blast or the flying general reads the board's
    topology (``WrapRule.seam_open``) because neither has a "whose turn is it" to ask.
    The board and its six copies cover every point a vector `validate_or_raise`'s
    refusal 7 allows can reach, without overlap, so the first match is the only one.
    """
    if radius:
        c_file, c_rank = at_file - radius, at_rank - radius
        if hex_distance(c_file, c_rank) <= radius:
            return at_rank * width + at_file
        is_open = permitted or (lambda _side: False)
        for pair in WRAP_PAIRS.values():
            vector = pair.copy(radius)
            for side, offset in ((pair.high, vector), (pair.low, (-vector[0], -vector[1]))):
                near_file, near_rank = c_file - offset[0], c_rank - offset[1]
                if hex_distance(near_file, near_rank) <= radius:
                    if is_open(side):
                        return (near_rank + radius) * width + (near_file + radius)
                    return None
        return None  # never reached: refusal 7 bounds every vector to what the six copies cover
    wrapped_file = _wrap_axis(at_file, width, low_file, high_file)
    if wrapped_file is None:
        return None
    wrapped_rank = _wrap_axis(at_rank, height, low_rank, high_rank)
    if wrapped_rank is None:
        return None
    return wrapped_rank * width + wrapped_file


@dataclass(frozen=True)
class HexCopy:
    """One of a hexagon's six neighbouring copies (Q281a2, unreachable until the gate).

    ``side`` is the frame side a piece leaves by to reach it, one of `schema.WRAP_PAIRS`'
    ``low``/``high`` names. ``vector`` is the (file, rank) offset from a cell of this
    copy to the same cell of the board proper, so the drawing and the designer
    (Q281b, Q281c) can place a copy, or work a click's offset out across one, without
    doing any hex arithmetic of their own. ``permission`` is ``(player 0, player 1)``:
    which players may actually cross to it, which is `WrapRule.may_leave` for ``side``.
    """

    side: str
    vector: tuple[int, int]
    permission: tuple[bool, bool]


@dataclass(frozen=True)
class StrikeRule:
    """What happens instead of removal when two pieces meet (schema 6; schema 10).

    **One rule, two spellings.** ``rules.strike`` says what the *victim* becomes, keyed
    by the victim alone. ``rules.contacts`` (schema 10, Q19a) says what *both* become,
    keyed by the pair -- and the first is the second with no attacker column, which is
    why they compile to one dataclass and are refused beside each other rather than
    ordered. Whatever the spec wrote, what is compiled here is the general form:

    * ``onto_to[mover * n_types + victim]`` -- the type the struck piece becomes, or
      ``-1`` for it leaving the board, which covers a link the table does not mention
      and an immune type as well. Three cases, one test.
    * ``mover_to[mover * n_types + victim]`` -- the type the *attacker* becomes, or
      ``-1`` for it being what it already was. A ``strike`` fills this with ``-1``, so
      it is the schema-6 rule exactly.

    ``compact_to`` is the projection back: the schema-6 one-array form when the table
    has no attacker column and every mover's row agrees. The boundary sends that form
    wherever it exists, which is what keeps every ruleset written before schema 10
    compiling byte for byte to the tables it always did -- and what makes "contacts
    generalises strike" a checkable claim rather than a description.

    Neither array is ``CompiledSpec.demotions``. That one already exists and means
    shogi's "what a promoted piece reverts to in its captor's hand", keyed by
    ``piece.promoted_from``; this is a different relation over the same piece set, and
    folding the two together would give every shogi ruleset a strike rule nobody wrote.
    """

    effect: str  # "demote" or "promote"; "contact" for the general table
    position: str  # "stay", "advance" or "swap"
    immune_types: frozenset[int]
    n_types: int
    onto_to: tuple[int, ...]
    mover_to: tuple[int, ...]
    spelling: str = "strike"  # the key the spec wrote: "strike" or "contacts"
    # Schema 11 (Q19b): the same two tables for a meeting with a *friendly* piece, keyed
    # the same way. Empty when no link says `own`, which is every table before schema 11.
    own_onto_to: tuple[int, ...] = ()
    own_mover_to: tuple[int, ...] = ()
    # Schema 12 (Q20): under ``position: "choose"``, ``landing[player][square]`` lists the
    # squares a struck piece standing on ``square`` may be placed on, ascending. Empty
    # under every other position.
    landing: tuple[tuple[tuple[int, ...], ...], ...] = ()
    # Schema 17 (Q540): under ``position: "home"``, ``rebirth[(owner * n_types + type) *
    # n_squares + square]`` is where a piece of ``owner`` and ``type`` taken on ``square``
    # is reborn, or -1 where it has no home and is removed. Empty under every other
    # position, and empty when the rule was compiled without a setup to read
    # (``schema._compiled_strike`` asks only whether anything survives).
    rebirth: tuple[int, ...] = ()
    # Schema 17 (Q540): ``mover_flips[mover * n_types + victim]`` -- the attacker changes
    # sides (``strike.switches_sides``, Andernach). Empty when no type switches, which is
    # every rule before schema 17, so the boundary sends nothing new for them.
    mover_flips: tuple[bool, ...] = ()

    @property
    def meets_own(self) -> bool:
        """Whether any link lets a piece move onto its own side's piece."""
        return bool(self.own_onto_to)

    @property
    def survives_anything(self) -> bool:
        """Whether any type at all is left on the board by a strike."""
        return any(target >= 0 for target in self.onto_to + self.own_onto_to)

    @property
    def removes_in_place(self) -> bool:
        """Whether some capture removes its victim and leaves the attacker where it stood.

        Schema 17's rifle (``effect: "capture"``, ``position: "stay"``): the victim's entry
        is ``REMOVED_IN_PLACE`` rather than -1, which is still "leaves the board" to every
        reader that asks ``>= 0``, and is "and the attacker does not move" to the core.
        """
        return any(target == REMOVED_IN_PLACE for target in self.onto_to)

    @property
    def changes_mover(self) -> bool:
        """Whether any meeting turns the *attacker* into something else, or sends it over."""
        return any(target >= 0 for target in self.mover_to + self.own_mover_to) or any(
            self.mover_flips
        )

    @property
    def compact_to(self) -> tuple[int, ...] | None:
        """The schema-6 array, when this table can be said keyed by the victim alone."""
        if self.changes_mover or self.meets_own:
            return None
        n = self.n_types
        row = self.onto_to[:n]
        for mover in range(1, n):
            if self.onto_to[mover * n : (mover + 1) * n] != row:
                return None
        return row


@dataclass(frozen=True)
class ExplosionRule:
    """A capture that destroys its own neighbourhood.

    ``blast[square]`` is the precomputed neighbourhood, so the hot path never does
    arithmetic on file and rank to find out what a capture takes with it. The capturing
    piece always dies; ``immune_types`` is what survives standing next to one.

    ``by_types`` and ``of_types`` are schema 10 (Q18) and narrow *when* the blast happens
    rather than what it reaches: whose capture explodes, and what has to be taken for it
    to. ``None`` means every type, which is atomic chess -- and it is ``None`` rather than
    a full set on purpose, because the boundary sends an empty list for it and the
    canonical tables of every ruleset written before schema 10 therefore do not move.

    Neither can be an *empty* set here. A rule that no capturer can fire, or that no
    victim can trigger, is not an explosion rule at all and ``_compile_explosion`` returns
    ``None`` for it -- the answer at the boundary rather than the question, and the shape
    ``_compile_strike`` already has for a chain that leaves nothing alive.
    """

    immune_types: frozenset[int]
    blast: tuple[tuple[int, ...], ...]
    by_types: frozenset[int] | None = None
    of_types: frozenset[int] | None = None

    @property
    def fires_on_every_capture(self) -> bool:
        """Whether this is atomic's explosion: any capturer, any victim."""
        return self.by_types is None and self.of_types is None


@dataclass(frozen=True)
class DropRule:
    """Where each piece type may be returned to the board, per player.

    ``regions[player][piece_type]`` of ``None`` means any empty square. Crazyhouse uses
    it for exactly one piece -- a pawn may not be dropped on a promotion rank, because
    it would have to promote immediately and there is no move in which to do it.
    """

    from_captures: bool
    regions: tuple[tuple[frozenset[int] | None, ...], tuple[frozenset[int] | None, ...]]
    no_doubled: frozenset[int]  # not onto a file already holding one of your own
    no_mate: frozenset[int]  # may not deliver checkmate on the drop itself
    # Schema 10 (Q51): what each player holds before anything has been captured.
    #
    # ``stock[type]`` is what the hand starts at -- 0 for every type the spec does not
    # stock, which is every ruleset before schema 10 -- and ``unlimited[type]`` says the
    # slot is never spent. An unlimited slot starts at 1 rather than at a large number,
    # because it is a *fact about the game* and not a resource: the core skips its
    # decrement, so the count stays 1 for the whole game and the hand stays constant,
    # which is what keeps the schema-2 argument for leaving hands out of the Zobrist key
    # true. ``unmake`` needs nothing either -- the undo log records a signed delta and an
    # unlimited slot records no delta at all.
    stock: tuple[int, ...] = ()
    unlimited: tuple[bool, ...] = ()
    # Schema 18 (Q784): the placement phase. While a side holds anything in hand, its only
    # legal moves are drops, which with a stock and no capture hand is *while its stock is
    # unspent* (Fairy-Stockfish's ``mustDrop``). A royal may then start in the stock.
    placement: bool = False
    # Schema 18 (Q785): drop either face. ``faces[slot]`` is the other face a piece held
    # in that slot may be dropped as -- the piece whose ``promoted_from`` it is -- or -1.
    # Empty unless the hand says ``either_face``, so every earlier game's tables are the
    # ones it had. A drop's move word carries the face that stands; ``hand_slot`` says
    # which slot it spends.
    faces: tuple[int, ...] = ()
    # Q1059a (``drops.gating``, Seirawan's gating): ``gates[player]`` is the set of squares on
    # which a piece of that player's, leaving its setup square for the first time, may let
    # one held piece in on the square it vacates, as part of the same move (a castle
    # vacates two). ``None`` for a hand that drops anywhere; under a set a held piece never
    # enters by a drop. The zone is ``schema.promotion_squares``' reading of the key
    # (``ranks`` or ``squares``, from the first player's side, reflected for the second),
    # less the walls.
    gates: tuple[frozenset[int], frozenset[int]] | None = None

    def hand_slot(self, face: int) -> int:
        """The hand slot a drop that places ``face`` spends (itself, or its base)."""
        for slot, paired in enumerate(self.faces):
            if paired == face:
                return slot
        return face


@dataclass(frozen=True)
class ReachRule:
    """A win by arrival: king of the hill, racing kings, breakthrough.

    ``piece_types`` empty means any piece, which is what breakthrough wants; otherwise
    the types whose arrival decides, ascending. SR3a (`docs/plan-schema-next.md` §1 rank
    4): ``reach.piece`` takes a list, so a side may choose its runner; one id is a tuple
    of one, and crosses the boundary exactly as the single type always did.
    Squares are stored per player because most such games aim each side at the other's
    end of the board, but racing kings sends both to the same rank.
    """

    piece_types: tuple[int, ...]
    squares: tuple[frozenset[int], frozenset[int]]
    result: str
    safe: bool = False  # the arriving piece must not be capturable where it lands
    # Schema 18 (Q786, Jeson Mor): ``reach.on: "leave"``. The game is decided by the move
    # that takes the named piece *off* one of its side's goal squares, read off the move's
    # origin in ``make`` (``Game::goal_left_by``), and a piece standing on a goal decides
    # nothing. False is the arrival every game before it reads.
    leaves: bool = False

    def admits(self, piece_type: int) -> bool:
        """Whether a piece of ``piece_type`` is one this goal is about."""
        return not self.piece_types or piece_type in self.piece_types

    @property
    def mask(self) -> int:
        """The types it admits as a bit mask over declared types, ``-1`` for any piece."""
        mask = 0
        for piece_type in self.piece_types:
            mask |= 1 << piece_type
        return mask if self.piece_types else -1


@dataclass(frozen=True)
class ExtinctionRule:
    """A loss by running out: kinglet's pawns, three-kings' kings, the horde's army.

    ``piece_types`` empty means "any piece at all", so the rule counts the whole army.

    ``per_type`` is schema 18 (Q783, Extinction chess): each type in ``piece_types`` is
    counted on its own, so a side has run out when any one of them is below
    ``min_count``, not when all of them together are. The validator refuses it over
    ``any`` and over a single type, where it would state nothing.

    ``own`` is schema 27 (Q936, ``"per_type": "own"``, `docs/reference/extinction-any-kind.md`
    §2): counted apart as ``per_type`` is (so ``per_type`` is true beside it), but each
    listed type only against a side that starts with at least one on the board.
    ``side_types`` holds the two lists, resolved from the opening board by
    ``_side_types``; under ``true`` both are ``piece_types``, and under the summed rule both
    are empty. The set is fixed at ply zero: a kind a side gains later by promotion, a side
    switch or a drop is never counted against it.
    """

    piece_types: tuple[int, ...]
    any_piece: bool
    min_count: int
    result: str
    per_type: bool = False
    own: bool = False
    side_types: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())


@dataclass(frozen=True)
class TurnRule:
    """How many moves a turn is, with the opening turns listed separately.

    ``lengths`` is the ramp, already resolved: entry *n* is the length of turn *n*, and
    everything past the end is ``moves``. Resolved here rather than in the core so the
    core walks a table like everything else.
    """

    moves: int
    lengths: tuple[int, ...]

    def length_of(self, turn: int) -> int:
        return self.lengths[turn] if turn < len(self.lengths) else self.moves


@dataclass(frozen=True)
class DuckRule:
    """The duck (generation 42, N4b, `docs/archive/plan-npc.md` §4.6): one piece nobody owns, which
    the mover moves as the second half of every turn.

    * ``piece_type`` -- its declared type, standing in the third owner's block at facing 0;
    * ``by_moves`` -- ``goes: "by_moves"``: it walks its type's frame-0 leap and slide entries
      (the validator refuses moves that read differently for the two players, so frame 0 is
      both players'); ``False`` is ``"anywhere"``, any empty square but its own;
    * ``start`` -- its opening square, or -1: off the board, put on by the first duck move;
    * ``squares`` -- the squares it may stand on, ascending: every square but a wall or an
      absent cell, which only the compiler knows.

    Beside it the compiled turn is two moves (`_compile`), the duck's column of the capture
    matrix is cleared (nothing takes it) and so is its row (it takes nothing).
    """

    piece_type: int
    by_moves: bool
    start: int
    squares: tuple[int, ...]


def _compile_duck(
    rules: dict[str, Any], index_of: dict[str, int], width: int, height: int,
    obstacles: frozenset[int],
) -> DuckRule | None:
    """`DuckRule` for a spec with ``rules.duck``, ``None`` otherwise."""
    duck = rules.get("duck")
    if not isinstance(duck, dict):
        return None
    start = -1
    if isinstance(duck.get("start"), str):
        coords = parse_square(duck["start"], width, height)
        assert coords is not None  # guaranteed by validation
        start = coords[1] * width + coords[0]
    return DuckRule(
        piece_type=index_of[duck["piece"]],
        by_moves=duck.get("goes", "anywhere") == "by_moves",
        start=start,
        squares=tuple(square for square in range(width * height) if square not in obstacles),
    )


def _clear_duck_captures(
    matrix: tuple[tuple[bool, ...], ...], duck: DuckRule | None
) -> tuple[tuple[bool, ...], ...]:
    """Generation 42: nothing takes the duck (its column cleared) and it takes nothing (its
    row cleared). The matrix it was handed without a duck."""
    if duck is None:
        return matrix
    kind = duck.piece_type
    return tuple(
        tuple(allowed and victim != kind and attacker != kind for victim, allowed in enumerate(row))
        for attacker, row in enumerate(matrix)
    )


# TC1 (`docs/plan-turncoat.md` §3.2): who moves a piece nobody owns, by its type -- the
# **control kind** the core reads for a byte in the third owner's block (`Tables::value_kind`).
# 0 is no kind: a type no nobody's piece is ever of. Mirrored by `rust/src/tables.rs`'s
# `CONTROL_*`. `NEUTRAL` (a nobody's piece that stands) and `MAJORITY` (moved by the majority
# holder) are TC3's, read off `NeutralRule`.
CONTROL_NONE = 0
CONTROL_ROUTE = 1
CONTROL_DUCK = 2
CONTROL_PORTAL = 3
CONTROL_NEUTRAL = 4
CONTROL_MAJORITY = 5


@dataclass(frozen=True)
class ControlRule:
    """Who moves each type's pieces when nobody owns them (TC1, `docs/plan-turncoat.md` §3).

    A piece has an **allegiance** (which owner block its byte sits in: a player's, or nobody's)
    and a **control** (who may move it now). A player's piece is controlled by its owner, plus
    whatever a sight grants (``command``); nobody's piece by its type's ``kind`` here:
    ``CONTROL_ROUTE`` walked by the phase (``rules.npcs``), ``CONTROL_DUCK`` moved by the mover
    on the duck half (``rules.duck``), ``CONTROL_PORTAL`` a portal end, moved by its group's
    rule. One entry per type; the core expands it per board byte of the third block.

    Read off today's rules alone: the third block holds wild pieces, the duck (both may share
    it) or portal ends (refused beside either), and every non-duck type a wild game could set in
    it is a route walker, which is what ``Tables::is_wild_piece`` always answered. ``None`` when
    nobody owns anything (``n_owners`` 2), so a two-army game crosses an empty table.
    """

    kind: tuple[int, ...]


def _compile_control(
    n_types: int, n_owners: int, npcs: NpcRule | None, duck: DuckRule | None,
    portals: PortalRule | None, neutrals: NeutralRule | None = None,
) -> ControlRule | None:
    """`ControlRule` for a game with a third owner's block, ``None`` otherwise. TC3's
    neutral kinds come from `NeutralRule`, which the validator refuses beside the other
    three, so a type has one kind."""
    if n_owners < 3:
        return None
    if neutrals is not None:
        return ControlRule(kind=neutrals.kind)
    kind = [CONTROL_ROUTE if npcs is not None else CONTROL_NONE] * n_types
    if portals is not None:
        for piece_type in portals.end_types:
            kind[piece_type] = CONTROL_PORTAL
    if duck is not None:
        kind[duck.piece_type] = CONTROL_DUCK
    return ControlRule(kind=tuple(kind))


# Q1069a: what each type counts as in a dead set of material (`schema.INSUFFICIENT_*`).
INSUFFICIENT_OTHER = 0
INSUFFICIENT_ROYAL = 1
INSUFFICIENT_LEAPER = 2
INSUFFICIENT_RIDER = 3


@dataclass(frozen=True)
class InsufficientMaterialRule:
    """Q1069a: the draw by insufficient material (`docs/reference/insufficient-material.md`).

    ``roles[piece_type]`` is what a piece of that type is in FIDE's dead sets, read by how it
    moves (`schema.insufficient_material_roles`, which the validator also asks):
    ``INSUFFICIENT_ROYAL`` for the one royal, ``INSUFFICIENT_LEAPER`` for a type that jumps
    exactly (1, 2) every way, ``INSUFFICIENT_RIDER`` for one that rides the four diagonals,
    and ``INSUFFICIENT_OTHER`` for every other type, whose presence means the material is
    not dead. ``colour[square]`` is the square's class in the riders' move graph (the
    component a rider standing there can reach), 0 or 1 on the 8x8 board the rule needs:
    the square colour, worked out from the tables rather than from a file and rank.

    The position is drawn when the board holds, besides the two royals, nothing, exactly
    one leaper or rider (a lone minor piece), or only riders, all on one class (any number
    of them on one square colour). The core asks `Game::material_is_dead`; it
    computes no square and searches nothing.
    """

    roles: tuple[int, ...]
    colour: tuple[int, ...]


def _compile_insufficient_material(
    spec: dict[str, Any],
    index_of: dict[str, int],
    n_types: int,
    n_squares: int,
    slides: Sequence[Sequence[Sequence[Sequence[Entry]]]],
) -> InsufficientMaterialRule | None:
    """``InsufficientMaterialRule`` where ``rules.insufficient_material`` states a rule, else
    ``None``: on a game that is not chess once its other types are gone the key is inert
    (`schema.insufficient_material_live`), so every type given a role here moves exactly as
    its counterpart in FIDE's game."""
    if not insufficient_material_live(spec):
        return None
    by_id, _ = insufficient_material_roles(spec)
    roles = [INSUFFICIENT_OTHER] * n_types
    for piece_id, role in by_id.items():
        roles[index_of[piece_id]] = role
    # The riders' classes: union every square with each square a rider standing there can
    # reach, in the first frame (the riders' moves read alike for both players).
    parent = list(range(n_squares))

    def root(square: int) -> int:
        while parent[square] != square:
            parent[square] = parent[parent[square]]
            square = parent[square]
        return square

    riders = [t for t in range(n_types) if roles[t] == INSUFFICIENT_RIDER]
    for piece_type in riders[:1]:
        for square in range(n_squares):
            for landings, _modes, _flag in slides[0][piece_type][square]:
                for landing in landings:
                    parent[root(landing)] = root(square)
    classes: dict[int, int] = {}
    colour = tuple(classes.setdefault(root(square), len(classes)) for square in range(n_squares))
    if riders and len(classes) != 2:
        raise SpecError([
            "rules.insufficient_material: the diagonal riders do not split the board into "
            "two colours"
        ])
    return InsufficientMaterialRule(
        roles=tuple(roles), colour=colour if riders else (0,) * n_squares
    )


@dataclass(frozen=True)
class NeutralRule:
    """TC3 (`docs/plan-turncoat.md` §4): pieces nobody owns from the setup, and who may move
    them.

    ``kind`` is each type's control kind in the third owner's block: ``CONTROL_MAJORITY`` for
    a type with ``control: "majority"`` that some placement puts there, ``CONTROL_NEUTRAL``
    for any other type placed there (it stands), ``CONTROL_NONE`` for the rest.
    ``neighbours[piece_type][square]`` is what *next to* means for a majority type standing
    on ``square`` (its control's geometry, or the king's step, through `_compile_square`, so
    walls, absent cells and wrap are resolved here), empty for every other type. The core
    counts each player's pieces over that list and computes no square (rule 4).

    ``takes_royal`` (Q1077, `docs/plan-turncoat.md` §4.7) is the majority types whose held
    pieces may take the royal of the player they are held against, and so attack it: their
    frame-0 attack sources fill the third owner's block (`_build_attack_sources`), asked only
    for the holder. Empty in every game without the key, which is the game TC3 built.
    """

    kind: tuple[int, ...]
    neighbours: tuple[tuple[tuple[int, ...], ...], ...]
    takes_royal: tuple[int, ...] = ()

    @property
    def has_majority(self) -> bool:
        return CONTROL_MAJORITY in self.kind


def _compile_neutrals(
    spec: dict[str, Any], pieces: list[dict[str, Any]],
    width: int, height: int, orientation: str, wrap: WrapRule | None,
    obstacles: frozenset[int], radius: int,
) -> NeutralRule | None:
    """`NeutralRule`, or ``None`` when no placement is nobody's (every game before TC3)."""
    placed = neutral_ids(spec)
    if not placed:
        return None
    held = majority_ids(spec)
    hex_lattice = spec["board"].get("lattice") == "hex"
    kind = [CONTROL_NONE] * len(pieces)
    neighbours: list[tuple[tuple[int, ...], ...]] = [() for _ in pieces]
    for piece_type, piece in enumerate(pieces):
        pid = piece["id"]
        if pid not in placed:
            continue
        kind[piece_type] = CONTROL_MAJORITY if pid in held else CONTROL_NEUTRAL
        if pid not in held:
            continue
        neighbours[piece_type] = _next_to(
            piece, hex_lattice, width, height, orientation, wrap, obstacles, radius,
        )
    royal_takers = takes_royal_ids(spec)
    return NeutralRule(
        kind=tuple(kind), neighbours=tuple(neighbours),
        takes_royal=tuple(
            piece_type for piece_type, piece in enumerate(pieces) if piece["id"] in royal_takers
        ),
    )


def _next_to(
    piece: dict[str, Any], hex_lattice: bool,
    width: int, height: int, orientation: str, wrap: WrapRule | None,
    obstacles: frozenset[int], radius: int,
) -> tuple[tuple[int, ...], ...]:
    """``[square]`` -> the squares *next to* a piece of this type standing there
    (`docs/plan-turncoat.md` §4.2): its control's geometry, or the lattice's king's step,
    landed through `_compile_square` (walls, absent cells, wrap). Shared by majority control
    (TC3) and a hire's or a bribe's adjacency (TC6)."""
    vectors, symmetry = control_geometry(piece, hex_lattice)
    step = {"moves": [{
        "kind": "leap", "vectors": [list(v) for v in vectors], "symmetry": symmetry,
    }]}
    per_square: list[tuple[int, ...]] = []
    for square in range(width * height):
        built = _compile_square(
            step, 0, square, width, height, orientation, wrap, obstacles, False, 0, radius,
        )
        per_square.append(tuple(sorted({t for entry in built[0] for t in entry[0]})))
    return tuple(per_square)


@dataclass(frozen=True)
class TerritoryRule:
    """Squares that remember who moved onto them.

    ``permanent`` is the cheap half and the only one that needs no Zobrist term: a
    colour that never changes is monotone, so a ply that lays one down is irreversible
    and resets the no-progress clock, so colour is constant inside any window a
    repetition test compares across. Two positions the repetition rule ever compares
    therefore always share the same colouring, exactly as they always share the same
    hands, and the key does not have to carry it.

    ``repaint`` gives that up: a square can go back and forth, so two positions with the
    same pieces can differ in colour, and the key has to say so.
    """

    mode: str  # permanent | repaint
    win_at: int  # 0 when the colouring is not itself a win condition


@dataclass(frozen=True)
class ScoringRule:
    """What to compare when the clock runs out, instead of calling it a draw.

    ``squares`` is per player, like every other region in this file, because "hold the
    centre" is symmetric and "reach their half" is not, and the compiler cannot tell
    which one a spec meant without being told the orientation.
    """

    count: str  # material | pieces | region | territory | points
    squares: tuple[frozenset[int], frozenset[int]] | None


@dataclass(frozen=True)
class NpcRule:
    """Wild pieces (schema 9): pieces neither player owns, each walking a route.

    Three board-sized tables and a start list, and nothing else -- because a route is a
    cycle of distinct squares and no square is on two routes, so *which square is next*
    is a fact about the square a wild piece stands on. That is what lets the core step
    every wild piece with one table read and never ask which one it is looking at, and
    it is why the validator refuses two routes that share a square. Schema 23's
    ``also_at`` puts several wild pieces on one route, which changes only ``starts``:
    they share the route's type and bounty, so every per-square entry is still the
    route's. The core steps the pieces of one route front first (``npc_step_order`` in
    ``game.rs``).

    ``next[square]`` is the square a wild piece standing here steps to, ``-1`` off every
    route and the square itself for a one-square route (a piece that never moves).
    ``bounty[square]`` is what taking the wild piece that walks this square is worth,
    ``0.0`` off every route. ``piece[square]`` is its type, ``-1`` off every route.
    ``starts`` is where each stands at ply zero, in ``rules.npcs`` order -- per entry the
    route's first square, then its ``also_at`` squares in route order -- with a mirrored
    entry's starts following its original's.

    **No movement frames.** A wild piece never reads a movement table; its type's
    ``moves`` are what it does in a player's hands, which under ``drops`` is a real case.
    Its threat is the square it steps to next, and that is one attack source per route
    square in a third owner's block of ``attack_sources``, built here from ``next``.
    """

    next: tuple[int, ...]
    bounty: tuple[float, ...]
    piece: tuple[int, ...]
    starts: tuple[tuple[int, int], ...]
    # Generation 36 (N1, `docs/archive/plan-npc.md` §4.1): ``(every, starts)`` per respawning half --
    # an entry with ``respawn``, then its mirrored half -- in ``rules.npcs`` order, which is
    # the order the phase's step 3 returns them in. ``starts`` are the half's own, first free
    # first. ``()`` when no entry comes back, which is every game before the rule.
    respawn: tuple[tuple[int, tuple[int, ...]], ...] = ()
    # Generation 38 (N5, `docs/archive/plan-npc.md` §4.7): a turning-back entry's route is a **line**,
    # walked first to last and back. ``prev[square]`` is the square a piece walking it back
    # steps to: the previous square of its line, the square itself at the line's first square
    # (nothing before it, as ``next`` holds the square itself at the line's last), and ``-1``
    # off every line, which is how the core knows a square is on one. ``()`` when no entry
    # turns back, which is every game before the rule; a line's squares also carry no attack
    # source (the square it threatens depends on what blocks it, so the rule is refused where
    # check is asked).
    prev: tuple[int, ...] = ()

    def on_line(self, square: int) -> bool:
        """Whether ``square`` is on a turning-back entry's line (generation 38)."""
        return bool(self.prev) and self.prev[square] >= 0


# Generation 40 (N3c): ``PounceRule.kind``'s code for a type that pounces on its attackers,
# the core's ``POUNCE_ATTACKERS``; the analysis tally reads it off the compiled rule.
POUNCE_ATTACKERS = POUNCE_KINDS.index("attackers") + 1


@dataclass(frozen=True)
class PounceRule:
    """Wild pieces that pounce (generation 39, N3b, `docs/archive/plan-npc.md` §3.2, §4.3), resolved.

    ``kind[type]``: 0 for a type that does not pounce, else ``POUNCE_KINDS.index(kind) + 1``
    (two entries of one type must agree, the validator says). The **prey order** -- a royal,
    then the higher declared value, then the nearer, then the mover's reading order -- is
    three tables, so the core sorts nothing geometric:

    * ``value_rank[type]``: the type's place among the declared values, highest 0, equal
      values equal;
    * ``reading_rank[player][square]``: the square's place in that player's reading order.
      Player 0 reads rank ascending, then file ascending; player 1 reads the image of each of
      player 0's squares under ``schema.reflect_square`` with the setup's orientation, the one
      reflection the spec already places the second army by, so a position and its mirror
      image choose mirror-image victims;
    * ``distance[a * n_squares + b]``: king's distance, the lattice's own on a hexagon.

    ``reach[type][origin]``: a ``"prey"`` pouncing type's captures from ``origin`` as ``(target, prefix,
    window, screens)``, the attack sources' own frame-0 entries (player 0, facing 0: a pounce
    is refused on a type whose moves read differently for the two players, a facing, a
    region or a move condition) grouped by origin instead of by target. A source that needs
    an unmoved piece never fires for a wild piece and is left out. ``()`` for a type that
    does not pounce on its prey (an ``"attackers"`` pouncer reads the players' own inverse
    table, generation 40). The attack table itself is not touched (§3.3), which is what keeps every
    other game's tables what they were.
    """

    kind: tuple[int, ...]
    value_rank: tuple[int, ...]
    reading_rank: tuple[tuple[int, ...], tuple[int, ...]]
    distance: tuple[int, ...]
    reach: tuple[tuple[tuple[tuple[int, tuple[int, ...], tuple[int, ...], int], ...], ...], ...]
    # Generation 41 (N3d, `docs/archive/plan-npc.md` §4.5): ``split[type]``, the most wild pieces of a
    # splitting type standing at once (0 for a type that does not split), or ``()`` where no
    # type splits, so a pounce game without the rule compiles to what it compiled before.
    split: tuple[int, ...] = ()


# Schema 19 (Q771): the eight king's directions a transit can come out in, in this fixed
# order -- the bit a momentum mask sets for a step is its index here. Absolute board
# directions, file then rank, after the second player's reflection: a direction is the
# *sign* of the vector the piece travelled (`docs/archive/plan-portals.md` §2.2).
PORTAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
)
# A momentum mask is sixteen bits per landing: bit ``d`` says a move-mode landing there
# travelled in direction ``d``, bit ``8 + d`` a capture-mode one. Split by mode because a
# transit is judged where the piece comes out -- a pawn's push may come out only onto an
# empty square and its diagonal only onto a piece it may take.
MOMENTUM_CAPTURE_SHIFT = 8


def momentum_bits(step_x: int, step_y: int, mode: int) -> int:
    """The momentum mask of one landing reached by a step of this sign, under ``mode``."""
    sign = ((step_x > 0) - (step_x < 0), (step_y > 0) - (step_y < 0))
    if sign == (0, 0):
        return 0
    direction = PORTAL_DIRECTIONS.index(sign)
    bits = 0
    if mode & MODE_MOVE:
        bits |= 1 << direction
    if mode & MODE_CAPTURE:
        bits |= 1 << (MOMENTUM_CAPTURE_SHIFT + direction)
    return bits


@dataclass(frozen=True)
class PortalRule:
    """Groups of portals (schemas 19 and 21, `docs/archive/plan-portals.md` §2,
    `docs/archive/plan-portals-2.md` §2.2), resolved into tables.

    **Each end is its own internal piece type**, appended after the declared pieces in
    ``rules.portals`` order, group then end (pair 1 orange, pair 1 blue, group 2's ends a,
    b, c, ...), with its group's moves and the id ``portal_end_ids`` gives
    it (``1o``, ``1b``, ``2a``, ...). It stands on the board as a byte of the third owner, the block a wild piece uses
    -- nobody's -- which the validator keeps free by refusing ``npcs`` beside portals. So
    every reader that already leaves a wild piece out of the two armies leaves a portal out
    too, and the board byte names which end stands where.

    What a portal *does* is three tables the core walks and never works out:

    * ``exits[player][square][direction]`` -- the square one king's step from ``square``
      in that direction under ``player``'s wrap permission, or -1 (off the board, a wall).
    * the momentum masks (``leap_dirs``, ``slide_dirs``, ``hop_dirs``), parallel to the
      movement tables' landings: which directions, under which mode, a piece travelled to
      arrive there (`momentum_bits`). Several bits where several moves of one piece reach
      one square, which is the player's choice of exit (§2.3).
    * ``attack_dirs[owner][square]`` -- the capture directions of every attack source, in
      ``attack_sources`` order, so check through a portal is read from the same table.

    The rest is per type: ``dest`` (the type of the end a piece that goes in here comes
    out of; -1 for a way out only and for a piece that is not a portal), ``admits[type]``
    (per player, a mask over the declared piece types: bit ``t`` set when that player's
    pieces of type ``t`` may go in by this end, from the end's ``enter`` and ``carries``),
    ``movers`` (which players may move it). ``group_of`` is each end's group, in end
    order. For a pair, ``dest`` is the other end and ``admits`` is every declared type where
    the passage opens the end: schema 19's ``partner`` and ``open_to``, renamed (schema 21).

    Schema 20 (Q807): ``no_back[type]``, true on every end of a group that sets
    ``no_straight_back`` -- once moved, the end may not be moved straight back onto the
    square it left on the very next move. The validator refuses a bar no move can meet
    (Q879; stored specs too, since Q1008d), so none compiles inert; the core keeps a bar it
    sets only when the player to move next is one of the end's ``movers`` (Q883). The core
    holds the last such move as ``Game::portal_bar`` and keys it with ``zobrist_bar`` (``[square]`` for the square the end
    left, ``[n_squares + square]`` for the one it stands on), drawn after every other key and
    only when some pair sets the flag, so no other key moves. Both empty otherwise.
    """

    declared_types: int
    end_types: tuple[int, ...]
    squares: tuple[int, ...]
    group_of: tuple[int, ...]
    dest: tuple[int, ...]
    admits: tuple[tuple[int, int], ...]
    movers: tuple[frozenset[int], ...]
    exits: tuple[tuple[tuple[int, ...], ...], ...]
    leap_dirs: Any
    slide_dirs: Any
    hop_dirs: Any
    attack_dirs: tuple[tuple[tuple[int, ...], ...], ...]
    no_back: tuple[bool, ...] = ()
    zobrist_bar: tuple[int, ...] = ()
    # Schema 21: per type, whether the end names the pieces it carries -- for the page's
    # words only (`viz.wrapped.portal_payload`); the core reads `admits`, which says it.
    carried: tuple[bool, ...] = ()
    # Schema 22 (Q843c): each end's rule set, in end order -- ``"momentum"`` or ``"fold"``
    # -- and the fold tables (`FoldRule`), ``None`` where no group is a fold.
    kinds: tuple[str, ...] = ()
    folds: FoldRule | None = None

    def is_end(self, piece_type: int) -> bool:
        return piece_type in self.end_types

    def is_fold(self, piece_type: int) -> bool:
        """Whether ``piece_type`` is a fold end (schema 22)."""
        if piece_type not in self.end_types or not self.kinds:
            return False
        return self.kinds[self.end_types.index(piece_type)] == "fold"

    def admitted(self, piece_type: int, player: int, mover: int) -> bool:
        """Whether ``player``'s piece of type ``mover`` may go in by end ``piece_type``."""
        return bool(self.admits[piece_type][player] >> mover & 1)


@dataclass(frozen=True)
class FoldRule:
    """Fold portals (schema 22; revised at schema 24, `docs/archive/plan-portals-2.md` §3R).

    A fold end is an internal type like a momentum end (`PortalRule`), but it never stands
    on the board byte: where each stands is game state (``Game::fold_at``), starting at
    ``starts`` (one square per end in end order, -1 for a momentum end). A landing on an
    empty fold may go through to the square of the end it leads to, which the core reads
    off ``PortalRule.dest`` and ``fold_at``; nothing about a fold is geometry, so this is
    all the compiler adds:

    * ``zobrist`` -- ``[end * n_squares + square]``, where each fold stands, drawn after
      every other key and only when a fold exists, so no other key moves.
    """

    starts: tuple[int, ...]
    zobrist: tuple[int, ...]


@dataclass(frozen=True)
class PromotionRule:
    """Where a pawn promotes, into what, and whether it may decline.

    The zone is a set of squares because a promotion *zone* is a real rule -- grand chess
    promotes anywhere in the last three ranks, and a hexagon's far edge is a zigzag rather
    than a rank (schema 14) -- and targets are per player because chigorin gives the two
    sides different armies to promote into.

    ``on_leaving`` is schema 15 (Q452): the piece promotes on a move that *starts* in its
    zone as well as on one that ends there, wherever it lands -- real shogi's rule, and
    what Fairy-Stockfish does for every piece in ``promotedPieceType``. With ``mandatory``
    every such move must promote. It is ``True`` only where the entry says so **and** the
    piece has a move from a square of its zone to a square outside it, so a flag that can
    never fire -- a chess pawn's, whose zone is the rank it cannot move from -- compiles
    to the game that was already there (`_leaves_its_zone`). A move that ends in the zone
    promotes as it always did, so the flag only ever adds promotions to moves out of it.
    En passant never promotes, on arriving or on leaving: its move word has no promotion.

    ``only_lost`` is schema 18 (Q783, Grand chess): a target is offered only while the
    mover's side has fewer pieces of that type on the board than ``opening[player]``
    says it started with, so a pawn promotes only into what its side has lost. A
    promotion with every target withheld is no move where the rule is ``mandatory`` and
    the plain move where it is not. ``opening`` is the compiled setup counted per
    player and type (`_settle_only_lost`), filled only where the flag is set, so an entry
    without it is the record it always was.
    """

    piece_type: int
    squares: tuple[frozenset[int], frozenset[int]]  # absolute destination squares per player
    targets: tuple[tuple[int, ...], tuple[int, ...]]  # per player
    mandatory: bool
    on_leaving: bool = False
    only_lost: bool = False
    opening: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class EntryPromotionRule:
    """What a piece becomes on *entering play* inside its promotion zone (schema 9).

    ``rules.promote_on_entry``. A piece reaches a square without moving there by exactly
    two routes -- it is dropped there, or a strike or contact leaves it standing there as
    a new type -- and neither is a move in which a promotion could be chosen. So the answer is
    resolved here, once, into one table: ``becomes[player][piece_type][square]`` is the
    type the piece turns into on arrival, or ``-1`` where nothing happens, which is every
    square outside the type's zone and every type with no promotion rule.

    Three readings are fixed by the table rather than left to the core:

    * it takes the **first** target the promotion rule lists for that player -- a
      designer's ordering is the only choice available where no player is choosing;
    * it fires whether or not the rule is ``mandatory`` -- ``mandatory`` is about a move's
      option to decline, and there is no move here;
    * it fires **once**: the piece it becomes stands, exactly as a piece promoted by
      moving does not promote a second time, so a chain of promotion rules cannot loop.

    ``None`` on the spec is every game before this one: a pawn dropped or struck onto the
    far rank stands there, and a chess pawn on the far rank never moves again.
    """

    becomes: tuple[tuple[tuple[int, ...], ...], ...]

    def at(self, player: int, piece_type: int, square: int) -> int:
        return self.becomes[player][piece_type][square]


@dataclass(frozen=True)
class CastlingRule:
    """Either a symmetric step count, or an explicit destination file per side.

    ``king_files`` is what chess960 needs and a step count cannot express: the king
    starts on a different file every game and always finishes on c or g, so the
    distance it travels is a consequence rather than a rule. ``None`` means the spec
    used ``king_steps`` and the old reading applies.
    """

    king_type: int
    rook_type: int
    king_steps: int
    king_files: tuple[int, int] | None = None  # (queenside, kingside) destination
    rook_files: tuple[int, int] | None = None


# Schema 26 (Q918a): two void rounds in a row bar the pair (`docs/reference/rounds.md` §2.7).
# The user ruled it fixed, *"nor configurable"*, so it is a constant here and never a key.
# It crosses as ``SimRule.void_run`` so the core reads it and never decides it; changing it
# would change every simultaneous game under an unchanged hash, so it is never an edit.
VOID_RUN = 2

# Schema 26: ``SimRule.tie``. Absent is the user's table (both eliminated).
TIE_BOTH = 0
TIE_INITIATIVE = 1
TIE_IMPOSSIBLE = 2


@dataclass(frozen=True)
class SimRule:
    """Simultaneous moves (schema 26, Q918a; `docs/reference/rounds.md` §5.3).

    * ``tie`` -- ``TIE_BOTH`` (absent: both eliminated), ``TIE_INITIATIVE`` or
      ``TIE_IMPOSSIBLE`` (the validator proved no tie occurs; the core treats it as both).
    * ``initiative_start`` -- who holds the initiative at the start: the second player (§3.1,
      E), resolved here so the core never decides it. Read only under the initiative.
    * ``void_run`` -- ``VOID_RUN``.
    * ``collision_strikes`` -- whether a collision's loser goes through the strike rule:
      true for ``rules.strike``, false for ``rules.contacts`` (a collision never fires a
      contact link, §4) and where there is no rule.
    * ``collision_allowed[a * n_types + b]`` -- a piece of type ``a`` eliminates one of
      type ``b`` when they collide: ``pieces[].beats`` with its default (``captures``, else
      every piece) resolved, over every type; a portal's row and column are false.
    * ``zobrist`` -- the initiative's key (only under the initiative, else 0), then one key
      per square for each of the two squares of a standoff entry (``[square]`` for player
      0's square, ``[n_squares + square]`` for player 1's), then one per entry state (1, a
      run of one void round; 2, barred). Drawn after every other key.
    * ``swap_collides``, ``both_score`` -- schema 27's chase switches (Q942a,
      `docs/archive/plan-round-chase.md` §3 R2, X1), each false when absent. R1's ``rest``
      left this rule in schema 29 (Q983a) for the piece-type property
      ``CompiledSpec.rest``, which an alternating game may declare too.
    * ``deal_initiative`` -- schema 27 (Q935a, `docs/plan-spy.md` E10): in a spy game under
      the initiative, the token is dealt to the announcer when the first announced round
      begins, so the answerer cannot win at will every tie it chooses to enter. False in
      every other game.
    * ``speed`` -- schema 28 (Q943a, `docs/archive/plan-speed.md` §4.2): ``[type]``, each type's
      speed (``pieces[].speed``, 0 where none is written, and 0 for a portal end), over every
      type. **Empty when no piece declares one**, so no earlier game's tables move and the
      core never enters its ordered round.
    """

    tie: int
    initiative_start: int
    void_run: int
    collision_strikes: bool
    collision_allowed: tuple[bool, ...]
    zobrist_initiative: int
    zobrist_squares: tuple[int, ...]
    zobrist_states: tuple[int, ...]
    swap_collides: bool = False
    both_score: bool = False
    deal_initiative: bool = False
    speed: tuple[int, ...] = ()

    @property
    def chases(self) -> bool:
        """Whether any of schema 27's chase switches is on."""
        return self.swap_collides or self.both_score


def _compile_simultaneous(
    rules: dict[str, Any],
    pieces: list[dict[str, Any]],
    index_of: dict[str, int],
    n_types: int,
    draws: tuple[int, tuple[int, ...], tuple[int, ...]] | None,
) -> SimRule | None:
    """`SimRule` for a simultaneous spec, ``None`` for an alternating one (§5.3)."""
    rule = rules.get("simultaneous")
    if not isinstance(rule, dict) or draws is None:
        return None
    tie = {"initiative": TIE_INITIATIVE, "impossible": TIE_IMPOSSIBLE}.get(
        str(rule.get("tie")), TIE_BOTH
    )
    declared = [piece["id"] for piece in pieces]
    allowed = [False] * (n_types * n_types)
    for piece in pieces:
        beaten = beats_of(piece, declared)
        for victim in beaten:
            allowed[index_of[piece["id"]] * n_types + index_of[victim]] = True
    source = rules.get("strike") if isinstance(rules.get("strike"), dict) else None
    initiative, squares, states = draws
    return SimRule(
        tie=tie,
        initiative_start=1,
        void_run=VOID_RUN,
        collision_strikes=source is not None and strike_is_live(source),
        collision_allowed=tuple(allowed),
        zobrist_initiative=initiative,
        zobrist_squares=squares,
        zobrist_states=states,
        swap_collides=simultaneous_switch(rules, "swap"),
        both_score=simultaneous_switch(rules, "both"),
        deal_initiative=tie == TIE_INITIATIVE and bool(spy_ids(pieces)),
        speed=_compile_speeds(pieces, index_of, n_types),
    )


def _compile_speeds(
    pieces: list[dict[str, Any]], index_of: dict[str, int], n_types: int
) -> tuple[int, ...]:
    """``SimRule.speed``: each type's speed, or empty when no piece declares one (§4.1's
    inert default)."""
    if not has_speeds(pieces):
        return ()
    speed = [0] * n_types
    for pid, value in piece_speeds(pieces).items():
        speed[index_of[pid]] = value
    return tuple(speed)


@dataclass(frozen=True)
class PurseRule:
    """TC4 and TC5 (`docs/plan-turncoat.md` §5.2, §5.4): coins each player holds, part of the
    position, and the market they buy from (``rules.purse``).

    ``start`` is both players' opening amount, 0 to ``PURSE_MAX``. A bounty paid under the
    rule also fills the payee's purse, capped at ``PURSE_MAX`` (what passes the cap is
    lost). ``zobrist[player * (PURSE_MAX + 1) + amount]`` is the key of ``player`` holding
    ``amount``, drawn last of all and only under the rule (`_build_zobrist`), so no other
    game's key moves.

    The market (TC5): ``market[i]`` is a type a player may buy and ``price[i]`` what it
    costs; ``squares[player][i]`` the empty-or-not squares a bought ``market[i]`` may be
    placed on (`schema.purse_squares`: the region, reflected, less walls and absent cells,
    inside the type's own region), ascending. The core picks the empty ones and computes no
    square.

    Hiring and bribing (TC6, §5.5): ``hire[type]`` is what hiring a nobody's piece of the type
    costs (its price; 0 where it cannot be hired) and ``bribe[type]`` what bribing an enemy
    piece of it costs (twice its price; 0 where it cannot be bribed), each over every type or
    **empty when nothing can be hired** (or bribed). ``neighbours[type][square]`` is what
    *next to* means for such a type (`_next_to`), empty for every other type and empty
    altogether when neither list is: the hirer or briber needs a piece of theirs on one.

    Who is paid (TC9c, §5.10): ``to_loser`` pays a capture's bounty to the player who owned
    the piece taken (nobody, for a piece nobody owns), not to the taker; the tally follows
    the same payee. ``False`` is ``rules.purse.bounty_to: "taker"``, today's rule.
    """

    start: int
    zobrist: tuple[int, ...]
    market: tuple[int, ...] = ()
    price: tuple[int, ...] = ()
    squares: tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] = ((), ())
    hire: tuple[int, ...] = ()
    bribe: tuple[int, ...] = ()
    neighbours: tuple[tuple[tuple[int, ...], ...], ...] = ()
    to_loser: bool = False

    @property
    def bribes(self) -> bool:
        """Whether an enemy piece can be bought (``rules.purse.bribe`` with a priced type)."""
        return bool(self.bribe)


def _compile_purse(
    spec: dict[str, Any], pieces: list[dict[str, Any]], index_of: dict[str, int],
    zobrist: tuple[int, ...],
    width: int, height: int, orientation: str, wrap: WrapRule | None,
    obstacles: frozenset[int], radius: int,
) -> PurseRule | None:
    """``CompiledSpec.purse`` from ``rules.purse`` (TC5, TC6), or ``None`` without one."""
    purse = purse_rule(spec)
    if purse is None:
        return None
    market = purse_market(spec)
    prices = piece_prices(spec)
    hired = hire_ids(spec)
    bribed = bribe_ids(spec)
    hire = [0] * len(pieces)
    bribe = [0] * len(pieces)
    neighbours: list[tuple[tuple[int, ...], ...]] = [() for _ in pieces]
    hex_lattice = spec["board"].get("lattice") == "hex"
    for piece_type, piece in enumerate(pieces):
        pid = piece.get("id")
        if pid in hired:
            hire[piece_type] = prices[pid]
        if pid in bribed:
            bribe[piece_type] = 2 * prices[pid]
        if pid in hired or pid in bribed:
            neighbours[piece_type] = _next_to(
                piece, hex_lattice, width, height, orientation, wrap, obstacles, radius,
            )
    contested = bool(hired or bribed)
    return PurseRule(
        start=int(purse["start"]),
        zobrist=zobrist,
        market=tuple(index_of[pid] for pid in market),
        price=tuple(prices[pid] for pid in market),
        squares=(
            tuple(tuple(sorted(purse_squares(spec, pid, 0))) for pid in market),
            tuple(tuple(sorted(purse_squares(spec, pid, 1))) for pid in market),
        ),
        hire=tuple(hire) if hired else (),
        bribe=tuple(bribe) if bribed else (),
        neighbours=tuple(neighbours) if contested else (),
        to_loser=purse_pays_loser(spec),
    )


def _compile_bounties(
    pieces: list[dict[str, Any]], index_of: dict[str, int], n_types: int
) -> tuple[int, ...]:
    """``CompiledSpec.bounty``: each type's ``pieces[].bounty`` (TC4), 0 where none is
    written, over every type (a portal end's is 0), or **empty when no type carries one**."""
    bounties = piece_bounties(pieces)
    if not bounties:
        return ()
    bounty = [0] * n_types
    for pid, worth in bounties.items():
        bounty[index_of[pid]] = worth
    return tuple(bounty)


def _compile_rests(
    pieces: list[dict[str, Any]], index_of: dict[str, int], n_types: int
) -> tuple[int, ...]:
    """``CompiledSpec.rest``: each type's rest (schema 29, Q983a, `docs/plan-rest.md` §4.3),
    0 where none is written, over every type (a portal end's is 0), or **empty when no piece
    declares one**, so no other game's tables move and the core never enters its rest."""
    if not has_rests(pieces):
        return ()
    rest = [0] * n_types
    for pid, value in piece_rests(pieces).items():
        rest[index_of[pid]] = value
    return tuple(rest) if any(rest) else ()


# Schema 30 (Q999a, `docs/archive/plan-passive.md` §3.2): one bit per effect, as the core reads them.
SIGHT_FREEZE = 1
SIGHT_DISARM = 2
SIGHT_SHIELD = 4
# Q1007c (`docs/archive/plan-schema-31.md` §3.7): the piece's opponent may also move it, and with
# ``captures_own`` make it take its own side's (non-royal) pieces. One command source carries
# ``SIGHT_COMMAND`` alone or with ``SIGHT_COMMAND_OWN``.
SIGHT_COMMAND = 8
SIGHT_COMMAND_OWN = 16
SIGHT_BITS = {
    "freeze": SIGHT_FREEZE, "disarm": SIGHT_DISARM, "shield": SIGHT_SHIELD,
    "command": SIGHT_COMMAND,
}

# One way a piece can be under an effect, inverted out of the forward lines:
# ``(origin, piece_value, effect bit, path)``, filed under the affected player, the affected
# type and the square seen. The source's piece stands on ``origin`` (the byte equal to
# ``piece_value``) and every square of ``path`` is empty: then the effect binds. A ray's path
# is the squares between the source and the seen square, those nearer than ``min_range``
# included; an aura's is empty.
SightSource = tuple[int, int, int, tuple[int, ...]]
# One forward line: ``(entry number, squares, seen)``, ``seen[i]`` false for a square nearer
# than ``min_range`` (not seen, but it still stops a ray).
SightLine = tuple[int, tuple[int, ...], tuple[bool, ...]]


@dataclass(frozen=True)
class SightEntry:
    """One compiled line of sight of a type: its effect's bit, ``whose`` as written, the
    affected types as a mask over declared types, and whether it is an aura."""

    effect: int
    whose: str
    affects: int
    through: bool


@dataclass(frozen=True)
class SightRule:
    """Lines of sight (schema 30, Q999a; `docs/archive/plan-passive.md` §2, §3.2).

    * ``entries[type]`` -- each type's sights, in the order written; empty for a type that
      sees nothing.
    * ``lines[frame][type][square]`` -- the forward lines, read off the slide tables
      `_compile_square_full` builds for each sight's geometry in that frame: what the type
      sees from that square on an empty board, before blocking. The page's hover and the
      oracle read these; the core never does. Where a wild type sees (generation 37, N2), one
      frame more, ``2 * n_facings``, the third owner's (`frame_at` of a wild byte).
    * ``sources[player][piece_type][square]`` -- the reverse table the core walks
      (`Game::effects_at`): every source whose effect could bind ``player``'s piece of
      ``piece_type`` standing on ``square``. Filed by the affected type, so a walk reads only
      the sources that affect what stands there (Q999a's benches: Madrasi's sights each affect
      one type).
    * ``has_shield`` -- whether any sight shields.
    * ``has_restraint`` -- whether any sight freezes or disarms.
    * ``has_command`` -- whether any sight commands (Q1007c): the core then walks the
      opponent's commanded pieces for a second source of moves. A command never touches a
      royal type, so no command source is filed under one.
    * ``bears_on_check`` -- whether any sight can change a check: a freeze or a disarm, or a
      shield that can cover a royal other than its own source (a royal never sees itself, so
      a king's shield bears on check only where a side can hold two royals). The core's pin
      shortcut is off when it is set.
    """

    entries: tuple[tuple[SightEntry, ...], ...]
    lines: tuple[tuple[tuple[tuple[SightLine, ...], ...], ...], ...]
    sources: tuple[tuple[tuple[tuple[SightSource, ...], ...], ...], ...]
    has_shield: bool
    has_restraint: bool
    bears_on_check: bool
    has_command: bool = False


def _compile_sights(
    spec: dict[str, Any],
    pieces: list[dict[str, Any]],
    index_of: dict[str, int],
    n_types: int,
    n_facings: int,
    facing_axes: tuple[int, ...],
    n_squares: int,
    width: int,
    height: int,
    orientation: str,
    wrap: WrapRule | None,
    obstacles: frozenset[int],
    blocks_leaps: bool,
    radius: int,
    royal_types: tuple[int, ...],
    multi_royal: bool,
) -> SightRule | None:
    """`SightRule` for a spec whose types declare a sight, ``None`` otherwise (§3.1's inert
    default: no table, and the core never enters a sight branch).

    A sight's squares are those a slide of its geometry lands on (§2.2), so each entry is
    compiled as a one-descriptor slide by `_compile_square_full`, which decides every step
    (walls, the edge, absent cells, wrap and the hex lattice); a wrapped line stops on a square
    already visited, so a piece never sees itself."""
    sights = piece_sights(spec)
    if not sights:
        return None
    rules = spec.get("rules") if isinstance(spec.get("rules"), dict) else {}
    declared = [piece["id"] for piece in spec["pieces"]]
    royal_ids = {pieces[royal]["id"] for royal in royal_types}
    entries: list[tuple[SightEntry, ...]] = []
    for piece in pieces:
        row: list[SightEntry] = []
        for sight in sights.get(piece["id"], ()):
            mask = 0
            for pid in sight_affected_ids(sight, declared, rules, royal_ids):
                mask |= 1 << index_of[pid]
            effect = SIGHT_BITS[sight.effect]
            if sight.captures_own:
                effect |= SIGHT_COMMAND_OWN
            row.append(SightEntry(effect, sight.whose, mask, sight.through))
        entries.append(tuple(row))

    n_frames = 2 * n_facings
    lines: list[list[list[tuple[SightLine, ...]]]] = []
    sources: list[list[list[list[SightSource]]]] = [
        [[[] for _ in range(n_squares)] for _ in range(n_types)] for _ in range(2)
    ]
    for frame in range(n_frames):
        player, facing = divmod(frame, n_facings)
        frame_lines: list[list[tuple[SightLine, ...]]] = []
        for piece_type, piece in enumerate(pieces):
            written = sights.get(piece["id"], ())
            if not written:
                frame_lines.append([() for _ in range(n_squares)])
                continue
            stands = not (facing and facing % (n_facings // facing_axes[piece_type]))
            piece_value = (player * n_facings + facing) * n_types + piece_type + 1
            type_lines: list[tuple[SightLine, ...]] = []
            for origin in range(n_squares):
                found: list[SightLine] = []
                for number, sight in enumerate(written):
                    slide = {
                        "kind": "slide", "vectors": [list(v) for v in sight.vectors],
                        "symmetry": sight.symmetry, "range": sight.range,
                        "min_range": sight.min_range,
                    }
                    built = _compile_square(
                        {"moves": [slide]}, player, origin, width, height, orientation, wrap,
                        obstacles, blocks_leaps, facing_quarter_turns(facing, n_facings), radius,
                    )
                    for ray, modes, _ in built[1]:
                        found.append((number, tuple(ray), tuple(mode != 0 for mode in modes)))
                type_lines.append(tuple(found))
                if not stands:
                    continue  # a facing this type can never be in: no source, as for attacks
                for number, ray, seen in found:
                    entry = entries[piece_type][number]
                    affected = {
                        "enemy": (1 - player,), "own": (player,), "any": (0, 1),
                    }[entry.whose]
                    for index, square in enumerate(ray):
                        if not seen[index]:
                            continue
                        path = () if entry.through else ray[:index]
                        for side in affected:
                            for victim in range(n_types):
                                if entry.affects >> victim & 1:
                                    sources[side][victim][square].append(
                                        (origin, piece_value, entry.effect, path)
                                    )
            frame_lines.append(type_lines)
        lines.append(frame_lines)

    # Generation 37 (N2, `docs/archive/plan-npc.md` §4.2): a wild piece's sight. A wild piece stands
    # in the third owner's one frame (its byte is `WILD * n_facings * n_types + type + 1`), its
    # lines read the same for both players (the validator refuses one that does not), so they
    # are compiled in player 0's frame, and its ``whose`` is ``"any"``: one source per seen
    # square for each player's piece. The forward lines get a frame of their own after the
    # players' (`frame_at` of a wild byte), and only where a wild type sees, so no other
    # game's tables move.
    listed_npcs = rules.get("npcs") if isinstance(rules, dict) else None
    wild_ids = {
        npc.get("piece") for npc in (listed_npcs if isinstance(listed_npcs, list) else [])
        if isinstance(npc, dict)
    }
    # Generation 42 (N4b): the duck stands in the same block and sees as a wild piece does.
    listed_duck = rules.get("duck") if isinstance(rules, dict) else None
    if isinstance(listed_duck, dict):
        wild_ids.add(listed_duck.get("piece"))
    wild_seeing = [
        piece_type for piece_type, piece in enumerate(pieces)
        if piece["id"] in wild_ids and sights.get(piece["id"])
    ]
    if wild_seeing:
        wild_lines: list[list[tuple[SightLine, ...]]] = [
            [() for _ in range(n_squares)] for _ in range(n_types)
        ]
        for piece_type in wild_seeing:
            piece_value = (WILD * n_facings) * n_types + piece_type + 1
            written = sights[pieces[piece_type]["id"]]
            for origin in range(n_squares):
                seen_from: list[SightLine] = []
                for number, sight in enumerate(written):
                    slide = {
                        "kind": "slide", "vectors": [list(v) for v in sight.vectors],
                        "symmetry": sight.symmetry, "range": sight.range,
                        "min_range": sight.min_range,
                    }
                    built = _compile_square(
                        {"moves": [slide]}, 0, origin, width, height, orientation, wrap,
                        obstacles, blocks_leaps, 0, radius,
                    )
                    for ray, modes, _ in built[1]:
                        seen_from.append((number, tuple(ray), tuple(mode != 0 for mode in modes)))
                wild_lines[piece_type][origin] = tuple(seen_from)
                for number, ray, seen in seen_from:
                    entry = entries[piece_type][number]
                    for index, square in enumerate(ray):
                        if not seen[index]:
                            continue
                        path = () if entry.through else ray[:index]
                        for side in (0, 1):
                            for victim in range(n_types):
                                if entry.affects >> victim & 1:
                                    sources[side][victim][square].append(
                                        (origin, piece_value, entry.effect, path)
                                    )
        lines.append(wild_lines)

    royal_mask = 0
    for royal in royal_types:
        royal_mask |= 1 << royal
    restraint = any(
        entry.effect & (SIGHT_FREEZE | SIGHT_DISARM) for row in entries for entry in row
    )
    # A shield bears on check where it can cover a royal: one of another type, or, where a
    # side can hold two royals, one of its own source's type (a piece never sees itself).
    royal_shield = any(
        entry.effect == SIGHT_SHIELD
        and entry.affects & royal_mask & ~(0 if multi_royal else 1 << source)
        for source, row in enumerate(entries) for entry in row
    )
    return SightRule(
        entries=tuple(entries),
        lines=tuple(tuple(tuple(row) for row in frame) for frame in lines),
        sources=tuple(
            tuple(tuple(tuple(found) for found in by_type) for by_type in side)
            for side in sources
        ),
        has_shield=any(entry.effect == SIGHT_SHIELD for row in entries for entry in row),
        has_restraint=restraint,
        bears_on_check=restraint or royal_shield,
        has_command=any(entry.effect & SIGHT_COMMAND for row in entries for entry in row),
    )


@dataclass(frozen=True)
class SpyRule:
    """The spy (schema 27, Q935a; `docs/plan-spy.md` §1, §2.4).

    * ``types`` -- each player's spy types: the flagged types that player has on the
      opening board (`_side_types`, Q936's helper, so a mirrored army has one spy type for
      both and disjoint armies one each). A player **has a spy** while at least one piece
      of any of its spy types stands on the board, the count summed over the list: a side
      given two spies has lost "the spy" when it has none left (§1.2), so two types are
      not two separate losses (`Game::has_spy`, never `short_of`, which counts each listed
      type apart).

    Nothing else: the phase is a function of the board and the side to move (§2.5), and the
    initiative's deal is ``SimRule.deal_initiative``.
    """

    types: tuple[tuple[int, ...], tuple[int, ...]]


def _compile_spy(
    pieces: list[dict[str, Any]], index_of: dict[str, int], board: bytes, n_types: int,
    n_facings: int,
) -> SpyRule | None:
    """`SpyRule` for a spec with a spy, ``None`` otherwise."""
    flagged = [index_of[pid] for pid in spy_ids(pieces) if pid in index_of]
    if not flagged:
        return None
    return SpyRule(types=_side_types(board, n_types, n_facings, flagged))


@dataclass(frozen=True)
class PushRule:
    """Q1074b1 (`docs/plan-push.md` §1 to §3): the push rule's tables. ``None`` on
    `CompiledSpec` in every game no move of which pushes, so no other game's tables move.

    The masks themselves are the movement tables' landing modes (`MODE_PUSH_ENEMY`,
    `MODE_PUSH_OWN`). Beside them:

    * ``chain`` (`PUSH_CHAIN`) and ``edge_removes``: the game-wide physics (§2);
    * ``directions``: each push direction as ``(player, dx, dy)``, the step a pushed piece
      takes in the board's own coordinates, keyed by the mover (whose wrap it reads, D1);
    * ``next[dir][square]``: where a piece on ``square`` goes, `PUSH_OFF` off the board (the
      edge, or a hexagon's absent cell, W2), `PUSH_STOP` (a wall) or, under ``walls``
      ``"crush"`` (Q1074b7), `PUSH_CRUSH` (a wall that removes the piece pressed on it);
      `_land` decides it, as it decides every step, so the core never computes a square;
    * ``leap_dirs``/``slide_dirs``/``hop_dirs`` ``[frame][type][square]``: per entry, per
      landing, the direction id of the push there, `PUSH_NO_DIRECTION` where it pushes nothing
      (parallel to the movement tables' landings, as portal momentum masks are);
    * ``allowed[pusher][victim]``: the type lists (`schema.push_lists`), owner-blind (whose is
      the landing's mode bits); never a wild type or the duck, either way (N1, N2);
    * ``weight[type]``: empty unless ``chain`` is ``"weight"``; a pushed piece's load, 0 for
      a type nothing may push;
    * ``strength[type]``: empty unless ``chain`` is ``"weight"``; a pusher's strength, its
      ``strength`` or else its weight (Q1074b4s, `schema.piece_strengths`), 0 for a type that
      never pushes: a line moves when its summed weight is at most this;
    * ``stand[player][type]``: per square, whether a pushed piece of that type may stand there
      (its ``region``, W5); ``()`` for a type that may stand anywhere, and ``()`` whole unless
      some type has a region;
    * ``attack_kinds``: parallel to `CompiledSpec.attack_sources`, what each source attacks by
      (`ATTACK_BY_CAPTURE`, `ATTACK_BY_PUSH`), so the core can read a push source against
      ``allowed`` and a capture source against the capture matrix (R1);
    * ``zobrist_bar`` (Q1074b6, K5, ``rules.push.no_push_back``): the push-back bar's keys,
      ``[x]`` and ``[n_squares + y]`` for a piece standing on ``y`` that was pushed there
      from ``x`` (``Game::push_bar``); drawn last of all and only under the rule
      (`_build_zobrist`), ``()`` otherwise, which is the one flag the core reads.
    """

    chain: int
    edge_removes: bool
    directions: tuple[tuple[int, int, int], ...]
    next: tuple[tuple[int, ...], ...]
    leap_dirs: Any
    slide_dirs: Any
    hop_dirs: Any
    allowed: tuple[tuple[bool, ...], ...]
    weight: tuple[int, ...]
    strength: tuple[int, ...]
    stand: tuple[tuple[tuple[bool, ...], ...], ...]
    attack_kinds: tuple[tuple[tuple[int, ...], ...], ...]
    zobrist_bar: tuple[int, ...] = ()

    @property
    def no_back(self) -> bool:
        """Whether a piece pushed may not push straight back on the next move (K5)."""
        return bool(self.zobrist_bar)


@dataclass(frozen=True)
class CompiledSpec:
    """Immutable, position-independent tables derived from one rule spec."""

    name: str
    hash: str
    width: int
    height: int
    # Schema 14: `"square"` or `"hex"`. A declaration that no table reads; the prose and the
    # three drawing surfaces read it to say and show the board is hexagonal.
    lattice: str
    n_squares: int
    n_types: int

    piece_ids: tuple[str, ...]
    piece_names: tuple[str, ...]
    piece_values: tuple[float, ...]
    royal_types: tuple[int, ...]

    # How many frames the movement tables hold: `2 * n_facings`, and the one number
    # every `[frame][...]` table's first dimension is a function of.
    n_frames: int
    # How many orientations this spec's pieces have between them -- the largest declared
    # `facing.axes`, or 1 where nothing faces. One for every ruleset before schema 8,
    # which is what makes a frame the player index and every one of those tables
    # byte-identical to what it compiled to before frames existed.
    n_facings: int
    # [piece_type] -> how many orientations *that* type has, and which one it enters play
    # in. A type with no facing has one orientation and enters play in it, so these are
    # all-ones and all-zeros for every spec before schema 8.
    facing_axes: tuple[int, ...]
    facing_initial: tuple[int, ...]

    # [frame][piece_type][square] -> entries. See `frame_index`: the first index is a
    # frame rather than a player, which is the same thing today and will not be once a
    # piece can face.
    leaps: tuple[tuple[tuple[tuple[Entry, ...], ...], ...], ...]
    slides: tuple[tuple[tuple[tuple[Entry, ...], ...], ...], ...]
    hops: tuple[tuple[tuple[tuple[HopEntry, ...], ...], ...], ...]
    # Schema 13: sweep rays, shaped as `slides`. Every slot is empty unless a piece sweeps.
    sweeps: tuple[tuple[tuple[tuple[SweepEntry, ...], ...], ...], ...]

    # [player][target square] -> every way that player could attack that square
    attack_sources: tuple[tuple[tuple[AttackSource, ...], ...], ...]

    # [player][piece_type] -> squares whose generated moves still need a uniqueness
    # check. Empty for every well-formed piece, so the generator normally skips it.
    dedup_squares: tuple[tuple[frozenset[int], ...], ...]

    # Whether any piece attacks by hopping. Hops invert the usual blocking logic --
    # adding a piece to a line can *create* an attack -- which defeats the pin
    # shortcut in legal-move filtering, so such rulesets take the slow, safe path.
    has_hop_attacks: bool

    # [attacker type][victim type] -> may the first take the second (schema 3).
    #
    # All true unless a piece declares ``captures``, which is what makes the feature
    # inert on every spec written before it existed. It cannot be baked into the
    # per-square tables the way a region or a lame leap can: what a move is allowed to
    # do depends on the *victim's* type, which is a property of the position rather
    # than of the geometry. So the core consults it at the capture test and at every
    # consumer of ``attack_sources`` -- the attacker's type travels there already, as
    # the piece value.
    capture_allowed: tuple[tuple[bool, ...], ...]
    # Whether any piece restricts what it may capture. False lets the core skip the
    # lookup entirely, which is every committed ruleset.
    has_capture_matrix: bool

    initial_board: bytes
    initial_unmoved: int

    win: str
    stalemate: str
    max_plies: int
    repetition_draw: int
    no_progress_draw: int
    en_passant: bool
    # One rule per promoting piece. Shogi has six; chess has one; most rulesets that
    # want promotion at all want exactly one, so the common case is a one-element tuple.
    promotions: tuple[PromotionRule, ...]
    # What each piece reverts to when captured, indexed by piece type: a promoted shogi
    # piece goes back into its captor's hand as the modest thing it started as.
    demotions: tuple[int, ...]
    castling: CastlingRule | None

    # Schema 2 endings. Each is off by default and costs a compiled-away branch when
    # unused, so a ruleset that wants none of them pays for none of them.
    bare_royal: str  # "" when off, else the result for the player who bares
    check_limit: int  # 0 when off
    reach: ReachRule | None
    extinction: ExtinctionRule | None
    # Schema 3: how a game decides itself at the ply limit or the no-progress clock.
    # ``None`` keeps the draw those clocks have always produced.
    scoring: ScoringRule | None
    # Schema 4: whether the board keeps a record of who has been where.
    territory: TerritoryRule | None
    # Schema 4: how many moves a turn is. ``None`` is one, which is every game so far.
    turn: TurnRule | None
    # Schema 5: which edges each player may cross. ``None`` is a board with edges,
    # which is every game before this one, and it keeps the compile path identical.
    wrap: WrapRule | None
    # Schema 6: squares nothing may occupy. Resolved once here and then *dissolved* --
    # every consumer reads the per-square tables, which already have the walls compiled
    # out of them. Kept on the spec because a board has to be drawable, not because the
    # engine consults it.
    #
    # Q281a2 (unreachable until the gate) folds a hexagon's absent frame cells in here
    # too, so this holds more than its name says: every square nothing may ever stand
    # on, whether a declared wall or a corner the shape simply does not have. Every
    # reader that asks "can a piece be here" gets the right answer from one set; the
    # only place a corner is still visible is the drawing, which reads ``absent``.
    obstacles: frozenset[int]
    # Schema 6: what happens instead of removal when a piece is taken. ``None`` is
    # removal, which is every game before this one.
    strike: StrikeRule | None
    # Schema 9: pieces neither player owns. ``None`` is every game before this one, and
    # ``n_owners`` is 2 then and 3 otherwise -- the one number the byte encoding, the
    # Zobrist draw and the attack-source table grow by.
    npcs: NpcRule | None
    n_owners: int
    # Schema 9: a piece that enters play inside its promotion zone promotes at once.
    # ``None`` is every game before this one, and the reason the field is a rule rather
    # than a flag is that the compiler has already answered *into what*.
    entry_promotion: EntryPromotionRule | None
    drops: DropRule | None
    explosion: ExplosionRule | None
    must_capture: bool
    forbid_immobility: bool
    royal_facing_forbidden: bool
    forbid_check: bool
    checkmate: str
    orientation: str

    # [player][piece_type] -> squares from which that piece has no move at all. A
    # shogi pawn on the far rank is the case that matters: declining to promote there
    # would strand it forever, which is why shogi forbids the choice rather than the
    # position.
    immobile_squares: tuple[tuple[frozenset[int], ...], ...]

    # [player][piece_type][square] -> how many moves that piece needs, on an empty
    # board, to get from that square to the nearest square a ``reach`` rule wins on.
    # ``UNREACHABLE`` where it cannot get there at all, and empty when the ruleset has
    # no reach rule.
    #
    # A *fact about the rules*, which is why it is compiled here beside the attack
    # sources rather than derived beside the piece values. An evaluator that wanted "how
    # far is this piece from a winning square" and answered it with geometry would be
    # wrong about every leaper: a knight's distance to a rank is not the number of
    # ranks, and a piece that only moves forward may not be able to come back at all.
    goal_distance: tuple[tuple[tuple[int, ...], ...], ...]
    # The largest finite entry above: the diameter of this board *for these pieces*,
    # which is the only honest normaliser for a distance measured in moves.
    goal_diameter: int
    # Q708 (F1): [player][goal][piece_type][square] -> the same distance to one goal
    # square alone, goals in ascending square order. Empty unless a priced race gives a
    # side more than one goal square: the evaluator reads it only when some of a side's
    # goals are held by a piece that side can never capture (`evaluation.sealed_goals`).
    goal_distance_each: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]

    # Whether a player can ever hold more than one royal piece at once. The check
    # filter is the hottest query in the engine and the single-royal case can answer
    # "where is it?" with one ``bytes.find``; anything more has to collect them all.
    # Deciding once, here, keeps the overwhelmingly common ruleset on the fast path.
    multi_royal: bool

    # [player][origin] -> the destinations that open an en passant window, each with the
    # square the mover stepped over. Empty for every ruleset without en passant.
    ep_passes: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]

    # [square] -> ((the far end of a file line, the squares strictly between), ...).
    # Xiangqi's flying general, as a fact about the board rather than as arithmetic. See
    # `_facing_lines`. Empty for every ruleset without `royal_facing: forbidden`.
    facing_lines: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]

    zobrist_pieces: tuple[tuple[int, ...], ...]
    zobrist_side: int
    zobrist_ep: tuple[int, ...]
    # Per square colour, for a repainting territory rule. Unused by everything else.
    zobrist_colour: tuple[tuple[int, ...], ...]
    # Per moves-remaining, for a turn longer than one move.
    zobrist_turn: tuple[int, ...]

    # The JSON this was compiled from, as text.
    #
    # The tables above are what a rule *means* to this engine; they are not what it
    # means in general, and translating a ruleset for a foreign engine needs the
    # descriptors rather than the per-square expansion of them -- "slides orthogonally
    # for ever" is not recoverable from a list of squares without inferring it back.
    # Carried as text rather than as the dict so the dataclass stays frozen and
    # hashable, and last so that every existing constructor call keeps working.
    #
    # ``chess_variant_sandbox/fairy/`` is the only reader. Nothing in ``engine/`` consults it, and
    # nothing may: a second interpretation of the DSL is exactly what this project
    # spends its design budget avoiding.
    source: str = ""

    # Schema 10 (Q51): the types that declared no move descriptor at all.
    #
    # A fact about the spec rather than a table, and it crosses nothing -- the core needs
    # nothing for it, because a type with no descriptors already has no movement table
    # and the generator emits nothing for it. It is here so that readers holding only the
    # compiled spec can *ask* -- `fairy` refuses such a piece, `describe` says so out
    # loud -- rather than each scanning the movement tables for emptiness and arriving at
    # slightly different answers. Empty for every ruleset before schema 10, and last for
    # the reason `source` is last.
    moveless_types: frozenset[int] = frozenset()

    # Q113: the types that can never return to a square they left.
    #
    # A move by one of these is *progress*, and both of the core's two tests for progress
    # read it: the repetition window since Q113, the no-progress clock since Q135. Each
    # used to be keyed on an unrelated rule's spelling -- the window on ``en_passant``,
    # the clock on ``promotion`` -- so *a pawn move is progress*, a fact about chess, was
    # switched on twice over by declarations that say nothing of the sort.
    # Derived from the movement tables by ``_one_way_types``, which is where the geometry
    # already is; the core reads it as a flag per type and walks no graph.
    #
    # Empty for a game whose every piece can come back, which is what keeps the boundary
    # bytes of such a ruleset exactly what they were before this existed.
    one_way_types: frozenset[int] = frozenset()

    # Schema 13: whether any piece sweeps. False keeps every boundary key of the fourth
    # table empty, which is what keeps earlier rulesets' fingerprints where they were.
    has_sweep: bool = False

    # Q281a2 (unreachable until the gate, `docs/reference/hex-build-plan.md` *Q281a2*):
    # a hexagon's frame cells folded into ``obstacles`` above, kept apart too so the
    # drawing can tell a corner from a wall. This and ``copies`` below are both empty
    # wherever `board.shape` is not `"hexagon"`.
    absent: frozenset[int] = frozenset()
    # Q281a2: a hexagon's six neighbouring copies, with their side names and
    # permissions, for the drawing and the designer (Q281b, Q281c) -- neither has to
    # work a copy vector out for itself. Empty on a grid.
    copies: tuple[HexCopy, ...] = ()
    # Q281c3: a joined rectangle's copies, shaped exactly like ``copies`` -- one past each
    # joined edge (``left``/``right`` a whole width away, ``bottom``/``top`` a whole
    # height) and, on a torus, the four corner copies reached by crossing both, named
    # ``"<file side>+<rank side>"``. For the designer's short step across a seam, so it
    # works no square out for itself. Kept apart from ``copies`` because the drawings
    # read that one as a hexagon's; empty on a hexagon and on a board with edges, and
    # read by nothing that plays a game.
    grid_copies: tuple[HexCopy, ...] = ()
    # Schema 19 (Q771): pairs of portals, or ``None`` -- every game before it, whose
    # boundary keys for the rule all cross empty. See ``PortalRule``.
    portals: PortalRule | None = None
    # Schema 25 (Q869): [piece_type] -> whether a move may turn that type, for a spec in
    # which some facing piece says ``turns: false``. Empty where every facing piece turns,
    # which is every spec before schema 25 and what keeps their boundary bytes where they
    # were; read it through ``turns``.
    facing_turns: tuple[bool, ...] = ()
    # Schema 26 (Q918a): simultaneous moves, or ``None`` -- every alternating game, whose
    # boundary keys for the rule all cross empty. See ``SimRule``.
    simultaneous: SimRule | None = None
    # Schema 27 (Q935a): the spy, or ``None`` -- every game without one, whose boundary keys
    # for the rule cross empty. See ``SpyRule``. A spy game always has ``simultaneous``.
    spy: SpyRule | None = None
    # Schema 29 (Q983a, `docs/plan-rest.md` §3, §4.3): [piece_type] -> how many of its
    # player's turns a piece of that type rests after it makes its player's move, 0 for
    # never. The type is the one that **stands at the end of the move** (a pawn promoted to
    # a queen rests as the queen), which the core reads off the board, so nothing downstream
    # needs a rule of its own. Empty when no type declares a rest -- every game before
    # schema 29 -- which is what keeps their boundary bytes and keys where they were.
    rest: tuple[int, ...] = ()
    # One key per (turns left, player, square) for a resting piece,
    # ``((left - 1) * 2 + player) * n_squares + square``, ``left`` 1 to ``max(rest)``; drawn
    # last of all and only under the rule (`_build_zobrist`). For a game whose largest rest
    # is 1 this is schema 27's layout, ``player * n_squares + square``. Empty otherwise.
    zobrist_rest: tuple[int, ...] = ()
    # Generation 36 (N1): the respawn clocks' keys, ``[half's base + clock - 1]`` in
    # ``npcs.respawn`` order, drawn after the rest's and only where an entry comes back.
    zobrist_respawn: tuple[int, ...] = ()
    # Generation 38 (N5): a wild piece's heading on a line, ``[square]`` the key of a piece
    # there walking its line back (a piece walking forward is keyed by nothing), board-sized
    # with 0 off every line, drawn after the respawn clocks' and only where an entry turns
    # back. Empty otherwise.
    zobrist_heading: tuple[int, ...] = ()
    # Schema 30 (Q999a, `docs/archive/plan-passive.md` §3.2): lines of sight, or ``None`` -- every
    # game in which no type sees, whose boundary keys for the rule all cross empty. An effect
    # is a function of the board, so the rule adds no state and no key.
    sight: SightRule | None = None
    # Generation 39 (N3b, `docs/archive/plan-npc.md` §4.3): wild pieces that pounce, or ``None`` -- every
    # game in which no ``npcs`` entry says ``pounce``, whose boundary keys for it cross empty.
    pounce: PounceRule | None = None
    # Generation 42 (N4b, `docs/archive/plan-npc.md` §4.6): the duck, or ``None`` -- every game without
    # ``rules.duck``, whose boundary key for it crosses empty.
    duck: DuckRule | None = None
    # TC1 (`docs/plan-turncoat.md` §3.2): each type's control kind in the third owner's block,
    # or ``None`` -- every game in which nobody owns anything, whose boundary key crosses empty.
    control: ControlRule | None = None
    # TC3 (`docs/plan-turncoat.md` §4): pieces nobody owns from the setup, or ``None`` -- every
    # game without a ``player: "nobody"`` placement, whose boundary keys for it cross empty.
    neutrals: NeutralRule | None = None
    # TC4 (`docs/plan-turncoat.md` §5.1): [piece_type] -> what taking a piece of that type
    # pays the player whose move took it (``pieces[].bounty``), 0 for none, over every type;
    # **empty when no type carries one**, so no other game's tables move. A wild piece's
    # bounty stays its route's (``npcs.bounty``).
    bounty: tuple[int, ...] = ()
    # TC4 and TC5 (§5.2, §5.4): the purse and its market, from ``rules.purse``, or ``None``
    # -- every game without one. See ``PurseRule``.
    purse: PurseRule | None = None
    # Q1069a: the draw by insufficient material, from ``rules.insufficient_material``, or
    # ``None`` -- every game without it. See ``InsufficientMaterialRule``.
    insufficient_material: InsufficientMaterialRule | None = None
    # Q1074b1: the push rule's tables, or ``None`` -- every game no move of which pushes.
    push: PushRule | None = None

    @property
    def rest_max(self) -> int:
        """The largest rest any type declares, 0 without the rule."""
        return max(self.rest, default=0)

    def turns(self, piece_type: int) -> bool:
        """Whether a move may turn this type: it has more than one orientation and its
        facing does not say ``turns: false``. What the core asks before it generates a
        rotation (`Tables::turns_of`)."""
        if self.facing_axes[piece_type] < 2:
            return False
        return not self.facing_turns or self.facing_turns[piece_type]

    def sweep_path(self, board: bytes, origin: int, target: int) -> tuple[int, ...]:
        """The squares a sweep from ``origin`` to ``target`` passes over, in ray order.

        Read off the fourth table for the piece standing on ``origin``, as
        `Tables::sweep_path` reads it in the core. Empty when none of its sweeps reaches
        ``target``: one ray reaches one target, because ``wrap`` is refused beside a sweep.
        """
        value = board[origin]
        if not self.has_sweep or not value:
            return ()
        for ray, *_ in self.sweeps[self.frame_at(value)][self.type_of(value)][origin]:
            if target in ray:
                return ray[: ray.index(target)]
        return ()

    def sweep_victims(self, board: bytes, move: int) -> tuple[tuple[int, int], ...]:
        """The pieces ``move`` sweeps off ``board``, as ``(square, value)`` in ray order:
        every occupied square of its path, read off the position it is played in.

        The one answer the Python side gives to *what does a sweep remove*, mirroring
        `Tables::sweep_victims` in the core; every reader that names or prices a sweep's
        victims (`minimax._capture_gain`, `Engine.describe_move`, `viz.wrapped.sweep_transit`)
        asks this. Empty for a move that is not a ``MOVE_SWEEP``.
        """
        if move_type(move) != MOVE_SWEEP:
            return ()
        return tuple(
            (square, board[square])
            for square in self.sweep_path(board, move_from(move), move_to(move))
            if board[square]
        )

    def push_takes(self, board: bytes, move: int) -> bool:
        """Q1074b3 (`docs/plan-push.md` §3.3): whether ``move`` is a push that throws an
        enemy piece off the board, read off the position it is played in.

        The one answer the Python side gives to *does this push take*, mirroring
        `Tables::push_takes` in the core: a removal counts as a capture where a capture
        takes an enemy piece (`must_capture`, quiescence, the ordering's gain). A push that
        throws the pusher's own piece off changes material (the clocks, the window) but
        takes nothing. ``False`` for any word that is not a ``MOVE_PUSH``.
        """
        return self.push_taken(board, move) != 0

    def push_removed(self, board: bytes, move: int) -> int:
        """Q1074b4 (`docs/plan-push.md` §2, C5): the board byte ``move`` throws off the board,
        whoever's it is, or 0 where it throws none off (``0`` too for any word that is not a
        ``MOVE_PUSH``). Under ``chain`` ``"one"`` that is the target's piece; under ``"row"``
        and ``"weight"`` the last of the line (the target's piece and every piece behind it
        up to the first empty square, the pusher's origin counting as empty). Read off the
        position the move is played in; the move was offered, so its line may move.
        `Tables::push_taken` walks the same line in the core."""
        line = self.push_line(board, move)
        if line and line[-1][1] in (PUSH_OFF, PUSH_CRUSH):
            return board[line[-1][0]]
        return 0

    def push_line(self, board: bytes, move: int) -> list[tuple[int, int]]:
        """Q1074b5: what ``move`` pushes, read off the position it is played in, as
        ``(square, to)`` for each piece of its line in order from the target, ``to`` the square
        it is pushed onto or `PUSH_OFF` for the one thrown off the board (`PUSH_CRUSH` for one
        crushed on a wall, Q1074b7). Empty for any word
        that is not a ``MOVE_PUSH``. The walk `push_removed` reads its removal from, and the
        page its animation (`viz.wrapped.push_transit`); the move was offered, so its line may
        move."""
        if move_type(move) != MOVE_PUSH or self.push is None:
            return []
        origin, square = move_from(move), move_to(move)
        following = self.push.next[move_push_direction(move)]
        line: list[tuple[int, int]] = []
        for _ in range(self.n_squares):
            behind = following[square]
            line.append((square, behind))
            if behind < 0 or behind == origin or not board[behind]:
                return line
            square = behind
        return line

    def push_taken(self, board: bytes, move: int) -> int:
        """Q1074b4: the enemy piece ``move`` throws off the board (`push_removed`), or 0
        where it throws none off or throws off the pusher's own. `Tables::push_taken`'s
        copy."""
        removed = self.push_removed(board, move)
        if removed and self.owner_of(removed) != self.owner_of(board[move_from(move)]):
            return removed
        return 0

    def square_index(self, file: int, rank: int) -> int:
        return rank * self.width + file

    def square_file(self, square: int) -> int:
        return square % self.width

    def square_rank(self, square: int) -> int:
        return square // self.width

    def relative_rank(self, square: int, player: int) -> int:
        """Rank counted from ``player``'s own back rank."""
        rank = square // self.width
        return rank if player == 0 else self.height - 1 - rank

    def piece_at(self, board: bytes, square: int) -> tuple[int, int] | None:
        """Return ``(player, piece_type)`` at ``square``, or ``None`` if empty."""
        value = board[square]
        if value == 0:
            return None
        return self.owner_of(value), self.type_of(value)

    def encode_piece(self, player: int, piece_type: int, facing: int = 0) -> int:
        """The board byte for one piece, **player-major**.

        ``player * n_facings * n_types + facing * n_types + type + 1``. The ordering is
        load-bearing rather than arbitrary: it keeps one player's values contiguous, so
        "is this piece mine?" stays the range test it has been since the first commit and
        only its stride moves. With ``n_facings`` of 1 it is the arithmetic it always was.

        ``player`` may be ``WILD`` (2) for a piece neither player owns, since schema 9: a
        third contiguous block above both armies, at facing 0, because a wild piece never
        turns and never reads a frame.
        """
        return (player * self.n_facings + facing) * self.n_types + piece_type + 1

    def owner_of(self, value: int) -> int:
        """Who owns a board byte: 0, 1, or ``WILD`` for a piece neither player owns.

        Never ``value // n_types``: that is the frame.
        """
        return (value - 1) // (self.n_facings * self.n_types)

    def type_of(self, value: int) -> int:
        return (value - 1) % self.n_types

    def frame_at(self, value: int) -> int:
        """The frame a piece reads its movement out of, straight off its board byte."""
        return (value - 1) // self.n_types

    def facing_of(self, value: int) -> int:
        return self.frame_at(value) % self.n_facings

    @property
    def has_facing(self) -> bool:
        return self.n_facings > 1

    @property
    def side_stride(self) -> int:
        """How many board values one player owns. `(value - 1) // this` is the owner."""
        return self.n_facings * self.n_types

    def facing_headings(self) -> tuple[tuple[int, ...], ...]:
        """`[player][facing] -> degrees clockwise` a piece at that facing points on screen.

        For the drawing surfaces, which have to say which way a piece is pointing or the
        game cannot be played by a person. Derived by putting the piece's own forward
        vector through the same two transforms `_compile_square` puts every vector
        through -- the facing rotation, then the player's reflection -- rather than
        worked out from the facing index, because the answer is **not** `facing * 90` for
        both players.

        Under `mirror` the second player's vectors are reflected, and a reflection
        reverses handedness: the same facing index turns clockwise on screen for the
        first player and anticlockwise for the second. That is the chirality asymmetry
        `setup.mirror` has always had -- invisible in chess because a knight is achiral
        -- and a renderer that assumed `facing * 90` would draw one army pointing the
        wrong way, which is precisely the kind of geometry-by-arithmetic this project
        keeps out of the core.
        """
        headings: list[tuple[int, ...]] = []
        for player in (0, 1):
            per_facing: list[int] = []
            for facing in range(self.n_facings):
                dx, dy = _rotate(0, 1, facing_quarter_turns(facing, self.n_facings))
                if player == 1:
                    dx, dy = (-dx, -dy) if self.orientation == "rotate" else (dx, -dy)
                # Screen y runs the other way from board rank, and the icons are drawn
                # pointing up, so "up" is zero and the angle grows clockwise.
                per_facing.append({(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}[(dx, dy)])
            headings.append(tuple(per_facing))
        return (headings[0], headings[1])

    def facing_step(self, piece_type: int) -> int:
        """How many facing indices one turn of this piece covers.

        A piece with two axes on a board whose spec has four steps by two: its
        orientations are the spec's forward and backward, and the two sideways facings
        are ones it can never be in.
        """
        return self.n_facings // self.facing_axes[piece_type]

    @property
    def coat_facing(self) -> tuple[tuple[int, ...], ...]:
        """TC2 (`docs/plan-turncoat.md` §6): ``[from_owner][facing] -> facing`` under the other
        owner that points the same way on the board. Empty where nothing faces.

        A piece that changes sides on the board keeps its heading: its colour changes, its
        arrow does not. See ``_coat_facing``.
        """
        if not self.has_facing:
            return ()
        return _coat_facing(self.facing_headings())

    def has_royalty(self) -> bool:
        return bool(self.royal_types)

    def has_v2_endings(self) -> bool:
        """Whether ``terminal`` has to consider anything beyond the schema-1 endings."""
        return bool(
            self.bare_royal or self.check_limit or self.reach or self.extinction
            or self.scoring or self.territory
        )


def _obstacles_with_absent(
    rules: Any, board: Any, width: int, height: int, orientation: str
) -> frozenset[int]:
    """Every square nothing may stand on: walls, and (Q281a2) a hexagon's absent frame
    cells folded in beside them.

    One helper for both direct callers (`compile_spec`, `pieces_that_can_leave_their_zone`)
    so a third caller cannot forget the fold -- the mistake `docs/reference/hex-build-
    plan.md` *Q281a2* names by its second point. ``CompiledSpec.obstacles`` then means
    more than its name says, which is documented on the field itself.
    """
    return obstacle_squares(rules, width, height, orientation) | absent_squares(board)


def _hexagon_radius(board: Any) -> int:
    """A hexagon board's R, or 0 for a grid (Q281a2). 0 also selects the grid reading of
    ``_land`` everywhere it is threaded through, so a malformed or not-yet-hexagon board
    never takes the hex path.
    """
    if not isinstance(board, dict) or board.get("shape") != "hexagon":
        return 0
    width = board.get("width")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1 or width % 2 == 0:
        return 0
    return (width - 1) // 2


def compile_spec(spec: dict[str, Any]) -> CompiledSpec:
    """Validate ``spec`` and build its tables. Raises ``SpecError`` if invalid."""
    compiled, _ = _compile(spec, None)
    assert compiled is not None  # only a shared frame pass can stop early
    return compiled


def compile_two_readings(
    spec: dict[str, Any], restated: dict[str, Any]
) -> tuple[CompiledSpec, CompiledSpec | None]:
    """``spec`` compiled, and ``restated`` compiled beside it sharing what it can.

    The one caller is `relate._orientation_played`, which asks whether a ``"rotate"``
    spec's tables equal those of the same game ``restated`` under the default ``mirror``
    with every placement and wall written square by square. That needed two whole
    compiles, and the movement frames -- `_compile_square_full` once per frame, type and
    square -- are most of each (Q866). This builds them once and a half:

    * **The first player's frames are shared.** `_compile_square_full` reads the
      orientation only for the second player (the step's reflection and
      `_region_squares`), so wherever everything else it reads is equal (`_Frames.inputs`:
      the pieces, the board, the wrap, the walls, the facings, the portals) the first
      player's half of ``restated``'s frames *is* ``spec``'s, and is taken rather than
      rebuilt. Where anything differs nothing is shared and both compile whole.
    * **The second player's frames are compared as they are built**, and the first square
      whose entries differ from ``spec``'s ends the restated compile: the answer is then
      ``(compiled, None)``, meaning *the two compile to different tables*. That is exact,
      not a heuristic. `bridge._flatten_entries` writes every slot's entries in frame,
      type, square order, each with its landing count beside its landings and a mode per
      landing (true of every entry built here: leaps, rays and lame legs alike), so two
      flattenings are equal only when every slot's entries are, and the movement arrays
      are canonical-table keys that `relate._played` keeps. Nothing after the frame pass
      rewrites them (`_freeze` is the identity on content).

    ``(compiled, compiled_restated)`` otherwise, and the restated one is exactly
    `compile_spec(restated)`: the same frames, then the same everything else. Raises as
    `compile_spec` does for ``spec``; for ``restated`` it raises only where its compile
    gets that far, which for this caller is the same answer (any failure to ask keeps the
    orientation as written). `tests/test_relate.py` holds the old two compiles against
    this over every committed rotating spec.
    """
    compiled, frames = _compile(spec, None)
    assert compiled is not None and frames is not None
    restated_compiled, _ = _compile(restated, frames)
    return compiled, restated_compiled


@dataclass(frozen=True)
class _Frames:
    """One compile's movement frames, `[frame][type][square]`, and what they were built
    from, so that `compile_two_readings` can share the first player's half."""

    # Everything `_compile_square_full` reads but the player and the orientation.
    inputs: tuple[Any, ...]
    leaps: list[list[list[tuple[Entry, ...]]]]
    slides: list[list[list[tuple[Entry, ...]]]]
    hops: list[list[list[tuple[HopEntry, ...]]]]
    sweeps: list[list[list[tuple[SweepEntry, ...]]]]
    dedup: list[list[frozenset[int]]]
    # Schema 19: the momentum masks, parallel to leaps, slides and hops. Built only where
    # there are portals; every other game compiles exactly as it did.
    leap_dirs: list[list[list[Any]]]
    slide_dirs: list[list[list[Any]]]
    hop_dirs: list[list[list[Any]]]
    # Q1074b1: each square's push steps (`PushSteps`), ``[frame][type][square]``. Built only
    # in a push game; every other game compiles exactly as it did.
    push_at: list[list[list[PushSteps]]]


def _compile_frames(
    pieces: list[dict[str, Any]],
    n_facings: int,
    n_squares: int,
    width: int,
    height: int,
    orientation: str,
    wrap: WrapRule | None,
    obstacles: frozenset[int],
    blocks_leaps: bool,
    radius: int,
    has_portals: bool,
    share: _Frames | None,
    push: bool = False,
) -> _Frames | None:
    """Every piece's entries on every square, one pass per *frame*.

    A frame is (player, facing) flattened player-major, so with nothing facing it is the
    player and this runs twice, exactly as it did before frames had a name.

    ``share`` is another spec's frames (`compile_two_readings`). Where its inputs equal
    these, the first player's frames are taken from it, and the second player's are
    compared with it as they are built: ``None`` at the first square that differs, and a
    square that does not differ takes its dedup flag from ``share`` (a function of the
    entries alone, `_needs_runtime_dedup`).
    """
    inputs = (
        pieces, n_facings, width, height, wrap, obstacles, blocks_leaps, radius,
        has_portals, push,
    )
    shared = share if share is not None and share.inputs == inputs else None
    frames = _Frames(inputs, [], [], [], [], [], [], [], [], [])
    for frame in range(n_facings * 2):
        player, facing = divmod(frame, n_facings)
        if shared is not None and player == 0:
            # Copied a level down, so that nothing appended here can reach `share`.
            frames.leaps.append(list(shared.leaps[frame]))
            frames.slides.append(list(shared.slides[frame]))
            frames.hops.append(list(shared.hops[frame]))
            frames.sweeps.append(list(shared.sweeps[frame]))
            frames.dedup.append(list(shared.dedup[frame]))
            frames.leap_dirs.append(list(shared.leap_dirs[frame]))
            frames.slide_dirs.append(list(shared.slide_dirs[frame]))
            frames.hop_dirs.append(list(shared.hop_dirs[frame]))
            frames.push_at.append(list(shared.push_at[frame]))
            continue
        frame_leaps: list[list[tuple[Entry, ...]]] = []
        frame_slides: list[list[tuple[Entry, ...]]] = []
        frame_hops: list[list[tuple[HopEntry, ...]]] = []
        frame_sweeps: list[list[tuple[SweepEntry, ...]]] = []
        frame_dedup: list[frozenset[int]] = []
        frame_dirs: tuple[list[Any], list[Any], list[Any]] = ([], [], [])
        frame_push: list[list[PushSteps]] = []
        for piece_type, piece in enumerate(pieces):
            type_leaps: list[tuple[Entry, ...]] = []
            type_slides: list[tuple[Entry, ...]] = []
            type_hops: list[tuple[HopEntry, ...]] = []
            type_sweeps: list[tuple[SweepEntry, ...]] = []
            type_dirs: tuple[list[Any], list[Any], list[Any]] = ([], [], [])
            type_push: list[PushSteps] = []
            ambiguous: set[int] = set()
            for square in range(n_squares):
                push_at: list[PushSteps] | None = [] if push else None
                built, momentum = _compile_square_full(
                    piece, player, square, width, height, orientation, wrap,
                    obstacles, blocks_leaps,
                    facing_quarter_turns(facing, n_facings),
                    radius,
                    momentum=has_portals,
                    push_out=push_at,
                )
                type_push.append(push_at[0] if push_at else ((), (), ()))
                type_leaps.append(built[0])
                type_slides.append(built[1])
                type_hops.append(built[2])
                type_sweeps.append(built[3])
                for kept, dirs in zip(type_dirs, momentum, strict=True):
                    kept.append(dirs)
                if shared is not None:
                    if (
                        built[0] != shared.leaps[frame][piece_type][square]
                        or built[1] != shared.slides[frame][piece_type][square]
                        or built[2] != shared.hops[frame][piece_type][square]
                        or built[3] != shared.sweeps[frame][piece_type][square]
                    ):
                        return None
                    if square in shared.dedup[frame][piece_type]:
                        ambiguous.add(square)
                elif _needs_runtime_dedup(*built):
                    ambiguous.add(square)
            frame_leaps.append(type_leaps)
            frame_slides.append(type_slides)
            frame_hops.append(type_hops)
            frame_sweeps.append(type_sweeps)
            frame_dedup.append(frozenset(ambiguous))
            for kept_frame, kept_type in zip(frame_dirs, type_dirs, strict=True):
                kept_frame.append(kept_type)
            frame_push.append(type_push)
        frames.leaps.append(frame_leaps)
        frames.slides.append(frame_slides)
        frames.hops.append(frame_hops)
        frames.sweeps.append(frame_sweeps)
        frames.dedup.append(frame_dedup)
        frames.leap_dirs.append(frame_dirs[0])
        frames.slide_dirs.append(frame_dirs[1])
        frames.hop_dirs.append(frame_dirs[2])
        frames.push_at.append(frame_push)
    return frames


def _compile(
    spec: dict[str, Any], share: _Frames | None
) -> tuple[CompiledSpec | None, _Frames | None]:
    """`compile_spec`, with its frames, and ``share`` as `_compile_frames` reads it:
    ``(None, None)`` when a shared pass stopped at a frame that differs."""
    validate_or_raise(spec)

    board_spec = spec["board"]
    width = board_spec["width"]
    height = board_spec["height"]
    n_squares = width * height
    rules = spec.get("rules", {})
    # Schema 19: each portal end is an internal piece type after the declared ones, so
    # every table below is built for it exactly as for a declared piece. Spelling 31: the
    # portals' movement types (`schema.portal_movement_types`) come first, among the
    # declared, where every portal game before it declared its stand-in piece: no piece ever
    # stands as one, and its row keeps every portal game's type count, order and tables.
    movement = portal_movement_types(spec)
    declared_pieces = list(spec["pieces"]) + movement
    declared_types = len(declared_pieces)
    pieces = declared_pieces + _portal_end_pieces(spec)
    n_types = len(pieces)
    has_portals = n_types > declared_types

    piece_ids = tuple(p["id"] for p in pieces)
    index_of = {pid: i for i, pid in enumerate(piece_ids)}
    orientation = spec.get("setup", {}).get("orientation", "mirror")
    shape = board_spec.get("shape", "grid")
    radius = _hexagon_radius(board_spec)
    wrap = _compile_wrap(rules.get("wrap"), shape)
    obstacles = _obstacles_with_absent(rules, board_spec, width, height, orientation)
    absent = absent_squares(board_spec)
    blocks_leaps = bool((rules.get("obstacles") or {}).get("blocks_leaps", False))

    # Schema 22: whether a portal group is a fold; every other game compiles as it did.
    has_folds = has_portals and portal_has_folds(rules)

    facing_axes = tuple(
        int((piece.get("facing") or {}).get("axes", 1)) for piece in pieces
    )
    n_facings = max(facing_axes)
    facing_initial = tuple(
        FACING_NAMES.index((piece.get("facing") or {}).get("initial", "forward"))
        // (4 // n_facings)
        for piece in pieces
    )
    # Schema 25: a facing no move turns. Empty unless one says so (see the field).
    facing_turns = (
        tuple(facing_turns_of(piece) for piece in pieces)
        if any(
            isinstance(piece.get("facing"), dict) and not facing_turns_of(piece)
            for piece in pieces
        )
        else ()
    )
    n_frames = n_facings * 2
    # Q1074b1: whether some move pushes; every other game compiles exactly as it did.
    has_push = push_rule(spec) is not None
    frames = _compile_frames(
        _without_self_destroying_captures(pieces, rules, index_of),
        n_facings, n_squares, width, height, orientation, wrap, obstacles,
        blocks_leaps, radius, has_portals, share, push=has_push,
    )
    if frames is None:
        return None, None
    leaps, slides, hops, sweeps = frames.leaps, frames.slides, frames.hops, frames.sweeps
    dedup = frames.dedup
    leap_dirs, slide_dirs, hop_dirs = frames.leap_dirs, frames.slide_dirs, frames.hop_dirs

    duck = _compile_duck(rules, index_of, width, height, obstacles)
    capture_allowed = _clear_duck_captures(
        _apply_portal_immunity(
            _apply_npc_immunity(
                _build_capture_matrix(pieces, index_of), rules, piece_ids, index_of
            ),
            declared_types,
            {index_of[piece["id"]] for piece in movement},
        ),
        duck,
    )
    npcs = _compile_npcs(rules, pieces, index_of, width, height, orientation)
    # A third owner's block for wild pieces (schema 9), for portals (schema 19) or for the
    # duck (generation 42), which are nobody's too; the validator refuses portals beside
    # either.
    # TC3: and for pieces nobody owns from the setup (refused beside the other three).
    neutrals = _compile_neutrals(
        spec, pieces, width, height, orientation, wrap, obstacles, radius,
    )
    n_owners = 3 if has_third_owner(spec) else 2
    promotions = _settle_on_leaving(
        _compile_promotions(rules.get("promotion"), index_of, width, height, orientation),
        leaps, slides, hops, sweeps, facing_axes, n_facings,
    )
    board, unmoved = _build_initial_position(
        spec, index_of, width, height, n_types, n_facings, facing_initial, npcs,
        _portal_starts(spec, declared_types, width, height),
        duck,
    )
    digest = spec_hash(spec)
    simultaneous = rules.get("simultaneous")
    (
        zob_pieces, zob_side, zob_ep, zob_colour, zob_turn, zob_bar, zob_fold, zob_sim, zob_rest,
        zob_respawn, zob_heading, zob_purse, zob_push_bar,
    ) = (
        _build_zobrist(
            digest, n_squares, n_types, width, n_facings, n_owners,
            portal_bar=any(group["no_straight_back"] for group in portal_groups(rules)),
            fold_ends=n_types - declared_types if has_folds else 0,
            simultaneous=isinstance(simultaneous, dict),
            initiative=isinstance(simultaneous, dict) and simultaneous.get("tie") == "initiative",
            rest_max=max(piece_rests(spec).values(), default=0),
            respawn_keys=sum(every for every, _ in npcs.respawn) if npcs is not None else 0,
            heading_squares=tuple(
                square for square in range(n_squares) if npcs is not None and npcs.on_line(square)
            ),
            purse=purse_rule(spec) is not None,
            push_bar=bool((push_rule(spec) or {}).get("no_push_back")),
        )
    )
    royal_types = tuple(i for i, p in enumerate(pieces) if p.get("royal", False))
    win = rules.get("win", "checkmate")
    demotions = tuple(
        index_of.get(piece.get("promoted_from"), i) for i, piece in enumerate(pieces)
    )
    # One slot for one mechanism. The validator refuses the two spellings together, so
    # at most one of these is ever a rule -- and `contacts` lands in `CompiledSpec.strike`
    # rather than beside it because everything downstream, from `rust_supports` to the
    # Fairy refusal, has exactly one question to ask.
    strike_source = rules.get("strike") or rules.get("contacts")
    landing = (
        _compile_landing(
            strike_source["landing"], n_squares, width, height, orientation, wrap,
            obstacles, radius,
        )
        if isinstance(strike_source, dict)
        and strike_source.get("position") == "choose"
        and isinstance(strike_source.get("landing"), dict)
        else ()
    )
    strike = _compile_strike(
        rules.get("strike"), index_of, n_types, landing,
        rebirth=(board, width, n_facings, board_spec.get("lattice", "square") == "hex"),
    )
    if strike is None:
        strike = _compile_contacts(rules.get("contacts"), index_of, n_types, landing)
    drops = _compile_drops(
        rules.get("drops"), index_of, n_types, width, height, orientation, obstacles,
        pieces,
    )
    promotions = _settle_only_lost(promotions, board, n_types, n_facings, drops)
    attack_dirs: list[list[list[int]]] | None = [] if has_portals else None
    attack_kinds: list[list[list[int]]] | None = [] if has_push else None
    attack_sources = _build_attack_sources(
        leaps,
        slides,
        hops,
        sweeps,
        n_squares,
        n_types,
        n_facings,
        npcs=npcs,
        third_block=has_portals or duck is not None or neutrals is not None,
        momentum=(leap_dirs, slide_dirs, hop_dirs) if has_portals else None,
        dirs_out=attack_dirs,
        facing_axes=facing_axes,
        kinds_out=attack_kinds,
        held_takers=neutrals.takes_royal if neutrals is not None else (),
    )
    one_way_types = _one_way_types(
        leaps,
        slides,
        hops,
        facing_axes,
        promotions,
        n_types,
        n_squares,
        n_facings,
        width,
        sweeps,
        facing_turns,
    )
    portals = None
    if has_portals:
        portals = _compile_portals(
            rules, declared_types, n_types, width, height, wrap, obstacles,
            (leap_dirs, slide_dirs, hop_dirs), attack_dirs or [], zob_bar,
            piece_ids[:declared_types],
            tuple(piece["id"] for piece in movement),
        )
        if has_folds:
            portals = replace(portals, folds=FoldRule(
                starts=_fold_starts(rules, width, height), zobrist=zob_fold,
            ))
        # Any type that can go in by a portal can come back to a square it left -- a pawn
        # through a pair that runs down the board -- so no type some end admits is one-way
        # (`docs/archive/plan-portals.md` §2.6). Without this a pawn move would clear the
        # repetition window while a cycle through a portal exists. Schema 21
        # (`docs/archive/plan-portals-2.md` §2.3): a type every filter keeps out stays one-way.
        admitted = 0
        for masks in portals.admits:
            admitted |= masks[0] | masks[1]
        one_way_types = frozenset(t for t in one_way_types if not admitted >> t & 1)

    multi_royal = _royalty_can_multiply(
        royal_types, board, n_types, n_facings, promotions, strike, drops, demotions, win
    )
    compiled = CompiledSpec(
        name=spec["name"],
        hash=digest,
        source=json.dumps(spec, sort_keys=True),
        moveless_types=frozenset(
            index for index, piece in enumerate(pieces) if not piece.get("moves")
        ),
        one_way_types=one_way_types,
        has_sweep=any(
            entries
            for frame_sweeps in sweeps
            for type_sweeps in frame_sweeps
            for entries in type_sweeps
        ),
        width=width,
        height=height,
        lattice=board_spec.get("lattice", "square"),
        n_squares=n_squares,
        n_types=n_types,
        piece_ids=piece_ids,
        piece_names=tuple(p.get("name", p["id"]) for p in pieces),
        piece_values=tuple(float(p.get("value", 1)) for p in pieces),
        royal_types=royal_types,
        leaps=_freeze(leaps),
        slides=_freeze(slides),
        hops=_freeze(hops),
        sweeps=_freeze(sweeps),
        attack_sources=attack_sources,
        capture_allowed=capture_allowed,
        has_capture_matrix=any(
            not allowed for row in capture_allowed for allowed in row
        ),
        dedup_squares=tuple(tuple(types) for types in dedup),
        n_frames=n_frames,
        n_facings=n_facings,
        facing_axes=facing_axes,
        facing_initial=facing_initial,
        facing_turns=facing_turns,
        immobile_squares=tuple(
            tuple(
                frozenset(
                    square
                    for square in range(n_squares)
                    if not leaps[frame][piece_type][square]
                    and not slides[frame][piece_type][square]
                    and not hops[frame][piece_type][square]
                    and not sweeps[frame][piece_type][square]
                )
                for piece_type in range(n_types)
            )
            for frame in range(n_frames)
        ),
        has_hop_attacks=any(
            any(modes[i] & (MODE_CAPTURE | MODE_PUSH_ENEMY) for i in range(len(modes)))
            for frame_hops in hops
            for type_hops in frame_hops
            for square_hops in type_hops
            for _, modes, _, _, _ in square_hops
        ),
        initial_board=board,
        initial_unmoved=unmoved,
        win=win,
        stalemate=rules.get("stalemate", "draw"),
        max_plies=rules.get("max_plies", 400),
        repetition_draw=_compile_repetition_draw(rules.get("repetition_draw", 3)),
        no_progress_draw=rules.get("no_progress_draw", 100),
        en_passant=bool(rules.get("en_passant", False)),
        promotions=promotions,
        demotions=demotions,
        castling=_compile_castling(rules.get("castling"), index_of, width),
        bare_royal=rules.get("bare_royal") or "",
        check_limit=int(rules.get("check_limit", 0)),
        reach=_compile_reach(rules.get("reach"), index_of, width, height),
        extinction=_compile_extinction(
            rules.get("extinction"), index_of, board, n_types, n_facings
        ),
        scoring=_compile_scoring(rules.get("scoring"), width, height, orientation),
        territory=_compile_territory(rules.get("territory")),
        # Generation 42 (N4b): a duck game's turn is two moves, the piece's and the duck's
        # (the validator refuses `rules.turn` beside it).
        turn=TurnRule(moves=2, lengths=()) if duck is not None else _compile_turn(rules.get("turn")),
        wrap=wrap,
        obstacles=obstacles,
        absent=absent,
        copies=_compile_copies(shape, radius, wrap),
        grid_copies=_compile_grid_copies(shape, width, height, wrap),
        strike=strike,
        npcs=npcs,
        duck=duck,
        n_owners=n_owners,
        entry_promotion=_compile_entry_promotion(
            bool(rules.get("promote_on_entry", False)), promotions, n_types, width, height
        ),
        drops=drops,
        explosion=_compile_explosion(
            rules.get("explode_on_capture"), index_of, width, height, wrap, obstacles, radius
        ),
        must_capture=bool(rules.get("must_capture", False)),
        forbid_immobility=bool(rules.get("forbid_immobility", False)),
        royal_facing_forbidden=rules.get("royal_facing") == "forbidden",
        forbid_check=bool(rules.get("forbid_check", False)),
        checkmate=rules.get("checkmate", "loss"),
        orientation=orientation,
        multi_royal=multi_royal,
        ep_passes=_ep_passes(
            bool(rules.get("en_passant", False)),
            promotions,
            slides,
            n_squares,
            n_frames,
        ),
        facing_lines=_facing_lines(
            rules.get("royal_facing") == "forbidden", width, height, obstacles, wrap, radius
        ),
        **_goal_distances(
            _compile_reach(rules.get("reach"), index_of, width, height),
            leaps,
            slides,
            n_types,
            n_squares,
            n_facings,
            sweeps,
            royal_types,
        ),
        zobrist_pieces=zob_pieces,
        zobrist_side=zob_side,
        zobrist_ep=zob_ep,
        zobrist_colour=zob_colour,
        zobrist_turn=zob_turn,
        portals=portals,
        control=_compile_control(n_types, n_owners, npcs, duck, portals, neutrals),
        neutrals=neutrals,
        simultaneous=_compile_simultaneous(rules, declared_pieces, index_of, n_types, zob_sim),
        spy=_compile_spy(declared_pieces, index_of, board, n_types, n_facings),
        rest=_compile_rests(declared_pieces, index_of, n_types),
        zobrist_rest=zob_rest,
        zobrist_respawn=zob_respawn,
        zobrist_heading=zob_heading,
        bounty=_compile_bounties(declared_pieces, index_of, n_types),
        purse=_compile_purse(
            spec, pieces, index_of, zob_purse,
            width, height, orientation, wrap, obstacles, radius,
        ),
        sight=_compile_sights(
            spec, pieces, index_of, n_types, n_facings, facing_axes, n_squares, width, height,
            orientation, wrap, obstacles, blocks_leaps, radius, royal_types, multi_royal,
        ),
        pounce=_compile_pounce(
            rules, index_of, tuple(float(p.get("value", 1)) for p in pieces), attack_sources,
            width, height, orientation, board_spec.get("lattice", "square") == "hex",
        ),
        insufficient_material=_compile_insufficient_material(
            spec, index_of, n_types, n_squares, slides
        ),
        push=_compile_push(
            spec, frames, pieces, index_of, n_types, n_squares, width,
            height, orientation, wrap, obstacles, absent, radius, attack_kinds, zob_push_bar,
        ) if has_push else None,
    )
    if obstacles:
        _reject_walled_in(compiled)
    if compiled.has_facing:
        _reject_null_rotations(compiled)
    return compiled, frames


def _reject_null_rotations(compiled: CompiledSpec) -> None:
    """Refuse a facing whose orientations are not actually different.

    **A rotation that changes no move is a pass**, and a pass is not a small thing to
    hand a game by accident: zugzwang stops existing, a losing side can always stand
    still, and the repetition rule becomes the only thing that ends anything. A rook
    reads the same rays whichever way it is turned, so ``facing`` on one would generate
    three legal moves per position that do nothing at all.

    Checked against the *compiled* tables rather than against the declared vectors, so
    it is exact: a region, a wall, a rank condition and a seam all feed into what a
    frame actually contains, and two orientations that a symmetry argument says differ
    can still compile to the same table on a particular board.

    Only the first player's frames are compared. The second player's are the same
    descriptors under one further reflection, which cannot make two equal tables unequal.
    """
    for piece_type in range(compiled.n_types):
        axes = compiled.facing_axes[piece_type]
        if axes < 2:
            continue
        step = compiled.facing_step(piece_type)
        seen: dict[Any, int] = {}
        for turn in range(axes):
            facing = turn * step
            frame = frame_index(0, facing, compiled.n_facings)
            table = (
                compiled.leaps[frame][piece_type],
                compiled.slides[frame][piece_type],
                compiled.hops[frame][piece_type],
                compiled.sweeps[frame][piece_type],
            )
            if table in seen:
                first = FACING_NAMES[facing_quarter_turns(seen[table], compiled.n_facings)]
                second = FACING_NAMES[facing_quarter_turns(facing, compiled.n_facings)]
                name = compiled.piece_ids[piece_type]
                if not compiled.turns(piece_type):
                    # Schema 25: no pass to hand out, but a facing that changes no move
                    # is the same game under a second hash, and a sentence a player is
                    # told for nothing.
                    raise SpecError(
                        [
                            f"piece {name!r} facing {second} moves exactly as it does "
                            f"facing {first}, so its facing changes nothing. Give the "
                            "facing to a piece whose moves are not symmetric under it."
                        ]
                    )
                raise SpecError(
                    [
                        f"piece {name!r} facing {second} moves "
                        f"exactly as it does facing {first}, so turning between them "
                        "would be a move that does nothing -- which is a pass, and a "
                        "pass changes every game it is added to. Give the facing to a "
                        "piece whose moves are not symmetric under it."
                    ]
                )
            seen[table] = facing


def _reject_walled_in(compiled: CompiledSpec) -> None:
    """Refuse a wall set that leaves a player with nothing that can move at all.

    This is the degenerate case a proposal will find, and it is cheap here because
    ``immobile_squares`` already answers it: it is the set of squares a piece has no
    table entry from, which is exactly what a piece bricked up behind walls has. Note
    what this is *not* -- it is not "has a legal move", which is a question about a
    position and therefore the engine's to answer, not the compiler's. A piece with
    entries that its own army happens to block is a normal chess opening; a piece with
    no entries at all is a spec mistake.
    """
    for player in (0, 1):
        mobile = False
        for square in range(compiled.n_squares):
            value = compiled.initial_board[square]
            if value == 0 or compiled.owner_of(value) != player:
                continue
            frame = compiled.frame_at(value)
            if square not in compiled.immobile_squares[frame][compiled.type_of(value)]:
                mobile = True
                break
        if not mobile:
            raise SpecError(
                [
                    f"rules.obstacles walls player {player} in: no piece of theirs has "
                    "a move from where it starts"
                ]
            )


# ------------------------------------------------------------------ move tables


def _region_squares(
    names: Any, player: int, width: int, height: int, orientation: str
) -> frozenset[int] | None:
    """Resolve a named square set for one player, reflecting it for the second.

    ``None`` means "no restriction", which is not the same as the empty set: a piece
    with an empty region could never move at all, and that is a spec mistake rather
    than a default.
    """
    if not names:
        return None
    squares = set()
    for name in names:
        coords = parse_square(name, width, height)
        assert coords is not None  # guaranteed by validation
        file, rank = coords
        if player == 1:
            rank = height - 1 - rank
            if orientation == "rotate":
                file = width - 1 - file
        squares.add(rank * width + file)
    return frozenset(squares)


def _lame_path(dx: int, dy: int) -> tuple[int, int]:
    """The square a blockable leap is stopped at.

    One step along the dominant axis, or the midpoint when the two are equal. That
    single rule gives the xiangqi horse its leg -- a (1,2) move blocked at (0,1) -- and
    the elephant its eye, a (2,2) move blocked at (1,1), which is how both are actually
    described.

    **The midpoint is halved toward the mover, so a one-square diagonal step passes over
    nothing**: ``(±1, ±1)`` halves to ``(0, 0)``, its own square, on all four diagonals
    alike, as a one-square orthogonal step's "leg" is its own target. Q880: the halving
    used to floor, so ``(-1, 1)`` went over ``(-1, 0)`` and ``(1, -1)`` over ``(0, -1)``
    while their mirror images went over nothing, and a wall beside a king or a pawn under
    ``blocks_leaps`` stopped its diagonal step on one side only (Quarry's d5 king could
    not reach c6 for c5). Every even midpoint, the elephant's included, is unmoved.
    """
    if abs(dx) == abs(dy):
        return _halved_toward_zero(dx), _halved_toward_zero(dy)
    if abs(dx) > abs(dy):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dy > 0 else -1)


def _lame_ride(
    landing: Callable[[int, int], int | None],
    file: int,
    rank: int,
    square: int,
    step_x: int,
    step_y: int,
    width: int,
    mode: int,
    min_range: int,
    jumps: int,
    confine: frozenset[int] | None,
    obstacles: frozenset[int],
    legs: Legs | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """One direction of a ``lame`` descriptor: the ray ``(leg, landing, leg, landing, ...)``.

    ``legs`` (SR1a) are the jump's stated legs in the mover's frame, each one a modeless
    square before its landing, so a ride of stated legs is ``(leg, leg, landing, ...)``;
    ``None`` is the one computed leg, ``_lame_path``'s, and builds the very table it
    always did.

    Jump after jump along the step, each one blocked at its leg (``_lame_path``, the
    language's one notion of "the way past") and each landing ending the ride when
    occupied, captured if it may be (Q1007a, `docs/archive/plan-schema-31.md` §2.2). Expressed as a
    slide whose legs carry no mode: the generator already walks off a ray the moment it
    meets an occupied square it cannot use, which is exactly what being blocked means, and
    the attack inversion gives each landing every square before it, legs included.

    ``jumps`` is the descriptor's ``range`` (1, today's single lame leap, unless it says
    otherwise; an unlimited one arrives as the slide's cap). A landing nearer than
    ``min_range`` or outside ``confine`` carries mode 0, so it still blocks and the ride
    goes on past it, as a slide's does; trailing landings with no mode are left off, so a
    ride with no landing it may use is ``None``. ``range: 1`` builds the two-square ray the
    single lame leap always had, and drops it in exactly the cases it was dropped:

    * a landing off the board, back on a square the ride has already been on (a wrapped
      board; the origin included, so a wrapped leap onto its own square), or on a wall
      ends the ride;
    * so does a leg on a wall (blocked for ever: a lame leap is *defined* by being stoppable
      at its leg, whatever ``blocks_leaps`` says), off the board (the landing wrapped and
      the leg did not reach the seam, a leap nobody can make), on the landing or on a
      square already ridden (a one-square step has no leg: ``_lame_path`` puts it on the
      target, or on the jump's own start for a diagonal one; the validator refuses it).
    """
    offsets = legs if legs is not None else (_lame_path(step_x, step_y),)
    path: list[int] = []
    modes: list[int] = []
    visited = {square}
    at_file, at_rank = file, rank
    for jump in range(1, jumps + 1):
        target = landing(at_file + step_x, at_rank + step_y)
        if target is None or target in visited or target in obstacles:
            break
        blocks: list[int] = []
        for block_x, block_y in offsets:
            block = landing(at_file + block_x, at_rank + block_y)
            if (
                block is None or block in obstacles or block == target or block in visited
                or block in blocks
            ):
                break
            blocks.append(block)
        else:
            visited.update((*blocks, target))
            path.extend((*blocks, target))
            usable = jump >= min_range and (confine is None or target in confine)
            modes.extend((*(0 for _ in blocks), mode if usable else 0))
            at_file, at_rank = target % width, target // width
            continue
        break
    while modes and not modes[-1]:
        modes.pop()
        path.pop()
    if not modes:
        return None
    return tuple(path), tuple(modes)


def _turning_rides(
    landing: Callable[[int, int], int | None],
    file: int,
    rank: int,
    square: int,
    steps: list[tuple[int, int]],
    path: str,
    width: int,
    height: int,
    mode: int,
    max_range: int,
    min_range: int,
    confine: frozenset[int] | None,
    obstacles: frozenset[int],
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Every path of one turning slide from one square (T1, `docs/archive/plan-pieces-moves.md` §5.2):
    ``(squares, modes, steps)`` per path, ``steps`` the index into ``steps`` of the direction
    that reached each square (for the momentum signs).

    ``steps`` are the descriptor's directions in the mover's frame, laid out round the origin
    by `schema.turning_order`, on either lattice (a hex lattice's in axial coordinates, SR1c). A ``circle`` starts with any of them and takes the next one
    round at every step, one way or the other; a ``zigzag`` alternates its first direction
    with a neighbour less than half a turn away, either one. Every landing goes through
    ``landing`` (the edge, a seam), so a path ends off the board, on a wall, or on a square it
    has already been on, its origin included, exactly as a slide's ray does; a landing
    nearer than ``min_range`` or outside ``confine`` carries mode 0 and is passed, as on a
    slide, and trailing landings with no mode are dropped. ``range`` (0 for no cap) counts
    steps. Each path is a slide entry of its own, so blocking is the core's walk of a ray:
    the first occupied landing ends it, taken if it may be. Paths that share a square are
    folded by the runtime dedup flag (`_needs_runtime_dedup`). Identical paths are kept
    once and the list is sorted, so the table is a function of the spec alone.
    """
    count = len(steps)
    # The longest a path can be: it never revisits a square.
    limit = max_range if max_range > 0 else width * height
    walks: list[list[int]] = []
    for start in range(count):
        ahead, behind = (start + 1) % count, (start - 1) % count
        if path == "circle":
            walks += [[(start + k) % count for k in range(limit)],
                      [(start - k) % count for k in range(limit)]]
            continue
        # Counter-clockwise to the one ahead, or clockwise to the one behind, each only
        # where that turn is less than half a turn (with two directions they are one).
        for neighbour, first, second in ((ahead, start, ahead), (behind, behind, start)):
            if neighbour != start and turns_less_than_half(steps[first], steps[second]):
                walks.append([start if k % 2 == 0 else neighbour for k in range(limit)])
    rides: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()
    for walk in walks:
        squares: list[int] = []
        modes: list[int] = []
        used: list[int] = []
        visited = {square}
        at_file, at_rank = file, rank
        for distance, index in enumerate(walk, start=1):
            step_x, step_y = steps[index]
            target = landing(at_file + step_x, at_rank + step_y)
            if target is None or target in visited or target in obstacles:
                break
            visited.add(target)
            squares.append(target)
            used.append(index)
            usable = distance >= min_range and (confine is None or target in confine)
            modes.append(mode if usable else 0)
            at_file, at_rank = target % width, target // width
        while modes and not modes[-1]:
            modes.pop()
            squares.pop()
            used.pop()
        if modes:
            rides.add((tuple(squares), tuple(modes), tuple(used)))
    return sorted(rides)


def _halved_toward_zero(value: int) -> int:
    """Half of ``value``, rounded toward zero, so ``-v`` halves to minus what ``v`` does."""
    return value // 2 if value >= 0 else -(-value // 2)


def _descriptor_directions(
    vectors: Sequence[Sequence[int]], symmetry: str, facing: int
) -> set[tuple[int, int]]:
    """Every direction one move descriptor implies, in the first player's orientation.

    A set, so a descriptor naming the same direction twice states it once. It is a set
    and not a sequence because nothing here has an order worth keeping -- the caller
    imposes one, and `_compile_square` says why it has to.
    """
    directions: set[tuple[int, int]] = set()
    for vector in vectors:
        directions |= expand_symmetry((vector[0], vector[1]), symmetry)
    if not facing:
        return directions

    # A rotationally symmetric descriptor is its own rotation, so a rook facing any way
    # is a rook and the set comes back unchanged -- which is what makes a facing worth
    # giving only to a piece that is directional.
    return {_rotate(dx, dy, facing) for dx, dy in directions}


def _descriptor_legs(
    descriptor: dict[str, Any], symmetry: str, facing: int, player: int, orientation: str
) -> dict[tuple[int, int], Legs]:
    """A descriptor's stated legs (SR1a), keyed as `_descriptor_directions` keys a
    direction and turned into the mover's frame; empty when it states none.

    Each leg goes through exactly what its vector goes through: the symmetry's map that
    turned the vector (`legs.stated_legs`), the facing, then the second player's
    reflection, so a leg means the same relative square to both armies and every facing.
    """
    if "legs" not in descriptor:
        return {}
    found, _ = stated_legs(descriptor["vectors"], descriptor["legs"], symmetry)

    def frame(x: int, y: int) -> tuple[int, int]:
        if facing:
            x, y = _rotate(x, y, facing)
        if player == 0:
            return x, y
        return (-x, -y) if orientation == "rotate" else (x, -y)

    return {
        (_rotate(dx, dy, facing) if facing else (dx, dy)): tuple(
            frame(x, y) for x, y in legs
        )
        for (dx, dy), legs in found.items()
    }


_Built = tuple[tuple[Entry, ...], tuple[Entry, ...], tuple[HopEntry, ...], tuple[SweepEntry, ...]]
# Schema 19: the momentum masks of one square's leaps, slides and hops, one tuple of
# masks per entry, parallel to the entry's landings (`momentum_bits`).
_Momentum = tuple[tuple[int, ...], ...]
_NO_MOMENTUM: tuple[_Momentum, _Momentum, _Momentum] = ((), (), ())
# Q1074b1: one square's push steps, parallel to its leaps, slides and hops: per entry, per
# landing, the step a pushed piece takes, or ``None`` where the landing pushes nothing.
PushSteps = tuple[
    tuple[tuple[tuple[int, int] | None, ...], ...],
    tuple[tuple[tuple[int, int] | None, ...], ...],
    tuple[tuple[tuple[int, int] | None, ...], ...],
]


def _compile_square(
    piece: dict[str, Any],
    player: int,
    square: int,
    width: int,
    height: int,
    orientation: str = "mirror",
    wrap: WrapRule | None = None,
    obstacles: frozenset[int] = frozenset(),
    blocks_leaps: bool = False,
    facing: int = 0,
    radius: int = 0,
) -> _Built:
    """Build the four entry lists for one piece type standing on one square.

    `_compile_square_full` without the momentum masks, which only a portal game asks for.
    """
    return _compile_square_full(
        piece, player, square, width, height, orientation, wrap, obstacles, blocks_leaps,
        facing, radius,
    )[0]


def _compile_square_full(
    piece: dict[str, Any],
    player: int,
    square: int,
    width: int,
    height: int,
    orientation: str = "mirror",
    wrap: WrapRule | None = None,
    obstacles: frozenset[int] = frozenset(),
    blocks_leaps: bool = False,
    facing: int = 0,
    radius: int = 0,
    momentum: bool = False,
    push_out: list[PushSteps] | None = None,
) -> tuple[_Built, tuple[_Momentum, _Momentum, _Momentum]]:
    """Build the four entry lists for one piece type standing on one square.

    ``momentum`` (schema 19, a game with portals) also returns, for every landing of every
    leap, slide and hop, the sign of the vector that reached it under each mode
    (`momentum_bits`): the direction a piece comes out of a portal in. It then leaves the
    overlap resolution out (`_resolve_overlaps`), so every landing keeps the direction it
    was reached by and the runtime uniqueness pass folds the duplicates instead: on a seam
    a leap and a slide can reach one square from opposite sides.

    ``push_out`` (Q1074b1, a push game) receives the step a pushed piece takes from every
    landing (`PushSteps`, parallel to the leaps, slides and hops returned), in the board's
    coordinates (§3.1): a ray's own step (D1, D5), a leap's or a lame's vector over the gcd of
    its components (D2, D4), or its rank or file component (D3), read in the mover's frame
    (D8). The entries' modes gain `MODE_PUSH_ENEMY`/`MODE_PUSH_OWN` (`_push_mode`). As under
    ``momentum``, the overlap resolution is left out, so every landing keeps the way it was
    reached and the runtime uniqueness pass folds the duplicates: on a seam one square can be
    pushed two ways (a cylinder's rook reaches its rank both ways round). Within one merged
    entry (the leaps of one kind, the descriptors of one ray) a landing has one step; two
    descriptors that disagree there are refused. ``None`` changes nothing.

    ``radius`` is a hexagon's R (Q281a2, unreachable until the gate); 0 is every board
    before it and every rectangle after, and it is what selects the grid or hex reading
    of ``landing`` below -- a spec is one shape or the other, never both, so nothing here
    branches per square.

    ``obstacles`` is the schema-6 counterpart to ``wrap`` and works the same way: it is
    entirely static, so a wall is answered here once and never again. A ray ends at one,
    a leap onto one is never emitted, and under ``blocks_leaps`` a leap *over* one is
    not either. Nothing downstream -- not the inverted attack sources, not the core --
    ever learns that walls exist.

    ``facing`` is which way the piece is pointing, **in quarter turns clockwise**, and
    it is what schema 8 added. It rotates the piece's declared vectors and does nothing
    else: a facing is a different *reading* of the same descriptor set, not a new kind of
    descriptor, which is why it costs another frame rather than another dimension and why
    every ray, region, condition and wall test below is untouched by it.

    The rotation is applied to the vector as the spec wrote it, *before* the second
    player's reflection, so that a facing means the same relative thing to both armies --
    exactly as a base vector does. Applied afterwards, the mirror would make the two
    players' "turn right" the same absolute direction, which is the kind of secret
    asymmetry ``wrap``'s absolute edge names exist to prevent.

    ``wrap`` is the only thing here that can make a step land somewhere the raw
    arithmetic did not, and this is deliberately the only place in the project that
    works out *where a step lands* at all: everything downstream -- the inverted attack
    sources, the goal-distance search, the runtime dedup flag, and the whole Rust core
    -- reads the tables this builds and never asks what a file is. Other modules know
    the rule exists -- the validator checks its spelling, the prose states it, the grid
    files a game by it -- and none of them does geometry.
    """
    file = square % width
    rank = square // width
    relative_rank = rank if player == 0 else height - 1 - rank
    low_file, high_file = wrap.files[player] if wrap else (False, False)
    low_rank, high_rank = wrap.ranks[player] if wrap else (False, False)
    permitted = (lambda side: wrap.may_leave(side, player)) if (radius and wrap) else None

    def landing(at_file: int, at_rank: int) -> int | None:
        """The square an offset lands on, or ``None`` if it left the board."""
        return _land(
            at_file, at_rank, width, height, low_file, high_file, low_rank, high_rank,
            radius, permitted,
        )

    # A confined piece -- a xiangqi general in its palace, an elephant on its own bank
    # of the river -- simply has no table entry for a square it may not reach, so the
    # restriction costs nothing once compiled.
    confine = _region_squares(piece.get("region"), player, width, height, orientation)
    if confine is not None and square not in confine:
        return ((), (), (), ()), _NO_MOMENTUM

    # Nothing stands on a wall, so a piece that starts on one has no moves from it.
    # Validation already refuses a setup that puts one there; this is what makes the
    # table honest for a square no piece can ever reach either.
    if square in obstacles:
        return ((), (), (), ()), _NO_MOMENTUM

    # Leaps merge into a target -> mode map; slides and hops merge per direction.
    leap_targets: dict[bool, dict[int, int]] = {}
    lame_entries: list[Entry] = []
    # Schema 19: the momentum of each leap target and lame leap, built beside them.
    leap_momentum: dict[bool, dict[int, int]] = {}
    lame_momentum: list[tuple[int, ...]] = []
    # T1: each turning slide's paths, and their momentum, built beside the lame rides.
    turning_entries: list[Entry] = []
    turning_momentum: list[tuple[int, ...]] = []
    # (kind, step x, step y, needs an unmoved origin, screen range, screen owner).
    # The last two are hop-only and are zero for everything else, which is what keeps a
    # slide and a hop along the same vector from merging into one ray.
    _RayKey = tuple[str, int, int, bool, int, int]
    ray_groups: dict[_RayKey, dict[int, int]] = {}
    # The squares each ray actually visits, recorded as it is walked rather than
    # re-derived afterwards. Two descriptors sharing a key share an origin and a step,
    # so the shorter walk is a prefix of the longer one and this only ever extends --
    # and with a seam in play there is no arithmetic that could rebuild it.
    ray_squares: dict[_RayKey, list[int]] = {}
    # Schema 22: each ray's mode at every distance a continued walk may reach.
    # Q1074b1: the push steps of each merged entry, beside its modes.
    leap_push: dict[bool, dict[int, tuple[int, int]]] = {}
    lame_push: list[tuple[tuple[int, int] | None, ...]] = []
    ray_push: dict[Any, dict[int, tuple[int, int]]] = {}

    for descriptor in piece["moves"]:
        condition = descriptor.get("condition") or {}
        if "from_rank" in condition and condition["from_rank"] != relative_rank:
            continue  # cannot ever apply here; drop it from the table entirely
        if "region" in condition:
            from_region = _region_squares(
                condition["region"], player, width, height, orientation
            )
            if from_region is not None and square not in from_region:
                continue
        requires_unmoved = bool(condition.get("first_move", False))

        kind = descriptor["kind"]
        mode = _MODE_BITS[descriptor.get("mode", "both")]
        pushed = move_push(descriptor) if push_out is not None else None
        if pushed is not None:
            mode = _push_mode(mode, pushed)
        symmetry = descriptor.get("symmetry", "none")
        min_range = descriptor.get("min_range", 1)
        max_range = descriptor.get("range", DEFAULT_RANGE.get(kind, 1))

        turning = path_of(descriptor)
        if kind == "slide" and turning != "straight":
            # T1: a turning path reads the whole set of directions at once, in the mover's
            # frame (the second player's reflection, as below), so it is built here rather
            # than one direction at a time. On a hex lattice (SR1c) the frame is axial and
            # the order is the hexagon's angle order: `schema.turning_order`'s integer
            # comparator reads axial vectors exactly as their angles in the plane, since the
            # map into the plane keeps every cross product's sign (its comment). The second
            # player's frame there is the half turn, which keeps a turn's sense, and every
            # landing (a hexagon's absent cell, a seam) is `landing`'s, as on a slide.
            frame = [
                (dx, dy) if player == 0 else (-dx, -dy) if orientation == "rotate" else (dx, -dy)
                for dx, dy in _descriptor_directions(descriptor["vectors"], symmetry, facing)
            ]
            steps = turning_order(frame)
            for squares, path_modes, used in _turning_rides(
                landing, file, rank, square, steps, turning, width, height, mode, max_range,
                min_range, confine, obstacles,
            ):
                turning_entries.append((squares, path_modes, requires_unmoved))
                turning_momentum.append(tuple(
                    momentum_bits(*steps[index], m) if m else 0
                    for index, m in zip(used, path_modes, strict=True)
                ))
            continue

        # Sorted, and the sort is what makes the compiled tables a function of the spec
        # alone. A set of tuples iterates in hash order, and CPython's tuple hash is
        # built from `Py_uhash_t` -- 64 bits here, 32 under Emscripten. A lame leap is
        # appended to `lame_entries` as its direction is walked, so an unsorted walk put
        # the interpreter's hash width into the slide table's tail: that permuted
        # xiangqi's `slide_squares` under Pyodide and parted its games from the server's
        # at ply 1. Every other order below is re-imposed on a key (`sorted(ray_groups)`,
        # `sorted(leap_targets)`), so this is the one walk whose order reaches a table.
        frame_legs = _descriptor_legs(descriptor, symmetry, facing, player, orientation)
        for dx, dy in sorted(_descriptor_directions(descriptor["vectors"], symmetry, facing)):
            # Vectors are written from the first player's view. Under the default
            # mirror the second player's forward direction is the rank reflection and
            # files are untouched; under ``rotate`` the whole board turns 180 degrees,
            # which is what legan needs -- its pawns advance along a diagonal, and a
            # rank-only flip would leave the two armies moving in unrelated directions.
            # Either way this matches the reflection used to build the second army.
            if player == 0:
                step_x, step_y = dx, dy
            elif orientation == "rotate":
                step_x, step_y = -dx, -dy
            else:
                step_x, step_y = dx, -dy

            if kind == "leap":
                target = landing(file + step_x, rank + step_y)
                # A wrapped leap can come all the way round onto its own square, which
                # is a null move rather than a move; nothing else can produce one.
                if target is None or target == square:
                    continue
                if confine is not None and target not in confine:
                    continue
                if target in obstacles:
                    continue
                # "A wall a knight may vault" and "a wall a knight may not" are two
                # different games, and neither is obviously the default -- so the spec
                # says which. The square a jump goes *over* is ``_lame_path``: this
                # project already has exactly one rule for that, it is the rule the
                # xiangqi horse is described by, and inventing a second notion of "the
                # way past" would leave two answers to one question.
                if blocks_leaps and any(
                    over is not None and over in obstacles
                    for over in (
                        landing(file + block_x, rank + block_y)
                        for block_x, block_y in frame_legs.get(
                            (dx, dy), (_lame_path(step_x, step_y),)
                        )
                    )
                ):
                    continue
                bucket = leap_targets.setdefault(requires_unmoved, {})
                bucket[target] = bucket.get(target, 0) | mode
                if pushed is not None:
                    _push_landing(
                        leap_push.setdefault(requires_unmoved, {}), target,
                        _push_step(step_x, step_y, pushed, True),
                    )
                if momentum:
                    signs = leap_momentum.setdefault(requires_unmoved, {})
                    signs[target] = signs.get(target, 0) | momentum_bits(step_x, step_y, mode)
                continue

            if kind == "lame":
                ride = _lame_ride(
                    landing, file, rank, square, step_x, step_y, width, mode, min_range,
                    max_range if max_range > 0 else (
                        width * height if wrap and wrap.active_for(player)
                        else max(width, height)
                    ),
                    confine, obstacles, frame_legs.get((dx, dy)),
                )
                if ride is not None:
                    path, path_modes = ride
                    lame_entries.append((path, path_modes, requires_unmoved))
                    lame_step = (
                        _push_step(step_x, step_y, pushed, True) if pushed is not None else None
                    )
                    lame_push.append(tuple(
                        lame_step if at_mode & MODE_PUSH else None for at_mode in path_modes
                    ))
                    lame_momentum.append(
                        tuple(momentum_bits(step_x, step_y, m) if m else 0 for m in path_modes)
                    )
                continue

            # Q1063: a sweep's slot for the screen range holds its sweep length (0 for a
            # sweep that clears its whole line), so two sweeps along one vector that clear
            # different lengths are two rays, and every sweep of one length is one ray.
            key = (
                kind,
                step_x,
                step_y,
                requires_unmoved,
                sweep_length(descriptor) if kind == "sweep"
                else descriptor.get("screen_range", 0),
                _SCREEN_BITS[str(descriptor.get("screen", "any"))],
            )
            modes = ray_groups.setdefault(key, {})
            walked = ray_squares.setdefault(key, [])
            # An unbounded ray runs to the far edge, and on a board with no far edge it
            # runs until it comes back on itself instead. The cap is the square count
            # because a torus diagonal's period is neither the width nor the height.
            limit = max_range if max_range > 0 else (
                width * height if wrap and wrap.active_for(player)
                else max(width, height)
            )
            current_file, current_rank = file, rank
            visited = {square}
            for distance in range(1, limit + 1):
                current = landing(current_file + step_x, current_rank + step_y)
                if current is None:
                    break
                # A ray stops when it reaches a square it has already been on, the
                # origin included. Without a seam this never fires; with one it is what
                # stops a rook circling a cylinder for ever, and it is exact for a
                # diagonal whose period is neither dimension.
                if current in visited:
                    break
                # A wall ends the line, and this is emphatically *not* what ``confine``
                # does two lines down: a square outside a piece's region gets mode 0 and
                # the ray carries on past it, because a palace is somewhere a piece may
                # not *stop*. An obstacle is somewhere nothing may *be*, so there is
                # nothing on the far side of it to see. Getting this backwards lets a
                # rook shoot through a wall, and perft would happily count it. A hop's
                # ray is walked by this same loop and so stops at a wall too, which is
                # the reading a cannon deserves: a wall is not a screen, and there is
                # nothing behind it to land on.
                if current in obstacles:
                    break
                visited.add(current)
                current_file, current_rank = current % width, current // width
                index = distance - 1
                if index < len(walked):
                    assert walked[index] == current  # same origin, same step
                else:
                    walked.append(current)
                if distance >= min_range and (confine is None or current in confine):
                    modes[index] = modes.get(index, 0) | mode
                    if pushed is not None:
                        _push_landing(
                            ray_push.setdefault(key, {}), index,
                            _push_step(step_x, step_y, pushed, False),
                        )

    leap_entries = [
        (tuple(sorted(targets)), tuple(targets[t] for t in sorted(targets)), unmoved_flag)
        for unmoved_flag, targets in sorted(leap_targets.items())
        if targets
    ]
    leap_signs = tuple(
        tuple(leap_momentum.get(unmoved_flag, {}).get(t, 0) for t in sorted(targets))
        for unmoved_flag, targets in sorted(leap_targets.items())
        if targets
    )

    slide_entries: list[Entry] = []
    hop_entries: list[HopEntry] = []
    sweep_entries: list[SweepEntry] = []
    slide_signs: list[tuple[int, ...]] = []
    hop_signs: list[tuple[int, ...]] = []
    slide_push: list[tuple[tuple[int, int] | None, ...]] = []
    hop_push: list[tuple[tuple[int, int] | None, ...]] = []
    for key in sorted(ray_groups):
        kind, step_x, step_y, requires_unmoved, screen_range, screen_owner = key
        ray = ray_squares.get(key, [])
        length = len(ray)
        if length == 0:
            continue
        modes = ray_groups[key]
        entry: Entry = (
            tuple(ray),
            tuple(modes.get(i, 0) for i in range(length)),
            requires_unmoved,
        )
        if not any(entry[1]):
            continue
        ray_signs = tuple(momentum_bits(step_x, step_y, mode) for mode in entry[1])
        ray_steps = tuple(
            ray_push.get(key, {}).get(index) if mode & MODE_PUSH else None
            for index, mode in enumerate(entry[1])
        )
        if kind == "slide":
            slide_entries.append(entry)
            slide_signs.append(ray_signs)
            slide_push.append(ray_steps)
        elif kind == "sweep":
            # Schema 13: a ray the core walks through pieces. A region square keeps mode
            # 0 and stays on the ray, so a sweeper passes over it without stopping.
            # Q1063: with its sweep length, 0 where the ray is no longer than it (the edge
            # cut it short), so a sweep that clears all it can reach has one table.
            sweep_entries.append(
                (*entry, screen_range if screen_range < length else 0)
            )
        else:
            hop_entries.append((*entry, screen_range, screen_owner))
            hop_signs.append(ray_signs)
            hop_push.append(ray_steps)

    if push_out is not None:
        push_out.append((
            tuple(
                tuple(
                    leap_push.get(unmoved_flag, {}).get(t) if targets[t] & MODE_PUSH else None
                    for t in sorted(targets)
                )
                for unmoved_flag, targets in sorted(leap_targets.items())
                if targets
            ),
            tuple(slide_push + lame_push + [(None,) * len(e[0]) for e in turning_entries]),
            tuple(hop_push),
        ))
    if momentum or push_out is not None:
        return (
            (
                tuple(leap_entries), tuple(slide_entries + lame_entries + turning_entries),
                tuple(hop_entries), tuple(sweep_entries),
            ),
            (
                leap_signs, tuple(slide_signs + lame_momentum + turning_momentum),
                tuple(hop_signs),
            ),
        )
    return _resolve_overlaps(
        leap_entries, slide_entries + lame_entries + turning_entries, hop_entries,
        sweep_entries,
    ), _NO_MOMENTUM


def _push_mode(mode: int, pushed: dict[str, Any]) -> int:
    """A move entry's landing mode with its push (§1.3): the push bits ``whose`` names, and
    the capture bit off where it pushes enemies and ``captures`` is false."""
    whose = pushed["whose"]
    if whose in ("enemy", "any"):
        mode |= MODE_PUSH_ENEMY
        if not pushed["captures"]:
            mode &= ~MODE_CAPTURE
    if whose in ("own", "any"):
        mode |= MODE_PUSH_OWN
    return mode


def _push_step(step_x: int, step_y: int, pushed: dict[str, Any], leap: bool) -> tuple[int, int]:
    """The step a pushed piece takes (§3.1): along the move, a ray's own step (D1, D5) or a
    leap's over its components' gcd (D2, D4); along a rank or file, that component's sign (D3).
    Already in the mover's frame (D8)."""
    along = pushed["along"]
    if along == "rank":
        step = ((step_x > 0) - (step_x < 0), 0)
    elif along == "file":
        step = (0, (step_y > 0) - (step_y < 0))
    elif leap:
        divisor = gcd(abs(step_x), abs(step_y)) or 1
        step = (step_x // divisor, step_y // divisor)
    else:
        step = (step_x, step_y)
    if step == (0, 0):
        raise SpecError([
            f"a push along the {along} on the step ({step_x}, {step_y}), which has no "
            f"{along} component (D3)"
        ])
    return step


def _push_landing(steps: dict[int, tuple[int, int]], at: int, step: tuple[int, int]) -> None:
    """Record the step a push at ``at`` (a landing, or a ray's index) takes in one merged
    entry, which holds one step per landing."""
    known = steps.setdefault(at, step)
    if known != step:
        raise SpecError([
            f"two moves of one piece push the piece on one landing two ways ({known} and "
            f"{step}) through one table entry: give the moves one 'along'"
        ])


def _push_next(
    player: int, step: tuple[int, int], width: int, height: int, wrap: WrapRule | None,
    radius: int, obstacles: frozenset[int], absent: frozenset[int], crush: bool = False,
) -> tuple[int, ...]:
    """Where a piece on each square goes when ``player``'s push moves it by ``step``: through
    `_land` with the mover's wrap (D1, H3), `PUSH_OFF` off the board or onto a hexagon's
    absent cell (W2), `PUSH_STOP` onto a wall (W1) or back onto itself; under ``crush``
    (``walls: "crush"``, Q1074b7) a wall is `PUSH_CRUSH`, which removes the piece."""
    low_file, high_file = wrap.files[player] if wrap else (False, False)
    low_rank, high_rank = wrap.ranks[player] if wrap else (False, False)
    permitted = (lambda side: wrap.may_leave(side, player)) if (radius and wrap) else None
    row: list[int] = []
    for square in range(width * height):
        target = _land(
            square % width + step[0], square // width + step[1], width, height,
            low_file, high_file, low_rank, high_rank, radius, permitted,
        )
        if target is None or target in absent:
            row.append(PUSH_OFF)
        elif target in obstacles:
            row.append(PUSH_CRUSH if crush else PUSH_STOP)
        elif target == square:
            row.append(PUSH_STOP)
        else:
            row.append(target)
    return tuple(row)


def _compile_push(
    spec: dict[str, Any],
    frames: _Frames,
    pieces: list[dict[str, Any]],
    index_of: dict[str, int],
    n_types: int,
    n_squares: int,
    width: int,
    height: int,
    orientation: str,
    wrap: WrapRule | None,
    obstacles: frozenset[int],
    absent: frozenset[int],
    radius: int,
    attack_kinds: list[list[list[int]]] | None,
    zobrist_bar: tuple[int, ...] = (),
) -> PushRule:
    """The push rule's tables (`PushRule`), from the frames' push landings."""
    rule = push_rule(spec)
    assert rule is not None
    rules = spec.get("rules", {})
    duck = duck_piece(rules)
    # Nobody's types are never pushed (N1, N2), and neither are portal ends (refused beside a
    # push, W4), which are not declared pieces.
    pushable = {piece["id"] for piece in spec["pieces"]} - {
        npc["piece"] for npc in rules.get("npcs") or []
    } - ({duck} if duck else set())
    n_facings = len(frames.push_at) // 2
    keys = sorted({
        (frame // n_facings, step[0], step[1])
        for frame, frame_push in enumerate(frames.push_at)
        for type_push in frame_push
        for square_push in type_push
        for table_steps in square_push
        for entry_steps in table_steps
        for step in entry_steps
        if step is not None
    })
    number = {key: index for index, key in enumerate(keys)}

    def parallel(which: int) -> Any:
        return tuple(
            tuple(
                tuple(
                    tuple(
                        tuple(
                            PUSH_NO_DIRECTION if step is None
                            else number[(frame // n_facings, step[0], step[1])]
                            for step in entry_steps
                        )
                        for entry_steps in frames.push_at[frame][kind][square][which]
                    )
                    for square in range(n_squares)
                )
                for kind in range(n_types)
            )
            for frame in range(len(frames.push_at))
        )

    lists = push_lists(spec)
    ids = [piece["id"] for piece in pieces]

    def may_push(pusher: str, victim: str) -> bool:
        if pusher not in lists or victim not in pushable:
            return False
        named = lists[pusher]
        return named is None or victim in named

    allowed = tuple(tuple(may_push(pusher, victim) for victim in ids) for pusher in ids)
    weights = piece_weights(spec)
    strengths = piece_strengths(spec)
    regions = [piece.get("region") for piece in pieces]
    stand: tuple[tuple[tuple[bool, ...], ...], ...] = ()
    if any(regions):
        stand = tuple(
            tuple(
                () if region is None else tuple(
                    square in confine for square in range(n_squares)
                )
                for region in regions
                for confine in [
                    _region_squares(region, player, width, height, orientation) or frozenset()
                ]
            )
            for player in range(2)
        )
    return PushRule(
        chain=PUSH_CHAIN[rule["chain"]],
        edge_removes=rule["edge"] == "removes",
        directions=tuple(keys),
        next=tuple(
            _push_next(
                player, (dx, dy), width, height, wrap, radius, obstacles, absent,
                rule["walls"] == "crush",
            )
            for player, dx, dy in keys
        ),
        leap_dirs=parallel(0),
        slide_dirs=parallel(1),
        hop_dirs=parallel(2),
        allowed=allowed,
        # A weight only a pushed piece's load reads: 0 for a type nothing may push, so a
        # pusher's one number spelt as a weight or as a strength is one table (Q1074b4s).
        weight=(
            tuple(
                weights.get(piece["id"], 0) if any(row[kind] for row in allowed) else 0
                for kind, piece in enumerate(pieces)
            )
            if rule["chain"] == "weight" else ()
        ),
        strength=(
            tuple(
                strengths.get(piece["id"], 0) if piece["id"] in lists else 0
                for piece in pieces
            )
            if rule["chain"] == "weight" else ()
        ),
        stand=stand,
        attack_kinds=tuple(
            tuple(tuple(kinds) for kinds in owner) for owner in (attack_kinds or [])
        ),
        zobrist_bar=zobrist_bar,
    )


def _resolve_overlaps(
    leaps: list[Entry], slides: list[Entry], hops: list[HopEntry], sweeps: list[SweepEntry]
) -> tuple[tuple[Entry, ...], tuple[Entry, ...], tuple[HopEntry, ...], tuple[SweepEntry, ...]]:
    """Strip redundant reachability so two primitives cannot emit the same move twice.

    Merging by direction already handles same-kind overlap; what is left is a piece
    that reaches one square by two different primitives. One case is resolvable
    statically: an unconditional leap always fires, and a slide to the same square
    fires only under the strictly narrower condition of a clear path, so the slide's
    bits for that square are pure duplication and can go.

    Hops are left alone. A hop needs exactly one screening piece and a slide needs
    none, so the two never fire from the same position -- statically they look like a
    conflict, but suppressing either would delete a real move.

    Sweeps are left alone too (schema 13). A sweep over a piece and a leap to the same
    square are two different moves; over an empty path the words are equal and the
    runtime dedup folds them.
    """
    always_covered: dict[int, int] = {}
    for targets, modes, requires_unmoved in leaps:
        if requires_unmoved:
            continue
        for index, target in enumerate(targets):
            always_covered[target] = always_covered.get(target, 0) | modes[index]

    if always_covered:
        cleaned: list[Entry] = []
        for ray, modes, requires_unmoved in slides:
            adjusted = tuple(
                mode & ~always_covered.get(ray[index], 0) for index, mode in enumerate(modes)
            )
            if any(adjusted):
                cleaned.append((ray, adjusted, requires_unmoved))
        slides = cleaned

    return tuple(leaps), tuple(slides), tuple(hops), tuple(sweeps)


def _needs_runtime_dedup(
    leaps: tuple[Entry, ...],
    slides: tuple[Entry, ...],
    hops: tuple[HopEntry, ...],
    sweeps: tuple[SweepEntry, ...] = (),
) -> bool:
    """Whether any square is still reachable two ways after static resolution.

    True only for genuinely odd pieces -- a conditional leap shadowing an
    unconditional one, or a leap and a hop converging. Flagging is per piece per
    square, so the generator pays for the oddity exactly where it exists. A sweep that
    shares a target with anything is flagged, because over an empty path its word is a
    plain move's.
    """
    seen: dict[int, int] = {}
    groups = [
        (targets, modes) for targets, modes, _ in leaps
    ] + [
        (ray, modes) for ray, modes, _ in slides
    ] + [
        (ray, modes) for ray, modes, _, _, _ in hops
    ] + [
        (ray, modes) for ray, modes, *_ in sweeps
    ]
    for squares, modes in groups:
        local: dict[int, int] = {}
        for index, square in enumerate(squares):
            local[square] = local.get(square, 0) | modes[index]
        for square, mode in local.items():
            if seen.get(square, 0) & mode:
                return True
            seen[square] = seen.get(square, 0) | mode
    return False


def _freeze(table: list[list[list[Any]]]) -> Any:
    """Deep-freeze a [player][piece type][square] table into nested tuples.

    Deliberately untyped in the middle: leaps and slides hold ``Entry`` and hops hold
    ``HopEntry``, and spelling that out here would mean two identical functions.
    """
    return tuple(tuple(tuple(squares) for squares in types) for types in table)


# The landing modes that attack: a capture, and (Q1074b1, R1) a push of an enemy piece. No
# stored spec's tables hold a push bit, so every one keeps its sources.
_ATTACKS = MODE_CAPTURE | MODE_PUSH_ENEMY


def _build_attack_sources(
    leaps: list[list[list[tuple[Entry, ...]]]],
    slides: list[list[list[tuple[Entry, ...]]]],
    hops: list[list[list[tuple[HopEntry, ...]]]],
    sweeps: list[list[list[tuple[SweepEntry, ...]]]],
    n_squares: int,
    n_types: int,
    n_facings: int = 1,
    npcs: NpcRule | None = None,
    third_block: bool = False,
    momentum: tuple[Any, Any, Any] | None = None,
    dirs_out: list[list[list[int]]] | None = None,
    facing_axes: tuple[int, ...] = (),
    kinds_out: list[list[list[int]]] | None = None,
    held_takers: tuple[int, ...] = (),
) -> tuple[tuple[tuple[AttackSource, ...], ...], ...]:
    """Invert the move tables into "who could attack this square?".

    Legal-move filtering under ``win: checkmate`` asks that question once per candidate
    move, which makes it the single hottest query in the engine. Answering it forward --
    walk every enemy piece, generate its moves, look for the square -- costs a full
    board scan each time. Inverted, it costs a walk over a short precomputed list, most
    of whose entries fail on their first byte comparison.

    Indexed by the *attacking owner* and not by frame, and that is deliberate: a source
    already carries the exact piece value it matches, so a facing-carrying value is
    simply another row and check detection needs no new dimension. The table grows with
    the facing count and nothing that reads it changes -- which is the same property that
    made a wrapped board and a walled one cost the core nothing.

    **Only the frames a type can stand in** (Q869). The movement tables hold every type in
    every frame, so a rook has a "facing backward" reading on a board where only the pawn
    faces; no rook byte ever carries it, and a source for it fails its first comparison on
    every check test. ``facing_axes`` leaves those out: on the rear guard they were half
    the slider sources and a quarter of the perft time, and nothing reads a source whose
    value never stands on a square, so no answer moves.

    **A third owner's block, from the routes rather than from the tables** (schema 9). A
    wild piece threatens exactly the square it will step to next, so its block is one
    source per route square -- origin the square, value the wild byte, no path, no
    screens -- and the core walks it with the same loop it walks a player's. The block
    exists only when the spec has wild pieces, so every earlier ruleset's table is still
    two blocks long and byte-identical.
    """
    table: list[list[list[AttackSource]]] = [
        [[] for _ in range(n_squares)] for _ in range(2)
    ]
    # Schema 19: the capture directions of every source, in lockstep with ``table``
    # (`PortalRule.attack_dirs`). Read only when ``momentum`` is given.
    dirs: list[list[list[int]]] = [[[] for _ in range(n_squares)] for _ in range(2)]
    # Q1074b1: what each source attacks by, in lockstep with ``table`` (`PushRule.attack_kinds`).
    # A push of an enemy attacks its landing as a capture does (R1); read only in a push game.
    kinds: list[list[list[int]]] = [[[] for _ in range(n_squares)] for _ in range(2)]
    leap_dirs, slide_dirs, hop_dirs = momentum if momentum is not None else (None, None, None)

    def add(
        player: int, target: int, source: AttackSource, sign: int = 0, mode: int = MODE_CAPTURE
    ) -> None:
        table[player][target].append(source)
        dirs[player][target].append(sign >> MOMENTUM_CAPTURE_SHIFT)
        kinds[player][target].append(
            (ATTACK_BY_CAPTURE if mode & MODE_CAPTURE else 0)
            | (ATTACK_BY_PUSH if mode & MODE_PUSH_ENEMY else 0)
        )

    if third_block and npcs is None:
        # Schema 19: portals are the third owner's and attack nothing, so its block is
        # there and empty -- the core indexes the attack table by owner.
        table.append([[] for _ in range(n_squares)])
        dirs.append([[] for _ in range(n_squares)])
        kinds.append([[] for _ in range(n_squares)])
    if npcs is not None:
        wild: list[list[AttackSource]] = [[] for _ in range(n_squares)]
        for origin, target in enumerate(npcs.next):
            if target < 0 or target == origin or npcs.on_line(origin):
                continue
            value = (WILD * n_facings) * n_types + npcs.piece[origin] + 1
            wild[target].append((origin, value, (), (), 0, False, SCREEN_ANY, 0, ()))
        table.append(wild)
        dirs.append([[0] * len(sources) for sources in wild])
        kinds.append([[ATTACK_BY_CAPTURE] * len(sources) for sources in wild])

    def readings() -> Any:
        for frame in range(len(leaps)):
            player, facing = divmod(frame, n_facings)
            for piece_type in range(len(leaps[frame])):
                if facing and piece_type < len(facing_axes) and facing % (
                    n_facings // facing_axes[piece_type]
                ):
                    continue  # a facing this type can never be in
                yield player, frame, piece_type, (
                    (player * n_facings + facing) * n_types + piece_type + 1
                )
        # Q1077 (`docs/plan-turncoat.md` §4.7): a held neutral that may take a royal attacks
        # as its holder would move it, in frame 0, filed under the third owner's block with
        # its nobody's byte. The core asks these only for the player who holds the piece.
        for piece_type in held_takers:
            yield WILD, 0, piece_type, (WILD * n_facings) * n_types + piece_type + 1

    for player, frame, piece_type, piece_value in readings():
        for origin in range(n_squares):
            for number, (targets, modes, unmoved) in enumerate(
                leaps[frame][piece_type][origin]
            ):
                for index, target in enumerate(targets):
                    if modes[index] & _ATTACKS:
                        add(
                            player, target,
                            (origin, piece_value, (), (), 0, unmoved, SCREEN_ANY, 0, ()),
                            leap_dirs[frame][piece_type][origin][number][index]
                            if leap_dirs else 0,
                            modes[index],
                        )

            for number, (ray, modes, unmoved) in enumerate(
                slides[frame][piece_type][origin]
            ):
                for index, target in enumerate(ray):
                    if modes[index] & _ATTACKS:
                        add(
                            player, target,
                            (origin, piece_value, ray[:index], (), 0, unmoved, SCREEN_ANY,
                             0, ()),
                            slide_dirs[frame][piece_type][origin][number][index]
                            if slide_dirs else 0,
                            modes[index],
                        )

            for number, entry in enumerate(hops[frame][piece_type][origin]):
                ray, modes, unmoved, screen_range, screen_owner = entry
                for index, target in enumerate(ray):
                    if not modes[index] & _ATTACKS:
                        continue
                    path = ray[:index]
                    sign = (
                        hop_dirs[frame][piece_type][origin][number][index] if hop_dirs else 0
                    )
                    if screen_range and index >= screen_range:
                        # Only the last ``screen_range`` squares may hold the screen,
                        # so everything before them has to be empty.
                        split = index - screen_range
                        add(
                            player, target,
                            (origin, piece_value, path[:split], path[split:], 1,
                             unmoved, screen_owner, 0, ()),
                            sign, modes[index],
                        )
                    else:
                        add(
                            player, target,
                            (origin, piece_value, (), path, 1, unmoved, screen_owner,
                             0, ()),
                            sign, modes[index],
                        )

            # Schema 13, decision A: one source per ray square. The square is attacked
            # when it is a capture landing, or when a landing past it would sweep it,
            # so a sweep over the enemy royal is check. The core walks the rest of the
            # ray with `modes` to find that landing.
            #
            # Q1063: past a sweep's length the ray is a slide's. A square there is
            # attacked only as a capture landing over a clear ride, never by a landing
            # beyond it (that would have to pass over the piece on it), and the core
            # reads the length off the byte's high bits to know where the ride starts.
            for ray, modes, unmoved, swept in sweeps[frame][piece_type][origin]:
                for index, target in enumerate(ray):
                    rides = 0 < swept <= index
                    bits = (SWEEP_LANDS if modes[index] & 2 else 0) | (
                        SWEEP_BEYOND if not rides and any(modes[index + 1:]) else 0
                    )
                    if bits:
                        add(
                            player, target,
                            (origin, piece_value, ray[:index], ray[index:], 0, unmoved,
                             SCREEN_ANY, bits | (swept << SWEEP_LENGTH_SHIFT), modes),
                        )

    if dirs_out is not None and momentum is not None:
        dirs_out.extend(dirs)
    if kinds_out is not None:
        kinds_out.extend(kinds)
    return tuple(tuple(tuple(sources) for sources in player_table) for player_table in table)


# --------------------------------------------------------------- initial position


def _compile_wrap(wrap: Any, shape: str | None = None) -> WrapRule | None:
    """Resolve ``rules.wrap`` into one per-player permission set, through the one
    resolver every reader of it shares (`schema.wrap_stated`).

    ``None`` when nothing is declared or everything says ``none``, which is what keeps
    a board with edges compiling down exactly the instructions it always did -- the
    stepping helper is not merely equivalent on that path, it is skipped.

    ``shape`` selects the keys: a rectangle's two axes, or (Q281a2, unreachable until
    the gate) a hexagon's three pairs, `schema.WRAP_PAIRS`. A hexagon cannot reach this
    until `validate_or_raise` admits ``board.shape`` (Q281a4), but the reading is built
    now so the gate has nothing left to teach the compiler.
    """
    stated = wrap_stated(wrap, shape)
    if shape == "hexagon":

        def _seam(key: str, pair: Any) -> tuple[tuple[bool, bool], tuple[bool, bool]]:
            p0, p1 = stated[key]
            return (
                (p0 in (pair.low, "both"), p0 in (pair.high, "both")),
                (p1 in (pair.low, "both"), p1 in (pair.high, "both")),
            )

        seams = {key: _seam(key, pair) for key, pair in WRAP_PAIRS.items()}
        rule = WrapRule(n_s=seams["n_s"], ne_sw=seams["ne_sw"], nw_se=seams["nw_se"])
        return rule if rule.seams_join else None
    files = tuple(
        (value in ("left", "both"), value in ("right", "both")) for value in stated["files"]
    )
    ranks = tuple(
        (value in ("bottom", "both"), value in ("top", "both")) for value in stated["ranks"]
    )
    rule = WrapRule(files=files, ranks=ranks)  # type: ignore[arg-type]
    return rule if rule.files_join or rule.ranks_join else None


def _compile_copies(shape: str, radius: int, wrap: WrapRule | None) -> tuple[HexCopy, ...]:
    """A hexagon's six copies, with their side names and permissions (Q281a2).

    Empty on a grid. The drawing and the designer (Q281b, Q281c) read this rather than
    working a copy vector out for themselves, `docs/reference/hex-build-plan.md`
    *Q281a2* names as the reason `CompiledSpec.copies` exists at all.
    """
    if shape != "hexagon":
        return ()
    copies = []
    for pair in WRAP_PAIRS.values():
        vector = pair.copy(radius)
        for side, offset in ((pair.high, vector), (pair.low, (-vector[0], -vector[1]))):
            permission = (
                (wrap.may_leave(side, 0), wrap.may_leave(side, 1)) if wrap else (False, False)
            )
            copies.append(HexCopy(side=side, vector=offset, permission=permission))
    return tuple(copies)


def _compile_grid_copies(
    shape: str, width: int, height: int, wrap: WrapRule | None
) -> tuple[HexCopy, ...]:
    """A joined rectangle's copies, with their side names and permissions (Q281c3).

    ``vector`` means what a hexagon copy's does: a cell's place in the copy minus its
    place on the board, which on a rectangle is a whole width or height, since
    `_resolve` wraps an axis by exactly that. A corner copy is past a file edge and a
    rank edge at once, and a player may reach it only when both are theirs to cross.
    Empty on a hexagon and on a board with edges.
    """
    if shape == "hexagon" or wrap is None:
        return ()
    files = [("left", -width, 0), ("right", width, 1)] if wrap.files_join else []
    ranks = [("bottom", -height, 0), ("top", height, 1)] if wrap.ranks_join else []
    copies = [
        HexCopy(side=side, vector=(dx, 0),
                permission=(wrap.files[0][high], wrap.files[1][high]))
        for side, dx, high in files
    ] + [
        HexCopy(side=side, vector=(0, dy),
                permission=(wrap.ranks[0][high], wrap.ranks[1][high]))
        for side, dy, high in ranks
    ]
    for file_side, dx, file_high in files:
        for rank_side, dy, rank_high in ranks:
            copies.append(HexCopy(
                side=f"{file_side}+{rank_side}", vector=(dx, dy),
                permission=tuple(  # type: ignore[arg-type]
                    wrap.files[p][file_high] and wrap.ranks[p][rank_high] for p in (0, 1)
                ),
            ))
    return tuple(copies)


def _ep_passes(
    en_passant: bool,
    promotions: tuple[PromotionRule, ...],
    slides: Any,
    n_squares: int,
    n_frames: int = 2,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """``[frame][origin] -> ((destination, the square stepped over), ...)``.

    Keyed by **frame** rather than by player, which is the one table schema 8 had to
    reindex. A pawn that can turn steps over a different square depending on which way
    it points, and an origin alone cannot say which -- so this would have been the fifth
    entry in the "arithmetic on geometry" list, except that it had already stopped being
    arithmetic and the repair was an index rather than a refusal.

    En passant used to be the one rule the core answered with *arithmetic* rather than
    by reading a table: it halved the move's delta to find the square the pawn crossed.
    That is right on a board with edges and wrong on one without, because a wrapped
    move's raw delta is not its real one -- and it fails silently, since nothing about
    it changes a move count and every perft in the suite would still pass.

    Stated as a table it is also a better definition than the arithmetic was. The square
    a pawn steps over is the first square of the line it moved along, which is what a
    ray already knows; a *leap* passes over nothing at all, so the old code computing a
    midpoint for one was answering a question that has no answer. No committed ruleset
    has an en passant piece that leaps, so the two agree everywhere they are both
    defined.

    **A hop is read here too and it does not belong** (Q121). It has a ray, so this used
    to take its second square exactly as it takes a slide's -- but a hop lands past its
    *screen*, and landing on the second square of the ray means the screen stands on the
    first. So the window a hop opens is always on an occupied square, and every arm of
    the generator that takes in passing requires an empty one. Nothing can vacate it
    either: `make` sets the window only when the move captured nothing, so no strike
    cleared it, and the wild phase closes the window outright when it touches that
    square. So a hop's window is dead by construction rather than by omission, which is
    why the repair is here and not in the generator. `schema.can_double_step` says the
    same of the descriptor, and 32 committed and archived specs declare `en_passant`
    with no hop on the promoting piece, so no compiled table moves.
    """
    empty = tuple(() for _ in range(n_squares))
    if not en_passant or not promotions:
        return tuple(empty for _ in range(n_frames))
    piece_type = promotions[0].piece_type
    table: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for frame in range(n_frames):
        per_square: list[tuple[tuple[int, int], ...]] = []
        for origin in range(n_squares):
            passes: dict[int, int] = {}
            for entry in slides[frame][piece_type][origin]:
                ray, modes = entry[0], entry[1]
                if len(ray) >= 2 and modes[1] & MODE_MOVE:
                    passes.setdefault(ray[1], ray[0])
            per_square.append(tuple(sorted(passes.items())))
        table.append(tuple(per_square))
    return tuple(table)


def _facing_lines(
    forbidden: bool,
    width: int,
    height: int,
    obstacles: frozenset[int],
    wrap: WrapRule | None,
    radius: int = 0,
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]:
    """``[square] -> ((the far end of a file line, the squares strictly between), ...)``.

    The flying general, stated as a fact about the board. ``royals_face`` in the core
    used to walk a file by ``+= width`` asking what stood on each square, which is the
    same mistake en passant's midpoint was in schema 5 and the same one castling still
    is: **arithmetic that is only right on a board whose edges are edges and whose empty
    squares are the only empty squares.** It was refused beside `wrap` for the first and
    beside `obstacles` for the second, and both refusals go away with it.

    A wall closes a line, which is the reading the rule deserves rather than a
    concession: the flying general is about an *open* file, and a wall is precisely what
    closes one. So the ray simply breaks at one, exactly as every other ray in this
    module does, and no square beyond it is reachable.

    Joined ranks give a square **two** entries for the same far end -- one each way round
    -- and the rule is satisfied if either path is clear. That is not a special case
    either; it is what "is there an open line between them" means once a file is a
    circle, and it is the thing the arithmetic could not say at all.

    Built only when the rule is declared. Every other ruleset gets an empty table, so
    nothing else's compiled bytes move and only xiangqi's fingerprint changes.

    ``radius`` is a hexagon's R (Q281a2, unreachable until the gate). The line still
    steps by rank alone, ``(0, 1)`` and ``(0, -1)``, through the one landing function
    (`_land`): on a rectangle that is exactly today's file, and on a hexagon it is
    whichever line that step traces, wrapping through a joined side the same way a move
    would. It reads the board's topology (`WrapRule.seam_open`), not a player's
    permission, for the reason given above.
    """
    if not forbidden:
        return ()

    # A blast, not a move: facing belongs to the *board*, so it reads the topology rather
    # than a player's permission. Schema 5 drew that distinction for `explosion.blast`
    # and it applies here for the same reason -- two royals face each other or they do
    # not, and there is no "whose turn is it" in the question. Only ranks matter: a file
    # is a file however the two file edges are joined.
    join = bool(wrap and wrap.ranks_join)
    seam_open = wrap.seam_open if wrap else (lambda _side: False)
    limit = width * height if (join or (radius and wrap and wrap.seams_join)) else height

    table: list[tuple[tuple[int, tuple[int, ...]], ...]] = []
    for square in range(width * height):
        if square in obstacles:
            table.append(())
            continue
        file, rank = square % width, square // width
        entries: list[tuple[int, tuple[int, ...]]] = []
        for step in (1, -1):
            path: list[int] = []
            visited = {square}
            current_file, current_rank = file, rank
            for _ in range(limit):
                current = _land(
                    current_file, current_rank + step, width, height,
                    False, False, join, join, radius, seam_open,
                )
                if current is None:
                    break
                current_file, current_rank = current % width, current // width
                # The same two stops every ray in this module makes, in the same order.
                if current in visited:
                    break
                if current in obstacles:
                    break
                visited.add(current)
                entries.append((current, tuple(path)))
                path.append(current)
        table.append(tuple(entries))
    return tuple(table)


def _compile_turn(turn: Any) -> TurnRule | None:
    if not turn:
        return None
    moves = int(turn.get("moves", 1))
    ramp = tuple(int(value) for value in turn.get("ramp") or ())
    if moves == 1 and all(value == 1 for value in ramp):
        return None  # a turn of one move is what every ruleset already does
    return TurnRule(moves=moves, lengths=ramp)


def _compile_territory(territory: Any) -> TerritoryRule | None:
    if not territory:
        return None
    return TerritoryRule(
        mode=str(territory.get("mode", "permanent")),
        win_at=int(territory.get("win_at", 0)),
    )


def _compile_scoring(
    scoring: Any, width: int, height: int, orientation: str
) -> ScoringRule | None:
    if not scoring:
        return None
    region = scoring.get("region")
    squares = None
    if region:
        # `_region_squares` answers `None` for "no restriction", which cannot happen
        # here: the validator requires a non-empty list whenever the count is "region".
        first = _region_squares(region, 0, width, height, orientation)
        second = _region_squares(region, 1, width, height, orientation)
        assert first is not None and second is not None
        squares = (first, second)
    return ScoringRule(count=str(scoring.get("count", "material")), squares=squares)


def _build_capture_matrix(
    pieces: list[dict[str, Any]], index_of: dict[str, int]
) -> tuple[tuple[bool, ...], ...]:
    """``[attacker][victim] -> may take``, from each piece's ``captures`` list.

    Absent means everything, which is what every ruleset written before schema 3 meant.
    An empty list means nothing, which is a piece that moves and blocks and never takes.
    The two are different and the distinction is load-bearing, so this reads ``is None``
    rather than truthiness.
    """
    n_types = len(pieces)
    matrix: list[tuple[bool, ...]] = []
    for piece in pieces:
        allowed = piece.get("captures")
        if allowed is None:
            matrix.append((True,) * n_types)
        else:
            permitted = {index_of[pid] for pid in allowed if pid in index_of}
            matrix.append(tuple(victim in permitted for victim in range(n_types)))
    return tuple(matrix)


def _apply_npc_immunity(
    matrix: tuple[tuple[bool, ...], ...],
    rules: dict[str, Any],
    piece_ids: tuple[str, ...],
    index_of: dict[str, int],
) -> tuple[tuple[bool, ...], ...]:
    """Schema 10: take a wild type out of the capture matrix for everyone not named.

    **The whole of Q78 in the engine, and it is one loop over a table that already
    exists.** ``npcs[].immune`` is a different *spelling* of the schema-3 capture matrix,
    not a second mechanism: the three capture tests in ``generate_pseudo_legal`` and
    ``may_capture`` behind ``attacks_square_as`` read this matrix and are untouched, so
    the core needs to know nothing about it. *Only a knight may take a wolf* costs
    ``archive/wolves.json`` five six-element ``captures`` lists and this one key, and
    `tests/test_schema_ten.py` asserts the two compile to byte-identical tables.

    Which attackers lose the permission is ``schema.npc_immune_attackers``' answer, not a
    second reading of the key here -- the same sharing ``npc_routes`` gets, and for the
    same reason.

    Absent ``immune`` clears nothing, so this returns the matrix it was handed for every
    ruleset written before schema 10.
    """
    npcs = rules.get("npcs")
    if not isinstance(npcs, list):
        return matrix
    cleared: dict[int, set[int]] = {}
    for npc in npcs:
        if not isinstance(npc, dict) or npc.get("piece") not in index_of:
            continue
        victim = index_of[npc["piece"]]
        for pid in npc_immune_attackers(rules, piece_ids, npc):
            cleared.setdefault(index_of[pid], set()).add(victim)
    if not cleared:
        return matrix
    return tuple(
        tuple(
            allowed and victim not in cleared.get(attacker, ())
            for victim, allowed in enumerate(row)
        )
        for attacker, row in enumerate(matrix)
    )


def _build_initial_position(
    spec: dict[str, Any],
    index_of: dict[str, int],
    width: int,
    height: int,
    n_types: int,
    n_facings: int = 1,
    facing_initial: tuple[int, ...] = (),
    npcs: NpcRule | None = None,
    portal_starts: tuple[tuple[int, int], ...] = (),
    duck: DuckRule | None = None,
) -> tuple[bytes, int]:
    board = bytearray(width * height)
    # Schema 19: the portal ends, each its own internal type in the third owner's block
    # at facing 0. Placed before the unmoved mask is taken, so an end carries its bit as
    # any piece does: a portal's move may have a `first_move` condition.
    for square, piece_type in portal_starts:
        board[square] = (WILD * n_facings) * n_types + piece_type + 1
    setup = spec["setup"]
    mirror = setup.get("mirror", True)
    rotate = setup.get("orientation", "mirror") == "rotate"

    nobodys: list[tuple[int, int]] = []
    for placement in setup["placements"]:
        piece_type = index_of[placement["piece"]]
        if is_nobodys(placement):
            # TC3: placed after the unmoved mask, as a wild piece is (no first move, no
            # castling right), in the third block at facing 0; mirrored, still nobody's.
            for square_name in placement["squares"]:
                coords = parse_square(square_name, width, height)
                assert coords is not None  # guaranteed by validation
                images = {coords}
                if mirror:
                    images.add(reflect_square(*coords, width, height, "rotate" if rotate else "mirror"))
                nobodys.extend((rank * width + file, piece_type) for file, rank in images)
            continue
        player = placement.get("player", 0)
        # Which way these pieces start pointing: the placement's answer where it gives
        # one, the piece type's otherwise. Relative to each army's own forward, which is
        # what makes a mirrored setup mirror the orientations too -- the facing index
        # rotates the vectors, and those are already written from one player's view.
        facing = facing_initial[piece_type] if facing_initial else 0
        if "facing" in placement and n_facings > 1:
            facing = FACING_NAMES.index(placement["facing"]) // (4 // n_facings)
        for square_name in placement["squares"]:
            coords = parse_square(square_name, width, height)
            assert coords is not None  # guaranteed by validation
            file, rank = coords
            board[rank * width + file] = (
                (player * n_facings + facing) * n_types + piece_type + 1
            )
            if mirror:
                mirrored_rank = height - 1 - rank
                mirrored_file = width - 1 - file if rotate else file
                board[mirrored_rank * width + mirrored_file] = (
                    ((1 - player) * n_facings + facing) * n_types + piece_type + 1
                )

    unmoved = 0
    for square, value in enumerate(board):
        if value:
            unmoved |= 1 << square
    # Wild pieces last, *after* the unmoved mask is taken: a wild square carries no bit,
    # because the bit is a castling right and a first-move permission and a wild piece
    # has neither. Placing them earlier would set a bit that the phase then clears on
    # its first step, and `rights_changed` would call every opening ply irreversible.
    for square, piece_type in nobodys:
        board[square] = (WILD * n_facings) * n_types + piece_type + 1
    if npcs is not None:
        for square, piece_type in npcs.starts:
            board[square] = (WILD * n_facings) * n_types + piece_type + 1
    # Generation 42: the duck, where it starts on the board, carries no bit either.
    if duck is not None and duck.start >= 0:
        board[duck.start] = (WILD * n_facings) * n_types + duck.piece_type + 1
    return bytes(board), unmoved


def _royalty_can_multiply(
    royal_types: tuple[int, ...],
    board: bytes,
    n_types: int,
    n_facings: int,
    promotions: tuple[PromotionRule, ...],
    strike: StrikeRule | None,
    drops: DropRule | None,
    demotions: tuple[int, ...],
    win: str,
) -> bool:
    """Whether either player might ever hold two royal pieces at the same time.

    Read off the *compiled* rules, never the spec: ``promotions`` already holds both
    spellings of ``rules.promotion`` and both of ``to`` / ``to_by_player``, and is what
    ``promote_on_entry`` draws from; ``strike.onto_to`` has already dropped an immune
    source, ``effect: capture`` and a chain that keeps nothing alive. An earlier version
    read the spec a second time, knew the dict form of ``promotion`` only, and compiled a
    promotion *list* ending on a king and a strike chain ending on a king both ``False``
    -- a second interpretation of a rule is a second chance to interpret it differently,
    and the core then guarded one king of two (``docs/game-ideas.md``, row 2).

    A royal enters play after ply zero by exactly these routes:

    * a promotion whose targets include a royal type;
    * a strike or contact whose victim -- or, since schema 10, whose attacker --
      becomes a royal type;
    * a drop from a hand a royal can reach. A captured piece enters the captor's hand as
      its ``demotions`` entry, so a common piece ``promoted_from`` a royal arrives royal
      under any ending, and a royal arrives as itself wherever taking one leaves the
      game running -- which is every ``win`` but ``checkmate``, where a royal cannot be
      taken, and ``royal_capture``, where taking the only one ends the game.

    Two royals at ply zero -- a second royal type, or two of one type placed -- are read
    off the board rather than the spec for the same reason.
    """
    if not royal_types:
        return False
    if len(royal_types) > 1:
        return True

    royal = royal_types[0]
    # A royal in the stock (schema 18's placement phase, Q784) is a royal that enters
    # play by a drop, so it counts beside the ones placed: a side holding one and placing
    # one would hold two. Schema 10 let a stock name a royal beside a placed one and the
    # core then guarded one king of two; the validator now refuses that spelling, and
    # this reads it honestly whatever the validator says.
    stocked = drops.stock[royal] if drops is not None and drops.stock else 0
    if stocked and drops is not None and drops.unlimited and drops.unlimited[royal]:
        return True
    for player in (0, 1):
        placed = sum(
            1
            for value in board
            if value
            and (value - 1) // (n_facings * n_types) == player
            and (value - 1) % n_types == royal
        )
        if placed + stocked > 1:
            return True

    if any(royal in targets for rule in promotions for targets in rule.targets):
        return True
    # Both columns of the contact table (schema 10): a meeting can make a royal out of
    # the attacker as easily as out of the piece it meets, and a `strike` fills the
    # attacker's column with -1 so this is the schema-6 reading unchanged.
    if strike is not None and any(
        royal in table
        for table in (strike.onto_to, strike.mover_to, strike.own_onto_to, strike.own_mover_to)
    ):
        return True
    if drops is not None and drops.from_captures:
        royal_taken_continues = win not in ("checkmate", "royal_capture")
        for piece_type, demoted in enumerate(demotions):
            if demoted == royal and (piece_type != royal or royal_taken_continues):
                return True
    return False


def _compile_promotions(
    promotion: Any, index_of: dict[str, int], width: int, height: int, orientation: str
) -> tuple[PromotionRule, ...]:
    if not promotion:
        return ()
    entries = promotion if isinstance(promotion, list) else [promotion]
    return tuple(
        _compile_promotion(entry, index_of, width, height, orientation) for entry in entries
    )


def _compile_promotion(
    promotion: dict[str, Any],
    index_of: dict[str, int],
    width: int,
    height: int,
    orientation: str,
) -> PromotionRule:
    # The zone and the targets are resolved by `schema.promotion_squares` and
    # `schema.promotion_targets`, which the validator's `promote_on_entry` coherence
    # check reads too, so the two cannot disagree about where a zone is. The second
    # player's zone is the first player's reflected, so a spec states the rule once from
    # one side and the two are guaranteed to stay each other's mirror.
    first, second = promotion_squares(promotion, width, height, orientation)
    for_first, for_second = promotion_targets(promotion)
    return PromotionRule(
        piece_type=index_of[promotion["piece"]],
        squares=(first, second),
        targets=(
            tuple(index_of[t] for t in for_first),
            tuple(index_of[t] for t in for_second),
        ),
        mandatory=bool(promotion.get("mandatory", True)),
        on_leaving=promotion.get("on_leaving", False) is True,
        only_lost=promotion.get("only_lost", False) is True,
    )


def _settle_only_lost(
    promotions: tuple[PromotionRule, ...],
    board: bytes,
    n_types: int,
    n_facings: int,
    drops: DropRule | None = None,
) -> tuple[PromotionRule, ...]:
    """Schema 18's ``only_lost``, given the counts it is read against.

    The opening count of each type for each player, read off the compiled opening board
    and not off the JSON, so it is the army `_build_initial_position` placed, mirror and
    all. A wild piece is nobody's and is not counted. Under the placement phase (Q784)
    the stock is counted too: a side plays it all out before any of its pieces moves, so
    by the time a promotion can be offered every stocked piece is on the board or lost.
    Since SR2 (H1, generation 50) the hand may also fill from captures; a piece in it is
    off the board and so counts as lost until it is dropped back, which is the rule as
    written (the board is what is counted), so the opening count stands.
    """
    if not any(rule.only_lost for rule in promotions):
        return promotions
    counts = [[0] * n_types, [0] * n_types]
    for value in board:
        if not value:
            continue
        owner = (value - 1) // (n_facings * n_types)
        if owner < 2:
            counts[owner][(value - 1) % n_types] += 1
    if drops is not None and drops.placement and drops.stock:
        for piece_type, held in enumerate(drops.stock):
            if not (drops.unlimited and drops.unlimited[piece_type]):
                counts[0][piece_type] += held
                counts[1][piece_type] += held
    opening = (tuple(counts[0]), tuple(counts[1]))
    return tuple(
        replace(rule, opening=opening) if rule.only_lost else rule for rule in promotions
    )


def _settle_on_leaving(
    promotions: tuple[PromotionRule, ...],
    leaps: Any,
    slides: Any,
    hops: Any,
    sweeps: Any,
    facing_axes: tuple[int, ...],
    n_facings: int,
) -> tuple[PromotionRule, ...]:
    """Schema 15's flag, kept only where it can fire.

    ``on_leaving`` adds a promotion to a move from a square of the zone to a square
    outside it, so a piece with no such move has nothing for it to add: a chess pawn's
    zone is the far rank, where it has no move at all. There the flag is dropped, and the
    spec compiles to the game it spells, table for table -- the ``{"effect": "capture"}``
    rule one key over. Everything downstream reads this answer, never the JSON: the
    boundary (`bridge.build_tables`), the one-way graph and Fairy-Stockfish's refusal.
    """
    if not any(rule.on_leaving for rule in promotions):
        return promotions
    return tuple(
        rule
        if not rule.on_leaving
        or _leaves_its_zone(rule, leaps, slides, hops, sweeps, facing_axes, n_facings)
        else replace(rule, on_leaving=False)
        for rule in promotions
    )


def _leaves_its_zone(
    rule: PromotionRule,
    leaps: Any,
    slides: Any,
    hops: Any,
    sweeps: Any,
    facing_axes: tuple[int, ...],
    n_facings: int,
) -> bool:
    """Whether the rule's piece has a move from a square of its zone to one outside it.

    Asked of the compiled tables, for either player and any facing the piece can hold.
    `pieces_that_can_leave_their_zone` asks the same of `_compile_square` directly.
    """
    piece = rule.piece_type
    return any(
        _moves_out_of(
            (
                leaps[frame][piece][origin],
                slides[frame][piece][origin],
                hops[frame][piece][origin],
                sweeps[frame][piece][origin],
            ),
            rule.squares[player],
        )
        for player, frame in _frames_of(piece, facing_axes, n_facings)
        for origin in rule.squares[player]
    )


def _frames_of(
    piece_type: int, facing_axes: tuple[int, ...], n_facings: int
) -> list[tuple[int, int]]:
    """Every ``(player, frame)`` this type can stand in: both players, its own facings."""
    axes = facing_axes[piece_type]
    return [
        (player, frame_index(player, turn * (n_facings // axes), n_facings))
        for player in (0, 1)
        for turn in range(axes)
    ]


def _moves_out_of(built: tuple[Any, ...], zone: frozenset[int]) -> bool:
    """Whether one square's four entry lists reach any square outside ``zone``.

    Over-approximate in the direction that keeps a flag: a ray square counts whether or
    not its mode lets the piece stop there, and a hop's whole ray counts though it needs a
    screen -- `_one_way_types`' two readings, and for the same reason. A kept flag that
    never fires costs nothing but a line in the tables; a dropped one that could would
    be a different game.
    """
    return any(target not in zone for table in built for entry in table for target in entry[0])


def pieces_that_can_leave_their_zone(spec: dict[str, Any]) -> frozenset[str]:
    """The promoting pieces of a valid spec with a move from their zone to outside it.

    Where schema 15's ``on_leaving`` can fire, whether or not the spec says it: the
    question `_settle_on_leaving` asks of the compiled tables, asked here of
    `_compile_square` for the zone squares alone, so a caller choosing where to put the
    flag (`loop.operators._promote_on_leaving`) does not pay for a whole compile.
    """
    rules = spec.get("rules", {})
    board = spec["board"]
    width, height = board["width"], board["height"]
    pieces = spec["pieces"]
    index_of = {piece["id"]: i for i, piece in enumerate(pieces)}
    orientation = spec.get("setup", {}).get("orientation", "mirror")
    shape = board.get("shape", "grid")
    radius = _hexagon_radius(board)
    wrap = _compile_wrap(rules.get("wrap"), shape)
    obstacles = _obstacles_with_absent(rules, board, width, height, orientation)
    blocks_leaps = bool((rules.get("obstacles") or {}).get("blocks_leaps", False))
    facing_axes = tuple(int((piece.get("facing") or {}).get("axes", 1)) for piece in pieces)
    n_facings = max(facing_axes)
    leaving: set[str] = set()
    for rule in _compile_promotions(
        rules.get("promotion"), index_of, width, height, orientation
    ):
        piece = pieces[rule.piece_type]
        if any(
            _moves_out_of(
                _compile_square(
                    piece, player, origin, width, height, orientation, wrap, obstacles,
                    blocks_leaps, facing_quarter_turns(frame % n_facings, n_facings),
                    radius,
                ),
                rule.squares[player],
            )
            for player, frame in _frames_of(rule.piece_type, facing_axes, n_facings)
            for origin in rule.squares[player]
        ):
            leaving.add(piece["id"])
    return frozenset(leaving)


def _compile_entry_promotion(
    wanted: bool,
    promotions: tuple[PromotionRule, ...],
    n_types: int,
    width: int,
    height: int,
) -> EntryPromotionRule | None:
    """Resolve ``promote_on_entry`` into ``becomes[player][type][square]``.

    ``None`` when the flag is off, which is every game before schema 9 and is what keeps
    every earlier ruleset's tables byte-identical -- the boundary carries nothing for a
    rule that is not there. With the flag on, every square of every promoting type's
    zone holds the *first* target that type's rule lists for that player, and every
    other cell holds ``-1``. The validator has already refused a flag no piece could
    ever reach, so a rule that compiles is a rule that can fire.
    """
    if not wanted:
        return None
    n_squares = width * height
    becomes = [[[-1] * n_squares for _ in range(n_types)] for _ in (0, 1)]
    for rule in promotions:
        for player in (0, 1):
            targets = rule.targets[player]
            if not targets:
                continue
            row = becomes[player][rule.piece_type]
            for square in rule.squares[player]:
                row[square] = targets[0]
    return EntryPromotionRule(
        becomes=tuple(tuple(tuple(row) for row in per_type) for per_type in becomes)
    )


def _compile_castling(
    castling: dict[str, Any] | None, index_of: dict[str, int], width: int
) -> CastlingRule | None:
    if not castling:
        return None

    king_files = rook_files = None
    if "king_file" in castling:
        files = castling["king_file"]
        king_files = (
            _file_index(files["queenside"]),
            _file_index(files["kingside"]),
        )
        if "rook_file" in castling:
            rooks = castling["rook_file"]
            rook_files = (_file_index(rooks["queenside"]), _file_index(rooks["kingside"]))
        else:
            # Chess's own convention: the rook lands on the square the king crossed.
            rook_files = (king_files[0] + 1, king_files[1] - 1)

    return CastlingRule(
        king_type=index_of[castling["king"]],
        rook_type=index_of[castling["rook"]],
        king_steps=castling.get("king_steps", 2),
        king_files=king_files,
        rook_files=rook_files,
    )


def _file_index(name: str) -> int:
    return ord(name.lower()) - ord("a")


def _compile_reach(
    reach: dict[str, Any] | None, index_of: dict[str, int], width: int, height: int
) -> ReachRule | None:
    if not reach:
        return None
    reflection = reach_reflection(reach)
    squares = set()
    second = set()
    for name in reach["squares"]:
        coords = parse_square(name, width, height)
        assert coords is not None  # guaranteed by validation
        file, rank = coords
        squares.add(rank * width + file)
        if reflection is not None:
            file, rank = reflect_square(file, rank, width, height, reflection)
        second.add(rank * width + file)

    return ReachRule(
        piece_types=tuple(sorted({index_of[piece] for piece in reach_pieces(reach)})),
        squares=(frozenset(squares), frozenset(second)),
        result=reach.get("result", "win"),
        safe=bool(reach.get("safe", False)),
        leaves=reach.get("on", "arrive") == "leave",
    )


# What ``goal_distance`` holds for a square a piece can never get to. One more than any
# real distance can be, so "unreachable" compares as worse than the worst reachable
# square without being a special case at every read.
UNREACHABLE = 255


def _goal_distances(
    reach: ReachRule | None,
    leaps: Any,
    slides: Any,
    n_types: int,
    n_squares: int,
    n_facings: int = 1,
    sweeps: Any = None,
    royal_types: tuple[int, ...] = (),
) -> dict[str, Any]:
    """How many moves each piece needs, from each square, to reach a winning square.

    A breadth-first search over the compiled tables on an otherwise empty board, run
    *backwards* from the goal squares. Backwards because movement is not symmetric --
    a pawn cannot retreat, and a forward-only piece's distance *to* a rank is not its
    distance *from* it -- so the edges have to be reversed before they are walked.

    Bounded work: ``2 x n_types x n_squares`` entries, one BFS per (player, piece type),
    once per ruleset at compile time. Nothing about it depends on a position, which is
    why it belongs here beside the attack sources rather than beside the piece values:
    it is a fact about the rules, not an agent's opinion.

    Two honest limitations, both stated rather than papered over. **Hops contribute no
    edges**, because a hop needs a screen and the board this walks is empty -- so a
    cannon's distance is measured by its ordinary moves, which is what it has when
    nothing is in the way. And **only quiet moves count**: a piece that can reach a
    square by capture and not otherwise cannot reach it here, which is right for a board
    with nothing on it and wrong for the game, and the search is what closes that gap.

    A third arrived with facing: **a piece that can turn is measured as though turning
    were free.** Every frame's edges are unioned, so the answer is the distance the piece
    would have if orientation cost nothing -- an under-estimate by however many turns the
    journey needs. The alternative is a distance per (square, facing), which the boundary
    would have to carry and the evaluator index, for a term that is a heuristic already
    and is read only by a ruleset combining `reach` with a facing.

    **Schema 18's leaving (Q786) is read as reach, then step off.** Under ``on: "leave"``
    a goal square is worth getting to only if the piece can move off it again, and doing
    so is one more move: the walk starts from the goal squares this piece type has a move
    out of, each at distance one, so a piece standing on one is a move from winning and a
    goal it could never leave is no goal. Every racer pays the same extra move, so the
    race is the arrival race shifted by one wherever a piece can step off, and the term
    still does not see that a piece must *stand* on the goal before the move that wins
    (the coverage table's ``partly``, in ``docs/reference/``).
    """
    if reach is None:
        return {"goal_distance": (), "goal_diameter": 1, "goal_distance_each": ()}

    # Q708 (F1): the same search from each goal square alone, for a race the evaluator
    # prices (`evaluation._reach_is_priceable`'s two exclusions, restated: no `safe`, no
    # royal racer) where a side has more than one goal square. The evaluator then reads
    # the distance to the nearest goal a side can still use, when the others are held by
    # a piece it can never capture. With one goal square a side's goals are all held or
    # none, and the union table above already answers both, so nothing is sent.
    goals = [sorted(square for square in reach.squares[player] if square < n_squares)
             for player in (0, 1)]
    priced = not reach.safe and (
        not reach.piece_types or not any(t in royal_types for t in reach.piece_types)
    )
    each_wanted = priced and max(len(goals[0]), len(goals[1])) > 1
    each: list[tuple[tuple[tuple[int, ...], ...], ...]] = []

    # Reverse adjacency, per (player, piece type): who can move *into* this square.
    distances: list[tuple[tuple[int, ...], ...]] = []
    diameter = 1
    for player in (0, 1):
        per_type: list[tuple[int, ...]] = []
        per_goal: list[list[tuple[int, ...]]] = [[] for _ in goals[player]]
        for piece_type in range(n_types):
            into: list[list[int]] = [[] for _ in range(n_squares)]
            # Schema 18 (Q786): the squares this piece has a quiet move out of, which is
            # what a goal under ``on: "leave"`` must be to count.
            out: list[bool] = [False] * n_squares
            for facing in range(n_facings):
                frame = player * n_facings + facing
                for origin in range(n_squares):
                    # A sweep reaches every ray square whatever stands on it, and an
                    # empty board is all this walks (schema 13).
                    swept = (sweeps[frame][piece_type][origin],) if sweeps else ()
                    for entries in (leaps[frame][piece_type][origin],
                                    slides[frame][piece_type][origin], *swept):
                        for targets, modes, *_ in entries:
                            for index, target in enumerate(targets):
                                if modes[index] & MODE_MOVE:
                                    into[target].append(origin)
                                    out[origin] = True

            sources = goals[player]
            start = 0
            if reach.leaves:
                sources = [goal for goal in sources if out[goal]]
                start = 1
            distance = _walk_back(into, sources, n_squares, start)
            finite = [d for d in distance if d != UNREACHABLE]
            if finite:
                diameter = max(diameter, max(finite))
            per_type.append(tuple(distance))
            if each_wanted:
                for index, goal in enumerate(goals[player]):
                    # A goal this piece cannot step off is no goal under ``leave``: its
                    # table is all unreachable, as a sealed goal's reading would be.
                    own = [goal] if not reach.leaves or out[goal] else []
                    per_goal[index].append(tuple(_walk_back(into, own, n_squares, start)))
        distances.append(tuple(per_type))
        each.append(tuple(tuple(table) for table in per_goal))
    return {
        "goal_distance": tuple(distances),
        "goal_diameter": diameter,
        "goal_distance_each": tuple(each) if each_wanted else (),
    }


def _walk_back(
    into: list[list[int]], sources: list[int], n_squares: int, start: int = 0
) -> list[int]:
    """Moves to the nearest of ``sources``, from every square: a breadth-first search over
    ``into``, the reversed move edges ``_goal_distances`` built. ``start`` is what a source
    itself counts: 0 for arriving, 1 for leaving (one move off the goal, schema 18)."""
    distance = [UNREACHABLE] * n_squares
    frontier = list(sources)
    for square in frontier:
        distance[square] = start
    while frontier:
        nxt: list[int] = []
        for square in frontier:
            step = distance[square] + 1
            for origin in into[square]:
                if distance[origin] > step:
                    distance[origin] = step
                    nxt.append(origin)
        frontier = nxt
    return distance


def _one_way_types(
    leaps: Any,
    slides: Any,
    hops: Any,
    facing_axes: tuple[int, ...],
    promotions: tuple[PromotionRule, ...],
    n_types: int,
    n_squares: int,
    n_facings: int,
    width: int,
    sweeps: Any = None,
    facing_turns: tuple[bool, ...] = (),
) -> frozenset[int]:
    """Which piece types can never return to a square they left (Q113).

    **A piece is one-way when its own move graph is acyclic.** Nodes are
    ``(facing, square)``; edges are every square the compiled tables let it reach from
    each node -- quiet, capturing and hopping alike -- plus every rotation it may make,
    minus the moves that promote it into something else. If no path leads back to a node,
    no move of that piece can be taken back, and that is what makes the move *progress*.
    Both of ``game.rs``'s tests for progress read this flag -- it clears the repetition
    window and it resets the no-progress clock.

    **This replaces two rules that were hiding in code.** The window used to ask
    ``piece_type == ep_type`` and ``ep_type`` is set only when ``en_passant`` is declared
    (Q113); the clock used to ask ``is_promoting(piece_type)``, so declaring a
    ``promotion`` rule made a piece's every move count, including a rotation (Q135). Both
    spell out *a pawn move is progress*, a fact about chess, and `CLAUDE.md`'s fifth rule
    says the engine may not hold one.
    ``docs/reference/derived-irreversibility.md`` is the costing and the corpus table.

    Four readings are the graph's rather than this function's, which is the point of
    answering it here where the geometry already is:

    * **A facing makes a piece reversible.** The core generates a rotation to *any* other
      orientation in one move, so two rotations at one square are a two-cycle. The
      decision's "no ``facing``" is a consequence, not a clause. A facing no move turns
      (schema 25, ``facing_turns``) has no rotation edges: each orientation is a graph of
      its own, and a pawn placed facing backward is as one-way as one facing forward.
    * **A seam is answered exactly.** A rank seam under a forward-only piece is a cycle;
      a file seam under the same piece is not. Nothing here knows what a seam is --
      ``_compile_square`` already put it in the table.
    * **A promoting move is a sink, not an edge.** It is ``MOVE_PROMOTION``, irreversible
      by kind already, and it ends the piece. Only a *mandatory* promotion whose targets
      exclude the piece itself is removed; a declinable one leaves a ``MOVE_NORMAL``
      behind and keeps its edge. The **node** stays: a piece can be dropped or struck onto
      a promotion square and stand there, and from there it may move on -- which on a
      torus is the difference between a right answer and a lucky one. Schema 15's
      ``on_leaving`` makes a mandatory rule's moves *out of* the zone promote too, and
      the graph needs no word for it: a sink has no incoming move edge, so no cycle of
      moves passes through one, and taking its outgoing edges away as well would change
      no answer.
    * **A piece with no move is not one-way, it is still.** Vacuous acyclicity is not a
      claim worth making and the flag would never be read.

    Deliberately over-approximate in two places, both of which can only *lose* a claim
    and never make a false one: a hop contributes its whole ray though it needs a screen
    to travel it, and a ray square contributes an edge whether or not its mode lets the
    piece stop there. Answered for both players and granted only if both agree, so one
    army's seam cannot hand the other a claim it has not got.

    Bounded work: one depth-first search per ``(player, piece type)`` over
    ``n_facings x n_squares`` nodes, once per ruleset -- the same shape and bound as
    ``_goal_distances``, which walks the same tables.
    """
    one_way: set[int] = set()
    for piece_type in range(n_types):
        axes = facing_axes[piece_type]
        facings = [turn * (n_facings // axes) for turn in range(axes)]
        # Squares this type cannot arrive on and still be this type.
        sinks: list[frozenset[int]] = []
        for player in (0, 1):
            sinks.append(
                frozenset(
                    square
                    for rule in promotions
                    if rule.piece_type == piece_type
                    and rule.mandatory
                    and piece_type not in rule.targets[player]
                    for square in rule.squares[player]
                )
            )
        turns = not facing_turns or facing_turns[piece_type]
        if all(
            _acyclic(
                leaps, slides, hops, piece_type, player, facings,
                n_squares, n_facings, sinks[player], sweeps, turns,
            )
            for player in (0, 1)
        ):
            one_way.add(piece_type)
    return frozenset(one_way)


def _acyclic(
    leaps: Any,
    slides: Any,
    hops: Any,
    piece_type: int,
    player: int,
    facings: list[int],
    n_squares: int,
    n_facings: int,
    sinks: frozenset[int],
    sweeps: Any = None,
    turns: bool = True,
) -> bool:
    """Whether one piece's move graph for one player holds no cycle, and any move at all.

    Iterative three-colour depth-first search: grey is on the current path, so an edge
    into grey is a cycle. Iterative rather than recursive because a slide's ray on a large
    board makes the graph deeper than the interpreter's stack is willing to go.
    """
    adjacency: list[list[int]] = [[] for _ in range(len(facings) * n_squares)]
    moves = 0
    for slot, facing in enumerate(facings):
        frame = frame_index(player, facing, n_facings)
        for origin in range(n_squares):
            node = slot * n_squares + origin
            # A sweeper's rays are edges of its move graph too (schema 13).
            swept = (sweeps[frame][piece_type][origin],) if sweeps else ()
            for table in (
                leaps[frame][piece_type][origin],
                slides[frame][piece_type][origin],
                hops[frame][piece_type][origin],
                *swept,
            ):
                for entry in table:
                    for target in entry[0]:
                        moves += 1
                        if target not in sinks:
                            adjacency[node].append(slot * n_squares + target)
            for other in range(len(facings) if turns else 0):
                if other != slot:
                    adjacency[node].append(other * n_squares + origin)
    if not moves:
        return False

    WHITE, GREY, BLACK = 0, 1, 2
    colour = [WHITE] * len(adjacency)
    for start in range(len(adjacency)):
        if colour[start] != WHITE:
            continue
        colour[start] = GREY
        stack = [(start, iter(adjacency[start]))]
        while stack:
            node, remaining = stack[-1]
            for nxt in remaining:
                if colour[nxt] == GREY:
                    return False
                if colour[nxt] == WHITE:
                    colour[nxt] = GREY
                    stack.append((nxt, iter(adjacency[nxt])))
                    break
            else:
                colour[node] = BLACK
                stack.pop()
    return True


def _compile_repetition_draw(value: Any) -> int:
    """The repetition threshold, with schema 10's ``"never"`` resolved to a number.

    **This is the whole of Q38 in the engine, and it is deliberately not a branch in the
    core.** ``game.rs`` tests ``repetition_count() >= t.repetition_draw`` against a
    ``u32``, and a repetition count is bounded by the length of the game -- so a threshold
    above every reachable count is a rule that can never fire, with no second test on the
    hot path and no new boundary field. Schema 6's rule, in one line: the boundary carries
    the answer, not the question.

    ``NO_REPETITION_DRAW`` is ``u32::MAX``, which is unreachable by construction rather
    than by argument about ``max_plies``: nothing in the core adds to this value, it is
    only ever compared against, and a count that large cannot be stored in the history it
    is counted from.
    """
    return NO_REPETITION_DRAW if value == REPETITION_NEVER else int(value)


def _without_self_destroying_captures(
    pieces: list[dict[str, Any]], rules: dict[str, Any], index_of: dict[str, int]
) -> list[dict[str, Any]]:
    """The pieces as the movement tables read them: a royal that cannot capture loses its
    captures (Q1032).

    Under ``explode_on_capture`` the captor always dies, so a royal whose every capture
    fires the blast -- it is in ``by`` (or ``by`` is absent) and ``of`` is absent -- can
    take nothing without losing the game. Atomic's and kamikaze's convention, and their
    descriptions' words, is that such a king *cannot capture*: it is offered no capture,
    and since it attacks nothing it gives no check either, so two kings may stand side by
    side. Stripping the capture bit here says both at once; the attack table is inverted
    from the capture bits and needs no exception. A royal outside ``by`` (queen-bomb) or
    one whose captures fire only on the ``of`` victims (queens-wake) captures as in chess
    and attacks as in chess. Schema 2's ``silent_types`` silenced every royal under any
    blast, which was this rule before ``by``/``of`` existed, and it left the capture in
    the tables, so a king could step beside the enemy king and be taken by it
    (``docs/reference/engine-and-spec-record.md``, "A royal that cannot capture").
    """
    explosion = rules.get("explode_on_capture")
    if not isinstance(explosion, dict):
        return pieces
    by = _explosion_selection(explosion, "by", index_of)
    of = _explosion_selection(explosion, "of", index_of)
    if by == frozenset() or of is not None:
        return pieces  # no blast at all, or one only some victims set off
    stripped = list(pieces)
    for index, piece in enumerate(pieces):
        if not piece.get("royal", False) or (by is not None and index not in by):
            continue
        moves = []
        for descriptor in piece.get("moves", []):
            mode = descriptor.get("mode", "both")
            if mode == "capture":
                continue
            moves.append({**descriptor, "mode": "move"} if mode == "both" else descriptor)
        stripped[index] = {**piece, "moves": moves}
    return stripped


def _explosion_selection(
    explosion: dict[str, Any], key: str, index_of: dict[str, int]
) -> frozenset[int] | None:
    """One of schema 10's ``by`` / ``of`` lists, resolved to types.

    Three answers, and the two degenerate ones are why this is not a set comprehension:

    * ``None`` -- the key is absent, or names every declared type. Both are "any", so a
      spelled-out full list is an inert *spelling* and compiles to the game that was
      already there, the way ``{"effect": "capture"}`` does. This is also what keeps the
      boundary field at its inert default for every pre-schema-10 ruleset.
    * the **empty set** -- the key is present and names nobody. There is then no capture
      that can fire the rule, so the caller drops the whole explosion.
    * a non-empty set -- the types that do fire it.
    """
    listed = explosion.get(key)
    if not isinstance(listed, list):
        return None
    types = frozenset(index_of[pid] for pid in listed if pid in index_of)
    if not types:
        return types
    return None if len(types) == len(index_of) else types


def _compile_explosion(
    explosion: dict[str, Any] | None,
    index_of: dict[str, int],
    width: int,
    height: int,
    wrap: WrapRule | None = None,
    obstacles: frozenset[int] = frozenset(),
    hex_radius: int = 0,
) -> ExplosionRule | None:
    if not explosion:
        return None
    by = _explosion_selection(explosion, "by", index_of)
    of = _explosion_selection(explosion, "of", index_of)
    if by == frozenset() or of == frozenset():
        # Schema 10 (Q18). `by: []` is a rule no capture can fire, so it *is* the game
        # that was already there written the long way -- the shape `_compile_strike` has
        # for a chain that leaves nothing alive. Dropping it here is what lets the core
        # keep one test rather than two, and what `complexity._RULE_ACTIVE` agrees with
        # so the declaration is not charged 1.5 for nothing (Q56's ruling,
        # docs/reference/bugs-investigations.md, "Ruled on in advance, for Q86").
        return None
    blast_radius = explosion.get("radius", 1)
    # Generation 48 (SR1b): a stated shape replaces the radius's box, its offsets in the
    # box's own order (file offset, then rank offset) so a shape that *is* the box compiles
    # to the box's table. Absent, the box is built exactly as before.
    if "blast" in explosion:
        offsets = sorted(blast_directions(explosion["blast"]))
    else:
        offsets = [
            (dx, dy)
            for dx in range(-blast_radius, blast_radius + 1)
            for dy in range(-blast_radius, blast_radius + 1)
        ]
    # A blast reads the board's shape, not a player's permission: the two file edges are
    # one seam whoever may cross it, so a capture beside the join takes what is on the
    # other side of it. With no seam these are both false and ``_wrap_axis`` (or, on a
    # hexagon, the six-copy search with every side closed) is the bounds test it
    # replaces. ``hex_radius`` is a hexagon's R (Q281a2, unreachable until the gate).
    join_files = bool(wrap and wrap.files_join)
    join_ranks = bool(wrap and wrap.ranks_join)
    seam_open = wrap.seam_open if wrap else (lambda _side: False)
    blast = []
    for square in range(width * height):
        file, rank = square % width, square // width
        neighbours = []
        for dx, dy in offsets:
            if dx == 0 and dy == 0:
                continue
            nf, nr = file + dx, rank + dy
            neighbour = _land(
                nf, nr, width, height, join_files, join_files, join_ranks, join_ranks,
                hex_radius, seam_open,
            )
            # A blast clears squares, and a wall is not a square that can be
            # cleared: nothing is ever standing on one, so listing it would be a
            # per-capture read of a square that is always empty.
            if (
                neighbour is not None
                and neighbour != square
                and neighbour not in obstacles
                and neighbour not in neighbours
            ):
                neighbours.append(neighbour)
        blast.append(tuple(neighbours))
    return ExplosionRule(
        immune_types=frozenset(index_of[p] for p in explosion.get("immune", [])),
        blast=tuple(blast),
        by_types=by,
        of_types=of,
    )


def _compile_landing(
    landing: dict[str, Any],
    n_squares: int,
    width: int,
    height: int,
    orientation: str,
    wrap: WrapRule | None,
    obstacles: frozenset[int],
    radius: int = 0,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Resolve ``landing`` into ``[player][square] -> candidate squares`` (schema 12, Q20).

    Each vector is resolved as a leap from the struck piece's square, by
    ``_compile_square`` itself, so the mover's frame, a seam and a wall mean here what
    they mean for a move. The square itself is never a candidate. Ascending order is the
    order the core fans a capture out in, and a stored landing index names an entry here.
    """
    thrower = {
        "moves": [
            {
                "kind": "leap",
                "vectors": landing["vectors"],
                "symmetry": landing.get("symmetry", "none"),
            }
        ]
    }
    table = []
    for player in (0, 1):
        per_square = []
        for square in range(n_squares):
            leaps, *_ = _compile_square(
                thrower, player, square, width, height, orientation, wrap, obstacles,
                radius=radius,
            )
            per_square.append(tuple(sorted({t for squares, _, _ in leaps for t in squares})))
        table.append(tuple(per_square))
    return tuple(table)


def _compile_strike(
    strike: dict[str, Any] | None,
    index_of: dict[str, int],
    n_types: int,
    landing: tuple[tuple[tuple[int, ...], ...], ...] = (),
    rebirth: tuple[Sequence[int], int, int, bool] | None = None,
) -> StrikeRule | None:
    """Resolve the strike chain into one target type per victim type.

    ``{"effect": "capture"}`` compiles to ``None`` -- the game that was already there --
    which is what makes the rule inert when it is spelled but not used, the way
    ``turn`` and ``wrap`` are. A chain that leaves nothing alive compiles to ``None``
    too, for the same reason: it *is* plain capture, written the long way.

    Schema 17 (Q540) gives ``"capture"`` two positions that say something (``stay``,
    rifle; ``home``, circe) and adds ``switches_sides`` (Andernach) under any effect.
    A1 adds ``by``: only the listed types' captures strike, a row mask over the table.
    ``rebirth`` is ``(initial board, width, n_facings, hex lattice)``, from which
    ``home`` builds its table; without it the rule says only whether anything survives.
    """
    if not strike:
        return None
    effect = strike.get("effect", "capture")
    position = strike.get("position", "advance")
    immune = frozenset(index_of[p] for p in strike.get("immune", []) if p in index_of)
    switching = frozenset(
        index_of[p] for p in strike.get("switches_sides") or [] if p in index_of
    )
    # A1 (`docs/archive/plan-pieces-moves.md` §6.2): only a capture by a listed type strikes, so
    # every other mover's row is plain capture. `None` is every type, the rule as it was.
    by = strike.get("by")
    strikers = (
        frozenset(index_of[p] for p in by if p in index_of) if by is not None else None
    )
    strike_to = [-1] * n_types
    if effect == "capture":
        # Schema 17 (Q540). Under "capture" a struck piece is the piece it was, so two
        # positions say something: `stay` removes it and leaves the attacker where it
        # stood (rifle), and `home` puts it back on its home square (circe). Every other
        # position is plain capture, and with no `switches_sides` the rule is `None`.
        for victim in range(n_types):
            if victim in immune:
                continue
            if position == "stay":
                strike_to[victim] = REMOVED_IN_PLACE
            elif position == "home":
                strike_to[victim] = victim
    else:
        for link in strike.get("chain") or []:
            source, target = link["from"], link["to"]
            victim = index_of[source]
            if victim in immune:
                continue  # a strike never reaches it; this is a plain capture
            strike_to[victim] = index_of[target]

    # Schema 17: the attacker changes sides on every capture that reaches its victim. A
    # capture of an immune piece is a plain capture, and stays one for the attacker too.
    mover_flips: tuple[bool, ...] = ()
    if switching:
        mover_flips = tuple(
            mover in switching and victim not in immune
            and (strikers is None or mover in strikers)
            for mover in range(n_types)
            for victim in range(n_types)
        )
        if not any(mover_flips):
            mover_flips = ()

    # The victim column, said the general way: every attacker meets the same table,
    # which is precisely what "keyed by the victim alone" means. `compact_to` reads it
    # straight back out, so the boundary still sends one array of `n_types`. Under `by`
    # (A1) an unlisted mover's row is all -1, the rows differ, `compact_to` is `None`, and
    # the table crosses as the contact arrays, as Andernach's does.
    if strikers is None:
        onto_to = tuple(strike_to) * n_types
    else:
        onto_to = tuple(
            target if mover in strikers else -1
            for mover in range(n_types)
            for target in strike_to
        )
    rule = StrikeRule(
        effect=effect,
        position=position,
        immune_types=immune,
        n_types=n_types,
        onto_to=onto_to,
        mover_to=(-1,) * (n_types * n_types),
        spelling="strike",
        landing=landing,
        mover_flips=mover_flips,
    )
    if not (rule.survives_anything or rule.removes_in_place or rule.changes_mover):
        return None
    if position == "home" and rebirth is not None:
        rule = replace(rule, rebirth=_compile_rebirth(*rebirth, n_types))
    return rule


def _compile_rebirth(
    board: Sequence[int], width: int, n_facings: int, hex_lattice: bool, n_types: int
) -> tuple[int, ...]:
    """Where each piece is reborn under ``position: "home"`` (schema 17, Q540; circe).

    ``rebirth[(owner * n_types + type) * n_squares + square]``: the home square of a piece
    of ``owner`` and ``type`` taken on ``square``, or -1 where its side's setup has no
    piece of that type (a promoted queen in a game with no queen in the setup is removed).

    A type's home squares are where the setup stands it for that side. Which attribute
    picks among them depends on how they stand (Q963, orthodox Circe, P. Monreal 1967):

    * **Colour first** for a type with at most one home on each colour of a square board
      (chess's rook, knight and bishop; a queen or king with one home): the home on the
      capture square's colour, failing that all of them.
    * **File first** for every other type (chess's pawns, a home on every file; and every
      type on a hex board, which has three colours, so the colour step is skipped there):
      the home on the capture's file, failing that the one on the capture square's colour,
      failing that all of them.

    Within the set kept, the nearest by file, then by rank, then the lower square. On
    chess's setup this is orthodox Circe: a pawn goes back to its file, a rook, knight or
    bishop to the home square of the capture square's colour (a white knight taken on b4
    to g1, so a bishop keeps its square colour), a queen to her own square. Until Q963 the
    file came first for every type, so a knight taken on b4 went to b1.
    """
    n_squares = len(board)
    stride = n_facings * n_types
    homes: dict[tuple[int, int], list[int]] = {}
    for square, value in enumerate(board):
        if value:
            owner, piece_type = (value - 1) // stride, (value - 1) % n_types
            homes.setdefault((owner, piece_type), []).append(square)

    def colour(square: int) -> int:
        return (square % width + square // width) % 2

    table = [-1] * (2 * n_types * n_squares)
    for (owner, piece_type), squares in homes.items():
        if owner > 1:
            continue  # a wild piece is never struck: `npcs` is refused beside a strike
        colour_first = not hex_lattice and all(
            sum(1 for home in squares if colour(home) == shade) <= 1 for shade in (0, 1)
        )
        for taken in range(n_squares):
            file, rank = taken % width, taken // width
            same_file = [home for home in squares if home % width == file]
            same_colour = (
                [home for home in squares if colour(home) == colour(taken)]
                if not hex_lattice
                else []
            )
            # A colour-first type never asks the file: on a home file of the other colour
            # its home is the other one of the pair (a knight taken on b4 goes to g1).
            kept = (same_colour if colour_first else same_file or same_colour) or squares
            best = min(
                kept,
                key=lambda home: (abs(home % width - file), abs(home // width - rank), home),
            )
            table[(owner * n_types + piece_type) * n_squares + taken] = best
    return tuple(table)


def _compile_contacts(
    contacts: dict[str, Any] | None,
    index_of: dict[str, int],
    n_types: int,
    landing: tuple[tuple[tuple[int, ...], ...], ...] = (),
) -> StrikeRule | None:
    """Resolve the contact table into the two ``n_types x n_types`` arrays (schema 10).

    The same three drops ``_compile_strike`` makes, for the same reason: a table with no
    link, a table whose every target is immune, and a table that changes neither piece
    are all plain capture written the long way, and the boundary carries the answer
    rather than the question. Q56 ruled on the empty-list reading before the key existed
    (``docs/reference/bugs-investigations.md``, "Ruled on in advance, for Q86") -- an empty table is no rule, so ``complexity._RULE_ACTIVE`` agrees with
    the compiler and the declaration is not charged for nothing.
    """
    if not contacts:
        return None
    immune = frozenset(index_of[p] for p in contacts.get("immune", []) if p in index_of)
    links = contacts.get("links") or []
    onto_to = [-1] * (n_types * n_types)
    mover_to = [-1] * (n_types * n_types)
    # Schema 11: a link with `own` fills a second pair of tables. They stay empty unless
    # some link uses it, so every earlier table compiles to what it did.
    meets_own = any(link.get("own") is True for link in links)
    own_onto_to = [-1] * (n_types * n_types) if meets_own else []
    own_mover_to = [-1] * (n_types * n_types) if meets_own else []
    for link in links:
        mover, onto = index_of[link["mover"]], index_of[link["onto"]]
        own = link.get("own") is True
        if onto in immune and not own:
            continue  # a contact never reaches it; this is a plain capture
        cell = mover * n_types + onto
        into, mover_into = (own_onto_to, own_mover_to) if own else (onto_to, mover_to)
        if "onto_to" in link:
            into[cell] = index_of[link["onto_to"]]
        if "mover_to" in link:
            mover_into[cell] = index_of[link["mover_to"]]

    rule = StrikeRule(
        effect="contact",
        position=contacts.get("position", "advance"),
        immune_types=immune,
        n_types=n_types,
        onto_to=tuple(onto_to),
        mover_to=tuple(mover_to),
        spelling="contacts",
        own_onto_to=tuple(own_onto_to),
        own_mover_to=tuple(own_mover_to),
        landing=landing,
    )
    return rule if rule.survives_anything or rule.changes_mover else None


def _portal_end_pieces(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The internal piece types schema 19 adds, one per portal end, in group order.

    Each is its group's movement type (`schema.portal_movement_types`) under the end's own
    id (``1o``, ``1b``, ``2a``, ...), so every table is built for it as for a declared piece:
    its moves are the group's. Empty without ``rules.portals``, which is what keeps every
    other game's tables the size they were.
    """
    rules = spec.get("rules") or {}
    groups = portal_groups(rules)
    if not groups:
        return []
    movers = {piece["id"]: piece for piece in portal_movement_types(spec)}
    ids = iter(portal_end_ids(groups))
    ends: list[dict[str, Any]] = []
    for group, mover in zip(groups, portal_movement_of_groups(spec), strict=True):
        portal = movers[mover]
        for end in group["ends"]:
            called = end["key"] if group["spelling"] == "pair" else end["letter"]
            ends.append(
                {
                    "id": next(ids),
                    "name": f"{portal.get('name', portal['id'])} {group['number']} {called}",
                    "value": 0,
                    "moves": portal.get("moves", []),
                }
            )
    return ends


def _portal_starts(
    spec: dict[str, Any], declared_types: int, width: int, height: int
) -> tuple[tuple[int, int], ...]:
    """``(square, internal type)`` for every momentum portal end at ply zero.

    A fold end (schema 22) never stands on the board byte, so it is left out here and
    starts at `_fold_starts`' square instead.
    """
    starts: list[tuple[int, int]] = []
    piece_type = declared_types
    for group in portal_groups(spec.get("rules") or {}):
        for end in group["ends"]:
            coords = parse_square(end["at"], width, height)
            assert coords is not None  # guaranteed by validation
            if group["kind"] != "fold":
                starts.append((coords[1] * width + coords[0], piece_type))
            piece_type += 1
    return tuple(starts)


def _fold_starts(rules: dict[str, Any], width: int, height: int) -> tuple[int, ...]:
    """Schema 22: where each portal end starts if it is a fold, in end order; -1 for a
    momentum end, which stands on the board byte instead (`_portal_starts`)."""
    starts: list[int] = []
    for group in portal_groups(rules):
        for end in group["ends"]:
            coords = parse_square(end["at"], width, height)
            assert coords is not None  # guaranteed by validation
            starts.append(coords[1] * width + coords[0] if group["kind"] == "fold" else -1)
    return tuple(starts)


def _apply_portal_immunity(
    matrix: tuple[tuple[bool, ...], ...], declared_types: int, portal_types: set[int]
) -> tuple[tuple[bool, ...], ...]:
    """Schema 19: no type may take a portal end -- point 5 of the brief, *a portal is not
    captured* -- stated in the schema-3 capture matrix every capture test already reads.

    So an end is an uncapturable piece to every ordinary move: a slide stops at it and a
    leap does not land on it. Going *in* is a transit, which the core generates on its own
    walk (`Game::push_transits`); an end closed to the mover is then just the obstacle
    point 11 asks for. A portal's movement type goes too (`schema.portal_movement_types`):
    it never stands on the board, and the prose then says of it what is true of its ends
    (`describe._moveless_fates`).
    Unchanged where nothing was added.
    """
    if len(matrix) == declared_types:
        return matrix
    return tuple(
        tuple(
            allowed and victim < declared_types and victim not in portal_types
            for victim, allowed in enumerate(row)
        )
        for row in matrix
    )


def _portal_exits(
    width: int, height: int, wrap: WrapRule | None, obstacles: frozenset[int]
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """``[player][square][direction]`` -> the square one king's step away, or -1.

    Through `_land`, the one place a step is landed, under each player's own wrap
    permission (a move reads the mover's): off the board or onto a wall is -1. Portals are
    refused on a hexagon, so this is the grid reading.
    """
    exits: list[tuple[tuple[int, ...], ...]] = []
    for player in (0, 1):
        low_file, high_file = wrap.files[player] if wrap else (False, False)
        low_rank, high_rank = wrap.ranks[player] if wrap else (False, False)
        per_square: list[tuple[int, ...]] = []
        for square in range(width * height):
            file, rank = square % width, square // width
            row: list[int] = []
            for step_x, step_y in PORTAL_DIRECTIONS:
                landed = _land(
                    file + step_x, rank + step_y, width, height,
                    low_file, high_file, low_rank, high_rank,
                )
                row.append(-1 if landed is None or landed in obstacles else landed)
            per_square.append(tuple(row))
        exits.append(tuple(per_square))
    return tuple(exits)


def _compile_portals(
    rules: dict[str, Any],
    declared_types: int,
    n_types: int,
    width: int,
    height: int,
    wrap: WrapRule | None,
    obstacles: frozenset[int],
    momentum: tuple[Any, Any, Any],
    attack_dirs: list[list[list[int]]],
    zobrist_bar: tuple[int, ...] = (),
    declared_ids: tuple[str, ...] = (),
    movement_ids: tuple[str, ...] = (),
) -> PortalRule:
    """``rules.portals`` resolved into the tables `PortalRule` describes.

    Every group, in either spelling, through the one normaliser (`portal_groups`): an end's
    ``dest`` is the type of the end its ``to`` names, and its ``admits`` mask per player is
    every declared type its ``carries`` names (every declared type where it has none) when
    the player is in its ``enter``, else nothing (`docs/archive/plan-portals-2.md` §2.2-2.3).
    """
    groups = portal_groups(rules)
    index_of = {pid: index for index, pid in enumerate(declared_ids)}
    # Every declared piece but the portals' movement types, which never stand on the board.
    movement = set(movement_ids)
    every = sum(
        1 << index for index, pid in enumerate(declared_ids) if pid not in movement
    )
    dest = [-1] * n_types
    no_back = [False] * n_types
    admits = [(0, 0)] * n_types
    movers: list[frozenset[int]] = [frozenset()] * n_types
    carried = [False] * n_types
    end_types: list[int] = []
    squares: list[int] = []
    group_of: list[int] = []
    kinds: list[str] = []
    first = declared_types
    for index, group in enumerate(groups):
        for place, end in enumerate(group["ends"]):
            piece_type = first + place
            if end["to"] is not None:
                dest[piece_type] = first + end["to"]
            named = end["carries"]
            carried[piece_type] = named is not None
            mask = every if named is None else sum(
                1 << index_of[pid] for pid in named if pid in index_of
            )
            admits[piece_type] = (
                mask if 0 in end["enter"] else 0, mask if 1 in end["enter"] else 0,
            )
            movers[piece_type] = end["moved_by"]
            # A bar the validator admits is one a move can meet (Q879, refused for stored
            # specs too since Q1008d), so the flag is the spec's own.
            no_back[piece_type] = bool(group["no_straight_back"])
            end_types.append(piece_type)
            kinds.append(group["kind"])
            coords = parse_square(end["at"], width, height)
            assert coords is not None  # guaranteed by validation
            squares.append(coords[1] * width + coords[0])
            group_of.append(index)
        first += len(group["ends"])
    leap_dirs, slide_dirs, hop_dirs = momentum
    return PortalRule(
        declared_types=declared_types,
        end_types=tuple(end_types),
        squares=tuple(squares),
        group_of=tuple(group_of),
        dest=tuple(dest),
        admits=tuple(admits),
        carried=tuple(carried),
        movers=tuple(movers),
        exits=_portal_exits(width, height, wrap, obstacles),
        leap_dirs=_freeze(leap_dirs),
        slide_dirs=_freeze(slide_dirs),
        hop_dirs=_freeze(hop_dirs),
        attack_dirs=tuple(
            tuple(tuple(per_square) for per_square in owner) for owner in attack_dirs
        ),
        # Schema 20: empty unless some pair bars going straight back, so every schema 19
        # game's tables are what they were.
        no_back=tuple(no_back) if any(no_back) else (),
        zobrist_bar=zobrist_bar if any(no_back) else (),
        kinds=tuple(kinds),
    )


def _compile_npcs(
    rules: dict[str, Any],
    pieces: list[dict[str, Any]],
    index_of: dict[str, int],
    width: int,
    height: int,
    orientation: str,
) -> NpcRule | None:
    """Resolve every wild piece's route into the three per-square tables.

    ``None`` when the spec declares no wild piece, which is every game before schema 9.
    The routes come from ``schema.npc_routes`` -- the same resolver the validator used
    to refuse overlaps and walls -- so a route means one thing to both. A mirrored entry
    is a second wild piece walking the reflected squares in the same order, and it
    carries its original's type and bounty.

    The bounty defaults to the type's declared ``value``: what a piece is worth is what
    taking it pays, unless the designer says otherwise.
    """
    npcs = rules.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return None
    n_squares = width * height
    nxt = [-1] * n_squares
    bounty = [0.0] * n_squares
    piece_of = [-1] * n_squares
    prev = [-1] * n_squares
    for label, squares in npc_routes(rules, width, height, orientation):
        index = int(label.split("[")[1].split("]")[0])
        entry = npcs[index]
        piece_type = index_of[entry["piece"]]
        stated = entry.get("bounty")
        worth = float(stated if stated is not None else pieces[piece_type].get("value", 1))
        line = npc_turns_back(entry)
        last = len(squares) - 1
        for position, square in enumerate(squares):
            nxt[square] = squares[(position + 1) % len(squares)]
            bounty[square] = worth
            piece_of[square] = piece_type
            if line:
                # Generation 38 (N5): a line never wraps; each end holds itself.
                nxt[square] = squares[position + 1] if position < last else square
                prev[square] = squares[position - 1] if position > 0 else square
    # Schema 23: ``also_at`` is more starts on a route and nothing else -- the extra pieces
    # share the entry's type and bounty, so the three tables above are the route's.
    starts = tuple(
        (square, index_of[npcs[index]["piece"]])
        for index, _, square in npc_starts(rules, width, height, orientation)
    )
    return NpcRule(
        next=tuple(nxt), bounty=tuple(bounty), piece=tuple(piece_of), starts=starts,
        respawn=_respawn_halves(rules, width, height, orientation),
        prev=tuple(prev) if any(npc_turns_back(npc) for npc in npcs) else (),
    )


def _compile_pounce(
    rules: dict[str, Any],
    index_of: dict[str, int],
    piece_values: tuple[float, ...],
    attack_sources: tuple[tuple[tuple[AttackSource, ...], ...], ...],
    width: int,
    height: int,
    orientation: str,
    hexagonal: bool,
) -> PounceRule | None:
    """Generation 39 (N3b): the pounce tables (``PounceRule``), or ``None`` where no entry
    pounces, which is every game before the rule."""
    npcs = rules.get("npcs")
    n_types = len(piece_values)
    kind = [0] * n_types
    for npc in npcs if isinstance(npcs, list) else []:
        stated = npc_pounce(npc)
        if stated is not None:
            kind[index_of[npc["piece"]]] = POUNCE_KINDS.index(stated) + 1
    if not any(kind):
        return None
    n_squares = width * height
    values = sorted(set(piece_values), reverse=True)
    value_rank = tuple(values.index(value) for value in piece_values)
    second = [0] * n_squares
    for square in range(n_squares):
        file, rank = reflect_square(square % width, square // width, width, height, orientation)
        second[rank * width + file] = square
    distance = []
    for a in range(n_squares):
        for b in range(n_squares):
            dx, dy = b % width - a % width, b // width - a // width
            distance.append(hex_distance(dx, dy) if hexagonal else max(abs(dx), abs(dy)))
    # Generation 40 (N3c): only a ``"prey"`` pouncer reads its own reach; an ``"attackers"``
    # one reads the players' inverse table, which is already compiled.
    prey = POUNCE_KINDS.index("prey") + 1
    reach: list[list[list[tuple[int, tuple[int, ...], tuple[int, ...], int]]]] = [
        [[] for _ in range(n_squares)] if kind[piece_type] == prey else []
        for piece_type in range(n_types)
    ]
    for target in range(n_squares):
        for origin, value, prefix, window, screens, unmoved, _, sweep, _ in attack_sources[0][target]:
            piece_type = value - 1  # player 0, facing 0
            if piece_type < n_types and kind[piece_type] == prey and not unmoved and not sweep:
                reach[piece_type][origin].append((target, tuple(prefix), tuple(window), screens))
    # Generation 41 (N3d, §4.5): the cap per type (entries of one type agree, the validator
    # says), empty where nothing splits.
    split = [0] * n_types
    for npc in npcs if isinstance(npcs, list) else []:
        if npc_split(npc):
            split[index_of[npc["piece"]]] = npc_split(npc)
    return PounceRule(
        split=tuple(split) if any(split) else (),
        kind=tuple(kind),
        value_rank=value_rank,
        reading_rank=(tuple(range(n_squares)), tuple(second)),
        distance=tuple(distance),
        reach=tuple(tuple(tuple(entries) for entries in per_type) for per_type in reach),
    )


def _respawn_halves(
    rules: dict[str, Any], width: int, height: int, orientation: str
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Generation 36 (N1): each respawning half's clock and starts, ``NpcRule.respawn``.

    Read off ``schema.npc_starts``, the one reading of where a route's pieces start, split
    by the mirror label it gives: an entry's own starts, then its mirrored half's, each a
    clock of its own (§4.1).
    """
    npcs = rules["npcs"]
    halves: dict[tuple[int, bool], list[int]] = {}
    for index, label, square in npc_starts(rules, width, height, orientation):
        if npc_respawn(npcs[index]):
            halves.setdefault((index, label.endswith("(mirrored)")), []).append(square)
    return tuple(
        (npc_respawn(npcs[index]), tuple(squares))
        for (index, _), squares in sorted(halves.items())
    )


def _compile_drops(
    drops: dict[str, Any] | None,
    index_of: dict[str, int],
    n_types: int,
    width: int,
    height: int,
    orientation: str,
    obstacles: frozenset[int] = frozenset(),
    pieces: Sequence[dict[str, Any]] = (),
) -> DropRule | None:
    """Where each piece may be dropped, with the walls already taken out.

    **A drop is the one move that reaches a square without travelling to it**, so none of
    what `_compile_square` does for a ray or a leap applies to it: the core picks the
    empty squares, and a wall is empty. Dissolving the obstacle here rather than sending
    it across the boundary is the same answer the rest of the rule gets -- by the time
    anything crosses, the walls are gone. A piece with no region of its own gets one when
    there are walls, because "anywhere" has to stop meaning "including on a wall".

    Found by `chess_variant_sandbox/fuzz.py`, which is the fourth instance of the root cause item 1
    enumerates: code that works a square out for itself is blind to what the compiler
    dissolved. `generate_drops` was on that list as *correct today*, and the entry was
    about its `no_doubled` file bucketing -- which is still correct. This is a different
    blindness in the same function.
    """
    if not drops:
        return None
    region = drops.get("region") or {}
    every_square = frozenset(range(width * height))
    per_player = []
    for player in (0, 1):
        row: list[frozenset[int] | None] = [None] * n_types
        for piece, squares in region.items():
            row[index_of[piece]] = _region_squares(
                squares, player, width, height, orientation
            )
        if obstacles:
            for piece_type in range(n_types):
                allowed = row[piece_type]
                row[piece_type] = (
                    every_square if allowed is None else allowed
                ) - obstacles
        per_player.append(tuple(row))
    stated = drops.get("stock") or {}
    stock = [0] * n_types
    unlimited = [False] * n_types
    for piece, held in stated.items():
        if piece not in index_of:
            continue
        if held == STOCK_UNLIMITED:
            stock[index_of[piece]] = 1
            unlimited[index_of[piece]] = True
        else:
            stock[index_of[piece]] = int(held)
    return DropRule(
        from_captures=bool(drops.get("from_captures", True)),
        regions=(per_player[0], per_player[1]),
        no_doubled=frozenset(index_of[p] for p in drops.get("no_doubled", [])),
        no_mate=frozenset(index_of[p] for p in drops.get("no_mate", [])),
        # Empty rather than a run of zeros when nothing is stocked, so the boundary
        # fields sit at their inert defaults and no committed ruleset's tables move.
        stock=tuple(stock) if any(stock) else (),
        unlimited=tuple(unlimited) if any(unlimited) else (),
        placement=drops.get("placement") is True,
        faces=_drop_faces(drops, index_of, n_types, pieces),
        gates=_gates(drops.get("gating"), width, height, orientation, obstacles),
    )


def _gates(
    gating: Any, width: int, height: int, orientation: str, obstacles: frozenset[int]
) -> tuple[frozenset[int], frozenset[int]] | None:
    """Q1059a: ``drops.gating``'s squares per player, or ``None`` without the key.

    Stated as a promotion zone is (``ranks`` or ``squares``, the first player's side) and
    read by the same function, so the validator's zone and the compiler's are one idea.
    Whether a piece may gate is the core's question (its ``unmoved`` bit, which only a
    setup square carries), so the table is the zone and nothing about who stands on it.
    """
    if not isinstance(gating, dict):
        return None
    first, second = promotion_squares(gating, width, height, orientation)
    return first - obstacles, second - obstacles


def _drop_faces(
    drops: dict[str, Any],
    index_of: dict[str, int],
    n_types: int,
    pieces: Sequence[dict[str, Any]],
) -> tuple[int, ...]:
    """Schema 18's ``either_face`` (Q785): each hand slot's other face, or -1.

    The pairs are ``promoted_from``'s, read the other way: a piece that names a base as
    its ``promoted_from`` is that base's other face, so a captured dragon, which enters
    the hand as a rook, may come back as either. The validator refuses a base with two
    faces and every spelling under which a hand could hold a face itself
    (`schema._check_either_face_coherence`), so each slot has at most one face and each
    face one slot. Empty without the flag, so no earlier game's tables move.
    """
    if drops.get("either_face") is not True:
        return ()
    faces = [-1] * n_types
    for piece in pieces:
        base = piece.get("promoted_from")
        if base in index_of and base != piece["id"]:
            faces[index_of[base]] = index_of[piece["id"]]
    return tuple(faces)


def _side_types(
    board: bytes, n_types: int, n_facings: int, types: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Each player's subset of ``types``: the ones it has at least one of on the opening
    board, in the order listed, each once.

    Schema 27's per-side type list (Q936; `docs/plan-spy.md` §2.4), read off the compiled
    opening board rather than the JSON, so it is the army `_build_initial_position` placed,
    mirror and all. A wild piece is nobody's and counts for no one. ``"per_type": "own"``
    resolves its two lists here, and the spy's two lists of spy types come from the same
    place (``SpyRule``, Q935a), so the two rules share one compiled shape; the core counts
    them differently (``Game::short_of`` each type apart, ``Game::has_spy`` summed).
    """
    started: tuple[set[int], set[int]] = (set(), set())
    for value in board:
        if not value:
            continue
        owner = (value - 1) // (n_facings * n_types)
        if owner < 2:
            started[owner].add((value - 1) % n_types)
    lists: list[tuple[int, ...]] = []
    for player in (0, 1):
        seen: list[int] = []
        for piece_type in types:
            if piece_type in started[player] and piece_type not in seen:
                seen.append(piece_type)
        lists.append(tuple(seen))
    return lists[0], lists[1]


def _compile_extinction(
    extinction: dict[str, Any] | None,
    index_of: dict[str, int],
    board: bytes,
    n_types: int,
    n_facings: int,
) -> ExtinctionRule | None:
    if not extinction:
        return None
    pieces = extinction["pieces"]
    any_piece = pieces == "any"
    piece_types = () if any_piece else tuple(index_of[p] for p in pieces)
    spelled = extinction.get("per_type", False)
    # The validator refuses the flag over ``any`` and over one type, so where it is
    # accepted it always counts two or more types apart.
    per_type = spelled in (True, "own") and not any_piece
    own = per_type and spelled == "own"
    side_types: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    if own:
        side_types = _side_types(board, n_types, n_facings, piece_types)
    elif per_type:
        side_types = (piece_types, piece_types)
    return ExtinctionRule(
        piece_types=piece_types,
        any_piece=any_piece,
        min_count=extinction.get("min_count", 1),
        result=extinction.get("result", "loss"),
        per_type=per_type,
        own=own,
        side_types=side_types,
    )


# ----------------------------------------------------------------------- zobrist


def _build_zobrist(
    digest: str,
    n_squares: int,
    n_types: int,
    width: int,
    n_facings: int = 1,
    n_owners: int = 2,
    portal_bar: bool = False,
    fold_ends: int = 0,
    simultaneous: bool = False,
    initiative: bool = False,
    rest_max: int = 0,
    respawn_keys: int = 0,
    heading_squares: tuple[int, ...] = (),
    purse: bool = False,
    push_bar: bool = False,
) -> tuple[
    tuple[tuple[int, ...], ...],
    int,
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, tuple[int, ...], tuple[int, ...]] | None,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Seed the Zobrist table from the spec hash.

    Seeding from the ruleset rather than the clock means a position hash is stable
    across threads, processes and runs, so transposition tables and repetition records
    from a tournament worker mean the same thing to the analysis that reads them back.
    """
    rng = random.Random(int(digest, 16))
    # One row per board value the encoding can produce, which with facing is
    # ``n_facings`` times as many. A facing-carrying value gets its own term for
    # nothing, which is most of the argument for putting the facing in the piece byte:
    # two positions that differ only in which way a piece is pointing are different
    # positions, and this is what makes the key say so.
    #
    # A third owner (schema 9) is a third block of rows *after* the two armies', so a
    # ruleset without wild pieces draws exactly the rows it always drew, in the order it
    # always drew them, and every number after them is where it always was.
    pieces = tuple(
        tuple(rng.getrandbits(64) for _ in range(n_squares))
        for _ in range(n_owners * n_facings * n_types + 1)
    )
    side = rng.getrandbits(64)
    ep = tuple(rng.getrandbits(64) for _ in range(width + 1))
    # Square colours, for a *repainting* territory rule only. Drawn unconditionally so
    # that adding a territory rule to a spec does not shift every other random number
    # after it -- the seed is the spec hash, and a table whose layout depends on which
    # rules happen to be present would make two similar rulesets incomparable for no
    # reason. A permanent colouring never reads this: laying a colour down is
    # irreversible, so it clears the repetition window, so the key never has to tell two
    # differently-coloured positions apart.
    colours = tuple(
        tuple(rng.getrandbits(64) for _ in range(n_squares)) for _ in range(3)
    )
    # Moves remaining in the turn, for a ruleset whose turn is longer than one move.
    # Drawn unconditionally for the same reason the colours are: the seed is the spec
    # hash, and a table whose layout depended on which rules are present would shift
    # every number after it.
    turns = tuple(rng.getrandbits(64) for _ in range(MAX_MOVES_PER_TURN + 1))
    # Schema 20 (Q807): the straight-back bar, ``[left]`` and ``[n_squares + standing on]``.
    # Drawn last and only where a pair sets ``no_straight_back``: after every other draw,
    # so no key above moves, and only where read, so no other game's tables grow.
    bar = (
        tuple(rng.getrandbits(64) for _ in range(2 * n_squares)) if portal_bar else ()
    )
    # Schema 22 (Q843c): where each fold stands, ``[end * n_squares + square]`` for every
    # portal end (a momentum end's rows are never read). Drawn after the bar and only where a
    # group is a fold, so no key above moves and no other game's tables grow.
    fold = tuple(rng.getrandbits(64) for _ in range(fold_ends * n_squares))
    # Schema 26 (Q918a): the initiative's key, drawn last and only under the initiative,
    # then the standoff's (`SimRule`), only in a simultaneous game: after every other draw,
    # so no key above moves, and only where read, so no other game's tables grow.
    sim = None
    if simultaneous:
        holder = rng.getrandbits(64) if initiative else 0
        squares = tuple(rng.getrandbits(64) for _ in range(2 * n_squares))
        states = tuple(rng.getrandbits(64) for _ in range(VOID_RUN))
        sim = (holder, squares, states)
    # Schema 29 (Q983a, `docs/plan-rest.md` §4.2): the resting pieces, one key per (turns
    # left, player, square), drawn last of all and only where a type rests, so no other
    # game's key moves. Where schema 27's rested squares were drawn: after the standoff's in
    # a round game, and in an alternating game (which has no standoff) after the folds.
    resting = tuple(rng.getrandbits(64) for _ in range(rest_max * 2 * n_squares))
    # Generation 36 (N1, `docs/archive/plan-npc.md` §4.1): the respawn clocks, one key per running
    # value of each half's clock, drawn last of all and only where an entry comes back, so no
    # other game's key moves.
    respawn = tuple(rng.getrandbits(64) for _ in range(respawn_keys))
    # Generation 38 (N5, `docs/archive/plan-npc.md` §4.7): a wild piece walking its line back, one key
    # per line square in ascending order, laid out board-sized (0 off every line), drawn last
    # of all and only where an entry turns back, so no other game's key moves.
    heading: tuple[int, ...] = ()
    if heading_squares:
        drawn = {square: rng.getrandbits(64) for square in heading_squares}
        heading = tuple(drawn.get(square, 0) for square in range(n_squares))
    # TC4 (`docs/plan-turncoat.md` §5.2): the purse, one key per (player, amount), amount 0
    # to ``PURSE_MAX``, drawn last of all and only under the rule, so no other game's key
    # moves.
    coins = tuple(rng.getrandbits(64) for _ in range(2 * (PURSE_MAX + 1))) if purse else ()
    # Q1074b6 (K5, `docs/plan-push.md` §3.12): the push-back bar, ``[left]`` and
    # ``[n_squares + standing on]``, drawn last of all and only under
    # ``rules.push.no_push_back``, so no other game's key moves.
    push_bar_keys = (
        tuple(rng.getrandbits(64) for _ in range(2 * n_squares)) if push_bar else ()
    )
    return (
        pieces, side, ep, colours, turns, bar, fold, sim, resting, respawn, heading, coins,
        push_bar_keys,
    )
