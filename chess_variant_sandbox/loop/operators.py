"""Free, local variation: one small mechanical edit to a spec.

The loop had exactly one operator and it was the expensive one. Every candidate came
from an unconstrained model call that regenerates a whole spec, which means there was no
notion of a *local* move in spec space, no landscape to climb, and no way to measure how
much a child inherits from its parent. It also spends the budget in the wrong place: a
model call costs more than the tournament that follows it, and `propose.py` already asks
the model not to spend it on a single-step tweak -- an instruction that is a missing
operator wearing a prompt.

So this is the other half of the operator set, and the division falls out of the cost
structure rather than out of theory:

* **mechanical** (here) -- free, milliseconds, dense local exploration;
* **model-directed** (`llm/propose.py`) -- the jumps: a new piece with a reason to
  exist, a win condition that changes what the game is about.

Every edit is one edit. An operator that changed three things at once would be a small
regeneration, which is the thing this exists not to be.

**Invalid output is discarded, not repaired.** The transforms repair what they can --
removing a piece takes its placements and its rule references with it, and
`Operator.apply` takes away any rule the edit left with nothing to act on
(`transform.settle_inert_rules`, asking the validator's own predicates) -- but a shrunk
board can still strand a placement, and a repair loop that parses the validator's English
is a repair loop that breaks when someone rewords an error. At these prices, drawing
again is simply cheaper than being clever. If repair ever becomes the bottleneck, the
fix is machine-readable validation errors, not a parser.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.loop.grid import descriptor
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.relate import identity_key
from chess_variant_sandbox.spec.schema import (
    CLOCK_BOUNDS,
    REPETITION_NEVER,
    absent_squares,
    can_double_step,
    obstacle_squares,
    parse_square,
    promoting_piece_ids,
    reach_reflection,
    reflect_square,
    sweep_conflicts,
    turn_is_long,
    uses_sweep,
)
from chess_variant_sandbox.spec.transform import (
    capture_is_possible,
    drop_piece,
    pieces,
    pieces_that_can_stand,
    promotion_rank,
    resize_board,
    restrict_captures,
    royal_id,
    set_obstacle,
    set_scoring,
    set_strike,
    set_territory,
    set_turn,
    set_wrap,
    settle_en_passant,
    settle_inert_rules,
    settle_schema_version,
    split_piece,
    strip_rule,
)

# How many operators to try before giving up on producing a valid, different child.
# Generous, because each attempt costs a validation and nothing else.
ATTEMPTS = 16

# The board dimensions the schema allows, which is what a resize has to stay inside.
_MIN_DIM, _MAX_DIM = 3, 16

_SYMMETRIES = ("none", "lateral", "rotate4", "dihedral8")
_SHORT_VECTORS = ([1, 0], [1, 1], [0, 1], [1, 2], [2, 1], [1, 3], [2, 3])


@dataclass(frozen=True)
class Mutation:
    """One child, and what was done to make it."""

    spec: dict[str, Any]
    operator: str
    summary: str

    @property
    def hash(self) -> str:
        return spec_hash(self.spec)


@dataclass(frozen=True)
class Operator:
    """A named local edit. Returns a summary of what it did, or ``None`` if it could
    not apply to this spec at all -- an operator that removes a slider has nothing to do
    in a game with no sliders, and that is not a failure.

    ``apply`` is the edit and then `transform.settle_inert_rules`, so **no operator hands
    back a child the validator refuses as declaring nothing** (Q110). An edit can take
    away what another rule acts on -- the last capture a ``must_capture`` insists on, the
    only route by which a promoting piece stands, the rook a castling needs -- and the
    rule goes with it, as castling goes with its rook in ``drop_piece``. Once here rather
    than once per operator, because Q110 found those pairs spread across twelve operators
    and invisible to the corpus sweep, and the settle is a no-op on any child the
    validator accepts. What went is added to the sentence, because a child described as
    one change that is really two is a listing that lies.
    """

    name: str
    edit: Callable[[dict[str, Any], random.Random], str | None]

    def apply(self, spec: dict[str, Any], rng: random.Random) -> str | None:
        summary = self.edit(spec, rng)
        if summary is None:
            return None
        gone = settle_inert_rules(spec)
        if not gone:
            return summary
        words = [_gone_words(key) for key in gone]
        listed = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
        return f"{summary}, and {listed} had nothing left to act on, so {'it' if len(words) == 1 else 'they'} went too"


# How `Operator.apply` names a rule `settle_inert_rules` took away, in a player's words.
_GONE_WORDS = {
    "must_capture": "compulsory capture",
    "explode_on_capture": "the explosion",
    "drops": "the hand",
    "bare_royal": "the bare-royal ending",
    "check_limit": "the check count",
    "forbid_check": "the ban on giving check",
    "royal_facing": "the ban on royals facing",
    "en_passant": "en passant",
    "castling": "castling",
    "reach": "the win by arrival",
    "promote_on_entry": "promotion on entry",
    "scoring": "the count of points at the clock",
}


def _gone_words(key: str) -> str:
    """One key `settle_inert_rules` returned, in a player's words."""
    if key.startswith("promotion:"):
        return f"the {key.split(':', 1)[1]}'s promotion"
    if key.startswith("immune:"):
        return f"the wild {key.split(':', 1)[1]}'s immunity"
    return _GONE_WORDS.get(key, key.replace("_", " "))


def _non_royal(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [piece for piece in pieces(spec) if not piece.get("royal")]


def _movers(spec: dict[str, Any], kind: str | None = None) -> list[tuple[dict, dict]]:
    """Every (piece, move) pair, optionally of one movement kind."""
    return [
        (piece, move)
        for piece in pieces(spec)
        for move in piece.get("moves", [])
        if kind is None or move.get("kind") == kind
    ]


def _remove_a_piece(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Remove one non-royal piece type, everywhere the spec names it.

    Never the only type the setup places: its placements go with it, and a setup with none
    is refused (*"setup.placements must be a non-empty array"*). A piece that only a hand
    holds -- `place-a-stone`'s stone on `dull` -- leaves the placed type as the one such
    piece, and removing it was a refused child (Q484: 8, one and two steps out). The
    filter changes the draw only on such a spec, so every other child is the one it was.
    """
    candidates = _non_royal(spec)
    if len(candidates) < 2:  # a game needs something besides its royal piece
        return None
    placed = {
        entry.get("piece")
        for entry in spec.get("setup", {}).get("placements", [])
        if isinstance(entry, dict)
    }
    if len(placed) == 1:
        candidates = [piece for piece in candidates if piece["id"] not in placed]
    piece = rng.choice(candidates)
    drop_piece(spec, str(piece["id"]))
    return f"removed the {piece.get('name', piece['id'])}"


def _clip_a_slider(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Shorten one slide.

    A move may also declare a *minimum* distance -- xiangqi's cannon and shogi's lance
    both do -- and a range under it is refused as *"min_range may not exceed range"*.
    The clip is drawn from the distances that leave a move rather than from (1, 2, 3)
    flat, which is the difference between an operator that is sometimes refused and one
    that is never wrong (Q85: 28 of 270 children).
    """
    movers = [(p, m) for p, m in _movers(spec, "slide") if m.get("range", 0) != 1]
    if not movers:
        return None
    piece, move = rng.choice(movers)
    floor = move.get("min_range", 1)
    lengths = [n for n in (1, 2, 3) if n >= (floor if isinstance(floor, int) else 1)]
    if not lengths:
        return None  # nothing shorter than this move's own minimum is a move
    move["range"] = rng.choice(lengths)
    settle_en_passant(spec)  # a promoting piece clipped to one square answers nothing
    return f"limited the {piece.get('name', piece['id'])} to {move['range']} squares"


def _free_a_slider(spec: dict[str, Any], rng: random.Random) -> str | None:
    movers = [(p, m) for p, m in _movers(spec, "slide") if m.get("range", 0) not in (0,)]
    if not movers:
        return None
    piece, move = rng.choice(movers)
    move.pop("range", None)
    move.pop("min_range", None)
    return f"let the {piece.get('name', piece['id'])} slide any distance"


def _leap_becomes_slide(spec: dict[str, Any], rng: random.Random) -> str | None:
    movers = _movers(spec, "leap")
    if not movers:
        return None
    piece, move = rng.choice(movers)
    move["kind"] = "slide"
    move["range"] = rng.choice((2, 3, 0))
    return f"turned the {piece.get('name', piece['id'])}'s leap into a slide"


def _slide_becomes_leap(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Turn one slide into a leap.

    A leap passes over nothing, so this is the other way the promoting piece can lose
    the two-square move en passant answers -- `_clip_a_slider` is the first.
    `settle_en_passant` takes the rule with it rather than handing `mutate` a child the
    validator refuses; on chess it is the pawn's own opening slide.
    """
    movers = _movers(spec, "slide")
    if not movers:
        return None
    piece, move = rng.choice(movers)
    move["kind"] = "leap"
    move.pop("range", None)
    move.pop("min_range", None)
    settle_en_passant(spec)
    return f"turned the {piece.get('name', piece['id'])}'s slide into a leap"


def _add_a_direction(spec: dict[str, Any], rng: random.Random) -> str | None:
    movers = _movers(spec)
    if not movers:
        return None
    piece, move = rng.choice(movers)
    vector = list(rng.choice(_SHORT_VECTORS))
    if vector in move.get("vectors", []):
        return None
    move.setdefault("vectors", []).append(vector)
    return f"gave the {piece.get('name', piece['id'])} a {vector[0]},{vector[1]} step"


def _remove_a_direction(spec: dict[str, Any], rng: random.Random) -> str | None:
    movers = [(p, m) for p, m in _movers(spec) if len(m.get("vectors", [])) > 1]
    if not movers:
        return None
    piece, move = rng.choice(movers)
    dropped = move["vectors"].pop(rng.randrange(len(move["vectors"])))
    return f"took a {dropped[0]},{dropped[1]} step off the {piece.get('name', piece['id'])}"


def _change_symmetry(spec: dict[str, Any], rng: random.Random) -> str | None:
    if spec.get("board", {}).get("lattice") == "hex":
        return None  # every symmetry but "none" is refused on a hex lattice
    movers = _movers(spec)
    if not movers:
        return None
    piece, move = rng.choice(movers)
    current = move.get("symmetry", "none")
    choices = [s for s in _SYMMETRIES if s != current]
    move["symmetry"] = rng.choice(choices)
    return f"changed how the {piece.get('name', piece['id'])}'s moves reflect ({current} to {move['symmetry']})"


def _across_a_board_bin(
    width: int, height: int, rng: random.Random
) -> tuple[int, int] | None:
    """The nearest board, one dimension away, that the grid would file differently.

    Measured before this existed, over five parents at 200 draws each: a plus-or-minus-
    one edit reaches exactly three shapes from any parent, and crosses a board bin only
    when the parent already sits within one row of an edge. Los Alamos at 36 squares
    crossed 86 times in 147 valid draws; gardner, chess, courier and shogi crossed zero.
    From an 8x8 the reachable boards are 8x7, 8x9 and 9x8 -- 64 plus or minus one row --
    and the nearest edge is sixteen squares away, so it is not close.

    The consequence was sharp: a model-free run could not reach a differently sized game
    however long it ran, while the archive it was filling reported coverage of a board
    axis as though it could. The grid is the thing being covered, so the operator moves
    in the grid's units, and the bin edges are read off ``AXES`` rather than repeated
    here -- two definitions of where a board becomes "large" is how they drift apart.
    """
    from chess_variant_sandbox.loop.grid import AXES

    bin_of = AXES[0].of
    current = bin_of({"board": {"width": width, "height": height}})
    options: list[tuple[int, int, int]] = []
    for size in range(_MIN_DIM, _MAX_DIM + 1):
        for (candidate_width, candidate_height), distance in (
            ((size, height), abs(size - width)),
            ((width, size), abs(size - height)),
        ):
            if distance == 0:
                continue
            shape = {"board": {"width": candidate_width, "height": candidate_height}}
            if bin_of(shape) != current:
                options.append((distance, candidate_width, candidate_height))
    if not options:
        return None
    # The *smallest* crossing step, so this stays a local move: one that lands in a
    # neighbouring cell rather than one that regenerates the board.
    nearest = min(distance for distance, _, _ in options)
    return rng.choice([(w, h) for distance, w, h in options if distance == nearest])


def _resize(spec: dict[str, Any], rng: random.Random) -> str | None:
    board = spec.get("board", {})
    if board.get("shape") == "hexagon":
        # A hexagon's size is its radius, and a new radius moves the cut under the army:
        # one size up, f1 is off the board. `resize_board` keeps the shape; this declines.
        return None
    width, height = int(board.get("width", 8)), int(board.get("height", 8))
    if rng.random() < 0.5:
        # Half the draws stay the smallest edit there is. Intra-cell moves are what
        # "local" means and are not the defect; being *unable* to leave was.
        if rng.random() < 0.5:
            width += rng.choice((-1, 1))
        else:
            height += rng.choice((-1, 1))
    else:
        crossing = _across_a_board_bin(width, height, rng)
        if crossing is None:
            return None  # already the only bin reachable from here
        width, height = crossing
    if not (_MIN_DIM <= width <= _MAX_DIM and _MIN_DIM <= height <= _MAX_DIM):
        return None
    resize_board(spec, width, height)
    return f"resized the board to {width}x{height}"


def _restrict_a_capture(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Take one enemy type off what a piece may capture, or give one back.

    The local move in the region schema 3 opened. A capture matrix is not reachable from
    movement edits at all -- no amount of changing how a piece moves changes what it may
    take -- so without this the free operator cannot enter the neighbourhood the model
    can now propose into, which is the same failure the board axis had.

    One type at a time, in both directions, because a step that could only tighten would
    walk every candidate to a game where nothing captures anything.
    """
    candidates = [p for p in pieces(spec) if p.get("id")]
    if len(candidates) < 2:
        return None  # a dominance relation needs something to dominate
    piece = rng.choice(candidates)
    others = sorted(
        str(other["id"]) for other in candidates if other["id"] != piece["id"]
    )
    if not others:
        return None

    current = piece.get("captures")
    if current is None:
        # Unrestricted: the first step is to forbid one type.
        forbidden = rng.choice(others)
        allowed = [pid for pid in others if pid != forbidden]
        restrict_captures(spec, str(piece["id"]), allowed)
        return f"{piece.get('name', piece['id'])} can no longer capture {forbidden}"

    allowed = [pid for pid in current if isinstance(pid, str)]
    missing = [pid for pid in others if pid not in allowed]
    if missing and (not allowed or rng.random() < 0.5):
        restored = rng.choice(missing)
        restrict_captures(spec, str(piece["id"]), sorted([*allowed, restored]))
        return f"{piece.get('name', piece['id'])} can capture {restored} again"
    if not allowed:
        return None
    forbidden = rng.choice(sorted(allowed))
    remaining = [pid for pid in allowed if pid != forbidden]
    restrict_captures(spec, str(piece["id"]), remaining)
    return f"{piece.get('name', piece['id'])} can no longer capture {forbidden}"


def _shift_promotion(spec: dict[str, Any], rng: random.Random) -> str | None:
    promotion = spec.get("rules", {}).get("promotion")
    if not isinstance(promotion, dict) or "squares" in promotion:
        return None  # a zone that is a shape has no rank to shift
    height = int(spec.get("board", {}).get("height", 8))
    current = promotion.get("rank")
    rank = (height - 1 if current in (None, "last") else int(current)) + rng.choice((-2, -1, 1))
    if not 1 <= rank <= height - 1:
        return None
    promotion_rank(spec, rank)
    return f"moved the promotion rank to {rank + 1}"


# A step's sentence names a rule the way the designer's form does (M7), never by its key:
# *changed the turn limit from 200 to 300 turns*, not *changed max plies from 200 to 300*.
# The two clocks count turns, one player's move each; repetition counts times.
_RULE_NAMES = {
    "en_passant": "en passant",
    "castling": "castling",
    "max_plies": "the turn limit",
    "no_progress_draw": "the no-progress rule",
    "repetition_draw": "the draw by repetition",
}


def _toggle_a_rule(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add or remove one of the optional rules, whichever way round it currently is."""
    rules = spec.setdefault("rules", {})
    key = rng.choice(("en_passant", "castling", "stalemate"))

    if key == "stalemate":
        current = rules.get("stalemate", "draw")
        rules["stalemate"] = rng.choice([v for v in ("draw", "loss", "win") if v != current])
        # The rule value is a noun and the sentence needs a verb: "loss" + "s" produced
        # "now losss the game" in a real run, which is the sort of thing that ends up in
        # a library listing for as long as the archive lives.
        verb = {"draw": "draws", "loss": "loses", "win": "wins"}[rules["stalemate"]]
        return f"a player with no legal move now {verb} the game"

    if key in rules:
        strip_rule(spec, key)
        return f"removed {_RULE_NAMES[key]}"

    if key == "en_passant":
        # Four things the validator refuses, asked here so the operator does not offer
        # a child that is thrown away: en passant is defined against *the promoting*
        # piece, so there has to be exactly one promotion rule, and that piece has to
        # have a move that travels two squares. The guard used to ask whether *any*
        # piece had a `range: 2` slide, which is a different question with the same
        # shape -- a spec whose rook double-steps and whose pawn does not passed it --
        # and `can_double_step` is imported rather than reimplemented so the two
        # readings cannot drift. The fourth is a turn of more than one move, where the
        # mover could answer their own double step; the other half of that pair is in
        # `transform.set_turn`, which drops the rule rather than refusing the edit.
        if turn_is_long(rules):
            return None
        promotions = spec.get("rules", {}).get("promotion")
        entries = promotions if isinstance(promotions, list) else [promotions]
        if len(entries) != 1 or not isinstance(entries[0], dict):
            return None
        promoter = next(
            (piece for piece in pieces(spec) if piece.get("id") == entries[0].get("piece")),
            None,
        )
        if promoter is None or not can_double_step(promoter):
            return None
        rules["en_passant"] = True
        return "added en passant"

    return None


def _limit_the_game(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Move one of the three clocks, and -- schema 10 -- cross the repetition rule's axis.

    ``repetition_draw`` is the one clock whose value is not only a size: since schema 10
    it may say ``"never"``, which removes the rule rather than lengthening it. That is a
    different *kind* of game (``distance.no_repetition_draw``), so an operator has to be
    able to reach it and to come back from it -- otherwise the axis exists in the schema
    and no proposal can cross it, which is the failure surface 8 of `new-move-primitive`
    exists to prevent.

    Scaling would not do it on its own. ``max(2, int(current * scale))`` is a floor at 2
    and there is no multiplier that reaches *never*, and in the other direction the
    sentinel is not an ``int`` at all, so the old body simply returned ``None`` and the
    axis was one-way by accident.

    **The scale has a ceiling as well as a floor, and it had only the floor** (Q84, Q85).
    ``max(2, ...)`` kept the bottom of every clock and nothing kept the top, so a parent
    at ``repetition_draw: 10`` doubled to 20 against a cap of 10 and the child was refused
    -- and a draw spent on a candidate that can never be played is indistinguishable from
    an operator that had nothing to do. No committed ruleset sits near a cap, which is why
    this was invisible: the clocks *drift* there, three doublings of chess's
    ``no_progress_draw`` at a time, so it bites a design run and never a test of one.
    The bounds are read from `schema.CLOCK_BOUNDS` rather than repeated, because a
    producer that guesses at the bound it must respect is the defect itself, one level up.

    A clamp that lands back on ``current`` is no edit at all, and says so by returning
    ``None`` rather than a summary claiming a change that did not happen. Note it clamps
    *to* the bound and never past it: ``never_forced``'s deliberate
    ``no_progress_draw == max_plies`` is a legal value one integer from a refusal, and an
    operator that treated the edge as forbidden would be the same bug with the sign
    flipped.
    """
    rules = spec.setdefault("rules", {})
    key = rng.choice(("max_plies", "no_progress_draw", "repetition_draw"))
    current = rules.get(key)
    if key == "repetition_draw":
        if current == REPETITION_NEVER:
            rules[key] = rng.choice((2, 3, 4, 5))
            settle_schema_version(spec)
            return f"gave the game a draw by repetition again, after {rules[key]}"
        if isinstance(current, int) and not isinstance(current, bool) and rng.random() < 0.5:
            rules[key] = REPETITION_NEVER
            settle_schema_version(spec)
            return "removed the draw by repetition"
    if not isinstance(current, int) or isinstance(current, bool):
        return None
    low, high = CLOCK_BOUNDS[key]
    scale = rng.choice((0.5, 0.75, 1.5, 2.0))
    scaled = min(high, max(low, int(current * scale)))
    if scaled == current:
        return None  # the clock is already against the bound this draw pushed it at
    rules[key] = scaled
    counted = "times" if key == "repetition_draw" else "turns"
    return f"changed {_RULE_NAMES[key]} from {current} to {rules[key]} {counted}"


def _reprice_a_piece(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Change what a piece is declared to be worth.

    A weak operator on purpose: the declared value is only a 25% prior on the value the
    evaluator derives from measured mobility, so this nudges how the agents treat a piece
    without changing what it can do. Kept because that nudge is sometimes exactly the
    difference between a piece being traded off and being played.
    """
    candidates = [p for p in _non_royal(spec) if isinstance(p.get("value"), (int, float))]
    if not candidates:
        return None
    piece = rng.choice(candidates)
    before = piece["value"]
    piece["value"] = max(1, round(float(before) * rng.choice((0.5, 0.75, 1.5, 2.0))))
    if piece["value"] == before:
        return None
    return f"repriced the {piece.get('name', piece['id'])} from {before} to {piece['value']}"


def _move_a_placement(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Slide one placed piece one file along, which changes the opening shape only."""
    placements = spec.get("setup", {}).get("placements", [])
    candidates = [p for p in placements if len(p.get("squares", [])) == 1]
    if not candidates:
        return None
    placement = rng.choice(candidates)
    square = placement["squares"][0]
    files = "abcdefghijklmnop"
    index = files.index(square[0]) + rng.choice((-1, 1))
    width = int(spec.get("board", {}).get("width", 8))
    if not 0 <= index < width:
        return None
    moved = f"{files[index]}{square[1:]}"
    if any(moved in other.get("squares", []) for other in placements):
        return None

    # A wall is a square nothing may stand on, and a hexagon's corners (schema 16) are not
    # squares at all; the walled Glinski hexagon spells its corners as walls.
    height = int(spec.get("board", {}).get("height", 8))
    orientation = spec.get("setup", {}).get("orientation", "mirror")
    walls = obstacle_squares(spec.get("rules", {}), width, height, orientation)
    walls |= absent_squares(spec.get("board"))
    if (int(square[1:]) - 1) * width + index in walls:
        return None

    placement["squares"] = [moved]
    return f"moved the {placement['piece']} from {square} to {moved}"


def _drop_royalty(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Change what winning means: capture the royal rather than mate it, or the reverse."""
    rules = spec.setdefault("rules", {})
    current = rules.get("win", "checkmate")
    if royal_id(spec) is None:
        return None
    rules["win"] = "royal_capture" if current == "checkmate" else "checkmate"
    return f"the game is now won by {rules['win'].replace('_', ' ')}"


def _hand_the_pieces_back(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Turn captures into reinforcements, or stop doing so.

    One of the two operators that could cross a cell boundary of the behaviour grid on
    its own **when it was written**, `win-by-arrival` being the other. Most mechanical
    edits are *intra*-cell by construction -- clipping a slider or repricing a piece
    changes the game without changing what kind of game it is -- so without these the
    grid could only ever be filled by the model operator. Most of the set can cross one
    now; the standing count is a measurement
    (`test_every_grid_axis_has_an_operator_that_can_cross_it`) rather than a number kept
    in a docstring.

    **It was the worst-affected member of the class Q85 was sent after** (found by the
    corpus sweep this task built, not by the brief). `drops` is a schema-2 rule and this
    was the one add-or-remove operator that never called `settle_schema_version`, so on
    every schema-1 parent -- chess, gardner, courier, shatranj, eighteen of the
    fifty-three -- *every* child was refused with `rules uses drops, added in schema 2`
    and the operator was a silent no-op. 108 of its 318 children in the sweep, 78 to the
    stamp and 30 to the `strike` clash below.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("drops"):
        if any(
            isinstance(piece, dict) and piece.get("moves") == [] for piece in pieces(spec)
        ):
            # Schema 10 (Q51). A piece with no moves is legal only while a `stock` puts
            # it in a hand, so popping the hand strands it and the validator refuses the
            # child by name -- every draw, on any spec with a stock. Asked here rather
            # than left to the validator for the module's own reason: what is checked
            # here is what would otherwise make *every* draw a rejection.
            return None
        rules.pop("drops", None)
        settle_schema_version(spec)
        return "captured pieces are gone for good again"
    if rules.get("strike") or rules.get("contacts") or uses_sweep(spec):
        return None  # a hand is refused beside a rule that takes several pieces at once
    if any(
        isinstance(npc, dict) and "immune" in npc for npc in rules.get("npcs") or ()
    ):
        # Schema 10 (Q78), and the same one-sided refusal `strike` had: a wild piece with
        # an immunity can reach a player's hand under `drops`, and capture permission is
        # a fact about a type rather than about who owns it, so the validator refuses the
        # pair. `_set_a_patrol` already declines to offer an immunity beside `drops`; this
        # is the other direction, which nobody wrote.
        return None
    if not capture_is_possible(spec):
        # A hand that fills from captures, in a game where nothing can be taken, is
        # refused as declaring nothing (inert-rule case 5) on every draw (Q110).
        return None
    rules["drops"] = {"from_captures": True}
    settle_schema_version(spec)
    return "captured pieces now change sides and may be dropped back onto any empty square"


def _win_by_arrival(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add or remove a win by getting the royal piece to the middle of the board.

    The other of that pair. The centre is chosen rather than a rank because it
    is the one target that means something on any board shape, and the squares are
    computed from the board rather than named, so this works on a 3x4 as readily as on a
    12x8.

    Same silence as `hand-the-pieces-back`, same cause: `reach` is schema 2 and this did
    not settle the stamp, so twelve schema-1 parents lost every child to `rules uses
    reach, added in schema 2`. 72 of 282 in the sweep.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("reach"):
        rules.pop("reach", None)
        settle_schema_version(spec)
        return "arriving somewhere no longer wins the game"

    # A royal that can stand, and the first declared one where it can: a `reach` about a
    # piece that never exists is refused (inert-rule case 13), and `royal_id` alone would
    # pick a royal declared first and placed nowhere on every draw (Q110).
    standing = pieces_that_can_stand(spec)
    royal = next(
        (
            str(piece["id"])
            for piece in pieces(spec)
            if piece.get("royal") and piece.get("id") in standing
        ),
        None,
    )
    if royal is None:
        return None
    width = int(spec.get("board", {}).get("width", 8))
    height = int(spec.get("board", {}).get("height", 8))
    files = "abcdefghijklmnop"
    middle_files = {(width - 1) // 2, width // 2}
    middle_ranks = {(height - 1) // 2, height // 2}
    squares = sorted(
        f"{files[file]}{rank + 1}" for file in middle_files for rank in middle_ranks
    )
    rules["reach"] = {"piece": royal, "squares": squares, "mirror": False, "result": "win"}
    settle_schema_version(spec)
    return f"the game is now also won by walking the {royal} to the centre"


# Ids a split can take. Single characters, because the schema allows one or two and a
# one-character id is what every committed ruleset uses; the vowels are avoided because
# they collide with nothing in particular and this only needs to find *an* unused one.
_SPARE_IDS = "XYZWVUTSGHJLM"


def _add_a_piece(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Split one piece type into two, and give the new one a different move.

    The gap this closes was found by measuring rather than by reading: over the five
    parents stages 13 and 15 used, *nothing* in the operator set could cross the grid's
    army axis, because every edit that touched the piece list removed a type and none
    added one. A free operator that can only ever shrink an army is exploring half of
    that dimension, and the coverage number said otherwise.

    Local by construction -- the new piece takes over some of the copies of an existing
    one, so no square is invented and the number of *pieces* does not change, only the
    number of *kinds*. Then one of its moves is perturbed, because two piece types that
    move alike are one piece type with two names: the child would be its parent spelled
    a second way, and measuring it would learn nothing.
    """
    candidates = [
        piece
        for piece in _non_royal(spec)
        if any(
            placement.get("piece") == piece.get("id")
            and len(placement.get("squares", [])) > 1
            for placement in spec.get("setup", {}).get("placements", [])
        )
    ]
    if not candidates:
        return None
    taken = {piece.get("id") for piece in pieces(spec)}
    spare = [pid for pid in _SPARE_IDS if pid not in taken]
    if not spare:
        return None

    source = rng.choice(candidates)
    new_id = spare[0]
    before = len(pieces(spec))
    split_piece(spec, source.get("id"), new_id)
    if len(pieces(spec)) == before:
        return None

    # ...and make it a different piece. A copy under a new name is not a new kind.
    clone = pieces(spec)[-1]
    moves = clone.get("moves") or []
    if not moves:
        return None
    move = rng.choice(moves)
    vector = list(rng.choice(_SHORT_VECTORS))
    if vector in move.get("vectors", []):
        if spec.get("board", {}).get("lattice") == "hex":
            return None  # every symmetry but "none" is refused on a hex lattice
        move["symmetry"] = rng.choice(
            [s for s in _SYMMETRIES if s != move.get("symmetry", "none")]
        )
        change = f"a {move['symmetry']} symmetry"
    else:
        move.setdefault("vectors", []).append(vector)
        change = f"a {vector[0]},{vector[1]} step"
    return (
        f"some of the {source.get('name', source['id'])}s are now a new piece "
        f"{new_id} with {change}"
    )


# What each scoring count is, in a sentence a player would read. One entry per member of
# `schema.SCORING_COUNTS`, including the ones `transform.set_scoring` will not choose --
# a table that only covers what today's producer emits is a table that raises the day the
# producer learns a new value, which is exactly how `points` got here.
_SCORE_SUBJECTS = {
    "territory": "ground held",
    "pieces": "pieces left standing",
    "material": "the value of what is left standing",
    "region": "of its own pieces standing in the scoring region",
    "points": "points collected by taking wild pieces",
}


def _score_the_clock(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add or remove a rule that decides the game on a count instead of drawing.

    One of the three operators stage 18 added, and the reason it exists is the reason
    stage 13 gave for `restrict-a-capture`: no amount of changing how a piece moves
    changes what happens when the clock runs out. Without these the grid's `score` and
    `ground` endings -- two of its six -- were reachable only by a model call, so the
    cheap half of the search was blind to a third of its own ending axis.

    **The prose is looked up per count, and the table has to hold every count**
    (Q57, Q85). It held three, written before schema 9 added `points`, and
    `transform.set_scoring` picks exactly that one for a spec that already has wild
    pieces -- so `[count]` raised `KeyError('points')` out of `operator.apply`, which
    `mutate` does not guard, and the generation died. Reproduced on `archive/wolves.json`
    at every seed. `mutate` is still unguarded on purpose: an operator that raises is a
    defect, and wrapping it would turn every future one into a draw that vanishes
    quietly, which is the whole class this operator was one of three examples of.
    `_SCORE_SUBJECTS` is held to `schema.SCORING_COUNTS` by a test rather than by care.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("scoring"):
        strip_rule(spec, "scoring")
        settle_schema_version(spec)
        return "a game that runs out of clock is a draw again"
    set_scoring(spec)
    count = rules["scoring"]["count"]
    return (
        "a game that runs out of clock now goes to whoever has more "
        f"{_SCORE_SUBJECTS[count]}"
    )


def _claim_the_ground(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add or remove a territory rule: moving onto a square claims it.

    The mode is drawn rather than fixed, because a claim you can take back and one you
    cannot are two different games and both are real -- `rps-v2` is permanent and
    `rps-v4` repaints, and the same designer built both. It does not change the cell
    either way, so this is diversity inside a bin rather than a second bin.

    Removing it also removes a `scoring` rule that counts territory, because a count
    with nothing to count is a spec the validator rejects by name -- and an operator
    that produced one would spend a draw on a candidate that could never be played.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("territory"):
        strip_rule(spec, "territory")
        if (rules.get("scoring") or {}).get("count") == "territory":
            strip_rule(spec, "scoring")
        settle_schema_version(spec)
        return "squares no longer remember who stood on them"
    mode = rng.choice(("permanent", "repaint"))
    set_territory(spec, mode=mode)
    win_at = rules["territory"]["win_at"]
    held = "keeps its colour" if mode == "permanent" else "can be taken back"
    return f"moving onto a square claims it -- it {held} -- and {win_at} squares wins"


def _lengthen_the_turn(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add or remove a turn longer than one move.

    Always with the opening ramp; see `spec.transform.set_turn` for the measurement that
    makes that non-optional. This is the one rule here whose *value* decides whether
    there is a rule at all, which is why removing it strips the key rather than writing
    `moves: 1` -- a turn of one move has several spellings and `complexity._RULE_ACTIVE`
    exists because of it.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("turn"):
        strip_rule(spec, "turn")
        settle_schema_version(spec)
        return "a turn is one move again"
    if rules.get("npcs"):
        # The other half of `_set_a_patrol`'s guard, which has always refused a patrol
        # beside a `turn` and was never refused back: a wild piece walks between the two
        # halves of a long turn and the validator says so by name. Every draw on
        # `archive/wolves` was refused before this.
        return None
    had_en_passant = bool(rules.get("en_passant"))
    set_turn(spec)
    sentence = (
        "a turn is now two moves, after a first turn of one so neither side is a move ahead"
    )
    if had_en_passant and not rules.get("en_passant"):
        # `set_turn` settles it: a player who moves twice could take their own pawn in
        # passing, so the rule cannot stand beside the longer turn.
        sentence += ", and en passant is gone with the one-move turn"
    return sentence


def _join_the_edges(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add or remove a seam joining two opposite edges of the board.

    The only operator that changes the board's *shape* rather than its size, and the
    cheapest large change in the set: it takes no piece away, adds no rule anybody has
    to be taught twice, and gives every long-range piece somewhere new to go.

    **On a hexagon (schema 16) it joins the one clear pair**, both ways: a pair whose two
    sides hold no piece of either army at the start, so the armies do not meet across the
    seam on move one. It is `set_wrap`'s *files, not ranks* measured again: on Glinski's
    opening, joining `n_s` puts White's knight on Black's king before anyone has moved,
    `ne_sw` gives 21 moves across it, and `nw_se`, the empty flanks, none. It declines
    when no pair is clear, and so does any vector the joined board would refuse.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("wrap"):
        strip_rule(spec, "wrap")
        settle_schema_version(spec)
        return "the board has edges again"
    if uses_sweep(spec):
        return None  # round a seam a sweep could reach one square by two paths
    if (spec.get("board") or {}).get("shape") == "hexagon":
        clear = _clear_pairs(spec)
        if not clear:
            return None
        pair = rng.choice(clear)
        set_wrap(spec, pair)
        if validate_spec(spec):
            # A vector longer than R + 1, or `no_doubled` across a folding pair: the
            # joined board refuses it, and every draw here would.
            strip_rule(spec, "wrap")
            return None
        return f"the {_PAIR_WORDS[pair]} sides are joined -- a piece that leaves by one comes back in by the other"
    set_wrap(spec, rng.choice(("files", "ranks")))
    axis = "left and right" if "files" in rules["wrap"] else "top and bottom"
    return f"the {axis} edges are joined -- a piece that runs off one continues from the other"


# A hexagon's pairs of sides, as the operator's summary names them.
_PAIR_WORDS = {
    "n_s": "top and bottom",
    "ne_sw": "upper-right and lower-left",
    "nw_se": "upper-left and lower-right",
}


def _clear_pairs(spec: dict[str, Any]) -> list[str]:
    """A hexagon's pairs of sides on which no piece of either army starts, in pair order.

    A side is the cells one step from leaving by it: on a hexagon of radius R, with
    ``(q, r)`` a cell's offset from the centre, the `n`/`s` sides are ``|r| = R``, the
    `ne`/`sw` sides ``|q + r| = R`` and the `nw`/`se` sides ``|q| = R``.
    """
    board = spec.get("board") or {}
    width = int(board.get("width", 0))
    height = int(board.get("height", 0))
    radius = (width - 1) // 2
    setup = spec.get("setup") or {}
    orientation = str(setup.get("orientation", "mirror"))
    standing: set[tuple[int, int]] = set()
    for placement in setup.get("placements", []):
        if not isinstance(placement, dict):
            continue
        for name in placement.get("squares", []):
            coords = parse_square(name, width, height) if isinstance(name, str) else None
            if coords is None:
                continue
            standing.add(coords)
            if setup.get("mirror", True):
                standing.add(reflect_square(*coords, width, height, orientation))
    offsets = [(file - radius, rank - radius) for file, rank in standing]
    on_side = {
        "n_s": lambda q, r: abs(r) == radius,
        "ne_sw": lambda q, r: abs(q + r) == radius,
        "nw_se": lambda q, r: abs(q) == radius,
    }
    return [
        pair for pair, touches in on_side.items()
        if not any(touches(q, r) for q, r in offsets)
    ]


def _wall_off_a_square(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add a wall, or take every wall away.

    The square is drawn from the middle of the board and from squares nothing stands
    on, which is not fastidiousness: a wall under the opening array is rejected by the
    validator, and a wall on the edge changes almost nothing about the lines that
    matter. Both halves are needed for the same reason every other add-or-remove
    operator has two -- an axis a candidate can only be pushed along in one direction
    is half an axis.

    **A placement is not the only thing that speaks for a square** (Q85). A drop region,
    a `reach` target, a scoring region and a wild piece's route are each refused by name
    over a wall, and reading only the opening array made this a total loss on
    `crazyhouse` and `crazyhouse-torus`, whose drop regions are the whole board. The four
    are asked in the same breath as the placements rather than left to the validator,
    because what is checked here is what would otherwise make *every* draw a rejection.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("obstacles"):
        strip_rule(spec, "obstacles")
        settle_schema_version(spec)
        return "the walls are gone; every square is a square again"

    board = spec.get("board", {})
    width = int(board.get("width", 8))
    height = int(board.get("height", 8))
    occupied = {
        square
        for placement in spec.get("setup", {}).get("placements", [])
        if isinstance(placement, dict)
        for square in placement.get("squares", [])
    }
    for region in (rules.get("drops") or {}).get("region", {}).values():
        occupied.update(region if isinstance(region, list) else ())
    goal = rules.get("reach") or {}
    occupied.update(goal.get("squares", []) or ())

    # The goal reaches the second player by its own reflection, which is not always the
    # rank image the free list below keeps clear.
    reflection = reach_reflection(goal)
    for name in goal.get("squares", []) or ():
        coords = parse_square(name, width, height) if isinstance(name, str) else None
        if reflection is not None and coords is not None:
            file, rank = reflect_square(*coords, width, height, reflection)
            occupied.add(f"{chr(ord('a') + file)}{rank + 1}")
    occupied.update((rules.get("scoring") or {}).get("region", []) or ())
    # Schema 14's zone as squares, refused under a wall by name. It went unasked until the
    # native hexagon (schema 16): the walled Glinski hexagon, the one spec with such a
    # zone, already has walls, so this operator only ever took them away there.
    promotion = rules.get("promotion")
    for entry in promotion if isinstance(promotion, list) else [promotion]:
        if isinstance(entry, dict):
            occupied.update(entry.get("squares", []) or ())
    for npc in rules.get("npcs") or ():
        occupied.update(npc.get("route", []) if isinstance(npc, dict) else ())
    # `set_obstacle` mirrors the wall, so the wall's image has to be free too, and the
    # image is the spec's own reflection (`schema.reflect_square`). It was the rank flip,
    # which is the image only under `orientation: "mirror"`: under `"rotate"` a wall could
    # land on a wild piece's route or its mirror, refused on every such draw (Q436). The
    # reflection is its own inverse, so a wall and its image missing every named square is
    # the same as both missing every named square and its image.
    orientation = str(spec.get("setup", {}).get("orientation", "mirror"))

    def named(file: int, rank: int) -> str:
        return f"{chr(ord('a') + file)}{rank + 1}"

    # Cells only: a hexagon's corners (schema 16) are not squares a wall could go on.
    absent = absent_squares(board)
    free = [
        named(file, rank)
        for rank in range(height)
        for file in range(width)
        if rank * width + file not in absent
        and named(file, rank) not in occupied
        and named(*reflect_square(file, rank, width, height, orientation)) not in occupied
    ]
    if not free:
        return None
    square = rng.choice(free)
    set_obstacle(spec, square)
    return f"{square} and its mirror are walls now -- no piece may stand there or move through"


def _soften_the_capture(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make a capture demote its victim instead of removing it, or put it back.

    The chain comes from the declared piece values, so this operator invents no numbers
    of its own; see `transform.set_strike`. It is the one edit in the set that changes
    no legal move at all -- every position has exactly the moves it had before, and only
    what they do is different.
    """
    rules = spec.setdefault("rules", {})
    if rules.get("strike"):
        placed = _place_the_survivor(spec, "strike", rng)
        if placed is not None:
            return placed
        strip_rule(spec, "strike")
        settle_schema_version(spec)
        return "a captured piece is simply removed again"
    if (
        rules.get("explode_on_capture")
        or rules.get("drops")
        or rules.get("npcs")
        or rules.get("contacts")
        or uses_sweep(spec)
    ):
        # Two rules cannot both say what happens to a taken piece. `npcs` is the third
        # member and was missing: `_set_a_patrol` has always refused a patrol beside a
        # strike, and the refusal was never symmetric, so every draw on `archive/wolves`
        # was thrown away. `contacts` is the fourth and arrived the same way (Q108) --
        # a strike beside a contact table is refused as *two spellings of one rule*.
        return None
    before = rules.get("strike")
    set_strike(spec, effect=rng.choice(("demote", "promote")), position="stay")
    if rules.get("strike") is before:
        return None  # nothing to build a chain from
    effect = rules["strike"]["effect"]
    return (
        f"taking a piece {effect}s it where it stands instead of removing it, so "
        "material is no longer conserved by trades"
    )


def _become_what_you_take(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make a capture change the *attacker* as well, or put it back.

    `_soften_the_capture`'s sibling, and it exists for the reason that operator exists
    one spelling over: schema 10 gave the DSL a contact table -- what **both** pieces in
    a capture become, keyed by the pair -- and **no operator could add, remove or alter
    one** (Q100's finding, audited over the whole bump by Q108 and true of three of the
    five features). A rule in the schema that no local edit can reach is an axis only the
    model operator can cross, which is surface 8 of `new-move-primitive` and the thing it
    exists to prevent.

    Separate from `soften-the-capture` rather than a third draw inside it, even though
    the validator refuses the two spellings together and only one can be on at a time.
    Folding them would halve how often either is reached, and they are not variants of
    one edit: a strike is a fact about the victim and this is a fact about the meeting.

    The chain comes from `transform.set_contacts`, which derives it from the declared
    values exactly as `set_strike` does, so this operator invents no numbers either. The
    three rules that also say what happens to a taken piece are asked here rather than
    left to the validator, on the module's own rule: what is checked here is what would
    otherwise make *every* draw a rejection.
    """
    from chess_variant_sandbox.spec.transform import set_contacts

    rules = spec.setdefault("rules", {})
    if rules.get("contacts"):
        placed = _place_the_survivor(spec, "contacts", rng)
        if placed is not None:
            return placed
        strip_rule(spec, "contacts")
        settle_schema_version(spec)
        return "taking a piece leaves the taker as it was again"
    if (
        rules.get("strike") or rules.get("explode_on_capture") or rules.get("drops")
        or rules.get("npcs") or uses_sweep(spec)
    ):
        return None  # two rules cannot both say what happens to a taken piece
    before = rules.get("contacts")
    set_contacts(spec, effect=rng.choice(("promote", "demote")))
    if rules.get("contacts") is before:
        return None  # one non-royal piece, so there is no rung to move to
    return (
        f"a piece that takes {_rung(spec, rules['contacts']['links'])} becomes that "
        "piece, so a capture changes what the taker is and not only what is left"
    )


def _place_the_survivor(spec: dict[str, Any], key: str, rng: random.Random) -> str | None:
    """Cross `position: "choose"` on a standing strike or contact table, or return ``None``.

    The half-draw both capture operators take before removing their rule, so the host rule
    survives the crossing (schema 12, Q20). Leaving puts the rule back under ``stay``,
    `set_strike`'s default. Entering needs a rule that keeps a struck piece alive under
    ``stay`` or ``swap``, and writes `transform.step_landing`, whose at most eight
    candidates sit far inside the 128-candidate bound. On a hex board that is the six
    neighbours under ``none``; the square board's ``rotate4`` there was refused (Q484).
    """
    from chess_variant_sandbox.spec.transform import clear_landing, keeps_a_survivor, set_landing

    if rng.random() >= 0.5:
        return None
    rules = spec.get("rules", {})
    rule = rules.get(key)
    if not isinstance(rule, dict):
        return None
    if rule.get("position") == "choose":
        clear_landing(spec, key, "stay")
        settle_schema_version(spec)
        return "a struck piece that survives stays where it stood again"
    if rule.get("position") not in ("stay", "swap") or not keeps_a_survivor(spec, key):
        return None
    set_landing(spec, key)
    settle_schema_version(spec)
    where = (
        "any empty cell next to where it stood"
        if spec.get("board", {}).get("lattice") == "hex"
        else "any empty square a king's step from where it stood"
    )
    return (
        f"the taker places a struck piece that survives on {where}; with none free, it "
        "cannot be taken"
    )


def _merge_your_own(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Let a piece move onto one of its own and merge with it, or take that back.

    Schema 11's `own: true` contact link (Q19b), which no other operator writes. The link
    comes from `transform.merge_links`, read off the declared values, so this invents no
    pairing of its own and every refusal the validator has for an own link is excluded
    there by construction.

    Leaving has two halves, drawn evenly. One drops `own` and keeps the link, so the table
    stands and the same pair now meets as enemies. The other deletes the link, which undoes
    an entry, and takes the table with its last link. A deletion can remove the only link
    that kept a target alive, so it settles `choose` as `transform.settle_landing` does.
    """
    from chess_variant_sandbox.spec.transform import merge_links, set_own_link, settle_landing

    rules = spec.setdefault("rules", {})
    contacts = rules.get("contacts")
    links = contacts.get("links") if isinstance(contacts, dict) else None
    own = [
        index
        for index, link in enumerate(links or [])
        if isinstance(link, dict) and link.get("own") is True
    ]
    if own and isinstance(links, list):
        index = rng.choice(own)
        link = links[index]
        pair = (link.get("mover"), link.get("onto"))
        enemy = {
            (other.get("mover"), other.get("onto"))
            for other in links
            if isinstance(other, dict) and other.get("own") is not True
        }
        if rng.random() < 0.5 and pair not in enemy:
            link.pop("own")
            settle_schema_version(spec)
            return f"the {pair[0]} no longer merges with its own {pair[1]}; it meets an enemy one instead"
        links.pop(index)
        if not links:
            strip_rule(spec, "contacts")
        settle_landing(spec)
        settle_schema_version(spec)
        return f"the {pair[0]} can no longer move onto its own {pair[1]}"

    if rules.get("must_capture"):
        return None  # whether moving onto your own piece is a capture is not settled
    if not rules.get("contacts") and (
        rules.get("strike") or rules.get("explode_on_capture") or rules.get("drops") or rules.get("npcs")
    ):
        return None  # two rules cannot both say what happens to a taken piece
    if uses_sweep(spec):
        return None  # a contact table is refused beside a sweep
    candidates = merge_links(spec)
    if not candidates:
        return None
    link = rng.choice(candidates)
    set_own_link(spec, link)
    settle_schema_version(spec)
    names = {str(piece.get("id")): piece.get("name", piece.get("id")) for piece in pieces(spec)}
    return (
        f"a {names[link['mover']]} may move onto its own {names[link['onto']]}, and the two "
        f"merge into a {names[link['mover_to']]}"
    )


def _rung(spec: dict[str, Any], links: list[dict[str, Any]]) -> str:
    """Which way a contact table's links point, in a sentence a player would read.

    Read off the declared values rather than off the ``effect`` word `set_contacts` was
    called with, and for that function's own reason: an army whose pieces are all worth
    the same is ordered by declaration, so *both* effects produce a table that raises
    nothing and neither "up" nor "down" would be true of it. This is the reading
    `distance._contact_features` takes for `strike_promotes`, one level up.
    """
    values = {
        str(piece.get("id")): float(piece.get("value", 1))
        for piece in pieces(spec)
        if isinstance(piece, dict)
    }
    deltas = {
        values.get(str(link.get("onto")), 0.0) - values.get(str(link.get("mover")), 0.0)
        for link in links
    }
    if all(delta > 0 for delta in deltas):
        return "a piece worth more than itself"
    if all(delta < 0 for delta in deltas):
        return "a piece worth less than itself"
    return "the piece next to it in the army's own order"


def _place_a_stone(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Give both players a stone they may place instead of moving, or take it away.

    The last of schema 10's five features to get an operator (Q108), and the only one
    that needed a piece that was not there: a hand fills from captures, so before
    `rules.drops.stock` the DSL could not say *a stone* at all -- nothing is ever
    captured that was never on the board. That is why `_add_a_piece` cannot reach it,
    and why this is a new operator rather than a branch on one: a split copies an
    existing type's moves and perturbs one, and a stone has no moves to perturb.

    It is also the one edit here that turns a chess into a *placement* game -- a move
    that puts a new piece on an empty square, available in every position -- which the
    behaviour grid has no other way of being handed. `reference/schema-showcase-coverage.md`
    rows 27--29 call the same thing "the only genuinely new design of the set".

    What it invents is a piece id and nothing else, and both halves of that are on the
    record: the id comes from `_SPARE_IDS` exactly as `_add_a_piece`'s does, and the
    supply is `STOCK_UNLIMITED` rather than a count, so there is no number to have
    chosen. See `transform.set_stone`.

    The three rules that refuse a hand are asked here rather than left to the validator,
    on the module's own rule: what is checked here is what would otherwise make *every*
    draw a rejection.
    """
    from chess_variant_sandbox.spec.schema import piece_has_stock
    from chess_variant_sandbox.spec.transform import set_stone

    rules = spec.setdefault("rules", {})
    stones = [
        piece
        for piece in pieces(spec)
        if piece.get("moves") == [] and piece_has_stock(rules, piece.get("id"))
    ]
    if stones:
        stone = rng.choice(stones)
        # `drop_piece` takes the stock entry with the piece, and the hand with the last
        # stock when nothing else fills it -- so this is one call and not three.
        drop_piece(spec, str(stone["id"]))
        settle_schema_version(spec)
        return f"the {stone.get('name', stone['id'])} is gone; every move is a move again"

    if rules.get("strike") or rules.get("contacts") or uses_sweep(spec):
        return None  # a hand is refused beside a rule that takes several pieces at once
    if any(isinstance(npc, dict) and "immune" in npc for npc in rules.get("npcs") or ()):
        return None  # an immune type that can reach a hand is refused by name
    taken = {str(piece.get("id")) for piece in pieces(spec)}
    spare = [pid for pid in _SPARE_IDS if pid not in taken]
    if not spare:
        return None
    set_stone(spec, spare[0])
    settle_schema_version(spec)
    return (
        "each player may place a stone on any empty square instead of moving, as often "
        "as they like; a stone never moves again and blocks what stands beside it"
    )


def _set_the_blast(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make a capture take its neighbours with it, stop doing so, or say whose captures do.

    Three branches rather than two, because schema 10 (Q18) split the rule into two
    questions and the operator set could reach neither: `explode_on_capture` had **no
    operator at all** -- a gap older than schema 10 and found by auditing it (Q108) --
    so `by` and `of` had no rule to narrow. `by` is the capturer types whose captures
    explode and `of` the victim types whose being taken does; both are
    `distance.explodes_selectively`, which is a different *kind* of game from atomic's
    all-or-nothing blast and was reachable only by a model call.

    One declared type, drawn the way `_set_a_patrol` draws its hunter: from what the spec
    already has, inventing nothing. Non-royal on both lists, and that is not fastidious --
    the capturer and the victim go regardless of `immune`, so `by: [<the royal>]` is a
    rule that says taking anything with your king loses you the game, and an operator
    whose whole output is a resignation is manufacturing noise rather than variety.

    The three rules that also say what happens to a taken piece are asked here rather
    than left to the validator, on the module's own rule: what is checked here is what
    would otherwise make *every* draw a rejection.
    """
    from chess_variant_sandbox.spec.transform import set_explosion

    rules = spec.setdefault("rules", {})
    explosion = rules.get("explode_on_capture")
    if isinstance(explosion, dict) and explosion:
        narrowed = [key for key in ("by", "of") if key in explosion]
        if narrowed and rng.random() < 0.5:
            for key in narrowed:
                explosion.pop(key)
            settle_schema_version(spec)
            return "every capture explodes again, whoever makes it and whatever it takes"
        if not narrowed and rng.random() < 0.5:
            candidates = _non_royal(spec)
            if not candidates:
                return None
            key = rng.choice(("by", "of"))
            piece = rng.choice(candidates)
            explosion[key] = [str(piece["id"])]
            settle_schema_version(spec)
            side = "made by" if key == "by" else "of"
            return f"only a capture {side} the {piece.get('name', piece['id'])} explodes"
        strip_rule(spec, "explode_on_capture")
        settle_schema_version(spec)
        return "a capture no longer takes its neighbours with it"

    if rules.get("strike") or rules.get("contacts") or rules.get("npcs") or uses_sweep(spec):
        return None  # two rules cannot both say what happens to a taken piece
    if not capture_is_possible(spec):
        return None  # a blast where nothing is taken is refused (inert-rule case 4, Q110)
    set_explosion(spec)
    settle_schema_version(spec)
    return (
        "a capture now destroys everything standing next to the square it happened on, "
        "so trades take the neighbourhood with them"
    )


def _turn_to_face(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Give a piece an orientation it may spend a move turning, or take one back.

    Which piece is the whole of the choice, and it is constrained rather than free: a
    facing on a piece whose moves are symmetric under the turn is a move that does
    nothing, which is a pass, and the compiler refuses it by name. So the candidates are
    the pieces whose declared directions actually move under the rotation -- worked out
    from `expand_symmetry`, which is the same expansion the compiler does, rather than
    from a list of piece names.

    Two axes rather than four when only the half turn changes anything: a piece that
    reads the same rays a quarter turn round would be refused at four and is fine at two.
    """
    from chess_variant_sandbox.spec.complexity import expand_symmetry
    from chess_variant_sandbox.spec.transform import clear_facing, set_facing

    pieces = [p for p in spec.get("pieces", []) if isinstance(p, dict)]
    facing = [p for p in pieces if isinstance(p.get("facing"), dict)]
    if facing:
        clear_facing(spec, rng.choice(facing)["id"])
        settle_schema_version(spec)
        return "the piece that could turn no longer has a direction to turn"

    def directions(piece: dict[str, Any]) -> frozenset[tuple[int, int]]:
        found: set[tuple[int, int]] = set()
        for move in piece.get("moves", []) or []:
            symmetry = str(move.get("symmetry", "none"))
            for vector in move.get("vectors", []) or []:
                found |= expand_symmetry((vector[0], vector[1]), symmetry)
        return frozenset(found)

    def turned(vectors: frozenset[tuple[int, int]], quarters: int) -> frozenset:
        out = set()
        for dx, dy in vectors:
            for _ in range(quarters):
                dx, dy = dy, -dx
            out.add((dx, dy))
        return frozenset(out)

    # A hex lattice has no quarter turn, so there a piece may only face two ways.
    quarter_turns = spec.get("board", {}).get("lattice") != "hex"
    candidates: list[tuple[dict[str, Any], int]] = []
    for piece in pieces:
        if piece.get("royal") or piece.get("facing"):
            continue  # a royal may not turn; the validator says so by name
        vectors = directions(piece)
        if not vectors:
            continue
        if quarter_turns and turned(vectors, 1) != vectors:
            candidates.append((piece, 4))
        elif turned(vectors, 2) != vectors:
            candidates.append((piece, 2))
    if not candidates:
        return None
    piece, axes = rng.choice(candidates)
    set_facing(spec, piece["id"], axes=axes)
    ways = "four ways" if axes == 4 else "forwards or backwards"
    return (
        f"{piece['id']} now points {ways} and reads its moves turned that way; it may "
        "spend a move turning instead of moving"
    )


def _hunt_the_wild(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Give the wild pieces already on the board an immunity, or take one away.

    Half of `_set_a_patrol`'s first branch since Q108, and it is the half that was
    missing. Schema 10's `npcs[].immune` -- *only these types may take this wild piece* --
    could be **entered** (the operator draws one when it looses a patrol) and could not be
    **left**: the only edit that touched a spec with wild pieces already on it deleted
    every one of them, which crosses `npcs` and takes `immune` down with it rather than
    crossing `immune`'s own axis. `distance.npcs_are_hard_to_kill` says the two are
    different kinds of game, so half an axis is a design run that can drift onto it and
    never off.

    ``None`` rather than a summary means *not this draw*, and `_set_a_patrol` falls
    through to clearing the wild pieces -- so the operator keeps both of the halves it
    already had and gains a third rather than trading one away.

    **Capture permission is a fact about a type**, and the validator refuses two wild
    pieces of one type with different immunities by name, so the edit is applied to every
    wild piece of the type it picks and not to one entry. Not offered beside `drops`,
    where an immune type that reaches a hand is refused; that is the guard the add branch
    already has, asked here too.
    """
    npcs = [npc for npc in spec.get("rules", {}).get("npcs") or () if isinstance(npc, dict)]
    if not npcs or rng.random() < 0.5:
        return None
    kind = rng.choice(sorted({str(npc.get("piece")) for npc in npcs}))
    family = [npc for npc in npcs if str(npc.get("piece")) == kind]
    if any("immune" in npc for npc in family):
        for npc in family:
            npc.pop("immune", None)
        settle_schema_version(spec)
        return f"anything may take a wild {kind} again, not only the one type that could"
    if spec.get("rules", {}).get("drops"):
        return None  # an immune type that can reach a hand is refused by name
    hunters = [piece for piece in pieces(spec) if str(piece.get("id")) != kind]
    if not hunters:
        return None
    hunter = str(rng.choice(hunters)["id"])
    for npc in family:
        npc["immune"] = [hunter]
    settle_schema_version(spec)
    return f"only {hunter} may take a wild {kind} now; to anything else it is a hazard"


def _set_a_patrol(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Loose a wild piece on a small loop, or take every wild piece away.

    The loop is four squares in a block -- the smallest route that is a walk rather than
    a post -- drawn from blocks whose eight squares (the loop and its mirror image) are
    all free of the opening array and of every wall, and do not meet each other. The
    type is any non-royal piece the spec already has, so the operator invents no piece
    and no number: the bounty is the type's declared value, which is `set_npc`'s default.
    Both halves, for the reason every add-or-remove operator here has two.

    Refused by the validator rather than checked here wherever it can be: a route across
    a wall, under a placement or beside a strike is a rejected candidate, and drawing
    again is cheaper than being clever. What is checked here is only what would make
    *every* draw a rejection.
    """
    from chess_variant_sandbox.spec.schema import (
        absent_squares,
        obstacle_squares,
        reflect_square,
    )
    from chess_variant_sandbox.spec.transform import clear_npcs, set_npc

    rules = spec.setdefault("rules", {})
    if rules.get("npcs"):
        hunted = _hunt_the_wild(spec, rng)
        if hunted is not None:
            return hunted
        clear_npcs(spec)
        settle_schema_version(spec)
        return "the wild pieces are gone; every piece on the board belongs to somebody"
    if (
        rules.get("strike")
        or rules.get("explode_on_capture")
        or rules.get("turn")
        or rules.get("contacts")
        or uses_sweep(spec)
    ):
        return None  # the validator refuses each of these beside a wild piece

    candidates = _non_royal(spec)
    if not candidates:
        return None
    board = spec.get("board", {})
    width = int(board.get("width", 8))
    height = int(board.get("height", 8))
    orientation = str(spec.get("setup", {}).get("orientation", "mirror"))
    mirrored = bool(spec.get("setup", {}).get("mirror", True))

    def index(name: str) -> int:
        return (int(name[1:]) - 1) * width + (ord(name[0]) - ord("a"))

    # A hexagon's corners are not cells (schema 16), and a route there is refused by name.
    taken: set[int] = set(obstacle_squares(rules, width, height, orientation))
    taken |= absent_squares(board)
    for placement in spec.get("setup", {}).get("placements", []):
        if not isinstance(placement, dict):
            continue
        for name in placement.get("squares", []):
            square = index(name)
            taken.add(square)
            if mirrored:
                f, r = reflect_square(square % width, square // width, width, height, orientation)
                taken.add(r * width + f)

    def name_of(square: int) -> str:
        return f"{chr(ord('a') + square % width)}{square // width + 1}"

    loops: list[list[int]] = []
    for rank in range(height - 1):
        for file in range(width - 1):
            loop = [
                rank * width + file,
                (rank + 1) * width + file,
                (rank + 1) * width + file + 1,
                rank * width + file + 1,
            ]
            image = [
                r * width + f
                for f, r in (
                    reflect_square(s % width, s // width, width, height, orientation)
                    for s in loop
                )
            ]
            squares = set(loop) | set(image)
            if len(squares) == 8 and not squares & taken:
                loops.append(loop)
    if not loops:
        return None
    loop = rng.choice(loops)
    piece = rng.choice(candidates)
    # Schema 10 (Q78). Half the draws give the wild piece an immunity, because "only a
    # knight may take it" is a different *kind* of game from "anything may take it" --
    # `distance.npcs_are_hard_to_kill` says so -- and an axis no operator can cross is an
    # axis only the model can reach. The hunter is one other declared type, chosen the
    # way the wild type is: from what the spec already has, inventing nothing. Refused
    # beside `drops` by the validator, so the draw is simply not offered there.
    hunters = [p for p in pieces(spec) if p.get("id") != piece["id"]]
    immune = (
        [rng.choice(hunters)["id"]]
        if hunters and not rules.get("drops") and rng.random() < 0.5
        else None
    )
    set_npc(spec, piece["id"], [name_of(s) for s in loop], immune=immune)
    hunted = (
        f"; only {immune[0]} may take one" if immune else ""
    )
    return (
        f"a wild {piece['id']} walks {', '.join(name_of(s) for s in loop)} after every "
        "move and eats what it walks into, and so does its mirror image; taking one is "
        f"worth {piece.get('value', 1)} points{hunted}"
    )


def _promote_what_arrives(spec: dict[str, Any], rng: random.Random) -> str | None:
    """A piece that arrives in its promotion zone promotes at once, or stops doing so.

    Both halves, for the reason every add-or-remove operator here has two. The rule is
    one boolean and what the piece becomes is already written in the promotion rule, so
    the operator invents nothing. What is checked here is only what would make *every*
    draw a rejection: no promotion rule, or no way for a piece to enter play without
    moving.

    **And "every draw" is every draw, because this operator has no draw** (Q85). It reads
    no random number: given a spec there is exactly one answer, so a refusal here is not
    a candidate to draw again for, it is the operator being unable to apply -- and
    `crazyhouse` and the three `hierarchy-rise` games lost all six of six that way, a
    drop region and a strike chain that between them never put a promoting piece inside
    its zone. The old docstring called that *"the validator's to refuse, by name, and
    drawing again is cheaper than being clever"*, which is the right argument for the
    operators that do draw and no argument at all for this one.

    So the question is put to the validator on a copy rather than answered a second time
    here. `_check_entry_promotion_coherence` is a walk over drop regions, strike chains
    and promotion zones, and a paraphrase of it in this module would be a second reading
    of the DSL -- which is the one thing `engine/compile.py` owns. It costs the
    validation `mutate` was about to run anyway.
    """
    from chess_variant_sandbox.spec.transform import set_promote_on_entry

    rules = spec.setdefault("rules", {})
    if rules.get("promote_on_entry"):
        strip_rule(spec, "promote_on_entry")
        settle_schema_version(spec)
        return "a piece that arrives in its promotion zone now stands there unpromoted"
    if not rules.get("promotion"):
        return None
    strike = rules.get("strike")
    contacts = rules.get("contacts")
    survives = (
        isinstance(strike, dict)
        and strike.get("effect", "capture") != "capture"
        and bool(strike.get("chain"))
    ) or (
        # Schema 10 (Q19a) puts a piece into play on both of a contact's columns, and
        # `_check_entry_promotion_coherence` reads both -- so the cheap pre-check has to
        # as well, or the rule is unreachable on every contact-table game for a reason
        # the validator does not hold. The trial below is still what decides.
        isinstance(contacts, dict) and bool(contacts.get("links"))
    )
    if not rules.get("drops") and not survives:
        return None
    trial = copy.deepcopy(spec)
    set_promote_on_entry(trial)
    if validate_spec(trial):
        return None  # nothing in this game can enter a promotion zone without moving
    set_promote_on_entry(spec)
    return (
        "a piece dropped or struck into its promotion zone promotes at once, to the "
        "first piece its promotion rule lists"
    )


def _promote_on_leaving(spec: dict[str, Any], rng: random.Random) -> str | None:
    """One piece also promotes on a move out of its zone, or stops doing so.

    Schema 15's flag (Q452), which no other operator writes. Both halves, one entry at a
    time: the flag belongs to a promotion entry, so taking it off one piece and putting it
    on another are two edits.

    **Only a piece that can leave its zone is offered.** On any other the flag compiles
    away (`compile._settle_on_leaving`) and the child would be its parent's game under a
    new hash -- a chess pawn's zone is the rank it cannot move from. Which pieces can
    leave is a question about the movement tables, so the compiler answers it
    (`compile.pieces_that_can_leave_their_zone`) rather than a second reading here; that
    is `_promote_what_arrives`' reason for asking the validator.
    """
    from chess_variant_sandbox.engine.compile import pieces_that_can_leave_their_zone

    promotion = (spec.get("rules") or {}).get("promotion")
    listed = promotion if isinstance(promotion, list) else [promotion]
    entries = [entry for entry in listed if isinstance(entry, dict)]
    names = {
        piece.get("id"): piece.get("name", piece.get("id")) for piece in spec.get("pieces", [])
    }
    flagged = [entry for entry in entries if entry.get("on_leaving") is True]
    if flagged:
        entry = rng.choice(flagged)
        entry.pop("on_leaving")
        settle_schema_version(spec)
        return f"the {names.get(entry.get('piece'))} promotes only on a move into its zone again"
    if not entries:
        return None
    live = sorted(pieces_that_can_leave_their_zone(spec))
    if not live:
        return None  # no promoting piece here can move out of its zone
    piece = rng.choice(live)
    next(entry for entry in entries if entry.get("piece") == piece)["on_leaving"] = True
    settle_schema_version(spec)
    return (
        f"the {names.get(piece)} may also promote on a move out of its promotion zone, "
        "as in shogi"
    )


def _sweep_a_slider(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Turn one slide into a sweep, or a sweep back into a slide.

    Schema 13's move kind (Q21), which no other operator writes. The descriptor keeps its
    vectors, symmetry, range and condition, so the piece reaches the same squares and only
    what stands between stops mattering. A promoting piece is never offered, and neither
    is a spec holding a rule the validator refuses beside a sweep: both would make every
    draw a rejection.
    """
    sweeps = _movers(spec, "sweep")
    if sweeps:
        piece, move = rng.choice(sweeps)
        move["kind"] = "slide"
        settle_schema_version(spec)
        return (
            f"the {piece.get('name', piece['id'])} slides again and stops at the first "
            "piece in its way"
        )
    rules = spec.get("rules", {})
    if sweep_conflicts(rules):
        return None
    promoting = promoting_piece_ids(rules)
    movers = [(p, m) for p, m in _movers(spec, "slide") if p.get("id") not in promoting]
    if not movers:
        return None
    piece, move = rng.choice(movers)
    move["kind"] = "sweep"
    settle_schema_version(spec)
    return (
        f"the {piece.get('name', piece['id'])} sweeps: it jumps to any square on its lines "
        "and removes every piece it passes over, its own included"
    )


OPERATORS: tuple[Operator, ...] = (
    Operator("wall-off-a-square", _wall_off_a_square),
    Operator("soften-the-capture", _soften_the_capture),
    Operator("become-what-you-take", _become_what_you_take),
    Operator("merge-your-own", _merge_your_own),
    Operator("set-the-blast", _set_the_blast),
    Operator("place-a-stone", _place_a_stone),
    Operator("set-a-patrol", _set_a_patrol),
    Operator("promote-what-arrives", _promote_what_arrives),
    Operator("promote-on-leaving", _promote_on_leaving),
    Operator("remove-a-piece", _remove_a_piece),
    Operator("add-a-piece", _add_a_piece),
    Operator("clip-a-slider", _clip_a_slider),
    Operator("free-a-slider", _free_a_slider),
    Operator("leap-becomes-slide", _leap_becomes_slide),
    Operator("slide-becomes-leap", _slide_becomes_leap),
    Operator("add-a-direction", _add_a_direction),
    Operator("remove-a-direction", _remove_a_direction),
    Operator("change-symmetry", _change_symmetry),
    Operator("resize-the-board", _resize),
    Operator("restrict-a-capture", _restrict_a_capture),
    Operator("shift-promotion", _shift_promotion),
    Operator("toggle-a-rule", _toggle_a_rule),
    Operator("limit-the-game", _limit_the_game),
    Operator("reprice-a-piece", _reprice_a_piece),
    Operator("move-a-placement", _move_a_placement),
    Operator("change-the-win-condition", _drop_royalty),
    Operator("hand-the-pieces-back", _hand_the_pieces_back),
    Operator("win-by-arrival", _win_by_arrival),
    Operator("score-the-clock", _score_the_clock),
    Operator("claim-the-ground", _claim_the_ground),
    Operator("lengthen-the-turn", _lengthen_the_turn),
    Operator("join-the-edges", _join_the_edges),
    Operator("turn-to-face", _turn_to_face),
    Operator("sweep-a-slider", _sweep_a_slider),
)


# The one refusal that is `mutate`'s own rather than the validator's.
SAME_GAME = "the child is its parent's game under another spelling"


def refusal(parent: dict[str, Any], child: dict[str, Any]) -> list[str]:
    """Why `mutate` would discard ``child`` as a child of ``parent``, or nothing.

    The validator's errors first, then whether the child is the parent's game again. An
    equal hash is the cheap half of that question. `relate.identity_key` is the rest: a
    symmetry the vectors already have, or a step the symmetry already reaches, moves the
    hash and compiles to the parent's tables, and a child like that would be given a
    tournament of its own. On the hash alone, 119 of 420 `change-symmetry` children and
    70 of 415 `add-a-direction` children of the corpus sweep went through (Q327, six
    seeds), and every one of them compiles to its parent's tables.

    One function so the corpus sweep in `tests/test_operators.py` counts exactly what a
    design run throws away.
    """
    errors = validate_spec(child)
    if errors:
        return errors
    if spec_hash(child) == spec_hash(parent) or identity_key(child) == identity_key(parent):
        return [SAME_GAME]
    return []


def mutate(
    spec: dict[str, Any],
    rng: random.Random,
    operators: tuple[Operator, ...] = OPERATORS,
    attempts: int = ATTEMPTS,
    targets: Iterable[Sequence[int]] | None = None,
    avoid: set[str] | None = None,
) -> Mutation | None:
    """One valid, different child of ``spec``, or ``None`` if none was found.

    ``None`` is a real answer rather than an error: a spec with one piece and no rules
    has very few local edits available, and the caller's response is to ask the model
    instead, which is the operator that can add something that was not there.

    ``targets`` are cells of the behaviour grid worth reaching -- usually every empty
    cell one step from the parent's. A mechanical operator cannot be *told* to make a
    game of a particular kind the way a model can, but it can be drawn from repeatedly
    and asked whether the result landed in any of them, which costs a validation each and
    is free next to a tournament.

    ``avoid`` is the set of mechanic hashes already evaluated. A local edit can easily
    reproduce a game the archive already holds -- measured on a ten-generation run, one
    edit was rediscovered five times, each costing a generation to the duplicate check.
    Drawing again costs a validation; a duplicate costs a whole generation.

    A *set* of targets rather than a single cell, and that distinction was measured: most edits here
    are intra-cell by construction -- clipping a slider changes the game without changing
    what kind of game it is -- and when this was measured only three operators crossed an
    axis at all (nineteen of thirty-one do since Q108, and the figure is measured rather
    than kept here). Aiming at
    one named cell landed roughly once in ten generations; aiming at "anywhere new
    nearby" is the same question the archive is actually asking, and it lands far more
    often. Every valid child is kept as the fallback, so aiming never *costs* a
    generation: it either reaches new ground or returns the local edit it would have
    returned anyway.
    """
    wanted = {tuple(cell) for cell in targets} if targets else set()
    seen = set(avoid or ())
    fallback: Mutation | None = None

    for _ in range(attempts):
        operator = rng.choice(operators)
        child = copy.deepcopy(spec)
        summary = operator.apply(child, rng)
        if summary is None:
            continue
        if refusal(spec, child):
            # Invalid, or the parent's game again: a different spec is not a different
            # game, and the archive would reject it as a duplicate anyway.
            continue
        if spec_hash(child) in seen:
            continue
        child["name"] = _name_for(spec, summary)
        child["description"] = (
            f"{spec.get('name', 'a game')} with one change: {summary}."
        )
        # The parent's provenance does not transfer. A mechanical edit of gardner is not
        # Martin Gardner's game, and a spec that says it is would be a fabricated
        # attribution travelling with the file forever -- worse than no provenance at
        # all. Where this candidate came from is the archive's business, and the archive
        # records it. (Outside MECHANIC_KEYS, so removing it moves no hash.)
        child.pop("provenance", None)
        mutation = Mutation(spec=child, operator=operator.name, summary=summary)
        if not wanted or descriptor(child) in wanted:
            return mutation
        fallback = fallback or mutation
    return fallback


def _name_for(parent: dict[str, Any], summary: str) -> str:
    """A name that says what the child is, since nothing here writes prose.

    The hash ignores names, so this is for the reader and for the library listing, and
    it has to stay short enough to be one.
    """
    base = str(parent.get("name", "game")).split(" (")[0]
    return f"{base} ({summary})"[:80]
