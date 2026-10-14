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
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.relate import identity_key
from chess_variant_sandbox.spec.schema import (
    CLOCK_BOUNDS,
    FACING_NAMES,
    HEX_SYMMETRIES,
    REPETITION_NEVER,
    SQUARE_SYMMETRIES,
    absent_squares,
    can_double_step,
    castling_spans,
    obstacle_squares,
    opening_counts,
    parse_square,
    portal_groups,
    portal_open_ends,
    portal_piece_ids,
    promoting_piece_ids,
    reach_reflection,
    reflect_square,
    strike_is_live,
    sweep_conflicts,
    turn_is_long,
    uses_sweep,
)
from chess_variant_sandbox.spec.transform import (
    capture_is_possible,
    drop_piece,
    empties_a_filter,
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
    "contacts": "the table of what a capture turns pieces into",
    "keeping": "the links keeping a capturer as it was",
}


def _gone_words(key: str) -> str:
    """One key `settle_inert_rules` returned, in a player's words."""
    if key.startswith("promotion:"):
        return f"the {key.split(':', 1)[1]}'s promotion"
    if key.startswith("immune:"):
        return f"the wild {key.split(':', 1)[1]}'s immunity"
    if key.startswith("carries:"):
        return f"the filter on the portal at {key.split(':', 1)[1]}"
    return _GONE_WORDS.get(key, key.replace("_", " "))


def _has_portals(spec: dict[str, Any]) -> bool:
    """Schema 19: whether the spec has portals, beside which the validator refuses a
    strike, a contact table, an explosion, a colouring, a sweep and a wild piece -- so an
    operator that would add one of those is not offered there."""
    return bool((spec.get("rules") or {}).get("portals"))


def _portal_squares(spec: dict[str, Any]) -> set[str]:
    """The squares a portal end starts on, which no placement or wall may take."""
    return {
        str(end["at"])
        for group in portal_groups(spec.get("rules") or {})
        for end in group["ends"]
    }


def _non_royal(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [piece for piece in pieces(spec) if not piece.get("royal")]


def _movers(spec: dict[str, Any], kind: str | None = None) -> list[tuple[dict, dict]]:
    """Every (piece, move) pair, optionally of one movement kind.

    A portal type's moves are left out (schema 19): they must stay `move` mode and read the
    same for both players, which a direction or a symmetry drawn for an army's piece
    breaks (Q772a found `add-a-direction` and `change-symmetry` refused on every portal
    showcase). `open-a-portal` is the operator that edits portals.
    """
    portal_types = portal_piece_ids(spec.get("rules"))
    return [
        (piece, move)
        for piece in pieces(spec)
        if piece.get("id") not in portal_types
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
    # Schema 21 (Q849): a portal end only this piece may go in by would let nobody in
    # without it, which no filter can say, and ``drop_piece`` would open it to everyone.
    candidates = [piece for piece in candidates if not empties_a_filter(spec, piece["id"])]
    if not candidates:
        return None
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


def _no_leg_for(move: dict[str, Any], vector: list[int]) -> bool:
    """Whether ``vector`` is a one-square step on a ``lame`` move, which the validator
    refuses on admission (Q884): a lame leap's blockable square is the one it passes over,
    and a one-square step passes over none."""
    return move.get("kind") == "lame" and max(abs(vector[0]), abs(vector[1])) == 1


def _add_a_direction(spec: dict[str, Any], rng: random.Random) -> str | None:
    movers = _movers(spec)
    if not movers:
        return None
    piece, move = rng.choice(movers)
    vector = list(rng.choice(_SHORT_VECTORS))
    if vector in move.get("vectors", []) or _no_leg_for(move, vector):
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


def _lattice_symmetries(spec: dict[str, Any]) -> tuple[str, ...]:
    """The symmetries this spec's lattice accepts: the hexagon's two on a hex board (schema
    18b), the square's three elsewhere, and ``none`` on both."""
    hexy = (spec.get("board") or {}).get("lattice") == "hex"
    return HEX_SYMMETRIES if hexy else SQUARE_SYMMETRIES


def _set_symmetry(spec: dict[str, Any], move: dict[str, Any], rng: random.Random) -> str:
    """Give ``move`` another of its lattice's symmetries, and stamp the version it needs.

    The hexagon's two are schema 18's, so a hex child is settled, both ways: turned, it
    needs 18, and written back under ``none`` it is the game its parent's stamp named. A
    square child is left as it was, since the square's three are as old as the schema.
    """
    current = move.get("symmetry", "none")
    move["symmetry"] = rng.choice([s for s in _lattice_symmetries(spec) if s != current])
    if _lattice_symmetries(spec) is HEX_SYMMETRIES:
        settle_schema_version(spec)
    return str(current)


def _change_symmetry(spec: dict[str, Any], rng: random.Random) -> str | None:
    movers = _movers(spec)
    if not movers:
        return None
    if _lattice_symmetries(spec) is HEX_SYMMETRIES:
        return _turn_a_hex_move(spec, movers, rng)
    piece, move = rng.choice(movers)
    current = _set_symmetry(spec, move, rng)
    return f"changed how the {piece.get('name', piece['id'])}'s moves reflect ({current} to {move['symmetry']})"


def _turn_a_hex_move(
    spec: dict[str, Any], movers: list[tuple[dict, dict]], rng: random.Random
) -> str | None:
    """`change-symmetry` on a hex board: one of the hexagon's turns that moves the move.

    A hex army lists its moves whole -- Glinski's rook is its six steps under ``none`` --
    so most of the hexagon's turns are the same move spelled shorter, a child `refusal`
    throws away as the parent's game. Drawing only among the turns that change where a
    move goes spends the draw on a game. The square board keeps its draw as it was.
    """

    def directions(move: dict[str, Any], symmetry: str) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for vector in move.get("vectors") or []:
            out |= expand_symmetry((vector[0], vector[1]), symmetry)
        return out

    choices = [
        (piece, move, symmetry)
        for piece, move in movers
        for symmetry in HEX_SYMMETRIES
        if symmetry != move.get("symmetry", "none")
        and directions(move, symmetry) != directions(move, move.get("symmetry", "none"))
    ]
    if not choices:
        return None
    piece, move, symmetry = rng.choice(choices)
    current = move.get("symmetry", "none")
    move["symmetry"] = symmetry
    settle_schema_version(spec)
    return f"changed how the {piece.get('name', piece['id'])}'s moves reflect ({current} to {symmetry})"


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
    # Schema 19: a portal neither captures nor is captured, so it is no party to this.
    portal = portal_piece_ids(spec.get("rules"))
    candidates = [p for p in pieces(spec) if p.get("id") and p.get("id") not in portal]
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
    if not isinstance(promotion, dict) or "squares" in promotion or "ranks" in promotion:
        # A zone that is a shape has no rank to shift, and a band of ranks is not one
        # rank: writing `rank` beside `ranks` is refused (`archive/grand`, Q783, was the
        # first archived band and the first spec the sweep found this on).
        return None
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
    # A portal type is nobody's material and must be worth 0 (schema 19).
    portal_types = portal_piece_ids(spec.get("rules"))
    candidates = [
        p for p in _non_royal(spec)
        if isinstance(p.get("value"), (int, float)) and p.get("id") not in portal_types
    ]
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
    if moved in _portal_squares(spec):
        return None  # schema 19: a portal end stands there

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
        if isinstance(rules["drops"], dict) and rules["drops"].get("placement") is True:
            # Schema 18d (Q784). The hand holds an army to place, and popping it leaves
            # that army (and often the royal) nowhere; `place-the-army` puts it back on
            # the board first.
            return None
        rules.pop("drops", None)
        settle_schema_version(spec)
        return "captured pieces are gone for good again"
    if rules.get("strike") or rules.get("contacts") or uses_sweep(spec):
        return None  # a hand is refused beside a rule that takes several pieces at once
    promotion = rules.get("promotion")
    if any(
        isinstance(entry, dict) and entry.get("only_lost") is True
        for entry in (promotion if isinstance(promotion, list) else [promotion])
    ):
        return None  # schema 18c: a hand gives back what `only_lost` counts as lost
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

    reach = _centre_reach(spec)
    if reach is None:
        return None
    rules["reach"] = reach
    settle_schema_version(spec)
    return f"the game is now also won by walking the {reach['piece']} to the centre"


def _centre_reach(spec: dict[str, Any]) -> dict[str, Any] | None:
    """`_win_by_arrival`'s rule: the first royal that can stand, aimed at the centre.

    ``None`` where no royal can stand. Shared with `_win_by_leaving`, which enters a
    game without a goal by writing this rule and its own flag in one edit.
    """
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
    # A centre square nobody can stand on is refused by name (a wall, or a hexagon's
    # missing corner): Q772a's walled `divide` has its whole middle rank walled, so there
    # the rule has no centre and is not offered.
    orientation = str(spec.get("setup", {}).get("orientation", "mirror"))
    unstandable = obstacle_squares(spec.get("rules", {}), width, height, orientation)
    unstandable |= absent_squares(spec.get("board"))
    squares = sorted(
        f"{files[file]}{rank + 1}"
        for file in middle_files
        for rank in middle_ranks
        if rank * width + file not in unstandable
    )
    if not squares:
        return None
    return {"piece": royal, "squares": squares, "mirror": False, "result": "win"}


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
    if vector in move.get("vectors", []) or _no_leg_for(move, vector):
        _set_symmetry(spec, move, rng)
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
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
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
    occupied |= _portal_squares(spec)
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
    # Castling's spans (schema 18, Q781): a wall where a king and rook cross or land is
    # refused by name, and one elsewhere on their rank is Omega's walled rank.
    setup = spec.get("setup")
    for span_rank, files in castling_spans(
        rules, setup if isinstance(setup, dict) else {}, width, height,
        str((setup or {}).get("orientation", "mirror")),
    ):
        occupied.update(f"{chr(ord('a') + file)}{span_rank + 1}" for file in files)
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
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
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
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
    from chess_variant_sandbox.spec.transform import set_contacts

    rules = spec.setdefault("rules", {})
    if rules.get("contacts"):
        kept = _keep_the_mover(spec, rng)
        if kept is not None:
            return kept
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


def _keep_the_mover(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Cross schema 18's keeping link on a standing contact table, or return ``None``.

    The third half-draw `become-what-you-take` takes before removing its rule (Q781), so
    the table survives the crossing. Leaving deletes every keeping link while another
    link remains to hold the table up. Entering picks one piece with a promotion of its
    own and adds a link keeping it as it is for every enemy target the table does not
    already pair it with: *a queen that takes stays a queen*, which is Einstein chess's
    one rule the audit could not say, and one edit rather than five. Royal pieces and
    immune targets are left out, as the validator would refuse them by name.
    """
    from chess_variant_sandbox.spec.schema import is_keeping_link, keeping_link_holds

    if rng.random() >= 1 / 3:
        return None
    rules = spec.get("rules", {})
    contacts = rules.get("contacts")
    links = contacts.get("links") if isinstance(contacts, dict) else None
    if not isinstance(contacts, dict) or not isinstance(links, list):
        return None
    names = {
        str(piece.get("id")): str(piece.get("name", piece.get("id")))
        for piece in pieces(spec)
        if isinstance(piece, dict)
    }
    keeping = [link for link in links if is_keeping_link(link)]
    if keeping:
        if len(keeping) == len(links):
            return None  # the table would go with them; the strip arm says that
        contacts["links"] = [link for link in links if not is_keeping_link(link)]
        settle_schema_version(spec)
        movers = sorted({names.get(str(link.get("mover")), "?") for link in keeping})
        return f"a {' or '.join(movers)} that takes is promoted as its promotion says again"
    royal = {
        str(piece.get("id")) for piece in pieces(spec)
        if isinstance(piece, dict) and piece.get("royal")
    }
    immune = set(contacts.get("immune") or ())
    paired = {
        (link.get("mover"), link.get("onto"))
        for link in links
        if isinstance(link, dict) and link.get("own") is not True
    }
    promoting = promoting_piece_ids(rules)
    targets = [pid for pid in names if pid not in royal and pid not in immune]
    choices = {
        mover: [onto for onto in targets if (mover, onto) not in paired]
        for mover in sorted(promoting)
        if mover in names and mover not in royal
    }
    choices = {mover: ontos for mover, ontos in choices.items() if ontos}
    if not choices:
        return None
    mover = rng.choice(sorted(choices))
    added = [{"mover": mover, "onto": onto, "mover_to": mover} for onto in choices[mover]]
    assert all(keeping_link_holds(link, promoting) for link in added)
    links.extend(added)
    settle_schema_version(spec)
    name = names[mover]
    return (
        f"a {name} that takes stays a {name}, whatever its promotion would have made it "
        "on that move"
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
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
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
        # Schema 21 (Q849): not a stone that a portal end alone lets in.
        stones = [stone for stone in stones if not empties_a_filter(spec, stone["id"])]
        if not stones:
            return None
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
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
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
    drops = rules.get("drops")
    if isinstance(drops, dict) and drops.get("either_face") is True:
        return None  # nor can a blast and a hand that drops either face (schema 18e)
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


def _fix_a_facing(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Stop a facing piece turning, or let one that cannot turn do so again (schema 25).

    Both halves of ``pieces[].facing.turns``. Fixing a facing is worth nothing unless the
    setup holds the piece two ways -- every way into play later takes its initial facing,
    so the validator refuses a fixed facing every copy shares -- so when all its
    placements point it the same way, one of them is pointed another way the piece has:
    a rank of pawns placed facing backwards, the rear guard the bump was for. Freeing one
    drops the key, which is the default, and leaves the placements where they stood.
    """
    pieces = [p for p in spec.get("pieces", []) if isinstance(p, dict)]
    facing = [p for p in pieces if isinstance(p.get("facing"), dict)]
    fixed = [p for p in facing if p["facing"].get("turns", True) is False]
    if fixed:
        piece = rng.choice(fixed)
        piece["facing"].pop("turns")
        settle_schema_version(spec)
        return f"{piece['id']} may now spend a move turning to face another way"
    placements = [
        p for p in (spec.get("setup") or {}).get("placements", []) if isinstance(p, dict)
    ]
    candidates = [
        p for p in facing if any(placement.get("piece") == p["id"] for placement in placements)
    ]
    if not candidates:
        return None
    piece = rng.choice(candidates)
    initial = piece["facing"].get("initial", "forward")
    own = [placement for placement in placements if placement.get("piece") == piece["id"]]
    pointed = ""
    if all(placement.get("facing", initial) == initial for placement in own):
        axes = int(piece["facing"].get("axes", 2))
        others = [
            name for index, name in enumerate(FACING_NAMES)
            if index % (len(FACING_NAMES) // axes) == 0 and name != initial
        ]
        placement = rng.choice(own)
        turned = rng.choice(others)
        squares = [square for square in placement.get("squares", [])]
        if len(squares) > 1:
            # Half of them, so the game keeps the piece both ways round.
            moved = sorted(rng.sample(squares, len(squares) // 2))
            placement["squares"] = [square for square in squares if square not in moved]
            split = {key: value for key, value in placement.items() if key != "squares"}
            split.update(squares=moved, facing=turned)
            spec["setup"]["placements"].append(split)
        else:
            moved = squares
            placement["facing"] = turned
        pointed = f"; those on {', '.join(moved)} now point {turned}"
    piece["facing"]["turns"] = False
    settle_schema_version(spec)
    return f"{piece['id']} no longer turns: each one points the way it was placed{pointed}"


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
        or rules.get("portals")
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
    survives = strike_is_live(strike) or (
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


def _count_each_kind(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Running out of any one kind loses, or the listed kinds count together again.

    Schema 18c's ``extinction.per_type`` (Q783, Extinction chess), both halves. A game
    whose extinction lists two or more types gains or loses the flag; a game with no
    extinction rule gains one over two or more of the non-royal types both armies start
    with, counted apart, which is the one edit that reaches the feature from a game
    without it (no committed game lists two types to count together). A rule over
    ``any`` is the whole army and has no kinds to count apart, so it is left alone.
    """
    rules = spec.setdefault("rules", {})
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and extinction.get("per_type") is True:
        extinction.pop("per_type")
        settle_schema_version(spec)
        return f"running out of {', '.join(extinction['pieces'])} counts them together again"
    if isinstance(extinction, dict):
        listed = extinction.get("pieces")
        if not isinstance(listed, list) or len(set(listed)) < 2:
            return None
        trial = copy.deepcopy(spec)
        trial["rules"]["extinction"]["per_type"] = True
        settle_schema_version(trial)
        if validate_spec(trial):
            return None
        spec.clear()
        spec.update(trial)
        return f"a player who runs out of any one of {', '.join(listed)} loses"
    if extinction:
        return None
    first, second = opening_counts(spec)
    kinds = sorted(
        piece["id"]
        for piece in _non_royal(spec)
        if first.get(piece["id"], 0) >= 1 and second.get(piece["id"], 0) >= 1
    )
    if len(kinds) < 2:
        return None
    chosen = sorted(rng.sample(kinds, rng.randint(2, len(kinds))))
    trial = copy.deepcopy(spec)
    trial["rules"]["extinction"] = {"pieces": chosen, "per_type": True}
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return f"a player who runs out of any one of {', '.join(chosen)} loses"


def _promote_only_lost(spec: dict[str, Any], rng: random.Random) -> str | None:
    """A piece promotes only into what its side has lost, or into any target again.

    Schema 18c's ``promotion.only_lost`` (Q783, Grand chess), both halves, one entry at a
    time. Offered only where the validator accepts it -- every target in both armies'
    setup, and no hand, circe strike or entry promotion beside it -- which is a question
    the validator answers rather than a second reading here.
    """
    promotion = (spec.get("rules") or {}).get("promotion")
    listed = promotion if isinstance(promotion, list) else [promotion]
    entries = [entry for entry in listed if isinstance(entry, dict)]
    names = {
        piece.get("id"): piece.get("name", piece.get("id")) for piece in spec.get("pieces", [])
    }
    flagged = [entry for entry in entries if entry.get("only_lost") is True]
    if flagged:
        entry = rng.choice(flagged)
        entry.pop("only_lost")
        settle_schema_version(spec)
        return f"the {names.get(entry.get('piece'))} may promote into any of its targets again"
    for index in rng.sample(range(len(entries)), len(entries)):
        trial = copy.deepcopy(spec)
        trial_promotion = trial["rules"]["promotion"]
        trial_entries = [
            entry
            for entry in (trial_promotion if isinstance(trial_promotion, list) else [trial_promotion])
            if isinstance(entry, dict)
        ]
        trial_entries[index]["only_lost"] = True
        settle_schema_version(trial)
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        piece = entries[index].get("piece")
        return f"the {names.get(piece)} promotes only into a piece its side has lost"
    return None


def _place_the_army(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Play a placement phase before the game, or put the army back on the board.

    Schema 18d's ``drops.placement`` (Q784, Placement chess and Sittuyin), both halves. A
    game with a stock gains or loses the flag. A game with no hand at all gains one by
    the edit that makes a placement game out of a game: the first player's back rank
    (every type that stands only there) goes into both players' hands, each type
    droppable on the squares the rank held, and castling goes with it, since a dropped
    king has moved. The way back puts each stocked piece on the first free squares of
    its region again, which is the only way off the flag while a royal is stocked: a
    royal in hand outside the phase is refused. ``rng`` is unused; every branch is the
    one edit its spec allows.
    """
    del rng
    rules = spec.setdefault("rules", {})
    drops = rules.get("drops")
    if isinstance(drops, dict) and drops.get("placement") is True:
        return _unplace_the_army(spec)
    if isinstance(drops, dict) and drops:
        if not isinstance(drops.get("stock"), dict) or not drops["stock"]:
            return None
        trial = copy.deepcopy(spec)
        trial["rules"]["drops"]["placement"] = True
        settle_schema_version(trial)
        if validate_spec(trial):
            return None
        spec.clear()
        spec.update(trial)
        return "a player holding a piece must place it before anything on the board moves"
    if rules.get("strike") or rules.get("contacts") or uses_sweep(spec):
        return None  # a hand is refused beside a rule that takes several pieces at once
    setup = spec.get("setup")
    board = spec.get("board") or {}
    width, height = board.get("width"), board.get("height")
    if (
        not isinstance(setup, dict)
        or setup.get("mirror", True) is not True
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        return None
    placements = [p for p in setup.get("placements") or [] if isinstance(p, dict)]
    ranks: dict[str, set[int]] = {}
    for placement in placements:
        if placement.get("player", 0) != 0:
            continue
        for name in placement.get("squares") or []:
            coords = parse_square(name, width, height) if isinstance(name, str) else None
            if coords is not None:
                ranks.setdefault(str(placement.get("piece")), set()).add(coords[1])
    stocked = sorted(piece for piece, held in ranks.items() if held == {0})
    if not stocked:
        return None
    kept: list[dict[str, Any]] = []
    stock: dict[str, int] = {}
    region: list[str] = []
    for placement in placements:
        piece = str(placement.get("piece"))
        if placement.get("player", 0) == 0 and piece in stocked:
            squares = [name for name in placement.get("squares") or [] if isinstance(name, str)]
            stock[piece] = stock.get(piece, 0) + len(squares)
            region.extend(squares)
        else:
            kept.append(placement)
    trial = copy.deepcopy(spec)
    trial["setup"]["placements"] = kept
    castling = trial["rules"].get("castling")
    if isinstance(castling, dict) and {castling.get("king"), castling.get("rook")} & set(stock):
        trial["rules"].pop("castling")
    region.sort(key=lambda name: parse_square(name, width, height) or (0, 0))
    trial["rules"]["drops"] = {
        "from_captures": False,
        "placement": True,
        "stock": stock,
        "region": {piece: list(region) for piece in stocked},
    }
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return (
        "the back rank starts in hand, and each player places it, a piece a move, on the "
        "squares it stood on before anything on the board moves"
    )


def _either_face(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Let a held piece come back as its other face, or as itself only again.

    Schema 18e's ``drops.either_face`` (Q785, Kyoto shogi), both halves: on a game whose
    hand holds a piece with a ``promoted_from`` pair (shogi, dobutsu) the flag goes on,
    and off a game that has it, it comes off. The validator decides whether the pairs
    allow it (one face per base, none stocked, royal or chained), and a refusal is a
    draw with nothing to do. ``rng`` is unused; each branch is the one edit its spec
    allows.
    """
    del rng
    drops = spec.get("rules", {}).get("drops")
    if not isinstance(drops, dict) or not drops:
        return None
    trial = copy.deepcopy(spec)
    if drops.get("either_face") is True:
        trial["rules"]["drops"].pop("either_face")
        said = "a piece in hand comes back as itself only"
    else:
        trial["rules"]["drops"]["either_face"] = True
        said = (
            "a piece in hand may come back as either face: itself, or the piece that "
            "reverts to it when captured"
        )
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _win_by_leaving(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Decide a game by leaving its goal square rather than by arriving there, or back.

    Schema 18f's ``reach.on: "leave"`` (Q786, Jeson Mor), both halves: on a game with a
    ``reach`` rule the flag goes on, and off a game that has it, it comes off, so the
    arrival twin and the leaving game are one edit apart. A game with no goal at all gets
    `_win_by_arrival`'s centre with the flag already on, since a flag with no rule to
    ride on is no edit and most parents have no goal. ``safe`` is refused beside leaving
    (it reads the piece standing on the goal), and the validator says so; a refusal is a
    draw with nothing to do. ``rng`` is unused; each branch is the one edit its spec
    allows.
    """
    del rng
    reach = spec.get("rules", {}).get("reach")
    trial = copy.deepcopy(spec)
    if not isinstance(reach, dict) or not reach:
        centre = _centre_reach(trial)
        if centre is None:
            return None
        trial.setdefault("rules", {})["reach"] = {**centre, "on": "leave"}
        said = (
            f"the game is now also won by walking the {centre['piece']} to the centre and "
            "stepping off it again"
        )
    elif reach.get("on") == "leave":
        trial["rules"]["reach"].pop("on")
        said = "the game is decided by arriving on the goal square again, not by leaving it"
    else:
        trial["rules"]["reach"]["on"] = "leave"
        said = (
            "the game is now decided by moving off the goal square, so a piece has to get "
            "there and then step away"
        )
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


_PORTAL_STEP = {
    "kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "move",
}
_PORTAL_IDS = ("O", "Y", "Z", "J", "U", "V")


def _has_movers(group: dict[str, Any]) -> bool:
    """Whether anybody may move an end of this group, in either spelling."""
    if "ends" in group:
        return any(end.get("moved_by") for end in group["ends"] if isinstance(end, dict))
    return any((group.get("moved_by") or {}).values())


def _take_movers(group: dict[str, Any]) -> bool:
    """Nobody may move any end of this group any more; whether anybody could."""
    had = _has_movers(group)
    group.pop("moved_by", None)
    for end in group.get("ends") or ():
        end.pop("moved_by", None)
    return had


def _give_movers(group: dict[str, Any]) -> None:
    """Either player may move every end of this group, in either spelling."""
    if "ends" in group:
        for end in group["ends"]:
            end["moved_by"] = [0, 1]
    else:
        group["moved_by"] = {"orange": [0, 1], "blue": [0, 1]}


def _open_a_portal(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Put a pair of portals on the board, or change one of its settings, or take it away.

    Schema 19 (Q771, `docs/plan-portals.md`), all three of its names both ways, and schema
    20's ``no_straight_back`` (Q807) both ways. On a game with no portals a pair goes on two
    empty squares that are each other's image under the setup's mirror, so neither side
    starts with the better door; the portal is a new piece worth nothing that steps one
    square and that nobody moves yet. On a game with portals one draw lets either player
    move a pair's two ends (or stops it, and with it any bar on going straight back), one
    makes a pair one way for each player (or two way again), one bars a moved portal from
    going straight back (letting either player move it, if nobody could) or lifts the bar,
    and one takes every portal away with the piece they moved like. Every branch is checked against the validator and drawn again rather
    than handed back refused: what portals are refused beside is the validator's list.
    """
    rules = spec.get("rules") or {}
    trial = copy.deepcopy(spec)
    trial_rules = trial.setdefault("rules", {})
    if rules.get("portals"):
        pairs = trial_rules["portals"]
        pair = rng.choice(pairs)
        number = pairs.index(pair) + 1
        draw = rng.random()
        if draw < 1 / 4:
            if _take_movers(pair):
                # A bar needs somebody to move the pair (schema 20's refusal).
                pair.pop("no_straight_back", None)
                said = f"nobody may move portal pair {number} any more"
            else:
                _give_movers(pair)
                portal = next(p for p in pieces(trial) if p.get("id") == pair["piece"])
                portal["moves"] = portal.get("moves") or [copy.deepcopy(_PORTAL_STEP)]
                said = (
                    f"either player may now spend a turn moving a portal of pair {number} "
                    "one step onto an empty square"
                )
        elif draw < 1 / 2:
            if "ends" in pair:
                return None  # a map's doors are `filter-a-portal`'s `enter`
            if pair.pop("passage", None):
                said = f"both players' pieces may go in by either portal of pair {number} again"
            else:
                pair["passage"] = {"0": "orange_to_blue", "1": "blue_to_orange"}
                said = (
                    f"portal pair {number} is one way now: the first player's pieces go in "
                    f"only at {pair['orange']}, the second player's only at {pair['blue']}"
                )
        elif draw < 3 / 4:
            if pair.pop("no_straight_back", None):
                said = (
                    f"a portal of pair {number} may be moved straight back where it came "
                    "from again"
                )
            else:
                pair["no_straight_back"] = True
                said = (
                    f"a portal of pair {number} that has just been moved may not be moved "
                    "straight back on the very next move"
                )
                if not _has_movers(pair):
                    _give_movers(pair)
                    portal = next(p for p in pieces(trial) if p.get("id") == pair["piece"])
                    portal["moves"] = portal.get("moves") or [copy.deepcopy(_PORTAL_STEP)]
                    said = (
                        f"either player may now spend a turn moving a portal of pair "
                        f"{number} one step onto an empty square, but not straight back on "
                        "the very next move"
                    )
        else:
            named = portal_piece_ids(trial_rules)
            del trial_rules["portals"]
            trial["pieces"] = [p for p in pieces(trial) if p.get("id") not in named]
            said = "the portals are gone, and the piece they moved like with them"
    else:
        board = trial.get("board", {})
        if board.get("lattice") == "hex" or any("region" in p for p in pieces(trial)):
            return None
        used_ids = {str(p.get("id")) for p in pieces(trial)}
        portal_id = next((pid for pid in _PORTAL_IDS if pid not in used_ids), None)
        if portal_id is None:
            return None
        width, height = int(board.get("width", 8)), int(board.get("height", 8))
        setup = trial.get("setup", {})
        orientation = str(setup.get("orientation", "mirror"))
        mirrored = bool(setup.get("mirror", True))
        taken = set(obstacle_squares(trial_rules, width, height, orientation))
        taken |= absent_squares(board)
        for placement in setup.get("placements", []):
            for name in placement.get("squares", []) if isinstance(placement, dict) else ():
                file, rank = ord(name[0]) - ord("a"), int(name[1:]) - 1
                taken.add(rank * width + file)
                if mirrored:
                    seen = reflect_square(file, rank, width, height, orientation)
                    taken.add(seen[1] * width + seen[0])
        doors: list[tuple[int, int]] = []
        for square in range(width * height):
            file, rank = square % width, square // width
            image_file, image_rank = reflect_square(file, rank, width, height, orientation)
            image = image_rank * width + image_file
            if square < image and square not in taken and image not in taken:
                doors.append((square, image))
        if not doors:
            return None
        orange, blue = rng.choice(doors)

        def name_of(square: int) -> str:
            return f"{chr(ord('a') + square % width)}{square // width + 1}"

        trial["pieces"] = [
            *pieces(trial),
            {"id": portal_id, "name": "portal", "value": 0,
             "moves": [copy.deepcopy(_PORTAL_STEP)]},
        ]
        pair = {"piece": portal_id, "orange": name_of(orange), "blue": name_of(blue)}
        said = (
            f"a pair of portals stands on {name_of(orange)} and {name_of(blue)}: a piece "
            "whose move ends on one comes out one step beyond the other, the way it was going"
        )
        # A quarter of the draws open the pair with one of its two settings already on,
        # so a game with no portals is one edit from each name the bump added.
        setting = rng.random()
        if setting < 0.25:
            pair["moved_by"] = {"orange": [0, 1], "blue": [0, 1]}
            said += ", and either player may spend a turn moving one a step"
        elif setting < 0.5:
            pair["passage"] = {"0": "orange_to_blue", "1": "blue_to_orange"}
            said += (
                f", one way: the first player's pieces go in only at {name_of(orange)}, the "
                f"second player's only at {name_of(blue)}"
            )
        trial_rules["portals"] = [pair]
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _portal_free_squares(trial: dict[str, Any]) -> list[str]:
    """The squares a new portal end may start on: on the board, not a wall, empty in the
    setup (and in its mirror image), and not already a portal end. Names, in board order."""
    board = trial.get("board", {})
    width, height = int(board.get("width", 8)), int(board.get("height", 8))
    setup = trial.get("setup", {})
    orientation = str(setup.get("orientation", "mirror"))
    mirrored = bool(setup.get("mirror", True))
    rules = trial.get("rules") or {}
    taken = set(obstacle_squares(rules, width, height, orientation)) | absent_squares(board)
    for placement in setup.get("placements", []):
        for name in placement.get("squares", []) if isinstance(placement, dict) else ():
            file, rank = ord(name[0]) - ord("a"), int(name[1:]) - 1
            taken.add(rank * width + file)
            if mirrored:
                seen = reflect_square(file, rank, width, height, orientation)
                taken.add(seen[1] * width + seen[0])
    ends = {
        str(end["at"]) for group in portal_groups(rules) for end in group["ends"]
    }
    return [
        name for square in range(width * height)
        if square not in taken
        and (name := f"{chr(ord('a') + square % width)}{square // width + 1}") not in ends
    ]


def _as_map(pair: dict[str, Any]) -> list[dict[str, Any]]:
    """A pair's two ends written as a map's, each keeping who moves it and who goes in."""
    words: dict[str, Any] = pair["passage"] if isinstance(pair.get("passage"), dict) else {}
    moved_by: dict[str, Any] = (
        pair["moved_by"] if isinstance(pair.get("moved_by"), dict) else {}
    )
    carries: dict[str, Any] = pair["carries"] if isinstance(pair.get("carries"), dict) else {}
    ends: list[dict[str, Any]] = []
    for side, key in enumerate(("orange", "blue")):
        enter = [
            player for player in (0, 1)
            if portal_open_ends(words.get(str(player), "both_ways"))[side]
        ]
        end: dict[str, Any] = {"at": pair[key]}
        if enter:
            end["to"] = 1 - side
            if enter != [0, 1]:
                end["enter"] = enter
            if key in carries:
                end["carries"] = copy.deepcopy(carries[key])
        if moved_by.get(key):
            end["moved_by"] = list(moved_by[key])
        ends.append(end)
    return ends


def _as_pair(group: dict[str, Any]) -> dict[str, Any]:
    """A map's first two ends written as a pair: each leads to the other, who goes in by
    each is kept as a passage, and the rest of the map goes."""
    first, second = group["ends"][0], group["ends"][1]
    pair: dict[str, Any] = {"piece": group["piece"], "orange": first["at"], "blue": second["at"]}
    moved_by = {key: list(end["moved_by"]) for key, end in (("orange", first), ("blue", second))
                if end.get("moved_by")}
    if moved_by:
        pair["moved_by"] = moved_by
    entered = [set(end.get("enter", [0, 1])) if "to" in end else set() for end in (first, second)]
    words = {}
    for player in (0, 1):
        opened = (player in entered[0], player in entered[1])
        word = {(True, True): "both_ways", (True, False): "orange_to_blue",
                (False, True): "blue_to_orange", (False, False): "closed"}[opened]
        if word != "both_ways":
            words[str(player)] = word
    if words:
        pair["passage"] = words
    carries = {key: list(end["carries"]) for key, end in (("orange", first), ("blue", second))
               if end.get("carries") and "to" in end}
    if carries:
        pair["carries"] = carries
    if group.get("no_straight_back") and moved_by:
        pair["no_straight_back"] = True
    if group.get("kind") == "fold":
        pair["kind"] = "fold"
    return pair


def _reshape_portals(spec: dict[str, Any], rng: random.Random, shape: str) -> str | None:
    """`ring-the-portals` and `funnel-the-portals` (schema 21, `plan-portals-2.md` §2): a
    pair becomes a map of three, or a map becomes a pair again.

    A ring: the pair's two ends and a third on an empty square lead round, each to the
    next. A funnel: the pair's two ends both lead to a third, a way out only. Who moves and
    who goes in by the two old ends is kept. On a game whose portals include a map, a
    draw writes one back as a pair from its first two ends, so the axis is crossed both
    ways. Checked against the validator and drawn again rather than handed back refused.
    """
    rules = spec.get("rules") or {}
    groups = rules.get("portals")
    if not isinstance(groups, list) or not groups:
        return None
    trial = copy.deepcopy(spec)
    held = trial["rules"]["portals"]
    maps = [index for index, group in enumerate(held) if "ends" in group]
    pairs = [index for index, group in enumerate(held) if "ends" not in group]
    if maps and (not pairs or rng.random() < 0.5):
        index = rng.choice(maps)
        held[index] = _as_pair(held[index])
        said = (f"portal group {index + 1} is a pair again, on {held[index]['orange']} and "
                f"{held[index]['blue']}")
    elif pairs:
        index = rng.choice(pairs)
        if sum(len(g["ends"]) for g in portal_groups(trial["rules"])) >= 8:
            return None
        free = _portal_free_squares(trial)
        if not free:
            return None
        square = rng.choice(free)
        pair = held[index]
        ends = _as_map(pair)
        if shape == "ring":
            ends[0]["to"], ends[1]["to"] = 1, 2
            ends.append({"at": square, "to": 0})
            said = (f"portal pair {index + 1} is a ring of three now: {ends[0]['at']} leads "
                    f"to {ends[1]['at']}, {ends[1]['at']} to {square} and {square} back to "
                    f"{ends[0]['at']}")
        else:
            for end in ends:
                end["to"] = 2
            ends.append({"at": square})
            said = (f"portal pair {index + 1} is a funnel now: {ends[0]['at']} and "
                    f"{ends[1]['at']} both lead out beside {square}, which nobody goes in by")
        group: dict[str, Any] = {"piece": pair["piece"], "ends": ends}
        if pair.get("no_straight_back"):
            group["no_straight_back"] = True
        if pair.get("kind") == "fold":
            group["kind"] = "fold"
        held[index] = group
    else:
        return None
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _ring_the_portals(spec: dict[str, Any], rng: random.Random) -> str | None:
    return _reshape_portals(spec, rng, "ring")


def _funnel_the_portals(spec: dict[str, Any], rng: random.Random) -> str | None:
    return _reshape_portals(spec, rng, "funnel")


def _filter_a_portal(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Let only some pieces go in by one portal end, or every piece again; or, on a map,
    let only one player's pieces go in by an end, or both again (schema 21's ``carries``
    and ``enter``, `plan-portals-2.md` §2.3).

    A filter names one piece that can move, drawn at random, so the end is a door to it
    and a solid block to every other. Checked against the validator and drawn again
    rather than handed back refused.
    """
    rules = spec.get("rules") or {}
    if not isinstance(rules.get("portals"), list) or not rules["portals"]:
        return None
    trial = copy.deepcopy(spec)
    held = trial["rules"]["portals"]
    portal_ids = portal_piece_ids(trial["rules"])
    movers = sorted(
        str(p["id"]) for p in pieces(trial) if p.get("moves") and p.get("id") not in portal_ids
    )
    index = rng.randrange(len(held))
    group = held[index]
    number = index + 1
    if "ends" in group:
        doors = [end for end in group["ends"] if "to" in end]
        if not doors:
            return None
        end = rng.choice(doors)
        if rng.random() < 0.5:
            if end.pop("enter", None) is not None:
                said = f"both players' pieces may go in by the portal on {end['at']} again"
            else:
                player = rng.choice((0, 1))
                end["enter"] = [player]
                said = (f"only the {('first', 'second')[player]} player's pieces may go in "
                        f"by the portal on {end['at']}")
        elif end.pop("carries", None) is not None:
            said = f"every piece may go in by the portal on {end['at']} again"
        else:
            if len(movers) < 2:
                return None
            end["carries"] = [rng.choice(movers)]
            said = f"only {end['carries'][0]} may go in by the portal on {end['at']}"
    else:
        key = rng.choice(("orange", "blue"))
        carries = group.get("carries") if isinstance(group.get("carries"), dict) else {}
        if key in carries:
            carries.pop(key)
            if not carries:
                group.pop("carries", None)
            said = f"every piece may go in by the portal on {group[key]} of pair {number} again"
        else:
            if len(movers) < 2:
                return None
            group.setdefault("carries", {})[key] = [rng.choice(movers)]
            said = (f"only {group['carries'][key][0]} may go in by the portal on "
                    f"{group[key]} of pair {number}")
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _fold_a_portal(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make one portal group fold portals, or momentum portals again (schema 24's ``kind``,
    `plan-portals-2.md` §3R).

    A fold is not a piece: a piece may stand in it, and a move that ends on an empty one may
    stay there or go through and end on the end it leads to, taking a piece there as any
    move does; a move that passes over it is not changed. Checked against the validator and
    drawn again rather than handed back refused -- a fold that starts under a piece cannot
    become a momentum portal, which may not.
    """
    rules = spec.get("rules") or {}
    if not isinstance(rules.get("portals"), list) or not rules["portals"]:
        return None
    trial = copy.deepcopy(spec)
    held = trial["rules"]["portals"]
    index = rng.randrange(len(held))
    group = held[index]
    shape = "group" if "ends" in group else "pair"
    if group.pop("kind", None) == "fold":
        said = (f"portal {shape} {index + 1} is momentum portals again: a piece that goes in "
                "comes out one step beyond the other end")
    else:
        group["kind"] = "fold"
        said = (f"portal {shape} {index + 1} is fold portals now: a move that ends on one "
                "may stay there or come out of the other")
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _unplace_the_army(spec: dict[str, Any]) -> str | None:
    """`_place_the_army`'s way back: the flag off, and the army on the board where needed."""
    rules = spec["rules"]
    drops = rules["drops"]
    royal = {str(piece.get("id")) for piece in pieces(spec) if piece.get("royal")}
    stock = drops.get("stock") if isinstance(drops.get("stock"), dict) else {}
    if not royal & set(stock):
        drops.pop("placement")
        settle_schema_version(spec)
        return "a player holding a piece may wait to place it again"
    setup = spec.get("setup")
    board = spec.get("board") or {}
    width, height = board.get("width"), board.get("height")
    region = drops.get("region") or {}
    if not isinstance(setup, dict) or not isinstance(width, int) or not isinstance(height, int):
        return None
    mirrored = setup.get("mirror", True) is True
    orientation = str(setup.get("orientation", "mirror"))
    taken = {
        name
        for placement in setup.get("placements") or []
        if isinstance(placement, dict)
        for name in placement.get("squares") or []
    }
    added: list[dict[str, Any]] = []
    for piece, held in stock.items():
        if not isinstance(held, int) or isinstance(held, bool):
            return None  # an inexhaustible slot has no army to put back
        free = [name for name in region.get(piece) or [] if name not in taken]
        if len(free) < held:
            return None
        chosen = free[:held]
        taken.update(chosen)
        added.append({"piece": piece, "squares": chosen})
        if not mirrored:
            images = []
            for name in chosen:
                coords = parse_square(name, width, height)
                if coords is None:
                    return None
                file, rank = reflect_square(*coords, width, height, orientation)
                images.append(f"{chr(ord('a') + file)}{rank + 1}")
            if taken & set(images):
                return None
            taken.update(images)
            added.append({"piece": piece, "player": 1, "squares": images})
    trial = copy.deepcopy(spec)
    trial["setup"]["placements"] = [*trial["setup"].get("placements", []), *added]
    if trial["rules"]["drops"].get("from_captures", True) is False:
        trial["rules"].pop("drops")
    else:
        for key in ("stock", "placement", "region"):
            trial["rules"]["drops"].pop(key, None)
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return "the pieces in hand start on the board again, where each was first placed"


def _leave_something_behind(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Change what a capture leaves behind, or make it ordinary again (schema 17, Q540).

    Schema 17's four features, which no other operator writes: a capturer that stays
    where it stood (rifle), a victim reborn on its home square (circe), a capturer that
    changes sides (Andernach) and a blast that takes only the two pieces (kamikaze).
    Leaving takes the feature off and keeps its host where it has one: a kamikaze blast
    becomes atomic's radius 1, and a switching strike keeps its chain when it has one.

    Entering draws one of the three strike spellings on a game with no rule that already
    says what a capture leaves behind, adds `switches_sides` to a standing strike, or
    narrows a standing blast to radius 0 (taking off `immune`, which radius 0 refuses).
    Every royal is immune and never switches, which the validator requires; the switching
    pieces are every non-royal one, Andernach's reading.
    """
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
    rules = spec.setdefault("rules", {})
    strike = rules.get("strike")
    explosion = rules.get("explode_on_capture")
    if isinstance(explosion, dict) and explosion.get("radius") == 0:
        explosion["radius"] = 1
        settle_schema_version(spec)
        return "a capture takes its neighbours with it again, not only the two pieces"
    if isinstance(strike, dict) and "switches_sides" in strike:
        strike.pop("switches_sides")
        if not strike_is_live(strike):
            strip_rule(spec, "strike")
        settle_schema_version(spec)
        return "a piece that captures stays on its own side again"
    if (
        isinstance(strike, dict)
        and strike.get("effect", "capture") == "capture"
        and strike.get("position") in ("stay", "home")
    ):
        strip_rule(spec, "strike")
        settle_schema_version(spec)
        return "a capture is an ordinary capture again: the taker moves in, the taken is gone"

    if isinstance(explosion, dict) and explosion:
        explosion["radius"] = 0
        explosion.pop("immune", None)
        settle_schema_version(spec)
        return "a capture destroys the taker with what it took, and nothing beside them"
    if (
        explosion or rules.get("drops") or rules.get("npcs") or rules.get("contacts")
        or uses_sweep(spec)
    ):
        return None  # two rules cannot both say what happens to a taken piece
    switching = sorted(str(piece["id"]) for piece in _non_royal(spec))
    if isinstance(strike, dict) and strike:
        if not strike_is_live(strike) or not switching:
            return None
        strike["switches_sides"] = switching
        settle_schema_version(spec)
        return "a piece that captures also changes sides on the square it took"
    if not capture_is_possible(spec):
        return None
    royals = sorted(str(piece["id"]) for piece in pieces(spec) if piece.get("royal"))
    rule: dict[str, Any] = {"effect": "capture"}
    if royals:
        rule["immune"] = royals
    promotion = rules.get("promotion")
    only_lost = any(
        isinstance(entry, dict) and entry.get("only_lost") is True
        for entry in (promotion if isinstance(promotion, list) else [promotion])
    )
    # Schema 18c: circe gives back what `only_lost` counts as lost, so it is not offered.
    choice = rng.choice(("stay", "switch") if only_lost else ("stay", "home", "switch"))
    if choice == "switch":
        if not switching:
            return None
        rule["switches_sides"] = switching
        said = "a piece that captures changes sides on the square it took"
    else:
        rule["position"] = choice
        said = (
            "a capture removes the taken piece and leaves the taker where it stood"
            if choice == "stay"
            else "a taken piece is reborn on its home square, or is gone if that is occupied"
        )
    rules["strike"] = rule
    settle_schema_version(spec)
    return said


def _sweep_a_slider(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Turn one slide into a sweep, or a sweep back into a slide.

    Schema 13's move kind (Q21), which no other operator writes. The descriptor keeps its
    vectors, symmetry, range and condition, so the piece reaches the same squares and only
    what stands between stops mattering. A promoting piece is never offered, and neither
    is a spec holding a rule the validator refuses beside a sweep: both would make every
    draw a rejection.
    """
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
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


def _share_a_walk(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Put another wild piece on a walk that has one, or take the others off it (schema 23's
    ``also_at``, `designer-case-study-wolves.md` §2.2).

    The new one starts as far round the route from the first as the route allows -- half a
    lap, the user's patrol -- because that is the start that leaves the two longest apart
    and the one a designer asked for. Checked against the validator and drawn again rather
    than handed back refused: a start under the opening array is refused by name.
    """
    rules = spec.get("rules") or {}
    npcs = rules.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return None
    walks = [
        index for index, npc in enumerate(npcs)
        if isinstance(npc, dict) and isinstance(npc.get("route"), list)
        and len(npc["route"]) > 1
    ]
    if not walks:
        return None
    trial = copy.deepcopy(spec)
    index = rng.choice(walks)
    npc = trial["rules"]["npcs"][index]
    kind = npc.get("piece")
    if npc.pop("also_at", None):
        said = f"the wild {kind} on {npc['route'][0]} walks its route alone again"
    else:
        start = npc["route"][len(npc["route"]) // 2]
        npc["also_at"] = [start]
        said = (f"a second wild {kind} starts on {start} and walks the same route as the one "
                f"on {npc['route'][0]}, half a lap behind it")
    settle_schema_version(trial)
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


OPERATORS: tuple[Operator, ...] = (
    Operator("wall-off-a-square", _wall_off_a_square),
    Operator("soften-the-capture", _soften_the_capture),
    Operator("become-what-you-take", _become_what_you_take),
    Operator("merge-your-own", _merge_your_own),
    Operator("set-the-blast", _set_the_blast),
    Operator("leave-something-behind", _leave_something_behind),
    Operator("place-a-stone", _place_a_stone),
    Operator("set-a-patrol", _set_a_patrol),
    Operator("share-a-walk", _share_a_walk),
    Operator("promote-what-arrives", _promote_what_arrives),
    Operator("promote-on-leaving", _promote_on_leaving),
    Operator("count-each-kind", _count_each_kind),
    Operator("promote-only-lost", _promote_only_lost),
    Operator("place-the-army", _place_the_army),
    Operator("either-face", _either_face),
    Operator("win-by-leaving", _win_by_leaving),
    Operator("open-a-portal", _open_a_portal),
    Operator("ring-the-portals", _ring_the_portals),
    Operator("funnel-the-portals", _funnel_the_portals),
    Operator("filter-a-portal", _filter_a_portal),
    Operator("fold-a-portal", _fold_a_portal),
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
    Operator("fix-a-facing", _fix_a_facing),
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
