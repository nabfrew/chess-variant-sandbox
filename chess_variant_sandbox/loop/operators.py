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
import json
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
    DEFAULT_RANGE,
    FACING_NAMES,
    HEX_SYMMETRIES,
    POUNCE_KINDS,
    PRICE_MAX,
    PURSE_MAX,
    REPETITION_NEVER,
    REST_MAX,
    SIGHT_EFFECTS,
    SIGHT_MAX,
    SPLIT_MAX,
    SPLIT_MIN,
    SQUARE_SYMMETRIES,
    TURNING_PATHS,
    _any_equal_speed_meeting,
    _check_extinction_coherence,
    _sight_entry,
    absent_squares,
    beats_of,
    can_double_step,
    castling_spans,
    duck_piece,
    format_square,
    has_equal_speed_opponent,
    has_rests,
    has_sights,
    has_speeds,
    hire_ids,
    is_nobodys,
    is_simultaneous,
    majority_ids,
    neutral_ids,
    npc_pounce,
    npc_respawn,
    npc_split,
    npc_start_count,
    npc_turns_back,
    obstacle_squares,
    opening_counts,
    parse_square,
    path_of,
    path_refusals,
    piece_bounties,
    piece_has_stock,
    piece_prices,
    piece_rests,
    piece_sights,
    piece_speeds,
    placed_ids,
    portal_groups,
    portal_open_ends,
    promoting_piece_ids,
    purse_bribes,
    purse_market,
    purse_rule,
    reach_reflection,
    reads_alike_for_both,
    reflect_square,
    simultaneous_tie_pair,
    spy_ids,
    strike_is_live,
    strikers,
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
    "tie": "the tie rule for collisions",
    "swap": "the collision of swapping pieces",
    "both": "the score settling an ending both players reach at once",
    "speed": "the pieces' speeds",
    "speed ranks": "the gap in the pieces' speeds",
}


def _gone_words(key: str) -> str:
    """One key `settle_inert_rules` returned, in a player's words."""
    if key.startswith("promotion:"):
        return f"the {key.split(':', 1)[1]}'s promotion"
    if key.startswith("immune:"):
        return f"the wild {key.split(':', 1)[1]}'s immunity"
    if key.startswith("beats:"):
        return f"the {key.split(':', 1)[1]}'s own collision table"
    if key.startswith("speed:"):
        return f"the {key.split(':', 1)[1]}'s speed"
    if key.startswith("rest:"):
        return f"the {key.split(':', 1)[1]}'s rest"
    if key.startswith("sight:"):
        return f"a line of sight of the {key.split(':', 1)[1]}"
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

    A portal's moves are its group's (spelling 31), not a piece's, so they are never drawn
    here: they must stay move-only and read the same for both players, which a direction or
    a symmetry drawn for an army's piece breaks (Q772a). `open-a-portal` edits portals.
    """
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
    # Schema 21 (Q849): a portal end only this piece may go in by would let nobody in
    # without it, which no filter can say, and ``drop_piece`` would open it to everyone.
    candidates = [piece for piece in candidates if not empties_a_filter(spec, piece["id"])]
    # Q937: nor a type whose removal leaves a side short of `extinction.min_count` of the
    # counted kinds (horde's pawns under `any`), a game over at ply zero the validator
    # refuses. Asked of the child by the validator's own check, so the two cannot disagree.
    candidates = [piece for piece in candidates if not _starts_run_out(spec, str(piece["id"]))]
    if not candidates:
        return None
    piece = rng.choice(candidates)
    drop_piece(spec, str(piece["id"]))
    return f"removed the {piece.get('name', piece['id'])}"


def _starts_run_out(spec: dict[str, Any], piece_id: str) -> bool:
    """Whether removing ``piece_id`` leaves a side starting with fewer than
    ``extinction.min_count`` of the counted kinds."""
    child = copy.deepcopy(spec)
    drop_piece(child, piece_id)
    rules = child.get("rules")
    errors: list[str] = []
    _check_extinction_coherence(child, rules if isinstance(rules, dict) else {}, errors)
    return bool(errors)


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
    if path_of(move) != "straight":
        lengths = [n for n in lengths if n > 1]  # T1: a path of one step never turns
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
    move.pop("path", None)  # T1: a leap is one jump, with no path to turn
    settle_en_passant(spec)
    return f"turned the {piece.get('name', piece['id'])}'s slide into a leap"


def _ride_a_leap(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Q1007a (`docs/archive/plan-schema-31.md` §2.5): a lame leap rides, or a lame rider stops.

    A single lame leap gains a ``range`` (0, 2 or 3 jumps, and now and then a ``min_range``
    of 2): jump after jump, each blocked on its leg. A lame rider loses both and is the
    single leap again, so the axis crosses both ways. Option 1, a leap ridden as a slide
    over its vector (the nightrider), is `leap-becomes-slide`'s and `slide-becomes-leap`'s.
    A lame move keeps reaching a second square either way, so en passant is untouched
    (`schema.can_double_step`).
    """
    movers = _movers(spec, "lame")
    if not movers:
        return None
    piece, move = rng.choice(movers)
    name = piece.get("name", piece["id"])
    if move.get("range", 1) != 1:
        move.pop("range", None)
        move.pop("min_range", None)
        return f"made the {name}'s blockable leap a single jump again"
    move["range"] = rng.choice((0, 2, 3))
    if rng.random() < 0.25:
        move["min_range"] = 2
    return f"let the {name} leap again and again, each jump blockable on its way"


def _bend_a_line(spec: dict[str, Any], rng: random.Random) -> str | None:
    """T1 (`docs/archive/plan-pieces-moves.md` §5.5): a straight slide's path turns, a circle or a
    zigzag, wherever the validator lets that set of directions turn (`path_refusals`).

    Each child must pass the validator whole, which also refuses a path beside portals, on
    the piece en passant is about and on a wild piece, so those are drawn and declined
    rather than copied here. `straighten` is the way back."""
    candidates = [
        (index, position, path)
        for index, piece in enumerate(pieces(spec))
        for position, move in enumerate(piece.get("moves", []))
        if move.get("kind") == "slide" and path_of(move) == "straight"
        and move.get("range", 0) != 1
        for path in TURNING_PATHS
    ]
    for index, position, path in rng.sample(candidates, len(candidates)):
        trial = copy.deepcopy(spec)
        piece = pieces(trial)[index]
        piece["moves"][position]["path"] = path
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        name = piece.get("name", piece["id"])
        return (
            f"the {name}'s line turns as it goes, round in a circle, each step the next "
            "direction round" if path == "circle"
            else f"the {name}'s line zigzags, each step turning to a neighbouring direction "
            "and back"
        )
    return None


def _straighten(spec: dict[str, Any], rng: random.Random) -> str | None:
    """T1: a slide whose path turns runs straight again (``path`` taken out)."""
    movers = [(p, m) for p, m in _movers(spec, "slide") if path_of(m) != "straight"]
    if not movers:
        return None
    piece, move = rng.choice(movers)
    move.pop("path", None)
    return f"the {piece.get('name', piece['id'])}'s line runs straight again"


def _still_a_piece(spec: dict[str, Any], rng: random.Random) -> str | None:
    """I1 (`docs/archive/plan-pieces-moves.md` §4.5): a piece on the board stops moving, or one that
    never moves is given a king's step.

    Still: a non-royal type the setup places, that castles and promotes in no rule, loses
    every move (`moves: []`): it stands, blocks and can be taken. Freed: a type with no
    moves that is not a stocked stone alone (that is `place-a-stone`'s) and not wild gets
    one square in every direction. Each child must pass the validator, which refuses a
    still piece some rule still asks to move."""
    from chess_variant_sandbox.spec.schema import piece_has_stock

    found = spec.get("rules")
    rules: dict[str, Any] = found if isinstance(found, dict) else {}
    wild = {npc.get("piece") for npc in rules.get("npcs") or [] if isinstance(npc, dict)}
    wild.add(duck_piece(rules))  # generation 42: the duck's moves are `loose-a-duck`'s
    placed = {
        placement.get("piece")
        for placement in (spec.get("setup") or {}).get("placements") or []
        if isinstance(placement, dict)
    }
    hexy = (spec.get("board") or {}).get("lattice") == "hex"
    step = (
        {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate6"} if hexy
        else {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
    )
    candidates = [
        piece for piece in pieces(spec)
        if isinstance(piece.get("id"), str) and piece["id"] not in wild
        and (
            (piece.get("moves") == [] and (piece["id"] in placed
                                           or not piece_has_stock(rules, piece["id"])))
            or (piece.get("moves") and not piece.get("royal") and piece["id"] in placed)
        )
    ]
    # Either way first, half the time each, so a game with both kinds crosses both ways.
    freeing = rng.random() < 0.5
    ordered = rng.sample(candidates, len(candidates))
    ordered.sort(key=lambda piece: bool(piece.get("moves")) == freeing)
    for piece in ordered:
        trial = copy.deepcopy(spec)
        target = next(p for p in pieces(trial) if p.get("id") == piece["id"])
        name = piece.get("name", piece["id"])
        if target.get("moves"):
            target["moves"] = []
            said = f"the {name} never moves: it stands where it starts until it is taken"
        else:
            target["moves"] = [copy.deepcopy(step)]
            said = f"the {name} may now step one square in any direction"
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        return said
    return None


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
    """Give ``move`` another of its lattice's symmetries: the hexagon's on a hex board."""
    current = move.get("symmetry", "none")
    move["symmetry"] = rng.choice([s for s in _lattice_symmetries(spec) if s != current])
    return str(current)


def _pouncing_types(spec: dict[str, Any]) -> set[str]:
    """The wild types some `rules.npcs` entry makes pounce (generation 39)."""
    npcs = (spec.get("rules") or {}).get("npcs") or []
    return {
        str(npc.get("piece")) for npc in npcs
        if isinstance(npc, dict) and npc_pounce(npc) is not None
    }


def _change_symmetry(spec: dict[str, Any], rng: random.Random) -> str | None:
    movers = _movers(spec)
    if not movers:
        return None
    if _lattice_symmetries(spec) is HEX_SYMMETRIES:
        return _turn_a_hex_move(spec, movers, rng)
    piece, move = rng.choice(movers)
    if str(piece.get("id")) in _pouncing_types(spec):
        # Generation 39 (N3b): a pouncer's moves must read the same for both players (a wild
        # piece has no forward), so it draws only among the symmetries that keep them so
        # (N7's hydra was the first library game with a pouncer to meet this).
        orientation = (spec.get("setup") or {}).get("orientation", "mirror")
        current = move.get("symmetry", "none")
        alike = [
            s for s in _lattice_symmetries(spec)
            if s != current and reads_alike_for_both(move.get("vectors"), s, orientation)
        ]
        if not alike:
            return None
        move["symmetry"] = rng.choice(alike)
        return f"changed how the {piece.get('name', piece['id'])}'s moves reflect ({current} to {move['symmetry']})"
    current = _set_symmetry(spec, move, rng)
    # T1: a turning path needs its directions to go round (a circle) or to have a near pair
    # (a zigzag); where the new symmetry leaves none, the line runs straight, by the
    # validator's own reading (`path_refusals`).
    if path_of(move) != "straight" and path_refusals(move, "move"):
        move.pop("path", None)
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
    # Generation 42 (N4b): the duck takes nothing and nothing may name it in a capture list.
    duck = duck_piece(spec.get("rules"))
    candidates = [p for p in pieces(spec) if p.get("id") and p.get("id") != duck]
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
            return f"gave the game a draw by repetition again, after {rules[key]}"
        if isinstance(current, int) and not isinstance(current, bool) and rng.random() < 0.5:
            rules[key] = REPETITION_NEVER
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
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (the duck is for a game without check)
    rules = spec.setdefault("rules", {})
    current = rules.get("win", "checkmate")
    if royal_id(spec) is None:
        return None
    if is_simultaneous(rules):
        return None  # schema 26: there is no checkmate when both players move at once
    rules["win"] = "royal_capture" if current == "checkmate" else "checkmate"
    said = f"the game is now won by {rules['win'].replace('_', ' ')}"
    # N5: a wild piece that turns back is refused beside checkmate, so under it the walks go
    # round again (the guard both ways: `pace-a-line` moves the other way).
    npcs = rules.get("npcs")
    if rules["win"] == "checkmate" and isinstance(npcs, list) and any(map(npc_turns_back, npcs)):
        for npc in npcs:
            if isinstance(npc, dict):
                npc.pop("turns_back", None)
        said += ", and the wild pieces walk their routes round again"
    return said


def _counts_own_kinds(rules: dict[str, Any]) -> bool:
    """Schema 27's ``"per_type": "own"``, which the validator refuses beside a hand."""
    extinction = rules.get("extinction")
    return isinstance(extinction, dict) and extinction.get("per_type") == "own"


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
    if _turncoats(spec):
        return None  # Q1007c: a capture hand is refused beside captures_own (§3.6)
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
        return "captured pieces are gone for good again"
    if rules.get("strike") or rules.get("contacts") or uses_sweep(spec):
        return None  # a hand is refused beside a rule that takes several pieces at once
    if _counts_own_kinds(rules):
        return None  # schema 27 (Q936): a hand is refused beside `"per_type": "own"`
    if any(npc_split(npc) for npc in rules.get("npcs") or ()):
        return None  # generation 41 (N3d): copies would mint pieces for a capture hand
    if spy_ids(spec):
        return None  # schema 27: a hand that fills from captures or holds a spy gives one back
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside majority control.
    if majority_ids(spec):
        return None
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
    rules = spec.setdefault("rules", {})
    if rules.get("reach"):
        rules.pop("reach", None)
        return "arriving somewhere no longer wins the game"

    reach = _centre_reach(spec)
    if reach is None:
        return None
    rules["reach"] = reach
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
        # TC4: a type's bounty is paid into the points the count read; it goes with it,
        # unless (TC5) a purse still pays it.
        if piece_bounties(spec) and not isinstance(spec["rules"].get("purse"), dict):
            for piece in pieces(spec):
                piece.pop("bounty", None)
            return "a game that runs out of clock is a draw again, and taking pays nothing"
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside majority control.
    if majority_ids(spec):
        return None
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
    rules = spec.setdefault("rules", {})
    if rules.get("territory"):
        strip_rule(spec, "territory")
        if (rules.get("scoring") or {}).get("count") == "territory":
            strip_rule(spec, "scoring")
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside majority control.
    if majority_ids(spec):
        return None
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (a turn already has its two halves)
    if purse_rule(spec) is not None:
        return None  # TC5 (`docs/plan-turncoat.md` §5.9): a purse is refused beside a long turn
    rules = spec.setdefault("rules", {})
    if rules.get("turn"):
        strip_rule(spec, "turn")
        return "a turn is one move again"
    if rules.get("npcs"):
        # The other half of `_set_a_patrol`'s guard, which has always refused a patrol
        # beside a `turn` and was never refused back: a wild piece walks between the two
        # halves of a long turn and the validator says so by name. Every draw on
        # `archive/wolves` was refused before this.
        return None
    if is_simultaneous(rules):
        return None  # schema 26: a round is one secret move each
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside a piece nobody owns.
    if neutral_ids(spec):
        return None
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (a strike)
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
    rules = spec.setdefault("rules", {})
    if rules.get("strike"):
        placed = _place_the_survivor(spec, "strike", rng)
        if placed is not None:
            return placed
        strip_rule(spec, "strike")
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
    if is_simultaneous(rules):
        return None  # schema 26: a strike that stays is refused beside rounds
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside a piece nobody owns.
    if neutral_ids(spec):
        return None
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (a strike)
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
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
        return "taking a piece leaves the taker as it was again"
    if (
        rules.get("strike") or rules.get("explode_on_capture") or rules.get("drops")
        or rules.get("npcs") or uses_sweep(spec)
    ):
        return None  # two rules cannot both say what happens to a taken piece
    before = rules.get("contacts")
    if spy_ids(spec):
        return None  # schema 27: a contact chain may turn a piece into the spy, which a spy game refuses
    set_contacts(spec, effect=rng.choice(("promote", "demote")))
    if rules.get("contacts") is before:
        return None  # one non-royal piece, so there is no rung to move to
    # I1: a link whose mover never moves never fires, and is refused by name.
    still = {piece.get("id") for piece in pieces(spec) if piece.get("moves") == []}
    links = [link for link in rules["contacts"]["links"] if link.get("mover") not in still]
    if not links:
        rules.pop("contacts")
        return None
    rules["contacts"]["links"] = links
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
        return "a struck piece that survives stays where it stood again"
    if rule.get("position") not in ("stay", "swap") or not keeps_a_survivor(spec, key):
        return None
    set_landing(spec, key)
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside a piece nobody owns.
    if neutral_ids(spec):
        return None
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (contacts)
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
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
            return f"the {pair[0]} no longer merges with its own {pair[1]}; it meets an enemy one instead"
        links.pop(index)
        if not links:
            strip_rule(spec, "contacts")
        settle_landing(spec)
        return f"the {pair[0]} can no longer move onto its own {pair[1]}"

    if spy_ids(spec):
        return None  # schema 27: a merge may turn a piece into the spy, which a spy game refuses

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
        return f"the {stone.get('name', stone['id'])} is gone; every move is a move again"

    if rules.get("strike") or rules.get("contacts") or uses_sweep(spec):
        return None  # a hand is refused beside a rule that takes several pieces at once
    if _counts_own_kinds(rules):
        return None  # schema 27 (Q936): a hand is refused beside `"per_type": "own"`
    if spy_ids(spec):
        return None  # schema 27: a hand that fills from captures or holds a spy gives one back
    if any(isinstance(npc, dict) and "immune" in npc for npc in rules.get("npcs") or ()):
        return None  # an immune type that can reach a hand is refused by name
    taken = {str(piece.get("id")) for piece in pieces(spec)}
    spare = [pid for pid in _SPARE_IDS if pid not in taken]
    if not spare:
        return None
    set_stone(spec, spare[0])
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside majority control.
    if majority_ids(spec):
        return None
    if purse_rule(spec) is not None:
        return None  # TC5 (`docs/plan-turncoat.md` §5.9): who is paid for a blast's victims
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (a blast)
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
            return "every capture explodes again, whoever makes it and whatever it takes"
        if not narrowed and rng.random() < 0.5:
            candidates = _non_royal(spec)
            if not candidates:
                return None
            key = rng.choice(("by", "of"))
            piece = rng.choice(candidates)
            explosion[key] = [str(piece["id"])]
            side = "made by" if key == "by" else "of"
            return f"only a capture {side} the {piece.get('name', piece['id'])} explodes"
        strip_rule(spec, "explode_on_capture")
        return "a capture no longer takes its neighbours with it"

    if rules.get("strike") or rules.get("contacts") or rules.get("npcs") or uses_sweep(spec):
        return None  # two rules cannot both say what happens to a taken piece
    drops = rules.get("drops")
    if isinstance(drops, dict) and drops.get("either_face") is True:
        return None  # nor can a blast and a hand that drops either face (schema 18e)
    if not capture_is_possible(spec):
        return None  # a blast where nothing is taken is refused (inert-rule case 4, Q110)
    if has_sights(spec):
        return None  # schema 30: a blast is refused beside a line of sight
    set_explosion(spec)
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
    return f"{piece['id']} no longer turns: each one points the way it was placed{pointed}"


# Schema 26 (Q918a): what a simultaneous game cannot be played beside and an operator may
# simply drop on the way in -- each is one rule a round has no use for. Anything else §4 of
# `docs/reference/rounds.md` refuses (a sweep, a moved portal, a strike that stays or
# chooses, a long turn) is a design of its own, so the operator does not apply there.
_DROPPED_FOR_ROUNDS = ("castling", "en_passant", "checkmate", "forbid_check", "check_limit")


def _play_at_once(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make both players move at once, or make them alternate again (schema 26).

    Into rounds: the checkmate win becomes royal capture (there is no check), and the
    rules a round has no use for are dropped (a wild piece's pounce among them). Out: the rule, every `beats` and the tie go,
    and the royal capture win stays, which an alternating game plays as it is.
    """
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside a piece nobody owns.
    if neutral_ids(spec):
        return None
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (nobody is the mover in a round)
    if purse_rule(spec) is not None:
        return None  # TC5 (`docs/plan-turncoat.md` §5.9): two buyers in one round
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
    rules = spec.setdefault("rules", {})
    if is_simultaneous(rules):
        del rules["simultaneous"]
        for piece in spec.get("pieces", []):
            if isinstance(piece, dict):
                piece.pop("beats", None)
                # Schema 27: a spy decides when rounds begin, so it goes with them; schema
                # 28's speed orders a round's moves, so it goes too.
                piece.pop("spy", None)
                piece.pop("speed", None)
        return "the players take turns again"
    del rng
    trial = copy.deepcopy(spec)
    trial_rules = trial["rules"]
    for key in _DROPPED_FOR_ROUNDS:
        trial_rules.pop(key, None)
    if trial_rules.get("royal_facing") == "forbidden":
        trial_rules.pop("royal_facing")
    if trial_rules.get("win", "checkmate") == "checkmate":
        trial_rules["win"] = "royal_capture"
    drops = trial_rules.get("drops")
    if isinstance(drops, dict):
        drops.pop("no_mate", None)
    # Generation 39 (N3b): a pounce is refused in rounds (no colour-blind order between two
    # players who both moved), so the wild pieces only walk there.
    for npc in trial_rules.get("npcs") or []:
        if isinstance(npc, dict):
            npc.pop("pounce", None)
            npc.pop("split", None)  # generation 41 (N3d): only beside a pounce
    trial_rules["simultaneous"] = {}
    # The rest of §4's refusals (a sweep, a moved portal, a strike that stays or chooses,
    # a long turn, a royal in a placement stock) are designs of their own: not applied.
    if any(error.startswith("rules.simultaneous beside") for error in validate_spec(trial)):
        return None
    spec.clear()
    spec.update(trial)
    return "both players now move at once, each choosing a move without seeing the other's"


def _set_a_collision(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Say who a piece beats when two pieces move onto one square, change it, or drop it
    (schema 26's `pieces[].beats`). Only in a game played in rounds, and only on a type that
    can collide: under speeds (schema 28, `docs/archive/plan-speed.md` §4.1) one that the other
    player can meet at its own speed, asked of the validator's own predicate
    (`schema.has_equal_speed_opponent`), so the two cannot disagree."""
    if not is_simultaneous(spec):
        return None
    pieces = [p for p in spec.get("pieces", []) if isinstance(p, dict)]
    if not pieces:
        return None
    ids = [p["id"] for p in pieces]
    stated = [p for p in pieces if isinstance(p.get("beats"), list)]
    if has_speeds(spec):
        pieces = [
            p for p in pieces
            if isinstance(p.get("id"), str) and has_equal_speed_opponent(spec, p["id"])
        ]
        if not pieces and not stated:
            return None
    if stated and (not pieces or rng.random() < 0.5):
        piece = rng.choice(stated)
        piece.pop("beats")
        return f"{piece['id']} beats in a collision what it may capture again"
    piece = rng.choice(pieces)
    default = beats_of({key: value for key, value in piece.items() if key != "beats"}, ids)
    choices = [
        sorted(set(ids) - {other}) for other in ids
    ] + [[]] + [[other] for other in ids]
    choices = [beats for beats in choices if frozenset(beats) != default and beats != piece.get("beats")]
    if not choices:
        return None
    piece["beats"] = rng.choice(choices)
    beaten = ", ".join(piece["beats"]) or "nothing"
    return f"in a collision {piece['id']} now beats {beaten}"


def _break_the_tie(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Decide a collision tie by the initiative, promise there is none, or eliminate both
    again (schema 26's `rules.simultaneous.tie`). Not entered under speeds where no type of
    one player shares a speed with a type of the other (schema 28, `docs/archive/plan-speed.md`
    §4.1): nothing collides there, and the validator refuses a tie, by the same predicate
    (`schema._any_equal_speed_meeting`)."""
    del rng
    if not is_simultaneous(spec):
        return None
    rule = spec["rules"]["simultaneous"]
    if "tie" in rule:
        del rule["tie"]
        return "a tied collision eliminates both pieces again"
    if has_speeds(spec) and not _any_equal_speed_meeting(spec):
        return None
    if simultaneous_tie_pair(spec) is None:
        rule["tie"] = "impossible"
        return "no collision can tie, and the file now says so"
    rule["tie"] = "initiative"
    return "a tied collision goes to the player holding the initiative, which then passes"


# Schema 27 (Q942a, `docs/archive/plan-round-chase.md` §3): the switches against a royal chased
# round after round, each with the one value that turns it on. The third, the round's rest,
# is schema 29's `pieces[].rest` (Q983a): `rest` here turns on `"rest": 1` on every piece a
# player moves, which is the old switch, and off takes every rest away.
_CHASE_SWITCHES = {"swap": "collision", "both": "score"}
_CHASE_WORDS = {
    "rest": (
        "every piece rests for its player's next turn after it moves",
        "no piece rests any more: a piece may move round after round again",
    ),
    "swap": ("two pieces that swap squares now meet as a collision", "swapping pieces pass again"),
    "both": (
        "an ending both players reach in one round goes to the player ahead",
        "an ending both players reach in one round is a draw again",
    ),
}


def _curb_the_chase(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Turn one of the chase remedies on or off: schema 27's `rules.simultaneous.swap` and
    `.both`, and the rest (schema 29's `pieces[].rest`, the old round switch): on, every
    piece a player moves rests 1; off, no piece rests. Only in a game played in rounds;
    `both` brings a material count when the game has no scoring rule, and leaves it when the
    switch goes."""
    if not is_simultaneous(spec):
        return None
    rules = spec["rules"]
    rule = rules["simultaneous"]
    stated = [key for key in _CHASE_SWITCHES if key in rule]
    absent = [key for key in _CHASE_SWITCHES if key not in rule]
    (stated if has_rests(spec) else absent).append("rest")
    if stated and (not absent or rng.random() < 0.5):
        key = rng.choice(stated)
        if key == "rest":
            for piece in pieces(spec):
                piece.pop("rest", None)
        else:
            del rule[key]
        return _CHASE_WORDS[key][1]
    for key in rng.sample(absent, len(absent)):
        trial = copy.deepcopy(spec)
        if key == "rest":
            for piece in _restable(trial):
                piece["rest"] = 1
        else:
            trial["rules"]["simultaneous"][key] = _CHASE_SWITCHES[key]
        if key == "both" and not isinstance(trial["rules"].get("scoring"), dict):
            trial["rules"]["scoring"] = {"count": "material"}
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        return _CHASE_WORDS[key][0]
    return None


def _restable(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The piece types a rest may be written on (schema 29, `docs/plan-rest.md` §3.1): a type
    a player moves, so never a wild piece, and one with a move or a stock."""
    found = spec.get("rules")
    rules: dict[str, Any] = found if isinstance(found, dict) else {}
    skipped = {
        npc.get("piece") for npc in (rules.get("npcs") or []) if isinstance(npc, dict)
    }
    return [
        piece for piece in pieces(spec)
        if isinstance(piece.get("id"), str) and piece["id"] not in skipped
        and (piece.get("moves") or piece_has_stock(rules, piece["id"]))
    ]


def _tire_a_piece(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make a piece type rest after it moves, rest longer or shorter, or never rest (schema 29,
    `docs/plan-rest.md` §5.7), in any game, alternating or in rounds. A type that never rests
    is made to rest 1; one that rests is moved a turn up or down (0 takes the key away, and
    none goes past `REST_MAX`). Each child must pass the validator, so a game of long turns,
    where a rest is refused, gets none."""
    if turn_is_long(spec.get("rules")):
        return None
    candidates = _restable(spec)
    for piece in rng.sample(candidates, len(candidates)):
        trial = copy.deepcopy(spec)
        target = next(p for p in pieces(trial) if p.get("id") == piece["id"])
        now = piece_rests(trial).get(piece["id"], 0)
        new = 1 if now == 0 else now + rng.choice((-1, 1))
        if new > REST_MAX:
            new = now - 1
        if new <= 0:
            target.pop("rest", None)
        else:
            target["rest"] = new
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        name = str(piece.get("name", piece["id"]))
        if new <= 0:
            return f"the {name} no longer rests after it moves"
        turns = "turn" if new == 1 else f"{new} turns"
        return f"the {name} now rests for its player's next {turns} after it moves"
    return None


# Schema 30 (Q999a, `docs/archive/plan-passive.md` §4.3): lines of sight. Whose pieces each effect
# touches when an operator casts it: an effect that restricts is cast on the enemy, a shield
# on its own side. Q1007c (`docs/archive/plan-schema-31.md` §3.8): a command on the enemy.
_GAZE_WHOSE = {"freeze": "enemy", "disarm": "enemy", "shield": "own", "command": "enemy"}


def _shields(spec: dict[str, Any]) -> bool:
    """Whether a line of sight shields (schema 30), beside which a sweep and a wild piece are
    refused."""
    return any(
        sight.effect == "shield" for sights in piece_sights(spec).values() for sight in sights
    )


def _commands(spec: dict[str, Any]) -> bool:
    """Whether a line of sight commands (Q1007c), beside which rounds, a reach, a territory, a
    strike, a contact, a rest and a sweep are refused (`docs/archive/plan-schema-31.md` §3.6)."""
    return any(
        sight.effect == "command" for sights in piece_sights(spec).values() for sight in sights
    )


def _turncoats(spec: dict[str, Any]) -> bool:
    """Whether a command may make its pieces take their own side (Q1007c's `captures_own`),
    beside which a hand filled from captures is refused."""
    return any(sight.captures_own for sights in piece_sights(spec).values() for sight in sights)


def _sight_names(spec: dict[str, Any]) -> dict[str, str]:
    """Each piece id's name, as `describe.sight_sentence` says the types a sight touches."""
    return {
        piece["id"]: str(piece.get("name", piece["id"]))
        for piece in pieces(spec) if isinstance(piece.get("id"), str)
    }


def _gaze_words(spec: dict[str, Any], entry: dict[str, Any]) -> str | None:
    """One sight entry in `describe`'s words, or ``None`` for one `piece_sights` cannot read."""
    from chess_variant_sandbox.spec.describe import sight_sentence

    read = _sight_entry(entry)
    if read is None:
        return None
    hexy = (spec.get("board") or {}).get("lattice") == "hex"
    return sight_sentence(read, _sight_names(spec), hexy)


def _gaze_of(move: dict[str, Any]) -> dict[str, Any] | None:
    """A slide-like reading of one capture-capable move's geometry, as a sight's keys: its
    vectors and symmetry, its range (1 for a leap or a lame leap, which is one jump per
    vector) and its `min_range`. ``None`` for a move that may not capture."""
    if move.get("mode", "both") == "move" and move.get("kind") != "sweep":
        return None
    vectors = move.get("vectors")
    if not isinstance(vectors, list) or not vectors:
        return None
    kind = move.get("kind")
    gaze: dict[str, Any] = {"vectors": copy.deepcopy(vectors)}
    if move.get("symmetry", "none") != "none":
        gaze["symmetry"] = move["symmetry"]
    if kind in ("leap", "lame"):
        gaze["range"] = 1
    else:
        reach = move.get("range", DEFAULT_RANGE.get(str(kind), 0))
        if reach:
            gaze["range"] = reach
        if move.get("min_range", 1) != 1 and (not reach or move["min_range"] <= reach):
            gaze["min_range"] = move["min_range"]
    return gaze


def _gazers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The piece types a sight may be written on: a player's type, never a wild piece
    (`docs/archive/plan-passive.md` §2.3)."""
    found = spec.get("rules")
    rules: dict[str, Any] = found if isinstance(found, dict) else {}
    skipped = {
        npc.get("piece") for npc in (rules.get("npcs") or []) if isinstance(npc, dict)
    }
    return [
        piece for piece in pieces(spec)
        if isinstance(piece.get("id"), str) and piece["id"] not in skipped
    ]


def _cast_a_gaze(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Give a piece type a line of sight copied from its own capture geometry (schema 30,
    `docs/archive/plan-passive.md` §4.3): a slide-like reading of one of its capture-capable moves,
    a ray, with an effect drawn from the four -- `enemy` for a freeze, a disarm or a command
    (Q1007c), `own` for a shield -- touching every type but, for a command, the royals. Each child must pass the validator, so a type whose
    gaze could never bind is passed over, a shield is not drawn beside a sweep, and nothing
    is drawn beside portals or an explosion, where a sight is refused. A shield beside wild
    pieces is allowed since generation 37 (a wild piece never takes a shielded piece)."""
    found = spec.get("rules")
    rules: dict[str, Any] = found if isinstance(found, dict) else {}
    if rules.get("portals") or rules.get("explode_on_capture"):
        return None
    effects = list(SIGHT_EFFECTS)
    if uses_sweep(spec):
        effects.remove("shield")
    candidates = [
        piece for piece in _gazers(spec)
        if len(piece.get("sight") or []) < SIGHT_MAX
        and any(_gaze_of(move) for move in piece.get("moves") or [] if isinstance(move, dict))
    ]
    for piece in rng.sample(candidates, len(candidates)):
        gazes = [
            gaze for move in piece.get("moves") or [] if isinstance(move, dict)
            for gaze in [_gaze_of(move)] if gaze is not None
        ]
        for effect in rng.sample(effects, len(effects)):
            entry = {**rng.choice(gazes), "effect": effect, "whose": _GAZE_WHOSE[effect]}
            # A gaze longer than a step is a ray or an aura, which sees past pieces, at even
            # odds, so a game can enter `through` in one edit as well as by `widen-a-gaze`.
            if entry.get("range", 0) != 1 and rng.random() < 0.5:
                entry["through"] = True
            # Q1007c: a command lets its pieces be made to take their own side at even odds,
            # so a game can enter `captures_own` in one edit as well as by `turn-the-gaze`.
            if effect == "command" and rng.random() < 0.5:
                entry["captures_own"] = True
            if entry in (piece.get("sight") or []):
                continue
            trial = copy.deepcopy(spec)
            target = next(p for p in pieces(trial) if p.get("id") == piece["id"])
            target.setdefault("sight", []).append(entry)
            if validate_spec(trial):
                continue
            spec.clear()
            spec.update(trial)
            words = _gaze_words(spec, entry)
            return f"the {piece.get('name', piece['id'])} now {words}"
    return None


def _blind_a_piece(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Take one line of sight away from a piece type (schema 30), and the type's `sight` with
    its last line."""
    seeing = [piece for piece in pieces(spec) if isinstance(piece.get("sight"), list)]
    if not seeing:
        return None
    piece = rng.choice(seeing)
    index = rng.randrange(len(piece["sight"]))
    words = _gaze_words(spec, piece["sight"][index])
    del piece["sight"][index]
    if not piece["sight"]:
        piece.pop("sight")
    name = piece.get("name", piece["id"])
    return f"the {name} no longer {words}" if words else f"the {name} sees less"


def _widen_a_gaze(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make one line of sight an aura that sees through pieces, or a ray that stops at the
    first again (schema 30's `through`). Only a line longer than one step, where the two
    differ: a one-step line has nothing between to see through."""
    lines = [
        (piece, entry) for piece in pieces(spec) if isinstance(piece.get("sight"), list)
        for entry in piece["sight"]
        if isinstance(entry, dict) and entry.get("range", 0) != 1
    ]
    if not lines:
        return None
    for piece, entry in rng.sample(lines, len(lines)):
        trial = copy.deepcopy(spec)
        target = next(p for p in pieces(trial) if p.get("id") == piece["id"])
        twin = target["sight"][piece["sight"].index(entry)]
        if twin.pop("through", None) is None:
            twin["through"] = True
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        name = piece.get("name", piece["id"])
        if twin.get("through"):
            return f"the {name}'s sight now passes through pieces: it {_gaze_words(spec, twin)}"
        return f"the {name}'s sight now stops at the first piece: it {_gaze_words(spec, twin)}"
    return None


def _turn_the_gaze(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Let one command make its pieces take their own side, or stop it doing so again (Q1007c's
    `captures_own`, `docs/archive/plan-schema-31.md` §3.3, §3.8). Only a command sight; each child must
    pass the validator, so a turncoat capture nothing could make, or one beside a capture hand,
    is not drawn."""
    lines = [
        (piece, entry) for piece in pieces(spec) if isinstance(piece.get("sight"), list)
        for entry in piece["sight"]
        if isinstance(entry, dict) and entry.get("effect") == "command"
    ]
    for piece, entry in rng.sample(lines, len(lines)):
        trial = copy.deepcopy(spec)
        target = next(p for p in pieces(trial) if p.get("id") == piece["id"])
        twin = target["sight"][piece["sight"].index(entry)]
        if twin.pop("captures_own", None) is not True:
            twin["captures_own"] = True
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        name = piece.get("name", piece["id"])
        return f"the {name} now {_gaze_words(spec, twin)}"
    return None


def _set_a_speed(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Give a piece type a speed, change one, or take the speeds away (schema 28,
    `docs/archive/plan-speed.md` §4.3). Only in a game played in rounds. In: a type at 0 is made
    faster than every other (one rank above the top), or one with a speed is moved a rank;
    the ranks are then closed up so none is skipped (`speed_ranks_in_use`). Out: every speed
    goes, and the round is today's again. Each child must pass the validator, so speeds that
    can never order a round, or a `beats`/`tie` they would leave inert, are not drawn."""
    if not is_simultaneous(spec):
        return None
    pieces = [p for p in spec.get("pieces", []) if isinstance(p, dict) and isinstance(p.get("id"), str)]
    if has_speeds(pieces) and rng.random() < 0.3:
        for piece in pieces:
            piece.pop("speed", None)
        return "every piece moves at one speed again"
    skipped = {
        npc.get("piece") for npc in (spec.get("rules", {}).get("npcs") or []) if isinstance(npc, dict)
    }
    movers = [p for p in pieces if p["id"] not in skipped]
    for piece in rng.sample(movers, len(movers)):
        trial = copy.deepcopy(spec)
        target = next(p for p in trial["pieces"] if isinstance(p, dict) and p.get("id") == piece["id"])
        speeds = piece_speeds(trial)
        now = speeds.get(piece["id"], 0)
        top = max((speeds[p["id"]] for p in movers), default=0)
        new = top + 1 if now == 0 else now + rng.choice((-1, 1))
        if new <= 0:
            target.pop("speed", None)
        else:
            target["speed"] = new
        _close_up_speeds(trial, skipped)
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        name = str(piece.get("name", piece["id"]))
        final = piece_speeds(spec).get(piece["id"], 0)
        if final == 0:
            return f"the {name} is among the slowest pieces again"
        return f"the {name} now has speed {final}: in a round it moves before slower pieces"
    return None


def _close_up_speeds(spec: dict[str, Any], skipped: set[Any]) -> None:
    """Renumber the speeds in use to 0, 1, 2, ... with none skipped, keeping their order
    (schema 28's ranks); a type that lands on 0 loses its key, and no speed is left on a
    type no player moves."""
    pieces = [p for p in spec.get("pieces", []) if isinstance(p, dict)]
    speeds = piece_speeds(pieces)
    ranks = sorted({speed for pid, speed in speeds.items() if pid not in skipped})
    renumber = {old: new for new, old in enumerate(ranks)}
    for piece in pieces:
        if piece.get("id") in skipped:
            piece.pop("speed", None)
            continue
        new = renumber.get(speeds.get(str(piece.get("id")), 0), 0)
        if new:
            piece["speed"] = new
        else:
            piece.pop("speed", None)


def _plant_a_spy(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make a piece type both players start with a spy, or take the spy away (schema 27,
    `docs/plan-spy.md`). In: the game must be one in rounds, so an alternating game is
    first made one as `play-at-once` makes it (the checkmate win becomes royal capture);
    the piece is any non-royal type both armies hold whose flag the validator accepts.
    Out: every spy flag goes, and the rounds stay, played blind from the first move."""
    if spy_ids(spec):
        for piece in pieces(spec):
            piece.pop("spy", None)
        return "there is no spy any more: both players choose every move in secret"
    base = copy.deepcopy(spec)
    if not is_simultaneous(base) and _play_at_once(base, rng) is None:
        return None
    first, second = placed_ids(base)
    candidates = [
        piece["id"] for piece in base["pieces"]
        if isinstance(piece, dict) and not piece.get("royal") and piece.get("id") in first & second
    ]
    for pid in rng.sample(candidates, len(candidates)):
        trial = copy.deepcopy(base)
        for piece in trial["pieces"]:
            if piece.get("id") == pid:
                piece["spy"] = True
        if validate_spec(trial):
            continue
        spec.clear()
        spec.update(trial)
        name = next(str(p.get("name", pid)) for p in trial["pieces"] if p.get("id") == pid)
        return (
            f"the {name} is now a spy: the players take turns while both keep theirs, and "
            "once one is lost the player without a spy moves first, in the open"
        )
    return None


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
        return f"anything may take a wild {kind} again, not only the one type that could"
    if spec.get("rules", {}).get("drops"):
        return None  # an immune type that can reach a hand is refused by name
    # Generation 42 (N4b): the duck's type is named by `rules.duck` alone, and nobody owns it,
    # so it hunts nothing (N6's shepherds duck was the first library game to meet this).
    duck = duck_piece(spec.get("rules", {}))
    hunters = [
        piece for piece in pieces(spec) if str(piece.get("id")) not in (kind, duck)
    ]
    if not hunters:
        return None
    hunter = str(rng.choice(hunters)["id"])
    for npc in family:
        npc["immune"] = [hunter]
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
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside a piece nobody owns.
    if neutral_ids(spec):
        return None
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
        return "the wild pieces are gone; every piece on the board belongs to somebody"
    if (
        rules.get("strike")
        or rules.get("explode_on_capture")
        or rules.get("turn")
        or rules.get("contacts")
        or rules.get("portals")
        or uses_sweep(spec)
        or _shields(spec)
    ):
        return None  # the validator refuses each of these beside a wild piece

    # T1: a wild type's moves are never read, so the validator refuses a turning path on one.
    candidates = [
        piece for piece in _non_royal(spec)
        if all(path_of(move) == "straight" for move in piece.get("moves", []))
    ]
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
        return f"the {names.get(entry.get('piece'))} promotes only on a move into its zone again"
    if not entries:
        return None
    live = sorted(pieces_that_can_leave_their_zone(spec))
    if not live:
        return None  # no promoting piece here can move out of its zone
    piece = rng.choice(live)
    next(entry for entry in entries if entry.get("piece") == piece)["on_leaving"] = True
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

    Schema 27's ``"own"`` (Q936) is the same edit where the two armies are made of
    different kinds: where ``true`` is refused because a side starts with none of a listed
    kind, ``"own"`` counts each kind only against a side that starts with it; and a game
    whose armies share fewer than two kinds gains one over every non-royal kind either
    army starts with. Neither draws from ``rng``, so a game the operator reached before
    draws what it drew.
    """
    rules = spec.setdefault("rules", {})
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and extinction.get("per_type") in (True, "own"):
        trial = copy.deepcopy(spec)
        trial["rules"]["extinction"].pop("per_type")
        if validate_spec(trial):
            return None
        spec.clear()
        spec.update(trial)
        listed = trial["rules"]["extinction"]["pieces"]
        return f"running out of {', '.join(listed)} counts them together again"
    if isinstance(extinction, dict):
        listed = extinction.get("pieces")
        if not isinstance(listed, list) or len(set(listed)) < 2:
            return None
        for reading in (True, "own"):
            trial = copy.deepcopy(spec)
            trial["rules"]["extinction"]["per_type"] = reading
            if not validate_spec(trial):
                spec.clear()
                spec.update(trial)
                own = " of their own" if reading == "own" else ""
                return f"a player who runs out of any one{own} of {', '.join(listed)} loses"
        return None
    if extinction:
        return None
    first, second = opening_counts(spec)
    kinds = sorted(
        piece["id"]
        for piece in _non_royal(spec)
        if first.get(piece["id"], 0) >= 1 and second.get(piece["id"], 0) >= 1
    )
    if len(kinds) < 2:
        either = sorted(
            piece["id"]
            for piece in _non_royal(spec)
            if first.get(piece["id"], 0) >= 1 or second.get(piece["id"], 0) >= 1
        )
        if len(either) < 2:
            return None
        trial = copy.deepcopy(spec)
        trial["rules"]["extinction"] = {"pieces": either, "per_type": "own"}
        if validate_spec(trial):
            return None
        spec.clear()
        spec.update(trial)
        return f"a player who runs out of any one of their own {', '.join(either)} loses"
    chosen = sorted(rng.sample(kinds, rng.randint(2, len(kinds))))
    trial = copy.deepcopy(spec)
    trial["rules"]["extinction"] = {"pieces": chosen, "per_type": True}
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
        if validate_spec(trial):
            return None
        spec.clear()
        spec.update(trial)
        return "a player holding a piece must place it before anything on the board moves"
    if rules.get("strike") or rules.get("contacts") or uses_sweep(spec):
        return None  # a hand is refused beside a rule that takes several pieces at once
    if _counts_own_kinds(rules):
        return None  # schema 27 (Q936): a hand is refused beside `"per_type": "own"`
    if spy_ids(spec):
        return None  # schema 27: a hand that fills from captures or holds a spy gives one back
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
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


# A group's move when somebody first may move its ends: one step any way, onto an empty
# square (a group's move omits ``mode``, which means ``"move"`` there; spelling 31).
_PORTAL_STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


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

    Schema 19 (Q771, `docs/archive/plan-portals.md`), all three of its names both ways, and schema
    20's ``no_straight_back`` (Q807) both ways. On a game with no portals a pair goes on two
    empty squares that are each other's image under the setup's mirror, so neither side
    starts with the better door; its group steps one square (`moves`), and nobody moves it
    yet. On a game with portals one draw lets either player
    move a pair's two ends (or stops it, and with it any bar on going straight back), one
    makes a pair one way for each player (or two way again), one bars a moved portal from
    going straight back (letting either player move it, if nobody could) or lifts the bar,
    and one takes every portal away. Every branch is checked against the validator and drawn again rather
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
                pair["moves"] = pair.get("moves") or [copy.deepcopy(_PORTAL_STEP)]
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
                    pair["moves"] = pair.get("moves") or [copy.deepcopy(_PORTAL_STEP)]
                    said = (
                        f"either player may now spend a turn moving a portal of pair "
                        f"{number} one step onto an empty square, but not straight back on "
                        "the very next move"
                    )
        else:
            del trial_rules["portals"]
            said = "the portals are gone"
    else:
        board = trial.get("board", {})
        if board.get("lattice") == "hex" or any("region" in p for p in pieces(trial)):
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

        pair = {
            "moves": [copy.deepcopy(_PORTAL_STEP)],
            "orange": name_of(orange), "blue": name_of(blue),
        }
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
    pair: dict[str, Any] = {"orange": first["at"], "blue": second["at"]}
    if group.get("moves"):
        pair["moves"] = group["moves"]
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
    """`ring-the-portals` and `funnel-the-portals` (schema 21, `docs/archive/plan-portals-2.md` §2): a
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
        group: dict[str, Any] = {"ends": ends}
        if pair.get("moves"):
            group = {"moves": pair["moves"], **group}
        if pair.get("no_straight_back"):
            group["no_straight_back"] = True
        if pair.get("kind") == "fold":
            group["kind"] = "fold"
        held[index] = group
    else:
        return None
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
    and ``enter``, `docs/archive/plan-portals-2.md` §2.3).

    A filter names one piece that can move, drawn at random, so the end is a door to it
    and a solid block to every other. Checked against the validator and drawn again
    rather than handed back refused.
    """
    rules = spec.get("rules") or {}
    if not isinstance(rules.get("portals"), list) or not rules["portals"]:
        return None
    trial = copy.deepcopy(spec)
    held = trial["rules"]["portals"]
    movers = sorted(str(p["id"]) for p in pieces(trial) if p.get("moves"))
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
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _fold_a_portal(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make one portal group fold portals, or momentum portals again (schema 24's ``kind``,
    `docs/archive/plan-portals-2.md` §3R).

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

    A1 (`docs/archive/plan-pieces-moves.md` §6.5) adds the strike's ``by`` both ways. *Arm one
    piece*: a standing rifle strike is given to one capturing type at random instead of
    being taken off (half the draws), and a new strike may be rifle given to one type (the
    archers). *Arm everyone*: a strike with ``by`` loses it, every piece striking again.
    """
    # TC3 (`docs/plan-turncoat.md` §4.5): the rule this adds is refused beside a piece nobody owns.
    if neutral_ids(spec):
        return None
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (a strike)
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
    rules = spec.setdefault("rules", {})
    strike = rules.get("strike")
    explosion = rules.get("explode_on_capture")
    if isinstance(explosion, dict) and explosion.get("radius") == 0:
        explosion["radius"] = 1
        return "a capture takes its neighbours with it again, not only the two pieces"
    if isinstance(strike, dict) and "by" in strike:
        strike.pop("by")
        return "a capture by any piece strikes again, not only one by the listed pieces"
    if isinstance(strike, dict) and "switches_sides" in strike:
        strike.pop("switches_sides")
        if not strike_is_live(strike):
            strip_rule(spec, "strike")
        return "a piece that captures stays on its own side again"
    if (
        isinstance(strike, dict)
        and strike.get("effect", "capture") == "capture"
        and strike.get("position") in ("stay", "home")
    ):
        if strike.get("position") == "stay" and rng.random() < 0.5:
            armed = _arm_one_piece(spec, strike, rng)
            if armed is not None:
                return armed
        strip_rule(spec, "strike")
        return "a capture is an ordinary capture again: the taker moves in, the taken is gone"

    if isinstance(explosion, dict) and explosion:
        explosion["radius"] = 0
        explosion.pop("immune", None)
        return "a capture destroys the taker with what it took, and nothing beside them"
    if spy_ids(spec):
        return None  # schema 27: a strike may switch a spy to the other side, which a spy game refuses
    if (
        explosion or rules.get("drops") or rules.get("npcs") or rules.get("contacts")
        or uses_sweep(spec)
    ):
        return None  # two rules cannot both say what happens to a taken piece
    # I1: a piece that never moves never captures, so it never switches.
    switching = sorted(str(piece["id"]) for piece in _non_royal(spec) if piece.get("moves"))
    if isinstance(strike, dict) and strike:
        if not strike_is_live(strike) or not switching:
            return None
        strike["switches_sides"] = switching
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
    # Schema 26: a strike that stays is refused beside rounds, so it is not offered there.
    stays = () if is_simultaneous(rules) else ("stay", "shoot")
    choice = rng.choice((*stays, "switch") if only_lost else (*stays, "home", "switch"))
    if choice == "shoot":
        rule["position"] = "stay"
        return _arm_one_piece(spec, rule, rng, adding=True)
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
    return said


def _arm_one_piece(
    spec: dict[str, Any], strike: dict[str, Any], rng: random.Random, adding: bool = False
) -> str | None:
    """Give a rifle strike to one capturing type (A1, `docs/archive/plan-pieces-moves.md` §6.5).

    ``strike`` is the standing rule, or with ``adding`` the one about to be added. One of
    the types whose captures it reaches is drawn, and only where there are two or more, so
    the list never names every one (the strike without ``by``). Checked against the
    validator; ``None`` where the result is refused, with the spec left alone.
    """
    listed = pieces(spec)
    firing = strikers(listed, strike)
    if not firing or len(firing) < 2:
        return None
    shooter = rng.choice(sorted(firing))
    trial = copy.deepcopy(spec)
    trial.setdefault("rules", {})["strike"] = {**copy.deepcopy(strike), "by": [shooter]}
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    name = next(
        (str(p.get("name", p["id"])) for p in listed if p.get("id") == shooter), shooter
    )
    return (
        f"the {name} shoots: its capture removes what it takes and it stays where it stood"
        + ("" if adding else "; every other piece captures as usual again")
    )


def _sweep_a_slider(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Turn one slide into a sweep, or a sweep back into a slide.

    Schema 13's move kind (Q21), which no other operator writes. The descriptor keeps its
    vectors, symmetry, range and condition, so the piece reaches the same squares and only
    what stands between stops mattering. A promoting piece is never offered, and neither
    is a spec holding a rule the validator refuses beside a sweep: both would make every
    draw a rejection.
    """
    if duck_piece(spec.get("rules")) is not None:
        return None  # generation 42 (N4b): refused beside the duck (a sweep)
    if _commands(spec):
        return None  # Q1007c: refused beside a command (docs/archive/plan-schema-31.md §3.6)
    if _has_portals(spec):
        return None  # refused beside portals (schema 19)
    if is_simultaneous(spec) and not _movers(spec, "sweep"):
        return None  # schema 26: a sweep is refused beside rounds
    sweeps = _movers(spec, "sweep")
    if sweeps:
        piece, move = rng.choice(sweeps)
        move["kind"] = "slide"
        return (
            f"the {piece.get('name', piece['id'])} slides again and stops at the first "
            "piece in its way"
        )
    if _shields(spec):
        return None  # schema 30: a sweep is refused beside a shield
    rules = spec.get("rules", {})
    if sweep_conflicts(rules):
        return None
    promoting = promoting_piece_ids(rules)
    movers = [(p, m) for p, m in _movers(spec, "slide") if p.get("id") not in promoting]
    if not movers:
        return None
    piece, move = rng.choice(movers)
    move["kind"] = "sweep"
    move.pop("path", None)  # T1: a sweep removes what lies on a line
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
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


# Generation 37 (N2, `docs/archive/plan-npc.md` §4.2): the effects a wild piece's sight may have, and its
# one shape here, a ring of the eight squares round it that sees past pieces (an aura): a wild
# piece has no forward, so its sight must read the same for both players, and the ring does.
_WILD_EFFECTS = ("freeze", "disarm", "shield")
_WILD_RING = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1, "through": True}


def _wild_eyes(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Give a wild piece an aura, or take every wild aura away (generation 37, N2's sight on a
    wild type, `docs/archive/plan-npc.md` §4.2).

    In a game where a wild type sees, no wild type does any more: the edit a designer makes is
    "the wild pieces are only pieces again". Otherwise one wild type, chosen at random, freezes,
    disarms or shields either side's pieces on the eight squares round it.
    """
    rules = spec.get("rules") or {}
    npcs = rules.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return None
    wild = sorted(
        {npc["piece"] for npc in npcs if isinstance(npc, dict) and isinstance(npc.get("piece"), str)}
    )
    by_id = {piece.get("id"): piece for piece in pieces(spec)}
    seeing = [pid for pid in wild if isinstance((by_id.get(pid) or {}).get("sight"), list)]
    trial = copy.deepcopy(spec)
    trial_by_id = {piece.get("id"): piece for piece in pieces(trial)}
    if seeing:
        for pid in seeing:
            trial_by_id[pid].pop("sight", None)
        said = "the wild pieces no longer affect what stands beside them"
    else:
        choices = [pid for pid in wild if pid in trial_by_id]
        if not choices:
            return None
        pid = rng.choice(choices)
        effect = rng.choice(_WILD_EFFECTS)
        entry = {**copy.deepcopy(_WILD_RING), "effect": effect, "whose": "any"}
        trial_by_id[pid]["sight"] = [entry]
        name = trial_by_id[pid].get("name", pid)
        said = f"the wild {name} now {_gaze_words(trial, entry)}"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _bring_them_back(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make a wild piece come back once taken, or let it stay gone (generation 36, N1's
    ``npcs[].respawn``, `docs/archive/plan-npc.md` §4.1).

    In a game where some wild piece comes back, none does any more: the edit a designer
    makes is "taken wild pieces stay gone", not one route of several. Otherwise one entry,
    chosen at random, is given a clock of four to twelve phases (two to six moves of each
    player), the range a hunt is played in -- long enough that taking one is worth
    something, short enough that the board fills again within a game.
    """
    rules = spec.get("rules") or {}
    npcs = rules.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return None
    entries = [index for index, npc in enumerate(npcs) if isinstance(npc, dict)]
    if not entries:
        return None
    trial = copy.deepcopy(spec)
    if any(npc_respawn(npcs[index]) for index in entries):
        for npc in trial["rules"]["npcs"]:
            if isinstance(npc, dict):
                npc.pop("respawn", None)
        said = "a wild piece that is taken no longer comes back"
    else:
        npc = trial["rules"]["npcs"][rng.choice(entries)]
        kind = npc.get("piece")
        every = rng.choice((4, 6, 8, 12))
        npc["respawn"] = every
        said = f"a wild {kind} that is taken comes back to its start {every} moves later"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _pace_a_line(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Walk a wild piece's route as a line that turns back, or as a cycle again (generation
    38, N5's ``npcs[].turns_back``, `docs/archive/plan-npc.md` §4.7).

    In a game where some wild piece turns back, none does any more: the edit is "the walks
    go round again", not one route of several. Otherwise one entry with a route of two or
    more squares and no ``also_at``, chosen at random, paces its route as a line. The
    rule is refused where check is asked, so in a checkmate game (with no ``forbid_check``
    and no safe goal) the game becomes one won by taking the king, which is the one edit that
    lets the line in; the validator decides the rest.
    """
    rules = spec.get("rules") or {}
    npcs = rules.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return None
    trial = copy.deepcopy(spec)
    if any(npc_turns_back(npc) for npc in npcs):
        for npc in trial["rules"]["npcs"]:
            if isinstance(npc, dict):
                npc.pop("turns_back", None)
        said = "the wild pieces walk their routes round again instead of turning back"
    else:
        entries = [
            index for index, npc in enumerate(npcs)
            if isinstance(npc, dict) and isinstance(npc.get("route"), list)
            and len(npc["route"]) > 1 and "also_at" not in npc
        ]
        if not entries:
            return None
        npc = trial["rules"]["npcs"][rng.choice(entries)]
        npc["turns_back"] = True
        said = (
            f"the wild {npc.get('piece')} paces its route as a line, turning back at its ends "
            "and wherever it is blocked"
        )
        if trial["rules"].get("win", "checkmate") == "checkmate" and royal_id(trial) is not None:
            trial["rules"]["win"] = "royal_capture"
            said += ", and the game is now won by taking the royal piece rather than mating it"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _set_the_hounds(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Make one kind of wild piece pounce, or none (generation 39, N3b's ``npcs[].pounce``,
    `docs/archive/plan-npc.md` §4.3; generation 40, N3c's ``"attackers"``, §4.4; generation 41, N3d's
    ``split``, §4.5).

    In a game where some wild piece pounces, none does any more. Otherwise one wild type,
    chosen at random, pounces on every entry that walks it (entries of one type must agree),
    on its prey or on its attackers, chosen at random too (on its attackers, half the time it
    splits too, up to a cap drawn from its starts to ``SPLIT_MAX``),
    and the returns and lines refused beside a pounce are dropped from every entry, so the
    walks go round and a taken wild piece is gone; the validator decides the rest (a type whose
    moves read differently for the two players, rounds, ``forbid_check``).
    """
    rules = spec.get("rules") or {}
    npcs = rules.get("npcs")
    if not isinstance(npcs, list) or not npcs:
        return None
    trial = copy.deepcopy(spec)
    if any(npc_pounce(npc) is not None for npc in npcs):
        for npc in trial["rules"]["npcs"]:
            if isinstance(npc, dict):
                npc.pop("pounce", None)
                npc.pop("split", None)
        said = "the wild pieces no longer pounce on what they attack"
    else:
        kinds = sorted({
            npc["piece"] for npc in npcs if isinstance(npc, dict) and isinstance(npc.get("piece"), str)
        })
        if not kinds:
            return None
        kind = rng.choice(kinds)
        style = rng.choice(POUNCE_KINDS)
        starts = sum(
            npc_start_count(npc) for npc in npcs if isinstance(npc, dict) and npc.get("piece") == kind
        )
        cap = (
            rng.randint(max(SPLIT_MIN, starts), SPLIT_MAX)
            if style == "attackers" and starts <= SPLIT_MAX and rng.random() < 0.5 else 0
        )
        for npc in trial["rules"]["npcs"]:
            if not isinstance(npc, dict):
                continue
            npc.pop("respawn", None)
            npc.pop("turns_back", None)
            if npc.get("piece") == kind:
                npc["pounce"] = style
                if cap:
                    npc["split"] = cap
        said = (
            f"after each move a wild {kind} pounces on a piece of the mover's "
            f"{'it attacks' if style == 'prey' else 'that attacks it'}, before the wild pieces step"
            + (f", growing a copy on each attacker's square while fewer than {cap} stand" if cap else "")
        )
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


_DUCK_STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
_DUCK_SHIELD = {
    "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1, "through": True,
    "effect": "shield", "whose": "any",
}


def _loose_a_duck(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Add a duck, take it away, or change how it goes, where it starts or what it shields
    (generation 42, N4b's ``rules.duck``, `docs/archive/plan-npc.md` §4.6).

    In a game without a duck: a new moveless type (a spare id) becomes the classic duck, which
    the mover puts on any empty square after every move. The duck is refused beside check, so a
    checkmate game becomes one won by taking the royal piece (``forbid_check`` and
    ``check_limit`` dropped), and an odd ply limit is made even; the validator decides the rest
    (rounds, a turn, portals, a blast, a strike, a sweep). In a game with one, one of four, at
    random: the duck goes (its type with it, since nothing else may name it); it walks by a
    king's step instead of going anywhere, or anywhere again; it starts on an empty square, or
    off the board again; its type shields the eight squares round it, or no longer does. A new
    duck walks, starts on a square or shields a quarter of the time each, so one edit reaches
    each of those from a game without a duck.
    """
    rules = spec.get("rules") if isinstance(spec.get("rules"), dict) else {}
    trial = copy.deepcopy(spec)
    trial_rules = trial.setdefault("rules", {})
    pid = duck_piece(rules)
    if pid is None:
        taken = {piece.get("id") for piece in pieces(spec)}
        spare = [letter for letter in "D" + _SPARE_IDS if letter not in taken]
        if not spare or royal_id(spec) is None and trial_rules.get("win") != "annihilation":
            return None
        pid = spare[0]
        trial["pieces"].append({"id": pid, "name": "duck", "value": 0, "moves": []})
        trial_rules["duck"] = {"piece": pid}
        said = "every turn ends with the mover putting a duck, which nothing takes, on any empty square"
        if trial_rules.get("win", "checkmate") == "checkmate":
            trial_rules["win"] = "royal_capture"
            trial_rules.pop("checkmate", None)
            said += ", and the game is now won by taking the royal piece rather than mating it"
        trial_rules.pop("forbid_check", None)
        trial_rules.pop("check_limit", None)
        plies = trial_rules.get("max_plies", 400)
        if isinstance(plies, int) and plies % 2:
            trial_rules["max_plies"] = plies + 1
        # A quarter of the time each, the new duck walks, starts on a square, or shields.
        added = trial["pieces"][-1]
        if rng.random() < 0.25:
            trial_rules["duck"]["goes"] = "by_moves"
            added["moves"] = [copy.deepcopy(_DUCK_STEP)]
            said += "; it walks a king's step instead of going anywhere"
        if rng.random() < 0.25:
            added["sight"] = [copy.deepcopy(_DUCK_SHIELD)]
            said += "; it shields the eight squares round it"
        if rng.random() < 0.25:
            board = trial.get("board") or {}
            width, height = board.get("width"), board.get("height")
            if isinstance(width, int) and isinstance(height, int):
                for square in rng.sample(range(width * height), width * height):
                    name = "abcdefghijklmnop"[square % width] + str(square // width + 1)
                    trial_rules["duck"]["start"] = name
                    if not validate_spec(trial):
                        said += f"; it starts on {name}"
                        break
                else:
                    trial_rules["duck"].pop("start")
    else:
        duck = trial_rules["duck"]
        piece = next(p for p in pieces(trial) if p.get("id") == pid)
        choice = rng.randrange(4)
        if choice == 0:
            del trial_rules["duck"]
            trial["pieces"] = [p for p in trial["pieces"] if p.get("id") != pid]
            said = "the duck is gone: a turn is one move again"
        elif choice == 1:
            if duck.get("goes") == "by_moves":
                duck.pop("goes")
                piece["moves"] = []
                said = "the duck goes to any empty square again"
            else:
                duck["goes"] = "by_moves"
                piece["moves"] = [copy.deepcopy(_DUCK_STEP)]
                said = "the duck walks a king's step instead of going anywhere"
        elif choice == 2:
            if "start" in duck:
                duck.pop("start")
                said = "the duck starts off the board again"
            else:
                board = trial.get("board") or {}
                width, height = board.get("width"), board.get("height")
                if not isinstance(width, int) or not isinstance(height, int):
                    return None
                squares = [
                    "abcdefghijklmnop"[square % width] + str(square // width + 1)
                    for square in range(width * height)
                ]
                rng.shuffle(squares)
                for name in squares:
                    duck["start"] = name
                    if not validate_spec(trial):
                        break
                else:
                    return None
                said = f"the duck starts on {duck['start']}"
        elif piece.get("sight"):
            piece.pop("sight")
            said = "the duck no longer shields the pieces round it"
        else:
            piece["sight"] = [copy.deepcopy(_DUCK_SHIELD)]
            said = "the duck shields the eight squares round it: nothing may take a piece there"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


_CROWD_STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
# TC3's object form: next to means the four orthogonal neighbours, not the king's eight.
_CROWD_CROSS = {"majority": {"vectors": [[1, 0]], "symmetry": "rotate4"}}


def _gather_a_crowd(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Put pieces nobody owns on the board, take them away, or change who may move them
    (TC3, ``setup.placements[].player: "nobody"`` and ``pieces[].control``,
    `docs/plan-turncoat.md` §4).

    In a game without one: a new type (a spare id) stepping as a king is placed for nobody on
    a random empty square in the half between the armies (mirrored, under a mirrored setup;
    stated twice, square and image, otherwise), and half the time the player with more pieces
    next to it may move it (``control: "majority"``; a quarter of those count only the four
    orthogonal neighbours). In a game with some, one of three, at random: they go (their
    placements, and a type nothing else places); majority control is given to one standing
    type or taken from every type; or a majority type's neighbours swap between the king's
    step and the orthogonal four. The validator decides the rest (rounds, wild pieces, the
    duck, portals, a strike, contacts, and beside majority a territory, a goal, a blast, a
    long turn).
    """
    trial = copy.deepcopy(spec)
    placed = neutral_ids(spec)
    setup = trial.get("setup") or {}
    if not placed:
        taken = {piece.get("id") for piece in pieces(spec)}
        spare = [letter for letter in "C" + _SPARE_IDS if letter not in taken]
        board = trial.get("board") or {}
        width, height = board.get("width"), board.get("height")
        if not spare or not isinstance(width, int) or not isinstance(height, int):
            return None
        pid = spare[0]
        added: dict[str, Any] = {
            "id": pid, "name": "condottiere", "value": 2, "moves": [copy.deepcopy(_CROWD_STEP)],
        }
        said = f"a {pid} that steps as a king stands on the board belonging to nobody"
        if rng.random() < 0.5:
            added["control"] = (
                copy.deepcopy(_CROWD_CROSS) if rng.random() < 0.25 else "majority"
            )
            said += (
                ", moved by whichever player has more pieces "
                + ("orthogonally next to it" if isinstance(added["control"], dict) else "next to it")
            )
        trial["pieces"].append(added)
        mirrored = setup.get("mirror", True) is not False
        orientation = str(setup.get("orientation", "mirror"))
        names = [
            "abcdefghijklmnop"[square % width] + str(square // width + 1)
            for square in range(width * (height // 2))
        ]
        rng.shuffle(names)
        row: dict[str, Any] = {"piece": pid, "player": "nobody", "squares": []}
        setup.setdefault("placements", []).append(row)
        for name in names:
            row["squares"] = [name]
            if not mirrored:
                coords = parse_square(name, width, height)
                if coords is None:
                    continue
                file, rank = reflect_square(*coords, width, height, orientation)
                image = "abcdefghijklmnop"[file] + str(rank + 1)
                if image != name:
                    row["squares"].append(image)
            if not validate_spec(trial):
                said += f", on {' and '.join(row['squares'])}" + (" and its mirror" if mirrored else "")
                break
        else:
            return None
    else:
        by_id = {piece.get("id"): piece for piece in pieces(trial)}
        held = sorted(pid for pid in placed if by_id.get(pid, {}).get("control", "stands") != "stands")
        choice = rng.randrange(3)
        if choice == 0:
            setup["placements"] = [
                row for row in setup.get("placements") or [] if not is_nobodys(row)
            ]
            still = {row.get("piece") for row in setup["placements"] if isinstance(row, dict)}
            for pid in sorted(placed):
                by_id[pid].pop("control", None)
            trial["pieces"] = [
                piece for piece in trial["pieces"]
                if piece.get("id") not in placed or piece.get("id") in still
                or not _unreferenced(trial, piece.get("id"))
            ]
            said = "nothing on the board belongs to nobody any more"
        elif choice == 1:
            if held:
                for pid in held:
                    by_id[pid].pop("control", None)
                said = "nobody may move the pieces nobody owns any more: they only stand"
            else:
                pid = rng.choice(sorted(placed))
                by_id[pid]["control"] = "majority"
                said = f"a {pid} nobody owns may be moved by whichever player has more pieces next to it"
        else:
            if not held:
                return None
            pid = rng.choice(held)
            if isinstance(by_id[pid]["control"], dict):
                by_id[pid]["control"] = "majority"
                said = f"a {pid} nobody owns is held by the majority of the eight squares round it"
            else:
                by_id[pid]["control"] = copy.deepcopy(_CROWD_CROSS)
                said = f"a {pid} nobody owns is held by the majority of its four orthogonal neighbours"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _unreferenced(spec: dict[str, Any], pid: Any) -> bool:
    """Whether nothing in ``spec`` but its own declaration names the piece ``pid``."""
    others = [piece for piece in pieces(spec) if piece.get("id") != pid]
    text = json.dumps({key: value for key, value in spec.items() if key != "pieces"}) + json.dumps(others)
    return f'"{pid}"' not in text


def _post_a_bounty(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Put a price on a piece's head, take it off, or move it (TC4, ``pieces[].bounty``,
    `docs/plan-turncoat.md` §5.1).

    In a game without one: a random non-royal type (never a wild type or the duck) is given a
    bounty of its declared value, rounded and kept to 1 to 20, and the clock is made to count
    points (``rules.scoring`` added with ``count: "points"``, or an existing count of points
    kept; a game whose clock counts something else is left alone). In a game with some, one of
    three, at random: every bounty goes (and a points count with nothing left to pay goes with
    it); one type's bounty is raised or lowered by one to three; or it moves to another type.
    The validator decides the rest.
    """
    trial = copy.deepcopy(spec)
    rules = trial.setdefault("rules", {})
    wild = {
        npc.get("piece") for npc in rules.get("npcs") or [] if isinstance(npc, dict)
    } | {duck_piece(rules)}
    candidates = [
        piece for piece in pieces(trial)
        if not piece.get("royal") and piece.get("id") not in wild
    ]
    paying = piece_bounties(trial)
    # TC5: under a purse a bounty is paid into it, so no points count is needed.
    purse = isinstance(rules.get("purse"), dict)
    if not paying:
        scoring = rules.get("scoring")
        if isinstance(scoring, dict) and scoring.get("count") != "points" and not purse:
            return None
        if not candidates:
            return None
        piece = rng.choice(candidates)
        value = piece.get("value", 1)
        worth = max(1, min(20, round(value if isinstance(value, (int, float)) else 1)))
        piece["bounty"] = worth
        said = f"taking a {piece['id']} pays its taker {worth}"
        if not isinstance(scoring, dict) and not purse:
            rules["scoring"] = {"count": "points"}
            said += ", and the points decide the game when the clock runs out"
    else:
        by_id = {piece.get("id"): piece for piece in pieces(trial)}
        choice = rng.randrange(3)
        if choice == 0:
            for pid in paying:
                by_id[pid].pop("bounty", None)
            said = "taking a piece pays nothing any more"
            # A points count left with nothing to pay is refused; it goes with the bounties.
            if validate_spec(trial) and (rules.get("scoring") or {}).get("count") == "points":
                rules.pop("scoring")
                said += ", and the clock counts no points"
        elif choice == 1:
            pid = rng.choice(sorted(paying))
            worth = paying[pid] + rng.choice((-3, -2, -1, 1, 2, 3))
            worth = max(1, min(20, worth))
            if worth == paying[pid]:
                return None
            by_id[pid]["bounty"] = worth
            said = f"taking a {pid} pays {worth} now"
        else:
            pid = rng.choice(sorted(paying))
            others = [piece for piece in candidates if piece.get("id") not in paying]
            if not others:
                return None
            other = rng.choice(others)
            other["bounty"] = by_id[pid].pop("bounty")
            said = f"the bounty on {pid} moves to {other['id']}"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _two_back_ranks(spec: dict[str, Any]) -> list[str]:
    """The first player's two back ranks by name, or ``[]`` on a board too small or malformed."""
    board = spec.get("board") or {}
    width, height = board.get("width"), board.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or height < 4:
        return []
    return [format_square(file, rank) for rank in (0, 1) for file in range(width)]


def _open_a_market(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Give the players a purse and a market, close it, or change it (TC5, ``rules.purse``,
    `docs/plan-turncoat.md` §5.4).

    In a game without one: one or two random non-royal types that move (never a spy, a wild
    type or the duck) go on sale at their declared value rounded (1 to 40), and each player
    starts with 0 to 4 coins, placing on the first rank or (one time in three) on the two back
    ranks. In a game with one, one of four, at random: the purse closes
    (the prices go with it, and a bounty left with no points count to pay into); a price
    moves by one or two; the opening coins move by one to three; or the region is named as
    the first two ranks, or, where one is named, dropped back to the first rank. The
    validator decides the rest.
    """
    trial = copy.deepcopy(spec)
    rules = trial.setdefault("rules", {})
    purse = rules.get("purse")
    by_id = {piece.get("id"): piece for piece in pieces(trial)}
    if not isinstance(purse, dict):
        wild = {
            npc.get("piece") for npc in rules.get("npcs") or [] if isinstance(npc, dict)
        } | {duck_piece(rules)}
        candidates = sorted(
            piece["id"] for piece in pieces(trial)
            if not piece.get("royal") and piece.get("spy") is not True and piece.get("moves")
            and piece.get("id") not in wild and isinstance(piece.get("id"), str)
        )
        if not candidates:
            return None
        sold = rng.sample(candidates, min(len(candidates), rng.choice((1, 2))))
        for pid in sold:
            value = by_id[pid].get("value", 1)
            by_id[pid]["price"] = max(
                1, min(PRICE_MAX, round(value if isinstance(value, (int, float)) else 1))
            )
        start = rng.randrange(5)
        rules["purse"] = {"start": start, "market": sold}
        said = (
            f"each player starts with {start} coins and may buy "
            + " or ".join(f"{pid} for {by_id[pid]['price']}" for pid in sold)
        )
        if rng.randrange(3) == 0:
            region = _two_back_ranks(trial)
            if region:
                rules["purse"]["region"] = region
                said += ", placing it on either of the two back ranks"
    else:
        choice = rng.randrange(4)
        market = [pid for pid in purse.get("market") or [] if pid in by_id]
        if choice == 0:
            rules.pop("purse")
            for piece in pieces(trial):
                piece.pop("price", None)
            said = "the market closes and nobody holds coins"
            scoring = rules.get("scoring")
            if piece_bounties(trial) and not (
                isinstance(scoring, dict) and scoring.get("count") == "points"
            ):
                for piece in pieces(trial):
                    piece.pop("bounty", None)
                said += ", and taking pays nothing"
        elif choice == 1:
            if not market:
                return None
            pid = rng.choice(market)
            old = by_id[pid].get("price", 1)
            price = max(1, min(PRICE_MAX, old + rng.choice((-2, -1, 1, 2))))
            if price == old:
                return None
            by_id[pid]["price"] = price
            said = f"a {pid} costs {price} now"
        elif choice == 2:
            old = purse.get("start", 0)
            start = max(0, min(PURSE_MAX, old + rng.choice((-3, -2, -1, 1, 2, 3))))
            if start == old:
                return None
            purse["start"] = start
            said = f"each player starts with {start} coins"
        elif "region" in purse:
            purse.pop("region")
            said = "bought pieces go on the back rank again"
        else:
            region = _two_back_ranks(trial)
            if not region:
                return None
            purse["region"] = region
            said = "bought pieces may go on either of the two back ranks"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _settle_the_purse(trial: dict[str, Any]) -> str:
    """After a hire or a bribe is taken away (TC6): drop the prices nothing charges any more,
    and a purse left with nothing to spend on, with a bounty left with nothing to pay into.
    Returns the words for what went, or ``""``."""
    rules = trial.setdefault("rules", {})
    purse = rules.get("purse")
    if not isinstance(purse, dict):
        return ""
    charged = set(purse_market(trial)) | hire_ids(trial)
    bribes = purse_bribes(trial)
    for piece in pieces(trial):
        if not bribes and piece.get("id") not in charged:
            piece.pop("price", None)
    if purse_market(trial) or hire_ids(trial) or bribes:
        return ""
    rules.pop("purse")
    for piece in pieces(trial):
        piece.pop("price", None)
    said = ", and nobody holds coins any more"
    scoring = rules.get("scoring")
    if piece_bounties(trial) and not (
        isinstance(scoring, dict) and scoring.get("count") == "points"
    ):
        for piece in pieces(trial):
            piece.pop("bounty", None)
        said += ", and taking pays nothing"
    return said


def _price_of(piece: dict[str, Any]) -> int:
    value = piece.get("value", 1)
    return max(1, min(PRICE_MAX, round(value if isinstance(value, (int, float)) else 1)))


def _hire_a_sword(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Put a mercenary up for hire, take hiring away, or change its price (TC6, a
    ``pieces[].price`` on a type placed for nobody, `docs/plan-turncoat.md` §5.5).

    In a game with nothing for hire: a type placed for nobody is priced at its value (1 to
    40), or, where nobody owns anything, a new king-stepping type (a spare id) is placed for
    nobody between the armies as `gather-a-crowd` places one and priced at 2; and a purse of 1
    to 4 coins is opened where there is none. In a game with some, one of two, at random:
    hiring goes (the prices on the pieces nobody owns, unless a market or a bribe still reads
    them, and a purse left with nothing to spend on); or one hired type's price moves by one
    or two. The validator decides the rest.
    """
    trial = copy.deepcopy(spec)
    rules = trial.setdefault("rules", {})
    hired = hire_ids(trial)
    by_id = {piece.get("id"): piece for piece in pieces(trial)}
    if not hired:
        placed = sorted(neutral_ids(trial))
        if placed:
            pid = rng.choice(placed)
            by_id[pid]["price"] = _price_of(by_id[pid])
            said = f"a {pid} nobody owns may be hired for {by_id[pid]['price']}"
        else:
            crowd = copy.deepcopy(trial)
            if _gather_a_crowd(crowd, random.Random(rng.randrange(1 << 30))) is None:
                return None
            new = [p for p in pieces(crowd) if p.get("id") not in by_id]
            if len(new) != 1:
                return None
            new[0].pop("control", None)
            new[0]["price"] = 2
            trial["pieces"] = crowd["pieces"]
            trial["setup"] = crowd["setup"]
            pid = new[0]["id"]
            said = f"a {pid} that steps as a king stands for nobody, for hire at 2"
        if not isinstance(rules.get("purse"), dict):
            start = rng.randrange(1, 5)
            rules["purse"] = {"start": start}
            said += f", and each player starts with {start} coins"
    elif rng.randrange(2) == 0:
        for pid in hired:
            if pid not in purse_market(trial) and not purse_bribes(trial):
                by_id[pid].pop("price", None)
        if hire_ids(trial):
            return None
        said = "nobody's pieces are not for hire any more" + _settle_the_purse(trial)
    else:
        pid = rng.choice(sorted(hired))
        old = by_id[pid]["price"]
        price = max(1, min(PRICE_MAX, old + rng.choice((-2, -1, 1, 2))))
        if price == old:
            return None
        by_id[pid]["price"] = price
        said = f"hiring a {pid} costs {price} now"
    if validate_spec(trial):
        return None
    spec.clear()
    spec.update(trial)
    return said


def _offer_a_bribe(spec: dict[str, Any], rng: random.Random) -> str | None:
    """Let a player buy an enemy piece next to one of theirs at twice its price, or stop it
    (TC6, ``rules.purse.bribe``, `docs/plan-turncoat.md` §5.5).

    In a game without bribes: a purse of 1 to 4 coins is opened where there is none, one
    non-royal type a player places that moves (never a spy, a wild type or the duck) is
    priced at its value where no placed type has a price, and bribes go on. In a game with
    them: bribes go off, with the prices nothing else reads and a purse left with nothing to
    spend on. The validator decides the rest.
    """
    trial = copy.deepcopy(spec)
    rules = trial.setdefault("rules", {})
    purse = rules.get("purse")
    if purse_bribes(trial):
        assert isinstance(purse, dict)
        purse.pop("bribe")
        said = "nobody may bribe an enemy piece any more" + _settle_the_purse(trial)
    else:
        said = "a player may bribe an enemy piece next to one of theirs at twice its price"
        if not isinstance(purse, dict):
            start = rng.randrange(1, 5)
            purse = rules["purse"] = {"start": start}
            said += f", each player starting with {start} coins"
        setup = trial.get("setup") or {}
        held = {
            row.get("piece") for row in setup.get("placements") or []
            if isinstance(row, dict) and not is_nobodys(row)
        }
        wild = {
            npc.get("piece") for npc in rules.get("npcs") or [] if isinstance(npc, dict)
        } | {duck_piece(rules)}
        if not set(piece_prices(trial)) & held:
            candidates = sorted(
                piece["id"] for piece in pieces(trial)
                if piece.get("id") in held and not piece.get("royal")
                and piece.get("spy") is not True and piece.get("moves")
                and piece.get("id") not in wild and isinstance(piece.get("id"), str)
            )
            if not candidates:
                return None
            pid = rng.choice(candidates)
            by_id = {piece.get("id"): piece for piece in pieces(trial)}
            by_id[pid]["price"] = _price_of(by_id[pid])
            said += f" (a {pid} for {2 * by_id[pid]['price']})"
        purse["bribe"] = True
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
    Operator("bring-them-back", _bring_them_back),
    Operator("wild-eyes", _wild_eyes),
    Operator("pace-a-line", _pace_a_line),
    Operator("set-the-hounds", _set_the_hounds),
    Operator("loose-a-duck", _loose_a_duck),
    Operator("gather-a-crowd", _gather_a_crowd),
    Operator("post-a-bounty", _post_a_bounty),
    Operator("open-a-market", _open_a_market),
    Operator("hire-a-sword", _hire_a_sword),
    Operator("offer-a-bribe", _offer_a_bribe),
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
    Operator("ride-a-leap", _ride_a_leap),
    Operator("still-a-piece", _still_a_piece),
    Operator("bend-a-line", _bend_a_line),
    Operator("straighten", _straighten),
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
    Operator("play-at-once", _play_at_once),
    Operator("set-a-collision", _set_a_collision),
    Operator("break-the-tie", _break_the_tie),
    Operator("sweep-a-slider", _sweep_a_slider),
    Operator("curb-the-chase", _curb_the_chase),
    Operator("plant-a-spy", _plant_a_spy),
    Operator("set-a-speed", _set_a_speed),
    Operator("tire-a-piece", _tire_a_piece),
    Operator("cast-a-gaze", _cast_a_gaze),
    Operator("blind-a-piece", _blind_a_piece),
    Operator("widen-a-gaze", _widen_a_gaze),
    Operator("turn-the-gaze", _turn_the_gaze),
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
