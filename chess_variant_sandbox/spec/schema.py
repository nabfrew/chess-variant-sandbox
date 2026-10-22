"""Rule-spec schema and validation.

This module is the trust boundary for LLM-proposed rules. A spec that passes
``validate_spec`` is guaranteed structurally sound enough for ``engine.compile`` to
consume without defensive checks of its own. Validation is deliberately hand-rolled
and stdlib-only: the engine has no third-party dependencies, and the set of legal
movement primitives *is* the design space, so it lives here in one readable table.

Unknown keys are rejected rather than ignored. A proposal that invents a field is a
proposal that believes the engine supports something it does not, and silently
dropping it would produce a game nobody asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from collections.abc import Set as AbstractSet
from math import gcd
from typing import TYPE_CHECKING, Any, NamedTuple

from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.reachability import (
    _in,
    _pieces_that_can_stand,
    _promotion_entries,
    _promotion_target_ids,
    _str_set,
)
from chess_variant_sandbox.spec.reachability import (
    pieces_a_player_can_hold as pieces_a_player_can_hold,
)

if TYPE_CHECKING:
    from chess_variant_sandbox.engine.compile import StrikeRule

SCHEMA_VERSION = 30

# Every version is accepted, and the number a spec declares is a claim about which
# feature set it uses rather than a date stamp. That is what keeps a bump cheap:
# ``schema_version`` is part of the mechanics hash, so rewriting the committed rulesets
# to say 3 would rehash every one of them and orphan the calibration runs under
# ``runs/``. A spec that declares 1 may not use anything added in 2 or 3, and the
# validator enforces that by name, so the version stays an honest description of what
# the rules need.
SUPPORTED_SCHEMA_VERSIONS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30,
)

# The movement primitive table. Extending the design space means adding a row here,
# giving it its meaning in engine/compile.py, walking its tables in rust/src/game.rs, and
# bumping SCHEMA_VERSION. Skill `new-move-primitive` lists every other surface it owes.
# "lame" is schema 2: a leap that can be blocked. The xiangqi horse and elephant are
# the reason -- they reach like leapers but a piece on the way stops them, which is
# what makes them weak enough to be interesting.
# "sweep" is schema 13: a line that jumps to any square on it and removes every piece it
# passes over, its own included (Q21).
MOVE_KINDS = ("slide", "leap", "hop", "lame", "sweep")

# What ``range`` means when a descriptor omits it. A leap is a single jump; a slide and
# a hop both run to the edge of the board unless told otherwise. Shared with the
# compiler so that a spec means the same thing to the validator and to the engine.
DEFAULT_RANGE = {"slide": 0, "hop": 0, "leap": 1, "lame": 1, "sweep": 0}

# How a move descriptor's base vectors are expanded into concrete directions.
#   none      - used as written (y is still flipped for the second player)
#   lateral   - (x, y) and (-x, y); left/right mirror, forward stays forward
#   rotate4   - four 90-degree rotations
#   dihedral8 - the full symmetry group of the square: 8 images, deduplicated
# Schema 18b adds the hexagon's two, accepted only on a hex lattice (and the square's three
# turns stay refused there): a hex step is written once and turned six ways.
#   rotate6    - six 60-degree turns in axial coordinates, (q, r) -> (-r, q + r)
#   dihedral12 - the full symmetry group of the hexagon: the six turns and their mirrors
SQUARE_SYMMETRIES = ("none", "lateral", "rotate4", "dihedral8")
HEX_SYMMETRIES = ("none", "rotate6", "dihedral12")
SYMMETRIES = (*SQUARE_SYMMETRIES, "rotate6", "dihedral12")

# The lattice a board's cells sit on (schema 14). `"hex"` adds no geometry. It fixes the
# convention a hex spec's vectors are read in -- a width by height rhombus in axial
# coordinates, file `q` and rank `r`, six neighbours `(+-1, 0) (0, +-1) (1, -1) (-1, 1)`
# -- so the validator can refuse the spellings that compile a plausible wrong hex game.
LATTICES = ("square", "hex")

# The board's outline (schema 16, Q281). `"grid"`, the default, is every cell of the width
# by height frame: a rectangle on the square lattice, a rhombus on the hex one.
# `"hexagon"` is the regular hexagon inside a hex rhombus, whose corners are *absent*
# (`absent_squares`): not cells, not walls, never named by a rule. It compiles to the
# rhombus cut by walls, and its three pairs of sides take `rules.wrap`'s pair keys.
SHAPES = ("grid", "hexagon")
_BOARD_KEYS = {"width", "height", "lattice", "shape"}

# What a move descriptor may do with the destination square.
MODES = ("move", "capture", "both")

STALEMATE_RESULTS = ("draw", "loss", "win")
# "none" is schema 2: the game has no built-in ending and is decided entirely by the
# rules the spec states -- which is what antichess needs, because there losing your
# last piece is a win and the built-in reading of that is the opposite.
WIN_CONDITIONS = ("checkmate", "royal_capture", "annihilation", "none")

# How a v2 ending resolves, always stated for the player the ending happens to: the one
# who reaches the square, runs out of the named piece, or is reduced to royalty alone.
ENDING_RESULTS = ("win", "loss", "draw")

# Whose result `rules.bare_royal` states, as every text that explains the key says it.
# The core decides for the side it found bared, so `"loss"` is shatranj's rule.
# `tests/test_schema_two.py` holds each text to this phrase and the phrase to the core.
BARE_ROYAL_HOLDER = "the player reduced to royalty alone"

# Everything added in schema 2. A spec declaring version 1 that uses any of these is
# rejected by name rather than silently ignored, so the declared version never
# overstates what the engine was asked to do.
V2_RULE_KEYS = frozenset(
    {
        "bare_royal",
        "check_limit",
        "reach",
        "extinction",
        "must_capture",
        "checkmate",
        "forbid_check",
        "royal_facing",
        "drops",
        "explode_on_capture",
        "forbid_immobility",
    }
)
V2_PROMOTION_KEYS = frozenset({"ranks", "mandatory", "to_by_player"})
V2_CASTLING_KEYS = frozenset({"king_file", "rook_file"})
V2_SETUP_KEYS = frozenset({"orientation"})
V2_MOVE_KEYS = frozenset({"screen_range"})
V2_CONDITION_KEYS = frozenset({"region"})
V2_PIECE_KEYS = frozenset({"region", "promoted_from"})

# Everything added in schema 3, which is one idea in two places: *what* a piece may
# capture, rather than only where it may move.
#
# ``captures`` is the capture matrix -- a per-piece list of the ids it is allowed to
# take. Absent means "everything", which is what makes the whole feature inert on every
# spec written before it existed. Per piece rather than a top-level matrix so that a
# mutation operator can edit one piece and so the permission sits beside the movement
# that produces the threat.
#
# ``screen`` says whose piece a hop may jump. The xiangqi cannon does not care; a game
# where pieces vault their own army does, and that is a design nobody could state.
V3_PIECE_KEYS = frozenset({"captures"})
V3_MOVE_KEYS = frozenset({"screen"})
V3_RULE_KEYS = frozenset({"scoring"})

# What a scoring rule counts when the clock runs out.
#
# All three are already on the state, which is the point: material value, a headcount,
# or how many of your own pieces are standing in a named static region. The last is
# "hold the centre", and it is a real approximation of territory with no mutable squares
# anywhere -- the expensive version of that is a whole stage of its own.
#
# ``points`` is schema 9: what a player has collected by taking wild pieces (``rules.npcs``),
# which is the one count that is not already on the board -- it is a tally on the state,
# and the reason the rule that pays it is also the rule that needs it.
SCORING_COUNTS = ("material", "pieces", "region", "territory", "points")

# The three clocks, and what the validator will accept for each: (low, high), inclusive.
#
# Named rather than written inline in `_validate_rules`, because two things outside this
# module have to *stay inside* these bounds and used to guess at them instead:
# `loop.operators._limit_the_game` scales a clock and `spec.transform.never_forced` pushes
# the no-progress clock out to the ply limit, and both wrote values the validator then
# refused -- silently, because a refused child looks exactly like an operator that had
# nothing to do. A producer that has to respect a bound reads it from the same place the
# refusal comes from, or the two drift apart the first time either moves.
#
# `repetition_draw` is the one clock whose value is not only a size: since schema 10 it
# may also say `REPETITION_NEVER`, which is not a number and is not clamped to one.
CLOCK_BOUNDS: dict[str, tuple[int, int]] = {
    "max_plies": (2, 10_000),
    "repetition_draw": (2, 10),
    "no_progress_draw": (2, 1_000),
}

# Whose piece may serve as a hop's screen.
SCREEN_OWNERS = ("any", "own", "enemy")

# Schema 4: the board itself keeps a record of who has been where.
#
# One key, and it is a rule rather than a piece property, because it changes what the
# *board* is: squares stop being interchangeable and start being worth something. That
# is a different kind of game -- the designer who invented the sequence in
# `docs/reference/rps-versions.md` reached for it the moment his capture game turned
# repetitive, and it is what "an irreversible process" looks like in a chess-shaped game.
V4_RULE_KEYS = frozenset({"territory", "turn"})

# Schema 5: a board with no edges.
#
# One key, and it belongs in ``rules`` rather than in ``board`` for the same reason
# ``territory`` does: ``complexity._RULE_COSTS`` is defined as what a player has to be
# *taught*, and "a piece that leaves the left edge comes back on the right" is a taught
# exception rather than a dimension. Putting it here also means the four surfaces a rule
# already has to reach -- the prose the judge reads, the JSON schema the model writes
# against, the exhaustive cost table, and a mechanical operator -- guard it for free.
V5_RULE_KEYS = frozenset({"wrap"})

# Schema 6: two rules that share a theme -- *a capture is not the only thing that can
# happen when two pieces meet, and not every square is a place two pieces can meet*.
#
# ``obstacles`` is a set of squares nothing may occupy. It is entirely static, so every
# question it raises is decidable at compile time and it dissolves into the per-square
# tables the way schema 5's seam did: a ray that meets one ends there, a leap whose
# target is one is never emitted. Nothing about it crosses the boundary at all.
#
# ``strike`` is what happens *instead of* removal when a piece is taken -- it may be
# demoted, promoted, left standing, or swapped with its attacker. Movegen does not
# change: a strike is generated as a capture, blocked like a capture, and permitted by
# the same ``captures`` matrix. Only ``make`` differs, which is why a strike ruleset's
# perft is exactly its capture ruleset's perft, at every depth.
V6_RULE_KEYS = frozenset({"obstacles", "strike"})

# Schema 8: a piece that has an orientation, and may spend its move turning.
#
# ``facing`` is a *piece* property rather than a rule, because it is a fact about one
# piece the way ``captures`` and ``region`` are: a spec can give it to a pawn and to
# nothing else, and the sentence a player has to be told is per piece. It reads the
# piece's own move vectors rotated, so it is a different reading of the same
# descriptors rather than a new descriptor -- which is why it costs the movement tables
# another *frame* and not another dimension.
V8_PIECE_KEYS = frozenset({"facing"})

# How many orientations a facing piece has. Two is forward and backward; four is the
# quarter turns. Three is not offered because the board is square-gridded and a third
# of a turn does not land on a vector.
FACING_AXES = (2, 4)

# The orientations, in quarter turns clockwise from the piece's own forward. Named
# rather than numbered because ``"initial": 2`` is a sentence nobody can read, and
# because the names are what ``describe.py`` says out loud.
FACING_NAMES = ("forward", "right", "backward", "left")

# ``turns`` is schema 25 (Q869, the rear pawn): ``false`` is a facing no move turns. The
# piece still reads its vectors through its orientation and a placement still sets it,
# but the core generates no rotation for it, so a pawn stood facing backward stays a
# pawn going backward. See ``_schema_twenty_five_uses`` and ``_check_fixed_facings``.
_FACING_KEYS = {"axes", "initial", "turns"}

# Schema 9: a piece that neither player owns.
#
# ``npcs`` is a list of *wild pieces*. Each is a piece of a declared type standing on the
# board from ply zero, owned by nobody, which after every player move steps to the next
# square of its ``route`` and eats whatever stands there that its type may capture --
# schema 3's matrix, not a second flag. Taking one is worth its ``bounty``, which
# ``scoring.count: "points"`` reads at the clocks. A wild piece does not move as its type
# moves: its ``moves`` are what it does in a player's hands, which under ``drops`` is a
# real case. Routes are cycles of distinct squares and no square is on two routes, so
# which square is *next* is a fact about the square alone and dissolves into one table.
#
# ``promote_on_entry`` is schema 9's second key (queue task Q13), and it is the user's
# answer to the first entry ``docs/bugs.md`` carried (now in
# ``docs/archive/bugs-resolved.md``): a piece that *enters play* inside
# its own promotion zone -- dropped there, or left standing there by a strike that turned
# it into a promoting type -- promotes at once, to the first piece its promotion rule
# lists, whether or not that rule is ``mandatory``. Off, which is every game before it, a
# pawn that arrives on the far rank without moving there simply stands, and a chess pawn
# on the far rank is a piece that can never move again. It rides one flag because the
# three ways in are one event: a piece appearing on a square without a move of its own to
# choose a promotion in. A piece that gets there *by moving* is the promotion rule's own
# business and is unchanged.
#
# One set for the whole version, so that the two keys share one version check.
# `_schema_nine_uses` is the one collector.
V9_RULE_KEYS = frozenset({"npcs", "promote_on_entry"})
# ``immune`` is schema 10 (Q78) and is documented at ``REPETITION_NEVER`` below.
_NPC_KEYS = {"piece", "route", "bounty", "mirror", "immune", "also_at"}

# Schema 19 (Q771, `docs/archive/plan-portals.md` §1): pairs of portals. A piece whose move ends on
# an end open to its player comes out one king's step beyond the other end, in the
# direction it was travelling (the sign of its vector). A pair's number is its place in the
# list, so numbers cannot clash or skip. Each end is moved by the players its ``moved_by``
# names, with its declared piece's moves, onto empty squares only; ``passage`` says, per
# player, which end that player's pieces may go in by. An end closed to a player is an
# uncapturable obstacle to that player's pieces.
#
# Schema 20 (Q807, `docs/archive/plan-portals.md` *Ferry's shuffle*): ``no_straight_back``, per pair.
# An end of a pair that sets it, once moved, may not be moved straight back onto the square
# it left on the very next move (a portal's ko). Absent is ``false``.
#
# Schema 21 (Q843a, `docs/archive/plan-portals-2.md` §2): ``rules.portals`` holds **groups**. A group
# is a pair, written as before (plus ``carries``, per end), or any other map of two to eight
# ends, written with ``ends``: each end has its square (``at``), the index of the end it
# leads to (``to``; absent, a way out only), whose pieces may go in by it (``enter``), who
# may move it (``moved_by``) and which pieces may go in by it (``carries``). One normaliser,
# `portal_groups`, reads both spellings into one list of ends with a destination each, and
# nothing past it knows a pair from any other group.
#
# Schema 22 (Q843c), revised at schema 24 (Q867b, `docs/archive/plan-portals-2.md` §3R): per group,
# ``kind``. Absent is today's **momentum** portal; ``"fold"`` is the second rule set: a fold
# is not a piece (a piece of either side may stand in it, and is taken there as anywhere),
# and a move that ends on an empty fold may stay there or go through, ending on the square
# of the end it leads to and taking an enemy piece there as any landing does; a move that
# passes over a fold is not changed. Written only as ``"fold"``: the default spelled out is
# one game under two hashes, so the validator refuses it (`_check_inert_rules`' argument).
_PORTAL_KEYS = {
    "piece", "orange", "blue", "moved_by", "passage", "carries", "no_straight_back", "kind",
}
_PORTAL_ENDS_KEYS = {"piece", "ends", "no_straight_back", "kind"}
PORTAL_KINDS = ("momentum", "fold")
_PORTAL_END_KEYS = {"at", "to", "enter", "moved_by", "carries"}
PORTAL_ENDS = ("orange", "blue")
PORTAL_PASSAGES = ("both_ways", "orange_to_blue", "blue_to_orange", "closed")
# At most four groups (the palette's four colours, the number on the glyph) and eight ends
# across them (the core's end arrays are eight wide).
MAX_PORTAL_GROUPS = 4
MAX_PORTAL_ENDS = 8
# The letters an ``ends`` group's ends are named by, in place order (``2a``, ``2b``, ...);
# a pair's keep ``o`` and ``b``, so the move text of every stored portal game stays true.
_END_LETTERS = "abcdefgh"


def portal_end_ids(groups: list[dict[str, Any]]) -> tuple[str, ...]:
    """The names the compiler gives each portal end's internal type, in group order.

    Group number, then the end's letter: ``o`` for a pair's orange end and ``b`` for its
    blue (``1o``, ``1b``, the move text's since N1), and ``a``, ``b``, ``c``, ... by place for
    an ``ends`` group (``2a``, ``2b``, ``2c``). ``groups`` is `portal_groups`' answer. A
    declared piece may not take one of these ids beside ``rules.portals``.
    """
    return tuple(
        f"{group['number']}{end['letter']}" for group in groups for end in group["ends"]
    )


def portal_open_ends(passage: Any) -> tuple[bool, bool]:
    """``(orange, blue)``: which ends a player's pieces may go in by, for one passage word.

    ``both_ways`` (and anything malformed, which is refused by name elsewhere) is both;
    ``orange_to_blue`` is only the orange end; ``closed`` is neither. Read by the normaliser,
    which turns a pair's ``passage`` into its two ends' ``enter``.
    """
    if passage == "orange_to_blue":
        return (True, False)
    if passage == "blue_to_orange":
        return (False, True)
    if passage == "closed":
        return (False, False)
    return (True, True)


def _players(value: Any) -> frozenset[int]:
    """A permissive read of a list of players: the well-formed ones."""
    return frozenset(
        player for player in _listed(value)
        if player in (0, 1) and not isinstance(player, bool)
    )


def _carried(value: Any) -> frozenset[str] | None:
    """A permissive read of a ``carries`` list: ``None`` (every piece) when absent."""
    if value is None:
        return None
    return frozenset(pid for pid in _listed(value) if isinstance(pid, str))


def portal_groups(rules: Any) -> list[dict[str, Any]]:
    """``rules.portals`` read permissively: the well-formed groups, in order (schema 21).

    The one reader of the spelling, shared by the validator, the compiler and every reader
    after them, so a group means one thing to all of them (`docs/archive/plan-portals-2.md` §2.2).
    Each group is ``{"number", "index", "piece", "no_straight_back", "kind", "spelling",
    "passage", "ends"}``; each end ``{"at", "to", "enter", "moved_by", "carries", "letter",
    "key"}``:

    * ``at`` -- its square's name, as written (the readers parse it against their board);
    * ``to`` -- the index in the group's ``ends`` of the end it leads to, or ``None`` for a
      way out only;
    * ``enter`` -- the players whose pieces may go in by it (empty where ``to`` is
      ``None``);
    * ``moved_by`` -- the players who may move it, as a turn;
    * ``carries`` -- the piece ids that may go in by it, or ``None`` for every piece;
    * ``letter`` -- its letter in its id (`portal_end_ids`), and ``key`` the place the
      validator's messages name it by (``orange``, ``blue``, ``ends[2]``).

    A pair is two ends, orange ``to`` blue and blue ``to`` orange, each ``enter`` read from
    ``passage`` per player. ``spelling`` and ``passage`` are kept for the one reader that
    writes the spelling back (the designer, `web/designing.py`); nothing that plays or says
    the rule reads them. ``kind`` is ``"momentum"``, today's portal, or ``"fold"`` (schema 22;
    anything else reads as momentum, and is refused by name elsewhere).
    """
    if not isinstance(rules, dict) or not isinstance(rules.get("portals"), list):
        return []
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(rules["portals"]):
        if not isinstance(group, dict):
            continue
        if "ends" in group:
            ends = _ends_spelled(group.get("ends"))
            spelling = "ends"
            passage: dict[int, Any] = {}
        else:
            ends, passage = _pair_spelled(group)
            spelling = "pair"
        groups.append({
            "number": len(groups) + 1,
            "index": index,
            "piece": group.get("piece"),
            "no_straight_back": group.get("no_straight_back") is True,
            "kind": "fold" if group.get("kind") == "fold" else "momentum",
            "spelling": spelling,
            "passage": passage,
            "ends": ends,
        })
    return groups


def _pair_spelled(pair: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """A pair's two ends, and its passage words per player."""
    moved_by = pair["moved_by"] if isinstance(pair.get("moved_by"), dict) else {}
    passage = pair["passage"] if isinstance(pair.get("passage"), dict) else {}
    carries = pair["carries"] if isinstance(pair.get("carries"), dict) else {}
    words = {player: passage.get(str(player), "both_ways") for player in (0, 1)}
    opened = [portal_open_ends(words[player]) for player in (0, 1)]
    ends = []
    for side, end in enumerate(PORTAL_ENDS):
        ends.append({
            "at": pair.get(end),
            "to": 1 - side,
            "enter": frozenset(player for player in (0, 1) if opened[player][side]),
            "moved_by": _players(moved_by.get(end)),
            "carries": _carried(carries.get(end)),
            "letter": end[0],
            "key": end,
        })
    return ends, words


def _ends_spelled(raw: Any) -> list[dict[str, Any]]:
    """An ``ends`` group's ends, read permissively: a malformed ``to`` is a way out."""
    ends: list[dict[str, Any]] = []
    listed = [end for end in _listed(raw) if isinstance(end, dict)]
    for place, end in enumerate(listed):
        to = end.get("to")
        if (
            not isinstance(to, int) or isinstance(to, bool) or to == place
            or not 0 <= to < len(listed)
        ):
            to = None
        enter = _players(end["enter"]) if "enter" in end else frozenset((0, 1))
        ends.append({
            "at": end.get("at"),
            "to": to,
            "enter": enter if to is not None else frozenset(),
            "moved_by": _players(end.get("moved_by")),
            "carries": _carried(end.get("carries")),
            "letter": _END_LETTERS[place] if place < len(_END_LETTERS) else str(place),
            "key": f"ends[{place}]",
        })
    return ends


def _listed(value: Any) -> list[Any]:
    """``value`` when it is a list, else nothing: a permissive read of a malformed key."""
    return value if isinstance(value, list) else []


def portal_piece_ids(rules: Any) -> set[str]:
    """The declared piece ids ``rules.portals`` names as a group's ``piece``."""
    return {
        group["piece"] for group in portal_groups(rules) if isinstance(group["piece"], str)
    }


def portal_type_needs_no_moves(rules: Any, piece_id: Any) -> bool:
    """Whether ``piece_id`` is a portal type that no player may ever move (schema 19).

    The second exception to ``pieces[].moves must be a non-empty array`` (the first is
    ``piece_has_stock``): a portal nobody moves is placed by the rule and stands where it
    was put, so it needs no move of its own.
    """
    named = [group for group in portal_groups(rules) if group["piece"] == piece_id]
    return bool(named) and not any(
        end["moved_by"] for group in named for end in group["ends"]
    )


def portal_extra_types(rules: Any) -> int:
    """How many internal piece types the compiler adds for portals: one per end."""
    return sum(len(group["ends"]) for group in portal_groups(rules))


def portal_map_shape(group: dict[str, Any]) -> str:
    """The shape of one group's map (`portal_groups`' group): ``pair`` (written as one),
    ``ring`` (every end leads to the next round one cycle), ``funnel`` (one end is a way out
    only and every other leads to it), or ``other``. The designer's three named shapes and
    the rest (`docs/archive/plan-portals-2.md` §5.4), for the readers that tell maps apart."""
    if group["spelling"] == "pair":
        return "pair"
    ends = group["ends"]
    tos = [end["to"] for end in ends]
    exits = [place for place, to in enumerate(tos) if to is None]
    if len(exits) == 1 and all(to in (None, exits[0]) for to in tos):
        return "funnel"
    if not exits and sorted(tos) == list(range(len(ends))):
        place, seen = 0, set()
        while place not in seen:
            seen.add(place)
            place = tos[place]
        if len(seen) == len(ends):
            return "ring"
    return "other"


# Schema 10. Five features, one version, and only one of them is a top-level key -- which
# is why the gate is `_schema_ten_uses` and not a `V10_RULE_KEYS` frozenset. Q84's costing
# sorted them: a new *value* (`repetition_draw: "never"`), nested keys
# (`npcs[].immune`, `explode_on_capture.by`/`.of`, `drops.stock`), a *relaxation*
# (a piece with no moves, once a stock puts it in a hand) and one new rule key.
#
# `repetition_draw: "never"` is the one value the clock could not previously say. The
# bound `[2, 10]` makes "there is no draw by repetition" inexpressible, so a spec whose
# designer says there is none has to misstate itself -- which is what `intransitive`'s
# committed spec does. Spelled as a word rather than as `0`, because `0` reads as "draw
# at once" to the core's `repetition_count() >= threshold` and as "no rule" to
# `fairy._rule_lines`; a sentinel that two readers disagree about is worse than no
# sentinel. The compiler turns it into a threshold the count can never reach, which is
# schema 6's rule that the boundary carries the answer and not the question.
REPETITION_NEVER = "never"

# Schema 10 (Q78): ``npcs[].immune``, a wild piece that only some types may take.
#
# **Read the key as "immune, except to these".** ``immune: ["N"]`` is *only a knight may
# take this wild piece*; ``immune: []`` is a wild piece nothing may take at all; the key
# absent is schema 9's reading, where anything may take it -- which is what makes the
# feature inert on every spec written before it.
#
# It costs the core nothing, and that is the whole argument for it. ``capture_allowed``
# already holds exactly this relation, so the compiler clears
# ``capture_allowed[attacker][wild type]`` for every attacker the list does not name and
# the schema-3 machinery does the rest -- the three capture tests in
# ``generate_pseudo_legal`` and ``may_capture`` behind ``attacks_square_as``, all
# unchanged. What it buys is one sentence for one rule: *only a knight may take a wolf*
# costs ``archive/wolves.json`` five ``captures`` lists, one per piece that is not the
# knight, each of them a six-element list whose only content is *everything except the
# wolf*. Under ``v10`` that is most of why its elegance reads 0.288 against its parent's
# 0.420.
#
# **Another wild piece is not bound by it.** The clearing skips every type that is itself
# wild in this spec, because a wild piece eats whatever it steps onto and an immunity
# stated about *who may take it* is a rule about the players. Without the exemption the
# re-spelling of ``wolves`` would differ from the original in one unreachable cell of the
# matrix -- ``W`` taking ``W`` -- and the byte-identical check that verifies this feature
# is a pure re-spelling would be checking something weaker.
#
# The one thing it must state, and Q84 asked for the decision out loud: ``capture_allowed``
# is keyed by *type* and not by owner, so a type that is both wild and in a player's hand
# under ``drops`` -- "tame the boar", which schema 9 deliberately allows -- would be
# uncapturable in the player's hands too. That is refused rather than read owner-blind: the
# rule's own name says *wild piece*, being wrong in the permissive direction is silent, and
# nothing becomes inexpressible by it because per-piece ``captures`` lists still say the
# owner-blind thing the long way. `npc_immune_attackers` is the one reading of the key,
# shared by the validator and the compiler.

# Schema 7: reach.mirror may be the string "rotate" instead of a boolean.
#
# A 180-degree rotation of the goal squares -- flipping both rank and file -- so that
# white's i9 becomes black's a1 on a 9x9 board.  The existing boolean could only
# reflect the rank, which is enough for racing kings or king of the hill but not for
# a diagonally symmetric game like intransitive.

# What happens to a struck piece. ``capture`` is removal, which is the game that was
# already there, and it compiles to no rule at all.
STRIKE_EFFECTS = ("capture", "demote", "promote")

# Where the two pieces end up after a strike.
#
# There are exactly three placements and no fourth, which is worth stating because it
# is not obvious:
#
#   ``advance``  the attacker takes the square and the struck piece leaves the board.
#                What capture has always done, so it is the default.
#   ``stay``     the attacker holds the square it came from, and a struck piece that
#                survives its chain transforms where it stands.
#   ``swap``     the attacker takes the square and a surviving struck piece lands on
#                the attacker's origin -- the one square a strike is guaranteed to
#                have freed.
#
#   ``choose``   schema 12 (Q20): the attacker takes the square, as under ``advance``, and
#                places a surviving struck piece on an empty square of ``landing``, read
#                from where the struck piece stood in the mover's frame. A capture with
#                no empty landing square is not generated.
#
#   ``home``     schema 17 (Q540): the attacker takes the square, as under ``advance``,
#                and a surviving struck piece is reborn on its home square -- where the
#                setup stands its type for its side -- or removed when that square is
#                taken (circe). A strike's position only; a contact table refuses it.
#
# Under ``advance`` there is nowhere to put a survivor, and the validator refuses that
# combination rather than quietly removing a piece the chain said should live.
#
# Schema 17 gives ``effect: "capture"`` two positions that say something: ``stay`` keeps
# the attacker on its origin while the victim leaves the board (rifle), and ``home`` puts
# the victim back on its home square as itself (circe). Under ``advance`` and ``swap`` a
# plain capture is still plain capture.
STRIKE_POSITIONS = ("stay", "advance", "swap", "choose", "home")

# The positions a contact table may use: every strike position but ``home``.
CONTACT_POSITIONS = ("stay", "advance", "swap", "choose")

# ``landing`` under ``position: "choose"``: a vector set with a move descriptor's
# ``vectors`` and ``symmetry`` and nothing else.
_LANDING_KEYS = {"vectors", "symmetry"}

# The most landing squares one target may resolve to, for either player. The move word's
# seven free bits (25 to 31) hold the landing index, so the bound is the word's.
# `engine.bridge.rust_supports` refuses it off the compiled table.
MAX_LANDING_CANDIDATES = 128

# Schema 10 (Q19a): ``rules.contacts``, the general form of the strike.
#
# ``strike`` is a map ``victim -> what the victim becomes``. A contact table is a map
# ``(mover, target) -> (what the mover becomes, what the target becomes)`` -- the same
# relation with the attacker's column filled in and the victim's column keyed by the
# attacker as well. Written as a list of links for the reason ``strike.chain`` is a list
# and not a map: ``llm.client.wire_schema`` deletes an object whose keys are not known in
# advance, so a table written as a map is a rule the proposer could never reach.
#
#   ``mover``     the attacking piece the link is about (required)
#   ``onto``      the piece it meets (required)
#   ``mover_to``  what the attacker becomes; absent means unchanged
#   ``onto_to``   what the target becomes; absent means removed, as an ordinary capture
#
# **The two spellings are refused together**, and that is a decision rather than a
# limitation: they are one mechanism, they compile to one ``StrikeRule``, and a spec
# carrying both would be two rules answering the same question -- the shape ``npcs``
# beside ``strike`` already has. What makes the generalisation real is that the compiler
# resolves *both* into the same pair of tables and the boundary sends the schema-6 array
# whenever a table has no attacker column, so every ``strike`` ruleset compiles byte for
# byte to what it always did (``tests/test_schema_ten.py``).
#
#   ``own``       schema 11 (Q19b): ``true`` makes the link about the mover's own piece
#
# An own link is a move no other rule generates. The mover reaches a friendly ``onto``
# along its capture moves, and the link says what both become. It is keyed apart from the
# enemy link for the same pair, so a table can say both. A royal cannot be the target,
# because every royal is immune and an own link onto an immune piece is refused.
_CONTACTS_KEYS = {"links", "position", "immune", "landing"}
_CONTACT_LINK_KEYS = {"mover", "onto", "mover_to", "onto_to", "own"}

# Which edge of an axis a piece may cross, named absolutely.
#
# Absolute, from the standard diagram -- the first player at the bottom, file ``a`` on
# the left -- rather than relative to each player's forward direction. Relative names
# would make ``{"0": "left", "1": "left"}`` secretly asymmetric, which is precisely the
# thing this feature exists to be able to state out loud.
#
# A permission is directional; the topology is not. ``left`` and ``right`` identify the
# *same* seam, because file ``a`` is adjacent to file ``h`` either way, and differ only
# in which direction a piece may traverse it. So the board wraps on an axis when any of
# that axis's four flags is set -- which is what an explosion's blast reads -- while a
# move consults the moving player's own permission.
WRAP_FILE_EDGES = ("none", "left", "right", "both")
WRAP_RANK_EDGES = ("none", "bottom", "top", "both")


class WrapPair(NamedTuple):
    """One pair of a hexagon's opposite sides, as `rules.wrap` names it (Q281).

    ``low`` is the side the frame's indices start from and ``high`` its opposite, as
    ``left`` and ``bottom`` are on a rectangle. ``copy(radius)`` is where the board's copy
    lies across the high side, as a (file, rank) offset; the copy across the low side is
    its negation. A step that leaves by a side lands on the cell it reaches, less that
    side's copy vector.
    """

    low: str
    high: str
    copy: Callable[[int], tuple[int, int]]


# A hexagon's three pairs of opposite sides, named as a flat-topped hex map names them,
# with the first player at the bottom (`docs/game-ideas.md` *Q281 specified*, *The
# spelling*). The three high copies are the lattice's basis: `n` is `ne` less `se`.
WRAP_PAIRS: dict[str, WrapPair] = {
    "n_s": WrapPair("s", "n", lambda radius: (-radius - 1, 2 * radius + 1)),
    "ne_sw": WrapPair("sw", "ne", lambda radius: (radius, radius + 1)),
    "nw_se": WrapPair("nw", "se", lambda radius: (2 * radius + 1, -radius)),
}


def wrap_axes(shape: Any = None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The keys `rules.wrap` states a seam under on this shape, each with its values.

    A grid has two axes, a hexagon three pairs of sides. Every reader of `rules.wrap`
    loops over this rather than spelling the keys out, so a third shape is one edit here.
    """
    if shape == "hexagon":
        return tuple(
            (key, ("none", pair.low, pair.high, "both")) for key, pair in WRAP_PAIRS.items()
        )
    return (("files", WRAP_FILE_EDGES), ("ranks", WRAP_RANK_EDGES))


def wrap_stated(wrap: Any, shape: Any = None) -> dict[str, tuple[Any, Any]]:
    """What each player has on each of the shape's wrap keys, as ``{key: (p0, p1)}``.

    The one reading of the ``X`` / ``X_by_player`` idiom: the per-player object wins where
    it is stated, a player it leaves out falls back to ``X``, and an absent key reads
    ``"none"``. Every key of the shape is in the answer, in `wrap_axes` order. Values are
    returned as written, so the validator's callers see malformed ones and the readers of
    a valid spec see one of the key's four.
    """
    stated: dict[str, tuple[Any, Any]] = {}
    for key, _ in wrap_axes(shape):
        if not isinstance(wrap, dict):
            stated[key] = ("none", "none")
            continue
        shared = wrap.get(key, "none")
        by_player = wrap.get(f"{key}_by_player")
        if isinstance(by_player, dict) and by_player:
            stated[key] = (by_player.get("0", shared), by_player.get("1", shared))
        else:
            stated[key] = (shared, shared)
    return stated


def hex_distance(q: int, r: int) -> int:
    """Steps between two hex cells an axial offset ``(q, r)`` apart."""
    return max(abs(q), abs(r), abs(q + r))


def absent_squares(board: Any) -> frozenset[int]:
    """The frame's squares that are not cells of a hexagon board, as square indices.

    Empty unless ``board.shape`` is ``"hexagon"``. A hexagon of radius R sits in an odd
    square frame ``2R + 1`` wide, and its cells are those with ``R <= file + rank <= 3R``.
    Malformed input gives an empty set, as `obstacle_squares` does, because the validator
    calls this while it is still collecting the errors about it.
    """
    if not isinstance(board, dict) or board.get("shape") != "hexagon":
        return frozenset()
    width, height = board.get("width"), board.get("height")
    if (
        not isinstance(width, int) or isinstance(width, bool) or width != height
        or not 1 <= width <= MAX_BOARD_DIM or width % 2 == 0
    ):
        return frozenset()
    radius = (width - 1) // 2
    return frozenset(
        rank * width + file
        for rank in range(width)
        for file in range(width)
        if not radius <= file + rank <= 3 * radius
    )

# The longest turn the schema will accept. Not a technical limit -- a turn of twenty
# moves is a different game from a turn of two and nothing here would stop it -- but a
# search's branching factor is the legal move count raised to the power of the turn
# length, and past a handful the roster stops being able to read the game at all.
MAX_MOVES_PER_TURN = 4

# How a colour behaves once laid down.
#
# "permanent" is the cheap half and the one worth building first: a colour that can
# never change is monotone, so every colouring ply is irreversible, so it resets the
# no-progress clock, so colour is constant inside any window a repetition test compares
# across -- and the state needs no Zobrist term at all. That is the same trick `hands`
# already uses. "repaint" gives that up and pays for it.
TERRITORY_MODES = ("permanent", "repaint")

# How the second player's army and move vectors are derived from the first player's.
#   mirror   reflect the rank, leave the file -- chess, and the historical default
#   rotate   180-degree rotation, which also mirrors the file -- legan, xiangqi
ORIENTATIONS = ("mirror", "rotate")

MAX_BOARD_DIM = 16
MAX_PIECE_TYPES = 24
MAX_MOVE_DESCRIPTORS = 12
MAX_VECTOR_COMPONENT = 8

_MOVE_KEYS = {
    "kind", "vectors", "symmetry", "range", "min_range", "mode", "condition",
    "screen_range", "screen",
}
_CONDITION_KEYS = {"from_rank", "first_move", "region"}
_PIECE_KEYS = {
    "id", "name", "royal", "value", "moves", "region", "promoted_from", "captures",
    "facing", "beats", "spy", "speed", "rest", "sight",
}
# ``appearance`` and ``provenance`` are deliberately *outside* the mechanics.
# ``canonical.MECHANIC_KEYS`` is what the content hash covers, so anything added inside
# ``pieces`` rehashes every spec that uses it and orphans the calibration runs -- and
# neither of these changes a game. Appearance says how to draw a piece; provenance says
# where the ruleset came from. Both are optional, and nothing in ``engine/`` reads them.
_SPEC_KEYS = {
    "schema_version",
    "name",
    "description",
    "board",
    "pieces",
    "setup",
    "rules",
    "appearance",
    "provenance",
}
_APPEARANCE_KEYS = {"icons"}
# ``exhibit`` is a sentence saying the project keeps this ruleset on purpose for what it
# shows rather than as a candidate -- `archive/stones`, an example of a game that cannot
# really be played. The pages print it wherever the game's number is shown
# (`web/library.exhibit`), so a reader does not take the number for a score.
# ``added`` is the day the file joined the library, ``YYYY-MM-DD`` (Q826): what the landing
# page's *New in the library* shelf and the games page's *date added* order read. Outside
# the mechanics like the rest of provenance, so stamping it moved no hash
# (`.claude/scripts/backfill_added.py`, `tests/test_added.py`).
# ``preview`` is a sentence saying the game is shown on the site as a preview: a new kind of
# game, still being tested, playable and kept off every scored surface (Q931,
# `web/library.preview`). Outside the mechanics, so marking or unmarking a game moves no
# hash and no measurement.
_PROVENANCE_KEYS = {
    "origin", "author", "year", "source", "history", "theme", "parents", "exhibit", "added",
    "preview",
}
_SETUP_KEYS = {"mirror", "placements", "orientation"}
_PLACEMENT_KEYS = {"piece", "player", "squares", "facing"}
_RULES_KEYS = {
    "win",
    "stalemate",
    "promotion",
    "en_passant",
    "castling",
    "max_plies",
    "repetition_draw",
    "no_progress_draw",
    "bare_royal",
    "check_limit",
    "reach",
    "extinction",
    "must_capture",
    "checkmate",
    "forbid_check",
    "royal_facing",
    "drops",
    "explode_on_capture",
    "forbid_immobility",
    "scoring",
    "territory",
    "turn",
    "wrap",
    "obstacles",
    "strike",
    "contacts",
    "npcs",
    "promote_on_entry",
    "portals",
    "simultaneous",
}
_OBSTACLES_KEYS = {"squares", "mirror", "blocks_leaps"}
_STRIKE_KEYS = {"effect", "chain", "position", "immune", "landing", "switches_sides"}
_STRIKE_LINK_KEYS = {"from", "to"}
# The three pairs are schema 16's, a hexagon's alone (`WRAP_PAIRS`); `files` and `ranks`
# are a grid's alone. `_validate_wrap` refuses each shape's keys on the other.
_WRAP_KEYS = {
    "files", "ranks", "files_by_player", "ranks_by_player",
    "n_s", "ne_sw", "nw_se", "n_s_by_player", "ne_sw_by_player", "nw_se_by_player",
}
# ``on_leaving`` is schema 15 (Q452, traditional shogi's B1): the entry also promotes its
# piece on a move that *starts* in the zone, as Fairy-Stockfish and real shogi do. See
# ``_schema_fifteen_uses`` and ``compile.PromotionRule.on_leaving``. ``only_lost`` is
# schema 18 (Q783, Grand chess): a target is offered only while its side has fewer of it
# than it started with. See ``_check_only_lost_coherence`` and
# ``compile.PromotionRule.only_lost``.
_PROMOTION_KEYS = {
    "piece", "rank", "to", "ranks", "squares", "mandatory", "to_by_player", "on_leaving",
    "only_lost",
}
_CASTLING_KEYS = {"king", "rook", "king_steps", "king_file", "rook_file"}
# ``stock`` is schema 10 (Q51): what each player holds in hand before a piece has been
# taken. See ``piece_has_stock`` and ``_validate_drops``.
# ``placement`` is schema 18 (Q784): while a side holds anything in hand, its only legal
# moves are drops, and a royal may start in the stock. See ``_check_placement_coherence``.
# ``either_face`` is schema 18 (Q785): a held piece may be dropped as itself or as the
# piece whose ``promoted_from`` it is. See ``_check_either_face_coherence``.
_DROPS_KEYS = {
    "from_captures", "region", "no_doubled", "no_mate", "stock", "placement", "either_face",
}

# What ``drops.stock`` may say instead of a number: a supply that never runs out.
#
# The half of Q51 that needed a word rather than an integer. A stone a player may always
# place is not a large number of stones -- a number is a resource to be spent and this is
# not one -- and the core answers it by never decrementing the slot rather than by
# counting down from something big.
STOCK_UNLIMITED = "unlimited"

# The largest finite stock. Not a technical limit: a hand is a `u32` and the drop
# generator walks the empty squares once per type it holds whatever the count. It is the
# same kind of bound as ``MAX_MOVES_PER_TURN`` -- past a handful the opening is a hundred
# placements deep and the roster stops being able to read the game.
MAX_STOCK = 32
# ``by`` and ``of`` are schema 10 (Q18): *whose* capture explodes and *what* has to be
# taken for it to. Absent means every type, which is atomic chess and is what makes the
# pair inert on every spec written before them. See ``_validate_explosion``.
_EXPLODE_KEYS = {"radius", "immune", "by", "of"}
_REACH_KEYS = {"piece", "squares", "mirror", "result", "safe", "on"}
# ``on`` is schema 18 (Q786, Jeson Mor): ``"arrive"`` is the rule as it always was, a piece
# standing on a goal square decides the game; ``"leave"`` reads the move's origin instead,
# so the game is decided by the move that takes the named piece *off* a goal square. See
# ``_validate_reach``.
REACH_ON = ("arrive", "leave")
# ``per_type`` is schema 18 (Q783, Extinction chess): each listed type is counted on its
# own, so a side loses when any one of them runs out. See ``_validate_extinction``.
_EXTINCTION_KEYS = {"pieces", "min_count", "result", "per_type"}
_SCORING_KEYS = {"count", "region"}
_TERRITORY_KEYS = {"mode", "win_at"}
_TURN_KEYS = {"moves", "ramp"}


class SpecError(Exception):
    """Raised when a spec is structurally invalid.

    Carries every problem found, not just the first: a proposal LLM given one error
    at a time will fix that one and reintroduce another.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_spec(spec: Any, *, new: bool = True) -> list[str]:
    """Return a list of problems with ``spec``. An empty list means it is valid.

    ``new`` says the spec is being admitted: a draft, a proposal, an operator's child, a
    file under ``validate``, a game promoted into the library. Every gate keeps the
    default. ``validate_or_raise``, the check ``compile_spec`` opens with, passes
    ``new=False``, because building an engine is also how a *stored* spec is loaded: a
    retired game's page and its match replays on the site, and a calibration corpus's
    entry re-scored. What that skips is three refusals, each of a rule that never fires,
    added after such a spec could have been played: ``_refuse_a_promotion_rank_past_the_board``
    (Q434), ``_refuse_a_portal_bar_no_move_meets`` (Q879) and ``_refuse_a_one_square_lame_leap``
    (Q884). The engine plays that spec exactly as it always did; the refusal is about not
    admitting another one.
    """
    errors: list[str] = []

    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]

    _check_keys(spec, _SPEC_KEYS, "spec", errors, required={"name", "board", "pieces", "setup"})

    version = spec.get("schema_version", SCHEMA_VERSION)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"schema_version {version!r} is not supported (expected one of "
            f"{', '.join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)})"
        )
        version = SCHEMA_VERSION

    if not isinstance(spec.get("name"), str) or not spec.get("name", "").strip():
        errors.append("name must be a non-empty string")

    width, height, lattice = _validate_board(spec.get("board"), errors)
    hex_lattice = lattice == "hex"
    piece_ids = _validate_pieces(
        spec.get("pieces"), width, height, errors, spec.get("rules"), hex_lattice
    )
    _validate_setup(
        spec.get("setup"), piece_ids, width, height, errors, spec.get("pieces"), hex_lattice
    )
    _validate_rules(
        spec.get("rules", {}), piece_ids, width, height, errors, hex_lattice,
        board_shape(spec),
    )
    if new:
        _refuse_a_promotion_rank_past_the_board(spec.get("rules"), height, errors)
        _refuse_a_portal_bar_no_move_meets(spec, errors)
        _refuse_a_one_square_lame_leap(spec, errors)
    _validate_appearance(spec.get("appearance"), piece_ids, errors)
    _validate_provenance(spec.get("provenance"), errors)
    _validate_coherence(spec, errors)
    if version < 2:
        _validate_version_one_stays_within_version_one(spec, errors)
    if version < 3:
        _validate_version_two_stays_within_version_two(spec, errors)
    if version < 4:
        rules = spec.get("rules")
        used = V4_RULE_KEYS.intersection(rules) if isinstance(rules, dict) else set()
        if used:
            errors.append(
                f"rules uses {', '.join(sorted(used))}, added in schema 4; declare "
                '"schema_version": 4 to use it'
            )
    if version < 5:
        rules = spec.get("rules")
        used = V5_RULE_KEYS.intersection(rules) if isinstance(rules, dict) else set()
        if used:
            errors.append(
                f"rules uses {', '.join(sorted(used))}, added in schema 5; declare "
                '"schema_version": 5 to use it'
            )
    if version < 6:
        rules = spec.get("rules")
        used = V6_RULE_KEYS.intersection(rules) if isinstance(rules, dict) else set()
        if used:
            errors.append(
                f"rules uses {', '.join(sorted(used))}, added in schema 6; declare "
                '"schema_version": 6 to use it'
            )
    if version < 7:
        rules = spec.get("rules")
        if isinstance(rules, dict):
            reach = rules.get("reach")
            if isinstance(reach, dict) and reach.get("mirror") == "rotate":
                errors.append(
                    'rules.reach.mirror "rotate" was added in schema 7; declare '
                    '"schema_version": 7 to use it'
                )
    if version < 8:
        pieces = spec.get("pieces")
        used = (
            {key for piece in pieces if isinstance(piece, dict)
             for key in V8_PIECE_KEYS.intersection(piece)}
            if isinstance(pieces, list) else set()
        )
        setup = spec.get("setup")
        placements = setup.get("placements") if isinstance(setup, dict) else None
        if isinstance(placements, list) and any(
            isinstance(placement, dict) and "facing" in placement
            for placement in placements
        ):
            used.add("setup.placements[].facing")
        if used:
            errors.append(
                f"a piece uses {', '.join(sorted(used))}, added in schema 8; declare "
                '"schema_version": 8 to use it'
            )
    if version < 9:
        used = _schema_nine_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 9; declare "
                '"schema_version": 9 to use it'
            )
    if version < 10:
        used = _schema_ten_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 10; declare "
                '"schema_version": 10 to use it'
            )
    if version < 11:
        used = _schema_eleven_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 11; declare "
                '"schema_version": 11 to use it'
            )
    if version < 12:
        used = _schema_twelve_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 12; declare "
                '"schema_version": 12 to use it'
            )
    if version < 13:
        used = _schema_thirteen_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 13; declare "
                '"schema_version": 13 to use it'
            )
    if version < 14:
        used = _schema_fourteen_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 14; declare "
                '"schema_version": 14 to use it'
            )
    if version < 15:
        used = _schema_fifteen_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 15; declare "
                '"schema_version": 15 to use it'
            )
    if version < 16:
        used = _schema_sixteen_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 16; declare "
                '"schema_version": 16 to use it'
            )
    if version < 17:
        used = _schema_seventeen_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 17; declare "
                '"schema_version": 17 to use it'
            )
    if version < 18:
        used = _schema_eighteen_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 18; declare "
                '"schema_version": 18 to use it'
            )
    if version < 19:
        used = _schema_nineteen_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 19; declare "
                '"schema_version": 19 to use it'
            )
    if version < 20:
        used = _schema_twenty_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 20; declare "
                '"schema_version": 20 to use it'
            )
    if version < 21:
        used = _schema_twenty_one_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 21; declare "
                '"schema_version": 21 to use it'
            )
    if version < 22:
        used = _schema_twenty_two_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 22; declare "
                '"schema_version": 22 to use it'
            )
    if version < 23:
        used = _schema_twenty_three_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 23; declare "
                '"schema_version": 23 to use it'
            )
    if version < 24:
        used = _schema_twenty_four_uses(spec)
        if used and version >= 22:
            # Folds were schema 22's until their rules changed (the user, 2026-10-12);
            # the first rules are gone, so a fold under 22 or 23 names no game.
            errors.append(
                "folds changed at schema 24 (the user, 2026-10-12): a piece that lands on a "
                'fold stays or goes through; declare "schema_version": 24'
            )
        elif used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 24; declare "
                '"schema_version": 24 to use it'
            )
    if version < 25:
        used = _schema_twenty_five_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 25; declare "
                '"schema_version": 25 to use it'
            )
    if version < 26:
        used = _schema_twenty_six_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 26; declare "
                '"schema_version": 26 to use it'
            )
    if version < 27:
        used = _schema_twenty_seven_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 27; declare "
                '"schema_version": 27 to use it'
            )
    if version < 28:
        used = _schema_twenty_eight_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 28; declare "
                '"schema_version": 28 to use it'
            )
    if version < 29:
        used = _schema_twenty_nine_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 29; declare "
                '"schema_version": 29 to use it'
            )
    if version < 30:
        used = _schema_thirty_uses(spec)
        if used:
            errors.append(
                f"the spec uses {', '.join(sorted(used))}, added in schema 30; declare "
                '"schema_version": 30 to use it'
            )
    # Schema 29 (Q983a, `docs/plan-rest.md` §4.4): the old rest is refused at every version,
    # by admission and by the loader alike, since a loader that still compiled it would be
    # a second rest beside `pieces[].rest`.
    _refuse_the_retired_rest(spec, errors)

    return errors


def _schema_nine_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 9 added, by name.

    One collector rather than one ``if`` per feature, so that the second key schema 9
    carries (queue task Q13) joins here and the message stays one sentence: a spec that
    reaches for two schema-9 features under an older stamp is told both at once.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    used |= {f"rules.{key}" for key in V9_RULE_KEYS.intersection(rules)}
    scoring = rules.get("scoring")
    if isinstance(scoring, dict) and scoring.get("count") == "points":
        used.add('rules.scoring.count "points"')
    return used


def _schema_ten_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 10 added, by name.

    The same collector shape as ``_schema_nine_uses`` and for a stronger reason: schema 10
    carries five features and exactly *one* of them is a top-level key under ``rules``, so
    a ``V10_RULE_KEYS`` frozenset written the schema-4-to-6 way would gate one fifth of the
    bump and wave the rest through under any stamp. Q84 sorted them -- a new value on an
    existing key, three nested keys, one relaxation and one rule key -- and only the last
    is indexable.

    **The set of names this returns is the bump's inventory.** ``describe.py`` has to be
    able to say each of them, ``docs/reference/schema-showcase-coverage.md`` audits against
    them, and each owes a demonstrating ruleset. Everything is spelled as its own ``if`` so
    that a spec reaching for two of them under an older stamp is told about both at once.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    if rules.get("repetition_draw") == REPETITION_NEVER:
        used.add('rules.repetition_draw "never"')
    npcs = rules.get("npcs")
    if isinstance(npcs, list) and any(
        isinstance(npc, dict) and "immune" in npc for npc in npcs
    ):
        used.add("rules.npcs[].immune")
    explosion = rules.get("explode_on_capture")
    if isinstance(explosion, dict):
        used |= {
            f"rules.explode_on_capture.{key}"
            for key in ("by", "of")
            if key in explosion
        }
    drops = rules.get("drops")
    if isinstance(drops, dict) and "stock" in drops:
        used.add("rules.drops.stock")
    # The bump's one indexable feature, and the only one a `V10_RULE_KEYS` frozenset
    # could have caught -- which is why the collector exists rather than the frozenset.
    if "contacts" in rules:
        used.add("rules.contacts")
    # The relaxation, named separately from the key that gates it. A moveless piece and a
    # stock arrive together in a valid spec, but they are two of the bump's five features
    # and the collector's set *is* the inventory -- `describe.py` owes a sentence for each
    # and the showcase audit reads this list, so folding one into the other would hide it.
    pieces = spec.get("pieces")
    if isinstance(pieces, list) and any(
        isinstance(piece, dict) and piece.get("moves") == [] for piece in pieces
    ):
        used.add("a piece with no moves of its own")
    return used


def _schema_eleven_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 11 added, by name.

    One feature: a contact link about the mover's *own* piece (Q19b). It is a nested key,
    so no key set could gate it. It has a version to itself because it moves perft.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    contacts = rules.get("contacts")
    links = contacts.get("links") if isinstance(contacts, dict) else None
    if isinstance(links, list) and any(
        isinstance(link, dict) and "own" in link for link in links
    ):
        used.add("rules.contacts.links[].own")
    return used


def _schema_twelve_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 12 added, by name.

    One feature on two spellings: ``position: "choose"`` and its ``landing`` set (Q20),
    on ``strike`` and on ``contacts``. A new value and a nested key, so no key set could
    gate either.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        if not isinstance(rule, dict):
            continue
        if rule.get("position") == "choose":
            used.add(f"rules.{key}.position=choose")
        if "landing" in rule:
            used.add(f"rules.{key}.landing")
    return used


def _schema_thirteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 13 added, by name.

    One feature: the ``sweep`` move kind (Q21). A new value on a move descriptor, gated the
    way ``lame`` is.
    """
    return {"the 'sweep' move kind"} if uses_sweep(spec) else set()


def _schema_fourteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 14 added, by name.

    Two features (Q229): the ``board.lattice`` declaration and a promotion zone stated as
    ``squares``. Nested keys both, so no key set could gate either. The key is schema 14's
    whatever its value, so ``"square"`` written out is named too; it compiles to the board
    it always was and owes no showcase.
    """
    used: set[str] = set()
    board = spec.get("board")
    if isinstance(board, dict) and "lattice" in board:
        used.add(f"board.lattice={board['lattice']}")
    rules = spec.get("rules")
    if isinstance(rules, dict) and any(
        isinstance(entry, dict) and "squares" in entry for entry in _promotion_entries(rules)
    ):
        used.add("rules.promotion.squares")
    return used


def _schema_fifteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 15 added, by name.

    One feature (Q452, traditional shogi's B1): ``promotion.on_leaving``, a promotion
    entry that also promotes its piece on a move *out of* its zone. A nested key, so no
    key set could gate it. The key is schema 15's whatever its value, as
    ``board.lattice`` is schema 14's, so ``false`` written out is named too; it compiles
    to the game that was already there and owes no showcase.
    """
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return set()
    used: set[str] = set()
    for entry in _promotion_entries(rules):
        if isinstance(entry, dict) and "on_leaving" in entry:
            used.add(
                "rules.promotion.on_leaving"
                if entry["on_leaving"] is True
                else "rules.promotion.on_leaving=false"
            )
    return used


def _schema_sixteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 16 added, by name.

    Two features (Q281): ``board.shape`` and a hexagon's three pairs of sides under
    ``rules.wrap``. Nested keys both, so no key set could gate either. The shape is
    named with its value, as ``board.lattice`` is, so ``"grid"`` written out is named
    too; it compiles to the board it always was and owes no showcase. A pair key is
    named whatever its value, because a schema-15 spec may not write even an inert one.
    """
    used: set[str] = set()
    board = spec.get("board")
    if isinstance(board, dict) and "shape" in board:
        used.add(f"board.shape={board['shape']}")
    rules = spec.get("rules")
    wrap = rules.get("wrap") if isinstance(rules, dict) else None
    if isinstance(wrap, dict) and any(
        key in wrap or f"{key}_by_player" in wrap for key in WRAP_PAIRS
    ):
        used.add("rules.wrap.hexagon")
    return used


def _schema_seventeen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 17 added, by name.

    Four features (Q540, *what a capture leaves behind*), all new values or nested keys,
    so no key set could gate any of them:

    * ``rules.strike.position=stay(capture)`` -- rifle: a capture that leaves the
      attacker where it stood. Schema 6 to 16 refused the spelling as stating nothing
      (Q538); the value is old, the combination is the feature.
    * ``rules.strike.position=home`` -- circe: a struck piece reborn on its home square.
    * ``rules.strike.switches_sides`` -- Andernach: the attacker changes sides. Named
      whatever its value, as ``board.lattice`` is, because an older spec may not write
      even an empty one.
    * ``rules.explode_on_capture.radius=0`` -- kamikaze: the blast takes the capturer and
      its victim and nothing around them.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    strike = rules.get("strike")
    if isinstance(strike, dict):
        position = strike.get("position", "advance")
        if position == "stay" and strike.get("effect", "capture") == "capture":
            used.add("rules.strike.position=stay(capture)")
        if position == "home":
            used.add("rules.strike.position=home")
        if "switches_sides" in strike:
            used.add("rules.strike.switches_sides")
    explosion = rules.get("explode_on_capture")
    if isinstance(explosion, dict) and explosion.get("radius") == 0 and not isinstance(
        explosion.get("radius"), bool
    ):
        used.add("rules.explode_on_capture.radius=0")
    return used


def _schema_eighteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 18 added, by name.

    Schema 18 is built in several tasks (`docs/archive/plan-schema-17-18.md` §9.3, 18a to
    18f), one version for all of them: each later task adds its names here, and until it lands its
    feature is refused by name at every version, so nothing accepted earlier is read
    differently afterwards. 18a (Q781) opens it with two lifted restrictions, so both
    are predicates over the spec rather than keys:

    * ``rules.contacts.mover_to=mover(promotion)`` -- Einstein's keeping link: a contact
      link whose ``mover_to`` is the mover itself, accepted where the mover has a
      promotion of its own (``keeping_link_holds``). The contact overrides the promotion
      (Q539), so a queen that captures stays a queen where its promotion would demote
      it. Schema 10 to 17 refused the spelling as changing nothing, which it still does
      where no promotion could fire.
    * ``rules.castling(walled rank)`` -- Omega's castling: a wall or an absent cell on a
      rank castling starts from, but on no square any castling crosses or lands on
      (``castling_rank_walled_beyond_its_spans``). A walled span stays refused.

    18c (Q783) adds two nested keys, each named whatever its value, as schema 15's
    ``on_leaving`` is, because an older spec may not write even an inert one; ``false``
    is named with its value and compiles to the game that was already there:

    * ``rules.extinction.per_type`` -- Extinction chess: each listed type is counted on
      its own, so a side loses when any one of them runs out, not when all of them
      together do.
    * ``rules.promotion.only_lost`` -- Grand chess: a promotion target is offered only
      while the mover's side has fewer of that type than it started with
      (``_check_only_lost_coherence``).

    18d (Q784) adds one nested key, named whatever its value as 18c's are:

    * ``rules.drops.placement`` -- Placement chess and Sittuyin: while a side holds
      anything in hand its only legal moves are drops, so a stock is played out before
      a piece on the board moves. A royal may start in the stock
      (``royal_starts_in_hand``, ``_check_placement_coherence``).

    18e (Q785) adds one nested key, named whatever its value as 18d's is:

    * ``rules.drops.either_face`` -- Kyoto shogi: a piece in hand may be dropped as
      itself or as its other face, the piece whose ``promoted_from`` it is, so a captured
      dragon (held as a rook) may come back as either (``_check_either_face_coherence``).

    18f (Q786) adds one new value and its key, named whatever its value as 18c's are:

    * ``rules.reach.on=leave`` -- Jeson Mor: the game is decided by the move that takes
      the named piece off a goal square, not by a piece standing on one. The core keeps
      one position field for it, the side that has just left (``Game::goal_left_by``).
      ``rules.reach.on=arrive`` is the rule written out, and compiles to the game that was
      already there.

    18b (Q782) adds two new values on an existing key, each named wherever it is written,
    on a piece's move or on a strike's or contact table's landing:

    * ``symmetry=rotate6`` and ``symmetry=dihedral12`` -- the hexagon's turns, so a hex
      step is written once and turned six ways (or six ways and mirrored). A spelling and
      not a rule: `complexity.expand_symmetry` turns them, the compiler reads its answer,
      and a spec written with them compiles to the tables of its twin that lists every
      vector under ``none``. Accepted only on a hex lattice (`_symmetry_refusals`).
    """
    used: set[str] = set()
    used |= _hex_symmetries_used(spec)
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and "per_type" in extinction:
        used.add(
            "rules.extinction.per_type"
            if extinction["per_type"] is not False
            else "rules.extinction.per_type=false"
        )
    for entry in _promotion_entries(rules):
        if isinstance(entry, dict) and "only_lost" in entry:
            used.add(
                "rules.promotion.only_lost"
                if entry["only_lost"] is True
                else "rules.promotion.only_lost=false"
            )
    drops = rules.get("drops")
    if isinstance(drops, dict) and "placement" in drops:
        used.add(
            "rules.drops.placement"
            if drops["placement"] is True
            else "rules.drops.placement=false"
        )
    if isinstance(drops, dict) and "either_face" in drops:
        used.add(
            "rules.drops.either_face"
            if drops["either_face"] is True
            else "rules.drops.either_face=false"
        )
    contacts = rules.get("contacts")
    links = contacts.get("links") if isinstance(contacts, dict) else None
    promoting = promoting_piece_ids(rules)
    if isinstance(links, list) and any(
        keeping_link_holds(link, promoting) for link in links
    ):
        used.add("rules.contacts.mover_to=mover(promotion)")
    if castling_rank_walled_beyond_its_spans(spec):
        used.add("rules.castling(walled rank)")
    reach = rules.get("reach")
    if isinstance(reach, dict) and "on" in reach:
        used.add(f"rules.reach.on={reach['on']}")
    return used


def _schema_nineteen_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 19 added, by name (Q771, portals).

    One new top-level rule and its two nested keys, each named whatever its value, as
    schema 15's ``on_leaving`` is, because an older spec may not write even an inert one:

    * ``rules.portals`` -- pairs of portals: a piece whose move ends on an open end comes
      out one king's step beyond the other end, in the direction it travelled
      (`docs/archive/plan-portals.md` §§1-2).
    * ``rules.portals.moved_by`` -- which players may move each end (the free agent, §3).
    * ``rules.portals.passage`` -- per player, which end that player's pieces may go in
      by: both, one way, or neither, when the pair is an obstacle to them.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if not isinstance(rules, dict) or "portals" not in rules:
        return used
    used.add("rules.portals")
    portals = rules.get("portals")
    for pair in portals if isinstance(portals, list) else ():
        if isinstance(pair, dict):
            if "moved_by" in pair:
                used.add("rules.portals.moved_by")
            if "passage" in pair:
                used.add("rules.portals.passage")
    return used


def _schema_twenty_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 20 added, by name (Q807, no straight back).

    * ``rules.portals.no_straight_back`` -- an end of the pair, once moved, may not be moved
      straight back onto the square it left on the very next move: a portal's ko, the answer
      to Ferry's repetition by shuffling (`docs/archive/plan-portals.md` *Ferry's shuffle*). Named
      whatever its value, as schema 19's names are, because an older spec may not write even
      the inert ``false``.
    """
    rules = spec.get("rules")
    portals = rules.get("portals") if isinstance(rules, dict) else None
    for pair in portals if isinstance(portals, list) else ():
        if isinstance(pair, dict) and "no_straight_back" in pair:
            return {"rules.portals.no_straight_back"}
    return set()


def _schema_twenty_one_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 21 added, by name (Q843a, portal maps and filters).

    `docs/archive/plan-portals-2.md` §2. Each named whatever its value, as schema 19's names are,
    because an older spec may not write even an inert one:

    * ``rules.portals.ends`` -- a group written as a list of ends rather than a pair: a
      ring, a funnel or any other map of three to eight ends.
    * ``rules.portals.ends.to`` -- the end a piece that goes in by this one comes out of.
    * ``rules.portals.ends.enter`` -- whose pieces may go in by this end.
    * ``rules.portals.carries`` -- which pieces may go in by an end, in either spelling.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    portals = rules.get("portals") if isinstance(rules, dict) else None
    for group in portals if isinstance(portals, list) else ():
        if not isinstance(group, dict):
            continue
        if "carries" in group:
            used.add("rules.portals.carries")
        if "ends" not in group:
            continue
        used.add("rules.portals.ends")
        for end in _listed(group["ends"]):
            if not isinstance(end, dict):
                continue
            if "to" in end:
                used.add("rules.portals.ends.to")
            if "enter" in end:
                used.add("rules.portals.ends.enter")
            if "carries" in end:
                used.add("rules.portals.carries")
    return used


def _schema_twenty_two_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 22 added, by name: nothing any more.

    Schema 22 (Q843c) added fold portals, ``rules.portals.kind``. Their rules changed at
    schema 24 (Q867b, `docs/archive/plan-portals-2.md` §3R.6), and the first rules were deleted, not
    kept behind the old number (one implementation), so ``kind`` is
    `_schema_twenty_four_uses`' and schema 22 keeps no rule of its own. A version with
    nothing behind it is harmless: `settle_schema_version` stamps the lowest version a spec
    validates under, so no new spec is ever stamped 22.
    """
    del spec
    return set()


def _schema_twenty_three_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 23 added, by name (Q850, several wild pieces on
    one walk).

    `docs/reference/designer-case-study-wolves.md` §2.2. Named whatever its value, because
    an older spec may not write even an empty one:

    * ``rules.npcs.also_at`` -- squares of the entry's own route where a further wild piece
      of the same kind stands at ply zero, every one of them walking the route in step with
      the piece on the route's first square.
    """
    rules = spec.get("rules")
    npcs = rules.get("npcs") if isinstance(rules, dict) else None
    for npc in npcs if isinstance(npcs, list) else ():
        if isinstance(npc, dict) and "also_at" in npc:
            return {"rules.npcs.also_at"}
    return set()


def _schema_twenty_four_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 24 added, by name (Q867b, folds revised).

    `docs/archive/plan-portals-2.md` §3R. Named whatever its value, as schema 19's names are,
    because an older spec may not write even the default:

    * ``rules.portals.kind`` -- a group's rule set: ``"fold"``, a portal a piece of either
      side may stand in, where a move that ends on it empty may stay or go through and end
      on the end it leads to, taking an enemy piece there as any landing does (absent:
      today's momentum portal). Schema 22's first fold rules are gone with their version.
    """
    rules = spec.get("rules")
    portals = rules.get("portals") if isinstance(rules, dict) else None
    for group in portals if isinstance(portals, list) else ():
        if isinstance(group, dict) and "kind" in group:
            return {"rules.portals.kind"}
    return set()


def _schema_twenty_five_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 25 added, by name (Q869, a fixed facing).

    `docs/archive/plan-library-audit.md` *The rear pawn: a fixed facing*. Named whatever its value,
    because an older spec may not write even the default:

    * ``pieces[].facing.turns`` -- whether a move may turn the piece (absent: it may, as
      every facing piece could from schema 8). ``false`` keeps each piece pointing the way
      it was placed, for the whole game.
    """
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        facing = piece.get("facing") if isinstance(piece, dict) else None
        if isinstance(facing, dict) and "turns" in facing:
            return {"pieces[].facing.turns"}
    return set()


def _schema_twenty_six_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 26 added, by name (Q918a, simultaneous moves).

    `docs/reference/rounds.md` §5. Named whatever its value, because an older spec may not
    write even the default:

    * ``rules.simultaneous`` -- the game is played in rounds: both players choose a move on
      the same board, in secret, and both are revealed and resolved together;
    * ``rules.simultaneous.tie`` -- what decides a collision both pieces win or both lose
      (absent: both are eliminated), with its value;
    * ``pieces[].beats`` -- the pieces this one eliminates when the two collide.
    """
    used: set[str] = set()
    rules = spec.get("rules")
    if isinstance(rules, dict) and "simultaneous" in rules:
        used.add("rules.simultaneous")
        rule = rules["simultaneous"]
        if isinstance(rule, dict) and "tie" in rule:
            used.add(f"rules.simultaneous.tie {rule['tie']!r}")
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        if isinstance(piece, dict) and "beats" in piece:
            used.add("pieces[].beats")
            break
    return used


def _schema_twenty_seven_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 27 added, by name.

    One version, three tasks (`docs/plan-spy.md` §2.2), each adding its names here as it
    lands, as schema 18's six did:

    * ``rules.extinction.per_type.own`` -- Q936, `docs/reference/extinction-any-kind.md`
      §2: each listed kind counted on its own, and only against a side that starts with
      at least one, so two armies of different kinds can each lose by running out of any
      one of their own.

    * ``rules.simultaneous.rest`` -- Q942a, `docs/archive/plan-round-chase.md` §3 R1, the piece
      that made a player's move resting the next round -- was here until schema 29 retired it
      for ``pieces[].rest`` (Q983a, `docs/plan-rest.md` §4.4); it is refused at every version
      now, so this collector no longer names it;
    * ``rules.simultaneous.swap`` -- Q942a, R2: two moves that each end on the other's
      origin meet as a collision (``beats`` and the tie decide) instead of passing;
    * ``rules.simultaneous.both`` -- Q942a, X1: an ending that fires for both players in
      one round is settled by ``rules.scoring`` instead of drawn.

    * ``pieces.spy`` -- Q935a, `docs/plan-spy.md` §1, §2: a piece type that decides how the
      game is played: alternating while both players have one, announced rounds (the
      player without one moves first, in the open) once one is lost, blind rounds once
      both are.

    Each name is collected whatever its value, since an older spec may not write even a
    refused one.
    """
    used: set[str] = set()
    pieces = spec.get("pieces")
    if any(isinstance(piece, dict) and "spy" in piece for piece in _listed(pieces)):
        used.add("pieces.spy")
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return used
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and extinction.get("per_type") == EXTINCTION_PER_TYPE_OWN:
        used.add("rules.extinction.per_type.own")
    simultaneous = rules.get("simultaneous")
    if isinstance(simultaneous, dict):
        for key in SIMULTANEOUS_CHASE_KEYS:
            if key in simultaneous:
                used.add(f"rules.simultaneous.{key}")
    return used


def _schema_twenty_eight_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 28 added, by name (Q943a, `docs/plan-speed.md`).

    * ``pieces.speed`` -- a piece type's speed, a rank (absent: 0, the slowest). When the
      two pieces moving in a round differ in speed, the faster move is made first, as a move
      of an alternating game, and the slower is then made only if its piece still stands
      and the move can still be made on that board (§2.4, §2.5).

    Collected whatever its value, since an older spec may not write even a refused one.
    """
    if has_speeds(spec):
        return {"pieces.speed"}
    return set()


def _schema_twenty_nine_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 29 added, by name (Q983a, `docs/plan-rest.md`).

    * ``pieces.rest`` -- a piece type's rest, 1 to ``REST_MAX`` (absent: 0, it never rests).
      A piece that makes its player's move may not make that player's move during that
      player's next *rest* turns (an alternating turn or a round), unless the rest would
      leave the player no legal move, when it is lifted for that turn (§3).

    Collected whatever its value, since an older spec may not write even a refused one.
    """
    if has_rests(spec):
        return {"pieces.rest"}
    return set()


# Schema 29 (Q983a, `docs/plan-rest.md` §3.1): the longest rest a type may declare. At 4 a
# piece moves at most one turn in five; it bounds each player's resting list (one piece
# enters a turn and none stays longer than this) and the key's table. A showcase that needs
# more raises it here and in the core (`rust/src/tables.rs::REST_MAX`).
REST_MAX = 4

# The one message the old switch is refused with, at every version (§4.4).
RETIRED_REST = (
    'rules.simultaneous.rest was retired in schema 29: write "rest": 1 on each piece type '
    "instead (pieces[].rest), which is the same rest for a game in rounds"
)


def has_retired_rest(spec_or_rules: Any) -> bool:
    """Whether a spec (or its ``rules``) writes schema 27's ``rules.simultaneous.rest``, which
    schema 29 retired (`docs/plan-rest.md` §4.4), whatever its value."""
    rules = spec_or_rules
    if isinstance(spec_or_rules, dict) and "pieces" in spec_or_rules:
        rules = spec_or_rules.get("rules")
    return (
        isinstance(rules, dict)
        and isinstance(rules.get("simultaneous"), dict)
        and "rest" in rules["simultaneous"]
    )


def _refuse_the_retired_rest(spec: dict[str, Any], errors: list[str]) -> None:
    if has_retired_rest(spec):
        errors.append(RETIRED_REST)


def has_rests(spec_or_pieces: Any) -> bool:
    """Whether any piece type writes a ``rest`` (schema 29), whatever its value."""
    pieces = spec_or_pieces.get("pieces") if isinstance(spec_or_pieces, dict) else spec_or_pieces
    return any(isinstance(piece, dict) and "rest" in piece for piece in _listed(pieces))


def piece_rests(spec_or_pieces: Any) -> dict[str, int]:
    """Each declared piece id's rest (schema 29, `docs/plan-rest.md` §3.1): its ``rest``, and
    0 (it never rests, the default) where it writes none or something the validator refuses.
    The one reader of the key, shared by the validator, the compiler and every surface."""
    pieces = spec_or_pieces.get("pieces") if isinstance(spec_or_pieces, dict) else spec_or_pieces
    rests: dict[str, int] = {}
    for piece in _listed(pieces):
        if not isinstance(piece, dict) or not isinstance(piece.get("id"), str):
            continue
        value = piece.get("rest", 0)
        valid = isinstance(value, int) and not isinstance(value, bool) and 0 < value <= REST_MAX
        rests[piece["id"]] = value if valid else 0
    return rests


def _check_rest_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Schema 29 (Q983a): ``pieces[].rest`` and every refusal of `docs/plan-rest.md` §3.1, in
    words. The value is a count of turns, 1 to ``REST_MAX``; the default (0) is never written;
    a type no player moves never rests; and a turn of several moves is refused beside it."""
    written = [
        (index, piece) for index, piece in enumerate(pieces)
        if isinstance(piece, dict) and "rest" in piece
    ]
    if not written:
        return
    portals = portal_piece_ids(rules)
    wild = _never_moved(rules) - portals
    for index, piece in written:
        value = piece["rest"]
        if value == 0 and not isinstance(value, bool):
            errors.append(f"pieces[{index}].rest 0 is the default (it never rests); leave it out")
        elif (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            or value > REST_MAX
        ):
            errors.append(
                f"pieces[{index}].rest must be a whole number of turns from 1 to {REST_MAX}, "
                f"got {value!r}"
            )
        pid = piece.get("id")
        if pid in portals:
            errors.append(f"pieces[{index}].rest is on a portal, which nobody moves; leave it out")
        elif pid in wild:
            errors.append(
                f"pieces[{index}].rest is on a wild piece (rules.npcs), which moves in its own "
                "phase, never as a player's move; leave it out"
            )
        elif not _listed(piece.get("moves")) and not piece_has_stock(rules, pid):
            errors.append(
                f"pieces[{index}].rest is on a piece with no moves that no hand can drop: a "
                "piece that never moves never rests; leave it out"
            )
    if turn_is_long(rules):
        errors.append(
            "pieces[].rest beside a turn longer than one move (rules.turn moves or ramp): "
            "inside a turn of several moves a rest would need a second rule, for a piece that "
            "moved earlier in the same turn; use one move per turn, or leave the rest out"
        )


def _schema_thirty_uses(spec: dict[str, Any]) -> set[str]:
    """Everything in this spec that schema 30 added, by name (Q999a, `docs/plan-passive.md`).

    * ``pieces.sight.freeze`` -- a line of sight whose pieces may not move and attack nothing.
    * ``pieces.sight.disarm`` -- a line of sight whose pieces may not capture and attack nothing.
    * ``pieces.sight.shield`` -- a line of sight whose pieces may not be captured (a shielded
      royal is never in check).
    * ``pieces.sight.through`` -- a sight that sees past pieces (an aura) rather than stopping
      at the first one (a ray).

    ``whose`` and ``affects`` are values of a sight's keys, as ``mode`` is of a move's, and
    have no names of their own (§4.1). Collected whatever else the entry says.
    """
    used: set[str] = set()
    for piece in _listed(spec.get("pieces")):
        if not isinstance(piece, dict) or "sight" not in piece:
            continue
        for entry in _listed(piece["sight"]):
            if not isinstance(entry, dict):
                continue
            effect = entry.get("effect")
            if isinstance(effect, str) and effect in SIGHT_EFFECTS:
                used.add(f"pieces.sight.{effect}")
            if "through" in entry:
                used.add("pieces.sight.through")
    return used


# Schema 30 (Q999a, `docs/plan-passive.md` §2): lines of sight. A type declares at most
# ``SIGHT_MAX`` of them (§2.2); the cap bounds the core's reverse table and is raised by a
# showcase that needs more.
SIGHT_MAX = 4
SIGHT_EFFECTS = ("freeze", "disarm", "shield")
SIGHT_WHOSE = ("enemy", "own", "any")
_SIGHT_KEYS = {
    "vectors", "symmetry", "range", "min_range", "effect", "whose", "affects", "through",
}


class Sight(NamedTuple):
    """One line of sight as `piece_sights` reads it: a slide's geometry, its effect, whose
    pieces it touches, the types it touches (``None`` for every type) and whether it sees
    through pieces (an aura) or stops at the first (a ray)."""

    vectors: tuple[tuple[int, int], ...]
    symmetry: str
    range: int
    min_range: int
    effect: str
    whose: str
    affects: tuple[str, ...] | None
    through: bool


def has_sights(spec_or_pieces: Any) -> bool:
    """Whether any piece type writes a ``sight`` (schema 30), whatever its value."""
    pieces = spec_or_pieces.get("pieces") if isinstance(spec_or_pieces, dict) else spec_or_pieces
    return any(isinstance(piece, dict) and "sight" in piece for piece in _listed(pieces))


def _sight_entry(entry: Any) -> Sight | None:
    """One well-formed sight entry, or ``None`` for anything the validator refuses."""
    if not isinstance(entry, dict):
        return None
    vectors = entry.get("vectors")
    if not isinstance(vectors, list) or not vectors or not all(
        isinstance(v, list) and len(v) == 2
        and all(isinstance(c, int) and not isinstance(c, bool) for c in v)
        and v != [0, 0]
        for v in vectors
    ):
        return None
    symmetry = entry.get("symmetry", "none")
    rng = entry.get("range", 0)
    low = entry.get("min_range", 1)
    effect = entry.get("effect")
    whose = entry.get("whose")
    affects = entry.get("affects")
    through = entry.get("through", False)
    if (
        symmetry not in SYMMETRIES
        or isinstance(rng, bool) or not isinstance(rng, int) or rng < 0
        or isinstance(low, bool) or not isinstance(low, int) or low < 1
        or (rng > 0 and low > rng)
        or effect not in SIGHT_EFFECTS or whose not in SIGHT_WHOSE
        or through is not True and "through" in entry
        or (affects is not None and (
            not isinstance(affects, list) or not affects
            or not all(isinstance(pid, str) for pid in affects)
        ))
    ):
        return None
    return Sight(
        vectors=tuple((v[0], v[1]) for v in vectors),
        symmetry=symmetry,
        range=rng,
        min_range=low,
        effect=effect,
        whose=whose,
        affects=tuple(affects) if affects is not None else None,
        through=through is True,
    )


def piece_sights(spec_or_pieces: Any) -> dict[str, tuple[Sight, ...]]:
    """Each declared piece id's lines of sight (schema 30, `docs/plan-passive.md` §2.1), in the
    order written, for every id that writes at least one well-formed entry; an id with none is
    absent. The one reader of the key, shared by the validator, the compiler and every
    surface."""
    pieces = spec_or_pieces.get("pieces") if isinstance(spec_or_pieces, dict) else spec_or_pieces
    sights: dict[str, tuple[Sight, ...]] = {}
    for piece in _listed(pieces):
        if not isinstance(piece, dict) or not isinstance(piece.get("id"), str):
            continue
        read = tuple(
            sight for sight in (_sight_entry(entry) for entry in _listed(piece.get("sight")))
            if sight is not None
        )
        if read:
            sights[piece["id"]] = read
    return sights


def sight_affected_ids(sight: Sight, declared: Iterable[str], rules: Any) -> list[str]:
    """The ids a sight's effect can touch: its ``affects``, or every declared type that is
    neither wild nor a portal end (§2.3)."""
    blank = _never_moved(rules)
    if sight.affects is not None:
        return [pid for pid in sight.affects if pid not in blank]
    return [pid for pid in declared if pid not in blank]


def _piece_can_capture(piece: dict[str, Any], declared: set[str]) -> bool:
    """Whether a piece type has any capture: a move that may land on a piece (or a sweep),
    and a capture matrix that lets it take something."""
    return _capture_is_possible([piece], declared)


def _piece_can_be_captured(pid: str, pieces: list[Any]) -> bool:
    """Whether any type has a capture that may land on a ``pid``."""
    for attacker in pieces:
        if not isinstance(attacker, dict):
            return True
        moves = _listed(attacker.get("moves"))
        if all(
            not isinstance(move, dict)
            or (move.get("mode", "both") == "move" and move.get("kind") != "sweep")
            for move in moves
        ):
            continue
        allowed = attacker.get("captures")
        if not isinstance(allowed, list) or pid in allowed:
            return True
    return False


def sight_can_bind(spec: dict[str, Any], source: str, sight: Sight) -> str | None:
    """``None`` when this sight of ``source``'s can ever touch a piece its effect means
    something to, else why not, in words (§2.1, *a sight that can never bind*). The reading is
    per side: the types each player can ever hold (`simultaneous_holdings`, with promotion,
    hands, strikes and contacts), so a sight on one army's type that touches only the other
    army's own types is seen to touch nothing."""
    pieces = [piece for piece in _listed(spec.get("pieces")) if isinstance(piece, dict)]
    by_id = {piece["id"]: piece for piece in pieces if isinstance(piece.get("id"), str)}
    declared = set(by_id)
    rules = spec.get("rules") if isinstance(spec.get("rules"), dict) else {}
    held = simultaneous_holdings(spec)
    affected = set(sight_affected_ids(sight, list(by_id), rules))
    reachable: set[str] = set()
    for owner in (0, 1):
        if source not in held[owner]:
            continue
        sides = {"enemy": (1 - owner,), "own": (owner,), "any": (0, 1)}[sight.whose]
        for side in sides:
            reachable |= affected & held[side]
    if not reachable:
        return "no type it affects is ever held by the side it affects"
    if sight.effect == "freeze" and not any(_listed(by_id[pid].get("moves")) for pid in reachable):
        return "a freeze on types that have no move"
    if sight.effect == "disarm" and not any(
        _piece_can_capture(by_id[pid], declared) for pid in reachable
    ):
        return "a disarm on types that can capture nothing"
    if sight.effect == "shield" and not any(
        _piece_can_be_captured(pid, pieces) for pid in reachable
    ):
        return "a shield on types that nothing can capture"
    return None


def _check_sight_geometry(
    entry: dict[str, Any], loc: str, errors: list[str], hex_lattice: bool
) -> None:
    """A sight's geometry keys, by the rules and in the words of a slide's (§2.1)."""
    symmetry = entry.get("symmetry", "none")
    if symmetry not in SYMMETRIES:
        errors.append(f"{loc}.symmetry must be one of {SYMMETRIES}, got {symmetry!r}")
    else:
        errors.extend(
            _symmetry_refusals(symmetry, entry.get("vectors"), f"{loc}.symmetry", hex_lattice)
        )
    rng = entry.get("range", 0)
    if isinstance(rng, bool) or not isinstance(rng, int) or rng < 0:
        errors.append(f"{loc}.range must be a non-negative integer")
    elif rng > MAX_BOARD_DIM:
        errors.append(f"{loc}.range may not exceed {MAX_BOARD_DIM}")
    min_range = entry.get("min_range", 1)
    if isinstance(min_range, bool) or not isinstance(min_range, int) or min_range < 1:
        errors.append(f"{loc}.min_range must be a positive integer")
    elif isinstance(rng, int) and rng > 0 and min_range > rng:
        errors.append(f"{loc}.min_range ({min_range}) may not exceed range ({rng})")
    _validate_vectors(entry.get("vectors"), loc, errors)


def _check_sight_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Schema 30 (Q999a): ``pieces[].sight`` and every refusal of `docs/plan-passive.md` §2.1
    and §2.8, in words."""
    written = [
        (index, piece) for index, piece in enumerate(pieces)
        if isinstance(piece, dict) and "sight" in piece
    ]
    if not written:
        return
    declared = [
        piece["id"] for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    ]
    board = spec.get("board")
    hex_lattice = isinstance(board, dict) and board.get("lattice") == "hex"
    portals = portal_piece_ids(rules)
    wild = _never_moved(rules) - portals
    shields = False
    for index, piece in written:
        where = f"pieces[{index}].sight"
        sight = piece["sight"]
        if not isinstance(sight, list):
            errors.append(f"{where} must be an array of 1 to {SIGHT_MAX} lines of sight")
            continue
        if not sight:
            errors.append(f"{where} is empty, which sees nothing; leave it out")
        elif len(sight) > SIGHT_MAX:
            errors.append(
                f"{where} holds {len(sight)} lines of sight; a type may have at most {SIGHT_MAX}"
            )
        pid = piece.get("id")
        if pid in portals:
            errors.append(f"{where} is on a portal, which carries no sight; leave it out")
        elif pid in wild:
            errors.append(
                f"{where} is on a wild piece (rules.npcs), which is nobody's and carries no "
                "sight; leave it out"
            )
        for number, entry in enumerate(sight):
            loc = f"{where}[{number}]"
            if not isinstance(entry, dict):
                errors.append(f"{loc} must be an object")
                continue
            _check_keys(entry, _SIGHT_KEYS, loc, errors, required={"vectors", "effect", "whose"})
            _check_sight_geometry(entry, loc, errors, hex_lattice)
            effect = entry.get("effect")
            if "effect" in entry and effect not in SIGHT_EFFECTS:
                errors.append(f"{loc}.effect must be one of {SIGHT_EFFECTS}, got {effect!r}")
            shields |= effect == "shield"
            whose = entry.get("whose")
            if "whose" in entry and whose not in SIGHT_WHOSE:
                errors.append(f"{loc}.whose must be one of {SIGHT_WHOSE}, got {whose!r}")
            if "through" in entry and entry["through"] is not True:
                errors.append(
                    f"{loc}.through is true or absent: absent is a ray, stopped by the first "
                    "piece it meets, so false is the default; leave it out"
                )
            if "affects" in entry:
                affects = entry["affects"]
                if not isinstance(affects, list):
                    errors.append(f"{loc}.affects must be an array of piece ids")
                elif not affects:
                    errors.append(f"{loc}.affects is empty, which affects nothing; leave it out")
                else:
                    for named in affects:
                        if not isinstance(named, str) or named not in declared:
                            errors.append(f"{loc}.affects entry {named!r} is not a declared piece")
                        elif named in portals:
                            errors.append(
                                f"{loc}.affects names {named!r}, a portal, which nothing affects"
                            )
                        elif named in wild:
                            errors.append(
                                f"{loc}.affects names {named!r}, a wild piece (rules.npcs), which "
                                "nothing affects"
                            )
                    if len(set(map(str, affects))) != len(affects):
                        errors.append(f"{loc}.affects repeats a piece id")
            read = _sight_entry(entry)
            if read is not None and isinstance(pid, str) and pid not in portals | wild:
                why = sight_can_bind(spec, pid, read)
                if why is not None:
                    errors.append(f"{loc} affects nothing that could be there: {why}")
    if rules.get("portals"):
        errors.append(
            "pieces[].sight beside rules.portals: an attack through a portal would need the "
            "sight's reading of attack too, and a sight through a portal a rule of its own; "
            "leave one out"
        )
    if rules.get("explode_on_capture"):
        errors.append(
            "pieces[].sight beside rules.explode_on_capture: what a blast does to a shielded "
            "piece, and how its legality reads a frozen attacker, are rules of their own; leave "
            "one out"
        )
    if shields and uses_sweep(spec):
        errors.append(
            "a shield (pieces[].sight effect 'shield') beside a sweep move: whether a sweep "
            "passing over a shielded piece removes it or is stopped by it would be a second "
            "rule; leave one out"
        )
    if shields and rules.get("npcs"):
        errors.append(
            "a shield (pieces[].sight effect 'shield') beside rules.npcs: whether a wild "
            "piece's step takes a shielded piece would be a second rule; leave one out"
        )


def has_speeds(spec_or_pieces: Any) -> bool:
    """Whether any piece type writes a ``speed`` (schema 28), whatever its value."""
    pieces = spec_or_pieces.get("pieces") if isinstance(spec_or_pieces, dict) else spec_or_pieces
    return any(isinstance(piece, dict) and "speed" in piece for piece in _listed(pieces))


def piece_speeds(spec_or_pieces: Any) -> dict[str, int]:
    """Each declared piece id's speed (schema 28, `docs/plan-speed.md` §2.1): its ``speed``,
    and 0 (the slowest, the default) where it writes none or something the validator
    refuses. The one reader of the key, shared by the validator, the compiler and every
    surface."""
    pieces = spec_or_pieces.get("pieces") if isinstance(spec_or_pieces, dict) else spec_or_pieces
    speeds: dict[str, int] = {}
    for piece in _listed(pieces):
        if not isinstance(piece, dict) or not isinstance(piece.get("id"), str):
            continue
        value = piece.get("speed", 0)
        valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        speeds[piece["id"]] = value if valid else 0
    return speeds


def _never_moved(rules: Any) -> set[str]:
    """The declared types no player ever moves: portal ends and wild pieces."""
    listed = rules.get("npcs") if isinstance(rules, dict) else None
    wild = {npc.get("piece") for npc in _listed(listed) if isinstance(npc, dict)}
    return portal_piece_ids(rules) | {pid for pid in wild if isinstance(pid, str)}


def speed_ranks_in_use(spec: dict[str, Any]) -> list[int]:
    """The distinct speeds of the types a player can move, 0 counted for every one that
    names none, in order (§2.1: they must be 0 to *k*, none skipped)."""
    skipped = _never_moved(spec.get("rules"))
    return sorted({speed for pid, speed in piece_speeds(spec).items() if pid not in skipped})


def speeds_order_a_round(spec: dict[str, Any]) -> bool:
    """Whether some type one player can hold and some type the other can hold differ in
    speed (over `simultaneous_holdings`); otherwise every round is an equal-speed round and
    the speeds are inert (§4.1). False for a game that declares none."""
    if not has_speeds(spec):
        return False
    speeds = piece_speeds(spec)
    first, second = simultaneous_holdings(spec)
    return any(
        speeds[x] != speeds[y] for x in first for y in second if x in speeds and y in speeds
    )


def has_equal_speed_opponent(spec: dict[str, Any], piece_id: str) -> bool:
    """Whether, for some player who can hold ``piece_id``, the other player can hold a type
    of the same speed. A collision (and a swap that meets) happens only between equal
    speeds (§2.3), so a type with none never collides. True where no player can hold the
    type, which is not speed's business."""
    speeds = piece_speeds(spec)
    held = simultaneous_holdings(spec)
    holders = [player for player in (0, 1) if piece_id in held[player]]
    if not holders:
        return True
    mine = speeds.get(piece_id, 0)
    return any(speeds.get(other, 0) == mine for player in holders for other in held[1 - player])


def _any_equal_speed_meeting(spec: dict[str, Any]) -> bool:
    """Whether some type one player can hold and some type the other can hold share a
    speed, so that a collision can happen at all (§2.3)."""
    speeds = piece_speeds(spec)
    first, second = simultaneous_holdings(spec)
    return any(speeds.get(x, 0) == speeds.get(y, 0) for x in first for y in second)


def _check_speed_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Schema 28 (Q943a): ``pieces[].speed`` and every refusal of `docs/plan-speed.md` §4.1,
    in words. A speed orders the two moves of a round, so it needs the round rule; speeds
    are ranks, so one game has one spelling; and they must be able to order some round."""
    written = [
        (index, piece) for index, piece in enumerate(pieces)
        if isinstance(piece, dict) and "speed" in piece
    ]
    if not written:
        return
    if not is_simultaneous(rules):
        errors.append(
            "pieces[].speed orders the two moves of a round, and this game has none "
            "(rules.simultaneous); leave it out"
        )
        return
    malformed = False
    portals = portal_piece_ids(rules)
    wild = _never_moved(rules) - portals
    for index, piece in written:
        value = piece["speed"]
        if value == 0 and not isinstance(value, bool):
            errors.append(f"pieces[{index}].speed 0 is the default (the slowest); leave it out")
            malformed = True
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(
                f"pieces[{index}].speed must be a whole number of at least 1, got {value!r}"
            )
            malformed = True
        if piece.get("id") in portals:
            errors.append(f"pieces[{index}].speed is on a portal, which never moves; leave it out")
        if piece.get("id") in wild:
            errors.append(
                f"pieces[{index}].speed is on a wild piece (rules.npcs), which steps after both "
                "players' moves whatever its speed; leave it out"
            )
    if malformed:
        return
    whole = {**spec, "rules": rules, "pieces": pieces}
    ranks = speed_ranks_in_use(whole)
    if ranks != list(range(len(ranks))):
        dense = ", ".join(f"{old} as {new}" for new, old in enumerate(ranks) if old != new)
        errors.append(
            "pieces[].speed: speeds are ranks, and only their order is a rule, so the speeds "
            "in use (0 for every piece that names none) must be 0, 1, 2, ... with none "
            f"skipped; they are {ranks}: write {dense}"
        )
        return
    if not speeds_order_a_round(whole):
        errors.append(
            "pieces[].speed never orders a round: every piece one player can hold has the "
            "speed of every piece the other can hold; leave it out"
        )


# Schema 27 (Q936): the third value of ``rules.extinction.per_type``, beside ``true`` and
# ``false``. Each listed kind is counted on its own against a side that starts with it.
EXTINCTION_PER_TYPE_OWN = "own"


# Schema 26 (Q918a): the values ``rules.simultaneous.tie`` may take. Absent is the user's
# table, both pieces eliminated, and writing it out as ``"both"`` is refused so that one game
# has one spelling (`docs/reference/rounds.md` §5.1).
SIMULTANEOUS_TIES = ("initiative", "impossible")
# Schema 27 (Q942a, `docs/archive/plan-round-chase.md` §3, §4): the three switches against the royal
# chase, each off when absent and each with exactly one value that turns it on. Writing the
# off value out (``false``, or ``"chase"``/``"draw"``) is refused, one game one spelling.
# Schema 29 (Q983a) retired ``rest`` for ``pieces[].rest``: it is still a key the shape check
# lets through, so that the one message a spec carrying it gets is the retirement's
# (`RETIRED_REST`), never an unknown key beside it.
SIMULTANEOUS_CHASE_KEYS = ("swap", "both")
SIMULTANEOUS_SWAP_COLLISION = "collision"
SIMULTANEOUS_BOTH_SCORE = "score"
_SIMULTANEOUS_KEYS = {"tie", "rest", *SIMULTANEOUS_CHASE_KEYS}


def simultaneous_switch(rules: Any, key: str) -> bool:
    """Whether a simultaneous game turns on one of schema 27's chase switches (Q942a).

    The one reader of the spelling, shared by the validator, the compiler and every
    surface: ``swap`` is on at ``"collision"``, ``both`` at ``"score"``; anything else,
    absent included, is off. Schema 29 (Q983a) retired the third, ``rest``, for
    ``pieces[].rest`` (`piece_rests`)."""
    if isinstance(rules, dict) and "pieces" in rules:
        rules = rules.get("rules")
    if not is_simultaneous(rules):
        return False
    value = rules["simultaneous"].get(key)
    return value == {"swap": SIMULTANEOUS_SWAP_COLLISION, "both": SIMULTANEOUS_BOTH_SCORE}[key]


def simultaneous_swap_pair(spec: dict[str, Any]) -> tuple[str, str] | None:
    """A player-0 type and a player-1 type that may each take the other, or ``None``.

    A swap is two moves each ending on the other's origin while the other stands there, so
    each is a capture of the other: without such a pair no swap can happen and
    ``swap: "collision"`` can never fire (`docs/archive/plan-round-chase.md` §3 R2). Over
    `simultaneous_holdings`, so a type a player can come to hold counts; a piece takes a
    type when its ``captures`` allows it and it has a move that may capture. Schema 28
    (Q943a): only a pair of equal speed, since between unequal speeds the faster takes the
    slower where it stands and no swap happens (`docs/plan-speed.md` §2.5, case B)."""
    pieces = [piece for piece in spec.get("pieces") or [] if isinstance(piece, dict)]
    by_id = {piece["id"]: piece for piece in pieces if isinstance(piece.get("id"), str)}
    speeds = piece_speeds(pieces)

    def takes(attacker: str, victim: str) -> bool:
        piece = by_id[attacker]
        captures = piece.get("captures")
        if isinstance(captures, list) and victim not in captures:
            return False
        return any(
            isinstance(move, dict) and move.get("mode", "both") != "move"
            for move in piece.get("moves") or []
        )

    first, second = simultaneous_holdings(spec)
    for x in sorted(first):
        for y in sorted(second):
            if (
                x in by_id and y in by_id and speeds[x] == speeds[y]
                and takes(x, y) and takes(y, x)
            ):
                return x, y
    return None


def is_simultaneous(spec_or_rules: Any) -> bool:
    """Whether a spec (or its ``rules``) is played in rounds (schema 26).

    The one reader of the spelling: present as an object, the game is simultaneous."""
    rules = spec_or_rules
    if isinstance(spec_or_rules, dict) and "pieces" in spec_or_rules:
        rules = spec_or_rules.get("rules")
    return isinstance(rules, dict) and isinstance(rules.get("simultaneous"), dict)


def beats_of(piece: dict[str, Any], declared: Iterable[str]) -> frozenset[str]:
    """The ids ``piece`` eliminates in a collision: its ``beats``, else its ``captures``,
    else every declared piece (`docs/reference/rounds.md` §2.3)."""
    for key in ("beats", "captures"):
        named = piece.get(key)
        if isinstance(named, list):
            return frozenset(entry for entry in named if isinstance(entry, str))
    return frozenset(declared)


def simultaneous_holdings(spec: dict[str, Any]) -> tuple[set[str], set[str]]:
    """The piece ids each player can ever have on the board, for the ``impossible`` proof.

    An over-approximation in the safe direction, per player: placements (both under a
    mirrored setup), a stock, then the closure under promotion targets, a strike chain's
    or a contact link's results, a side switch (the type joins the other player), and a
    hand that fills from captures (the other player's types join). A pair it calls
    possible may never meet in play; a pair it leaves out never can.
    """
    pieces = [piece for piece in spec.get("pieces") or [] if isinstance(piece, dict)]
    declared = {piece["id"] for piece in pieces if isinstance(piece.get("id"), str)}
    rules: dict[str, Any] = spec["rules"] if isinstance(spec.get("rules"), dict) else {}
    held: list[set[str]] = [set(), set()]
    setup: dict[str, Any] = spec["setup"] if isinstance(spec.get("setup"), dict) else {}
    mirrored = setup.get("mirror", True) is not False
    for placement in setup.get("placements") or []:
        if not isinstance(placement, dict) or placement.get("piece") not in declared:
            continue
        players = (0, 1) if mirrored else (placement.get("player", 0),)
        for player in players:
            if player in (0, 1):
                held[player].add(placement["piece"])
    drops: dict[str, Any] | None = rules["drops"] if isinstance(rules.get("drops"), dict) else None
    if drops is not None and isinstance(drops.get("stock"), dict):
        for player in (0, 1):
            held[player] |= declared.intersection(drops["stock"])
    hands_fill = drops is not None and drops.get("from_captures", True) is not False
    demoted = {
        piece["id"]: piece["promoted_from"] for piece in pieces
        if isinstance(piece.get("id"), str) and piece.get("promoted_from") in declared
    }
    promotions = rules.get("promotion")
    promotions = promotions if isinstance(promotions, list) else [promotions]
    strike: dict[str, Any] = rules["strike"] if isinstance(rules.get("strike"), dict) else {}
    contacts: dict[str, Any] = (
        rules["contacts"] if isinstance(rules.get("contacts"), dict) else {}
    )
    switches = set(strike.get("switches_sides") or []) & declared
    while True:
        before = (frozenset(held[0]), frozenset(held[1]))
        for player in (0, 1):
            for rule in promotions:
                if not isinstance(rule, dict) or rule.get("piece") not in held[player]:
                    continue
                targets = rule.get("to")
                by_player = rule.get("to_by_player")
                if isinstance(by_player, dict):
                    targets = by_player.get(str(player))
                held[player] |= declared.intersection(targets or [])
            for link in strike.get("chain") or []:
                if isinstance(link, dict) and link.get("from") in held[player]:
                    held[player] |= declared.intersection([link.get("to")])
            for link in contacts.get("links") or []:
                if not isinstance(link, dict):
                    continue
                if link.get("mover") in held[player]:
                    held[player] |= declared.intersection([link.get("mover_to")])
                if link.get("onto") in held[player]:
                    held[player] |= declared.intersection([link.get("onto_to")])
            held[1 - player] |= held[player] & switches
            if hands_fill:
                held[player] |= {demoted.get(pid, pid) for pid in held[1 - player]}
        if (frozenset(held[0]), frozenset(held[1])) == before:
            return held[0], held[1]


def simultaneous_tie_pair(spec: dict[str, Any]) -> tuple[str, str] | None:
    """The first pair of a player-0 type and a player-1 type whose collision is a tie (both
    beat the other, or neither does), or ``None`` when no tie can occur (§3.1, F). Schema 28
    (Q943a): over pairs of equal speed only, since pieces of different speeds never collide
    (`docs/plan-speed.md` §2.3, §3.3)."""
    pieces = [piece for piece in spec.get("pieces") or [] if isinstance(piece, dict)]
    by_id = {piece["id"]: piece for piece in pieces if isinstance(piece.get("id"), str)}
    declared = list(by_id)
    speeds = piece_speeds(pieces)
    first, second = simultaneous_holdings(spec)
    for x in sorted(first):
        for y in sorted(second):
            if speeds.get(x, 0) != speeds.get(y, 0):
                continue
            x_beats = y in beats_of(by_id[x], declared)
            y_beats = x in beats_of(by_id[y], declared)
            if x_beats == y_beats:
                return x, y
    return None


def spy_ids(spec_or_pieces: Any) -> list[str]:
    """The ids of the piece types a spec flags ``"spy": true`` (schema 27, Q935a), in
    declared order. The one reader of the flag, shared by the validator, the compiler and
    every surface (`docs/plan-spy.md` §1.2)."""
    pieces = spec_or_pieces
    if isinstance(spec_or_pieces, dict):
        pieces = spec_or_pieces.get("pieces")
    return [
        piece["id"] for piece in _listed(pieces)
        if isinstance(piece, dict) and piece.get("spy") is True and isinstance(piece.get("id"), str)
    ]


def is_spy_game(spec: Any) -> bool:
    """Whether a spec is a spy game: some piece type is flagged ``"spy": true``."""
    return bool(spy_ids(spec))


def placed_ids(spec: dict[str, Any]) -> tuple[set[str], set[str]]:
    """The piece ids each player has on the opening board, from ``setup.placements`` (both
    players under a mirrored setup, else the placement's player)."""
    held: tuple[set[str], set[str]] = (set(), set())
    setup: dict[str, Any] = spec["setup"] if isinstance(spec.get("setup"), dict) else {}
    mirrored = setup.get("mirror", True) is not False
    for placement in _listed(setup.get("placements")):
        if not isinstance(placement, dict) or not isinstance(placement.get("piece"), str):
            continue
        if not placement.get("squares"):
            continue
        for player in (0, 1) if mirrored else (placement.get("player", 0),):
            if player in (0, 1):
                held[player].add(placement["piece"])
    return held


def _spy_gains(rules: dict[str, Any], spies: set[str]) -> list[str]:
    """Every route by which a player could come to hold a spy it did not start with
    (`docs/plan-spy.md` §1.2, *never gained*), each as the words of its refusal."""
    found: list[str] = []
    for entry in _promotion_entries(rules):
        if isinstance(entry, dict):
            for target in sorted(_promotion_target_ids(entry) & spies):
                found.append(f"rules.promotion turns a piece into the spy {target!r}")
    strike: dict[str, Any] = rules["strike"] if isinstance(rules.get("strike"), dict) else {}
    for link in _listed(strike.get("chain")):
        if isinstance(link, dict) and _in(link.get("to"), spies):
            found.append(f"rules.strike.chain turns a piece into the spy {link['to']!r}")
    switching = _str_set(_listed(strike.get("switches_sides")))
    for spy in sorted(switching & spies):
        found.append(
            f"rules.strike.switches_sides names the spy {spy!r}, which would join the other "
            "player when it captures"
        )
    contacts: dict[str, Any] = (
        rules["contacts"] if isinstance(rules.get("contacts"), dict) else {}
    )
    for link in _listed(contacts.get("links")):
        if not isinstance(link, dict):
            continue
        for key in ("mover_to", "onto_to"):
            if _in(link.get(key), spies):
                found.append(f"a rules.contacts link turns a piece into the spy {link[key]!r}")
    drops = rules.get("drops")
    if isinstance(drops, dict):
        stock = drops.get("stock")
        for spy in sorted(spies & set(stock if isinstance(stock, dict) else ())):
            found.append(f"rules.drops.stock puts the spy {spy!r} in a hand")
        if drops.get("from_captures", True) is not False:
            found.append(
                "rules.drops fills a hand from captures, and a taken spy would come back as "
                "its taker's (set from_captures to false)"
            )
    return found


def _spy_can_be_removed(spec: dict[str, Any], rules: dict[str, Any], spy: str, by: set[str]) -> bool:
    """Whether anything could take a spy off the board: a type ``by`` (the other player's
    holdings) that may capture it or beats it in a collision, an explosion, a wild piece,
    a strike or contact rule, or a promotion of the spy itself (a spy promoted is lost)."""
    if any(key in rules for key in ("explode_on_capture", "npcs", "strike", "contacts")):
        return True
    if any(
        isinstance(entry, dict) and entry.get("piece") == spy for entry in _promotion_entries(rules)
    ):
        return True
    pieces = [piece for piece in _listed(spec.get("pieces")) if isinstance(piece, dict)]
    declared = [piece["id"] for piece in pieces if isinstance(piece.get("id"), str)]
    for piece in pieces:
        if piece.get("id") not in by:
            continue
        if spy in beats_of(piece, declared):
            return True
        captures = piece.get("captures")
        if isinstance(captures, list) and spy not in captures:
            continue
        if any(
            isinstance(move, dict) and move.get("mode", "both") != "move"
            for move in _listed(piece.get("moves"))
        ):
            return True
    return False


def _check_spy_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Schema 27 (Q935a): ``pieces[].spy`` and every refusal of `docs/plan-spy.md` §1.7, in
    words. A spy decides when a game plays in rounds, so it needs the round rule; it may be
    lost but never gained, so the phases only move forward; both players start with one; and
    something must be able to remove it, or the rule never fires."""
    flagged: list[tuple[int, dict[str, Any]]] = []
    for index, piece in enumerate(pieces):
        if not isinstance(piece, dict) or "spy" not in piece:
            continue
        value = piece["spy"]
        if value is False:
            errors.append(f"pieces[{index}].spy false is the default (not a spy); leave it out")
        elif value is not True:
            errors.append(f"pieces[{index}].spy must be true, got {value!r}")
        else:
            flagged.append((index, piece))
    if not flagged:
        return
    if not is_simultaneous(rules):
        errors.append(
            "pieces[].spy: a spy decides when a game plays in rounds; declare "
            "rules.simultaneous (at least {})"
        )
        return
    spies = {piece["id"] for _, piece in flagged if isinstance(piece.get("id"), str)}
    for index, piece in flagged:
        if piece.get("royal") is True:
            errors.append(
                f"pieces[{index}].spy on a royal piece: losing a royal ends the game, so its "
                "loss could never start a round"
            )
    portals = portal_piece_ids(rules)
    wild = {
        npc.get("piece") for npc in _listed(rules.get("npcs")) if isinstance(npc, dict)
    }
    for index, piece in flagged:
        if piece.get("id") in portals:
            errors.append(f"pieces[{index}].spy on a portal, which is nobody's piece; leave it out")
        if piece.get("id") in wild:
            errors.append(
                f"pieces[{index}].spy on a wild piece (rules.npcs), which is nobody's piece; "
                "leave it out"
            )
    started = placed_ids(spec)
    for player in (0, 1):
        if not started[player] & spies:
            errors.append(
                f"player {player} starts with no spy on the board, so the game would begin in "
                "rounds; place a spy for each player"
            )
    for route in _spy_gains(rules, spies):
        errors.append(
            f"pieces[].spy beside {route}: a spy may be lost but never gained, so the game "
            "never goes back to taking turns"
        )
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and extinction.get("per_type") in (True, EXTINCTION_PER_TYPE_OWN):
        for spy in sorted(spies & _str_set(_listed(extinction.get("pieces")))):
            errors.append(
                f"rules.extinction counts the spy {spy!r} on its own (per_type): losing the spy "
                "would end the game, so its rounds would never be played"
            )
    holdings = simultaneous_holdings({**spec, "rules": rules, "pieces": pieces})
    for player in (0, 1):
        mine = started[player] & spies
        if mine and not any(
            _spy_can_be_removed(spec, rules, spy, holdings[1 - player]) for spy in mine
        ):
            errors.append(
                f"nothing can remove player {player}'s spy (no piece of the other player may "
                "capture it or beat it in a collision), so the rule never fires"
            )


def _check_chase_switches(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Schema 27 (Q942a): the chase switches' spellings and refusals
    (`docs/archive/plan-round-chase.md` §3 R2, X1). R1's ``rest`` was retired by schema 29
    (`RETIRED_REST`, refused in `validate_spec` at every version)."""
    rule = rules["simultaneous"]
    if "swap" in rule:
        swap = rule["swap"]
        if swap == "chase":
            errors.append(
                'rules.simultaneous.swap "chase" is the default (two pieces swapping squares '
                "pass each other); leave it out"
            )
        elif swap != SIMULTANEOUS_SWAP_COLLISION:
            errors.append(
                f'rules.simultaneous.swap must be "{SIMULTANEOUS_SWAP_COLLISION}", got {swap!r}'
            )
        elif simultaneous_swap_pair({**spec, "rules": rules, "pieces": pieces}) is None:
            speed = " and has its speed (pieces[].speed)" if has_speeds(pieces) else ""
            errors.append(
                'rules.simultaneous.swap "collision" can never fire: no piece of one player '
                f"can take a piece of the other that can take it back{speed}, so no two moves "
                "can each end on the other's square; leave it out"
            )
    if "both" in rule:
        both = rule["both"]
        if both == "draw":
            errors.append(
                'rules.simultaneous.both "draw" is the default (an ending both players reach '
                "in one round is a draw); leave it out"
            )
        elif both != SIMULTANEOUS_BOTH_SCORE:
            errors.append(
                f'rules.simultaneous.both must be "{SIMULTANEOUS_BOTH_SCORE}", got {both!r}'
            )
        elif not isinstance(rules.get("scoring"), dict):
            errors.append(
                'rules.simultaneous.both "score" settles an ending both players reach in one '
                "round by rules.scoring, and there is none; add rules.scoring or leave it out"
            )


def _check_simultaneous_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Schema 26 (Q918a): the rule's own shape, ``beats``, and every combination §4 of
    `docs/reference/rounds.md` refuses, each in words naming both rules."""
    declared = [p["id"] for p in pieces if isinstance(p, dict) and isinstance(p.get("id"), str)]
    if not is_simultaneous(rules):
        if "simultaneous" in rules:
            errors.append(
                'rules.simultaneous must be an object, such as {} or {"tie": "initiative"}'
            )
        for index, piece in enumerate(pieces):
            if isinstance(piece, dict) and "beats" in piece:
                errors.append(
                    f"pieces[{index}].beats says who wins a collision, and only matters in a "
                    "simultaneous game (rules.simultaneous); leave it out"
                )
        return
    rule = rules["simultaneous"]
    _check_keys(rule, _SIMULTANEOUS_KEYS, "rules.simultaneous", errors)
    tie = rule.get("tie")
    if "tie" in rule:
        if tie == "both":
            errors.append(
                'rules.simultaneous.tie "both" is the default (both pieces are eliminated); '
                "leave it out"
            )
        elif tie not in SIMULTANEOUS_TIES:
            errors.append(
                f"rules.simultaneous.tie must be one of {SIMULTANEOUS_TIES}, got {tie!r}"
            )
    _check_chase_switches(spec, pieces, rules, errors)
    portal_ids = portal_piece_ids(rules)
    for index, piece in enumerate(pieces):
        if not isinstance(piece, dict) or not isinstance(piece.get("beats"), list):
            continue
        if piece.get("id") in portal_ids:
            errors.append(
                f"pieces[{index}].beats is on a portal, and a portal never collides; leave it out"
            )
        named = frozenset(entry for entry in piece["beats"] if isinstance(entry, str))
        captures = piece.get("captures")
        default = (
            frozenset(entry for entry in captures if isinstance(entry, str))
            if isinstance(captures, list) else frozenset(declared)
        )
        if named == default and len(named) == len(piece["beats"]):
            errors.append(
                f"pieces[{index}].beats is what the piece beats when beats is left out (its "
                "captures, or every piece); leave it out"
            )
        elif (
            has_speeds(pieces) and isinstance(piece.get("id"), str)
            and not has_equal_speed_opponent({**spec, "rules": rules, "pieces": pieces}, piece["id"])
        ):
            errors.append(
                f"pieces[{index}].beats is read only when two pieces of one speed collide, and "
                "no piece of the other player has this one's speed (pieces[].speed); leave it out"
            )

    def refuse(what: str, why: str) -> None:
        errors.append(f"rules.simultaneous beside {what}: {why}")

    if rules.get("win", "checkmate") == "checkmate":
        refuse(
            "rules.win 'checkmate'" + ("" if "win" in rules else " (the default)"),
            "there is no check when both players move at once; use 'royal_capture', "
            "'annihilation' or 'none'",
        )
    if "checkmate" in rules:
        refuse("rules.checkmate", "there is no checkmate when both players move at once")
    if rules.get("forbid_check") is True:
        refuse("rules.forbid_check", "there is no check when both players move at once")
    if rules.get("check_limit") is not None:
        refuse("rules.check_limit", "there is no check when both players move at once")
    if rules.get("royal_facing") == "forbidden":
        refuse(
            "rules.royal_facing 'forbidden'",
            "two legal moves made together can open the file, which neither did alone",
        )
    if isinstance(rules.get("castling"), dict):
        refuse("rules.castling", "castling moves two pieces in one move, and a round has one each")
    if rules.get("en_passant") is True:
        refuse(
            "rules.en_passant",
            "it reads the last move, and the pawn that stepped twice may move again this round",
        )
    if turn_is_long(rules):
        refuse("a turn longer than one move", "a round is one secret move each")
    if any(
        isinstance(piece, dict) and any(
            isinstance(move, dict) and move.get("kind") == "sweep"
            for move in piece.get("moves") or []
        )
        for piece in pieces
    ):
        refuse(
            "a sweep",
            "which pieces a sweep removes depends on who stood on its path when, and a "
            "round has no such time",
        )
    if any(end["moved_by"] for group in portal_groups(rules) for end in group["ends"]):
        refuse(
            "a portal a player moves (moved_by)",
            "a piece going through a portal moved in the same round needs a third timing rule",
        )
    for key in ("strike", "contacts"):
        stated = rules.get(key)
        if not isinstance(stated, dict):
            continue
        position = stated.get("position", "advance")
        if position == "choose":
            refuse(
                f"rules.{key}.position 'choose'",
                "the square is chosen before the other player's move is known",
            )
        if key == "strike" and position == "stay":
            refuse(
                "rules.strike.position 'stay'",
                "a captor that collided did move, so staying would mean going back",
            )
        if key == "contacts" and position == "swap" and any(
            isinstance(link, dict) and link.get("own") is True
            for link in stated.get("links") or []
        ):
            refuse(
                "an own rules.contacts link with position 'swap'",
                "the mover's own piece would be put on its origin before the meetings, "
                "where the other player's piece may arrive",
            )
    drops = rules.get("drops")
    if isinstance(drops, dict):
        if drops.get("no_mate"):
            refuse("rules.drops.no_mate", "it is about checkmate, and there is none")
        stock = drops.get("stock")
        royal_ids = {
            p.get("id") for p in pieces if isinstance(p, dict) and p.get("royal") is True
        }
        if drops.get("placement") is True and isinstance(stock, dict) and royal_ids & set(stock):
            refuse(
                "a royal in a placement stock",
                "both players place at once, and a royal could be traded before the game starts",
            )
    if tie in SIMULTANEOUS_TIES and has_speeds(pieces) and not _any_equal_speed_meeting(
        {**spec, "rules": rules, "pieces": pieces}
    ):
        # Schema 28 (Q943a, `docs/plan-speed.md` §3.3): pieces of different speeds never
        # collide, so where no two opposing types share a speed no collision can happen.
        errors.append(
            f"rules.simultaneous.tie {tie!r} can never be read: pieces of different speeds "
            "never collide (pieces[].speed), and no piece of one player has the speed of a "
            "piece of the other; leave it out"
        )
    elif tie in SIMULTANEOUS_TIES:
        pair = simultaneous_tie_pair({**spec, "rules": rules, "pieces": pieces})
        if tie == "impossible" and pair is not None:
            errors.append(
                f'rules.simultaneous.tie "impossible", but a {pair[0]} of the first player and '
                f"a {pair[1]} of the second can collide and tie (each beats the other, or "
                "neither does)"
            )
        if tie == "initiative" and pair is None:
            errors.append(
                'rules.simultaneous.tie "initiative" can never decide anything, since no '
                'collision in this game can tie; write "impossible", or leave it out'
            )


def facing_turns(piece: Any) -> bool:
    """Whether a move may turn this piece: it has a facing and does not say ``turns: false``.

    The one reader of the flag, shared by the validator, the compiler and every surface
    that asks whether a facing is a choice a player makes or a fact about a piece."""
    facing = piece.get("facing") if isinstance(piece, dict) else None
    return isinstance(facing, dict) and facing.get("turns", True) is not False


def portal_has_folds(rules: Any) -> bool:
    """Whether any portal group is a fold (schema 22)."""
    return any(group["kind"] == "fold" for group in portal_groups(rules))


def _hex_symmetries_used(spec: dict[str, Any]) -> set[str]:
    """Schema 18b's names for the hexagon's turns, wherever a descriptor writes one."""
    written: list[Any] = []
    pieces = spec.get("pieces")
    for piece in pieces if isinstance(pieces, list) else ():
        moves = piece.get("moves") if isinstance(piece, dict) else None
        for move in moves if isinstance(moves, list) else ():
            if isinstance(move, dict):
                written.append(move.get("symmetry"))
    rules = spec.get("rules")
    for key in ("strike", "contacts"):
        rule = rules.get(key) if isinstance(rules, dict) else None
        landing = rule.get("landing") if isinstance(rule, dict) else None
        if isinstance(landing, dict):
            written.append(landing.get("symmetry"))
    return {f"symmetry={name}" for name in ("rotate6", "dihedral12") if name in written}


def is_keeping_link(link: Any) -> bool:
    """Whether a contact link says the mover becomes itself (schema 18's keeping link)."""
    return (
        isinstance(link, dict)
        and "mover_to" in link
        and link.get("mover_to") == link.get("mover")
    )


def keeping_link_holds(link: Any, promoting: AbstractSet[str]) -> bool:
    """Whether a keeping link states a rule: the mover has a promotion it overrides.

    Q781's predicate (`docs/archive/plan-schema-17-18.md` §8). The core writes the promoted
    piece, then the contact's ``mover_to`` over it (Q539: the contact wins), so ``mover_to``
    equal to the mover undoes the promotion on that capture, and is the plain capture
    written the long way wherever no promotion could have fired. The mover's own promotion
    entries are the test: a link is about one meeting, and whether a promotion fires on it
    is a question of where the capture lands, which a link does not say. An ``own`` link
    (schema 11) is a merge and not a capture, and is not lifted.
    """
    return (
        is_keeping_link(link)
        and link.get("own") is not True
        and link.get("mover") in promoting
    )


def opening_counts(spec: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    """How many of each piece id each player has at ply zero, the mirrored army included.

    What schema 18's two counting rules read their setup against: ``promotion.only_lost``
    offers a target while the mover has fewer than this, and ``extinction.per_type``
    refuses a side that would have run out before its first move. Read off the setup
    exactly as ``compile`` places it (``_placed_squares``), so the validator and the
    compiled ``initial_board`` cannot disagree about who starts with what.
    """
    counts: tuple[dict[str, int], dict[str, int]] = ({}, {})
    board = spec.get("board")
    setup = spec.get("setup")
    pieces = spec.get("pieces")
    if not (isinstance(board, dict) and isinstance(setup, dict) and isinstance(pieces, list)):
        return counts
    width, height = board.get("width"), board.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return counts
    orientation = setup.get("orientation", "mirror")
    for piece in pieces:
        piece_id = piece.get("id") if isinstance(piece, dict) else None
        if not isinstance(piece_id, str):
            continue
        for player, _file, _rank in _placed_squares(setup, piece_id, width, height, orientation):
            counts[player][piece_id] = counts[player].get(piece_id, 0) + 1
    return counts


def _check_only_lost_coherence(
    spec: dict[str, Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """``promotion.only_lost`` (schema 18, Q783) against what *lost* has to mean.

    The core counts the mover's pieces of a target type on the board and offers the
    target while that is below the side's opening count. That is *lost* only where a
    piece off the board stays off it, so the rules under which a taken piece comes back
    are refused by name rather than ordered (`docs/archive/plan-schema-17-18.md` §9.3,
    *18c*):

    * **beside ``drops`` that fill from captures** -- a taken piece goes to a hand and may
      be dropped again, so a piece can be lost and not gone. A stone-only hand
      (``from_captures: false``, `archive/stones`) gives nothing back and is accepted;
    * **beside a circe strike** (``strike.position: "home"``) -- a taken piece is reborn
      on its home square, so it is lost only when that square was full;
    * **beside ``promote_on_entry``** -- a piece entering its zone takes the first target
      listed, with no move in which to offer it or withhold it.

    And a target the side starts with none of could never be offered at all: nothing can
    be lost of it. That is a target written for nothing, refused per player as the
    inert-rule refusals are.
    """
    flagged = [
        entry
        for entry in _promotion_entries(rules)
        if isinstance(entry, dict) and entry.get("only_lost") is True
    ]
    if not flagged:
        return
    drops = rules.get("drops")
    if isinstance(drops, dict) and drops and drops.get("from_captures", True) is not False:
        errors.append(
            "rules.promotion.only_lost beside rules.drops: a taken piece goes to a hand and "
            "can come back, so what a side has lost has two answers; drop one of the two"
        )
    strike = rules.get("strike")
    if isinstance(strike, dict) and strike.get("position") == "home":
        errors.append(
            'rules.promotion.only_lost beside a strike whose position is "home": a taken '
            "piece is reborn on its home square, so what a side has lost has two answers; "
            "drop one of the two"
        )
    if rules.get("promote_on_entry") is True:
        errors.append(
            "rules.promotion.only_lost beside rules.promote_on_entry: a piece that enters "
            "its zone takes the first target listed, with no move in which to withhold it; "
            "drop one of the two"
        )
    # Who starts with what is a fact only of a board and an army that can be read. When
    # either was refused, a count of none would repeat that refusal, not state a fact.
    board = spec.get("board")
    setup = spec.get("setup")
    if not (
        isinstance(board, dict)
        and all(
            isinstance(board.get(side), int)
            and not isinstance(board.get(side), bool)
            and 2 <= board[side] <= MAX_BOARD_DIM
            for side in ("width", "height")
        )
        and isinstance(setup, dict)
        and isinstance(setup.get("placements"), list)
    ):
        return
    counts = opening_counts(spec)
    # Schema 18 (Q784): under the placement phase a stocked piece is on the board before
    # any piece of its side moves (a hand that fills from captures is refused above), so
    # the side starts with it as surely as with one placed.
    if isinstance(drops, dict) and drops.get("placement") is True:
        for player in (0, 1):
            stock = drops.get("stock")
            for piece, held in (stock if isinstance(stock, dict) else {}).items():
                if isinstance(held, int) and not isinstance(held, bool):
                    counts[player][piece] = counts[player].get(piece, 0) + held
    for entry in flagged:
        if not isinstance(entry.get("piece"), str) or (
            "to" not in entry and "to_by_player" not in entry
        ):
            continue
        try:
            per_player = promotion_targets(entry)
        except (KeyError, TypeError):
            continue
        for player, targets in enumerate(per_player):
            for target in targets if isinstance(targets, list) else []:
                if isinstance(target, str) and counts[player].get(target, 0) == 0:
                    errors.append(
                        f"rules.promotion.only_lost offers {target!r} to player {player} "
                        f"only once one is lost, and player {player} starts with none; "
                        "take it out of the targets"
                    )


def royal_starts_in_hand(rules: Any, royal_ids: AbstractSet[str]) -> bool:
    """Whether a royal starts in the stock under ``drops.placement`` (schema 18, Q784).

    The one spelling that lifts *"no royal piece appears in setup.placements"*: a royal
    played from the hand before anything on the board moves is a royal a side has from
    the first ply, as surely as one placed. ``_check_placement_coherence`` refuses every
    other way of stocking one.
    """
    if not isinstance(rules, dict):
        return False
    drops = rules.get("drops")
    if not isinstance(drops, dict) or drops.get("placement") is not True:
        return False
    stock = drops.get("stock")
    return isinstance(stock, dict) and any(
        stock.get(royal) == 1 for royal in royal_ids
    )


def _check_placement_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """A royal in a stock (schema 10's ``drops.stock``, read again by schema 18, Q784).

    A stock could name a royal since schema 10, and nothing read it: a royal placed and
    stocked compiled as one royal (`compile._royalty_can_multiply` counted the board
    alone), so the core guarded one king of two, and a royal stocked and never placed was
    refused only because no royal was placed. Schema 18's placement phase is the one
    reading under which a royal in hand is a game, so the rest are refused by name at
    every version:

    * **outside ``drops.placement``** -- a side could move its army with its royal still in
      hand, neither in play nor lost;
    * **more than one, or ``"unlimited"``** -- a royal is stocked once, as it is placed once
      where it is not placed at all;
    * **beside a placed royal of the same type** -- a side would hold two.
    """
    drops = rules.get("drops")
    if not isinstance(drops, dict):
        return
    stock = drops.get("stock")
    if not isinstance(stock, dict):
        return
    royal_ids = {
        piece["id"]
        for piece in pieces
        if isinstance(piece, dict) and piece.get("royal") and isinstance(piece.get("id"), str)
    }
    stocked = sorted(royal for royal in royal_ids if royal in stock)
    if not stocked:
        return
    placement = drops.get("placement") is True
    counts = opening_counts(spec)
    for royal in stocked:
        held = stock[royal]
        if not placement:
            errors.append(
                f"rules.drops.stock holds the royal {royal!r} outside the placement phase: "
                "a side could move its army with its royal still in hand; add "
                '"placement": true to rules.drops or take the royal out of the stock'
            )
        if held != 1:
            errors.append(
                f"rules.drops.stock holds {held!r} of the royal {royal!r}; a royal is "
                "stocked once"
            )
        if counts[0].get(royal, 0) or counts[1].get(royal, 0):
            errors.append(
                f"the royal {royal!r} is both placed by setup.placements and stocked by "
                "rules.drops.stock, so a side would hold two; place it or stock it"
            )


def drop_face_pairs(pieces: Any) -> dict[str, list[str]]:
    """Each base's other faces, ``promoted_from`` read the other way (schema 18, Q785).

    A piece naming a declared base as its ``promoted_from`` is one of that base's faces.
    Where ``drops.either_face`` is set, more than one face per base is refused
    (``_check_either_face_coherence``), so there every list has one entry.
    """
    if not isinstance(pieces, list):
        return {}
    declared = {
        piece["id"]
        for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    pairs: dict[str, list[str]] = {}
    for piece in pieces:
        if not isinstance(piece, dict) or not isinstance(piece.get("id"), str):
            continue
        base = piece.get("promoted_from")
        if isinstance(base, str) and base in declared and base != piece["id"]:
            pairs.setdefault(base, []).append(piece["id"])
    return pairs


def _check_either_face_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """``drops.either_face`` (schema 18, Q785) against what a taken piece becomes.

    A held piece may be dropped as itself or as its other face, the piece whose
    ``promoted_from`` it is. The rule says what a taken piece may come back as, so the
    rules that already say something else about a taken piece are refused by name rather
    than ordered (`docs/archive/plan-schema-17-18.md` §3 item 7, §8):

    * **a strike whose position is not ``advance``** -- rifle's attacker stays, circe's
      victim is reborn on its home square, and a chosen landing places a survivor: each is
      a second answer to where a taken piece goes;
    * **``strike.switches_sides``** -- Andernach's attacker changes sides on a capture;
    * **``explode_on_capture``** -- a blast takes pieces no capture took.

    A live strike is refused beside any hand already (``_check_strike_coherence``); these
    name the flag so the message says which rule to give up. And the pairs must name one
    slot per face, or the core could not say which slot a drop spends: a base with two
    faces, a piece that is both a face and a base (a chain of ``promoted_from``), a face
    in ``drops.stock`` and a royal in a pair are refused. A flag with no pair a hand can
    hold states nothing, and is refused as the inert rules are.
    """
    drops = rules.get("drops")
    if not isinstance(drops, dict) or drops.get("either_face") is not True:
        return
    strike = rules.get("strike")
    if isinstance(strike, dict):
        position = strike.get("position", "advance")
        if position != "advance":
            errors.append(
                f"rules.drops.either_face beside a strike whose position is {position!r}: "
                "the strike already says where a taken piece goes, so what it comes back "
                "as has two answers; drop one of the two"
            )
        if strike.get("switches_sides"):
            errors.append(
                "rules.drops.either_face beside rules.strike.switches_sides: a capture "
                "already changes a piece's side, so what a taken piece becomes has two "
                "answers; drop one of the two"
            )
    if rules.get("explode_on_capture"):
        errors.append(
            "rules.drops.either_face beside rules.explode_on_capture: a blast takes pieces "
            "no capture took, so what a taken piece becomes has two answers; drop one of "
            "the two"
        )
    pairs = drop_face_pairs(pieces)
    if not pairs:
        errors.append(
            "rules.drops.either_face pairs nothing: no piece names another as its "
            "promoted_from, so every piece has one face; drop the flag"
        )
        return
    faces = {face for listed in pairs.values() for face in listed}
    royal_ids = {
        piece["id"]
        for piece in pieces
        if isinstance(piece, dict) and piece.get("royal") and isinstance(piece.get("id"), str)
    }
    stock = drops.get("stock")
    stocked = set(stock) if isinstance(stock, dict) else set()
    for base, listed in sorted(pairs.items()):
        if len(listed) > 1:
            errors.append(
                f"rules.drops.either_face: {base!r} is the promoted_from of "
                f"{', '.join(repr(face) for face in sorted(listed))}, and a piece in hand "
                "has one other face; keep one"
            )
        if base in faces:
            errors.append(
                f"rules.drops.either_face: {base!r} is both a face (it names a "
                "promoted_from) and a base (another piece names it), so a hand could hold "
                "a face and a drop could not say which slot it spends"
            )
        for piece in (base, *listed):
            if piece in royal_ids:
                errors.append(
                    f"rules.drops.either_face pairs the royal piece {piece!r}; a royal "
                    "has one face, so leave it out of promoted_from"
                )
    for face in sorted(faces & stocked):
        errors.append(
            f"rules.drops.stock holds {face!r}, the other face of a piece in hand under "
            "rules.drops.either_face; stock its base, which may be dropped as either"
        )
    if all(_never_in_a_hand(spec, rules, drops, base) is not None for base in pairs):
        errors.append(
            "rules.drops.either_face can never fire: no piece with another face can reach "
            "a hand; drop the flag"
        )


def _check_extinction_coherence(spec: dict[str, Any], rules: dict[str, Any], errors: list[str]) -> None:
    """``rules.extinction``: no side may start already run out.

    Counted on its own (``per_type``, schema 18, Q783), a listed type a side starts with
    fewer than ``min_count`` of ends the game before its first move. Summed (Q937), the
    same is true of a side whose listed types (``any``: its whole army) add up to fewer
    than ``min_count``: the core counts the board at ply zero as at every other ply
    (``terminal_prefix``), so the game is over before anyone moves. Both read the board
    as ``compile`` places it; a stock is not counted, as the core does not count it.

    Counted against a side that starts with it (``"own"``, schema 27, Q936), the same
    holds for each kind a side starts with, and four spellings state nothing new
    (`docs/reference/extinction-any-kind.md` §2): a listed kind neither side starts with
    names nothing; every kind started by both is ``true``'s game under a second hash; no
    side with two counted kinds is the summed count's; and beside ``drops`` a kind can
    start in hand, with an opening count of zero on the board.
    """
    extinction = rules.get("extinction")
    if not isinstance(extinction, dict):
        return
    pieces = extinction.get("pieces")
    count = extinction.get("min_count", 1)
    if isinstance(count, bool) or not isinstance(count, int):
        return
    if pieces != "any" and not isinstance(pieces, list):
        return
    counts = opening_counts(spec)
    if extinction.get("per_type") == EXTINCTION_PER_TYPE_OWN:
        if isinstance(pieces, list):
            _check_own_extinction(rules, pieces, count, counts, errors)
        return
    if extinction.get("per_type") is True:
        if not isinstance(pieces, list):
            return
        for player in (0, 1):
            for piece in sorted({p for p in pieces if isinstance(p, str)}):
                if counts[player].get(piece, 0) < count:
                    errors.append(
                        f"rules.extinction.per_type: player {player} starts with "
                        f"{counts[player].get(piece, 0)} of {piece!r}, fewer than min_count "
                        f"{count}, so the game would be over before the first move"
                    )
        return
    if extinction.get("per_type", False) is not False:
        return  # a malformed per_type is `_validate_extinction`'s to name
    for player in (0, 1):
        if pieces == "any":
            have = sum(counts[player].values())
            named = "pieces"
        else:
            kinds = sorted({p for p in pieces if isinstance(p, str)})
            have = sum(counts[player].get(piece, 0) for piece in kinds)
            named = "of " + ", ".join(repr(piece) for piece in kinds)
        if have < count:
            errors.append(
                f"rules.extinction: player {player} starts with {have} {named}, fewer than "
                f"min_count {count}, so the game would be over before the first move"
            )


def _check_own_extinction(
    rules: dict[str, Any],
    pieces: list[Any],
    count: int,
    counts: tuple[dict[str, int], dict[str, int]],
    errors: list[str],
) -> None:
    """``"per_type": "own"`` (schema 27, Q936) against the opening board."""
    kinds = sorted({p for p in pieces if isinstance(p, str)})
    if len(kinds) < 2:
        return  # `_validate_extinction` names a single listed type
    if isinstance(rules.get("drops"), dict):
        errors.append(
            'rules.extinction.per_type "own" beside rules.drops: a kind that starts in hand '
            "has no opening count on the board, and whether a held kind counts is not "
            "settled; drop one of the two"
        )
        return
    for piece in kinds:
        if not any(counts[player].get(piece, 0) for player in (0, 1)):
            errors.append(
                f'rules.extinction.per_type "own": neither player starts with {piece!r}, '
                "so it is counted against nobody; drop it from pieces"
            )
    for player in (0, 1):
        for piece in kinds:
            have = counts[player].get(piece, 0)
            if 0 < have < count:
                errors.append(
                    f'rules.extinction.per_type "own": player {player} starts with {have} '
                    f"of {piece!r}, fewer than min_count {count}, so the game would be over "
                    "before the first move"
                )
    if all(counts[player].get(piece, 0) for piece in kinds for player in (0, 1)):
        errors.append(
            'rules.extinction.per_type "own": both players start with every listed kind, '
            "so each is counted against both; say per_type true"
        )
        return
    counted = [sum(1 for piece in kinds if counts[player].get(piece, 0)) for player in (0, 1)]
    if max(counted) < 2:
        errors.append(
            'rules.extinction.per_type "own": no player starts with two of the listed '
            "kinds, so counting each on its own is the count together; drop per_type"
        )


def strike_is_live(strike: Any) -> bool:
    """Whether a ``rules.strike`` spelling states a rule rather than plain capture.

    Read off the spelling, for the coherence checks that run before (and without) a
    compile: an effect other than ``"capture"``, or since schema 17 a ``"capture"``
    whose position is ``stay`` (rifle) or ``home`` (circe), or a ``switches_sides`` list
    (Andernach). ``strike_changes_anything`` is the compiler's sharper answer, which also
    drops a chain whose every link is immune.
    """
    if not isinstance(strike, dict):
        return False
    return (
        strike.get("effect", "capture") != "capture"
        or strike.get("position", "advance") in ("stay", "home")
        or bool(strike.get("switches_sides"))
    )


def uses_sweep(spec: dict[str, Any]) -> bool:
    """Whether any piece in this spec has a ``sweep`` descriptor.

    Public because the mutation operators ask it before adding a rule a sweep refuses.
    """
    pieces = spec.get("pieces")
    return isinstance(pieces, list) and any(
        isinstance(move, dict) and move.get("kind") == "sweep"
        for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("moves"), list)
        for move in piece["moves"]
    )


def promoting_piece_ids(rules: Any) -> set[str]:
    """The ids ``rules.promotion`` names as promoting pieces, in either spelling."""
    if not isinstance(rules, dict):
        return set()
    promotion = rules.get("promotion")
    entries = promotion if isinstance(promotion, list) else [promotion]
    return {
        entry["piece"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("piece"), str)
    }


# The rules a sweep may not share a game with, and why (Q21).
_SWEEP_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("strike", "a strike says what one taken piece becomes, a sweep takes several, and "
     "ordering the two would choose a game silently"),
    ("contacts", "a contact says what one taken piece becomes, a sweep takes several, and "
     "ordering the two would choose a game silently"),
    ("explode_on_capture", "two multi-square clears on one move would be undone in an "
     "order nobody stated"),
    ("drops", "a sweep over several pieces would put several in a hand at once"),
    ("npcs", "a bounty is paid per victim, and a swept wild piece would be a second "
     "victim with nowhere to be paid"),
    ("wrap", "round a seam two rays can reach one square by two paths, and a move names "
     "only its target"),
)


def sweep_conflicts(rules: Any) -> list[str]:
    """The rule keys in ``rules`` that are refused beside a sweep, in declaration order.

    Read as the rule means it, not by the key's presence: an inert strike, an empty contact
    table and a wrap with every edge ``"none"`` compile to nothing and refuse nothing.
    """
    if not isinstance(rules, dict):
        return []
    active = {
        "strike": strike_is_live(rules.get("strike")),
        "contacts": isinstance(rules.get("contacts"), dict)
        and bool(rules["contacts"].get("links")),
        "explode_on_capture": bool(rules.get("explode_on_capture")),
        "drops": bool(rules.get("drops")),
        "npcs": bool(rules.get("npcs")),
        "wrap": _wrap_joins_an_edge(rules.get("wrap")),
    }
    return [key for key, _ in _SWEEP_EXCLUDES if active[key]]


def _wrap_joins_an_edge(wrap: Any) -> bool:
    # Both shapes' keys: a valid spec states only its own shape's, so the union is exact,
    # and a caller holding `rules` alone need not be handed the board.
    return any(
        value not in (None, "none")
        for shape in SHAPES
        for values in wrap_stated(wrap, shape).values()
        for value in values
    )


def _check_sweep_coherence(
    pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """The six refusals a sweep carries (Q21).

    Five rules say what a taken piece becomes or how a move is named, and a sweep takes
    several pieces along a path a move word does not carry. The sixth keeps a sweeper from
    promoting: a move word's kind holds a promotion or a sweep, not both. That refusal is
    also why no sweeper is ever the en passant piece.
    """
    sweepers = {
        piece["id"]
        for piece in pieces
        if isinstance(piece, dict)
        and isinstance(piece.get("id"), str)
        and isinstance(piece.get("moves"), list)
        and any(isinstance(m, dict) and m.get("kind") == "sweep" for m in piece["moves"])
    }
    if not sweepers:
        return
    why = dict(_SWEEP_EXCLUDES)
    for key in sweep_conflicts(rules):
        errors.append(f"rules.{key} is refused beside a piece that sweeps: {why[key]}")
    for pid in sorted(str(p) for p in promoting_piece_ids(rules) & sweepers):
        errors.append(
            f"rules.promotion makes {pid!r} a promoting piece, and it sweeps; a move is a "
            "promotion or a sweep, never both, so a sweeper may not promote (promoting "
            "into one is fine)"
        )


def _validate_version_two_stays_within_version_two(
    spec: dict[str, Any], errors: list[str]
) -> None:
    """Reject a spec that declares schema 1 or 2 and then uses schema 3.

    Same reasoning as the version-1 check below, and the same consequence if it were
    skipped: the declared version is part of the mechanics hash, so two specs whose
    rules differ only in a feature one of them was not entitled to use must not be able
    to collide.
    """
    pieces = spec.get("pieces")
    if not isinstance(pieces, list):
        return
    rules = spec.get("rules")
    used: set[str] = set()
    if isinstance(rules, dict):
        used |= V3_RULE_KEYS.intersection(rules)
    for piece in pieces:
        if not isinstance(piece, dict):
            continue
        used |= V3_PIECE_KEYS.intersection(piece)
        moves = piece.get("moves")
        if not isinstance(moves, list):
            continue
        for move in moves:
            if isinstance(move, dict):
                used |= V3_MOVE_KEYS.intersection(move)
    if used:
        errors.append(
            f"this spec uses {', '.join(sorted(used))}, added in schema 3; "
            'declare "schema_version": 3 to use it'
        )


def _validate_version_one_stays_within_version_one(
    spec: dict[str, Any], errors: list[str]
) -> None:
    """Reject a spec that declares schema 1 and then uses schema 2.

    Accepting it silently would make the declared version meaningless, and the version
    is part of the mechanics hash: two specs whose rules differ only in a feature one
    of them was not entitled to use must not be able to collide.
    """
    def report(where: str, used: AbstractSet[str]) -> None:
        if used:
            errors.append(
                f"{where} uses {', '.join(sorted(used))}, added in schema 2; "
                "declare \"schema_version\": 2 to use it"
            )

    rules = spec.get("rules")
    if isinstance(rules, dict):
        report("rules", V2_RULE_KEYS.intersection(rules))
        promotion = rules.get("promotion")
        if isinstance(promotion, dict):
            report("rules.promotion", V2_PROMOTION_KEYS.intersection(promotion))
        castling = rules.get("castling")
        if isinstance(castling, dict):
            report("rules.castling", V2_CASTLING_KEYS.intersection(castling))

    setup = spec.get("setup")
    if isinstance(setup, dict):
        report("setup", V2_SETUP_KEYS.intersection(setup))

    if isinstance(rules, dict) and rules.get("win") == "none":
        errors.append(
            "rules.win 'none' was added in schema 2; declare \"schema_version\": 2 to use it"
        )

    pieces = spec.get("pieces")
    if isinstance(pieces, list):
        used: set[str] = set()
        for piece in pieces:
            if not isinstance(piece, dict) or not isinstance(piece.get("moves"), list):
                continue
            if "region" in piece:
                used.add("region")
            for move in piece["moves"]:
                if not isinstance(move, dict):
                    continue
                used |= V2_MOVE_KEYS.intersection(move)
                if move.get("kind") == "lame":
                    used.add("the 'lame' move kind")
                condition = move.get("condition")
                if isinstance(condition, dict):
                    used |= V2_CONDITION_KEYS.intersection(condition)
        report("a piece", used)

    # Multiple royals is the one schema-2 feature with no key of its own: it is the
    # removal of a restriction rather than the addition of a field. Under schema 1 the
    # check filter guarded a single royal square, so every route to a second one stays
    # closed there. What those routes are is `royalty_can_multiply`, stated once below.
    if (
        isinstance(rules, dict)
        and rules.get("win", "checkmate") == "checkmate"
        and royalty_can_multiply(spec)
    ):
        errors.append(
            "more than one royal piece under 'checkmate' requires \"schema_version\": 2"
        )


def royalty_can_multiply(spec: dict[str, Any]) -> bool:
    """Whether either player might ever hold two royal pieces at the same time.

    The validator's reading of the question ``engine.compile._royalty_can_multiply``
    answers for the core. The two must find the same routes: the schema-2 gate above
    refuses under schema 1 exactly the games the core would have to be told about, and a
    route only one of them knows is a spec accepted as single-royal and then compiled as
    something else, or the reverse.

    **Why this reads the spec again rather than delegating.** Delegation was the first
    answer tried and it is not available. The compiler's read is defined over the
    *compiled* rules -- a board of bytes, promotion targets as type indices,
    ``strike_to``, ``demotions`` -- none of which exist until a spec has been compiled,
    and ``compile_spec`` opens with ``validate_or_raise``: a validator that reached for
    it would call the function that calls the validator, and the recursion is only the
    loudest of the three problems. ``engine/compile.py`` imports this module (so the
    import is a cycle), and the compiler's contract is that a validated spec needs no
    defensive checks of its own (so it may not be run on input nothing has checked yet).
    Validation precedes compilation by construction and cannot borrow from it.

    So the routes are read twice, in the two vocabularies, and
    ``tests/test_multi_royal.py`` holds the two answers together over the committed
    corpus and over one spec per route -- the arrangement ``fairy/verify.py`` uses for
    the only other place two readings of the DSL have to coexist.

    The routes, in the compiler's order:

    * two royal piece types, or two royals placed for one player at ply zero;
    * a promotion whose targets include a royal -- either spelling of ``rules.promotion``
      (one rule or a list of them) and either of ``to`` / ``to_by_player``;
    * a strike chain link that turns a victim the chain can reach into a royal;
    * a drop from a hand a royal can reach: a piece ``promoted_from`` a royal comes back
      royal, and a royal comes back as itself wherever taking one leaves the game
      running.

    Defensive throughout, because ``validate_spec`` collects errors rather than stopping:
    this runs over specs already known to be malformed.
    """
    pieces = spec.get("pieces")
    if not isinstance(pieces, list):
        return False
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        rules = {}
    royal_ids = {
        p["id"]
        for p in pieces
        if isinstance(p, dict) and p.get("royal") and isinstance(p.get("id"), str)
    }
    if not royal_ids:
        return False
    if len(royal_ids) > 1:
        return True

    setup = spec.get("setup")
    placements = setup.get("placements") if isinstance(setup, dict) else None
    if isinstance(placements, list):
        # `setup.mirror` needs no term of its own: a placement for the second player is
        # already refused while it is on (`_validate_setup`), and the reflection hands
        # that player the same count rather than a further one.
        per_player: dict[Any, int] = {}
        for placement in placements:
            if not isinstance(placement, dict) or not _in(placement.get("piece"), royal_ids):
                continue
            squares = placement.get("squares")
            if isinstance(squares, list):
                player = placement.get("player", 0)
                if not isinstance(player, (int, str)):
                    player = repr(player)
                per_player[player] = per_player.get(player, 0) + len(squares)
        if any(count > 1 for count in per_player.values()):
            return True

    for entry in _promotion_entries(rules):
        if isinstance(entry, dict) and royal_ids.intersection(_promotion_target_ids(entry)):
            return True

    strike = rules.get("strike")
    if isinstance(strike, dict) and strike.get("effect", "capture") != "capture":
        immune = strike.get("immune")
        immune_ids = set(immune) if isinstance(immune, list) else set()
        chain = strike.get("chain")
        # An immune source is a plain capture, so its link never fires -- which is how
        # the compiler drops it before `strike_to` is built.
        for link in chain if isinstance(chain, list) else []:
            if (
                isinstance(link, dict)
                and link.get("from") not in immune_ids
                and link.get("to") in royal_ids
            ):
                return True

    # The same route through the other spelling (schema 10, Q19a). Two columns rather
    # than one: a contact can make a royal out of the *attacker* as well as out of the
    # piece it meets, and `_royal_types` reads both in the compiler's vocabulary.
    contacts = rules.get("contacts")
    if isinstance(contacts, dict):
        immune = contacts.get("immune")
        immune_ids = set(immune) if isinstance(immune, list) else set()
        links = contacts.get("links")
        for link in links if isinstance(links, list) else []:
            if not isinstance(link, dict):
                continue
            if link.get("mover_to") in royal_ids:
                return True
            if link.get("onto") not in immune_ids and link.get("onto_to") in royal_ids:
                return True

    drops = rules.get("drops")
    if isinstance(drops, dict) and drops.get("from_captures", True):
        # A captured piece enters the captor's hand as its `promoted_from` where it
        # declares one, so a common piece demoting to a royal comes back royal under any
        # ending; a royal comes back as itself wherever taking one leaves the game
        # running -- every `win` but `checkmate`, where a royal cannot be taken, and
        # `royal_capture`, where taking the only one ends the game.
        royal_taken_continues = rules.get("win", "checkmate") not in (
            "checkmate",
            "royal_capture",
        )
        declared = {p.get("id") for p in pieces if isinstance(p, dict) and "id" in p}
        for piece in pieces:
            if not isinstance(piece, dict):
                continue
            demoted = piece.get("promoted_from")
            if demoted not in declared:
                demoted = piece.get("id")
            if demoted in royal_ids and (
                piece.get("id") not in royal_ids or royal_taken_continues
            ):
                return True

    return False


def _validate_appearance(
    appearance: Any, piece_ids: list[str] | set[str], errors: list[str]
) -> None:
    """How the pieces should be drawn. Never how they move.

    ``icons`` maps a piece id to a name in the icon library, for the case the library
    cannot guess from the piece's own name -- which is most invented pieces. A miss is
    not an error at any point downstream: the renderer draws the piece's letter, as it
    does today for every piece with no icon.

    An entry may also be a **pair** of names, one per player, and schema 8 is what
    forced it. A single drawing has always served both players by being turned upside
    down for the second -- which stops working the moment a piece has a facing, because
    the board turns it to *point* and a first player's pawn facing backwards then looks
    exactly like a second player's pawn facing forwards. A piece that carries a
    direction cannot also carry its side by being upside down, so it has to carry it in
    the drawing.
    """
    if appearance is None:
        return
    if not isinstance(appearance, dict):
        errors.append("appearance must be an object")
        return
    _check_keys(appearance, _APPEARANCE_KEYS, "appearance", errors)

    icons = appearance.get("icons")
    if icons is None:
        return
    if not isinstance(icons, dict):
        errors.append("appearance.icons must be an object mapping piece id to icon name")
        return
    for piece_id, icon in icons.items():
        if piece_id not in piece_ids:
            errors.append(f"appearance.icons names unknown piece {piece_id!r}")
        entries = icon if isinstance(icon, list) else [icon]
        if isinstance(icon, list) and len(icon) != 2:
            errors.append(
                f"appearance.icons[{piece_id!r}] as a list must hold exactly two "
                "entries, one per player"
            )
        if not all(isinstance(entry, str) and entry.strip() for entry in entries):
            errors.append(
                f"appearance.icons[{piece_id!r}] must be a non-empty string, or a pair "
                "of them for a piece drawn differently for each player"
            )


def _is_day(value: Any) -> bool:
    """A calendar day written ``YYYY-MM-DD``, and a real one."""
    from datetime import date

    if not isinstance(value, str) or len(value) != 10:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validate_provenance(provenance: Any, errors: list[str]) -> None:
    """Where a ruleset came from: who made it, when, and what it was asked to explore.

    Free text by design. A published variant's history is prose and refuses to be a
    schema; what matters is that it travels with the spec rather than in someone's head,
    and that it cannot reach the engine.
    """
    if provenance is None:
        return
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
        return
    _check_keys(provenance, _PROVENANCE_KEYS, "provenance", errors)

    for key in ("origin", "author", "source", "history", "theme", "exhibit", "preview"):
        value = provenance.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"provenance.{key} must be a string")
    exhibit = provenance.get("exhibit")
    if isinstance(exhibit, str) and not exhibit.strip():
        errors.append("provenance.exhibit must say what the ruleset is kept to show")
    preview = provenance.get("preview")
    if isinstance(preview, str) and not preview.strip():
        errors.append("provenance.preview must say why the game is shown as a preview")

    added = provenance.get("added")
    if added is not None and not _is_day(added):
        errors.append("provenance.added must be a day written YYYY-MM-DD")

    year = provenance.get("year")
    if year is not None and not isinstance(year, (int, str)):
        errors.append("provenance.year must be a year or a string such as '15th century'")

    parents = provenance.get("parents")
    if parents is not None and (
        not isinstance(parents, list) or not all(isinstance(p, str) for p in parents)
    ):
        errors.append("provenance.parents must be a list of ruleset hashes")


def turn_is_long(rules: dict[str, Any] | None) -> bool:
    """Does ``rules.turn`` ever give a player more than one move in a row?

    The validator's own reading of "a long turn", shared by every rule that is refused
    beside one and by ``transform.settle_en_passant``, so the two cannot drift. ``ramp``
    counts: a turn of one move whose opening ramps to two is still a player moving twice.
    """
    turn = (rules or {}).get("turn")
    if not isinstance(turn, dict):
        return False
    moves = turn.get("moves")
    if isinstance(moves, int) and not isinstance(moves, bool) and moves > 1:
        return True
    ramp = turn.get("ramp")
    return isinstance(ramp, list) and any(
        isinstance(entry, int) and not isinstance(entry, bool) and entry > 1
        for entry in ramp
    )


def _validate_coherence(spec: dict[str, Any], errors: list[str]) -> None:
    """Cross-field checks: individually valid settings that cannot hold together.

    These are the failures an LLM proposal actually makes. Each one would otherwise
    compile into a game that is quietly unplayable — a checkmate rule with nothing to
    checkmate, or an en passant rule with no piece able to trigger it — and would burn
    a full tournament before anyone noticed.
    """
    pieces = spec.get("pieces")
    rules = spec.get("rules", {})
    if not isinstance(pieces, list) or not isinstance(rules, dict):
        return

    has_royal = any(isinstance(p, dict) and p.get("royal", False) for p in pieces)
    win = rules.get("win", "checkmate")
    if win in ("checkmate", "royal_capture") and not has_royal:
        errors.append(f"rules.win is {win!r} but no piece is marked royal")
    if win == "annihilation" and has_royal:
        errors.append("rules.win is 'annihilation' but a piece is marked royal")
    if win == "none" and not any(
        rules.get(key) for key in ("reach", "extinction", "check_limit", "bare_royal")
    ) and rules.get("stalemate", "draw") == "draw":
        errors.append(
            "rules.win is 'none' but the spec states no ending of its own; the game "
            "could never be won"
        )

    if rules.get("en_passant") and not rules.get("promotion"):  # noqa: SIM102
        errors.append(
            "rules.en_passant requires rules.promotion: en passant is defined against "
            "the promoting piece, so there must be one"
        )

    # The core writes the en-passant window per *move* and flips the side only when the
    # turn runs out, so inside a two-move turn a double step is still answerable by the
    # player who made it -- a pawn taking its own pawn in passing. Refused rather than
    # repaired in the core, which is the `npcs` precedent below: the core fix has to
    # answer a design question nobody has asked (does the window close at the move or at
    # the turn, and may the opponent answer a double step made on the first move?), and
    # it would stale every stored measurement by build fingerprint. `docs/bugs.md` holds
    # the reasoning.
    if rules.get("en_passant") and turn_is_long(rules):
        errors.append(
            "rules.en_passant beside a turn longer than one move: the window a double "
            "step opens is still open on the mover's own next move, so a player could "
            "take their own pawn in passing; use one move per turn, or drop en passant"
        )

    setup = spec.get("setup")
    if has_royal and isinstance(setup, dict) and isinstance(setup.get("placements"), list):
        royal_ids = {
            p["id"]
            for p in pieces
            if isinstance(p, dict) and p.get("royal") and isinstance(p.get("id"), str)
        }
        placed = {
            p.get("piece")
            for p in setup["placements"]
            if isinstance(p, dict) and isinstance(p.get("piece"), str)
        }
        missing = royal_ids - placed
        if missing == royal_ids and not royal_starts_in_hand(rules, royal_ids):
            errors.append("no royal piece appears in setup.placements")

    # A malformed id or promoted_from is refused by its shape check; only strings are
    # asked about here, so a list in either place is refused rather than crashing a set.
    piece_ids_all = {
        p["id"] for p in pieces if isinstance(p, dict) and isinstance(p.get("id"), str)
    }
    for piece in pieces:
        if not isinstance(piece, dict):
            continue
        base = piece.get("promoted_from")
        if isinstance(base, str) and base not in piece_ids_all:
            errors.append(
                f"piece {piece.get('id')!r} promoted_from {base!r}, which is not a declared piece"
            )

    _check_obstacle_coherence(spec, rules, errors)
    _check_hexagon_coherence(spec, pieces, rules, errors)
    _check_strike_coherence(pieces, rules, errors)
    _check_contact_coherence(pieces, rules, errors)
    _check_sweep_coherence(pieces, rules, errors)
    _check_npc_coherence(spec, pieces, rules, errors)
    _check_entry_promotion_coherence(spec, rules, errors)
    _check_only_lost_coherence(spec, rules, errors)
    _check_extinction_coherence(spec, rules, errors)
    _check_placement_coherence(spec, pieces, rules, errors)
    _check_either_face_coherence(spec, pieces, rules, errors)
    _check_portal_coherence(spec, pieces, rules, errors)
    _check_inert_rules(spec, pieces, rules, errors)
    # Three owners share the one byte once anything is wild, and the bound has to know.
    # Portals (schema 19) are nobody's too, and each pair adds two internal types.
    _check_facing_coherence(
        pieces, errors,
        owners=3 if rules.get("npcs") or rules.get("portals") else 2,
        extra_types=portal_extra_types(rules),
    )
    _check_fixed_facings(spec, pieces, errors)
    _check_simultaneous_coherence(spec, pieces, rules, errors)
    _check_spy_coherence(spec, pieces, rules, errors)
    _check_speed_coherence(spec, pieces, rules, errors)
    _check_rest_coherence(spec, pieces, rules, errors)
    _check_sight_coherence(spec, pieces, rules, errors)

    if isinstance(rules.get("castling"), dict):
        castling = rules["castling"]
        king_id = castling.get("king")
        if has_royal and king_id is not None:
            royal_ids = _str_set(p.get("id") for p in pieces if isinstance(p, dict) and p.get("royal"))
            if not _in(king_id, royal_ids):
                errors.append(
                    f"rules.castling.king {king_id!r} is not a royal piece; castling protects "
                    "royalty, so pointing it at a common piece is almost always a mistake"
                )


def _check_facing_coherence(
    pieces: list[Any], errors: list[str], owners: int = 2, extra_types: int = 0
) -> None:
    """The board byte has to hold every piece value this ruleset can produce.

    A piece value is ``owner * facings * types + facing * types + type + 1`` and the
    board is one byte per square, so the whole encoding has to stay under 256. At
    ``MAX_PIECE_TYPES`` and four facings for two players it is 193, which is why this
    never fired before schema 9 -- and it is checked here rather than asserted in a
    comment because it is one ``MAX_PIECE_TYPES`` bump away from wrapping silently, and a
    wrapped piece value is one piece playing as another.

    **Schema 9 is the bump that makes it fire.** A wild piece is a third owner, and three
    owners at four facings is 21 piece types, not 24; ``owners`` is 3 whenever the spec
    declares ``rules.npcs``.
    """
    facings = max(
        (
            axes
            for piece in pieces
            if isinstance(piece, dict) and isinstance(piece.get("facing"), dict)
            for axes in [piece["facing"].get("axes")]
            if isinstance(axes, int) and not isinstance(axes, bool)
        ),
        default=1,
    )
    types = len(pieces) + extra_types
    highest = owners * facings * types
    if highest >= 256:
        who = "two players" if owners == 2 else "two players and a wild third owner"
        errors.append(
            f"{types} piece types at {facings} facings for {who} need "
            f"{highest + 1} board values, and a square is one byte; reduce the piece "
            "count or the facings"
        )


def _check_fixed_facings(spec: dict[str, Any], pieces: list[Any], errors: list[str]) -> None:
    """A facing no move turns (schema 25) has to be held two ways, or it is no facing.

    Nothing but the setup puts a piece in a facing other than its ``initial`` one: every
    way of entering play later (a promotion, a drop, a strike that changes a piece's type
    or side) takes the initial facing, and only a rotation could leave it. So when no
    placement of a ``turns: false`` piece states another facing, every copy points the way
    it entered for the whole game, which is the same game as its moves written that way
    round with no facing -- two spellings of one game under two hashes, and a facing a
    player is told about that never matters. Refused by name, as a null rotation is.
    """
    setup = spec.get("setup")
    placements = setup.get("placements") if isinstance(setup, dict) else None
    for index, piece in enumerate(pieces):
        if not isinstance(piece, dict) or not isinstance(piece.get("facing"), dict):
            continue
        if facing_turns(piece):
            continue
        initial = piece["facing"].get("initial", "forward")
        held = {repr(initial)} | {
            repr(placement.get("facing", initial))
            for placement in (placements if isinstance(placements, list) else ())
            if isinstance(placement, dict) and placement.get("piece") == piece.get("id")
        }
        if len(held) < 2:
            errors.append(
                f"pieces[{index}].facing.turns is false and every {piece.get('id')!r} "
                f"starts facing {initial}, the way it enters play, so none ever faces "
                "another way: write its moves that way round and give it no facing, or "
                "place some of them facing another way"
            )


def _check_obstacle_coherence(
    spec: dict[str, Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Every rule that names a square a piece stands on, checked against the walls.

    All seven are the same mistake -- *a rule that names a square nobody can be on* --
    and each is worth saying by name, because none is a default the engine could pick
    for the designer. A wall under the opening array is a game that cannot be set up; a
    winning square or a drop zone behind one is a rule that can never fire; a wild
    piece's route through one is a walk onto a square nothing may occupy, which is the
    sixth and is schema 9's; a promotion zone stated as squares is the seventh, schema 14's.

    **Castling is the fifth, and it was missing until it shipped a defect.** It is not
    like the other four: it names no square in the spec at all, it names a *span*, and
    the span is worked out at play time from where the king stands. That is why it was
    not here -- and why the core got it wrong, since ``generate_castling`` decides a
    path is clear by asking whether a piece stands on each square, and a wall is not a
    piece standing anywhere. The king did not merely castle *through* a wall; it ended
    up standing on one. See ``_castling_ranks`` for why a rank is the right granularity.

    **Schema 16 asks the same of a hexagon's absent cells** (refusal 4), which are not
    walls but are just as unstandable, and adds an eighth set, ``obstacles.squares``: a
    wall on a corner that is not a cell is walls spelled twice. The messages say *off the
    hexagon*, because nothing is in the way there; there is nothing there. Castling is
    held to the one rank with no absent cell, the middle one, by the rule for walls.
    """
    board = spec.get("board")
    if not isinstance(board, dict):
        return
    width, height = board.get("width"), board.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return
    raw_setup = spec.get("setup")
    setup: dict[str, Any] = raw_setup if isinstance(raw_setup, dict) else {}
    orientation = setup.get("orientation", "mirror")
    walls = obstacle_squares(rules, width, height, orientation)
    if walls:
        _check_named_squares(rules, setup, width, height, orientation, walls, False, errors)
    absent = absent_squares(board)
    if absent:
        _check_named_squares(rules, setup, width, height, orientation, absent, True, errors)


def _check_hexagon_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Schema 16's refusals 7 to 9: what a hexagon's geometry cannot answer.

    * **7. A vector longer than R + 1 on a hexagon with a side joined**, in hex length
      ``max(|a|, |b|, |a + b|)``: every move descriptor's, a ``strike`` or ``contacts``
      landing's, and a blast of radius r where 2r > R + 1. A step of at most R + 1 from
      any cell lands on the board or in one of the six copies across its sides, which is
      what makes the compiler's landing exact; a longer one can land two copies out, and
      with only some sides joined the way round the seams to get there has no single
      answer. Reducing modulo the lattice would be exact on the full torus and wrong
      everywhere else, so it is refused rather than answered, as ``lame`` is on the hex
      lattice.
    * **8. ``drops.no_doubled`` on a hexagon whose ``n_s`` or ``ne_sw`` pair is joined.**
      The core buckets squares into files by ``square % width``, and a file is a line
      only while ``(0, 1)`` comes back onto it; across ``n`` or ``ne`` it comes back onto
      another file (``c11`` to ``i1``). With ``nw_se`` alone joined it does not.
    * **9. ``obstacles.blocks_leaps`` on a hexagon.** The leg ``_lame_path`` picks is the
      square lattice's, and on a hexagon it can fall on an absent cell, which would block
      a leap with a square that is not there.
    """
    if board_shape(spec) != "hexagon":
        return
    board = spec["board"]
    width = board.get("width")
    if not isinstance(width, int) or isinstance(width, bool) or width % 2 == 0:
        return  # refused as a frame already; no radius to measure against
    radius = (width - 1) // 2

    obstacles = rules.get("obstacles")
    if isinstance(obstacles, dict) and obstacles.get("blocks_leaps") is True:
        errors.append(
            "rules.obstacles.blocks_leaps on a hexagon: the leg a blocked leap is checked "
            "on is the square lattice's, and on a hexagon it can fall on a corner that is "
            "not a cell, blocking the leap with a square that is not there"
        )

    stated = wrap_stated(rules.get("wrap"), "hexagon")
    if not any(value not in (None, "none") for values in stated.values() for value in values):
        return

    drops = rules.get("drops")
    folding = [
        key for key in ("n_s", "ne_sw")
        if any(value not in (None, "none") for value in stated[key])
    ]
    if isinstance(drops, dict) and drops.get("no_doubled") and folding:
        errors.append(
            f"rules.drops.no_doubled on a hexagon whose {' and '.join(folding)} sides are "
            "joined: a file is counted as a line, and a step up across the top or the "
            "upper right comes back onto another file; join only nw_se, or drop the rule"
        )

    bound = radius + 1
    long: list[str] = []
    for index, piece in enumerate(pieces):
        moves = piece.get("moves") if isinstance(piece, dict) else None
        for number, move in enumerate(moves if isinstance(moves, list) else ()):
            vectors = move.get("vectors") if isinstance(move, dict) else None
            if _longest_hex_vector(vectors) > bound:
                long.append(f"pieces[{index}].moves[{number}]")
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        landing = rule.get("landing") if isinstance(rule, dict) else None
        if isinstance(landing, dict) and _longest_hex_vector(landing.get("vectors")) > bound:
            long.append(f"rules.{key}.landing")
    if long:
        errors.append(
            f"{', '.join(long)} steps further than {bound} cells, one more than the "
            f"hexagon's radius {radius}; with a side joined, a longer step can land two "
            "copies of the board away and has no single way round the seams. Shorten "
            "the vector or join no side"
        )
    explosion = rules.get("explode_on_capture")
    blast = explosion.get("radius", 1) if isinstance(explosion, dict) else None
    if isinstance(blast, int) and not isinstance(blast, bool) and 2 * blast > bound:
        errors.append(
            f"rules.explode_on_capture.radius {blast} reaches across more than {bound} "
            f"cells, one more than the hexagon's radius {radius}; with a side joined, a "
            "blast that wide can meet itself round the seams. Use a smaller radius or "
            "join no side"
        )


def _longest_hex_vector(vectors: Any) -> int:
    """The longest well-formed vector in ``vectors``, in hex steps; 0 when none is."""
    return max(
        (
            hex_distance(vector[0], vector[1])
            for vector in vectors
            if isinstance(vector, list)
            and len(vector) == 2
            and all(isinstance(c, int) and not isinstance(c, bool) for c in vector)
        ),
        default=0,
    ) if isinstance(vectors, list) else 0


def _check_named_squares(
    rules: dict[str, Any],
    setup: dict[str, Any],
    width: int,
    height: int,
    orientation: str,
    blocked: AbstractSet[int],
    absent: bool,
    errors: list[str],
) -> None:
    """`_check_obstacle_coherence`'s sets against one kind of unstandable square.

    ``absent`` says the squares are a hexagon's missing corners rather than walls; the
    checks are the same and only the words differ.
    """

    # `reflection` is how a named set reaches the second player: a `reflect_square`
    # orientation, or None when both players read the same squares.
    def offenders(names: Any, reflection: str | None) -> list[str]:
        found: list[str] = []
        if not isinstance(names, list):
            return found
        for name in names:
            if not isinstance(name, str):
                continue
            coords = parse_square(name, width, height)
            if coords is None:
                continue
            file, rank = coords
            candidates = [(file, rank)]
            if reflection is not None:
                candidates.append(reflect_square(file, rank, width, height, reflection))
            if any(r * width + f in blocked for f, r in candidates):
                found.append(name)
        return found

    placements = setup.get("placements")
    if isinstance(placements, list):
        army = orientation if setup.get("mirror", True) else None
        standing: list[str] = []
        for placement in placements:
            if isinstance(placement, dict):
                standing += offenders(placement.get("squares"), army)
        if standing and absent:
            errors.append(
                "setup.placements stands a piece off the hexagon at "
                f"{', '.join(sorted(set(standing)))}; those squares are not cells of "
                "this board"
            )
        elif standing:
            errors.append(
                "setup.placements stands a piece on an obstacle at "
                f"{', '.join(sorted(set(standing)))}; nothing may occupy a wall"
            )

    named: list[tuple[str, Any, str | None]] = []
    reach = rules.get("reach")
    if isinstance(reach, dict):
        named.append(("rules.reach.squares", reach.get("squares"), reach_reflection(reach)))
    scoring = rules.get("scoring")
    if isinstance(scoring, dict):
        named.append(("rules.scoring.region", scoring.get("region"), orientation))
    drops = rules.get("drops")
    if isinstance(drops, dict):
        # A map from piece id to squares, not a list of squares -- which is why this
        # entry did nothing at all until `chess_variant_sandbox/fuzz.py` generated a spec that dropped
        # a piece onto a wall and the check that was supposed to refuse it stayed quiet.
        # `offenders` returns immediately on anything that is not a list, so the fourth
        # of the four rules this function claims to check was never checked.
        region = drops.get("region")
        if isinstance(region, dict):
            for piece, squares in region.items():
                named.append((f"rules.drops.region[{piece!r}]", squares, orientation))
    listed = isinstance(rules.get("promotion"), list)
    for index, entry in enumerate(_promotion_entries(rules)):
        if isinstance(entry, dict) and "squares" in entry:
            where = f"rules.promotion[{index}]" if listed else "rules.promotion"
            named.append((f"{where}.squares", entry["squares"], orientation))
    obstacles = rules.get("obstacles")
    if absent and isinstance(obstacles, dict):
        # The eighth, a hexagon's alone: a wall where there is no cell. Its mirror is a
        # corner too, since the half turn maps the hexagon onto itself.
        named.append(("rules.obstacles.squares", obstacles.get("squares"), None))
    for where, names, reflection in named:
        bad = offenders(names, reflection)
        if bad and absent:
            errors.append(
                f"{where} names a square off the hexagon at {', '.join(sorted(set(bad)))}; "
                "a rule cannot be about a square that is not a cell of the board"
            )
        elif bad:
            errors.append(
                f"{where} names an obstacle at {', '.join(sorted(set(bad)))}; a rule "
                "cannot be about a square nobody can stand on"
            )

    # The sixth: a wild piece's route, mirrors included, resolved the way the compiler
    # resolves it so the two cannot disagree about which squares a route visits.
    for where, squares in npc_routes(rules, width, height, orientation):
        walled = sorted(
            format_square(square % width, square // width)
            for square in squares
            if square in blocked
        )
        if walled and absent:
            errors.append(
                f"{where} walks off the hexagon at {', '.join(walled)}; a rule cannot be "
                "about a square that is not a cell of the board"
            )
        elif walled:
            errors.append(
                f"{where} walks through an obstacle at {', '.join(walled)}; a rule "
                "cannot be about a square nobody can stand on"
            )

    # Since schema 18 (Q781) the span, not the rank: see `castling_spans`. A wall beyond
    # every rook is Omega's walled rank, which `_schema_eighteen_uses` names.
    crossed: dict[int, set[int]] = {}
    for rank, files in castling_spans(rules, setup, width, height, orientation):
        crossed.setdefault(rank, set()).update(
            file for file in files if rank * width + file in blocked
        )
    for rank, in_span in sorted(crossed.items()):
        if not in_span:
            continue
        standing = [format_square(file, rank) for file in sorted(in_span)]
        if absent:
            errors.append(
                f"rules.castling happens along rank {rank + 1}, which runs off the hexagon "
                f"at {', '.join(standing)} between the king and a rook it castles with, "
                "or where one of them lands; the king and the rook walk a rank by file and "
                "would step onto squares that are not cells"
            )
        else:
            errors.append(
                f"rules.castling happens along rank {rank + 1}, which holds an obstacle at "
                f"{', '.join(standing)} between the king and a rook it castles with, or "
                "where one of them lands; the castling move would carry a piece through "
                "or onto a wall, and nothing may stand on one"
            )


def _castling_ranks(
    rules: dict[str, Any],
    setup: dict[str, Any],
    width: int,
    height: int,
    orientation: str,
) -> list[int]:
    """The ranks a castling move can travel along, from where the king starts.

    **A rank, rather than the exact span, and the reason is that the span is not knowable
    from the spec.** Castling requires an *unmoved* king, and ``make`` clears that bit
    for every square it writes, so the king can only ever castle from where ``setup``
    put it -- that much is exact. But the rook it castles with is whichever unmoved rook
    of the right type stands on that rank at the time, the span runs between the two,
    and both destinations depend on the rule's spelling. Reconstructing all of that here
    would be a second implementation of ``generate_castling`` living in the validator,
    which is the duplication this project spends its design budget avoiding.

    A rank is sound instead of exact: both pieces move along it and stop on it, so every
    square either could reach is in the rank, and a wall anywhere else stays legal --
    which is what keeps ``quarry`` (walls on c4/f4, castling on rank 1) accepted.
    """
    castling = rules.get("castling")
    if not isinstance(castling, dict) or not isinstance(setup.get("placements"), list):
        return []
    return sorted(_placed_ranks(setup, castling.get("king"), width, height, orientation))


def castling_spans(
    rules: dict[str, Any],
    setup: dict[str, Any],
    width: int,
    height: int,
    orientation: str,
) -> list[tuple[int, frozenset[int]]]:
    """Every file a castling move can cross or land on, as ``(rank, files)`` (schema 18).

    One entry per king and rook of one side that start on one rank. ``_castling_ranks`` says
    a whole rank because the span is worked out at play time, and Q781 bounds it from the
    spec all the same (`docs/archive/plan-schema-17-18.md` §9.1): castling needs an unmoved
    king and an unmoved rook, and an unmoved piece only ever stands on a square ``setup``
    put it on (``make`` clears the bit on every square it writes; schema 17's circe rebirth
    restores it only on a setup square of the reborn piece's own type). So the rook is one
    of the setup rooks on the king's rank, of the king's side, and what the core walks is:

    * step castling (``generate_castling``): every file from the king to the rook, both
      included. The king lands strictly inside it and the rook beside the king, or the
      move is not generated;
    * file castling (``king_file``): the same, widened to the two fixed destination files
      of the side the rook stands on, which the core also checks are empty.

    Sound and exact: a wall outside every span is a square no castling ever crosses or
    lands on, which is what lets Omega castle with walls at a2 and l2. Malformed parts
    are skipped; the validator names them elsewhere.
    """
    castling = rules.get("castling")
    if not isinstance(castling, dict) or not isinstance(setup.get("placements"), list):
        return []
    kings = _placed_squares(setup, castling.get("king"), width, height, orientation)
    rooks = _placed_squares(setup, castling.get("rook"), width, height, orientation)
    destinations: dict[str, tuple[int, int]] = {}
    king_files = castling.get("king_file")
    if isinstance(king_files, dict):
        rook_files = castling.get("rook_file")
        for side in ("kingside", "queenside"):
            king_to = parse_file(king_files.get(side), width)
            if king_to is None:
                continue
            if isinstance(rook_files, dict):
                rook_to = parse_file(rook_files.get(side), width)
            else:
                # The compiler's default (`_compile_castling`): beside the king, inward.
                rook_to = king_to - 1 if side == "kingside" else king_to + 1
            if rook_to is not None:
                destinations[side] = (king_to, rook_to)
    spans: list[tuple[int, frozenset[int]]] = []
    for player, king_file, rank in sorted(kings):
        for owner, rook_file, rook_rank in sorted(rooks):
            if owner != player or rook_rank != rank or rook_file == king_file:
                continue
            files = {king_file, rook_file}
            side = "kingside" if rook_file > king_file else "queenside"
            files.update(destinations.get(side, ()))
            low, high = min(files), max(files)
            spans.append(
                (rank, frozenset(file for file in range(low, high + 1) if 0 <= file < width))
            )
    return spans


def castling_rank_walled_beyond_its_spans(spec: dict[str, Any]) -> bool:
    """Whether castling's rank holds a wall or an absent cell that no castling reaches.

    Schema 18's lifted restriction (Q781, Omega chess): before it, any wall or absent cell
    on a rank a castling king starts on was refused, because the validator read the rank
    and not the span. True only when such a square exists and every span
    (``castling_spans``) is clear of them; a walled span is still refused, at every
    version, by ``_check_obstacle_coherence``.
    """
    board = spec.get("board")
    rules = spec.get("rules")
    setup = spec.get("setup")
    if not (isinstance(board, dict) and isinstance(rules, dict) and isinstance(setup, dict)):
        return False
    width, height = board.get("width"), board.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return False
    if not isinstance(rules.get("castling"), dict):
        return False
    orientation = setup.get("orientation", "mirror")
    blocked = obstacle_squares(rules, width, height, orientation) | absent_squares(board)
    if not blocked:
        return False
    ranks = set(_castling_ranks(rules, setup, width, height, orientation))
    if not any(square // width in ranks for square in blocked):
        return False
    return not any(
        rank * width + file in blocked
        for rank, files in castling_spans(rules, setup, width, height, orientation)
        for file in files
    )


def _placed_squares(
    setup: dict[str, Any],
    piece_id: Any,
    width: int,
    height: int,
    orientation: str,
) -> set[tuple[int, int, int]]:
    """``(player, file, rank)`` for every square ``piece_id`` starts on, both armies.

    ``_placed_ranks`` with the owner kept, which castling needs: a king castles only with
    a rook of its own side.
    """
    placements = setup.get("placements")
    mirrored = bool(setup.get("mirror", True))
    squares: set[tuple[int, int, int]] = set()
    for placement in placements if isinstance(placements, list) else []:
        if not isinstance(placement, dict) or placement.get("piece") != piece_id:
            continue
        names = placement.get("squares")
        player = placement.get("player", 0)
        if not isinstance(names, list) or player not in (0, 1):
            continue
        for name in names:
            coords = parse_square(name, width, height) if isinstance(name, str) else None
            if coords is None:
                continue
            file, rank = coords
            squares.add((player, file, rank))
            if mirrored:
                image = reflect_square(file, rank, width, height, orientation)
                squares.add((1 - player, *image))
    return squares


def _placed_ranks(
    setup: dict[str, Any],
    piece_id: Any,
    width: int,
    height: int,
    orientation: str,
) -> set[int]:
    """The ranks ``piece_id`` stands on at ply zero, the mirrored army included.

    Ply zero is the whole question wherever ``unmoved`` is: ``make`` clears the bit for
    every square it writes, so a piece that has been anywhere is not where it started.
    Shared by ``_castling_ranks`` and by the inert-castling refusal, which have to agree
    about where a king and a rook start or the second is refusing the first's game.
    """
    placements = setup.get("placements")
    mirrored = bool(setup.get("mirror", True))
    ranks: set[int] = set()
    for placement in placements if isinstance(placements, list) else []:
        if not isinstance(placement, dict) or placement.get("piece") != piece_id:
            continue
        squares = placement.get("squares")
        if not isinstance(squares, list):
            continue
        for name in squares:
            if not isinstance(name, str):
                continue
            coords = parse_square(name, width, height)
            if coords is None:
                continue
            file, rank = coords
            ranks.add(rank)
            if mirrored:
                ranks.add(reflect_square(file, rank, width, height, orientation)[1])
    return ranks


def _check_strike_coherence(
    pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """A struck royal is a captured royal, and two rules may not both answer the same
    question.

    A demotable king makes ``in_check`` a question about a piece type that does not
    exist yet: the check filter, the mate test and every goal term would each need
    their own answer to "is the thing this king turns into still royal". Refusing the
    combination is the ``royal_facing`` guard's shape -- say no rather than play a
    plausible wrong game.
    """
    strike = rules.get("strike")
    if not strike_is_live(strike):
        return
    assert isinstance(strike, dict)

    # Two rules cannot both say what happens to a taken piece. An exploding strike would
    # transform the victim and then destroy it; a strike under ``drops`` would put the
    # same piece in the captor's hand *and* leave it standing. Both are refusals rather
    # than orderings, because neither ordering is the obvious one and choosing one
    # silently is how a plausible wrong game gets played.
    for other, why in (
        ("explode_on_capture", "an explosion destroys what a strike just transformed"),
        ("drops", "a struck piece cannot both stand on the board and sit in a hand"),
    ):
        if rules.get(other):
            errors.append(
                f"rules.strike and rules.{other} both say what happens to a taken "
                f"piece; {why}"
            )

    immune = strike.get("immune")
    immune_ids = _str_set(immune) if isinstance(immune, list) else set()
    chain = strike.get("chain")
    struck_ids = _str_set(
        link.get("from")
        for link in (chain if isinstance(chain, list) else [])
        if isinstance(link, dict)
    )
    switching = strike.get("switches_sides")
    switching_ids = _str_set(switching) if isinstance(switching, list) else set()
    for piece in pieces:
        if not isinstance(piece, dict) or not piece.get("royal"):
            continue
        pid = piece.get("id")
        # Schema 17: a royal that changed sides would leave one side with two kings and
        # the other with none, which is not a rule anyone states on purpose.
        if _in(pid, switching_ids):
            errors.append(
                f"rules.strike.switches_sides names the royal piece {pid!r}; a royal "
                "never changes sides, so leave it out of the list"
            )
        if _in(pid, immune_ids):
            continue
        if _in(pid, struck_ids):
            errors.append(
                f"rules.strike.chain names the royal piece {pid!r}; a struck royal must "
                "be a captured royal, so list it in rules.strike.immune instead"
            )
        else:
            errors.append(
                f"the royal piece {pid!r} is not in rules.strike.immune; a struck royal "
                "must be a captured royal, so every royal piece has to be named immune"
            )


def _check_contact_coherence(
    pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """The contact table against every other rule that answers the same question.

    ``strike`` and ``contacts`` are **two spellings of one mechanism** and are refused
    together. That is the whole of the relationship between them at the spec level: they
    compile to the same ``StrikeRule``, so a spec carrying both would be two rules
    saying what happens when two pieces meet, and neither ordering is the obvious one --
    the shape ``npcs`` beside ``strike`` already has, and the reason ``contacts`` is not
    simply a rewrite of ``strike``.

    The other three refusals are ``_check_strike_coherence``'s, restated over this
    spelling rather than shared with it, because the messages have to name the key the
    designer actually wrote or the error points at a rule that is not in the file.
    """
    contacts = rules.get("contacts")
    if not isinstance(contacts, dict) or not contacts.get("links"):
        return
    if not isinstance(contacts["links"], list):
        return

    if rules.get("strike"):
        errors.append(
            "rules.contacts and rules.strike are two spellings of one rule; a strike is "
            "a contact table with no mover column, so write one or the other"
        )
    for other, why in (
        ("explode_on_capture", "an explosion destroys what a contact just transformed"),
        ("drops", "a transformed piece cannot both stand on the board and sit in a hand"),
        ("npcs", "a wild piece eats what it steps onto, and a contact says what that meeting leaves behind"),
    ):
        if rules.get(other):
            errors.append(
                f"rules.contacts and rules.{other} both say what happens to a taken "
                f"piece; {why}"
            )

    immune = contacts.get("immune")
    immune_ids = _str_set(immune) if isinstance(immune, list) else set()
    links = [link for link in contacts["links"] if isinstance(link, dict)]
    # Schema 11: `must_capture` asks whether a move takes an enemy piece, and nothing
    # settles whether a move onto an own piece is one. Refused rather than guessed.
    if rules.get("must_capture") and any(link.get("own") is True for link in links):
        errors.append(
            "rules.contacts has a link onto an own piece beside rules.must_capture; "
            "whether moving onto your own piece counts as a capture is not settled"
        )
    met = _str_set(link.get("onto") for link in links)
    for piece in pieces:
        if not isinstance(piece, dict) or not piece.get("royal"):
            continue
        pid = piece.get("id")
        if not _in(pid, immune_ids):
            errors.append(
                f"the royal piece {pid!r} is not in rules.contacts.immune; a struck "
                "royal must be a captured royal, so every royal piece has to be named "
                "immune"
                + (
                    f" -- and rules.contacts.links names {pid!r} on the 'onto' side"
                    if _in(pid, met)
                    else ""
                )
            )
        # A royal that *stops being royal* is the same problem read from the attacker's
        # side: check would be a question about a piece type that is not there yet.
        for index, link in enumerate(links):
            if link.get("mover") == pid and "mover_to" in link:
                errors.append(
                    f"rules.contacts.links[{index}] turns the royal piece {pid!r} into "
                    f"{link['mover_to']!r}; a royal that stops being royal makes check a "
                    "question about a piece that is not on the board"
                )


def _check_npc_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Wild pieces against everything else in the spec, each refusal by name.

    Five kinds of refusal, and none is a default the engine could pick for the designer:

    * **a wild royal** -- check is about a *player's* royal, and a royal with no side has
      nobody to lose it;
    * **two routes sharing a square**, or one route visiting a square twice -- which
      square is *next* has to be a fact about the square alone, or ``npc_next`` is not a
      table and the core has to know which wild piece it is looking at. Several wild
      pieces on *one* route are schema 23's ``also_at``, which keeps the tables a route's;
    * **a wild piece starting under the opening array** -- two pieces on one square, for
      every start, ``also_at``'s included;
    * **``strike``, ``explode_on_capture`` and a ``turn`` longer than one move** -- the
      first two are two rules saying what happens when a piece is taken (the shape
      ``_check_strike_coherence`` refuses), and the third puts the phase *between turns*
      where a legality filter would have to know whether the move it is testing ends the
      turn. One condition in two places, liftable, and not worth its test surface in the
      bump that introduces the rule;
    * **``scoring.count: "points"`` with nothing that pays** -- a count with nothing to
      count, the ``territory`` shape.

    A wild piece's route through a wall is refused beside the other five rules that name
    a square nobody may stand on, in ``_check_obstacle_coherence``.
    """
    npcs = rules.get("npcs")
    scoring = rules.get("scoring")
    wants_points = isinstance(scoring, dict) and scoring.get("count") == "points"
    if not isinstance(npcs, list) or not npcs:
        if wants_points:
            errors.append(
                'rules.scoring.count "points" needs rules.npcs with a bounty above zero, '
                "or there is nothing to count"
            )
        return

    board = spec.get("board")
    if not isinstance(board, dict):
        return
    width, height = board.get("width"), board.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return
    raw_setup = spec.get("setup")
    setup: dict[str, Any] = raw_setup if isinstance(raw_setup, dict) else {}
    orientation = setup.get("orientation", "mirror")

    by_id = {p["id"]: p for p in pieces if isinstance(p, dict) and isinstance(p.get("id"), str)}
    for index, npc in enumerate(npcs):
        if not isinstance(npc, dict):
            continue
        npc_piece = npc.get("piece")
        piece = by_id.get(npc_piece) if isinstance(npc_piece, str) else None
        if piece is not None and piece.get("royal"):
            errors.append(
                f"rules.npcs[{index}].piece {npc.get('piece')!r} is royal; a wild royal "
                "has no side to lose it, and check is about a player's royal"
            )

    routes = npc_routes(rules, width, height, orientation)
    seen: dict[int, str] = {}
    for where, squares in routes:
        for square in squares:
            name = format_square(square % width, square // width)
            if square in seen:
                errors.append(
                    f"{where} and {seen[square]} both walk {name}; routes may not share "
                    "a square, because which square is next has to be a fact about the "
                    "square alone. To put more than one wild piece on one route, list "
                    "the others' starting squares in that entry's also_at"
                )
            else:
                seen[square] = where

    placements = setup.get("placements")
    standing: set[int] = set()
    if isinstance(placements, list):
        mirrored = bool(setup.get("mirror", True))
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            named = placement.get("squares")
            for name in named if isinstance(named, list) else []:
                coords = parse_square(name, width, height) if isinstance(name, str) else None
                if coords is None:
                    continue
                file, rank = coords
                standing.add(rank * width + file)
                if mirrored:
                    other_file, other_rank = reflect_square(
                        file, rank, width, height, orientation
                    )
                    standing.add(other_rank * width + other_file)
    # Every start, not only a route's first square (schema 23's ``also_at``).
    for _, where, square in npc_starts(rules, width, height, orientation):
        if square in standing:
            start = format_square(square % width, square // width)
            errors.append(
                f"{where} starts on {start}, where setup.placements stands a piece; a "
                "wild piece needs an empty square to start from"
            )

    strike = rules.get("strike")
    if strike_is_live(strike):
        errors.append(
            "rules.npcs and rules.strike both say what happens when a piece is taken; a "
            "struck wild piece would have to become a wild piece of another type, or a "
            "player's, and neither is a rule anybody wrote"
        )
    if rules.get("explode_on_capture"):
        errors.append(
            "rules.npcs and rules.explode_on_capture both say what happens when a piece "
            "is taken; a blast that destroys a wild piece pays nobody its bounty, and a "
            "wild piece that eats does not explode"
        )
    if turn_is_long(rules):
        errors.append(
            "rules.npcs beside a turn longer than one move: a wild piece walks between "
            "turns, and a legality filter would have to know whether the move it is "
            "testing ends the turn; use one move per turn"
        )

    if wants_points:
        def bounty_of(npc: Any) -> float:
            if not isinstance(npc, dict):
                return 0.0
            stated = npc.get("bounty")
            if isinstance(stated, (int, float)) and not isinstance(stated, bool):
                return float(stated)
            piece = by_id.get(npc.get("piece"))
            value = piece.get("value", 1) if isinstance(piece, dict) else 1
            return float(value) if isinstance(value, (int, float)) else 1.0

        if not any(bounty_of(npc) > 0 for npc in npcs):
            errors.append(
                'rules.scoring.count "points" needs a wild piece with a bounty above '
                "zero, or there is nothing to count"
            )


def _check_entry_promotion_coherence(
    spec: dict[str, Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """``promote_on_entry`` against the rules that would have to feed it.

    Three refusals, and each is a flag that could never fire -- which is the ``points``
    shape: a rule with nothing to act on is a spec mistake rather than a default, and
    saying so by name is cheaper than charging a player for a sentence about nothing.

    * **no promotion rule** -- nothing promotes, so nothing can promote on entry;
    * **no way to enter play** -- a piece reaches a square without moving there by
      exactly two routes, a drop and a strike that leaves the struck piece standing, and
      a spec with neither has no event for the flag to fire on;
    * **no promoting piece can enter its zone** -- the routes exist but every promoting
      type is kept out: not on the strike chain's ``to`` side, and either never in a
      hand, or held but droppable only outside its zone (crazyhouse's pawn, which
      ``drops.region`` keeps off the far rank, is the case). Checked for the first
      player only, because a drop region and a promotion zone are both stated for the
      first player and reflected for the second, so the two intersections are each
      other's reflection.

    *Never in a hand* is ``_never_in_a_hand`` (Q483): a stone-only hand
    (``from_captures: false``, ``archive/stones``) holds only what its stock names, and
    until Q483 a promoting piece with no ``drops.region`` entry was read as droppable
    anywhere without that question being asked, so the flag was accepted beside a hand
    no promoting piece could reach.

    ``false`` is the absence of the rule and is not checked, on the grounds
    ``"en_passant": false`` is not.
    """
    flag = rules.get("promote_on_entry")
    if flag is not True:
        return
    entries = [
        entry
        for entry in _promotion_entries(rules)
        if isinstance(entry, dict) and isinstance(entry.get("piece"), str)
    ]
    if not entries:
        errors.append(
            "rules.promote_on_entry needs rules.promotion: nothing promotes, so nothing "
            "can promote on entering play"
        )
        return

    # `{}` is the absence of a hand, as `_check_inert_rules` and `_compile_drops` read it.
    drops = rules.get("drops") if isinstance(rules.get("drops"), dict) and rules["drops"] else None
    strike = rules.get("strike")
    survivors: set[str] = set()
    if (
        isinstance(strike, dict)
        and strike.get("effect", "capture") != "capture"
        and isinstance(strike.get("chain"), list)
    ):
        immune = strike.get("immune")
        immune_ids = set(immune) if isinstance(immune, list) else set()
        survivors = {
            link["to"]
            for link in strike["chain"]
            if isinstance(link, dict) and link.get("from") not in immune_ids
            and isinstance(link.get("to"), str)
        }
    # Schema 17 (Q540): a piece reborn on its home square enters play there, as itself,
    # and a capturer that switches sides enters play for the other side; both go through
    # the core's `arriving_value`, as a survivor does.
    if isinstance(strike, dict):
        immune = strike.get("immune")
        immune_ids = set(immune) if isinstance(immune, list) else set()
        if strike.get("effect", "capture") == "capture" and strike.get("position") == "home":
            survivors |= {
                piece["id"]
                for piece in spec.get("pieces") or []
                if isinstance(piece, dict) and isinstance(piece.get("id"), str)
                and piece["id"] not in immune_ids
            }
        switching = strike.get("switches_sides")
        if isinstance(switching, list):
            survivors |= {pid for pid in switching if isinstance(pid, str)}
    # The same walk over the other spelling (schema 10, Q19a). A contact enters a piece
    # into play on both columns: the target that survives as a new type, and the mover,
    # which under `stay` becomes something else without having moved anywhere.
    contacts = rules.get("contacts")
    if isinstance(contacts, dict) and isinstance(contacts.get("links"), list):
        immune = contacts.get("immune")
        immune_ids = set(immune) if isinstance(immune, list) else set()
        for link in contacts["links"]:
            if not isinstance(link, dict):
                continue
            if isinstance(link.get("mover_to"), str):
                survivors.add(link["mover_to"])
            if link.get("onto") not in immune_ids and isinstance(link.get("onto_to"), str):
                survivors.add(link["onto_to"])
    if drops is None and not survivors:
        errors.append(
            "rules.promote_on_entry: nothing enters play except by moving; it needs "
            "rules.drops, a rules.strike whose chain leaves a struck piece standing, or "
            "a rules.contacts link that turns a piece into another"
        )
        return

    board = spec.get("board")
    if not isinstance(board, dict):
        return
    width, height = board.get("width"), board.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return
    raw_setup = spec.get("setup")
    orientation = raw_setup.get("orientation", "mirror") if isinstance(raw_setup, dict) else "mirror"
    region = (drops.get("region") or {}) if drops is not None else {}
    kept_out: list[str] = []
    for entry in entries:
        pid = entry["piece"]
        if pid in survivors:
            return  # a strike can leave it standing on any square, the zone included
        if drops is None:
            kept_out.append(f"{pid} is not on the strike chain's 'to' side")
            continue
        never_held = _never_in_a_hand(spec, rules, drops, pid)
        if never_held is not None:
            kept_out.append(
                f"{pid} {never_held}"
                + (" and is not on the strike chain's 'to' side" if survivors else "")
            )
            continue
        squares = region.get(pid) if isinstance(region, dict) else None
        if squares is None:
            return  # droppable anywhere, the zone included
        zone = promotion_squares(entry, width, height, orientation)[0]
        droppable = {
            coords[1] * width + coords[0]
            for name in (squares if isinstance(squares, list) else [])
            if isinstance(name, str)
            for coords in [parse_square(name, width, height)]
            if coords is not None
        }
        if zone & droppable:
            return
        kept_out.append(
            f"{pid} may not be dropped on {_zone_words(zone, width)}"
            + (" and is not on the strike chain's 'to' side" if survivors else "")
        )
    errors.append(
        "rules.promote_on_entry can never fire: no piece with a promotion rule can "
        "enter play inside its zone -- " + "; ".join(kept_out)
    )


def _never_in_a_hand(
    spec: dict[str, Any], rules: dict[str, Any], drops: dict[str, Any], pid: str
) -> str | None:
    """Why ``pid`` can never be in a player's hand, or ``None`` when it can.

    A hand fills by the two routes ``reachability.pieces_a_player_can_hold`` reads, and
    by nothing else the schema has: a ``drops.stock`` that names the type, and a capture,
    which puts the taken piece in the captor's hand as its ``promoted_from`` where it
    declares one. ``drops.region`` says where a held piece may go and is not a way of
    holding one (the reading ``_validate_npc_immune`` makes for the same reason).

    Over-approximates in the walk's direction: any type that can stand is taken to be
    capturable, because *whether anything ever takes it* is play. What this refuses is a
    fact about the spec -- no route puts the type in a hand at all.

    ``None`` too on a malformed ``stock`` or ``from_captures``, which are refused by name
    elsewhere, and on a type that can never stand, which inert-rule case 13 refuses: an
    inert-rule refusal has to be a fact, not a consequence of a mistake somewhere else.
    """
    stock = drops.get("stock")
    if stock is not None and not isinstance(stock, dict):
        return None
    if pid in (stock or {}):
        return None
    from_captures = drops.get("from_captures", True)
    if not isinstance(from_captures, bool):
        return None
    if not from_captures:
        return (
            "never reaches a hand, since rules.drops.from_captures is false and "
            "rules.drops.stock does not name it"
        )
    listed = spec.get("pieces")
    listed = listed if isinstance(listed, list) else []
    declared = {
        piece["id"]
        for piece in listed
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    demoted = {
        piece["id"]: piece["promoted_from"]
        for piece in listed
        if isinstance(piece, dict)
        and isinstance(piece.get("id"), str)
        and isinstance(piece.get("promoted_from"), str)
        and piece["promoted_from"] in declared
    }
    if not _routes_are_readable(spec, listed, rules):
        return None
    standing = _pieces_that_can_stand(spec, listed, rules, declared)
    if pid not in standing or any(demoted.get(other, other) == pid for other in standing):
        return None
    return (
        f"never reaches a hand, since rules.drops.stock does not name it and a captured "
        f"{pid} is held as {demoted[pid]}"
    )


def _routes_are_readable(spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any]) -> bool:
    """Whether ``reachability`` can read every route into play this spec names.

    It asks set membership of a placement's, a wild piece's and a ``promoted_from``'s piece
    id, and reads ``rules.drops`` as an object; a list where an id belongs, or a drops that
    is not an object, is refused by its shape check. Then which pieces can stand is not
    known, so the refusals that read it say nothing rather than crash or echo the typo.
    """
    drops = rules.get("drops")
    if drops is not None and not isinstance(drops, dict):
        return False
    setup = spec.get("setup")
    placements = setup.get("placements") if isinstance(setup, dict) else None
    npcs = rules.get("npcs")
    named = [
        entry.get("piece")
        for group in (placements, npcs)
        for entry in (group if isinstance(group, list) else [])
        if isinstance(entry, dict)
    ] + [piece.get("promoted_from") for piece in pieces if isinstance(piece, dict)]
    return all(value is None or isinstance(value, str) for value in named)


def _zone_words(zone: AbstractSet[int], width: int) -> str:
    """A resolved zone as a reader names it: ``rank 8`` when it is whole ranks, else squares."""
    ranks = sorted({square // width for square in zone})
    if len(zone) == len(ranks) * width:
        return f"rank {', '.join(str(rank + 1) for rank in ranks)}"
    return ", ".join(format_square(square % width, square // width) for square in sorted(zone))


_NEVER_ON_THE_BOARD = (
    "it is not placed by setup.placements, is no promotion's target, is not left "
    "standing by a strike chain or a contact link, cannot come out of a hand or a "
    "stock, and is not a wild piece"
)


def _check_inert_rules(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """Rules that are charged for and declare nothing, refused by name.

    A rule that does nothing is not free. It is one game with two spellings: the spec
    with it and the spec without it play the same game under two hashes, so one game
    would be filed, measured and archived twice. ``complexity._RULE_COSTS``
    also charges for the declaration whatever it compiles to, so the longer spelling
    reads as more rule than the game has; under ``v10``, where ``rule_mass`` is the
    *denominator* of ``rule_yield``, it deflated the ratio the index rewarded. Refusing
    it by name is the shape
    ``_check_entry_promotion_coherence`` and the ``scoring.count "points"`` refusal
    already have: a rule with nothing to act on is a spec mistake rather than a default
    the engine could pick for the designer, and a proposal LLM told which rule is empty
    and why can drop it or give it something to do.

    Every case here is **decidable from the spec alone**, and what "inert" is worth is
    settled by reading the source rather than by playing. The standard is three steps,
    asked of the *compiled field* and never of the JSON key: name every field of
    ``bridge.build_tables``'s boundary dict whose **value** differs between the declared
    and the undeclared spelling; ``grep`` each one through ``rust/src/`` and read every
    site it has; and call the spelling inert only where each of those readers implements
    the behaviour the rule's *name* describes **and** is a provable no-op under the
    refused spelling. A reader doing anything else is a second meaning, and the clause is
    then not an inert-rule refusal at all.

    **Source proves; behaviour only refutes.** The earlier standard here was six seeds of
    greedy against greedy on the nearest baseline, and it is not one: inertness is a
    universal claim, so one divergent game refutes it and no number of identical games
    settles it, and the power of the check is a property of the *agents* rather than of
    the rule. ``en_passant`` passed it and is not inert; on that very spec ``random`` v
    ``random`` is identical at every seed while ``greedy`` v ``greedy`` diverges. Each
    clause's field list, read sites and verdict are in ``docs/reference/bugs-investigations.md``,
    "Rules that declare nothing" -- **including three that did not come back clean**, which stand as
    refusals for other reasons while what to do about them is decided.

    Nothing here is a judgement about how a game plays out -- three boundaries the
    enumeration drew are load-bearing, and they are stated here because they are what a
    later reader will be tempted to cross:

    * **An inert *spelling* stays legal.** ``turn: {"moves": 1}``, ``wrap: {"files":
      "none"}``, ``strike: {"effect": "capture"}`` and the spelled-out defaults all
      compile to the game that was already there and are charged **nothing** for it
      (``complexity._RULE_ACTIVE``, ``_RULE_DEFAULTS``). They are also the standing proof
      that schemas 4, 5 and 6 were inert bumps, and ``fairy._check_unsupported_rules``
      reads the *compiled* spec precisely so that a spec carrying one still translates.
      Refusing them would change no charge and cost all three.
    * **An unread colouring is not inert.** ``territory`` at ``win_at: 0`` with nothing
      scoring it looks like scenery and is not: a ``permanent`` claim is irreversible, so
      it clears the repetition window and resets the no-progress clock, and a ``repaint``
      colouring enters the Zobrist key. Measured DIFFERENT on every seed. The same trap
      catches ``explode_on_capture.immune`` covering every piece -- immunity protects
      *neighbours*, and the capturer and the victim go regardless.
    * **Whether a rule ever fires *in play* is not a question for the validator.** A
      promotion nobody reaches, a ``reach`` square nobody arrives on, a ``check_limit``
      never met, a ``drops`` for a piece nothing takes: each is a walk of the game tree,
      and the cheap gates and the tournament ask it better than a second engine living in
      the validator could. The line between the two is sharpest at the piece: *this piece
      can never exist* is a fact about the spec, *this piece exists and never gets there*
      is play.

    ``false`` and the empty object are the absence of a rule and are never refused, on
    the grounds ``"en_passant": false`` is not. Nothing here is gated on
    ``schema_version``: a gate exists to stop an old stamp reaching a new feature, and a
    refusal points the other way.
    """
    declared = {
        piece["id"]
        for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    has_royal = any(isinstance(p, dict) and p.get("royal") for p in pieces)

    # -- nothing can ever be taken: three rules about capturing, and a hand that cannot
    # fill. One test serves all four, and it is the cross of the two tables the schema
    # gives for taking a piece: a capture-mode move, and a matrix that permits the pair.
    # A wild piece eats under the same matrix, so it needs no term of its own.
    if not _capture_is_possible(pieces, declared):
        nothing_taken = (
            "no piece has a capture-mode move that rules the capture matrix permits, so "
            "nothing can ever be taken"
        )
        if rules.get("must_capture"):
            errors.append(
                f"rules.must_capture can never bind: {nothing_taken}, so there is never "
                "a capture to prefer"
            )
        # This clause was a **second meaning** until Q133 (`docs/reference/bugs-investigations.md`, "Rules that declare nothing", case 4): the
        # declaration also switched on *a side whose royal is off the board loses at
        # once*, at two sites in `game.rs` that have nothing to do with capturing, so a
        # spec refused here for generating no blast was being refused out of an unrelated
        # ending as well. Q133 separated them -- the ending is derived from the win
        # condition and the setup (`Tables::royal_loss_ends_it`,
        # `docs/reference/derived-royal-loss.md`) -- so the key means one thing and this
        # refusal is about that thing.
        if rules.get("explode_on_capture"):
            errors.append(
                f"rules.explode_on_capture can never fire: {nothing_taken}, so nothing "
                "ever explodes"
            )
        drops = rules.get("drops")
        if (
            isinstance(drops, dict)
            and drops
            and drops.get("from_captures", True)
            and not drops.get("stock")
        ):
            errors.append(
                f"rules.drops can never fill a hand: {nothing_taken}, and a hand fills "
                "only from a capture"
            )

    # -- the other way a hand never fills, and the only one the capture test misses.
    # `{}` is already free (falsy: `_count_special_rules` skips it and `_compile_drops`
    # returns None); `{"from_captures": false}` compiles to a real rule and is charged
    # 2.0. **Q51 moved this line**, exactly as Q57 said it would: a hand can now be
    # stocked, so the condition reads "no from_captures *and* no stock".
    drops = rules.get("drops")
    if (
        isinstance(drops, dict)
        and drops
        and not drops.get("from_captures", True)
        and not drops.get("stock")
    ):
        errors.append(
            'rules.drops has "from_captures": false and nothing else fills a hand, so '
            "there is never anything to drop"
        )

    # -- rules about a royal piece, in a game that has none. `win` is already refused
    # against a missing royal above; these four are what is left, and each is charged
    # 1.0 for a sentence about a piece that is not there. The core already guards
    # `forbid_check` by name (`if t.forbid_check && !t.royal_types.is_empty()`), which
    # is the engine saying this rule is nothing while the validator does not.
    #
    # `bare_royal`'s refusal was true for the wrong reason until Q134 and is now a
    # proof. The core's arm asked *no non-royal piece of mine*, which is vacuously true
    # for a side with nothing left, so the rule fired -- on the annihilation it read by
    # accident rather than on any royal. It asks for a non-empty set of royals now, and
    # with no piece marked royal no board can satisfy that.
    if not has_royal:
        for key, what in (
            ("bare_royal", "an ending for a player left with nothing but a royal piece"),
            ("check_limit", "a count of the checks delivered to a royal piece"),
            ("forbid_check", "a ban on the moves that put a royal piece in check"),
        ):
            if rules.get(key):
                errors.append(
                    f"rules.{key} is {what}, and no piece is marked royal, so it can "
                    "never fire"
                )
        if rules.get("royal_facing") == "forbidden":
            errors.append(
                'rules.royal_facing "forbidden" is a ban on two royal pieces seeing each '
                "other down an open line, and no piece is marked royal, so it can never "
                "fire"
            )

    # -- a counter that cannot advance. If no legal move may give check then no check is
    # ever delivered, and `check_limit` counts deliveries. Guarded on
    # `explode_on_capture` and the guard is the rule rather than a nicety: the
    # `forbid_check` filter plays the move *without the explosion arm* and says so in
    # its own comment, while `delivered` is computed after the blast, so under an
    # explosion a line opened by the blast still ticks the counter.
    if (
        rules.get("check_limit")
        and rules.get("forbid_check")
        and not rules.get("explode_on_capture")
        and has_royal
    ):
        errors.append(
            'rules.check_limit beside "forbid_check": true can never be reached: no '
            "legal move may give check, so the counter never advances"
        )

    # -- en passant with no double step to answer. The rule answers a piece slipping
    # *past* another, and `_ep_passes` builds its table from the promoting piece's own
    # slides -- the second square of a ray, where that square carries MODE_MOVE. No such
    # move and the table is empty, which is what `chess-oriented-pawns` was charged 1.0
    # for until it was repaired. The two refusals above -- a missing `promotion`, and
    # more than one -- come first and are what makes "the promoting piece" a single
    # piece here; a spec already collecting either is left alone.
    #
    # **A hop was read as a double step until Q121 and is not one.** It has a ray, but it
    # lands past its screen, so the square it would "pass over" is the square its screen
    # stands on -- occupied, and every arm that takes in passing needs an empty one. The
    # compiler stopped building those windows and this stopped counting them, together.
    #
    # **The message says "no capture", not "never fires", and the difference was
    # measured.** `ep_type` used to reach `game.rs` at three places and only two of them
    # were en passant: the third made a move by the promoting piece irreversible, so
    # dropping the key cleared the repetition window and the no-progress clock too. On
    # `chess-oriented-pawns` that was worth +0.1958 of index against +0.0080 for the
    # shorter rule list (`docs/reference/en-passant-repair-measurement.md`).
    #
    # **Q113 separated them**, and the third place is now `tables.one_way`, derived from
    # the movement tables rather than switched on by this key
    # (`docs/reference/derived-irreversibility.md`). So the key does exactly what its name
    # says, and a declaration that can generate no capture now really does nothing --
    # which makes this refusal sound where it used to be right for the wrong reason. The
    # wording stays "no capture" all the same: it is the narrower claim and it is the one
    # `can_double_step` actually checks.
    if rules.get("en_passant"):
        entries = _promotion_entries(rules)
        entry = entries[0] if len(entries) == 1 else None
        piece_id = entry.get("piece") if isinstance(entry, dict) else None
        promoter = next(
            (
                piece
                for piece in pieces
                if isinstance(piece, dict) and piece.get("id") == piece_id
            ),
            None,
        )
        if promoter is not None and not can_double_step(promoter):
            errors.append(
                f"rules.en_passant can generate no capture here: it answers a two-square "
                f"move, and {piece_id!r}, the promoting piece it is defined against, has "
                "none -- a leap passes over nothing, a hop passes over its own screen, "
                "and a slide needs a range of at least 2 (or 0 for unlimited), a "
                "min_range of at most 2, and a mode that moves. It is charged 1.0 all "
                "the same."
            )

    # -- castling for a rook that is never unmoved on the king's rank. Both pieces must
    # be unmoved, `generate_castling` looks for the rook along the king's own rank, and
    # the unmoved bit is set once at ply zero and only cleared afterwards -- so a rook
    # arriving by move, drop, promotion or strike is never a castling partner. This is
    # the narrow case: a castling the *walls* block, or one nobody ever plays, is a
    # question about the game and is `_check_obstacle_coherence`'s or the tournament's.
    castling = rules.get("castling")
    board = spec.get("board")
    raw_setup = spec.get("setup")
    if (
        isinstance(castling, dict)
        and isinstance(raw_setup, dict)
        and isinstance(raw_setup.get("placements"), list)
        and isinstance(board, dict)
        and isinstance(board.get("width"), int)
        and isinstance(board.get("height"), int)
        and _in(castling.get("king"), declared)
        and _in(castling.get("rook"), declared)
    ):
        width, height = board["width"], board["height"]
        orientation = raw_setup.get("orientation", "mirror")
        king_ranks = set(_castling_ranks(rules, raw_setup, width, height, orientation))
        rook_ranks = _placed_ranks(
            raw_setup, castling["rook"], width, height, orientation
        )
        if not king_ranks:
            errors.append(
                f"rules.castling can never happen: its king {castling['king']!r} is not "
                "placed by setup.placements, and only a king still standing where setup "
                "left it is unmoved"
            )
        elif not king_ranks & rook_ranks:
            ranks = ", ".join(str(rank + 1) for rank in sorted(king_ranks))
            errors.append(
                f"rules.castling can never happen: no {castling['rook']!r} starts on "
                f"rank {ranks}, where the king does, and both pieces have to be unmoved "
                "on the one rank; a rook arriving by move, drop, promotion or strike is "
                "never unmoved"
            )

    # -- a rule about a piece that can never stand on the board. Not "cannot reach the
    # square" -- "cannot exist". The routes into play are a closed set and the compiler
    # already enumerates them for royalty; this asks the same question of any piece.
    standing = (
        _pieces_that_can_stand(spec, pieces, rules, declared)
        if _routes_are_readable(spec, pieces, rules)
        else declared
    )
    reach = rules.get("reach")
    if isinstance(reach, dict):
        piece_id = reach.get("piece")
        if _in(piece_id, declared) and piece_id not in standing:
            errors.append(
                f"rules.reach is about {piece_id!r}, which can never stand on the "
                f"board: {_NEVER_ON_THE_BOARD}"
            )
    promotion = rules.get("promotion")
    for index, entry in enumerate(_promotion_entries(rules)):
        if not isinstance(entry, dict):
            continue
        piece_id = entry.get("piece")
        if _in(piece_id, declared) and piece_id not in standing:
            where = (
                f"rules.promotion[{index}]" if isinstance(promotion, list)
                else "rules.promotion"
            )
            errors.append(
                f"{where} is about {piece_id!r}, which can never stand on the board: "
                f"{_NEVER_ON_THE_BOARD}"
            )


def _capture_is_possible(pieces: list[Any], declared: set[str]) -> bool:
    """Whether any piece can ever take any other.

    The cross of the two tables the schema gives for taking a piece: a move descriptor
    whose ``mode`` is not ``"move"``, and a schema-3 ``captures`` matrix that permits the
    pair. Absent means everything; ``[]`` never takes and still blocks.

    Answers *yes* on anything malformed, so that a spec already collecting errors is
    never also told a rule of its is empty -- an inert-rule refusal has to be a fact, not
    a consequence of a typo somewhere else.

    **This is a claim about the core, and it was false until 2026-09-07 (Q118).** The
    generator's en passant arm pushed ``MOVE_EN_PASSANT`` onto an *empty* square without
    ever asking ``may_take`` -- the other three arms ask it of the piece standing on the
    square being moved to, and an en passant capture's victim stands somewhere else -- so
    a spec with ``captures: []`` on every piece, which this function answers *no* about,
    still took a piece in passing. ``game.rs::generate_pseudo_legal`` now asks the matrix
    for that arm too, once per position, and the three refusals built on this answer
    (``must_capture``, ``explode_on_capture``, ``drops``) are sound again.
    ``tests/test_engine.py::TestTheCaptureMatrixBindsTheCaptureInPassing`` holds the two
    readings together, because this is exactly the coupling ``docs/reference/bugs-investigations.md`` ("Rules that declare nothing")
    warns about:
    **inert is a property of a (spelling, core) pair**, so an arm added to the generator
    that does not consult the matrix un-proves three shipped clauses silently.
    """
    for attacker in pieces:
        if not isinstance(attacker, dict):
            return True
        moves = attacker.get("moves")
        if not isinstance(moves, list):
            return True
        if any(not isinstance(move, dict) for move in moves):
            return True
        # Schema 13: a sweep's path is a capture site its mode does not govern.
        if all(
            move.get("mode", "both") == "move" and move.get("kind") != "sweep"
            for move in moves
        ):
            continue
        allowed = attacker.get("captures")
        if not isinstance(allowed, list):
            return bool(declared)  # absent means every piece
        if declared.intersection(_str_set(allowed)):
            return True
    return False


def can_double_step(piece: dict[str, Any]) -> bool:
    """Whether a piece has a move that could put it two squares along one line.

    The question ``compile._ep_passes`` asks of the promoting piece, read off the
    descriptors instead of off the compiled table: is there a ray with a second square,
    and does that square carry ``MODE_MOVE``? Per kind, which is all it takes --

    * a **leap** has no ray at all, so it passes over nothing and never qualifies (the
      compiler used to halve its delta for a midpoint, which is a question with no
      answer; ``_ep_passes``'s docstring has that history);
    * a **lame** leap compiles to a two-square ray, ``(leg, target)``, and so always
      qualifies;
    * a **slide** walks its vector, so its second square exists unless ``range`` is 1,
      and carries a mode unless ``min_range`` is above 2 or the descriptor only
      captures;
    * a **hop** has a ray too and still never qualifies (Q121), because it lands past
      its *screen* -- so reaching the second square of the ray means a piece stands on
      the first, and the window would open on an occupied square that no en passant
      capture can land on.

    **Sound and deliberately incomplete, in that order.** An edge, a seam, a wall, a
    region or a per-square condition can each cut a long ray short, and this validator
    does no geometry (``_castling_ranks`` refuses the same duplication for the same
    reason) -- so a descriptor that *could* reach two squares counts as reaching them.
    That direction is the safe one: it refuses fewer specs than the compiled table would
    and never more, and the cost is only that an en passant made inert by the board
    rather than by the descriptor is still accepted.

    **Public because the mutation operator asks the same question.**
    ``loop.operators._toggle_a_rule`` decides whether adding ``en_passant`` is worth
    doing, and a guard of its own would be a second reading of this rule that could
    drift from this one -- which is precisely the defect it had, asking after *any*
    piece's ``range: 2`` where the rule is about the promoting one. One function, one
    answer, and a child the operator offers is never a child the validator refuses.
    """
    moves = piece.get("moves")
    if not isinstance(moves, list):
        return True  # malformed: something else is already refusing it
    for move in moves:
        if not isinstance(move, dict):
            return True
        kind = move.get("kind")
        # A sweep never qualifies: a promoting piece may not sweep (schema 13).
        if kind in ("leap", "hop", "sweep"):
            continue
        if kind == "lame":
            return True
        if kind != "slide":
            return True  # a kind this function has not been taught
        if move.get("mode", "both") == "capture":
            continue
        distance = move.get("range", DEFAULT_RANGE.get(str(kind), 1))
        minimum = move.get("min_range", 1)
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (distance, minimum)
        ):
            return True
        if distance != 1 and minimum <= 2:
            return True
    return False


def promotion_squares(
    promotion: dict[str, Any], width: int, height: int, orientation: str = "mirror"
) -> tuple[frozenset[int], frozenset[int]]:
    """The absolute squares one promotion rule fires on, per player.

    Stated once from the first player's side -- ``rank`` (``"last"`` or an index), a
    ``ranks`` band, or schema 14's ``squares`` -- and carried to the second player by
    ``reflect_square`` under ``setup.orientation``, as ``piece.region`` is. A rank zone is
    its ranks times every file, and a whole rank is its own image under either
    orientation, so every rank-spelled rule resolves to the zone it always had.

    Shared with the compiler, so the validator's idea of a zone and the engine's are one
    idea; ``npc_routes`` is the precedent. Malformed names and ranks off the board are
    skipped, because the validator calls this while still collecting errors about them.
    """
    if "squares" in promotion:
        names = promotion["squares"] if isinstance(promotion["squares"], list) else []
        cells = [parse_square(name, width, height) for name in names if isinstance(name, str)]
        first = frozenset(
            rank * width + file for coords in cells if coords is not None
            for file, rank in [coords]
        )
    else:
        if "ranks" in promotion:
            stated = promotion["ranks"] if isinstance(promotion["ranks"], list) else []
        else:
            rank = promotion.get("rank", "last")
            stated = [height - 1 if rank == "last" else rank]
        first = frozenset(
            rank * width + file
            for rank in stated
            if isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < height
            for file in range(width)
        )
    second = frozenset(
        other_rank * width + other_file
        for square in first
        for other_file, other_rank in [
            reflect_square(square % width, square // width, width, height, orientation)
        ]
    )
    return first, second


def promotion_targets(promotion: dict[str, Any]) -> tuple[list[str], list[str]]:
    """What one promotion rule lets a piece become, per player, in the order stated.

    The order is load-bearing since schema 9: ``promote_on_entry`` promotes a piece
    that arrives without a move to choose in, and it takes the **first** piece listed.
    """
    if "to_by_player" in promotion:
        by_player = promotion["to_by_player"]
        return list(by_player["0"]), list(by_player["1"])
    shared = list(promotion["to"])
    return shared, list(shared)


def validate_or_raise(spec: Any) -> dict[str, Any]:
    """Validate and return the spec, raising :class:`SpecError` if it is invalid.

    The check ``compile_spec`` opens with, and its only caller. It loads rather than
    admits, so it asks ``validate_spec`` with ``new=False``: a stored spec still builds
    its engine. A gate that admits a spec asks ``validate_spec`` itself.
    """
    errors = validate_spec(spec, new=False)
    if errors:
        raise SpecError(errors)
    return spec


# --------------------------------------------------------------------------- board


def _validate_board(board: Any, errors: list[str]) -> tuple[int, int, str]:
    """The board's size, and (schema 14) the lattice its cells sit on."""
    if not isinstance(board, dict):
        errors.append("board must be an object with integer width and height")
        return (0, 0, "square")

    _check_keys(board, _BOARD_KEYS, "board", errors, required={"width", "height"})

    lattice = board.get("lattice", "square")
    if lattice not in LATTICES:
        errors.append(f"board.lattice must be one of {LATTICES}, got {lattice!r}")
        lattice = "square"

    width = board.get("width")
    height = board.get("height")
    ok = True
    for label, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"board.{label} must be an integer")
            ok = False
        elif not 2 <= value <= MAX_BOARD_DIM:
            errors.append(f"board.{label} must be between 2 and {MAX_BOARD_DIM}, got {value}")
            ok = False

    # Schema 16's three refusals, in the order a reader fixes them: the word, the lattice
    # it needs, the frame it fits.
    shape = board.get("shape", "grid")
    if shape not in SHAPES:
        errors.append(f"board.shape must be one of {SHAPES}, got {shape!r}")
    elif shape == "hexagon":
        if lattice != "hex":
            errors.append(
                'board.shape "hexagon" needs board.lattice "hex": a hexagon of cells is '
                "cut from a hex rhombus, and a square lattice has no hexagon to cut"
            )
        if ok and isinstance(width, int) and (width != height or width % 2 == 0):
            errors.append(
                'board.shape "hexagon" needs an odd width equal to its height, got '
                f"{width} by {height}: a hexagon of cells is as tall as it is wide, and "
                "odd: 11 by 11 is six cells a side"
            )
    return (width, height, lattice) if ok else (0, 0, lattice)  # type: ignore[return-value]


def board_shape(spec: Any) -> str:
    """The spec's ``board.shape``, ``"grid"`` when it states none or is malformed."""
    board = spec.get("board") if isinstance(spec, dict) else None
    shape = board.get("shape", "grid") if isinstance(board, dict) else "grid"
    return shape if shape in SHAPES else "grid"


# -------------------------------------------------------------------------- pieces


def _validate_pieces(
    pieces: Any,
    width: int,
    height: int,
    errors: list[str],
    rules: Any = None,
    hex_lattice: bool = False,
) -> list[str]:
    if not isinstance(pieces, list) or not pieces:
        errors.append("pieces must be a non-empty array")
        return []
    if len(pieces) > MAX_PIECE_TYPES:
        errors.append(f"at most {MAX_PIECE_TYPES} piece types, got {len(pieces)}")
    elif len(pieces) + portal_extra_types(rules) > MAX_PIECE_TYPES:
        errors.append(
            f"at most {MAX_PIECE_TYPES} piece types, and rules.portals adds one per end: "
            f"{len(pieces)} declared and {portal_extra_types(rules)} portal ends"
        )

    ids: list[str] = []
    for index, piece in enumerate(pieces):
        where = f"pieces[{index}]"
        if not isinstance(piece, dict):
            errors.append(f"{where} must be an object")
            continue
        _check_keys(piece, _PIECE_KEYS, where, errors, required={"id", "moves"})

        pid = piece.get("id")
        if not isinstance(pid, str) or not pid or len(pid) > 2:
            errors.append(f"{where}.id must be a string of 1-2 characters")
        elif pid in ids:
            errors.append(f"{where}.id {pid!r} is duplicated")
        else:
            ids.append(pid)

        if "name" in piece and not isinstance(piece["name"], str):
            errors.append(f"{where}.name must be a string")
        if "royal" in piece and not isinstance(piece["royal"], bool):
            errors.append(f"{where}.royal must be a boolean")
        value = piece.get("value", 1)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{where}.value must be a number")
        elif not 0 <= value <= 100:
            errors.append(f"{where}.value must be between 0 and 100")

        _validate_squares(piece.get("region"), f"{where}.region", width, height, errors)
        if "promoted_from" in piece and not isinstance(piece["promoted_from"], str):
            errors.append(f"{where}.promoted_from must be a piece id")
        _validate_moves(
            piece.get("moves"),
            where,
            width,
            height,
            errors,
            stocked=piece_has_stock(rules, piece.get("id"))
            or portal_type_needs_no_moves(rules, piece.get("id")),
            hex_lattice=hex_lattice,
        )
        _validate_facing(piece, where, errors, hex_lattice)

    # The capture matrix resolves ids, so it can only be checked once every id is known
    # -- a cycle is the point of the feature and rock has to be able to name scissors
    # whichever order they are declared in.
    for index, piece in enumerate(pieces):
        if isinstance(piece, dict):
            _validate_captures(piece.get("captures"), f"pieces[{index}]", ids, errors)
            _validate_captures(
                piece.get("beats"), f"pieces[{index}]", ids, errors, key="beats"
            )

    return ids


def _validate_facing(
    piece: dict[str, Any], where: str, errors: list[str], hex_lattice: bool = False
) -> None:
    """Which way a piece points, and how many ways there are to point.

    Three refusals, and each is a rule that would otherwise fail somewhere quieter than
    here:

    * **A royal piece may not face.** ``royal_square`` is the hottest query in the
      engine and is memoised against one exact piece value; a royal that can turn is
      four values, on the path the check filter runs once per suspect move. It also
      buys nothing -- check is about what attacks a royal, not about which way it looks
      -- so this is a spec mistake rather than a limitation, and it is said by name.
    * **``initial`` has to be an orientation this piece can be in.** A piece with two
      axes has a forward and a backward and no sideways, and ``"initial": "left"``
      would otherwise compile to a facing it could never turn back to.
    * **A piece that cannot move cannot usefully turn**, which the schema cannot check
      here and the compiler does: a facing is a rotation of the move vectors, so a
      piece with none has four identical readings of nothing.
    """
    facing = piece.get("facing")
    if facing is None:
        return
    if not isinstance(facing, dict):
        errors.append(f"{where}.facing must be an object")
        return
    _check_keys(facing, _FACING_KEYS, f"{where}.facing", errors, required={"axes"})

    axes = facing.get("axes")
    if isinstance(axes, bool) or not isinstance(axes, int) or axes not in FACING_AXES:
        errors.append(
            f"{where}.facing.axes must be one of "
            f"{', '.join(str(a) for a in FACING_AXES)}"
        )
        return
    if hex_lattice and axes == 4:
        errors.append(
            f"{where}.facing.axes 4 turns the piece by quarter turns, and a hex lattice has "
            "none: a quarter turn of a neighbour is not a neighbour. Use 2, a half turn"
        )

    initial = facing.get("initial", "forward")
    if initial not in FACING_NAMES:
        errors.append(
            f"{where}.facing.initial must be one of {', '.join(FACING_NAMES)}"
        )
    elif FACING_NAMES.index(initial) % (len(FACING_NAMES) // axes):
        errors.append(
            f"{where}.facing.initial {initial!r} is not one of the {axes} orientations "
            f"this piece has: with {axes} axes it may only be "
            + " or ".join(
                repr(FACING_NAMES[i])
                for i in range(0, len(FACING_NAMES), len(FACING_NAMES) // axes)
            )
        )

    if "turns" in facing and not isinstance(facing["turns"], bool):
        errors.append(f"{where}.facing.turns must be a boolean")

    if piece.get("royal", False):
        errors.append(
            f"{where}.facing on a royal piece: a royal that can turn is several piece "
            "values rather than one, and the check filter finds a royal by value. "
            "Give the facing to a piece that is not royal."
        )


def _validate_captures(
    captures: Any, where: str, piece_ids: list[str], errors: list[str], key: str = "captures"
) -> None:
    """Which enemy types this piece may take.

    Absent means every type, which is what every ruleset written before schema 3 meant
    and is why the feature moves no hash. An empty list is legal and means the piece
    never captures anything -- a courier, a blocker, a piece that only claims ground --
    and it is distinct from absence in exactly the way that matters.
    """
    if captures is None:
        return
    if not isinstance(captures, list):
        errors.append(f"{where}.{key} must be an array of piece ids")
        return
    for entry in captures:
        if not isinstance(entry, str):
            errors.append(f"{where}.{key} entry {entry!r} must be a piece id")
        elif entry not in piece_ids:
            errors.append(f"{where}.{key} entry {entry!r} is not a declared piece")
    named = [entry for entry in captures if isinstance(entry, str)]
    if len(set(named)) != len(named):
        errors.append(f"{where}.{key} repeats a piece id")


def _validate_squares(
    squares: Any, where: str, width: int, height: int, errors: list[str]
) -> None:
    """A named set of board squares: a palace, a river bank, a promotion zone."""
    if squares is None:
        return
    if not isinstance(squares, list) or not squares:
        errors.append(f"{where} must be a non-empty array of algebraic squares")
        return
    for square in squares:
        if not isinstance(square, str) or parse_square(square, width, height) is None:
            errors.append(f"{where} entry {square!r} is not a square on this board")


def _validate_moves(
    moves: Any,
    where: str,
    width: int,
    height: int,
    errors: list[str],
    stocked: bool = False,
    hex_lattice: bool = False,
) -> None:
    """Every way this piece may move, and (schema 10) the one piece that has none.

    ``stocked`` lifts the non-empty claim, and only for a piece ``rules.drops.stock``
    names -- ``piece_has_stock`` states why the relaxation is gated rather than removed.
    An empty list is still an empty list to everything downstream: the compiler builds no
    movement table for the type, ``_build_attack_sources`` gives it no attack source, and
    the core never generates a move from it. It is placed, it blocks, it can be taken.
    """
    if not isinstance(moves, list) or (not moves and not stocked):
        errors.append(
            f"{where}.moves must be a non-empty array"
            + (
                " unless rules.drops.stock puts the piece in a hand, which is the one "
                "way a piece with no moves can still enter play"
                if isinstance(moves, list)
                else ""
            )
        )
        return
    if len(moves) > MAX_MOVE_DESCRIPTORS:
        errors.append(f"{where}.moves may hold at most {MAX_MOVE_DESCRIPTORS} descriptors")

    for index, move in enumerate(moves):
        loc = f"{where}.moves[{index}]"
        if not isinstance(move, dict):
            errors.append(f"{loc} must be an object")
            continue
        _check_keys(move, _MOVE_KEYS, loc, errors, required={"kind", "vectors"})

        kind = move.get("kind")
        if kind not in MOVE_KINDS:
            errors.append(f"{loc}.kind must be one of {MOVE_KINDS}, got {kind!r}")
        elif hex_lattice and kind == "lame":
            errors.append(
                f"{loc}.kind 'lame' is refused on a hex lattice: its leg is chosen by the "
                "square lattice's longer axis, and a hex leap has two shortest paths and no "
                "stated leg"
            )

        symmetry = move.get("symmetry", "none")
        if symmetry not in SYMMETRIES:
            errors.append(f"{loc}.symmetry must be one of {SYMMETRIES}, got {symmetry!r}")
        else:
            errors.extend(
                _symmetry_refusals(symmetry, move.get("vectors"), f"{loc}.symmetry", hex_lattice)
            )

        mode = move.get("mode", "both")
        if mode not in MODES:
            errors.append(f"{loc}.mode must be one of {MODES}, got {mode!r}")

        if "screen" in move:
            owner = move.get("screen")
            if owner not in SCREEN_OWNERS:
                errors.append(
                    f"{loc}.screen must be one of {SCREEN_OWNERS}, got {owner!r}"
                )
            elif kind != "hop":
                errors.append(f"{loc}.screen only means something on a hop, not a {kind}")

        # A leap is one jump per vector and a lame leap is one blockable jump, so a
        # distance means nothing on either. The compiler read `range` off a leap and
        # never used it, and `fairy/` read the same key and translated it as a
        # limited rider -- two readings of one spec, which is the thing this project
        # refuses to have. Reach further by listing more vectors: `[[1,1]]..[[7,7]]`
        # is a diagonal leaper on a board up to nine wide.
        for distance_key in ("range", "min_range"):
            if distance_key in move and kind in ("leap", "lame"):
                errors.append(
                    f"{loc}.{distance_key} means nothing on a {kind}, which is one jump "
                    "per vector; to reach further, list more vectors"
                )

        rng = move.get("range", DEFAULT_RANGE.get(str(kind), 1))
        if isinstance(rng, bool) or not isinstance(rng, int) or rng < 0:
            errors.append(f"{loc}.range must be a non-negative integer")
        elif rng > MAX_BOARD_DIM:
            errors.append(f"{loc}.range may not exceed {MAX_BOARD_DIM}")

        # min_range exists so a descriptor can cover *only* the far end of a ray --
        # a pawn's double step is "exactly two, path must be clear", not "one or two".
        # Without it every such rule would double-generate its shorter sibling.
        min_range = move.get("min_range", 1)
        if isinstance(min_range, bool) or not isinstance(min_range, int) or min_range < 1:
            errors.append(f"{loc}.min_range must be a positive integer")
        elif isinstance(rng, int) and rng > 0 and min_range > rng:
            errors.append(f"{loc}.min_range ({min_range}) may not exceed range ({rng})")

        # How far past its screen a hop may land. Zero means the rest of the ray, which
        # is the xiangqi cannon; one is the grasshopper, which must land immediately
        # beyond. Meaningless on a slide or a leap, and saying it there is a mistake
        # worth reporting rather than ignoring.
        screen_range = move.get("screen_range", 0)
        if isinstance(screen_range, bool) or not isinstance(screen_range, int) or screen_range < 0:
            errors.append(f"{loc}.screen_range must be a non-negative integer")
        elif screen_range > MAX_BOARD_DIM:
            errors.append(f"{loc}.screen_range may not exceed {MAX_BOARD_DIM}")
        if "screen_range" in move and kind != "hop":
            errors.append(f"{loc}.screen_range only applies to a hop, not to a {kind!r}")

        _validate_vectors(move.get("vectors"), loc, errors)
        _validate_condition(move.get("condition"), loc, width, height, errors)


def _symmetry_refusals(
    symmetry: str, vectors: Any, loc: str, hex_lattice: bool
) -> list[str]:
    """Why a known ``symmetry`` cannot turn these ``vectors`` on this lattice, or nothing.

    Each lattice takes its own turns (schema 18b): the square's quarter turns carry a hex
    neighbour off the hex lattice, and the hexagon's 60-degree turns carry a square step
    off the square one, so either on the other lattice is a plausible wrong game. And a
    hex turn can lengthen a component, as ``[2, 2]`` turned once is ``[-2, 4]``, so every
    turned vector must stay within ``MAX_VECTOR_COMPONENT``: the shorthand says nothing
    its vectors written out under ``none`` could not.
    """
    if symmetry not in (HEX_SYMMETRIES if hex_lattice else SQUARE_SYMMETRIES):
        if hex_lattice:
            return [
                f"{loc} {symmetry!r} expands vectors by the square's symmetries, which "
                "carry some of them off the hex lattice; on a hex board turn them with "
                "'rotate6' or 'dihedral12', or list every vector under 'none'"
            ]
        return [
            f"{loc} {symmetry!r} turns vectors by the hexagon's 60-degree turns, which "
            "only a hex lattice has; on a square board use 'lateral', 'rotate4' or "
            "'dihedral8'"
        ]
    if symmetry not in ("rotate6", "dihedral12") or not isinstance(vectors, list):
        return []
    long = [
        vector
        for vector in vectors
        if isinstance(vector, list)
        and len(vector) == 2
        and all(isinstance(c, int) and not isinstance(c, bool) for c in vector)
        and any(
            abs(c) > MAX_VECTOR_COMPONENT
            for turned in expand_symmetry((vector[0], vector[1]), symmetry)
            for c in turned
        )
    ]
    if not long:
        return []
    return [
        f"{loc} {symmetry!r} turns {', '.join(repr(v) for v in long)} past the vector "
        f"components' +/-{MAX_VECTOR_COMPONENT}; shorten the vector"
    ]


def _validate_vectors(vectors: Any, loc: str, errors: list[str]) -> None:
    if not isinstance(vectors, list) or not vectors:
        errors.append(f"{loc}.vectors must be a non-empty array of [dx, dy] pairs")
        return
    for vector in vectors:
        if (
            not isinstance(vector, list)
            or len(vector) != 2
            or any(isinstance(c, bool) or not isinstance(c, int) for c in vector)
        ):
            errors.append(f"{loc}.vectors entries must be [dx, dy] integer pairs, got {vector!r}")
            continue
        if vector == [0, 0]:
            errors.append(f"{loc}.vectors may not contain the null vector [0, 0]")
        if any(abs(c) > MAX_VECTOR_COMPONENT for c in vector):
            errors.append(
                f"{loc}.vectors components must be within +/-{MAX_VECTOR_COMPONENT}, got {vector!r}"
            )


def _validate_condition(
    condition: Any, loc: str, width: int, height: int, errors: list[str]
) -> None:
    if condition is None:
        return
    if not isinstance(condition, dict):
        errors.append(f"{loc}.condition must be an object")
        return
    _check_keys(condition, _CONDITION_KEYS, f"{loc}.condition", errors)

    if "from_rank" in condition:
        rank = condition["from_rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            errors.append(f"{loc}.condition.from_rank must be a non-negative integer")
    if "first_move" in condition and not isinstance(condition["first_move"], bool):
        errors.append(f"{loc}.condition.first_move must be a boolean")
    _validate_squares(
        condition.get("region"), f"{loc}.condition.region", width, height, errors
    )


# --------------------------------------------------------------------------- setup


def _validate_placement_facing(
    facing: Any, declared: Any, loc: str, errors: list[str]
) -> None:
    """One placement's starting orientation, against what its piece can actually be in."""
    if not isinstance(declared, dict):
        errors.append(
            f"{loc}.facing is set but the piece has no facing to start in; give the "
            "piece a facing first"
        )
        return
    if facing not in FACING_NAMES:
        errors.append(f"{loc}.facing must be one of {', '.join(FACING_NAMES)}")
        return
    axes = declared.get("axes")
    if (
        isinstance(axes, int)
        and not isinstance(axes, bool)
        and axes in FACING_AXES
        and FACING_NAMES.index(facing) % (len(FACING_NAMES) // axes)
    ):
        errors.append(
            f"{loc}.facing {facing!r} is not one of the {axes} orientations this "
            "piece has"
        )


def _validate_setup(
    setup: Any,
    piece_ids: list[str],
    width: int,
    height: int,
    errors: list[str],
    pieces_by_id: list[Any] | None = None,
    hex_lattice: bool = False,
) -> None:
    if not isinstance(setup, dict):
        errors.append("setup must be an object with a placements array")
        return
    _check_keys(setup, _SETUP_KEYS, "setup", errors, required={"placements"})

    if "mirror" in setup and not isinstance(setup["mirror"], bool):
        errors.append("setup.mirror must be a boolean")

    orientation = setup.get("orientation", "mirror")
    if orientation not in ORIENTATIONS:
        errors.append(f"setup.orientation must be one of {ORIENTATIONS}, got {orientation!r}")
    elif hex_lattice and orientation != "rotate":
        errors.append(
            'setup.orientation must be "rotate" on a hex lattice, written out: the rank '
            "reflection sends a neighbour to a cell that is not one, and the half turn is "
            "the hex symmetry that carries one army onto the other"
        )

    placements = setup.get("placements")
    if not isinstance(placements, list) or not placements:
        errors.append("setup.placements must be a non-empty array")
        return

    mirror = setup.get("mirror", True)
    # With a mirror the second army is derived, so only one army is placed; without
    # one the two armies share the board and a square holds one piece whoever's it is.
    seen: dict[tuple[int, str], str] = {}
    for index, placement in enumerate(placements):
        loc = f"setup.placements[{index}]"
        if not isinstance(placement, dict):
            errors.append(f"{loc} must be an object")
            continue
        _check_keys(placement, _PLACEMENT_KEYS, loc, errors, required={"piece", "squares"})

        piece = placement.get("piece")
        if piece not in piece_ids:
            errors.append(f"{loc}.piece {piece!r} is not a declared piece id")

        player = placement.get("player", 0)
        if player not in (0, 1):
            errors.append(f"{loc}.player must be 0 or 1, got {player!r}")
        elif mirror and player != 0:
            errors.append(
                f"{loc}.player must be 0 when setup.mirror is true "
                "(the second player's army is derived by reflection)"
            )

        # Which way these pieces start pointing. On the *placement* rather than only on
        # the piece, because "which way is this one facing" is a fact about a piece on
        # the board in exactly the way its square is -- a game whose pawns all start
        # pointing different ways is not a game with four kinds of pawn in it. The
        # piece's own `facing.initial` is the default and is what a piece takes when it
        # *enters* play later, which the setup has nothing to say about.
        if "facing" in placement:
            declared = next(
                (
                    p.get("facing")
                    for p in (pieces_by_id if isinstance(pieces_by_id, list) else [])
                    if isinstance(p, dict) and p.get("id") == piece
                ),
                None,
            )
            _validate_placement_facing(placement["facing"], declared, loc, errors)

        squares = placement.get("squares")
        if not isinstance(squares, list) or not squares:
            errors.append(f"{loc}.squares must be a non-empty array of algebraic squares")
            continue
        for square in squares:
            if not isinstance(square, str) or parse_square(square, width, height) is None:
                errors.append(f"{loc}.squares entry {square!r} is not a square on this board")
                continue
            key = (player if mirror and isinstance(player, int) else 0, square)
            if key in seen:
                errors.append(f"{loc}.squares {square!r} is already occupied by {seen[key]!r}")
            else:
                seen[key] = str(piece)


def parse_square(square: str, width: int, height: int) -> tuple[int, int] | None:
    """Parse algebraic notation (``a1``, ``h8``) into zero-based ``(file, rank)``.

    Returns ``None`` when the square is malformed or off the board. Files run ``a``
    upward; ranks are 1-based in text and 0-based in the returned tuple.
    """
    if not square or len(square) < 2:
        return None
    file_char, rank_text = square[0], square[1:]
    if not file_char.isalpha() or not rank_text.isdigit():
        return None
    file_index = ord(file_char.lower()) - ord("a")
    rank_index = int(rank_text) - 1
    if not (0 <= file_index < width and 0 <= rank_index < height):
        return None
    return file_index, rank_index


def format_square(file_index: int, rank_index: int) -> str:
    """Inverse of :func:`parse_square`."""
    return f"{chr(ord('a') + file_index)}{rank_index + 1}"


# --------------------------------------------------------------------------- rules


def _validate_rules(
    rules: Any,
    piece_ids: list[str],
    width: int,
    height: int,
    errors: list[str],
    hex_lattice: bool = False,
    shape: str = "grid",
) -> None:
    if not isinstance(rules, dict):
        errors.append("rules must be an object")
        return
    _check_keys(rules, _RULES_KEYS, "rules", errors)
    if hex_lattice:
        _refuse_square_readings(rules, errors)
    _check_landing_symmetries(rules, errors, hex_lattice)

    win = rules.get("win", "checkmate")
    if win not in WIN_CONDITIONS:
        errors.append(f"rules.win must be one of {WIN_CONDITIONS}, got {win!r}")

    stalemate = rules.get("stalemate", "draw")
    if stalemate not in STALEMATE_RESULTS:
        errors.append(f"rules.stalemate must be one of {STALEMATE_RESULTS}, got {stalemate!r}")

    # Both are stated from the point of view of the player it happens *to*, matching
    # rules.stalemate. Losers chess is the reason checkmate needs saying at all.
    checkmate = rules.get("checkmate", "loss")
    if checkmate not in ("loss", "win"):
        errors.append(f"rules.checkmate must be 'loss' or 'win', got {checkmate!r}")

    for key, (low, high) in CLOCK_BOUNDS.items():
        if key not in rules:
            continue
        value = rules[key]
        # Schema 10: only `repetition_draw` may say "never", and only that word. The other
        # two clocks keep their bounds -- `max_plies` is what stops a game at all, and a
        # no-progress clock that never fires is a game the corpus already has under a
        # large integer. A draw by repetition is different in kind: there is no integer
        # that means "this game does not have that rule".
        if key == "repetition_draw" and value == REPETITION_NEVER:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(
                f"rules.{key} must be an integer"
                + (f' or {REPETITION_NEVER!r}' if key == "repetition_draw" else "")
            )
        elif not low <= value <= high:
            errors.append(f"rules.{key} must be between {low} and {high}, got {value}")

    for flag in (
        "en_passant", "must_capture", "forbid_check", "forbid_immobility",
        "promote_on_entry",
    ):
        if flag in rules and not isinstance(rules[flag], bool):
            errors.append(f"rules.{flag} must be a boolean")

    facing = rules.get("royal_facing")
    if facing is not None and facing not in ("allowed", "forbidden"):
        errors.append(
            f"rules.royal_facing must be 'allowed' or 'forbidden', got {facing!r}"
        )

    bare = rules.get("bare_royal")
    if bare is not None and bare not in ENDING_RESULTS:
        errors.append(f"rules.bare_royal must be one of {ENDING_RESULTS}, got {bare!r}")

    limit = rules.get("check_limit")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20
    ):
        errors.append("rules.check_limit must be an integer between 1 and 20")

    _validate_turn(rules.get("turn"), errors)
    _validate_wrap(rules.get("wrap"), errors, shape)
    _validate_obstacles(rules.get("obstacles"), width, height, errors)
    _validate_strike(rules.get("strike"), piece_ids, errors)
    _validate_contacts(rules.get("contacts"), piece_ids, errors, promoting_piece_ids(rules))
    _validate_npcs(rules.get("npcs"), piece_ids, width, height, errors, rules)
    _validate_portals(rules.get("portals"), piece_ids, width, height, errors)
    _validate_territory(rules.get("territory"), width, height, errors)
    _validate_scoring(
        rules.get("scoring"), width, height, errors, bool(rules.get("territory"))
    )
    _validate_drops(rules.get("drops"), piece_ids, width, height, errors)
    _validate_explosion(rules.get("explode_on_capture"), piece_ids, errors)
    _validate_reach(rules.get("reach"), piece_ids, width, height, errors)
    _validate_extinction(rules.get("extinction"), piece_ids, errors)
    promotion = rules.get("promotion")
    if isinstance(promotion, list):
        if not promotion:
            errors.append("rules.promotion may not be an empty array")
        seen: set[Any] = set()
        for index, entry in enumerate(promotion):
            _validate_promotion(
                entry, piece_ids, width, height, errors, f"rules.promotion[{index}]",
                hex_lattice,
            )
            if isinstance(entry, dict):
                piece = entry.get("piece")
                if isinstance(piece, str) and piece in seen:
                    errors.append(
                        f"rules.promotion has two rules for {piece!r}; a piece promotes one way"
                    )
                if isinstance(piece, str):
                    seen.add(piece)
        if len(promotion) > 1 and rules.get("en_passant"):
            errors.append(
                "rules.en_passant needs a single promoting piece to be defined against, "
                f"but rules.promotion names {len(promotion)}"
            )
    else:
        _validate_promotion(
            promotion, piece_ids, width, height, errors, hex_lattice=hex_lattice
        )
    _validate_castling(rules.get("castling"), piece_ids, width, errors)


def _refuse_square_readings(rules: dict[str, Any], errors: list[str]) -> None:
    """The rules whose square-lattice reading is a plausible wrong game on a hex lattice.

    ``reach.mirror``, and a landing's ``symmetry`` (`_check_landing_symmetries`, which
    asks it of both lattices). The pieces' own descriptors, their facing, the setup's
    orientation and a rank-shaped promotion zone are refused where each is validated;
    these two live on rules that are validated without the lattice.
    """
    reach = rules.get("reach")
    if isinstance(reach, dict) and reach.get("mirror", True) not in ("rotate", False):
        errors.append(
            'rules.reach.mirror must be "rotate" or false on a hex lattice: true, the '
            "default, reflects the goal by rank, which is not a hex symmetry"
        )


def _check_landing_symmetries(
    rules: dict[str, Any], errors: list[str], hex_lattice: bool
) -> None:
    """A landing's ``symmetry`` must be its lattice's, as a move's must (schema 18b)."""
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        landing = rule.get("landing") if isinstance(rule, dict) else None
        if not isinstance(landing, dict):
            continue
        symmetry = landing.get("symmetry", "none")
        if symmetry in SYMMETRIES:
            errors.extend(
                _symmetry_refusals(
                    symmetry, landing.get("vectors"), f"rules.{key}.landing.symmetry",
                    hex_lattice,
                )
            )


def _validate_explosion(explosion: Any, piece_ids: list[str], errors: list[str]) -> None:
    """A capture that takes the neighbourhood with it: atomic chess.

    Schema 10 adds ``by`` and ``of``, which narrow *when* the blast happens rather than
    what it reaches. ``by`` is the capturer types whose captures explode; ``of`` is the
    victim types whose being taken explodes; ``immune``, which was already here, is the
    *bystanders* who survive one. The three are three different questions about the same
    event and none of them can be said with either of the others -- which is why `immune`
    covering everything is famously **not** inert (the capturer and the victim go
    regardless) while ``by: []`` is.

    Both default to every type, so a spec written before them means what it always did.
    """
    if explosion is None:
        return
    if not isinstance(explosion, dict):
        errors.append("rules.explode_on_capture must be an object")
        return
    _check_keys(explosion, _EXPLODE_KEYS, "rules.explode_on_capture", errors)

    radius = explosion.get("radius", 1)
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 3:
        errors.append("rules.explode_on_capture.radius must be an integer between 0 and 3")
    elif radius == 0 and explosion.get("immune"):
        # Schema 17 (Q540, kamikaze): a blast of radius 0 reaches no square but the
        # capture's own, and the capturer always goes, so an immune list protects nobody.
        errors.append(
            "rules.explode_on_capture.immune protects pieces standing next to a blast, and "
            "a blast of radius 0 reaches nobody next to it; leave immune out"
        )

    for key in ("immune", "by", "of"):
        listed = explosion.get(key, [])
        if not isinstance(listed, list):
            errors.append(
                f"rules.explode_on_capture.{key} must be an array of piece ids"
            )
            continue
        for piece in listed:
            if piece not in piece_ids:
                errors.append(
                    f"rules.explode_on_capture.{key} entry {piece!r} is not a "
                    "declared piece"
                )


def _validate_drops(
    drops: Any, piece_ids: list[str], width: int, height: int, errors: list[str]
) -> None:
    """Pieces returning to the board from hand: crazyhouse, shogi, and their kin."""
    if drops is None:
        return
    if not isinstance(drops, dict):
        errors.append("rules.drops must be an object")
        return
    _check_keys(drops, _DROPS_KEYS, "rules.drops", errors)

    if "from_captures" in drops and not isinstance(drops["from_captures"], bool):
        errors.append("rules.drops.from_captures must be a boolean")
    if "placement" in drops and not isinstance(drops["placement"], bool):
        errors.append("rules.drops.placement must be a boolean")
    if "either_face" in drops and not isinstance(drops["either_face"], bool):
        errors.append("rules.drops.either_face must be a boolean")

    # Shogi's two restrictions on where a pawn may be dropped. Both are about the
    # pawn specifically, but neither is about pawns in principle, so both are lists.
    for key in ("no_doubled", "no_mate"):
        entry = drops.get(key)
        if entry is None:
            continue
        if not isinstance(entry, list):
            errors.append(f"rules.drops.{key} must be an array of piece ids")
            continue
        for piece in entry:
            if piece not in piece_ids:
                errors.append(f"rules.drops.{key} entry {piece!r} is not a declared piece")

    _validate_stock(drops.get("stock"), piece_ids, errors)

    region = drops.get("region")
    if region is None:
        return
    if not isinstance(region, dict):
        errors.append("rules.drops.region must be an object keyed by piece id")
        return
    for piece, squares in region.items():
        if piece not in piece_ids:
            errors.append(f"rules.drops.region key {piece!r} is not a declared piece")
        _validate_squares(squares, f"rules.drops.region[{piece!r}]", width, height, errors)


def _validate_stock(stock: Any, piece_ids: list[str], errors: list[str]) -> None:
    """What each player holds before anything has been captured (schema 10, Q51).

    Both players get the same stock, for the reason ``setup.mirror`` exists: a supply one
    player has and the other does not is a first-seat effect the index would mark down
    for a reason that is not about the rule.

    ``"unlimited"`` is a supply that never runs out and is the point of the key: the DSL
    could not previously say *a stone*, because a hand fills only from captures and
    nothing is ever captured that was never on the board. A number is the other half and
    is free -- the hand already counts down.
    """
    if stock is None:
        return
    if not isinstance(stock, dict) or not stock:
        errors.append(
            "rules.drops.stock must be a non-empty object keyed by piece id, whose "
            f"values are counts of 1 to {MAX_STOCK} or {STOCK_UNLIMITED!r}"
        )
        return
    for piece, held in stock.items():
        if piece not in piece_ids:
            errors.append(f"rules.drops.stock key {piece!r} is not a declared piece")
        if held == STOCK_UNLIMITED:
            continue
        if isinstance(held, bool) or not isinstance(held, int):
            errors.append(
                f"rules.drops.stock[{piece!r}] must be an integer or {STOCK_UNLIMITED!r}"
            )
        elif not 1 <= held <= MAX_STOCK:
            errors.append(
                f"rules.drops.stock[{piece!r}] must be between 1 and {MAX_STOCK}, "
                f"got {held}"
            )


def _validate_reach(
    reach: Any, piece_ids: list[str], width: int, height: int, errors: list[str]
) -> None:
    """A win by arrival: king of the hill, racing kings, breakthrough."""
    if reach is None:
        return
    if not isinstance(reach, dict):
        errors.append("rules.reach must be an object")
        return
    _check_keys(reach, _REACH_KEYS, "rules.reach", errors, required={"squares"})

    piece = reach.get("piece")
    if piece is not None and piece not in piece_ids:
        errors.append(f"rules.reach.piece {piece!r} is not a declared piece")

    mirror = reach.get("mirror")
    if mirror is not None and not isinstance(mirror, bool) and mirror != "rotate":
        errors.append("rules.reach.mirror must be a boolean or \"rotate\"")
    if "safe" in reach and not isinstance(reach.get("safe"), bool):
        errors.append("rules.reach.safe must be a boolean")
    # Schema 18 (Q786): win by leaving a square. ``safe`` asks whether the piece standing
    # on the goal could be taken there, and under ``leave`` the piece that decides the
    # game has just gone, so the two are refused together rather than given a reading.
    on = reach.get("on", "arrive")
    if on not in REACH_ON:
        errors.append(f"rules.reach.on must be one of {REACH_ON}, got {on!r}")
    elif on == "leave" and reach.get("safe") is True:
        errors.append(
            'rules.reach.safe asks whether the piece on the goal square could be taken '
            'there, and under "on": "leave" the piece that decides the game has just left '
            "it; leave safe out"
        )
    result = reach.get("result", "win")
    if result not in ENDING_RESULTS:
        errors.append(f"rules.reach.result must be one of {ENDING_RESULTS}, got {result!r}")

    squares = reach.get("squares")
    if not isinstance(squares, list) or not squares:
        errors.append("rules.reach.squares must be a non-empty array of algebraic squares")
        return
    for square in squares:
        if not isinstance(square, str) or parse_square(square, width, height) is None:
            errors.append(f"rules.reach.squares entry {square!r} is not a square on this board")


def _validate_extinction(extinction: Any, piece_ids: list[str], errors: list[str]) -> None:
    """A loss by running out: kinglet's pawns, three-kings' kings, the horde's army."""
    if extinction is None:
        return
    if not isinstance(extinction, dict):
        errors.append("rules.extinction must be an object")
        return
    _check_keys(extinction, _EXTINCTION_KEYS, "rules.extinction", errors, required={"pieces"})

    pieces = extinction.get("pieces")
    if pieces == "any":
        pass  # the horde: any piece at all, so the army as a whole
    elif not isinstance(pieces, list) or not pieces:
        errors.append("rules.extinction.pieces must be 'any' or a non-empty array of piece ids")
    else:
        for piece in pieces:
            if piece not in piece_ids:
                errors.append(f"rules.extinction.pieces entry {piece!r} is not a declared piece")

    count = extinction.get("min_count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
        errors.append("rules.extinction.min_count must be an integer between 1 and 64")
    result = extinction.get("result", "loss")
    if result not in ENDING_RESULTS:
        errors.append(f"rules.extinction.result must be one of {ENDING_RESULTS}, got {result!r}")

    # Schema 18 (Q783): each listed type counted on its own. Two spellings would state
    # nothing, and are refused as `_check_inert_rules` refuses a rule with nothing to act
    # on: over ``any`` there is no list to count apart, and over one type the count apart
    # is the count together, so the spec with the key and the spec without it would be
    # one game under two hashes.
    #
    # Schema 27 (Q936): ``"own"`` counts each apart too, and only against a side that
    # starts with the kind; `_check_extinction_coherence` holds the readings that need
    # the opening board. Over ``any`` and over one type it states nothing, as ``true``.
    per_type = extinction.get("per_type", False)
    if not isinstance(per_type, bool) and per_type != EXTINCTION_PER_TYPE_OWN:
        errors.append('rules.extinction.per_type must be true, false or "own"')
    elif per_type and pieces == "any":
        errors.append(
            "rules.extinction.per_type counts each listed type on its own, and 'any' lists "
            "no types; name the types, or drop per_type"
        )
    elif per_type and isinstance(pieces, list) and len(_str_set(pieces)) < 2:
        errors.append(
            "rules.extinction.per_type counts each listed type on its own, and with one "
            "type listed that is the count the rule already makes; list two or more "
            "types, or drop per_type"
        )


def _validate_wrap(wrap: Any, errors: list[str], shape: str = "grid") -> None:
    """A board whose opposite edges are the same edge.

    Two axes, each stated either once for both players or once per player. The pair of
    keys is the ``to`` / ``to_by_player`` idiom ``rules.promotion`` already uses, and it
    is here for the same reason: the symmetric case is the common one and deserves one
    word, while the asymmetric case has to be sayable at all.

    Edge names are absolute -- the first player at the bottom, file ``a`` on the left --
    so ``{"0": "left", "1": "right"}`` says exactly what it looks like. Naming them
    relative to each player's own forward direction would have made two identical strings
    mean two different seams.

    **A hexagon (schema 16) joins three pairs of sides instead of two axes**, `n_s`,
    `ne_sw` and `nw_se`, spelled the same way, with the compass names of a flat-topped
    hex map. Each shape's keys are refused on the other: `files` on a hexagon would have
    to glue by a vector that is not its rectangle meaning, a second reading of one key,
    and a rectangle has no third pair. There is no relative ``"forward"`` value, for the
    reason above (the user, 2026-09-28).
    """
    if wrap is None:
        return
    if not isinstance(wrap, dict):
        errors.append("rules.wrap must be an object")
        return
    _check_keys(wrap, _WRAP_KEYS, "rules.wrap", errors)

    own = {key for axis, _ in wrap_axes(shape) for key in (axis, f"{axis}_by_player")}
    foreign = sorted(key for key in wrap if key in _WRAP_KEYS and key not in own)
    if foreign and shape == "hexagon":
        errors.append(
            f"rules.wrap.{', rules.wrap.'.join(foreign)} joins a rectangle's edges, and "
            "this board is a hexagon: join its sides with n_s, ne_sw and nw_se"
        )
    elif foreign:
        errors.append(
            f"rules.wrap.{', rules.wrap.'.join(foreign)} joins a hexagon's sides, and this "
            'board is not one (board.shape "hexagon"): join its edges with files and ranks'
        )

    for axis, allowed in wrap_axes(shape):
        by_player = f"{axis}_by_player"
        if axis in wrap and by_player in wrap:
            errors.append(
                f"rules.wrap has both {axis!r} and {by_player!r}; use one or the other"
            )
        if axis in wrap and wrap[axis] not in allowed:
            errors.append(
                f"rules.wrap.{axis} must be one of {allowed}, got {wrap[axis]!r}"
            )
        if by_player not in wrap:
            continue
        value = wrap[by_player]
        if not isinstance(value, dict) or set(value) != {"0", "1"}:
            errors.append(
                f"rules.wrap.{by_player} must be an object keyed by '0' and '1'"
            )
            continue
        for player in ("0", "1"):
            if value[player] not in allowed:
                errors.append(
                    f"rules.wrap.{by_player}[{player!r}] must be one of {allowed}, "
                    f"got {value[player]!r}"
                )


def reflect_square(
    file: int, rank: int, width: int, height: int, orientation: str
) -> tuple[int, int]:
    """The second player's image of a square, under this setup orientation.

    The one rule the whole spec reflects by: a mirror flips the rank, a rotation flips
    both. Shared with ``engine.compile`` so that a named square means the same thing to
    the validator and to the compiler.
    """
    if orientation == "rotate":
        return width - 1 - file, height - 1 - rank
    return file, height - 1 - rank


def reach_reflection(reach: Any) -> str | None:
    """How ``reach.squares`` is carried to the second player, as a ``reflect_square`` name.

    ``reach.mirror`` names its reflection itself and never reads ``setup.orientation``:
    ``true`` is the rank reflection, ``"rotate"`` the half turn, ``false`` none at all.
    Schema 7 added ``"rotate"`` because `intransitive` wanted a rotated goal under a
    mirrored army. Shared with ``engine.compile`` so the validator and the compiler put the
    second player's goal on the same squares.
    """
    if not isinstance(reach, dict):
        return None
    mirror = reach.get("mirror", True)
    if mirror == "rotate":
        return "rotate"
    return "mirror" if mirror else None


def obstacle_squares(
    rules: Any, width: int, height: int, orientation: str = "mirror"
) -> frozenset[int]:
    """Which squares are walls, as square indices. Empty when there are none.

    An obstacle belongs to the *board*, not to a player, so the resolved set is one set
    that both players see. ``mirror`` therefore means "and their reflections" rather
    than "a different set for the second player": it is shorthand for writing a
    symmetric wall out, not a way to give one player a wall the other does not have.
    That is why it defaults to false where ``setup.mirror`` and ``reach.mirror`` default
    to true -- those describe an army, and an army has a side.

    Malformed names are skipped rather than raised on, because the validator calls this
    while it is still collecting the errors about them.
    """
    if not isinstance(rules, dict):
        return frozenset()
    obstacles = rules.get("obstacles")
    if not isinstance(obstacles, dict):
        return frozenset()
    names = obstacles.get("squares")
    if not isinstance(names, list):
        return frozenset()
    mirror = obstacles.get("mirror", False)
    squares: set[int] = set()
    for name in names:
        if not isinstance(name, str):
            continue
        coords = parse_square(name, width, height)
        if coords is None:
            continue
        file, rank = coords
        squares.add(rank * width + file)
        if mirror is True:
            other_file, other_rank = reflect_square(
                file, rank, width, height, orientation
            )
            squares.add(other_rank * width + other_file)
    return frozenset(squares)


def _validate_obstacles(
    obstacles: Any, width: int, height: int, errors: list[str]
) -> None:
    """Squares nothing may occupy.

    Entirely static, which is the whole reason the rule is cheap: every question it
    raises -- does this ray end here, can this leap land there -- is answered once at
    compile time and never again.
    """
    if obstacles is None:
        return
    if not isinstance(obstacles, dict):
        errors.append("rules.obstacles must be an object")
        return
    _check_keys(
        obstacles, _OBSTACLES_KEYS, "rules.obstacles", errors, required={"squares"}
    )

    for flag in ("mirror", "blocks_leaps"):
        if flag in obstacles and not isinstance(obstacles[flag], bool):
            errors.append(f"rules.obstacles.{flag} must be a boolean")

    _validate_squares(
        obstacles.get("squares"), "rules.obstacles.squares", width, height, errors
    )


def piece_has_stock(rules: Any, piece_id: Any) -> bool:
    """Whether ``rules.drops.stock`` puts this type in both players' hands (schema 10).

    **The predicate the moveless relaxation is gated on**, and it is gated for the reason
    Q29's is: a relaxation has no key and no value, so it cannot be indexed, and removing
    the restriction outright would let *every* spec ever written declare a piece that can
    never move -- schema 1 included, since a version gate only stops an old stamp reaching
    a new *feature*. Multiple royals took exactly this shape in schema 2
    (``royalty_can_multiply``), and this is that shape again with the restriction being
    ``pieces[].moves must be a non-empty array``.

    The condition is *this piece can be placed without moving and never runs out of
    reasons to exist*, not *this piece is useless*: a stocked stone is placed, it blocks,
    it can be taken, and it is a game. A moveless piece with no stock is a piece that
    stands where it was put and can do nothing for the rest of the game, and that stays
    refused because it is almost always a mistake rather than a design.
    """
    if not isinstance(rules, dict):
        return False
    drops = rules.get("drops")
    if not isinstance(drops, dict):
        return False
    stock = drops.get("stock")
    return isinstance(stock, dict) and _in(piece_id, stock)


def npc_immune_attackers(
    rules: Any, piece_ids: Iterable[str], npc: Any
) -> set[str]:
    """Which declared types ``npc``'s ``immune`` list actually takes the wild type from.

    The one reading of schema 10's ``npcs[].immune``, shared by the validator -- which
    refuses a list that clears nothing -- and by ``engine.compile``, which does the
    clearing. Shared for the reason ``npc_routes`` and ``obstacle_squares`` are: two
    readers of one sentence is how a nested key ends up meaning two things.

    Empty when the key is absent, which is what makes the feature inert. Wild types are
    exempt: a wild piece eats whatever it steps onto, and this immunity is a rule about
    the players.

    Malformed entries are read permissively rather than raised on, because the validator
    calls this while it is still collecting the errors about them.
    """
    if not isinstance(npc, dict) or not isinstance(npc.get("immune"), list):
        return set()
    npcs = (rules or {}).get("npcs") if isinstance(rules, dict) else None
    wild = _str_set(
        entry.get("piece")
        for entry in (npcs if isinstance(npcs, list) else [])
        if isinstance(entry, dict)
    )
    named = {pid for pid in npc["immune"] if isinstance(pid, str)}
    return {pid for pid in piece_ids if pid not in named and pid not in wild}


def npc_routes(
    rules: Any, width: int, height: int, orientation: str = "mirror"
) -> list[tuple[str, tuple[int, ...]]]:
    """Every wild piece's route, as square indices, mirrors expanded.

    One entry per wild piece that will stand on the board -- a mirrored entry yields two,
    the second labelled as the mirror and walking the reflected squares in the same
    order. Shared with ``engine.compile`` so that a route means the same thing to the
    validator and to the compiler, exactly as ``obstacle_squares`` is.

    Malformed entries are skipped rather than raised on, because the validator calls
    this while it is still collecting the errors about them.
    """
    if not isinstance(rules, dict):
        return []
    npcs = rules.get("npcs")
    if not isinstance(npcs, list):
        return []
    routes: list[tuple[str, tuple[int, ...]]] = []
    for index, npc in enumerate(npcs):
        if not isinstance(npc, dict) or not isinstance(npc.get("route"), list):
            continue
        squares: list[int] = []
        mirrored: list[int] = []
        for name in npc["route"]:
            coords = parse_square(name, width, height) if isinstance(name, str) else None
            if coords is None:
                continue
            file, rank = coords
            squares.append(rank * width + file)
            other_file, other_rank = reflect_square(file, rank, width, height, orientation)
            mirrored.append(other_rank * width + other_file)
        if not squares:
            continue
        routes.append((f"rules.npcs[{index}]", tuple(squares)))
        if npc.get("mirror") is True:
            routes.append((f"rules.npcs[{index}] (mirrored)", tuple(mirrored)))
    return routes


def npc_starts(
    rules: Any, width: int, height: int, orientation: str = "mirror"
) -> list[tuple[int, str, int]]:
    """Where every wild piece stands at ply zero: ``(entry index, label, square)``.

    Per entry, the route's first square and then each ``also_at`` square (schema 23) in
    the order written, which the validator holds to route order; a mirrored entry's starts
    follow its original's, each reflected as its route is. With no ``also_at`` this is
    one start per route of ``npc_routes``, in the same order, which is what keeps every
    earlier game's opening position where it was. Shared with ``engine.compile`` for the
    reason ``npc_routes`` is.

    An ``also_at`` square that is malformed or not on its route is skipped rather than
    raised on: the validator is still collecting the errors about it.
    """
    if not isinstance(rules, dict):
        return []
    npcs = rules.get("npcs")
    if not isinstance(npcs, list):
        return []
    starts: list[tuple[int, str, int]] = []
    for index, npc in enumerate(npcs):
        if not isinstance(npc, dict) or not isinstance(npc.get("route"), list):
            continue
        route = [name for name in npc["route"] if isinstance(name, str)]
        named = [route[0]] if route else []
        also = npc.get("also_at")
        if isinstance(also, list):
            named += [name for name in also if isinstance(name, str) and name in route[1:]]
        own: list[tuple[str, int]] = []
        mirrored: list[tuple[str, int]] = []
        for position, name in enumerate(named):
            coords = parse_square(name, width, height)
            if coords is None:
                continue
            file, rank = coords
            other_file, other_rank = reflect_square(file, rank, width, height, orientation)
            suffix = "" if position == 0 else f".also_at {name}"
            own.append((f"rules.npcs[{index}]{suffix}", rank * width + file))
            mirrored.append(
                (f"rules.npcs[{index}]{suffix} (mirrored)", other_rank * width + other_file)
            )
        starts += [(index, where, square) for where, square in own]
        if npc.get("mirror") is True:
            starts += [(index, where, square) for where, square in mirrored]
    return starts


def _validate_npc_also_at(npc: dict[str, Any], loc: str, errors: list[str]) -> None:
    """Schema 23: more wild pieces of the entry's kind on the entry's own route.

    Squares of the route other than its first (which is the entry's own piece's start),
    each once, in route order -- so a game has one spelling: the extras share the entry's
    type, bounty and immunity by construction, and listing them any other way, or
    rotating the route so a different piece is "first", would say the same game twice.
    Each piece walks the route in step with the others; the core steps them front first
    (``docs/reference/engine-and-spec-record.md``, schema 23).
    """
    if "also_at" not in npc:
        return
    also = npc["also_at"]
    route = npc.get("route")
    if not isinstance(also, list) or not also:
        errors.append(f"{loc}.also_at must be a non-empty array of squares on its route")
        return
    if not isinstance(route, list):
        return
    names = [name for name in route if isinstance(name, str)]
    positions: list[int] = []
    for name in also:
        if not isinstance(name, str):
            errors.append(f"{loc}.also_at entry {name!r} is not a square")
        elif names and name == names[0]:
            errors.append(
                f"{loc}.also_at names {name}, the route's first square, where the entry's "
                "own wild piece starts; list only the others"
            )
        elif name not in names:
            errors.append(
                f"{loc}.also_at names {name}, which is not on its route; a further wild "
                "piece walks the entry's own route"
            )
        else:
            positions.append(names.index(name))
    repeated = sorted({name for name in also if isinstance(name, str) and also.count(name) > 1})
    if repeated:
        errors.append(
            f"{loc}.also_at names {', '.join(repeated)} twice; one wild piece stands on a "
            "square"
        )
    elif positions != sorted(positions):
        in_order = [names[position] for position in sorted(positions)]
        errors.append(
            f"{loc}.also_at must list its squares in route order ({', '.join(in_order)}), "
            "so the game has one spelling"
        )


def _validate_npcs(
    npcs: Any,
    piece_ids: list[str],
    width: int,
    height: int,
    errors: list[str],
    rules: dict[str, Any] | None = None,
) -> None:
    """Pieces neither player owns, each walking a route of its own (schema 9)."""
    if npcs is None:
        return
    if not isinstance(npcs, list) or not npcs:
        errors.append("rules.npcs must be a non-empty array of wild pieces")
        return
    for index, npc in enumerate(npcs):
        loc = f"rules.npcs[{index}]"
        if not isinstance(npc, dict):
            errors.append(f"{loc} must be an object with 'piece' and 'route'")
            continue
        _check_keys(npc, _NPC_KEYS, loc, errors, required={"piece", "route"})

        piece = npc.get("piece")
        if piece not in piece_ids:
            errors.append(f"{loc}.piece {piece!r} is not a declared piece")

        route = npc.get("route")
        _validate_squares(route, f"{loc}.route", width, height, errors)
        if isinstance(route, list):
            repeated = sorted(
                {name for name in route if isinstance(name, str) and route.count(name) > 1}
            )
            if repeated:
                errors.append(
                    f"{loc}.route visits {', '.join(repeated)} twice; a route is a cycle "
                    "of distinct squares, because which square is next has to be a fact "
                    "about the square"
                )

        bounty = npc.get("bounty")
        if bounty is not None and (
            isinstance(bounty, bool) or not isinstance(bounty, (int, float)) or bounty < 0
        ):
            errors.append(f"{loc}.bounty must be a number of 0 or more")

        if "mirror" in npc and not isinstance(npc["mirror"], bool):
            errors.append(f"{loc}.mirror must be a boolean")

        _validate_npc_also_at(npc, loc, errors)

        _validate_npc_immune(npc, loc, rules or {}, piece_ids, errors)

    _check_one_immunity_per_type(npcs, errors)


def _validate_portals(
    portals: Any,
    piece_ids: list[str],
    width: int,
    height: int,
    errors: list[str],
) -> None:
    """The shape of ``rules.portals`` (schemas 19 and 21): groups, each a pair or a map.

    What may stand beside portals, where they may stand and what ``carries`` may name is
    ``_check_portal_coherence``'s.
    """
    if portals is None:
        return
    if not isinstance(portals, list) or not portals:
        errors.append("rules.portals must be a non-empty array of portal groups")
        return
    if len(portals) > MAX_PORTAL_GROUPS:
        errors.append(
            f"rules.portals holds {len(portals)} groups; at most {MAX_PORTAL_GROUPS}, because "
            "a group's number and colour are shown on its portals"
        )
    n_ends = sum(
        len(group["ends"]) if isinstance(group.get("ends"), list) else 2
        for group in portals if isinstance(group, dict)
    )
    if n_ends > MAX_PORTAL_ENDS:
        errors.append(
            f"rules.portals has {n_ends} ends; at most {MAX_PORTAL_ENDS} across every group"
        )
    for index, group in enumerate(portals):
        loc = f"rules.portals[{index}]"
        if not isinstance(group, dict):
            errors.append(
                f"{loc} must be an object with 'piece' and either 'orange' and 'blue' or 'ends'"
            )
            continue
        piece = group.get("piece")
        if piece not in piece_ids:
            errors.append(f"{loc}.piece {piece!r} is not a declared piece")
        if "no_straight_back" in group and not isinstance(group["no_straight_back"], bool):
            errors.append(f"{loc}.no_straight_back must be true or false")
        if "kind" in group:
            if group["kind"] not in PORTAL_KINDS:
                errors.append(f"{loc}.kind must be \"fold\", got {group['kind']!r}")
            elif group["kind"] == "momentum":
                errors.append(
                    f"{loc}.kind \"momentum\" is the default: drop it, so the group has one "
                    "spelling and the game one hash"
                )
        if "ends" in group:
            _validate_portal_ends(group, loc, width, height, errors)
        else:
            _validate_portal_pair(group, loc, width, height, errors)


def _players_problem(players: Any) -> bool:
    """Whether a list of players is malformed: not a list of distinct 0s and 1s."""
    return (
        not isinstance(players, list)
        or any(p not in (0, 1) or isinstance(p, bool) for p in players)
        or len(set(players)) != len(players)
    )


def _carries_problem(carries: Any) -> str | None:
    """What is wrong with the shape of one end's ``carries``, or ``None``."""
    if not isinstance(carries, list) or not all(isinstance(pid, str) for pid in carries):
        return "must be a list of piece ids"
    if not carries:
        return (
            "is empty, so no piece may go in by this end; say that with who may enter "
            "(passage 'closed' on a pair, no 'to' on an end of a map) instead"
        )
    if len(set(carries)) != len(carries):
        return "names a piece twice"
    return None


def _validate_portal_pair(
    pair: dict[str, Any], loc: str, width: int, height: int, errors: list[str],
) -> None:
    """A group written as a pair: ``orange`` and ``blue`` (schema 19), ``carries`` (21).

    A missing end is said as a square that is not on the board, as a blank one is, so a
    pair half written reads the same however it is half written.
    """
    _check_keys(pair, _PORTAL_KEYS, loc, errors, required={"piece"})
    for end in PORTAL_ENDS:
        name = pair.get(end)
        if not isinstance(name, str) or parse_square(name, width, height) is None:
            errors.append(f"{loc}.{end} must be a square on the board, got {name!r}")
    if (
        pair.get("orange") == pair.get("blue")
        and isinstance(pair.get("orange"), str)
        and parse_square(pair["orange"], width, height) is not None
    ):
        errors.append(f"{loc} puts both ends on {pair['orange']}; a pair needs two squares")
    moved_by = pair.get("moved_by")
    if moved_by is not None:
        if not isinstance(moved_by, dict):
            errors.append(f"{loc}.moved_by must be an object keyed by 'orange' and 'blue'")
        else:
            for key in moved_by:
                if key not in PORTAL_ENDS:
                    errors.append(
                        f"{loc}.moved_by has unknown key {key!r}; it names the ends "
                        "'orange' and 'blue'"
                    )
            for end, players in moved_by.items():
                if end in PORTAL_ENDS and _players_problem(players):
                    errors.append(
                        f"{loc}.moved_by.{end} must be a list of distinct players, "
                        "drawn from 0 and 1 ([] for nobody)"
                    )
    passage = pair.get("passage")
    if passage is not None:
        if not isinstance(passage, dict):
            errors.append(f"{loc}.passage must be an object keyed by player, '0' and '1'")
        else:
            for key, word in passage.items():
                if key not in ("0", "1"):
                    errors.append(
                        f"{loc}.passage has unknown key {key!r}; it is keyed by player, "
                        "'0' and '1'"
                    )
                elif word not in PORTAL_PASSAGES:
                    errors.append(
                        f"{loc}.passage.{key} must be one of {PORTAL_PASSAGES}, got "
                        f"{word!r}"
                    )
    carries = pair.get("carries")
    if carries is not None:
        if not isinstance(carries, dict) or not carries:
            errors.append(
                f"{loc}.carries must be a non-empty object keyed by 'orange' and 'blue'"
            )
        else:
            for end, listed in carries.items():
                if end not in PORTAL_ENDS:
                    errors.append(
                        f"{loc}.carries has unknown key {end!r}; it names the ends "
                        "'orange' and 'blue'"
                    )
                elif (problem := _carries_problem(listed)) is not None:
                    errors.append(f"{loc}.carries.{end} {problem}")


def _validate_portal_ends(
    group: dict[str, Any], loc: str, width: int, height: int, errors: list[str],
) -> None:
    """A group written with ``ends`` (schema 21, `docs/archive/plan-portals-2.md` §2.1).

    Refused: a destination outside the group or itself; an end nobody can use (no ``to``,
    and no end leads to it); ``enter`` on an end that leads nowhere, or naming nobody (the
    same end as one with no ``to``); and a group of two ends, which is always a pair --
    two that lead to each other, or one way, which a pair's ``passage`` says -- so a pair
    has one spelling and one hash.
    """
    _check_keys(group, _PORTAL_ENDS_KEYS, loc, errors, required={"piece", "ends"})
    ends = group.get("ends")
    if not isinstance(ends, list) or not all(isinstance(end, dict) for end in ends):
        errors.append(f"{loc}.ends must be a list of ends, each an object with 'at'")
        return
    if len(ends) < 3:
        errors.append(
            f"{loc}.ends has {len(ends)} ends; two ends are a pair, written with 'orange' "
            "and 'blue' (and 'passage' for one way), so a group of ends has three or more"
        )
    led_to = {
        end.get("to") for end in ends
        if isinstance(end.get("to"), int) and not isinstance(end.get("to"), bool)
    }
    seen: dict[str, int] = {}
    for place, end in enumerate(ends):
        where = f"{loc}.ends[{place}]"
        _check_keys(end, _PORTAL_END_KEYS, where, errors, required={"at"})
        name = end.get("at")
        if not isinstance(name, str) or parse_square(name, width, height) is None:
            errors.append(f"{where}.at must be a square on the board, got {name!r}")
        elif name in seen:
            errors.append(
                f"{where}.at {name} is also {loc}.ends[{seen[name]}]; a group's ends need "
                "squares of their own"
            )
        else:
            seen[name] = place
        to = end.get("to")
        if "to" in end:
            if not isinstance(to, int) or isinstance(to, bool) or not 0 <= to < len(ends):
                errors.append(
                    f"{where}.to must be the index of another end of this group, 0 to "
                    f"{len(ends) - 1}, got {to!r}; a destination is always in its own group"
                )
            elif to == place:
                errors.append(f"{where}.to is the end itself; an end leads to another end")
        elif place not in led_to:
            errors.append(
                f"{where} leads nowhere and no end leads to it, so nobody can use it; give "
                "it a 'to' or lead another end to it"
            )
        if "enter" in end:
            if _players_problem(end["enter"]):
                errors.append(
                    f"{where}.enter must be a list of distinct players, drawn from 0 and 1"
                )
            elif "to" not in end:
                errors.append(
                    f"{where}.enter says who may go in, but the end leads nowhere (no 'to'); "
                    "drop it"
                )
            elif not end["enter"]:
                errors.append(
                    f"{where}.enter is empty, so nobody goes in by it: that is an end with no "
                    "'to', a way out only; write it that way"
                )
        if "moved_by" in end and _players_problem(end["moved_by"]):
            errors.append(
                f"{where}.moved_by must be a list of distinct players, drawn from 0 and 1 "
                "([] for nobody)"
            )
        if "carries" in end:
            if (problem := _carries_problem(end["carries"])) is not None:
                errors.append(f"{where}.carries {problem}")
            elif "to" not in end:
                errors.append(
                    f"{where}.carries says which pieces may go in, but the end leads nowhere "
                    "(no 'to'); drop it"
                )


def _check_portal_coherence(
    spec: dict[str, Any], pieces: list[Any], rules: dict[str, Any], errors: list[str]
) -> None:
    """What may stand beside ``rules.portals``, and where the portals may stand (schema 19).

    `docs/archive/plan-portals.md` §1. A portal type is placed by the rule and by nothing else,
    is nobody's material, never captures and never turns: so it may not be placed by the
    setup, promote or be promoted into, be held, be a goal's piece or a counted one, be
    named by a capture list, be royal, or be worth anything. Its moves are a player's
    *steps onto empty squares* (`mode: "move"`, which also keeps a portal off another
    portal) and read the same for both players, because an end either player may move has
    no forward of its own.

    The rules refused beside it in the first build are each a reading nobody has asked for:
    a hex lattice (a sign does not name one of six neighbours), wild pieces (a route is a
    static table and a portal moves), a sweep, a strike or contact table and an explosion
    (what a capture leaves behind at an exit is a second landing rule), territory (which
    square a transit paints), and a confined piece (a region an exit could leave).
    """
    if "portals" not in rules or not isinstance(rules.get("portals"), list):
        return
    groups = portal_groups(rules)
    board = spec.get("board")
    width = board.get("width") if isinstance(board, dict) else None
    height = board.get("height") if isinstance(board, dict) else None
    if not isinstance(width, int) or not isinstance(height, int):
        return
    setup: dict[str, Any] = spec["setup"] if isinstance(spec.get("setup"), dict) else {}
    orientation = str(setup.get("orientation", "mirror"))

    if isinstance(board, dict) and board.get("lattice") == "hex":
        errors.append(
            "rules.portals is refused on a hex lattice in this schema: the direction a piece "
            "comes out in is read from the sign of its vector, which on six neighbours does "
            "not name one of them"
        )
    for key, why in (
        ("npcs", "a wild piece walks a fixed route and a portal moves"),
        ("strike", "what a capture leaves behind beside an exit is a second landing rule"),
        ("contacts", "what a capture leaves behind beside an exit is a second landing rule"),
        ("explode_on_capture", "a blast beside an exit is a second landing rule"),
        ("territory", "which square a transit colours is not settled"),
    ):
        if key in rules:
            errors.append(f"rules.portals is refused beside rules.{key}: {why}")
    if uses_sweep(spec):
        errors.append(
            "rules.portals is refused beside a 'sweep' move: what a sweep through a portal "
            "removes is not settled"
        )

    declared = {
        piece["id"]: piece
        for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    for pid, piece in declared.items():
        if "region" in piece:
            errors.append(
                f"piece {pid!r} has a region, which rules.portals refuses in this schema: an "
                "exit could put the piece outside it"
            )
    clashing = sorted(set(portal_end_ids(groups)) & set(declared))
    if clashing:
        errors.append(
            f"piece id {', '.join(repr(c) for c in clashing)} is the name of a portal end "
            "beside rules.portals; choose another id"
        )

    portal_ids = portal_piece_ids(rules)
    for pid in sorted(portal_ids & set(declared)):
        _check_portal_piece(pid, declared[pid], rules, orientation, errors)
        _check_portal_type_used_nowhere_else(spec, pid, pieces, rules, errors)
    _check_fold_pieces(groups, declared, errors)

    # The squares: on the board, not a wall or an absent cell, empty in the setup, and no
    # two ends on one square.
    walls = obstacle_squares(rules, width, height, orientation) | absent_squares(board)
    occupied: set[tuple[int, int]] = set()
    for pid in declared:
        occupied |= {(f, r) for _, f, r in _placed_squares(setup, pid, width, height, orientation)}
    seen: dict[str, str] = {}
    for group in groups:
        for end in group["ends"]:
            name = end["at"]
            coords = parse_square(name, width, height) if isinstance(name, str) else None
            if coords is None:
                continue
            loc = f"rules.portals[{group['index']}].{end['key']}"
            file, rank = coords
            if rank * width + file in walls:
                errors.append(f"{loc} {name} is a wall or not on the board's shape")
            # Schema 22 (§3.2 rule 10): a fold is occupiable, so it may start under a piece.
            if (file, rank) in occupied and group["kind"] != "fold":
                errors.append(f"{loc} {name} is a square setup.placements puts a piece on")
            if name in seen:
                errors.append(f"{loc} {name} is also {seen[name]}; two portals on one square")
            seen.setdefault(name, loc)
    # Schema 20: a bar on going straight back needs somebody to move an end, or it is one
    # game under two hashes (`_check_inert_rules`' argument).
    for group in groups:
        if group["no_straight_back"] and not any(end["moved_by"] for end in group["ends"]):
            errors.append(
                f"rules.portals[{group['index']}].no_straight_back is true but nobody may "
                "move any end of the group, so there is never a move to bar; drop it or set "
                "moved_by"
            )
    _check_portal_carries(groups, declared, errors)


def _refuse_a_one_square_lame_leap(spec: dict[str, Any], errors: list[str]) -> None:
    """Q884: a ``lame`` vector of one square (``[1, 0]``, ``[0, 1]``, ``[1, 1]``, any signs).

    A lame leap is blocked by a piece on the square it passes over (`compile._lame_path`),
    and a one-square step passes over none: its leg is its own target. The compiler drops
    such a move (``block in (target, square)``), so a spec that wrote one got a piece
    silently lacking it. The xiangqi horse's ``[1, 2]`` and elephant's ``[2, 2]`` have a leg
    and are untouched.

    It runs only on a new spec (``validate_spec``'s ``new``), like
    ``_refuse_a_portal_bar_no_move_meets``: the move was never played, so a stored spec that
    carries one plays exactly as it always did and still builds its engine.
    """
    pieces = spec.get("pieces")
    for p_index, piece in enumerate(pieces if isinstance(pieces, list) else ()):
        moves = piece.get("moves") if isinstance(piece, dict) else None
        for m_index, move in enumerate(moves if isinstance(moves, list) else ()):
            vectors = move.get("vectors") if isinstance(move, dict) else None
            if not isinstance(move, dict) or move.get("kind") != "lame":
                continue
            for v_index, vector in enumerate(vectors if isinstance(vectors, list) else ()):
                if (
                    isinstance(vector, list) and len(vector) == 2
                    and all(isinstance(c, int) and not isinstance(c, bool) for c in vector)
                    and vector != [0, 0] and max(abs(vector[0]), abs(vector[1])) == 1
                ):
                    errors.append(
                        f"pieces[{p_index}].moves[{m_index}].vectors[{v_index}] {vector} is a "
                        "one-square 'lame' leap, which has no square to be blocked on; use an "
                        "ordinary 'leap' or a 'slide' of range 1 instead"
                    )


def _refuse_a_portal_bar_no_move_meets(spec: dict[str, Any], errors: list[str]) -> None:
    """Q879: every ``no_straight_back`` some end of its group is moved by, asked whether a
    move can ever meet it (`_check_portal_bar_can_bind`).

    It runs only on a new spec (``validate_spec``'s ``new``), as
    ``_refuse_a_promotion_rank_past_the_board`` does and for its reason: the bar never
    removes a move, so a spec that already carries one plays as it always did, and a game
    stored before the refusal -- a visitor's published page, its match replays, a corpus
    entry -- still builds its engine. A group nobody moves is `_check_portal_coherence`'s,
    refused always, as it was from schema 20.
    """
    for refusals in _portal_bar_refusals(spec):
        errors.extend(refusals or ())


def portal_bars_that_can_bind(spec: dict[str, Any]) -> tuple[bool, ...]:
    """Per portal group, in `portal_groups` order: whether its ``no_straight_back`` can ever
    bar a move -- false for a group without one, and for one `_check_portal_bar_can_bind`
    finds no move can meet.

    Q883: the compiler's reading of the bar (`engine.compile._compile_portals`), so a bar
    the core keeps and keys is exactly one this refusal admits. A stored spec built with
    ``new=False`` may carry an inert one; it compiles to no bar at all, so two positions
    alike in everything that affects play get one key.
    """
    return tuple(bool(refusals == []) for refusals in _portal_bar_refusals(spec))


def _portal_bar_refusals(spec: dict[str, Any]) -> list[list[str] | None]:
    """Per portal group: ``None`` without a bar some end is moved by, else what
    `_check_portal_bar_can_bind` says against it (empty when a move can meet it)."""
    rules = spec.get("rules")
    if not isinstance(rules, dict) or not isinstance(rules.get("portals"), list):
        return []
    groups = portal_groups(rules)
    pieces = spec.get("pieces")
    declared = {
        piece["id"]: piece
        for piece in (pieces if isinstance(pieces, list) else ())
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    setup = spec.get("setup")
    orientation = str(setup.get("orientation", "mirror")) if isinstance(setup, dict) else "mirror"
    out: list[list[str] | None] = []
    for group in groups:
        if group["no_straight_back"] and any(end["moved_by"] for end in group["ends"]):
            refusals: list[str] = []
            _check_portal_bar_can_bind(group, groups, declared, rules, orientation, refusals)
            out.append(refusals)
        else:
            out.append(None)
    return out


def _check_portal_bar_can_bind(
    group: dict[str, Any], groups: list[dict[str, Any]], declared: dict[str, Any],
    rules: dict[str, Any], orientation: str, errors: list[str],
) -> None:
    """A ``no_straight_back`` some end of its group is moved by, but no move can meet.

    The bar binds one move only: the move straight after an end of the group was moved,
    and only that end, back onto the square it left. So it can ever refuse a move only
    where (a) whoever moves next may move the same end -- the other player, unless a turn
    is longer than one move -- and (b) the end's own moves can undo one another. Both are
    read off the spec: (a) from each end's ``moved_by`` and ``rules.turn``, (b) from the
    group piece's directions. Each move a portal makes goes a whole number of its vector's
    steps (a leap, a slide, a hop or a lame leap alike), so on a board with no wrap a move
    back needs a direction and its opposite; that clause is not asked round a wrap (where
    two steps one way can come back) or beside a fold (a move through one leaves the line).

    Without the bar the game is the same game: the bar never removes a move. Before Q883 the
    core still keyed the position by the bar it set, so a draw by repetition could read one
    position as two; the compiler now reads this same check (`portal_bars_that_can_bind`)
    and hands such a bar to the core as none, and the core keeps a bar only when the
    player to move next may move that end. Sluice's draft
    (Q844, `docs/reference/portal-showcase-ideas.md`) carried one: each gate moved by the
    player it threatens alone.
    """
    loc = f"rules.portals[{group['index']}].no_straight_back is true but"
    movers = [end for end in group["ends"] if end["moved_by"]]
    if not turn_is_long(rules) and all(len(end["moved_by"]) < 2 for end in movers):
        each = ", ".join(
            f"{end['key']} only by player {next(iter(end['moved_by']))}" for end in movers
        )
        errors.append(
            f"{loc} it could never bar a move: it binds only the very next move, which is "
            f"always the other player's, and each end that moves is moved by one player "
            f"only ({each}); drop it, or let both players move an end"
        )
        return
    if _wrap_joins_an_edge(rules.get("wrap")) or any(g["kind"] == "fold" for g in groups):
        return
    piece = declared.get(group["piece"]) if isinstance(group["piece"], str) else None
    moves = piece.get("moves") if isinstance(piece, dict) else None
    directions: set[tuple[int, int]] = set()
    for move in moves if isinstance(moves, list) else ():
        vectors = move.get("vectors") if isinstance(move, dict) else None
        symmetry = move.get("symmetry", "none") if isinstance(move, dict) else None
        if not isinstance(vectors, list) or symmetry not in SQUARE_SYMMETRIES:
            return  # refused by name elsewhere; nothing to read a direction from
        for vector in vectors:
            if not (
                isinstance(vector, list) and len(vector) == 2
                and all(isinstance(c, int) and not isinstance(c, bool) for c in vector)
                and vector != [0, 0]
            ):
                continue
            for dx, dy in expand_symmetry((vector[0], vector[1]), symmetry):
                step = gcd(dx, dy)
                directions.add((dx // step, dy // step))
                # The second player's reading, whichever way the setup turns it.
                flipped = (-dx, -dy) if orientation == "rotate" else (dx, -dy)
                directions.add((flipped[0] // step, flipped[1] // step))
    if directions and not any((-dx, -dy) in directions for dx, dy in directions):
        errors.append(
            f"{loc} it could never bar a move: piece {group['piece']!r} has no move in the "
            "opposite direction to another, so an end it moves can never go straight back "
            "onto the square it left; drop it, or give the piece a move back"
        )


def _check_fold_pieces(
    groups: list[dict[str, Any]], declared: dict[str, Any], errors: list[str],
) -> None:
    """Schema 22: what a fold's own piece may not say (`docs/archive/plan-portals-2.md` §3.2).

    A fold stands on no square of the board byte, so it has no unmoved state of its own:
    the square's ``unmoved`` bit belongs to whatever piece stands in it. A ``first_move``
    condition on a fold's moves would read the wrong thing, so it is refused.
    """
    for group in groups:
        if group["kind"] != "fold" or not _in(group["piece"], declared):
            continue
        moves = declared[group["piece"]].get("moves")
        for index, move in enumerate(moves if isinstance(moves, list) else ()):
            condition = move.get("condition") if isinstance(move, dict) else None
            if isinstance(condition, dict) and "first_move" in condition:
                errors.append(
                    f"piece {group['piece']!r} moves[{index}] may not have a first_move "
                    f"condition: rules.portals[{group['index']}] is a fold, which stands on "
                    "no square of its own and so has no first move"
                )


def _check_portal_carries(
    groups: list[dict[str, Any]], declared: dict[str, Any], errors: list[str],
) -> None:
    """Schema 21: what an end's ``carries`` may name (`docs/archive/plan-portals-2.md` §2.3).

    Declared pieces that are not portals, royals included. Refused: a filter on an end
    nobody may go in by (it states nothing), and one naming every piece that can move (it
    lets in what no filter would). A promoted piece is filtered by what it is now, which is
    the compiler's reading and needs nothing here.
    """
    portal_ids = _str_set(group["piece"] for group in groups)
    movers = {pid for pid, piece in declared.items() if piece.get("moves")} - portal_ids
    for group in groups:
        for end in group["ends"]:
            carries = end["carries"]
            if carries is None:
                continue
            group_loc = f"rules.portals[{group['index']}]"
            loc = (
                f"{group_loc}.{end['key']}.carries" if group["spelling"] == "ends"
                else f"{group_loc}.carries.{end['key']}"
            )
            for pid in sorted(carries):
                if pid not in declared:
                    errors.append(f"{loc} names {pid!r}, which is not a declared piece")
                elif pid in portal_ids:
                    errors.append(f"{loc} names {pid!r}, a portal; a portal never goes in")
            if not end["enter"]:
                errors.append(
                    f"{loc} says which pieces may go in, but nobody may go in by this end; "
                    "drop it"
                )
            elif movers and movers <= carries:
                errors.append(
                    f"{loc} names every piece that can move, so it lets in what no filter "
                    "would; drop it"
                )


def _check_portal_piece(
    pid: str, piece: dict[str, Any], rules: dict[str, Any], orientation: str,
    errors: list[str],
) -> None:
    """One portal type's own declaration: worth nothing, never royal, symmetric steps."""
    where = f"piece {pid!r} (a portal, rules.portals)"
    if piece.get("royal"):
        errors.append(f"{where} may not be royal")
    value = piece.get("value", 1)
    if value != 0:
        errors.append(f"{where} must have value 0: a portal is nobody's material")
    for key in ("facing", "captures", "promoted_from"):
        if key in piece:
            errors.append(f"{where} may not declare {key}")
    moves = piece.get("moves")
    if not isinstance(moves, list):
        return
    for index, move in enumerate(moves):
        if not isinstance(move, dict):
            continue
        loc = f"{where} moves[{index}]"
        if move.get("mode", "both") != "move":
            errors.append(
                f"{loc} must say mode \"move\": a portal moves only onto an empty square, "
                "so never onto a piece or another portal"
            )
        if move.get("kind") == "sweep":
            errors.append(f"{loc} may not sweep")
        condition = move.get("condition")
        if isinstance(condition, dict) and ({"from_rank", "region"} & set(condition)):
            errors.append(
                f"{loc} may not have a from_rank or region condition: those are read from "
                "a player's side, and a portal belongs to neither"
            )
        vectors = move.get("vectors")
        symmetry = move.get("symmetry", "none")
        if not isinstance(vectors, list) or symmetry not in SQUARE_SYMMETRIES:
            continue
        directions: set[tuple[int, int]] = set()
        for vector in vectors:
            if (
                isinstance(vector, list) and len(vector) == 2
                and all(isinstance(c, int) and not isinstance(c, bool) for c in vector)
            ):
                directions |= expand_symmetry((vector[0], vector[1]), symmetry)
        flipped = {
            (-dx, -dy) if orientation == "rotate" else (dx, -dy) for dx, dy in directions
        }
        if flipped != directions:
            errors.append(
                f"{loc} reads differently for the two players: a portal has no forward of "
                "its own, so its moves must be the same seen from either side (use a "
                "symmetry that covers both, such as rotate4 or dihedral8)"
            )
    if not moves and not portal_type_needs_no_moves(rules, pid):
        errors.append(
            f"{where} has no moves, but rules.portals lets a player move one of its ends"
        )


def _check_portal_type_used_nowhere_else(
    spec: dict[str, Any], pid: str, pieces: list[Any], rules: dict[str, Any],
    errors: list[str],
) -> None:
    """A portal type is placed by ``rules.portals`` and named by no other rule."""
    where = f"piece {pid!r} is a portal (rules.portals)"
    setup = spec.get("setup")
    placements = setup.get("placements") if isinstance(setup, dict) else None
    if isinstance(placements, list) and any(
        isinstance(p, dict) and p.get("piece") == pid for p in placements
    ):
        errors.append(f"{where} and may not be placed by setup.placements")
    for entry in _promotion_entries(rules):
        if not isinstance(entry, dict):
            continue
        if entry.get("piece") == pid or pid in _promotion_target_ids(entry):
            errors.append(f"{where} and may not promote or be promoted into")
    drops = rules.get("drops")
    if isinstance(drops, dict):
        named = {
            key: drops.get(key) for key in ("stock", "region", "no_doubled", "no_mate")
        }
        if any(
            (isinstance(value, (dict, list)) and pid in value) for value in named.values()
        ):
            errors.append(f"{where} and may not be named by rules.drops: it is never held")
    reach = rules.get("reach")
    if isinstance(reach, dict) and reach.get("piece") == pid:
        errors.append(f"{where} and may not be rules.reach's piece")
    extinction = rules.get("extinction")
    if isinstance(extinction, dict) and pid in (extinction.get("pieces") or []):
        errors.append(f"{where} and may not be counted by rules.extinction")
    castling = rules.get("castling")
    if isinstance(castling, dict) and pid in (castling.get("king"), castling.get("rook")):
        errors.append(f"{where} and may not castle")
    for piece in pieces:
        if not isinstance(piece, dict):
            continue
        if isinstance(piece.get("captures"), list) and pid in piece["captures"]:
            errors.append(
                f"piece {piece.get('id')!r} lists {pid!r} in captures, but a portal is "
                "never captured"
            )
        if piece.get("promoted_from") == pid:
            errors.append(f"piece {piece.get('id')!r} is promoted_from {pid!r}, a portal")


def _check_one_immunity_per_type(npcs: list[Any], errors: list[str]) -> None:
    """Two wild pieces of one type may not disagree about who can take them.

    ``capture_allowed`` is a matrix over *types*, so an immunity is a fact about the type
    and there is one row for it however many wild pieces walk around. Two entries of the
    same type with different ``immune`` lists would silently compile to the intersection
    -- only an attacker both lists allow could take either -- which is a game nobody
    wrote, and the permissive direction is the one that fails quietly.

    ``wolves`` reaches this and passes: its two routes carry the same type and, in the
    schema-10 re-spelling, the same list. A designer who wants two hazards with different
    immunities wants two types, which the schema already offers.
    """
    stated: dict[str, frozenset[str] | None] = {}
    disagree: set[str] = set()
    for npc in npcs:
        if not isinstance(npc, dict) or not isinstance(npc.get("piece"), str):
            continue
        immune = npc.get("immune")
        here = (
            frozenset(pid for pid in immune if isinstance(pid, str))
            if isinstance(immune, list)
            else None
        )
        piece = npc["piece"]
        if piece in stated and stated[piece] != here:
            disagree.add(piece)
        stated[piece] = here
    for piece in sorted(disagree):
        errors.append(
            f"rules.npcs has two wild {piece!r} with different 'immune' lists; capture "
            "permission is a fact about a piece type, so one type has one immunity. "
            "Declare a second type if the two hazards differ."
        )


def _validate_npc_immune(
    npc: dict[str, Any],
    loc: str,
    rules: dict[str, Any],
    piece_ids: list[str],
    errors: list[str],
) -> None:
    """Schema 10: the types that may still take this wild piece, and nothing else may.

    Three refusals, and each is decidable from the spec alone:

    * a list of ids, and every id declared -- the ``captures`` shape;
    * a list that takes the wild type away from *nobody* is the key doing nothing, which
      is Q57's rule (``_check_inert_rules``) applied one level down: a declaration with no
      effect is the same game under a second hash, and ``rule_mass`` charges for ``npcs``
      whatever it compiles to (under ``v10`` that deflated ``rule_yield`` rather than
      merely failing to help);
    * and the one Q84 asked to be decided out loud: ``capture_allowed`` is keyed by type
      rather than by owner, so a wild type that can also reach a player's *hand* would
      carry its immunity there. Refused rather than read owner-blind.
    """
    immune = npc.get("immune")
    if immune is None:
        return
    if not isinstance(immune, list):
        errors.append(f"{loc}.immune must be an array of piece ids")
        return
    for pid in immune:
        if pid not in piece_ids:
            errors.append(f"{loc}.immune entry {pid!r} is not a declared piece")
    if any(pid not in piece_ids for pid in immune):
        return

    if not npc_immune_attackers(rules, piece_ids, npc):
        errors.append(
            f"{loc}.immune names every type that could take this wild piece, so it "
            "protects it from nothing; leave the key out"
        )

    drops = rules.get("drops")
    piece = npc.get("piece")
    if isinstance(drops, dict) and drops and isinstance(piece, str):
        # The two routes into a hand, and only those two. `drops.region` says *where* a
        # held piece may go and is not a way of getting one, so it is deliberately not
        # here -- a spec whose only `drops` content is a region for some other type puts
        # this type in nobody's hand and is not refused.
        in_hand = bool(drops.get("from_captures", True)) or piece in (
            drops.get("stock") or {}
        )
        if in_hand:
            errors.append(
                f"{loc}.immune beside rules.drops: {piece!r} can reach a player's hand, "
                "and capture permission is a fact about a piece type rather than about "
                "who owns it, so the immunity would follow it out of the hand. Say it "
                "with per-piece 'captures' lists if that is the game you mean."
            )


# The piece-naming fields of one link, required and optional, for each spelling.
_STRIKE_LINK_ROLES = {
    "strike": (("from", "to"), ()),
    "contacts": (("mover", "onto"), ("mover_to", "onto_to")),
}


def strike_survives_anything(rule: Any, key: str, piece_ids: Sequence[str]) -> bool:
    """Whether ``rules[key]`` ever leaves a struck piece on the board (Q271).

    ``key`` is ``"strike"`` or ``"contacts"``. This is the compiler's own answer,
    ``StrikeRule.survives_anything``, asked of the rule as ``compile_spec`` compiles it.
    The ``advance`` and ``choose`` refusals used to read the spec a second time, and that
    reading counted a link from an immune piece as a survivor, although a strike never
    reaches one (Q95's find). So a rule whose every link was immune validated under
    ``choose`` with nothing it could ever place, and was refused under ``advance`` for a
    survivor that could never exist. The compiler drops such a rule, and the two
    refusals now ask it rather than restate it.

    This is the question about a *survivor*, which is what those two refusals are about.
    Whether the rule is there at all is `strike_changes_anything`, and for a contacts
    table the two differ.
    """
    compiled = _compiled_strike(rule, key, piece_ids)
    return compiled is not None and compiled.survives_anything


def strike_changes_anything(rule: Any, key: str, piece_ids: Sequence[str]) -> bool:
    """Whether the compiler keeps ``rules[key]`` at all, rather than plain capture (Q417).

    ``key`` is ``"strike"`` or ``"contacts"``, and the answer is the compiler's whole one:
    ``_compile_strike`` or ``_compile_contacts`` returning a rule rather than ``None``. A
    strike is kept when something survives, so for a strike this is
    `strike_survives_anything`. A contacts table is kept when something survives *or*
    only the attacker changes (``survives_anything or changes_mover``), so a table whose
    links all say ``mover_to`` alone is a rule with no survivor. A table whose every
    link lands on an immune target is neither, since a contact never reaches one, and is
    plain capture written the long way.

    This is the question a reader asks whose answer is "is capture still capture here",
    the grid's ``taking`` axis and ``distance``'s contact features. Before it, they had
    only `strike_survives_anything`, which answers half of it for a table.
    """
    return _compiled_strike(rule, key, piece_ids) is not None


def _compiled_strike(rule: Any, key: str, piece_ids: Sequence[str]) -> StrikeRule | None:
    """``rules[key]`` as ``compile_spec`` compiles it, or ``None`` where it drops it.

    **Why this may borrow from the compiler when ``royalty_can_multiply`` may not.** That
    question is defined over a whole compiled spec, which only ``compile_spec`` makes, and
    ``compile_spec`` opens with the validator. This one is a leaf. ``_compile_strike`` and
    ``_compile_contacts`` read one rule and a map of piece ids and never validate, so
    nothing recurses. The import waits for the call, when both modules are loaded, so
    there is no cycle. And the compiler's contract, a validated input and no defensive
    checks, is kept by handing it only the links that name declared pieces. A link that
    names anything else is refused by name elsewhere and leaves nothing standing here.
    """
    from chess_variant_sandbox.engine.compile import _compile_contacts, _compile_strike

    if not isinstance(rule, dict) or key not in _STRIKE_LINK_ROLES:
        return None
    declared = set(piece_ids)

    def names_declared(piece: Any) -> bool:
        return isinstance(piece, str) and piece in declared

    required, optional = _STRIKE_LINK_ROLES[key]
    listed = "chain" if key == "strike" else "links"
    links = rule.get(listed)
    immune = rule.get("immune")
    switching = rule.get("switches_sides")
    readable = {
        **rule,
        "immune": [p for p in immune if names_declared(p)] if isinstance(immune, list) else [],
        "switches_sides": (
            [p for p in switching if names_declared(p)] if isinstance(switching, list) else []
        ),
        listed: [
            link
            for link in (links if isinstance(links, list) else [])
            if isinstance(link, dict)
            and all(names_declared(link.get(role)) for role in required)
            and all(names_declared(link[role]) for role in optional if role in link)
        ],
    }
    index_of = {piece: index for index, piece in enumerate(piece_ids)}
    compile_rule = _compile_strike if key == "strike" else _compile_contacts
    return compile_rule(readable, index_of, len(piece_ids))


def _validate_strike(strike: Any, piece_ids: list[str], errors: list[str]) -> None:
    """What happens instead of removal when a piece is taken.

    The shape first, then, on a strike whose shape is sound, whether it says anything
    (Q538). Asked only of a well-formed rule, so that refusal is a fact about the rule
    rather than a consequence of a typo the shape check already named.
    """
    before = len(errors)
    _validate_strike_shape(strike, piece_ids, errors)
    if len(errors) == before:
        errors.extend(strike_states_nothing(strike, "strike", piece_ids))


def strike_states_nothing(strike: Any, key: str, piece_ids: Sequence[str]) -> list[str]:
    """The refusal for a strike that states a rule the core plays as plain capture (Q538).

    ``key`` is ``"strike"`` or ``"contacts"``: the table is the strike's other spelling
    and had the same gap: a table whose every link lands on an immune target validated
    under ``stay``, ``swap`` and ``advance`` and played as if it were absent.

    Empty for a strike that is kept, for the bare spelling, and for anything that is not
    an object. The compiler drops a strike whose ``effect`` is ``"capture"`` and one whose
    every link starts at an immune piece (``_compile_strike``), and the core then plays
    an ordinary capture: the victim is removed and the attacker moves onto its square.
    ``position``, ``chain`` and ``immune`` are all sentences about a *struck piece that
    survives* -- where it goes, what it becomes, who is spared the change -- so beside no
    survivor each is a rule stated and never played. ``rifle-b`` was the find
    (``docs/reference/variant-expressibility-2026-09.md``, *Found along the way* 1):
    ``{"effect": "capture", "position": "stay"}`` reads as rifle chess, and ``Pe4xd5``
    left the pawn on d5, under a hash that differs from chess's.

    **The bare spelling stays legal**, which is ``_check_inert_rules``' first boundary:
    ``{"effect": "capture"}``, with ``position: "advance"`` or an empty ``immune`` spelled
    out, is the default said aloud and is the standing proof that schema 6 was an inert
    bump. What is refused is a strike that *says more than that* and is played as that.
    ``choose`` is left to its own refusal in ``_validate_strike_shape``, which is this
    one's older half. Public because ``transform.settle_inert_rules`` drops what this
    refuses, asking this rather than a copy.
    """
    if not isinstance(strike, dict) or not strike:
        return []
    position = strike.get("position", "advance")
    if position == "choose" or strike_changes_anything(strike, key, piece_ids):
        return []
    if key == "contacts":
        if not strike.get("links"):
            return []  # no table: the shape check's refusal, not this one
        return [
            "rules.contacts lands every link on a piece in rules.contacts.immune, and a "
            f"contact never reaches an immune target, so no link ever fires: {PLAYED_AS_CAPTURE}. "
            "Take a link's 'onto' out of immune, or leave rules.contacts out"
        ]
    effect = strike.get("effect", "capture")
    if effect != "capture":
        return [
            "rules.strike.chain starts every link at a piece in rules.strike.immune, and "
            f"a strike never reaches an immune piece, so the {effect} chain never fires: "
            f"{PLAYED_AS_CAPTURE}. Take a link's 'from' out of immune, or leave "
            "rules.strike out"
        ]
    stated = []
    if position != "advance":
        stated.append(f"position {position!r}")
    if strike.get("chain") is not None:
        stated.append("a chain")
    if strike.get("immune"):
        stated.append("an immune list")
    if not stated:
        return []
    return [
        f"rules.strike has effect 'capture' and {' and '.join(stated)}, and position, "
        "chain and immune are all about a struck piece that survives, which under "
        f"'capture' none does: {PLAYED_AS_CAPTURE}. Under 'capture' only position "
        "'stay' (the attacker keeps its square: rifle chess) and 'home' (the taken piece "
        "is reborn on its home square: circe) say something. Use one of those, give the "
        "strike effect 'demote' or 'promote' and a chain, or leave rules.strike out"
    ]


# The clause every `strike_states_nothing` refusal ends on, and what a test looks for.
PLAYED_AS_CAPTURE = (
    "the rule changes nothing, and every capture is an ordinary one, the victim removed "
    "and the attacker moving onto its square"
)


def _validate_strike_shape(strike: Any, piece_ids: list[str], errors: list[str]) -> None:
    """The strike's keys, values and chain, and the refusals that read its position."""
    if strike is None:
        return
    if not isinstance(strike, dict):
        errors.append("rules.strike must be an object")
        return
    _check_keys(strike, _STRIKE_KEYS, "rules.strike", errors)

    effect = strike.get("effect", "capture")
    if effect not in STRIKE_EFFECTS:
        errors.append(
            f"rules.strike.effect must be one of {STRIKE_EFFECTS}, got {effect!r}"
        )

    position = strike.get("position", "advance")
    if position not in STRIKE_POSITIONS:
        errors.append(
            f"rules.strike.position must be one of {STRIKE_POSITIONS}, got {position!r}"
        )
    _validate_landing(strike, "strike", errors)

    # Schema 17 (Q540): the attacker changes sides on a capture (Andernach).
    switching = strike.get("switches_sides")
    if switching is not None:
        if not isinstance(switching, list) or not switching:
            errors.append(
                "rules.strike.switches_sides must be a non-empty array of piece ids: the "
                "pieces that change sides when they capture"
            )
        else:
            for piece in switching:
                if piece not in piece_ids:
                    errors.append(
                        f"rules.strike.switches_sides entry {piece!r} is not a declared piece"
                    )
            if len(set(map(str, switching))) != len(switching):
                errors.append("rules.strike.switches_sides names a piece twice")

    immune = strike.get("immune", [])
    if not isinstance(immune, list):
        errors.append("rules.strike.immune must be an array of piece ids")
    else:
        for piece in immune:
            if piece not in piece_ids:
                errors.append(
                    f"rules.strike.immune entry {piece!r} is not a declared piece"
                )

    # A list of links rather than a map from piece id to piece id, and the reason is the
    # wire. ``llm.client.wire_schema`` cannot express an object whose keys are not known
    # in advance -- it *deletes* the property rather than weakening it -- so a chain
    # written as a map would be a rule the proposer could never reach, which this
    # project counts as a rule that does not exist. There is no "null" link either: a
    # piece the chain does not mention is already removed when it is struck, so a link
    # pointing at nothing would be a second spelling of absence.
    chain = strike.get("chain")
    if chain is None:
        if effect != "capture":
            errors.append(
                f"rules.strike.effect is {effect!r} but no chain says where a struck "
                "piece goes"
            )
        return
    if not isinstance(chain, list) or not chain:
        errors.append("rules.strike.chain must be a non-empty array of from/to links")
        return

    links: dict[str, str] = {}
    for index, link in enumerate(chain):
        loc = f"rules.strike.chain[{index}]"
        if not isinstance(link, dict):
            errors.append(f"{loc} must be an object with 'from' and 'to'")
            continue
        _check_keys(link, _STRIKE_LINK_KEYS, loc, errors, required={"from", "to"})
        source, target = link.get("from"), link.get("to")
        for role, value in (("from", source), ("to", target)):
            if value not in piece_ids:
                errors.append(f"{loc}.{role} {value!r} is not a declared piece")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if source == target:
            errors.append(
                f"{loc} points {source!r} at itself, which is a strike that changes "
                "nothing"
            )
        if source in links:
            errors.append(
                f"rules.strike.chain has two links from {source!r}; a struck piece "
                "becomes one thing"
            )
        links[source] = target

    # One question decides both refusals, and it is the compiler's: a link from an immune
    # piece keeps nothing alive, because a strike never reaches that piece (Q271).
    if position == "advance" and strike_survives_anything(strike, "strike", piece_ids):
        survivors = [
            source
            for source, target in sorted(links.items())
            if strike_survives_anything(
                {**strike, "chain": [{"from": source, "to": target}]}, "strike", piece_ids
            )
        ]
        errors.append(
            "rules.strike.position 'advance' has nowhere to put a struck piece that "
            f"survives ({', '.join(survivors)}); the attacker takes its square, so "
            "use 'stay', 'swap' or 'choose'"
        )
    if position == "choose" and not strike_survives_anything(strike, "strike", piece_ids):
        errors.append(
            "rules.strike.position 'choose' needs a chain that keeps a struck piece "
            "alive, and a strike never reaches a piece in rules.strike.immune; with "
            "nothing to place it is 'advance' written the long way"
        )

    # A descending chain has to terminate. An ascending one need not -- a cycle under
    # ``promote`` is a legal and interesting game, where taking the top piece hands the
    # defender the bottom one -- but a cycle under ``demote`` is an infinite ladder, and
    # material would never leave the board.
    if effect == "demote":
        for start in links:
            seen: set[str] = set()
            node: str | None = start
            while node is not None and node in links:
                if node in seen:
                    errors.append(
                        f"rules.strike.chain cycles at {node!r}; a demotion chain has "
                        "to end, or material never leaves the board"
                    )
                    break
                seen.add(node)
                node = links[node]


def _validate_contacts(
    contacts: Any,
    piece_ids: list[str],
    errors: list[str],
    promoting: AbstractSet[str] = frozenset(),
) -> None:
    """The contact table: what two pieces that meet both become (schema 10, Q19a).

    ``strike`` says what the *victim* becomes and says it keyed by the victim alone.
    This says what *both* become, keyed by the pair -- so a link is a sentence about one
    meeting, and the table is every meeting the designer wants to be about more than
    removal. Everything else a strike carries is gone: there is no ``effect``, because a
    table that names both new types has already said whether this is a promotion or a
    demotion and saying it twice is a second place the rule could be stated differently.

    There is no cycle check, and its absence is deliberate rather than an omission. A
    demotion *chain* had to terminate because it is a ladder a piece walks down without
    anything else changing; a contact needs a capture-move each time, and the
    no-progress clock reads the board's declared material, so a table that conserves
    material never resets it and a repetition of one is caught by the clock that already
    exists.

    The shape first, then, on a table whose shape is sound, whether it changes anything
    (Q538, ``strike_states_nothing``), as for the strike.
    """
    before = len(errors)
    _validate_contacts_shape(contacts, piece_ids, errors, promoting)
    if len(errors) == before:
        errors.extend(strike_states_nothing(contacts, "contacts", piece_ids))


def _validate_contacts_shape(
    contacts: Any,
    piece_ids: list[str],
    errors: list[str],
    promoting: AbstractSet[str] = frozenset(),
) -> None:
    """The table's keys, values and links, and the refusals that read its position.

    ``promoting`` is the pieces with a promotion rule (``promoting_piece_ids``): a link that
    keeps such a mover as it is overrides that promotion, and is schema 18's keeping link
    rather than a link that changes nothing.
    """
    if contacts is None:
        return
    if not isinstance(contacts, dict):
        errors.append("rules.contacts must be an object")
        return
    _check_keys(contacts, _CONTACTS_KEYS, "rules.contacts", errors)

    position = contacts.get("position", "advance")
    if position not in CONTACT_POSITIONS:
        errors.append(
            f"rules.contacts.position must be one of {CONTACT_POSITIONS}, got {position!r}"
        )
    _validate_landing(contacts, "contacts", errors)

    immune = contacts.get("immune", [])
    if not isinstance(immune, list):
        errors.append("rules.contacts.immune must be an array of piece ids")
    else:
        for piece in immune:
            if piece not in piece_ids:
                errors.append(
                    f"rules.contacts.immune entry {piece!r} is not a declared piece"
                )

    links = contacts.get("links")
    if links is None:
        errors.append("rules.contacts needs links: a table with no link is no rule")
        return
    if not isinstance(links, list) or not links:
        errors.append(
            "rules.contacts.links must be a non-empty array of mover/onto links"
        )
        return

    seen: set[tuple[str, str, bool]] = set()
    for index, link in enumerate(links):
        loc = f"rules.contacts.links[{index}]"
        if not isinstance(link, dict):
            errors.append(f"{loc} must be an object with 'mover' and 'onto'")
            continue
        _check_keys(link, _CONTACT_LINK_KEYS, loc, errors, required={"mover", "onto"})
        mover, onto = link.get("mover"), link.get("onto")
        # Absence already says "an enemy target", so `false` would be a second spelling.
        if "own" in link and link["own"] is not True:
            errors.append(f"{loc}.own must be true; leave it out for an enemy target")
        own = link.get("own") is True
        if own and isinstance(immune, list) and onto in immune:
            errors.append(
                f"{loc} lets {mover!r} move onto its own {onto!r}, which "
                "rules.contacts.immune says no contact reaches"
            )
        for role, value in (("mover", mover), ("onto", onto)):
            if value not in piece_ids:
                errors.append(f"{loc}.{role} {value!r} is not a declared piece")
        # Absence is the spelling for "unchanged" and for "removed", so a key present
        # and pointing at nothing would be a second spelling of one of them -- the
        # `strike.chain` ruling, which this table inherits along with the shape.
        for role in ("mover_to", "onto_to"):
            if role not in link:
                continue
            becomes = link[role]
            if becomes not in piece_ids:
                errors.append(f"{loc}.{role} {becomes!r} is not a declared piece")
        if "mover_to" not in link and "onto_to" not in link:
            errors.append(
                f"{loc} says neither what the mover becomes nor what the target becomes"
                + (
                    "; a link onto an own piece has to change one of them"
                    if own
                    else ", so it is an ordinary capture written the long way"
                )
            )
        if link.get("mover_to") == mover and not keeping_link_holds(link, promoting):
            errors.append(
                f"{loc}.mover_to points {mover!r} at itself, which changes nothing"
                + (
                    "; a link may keep its mover only where the mover has a promotion "
                    "for it to override (schema 18), and a link onto an own piece never"
                    if not own
                    else ""
                )
            )
        if link.get("onto_to") == onto:
            errors.append(f"{loc}.onto_to points {onto!r} at itself, which changes nothing")
        if isinstance(mover, str) and isinstance(onto, str):
            whose = "its own " if own else ""
            if (mover, onto, own) in seen:
                errors.append(
                    f"rules.contacts.links has two links for {mover!r} onto {whose}{onto!r}; "
                    "one meeting has one outcome"
                )
            seen.add((mover, onto, own))

    # The strike's one question, asked of the table: a link onto an immune target keeps
    # nothing alive, because a contact reaches that target only through `own` (Q271).
    survives = strike_survives_anything(contacts, "contacts", piece_ids)
    if position == "advance" and survives:
        errors.append(
            "rules.contacts.position 'advance' has nowhere to put a target that "
            "survives; the mover takes its square, so use 'stay', 'swap' or 'choose'"
        )
    if position == "choose" and not survives:
        errors.append(
            "rules.contacts.position 'choose' needs a link that keeps a target alive, "
            "and a link onto a piece in rules.contacts.immune keeps none; with nothing "
            "to place it is 'advance' written the long way"
        )


def _validate_landing(rule: dict[str, Any], key: str, errors: list[str]) -> None:
    """``landing`` exactly when ``position`` is ``choose`` (schema 12, Q20).

    Called before the chain or links are read, so a refusal there does not hide this one.
    """
    loc = f"rules.{key}.landing"
    position = rule.get("position", "advance")
    landing = rule.get("landing")
    if landing is None:
        if position == "choose":
            errors.append(
                f"rules.{key}.position 'choose' needs {loc}, the squares a survivor may "
                "be placed on"
            )
        return
    if position != "choose":
        errors.append(f"{loc} only means something under position 'choose'")
    if not isinstance(landing, dict):
        errors.append(f"{loc} must be an object with 'vectors' and 'symmetry'")
        return
    _check_keys(landing, _LANDING_KEYS, loc, errors, required={"vectors"})
    symmetry = landing.get("symmetry", "none")
    if symmetry not in SYMMETRIES:
        errors.append(f"{loc}.symmetry must be one of {SYMMETRIES}, got {symmetry!r}")
    _validate_vectors(landing.get("vectors"), loc, errors)


def _validate_turn(turn: Any, errors: list[str]) -> None:
    """How many moves a turn is, and how the opening ramps up to it.

    ``ramp`` is the half a designer found worth keeping when he discarded the rest: with
    two moves a turn, whoever starts is a whole move ahead forever, and giving the first
    turn one move instead levels it. It lists the lengths of the opening turns in order,
    so ``moves: 2, ramp: [1]`` is the 1-2-2-2 pattern.

    This is the one rule here that a search feels rather than reads. Every other rule
    changes what a move is; this changes how many of them a node has, so the branching
    factor is the legal move count raised to the turn length -- which is why the schema
    caps it rather than trusting a proposal to be reasonable.
    """
    if turn is None:
        return
    if not isinstance(turn, dict):
        errors.append("rules.turn must be an object")
        return
    _check_keys(turn, _TURN_KEYS, "rules.turn", errors, required={"moves"})

    moves = turn.get("moves")
    if isinstance(moves, bool) or not isinstance(moves, int) or not 1 <= moves <= MAX_MOVES_PER_TURN:
        errors.append(
            f"rules.turn.moves must be an integer between 1 and {MAX_MOVES_PER_TURN}"
        )

    ramp = turn.get("ramp")
    if ramp is None:
        return
    if not isinstance(ramp, list) or not ramp:
        errors.append("rules.turn.ramp must be a non-empty array of turn lengths")
        return
    if len(ramp) > 8:
        errors.append("rules.turn.ramp may describe at most eight opening turns")
    for entry in ramp:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, int)
            or not 1 <= entry <= MAX_MOVES_PER_TURN
        ):
            errors.append(
                f"rules.turn.ramp entry {entry!r} must be an integer between 1 and "
                f"{MAX_MOVES_PER_TURN}"
            )


def _validate_territory(
    territory: Any, width: int, height: int, errors: list[str]
) -> None:
    """Squares that remember who moved through them.

    ``win_at`` of 0 means the colouring is scenery unless something else reads it --
    ``rules.scoring`` with ``count: "territory"`` is the other consumer, and it decides
    a game on ground held when the clock runs out rather than the moment a threshold is
    crossed. Both at once is legal and is what version 5 of the RPS sequence wanted.
    """
    if territory is None:
        return
    if not isinstance(territory, dict):
        errors.append("rules.territory must be an object")
        return
    _check_keys(territory, _TERRITORY_KEYS, "rules.territory", errors, required={"mode"})

    mode = territory.get("mode")
    if mode not in TERRITORY_MODES:
        errors.append(
            f"rules.territory.mode must be one of {TERRITORY_MODES}, got {mode!r}"
        )

    win_at = territory.get("win_at", 0)
    if isinstance(win_at, bool) or not isinstance(win_at, int) or win_at < 0:
        errors.append("rules.territory.win_at must be a non-negative integer")
    elif win_at > width * height:
        errors.append(
            f"rules.territory.win_at is {win_at}, which is more squares than the board "
            f"has ({width * height}); nobody could reach it"
        )


def _validate_scoring(
    scoring: Any, width: int, height: int, errors: list[str], has_territory: bool = False
) -> None:
    """How a game decides itself when the clock runs out instead of drawing.

    The ply limit and the no-progress clock are otherwise always a draw, which means a
    game with a real advantage in it can still end level -- the failure the Go lesson
    names: "a small advantage will straight up win you the game" is a statement about
    scoring, and without one every long game is a draw however far ahead someone is.
    """
    if scoring is None:
        return
    if not isinstance(scoring, dict):
        errors.append("rules.scoring must be an object")
        return
    _check_keys(scoring, _SCORING_KEYS, "rules.scoring", errors, required={"count"})

    count = scoring.get("count")
    if count not in SCORING_COUNTS:
        errors.append(
            f"rules.scoring.count must be one of {SCORING_COUNTS}, got {count!r}"
        )

    if count == "territory" and not has_territory:
        # The dependency a spec can get wrong: a count with nothing to count. The
        # reverse -- territory with no reader -- is legal, and is simply scenery.
        errors.append(
            'rules.scoring.count "territory" needs rules.territory, or there is nothing '
            "coloured to compare"
        )

    region = scoring.get("region")
    if count == "region":
        if region is None:
            errors.append('rules.scoring.count "region" needs rules.scoring.region')
        else:
            _validate_squares(region, "rules.scoring.region", width, height, errors)
    elif region is not None:
        errors.append(
            f"rules.scoring.region only means something when count is 'region', "
            f"not {count!r}"
        )


def _validate_promotion(
    promotion: Any,
    piece_ids: list[str],
    width: int,
    height: int,
    errors: list[str],
    where: str = "rules.promotion",
    hex_lattice: bool = False,
) -> None:
    if promotion is None:
        return
    if not isinstance(promotion, dict):
        errors.append(f"{where} must be an object")
        return
    _check_keys(promotion, _PROMOTION_KEYS, where, errors, required={"piece"})

    promoting = promotion.get("piece")
    if promoting not in piece_ids:
        errors.append(f"{where}.piece {promoting!r} is not a declared piece")

    if "rank" in promotion and "ranks" in promotion:
        errors.append(f"{where} has both 'rank' and 'ranks'; use one or the other")

    rank = promotion.get("rank", "last")
    if rank != "last" and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
        errors.append(f"{where}.rank must be 'last' or a non-negative integer")
    # A lone rank past the board is refused too, but only for a new spec:
    # `_refuse_a_promotion_rank_past_the_board`, below `validate_spec`.

    # A promotion *region* rather than a single rank: grand chess promotes anywhere in
    # the last three, and whether promotion is compulsory is what makes that meaningful.
    if "ranks" in promotion:
        ranks = promotion["ranks"]
        if not isinstance(ranks, list) or not ranks:
            errors.append(f"{where}.ranks must be a non-empty array of ranks")
        else:
            for entry in ranks:
                if isinstance(entry, bool) or not isinstance(entry, int) or not 0 <= entry < max(height, 1):
                    errors.append(
                        f"{where}.ranks entry {entry!r} is not a rank on this board"
                    )

    # Schema 14: a zone stated as named squares, for a zone that is a shape rather than a
    # band. On a hex lattice it is the only spelling, because a hexagon's far edge is not
    # a rank; a wall inside the zone is refused by `_check_obstacle_coherence`.
    if "squares" in promotion:
        if "rank" in promotion or "ranks" in promotion:
            errors.append(
                f"{where} has 'squares' beside 'rank' or 'ranks'; a zone is stated one way"
            )
        squares = promotion["squares"]
        _validate_squares(
            [] if squares is None else squares, f"{where}.squares", width, height, errors
        )
    elif hex_lattice:
        stated = " and ".join(
            f"'{key}'" for key in ("rank", "ranks") if key in promotion
        ) or "the default 'rank': 'last'"
        errors.append(
            f"{where} states its zone by {stated}, and a hexagon's far edge is not a rank; "
            "on a hex lattice state the zone as 'squares'"
        )

    if "mandatory" in promotion and not isinstance(promotion["mandatory"], bool):
        errors.append(f"{where}.mandatory must be a boolean")
    # Schema 15: promote on a move that starts in the zone as well as on one that ends
    # there. What that means is `compile.PromotionRule.on_leaving`'s to say.
    if "on_leaving" in promotion and not isinstance(promotion["on_leaving"], bool):
        errors.append(f"{where}.on_leaving must be a boolean")
    # Schema 18 (Q783): a target only while its side has lost one. What it is read
    # against is `_check_only_lost_coherence`'s to say.
    if "only_lost" in promotion and not isinstance(promotion["only_lost"], bool):
        errors.append(f"{where}.only_lost must be a boolean")

    if "to" not in promotion and "to_by_player" not in promotion:
        errors.append(f"{where} needs 'to' or 'to_by_player'")
    if "to" in promotion and "to_by_player" in promotion:
        errors.append(f"{where} has both 'to' and 'to_by_player'; use one or the other")

    if "to" in promotion:
        _validate_promotion_targets(promotion["to"], promoting, piece_ids, f"{where}.to", errors)

    # Chigorin gives the two sides different armies and different promotion choices.
    if "to_by_player" in promotion:
        by_player = promotion["to_by_player"]
        if not isinstance(by_player, dict):
            errors.append(f"{where}.to_by_player must be an object keyed by '0' and '1'")
        else:
            unknown = sorted(set(by_player) - {"0", "1"})
            if unknown:
                errors.append(
                    f"{where}.to_by_player has unknown players: {', '.join(unknown)}"
                )
            for player in ("0", "1"):
                if player not in by_player:
                    errors.append(f"{where}.to_by_player is missing player {player!r}")
                    continue
                _validate_promotion_targets(
                    by_player[player], promoting, piece_ids,
                    f"{where}.to_by_player[{player!r}]", errors,
                )


def _refuse_a_promotion_rank_past_the_board(
    rules: Any, height: int, errors: list[str]
) -> None:
    """A lone promotion ``rank`` is bounded by the board, as a ``ranks`` band's entries are.

    ``promotion_squares`` skips a rank off the board, so the zone is empty and the rule
    can never fire, and the words are the band's so the two spellings read alike. It is
    apart from ``_validate_promotion`` for one reason: it runs only on a new spec
    (``validate_spec``'s ``new``). `tunnel-horizontal` said ``rank: 9`` on an 8-high
    board and was played and kept as that spec (Q102 found it; Q434 retired it), so the
    site still draws its page and replays its matches from those rules, and the
    calibration corpora under ``runs/`` still hold it. A malformed rank or board is
    ``_validate_promotion``'s and ``_validate_board``'s to report, and is skipped here.
    """
    if not isinstance(rules, dict) or height <= 0:
        return
    promotion = rules.get("promotion")
    stated = (
        [(f"rules.promotion[{index}]", entry) for index, entry in enumerate(promotion)]
        if isinstance(promotion, list)
        else [("rules.promotion", promotion)]
    )
    for where, entry in stated:
        rank = entry.get("rank") if isinstance(entry, dict) else None
        if isinstance(rank, int) and not isinstance(rank, bool) and rank >= height:
            errors.append(f"{where}.rank {rank!r} is not a rank on this board")


def _validate_promotion_targets(
    targets: Any, promoting: Any, piece_ids: list[str], where: str, errors: list[str]
) -> None:
    if not isinstance(targets, list) or not targets:
        errors.append(f"{where} must be a non-empty array of piece ids")
        return
    for target in targets:
        if target not in piece_ids:
            errors.append(f"{where} entry {target!r} is not a declared piece")
        elif target == promoting:
            errors.append(f"{where} may not include the promoting piece itself")


def _validate_castling(
    castling: Any, piece_ids: list[str], width: int, errors: list[str]
) -> None:
    if castling is None:
        return
    if not isinstance(castling, dict):
        errors.append("rules.castling must be an object")
        return
    _check_keys(
        castling, _CASTLING_KEYS, "rules.castling", errors, required={"king", "rook"}
    )
    for key in ("king", "rook"):
        if castling.get(key) not in piece_ids:
            errors.append(f"rules.castling.{key} {castling.get(key)!r} is not a declared piece")

    if "king_steps" in castling and "king_file" in castling:
        errors.append(
            "rules.castling has both 'king_steps' and 'king_file'; a step count and a "
            "destination file are two ways of saying the same thing"
        )

    steps = castling.get("king_steps", 2)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 4:
        errors.append("rules.castling.king_steps must be an integer between 1 and 4")

    # Castling described by where the king lands rather than how far it travels. Chess
    # itself does not need this, but chess960 does -- the king starts on a different
    # file every game and always finishes on c or g -- and so do janus and embassy,
    # whose two castlings are different distances.
    for key in ("king_file", "rook_file"):
        if key not in castling:
            continue
        files = castling[key]
        if not isinstance(files, dict):
            errors.append(f"rules.castling.{key} must be an object with 'kingside' and 'queenside'")
            continue
        unknown = sorted(set(files) - {"kingside", "queenside"})
        if unknown:
            errors.append(f"rules.castling.{key} has unknown sides: {', '.join(unknown)}")
        for side in ("kingside", "queenside"):
            if side not in files:
                errors.append(f"rules.castling.{key} is missing {side!r}")
                continue
            if parse_file(files[side], width) is None:
                errors.append(
                    f"rules.castling.{key}.{side} {files[side]!r} is not a file on this board"
                )
    if "rook_file" in castling and "king_file" not in castling:
        errors.append("rules.castling.rook_file requires king_file")


def parse_file(name: Any, width: int) -> int | None:
    """Parse a single file letter (``a``, ``h``, ``l``) into a zero-based index."""
    if not isinstance(name, str) or len(name) != 1 or not name.isalpha():
        return None
    index = ord(name.lower()) - ord("a")
    return index if 0 <= index < width else None


# --------------------------------------------------------------------------- shared


def _check_keys(
    obj: dict[str, Any],
    allowed: set[str],
    where: str,
    errors: list[str],
    required: set[str] | None = None,
) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        errors.append(f"{where} has unknown keys: {', '.join(unknown)}")
    for key in sorted(required or set()):
        if key not in obj:
            errors.append(f"{where} is missing required key {key!r}")
