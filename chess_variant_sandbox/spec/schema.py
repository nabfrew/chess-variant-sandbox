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
from typing import TYPE_CHECKING, Any, NamedTuple

from chess_variant_sandbox.spec.reachability import (
    _pieces_that_can_stand,
    _promotion_entries,
    _promotion_target_ids,
)
from chess_variant_sandbox.spec.reachability import (
    pieces_a_player_can_hold as pieces_a_player_can_hold,
)

if TYPE_CHECKING:
    from chess_variant_sandbox.engine.compile import StrikeRule

SCHEMA_VERSION = 16

# Every version is accepted, and the number a spec declares is a claim about which
# feature set it uses rather than a date stamp. That is what keeps a bump cheap:
# ``schema_version`` is part of the mechanics hash, so rewriting the committed rulesets
# to say 3 would rehash every one of them and orphan the calibration runs under
# ``runs/``. A spec that declares 1 may not use anything added in 2 or 3, and the
# validator enforces that by name, so the version stays an honest description of what
# the rules need.
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

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
SYMMETRIES = ("none", "lateral", "rotate4", "dihedral8")

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

_FACING_KEYS = {"axes", "initial"}

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
_NPC_KEYS = {"piece", "route", "bounty", "mirror", "immune"}

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
# Under ``advance`` there is nowhere to put a survivor, and the validator refuses that
# combination rather than quietly removing a piece the chain said should live.
STRIKE_POSITIONS = ("stay", "advance", "swap", "choose")

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
    "facing",
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
_PROVENANCE_KEYS = {
    "origin", "author", "year", "source", "history", "theme", "parents", "exhibit",
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
}
_OBSTACLES_KEYS = {"squares", "mirror", "blocks_leaps"}
_STRIKE_KEYS = {"effect", "chain", "position", "immune", "landing"}
_STRIKE_LINK_KEYS = {"from", "to"}
# The three pairs are schema 16's, a hexagon's alone (`WRAP_PAIRS`); `files` and `ranks`
# are a grid's alone. `_validate_wrap` refuses each shape's keys on the other.
_WRAP_KEYS = {
    "files", "ranks", "files_by_player", "ranks_by_player",
    "n_s", "ne_sw", "nw_se", "n_s_by_player", "ne_sw_by_player", "nw_se_by_player",
}
# ``on_leaving`` is schema 15 (Q452, traditional shogi's B1): the entry also promotes its
# piece on a move that *starts* in the zone, as Fairy-Stockfish and real shogi do. See
# ``_schema_fifteen_uses`` and ``compile.PromotionRule.on_leaving``.
_PROMOTION_KEYS = {
    "piece", "rank", "to", "ranks", "squares", "mandatory", "to_by_player", "on_leaving",
}
_CASTLING_KEYS = {"king", "rook", "king_steps", "king_file", "rook_file"}
# ``stock`` is schema 10 (Q51): what each player holds in hand before a piece has been
# taken. See ``piece_has_stock`` and ``_validate_drops``.
_DROPS_KEYS = {"from_captures", "region", "no_doubled", "no_mate", "stock"}

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
_REACH_KEYS = {"piece", "squares", "mirror", "result", "safe"}
_EXTINCTION_KEYS = {"pieces", "min_count", "result"}
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
    entry re-scored. What that skips is one refusal,
    ``_refuse_a_promotion_rank_past_the_board``, added (Q434) after such a spec had been
    played. It declares a rule that never fires, so the engine plays that spec exactly as
    it always did; the refusal is about not admitting another one.
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
        placements = (spec.get("setup") or {}).get("placements")
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
        "strike": isinstance(rules.get("strike"), dict)
        and rules["strike"].get("effect", "capture") != "capture",
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
        piece.get("id")
        for piece in pieces
        if isinstance(piece, dict)
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
    royal_ids = {p.get("id") for p in pieces if isinstance(p, dict) and p.get("royal")}
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
            if not isinstance(placement, dict) or placement.get("piece") not in royal_ids:
                continue
            squares = placement.get("squares")
            if isinstance(squares, list):
                player = placement.get("player", 0)
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

    for key in ("origin", "author", "source", "history", "theme", "exhibit"):
        value = provenance.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"provenance.{key} must be a string")
    exhibit = provenance.get("exhibit")
    if isinstance(exhibit, str) and not exhibit.strip():
        errors.append("provenance.exhibit must say what the ruleset is kept to show")

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
        royal_ids = {p["id"] for p in pieces if isinstance(p, dict) and p.get("royal")}
        placed = {
            p.get("piece")
            for p in setup["placements"]
            if isinstance(p, dict)
        }
        missing = royal_ids - placed
        if missing == royal_ids:
            errors.append("no royal piece appears in setup.placements")

    piece_ids_all = {p["id"] for p in pieces if isinstance(p, dict) and "id" in p}
    for piece in pieces:
        if not isinstance(piece, dict):
            continue
        base = piece.get("promoted_from")
        if base is not None and base not in piece_ids_all:
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
    _check_inert_rules(spec, pieces, rules, errors)
    # Three owners share the one byte once anything is wild, and the bound has to know.
    _check_facing_coherence(pieces, errors, owners=3 if rules.get("npcs") else 2)

    if isinstance(rules.get("castling"), dict):
        castling = rules["castling"]
        king_id = castling.get("king")
        if has_royal and king_id is not None:
            royal_ids = {p["id"] for p in pieces if isinstance(p, dict) and p.get("royal")}
            if king_id not in royal_ids:
                errors.append(
                    f"rules.castling.king {king_id!r} is not a royal piece; castling protects "
                    "royalty, so pointing it at a common piece is almost always a mistake"
                )


def _check_facing_coherence(pieces: list[Any], errors: list[str], owners: int = 2) -> None:
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
    highest = owners * facings * len(pieces)
    if highest >= 256:
        who = "two players" if owners == 2 else "two players and a wild third owner"
        errors.append(
            f"{len(pieces)} piece types at {facings} facings for {who} need "
            f"{highest + 1} board values, and a square is one byte; reduce the piece "
            "count or the facings"
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


def hexagon_joins_a_side(spec: Any) -> bool:
    """Whether the spec is a hexagon with any of its six sides open to anyone."""
    if board_shape(spec) != "hexagon":
        return False
    rules = spec.get("rules")
    wrap = rules.get("wrap") if isinstance(rules, dict) else None
    return any(
        value not in (None, "none")
        for values in wrap_stated(wrap, "hexagon").values()
        for value in values
    )


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

    walled_ranks = sorted({square // width for square in blocked})
    for rank in _castling_ranks(rules, setup, width, height, orientation):
        if rank not in walled_ranks:
            continue
        standing = sorted(
            format_square(square % width, rank)
            for square in blocked
            if square // width == rank
        )
        if absent:
            errors.append(
                f"rules.castling happens along rank {rank + 1}, which runs off the hexagon "
                f"at {', '.join(standing)}; the king and the rook walk a rank by file and "
                "would step onto squares that are not cells, so on a hexagon castling "
                "stays on the middle rank"
            )
        else:
            errors.append(
                f"rules.castling happens along rank {rank + 1}, which holds an obstacle at "
                f"{', '.join(standing)}; the king and the rook cross that rank and may end "
                "up on any square of it, and nothing may stand on a wall"
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
    if not isinstance(strike, dict) or strike.get("effect", "capture") == "capture":
        return

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
    immune_ids = set(immune) if isinstance(immune, list) else set()
    chain = strike.get("chain")
    struck_ids = {
        link.get("from")
        for link in (chain if isinstance(chain, list) else [])
        if isinstance(link, dict)
    }
    for piece in pieces:
        if not isinstance(piece, dict) or not piece.get("royal"):
            continue
        pid = piece.get("id")
        if pid in immune_ids:
            continue
        if pid in struck_ids:
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
    immune_ids = set(immune) if isinstance(immune, list) else set()
    links = [link for link in contacts["links"] if isinstance(link, dict)]
    # Schema 11: `must_capture` asks whether a move takes an enemy piece, and nothing
    # settles whether a move onto an own piece is one. Refused rather than guessed.
    if rules.get("must_capture") and any(link.get("own") is True for link in links):
        errors.append(
            "rules.contacts has a link onto an own piece beside rules.must_capture; "
            "whether moving onto your own piece counts as a capture is not settled"
        )
    met = {link.get("onto") for link in links}
    for piece in pieces:
        if not isinstance(piece, dict) or not piece.get("royal"):
            continue
        pid = piece.get("id")
        if pid not in immune_ids:
            errors.append(
                f"the royal piece {pid!r} is not in rules.contacts.immune; a struck "
                "royal must be a captured royal, so every royal piece has to be named "
                "immune"
                + (
                    f" -- and rules.contacts.links names {pid!r} on the 'onto' side"
                    if pid in met
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
      table and the core has to know which wild piece it is looking at;
    * **a wild piece starting under the opening array** -- two pieces on one square;
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

    by_id = {p["id"]: p for p in pieces if isinstance(p, dict) and "id" in p}
    for index, npc in enumerate(npcs):
        if not isinstance(npc, dict):
            continue
        piece = by_id.get(npc.get("piece"))
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
                    "square alone"
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
            for name in placement.get("squares") or []:
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
    for where, squares in routes:
        if squares and squares[0] in standing:
            start = format_square(squares[0] % width, squares[0] // width)
            errors.append(
                f"{where} starts on {start}, where setup.placements stands a piece; a "
                "wild piece needs an empty square to start from"
            )

    strike = rules.get("strike")
    if (
        isinstance(strike, dict)
        and strike.get("effect", "capture") != "capture"
        and strike.get("chain")
    ):
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
        and piece.get("promoted_from") in declared
    }
    standing = _pieces_that_can_stand(spec, listed, rules, declared)
    if pid not in standing or any(demoted.get(other, other) == pid for other in standing):
        return None
    return (
        f"never reaches a hand, since rules.drops.stock does not name it and a captured "
        f"{pid} is held as {demoted[pid]}"
    )


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
        and castling.get("king") in declared
        and castling.get("rook") in declared
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
    standing = _pieces_that_can_stand(spec, pieces, rules, declared)
    reach = rules.get("reach")
    if isinstance(reach, dict):
        piece_id = reach.get("piece")
        if piece_id in declared and piece_id not in standing:
            errors.append(
                f"rules.reach is about {piece_id!r}, which can never stand on the "
                f"board: {_NEVER_ON_THE_BOARD}"
            )
    promotion = rules.get("promotion")
    for index, entry in enumerate(_promotion_entries(rules)):
        if not isinstance(entry, dict):
            continue
        piece_id = entry.get("piece")
        if piece_id in declared and piece_id not in standing:
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
        if declared.intersection(allowed):
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
            stocked=piece_has_stock(rules, piece.get("id")),
            hex_lattice=hex_lattice,
        )
        _validate_facing(piece, where, errors, hex_lattice)

    # The capture matrix resolves ids, so it can only be checked once every id is known
    # -- a cycle is the point of the feature and rock has to be able to name scissors
    # whichever order they are declared in.
    for index, piece in enumerate(pieces):
        if isinstance(piece, dict):
            _validate_captures(piece.get("captures"), f"pieces[{index}]", ids, errors)

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

    if piece.get("royal", False):
        errors.append(
            f"{where}.facing on a royal piece: a royal that can turn is several piece "
            "values rather than one, and the check filter finds a royal by value. "
            "Give the facing to a piece that is not royal."
        )


def _validate_captures(
    captures: Any, where: str, piece_ids: list[str], errors: list[str]
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
        errors.append(f"{where}.captures must be an array of piece ids")
        return
    for entry in captures:
        if not isinstance(entry, str):
            errors.append(f"{where}.captures entry {entry!r} must be a piece id")
        elif entry not in piece_ids:
            errors.append(f"{where}.captures entry {entry!r} is not a declared piece")
    if len(set(captures)) != len(captures):
        errors.append(f"{where}.captures repeats a piece id")


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
        elif hex_lattice and symmetry != "none":
            errors.append(
                f"{loc}.symmetry {symmetry!r} expands vectors by the square's symmetries, "
                "which carry some of them off the hex lattice; on a hex board list every "
                "vector under 'none'"
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
                    for p in (pieces_by_id or [])
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
            key = (player if isinstance(player, int) else 0, square)
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
    _validate_contacts(rules.get("contacts"), piece_ids, errors)
    _validate_npcs(rules.get("npcs"), piece_ids, width, height, errors, rules)
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
                if piece in seen:
                    errors.append(
                        f"rules.promotion has two rules for {piece!r}; a piece promotes one way"
                    )
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

    ``reach.mirror`` and a landing's ``symmetry``. The pieces' own descriptors, their
    facing, the setup's orientation and a rank-shaped promotion zone are refused where
    each is validated; these two live on rules that are validated without the lattice.
    """
    reach = rules.get("reach")
    if isinstance(reach, dict) and reach.get("mirror", True) not in ("rotate", False):
        errors.append(
            'rules.reach.mirror must be "rotate" or false on a hex lattice: true, the '
            "default, reflects the goal by rank, which is not a hex symmetry"
        )
    for key in ("strike", "contacts"):
        rule = rules.get(key)
        landing = rule.get("landing") if isinstance(rule, dict) else None
        symmetry = landing.get("symmetry", "none") if isinstance(landing, dict) else "none"
        if symmetry in SYMMETRIES and symmetry != "none":
            errors.append(
                f"rules.{key}.landing.symmetry {symmetry!r} expands vectors by the square's "
                "symmetries, which carry some of them off the hex lattice; on a hex board "
                "list every vector under 'none'"
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
    if isinstance(radius, bool) or not isinstance(radius, int) or not 1 <= radius <= 3:
        errors.append("rules.explode_on_capture.radius must be an integer between 1 and 3")

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
    return isinstance(stock, dict) and piece_id in stock


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
    wild = {
        entry.get("piece")
        for entry in (npcs or [])
        if isinstance(entry, dict)
    }
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

        _validate_npc_immune(npc, loc, rules or {}, piece_ids, errors)

    _check_one_immunity_per_type(npcs, errors)


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
    readable = {
        **rule,
        "immune": [p for p in immune if names_declared(p)] if isinstance(immune, list) else [],
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
        f"'capture' none does: {PLAYED_AS_CAPTURE}. A capture that leaves the attacker "
        "where it stood (rifle chess) is not in the DSL. Give the strike effect 'demote' "
        "or 'promote' and a chain, or leave rules.strike out"
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


def _validate_contacts(contacts: Any, piece_ids: list[str], errors: list[str]) -> None:
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
    _validate_contacts_shape(contacts, piece_ids, errors)
    if len(errors) == before:
        errors.extend(strike_states_nothing(contacts, "contacts", piece_ids))


def _validate_contacts_shape(contacts: Any, piece_ids: list[str], errors: list[str]) -> None:
    """The table's keys, values and links, and the refusals that read its position."""
    if contacts is None:
        return
    if not isinstance(contacts, dict):
        errors.append("rules.contacts must be an object")
        return
    _check_keys(contacts, _CONTACTS_KEYS, "rules.contacts", errors)

    position = contacts.get("position", "advance")
    if position not in STRIKE_POSITIONS:
        errors.append(
            f"rules.contacts.position must be one of {STRIKE_POSITIONS}, got {position!r}"
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
        if link.get("mover_to") == mover:
            errors.append(f"{loc}.mover_to points {mover!r} at itself, which changes nothing")
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
