"""How two rulesets differ, counted in rules rather than in JSON leaves.

`distance.py` answers *how far apart* with a number and no words. This answers *what
differs*, one rule at a time, where a rule is the board, one piece's movement, the setup,
who may take whom, or one key under `rules`. The count is the weight on a link in the
library's graph and each difference carries a kind, a size and a sentence.

The whole module reads two spec files. It plays nothing and asks the engine nothing --
comparing two written descriptions is not an engine question. It compiles in one place
only: whether `setup.orientation: rotate` is a rule or a spelling depends on how the second
player reads every square and vector written from the first player's side, which is the
compiler's to say, so `_orientation_played` asks it rather than re-reading the DSL here.
Two specs that describe one game under two spellings the normal form does not fold are
still two specs here; merging those is `web/graphing.py`'s and it uses the compiler's own
table fingerprint.

**Why a normal form.** A raw diff answers about spelling. One replaced piece in `almost`
costs 12 JSON leaves against chess and a reordered placement list costs 85, while both are
one rule. So before anything is compared, symmetries expand to direction sets, like move
descriptors merge, placements expand to cells, lists lose their order, written defaults
drop out and piece letters stop mattering. `docs/reference/game-graph-probe.md` measures
what each of those repairs and what is left.

**Consequences fold into their cause.** An edit carries things with it. A removed piece
takes its squares, its mentions in rule lists and the rules that name it; an added piece
takes the squares it displaced; a resized board rewrites the promotion rank; a piece that
loses its double step takes en passant. Each fold is a rule about what one edit entails,
never a threshold, and each is named where it is written below. Over 5,549 operator
children -- every one of which is one edit from its parent by construction -- this reads
the step as one rule 96% of the time, against 86% without the folds.

**Nothing here knows a game.** Kinds are keyed by schema position (`RULE_KINDS` covers
`schema._RULES_KEYS`), pieces by id and square, and no piece name or ruleset name appears
anywhere in the module. Piece ids are renamed *by schema position* too, from
`_PIECE_POSITIONS`, rather than by matching the string wherever it occurs: a piece may be
given any id the schema accepts, so a string match would rewrite a square name or an
ending word that happens to spell one. `tests/test_relate.py` pins both tables against
`schema.py`, so a bump that adds a `rules` key fails until the key is classified.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.schema import (
    DEFAULT_RANGE,
    absent_squares,
    can_double_step,
    format_square,
    has_sights,
    obstacle_squares,
    parse_square,
    piece_sights,
    portal_stand_in_shape,
    reflect_square,
    wrap_stated,
)
from chess_variant_sandbox.spec.spelling import current_spelling

# Bumped when a change here would move a stored count or sentence. A cached link carries
# it, so a reader that cannot match it recomputes instead of showing an older reading.
# r2: direction sets sorted inside the key, and `orientation` read as played (Q503c2).
# r3: a matched pair whose names differ is named from both sides (Q861). No count moves;
# the bump is for the stored sentences, which the graph's pair file and the site's
# `links` rows hold under this stamp and so recompute instead of showing the old words.
RELATE_VERSION = "r3"

# Which kind of rule each `rules` key is. Exhaustive over `schema._RULES_KEYS`, and a
# test fails when the schema grows a key with no row here -- a link with no kind is a
# link the map cannot colour or filter. `wrap` and `obstacles` are board rules: they are
# declared under `rules` and what they change is the shape of the board.
RULE_KINDS: dict[str, str] = {
    "wrap": "board",
    "obstacles": "board",
    "portals": "board",
    "castling": "pieces",
    "promotion": "pieces",
    "promote_on_entry": "pieces",
    "forbid_immobility": "pieces",
    "en_passant": "capture",
    "must_capture": "capture",
    "explode_on_capture": "capture",
    "strike": "capture",
    "contacts": "capture",
    "win": "ending",
    "stalemate": "ending",
    "bare_royal": "ending",
    "check_limit": "ending",
    "reach": "ending",
    "extinction": "ending",
    "checkmate": "ending",
    "forbid_check": "ending",
    "scoring": "ending",
    "territory": "ending",
    "royal_facing": "ending",
    "max_plies": "clock",
    "repetition_draw": "clock",
    "no_progress_draw": "clock",
    "drops": "hand",
    "npcs": "wild",
    "turn": "turn",
    "simultaneous": "turn",
}

# What each `rules` key is *about*, for the sentence on a link. Exhaustive over
# `schema._RULES_KEYS` beside `RULE_KINDS`, and pinned by the same test: `the reach rule`
# tells a reader nothing, and a link whose whole content is a schema key is a link nobody
# can act on. These name the question the two games answer differently; what each of them
# answers is `describe.py`'s and arrives with the prose diff.
RULE_WORDS: dict[str, str] = {
    "wrap": "which edges of the board are joined",
    "obstacles": "which squares are walled off",
    "portals": "which squares send a piece out of another",
    "castling": "the castling move",
    "promotion": "what a piece becomes at the far end",
    "promote_on_entry": "what a piece becomes on entering a region",
    "forbid_immobility": "whether a piece may be left with no move",
    "en_passant": "taking a piece in passing",
    "must_capture": "whether a capture has to be taken",
    "explode_on_capture": "a capture taking its neighbours with it",
    "strike": "taking a piece without moving onto it",
    "contacts": "what a piece becomes when it takes one",
    "win": "how the game is won",
    "stalemate": "what a player with no move gets",
    "bare_royal": "what becomes of a lone royal piece",
    "check_limit": "how many checks lose the game",
    "reach": "which squares win the game when reached",
    "extinction": "which piece type being wiped out loses",
    "checkmate": "what trapping a royal piece means",
    "forbid_check": "whether a player may be left in check",
    "scoring": "what the game is scored on",
    "territory": "holding ground",
    "royal_facing": "which way a royal piece may not face",
    "max_plies": "how long a game may run",
    "repetition_draw": "how often a position may repeat",
    "no_progress_draw": "how long without progress before a draw",
    "drops": "putting captured pieces back on the board",
    "npcs": "the pieces that belong to nobody",
    "turn": "how many moves a turn is",
    "simultaneous": "whether both players move at once, and who survives a collision",
}

# The kinds, in the order a legend lists them. Every `Difference.kind` is one of these.
KINDS: tuple[str, ...] = (
    "board",
    "pieces",
    "setup",
    "capture",
    "ending",
    "hand",
    "wild",
    "turn",
    "clock",
    "valuation",
)

# The five `rules` values the compiler reads as "what the game would do anyway". A spec
# that writes one is saying nothing, so writing it must not cost a link.
_WRITTEN_DEFAULTS: dict[str, Any] = {
    "win": "checkmate",
    "stalemate": "draw",
    "max_plies": 400,
    "repetition_draw": 3,
    "no_progress_draw": 100,
}

# The marker an id is renamed to when the piece belongs to only one of the two games.
# `_frozen` drops it from lists and from dict keys, so a promotion list that lost a
# removed piece compares equal to one that never held it.
GONE = "\x00gone"

# The three ways two armies can be paired, each a sequence of stages of evidence. The
# matching that yields the fewest differences wins -- see `_compare`.
_MATCHINGS: tuple[tuple[str, ...], ...] = (
    ("same", "moves", "letter", "closest"),
    ("cells", "letter", "closest"),
    ("letter", "closest"),
)


@dataclass(frozen=True)
class Difference:
    """One rule two games do not share.

    `unit` is the schema position that differs (`board.size`, `piece.moves`,
    `rules.promotion`), which is what an operator table and a link's colour are keyed on.
    `sentence` is the same fact in words. `size` is the magnitude where the unit has one
    -- cells in a setup, pieces carrying a capture list -- and `None` where it does not.

    **A sentence reads as what the *second* spec does differently**, so a page listing one
    game's neighbours passes its own game first and each sentence is about the neighbour.
    The count is symmetric and only the wording has a direction; `unit` carries it too,
    which is the difference between `piece.added` and `piece.removed`.
    """

    kind: str
    unit: str
    sentence: str
    size: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "unit": self.unit,
            "sentence": self.sentence,
            "size": self.size,
        }


@dataclass(frozen=True)
class Relation:
    """What two games differ in, and how many rules that is."""

    differences: tuple[Difference, ...]

    @property
    def units(self) -> int:
        return len(self.differences)

    @property
    def sentence(self) -> str:
        """Every difference in one line, in the order they were found."""
        return "; ".join(difference.sentence for difference in self.differences)

    def as_dict(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "sentence": self.sentence,
            "differences": [difference.as_dict() for difference in self.differences],
        }


# ------------------------------------------------------------------ the normal form


@dataclass(frozen=True, eq=False)
class NormalForm:
    """One spec with its spelling folded out, ready to be compared with another.

    Built once per spec and reused across every pair it appears in, because the per-spec
    work -- expanding symmetries, parsing squares, dropping defaults -- is most of the
    cost and a library of *n* games has *n* of it against *n squared* pairs.

    Not hashable on purpose (`eq=False`): `key` is the identity this is cached and
    compared by, and two forms with the same key describe the same game whatever the two
    files spell.
    """

    size: tuple[int, int]
    lattice: str
    wrap: Any
    walls: Any
    # `setup.orientation` as played: `"mirror"` wherever rotating changes nothing the
    # compiler builds, so a rotated army of pieces that read left and right alike is the
    # same game as both armies written out. See `_orientation_played`.
    orientation: str
    # The raw piece objects, by id. Kept raw because a comparison renames ids, and a
    # renaming has to reach inside `captures`, `region` and `promoted_from`.
    pieces: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Each piece's movement with symmetries expanded and like descriptors merged.
    forms: dict[str, frozenset[Any]] = field(default_factory=dict)
    # Where each piece starts, as (player, file, rank), for the by-square matching.
    stands: dict[str, frozenset[tuple[int, int, int]]] = field(default_factory=dict)
    # Every starting piece as (player, file, rank, id, facing), mirror or rotation applied.
    cells: frozenset[tuple[int, int, int, str, str]] = frozenset()
    # `rules` with the written defaults, the inert values and the board rules taken out.
    rules: dict[str, Any] = field(default_factory=dict)
    # Whether this game's rules could ever bring en passant into play. See `_compare`.
    en_passant_fires: bool = False
    key: str = ""
    # What each piece is called, by id. Outside `key`, which a renaming does not move,
    # but a sentence names pieces, so the pair cache is keyed on it too (`relate_forms`).
    names: tuple[tuple[str, str], ...] = ()


def normal_form(spec: dict[str, Any]) -> NormalForm:
    """This spec's normal form, memoised on the spec itself.

    Content-addressed, as `web/library.py` memoises a compiled engine: a spec is data and
    the same data always normalises the same way.
    """
    return _normal_cached(json.dumps(spec, sort_keys=True, default=str))


@lru_cache(maxsize=1024)
def _normal_cached(blob: str) -> NormalForm:
    return _build(json.loads(blob))


def identity_key(spec: dict[str, Any]) -> str:
    """This spec's normal form as a short digest.

    Two specs with the same key differ in nothing this module can count, so they are one
    game under two spellings as far as the written rules go. Used to group a library's
    duplicates and to ask whether an operator's child landed on a game that already
    exists.
    """
    return normal_form(spec).key


def _build(spec: dict[str, Any]) -> NormalForm:
    # Read in the one current spelling (Q1008b): a promotion written as one object or as a
    # list, a lone numbered rank or `ranks`, are one rule and must not read as a difference.
    spec = current_spelling(spec)
    # A portal group's moves read as the stand-in piece every portal game had before
    # spelling 31 (`schema.portal_stand_in_shape`), so no relation moves with the re-spelling.
    spec = portal_stand_in_shape(spec)
    board = spec.get("board") or {}
    width, height = _int(board.get("width"), 8), _int(board.get("height"), 8)
    setup = spec.get("setup") or {}
    written = str(setup.get("orientation", "mirror"))
    pieces = {str(piece["id"]): piece for piece in spec.get("pieces") or [] if "id" in piece}
    pieces = _live_sights(spec, pieces)
    forms = {pid: _move_form(piece) for pid, piece in pieces.items()}
    cells = _setup_cells(spec, width, height, written)
    walls = _wall_form(spec, width, height, written)
    orientation = _orientation_played(spec, written, cells)
    stands: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    for player, file_, rank, pid, _facing in cells:
        stands[pid].add((player, file_, rank))

    form = NormalForm(
        size=(width, height),
        lattice=str(board.get("lattice", "square")),
        wrap=_wrap_form(spec),
        walls=walls,
        orientation=orientation,
        pieces=pieces,
        forms=forms,
        stands={pid: frozenset(squares) for pid, squares in stands.items()},
        cells=frozenset(cells),
        rules=_live_rules(spec),
        en_passant_fires=_en_passant_fires(spec, pieces),
        key="",
        names=tuple(sorted((pid, str(piece.get("name", pid))) for pid, piece in pieces.items())),
    )
    object.__setattr__(form, "key", _key_of(form))
    return form


def _key_of(form: NormalForm) -> str:
    body = (
        form.size,
        form.lattice,
        form.wrap,
        form.walls,
        form.orientation,
        tuple(sorted(
            (
                pid,
                # Each direction set sorted too: a frozenset's repr keeps the order it
                # was built in, so one set of directions written two ways had two.
                repr(sorted(
                    ((attrs, tuple(sorted(dirs))) for attrs, dirs in form.forms[pid]),
                    key=repr,
                )),
                piece.get("value"),
                bool(piece.get("royal")),
                _frozen({
                    key: piece.get(key)
                    for key in ("captures", "facing", "region", "promoted_from")
                } | ({"beats": piece["beats"]} if "beats" in piece else {})
                  | ({"spy": piece["spy"]} if "spy" in piece else {})
                  | ({"speed": piece["speed"]} if "speed" in piece else {})
                  | ({"rest": piece["rest"]} if "rest" in piece else {})
                  | ({"sight": _sight_form(piece)} if "sight" in piece else {})),
            )
            for pid, piece in form.pieces.items()
        )),
        tuple(sorted(form.cells)),
        _frozen(form.rules),
        form.en_passant_fires,
    )
    return hashlib.sha256(repr(body).encode("utf-8")).hexdigest()[:16]


def _orientation_played(
    spec: dict[str, Any], written: str, cells: set[tuple[int, int, int, str, str]]
) -> str:
    """`setup.orientation` as a rule of play: `"mirror"` unless rotating changes the game.

    The setup is already keyed by the cells it puts pieces on, so what is left of
    `orientation` is how the second player reads everything written from the first
    player's side: vectors, regions, zones, routes, facings. Whether that reading differs
    is the compiler's to say (`CLAUDE.md` rule 4), so this asks it rather than reasoning
    about symmetry here. A rotated spec is rewritten under the default orientation with
    both armies and every wall stated square by square, and if the two compile to the same
    tables the rotation was spelling. Any doubt -- a spec that does not compile, a table
    that differs anywhere -- keeps the orientation as written, which can only split one
    game into two keys, never join two games into one.

    The two compiles are one and a half (Q866): `compile.compile_two_readings` builds the
    first player's movement frames once, since only the second player reads the
    orientation, and stops the restated compile at the first second-player square whose
    entries differ, which is already a table that differs. Memoising could not have done
    it (every design child is another spec), and neither could reflecting the rotated
    reading into the mirrored one here, which is interpreting the DSL outside the
    compiler.
    """
    if written != "rotate":
        return written
    try:
        from chess_variant_sandbox.engine.bridge import canonical_tables
        from chess_variant_sandbox.engine.compile import compile_two_readings

        compiled, stated = compile_two_readings(spec, _restated(spec, written, cells))
        same = stated is not None and _played(canonical_tables(compiled)) == _played(
            canonical_tables(stated)
        )
    except Exception:  # noqa: BLE001 -- any failure to ask is a "keep it as written"
        return written
    return "mirror" if same else written


def _restated(
    spec: dict[str, Any], written: str, cells: set[tuple[int, int, int, str, str]]
) -> dict[str, Any]:
    """``spec`` under the default orientation, both armies and every wall square by square.

    The second spec `_orientation_played` compiles. It changes the spelling of the setup
    and the walls and nothing else, so it differs from ``spec`` only in how the second
    player reads what is written from the first player's side.
    """
    stated = json.loads(json.dumps(spec))
    placements: list[dict[str, Any]] = []
    for player, file_, rank, pid, facing in sorted(cells):
        row: dict[str, Any] = {
            "piece": pid, "player": player, "squares": [format_square(file_, rank)],
        }
        if facing:
            row["facing"] = facing
        placements.append(row)
    stated["setup"] = {"orientation": "mirror", "mirror": False, "placements": placements}
    rules = stated.get("rules") or {}
    obstacles = rules.get("obstacles")
    if isinstance(obstacles, dict) and obstacles.get("squares"):
        board = spec.get("board") or {}
        width = _int(board.get("width"), 8)
        walls = obstacle_squares(
            spec.get("rules"), width, _int(board.get("height"), 8), written
        )
        rules["obstacles"] = {
            **obstacles,
            "mirror": False,
            "squares": sorted(format_square(sq % width, sq // width) for sq in walls),
        }
    return stated


def _played(tables: dict[str, Any]) -> dict[str, Any]:
    """Compiled tables less the Zobrist keys, which are seeded from the spec's own hash.

    They name the spelling rather than the game -- two spellings of one game have two
    hashes and so two sets of keys -- and nothing else in the tables reads the hash.
    """
    return {key: value for key, value in tables.items() if not key.startswith("zobrist_")}


def _int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


# ------------------------------------------------------- renaming, by schema position
#
# Where a piece id may appear, written as the shape of the value that holds it. Renaming
# by *matching the string anywhere* is what the probe did, and it is a guess: a piece may
# be given any id the schema accepts, so a rule whose value happens to spell one -- a
# square name, a count word, an ending -- would be rewritten as a piece reference. These
# say which positions are references and nothing else is touched.
#
# One row per `rules` key that can name a piece, read off `schema.py`'s own checks. A key
# absent here names none, and `tests/test_relate.py` pins that against the validator.

_ID = "<id>"  # this value is one piece id
_IDS = "<ids>"  # a list of piece ids
_KEYED = "<keyed>"  # a map whose *keys* are piece ids
_ANY = "*"  # every key of this map, whatever it is called

_PIECE_POSITIONS: dict[str, Any] = {
    "castling": {"king": _ID, "rook": _ID},
    "promotion": [{"piece": _ID, "to": _IDS, "to_by_player": {_ANY: _IDS}}],
    "reach": {"piece": _ID},
    "extinction": {"pieces": _IDS},
    "explode_on_capture": {"immune": _IDS, "by": _IDS, "of": _IDS},
    "drops": {
        "no_doubled": _IDS, "no_mate": _IDS, "stock": _KEYED, "region": _KEYED,
    },
    "npcs": [{"piece": _ID, "immune": _IDS}],
    # Schema 21: an end's `carries`, on a pair keyed by the end, on a map's end a list.
    "portals": [{"piece": _ID, "carries": {_ANY: _IDS}, "ends": [{"carries": _IDS}]}],
    "strike": {
        "immune": _IDS, "chain": [{"from": _ID, "to": _ID}], "switches_sides": _IDS,
        "by": _IDS,  # A1: the pieces whose captures strike
    },
    "contacts": {
        "immune": _IDS,
        "links": [{"mover": _ID, "onto": _ID, "mover_to": _ID, "onto_to": _ID}],
    },
}

# The same, for the fields on a piece that name another piece. `region` is squares
# and `facing` is an angle, so neither is here. `beats` is schema 26's.
_PIECE_FIELDS: dict[str, Any] = {
    "captures": _IDS, "promoted_from": _ID, "beats": _IDS,
    # Schema 30 (Q999a): a line of sight names the types it touches.
    "sight": [{"affects": _IDS}],
}


def _renamed(value: Any, shape: Any, ids: dict[str, str]) -> Any:
    """A copy of one rule's value with the piece ids at its schema positions replaced.

    `shape` is the row from `_PIECE_POSITIONS`; `None` means this rule names no piece, and
    the value comes back untouched. A list shape applies its one item shape to every item,
    and tolerates the bare-object spelling a list-or-object key allows.
    """
    if shape is None or value is None:
        return value
    if shape is _ID:
        return ids.get(value, value) if isinstance(value, str) else value
    if shape is _IDS:
        if not isinstance(value, list):
            return value
        return [ids.get(item, item) if isinstance(item, str) else item for item in value]
    if shape is _KEYED:
        if not isinstance(value, dict):
            return value
        return {ids.get(key, key): inner for key, inner in value.items()}
    if isinstance(shape, list):
        if isinstance(value, list):
            return [_renamed(item, shape[0], ids) for item in value]
        return _renamed(value, shape[0], ids)
    if isinstance(shape, dict) and isinstance(value, dict):
        wildcard = shape.get(_ANY)
        return {
            key: _renamed(inner, shape.get(key, wildcard), ids)
            for key, inner in value.items()
        }
    return value


def _folded(value: Any, shape: Any, ids: dict[str, str]) -> Any:
    """One rule's value, renamed at its schema positions and then frozen."""
    return _frozen(_renamed(value, shape, ids))


def _frozen(value: Any, ordered: bool = False) -> Any:
    """A hashable, order-free copy of a value that has already been renamed.

    Lists lose their order, because a spec may declare the same set of squares or targets
    in any order and none of them is a different rule. A patrol route is the exception:
    its order is the route, so a `route` key keeps it, and so does any list of plain
    numbers, which is a vector or a pair of coordinates rather than a set.

    **Whatever names a piece only one game has drops out entirely** -- a list item, a dict
    key, a dict entry -- and a list or map left empty by that drops out in turn. That is
    what `transform.drop_piece` does to a spec when a piece goes: a strike link, a
    promotion entry, a patrol, a stock and a `promoted_from` that named it all leave with
    it, and a chain with nothing left in it leaves too, because a rule pointing at nothing
    is a spec the validator refuses. What is left is the rule as it reads without that
    piece. An empty list that was *written* empty stays: `captures: []` means the piece may
    take nothing, which is not the same as not saying.
    """
    if isinstance(value, dict):
        entries = []
        for key, inner in value.items():
            if key == GONE:
                continue
            frozen = _frozen(inner, ordered or key == "route")
            if _names_gone(frozen) or _emptied(inner, frozen):
                continue
            entries.append((key, frozen))
        return tuple(sorted(entries))
    if isinstance(value, list):
        items = [
            frozen
            for inner in value
            if not _names_gone(frozen := _frozen(inner, ordered))
            and not _emptied(inner, frozen)
        ]
        if ordered or all(isinstance(item, (int, float)) for item in items):
            return tuple(items)
        return tuple(sorted(items, key=repr))
    return value


def _names_gone(frozen: Any) -> bool:
    """Whether an already-frozen value mentions a piece only one of the games has."""
    if isinstance(frozen, tuple):
        return any(_names_gone(item) for item in frozen)
    return frozen == GONE


def _emptied(raw: Any, frozen: Any) -> bool:
    """Whether a list or map held something and now holds nothing, because a piece went."""
    return frozen == () and isinstance(raw, (dict, list)) and bool(raw) and _mentions(raw)


def _stated(piece: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Whichever of these keys the piece actually declares.

    An absent key and a key written `null` are the same statement, and a key that leaves
    with a removed piece has to compare equal to one that was never there.
    """
    return {key: piece[key] for key in keys if piece.get(key) is not None}


def _stated_sight(piece: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """`_stated`, with the piece's lines of sight (schema 30) in their folded form."""
    stated = _stated(piece, keys)
    if "sight" in piece:
        stated["sight"] = _sight_form(piece)
    return stated


def _sight_form(piece: dict[str, Any]) -> list[dict[str, Any]]:
    """A piece's lines of sight with their spelling folded out: the directions expanded, the
    defaults written, `affects` kept as a list so a renaming reaches it. Read through
    `schema.piece_sights`, the one reader of the key."""
    out = []
    for sight in piece_sights([piece]).get(str(piece.get("id")), ()):
        dirs: set[tuple[int, int]] = set()
        for vector in sight.vectors:
            dirs |= expand_symmetry(vector, sight.symmetry)
        entry: dict[str, Any] = {
            "dirs": sorted(dirs),
            "range": sight.range,
            "min_range": sight.min_range,
            "effect": sight.effect,
            "whose": sight.whose,
            # A one-step line has nothing between to see through: a ray and an aura there
            # see the same squares and compile to the same table, so they are one game.
            "through": sight.through and sight.range != 1,
        }
        if sight.affects is not None:
            entry["affects"] = list(sight.affects)
        if sight.captures_own:
            entry["captures_own"] = True  # Q1007c; written only when on, so no form moves
        out.append(entry)
    return out


def _live_sights(
    spec: dict[str, Any], pieces: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The pieces, with a `sight` taken off wherever the compiler keeps none (schema 30,
    `docs/archive/plan-passive.md` §4.3): a sight the compiled game does not carry is no rule, as an
    inert `rules` value is none (`_live_rules`). The compiler builds a type's sight entries
    from `schema.piece_sights` and no table at all when that reads none
    (`compile._compile_sights`), so asking the same reader is asking the compiler, without a
    compile per normal form (`tests/test_passive_surfaces.py` holds the two together)."""
    if not has_sights(spec):
        return pieces
    kept = piece_sights(spec)
    return {
        pid: piece if pid in kept or "sight" not in piece
        else {key: value for key, value in piece.items() if key != "sight"}
        for pid, piece in pieces.items()
    }


def _mentions(value: Any) -> bool:
    """Whether an already-renamed value names a piece only one of the games has."""
    if isinstance(value, dict):
        return any(key == GONE or _mentions(inner) for key, inner in value.items())
    if isinstance(value, list):
        return any(_mentions(item) for item in value)
    return value == GONE


def _move_form(piece: dict[str, Any]) -> frozenset[Any]:
    """A piece's movement with symmetries expanded and like descriptors merged.

    Two specs can write one movement many ways -- four vectors or one with a symmetry, one
    descriptor or three with the same attributes -- and none of those is a different rule.
    What is left is a set of (attributes, directions) pairs, which is the movement itself.
    """
    merged: dict[Any, set[tuple[int, int]]] = defaultdict(set)
    for move in piece.get("moves") or []:
        if not isinstance(move, dict):
            continue
        kind = str(move.get("kind", "slide"))
        attrs = (
            kind,
            move.get("mode", "both"),
            move.get("range", DEFAULT_RANGE.get(kind, 1)),
            move.get("min_range", 1),
            _frozen(move.get("condition") or {}),
            move.get("screen_range"),
            move.get("screen"),
            # T1: a slide whose path turns is another movement over the same directions;
            # "straight", the off value, is the key's absence.
            move.get("path", "straight"),
        )
        for vector in move.get("vectors") or []:
            merged[attrs] |= expand_symmetry(
                (int(vector[0]), int(vector[1])), str(move.get("symmetry", "none"))
            )
    return frozenset((attrs, frozenset(dirs)) for attrs, dirs in merged.items())


def _directions(form: frozenset[Any]) -> set[Any]:
    return {(attrs[0], step) for attrs, dirs in form for step in dirs}


def _setup_cells(
    spec: dict[str, Any], width: int, height: int, orientation: str
) -> set[tuple[int, int, int, str, str]]:
    """Every starting piece as (player, file, rank, id, facing), the mirror applied.

    A placement is a list of square names and a spec may write that list in any order, so
    it is expanded to the cells it means before anything compares two setups. A spec that
    states one army and derives the other is expanded too: what differs is the position
    the two games open in, not how few lines each took to say it.
    """
    setup = spec.get("setup") or {}
    cells: set[tuple[int, int, int, str, str]] = set()
    for placement in setup.get("placements") or []:
        if not isinstance(placement, dict) or "piece" not in placement:
            continue
        player = _int(placement.get("player"), 0)
        facing = str(placement.get("facing", ""))
        for name in placement.get("squares") or []:
            square = parse_square(str(name), width, height)
            if square is None:
                continue
            cells.add((player, square[0], square[1], str(placement["piece"]), facing))
            if setup.get("mirror", True):
                there = reflect_square(square[0], square[1], width, height, orientation)
                cells.add(
                    (1 - player, there[0], there[1], str(placement["piece"]), facing)
                )
    return cells


def anchored(
    cells: set[tuple[int, int, int, Any, str]],
    width: int,
    height: int,
    orientation: str,
) -> frozenset[Any]:
    """Each army slid to its own corner, so a setup that moved with the board is one setup.

    Squares in this DSL are named from the first player's corner, so making the board
    bigger leaves the first army where it was and pushes the second one away. A person
    reading the two files says the board changed and the setup did not, and this is what
    lets the count say the same.
    """
    out: set[Any] = set()
    for player in (0, 1):
        army = [cell for cell in cells if cell[0] == player]
        if player == 1:
            army = [
                (p, *reflect_square(x, y, width, height, orientation), piece, facing)
                for p, x, y, piece, facing in army
            ]
        if not army:
            continue
        low_x = min(cell[1] for cell in army)
        low_y = min(cell[2] for cell in army)
        out |= {(p, x - low_x, y - low_y, piece, facing) for p, x, y, piece, facing in army}
    return frozenset(out)


def _live_rules(spec: dict[str, Any]) -> dict[str, Any]:
    """`rules` with the written defaults, the inert values and the board rules removed.

    `wrap` and `obstacles` come out because the board half of this compares them in a
    shape of their own -- `wrap` per axis and per player, obstacles as the squares they
    actually mark -- and a raw comparison of either would read a spelling as a rule.
    """
    rules = spec.get("rules") or {}
    return {
        key: value
        for key, value in rules.items()
        if key not in ("wrap", "obstacles")
        # Schema 26: `"simultaneous": {}` is the whole rule, not an empty one.
        and (value not in (None, False, 0, "", [], {}) or key == "simultaneous")
        and _WRITTEN_DEFAULTS.get(key) != value
    }


def _en_passant_fires(spec: dict[str, Any], pieces: dict[str, dict[str, Any]]) -> bool:
    """Whether this game's rules could ever bring en passant into play.

    En passant answers a two-square move by the promoting piece, so a game with no
    promotion rule, or whose promoting piece cannot double-step, has no en passant to
    have. Asked through `schema.can_double_step`, which is the validator's own reading --
    `transform.settle_en_passant` drops the rule on exactly this test, which is why an
    edit that shortens a pawn's move arrives here with the rule already gone.
    """
    rules = spec.get("rules") or {}
    promotion = rules.get("promotion")
    entries = promotion if isinstance(promotion, list) else [promotion]
    if len(entries) != 1 or not isinstance(entries[0], dict):
        return False
    promoter = pieces.get(str(entries[0].get("piece")))
    return promoter is not None and can_double_step(promoter)


def _wrap_form(spec: dict[str, Any]) -> Any:
    """Which edges are joined, per axis and per player, or `None` where none are."""
    wrap = (spec.get("rules") or {}).get("wrap")
    shape = (spec.get("board") or {}).get("shape")
    out = [tuple(str(edge) for edge in edges) for edges in wrap_stated(wrap, shape).values()]
    return None if all(edge == "none" for pair in out for edge in pair) else tuple(out)


def _wall_form(spec: dict[str, Any], width: int, height: int, orientation: str) -> Any:
    """Which squares are walled off, as squares rather than as the lines that name them.

    A hexagon's absent cells are walls here (Q281), and `board.shape` is read nowhere
    else: the native hexagon compiles to the rhombus cut by walls, so the two are one
    game in two spellings, and one node (`archive/plan-game-graph.md`).
    """
    rules = spec.get("rules") or {}
    squares = obstacle_squares(rules, width, height, orientation) | absent_squares(
        spec.get("board")
    )
    if not squares:
        return None
    return (frozenset(squares), bool((rules.get("obstacles") or {}).get("blocks_leaps")))


def _without_ranks(promotion: Any) -> Any:
    """`rules.promotion` with the rank it fires on taken out, leaving what it does."""
    entries = promotion if isinstance(promotion, list) else [promotion]
    return [
        {key: value for key, value in entry.items() if key not in ("rank", "ranks")}
        if isinstance(entry, dict)
        else entry
        for entry in entries
    ]


# ---------------------------------------------------------------- the piece matching


def _match(
    first: NormalForm, second: NormalForm, stages: tuple[str, ...]
) -> tuple[dict[str, str], list[str], list[str]]:
    """Pair the second game's pieces with the first's, one stage of evidence at a time.

    `same` is letter and movement together, `moves` is movement and royalty, `cells` is
    standing on the same starting squares, `letter` is the id alone, and `closest` pairs
    whatever is left by how many directions they share. Which stages run is the caller's,
    because no single order of evidence is right for every pair: pairing by movement reads
    `almost` as one replaced piece, and pairing by square is what reads `knightmate` as two
    pieces swapping roles rather than as seven edits.

    Returns the mapping from the second game's ids to the first's, and the ids each game
    has that the other does not.
    """
    left_a, left_b = list(first.pieces), list(second.pieces)
    mapping: dict[str, str] = {}

    def take(id_a: str, id_b: str) -> None:
        mapping[id_b] = id_a
        left_a.remove(id_a)
        left_b.remove(id_b)

    if "same" in stages:
        for pid in [
            pid for pid in left_a
            if pid in left_b and first.forms[pid] == second.forms[pid]
        ]:
            take(pid, pid)
    if "moves" in stages:
        for id_a in list(left_a):
            same = [
                id_b for id_b in left_b
                if second.forms[id_b] == first.forms[id_a]
                and bool(second.pieces[id_b].get("royal"))
                == bool(first.pieces[id_a].get("royal"))
            ]
            if same:
                take(id_a, same[0])
    if "cells" in stages:
        shared = sorted(
            (
                (len(first.stands.get(id_a, frozenset())
                     & second.stands.get(id_b, frozenset())), id_a, id_b)
                for id_a in left_a
                for id_b in left_b
            ),
            reverse=True,
        )
        for overlap, id_a, id_b in shared:
            if overlap and id_a in left_a and id_b in left_b:
                take(id_a, id_b)
    if "letter" in stages:
        for pid in [pid for pid in left_a if pid in left_b]:
            take(pid, pid)
    while "closest" in stages and left_a and left_b:
        best = max(
            ((id_a, id_b) for id_a in left_a for id_b in left_b),
            key=lambda pair: _overlap(
                _directions(first.forms[pair[0]]), _directions(second.forms[pair[1]])
            ),
        )
        take(*best)
    return mapping, left_a, left_b


def _overlap(one: set[Any], other: set[Any]) -> float:
    return len(one & other) / len(one | other) if one | other else 1.0


# -------------------------------------------------------------------- the comparison


def relate(first: dict[str, Any], second: dict[str, Any]) -> Relation:
    """What differs between two specs, under the piece matching that finds the fewest.

    Cheap enough to call over a whole library: the per-spec work is memoised and one pair
    costs under a millisecond, so a game's neighbours among seventy others is a twentieth
    of a second the first time and nothing afterwards.
    """
    return relate_forms(normal_form(first), normal_form(second))


def relate_forms(first: NormalForm, second: NormalForm) -> Relation:
    """`relate` over two normal forms already in hand, memoised on their identity keys.

    And on the pieces' names: two spellings of one game share a key, and a sentence read
    for one of them would name the other's pieces.
    """
    if first.key == second.key:
        return Relation(())
    pair = (first.key, first.names, second.key, second.names)
    held = _PAIRS.get(pair)
    if held is None:
        # Whichever way of pairing the two armies finds the fewest rules. A tie keeps the
        # first, so the reading is the same twice running.
        fewest: list[Difference] = _compare(first, second, _MATCHINGS[0])
        for stages in _MATCHINGS[1:]:
            reading = _compare(first, second, stages)
            if len(reading) < len(fewest):
                fewest = reading
        held = Relation(tuple(fewest))
        if len(_PAIRS) >= _PAIRS_KEPT:
            _PAIRS.clear()
        _PAIRS[pair] = held
    return held


def units(first: dict[str, Any], second: dict[str, Any]) -> int:
    """How many rules two specs differ in."""
    return relate(first, second).units


# Every pair compared in this process, by the two normal forms' identity keys. Bounded
# and cleared whole rather than evicted one at a time: the cost of a miss is a
# millisecond, and a library of 500 games is 125,000 pairs, which is what the bound is
# sized for. `RELATE_VERSION` is not in the key because a process runs one version of
# this file; a *stored* link carries it.
_PAIRS: dict[tuple[Any, ...], Relation] = {}
_PAIRS_KEPT = 200_000


def _compare(
    first: NormalForm, second: NormalForm, stages: tuple[str, ...]
) -> list[Difference]:
    """One reading of what differs, under one way of pairing the two armies.

    One difference is one thing a person would call a rule. Where an edit carries a
    consequence, the consequence is charged to its cause rather than counted twice -- each
    such fold is marked below, and `docs/archive/plan-game-graph.md` §2 is the table of
    what each one repairs.
    """
    found: list[Difference] = []
    _board_differences(first, second, found)

    mapping, only_a, only_b = _match(first, second, stages)
    back = {id_a: id_b for id_b, id_a in mapping.items()}
    gone_a = set(only_a)
    gone_b = set(only_b)
    ids_a: dict[str, str] = {pid: GONE for pid in gone_a}
    ids_b: dict[str, str] = {**mapping, **{pid: GONE for pid in gone_b}}

    for pid in sorted(gone_a):
        name = str(first.pieces[pid].get("name", pid))
        found.append(Difference(
            "pieces", "piece.removed", f"it has no {name}"
        ))
    for pid in sorted(gone_b):
        name = str(second.pieces[pid].get("name", pid))
        found.append(Difference(
            "pieces", "piece.added", f"it also has a {name}"
        ))

    turns = False
    takers: list[str] = []
    for id_a, id_b in sorted(back.items()):
        one, other = first.pieces[id_a], second.pieces[id_b]
        name = str(one.get("name", id_a))
        words = _PairWords(name, str(other.get("name", id_b)))
        moved = first.forms[id_a] != second.forms[id_b]
        crowned = bool(one.get("royal")) != bool(other.get("royal"))
        if moved:
            found.append(Difference("pieces", "piece.moves", words.moves()))
        if crowned:
            found.append(Difference(
                "ending", "piece.royal", words.royal(bool(other.get("royal")))
            ))
        # A price follows what a piece is. A piece whose movement or royalty changed and
        # whose price changed with it is one edit, not two.
        if one.get("value") != other.get("value") and not (moved or crowned):
            found.append(Difference("valuation", "piece.value", words.priced()))
        if _folded(one.get("captures"), _IDS, ids_a) != _folded(
            other.get("captures"), _IDS, ids_b
        ):
            takers.append(name)
        if _frozen(one.get("facing")) != _frozen(other.get("facing")):
            turns = True
            found.append(Difference("pieces", "piece.facing", words.turns()))
        # Schema 27 (Q935a): a spy is a fact about the piece that decides the turn; schema
        # 28's speed (Q943a) one that decides when its move is made in a round; schema 29's
        # rest (Q983a) one that decides when it may move again.
        # Schema 30's sight (Q999a) one that decides what the pieces it sees may do.
        rest = ("region", "promoted_from", "beats", "spy", "speed", "rest")
        if _folded(_stated_sight(one, rest), _PIECE_FIELDS, ids_a) != _folded(
            _stated_sight(other, rest), _PIECE_FIELDS, ids_b
        ):
            found.append(Difference("pieces", "piece.other", words.stands()))
    # Who may take whom is one rule however many pieces carry a list. `chess / wolves`
    # declares it on five pieces and costs 55 JSON leaves for one decision.
    if takers:
        found.append(Difference(
            "capture",
            "piece.captures",
            f"who may take whom, declared on {len(takers)} piece"
            f"{'' if len(takers) == 1 else 's'}",
            len(takers),
        ))

    _setup_differences(first, second, mapping, gone_a, gone_b, turns, found)
    _rule_differences(first, second, ids_a, ids_b, found)
    return found


@dataclass(frozen=True)
class _PairWords:
    """The sentences about one matched pair of pieces, from the second game's side.

    Pieces are matched by what they do, not by what they are called, so a pair can carry
    two names: chess's knight against a draft's *diagonal knight*. A sentence reads as
    what the second game does differently (`Difference`), so where the names differ it
    names the second game's piece as *its* and the first game's beside it -- *its knight
    moves differently from the diagonal knight* on the draft's page, under chess. Naming
    only one would name the page's own piece in a row about the other game, which is
    what F9 of `docs/reference/first-visit-study.md` found. Where the names agree the
    sentence names the piece once, as it always has.
    """

    first: str
    second: str

    @property
    def same(self) -> bool:
        return self.first.strip().casefold() == self.second.strip().casefold()

    def moves(self) -> str:
        if self.same:
            return f"the {self.first} moves differently"
        return f"its {self.second} moves differently from the {self.first}"

    def royal(self, second_is: bool) -> str:
        if self.same:
            return f"whether the {self.first} is royal"
        if second_is:
            return f"its {self.second} is royal where the {self.first} is not"
        return f"its {self.second} is not royal where the {self.first} is"

    def priced(self) -> str:
        if self.same:
            return f"the {self.first} is priced differently"
        return f"its {self.second} is priced differently from the {self.first}"

    def turns(self) -> str:
        if self.same:
            return f"whether the {self.first} turns"
        return f"its {self.second} turns differently from the {self.first}"

    def stands(self) -> str:
        if self.same:
            return f"where the {self.first} may stand"
        return f"its {self.second} may stand in other places than the {self.first}"


def _board_differences(
    first: NormalForm, second: NormalForm, found: list[Difference]
) -> None:
    if first.size != second.size:
        found.append(Difference(
            "board",
            "board.size",
            f"its board is {second.size[0]}x{second.size[1]} rather than "
            f"{first.size[0]}x{first.size[1]}",
            abs(first.size[0] * first.size[1] - second.size[0] * second.size[1]),
        ))
    if first.lattice != second.lattice:
        found.append(Difference("board", "board.lattice", "the squares are laid out differently"))
    if first.wrap != second.wrap:
        found.append(Difference("board", "board.wrap", "which edges are joined"))
    # Walls are compared as squares, which only means the same thing on one board. Where
    # the boards differ, all that can honestly be said is whether one of them has walls.
    if first.size == second.size:
        if first.walls != second.walls:
            found.append(Difference("board", "board.walls", "which squares are walled off"))
    elif (first.walls is None) != (second.walls is None):
        found.append(Difference("board", "board.walls", "only one of them has walls"))


def _setup_differences(
    first: NormalForm,
    second: NormalForm,
    mapping: dict[str, str],
    gone_a: set[str],
    gone_b: set[str],
    turns: bool,
    found: list[Difference],
) -> None:
    """Where the pieces start, as cells rather than as placement lines.

    **A piece only one game has takes its squares with it, and the squares it took from
    somebody else.** A split gives a new piece some of another piece's copies, so the
    parent has the old piece on those squares and the child has the new one -- one edit,
    which without this reads as the piece plus a changed setup. So every square a
    one-sided piece stands on is set aside in *both* games, and what is left is the rest
    of the setup.
    """
    displaced = {
        (player, x, y)
        for player, x, y, pid, _facing in first.cells
        if pid in gone_a
    } | {
        (player, x, y)
        for player, x, y, pid, _facing in second.cells
        if pid in gone_b
    }
    cells_a = {
        cell for cell in first.cells
        if cell[3] not in gone_a and cell[:3] not in displaced
    }
    cells_b = {
        (player, x, y, mapping.get(pid, pid), facing)
        for player, x, y, pid, facing in second.cells
        if pid not in gone_b and (player, x, y) not in displaced
    }
    # A facing written on a placement belongs to the facing rule when that rule differs.
    if turns:
        cells_a = {(*cell[:4], "") for cell in cells_a}
        cells_b = {(*cell[:4], "") for cell in cells_b}

    if first.size == second.size:
        apart = len(cells_a ^ cells_b)
    else:
        apart = len(
            anchored(cells_a, *first.size, first.orientation)
            ^ anchored(cells_b, *second.size, second.orientation)
        )
    if apart:
        found.append(Difference(
            "setup", "setup.cells", f"where the pieces start, in {apart} cells", apart
        ))
    if first.orientation != second.orientation:
        found.append(Difference(
            "setup", "setup.orientation", "one army is mirrored and the other rotated"
        ))


def _rule_differences(
    first: NormalForm,
    second: NormalForm,
    ids_a: dict[str, str],
    ids_b: dict[str, str],
    found: list[Difference],
) -> None:
    """One key under `rules` is one rule, with two folds.

    **A rule that names a piece only one game has folds into that piece.** Removing a
    piece takes castling, promotion and a drop's stock with it -- `transform.drop_piece`
    does exactly that, because a rule pointing at nothing is a spec the validator refuses
    -- and adding a stone brings the hand that is the only way it reaches the board. Those
    are one edit and its consequence, so the rule is not counted a second time.

    **En passant is compared only where both games could bring it into play.** It answers
    a two-square move by the promoting piece; where one game's promoting piece cannot make
    one, the difference is already charged to that piece's movement or to its removal.

    **A promotion rank follows the board.** Where the two boards differ and the only thing
    promotion says differently is which rank it fires on, that rank is part of the resize:
    a board with a new far edge has to settle where the walk ends, and both answers --
    moving the rank with the edge and leaving it where it was -- are that one decision.
    """
    for key in sorted(first.rules.keys() | second.rules.keys()):
        if key == "en_passant" and not (first.en_passant_fires and second.en_passant_fires):
            continue
        shape = _PIECE_POSITIONS.get(key)
        one = _renamed(first.rules.get(key), shape, ids_a)
        other = _renamed(second.rules.get(key), shape, ids_b)
        if _frozen(one) == _frozen(other):
            continue
        if (key not in second.rules and _mentions(one)) or (
            key not in first.rules and _mentions(other)
        ):
            continue
        if (
            key == "promotion"
            and first.size != second.size
            and one is not None
            and other is not None
            and _frozen(_without_ranks(one)) == _frozen(_without_ranks(other))
        ):
            continue
        found.append(Difference(
            RULE_KINDS.get(key, "ending"), f"rules.{key}", _rule_sentence(key)
        ))


def _rule_sentence(key: str) -> str:
    """One `rules` key in words, for a link a person reads.

    What the rule is *about*, not what either game says about it. The second is the prose
    diff `docs/plan-designer-ux.md` C3 asks for and is built once, on `describe.py`; a
    link only has to say which rule the two games answer differently.
    """
    return RULE_WORDS.get(key) or f"the {key.replace('_', ' ')} rule"


# ---------------------------------------------------------------- the similarity index
#
# `units` counts steps and every step counts one. The index ranks instead, over four
# tiers a person weighs differently: which rules a game involves, which movement
# mechanisms its pieces have, which pieces it has, and where they stand. Each tier weighs
# `SIMILARITY_RATIO` times the tier below, so eight squares equal one piece, eight pieces
# one mechanism and eight mechanisms one rule. `docs/archive/plan-game-graph.md` §9.2
# defines every count and §9.4 holds the pre-registered check the ratio was chosen against.
#
# Both live side by side. A link exists because two games are one, two or three rules
# apart, which is `units`; the index orders the links and explains them.

# How much more each tier weighs than the tier below it. One constant, and nothing else
# depends on its value -- §9.3 puts the steepness to the user and 8 is the answer the
# check passed at.
SIMILARITY_RATIO = 8

# A rule both games involve and set differently, against a rule only one of them involves.
# Without it a family that differs in one setting ties with every one-rule variant.
SETTING_WEIGHT = 0.5

_RULE_WEIGHT = SIMILARITY_RATIO**3

# The three clocks are the harness's rather than the game's, so they leave tier 1 and are
# weighed beside a piece's price, one tier-4 difference each.
_CLOCK_KEYS = frozenset(key for key, kind in RULE_KINDS.items() if kind == "clock")

# Each count, with the noun the sentence gives it. The first is always said, because *no
# rules* is the most important thing a reading can report; the rest are said when they are
# not zero.
_TIER_WORDS: tuple[tuple[str, str, str], ...] = (
    ("rules", "rule", "rules"),
    ("settings", "rule set differently", "rules set differently"),
    ("mechanisms", "mechanism", "mechanisms"),
    ("pieces", "piece", "pieces"),
    ("squares", "square", "squares"),
    ("books", "clock or price", "clocks or prices"),
)


@dataclass(frozen=True)
class Similarity:
    """How alike two games are, as one number over six counts.

    `rules` is a rule one game involves and the other does not, `settings` a rule both
    involve and set differently, `mechanisms` a movement mechanism one side's pieces have
    and the other's do not, `pieces` a piece identity one side has, `squares` a starting
    square whose occupant differs, and `books` a draw clock or a piece's price.

    Symmetric, and 0 exactly when the two normal forms are one.
    """

    rules: int
    settings: int
    mechanisms: int
    pieces: int
    squares: int
    books: int

    @property
    def index(self) -> float:
        """The tiers weighed together. Lower is nearer, and 0 is the same game."""
        return (
            _RULE_WEIGHT * (self.rules + SETTING_WEIGHT * self.settings)
            + SIMILARITY_RATIO**2 * self.mechanisms
            + SIMILARITY_RATIO * self.pieces
            + self.squares
            + self.books
        )

    @property
    def counts(self) -> dict[str, int]:
        return {name: getattr(self, name) for name, _, _ in _TIER_WORDS}

    @property
    def sentence(self) -> str:
        """The counts in words, as *no rules, 6 mechanisms, 1 piece*."""
        said = []
        for position, (name, one, many) in enumerate(_TIER_WORDS):
            number = getattr(self, name)
            if number or position == 0:
                count = str(number) if number else "no"
                said.append(f"{count} {one if number == 1 else many}")
        return ", ".join(said)

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "counts": self.counts, "words": self.sentence}


@dataclass(frozen=True)
class _Reading:
    """What one game brings to a comparison, built once and reused across its pairs."""

    rules: frozenset[str]
    mechanisms: frozenset[Any]
    identities: dict[str, Any]
    kinds: frozenset[Any]
    # Who stands where, each piece labelled by what it is, as written and slid to the
    # corner. Which one a pair uses depends on whether the two boards are the same size.
    stands: dict[tuple[int, int, int], Any]
    slid: dict[tuple[int, int, int], Any]


# Every normal form read this process, by identity key. Bounded and cleared whole, as
# `_PAIRS` is, and for the same reason: a miss costs a fraction of a millisecond.
_READINGS: dict[str, _Reading] = {}
_READINGS_KEPT = 5_000


def rules_involved(form: NormalForm) -> frozenset[str]:
    """Tier 1. Which rules this game involves, the clocks apart.

    The live `rules` keys, plus the three things declared elsewhere that change what a
    move is -- joined edges, walls and a lattice other than square -- plus who may take
    whom when any piece declares it. A set of keys has no board and no piece count in it.
    """
    involved = {key for key in form.rules if key not in _CLOCK_KEYS}
    if form.wrap is not None:
        involved.add("board.wrap")
    if form.walls is not None:
        involved.add("board.walls")
    if form.lattice != "square":
        involved.add(f"board.lattice.{form.lattice}")
    if any(piece.get("captures") is not None for piece in form.pieces.values()):
        involved.add("piece.captures")
    # Schema 27 (Q935a): a spy changes what a turn is, as `rules.simultaneous` does.
    if any(piece.get("spy") is True for piece in form.pieces.values()):
        involved.add("piece.spy")
    # Schema 28 (Q943a): speeds change how a round is resolved, so they are a rule too.
    if any("speed" in piece for piece in form.pieces.values()):
        involved.add("piece.speed")
    # Schema 29 (Q983a): a rest changes which piece may make a turn's move, so it is a rule.
    if any("rest" in piece for piece in form.pieces.values()):
        involved.add("piece.rest")
    # Schema 30 (Q999a): a line of sight changes what the pieces it sees may do, so it is a
    # rule; held only where the compiler keeps it (`_live_sights`).
    if any("sight" in piece for piece in form.pieces.values()):
        involved.add("piece.sight")
    return frozenset(involved)


def mechanisms(form: NormalForm) -> frozenset[Any]:
    """Tier 2. Every movement mechanism any piece of this game has.

    One mechanism is one set of move attributes over one orbit of directions, with the
    directions of that orbit the piece actually has; the orbit of `(dx, dy)` is the sorted
    pair of absolute values. So a piece whose movement is two mechanisms the game already
    has adds none, and a game's set of mechanisms holds no board and no piece count.
    """
    found: set[Any] = set()
    for moves in form.forms.values():
        for attrs, dirs in moves:
            orbits: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
            for dx, dy in dirs:
                low, high = sorted((abs(dx), abs(dy)))
                orbits[(low, high)].add((dx, dy))
            for orbit, members in orbits.items():
                found.add((attrs, orbit, frozenset(members)))
    return frozenset(found)


def piece_identity(form: NormalForm, pid: str) -> tuple[Any, ...]:
    """Tier 3. What one piece is, with its letter, its name and its price left out.

    Its whole movement, whether it is royal, which way it may face and where it may
    stand. A region is compared as written, so one palace on two board sizes reads as two
    identities.
    """
    piece = form.pieces[pid]
    return (
        form.forms[pid],
        bool(piece.get("royal")),
        _frozen(piece.get("facing")),
        _frozen(piece.get("region")),
    )


def similarity(first: dict[str, Any], second: dict[str, Any]) -> Similarity:
    """How alike two specs are. Reads the two files and asks the engine nothing."""
    return similarity_forms(normal_form(first), normal_form(second))


def similarity_forms(first: NormalForm, second: NormalForm) -> Similarity:
    """`similarity` over two normal forms already in hand."""
    if first.key == second.key:
        return Similarity(0, 0, 0, 0, 0, 0)
    one, other = _reading(first), _reading(second)
    relation = relate_forms(first, second)
    differing = {difference.unit for difference in relation.differences}

    # Tier 1, in two parts. A rule one game involves and the other does not, then a rule
    # both involve and set differently, which `relate` has already found.
    rules = len(one.rules ^ other.rules)
    settings = sum(1 for key in one.rules & other.rules if _unit_of(key) in differing)

    # Tier 3. A replaced piece is one difference and an added piece is one, so the larger
    # of the two one-sided counts is what a person would say changed.
    pieces = max(len(one.kinds - other.kinds), len(other.kinds - one.kinds))

    # Tier 4. A square of one army whose occupant differs, empty against occupied
    # included. A square held by an identity only one game has is set aside, as
    # `_setup_differences` sets it aside, or a changed pawn would carry sixteen squares.
    shared = one.kinds & other.kinds
    apart = 0
    if first.size == second.size:
        here, there = one.stands, other.stands
    else:
        here, there = one.slid, other.slid
    for square in here.keys() | there.keys():
        mine, yours = here.get(square), there.get(square)
        if (mine and mine[0] not in shared) or (yours and yours[0] not in shared):
            continue
        apart += mine != yours
    apart += first.size != second.size
    apart += first.orientation != second.orientation

    # What the four tiers do not name: the draw clocks and a piece's price, one difference
    # each. Counted over the differences rather than over their units, because two pieces
    # priced differently are two.
    books = sum(
        1
        for difference in relation.differences
        if difference.unit == "piece.value"
        or difference.unit.removeprefix("rules.") in _CLOCK_KEYS
    )
    return Similarity(
        rules=rules,
        settings=settings,
        mechanisms=len(one.mechanisms ^ other.mechanisms),
        pieces=pieces,
        squares=apart,
        books=books,
    )


def _unit_of(key: str) -> str:
    """One entry of `rules_involved` as the unit `relate` would report it under."""
    return key if key.startswith(("board.", "piece.")) else f"rules.{key}"


def _reading(form: NormalForm) -> _Reading:
    held = _READINGS.get(form.key)
    if held is None:
        identities = {pid: piece_identity(form, pid) for pid in form.pieces}
        labelled = {
            (player, x, y, identities[pid], facing)
            for player, x, y, pid, facing in form.cells
        }
        held = _Reading(
            rules=rules_involved(form),
            mechanisms=mechanisms(form),
            identities=identities,
            kinds=frozenset(identities.values()),
            stands=_by_square(labelled),
            slid=_by_square(anchored(labelled, *form.size, form.orientation)),
        )
        if len(_READINGS) >= _READINGS_KEPT:
            _READINGS.clear()
        _READINGS[form.key] = held
    return held


def _by_square(cells: Any) -> dict[tuple[int, int, int], Any]:
    return {(player, x, y): (label, facing) for player, x, y, label, facing in cells}
