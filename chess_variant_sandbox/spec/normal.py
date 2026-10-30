"""A game's name: the hash of its normal form, or the name it already had (Q1008a).

Until Q1008a a game was named by the hash of its file's mechanics as written, so two
spellings of one game were two games, and a file could never be re-spelled without renaming
its game (`docs/reference/schema-reboot-assessment.md` §1). Now:

* **The normal form** (`normal_form`) is the game with the spelling taken out: no
  ``schema_version``, no rule written at the value that means "no rule", no key written at
  its default, and one spelling where the language has two for one meaning. Every rewrite is
  listed in `normal_form`'s docstring and each is one the compiler cannot tell apart, which
  `tests/test_normal_form.py` holds table for table over the library.
* **The legacy table** (`legacy_names.LEGACY_NAMES`) maps the normal-form hash of every game
  that had a name before Q1008a to that name. It was generated once, with the spelling hash,
  and is never regenerated from the files, which Q1008b re-spells.
* **A game's identity** (`canonical.spec_hash`) is its entry in that table if it has one, else
  its normal-form hash. No game that had a name changed it, and the position keys are seeded
  from the identity, so every existing game compiles to the tables it always did.

The normal form is a *hashing* form. It is a valid spec once a ``schema_version`` is put back,
but it resolves what a file may leave to the board (``rank: "last"`` becomes the rank it is), so
it is not necessarily the spelling a person or an operator should keep.

Two pairs the assessment listed as second spellings are deliberately **not** merged, because
they are not one meaning: ``strike`` and ``contacts`` differ in a game in rounds (a collision's
loser goes through ``strike`` and never through ``contacts``, `compile.SimRule`), and a portal
pair and a two-end ``ends`` group name their ends differently (``1o``/``1b`` against
``1a``/``1b``), which is the move text of every stored portal game.
"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chess_variant_sandbox.spec.canonical import canonical_json, hash_text, spelling_hash
from chess_variant_sandbox.spec.legacy_names import LEGACY_NAMES
from chess_variant_sandbox.spec.schema import (
    CONTACT_LINK_OFF_VALUES,
    DEFAULT_RANGE,
    DUCK_OFF_VALUES,
    MOVE_OFF_VALUES,
    NPC_OFF_VALUES,
    PIECE_OFF_VALUES,
    PORTAL_OFF_VALUES,
    PURSE_OFF_VALUES,
    SIGHT_OFF_VALUES,
    SIMULTANEOUS_OFF_VALUES,
    insufficient_material_live,
    portal_stand_in_shape,
    stated_off,
    strike_is_live,
    sweep_range_stated_off,
    wrap_stated,
)

# A rules key written at one of these is the rule's absence: the compiler reads the key with
# `if not value` (or a default of `False`/`0`), so `"en_passant": false`, `"drops": {}` and
# `"check_limit": 0` are each the game without the key. `simultaneous` is not here (`{}` is the
# rule, Q918a), nor are the three clocks, whose absence is a number rather than nothing.
_ABSENT_WHEN_EMPTY = frozenset({
    "promotion", "en_passant", "castling", "bare_royal", "check_limit", "reach",
    "extinction", "must_capture", "forbid_check", "drops", "explode_on_capture",
    "forbid_immobility", "scoring", "territory", "turn", "wrap", "obstacles", "strike",
    "contacts", "npcs", "promote_on_entry", "portals", "insufficient_material",
})
_EMPTY: tuple[Any, ...] = (False, None, 0, "", [], {})

# Keys at the value the compiler reads when they are absent (`compile.py`'s `.get` defaults).
_RULE_DEFAULTS: dict[str, Any] = {
    "win": "checkmate", "stalemate": "draw", "checkmate": "loss", "royal_facing": "allowed",
    "max_plies": 400, "repetition_draw": 3, "no_progress_draw": 100,
}
_BOARD_DEFAULTS: dict[str, Any] = {"lattice": "square", "shape": "grid"}
_SETUP_DEFAULTS: dict[str, Any] = {"mirror": True, "orientation": "mirror"}
_PLACEMENT_DEFAULTS: dict[str, Any] = {"player": 0}
_PIECE_DEFAULTS: dict[str, Any] = {"royal": False}
_MOVE_DEFAULTS: dict[str, Any] = {
    "symmetry": "none", "min_range": 1, "mode": "both", "screen": "any", "screen_range": 0,
}
_CONDITION_DEFAULTS: dict[str, Any] = {"first_move": False}
_PROMOTION_DEFAULTS: dict[str, Any] = {"mandatory": True, "on_leaving": False, "only_lost": False}


def normal_form(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec``'s mechanics with the spelling taken out. Never mutates ``spec``.

    The rewrites, each one the compiler reads identically:

    1. Only the mechanics (`canonical.MECHANIC_KEYS`), and not ``schema_version``.
    2. A key at its default is dropped: on the board (``lattice: "square"``, ``shape:
       "grid"``), the setup (``mirror: true``, ``orientation: "mirror"``), a placement
       (``player: 0``), a piece (``royal: false``), a move (``symmetry: "none"``, ``range`` at
       its kind's `schema.DEFAULT_RANGE`, ``min_range: 1``, ``mode: "both"``, ``screen:
       "any"``, ``screen_range: 0``, an empty ``condition`` and ``first_move: false``), a
       promotion entry (``mandatory: true``, ``on_leaving: false``, ``only_lost: false``) and
       the rules (`_RULE_DEFAULTS`).
    3. A rule that states nothing is dropped: one of `_ABSENT_WHEN_EMPTY` at ``false``,
       ``0``, ``{}`` or ``[]``; a ``turn`` of one move; a ``wrap`` where every edge is
       ``"none"``; a ``strike`` that `schema.strike_is_live` says is plain capture.
    4. One spelling where there are two: ``promotion`` is always a list; a promotion zone by
       rank is always ``ranks``, ascending, with ``"last"`` and the default resolved against
       the board's height; ``to_by_player`` naming one list for both players is ``to``; a
       ``wrap`` is written key by key from `schema.wrap_stated`, ``X_by_player`` only where
       the two players differ.
    5. A key written at its off value is dropped (`drop_off_values`, Q1008c): before
       Q1008c the validator refused each of these, so no game that had a name writes one.
    6. Portal moves are written in the shape every portal game was named in (spelling 31,
       Q1007b, `docs/archive/plan-schema-31.md` §4.3, `_portal_moves_as_pieces`): each group's
       ``moves`` as a piece appended to ``pieces`` (`schema.portal_movement_types`) that the
       group names as its ``piece``. Not an authoring spelling, which the validator refuses;
       the old file and its conversion are one game under one name.

    Each rewrite applies only where its input is well formed, so a malformed spec keeps what
    the validator will name, and the function never raises on JSON.
    """
    return _normalise(json.loads(canonical_json(spec, mechanics_only=True)))


def normal_hash(spec: dict[str, Any]) -> str:
    """The hash of ``spec``'s normal form: a new game's name (`canonical.spec_hash`)."""
    return _normal_hash_of(canonical_json(spec, mechanics_only=True))


def identity(spec: dict[str, Any]) -> str:
    """``spec``'s name: its legacy name if it had one, else its normal-form hash."""
    return _identity_of(canonical_json(spec, mechanics_only=True))


# Both read the mechanics as written, which is also what the spelling hash reads, so a spec
# seen once costs its serialisation thereafter: the library is named on every page.
@functools.lru_cache(maxsize=8192)
def _identity_of(mechanics: str) -> str:
    normal = _normal_hash_of(mechanics)
    return LEGACY_NAMES.get(normal, normal)


@functools.lru_cache(maxsize=8192)
def _normal_hash_of(mechanics: str) -> str:
    return hash_text(_canonical(_normalise(json.loads(mechanics))))


def _normalise(normal: dict[str, Any]) -> dict[str, Any]:
    """`normal_form`'s rewrites, made in place on a fresh copy of the mechanics."""
    normal.pop("schema_version", None)
    _portal_moves_as_pieces(normal)
    board = normal.get("board")
    if isinstance(board, dict):
        _drop_defaults(board, _BOARD_DEFAULTS)
    setup = normal.get("setup")
    if isinstance(setup, dict):
        _drop_defaults(setup, _SETUP_DEFAULTS)
        for placement in _dicts(setup.get("placements")):
            _drop_defaults(placement, _PLACEMENT_DEFAULTS)
    for piece in _dicts(normal.get("pieces")):
        _drop_defaults(piece, _PIECE_DEFAULTS)
        for move in _dicts(piece.get("moves")):
            _normal_move(move)
    rules = normal.get("rules")
    if isinstance(rules, dict):
        height = board.get("height") if isinstance(board, dict) else None
        shape = board.get("shape") if isinstance(board, dict) else None
        _normal_rules(rules, height, shape)
    drop_off_values(normal)
    # Q1069a: the draw by insufficient material written on a game that is not chess
    # underneath states nothing (`schema.insufficient_material_live`, read off this form).
    if (
        isinstance(rules, dict) and "insufficient_material" in rules
        and not insufficient_material_live(normal, normalised=True)
    ):
        del rules["insufficient_material"]
    return normal


def _portal_moves_as_pieces(spec: dict[str, Any]) -> None:
    """Spelling 31's group ``moves`` written, in place, as spelling 30's stand-in piece
    (`schema.portal_stand_in_shape`, the one helper the relations and the rules text read).

    Every portal game git ever held (59 specs, one shape) declared one stand-in, ``{"id":
    "O", "name": "portal", "value": 0, "moves": [...]}``, last in ``pieces`` and named by
    every group as its ``piece``, with ``"mode": "move"`` on each move. Spelling 31 moves the
    list onto the groups; this puts it back, so a converted file hashes to the normal form
    its old spelling had and the legacy table still finds its name. A spec that already
    names a stand-in (the old spelling, as a stored spec may still be) is its own hashing
    shape and is left as it is.
    """
    shaped = portal_stand_in_shape(spec)
    if shaped is not spec:
        spec["pieces"] = shaped["pieces"]
        spec["rules"] = shaped["rules"]


def drop_off_values(spec: dict[str, Any]) -> None:
    """Take out, in place, every key written at the value that states no rule (Q1008c).

    `schema`'s ``*_OFF_VALUES``: a piece type's ``rest: 0``, ``speed: 0``, ``spy: false`` and
    ``sight: []``, a move's ``path: "straight"`` (T1), a line of sight's ``through:
    false``, ``rules.simultaneous``'s ``tie: "both"``, ``swap: "chase"`` and ``both:
    "draw"``, a portal group's ``kind: "momentum"``, a contact link's ``own: false`` and a wild-piece
    entry's ``respawn: 0`` (N1) and ``turns_back: false`` (N5), and ``rules.duck``'s ``goes:
    "anywhere"`` (N4b), and ``rules.purse``'s ``bribe: false`` (TC6), and a sweep's ``sweep_range``
    equal to its ``range`` (Q1063, `schema.sweep_range_stated_off`). The validator accepts each and every reader takes it
    as the key's absence. Shared by the normal form and `spelling.current_spelling`, so a
    spec that writes one is named as, and tidied to, the game without it.
    """
    for piece in _dicts(spec.get("pieces")):
        for entry in _dicts(piece.get("sight")):
            _drop_off(entry, SIGHT_OFF_VALUES)
        _drop_off(piece, PIECE_OFF_VALUES)
        for move in _dicts(piece.get("moves")):
            _drop_off(move, MOVE_OFF_VALUES)
            # Q1063: a sweep's ``sweep_range`` at its default, the move's own range.
            if sweep_range_stated_off(move):
                del move["sweep_range"]
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return
    if isinstance(rules.get("simultaneous"), dict):
        _drop_off(rules["simultaneous"], SIMULTANEOUS_OFF_VALUES)
    for group in _dicts(rules.get("portals")):
        _drop_off(group, PORTAL_OFF_VALUES)
        for move in _dicts(group.get("moves")):
            _drop_off(move, MOVE_OFF_VALUES)
    contacts = rules.get("contacts")
    if isinstance(contacts, dict):
        for link in _dicts(contacts.get("links")):
            _drop_off(link, CONTACT_LINK_OFF_VALUES)
    for npc in _dicts(rules.get("npcs")):
        _drop_off(npc, NPC_OFF_VALUES)
    if isinstance(rules.get("duck"), dict):
        _drop_off(rules["duck"], DUCK_OFF_VALUES)
    if isinstance(rules.get("purse"), dict):
        _drop_off(rules["purse"], PURSE_OFF_VALUES)


def _drop_off(where: dict[str, Any], off_values: dict[str, Any]) -> None:
    for key in [key for key in where if stated_off(where, key, off_values)]:
        del where[key]


# ------------------------------------------------------------------------------ rewrites


def _normal_move(move: dict[str, Any]) -> None:
    kind = move.get("kind")
    if isinstance(kind, str) and kind in DEFAULT_RANGE:
        _drop_defaults(move, {"range": DEFAULT_RANGE[kind]})
    _drop_defaults(move, _MOVE_DEFAULTS)
    condition = move.get("condition")
    if isinstance(condition, dict):
        _drop_defaults(condition, _CONDITION_DEFAULTS)
        if not condition:
            del move["condition"]


def _normal_rules(rules: dict[str, Any], height: Any, shape: Any) -> None:
    if isinstance(rules.get("promotion"), dict):
        rules["promotion"] = [rules["promotion"]]
    for entry in _dicts(rules.get("promotion")):
        _normal_promotion(entry, height)
    if isinstance(rules.get("wrap"), dict):
        rules["wrap"] = _normal_wrap(rules["wrap"], shape)
    turn = rules.get("turn")
    if isinstance(turn, dict) and _one_move_turn(turn):
        del rules["turn"]
    if isinstance(rules.get("strike"), dict) and not strike_is_live(rules["strike"]):
        del rules["strike"]
    _drop_defaults(rules, _RULE_DEFAULTS)
    for key in [key for key in rules if key in _ABSENT_WHEN_EMPTY]:
        if any(_same(rules[key], empty) for empty in _EMPTY):
            del rules[key]


def _normal_promotion(entry: dict[str, Any], height: Any) -> None:
    _drop_defaults(entry, _PROMOTION_DEFAULTS)
    by_player = entry.get("to_by_player")
    if (
        "to" not in entry
        and isinstance(by_player, dict)
        and set(by_player) == {"0", "1"}
        and _same(by_player["0"], by_player["1"])
    ):
        entry["to"] = by_player["0"]
        del entry["to_by_player"]
    if "squares" in entry or ("rank" in entry and "ranks" in entry):
        return
    if "ranks" in entry:
        ranks = entry["ranks"]
        if isinstance(ranks, list) and all(_is_int(rank) for rank in ranks):
            entry["ranks"] = sorted(set(ranks))
        return
    rank = entry.get("rank", "last")
    if rank == "last" and _is_int(height):
        rank = height - 1
    if _is_int(rank):
        entry.pop("rank", None)
        entry["ranks"] = [rank]


def _normal_wrap(wrap: dict[str, Any], shape: Any) -> dict[str, Any]:
    """``wrap`` written one way: each key's edge, or both players' where they differ."""
    stated = wrap_stated(wrap, shape)
    known = {key for name in stated for key in (name, f"{name}_by_player")}
    if not set(wrap) <= known or not all(
        isinstance(edge, str) for pair in stated.values() for edge in pair
    ):
        return wrap
    written: dict[str, Any] = {}
    for key, (first, second) in stated.items():
        if first == second:
            if first != "none":
                written[key] = first
        else:
            written[f"{key}_by_player"] = {"0": first, "1": second}
    return written


def _one_move_turn(turn: dict[str, Any]) -> bool:
    """`compile._compile_turn`'s test for a turn that is every game's turn."""
    moves, ramp = turn.get("moves", 1), turn.get("ramp") or []
    return (
        set(turn) <= {"moves", "ramp"}
        and _same(moves, 1)
        and isinstance(ramp, list)
        and all(_same(length, 1) for length in ramp)
    )


def _drop_defaults(where: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, default in defaults.items():
        if key in where and _same(where[key], default):
            del where[key]


def _same(value: Any, other: Any) -> bool:
    """Equal and of one JSON type: ``true`` is not ``1`` and ``1.0`` is not ``1`` here."""
    return type(value) is type(other) and value == other


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# -------------------------------------------------------------------------- legacy names


class NameCollision(ValueError):
    """Two specs with two names are one game: the legacy table cannot keep both names."""


def legacy_table(specs: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, str]:
    """``{normal-form hash: spelling hash}`` over ``(where, spec)`` pairs, sorted.

    The generator of `legacy_names.LEGACY_NAMES`. The spelling hash is the name every spec had
    before Q1008a, so this must run on the spellings that were named, before anything re-spells
    them. One spelling seen twice is one entry; **two spellings of one game raise**
    `NameCollision`, naming every pair, because only one of their names can survive.
    """
    table: dict[str, str] = {}
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for where, spec in specs:
        normal, named = normal_hash(spec), spelling_hash(spec)
        if normal in table and table[normal] != named:
            clashes.append(f"{seen[normal]} ({table[normal]}) and {where} ({named})")
            continue
        table[normal] = named
        seen.setdefault(normal, where)
    if clashes:
        raise NameCollision(
            "two spellings of one game would share a name: " + "; ".join(clashes)
        )
    return dict(sorted(table.items()))


def merge_legacy(table: dict[str, str], more: dict[str, str]) -> dict[str, str]:
    """``table`` with ``more``'s entries added, refusing one that would rename a game."""
    clashes = [
        f"{normal}: {table[normal]} against {named}"
        for normal, named in more.items()
        if normal in table and table[normal] != named
    ]
    if clashes:
        raise NameCollision("an entry already names this game otherwise: " + "; ".join(clashes))
    return dict(sorted({**table, **more}.items()))


LEGACY_MODULE = Path(__file__).with_name("legacy_names.py")

_LEGACY_HEADER = '''"""The names games had before Q1008a, keyed by their normal-form hash. Generated; never edit.

Generated once (Q1008a, 2026-10-23) by `normal.legacy_table` over every spec any branch of
git ever held, the retired specs `tests/retired_specs.py` builds, the store's rewrite of the
retired round preview, and every spec stored in the main checkout's `runs/` (campaign drafts,
calibration corpora). The one collision was the shogi-uniform mutant, written in two spellings
by four corpora; the name `runs/v9`, `runs/v11` and `runs/v12-calibrate` hold kept it, and
`runs/q478-calibration`'s was left out (`docs/reference/schema-reboot-assessment.md` §7).
Extended once (Q1008s, 2026-10-23) by the site store's 148 stored specs, read as stored: nine
names added, the members' designs, and no collision. Extended only by `python -m
chess_variant_sandbox.spec.normal add <specs.jsonl>`, which refuses an entry that would rename a
game. Never regenerate it from files that have since been re-spelled: the value is each game's
spelling hash *as it was named*, and that is what keeps every name.
"""

LEGACY_NAMES: dict[str, str] = {
'''


def render_legacy_module(table: dict[str, str]) -> str:
    """The source of `legacy_names.py` holding ``table``."""
    rows = "".join(f'    "{normal}": "{named}",\n' for normal, named in sorted(table.items()))
    return _LEGACY_HEADER + rows + "}\n"


def main(argv: list[str] | None = None) -> int:
    """``add <specs.jsonl>``: merge the names of the specs listed, one JSON spec a line.

    The store's half of the table (`docs/reference/schema-reboot-assessment.md` §5 step 2): a
    deploy dumps every stored game's spec, as stored and **before** any migration re-spells it,
    and runs this on the dump. A collision, within the dump or with an entry already here,
    stops it with nothing written.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "add":
        print("usage: python -m chess_variant_sandbox.spec.normal add <specs.jsonl>")
        return 2

    lines = Path(args[1]).read_text(encoding="utf-8").splitlines()
    specs = [(f"{args[1]}:{n}", json.loads(line)) for n, line in enumerate(lines, 1) if line]
    try:
        merged = merge_legacy(LEGACY_NAMES, legacy_table(specs))
    except NameCollision as error:
        print(error, file=sys.stderr)
        return 1
    # LF, as `.gitattributes` holds every tracked file: the fingerprints hash bytes.
    LEGACY_MODULE.write_text(render_legacy_module(merged), encoding="utf-8", newline="\n")
    print(f"{len(merged) - len(LEGACY_NAMES)} names added, {len(merged)} in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
