"""The one current spelling of a rule file, and the converter into it (Q1008b).

`docs/reference/schema-reboot-assessment.md` §5 step 3. Since Q1008a a game is named by its
normal form (`normal.normal_form`), so a file can be re-spelled without renaming its game, and
`rulesets/`, `archive/` and the site's stored specs were rewritten once into the spelling
`current_spelling` writes. It stays as the import path for an old file someone still holds::

    python -m chess_variant_sandbox.spec.spelling FILE...          # rewrite each in place
    python -m chess_variant_sandbox.spec.spelling --check FILE...  # exit 1 naming any that is not

**What the current spelling is.** Every rewrite but the one translation below is one
`normal_form` also makes, so a converted
spec has its original's identity and compiles to its tables (`tests/test_spelling.py` holds
both over the library and its re-spellings):

1. ``schema_version`` is `SCHEMA_VERSION`, written first when a file left it out. The stamp
   is the spelling's version (`schema.SCHEMA_VERSION`): it is in no game's name, and since
   Q1008c the validator reads nothing from it.
2. A *rule* stated at its "no rule" value is left out: one of `normal._ABSENT_WHEN_EMPTY` at
   ``false``, ``0``, ``""``, ``[]`` or ``{}``; a ``turn`` of one move; a ``wrap`` whose every
   edge is ``"none"``; a ``strike`` that `schema.strike_is_live` calls plain capture; and,
   since Q1008c, a key at its off value (`normal.drop_off_values`: ``rest: 0``, ``tie:
   "both"``, ``kind: "momentum"`` and the like), which the validator no longer refuses, so a
   person who writes one is tidied rather than refused.
3. One spelling where the language has two for one statement: ``promotion`` is one object
   when it has one entry and a list only when it has several (the spelling most of the
   library writes and the model's contract describes); a
   numbered promotion zone is ``ranks``, ascending and without repeats; ``to_by_player``
   naming one list for both players is ``to``; a ``wrap`` is written key by key from
   `schema.wrap_stated`, ``X_by_player`` only where the players differ.

**What it leaves as written, on purpose.**

* A *setting* written at its default (``"win": "checkmate"``, ``"mode": "both"``, ``"mirror":
  true``, ``"player": 0``, a clock at its default). A setting always has a value, so writing
  it states nothing twice; the model's contract requires ``win``, and the rules text reads some
  of them (a placement's ``player`` is "player 0 only" in a setup stated per player). The
  normal form drops them for naming only.
* ``"rank": "last"``, and a zone left at its default. "The last rank" is a statement about
  the board, which an operator that resizes the board keeps true; ``"ranks": [7]`` is not.
* A lone ``rank`` past the board's edge: `schema._promotion_ranks_past_the_board` refuses
  it, but `schema.stored_spec_errors` forgives it for the one game kept with one
  (`schema.RANK_PAST_THE_BOARD_KEPT`), and ``ranks`` would be refused always, so that game
  keeps it. Q1008c kept it: *tunnel-horizontal*'s dead promotion is the rule its en passant
  is defined against, so dropping it would be refused, or change the game.

4. Spelling 31 (Q1007b, `docs/archive/plan-schema-31.md` §4.2): a portal group's moves are written on
   the group (``rules.portals[].moves``), not on a stand-in piece the group names by
   ``piece`` (`_portal_moves_on_groups`). The normal form writes them back as that stand-in
   (`normal._portal_moves_as_pieces`), so the converted file keeps its game's name and tables.

**One translation.** A retired ``rules.simultaneous.rest`` is written as ``pieces[].rest``
(`_rest_on_pieces`), the rewrite the store's ``0039`` made. It is not a spelling, so the
converted game is named by its new spelling; no game in the library or the store carries it.
* ``strike`` against ``contacts``, and a portal pair against a two-end ``ends`` group: they are
  not one meaning (`normal`'s docstring), so each stays as written.
* ``name``, ``description``, ``appearance`` and ``provenance``.

`respell_text` rewrites a file's text by splicing: only the values that changed are written
again, in the file's own layout, so a hand-laid-out file keeps its layout.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.spec.normal import (
    _ABSENT_WHEN_EMPTY,
    _EMPTY,
    _is_int,
    _normal_wrap,
    _one_move_turn,
    _same,
    drop_off_values,
)
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, strike_is_live

# --------------------------------------------------------------------------- the spelling


def current_spelling(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` written in the one current spelling. Never mutates ``spec``; keeps key order.

    Each rewrite applies only where its input is well formed, as `normal.normal_form`'s do,
    so a malformed spec keeps what the validator will name.
    """
    out = json.loads(json.dumps(spec))
    if not isinstance(out, dict):
        return out
    if "schema_version" not in out:
        out = {"schema_version": SCHEMA_VERSION, **out}
    out["schema_version"] = SCHEMA_VERSION
    _rest_on_pieces(out)
    _portal_moves_on_groups(out)
    rules = out.get("rules")
    if isinstance(rules, dict):
        board = out.get("board")
        height = board.get("height") if isinstance(board, dict) else None
        shape = board.get("shape") if isinstance(board, dict) else None
        out["rules"] = _rules(rules, height, shape)
    drop_off_values(out)
    return out


def _rest_on_pieces(spec: dict[str, Any]) -> None:
    """Schema 27's ``rules.simultaneous.rest``, retired by schema 29, written the new way.

    The rewrite of `docs/plan-rest.md` §4.5 (`tests/rest_equivalence.py::rewrite`, and the
    store's ``0039_rest_on_pieces.sql``): the switch goes, and where it was ``true`` every
    piece type a player moves (one with a move, named by no ``rules.npcs`` entry) rests 1.
    The one rewrite here the normal form does not make: it is a translation, not a
    spelling, so the converted game is named by its new spelling, as `0039`'s rows are.
    """
    rules = spec.get("rules")
    if not isinstance(rules, dict):
        return
    rule = rules.get("simultaneous")
    if not isinstance(rule, dict) or "rest" not in rule:
        return
    if rule.pop("rest") is not True:
        return
    npcs = rules.get("npcs")
    wild = {npc.get("piece") for npc in npcs if isinstance(npc, dict)} if isinstance(
        npcs, list
    ) else set()
    for piece in spec.get("pieces") or []:
        if isinstance(piece, dict) and piece.get("moves") and piece.get("id") not in wild:
            piece["rest"] = 1


def _portal_moves_on_groups(spec: dict[str, Any]) -> None:
    """Spelling 30's portal stand-in written the spelling-31 way, in place (§4.2).

    Each group that names a declared piece by ``piece`` takes that piece's moves as its own
    ``moves``, in ``piece``'s place, and the key left out where the piece had none; ``piece``
    goes. Then every piece a group named leaves ``pieces``. A group naming no declared piece
    keeps its ``piece`` for the validator to name, and the piece stays. In every group's
    moves, ``"mode": "move"`` is left out: a portal's move is move-only, so it is what a
    group's move means without one (one spelling for one statement).
    """
    rules = spec.get("rules")
    pieces = spec.get("pieces")
    if not isinstance(rules, dict) or not isinstance(pieces, list):
        return
    groups = rules.get("portals")
    if not isinstance(groups, list):
        return
    declared = {
        piece["id"]: piece for piece in pieces
        if isinstance(piece, dict) and isinstance(piece.get("id"), str)
    }
    named: set[str] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        if isinstance(group.get("moves"), list):
            group["moves"] = _move_only_unsaid(group["moves"])
        if group.get("piece") not in declared:
            continue
        piece_id = group["piece"]
        named.add(piece_id)
        moves = declared[piece_id].get("moves")
        written = _move_only_unsaid(moves if isinstance(moves, list) else [])
        converted: dict[str, Any] = {}
        for key, value in group.items():
            if key == "piece":
                if written:
                    converted["moves"] = written
            elif not (key == "moves" and written):
                converted[key] = value
        groups[index] = converted
    if named:
        spec["pieces"] = [
            piece for piece in pieces
            if not (isinstance(piece, dict) and piece.get("id") in named)
        ]


def _move_only_unsaid(moves: list[Any]) -> list[Any]:
    """A portal group's moves with each ``"mode": "move"`` left out (`schema.portal_moves`)."""
    return [
        {key: value for key, value in move.items() if not (key == "mode" and value == "move")}
        if isinstance(move, dict) else move
        for move in moves
    ]


def _rules(rules: dict[str, Any], height: Any, shape: Any) -> dict[str, Any]:
    """The rules with each rule in its one spelling and each stated-off rule left out.

    In `normal._normal_rules`'s order, so the two agree on what an odd input becomes.
    """
    out: dict[str, Any] = {}
    for key, value in rules.items():
        if key == "promotion" and isinstance(value, (dict, list)):
            entries = [value] if isinstance(value, dict) else value
            entries = [_promotion(entry, height) if isinstance(entry, dict) else entry
                       for entry in entries]
            # Asked of the list, as the normal form asks it, before one entry is unwrapped:
            # only `[]` states nothing.
            if entries:
                single = len(entries) == 1 and isinstance(entries[0], dict)
                out[key] = entries[0] if single else entries
            continue
        if key == "wrap" and isinstance(value, dict):
            value = _normal_wrap(value, shape)
        if not _stated_off(key, value):
            out[key] = value
    return out


def _stated_off(key: str, value: Any) -> bool:
    """Whether a rule is written at the value that means no rule."""
    if key == "turn" and isinstance(value, dict) and _one_move_turn(value):
        return True
    if key == "strike" and isinstance(value, dict) and not strike_is_live(value):
        return True
    return key in _ABSENT_WHEN_EMPTY and any(_same(value, empty) for empty in _EMPTY)


def _promotion(entry: dict[str, Any], height: Any) -> dict[str, Any]:
    """One promotion entry: a numbered zone as ``ranks``, one list for both players as ``to``."""
    zoned = "squares" not in entry and not ("rank" in entry and "ranks" in entry)
    by_player = entry.get("to_by_player")
    one_list = (
        "to" not in entry
        and isinstance(by_player, dict)
        and set(by_player) == {"0", "1"}
        and _same(by_player["0"], by_player["1"])
    )
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "to_by_player" and one_list:
            out["to"] = by_player["0"]  # type: ignore[index]
        elif (
            key == "rank" and zoned and _is_int(value)
            and _is_int(height) and 0 <= value < height
        ):
            out["ranks"] = [value]
        elif key == "ranks" and zoned and isinstance(value, list) and all(map(_is_int, value)):
            out["ranks"] = sorted(set(value))
        else:
            out[key] = value
    return out


def is_current(spec: dict[str, Any]) -> bool:
    """Whether ``spec`` is already in the current spelling."""
    return _same_json(current_spelling(spec), spec)


def _same_json(one: Any, other: Any) -> bool:
    """Equal as JSON, key order included: ``true`` is not ``1``."""
    return json.dumps(one) == json.dumps(other)


# ------------------------------------------------------------------- the file, re-spelled


@dataclass
class _Node:
    """A JSON value's span in its text, and its members' or items' spans."""

    start: int
    end: int
    members: list[tuple[str, int, _Node]] = field(default_factory=list)  # key, key start
    items: list[_Node] = field(default_factory=list)


_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SPACE = re.compile(r"[ \t\n\r]*")
_DECODER = json.JSONDecoder()


def _parse(text: str, at: int) -> _Node:
    at = _SPACE.match(text, at).end()  # type: ignore[union-attr]
    char = text[at]
    if char == "{":
        node = _Node(at, at)
        at = _SPACE.match(text, at + 1).end()  # type: ignore[union-attr]
        if text[at] == "}":
            node.end = at + 1
            return node
        while True:
            key_start = at
            key, at = _DECODER.raw_decode(text, at)
            at = _SPACE.match(text, at).end()  # type: ignore[union-attr]
            value = _parse(text, at + 1)  # past the colon
            node.members.append((key, key_start, value))
            at = _SPACE.match(text, value.end).end()  # type: ignore[union-attr]
            if text[at] == "}":
                node.end = at + 1
                return node
            at = _SPACE.match(text, at + 1).end()  # type: ignore[union-attr]
    if char == "[":
        node = _Node(at, at)
        at = _SPACE.match(text, at + 1).end()  # type: ignore[union-attr]
        if text[at] == "]":
            node.end = at + 1
            return node
        while True:
            item = _parse(text, at)
            node.items.append(item)
            at = _SPACE.match(text, item.end).end()  # type: ignore[union-attr]
            if text[at] == "]":
                node.end = at + 1
                return node
            at = at + 1
    if char == '"':
        return _Node(at, _DECODER.raw_decode(text, at)[1])
    for word in ("true", "false", "null"):
        if text.startswith(word, at):
            return _Node(at, at + len(word))
    number = _NUMBER.match(text, at)
    if number is None:
        raise ValueError(f"not JSON at character {at}")
    return _Node(at, number.end())


@dataclass(frozen=True)
class _Layout:
    """How the file lays out a value that has to be written again."""

    plain: bool  # the whole file is `json.dumps(indent=2)`
    ascii: bool
    width: int = 100

    def write(self, value: Any, indent: int, inline: bool, room: int) -> str:
        if self.plain:
            text = json.dumps(value, indent=2, ensure_ascii=self.ascii)
            return text.replace("\n", "\n" + " " * indent)
        flat = _flat(value, self.ascii)
        if inline and len(flat) <= room:
            return flat
        return _laid_out(value, indent, self.width, self.ascii)


def _flat(value: Any, ascii_: bool) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{json.dumps(k, ensure_ascii=ascii_)}: {_flat(v, ascii_)}" for k, v in value.items()
        ) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_flat(v, ascii_) for v in value) + "]"
    return json.dumps(value, ensure_ascii=ascii_)


def _laid_out(value: Any, indent: int, width: int, ascii_: bool) -> str:
    """A hand-laid-out file's style: a value on one line where it fits, else one per line."""
    if not isinstance(value, (dict, list)) or not value:
        return _flat(value, ascii_)
    pad = " " * (indent + 2)
    lines = []
    entries = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
    for index, (key, item) in enumerate(entries):
        head = f"{json.dumps(key, ensure_ascii=ascii_)}: " if isinstance(value, dict) else ""
        tail = "," if index < len(entries) - 1 else ""
        flat = _flat(item, ascii_)
        if len(pad) + len(head) + len(flat) + len(tail) <= width:
            lines.append(pad + head + flat + tail)
        else:
            lines.append(pad + head + _laid_out(item, indent + 2, width, ascii_) + tail)
    opening, closing = ("{", "}") if isinstance(value, dict) else ("[", "]")
    return opening + "\n" + "\n".join(lines) + "\n" + " " * indent + closing


def respell_text(text: str) -> str:
    """A rule file's text in the current spelling, laid out as it was.

    Only what `current_spelling` changes is written again: a removed member is cut with its
    comma, a value whose shape changed is written in the file's layout at its own indentation,
    and everything else -- the words, the order, the line breaks -- is the file's own.
    """
    old = json.loads(text)
    new = current_spelling(old)
    if _same_json(old, new):
        return text
    layout = _Layout(
        plain=any(text.rstrip("\n") == json.dumps(old, indent=2, ensure_ascii=a)
                  for a in (False, True)),
        ascii=text.isascii(),
    )
    root = _parse(text, 0)
    edits: list[tuple[int, int, str]] = []
    if "schema_version" not in old and root.members:
        # Written first, on a line of its own like the member it goes before.
        first = root.members[0][1]
        indent = text[text.rfind("\n", 0, first) + 1:first]
        edits.append((first, first, f'"schema_version": {SCHEMA_VERSION},\n{indent}'))
        new = {key: value for key, value in new.items() if key != "schema_version"}
    _splice(text, root, old, new, layout, edits)
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    if not _same_json(json.loads(text), current_spelling(old)):
        raise AssertionError("the re-spelt text does not read back as the current spelling")
    return text


def _splice(
    text: str, node: _Node, old: Any, new: Any, layout: _Layout,
    edits: list[tuple[int, int, str]],
) -> None:
    if _same_json(old, new):
        return
    if isinstance(old, dict) and isinstance(new, dict) and node.members and new:
        kept = [key for key in old if key in new]
        if kept == list(new):
            _cut_members(text, node, [key for key in old if key not in new], edits)
            for key, _, child in node.members:
                if key in new:
                    _splice(text, child, old[key], new[key], layout, edits)
            return
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new) and old:
        for child, one, other in zip(node.items, old, new, strict=True):
            _splice(text, child, one, other, layout, edits)
        return
    if isinstance(old, list) and isinstance(new, list) and new and len(new) < len(old):
        gone = _items_cut(old, new)
        if gone is not None:
            _cut_items(node, gone, edits)
            return
    line = text.rfind("\n", 0, node.start) + 1
    indent = len(text[line:node.start]) - len(text[line:node.start].lstrip(" "))
    end_of_line = text.find("\n", node.end)
    after = len(text[node.end:end_of_line if end_of_line >= 0 else len(text)])
    inline = "\n" not in text[node.start:node.end]
    room = layout.width - (node.start - line) - after
    edits.append((node.start, node.end, layout.write(new, indent, inline, room)))


def _items_cut(old: list[Any], new: list[Any]) -> list[int] | None:
    """The indices of ``old`` to cut so that what is left is ``new`` item for item, or
    ``None`` when ``new`` is not ``old`` with some items taken out."""
    gone: list[int] = []
    at = 0
    for index, item in enumerate(old):
        if at < len(new) and _same_json(item, new[at]):
            at += 1
        else:
            gone.append(index)
    return gone if at == len(new) else None


def _cut_items(node: _Node, gone: list[int], edits: list[tuple[int, int, str]]) -> None:
    """Cut the items of ``node`` at the indices ``gone``, with the comma that joins each run,
    as `_cut_members` cuts members. ``node`` keeps at least one item."""
    items = node.items
    index = 0
    while index < len(items):
        if index not in gone:
            index += 1
            continue
        last = index
        while last + 1 < len(items) and last + 1 in gone:
            last += 1
        if index > 0:
            edits.append((items[index - 1].end, items[last].end, ""))
        else:
            edits.append((items[0].start, items[last + 1].start, ""))
        index = last + 1


def _cut_members(
    text: str, node: _Node, gone: list[str], edits: list[tuple[int, int, str]]
) -> None:
    """Cut each member named in ``gone`` from ``node``'s text, with the comma that joins it.

    A run of cut members after a kept one goes from that member's end to the run's end, so
    the run's leading comma goes with it; a run that opens the object goes from its first key
    to the next kept key. ``node`` keeps at least one member (`_splice` writes ``{}`` else).
    """
    members = node.members
    index = 0
    while index < len(members):
        if members[index][0] not in gone:
            index += 1
            continue
        last = index
        while last + 1 < len(members) and members[last + 1][0] in gone:
            last += 1
        if index > 0:
            edits.append((members[index - 1][2].end, members[last][2].end, ""))
        else:
            edits.append((members[0][1], members[last + 1][1], ""))
        index = last + 1


# ------------------------------------------------------------------------------- the CLI


def main(argv: list[str] | None = None) -> int:
    """Rewrite each file named into the current spelling, or with ``--check`` only say which."""
    args = sys.argv[1:] if argv is None else argv
    check = "--check" in args
    paths = [Path(arg) for arg in args if arg != "--check"]
    if not paths:
        print("usage: python -m chess_variant_sandbox.spec.spelling [--check] FILE...")
        return 2
    behind = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        respelt = respell_text(text)
        if respelt == text:
            continue
        behind.append(path)
        if not check:
            # LF, as `.gitattributes` holds every tracked file.
            path.write_text(respelt, encoding="utf-8", newline="\n")
    verb = "not in the current spelling" if check else "re-spelt"
    for path in behind:
        print(f"{verb}: {path}")
    print(f"{len(behind)} of {len(paths)} {verb}")
    return 1 if check and behind else 0


if __name__ == "__main__":
    raise SystemExit(main())
