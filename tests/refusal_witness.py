# compute-lease: light
"""A witness for every refusal, and what the search offers for it (R-L3, B5).

`docs/reference/refusal-offers.md` §9 row B5. A **witness** is a refused draft: a committed
ruleset (or a test arm) with a few edits, which `validate_spec` refuses with one
`refusals.toml` entry (`refusals.entry_of` says so). The witnesses are data, one file per
family under `tests/refusal_witnesses/`, never games of the library:

    {"family": "rounds",
     "witnesses": {"R2": {"from": "tests/_simultaneous_gardner.json",
                          "edits": [["set", "rules.checkmate", {...}]],
                          "wanted": "rules.checkmate",
                          "meets": ["R2"]}},
     "none": {"<id>": "why no witness could be written"}}

``edits`` are applied in order to the file as written (no tidy: the tidy could remove the
refusal): ``["set", path, value]``, ``["drop", path]``, ``["add", path, value]`` (append to a
list), paths in `spec/instead.py`'s spelling. ``wanted`` is the key path the witness asks for
(the search keeps it), or null for a draft refused as it stands; the search's ``base`` is then
the trial without ``wanted``, the draft it was asked from. ``meets`` is every problem the
validator says, as entry ids in its order (null: a problem with no entry), pinned so that a
moved sentence or refusal is seen (§8 test 6).

B2's twenty witnesses (`tests/test_instead.py::WITNESSES`) count for their own entry and are
not written twice.

**Which families are covered** is `FAMILIES`: B5a's fifteen and B5b's twelve, which is every
family `refusals.toml` has (`tests/test_refusal_witnesses.py` holds that no family is left out).

As a command it prints what the search finds, per kind of §5 and reason (``--rows``: one
row per witness, the Markdown of the record's table):

    uv run python tests/refusal_witness.py [--rows]
"""

from __future__ import annotations

import copy
import functools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:  # run as a command
    sys.path.insert(0, str(ROOT / "tests"))

from chess_variant_sandbox.spec import validate_spec  # noqa: E402
from chess_variant_sandbox.spec.instead import (  # noqa: E402
    EVERY,
    Search,
    edit_text,
    parse_path,
    search,
)
from chess_variant_sandbox.spec.refusals import _classification, entry_of  # noqa: E402

STORE = ROOT / "tests" / "refusal_witnesses"

# The families whose every refusal has a witness here (`refusals.toml`'s ``family``): B5a's
# half, then B5b's (§9 row B5), which is every other family, so every refusal of the file.
FAMILIES: tuple[str, ...] = (
    "rounds", "taken piece", "hand", "sweeps", "sight", "wild", "duck", "portals", "nobody",
    "purse", "spy", "paths", "lattice", "royalty", "reach",
    # B5b
    "push", "upkeep", "pass", "income", "gating", "takes_royal", "shared control",
    "scoring goals", "turn parts", "hop runs", "region", "blast",
)


@dataclass(frozen=True)
class Witness:
    entry: str
    source: str  # where it is written: a family file, or B2's name in test_instead
    origin: str  # the file the draft is made from
    edits: tuple[tuple[Any, ...], ...]
    wanted: str | None
    meets: tuple[str | None, ...] | None  # None: pinned elsewhere (B2's)
    b2: str | None = None


def entries() -> dict[str, dict[str, Any]]:
    """Every `[[refusal]]` entry of `refusals.toml`, by id."""
    return {str(raw["id"]): raw for raw in _classification()}


def covered() -> dict[str, dict[str, Any]]:
    """The entries of `FAMILIES` that need a witness (not the inert ones), by id."""
    return {
        rid: raw for rid, raw in entries().items()
        if raw["family"] in FAMILIES and raw["reason"] != "inert"
    }


def _files() -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(STORE.glob("*.json"))]


@functools.cache
def stored() -> dict[str, Witness]:
    """Every witness of the store, and B2's by their own entry."""
    out: dict[str, Witness] = {}
    for data in _files():
        for rid, raw in data.get("witnesses", {}).items():
            assert rid not in out, f"{rid} has two witnesses"
            out[rid] = Witness(
                rid, data["family"], raw["from"], tuple(tuple(e) for e in raw.get("edits", ())),
                raw.get("wanted"), tuple(raw["meets"]),
            )
    from test_instead import WITNESSES  # B2's, by their own entry

    for name, (origin, _refuse, wanted, _from_base, _met, own, _first) in WITNESSES.items():
        if own in out:
            raise AssertionError(f"{own} is B2's witness {name!r}; the store writes it twice")
        out[own] = Witness(own, "test_instead", origin, (), wanted, None, b2=name)
    return out


@functools.cache
def unwritable() -> dict[str, str]:
    """The entries listed with why no witness could be written."""
    out: dict[str, str] = {}
    for data in _files():
        out.update(data.get("none", {}))
    return out


def _edit(node: Any, path: tuple[Any, ...], op: str, value: Any) -> None:
    head, rest = path[0], path[1:]
    if head == EVERY:
        for index in range(len(node)):
            _edit(node, (index, *rest), op, value)
        return
    if rest:
        if op != "drop" and isinstance(head, str) and head not in node:
            node[head] = [] if isinstance(rest[0], int) or rest[0] == EVERY else {}
        _edit(node[head], rest, op, value)
    elif op == "drop":
        del node[head]
    elif op == "add":
        node.setdefault(head, []).append(copy.deepcopy(value))
    else:
        node[head] = copy.deepcopy(value)


def made(witness: Witness) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    """The witness's trial, wanted and base, built afresh."""
    if witness.b2 is not None:
        from test_instead import _made

        return _made(witness.b2)
    trial = json.loads((ROOT / witness.origin).read_text(encoding="utf-8"))
    for op, path, *value in witness.edits:
        _edit(trial, parse_path(path), op, value[0] if value else None)
    base = None
    if witness.wanted is not None:
        base = copy.deepcopy(trial)
        _edit(base, parse_path(witness.wanted), "drop", None)
    return trial, witness.wanted, base


def met(trial: dict[str, Any]) -> list[str | None]:
    return [getattr(entry_of(problem), "id", None) for problem in validate_spec(trial)]


# ------------------------------------------------------------------------- the table

def kind(pair: list[str] | tuple[str, str]) -> str:
    """§5's kind, read mechanically from ``pair`` (as `tests/test_instead.py` reads it)."""
    from test_instead import _kind

    return _kind(pair)


# §5's "found by": the classes the design expected the first offer's edits to be.
EXPECTED: dict[str, frozenset[str]] = {
    "whole": frozenset("RP"), "value": frozenset("V"), "narrower": frozenset("SVP"),
    "class": frozenset("PD"),
}


def edit_class(edit: tuple[Any, ...]) -> str:
    """The class an offer's edit is (§4.1), read from what it did."""
    if edit[0] == "set":
        return "V"
    path = parse_path(edit[1])
    if path[0] == "rules":
        return "R" if len(path) == 2 else "S"
    return "P"


@dataclass(frozen=True)
class Found:
    entry: str
    kind: str
    reason: str
    search: Search
    seconds: float

    @property
    def first(self) -> tuple[tuple[Any, ...], ...] | None:
        return self.search.offers[0].edits if self.search.offers else None

    @property
    def classes(self) -> str:
        first = self.first
        return "" if first is None else "".join(sorted({edit_class(e) for e in first}))

    @property
    def agrees(self) -> bool:
        return self.first is not None and set(self.classes) <= EXPECTED[self.kind]


def find(rid: str) -> Found:
    raw = entries()[rid]
    trial, wanted, base = made(stored()[rid])
    began = time.perf_counter()
    found = search(trial, wanted=wanted, base=base)
    return Found(rid, kind(raw["pair"]), raw["reason"], found, time.perf_counter() - began)


def find_all() -> list[Found]:
    return [find(rid) for rid in covered() if rid in stored()]


KINDS = ("whole", "value", "narrower", "class")
REASONS = ("contradictory", "too_expensive", "not_built")


def counts(found: list[Found]) -> dict[tuple[str, str], dict[str, int]]:
    """Per (kind, reason): witnesses, offers, none, agreeing, and the first offer's classes."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for item in found:
        cell = out.setdefault((item.kind, item.reason), {"witnesses": 0, "offer": 0, "none": 0,
                                                         "agrees": 0})
        cell["witnesses"] += 1
        cell["offer" if item.first is not None else "none"] += 1
        cell["agrees"] += item.agrees
        if item.first is not None:
            cell[item.classes] = cell.get(item.classes, 0) + 1
    return out


def _row(item: Found) -> str:
    first = item.first
    said = "no offer" if first is None else "; ".join(f"`{edit_text(e)}`" for e in first)
    units = "" if first is None else str(item.search.offers[0].units)
    return (f"| `{item.entry}` | {item.kind} | {item.reason.replace('_', ' ')} | {said} | {units} "
            f"| {item.classes or '-'} | {'yes' if item.agrees else 'no'} "
            f"| {item.search.validations} | {item.seconds * 1000:.0f} |")


def main(argv: list[str]) -> int:
    found = find_all()
    missing = [rid for rid in covered() if rid not in stored() and rid not in unwritable()]
    if "--rows" in argv:
        print("| entry | kind | reason | first offer | units | classes | as expected "
              "| validations | ms |")
        print("|---|---|---|---|---|---|---|---|---|")
        for item in found:
            print(_row(item))
        return 0
    table = counts(found)
    print("kind      reason         witnesses  offer  none  as expected  first offer's classes")
    for (kind_, reason), cell in sorted(table.items(), key=lambda kv: (KINDS.index(kv[0][0]),
                                                                       REASONS.index(kv[0][1]))):
        classes = ", ".join(f"{k} {v}" for k, v in sorted(cell.items())
                            if k not in ("witnesses", "offer", "none", "agrees"))
        print(f"{kind_:<9} {reason:<14} {cell['witnesses']:>9}  {cell['offer']:>5}  "
              f"{cell['none']:>4}  {cell['agrees']:>11}  {classes}")
    total = len(found)
    offered = sum(1 for item in found if item.first is not None)
    print(f"all: {total} witnesses, {offered} offer, {total - offered} none, "
          f"{sum(item.agrees for item in found)} as expected; "
          f"{len(unwritable())} listed with no witness; {len(missing)} missing")
    seconds = sorted(item.seconds for item in found)
    if seconds:
        print(f"time: median {seconds[len(seconds) // 2] * 1000:.0f} ms, "
              f"max {seconds[-1] * 1000:.0f} ms, all {sum(seconds):.1f} s; validations max "
              f"{max(item.search.validations for item in found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
