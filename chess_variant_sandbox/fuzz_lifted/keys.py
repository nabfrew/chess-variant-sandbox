"""Which lifted pairs a spec holds, read off its spelling.

`spec/refusals.toml`'s `[[lifted]]` entries are the list (R-L1a): each names two
key paths in the words `spec/refusals.py` uses, the task that lifted it, and whether a narrower
case stays refused. This module reads that file as data and answers, for a spec, which of its
pairs are present. A detector reads the spelling only; it never compiles, so it costs nothing
beside a validation.

A key path the detectors do not know is reported by `unknown_keys`, and a test holds that list
empty, so a new `[[lifted]]` row with a new path fails the fast suite instead of going silently
uncovered.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from chess_variant_sandbox.spec.capture import capture_list
from chess_variant_sandbox.spec.refusals import REASONS_FILE
from chess_variant_sandbox.spec.schema import (
    contacts_only_remove,
    duck_piece,
    income_rule,
    portal_groups,
    reads_alike_for_both,
    run_hop_ids,
    shared_ids,
    strike_is_live,
    strike_only_removes,
    wild_type_ids,
)

DEBT = REASONS_FILE

# The batches of `docs/plan-schema-next.md` §3.2's table, by the task prefix that built them.
# A-F are the batches R-L6a covers; "none" is the Python-only work (SR1a-c, SR2, SR2b, SR11a)
# and the push's own round 2 (Q1074b8), which no batch holds.
_BATCH_OF_PREFIX = (
    ("SR1d", "A"), ("SR3", "A"), ("SR4", "B"), ("SR5", "C"), ("SR6", "D"), ("SR7", "E"),
    ("SR8", "F"),
)
BATCH_ORDER = ("F", "E", "D", "C", "B", "A", "none")


def batch_of(by: str) -> str:
    """The batch a `[[lifted]]` row's `by` names: the latest task it lists decides
    (`"SR2b, SR4h"` is batch B's, since SR4h built the core half)."""
    found = "none"
    for task in (part.strip() for part in by.split(",")):
        for prefix, batch in _BATCH_OF_PREFIX:
            if task.startswith(prefix) and (
                found == "none" or BATCH_ORDER.index(batch) < BATCH_ORDER.index(found)
            ):
                found = batch
    return found


@dataclass(frozen=True)
class Lifted:
    """One `[[lifted]]` row."""

    id: str
    pair: tuple[str, str]
    by: str
    narrowed: bool

    @property
    def batch(self) -> str:
        return batch_of(self.by)


@functools.cache
def lifted(path: Path = DEBT) -> tuple[Lifted, ...]:
    """Every `[[lifted]]` row, in the file's order."""
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return tuple(
        Lifted(
            id=row["id"], pair=(row["pair"][0], row["pair"][1]), by=row["by"],
            narrowed=bool(row.get("narrowed", False)),
        )
        for row in data.get("lifted", [])
    )


# ------------------------------------------------------------------------- readers

def _rules(spec: dict[str, Any]) -> dict[str, Any]:
    rules = spec.get("rules")
    return rules if isinstance(rules, dict) else {}


def _pieces(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in spec.get("pieces", []) or [] if isinstance(p, dict)]


def _moves(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for p in _pieces(spec) for m in p.get("moves", []) or [] if isinstance(m, dict)]


def _sights(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for p in _pieces(spec) for s in p.get("sight", []) or [] if isinstance(s, dict)]


def _promotions(spec: dict[str, Any]) -> list[dict[str, Any]]:
    promotion = _rules(spec).get("promotion")
    if isinstance(promotion, dict):
        return [promotion]
    if isinstance(promotion, list):
        return [p for p in promotion if isinstance(p, dict)]
    return []


def _entries(spec: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """R-L4b stage 3: the capture list's entries of ``kind``, read off either spelling
    (`spec.capture.capture_list`, the one reader)."""
    entries = capture_list(spec) or []
    return [entry for entry in entries if entry.get("do") == kind]


def _rule_of(spec: dict[str, Any], kind: str) -> dict[str, Any] | None:
    """A kind's rule as its old key's object (its first entry's keys), or ``None``."""
    found = _entries(spec, kind)
    if not found:
        return None
    return {k: v for k, v in found[0].items() if k not in ("do", "on", "path_by")}


def _strike(spec: dict[str, Any]) -> dict[str, Any] | None:
    strike = _rule_of(spec, "strike")
    return strike if strike_is_live(strike) else None


def _contacts(spec: dict[str, Any]) -> dict[str, Any] | None:
    contacts = _rule_of(spec, "contacts")
    return contacts if isinstance(contacts, dict) and contacts.get("links") else None


def _drops(spec: dict[str, Any]) -> dict[str, Any] | None:
    drops = _rules(spec).get("drops")
    return drops if isinstance(drops, dict) else None


def _from_captures(spec: dict[str, Any]) -> bool:
    return bool(_entries(spec, "hand"))


def _blast_spares_or_is_read(spec: dict[str, Any]) -> bool:
    blasts = _entries(spec, "blast")
    meets = _entries(spec, "strike") + _entries(spec, "contacts")
    return bool(blasts) and (
        any(b.get("spares_taken") for b in blasts) or any("blast" in m["on"] for m in meets)
    )


def _path_by(spec: dict[str, Any]) -> bool:
    return any("path_by" in entry for entry in capture_list(spec) or [])


def _stocked(spec: dict[str, Any]) -> bool:
    drops = _drops(spec)
    return drops is not None and bool(drops.get("stock") or drops.get("stock_by_player"))


def _nobody(spec: dict[str, Any]) -> bool:
    placements = (spec.get("setup") or {}).get("placements", []) or []
    return any(isinstance(p, dict) and p.get("player") == "nobody" for p in placements)


def _majority(spec: dict[str, Any]) -> bool:
    for piece in _pieces(spec):
        control = piece.get("control")
        if control == "majority" or (isinstance(control, dict) and "majority" in control):
            return True
    return False


def _player_moved(spec: dict[str, Any]) -> bool:
    return _majority(spec) or any(p.get("control") == "shared" for p in _pieces(spec))


def _reads_unalike(spec: dict[str, Any]) -> bool:
    orientation = (spec.get("setup") or {}).get("orientation", "mirror")
    return any(
        not reads_alike_for_both(m.get("vectors"), m.get("symmetry", "none"), orientation)
        for m in _moves(spec)
    )


def _placement_facing(spec: dict[str, Any]) -> bool:
    placements = (spec.get("setup") or {}).get("placements", []) or []
    return any(isinstance(p, dict) and "facing" in p for p in placements)


def _turn_long(spec: dict[str, Any]) -> bool:
    turn = _rules(spec).get("turn")
    if not isinstance(turn, dict):
        return False
    if turn.get("parts"):
        return True
    return (turn.get("moves", 1) or 1) > 1 or any(n > 1 for n in turn.get("ramp", []) or [])


def _explode(spec: dict[str, Any]) -> bool:
    return bool(_entries(spec, "blast"))


def _portals(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return portal_groups(_rules(spec)) if _rules(spec).get("portals") else []


def _portals_moved(spec: dict[str, Any]) -> bool:
    for group in _portals(spec):
        moved = group.get("moved_by")
        if isinstance(moved, dict) and any(moved.get(end) for end in ("orange", "blue")):
            return True
        if any(isinstance(end, dict) and end.get("moved_by") for end in group.get("ends", []) or []):
            return True
    return False


def _sight_effect(effect: str) -> Callable[[dict[str, Any]], bool]:
    return lambda spec: any(s.get("effect") == effect for s in _sights(spec))


def _sight_affects_wild(spec: dict[str, Any]) -> bool:
    wild = wild_type_ids(_rules(spec))
    duck = duck_piece(_rules(spec))
    if duck:
        wild = wild | {duck}
    return any(set(s.get("affects", []) or []) & wild for s in _sights(spec))


def _wild_or_duck(spec: dict[str, Any]) -> bool:
    return bool(_rules(spec).get("npcs")) or bool(duck_piece(_rules(spec)))


def _lame_rider(spec: dict[str, Any]) -> bool:
    return any(m.get("kind") == "lame" and m.get("range", 1) != 1 for m in _moves(spec))


def _castling_not_royal(spec: dict[str, Any]) -> bool:
    castling = _rules(spec).get("castling")
    if not isinstance(castling, dict):
        return False
    royal = {p.get("id") for p in _pieces(spec) if p.get("royal")}
    return castling.get("king") not in royal


def _strike_position(position: str) -> Callable[[dict[str, Any]], bool]:
    def has(spec: dict[str, Any]) -> bool:
        strike = _strike(spec)
        return strike is not None and strike.get("position", "advance") == position
    return has


def _per_type_own(spec: dict[str, Any]) -> bool:
    extinction = _rules(spec).get("extinction")
    return isinstance(extinction, dict) and extinction.get("per_type") == "own"


def _npcs_field(field: str, value: Any = None) -> Callable[[dict[str, Any]], bool]:
    def has(spec: dict[str, Any]) -> bool:
        npcs = _rules(spec).get("npcs") or []
        return any(
            isinstance(e, dict) and (e.get(field) == value if value is not None else bool(e.get(field)))
            for e in npcs
        )
    return has


def _hex(spec: dict[str, Any]) -> bool:
    return (spec.get("board") or {}).get("lattice") == "hex"


def _promotion_field(field: str) -> Callable[[dict[str, Any]], bool]:
    return lambda spec: any(p.get(field) for p in _promotions(spec))


# Every key path the `[[lifted]]` list names, in its own words, to the spelling that holds it.
DETECTORS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "rules.en_passant": lambda s: _rules(s).get("en_passant") is True,
    "rules.turn": _turn_long,
    "pieces[].rest": lambda s: any((p.get("rest") or 0) > 0 for p in _pieces(s)),
    "rules.npcs": lambda s: bool(_rules(s).get("npcs")),
    'pieces[].control: "majority"': _majority,
    "rules.purse": lambda s: isinstance(_rules(s).get("purse"), dict),
    "rules.duck": lambda s: bool(duck_piece(_rules(s))),
    "rules.simultaneous": lambda s: isinstance(_rules(s).get("simultaneous"), dict),
    "rules.castling": lambda s: isinstance(_rules(s).get("castling"), dict),
    "rules.portals[].moved_by": _portals_moved,
    'rules.strike.position: "stay"': _strike_position("stay"),
    'rules.strike.position: "home"': _strike_position("home"),
    'setup.placements[].player: "nobody"': _nobody,
    "rules.purse.market": lambda s: bool((_rules(s).get("purse") or {}).get("market")),
    "rules.strike": lambda s: _strike(s) is not None,
    "rules.strike (removing its victim)": lambda s: strike_only_removes(_strike(s)),
    "rules.strike.by": lambda s: bool((_strike(s) or {}).get("by")),
    "rules.strike.switches_sides": lambda s: bool((_strike(s) or {}).get("switches_sides")),
    "rules.strike.chain": lambda s: bool((_strike(s) or {}).get("chain"))
    and (_strike(s) or {}).get("effect", "capture") != "capture",
    "rules.explode_on_capture": _explode,
    "rules.drops": lambda s: _drops(s) is not None,
    "rules.drops.from_captures": _from_captures,
    "rules.drops.stock": _stocked,
    "rules.drops.either_face": lambda s: bool((_drops(s) or {}).get("either_face")),
    "rules.contacts": lambda s: _contacts(s) is not None,
    "rules.contacts.links": lambda s: _contacts(s) is not None,
    "rules.contacts (links removing their target)": lambda s: _contacts(s) is not None
    and contacts_only_remove(_contacts(s)),
    "rules.contacts (an own link)": lambda s: any(
        isinstance(link, dict) and link.get("own") for link in (_contacts(s) or {}).get("links", [])
    ),
    "rules.must_capture": lambda s: _rules(s).get("must_capture") is True,
    "rules.promotion": lambda s: bool(_promotions(s)),
    "rules.promotion.only_lost": _promotion_field("only_lost"),
    "rules.promote_on_entry": lambda s: _rules(s).get("promote_on_entry") is True,
    'rules.extinction.per_type: "own"': _per_type_own,
    'pieces[].moves[].kind: "sweep"': lambda s: any(m.get("kind") == "sweep" for m in _moves(s)),
    "rules.wrap": lambda s: isinstance(_rules(s).get("wrap"), dict) and any(
        v not in ("none", None) for v in _rules(s)["wrap"].values()
    ),
    "rules.portals": lambda s: bool(_portals(s)),
    'pieces[].sight effect "shield"': _sight_effect("shield"),
    'pieces[].sight effect "command"': _sight_effect("command"),
    "pieces[].sight": lambda s: bool(_sights(s)),
    "pieces[].sight captures_own": lambda s: any(x.get("captures_own") for x in _sights(s)),
    "pieces[].sight affects": _sight_affects_wild,
    "a wild type or the duck": _wild_or_duck,
    "rules.reach": lambda s: isinstance(_rules(s).get("reach"), dict),
    "rules.territory": lambda s: isinstance(_rules(s).get("territory"), dict),
    "rules.npcs[].immune": _npcs_field("immune"),
    "rules.npcs[].turns_back": _npcs_field("turns_back"),
    "rules.npcs[].also_at": _npcs_field("also_at"),
    'rules.npcs[].pounce: "prey"': _npcs_field("pounce", "prey"),
    "rules.npcs[].respawn": _npcs_field("respawn"),
    'rules.win: "checkmate"': lambda s: _rules(s).get("win", "checkmate") == "checkmate",
    "rules.forbid_check": lambda s: _rules(s).get("forbid_check") is True,
    "rules.reach.safe": lambda s: (_rules(s).get("reach") or {}).get("safe") is True,
    "rules.npcs[].pounce": _npcs_field("pounce"),
    "pieces[].moves[].path": lambda s: any(m.get("path", "straight") != "straight" for m in _moves(s)),
    "rules.explode_on_capture, rules.strike or rules.contacts": lambda s: _explode(s)
    or _strike(s) is not None or _contacts(s) is not None,
    'board.lattice: "hex"': _hex,
    "pieces[].region": lambda s: any(p.get("region") for p in _pieces(s)),
    "pieces[].spy": lambda s: any(p.get("spy") for p in _pieces(s)),
    "pieces[].price": lambda s: any(p.get("price") for p in _pieces(s)),
    'moves[].kind: "lame" (a rider)': _lame_rider,
    'pieces[].moves[].kind: "lame" (stating its legs)': lambda s: any(
        m.get("kind") == "lame" and m.get("legs") for m in _moves(s)
    ),
    "rules.obstacles.blocks_leaps (leaps stating their legs)": lambda s: bool(
        (_rules(s).get("obstacles") or {}).get("blocks_leaps")
    ),
    "rules.castling.king": _castling_not_royal,
    "a piece that is not royal": _castling_not_royal,
    "pieces[].moves[].push": lambda s: any(m.get("push") for m in _moves(s)),
    "pieces[].sight (but a command that captures_own: push.N4)": lambda s: bool(_sights(s)),
    # SR8i (O9, O13, O14, O17, O18, O19): a held piece in its holder's frame.
    'pieces[].control: "majority" or "shared"': _player_moved,
    "pieces[].facing": lambda s: any(isinstance(p.get("facing"), dict) for p in _pieces(s)),
    "setup.placements[].facing": _placement_facing,
    "pieces[].moves (reading differently for the two players)": _reads_unalike,
    "pieces[].moves[].condition": lambda s: any(isinstance(m.get("condition"), dict) for m in _moves(s)),
    "pieces[].moves[].screen own or enemy": lambda s: any(
        m.get("screen") in ("own", "enemy") for m in _moves(s)
    ),
    # U2 (`unheld-pairs-plan.md` §2.2): nine pairs the core already played.
    "rules.income": lambda s: income_rule(s) is not None,
    'rules.scoring.count "points"': lambda s: (_rules(s).get("scoring") or {}).get("count") == "points",
    "rules.drops.gating": lambda s: "gating" in (_drops(s) or {}),
    "rules.reach.removes": lambda s: (_rules(s).get("reach") or {}).get("removes") is True,
    "rules.extinction": lambda s: isinstance(_rules(s).get("extinction"), dict),
    "rules.bare_royal": lambda s: _rules(s).get("bare_royal") not in (None, False, "none"),
    'pieces[].control: "shared"': lambda s: bool(shared_ids(s)),
    "pieces[].price (on a shared type)": lambda s: any(
        p.get("price") and p.get("control") == "shared" for p in _pieces(s)
    ),
    "pieces[].moves[].run or displace": lambda s: bool(run_hop_ids(s)),
    'pieces[].moves[].displace: "back"': lambda s: any(m.get("displace") == "back" for m in _moves(s)),
    # R-L4b stage 3: the capture list's entries, read off `capture_list` (either spelling).
    "rules.capture: a blast": _explode,
    "rules.capture: a hand": _from_captures,
    "rules.capture: a strike": lambda s: _strike(s) is not None,
    "rules.capture: contacts": lambda s: _contacts(s) is not None,
    "rules.capture: a blast that spares the taken piece": lambda s: any(
        b.get("spares_taken") for b in _entries(s, "blast")
    ),
    "rules.capture: a blast that spares the taken piece or is read": _blast_spares_or_is_read,
    'rules.capture: a strike placed "advance", "swap" or "choose"': lambda s: _strike(s) is not None
    and (_strike(s) or {}).get("position", "advance") in ("advance", "swap", "choose"),
    'rules.capture: a strike placed "choose"': lambda s: (_strike(s) or {}).get("position") == "choose",
    'rules.capture: contacts placed "choose"': lambda s: (_contacts(s) or {}).get("position") == "choose",
    'rules.capture: a strike placed "stay", "swap", "home" or "choose"': lambda s: _strike(s) is not None
    and (_strike(s) or {}).get("position", "advance") != "advance",
    "rules.capture: a strike that keeps its victim": lambda s: _strike(s) is not None
    and not strike_only_removes(_strike(s)),
    "rules.capture: contacts that keep their target": lambda s: _contacts(s) is not None
    and not contacts_only_remove(_contacts(s)),
    "rules.capture: a strike that switches sides": lambda s: bool((_strike(s) or {}).get("switches_sides")),
    "rules.capture: an entry reading some sweepers' path (path_by)": _path_by,
    "rules.capture: a blast beside a strike or contacts": lambda s: _explode(s)
    and (_strike(s) is not None or _contacts(s) is not None),
    "rules.purse.upkeep": lambda s: bool((_rules(s).get("purse") or {}).get("upkeep")),
    # R-L4b5: a blast beside a strike or contacts that does not stay (T1.rounds, T4.rounds).
    "rules.capture: a blast beside a strike that moves its taker": lambda s: _explode(s)
    and _strike(s) is not None and (_strike(s) or {}).get("position", "advance") != "stay",
    "rules.capture: a blast beside contacts that move their taker": lambda s: _explode(s)
    and _contacts(s) is not None and (_contacts(s) or {}).get("position", "advance") != "stay",
}


def unknown_keys(rows: tuple[Lifted, ...] | None = None) -> list[str]:
    """Key paths a `[[lifted]]` row names that no detector reads."""
    rows = lifted() if rows is None else rows
    return sorted({key for row in rows for key in row.pair if key not in DETECTORS})


def holds(spec: dict[str, Any], key: str) -> bool:
    """Whether ``spec`` holds the key path ``key``."""
    try:
        return bool(DETECTORS[key](spec))
    except (TypeError, AttributeError, KeyError, ValueError):
        return False


def pairs_present(spec: dict[str, Any], rows: tuple[Lifted, ...] | None = None) -> list[str]:
    """The ids of the `[[lifted]]` rows whose two key paths ``spec`` holds together."""
    rows = lifted() if rows is None else rows
    return [row.id for row in rows if holds(spec, row.pair[0]) and holds(spec, row.pair[1])]
