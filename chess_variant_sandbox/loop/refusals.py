"""What a refusal rests on: a fact of the rules, or a reading of these players.

One rule, read by two places (R-S17a, R-S17c; `docs/reference/wrong-no.md` §2 items 1 and 2):
the site's words for a refusal (`web/sitescore.py`) and the design loop's memory of one
(`loop/orchestrator.py`, `loop/archive.py`). A refusal splits by *which check failed*, never
by a new verdict:

- **by its rules** (`RULES`): true whoever plays. No move at the start, over before the first
  move, the rules not fitting together, and under one and a half moves a turn (the cheap
  check's, and the weights' `mean_branching`).
- **as these players play it** (`PLAYERS`): every check that reads games the computer players
  played, random play included. Such a verdict can change with stronger players or a line
  drawn elsewhere, so the loop measures it again when either changes, and the site says so.

A sentence nothing here recognises reads as `PLAYERS`, the kind that admits it may be wrong:
the costly error is the wrong "no" (the owner, 2026-11-15).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

RULES = "rules"
PLAYERS = "players"
BOTH = "both"

# The scored tests that are facts of the rules. `mean_branching` reads the positions the
# games reached, but under one and a half moves a turn is a property of the rules at any
# strength (`wrong-no.md` §3, the cheap check's structural refusals).
RULES_GATES = frozenset({"mean_branching"})

# The cheap check's refusals (`loop/gates.py::smoke_test`) that are facts of the rules, by
# the start of the sentence it writes. Every other one reads random games.
SMOKE_RULES = (
    "the rules don't fit together",
    "the first player has no legal move",
    "a player has no legal move",
    "the game is over before the first move",
)
SMOKE_BRANCHING = re.compile(r"^a player has [\d.]+ moves to choose from on average")

# The scoring's sentences (`metrics/index.py`, `metrics/seat.py`, `metrics/seeds.py`), as
# far as telling which test they name needs. `web/sitescore.py` words them from the same
# patterns.
PLACE = re.compile(r"^run (\d+): (.*)$", re.S)
UNREADABLE = re.compile(
    r"^this roster cannot read this ruleset: (.*?), so (.*) would describe the players "
    r"rather than the game$", re.S)
ABSENT = re.compile(r"^could not be measured on this \w+: (.*)$", re.S)
UNMEASURED = re.compile(r"^(\w+) was not measured$")
SEAT_RUN = re.compile(
    r"^(\w+)=\S+ below minimum \S+: the (first|second) seat went (\d+)-(\d+) with (\d+) "
    r"draws over the (\d+) games", re.S)
SEAT_POOLED = re.compile(
    r"^(\w+)=\S+ below minimum \S+: the (first|second) seat took (\d+)% of the (\d+) "
    r"decided games(, each different game counted once)?.* rules out a seat held to "
    r"(\d+)% or less", re.S)
# The third verdict's own sentence (`seeds._violations`): no verdict, not a failure.
NOT_MEASURED = re.compile(r"^Not measured: (.*?)\.?$", re.S)
GATE = re.compile(r"^(\w+)=(\S+) (below minimum|above maximum) (\S+?)(?: \(.*\))?$", re.S)
_SPLIT = re.compile(r"^(\w+ of its \w+ runs|Of its \w+ runs)")

# What `violation_check` names a refusal that is not one scored test.
ABSENT_CHECK = "absent"
UNKNOWN_CHECK = "unknown"
# What `count` names a refusal that names no failed test: *this roster cannot read* or
# *not measured*, the players' part of a "no" by definition.
UNREAD_CHECK = "unread"


def smoke_kind(reason: str) -> str:
    """One of the cheap check's refusals as `RULES` or `PLAYERS` (random play)."""
    text = str(reason).strip()
    if text.startswith(SMOKE_RULES) or SMOKE_BRANCHING.match(text):
        return RULES
    return PLAYERS


def gate_kind(component: str) -> str:
    """A scored test, by the component it reads, as `RULES` or `PLAYERS`."""
    return RULES if component in RULES_GATES else PLAYERS


def component_of(text: str) -> str | None:
    """The component a scored refusal names, or None for any other sentence."""
    found = GATE.match(text) or SEAT_RUN.match(text) or SEAT_POOLED.match(text)
    return found.group(1) if found else None


def violation_check(text: str) -> tuple[str, str] | None:
    """One stored scoring refusal as the check it names and its kind, or None for a
    sentence that is not a failed test: a split, *not measured*, or *this roster cannot
    read*. The check is the component, `ABSENT_CHECK` for a measurement the score needs and
    did not get, or `UNKNOWN_CHECK`."""
    text = str(text).strip()
    placed = PLACE.match(text)
    if placed:
        text = placed.group(2)
    if NOT_MEASURED.match(text) or UNREADABLE.match(text) or _SPLIT.match(text):
        return None
    component = component_of(text)
    if component is not None:
        return component, gate_kind(component)
    if ABSENT.match(text) or UNMEASURED.match(text):
        return ABSENT_CHECK, PLAYERS
    return UNKNOWN_CHECK, PLAYERS


# The cheap rungs' sentences (`loop/gates.py`, `loop/screen.py`) by how they start, to the
# check each names. Only for reading an archive written before a refusal stored its checks
# (`refusal_of`); a refusal written since carries them, and a test holds the two to one name.
SENTENCE_CHECKS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), check) for pattern, check in (
        (r"^the rules don't fit together", "rules_fit"),
        (r"^the computer can't play one of these rules", "core_plays"),
        (r"^(the first player|a player) has no legal move", "opening_moves"),
        (r"^the game is over before the first move", "over_at_start"),
        (r"^random games of these rules take too long", "clock"),
        (r"^a player has [\d.]+ moves to choose from", "mean_branching"),
        (r"^\d+% of random games ended", "stall_rate"),
        (r"^none of \d+ random games had a winner", "decided"),
        (r"^\d+% of games end within", "forced_rate"),
        (r"^one side wins", "seat_skew"),
        (r"^no computer player here can test it", "roster"),
        (r"^no games were played", "games"),
        (r"^no computer player won any", "decisive"),
        (r"^\d+% of games were over within", "quick_kill_rate"),
    )
)

SMOKE = "smoke"
SCREEN = "screen"
FULL = "full"
RUNGS = (SMOKE, SCREEN, FULL)


def sentence_check(text: str) -> str:
    """The check one of the cheap rungs' sentences names, or `UNKNOWN_CHECK`."""
    text = str(text).strip()
    for pattern, check in SENTENCE_CHECKS:
        if pattern.match(text):
            return check
    return UNKNOWN_CHECK


def kind_of(kinds: Iterable[str]) -> str:
    """The kind of a refusal made of checks of these kinds: `BOTH` where they differ, and
    `PLAYERS` where there are none, the kind that admits it may be wrong."""
    found = set(kinds)
    if not found:
        return PLAYERS
    return BOTH if len(found) > 1 else next(iter(found))


# ------------------------------------------------------------------ the loop's memory (R-S17c)
#
# A refused child used to be remembered by its hash alone and never measured again, whatever
# changed (`wrong-no.md` §1.2). Its archive entry now carries `refusal`: the rung that refused
# it, each check with the number it read, the line and the margin, the kind by the rule above,
# and the basis it was refused on -- the rung's players, the rung's lines and the weights
# version. `retriable` is the one test of whether that basis still holds.


def _row(check: str, kind: str, reason: str, found: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"check": check, "kind": kind, "reason": reason}
    for key in ("value", "line", "side", "margin"):
        row[key] = (found or {}).get(key)
    return row


def cheap_refusal(
    rung: str,
    reasons: Sequence[str],
    checks: Sequence[dict[str, Any]],
    roster: Sequence[str],
    lines: dict[str, float],
    weights_version: str,
    seed: int,
) -> dict[str, Any]:
    """The refusal of the cheap check or the screen, from its sentences and the rows its
    gate wrote beside them (`gates.check_row`), one for one and in order."""
    rows = []
    for at, reason in enumerate(reasons):
        found = checks[at] if at < len(checks) else None
        check = str(found["check"]) if found else sentence_check(reason)
        rows.append(_row(check, smoke_kind(reason), str(reason), found))
    return {
        "rung": rung,
        "kind": kind_of(row["kind"] for row in rows),
        "checks": rows,
        "roster": list(roster),
        "lines": dict(lines),
        "weights_version": weights_version,
        "seed": seed,
    }


def _numbers(text: str) -> dict[str, Any] | None:
    """The value, line, side and margin of a scored refusal that states them."""
    gate = GATE.match(str(text).strip())
    if not gate:
        return None
    try:
        value, line = float(gate.group(2)), float(gate.group(4))
    except ValueError:
        return None
    return {
        "value": value,
        "line": line,
        "side": "min" if gate.group(3) == "below minimum" else "max",
        "margin": round(abs(value - line), 4),
    }


def full_refusal(
    violations: Sequence[str],
    roster: Sequence[str],
    weights_version: str,
    seed: int,
) -> dict[str, Any]:
    """The refusal of the full roster, from the scoring's sentences. A sentence that is not
    a failed test (*this roster cannot read*, *not measured*) adds no check; a refusal with
    none is `PLAYERS`, since what it lacks is a reading these players could give."""
    rows = []
    for text in violations:
        found = violation_check(text)
        if found is not None:
            rows.append(_row(found[0], found[1], str(text), _numbers(text)))
    return {
        "rung": FULL,
        "kind": kind_of(row["kind"] for row in rows),
        "checks": rows,
        "roster": list(roster),
        "weights_version": weights_version,
        "seed": seed,
    }


def basis_differs(refusal: dict[str, Any], basis: dict[str, Any]) -> bool:
    """Whether anything the refusal was read on has changed: the rung's players, its lines,
    or, at the full roster, the weights version. Compared on what `basis` names, so a rung
    is never re-tried for a change it does not read: the cheap rungs read no weights, and
    under one seed and one roster they would play the very games that refused it again."""
    return any(_plain(refusal.get(key)) != _plain(value) for key, value in basis.items())


def retriable(refusal: dict[str, Any] | None, basis: dict[str, Any]) -> bool:
    """Whether the loop measures a refused child again rather than skipping its hash.

    Only a refusal that stored its basis (an entry written before R-S17c has none and is
    final, as it always was), only one read entirely off the players (`PLAYERS`: a refusal
    with any check by the rules stays final, true whoever plays), and only when its basis
    has moved."""
    if not refusal or refusal.get("reconstructed") or refusal.get("kind") != PLAYERS:
        return False
    return basis_differs(refusal, basis)


def _plain(value: Any) -> Any:
    """A value as JSON gives it back, so a tuple and the list it was stored as are equal."""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def refusal_of(entry: Any) -> dict[str, Any] | None:
    """An archive entry's refusal: the stored one, or for an entry written before refusals
    were stored, one read back from its sentences (`reconstructed`, with no basis, so it is
    never `retriable`). None for an entry nothing refused."""
    stored = getattr(entry, "refusal", None)
    if stored:
        return dict(stored)
    detail = getattr(entry, "detail", None) or {}
    status = getattr(entry, "status", "")
    rows: list[dict[str, Any]] = []
    if status == "rejected":
        screened = detail.get("screen")
        rung, said = (SCREEN, screened) if screened else (SMOKE, detail.get("smoke"))
        for text in (said or {}).get("reasons") or []:
            rows.append(_row(sentence_check(text), smoke_kind(text), str(text), None))
    elif status == "evaluated" and not getattr(entry, "viable", True):
        rung = FULL
        for text in detail.get("violations") or []:
            found = violation_check(text)
            if found is not None:
                rows.append(_row(found[0], found[1], str(text), _numbers(text)))
    else:
        return None
    return {
        "rung": rung,
        "kind": kind_of(row["kind"] for row in rows),
        "checks": rows,
        "reconstructed": True,
    }


def count(entries: Iterable[Any]) -> Counter[tuple[str, str, str]]:
    """How many children were refused, by rung, check and kind: one count per check, so a
    child refused on two checks counts under both, and one refused with no failed test named
    (the roster could not read it, or it was not measured) counts once as `UNREAD_CHECK`.
    Starting points (no parent) are not children."""
    counted: Counter[tuple[str, str, str]] = Counter()
    for entry in entries:
        if getattr(entry, "parent", None) is None:
            continue
        refusal = refusal_of(entry)
        if refusal is None:
            continue
        rows = refusal.get("checks") or [{"check": UNREAD_CHECK, "kind": refusal["kind"]}]
        for row in rows:
            counted[(str(refusal["rung"]), str(row["check"]), str(row["kind"]))] += 1
    return counted
