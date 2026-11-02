"""A game's small endings, as plain facts for its page (N3 of `docs/reference/no-forced-mate.md`).

The endgame solver (N1, `engine/endgame.py`) was run over the library by N2 (§8 of that
record), which wrote one record a game, `runs/endgames/<hash>.json`: every ending with the
royals and at most one other piece, solved to the end, and the stalled records of the four
library seeds replayed against those tables. This reads that record and says what it found,
in sentences a visitor reads under the game's full rules:

- which one-piece endings force a win, from what share of positions and in at most how many
  moves; which cannot be forced; which the rules decide the moment they arise;
- a pair of bare royals where one side can force a win against the other (the wolves can
  trap a king, a duck can box one in), and nothing about them where neither can;
- the one-piece endings too big to solve under the run's cap;
- how often the computer players' stalled games reached such an ending, and in how many of
  those one side had a forced win it did not take.

**Facts only.** Nothing here is a draw rule, an adjudication or advice: it says what can be
forced, never what to do or how a game should end (§5 of the record).

**A record, never a live engine.** The page reads the persisted record (CLAUDE.md: metrics
read records) and solves nothing. The record carries its basis (`basis.build`, the core
build it was solved with, and `basis.date`); a reader whose core is another build shows the
facts marked *provisional*, with the date, never as current
(`.claude/rules/metrics-and-calibration.md`). The record is keyed by the game's hash, so an
edit to the rules is another game and finds no record.

**Absent without error.** No record, a game the solver refuses (a capture hand, the purse,
rounds, spies), a game without royals (whose small endings N2 counted only in aggregate), a
record that does not fit the spec, or one where nothing small enough was solved: `sheet`
answers `None` and the page shows no section at all, the same way in every case.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chess_variant_sandbox.measurement import built_fingerprint

DIRECTORY = "endgames"
"""Where the records sit under `runs/`: `runs/endgames/<hash>.json`."""

TITLE = "Endings with few pieces"

RARE = 0.01
"""Below this share of the defender-to-move positions, a forced win is said to come only from
a rare few positions (the defender already boxed in, §8.3), not given as a percentage that
rounds to nought."""

_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)

_cache: dict[str, tuple[tuple[int, int], dict[str, Any] | None]] = {}


def record_path(runs_dir: Path | str, digest: str) -> Path:
    return Path(runs_dir) / DIRECTORY / f"{digest}.json"


def load(runs_dir: Path | str, digest: str) -> dict[str, Any] | None:
    """The game's record, or `None` where there is none or it does not read; kept until the
    file changes, so a page asked for often reads the disk once."""
    if not re.fullmatch(r"[0-9a-f]{8,64}", digest or ""):
        return None
    path = record_path(runs_dir, digest)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = str(path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    held = _cache.get(key)
    if held is not None and held[0] == stamp:
        return held[1]
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        found = None
    if not isinstance(found, dict) or found.get("hash") != digest:
        found = None
    _cache[key] = (stamp, found)
    return found


def sheet_for(
    runs_dir: Path | str, digest: str, spec: Mapping[str, Any], built: str | None = None
) -> dict[str, Any] | None:
    """The page's payload for one game, or `None` where it shows nothing."""
    record = load(runs_dir, digest)
    if record is None:
        return None
    try:
        return sheet(record, spec, built)
    except (KeyError, IndexError, TypeError, ValueError):
        # A record from another shape of solver output, or one that does not fit this spec,
        # is no fact sheet; the page says nothing rather than fail.
        return None


# ------------------------------------------------------------------------ the words


def _article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _day(iso: Any) -> str:
    found = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(iso or ""))
    if not found:
        return ""
    return f"{int(found.group(3))} {_MONTHS[int(found.group(2)) - 1]} {found.group(1)}"


def _either(names: Sequence[str]) -> str:
    """`a pawn, a knight or a rook`."""
    said = [_article(name) for name in names]
    if len(said) <= 1:
        return "".join(said)
    return ", ".join(said[:-1]) + " or " + said[-1]


def _count(n: int) -> str:
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return words[n] if 0 <= n < len(words) else str(n)


def _moves(plies: int, turn: int) -> str:
    """The winner's moves in a line of `plies` plies: a mate in n moves is 2n - 1 plies, and
    a duck game's turn is two plies (a move, then the duck)."""
    n = max(1, math.ceil(plies / (2 * turn)))
    return f"{n} move{'' if n == 1 else 's'}"


def _share(set_: Mapping[str, Any], player: int) -> float:
    """The share of the positions with `player`'s opponent to move that are lost for it."""
    theirs = set_["mover"][1 - player]
    total = theirs["won"] + theirs["lost"] + theirs["drawn"]
    return theirs["lost"] / total if total else 0.0


def _percent(share: float) -> str:
    return f"{max(1, round(100 * share))}%"


def _at_once(set_: Mapping[str, Any]) -> bool:
    """A set the rules decide on the board, before anybody moves (`n2_table.py`'s test)."""
    return (
        set_["drawn"] == 0 and set_["won"] + set_["lost"] > 0
        and all(m["longest_win"] == 0 and m["longest_loss"] == 0 for m in set_["mover"])
    )


def _forced_words(shares: Sequence[float], plies: Sequence[int], turn: int) -> str:
    worst = min(shares)
    moves = _moves(max(plies), turn)
    if max(shares) < RARE:
        return (f"a win can be forced only from a rare few positions (fewer than 1 in 100), "
                f"in at most {moves}")
    low, high = _percent(worst), _percent(max(shares))
    where = low if low == high else f"{low} to {high}"
    return f"a win can be forced from {where} of positions, in at most {moves}"


# ----------------------------------------------------------------------- the sheet


def sheet(
    record: Mapping[str, Any], spec: Mapping[str, Any], built: str | None = None
) -> dict[str, Any] | None:
    """The facts of one record, worded, with their basis; `None` where there is nothing to
    say. `built` is the reader's core build (`measurement.built_fingerprint` by default)."""
    if record.get("refused") or not record.get("royals"):
        return None
    sets = [s for s in record.get("sets") or [] if isinstance(s, Mapping)]
    if not sets or "error" in sets[0]:
        return None
    pieces = list(spec.get("pieces") or [])
    names = [str(p.get("name") or p.get("id") or "piece") for p in pieces]
    rules = spec.get("rules") or {}
    turn = 2 if isinstance(rules, Mapping) and rules.get("duck") else 1

    base = [list(side) for side in sets[0]["material"][:2]]
    if any(i >= len(names) for side in base for i in side):
        return None
    royal = [" and ".join(names[i] for i in side) for side in base]

    # One-piece endings, by the extra piece's type: for each player who could hold it, the
    # solved set (or the reason it was not solved).
    found: dict[int, dict[int, Mapping[str, Any]]] = {}
    for set_ in sets:
        material = set_["material"]
        for p in (0, 1):
            mine, theirs = Counter(material[p]), Counter(material[1 - p])
            if theirs != Counter(base[1 - p]):
                continue
            extra = mine - Counter(base[p])
            if sum(extra.values()) != 1 or sum(mine.values()) != len(base[p]) + 1:
                continue
            (kind,) = extra
            if kind >= len(names):
                return None
            found.setdefault(kind, {}).setdefault(p, set_)

    lines: list[str] = []
    too_big: list[str] = []
    for kind in sorted(found):
        by_player = found[kind]
        name = names[kind]
        if any("error" in s for s in by_player.values()):
            too_big.append(name)
            continue
        who = f"{_article(royal[0])} and {_article(name)} against a lone {royal[1]}"
        if any(_at_once(s) for s in by_player.values()):
            lines.append(f"{_capital(who)}: the rules decide it as soon as it arises.")
            continue
        forced = {p: bool(s["forced"][p]) for p, s in by_player.items()}
        if all(forced.values()):
            words = _forced_words(
                [_share(s, p) for p, s in by_player.items()],
                [s["mover"][p]["longest_win"] for p, s in by_player.items()], turn)
            lines.append(f"{_capital(who)}: {words}.")
        elif not any(forced.values()):
            lines.append(f"{_capital(who)}: a win cannot be forced.")
        else:
            p = next(q for q, yes in forced.items() if yes)
            s = by_player[p]
            words = _forced_words([_share(s, p)], [s["mover"][p]["longest_win"]], turn)
            lines.append(
                f"{_capital(who)}: with the extra piece, the {('first', 'second')[p]} player "
                f"can force a win and the {('second', 'first')[p]} cannot; {words}."
            )

    # Bare royals, said only where somebody can force a win (it is news only then).
    bare = sets[0]
    bare_lines: list[str] = []
    if not _at_once(bare) and any(bare["forced"][:2]):
        players = [p for p in (0, 1) if bare["forced"][p]]
        words = _forced_words(
            [_share(bare, p) for p in players],
            [bare["mover"][p]["longest_win"] for p in players], turn)
        bare_lines.append(
            f"{_capital(_article(royal[0]))} against a lone {royal[1]}, with nothing else "
            f"on the board: {words}."
        )

    if not lines and not bare_lines:
        return None
    if too_big:
        lines.append(
            f"With {_either(too_big)} added there are too many positions to check, so those "
            "endings are not worked out."
        )

    basis = record.get("basis") or {}
    current = bool(basis.get("build")) and basis.get("build") == (
        built if built is not None else built_fingerprint())
    day = _day(basis.get("date"))
    if current:
        stamp = f"Worked out by computer{f' on {day}' if day else ''}, checking every position."
    else:
        stamp = (f"Provisional: worked out{f' on {day}' if day else ''} with an earlier version "
                 "of the game engine, and it will be worked out again.")

    return {
        "title": TITLE,
        "intro": (f"What can be forced when {_article(royal[0])} has at most one other piece "
                  f"against a lone {royal[1]}, with best play from both sides."),
        "lines": bare_lines + lines,
        "stalls": _stalls(record, royal),
        "stamp": stamp,
        "status": "current" if current else "provisional",
        "measuredAt": str(basis.get("date") or ""),
        "build": str(basis.get("build") or ""),
    }


def _stalls(record: Mapping[str, Any], royal: Sequence[str]) -> str:
    """How the computer players' stalled games met these endings, or empty where the record
    holds none (`stalls` lists every `ply_limit` and `no_progress` record of the four library
    seeds, each with its tail's labels)."""
    stalls = [s for s in record.get("stalls") or [] if isinstance(s, Mapping)]
    if not stalls:
        return ""
    reached = missed = 0
    for one in stalls:
        labels = one.get("labels") or {}
        if labels.get("won", 0) + labels.get("cursed", 0) + labels.get("drawn", 0):
            reached += 1
            missed += bool(labels.get("won", 0))
    seeds = (record.get("basis") or {}).get("seeds") or []
    runs = f" (over {_count(len(seeds))} runs)" if len(seeds) > 1 else ""
    k = int(record.get("k") or 1)
    small = "one other piece" if k == 1 else f"{_count(k)} other pieces"
    n = len(stalls)
    games = f"{n} game{'' if n == 1 else 's'}"
    lead = (f"Of the computer players' {games} here that were stopped for going on too "
            f"long{runs}, ")
    if not reached:
        return lead + f"none ended with at most {small} besides the {royal[0]}s."
    said = (f"in {missed} of those one side had a win it could force and did not take it"
            if missed else "in none of those did either side have a win it could force")
    return lead + f"{reached} ended with at most {small} besides the {royal[0]}s, and {said}."
