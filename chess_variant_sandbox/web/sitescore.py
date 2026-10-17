"""A score's own sentences in a visitor's words, sent beside the maintainer's.

`docs/plan-understanding.md` §3.3, rows G29 to G31 (task W3). The scoring writes its
verdicts for the person who runs it: *quick_kill_rate_worst_half=0.538 above maximum 0.5*,
*this roster cannot read this ruleset*, *the first seat took 41 of the 61 decided games
between the two strongest rungs*. Those sentences are right on a checkout and stay there.
The site draws these instead, where the page is the site (`ACCOUNTS`), the same way
`metrics/glossary.SITE_WORDS` gives each measured part a visitor's title.

**Words only.** Every number here is one the measurement stored, printed as it was
stored or rounded for a sentence; nothing is worked out, compared or re-decided. A
sentence this does not recognise reads as *it failed one of the tests a game must pass*
rather than as the maintainer's text, so a new kind of verdict can never put the
scoring's identifiers on the site; the test that holds every stored shape to a site
sentence (`tests/test_sitescore.py`) is where a new shape is noticed.
"""

from __future__ import annotations

import copy
import re
from typing import Any

# What a visitor reads for each test a game must pass (G29): its name, and why it is a
# test, keyed by the component the test reads. `units` follows a number where the bare
# number would not say what it counts. *Worse half* is defined once, above the list
# (`detail.js` `scoreHtml`): whichever half of the computer players, weaker or stronger,
# does worse.
SITE_GATES: dict[str, dict[str, str]] = {
    "search_advantage": {
        "title": "thinking beats random moves",
        "reason": "a computer player that thinks ahead must beat one that moves at random, "
                  "or there is nothing to play",
    },
    "stall_rate_worst_half": {
        "title": "games that never finish, worse half",
        "reason": "a game that mostly fails to finish is not a game, and the half of the "
                  "computer players that mostly fails to finish it is not rescued by the "
                  "other half finishing",
    },
    "mean_branching": {
        "title": "choices per turn",
        "reason": "with no choices there is no play",
    },
    "decisive_rate_worst_half": {
        "title": "games with a winner, worse half",
        "reason": "a game that good players almost always draw is not a contest, and the "
                  "games their weaker opponents lost outright do not make it one",
    },
    "mean_plies_worst_half": {
        "title": "game length, worse half",
        "units": "turns",
        "reason": "a game decided before either player has developed anything is not a "
                  "game, and an average with long shuffles among the other half would "
                  "describe no game that was played",
    },
    "skill_gradient": {
        "title": "does thinking longer win",
        "units": "rating points",
        "reason": "thinking twice as long must be worth at least a few rating points, or "
                  "skill does not exist here",
    },
    "quick_kill_rate_worst_half": {
        "title": "quick kills, worse half",
        "reason": "a game where half of all games are over in a handful of moves is decided "
                  "by an opening trap, not by play; it is read on the half of the computer "
                  "players that falls into the trap",
    },
    "seat_decided_chance": {
        "title": "won by one side",
        "reason": "the number is the chance that a fair game would split as one-sidedly "
                  "between the two strongest computer players; a game they win from one "
                  "side nearly every time, whoever plays it, is decided by the side and not "
                  "by play",
    },
    "top_three_seat_decided_chance": {
        "title": "won by one side, strongest three",
        "reason": "the same chance, over the games among the three strongest computer "
                  "players; a game they win from one side far more often than the other is "
                  "decided by the side and not by play",
    },
}

_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve")

FAILED = "it failed one of the tests a game must pass"


def _word(count: int) -> str:
    return _WORDS[count] if 0 <= count < len(_WORDS) else str(count)


def _title(component: str) -> str:
    from chess_variant_sandbox.metrics.glossary import SITE_WORDS, title_of

    found = SITE_GATES.get(component) or SITE_WORDS.get(component) or {}
    return found.get("title") or title_of(component)


def _strongest(component: str) -> str:
    return "three" if component.startswith("top_three") else "two"


def _percent(share: float, tolerated: float) -> tuple[str, str]:
    """A share and the share allowed, as whole percentages unless that makes them equal."""
    whole = (f"{share:.0%}", f"{tolerated:.0%}")
    return whole if whole[0] != whole[1] else (f"{share:.1%}", f"{tolerated:.1%}")


_PLACE = re.compile(r"^run (\d+): (.*)$", re.S)
_UNREADABLE = re.compile(
    r"^this roster cannot read this ruleset: (.*?), so (.*) would describe the players "
    r"rather than the game$", re.S)
_ABSENT = re.compile(r"^could not be measured on this \w+: (.*)$", re.S)
_UNMEASURED = re.compile(r"^(\w+) was not measured$")
_SEAT_RUN = re.compile(
    r"^(\w+)=\S+ below minimum \S+: the (first|second) seat went (\d+)-(\d+) with (\d+) "
    r"draws over the (\d+) games", re.S)
# Since Q953a the per-run sentence adds the split it tested, each different game once.
_SEAT_RUN_DISTINCT = re.compile(r", (\d+)-(\d+) with each different game counted once")
_SEAT_POOLED = re.compile(
    r"^(\w+)=\S+ below minimum \S+: the (first|second) seat took (\d+)% of the (\d+) "
    r"decided games(, each different game counted once)?.* rules out a seat held to "
    r"(\d+)% or less", re.S)
_GATE = re.compile(r"^(\w+)=(\S+) (below minimum|above maximum) (\S+?)(?: \(.*\))?$", re.S)


def violation(text: str, seeds: list[int] | None = None) -> str:
    """One of the scoring's refusals, in a visitor's words.

    `seeds` are the score's runs, so a refusal the four runs were pooled for, filed as
    *run 2000: ...*, is named by its place among them (*run 3 of 4*), as the page names a
    refusing run (A15).
    """
    text = str(text).strip()
    placed = _PLACE.match(text)
    if placed:
        seed, rest = int(placed.group(1)), placed.group(2)
        order = sorted(seeds or [])
        where = (f"run {order.index(seed) + 1} of {len(order)}" if seed in order
                 else "one run")
        return f"{where}: {violation(rest)}"
    found = _UNREADABLE.match(text)
    if found:
        why, parts = found.groups()
        weaker = "rates *worse*" in why
        count = len([part for part in parts.split(",") if part.strip()])
        said = ("the computer players did worse with more thinking" if weaker
                else "the computer players didn't get clearly stronger with more thinking")
        return (f"{said}, so this run has no score: {_word(count)} of the things measured "
                "would describe the computer players rather than the game")
    found = _ABSENT.match(text)
    if found:
        names = [_title(part.strip()) for part in found.group(1).split(",") if part.strip()]
        return f"{', '.join(names)} could not be measured on this game"
    found = _UNMEASURED.match(text)
    if found:
        return f"{_title(found.group(1))} could not be measured on this game"
    found = _SEAT_RUN.match(text)
    if found:
        # The first seat's record; the side that won most is whichever it favours.
        component, _, wins, losses, draws, _ = found.groups()
        side = "first" if int(wins) >= int(losses) else "second"
        most, least = sorted((int(wins), int(losses)), reverse=True)
        said = (f"the {_strongest(component)} strongest computer players won from the "
                f"{side} side almost every time: {most} games to {least}, with {draws} "
                "drawn")
        once = _SEAT_RUN_DISTINCT.search(text)
        if once and (int(once.group(1)), int(once.group(2))) != (int(wins), int(losses)):
            many, few = sorted((int(once.group(1)), int(once.group(2))), reverse=True)
            said += f" ({many} to {few} counting each different game once)"
        return said
    found = _SEAT_POOLED.match(text)
    if found:
        component, side, share, decided, each_once, allowed = found.groups()
        different = " different" if each_once else ""
        return (f"the {_strongest(component)} strongest computer players won {share}% of "
                f"their {decided}{different} decided games from the {side} side, far more "
                f"than the {allowed}% allowed")
    found = _GATE.match(text)
    if found:
        component, value, side, bound = found.groups()
        units = SITE_GATES.get(component, {}).get("units", "")
        unit = f" {units}" if units else ""
        limit = (f"under the {bound}{unit} it must reach" if side == "below minimum"
                 else f"over the {bound}{unit} it must stay under")
        return f"{_title(component)} came to {value}{unit}, {limit}"
    return FAILED


def split(verdict: dict[str, Any] | None, viable: bool | None) -> str:
    """How the runs split, in a visitor's words (*pass it*, *rule it out*), or nothing.

    From the two counts the measurement stored, as `metrics/seeds.py` words them: nothing
    when every run agreed or there was one.
    """
    if not verdict or not verdict.get("split"):
        return ""
    accept, refuse = int(verdict.get("accept") or 0), int(verdict.get("refuse") or 0)
    total = accept + refuse
    if total < 2 or not accept or not refuse:
        return ""
    ending = "it stays in" if viable else "it is ruled out"
    head = (f"{_word(accept).capitalize()} of its {_word(total)} runs "
            f"{'passes' if accept == 1 else 'pass'} it and {_word(refuse)} "
            f"{'rules' if refuse == 1 else 'rule'} it out")
    if accept == refuse:
        return f"{head}. A tie counts {'for' if viable else 'against'} it, so {ending}."
    return f"{head}, so {ending}."


def lean(seat: dict[str, Any] | None) -> str:
    """How far a ranked game leans toward one side, in a visitor's words, or nothing.

    From the band whose lean the seat check printed, and only where it printed one.
    """
    if not seat or seat.get("refuses") or not seat.get("lean"):
        return ""
    tolerated = float(seat.get("tolerated_share") or 0)
    # Counts rather than two rounded shares: *67% against the 67% allowed* is what a share
    # a hair over two in three reads as.
    allowed = ("two in three" if abs(tolerated - 2 / 3) < 0.001
               else _percent(tolerated, tolerated)[1])
    for band in (seat.get("bands") or {}).values():
        if not band.get("lean"):
            continue
        took = max(int(band.get("wins") or 0), int(band.get("losses") or 0))
        decided = int(band.get("decided") or 0)
        # A printed lean is always past the line: since Q625 the check compares the exact
        # split, so a split of exactly two in three prints none (replayed, Q644).
        return (f"the {_strongest(str(band.get('component', '')))} strongest computer "
                f"players won {took} of their {decided} decided games from the "
                f"{band.get('favoured', 'one')} side, more than the {allowed} allowed, though not "
                "by enough to rule the game out")
    return ""


def worded(score: dict[str, Any] | None) -> dict[str, Any] | None:
    """A copy of a stored score whose sentences each carry a visitor's words as `site`.

    The score itself is untouched (a game's score is memoised and read by everything).
    Each run gains `site.violations`, the verdict `site.split`, the seat `site.lean`, and
    the score `site.violations`: every sentence the score panel and the card draw, beside
    the one the measurement wrote.
    """
    if not score:
        return score
    found = copy.deepcopy(score)
    seeds = [int(seed) for seed in found.get("seeds") or []]
    runs(found.get("runs"))
    verdict = found.get("verdict")
    said = split(verdict, found.get("viable"))
    if isinstance(verdict, dict):
        verdict["site"] = {"split": said}
    if isinstance(found.get("seat"), dict):
        found["seat"]["site"] = {"lean": lean(found["seat"])}
    found["site"] = {"violations": [
        said if verdict and text == verdict.get("split") else violation(text, seeds)
        for text in found.get("violations") or []
    ]}
    return found


def runs(listed: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Each run's refusals in a visitor's words, as its `site.violations`, in place."""
    for run in listed or []:
        run["site"] = {"violations": [violation(text) for text in run.get("violations") or []]}
    return listed


def guide(explained: dict[str, Any] | None) -> dict[str, Any] | None:
    """The scoring rules' own guide (`glossary.explain`), each test carrying its `site`."""
    if not explained:
        return explained
    found = dict(explained)
    found["gates"] = [
        {**gate, "site": {"title": _title(gate["component"]),
                          **SITE_GATES.get(gate["component"], {})}}
        for gate in explained.get("gates") or []
    ]
    return found
