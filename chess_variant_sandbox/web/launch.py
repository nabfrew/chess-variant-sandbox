"""The launch set, the front page's featured game, and the first game of the month.

`docs/plan-launch-set.md` §3 (the nine and the reserves, in order) and §5 (the featured
game and this month's game), as the user answered them on 2026-10-03
(`docs/archive/decisions.md`): there is no person test, so the nine are the pool and not a
gate; the front page features the five a chess player can start without reading the rules,
in turn by day (answer 6 A); the first game of the month is *reckoning* (answer 7 A).

**One place.** Every page that names the launch set reads it from here: the front page's
first step (L6, below), the front door's list (L1, Q559), this month's game (C2, Q603) and
the refresher's priority (`refresher.LAUNCH_SET`, which a test holds to this list).
The refresher keeps a copy rather than importing this one, on purpose (Q728): importing any
`web` module runs `web/__init__`, which loads the server, which imports `refreshing` and so
`refresher` itself. A lazy import would break the cycle only by making the measuring loop
load the web server for nine names, and the core layer would then depend on the page layer;
`tests/test_launch.py`'s equality test is the cheaper bond.

**Each game is pinned by its hash as well as its name.** A game's identity is its
mechanics hash (`canonical.MECHANIC_KEYS`): an in-place edit of its rules is a new game,
and the launch set must not quietly come to mean it. `tests/test_launch.py` fails, naming
the game, when a pinned hash or name is no longer in the library; the fix is a deliberate
edit here, never a lookup that follows the name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class Launch:
    """One game of the launch set: its name, its hash, and the one sentence that says
    how it differs from chess (`plan-launch-set.md` §3.2's *how it is learned*), which the
    first step shows under its board."""

    name: str
    hash: str
    sentence: str


# The nine, in `plan-launch-set.md` §3.2's order: the order the front door lists them in.
LAUNCH_SET: tuple[Launch, ...] = (
    Launch("dobutsu", "a2f0c65ceedf",
           "Shogi for children on twelve squares: captured pieces come back as yours, and "
           "your lion reaching the far row wins."),
    Launch("los alamos", "fcbb53d53ce4",
           "Chess on six by six, without bishops, castling or the pawn's double step."),
    Launch("cylinder chess", "a9306688fe57",
           "Chess, except that the left and right edges are joined: a piece leaving one "
           "side comes back on the other."),
    Launch("three-check", "b1f536573c22",
           "Chess, but the third check wins."),
    Launch("crazyhouse", "bb5d97520084",
           "Chess, and a piece you capture joins your hand: instead of a move you may put "
           "it on any empty square."),
    Launch("quarry", "51d9b36b4e93",
           "Chess, and four squares in the middle are walls that nothing crosses, not even "
           "a knight."),
    Launch("reckoning", "ffc7b604bea0",
           "Six by six, with small short-ranged armies: trap the king, or have more pieces "
           "standing when the game runs out of moves."),
    Launch("wolves", "ceb4916bdabc",
           "Chess with four wolves that belong to nobody: they walk their paths after every "
           "move and eat what they find."),
    Launch("hex glinski hexagon", "6eaaad41673b",
           "Chess on a board of hexagons, where pieces move in six and twelve directions."),
)

# The reserves, in §3.3's order. A reserve joins the set only by a deliberate edit here.
RESERVES: tuple[Launch, ...] = (
    Launch("intransitive", "29d5213b02b2", ""),
    Launch("atomic", "6bd4d9c5769e", ""),
    Launch("one-way cylinder chess", "b02bbe8da72c", ""),
    Launch("succession", "e7293070b5be", ""),
    Launch("shogi", "54c731ca6e95", ""),
    Launch("xiangqi", "151582aed409", ""),
)

# The front page's rotation (answer 6 A), in the order the user gave it: one a day, the
# same for everybody, by the calendar day in UTC. Each must be in `LAUNCH_SET`.
FEATURED: tuple[str, ...] = (
    "cylinder chess",
    "three-check",
    "quarry",
    "los alamos",
    "crazyhouse",
)

# The day the rotation shows `FEATURED[0]`. §5.1 starts it at *cylinder chess* on the day
# the invitations go out; until that day is known it is the day the rotation landed. Moving
# it re-aims the cycle and changes nothing else.
ROTATION_START = date(2026, 10, 3)

# The first game of the month (answer 7 A), for this month's game (C2, Q603) to read.
FIRST_MONTH_GAME = "reckoning"

# This month's game (C2, Q603; `plan-launch-set.md` §5.2): one game a month from the launch
# set, the same for everybody, by the calendar month in UTC. `FIRST_MONTH_GAME` in
# `FIRST_MONTH`, then §3.2's order from *quarry* onward (the invented games first), wrapping
# round. Each must be in `LAUNCH_SET`, and each of the nine takes a month in turn. A game
# joining the set takes its month by a deliberate edit here, which moves the months after
# the edit and none before it only if it is appended; inserting re-aims the cycle.
MONTHLY: tuple[str, ...] = (
    FIRST_MONTH_GAME,
    "quarry",
    "wolves",
    "hex glinski hexagon",
    "dobutsu",
    "los alamos",
    "cylinder chess",
    "three-check",
    "crazyhouse",
)

# The calendar month that shows `MONTHLY[0]`: October 2026, the month the launch set and
# its first game were answered (2026-10-03).
FIRST_MONTH = (2026, 10)

_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)

# The month's line (C2's test 7: it *says how it is chosen*), and why there is one at all:
# one game for everybody gathers people onto it, which a game between people by
# correspondence needs. No reward for playing it (test 6), and the month is the calendar's,
# never time left (test 2).
MONTH_WHY = (
    "One game a month for everybody, so there are people to play it with: challenge "
    "someone to a game by correspondence, a move whenever you have a minute."
)
MONTH_HOW = "It changes on the first of each month, taking the games to start with in turn."


def month_name(day: date) -> str:
    """The game of the month that holds `day`: one a month through `MONTHLY`."""
    months = (day.year - FIRST_MONTH[0]) * 12 + (day.month - FIRST_MONTH[1])
    return MONTHLY[months % len(MONTHLY)]


def month_game(games: Iterable[Any], day: date) -> dict[str, Any] | None:
    """The front page's *this month's game* on `day`: its name, hash, sentence, the month
    in words, and the lines saying why and how it is chosen.

    `games` is what the listing offers. The month's game when it is offered, else the next
    in `MONTHLY` that is, as `first_step` does; None only when none of the nine is offered.
    """
    listed = {game.hash for game in games}
    start = MONTHLY.index(month_name(day))
    for step in range(len(MONTHLY)):
        entry = by_name(MONTHLY[(start + step) % len(MONTHLY)])
        if entry.hash not in listed:
            continue
        return {
            "name": entry.name,
            "hash": entry.hash,
            "sentence": entry.sentence,
            "month": f"{_MONTH_NAMES[day.month - 1]} {day.year}",
            "why": MONTH_WHY,
            "how": MONTH_HOW,
        }
    return None


def by_name(name: str) -> Launch:
    """The launch-set entry called `name` (reserves included), or `KeyError`."""
    for entry in (*LAUNCH_SET, *RESERVES):
        if entry.name == name:
            return entry
    raise KeyError(name)


def today() -> date:
    """The calendar day the rotation reads: UTC, so it is one day for everybody."""
    return datetime.now(timezone.utc).date()


def featured_name(day: date) -> str:
    """The game the front page features on `day`: one a day through `FEATURED`."""
    return FEATURED[(day - ROTATION_START).days % len(FEATURED)]


class _Listed(Protocol):
    hash: str
    name: str


class LaunchSetMissing(LookupError):
    """A pinned game the library no longer holds, by name and hash."""


def resolve(entries: Iterable[Launch], games: Iterable[_Listed]) -> list[Any]:
    """The library's game for each entry, in order, matched by hash and name.

    Raises `LaunchSetMissing` naming every entry whose hash is not in `games` or whose game
    is called something else now: the launch set never silently drops or swaps a game.
    """
    by_hash = {game.hash: game for game in games}
    found: list[Any] = []
    missing: list[str] = []
    for entry in entries:
        game = by_hash.get(entry.hash)
        if game is None:
            missing.append(f"{entry.name} ({entry.hash}) is not in the library")
        elif game.name != entry.name:
            missing.append(f"{entry.name} ({entry.hash}) is called {game.name!r} now")
        else:
            found.append(game)
    if missing:
        raise LaunchSetMissing("; ".join(missing))
    return found


# --------------------------------------------------------------- the first step's opponents

# Never offered as the first step's opponent: a person, the player that moves at random
# (K3: *the gentlest opponent above random*), and *hardest*, which the site caps.
_NOT_FIRST = ("human", "random", "minimax:nodes=200000")


def _offered(roster: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [entry for entry in roster if str(entry.get("agent")) not in _NOT_FIRST]


def opponents(
    roster: Iterable[Mapping[str, Any]], ratings: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The first step's gentlest opponent and the next one up (§5.1, with K3's rule).

    `roster` is the ladder the page offers on every board (`opponents.ladder()` as dicts),
    weakest first; `ratings` the game's current rating table (`Game.ratings`), or None.
    With a table, the order is the table's: on some games *greedy* is stronger than
    *easy*, which is why it is read rather than assumed. Without one, the roster's own
    order, which is the order the picker already shows. `basis` says which, so the page
    claims no more than it knows.
    """
    offered = _offered(roster)
    if not offered:
        return None
    rated: dict[str, float] = {}
    for row in (ratings or {}).get("standings") or []:
        if isinstance(row, Mapping) and row.get("rating") is not None:
            rated[str(row.get("name"))] = float(row["rating"])
    in_table = [entry for entry in offered if str(entry["agent"]) in rated]
    if len(in_table) >= 2:
        ordered = sorted(in_table, key=lambda entry: rated[str(entry["agent"])])
        basis = "ratings"
    else:
        ordered = offered
        basis = "ladder"

    def one(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {"agent": entry["agent"], "name": entry.get("name") or entry.get("label")}

    return {
        "gentle": one(ordered[0]),
        "harder": one(ordered[1]) if len(ordered) > 1 else None,
        "basis": basis,
    }


def first_step(
    games: Iterable[Any],
    roster: Iterable[Mapping[str, Any]],
    day: date,
    preview: Any = None,
) -> dict[str, Any] | None:
    """The front page's first step on `day`: the featured game, its sentence, its board and
    its two opponents.

    `games` is what the listing offers (a game the site has withdrawn is not in it). The
    day's game when it is offered, else the next in the rotation that is, so a withdrawn
    game never leaves the front page empty; None only when none of the five is offered.
    `preview(game)` draws the opening position (`designing.board_payload`), or is None.
    """
    listed = {game.hash: game for game in games}
    roster = list(roster)
    start = FEATURED.index(featured_name(day))
    for step in range(len(FEATURED)):
        entry = by_name(FEATURED[(start + step) % len(FEATURED)])
        game = listed.get(entry.hash)
        if game is None:
            continue
        return {
            "name": entry.name,
            "hash": entry.hash,
            "sentence": entry.sentence,
            "day": day.isoformat(),
            "opponents": opponents(roster, getattr(game, "ratings", None)),
            # The game's hash on the drawing too, so its wood is the library card's.
            "preview": None if preview is None else {**preview(game), "hash": entry.hash},
        }
    return None


# --------------------------------------------------------------- the landing page's picks

# Q788 (the user, 2026-10-09: *"I'd like the new games highlighted, as well as
# intransitive"*). The landing page shows the library's newest games and *Intransitive*,
# and `/game` repeats neither.
#
# Q826 (the user, 2026-10-10: *"Make the highlighting of recently added games update
# automatically (currently it's out of date)"*). The shelf was a hand-kept list here
# (`NEW_GAMES`, four schema-17 games of 9 October) that nobody edited when fourteen more
# games joined the next two days. It is derived now, from the day each game joined the
# library, which its file records as `provenance.added` (`library.added`;
# `.claude/scripts/backfill_added.py` wrote every existing file's day from git, and
# `corpus.promote` and `archive` stamp a new one). **A game is new for `NEW_WINDOW`
# after the day it joined**; the shelf shows the newest `NEW_SHOWN` of them, newest day
# first and by name within a day. Two weeks because the library grows in bursts (fourteen
# games in two days, then nothing for a fortnight): a week would empty the shelf between
# bursts, and a month would keep a game *new* after a visitor has seen it on every visit.
# **When nothing joined in the window** the shelf shows the `NEW_FALLBACK` most recently
# added, each with its day, under the heading *Recently added* rather than *New*
# (`landing.js` swaps the words when no card is `fresh`), so the section never goes empty
# and never calls a month-old game new. A member's game is not on this shelf: it has its
# own, *Made by members*, newest published first (`site/showcase.made_here`).

# How long a game counts as new after the day it joined the library.
NEW_WINDOW_DAYS = 14

# How many of the new games the landing page's shelf shows, newest first.
NEW_SHOWN = 8

# How many it shows, marked as not new, when nothing joined inside the window.
NEW_FALLBACK = 4

# The landing page's spotlight: *Intransitive*, the reserve the user named, with why it is
# worth a look in a visitor's words.
SPOTLIGHT = Launch(
    "intransitive", "29d5213b02b2",
    "Rock, paper, scissors made into a strategy game on a nine by nine board: each piece "
    "takes only the kind it beats, and the first to bring a piece to the far corner wins.")
SPOTLIGHT_WHY = (
    "No piece is the strongest, because every kind beats one other and loses to the third, "
    "so it plays like no chess variant does. It was designed by webgoatguy, who showed how "
    "he made it on YouTube, and it is the game this project has studied most.")


def _day_words(day: date) -> str:
    return f"{day.day} {_MONTH_NAMES[day.month - 1]} {day.year}"


def first_sentence(text: str) -> str:
    """The first sentence of a game's description: what the shelf prints under its name."""
    text = " ".join(str(text or "").split())
    for at, mark in enumerate(text):
        if mark in ".!?" and (at + 1 == len(text) or text[at + 1] == " "):
            return text[:at + 1]
    return text


def new_cards(games: Iterable[Any], day: date | None = None) -> list[dict[str, Any]]:
    """The library's newest games among `games`, newest first (Q826).

    The games that joined within `NEW_WINDOW_DAYS` of `day` (today, UTC), at most
    `NEW_SHOWN`, each `fresh`; when none did, the `NEW_FALLBACK` most recently added, none
    `fresh`. Each card: name, hash, the first sentence of its description, the day it
    joined in words and as `YYYY-MM-DD`. A game with no recorded day is never on it.
    """
    # Here rather than at the top: `web/__init__` loads the server, which imports this.
    from chess_variant_sandbox.web import library

    day = today() if day is None else day
    dated = []
    for game in games:
        joined = library.added(game) if hasattr(game, "spec") else None
        if joined is not None and date.fromisoformat(joined) <= day:
            dated.append((date.fromisoformat(joined), game))
    dated.sort(key=lambda pair: (-pair[0].toordinal(), str(pair[1].name).lower()))
    fresh = [pair for pair in dated if (day - pair[0]).days < NEW_WINDOW_DAYS]
    shown = fresh[:NEW_SHOWN] if fresh else dated[:NEW_FALLBACK]
    return [
        {"name": str(game.name), "hash": str(game.hash),
         "sentence": first_sentence(getattr(game, "description", "")),
         "added": _day_words(joined), "day": joined.isoformat(), "fresh": bool(fresh)}
        for joined, game in shown
    ]


def spotlight_card(games: Iterable[Any]) -> dict[str, str] | None:
    """The landing page's spotlight when the listing offers it, else None."""
    if SPOTLIGHT.hash not in {game.hash for game in games}:
        return None
    return {"name": SPOTLIGHT.name, "hash": SPOTLIGHT.hash,
            "sentence": SPOTLIGHT.sentence, "why": SPOTLIGHT_WHY}


def launch_cards(games: Iterable[Any]) -> list[dict[str, str]]:
    """The launch set as the listing offers it, in order: name, hash and sentence of each
    of the nine that is offered (for the front door, L1)."""
    listed = {game.hash for game in games}
    return [
        {"name": entry.name, "hash": entry.hash, "sentence": entry.sentence}
        for entry in LAUNCH_SET
        if entry.hash in listed
    ]
