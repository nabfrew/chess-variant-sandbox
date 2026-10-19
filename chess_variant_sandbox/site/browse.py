"""The games page's own columns, and the three orderings §5 offers over them.

Task 1.5 of `docs/plan-website.md`. The local application's `/api/library` already
answers *what games are there* off the files; this is what the site knows about them that
a checkout does not -- what people scored them out of five, how many games people have
played on them, and whether anybody has beaten the top computer player yet.

The user, 2026-09-10: *"Include metrics for games as sorting options. Nr human games
played. 'Unbeaten' (filterable tag given to games where no human has beaten the
leaderboard.) rating"*. So three sort keys and one filter, offered switchably and never
merged: stars are an opinion, human games are what people did, and the index is the
measurement. A single number over all three would answer no question anybody asked.

Box B2 of `docs/archive/plan-meta-game.md` adds two more filters over the same columns.
*Untried* reads `unbeaten.UNTRIED` for the signed-in viewer, one state among the four a
person's own record already has; a visitor has no record to read, so the filter keeps
nothing for one rather than refusing the listing outright, the same *absent is not zero*
the unvoted and unplayed games already live by. *Unplayed* is `humanGames == 0`, the same
count the human-games sort already carries -- a game nobody has finished a leaderboard
match on, which is what M5 of `docs/archive/plan-meta-game.md` §2.2 calls *the first human game
here*.

**The index sorts on the card's own number.** `canonical.json` is the single source for a
card's index and its ladder (`.claude/rules/viz-and-web.md`), and a card that shows one
number and sorts by another is exactly what that rule exists to prevent. The store's
`measurements.headline` is a copy of the same file, loaded by task 1.6, and it is what a
reader with no `runs/` in reach will use -- a user game in phase 2, say. On the box the
two are the same number because the loader read the directory the page reads.

**A stale measurement sorts by its provisional number** (EV2, `docs/archive/plan-evolving-score.md`
§5.2, decided A): the card carries the number with `status` *provisional* and draws the
mark, and nothing here compares it with a current one. A game with no number at all sorts
last, in both directions. An unvoted game and an unplayed game sort last the same way:
absent is not zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site import rating, unbeaten, votes

STARS = "stars"
HUMAN_GAMES = "human_games"
INDEX = "index"
NEWEST = "newest"
"""Newest on trial first, by `games.published` (the *On trial* tab's own order, Q503b)."""
ADDED = "added"
"""Newest first by the day a game joined: a library file's `provenance.added`, a member's
game's publish day (Q826, the games page's *date added*)."""
SORTS = (STARS, HUMAN_GAMES, INDEX, NEWEST, ADDED)
"""What `/api/library?sort=` takes. Anything else, including nothing, leaves the listing
in the order the library gave it -- committed first, then archived, each by name."""

UNBEATEN = "unbeaten"
UNTRIED = unbeaten.UNTRIED
UNPLAYED = "unplayed"
PROVEN = "proven"
"""Games whose page offers at least one proven position (Q972): the card's
`provenPositions`, which the server counts off the same record the page's section reads
(`proven.offered`)."""
FILTERS = (UNBEATEN, UNTRIED, UNPLAYED, PROVEN)


class BrowseError(ValueError):
    """A listing was asked for in terms it does not have.

    A `ValueError` like `matches.MatchError`, so the server answers 400 with a sentence
    naming what it does take.
    """


@dataclass(frozen=True)
class Columns:
    """What the site adds to one card."""

    game: str
    stars: votes.Stars
    human_games: int
    tag: unbeaten.Tag
    untried: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        """The card's new keys, spelled the way the page reads them.

        `unbeaten` is null on an unrated game and the page says *unrated* rather than
        drawing a tag; `rated` is what the play controls read, since §4 offers an unrated
        game sandbox only. `untried` is null with no viewer signed in, the same absence
        for the same reason.
        """
        first = self.tag.first
        return {
            "stars": self.stars.mean,
            "votes": self.stars.count,
            "humanGames": self.human_games,
            "rated": self.tag.rated,
            "unbeaten": self.tag.unbeaten,
            "untried": self.untried,
            "top": list(self.tag.top),
            "topRating": self.tag.top_rating,
            "clears": self.tag.clears,
            "firstClear": None if first is None else {
                "handle": first.handle,
                "when": None if first.when is None else first.when.isoformat(),
                "seat": first.seat,
                "agent": first.agent,
            },
        }


def columns(
    connection: Any, games: Iterable[str] = (), viewer: int | None = None
) -> dict[str, Columns]:
    """The site's columns for every game in the listing, in four queries.

    One pass over the leaderboard rows feeds the count, the tag and, where a viewer is
    given, that person's own *untried* mark -- three readings of the same rows, since
    asking three times would be asking the same question of the same index three times.
    """
    wanted = tuple(games)
    rows = match_rows.leaderboard_matches(connection)
    tables = rating.load_many(connection, wanted)
    if not wanted:
        wanted = tuple(sorted(set(tables) | {row.game for row in rows}))
    tags = unbeaten.tags_from(wanted, rows, tables, unbeaten.handles(connection))
    starred = votes.by_game(connection, wanted)
    played = _human_games(rows)
    untried = _untried(rows, wanted, viewer)
    return {
        game: Columns(
            game=game,
            stars=starred.get(game, votes.NONE),
            human_games=played.get(game, 0),
            tag=tags[game],
            untried=untried.get(game),
        )
        for game in wanted
    }


def names(connection: Any, games: Sequence[str]) -> dict[str, str]:
    """What each of these games is called.

    The listing carries a name on every card already. This is for the pages that hold
    hashes and nothing else -- the profile is the first -- so that a person reads
    *Courier chess* rather than twelve hexadecimal digits.
    """
    if not games:
        return {}
    with connection.cursor() as cursor:
        cursor.execute("select hash, name from games where hash = any(%s)", (list(games),))
        return {str(digest): str(name) for digest, name in cursor.fetchall()}


def _human_games(rows: Sequence[match_rows.Row]) -> dict[str, int]:
    """Finished leaderboard games with a person in them, per game.

    Sandbox counts for nothing (§4) and the rows loaded from the local application count
    for nothing either -- §3 keeps them for the anchor's offline reading and says they are
    counted by no leaderboard. So a game the maintainer played twenty times on their own
    machine still reads nought here, which is the true answer to *how much has this been
    played on the site*.
    """
    found: dict[str, int] = {}
    for row in rows:
        if row.finished is None or row.source is not None:
            continue
        if row.is_computer(0) and row.is_computer(1):
            continue
        found[row.game] = found.get(row.game, 0) + 1
    return found


def _untried(
    rows: Sequence[match_rows.Row], games: Iterable[str], viewer: int | None
) -> dict[str, bool]:
    """Which of these games this viewer has not tried, `unbeaten.UNTRIED`'s own reading
    of the same rows -- so *untried* on a card means exactly what it means on a profile.

    Empty with no viewer, since nobody is signed in to ask about, and an empty map is
    what leaves the `Columns` above null: absent, not false.
    """
    if viewer is None:
        return {}
    return {
        game: unbeaten.record_from(game, viewer, rows).state == unbeaten.UNTRIED
        for game in games
    }


def decorate(
    summaries: Iterable[Mapping[str, Any]], found: Mapping[str, Columns]
) -> list[dict[str, Any]]:
    """The library's cards with the site's columns on them.

    A game the store has never heard of keeps its card and gains nothing, which is what a
    checkout with a fresh database looks like before `chess-variant-sandbox site load` has run.
    """
    cards = []
    for summary in summaries:
        card = dict(summary)
        extra = found.get(str(card.get("hash", "")))
        if extra is not None:
            card.update(extra.as_dict())
        cards.append(card)
    return cards


def order(
    cards: Sequence[Mapping[str, Any]], sort: str = "", keep: str = ""
) -> list[dict[str, Any]]:
    """The listing, filtered and ordered as the query asked.

    An unrecognised sort or filter is refused by name rather than ignored, because a
    listing silently in the wrong order is a listing that lies about its own ranking.
    """
    if sort and sort not in SORTS:
        raise BrowseError(f"sort is one of {', '.join(SORTS)}, not {sort!r}")
    if keep and keep not in FILTERS:
        raise BrowseError(f"filter is one of {', '.join(FILTERS)}, not {keep!r}")
    found = [dict(card) for card in cards]
    if keep == UNBEATEN:
        found = [card for card in found if card.get("unbeaten") is True]
    elif keep == UNTRIED:
        # Null with no viewer signed in (`_untried`), so this keeps nothing for a
        # visitor rather than refusing the whole listing -- the same *absent is not
        # zero* the unbeaten and unplayed filters already live by.
        found = [card for card in found if card.get("untried") is True]
    elif keep == UNPLAYED:
        found = [card for card in found if card.get("humanGames") == 0]
    elif keep == PROVEN:
        found = [card for card in found if (card.get("provenPositions") or 0) > 0]
    if not sort:
        return found
    return sorted(found, key=lambda card: _key(card, sort))


# ------------------------------------------------------- the same columns, keyed by node
#
# `docs/archive/plan-game-graph.md` §8.3, G12. The map draws nodes, not files, and what
# the site knows about a game is recorded per file -- so the overlay that puts the site's
# numbers on the map has to fold a node's spellings together before anybody sees a dot.
# That fold is here, beside the columns it folds, and not in the page: *the page never
# averages*.
#
# The two numbers are the listing's own first two orderings, read per game and the same
# for every viewer. *Untried* is the one per-viewer field, absent for a visitor, exactly
# as it is absent on a card.

# The star scale, as the vote control offers it. Both ends of the `stars` size key, and
# fixed rather than the library's own lowest and highest: a person voting knows what five
# means, so a ramp that moved when somebody voted would be a second scale over the first.
STARS_LOW = votes.LOWEST
STARS_HIGH = votes.HIGHEST

STARS_SIZE = "stars"
PLAYED_SIZE = "played"

OVERLAY = "people"
"""What `/api/graph` calls this reading in its `overlays`, and the one name the page
matches on before it asks for anything. A checkout has no store, so its list is empty and
its page asks for nothing."""


def people(
    nodes: Mapping[str, Sequence[str]],
    found: Mapping[str, Columns],
    viewer: int | None = None,
) -> dict[str, Any]:
    """The overlay `GET /api/graph/people` answers, keyed by the nodes the map draws.

    `nodes` maps a node's hash to every spelling merged into it, which is the map's own
    twin grouping; `found` is `columns` over those spellings. A node with two spellings
    adds their matches and their votes and takes the mean over all the votes, so one game
    written two ways reads as one game.

    `sizes` and `games[...].size` are the shape of §8.3 in the archive copy, the same one
    `web/graphing.py` builds for the payload's own keys: the server names the key, says
    what it means, prints both ends, and hands each dot its place between them. A key
    nothing can answer -- stars with no vote cast, played with no game finished -- is left
    out of the menu, which is what G11 settled for a `score` key on a library with no ramp.
    """
    merged = {
        digest: _merged(spellings, found, viewer)
        for digest, spellings in nodes.items()
    }
    played = [one["played"] for one in merged.values()]
    busiest = max(played) if played else 0
    voted = any(one["votes"] for one in merged.values())
    games = {
        digest: {**one, "size": _people_size(one, busiest, voted)}
        for digest, one in merged.items()
    }
    return {
        "sizes": _people_sizes(busiest, voted),
        "filters": _people_filters(viewer),
        "games": games,
    }


def _merged(
    spellings: Sequence[str], found: Mapping[str, Columns], viewer: int | None
) -> dict[str, Any]:
    """One node's numbers, over every file that says this game.

    `stars` is null where nobody has voted and `played` is a true nought, which is the
    same *absent is not zero* the cards live by: an unvoted game is not a game everybody
    hated, and an unplayed one has genuinely been played no times.

    `unbeaten` is null until some spelling is rated, and true only when every rated
    spelling is still unbeaten -- beating the game under either spelling beats the game.
    `untried` is null for a visitor and true only when this person has tried neither.
    """
    held = [found[one] for one in spellings if one in found]
    count = sum(one.stars.count for one in held)
    total = sum(
        one.stars.mean * one.stars.count for one in held if one.stars.mean is not None
    )
    rated = [one.tag for one in held if one.tag.rated]
    tried = [one.untried for one in held if one.untried is not None]
    playings = sum(one.human_games for one in held)
    return {
        "stars": None if count == 0 else round(total / count, 2),
        "votes": count,
        "played": playings,
        "unbeaten": all(tag.unbeaten for tag in rated) if rated else None,
        "untried": all(tried) if viewer is not None and tried else None,
        "unplayed": playings == 0,
    }


def _people_size(
    one: Mapping[str, Any], busiest: int, voted: bool
) -> dict[str, float | None]:
    """One dot's place between each key's two ends, 0 to 1, or null for *no value here*.

    Null is the unvoted game under `stars`, which draws at the smallest size and is not
    the same statement as a game everybody gave one star. Under `played` a nought is a
    number, so a game nobody has finished is 0 rather than null.
    """
    stars = one["stars"]
    return {
        STARS_SIZE: (
            None
            if not voted or stars is None
            else round((float(stars) - STARS_LOW) / (STARS_HIGH - STARS_LOW), 4)
        ),
        PLAYED_SIZE: None if busiest <= 0 else round(one["played"] / busiest, 4),
    }


def _people_sizes(busiest: int, voted: bool) -> list[dict[str, Any]]:
    """The menu the two numbers add, in menu order, each with its rule and its two ends.

    An end is printed in the server's own words, because a page that formatted one would
    be a second opinion about what the end is.
    """
    sizes: list[dict[str, Any]] = []
    if voted:
        sizes.append({
            "key": STARS_SIZE,
            "label": "what people scored it",
            "words": (
                "The mean of the one-to-five stars people gave the game. A game nobody"
                " has voted on has no value under this key and draws the smallest dot,"
                " and a dot's panel prints the votes beside the stars."
            ),
            "low": STARS_LOW,
            "high": STARS_HIGH,
            "lowWords": "one star",
            "highWords": "five stars",
        })
    if busiest > 0:
        sizes.append({
            "key": PLAYED_SIZE,
            "label": "how much it is played",
            "words": (
                "Finished leaderboard games with a person in a seat. Sandbox games and"
                " the rows loaded from the local application count for nothing."
            ),
            "low": 0,
            "high": busiest,
            "lowWords": "never played",
            "highWords": f"{busiest} games" if busiest != 1 else "1 game",
        })
    return sizes


def _people_filters(viewer: int | None) -> list[dict[str, str]]:
    """The listing's three, in the listing's own words.

    *Untried* reads this person's own record, so a visitor has none to read and is not
    offered the filter rather than being offered one that would light nothing.
    """
    filters = [
        {"filter": UNBEATEN, "key": "unbeaten", "label": "nobody has beaten its strongest computer player yet"},
        {"filter": UNTRIED, "key": "untried", "label": "you haven't tried this yet"},
        {"filter": UNPLAYED, "key": "unplayed", "label": "not played here yet"},
    ]
    return [one for one in filters if viewer is not None or one["filter"] != UNTRIED]


def _key(card: Mapping[str, Any], sort: str) -> tuple[int, float, str]:
    """Best first, absent last, name as the tie-break.

    The leading 0 or 1 is what keeps an unmeasured, unvoted or unplayed game at the
    bottom rather than at the top of a descending sort: *nobody has scored this* is not
    *scored zero*, and a ranking that confused the two would put every new game first.
    """
    field = {
        STARS: "stars", HUMAN_GAMES: "humanGames", INDEX: "index", NEWEST: "onTrialSince",
        ADDED: "added",
    }[sort]
    value = card.get(field)
    name = str(card.get("name", "")).lower()
    if value is None:
        return (1, 0.0, name)
    if sort == NEWEST:
        return (0, -datetime.fromisoformat(str(value)).timestamp(), name)
    if sort == ADDED:
        return (0, -float(date.fromisoformat(str(value)[:10]).toordinal()), name)
    return (0, -float(value), name)
