"""The meta-game's milestone reader, box A1 of `docs/archive/plan-meta-game.md`.

The milestones it reads are defined in `docs/plan-meta-game.md` §2.2 to §2.4.

Pure readings in the idiom of `unbeaten.record_from`. Rows and tables go in, values come
out, and nothing here opens a connection. The caller hands over every leaderboard row on
the site rather than one person's, because *the first human game here* and the
*Unbeaten* tag read other people's rows too.

## What is read

A row counts when it is a finished leaderboard match with a result, both seats hold an
account, and the local application did not load it. Sandbox rows and `source =
'local-app'` rows are dropped at the door, so no reading below sees them. Every result is
read off a row and none is worked out here (`CLAUDE.md` rule 3).

## The milestones

Each is a dated line, dated by the finish of the match that completed it.

- M3, *both seats*, is a win over a computer player from each seat, one line per game.
- M4, *the whole ladder*, is a win over every computer player on the game's table, one
  line per game.
- M5, *the first human game here*, is the earliest counted match on a game with a person
  in a seat. Everyone seated in that match holds it.
- M6, *the tour*, is two lines. One is a try on every rated game. The other is a try on
  every kind the rated games cover, where a kind is one bin of one `loop.grid.AXES` axis.
- M7 is two lines across the library, a first game against another person and a win over
  another person from each seat. Both read any row that seats two people, so they light
  as soon as phase 3 writes one.

S2 is a count rather than a line. It counts the games where this person is the
highest-rated established person on the table.

Two readings are columns on the profile's table rather than lines under it. M4 counted
rather than dated is *n of k rungs*, and T3 is *distance to the top*, the gap between
this person's rating on a game and its top computer player's.

A try for the tour means what a try means on the profile, a counted match against a
computer player (`unbeaten.UNTRIED`). So a game the tour counts as tried never shows as
untried under *still open*.

M4 and M6 read the table and the library as they stand now. A rung or a game added later
reopens them for everyone, present or away, and the new rung or game shows under *still
open*. Every other line is fixed once its match is recorded.

## The designer's four (D1 to D4)

Box Q252 of `docs/plan-meta-game.md`. They read designed games' rows and the signed-in
seats on them rather than leaderboard matches, so they come in as `Design` and `Try`
(`site/designs.py` loads both) and `read_from` joins them to the lines above. Each is one
line however many games a person makes (the seven tests' sixth), the first of its kind:

- D1, *first draft kept*, the first designed game this person's save kept.
- D2, *passes the screen*, the first of theirs the playability check has passed. The
  store keeps the verdict and not the day of it, so it is dated by the save that kept the
  draft; a later save only ever sets the verdict, so the line never goes.
- D3, *played by somebody else*, the first match another signed-in person started on one
  of their games, practice or rated -- the user's decision B of 2026-09-14.
- D4, *forked*, the first game another person kept that was built from one of theirs,
  with who (`Milestone.by`).

`marks_from` is the same reading for one game's page: whether it passes the screen, how
many other people have tried it (a count and never who), and who forked it.

## The overall page

S3, box A3 of `docs/archive/plan-meta-game.md`. Two readings over the same counted rows,
for the page at `/players`.

`standings_from` is everybody with a finished game, each beside the three counts. The
counts are `unbeaten.counts_from` over `unbeaten.record_from`, which is what a profile
shows, so one person reads the same on both pages and there is no second way to count a
game tried.

`unclaimed_from` is what nobody has taken yet -- the rated games nobody has played a
counted game on (M5) and the rated games still *Unbeaten* (M2). Both are over the games
the caller says can still be started, since a game nobody may seat is not open to
anybody.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from chess_variant_sandbox.loop import grid
from chess_variant_sandbox.site import matches as match_rows
from chess_variant_sandbox.site import rating, unbeaten

BOTH_SEATS = "both-seats"
WHOLE_LADDER = "whole-ladder"
FIRST_HERE = "first-here"
EVERY_GAME = "every-game"
EVERY_KIND = "every-kind"
FIRST_PERSON = "first-person"
PERSON_BOTH_SEATS = "person-both-seats"
FIRST_DRAFT = "first-draft"
PASSES_SCREEN = "passes-screen"
PLAYED_BY_OTHERS = "played-by-others"
FORKED = "forked"
DESIGNER_KINDS = (FIRST_DRAFT, PASSES_SCREEN, PLAYED_BY_OTHERS, FORKED)
"""D1 to D4, the designer's four lines (`docs/plan-meta-game.md`, *Open boxes*)."""
KINDS = (
    BOTH_SEATS,
    WHOLE_LADDER,
    FIRST_HERE,
    EVERY_GAME,
    EVERY_KIND,
    FIRST_PERSON,
    PERSON_BOTH_SEATS,
    *DESIGNER_KINDS,
)

# Each rule in a visitor's words (`docs/plan-understanding.md` §3.3, S13): *rated game*
# for a leaderboard match, *side* for a seat, *every computer player this game has now*
# for the table. The words changed and the readings did not; `tests/test_milestones.py`
# holds every kind to one rule.
RULES: dict[str, str] = {
    BOTH_SEATS: (
        "A win over a computer player from each side of this game: once moving first, "
        "once moving second."
    ),
    WHOLE_LADDER: "A win over every computer player this game has now.",
    FIRST_HERE: "The first rated game anybody finished on this game.",
    EVERY_GAME: "A finished rated game on every game that has rated play now.",
    EVERY_KIND: (
        "A finished rated game of every kind the rated games cover now: one game of each "
        "kind, for every question a kind answers, such as how the game is won or how big "
        "the board is."
    ),
    FIRST_PERSON: "A first finished rated game against another person.",
    PERSON_BOTH_SEATS: (
        "A win over another person from each side, moving first and moving second, on "
        "any games."
    ),
    FIRST_DRAFT: "The first game you designed and kept on the site, dated by the save.",
    PASSES_SCREEN: (
        "The first game you designed that passes the playability check, dated by the save "
        "that kept it."
    ),
    PLAYED_BY_OTHERS: (
        "The first time another signed-in person started a game, practice or rated, on "
        "a game you designed."
    ),
    FORKED: (
        "The first time another person kept a game of their own built from one you "
        "designed."
    ),
}
"""What each milestone reads, one sentence each, so the page can name its rule (§3 test 7)."""

Tables = Mapping[str, Mapping[rating.PlayerKey, rating.Player]]

Cells = Mapping[str, Sequence[int]]
"""Each rated game's cell on the behaviour grid. An **empty** cell is a game whose rules
were deleted (`0010_deleted.sql`): it has no kind, because there are no rules to read one
off. Its rating table, its matches and its votes stay, so it is still a rated game and
still a row on a person's profile -- it just covers no bin of any axis."""


@dataclass(frozen=True)
class Milestone:
    """One dated line.

    `match` is the match that completed it and `seat` is this person's seat in it. `game`
    is the game the milestone is on, or for a library-wide one the completing match's game.

    A designer's line (`DESIGNER_KINDS`) has no seat, so `seat` is -1, and `match` is
    whatever dated it: the kept game's hash for D1 and D2, the other person's match for
    D3, and the fork's hash for D4.
    """

    kind: str
    when: datetime
    game: str
    seat: int
    match: str
    # D4 only: the account that forked the game, so the line can say *by ‹handle›*. D3
    # names nobody (the user's decision B of 2026-09-14), and every play line is about
    # the holder's own match. Defaulted so the play lines are built as they always were.
    by: int | None = None


@dataclass(frozen=True)
class OpenKind:
    """One bin of one grid axis, as *still open for you* shows it.

    The three lists follow the plan's wording, so they overlap. An untried game is also
    not yet won, and an *Unbeaten* game appears whatever this person's state on it.
    """

    bin: str
    tried: bool
    untried: tuple[str, ...]
    unwon: tuple[str, ...]
    unbeaten: tuple[str, ...]


@dataclass(frozen=True)
class OpenAxis:
    """One grid axis and every bin of it that holds a rated game, in `grid.AXES` order."""

    axis: str
    kinds: tuple[OpenKind, ...]


@dataclass(frozen=True)
class Rungs:
    """M4 counted, as the profile's table shows it. One rated game's *n of k rungs*."""

    beaten: int
    of: int


@dataclass(frozen=True)
class Distance:
    """T3 on one rated game.

    `points` is the top computer player's rating less this person's, so a positive number
    is how far below the top they sit and a negative one how far above. `rating` is this
    person's own, which the profile shows first (F8 of `docs/reference/first-visit-study.md`).
    """

    points: float
    established: bool
    rating: float = 0.0


@dataclass(frozen=True)
class Reading:
    """Everything A1 gives a profile. Milestones are newest first.

    `rungs` and `distance` are the profile table's two columns, keyed by game. A game
    absent from `distance` is one this person has no rating on, which is not a gap of
    nought (§2.1).
    """

    milestones: tuple[Milestone, ...]
    still_open: tuple[OpenAxis, ...]
    highest: int
    rungs: dict[str, Rungs]
    distance: dict[str, Distance]


@dataclass(frozen=True)
class Standing:
    """One person on the overall page, S3.

    `counts` is `unbeaten.Counts`, the same four numbers a profile's three counts and its
    *of N* come from. `handle` is whatever `unbeaten.handles` gave, so somebody who has
    left reads as `identity.FORMER` and never by the name they had.
    """

    user: int
    handle: str
    counts: unbeaten.Counts


@dataclass(frozen=True)
class Unclaimed:
    """What the overall page opens with: the firsts nobody has taken.

    `unplayed` is the games with no counted game on them at all, where M5 is there for
    the taking. `unbeaten` is the games whose top computer player nobody has beaten,
    where M2 is. The two overlap, because a game nobody has played is also one nobody has
    cleared. `offered` is how many games either list was read over.
    """

    unplayed: tuple[str, ...]
    unbeaten: tuple[str, ...]
    offered: int


@dataclass(frozen=True)
class Design:
    """One designed game's row, as the designer milestones read it.

    `created` is when the save first kept it and `screen_passed` whether the playability
    check has passed it at a save (`0016_drafts.sql`; a later save can set it, never clear
    it). `parent` is the game it was forked from, when the store held that game.
    """

    hash: str
    owner: int | None
    parent: str | None
    created: datetime
    screen_passed: bool


@dataclass(frozen=True)
class Try:
    """One signed-in person seated in one match on a game, practice or rated.

    Visitors with no account and computer players are no `Try`: the plain count D3 shows
    is of people, and only a signed-in seat is somebody.
    """

    game: str
    user: int
    started: datetime
    match: str


@dataclass(frozen=True)
class Marks:
    """What the game page says about a designed game (D2 to D4).

    `tried` is how many signed-in people other than its designer have started a game of
    it, a plain count and never who (the user's decision B of 2026-09-14). `forked_by` is
    the accounts, other than the designer, that kept a game built from it, in the order
    they first did.
    """

    passes_screen: bool
    tried: int
    forked_by: tuple[int, ...]


@dataclass(frozen=True)
class _Played:
    """A counted match from one person's side of the board."""

    when: datetime
    match: str
    game: str
    seat: int
    opponent: rating.PlayerKey
    score: float


# ----------------------------------------------------------------------------- reading


def read_from(
    user: int,
    rows: Iterable[match_rows.Row],
    tables: Tables,
    cells: Cells,
    designs: Iterable[Design] = (),
    tries: Iterable[Try] = (),
) -> Reading:
    """This person's milestones, what is still open for them, and S2's count.

    `rows` is every leaderboard row on the site, `tables` every game's rating table, and
    `cells` each rated game's `grid.descriptor`. `designs` and `tries` are what the
    designer lines read (`designer_from`): this person's designed games and the games
    forked from them, and the signed-in seats on those games. Left out, a reading has no
    designer lines, which is also what somebody who has designed nothing gets.
    """
    counted = _counted(rows)
    played = milestones_from(user, counted, tables, cells)
    made = designer_from(user, designs, tries)
    return Reading(
        milestones=tuple(sorted((*played, *made), key=_newest_first)),
        still_open=still_open_from(user, counted, tables, cells),
        highest=highest_count(user, tables),
        rungs=rungs_from(user, counted, tables),
        distance=distance_from(user, tables),
    )


def milestones_from(
    user: int, rows: Iterable[match_rows.Row], tables: Tables, cells: Cells
) -> tuple[Milestone, ...]:
    """Every milestone this person holds, newest first."""
    counted = _counted(rows)
    mine = _mine(user, counted)
    tries = [played for played in mine if played.opponent.is_computer]
    people = [played for played in mine if not played.opponent.is_computer]
    rated = _rated(tables)

    found: list[Milestone] = []
    for game, played in _by_game(tries).items():
        wins = [each for each in played if each.score == 1.0]
        found += _line(BOTH_SEATS, _each_seat(wins))
        found += _line(WHOLE_LADDER, _whole_ladder(wins, tables.get(game, {})))
    firsts = _first_human_games(counted)
    found += [_milestone(FIRST_HERE, played) for played in mine if played.match in firsts]
    found += _line(EVERY_GAME, _every_game(tries, rated))
    found += _line(EVERY_KIND, _every_kind(tries, rated, cells))
    found += _line(FIRST_PERSON, people[0] if people else None)
    found += _line(PERSON_BOTH_SEATS, _each_seat([p for p in people if p.score == 1.0]))
    return tuple(sorted(found, key=_newest_first))


def still_open_from(
    user: int, rows: Iterable[match_rows.Row], tables: Tables, cells: Cells
) -> tuple[OpenAxis, ...]:
    """The rated games not yet tried, not yet won and still *Unbeaten*, grouped by axis.

    States come from `unbeaten.record_from` and the tag from `unbeaten.tags_from`, so a
    game reads the same here as on the profile's table and the game's card.
    """
    counted = _counted(rows)
    rated = _rated(tables)
    records = {game: unbeaten.record_from(game, user, counted) for game in rated}
    tags = unbeaten.tags_from(rated, counted, tables)
    kinds = _kinds(rated, cells)

    found: list[OpenAxis] = []
    for index, axis in enumerate(grid.AXES):
        groups: list[OpenKind] = []
        for value, name in enumerate(axis.bins):
            games = kinds.get((index, value), [])
            if not games:
                continue
            groups.append(
                OpenKind(
                    bin=name,
                    tried=any(records[game].state != unbeaten.UNTRIED for game in games),
                    untried=tuple(g for g in games if records[g].state == unbeaten.UNTRIED),
                    unwon=tuple(
                        g for g in games
                        if records[g].state in (unbeaten.UNTRIED, unbeaten.TRIED)
                    ),
                    unbeaten=tuple(g for g in games if tags[g].unbeaten is True),
                )
            )
        found.append(OpenAxis(axis=axis.name, kinds=tuple(groups)))
    return tuple(found)


def rungs_from(user: int, rows: Iterable[match_rows.Row], tables: Tables) -> dict[str, Rungs]:
    """M4 as a count, one entry per rated game.

    The same wins `_whole_ladder` folds, stopped at how many of the table's computer
    players this person has beaten at least once. A win over a player the table no longer
    carries counts for nothing, so the column and the milestone reopen together when a
    rung is added.
    """
    wins = [
        played
        for played in _mine(user, _counted(rows))
        if played.opponent.is_computer and played.score == 1.0
    ]
    by_game = _by_game(wins)
    found: dict[str, Rungs] = {}
    for game in _rated(tables):
        ladder = {row.key.agent for row in rating.computers(tables[game])}
        beaten = {played.opponent.agent for played in by_game.get(game, [])} & ladder
        found[game] = Rungs(beaten=len(beaten), of=len(ladder))
    return found


def distance_from(user: int, tables: Tables) -> dict[str, Distance]:
    """T3, one entry per rated game this person holds a rating on.

    A game with no row for them is absent rather than a gap of nought, so the column is
    blank until they have played there. `established` travels with the gap because a
    provisional rating is one the table itself does not rank (`rating.ESTABLISHED_DEVIATION`).
    """
    me = rating.person(user)
    found: dict[str, Distance] = {}
    for game in _rated(tables):
        table = tables[game]
        mine = table.get(me)
        top = rating.top(table)
        if mine is None or not top:
            continue
        found[game] = Distance(
            points=top[0].rating - mine.rating,
            established=mine.established,
            rating=mine.rating,
        )
    return found


# ------------------------------------------------------------- D1 to D4, the designer


def designer_from(
    user: int, designs: Iterable[Design], tries: Iterable[Try]
) -> tuple[Milestone, ...]:
    """D1 to D4 for this person, newest first: at most one line of each.

    One line each however many games (§ the seven tests, test 6): the first draft kept,
    the first that passes the screen, the first time somebody else played one, and the
    first time somebody else forked one. Each is dated by the row that completed it and
    names the game it was on, so it is fixed once reached and nothing about it lapses.

    A game counts as this person's when its row names them as owner and it is a designed
    game (the caller's filter). Their own plays and their own forks of their own games
    are not somebody else's.
    """
    everything = tuple(designs)
    mine = sorted((d for d in everything if d.owner == user), key=_design_order)
    held = {design.hash for design in mine}
    found: list[Milestone] = []
    if mine:
        found.append(_design_line(FIRST_DRAFT, mine[0]))
    passed = [design for design in mine if design.screen_passed]
    if passed:
        found.append(_design_line(PASSES_SCREEN, passed[0]))
    others = sorted(
        (t for t in tries if t.game in held and t.user != user),
        key=lambda t: (t.started, t.match),
    )
    if others:
        first = others[0]
        found.append(
            Milestone(
                kind=PLAYED_BY_OTHERS, when=first.started, game=first.game, seat=-1,
                match=first.match,
            )
        )
    forks = sorted(
        (
            d for d in everything
            if d.parent in held and d.owner is not None and d.owner != user
        ),
        key=_design_order,
    )
    if forks:
        fork = forks[0]
        found.append(
            Milestone(
                kind=FORKED, when=fork.created, game=str(fork.parent), seat=-1,
                match=fork.hash, by=fork.owner,
            )
        )
    return tuple(sorted(found, key=_newest_first))


def marks_from(game: Design, designs: Iterable[Design], tries: Iterable[Try]) -> Marks:
    """The game page's three marks on one designed game, D2 to D4.

    `designs` may hold any rows; only those forked from `game` are read. `tries` likewise,
    only those on `game`. Its designer is counted in neither, and a game with no owner
    -- one loaded before any account existed -- has no designer to leave out.
    """
    tried = {t.user for t in tries if t.game == game.hash and t.user != game.owner}
    forks = sorted(
        (
            d for d in designs
            if d.parent == game.hash and d.owner is not None and d.owner != game.owner
        ),
        key=_design_order,
    )
    by: list[int] = []
    for fork in forks:
        if fork.owner is not None and fork.owner not in by:
            by.append(fork.owner)
    return Marks(passes_screen=game.screen_passed, tried=len(tried), forked_by=tuple(by))


def _design_order(design: Design) -> tuple[datetime, str]:
    return (design.created, design.hash)


def _design_line(kind: str, design: Design) -> Milestone:
    return Milestone(kind=kind, when=design.created, game=design.hash, seat=-1, match=design.hash)


# ------------------------------------------------------------------- S3, `/players`


LISTED_FROM = 5
"""L3 of `docs/plan-launch.md` §3.3: `/players` lists people, and the header links it, only
once this many have finished a rated game. Below it the page says who is here and what
puts a person on the list, and draws nothing ranked: with two or three names a standing
reads as a scoreboard of a crowd that is not there yet."""


def listed_people(rows: Iterable[match_rows.Row]) -> dict[int, list[match_rows.Row]]:
    """Everybody `/players` would list, each with their counted rows in finish order.

    A person is here because they are seated in a counted row; a computer seat is nobody.
    `standings_from` reads the same set, so the threshold and the list never disagree.
    """
    theirs: dict[int, list[match_rows.Row]] = {}
    for row in _counted(rows):
        for seat in (0, 1):
            if row.is_computer(seat):
                continue
            person = row.player(seat).user
            if person is not None:
                theirs.setdefault(person, []).append(row)
    return theirs


def players_listed(rows: Iterable[match_rows.Row]) -> bool:
    """Whether `/players` lists people yet: `LISTED_FROM` of them or more."""
    return len(listed_people(rows)) >= LISTED_FROM


def standings_from(
    rows: Iterable[match_rows.Row], tables: Tables, handles: Mapping[int, str]
) -> tuple[Standing, ...]:
    """Everybody with a finished game, each beside the three counts, strongest first.

    A person is here because they are seated in a counted row, which is the decision the
    user took on 2026-09-14: the page lists only people who have finished a game. Anybody
    who has not is on no list at all, rather than on it at nought (§3, test 5).

    The order is tops beaten, then games won, then games tried, then the handle. It is a
    standing and not a league table, so no position is drawn and nothing here counts
    matches -- a hundred games on one game move no column past one (§3, test 6).

    Two people who have both left share the handle `identity.FORMER`, so the account's own
    id is the last tie-break and the order is settled whatever the store returns.
    """
    rated = _rated(tables)
    theirs = listed_people(rows)
    found = [
        Standing(
            user=user,
            handle=handles.get(user, ""),
            counts=unbeaten.counts_from(
                unbeaten.record_from(game, user, mine) for game in rated
            ),
        )
        for user, mine in theirs.items()
    ]
    return tuple(sorted(found, key=_standing_order))


def _standing_order(standing: Standing) -> tuple[int, int, int, str, int]:
    counts = standing.counts
    return (-counts.top, -counts.won, -counts.tried, standing.handle.lower(), standing.user)


def unclaimed_from(
    rows: Iterable[match_rows.Row], tables: Tables, games: Sequence[str]
) -> Unclaimed:
    """The firsts nobody has taken yet, over the games that can still be started.

    `games` is the caller's, because whether a game takes a new match is the store's
    answer and not a reading over rows: a killed, retired or deleted game is on nobody's
    list of what to go and claim.

    Both lists are scarce because there is one library, never because a clock is running
    (§3, test 2). Neither shrinks while anybody is away and both grow when the library
    does.
    """
    counted = _counted(rows)
    played = {row.game for row in counted if not (row.is_computer(0) and row.is_computer(1))}
    tags = unbeaten.tags_from(games, counted, tables)
    return Unclaimed(
        unplayed=tuple(game for game in games if game not in played),
        unbeaten=tuple(game for game in games if tags[game].unbeaten is True),
        offered=len(games),
    )


def highest_people(table: Mapping[rating.PlayerKey, rating.Player]) -> tuple[rating.Player, ...]:
    """S2. The established people at the table's highest person rating.

    Everyone tied at that rating holds it. Nobody holds it until established, and a
    computer player never does.
    """
    people = [row for row in table.values() if not row.is_computer and row.established]
    if not people:
        return ()
    best = max(row.rating for row in people)
    return tuple(
        sorted((row for row in people if row.rating == best), key=lambda row: row.key.user or 0)
    )


def highest_count(user: int, tables: Tables) -> int:
    """S2's count on a profile, the games where this person is the highest-rated person."""
    me = rating.person(user)
    return sum(
        1 for table in tables.values() if any(row.key == me for row in highest_people(table))
    )


# ----------------------------------------------------------------------------- the rows


def _counted(rows: Iterable[match_rows.Row]) -> tuple[match_rows.Row, ...]:
    """The rows any milestone reads, in finish order with `id` breaking a tie."""
    kept = [
        (row.finished, row.id, row)
        for row in rows
        if row.mode == match_rows.LEADERBOARD
        and row.source is None
        and row.finished is not None
        and row.result is not None
        and row.seat0 is not None
        and row.seat1 is not None
    ]
    kept.sort(key=lambda found: (found[0], found[1]))
    return tuple(row for _, _, row in kept)


def _mine(user: int, rows: Iterable[match_rows.Row]) -> list[_Played]:
    me = rating.person(user)
    found: list[_Played] = []
    for row in rows:
        seat = row.seat_of(me)
        if seat < 0 or row.finished is None or row.result is None:
            continue
        found.append(
            _Played(
                when=row.finished,
                match=row.id,
                game=row.game,
                seat=seat,
                opponent=row.player(1 - seat),
                score=row.result if seat == 0 else 1.0 - row.result,
            )
        )
    return sorted(found, key=_order)


def _order(played: _Played) -> tuple[datetime, str]:
    return (played.when, played.match)


def _newest_first(line: Milestone) -> tuple[float, str, int]:
    return (-line.when.timestamp(), line.match, KINDS.index(line.kind))


def _by_game(played: Iterable[_Played]) -> dict[str, list[_Played]]:
    found: dict[str, list[_Played]] = {}
    for each in played:
        found.setdefault(each.game, []).append(each)
    return found


def _milestone(kind: str, played: _Played) -> Milestone:
    return Milestone(
        kind=kind, when=played.when, game=played.game, seat=played.seat, match=played.match
    )


def _line(kind: str, played: _Played | None) -> list[Milestone]:
    return [] if played is None else [_milestone(kind, played)]


def _rated(tables: Tables) -> tuple[str, ...]:
    """The games with computer players on their table, which is what makes a game rated."""
    return tuple(sorted(game for game, table in tables.items() if rating.top(table)))


# ----------------------------------------------------------------------- the milestones


def _each_seat(wins: Iterable[_Played]) -> _Played | None:
    """The win that completed a win from each seat, or None."""
    first: dict[int, _Played] = {}
    for win in wins:
        first.setdefault(win.seat, win)
        if len(first) == 2:
            return win
    return None


def _whole_ladder(
    wins: Iterable[_Played], table: Mapping[rating.PlayerKey, rating.Player]
) -> _Played | None:
    """The win that completed a win over every computer player on the table, or None."""
    ladder = {row.key.agent for row in rating.computers(table)}
    beaten: set[str | None] = set()
    for win in wins:
        if win.opponent.agent not in ladder or win.opponent.agent in beaten:
            continue
        beaten.add(win.opponent.agent)
        if beaten == ladder:
            return win
    return None


def _first_human_games(rows: Iterable[match_rows.Row]) -> set[str]:
    """The match id of each game's first counted match with a person in a seat."""
    first: dict[str, str] = {}
    for row in rows:
        if row.is_computer(0) and row.is_computer(1):
            continue
        first.setdefault(row.game, row.id)
    return set(first.values())


def _first_tries(tries: Iterable[_Played]) -> dict[str, _Played]:
    first: dict[str, _Played] = {}
    for played in sorted(tries, key=_order):
        first.setdefault(played.game, played)
    return first


def _every_game(tries: Iterable[_Played], rated: Sequence[str]) -> _Played | None:
    first = _first_tries(tries)
    if not rated or any(game not in first for game in rated):
        return None
    return max((first[game] for game in rated), key=_order)


def _every_kind(tries: Iterable[_Played], rated: Sequence[str], cells: Cells) -> _Played | None:
    """The try that completed a try on every kind the rated games cover, or None."""
    first = _first_tries(tries)
    completing: list[_Played] = []
    for games in _kinds(rated, cells).values():
        reached = [first[game] for game in games if game in first]
        if not reached:
            return None
        completing.append(min(reached, key=_order))
    return max(completing, key=_order) if completing else None


def _kinds(rated: Sequence[str], cells: Cells) -> dict[tuple[int, int], list[str]]:
    """The rated games in each bin of each axis, keyed by axis index and bin.

    A game with no cell covers no bin, so *every kind* is not held open by a game nobody
    can play any more and a deleted game is in nobody's *still open* list.
    """
    found: dict[tuple[int, int], list[str]] = {}
    for game in rated:
        for index, value in enumerate(_cell(cells, game)):
            found.setdefault((index, value), []).append(game)
    return dict(sorted(found.items()))


def _cell(cells: Cells, game: str) -> tuple[int, ...]:
    """A rated game's grid cell, refused by name when it is missing or the wrong length.

    Empty is the one length that is not a refusal, and it means the rules were deleted.
    The distinction is the map's to make: a game absent from it is a reading that lost
    track of a game, which is the bug this guard was put here to catch.
    """
    found = cells.get(game)
    if found is not None and not found:
        return ()
    if found is None or len(found) != len(grid.AXES):
        raise ValueError(f"rated game {game} needs a grid cell of {len(grid.AXES)} axes")
    return tuple(found)
