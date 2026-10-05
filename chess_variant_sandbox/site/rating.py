"""One Elo table per game, with people and computer players on it.

§5 of `docs/plan-website.md` is the design and this is task 1.11. A game has exactly one
rating table. A leaderboard-mode match moves both seats, a sandbox match moves nothing,
and nothing under `metrics/` reads any of it -- `tournament/ratings.py` measures the
*game* and is untouched. This table measures the *players*.

The update is Glicko-2 with a rating period of one match, as lichess runs it, because a
person should see their own match move their own number and nobody's rating should change
because somebody else played. `rate_period` is Glickman's published algorithm over a whole
period, so his worked example can be pinned against it; the site calls it through `update`
with one opponent.

The table is a **fold**. `replay` takes a seed and a list of finished matches in finish
order and produces the table; `rebuild` does that against the database and rewrites the
`ratings` rows. So a bug in the update is repaired by replaying the rows rather than by
patching numbers, and the seed is recoverable too -- `seeds_for` reads each player's first
rated match's before-columns, which no update ever touches.

Two kinds of player sit in one table. A person starts at 1500 with a deviation of 350 and
Glicko-2's default volatility, and is *established* once the deviation falls under 110. A
computer player is seeded once from the game's picker ladder at its ladder rating on the
site's scale, with a deviation of 100 and a volatility pinned at zero -- pinned because
its strength does not change, which is also why its deviation can only shrink. Human
results move computer players; §5 says why that has to be true.

## What task 1.4 calls

`matches` and the two modes (task 1.4) own the match row; this module owns what the row
means for the table. Two calls, in this order:

    at the start of a leaderboard match:
        top(load(connection, game))     -- every computer player within 5 Elo of the
                                           highest; `matches.top` is whether the computer
                                           seat is one of them
    at the finish of a leaderboard match:
        finish(connection, match_id)    -- reads the row, moves both seats, writes the
                                           two `ratings` rows and the row's eight
                                           before/after columns; returns what moved, or
                                           None when the row is not one the table counts

Neither commits. The caller owns the transaction, so a match row and the ratings it moved
land together or not at all.

## What the loader calls

Task 1.6 loads `runs/opponents/<hash>.json` into `measurements.ladder` and seeds the
game's computer players from it with `seed`. A second run adds nothing, because seeding
only ever adds players the table has not seen.

Match rows are never deleted. The fold assumes that, and the retention question (§8) is
open; option A, the recommended default, is what this is built on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

# ------------------------------------------------------------------- §5's constants

# Every number §5 names, in one block, because the design argues for each of them and a
# value that drifted away from the prose would be invisible.

SITE_ORIGIN = 1500.0
"""The site's scale is the ladder's plus this. A ladder centres on its roster's mean, so
a newcomer at 1500 sits in the middle of the roster and the numbers read as Elo."""

NEWCOMER_RATING = 1500.0
NEWCOMER_DEVIATION = 350.0
NEWCOMER_VOLATILITY = 0.06
"""Where a person starts. The volatility is Glicko-2's own default."""

COMPUTER_DEVIATION = 100.0
"""Roughly what the 20 to 24 games under each ladder rung leave."""

COMPUTER_VOLATILITY = 0.0
"""Pinned, and zero rather than merely small. §5 asks for a volatility fixed low and a
deviation that never inflates with time; volatility is exactly the model of strength
drifting between rating periods, and a computer player's strength does not drift. At zero
the inflation term vanishes and the deviation can only shrink, which makes that promise
arithmetic rather than an approximation."""

ESTABLISHED_DEVIATION = 110.0
"""Under this a player is ranked rather than shown with a `?`. Ten to fifteen matches for
a person; a computer player is established from its seed."""

TOP_BAND = 5.0
"""Every computer player within this of the highest-rated one is *the top*, so a tie the
ladder could not break stays a tie until a match breaks it."""

TAU = 0.5
"""Glicko-2's system constant, which bounds how fast volatility moves."""

CONVERGENCE = 1e-6
GLICKO2_SCALE = 173.7178
"""Glickman's constant for the step between the Elo scale and the internal one."""


class RatingError(ValueError):
    """A table was asked for something the match rows cannot answer.

    A `ValueError` like `db.ConfigError` and `migrate.MigrationError`, so `cli.main`
    prints it as one line rather than a stack trace.
    """


# ----------------------------------------------------------------------- who a row is


@dataclass(frozen=True)
class PlayerKey:
    """A person or a computer player, exactly one of the two.

    The DDL's `ratings_player_is_one_or_the_other` check in Python. A computer player is
    its agent string, budget and objective included, because that string is the identity:
    `minimax:nodes=400` and `minimax:nodes=1600` are two rated players.
    """

    user: int | None = None
    agent: str | None = None

    def __post_init__(self) -> None:
        if (self.user is None) == (self.agent is None):
            raise RatingError("a player is either a person or a computer player, not both or neither")

    @property
    def is_computer(self) -> bool:
        return self.agent is not None


def person(user_id: int) -> PlayerKey:
    return PlayerKey(user=user_id)


def computer(agent: str) -> PlayerKey:
    return PlayerKey(agent=agent)


@dataclass(frozen=True)
class Player:
    """One row of a game's table."""

    key: PlayerKey
    rating: float
    deviation: float
    volatility: float
    matches: int = 0

    @property
    def is_computer(self) -> bool:
        return self.key.is_computer

    @property
    def established(self) -> bool:
        return self.deviation < ESTABLISHED_DEVIATION

    @property
    def pinned(self) -> bool:
        """Whether the volatility is held where it was seeded.

        True for every computer player and false for every person, so the caller never
        has to remember which players drift.
        """
        return self.key.is_computer


Table = dict[PlayerKey, Player]
"""A game's whole rating table, keyed by player. The fold's state."""


def newcomer(key: PlayerKey) -> Player:
    """A person who has not played this game yet."""
    return Player(key, NEWCOMER_RATING, NEWCOMER_DEVIATION, NEWCOMER_VOLATILITY)


# --------------------------------------------------------------------------- Glicko-2


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _expected(mu: float, opponent_mu: float, opponent_phi: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(opponent_phi) * (mu - opponent_mu)))


def _volatility(phi: float, sigma: float, variance: float, delta: float) -> float:
    """Step 5 of the paper, solved by the Illinois variant of regula falsi.

    The published procedure, written out rather than approximated, because the whole
    point of pinning Glickman's worked example is that this arithmetic does not drift.
    """
    a = math.log(sigma * sigma)
    phi_squared = phi * phi
    delta_squared = delta * delta

    def f(x: float) -> float:
        exp_x = math.exp(x)
        top = exp_x * (delta_squared - phi_squared - variance - exp_x)
        bottom = 2.0 * (phi_squared + variance + exp_x) ** 2
        return top / bottom - (x - a) / (TAU * TAU)

    lower = a
    if delta_squared > phi_squared + variance:
        upper = math.log(delta_squared - phi_squared - variance)
    else:
        step = 1
        while f(a - step * TAU) < 0.0:
            step += 1
        upper = a - step * TAU

    f_lower, f_upper = f(lower), f(upper)
    while abs(upper - lower) > CONVERGENCE:
        middle = lower + (lower - upper) * f_lower / (f_upper - f_lower)
        f_middle = f(middle)
        if f_middle * f_upper <= 0.0:
            lower, f_lower = upper, f_upper
        else:
            f_lower /= 2.0
        upper, f_upper = middle, f_middle
    return math.exp(lower / 2.0)


def rate_period(player: Player, results: Sequence[tuple[Player, float]]) -> Player:
    """Glickman's Glicko-2, one rating period, published form.

    `results` pairs each opponent with the player's score against it -- 1, 0.5 or 0. The
    opponents are read at their pre-period values, so a match moves both seats from the
    same table.

    A player with no results keeps their rating and, for a person, widens by their
    volatility, which is the paper's step for an inactive period. A computer player's
    volatility is zero, so nothing widens -- see `COMPUTER_VOLATILITY`.
    """
    mu = (player.rating - SITE_ORIGIN) / GLICKO2_SCALE
    phi = player.deviation / GLICKO2_SCALE
    sigma = player.volatility

    if not results:
        widened = math.sqrt(phi * phi + sigma * sigma)
        return replace(player, deviation=widened * GLICKO2_SCALE)

    weighted = 0.0
    inverse_variance = 0.0
    for opponent, score in results:
        opponent_mu = (opponent.rating - SITE_ORIGIN) / GLICKO2_SCALE
        opponent_phi = opponent.deviation / GLICKO2_SCALE
        g = _g(opponent_phi)
        expected = _expected(mu, opponent_mu, opponent_phi)
        inverse_variance += g * g * expected * (1.0 - expected)
        weighted += g * (score - expected)

    variance = 1.0 / inverse_variance
    delta = variance * weighted

    if player.pinned:
        # The volatility step and the inflation it feeds are both about strength drifting
        # between periods, which a computer player's does not.
        sigma_after = sigma
        phi_star = phi
    else:
        sigma_after = _volatility(phi, sigma, variance, delta)
        phi_star = math.sqrt(phi * phi + sigma_after * sigma_after)

    phi_after = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + inverse_variance)
    mu_after = mu + phi_after * phi_after * weighted

    return Player(
        key=player.key,
        rating=mu_after * GLICKO2_SCALE + SITE_ORIGIN,
        deviation=phi_after * GLICKO2_SCALE,
        volatility=sigma_after,
        matches=player.matches + len(results),
    )


def update(player: Player, opponent: Player, score: float) -> Player:
    """One match, which is what a rating period is here."""
    return rate_period(player, [(opponent, score)])


# --------------------------------------------------------------------------- the fold


@dataclass(frozen=True)
class Finished:
    """One finished leaderboard match, as the fold reads it.

    `result` is seat 0's score -- 1, 0.5 or 0 -- which is what `matches.result` holds. A
    draw is half a point each and clears nothing; the update ignores the seat, as Elo
    does, and the seat stays on the row for the readings that want it.
    """

    id: str
    seat0: PlayerKey
    seat1: PlayerKey
    result: float


@dataclass(frozen=True)
class Move:
    """What one match did to one seat."""

    before: Player
    after: Player

    @property
    def key(self) -> PlayerKey:
        return self.after.key


@dataclass(frozen=True)
class Update:
    """What one match did to the table."""

    match_id: str
    seat0: Move
    seat1: Move


def apply_match(table: Table, match: Finished) -> tuple[Table, Update]:
    """Move both seats from the same pre-match table.

    A person the table has not seen enters as a newcomer here rather than being seeded in
    advance, because a person's seed is a constant and a match is the first thing that
    proves they exist. A computer player the table has not seen is a refusal: its seed is
    the ladder's, and inventing 1500 for it would quietly put a strong player at the
    middle of its own table.
    """
    left = _seat(table, match.seat0)
    right = _seat(table, match.seat1)
    moved = {
        left.key: Move(left, update(left, right, match.result)),
        right.key: Move(right, update(right, left, 1.0 - match.result)),
    }
    after = dict(table)
    after[left.key] = moved[left.key].after
    after[right.key] = moved[right.key].after
    return after, Update(match.id, moved[left.key], moved[right.key])


def _seat(table: Table, key: PlayerKey) -> Player:
    if key in table:
        return table[key]
    if key.is_computer:
        raise RatingError(
            f"{key.agent} is not on this game's table, so it has no seed; "
            f"a computer player enters from the game's picker ladder"
        )
    return newcomer(key)


def replay(seeds: Mapping[PlayerKey, Player], matches: Iterable[Finished]) -> Table:
    """The whole table, folded from its seed over the matches in finish order.

    Pure, so the fold can be checked without a database. The caller owns the order; the
    database-facing `finished` reads the rows the way `matches_by_game` indexes them.
    """
    table: Table = dict(seeds)
    for match in matches:
        table, _ = apply_match(table, match)
    return table


# -------------------------------------------------------------------------- the ladder


@dataclass(frozen=True)
class Rung:
    """One standing from `runs/opponents/<hash>.json`."""

    name: str
    rating: float


def rungs(ladder: Mapping[str, Any]) -> tuple[Rung, ...]:
    """The picker ladder's standings, which is all the seeding reads of it.

    The file carries games, wins, labels and a plan per rung; none of them is a seed. The
    ladder is read once, as a seed, and a later run of it changes nothing on a table that
    exists.
    """
    standings = ladder.get("standings") or ()
    return tuple(Rung(str(row["name"]), float(row["rating"])) for row in standings)


def ladder_shift(standings: Sequence[Rung], table: Mapping[PlayerKey, Player]) -> float:
    """How far this ladder's zero sits from the table's scale.

    A ladder centres on its own roster's mean, so the zero moves when the roster does and
    two runs of the same game put the same rung at two numbers. With nothing shared, the
    shift is the site's origin and the seed is the ladder rating plus 1500. With rungs in
    common -- a later run that added a fitted objective, a rule of thumb, Fairy-Stockfish
    -- the shift is the mean gap over those rungs, read against where the table has them
    *now* rather than where they were seeded, so a newcomer arrives beside the players
    human results have already placed.
    """
    shared = [
        (rung, table[computer(rung.name)])
        for rung in standings
        if computer(rung.name) in table
    ]
    if not shared:
        return SITE_ORIGIN
    return sum(row.rating - rung.rating for rung, row in shared) / len(shared)


def seeds_from_ladder(
    standings: Sequence[Rung], table: Mapping[PlayerKey, Player] | None = None
) -> Table:
    """The computer players this ladder adds, at their seats on the site's scale.

    Only the ones the table has not seen: §5's loader adds computer players it has not
    seen and never touches the ones it has, so a re-rated ladder reseeds nothing and a
    second load changes no rows.
    """
    from chess_variant_sandbox.web.opponents import never_rated

    existing: Mapping[PlayerKey, Player] = table or {}
    shift = ladder_shift(standings, existing)
    added: Table = {}
    for rung in standings:
        key = computer(rung.name)
        # Q683: a learned net's player is practice only, so no ladder seats it here.
        if key in existing or never_rated(rung.name):
            continue
        added[key] = Player(key, rung.rating + shift, COMPUTER_DEVIATION, COMPUTER_VOLATILITY)
    return added


# ----------------------------------------------------------------------- the top query


def computers(table: Mapping[PlayerKey, Player]) -> tuple[Player, ...]:
    """The table's computer players, strongest first."""
    found = [row for row in table.values() if row.is_computer]
    return tuple(sorted(found, key=lambda row: (-row.rating, row.key.agent or "")))


def top(table: Mapping[PlayerKey, Player]) -> tuple[Player, ...]:
    """Every computer player within `TOP_BAND` of the highest, read live.

    A tie the ladder could not break stays a tie until a match breaks it, so clearing a
    tied game means beating each of the tied and the page names them all. Empty on a game
    whose table has no computer players, which is a game with no ladder and so no
    leaderboard mode at all.
    """
    ranked = computers(table)
    if not ranked:
        return ()
    return tuple(row for row in ranked if row.rating >= ranked[0].rating - TOP_BAND)


def is_top(table: Mapping[PlayerKey, Player], key: PlayerKey) -> bool:
    """What task 1.4 writes into `matches.top` at the start of a match."""
    return any(row.key == key for row in top(table))


# --------------------------------------------------------- the database, thinly covered

# Everything below is a layer over the `ratings` and `matches` tables of
# `migrations/0001_store.sql` and holds no arithmetic of its own. Nothing here commits:
# the caller owns the transaction, so a match row and the ratings it moved land together.

def _row_to_player(row: Sequence[Any]) -> Player:
    user, agent, rating, deviation, volatility, matches = row
    key = PlayerKey(user=int(user) if user is not None else None, agent=agent)
    return Player(key, float(rating), float(deviation), float(volatility), int(matches))


def load(connection: Any, game: str) -> Table:
    """A game's whole rating table, people and computer players together."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select player_user, player_agent, rating, deviation, volatility, matches"
            " from ratings where game = %s",
            (game,),
        )
        rows = cursor.fetchall()
    return {player.key: player for player in map(_row_to_player, rows)}


def load_many(connection: Any, games: Iterable[str] = ()) -> dict[str, Table]:
    """Every game's table in one query, or just the ones asked for.

    What the games page needs: sixty cards each wanting to know who the top is, and sixty
    round trips to find out is sixty too many. A game with no rows at all is absent from
    the answer rather than present and empty, so the caller can tell *unrated* from
    *rated and nobody has played*.
    """
    wanted = tuple(games)
    clause = " where game = any (%s)" if wanted else ""
    with connection.cursor() as cursor:
        cursor.execute(
            "select game, player_user, player_agent, rating, deviation, volatility,"
            f" matches from ratings{clause}",
            (list(wanted),) if wanted else (),
        )
        rows = cursor.fetchall()
    found: dict[str, Table] = {}
    for row in rows:
        player = _row_to_player(row[1:])
        found.setdefault(str(row[0]), {})[player.key] = player
    return found


def store(connection: Any, game: str, players: Iterable[Player]) -> None:
    """Write rows, replacing any the table already has for those players.

    Two statements rather than one because "one row per player per game" is two partial
    unique indexes, each covering the half of the table its column is filled on, and
    `on conflict` has to name the index's predicate to infer a partial index at all.
    """
    upsert = (
        "insert into ratings"
        " (game, {column}, rating, deviation, volatility, matches, established)"
        " values (%s, %s, %s, %s, %s, %s, %s)"
        " on conflict (game, {column}) where {column} is not null do update set"
        " rating = excluded.rating, deviation = excluded.deviation,"
        " volatility = excluded.volatility, matches = excluded.matches,"
        " established = excluded.established, updated = now()"
    )
    person_sql = upsert.format(column="player_user")
    computer_sql = upsert.format(column="player_agent")
    with connection.cursor() as cursor:
        for player in players:
            who: int | str | None = player.key.agent if player.is_computer else player.key.user
            cursor.execute(
                computer_sql if player.is_computer else person_sql,
                (
                    game,
                    who,
                    player.rating,
                    player.deviation,
                    player.volatility,
                    player.matches,
                    player.established,
                ),
            )


def seed(connection: Any, game: str, ladder: Mapping[str, Any]) -> tuple[Player, ...]:
    """Seed a game's computer players from its picker ladder, once each.

    What task 1.6's loader calls, with `runs/opponents/<hash>.json` as it read it. A
    second run adds nothing and touches nothing, which is that task's own acceptance.
    Returns the players added, so the loader can say what it did.
    """
    added = seeds_from_ladder(rungs(ladder), load(connection, game))
    store(connection, game, added.values())
    return tuple(added.values())


def finished(connection: Any, game: str) -> tuple[Finished, ...]:
    """A game's rated matches, in finish order.

    Rated means leaderboard mode, finished, with a result, and not loaded from the local
    app -- the 54 pre-site rows carry `source = local-app`, were played with undo on hand
    and predate the modes, so they enter no fold and clear nothing. `id` breaks a tie on
    the timestamp, so the order two replays see is the same order.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, seat0_user, seat0_agent, seat1_user, seat1_agent, result"
            " from matches where game = %s and mode = 'leaderboard'"
            " and finished is not null and result is not null and source is null"
            " order by finished, id",
            (game,),
        )
        rows = cursor.fetchall()
    return tuple(
        Finished(
            id=str(row[0]),
            seat0=PlayerKey(user=row[1], agent=row[2]),
            seat1=PlayerKey(user=row[3], agent=row[4]),
            result=float(row[5]),
        )
        for row in rows
    )


def seeds_for(connection: Any, game: str) -> Table:
    """Where the table started, recovered from rows no update ever writes twice.

    A player's seed is the before-columns of their first rated match, which are written
    once at that match's finish and never again; a player who has not played is still at
    their seed, so their current row is it. So the table is reproducible from the match
    rows even after a later ladder run has replaced the file the computer players were
    seeded from. Volatility is not on the match row and does not need to be: a seed's is
    the constant for its kind.
    """
    found: Table = {
        key: player
        for key, player in load(connection, game).items()
        if player.matches == 0
    }
    with connection.cursor() as cursor:
        cursor.execute(
            "select distinct on (player) player, rating, deviation from ("
            "  select coalesce('u' || seat0_user::text, 'a' || seat0_agent) as player,"
            "    seat0_rating_before as rating, seat0_deviation_before as deviation,"
            "    finished, id"
            "  from matches where game = %(game)s and mode = 'leaderboard'"
            "    and finished is not null and source is null"
            "    and seat0_rating_before is not null"
            "  union all"
            "  select coalesce('u' || seat1_user::text, 'a' || seat1_agent),"
            "    seat1_rating_before, seat1_deviation_before, finished, id"
            "  from matches where game = %(game)s and mode = 'leaderboard'"
            "    and finished is not null and source is null"
            "    and seat1_rating_before is not null"
            ") seats order by player, finished, id",
            {"game": game},
        )
        rows = cursor.fetchall()
    for tag, rating, deviation in rows:
        key = person(int(tag[1:])) if tag[0] == "u" else computer(tag[1:])
        volatility = COMPUTER_VOLATILITY if key.is_computer else NEWCOMER_VOLATILITY
        found[key] = Player(key, float(rating), float(deviation), volatility)
    return found


def rebuild(
    connection: Any, game: str, seeds: Mapping[PlayerKey, Player] | None = None
) -> Table:
    """Rebuild a game's `ratings` rows by replaying its matches from the seed.

    The repair when the update was wrong, and the check that the table is a fold rather
    than an accumulation of edits. `seeds` defaults to `seeds_for`; pass the ladder's own
    seeds to rebuild a table whose match rows carry no before-columns.
    """
    start = seeds_for(connection, game) if seeds is None else seeds
    table = replay(start, finished(connection, game))
    store(connection, game, table.values())
    return table


def finish(connection: Any, match_id: str) -> Update | None:
    """Move both seats of a finished leaderboard match. What task 1.4 calls.

    Returns None -- and writes nothing -- for a row the table does not count: a sandbox
    match, one still running, one loaded from the local app. Otherwise it writes the two
    `ratings` rows and the row's own eight before/after columns, so the match carries what
    it did and the rating history is the match rows.

    A second call on the same match is refused rather than counted twice, which is what
    the row's own after-columns say.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "select id, game, mode, seat0_user, seat0_agent, seat1_user, seat1_agent,"
            " result, finished, source, seat0_rating_after from matches where id = %s",
            (match_id,),
        )
        row = cursor.fetchone()
    if row is None:
        raise RatingError(f"no match {match_id!r}")
    (found, game, mode, seat0_user, seat0_agent, seat1_user, seat1_agent,
     result, ended, source, rated) = row
    if mode != "leaderboard" or ended is None or result is None or source is not None:
        return None
    if rated is not None:
        raise RatingError(f"match {match_id!r} has already moved this game's table")

    match = Finished(
        id=str(found),
        seat0=PlayerKey(user=seat0_user, agent=seat0_agent),
        seat1=PlayerKey(user=seat1_user, agent=seat1_agent),
        result=float(result),
    )
    _, moved = apply_match(load(connection, game), match)
    store(connection, game, (moved.seat0.after, moved.seat1.after))
    with connection.cursor() as cursor:
        cursor.execute(
            "update matches set"
            " seat0_rating_before = %s, seat0_deviation_before = %s,"
            " seat0_rating_after = %s, seat0_deviation_after = %s,"
            " seat1_rating_before = %s, seat1_deviation_before = %s,"
            " seat1_rating_after = %s, seat1_deviation_after = %s"
            " where id = %s",
            (
                moved.seat0.before.rating,
                moved.seat0.before.deviation,
                moved.seat0.after.rating,
                moved.seat0.after.deviation,
                moved.seat1.before.rating,
                moved.seat1.before.deviation,
                moved.seat1.after.rating,
                moved.seat1.after.deviation,
                match_id,
            ),
        )
    return moved
