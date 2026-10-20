"""A game in progress as a row, in the two modes §4 gives it.

Task 1.4 of `docs/plan-website.md`. Two properties carry the whole row: a sandbox match
undoes, restarts and appears in no statistic, and a leaderboard match closed mid-game
reads as a resignation once the sweep has been past. Everything else here is one of those
two seen from a different angle -- the *top* bit written at the start, the rating update
at the finish, the refusal on a game with no picker ladder, and the readings tasks 1.5 and
1.6 are built on.

These need a real Postgres and skip with a reason without one, each inside a scratch
schema it drops -- the recipe task 1.2 settled, since `py-pglite` does not work on
Windows. The game played is `rulesets/dull.json` with `max_plies` turned down, because a
match that ends after two plies is the cheapest possible finish and the arithmetic of the
rating itself is pinned in `tests/test_site_rating.py`, not here.

Nothing in this file works out a legal move, an outcome or a winner. The engine answers
those and the store records what it said.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import site_store

from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import migrate, quota
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.play import Sessions as PlaySessions

REPOSITORY = Path(__file__).resolve().parents[1]
DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))

AGENT = "minimax:nodes=400"


def _dull(max_plies: int) -> dict[str, Any]:
    """`dull` with its ply limit moved, which is a different game and a different hash."""
    spec = json.loads(json.dumps(DULL))
    spec["rules"]["max_plies"] = max_plies
    return spec


# A game that ends after one move each, so a finish costs two calls. And one nobody has
# rated, which is what leaderboard mode is refused on.
SHORT = _dull(2)
SHORT_HASH = spec_hash(SHORT)
UNRATED = _dull(4)
UNRATED_HASH = spec_hash(UNRATED)


def _alone(spec: dict[str, Any]) -> dict[str, Any]:
    """One grunt a side, on one file: each side's only move is a step at the other, until
    the second player's grunt has nowhere to go. A game of nothing but forced moves."""
    spec = json.loads(json.dumps(spec))
    spec["setup"]["placements"] = [{"piece": "G", "squares": ["a1"]}]
    return spec


FORCED = _alone(_dull(4))
FORCED_HASH = spec_hash(FORCED)

# One picker ladder in the shape `runs/opponents/<hash>.json` has. Its ratings sum to
# zero, as a fitted one's do.
LADDER = {
    "ruleset_hash": SHORT_HASH,
    "standings": [
        {"name": AGENT, "rating": 300.0, "games": 24},
        {"name": "greedy", "rating": 0.0, "games": 24},
        {"name": "random", "rating": -300.0, "games": 24},
    ],
}


# ------------------------------------------------------------------------ the fixture


class Scratch:
    """A store with the schema in it, and a way to open a second connection onto it.

    Two connections are the point of the second one: *a game survives a server restart*
    is only proved by reading the row back through something that never held the
    position, which is what a restarted process is.
    """

    def __init__(self, url: str, schema: str) -> None:
        self.url = url
        self.schema = schema
        self.connection = self.open()

    def open(self) -> Any:
        connection = connect(self.url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {self.schema}")
        return connection

    def user(self, handle: str) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into users (handle, email) values (%s, %s) returning id",
                (handle, f"{handle}@example.invalid"),
            )
            found = cursor.fetchone()
        self.connection.commit()
        return int(found[0])

    def game(self, digest: str, spec: dict[str, Any], ladder: Any = None) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (digest, json.dumps(spec), spec["name"], "corpus"),
            )
        if ladder is not None:
            r.seed(self.connection, digest, ladder)
        self.connection.commit()
        return digest

    def rows(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("select count(*) from matches")
            return int(cursor.fetchone()[0])

    def backdate(self, match_id: str, ago: timedelta) -> None:
        """Make a match look like nobody has touched it for `ago`."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "update matches set started = started - %s, moved = moved - %s where id = %s",
                (ago, ago, match_id),
            )
        self.connection.commit()


# Past the sweep's deadline, and safely inside it. The deadline is `IDLE_LIMIT`, thirty
# days, so these bracket it by weeks rather than minutes.
STALE = timedelta(days=45)
FRESH = timedelta(days=1)


@contextlib.contextmanager
def _scratch() -> Iterator[Scratch]:
    url, name = site_store.emptied("no database to play a match in")
    scratch = Scratch(url, name)
    # Committed, so the `search_path` survives a test's rollback. `migrate.run` used to
    # commit it in passing, when every test migrated a schema of its own.
    scratch.connection.commit()
    try:
        yield scratch
    finally:
        scratch.connection.close()


@pytest.fixture
def store() -> Iterator[Scratch]:
    """A store holding one rated game, one unrated one, and one person."""
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        yield scratch


def _play_out(scratch: Scratch, match_id: str) -> dict[str, Any]:
    """One move each, which is the whole of a `max_plies = 2` game."""
    connection = scratch.connection
    snapshot = m.session(connection, match_id).snapshot()
    m.play(connection, match_id, snapshot["moves"][0]["move"])
    return m.play_agent(connection, match_id)


def _move(connection: Any, match_id: str) -> dict[str, Any]:
    """The person's move: the first one the engine offers, since which one is not the point."""
    snapshot = m.session(connection, match_id).snapshot()
    return m.play(connection, match_id, snapshot["moves"][0]["move"])


# --------------------------------------------------------------- a row, not a dictionary


class TestTheRow:
    def test_a_match_survives_the_process_that_started_it(self, store: Scratch) -> None:
        """The whole reason a game is a row. Read back through a connection that never
        saw the position, which is what a restarted unit is."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        first = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, first["moves"][0]["move"])
        m.play_agent(store.connection, match.id)
        store.connection.commit()

        second = store.open()
        try:
            reopened = m.session(second, match.id).snapshot()
        finally:
            second.close()

        assert len(reopened["history"]) == 2
        assert reopened["board"] != first["board"], "the moves did not come back"
        assert reopened["id"] == match.id

    def test_the_moves_are_what_is_stored_and_the_engine_is_what_reads_them(
        self, store: Scratch
    ) -> None:
        """A match row holds integers; every square on the board is derived from them.

        So the same row replays to the same position twice, which is what makes the
        rating fold's *replay the rows* repair true one layer down.
        """
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        snapshot = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, snapshot["moves"][0]["move"])
        once = m.session(store.connection, match.id).snapshot()
        twice = m.session(store.connection, match.id).snapshot()
        assert once["board"] == twice["board"]
        assert once["history"] == twice["history"]

    def test_a_game_the_store_has_never_seen_is_refused_by_name(self, store: Scratch) -> None:
        user = store.user("ada")
        with pytest.raises(m.MatchError) as error:
            m.start(store.connection, "0" * 12, m.SANDBOX, ["human", AGENT], user)
        assert "this game isn't on the site" in str(error.value)
        assert "site load" not in str(error.value), "a visitor is not told a command"


# ------------------------------------------------------------------------ the seed (Q987)


def _seed_of(connection: Any, match_id: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute("select seed from matches where id = %s", (match_id,))
        return int(cursor.fetchone()[0])


def _opening(store: Scratch, user: int, seed: int | None) -> tuple[str, int]:
    """A match against the core's `random` player, the person's first offered move and
    the reply: its id, and the move the computer chose. The reply is drawn from the
    seat's stream at `seed + ply`, so it is the seed that decides which one."""
    match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", "random"], user, seed=seed)
    _move(store.connection, match.id)
    m.play_agent(store.connection, match.id)
    store.connection.commit()
    return match.id, int(m.session(store.connection, match.id).played[1])


class TestTheSeed:
    """A match nobody named a seed for gets a fresh one, written in its row, and a named
    one is kept (Q987, `docs/reference/round-hardest-weakness.md` §7)."""

    def test_two_new_matches_get_different_seeds(self, store: Scratch) -> None:
        user = store.user("ada")
        made = [
            m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
            for _ in range(4)
        ]
        seeds = [_seed_of(store.connection, match.id) for match in made]
        assert len(set(seeds)) == len(seeds), "two matches were given the same seed"
        assert [match.seed for match in made] == seeds, "the row is not what was played"

    def test_a_named_seed_is_kept_and_zero_is_one_like_any_other(self, store: Scratch) -> None:
        user = store.user("ada")
        for named in (0, 7, 2**31 - 1):
            match = m.start(
                store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user, seed=named
            )
            assert _seed_of(store.connection, match.id) == named

    def test_a_stored_match_replays_identically(self, store: Scratch) -> None:
        """Read back through a connection that never saw it, a match is the same game:
        the same seed, the same moves, the same board. And a second match given that seed
        meets the same reply, so the seed in the row is what the reply was drawn from."""
        user = store.user("ada")
        match_id, reply = _opening(store, user, None)
        seed = _seed_of(store.connection, match_id)
        first = m.session(store.connection, match_id)

        second = store.open()
        try:
            reopened = m.session(second, match_id)
        finally:
            second.close()

        assert reopened.seed == first.seed == seed
        assert reopened.played == first.played
        assert reopened.snapshot()["board"] == first.snapshot()["board"]
        assert _opening(store, user, seed)[1] == reply

    def test_the_reply_moves_with_the_seed(self, store: Scratch) -> None:
        """Under one seed for every game the computer answered a position the same way
        in every game; under twelve named seeds it does not."""
        user = store.user("ada")
        assert len({_opening(store, user, named)[1] for named in range(12)}) > 1


# ------------------------------------------------------------ how far the machine looked


def _play_on(connection: Any, match_id: str, plies: int) -> dict[str, Any]:
    """Up to `plies` more moves, or to the end: the person's first offered move on their
    turn, the core's reply on a machine's. Which move the person makes is not the point."""
    snapshot = m.session(connection, match_id).snapshot()
    for _ in range(plies):
        if snapshot["outcome"] is not None:
            break
        if snapshot["awaitingAgent"]:
            snapshot = m.play_agent(connection, match_id)
        else:
            snapshot = m.play(connection, match_id, snapshot["moves"][0]["move"])
    return snapshot


def _depths(snapshot: dict[str, Any]) -> list[int | None]:
    return [move["depth"] for move in snapshot["history"]]


def _stored(connection: Any, match_id: str) -> Any:
    """The row's `depths` as the store holds it: a list, or None for a row with none."""
    with connection.cursor() as cursor:
        cursor.execute("select depths from matches where id = %s", (match_id,))
        return cursor.fetchone()[0]


class TestHowFarItLookedIsKept:
    """Q424, after Q288. The board says *looked 3 moves ahead* beside a searched machine
    move, and the number is the core's, reported once, with the reply. The site rebuilds
    a match from its row on every request and a rebuilt move is not searched again, so a
    reload used to lose every depth. The row keeps them beside the moves now, and a board
    read back through a connection that never saw the game says what the played one said.
    """

    @pytest.mark.parametrize(
        "seats", [["human", AGENT], [AGENT, AGENT]], ids=["a person", "a watch"]
    )
    def test_a_reloaded_board_says_how_far_each_move_looked(
        self, store: Scratch, seats: list[str]
    ) -> None:
        user = store.user("ada") if "human" in seats else None
        match = m.start(store.connection, UNRATED_HASH, m.SANDBOX, seats, user)
        played = _play_on(store.connection, match.id, 4)
        store.connection.commit()

        told = _depths(played)
        assert len(told) == 4
        for move, depth in zip(played["history"], told, strict=True):
            if move["by"] == "human":
                assert depth is None, told
            else:
                assert isinstance(depth, int), told

        second = store.open()
        try:
            reloaded = m.session(second, match.id).snapshot()
        finally:
            second.close()
        assert _depths(reloaded) == told
        assert reloaded["history"] == played["history"], "the reload is not the game played"
        assert _stored(store.connection, match.id) == told

    @pytest.mark.parametrize(
        ("digest", "seats"),
        [
            (UNRATED_HASH, ["random", "greedy"]),
            (UNRATED_HASH, ["greedy", "mcts:nodes=50"]),
            (FORCED_HASH, [AGENT, AGENT]),
        ],
        ids=["random and greedy", "the playout search", "a searching player forced"],
    )
    def test_a_move_nothing_searched_for_stores_nothing(
        self, store: Scratch, digest: str, seats: list[str]
    ) -> None:
        """A player that does not deepen, and a searching one with one legal move, which
        does not search at all. The core's zero from either would read as *played without
        looking ahead*, so what is stored is a null beside each move -- an array, written,
        rather than a row that never recorded anything."""
        store.game(FORCED_HASH, FORCED)
        match = m.start(store.connection, digest, m.SANDBOX, seats)
        snapshot = m.session(store.connection, match.id).snapshot()
        while snapshot["outcome"] is None:
            if digest == FORCED_HASH:
                assert len(snapshot["moves"]) == 1, "the forced game offered a choice"
            snapshot = m.play_agent(store.connection, match.id)
        store.connection.commit()

        plies = len(snapshot["history"])
        assert plies >= 3
        assert _stored(store.connection, match.id) == [None] * plies
        assert _depths(m.session(store.connection, match.id).snapshot()) == [None] * plies

    def test_a_take_back_and_a_restart_take_the_depths_with_the_moves(
        self, store: Scratch
    ) -> None:
        user = store.user("ada")
        match = m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", AGENT], user)
        played = _play_on(store.connection, match.id, 4)
        told = _depths(played)

        # Back past the reply and the person's move before it, to the person's turn.
        taken_back = m.undo(store.connection, match.id)
        assert _depths(taken_back) == told[:2]
        assert _stored(store.connection, match.id) == told[:2]
        assert _depths(m.session(store.connection, match.id).snapshot()) == told[:2]

        m.restart(store.connection, match.id)
        assert _stored(store.connection, match.id) == []
        assert _depths(_play_on(store.connection, match.id, 2))[0] is None

    def test_a_game_stored_before_the_column_loads_with_none_and_plays_on(
        self, tmp_path: Path
    ) -> None:
        """What the deploy does to a match somebody is in the middle of. A store really at
        `0013`, a row the code of that time wrote -- the move list and nothing beside it --
        and then `0014` over it. The board comes back with no depth beside any move, and
        the next reply keeps its own: nulls for the moves before it, the core's number for
        the one it played."""
        older = tmp_path / "migrations-before-0014"
        older.mkdir()
        for migration in migrate.available():
            if migration.number < 14:
                (older / migration.name).write_text(migration.sql, encoding="utf-8")

        url = site_store.database("no database to carry a match across 0014 in")
        schema = f"before_depths_{secrets.token_hex(6)}"
        admin = connect(url)
        with admin.cursor() as cursor:
            cursor.execute(f"create schema {schema}")
        admin.commit()
        scratch = Scratch(url, schema)
        scratch.connection.commit()
        try:
            migrate.run(scratch.connection, older)
            scratch.connection.commit()
            user = scratch.user("ada")
            scratch.game(UNRATED_HASH, UNRATED)
            # `start` writes no depth, so it is the start of that time too.
            match = m.start(
                scratch.connection, UNRATED_HASH, m.SANDBOX, ["human", AGENT], user
            )
            # The two moves the old code would have stored, found the way it found them.
            game = PlaySessions().start(
                library.Game(
                    hash=UNRATED_HASH, name="dull", description="", path=Path(),
                    source="site", spec=UNRATED,
                ),
                ("human", AGENT),
                seed=match.seed,
            )
            game.play(game.snapshot()["moves"][0]["move"])
            before = game.play_agent()
            assert isinstance(before["history"][1]["depth"], int)
            with scratch.connection.cursor() as cursor:
                cursor.execute(
                    "update matches set moves = %s::jsonb, moved = now() where id = %s",
                    (json.dumps(game.played), match.id),
                )
            scratch.connection.commit()

            report = migrate.run(scratch.connection)
            scratch.connection.commit()
            # First, not only: the deploy applies every later migration after it too
            # (`0015_provisional.sql`, Q312), as `test_site_links.py` pins `0013`.
            assert report.applied[0] == "0014_depths.sql", report.summary()
            assert _stored(scratch.connection, match.id) is None

            loaded = m.session(scratch.connection, match.id).snapshot()
            assert loaded["board"] == before["board"]
            assert _depths(loaded) == [None, None], "a depth nobody kept was made up"

            after = _play_on(scratch.connection, match.id, 2)
            scratch.connection.commit()
            now = _depths(after)
            assert now[:3] == [None, None, None] and isinstance(now[3], int), now
            assert _stored(scratch.connection, match.id) == now
        finally:
            scratch.connection.close()
            with admin.cursor() as cursor:
                cursor.execute(f"drop schema if exists {schema} cascade")
            admin.commit()
            admin.close()

    def test_the_store_refuses_a_depth_that_is_beside_no_move(self, store: Scratch) -> None:
        """A writer that changed the move list and left the depths would put every depth
        beside the wrong move, silently. The constraint makes that a refusal."""
        import psycopg

        user = store.user("ada")
        match = m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", AGENT], user)
        _play_on(store.connection, match.id, 2)
        store.connection.commit()
        for statement in (
            "update matches set moves = '[]'::jsonb where id = %s",
            "update matches set depths = '[null]'::jsonb where id = %s",
            "update matches set depths = '3'::jsonb where id = %s",
        ):
            with (
                pytest.raises(psycopg.errors.CheckViolation),
                store.connection.cursor() as cursor,
            ):
                cursor.execute(statement, (match.id,))
            store.connection.rollback()
        assert len(_stored(store.connection, match.id)) == 2


# ------------------------------------------------------------------------ sandbox mode


class TestALearnedSeat:
    """Q709: a learned seat records the net that played, and a net that cannot be loaded
    refuses to start or to move -- never the built-in evaluation under its name."""

    PROCEDURE = "aaaaaaaaaaa9"
    LEARNED = f"minimax:net={PROCEDURE},nodes=400"

    @pytest.fixture
    def nets_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
        """An empty `nets/` the seat resolves its net from, as on the box."""
        from chess_variant_sandbox.agents.nets import unregister_net

        monkeypatch.chdir(tmp_path)
        (tmp_path / "nets").mkdir()
        yield tmp_path / "nets"
        unregister_net(self.PROCEDURE, SHORT_HASH)

    def _file(self, nets_dir: Path, seed: int = 1) -> bytes:
        from chess_variant_sandbox.agents.nets import (
            ACCEPTED,
            Layout,
            NetRecord,
            random_net,
            save_net,
            unregister_net,
        )
        from chess_variant_sandbox.web.library import engine_for

        weights = random_net(Layout.of(engine_for(SHORT).spec).n_inputs, seed=seed,
                             accumulator=8, hidden1=4, hidden2=4)
        save_net(NetRecord(self.PROCEDURE, SHORT_HASH, ACCEPTED, weights), nets_dir)
        unregister_net(self.PROCEDURE, SHORT_HASH)
        return weights

    def test_the_row_names_the_net_that_played(self, store: Scratch, nets_dir: Path) -> None:
        from chess_variant_sandbox.agents.nets import ACCEPTED, NetRecord

        weights = self._file(nets_dir)
        digest = NetRecord(self.PROCEDURE, SHORT_HASH, ACCEPTED, weights).digest
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", self.LEARNED])
        assert m.played(store.connection, match.id) == (None, f"net:{digest}")
        assert match.seat1 is not None and match.seat1.agent == self.LEARNED
        assert m.session(store.connection, match.id).played_by == ("", f"net:{digest}")
        snapshot = _play_out(store, match.id)
        assert snapshot["history"][-1]["by"] == self.LEARNED
        assert m.played(store.connection, match.id) == (None, f"net:{digest}")
        # Every other seat records nothing beyond its agent string.
        plain = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT])
        assert m.played(store.connection, plain.id) == (None, None)

    @pytest.mark.parametrize("how", ["missing", "corrupt", "absent"])
    def test_a_net_that_cannot_load_refuses_to_start(
        self, store: Scratch, nets_dir: Path, how: str
    ) -> None:
        if how != "absent":
            self._file(nets_dir)
            weights = nets_dir / f"{SHORT_HASH}-{self.PROCEDURE}.net"
            if how == "missing":
                weights.unlink()
            else:
                weights.write_bytes(weights.read_bytes()[:-3] + b"bad")
        before = store.rows()
        with pytest.raises(m.MatchError) as error:
            m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", self.LEARNED])
        assert str(error.value) == m.UNPLAYABLE
        assert store.rows() == before

    @pytest.mark.parametrize("how", ["gone", "changed"])
    def test_a_net_gone_or_changed_mid_game_refuses_the_reply(
        self, store: Scratch, nets_dir: Path, how: str
    ) -> None:
        from chess_variant_sandbox.agents.nets import unregister_net

        self._file(nets_dir)
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", self.LEARNED])
        _move(store.connection, match.id)
        unregister_net(self.PROCEDURE, SHORT_HASH)
        if how == "gone":
            (nets_dir / f"{SHORT_HASH}-{self.PROCEDURE}.net").unlink()
        else:
            self._file(nets_dir, seed=2)
        with pytest.raises(m.MatchError) as error:
            m.play_agent(store.connection, match.id)
        assert str(error.value) == m.UNPLAYABLE
        assert m.row(store.connection, match.id).plies == 1


class TestSandbox:
    def test_it_undoes_back_to_the_persons_turn(self, store: Scratch) -> None:
        """One ply is not enough with a computer player opposite: taking back your own
        move and leaving the reply on the board is not a position anybody was in."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        opening = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, opening["moves"][0]["move"])

        after = m.undo(store.connection, match.id)

        assert after["history"] == []
        assert after["board"] == opening["board"]
        assert m.row(store.connection, match.id).plies == 0

    def test_it_restarts(self, store: Scratch) -> None:
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        opening = m.session(store.connection, match.id).snapshot()
        finished = _play_out(store, match.id)
        assert finished["outcome"] is not None, "the two-ply game did not end"

        after = m.restart(store.connection, match.id)

        assert after["board"] == opening["board"]
        assert after["outcome"] is None
        row = m.row(store.connection, match.id)
        assert row.plies == 0 and row.running and row.result is None

    def test_it_appears_in_no_statistic(self, store: Scratch) -> None:
        """§4: read by no statistic, no rating, no anchor row and no sort key."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        before = r.load(store.connection, SHORT_HASH)

        _play_out(store, match.id)

        assert m.row(store.connection, match.id).finished is not None, "it did not finish"
        assert r.load(store.connection, SHORT_HASH) == before, "a sandbox game moved a rating"
        assert r.finished(store.connection, SHORT_HASH) == ()
        assert m.leaderboard_matches(store.connection, SHORT_HASH) == ()
        assert m.human_games(store.connection, SHORT_HASH) == []
        assert m.Record(store.open).summary(SHORT_HASH, None)["started"] == 0

    def test_it_writes_no_top_bit(self, store: Scratch) -> None:
        """The bit is what the *Unbeaten* tag reads, and sandbox clears nothing."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        assert match.top is False


# ------------------------------------------------------------ a visitor with no account


class TestAVisitor:
    """Sandbox counts for nothing, so it needs no identity (§4).

    The same three properties `TestSandbox` asserts, on a row whose person is nobody:
    it plays, it takes moves back, and no statistic can see it. Leaderboard mode is the
    one that still asks for a name.
    """

    def test_they_play_take_back_and_start_again(self, store: Scratch) -> None:
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT])
        assert match.seat0 is None, "somebody was seated who never signed in"
        assert match.is_computer(1) and match.is_computer(0) is False

        opening = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, opening["moves"][0]["move"])
        assert m.row(store.connection, match.id).plies == 1

        assert m.undo(store.connection, match.id)["history"] == []
        m.play(store.connection, match.id, opening["moves"][0]["move"])
        assert m.restart(store.connection, match.id)["board"] == opening["board"]

    def test_their_game_survives_the_process_that_started_it(self, store: Scratch) -> None:
        """The reason this is a row and not a dictionary holds without a name on it."""
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT])
        first = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, first["moves"][0]["move"])
        store.connection.commit()

        second = store.open()
        try:
            reopened = m.session(second, match.id).snapshot()
        finally:
            second.close()
        assert len(reopened["history"]) == 1

    def test_their_game_appears_in_no_statistic(self, store: Scratch) -> None:
        """§4 again, read off a row with nobody in one seat."""
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT])
        before = r.load(store.connection, SHORT_HASH)

        _play_out(store, match.id)

        assert m.row(store.connection, match.id).finished is not None, "it did not finish"
        assert r.load(store.connection, SHORT_HASH) == before, "a visitor moved a rating"
        assert r.finished(store.connection, SHORT_HASH) == ()
        assert m.leaderboard_matches(store.connection, SHORT_HASH) == ()
        assert m.human_games(store.connection, SHORT_HASH) == []
        assert m.Record(store.open).summary(SHORT_HASH, None)["started"] == 0

    def test_leaderboard_mode_is_refused_in_words_they_can_read(
        self, store: Scratch
    ) -> None:
        """A rated result moves a table whose rows are people, so it asks for one."""
        with pytest.raises(m.MatchError) as error:
            m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT])
        assert "sign in" in str(error.value) and "practice" in str(error.value)
        assert store.rows() == 0


# ----------------------------------------------------------- two computer players (Q262)


class TestAWatch:
    """Two computer players at one board, which is nobody's game.

    The local application has always offered this and task 1.4 refused it here, so the
    site lost it (Q262). What is asserted is the whole of *nobody's game*: it plays, and
    it enters no rating, no badge, no milestone and no leaderboard count. The rationing
    is the other half -- the box pays for two searches a ply instead of one.
    """

    def _watched(self, store: Scratch, user: int | None = None) -> m.Row:
        return m.start(store.connection, SHORT_HASH, m.SANDBOX, [AGENT, AGENT], user)

    def test_both_seats_play_and_the_game_runs_to_its_end(self, store: Scratch) -> None:
        match = self._watched(store)
        assert match.is_computer(0) and match.is_computer(1)
        assert match.seat0 is not None and match.seat0.user is None

        first = m.play_agent(store.connection, match.id)
        assert first["history"], "the first computer player did not move"
        m.play_agent(store.connection, match.id)

        found = m.row(store.connection, match.id)
        assert found.plies == 2 and found.result is not None
        assert found.finished is not None, "the two-ply game did not end"

    def test_it_enters_no_rating_no_badge_no_milestone_and_no_count(
        self, store: Scratch
    ) -> None:
        """The four readings a finished match can reach, each asked after a whole watch."""
        from chess_variant_sandbox.loop import grid
        from chess_variant_sandbox.site import awards, browse, milestones

        ada = store.user("ada")
        before = r.load(store.connection, SHORT_HASH)
        match = self._watched(store)
        m.play_agent(store.connection, match.id)
        m.play_agent(store.connection, match.id)
        store.connection.commit()
        assert m.row(store.connection, match.id).finished is not None

        assert r.load(store.connection, SHORT_HASH) == before, "a watch moved a rating"
        assert r.finished(store.connection, SHORT_HASH) == ()
        assert awards.held(store.connection, game=SHORT_HASH) == []
        assert awards.held(store.connection, holder=ada) == []
        rows = m.leaderboard_matches(store.connection, running=True)
        assert rows == (), "a watch reached the leaderboard rows"
        tables = {SHORT_HASH: r.load(store.connection, SHORT_HASH)}
        cells = {SHORT_HASH: (0,) * len(grid.AXES)}
        reading = milestones.read_from(ada, rows, tables, cells)
        assert reading.milestones == () and reading.highest == 0
        counted = browse.columns(store.connection, [SHORT_HASH])
        assert counted[SHORT_HASH].human_games == 0
        assert m.human_games(store.connection, SHORT_HASH) == []

    def test_a_game_that_counts_needs_somebody_to_play_it(self, store: Scratch) -> None:
        """Leaderboard mode moves a table whose rows are people, and nobody is here."""
        user = store.user("ada")
        with pytest.raises(m.MatchError, match="counts for nothing"):
            m.start(store.connection, SHORT_HASH, m.LEADERBOARD, [AGENT, AGENT], user)
        assert store.rows() == 0, "the refused watch left a row behind"

    def test_it_is_a_rematch_of_nothing_and_nothing_is_a_rematch_of_it(
        self, store: Scratch
    ) -> None:
        """C1's column stays a count of games somebody played again."""
        user = store.user("ada")
        played = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        with pytest.raises(m.MatchError, match="nobody played"):
            m.start(
                store.connection,
                SHORT_HASH,
                m.SANDBOX,
                [AGENT, AGENT],
                user,
                rematch_of=played.id,
            )

        watched = self._watched(store, user)
        with pytest.raises(m.MatchError, match="no rematch of it"):
            m.start(
                store.connection,
                SHORT_HASH,
                m.SANDBOX,
                ["human", AGENT],
                user,
                rematch_of=watched.id,
            )

    def test_both_seats_are_rationed(self, store: Scratch) -> None:
        """A watch costs two searches a ply, so both seats meet `quota`'s cap.

        The cap is site-wide and about the box, which is why a watch of the strongest
        opponent is what fills it rather than a person's game.
        """
        strong = f"minimax:nodes={quota.STRONG_NODES}"
        for _ in range(quota.STRONG_SEATS // 2):
            m.start(store.connection, SHORT_HASH, m.SANDBOX, [strong, strong], _member(store))
        with pytest.raises(quota.QuotaError, match="none is free"):
            m.start(store.connection, SHORT_HASH, m.SANDBOX, [strong, strong], _member(store))

    def test_both_seats_are_counted_in_one_check(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q301: one seat short of the cap, a watch of two strong players is refused.

        Asked a seat at a time, both seats found the one free seat and the watch was
        seated one over the cap. Three is odd so the watches before cannot land on it.
        """
        monkeypatch.setattr(quota, "STRONG_SEATS", 3)
        strong = f"minimax:nodes={quota.STRONG_NODES}"
        m.start(store.connection, SHORT_HASH, m.SANDBOX, [strong, strong], _member(store))
        with pytest.raises(quota.QuotaError, match="only one is free"):
            m.start(store.connection, SHORT_HASH, m.SANDBOX, [strong, strong], _member(store))
        assert quota.strong_seats(store.connection) == 2
        m.start(store.connection, SHORT_HASH, m.SANDBOX, [strong, AGENT], _member(store))
        assert quota.strong_seats(store.connection) == 3

    def test_start_asks_the_quota_once_with_every_computer_seat(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shape of the fix, pinned: one call per match, never one per seat."""
        asked: list[tuple[int | None, tuple[str, ...]]] = []
        monkeypatch.setattr(
            quota, "check_new_match", lambda _c, user, *agents: asked.append((user, agents))
        )
        user = store.user("ada")
        m.start(store.connection, SHORT_HASH, m.SANDBOX, [AGENT, "minimax:nodes=1600"], user)
        m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        assert asked == [(user, (AGENT, "minimax:nodes=1600")), (user, (AGENT,))]

    def test_there_is_nobody_to_resign_it(self, store: Scratch) -> None:
        match = self._watched(store)
        with pytest.raises(m.MatchError, match="nobody is playing"):
            m.resign(store.connection, match.id)

    def test_taking_a_move_back_steps_one_ply(self, store: Scratch) -> None:
        """A watch has no position a person was asked to move in, so the loop that looks
        for one would unwind the whole game. Played on the longer game, because one ply
        back from one ply cannot tell the two apart."""
        match = m.start(store.connection, UNRATED_HASH, m.SANDBOX, [AGENT, AGENT])
        for _ in range(3):
            m.play_agent(store.connection, match.id)
        assert m.row(store.connection, match.id).plies == 3

        assert len(m.undo(store.connection, match.id)["history"]) == 2
        assert m.row(store.connection, match.id).plies == 2

    def test_two_people_is_still_phase_three(self, store: Scratch) -> None:
        user = store.user("ada")
        with pytest.raises(m.MatchError, match="one person"):
            m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", "human"], user)


# -------------------------------------------------------------------- leaderboard mode


class TestLeaderboard:
    def test_undo_and_restart_are_refused_in_words_a_player_can_read(
        self, store: Scratch
    ) -> None:
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        snapshot = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, snapshot["moves"][0]["move"])

        for act in (m.undo, m.restart):
            with pytest.raises(m.MatchError) as error:
                act(store.connection, match.id)
            assert "rated" in str(error.value) and "practice" in str(error.value)
        assert m.row(store.connection, match.id).plies == 1, "a refusal changed the board"

    def test_a_game_with_no_picker_ladder_is_refused_and_offered_sandbox(
        self, store: Scratch
    ) -> None:
        """§5: a game is rated once its ladder exists, and until then offers sandbox."""
        user = store.user("ada")
        with pytest.raises(m.MatchError) as error:
            m.start(store.connection, UNRATED_HASH, m.LEADERBOARD, ["human", AGENT], user)
        message = str(error.value)
        assert "practice" in message and "have no ratings yet" in message

        sandbox = m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", AGENT], user)
        assert sandbox.mode == m.SANDBOX

    def test_an_opponent_the_table_has_no_row_for_is_refused(self, store: Scratch) -> None:
        """A rated result against a player with no seed would have to invent one, and
        `rating.apply_match` refuses that outright. The refusal belongs before the game."""
        user = store.user("ada")
        with pytest.raises(m.MatchError) as error:
            m.start(
                store.connection, SHORT_HASH, m.LEADERBOARD, ["human", "minimax:nodes=99"], user
            )
        assert "that computer player has no rating on" in str(error.value)
        assert "minimax:" not in str(error.value), "never the agent string"

    def test_the_learned_player_is_never_rated_even_with_a_row(self, store: Scratch) -> None:
        """Q683: a game against a learned net is practice only. Refused by the agent
        string, before the table is read, so even a row somebody wrote for it by hand
        does not make the game rated; practice against it starts as any other."""
        learned = "minimax:net=aaaaaaaaaaaa,nodes=400"
        r.store(store.connection, SHORT_HASH,
                [r.Player(r.computer(learned), 1500.0, 50.0, r.COMPUTER_VOLATILITY)])
        user = store.user("ada")
        with pytest.raises(m.MatchError) as error:
            m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", learned], user)
        assert str(error.value) == m.NEVER_RATED
        # Practice starts as any other, where its net loads (Q709: without one it refuses).
        from chess_variant_sandbox.agents.nets import (
            ACCEPTED,
            Layout,
            NetRecord,
            random_net,
            register_net,
            unregister_net,
        )
        from chess_variant_sandbox.web.library import engine_for

        layout = Layout.of(engine_for(SHORT).spec).n_inputs
        register_net(NetRecord("aaaaaaaaaaaa", SHORT_HASH, ACCEPTED, random_net(layout, seed=1)))
        try:
            practice = m.start(
                store.connection, SHORT_HASH, m.SANDBOX, ["human", learned], user
            )
        finally:
            unregister_net("aaaaaaaaaaaa", SHORT_HASH)
        assert practice.mode == m.SANDBOX

    def test_the_games_rated_net_is_rated_like_any_other_player(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q913: where the learned player's net is the game's rated one (`nets.rated_net`),
        a rated game against it starts as against any player the table holds; another net
        on the same game is still refused by `never_rated`."""
        from chess_variant_sandbox.agents import nets
        from chess_variant_sandbox.agents.nets import (
            ACCEPTED,
            Layout,
            NetRecord,
            random_net,
            register_net,
            unregister_net,
        )
        from chess_variant_sandbox.web.library import engine_for

        learned = "minimax:net=aaaaaaaaaaaa,nodes=400"
        other = "minimax:net=bbbbbbbbbbbb,nodes=400"
        monkeypatch.setattr(
            nets, "rated_net",
            lambda digest, root=nets.STORE_DIR: "aaaaaaaaaaaa" if digest == SHORT_HASH else "",
        )
        r.store(store.connection, SHORT_HASH,
                [r.Player(r.computer(agent), 1500.0, 50.0, r.COMPUTER_VOLATILITY)
                 for agent in (learned, other)])
        user = store.user("ada")
        with pytest.raises(m.MatchError) as error:
            m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", other], user)
        assert str(error.value) == m.NEVER_RATED
        layout = Layout.of(engine_for(SHORT).spec).n_inputs
        register_net(NetRecord("aaaaaaaaaaaa", SHORT_HASH, ACCEPTED, random_net(layout, seed=1)))
        try:
            rated = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", learned], user)
        finally:
            unregister_net("aaaaaaaaaaaa", SHORT_HASH)
        assert rated.mode == m.LEADERBOARD

    def test_the_top_bit_is_written_at_the_start_and_does_not_move(
        self, store: Scratch
    ) -> None:
        """§5: *Unbeaten* is a live question, but a clear is decided at the time and stays.
        So the bit is the table as it was when the match began, never as it is now."""
        user = store.user("ada")
        top = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        below = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", "random"], user)
        assert top.top is True
        assert below.top is False

        # The ladder's top falls below `random`, so the live answer changes.
        r.store(
            store.connection,
            SHORT_HASH,
            [r.Player(r.computer(AGENT), 100.0, 100.0, 0.0)],
        )
        assert r.is_top(r.load(store.connection, SHORT_HASH), r.computer(AGENT)) is False
        assert m.row(store.connection, top.id).top is True

    def test_a_finish_moves_both_ratings_through_the_table(self, store: Scratch) -> None:
        """The two calls §5 hands task 1.4, seen from the other end: after the finish the
        game's table has moved and the row carries what it did."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        machine = r.load(store.connection, SHORT_HASH)[r.computer(AGENT)]

        snapshot = _play_out(store, match.id)
        assert snapshot["outcome"] is not None, "the two-ply game did not end"

        table = r.load(store.connection, SHORT_HASH)
        assert table[r.computer(AGENT)].rating != machine.rating
        assert table[r.computer(AGENT)].matches == 1
        assert r.person(user) in table, "the person never joined the table"
        assert table[r.person(user)].matches == 1

        with store.connection.cursor() as cursor:
            cursor.execute(
                "select seat0_rating_before, seat0_rating_after, seat1_rating_before,"
                " seat1_rating_after from matches where id = %s",
                (match.id,),
            )
            assert all(value is not None for value in cursor.fetchone())

    def test_the_finish_and_the_ratings_it_moved_land_together(self, store: Scratch) -> None:
        """Nothing in the store commits, so a rolled-back finish leaves no rating behind."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        store.connection.commit()
        machine = r.load(store.connection, SHORT_HASH)[r.computer(AGENT)]

        _play_out(store, match.id)
        store.connection.rollback()

        assert m.row(store.connection, match.id).running
        assert r.load(store.connection, SHORT_HASH)[r.computer(AGENT)] == machine

    def test_a_second_match_on_the_same_game_resigns_the_first(self, store: Scratch) -> None:
        """§4: leaving is a resignation, and opening a fresh board is leaving."""
        user = store.user("ada")
        first = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)

        found = m.row(store.connection, first.id)
        assert found.reason == "resigned" and found.left is True
        assert found.result == 0.0, "the person did not lose the game they walked away from"

    def test_a_sandbox_match_is_not_resigned_by_a_new_leaderboard_one(
        self, store: Scratch
    ) -> None:
        user = store.user("ada")
        sandbox = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        assert m.row(store.connection, sandbox.id).running


# ------------------------------------------------------------------------- a rematch


class TestARematch:
    """Box C1 of `docs/archive/plan-meta-game.md`. One nullable column, written from one place.

    Two properties carry it. The board's rematch control names the game it was pressed
    on, and every other way of starting a match leaves the column null -- which is what
    makes a count over it a number rather than a floor. Everything else here is the
    second property: a rematch is a fresh match in every other respect.
    """

    def test_the_next_game_names_the_one_it_followed(self, store: Scratch) -> None:
        user = store.user("ada")
        first = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        _play_out(store, first.id)

        again = m.start(
            store.connection,
            SHORT_HASH,
            m.SANDBOX,
            ["human", AGENT],
            user,
            rematch_of=first.id,
        )

        assert m.row(store.connection, again.id).rematch_of == first.id
        assert m.row(store.connection, first.id).rematch_of is None, "the chain runs one way"

    def test_a_match_started_any_other_way_has_it_null(self, store: Scratch) -> None:
        """The property the prediction rests on: null means *not a rematch*."""
        user = store.user("ada")
        first = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        _play_out(store, first.id)
        second = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        rated = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        visitor = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT])

        for found in (first, second, rated, visitor):
            assert m.row(store.connection, found.id).rematch_of is None

    def test_the_rematch_is_rated_like_any_other_match(self, store: Scratch) -> None:
        """A fresh match in every other respect, and this is the respect that counts."""
        user = store.user("ada")
        first = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        _play_out(store, first.id)
        before = r.load(store.connection, SHORT_HASH)

        again = m.start(
            store.connection,
            SHORT_HASH,
            m.LEADERBOARD,
            [AGENT, "human"],
            user,
            seed=7,
            rematch_of=first.id,
        )
        # The computer player has seat 0 this time, so it moves first: the seats asked
        # for are the seats that were written.
        m.play_agent(store.connection, again.id)
        opening = m.session(store.connection, again.id).snapshot()
        m.play(store.connection, again.id, opening["moves"][0]["move"])

        found = m.row(store.connection, again.id)
        assert found.id != first.id and found.seed == 7
        assert found.seat_of(r.person(user)) == 1, "the seats are the ones asked for"
        assert found.finished is not None and found.rematch_of == first.id
        assert r.load(store.connection, SHORT_HASH) != before, "the rematch rated nothing"
        assert found.rating_before(1) is not None, "the rating going in was not written"
        assert len(r.finished(store.connection, SHORT_HASH)) == 2

    def test_a_match_that_is_not_there_is_refused_by_name(self, store: Scratch) -> None:
        user = store.user("ada")
        with pytest.raises(m.MatchError) as error:
            m.start(
                store.connection,
                SHORT_HASH,
                m.SANDBOX,
                ["human", AGENT],
                user,
                rematch_of="nothing-here",
            )
        assert "nothing-here" in str(error.value)
        assert store.rows() == 0, "a refused rematch left a row behind"

    def test_a_rematch_is_another_game_of_the_same_rules(self, store: Scratch) -> None:
        user = store.user("ada")
        first = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)

        with pytest.raises(m.MatchError, match="same rules"):
            m.start(
                store.connection,
                UNRATED_HASH,
                m.SANDBOX,
                ["human", AGENT],
                user,
                rematch_of=first.id,
            )
        assert store.rows() == 1

    def test_only_a_player_of_that_game_can_play_it_again(self, store: Scratch) -> None:
        """The column holds nothing about a person the two rows do not already hold,
        which is only true while the two rows are the same person's."""
        ada = store.user("ada")
        bob = store.user("bob")
        hers = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], ada)

        with pytest.raises(m.MatchError, match="only a player"):
            m.start(
                store.connection,
                SHORT_HASH,
                m.SANDBOX,
                ["human", AGENT],
                bob,
                rematch_of=hers.id,
            )
        assert store.rows() == 1

    def test_a_visitor_plays_again_and_cannot_name_somebody_elses_game(
        self, store: Scratch
    ) -> None:
        """A visitor has no id to match on, so the shape of the row is what is checked."""
        theirs = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT])
        again = m.start(
            store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], rematch_of=theirs.id
        )
        assert m.row(store.connection, again.id).rematch_of == theirs.id

        ada = store.user("ada")
        hers = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], ada)
        with pytest.raises(m.MatchError, match="sign in"):
            m.start(
                store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], rematch_of=hers.id
            )


# -------------------------------------------------------------------------- the sweep


class TestTheSweep:
    def test_a_match_left_mid_game_reads_as_a_resignation(self, store: Scratch) -> None:
        """The row's own criterion. A tab closed leaves no signal at all, so the limit is
        what decides, and the resignation is a finish -- which is what rates it."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        snapshot = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, snapshot["moves"][0]["move"])
        store.backdate(match.id, STALE)
        before = r.load(store.connection, SHORT_HASH)[r.computer(AGENT)]

        swept = m.sweep(store.connection)

        assert [found.id for found in swept] == [match.id]
        found = m.row(store.connection, match.id)
        assert found.reason == "resigned"
        assert found.left is True
        assert found.finished is not None
        assert found.result == 0.0, "seat 0 was the person, so they lost"
        assert r.load(store.connection, SHORT_HASH)[r.computer(AGENT)].rating > before.rating

    def test_it_leaves_a_match_somebody_is_still_playing(self, store: Scratch) -> None:
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        assert m.sweep(store.connection) == ()
        assert m.row(store.connection, match.id).running

    def test_it_measures_idleness_from_the_last_move(self, store: Scratch) -> None:
        """A long game with a recent move is not idle, however long ago it began."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        snapshot = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, snapshot["moves"][0]["move"])
        with store.connection.cursor() as cursor:
            cursor.execute(
                "update matches set started = started - %s where id = %s",
                (STALE, match.id),
            )
        assert m.sweep(store.connection) == ()

    def test_a_match_idle_a_day_is_left_alone(self, store: Scratch) -> None:
        """The deadline is generous by the user's instruction, so a day's silence is not
        abandonment. The sweep closes a loophole, it does not punish a lost connection."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        snapshot = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, snapshot["moves"][0]["move"])
        store.backdate(match.id, FRESH)
        assert m.sweep(store.connection) == ()
        assert m.row(store.connection, match.id).running

    def test_it_resigns_and_never_deletes(self, store: Scratch) -> None:
        """The user's retention answer of 2026-09-11 is *the account goes, the games
        stay*, and the rating table is a fold over these rows."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        store.backdate(match.id, STALE)
        before = store.rows()
        m.sweep(store.connection)
        assert store.rows() == before

    def test_it_leaves_sandbox_alone(self, store: Scratch) -> None:
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        store.backdate(match.id, STALE)
        assert m.sweep(store.connection) == ()
        assert m.row(store.connection, match.id).running

    def test_the_limit_is_an_argument(self, store: Scratch) -> None:
        """Task 1.7 gathers the quota numbers; moving this one is a call site. A match a
        day idle rides out the default deadline but not a one-hour limit passed in."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        store.backdate(match.id, FRESH)
        assert m.sweep(store.connection) == ()
        assert len(m.sweep(store.connection, timedelta(hours=1))) == 1


# ------------------------------------------------- what tasks 1.5 and 1.6 read off these


class TestTheReadings:
    def test_leaderboard_matches_carries_the_seat_the_top_bit_and_finish_order(
        self, store: Scratch
    ) -> None:
        """The one query rows 1.5 and 1.6 build on, so neither writes SQL against this."""
        user = store.user("ada")
        # A commit after each finish, because `now()` is frozen inside one transaction and
        # finish order would then fall to the random id. On the site each finish is its own
        # committed request, so the timestamps genuinely differ; the test mirrors that.
        first = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", "random"], user)
        m.resign(store.connection, first.id)
        store.connection.commit()
        second = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        _play_out(store, second.id)
        store.connection.commit()
        m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)

        found = m.leaderboard_matches(store.connection, SHORT_HASH)

        assert [row.id for row in found] == [first.id, second.id], "not in finish order"
        assert [row.top for row in found] == [False, True]
        assert all(row.seat_of(r.person(user)) == 0 for row in found)
        assert [row.seat1.agent for row in found] == ["random", AGENT]
        assert all(row.source is None for row in found)

    def test_it_narrows_to_one_person_and_can_include_the_unfinished(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        bob = store.user("bob")
        hers = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        his = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], bob)

        assert m.leaderboard_matches(store.connection, user=ada) == ()
        running = m.leaderboard_matches(store.connection, user=ada, running=True)
        assert [row.id for row in running] == [hers.id]
        assert len(m.leaderboard_matches(store.connection, running=True)) == 2
        assert his.id != hers.id

    def test_the_anchor_reads_leaderboard_rows_only(self, store: Scratch) -> None:
        """`HumanRecord`'s reading, over rows instead of a file. The sandbox game was
        played with undo on hand and says nothing about how hard the game is."""
        user = store.user("ada")
        rated = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        _play_out(store, rated.id)
        sandbox = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], user)
        _play_out(store, sandbox.id)
        store.connection.commit()

        summary = m.Record(store.open).summary(SHORT_HASH, None)

        assert summary["started"] == 1 and summary["finished"] == 1
        assert summary["name"] == SHORT["name"]
        assert summary["byOpponent"][0]["agent"] == AGENT

    def test_the_abandonment_count_becomes_a_resignation_count(self, store: Scratch) -> None:
        """§4's reason for resigning rather than dropping: the anchor's first number is
        the games begun and never finished, and a mode is what closes that hole."""
        user = store.user("ada")
        match = m.start(store.connection, SHORT_HASH, m.LEADERBOARD, ["human", AGENT], user)
        store.connection.commit()
        record = m.Record(store.open)
        assert record.summary(SHORT_HASH, None)["abandoned"] == 1

        store.backdate(match.id, STALE)
        m.sweep(store.connection)
        store.connection.commit()

        summary = record.summary(SHORT_HASH, None)
        assert summary["abandoned"] == 0
        assert summary["resigned"] == 1
        assert summary["score"] == 0.0

    def test_the_site_never_writes_the_local_applications_file(self, store: Scratch) -> None:
        record = m.Record(store.open)
        for write in (record.note_start, record.note_result):
            with pytest.raises(m.MatchError):
                write()


# ------------------------------------------------------------------ two starts at once

STRONG = f"minimax:nodes={quota.STRONG_NODES}"


def _member(store: Scratch) -> int:
    """A fresh signed-in person: only they may play the strongest opponent (L15)."""
    return store.user(f"m{secrets.token_hex(4)}")

# How long one step of a race may take before the test says which step hung. Far past
# anything a start costs; only a lock nobody releases gets there.
RACE_DEADLINE = 20.0


class Race:
    """Two starts on two connections, the second begun while the first is uncommitted.

    The interleaving is laid out by hand rather than left to timing. The first start runs
    to its end and is not committed, which is the moment a check with nothing between it
    and the insert has no defence at: the first row is written and nobody else can see it
    yet. The second start then runs on a connection of its own, on a thread, and the race
    waits until that thread has either finished -- nothing held it back -- or is waiting
    on the first connection. Only then does the first commit.

    So without a lock the second reads the count before the first's row is visible, on
    every run, and with one it reads the count after the first has committed. Nothing
    here sleeps to let the other side catch up; the wait is on the server's own answer to
    *is this backend waiting on that one*.
    """

    def __init__(self, store: Scratch) -> None:
        self.store = store
        self.waited = False

    def run(
        self, first: Callable[[Any], m.Row], second: Callable[[Any], m.Row]
    ) -> tuple[m.Row, Any]:
        """The first start's row and the second's outcome: its row, or the refusal."""
        earlier, later, watcher = (self._connection() for _ in range(3))
        outcome: list[Any] = []

        def attempt() -> None:
            try:
                outcome.append(second(later))
                later.commit()
            except Exception as error:
                later.rollback()
                outcome.append(error)

        try:
            seated = first(earlier)
            thread = threading.Thread(target=attempt, daemon=True)
            thread.start()
            self.waited = self._done_or_waiting(thread, watcher, later, earlier)
            earlier.commit()
            thread.join(RACE_DEADLINE)
            assert not thread.is_alive(), "the second start never finished"
        finally:
            for connection in (earlier, later, watcher):
                connection.close()
        if isinstance(outcome[0], Exception) and not isinstance(outcome[0], quota.QuotaError):
            raise outcome[0]
        return seated, outcome[0]

    def _connection(self) -> Any:
        connection = self.store.open()
        with connection.cursor() as cursor:
            cursor.execute(f"set lock_timeout = '{site_store.LOCK_TIMEOUT}'")
        connection.commit()
        return connection

    @staticmethod
    def _done_or_waiting(
        thread: threading.Thread, watcher: Any, waiting: Any, holding: Any
    ) -> bool:
        """Block until the thread has finished or `waiting` waits on `holding`. True if
        it waited."""
        watcher.autocommit = True
        waiter, holder = waiting.info.backend_pid, holding.info.backend_pid
        deadline = time.monotonic() + RACE_DEADLINE
        while thread.is_alive():
            with watcher.cursor() as cursor:
                cursor.execute(
                    "select %s::int = any(pg_blocking_pids(%s::int))", (holder, waiter)
                )
                if cursor.fetchone()[0]:
                    return True
            assert time.monotonic() < deadline, "the second start neither ended nor waited"
            thread.join(0.002)
        return False


class TestTwoStartsAtOnce:
    """Q408: two requests one short of a limit, at the same moment, seat exactly one.

    `quota.check_new_match` reads a count and `start` writes the row after it. With
    nothing between the two, each request read the count before the other's row existed,
    each found the one place free, and both were seated -- Q301's shape, between two
    requests rather than two seats. Both limits the check enforces had it, so both are
    raced here.

    The refusal is compared with the one the same start meets on its own afterwards, so
    *the usual sentence* is the whole sentence rather than a fragment of it.
    """

    @pytest.mark.parametrize(
        "seats", [["human", STRONG], [STRONG, STRONG]], ids=["a game", "a watch"]
    )
    def test_one_strong_place_short_exactly_one_is_seated(
        self, store: Scratch, seats: list[str]
    ) -> None:
        needs = sum(quota.is_strong(seat) for seat in seats)
        for _ in range(quota.STRONG_SEATS - needs):
            m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", STRONG], _member(store))
        store.connection.commit()
        ada, bob, cy = store.user("ada"), store.user("bob"), store.user("cy")

        seated, second = Race(store).run(
            lambda connection: m.start(connection, SHORT_HASH, m.SANDBOX, seats, ada),
            lambda connection: m.start(connection, SHORT_HASH, m.SANDBOX, seats, bob),
        )

        assert isinstance(seated, m.Row)
        assert isinstance(second, quota.QuotaError), "both starts were seated"
        assert quota.strong_seats(store.connection) == quota.STRONG_SEATS
        with pytest.raises(quota.QuotaError) as alone:
            m.start(store.connection, SHORT_HASH, m.SANDBOX, seats, cy)
        assert str(second) == str(alone.value)

    def test_one_match_short_of_the_day_exactly_one_is_seated(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same person twice at once, a match short. Two is the allowance so the day
        before the race is one start rather than fifty-nine."""
        monkeypatch.setattr(quota, "MATCHES_A_DAY", 2)
        ada = store.user("ada")

        def start(connection: Any) -> m.Row:
            return m.start(connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], ada)

        start(store.connection)
        store.connection.commit()

        seated, second = Race(store).run(start, start)

        assert isinstance(seated, m.Row)
        assert isinstance(second, quota.QuotaError), "both starts were seated"
        assert quota.matches_today(store.connection, ada) == quota.MATCHES_A_DAY
        with pytest.raises(quota.QuotaError) as alone:
            start(store.connection)
        assert str(second) == str(alone.value)

    def test_starts_that_share_no_limit_do_not_wait_on_each_other(
        self, store: Scratch
    ) -> None:
        """What the lock costs a normal start. Two people's games against a cheaper
        opponent share neither limit, so neither waits for the other to commit."""
        ada, bob = store.user("ada"), store.user("bob")

        race = Race(store)
        seated, second = race.run(
            lambda connection: m.start(connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], ada),
            lambda connection: m.start(connection, SHORT_HASH, m.SANDBOX, ["human", AGENT], bob),
        )

        assert isinstance(seated, m.Row) and isinstance(second, m.Row)
        assert race.waited is False

    def test_a_connection_with_no_transaction_is_refused_rather_than_raced(
        self, store: Scratch
    ) -> None:
        """The lock ends with its transaction, which on an autocommit connection is before
        the row is written, so the check refuses to run there instead of passing."""
        ada = store.user("ada")
        loose = store.open()
        loose.commit()
        loose.autocommit = True
        try:
            with pytest.raises(RuntimeError, match="autocommit"):
                quota.check_new_match(loose, ada, AGENT)
        finally:
            loose.close()


# ------------------------------------------------------ taking a strong seat back

IDLE = quota.SEAT_ACTIVE + timedelta(minutes=1)
"""Long enough without a move that a game has given its strong seat back to the count."""


def _reply(connection: Any, match_id: str) -> dict[str, Any]:
    return m.play_agent(connection, match_id)


# Each way of going on with a game, with the seats and the plies it needs first: a
# person's move needs the person to move, a reply the machine, and a take-back something
# to take back. Every one of them leaves the row live.
GOING_ON: dict[str, tuple[list[str], int, Callable[[Any, str], Any]]] = {
    "a move": (["human", STRONG], 0, _move),
    "a reply": ([STRONG, "human"], 0, _reply),
    "a take-back": (["human", STRONG], 1, m.undo),
    "a restart": (["human", STRONG], 1, m.restart),
}


class TestTakingAStrongSeatBack:
    """Q410: a game that gave its strong seat up meets the cap to take it back.

    A strong seat is held while its game is live -- unfinished and moved inside
    `quota.SEAT_ACTIVE` -- so a game that sat past that, or ended, is out of the count.
    Going on with it seats the strongest opponent again, and before this a move, a reply,
    a take-back or a restart did so with no check at all: leave a game, let the site fill
    up, come back, and the site was one over.

    Each test fills the site with members' games being played, because the cap is
    site-wide, and then asks what a game coming back may do. The cap is two so that takes
    two starts. The games are `UNRATED`'s, whose four plies leave room to move and to
    come back.
    """

    LIMIT = 2

    @pytest.fixture(autouse=True)
    def _cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(quota, "STRONG_SEATS", self.LIMIT)

    @staticmethod
    def _live(store: Scratch, count: int) -> None:
        """`count` more strong seats, each in a member's game that is being played."""
        for _ in range(count):
            m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", STRONG], _member(store))
        store.connection.commit()

    @staticmethod
    def _left(
        store: Scratch,
        seats: list[str] | None = None,
        plies: int = 0,
        user: int | None = None,
    ) -> m.Row:
        """A game against the strongest opponent, `plies` person's moves in, then left."""
        match = m.start(
            store.connection, UNRATED_HASH, m.SANDBOX, seats or ["human", STRONG],
            _member(store) if user is None else user,
        )
        for _ in range(plies):
            _move(store.connection, match.id)
        store.connection.commit()
        store.backdate(match.id, IDLE)
        return match

    @pytest.mark.parametrize("going_on", list(GOING_ON))
    def test_a_game_left_idle_takes_its_seat_back_while_one_is_free(
        self, store: Scratch, going_on: str
    ) -> None:
        """It goes on, and the seat it took is in the count: the next game to come back
        finds the site full."""
        seats, plies, act = GOING_ON[going_on]
        returning = self._left(store, seats, plies)
        later = self._left(store)
        self._live(store, self.LIMIT - 1)
        assert quota.strong_seats(store.connection) == self.LIMIT - 1

        act(store.connection, returning.id)
        store.connection.commit()

        assert quota.strong_seats(store.connection) == self.LIMIT, "it went on seatless"
        with pytest.raises(quota.QuotaError):
            _move(store.connection, later.id)

    @pytest.mark.parametrize("going_on", list(GOING_ON))
    def test_at_the_cap_a_game_left_idle_is_refused_and_kept_as_it_stands(
        self, store: Scratch, going_on: str
    ) -> None:
        """Refused before anything is written or searched, in a sentence that says what
        to do: wait, or play somebody smaller. Carrying on against a smaller player is
        not offered, because a match's seats are fixed when it starts."""
        seats, plies, act = GOING_ON[going_on]
        returning = self._left(store, seats, plies)
        before = m.session(store.connection, returning.id).snapshot()
        self._live(store, self.LIMIT)

        with pytest.raises(quota.QuotaError) as refused:
            act(store.connection, returning.id)
        store.connection.rollback()

        said = str(refused.value)
        assert "come back in a few minutes" in said, said
        assert "start a new game against a weaker opponent" in said, said
        assert "hardest" in said and "kept as it stands" in said, said
        assert STRONG not in said, "a refusal names the player, never its agent string"
        after = m.session(store.connection, returning.id).snapshot()
        assert after["history"] == before["history"], "the refusal changed the board"
        assert quota.strong_seats(store.connection) == self.LIMIT

    @pytest.mark.parametrize("act", [m.undo, m.restart], ids=["a take-back", "a restart"])
    def test_at_the_cap_a_finished_game_is_not_reopened(
        self, store: Scratch, act: Callable[[Any, str], Any]
    ) -> None:
        """A game that ended holds no seat, so opening it again is taking one."""
        match = m.start(store.connection, SHORT_HASH, m.SANDBOX, ["human", STRONG], _member(store))
        _play_out(store, match.id)
        store.connection.commit()
        assert not m.row(store.connection, match.id).running, "the short game did not end"
        self._live(store, self.LIMIT)

        with pytest.raises(quota.QuotaError, match="come back in a few minutes"):
            act(store.connection, match.id)
        store.connection.rollback()

        assert not m.row(store.connection, match.id).running, "the refusal reopened it"

    def test_a_watch_left_idle_needs_both_its_seats_free(self, store: Scratch) -> None:
        """Q301's shape on the way back: two strong seats against one free is refused,
        in the watch's own sentence."""
        watch = self._left(store, [STRONG, STRONG])
        self._live(store, self.LIMIT - 1)

        with pytest.raises(quota.QuotaError, match="only one is free") as refused:
            m.play_agent(store.connection, watch.id)
        store.connection.rollback()

        assert "weaker player on at least one side" in str(refused.value)
        assert m.row(store.connection, watch.id).plies == 0
        assert quota.strong_seats(store.connection) == self.LIMIT - 1

    def test_a_game_being_played_goes_on_at_the_cap_and_is_never_counted_again(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its seat is already in the count, so its moves take neither the lock nor the
        count. The idle game refused beside it is what shows the site really is full."""
        played = m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", STRONG], _member(store))
        returning = self._left(store)
        self._live(store, self.LIMIT - 1)
        assert quota.strong_seats(store.connection) == self.LIMIT
        with pytest.raises(quota.QuotaError):
            _move(store.connection, returning.id)
        store.connection.rollback()

        def asked(*_args: Any) -> Any:
            raise AssertionError("a game being played was checked again")

        monkeypatch.setattr(quota, "strong_seats", asked)
        monkeypatch.setattr(quota, "_hold", asked)
        _move(store.connection, played.id)
        m.play_agent(store.connection, played.id)
        m.undo(store.connection, played.id)
        _move(store.connection, played.id)
        m.restart(store.connection, played.id)
        store.connection.commit()

    def test_a_game_in_its_last_minute_is_read_again_under_the_lock_and_goes_on(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Near the end of `SEAT_ACTIVE` a move is not waved past, because a slow request
        could commit after its seat had left somebody else's count. Read again under the
        lock it is still live, so it goes on without being counted twice."""
        played = m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", STRONG], _member(store))
        store.connection.commit()
        store.backdate(played.id, quota.SEAT_ACTIVE - timedelta(seconds=30))
        self._live(store, self.LIMIT - 1)
        assert quota.strong_seats(store.connection) == self.LIMIT
        held: list[int] = []
        hold = quota._hold
        monkeypatch.setattr(
            quota, "_hold", lambda connection, key: (held.append(key), hold(connection, key))
        )

        _move(store.connection, played.id)
        store.connection.commit()

        assert held == [quota.WHOLE_SITE]
        assert quota.strong_seats(store.connection) == self.LIMIT

    def test_taking_back_to_the_opening_keeps_the_seat(self, store: Scratch) -> None:
        """A take-back is somebody at the board. A game started long ago and moved a
        moment ago, taken back to its opening, is still being played; it used to hand
        its seat back there and meet the cap on its next move as though it had been left.
        """
        played = m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", STRONG], _member(store))
        _move(store.connection, played.id)
        with store.connection.cursor() as cursor:
            cursor.execute(
                "update matches set started = started - %s where id = %s", (IDLE, played.id)
            )
        store.connection.commit()
        self._live(store, self.LIMIT - 1)

        m.undo(store.connection, played.id)
        store.connection.commit()

        assert quota.strong_seats(store.connection) == self.LIMIT, "the take-back left it"
        _move(store.connection, played.id)

    def test_going_on_is_not_another_match_for_the_day(
        self, store: Scratch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The daily allowance counts matches started, and a game carried on or started
        again in place is the one match it already counted. So a person who has used the
        day can still come back to the game they used it on."""
        monkeypatch.setattr(quota, "MATCHES_A_DAY", 1)
        ada = store.user("ada")
        returning = self._left(store, user=ada, plies=1)
        with pytest.raises(quota.QuotaError, match="which is the limit of 1"):
            m.start(store.connection, UNRATED_HASH, m.SANDBOX, ["human", AGENT], ada)
        store.connection.rollback()

        m.restart(store.connection, returning.id)
        _move(store.connection, returning.id)
        store.connection.commit()

        assert quota.matches_today(store.connection, ada) == 1
        assert quota.strong_seats(store.connection) == 1

    def test_the_person_reads_why_where_they_moved(self, store: Scratch, app: SiteApp) -> None:
        """Through the route the board presses. The server answers a `QuotaError` with a
        400 carrying the sentence, and the board prints it beside its status line."""
        returning = self._left(store)
        self._live(store, self.LIMIT)
        snapshot = m.session(store.connection, returning.id).snapshot()
        app.connect = store.open
        app.viewer = None

        with pytest.raises(quota.QuotaError, match="come back in a few minutes"):
            app.post(
                f"/api/session/{returning.id}/move", {"move": snapshot["moves"][0]["move"]}
            )
        assert m.row(store.connection, returning.id).plies == 0

    @pytest.mark.parametrize("order", ["coming back first", "starting first"])
    def test_a_game_coming_back_and_a_start_one_place_short_seat_exactly_one(
        self, store: Scratch, order: str
    ) -> None:
        """Q408's race, with a game coming back on one side of it. Both go through the
        site's lock, so whichever is second counts the first's seat."""
        returning = self._left(store)
        self._live(store, self.LIMIT - 1)
        ada = store.user("ada")

        def back(connection: Any) -> m.Row:
            _move(connection, returning.id)
            return m.row(connection, returning.id)

        def begin(connection: Any) -> m.Row:
            return m.start(connection, UNRATED_HASH, m.SANDBOX, ["human", STRONG], ada)

        first, second = (back, begin) if order == "coming back first" else (begin, back)
        seated, refused = Race(store).run(first, second)

        assert isinstance(seated, m.Row)
        assert isinstance(refused, quota.QuotaError), "both took the last place"
        assert quota.strong_seats(store.connection) == self.LIMIT

    def test_one_game_taken_back_twice_at_once_is_one_seat(self, store: Scratch) -> None:
        """A double click on one idle game, one place short. The second press waits for
        the first, then finds the seat already its own game's rather than counting that
        game against itself."""
        returning = self._left(store, plies=1)
        self._live(store, self.LIMIT - 1)

        def again(connection: Any) -> m.Row:
            m.restart(connection, returning.id)
            return m.row(connection, returning.id)

        seated, second = Race(store).run(again, again)

        assert isinstance(seated, m.Row)
        assert isinstance(second, m.Row), "the second press was refused its own seat"
        assert quota.strong_seats(store.connection) == self.LIMIT


# ---------------------------------------------------------------------- through the app


@pytest.fixture(scope="module")
def app() -> SiteApp:
    return SiteApp(library.RULESET_DIR, library.RUNS_DIR)


class TestTheApp:
    def test_the_session_routes_play_into_the_store(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """`POST /api/session` and its actions, end to end, with a row behind them."""
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = user
        try:
            snapshot = app.post(
                "/api/session",
                {"hash": spec_hash(DULL), "seats": ["human", AGENT], "mode": m.LEADERBOARD},
            )
            played = app.post(
                f"/api/session/{snapshot['id']}/move", {"move": snapshot["moves"][0]["move"]}
            )
            app.post(f"/api/session/{snapshot['id']}/agent", {})
            reopened = app.get(f"/api/session/{snapshot['id']}")
        finally:
            app.viewer = None

        assert played["history"], "the move did not land"
        assert len(reopened["history"]) == 2
        found = m.row(store.connection, snapshot["id"])
        assert found.mode == m.LEADERBOARD and found.plies == 2 and found.top is True

    def test_a_session_route_draws_a_fresh_seed_and_keeps_a_named_one(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """No page names a seed, so each start draws one and writes it in the row (Q987);
        a request that names one, 0 included, gets exactly that, and junk is refused."""
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = user
        body = {"hash": spec_hash(DULL), "seats": ["human", AGENT]}
        try:
            drawn = [app.post("/api/session", body)["id"] for _ in range(3)]
            named = [app.post("/api/session", {**body, "seed": seed})["id"] for seed in (0, 5)]
            with pytest.raises(ValueError, match="whole number"):
                app.post("/api/session", {**body, "seed": "fresh"})
        finally:
            app.viewer = None

        seeds = [_seed_of(store.connection, match_id) for match_id in drawn]
        assert len(set(seeds)) == len(seeds)
        assert [_seed_of(store.connection, match_id) for match_id in named] == [0, 5]

    def test_undo_is_refused_on_the_leaderboard_and_offered_in_sandbox(
        self, store: Scratch, app: SiteApp
    ) -> None:
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = user
        try:
            rated = app.post(
                "/api/session",
                {"hash": spec_hash(DULL), "seats": ["human", AGENT], "mode": m.LEADERBOARD},
            )
            app.post(f"/api/session/{rated['id']}/move", {"move": rated["moves"][0]["move"]})
            with pytest.raises(ValueError, match="practice"):
                app.post(f"/api/session/{rated['id']}/undo", {})

            loose = app.post(
                "/api/session", {"hash": spec_hash(DULL), "seats": ["human", AGENT]}
            )
            app.post(f"/api/session/{loose['id']}/move", {"move": loose["moves"][0]["move"]})
            assert app.post(f"/api/session/{loose['id']}/undo", {})["history"] == []
        finally:
            app.viewer = None

    def test_the_default_mode_is_the_one_that_counts_for_nothing(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """A result that counts is something a player chooses, never something they get
        by leaving a field out."""
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = user
        try:
            snapshot = app.post(
                "/api/session", {"hash": spec_hash(DULL), "seats": ["human", AGENT]}
            )
        finally:
            app.viewer = None
        assert m.row(store.connection, snapshot["id"]).mode == m.SANDBOX

    def test_a_visitor_plays_and_is_refused_only_the_game_that_counts(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """What a person with no account gets: the sandbox, and a sentence about the rest.

        Task 1.3's cookie is what fills `viewer` in. Until it does, and for everybody who
        never signs in afterwards, a game is a sandbox row with nobody in their seat.
        """
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = None

        snapshot = app.post(
            "/api/session", {"hash": spec_hash(DULL), "seats": ["human", AGENT]}
        )
        played = app.post(
            f"/api/session/{snapshot['id']}/move", {"move": snapshot["moves"][0]["move"]}
        )
        assert played["history"], "the visitor's move did not land"

        with pytest.raises(m.MatchError, match="sign in"):
            app.post(
                "/api/session",
                {
                    "hash": spec_hash(DULL),
                    "seats": ["human", AGENT],
                    "mode": m.LEADERBOARD,
                },
            )

        found = m.row(store.connection, snapshot["id"])
        assert found.mode == m.SANDBOX and found.seat0 is None
        assert store.rows() == 1, "the refused game left a row behind"

    def test_the_game_page_offers_a_visitor_sandbox_alone(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """`signedIn` is what the mode control reads, so the menu and the server agree."""
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = None
        assert app.get(f"/api/game/{spec_hash(DULL)}")["standing"]["signedIn"] is False
        app.viewer = store.user("ada")
        try:
            assert app.get(f"/api/game/{spec_hash(DULL)}")["standing"]["signedIn"] is True
        finally:
            app.viewer = None

    def test_a_visitor_plays_a_draft_and_no_row_is_written(self, app: SiteApp) -> None:
        """The designer's play-a-draft, which is a session in memory and nothing else.

        A draft has no `games` row for a match to point at, and §2's own note on the
        route says it writes nothing -- so the store is handed a connection that refuses
        to open, and the game is played anyway.
        """

        def refuse() -> Any:
            raise AssertionError("playing a draft must not touch the store")

        app.connect = refuse
        app.viewer = None
        snapshot = app.post("/api/design/play", {"spec": DULL, "seats": ["human", AGENT]})
        played = app.post(
            f"/api/session/{snapshot['id']}/move", {"move": snapshot["moves"][0]["move"]}
        )
        assert played["history"], "the draft did not take a move"
        assert app.get(f"/api/session/{snapshot['id']}")["id"] == snapshot["id"]

    def test_a_visitor_watches_two_computer_players(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """Q262, through the route the board presses: two agent seats, nobody signed in.

        The page never picks a move. Both seats answer through `/agent`, which is the
        core's own player, and the row that comes out is nobody's game.
        """
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = None

        snapshot = app.post(
            "/api/session", {"hash": spec_hash(DULL), "seats": [AGENT, AGENT]}
        )
        assert snapshot["awaitingAgent"] is True, "nobody was asked to move"
        app.post(f"/api/session/{snapshot['id']}/agent", {})
        played = app.post(f"/api/session/{snapshot['id']}/agent", {})
        assert len(played["history"]) == 2

        found = m.row(store.connection, snapshot["id"])
        assert found.mode == m.SANDBOX and found.is_computer(0) and found.is_computer(1)
        assert found.top is False and found.rematch_of is None
        assert m.leaderboard_matches(store.connection, running=True) == ()

    def test_the_route_records_a_rematch_and_records_nothing_without_one(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """`POST /api/session` carries the match id the board was showing, or nothing.

        The same route serves the game page's start button, which sends no such field,
        so the two rows differ in exactly the column C1 added.
        """
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = user
        try:
            first = app.post(
                "/api/session", {"hash": spec_hash(DULL), "seats": ["human", AGENT]}
            )
            again = app.post(
                "/api/session",
                {
                    "hash": spec_hash(DULL),
                    "seats": ["human", AGENT],
                    "rematch_of": first["id"],
                },
            )
        finally:
            app.viewer = None

        assert m.row(store.connection, first["id"]).rematch_of is None
        assert m.row(store.connection, again["id"]).rematch_of == first["id"]

    def test_the_mode_travels_with_the_position(self, store: Scratch, app: SiteApp) -> None:
        """B1 of `docs/archive/plan-ux.md`. The board draws a *rated* mark, hides the take-back
        and offers a resignation off this field, so it has to be on every snapshot the
        page is handed and not only on the first.
        """
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = user
        try:
            rated = app.post(
                "/api/session",
                {"hash": spec_hash(DULL), "seats": ["human", AGENT], "mode": m.LEADERBOARD},
            )
            moved = app.post(
                f"/api/session/{rated['id']}/move", {"move": rated["moves"][0]["move"]}
            )
            reopened = app.get(f"/api/session/{rated['id']}")
            loose = app.post(
                "/api/session", {"hash": spec_hash(DULL), "seats": ["human", AGENT]}
            )
        finally:
            app.viewer = None

        assert rated["mode"] == m.LEADERBOARD
        assert moved["mode"] == m.LEADERBOARD, "a move must not lose it"
        assert reopened["mode"] == m.LEADERBOARD, "and neither must a reload"
        assert loose["mode"] == m.SANDBOX

    def test_a_resigned_game_reads_as_over_on_the_board_it_was_resigned_from(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """Giving a game up ends it in a position that is still playable.

        So the engine has no answer about it and the row is what says what happened. A
        board that did not show it offered a resigned game's moves and went on asking the
        opponent to reply, which the store then refused one call later.
        """
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = user
        try:
            rated = app.post(
                "/api/session",
                {"hash": spec_hash(DULL), "seats": [AGENT, "human"], "mode": m.LEADERBOARD},
            )
            given_up = app.post(f"/api/session/{rated['id']}/resign", {})
            reopened = app.get(f"/api/session/{rated['id']}")
        finally:
            app.viewer = None

        found = m.row(store.connection, rated["id"])
        assert found.reason == "resigned" and found.left is True
        # Seat 1 is the person, so seat 0 is the winner and the page is told which.
        assert given_up["outcome"] == {"winner": 0, "reason": "resigned"}
        assert given_up["moves"] == [], "a game that is over offers no move"
        assert given_up["awaitingAgent"] is False, "and asks nobody to make one"
        assert reopened["outcome"] == given_up["outcome"], "the row is what remembers it"

    def test_a_sandbox_game_can_be_resigned_and_moves_no_rating(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """Q578, the first-visit study's F12: a sandbox game had take back and new game
        and no way to end it with a result. Its board now offers the resignation a rated
        game has, through the same call; the row records the finish and the table, which
        counts no sandbox row, writes nothing."""
        user = store.user("ada")
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()

        def rating_rows() -> int:
            with store.connection.cursor() as cursor:
                cursor.execute("select count(*) from ratings")
                return int(cursor.fetchone()[0])

        before = rating_rows()
        app.connect = store.open
        app.viewer = user
        try:
            loose = app.post(
                "/api/session",
                {"hash": spec_hash(DULL), "seats": ["human", AGENT], "mode": m.SANDBOX},
            )
            given_up = app.post(f"/api/session/{loose['id']}/resign", {})
        finally:
            app.viewer = None

        found = m.row(store.connection, loose["id"])
        assert found.mode == m.SANDBOX
        assert found.reason == "resigned" and found.result == 0.0
        # Seat 0 is the person, so seat 1 wins, and the board is told the game is over.
        assert given_up["outcome"] == {"winner": 1, "reason": "resigned"}
        assert given_up["moves"] == []
        assert rating_rows() == before, "a sandbox finish rates nobody"

    def test_a_visitors_sandbox_game_can_be_resigned_too(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """Nobody is signed in, so there is no id to name; the person's seat is the one
        seat a computer player does not hold."""
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = None
        loose = app.post("/api/session", {"hash": spec_hash(DULL), "seats": [AGENT, "human"]})
        given_up = app.post(f"/api/session/{loose['id']}/resign", {})
        assert given_up["outcome"] == {"winner": 0, "reason": "resigned"}

    def test_a_take_back_is_a_game_going_on_again(self, store: Scratch, app: SiteApp) -> None:
        """The other side of reading a finish off the row: a sandbox game that ended and
        was taken back is running, and the board must not still say who won."""
        app.connect = store.open
        app.viewer = None

        loose = app.post("/api/session", {"hash": SHORT_HASH, "seats": ["human", AGENT]})
        app.post(f"/api/session/{loose['id']}/move", {"move": loose["moves"][0]["move"]})
        over = app.post(f"/api/session/{loose['id']}/agent", {})
        assert over["outcome"] is not None, "the short game did not end"

        again = app.post(f"/api/session/{loose['id']}/undo", {})
        assert again["outcome"] is None
        assert again["moves"], "and it can be played on"

    def test_a_watch_says_who_is_playing_and_a_game_with_a_person_in_it_does_not(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """Q265's sentence, on the site's watch as well as the designer's.

        Weak play by a weak player looks exactly like a broken game, and a site watch
        raises that question as much as a draft's does. The words are the picker's own
        labels, so the page holds no second opinion about what a seat is.
        """
        store.game(spec_hash(DULL), DULL, LADDER)
        store.connection.commit()
        app.connect = store.open
        app.viewer = None

        watched = app.post("/api/session", {"hash": spec_hash(DULL), "seats": [AGENT, AGENT]})
        played = app.post("/api/session", {"hash": spec_hash(DULL), "seats": ["human", AGENT]})

        assert "Both sides are played by" in watched["watching"]
        assert AGENT in watched["watching"] or "400" in watched["watching"]
        assert played["watching"] == ""

    def test_a_reload_serves_the_depths_the_replies_did(
        self, store: Scratch, app: SiteApp
    ) -> None:
        """Q424 through the routes the board presses: the reply says how far it looked,
        and `GET /api/session/<id>` -- a reload -- says the same beside the same move."""
        app.connect = store.open
        app.viewer = None

        started = app.post("/api/session", {"hash": UNRATED_HASH, "seats": ["human", AGENT]})
        session = f"/api/session/{started['id']}"
        app.post(f"{session}/move", {"move": started["moves"][0]["move"]})
        replied = app.post(f"{session}/agent", {})
        reopened = app.get(session)

        assert isinstance(replied["history"][1]["depth"], int)
        assert _depths(reopened) == _depths(replied) == [None, replied["history"][1]["depth"]]
