"""A person's finished games are kept move by move, and replay to what the store said.

Q502a, `docs/reference/general-player.md` §4.2, for E6 (*where a person wins*). The
acceptance is one sentence per class below: a finish a person sat at is kept with its
whole move list and result, whatever the sandbox's undo and restart do to the match row
afterwards; replaying that list through the engine reaches the same position and the
same verdict; what is kept names nobody; and a reader gets the games out with
`chess-variant-sandbox site games`.

The game is `dull` cut to one grunt a side on one file with `stalemate: loss`: every
move is forced and the second player is stalemated on ply 3, so the person in seat 0
wins whatever the computer player does, in three calls. Nothing here works out a legal
move or a winner; the engine answers and the store records what it said.

Everything needs a real Postgres and skips with a reason without one;
`tests/test_site.py` has the recipe for a throwaway cluster.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_store

from chess_variant_sandbox import cli
from chess_variant_sandbox.site import db, finishes
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.human import HumanGame, HumanRecord

DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))
AGENT = "minimax:nodes=400"


def _won_by_seat_0() -> dict[str, Any]:
    spec = json.loads(json.dumps(DULL))
    spec["name"] = "forced"
    spec["rules"]["max_plies"] = 8
    spec["rules"]["stalemate"] = "loss"
    spec["setup"]["placements"] = [{"piece": "G", "squares": ["a1"]}]
    return spec


FORCED = _won_by_seat_0()
FORCED_HASH = spec_hash(FORCED)

LADDER = {
    "ruleset_hash": FORCED_HASH,
    "standings": [
        {"name": AGENT, "rating": 300.0, "games": 24},
        {"name": "greedy", "rating": 0.0, "games": 24},
        {"name": "random", "rating": -300.0, "games": 24},
    ],
}


class Scratch:
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

    def one(self, sql: str, *values: Any) -> Any:
        with self.connection.cursor() as cursor:
            cursor.execute(sql, values)
            return cursor.fetchone()


@contextlib.contextmanager
def _scratch() -> Iterator[Scratch]:
    url, name = site_store.emptied("no database to keep a finished game in")
    scratch = Scratch(url, name)
    scratch.connection.commit()
    try:
        yield scratch
    finally:
        scratch.connection.close()


@pytest.fixture
def store() -> Iterator[Scratch]:
    """A store holding the forced game, rated, so both modes can be played on it."""
    with _scratch() as scratch:
        with scratch.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (FORCED_HASH, json.dumps(FORCED), FORCED["name"], "corpus"),
            )
        r.seed(scratch.connection, FORCED_HASH, LADDER)
        scratch.connection.commit()
        yield scratch


def _play_out(connection: Any, match_id: str) -> dict[str, Any]:
    """To the end: the person's first offered move on their turn, the core's reply on the
    machine's. In this game every move is forced, so which is offered first is moot."""
    snapshot = m.session(connection, match_id).snapshot()
    while snapshot["outcome"] is None:
        if snapshot["awaitingAgent"]:
            snapshot = m.play_agent(connection, match_id)
        else:
            snapshot = m.play(connection, match_id, snapshot["moves"][0]["move"])
    return snapshot


def _position(state: Any) -> tuple[Any, ...]:
    return (
        list(state.board), list(state.colour), state.side_to_move, state.ply,
        [list(state.hands[0]), list(state.hands[1])],
    )


# ------------------------------------------------------------- a finish is kept


class TestAFinishIsKept:
    def test_a_leaderboard_win_is_kept_with_every_move_and_its_result(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        end = _play_out(store.connection, match.id)
        store.connection.commit()

        assert end["outcome"]["winner"] == 0, "the forced game is the first seat's"
        [kept] = finishes.recorded(store.connection)
        stored_moves = store.one("select moves from matches where id = %s", match.id)[0]
        assert kept.match == match.id and kept.ordinal == 1
        assert list(kept.moves) == stored_moves and len(kept.moves) == 3
        assert kept.mode == m.LEADERBOARD and kept.person_seat == 0
        assert kept.opponent == AGENT and kept.signed_in
        assert kept.result == 1.0 and kept.winner == 0 and kept.person_won
        assert kept.reason == end["outcome"]["reason"] and kept.takebacks == 0

    def test_a_sandbox_win_survives_the_restart_that_follows_it(self, store: Scratch) -> None:
        """The hole this closes: a restart empties the match row's moves and verdict."""
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _play_out(store.connection, match.id)
        m.restart(store.connection, match.id)
        store.connection.commit()

        assert m.row(store.connection, match.id).plies == 0
        assert m.row(store.connection, match.id).result is None
        [kept] = finishes.recorded(store.connection)
        assert len(kept.moves) == 3 and kept.person_won

    def test_a_game_taken_back_past_its_end_and_won_again_is_two_finishes(
        self, store: Scratch
    ) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _play_out(store.connection, match.id)
        m.undo(store.connection, match.id)
        _play_out(store.connection, match.id)
        store.connection.commit()

        first, second = finishes.recorded(store.connection)
        assert (first.ordinal, second.ordinal) == (1, 2)
        assert (first.takebacks, second.takebacks) == (0, 1), "the undo is not on the win"
        assert first.moves == second.moves

    def test_a_restart_puts_the_take_backs_back_to_nothing(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        snapshot = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, snapshot["moves"][0]["move"])
        m.undo(store.connection, match.id)
        m.restart(store.connection, match.id)
        _play_out(store.connection, match.id)

        [kept] = finishes.recorded(store.connection)
        assert kept.takebacks == 0

    def test_a_resignation_is_kept_as_the_moves_it_was_resigned_at(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        snapshot = m.session(store.connection, match.id).snapshot()
        m.play(store.connection, match.id, snapshot["moves"][0]["move"])
        m.resign(store.connection, match.id)

        [kept] = finishes.recorded(store.connection)
        assert len(kept.moves) == 1 and kept.reason == m.RESIGNED
        assert kept.result == 0.0 and kept.winner == 1
        assert finishes.replay(FORCED, kept.moves).outcome is None, (
            "a resignation is an ending the engine cannot see; the row is what says it"
        )

    def test_a_game_nobody_sat_at_is_not_kept(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "greedy"])
        _play_out(store.connection, match.id)

        assert m.row(store.connection, match.id).result is not None
        assert finishes.recorded(store.connection) == []
        assert store.one("select count(*) from finishes")[0] == 0

    def test_the_session_routes_keep_the_finish(self, store: Scratch) -> None:
        """Through `POST /api/session` and its actions, as the board plays it."""
        app = SiteApp(library.RULESET_DIR, library.RUNS_DIR)
        app.connect = store.open
        snapshot = app.post("/api/session", {"hash": FORCED_HASH, "seats": ["human", AGENT]})
        while snapshot["outcome"] is None:
            action = "agent" if snapshot["awaitingAgent"] else "move"
            body = {} if action == "agent" else {"move": snapshot["moves"][0]["move"]}
            snapshot = app.post(f"/api/session/{snapshot['id']}/{action}", body)
        app.post(f"/api/session/{snapshot['id']}/restart", {})

        [kept] = finishes.recorded(store.connection)
        assert kept.match == snapshot["id"] and kept.person_won and not kept.signed_in


# --------------------------------------------------- the moves replay to the verdict


class TestAFinishReplays:
    def test_a_kept_win_replays_to_the_same_position_and_result(
        self, store: Scratch
    ) -> None:
        """The acceptance: the list the store kept, put to the engine move by move from
        the opening, reaches the position the match ended in and the verdict it was
        given -- read back through a connection that never saw the game."""
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _play_out(store.connection, match.id)
        store.connection.commit()
        ended = m.session(store.connection, match.id)

        second = store.open()
        try:
            [kept] = finishes.recorded(second)
            spec = store.one("select spec from games where hash = %s", kept.game)[0]
        finally:
            second.close()
        replayed = finishes.replay(spec, kept.moves)

        assert _position(replayed.state) == _position(ended.state)
        assert replayed.outcome is not None
        assert replayed.outcome.winner == kept.winner == 0
        assert replayed.outcome.reason == kept.reason
        assert len(replayed.positions) == len(kept.moves) + 1

    def test_a_list_the_engine_refuses_is_refused_by_ply(self) -> None:
        opening = finishes.replay(FORCED, []).state
        with pytest.raises(ValueError, match="ply 0"):
            finishes.replay(FORCED, [-1])
        assert opening.ply == 0

    def test_moves_past_the_end_are_refused(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _play_out(store.connection, match.id)
        [kept] = finishes.recorded(store.connection)
        with pytest.raises(ValueError, match="already over"):
            finishes.replay(FORCED, [*kept.moves, kept.moves[-1]])


# ------------------------------------------------------------- what is kept, of whom


class TestWhatIsKeptNamesNobody:
    def test_a_finish_row_holds_the_game_and_nothing_about_the_player(self) -> None:
        """The table's columns, all of them: the match it belongs to and the game itself.
        Who sat where stays on the match row, the one place leaving empties."""
        with _scratch() as scratch, scratch.connection.cursor() as cursor:
            cursor.execute(
                "select column_name from information_schema.columns"
                " where table_schema = %s and table_name = 'finishes'",
                (scratch.schema,),
            )
            columns = {found[0] for found in cursor.fetchall()}
        assert columns == {
            "match", "ordinal", "moves", "result", "reason", "takebacks", "finished",
        }

    def test_a_visitors_finish_is_the_game_and_no_one(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _play_out(store.connection, match.id)

        [kept] = finishes.recorded(store.connection)
        assert not kept.signed_in
        assert m.row(store.connection, match.id).seat0 is None
        line = kept.as_dict()
        assert not {"user", "handle", "email", "seat0_user", "seat1_user"} & set(line)

    def test_one_persons_games_are_found_by_handle_and_never_a_visitors(
        self, store: Scratch
    ) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        for user in (ada, bea, None):
            match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT], user)
            _play_out(store.connection, match.id)

        assert len(finishes.recorded(store.connection)) == 3
        assert len(finishes.recorded(store.connection, handle="Ada")) == 1
        assert finishes.recorded(store.connection, handle="nobody") == []

    def test_the_filters_narrow_by_game_mode_and_winner(self, store: Scratch) -> None:
        ada = store.user("ada")
        won = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _play_out(store.connection, won.id)
        lost = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "human"], ada)
        _play_out(store.connection, lost.id)

        everything = finishes.recorded(store.connection, game=FORCED_HASH[:8])
        assert len(everything) == 2
        [only_won] = finishes.recorded(store.connection, won=True)
        assert only_won.match == won.id
        [sandbox] = finishes.recorded(store.connection, mode=m.SANDBOX)
        assert sandbox.match == lost.id and sandbox.person_seat == 1 and sandbox.result == 0.0
        assert finishes.recorded(store.connection, game="ffff") == []


# ------------------------------------------------------------------ getting them out


class TestTheCommand:
    def test_it_prints_one_line_per_finish_that_replays(
        self,
        store: Scratch,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _play_out(store.connection, match.id)
        visitor = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _play_out(store.connection, visitor.id)
        store.connection.commit()
        monkeypatch.setattr(db, "database_url", lambda: store.url)
        monkeypatch.setattr(db, "connect", lambda url: store.open())

        assert cli.main(["site", "games", "--player", "ada", "--won"]) == 0
        lines = capsys.readouterr().out.splitlines()

        [line] = [json.loads(text) for text in lines]
        assert line["match"] == match.id and "ada" not in json.dumps(line)
        replayed = finishes.replay(FORCED, line["moves"])
        assert replayed.outcome is not None and replayed.outcome.winner == line["winner"]


# ------------------------------------------- the anchor's rows carry the moves too


class TestTheAnchorRowsCarryMoves:
    def test_the_sites_human_games_carry_a_finished_matchs_moves(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _play_out(store.connection, match.id)
        [game] = m.human_games(store.connection, FORCED_HASH)
        [kept] = finishes.recorded(store.connection)
        assert game.moves == kept.moves and game.result == 1.0

    def test_the_local_apps_record_keeps_the_moves_and_reads_old_rows(
        self, tmp_path: Path
    ) -> None:
        """`web/human.py`'s file, which needs no database: a row written now carries the
        moves; a row from before carries none and still loads."""
        record = HumanRecord(tmp_path)
        record.note_start("s1", FORCED_HASH, "forced", AGENT, 0, 7)
        record.note_result("s1", FORCED_HASH, "forced", AGENT, 0, 7, 3, 0, "stalemate_loss",
                           moves=[4, 9, 2])
        record.note_result("s1", FORCED_HASH, "forced", AGENT, 0, 7, 5, 1, "later",
                           moves=[1])
        old = HumanGame("s0", FORCED_HASH, "forced", AGENT, 0, 1, 0.0).as_dict()
        del old["moves"]
        with open(record.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(old) + "\n")

        by_session = {game.session: game for game in record.games()}
        assert by_session["s1"].moves == (4, 9, 2), "the first finish's moves, once"
        assert by_session["s0"].moves == ()
