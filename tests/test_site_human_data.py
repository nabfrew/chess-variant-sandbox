"""What people's games record for a baseline: think time, take-backs, the rating at the time.

Q767, H1 to H3 of `docs/reference/human-data.md` §1.2 (`0032_human_data.sql`). One sentence per
class below: a person's move against a computer player keeps how long they took, measured
on the server, null over half an hour and on every computer move, and a game between two
people or two computer players keeps none (H1); each take-back and restart appends what it
removed (H2); a practice game against a computer player keeps each side's rating as the
table stood, and moves nothing (H3); a finish copies all three and the export carries
them; and no page's payload does, while the privacy page names them.

The game is `test_site_finishes.py`'s forced one: one grunt a side on one file, every move
forced, the person in seat 0 wins on ply 3. Gaps are made by moving the row's `started`
or `moved` back on the database's own clock, which is the clock the think is read on;
inside one transaction `now()` stands still, so the gap is exact.

Everything needs a real Postgres and skips with a reason without one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from test_site_finishes import (
    AGENT,
    FORCED,
    FORCED_HASH,
    LADDER,
    Scratch,
    _play_out,
    _scratch,
)

from chess_variant_sandbox import cli
from chess_variant_sandbox.site import correspondence, db, finishes
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.pages import privacy
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.web import library


@pytest.fixture
def store() -> Iterator[Scratch]:
    """`test_site_finishes.py`'s store: the forced game, its table seeded from the ladder."""
    with _scratch() as scratch:
        with scratch.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (FORCED_HASH, json.dumps(FORCED), FORCED["name"], "corpus"),
            )
        r.seed(scratch.connection, FORCED_HASH, LADDER)
        scratch.connection.commit()
        yield scratch


def _first(connection: Any, match_id: str) -> int:
    return int(m.session(connection, match_id).snapshot()["moves"][0]["move"])


def _person_moves(connection: Any, match_id: str) -> dict[str, Any]:
    return m.play(connection, match_id, _first(connection, match_id))


def _since(store: Scratch, match_id: str, seconds: float) -> None:
    """Make the last hand-over to the person `seconds` ago, on the database's clock."""
    with store.connection.cursor() as cursor:
        cursor.execute(
            "update matches set started = now() - make_interval(secs => %s),"
            " moved = case when moved is null then null"
            " else now() - make_interval(secs => %s) end where id = %s",
            (seconds, seconds, match_id),
        )


def _column(store: Scratch, column: str, match_id: str) -> Any:
    return store.one(f"select {column} from matches where id = %s", match_id)[0]


# --------------------------------------------------------------------- H1: think time


class TestThinkTime:
    def test_each_person_move_keeps_its_think_and_each_reply_none(
        self, store: Scratch
    ) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _since(store, match.id, 12)
        _person_moves(store.connection, match.id)
        assert _column(store, "thinks", match.id) == [12000]

        m.play_agent(store.connection, match.id)
        assert _column(store, "thinks", match.id) == [12000, None]

        _since(store, match.id, 7.25)
        _person_moves(store.connection, match.id)
        assert _column(store, "thinks", match.id) == [12000, None, 7250]
        assert m.row(store.connection, match.id).finished is not None

        [kept] = finishes.recorded(store.connection)
        assert kept.thinks == (12000, None, 7250)

    def test_a_gap_over_half_an_hour_is_away_not_thinking(self, store: Scratch) -> None:
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _since(store, match.id, 30 * 60 + 1)
        _person_moves(store.connection, match.id)
        m.play_agent(store.connection, match.id)
        _since(store, match.id, 30 * 60)
        _person_moves(store.connection, match.id)

        assert m.THINK_CAP.total_seconds() == 30 * 60, "the plan's cap, fixed"
        assert _column(store, "thinks", match.id) == [None, None, 30 * 60 * 1000]

    def test_a_computer_moving_first_has_no_think(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "human"])
        m.play_agent(store.connection, match.id)
        _since(store, match.id, 3)
        _person_moves(store.connection, match.id)
        assert _column(store, "thinks", match.id) == [None, 3000]

    def test_a_row_from_before_the_column_gains_it_with_nulls(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _person_moves(store.connection, match.id)
        m.play_agent(store.connection, match.id)
        with store.connection.cursor() as cursor:
            cursor.execute("update matches set thinks = null where id = %s", (match.id,))
        _since(store, match.id, 2)
        _person_moves(store.connection, match.id)
        assert _column(store, "thinks", match.id) == [None, None, 2000]

    def test_a_game_nobody_sits_at_keeps_none(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "greedy"])
        _play_out(store.connection, match.id)
        assert _column(store, "thinks", match.id) is None

    def test_a_game_between_two_people_keeps_none(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        match = correspondence.start(store.connection, FORCED_HASH, [ada, bea])
        correspondence.move(store.connection, match.id, ada, _first(store.connection, match.id))
        correspondence.move(store.connection, match.id, bea, _first(store.connection, match.id))
        assert _column(store, "thinks", match.id) is None
        assert len(_column(store, "times", match.id)) == 2, "they keep `times` instead"

    def test_the_store_refuses_thinks_that_do_not_run_beside_the_moves(
        self, store: Scratch
    ) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        store.connection.commit()
        with (
            pytest.raises(Exception, match="matches_thinks_beside_moves"),
            store.connection.cursor() as cursor,
        ):
            cursor.execute(
                "update matches set thinks = '[1]'::jsonb where id = %s", (match.id,)
            )
        store.connection.rollback()


# -------------------------------------------------------------- H2: what was undone


class TestWhatATakeBackUndid:
    def test_a_take_back_appends_the_ply_and_the_moves_it_removed(
        self, store: Scratch
    ) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _since(store, match.id, 5)
        _person_moves(store.connection, match.id)
        m.play_agent(store.connection, match.id)
        played = list(_column(store, "moves", match.id))
        m.undo(store.connection, match.id)

        assert _column(store, "undone", match.id) == [
            {"ply": 0, "moves": played, "thinks": [5000, None]}
        ]
        assert _column(store, "thinks", match.id) == []
        assert _column(store, "moves", match.id) == []

        _play_out(store.connection, match.id)
        [kept] = finishes.recorded(store.connection)
        assert kept.takebacks == 1
        assert kept.undone == ({"ply": 0, "moves": played, "thinks": [5000, None]},)

    def test_a_take_back_past_the_end_appends_the_moves_after_the_ply(
        self, store: Scratch
    ) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _play_out(store.connection, match.id)
        played = list(_column(store, "moves", match.id))
        m.undo(store.connection, match.id)
        [entry] = _column(store, "undone", match.id)
        assert entry["ply"] == 2 and entry["moves"] == played[2:]
        assert _column(store, "moves", match.id) == played[:2]

    def test_a_restart_appends_the_whole_list_and_says_so(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        _person_moves(store.connection, match.id)
        played = list(_column(store, "moves", match.id))
        m.restart(store.connection, match.id)
        m.restart(store.connection, match.id)

        [entry] = _column(store, "undone", match.id)
        assert entry["ply"] == 0 and entry["moves"] == played and entry["restart"] is True
        assert _column(store, "thinks", match.id) == []

    def test_a_rated_game_has_nothing_undone(self, store: Scratch) -> None:
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _person_moves(store.connection, match.id)
        with pytest.raises(m.MatchError):
            m.undo(store.connection, match.id)
        assert _column(store, "undone", match.id) is None


# ------------------------------------------------------- H3: the rating at the time


class TestTheRatingAtTheTime:
    def test_a_newcomers_practice_keeps_the_starting_values_and_says_so(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        table = r.load(store.connection, FORCED_HASH)
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT], ada)
        machine = table[r.computer(AGENT)]
        assert _column(store, "standing", match.id) == {
            "ratings": [r.NEWCOMER_RATING, machine.rating],
            "deviations": [r.NEWCOMER_DEVIATION, machine.deviation],
            "starting": [True, False],
        }

    def test_a_person_with_a_row_keeps_it_and_practice_moves_nothing(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        rated = m.start(store.connection, FORCED_HASH, m.LEADERBOARD, ["human", AGENT], ada)
        _play_out(store.connection, rated.id)
        assert _column(store, "standing", rated.id) is None, "a rated row keeps `before`"

        table = r.load(store.connection, FORCED_HASH)
        practice = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "human"], ada)
        _play_out(store.connection, practice.id)

        person, machine = table[r.person(ada)], table[r.computer(AGENT)]
        standing = _column(store, "standing", practice.id)
        assert standing == {
            "ratings": [machine.rating, person.rating],
            "deviations": [machine.deviation, person.deviation],
            "starting": [False, False],
        }
        assert r.load(store.connection, FORCED_HASH) == table, "a read only"
        [kept] = finishes.recorded(store.connection, mode=m.SANDBOX)
        assert kept.standing == standing

    def test_a_visitor_has_no_rating_to_keep(self, store: Scratch) -> None:
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        standing = _column(store, "standing", match.id)
        assert standing["ratings"][0] is None and standing["starting"][0] is None
        assert standing["ratings"][1] is not None

    def test_a_game_with_no_table_and_a_watch_keep_none(self, store: Scratch) -> None:
        watch = m.start(store.connection, FORCED_HASH, m.SANDBOX, [AGENT, "greedy"])
        assert _column(store, "standing", watch.id) is None
        with store.connection.cursor() as cursor:
            cursor.execute("delete from ratings where game = %s", (FORCED_HASH,))
        practice = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT])
        assert _column(store, "standing", practice.id) is None


# ------------------------------------------------------ out by the export, not a page


class TestWhereItGoes:
    def test_the_export_carries_all_three(
        self,
        store: Scratch,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        ada = store.user("ada")
        match = m.start(store.connection, FORCED_HASH, m.SANDBOX, ["human", AGENT], ada)
        _since(store, match.id, 4)
        _person_moves(store.connection, match.id)
        m.undo(store.connection, match.id)
        _play_out(store.connection, match.id)
        store.connection.commit()
        monkeypatch.setattr(db, "database_url", lambda: store.url)
        monkeypatch.setattr(db, "connect", lambda url: store.open())

        assert cli.main(["site", "games"]) == 0
        [line] = [json.loads(text) for text in capsys.readouterr().out.splitlines()]
        assert len(line["thinks"]) == len(line["moves"]) == 3
        assert line["thinks"][1] is None
        assert line["undone"][0]["thinks"] == [4000]
        assert line["standing"]["starting"] == [True, False]

    def test_no_page_payload_carries_them(self, store: Scratch) -> None:
        app = SiteApp(library.RULESET_DIR, library.RUNS_DIR)
        app.connect = store.open
        payloads = [app.post("/api/session", {"hash": FORCED_HASH, "seats": ["human", AGENT]})]
        snapshot = payloads[0]
        payloads.append(app.post(
            f"/api/session/{snapshot['id']}/move", {"move": snapshot["moves"][0]["move"]}
        ))
        payloads.append(app.post(f"/api/session/{snapshot['id']}/undo", {}))
        payloads.append(app.get(f"/api/session/{snapshot['id']}"))
        assert _column(store, "undone", snapshot["id"]) is not None, "it was recorded"
        for payload in payloads:
            text = json.dumps(payload)
            for word in ('"thinks"', '"undone"', '"standing"'):
                assert word not in text

    def test_the_privacy_page_names_them(self) -> None:
        page = privacy()
        assert "how long you took over each move" in page
        assert "which moves a take-back undid" in page
        assert "the ratings each side had on that game when it started" in page
