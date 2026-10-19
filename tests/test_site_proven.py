"""Proven positions on the site (R9b, Q605, `docs/plan-website.md` *Proven positions*).

R9a proved positions from the library's recorded games won for the side to move and wrote
them as `runs/puzzles/<hash>.json`. The site (`site/proven.py`) loads the record, replaying
every prefix through the engine; offers each position to **convert** -- a practice game
from the position against *hardest*, which the game's own ending decides -- and, where the
proof shows a single winning move, to **find the move**, answered from the stored scores;
and keeps a member's solved marks for their own page alone.

Proved here: the loader (order, what it drops and counts, the cache), *find the move*'s
four answers (a proved win is right, a proved loss loses, an *open* move is *not settled*
and never *wrong*, a position with several wins is not offered), the drawing it sends (the
moves without their scores), the basis line (test 7), and, against a store, a try from a
prefix: its row, its snapshot, that the engine's outcome decides it and a win marks it,
that a loss marks nothing, no take-back, a restart to the position, refusals (rated, a
visitor, a watch), the marks' privacy (yours only, deleted on leaving), the game page's
section and the profile's lines, and the export that says which moves were the board's.

On specs, never chess (rule 5): `dull` set up as one grunt a side, so that after two
forced moves the first player has one capture that ends the game. Nothing here works out a
legal move or an outcome; the engine answers, and the records are written from what it
answered. The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it.
"""

from __future__ import annotations

import html
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_site_matches import Scratch, _scratch

from chess_variant_sandbox import puzzles
from chess_variant_sandbox.corpus import RUNS_DIR, find
from chess_variant_sandbox.site import admin, finishes, proven, routes
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.library import engine_for

REPOSITORY = Path(__file__).resolve().parents[1]
DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))


def _spec() -> dict[str, Any]:
    """One grunt a side: b1 for the first player, c4 for the second. Each has one move,
    and then the first player may take the second's last piece."""
    spec = json.loads(json.dumps(DULL))
    spec["name"] = "proven test"
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "G", "squares": ["b1"], "player": 0},
        {"piece": "G", "squares": ["c4"], "player": 1},
    ]}
    spec["rules"]["max_plies"] = 40
    return spec


SPEC = _spec()
DIGEST = spec_hash(SPEC)
ENGINE = engine_for(SPEC)


def _prefix(*texts: str) -> list[int]:
    """The moves named, each looked up among the engine's legal moves as it goes."""
    state = ENGINE.initial_state()
    moves = []
    for text in texts:
        move = ENGINE.find_move(state, text)
        moves.append(move)
        state = ENGINE.apply(state, move)
    return moves


PREFIX = _prefix("Gb1-b2", "Gc4-c3")
AT = ENGINE.play(ENGINE.initial_state(), PREFIX)
LEGAL = ENGINE.legal_moves(AT)
TAKE = ENGINE.find_move(AT, "Gb2xc3")
STEP = ENGINE.find_move(AT, "Gb2-b3")


def _position(
    position_id: str,
    proofs: dict[int, str],
    moves: list[int] | None = None,
    result: str = "won",
    ends_in: int = 1,
    side: int = 0,
) -> dict[str, Any]:
    """A record's position in R9a's shape, the root moves in the engine's order with the
    proof each is given here (a record is data; what is tested is what the site does
    with it)."""
    scores = []
    for move in LEGAL:
        proof = proofs[move]
        entry: dict[str, Any] = {
            "move": move, "text": ENGINE.describe_move(AT, move),
            "score": {"won": 999_999.0, "lost": -999_999.0}.get(proof, 0.5), "proof": proof,
        }
        if proof != "open":
            entry["ends_in"] = ends_in
        scores.append(entry)
    winning = sum(1 for entry in scores if entry["proof"] == "won")
    single = result == "won" and winning == 1
    return {
        "id": position_id, "seed": 0, "game": 0, "ply": len(PREFIX), "pieces": 2,
        "moves": list(PREFIX if moves is None else moves), "side": side, "result": result,
        "ends_in": ends_in, "winning": winning, "single": single,
        "others_lost": single and all(e["proof"] == "lost" for e in scores if e["proof"] != "won"),
        "scores": scores,
    }


ONE = _position("0-0-2", {TAKE: "won", STEP: "open"}, ends_in=1)
LOSES = _position("0-1-2", {TAKE: "won", STEP: "lost"}, ends_in=3)
BOTH = _position("0-2-2", {TAKE: "won", STEP: "won"}, ends_in=2)


def _record(positions: list[dict[str, Any]], digest: str = DIGEST) -> dict[str, Any]:
    return {
        "format": puzzles.FORMAT, "ruleset_hash": digest, "name": "proven test",
        "made": "2026-10-18",
        "basis": {
            "probe": "Engine.root_scores", "nodes": 20_000_000, "max_depth": 400,
            "proven_at": puzzles.PROVEN, "goal": 16.0, "immune": 1.0,
            "built": "0123456789abcdef", "runs": [f"{digest}-{s}" for s in (0, 1000, 2000, 3000)],
            "max_pieces": 6, "distinct": 40, "near": 12, "cap": 240,
        },
        "counts": {"candidates": 5, "forced": 0, "won": 3, "lost": 0, "open": 2,
                   "single": 2, "others_lost": 1},
        "positions": positions,
    }


def _write(runs: Path, record: dict[str, Any], digest: str = DIGEST) -> Path:
    where = proven.path(runs, digest)
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps(record), encoding="utf-8")
    return where


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    """A runs directory holding the test game's record: three offered positions."""
    _write(tmp_path, _record([ONE, LOSES, BOTH]))
    return tmp_path


def _load(runs: Path) -> proven.Record:
    found = proven.load(runs, DIGEST, SPEC)
    assert found is not None
    return found


# ------------------------------------------------------------------------ the loader


class TestTheLoader:
    def test_a_record_is_offered_shortest_win_first(self, runs: Path) -> None:
        record = _load(runs)
        assert [found.id for found in record.positions] == ["0-0-2", "0-2-2", "0-1-2"]
        assert [found.number for found in record.positions] == [1, 2, 3]
        assert record.dropped == 0 and record.singles == 2
        first = record.find("0-0-2")
        assert first is not None and first.moves == tuple(PREFIX) and first.side == 0
        assert first.ply == len(PREFIX)

    def test_a_position_that_no_longer_replays_is_dropped_and_counted(
        self, tmp_path: Path
    ) -> None:
        # A prefix whose second move is not legal where it is played, a position whose
        # root moves are not the engine's, one the record says is lost (nothing to
        # convert), one whose side to move is not the record's, and one with no moves.
        illegal = _position("0-3-2", {TAKE: "won", STEP: "open"}, moves=[PREFIX[0], PREFIX[0]])
        other_moves = json.loads(json.dumps(ONE))
        other_moves["id"] = "0-4-2"
        other_moves["scores"] = other_moves["scores"][:1]
        lost = _position("0-5-2", {TAKE: "lost", STEP: "lost"}, result="lost")
        wrong_side = _position("0-6-2", {TAKE: "won", STEP: "open"}, side=1)
        broken = {"id": "0-7-2", "result": "won"}
        _write(tmp_path, _record([ONE, illegal, other_moves, lost, wrong_side, broken]))
        record = _load(tmp_path)
        assert [found.id for found in record.positions] == ["0-0-2"]
        assert record.dropped == 5
        assert "5 more no longer replay" in proven.basis_said(record)

    def test_no_record_another_games_or_another_format_is_none(self, tmp_path: Path) -> None:
        assert proven.load(tmp_path, DIGEST, SPEC) is None
        _write(tmp_path, _record([ONE], digest="aaaaaaaaaaaa"))
        assert proven.load(tmp_path, DIGEST, SPEC) is None
        wrong = _record([ONE])
        wrong["format"] = puzzles.FORMAT + 1
        _write(tmp_path, wrong)
        assert proven.load(tmp_path, DIGEST, SPEC) is None
        _write(tmp_path, _record([ONE]))
        assert proven.load(tmp_path, DIGEST, None) is None
        proven.path(tmp_path, DIGEST).write_text("{ not json", encoding="utf-8")
        assert proven.load(tmp_path, DIGEST, SPEC) is None

    def test_a_changed_file_is_read_again(self, tmp_path: Path) -> None:
        where = _write(tmp_path, _record([ONE]))
        assert len(_load(tmp_path).positions) == 1
        where.write_text(json.dumps(_record([ONE, LOSES, BOTH])), encoding="utf-8")
        os.utime(where, ns=(where.stat().st_atime_ns, where.stat().st_mtime_ns + 10**9))
        assert len(_load(tmp_path).positions) == 3

    @pytest.mark.parametrize(
        "path", sorted((Path(RUNS_DIR) / "puzzles").glob("*.json")) or [None],
        ids=lambda path: "none-here" if path is None else path.stem,
    )
    def test_every_delivered_record_is_offered_whole(self, path: Path | None) -> None:
        """The site's loader drops nothing from a record R9a delivered: under today's rules
        every prefix replays to its position. Skips where no record is on disk."""
        if path is None:
            pytest.skip("no runs/puzzles/ record here")
        data = json.loads(path.read_text(encoding="utf-8"))
        record = proven.load(path.parent.parent, path.stem, find(path.stem).spec)
        assert record is not None
        assert record.dropped == 0
        assert len(record.positions) == data["counts"]["won"]
        assert record.singles == data["counts"]["single"]


# ------------------------------------------------------------------- find the move


class TestFindTheMove:
    def test_the_proved_move_is_right(self, runs: Path) -> None:
        said = proven.answer(_load(runs), "0-0-2", TAKE)
        assert said["right"] is True and said["proof"] == "won"
        assert said["said"].startswith("Yes: Gb2xc3 is the move the computer proved wins")

    def test_an_open_move_is_not_settled_and_never_wrong(self, runs: Path) -> None:
        said = proven.answer(_load(runs), "0-0-2", STEP)
        assert said["right"] is False and said["proof"] == "open"
        assert "not the move the computer proved" in said["said"]
        assert "could not settle" in said["said"]
        for word in ("wrong", "loses", "No:"):
            assert word not in said["said"]

    def test_a_move_proved_lost_is_said_to_lose(self, runs: Path) -> None:
        said = proven.answer(_load(runs), "0-1-2", STEP)
        assert said["right"] is False and said["proof"] == "lost"
        assert said["said"].startswith("No: the computer proved Gb2-b3 loses.")
        assert "Every move here but one loses." in said["said"]

    def test_a_position_with_two_winning_moves_is_not_asked(self, runs: Path) -> None:
        with pytest.raises(ValueError, match="more than one winning move"):
            proven.answer(_load(runs), "0-2-2", TAKE)

    def test_showing_the_move_names_it_and_is_no_pick(self, runs: Path) -> None:
        said = proven.answer(_load(runs), "0-0-2", None, show=True)
        assert said["shown"] == "Gb2xc3" and said["right"] is False and said["proof"] is None

    def test_a_move_not_in_the_position_or_a_position_not_here_is_refused(
        self, runs: Path
    ) -> None:
        with pytest.raises(ValueError, match="not one of the moves"):
            proven.answer(_load(runs), "0-0-2", 1)
        with pytest.raises(KeyError):
            proven.answer(_load(runs), "9-9-9", TAKE)

    def test_the_drawing_sends_the_moves_and_never_their_scores(self, runs: Path) -> None:
        drawn = proven.view(_load(runs), SPEC, "0-0-2")
        assert drawn["preview"]["board"] == list(AT.board)
        assert drawn["preview"]["toMove"] == 0
        assert [entry["move"] for entry in drawn["moves"]] == LEGAL
        assert all(set(entry) == {"move", "text"} for entry in drawn["moves"])
        assert "score" not in json.dumps(drawn["moves"]) and "proof" not in json.dumps(drawn)
        # Two winning moves: offered to convert, with nothing to pick from.
        assert proven.view(_load(runs), SPEC, "0-2-2")["moves"] == []


# ---------------------------------------------------------------- what the page reads


class TestTheWords:
    def test_the_basis_says_where_how_how_many_and_when(self, runs: Path) -> None:
        said = proven.basis_said(_load(runs))
        assert "this game's 4 runs" in said
        assert "with 6 pieces or fewer left" in said
        assert "within 12 turns of the end of a game one side won" in said
        assert "looking at up to 20 million positions" in said
        assert "Of 5 positions asked it proved 3; it could not settle 2." in said
        assert "Proved on 2026-10-18." in said
        assert "0123456789abcdef" not in said

    def test_no_sentence_speaks_in_the_maintainers_voice(self, runs: Path) -> None:
        record = _load(runs)
        sentences = [
            proven.basis_said(record), proven.NOT_HERE, proven.NO_TAKE_BACK,
            proven.SIGN_IN_TO_CONVERT, *proven.TRY_SAID.values(), m.PUZZLE_IS_PRACTICE,
            proven.view(record, SPEC, "0-0-2")["said"],
            *(proven.answer(record, "0-0-2", move)["said"] for move in LEGAL),
            proven.answer(record, "0-1-2", STEP)["said"],
            proven.answer(record, "0-0-2", None, show=True)["said"],
        ]
        for sentence in sentences:
            assert site_words.hits(sentence) == [], sentence

    def test_the_section_counts_the_readers_own_marks_and_nobody_elses(
        self, runs: Path
    ) -> None:
        record = _load(runs)
        visitor = proven.section(record, None, signed_in=False)
        assert visitor["yours"] is None and visitor["signInToConvert"]
        assert visitor["count"] == 3 and visitor["singles"] == 2
        assert [entry["id"] for entry in visitor["positions"]] == ["0-0-2", "0-2-2", "0-1-2"]
        mine = proven.section(
            record, {"0-0-2": {proven.CONVERTED, proven.FOUND}, "gone": {proven.FOUND}},
            signed_in=True,
        )
        assert mine["yours"] == {"converted": 1, "found": 1}
        assert mine["signInToConvert"] == ""
        assert mine["positions"][0]["converted"] and mine["positions"][0]["found"]
        assert not any(key in json.dumps(mine) for key in ("people", "everyone", "others"))

    def test_the_profile_counts_against_what_is_offered_now(self, runs: Path) -> None:
        record = _load(runs)
        lines = proven.profile_lines(
            {DIGEST: {"0-0-2": {proven.CONVERTED, proven.FOUND}, "0-2-2": {proven.CONVERTED},
                      "gone": {proven.CONVERTED}},
             "bbbbbbbbbbbb": {"x": {proven.FOUND}}},
            {DIGEST: record, "bbbbbbbbbbbb": None},
            {DIGEST: "proven test", "bbbbbbbbbbbb": "another"},
        )
        assert lines == [
            {"hash": "bbbbbbbbbbbb", "name": "another", "converted": 0, "found": 1,
             "of": None, "singles": None},
            {"hash": DIGEST, "name": "proven test", "converted": 2, "found": 1,
             "of": 3, "singles": 2},
        ]

    def test_a_try_seats_the_person_on_the_side_to_move_against_hardest(
        self, runs: Path
    ) -> None:
        first = _load(runs).find("0-0-2")
        assert first is not None
        assert proven.seats_for(first) == ["human", "minimax:nodes=200000"]
        assert proven.OPPONENT == "minimax:nodes=200000"
        second = proven.Position(
            id="x", number=1, moves=(), side=1, ends_in=1, pieces=2, single=False,
            others_lost=False, scores=(),
        )
        assert proven.seats_for(second) == ["minimax:nodes=200000", "human"]

    def test_the_routes_are_registered_for_anybody(self) -> None:
        for method, path in (("GET", "/api/proven/<hash>/<position>"),
                             ("POST", "/api/proven/<hash>/<position>/find")):
            found = [route for route in routes.OWN
                     if route.method == method and route.path == path]
            assert found and found[0].visitor == ""
        assert routes.find("GET", f"/api/proven/{DIGEST}/0-0-2") is not None
        assert routes.find("POST", f"/api/proven/{DIGEST}/0-0-2/find") is not None


# --------------------------------------------------------------------- the store


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(DIGEST, SPEC)
        yield scratch


def _try(store: Scratch, user: int | None, position: dict[str, Any] = ONE) -> m.Row:
    found = m.start(
        store.connection, DIGEST, m.SANDBOX, ["human", proven.OPPONENT], user,
        puzzle=(str(position["id"]), position["moves"]),
    )
    store.connection.commit()
    return found


def _marks(store: Scratch) -> list[tuple[Any, ...]]:
    with store.connection.cursor() as cursor:
        cursor.execute("select person, game, position, way, match from proven_marks"
                       " order by person, position, way")
        return [tuple(row) for row in cursor.fetchall()]


class TestATry:
    def test_it_starts_from_the_prefix_as_practice(self, store: Scratch) -> None:
        ada = store.user("ada")
        found = _try(store, ada)
        assert found.mode == m.SANDBOX and found.source == m.PUZZLE
        assert m.puzzle_of(store.connection, found.id) == ("0-0-2", len(PREFIX))
        assert found.plies == len(PREFIX)
        snapshot = m.session(store.connection, found.id).snapshot()
        assert snapshot["board"] == list(AT.board) and snapshot["sideToMove"] == 0
        assert snapshot["puzzle"]["id"] == "0-0-2" and snapshot["puzzle"]["side"] == 0
        assert snapshot["puzzle"]["from"] == len(PREFIX)
        assert snapshot["puzzle"]["said"] == proven.TRY_SAID["said"]
        # An ordinary match carries no block at all.
        other = m.start(store.connection, DIGEST, m.SANDBOX, ["human", "random"], ada)
        assert "puzzle" not in m.session(store.connection, other.id).snapshot()
        assert m.puzzle_of(store.connection, other.id) is None

    def test_the_games_own_ending_converts_it_and_marks_it(self, store: Scratch) -> None:
        ada = store.user("ada")
        found = _try(store, ada)
        snapshot = m.play(store.connection, found.id, TAKE)
        store.connection.commit()
        # The engine ended the game, and named the person's side the winner.
        assert snapshot["outcome"]["winner"] == 0
        assert m.row(store.connection, found.id).result == 1.0
        assert _marks(store) == [(ada, DIGEST, "0-0-2", proven.CONVERTED, found.id)]
        assert proven.marks_on(store.connection, ada, DIGEST) == {"0-0-2": {proven.CONVERTED}}
        # Converting it again is still one mark.
        again = _try(store, ada)
        m.play(store.connection, again.id, TAKE)
        store.connection.commit()
        assert len(_marks(store)) == 1

    def test_a_try_that_ends_otherwise_marks_nothing_and_is_a_game_lost(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        found = _try(store, ada)
        m.play(store.connection, found.id, STEP)
        m.resign(store.connection, found.id, by=ada)
        store.connection.commit()
        assert m.row(store.connection, found.id).result == 0.0
        assert _marks(store) == []
        kept = finishes.recorded(store.connection, game=DIGEST)
        assert [(f.result, f.source, f.prefix) for f in kept] == [(0.0, m.PUZZLE, len(PREFIX))]

    def test_a_visitors_win_marks_nobody(self, store: Scratch) -> None:
        found = m.start(
            store.connection, DIGEST, m.SANDBOX, ["human", "minimax:nodes=6400"], None,
            puzzle=("0-0-2", PREFIX),
        )
        m.play(store.connection, found.id, TAKE)
        store.connection.commit()
        assert _marks(store) == []

    def test_no_take_back_and_a_restart_goes_back_to_the_position(
        self, store: Scratch
    ) -> None:
        ada = store.user("ada")
        found = _try(store, ada)
        m.play(store.connection, found.id, STEP)
        with pytest.raises(m.MatchError, match="no take-back"):
            m.undo(store.connection, found.id)
        store.connection.rollback()
        back = m.restart(store.connection, found.id)
        store.connection.commit()
        assert back["board"] == list(AT.board) and len(back["history"]) == len(PREFIX)
        assert m.row(store.connection, found.id).plies == len(PREFIX)

    def test_a_try_is_never_rated_a_watch_or_a_rematch(self, store: Scratch) -> None:
        ada = store.user("ada")
        with pytest.raises(m.MatchError, match="practice"):
            m.start(store.connection, DIGEST, m.LEADERBOARD, ["human", proven.OPPONENT], ada,
                    puzzle=("0-0-2", PREFIX))
        with pytest.raises(m.MatchError, match="one person"):
            m.start(store.connection, DIGEST, m.SANDBOX, [proven.OPPONENT, proven.OPPONENT],
                    ada, puzzle=("0-0-2", PREFIX))
        earlier = _try(store, ada)
        with pytest.raises(m.MatchError, match="started from the position"):
            m.start(store.connection, DIGEST, m.SANDBOX, ["human", proven.OPPONENT], ada,
                    rematch_of=earlier.id, puzzle=("0-0-2", PREFIX))

    def test_the_store_itself_holds_a_try_to_practice(self, store: Scratch) -> None:
        ada = store.user("ada")
        with store.connection.cursor() as cursor, pytest.raises(Exception, match="puzzle"):
            cursor.execute(
                "insert into matches (id, game, mode, seed, seat0_user, seat1_agent, source,"
                " puzzle, puzzle_ply) values ('x', %s, 'leaderboard', 0, %s, 'random',"
                " 'puzzle', '0-0-2', 2)",
                (DIGEST, ada),
            )
        store.connection.rollback()

    def test_leaving_deletes_a_persons_marks(self, store: Scratch) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        proven.mark(store.connection, ada, DIGEST, "0-0-2", proven.FOUND)
        proven.mark(store.connection, bo, DIGEST, "0-0-2", proven.FOUND)
        store.connection.commit()
        admin.forget(store.connection, "ada")
        store.connection.commit()
        assert [row[0] for row in _marks(store)] == [bo]


# ---------------------------------------------------------------- through the site


@pytest.fixture
def app(runs: Path) -> SiteApp:
    """The site over the test game's record, the game known as a library game."""
    site = SiteApp(library.RULESET_DIR, runs)
    game = library.Game(
        hash=DIGEST, name="proven test", description="", path=Path(), source="committed",
        spec=SPEC,
    )
    site.known = lambda: {DIGEST: game}  # type: ignore[method-assign]
    return site


def _as(app: SiteApp, viewer: int | None) -> SiteApp:
    app.viewer = viewer
    return app


class TestThroughTheSite:
    def test_a_member_converts_it_and_reads_it_on_their_own_page(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada, bo = store.user("ada"), store.user("bo")
        app.connect = store.open
        try:
            started = _as(app, ada).post("/api/session", {"hash": DIGEST, "puzzle": "0-0-2"})
            assert started["seats"] == ["human", proven.OPPONENT]
            assert started["puzzle"]["id"] == "0-0-2" and started["mode"] == m.SANDBOX
            done = app.post(f"/api/session/{started['id']}/move", {"move": TAKE})
            assert done["outcome"]["winner"] == done["puzzle"]["side"] == 0
            section = app._standing(DIGEST)["standing"]["proven"]
            assert section["yours"] == {"converted": 1, "found": 0}
            assert section["positions"][0]["converted"] is True
            lines = app.get("/api/profile")["proven"]
            assert lines == [{"hash": DIGEST, "name": "proven test", "converted": 1,
                              "found": 0, "of": 3, "singles": 2}]
            # Somebody else reads none of it, on the game or on their own page.
            assert _as(app, bo)._standing(DIGEST)["standing"]["proven"]["yours"] == {
                "converted": 0, "found": 0}
            assert app.get("/api/profile")["proven"] == []
        finally:
            app.viewer = None

    def test_finding_the_move_marks_a_right_pick_only(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada = store.user("ada")
        app.connect = store.open
        try:
            address = f"/api/proven/{DIGEST}/0-0-2"
            drawn = _as(app, ada).get(address)
            assert [entry["move"] for entry in drawn["moves"]] == LEGAL
            wrong = app.post(address + "/find", {"move": STEP})
            assert wrong["right"] is False and wrong["marked"] is False
            assert _marks(store) == []
            shown = app.post(address + "/find", {"show": True})
            assert shown["marked"] is False and _marks(store) == []
            right = app.post(address + "/find", {"move": TAKE})
            assert right["right"] is True and right["marked"] is True
            assert _marks(store) == [(ada, DIGEST, "0-0-2", proven.FOUND, None)]
            with pytest.raises(ValueError, match="pick one"):
                app.post(address + "/find", {"move": "Gb2xc3"})
        finally:
            app.viewer = None

    def test_a_visitor_finds_the_move_and_is_told_to_sign_in_to_convert(
        self, store: Scratch, app: SiteApp
    ) -> None:
        app.connect = store.open
        _as(app, None)
        section = app._standing(DIGEST)["standing"]["proven"]
        assert section["yours"] is None
        assert section["signInToConvert"] == proven.SIGN_IN_TO_CONVERT
        right = app.post(f"/api/proven/{DIGEST}/0-0-2/find", {"move": TAKE})
        assert right["right"] is True and right["marked"] is False
        assert _marks(store) == []
        with pytest.raises(m.MatchError, match="hardest is for people with an account"):
            app.post("/api/session", {"hash": DIGEST, "puzzle": "0-0-2"})
        assert store.rows() == 0

    def test_a_try_takes_nothing_but_the_game_and_the_position(
        self, store: Scratch, app: SiteApp
    ) -> None:
        ada = store.user("ada")
        app.connect = store.open
        try:
            for body in ({"mode": m.LEADERBOARD}, {"rematch_of": "x"}, {"seed": 7}):
                with pytest.raises(m.MatchError, match="practice"):
                    _as(app, ada).post(
                        "/api/session", {"hash": DIGEST, "puzzle": "0-0-2", **body}
                    )
            with pytest.raises(KeyError):
                app.post("/api/session", {"hash": DIGEST, "puzzle": "9-9-9"})
            assert store.rows() == 0
        finally:
            app.viewer = None

    def test_a_game_with_no_record_has_no_section_and_no_address(
        self, store: Scratch, app: SiteApp, tmp_path: Path
    ) -> None:
        app.connect = store.open
        proven.path(app.runs_dir, DIGEST).unlink()
        _as(app, None)
        assert app._standing(DIGEST)["standing"]["proven"] is None
        with pytest.raises(KeyError):
            app.get(f"/api/proven/{DIGEST}/0-0-2")


# ------------------------------------------------------------------- the page draws


_GAME = f"""
const GAMEX = {{ name: 'Proven test', hash: '{DIGEST}', sides: ['north', 'south'], standing: {{
  proven: {{ basis: 'Proved on 2026-10-18.', count: 2, singles: 1, signInToConvert: '',
    yours: {{ converted: 1, found: 0 }},
    positions: [
      {{ id: '0-0-2', number: 1, side: 0, endsIn: 1, single: true, converted: true, found: false }},
      {{ id: '0-1-2', number: 2, side: 1, endsIn: 3, single: false, converted: false, found: false }}
    ] }} }} }};
"""


class TestTheGamePageDraws:
    def test_the_section_lists_the_positions_in_the_servers_order(self) -> None:
        from test_web import _in_page

        panel = html.unescape(_in_page(_GAME, "provenHtml(GAMEX)"))
        assert "Proven positions" in panel and 'id="provenPick"' in panel
        assert panel.index("1 · North to move, a win in 1 turn · one winning move") < \
            panel.index("2 · South to move, a win in 3 turns")
        assert "you converted it" in panel
        assert "You have converted 1 of these 2, and found the move in 0 of the 1" in panel
        assert "Only you see this." in panel and "Proved on 2026-10-18." in panel
        assert "There is no clock, and nothing here expires." in panel

    def test_no_section_without_proven_positions(self) -> None:
        from test_web import _in_page

        assert _in_page(_GAME + "GAMEX.standing.proven = null;", "provenHtml(GAMEX)") == ""
        assert _in_page("const GAMEX = { standing: {} };", "provenHtml(GAMEX)") == ""

    def test_a_position_offers_convert_and_the_moves_without_scores(self) -> None:
        from test_web import _in_page

        position = (
            "const SPOT = { id: '0-0-2', number: 1, side: 0, endsIn: 1, single: true,"
            " said: 'The computer proved a win.', preview: null,"
            " moves: [{ move: 7, text: 'Gb2xc3' }, { move: 9, text: 'Gb2-b3' }] };"
        )
        drawn = html.unescape(_in_page(
            _GAME + position, "provenAtHtml(GAMEX, GAMEX.standing.proven, SPOT)"
        ))
        assert 'id="provenConvert"' in drawn and "convert it against hardest" in drawn
        assert 'data-proven-move="7"' in drawn and "Gb2-b3" in drawn
        assert "Find the move" in drawn and 'id="provenShow"' in drawn
        assert "North to move. The computer proved a win." in drawn
        visitor = html.unescape(_in_page(
            _GAME + position + "GAMEX.standing.proven.signInToConvert = 'Sign in to convert.';",
            "provenAtHtml(GAMEX, GAMEX.standing.proven, SPOT)",
        ))
        assert 'id="provenConvert"' not in visitor and "Sign in to convert." in visitor

    def test_the_board_says_how_a_try_ended_from_the_engines_outcome(self) -> None:
        from test_web import _in_page

        block = "{ id: 'x', side: 1, said: 'running', won: 'converted', lost: 'not this time' }"
        assert _in_page("", f"puzzleSaid({{ puzzle: {block}, outcome: null }})") == "running"
        assert _in_page(
            "", f"puzzleSaid({{ puzzle: {block}, outcome: {{ winner: 1 }} }})"
        ) == "converted"
        for winner in ("0", "null"):
            assert _in_page(
                "", f"puzzleSaid({{ puzzle: {block}, outcome: {{ winner: {winner} }} }})"
            ) == "not this time"
        assert _in_page("", "puzzleSaid({ outcome: null })") == ""

    def test_a_try_has_no_take_back(self) -> None:
        from test_web import _in_page

        with_one = _in_page("", "frameHtml(false, 'sandbox', false, null, false, false)")
        without = _in_page("", "frameHtml(false, 'sandbox', false, null, false, true)")
        assert 'id="undo"' in with_one and 'id="undo"' not in without
        assert 'id="resign"' in without

    def test_the_profile_draws_your_lines(self) -> None:
        profile = (REPOSITORY / "chess_variant_sandbox" / "site" / "static" / "profile.js"
                   ).read_text(encoding="utf-8")
        start = profile.index("function drawProven")
        body = profile[start: profile.index("\n}\n", start)]
        assert "if (!lines.length) return;" in body
        assert "Only you see this." in body
        assert "drawProven(payload.proven || []);" in profile
