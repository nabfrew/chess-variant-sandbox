"""C15 (`docs/plan-performance.md` §2): the site's reads held between requests.

Six caches, each changing only what a process keeps between requests and never what it
computes. Each is held to one thing here: an in-process write that moves what it holds is
seen at the very next read. (a) the library walk (`library.discover`'s hold), (b) a game
page's two compiles (`library._live_terms`, `_piece_values`), (c) the store's connections
(`db.Pool`), (d) `/api/design`'s corpus answers (`App._design_corpus`), (e) a card's rule
facts (`Game.shape`), (f) the sitemap's proven positions (`proven.listed`).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import site_store
from test_site_proven import BOTH, DIGEST, LOSES, ONE, SPEC, _record, _write

from chess_variant_sandbox.site import db, proven
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import designing, library
from chess_variant_sandbox.web.server import App

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _tree(tmp_path: Path, *names: str) -> tuple[Path, Path, Path]:
    """A library of its own: these committed rulesets, an empty runs/ and archive/."""
    rulesets, runs, archive = tmp_path / "rulesets", tmp_path / "runs", tmp_path / "archive"
    for directory in (rulesets, runs, archive):
        directory.mkdir()
    for name in names:
        (rulesets / name).write_bytes((RULESETS / name).read_bytes())
    return rulesets, runs, archive


def _hashes(games: list[library.Game]) -> set[str]:
    return {game.hash for game in games}


# ------------------------------------------------------------------- (a) the library walk


class TestTheWalkIsHeld:
    def test_a_held_listing_skips_the_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        roots = _tree(tmp_path, "chess.json")
        first = library.discover(*roots, hold=60.0)
        walks: list[Any] = []
        real = library._scan
        monkeypatch.setattr(library, "_scan", lambda *a: walks.append(a) or real(*a))
        assert _hashes(library.discover(*roots, hold=60.0)) == _hashes(first)
        assert walks == []
        # Without a hold, as every caller but `App.refresh`, the walk runs as before.
        library.discover(*roots)
        assert len(walks) == 1

    def test_an_in_process_write_is_seen_at_the_next_listing(self, tmp_path: Path) -> None:
        rulesets, runs, archive = _tree(tmp_path, "chess.json")
        before = _hashes(library.discover(rulesets, runs, archive, hold=60.0))
        with library.writing():
            (rulesets / "dull.json").write_bytes((RULESETS / "dull.json").read_bytes())
        after = _hashes(library.discover(rulesets, runs, archive, hold=60.0))
        assert after - before == {spec_hash(load_spec(RULESETS / "dull.json"))}

    def test_nothing_is_held_while_a_writer_is_inside(self, tmp_path: Path) -> None:
        rulesets, runs, archive = _tree(tmp_path, "chess.json")
        with library.writing():
            library.discover(rulesets, runs, archive, hold=60.0)
            (rulesets / "dull.json").write_bytes((RULESETS / "dull.json").read_bytes())
            assert len(library.discover(rulesets, runs, archive, hold=60.0)) == 2

    def test_a_write_from_outside_is_seen_once_the_hold_is_out(self, tmp_path: Path) -> None:
        rulesets, runs, archive = _tree(tmp_path, "chess.json")
        library.discover(rulesets, runs, archive, hold=0.2)
        (rulesets / "dull.json").write_bytes((RULESETS / "dull.json").read_bytes())
        assert len(library.discover(rulesets, runs, archive, hold=60.0)) == 1  # held
        time.sleep(0.25)
        assert len(library.discover(rulesets, runs, archive, hold=0.2)) == 2

    def test_every_post_is_a_writer(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        app = App(*_tree(tmp_path, "chess.json")[:2], archive_dir=tmp_path / "archive")
        inside: list[int] = []
        monkeypatch.setattr(App, "_post", lambda self, path, body: inside.append(
            library._writers))
        app.post("/api/anything", {})
        assert inside == [1] and library._writers == 0

    def test_a_game_nobody_has_listed_walks_whatever_is_held(self, tmp_path: Path) -> None:
        rulesets, runs, archive = _tree(tmp_path, "chess.json")
        app = App(rulesets, runs, archive_dir=archive)
        app.refresh()
        (rulesets / "dull.json").write_bytes((RULESETS / "dull.json").read_bytes())
        found = app.game(spec_hash(load_spec(RULESETS / "dull.json")))
        assert found.name == load_spec(RULESETS / "dull.json")["name"]


# ------------------------------------------------------- (b) a game page's two compiles


class TestTheGamePagesCompilesAreHeld:
    def test_each_answer_is_the_designers_own_and_a_copy(self) -> None:
        spec = load_spec(RULESETS / "chess.json")
        assert library._live_terms(spec) == designing.live_terms(spec)
        assert library._piece_values(spec) == designing.piece_values(spec)
        library._piece_values(spec)[0]["name"] = "edited by a caller"
        library._live_terms(spec).append("edited by a caller")
        assert library._piece_values(spec) == designing.piece_values(spec)
        assert library._live_terms(spec) == designing.live_terms(spec)

    def test_an_edited_spec_is_asked_again(self) -> None:
        spec = load_spec(RULESETS / "chess.json")
        library._piece_values(spec)
        edited = json.loads(json.dumps(spec))
        edited["pieces"][0]["name"] = "renamed in place"
        assert library._piece_values(edited)[0]["name"] == "renamed in place"


# ------------------------------------------------------------------ (c) the connections


class _Fake:
    """A driver's connection as `db.Pool` reads one."""

    def __init__(self) -> None:
        self.info = SimpleNamespace(transaction_status=0)
        self.closed = False
        self.broken = False
        self.asked: list[str] = []

    def execute(self, sql: str) -> None:
        if self.broken:
            raise OSError("server closed the connection")
        self.asked.append(sql)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _site(tmp_path: Path, opened: list[_Fake]) -> SiteApp:
    rulesets, runs, archive = _tree(tmp_path)

    def connect() -> _Fake:
        opened.append(_Fake())
        return opened[-1]

    return SiteApp(rulesets, runs, archive_dir=archive, connect=connect)


class TestTheConnectionsAreKept:
    def test_a_committed_connection_serves_the_next_unit_of_work(self, tmp_path: Path) -> None:
        opened: list[_Fake] = []
        app = _site(tmp_path, opened)
        for _ in range(5):
            with app._connection():
                pass
        assert len(opened) == 1 and not opened[0].closed

    def test_a_rolled_back_connection_is_closed(self, tmp_path: Path) -> None:
        opened: list[_Fake] = []
        app = _site(tmp_path, opened)
        with pytest.raises(RuntimeError), app._connection():
            raise RuntimeError("a unit of work that fails")
        assert opened[0].closed
        with app._connection():
            pass
        assert len(opened) == 2

    def test_another_connect_drops_what_the_old_one_opened(self, tmp_path: Path) -> None:
        opened: list[_Fake] = []
        app = _site(tmp_path, opened)
        with app._connection():
            pass
        mine: list[_Fake] = []
        app.connect = lambda: mine.append(_Fake()) or mine[-1]
        with app._connection() as connection:
            assert connection is mine[0]
        assert opened[0].closed

    def test_a_long_idle_connection_is_asked_and_a_dead_one_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[_Fake] = []
        app = _site(tmp_path, opened)
        with app._connection():
            pass
        monkeypatch.setattr(db, "POOL_PING_SECONDS", -1.0)
        with app._connection():
            pass
        assert opened[0].asked == ["select 1"] and len(opened) == 1
        opened[0].broken = True
        with app._connection() as connection:
            assert connection is opened[1]
        assert opened[0].closed

    def test_a_connection_left_inside_a_transaction_is_not_kept(self, tmp_path: Path) -> None:
        opened: list[_Fake] = []
        app = _site(tmp_path, opened)
        with app._connection() as connection:
            connection.commit = lambda: None
            connection.info.transaction_status = 2  # in a transaction, as if never committed
        assert opened[0].closed

    def test_against_a_real_store(self, tmp_path: Path) -> None:
        url, schema = site_store.emptied("the pool keeps a real connection between units")
        opened: list[Any] = []

        def open_one() -> Any:
            connection = db.connect(url)
            with connection.cursor() as cursor:
                cursor.execute(f"set search_path to {schema}")
            opened.append(connection)
            return connection

        app = SiteApp(*_tree(tmp_path)[:2], archive_dir=tmp_path / "archive", connect=open_one)
        try:
            for _ in range(3):
                with app._connection() as connection:
                    found = connection.execute("show search_path").fetchone()[0]
                    assert schema in found
            assert len(opened) == 1
            with pytest.raises(RuntimeError), app._connection():
                raise RuntimeError("rolled back")
            assert opened[0].closed
        finally:
            app._pool.clear()
            for connection in opened:
                connection.close()


# --------------------------------------------------------- (d) `/api/design`'s corpus


class TestTheDesignersCorpusIsHeld:
    def test_an_unmoved_corpus_is_not_read_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rulesets, runs, archive = _tree(tmp_path, "chess.json")
        app = App(rulesets, runs, archive_dir=archive)
        first = app.get("/api/design")
        from chess_variant_sandbox.web import cards

        monkeypatch.setattr(cards, "corpus_specs", lambda *dirs: pytest.fail("read again"))
        again = app.get("/api/design")
        assert json.dumps(again, sort_keys=True) == json.dumps(first, sort_keys=True)

    def test_a_ruleset_written_in_process_is_read_at_the_next_request(
        self, tmp_path: Path
    ) -> None:
        rulesets, runs, archive = _tree(tmp_path, "chess.json")
        app = App(rulesets, runs, archive_dir=archive)
        before = app.get("/api/design")
        (rulesets / "dull.json").write_bytes((RULESETS / "dull.json").read_bytes())
        after = app.get("/api/design")
        assert after["palette"] != before["palette"]
        fresh = App(rulesets, runs, archive_dir=archive).get("/api/design")
        for key in ("cards", "doors", "palette"):
            assert json.dumps(after[key], sort_keys=True) == json.dumps(fresh[key], sort_keys=True)


# ------------------------------------------------------------- (e) a card's rule facts


class TestACardsRuleFactsAreHeld:
    def test_the_card_is_unchanged_and_an_edited_spec_is_counted_again(
        self, tmp_path: Path
    ) -> None:
        rulesets, runs, archive = _tree(tmp_path, "chess.json")
        game = library.discover(rulesets, runs, archive)[0]
        card = game.summary()
        complexity = library.measure_complexity(game.spec)
        assert (card["pieceTypes"], card["ruleMass"], card["specialRules"]) == (
            complexity.piece_types, round(complexity.rule_mass, 1), complexity.special_rules)
        assert card["cell"] == library.label(library.descriptor(game.spec))
        card["cellBins"].append("edited by a caller")
        assert game.summary()["cellBins"] == list(library.bins_of(library.descriptor(game.spec)))
        # Other rules written over the same file, in process: the card counts them.
        other = load_spec(RULESETS / "dull.json")
        with library.writing():
            (rulesets / "chess.json").write_bytes((RULESETS / "dull.json").read_bytes())
        edited = library.discover(rulesets, runs, archive, hold=60.0)[0].summary()
        counted = library.measure_complexity(other)
        assert (edited["pieceTypes"], edited["ruleMass"], edited["cell"]) == (
            counted.piece_types, round(counted.rule_mass, 1),
            library.label(library.descriptor(other)))
        assert edited["cell"] != card["cell"] or edited["ruleMass"] != card["ruleMass"]


# --------------------------------------------------- (f) the sitemap's proven positions


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    where = tmp_path / "proof-store" / proven.PROOF_ANSWERS_NAME
    _restart(monkeypatch, where)
    yield where
    proven.proof_answers_key.cache_clear()


def _restart(monkeypatch: pytest.MonkeyPatch, where: Path) -> list[str]:
    """A new process: nothing in memory but the store's file. Returns each replay's game."""
    for name in ("_CACHE", "_COUNTED", "_FLIGHTS", "_LISTED"):
        monkeypatch.setattr(proven, name, {})
    monkeypatch.setattr(proven, "_PROOFS", proven.ProofAnswers(where))
    proven.proof_answers_key.cache_clear()
    replays: list[str] = []
    real = proven._read

    def counted(data: dict[str, Any], digest: str, spec: dict[str, Any]) -> Any:
        replays.append(digest)
        return real(data, digest, spec)

    monkeypatch.setattr(proven, "_read", counted)
    return replays


class TestTheSitemapsPositionsAreStored:
    def test_the_listing_is_the_records(self, tmp_path: Path, store: Path) -> None:
        _write(tmp_path, _record([ONE, LOSES, BOTH]))
        record = proven.load(tmp_path, DIGEST, SPEC)
        assert record is not None
        assert proven.listed(tmp_path, DIGEST, SPEC) == (
            record.made, tuple(one.id for one in record.positions))

    def test_a_restart_reads_it_back_without_replaying(
        self, tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, _record([ONE, LOSES, BOTH]))
        # The library listing's count stores the listing beside it.
        assert proven.offered(tmp_path, DIGEST, SPEC) == 3
        expected = proven.listed(tmp_path, DIGEST, SPEC)
        replays = _restart(monkeypatch, store)
        assert proven.listed(tmp_path, DIGEST, SPEC) == expected and replays == []

    def test_a_record_rewritten_in_process_is_replayed(
        self, tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        where = _write(tmp_path, _record([ONE, LOSES, BOTH]))
        assert proven.listed(tmp_path, DIGEST, SPEC) is not None
        held = where.stat()
        _write(tmp_path, _record([ONE, BOTH]))
        os.utime(where, ns=(held.st_atime_ns, held.st_mtime_ns + 10**9))
        found = proven.listed(tmp_path, DIGEST, SPEC)
        assert found is not None and len(found[1]) == 2
        replays = _restart(monkeypatch, store)
        assert proven.listed(tmp_path, DIGEST, SPEC) == found and replays == []

    def test_an_entry_from_before_the_listing_was_stored_replays(
        self, tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, _record([ONE, LOSES, BOTH]))
        expected = proven.listed(tmp_path, DIGEST, SPEC)
        held = json.loads(store.read_text(encoding="utf-8"))
        for entry in held["proven"].values():
            del entry["listed"]
        store.write_text(json.dumps(held), encoding="utf-8")
        replays = _restart(monkeypatch, store)
        assert proven.listed(tmp_path, DIGEST, SPEC) == expected and replays == [DIGEST]

    def test_no_record_and_an_empty_record_list_nothing(self, tmp_path: Path, store: Path) -> None:
        assert proven.listed(tmp_path, DIGEST, SPEC) is None
        _write(tmp_path, _record([]))
        assert proven.listed(tmp_path, DIGEST, SPEC) is None
        assert proven.load(tmp_path, DIGEST, SPEC) is None
