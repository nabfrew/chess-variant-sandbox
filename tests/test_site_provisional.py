"""The provisional store: a score a visitor's browser measured, held while the server checks it.

Q312, the first row of `docs/archive/plan-client-compute.md` §8 after the probe. Its acceptance, and
what each class below holds:

* a table and `runs/provisional/` -- `TestTheRecordsLiveUnderProvisional`;
* the two new basis fields -- `TestTheTwoBasisFields`;
* the five states and the refusal by fingerprint -- `TestTheFiveStates`,
  `TestTheRefusalByFingerprint`;
* the page's two labels -- `TestThePageLabels`;
* a provisional result shows only under a matching basis and appears in no sort, tag or
  canonical file -- `TestItShowsOnlyUnderAMatchingBasis`, `TestItRanksNothing`;
* a `legal` result shows to its designer alone, and under the user's answer A a
  spot-checked one shows to everybody, labelled -- `TestWhoSeesIt`, `TestTheSitePayloads`.

The store's tests need a real Postgres and skip with a reason without one; `test_site.py`'s
docstring says how to start a scratch cluster. The basis and page tests need neither.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import node_session
import pytest
import site_store

from chess_variant_sandbox import corpus, measurement
from chess_variant_sandbox.measurement import (
    BROWSER,
    LEGAL,
    Basis,
    sample_check,
)
from chess_variant_sandbox.site import admin, awards, browse, db, provisional
from chess_variant_sandbox.site.provisional import ProvisionalError
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.spec.canonical import load_spec

REPOSITORY = Path(__file__).resolve().parents[1]
# A small board, so compiling its tables for the basis costs nothing worth timing.
RULESET = REPOSITORY / "rulesets" / "dobutsu.json"

# What a browser's measurement sends: `slices.measure_in_slices`' four files. The text is
# a stand-in; nothing here reads a record, which is the tiers' business.
FILES = {
    "games.jsonl": '{"seed": 0, "moves": ["a1a2"]}\n',
    "tournament.json": '{"games": 1}\n',
    "analysis.json": '{"games": []}\n',
    "index-v11.json": '{"total": 0.61}\n',
}


def _spec() -> dict[str, Any]:
    return load_spec(RULESET)


def _now(spec: dict[str, Any] | None = None) -> Basis:
    """The server's basis for this game's one-run measurement, as it would stamp its own."""
    body = spec or _spec()
    return corpus.current_basis(spec_hash(body), spec=body, seeds=corpus.ONE_SEED)


def _sent(basis: Basis) -> dict[str, Any]:
    """What a browser sends for this basis: the same stamp, played in a browser."""
    return basis.in_browser().as_dict()


def _one_run(digest: str, spec: Any) -> Basis | None:
    return corpus.current_basis(digest, spec=spec, seeds=corpus.ONE_SEED)


# ------------------------------------------------------------------ the two basis fields


class TestTheTwoBasisFields:
    def test_a_server_stamp_is_written_exactly_as_before(self) -> None:
        """Neither field is written for the server's own measurement, so no stored file,
        and no `canonical.json` the loader copies, changes shape."""
        stamp = _now().as_dict()
        assert "played" not in stamp and "checked" not in stamp
        assert set(stamp) == {
            "ruleset_hash", "weights_version", "agents", "games_per_pairing", "seeds",
            "engine", "metrics", "tables", "analysis", "verdict_rule",
            # Q631: what the core is handed, and the agents' source for where it cannot
            # be read. Every server stamp gains both at the replay Q631 owes.
            "players", "agents_source",
        }
        assert Basis.from_dict(stamp) == _now()

    def test_a_browser_stamp_round_trips_at_every_level_of_checking(self) -> None:
        for checked in ("", LEGAL, sample_check(32), sample_check(1)):
            stamp = _now().in_browser(checked)
            written = stamp.as_dict()
            assert written["played"] == BROWSER and written["checked"] == checked
            assert Basis.from_dict(json.loads(json.dumps(written))) == stamp

    def test_a_pair_this_module_does_not_write_is_no_basis(self) -> None:
        """Unreadable is `None`, which every reader already treats as *no number*."""
        base = _now().as_dict()
        for played, checked in (
            ("server", ""), ("phone", LEGAL), ("", LEGAL), ("", "sample:32"),
            (BROWSER, "sample"), (BROWSER, "sample:0"), (BROWSER, "sample:032"),
            (BROWSER, "sample:-3"), (BROWSER, "sampled:32"), (BROWSER, "everything"),
        ):
            assert Basis.from_dict({**base, "played": played, "checked": checked}) is None, (
                played, checked,
            )

    def test_a_sample_names_a_positive_count(self) -> None:
        assert sample_check(32) == "sample:32"
        assert measurement.read_check("sample:32") == ("sample", 32)
        assert measurement.read_check(LEGAL) == (LEGAL, 0)
        for bad in (0, -1, True, 3.0):
            with pytest.raises(ValueError):
                sample_check(bad)  # type: ignore[arg-type]

    def test_a_browser_stamp_is_never_current_against_the_servers(self) -> None:
        """`played` is compared: an official reader shows no number for a browser's games."""
        server = _now()
        for checked in (LEGAL, sample_check(32)):
            found = server.in_browser(checked).stale_against(server)
            assert found.needs == "replay"
            assert "played in a visitor's browser, not by the server" in found.reasons
        backwards = server.stale_against(server.in_browser(LEGAL))
        assert backwards.needs == "replay"

    def test_how_far_it_was_checked_moves_no_staleness(self) -> None:
        """`checked` decides who sees a number and its label, never whether it holds."""
        server = _now()
        stored = server.in_browser(LEGAL)
        assert stored.stale_against(server.in_browser(sample_check(32))).current
        assert stored.stale_against(server.in_browser()).current

    def test_an_official_reader_shows_no_number_for_a_browser_stamp(
        self, tmp_path: Path
    ) -> None:
        """The second door: even a browser's stamp in a canonical file reads as stale."""
        path = tmp_path / "canonical.json"
        path.write_text(
            json.dumps({"runs": [], "basis": _now().in_browser(sample_check(32)).as_dict()}),
            encoding="utf-8",
        )
        found = measurement.load(path, _now())
        assert found is not None and not found.current
        assert "played in a visitor's browser" in found.staleness.summary()

    def test_the_measurement_module_moves_no_fingerprint(self) -> None:
        """The fields live in `measurement.py`, which no fingerprinted set reads, so adding
        them stales no stored score (`FROZEN_METRICS` and `FROZEN_ANALYSIS` in
        `test_measurement.py` pin the sets' values)."""
        itself = (REPOSITORY / "chess_variant_sandbox" / "measurement.py").resolve()
        for kind in ("engine", "metrics", "analysis"):
            assert itself not in {path.resolve() for path in measurement.fingerprinted(kind)}


# ----------------------------------------------------------- the refusal by fingerprint


class TestTheRefusalByFingerprint:
    def test_a_stamp_made_from_the_same_sources_is_taken(self) -> None:
        assert provisional.refusal(_sent(_now()), _now()) is None

    def test_an_old_cached_module_is_refused_by_its_build(self) -> None:
        now = _now()
        sources, _, build = now.engine.rpartition("-")
        old = replace(now, engine=f"{sources}-0123456789abcdef")
        said = provisional.refusal(_sent(old), now)
        assert said is not None
        assert "build 0123456789abcdef" in said and f"this server runs {build}" in said
        assert "fetches the current module" in said

    def test_every_other_fingerprint_is_refused_with_its_reason(self) -> None:
        now = _now()
        cases = {
            "the metrics changed": replace(now, metrics="0" * 16),
            "the analysis pass changed": replace(now, analysis="0" * 16),
            "compiles to different tables": replace(now, tables="0" * 16),
            "scored under weights v1": replace(now, weights_version="v1"),
            "the core, the schedule or the game loop": replace(
                now, engine="0" * 16 + "-" + now.engine.rpartition("-")[2]
            ),
            "not yet measured on seed 0": replace(now, seeds=(1000,)),
        }
        for reason, stamp in cases.items():
            said = provisional.refusal(_sent(stamp), now)
            assert said is not None and reason in said, (reason, said)

    def test_a_stamp_that_cannot_vouch_for_itself_is_refused(self) -> None:
        now = _now()
        assert "no basis the server can read" in str(provisional.refusal(None, now))
        assert "no basis the server can read" in str(provisional.refusal({"x": 1}, now))
        assert "does not say a browser played" in str(
            provisional.refusal(now.as_dict(), now)
        )
        assert "only the server says" in str(
            provisional.refusal(now.in_browser(sample_check(32)).as_dict(), now)
        )


# ---------------------------------------------------------------------------- the store


@dataclass
class Store:
    connection: Any
    runs: Path
    game: str
    spec: dict[str, Any]
    designer: int
    other: int
    url: str
    schema: str

    def submit(self, **overrides: Any) -> provisional.Result:
        now = _now(self.spec)
        values: dict[str, Any] = {
            "game": self.game,
            "submitter": self.designer,
            "submitted": _sent(now),
            "current": now,
            "headline": 0.6123,
            "viable": True,
            "files": FILES,
        }
        values.update(overrides)
        found = provisional.record(self.connection, self.runs, **values)
        self.connection.commit()
        return found

    def rows(self) -> list[tuple[int, str, str | None]]:
        with self.connection.cursor() as cursor:
            cursor.execute("select id, state, directory from provisional order by id")
            return [(int(r[0]), str(r[1]), r[2]) for r in cursor.fetchall()]

    def shown(self, viewer: int | None, current: Any = _one_run) -> dict[str, Any] | None:
        return provisional.shown(self.connection, [self.game], viewer, current).get(self.game)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    url, schema = site_store.emptied("no database to keep provisional results in")
    connection = db.connect(url)
    spec = _spec()
    digest = spec_hash(spec)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {schema}")
            ids = []
            for handle in ("ada", "bo"):
                cursor.execute(
                    "insert into users (handle, email) values (%s, %s) returning id",
                    (handle, f"{handle}@example.org"),
                )
                ids.append(int(cursor.fetchone()[0]))
            cursor.execute(
                "insert into games (hash, spec, name, owner, visibility, origin)"
                " values (%s, %s::jsonb, %s, %s, 'public', 'designed')",
                (digest, json.dumps(spec), spec["name"], ids[0]),
            )
        connection.commit()
        (tmp_path / "runs").mkdir()
        yield Store(connection, tmp_path / "runs", digest, spec, ids[0], ids[1], url, schema)
    finally:
        connection.rollback()
        connection.close()


class TestTheRecordsLiveUnderProvisional:
    def test_a_recorded_result_is_legal_with_its_records_under_provisional(
        self, store: Store
    ) -> None:
        found = store.submit()
        assert found.state == LEGAL
        assert found.basis == _now(store.spec).in_browser(LEGAL)
        assert found.directory == f"provisional/{found.id}"
        where = store.runs / "provisional" / str(found.id)
        assert {path.name for path in where.iterdir()} == set(FILES)
        for name, text in FILES.items():
            assert (where / name).read_bytes() == text.encode("utf-8")
        # Nothing else under the runs root: no `<hash>/`, no `<hash>-<seed>/`.
        assert [path.name for path in store.runs.iterdir()] == ["provisional"]
        assert not list(store.runs.rglob("canonical.json"))

    def test_the_stored_stamp_is_the_servers(self, store: Store) -> None:
        """The number and the stamp are the server's; the browser's are only checked."""
        found = store.submit()
        with store.connection.cursor() as cursor:
            cursor.execute("select basis, headline, viable from provisional")
            basis, headline, viable = cursor.fetchone()
        assert basis == _now(store.spec).in_browser(LEGAL).as_dict()
        assert (headline, viable) == (0.6123, True)
        assert found.headline == 0.6123

    def test_a_result_carries_the_four_files_and_never_a_canonical_one(
        self, store: Store
    ) -> None:
        for files in (
            {**FILES, "canonical.json": "{}"},
            {**FILES, "../games.jsonl": ""},
            {**FILES, "runs/x.json": ""},
        ):
            with pytest.raises(ProvisionalError, match="arrived incomplete"):
                store.submit(files=files)
        without = {k: v for k, v in FILES.items() if k != "games.jsonl"}
        with pytest.raises(ProvisionalError, match="without its games"):
            store.submit(files=without)
        store.connection.rollback()
        assert store.rows() == []
        assert not (store.runs / "provisional").exists()

    def test_only_the_designer_sends_a_result(self, store: Store) -> None:
        with pytest.raises(ProvisionalError, match="only the person who made a game"):
            store.submit(submitter=store.other)
        store.connection.rollback()
        with store.connection.cursor() as cursor:
            cursor.execute("update games set owner = null where hash = %s", (store.game,))
        with pytest.raises(ProvisionalError, match="measured by the server alone"):
            store.submit()
        store.connection.rollback()
        with store.connection.cursor() as cursor:
            cursor.execute("update games set revoked = now() where hash = %s", (store.game,))
        with pytest.raises(ProvisionalError, match="taken off the site"):
            store.submit()
        store.connection.rollback()
        assert store.rows() == []

    def test_a_newer_result_supersedes_and_the_sweep_takes_its_records(
        self, store: Store
    ) -> None:
        first = store.submit()
        second = store.submit(headline=0.7)
        assert store.rows() == [
            (first.id, provisional.SUPERSEDED, f"provisional/{first.id}"),
            (second.id, LEGAL, f"provisional/{second.id}"),
        ]
        # Still on disk until the sweep, which runs after the commit.
        assert (store.runs / "provisional" / str(first.id)).is_dir()
        assert provisional.sweep(store.connection, store.runs) == [f"provisional/{first.id}"]
        store.connection.commit()
        assert not (store.runs / "provisional" / str(first.id)).exists()
        assert (store.runs / "provisional" / str(second.id)).is_dir()
        assert store.rows()[0] == (first.id, provisional.SUPERSEDED, None)
        assert provisional.sweep(store.connection, store.runs) == []

    def test_a_confirmed_result_loses_its_records_and_a_mismatch_keeps_them(
        self, store: Store
    ) -> None:
        kept = store.submit()
        provisional.mismatched(store.connection, kept.id)
        store.connection.commit()
        confirmed = store.submit()
        provisional.confirmed(store.connection, confirmed.id)
        store.connection.commit()
        assert provisional.sweep(store.connection, store.runs) == [
            f"provisional/{confirmed.id}"
        ]
        store.connection.commit()
        assert (store.runs / "provisional" / str(kept.id)).is_dir()
        assert not (store.runs / "provisional" / str(confirmed.id)).exists()


class TestTheFiveStates:
    def test_the_states_are_section_fives(self) -> None:
        assert provisional.STATES == (
            "legal", "spot-checked", "confirmed", "mismatch", "superseded"
        )

    def test_a_result_moves_only_forward(self, store: Store) -> None:
        found = store.submit()
        checked = provisional.spot_checked(store.connection, found.id, 32)
        assert checked.state == provisional.SPOT_CHECKED
        assert checked.basis.checked == "sample:32" and checked.sample == 32
        assert checked.basis == found.basis.in_browser("sample:32")
        with pytest.raises(ProvisionalError, match="spot-checked cannot become spot-checked"):
            provisional.spot_checked(store.connection, found.id, 64)
        done = provisional.confirmed(store.connection, found.id)
        assert done.state == provisional.CONFIRMED
        for move in (provisional.mismatched, provisional.confirmed):
            with pytest.raises(ProvisionalError, match="confirmed cannot become"):
                move(store.connection, found.id)
        with pytest.raises(ProvisionalError, match="no provisional result numbered"):
            provisional.confirmed(store.connection, 10**9)

    def test_an_ended_result_never_comes_back(self, store: Store) -> None:
        first = store.submit()
        provisional.mismatched(store.connection, first.id)
        store.connection.commit()
        second = store.submit()
        store.submit()
        for ended in (first.id, second.id):
            for move in (provisional.confirmed, provisional.mismatched):
                with pytest.raises(ProvisionalError, match="cannot become"):
                    move(store.connection, ended)
            with pytest.raises(ProvisionalError, match="cannot become"):
                provisional.spot_checked(store.connection, ended, 32)

    def test_the_table_ties_the_state_to_its_stamp(self, store: Store) -> None:
        """A row cannot say more than its stamp, so a label cannot outrun the server."""
        import psycopg

        now = _now(store.spec)
        cases = (
            ("spot-checked", now.in_browser(LEGAL)),
            ("legal", now.in_browser(sample_check(32))),
            ("pending", now.in_browser(LEGAL)),
            ("legal", now),
            ("legal", now.in_browser("")),
        )
        for state, stamp in cases:
            with pytest.raises(psycopg.errors.CheckViolation), store.connection.cursor() as c:
                c.execute(
                    "insert into provisional (game, submitter, basis, state)"
                    " values (%s, %s, %s::jsonb, %s)",
                    (store.game, store.designer, json.dumps(stamp.as_dict()), state),
                )
            store.connection.rollback()
        for where in ("dobutsu-0", "provisional/../x", "provisional/"):
            with pytest.raises(psycopg.errors.CheckViolation), store.connection.cursor() as c:
                c.execute(
                    "insert into provisional (game, submitter, basis, state, directory)"
                    " values (%s, %s, %s::jsonb, 'legal', %s)",
                    (store.game, store.designer, json.dumps(now.in_browser(LEGAL).as_dict()),
                     where),
                )
            store.connection.rollback()

    def test_a_game_has_one_standing_result(self, store: Store) -> None:
        import psycopg

        store.submit()
        with pytest.raises(psycopg.errors.UniqueViolation), store.connection.cursor() as c:
            c.execute(
                "insert into provisional (game, submitter, basis, state)"
                " values (%s, %s, %s::jsonb, 'legal')",
                (store.game, store.designer,
                 json.dumps(_now(store.spec).in_browser(LEGAL).as_dict())),
            )
        store.connection.rollback()

    def test_a_stale_build_is_refused_and_leaves_nothing(self, store: Store) -> None:
        now = _now(store.spec)
        old = replace(now, engine=now.engine.rpartition("-")[0] + "-0123456789abcdef")
        # In a visitor's words (D37); which build it was goes to the server's log.
        with pytest.raises(ProvisionalError, match="changed how it scores games"):
            store.submit(submitted=_sent(old))
        store.connection.rollback()
        assert store.rows() == []
        assert not (store.runs / "provisional").exists()


# --------------------------------------------------------------------- who sees it, where


class TestWhoSeesIt:
    def test_a_legal_result_shows_to_its_designer_alone(self, store: Store) -> None:
        store.submit()
        mine = store.shown(store.designer)
        assert mine == {
            "index": 0.6123, "viable": True, "state": LEGAL, "sample": None,
            "yours": True, "seeds": [0], "weights": corpus.DEFAULT_WEIGHTS,
        }
        assert store.shown(store.other) is None
        assert store.shown(None) is None

    def test_a_spot_checked_result_shows_to_everybody_labelled(self, store: Store) -> None:
        """The user's answer A, kept on 2026-09-25 after T2."""
        found = store.submit()
        provisional.spot_checked(store.connection, found.id, 32)
        store.connection.commit()
        for viewer, yours in ((store.designer, True), (store.other, False), (None, False)):
            seen = store.shown(viewer)
            assert seen is not None, viewer
            assert (seen["state"], seen["sample"], seen["yours"]) == (
                provisional.SPOT_CHECKED, 32, yours,
            )

    def test_an_ended_result_shows_to_nobody(self, store: Store) -> None:
        for end in (provisional.mismatched, provisional.confirmed):
            found = store.submit()
            end(store.connection, found.id)
            store.connection.commit()
            for viewer in (store.designer, store.other, None):
                assert store.shown(viewer) is None, (end, viewer)
        store.submit()
        store.submit()
        assert store.shown(store.designer)["state"] == LEGAL

    def test_a_killed_game_shows_nothing(self, store: Store) -> None:
        found = store.submit()
        provisional.spot_checked(store.connection, found.id, 32)
        with store.connection.cursor() as cursor:
            cursor.execute("update games set revoked = now() where hash = %s", (store.game,))
        assert store.shown(store.designer) is None

    def test_deleting_a_game_supersedes_its_result_and_the_sweep_takes_the_records(
        self, store: Store
    ) -> None:
        found = store.submit()
        provisional.spot_checked(store.connection, found.id, 32)
        with store.connection.cursor() as cursor:
            cursor.execute("update games set retired = now() where hash = %s", (store.game,))
        admin.delete_game(store.connection, store.game, store.designer)
        store.connection.commit()
        assert store.rows() == [(found.id, provisional.SUPERSEDED, f"provisional/{found.id}")]
        assert provisional.sweep(store.connection, store.runs) == [f"provisional/{found.id}"]
        store.connection.commit()
        assert not (store.runs / "provisional" / str(found.id)).exists()


class TestItShowsOnlyUnderAMatchingBasis:
    def test_a_moved_basis_hides_it_from_everybody(self, store: Store) -> None:
        found = store.submit()
        provisional.spot_checked(store.connection, found.id, 32)
        store.connection.commit()
        assert store.shown(store.other) is not None
        moved = {
            "weights": lambda b: replace(b, weights_version="v99"),
            "core": lambda b: replace(b, engine=b.engine.rpartition("-")[0] + "-feedface"),
            "metrics": lambda b: replace(b, metrics="0" * 16),
            "tables": lambda b: replace(b, tables="0" * 16),
            "roster": lambda b: replace(b, agents=b.agents[:-1]),
        }
        for why, change in moved.items():
            def later(digest: str, spec: Any, change: Any = change) -> Basis:
                return change(_one_run(digest, spec))

            for viewer in (store.designer, store.other, None):
                assert store.shown(viewer, later) is None, (why, viewer)

    def test_no_rules_to_compile_is_no_basis_and_no_number(self, store: Store) -> None:
        store.submit()
        assert store.shown(store.designer, lambda digest, spec: None) is None


class TestItRanksNothing:
    def test_it_writes_no_measurement_and_no_canonical_file(self, store: Store) -> None:
        found = store.submit()
        provisional.spot_checked(store.connection, found.id, 32)
        store.connection.commit()
        with store.connection.cursor() as cursor:
            cursor.execute("select count(*) from measurements")
            assert cursor.fetchone()[0] == 0
        # The score badges read `measurements` alone, and find nothing to quote.
        assert awards.score_evidence(store.connection, store.game) is None
        assert not list(store.runs.rglob("canonical.json"))
        assert not list(store.runs.glob(f"{store.game}*"))

    def test_the_index_sort_never_reads_it(self) -> None:
        """A card carries the number under `provisional`, never `index`, so it sorts as a
        game with no score: last, in both directions (`browse`)."""
        cards = [
            {"hash": "a", "name": "a", "index": None,
             "provisional": {"index": 0.99, "state": "spot-checked"}},
            {"hash": "b", "name": "b", "index": 0.40},
            {"hash": "c", "name": "c", "index": 0.50},
        ]
        ordered = [card["hash"] for card in browse.order(cards, browse.INDEX)]
        assert ordered[-1] == "a"
        assert ordered[:2] == ["c", "b"]


# ----------------------------------------------------------------- the site's payloads


@pytest.fixture
def site(store: Store, tmp_path: Path) -> Iterator[tuple[SiteApp, Store]]:
    """The site over a library of one game, which is the store's."""
    for name in ("rulesets", "archive"):
        (tmp_path / name).mkdir()
    shutil.copy(RULESET, tmp_path / "rulesets" / RULESET.name)

    def open_one() -> Any:
        connection = db.connect(store.url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {store.schema}")
        return connection

    yield SiteApp(
        tmp_path / "rulesets", store.runs, archive_dir=tmp_path / "archive", connect=open_one
    ), store


class TestTheSitePayloads:
    def _as(self, app: SiteApp, viewer: int | None) -> tuple[Any, Any]:
        app.viewer = viewer
        listing = app.get("/api/library")
        card = next(g for g in listing["games"] if g["hash"] == spec_hash(_spec()))
        return card, app._standing(card["hash"])

    def test_a_legal_result_reaches_its_designer_on_the_card_and_the_page(
        self, site: tuple[SiteApp, Store]
    ) -> None:
        app, store = site
        store.submit()
        card, page = self._as(app, store.designer)
        assert card["provisional"]["state"] == LEGAL and card["provisional"]["yours"]
        assert page["provisional"] == card["provisional"]
        # Its own key, never the score's.
        assert card["index"] is None
        for viewer in (store.other, None):
            card, page = self._as(app, viewer)
            assert card["provisional"] is None and page["provisional"] is None

    def test_a_spot_checked_result_reaches_everybody(
        self, site: tuple[SiteApp, Store]
    ) -> None:
        app, store = site
        found = store.submit()
        provisional.spot_checked(store.connection, found.id, 32)
        store.connection.commit()
        for viewer in (store.designer, store.other, None):
            card, page = self._as(app, viewer)
            assert card["provisional"]["state"] == provisional.SPOT_CHECKED, viewer
            assert page["provisional"] == card["provisional"]
            assert card["index"] is None
        app.viewer = None
        ranked = app.get("/api/library", {"sort": browse.INDEX})
        assert all(game.get("index") is None for game in ranked["games"])


# ------------------------------------------------------------------ the page's two labels


def _in_page(expression: str) -> Any:
    """Run an expression against the page's own functions in node, over a stub DOM."""
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      global.document = { getElementById: () => el, querySelector: () => null,
                          querySelectorAll: () => [], addEventListener: () => {},
                          createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    body = stub + script.group(1) + "\nconsole.log(JSON.stringify(" + expression + "));"
    done = node_session.run(body)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _game(provisional_result: dict[str, Any] | None, **official: Any) -> dict[str, Any]:
    return {"index": None, "seeds": None, "staleness": None, "designIndex": None,
            "designRun": None, "designSeeds": None, "provisional": provisional_result,
            **official}


LEGAL_RESULT = {"index": 0.6123, "viable": True, "state": "legal", "sample": None,
                "yours": True, "seeds": [0], "weights": "v11"}
CHECKED = {**LEGAL_RESULT, "state": "spot-checked", "sample": 32, "yours": False}


class TestThePageLabels:
    def test_a_legal_result_says_not_yet_confirmed(self) -> None:
        drawn = _in_page(f"scoreTag({json.dumps(_game(LEGAL_RESULT))})")
        assert 'class="tag provisional"' in drawn
        assert '0.612 not yet confirmed <span class="muted">1 run</span>' in drawn
        assert "Measured on your computer, not yet confirmed, as the one run" in drawn
        assert "Only you can see this number" in drawn

    def test_a_spot_checked_result_says_so_and_whose_machine(self) -> None:
        drawn = _in_page(f"scoreTag({json.dumps(_game(CHECKED))})")
        assert '0.612 spot-checked <span class="muted">1 run</span>' in drawn
        assert "Measured on its designer&#39;s computer, spot-checked by the site" in drawn \
            or "Measured on its designer's computer, spot-checked by the site" in drawn
        assert "replayed 32 of its games" in drawn
        assert "not an official score" in drawn
        mine = _in_page(f"scoreTag({json.dumps(_game({**CHECKED, 'yours': True}))})")
        assert "Measured on your computer, spot-checked by the site" in mine

    def test_it_never_borrows_the_score_pill(self) -> None:
        for result in (LEGAL_RESULT, CHECKED):
            drawn = _in_page(f"scoreTag({json.dumps(_game(result))})")
            assert "tag score" not in drawn
        ruled = _in_page(f"scoreTag({json.dumps(_game({**CHECKED, 'viable': False}))})")
        assert 'class="tag gated provisional"' in ruled and "ruled out, spot-checked" in ruled

    def test_a_current_official_score_always_wins(self) -> None:
        official = _game(CHECKED, index=0.7, seeds=[0, 1000, 2000, 3000], viable=True,
                         weights="v11", staleness={"current": True})
        drawn = _in_page(f"scoreTag({json.dumps(official)})")
        assert "tag score" in drawn and "spot-checked" not in drawn

    def test_a_stale_official_score_gives_way_to_a_current_provisional_one(self) -> None:
        stale = _game(CHECKED, staleness={"current": False, "summary": "stale (re-score)"})
        drawn = _in_page(f"scoreTag({json.dumps(stale)})")
        assert "spot-checked" in drawn and "out of date" not in drawn

    def test_the_game_page_draws_it_under_its_label(self) -> None:
        drawn = _in_page(f"provisionalHtml({json.dumps(_game(CHECKED))})")
        assert "measured on its designer" in drawn and "spot-checked by the site" in drawn
        assert 'data-gloss="provisional"' in drawn
        assert "one run" in drawn and "scoring rules version 11" in drawn
        official = {**_game(CHECKED), "score": {"total": 0.7}, "staleness": {"current": True}}
        assert _in_page(f"provisionalHtml({json.dumps(official)})") == ""
        assert _in_page(f"provisionalHtml({json.dumps(_game(None))})") == ""
