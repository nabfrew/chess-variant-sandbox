"""The repository's own data as rows, loaded by hash and loaded once.

Task 1.6 of `docs/plan-website.md`. Three properties carry the whole row. A second run
writes nothing at all, which is what makes the loader safe to put in a deploy. A stale or
missing measurement is loaded as what it is and never as a number. And the maintainer's
local-application games land on their profile as history, rated by nothing and clearing no
*Unbeaten* tag.

The store tests need a real Postgres and skip with a reason without one, each on
the session's migrated store (`tests/site_store.py`), emptied first -- a real server rather than
`py-pglite`, which does not work on Windows (task 1.2). The fixture tree is built under `tmp_path` out of `rulesets/dull.json` with its
ply limit moved, which is a different game and a different hash each time, so the specs
are real ones and nothing is committed twice.

Nothing here works out a legal move, an outcome or a score. The files say what they say
and the loader copies them.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_store

from chess_variant_sandbox import corpus
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.site import admin, awards, badges
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library

DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))

TOP_AGENT = "minimax:nodes=400"


def _dull(max_plies: int, name: str) -> dict[str, Any]:
    """`dull` with its ply limit moved, which is a different game and a different hash."""
    spec = json.loads(json.dumps(DULL))
    spec["rules"]["max_plies"] = max_plies
    spec["name"] = name
    return spec


# Three games: one committed with a measurement and a ladder, one committed with a
# measurement nothing can read, one archived with neither.
RATED = _dull(2, "rated")
STALE = _dull(4, "stale")
BARE = _dull(6, "bare")
RATED_HASH = spec_hash(RATED)
STALE_HASH = spec_hash(STALE)
BARE_HASH = spec_hash(BARE)

LADDER = {
    "ruleset_hash": RATED_HASH,
    "games_per_pairing": 2,
    "seed": 0,
    "standings": [
        {"name": TOP_AGENT, "rating": 300.0, "games": 24},
        {"name": "greedy", "rating": 0.0, "games": 24},
        {"name": "random", "rating": -300.0, "games": 24},
    ],
}

HEADLINE = 0.6123

# Both ends of the resampled interval, which `index-<weights>.json` holds beside the
# summary rather than in it. `0008_awards.sql` brought them into the row, so that a badge
# testing a game's whole range need not open a file inside a vote's transaction.
CI_LOW = 0.5804
CI_HIGH = 0.6455

# Two games the maintainer played in the local application. One was won outright against
# the ladder's top rung and the other was begun and never finished. The win is the row
# that would clear *Unbeaten* if anything counted it, which is what these tests deny.
WON = {
    "session": "aWonGame01",
    "ruleset_hash": RATED_HASH,
    "ruleset_name": "rated",
    "opponent": TOP_AGENT,
    "human_seat": 0,
    "seed": 0,
    "started": 1788027650.442,
    "plies": 34,
    "result": 1.0,
    "winner": 0,
    "reason": "checkmate",
    "finished": 1788028650.117,
}
ABANDONED = {
    "session": "anOpenGame",
    "ruleset_hash": STALE_HASH,
    "ruleset_name": "stale",
    "opponent": "greedy",
    "human_seat": 1,
    "seed": 0,
    "started": 1788027720.009,
    "plies": 0,
    "result": None,
    "winner": None,
    "reason": "",
    "finished": None,
}


# ------------------------------------------------------------------------ the fixtures


def _canonical(digest: str, spec: dict[str, Any], weights: str) -> dict[str, Any]:
    """One `canonical.json`, with the basis a measurement taken here would carry."""
    basis = corpus.current_basis(digest, DEFAULT_WEIGHTS, spec).as_dict()
    basis["weights_version"] = weights
    return {
        "hash": digest,
        "name": str(spec["name"]),
        "index": HEADLINE,
        "weights_version": weights,
        "standings": [{"name": "random", "rating": -300.0, "games": 64}],
        "seeds": [0],
        "totals_used": 1,
        "runs": [{"seed": 0, "index": HEADLINE, "sd": 0.01, "viable": True,
                  "violations": [], "unreadable": [], "origin": "played"}],
        "basis": basis,
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A checkout: three rulesets, two summaries, one ladder, two human games."""
    rulesets = tmp_path / "rulesets"
    archive = tmp_path / "archive"
    runs = tmp_path / "runs"
    for directory in (rulesets, archive, runs / "opponents", runs / "human"):
        directory.mkdir(parents=True)

    (rulesets / "rated.json").write_text(json.dumps(RATED), encoding="utf-8")
    (rulesets / "stale.json").write_text(json.dumps(STALE), encoding="utf-8")
    (archive / "bare.json").write_text(json.dumps(BARE), encoding="utf-8")
    # A README sits in both directories in the real repository, and a loader that
    # stopped on one of them would ship nothing.
    (archive / "README.md").write_text("not a spec", encoding="utf-8")

    for digest, spec, weights in (
        (RATED_HASH, RATED, DEFAULT_WEIGHTS),
        # A weights version nothing here scores under, so this one reads stale by its
        # own basis rather than by anything the test asserted about the code.
        (STALE_HASH, STALE, "v1"),
    ):
        directory = runs / digest
        directory.mkdir()
        (directory / "canonical.json").write_text(
            json.dumps(_canonical(digest, spec, weights)), encoding="utf-8"
        )
        # The scoring result beside the summary. Cut down to the interval, which is the
        # only part of it the loader reads.
        (directory / f"index-{weights}.json").write_text(
            json.dumps({"interval": {"ci_low": CI_LOW, "ci_high": CI_HIGH}}),
            encoding="utf-8",
        )

    (runs / "opponents" / f"{RATED_HASH}.json").write_text(
        json.dumps(LADDER), encoding="utf-8"
    )
    _write_human(tmp_path, [WON, ABANDONED])
    return tmp_path


def _write_human(root: Path, games: list[dict[str, Any]]) -> None:
    lines = "\n".join(json.dumps(game) for game in games) + "\n"
    (root / "runs" / "human" / "games.jsonl").write_text(lines, encoding="utf-8")


class Scratch:
    """A store with the schema in it, on a schema of its own."""

    def __init__(self, url: str, schema: str) -> None:
        self.url = url
        self.schema = schema
        self.connection = self.open()

    def open(self) -> Any:
        connection = connect(self.url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {self.schema}")
        return connection

    def count(self, table: str) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(f"select count(*) from {table}")
            return int(cursor.fetchone()[0])

    def snapshot(self) -> dict[str, list[Any]]:
        """Every row of every table the loader touches, timestamps included.

        The strict reading of *a second run changes zero rows*, taken over the rows
        themselves rather than over a count or the loader's own report. An `updated` that
        moved would show here even though the values it sits beside did not.
        """
        found: dict[str, list[Any]] = {}
        for table in ("users", "games", "measurements", "ratings", "matches"):
            with self.connection.cursor() as cursor:
                cursor.execute(f"select * from {table} order by 1")
                found[table] = cursor.fetchall()
        return found


@contextlib.contextmanager
def _scratch() -> Iterator[Scratch]:
    url, name = site_store.emptied("no database to load into")
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
    with _scratch() as scratch:
        yield scratch


def _load(scratch: Scratch, root: Path) -> loader.Report:
    report = loader.load(scratch.connection, root)
    scratch.connection.commit()
    return report


def _maintainer(scratch: Scratch) -> int:
    with scratch.connection.cursor() as cursor:
        cursor.execute("select id from users where handle = %s", (loader.MAINTAINER,))
        return int(cursor.fetchone()[0])


def _measurement(scratch: Scratch, digest: str) -> tuple[str, float | None, Any]:
    with scratch.connection.cursor() as cursor:
        cursor.execute(
            "select state, headline, basis from measurements where hash = %s", (digest,)
        )
        found = cursor.fetchone()
    return str(found[0]), found[1], found[2]


def _range(scratch: Scratch, digest: str) -> tuple[float | None, float | None]:
    with scratch.connection.cursor() as cursor:
        cursor.execute(
            "select ci_low, ci_high from measurements where hash = %s", (digest,)
        )
        return tuple(cursor.fetchone())  # type: ignore[return-value]


# ------------------------------------------------------------- reading the files, purely


class TestReadingTheFiles:
    """No database. What the loader makes of a directory and a ladder file."""

    def test_a_directory_of_specs_is_read_by_hash_and_a_readme_is_not_a_spec(
        self, tree: Path
    ) -> None:
        found, problems = loader.read_specs(tree / "archive")
        assert set(found) == {BARE_HASH}
        assert problems == [], "a README is not a *.json and should not be mentioned"

    def test_a_directory_that_is_not_there_is_empty_rather_than_a_refusal(
        self, tmp_path: Path
    ) -> None:
        """`archive/` is optional in a trimmed checkout, and a deploy must not stop on it."""
        found, problems = loader.read_specs(tmp_path / "nowhere")
        assert (found, problems) == ({}, [])

    def test_a_file_that_is_not_a_spec_is_named_rather_than_swallowed(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")
        found, problems = loader.read_specs(tmp_path)
        assert found == {}
        assert len(problems) == 1 and "broken.json" in problems[0]

    def test_the_top_rung_is_the_ladders_own_name_and_its_own_rating(self) -> None:
        """Not the seed's number on the site's scale: the column sits beside the file."""
        assert loader.top_rung(LADDER) == (TOP_AGENT, 300.0)

    def test_a_game_with_no_ladder_has_no_top(self, tree: Path) -> None:
        assert loader.read_ladder(tree / "runs", STALE_HASH) is None
        assert loader.top_rung(None) == (None, None)


# -------------------------------------------------------------- the row's own acceptance


class TestASecondRunChangesNothing:
    def test_the_second_load_writes_no_row_and_moves_no_timestamp(
        self, store: Scratch, tree: Path
    ) -> None:
        first = _load(store, tree)
        assert first.written > 0, "the first load should have done something"
        before = store.snapshot()

        second = _load(store, tree)
        assert second.written == 0
        assert (second.games.added, second.games.updated) == (0, 0)
        assert (second.measurements.added, second.measurements.updated) == (0, 0)
        assert (second.ratings.added, second.matches.added) == (0, 0)
        assert store.snapshot() == before

    def test_the_counts_are_what_the_tree_holds(self, store: Scratch, tree: Path) -> None:
        report = _load(store, tree)
        assert report.games.rows == 3, "two committed and one archived"
        # `bare` has neither a summary nor a ladder, so it has no row at all -- a game
        # with no current measurement simply has none.
        assert report.measurements.rows == 2
        assert report.ratings.rows == len(LADDER["standings"])
        assert report.matches.rows == 2
        assert (store.count("games"), store.count("measurements")) == (3, 2)
        assert (store.count("ratings"), store.count("matches")) == (3, 2)

    def test_a_summary_that_appears_later_is_added_and_nothing_else_moves(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        directory = tree / "runs" / BARE_HASH
        directory.mkdir()
        (directory / "canonical.json").write_text(
            json.dumps(_canonical(BARE_HASH, BARE, DEFAULT_WEIGHTS)), encoding="utf-8"
        )
        report = _load(store, tree)
        assert (report.measurements.added, report.measurements.updated) == (1, 0)
        assert report.games.written == 0 and report.matches.written == 0


# ------------------------------------------------------------------------- what a row says


class TestTheLibrary:
    def test_every_game_is_owned_by_the_maintainer_under_its_own_origin(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        owner = _maintainer(store)
        with store.connection.cursor() as cursor:
            cursor.execute("select hash, owner, origin, visibility from games order by hash")
            rows = {row[0]: row[1:] for row in cursor.fetchall()}
        assert rows[RATED_HASH] == (owner, loader.CORPUS, loader.PUBLIC)
        assert rows[BARE_HASH] == (owner, loader.ARCHIVED, loader.PUBLIC)

    def test_the_maintainer_is_made_once_and_cannot_sign_in(
        self, store: Scratch, tree: Path
    ) -> None:
        """A row the loader invented owns the corpus and sits in the seats of games
        already played. Neither is a login, so `password` stays null until 1.3."""
        _load(store, tree)
        _load(store, tree)
        with store.connection.cursor() as cursor:
            cursor.execute("select handle, password, role from users")
            rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0] == (loader.MAINTAINER, None, loader.MAINTAINER_ROLE)

    def test_a_row_the_loader_did_not_write_is_left_exactly_as_it_is(
        self, store: Scratch, tree: Path
    ) -> None:
        """§3: the same rules uploaded twice are one row and the first uploader keeps the
        credit. The loader cannot tell a deliberate name on the site from a stale one in
        the repository, so it does not touch a row it did not write."""
        with store.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin, visibility)"
                " values (%s, %s::jsonb, %s, %s, %s)",
                (RATED_HASH, json.dumps(RATED), "somebody else's name", "designed", "private"),
            )
        store.connection.commit()

        report = _load(store, tree)
        assert report.games.added == 2, "the other two, and not this one"
        with store.connection.cursor() as cursor:
            cursor.execute("select name, origin, owner from games where hash = %s", (RATED_HASH,))
            assert cursor.fetchone() == ("somebody else's name", "designed", None)

    def test_a_renamed_corpus_game_is_corrected_rather_than_duplicated(
        self, store: Scratch, tree: Path
    ) -> None:
        """The filename is the mechanics hash and `name` sits outside it, so a rename in
        the repository is the same row under a different word."""
        _load(store, tree)
        renamed = json.loads(json.dumps(RATED))
        renamed["name"] = "rated, renamed"
        (tree / "rulesets" / "rated.json").write_text(json.dumps(renamed), encoding="utf-8")

        report = _load(store, tree)
        assert (report.games.added, report.games.updated) == (0, 1)
        assert store.count("games") == 3
        with store.connection.cursor() as cursor:
            cursor.execute("select name from games where hash = %s", (RATED_HASH,))
            assert cursor.fetchone()[0] == "rated, renamed"


class TestAGameThatLeavesTheTree:
    """Q287. A spec superseded in the corpus is retired, and the row stays.

    The loader is the only thing that stamps `games.retired`, so these are its four
    claims: it stamps what has gone, it stamps it once, it leaves everything that is not
    a corpus row alone, and it clears the stamp when the spec comes back.
    """

    def _retire(self, store: Scratch, tree: Path) -> loader.Report:
        """Load, take `rated.json` out of the corpus, load again."""
        _load(store, tree)
        (tree / "rulesets" / "rated.json").unlink()
        return _load(store, tree)

    def _retired(self, store: Scratch, digest: str) -> Any:
        with store.connection.cursor() as cursor:
            cursor.execute("select retired from games where hash = %s", (digest,))
            found = cursor.fetchone()
        return None if found is None else found[0]

    def test_a_spec_that_has_left_both_directories_is_stamped_and_named(
        self, store: Scratch, tree: Path
    ) -> None:
        report = self._retire(store, tree)
        assert (report.retired.updated, report.retired.unchanged) == (1, 0)
        assert self._retired(store, RATED_HASH) is not None
        assert any(RATED_HASH in note for note in report.notes)

    def test_the_row_and_everything_that_references_it_stay(
        self, store: Scratch, tree: Path
    ) -> None:
        """Measurements, ratings and the matches already played are what make deleting
        the row impossible; the stamp is what retirement is instead."""
        self._retire(store, tree)
        assert store.count("games") == 3
        with store.connection.cursor() as cursor:
            cursor.execute("select name, spec from games where hash = %s", (RATED_HASH,))
            name, spec = cursor.fetchone()
        assert (name, spec_hash(dict(spec))) == ("rated", RATED_HASH)
        assert r.load(store.connection, RATED_HASH) != {}
        assert m.row(store.connection, str(WON["session"])).game == RATED_HASH

    def test_a_second_load_keeps_the_first_stamp_and_writes_nothing(
        self, store: Scratch, tree: Path
    ) -> None:
        """The column answers *when it left*, not *when the last deploy ran*, so the
        loader's own acceptance holds over a tree with a retirement in it."""
        self._retire(store, tree)
        before = store.snapshot()

        again = _load(store, tree)
        assert again.written == 0
        assert (again.retired.updated, again.retired.unchanged) == (0, 1)
        assert store.snapshot() == before

    def test_a_draft_saved_on_the_site_is_never_retired(
        self, store: Scratch, tree: Path
    ) -> None:
        """A draft has no spec in the tree by definition. Retiring by absence without
        this would take every draft off the listing on the first deploy after it was
        saved."""
        with store.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin, visibility)"
                " values (%s, %s::jsonb, %s, %s, %s)",
                ("d" * 12, json.dumps(BARE), "somebody's draft", "designed", "public"),
            )
        store.connection.commit()

        report = self._retire(store, tree)
        assert report.retired.updated == 1, "the corpus game, and it alone"
        assert self._retired(store, "d" * 12) is None

    def test_a_spec_that_comes_back_under_the_same_hash_is_put_back(
        self, store: Scratch, tree: Path
    ) -> None:
        self._retire(store, tree)
        (tree / "rulesets" / "rated.json").write_text(json.dumps(RATED), encoding="utf-8")

        report = _load(store, tree)
        assert (report.games.added, report.games.updated) == (0, 1)
        assert report.retired.rows == 0
        assert self._retired(store, RATED_HASH) is None

    def test_an_empty_tree_retires_nothing(self, store: Scratch, tmp_path: Path) -> None:
        """A checkout with no `rulesets/` is a half-shipped tree, not an emptied corpus,
        and the loader never lets one take the box's whole library off the listing."""
        report = loader.load(store.connection, tmp_path)
        store.connection.commit()
        assert report.retired.rows == 0

    def test_a_game_the_loader_does_not_own_keeps_its_own_state(
        self, store: Scratch, tree: Path
    ) -> None:
        """An admin's kill switch and the loader's reading are separate acts on separate
        columns, so a deploy neither sets nor clears `revoked`."""
        _load(store, tree)
        with store.connection.cursor() as cursor:
            cursor.execute(
                "update games set revoked = now() where hash = %s", (STALE_HASH,)
            )
        store.connection.commit()

        self._retire(store, tree)
        with store.connection.cursor() as cursor:
            cursor.execute("select revoked, retired from games where hash = %s",
                           (STALE_HASH,))
            revoked, retired = cursor.fetchone()
        assert revoked is not None and retired is None


class TestAGameWhoseRulesWereDeleted:
    """Q303. A deleted game is never written again, whatever the tree says.

    `admin.delete_game` is irreversible by design, and the tree still holds the file the
    rules came from -- so a load that wrote them back would undo somebody's deletion on
    the next deploy. These hold the loader's end of that.
    """

    def _delete(self, store: Scratch, digest: str) -> None:
        """The row a delete leaves behind: no spec, the placeholder name, a stamp."""
        with store.connection.cursor() as cursor:
            cursor.execute("delete from measurements where hash = %s", (digest,))
            cursor.execute(
                "update games set spec = null, name = %s, deleted = now()"
                " where hash = %s",
                (admin.DELETED_NAME, digest),
            )
        store.connection.commit()

    def test_a_spec_still_in_the_tree_is_skipped_and_named(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        self._delete(store, RATED_HASH)

        report = _load(store, tree)

        assert any(RATED_HASH in note for note in report.notes), report.summary()
        with store.connection.cursor() as cursor:
            cursor.execute(
                "select spec, name, deleted from games where hash = %s", (RATED_HASH,)
            )
            spec, name, when = cursor.fetchone()
        assert spec is None and name == admin.DELETED_NAME and when is not None
        assert store.count(f"measurements where hash = '{RATED_HASH}'") == 0

    def test_nothing_stamps_it_retired_afterwards(
        self, store: Scratch, tree: Path
    ) -> None:
        """Retirement is the weaker state, and stamping a deleted row would only move
        `retired` about on a game that has already gone further than that."""
        _load(store, tree)
        self._delete(store, RATED_HASH)
        (tree / "rulesets" / "rated.json").unlink()

        report = _load(store, tree)

        assert report.retired.rows == 0
        with store.connection.cursor() as cursor:
            cursor.execute("select retired from games where hash = %s", (RATED_HASH,))
            assert cursor.fetchone()[0] is None

    def test_the_rows_that_reference_it_are_untouched(
        self, store: Scratch, tree: Path
    ) -> None:
        """The reason the row stays rather than going: a load must not move a rating or a
        match any more than the delete did."""
        _load(store, tree)
        self._delete(store, RATED_HASH)
        before = store.count("matches"), store.count("ratings")

        _load(store, tree)

        assert (store.count("matches"), store.count("ratings")) == before
        assert m.row(store.connection, str(WON["session"])).game == RATED_HASH


class TestTheMeasurements:
    def test_a_measurement_that_still_holds_carries_its_headline(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        state, headline, basis = _measurement(store, RATED_HASH)
        assert state == loader.CURRENT
        assert headline == pytest.approx(HEADLINE)
        assert basis["weights_version"] == DEFAULT_WEIGHTS

    def test_a_stale_measurement_is_loaded_as_stale_and_carries_no_number(
        self, store: Scratch, tree: Path
    ) -> None:
        """`.claude/rules/viz-and-web.md`: a stale measurement shows neither the index nor
        the ladder. Here there is no number in the column to show."""
        _load(store, tree)
        state, headline, basis = _measurement(store, STALE_HASH)
        assert state != loader.CURRENT and state != loader.MISSING
        assert headline is None
        assert basis["weights_version"] == "v1", "the basis is loaded as what it is"

    def test_a_measurement_with_no_basis_at_all_is_a_replay_and_not_a_guess(
        self, store: Scratch, tree: Path
    ) -> None:
        stored = _canonical(RATED_HASH, RATED, DEFAULT_WEIGHTS)
        del stored["basis"]
        (tree / "runs" / RATED_HASH / "canonical.json").write_text(
            json.dumps(stored), encoding="utf-8"
        )
        _load(store, tree)
        state, headline, basis = _measurement(store, RATED_HASH)
        assert (state, headline, basis) == ("replay", None, None)

    def test_a_game_with_a_ladder_and_no_summary_says_the_summary_is_missing(
        self, store: Scratch, tree: Path
    ) -> None:
        (tree / "runs" / RATED_HASH / "canonical.json").unlink()
        _load(store, tree)
        state, headline, _basis = _measurement(store, RATED_HASH)
        assert (state, headline) == (loader.MISSING, None)
        with store.connection.cursor() as cursor:
            cursor.execute("select top, top_rating from measurements where hash = %s",
                           (RATED_HASH,))
            assert cursor.fetchone() == (TOP_AGENT, 300.0)

    def test_the_whole_ladder_is_kept_because_the_page_names_every_rung(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        with store.connection.cursor() as cursor:
            cursor.execute("select ladder from measurements where hash = %s", (RATED_HASH,))
            assert cursor.fetchone()[0] == LADDER

    def test_a_game_with_no_files_has_no_measurement_row(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        with store.connection.cursor() as cursor:
            cursor.execute("select count(*) from measurements where hash = %s", (BARE_HASH,))
            assert cursor.fetchone()[0] == 0


class TestTheRange:
    """Box E1: the two ends of the interval follow the headline exactly."""

    def test_a_current_measurement_carries_both_ends(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        assert _range(store, RATED_HASH) == (pytest.approx(CI_LOW), pytest.approx(CI_HIGH))

    def test_a_stale_measurement_carries_neither(self, store: Scratch, tree: Path) -> None:
        """The index file is there and reads fine. A stale score is not a number, so
        neither are its ends."""
        assert (tree / "runs" / STALE_HASH / "index-v1.json").exists()
        _load(store, tree)
        assert _range(store, STALE_HASH) == (None, None)

    def test_a_measurement_whose_index_file_has_gone_keeps_its_headline(
        self, store: Scratch, tree: Path
    ) -> None:
        """Absent is not an error and never a zero: a corpus too small to resample has no
        interval at all, and a badge that needs one refuses on that."""
        (tree / "runs" / RATED_HASH / f"index-{DEFAULT_WEIGHTS}.json").unlink()
        _load(store, tree)
        _state, headline, _basis = _measurement(store, RATED_HASH)
        assert headline == pytest.approx(HEADLINE)
        assert _range(store, RATED_HASH) == (None, None)

    def test_an_index_file_that_will_not_parse_is_no_range_rather_than_a_stop(
        self, store: Scratch, tree: Path
    ) -> None:
        (tree / "runs" / RATED_HASH / f"index-{DEFAULT_WEIGHTS}.json").write_text(
            "{ not json", encoding="utf-8"
        )
        _load(store, tree)
        assert _range(store, RATED_HASH) == (None, None)

    def test_a_range_that_appears_later_is_an_update_and_not_a_second_row(
        self, store: Scratch, tree: Path
    ) -> None:
        """A store loaded before this column existed reads its range on the next deploy,
        through the same comparison that corrects anything else."""
        path = tree / "runs" / RATED_HASH / f"index-{DEFAULT_WEIGHTS}.json"
        path.unlink()
        _load(store, tree)
        path.write_text(
            json.dumps({"interval": {"ci_low": CI_LOW, "ci_high": CI_HIGH}}),
            encoding="utf-8",
        )
        report = _load(store, tree)
        assert (report.measurements.added, report.measurements.updated) == (0, 1)
        assert _range(store, RATED_HASH) == (pytest.approx(CI_LOW), pytest.approx(CI_HIGH))


class TestTheSeeding:
    def test_the_ladder_seeds_the_games_computer_players_once(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        table = r.load(store.connection, RATED_HASH)
        assert {key.agent for key in table} == {TOP_AGENT, "greedy", "random"}
        assert table[r.computer(TOP_AGENT)].rating == pytest.approx(300.0 + r.SITE_ORIGIN)

    def test_a_re_rated_ladder_reseeds_unplayed_rows_and_leaves_played_ones(
        self, store: Scratch, tree: Path
    ) -> None:
        """L3's reseed rule (`docs/archive/plan-leaderboard.md` §2.3), through the loader.

        A picker run three months later cannot rewrite what human results decided: the
        rung a person has played keeps its number. Every rung nobody has played is still
        only its seed, so it takes the newer ladder's seat; the rung the table had not seen
        is added. And a second load over the same files writes nothing.
        """
        _load(store, tree)
        with store.connection.cursor() as cursor:
            cursor.execute("insert into users (email) values ('one@example.test') returning id")
            person = int(cursor.fetchone()[0])
            cursor.execute(
                "insert into matches (id, game, mode, seat0_user, seat1_agent, seed, result,"
                " finished) values ('p1', %s, 'leaderboard', %s, %s, 0, 1.0, now())",
                (RATED_HASH, person, TOP_AGENT),
            )
        r.finish(store.connection, "p1")
        store.connection.commit()
        before = r.load(store.connection, RATED_HASH)

        rerated = json.loads(json.dumps(LADDER))
        for row, moved in zip(rerated["standings"], (90.0, 140.0, -20.0), strict=True):
            row["rating"] += moved
        rerated["standings"].append({"name": "mcts:nodes=1600", "rating": 50.0, "games": 24})
        (tree / "runs" / "opponents" / f"{RATED_HASH}.json").write_text(
            json.dumps(rerated), encoding="utf-8"
        )

        report = _load(store, tree)
        assert (report.ratings.added, report.ratings.updated) == (1, 2)
        assert report.ratings.unchanged == 1, "the rung a person played"
        after = r.load(store.connection, RATED_HASH)
        top = r.computer(TOP_AGENT)
        assert after[top] == before[top]
        assert after[r.person(person)] == before[r.person(person)]
        shift = before[top].rating - 390.0
        assert after[r.computer("greedy")].rating == pytest.approx(140.0 + shift)
        assert after[r.computer("random")].rating == pytest.approx(-320.0 + shift)

        snapshot = store.snapshot()
        second = _load(store, tree)
        assert second.ratings.written == 0
        assert store.snapshot() == snapshot

    def test_the_first_seeding_centres_on_the_spine(self, store: Scratch, tree: Path) -> None:
        """§2.4: 1500 is the mean of the standard players the ladder has, not of its roster.

        The fixture's ladder centres on its roster; a fourth, strong player pulls that
        mean up while the spine's three stay where they are, so the seats must not move.
        """
        wider = json.loads(json.dumps(LADDER))
        for row in wider["standings"]:
            row["rating"] -= 150.0
        wider["standings"].append({"name": "minimax:nodes=200000", "rating": 450.0})
        (tree / "runs" / "opponents" / f"{RATED_HASH}.json").write_text(
            json.dumps(wider), encoding="utf-8"
        )
        _load(store, tree)
        table = r.load(store.connection, RATED_HASH)
        spine = [table[r.computer(name)].rating for name in (TOP_AGENT, "greedy", "random")]
        assert sum(spine) / 3 == pytest.approx(r.SITE_ORIGIN)
        assert table[r.computer("minimax:nodes=200000")].rating == pytest.approx(2100.0)

    def test_a_game_with_no_ladder_gets_no_computer_players(
        self, store: Scratch, tree: Path
    ) -> None:
        """An empty table is what makes leaderboard mode refuse the game in a sentence a
        player can read, and offer sandbox instead."""
        _load(store, tree)
        assert r.load(store.connection, STALE_HASH) == {}


# ------------------------------------------------------- the maintainer's own 54 games


class TestTheMaintainersGames:
    def test_they_land_on_the_maintainers_profile(self, store: Scratch, tree: Path) -> None:
        _load(store, tree)
        owner = _maintainer(store)
        theirs = m.leaderboard_matches(store.connection, user=owner, running=True)
        assert {row.id for row in theirs} == {WON["session"], ABANDONED["session"]}

    def test_none_of_them_clears_the_unbeaten_tag(self, store: Scratch, tree: Path) -> None:
        """The win above is against the ladder's top rung on a game whose table exists, so
        it is exactly the row that would clear the tag if anything counted it.

        Two columns keep it out, and task 1.5 may read either. `top` is false, so the win
        reads as a win against a computer player that was not the top when the match
        started -- which §5 says clears nothing. And `source` is `local-app`, which is the
        filter every leaderboard reading applies.
        """
        _load(store, tree)
        won = m.row(store.connection, str(WON["session"]))
        assert won.top is False
        assert won.source == m.LOCAL_APP
        assert won.mode == m.LEADERBOARD, "leaderboard, so the anchor can still read it"

    def test_they_enter_no_rating_fold_and_leave_every_computer_player_at_its_seed(
        self, store: Scratch, tree: Path
    ) -> None:
        """*Rated by nothing* in the one place it can be checked. `rating.finished` is what
        the fold replays, and it drops a row with a `source`; so the table after a load is
        the ladder's seeding and nothing else, with every player still on nought matches."""
        _load(store, tree)
        assert r.finished(store.connection, RATED_HASH) == ()
        table = r.load(store.connection, RATED_HASH)
        assert all(player.matches == 0 for player in table.values())

    def test_the_result_column_is_seat_zeros_score_whichever_seat_the_person_took(
        self, store: Scratch, tree: Path
    ) -> None:
        """`HumanGame.result` is the person's score and `matches.result` is seat 0's, so
        the two are complements when the person sat second."""
        _load(store, tree)
        won = m.row(store.connection, str(WON["session"]))
        assert won.result == 1.0 and won.seat_of(r.person(_maintainer(store))) == 0
        open_game = m.row(store.connection, str(ABANDONED["session"]))
        assert open_game.result is None and open_game.seat_of(r.person(_maintainer(store))) == 1

    def test_the_anchor_reads_them_back_as_the_games_they_were(
        self, store: Scratch, tree: Path
    ) -> None:
        """What they were kept for (§5). `matches.human_games` is `web/human.py`'s own rows
        out of the table, and the person's score comes back the way the file wrote it."""
        _load(store, tree)
        found = {game.session: game for game in m.human_games(store.connection)}
        assert found[str(WON["session"])].result == 1.0
        assert found[str(WON["session"])].opponent == TOP_AGENT
        assert found[str(ABANDONED["session"])].result is None

    def test_a_game_finished_after_the_last_load_is_corrected(
        self, store: Scratch, tree: Path
    ) -> None:
        """The JSONL is append-only with the last line for a session winning, so a game
        that was open at the last load is the one row that has to move."""
        _load(store, tree)
        finished = dict(ABANDONED, result=0.0, winner=0, reason="checkmate",
                        finished=1788030000.0, plies=12)
        _write_human(tree, [WON, ABANDONED, finished])

        report = _load(store, tree)
        assert (report.matches.added, report.matches.updated) == (0, 1)
        # Seat 1 was the person and they lost, so seat 0 took the point.
        assert m.row(store.connection, str(ABANDONED["session"])).result == 1.0

    def test_a_human_game_on_a_ruleset_the_library_does_not_have_is_named_not_invented(
        self, store: Scratch, tree: Path
    ) -> None:
        orphan = dict(WON, session="anOrphan01", ruleset_hash="ffffffffffff")
        _write_human(tree, [WON, ABANDONED, orphan])

        report = _load(store, tree)
        assert report.matches.rows == 2
        assert any("ffffffffffff" in reason for reason in report.skipped)

    def test_a_loaded_row_carries_no_moves_because_the_file_never_held_any(
        self, store: Scratch, tree: Path
    ) -> None:
        """The local application's store kept a ply count and not a move list. A move list
        is what the site replays a position from, so inventing one would be inventing a
        game; the row reads nought plies instead and says `local-app` so every reading of
        length can exclude it."""
        _load(store, tree)
        assert m.row(store.connection, str(WON["session"])).plies == 0


# --------------------------------------------- the badges a load can complete (E8)


def _crowd(scratch: Scratch, digest: str) -> None:
    """Five people, each with a finished leaderboard game on this game and a five-star
    rating of it, written straight into the tables.

    Nothing here grants a badge. What the load makes of rows that were already there is
    the question, and §2.9's third caller of the game badges is this loader.
    """
    with scratch.connection.cursor() as cursor:
        for index in range(badges.CROWD_PEOPLE):
            cursor.execute(
                "insert into users (handle, email) values (%s, %s) returning id",
                (f"p{index}", f"p{index}@example.invalid"),
            )
            user = int(cursor.fetchone()[0])
            cursor.execute(
                "insert into matches (id, game, mode, seed, seat0_user, seat1_agent,"
                " moves, result, reason, top, started, finished)"
                " values (%s, %s, %s, 0, %s, %s, '[]'::jsonb, 0.0, 'checkmate', false,"
                " now(), now())",
                (f"crowd{index}{digest[:6]}", digest, m.LEADERBOARD, user, TOP_AGENT),
            )
            cursor.execute(
                "insert into votes (voter, game, stars) values (%s, %s, 5)",
                (user, digest),
            )
    scratch.connection.commit()


def _on_game(scratch: Scratch, digest: str) -> set[str]:
    return {award.kind for award in awards.held(scratch.connection, game=digest)}


class TestTheBadgesALoadCanComplete:
    """A load is one of the three events that can complete *Proved the clanker wrong*,
    because it is where a measurement comes to read current. The fixture's rated game
    scores 0.6123 with its range inside 0.65, which is what the badge is about.
    """

    def test_a_load_writes_both_badges_on_a_game_that_reads_current(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        _crowd(store, RATED_HASH)
        _load(store, tree)
        assert _on_game(store, RATED_HASH) == {badges.CROWD_PLEASER, badges.CLANKER}

    def test_a_stale_measurement_writes_crowd_pleaser_and_no_clanker(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        _crowd(store, STALE_HASH)
        _load(store, tree)
        assert _on_game(store, STALE_HASH) == {badges.CROWD_PLEASER}

    def test_the_load_that_reads_it_current_writes_the_clanker(
        self, store: Scratch, tree: Path
    ) -> None:
        """E8's *done when*: the next load that reads the measurement current writes U1.
        The same files under the weights they were scored with is what moves the state."""
        _load(store, tree)
        _crowd(store, STALE_HASH)
        _load(store, tree)
        loader.load(store.connection, tree, weights="v1")
        store.connection.commit()
        assert _on_game(store, STALE_HASH) == {badges.CROWD_PLEASER, badges.CLANKER}

    def test_a_game_nobody_has_rated_earns_nothing_on_a_load(
        self, store: Scratch, tree: Path
    ) -> None:
        _load(store, tree)
        assert _on_game(store, RATED_HASH) == set()
        assert _on_game(store, BARE_HASH) == set()
