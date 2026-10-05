"""Resting (Q503i) and the Baseline as a level (Q503n), `docs/plan-game-pipeline.md` §5.

**Resting.** A game on trial rests when it has been on trial `quota.RESTING_AFTER` (90
days) and nobody but its designer finished a game on it in the last `quota.RESTING_QUIET`
(60). `resting.rest`, which `site load` runs, stamps `games.rested`; `stages.stage` then
reads Resting, the listing drops it, its page and link still work, and its place on trial
is freed. A shortlisted game never rests. Its designer may publish it again, once, through
the same gates as a first publish.

**The Baseline.** Only the maintainer moves a game between `archive/` and `rulesets/`.
`site load` corrects the row's origin and records the move with the maintainer's reason
(the first sentence of `provenance.history`, when the move changed it). The game page
then reads *Baseline* with its date and reason, *What's new* carries the move, and no
site route can change an origin.

The store tests need a real Postgres and skip with a reason without one.
"""

from __future__ import annotations

import ast
import json
import secrets
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import site_store
from test_site_admin import Site, _running_site, _saved, _screened, _template

from chess_variant_sandbox.site import drafts, quota, resting, stages
from chess_variant_sandbox.site import load as loader
from chess_variant_sandbox.site import terms as site_terms
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library

READING = site_store.STAND_IN_READING
SITE_PACKAGE = Path(__file__).resolve().parents[1] / "chess_variant_sandbox" / "site"
DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ rows by hand


def _person(site: Site, handle: str) -> int:
    return site.person(handle)


def _finished(site: Site, game: str, person: int, ago: timedelta, reason: str = "checkmate") -> None:
    """One sandbox game on `game` with `person` in seat 0, finished `ago`."""
    match = secrets.token_hex(6)
    when = datetime.now(timezone.utc) - ago
    with site.store.cursor() as cursor:
        cursor.execute(
            "insert into matches (id, game, mode, seat0_user, seat1_agent, seed, started)"
            " values (%s, %s, 'sandbox', %s, 'random', 0, %s)",
            (match, game, person, when),
        )
        cursor.execute(
            "insert into finishes (match, ordinal, moves, result, reason, takebacks, finished)"
            " values (%s, 1, '[]'::jsonb, 1, %s, 0, %s)",
            (match, reason, when),
        )
    site.store.commit()


def _published(site: Site, game: str, ago: timedelta) -> None:
    with site.store.cursor() as cursor:
        cursor.execute("update games set published = now() - %s where hash = %s", (ago, game))
    site.store.commit()


def _column(site: Site, game: str, column: str) -> Any:
    with site.store.cursor() as cursor:
        cursor.execute(f"select {column} from games where hash = %s", (game,))  # noqa: S608
        return cursor.fetchone()[0]


def _sweep(site: Site) -> list[str]:
    done = resting.rest(site.store)
    site.store.commit()
    return [found.hash for found in done]


MOVED = datetime(2099, 1, 1, 23, 30, tzinfo=timezone.utc)
"""The time a test gives a level move, so the date the page says is fixed (Q594, Q730).
Late in UTC's day, so a session zone ahead of UTC would name the next day; after every
changelog entry, so the move heads What's new."""
MOVED_DAY = "1 Jan 2099"


def _moved_at(site: Site, game: str, when: datetime = MOVED) -> None:
    """Pin the game's level moves to `when`, the test's own clock."""
    with site.store.cursor() as cursor:
        cursor.execute("update level_moves set moved = %s where game = %s", (when, game))
    site.store.commit()


def _level(site: Site, game: str) -> str:
    level = str(stages.read(site.store, [game])[game]["level"])
    site.store.commit()
    return level


class Trial:
    """A running site with ada's game on trial, 91 days since she published it."""

    def __init__(self, site: Site, ada: Any, ada_id: int, game: str) -> None:
        self.site = site
        self.ada = ada
        self.ada_id = ada_id
        self.game = game


@pytest.fixture
def trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Trial]:
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    _screened(monkeypatch, passed=True)
    with _running_site(tmp_path) as site:
        ada = site.signed_in("ada")
        with site.store.cursor() as cursor:
            cursor.execute("select id from users where handle = 'ada'")
            ada_id = int(cursor.fetchone()[0])
        game = _saved(ada, _template(0, "ada's quiet game")).payload["hash"]
        site_store.on_trial(site.store, game)
        _published(site, game, quota.RESTING_AFTER + timedelta(days=1))
        yield Trial(site, ada, ada_id, game)


# ------------------------------------------------------------------------ the numbers


def test_the_plans_numbers_live_in_quota() -> None:
    assert (
        timedelta(days=90), timedelta(days=60), 1
    ) == (quota.RESTING_AFTER, quota.RESTING_QUIET, quota.WAKES_A_GAME)
    assert "A trial game rests after" in dict(quota.summary())


# ------------------------------------------------------------------------ the sweep


class TestTheSweep:
    def test_a_quiet_game_rests_and_leaves_the_listing(self, trial: Trial) -> None:
        site = trial.site
        assert trial.game in site.listed()
        assert quota.on_trial(site.store, trial.ada_id) == 1

        assert _sweep(site) == [trial.game]

        assert _level(site, trial.game) == stages.RESTING
        assert trial.game not in site.listed()
        # Its place on trial is freed, and it is still public: its page and link work.
        assert quota.on_trial(site.store, trial.ada_id) == 0
        site.store.commit()
        assert _column(site, trial.game, "visibility") == "public"
        page = site.browser().get(f"/api/game/{trial.game}")
        assert page.status == 200, page.text
        assert page.payload["level"] == stages.RESTING
        assert "off the listing" in page.payload["levelSays"]
        assert "Its designer can publish it again, once." in page.payload["levelSays"]

    def test_a_second_sweep_keeps_the_first_stamp(self, trial: Trial) -> None:
        _sweep(trial.site)
        first = _column(trial.site, trial.game, "rested")
        assert _sweep(trial.site) == []
        assert _column(trial.site, trial.game, "rested") == first

    def test_a_recent_finish_by_somebody_else_keeps_it_on_trial(self, trial: Trial) -> None:
        bea = _person(trial.site, "bea")
        _finished(trial.site, trial.game, bea, quota.RESTING_QUIET - timedelta(days=1))
        assert _sweep(trial.site) == []
        assert _level(trial.site, trial.game) == stages.ON_TRIAL

    @pytest.mark.parametrize(
        "why",
        ["the designer's own", "an old one", "a resignation"],
    )
    def test_a_finish_that_does_not_count_does_not_keep_it(
        self, trial: Trial, why: str
    ) -> None:
        recent = quota.RESTING_QUIET - timedelta(days=1)
        if why == "the designer's own":
            _finished(trial.site, trial.game, trial.ada_id, recent)
        elif why == "an old one":
            bea = _person(trial.site, "bea")
            _finished(trial.site, trial.game, bea, quota.RESTING_QUIET + timedelta(days=1))
        else:
            bea = _person(trial.site, "bea")
            _finished(trial.site, trial.game, bea, recent, reason="resigned")
        assert _sweep(trial.site) == [trial.game]

    def test_a_game_not_yet_ninety_days_on_trial_does_not_rest(self, trial: Trial) -> None:
        _published(trial.site, trial.game, quota.RESTING_AFTER - timedelta(days=1))
        assert _sweep(trial.site) == []

    def test_a_shortlisted_game_never_rests(self, trial: Trial) -> None:
        with trial.site.store.cursor() as cursor:
            cursor.execute("update games set nominated = now() where hash = %s", (trial.game,))
        trial.site.store.commit()
        assert _sweep(trial.site) == []
        assert _level(trial.site, trial.game) == stages.SHORTLISTED

    def test_a_draft_a_killed_game_and_the_library_never_rest(self, trial: Trial) -> None:
        site = trial.site
        draft = _saved(trial.ada, _template(1, "ada's draft")).payload["hash"]
        _published(site, draft, quota.RESTING_AFTER * 2)
        corpus = site.game()
        _published(site, corpus, quota.RESTING_AFTER * 2)
        with site.store.cursor() as cursor:
            cursor.execute("update games set revoked = now() where hash = %s", (trial.game,))
        site.store.commit()
        assert _sweep(site) == []

    def test_the_load_runs_it_and_names_what_it_rested(
        self, trial: Trial, tmp_path: Path
    ) -> None:
        report = loader.load(trial.site.store, tmp_path)
        trial.site.store.commit()
        assert report.rested.updated == 1
        assert any(trial.game in note and "rested" in note for note in report.notes)
        again = loader.load(trial.site.store, tmp_path)
        trial.site.store.commit()
        assert again.rested.written == 0


# ------------------------------------------------------------------------ the way back


class TestComingBackOnce:
    def test_its_designer_publishes_it_again_through_the_same_gates(
        self, trial: Trial
    ) -> None:
        site = trial.site
        _sweep(site)
        standing = drafts.standing(site.store, trial.game, trial.ada_id)
        site.store.commit()
        assert standing is not None
        assert standing["level"] == stages.RESTING
        assert standing["publishable"] is True
        assert "You can publish it again, once" in standing["says"]
        # No reading can be attached through the route until Q503e, so the press is
        # refused there as a first publish is.
        refused = trial.ada.post("/api/design/publish", {"hash": trial.game})
        assert refused.status == 400
        assert drafts.NEEDS_A_READING in refused.text

        done = drafts.publish(site.store, trial.game, trial.ada_id, READING, site_terms.VERSION)
        site.store.commit()
        assert done.already is False
        assert _level(site, trial.game) == stages.ON_TRIAL
        assert trial.game in site.listed()
        assert _column(site, trial.game, "woken") is not None
        # A fresh ninety days: the clock restarted, so the next sweep leaves it.
        assert _sweep(site) == []

    def test_a_game_that_rests_again_stays_resting(self, trial: Trial) -> None:
        site = trial.site
        _sweep(site)
        drafts.publish(site.store, trial.game, trial.ada_id, READING, site_terms.VERSION)
        site.store.commit()
        _published(site, trial.game, quota.RESTING_AFTER + timedelta(days=1))
        assert _sweep(site) == [trial.game]

        with pytest.raises(drafts.PublishError, match="once already"):
            drafts.publish(site.store, trial.game, trial.ada_id, READING, site_terms.VERSION)
        site.store.rollback()
        standing = drafts.standing(site.store, trial.game, trial.ada_id)
        site.store.commit()
        assert standing is not None and standing["publishable"] is False
        assert "stays here" in standing["says"]

    def test_only_its_designer_can_bring_it_back(self, trial: Trial) -> None:
        bea = _person(trial.site, "bea")
        _sweep(trial.site)
        with pytest.raises(drafts.PublishError, match="Only the person"):
            drafts.publish(trial.site.store, trial.game, bea, READING, site_terms.VERSION)
        trial.site.store.rollback()
        standing = drafts.standing(trial.site.store, trial.game, bea)
        trial.site.store.commit()
        assert standing is not None and standing["publishable"] is False

    def test_the_profile_names_it_with_its_next_step(self, trial: Trial) -> None:
        _sweep(trial.site)
        groups = stages.yours(trial.site.store, trial.ada_id)
        trial.site.store.commit()
        resting_group = next(group for group in groups if group["level"] == stages.RESTING)
        (game,) = resting_group["games"]
        assert game["hash"] == trial.game
        assert game["inDesigner"] is True
        assert "once" in game["next"]

    def test_a_resting_game_cannot_be_nominated(self, trial: Trial) -> None:
        from chess_variant_sandbox.site import shortlist

        _sweep(trial.site)
        with pytest.raises(shortlist.NominateError, match="Only a game on trial"):
            shortlist.nominate(trial.site.store, trial.game, trial.ada_id, READING)
        trial.site.store.rollback()


# --------------------------------------------------------------- Q503n, the Baseline


def _variant(max_plies: int, name: str, history: str = "") -> dict[str, Any]:
    spec = json.loads(json.dumps(DULL))
    spec["rules"]["max_plies"] = max_plies
    spec["name"] = name
    spec["provenance"] = {"origin": "human"}
    if history:
        spec["provenance"]["history"] = history
    return spec


class TestTheReason:
    @pytest.mark.parametrize(
        ("before", "after", "reason"),
        [
            # `promote --reason` puts the reason first and keeps what was there.
            ("Kept for its seam.", "Pins down wrap boards. Kept for its seam.",
             "Pins down wrap boards."),
            (None, "A published variant people play.", "A published variant people play."),
            # A file moved without a word keeps its history: no reason, never a stale one.
            ("Kept for its seam.", "Kept for its seam.", ""),
            ("Kept for its seam.", None, ""),
            ("", "Ends with a question? Then more.", "Ends with a question?"),
        ],
    )
    def test_it_is_the_first_sentence_the_move_wrote(
        self, before: str | None, after: str | None, reason: str
    ) -> None:
        def spec(history: str | None) -> dict[str, Any]:
            return {"provenance": {} if history is None else {"history": history}}

        assert loader.reason_for(spec(before), spec(after)) == reason


class TestWhatsNewFilesTheMovesByDate:
    def _move(self, when: datetime, joined: bool = True, reason: str = "") -> stages.Move:
        return stages.Move(
            "abc", "gardner", "archive" if joined else "corpus",
            "corpus" if joined else "archive", reason, when,
        )

    def test_a_move_joins_its_day_or_takes_its_own_place(self) -> None:
        payload = {
            "days": [
                {"date": "2026-09-29", "words": "29 September 2026",
                 "entries": [{"heading": "file's", "blocks": []}]},
                {"date": "2026-09-20", "words": "20 September 2026", "entries": []},
            ],
            "note": "",
        }
        moves = [
            self._move(datetime(2026, 9, 29, 12, tzinfo=timezone.utc), reason="Why."),
            self._move(datetime(2026, 9, 25, 12, tzinfo=timezone.utc), joined=False),
            self._move(datetime(2026, 10, 2, 12, tzinfo=timezone.utc)),
        ]
        found = stages.whats_new(payload, moves)
        assert [day["date"] for day in found["days"]] == [
            "2026-10-02", "2026-09-29", "2026-09-25", "2026-09-20"
        ]
        same = found["days"][1]["entries"]
        assert [entry["heading"] for entry in same] == ["gardner joins the Baseline", "file's"]
        lines = [block["runs"][0]["s"] for block in same[0]["blocks"]]
        assert "The reason given: Why." in lines
        assert any("Scores stay as they are until every game is re-scored" in line
                   for line in lines)
        # Plain text runs only: the page makes an element per run and sets its text.
        assert {run["t"] for block in same[0]["blocks"] for run in block["runs"]} == {"text"}
        assert found["days"][2]["entries"][0]["heading"] == (
            "gardner leaves the Baseline for the Library"
        )

    def test_no_moves_leaves_the_file_as_it_was(self) -> None:
        payload = {"days": [], "note": "The changelog holds no entries yet."}
        assert stages.whats_new(payload, []) == payload


@pytest.fixture
def tree_site(tmp_path: Path) -> Iterator[tuple[Site, Path]]:
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    with _running_site(tmp_path) as site:
        yield site, tmp_path


def _load(site: Site, root: Path) -> loader.Report:
    report = loader.load(site.store, root)
    site.store.commit()
    return report


class TestAGameThatJoinsTheBaseline:
    def test_the_page_says_the_moves_utc_day_as_whats_new_files_it(self) -> None:
        """No database. Half past midnight two hours east of UTC is the day before in
        UTC, which is where `whats_new` files the move (Q730)."""
        east = timezone(timedelta(hours=2))
        assert stages.day(datetime(2026, 10, 5, 0, 30, tzinfo=east)) == "4 Oct 2026"

    def test_it_reads_baseline_with_its_date_and_reason_and_whats_new_says_so(
        self, tree_site: tuple[Site, Path]
    ) -> None:
        site, root = tree_site
        kept = _variant(3, "joiner", "Kept for its long game.")
        digest = spec_hash(kept)
        (root / "archive" / "joiner.json").write_text(json.dumps(kept), encoding="utf-8")
        _load(site, root)
        assert _level(site, digest) == stages.LIBRARY
        assert site.browser().get(f"/api/game/{digest}").payload["levelSays"] == (
            stages.SAYS[stages.LIBRARY]
        )

        # What `promote --reason` does: the reason first in the history, the file moved.
        promoted = _variant(3, "joiner", "Pins down the long game. Kept for its long game.")
        assert spec_hash(promoted) == digest
        (root / "archive" / "joiner.json").unlink()
        (root / "rulesets" / "joiner.json").write_text(json.dumps(promoted), encoding="utf-8")
        report = _load(site, root)

        assert report.moved.added == 1
        assert any("joined the Baseline: Pins down the long game." in n for n in report.notes)
        assert _level(site, digest) == stages.BASELINE
        _moved_at(site, digest)
        page = site.browser().get(f"/api/game/{digest}").payload
        assert page["level"] == stages.BASELINE
        assert page["levelSays"] == (
            f"One of the games every score is measured against, since {MOVED_DAY}: "
            "Pins down the long game."
        )
        assert [step["word"] for step in page["track"]][-1] == stages.BASELINE

        news = site.browser().get("/api/changelog").payload
        first = news["days"][0]
        assert first["date"] == MOVED.date().isoformat()
        assert first["entries"][0]["heading"] == "joiner joins the Baseline"

        # A second load finds nothing new and records no second move.
        again = _load(site, root)
        assert again.moved.written == 0 and again.written == 0
        assert site.count("level_moves") == 1

    def test_a_move_without_a_word_shows_no_reason_and_leaving_says_it_left(
        self, tree_site: tuple[Site, Path]
    ) -> None:
        site, root = tree_site
        spec = _variant(5, "mover", "Kept for its seam.")
        digest = spec_hash(spec)
        (root / "rulesets" / "mover.json").write_text(json.dumps(spec), encoding="utf-8")
        _load(site, root)
        assert _level(site, digest) == stages.BASELINE
        # Always a Baseline game here, so its sentence is the plain one.
        assert site.browser().get(f"/api/game/{digest}").payload["levelSays"] == (
            stages.SAYS[stages.BASELINE]
        )

        (root / "rulesets" / "mover.json").rename(root / "archive" / "mover.json")
        report = _load(site, root)
        assert any("with no reason in its provenance" in note for note in report.notes)
        assert _level(site, digest) == stages.LIBRARY
        _moved_at(site, digest)
        said = site.browser().get(f"/api/game/{digest}").payload["levelSays"]
        assert said == (
            f"{stages.SAYS[stages.LIBRARY]} It left the games every score is measured "
            f"against on {MOVED_DAY}."
        )


# --------------------------------------------------------- no route changes an origin


def _sql_in(path: Path) -> list[str]:
    """Every string literal in a module, joined where Python joins adjacent ones."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(node.value.lower())
        elif isinstance(node, ast.JoinedStr):
            found.append(
                "".join(
                    part.value.lower() if isinstance(part, ast.Constant) else "{}"
                    for part in node.values
                    if isinstance(part, ast.Constant | ast.FormattedValue)
                )
            )
    return found


class TestNoRouteChangesAnOrigin:
    """The Baseline and the Library are the maintainer's, by files in the tree. So the
    loader is the one module on the site that writes `games.origin` or `level_moves`, a
    save writes only its own `designed` rows, and nothing the server runs imports it."""

    def test_only_the_loader_writes_an_origin_or_a_move(self) -> None:
        writers: dict[str, list[str]] = {}
        for path in sorted(SITE_PACKAGE.glob("*.py")):
            for text in _sql_in(path):
                updates = "update games" in text and "origin" in text.split("where")[0]
                if updates or "insert into level_moves" in text or "update level_moves" in text:
                    writers.setdefault(path.name, []).append(text)
        assert set(writers) == {"load.py"}, writers

    def test_a_save_inserts_its_own_designed_row_and_nothing_else_inserts_a_game(
        self,
    ) -> None:
        inserters = {
            path.name
            for path in SITE_PACKAGE.glob("*.py")
            for text in _sql_in(path)
            if "insert into games" in text
        }
        assert inserters == {"load.py", "drafts.py"}
        source = (SITE_PACKAGE / "drafts.py").read_text(encoding="utf-8")
        assert "DESIGNED" in source[source.index('"insert into games"'):][:600]

    def test_nothing_the_server_runs_imports_the_loader(self) -> None:
        """The package's `__init__` names `load` for the command line to reach; it
        calls nothing. Every module the server is built from leaves it alone."""
        for path in sorted(SITE_PACKAGE.glob("*.py")):
            if path.name in ("load.py", "__init__.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = {alias.name for alias in node.names}
                    assert not (
                        node.module.endswith("site") and "load" in names
                    ), path.name
                    assert not node.module.endswith("site.load"), path.name

    def test_a_save_of_a_baseline_games_rules_leaves_its_row_alone(
        self, tree_site: tuple[Site, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        site, root = tree_site
        _screened(monkeypatch, passed=True)
        spec = _variant(7, "held")
        digest = spec_hash(spec)
        (root / "rulesets" / "held.json").write_text(json.dumps(spec), encoding="utf-8")
        _load(site, root)
        ada = site.signed_in("ada")
        saved = ada.post("/api/design/save", {"spec": spec, "parent": ""})
        assert saved.status == 200, saved.text
        assert _column(site, digest, "origin") == "corpus"
        refused = ada.post("/api/design/publish", {"hash": digest})
        assert refused.status == 400
        assert _column(site, digest, "origin") == "corpus"
        assert _level(site, digest) == stages.BASELINE
