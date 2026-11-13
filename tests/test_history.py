"""A score's history (Q649, `docs/archive/plan-confidence.md` §3.2 step 3 and §4 T4).

What these hold down:

- **A replacement under another basis is kept.** The outgoing `canonical.json` and its runs
  go under `runs/<hash>/previous/<n>/` (runs played again moved, runs re-scored copied) and
  a line is appended to `runs/<hash>/history.jsonl`.
- **A re-run under the same basis keeps nothing**: no line, no `previous/`.
- **Nothing in history is ever a headline**: the score a reader is given is the new
  measurement's, the history travels beside it, and a site visitor is sent none of its
  ledger: each line reaches them as a *was* line in the trust line (`history.SHOW_WAS_TO_VISITORS`, on since 2026-10-03).
- It moves nothing: no metric imports it and it is in no fingerprint set.

Nothing plays: `measure`'s expensive calls are stubs, on `test_corpus`'s five-square game.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import private_paths
import pytest
from test_corpus import SPEC, _find, _Scored, world  # noqa: F401  (`world` is a fixture)

from chess_variant_sandbox import corpus, history, measurement
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.spec import edits, load_spec, spec_hash
from chess_variant_sandbox.spec.legacy_names import LEGACY_NAMES


@pytest.fixture
def stubbed(world, monkeypatch) -> dict[str, list[Any]]:  # noqa: F811
    """`measure`'s tournament, pass and scorer as stubs; the scorer scores
    ``totals[seed]`` so a test can say what the new measurement reads."""
    calls: dict[str, list[Any]] = {"play": [], "score": [], "totals": [0.6]}

    class _Ratings:
        standings: list[Any] = []

    @dataclass
    class _Tournament:
        seed: int
        records: list[Any] = field(default_factory=list)
        ratings: Any = field(default_factory=_Ratings)

        def save(self, out: Path) -> None:
            _game_on_disk(Path(out), self.seed, "new")
            (Path(out) / "tournament.json").write_text(json.dumps({
                "config": {"agents": list(corpus.CANONICAL_AGENTS),
                           "games_per_pairing": corpus.CANONICAL_GAMES, "seed": self.seed},
                "games": 0, "ratings": {"standings": []},
            }), encoding="utf-8")

    def play(spec, config, progress=False):
        calls["play"].append(config.seed)
        return _Tournament(seed=config.seed)

    def analyse(spec, records, seed=0, workers=0, **_ignored):
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

        return CorpusAnalysis()

    def score(spec, records, ratings, weights, analysis=None, **_ignored):
        scored = _Scored(analysis=analysis, weights_version=weights.version,
                         headline=calls["totals"][len(calls["score"]) % len(calls["totals"])])
        calls["score"].append(scored)
        return scored

    monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.LIBRARY_SEEDS)
    monkeypatch.setattr(corpus, "run_schedules", lambda spec, configs, **_: [play(spec, c) for c in configs])
    monkeypatch.setattr(corpus, "analyse_corpus", analyse)
    monkeypatch.setattr(corpus, "score_ruleset", score)
    return calls


@pytest.fixture
def game(world, stubbed) -> corpus.Candidate:  # noqa: F811
    """The game, archived, so its library expects four runs."""
    corpus.archive_game(_find(world), archive_dir=world["archive"], runs_dir=world["runs"])
    return _find(world)


def _stored(world, index: float = 0.8, **moved: Any) -> dict[str, Any]:  # noqa: F811
    """A four-run library measurement with its runs on disk, stamped with today's basis
    and ``moved`` over it (``{}`` is current)."""
    from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

    seeds = corpus.LIBRARY_SEEDS
    basis = {**corpus.current_basis(world["hash"], spec=SPEC, seeds=seeds).as_dict(), **moved}
    out = corpus.measurement_dir(world["hash"], world["runs"])
    out.mkdir(parents=True, exist_ok=True)
    totals = [index - 0.01, index + 0.01, index - 0.01, index + 0.01]
    (out / "canonical.json").write_text(json.dumps({
        "hash": world["hash"], "name": "lancers",
        "agents": list(corpus.CANONICAL_AGENTS),
        "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": list(seeds), "games": 1024,
        "index": index, "totals_used": 4, "sd": 0.02, "ci_low": index - 0.04,
        "ci_high": index + 0.04, "viable": True, "violations": [],
        "weights_version": basis["weights_version"], "families": {}, "standings": [],
        "runs": [{"seed": seed, "index": total, "sd": 0.02, "viable": True,
                  "violations": [], "unreadable": [], "origin": "played"}
                 for seed, total in zip(seeds, totals, strict=True)],
        "basis": basis,
    }), encoding="utf-8")
    (out / f"index-{DEFAULT_WEIGHTS}.json").write_text("{}", encoding="utf-8")
    for seed in seeds:
        run = corpus.run_dir(world["hash"], seed, world["runs"])
        run.mkdir(parents=True, exist_ok=True)
        _game_on_disk(run, seed, "old")
        (run / "tournament.json").write_text(json.dumps({
            "config": {"agents": list(corpus.CANONICAL_AGENTS),
                       "games_per_pairing": corpus.CANONICAL_GAMES, "seed": seed},
            "games": 4, "ratings": {"standings": []},
        }), encoding="utf-8")
        CorpusAnalysis().save(run / "analysis.json")
    return basis


def _game_on_disk(run: Path, seed: int, mark: str) -> None:
    """One real record in `games.jsonl` (a re-score reads it) and a `mark.txt` saying
    which measurement played it."""
    from chess_variant_sandbox.spec import spec_hash
    from chess_variant_sandbox.tournament.record import GameRecord, write_records

    run.mkdir(parents=True, exist_ok=True)
    write_records([GameRecord(
        ruleset_hash=spec_hash(SPEC), ruleset_name="lancers", seed=seed,
        agents=("random", "greedy"), moves=[], winner=1, reason="royal_capture", plies=6,
    )], run / "games.jsonl")
    (run / "mark.txt").write_text(f"{mark} {seed}", encoding="utf-8")


def _mark(run: Path) -> str:
    return (run / "mark.txt").read_text(encoding="utf-8")


def _tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*"))


def _contents(root: Path) -> dict[str, bytes | None]:
    """Every path under `root` with the bytes of each file: names alone miss a rewrite."""
    return {
        str(path.relative_to(root)).replace("\\", "/"): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


# The basis a stronger roster would stamp: the same spelling playing a net (`Basis.net`).
NET = {"net": "1d160257b8e5=net:abababababab"}


class TestAReplacementIsKept:
    def test_a_replay_keeps_the_outgoing_measurement_and_appends_a_line(
        self, world, game, stubbed  # noqa: F811
    ) -> None:
        old = _stored(world, index=0.8, **NET)
        canonical = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        before = canonical.read_bytes()

        new = corpus.refresh(game, runs_dir=world["runs"], workers=1)

        assert new.current and new.stored["index"] == 0.6
        assert stubbed["play"] == list(corpus.LIBRARY_SEEDS)
        kept = history.previous_root(world["hash"], world["runs"]) / "1"
        # The outgoing file, byte for byte, and its runs, moved whole: the old games are
        # under previous/1 and the run directories hold the new ones.
        assert (kept / "canonical.json").read_bytes() == before
        assert (kept / f"index-{DEFAULT_WEIGHTS}.json").exists()
        for seed in corpus.LIBRARY_SEEDS:
            name = f"{world['hash']}-{seed}"
            assert _mark(kept / name) == f"old {seed}"
            run = corpus.run_dir(world["hash"], seed, world["runs"])
            assert _mark(run) == f"new {seed}"

        [line] = history.read(world["hash"], world["runs"])
        assert line.n == 1 and line.path == "previous/1"
        assert line.index == 0.8 and line.runs == 4 and line.seeds == corpus.LIBRARY_SEEDS
        assert (line.ci_low, line.ci_high) == pytest.approx((0.76, 0.84))
        assert line.basis == old
        assert line.digest == history.digest_of(old)
        assert line.to == history.digest_of(new.basis.as_dict()) != line.digest
        assert line.needs == "replay"
        assert "net" in line.moved
        assert any("nets" in reason for reason in line.reasons)
        assert line.measured and line.replaced

    def test_a_second_replacement_is_kept_beside_the_first(self, world, game) -> None:  # noqa: F811
        _stored(world, index=0.8, **NET)
        corpus.refresh(game, runs_dir=world["runs"], workers=1)
        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["basis"] = {**payload["basis"], "weights_version": "v0"}
        path.write_text(json.dumps(payload), encoding="utf-8")

        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        lines = history.read(world["hash"], world["runs"])
        assert [line.n for line in lines] == [1, 2]
        assert lines[1].index == 0.6 and lines[1].needs == "rescore"
        assert (history.previous_root(world["hash"], world["runs"]) / "2").is_dir()

    def test_a_re_score_copies_the_runs_and_keeps_their_games_in_place(
        self, world, game, stubbed  # noqa: F811
    ) -> None:
        _stored(world, index=0.8, metrics="moved")

        new = corpus.refresh(game, runs_dir=world["runs"], workers=1)

        assert new.current and not stubbed["play"]
        kept = history.previous_root(world["hash"], world["runs"]) / "1"
        for seed in corpus.LIBRARY_SEEDS:
            name = f"{world['hash']}-{seed}"
            assert _mark(kept / name) == f"old {seed}"
            run = corpus.run_dir(world["hash"], seed, world["runs"])
            assert _mark(run) == f"old {seed}"
        [line] = history.read(world["hash"], world["runs"])
        assert line.needs == "rescore" and line.moved == ("metrics",)

    def test_a_measurement_that_fails_half_way_keeps_once_and_writes_no_line(
        self, world, game, stubbed, monkeypatch  # noqa: F811
    ) -> None:
        _stored(world, index=0.8, **NET)

        def broken(*_args, **_kwargs):
            raise corpus.CorpusError("the scorer fell over")

        real = corpus._store
        monkeypatch.setattr(corpus, "_store", broken)
        with pytest.raises(corpus.CorpusError):
            corpus.refresh(game, runs_dir=world["runs"], workers=1)
        assert history.read(world["hash"], world["runs"]) == []
        # The outgoing file is still the file in place: it was copied, not moved.
        stale = corpus.read_measurement(world["hash"], world["runs"], spec=SPEC)
        assert stale is not None and stale.stored["index"] == 0.8

        monkeypatch.setattr(corpus, "_store", real)
        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        assert [line.n for line in history.read(world["hash"], world["runs"])] == [1]
        assert [p.name for p in history.previous_root(world["hash"], world["runs"]).iterdir()] \
            == ["1"]
        kept = history.previous_root(world["hash"], world["runs"]) / "1"
        assert _mark(kept / f"{world['hash']}-0") == "old 0"


class TestASameBasisReRunKeepsNothing:
    def test_a_forced_re_run_of_a_current_measurement_appends_nothing(
        self, world, game, stubbed  # noqa: F811
    ) -> None:
        _stored(world, index=0.8)
        assert corpus.read_measurement(world["hash"], world["runs"], spec=SPEC).current

        corpus.measure(game, runs_dir=world["runs"], workers=1)

        assert stubbed["play"] == list(corpus.LIBRARY_SEEDS)
        assert history.read(world["hash"], world["runs"]) == []
        assert not history.previous_root(world["hash"], world["runs"]).exists()
        assert not history.history_path(world["hash"], world["runs"]).exists()

    def test_a_current_measurement_is_not_touched_by_refresh(self, world, game) -> None:  # noqa: F811
        _stored(world, index=0.8)
        before = _contents(world["runs"])
        assert before, "nothing stored, so nothing to leave alone"
        corpus.refresh(game, runs_dir=world["runs"], workers=1)
        assert _contents(world["runs"]) == before


class TestNothingInHistoryIsAHeadline:
    def test_the_library_shows_the_new_score_and_carries_the_old_beside_it(
        self, world, game  # noqa: F811
    ) -> None:
        from chess_variant_sandbox.web import library

        _stored(world, index=0.8, **NET)
        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        [found] = [g for g in library.discover(world["rulesets"], world["runs"],
                                               world["archive"]) if g.hash == world["hash"]]
        assert found.score is not None and found.score["index"] == 0.6
        assert found.summary()["index"] == 0.6
        detail = library.detail(found)
        assert detail["score"]["index"] == 0.6
        [was] = detail["history"]
        assert was["index"] == 0.8 and was["basis"]["net"] == NET["net"]
        assert "players" in was
        # And no reader of the score reads the ledger: the card carries no history.
        assert "history" not in found.summary()

    def test_a_stale_score_with_a_history_is_provisional_and_never_a_was(
        self, world, game  # noqa: F811
    ) -> None:
        """EV2: the stale file's own number is shown, marked provisional; the ledger's
        earlier score never stands in for it."""
        from chess_variant_sandbox.web import library

        _stored(world, index=0.8, **NET)
        corpus.refresh(game, runs_dir=world["runs"], workers=1)
        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["basis"] = {**payload["basis"], "engine": "another-core"}
        path.write_text(json.dumps(payload), encoding="utf-8")

        [found] = [g for g in library.discover(world["rulesets"], world["runs"],
                                               world["archive"]) if g.hash == world["hash"]]
        assert found.score is not None and found.score["status"] == "provisional"
        assert found.summary()["index"] == 0.6 and found.current_score is None
        assert [was["index"] for was in found.history] == [0.8]

    def test_a_new_line_reaches_a_cached_listing(self, world, game) -> None:  # noqa: F811
        from chess_variant_sandbox.web import library

        _stored(world, index=0.8)
        listed = library.discover(world["rulesets"], world["runs"], world["archive"])
        assert all(not g.history for g in listed)
        entry = history.entry_for(1, {"index": 0.5, "basis": None}, None, None, "replay", ())
        history.append(world["hash"], world["runs"], entry)

        again = library.discover(world["rulesets"], world["runs"], world["archive"])
        assert [g.history for g in again if g.hash == world["hash"]][0]


class TestTheSiteKeepsItFromVisitors:
    LINES = [{"n": 1, "index": 0.8, "runs": 4, "measured": "2026-09-28",
              "replaced": "2026-10-03", "basis": {"net": ""}, "moved": ["net"],
              "reasons": ["its nets were none"], "players": "8 players"}]

    def test_the_switch_is_on(self) -> None:
        """§5 Q1, answered A on 2026-10-03: a visitor sees the previous score."""
        assert history.SHOW_WAS_TO_VISITORS is True

    def test_a_visitor_is_sent_no_ledger_and_the_admin_every_line(self) -> None:
        """The visitor's *was* line rides in the trust line (`test_trust_line.py`), never
        the ledger with its bases."""
        from chess_variant_sandbox.site.server import history_shown

        found = {"score": {"index": 0.6}, "history": self.LINES}
        assert history_shown(found, is_admin=False) == []
        assert history_shown(found, is_admin=True) == self.LINES

    def test_a_visitor_reads_the_value_and_date_alone(self, monkeypatch) -> None:
        entries = [history.Entry.from_dict(line) for line in self.LINES]
        assert history.for_visitors(entries, 0.6) == [
            {"index": 0.8, "runs": 4, "measured": "2026-09-28", "weights_version": ""}]
        # The version word rides with each line, for the game page's chart (SH4).
        versioned = [history.Entry.from_dict({**self.LINES[0], "weights_version": "v12"})]
        assert history.for_visitors(versioned, 0.6)[0]["weights_version"] == "v12"
        # A replacement that moved no player is no *was* line for a visitor.
        other = [history.Entry.from_dict({**self.LINES[0], "moved": ["engine"]})]
        assert history.for_visitors(other, 0.6) == []
        monkeypatch.setattr(history, "SHOW_WAS_TO_VISITORS", False)
        assert history.for_visitors(entries, 0.6) == []

    def test_the_page_draws_was_lines_from_the_payload_only(self) -> None:
        from page_text import PAGE

        assert "function wasHtml(lines, game)" in PAGE
        assert "${earlierHtml(game)}" in PAGE

    # SH4: a visitor's game page as the site sends it (`history_shown` empties `history`),
    # with `for_visitors`' two lines, and a decoy beside them that the chart must not read.
    VISITOR = {
        "history": [], "status": "current", "index": 0.61, "measuredAt": "2026-11-08T09:00:00Z",
        "weights": "v13", "viable": True,
        "earlier": [{"index": 0.8, "runs": 4, "measured": "2026-10-20", "weights_version": "v13"},
                    {"index": 0.5, "runs": 4, "measured": "2026-09-28", "weights_version": "v12"}],
        "trust": {"more": ["Was 0.333 (4 runs) when measured by the earlier computer players."]},
        "score": {"index": 0.61, "interval": {"low": 0.2, "high": 0.9}},
    }

    @staticmethod
    def _fold(game: dict[str, Any]) -> str:
        """`detail.js`'s fold run in node with `history.js` under it, its markup printed."""
        import node_session
        import page_text

        detail = page_text.text("/static/page/detail.js")
        names = ("earlierLines", "earlierHtml", "earlierNow", "earlierChartHtml", "earlierLinkHtml",
                 "visitorsWasHtml", "wasHtml")
        pieces = []
        for name in names:
            start = detail.index(f"function {name}(")
            pieces.append(detail[start:detail.index("\n}\n", start) + 3])
        stub = """
          const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/"/g, '&quot;');
          const info = key => '(' + key + ')';
          const countWord = n => String(n);
          const rulesVersion = v => 'scoring rules ' + v;
          const isProvisional = e => !!e && e.status === 'provisional';
          const location = { search: '', pathname: '/', hash: '' };
        """
        source = (stub + page_text.text("/static/page/history.js") + "\n".join(pieces)
                  + f"\nconsole.log(JSON.stringify(earlierHtml({json.dumps(game)})));")
        done = node_session.run(source)
        assert done.returncode == 0, done.stderr
        return str(json.loads(done.stdout.strip().splitlines()[-1]))

    def test_a_visitors_chart_draws_from_for_visitors_lines_only(self) -> None:
        """Every point a visitor's chart draws is one of `earlier`'s lines (`for_visitors`)
        or the present current score; nothing else the payload carries becomes a point."""
        fold = self._fold(self.VISITOR)
        points = re.findall(r"<circle[^>]*><title>([^<]*)</title>", fold)
        assert [point.split(",")[0] for point in points] == ["0.500", "0.800", "0.610"]
        assert points[-1].endswith("the present score")
        assert "0.333" not in fold.split("<ul")[0]
        # The version is written beside each change, and the line is not joined across it.
        assert ">version 12<" in fold and ">version 13<" in fold
        assert 'stroke-dasharray="3 3"' in fold
        assert 'href="/score-history"' in fold and 'id="earlierScores"' in fold
        assert "weights v" not in fold.lower()

    def test_a_provisional_present_score_is_no_point(self) -> None:
        fold = self._fold({**self.VISITOR, "status": "provisional"})
        points = re.findall(r"<circle[^>]*><title>([^<]*)</title>", fold)
        assert [point.split(",")[0] for point in points] == ["0.500", "0.800"]
        assert "The present score is provisional" in fold

    def test_the_maintainer_draws_the_ledger(self) -> None:
        """With the ledger sent (the local application, an admin), the chart draws it and
        `earlier` is not read; a refused line is hollow."""
        ledger = [{"index": 0.3, "runs": 4, "measured": "2026-10-01", "weights_version": "v13",
                   "viable": False, "reasons": ["why"]}]
        fold = self._fold({**self.VISITOR, "history": ledger})
        points = re.findall(r"<circle([^>]*)><title>([^<]*)</title>", fold)
        assert [title.split(",")[0] for _, title in points] == ["0.300", "0.610"]
        assert 'fill="var(--panel)"' in points[0][0] and "refused" in points[0][1]
        assert "(refusedScore)" in fold

    def test_a_game_with_no_earlier_score_has_no_fold(self) -> None:
        assert self._fold({**self.VISITOR, "earlier": []}) == ""

    def test_admin_lists_earlier_scores(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.site import pages

        page = pages.admin("me")
        assert 'id="score-history"' in page
        script = (measurement.ROOT / "chess_variant_sandbox/site/static/admin.js").read_text(
            encoding="utf-8")
        assert "scoreHistory(payload.history || [])" in script
        runs = tmp_path / "runs"
        history.append("a" * 12, runs, history.entry_for(
            1, {"index": 0.5, "runs": [{}], "basis": None}, None, "2026-09-01", "replay",
            ("why",)))
        [row] = history.admin_rows(runs, {"a" * 12: "alpha"})
        assert row["name"] == "alpha" and row["entries"][0]["index"] == 0.5


OLD = "0123456789ab"  # the game's hash before an edit of its mechanics, on the boards
RETIRED = "deaddeaddead"  # a board game with no file now


def _boards(digest: str) -> dict[str, Any]:
    """A boards record in `history_boards.json`'s shape: two snapshots, the game under its
    old hash in the first (joined by file) and under both hashes in the second."""
    return {
        "about": "a test's boards", "written_by": "tests/test_history.py",
        "snapshots": [
            {"id": "v3", "weights_version": "v3", "date": "2026-08-20",
             "basis": "one run, calibration corpus", "source": "runs/v3/*/index-v3.json",
             "entries": [
                 {"hash": OLD, "name": "old lancers", "index": 0.5, "viable": True},
                 {"hash": RETIRED, "name": "gone", "index": 0.4, "viable": False}]},
            {"id": "v12a", "weights_version": "v12", "date": "2026-10-03",
             "basis": "mean of four runs", "source": "runs/v12-rescore/compare.json",
             "entries": [
                 {"hash": OLD, "name": "old lancers", "index": 0.1, "viable": False},
                 {"hash": digest, "name": "lancers", "index": 0.55, "viable": True}]},
        ],
        "same_game": [{"from": OLD, "to": digest, "how": "file", "file": "archive/x.json",
                       "task": "Q1, a test", "old_name": "old lancers", "name": "lancers"}],
    }


def _line(world, n: int, index: float, weights: str, measured: str, **moved: Any) -> None:  # noqa: F811
    outgoing = {"index": index, "runs": [{}] * 4, "viable": True,
                "weights_version": weights, "basis": {"weights_version": weights}}
    entry = history.entry_for(n, outgoing, {"weights_version": weights, **moved},
                              measured, "replay", ("why",))
    history.append(world["hash"], world["runs"], entry)


class TestTheLibraryHistory:
    """SH2 (`docs/plan-score-history.md` §2.2): the payload `GET /api/library/history`
    sends, its visitor's copy, and that the listing's cache holds it."""

    @pytest.fixture
    def listed(self, world, game, monkeypatch) -> Any:  # noqa: F811
        from chess_variant_sandbox.web import library

        monkeypatch.setattr(history, "boards", lambda: _boards(world["hash"]))
        # A listing of files written this instant is never cached (`library._SETTLE_NS`).
        monkeypatch.setattr(library, "_SETTLE_NS", 0)
        _stored(world, index=0.8)
        _line(world, 1, 0.6, "v11", "2026-10-01", net="a")
        _line(world, 2, 0.7, "v12", "2026-11-05", net="b")
        _line(world, 3, 0.75, "v12", "2026-11-06", engine="c")

        def listing() -> list[Any]:
            return library.discover(world["rulesets"], world["runs"], world["archive"])

        return listing

    def test_the_shape(self, world, listed) -> None:  # noqa: F811
        from chess_variant_sandbox.web import library

        payload = library.library_history(listed(), world["runs"])
        assert set(payload) == {"currentVersion", "snapshots", "games", "cohort", "joins",
                                "maintainer"}
        assert payload["currentVersion"] == DEFAULT_WEIGHTS and payload["maintainer"]
        # The boards', then the ledger's versions no older than the boards' last (v11 is a
        # board column already), then the current one.
        assert [(s["id"], s["kind"]) for s in payload["snapshots"]] == [
            ("v3", "board"), ("v12a", "board"), ("v12", "ledger"),
            (DEFAULT_WEIGHTS, "current")]
        for snapshot in payload["snapshots"]:
            assert {"id", "kind", "weights_version", "date", "basis"} <= set(snapshot)
        assert payload["snapshots"][2]["date"] == "2026-11-06"
        [row] = payload["games"]
        assert set(row) == {"hash", "name", "source", "points", "current", "days"}
        assert row["hash"] == world["hash"] and row["source"] == "archived"
        assert row["points"] == {
            "v3": {"index": 0.5, "viable": True, "as": "old lancers", "how": "file"},
            "v12a": {"index": 0.55, "viable": True},  # its own hash beats the join
            "v12": {"index": 0.75, "viable": True},   # the last line under v12
            DEFAULT_WEIGHTS: {"index": 0.8, "viable": True},
        }
        current = row["current"]
        assert current["status"] == "current" and current["revision"] == DEFAULT_WEIGHTS
        assert current["index"] == 0.8 and current["runs"] == 4 and current["measured"]
        assert [day["index"] for day in row["days"]] == [0.6, 0.7, 0.75]
        assert [day["why"] for day in row["days"]] == [
            "the players changed", "the players changed", "its engine changed"]
        assert row["days"][0]["basis"] == {"weights_version": "v11"}
        assert payload["cohort"] == [world["hash"]]
        assert payload["joins"] == _boards(world["hash"])["same_game"]
        json.dumps(payload)  # it travels as JSON

    def test_a_visitor_is_sent_the_numbers_and_no_basis(self, world, listed) -> None:  # noqa: F811
        from chess_variant_sandbox.web import library

        payload = library.library_history(listed(), world["runs"])
        sent = history.visitors_history(payload)
        assert sent["maintainer"] is False
        assert all("source" not in snapshot for snapshot in sent["snapshots"])
        [row] = sent["games"]
        assert row["points"] == payload["games"][0]["points"]
        for day in row["days"]:
            assert set(day) == {"index", "viable", "runs", "weights_version", "measured",
                                "replaced", "why"}
        # The maintainer's copy is untouched.
        assert "basis" in payload["games"][0]["days"][0]
        assert "source" in payload["snapshots"][0]

    def test_a_provisional_score_is_marked_and_never_a_point(self, world, listed) -> None:  # noqa: F811
        from chess_variant_sandbox.web import library

        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["basis"] = {**stored["basis"], "engine": "another-core"}
        path.write_text(json.dumps(stored), encoding="utf-8")

        payload = library.library_history(listed(), world["runs"])
        [row] = payload["games"]
        assert row["current"]["status"] == "provisional" and row["current"]["index"] == 0.8
        assert DEFAULT_WEIGHTS not in row["points"]
        assert payload["cohort"] == []
        assert payload["snapshots"][-1]["date"] is None
        sent = history.visitors_history(payload)
        assert sent["games"][0]["current"]["status"] == "provisional"

    def test_it_is_held_while_the_listing_is(self, world, listed) -> None:  # noqa: F811
        from chess_variant_sandbox.web import library

        first = library.library_history(listed(), world["runs"])
        assert library.library_history(listed(), world["runs"]) is first
        _line(world, 4, 0.78, DEFAULT_WEIGHTS, "2026-11-08", engine="d")
        again = library.library_history(listed(), world["runs"])
        assert again is not first and len(again["games"][0]["days"]) == 4

    def test_both_servers_answer_it(self, world, listed) -> None:  # noqa: F811
        from chess_variant_sandbox.site import routes
        from chess_variant_sandbox.web.server import App

        app = App(world["rulesets"], world["runs"], archive_dir=world["archive"])
        payload = app.get("/api/library/history")
        assert [row["hash"] for row in payload["games"]] == [world["hash"]]
        found = routes.find("GET", "/api/library/history")
        assert found is not None and found.visitor == "" and found in routes.ALLOWED

    def test_a_design_runs_candidate_is_not_in_it(self, world, listed) -> None:  # noqa: F811
        from chess_variant_sandbox.web import library

        everything = listed()
        assert {game.source for game in library.history_games(everything)} <= {
            *library.LIBRARY_SOURCES, library.PREVIEW}


class TestDelivery:
    def test_a_delivery_under_another_basis_keeps_what_it_replaces(
        self, world, tmp_path  # noqa: F811
    ) -> None:
        main = world["runs"]
        _stored(world, index=0.8, **NET)
        other = {"runs": tmp_path / "other", "hash": world["hash"]}
        incoming = _stored(other, index=0.6)
        before = (corpus.measurement_dir(world["hash"], main) / "canonical.json").read_bytes()

        assert history.deliver(other["runs"], main) == [world["hash"]]

        kept = history.previous_root(world["hash"], main) / "1"
        assert (kept / "canonical.json").read_bytes() == before
        assert (kept / f"{world['hash']}-0" / "games.jsonl").exists()
        stored = json.loads((corpus.measurement_dir(world["hash"], main) / "canonical.json")
                            .read_text(encoding="utf-8"))
        assert stored["index"] == 0.6 and stored["basis"] == incoming
        [line] = history.read(world["hash"], main)
        assert line.index == 0.8 and line.to == history.digest_of(incoming)
        assert line.needs == "replay" and "net" in line.moved

    def test_a_delivery_under_the_same_basis_keeps_nothing(self, world, tmp_path) -> None:  # noqa: F811
        main = world["runs"]
        _stored(world, index=0.8)
        other = {"runs": tmp_path / "other", "hash": world["hash"]}
        _stored(other, index=0.8)

        history.deliver(other["runs"], main)

        assert history.read(world["hash"], main) == []
        assert not history.previous_root(world["hash"], main).exists()


class TestItMovesNothing:
    def test_it_is_in_no_fingerprint_set(self) -> None:
        for name in ("history.py", "history_boards.json"):
            ours = (measurement.ROOT / "chess_variant_sandbox" / name).resolve()
            for kind in ("engine", "metrics", "analysis"):
                assert ours not in {path.resolve() for path in measurement.fingerprinted(kind)}, \
                    (name, kind)

    def test_no_metric_imports_it(self) -> None:
        metrics = measurement.ROOT / "chess_variant_sandbox" / "metrics"
        for path in metrics.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "import history" not in text and "history import" not in text, path


# ------------------------------------------------------------------ the boards before the ledger
# `docs/plan-score-history.md` §2.2 (SH1): `history_boards.json`'s shape, its joins, and that
# no entry carries a value its source does not.

TREE = Path(__file__).resolve().parents[1]
_HEX = re.compile(r"^[0-9a-f]{12}$")
# A record's build script: `docs/` is not in the public copy, so the two tests that run it
# are this checkout's.
_BUILD_PATH = "docs/reference/score-history/build_boards.py"
_BUILD = private_paths.path(_BUILD_PATH)


def _campaign_runs() -> Path | None:
    """The `runs/` holding the calibration campaigns: this tree's, or the main checkout's
    when this is a worktree. ``None`` where neither holds them (the box, a cloud worker)."""
    candidates = [TREE / "runs"]
    dot_git = TREE / ".git"
    if dot_git.is_file():
        pointer = dot_git.read_text(encoding="utf-8").strip().removeprefix("gitdir:").strip()
        candidates.append(Path(pointer).parents[2] / "runs")  # <main>/.git/worktrees/<name>
    for runs in candidates:
        if (runs / "v12-rescore" / "compare.json").exists():
            return runs
    return None


def _build() -> Any:
    spec = importlib.util.spec_from_file_location("build_boards", _BUILD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _norm(name: str) -> str:
    return name.strip().rstrip(".").lower()


class TestTheBoards:
    def test_the_snapshots_are_v3_to_v12_at_adoption_in_order(self) -> None:
        snapshots = history.boards()["snapshots"]
        assert [s["id"] for s in snapshots] == ["v3", "v5", "v6", "v8", "v9", "v10", "v11", "v12a"]
        dates = [s["date"] for s in snapshots]
        assert dates == sorted(dates) and dates[-1] == "2026-10-03"  # the ledger began then
        for snapshot in snapshots:
            assert set(snapshot) >= {"id", "weights_version", "date", "basis", "source", "entries"}
            assert snapshot["id"].startswith(snapshot["weights_version"])
            assert snapshot["basis"] and snapshot["entries"]
            hashes = [entry["hash"] for entry in snapshot["entries"]]
            assert len(hashes) == len(set(hashes)), snapshot["id"]
            for entry in snapshot["entries"]:
                assert set(entry) == {"hash", "name", "index", "viable"}
                assert _HEX.match(entry["hash"]) and entry["name"]
                assert entry["index"] is None or 0.0 <= entry["index"] <= 1.0
                assert entry["viable"] in (True, False, None)

    def test_no_snapshot_is_the_current_version(self) -> None:
        # The current column is `canonical.json`'s; every version after v12 is the ledger's.
        versions = {s["weights_version"] for s in history.boards()["snapshots"]}
        assert DEFAULT_WEIGHTS not in versions

    def test_a_borrowed_hash_names_the_snapshot_it_came_from(self) -> None:
        snapshots = {s["id"]: s for s in history.boards()["snapshots"]}
        borrowing = [s for s in snapshots.values() if "hashes_from" in s]
        assert borrowing
        for snapshot in borrowing:
            lender = {_norm(e["name"]): e["hash"] for e in snapshots[snapshot["hashes_from"]]["entries"]}
            for entry in snapshot["entries"]:
                assert lender[_norm(entry["name"])] == entry["hash"], entry

    def test_every_hash_join_is_a_legacy_name_identity(self) -> None:
        legacy = set(LEGACY_NAMES.values())
        found = [join for join in history.boards()["same_game"] if join["how"] == "hash"]
        assert found
        for join in found:
            assert join["from"] == join["to"] and join["to"] in legacy, join
            spec = load_spec(TREE / join["file"])
            assert spec_hash(spec) == join["to"] and spec["name"] == join["name"], join

    def test_every_file_join_names_a_file_in_the_tree(self) -> None:
        found = [join for join in history.boards()["same_game"] if join["how"] == "file"]
        assert found
        for join in found:
            assert join["file"].split("/")[0] in ("archive", "rulesets"), join
            assert spec_hash(load_spec(TREE / join["file"])) == join["to"] != join["from"], join
            assert join["task"], join
            if join["from"] in edits.EDITED:
                assert edits.current(join["from"]) == join["to"], join

    def test_every_join_starts_at_a_recorded_hash(self) -> None:
        data = history.boards()
        recorded = {e["hash"] for s in data["snapshots"] for e in s["entries"]}
        froms = [join["from"] for join in data["same_game"]]
        assert set(froms) <= recorded and len(froms) == len(set(froms))
        assert {join["how"] for join in data["same_game"]} <= {"hash", "file"}

    def test_no_entry_carries_a_value_its_source_does_not(self) -> None:
        runs = _campaign_runs()
        if runs is None:
            pytest.skip("the calibration campaigns' runs/ is not on this machine")
        for snapshot in history.boards()["snapshots"]:
            source = _source_entries(runs, snapshot)
            if source is None:
                continue  # v6: a table in a document, names only
            for entry in snapshot["entries"]:
                assert source[entry["hash"]] == entry, (snapshot["id"], entry)

    @private_paths.needs(_BUILD_PATH)
    def test_the_build_refuses_a_changed_snapshot(self) -> None:
        build = _build()
        held = history.boards()
        assert build.refusal(None, held) is None
        assert build.refusal(held, {**held, "same_game": []}) is None  # a join may change
        moved = json.loads(json.dumps(held))
        moved["snapshots"][0]["entries"][0]["index"] = 0.5
        assert "v3" in build.refusal(held, moved)
        assert build.refusal(held, {**held, "snapshots": held["snapshots"][:-1]})

    @private_paths.needs(_BUILD_PATH)
    def test_the_committed_file_is_what_the_build_writes(self) -> None:
        assert _build().render(history.boards()) == history.BOARDS_FILE.read_text(encoding="utf-8")


def _source_entries(runs: Path, snapshot: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """A snapshot's entries re-read from its source, by hash; ``None`` for v6's table."""
    source = snapshot["source"]
    if not source.startswith("runs/"):
        return None

    def entry(digest: str, name: str, index: Any, viable: Any) -> dict[str, Any]:
        return {"hash": digest, "name": name, "index": index, "viable": viable}

    if "*" in source:
        found = {}
        for path in runs.glob(source.removeprefix("runs/")):
            data = json.loads(path.read_text(encoding="utf-8"))
            found[data["ruleset_hash"]] = entry(data["ruleset_hash"], data["ruleset_name"],
                                                data.get("total"), data.get("viable"))
        return found
    data = json.loads((runs / source.removeprefix("runs/")).read_text(encoding="utf-8"))
    if snapshot["id"] == "v10":
        return {h: entry(h, r["name"], r.get("index"), r.get("viable")) for h, r in data.items()}
    if snapshot["id"] == "v11":
        rows = [r["canonical"] for r in data.values()]
        return {r["hash"]: entry(r["hash"], r["name"], r.get("index"), r.get("viable")) for r in rows}
    # v12a: the board records names; the entry's hash is v11's (`hashes_from`).
    by_name = {e["name"]: e["hash"] for e in snapshot["entries"]}
    return {by_name[r["name"]]: entry(by_name[r["name"]], r["name"], r.get("v12"), r.get("viable"))
            for r in data["board"]}
