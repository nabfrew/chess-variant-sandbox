"""A score's history (Q649, `docs/plan-confidence.md` §3.2 step 3 and §4 T4).

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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from test_corpus import SPEC, _find, _Scored, world  # noqa: F401  (`world` is a fixture)

from chess_variant_sandbox import corpus, history, measurement
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.tournament import fitting


@pytest.fixture
def stubbed(world, monkeypatch) -> dict[str, list[Any]]:  # noqa: F811
    """`measure`'s fit, tournament, pass and scorer as stubs; the scorer scores
    ``totals[seed]`` so a test can say what the new measurement reads."""
    calls: dict[str, list[Any]] = {"play": [], "score": [], "totals": [0.6]}

    class _Fit:
        accepted = False
        accept_score = 0.0

    class _Checked:
        outcome = fitting.MADE_HERE
        core = "loaded"
        fit = _Fit()
        stored = _Fit()
        failures = ()

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
    monkeypatch.setattr(corpus, "check_fit", lambda spec, workers=0: _Checked())
    monkeypatch.setattr(corpus, "run_tournament", play)
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
            {"index": 0.8, "runs": 4, "measured": "2026-09-28"}]
        # A replacement that moved no player is no *was* line for a visitor.
        other = [history.Entry.from_dict({**self.LINES[0], "moved": ["engine"]})]
        assert history.for_visitors(other, 0.6) == []
        monkeypatch.setattr(history, "SHOW_WAS_TO_VISITORS", False)
        assert history.for_visitors(entries, 0.6) == []

    def test_the_page_draws_was_lines_from_the_payload_only(self) -> None:
        from page_text import PAGE

        assert "function wasHtml(lines)" in PAGE
        assert "${wasHtml(game.history)}" in PAGE

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
        ours = (measurement.ROOT / "chess_variant_sandbox/history.py").resolve()
        for kind in ("engine", "metrics", "analysis"):
            assert ours not in {path.resolve() for path in measurement.fingerprinted(kind)}, kind

    def test_no_metric_imports_it(self) -> None:
        metrics = measurement.ROOT / "chess_variant_sandbox" / "metrics"
        for path in metrics.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "import history" not in text and "history import" not in text, path
