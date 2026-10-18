"""The convergence record (Q650, `docs/archive/plan-confidence.md` §3.2 step 4 and §4 T5).

What these hold down:

- **A pair that differs only in its players** gets `runs/<hash>/convergence.json`, in the
  shape `trust.read` consumes (round-tripped through it here): ``move``, ``sd``, ``to``.
- **A pair that also moved the scoring** is read after the outgoing runs are re-scored
  under the current metrics from their kept games; the re-score writes nothing.
- **A pair that moved anything else** (the core, the tables, the seeds) gets no verdict,
  and neither does one whose players did not change.
- A record already there is kept under the replaced measurement's `previous/<n>/`.

Nothing plays: `test_history`'s stubs, on `test_corpus`'s five-square game.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_corpus import SPEC, world  # noqa: F401  (`world` is a fixture)
from test_history import NET, _stored, _tree, game, stubbed  # noqa: F401  (fixtures)

from chess_variant_sandbox import convergence, corpus, history, measurement, trust


def _record(world) -> dict[str, Any] | None:  # noqa: F811
    path = corpus.measurement_dir(world["hash"], world["runs"]) / trust.CONVERGENCE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _trust(world, tmp_path: Path) -> trust.Trust:  # noqa: F811
    return trust.read(world["hash"], world["runs"], SPEC, nets_dir=tmp_path / "no-nets")


class TestComparing:
    BASE = {"ruleset_hash": "h", "weights_version": "v11", "agents": ["a"], "seeds": [0],
            "engine": "e", "metrics": "m", "analysis": "x", "tables": "t", "players": "p",
            "agents_source": "s"}

    def test_the_players_alone(self) -> None:
        pairing = convergence.compare(self.BASE, {**self.BASE, "net": "n=net:ab"})
        assert pairing == convergence.Pairing(("net",), (), ())

    def test_the_scoring_is_a_re_score_not_a_mix(self) -> None:
        pairing = convergence.compare(
            self.BASE, {**self.BASE, "players": "q", "metrics": "m2", "weights_version": "v12"})
        assert pairing.players == ("players",)
        assert pairing.rescore == ("weights_version", "metrics")
        assert not pairing.mixed

    def test_the_core_or_the_seeds_make_it_mixed(self) -> None:
        pairing = convergence.compare(
            self.BASE, {**self.BASE, "agents": ["b"], "engine": "e2", "seeds": [0, 1]})
        assert pairing.mixed == ("seeds", "engine")

    def test_the_agents_code_counts_only_where_the_players_cannot_be_read(self) -> None:
        moved = {**self.BASE, "agents_source": "s2"}
        assert not convergence.compare(self.BASE, moved).players
        unread = {**self.BASE, "players": ""}
        assert convergence.compare(unread, {**unread, "agents_source": "s2"}).players == (
            "agents_source",)


class TestAPlayersOnlyReplacement:
    def test_a_move_of_twice_the_spread_is_written_and_trust_questions(
        self, world, game, stubbed, tmp_path  # noqa: F811
    ) -> None:
        old = _stored(world, index=0.8, **NET)

        new = corpus.refresh(game, runs_dir=world["runs"], workers=1)

        record = _record(world)
        assert record is not None
        assert record["from"] == trust.measurement_digest(old)
        assert record["to"] == trust.measurement_digest(new.basis.as_dict())
        assert record["from_index"] == 0.8 and record["to_index"] == 0.6
        assert record["move"] == pytest.approx(-0.2)
        assert record["sd"] == pytest.approx(0.0115, abs=1e-4)
        assert record["verdict"] == convergence.MOVED and record["rescored"] is None
        assert record["players"] == ["net"] and record["previous"] == "previous/1"
        assert record["basis"]["rule"] == trust.RULE.convergence_sds
        # The shape `trust` reads: current against the new score, a finding.
        item = _trust(world, tmp_path).item(trust.CONVERGENCE)
        assert (item.state, item.reading) == (trust.CURRENT, trust.FINDS_MORE)
        assert item.detail == {"move": pytest.approx(0.2), "sd": record["sd"]}
        assert _trust(world, tmp_path).level == trust.QUESTIONED
        [line] = history.read(world["hash"], world["runs"])
        assert line.convergence == convergence.MOVED

    def test_a_move_under_twice_the_spread_holds_and_trust_checks(
        self, world, game, stubbed, tmp_path  # noqa: F811
    ) -> None:
        _stored(world, index=0.8, **NET)
        stubbed["totals"][:] = [0.79, 0.81, 0.8, 0.8]

        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        record = _record(world)
        assert record is not None and record["verdict"] == convergence.HELD
        assert record["move"] == 0.0
        item = _trust(world, tmp_path).item(trust.CONVERGENCE)
        assert (item.state, item.reading) == (trust.CURRENT, trust.AGREES)
        assert _trust(world, tmp_path).level == trust.CHECKED


class TestAReplacementThatAlsoMovedTheScoring:
    def test_the_outgoing_runs_are_re_scored_first_and_nothing_is_written_to_them(
        self, world, game, stubbed  # noqa: F811
    ) -> None:
        _stored(world, index=0.8, metrics="moved", **NET)
        stubbed["totals"][:] = [0.6, 0.62]

        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        # Four scores for the new runs, four for the kept ones under today's scorer.
        assert len(stubbed["score"]) == 8
        record = _record(world)
        assert record is not None
        # The outgoing side is the re-score's mean, not the stored 0.8.
        assert record["from_index"] == 0.61 and record["to_index"] == 0.61
        assert record["verdict"] == convergence.HELD
        assert record["rescored"]["moved"] == ["metrics"]
        assert record["rescored"]["reanalysed"] is False
        kept = history.previous_root(world["hash"], world["runs"]) / "1"
        # The re-score wrote no index file beside the kept games.
        assert not list(kept.glob(f"{world['hash']}-*/index-*.json"))

    def test_rescore_kept_reads_the_kept_games_and_writes_nothing(
        self, world, game, stubbed, monkeypatch  # noqa: F811
    ) -> None:
        _stored(world, index=0.8)
        kept = world["runs"] / "kept"
        kept.mkdir()
        for seed in corpus.LIBRARY_SEEDS:
            name = f"{world['hash']}-{seed}"
            (world["runs"] / name).rename(kept / name)
        analysed: list[int] = []
        real = corpus.analyse_corpus
        monkeypatch.setattr(corpus, "analyse_corpus",
                            lambda *a, **k: (analysed.append(1), real(*a, **k))[1])
        stubbed["totals"][:] = [0.5, 0.7]
        before = _tree(kept)

        totals = corpus.rescore_kept(game, kept, corpus.LIBRARY_SEEDS, reanalyse=False)
        assert totals == [0.5, 0.7, 0.5, 0.7] and not analysed
        corpus.rescore_kept(game, kept, corpus.LIBRARY_SEEDS, reanalyse=True)
        assert len(analysed) == 4
        assert _tree(kept) == before


class TestNoVerdict:
    def test_a_moved_core_is_a_mixed_cause(self, world, game) -> None:  # noqa: F811
        _stored(world, index=0.8, engine="another-core", **NET)

        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        assert _record(world) is None
        [line] = history.read(world["hash"], world["runs"])
        assert line.convergence.startswith(convergence.NO_VERDICT)
        assert "engine" in line.convergence

    def test_unchanged_players_say_nothing_about_players(self, world, game) -> None:  # noqa: F811
        _stored(world, index=0.8, weights_version="v0")

        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        assert _record(world) is None
        [line] = history.read(world["hash"], world["runs"])
        assert line.convergence == f"{convergence.NO_VERDICT}: the players did not change"

    def test_an_earlier_record_stays_where_no_new_verdict_is_written(
        self, world, game, tmp_path  # noqa: F811
    ) -> None:
        """A re-score back onto the basis the record compared into: no verdict of its own,
        and the record it found is neither moved nor dropped. (Its reading stale against
        another basis is `test_trust.py`'s.)"""
        _stored(world, index=0.8, **NET)
        corpus.refresh(game, runs_dir=world["runs"], workers=1)
        first = _record(world)
        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["basis"] = {**payload["basis"], "weights_version": "v0"}
        path.write_text(json.dumps(payload), encoding="utf-8")

        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        assert _record(world) == first
        assert history.read(world["hash"], world["runs"])[1].convergence.startswith(
            convergence.NO_VERDICT)
        item = _trust(world, tmp_path).item(trust.CONVERGENCE)
        assert (item.state, item.reading) == (trust.CURRENT, trust.FINDS_MORE)


class TestARecordIsNeverLost:
    def test_the_record_it_replaces_is_kept_under_previous(self, world, game) -> None:  # noqa: F811
        _stored(world, index=0.8, **NET)
        corpus.refresh(game, runs_dir=world["runs"], workers=1)
        first = _record(world)
        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["basis"] = {**payload["basis"], "net": "other=net:cdcdcdcdcdcd"}
        path.write_text(json.dumps(payload), encoding="utf-8")

        corpus.refresh(game, runs_dir=world["runs"], workers=1)

        kept = history.previous_root(world["hash"], world["runs"]) / "2" / trust.CONVERGENCE_FILE
        assert json.loads(kept.read_text(encoding="utf-8")) == first
        assert _record(world) != first

    def test_a_delivery_reads_convergence_too(self, world, tmp_path) -> None:  # noqa: F811
        main = world["runs"]
        _stored(world, index=0.8, **NET)
        other = {"runs": tmp_path / "other", "hash": world["hash"]}
        _stored(other, index=0.805)

        history.deliver(other["runs"], main, judge=corpus.judge_convergence)

        record = _record(world)
        assert record is not None and record["verdict"] == convergence.HELD
        assert history.read(world["hash"], main)[0].convergence == convergence.HELD


class TestItMovesNothing:
    def test_it_is_in_no_fingerprint_set(self) -> None:
        ours = (measurement.ROOT / "chess_variant_sandbox/convergence.py").resolve()
        for kind in ("engine", "metrics", "analysis"):
            assert ours not in {path.resolve() for path in measurement.fingerprinted(kind)}, kind

    def test_no_metric_imports_it(self) -> None:
        metrics = measurement.ROOT / "chess_variant_sandbox" / "metrics"
        for path in metrics.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "import convergence" not in text and "convergence import" not in text, path
