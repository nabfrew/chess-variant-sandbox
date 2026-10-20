"""Sampled re-certification (C6): a score kept through a change, by a sample, and labelled.

`docs/reference/compute-economy.md` §3.5, under the user's answer B of 2026-10-02: after a
change to the code that plays games which should move no game, a library score may stay
current because a sample of its games replayed byte for byte -- and the stamp says so
(`measurement.Certificate`), and the page shows the score *re-certified from a sample*.
What is held down here, on a five-square game of its own (never chess):

- **The stamp.** A certificate stands in for what played the games, so a score is current
  exactly while the code its sample ran on is; it stands in for nothing else (a roster, a
  net, a browser's games); a stamp without one is written as it always was; one that cannot be
  read is no basis at all.
- **The path.** `corpus.recertify` keeps the games and the numbers, stamps the sample, and
  plays again in full a run whose sample differs. A re-score carries the certificate; a
  full replay clears it.
- **The sample.** `adopt.spread` is the pre-registered 32 a run, spread over the
  opponents; a change outside the sample is missed, which is why the label exists.
- **The fit**, one game in eight of its kept games and its match's.
- **The page** draws the mark beside the number, in a visitor's words on the site.

The games are four on a five-square board, which is where the `slow` marks come from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_corpus import SPEC, _find, _Scored, world  # noqa: F401  (`world` is a fixture)
from test_web import _in_page

from chess_variant_sandbox import corpus
from chess_variant_sandbox.measurement import UNVERIFIED, Basis, Certificate, FitReading
from chess_variant_sandbox.tournament import adopt, fitting
from chess_variant_sandbox.tournament.record import read_records, write_records
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament
from chess_variant_sandbox.web import library

OLD_ENGINE = "a0b1c2d3e4f5a6b7-0011223344556677"
"""An `engine` stamp no checkout has: the code that played the games, since changed."""


def _basis(**fields: Any) -> Basis:
    base: dict[str, Any] = {
        "ruleset_hash": "c0ffee000001", "weights_version": "v12", "agents": ("random",),
        "games_per_pairing": 4, "seeds": (0,), "engine": "src-built", "metrics": "m",
        "tables": "t", "analysis": "a", "players": "p", "agents_source": "s",
    }
    return Basis(**{**base, **fields})


# --------------------------------------------------------------------- the stamp


class TestTheStamp:
    def test_a_sample_on_todays_code_keeps_a_score_current(self) -> None:
        now = _basis()
        played = _basis(engine=OLD_ENGINE, tables="old tables", players="old players")
        assert played.stale_against(now).needs == "replay"

        kept = played.certify(now, 128, 1024)

        assert kept.stale_against(now).current
        # What played the games is kept as it was, beside the certificate.
        assert kept.engine == OLD_ENGINE and kept.tables == "old tables"
        assert kept.certified is not None
        assert kept.certified.words() == "certified: sample 128/1024 on built"

    def test_a_later_change_stales_it_again_and_says_since_when(self) -> None:
        kept = _basis(engine=OLD_ENGINE).certify(_basis(), 128, 1024)
        later = _basis(engine="src-rebuilt")

        staleness = kept.stale_against(later)

        assert staleness.needs == "replay"
        assert any("since it was re-certified from a sample" in r for r in staleness.reasons)

    def test_it_stands_in_for_nothing_but_the_code_that_plays(self) -> None:
        now = _basis()
        for moved in (
            {"agents": ("greedy",)},
            {"played": "browser"},
            {"net": "proc=net:000000000000"},
            {"ruleset_hash": "c0ffee000002"},
        ):
            stamp = _basis(engine=OLD_ENGINE, **moved)
            assert stamp.certifiable(now).stored_needs == "replay", moved
        assert _basis(engine=OLD_ENGINE).certifiable(now).stored_needs == ""

    def test_a_stamp_without_one_is_written_as_it_always_was(self) -> None:
        assert "certified" not in _basis().as_dict()
        kept = _basis(engine=OLD_ENGINE).certify(_basis(), 32, 256)
        assert Basis.from_dict(kept.as_dict()) == kept

    def test_one_that_cannot_be_read_is_no_basis(self) -> None:
        stamp = _basis().as_dict()
        for broken in ({"sample": 0, "games": 256, "engine": "e", "tables": "t"},
                       {"sample": 300, "games": 256, "engine": "e", "tables": "t"},
                       {"sample": 32, "games": 256, "tables": "t"},
                       "sample 32/256"):
            assert Basis.from_dict({**stamp, "certified": broken}) is None, broken
        with pytest.raises(ValueError):
            Certificate(sample=1, games=1, engine="", tables="")

    def test_a_sampled_fit_is_this_cores_answer_for_the_score(self) -> None:
        """The fit store is unverified on the new core; the certificate vouches for it."""
        reading = FitReading(player="fit:aaaaaaaaaaaa", made_on="0011223344556677",
                             verified_on=frozenset({"0011223344556677"}))
        now = _basis(fit=UNVERIFIED, fit_reading=reading)
        played = _basis(engine=OLD_ENGINE, fit="fit:aaaaaaaaaaaa")
        assert played.stale_against(now).refit

        kept = played.certify(now, 128, 1024)

        assert kept.certified is not None and kept.certified.fit == "fit:aaaaaaaaaaaa"
        assert kept.stale_against(now).current
        # A store that has since become another player is a replay, sample or not.
        other = FitReading(player="fit:bbbbbbbbbbbb", made_on="x", verified_on=frozenset())
        assert kept.stale_against(
            _basis(fit=UNVERIFIED, fit_reading=other)
        ).needs == "replay"


# -------------------------------------------------------------------- the sample


def _records(per_pairing: int) -> list[Any]:
    config = TournamentConfig(agents=("greedy", "minimax:nodes=50,threats=0"),
                              games_per_pairing=per_pairing, seed=0, workers=1)
    return run_tournament(SPEC, config).records


class TestTheSampleSize:
    def test_it_is_four_with_each_agent_first_spread_over_the_opponents(self) -> None:
        """The pre-registered 32 of a run's 256: eight agents, four each (§3.5)."""
        agents = tuple(corpus.CANONICAL_AGENTS)
        config = TournamentConfig(agents=agents, games_per_pairing=4, seed=0)
        from chess_variant_sandbox.tournament.runner import schedule

        records = [type("R", (), {"agents": (first, second)})()
                   for _seed, first, second in schedule(config)]
        chosen = adopt.spread(records, config, adopt.RECERTIFY_PER_AGENT)  # type: ignore[arg-type]

        assert len(chosen) == 32 == adopt.RECERTIFY_PER_AGENT * len(agents)
        for agent in agents:
            mine = [index for index in chosen if records[index].agents[0] == agent]
            assert len(mine) == 4
            # Four different opponents, not four games against one.
            assert len({records[index].agents[1] for index in mine}) == 4


@pytest.mark.slow
class TestARunInPlace:
    ROSTER = TournamentConfig(agents=("greedy", "minimax:nodes=50,threats=0"),
                              games_per_pairing=2, seed=0, workers=1)

    @pytest.fixture
    def run(self, tmp_path: Path) -> Path:
        from chess_variant_sandbox.tournament.analysis import analyse_corpus

        out = tmp_path / "c0ffee-0"
        tournament = run_tournament(SPEC, self.ROSTER)
        tournament.save(out)
        analyse_corpus(SPEC, tournament.records, seed=0, workers=1).save(out / "analysis.json")
        return out

    def _check(self, run: Path) -> adopt.Recertification:
        return adopt.recertify_run(SPEC, 0, run, self.ROSTER, workers=1, per_agent=1)

    def _alter(self, run: Path, index: int) -> None:
        records = list(read_records(run / "games.jsonl"))
        assert len(records[index].moves) > 1, "the fixture needs a game of two moves"
        records[index].moves = list(records[index].moves)[:-1]
        write_records(records, run / "games.jsonl")

    def test_an_untouched_run_is_certified_and_nothing_is_written(self, run: Path) -> None:
        before = {path.name: path.read_bytes() for path in run.iterdir()}
        found = self._check(run)
        assert found.certified and found.sampled == 2 and found.games == 8
        assert {path.name: path.read_bytes() for path in run.iterdir()} == before

    def test_a_sampled_game_that_replays_differently_is_refused(self, run: Path) -> None:
        records = list(read_records(run / "games.jsonl"))
        sampled = adopt.spread(records, self.ROSTER, 1)
        self._alter(run, sampled[0])
        found = self._check(run)
        assert not found.certified and f"game {sampled[0]}" in found.reason

    def test_a_change_outside_the_sample_is_missed_which_is_why_it_is_labelled(
        self, run: Path
    ) -> None:
        records = list(read_records(run / "games.jsonl"))
        sampled = set(adopt.spread(records, self.ROSTER, 1))
        outside = next(index for index in range(len(records)) if index not in sampled)
        self._alter(run, outside)
        assert self._check(run).certified


# ---------------------------------------------------------------------- the path


ROSTER = ("greedy", "minimax:nodes=50,threats=0")


@pytest.fixture
def small(world, monkeypatch):  # noqa: F811
    """The game's canonical roster made two agents at one game a pairing, on one seed."""
    monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.ONE_SEED)
    monkeypatch.setattr(corpus, "canonical_agents", lambda digest, spec=None: ROSTER)
    monkeypatch.setattr(corpus, "CANONICAL_GAMES", 1)
    calls: list[_Scored] = []

    def score(spec, records, ratings, weights, analysis=None, **_ignored):
        calls.append(_Scored(analysis=analysis, weights_version=weights.version))
        return calls[-1]

    monkeypatch.setattr(corpus, "score_ruleset", score)
    corpus.measure(_find(world), runs_dir=world["runs"], workers=1)
    return world


def _canonical(game) -> Path:
    return corpus.measurement_dir(game["hash"], game["runs"]) / "canonical.json"


def _move_the_core(game) -> None:
    """What a change to the code that plays games does to the stamp: its engine moves."""
    path = _canonical(game)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["basis"]["engine"] = OLD_ENGINE
    path.write_text(json.dumps(stored), encoding="utf-8")


def _games(game) -> bytes:
    return (corpus.run_dir(game["hash"], 0, game["runs"]) / "games.jsonl").read_bytes()


def _played(game) -> list[dict[str, Any]]:
    """The run's games with the wall clock out, which no two plays of a game share."""
    path = corpus.run_dir(game["hash"], 0, game["runs"]) / "games.jsonl"
    return [{k: v for k, v in record.as_dict().items() if k not in adopt.IGNORED_FIELDS}
            for record in read_records(path)]


@pytest.mark.slow
class TestRecertifyingAScore:
    def test_the_score_is_kept_and_stamped_with_its_sample(self, small) -> None:
        _move_the_core(small)
        before = json.loads(_canonical(small).read_text(encoding="utf-8"))
        games = _games(small)
        assert corpus.status_of(small["hash"], small["runs"], small["rulesets"],
                                small["archive"]).needs == "replay"

        kept = corpus.recertify(_find(small), runs_dir=small["runs"], workers=1)

        assert kept.current and kept.basis is not None
        assert kept.basis.engine == OLD_ENGINE
        assert kept.basis.certified is not None
        assert (kept.basis.certified.sample, kept.basis.certified.games) == (4, 4)
        after = json.loads(_canonical(small).read_text(encoding="utf-8"))
        assert {k: v for k, v in after.items() if k != "basis"} == {
            k: v for k, v in before.items() if k != "basis"}
        assert _games(small) == games
        status = corpus.status_of(small["hash"], small["runs"], small["rulesets"],
                                  small["archive"])
        assert status.current and status.certified.startswith("re-certified: sample 4/4")

        # The page's payload says how large the sample was.
        [game] = [g for g in library.discover(small["rulesets"], small["runs"],
                                              small["archive"]) if g.hash == small["hash"]]
        assert game.status == library.CURRENT
        assert game.score is not None and game.score["certified"]["sample"] == 4
        assert game.summary()["certified"] == game.score["certified"]

    def test_a_run_whose_sample_differs_is_played_again(self, small) -> None:
        _move_the_core(small)
        run = corpus.run_dir(small["hash"], 0, small["runs"])
        records = list(read_records(run / "games.jsonl"))
        honest = _played(small)
        records[0].moves = list(records[0].moves)[:-1]
        write_records(records, run / "games.jsonl")

        kept = corpus.recertify(_find(small), runs_dir=small["runs"], workers=1)

        assert kept.current and kept.basis is not None
        # The only run did not hold, so the game was replayed and carries no label.
        assert kept.basis.certified is None
        assert _played(small) == honest

    def test_a_re_score_carries_the_label_and_a_full_replay_clears_it(self, small) -> None:
        _move_the_core(small)
        corpus.recertify(_find(small), runs_dir=small["runs"], workers=1)
        path = _canonical(small)
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["basis"]["metrics"] = "moved"
        path.write_text(json.dumps(stored), encoding="utf-8")

        rescored = corpus.refresh(_find(small), runs_dir=small["runs"], workers=1)

        assert rescored.current and rescored.basis is not None
        assert rescored.basis.certified is not None
        assert rescored.basis.engine == OLD_ENGINE

        replayed = corpus.refresh(_find(small), runs_dir=small["runs"], workers=1,
                                  force=True)
        assert replayed.current and replayed.basis is not None
        assert replayed.basis.certified is None
        assert replayed.basis.engine != OLD_ENGINE

    def test_what_a_sample_cannot_speak_for_is_replayed_as_before(self, small) -> None:
        path = _canonical(small)
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["basis"]["engine"] = OLD_ENGINE
        stored["basis"]["agents"] = ["random", "greedy"]
        path.write_text(json.dumps(stored), encoding="utf-8")
        lines: list[str] = []

        kept = corpus.recertify(_find(small), runs_dir=small["runs"], workers=1,
                                report=lines.append)

        assert kept.current and kept.basis is not None and kept.basis.certified is None
        assert any("a sample cannot stand in for this" in line for line in lines)


# ----------------------------------------------------------------------- the fit


@pytest.mark.slow
class TestTheFitByASample:
    @pytest.fixture
    def kept(self, tmp_path: Path) -> tuple[Any, Path]:
        found = fitting.check_fit(SPEC, store=tmp_path / "fitted", workers=1,
                                  keep=tmp_path / "kept")
        return found.fit, tmp_path / "kept"

    def test_a_check_keeps_the_games_of_the_store_that_stands(self, kept) -> None:
        fit, where = kept
        marker = json.loads((where / fitting.KEPT_FILE).read_text(encoding="utf-8"))
        assert marker["store_digest"] == fitting.store_digest(fit)
        assert list(read_records(where / fitting.FIT_GAMES_FILE))

    def test_its_kept_games_certify_it(self, kept) -> None:
        fit, where = kept
        assert adopt.recertify_fit(SPEC, fit, where, workers=1) == ""

    def test_a_kept_game_that_replays_differently_is_refused(self, kept) -> None:
        fit, where = kept
        name = (fitting.ACCEPT_GAMES_FILE
                if list(read_records(where / fitting.ACCEPT_GAMES_FILE))
                else fitting.FIT_GAMES_FILE)
        records = list(read_records(where / name))
        records[0].moves = list(records[0].moves)[:-1]
        write_records(records, where / name)
        assert adopt.recertify_fit(SPEC, fit, where, workers=1)

    def test_games_kept_for_another_store_certify_nothing(self, kept) -> None:
        fit, where = kept
        (where / fitting.KEPT_FILE).write_text('{"store_digest": "other"}', encoding="utf-8")
        assert "another store" in adopt.recertify_fit(SPEC, fit, where, workers=1)
        (where / fitting.KEPT_FILE).unlink()
        assert "no fit games are kept" in adopt.recertify_fit(SPEC, fit, where, workers=1)


# ---------------------------------------------------------------------- the page


CERTIFIED = {"sample": 128, "games": 1024, "build": "0011223344556677"}


def _card(**more: Any) -> dict[str, Any]:
    return {"hash": "c0ffee000001", "name": "the game", "index": 0.612, "status": "current",
            "revision": "v12", "measuredAt": "2026-10-20", "seeds": [0, 1000, 2000, 3000],
            "weights": "v12", "viable": True, "staleness": {"current": True},
            "certified": CERTIFIED, **more}


class TestThePage:
    def test_the_card_says_from_a_sample_inside_the_pill(self) -> None:
        for setup in ("", "ACCOUNTS = true;"):
            pill = _in_page(setup, f"scoreTag({json.dumps(_card())})")
            assert "0.612" in pill and "from a sample" in pill
            assert 'data-gloss="certifiedScore"' in pill
        plain = _in_page("", f"scoreTag({json.dumps(_card(certified=None))})")
        assert "from a sample" not in plain and "certifiedScore" not in plain

    def test_the_score_panel_marks_it_re_certified_with_the_sample(self) -> None:
        score = {"total": 0.612, "status": "current", "revision": "v12",
                 "weights_version": "v12", "families": {}, "scores": {},
                 "contributions": {}, "seeds": [0], "certified": CERTIFIED}
        panel = _in_page("", f"scoreHtml({json.dumps(score)}, {{}}, null, null)")
        assert "re-certified from a sample" in panel
        assert "Re-certified from a sample of 128 games" in panel
        site = _in_page("ACCOUNTS = true;",
                        f"scoreHtml({json.dumps(score)}, {{}}, null, null)")
        assert "kept from a sample" in site and "128 of its 1,024 games" in site
        assert "byte for byte" not in site and CERTIFIED["build"] not in site
        assert site_words.hits(site_words.visible(site)) == []

    def test_a_provisional_number_is_never_drawn_as_re_certified(self) -> None:
        estimate = _card(status="provisional")
        assert _in_page("", f"certifiedMarkHtml({json.dumps(estimate)})") == ""

    def test_the_site_glossary_says_it_in_a_visitors_words(self) -> None:
        site = _in_page("ACCOUNTS = true;", "glossEntry('certifiedScore')")
        local = _in_page("", "glossEntry('certifiedScore')")
        assert local[0] == "Re-certified from a sample"
        assert site[0] == "Kept from a sample" and site[1] != local[1]
        assert site_words.hits(site[1]) == []
