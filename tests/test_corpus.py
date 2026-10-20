"""The pathway out of a design run: test, archive, promote.

What these hold down is a ladder rather than three commands. A run's candidates live in
`runs/`, which git ignores, and until this existed there was no code path by which one
could become anything else -- every reference to `rulesets/` in the project reads it. So
the checks are mostly about the *refusals*: what a game has to be before the corpus will
take it, and that the browser and the command line cannot come to different answers.

Nothing here plays a real tournament except the one test that says so. `measure` is the
expensive rung and is exercised on a five-square board.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.history import HISTORY_FILE
from chess_variant_sandbox.loop.archive import Archive, Entry
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.tournament import fitting

SPEC = {
    "schema_version": 1,
    "name": "lancers",
    "description": "A small skirmish game.",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "name": "captain", "royal": True, "value": 0, "moves": [
            {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
        ]},
        {"id": "L", "name": "lancer", "value": 5, "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}
        ]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "L", "squares": ["a1", "e1"]},
    ]},
    "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
              "no_progress_draw": 40},
}


@pytest.fixture
def world(tmp_path: Path):
    """A design run with one candidate in it, and empty corpus directories beside it."""
    runs = tmp_path / "runs"
    archive = Archive(runs / "design")
    digest = spec_hash(SPEC)
    archive.record(
        Entry(
            hash=digest, name="lancers", generation=1, parent="beef", status="evaluated",
            index=0.5, objective=0.5, viable=True,
            detail={"theme": "a small skirmish", "operator": "model"},
        ),
        spec=SPEC,
    )
    archive.record(Entry(hash="beef", name="the parent", generation=0, parent=None,
                         status="evaluated", index=0.4, viable=True))
    return {
        "runs": runs,
        "rulesets": tmp_path / "rulesets",
        "archive": tmp_path / "archive",
        "hash": digest,
    }


def _find(world) -> corpus.Candidate:
    return corpus.find(
        world["hash"], runs_dir=world["runs"], ruleset_dir=world["rulesets"],
        archive_dir=world["archive"],
    )


def _measured(
    world,
    index: float = 0.8,
    viable: bool = True,
    seeds: tuple[int, ...] = corpus.LIBRARY_SEEDS,
) -> None:
    """A stored canonical-roster measurement, without playing 256 games for it.

    Stamped with the basis that holds *now*, because that is what a real measurement
    taken now would carry. A fixture that omitted it would be testing the stale path
    every time, which is a different test and is written as one below. The spec is
    passed for the same reason: without it the tables fingerprint is `""`, which is
    the "this checkout cannot compile that game" reading and not the one a fixture
    standing in for a real measurement wants.

    Four runs by default, because most of what reads this fixture is about promotion and
    `promotion_blocker` asks for the library's whole seed set whatever the library is
    currently expected to have.
    """
    out = corpus.measurement_dir(world["hash"], world["runs"])
    out.mkdir(parents=True, exist_ok=True)
    violations = [] if viable else ["decisive_rate=0.1 below minimum 0.2"]
    (out / "canonical.json").write_text(json.dumps({
        "hash": world["hash"], "name": "lancers",
        "agents": list(corpus.CANONICAL_AGENTS),
        "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": list(seeds),
        "games": 256 * len(seeds),
        "index": index, "totals_used": len(seeds), "sd": 0.02, "viable": viable,
        "violations": violations,
        "weights_version": DEFAULT_WEIGHTS, "families": {}, "standings": [],
        "runs": [{"seed": seed, "index": index, "sd": 0.02, "viable": viable,
                  "violations": violations, "unreadable": [], "origin": "played"}
                 for seed in seeds],
        "basis": corpus.current_basis(world["hash"], spec=SPEC, seeds=seeds).as_dict(),
    }), encoding="utf-8")


class TestTheRoster:
    """The first half of what `test` is for: naming the ladder a committed score is on.

    It lived only in prose. `cli.DEFAULT_AGENTS` and `loop.DEFAULT_AGENTS` are two other,
    smaller rosters, and a candidate measured by either is not comparable with anything
    in the corpus -- which is exactly the comparison the archive must never contain.
    """

    def test_it_is_not_either_of_the_smaller_rosters(self) -> None:
        from chess_variant_sandbox.cli import DEFAULT_AGENTS as cli_roster
        from chess_variant_sandbox.loop.orchestrator import DEFAULT_AGENTS as loop_roster

        assert cli_roster != corpus.CANONICAL_AGENTS
        assert loop_roster != corpus.CANONICAL_AGENTS
        assert len(corpus.CANONICAL_AGENTS) > len(loop_roster)

    def test_it_can_measure_a_skill_gradient_at_all(self) -> None:
        """The rule `LoopConfig` enforces on any roster: two budgets of one agent kind.

        Six of them here rather than two, because with four the two halves
        `gradient_persistence` compares shared a point.
        """
        from chess_variant_sandbox.loop.orchestrator import _require_measurable_ladder

        _require_measurable_ladder(corpus.CANONICAL_AGENTS)
        budgets = [a for a in corpus.CANONICAL_AGENTS if "nodes=" in a]
        assert len(budgets) == 6

    def test_a_score_on_another_roster_does_not_count_as_a_measurement(self, world) -> None:
        """The protection the whole module rests on.

        A design run writes an index file too. If that counted, `promote` would admit a
        game on a number taken from a different ladder, which is the one comparison this
        project is built to refuse.
        """
        smaller = ("random", "greedy", "minimax:nodes=400", "minimax:nodes=1600")
        basis = corpus.current_basis(world["hash"], spec=SPEC)
        out = corpus.measurement_dir(world["hash"], world["runs"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "canonical.json").write_text(json.dumps({
            "hash": world["hash"], "index": 0.9, "sd": 0.01, "viable": True,
            "agents": list(smaller),
            "basis": {**basis.as_dict(), "agents": list(smaller)},
        }), encoding="utf-8")

        # Readable, and not current. The measurement is returned rather than hidden --
        # "a score about a different ladder" is more use to a person than "no score" --
        # but nothing that turns on having been tested will accept it.
        stored = corpus.read_measurement(world["hash"], world["runs"])
        assert stored is not None
        assert not stored.current
        assert "different roster" in stored.staleness.summary()
        assert not _find(world).tested


class TestFindingIt:
    def test_a_candidate_is_found_in_the_run_that_made_it(self, world) -> None:
        candidate = _find(world)
        assert candidate.stage == "candidate"
        assert candidate.run == world["runs"] / "design"
        assert candidate.lineage["generation"] == 1
        assert candidate.lineage["parent_name"] == "the parent"
        assert candidate.lineage["theme"] == "a small skirmish"

    def test_where_it_lives_now_beats_where_it_came_from(self, world) -> None:
        """A promoted game keeps its copy in the run's `specs/`, and must not read as a
        candidate afterwards -- or the interface offers to promote the corpus."""
        corpus.archive_game(_find(world), archive_dir=world["archive"], runs_dir=world["runs"])
        assert _find(world).stage == "archived"

        _measured(world)
        corpus.promote(_find(world), reason="it pins something down",
                       ruleset_dir=world["rulesets"], runs_dir=world["runs"])
        assert _find(world).stage == "committed"

    def test_an_unknown_hash_says_where_it_looked(self, world) -> None:
        with pytest.raises(corpus.CorpusError, match="no game with hash"):
            corpus.find("0123456789ab", runs_dir=world["runs"],
                        ruleset_dir=world["rulesets"], archive_dir=world["archive"])


class TestArchiving:
    """The middle rung: kept, in the repository, out of the corpus."""

    def test_it_demands_no_measurement(self, world) -> None:
        """Archiving says "this was worth inventing", which a person may say about a
        game whose tournament nobody wants to pay for."""
        path = corpus.archive_game(
            _find(world), note="worth keeping", archive_dir=world["archive"],
            runs_dir=world["runs"],
        )
        assert path.exists()
        kept = json.loads(path.read_text(encoding="utf-8"))
        assert spec_hash(kept) == world["hash"], "archiving must not change the game"

    def test_the_provenance_is_the_archive_s_and_not_the_model_s(self, world) -> None:
        """A model asked to describe its own ancestry would invent one, so none of this
        is read from the spec."""
        path = corpus.archive_game(
            _find(world), note="worth keeping", archive_dir=world["archive"],
            runs_dir=world["runs"],
        )
        provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        assert provenance["origin"] == "evolved"
        assert provenance["parents"] == ["beef"]
        assert provenance["theme"] == "a small skirmish"
        assert "worth keeping." in provenance["history"]
        assert "revising the parent" in provenance["history"]

    def test_a_credit_names_the_designer_and_moves_no_hash(self, world) -> None:
        """`--credit` writes `provenance.author` and a sentence after the reason, which
        stays the history's first sentence; provenance is outside the mechanics hash."""
        path = corpus.archive_game(
            _find(world), note="Plays well", credit="  maker1 ", archive_dir=world["archive"],
            runs_dir=world["runs"],
        )
        kept = json.loads(path.read_text(encoding="utf-8"))
        assert kept["provenance"]["author"] == "maker1"
        assert kept["provenance"]["history"].startswith("Plays well. Designed by maker1.")
        assert spec_hash(kept) == world["hash"]
        assert spec_hash(kept) == spec_hash(_find(world).spec)

    def test_the_archive_command_takes_credit(self, world, capsys) -> None:
        from chess_variant_sandbox import cli

        code = cli.main([
            "archive", world["hash"], "--credit", "maker1", "--note", "Plays well.",
            "--runs", str(world["runs"]), "--out", str(world["archive"]),
        ])
        assert code == 0
        (path,) = world["archive"].glob("*.json")
        kept = json.loads(path.read_text(encoding="utf-8"))
        assert kept["provenance"]["author"] == "maker1"
        assert kept["provenance"]["history"].startswith("Plays well. Designed by maker1.")
        assert spec_hash(kept) == world["hash"]

    def test_no_credit_leaves_the_author_alone(self, world) -> None:
        path = corpus.archive_game(
            _find(world), note="x", archive_dir=world["archive"], runs_dir=world["runs"],
        )
        kept = json.loads(path.read_text(encoding="utf-8"))
        assert "Designed by" not in kept["provenance"]["history"]

    def test_a_name_renames_the_game_and_not_only_the_file(self, world) -> None:
        """`name` is outside the mechanics hash, so this is free -- and the alternative
        is a corpus entry called "gardner (some of the pawns are now a new piece X)"."""
        path = corpus.archive_game(
            _find(world), name="Longshot", archive_dir=world["archive"],
            runs_dir=world["runs"],
        )
        assert path.name == "longshot.json"
        renamed = json.loads(path.read_text(encoding="utf-8"))
        assert renamed["name"] == "longshot"
        assert spec_hash(renamed) == world["hash"]

    def test_a_model_style_name_is_normalised_to_the_one_convention(self, world) -> None:
        """A model proposes whatever style it likes; the corpus writes only the one
        convention `TestTheCorpusNames` pins, never what a model handed it."""
        path = corpus.archive_game(
            _find(world), name="Tunnel Chess Horizontal.", archive_dir=world["archive"],
            runs_dir=world["runs"],
        )
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["name"] == "tunnel chess horizontal"
        assert CORPUS_NAME.fullmatch(written["name"])
        # Normalising the name moves no hash: `name` is outside `canonical.MECHANIC_KEYS`.
        assert spec_hash(written) == world["hash"]

    def test_a_filename_already_taken_by_another_game_is_refused(self, world) -> None:
        world["archive"].mkdir(parents=True, exist_ok=True)
        other = copy.deepcopy(SPEC)
        other["board"] = {"width": 6, "height": 6}
        (world["archive"] / "lancers.json").write_text(json.dumps(other), encoding="utf-8")
        with pytest.raises(corpus.CorpusError, match="already a different game"):
            corpus.archive_game(_find(world), archive_dir=world["archive"],
                                runs_dir=world["runs"])


# The one naming convention for the corpus, `docs/archive/plan-ux.md`'s E1 (Q258, 2026-09-24). A
# name is what the library lists, so it reads as a name in running text: lowercase words
# separated by single spaces, with nothing after the last word. A hyphen joins the two halves
# of one compound word (*one-way*, *three-check*, *half-turn*) and never stands between
# words, which is why `chess-sweeping-bishops` is `chess sweeping bishops`, beside `chess
# oriented`. `name` is outside the mechanics hash, so bringing a name into line moves no
# hash; `tests/test_schema_nine.py::FROZEN` is what says so.
CORPUS_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?(?: [a-z0-9]+(?:-[a-z0-9]+)?)*")

ROOT = Path(__file__).resolve().parents[1]


def _corpus_names() -> dict[str, str]:
    """Every committed and archived spec's top-level `name`, by path."""
    return {
        f"{folder}/{path.name}": str(json.loads(path.read_text(encoding="utf-8"))["name"])
        for folder in ("rulesets", "archive")
        for path in sorted((ROOT / folder).glob("*.json"))
    }


class TestTheCorpusNames:
    """Names are keys as well as labels, so a name out of line is a failure with a path."""

    def test_every_name_follows_the_one_convention(self) -> None:
        off = {path: name for path, name in _corpus_names().items()
               if not CORPUS_NAME.fullmatch(name)}
        assert off == {}, f"names out of the convention: {off}"

    def test_the_convention_reads_the_names_it_was_written_against(self) -> None:
        for kept in ("one-way torus chess", "three-check", "weathervane half-turn",
                     "wolves rotated 2", "chess sweeping bishops"):
            assert CORPUS_NAME.fullmatch(kept), kept
        for refused in ("Marchfield", "tunnel chess horizontal.", "chess-sweeping-bishops",
                        "torus  chess", " chess", "chess ", ""):
            assert not CORPUS_NAME.fullmatch(refused), refused

    def test_no_two_games_share_a_name(self) -> None:
        """The library lists by name and the calibration lists key by it."""
        seen: dict[str, list[str]] = {}
        for path, name in _corpus_names().items():
            seen.setdefault(name, []).append(path)
        assert {name: paths for name, paths in seen.items() if len(paths) > 1} == {}

    def test_every_name_the_calibration_keys_by_is_a_committed_game(self) -> None:
        """A rename breaks these keys silently: a probe whose name no game carries any
        more is scored as a human game and widens the band."""
        from chess_variant_sandbox.metrics.calibration import EXCLUDED_FROM_BAND
        from chess_variant_sandbox.metrics.mutants import MUTANTS

        committed = {name for path, name in _corpus_names().items()
                     if path.startswith("rulesets/")}
        assert EXCLUDED_FROM_BAND - committed == set()
        assert {mutant.base for mutant in MUTANTS} - committed == set()


class TestPromoting:
    """The top rung, which is a claim about the corpus rather than about the game."""

    def test_an_untested_game_is_refused_and_told_what_to_run(self, world) -> None:
        with pytest.raises(corpus.CorpusError, match="chess-variant-sandbox test"):
            corpus.promote(_find(world), reason="because", ruleset_dir=world["rulesets"],
                           runs_dir=world["runs"])

    def test_a_game_the_gates_disqualify_is_refused(self, world) -> None:
        """Not a mediocre corpus entry: a game a searching player cannot win at is not a
        corpus entry at any score."""
        _measured(world, viable=False)
        with pytest.raises(corpus.CorpusError, match="not viable"):
            corpus.promote(_find(world), reason="because", ruleset_dir=world["rulesets"],
                           runs_dir=world["runs"])

    def test_no_reason_is_refused_even_when_everything_else_passes(self, world) -> None:
        """`rulesets/` is a set of games that each pin down something no other one does,
        and that sentence is the only part of this a command cannot supply."""
        _measured(world)
        with pytest.raises(corpus.CorpusError, match="needs a reason"):
            corpus.promote(_find(world), reason="   ", ruleset_dir=world["rulesets"],
                           runs_dir=world["runs"])

    def test_the_reason_and_the_measurement_are_written_into_provenance(self, world) -> None:
        _measured(world, index=0.812)
        path = corpus.promote(
            _find(world), reason="the only game here won by arrival on a corner",
            ruleset_dir=world["rulesets"], runs_dir=world["runs"],
        )
        history = json.loads(path.read_text(encoding="utf-8"))["provenance"]["history"]
        assert "won by arrival on a corner." in history
        assert "0.812" in history and "corpus roster" in history

    def test_promotion_from_the_archive_keeps_what_the_archive_wrote(self, world) -> None:
        """A game archived and then promoted still came out of a design run, and the
        second write no longer has the lineage to say so."""
        corpus.archive_game(_find(world), note="worth keeping",
                            archive_dir=world["archive"], runs_dir=world["runs"])
        _measured(world)
        path = corpus.promote(_find(world), reason="it pins something down",
                              ruleset_dir=world["rulesets"], runs_dir=world["runs"])
        history = json.loads(path.read_text(encoding="utf-8"))["provenance"]["history"]
        assert "Found by the design loop" in history
        assert "worth keeping." in history
        # ...and said once, not twice.
        assert history.count("Found by") == 1

    def test_a_game_is_on_one_rung_at_a_time(self, world) -> None:
        archived = corpus.archive_game(_find(world), archive_dir=world["archive"],
                                       runs_dir=world["runs"])
        _measured(world)
        corpus.promote(_find(world), reason="it pins something down",
                       ruleset_dir=world["rulesets"], runs_dir=world["runs"])
        assert not archived.exists(), "a promoted game must not also sit in the archive"

    def test_the_corpus_will_not_take_the_same_game_twice(self, world) -> None:
        _measured(world)
        corpus.promote(_find(world), reason="it pins something down",
                       ruleset_dir=world["rulesets"], runs_dir=world["runs"])
        with pytest.raises(corpus.CorpusError, match="already in"):
            corpus.promote(_find(world), reason="again", ruleset_dir=world["rulesets"],
                           runs_dir=world["runs"])

    def test_the_blocker_an_interface_shows_is_the_one_promote_enforces(self, world) -> None:
        """Written once and read twice, so a page cannot offer what a command refuses.

        The reason is deliberately not part of it: that is a property of the moment of
        asking rather than of the game, and a button greyed out until a box is filled
        explains less than the box does.
        """
        candidate = _find(world)
        assert "chess-variant-sandbox test" in (
            corpus.promotion_blocker(candidate, world["runs"], world["rulesets"]) or ""
        )
        _measured(world)
        assert corpus.promotion_blocker(candidate, world["runs"], world["rulesets"]) is None


@dataclass
class _Scored:
    """What `_store` reads off an `IndexResult`, and nothing else.

    The bootstrap is the whole cost of a real score and is downstream of the question
    these tests ask, which is *which passes ran*. The stub records what it was handed so
    a test can say what the scorer read.
    """

    analysis: Any
    weights_version: str = DEFAULT_WEIGHTS
    headline: float = 0.5
    viable: bool = True
    violations: tuple[str, ...] = ()
    families: dict[str, float] = field(default_factory=dict)
    scored: bool = True
    unreadable: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    interval: dict[str, Any] | None = None
    # The seat splits travel with a run, because the pooled seat reading adds them up
    # across the runs rather than averaging them. Empty here: these tests ask which
    # passes ran, and a stub that measured no seat is the honest shape for that.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def sd(self) -> float:
        return 0.0

    def save(self, path: Path) -> None:
        Path(path).write_text("{}", encoding="utf-8")

    def summary(self) -> str:
        return "scored"


class TestRepairingAStoredMeasurement:
    """`refresh` does the work the basis says is missing, and no other.

    Three repairs, and the middle one is the reason this class exists. Until the pass had
    a fingerprint of its own, an edit to `tournament/analysis.py` read as "the metrics
    changed", the repair was a re-score, and a re-score *reused* the stored
    `analysis.json` -- so the measurement was stamped current with the pass never having
    run. `docs/archive/bugs-resolved.md` (`v5`) has the demonstration on a copy of a real run. These tests
    pin the two directions: a stale pass is re-run before scoring, and a stale scorer or
    weights version alone does not pay for the pass.

    The games are real -- a four-game tournament on the five-square board, so the pass
    has positions to walk -- and the stored `analysis.json` is a sentinel the pass could
    never have written: an empty one. Whether it is still empty afterwards *is* the
    answer to whether the pass ran.
    """

    @pytest.fixture(autouse=True)
    def one_run(self, monkeypatch):
        """These tests ask which repair a stale fingerprint buys, not how many seeds.

        A committed game owes `LIBRARY_SEEDS`, so a measurement stamped on seed 0 would
        owe three replays as well, and the repair under test would be lost among them.
        `TestSeedBySeed` is where the seeds are the subject.
        """
        monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.ONE_SEED)

    @pytest.fixture
    def played(self, world):
        """A run directory with real games in it and an empty analysis pass."""
        from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

        out = corpus.run_dir(world["hash"], corpus.CANONICAL_SEED, world["runs"])
        run_tournament(
            SPEC, TournamentConfig(agents=("random", "greedy"), games_per_pairing=1,
                                   seed=0, workers=1),
        ).save(out)
        CorpusAnalysis().save(out / "analysis.json")
        return out

    @pytest.fixture
    def scorer(self, monkeypatch) -> list[_Scored]:
        calls: list[_Scored] = []

        def score(spec, records, ratings, weights, analysis=None, **_ignored):
            calls.append(_Scored(analysis=analysis, weights_version=weights.version))
            return calls[-1]

        monkeypatch.setattr(corpus, "score_ruleset", score)
        return calls

    @staticmethod
    def _stamp(world, basis: dict[str, Any]) -> None:
        """The game's measurement, in `runs/<hash>/`, beside the run that holds the games."""
        out = corpus.measurement_dir(world["hash"], world["runs"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "canonical.json").write_text(json.dumps({
            "hash": world["hash"], "name": "lancers",
            "agents": list(corpus.CANONICAL_AGENTS),
            "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": [0], "games": 4,
            "index": 0.5, "totals_used": 1, "sd": 0.0, "viable": True, "violations": [],
            "weights_version": basis["weights_version"], "families": {}, "standings": [],
            "runs": [{"seed": 0, "index": 0.5, "sd": 0.0, "viable": True,
                      "violations": [], "unreadable": [], "origin": "played"}],
            "basis": basis,
        }), encoding="utf-8")

    @staticmethod
    def _files(out: Path) -> dict[str, bytes]:
        return {name: (out / name).read_bytes() for name in ("games.jsonl", "analysis.json")}

    def test_a_stale_pass_is_re_run_before_the_score_is_taken(
        self, world, played, scorer
    ) -> None:
        # What an edit to `analysis.py` does to a stamped basis: both fingerprints move,
        # because the file is in both sets.
        basis = corpus.current_basis(world["hash"], spec=SPEC).as_dict()
        self._stamp(world, {**basis, "analysis": "moved", "metrics": "moved"})
        before = self._files(played)
        lines: list[str] = []

        stored = corpus.refresh(_find(world), runs_dir=world["runs"], workers=1,
                                report=lines.append)

        assert stored.current
        # The games were not touched, the analysis was rewritten, and it is no longer
        # the sentinel: the pass walked the games.
        after = self._files(played)
        assert after["games.jsonl"] == before["games.jsonl"]
        assert after["analysis.json"] != before["analysis.json"]
        assert json.loads(after["analysis.json"])["games"]
        # The scorer read the pass that was just run, not the file that was there.
        assert len(scorer) == 1
        assert scorer[0].analysis.games
        assert any("re-analysing" in line for line in lines)

    def test_a_stale_weights_version_is_repaired_by_a_re_score_alone(
        self, world, played, scorer
    ) -> None:
        basis = corpus.current_basis(world["hash"], spec=SPEC).as_dict()
        self._stamp(world, {**basis, "weights_version": "v0"})
        before = self._files(played)

        stored = corpus.refresh(_find(world), runs_dir=world["runs"], workers=1)

        assert stored.current
        assert stored.stored["weights_version"] == DEFAULT_WEIGHTS
        # Nothing but the score moved: the sentinel is still the sentinel, and the
        # scorer was handed it.
        assert self._files(played) == before
        assert len(scorer) == 1
        assert not scorer[0].analysis.games

    def test_a_stale_scorer_is_repaired_by_a_re_score_alone(
        self, world, played, scorer
    ) -> None:
        """`metrics/` moved and the pass did not: the cheap repair stays reachable."""
        basis = corpus.current_basis(world["hash"], spec=SPEC).as_dict()
        self._stamp(world, {**basis, "metrics": "moved"})
        before = self._files(played)

        stored = corpus.refresh(_find(world), runs_dir=world["runs"], workers=1)

        assert stored.current
        assert self._files(played) == before
        assert len(scorer) == 1

    def test_a_basis_that_never_recorded_the_pass_re_runs_it_when_metrics_moved(
        self, world, played, scorer
    ) -> None:
        """Every measurement stamped before the pass had a fingerprint, on the day after.

        Its `metrics` covered the pass, so a moved `metrics` cannot rule the pass out,
        and the stronger repair is taken rather than the file trusted.
        """
        basis = corpus.current_basis(world["hash"], spec=SPEC).as_dict()
        del basis["analysis"]
        self._stamp(world, {**basis, "metrics": "moved"})

        stored = corpus.refresh(_find(world), runs_dir=world["runs"], workers=1)

        assert stored.current
        assert json.loads((played / "analysis.json").read_bytes())["games"]
        # And the repaired measurement records the pass, so next time it can tell.
        assert stored.basis is not None and stored.basis.analysis

    def test_a_run_the_scorer_refuses_is_stored_as_a_sentence_and_not_a_number(
        self, world, played, monkeypatch
    ) -> None:
        """One seed, and the run has no total. `hierarchy chess` is this case on disk.

        Under `v10` a run that drew no blunder trials raises. The measurement is still
        written -- the game was played and the ladder stands -- with `index: null`, the
        scorer's own sentence, and a verdict of refuse. A 0.0 published here would be a
        number nothing measured, at the bottom of every leaderboard.
        """
        from chess_variant_sandbox.metrics.scoring import WeightsError

        def refuse(*_args, **_kwargs):
            raise WeightsError("weights v10 expect components not present: blunder")

        monkeypatch.setattr(corpus, "score_ruleset", refuse)
        basis = corpus.current_basis(world["hash"], spec=SPEC).as_dict()
        self._stamp(world, {**basis, "metrics": "moved"})

        stored = corpus.refresh(_find(world), runs_dir=world["runs"], workers=1)

        assert stored.current, "a run with no total is not a stale measurement"
        assert stored.stored["index"] is None
        assert stored.stored["sd"] is None
        assert stored.stored["totals_used"] == 0
        assert stored.stored["viable"] is False
        assert "blunder" in stored.stored["runs"][0]["violations"][0]
        assert stored.stored["runs"][0]["index"] is None
        # The run's own index file is not written either: there was nothing to write.
        assert not list(played.glob("index-*.json"))

    def test_a_current_measurement_runs_nothing(self, world, played, scorer) -> None:
        basis = corpus.current_basis(world["hash"], spec=SPEC).as_dict()
        self._stamp(world, basis)
        before = self._files(played)

        stored = corpus.refresh(_find(world), runs_dir=world["runs"], workers=1)

        assert stored.current
        assert self._files(played) == before
        assert not scorer


class TestSeedBySeed:
    """A game with more than one run, and what each run costs to bring up to date.

    The unit of work is a run rather than a measurement. A library game whose file names
    one seed and whose reader expects four owes three tournaments and nothing else; one
    whose scorer moved owes a re-score of every run it has and no tournament at all. The
    fit is a property of the spec, so it is paid once whatever the seeds cost -- and
    paying it per run would nearly double a four-run measurement for four identical
    answers.

    Nothing here plays. The three expensive calls are counted instead, because what these
    tests are about is *how many* of each run, and a real tournament would say the same
    thing in four minutes.
    """

    @pytest.fixture
    def counted(self, world, monkeypatch) -> dict[str, list[Any]]:
        """`measure`'s three expensive calls, replaced by counters.

        The stand-in tournament writes the two files a run directory is recognised by,
        with the canonical roster in them, so the seed it wrote can be read back and the
        write rule sees a canonical run rather than somebody's one-off.
        """
        calls: dict[str, list[Any]] = {"fit": [], "play": [], "analyse": [], "score": []}

        class _Fit:
            accepted = False
            accept_score = 0.0

        class _Checked:
            """`check_fit`'s answer for a store made on the loaded core: nothing played."""

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
                Path(out).mkdir(parents=True, exist_ok=True)
                (Path(out) / "games.jsonl").write_text("", encoding="utf-8")
                (Path(out) / "tournament.json").write_text(json.dumps({
                    "config": {
                        "agents": list(corpus.CANONICAL_AGENTS),
                        "games_per_pairing": corpus.CANONICAL_GAMES,
                        "seed": self.seed,
                    },
                    "games": 0, "ratings": {"standings": []},
                }), encoding="utf-8")

        def fit(spec, workers=0, **_ignored):
            calls["fit"].append(spec)
            return _Checked()

        def play(spec, config, progress=False):
            calls["play"].append(config.seed)
            return _Tournament(seed=config.seed)

        def analyse(spec, records, seed=0, workers=0, **_ignored):
            calls["analyse"].append(seed)
            from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

            return CorpusAnalysis()

        def score(spec, records, ratings, weights, analysis=None, **_ignored):
            calls["score"].append(_Scored(analysis=analysis, weights_version=weights.version))
            return calls["score"][-1]

        monkeypatch.setattr(corpus, "check_fit", fit)
        monkeypatch.setattr(corpus, "run_tournament", play)
        monkeypatch.setattr(corpus, "analyse_corpus", analyse)
        monkeypatch.setattr(corpus, "score_ruleset", score)
        return calls

    @pytest.fixture
    def library_game(self, world, monkeypatch) -> corpus.Candidate:
        """The game, archived, in a checkout whose library expects four runs."""
        corpus.archive_game(
            _find(world), archive_dir=world["archive"], runs_dir=world["runs"]
        )
        monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.LIBRARY_SEEDS)
        return _find(world)

    @staticmethod
    def _run_on_disk(world, seed: int) -> Path:
        """A run directory with a game in it, a canonical tournament and no index file."""
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
        from chess_variant_sandbox.tournament.record import GameRecord, write_records

        out = corpus.run_dir(world["hash"], seed, world["runs"])
        out.mkdir(parents=True, exist_ok=True)
        write_records([GameRecord(
            ruleset_hash=world["hash"], ruleset_name="lancers", seed=seed,
            agents=("random", "greedy"), moves=[], winner=1,
            reason="royal_capture", plies=6,
        )], out / "games.jsonl")
        (out / "tournament.json").write_text(json.dumps({
            "config": {
                "agents": list(corpus.CANONICAL_AGENTS),
                "games_per_pairing": corpus.CANONICAL_GAMES,
                "seed": seed,
            },
            "games": 4,
            "ratings": {"standings": [{"name": "greedy", "rating": 12.0, "games": 4}]},
        }), encoding="utf-8")
        CorpusAnalysis().save(out / "analysis.json")
        return out

    def test_a_one_run_file_in_a_four_run_library_owes_three_tournaments(
        self, world, library_game
    ) -> None:
        """The reason printed says how much of the work is missing runs.

        `replay 3 of 4 runs` is a different afternoon from `replay`, and the reader that
        shows a repair button has no other way to tell them apart.
        """
        _measured(world, seeds=(0,))

        stored = corpus.read_measurement(
            world["hash"], world["runs"], spec=SPEC, seeds=corpus.LIBRARY_SEEDS
        )
        assert stored is not None and not stored.current
        assert stored.staleness.missing_seeds == (1000, 2000, 3000)
        assert stored.staleness.stored_needs == ""
        summary = stored.staleness.summary()
        assert "replay 3 of 4 runs" in summary
        assert "not yet measured on seeds 1000, 2000 and 3000" in summary

    def test_only_the_missing_seeds_are_played_and_the_fit_is_paid_once(
        self, world, library_game, counted
    ) -> None:
        _measured(world, seeds=(0,))
        self._run_on_disk(world, corpus.CANONICAL_SEED)

        stored = corpus.refresh(library_game, runs_dir=world["runs"], workers=1)

        # Three tournaments, at the three seeds the file did not name, and one fit for
        # all of them.
        assert counted["play"] == [1000, 2000, 3000]
        assert len(counted["fit"]) == 1
        # The analysis pass runs at seed 0 for every run. A trial is fixed by the pass's
        # seed and the record it is drawn from, and the records differ by run, so the
        # trials already differ without the pass's own seed moving.
        assert counted["analyse"] == [0, 0, 0]
        assert stored.current
        assert stored.stored["seeds"] == list(corpus.LIBRARY_SEEDS)
        assert [run["seed"] for run in stored.stored["runs"]] == list(corpus.LIBRARY_SEEDS)

    def test_a_run_that_is_still_good_is_read_rather_than_scored_again(
        self, world, library_game, counted
    ) -> None:
        """The cheapest repair there is: a file read instead of a bootstrap.

        A run whose games and score are both current contributes its stored total, its
        components and its ladder as they stand. What makes that safe is the check: the
        game's file records what each run scored, so an index file that disagrees is a
        file written by a different pass and the run is scored again.
        """
        _measured(world, index=0.61, seeds=(0,))
        out = self._run_on_disk(world, corpus.CANONICAL_SEED)
        (out / f"index-{DEFAULT_WEIGHTS}.json").write_text(json.dumps({
            "total": 0.61, "viable": True, "violations": [],
            "scores": {"depth": 0.7}, "raw": {"depth": 3.0},
            "contributions": {"depth": 0.42}, "families": {"depth": 0.7},
            "unreadable": [], "combined": None,
            "interval": {"mean": 0.61, "sd": 0.02, "ci_low": 0.58, "ci_high": 0.64,
                         "deterministic_pairings": 0.0, "stratum_correction": 1.0,
                         "resamples": 1024},
        }), encoding="utf-8")

        stored = corpus.refresh(library_game, runs_dir=world["runs"], workers=1)

        # Three scores, not four: seed 0 was read off its own index file.
        assert len(counted["score"]) == 3
        first = stored.stored["runs"][0]
        assert first["seed"] == 0 and first["index"] == 0.61
        assert stored.standings and stored.standings[0]["name"] == "greedy"

    def test_an_index_file_that_disagrees_with_the_file_that_named_it_is_scored_again(
        self, world, library_game, counted
    ) -> None:
        _measured(world, index=0.61, seeds=(0,))
        out = self._run_on_disk(world, corpus.CANONICAL_SEED)
        (out / f"index-{DEFAULT_WEIGHTS}.json").write_text(json.dumps({
            "total": 0.49, "viable": True, "violations": [], "scores": {},
            "raw": {}, "contributions": {"depth": 0.42}, "families": {},
            "unreadable": [], "combined": None, "interval": None,
        }), encoding="utf-8")

        corpus.refresh(library_game, runs_dir=world["runs"], workers=1)

        assert len(counted["score"]) == 4, "the two numbers disagreed, so neither stood"

    def test_a_moved_scorer_re_scores_every_run_and_plays_none(
        self, world, library_game, counted
    ) -> None:
        """The other half of the split: what the runs the file *does* name owe."""
        _measured(world, seeds=corpus.LIBRARY_SEEDS)
        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["basis"] = {**payload["basis"], "metrics": "moved"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        for seed in corpus.LIBRARY_SEEDS:
            self._run_on_disk(world, seed)

        stored = corpus.refresh(library_game, runs_dir=world["runs"], workers=1)

        assert not counted["play"] and not counted["fit"]
        assert len(counted["score"]) == 4
        # Every score carries its repeats (Q953a), read off the runs' records on disk: the
        # four runs' one game each is one game played four times, and no game is in the
        # searching band or the top two.
        repeats = stored.stored["repeats"]
        assert (repeats["games"], repeats["all"]) == (4, 0.25)
        assert repeats["searching"] is None and repeats["top_two"] is None
        assert [run["seed"] for run in repeats["runs"]] == list(corpus.LIBRARY_SEEDS)

    def test_a_candidate_is_current_on_one_run_in_a_four_run_library(
        self, world, monkeypatch
    ) -> None:
        """A number nobody has committed to does not owe four tournaments.

        `expected_seeds` reads the rung, not the constant, so turning the library over to
        four runs leaves every visitor's design and every untested candidate current.
        """
        monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.LIBRARY_SEEDS)
        _measured(world, seeds=(0,))

        candidate = _find(world)
        assert candidate.stage == "candidate"
        measurement = candidate.measurement(world["runs"])
        assert measurement is not None and measurement.current
        assert measurement.stored["seeds"] == [0]

    def test_a_run_that_cannot_be_adopted_is_played_as_it_would_have_been(
        self, world, library_game, counted, tmp_path
    ) -> None:
        """Adoption's only fallback, and the reason it needs no other.

        A campaign run this checkout will not vouch for costs the tournament that was
        going to be played anyway, so every way adoption can fail has one consequence.
        """
        _measured(world, seeds=(0,))
        self._run_on_disk(world, corpus.CANONICAL_SEED)
        elsewhere = tmp_path / "campaign" / f"{world['hash']}-2000"
        elsewhere.mkdir(parents=True)
        (elsewhere / "games.jsonl").write_text("", encoding="utf-8")
        (elsewhere / "tournament.json").write_text(json.dumps({
            "config": {"agents": ["random", "greedy"], "games_per_pairing": 1,
                       "seed": 2000},
        }), encoding="utf-8")

        lines: list[str] = []
        corpus.refresh(
            library_game, runs_dir=world["runs"], workers=1,
            adopt_from=tmp_path / "campaign", report=lines.append,
        )

        assert counted["play"] == [1000, 2000, 3000]
        assert any("a different roster" in line for line in lines)
        assert any("holds no run to adopt" in line for line in lines)

    def test_an_adopted_run_is_re_scored_and_says_where_it_came_from(
        self, world, library_game, counted, monkeypatch, tmp_path
    ) -> None:
        """What adoption saves is the tournament, not the score.

        `metrics` moved after the campaign scored these runs, so an adopted run is scored
        here before it joins the mean -- and `runs[].origin` keeps the only claim in the
        file that is not the game's own.
        """
        from chess_variant_sandbox.tournament import adopt

        _measured(world, seeds=(0,))
        self._run_on_disk(world, corpus.CANONICAL_SEED)

        def take(spec, seed, source, target, config, workers=0, say=None):
            self._run_on_disk(world, seed)
            return adopt.Adoption(
                seed=seed, adopted=True, source=source,
                origin="adopted from campaign, 16 games and 6 trials re-derived",
            )

        monkeypatch.setattr(adopt, "campaign_run", lambda root, digest, seed: tmp_path)
        monkeypatch.setattr(adopt, "adopt_run", take)

        stored = corpus.refresh(
            library_game, runs_dir=world["runs"], workers=1, adopt_from=tmp_path
        )

        assert not counted["play"], "three tournaments were adopted instead of played"
        assert len(counted["score"]) == 4
        origins = [run["origin"] for run in stored.stored["runs"]]
        assert origins[0] == "played"
        assert all("adopted from campaign" in origin for origin in origins[1:])

    def test_a_replay_refuses_to_write_over_somebody_else_s_tournament(
        self, world, library_game, counted
    ) -> None:
        """`chess-variant-sandbox tournament` writes to `runs/<hash>-<seed>` only with an explicit `--out`,
        or for directories made before 2026-09-24; either way it may hold whatever roster it
        was asked for. A measurement that replayed over one would throw it away and stamp
        what replaced it as canonical."""
        _measured(world, seeds=(0,))
        self._run_on_disk(world, corpus.CANONICAL_SEED)
        foreign = corpus.run_dir(world["hash"], 2000, world["runs"])
        foreign.mkdir(parents=True, exist_ok=True)
        (foreign / "games.jsonl").write_text("", encoding="utf-8")
        (foreign / "tournament.json").write_text(json.dumps({
            "config": {"agents": ["random", "greedy"], "games_per_pairing": 1,
                       "seed": 2000},
        }), encoding="utf-8")

        with pytest.raises(corpus.CorpusError, match="another roster or seed"):
            corpus.refresh(library_game, runs_dir=world["runs"], workers=1)

    # ------------------------------------------------------------ the refit (Q80d)
    #
    # `docs/reference/refit-repair.md`: a fit neither made nor reproduced on the loaded
    # core is re-derived first, and the staleness read again. The real `check_fit` runs
    # here, with `fit_ruleset` answering from a canned fit, because what is under test is
    # what `refresh` does with the check's answer, not the 48-game match.

    FOREIGN = "0000000000000000"

    @staticmethod
    def _fit(digest: str, build: str, **changes: Any):
        from chess_variant_sandbox.agents.features import VOCABULARY
        from chess_variant_sandbox.agents.fitted import FittedObjective, procedure_digest, unit

        weights = [0.0] * len(VOCABULARY)
        weights[VOCABULARY.index("material")] = 1.0
        weights[VOCABULARY.index("centrality")] = 0.3
        fields: dict[str, Any] = {
            "ruleset_hash": digest, "procedure": procedure_digest(),
            "vector": unit(tuple(weights)), "accepted": True, "accept_score": 0.625,
            "accept_games": 48, "samples": 500, "loss": 0.5, "build": build,
        }
        return FittedObjective(**{**fields, **changes})

    @pytest.fixture
    def refit_world(self, world, library_game, counted, monkeypatch, tmp_path):
        """A library game measured before `Basis.fit`, on a fit made on another core.

        The stamp is today's in every field but `fit`, which it does not record, as every
        measurement in `runs/` did on the day Q80c landed.
        """
        from chess_variant_sandbox.agents.fitted import STORE_ENV, store_path

        store = tmp_path / "fits"
        monkeypatch.setenv(STORE_ENV, str(store))
        old = self._fit(world["hash"], self.FOREIGN)
        old.save(store_path(world["hash"], old.procedure, store))
        path = self._measured_silently(world)
        # The real check, over a canned procedure.
        monkeypatch.setattr(corpus, "check_fit", fitting.check_fit)
        return {"store": store, "old": old, "path": path}

    @staticmethod
    def _measured_silently(world) -> Path:
        """`_measured`, with the stamp's `fit` taken out as a pre-Q80c stamp has none."""
        _measured(world, seeds=corpus.LIBRARY_SEEDS)
        path = corpus.measurement_dir(world["hash"], world["runs"]) / "canonical.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["basis"].pop("fit", None)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _answer(self, monkeypatch, fresh) -> list[int]:
        calls: list[int] = []

        def fit_ruleset(spec, workers=0, **_ignored):
            calls.append(1)
            return fresh

        monkeypatch.setattr(fitting, "fit_ruleset", fit_ruleset)
        return calls

    def test_a_foreign_fit_reads_stale_refit_and_nothing_else(
        self, world, refit_world
    ) -> None:
        stored = corpus.read_measurement(
            world["hash"], world["runs"], spec=SPEC, seeds=corpus.LIBRARY_SEEDS
        )
        assert stored is not None and not stored.current
        assert stored.staleness.needs == "refit" and stored.staleness.refit
        assert stored.staleness.summary() == (
            f"stale (refit): its fit was made on core {self.FOREIGN} and has not been "
            "re-derived on this one"
        )

    def test_a_fit_that_reproduces_ends_current_with_no_tournament_played(
        self, world, library_game, counted, refit_world, monkeypatch
    ) -> None:
        from chess_variant_sandbox.measurement import built_fingerprint

        fitter = self._answer(
            monkeypatch,
            self._fit(world["hash"], built_fingerprint(), accept_score=0.6042),
        )
        before = refit_world["path"].read_bytes()
        said: list[str] = []

        stored = corpus.refresh(
            library_game, runs_dir=world["runs"], workers=1, report=said.append
        )

        assert stored.current
        assert len(fitter) == 1
        assert not counted["play"] and not counted["analyse"] and not counted["score"]
        # Nothing is rewritten: the store's sidecar is what now vouches for the stamp.
        assert refit_world["path"].read_bytes() == before
        assert fitting.reproduced_on(refit_world["old"], refit_world["store"]) == (
            built_fingerprint(),
        )
        assert any("reproduces on core" in line for line in said), said

    def test_a_refused_vector_that_drifts_plays_nothing(
        self, world, library_game, counted, monkeypatch, tmp_path
    ) -> None:
        """The player is the built-in objective before and after, so nothing moved."""
        from chess_variant_sandbox.agents.fitted import STORE_ENV, store_path
        from chess_variant_sandbox.measurement import built_fingerprint

        store = tmp_path / "fits"
        monkeypatch.setenv(STORE_ENV, str(store))
        old = self._fit(world["hash"], self.FOREIGN, accepted=False, accept_score=0.3)
        old.save(store_path(world["hash"], old.procedure, store))
        self._measured_silently(world)
        monkeypatch.setattr(corpus, "check_fit", fitting.check_fit)
        drifted = self._fit(
            world["hash"], built_fingerprint(), accepted=False, accept_score=0.35,
            samples=512,
        )
        self._answer(monkeypatch, drifted)

        stored = corpus.refresh(library_game, runs_dir=world["runs"], workers=1)

        assert stored.current
        assert not counted["play"] and not counted["score"]
        assert fitting.load_fit(world["hash"], old.procedure, store) == drifted

    @pytest.mark.parametrize("stamp", ["silent", "recorded"])
    def test_an_accepted_vector_that_changes_replays_exactly_its_own_seeds(
        self, world, library_game, counted, refit_world, monkeypatch, stamp
    ) -> None:
        """Both readings of the stamp reach the replay: one that records no fit through
        the check's own answer, and one that records the old player through the re-read."""
        from chess_variant_sandbox.measurement import built_fingerprint

        if stamp == "recorded":
            payload = json.loads(refit_world["path"].read_text(encoding="utf-8"))
            payload["basis"]["fit"] = corpus.player_digest(refit_world["old"])
            refit_world["path"].write_text(json.dumps(payload), encoding="utf-8")
        moved = self._fit(
            world["hash"], built_fingerprint(),
            vector=tuple(reversed(refit_world["old"].vector)),
        )
        fitter = self._answer(monkeypatch, moved)

        stored = corpus.refresh(library_game, runs_dir=world["runs"], workers=1)

        assert counted["play"] == list(corpus.LIBRARY_SEEDS)
        assert len(fitter) == 1, "the refit is the fit the replay plays; it is not paid twice"
        assert stored.current
        assert stored.basis is not None
        assert stored.basis.fit == corpus.player_digest(moved)


class TestTheCommandLineAsksForRuns:
    """`measure --seeds`, `measure --adopt`, and what `test` means now.

    Two words rather than a list of integers. Which seeds a run count *is* belongs to
    `corpus`, and a person choosing between one run and four is choosing between a
    visitor's score and a library one, not between arbitrary numbers.
    """

    @staticmethod
    def _parse(argv: list[str]):
        from chess_variant_sandbox.cli import _build_parser

        return _build_parser().parse_args(argv)

    def test_the_two_words_are_the_seed_sets_corpus_names(self) -> None:
        from chess_variant_sandbox.cli import _seed_set

        assert _seed_set(self._parse(["measure", "--seeds", "one"])) == corpus.ONE_SEED
        assert _seed_set(self._parse(["measure", "--seeds", "four"])) == corpus.LIBRARY_SEEDS
        # Absent means whatever the game's own rung expects, which is `expected_seeds`.
        assert _seed_set(self._parse(["measure"])) is None

    def test_a_seed_set_the_project_does_not_use_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            self._parse(["measure", "--seeds", "three"])

    def test_test_is_the_four_run_rung_before_promote(self) -> None:
        """`chess-variant-sandbox test` has always meant "play it on the corpus roster whatever is on
        disk". A committed score is now the mean of four runs, so that is four."""
        from chess_variant_sandbox.cli import _seed_set

        args = self._parse(["test", "abc123def456"])
        assert _seed_set(args) == corpus.LIBRARY_SEEDS
        assert args.adopt is None

    def test_adopt_names_a_campaign_root(self) -> None:
        args = self._parse(["measure", "--adopt", "runs/seat-fresh-analysed-2026-09"])
        assert args.adopt == Path("runs/seat-fresh-analysed-2026-09")
        assert self._parse(["measure"]).adopt is None


class TestPromotionAsksForTheWholeMeasurement:
    """Four of 71 library games change their verdict with the seed.

    Promotion is a claim about the corpus rather than a reading of one game, so the rung
    before it is the full four-run measurement -- whatever the library is currently
    expected to have, which is the one run S4 has not yet replaced.
    """

    def test_a_one_run_measurement_is_refused_and_named(self, world) -> None:
        _measured(world, seeds=(0,))
        blocker = corpus.promotion_blocker(_find(world), world["runs"], world["rulesets"])

        assert blocker is not None
        assert "not yet measured on seeds 1000, 2000 and 3000" in blocker
        # And the command that fixes it, which is the four-run rung and not `measure`.
        assert f"chess-variant-sandbox test {world['hash']}" in blocker

        with pytest.raises(corpus.CorpusError, match="out of date"):
            corpus.promote(_find(world), reason="it pins something down",
                           ruleset_dir=world["rulesets"], runs_dir=world["runs"])

    def test_four_runs_pass(self, world) -> None:
        _measured(world, seeds=corpus.LIBRARY_SEEDS)
        assert corpus.promotion_blocker(
            _find(world), world["runs"], world["rulesets"]
        ) is None


@pytest.mark.slow
def test_measuring_plays_the_corpus_roster_and_says_so(world, monkeypatch) -> None:
    """The expensive rung, run once for real on a five-square board.

    What it has to leave behind is a number that can sit in a column beside `chess`:
    played on `CANONICAL_AGENTS`, analysed, scored, and stamped with the roster it was
    taken on so nothing can later mistake it for a design run's number.

    One seed, because what is under test is the shape a run leaves behind and the
    library's four would play the same tournament four times over to say it.
    `TestSeedBySeed` is where the four are the subject.
    """
    monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.ONE_SEED)
    stored = corpus.measure(_find(world), runs_dir=world["runs"], workers=1)

    assert stored.stored["agents"] == list(corpus.CANONICAL_AGENTS)
    assert stored.stored["games"] > 0
    assert 0.0 <= stored.stored["index"] <= 1.0
    # Fresh, by construction: a measurement taken now carries the basis that holds now.
    assert stored.current

    # The ladder travels in the same file as the score. Two separate scans of `runs/`
    # used to supply them and could land in different directories, which is how a fresh
    # Elo table came to be rendered beside a year-old verdict as though it were one row.
    assert stored.standings

    # Two directories, and the split is what lets a game have more than one run. The run
    # holds the games and its own score; the measurement holds the file every reader
    # trusts and the mean's own index file, and no games at all.
    run = corpus.run_dir(world["hash"], corpus.CANONICAL_SEED, world["runs"])
    for name in ("games.jsonl", "tournament.json", "analysis.json"):
        assert (run / name).exists(), name
    # Written where `chess-variant-sandbox score` already looks, so nothing had to learn a new place.
    assert list(run.glob("index-*.json"))
    assert not (run / "canonical.json").exists()

    out = corpus.measurement_dir(world["hash"], world["runs"])
    assert (out / "canonical.json").exists()
    assert list(out.glob("index-*.json"))
    assert not (out / "games.jsonl").exists()
    assert stored.stored["seeds"] == [corpus.CANONICAL_SEED]
    assert [run["seed"] for run in stored.stored["runs"]] == [corpus.CANONICAL_SEED]
    read_back = corpus.read_measurement(world["hash"], world["runs"])
    assert read_back is not None
    assert read_back.current
    assert read_back.stored["index"] == stored.stored["index"]


class TestAPooledRescoreIsTheSerialOne:
    """R7 of `docs/reference/perf-survey-2026-09.md`: `measure` re-scores games side by side.

    Threads never change an answer, and neither may processes. Two games on the
    five-square board, one owing a re-score and one a re-analysis, with real games (a
    searching rung among them, so the blunder trials have something to grade and the
    runs score), a real pass and the real scorer, repaired once serially (`--workers 1`) and once through the
    pool from an identical copy. Every file either run leaves behind, and every line it
    prints, has to be the same.
    """

    @staticmethod
    def _variant() -> dict[str, Any]:
        other = copy.deepcopy(SPEC)
        other["name"] = "longer lancers"
        other["rules"]["no_progress_draw"] = 30
        return other

    @pytest.fixture
    def library(self, tmp_path: Path, monkeypatch) -> dict[str, Any]:
        from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
        from chess_variant_sandbox.tournament.analysis import analyse_corpus

        runs = tmp_path / "serial"
        archive = Archive(runs / "design")
        digests = []
        for spec, moved in ((SPEC, ("metrics",)), (self._variant(), ("analysis", "metrics"))):
            digest = spec_hash(spec)
            digests.append(digest)
            archive.record(
                Entry(hash=digest, name=spec["name"], generation=1, parent=None,
                      status="evaluated", index=0.5, viable=True),
                spec=spec,
            )
            out = corpus.run_dir(digest, corpus.CANONICAL_SEED, runs)
            # `threats=0`, the search before Q502i: at two games a pairing, the search
            # after it drew a skill gradient of zero here, and a non-viable run has no
            # index for the pooled and serial paths to agree on.
            played = run_tournament(
                spec, TournamentConfig(agents=("random", "greedy", "minimax:nodes=50,threats=0"),
                                       games_per_pairing=2, seed=0, workers=1),
            )
            played.save(out)
            analyse_corpus(spec, played.records, seed=corpus.CANONICAL_SEED,
                           workers=1).save(out / "analysis.json")
            basis = corpus.current_basis(
                digest, spec=spec, seeds=corpus.ONE_SEED
            ).as_dict()
            _stamp_at(runs, digest, spec["name"],
                      {**basis, **{kind: "moved" for kind in moved}})
        pooled = tmp_path / "pooled"
        shutil.copytree(runs, pooled)
        monkeypatch.setattr(corpus, "corpus_hashes", lambda *a, **k: list(digests))
        return {"serial": runs, "pooled": pooled, "hashes": digests}

    @staticmethod
    def _tree(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes().replace(
                str(root).encode(), b"<runs>"
            ).replace(json.dumps(str(root))[1:-1].encode(), b"<runs>")
            for path in sorted(root.rglob("*")) if path.is_file()
        }

    @staticmethod
    def _undated(tree: dict[str, bytes]) -> dict[str, bytes]:
        """The tree with each history line's `replaced` day checked and set aside (Q744).

        `replaced` is the UTC day the outgoing measurement was replaced on: the wall
        clock's, not the work's. The serial half and the pooled half run one after the
        other, and a pair that straddles UTC midnight writes two days, which is right, so
        the day is held to being a day and everything else on the line to the identity.
        """
        out = dict(tree)
        for name, body in tree.items():
            if not name.endswith("/" + HISTORY_FILE):
                continue
            lines = []
            for line in body.decode("utf-8").splitlines():
                entry = json.loads(line)
                date.fromisoformat(entry.pop("replaced"))
                lines.append(json.dumps(entry, sort_keys=True))
            assert lines, name
            out[name] = "\n".join(lines).encode("utf-8")
        return out

    # Slow (Q749): a real pool of spawned workers, 2.2 s on one core and 2.7-3.5 s in the
    # gate, most of it the workers starting.
    @pytest.mark.slow
    def test_the_pool_writes_and_says_what_the_serial_path_does(
        self, library, monkeypatch, capsys
    ) -> None:
        from chess_variant_sandbox import cli

        for digest in library["hashes"]:
            candidate = corpus.find(digest, runs_dir=library["serial"])
            assert corpus.poolable(candidate, library["serial"])

        # Which process ran each repair. A spawned worker imports its own `corpus`, so a
        # call counted here is one this process made.
        here: list[str] = []
        real = corpus.refresh

        def counted(candidate, *args, **kwargs):
            here.append(candidate.hash)
            return real(candidate, *args, **kwargs)

        monkeypatch.setattr(corpus, "refresh", counted)
        before = self._tree(library["serial"])
        assert before == self._tree(library["pooled"])

        assert cli.main(["measure", "--runs", str(library["serial"]), "--workers", "1"]) == 0
        serial_said = capsys.readouterr().out
        assert here == library["hashes"]

        here.clear()
        assert cli.main(["measure", "--runs", str(library["pooled"]), "--workers", "2"]) == 0
        pooled_said = capsys.readouterr().out
        assert here == []

        serial = self._undated(self._tree(library["serial"]))
        assert serial != before
        # Each game's replaced measurement went into its history, so the day was read.
        assert sum(name.endswith("/" + HISTORY_FILE) for name in serial) == len(
            library["hashes"]
        )
        assert self._undated(self._tree(library["pooled"])) == serial
        assert pooled_said.replace(str(library["pooled"]), "<runs>") == serial_said.replace(
            str(library["serial"]), "<runs>"
        )
        for digest in library["hashes"]:
            stored = corpus.find(digest, runs_dir=library["pooled"]).measurement(
                library["pooled"]
            )
            assert stored is not None and stored.current
            # A number, so the identity above compared scores and not two refusals.
            assert stored.stored["index"] is not None

    def test_what_owes_a_tournament_is_not_pooled(self, library) -> None:
        digest = library["hashes"][0]
        candidate = corpus.find(digest, runs_dir=library["serial"])
        run = corpus.run_dir(digest, corpus.CANONICAL_SEED, library["serial"])
        (run / "games.jsonl").unlink()
        assert not corpus.poolable(candidate, library["serial"])

        stamp = corpus.measurement_dir(digest, library["serial"]) / "canonical.json"
        stored = json.loads(stamp.read_text())
        stored["basis"]["engine"] = "moved"
        stamp.write_text(json.dumps(stored))
        assert not corpus.poolable(candidate, library["serial"])
        stamp.unlink()
        assert not corpus.poolable(candidate, library["serial"])


def _stamp_at(runs: Path, digest: str, name: str, basis: dict[str, Any]) -> None:
    """A one-run measurement in `runs/<hash>/`, stamped with `basis`."""
    out = corpus.measurement_dir(digest, runs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "canonical.json").write_text(json.dumps({
        "hash": digest, "name": name,
        "agents": list(corpus.CANONICAL_AGENTS),
        "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": [0], "games": 2,
        "index": 0.5, "totals_used": 1, "sd": 0.0, "viable": True, "violations": [],
        "weights_version": basis["weights_version"], "families": {}, "standings": [],
        "runs": [{"seed": 0, "index": 0.5, "sd": 0.0, "viable": True,
                  "violations": [], "unreadable": [], "origin": "played"}],
        "basis": basis,
    }), encoding="utf-8")
