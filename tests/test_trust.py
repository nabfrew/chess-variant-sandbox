"""How far to trust a score (Q646, `docs/plan-confidence.md` §2.3, T1).

What is held down here, on synthetic records for a five-square game (never chess):

- **Each row of §2.3's table**: every kind of evidence absent, finding more, agreeing (where
  it can), silent, stale and unreadable, at the pre-registered thresholds' both sides.
- **Each clause of the level**: a finding, current or stale, questions; a current
  agreement checks and a stale one counts for nothing; else early estimate.
- **The score's own state**: absent or out of date, no level, never a guessed one.
- **People** are read and reported and move the level (§5 Q3, answered A on 2026-10-03).
- **It moves nothing**: a pure function of the files, it writes none, `metrics/` never
  imports it, and it is in no fingerprint set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_corpus import SPEC

from chess_variant_sandbox import corpus, measurement, premise, trust
from chess_variant_sandbox.agents.nets import INPUTS_VERSION
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.spec import spec_hash

DIGEST = spec_hash(SPEC)
PROCEDURE = "1d160257b8e5"
WEIGHTS = "ab" * 32
PEOPLE_COUNT = trust.Rule(people_move_level=True)


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    return {"runs": tmp_path / "runs", "nets": tmp_path / "nets"}


def _basis() -> dict[str, Any]:
    return corpus.current_basis(DIGEST, spec=SPEC).as_dict()


def _measured(
    world: dict[str, Path],
    unreadable: tuple[tuple[str, ...], ...] = ((), (), (), ()),
    basis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A stored library score, stamped with today's basis unless told otherwise."""
    stamped = _basis() if basis is None else basis
    out = corpus.measurement_dir(DIGEST, world["runs"])
    out.mkdir(parents=True, exist_ok=True)
    seeds = corpus.LIBRARY_SEEDS
    (out / "canonical.json").write_text(json.dumps({
        "hash": DIGEST, "name": SPEC["name"], "agents": list(corpus.CANONICAL_AGENTS),
        "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": list(seeds),
        "games": 256 * len(seeds), "index": 0.7, "totals_used": 4, "sd": 0.02,
        "viable": True, "violations": [], "weights_version": DEFAULT_WEIGHTS,
        "families": {}, "standings": [],
        "runs": [{"seed": seed, "index": 0.7, "sd": 0.02, "viable": True, "violations": [],
                  "unreadable": list(withheld), "origin": "played"}
                 for seed, withheld in zip(seeds, unreadable, strict=True)],
        "basis": stamped,
    }), encoding="utf-8")
    return stamped


def _premise(
    world: dict[str, Path],
    fairy: float | None = None,
    ceiling: float = 0.35,
    stale: bool = False,
) -> None:
    """A stored strength check; ``fairy`` ``None`` where Fairy-Stockfish did not play."""
    basis = premise.PremiseBasis.current(DIGEST, SPEC).as_dict()
    if stale:
        basis["rule"] = "an older rule"
    fairy_part: dict[str, Any] = (
        {"played": False, "why": "stand-in"} if fairy is None
        else {"played": True, "score": fairy, "error": 0.03, "games": 32, "floor": 0.30,
              "passes": fairy >= 0.30}
    )
    premise.write({
        "hash": DIGEST, "name": SPEC["name"],
        "verdict": premise.PASSES,
        "ceiling": {"score": ceiling, "error": 0.02, "games": 32, "floor": 0.20,
                    "passes": ceiling >= 0.20},
        "fairy": fairy_part,
        "basis": basis,
    }, world["runs"])


def _net(
    world: dict[str, Path],
    verdict: str = "accepted",
    procedure: str = PROCEDURE,
    inputs_version: int = INPUTS_VERSION,
) -> None:
    world["nets"].mkdir(parents=True, exist_ok=True)
    (world["nets"] / f"{DIGEST}-{procedure}.json").write_text(json.dumps({
        "procedure": procedure, "ruleset_hash": DIGEST, "inputs_version": inputs_version,
        "verdict": verdict, "weights_sha256": WEIGHTS,
        "acceptance": {"score": 0.6 if verdict == "accepted" else 0.2, "error": 0.05,
                       "games": 48},
    }), encoding="utf-8")


def _converged(world: dict[str, Path], move: float, sd: float, into: str | None = None) -> None:
    path = corpus.measurement_dir(DIGEST, world["runs"]) / trust.CONVERGENCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "move": move, "sd": sd, "to": into or trust.measurement_digest(_basis()),
    }), encoding="utf-8")


def _read(
    world: dict[str, Path], people: trust.People | None = None, rule: trust.Rule = trust.RULE
) -> trust.Trust:
    return trust.read(DIGEST, world["runs"], SPEC, people, nets_dir=world["nets"], rule=rule)


# ------------------------------------------------------------------ the score's own state


class TestTheScore:
    def test_no_score_no_level(self, world) -> None:
        read = _read(world)
        assert (read.score, read.level, read.sentence) == (trust.ABSENT, None, "")
        assert [item.kind for item in read.items] == list(trust.KINDS)

    def test_an_out_of_date_score_has_no_level_even_with_a_finding(self, world) -> None:
        basis = _basis()
        basis["weights_version"] = "v1"
        _measured(world, unreadable=(("x",),) * 4, basis=basis)
        _net(world)
        read = _read(world)
        assert read.score == trust.STALE
        assert read.level is None
        assert read.item(trust.LADDER).reading == trust.FINDS_MORE

    def test_an_unreadable_score_is_no_score(self, world) -> None:
        out = corpus.measurement_dir(DIGEST, world["runs"])
        out.mkdir(parents=True)
        (out / "canonical.json").write_text("{not json", encoding="utf-8")
        assert _read(world).level is None

    def test_nothing_but_the_score_is_an_early_estimate(self, world) -> None:
        _measured(world)
        read = _read(world)
        assert (read.score, read.level, read.sentence) == (
            trust.CURRENT, trust.EARLY, "level.early")
        states = {item.kind: item.state for item in read.items}
        assert states == {trust.LADDER: trust.CURRENT, trust.SELF_DUEL: trust.ABSENT,
                          trust.FAIRY: trust.ABSENT, trust.NET: trust.ABSENT,
                          trust.CONVERGENCE: trust.ABSENT, trust.PEOPLE: trust.ABSENT}


# ------------------------------------------------------------------ E-L, the ladder


class TestTheLadder:
    def test_a_withheld_ladder_questions(self, world) -> None:
        _measured(world, unreadable=(("skill_gradient",),) * 4)
        read = _read(world)
        ladder = read.item(trust.LADDER)
        assert (ladder.reading, ladder.sentence) == (trust.FINDS_MORE, "ladder.withheld")
        assert ladder.count == trust.Count(4, 4, "runs")
        assert read.level == trust.QUESTIONED

    def test_the_score_withholds_only_what_every_run_withheld(self, world) -> None:
        # `seeds._unreadable` is the intersection: one blind run of four is not a
        # withheld ladder, and the trust line reads the score, not the run.
        _measured(world, unreadable=(("skill_gradient",), (), (), ()))
        ladder = _read(world).item(trust.LADDER)
        assert ladder.reading == trust.SILENT
        assert ladder.count == trust.Count(1, 4, "runs")

    @pytest.mark.parametrize(("blind", "read", "those"), [
        (1, 3, "those runs"), (2, 2, "those runs"), (3, 1, "that run")])
    def test_a_partly_blind_ladder_says_how_many_runs_read_it(
        self, world, blind: int, read: int, those: str
    ) -> None:
        # Q793 (`plan-confidence.md` §2.3.1): *dull* is blind on 2 of 4 runs and read
        # *steadily stronger*. Words only: the item stays silent and the level unmoved.
        _measured(world, unreadable=(("skill_gradient",),) * blind + ((),) * (4 - blind))
        read_ = _read(world)
        ladder = read_.item(trust.LADDER)
        assert (ladder.reading, ladder.sentence) == (trust.SILENT, "ladder.partly")
        assert ladder.count == trust.Count(blind, 4, "runs")
        assert read_.level == trust.EARLY
        text = trust.said(ladder)
        assert f"in only {read} of the 4 runs that measured this game" in text
        assert text.endswith(f"rest on {those} alone.")
        assert "steadily" not in text

    def test_no_blind_run_keeps_the_ordered_sentence(self, world) -> None:
        _measured(world)
        read = _read(world)
        ladder = read.item(trust.LADDER)
        assert ladder.sentence == "ladder.ordered"
        assert trust.said(ladder).startswith("Its computer players get steadily stronger")
        assert read.level == trust.EARLY

    def test_every_run_blind_keeps_the_withheld_sentence(self, world) -> None:
        _measured(world, unreadable=(("skill_gradient",),) * 4)
        read = _read(world)
        ladder = read.item(trust.LADDER)
        assert ladder.sentence == "ladder.withheld"
        assert trust.said(ladder).startswith(
            "Its computer players don't get clearly stronger with more thinking time here")
        assert read.level == trust.QUESTIONED

    def test_an_ordered_ladder_never_agrees(self, world) -> None:
        _measured(world)
        assert _read(world).item(trust.LADDER).reading == trust.SILENT

    def test_a_score_with_no_runs_list_reads_unreadable(self, world) -> None:
        _measured(world)
        path = corpus.measurement_dir(DIGEST, world["runs"]) / "canonical.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["runs"] = "lost"
        path.write_text(json.dumps(stored), encoding="utf-8")
        ladder = _read(world).item(trust.LADDER)
        assert (ladder.state, ladder.reading) == (trust.UNREADABLE, trust.SILENT)


# ------------------------------------------------------------------ E-C and E-F, premise.json


class TestTheStrengthCheck:
    def test_the_self_duel_is_reported_and_never_counted(self, world) -> None:
        _measured(world)
        _premise(world, ceiling=0.0)
        read = _read(world)
        duel = read.item(trust.SELF_DUEL)
        assert (duel.state, duel.reading) == (trust.CURRENT, trust.REPORTED)
        assert duel.count == trust.Count(0, 32, "games")
        assert read.level == trust.EARLY

    def test_fairy_stockfish_below_the_floor_questions(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.29)
        read = _read(world)
        fairy = read.item(trust.FAIRY)
        assert (fairy.reading, fairy.sentence) == (trust.FINDS_MORE, "fairy.beats-ours")
        assert read.level == trust.QUESTIONED

    def test_fairy_stockfish_at_the_floor_checks(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.30)
        read = _read(world)
        fairy = read.item(trust.FAIRY)
        assert (fairy.state, fairy.reading, fairy.sentence) == (
            trust.CURRENT, trust.AGREES, "fairy.held-its-own")
        assert fairy.count == trust.Count(10, 32, "games")
        assert read.level == trust.CHECKED
        assert read.sentence == "level.checked"

    def test_fairy_stockfish_that_did_not_play_is_silent(self, world) -> None:
        _measured(world)
        _premise(world, fairy=None)
        read = _read(world)
        fairy = read.item(trust.FAIRY)
        assert (fairy.reading, fairy.sentence) == (trust.SILENT, "fairy.not-played")
        assert read.level == trust.EARLY

    def test_a_stale_agreement_counts_for_nothing(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.45, stale=True)
        read = _read(world)
        fairy = read.item(trust.FAIRY)
        assert (fairy.state, fairy.reading) == (trust.STALE, trust.AGREES)
        assert fairy.detail["reasons"]
        assert read.level == trust.EARLY

    def test_a_stale_finding_still_questions(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.1, stale=True)
        read = _read(world)
        assert read.item(trust.FAIRY).state == trust.STALE
        assert read.level == trust.QUESTIONED

    def test_an_unreadable_check_is_silent_never_guessed(self, world) -> None:
        _measured(world)
        path = premise.premise_path(DIGEST, world["runs"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        read = _read(world)
        assert read.item(trust.FAIRY).state == trust.UNREADABLE
        assert read.item(trust.SELF_DUEL).state == trust.UNREADABLE
        assert read.level == trust.EARLY

    def test_a_check_that_records_no_basis_is_unreadable(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.1)
        path = premise.premise_path(DIGEST, world["runs"])
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["basis"] = None
        path.write_text(json.dumps(stored), encoding="utf-8")
        read = _read(world)
        assert read.item(trust.FAIRY).state == trust.UNREADABLE
        assert read.level == trust.EARLY


# ------------------------------------------------------------------ E-N, a trained player


class TestATrainedPlayer:
    def test_an_accepted_net_outside_the_score_questions(self, world) -> None:
        _measured(world)
        _net(world)
        read = _read(world)
        net = read.item(trust.NET)
        assert (net.state, net.reading, net.sentence) == (
            trust.CURRENT, trust.FINDS_MORE, "net.beats-ours")
        assert net.count == trust.Count(29, 48, "games")
        assert read.level == trust.QUESTIONED

    def test_a_refused_net_is_reported_not_counted(self, world) -> None:
        _measured(world)
        _net(world, verdict="refused")
        read = _read(world)
        net = read.item(trust.NET)
        assert (net.reading, net.sentence) == (trust.SILENT, "net.refused")
        assert read.level == trust.EARLY

    def test_a_net_the_score_plays_is_silent(self, world) -> None:
        basis = _basis()
        basis["net"] = f"{PROCEDURE}=net:{WEIGHTS[:12]}"
        _measured(world, basis=basis)
        _net(world)
        net = _read(world).item(trust.NET)
        assert (net.reading, net.sentence) == (trust.SILENT, "net.in-the-score")

    def test_a_different_net_in_the_score_does_not_answer_this_one(self, world) -> None:
        basis = _basis()
        basis["net"] = f"{PROCEDURE}=net:{'cd' * 6}"
        _measured(world, basis=basis)
        _net(world)
        assert _read(world).item(trust.NET).reading == trust.FINDS_MORE

    def test_a_net_on_an_older_layout_is_a_stale_finding_and_still_questions(self, world) -> None:
        _measured(world)
        _net(world, inputs_version=INPUTS_VERSION - 1)
        read = _read(world)
        assert read.item(trust.NET).state == trust.STALE
        assert read.level == trust.QUESTIONED

    def test_a_finding_wins_over_a_refusal(self, world) -> None:
        _measured(world)
        _net(world, verdict="refused", procedure="000000000001")
        _net(world)
        assert _read(world).item(trust.NET).reading == trust.FINDS_MORE

    def test_an_unreadable_store_is_silent(self, world) -> None:
        _measured(world)
        world["nets"].mkdir(parents=True)
        (world["nets"] / f"{DIGEST}-{PROCEDURE}.json").write_text("{", encoding="utf-8")
        read = _read(world)
        assert read.item(trust.NET).state == trust.UNREADABLE
        assert read.level == trust.EARLY

    def test_another_games_net_is_not_this_games(self, world) -> None:
        _measured(world)
        world["nets"].mkdir(parents=True)
        (world["nets"] / f"{'f' * 12}-{PROCEDURE}.json").write_text("{}", encoding="utf-8")
        assert _read(world).item(trust.NET).state == trust.ABSENT


# ------------------------------------------------------------------ E-V, convergence


class TestConvergence:
    def test_a_move_of_twice_the_spread_questions(self, world) -> None:
        _measured(world)
        _converged(world, move=-0.5, sd=0.25)
        read = _read(world)
        item = read.item(trust.CONVERGENCE)
        assert (item.state, item.reading, item.sentence) == (
            trust.CURRENT, trust.FINDS_MORE, "convergence.moved")
        assert read.level == trust.QUESTIONED

    def test_a_move_under_twice_the_spread_checks(self, world) -> None:
        _measured(world)
        _converged(world, move=0.499, sd=0.25)
        read = _read(world)
        assert read.item(trust.CONVERGENCE).sentence == "convergence.held"
        assert read.level == trust.CHECKED

    def test_a_record_about_another_measurement_is_stale(self, world) -> None:
        _measured(world)
        _converged(world, move=0.0, sd=0.25, into="000000000000")
        read = _read(world)
        assert read.item(trust.CONVERGENCE).state == trust.STALE
        assert read.level == trust.EARLY

    def test_a_stale_move_still_questions(self, world) -> None:
        _measured(world)
        _converged(world, move=1.0, sd=0.25, into="000000000000")
        assert _read(world).level == trust.QUESTIONED

    @pytest.mark.parametrize("payload", [
        {"move": 0.1}, {"move": "x", "sd": 0.1, "to": ""}, {"move": 0.1, "sd": -1, "to": ""},
        [],
    ])
    def test_an_unreadable_record_is_silent(self, world, payload: Any) -> None:
        _measured(world)
        path = corpus.measurement_dir(DIGEST, world["runs"]) / trust.CONVERGENCE_FILE
        path.write_text(json.dumps(payload), encoding="utf-8")
        read = _read(world)
        assert read.item(trust.CONVERGENCE).state == trust.UNREADABLE
        assert read.level == trust.EARLY


# ------------------------------------------------------------------ E-H, people


class TestPeople:
    def test_people_move_the_level_since_the_users_answer(self, world) -> None:
        """§5 Q3, answered A on 2026-10-03: `RULE.people_move_level` is on."""
        _measured(world)
        read = _read(world, trust.People(games=40, people=9, cleared=3, latest="2026-10-01"))
        item = read.item(trust.PEOPLE)
        assert (item.reading, item.counted, item.date) == (
            trust.FINDS_MORE, True, "2026-10-01")
        assert item.count == trust.Count(3, 9, "people")
        assert read.level == trust.QUESTIONED
        assert "reported only for now" not in read.rule

    def test_with_the_switch_off_people_are_reported_only(self, world) -> None:
        _measured(world)
        off = trust.Rule(people_move_level=False)
        read = _read(world, trust.People(games=40, people=9, cleared=3), rule=off)
        assert not read.item(trust.PEOPLE).counted and read.level == trust.EARLY
        assert "reported only for now" in read.rule

    @pytest.mark.parametrize(("people", "reading", "level"), [
        (trust.People(games=3, people=2, cleared=2), trust.FINDS_MORE, trust.QUESTIONED),
        (trust.People(games=3, people=1, cleared=1), trust.SILENT, trust.EARLY),
        (trust.People(games=20, people=5, cleared=0), trust.AGREES, trust.CHECKED),
        (trust.People(games=19, people=5, cleared=0), trust.SILENT, trust.EARLY),
        (trust.People(games=20, people=4, cleared=0), trust.SILENT, trust.EARLY),
        (trust.People(games=20, people=5, cleared=1), trust.SILENT, trust.EARLY),
    ])
    def test_the_thresholds_when_people_count(
        self, world, people: trust.People, reading: str, level: str
    ) -> None:
        _measured(world)
        read = _read(world, people, rule=PEOPLE_COUNT)
        assert read.item(trust.PEOPLE).reading == reading
        assert read.level == level

    def test_no_counts_is_absent(self, world) -> None:
        _measured(world)
        assert _read(world, rule=PEOPLE_COUNT).item(trust.PEOPLE).state == trust.ABSENT


# ------------------------------------------------------------------ the level


class TestTheLevel:
    def test_a_finding_beats_an_agreement(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.5)
        _net(world)
        assert _read(world).level == trust.QUESTIONED

    def test_two_agreements_are_still_only_checked(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.5)
        _converged(world, move=0.0, sd=0.1)
        assert _read(world).level == trust.CHECKED

    def test_the_clauses_alone(self) -> None:
        def item(reading: str, state: str = trust.CURRENT, counted: bool = True) -> trust.Item:
            return trust.Item(trust.FAIRY, state, reading, counted=counted)

        assert trust.level_of([]) == trust.EARLY
        assert trust.level_of([item(trust.FINDS_MORE, trust.STALE)]) == trust.QUESTIONED
        assert trust.level_of([item(trust.AGREES, trust.STALE)]) == trust.EARLY
        assert trust.level_of([item(trust.AGREES), item(trust.SILENT)]) == trust.CHECKED
        assert trust.level_of([item(trust.FINDS_MORE, counted=False)]) == trust.EARLY
        assert trust.level_of([item(trust.REPORTED)]) == trust.EARLY

    def test_the_rule_is_spelled_with_its_thresholds(self) -> None:
        spelled = trust.RULE.spelled()
        for threshold in ("0.5", "below 0.3", "2 sd", "2 or more people",
                          "20 or more games", "5 or more people"):
            assert threshold in spelled
        assert trust.Rule(0.30, 2.0, 2, 20, 5, True) == trust.RULE


# ------------------------------------------------------------------ it moves nothing


class TestItMovesNothing:
    def test_a_reading_is_a_pure_function_of_the_files_and_writes_none(self, world) -> None:
        _measured(world)
        _premise(world, fairy=0.5)
        _net(world, verdict="refused")
        _converged(world, move=0.01, sd=0.1)
        before = sorted(p.relative_to(world["runs"].parent) for p in world["runs"].parent.rglob("*"))
        people = trust.People(games=4, people=2, cleared=0)
        first = _read(world, people).as_dict()
        assert _read(world, people).as_dict() == first
        after = sorted(p.relative_to(world["runs"].parent) for p in world["runs"].parent.rglob("*"))
        assert before == after
        assert json.loads(json.dumps(first)) == first

    def test_it_is_in_no_fingerprint_set(self) -> None:
        ours = (measurement.ROOT / "chess_variant_sandbox/trust.py").resolve()
        for kind in ("engine", "metrics", "analysis"):
            assert ours not in {path.resolve() for path in measurement.fingerprinted(kind)}, kind

    def test_no_metric_imports_it(self) -> None:
        metrics = measurement.ROOT / "chess_variant_sandbox" / "metrics"
        for path in metrics.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "import trust" not in text and "trust import" not in text, path
