"""One generation in eight may explore past a narrow or unsure refusal (R-S17j).

`docs/reference/wrong-no.md` §10 is the design, and the roadmap's R-S17j block the
acceptance: a players' refusal within two standard errors of its line, read from its own
games, or one the full roster could not read, may parent a capped share of generations. The
refused entry is never rewritten and never shown or counted as viable; the mark is on its
children (`detail["past_refusal"]`). Two generations a family, one generation in eight as a
count, no second seed.

The rule is read on stored entries where it can be (`refusals.narrow`, `eligible`,
`family`, `orchestrator._past_refusal_parent`); the loop end to end plays its own small
games with no model, at `--mechanical-rate 1.0`, and the proposer's words are read off a
recorder standing in for the model.
"""

from __future__ import annotations

import copy
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
import test_loop_end_to_end as e2e

from chess_variant_sandbox.llm import feedback
from chess_variant_sandbox.loop import Archive, Entry, LoopConfig, evolve, orchestrator, refusals
from chess_variant_sandbox.loop.gates import check_row, count_row, rate_row
from chess_variant_sandbox.loop.grid import descriptor, label
from chess_variant_sandbox.metrics import load_weights, spread
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.tournament.ratings import rate
from chess_variant_sandbox.tournament.record import GameRecord

GOOD_SPEC = e2e.GOOD_SPEC
ROSTER = ("random", "minimax:nodes=80,threats=0,mopup=0", "minimax:nodes=320,threats=0,mopup=0")


class Always(random.Random):
    """A stream whose every draw is taken: the cap and the families are then all that
    limit the share."""

    def random(self) -> float:
        return 0.0


def _config(directory: Path, **overrides: Any) -> LoopConfig:
    settings: dict[str, Any] = {
        "generations": 1, "agents": ROSTER, "games_per_pairing": 1, "workers": 1,
        "smoke_games": 12, "mechanical_rate": 1.0, "judge": False, "batch": 3,
        "directory": directory,
    }
    settings.update(overrides)
    return LoopConfig(**settings)


def _refusal(rung: str, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A refusal as the loop stores one since R-S17j."""
    for row in rows:
        row.setdefault("kind", refusals.PLAYERS)
        row.setdefault("reason", row["check"])
    found = {
        "rung": rung, "kind": refusals.kind_of(row["kind"] for row in rows), "checks": rows,
        "roster": list(ROSTER), "weights_version": "v14", "seed": 0, refusals.SPREAD: True,
    }
    found.update(extra)
    return found


def _no_decisive() -> dict[str, Any]:
    """The screen's *no decisive game*: narrow by arithmetic (§10.2)."""
    return _refusal(refusals.SCREEN, [count_row("decisive", 0, 1, "min", 8)])


def _spec(tag: int) -> dict[str, Any]:
    spec = copy.deepcopy(GOOD_SPEC)
    spec["rules"]["max_plies"] = 100 + tag
    return spec


def _elite(archive: Archive, digest: str, tag: int, generation: int = 0) -> Entry:
    spec = _spec(tag)
    return archive.record(
        Entry(hash=digest, name=digest, generation=generation, parent=None, status="evaluated",
              index=0.5, index_ci_low=0.4, index_ci_high=0.6, seeds=[0], viable=True,
              cell=label(descriptor(spec))),
        spec=spec,
    )


def _refused(archive: Archive, digest: str, tag: int, parent: str | None,
             refusal: dict[str, Any] | None, generation: int = 0,
             status: str = "rejected") -> Entry:
    return archive.record(
        Entry(hash=digest, name=digest, generation=generation, parent=parent, status=status,
              notes="Rejected before the full roster: nothing came of it", refusal=refusal),
        spec=_spec(tag),
    )


def _child(archive: Archive, picked: tuple[Entry, dict[str, Any]], generation: int,
           viable: bool = False) -> Entry:
    """What a generation past a refusal leaves: one marked child."""
    entry, mark = picked
    status = "evaluated" if viable else "rejected"
    return archive.record(Entry(
        hash=f"child-{generation}", name="child", generation=generation, parent=entry.hash,
        status=status, viable=viable, index=0.5 if viable else None, seeds=[0] if viable else [],
        detail={refusals.PAST_REFUSAL: mark},
    ))


# --------------------------------------------------------------- (6) §10.2, row by row


class TestNarrowCheckByCheck:
    """Every row of §10.2's table: narrow and not narrow, or never."""

    def test_the_stall_line_is_narrow_on_every_refusal(self) -> None:
        # The line is 0.98 of 150; a refusal cannot be more than 0.02 past it.
        assert refusals.narrow(_p(rate_row("stall_rate", 1.0, 0.98, "max", 150)))
        assert refusals.narrow(_p(rate_row("stall_rate", 1.0, 0.98, "max", 12)))

    def test_no_random_winner_is_narrow_on_every_refusal(self) -> None:
        for games in (30, 150):
            assert refusals.narrow(_p(count_row("decided", 0, 1, "min", games)))

    def test_the_forced_rate_is_narrow_within_0_082_of_150(self) -> None:
        assert refusals.narrow(_p(rate_row("forced_rate", 0.58, 0.5, "max", 150)))
        assert not refusals.narrow(_p(rate_row("forced_rate", 0.6, 0.5, "max", 150)))

    def test_the_random_seat_skew_reads_its_decided_games(self) -> None:
        assert refusals.narrow(_p(rate_row("seat_skew", 1.0, 0.85, "max", 20)))
        assert refusals.narrow(_p(rate_row("seat_skew", 0.9, 0.85, "max", 150)))
        assert not refusals.narrow(_p(rate_row("seat_skew", 0.95, 0.85, "max", 150)))

    def test_the_screen_is_narrow_on_every_refusal(self) -> None:
        assert refusals.narrow(_p(count_row("decisive", 0, 1, "min", 8)))
        assert refusals.narrow(_p(rate_row("quick_kill_rate", 1.0, 0.75, "max", 8)))

    def test_a_check_by_the_rules_or_with_no_number_is_never_narrow(self) -> None:
        for check in ("rules_fit", "core_plays", "opening_moves", "over_at_start",
                      "mean_branching"):
            row = check_row(check, 1.0, 1.5, "min", n=150, se=10.0)
            row["kind"] = refusals.RULES
            assert not refusals.narrow(row), check
        # Machine time, a roster that cannot play, no games: no reading of the game, even
        # with a margin and a spread beside it.
        for check in ("clock", "roster", "games"):
            row = check_row(check, 100, 150, "min", n=150, se=100.0)
            row["kind"] = refusals.PLAYERS
            assert not refusals.narrow(row), check

    def test_a_row_without_its_games_is_never_narrow(self) -> None:
        row = check_row("stall_rate", 1.0, 0.98, "max")
        row["kind"] = refusals.PLAYERS
        assert not refusals.narrow(row)

    def test_the_full_rates_read_the_half_the_gate_reported(self) -> None:
        stalled = _ladder(stall_strong=True, games=4)
        said = spread.check_spread("stall_rate_worst_half", 0.7, stalled, rate(stalled), None)
        assert said == {"n": 4, "se": spread.binomial_se(0.7, 4)}
        row = _full_row("stall_rate_worst_half", 1.0, 0.7, "max", said)
        assert refusals.narrow(row)
        many = _ladder(stall_strong=True, games=40)
        said = spread.check_spread("stall_rate_worst_half", 0.7, many, rate(many), None)
        assert said is not None and said["n"] == 40
        assert not refusals.narrow(_full_row("stall_rate_worst_half", 1.0, 0.7, "max", said))
        for check in ("decisive_rate_worst_half", "quick_kill_rate_worst_half"):
            said = spread.check_spread(check, 0.5, stalled, rate(stalled), None)
            assert said is not None and said["n"] == 4, check

    def test_the_half_is_the_one_components_reports(self) -> None:
        from chess_variant_sandbox.metrics.components import _worst_half_rates

        records = _ladder(stall_strong=True, games=4)
        raw, _ = _worst_half_rates(records)
        for rate_name in ("stall_rate", "decisive_rate", "mean_plies", "quick_kill_rate"):
            half = spread.worst_half_games(records, rate_name)
            from chess_variant_sandbox.metrics.components import _outcome_rates

            assert _outcome_rates(half)[rate_name] == raw[rate_name + "_worst_half"]

    def test_mean_plies_reads_the_spread_of_its_own_games(self) -> None:
        # A game at 2.0 plies on every game has no spread: not narrow (§10.1's case).
        flat = _ladder(stall_strong=False, games=4, plies=[2, 2, 2, 2])
        said = spread.check_spread("mean_plies_worst_half", 8, flat, rate(flat), None)
        assert said is not None and said["se"] == 0.0
        assert not refusals.narrow(_full_row("mean_plies_worst_half", 2.0, 8, "min", said))
        varied = _ladder(stall_strong=False, games=4, plies=[2, 4, 6, 10])
        said = spread.check_spread("mean_plies_worst_half", 8, varied, rate(varied), None)
        assert said is not None
        assert refusals.narrow(_full_row("mean_plies_worst_half", 5.5, 8, "min", said))
        assert not refusals.narrow(_full_row("mean_plies_worst_half", 0.5, 8, "min", said))

    def test_the_search_advantage_reads_its_per_game_scores(self) -> None:
        from chess_variant_sandbox.metrics.components import _search_advantage

        records = _against_random([1, 1, 0, 1, 0.5, 1])
        ratings = rate(records)
        said = spread.check_spread("search_advantage", 0.55, records, ratings, None)
        assert said is not None and said["n"] == 6
        scores = [1.0, 1.0, 0.0, 1.0, 0.5, 1.0]
        assert said["se"] == pytest.approx(spread.mean_se(scores))
        assert _search_advantage(ratings) == pytest.approx(sum(scores) / 6)
        assert refusals.narrow(_full_row("search_advantage", 0.5, 0.55, "min", said))
        assert not refusals.narrow(_full_row("search_advantage", 0.0, 0.55, "min", said))

    def test_the_gradient_reads_the_index_resamples(self) -> None:
        interval = {"component_sd": {"skill_gradient": 3.0}}
        said = spread.check_spread("skill_gradient", 8, [_game("a", "b", 0, 10)] * 5,
                                   rate([]), interval)
        assert said == {"n": 5, "se": 3.0}
        assert refusals.narrow(_full_row("skill_gradient", 3.1, 8, "min", said))
        assert not refusals.narrow(_full_row("skill_gradient", 1.0, 8, "min", said))
        # Without the resamples' spread the gradient is never narrow.
        said = spread.check_spread("skill_gradient", 8, [], rate([]), None)
        assert said is not None
        assert not refusals.narrow(_full_row("skill_gradient", 7.9, 8, "min", said))

    def test_the_seat_gates_and_the_unread_are_never_narrow(self) -> None:
        seat = ("seat_chance=1.00e-04 below minimum 0.001: the first seat went 20-0 with 0 "
                "draws over the 20 games")
        found = refusals.full_refusal(
            [seat, "decision_cost was not measured", "something nobody wrote"],
            list(ROSTER), "v14", 0, spread=lambda check, line: {"n": 99, "se": 99.0},
        )
        assert [row["check"] for row in found["checks"]] == [
            "seat_chance", refusals.ABSENT_CHECK, refusals.UNKNOWN_CHECK]
        assert not any(refusals.narrow(row) for row in found["checks"])
        assert not refusals.eligible(found)
        # `mean_branching` at the full roster is by the rules: never asked for a spread.
        asked: list[str] = []
        refusals.full_refusal(["mean_branching=1.2 below minimum 1.5"], list(ROSTER), "v14", 0,
                              spread=lambda check, line: asked.append(check) or None)
        assert asked == []

    def test_a_refusal_that_names_no_check_is_unsure_and_eligible(self) -> None:
        unread = refusals.full_refusal(
            ["this roster cannot read this ruleset: the rating ladder orders the agents on "
             "only 0% of its budget steps, so skill_gradient would describe the players "
             "rather than the game"], list(ROSTER), "v14", 0, spread=lambda c, line: None)
        assert refusals.unsure(unread) and refusals.eligible(unread)


def _game(first: str, second: str, winner: int | None, plies: int,
          reason: str = "royal_captured") -> GameRecord:
    return GameRecord("h", "g", 0, (first, second), [], winner, reason, plies)


def _ladder(stall_strong: bool, games: int, plies: list[int] | None = None) -> list[GameRecord]:
    """Two halves of a ladder of greedy and two budgets: the strong half's games stall where
    asked, the weak half's are won."""
    low, mid, high = "greedy", "minimax:nodes=100", "minimax:nodes=400"
    strong = [
        _game(mid, high, None if stall_strong else 0,
              (plies[i % len(plies)] if plies else 60),
              "ply_limit" if stall_strong else "royal_captured")
        for i in range(games)
    ]
    weak = [_game(low, mid, 1, 30) for _ in range(games)]
    return strong + weak


def _against_random(scores: list[float]) -> list[GameRecord]:
    """The strongest player against a random mover, alternating seats, with these scores."""
    best = "minimax:nodes=400"
    records = []
    for at, score in enumerate(scores):
        if at % 2 == 0:
            winner = None if score == 0.5 else (0 if score == 1 else 1)
            records.append(_game(best, "random", winner, 20))
        else:
            winner = None if score == 0.5 else (1 if score == 1 else 0)
            records.append(_game("random", best, winner, 20))
    return records


def _p(row: dict[str, Any]) -> dict[str, Any]:
    """A gate's row as the refusal stores it, read off the players."""
    return {**row, "kind": refusals.PLAYERS}


def _full_row(check: str, value: float, line: float, side: str,
              said: dict[str, Any] | None) -> dict[str, Any]:
    row = check_row(check, value, line, side, **(said or {}))
    row["kind"] = refusals.PLAYERS
    return row


# ------------------------------------------------------------- eligible as a whole


class TestEligible:
    def test_narrow_and_read_off_the_players(self) -> None:
        assert refusals.eligible(_no_decisive())

    def test_one_far_check_keeps_it_out(self) -> None:
        far = _refusal(refusals.SMOKE, [
            rate_row("stall_rate", 1.0, 0.98, "max", 150),
            rate_row("forced_rate", 0.9, 0.5, "max", 150),
        ])
        assert not refusals.eligible(far)

    def test_any_check_by_the_rules_keeps_it_out(self) -> None:
        both = _refusal(refusals.SMOKE, [
            rate_row("stall_rate", 1.0, 0.98, "max", 150),
            {**check_row("opening_moves", 0, 1, "min"), "kind": refusals.RULES},
        ])
        assert both["kind"] == refusals.BOTH and not refusals.eligible(both)

    def test_a_reconstructed_or_pre_build_refusal_is_never_eligible(self) -> None:
        assert not refusals.eligible({**_no_decisive(), "reconstructed": True})
        old = _no_decisive()
        del old[refusals.SPREAD]
        assert not refusals.eligible(old)
        assert not refusals.eligible(None)

    def test_new_refusals_carry_their_games(self) -> None:
        cheap = refusals.cheap_refusal(
            refusals.SCREEN, ["no computer player won any of 8 games"],
            [count_row("decisive", 0, 1, "min", 8)], ROSTER, {}, "v14", 0)
        [row] = cheap["checks"]
        assert row["n"] == 8 and row["se"] == pytest.approx(spread.count_se(1, 8), abs=1e-6)
        assert refusals.eligible(cheap)


# ---------------------------------------------- (2) and (4): the cap and the families


def _families_archive(directory: Path, families: int) -> Archive:
    """``families`` elites, each with one narrowly refused child: that many families."""
    archive = Archive(directory)
    for at in range(families):
        _elite(archive, f"elite{at}", 2 * at)
        _refused(archive, f"refused{at}", 2 * at + 1, f"elite{at}", _no_decisive())
    return archive


class TestTheCap:
    def test_at_most_one_generation_in_eight_as_a_count(self, tmp_path) -> None:
        archive = _families_archive(tmp_path, 6)
        config = _config(tmp_path)
        picked_at = []
        for generation in range(64):
            picked = orchestrator._past_refusal_parent(
                archive, config, Always(), generation, explore=False)
            if picked is not None:
                picked_at.append(generation)
                _child(archive, picked, generation)
            assert 8 * len(picked_at) <= generation + 1
        # Every draw taken: the cap alone sets when, and the first seven never.
        assert picked_at == [7, 15, 23, 31, 39, 47, 55, 63]

    def test_at_the_rate_the_count_still_holds(self, tmp_path) -> None:
        archive = _families_archive(tmp_path, 6)
        config, rng, picked = _config(tmp_path), random.Random(5), 0
        for generation in range(200):
            found = orchestrator._past_refusal_parent(archive, config, rng, generation, True)
            if found is not None:
                picked += 1
                _child(archive, found, generation)
            assert 8 * picked <= generation + 1
        assert picked > 0

    def test_a_larger_share_is_refused(self, tmp_path) -> None:
        for wrong in (0.2, -0.1):
            with pytest.raises(ValueError, match="refused_rate"):
                _config(tmp_path, refused_rate=wrong)
        assert LoopConfig(directory=tmp_path).refused_rate == 0.125

    def test_the_kinds_that_are_never_a_parent(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        _elite(archive, "elite", 0)
        rules = _refusal(refusals.SMOKE, [
            {**check_row("opening_moves", 0, 1, "min"), "kind": refusals.RULES}])
        far = _refusal(refusals.SMOKE, [rate_row("forced_rate", 0.9, 0.5, "max", 150)])
        seat = refusals.full_refusal(
            ["seat_chance=1.00e-04 below minimum 0.001: the first seat went 20-0 with 0 "
             "draws over the 20 games"], list(ROSTER), "v14", 0, spread=lambda c, line: None)
        _refused(archive, "rules", 1, "elite", rules)
        _refused(archive, "far", 2, "elite", far)
        _refused(archive, "seat", 3, "elite", seat, status="evaluated")
        # Reconstructed: written before refusals were stored, read back from its sentence.
        archive.record(Entry(
            hash="old", name="old", generation=0, parent="elite", status="rejected",
            notes="Rejected before the full roster: no computer player won any of 8 games",
            detail={"smoke": {"passed": True, "reasons": []},
                    "screen": {"passed": False, "reasons": [
                        "no computer player won any of 8 games: either nobody can win"]}},
        ), spec=_spec(4))
        assert refusals.refusal_of(archive.latest("old"))["reconstructed"]
        assert refusals.open_families(archive) == {}
        for generation in range(64):
            assert orchestrator._past_refusal_parent(
                archive, _config(tmp_path), Always(), generation, False) is None


class TestFamilies:
    def test_seven_refused_siblings_are_one_family_tried_twice(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        _elite(archive, "elite", 0)
        for at in range(7):
            _refused(archive, f"sibling{at}", at + 1, "elite", _no_decisive())
        families = refusals.open_families(archive)
        assert list(families) == [("elite", refusals.SCREEN, ("decisive",))]
        assert len(families[("elite", refusals.SCREEN, ("decisive",))]) == 7
        picked_at = []
        for generation in range(200):
            picked = orchestrator._past_refusal_parent(
                archive, _config(tmp_path), Always(), generation, False)
            if picked is not None:
                picked_at.append(generation)
                _child(archive, picked, generation)
        assert picked_at == [7, 15]

    def test_a_viable_child_closes_the_family(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        _elite(archive, "elite", 0)
        _refused(archive, "sibling", 1, "elite", _no_decisive())
        picked = orchestrator._past_refusal_parent(archive, _config(tmp_path), Always(), 7, False)
        assert picked is not None
        _child(archive, picked, 7, viable=True)
        assert refusals.open_families(archive) == {}

    def test_a_chain_of_refusals_keys_to_its_nearest_unrefused_ancestor(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        _elite(archive, "elite", 0)
        child = _refused(archive, "child", 1, "elite", _no_decisive())
        grandchild = _refused(archive, "grandchild", 2, "child", _no_decisive())
        assert refusals.family(archive, grandchild) == refusals.family(archive, child)
        assert refusals.family(archive, child)["root"] == "elite"
        # From scratch, the top of the chain.
        _refused(archive, "top", 3, None, _no_decisive())
        below = _refused(archive, "below", 4, "top", _no_decisive())
        assert refusals.family(archive, below)["root"] == "top"

    def test_the_pick_is_over_families_with_an_empty_cell_first(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        # One family whose member sits in the elite's own cell, three of them...
        _elite(archive, "elite", 0)
        for at in range(3):
            _refused(archive, f"crowd{at}", 0, "elite", _no_decisive())
        # ...and one whose member sits in a cell no elite holds.
        spec = copy.deepcopy(GOOD_SPEC)
        spec["board"] = {"width": 8, "height": 8}
        spec["setup"]["placements"][0]["squares"] = ["d1"]
        assert descriptor(spec) != descriptor(_spec(0))
        archive.record(Entry(hash="lone", name="lone", generation=0, parent=None,
                             status="rejected", refusal=_refusal(refusals.SMOKE, [
                                 rate_row("stall_rate", 1.0, 0.98, "max", 150)])), spec=spec)
        for seed in range(20):
            picked = orchestrator._past_refusal_parent(
                archive, _config(tmp_path), _AlwaysSeeded(seed), 7, False)
            assert picked is not None and picked[0].hash == "lone"


class _AlwaysSeeded(Always):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)


# ------------------------------------------------------ (1) at rate 0, today's draws


class TestRateZero:
    def test_no_draw_is_made(self, tmp_path) -> None:
        archive = _families_archive(tmp_path, 3)
        rng = random.Random(3)
        before = rng.getstate()
        config = _config(tmp_path, refused_rate=0.0)
        for generation in (7, 63):
            assert orchestrator._past_refusal_parent(archive, config, rng, generation,
                                                     True) is None
        assert rng.getstate() == before

    def test_nothing_eligible_draws_nothing_either(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        _elite(archive, "elite", 0)
        rng = random.Random(3)
        before = rng.getstate()
        assert orchestrator._past_refusal_parent(archive, _config(tmp_path), rng, 63,
                                                 True) is None
        assert rng.getstate() == before

    def test_a_run_at_zero_is_the_run_without_the_mechanism(
        self, tmp_path, monkeypatch
    ) -> None:
        seed = _a_seed_that_draws(explore=False)
        runs = []
        for name, rate_ in (("zero", 0.0), ("without", None)):
            directory = tmp_path / name
            _refused_parent_archive(directory)
            # An elite beside it, so the run has a parent to revise either way.
            _elite(Archive(directory), "elite", 1, generation=7)
            with monkeypatch.context() as patch:
                if rate_ is None:
                    patch.setattr(orchestrator, "_past_refusal_parent",
                                  lambda *args, **kwargs: None)
                config = _config(directory, seed=seed, refused_rate=rate_ or 0.0, batch=1)
                archive = evolve(config, report=lambda _: None)
            runs.append(_rows(archive))
        assert runs[0] == runs[1]
        assert not any(row[-1] for row in runs[0])


# ------------------------------------------- (3) and (5): the loop end to end


def _a_seed_that_draws(explore: bool) -> int:
    """A seed whose second draw takes the share after its first, the explore draw."""
    chance = 0.125 / 0.7 if explore else 0.125
    for seed in range(1000):
        rng = random.Random(seed)
        first, second = rng.random(), rng.random()
        if (not explore or first >= 0.3) and second < chance:
            return seed
    raise AssertionError("no seed in a thousand draws the share")


def _refused_parent_archive(directory: Path) -> str:
    """An archive at generation 7 holding one narrowly refused game and nothing else."""
    archive = Archive(directory)
    digest = spec_hash(GOOD_SPEC)
    archive.record(
        Entry(hash=digest, name="lancers", generation=7, parent=None, status="rejected",
              notes="Rejected before the full roster: no computer player won any of 8 games",
              refusal=_no_decisive()),
        spec=copy.deepcopy(GOOD_SPEC),
    )
    return digest


def _rows(archive: Archive) -> list[tuple[Any, ...]]:
    return [
        (e.hash, e.status, e.viable, e.index, e.parent, e.separated, e.refusal,
         (e.detail or {}).get(refusals.PAST_REFUSAL))
        for e in archive.entries
    ]


@pytest.fixture(scope="module")
def explored(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, list[str], bytes]:
    """One generation past a refusal, played once for the class."""
    directory = tmp_path_factory.mktemp("explored")
    digest = _refused_parent_archive(directory)
    before = (directory / "lineage.jsonl").read_bytes()
    lines: list[str] = []
    evolve(_config(directory, seed=_a_seed_that_draws(explore=False)), report=lines.append)
    return directory, digest, lines, before


class TestAGenerationPastARefusal:
    def test_its_children_carry_the_mark(self, explored) -> None:
        directory, digest, lines, _ = explored
        archive = Archive(directory)
        children = archive.entries[1:]
        assert children
        for child in children:
            mark = child.detail[refusals.PAST_REFUSAL]
            assert child.parent == digest and child.generation == 8
            assert mark["parent"] == digest and mark["rung"] == refusals.SCREEN
            assert mark["checks"] == ["decisive"]
            assert mark["family"] == {"root": digest, "rung": refusals.SCREEN,
                                      "checks": ["decisive"]}
            # A refused game is nothing to beat.
            assert child.separated is None
        said = next(line for line in lines if line.startswith("exploring past a refusal"))
        assert digest in said and "as these players played it" in said
        assert "index" not in said and "score" not in said

    def test_the_refused_game_is_never_shown_or_counted_as_viable(self, explored) -> None:
        directory, digest, _, before = explored
        archive = Archive(directory)
        assert (directory / "lineage.jsonl").read_bytes().startswith(before)
        [parent] = [e for e in archive.entries if e.hash == digest]
        assert parent.status == "rejected" and not parent.viable
        assert digest not in {e.hash for e in archive.grid().elites.values()}
        assert digest not in {e.hash for e in archive.leaderboard(100)}
        assert digest not in {e.hash for e in archive.top_group(100)}
        summary = archive.summary()
        assert digest not in summary
        explored_, measured, viable = refusals.counted(archive.entries)
        assert explored_ == 1
        assert measured == sum(1 for e in archive.entries[1:] if e.status == "evaluated")
        assert viable == sum(1 for e in archive.entries[1:] if e.viable)
        assert (f"1 generation explored past a refusal: {measured} of their children "
                f"measured, {viable} of those viable") in summary

    def test_a_run_is_the_same_run_twice_and_resumes_the_same(self, explored, tmp_path) -> None:
        directory, _, _, _ = explored
        again = tmp_path / "again"
        _refused_parent_archive(again)
        seed = _a_seed_that_draws(explore=False)
        assert _rows(evolve(_config(again, seed=seed), report=lambda _: None)) == _rows(
            Archive(directory))
        # Resumed, each copy makes the same next generation.
        resumed = []
        for name in ("one", "two"):
            copy_of = tmp_path / name
            shutil.copytree(directory, copy_of)
            resumed.append(_rows(evolve(_config(copy_of, seed=seed), report=lambda _: None)))
        assert resumed[0] == resumed[1]


class TestWhatTheProposerIsTold:
    def test_the_refusal_by_its_kind_and_no_number(self, tmp_path) -> None:
        """The leak test for a refused parent: R-S17h's words, never a number, a line, a
        spread or a component."""
        archive = Archive(tmp_path)
        refusal = _refusal(refusals.SMOKE, [rate_row("forced_rate", 0.5432, 0.5, "max", 150)])
        archive.record(Entry(hash="p", name="parent", generation=7, parent=None,
                             status="rejected", notes="It ended too soon.", refusal=refusal),
                       spec=copy.deepcopy(GOOD_SPEC))

        class Recorder:
            model = "claude-opus-4-8"

            def __init__(self) -> None:
                self.prompts: list[str] = []

            def structured(self, system, user, schema, max_tokens=16000):
                self.prompts.append(user)
                return copy.deepcopy(GOOD_SPEC)

        recorder = Recorder()
        seed = _a_seed_that_draws(explore=True)
        lines: list[str] = []
        orchestrator._run_generation(
            8, _config(tmp_path, mechanical_rate=0.0, seed=seed), archive,
            load_weights(), random.Random(seed), recorder, None,
            orchestrator.PreferenceLog(tmp_path), lines.append,
        )
        assert any(line.startswith("exploring past a refusal") for line in lines)
        prompt = recorder.prompts[0]
        assert feedback.PLAYERS_WORDS in prompt
        [row] = refusal["checks"]
        for number in ("0.5432", "0.0432", str(row["se"])):
            assert number not in prompt
        lowered = prompt.lower()
        for component in load_weights().components():
            assert component not in lowered
        for term in ("margin", "standard error", "narrow"):
            assert term not in lowered
        assert all(refusals.PAST_REFUSAL in e.detail for e in archive.entries[1:])


# ------------------------------------------------- a full roster's row carries its games


def test_a_full_roster_refusal_stores_its_games_and_spread(tmp_path, monkeypatch) -> None:
    import test_loop_refusals as refused

    refused._refuse_on(monkeypatch, refused.ROSTER_A)
    archive = evolve(
        refused._config(tmp_path, refused.ROSTER_A),
        proposer=refused._proposer(monkeypatch, GOOD_SPEC), report=lambda _: None,
    )
    [entry] = archive.entries
    [row] = entry.refusal["checks"]
    assert row["check"] == "skill_gradient"
    assert row["n"] and row["n"] > 0
    stored = (tmp_path / "index" / f"{entry.hash}.json").read_text(encoding="utf-8")
    import json

    interval = json.loads(stored)["interval"]
    assert row["se"] == pytest.approx(interval["component_sd"]["skill_gradient"], abs=1e-6)
    assert entry.refusal[refusals.SPREAD]
