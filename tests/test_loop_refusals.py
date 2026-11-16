"""A refused child is remembered by what refused it, and measured again when that moves (R-S17c).

`docs/reference/wrong-no.md` §1.2: the loop's "no" was final and silent. A refused or
non-viable child was never a parent and its hash was never evaluated again in that archive,
whatever changed. Now its entry stores the rung, the checks with their margins, the kind
(*by its rules* or *as these players play it*, the site's rule: `loop/refusals.py`) and the
basis it was read on, and `evolve` measures a players' refusal again where the basis differs.

The model is the real `Claude` over the fake transport of `test_loop_end_to_end.py`; the
games, the screen and the scoring are the real ones.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import test_loop_end_to_end as e2e

from chess_variant_sandbox.llm.client import PROPOSAL_MODEL
from chess_variant_sandbox.loop import (
    Archive,
    Entry,
    LoopConfig,
    evolve,
    gates,
    orchestrator,
    refusals,
)
from chess_variant_sandbox.web import sitescore

GOOD_SPEC = e2e.GOOD_SPEC
# The roster `test_loop_end_to_end._config` plays, on which `GOOD_SPEC` is viable.
ROSTER_B = (
    "random", "minimax:nodes=80,threats=0,mopup=0", "minimax:nodes=320,threats=0,mopup=0"
)
# Another roster: a different player at the bottom, the same two budgets.
ROSTER_A = (
    "greedy", "minimax:nodes=80,threats=0,mopup=0", "minimax:nodes=320,threats=0,mopup=0"
)

# Its one piece steps only toward its own back edge, from the back edge: no move at the start.
STUCK = copy.deepcopy(GOOD_SPEC)
STUCK["name"] = "stuck"
STUCK["pieces"] = [
    {"id": "K", "name": "captain", "royal": True, "value": 0, "moves": [
        {"kind": "leap", "vectors": [[0, -1]], "mode": "both"}
    ]}
]
STUCK["setup"] = {"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]}]}

# A scored refusal that reads the players, as `metrics/index.py` words one.
PLAYERS_SAID = "skill_gradient=3.1 below minimum 8"


def _config(tmp_path, agents: tuple[str, ...], **overrides: Any) -> LoopConfig:
    settings: dict[str, Any] = {
        "generations": 1,
        "agents": agents,
        "games_per_pairing": 1,
        "workers": 1,
        "smoke_games": 12,
        "mechanical_rate": 0.0,
        "judge": False,
        "directory": tmp_path,
    }
    settings.update(overrides)
    return LoopConfig(**settings)


def _proposer(monkeypatch, *specs: dict[str, Any]):
    anthropic = pytest.importorskip("anthropic")
    from chess_variant_sandbox.llm.client import Claude

    transport = e2e.TestTheRequestTheLoopActuallySends.Transport(
        [copy.deepcopy(spec) for spec in specs]
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: transport)
    return Claude(model=PROPOSAL_MODEL, effort="high")


def _refuse_on(monkeypatch, roster: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Score as usual, but refuse on a players' check whatever plays on ``roster``: the
    stand-in for a game these players cannot find the play in."""
    real = orchestrator.score_ruleset
    seen: list[tuple[str, ...]] = []

    def scored(spec, records, *args, **kwargs):
        result = real(spec, records, *args, **kwargs)
        played = tuple(sorted({record.agents[0] for record in records}
                              | {record.agents[1] for record in records}))
        seen.append(played)
        if set(played) == set(roster):
            return dataclasses.replace(result, viable=False, violations=[PLAYERS_SAID])
        return result

    monkeypatch.setattr(orchestrator, "score_ruleset", scored)
    return seen


def _quiet() -> dict[str, Any]:
    return {"report": lambda _: None}


@pytest.fixture(scope="module")
def refused_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One run under `ROSTER_A` whose one child these players refuse, played once for the
    class; each test works on its own copy."""
    directory = tmp_path_factory.mktemp("refused")
    with pytest.MonkeyPatch.context() as patch:
        _refuse_on(patch, ROSTER_A)
        evolve(_config(directory, ROSTER_A), proposer=_proposer(patch, GOOD_SPEC), **_quiet())
    return directory


def _copy(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(source, target)
    return target


class TestAPlayersRefusalIsMeasuredAgain:
    def test_it_is_stored_with_its_basis_and_skipped_while_that_holds(
        self, refused_run, tmp_path, monkeypatch
    ) -> None:
        directory = _copy(refused_run, tmp_path)
        first = Archive(directory)
        refused = first.entries[-1]
        assert refused.status == "evaluated" and not refused.viable
        assert refused.refusal is not None
        assert refused.refusal["rung"] == refusals.FULL
        assert refused.refusal["kind"] == refusals.PLAYERS
        assert refused.refusal["roster"] == list(ROSTER_A)
        [check] = refused.refusal["checks"]
        assert check["check"] == "skill_gradient" and check["kind"] == refusals.PLAYERS
        assert (check["value"], check["line"], check["side"]) == (3.1, 8.0, "min")
        assert check["margin"] == pytest.approx(4.9)
        assert not first.grid().elites

        # The same roster again: the basis has not moved, so its hash is skipped as before.
        again = evolve(
            _config(directory, ROSTER_A), proposer=_proposer(monkeypatch, GOOD_SPEC),
            **_quiet(),
        )
        assert [e.status for e in again.entries] == ["evaluated", "duplicate"]

    def test_under_another_roster_it_is_measured_again_and_can_become_an_elite(
        self, refused_run, tmp_path, monkeypatch
    ) -> None:
        directory = _copy(refused_run, tmp_path)
        [refused] = Archive(directory).entries
        lines: list[str] = []
        later = evolve(
            _config(directory, ROSTER_B), proposer=_proposer(monkeypatch, GOOD_SPEC),
            report=lines.append,
        )
        assert [e.status for e in later.entries] == ["evaluated", "evaluated"]
        measured = later.entries[-1]
        assert measured.hash == refused.hash
        assert measured.viable and measured.refusal is None
        assert [e.hash for e in later.grid().elites.values()] == [refused.hash]
        assert any("measuring it again" in line for line in lines)
        # The old refusal is history now: nothing shows it as a failure.
        assert later.superseded(later.entries[0])
        assert later.failures() == []

    def test_a_refusal_by_construction_is_never_measured_again(
        self, tmp_path, monkeypatch
    ) -> None:
        first = evolve(
            _config(tmp_path, ROSTER_A), proposer=_proposer(monkeypatch, STUCK), **_quiet()
        )
        [refused] = first.entries
        assert refused.status == "rejected"
        assert refused.refusal is not None
        assert refused.refusal["rung"] == refusals.SMOKE
        assert refused.refusal["kind"] == refusals.RULES
        assert [c["check"] for c in refused.refusal["checks"]] == ["opening_moves"]

        # Another roster, and the cheap check's lines moved as well: still final.
        monkeypatch.setattr(gates, "LINES", {**gates.LINES, "stall_rate": 0.99})
        later = evolve(
            _config(tmp_path, ROSTER_B), proposer=_proposer(monkeypatch, STUCK), **_quiet()
        )
        assert [e.status for e in later.entries] == ["rejected", "duplicate"]


class TestTheRuleOfARetry:
    """`_retry_rule` read on stored entries alone: no game is played."""

    @staticmethod
    def _entry(archive: Archive, refusal: dict[str, Any]) -> Entry:
        return archive.record(
            Entry(hash="abc", name="g", generation=1, parent="p",
                  status="rejected", refusal=refusal),
            spec=copy.deepcopy(GOOD_SPEC),
        )

    def _smoke(self, kind: str) -> dict[str, Any]:
        return {"rung": refusals.SMOKE, "kind": kind, "checks": [],
                "roster": list(orchestrator.SMOKE_ROSTER), "lines": dict(gates.LINES),
                "weights_version": "v1", "seed": 0}

    def test_a_cheap_rung_is_re_tried_when_its_line_moves_and_not_for_new_weights(
        self, tmp_path, monkeypatch
    ) -> None:
        archive = Archive(tmp_path)
        entry = self._entry(archive, self._smoke(refusals.PLAYERS))
        config = _config(tmp_path, ROSTER_B)
        # Another weights version: the cheap check reads none, and would replay the very
        # games that refused it.
        assert not orchestrator._retry_rule(archive, config, "v99")(entry)
        monkeypatch.setattr(gates, "LINES", {**gates.LINES, "stall_rate": 0.99})
        assert orchestrator._retry_rule(archive, config, "v99")(entry)

    def test_the_full_roster_is_re_tried_for_new_weights_or_players(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        refusal = refusals.full_refusal([PLAYERS_SAID], list(ROSTER_B), "v14", 0)
        entry = self._entry(archive, refusal)
        assert not orchestrator._retry_rule(archive, _config(tmp_path, ROSTER_B), "v14")(entry)
        assert orchestrator._retry_rule(archive, _config(tmp_path, ROSTER_B), "v15")(entry)
        assert orchestrator._retry_rule(archive, _config(tmp_path, ROSTER_A), "v14")(entry)

    def test_a_refusal_with_any_check_by_the_rules_stays_final(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        both = refusals.full_refusal(
            [PLAYERS_SAID, "mean_branching=1.2 below minimum 1.5"], list(ROSTER_B), "v1", 0
        )
        assert both["kind"] == refusals.BOTH
        entry = self._entry(archive, both)
        assert not orchestrator._retry_rule(archive, _config(tmp_path, ROSTER_A), "v99")(entry)

    def test_a_roster_that_cannot_read_the_game_is_the_players(self) -> None:
        unread = refusals.full_refusal(
            ["this roster cannot read this ruleset: the rating ladder orders the agents on "
             "only 0% of its budget steps, so skill_gradient would describe the players "
             "rather than the game"], list(ROSTER_B), "v1", 0)
        assert unread["kind"] == refusals.PLAYERS and unread["checks"] == []
        child = Entry(hash="h", name="n", generation=1, parent="p", status="evaluated",
                      refusal=unread)
        assert refusals.count([child]) == {
            (refusals.FULL, refusals.UNREAD_CHECK, refusals.PLAYERS): 1
        }


class TestOldArchives:
    OLD = {
        "hash": "0123456789ab", "name": "old", "generation": 2, "parent": "fedcba987654",
        "status": "rejected", "notes": "Rejected before the full roster: no computer player "
        "won any of 8 games: either nobody can win this game, or winning it takes a stronger "
        "player than these two small ones",
        "detail": {"smoke": {"passed": True, "reasons": [], "stats": {}},
                   "screen": {"passed": False, "reasons": [
                       "no computer player won any of 8 games: either nobody can win this "
                       "game, or winning it takes a stronger player than these two small ones"
                   ], "stats": {}}},
        "timestamp": 1.0,
    }

    def test_an_entry_without_a_refusal_loads_and_stays_final(self, tmp_path) -> None:
        (tmp_path / "lineage.jsonl").write_text(json.dumps(self.OLD) + "\n", encoding="utf-8")
        archive = Archive(tmp_path)
        [entry] = archive.entries
        assert entry.refusal is None
        assert entry.status == "rejected" and entry.name == "old"
        retry = orchestrator._retry_rule(archive, _config(tmp_path, ROSTER_A), "v99")
        assert archive.settled_hashes(retry) == {"0123456789ab"}
        # Read back from its sentences, for the count, and never with a basis to re-try on.
        found = refusals.refusal_of(entry)
        assert found is not None and found["reconstructed"]
        assert (found["rung"], found["kind"]) == (refusals.SCREEN, refusals.PLAYERS)
        assert [row["check"] for row in found["checks"]] == ["decisive"]
        assert not refusals.retriable(found, {"roster": []})
        assert refusals.count(archive.entries) == {
            (refusals.SCREEN, "decisive", refusals.PLAYERS): 1
        }

    def test_a_new_entry_round_trips(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        refusal = refusals.full_refusal([PLAYERS_SAID], list(ROSTER_B), "v14", 0)
        archive.record(Entry(hash="h", name="n", generation=1, parent="p",
                             status="evaluated", refusal=refusal))
        assert Archive(tmp_path).entries[0].refusal == refusal


class TestDeterminism:
    def test_a_run_with_a_refusal_is_a_function_of_its_seed_and_configuration(
        self, tmp_path, monkeypatch
    ) -> None:
        _refuse_on(monkeypatch, ROSTER_A)
        runs = []
        for name in ("one", "two"):
            archive = evolve(
                _config(tmp_path / name, ROSTER_A, generations=2),
                proposer=_proposer(monkeypatch, GOOD_SPEC, STUCK), **_quiet(),
            )
            runs.append([
                (e.hash, e.status, e.viable, e.index, e.refusal) for e in archive.entries
            ])
        assert runs[0] == runs[1]
        assert [row[1] for row in runs[0]] == ["evaluated", "rejected"]


class TestOneRuleForTheSiteAndTheLoop:
    def test_the_site_words_a_refusal_by_the_loops_rule(self) -> None:
        assert sitescore.smoke_kind is refusals.smoke_kind
        assert sitescore.RULES_GATES is refusals.RULES_GATES
        assert (sitescore.RULES, sitescore.PLAYERS) == (refusals.RULES, refusals.PLAYERS)

    def test_every_row_a_gate_writes_names_the_check_its_sentence_does(self) -> None:
        stuck = gates.smoke_test(STUCK, games=12, seed=0, workers=1)
        assert not stuck.passed
        for reason, row in zip(stuck.reasons, stuck.checks, strict=True):
            assert refusals.sentence_check(reason) == row["check"]
        sentences = {
            "a player has 1.20 moves to choose from on average; there is nothing to decide":
                ("mean_branching", refusals.RULES),
            f"99% of random games {gates.STALLED}: the pieces can't force a result":
                ("stall_rate", refusals.PLAYERS),
            "none of 40 random games had a winner; from this start, the way to win may be "
            "out of reach": ("decided", refusals.PLAYERS),
            "80% of games were over within 12 turns, which is a trap in the opening rather "
            "than a game": ("quick_kill_rate", refusals.PLAYERS),
        }
        for text, (check, kind) in sentences.items():
            assert (refusals.sentence_check(text), refusals.smoke_kind(text)) == (check, kind)
