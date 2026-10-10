"""Runs played elsewhere, and what it takes before this checkout claims them.

The campaign that played three more seeds of every library game left 213 run directories
behind, and a run directory holds exactly what a played one holds. The question these
tests settle is not whether the files can be copied -- they can -- but what entitles the
measurement written afterwards to stamp today's `engine`, `tables` and `analysis` over
games this checkout did not play.

The answer is a sample. Some of the games are played again here and compared move for
move, some of their analyses are graded again, and some of the blunder trials are drawn
again. A run that differs anywhere is not adopted, and the caller replays it -- so every
failure below has the same consequence and none of them needs a fallback of its own.

The tournaments are four games on a five-square board, which is where this file's `slow`
marks come from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox.tournament import adopt
from chess_variant_sandbox.tournament.analysis import analyse_corpus
from chess_variant_sandbox.tournament.record import read_records, write_records
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.slow

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

SEED = 1000

# Two searching players, so every game is one the analysis pass grades and every blunder
# trial has a game to be drawn from. Neither carries `fit=`, so nothing here needs a
# fitted store in place first.
#
# The minimax player spells `threats=0`, the search before Q502i's repair, as a small-sample
# fixture pinned to the old search does (`.claude/rules/agents-and-eval.md`). With the repair
# on, 50 nodes go to chasing threats and every one of the four games ends in three plies, so
# no game reaches the twelve a blunder trial is drawn from (`analysis._blunder_choices`) and
# the trial test below had nothing to test (red from Q502i until Q743). With it off, two of
# the four run 27 and 64 plies.
ROSTER = TournamentConfig(
    agents=("greedy", "minimax:nodes=50,threats=0"),
    games_per_pairing=1, seed=SEED, workers=1,
)


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    """A finished run in a campaign root, exactly as the real one left them."""
    source = tmp_path / "campaign" / f"deadbeef-{SEED}"
    tournament = run_tournament(SPEC, ROSTER)
    tournament.save(source)
    # Seed 0 for the pass, whatever the run's own seed is. Every trial is fixed by the
    # pass's seed and the record it is drawn from, so the trials differ between runs
    # because the records do.
    analyse_corpus(SPEC, tournament.records, seed=0, workers=1).save(
        source / "analysis.json"
    )
    return source


def _adopt(campaign: Path, target: Path) -> adopt.Adoption:
    return adopt.adopt_run(SPEC, SEED, campaign, target, ROSTER, workers=1)


def _bytes(directory: Path) -> dict[str, bytes]:
    return {name: (directory / name).read_bytes() for name in adopt.CARRIED}


class TestARunThatHolds:
    def test_it_is_copied_and_says_where_it_came_from(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "runs" / f"deadbeef-{SEED}"
        before = _bytes(campaign)

        taken = _adopt(campaign, target)

        assert taken.adopted and not taken.reason
        assert _bytes(target) == before
        # The origin is what the game's file records about this run, and it is the only
        # claim in that file that is not the game's own.
        assert "adopted from campaign" in taken.origin
        assert "re-derived" in taken.origin

    def test_the_campaign_is_never_written_to(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        before = _bytes(campaign)
        _adopt(campaign, tmp_path / "runs" / f"deadbeef-{SEED}")
        assert _bytes(campaign) == before

    def test_the_index_file_is_not_carried_over(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        """`metrics` moved after the campaign scored these runs, so its score is not a
        score this checkout would take. The caller re-scores every adopted run."""
        (campaign / "index-v10.json").write_text("{}", encoding="utf-8")
        target = tmp_path / "runs" / f"deadbeef-{SEED}"

        _adopt(campaign, target)

        assert not list(target.glob("index-*.json"))


class TestARunThatDoesNot:
    """Four refusals, and each one names what it saw.

    A refusal is never fatal. The seed keeps the replay it was going to have, which is
    why none of these has to decide anything beyond *not this*.
    """

    def test_a_game_that_replays_differently_here(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        records = list(read_records(campaign / "games.jsonl"))
        records[1].plies += 1
        write_records(records, campaign / "games.jsonl")

        taken = _adopt(campaign, tmp_path / "runs" / f"deadbeef-{SEED}")

        assert not taken.adopted
        assert "replays differently here" in taken.reason
        assert not (tmp_path / "runs" / f"deadbeef-{SEED}").exists()

    def test_an_analysis_entry_the_pass_would_not_write(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        stored = json.loads((campaign / "analysis.json").read_text(encoding="utf-8"))
        stored["games"][0]["moved_instances"] += 1
        (campaign / "analysis.json").write_text(json.dumps(stored), encoding="utf-8")

        taken = _adopt(campaign, tmp_path / "runs" / f"deadbeef-{SEED}")

        assert not taken.adopted
        assert "analyses differently here" in taken.reason

    def test_a_blunder_trial_the_sampler_would_not_draw(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        stored = json.loads((campaign / "analysis.json").read_text(encoding="utf-8"))
        assert stored["blunders"], "the fixture needs a trial for this to test anything"
        stored["blunders"][0]["ply"] += 1
        (campaign / "analysis.json").write_text(json.dumps(stored), encoding="utf-8")

        taken = _adopt(campaign, tmp_path / "runs" / f"deadbeef-{SEED}")

        assert not taken.adopted
        assert "blunder trial" in taken.reason

    def test_a_tournament_played_on_another_roster(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        stored = json.loads((campaign / "tournament.json").read_text(encoding="utf-8"))
        stored["config"]["agents"] = ["random", "greedy", "minimax:nodes=50"]
        (campaign / "tournament.json").write_text(json.dumps(stored), encoding="utf-8")

        taken = _adopt(campaign, tmp_path / "runs" / f"deadbeef-{SEED}")

        assert not taken.adopted
        assert "a different roster" in taken.reason

    def test_a_tournament_played_from_another_seed(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        stored = json.loads((campaign / "tournament.json").read_text(encoding="utf-8"))
        stored["config"]["seed"] = 0
        (campaign / "tournament.json").write_text(json.dumps(stored), encoding="utf-8")

        taken = _adopt(campaign, tmp_path / "runs" / f"deadbeef-{SEED}")

        assert not taken.adopted
        assert "from seed 0" in taken.reason

    def test_a_target_directory_that_already_holds_a_run(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        """Nothing is overwritten, and the run that was there is what is kept.

        A run directory is trusted by co-location, so replacing one under a measurement
        that already names it would change what a stored score was taken over without
        changing anything a reader could see.
        """
        target = tmp_path / "runs" / f"deadbeef-{SEED}"
        target.mkdir(parents=True)
        (target / "games.jsonl").write_text("mine\n", encoding="utf-8")

        taken = _adopt(campaign, target)

        assert not taken.adopted
        assert "already holds a run" in taken.reason
        assert (target / "games.jsonl").read_text(encoding="utf-8") == "mine\n"


class TestFindingACampaignsRun:
    def test_a_campaign_root_is_read_where_it_lies(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        root = campaign.parent
        assert adopt.campaign_run(root, "deadbeef", SEED) == campaign
        assert adopt.campaign_run(root, "deadbeef", 2000) is None
        assert adopt.campaign_run(root, "0123456789ab", SEED) is None
