"""The cheap gate and the design archive.

Neither needs a model, which is the point: the loop's cost control and its memory are
both testable offline, and only the proposal and judging stages require credentials.
"""

from __future__ import annotations

import copy
import json

import pytest

from chess_variant_sandbox.loop import Archive, Entry, smoke_test
from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR

PLAYABLE = {
    "schema_version": 1,
    "name": "playable",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "royal": True, "moves": [
            {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
        ]},
        {"id": "R", "value": 5, "moves": [
            {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
        ]},
    ],
    "setup": {"mirror": True, "placements": [
        {"piece": "K", "squares": ["c1"]},
        {"piece": "R", "squares": ["a1", "e1"]},
    ]},
    "rules": {"win": "royal_capture", "max_plies": 150, "no_progress_draw": 50},
}


class TestSmokeGate:
    def test_a_playable_ruleset_passes(self) -> None:
        result = smoke_test(PLAYABLE, games=40)
        assert result.passed, result.reasons
        assert result.stats["mean_branching"] > 1.5

    def test_a_ruleset_that_does_not_compile_is_rejected_not_raised(self) -> None:
        """The loop must survive a bad proposal; an exception here would end the run."""
        broken = copy.deepcopy(PLAYABLE)
        broken["pieces"][0]["moves"] = [{"kind": "teleport", "vectors": [[1, 1]]}]
        result = smoke_test(broken, games=10)
        assert not result.passed
        assert result.reasons[0].startswith("the rules don't fit together: ")

    def test_a_position_with_no_opening_move_is_rejected(self) -> None:
        """Cheapest possible rejection, and it costs nothing to check first."""
        stuck = copy.deepcopy(PLAYABLE)
        stuck["board"] = {"width": 2, "height": 2}
        stuck["pieces"][0]["moves"] = [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "min_range": 1, "mode": "capture"}
        ]
        stuck["pieces"][1]["moves"] = [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "min_range": 1, "mode": "capture"}
        ]
        stuck["setup"] = {"mirror": True, "placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["b1"]}
        ]}
        result = smoke_test(stuck, games=5)
        assert not result.passed

    def test_a_game_that_never_ends_is_rejected(self) -> None:
        endless = copy.deepcopy(PLAYABLE)
        endless["pieces"] = [
            {"id": "K", "royal": True, "moves": [
                {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"}
            ]}
        ]
        endless["setup"] = {"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]}]}
        endless["rules"] = {"win": "royal_capture", "max_plies": 60, "no_progress_draw": 60}
        result = smoke_test(endless, games=30)
        assert not result.passed
        assert any("the way to win may be out of reach" in reason for reason in result.reasons)

    def test_a_draw_heavy_ruleset_is_not_rejected_for_seat_skew(self) -> None:
        """Regression: the skew check once fired off a single decisive game in sixty.

        A game can be legitimately draw-heavy under random play. Judging seat balance
        from one result rejects sound designs for a coin flip.
        """
        drawish = copy.deepcopy(PLAYABLE)
        drawish["rules"] = {"win": "royal_capture", "max_plies": 40, "no_progress_draw": 12}
        result = smoke_test(drawish, games=60)
        assert not any("lopsided" in reason for reason in result.reasons), result.reasons

    def test_a_sample_the_clock_cut_short_is_refused_not_graded(self, monkeypatch) -> None:
        """Refuse-or-whole: the verdict never depends on how many games beat the clock.

        The clock is injected, never timed: the core's `random_playouts` is wrapped so its
        budget "expires" after `finished` of the asked games, the way a slow machine's
        would. Grading those would make a pass a function of machine speed.
        """
        from chess_variant_sandbox.engine import Engine
        from chess_variant_sandbox.loop import gates

        games, finished = 40, 25

        class ClockCutCore:
            def __init__(self, core: object) -> None:
                self._core = core

            def __getattr__(self, name: str) -> object:
                return getattr(self._core, name)

            def random_playouts(self, count, seed, _budget, threads):  # type: ignore[no-untyped-def]
                whole = self._core.random_playouts(count, seed, float("inf"), threads)
                return whole[:finished]

        def cut_engine(spec):  # type: ignore[no-untyped-def]
            engine = Engine(spec)
            engine.core = ClockCutCore(engine.core)
            return engine

        assert smoke_test(PLAYABLE, games=games).passed  # the whole sample passes
        monkeypatch.setattr(gates, "Engine", cut_engine)
        result = smoke_test(PLAYABLE, games=games, time_budget=5.0)
        assert not result.passed
        assert result.reasons == [
            "random games of these rules take too long to finish, so they couldn't be "
            f"checked in the 5 seconds allowed (only {finished} of {games} finished)"
        ]
        assert "stall_rate" not in result.stats  # nothing was graded

    def test_the_gate_honours_its_time_budget(self) -> None:
        """A pathological proposal must not be able to stall the whole run."""
        import time

        started = time.perf_counter()
        smoke_test(PLAYABLE, games=100_000, time_budget=1.0)
        assert time.perf_counter() - started < 12.0


def _in_rounds(spec: dict) -> dict:
    """`spec` played in rounds (schema 26, `rules.simultaneous`)."""
    rounds = copy.deepcopy(spec)
    rounds["schema_version"] = 26
    rounds["rules"]["simultaneous"] = {}
    return rounds


class TestTheGatesPlayRounds:
    """Q932: the cheap rungs play a game in rounds, on the same criteria as any game."""

    def test_a_playable_game_in_rounds_passes(self) -> None:
        result = smoke_test(_in_rounds(PLAYABLE), games=40)
        assert result.passed, result.reasons
        assert result.stats["games"] == 40 and result.stats["mean_branching"] > 1.5

    def test_two_lone_kings_in_rounds_fail_for_never_having_a_winner(self) -> None:
        """Every meeting of two kings that capture each other is a tie that removes both,
        so no random game is ever won: the ordinary no-winner criterion, after playouts."""
        lone = _in_rounds(PLAYABLE)
        lone["pieces"] = lone["pieces"][:1]
        lone["setup"]["placements"] = lone["setup"]["placements"][:1]
        result = smoke_test(lone, games=40)
        assert not result.passed
        assert result.stats["games"] == 40
        assert result.reasons == [
            "none of 40 random games had a winner; from this start, the way to win may be "
            "out of reach"
        ]

    def test_a_game_in_rounds_with_no_first_move_fails_before_playing(self) -> None:
        stuck = _in_rounds(PLAYABLE)
        for piece in stuck["pieces"]:
            piece["moves"] = [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "min_range": 1, "mode": "capture"}
            ]
        result = smoke_test(stuck, games=5)
        assert not result.passed
        assert result.reasons == ["a player has no legal move from the starting position"]

    def test_the_round_roster_swaps_minimax_for_matrix_at_the_same_budgets(self) -> None:
        from chess_variant_sandbox.loop.orchestrator import DEFAULT_AGENTS
        from chess_variant_sandbox.loop.screen import SCREEN_AGENTS, roster_for, round_roster

        assert round_roster(SCREEN_AGENTS) == ("matrix:nodes=100", "matrix:nodes=400")
        assert round_roster(DEFAULT_AGENTS) == (
            "random", "greedy", "matrix:nodes=400", "matrix:nodes=1600"
        )
        assert roster_for(PLAYABLE, DEFAULT_AGENTS) == DEFAULT_AGENTS
        for refused in ("mcts:nodes=400", "minimax:h=abc,nodes=400"):
            with pytest.raises(ValueError, match="cannot play a game in rounds"):
                round_roster((refused,))

    def test_the_screen_plays_a_round_game_and_refuses_a_roster_with_no_round_player(
        self,
    ) -> None:
        from pathlib import Path

        from chess_variant_sandbox.loop.screen import screen
        from chess_variant_sandbox.spec.canonical import load_spec

        hole = load_spec(Path(__file__).parent.parent / "archive" / "watering-hole.json")
        played = screen(hole, workers=1)
        assert played.passed, played.reasons
        assert played.stats["games"] == 8 and played.stats["decisive"] >= 1

        refused = screen(hole, agents=("mcts:nodes=100", "mcts:nodes=400"), workers=1)
        assert not refused.passed and "cannot play a game in rounds" in refused.reasons[0]


class TestArchive:
    def test_entries_survive_a_restart(self, tmp_path) -> None:
        first = Archive(tmp_path)
        first.record(Entry(hash="aaa", name="one", generation=0, parent=None, status="evaluated",
                           index=0.5, objective=0.5, viable=True), spec=PLAYABLE)
        second = Archive(tmp_path)
        assert [e.hash for e in second.entries] == ["aaa"]
        assert second.load_spec("aaa") == PLAYABLE
        assert second.next_generation() == 1

    def test_unviable_candidates_never_reach_the_leaderboard(self, tmp_path) -> None:
        """A gate failure is not a low score; building on one wastes the next generation."""
        archive = Archive(tmp_path)
        archive.record(Entry(hash="good", name="g", generation=0, parent=None,
                             status="evaluated", index=0.4, objective=0.4, viable=True))
        archive.record(Entry(hash="broken", name="b", generation=1, parent=None,
                             status="evaluated", index=0.9, objective=0.9, viable=False))
        assert [e.hash for e in archive.leaderboard()] == ["good"]

    def test_the_leaderboard_is_ordered_by_index(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        for digest, index in (("a", 0.3), ("b", 0.8), ("c", 0.55)):
            archive.record(Entry(hash=digest, name=digest, generation=0, parent=None,
                                 status="evaluated", index=index, objective=index, viable=True))
        assert [e.hash for e in archive.leaderboard()] == ["b", "c", "a"]

    def test_seen_hashes_covers_rejects_so_they_are_not_retried(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        archive.record(Entry(hash="dud", name="d", generation=0, parent=None, status="rejected"))
        assert "dud" in archive.seen_hashes()

    def test_failures_are_kept_for_the_next_prompt(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        archive.record(Entry(hash="x", name="x", generation=0, parent=None,
                             status="rejected", notes="locks up immediately"))
        assert archive.failures()[0].notes == "locks up immediately"

    def test_a_corrupt_line_does_not_lose_the_rest_of_the_history(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        archive.record(Entry(hash="a", name="a", generation=0, parent=None, status="evaluated",
                             index=0.5, viable=True))
        with open(archive.path, "a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        assert len(Archive(tmp_path).entries) == 1

    def test_a_restatement_changes_the_words_and_nothing_else(self, tmp_path) -> None:
        """Q595: a saved draft's new name and description, in place. The entries stay
        where they were, other hashes' lines are untouched, and a corrupt line is written
        back as it was read rather than lost."""
        archive = Archive(tmp_path)
        spec = {**PLAYABLE, "name": "old", "description": ""}
        archive.record(Entry(hash="a", name="old", generation=0, parent=None,
                             status="proposed"), spec=spec)
        archive.record(Entry(hash="b", name="other", generation=0, parent=None,
                             status="proposed"))
        with open(archive.path, "a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        before = archive.path.read_text(encoding="utf-8").splitlines()

        archive.restate("a", "new", "now it has one")

        after = archive.path.read_text(encoding="utf-8").splitlines()
        assert after[1:] == before[1:]
        assert json.loads(after[0])["name"] == "new"
        reread = Archive(tmp_path)
        assert [(e.hash, e.name) for e in reread.entries] == [("a", "new"), ("b", "other")]
        assert [(e.hash, e.name) for e in archive.entries] == [("a", "new"), ("b", "other")]
        kept = reread.load_spec("a")
        assert kept == {**PLAYABLE, "name": "new", "description": "now it has one"}


class TestEveryNumberSaysItsRuns:
    """Q416: the design archive's numbers name the runs they rest on, or are not shown.

    `CLAUDE.md`: a score carries its basis, and every number says which. A design run's
    number is one run on its one seed, and the entry says so in `seeds`; an entry written
    before the field existed says nothing, and is shown as *no score* rather than with a
    count somebody guessed.
    """

    def test_the_seeds_survive_a_restart(self, tmp_path) -> None:
        Archive(tmp_path).record(Entry(hash="a", name="a", generation=0, parent=None,
                                       status="evaluated", index=0.5, viable=True,
                                       seeds=[7]))
        entry = Archive(tmp_path).entries[0]
        assert entry.seeds == [7]
        assert entry.runs == 1

    def test_an_entry_written_before_the_field_has_no_count(self, tmp_path) -> None:
        """The shape every archive on disk had before Q416: no `seeds` key at all."""
        Archive(tmp_path)
        (tmp_path / "lineage.jsonl").write_text(json.dumps({
            "hash": "old", "name": "old", "generation": 0, "parent": None,
            "status": "evaluated", "index": 0.9, "objective": 0.9, "viable": True,
        }) + "\n", encoding="utf-8")
        entry = Archive(tmp_path).entries[0]
        assert entry.seeds == [] and entry.runs == 0
        assert entry.scored_as() == "no score"

    def test_the_board_says_the_one_run_in_the_pages_words(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        archive.record(Entry(hash="a", name="alpha", generation=0, parent=None,
                             status="evaluated", index=0.5, index_sd=0.02, objective=0.5,
                             viable=True, seeds=[0]))
        board = archive.summary()
        assert "1. alpha [a] index 0.500 +/-0.020 as the one run (objective 0.500)" in board
        assert "no score" not in board

    def test_a_mean_is_said_as_one(self) -> None:
        entry = Entry(hash="m", name="m", generation=0, parent=None, status="evaluated",
                      index=0.5, viable=True, seeds=[0, 1000, 2000, 3000])
        assert entry.scored_as() == "index 0.500 as the mean of four runs"

    def test_an_entry_with_no_count_shows_no_number(self, tmp_path) -> None:
        archive = Archive(tmp_path)
        archive.record(Entry(hash="new", name="new", generation=0, parent=None,
                             status="evaluated", index=0.4, objective=0.4, viable=True,
                             seeds=[0]))
        archive.record(Entry(hash="old", name="old", generation=0, parent=None,
                             status="evaluated", index=0.9, index_sd=0.03, objective=0.8,
                             judge=0.7, viable=True))
        board = archive.summary()
        # Still ranked where it was, since ranking is the loop's and unchanged -- but its
        # number, its spread and its objective are not printed without a count.
        assert "1. old [old] no score (judge 0.70)" in board
        assert "0.900" not in board and "0.030" not in board and "0.800" not in board
        assert "2. new [new] index 0.400 as the one run (objective 0.400)" in board
        assert "written before the archive recorded how many runs" in board

    def test_the_grid_says_the_count_as_the_board_does(self, tmp_path) -> None:
        """`chess-variant-sandbox leaderboard` prints the grid's elites too (`v11` C3)."""
        archive = Archive(tmp_path)
        archive.record(Entry(hash="a", name="alpha", generation=0, parent=None,
                             status="evaluated", index=0.5, index_sd=0.02, viable=True,
                             seeds=[0], cell="1-1-0-0-1-0-0"))
        archive.record(Entry(hash="o", name="older", generation=0, parent=None,
                             status="evaluated", index=0.9, viable=True,
                             cell="2-1-0-0-1-0-0"))
        said = archive.grid().summary()
        assert "alpha  index 0.500 +/-0.020 as the one run" in said
        assert "older  no score" in said and "0.900" not in said

    def test_the_command_prints_the_count(self, tmp_path, capsys) -> None:
        from chess_variant_sandbox import cli

        Archive(tmp_path).record(Entry(hash="a", name="alpha", generation=0, parent=None,
                                       status="evaluated", index=0.5, objective=0.5,
                                       viable=True, seeds=[0]))
        assert cli.main(["leaderboard", "--out", str(tmp_path)]) == 0
        assert "alpha [a] index 0.500 as the one run" in capsys.readouterr().out


def test_feedback_never_leaks_the_scoring_function() -> None:
    """The guarantee the whole proposer design rests on.

    If component names, scores or weights reach the proposing model, it optimises the
    proxy instead of the game -- see llm/feedback.py.
    """
    from chess_variant_sandbox.llm.feedback import describe
    from chess_variant_sandbox.metrics import load_weights, score_components
    from chess_variant_sandbox.metrics.components import Components

    # Checked against every shipped weights version, not just the current one. A new
    # component arrives with a new prose rule, and the rule is written by someone
    # looking at the component name -- which is exactly when the name leaks.
    for version in ("v2", "v3"):
        weights = load_weights(version)
        raw = {name: 0.5 for name in weights.components()}
        raw.update({"mean_plies": 40, "mean_branching": 20, "decisive_rate": 0.6,
                    "skill_gradient": 5, "search_advantage": 0.55, "rule_mass": 200})
        result = score_components(
            "h", "g", Components(raw=raw, diagnostics={"games": 10}), weights
        )

        # Every diagnostic the loop forwards, not just the ones an empty run produces.
        # Widening the channel is what makes this test matter more, not less.
        diagnostics = {
            "games": 144,
            "outcome_reasons": {"checkmate": 90, "no progress": 54},
            "decisive_by_tier": {"low_decisive": 0.71, "high_decisive": 0.42},
            "moves_by_type": [3120, 880, 61],
            "idle_piece_types": 1,
            "criticality_samples": 512,
            "blunder_trials": 16,
            "blunder_changed_result": 0.31,
        }
        spec = {"pieces": [{"id": "K", "name": "captain"}, {"id": "R", "name": "lancer"},
                           {"id": "E", "name": "elephant"}],
                "setup": {"placements": [{"piece": p, "squares": ["a1"]} for p in "KRE"]}}
        result = score_components(
            "h", "g", Components(raw=raw, diagnostics=diagnostics), weights
        )

        text = describe(result, spec).lower()
        for component in weights.components():
            assert component.replace("_", " ") not in text
            assert component not in text
        for term in ("index", "weight", "score", "0.5", "component"):
            assert term not in text

        # ...and the observations really are there, or the test above passes vacuously.
        assert "elephant" in text
        assert "barely used" in text
        assert "between the weaker players" in text


# The rule stage 12 adopts, and the form it survived contact with the corpus in.
#
# The plan states it as "a component that is scored must not have a matching sentence in
# the brief". Applied literally that deletes five of the brief's seven bullets, and the
# measurement is what showed it: the brief and the component set were written from the
# same list of what makes a game good, so *topical* overlap is not a leak -- it is the
# reason the index is an index of quality rather than of something else. A brief that
# may not mention depth, balance, variety or pacing is not a brief.
#
# What separates a leak from a design goal is two things.
#
# **A brief is constant across candidates and therefore carries no gradient.** Every
# proposal sees the same words, so nothing in them says which way to move *this* game.
# `feedback.py` is the opposite -- it describes one candidate's measurements -- which is
# why the older test guards that channel far more tightly, and why it should.
#
# **A quality is not a measurement.** "Two strong players should not play the same game
# twice" is a quality: no procedure, no threshold, nothing to aim at but the game. "The
# number of options per turn into a comfortable range" states a *band*, which is the
# shape of the curve, which is the part of an objective function that can be gamed. So
# the enforced rule is: a scored component's target value, band or procedure must not
# appear in the brief. Its subject may.
#
# Every scored component must appear here, so that adding one to a weights file forces
# whoever adds it to write down what its measurement would sound like in English before
# the test can pass.
_OPERATIONAL: dict[str, tuple[str, ...]] = {
    "skill_gradient": ("elo per doubling", "rating rises with the node budget"),
    "gradient_persistence": ("still rises at the top of the ladder",),
    "decision_criticality": ("share of the legal moves", "how many moves are within"),
    "blunder_recoverability": ("share of forced mistakes", "band around half"),
    "draw_slope": ("decisive rate rises with strength",),
    "stall_rate": ("share of games ending on the ply limit", "no-progress clock"),
    "first_player_advantage": ("first seat should score", "self-play score near"),
    "opening_diversity": ("entropy over opening", "distinct opening prefixes"),
    "distinct_game_rate": ("share of games with a distinct move list",),
    "mean_branching": ("into a comfortable range", "average legal move count"),
    "ply_spread": ("coefficient of variation", "spread of game lengths relative"),
    "rule_yield": ("rating spread divided by", "per unit of rule mass"),
    "piece_utilisation": ("entropy over piece types", "share of moves by each piece"),
    # Arrived in v5, and its sentence left the brief in the same change. This is the
    # rule's first live application: it was written down while the component was
    # measured-but-unweighted, and it fired the moment that stopped being true.
    "lead_conversion": (
        "a small advantage should be convertible into a win by someone who plays well",
        "make an advantage something a good player can convert",
    ),
    "progress_irreversibility": ("plies that cannot be undone", "share of irreversible"),
    "instance_participation": ("share of pieces that ever moved",),
    # Weighted from v11 (docs/archive/plan-variety.md §5.4, C2). The brief's *Variety* bullet asks
    # that two strong players not play the same game twice, which is the family's question
    # and not this measurement; these are what the measurement sounds like.
    "position_divergence": (
        "positions in common with",
        "nearest other game",
        "pass through different positions",
        "share of positions",
    ),
}

# Measured but unweighted, so the brief may still ask for these. Empty at the moment,
# and kept rather than deleted because the table is the mechanism: a candidate component
# goes in here while it is being measured, and moving it to `_OPERATIONAL` is what
# forces whoever weights it to go and look at the brief.
_ALLOWED_WHILE_UNSCORED: dict[str, tuple[str, ...]] = {}


# Weights versions still in use, read off the weights directory rather than listed here.
# v1 and v2 are kept so their recorded scores stay reproducible; their component sets are
# history. A hand-maintained tuple went stale the moment `v9` was cut -- silently, and in
# the direction that exempts the *default* version from every invariant below, which is
# the one version that must not be exempt.
_HISTORICAL_VERSIONS = ("v1", "v2")
_LIVE_VERSIONS = tuple(
    sorted(
        (p.stem for p in WEIGHTS_DIR.glob("*.json") if p.stem not in _HISTORICAL_VERSIONS),
        key=lambda name: int(name.lstrip("v")),
    )
)


def _scored_now() -> set[str]:
    from chess_variant_sandbox.metrics import load_weights

    return {
        name
        for version in _LIVE_VERSIONS
        for name in load_weights(version).components()
    }


def _what_the_proposer_is_shown() -> str:
    from chess_variant_sandbox.llm.propose import DESIGN_BRIEF
    from chess_variant_sandbox.loop.orchestrator import _FOCUS

    return (DESIGN_BRIEF + " " + " ".join(_FOCUS.values())).lower()


def test_the_design_brief_does_not_state_a_scored_measurement() -> None:
    """Names are not the only channel a scoring function leaks through.

    Show a model the objective and it optimises the proxy, and it does not need the
    variable name to do that -- a sentence naming the target value is a *better*
    specification of the proxy than the name would be, because it is actionable
    without any further guessing.
    """

    shown = _what_the_proposer_is_shown()
    for name in sorted(_scored_now()):
        assert name in _OPERATIONAL, (
            f"{name} is scored and has no operational statement listed. Write what its "
            "measurement sounds like in English, then check the brief for it."
        )
        for phrase in _OPERATIONAL[name]:
            assert phrase not in shown, f"{name}'s measurement is in the brief: {phrase!r}"


@pytest.mark.skipif(
    not _ALLOWED_WHILE_UNSCORED,
    reason="no unscored component is asked for in the brief today (_ALLOWED_WHILE_UNSCORED "
    "is empty); this bites when a row is added",
)
def test_an_unscored_components_sentence_is_still_in_the_brief() -> None:
    """The other half of the rule, asserted so it cannot rot into a silent leak.

    A component that is measured and weighted by nothing may be asked for in prose:
    asking a designer for a quality is not showing them the scoring function. This test
    fails the moment that stops being true, which is exactly the moment the sentence has
    to come out -- and it did, for `lead_conversion`, when v5 scored it.
    """
    scored = _scored_now()
    shown = _what_the_proposer_is_shown()
    for name, phrases in _ALLOWED_WHILE_UNSCORED.items():
        assert name not in scored, (
            f"{name} is now scored, so its sentence must leave the brief: {phrases}"
        )
        assert any(phrase in shown for phrase in phrases), (
            f"{name}'s sentence is no longer in the brief; move it into _OPERATIONAL "
            "or drop this row"
        )


def test_the_preamble_names_every_primitive_and_every_rule_a_player_is_taught() -> None:
    """The fourth silent surface a schema bump has to reach, and it had gone two stale.

    The other three are known: the engine (a perft moves), `spec/describe.py` (the judge
    is told about a different game) and `spec/jsonschema.py` (the rule cannot be proposed
    at all). This one used to be the quietest, because a response-format grammar was
    holding the reply to the schema whatever the prose said. It is not any more: a rule
    spec is too big for the grammar compiler, so nothing on the wire constrains a
    proposal and the prompt is the whole contract. The preamble is the layer of it that
    says what a feature is *for*, and a list headed "movement primitives" that names
    three of the four is read as the four. `lame` had never been in it, and `rules.wrap`
    was absent entirely: five committed rulesets, one of them the highest-scoring game in
    the corpus, expressed in a rule the proposer was never told exists.

    Held to `complexity._RULE_COSTS` for the reason `describe.py` is: rule mass already
    decides which keys are exceptions a player has to be *taught*, so the honest question
    is whether the same keys are the ones the proposer is taught. The three costed at
    zero are bookkeeping every game needs and nobody learns.

    The key has to appear in code voice rather than as English -- `rules.win` counts and
    the word "wins" does not -- or half of this passes vacuously on prose that happens to
    use the same nouns.

    This deliberately does *not* extend `_what_the_proposer_is_shown`. The preamble
    shares phrases with `_OPERATIONAL` ("the no-progress clock") because those are the
    schema's own key names, and `rules.scoring` cannot be explained without naming when
    it fires. A mechanical reference in the DSL's vocabulary is not a statement of what
    makes a game good; the brief is, and that is the text the leak rule guards.
    """
    from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE
    from chess_variant_sandbox.spec.complexity import _RULE_COSTS
    from chess_variant_sandbox.spec.schema import MOVE_KINDS

    for kind in MOVE_KINDS:
        assert f"**{kind}**" in SCHEMA_PREAMBLE, (
            f"the {kind!r} move kind is a movement primitive the proposer is not shown"
        )

    for key, cost in _RULE_COSTS.items():
        if not cost:
            continue
        assert f"`{key}" in SCHEMA_PREAMBLE or f"`rules.{key}" in SCHEMA_PREAMBLE, (
            f"rules.{key} is charged as a special rule and the proposer is never told "
            "what it is for"
        )


def test_the_brief_does_not_forbid_a_game_the_corpus_already_scores() -> None:
    """A design opinion may narrow the search; it may not close a door the DSL opens.

    The brief used to say asymmetry between the armies "is not allowed: use the mirrored
    setup", which is a rule about the *engine* stated in the brief, and it was false. The
    DSL says asymmetry in five places, and two of the corpus's better games are reachable
    only through them -- `horde`, a scored band member, and `onewaycylinder`, whose two
    players cross one seam in opposite directions and which outscores chess.

    Asserted against the committed rulesets rather than against the sentence, so that it
    keeps meaning something if the wording changes: every key that makes a game
    asymmetric has a ruleset using it, and the proposer is told about each one.
    """
    import json
    from pathlib import Path

    from chess_variant_sandbox.llm.propose import DESIGN_BRIEF, SCHEMA_PREAMBLE

    rulesets = Path(__file__).resolve().parents[1] / "rulesets"
    specs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(rulesets.glob("*.json"))
    }

    # Separately-stated armies, and a per-player seam. One committed ruleset each is
    # enough: the claim is that the door is open, not that everything walks through it.
    assert specs["horde"]["setup"]["mirror"] is False
    assert "files_by_player" in specs["onewaycylinder"]["rules"]["wrap"]

    for key in ("setup.mirror", "to_by_player", "files_by_player", "ranks_by_player",
                "reach.mirror"):
        assert key in SCHEMA_PREAMBLE, f"{key} states an asymmetry and is never mentioned"

    # And the brief must not take back in prose what the schema offers. A narrow phrase
    # guard, in the same shape and for the same reason as the judge's "0 to 1": the
    # wording will change and the prohibition must not come back with it.
    lowered = DESIGN_BRIEF.lower()
    assert "asymmetry" not in lowered or "not allowed" not in lowered
    assert "must be symmetric" not in lowered
    assert "use the mirrored setup" not in lowered


def test_the_written_verdict_carries_no_number() -> None:
    """The judge produces prose now, and only prose.

    It used to produce a 0-1 score as well, blended into the index at ``llm_weight``.
    ``v4`` zeroed that weight and stage 7 replaced the blend with a pairwise ordering,
    which left a scoring path nothing called -- and a scoring path nothing calls is how
    a second ranker returns quietly. If a number ever comes back into this schema it has
    exactly one place to go: back into the ranking, next to an ordering derived from the
    same opinion, counted twice.
    """
    from dataclasses import fields

    from chess_variant_sandbox.llm.judge import JUDGE_BRIEF, VERDICT_SCHEMA, Judgement

    numeric = {"number", "integer"}
    for name, schema in VERDICT_SCHEMA["properties"].items():
        assert schema.get("type") not in numeric, name
    assert VERDICT_SCHEMA["additionalProperties"] is False

    for field in fields(Judgement):
        assert field.type not in ("float", "int"), field.name

    # And the brief must not ask for one either, or the model will hedge into a scale
    # in the prose and the next reader will pull it back out.
    lowered = JUDGE_BRIEF.lower()
    assert "0 to 1" not in lowered
    assert "score on a" not in lowered


def test_feedback_names_the_pieces_nobody_played() -> None:
    """The most common failure of an invented ruleset, and nothing else reports it.

    A piece can be added, validated, placed, and never be worth moving. The count is in
    the diagnostics already; what was missing was saying it in the language of the game.
    """
    from chess_variant_sandbox.llm.feedback import describe
    from chess_variant_sandbox.metrics import load_weights, score_components
    from chess_variant_sandbox.metrics.components import Components

    weights = load_weights("v3")
    raw = {name: 0.5 for name in weights.components()}
    result = score_components(
        "h", "g",
        Components(
            raw=raw,
            diagnostics={"games": 100, "moves_by_type": [900, 900, 12]},
        ),
        weights,
    )
    spec = {"pieces": [{"id": "K", "name": "king"}, {"id": "R"}, {"id": "H", "name": "herald"}],
            "setup": {"placements": [{"piece": p, "squares": ["a1"]} for p in "KRH"]}}
    text = describe(result, spec)

    assert "herald" in text
    assert "Barely used: herald." in text
    # A piece with no name at all is still named by its letter rather than by an index.
    assert " R " in text or "R " in text


def test_failed_siblings_reach_the_proposer(tmp_path) -> None:
    """The largest unused signal in the archive: what was already tried from here.

    ``avoid`` is a list of hashes -- it says that something was tried and nothing about
    what happened. A revision that cannot see why its siblings failed rediscovers them.
    """
    from chess_variant_sandbox.llm.feedback import describe_failures

    archive = Archive(tmp_path)
    archive.record(Entry(hash="p", name="parent", generation=0, parent=None,
                         status="evaluated", index=0.6, objective=0.6, viable=True))
    archive.record(Entry(hash="c1", name="wide open", generation=1, parent="p",
                         status="rejected", notes="Rejected before the tournament: "
                         "no game reached a winner"))
    archive.record(Entry(hash="c2", name="fortress", generation=2, parent="p",
                         status="evaluated", index=0.4, objective=0.4, viable=False,
                         notes="Games almost always ended level."))
    archive.record(Entry(hash="c3", name="elsewhere", generation=3, parent="other",
                         status="rejected", notes="not a sibling"))

    siblings = archive.failed_children("p")
    assert [e.hash for e in siblings] == ["c1", "c2"]

    text = describe_failures(siblings)
    assert "wide open" in text and "no game reached a winner" in text
    # A candidate that was played and then disqualified is a failure too, and the most
    # informative of the three kinds.
    assert "fortress" in text and "played and disqualified" in text
    assert "elsewhere" not in text

    # And the sentence reaches the model: `_produce` is what hands `failures` to the
    # proposer, so ask the prompt it sends rather than the text it was given.
    import random

    from chess_variant_sandbox.loop.orchestrator import LoopConfig, _produce

    class Recorder:
        model = "claude-opus-4-8"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def structured(self, system, user, schema, max_tokens=16000):
            self.prompts.append(user)
            return copy.deepcopy(PLAYABLE)

    recorder = Recorder()
    _produce(
        1, LoopConfig(mechanical_rate=0.0, directory=tmp_path), archive, random.Random(0),
        recorder, archive.entries[0], PLAYABLE, "", "", lambda _: None,
    )
    prompt = recorder.prompts[0]
    assert "Already tried from here" in prompt
    assert "wide open" in prompt and "no game reached a winner" in prompt
    assert "fortress" in prompt and "played and disqualified" in prompt
    assert "elsewhere" not in prompt


# --------------------------------------------------------- what the API will accept

def _every_schema_this_project_sends() -> dict[str, dict]:
    """Every response schema that reaches `Claude.structured`.

    Named rather than discovered, and the list is the test: a fifth LLM stage that adds
    a schema and forgets this line gets no protection, which is the same shape of gap
    `HEURISTIC_COMPONENTS` closes for the glossary.
    """
    from chess_variant_sandbox.llm.compare import COMPARISON_SCHEMA
    from chess_variant_sandbox.llm.heuristics import DEFAULT_K, heuristic_schema
    from chess_variant_sandbox.llm.judge import VERDICT_SCHEMA
    from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

    return {
        "rule_spec_schema": rule_spec_schema(),
        "heuristic_schema": heuristic_schema(DEFAULT_K),
        "VERDICT_SCHEMA": VERDICT_SCHEMA,
        "COMPARISON_SCHEMA": COMPARISON_SCHEMA,
    }


def _refused(node, path="$"):
    """Every constraint in ``node`` that the API answers 400 to.

    Walks the schema as a *schema*, not as a dict: `rule_spec_schema` has properties
    literally named ``0`` and ``1``, and a naive walk would read a property called
    ``minimum`` as the keyword ``minimum``.

    Three of the checks are on the *shape* of a node rather than on a keyword it
    carries, and they are the half a keyword list cannot see: a type union, a map whose
    keys the schema does not fix, and an object that never says it is closed. All three
    are in the published subset as unsupported, and `rule_spec_schema` produced two of
    them for its whole life.
    """
    from chess_variant_sandbox.llm.client import REFUSED_KEYWORDS, SCHEMA_MAPS, SCHEMA_VALUES

    found = []
    if isinstance(node, list):
        for index, item in enumerate(node):
            found += _refused(item, f"{path}[{index}]")
        return found
    if not isinstance(node, dict):
        return found
    for key, value in node.items():
        if key in REFUSED_KEYWORDS:
            found.append((path, key, value))
        elif key == "minItems" and value not in (0, 1):
            found.append((path, "minItems", value))
        elif key in SCHEMA_MAPS and isinstance(value, dict):
            for name, sub in value.items():
                found += _refused(sub, f"{path}.{key}.{name}")
        elif key in SCHEMA_VALUES:
            found += _refused(value, f"{path}.{key}")

    if isinstance(node.get("type"), list):
        found.append((path, "type union", node["type"]))
    if node.get("type") == "object":
        extra = node.get("additionalProperties", "<absent>")
        if extra is not False:
            found.append((path, "additionalProperties", "a schema" if isinstance(extra, dict) else extra))
    return found


def test_no_schema_this_project_sends_carries_a_constraint_the_api_refuses() -> None:
    """The invariant stage 23 was built to close, and the one that would have caught it.

    `spec/jsonschema.py` exists because a rule the generated schema cannot express
    cannot be proposed at all. This is the same failure one level further out: three of
    the four schemas below carried `minimum`, `maxItems` or `minItems > 1`, the API
    answers 400 to every one of those, and so *no LLM stage in this project had ever
    made a successful call* -- while every offline test passed, because a stub client
    never sends anything.

    Asserted on the wire form rather than on the generated form on purpose. A generator
    should state the constraint its validator implies; knowing which of those survive
    the request is `wire_schema`'s job and nowhere else's.
    """
    from chess_variant_sandbox.llm.client import wire_schema

    for name, schema in _every_schema_this_project_sends().items():
        remaining = _refused(wire_schema(schema))
        assert not remaining, f"{name} still carries {remaining}"


def test_a_dropped_constraint_is_restated_rather_than_lost() -> None:
    """A model not told how long a vector is will not send that many numbers."""
    from chess_variant_sandbox.agents.features import VOCABULARY
    from chess_variant_sandbox.llm.client import wire_schema
    from chess_variant_sandbox.llm.heuristics import DEFAULT_K, heuristic_schema

    wire = wire_schema(heuristic_schema(DEFAULT_K))
    weights = wire["properties"]["heuristics"]["items"]["properties"]["weights"]
    assert f"exactly {len(VOCABULARY)} items" in weights["description"]
    # The original description survives; the restatement is added to it, never instead.
    assert "one number per measurement" in weights["description"]
    assert f"exactly {DEFAULT_K} items" in wire["properties"]["heuristics"]["description"]

    bounded = wire_schema({"type": "number", "minimum": -8, "maximum": 8})
    assert bounded["description"] == "between -8 and 8"


def test_a_property_named_like_a_keyword_survives_the_rewrite() -> None:
    """`rule_spec_schema` has properties named `0` and `1`; a property named `minimum`
    is the same hazard, and a schema walked as an undifferentiated dict loses it."""
    from chess_variant_sandbox.llm.client import wire_schema

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "minimum": {"type": "string"},
            "maxItems": {"type": "integer", "maximum": 4},
        },
    }
    wire = wire_schema(schema)
    assert set(wire["properties"]) == {"minimum", "maxItems"}
    assert wire["properties"]["minimum"] == {"type": "string"}
    # ...while the real keyword inside that property is still removed.
    assert "maximum" not in wire["properties"]["maxItems"]


def test_minitems_is_clamped_rather_than_dropped() -> None:
    """Zero and one are accepted, so "at least one" survives as a real constraint."""
    from chess_variant_sandbox.llm.client import wire_schema

    assert wire_schema({"type": "array", "minItems": 1})["minItems"] == 1
    assert wire_schema({"type": "array", "minItems": 5})["minItems"] == 1


def test_a_type_union_becomes_the_anyof_the_api_accepts() -> None:
    """Same meaning, in the dialect the request takes.

    `rules.promotion`'s rank is "'last', or a rank index", which is a union of two types
    and is written as one. The wire takes unions only as `anyOf`, and sibling keywords
    apply to whichever arm matched, so nothing else about the node moves.
    """
    from chess_variant_sandbox.llm.client import wire_schema

    wire = wire_schema({"type": ["string", "integer"], "description": "'last', or an index."})
    assert "type" not in wire
    assert wire["anyOf"] == [{"type": "string"}, {"type": "integer"}]
    assert wire["description"] == "'last', or an index."


def test_a_property_the_wire_cannot_express_is_removed_from_its_parent() -> None:
    """The one thing dropped rather than restated.

    A map whose keys are not known in advance -- `{"additionalProperties": {...}}` -- has
    no form the request accepts, and there is no weaker one to fall back to: a closed
    object with no properties is one that must be empty. So the property goes, and its
    parent loses it from `required` too.

    No schema this project currently sends contains one. `rule_spec_schema` does, at
    `rules.drops.region`, and it is not sent at all -- see
    `test_the_rule_spec_schema_is_too_big_to_compile_and_that_is_not_a_bug`.
    """
    from chess_variant_sandbox.llm.client import wire_schema

    wire = wire_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kept", "by_piece"],
            "properties": {
                "kept": {"type": "string"},
                "by_piece": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    )
    assert set(wire["properties"]) == {"kept"}
    assert wire["required"] == ["kept"]


def test_no_schema_this_project_sends_asks_for_a_map_with_unknown_keys() -> None:
    """Because such a property would be silently dropped from the request.

    The check is on the *sent* form, so a schema that is not sent -- `rule_spec_schema`,
    which the grammar compiler refuses whole -- is not held to it.
    """
    from chess_variant_sandbox.llm.client import (
        SCHEMA_MAPS,
        SCHEMA_VALUES,
        constrainable,
        wire_schema,
    )

    def open_maps(node, path="$"):
        found = []
        if isinstance(node, list):
            for index, item in enumerate(node):
                found += open_maps(item, f"{path}[{index}]")
            return found
        if not isinstance(node, dict):
            return found
        if node.get("type") == "object" and isinstance(node.get("additionalProperties"), dict):
            found.append(path)
        for key, value in node.items():
            if key in SCHEMA_MAPS and isinstance(value, dict):
                for name, sub in value.items():
                    found += open_maps(
                        sub, f"{path}.{name}" if key == "properties" else f"{path}.{key}.{name}"
                    )
            elif key in SCHEMA_VALUES:
                found += open_maps(value, f"{path}.{key}")
        return found

    for name, schema in _every_schema_this_project_sends().items():
        if not constrainable(schema):
            continue
        assert open_maps(schema) == [], name
        assert open_maps(wire_schema(schema)) == [], name


def test_an_object_that_did_not_say_it_was_closed_is_closed() -> None:
    from chess_variant_sandbox.llm.client import wire_schema

    wire = wire_schema({"type": "object", "properties": {"a": {"type": "string"}}})
    assert wire["additionalProperties"] is False


# ------------------------------------------------------ what the *model* will accept


def _accepts(model: str) -> dict[str, bool]:
    """What each model this project calls takes on the request itself.

    Restated here rather than imported, for the same reason `_refused` re-walks a schema
    instead of calling `wire_schema`: a table that checks itself checks nothing. The
    first row is measured -- `claude-haiku-4-5` answers 400 to adaptive thinking, which
    is how the judge was found dead -- and the effort column is the published support
    matrix, where `output_config.effort` starts at Opus 4.5 and the 4.6 family.
    """
    return {
        "claude-opus-5": {"thinking": True, "effort": True},
        "claude-haiku-4-5": {"thinking": False, "effort": False},
    }[model]


def test_every_model_this_project_names_has_a_wire_row() -> None:
    """A stage pointed at a new model is a failing test, not a bill.

    The same shape of guard as `_every_schema_this_project_sends`: the defect was never
    that the table was wrong, it was that there was no table and every model got the
    parameters the newest one takes.
    """
    from chess_variant_sandbox.llm.client import JUDGE_MODEL, MODEL_WIRE, PROPOSAL_MODEL

    for model in (PROPOSAL_MODEL, JUDGE_MODEL):
        assert model in MODEL_WIRE, model
        assert model in ("claude-opus-5", "claude-haiku-4-5"), (
            f"{model} has no independently stated row in this test"
        )


def test_usage_records_what_a_cache_entry_cost_as_well_as_what_it_saved() -> None:
    """The read meter alone cannot tell a free miss from a paid one.

    `Usage` tracked `cache_read_input_tokens` and not `cache_creation_input_tokens`, so
    a run could say "nothing came from cache" and leave it ambiguous whether that was
    free -- a prefix too short to cache -- or the expensive failure, an entry opened at
    1.25x on every call and expired before anything read it. That distinction is the
    whole of the evidence for `CACHE_TTL`, so it has to be recorded.
    """
    from chess_variant_sandbox.llm.client import Usage

    class _Meters:
        input_tokens = 12
        output_tokens = 34
        cache_read_input_tokens = 900
        cache_creation_input_tokens = 300

    usage = Usage()
    usage.add(_Meters())
    assert usage.as_dict() == {
        "calls": 1,
        "input_tokens": 12,
        "output_tokens": 34,
        "cache_read_tokens": 900,
        "cache_write_tokens": 300,
    }
    assert usage.cache_reads_per_write == 3.0

    # The signature of the failure: written every time, read never.
    wasted = Usage(cache_write_tokens=10_000, calls=4)
    assert wasted.cache_reads_per_write == 0.0
    # And a run that never cached at all is not that failure, and must not read as it.
    assert Usage(calls=4).cache_reads_per_write == 0.0
    assert Usage(calls=4).cache_write_tokens == 0


def test_every_model_this_project_names_has_a_cache_minimum_row() -> None:
    """The same guard as the wire table, for the third wire fact.

    A prefix below its model's minimum is not cached and the API does not say so, so a
    stage pointed at a new model would go on emitting a marker that silently does
    nothing -- which is the failure this table exists to make visible.
    """
    from chess_variant_sandbox.llm.client import CACHE_MINIMUM, JUDGE_MODEL, PROPOSAL_MODEL

    for model in (PROPOSAL_MODEL, JUDGE_MODEL):
        assert model in CACHE_MINIMUM, model
    assert CACHE_MINIMUM[PROPOSAL_MODEL] == 512
    assert CACHE_MINIMUM[JUDGE_MODEL] == 4_096


def test_only_the_prompt_big_enough_to_cache_carries_a_breakpoint() -> None:
    """What the marker is for, asked of the prompts this project really sends.

    `llm/compare.py` and `llm/judge.py` carried a `cache_control` marker for their whole
    lives on prefixes of 253 and 858 tokens, against Haiku 4.5's minimum of 4,096. Nothing failed and nothing cached. `llm/propose.py` is the one that pays, and
    it must go on paying: this asserts both halves, because a change that suppressed the
    marker everywhere would look like a fix and cost the expensive call its cache.
    """
    import json

    from chess_variant_sandbox.llm.client import JUDGE_MODEL, PROPOSAL_MODEL, cached_system
    from chess_variant_sandbox.llm.compare import COMPARISON_BRIEF
    from chess_variant_sandbox.llm.judge import JUDGE_BRIEF
    from chess_variant_sandbox.llm.propose import DESIGN_BRIEF, SCHEMA_PREAMBLE
    from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

    proposal_prefix = "\n\n".join(
        [
            DESIGN_BRIEF,
            SCHEMA_PREAMBLE,
            "The schema:",
            json.dumps(rule_spec_schema(), indent=2),
        ]
    )
    blocks = cached_system(PROPOSAL_MODEL, proposal_prefix)
    # An hour, not the default five minutes: a generation is a tournament long, so an
    # entry that expires between proposals is a 1.25x surcharge nothing ever reads.
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    for brief in (COMPARISON_BRIEF, JUDGE_BRIEF):
        judged = cached_system(JUDGE_MODEL, brief)
        assert "cache_control" not in judged[0], brief[:40]
        # Suppressing the marker must not move a byte of what the model is asked.
        assert judged[0]["text"] == brief


def test_suppressing_the_marker_leaves_the_prompt_alone() -> None:
    """A prompt that does not cache is the same prompt, minus one key."""
    from chess_variant_sandbox.llm.client import JUDGE_MODEL, PROPOSAL_MODEL, cached_system

    stable, volatile = "s" * 40_000, "v" * 100
    cached = cached_system(PROPOSAL_MODEL, stable, volatile)
    plain = cached_system(JUDGE_MODEL, "short", volatile)

    assert [block["text"] for block in cached] == [stable, volatile]
    assert [block["text"] for block in plain] == ["short", volatile]
    assert "cache_control" not in plain[1]
    assert "cache_control" not in cached[1]


def test_no_request_carries_a_parameter_its_model_refuses() -> None:
    """The bug this section exists for, in one assertion.

    `structured` sent `thinking={"type": "adaptive"}` and an `effort` to every model
    unconditionally. `claude-haiku-4-5` accepts neither, so every judging call in this
    project answered 400 and the pairwise judge had never produced a comparison -- while
    the whole offline suite passed, because a stub client is handed a schema and never a
    request.
    """
    from chess_variant_sandbox.llm.client import JUDGE_MODEL, PROPOSAL_MODEL, request_options

    for model in (PROPOSAL_MODEL, JUDGE_MODEL):
        takes = _accepts(model)
        for name, schema in _every_schema_this_project_sends().items():
            options = request_options(model, "medium", schema)
            assert ("thinking" in options) is takes["thinking"], (model, name)
            assert ("effort" in options["output_config"]) is takes["effort"], (model, name)
            # Whatever else moves, the format block is the point of the call.
            sent = options["output_config"]["format"]
            assert sent["type"] == "json_schema"
            assert not _refused(sent["schema"]), (model, name)


def test_the_proposer_still_thinks_as_hard_as_it_is_told_to() -> None:
    """The fix must not quietly cost the expensive call its thinking.

    A proposal directs a whole tournament and is the one place this project deliberately
    does not economise; a "make the judge work" change that also turned the proposer's
    effort off would be a silent downgrade of the thing the loop is for.
    """
    from chess_variant_sandbox.llm.client import PROPOSAL_MODEL, request_options

    options = request_options(PROPOSAL_MODEL, "high", {"type": "object", "properties": {}})
    assert options["thinking"] == {"type": "adaptive"}
    assert options["output_config"]["effort"] == "high"


def _violations(instance, schema, path="$"):
    """Where ``instance`` breaks ``schema``, over the constructs the wire form uses.

    Deliberately not a general JSON Schema implementation, and not a dependency: what is
    being checked is a schema this repository generates, so the vocabulary is known --
    types, enums, properties, required, closed objects, arrays and `anyOf`. The
    constraints the wire drops are absent by construction and there is nothing here to
    check them with.
    """
    bad: list[str] = []
    if "anyOf" in schema:
        if not any(not _violations(instance, arm, path) for arm in schema["anyOf"]):
            return [f"{path}: matches no arm of anyOf"]
        return bad

    kind = schema.get("type")
    if isinstance(kind, list):
        # A type union, which the generated schema writes for `promotion.rank` and the
        # wire form rewrites as `anyOf`. Either spelling means the same thing here.
        return _violations(instance, {**schema, "anyOf": [{"type": one} for one in kind]}, path)
    if kind == "object":
        if not isinstance(instance, dict):
            return [f"{path}: expected an object"]
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties")
        bad += [
            f"{path}.{name}: required but absent"
            for name in schema.get("required", [])
            if name not in instance
        ]
        for name, value in instance.items():
            if name in properties:
                bad += _violations(value, properties[name], f"{path}.{name}")
            elif isinstance(extra, dict):
                # A map with unknown keys: every value is held to the same schema.
                bad += _violations(value, extra, f"{path}.{name}")
            elif extra is False:
                bad.append(f"{path}.{name}: no such property")
        return bad
    if kind == "array":
        if not isinstance(instance, list):
            return [f"{path}: expected an array"]
        for index, item in enumerate(instance):
            bad += _violations(item, schema.get("items", {}), f"{path}[{index}]")
        return bad

    if "enum" in schema and instance not in schema["enum"]:
        bad.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    expected = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    if kind in expected and not isinstance(instance, expected[kind]):
        bad.append(f"{path}: expected {kind}, got {instance!r}")
    return bad


def test_the_rule_spec_schema_is_too_big_to_compile_and_that_is_not_a_bug() -> None:
    """A response format is a decoding grammar, and the compiler has size limits.

    **Measured, in the API's own words.** `rule_spec_schema` has 67 properties outside
    `required` against a limit of 24: *"Schemas contains too many optional parameters
    (67), which would make grammar compilation inefficient"*. The obvious escape -- every
    property required and each allowed to be null -- trades those 67 optionals for 101
    unions and is refused again: *"too many parameters with union types (101 ...). This
    causes exponential compilation"*.

    So it is not a schema to shrink. It is 67 optionals because that is what a rule DSL
    is -- a game states the rules it has and nothing about the rest, and `rules` alone is
    48 of them -- and cutting it to 24 would mean choosing which two thirds of the
    vocabulary a proposal may reach, which is the defect `spec/jsonschema.py` exists to
    prevent. `llm/propose.py` states the contract in the prompt instead.

    Asserted in both directions: if the limit rises or the schema shrinks, the guarantee
    can come back, and this test is where that is noticed.
    """
    from chess_variant_sandbox.llm.client import MAX_OPTIONAL_PARAMETERS, constrainable
    from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

    assert MAX_OPTIONAL_PARAMETERS == 24
    assert not constrainable(rule_spec_schema())
    for name, schema in _every_schema_this_project_sends().items():
        if name != "rule_spec_schema":
            assert constrainable(schema), name


def test_the_proposer_asks_for_no_response_format_it_cannot_have() -> None:
    """And still asks for one everywhere it can.

    The decision is made by measuring the schema rather than by remembering this, so a
    schema that one day fits gets its guarantee back on its own.
    """
    from chess_variant_sandbox.llm.client import PROPOSAL_MODEL, request_options
    from chess_variant_sandbox.llm.heuristics import DEFAULT_K, heuristic_schema
    from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

    unconstrained = request_options(PROPOSAL_MODEL, "high", None)
    assert "format" not in unconstrained["output_config"]
    # The parameters the model *does* take are unaffected: this is about the reply.
    assert unconstrained["thinking"] == {"type": "adaptive"}
    assert unconstrained["output_config"]["effort"] == "high"

    constrained = request_options(PROPOSAL_MODEL, "high", heuristic_schema(DEFAULT_K))
    assert constrained["output_config"]["format"]["type"] == "json_schema"

    # And the proposer reaches for the same predicate rather than a remembered answer.
    from chess_variant_sandbox.llm import propose

    assert not propose.constrainable(rule_spec_schema())


def test_the_proposer_can_state_the_games_this_project_already_has() -> None:
    """The other half of "does the request work": can the contract express anything good?

    A schema no real ruleset satisfies is a proposer that can only produce rejects, and
    the failure looks like the model being careless rather than like the contract being
    wrong. Every committed ruleset is an instance of the schema the proposer is shown --
    including `crazyhouse`, whose per-piece drop region has no *response* schema form and
    needs none, because the schema is prose in the prompt now.

    Asserted as "nothing is unreachable" rather than against a list of what is. A list
    goes stale the next time a ruleset is added -- which is how `crazyhouse-torus` broke
    the version of this test that carried one -- and the property worth holding does not.
    """
    import json
    from pathlib import Path

    from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
    from chess_variant_sandbox.spec.schema import SCHEMA_VERSION

    schema = rule_spec_schema()
    unreachable = {}
    for path in sorted((Path(__file__).resolve().parents[1] / "rulesets").glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        # Two blocks a proposal is deliberately not asked for, and a version every
        # proposal states by construction -- neither says anything about expressiveness.
        spec = {k: v for k, v in spec.items() if k not in ("appearance", "provenance")}
        spec["schema_version"] = SCHEMA_VERSION
        problems = _violations(spec, schema)
        if problems:
            unreachable[path.stem] = problems

    assert unreachable == {}, unreachable


def test_the_rewrite_is_idempotent() -> None:
    """A schema already in the accepted dialect passes through unchanged.

    Which is what makes the shape rules safe to add to a file whose other half is
    *measured*: `heuristic_schema` is the one schema known to make successful calls --
    stage 23 made seven of them -- and its wire form is byte-identical across this
    change, as are the judge's two. Only `rule_spec_schema` moved, and it had never been
    sent. Idempotence is the durable form of that check: a rule that fired twice would
    restate a constraint twice, or rewrite its own rewrite.
    """
    import json

    from chess_variant_sandbox.llm.client import wire_schema

    for name, schema in _every_schema_this_project_sends().items():
        once = wire_schema(schema)
        assert json.dumps(wire_schema(once), sort_keys=True) == json.dumps(
            once, sort_keys=True
        ), name


class TestReadingAnUnconstrainedReply:
    """What `json_object` has to survive once no grammar guarantees the reply's shape.

    Only the rule proposal is unconstrained, and it is the expensive call: a reply
    wrapped in a fence or preceded by a sentence would otherwise cost a corrective retry
    -- or, before this, a `JSONDecodeError` out of the middle of a generation.
    """

    def test_a_bare_object_is_just_json(self) -> None:
        from chess_variant_sandbox.llm.client import json_object

        assert json_object('{"name": "lancers", "board": {"width": 5}}') == {
            "name": "lancers",
            "board": {"width": 5},
        }

    def test_a_fenced_object_survives(self) -> None:
        from chess_variant_sandbox.llm.client import json_object

        assert json_object('Here it is:\n```json\n{"name": "lancers"}\n```\n') == {
            "name": "lancers"
        }

    def test_a_brace_inside_a_string_does_not_end_the_object(self) -> None:
        """The reason this scans rather than searching for the last `}`.

        A ruleset's `description` is model-written prose and may contain anything.
        """
        from chess_variant_sandbox.llm.client import json_object

        assert json_object('prose {"description": "a } brace", "name": "x"} more prose') == {
            "description": "a } brace",
            "name": "x",
        }

    def test_an_escaped_quote_does_not_end_the_string(self) -> None:
        from chess_variant_sandbox.llm.client import json_object

        assert json_object(r'{"name": "the \"lancers\"}", "id": "L"}')["id"] == "L"

    def test_a_reply_with_no_object_is_an_error_rather_than_a_guess(self) -> None:
        import pytest

        from chess_variant_sandbox.llm.client import json_object

        with pytest.raises(ValueError):
            json_object("I would rather not design a game today.")
        with pytest.raises(ValueError):
            json_object('{"name": "unclosed"')
        # A JSON *array* is valid JSON and is not a ruleset.
        with pytest.raises(ValueError):
            json_object("[1, 2, 3]")


def test_feedback_reads_the_seat_from_the_verdict_and_never_from_the_figure() -> None:
    """The first-player figure is 32 self-play games wide and cannot tell an even seat
    from a 0.28 lean, so the two sentences it used to drive -- a large advantage above
    0.2, an even match below 0.08 -- told the proposer a fiction either way. The seat
    sentence now mirrors the seat gate's refusal, which fires only on a sweep of the
    strongest rungs' games (`docs/reference/metrics-and-calibration-record.md`, Q235)."""
    from chess_variant_sandbox.llm.feedback import describe
    from chess_variant_sandbox.metrics import load_weights, score_components
    from chess_variant_sandbox.metrics.components import Components
    from chess_variant_sandbox.metrics.seat import SEAT_DECIDED_CHANCE, TOP_RUNG_SEAT_DETAIL

    weights = load_weights("v3")
    raw = {name: 0.5 for name in weights.components()}
    raw.update({"mean_plies": 40, "mean_branching": 20, "decisive_rate": 0.6,
                "skill_gradient": 5, "search_advantage": 0.55, "rule_mass": 200,
                "first_player_advantage": 0.9, SEAT_DECIDED_CHANCE: 2 / 65536})
    sweep = {"rungs": [1600, 6400], "games": 16, "wins": 16, "draws": 0, "losses": 0}
    result = score_components(
        "h", "g", Components(raw=raw, diagnostics={"games": 196, TOP_RUNG_SEAT_DETAIL: sweep}),
        weights,
    )
    assert not result.viable
    text = describe(result).lower()
    assert "the same seat won nearly every game" in text
    assert "seat decided" not in text and "seat_decided" not in text

    # A figure of 0.9 with no refusal says nothing about the seat, in either direction.
    raw[SEAT_DECIDED_CHANCE] = 1.0
    text = describe(
        score_components("h", "g", Components(raw=raw, diagnostics={"games": 196}), weights)
    ).lower()
    assert "the same seat won" not in text
    assert "moving first is a large" not in text
    assert "evenly matched" not in text



class TestAPortalIsNotListedAmongThePieces:
    """Q877, after Q875: the count vector follows the compiled types, the declared pieces and
    then one type per portal end, so it runs past the spec's piece list on a portal game. The
    feedback's piece lines and the `idle_piece_types` diagnostic took every entry for a piece
    and named the ends "piece 6", "piece 7" (a fixed end as barely used). A portal is the
    board's, held by no player, and both now read the vector as `piece_utilisation` does."""

    @staticmethod
    def _beside():
        from test_schema_twenty_four import GAMES

        from chess_variant_sandbox.engine import Engine

        spec = GAMES["beside"]()
        n_types = Engine(spec).spec.n_types
        assert n_types > len(spec["pieces"]), "the spec must carry portal ends"
        return spec, n_types

    def test_the_piece_lines_name_only_the_declared_pieces_a_player_holds(self) -> None:
        from chess_variant_sandbox.llm.feedback import _piece_lines
        from chess_variant_sandbox.metrics.components import held_type_indices

        spec, n_types = self._beside()
        held = held_type_indices(spec, n_types)
        # Every type plays, so that nothing is idle for want of moves, and the board's own
        # moves (the ends) are many, so that counting them would shift every percentage.
        totals = [100 if index in held else 40 for index in range(n_types)]
        text = " ".join(_piece_lines({"moves_by_type": totals}, spec))

        assert "piece " not in text.replace("Moves played by piece:", "")
        for index, piece in enumerate(spec["pieces"]):
            name = str(piece.get("name") or piece["id"])
            assert (name in text) == (index in held), (name, text)
        assert f"({100 / (100 * len(held)) * 100:.0f}%)" in text

    def test_a_fixed_end_is_not_barely_used(self) -> None:
        from chess_variant_sandbox.llm.feedback import _piece_lines
        from chess_variant_sandbox.metrics.components import held_type_indices

        spec, n_types = self._beside()
        held = held_type_indices(spec, n_types)
        totals = [100 if index in held else 0 for index in range(n_types)]
        assert not any("Barely used" in line for line in _piece_lines(
            {"moves_by_type": totals}, spec
        ))

    def test_idle_piece_types_counts_only_the_army(self) -> None:
        from chess_variant_sandbox.metrics.components import (
            _analysis_components,
            held_type_indices,
        )
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, GameAnalysis

        spec, n_types = self._beside()
        held = held_type_indices(spec, n_types)
        totals = [100 if index in held else 0 for index in range(n_types)]
        game = GameAnalysis(seed=0, agents=("a", "b"), moves_by_type=totals)
        diagnostics: dict = {}
        _analysis_components(CorpusAnalysis(games=[game]), diagnostics, spec)

        assert diagnostics["held_piece_types"] == len(held)
        assert diagnostics["idle_piece_types"] == 0


def test_the_prompt_states_the_portal_bar_as_the_validator_enforces_it() -> None:
    """Q881: the bar is refused when no move can meet it (Q879), and the prompt says when."""
    from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE
    from chess_variant_sandbox.spec.schema import validate_spec

    assert "only on a pair somebody moves" not in SCHEMA_PREAMBLE
    assert "either both players must be able to move an end" in SCHEMA_PREAMBLE
    assert "a turn must be two moves" in SCHEMA_PREAMBLE
    assert "the portal piece must have a move back" in SCHEMA_PREAMBLE
    # The sentence is pinned to the validator's words for each refusal.
    from test_schema_twenty import _sluice_draft

    errors = validate_spec(_sluice_draft())
    assert any("each end that moves is moved by one player only" in e for e in errors)
    spec = _sluice_draft()
    spec["rules"]["portals"][0]["ends"][2]["moved_by"] = [0, 1]
    spec["rules"]["turn"] = {"moves": 2}
    spec["rules"].pop("en_passant", None)
    assert not any("rules.portals[0].no_straight_back" in e for e in validate_spec(spec))
