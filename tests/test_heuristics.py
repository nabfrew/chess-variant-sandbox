"""The heuristic player stage: identity, the boundary, the leak surface, the ladder.

The tests that matter here are the ones that would let a plausible-looking change do
real damage silently:

* a heuristic reaching the ladder components, which would make a per-ruleset player
  contaminate the one thing every ruleset is compared on;
* the Rust port and the Python oracle disagreeing about what a term means, which shows
  up as a different game rather than as a failing assertion;
* an empty weight vector reaching the feature evaluator, which would move every
  committed number in the project;
* the generated schema and the vocabulary drifting apart, which would make a term
  unproposable while every test still passed.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents import (
    NATIVE_VALUES,
    NATIVE_WEIGHTS,
    build_agent,
    native_spec,
    parse_agent,
)
from chess_variant_sandbox.agents.evaluation import Evaluator, PieceValues, mean_abs_non_royal
from chess_variant_sandbox.agents.features import (
    GLOSSES,
    VOCABULARY,
    VOCABULARY_VERSION,
    FeatureEvaluator,
    builtin_vector,
    feature_context,
)
from chess_variant_sandbox.agents.heuristics import (
    Heuristic,
    HeuristicError,
    HeuristicSet,
    other_spelling,
    register_heuristic,
    resolve,
    weights_for,
)
from chess_variant_sandbox.agents.objective import objective_for
from chess_variant_sandbox.agents.seeds import anchor, handwritten_set, random_set
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.heuristics import HEURISTIC_COMPONENTS, heuristic_components

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"


def load(name: str) -> dict:
    return json.loads((RULESETS / f"{name}.json").read_text(encoding="utf-8"))


def vector(**terms: float) -> tuple[float, ...]:
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return tuple(weights)


class TestIdentity:
    """A heuristic is named by its weights and by nothing else."""

    def test_scaling_a_vector_does_not_make_a_new_player(self) -> None:
        """Argmax is invariant under a positive scale, so these are one heuristic.

        Without normalising before hashing, a set of six proposals routinely contains
        three spellings of one plan and `k` silently becomes four -- which matters
        because every statistic the pass reports is taken over `k`.
        """
        first = Heuristic("a", "take material", vector(material=1.0, mobility=0.05))
        second = Heuristic("b", "the same thing", vector(material=7.0, mobility=0.35))
        assert first.digest == second.digest

    def test_the_name_and_the_plan_are_not_part_of_identity(self) -> None:
        weights = vector(material=1.0)
        assert Heuristic("x", "one story", weights).digest == (
            Heuristic("y", "another story", weights).digest
        )

    def test_a_different_balance_is_a_different_player(self) -> None:
        assert Heuristic("a", "", vector(material=1.0, mobility=0.05)).digest != (
            Heuristic("b", "", vector(material=1.0, mobility=0.5)).digest
        )

    def test_an_all_zero_vector_is_refused(self) -> None:
        """It is a player that moves at random: a legitimate control, never a proposal."""
        with pytest.raises(HeuristicError, match="moves at random"):
            Heuristic("nothing", "", tuple([0.0] * len(VOCABULARY)))

    def test_a_vector_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(HeuristicError, match="weights"):
            Heuristic("short", "", (1.0, 0.0))

    def test_an_edited_store_fails_at_load(self) -> None:
        """A store is a cache with a verified key, not an input.

        This is what lets the determinism contract survive content addressing: a game is
        still a pure function of `(ruleset_hash, seed, agent_configs)` precisely because
        a digest names exactly one vector, and a file that disagrees is refused rather
        than believed.
        """
        heuristic = Heuristic("a", "", vector(material=1.0, mobility=0.05))
        payload = HeuristicSet("deadbeef", (heuristic,)).as_dict()
        # The ratio, not the scale: halving every weight would be the same player, which
        # is exactly what the digest is designed to say.
        payload["heuristics"][0]["weights"][VOCABULARY.index("mobility")] = 0.5
        with pytest.raises(HeuristicError, match="has been edited"):
            HeuristicSet.from_dict(payload)

    def test_a_store_from_another_vocabulary_is_refused(self) -> None:
        payload = HeuristicSet("deadbeef", (Heuristic("a", "", vector(material=1.0)),)).as_dict()
        payload["vocabulary_version"] = VOCABULARY_VERSION + 1
        with pytest.raises(HeuristicError, match="regenerate"):
            HeuristicSet.from_dict(payload)

    def test_resolving_an_unknown_digest_raises_rather_than_defaulting(self) -> None:
        """Never a silent fallback: that would be a different player under one name."""
        with pytest.raises(HeuristicError, match="no heuristic"):
            resolve("nosuchdigest")

    def test_a_digest_that_looks_like_a_number_survives_the_option_parser(self) -> None:
        """`parse_agent` coerced anything numeric-looking, and a digest looks numeric.

        Two shapes, both met in practice and neither exotic:

        * ``096717939346`` -- all digits, so ``int`` took it and the leading zero was
          gone. One digest in sixteen starts with a zero.
        * ``37625e727351`` -- valid float syntax, so ``float`` took it, the exponent
          overflowed, and the digest arrived as ``inf``. Roughly one in two hundred, and
          the first model-proposed set in the project contained one -- it stopped a
          corpus run halfway through with a lookup for ``'000000000inf'``.

        The first was repairable after the fact by re-padding; the second is not,
        because ``float`` has already discarded the digits. Both are refused at the
        parser now, on the general rule that a coercion which changes the text has lost
        the identity it was supposed to preserve.
        """
        for digest in ("096717939346", "37625e727351"):
            config = parse_agent(f"greedy:h={digest}")
            assert config.options["h"] == digest
            assert str(config) == f"greedy:h={digest}"

        # And an ordinary numeric option is still a number, which is the half of the
        # coercion that has to keep working.
        numeric = parse_agent("minimax:nodes=5000,mobility=0.1,max_depth=2")
        assert numeric.options == {"nodes": 5000, "mobility": 0.1, "max_depth": 2}

    def test_a_registered_digest_resolves_through_the_parser(self) -> None:
        """End to end: the weights that come back out are the ones that went in."""
        weights = vector(material=1.0, centrality=0.4)
        digest = register_heuristic(Heuristic("hold", "", weights))
        config = parse_agent(f"greedy:h={digest}")
        assert weights_for(dict(config.options)) == weights


class TestADigestNamesOnePlayer:
    """The digest is taken over the unit-length weights; the core plays the raw ones (Q433).

    Scaling every weight leaves a search's argmax alone in exact arithmetic, which is why
    two spellings share a digest. The core's arithmetic is not exact: Q431 found gardner's
    built-in vector at 0.707 of unit length playing a different game from the same vector
    at unit length, 0.4167 where two spellings of one player score exactly 0.5. So a
    second raw vector under a held digest is a second player under the first one's name,
    and registering it is refused rather than dropped in silence.
    """

    def test_a_second_scale_under_a_held_digest_is_refused(self) -> None:
        held = Heuristic("held", "", vector(material=1.0, centrality=0.37))
        scaled = Heuristic("scaled", "", vector(material=3.0, centrality=1.11))
        assert held.digest == scaled.digest and held.weights != scaled.weights
        register_heuristic(held)
        with pytest.raises(HeuristicError, match="already names"):
            register_heuristic(scaled)
        assert resolve(held.digest).weights == held.weights

    def test_other_piece_values_under_a_held_digest_are_refused(self) -> None:
        """The values are rounded into the digest too, and the core plays them unrounded."""
        weights = vector(material=1.0, centrality=0.37)
        held = Heuristic("held", "", weights, values=(1.0, 3.0))
        nudged = Heuristic("nudged", "", weights, values=(1.0, 3.0000001))
        assert held.digest == nudged.digest and held.values != nudged.values
        register_heuristic(held)
        with pytest.raises(HeuristicError, match="already names"):
            register_heuristic(nudged)
        assert resolve(held.digest).values == held.values

    def test_the_same_player_registers_again_under_any_name(self) -> None:
        """Identity is the numbers: a second name for one vector is the first player."""
        weights = vector(material=1.0, centrality=0.37)
        digest = register_heuristic(Heuristic("first", "one story", weights))
        assert register_heuristic(Heuristic("first", "one story", weights)) == digest
        assert register_heuristic(Heuristic("second", "another story", weights)) == digest
        assert resolve(digest).name == "first"

    def test_a_set_that_clashes_registers_none_of_itself(self) -> None:
        """A store is refused whole, so a roster that skips it offers none of its players."""
        register_heuristic(Heuristic("held", "", vector(material=1.0, centrality=0.37)))
        fresh = Heuristic("fresh", "", vector(material=1.0, mobility=0.29))
        clash = Heuristic("clash", "", vector(material=3.0, centrality=1.11))
        with pytest.raises(HeuristicError, match="already names"):
            HeuristicSet("deadbeef", (fresh, clash)).register()
        with pytest.raises(HeuristicError, match="no heuristic"):
            resolve(fresh.digest)

    def test_a_set_that_spells_one_player_twice_is_refused(self) -> None:
        once = Heuristic("once", "", vector(material=1.0, mobility=0.29))
        twice = Heuristic("twice", "", vector(material=2.0, mobility=0.58))
        assert once.digest == twice.digest and once.weights != twice.weights
        with pytest.raises(HeuristicError, match="already names"):
            HeuristicSet("deadbeef", (once, twice)).register()
        with pytest.raises(HeuristicError, match="no heuristic"):
            resolve(once.digest)

    def test_the_clash_can_be_asked_about_without_registering(self) -> None:
        """What a store writer asks before it writes, so it refuses before, not after."""
        held = Heuristic("held", "", vector(material=1.0, centrality=0.37))
        scaled = Heuristic("scaled", "", vector(material=3.0, centrality=1.11))
        renamed = Heuristic("renamed", "another story", held.weights)
        assert other_spelling(scaled, [held]) is held
        assert other_spelling(renamed, [held]) is None
        assert other_spelling(scaled) is None
        with pytest.raises(HeuristicError, match="no heuristic"):
            resolve(held.digest)
        register_heuristic(held)
        assert other_spelling(scaled) is held

    def test_every_committed_store_loads_beside_every_other(self) -> None:
        """All of `heuristics/` and `heuristics/model/` in one registry, as a server holds them.

        Q431 counted 34 stores and 65 digests with one raw vector each. A store written
        later that spells a held digest at another scale fails here, naming both.
        """
        root = RULESETS.parent / "heuristics"
        stores = sorted(root.glob("*.json")) + sorted((root / "model").glob("*.json"))
        assert stores
        for path in stores:
            HeuristicSet.load(path).register()


class TestTheBuiltInPathIsUntouched:
    """An empty weight vector must never reach the feature evaluator."""

    def test_a_built_in_agent_gets_the_original_evaluator(self) -> None:
        engine = Engine(load("chess"))
        assert isinstance(objective_for(engine, 0.05, ()), Evaluator)
        assert isinstance(build_agent("greedy", engine).evaluator, Evaluator)
        assert isinstance(build_agent("minimax:nodes=100", engine).evaluator, Evaluator)

    def test_a_heuristic_agent_gets_the_feature_evaluator(self) -> None:
        engine = Engine(load("chess"))
        digest = register_heuristic(Heuristic("m", "", vector(material=1.0)))
        assert isinstance(build_agent(f"greedy:h={digest}", engine).evaluator, FeatureEvaluator)

    @pytest.mark.parametrize("agent", ["random", "greedy", "minimax:nodes=400", "mcts:nodes=100"])
    def test_every_built_in_sends_an_empty_weight_vector(self, agent: str) -> None:
        """The last slot of `NativeSpec`. Empty is what keeps the numbers where they are.

        Read from the end rather than by index: the slot was seventh when it was written
        and became eighth the moment stage 17A's goal weight landed beside it, and a
        test that pins the position fails for the wrong reason next time.
        """
        spec = native_spec(agent)
        assert spec is not None
        assert spec[NATIVE_WEIGHTS] == () and spec[NATIVE_VALUES] == ()

    def test_the_anchor_reproduces_the_built_in_ordering(self) -> None:
        """`builtin_vector` is an anchor, and the claim made for it is about decisions.

        Not about bits: dividing and multiplying by the same float is not the identity.
        So the test is that the two objectives rank the same move first, not that they
        return the same number.
        """
        engine = Engine(load("chess"))
        context = feature_context(engine.spec)
        heuristic = FeatureEvaluator(context, builtin_vector(context))
        builtin = Evaluator.for_engine(engine, mobility_weight=0.05)

        state = engine.initial_state()
        pairs = []
        for move in engine.legal_moves(state):
            child = engine.apply(state, move)
            count = len(engine.legal_moves(child))
            pairs.append(
                (
                    heuristic.evaluate(engine, child, 0, count),
                    builtin.evaluate(engine, child, 0, count),
                )
            )
        by_heuristic = sorted(range(len(pairs)), key=lambda i: -pairs[i][0])
        by_builtin = sorted(range(len(pairs)), key=lambda i: -pairs[i][1])
        assert by_heuristic == by_builtin

    def test_the_anchor_reproduces_the_built_in_ordering_with_a_hand(self) -> None:
        """The same claim where the built-in counts a hand (`v11` A6).

        The vocabulary's `material` column is the board alone, so on a ruleset with
        `drops` the anchor carries the hand on `hand_material` as well. A position with
        both hands holding something and a drop among the moves is where a vector
        without that weight would put a drop first that the built-in does not, so the
        comparison is taken there, over every child.
        """
        engine = Engine(load("crazyhouse"))
        context = feature_context(engine.spec)
        heuristic = FeatureEvaluator(context, builtin_vector(context))
        builtin = Evaluator.for_engine(engine, mobility_weight=0.05)
        assert builtin_vector(context)[VOCABULARY.index("hand_material")] > 0.0

        rng = random.Random(3)
        state = engine.initial_state()
        while not (any(state.hands[0]) and any(state.hands[1])):
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
        player = state.side_to_move

        pairs = []
        for move in engine.legal_moves(state):
            child = engine.apply(state, move)
            count = len(engine.legal_moves(child))
            pairs.append(
                (
                    heuristic.evaluate(engine, child, player, count),
                    builtin.evaluate(engine, child, player, count),
                )
            )
        by_heuristic = sorted(range(len(pairs)), key=lambda i: -pairs[i][0])
        by_builtin = sorted(range(len(pairs)), key=lambda i: -pairs[i][1])
        assert by_heuristic == by_builtin


# Each term that its gloss says holds "only where" a rule is declared, and how a spec
# declares that rule. Terms with no entry (material, mobility, ...) read on every game.
DECLARED_BY: dict[str, Any] = {
    "territory_margin": lambda spec: "territory" in spec["rules"],
    "score_margin": lambda spec: "scoring" in spec["rules"],
    "score_urgency": lambda spec: "scoring" in spec["rules"],
    "check_margin": lambda spec: "check_limit" in spec["rules"],
    "extinction_margin": lambda spec: "extinction" in spec["rules"],
    "hand_material": lambda spec: "drops" in spec["rules"],
    "promotion_occupancy": lambda spec: bool(spec["rules"].get("promotion")),
    "reach_progress": lambda spec: "reach" in spec["rules"],
    "facing_alignment": lambda spec: any("facing" in piece for piece in spec["pieces"]),
}


class TestTheVocabulary:
    """Every term reads a declared rule, and nothing hardcodes a game."""

    def test_every_term_has_a_gloss_and_a_function(self) -> None:
        assert set(GLOSSES) == set(VOCABULARY)

    def test_a_gloss_never_states_a_preferred_value(self) -> None:
        """A gloss may name a quantity; it may never state a band or a direction.

        "How many moves the mover has" is a term definition. "The number of options in a
        comfortable range" is a curve, and a curve is the part of an objective an
        optimiser can game. The proposer sees these, so the distinction is the same one
        the design brief is held to.
        """
        forbidden = (
            "should be", "ideally", "comfortable", "too many", "too few",
            "higher is better", "lower is better", "aim for", "good ", "best ",
        )
        for term, gloss in GLOSSES.items():
            lowered = gloss.lower()
            for phrase in forbidden:
                assert phrase not in lowered, f"{term}'s gloss states a preference: {phrase!r}"

    @pytest.mark.parametrize(
        "name",
        ["chess", "shogi", "xiangqi", "rps-v2", "rps-v5", "racingkings", "threecheck",
         "horde", "crazyhouse", "antichess", "dobutsu", "atomic", "dull", "gardner"],
    )
    def test_every_term_evaluates_on_every_committed_ruleset(self, name: str) -> None:
        """A term whose rule the ruleset does not declare reads zero and is inert."""
        raw = load(name)
        engine = Engine(raw)
        context = feature_context(engine.spec)
        state = engine.initial_state()
        count = len(engine.legal_moves(state))
        for index, term in enumerate(VOCABULARY):
            weights = [0.0] * len(VOCABULARY)
            weights[index] = 1.0
            value = FeatureEvaluator(context, tuple(weights)).evaluate(engine, state, 0, count)
            assert value == value, f"{term} is NaN on {name}"
            assert abs(value) <= 2.0, f"{term} is {value} on {name}, outside its stated range"
            # The docstring's claim, asserted: a term tied to a rule this spec does not
            # declare is exactly zero. `DECLARED_BY` is written here, from the glosses'
            # "only where ..." sentences, and shares nothing with the evaluator.
            if term in DECLARED_BY and not DECLARED_BY[term](raw):
                assert value == 0.0, f"{term} reads {value} on {name}, which declares no such rule"

    def test_no_term_names_a_piece(self) -> None:
        """The same rule `tests/test_dsl_generality.py` enforces on the engine.

        Asserted against the vocabulary and its glosses rather than against the whole
        file, because the file's prose has to be able to *discuss* a chess piece in
        order to explain why a piece-shaped term was excluded.
        """
        shown = (" ".join(VOCABULARY) + " " + " ".join(GLOSSES.values())).lower()
        for piece in ("knight", "bishop", "rook", "queen", "pawn", "king", "castle"):
            assert piece not in shown, f"the vocabulary names a chess piece: {piece}"


class TestTheLadderIsNotContaminated:
    """A per-ruleset player must not reach the components every ruleset is compared on."""

    def test_a_heuristic_record_moves_no_ladder_component(self) -> None:
        """The structural version of "we promise to keep the corpora apart".

        `rating_spread` reads the whole rating table and no name filter can reach it, so
        the real guarantee is that the heuristic pass keeps its own corpus. This test is
        what survives someone forgetting that.
        """
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament.ratings import rate
        from chess_variant_sandbox.tournament.record import GameRecord

        def record(seed: int, agents: tuple[str, str]) -> GameRecord:
            return GameRecord(
                ruleset_hash="h", ruleset_name="chess", seed=seed, agents=agents,
                moves=list(range(30)), winner=seed % 2, reason="checkmate", plies=30,
                branching=[20] * 30, material=[0.0] * 30, captures=[seed % 2] * 30,
            )

        clean = [
            record(seed, ("minimax:nodes=100", "minimax:nodes=400")) for seed in range(6)
        ] + [record(seed + 6, ("greedy", "minimax:nodes=1600")) for seed in range(6)]
        polluted = clean + [
            record(seed + 12, ("greedy:h=abc123abc123", "minimax:h=abc123abc123,nodes=400"))
            for seed in range(6)
        ]

        spec = load("chess")
        ladder = ("skill_gradient", "gradient_persistence", "rating_spread", "search_advantage")
        before = compute_components(spec, clean, rate(clean)).raw
        after = compute_components(spec, polluted, rate(polluted)).raw
        for name in ladder:
            if name in before:
                assert before[name] == after[name], f"a heuristic record moved {name}"


class TestTheDiagnosticsAreNeverScored:
    """Built, measured, reported -- and adopted later on evidence, or never.

    The `threat_asymmetry` pattern, and the same test that holds it: a component the
    project declined is not one waiting for a version to be quietly added to.
    """

    def test_no_weights_file_scores_a_heuristic_measurement(self) -> None:
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR, load_weights

        for path in sorted(Path(WEIGHTS_DIR).glob("*.json")):
            scored = set(load_weights(path.stem).components())
            overlap = scored & set(HEURISTIC_COMPONENTS)
            assert not overlap, f"{path.stem} scores {overlap}"

    def test_an_absent_pass_produces_no_measurements(self) -> None:
        """Absent is reported as absent, never as a zero."""
        assert heuristic_components(None) == {}

    def test_an_unsearchable_ruleset_is_unmeasured_rather_than_shallow(self) -> None:
        """The wrong instrument is not a bad game -- the `withheld_components` rule."""
        from chess_variant_sandbox.tournament.heuristics import HeuristicReport, MatchRow

        row = MatchRow("a", "d", "", deep_score=0.5, legibility=1.0,
                       agreement_shallow=1.0, agreement_deep=1.0)
        shallow_search = HeuristicReport("h", "s", nodes=100, depth_reached=1.5, rows=[row])
        produced = heuristic_components(shallow_search)
        assert "heuristic_headroom" not in produced
        assert produced["heuristic_depth_reached"] == 1.5


class TestSeeds:
    def test_the_anchor_is_index_zero_of_every_set(self) -> None:
        """One point per ruleset that owes nothing to any sample."""
        for name in ("chess", "rps-v2", "racingkings"):
            engine = Engine(load(name))
            for builder in (handwritten_set, random_set):
                assert builder(engine.spec).heuristics[0].digest == anchor(engine.spec).digest

    def test_a_plan_is_only_offered_where_its_rule_is_declared(self) -> None:
        chess = handwritten_set(Engine(load("chess")).spec, k=12)
        territory = handwritten_set(Engine(load("rps-v2")).spec, k=12)
        assert not any(h.name == "claim-ground" for h in chess.heuristics)
        assert any(h.name == "claim-ground" for h in territory.heuristics)

    def test_a_set_has_no_duplicate_players(self) -> None:
        for name in ("chess", "dull", "shogi"):
            heuristics = handwritten_set(Engine(load(name)).spec, k=12).heuristics
            digests = [h.digest for h in heuristics]
            assert len(digests) == len(set(digests))

    # A 5x5 game built to declare exactly the rules a test names. `S` may take only `T`,
    # which is the capture matrix; `L` is royal only where a test asks for it.
    @staticmethod
    def declaring(*, matrix: bool = True, royal: bool = True, **rules: Any) -> Any:
        stepper: dict[str, Any] = {
            "id": "S", "name": "stepper", "value": 1,
            "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}],
        }
        if matrix:
            stepper["captures"] = ["T"]
        spec = {
            "schema_version": 4,
            "name": "plan probe",
            "board": {"width": 5, "height": 5},
            "pieces": [
                {"id": "L", "name": "leader", "value": 0 if royal else 1, "royal": royal,
                 "moves": [{"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"}]},
                stepper,
                {"id": "T", "name": "walker", "value": 1,
                 "moves": [{"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"}]},
            ],
            "setup": {"mirror": False, "placements": [
                {"piece": "L", "player": 0, "squares": ["a1"]},
                {"piece": "L", "player": 1, "squares": ["e5"]},
                {"piece": "S", "player": 0, "squares": ["b1"]},
                {"piece": "S", "player": 1, "squares": ["d5"]},
                {"piece": "T", "player": 0, "squares": ["c1"]},
                {"piece": "T", "player": 1, "squares": ["c5"]},
            ]},
            "rules": {"win": "royal_capture" if royal else "annihilation", **rules},
        }
        return Engine(spec).spec

    def test_a_capture_matrix_game_is_offered_the_plan_that_reads_its_matrix(self) -> None:
        """Q162: the one plan about the matrix came last and never fitted in `k = 6`.

        Four plans every game gets came first in the list, so a game got one plan about
        its own rules, whichever the list named first. A royal piece named first, so a
        matrix game with one was never offered the prey term.
        """
        compiled = self.declaring()
        assert compiled.has_capture_matrix and compiled.royal_types
        prey = VOCABULARY.index("prey_weight")
        played = handwritten_set(compiled).heuristics
        assert len(played) == 6
        assert any(h.weights[prey] for h in played[1:])
        # And the other declared rule keeps its plan: nothing about the game was traded
        # for the matrix, only a plan any game would have got.
        assert any(h.weights[VOCABULARY.index("royal_exposure")] for h in played[1:])

    def test_the_game_s_own_plans_come_before_the_ones_every_game_gets(self) -> None:
        """Declared-rule plans are chosen first; the `always` plans fill what is left.

        Chosen in list order and written in list order, so the generic plan that drops
        is the last one listed, and a game declaring one rule keeps all four.
        """
        from chess_variant_sandbox.agents.seeds import describe_plans

        plans = describe_plans()
        always = [p["name"] for p in plans if p["requires"] == "always"]
        order = [p["name"] for p in plans]

        three = self.declaring(reach={"piece": "S", "squares": ["c3"], "mirror": False})
        declared = {"hunt-the-king", "race", "eat-what-you-can"}
        names = [h.name for h in handwritten_set(three).heuristics[1:]]
        assert declared <= set(names)
        assert [n for n in names if n in always] == always[: 5 - len(declared)]
        assert names == sorted(names, key=order.index)

        # One declared rule: the four generic plans are all still there, as before Q162.
        one = [h.name for h in handwritten_set(self.declaring(royal=False)).heuristics[1:]]
        assert one == [*always, "eat-what-you-can"]

        # More declared plans than slots: the first ones listed win, generic plans none.
        tight = [h.name for h in handwritten_set(three, k=3).heuristics[1:]]
        assert tight == ["hunt-the-king", "race"]

    def test_every_committed_hand_written_store_is_what_the_generator_writes(self) -> None:
        """A store is generated once and committed, so it can fall behind the generator.

        Q162 found three whose anchor predated A6 (the hand counted as material) and
        twenty-four whose members predated the plan order. Nothing noticed either. This
        is what would: a hand-written set is a pure function of its spec, so the file
        must be byte-for-byte what `handwritten_set` writes now. Proposed sets under
        `model/` cannot be regenerated without a model and are not checked, and nor is a
        store whose ruleset is no longer committed, which is history rather than drift.
        """
        from chess_variant_sandbox.spec import load_spec, spec_hash

        root = RULESETS.parent
        committed = [*RULESETS.glob("*.json"), *(root / "archive").rglob("*.json")]
        by_hash = {spec_hash(load_spec(path)): path for path in committed}
        checked = 0
        for store in sorted((root / "heuristics").glob("*.json")):
            text = store.read_text(encoding="utf-8")
            data = json.loads(text)
            path = by_hash.get(data["ruleset_hash"])
            if data.get("model") != "hand-written" or path is None:
                continue
            regenerated = handwritten_set(Engine(load_spec(path)).spec, k=int(data["k"]))
            written = json.dumps(regenerated.as_dict(), indent=2) + "\n"
            where = path.relative_to(root).as_posix()
            assert written == text, (
                f"{store.name} ({where}) is not what handwritten_set writes today; move "
                f"it aside and run `chess-variant-sandbox heuristics {where} --generate-only`"
            )
            checked += 1
        assert checked


class TestTheModelContract:
    """A term the schema cannot express cannot be proposed at all."""

    def test_the_schema_matches_the_vocabulary(self) -> None:
        from chess_variant_sandbox.llm.heuristics import heuristic_schema

        weights = heuristic_schema(5)["properties"]["heuristics"]["items"]["properties"][
            "weights"
        ]
        assert weights["minItems"] == len(VOCABULARY)
        assert weights["maxItems"] == len(VOCABULARY)
        for term in VOCABULARY:
            assert term in weights["description"]

    def test_the_schema_is_closed(self) -> None:
        from chess_variant_sandbox.llm.heuristics import heuristic_schema

        schema = heuristic_schema(5)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["heuristics"]["items"]["additionalProperties"] is False

    def test_the_prompt_is_a_pure_function_of_the_vocabulary(self) -> None:
        """What makes it constant across candidates, and therefore gradient-free."""
        from chess_variant_sandbox.llm.heuristics import heuristic_brief

        assert heuristic_brief() == heuristic_brief()

    def test_the_brief_names_no_measurement_the_index_scores(self) -> None:
        """The proposer is asked how to play, never how to make a game score well."""
        from chess_variant_sandbox.llm.heuristics import heuristic_brief
        from chess_variant_sandbox.metrics.scoring import WEIGHTS_DIR, load_weights

        shown = (heuristic_brief() + " " + " ".join(GLOSSES.values())).lower()
        scored: set[str] = set()
        for path in sorted(Path(WEIGHTS_DIR).glob("*.json")):
            scored |= set(load_weights(path.stem).components())
        for component in scored:
            assert component not in shown
            assert component.replace("_", " ") not in shown
        for term in ("index", "interestingness", "weights version", "viable"):
            assert term not in shown

    def test_an_invalid_proposal_is_rejected_by_name(self) -> None:
        from chess_variant_sandbox.llm.heuristics import _validate

        heuristics, errors = _validate(
            {"heuristics": [{"name": "a", "plan": "", "weights": [0.0] * len(VOCABULARY)}]}, 1
        )
        assert not heuristics
        assert any("random" in error for error in errors)

    def test_two_scalings_of_one_plan_are_rejected_as_one_player(self) -> None:
        from chess_variant_sandbox.llm.heuristics import _validate

        first = list(vector(material=1.0))
        second = list(vector(material=3.0))
        _, errors = _validate(
            {
                "heuristics": [
                    {"name": "a", "plan": "", "weights": first},
                    {"name": "b", "plan": "", "weights": second},
                ]
            },
            2,
        )
        assert any("same player" in error for error in errors)


class TestTheBoundary:
    """The Rust port and the Python oracle must be the same player, to the bit.

    `tests/test_backends.py` holds the built-in agents to this by playing one game twice
    over the same Rust engine, once with the Python loop and once through the core. A
    heuristic needs the same check *constructed deliberately*, for the reason `screen`
    and `turn` needed it: no committed ruleset carries a heuristic, so without building
    the arm there is no boundary check at all.

    This is not a formality. A `dispersion` term passed every other test in this file and
    diverged here on nine of fourteen rulesets, because it was a small difference between
    two large means and a last-bit disagreement flipped which move came out. It was
    removed rather than tolerated.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "name", ["chess", "gardner", "rps-v2", "shogi", "crazyhouse", "xiangqi", "horde"]
    )
    def test_every_term_plays_the_same_game_on_both_sides(self, name: str) -> None:
        from chess_variant_sandbox.tournament.play import play_game, play_game_python

        engine = Engine(load(name))
        for index, term in enumerate(VOCABULARY):
            weights = [0.0] * len(VOCABULARY)
            weights[index] = 1.0
            digest = register_heuristic(Heuristic(f"only-{term}", term, tuple(weights)))
            agent = f"greedy:h={digest}"
            through_core = play_game(engine, (agent, agent), seed=3, collect_series=False)
            through_python = play_game_python(engine, (agent, agent), seed=3, collect_series=False)
            assert through_core.moves == through_python.moves, (
                f"{name}: the core and the oracle disagree about {term}"
            )
            assert through_core.winner == through_python.winner
            assert through_core.reason == through_python.reason

    def test_a_vector_of_the_wrong_length_is_refused_at_the_boundary(self) -> None:
        """A heuristic from another vocabulary must not be reinterpreted as this one.

        The agent is the boundary's whole shape, twelve fields since Q502i. It used to be seven
        wide, so PyO3 refused the tuple's length before the core ever saw the vector, and
        a bare `ValueError` made that pass for the vocabulary check (Q142).
        """
        engine = Engine(load("gardner"))
        from chess_variant_sandbox.tournament.play import eval_values

        bad = ("greedy", 0, 0.05, 0, 0, 0.0, 0.0, (1.0, 2.0), (), 0.0, b"", 0)
        with pytest.raises(ValueError, match=f"expected {len(VOCABULARY)}, got 2"):
            engine.select_move(engine.initial_state(), bad, eval_values(engine.spec), 0)


class TestTheProposerEndToEnd:
    """Everything about the model stage except the network call.

    Stage 23 wants the proposer *validated* -- run against a real model and scored against
    the hand-written and random controls -- and that needs credentials. This is the part
    that does not: a stub client returns a canned response and the whole path runs, so
    "never been run" becomes "run end to end, with only the request unexercised".

    Worth having on its own terms. No LLM stage in this project has an offline test, and
    the failure modes here are the ordinary ones -- a prompt that does not assemble, a
    schema the validator disagrees with, a response that parses but does not become
    players -- none of which needs a model to catch.
    """

    class _Stub:
        """Answers `structured` from a fixed payload, and records what it was asked."""

        model = "stub"

        def __init__(self, payload: dict, expect_k: int) -> None:
            self.payload = payload
            self.expect_k = expect_k
            self.calls: list[tuple] = []

        def structured(self, system, user, schema, max_tokens=None):  # noqa: ANN001
            self.calls.append((system, user, schema))
            return self.payload

    @staticmethod
    def _payload(k: int) -> dict:
        plans = []
        for index in range(k):
            weights = [0.0] * len(VOCABULARY)
            # Distinct directions, so no two collapse to one player.
            weights[index % len(VOCABULARY)] = 1.0
            weights[(index + 1) % len(VOCABULARY)] = 0.25 + 0.1 * index
            plans.append({"name": f"plan-{index}", "plan": "do a thing", "weights": weights})
        return {"heuristics": plans}

    def test_a_valid_response_becomes_a_registered_set(self) -> None:
        from chess_variant_sandbox.llm.heuristics import propose_heuristics

        engine = Engine(load("chess"))
        stub = self._Stub(self._payload(5), 5)
        result = propose_heuristics(load("chess"), engine.spec, k=5, client=stub)

        # The anchor is prepended and is never proposed, so k+1 come back.
        assert result.k == 6
        assert result.heuristics[0].digest == anchor(engine.spec).digest
        assert result.ruleset_hash == engine.spec.hash
        assert result.model == "stub"
        # And every one of them actually plays.
        result.register()
        for heuristic in result.heuristics:
            assert weights_for({"h": heuristic.digest}) == heuristic.weights

    def test_the_prompt_carries_the_rules_and_the_vocabulary(self) -> None:
        from chess_variant_sandbox.llm.heuristics import propose_heuristics

        engine = Engine(load("rps-v2"))
        stub = self._Stub(self._payload(5), 5)
        propose_heuristics(load("rps-v2"), engine.spec, k=5, client=stub)
        system, user, schema = stub.calls[0]

        stable = system[0]["text"] if isinstance(system, list) else system
        assert "territory_margin" in stable, "the vocabulary is not in the cached prefix"
        # The rules are per-candidate and must be after the cache breakpoint.
        assert "board" in user.lower()
        assert schema["properties"]["heuristics"]["minItems"] == 5

    def test_an_invalid_response_gets_exactly_one_corrective_retry(self) -> None:
        """The `propose.py` contract: one retry carrying the errors, then give up."""
        from chess_variant_sandbox.agents.heuristics import HeuristicError
        from chess_variant_sandbox.llm.heuristics import propose_heuristics

        engine = Engine(load("chess"))
        broken = {"heuristics": [{"name": "z", "plan": "", "weights": [0.0] * len(VOCABULARY)}]}
        stub = self._Stub(broken, 1)
        with pytest.raises(HeuristicError, match="after one correction"):
            propose_heuristics(load("chess"), engine.spec, k=1, client=stub)

        assert len(stub.calls) == 2, "expected exactly one retry"
        retry_user = stub.calls[1][1]
        assert "rejected" in retry_user
        assert "moves at random" in retry_user, "the retry must carry the specific error"


class TestPieceValues:
    """Stage 30A: what a player is allowed to think the pieces are worth.

    The failure this whole group is written against is not arithmetic. It is two
    ``tuple[float, ...]`` of identical shape and different meaning being swapped at a
    call site, whose symptom is a slightly different player rather than an exception --
    so the tests here are about *which table each reader got*, not about whether the
    sums are right.
    """

    def spec(self, name: str = "xiangqi") -> Any:
        return Engine(load(name)).spec

    def derived(self, name: str = "xiangqi") -> tuple[float, ...]:
        return Evaluator.for_spec(self.spec(name)).values

    # -- the type ------------------------------------------------------------

    def test_no_override_puts_the_one_table_in_both_slots(self) -> None:
        """The built-in-shaped case has to be the arithmetic it always was.

        Not an equal copy: ``material`` has three branches and float addition is not
        associative, so ``FeatureEvaluator`` checks identity rather than equality when
        it decides whether to build a second evaluator at all.
        """
        derived = self.derived()
        pair = PieceValues.of(derived, (), self.spec().royal_types)
        assert pair.objective is pair.derived is derived

    def test_an_override_is_rescaled_to_the_derived_tables_mean(self) -> None:
        """The rule the whole stage turns on, on the ruleset that would have caught it.

        Xiangqi's derived table runs from 1.25 to 74, so an override normalised to mean
        1 -- the tempting convention, and the unit ``GOAL_PIECES`` is stated in -- would
        make ``_material`` about a thirtieth of its usual size there. Every other test
        in this file would pass.
        """
        spec = self.spec()
        derived = self.derived()
        # A reallocation a person might actually state: the region-locked defenders are
        # worth something, and the cannon rather less than the chariot.
        override = (3.0, 5.0, 4.0, 4.0, 9.0, 4.5, 0.0)
        pair = PieceValues.of(derived, override, spec.royal_types)
        assert mean_abs_non_royal(pair.objective, spec.royal_types) == pytest.approx(
            mean_abs_non_royal(derived, spec.royal_types), rel=1e-12
        )
        # A reallocation, not a volume knob: five times every slider is one opponent.
        louder = PieceValues.of(derived, tuple(5 * v for v in override), spec.royal_types)
        assert louder.objective == pytest.approx(pair.objective, rel=1e-12)

    def test_the_override_keeps_material_at_the_size_it_already_had(self) -> None:
        """The bug the mean-1 convention would have shipped, stated as a magnitude.

        ``_material`` divides by ``total_material``, which is derived and stays derived,
        so the two are only comparable if the numerator is on the derived scale. Read
        off a board with one piece missing, because the opening is symmetric and both
        readings would be zero there.
        """
        spec = self.spec()
        derived = self.derived()
        squares = bytearray(spec.initial_board)
        for square, value in enumerate(squares):
            # Any piece belonging to the second player, so the sides are unbalanced.
            if value and (value - 1) // spec.n_types == 1:
                squares[square] = 0
                break
        board = bytes(squares)

        context = feature_context(spec)
        plain = context.base.board_material(board, 0) / context.total_material
        override = (3.0, 5.0, 4.0, 4.0, 9.0, 4.5, 0.0)
        pair = PieceValues.of(derived, override, spec.royal_types)
        valued = replace(context.base, values=pair.objective).board_material(board, 0)
        valued /= context.total_material

        assert plain != 0.0
        # The same order of magnitude, which is the whole claim.
        assert 0.1 < abs(valued / plain) < 10.0

    # -- what reads which ----------------------------------------------------

    def test_ordering_and_the_normaliser_never_see_the_override(self) -> None:
        """The confusion ``PieceValues`` exists to prevent, asserted rather than typed.

        A type stops the two tables being passed to the wrong parameter; it cannot say
        that a call site picked the right one. Capture ordering is not the objective and
        ``total_material`` is what puts every term in the vocabulary on one scale, so
        both stay derived whatever a player thinks.
        """
        spec = self.spec()
        derived = self.derived()
        override = (3.0, 5.0, 4.0, 4.0, 9.0, 4.5, 0.0)
        context = feature_context(spec)
        before = context.total_material

        evaluator = FeatureEvaluator(
            context,
            vector(material=1.0),
            PieceValues.of(derived, override, spec.royal_types),
        )
        assert evaluator.values == derived
        assert evaluator.piece_values.derived == derived
        assert evaluator.piece_values.objective != derived
        assert context.total_material == before

    def test_a_zero_material_weight_makes_an_override_inert(self) -> None:
        """``_active`` skips a zero weight because float addition is not associative.

        So a zero ``material`` weight means *absent*, and a values override carried
        beside one is inert by construction. Said here rather than left to be discovered
        by somebody wondering why their sliders did nothing.
        """
        spec = self.spec()
        derived = self.derived()
        override = (3.0, 5.0, 4.0, 4.0, 9.0, 4.5, 0.0)
        context = feature_context(spec)
        engine = Engine(load("xiangqi"))
        state = engine.initial_state()
        state = engine.apply(state, engine.legal_moves(state)[0])

        weights = vector(mobility=1.0, advance=0.5)
        plain = FeatureEvaluator(context, weights)
        valued = FeatureEvaluator(
            context, weights, PieceValues.of(derived, override, spec.royal_types)
        )
        assert plain.evaluate(engine, state, 0) == valued.evaluate(engine, state, 0)

    # -- what is refused -----------------------------------------------------

    def test_a_royal_piece_may_not_be_priced(self) -> None:
        """``ROYAL_VALUE = 0.0`` is load-bearing and this is the door back in.

        ``_reach_is_priceable`` records what a royal term nothing opposes does: king of
        the hill collapses from 54.7 mean plies to 20 and ``quick_kill_rate`` from 0.041
        to over 0.54, at every weight tested. A settable royal value is that from the
        other direction, so it is refused at construction rather than zeroed.
        """
        spec = self.spec()
        override = [1.0] * spec.n_types
        override[spec.royal_types[0]] = 2.0
        with pytest.raises(ValueError, match="royal"):
            PieceValues.of(self.derived(), tuple(override), spec.royal_types)

    def test_a_table_written_for_another_game_is_refused(self) -> None:
        with pytest.raises(ValueError, match="different game"):
            PieceValues.of(self.derived(), (1.0, 2.0), self.spec().royal_types)

    def test_a_heuristic_bounds_the_length_without_a_board(self) -> None:
        """``Heuristic`` holds a vector and no board, so it can only bound the length.

        The exact length is a fact about a ruleset and is checked where a spec is in
        hand. What this class can refuse is a table longer than any ruleset may have,
        one that is not finite, and one that prices nothing at all.
        """
        with pytest.raises(HeuristicError, match="at most 24"):
            Heuristic(name="x", plan="", weights=vector(material=1.0), values=(1.0,) * 25)
        with pytest.raises(HeuristicError, match="finite"):
            Heuristic(
                name="x", plan="", weights=vector(material=1.0), values=(1.0, float("inf"))
            )
        with pytest.raises(HeuristicError, match="absence of one"):
            Heuristic(name="x", plan="", weights=vector(material=1.0), values=(0.0, 0.0))

    def test_values_with_no_weights_are_refused_rather_than_ignored(self) -> None:
        """An override is half of a heuristic. The other half names the objective.

        With no weights the built-in evaluator runs, and it *is* the spec-derived
        valuation -- so an override there would be accepted and then quietly not used,
        which is the shape of failure this project treats as worse than a crash.
        """
        engine = Engine(load("xiangqi"))
        with pytest.raises(ValueError, match="half of a heuristic"):
            objective_for(engine, 0.05, values=(1.0,) * engine.spec.n_types)

    # -- identity ------------------------------------------------------------

    def test_the_digest_covers_the_values_only_when_there_are_any(self) -> None:
        """Both halves of the convention, and the second keeps the corpus valid.

        Two players that price the pieces differently must not share a name. And a
        player with no opinion must hash exactly as it did before this field existed, or
        every committed set fails ``from_dict``'s edited-file check at once.
        """
        weights = vector(material=1.0, mobility=0.05)
        plain = Heuristic(name="a", plan="", weights=weights)
        # The digest as it was computed before stage 30, spelled out rather than
        # imported: the point is that the old definition still names this player.
        legacy = hashlib.sha256(
            json.dumps(
                {
                    "vocabulary_version": VOCABULARY_VERSION,
                    "weights": list(plain.normalised),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        assert plain.digest == legacy
        assert "values" not in plain.as_dict()

        one = Heuristic(name="a", plan="", weights=weights, values=(1.0, 2.0, 0.0))
        two = Heuristic(name="a", plan="", weights=weights, values=(2.0, 1.0, 0.0))
        assert len({plain.digest, one.digest, two.digest}) == 3

    def test_a_set_carried_across_a_vocabulary_bump_is_the_player_it_was(self) -> None:
        """What made padding the committed stores honest rather than convenient.

        `facing_alignment` was appended to a fixed-length vocabulary, so every stored
        fourteen-weight set is refused by its own length. The 34 committed sets were
        carried across with `0.0` in the new slot rather than regenerated -- regenerating
        would have cost model calls the project has already decided not to spend, and
        orphaning them would have cost the picker its rules of thumb on 28 rulesets.

        The claim that makes the padding the identity is here, in two halves. A zero
        weight is **dropped** from `_active` rather than multiplied by, so the new term is
        never called and no float in `evaluate` is perturbed by it; and every committed
        set actually carries a zero there rather than something small. Neither half is
        worth anything without the other.
        """
        from chess_variant_sandbox.agents.features import (
            VOCABULARY,
            FeatureEvaluator,
            feature_context,
        )

        engine = Engine(load("chess"))
        context = feature_context(engine.spec)
        index = VOCABULARY.index("facing_alignment")

        padded = vector(material=1.0, mobility=0.05)
        assert padded[index] == 0.0
        assert index not in {slot for slot, _ in FeatureEvaluator(context, padded)._active}

        root = Path(__file__).resolve().parent.parent / "heuristics"
        files = sorted(root.glob("*.json")) + sorted(root.glob("*/*.json"))
        assert len(files) >= 28, f"only {len(files)} stores found; the check is vacuous"
        for path in files:
            for heuristic in HeuristicSet.load(path).heuristics:
                assert heuristic.weights[index] == 0.0, (path.name, heuristic.name)

    def test_every_committed_store_still_loads_and_hashes_as_stored(self) -> None:
        """The check that says this change cost nothing already on disk.

        ``Heuristic.from_dict`` refuses a file whose stated digest does not match what
        its weights hash to, which is what makes a store a cache with a verified key.
        Adding a field to the hashed blob unconditionally would have failed every one of
        these, and the count is asserted so the check cannot go vacuous.
        """
        root = Path(__file__).resolve().parent.parent / "heuristics"
        files = sorted(root.glob("*.json")) + sorted(root.glob("*/*.json"))
        assert len(files) >= 28, f"only {len(files)} stores found; the check is vacuous"
        for path in files:
            stored = json.loads(path.read_text(encoding="utf-8"))
            found = HeuristicSet.load(path)
            assert found.set_hash == stored["set_hash"]
            for heuristic, written in zip(found.heuristics, stored["heuristics"], strict=True):
                assert heuristic.digest == written["digest"]

    def test_a_store_that_outlived_its_ruleset_diagnoses_itself(self) -> None:
        """The ``piece_ids`` echo, and the reason it must refuse only when it can.

        A value vector is positional and says nothing about which piece each slot is, so
        a set filed against a ruleset whose pieces have changed means something else in
        silence. It is written only beside a values array and checked only when the
        caller supplies what the ruleset actually declares -- which is what keeps every
        existing file, none of which states the key, loading unchanged.
        """
        spec = self.spec()
        priced = Heuristic(
            name="a", plan="", weights=vector(material=1.0), values=(1.0,) * spec.n_types
        )
        stored = HeuristicSet(
            ruleset_hash=spec.hash, heuristics=(priced,), piece_ids=spec.piece_ids
        ).as_dict()
        assert stored["piece_ids"] == list(spec.piece_ids)
        assert HeuristicSet.from_dict(stored, spec.piece_ids).piece_ids == spec.piece_ids
        with pytest.raises(HeuristicError, match="different game"):
            HeuristicSet.from_dict(stored, ("Q", "K"))

        # No claim on either side is not a mismatch, which is what every file on disk
        # relies on.
        bare = HeuristicSet(
            ruleset_hash=spec.hash,
            heuristics=(Heuristic(name="a", plan="", weights=vector(material=1.0)),),
        ).as_dict()
        assert "piece_ids" not in bare
        assert HeuristicSet.from_dict(bare, spec.piece_ids).piece_ids == ()


class TestThePass:
    """`chess-variant-sandbox heuristics`, the pass's one caller, and what it measures
    (Q751, archive/plan-tests F9: `measure_heuristics` and every helper under it had never run).

    On `dull` (one piece type, a small board), where the first rung of `NODE_LADDER`
    already clears `MIN_DEPTH`, so the whole pass is a few dozen short games."""

    ARGS = ("--k", "3", "--games", "1", "--workers", "1")

    def test_the_command_writes_the_set_and_the_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from chess_variant_sandbox import cli
        from chess_variant_sandbox.tournament.heuristics import (
            MIN_DEPTH,
            NODE_LADDER,
            HeuristicReport,
        )

        spec = RULESETS / "dull.json"
        out = tmp_path / "report.json"
        argv = ["heuristics", str(spec), *self.ARGS, "--store", str(tmp_path / "sets"),
                "--out", str(out)]
        assert cli.main(argv) == 0
        printed = capsys.readouterr().out
        compiled = Engine(load(spec.stem)).spec

        assert (tmp_path / "sets" / f"{compiled.hash}.json").is_file()
        stored = json.loads(out.read_text(encoding="utf-8"))
        report = HeuristicReport.load(out)
        assert report.as_dict() == stored
        assert report.ruleset_hash == compiled.hash
        assert report.k == 3 and report.games_per_pairing == 1
        assert "baseline" not in stored
        # The budget is chosen, not given: the smallest rung at which the deep arm is deep.
        assert report.nodes == NODE_LADDER[0]
        assert report.depth_reached >= MIN_DEPTH and report.measurable
        # Index 0 is the built-in objective, which no sample produced.
        assert report.rows[0].digest == anchor(compiled).digest
        for row, saved in zip(report.rows, stored["rows"], strict=True):
            for score in (row.deep_score, row.legibility, row.agreement_shallow,
                          row.agreement_deep):
                assert 0.0 <= score <= 1.0
            assert saved["headroom"] == round(max(0.0, min(1.0, 2 * (row.deep_score - 0.5))), 4)
            assert saved["agreement"] == round(min(row.agreement_shallow, row.agreement_deep), 4)
        assert f"at {NODE_LADDER[0]} nodes" in printed
        for name, value in heuristic_components(report).items():
            assert f"{name:<28}{value}" in printed

        # A second run reads the committed set back rather than writing another.
        assert cli.main(argv) == 0
        assert "reusing" in capsys.readouterr().out
        assert HeuristicReport.load(out).as_dict() == stored

    def test_the_report_does_not_depend_on_the_worker_count(self) -> None:
        from chess_variant_sandbox.tournament.heuristics import measure_heuristics

        spec = load("dull")
        chosen = handwritten_set(Engine(spec).spec, k=2)
        one = measure_heuristics(spec, chosen, games=1, nodes=6_400, workers=1)
        two = measure_heuristics(spec, chosen, games=1, nodes=6_400, workers=2)
        assert one.as_dict() == two.as_dict()

    def test_the_budget_climbs_the_ladder_until_the_deep_arm_is_deep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chess_variant_sandbox.tournament import heuristics as passing

        engine = Engine(load("dull"))
        chosen = handwritten_set(engine.spec, k=1)
        tried: list[int] = []

        def depth(at: dict[int, float]) -> None:
            tried.clear()

            def reached(_engine: Any, _heuristic: Any, nodes: int, _seed: int) -> float:
                tried.append(nodes)
                return at[nodes]

            monkeypatch.setattr(passing, "_depth_reached", reached)

        low, high, top = passing.NODE_LADDER
        depth({low: 2.0, high: passing.MIN_DEPTH, top: 9.0})
        assert passing._choose_budget(engine, chosen, 0) == high
        assert tried == [low, high]  # it stops at the first rung that is deep enough
        # A ruleset no rung can search ends on the top rung, which the report then calls
        # unmeasured (`measurable`) rather than shallow.
        depth({low: 1.0, high: 2.0, top: 3.0})
        assert passing._choose_budget(engine, chosen, 0) == top

    def test_the_shallow_arm_is_two_plies_on_any_board(self) -> None:
        """The yardstick's claim: `SHALLOW_NODES` is always enough to complete
        `SHALLOW_DEPTH`, so the arm is the same length on every board."""
        from chess_variant_sandbox.tournament.heuristics import (
            SHALLOW_DEPTH,
            _shallow,
            plies_completed,
        )

        for name in ("dull", "gardner", "dobutsu"):
            engine = Engine(load(name))
            assert plies_completed(engine, _shallow(None), 0) == SHALLOW_DEPTH, name

    def test_the_agreement_arms_differ_only_in_their_objective(self) -> None:
        from chess_variant_sandbox.tournament.heuristics import _shallow

        plain, digest = _shallow(None), _shallow("abc")
        assert plain == "minimax:max_depth=2,nodes=20000"
        assert digest == "minimax:h=abc,max_depth=2,nodes=20000"

    def test_the_diagnostics_say_what_the_components_leave_out(self) -> None:
        from chess_variant_sandbox.tournament.heuristics import (
            HeuristicReport,
            MatchRow,
            heuristic_diagnostics,
        )

        def row(plan: str, deep: float, legible: float, agrees: float) -> MatchRow:
            return MatchRow(plan, plan, plan, deep_score=deep, legibility=legible,
                            agreement_shallow=agrees, agreement_deep=1.0)

        rows = [row("built-in", 0.6, 0.7, 0.5), row("a", 1.0, 0.9, 0.8), row("b", 0.5, 0.4, 0.2)]
        report = HeuristicReport("h", "s", nodes=6_400, depth_reached=5.0, rows=rows)
        said = heuristic_diagnostics(report)
        assert said["k"] == 3 and said["measurable"] is True
        assert (said["headroom_best"], said["headroom_worst"], said["headroom_builtin"]) == (
            1.0, 0.0, 0.2)
        assert said["best_plan"] == "a" and said["beats_random"] == 0.9
        assert said["headroom_legibility_correlation"] == round(
            statistics.correlation([0.2, 1.0, 0.0], [0.7, 0.9, 0.4]), 3)
        # Under three rows, or nothing varying, there is no correlation to report.
        assert heuristic_diagnostics(
            HeuristicReport("h", "s", 6_400, 5.0, rows[:2]))["headroom_legibility_correlation"] is None
        flat = [row("x", 0.6, 0.5, 0.1), row("y", 0.9, 0.5, 0.1), row("z", 0.7, 0.5, 0.1)]
        assert heuristic_diagnostics(
            HeuristicReport("h", "s", 6_400, 5.0, flat))["headroom_legibility_correlation"] is None
        assert heuristic_diagnostics(None) == {}
