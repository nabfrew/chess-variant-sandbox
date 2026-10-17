"""Schema 10: five features, one version, and all five inert by default.

The bump is Q84's central recommendation (`docs/game-ideas.md`, *M2's nine primitives,
costed against each other*): the five M2 primitives that move **no committed perft at all**
share one version number, one collector, one inertness proof and one bench, because
splitting them would buy five inertness runs and four rebases of the same collector.

What is asserted here, in the order the features were written:

* **Q38** -- `rules.repetition_draw` may say `"never"`. The one clock value the DSL could
  not express, and the reason `intransitive`'s committed spec misstates its own designer.
  It costs the core nothing: *never* compiles to a threshold a repetition count cannot
  reach, so the test at `game.rs:2619` can never fire and there is no second branch on the
  hot path.
* **Q78** -- `npcs[].immune`, a wild piece only some types may take. A pure *re-spelling*
  of the schema-3 capture matrix, and its acceptance test is the cheapest of the nine:
  compile `archive/wolves.json` and its one-key re-spelling and assert the tables are the
  same bytes.
* **Q18** -- `explode_on_capture.by` / `.of`, which narrow *when* the blast fires rather
  than what it reaches. Two boolean tables, and `atomic.json` must not move -- which is
  what the empty-list-means-any encoding at the boundary is for.

**The inertness proof is not here.** It is
`test_schema_nine.py::TestTheBumpMovedNothing`, which pins `(mechanics hash, tables
fingerprint)` for every committed and archived ruleset: schema 10 moves neither, so that
table is the mechanical check and this file would only duplicate it. What is here is the
other half -- that each feature, when a spec *does* state it, changes the game, is gated
against an older stamp, and reaches every surface `new-move-primitive` names.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.evaluation import (
    DECLARED_WEIGHT,
    GOAL_PIECES,
    Evaluator,
    _mean_non_royal,
    has_goal_terms,
    inexhaustible_types,
)
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import NO_REPETITION_DRAW, compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_DROP,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.loop.operators import OPERATORS, mutate
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import _SCALES, distance, features
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import (
    MAX_STOCK,
    REPETITION_NEVER,
    SCHEMA_VERSION,
    STOCK_UNLIMITED,
    _schema_ten_uses,
    piece_has_stock,
    portal_end_ids,
    portal_groups,
    portal_piece_ids,
)

ROOT = Path(__file__).resolve().parents[1]
# Every committed and archived spec, for the walks that are one test per spec (Q749).
CORPUS_PATHS = sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))


@pytest.fixture(scope="module")
def chess() -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


def _never(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with the draw by repetition removed, stamped for schema 10."""
    child = copy.deepcopy(spec)
    child["schema_version"] = 10
    child.setdefault("rules", {})["repetition_draw"] = REPETITION_NEVER
    return child


def _wolves() -> dict[str, Any]:
    """`archive/wolves.json` as committed: five `captures` lists and no immunity."""
    return json.loads((ROOT / "archive" / "wolves.json").read_text(encoding="utf-8"))


def _hunted(spec: dict[str, Any]) -> dict[str, Any]:
    """`wolves` re-spelled the schema-10 way: one key instead of five lists.

    The exact edit Q84 costed -- drop the `captures` list from every piece that is not the
    knight, and say `"immune": ["N"]` on each wild piece instead.
    """
    child = copy.deepcopy(spec)
    child["schema_version"] = 10
    for piece in child["pieces"]:
        if piece["id"] != "N":
            piece.pop("captures", None)
    for npc in child["rules"]["npcs"]:
        npc["immune"] = ["N"]
    return child


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _move(engine: Engine, state: Any, origin: str, target: str) -> int:
    """The one legal move from ``origin`` to ``target``, by name rather than by index."""
    wanted = (_square(origin, engine.spec.width), _square(target, engine.spec.width))
    found = [
        move
        for move in engine.legal_moves(state)
        if (move_from(move), move_to(move)) == wanted
    ]
    assert len(found) == 1, f"{origin}{target}: {len(found)} moves"
    return found[0]


class TestTheVersionIsBumped:
    def test_the_schema_version_is_ten(self) -> None:
        # Schema 11 (Q19b) came after, so ten is a floor here rather than the number.
        assert SCHEMA_VERSION >= 10

    def test_the_collector_names_what_the_bump_added(self, chess: dict[str, Any]) -> None:
        """The bump's inventory, which everything downstream is stated against.

        Q84 found that only one of schema 10's five features is indexable by a
        `V10_RULE_KEYS` frozenset at all -- the rest are a new value, nested keys and a
        relaxation -- so the gate is a collector and *this* is the list a showcase audit,
        `describe.py` and the version message are all measured against.
        """
        assert _schema_ten_uses(chess) == set()
        assert _schema_ten_uses(_never(chess)) == {'rules.repetition_draw "never"'}
        assert _schema_ten_uses(_hunted(_wolves())) == {"rules.npcs[].immune"}

    def test_two_features_under_an_old_stamp_are_both_named_at_once(
        self, chess: dict[str, Any]
    ) -> None:
        """One collector rather than one `if` per feature, which is what it buys.

        A proposal told about one schema-10 feature at a time fixes that one and
        reintroduces the other -- the reason `SpecError` carries every problem.
        """
        spec = _hunted(_wolves())
        spec["rules"]["repetition_draw"] = REPETITION_NEVER
        assert _schema_ten_uses(spec) == {
            'rules.repetition_draw "never"',
            "rules.npcs[].immune",
        }
        spec["schema_version"] = 9
        errors = [e for e in validate_spec(spec) if "schema 10" in e]
        assert len(errors) == 1
        assert 'rules.repetition_draw "never"' in errors[0]
        assert "rules.npcs[].immune" in errors[0]


# ------------------------------------------------------- Q38: the clock that says never


class TestRepetitionDrawMayBeNever:
    def test_the_validator_accepts_it_under_ten(self, chess: dict[str, Any]) -> None:
        assert validate_spec(_never(chess)) == []

    def test_an_older_stamp_is_refused_by_name(self, chess: dict[str, Any]) -> None:
        spec = _never(chess)
        spec["schema_version"] = 9
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert 'rules.repetition_draw "never"' in errors[0]
        assert "schema 10" in errors[0]

    def test_only_that_word_and_only_that_clock(self, chess: dict[str, Any]) -> None:
        """The sentinel is one word on one key, and everything else stays refused.

        `max_plies` is what stops a game at all and `no_progress_draw` at a large integer
        is a game the corpus already has, so neither is widened. A misspelling has to be
        an error rather than a clock of some other length.
        """
        for key in ("max_plies", "no_progress_draw"):
            spec = copy.deepcopy(chess)
            spec["schema_version"] = 10
            spec["rules"][key] = REPETITION_NEVER
            assert any(key in error for error in validate_spec(spec)), key

        for value in ("Never", "none", "off", 0, True):
            spec = copy.deepcopy(chess)
            spec["schema_version"] = 10
            spec["rules"]["repetition_draw"] = value
            assert validate_spec(spec), value

    def test_it_compiles_to_a_threshold_that_can_never_be_reached(
        self, chess: dict[str, Any]
    ) -> None:
        """Q84's finding, asserted rather than restated: no `rust/src/` change.

        The core tests `repetition_count() >= t.repetition_draw`. A `u32::MAX` threshold
        makes that test dead code for this ruleset, and every ruleset that states a number
        compiles to exactly the number it stated.
        """
        assert compile_spec(_never(chess)).repetition_draw == NO_REPETITION_DRAW
        assert compile_spec(chess).repetition_draw == 3
        assert build_tables(compile_spec(_never(chess)))["repetition_draw"] == (
            NO_REPETITION_DRAW
        )
        assert rust_supports(compile_spec(_never(chess))) is None

    def test_the_core_plays_the_removal(self, chess: dict[str, Any]) -> None:
        """A shuffle that draws under a threshold of 2 and does not draw under *never*.

        The play-test `new-move-primitive` asks for: a game from a real start reaching a
        position where the feature changes an outcome. Both engines are handed the same
        eight plies, so the only difference is the rule.
        """
        shuffle = [("g1", "f3"), ("g8", "f6"), ("f3", "g1"), ("f6", "g8")]

        drawing = copy.deepcopy(chess)
        drawing["rules"]["repetition_draw"] = 2
        outcomes = []
        for spec in (drawing, _never(chess)):
            engine = Engine(spec)
            state = engine.initial_state()
            verdict = None
            for _ in range(2):
                for origin, target in shuffle:
                    state = engine.apply(state, _move(engine, state, origin, target))
                    verdict = engine.terminal(state)
                    if verdict is not None:
                        break
                if verdict is not None:
                    break
            outcomes.append(verdict)

        drawn, never = outcomes
        assert drawn is not None and drawn.winner is None
        assert "repetition" in drawn.reason
        assert never is None

    def test_the_description_says_never_rather_than_a_number(
        self, chess: dict[str, Any]
    ) -> None:
        """The judge, the proposer and a player mid-game read this.

        A threshold of 4294967295 rendered as a number is a sentence that is technically
        true and tells a reader nothing, so the clause moves out of the list.
        """
        prose = describe_spec(_never(chess))
        assert "Repeating a position is never a draw here." in prose
        assert str(NO_REPETITION_DRAW) not in prose
        assert "The game is drawn after 3 repetitions" in describe_spec(chess)

    def test_it_has_a_distance_feature_of_its_own(self, chess: dict[str, Any]) -> None:
        """Surface 7, and it closes a gap older than the feature.

        `distance.py` had no clock feature at all, so two specs differing only in
        `repetition_draw` reported distance zero. Harmless at 3 against 4; not harmless
        once the value can remove the rule, because the loop would treat a game with no
        draw by repetition as a rediscovery of one with it and skip the tournament.
        """
        assert "no_repetition_draw" in _SCALES
        assert features(chess)["no_repetition_draw"] == 0.0
        assert features(_never(chess))["no_repetition_draw"] == 1.0
        assert distance(chess, _never(chess)) > 0.0

        # A different *number* is still the same kind of game, deliberately.
        four = copy.deepcopy(chess)
        four["rules"]["repetition_draw"] = 4
        assert distance(chess, four) == 0.0

    def test_it_is_still_bookkeeping_in_the_cost_table(self, chess: dict[str, Any]) -> None:
        """A clock costs nothing, and removing one does not start costing something.

        `_RULE_COSTS["repetition_draw"]` is 0.0 because every game needs a way to stop and
        nobody learns the number. *Never* is the same key at a different setting, not a new
        exception a player is taught on top -- and the sentence they do have to be told is
        already carried by `describe.py`.
        """
        assert (
            measure_complexity(_never(chess)).rule_mass
            == measure_complexity(chess).rule_mass
        )

    def test_fairy_writes_it_as_no_fold_rule(self, chess: dict[str, Any]) -> None:
        """Q94's warning, which no key guard could have raised, and Q703's translation.

        `repetition_draw` is in `_RULES_TRANSLATED`, so the exhaustive catch-all stopped
        applying to it long ago, and a `_rule_lines` that wrote the compiled value would
        have written `nFoldRule = 4294967295` -- a variant definition stating a different
        game, with `verify.py` still comparing move lists that agreed. It was refused
        until Q703 found `nFoldRule = 0` switches the rule off there.
        """
        assert "nFoldRule = 3" in translate(compile_spec(chess)).config
        written = translate(compile_spec(_never(chess)))
        assert "nFoldRule = 0" in written.config
        assert str(NO_REPETITION_DRAW) not in written.config

    def test_the_json_schema_offers_it_to_a_proposer(self) -> None:
        """Otherwise the rule is in the schema and cannot be proposed at all."""
        clock = rule_spec_schema()["properties"]["rules"]["properties"]["repetition_draw"]
        assert {"const": REPETITION_NEVER} in clock["anyOf"]

    def test_an_operator_can_cross_the_axis_in_both_directions(
        self, chess: dict[str, Any]
    ) -> None:
        """Surface 8: a rule that reaches a new kind of game needs an operator that gets there.

        The old body could not: `max(2, int(current * scale))` has a floor at 2 and no
        multiplier reaches *never*, and coming back the sentinel is not an `int` at all, so
        the operator returned `None` and the axis was one-way by accident.
        """
        limit = next(op for op in OPERATORS if op.name == "limit-the-game")

        def reachable(start: dict[str, Any], wanted: Any) -> bool:
            for seed in range(80):
                child = copy.deepcopy(start)
                if limit.apply(child, random.Random(seed)) is None:
                    continue
                if child["rules"]["repetition_draw"] == wanted:
                    assert validate_spec(child) == [], child["rules"]["repetition_draw"]
                    return True
            return False

        assert reachable(chess, REPETITION_NEVER)
        assert reachable(_never(chess), 3) or reachable(_never(chess), 2)

    def test_the_stamp_settles_itself_on_both_crossings(self, chess: dict[str, Any]) -> None:
        """`settle_schema_version` raises the stamp on the way in and lowers it on the way out.

        Without it the operator would produce a spec using a schema-10 value under the
        parent's older stamp, which the validator refuses -- so the axis would exist and
        no mutation could cross it.
        """
        limit = next(op for op in OPERATORS if op.name == "limit-the-game")
        for seed in range(80):
            child = copy.deepcopy(chess)
            if limit.apply(child, random.Random(seed)) is None:
                continue
            if child["rules"]["repetition_draw"] == REPETITION_NEVER:
                assert child["schema_version"] == 10
                back = copy.deepcopy(child)
                for inner in range(80):
                    candidate = copy.deepcopy(back)
                    if limit.apply(candidate, random.Random(inner)) is None:
                        continue
                    if candidate["rules"]["repetition_draw"] != REPETITION_NEVER:
                        assert candidate["schema_version"] < 10
                        return
                pytest.fail("no mutation came back from 'never'")
        pytest.fail("no mutation reached 'never'")

    def test_a_mutated_child_is_a_different_game_and_a_valid_one(
        self, chess: dict[str, Any]
    ) -> None:
        """End to end through `mutate`, which is what the loop actually calls."""
        for seed in range(200):
            child = mutate(chess, random.Random(seed))
            if child is None:
                continue
            if child.spec["rules"].get("repetition_draw") == REPETITION_NEVER:
                assert validate_spec(child.spec) == []
                assert spec_hash(child.spec) != spec_hash(chess)
                return
        pytest.skip("no seed in range reached the sentinel through mutate()")


# ------------------------------------------------- Q78: a wild piece only some may take


class TestNpcsMayBeImmune:
    def test_the_re_spelling_compiles_to_the_same_tables(self) -> None:
        """The acceptance test Q84 named, and the cheapest of the nine.

        `npcs[].immune` is a different spelling of the schema-3 capture matrix rather than
        a second mechanism, so *only a knight may take a wolf* has to compile to the
        matrix five `captures` lists already produced -- byte for byte, not nearly.

        The Zobrist draw is the one exception and it is expected: the seed is the
        mechanics hash, the hash covers the spelling, so a shorter spec lays its
        transposition table out differently. Same game, same distribution, not the same
        individual games -- which is Q84's own caveat, asserted here rather than assumed.
        """
        first = build_tables(compile_spec(_wolves()))
        second = build_tables(compile_spec(_hunted(_wolves())))
        assert first["capture_allowed"] == second["capture_allowed"]
        differing = {key for key in first if first[key] != second[key]}
        assert differing == {
            "zobrist_pieces",
            "zobrist_side",
            "zobrist_ep",
            "zobrist_colour",
            "zobrist_turn",
        }

    def test_it_is_the_elegance_win_it_was_costed_as(self) -> None:
        """Five `captures` lists gone, and the spec is shorter.

        `capture_restrictions` is what `measure_complexity` counts per piece that names
        what it may take, and `spec_bytes` is the elegance metric itself.
        """
        before = measure_complexity(_wolves())
        after = measure_complexity(_hunted(_wolves()))
        assert before.capture_restrictions == 5
        assert after.capture_restrictions == 0
        assert after.spec_bytes < before.spec_bytes
        # And the rule itself is not costed on top of `npcs`: it parameterises a rule
        # already counted, the way `screen` and a turn `ramp` do.
        assert after.special_rules == before.special_rules

    def test_the_matrix_says_what_the_sentence_says(self) -> None:
        compiled = compile_spec(_hunted(_wolves()))
        wolf = compiled.piece_ids.index("W")
        for attacker, pid in enumerate(compiled.piece_ids):
            may = compiled.capture_allowed[attacker][wolf]
            assert may is (pid in ("N", "W")), pid
        assert compiled.has_capture_matrix

    def test_another_wild_piece_is_not_bound_by_it(self) -> None:
        """The exemption that makes the re-spelling exact, asserted as a rule.

        A wild piece eats whatever it steps onto; the immunity is a rule about the
        players. Without this the re-spelling would differ from `wolves` in one
        unreachable cell and the byte-identical check above would be checking less.
        """
        compiled = compile_spec(_hunted(_wolves()))
        wolf = compiled.piece_ids.index("W")
        assert compiled.capture_allowed[wolf][wolf]

    def test_an_older_stamp_is_refused_by_name(self) -> None:
        spec = _hunted(_wolves())
        spec["schema_version"] = 9
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert "rules.npcs[].immune" in errors[0]

    def test_a_list_that_protects_nothing_is_refused(self) -> None:
        """Q57's rule one level down: a declaration that does nothing is not free.

        `rule_mass` is the denominator of `rule_yield`, so an inert nested key deflates
        the ratio the index rewards rather than merely failing to help.
        """
        spec = _wolves()
        spec["schema_version"] = 10
        for npc in spec["rules"]["npcs"]:
            npc["immune"] = [piece["id"] for piece in spec["pieces"]]
        errors = validate_spec(spec)
        assert any("protects it from nothing" in error for error in errors)

    def test_an_undeclared_id_is_refused(self) -> None:
        spec = _hunted(_wolves())
        spec["rules"]["npcs"][0]["immune"] = ["Z"]
        assert any("'Z' is not a declared piece" in e for e in validate_spec(spec))

    def test_two_wild_pieces_of_one_type_may_not_disagree(self) -> None:
        """The matrix is over types, so an immunity is a fact about the type.

        Two entries with different lists would compile to the *intersection* -- only an
        attacker both allow could take either -- which is a game nobody wrote, and the
        permissive direction is the one that fails quietly. `wolves` reaches this and
        passes: two routes, one type, one list.
        """
        assert validate_spec(_hunted(_wolves())) == []
        spec = _hunted(_wolves())
        spec["rules"]["npcs"][1]["immune"] = ["B"]
        assert any("different 'immune' lists" in e for e in validate_spec(spec))
        # An entry with no list at all disagrees with one that has a list.
        partial = _hunted(_wolves())
        partial["rules"]["npcs"][1].pop("immune")
        assert any("different 'immune' lists" in e for e in validate_spec(partial))

    def test_it_is_refused_beside_drops(self) -> None:
        """Q84's caveat, decided rather than read owner-blind.

        `capture_allowed` is keyed by type and not by owner, so a wild type that can also
        reach a player's hand -- "tame the boar", which schema 9 deliberately allows --
        would carry its immunity out of the hand with it. The rule's own name says *wild
        piece*, and being wrong in the permissive direction is silent, so the combination
        is refused. Nothing becomes inexpressible: per-piece `captures` lists still say
        the owner-blind thing the long way, which is what `wolves` did.
        """
        for drops in (
            {"from_captures": True},
            # Schema 10's own second route into a hand, which Q51 opened in the same bump.
            {"from_captures": False, "stock": {"W": 2}},
        ):
            spec = _hunted(_wolves())
            spec["rules"]["drops"] = drops
            assert any(
                "can reach a player's hand" in error for error in validate_spec(spec)
            ), drops

        # A `drops` that puts some *other* type in a hand is not this refusal's business:
        # `region` says where a held piece may go, not how one is got.
        allowed = _hunted(_wolves())
        allowed["rules"]["drops"] = {"from_captures": False, "stock": {"P": 2}}
        assert not any(
            "can reach a player's hand" in error for error in validate_spec(allowed)
        )

    def test_it_narrows_a_legal_move_from_the_start_of_a_real_game(self) -> None:
        """The play-test, from `wolves`' own opening array and four ordinary plies.

        A wolf starts on a4 and steps a4, b5, c4, b3, a3 after every move, so after
        `1. Nf3 Nf6 2. Ng1 Ng8` -- four plies that touch neither the route nor the
        b-file -- one stands on a3, where the b2 pawn attacks it. Three spellings are
        played through the same four plies:

        * with no `captures` list anywhere, the pawn may take it;
        * with `immune: ["N"]`, it may not;
        * and **the committed `wolves` agrees with the second**, which is the claim the
          whole feature rests on: the same game, one key instead of five lists.
        """
        loose = _wolves()
        for piece in loose["pieces"]:
            piece.pop("captures", None)  # anything may take a wolf
        loose["schema_version"] = 10

        line = [("g1", "f3"), ("g8", "f6"), ("f3", "g1"), ("f6", "g8")]

        def pawn_targets(spec: dict[str, Any]) -> set[int]:
            engine = Engine(spec)
            state = engine.initial_state()
            for origin, target in line:
                state = engine.apply(state, _move(engine, state, origin, target))
            wild = engine.spec.npcs
            assert wild is not None and state.board[_square("a3")] != 0
            return {
                move_to(move)
                for move in engine.legal_moves(state)
                if move_from(move) == _square("b2")
            }

        assert _square("a3") in pawn_targets(loose)
        assert _square("a3") not in pawn_targets(_hunted(_wolves()))
        assert pawn_targets(_hunted(_wolves())) == pawn_targets(_wolves())

    def test_the_description_says_who_may_take_it(self) -> None:
        assert "Only N may take it." in describe_spec(_hunted(_wolves()))
        assert "may take it." not in describe_spec(_wolves())

    def test_it_has_a_distance_feature_of_its_own(self) -> None:
        assert "npcs_are_hard_to_kill" in _SCALES
        assert features(_wolves())["npcs_are_hard_to_kill"] == 0.0
        assert features(_hunted(_wolves()))["npcs_are_hard_to_kill"] == 1.0
        # Same board, same army, same route, one key apart: not distance zero.
        loose = _wolves()
        for piece in loose["pieces"]:
            piece.pop("captures", None)
        assert distance(loose, _hunted(_wolves())) > 0.0

    def test_fairy_still_refuses_the_whole_rule(self) -> None:
        """Q94's other warning: this one *is* covered, and here is the check.

        `npcs` is in `_RULES_REFUSED`, so a nested key inside it needs no listing of its
        own -- the rule never crosses. The refusal is asserted rather than assumed because
        "covered by the rule above" is exactly the reasoning the catch-all exists to stop
        anyone relying on without looking.
        """
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(_hunted(_wolves())))
        assert "rules.npcs" in str(refusal.value)

    def test_the_json_schema_offers_it_to_a_proposer(self) -> None:
        npcs = rule_spec_schema()["properties"]["rules"]["properties"]["npcs"]
        assert "immune" in npcs["items"]["properties"]

    def test_an_operator_can_reach_it(self, chess: dict[str, Any]) -> None:
        """Surface 8. An axis no operator can cross is an axis only the model can reach."""
        patrol = next(op for op in OPERATORS if op.name == "set-a-patrol")
        for seed in range(120):
            child = copy.deepcopy(chess)
            if patrol.apply(child, random.Random(seed)) is None:
                continue
            npcs = child["rules"].get("npcs") or []
            if npcs and "immune" in npcs[0]:
                assert validate_spec(child) == []
                assert child["schema_version"] == 10
                assert features(child)["npcs_are_hard_to_kill"] == 1.0
                return
        pytest.fail("no seed in range gave a wild piece an immunity")


# --------------------------------------------- Q18: which captures explode, and of what


def _selective(**narrowing: list[str]) -> dict[str, Any]:
    """`atomic` with its explosion narrowed, stamped for schema 10."""
    spec = json.loads((ROOT / "rulesets" / "atomic.json").read_text(encoding="utf-8"))
    spec["schema_version"] = 10
    spec["rules"]["explode_on_capture"].update(narrowing)
    return spec


class TestOnlySomeCapturesExplode:
    def test_atomic_itself_does_not_move(self) -> None:
        """The acceptance criterion names this ruleset, so it gets its own assertion.

        `by` and `of` absent compile to `None`, the boundary sends `[]` for both, and `[]`
        is their inert default -- so `canonical_tables` drops them and atomic's
        fingerprint is the one it had. A full-length vector of ones would have moved every
        ruleset with an explosion rule and bought a corpus-wide replay for a feature none
        of them uses.
        """
        atomic = json.loads((ROOT / "rulesets" / "atomic.json").read_text(encoding="utf-8"))
        compiled = compile_spec(atomic)
        assert compiled.explosion is not None
        assert compiled.explosion.fires_on_every_capture
        assert build_tables(compiled)["explosion_by"] == []
        assert build_tables(compiled)["explosion_of"] == []
        assert "explosion_by" not in canonical_tables(compiled)
        assert "explosion_of" not in canonical_tables(compiled)
        assert rust_supports(compiled) is None

    def test_the_validator_accepts_the_pair_under_ten(self) -> None:
        assert validate_spec(_selective(by=["Q"])) == []
        assert validate_spec(_selective(of=["P"])) == []
        assert validate_spec(_selective(by=["Q"], of=["P"])) == []

    def test_an_older_stamp_is_refused_by_name(self) -> None:
        for key in ("by", "of"):
            spec = _selective(**{key: ["Q"]})
            spec["schema_version"] = 9
            errors = validate_spec(spec)
            assert len(errors) == 1, errors
            assert f"rules.explode_on_capture.{key}" in errors[0]

    def test_an_undeclared_id_is_refused(self) -> None:
        for key in ("by", "of"):
            spec = _selective(**{key: ["Z"]})
            assert any(
                f"explode_on_capture.{key} entry 'Z'" in error
                for error in validate_spec(spec)
            ), key

    def test_it_compiles_to_the_types_it_names(self) -> None:
        compiled = compile_spec(_selective(by=["Q"], of=["P"]))
        assert compiled.explosion is not None
        queen = compiled.piece_ids.index("Q")
        pawn = compiled.piece_ids.index("P")
        assert compiled.explosion.by_types == frozenset({queen})
        assert compiled.explosion.of_types == frozenset({pawn})
        assert not compiled.explosion.fires_on_every_capture
        assert build_tables(compiled)["explosion_by"][queen] == 1
        assert sum(build_tables(compiled)["explosion_by"]) == 1

    def test_a_list_naming_every_type_is_atomic_written_the_long_way(self) -> None:
        """An inert *spelling* stays legal and compiles to the game already there.

        `_check_inert_rules` draws that line deliberately -- `turn: {"moves": 1}` and
        `{"effect": "capture"}` are the precedents -- and it is also what keeps the
        boundary field inert: a full list normalises to `None`, so it sends `[]`.
        """
        every = [piece["id"] for piece in _selective()["pieces"]]
        compiled = compile_spec(_selective(by=every, of=every))
        assert compiled.explosion is not None
        assert compiled.explosion.fires_on_every_capture
        assert "explosion_by" not in canonical_tables(compiled)
        assert "explosion_of" not in canonical_tables(compiled)
        # And the rest of the compiled game is atomic's. Compared key by key rather than
        # whole, because the stamp is part of the mechanics hash and the hash seeds the
        # Zobrist draw -- so a spec saying 10 lays its transposition table out
        # differently however identically it plays.
        plain = canonical_tables(
            compile_spec(json.loads((ROOT / "rulesets" / "atomic.json").read_text(encoding="utf-8")))
        )
        theirs = canonical_tables(compiled)
        assert {
            key: value for key, value in theirs.items() if not key.startswith("zobrist")
        } == {
            key: value for key, value in plain.items() if not key.startswith("zobrist")
        }

    def test_an_empty_list_is_no_explosion_at_all(self) -> None:
        """`by: []` is a rule no capture can fire, so the compiler drops the whole rule.

        The `_compile_strike` shape for a chain that leaves nothing alive: the boundary
        carries the answer, not the question, and the core keeps one test rather than two.
        """
        for key in ("by", "of"):
            compiled = compile_spec(_selective(**{key: []}))
            assert compiled.explosion is None, key
            assert rust_supports(compiled) is None, key

    def test_an_empty_list_is_charged_nothing(self) -> None:
        """Q56's advance ruling (`docs/reference/bugs-investigations.md`), which this change owed in the same commit.

        `_RULE_COSTS` charges `explode_on_capture` 1.5 on truthiness alone, so without an
        `_RULE_ACTIVE` clause `{"by": []}` would be 1.5 for a no-op -- and `rule_mass` is
        the *denominator* of `rule_yield`, so a missing charge does not merely fail to
        help, it inflates the ratio the index rewards. A missing charge also fails
        silently, which is why the ruling said this is the half that must not be forgotten.
        """
        plain = json.loads((ROOT / "rulesets" / "atomic.json").read_text(encoding="utf-8"))
        without = copy.deepcopy(plain)
        without["rules"].pop("explode_on_capture")
        assert measure_complexity(_selective(by=[])).special_rules == (
            measure_complexity(without).special_rules
        )
        assert measure_complexity(plain).special_rules > (
            measure_complexity(without).special_rules
        )
        # And a list naming everything is a real rule, charged in full.
        every = [piece["id"] for piece in plain["pieces"]]
        assert measure_complexity(_selective(by=every)).special_rules == (
            measure_complexity(plain).special_rules
        )
        # `{}` and `false` were free before this clause and stay free.
        for value in ({}, False):
            spec = copy.deepcopy(plain)
            spec["rules"]["explode_on_capture"] = value
            assert measure_complexity(spec).special_rules == (
                measure_complexity(without).special_rules
            ), value

    def test_the_hold_came_off_when_the_core_could_answer(self) -> None:
        """The eleventh hold, retired the way three of the ten before it were.

        `_CORE_KNOWS_SELECTIVE_EXPLOSION` was False for one commit, while the compiler
        could say "only a knight's captures explode" and `game.rs` still exploded on every
        one -- a *plausible wrong game*, which is the failure `rust_supports` exists to
        make loud. `ExplosionRule::fires` closed it, so the refusal is gone and a narrowed
        explosion is playable.
        """
        assert rust_supports(compile_spec(_selective(by=["N"]))) is None
        Engine(_selective(by=["N"]))

    def test_the_core_plays_the_narrowing(self) -> None:
        """The play-test, and the one that would have caught a wrong core.

        `1. e4 d5 2. exd5` from atomic's own start. Under atomic every capture explodes,
        so d5 is empty afterwards -- capturer and victim both go, and the pawns beside
        them survive only because atomic spares pawns. Under `by: ["N"]` the same four
        plies are an ordinary capture and a white pawn stands on d5.

        Three spellings, one line, so the difference cannot be anything but the rule.
        """
        line = [("e2", "e4"), ("d7", "d5"), ("e4", "d5")]

        def after(spec: dict[str, Any]) -> int:
            engine = Engine(spec)
            state = engine.initial_state()
            for origin, target in line:
                state = engine.apply(state, _move(engine, state, origin, target))
            return state.board[_square("d5")]

        plain = json.loads((ROOT / "rulesets" / "atomic.json").read_text(encoding="utf-8"))
        assert after(plain) == 0, "atomic: the capturer goes up with its victim"
        assert after(_selective(by=["N"])) != 0, "only a knight is a bomb here"
        assert after(_selective(of=["N"])) != 0, "only taking a knight sets one off"
        # And a narrowing that *does* admit this capture explodes exactly as atomic does.
        assert after(_selective(by=["P"], of=["P"])) == 0

    def test_the_legality_filter_reads_the_same_narrowing(self) -> None:
        """`make` is not the only site: a move is legal only if it leaves no royal attacked.

        The filter replays the blast on a scratch board, so a narrowing it did not read
        would make a king's capture suicide in one place and not the other -- the two
        sites disagreeing is a position where the engine offers a move it then refuses to
        have played. Asserted as an invariant over a whole short game rather than at one
        position: every move the generator offers is a move `make` can play.
        """
        for spec in (_selective(by=["N"]), _selective(of=["P"])):
            engine = Engine(spec)
            state = engine.initial_state()
            for _ in range(12):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                for move in moves:
                    engine.apply(state, move)  # no panic, no illegal write
                state = engine.apply(state, moves[len(moves) // 2])

    def test_the_description_says_which_capture(self) -> None:
        """Asserted against `_describe_explosion` itself.

        `atomic.json`'s own `description` field says the same sentence in prose, so a
        substring search over the whole rendering would pass without the generated line
        existing at all.
        """
        from chess_variant_sandbox.spec.describe import _describe_explosion

        def sentence(**narrowing: list[str]) -> list[str]:
            return _describe_explosion(
                _selective(**narrowing)["rules"]["explode_on_capture"]
            )

        assert sentence(by=["Q"]) == [
            "A capture by Q destroys the capturing piece and everything beside it, "
            "except P."
        ]
        assert sentence(of=["P"])[0].startswith("A capture of P destroys")
        assert sentence(by=["Q"], of=["P"])[0].startswith("A capture by Q of P destroys")
        assert sentence()[0].startswith("A capture destroys")
        # A rule no capture can fire is described as no rule, matching the compiler.
        assert sentence(by=[]) == []
        assert sentence(of=[]) == []

    def test_it_has_a_distance_feature_of_its_own(self) -> None:
        plain = json.loads((ROOT / "rulesets" / "atomic.json").read_text(encoding="utf-8"))
        assert "explodes_selectively" in _SCALES
        assert features(plain)["explodes_selectively"] == 0.0
        assert features(_selective(by=["Q"]))["explodes_selectively"] == 1.0
        assert distance(plain, _selective(by=["Q"])) > 0.0
        # An empty list reads as no explosion, the same answer the compiler gives.
        assert features(_selective(by=[]))["explodes"] == 0.0

    def test_fairy_refuses_it_by_name(self) -> None:
        """Q94's first warning: the key guard catches this one, and the branch decides it.

        `blastOnCapture` is a switch over there -- every capture explodes -- and
        `blastImmuneTypes` narrows only what the blast *reaches*. Listing the keys in
        `_RULE_OBJECT_KEYS` stops the silence; this is the branch that says what they mean.
        """
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(_selective(by=["Q"])))
        assert any(
            "explode_on_capture.by / .of" in reason for reason in refusal.value.reasons
        )
        # Read off the compiled spec, so neither degenerate spelling refuses anything.
        every = [piece["id"] for piece in _selective()["pieces"]]
        translate(compile_spec(_selective(by=every)))
        translate(compile_spec(_selective(by=[])))

    def test_the_page_is_told_which_captures_explode(self) -> None:
        """Surface 11, and Q84 named this one specifically.

        The page cannot work out from two positions which board change was which, so a
        rule narrowing *when* the blast fires has to reach it or a capture by a piece that
        is not a bomb animates an explosion that never happened.
        """
        from chess_variant_sandbox.web.play import Session

        info = Session._explosion_info(_Spec(compile_spec(_selective(by=["Q"]))))
        assert info is not None
        queen = compile_spec(_selective(by=["Q"])).piece_ids.index("Q")
        assert info["by"] is not None and info["by"][queen] is True
        assert sum(info["by"]) == 1
        assert info["of"] is None

        plain = compile_spec(json.loads((ROOT / "rulesets" / "atomic.json").read_text(encoding="utf-8")))
        atomic_info = Session._explosion_info(_Spec(plain))
        assert atomic_info is not None
        assert atomic_info["by"] is None and atomic_info["of"] is None

    def test_the_json_schema_offers_the_pair_to_a_proposer(self) -> None:
        explosion = rule_spec_schema()["properties"]["rules"]["properties"][
            "explode_on_capture"
        ]
        assert {"by", "of"} <= set(explosion["properties"])


    def test_a_short_game_keeps_its_incremental_hash(self) -> None:
        """The walk `new-move-primitive` asks for, on an inline spec.

        `test_schema_two.py`'s version covers every committed and archived ruleset and no
        committed ruleset should carry this rule yet, so the walk lives here. It is the
        check a perft cannot make: `make` writes the blast into the Zobrist key square by
        square, `unmake` restores it from the undo log, and a rule that narrows *when* the
        blast fires is a chance for the two to disagree -- which shows up as a drifted key
        long before it shows up as a wrong move count.
        """
        for spec in (
            _selective(by=["N"]),
            _selective(of=["P"]),
            _selective(by=["Q", "R"], of=["Q", "R", "N"]),
        ):
            engine = Engine(spec)
            rng = random.Random(7)
            state = engine.initial_state()
            for _ in range(20):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                state = engine.apply(state, rng.choice(moves))
                assert state.key == engine._compute_hash(state), "hash drifted"



# ------------------------------------------------- Q51: a hand that starts with something


def _stone(held: Any = "unlimited", moves: list[Any] | None = None) -> dict[str, Any]:
    """chess plus a stone: a piece with no moves, played from a stocked hand.

    The demonstrating shape rather than a contrivance -- placement is the whole content
    of the piece, which is what the feature exists to make expressible.
    """
    spec = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
    spec["schema_version"] = 10
    spec["pieces"].append(
        {"id": "S", "name": "Stone", "value": 1, "moves": moves if moves else []}
    )
    spec["rules"]["drops"] = {"from_captures": False, "stock": {"S": held}}
    return spec


class TestAHandMayStartWithSomethingInIt:
    def test_the_validator_accepts_a_stone(self) -> None:
        assert validate_spec(_stone()) == []
        assert validate_spec(_stone(3)) == []

    def test_an_older_stamp_is_refused_by_name(self) -> None:
        spec = _stone()
        spec["schema_version"] = 9
        errors = [error for error in validate_spec(spec) if "schema 10" in error]
        assert len(errors) == 1
        assert "rules.drops.stock" in errors[0]
        assert "a piece with no moves of its own" in errors[0]

    def test_the_bound_and_the_word_are_the_only_two_spellings(self) -> None:
        for held in ("Unlimited", "endless", 0, -1, MAX_STOCK + 1, True, 1.5):
            assert validate_spec(_stone(held)), held
        for held in (1, MAX_STOCK, STOCK_UNLIMITED):
            assert validate_spec(_stone(held)) == [], held

    def test_an_undeclared_id_is_refused(self) -> None:
        spec = _stone()
        spec["rules"]["drops"]["stock"] = {"Z": 1}
        assert any("stock key 'Z'" in error for error in validate_spec(spec))

    def test_the_relaxation_is_gated_on_the_stock_and_not_removed(self) -> None:
        """Q29's shape, and the reason it is a predicate rather than a deletion.

        Removing `pieces[].moves must be a non-empty array` outright would let *every*
        spec ever written declare a piece that can never move -- schema 1 included, since
        a version gate only stops an old stamp reaching a new key. So it is conditioned
        on `piece_has_stock`, exactly as schema 2's multiple royals are conditioned on
        `royalty_can_multiply`.
        """
        assert piece_has_stock(_stone()["rules"], "S")
        assert not piece_has_stock(_stone()["rules"], "P")

        orphan = _stone()
        orphan["rules"].pop("drops")
        errors = validate_spec(orphan)
        assert any("moves must be a non-empty array" in error for error in errors)

        # A stock for a *different* piece does not license this one.
        elsewhere = _stone()
        elsewhere["rules"]["drops"]["stock"] = {"P": 2}
        assert any(
            "moves must be a non-empty array" in error
            for error in validate_spec(elsewhere)
        )

    def test_a_stocked_piece_can_stand_on_the_board(self) -> None:
        """The sixth route into `_pieces_that_can_stand`, and the first that needs no other.

        Without it a promotion into the stone -- or any other rule about it -- would be
        refused as being about a piece that can never appear, because the five routes
        that existed all needed the piece to have been somewhere else first.
        """
        spec = _stone()
        # Rank 7, the last of chess's eight. This said 8 until Q434 bounded a lone rank by
        # the board: past the board, the zone was empty and the rule never fired.
        spec["rules"]["promotion"] = {"piece": "P", "rank": 7, "to": ["S"]}
        assert validate_spec(spec) == []

    def test_the_hand_starts_full_and_never_empties(self) -> None:
        engine = Engine(_stone())
        stone = engine.spec.piece_ids.index("S")
        state = engine.initial_state()
        assert state.hands[0][stone] == 1 and state.hands[1][stone] == 1

        for _ in range(6):
            drop = next(
                move for move in engine.legal_moves(state) if move_type(move) == MOVE_DROP
            )
            state = engine.apply(state, drop)
            assert state.hands[state.side_to_move][stone] == 1, "a supply is not spent"
            assert state.key == engine._compute_hash(state), "hash drifted"

    def test_a_finite_stock_is_spent(self) -> None:
        """The other half, and it needs no core change at all: the hand already counts down."""
        engine = Engine(_stone(2))
        stone = engine.spec.piece_ids.index("S")
        state = engine.initial_state()
        assert state.hands[0][stone] == 2 and state.hands[1][stone] == 2

        # Four plies: each side plays two stones, and the side that just moved is the one
        # whose hand went down.
        for expected in (1, 1, 0, 0):
            mover = state.side_to_move
            drop = next(
                move for move in engine.legal_moves(state) if move_type(move) == MOVE_DROP
            )
            state = engine.apply(state, drop)
            assert state.hands[mover][stone] == expected
        assert not [
            move for move in engine.legal_moves(state) if move_type(move) == MOVE_DROP
        ], "an empty hand offers nothing"

    def test_the_stone_is_placed_blocks_and_can_be_taken(self) -> None:
        """A piece with no moves is still a piece, which is the whole design.

        The three things it does are the three the relaxation had to keep true: it
        arrives, it occupies a square nothing may pass through, and it can be captured.
        """
        engine = Engine(_stone())
        stone = engine.spec.piece_ids.index("S")
        state = engine.initial_state()
        # Dropped in front of black's e-pawn, which attacks the square diagonally.
        drop = next(
            move
            for move in engine.legal_moves(state)
            if move_type(move) == MOVE_DROP and move_to(move) == _square("d6")
        )
        state = engine.apply(state, drop)
        assert engine.spec.piece_at(state.board, _square("d6")) == (0, stone)

        # It never generates a move of its own, on either side of it.
        assert not [
            move
            for move in engine.legal_moves(state)
            if move_from(move) == _square("d6")
        ]
        # And it is a real occupant: black is to move and may take it, so a piece with
        # no moves is still something that has to be dealt with.
        assert _square("d6") in {
            move_to(move)
            for move in engine.legal_moves(state)
            if move_from(move) == _square("e7")
        }
        # It also blocks: black's own d-pawn cannot step through it.
        assert _square("d6") not in {
            move_to(move)
            for move in engine.legal_moves(state)
            if move_from(move) == _square("d7")
        }

    def test_unmake_needs_nothing_for_a_supply(self) -> None:
        """Q84's finding, held against a second reading rather than claimed.

        `unmake` restores a hand from a signed delta in the undo log, so an unlimited slot
        records no delta -- the same shape schema 6 found for a strike. `perft` walks
        make/unmake; `apply` builds a fresh state every ply and never unmakes anything. A
        hand the undo log got wrong makes the two disagree, and nothing else here would.
        """
        def by_apply(engine: Engine, position: Any, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                by_apply(engine, engine.apply(position, move), depth - 1)
                for move in engine.legal_moves(position)
            )

        # Both spellings: the unlimited slot, which records no delta, and the finite one,
        # where the delta is recorded and has to come back.
        for spec in (_stone(), _stone(2)):
            engine = Engine(spec)
            state = engine.initial_state()
            walked = by_apply(engine, state, 2)
            assert walked > 0
            assert engine.perft(state, 2) == walked

    def test_the_stock_is_the_same_for_both_players(self) -> None:
        """`setup.mirror`'s argument: a supply one side has is a first-seat effect."""
        engine = Engine(_stone(3))
        state = engine.initial_state()
        assert state.hands[0] == state.hands[1]

    def test_the_description_says_both_halves(self) -> None:
        prose = describe_spec(_stone())
        assert "endless supply of S" in prose
        assert "never moves" in prose
        assert "Each player starts holding 3 S" in describe_spec(_stone(3))

    def test_it_has_distance_features_of_its_own(self) -> None:
        chess = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        for name in ("has_stock", "stock_is_endless", "has_a_moveless_piece"):
            assert name in _SCALES
            assert features(chess)[name] == 0.0
            assert features(_stone())[name] == 1.0
        assert features(_stone(3))["stock_is_endless"] == 0.0
        assert distance(_stone(3), _stone()) > 0.0

    def test_the_inert_rule_refusals_moved_with_it(self) -> None:
        """Q57 said this line would move, and it moved.

        `drops` with `from_captures: false` used to be a hand nothing could ever fill, and
        was refused. With a stock it is the ordinary spelling of a placement game, and the
        refusal has to stop applying -- while still applying to a `drops` with neither.
        """
        assert validate_spec(_stone()) == []
        empty = _stone()
        empty["rules"]["drops"] = {"from_captures": False}
        empty["pieces"] = [p for p in empty["pieces"] if p["id"] != "S"]
        assert any(
            "there is never anything to drop" in error for error in validate_spec(empty)
        )

    def test_fairy_refuses_only_the_unlimited_half(self) -> None:
        """The two halves did separate. A finite stock is the startFen's pocket (Q784),
        and a piece with no moves is an empty notation (Q703). An unlimited slot is
        still refused: a hand there counts down."""
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(_stone()))
        assert any("rules.drops.stock" in r for r in refusal.value.reasons)
        assert not any("no moves" in r for r in refusal.value.reasons)
        written = translate(compile_spec(_stone(3)))
        assert f"= {written.chars[-1].lower()}:\n" in written.config

    def test_a_ruleset_without_a_stock_still_crosses(self) -> None:
        """Read off the compiled rule, so shogi and crazyhouse are untouched."""
        for name in ("crazyhouse", "shogi"):
            spec = json.loads((ROOT / "rulesets" / f"{name}.json").read_text(encoding="utf-8"))
            compiled = compile_spec(spec)
            assert compiled.drops is not None and not compiled.drops.stock
            assert not compiled.moveless_types

    def test_the_json_schema_offers_it_to_a_proposer(self) -> None:
        drops = rule_spec_schema()["properties"]["rules"]["properties"]["drops"]
        assert "stock" in drops["properties"]

    def test_no_committed_ruleset_with_a_hand_moved(self) -> None:
        """The boundary fields are empty when nothing is stocked, which is all of them."""
        for name in ("crazyhouse", "shogi", "dobutsu", "crazyhouse-torus"):
            compiled = compile_spec(
                json.loads((ROOT / "rulesets" / f"{name}.json").read_text(encoding="utf-8"))
            )
            tables = build_tables(compiled)
            assert tables["drops_stock"] == [] and tables["drops_unlimited"] == []
            assert "drops_stock" not in canonical_tables(compiled)
            assert "drops_unlimited" not in canonical_tables(compiled)


class TestASupplyThatNeverRunsOutIsNotMaterial:
    """Q119. `archive/stones` indexed 0.3144 and the price was why -- not the whole of why,
    Q131 later showed: an unlimited stock of a genuinely inert piece is a second,
    independent defect, and a third stood behind it in `_live_values`, now closed (Q137):
    the denominator is the prey pool, not the enemy army. This class holds only the price
    this module fixes.

    The material term read the board and not the hand, so a drop moved a piece from
    somewhere uncounted to somewhere counted and paid the mover its value for the act of
    placing it. A finite stock bounded that at `stock x value`, spent once --
    `archive/three-stones` measured 0.8058 with the distortion in it. An unlimited stock
    does not bound it at all: `archive/stones` put 0.5198 of its moves into drops, ran 204
    of 256 games into the ply limit, and indexed 0.3144 against chess's 0.7943.

    So an inexhaustible type is priced at nothing, which is what it is worth: being ahead
    on a piece your opponent may conjure whenever they like is not being ahead. Counting
    the hand (`v11` A6) closed the finite case and not this one, because an endless slot
    never empties: its drop adds a stone to the board and takes none from the hand. What
    none of this claims is anything about what a moveless piece with a stock that runs
    out is *worth*; see the finite-stock test here.
    """

    def test_an_unlimited_type_is_worth_nothing(self) -> None:
        spec = compile_spec(_stone())
        values = Evaluator.for_spec(spec).values
        stone = spec.piece_ids.index("S")
        assert inexhaustible_types(spec) == {stone}
        assert values[stone] == 0.0

    def test_it_is_out_of_the_scale_too_so_the_rest_is_the_host_game(self) -> None:
        """A piece that is not material cannot be the unit material is counted in.

        The stone's reach is 0, which drove `min(reach)` to the `1.0` fallback and left
        every *other* piece in the game on a different scale from the same piece in chess
        -- a pawn at 1.3984 rather than 1.0. Taking the stone out of the scale as well as
        out of the sum makes `stones` price its chess pieces exactly as chess does, which
        is the strongest available statement of what the game now is: chess plus a
        placement that costs a tempo and buys nothing.
        """
        chess_values = Evaluator.for_spec(
            compile_spec(json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8")))
        ).values
        stones = Evaluator.for_spec(compile_spec(_stone())).values
        assert len(stones) == len(chess_values) + 1
        assert stones[: len(chess_values)] == chess_values

    def test_placing_one_is_no_longer_free_material(self) -> None:
        """The defect itself, at the position rather than in the table."""
        engine = Engine(_stone())
        evaluator = Evaluator.for_spec(engine.spec)
        state = engine.initial_state()
        mover = state.side_to_move
        before = evaluator.material(state, mover)

        drop = next(
            move for move in engine.legal_moves(state) if move_type(move) == MOVE_DROP
        )
        after = evaluator.material(engine.apply(state, drop), mover)
        assert after == before, "a drop from an endless hand is material out of nothing"

    def test_a_finite_stock_is_priced_by_the_fallback_and_its_drop_is_free(self) -> None:
        """The price is still the fallback; paying for the drop closed with the hand (A6).

        `_measured_values` counts *capture* reach, so a moveless piece comes back 0 and
        the blend prices it at `DECLARED_WEIGHT` times its declared value alone -- 0.25
        here -- which also drags the rest of the table off the host game's scale. That
        half is deliberately **not** fixed: the one measurement there is says it plays
        fine (`archive/three-stones`, 0.8058, viable, 0.8398 of seats spending all
        three), so replacing it would mean inventing a constant. It is pinned here so it
        is a known edge and not a rumour.

        The other half closed on 2026-09-24 (`v11` A6). The material term used to read
        the board and not the hand, so a drop from a finite stock paid the mover the
        stone's value, `stock x value` spent once. The hand is material now, so a drop
        moves a stone from one counted place to another and pays nothing -- while the
        board alone, which is what the term used to read, still moves.
        """
        spec = compile_spec(_stone(3))
        values = Evaluator.for_spec(spec).values
        stone = spec.piece_ids.index("S")
        assert inexhaustible_types(spec) == frozenset()
        assert values[stone] == pytest.approx(DECLARED_WEIGHT)

        chess_values = Evaluator.for_spec(
            compile_spec(json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8")))
        ).values
        assert values[: len(chess_values)] != chess_values, "the scale moves too"

        engine = Engine(_stone(3))
        evaluator = Evaluator.for_spec(engine.spec)
        state = engine.initial_state()
        mover = state.side_to_move
        before = evaluator.material(state, mover)
        drop = next(
            move for move in engine.legal_moves(state) if move_type(move) == MOVE_DROP
        )
        dropped = engine.apply(state, drop)
        after = evaluator.material(dropped, mover)
        assert after == pytest.approx(before, abs=1e-12), "a drop from a hand is not material"
        assert evaluator.board_material(dropped.board, mover) > evaluator.board_material(
            state.board, mover
        ), "the board alone, which the term used to read, still pays for it"

    def test_the_walk_below_reads_the_whole_corpus(self) -> None:
        assert len(CORPUS_PATHS) > 50, "the corpus moved out from under this test"

    # One test per spec (Q749): the walk was one test of 3.3 s in the gate.
    @pytest.mark.parametrize("path", CORPUS_PATHS, ids=lambda path: path.stem)
    def test_the_price_moves_only_where_a_spec_says_unlimited(self, path: Path) -> None:
        """Inertness, over every committed and archived spec rather than by argument.

        A non-royal piece worth exactly nothing is the new branch and nothing else, so
        this fires if the branch ever reaches a game that did not ask for it -- and it
        fires the other way too, if some future derivation prices an ordinary piece at
        zero, which is the bug this fix would otherwise have introduced everywhere.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        spec = compile_spec(raw)
        values = Evaluator.for_spec(spec).values
        free = {i for i, v in enumerate(values) if v == 0.0}
        free -= set(spec.royal_types)
        # A portal (schema 19) is nobody's material by rule, not by supply: its declared
        # type and its ends are left out, as the royals are (Q772a's showcases).
        rules = raw.get("rules", {})
        portals = portal_piece_ids(rules) | set(portal_end_ids(portal_groups(rules)))
        free -= {i for i, pid in enumerate(spec.piece_ids) if pid in portals}
        assert free == inexhaustible_types(spec), path.name


class TestTheGoalWeightIsStatedInPiecesThatAreMaterial:
    """Q124(b): the Go-shaped hole in `_mean_non_royal`, closed.

    `GOAL_PIECES` is stated in mean non-royal pieces, and the mean averaged
    `INEXHAUSTIBLE_VALUE`'s zero in with every real piece -- so a game with **both** an
    endless supply and a rule to win by got its goal term understated by construction,
    by exactly the share of its army that is inexhaustible. Go is a game with both.

    The fix is the sentence `Evaluator.for_spec` already applies to the two scales, one
    line further on: a piece that is not material cannot be the unit material is counted
    in. It invents nothing, and it is inert on the whole corpus -- `archive/stones` is
    the only spec with an endless supply and it is won by checkmate, so the weight it
    derives is never read. `tests/test_backends.py` holds the core to the same mean.
    """

    def test_a_stocked_game_states_its_goal_in_the_same_unit_as_the_host(self) -> None:
        """`stones` plus a goal rule weighs a win exactly as chess plus that rule does.

        The strongest available statement of what the fix means: `stones` already prices
        its chess pieces exactly as chess does (the test above), so once the stone is out
        of the mean as well as out of the sum, the two games agree on what a win is worth
        to the digit. Before, the stone's zero dragged the unit to five sixths of it.
        """
        stoned = _stone()
        stoned["rules"]["check_limit"] = 2
        plain = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        plain["schema_version"] = 2
        plain["rules"]["check_limit"] = 2

        stoned_eval = Evaluator.for_spec(compile_spec(stoned))
        plain_eval = Evaluator.for_spec(compile_spec(plain))
        assert stoned_eval.goal_weight == plain_eval.goal_weight

        # And the size of what was fixed, stated rather than implied: six non-royal
        # types where five are material, so the unit was 5/6 of what it should be.
        diluted = sum(stoned_eval.values) / (
            len(stoned_eval.values) - len(stoned_eval.spec.royal_types)
        )
        assert diluted * GOAL_PIECES == pytest.approx(plain_eval.goal_weight * 5 / 6)

    def test_the_corpus_is_the_one_the_count_below_reads(self) -> None:
        assert len(CORPUS_PATHS) == 98, "the corpus moved out from under this test"  # 98 since Q936s

    # One test per spec (Q749): the walk was one test of 2.9 s in the gate. `stones` is
    # the only spec with an inexhaustible type, asked of each spec by name.
    @pytest.mark.parametrize("path", CORPUS_PATHS, ids=lambda path: path.stem)
    def test_the_unit_is_untouched_wherever_nothing_is_endless(self, path: Path) -> None:
        """Inert on 69 of 70, and on the stone game for a reason rather than by luck.

        The mean can only move on a spec with an inexhaustible type, and `stones` is the
        only one. It is won by checkmate, so `has_goal_terms` is false and nothing reads
        the weight -- which is why this could be closed before the game that needs it
        exists, rather than after. The 61st spec, `weathervane-chaos-torus` (2026-09-11),
        and the 62nd, `weathervane-half-turn` (2026-09-13), are also won by checkmate and
        have no inexhaustible type, so they join the plain majority and move neither count.
        The 63rd, `wolves-bounty` (2026-09-13), has goal terms through its scoring rule but
        no inexhaustible type, so it joins the plain majority too. So does the 64th,
        `wolves-rotated` (2026-09-14), which is won by checkmate, and the 65th to 67th,
        its three sibling rotations (2026-09-14). The 68th, `merger` (schema 11), the
        69th, `hierarchy-rise-throw` (schema 12), the 70th, `chess-sweeping-bishops`
        (schema 13), and the 71st, `hex-glinski-promoting` (schema 14), are won by
        checkmate and declare no `drops`. Q280 retired `hex-glinski` on 2026-09-17, which
        was also won by checkmate, so the count fell by one and no verdict moved, and Q434
        retired `tunnel-horizontal` on 2026-09-25 the same way. Q281a4's three hexagon
        showcases (schema 16, 2026-09-28) are won by checkmate and declare no `drops`, and
        so are Q540's four schema-17 showcases (2026-10-09), and Q781's two schema-18
        showcases, `einstein` and `omega`. Q786's `jeson-mor` has goal terms (a priced
        race) but no inexhaustible type, so it joins the plain majority too, and Q782's two
        hex-symmetry showcases are won by checkmate and declare no `drops`.
        """
        spec = compile_spec(json.loads(path.read_text(encoding="utf-8")))
        values = Evaluator.for_spec(spec).values
        undiluted = _mean_non_royal(values, spec.royal_types, inexhaustible_types(spec))
        diluted = _mean_non_royal(values, spec.royal_types)
        assert bool(inexhaustible_types(spec)) == (path.stem == "stones"), path.stem
        if inexhaustible_types(spec):
            assert not has_goal_terms(spec), path.stem
        else:
            assert undiluted == diluted, path.stem


def _capture_then_redrop(
    engine: Engine,
    evaluator: Evaluator,
    seed: int,
    limit: int = 200,
    board_only: bool = False,
) -> tuple[int, float, float]:
    """Walk until one piece is taken and the taker plays it back. Both deltas, and its type.

    ``board_only`` reads ``Evaluator.board_material``, the board without the hand, which
    is what the material term was before `v11` A6 and is still the vocabulary's
    ``material`` column. The walk is the same either way, because the capture it stops at
    is recognised the same way in both readings.

    Random play at a fixed seed, which is deterministic in the sense this repo means:
    the engine and the seed decide the walk, so the position it finds is the same
    position every time. Each delta is measured across its own single ply -- material
    immediately before against material immediately after, always from the *taker's*
    point of view -- so the opponent's replies in between cannot leak into either one.

    A capture that **reverts** is walked past rather than reported. Shogi's
    ``capture_reverts`` puts a promoted rook into the taker's hand as a pawn, so the
    piece that left the board and the piece that comes back are two different pieces
    and "the same piece counted at both ends" is not the statement being made. The
    accounting there is not in dispute and needs no test of its own. It is recognised
    by its delta: a capture that does not revert pays the taker the victim twice, once
    off the opponent's board and once into the taker's hand.
    """
    def reading(position: GameState, side: int) -> float:
        if board_only:
            return evaluator.board_material(position.board, side)
        return evaluator.material(position, side)

    # What a capture that does not revert pays the taker: the victim off their board,
    # and, unless only the board is read, the same piece into the taker's hand.
    ends = 1.0 if board_only else 2.0

    rng = random.Random(seed)
    state = engine.initial_state()
    for _ in range(limit):
        moves = engine.legal_moves(state)
        if not moves:
            break
        taker = state.side_to_move
        before = reading(state, taker)
        move = rng.choice(moves)
        after_state = engine.apply(state, move)
        grew = [
            piece_type
            for piece_type, count in enumerate(after_state.hands[taker])
            if count > state.hands[taker][piece_type]
        ]
        if not grew:
            state = after_state
            continue
        held = grew[0]
        capture_delta = reading(after_state, taker) - before
        if capture_delta != pytest.approx(ends * evaluator.values[held]):
            state = after_state
            continue

        # Now the same side plays it back, at its first opportunity to do so.
        state = after_state
        for _ in range(limit):
            legal = engine.legal_moves(state)
            if not legal:
                break
            if state.side_to_move == taker:
                # A drop names no origin: the piece type travels in the promotion field.
                drops = [
                    m
                    for m in legal
                    if move_type(m) == MOVE_DROP and move_promotion(m) == held
                ]
                if drops:
                    was = reading(state, taker)
                    played = engine.apply(state, drops[0])
                    now = reading(played, taker)
                    return held, capture_delta, now - was
            state = engine.apply(state, rng.choice(legal))
        break
    raise AssertionError("no capture and re-drop inside the walk")


class TestACaptureThatChangesSidesIsCountedAtBothEnds:
    """Q124, settled by Q228 and built by `v11` A6: the hand is counted at `h = 1`.

    `docs/bugs.md` once recorded this as a **double count** -- one crazyhouse knight worth
    6.6429 across two plies -- and the arithmetic reproduced while the diagnosis did not.
    A capture under `from_captures` does not destroy a piece, it changes its owner, so
    after the capture and the re-drop the board really does differ by two knights and the
    score of both *endpoints* is correct. What was wrong was the position in between:
    `Evaluator.material` took a board and nothing else, so a piece in a hand scored
    nothing and the whole of the second 3.3214 was paid out as a bonus for the *act of
    dropping* a piece its owner already had.

    The fix prices the hand at some *h* of the board value -- capture (1+h)v, drop
    (1-h)v, total 2v for every *h* -- and until 2026-09-24 these tests pinned `h = 0`,
    because *h* was unmeasured. Q228's paired duel measured it: the objective with the
    hand counted at `h = 1` beat `h = 0` by 0.93 to 0.99 at 6,400 nodes on all four
    `from_captures` games and lost nowhere (`docs/reference/hand-value-duel.md`). `h = 1`
    is conservation: the capture pays 2v and the drop pays nothing. These tests pin that
    split, and the total that every *h* shares.
    """

    def test_drops_change_no_pieces_price(self) -> None:
        """Where 6.6429 comes from: it is twice chess's knight, not a crazyhouse knight.

        `_measured_values` counts capture reach and a `drops` rule changes no piece's
        reach, so crazyhouse derives the identical table to chess -- which is why the
        entry's number is exactly `2 * 3.3214` and why no rescaling of the table could
        express the error.
        """
        chess_values = Evaluator.for_spec(
            compile_spec(json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8")))
        ).values
        crazy = compile_spec(
            json.loads((ROOT / "rulesets" / "crazyhouse.json").read_text(encoding="utf-8"))
        )
        assert Evaluator.for_spec(crazy).values == chess_values
        assert crazy.drops is not None and crazy.drops.from_captures
        knight = crazy.piece_ids.index("N")
        assert chess_values[knight] == pytest.approx(3.3214, abs=5e-5)

    def test_the_capture_pays_the_whole_piece_twice_and_the_drop_nothing(self) -> None:
        """`h = 1` at the position: 2v at the capture, nothing at the re-drop."""
        engine = Engine(json.loads((ROOT / "rulesets" / "crazyhouse.json").read_text(encoding="utf-8")))
        evaluator = Evaluator.for_spec(engine.spec)
        held, capture_delta, drop_delta = _capture_then_redrop(engine, evaluator, seed=11)

        worth = evaluator.values[held]
        assert worth > 0.0
        # The capture changes the piece's owner: it leaves their board and enters my hand.
        assert capture_delta == pytest.approx(2.0 * worth)
        # The drop moves a piece its owner already had, and pays nothing for the act.
        assert drop_delta == pytest.approx(0.0, abs=1e-9)
        # And 2v is the honest total, which every *h* shares -- after both plies the taker
        # has one more and the victim one fewer. *h* only ever moved the split.
        assert capture_delta + drop_delta == pytest.approx(2.0 * worth)

    def test_the_board_alone_still_splits_it_as_h_zero_did(self) -> None:
        """What the term read before A6, kept as the vocabulary's `material` column.

        Pinned so the two readings cannot be confused: the board alone pays v at each
        end, and it is the hand that turns that into 2v and nothing.
        """
        engine = Engine(json.loads((ROOT / "rulesets" / "crazyhouse.json").read_text(encoding="utf-8")))
        evaluator = Evaluator.for_spec(engine.spec)
        held, capture_delta, drop_delta = _capture_then_redrop(engine, evaluator, seed=11)
        same, board_capture, board_drop = _capture_then_redrop(
            engine, evaluator, seed=11, board_only=True
        )

        worth = evaluator.values[held]
        assert same == held, "the same seed stops at the same capture in both readings"
        assert board_capture == pytest.approx(worth)
        assert board_drop == pytest.approx(worth)
        # The hand is exactly the difference between the two readings at each end.
        assert capture_delta - board_capture == pytest.approx(worth)
        assert drop_delta - board_drop == pytest.approx(-worth)

    def test_every_from_captures_ruleset_does_it(self) -> None:
        """Four rulesets, one arithmetic -- so the term names a family, not a ruleset."""
        for name in ("crazyhouse", "crazyhouse-torus", "shogi", "dobutsu"):
            engine = Engine(json.loads((ROOT / "rulesets" / f"{name}.json").read_text(encoding="utf-8")))
            assert engine.spec.drops is not None and engine.spec.drops.from_captures
            evaluator = Evaluator.for_spec(engine.spec)
            held, capture_delta, drop_delta = _capture_then_redrop(
                engine, evaluator, seed=11
            )
            assert capture_delta == pytest.approx(2.0 * evaluator.values[held]), name
            assert drop_delta == pytest.approx(0.0, abs=1e-9), name

    def test_the_material_term_sees_a_stocked_hand(self) -> None:
        """`archive/three-stones` states the same term without a capture in it.

        Its hands start full: 1.5000 between the two sides under the derived table, 0.75
        a side. The opening balance is still exactly zero, because the two hands are
        equal -- and a drop, which used to pay the mover its stone, now pays nothing.
        That is why pricing the hand closed case 2 and case 3 of the
        `docs/reference/bugs-investigations.md` entry on drops with one term, and it is
        the fifth of the five specs the term moves.
        """
        engine = Engine(json.loads((ROOT / "archive" / "three-stones.json").read_text(encoding="utf-8")))
        evaluator = Evaluator.for_spec(engine.spec)
        state = engine.initial_state()

        assert evaluator.material(state, 0) == 0.0
        assert evaluator.hand_value(state.hands, 0) == 0.0
        in_hand = sum(
            evaluator.values[piece_type] * count
            for side in state.hands
            for piece_type, count in enumerate(side)
        )
        assert in_hand == pytest.approx(1.5)

        mover = state.side_to_move
        drop = next(
            move for move in engine.legal_moves(state) if move_type(move) == MOVE_DROP
        )
        dropped = engine.apply(state, drop)
        assert evaluator.hand_value(dropped.hands, mover) == pytest.approx(-0.25)
        assert evaluator.material(dropped, mover) == pytest.approx(0.0, abs=1e-12)

    # Q787's three showcases: `pawn-knockdown` declares no `drops` (72), and the two
    # shogi ones hold a hand filled from captures, so they move with their parent.
    # Q781's `einstein` and `omega` declare none (74), and Q783's `extinction-chess`
    # and `grand` none either (76), and Q786's `jeson-mor` none (77), and Q782's
    # `hex-glinski-turned` and `-turned-mirrored` none (79), and Q772a's seven portal
    # showcases none (86), and Q807's `ferry-no-straight-back` none (87), and Q850's
    # `wolves-patrol` none (88), and Q772e's `divide-two-way` none (89), and Q772f's
    # `divide-fixed` none (90): every spec not named below, 90 of the 101. The library
    # audit (Q868, 2026-10-12) retired fourteen of the ninety, none holding a hand: 76 of 87.
    # Its second pass (Q870, 2026-10-13) retired four hexagon variants for `pinwheel`, none
    # holding a hand, and `pinwheel` declares none: 73 of 84.
    INERT = ("stones",)
    MOVED = (
        "crazyhouse",
        "crazyhouse-torus",
        "dobutsu",
        # Q785's `kyoto` holds a hand filled from captures.
        "kyoto",
        # Q784's two placement showcases hold their armies in a stock.
        "placement-chess",
        "shogi",
        "shogi-backward-drops",
        "shogi-drop-promoted",
        "sittuyin",
        "three-stones",
    )

    def test_the_three_kinds_are_counted_on_the_whole_corpus(self) -> None:
        assert len(CORPUS_PATHS) == 98, "the corpus moved out from under this test"  # 98 since Q936s
        stems = {path.stem for path in CORPUS_PATHS}
        assert set(self.INERT) | set(self.MOVED) <= stems
        assert len(stems) - len(self.INERT) - len(self.MOVED) == 87  # 73 until Q844b's three portal showcases; 76 until Q918e's four round showcases, which hold no hand; 80 until Q942c's four, which hold none either; 84 until Q935f's two spy showcases, which hold none; 86 until Q936s's riders and leapers, which holds none

    # One test per spec (Q749): the walk was one test, 4.4 s in the gate once the two
    # walks above stopped compiling the corpus first in the same worker.
    @pytest.mark.parametrize("path", CORPUS_PATHS, ids=lambda path: path.stem)
    def test_the_term_is_inert_on_everything_that_holds_nothing(self, path: Path) -> None:
        """The blast radius, over the corpus rather than by argument: 64, 1 and 5.

        64 of the 70 committed and archived specs declare no `drops` at all, so the hand
        loop never executes on them -- bit-identical, not merely equal. `stones`
        holds a hand and is inert too, because its only stocked type is priced at
        `INEXHAUSTIBLE_VALUE`. The five that move are the four `from_captures` rulesets
        and `three-stones`. The 61st spec, `weathervane-chaos-torus` (2026-09-11), the
        62nd, `weathervane-half-turn` (2026-09-13), and the 63rd, `wolves-bounty`
        (2026-09-13), declare no `drops` either, so they join the plain count rather than
        moving the other two, as do the 64th, `wolves-rotated` (2026-09-14), and the 65th
        to 67th, its three sibling rotations (2026-09-14). The 68th, `merger` (schema 11),
        the 69th, `hierarchy-rise-throw` (schema 12), the 70th,
        `chess-sweeping-bishops` (schema 13), and the 71st, `hex-glinski-promoting`
        (schema 14), declare no `drops` either. Q280 retired `hex-glinski` on 2026-09-17,
        which declared none either, so the no-hand count fell from 66 to 65, and Q434
        retired `tunnel-horizontal` on 2026-09-25, which took it to 64. Q281a4's three
        hexagon showcases (schema 16) declare none either, which takes it to 67, and Q540's
        four schema-17 showcases to 71. Q781's two schema-18 showcases declare none either,
        and nor do Q783's two, nor Q786's `jeson-mor`, nor Q782's two hex showcases.
        """
        # Off the compiled spec rather than off a position, for the reason
        # `inexhaustible_types` is: the question is "can this hold anything", which
        # the rules settle, not "does it hold anything now".
        spec = compile_spec(json.loads(path.read_text(encoding="utf-8")))
        if spec.drops is None:
            kind = "no hand"
        else:
            values = Evaluator.for_spec(spec).values
            opening = 2.0 * sum(
                values[piece_type] * count for piece_type, count in enumerate(spec.drops.stock)
            )
            kind = "inert" if opening == 0.0 and not spec.drops.from_captures else "moved"
        expected = (
            "inert" if path.stem in self.INERT
            else "moved" if path.stem in self.MOVED
            else "no hand"
        )
        assert kind == expected, path.stem


class _Spec:
    """The one attribute `Session._explosion_info` reads, without starting a session."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec


# ------------------------------------------------------- Q19a: the contact table


def _fall() -> dict[str, Any]:
    """`archive/hierarchy-fall.json` as committed: a demotion chain under schema 6."""
    return json.loads((ROOT / "archive" / "hierarchy-fall.json").read_text(encoding="utf-8"))


def _fall_as_contacts() -> dict[str, Any]:
    """The same game, re-spelled as a contact table with no attacker column.

    Every mover meets the same row, which is what "keyed by the victim alone" means --
    so this must compile to the bytes `hierarchy-fall` already compiles to. It is the
    test Q86 said to write first, and the whole claim that `contacts` *generalises*
    `strike` rather than sitting beside it rests on it.
    """
    child = copy.deepcopy(_fall())
    child["schema_version"] = 10
    strike = child["rules"].pop("strike")
    movers = [piece["id"] for piece in child["pieces"]]
    child["rules"]["contacts"] = {
        "position": strike["position"],
        "immune": list(strike["immune"]),
        "links": [
            {"mover": mover, "onto": link["from"], "onto_to": link["to"]}
            for mover in movers
            for link in strike["chain"]
        ],
    }
    return child


def _contacts(
    chess: dict[str, Any], links: list[dict[str, Any]], position: str = "advance"
) -> dict[str, Any]:
    """chess plus a contact table, stamped for schema 10."""
    child = copy.deepcopy(chess)
    child["schema_version"] = 10
    child["rules"]["contacts"] = {
        "position": position,
        "immune": ["K"],
        "links": links,
    }
    return child


class TestTheContactTable:
    """`rules.contacts`: (mover, target) -> (what the mover becomes, what it becomes).

    The largest of schema 10's five and the only one that is a new key under `rules`. It
    is one mechanism with `strike`, not a second beside it: the two are refused together,
    they compile into one `StrikeRule`, and the boundary sends schema 6's single array
    whenever the table has no attacker column -- which is what keeps every strike ruleset
    byte-identical while the core carries only the general rule.

    `own: true` -- a contact with a *friendly* piece -- is deliberately not here. That is
    the half that moves every perft, because a friendly target is a new mode bit and a
    third arm in the generator, and it takes a version of its own.
    """

    def test_a_strike_re_spelled_compiles_to_the_same_bytes(self) -> None:
        """The claim, checked the way Q78's was: identical canonical tables.

        Not "plays the same" -- the same bytes cross the boundary, so the core cannot be
        told the two apart. Anything weaker would be checking that two readings of one
        rule happen to agree today.

        The Zobrist draw is the one expected exception, exactly as it is for Q78's
        re-spelling: the seed is the mechanics hash and the hash covers the spelling, so
        a spec that says the same thing differently lays its transposition table out
        differently. Same game, same distribution, not the same individual games.
        """
        assert validate_spec(_fall_as_contacts()) == []
        original = build_tables(compile_spec(_fall()))
        respelled = build_tables(compile_spec(_fall_as_contacts()))
        assert respelled["strike_to"] == original["strike_to"]
        assert respelled["strike_position"] == original["strike_position"]
        differing = {key for key in original if original[key] != respelled[key]}
        assert differing == {
            "zobrist_pieces",
            "zobrist_side",
            "zobrist_ep",
            "zobrist_colour",
            "zobrist_turn",
        }

    def test_the_re_spelling_is_a_different_spec_and_the_same_game(self) -> None:
        """The hash moves, because the file changed; nothing about the play does."""
        assert spec_hash(_fall_as_contacts()) != spec_hash(_fall())
        engine, respelled = Engine(_fall()), Engine(_fall_as_contacts())
        assert engine.perft(engine.initial_state(), 3) == respelled.perft(
            respelled.initial_state(), 3
        )

    def test_the_compact_form_is_the_projection_and_not_a_second_table(
        self, chess: dict[str, Any]
    ) -> None:
        """`compact_to` exists exactly when the attacker's column is empty and the rows agree."""
        compiled = compile_spec(_fall_as_contacts())
        assert compiled.strike is not None
        assert compiled.strike.compact_to == compile_spec(_fall()).strike.compact_to
        assert not compiled.strike.changes_mover

        pair = compile_spec(_contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}]))
        assert pair.strike is not None
        assert pair.strike.changes_mover
        assert pair.strike.compact_to is None

        # Two movers disagreeing about one victim is the other way to need the pair.
        vampire = compile_spec(
            _contacts(
                chess,
                [{"mover": "N", "onto": "P", "onto_to": "R"}, {"mover": "B", "onto": "P", "onto_to": "Q"}],
                position="stay",
            )
        )
        assert vampire.strike is not None and vampire.strike.compact_to is None

    def test_the_boundary_sends_one_encoding_or_the_other(
        self, chess: dict[str, Any]
    ) -> None:
        """The inert default, which is what keeps every earlier fingerprint still."""
        plain = canonical_tables(compile_spec(chess))
        assert not {"contact_onto_to", "contact_mover_to"} & set(plain)

        respelled = build_tables(compile_spec(_fall_as_contacts()))
        assert respelled["contact_onto_to"] == [] and respelled["contact_mover_to"] == []
        assert len(respelled["strike_to"]) == compile_spec(_fall()).n_types

        pair = build_tables(
            compile_spec(_contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}]))
        )
        n = compile_spec(chess).n_types
        assert pair["strike_to"] == []
        assert len(pair["contact_onto_to"]) == n * n
        assert len(pair["contact_mover_to"]) == n * n

    def test_an_older_stamp_is_refused_by_name(self, chess: dict[str, Any]) -> None:
        spec = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])
        spec["schema_version"] = 9
        errors = validate_spec(spec)
        assert any("rules.contacts" in error and "schema 10" in error for error in errors)

    def test_the_two_spellings_are_refused_together(self, chess: dict[str, Any]) -> None:
        spec = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])
        spec["rules"]["strike"] = {
            "effect": "demote",
            "position": "stay",
            "chain": [{"from": "Q", "to": "R"}],
            "immune": ["K"],
        }
        errors = validate_spec(spec)
        assert any("two spellings of one rule" in error for error in errors), errors

    @pytest.mark.parametrize(
        "other, value",
        [
            ("explode_on_capture", {"radius": 1}),
            ("drops", {"from_captures": True}),
            ("npcs", [{"piece": "P", "route": ["d4", "d5"]}]),
        ],
    )
    def test_it_is_refused_beside_the_other_rules_about_a_taken_piece(
        self, chess: dict[str, Any], other: str, value: Any
    ) -> None:
        """`_check_strike_coherence`'s refusals, restated over this spelling.

        Restated rather than shared, because an error naming `rules.strike` for a file
        that says `rules.contacts` points a reader at a rule that is not there.
        """
        spec = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])
        spec["rules"][other] = value
        errors = validate_spec(spec)
        assert any(
            f"rules.contacts and rules.{other}" in error for error in errors
        ), errors

    def test_a_link_that_changes_nothing_is_refused(self, chess: dict[str, Any]) -> None:
        errors = validate_spec(_contacts(chess, [{"mover": "P", "onto": "R"}]))
        assert any("ordinary capture written the long way" in e for e in errors), errors

    def test_a_table_with_no_link_is_no_rule(self, chess: dict[str, Any]) -> None:
        """The Q56 ruling (`docs/reference/bugs-investigations.md`), held on all three surfaces at once."""
        spec = _contacts(chess, [])
        assert any("non-empty" in error for error in validate_spec(spec)), validate_spec(spec)
        # And whatever the validator says, the compiler and the cost table agree with
        # each other: no rule, and nothing charged for one.
        spec["rules"]["contacts"] = {"links": []}
        assert compile_spec(chess).strike is None
        assert measure_complexity(spec).rule_mass == measure_complexity(chess).rule_mass

    def test_advance_may_transform_the_mover_but_not_keep_the_target(
        self, chess: dict[str, Any]
    ) -> None:
        """The one asymmetry the position rules gained, and it is not an oversight.

        A survivor needs a square and `advance` has none, which is schema 6's refusal.
        The attacker's half needs no square -- it is already standing on the target -- so
        `advance` is exactly where "a capture promotes the capturer" wants to be said.
        """
        assert validate_spec(_contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])) == []
        errors = validate_spec(_contacts(chess, [{"mover": "P", "onto": "R", "onto_to": "N"}]))
        assert any("nowhere to put a target that survives" in e for e in errors), errors

    def test_a_royal_has_to_be_immune_on_both_columns(self, chess: dict[str, Any]) -> None:
        bare = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])
        bare["rules"]["contacts"]["immune"] = []
        assert any("has to be named immune" in e for e in validate_spec(bare)), bare

        demoted_king = _contacts(chess, [{"mover": "K", "onto": "R", "mover_to": "N"}])
        errors = validate_spec(demoted_king)
        assert any("a royal that stops being royal" in e for e in errors), errors

    def test_an_undeclared_id_is_refused(self, chess: dict[str, Any]) -> None:
        errors = validate_spec(_contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "Z"}]))
        assert any("'Z' is not a declared piece" in error for error in errors), errors

    def test_own_is_refused_by_name_under_this_stamp(self, chess: dict[str, Any]) -> None:
        """Q19b's half took schema 11, because a friendly target moves perft.

        `tests/test_schema_eleven.py` holds the feature; this holds the stamp.
        """
        spec = _contacts(chess, [{"mover": "P", "onto": "R", "own": True, "mover_to": "Q"}])
        errors = validate_spec(spec)
        assert any("schema 11" in error for error in errors), errors

    def test_a_contact_makes_a_second_royal_the_compiler_can_see(
        self, chess: dict[str, Any]
    ) -> None:
        """Both columns, because a meeting can crown the attacker as easily as the victim."""
        crowned = compile_spec(
            _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "K"}])
        )
        assert crowned.multi_royal
        assert compile_spec(_fall_as_contacts()).multi_royal is False

    def test_the_core_plays_the_mover_half(self, chess: dict[str, Any]) -> None:
        """`1. e4 d5 2. exd5` and what stands on d5 afterwards.

        Three spellings on one line, so the difference cannot be anything but the rule:
        a pawn that becomes a knight by taking a pawn; the same table keyed on a knight
        instead, which this line never reaches; and the target's half, where the pawn
        stays on e4 and a black knight stands on d5.
        """
        line = [("e2", "e4"), ("d7", "d5"), ("e4", "d5")]

        def after(spec: dict[str, Any], square: str) -> str | None:
            engine = Engine(spec)
            state = engine.initial_state()
            for origin, target in line:
                state = engine.apply(state, _move(engine, state, origin, target))
            value = state.board[_square(square)]
            return None if value == 0 else engine.spec.piece_ids[engine.spec.type_of(value)]

        assert after(chess, "d5") == "P"
        becomes = _contacts(chess, [{"mover": "P", "onto": "P", "mover_to": "N"}])
        assert after(becomes, "d5") == "N", "the mover took the square and changed"
        keyed_on_a_knight = _contacts(chess, [{"mover": "N", "onto": "P", "mover_to": "Q"}])
        assert after(keyed_on_a_knight, "d5") == "P", "no link for a pawn taking a pawn"
        standing = _contacts(
            chess, [{"mover": "P", "onto": "P", "onto_to": "N"}], position="stay"
        )
        assert after(standing, "e4") == "P" and after(standing, "d5") == "N"

    def test_the_legality_filter_reads_the_same_table(self, chess: dict[str, Any]) -> None:
        """`make` is not the only site, and a perft is the sharpest way to say so.

        Every node of a perft goes through `play_for_filter`, so a filter that read the
        table differently from `make` would count a different number of leaves. The
        re-spelling gives a reference number that is not a literal: whatever
        `hierarchy-fall` counts, its contact table has to count the same.
        """
        original, respelled = Engine(_fall()), Engine(_fall_as_contacts())
        for depth in (1, 2, 3):
            assert respelled.perft(respelled.initial_state(), depth) == original.perft(
                original.initial_state(), depth
            )
        # And a table with an attacker column survives make/unmake, which is what a
        # perft over it actually tests.
        mover = Engine(_contacts(chess, [{"mover": "P", "onto": "P", "mover_to": "N"}]))
        assert mover.perft(mover.initial_state(), 3) == Engine(chess).perft(
            Engine(chess).initial_state(), 3
        )

    def test_a_short_game_keeps_its_incremental_hash(self, chess: dict[str, Any]) -> None:
        """The walk `new-move-primitive` asks for, on inline specs.

        A contact writes up to three squares into the Zobrist key and `unmake` restores
        them from the undo log; the attacker's half writes a square that schema 6 never
        wrote under `advance`, which is exactly where the two can disagree. A drifted key
        shows up here long before it shows up as a wrong move count.
        """
        for spec in (
            _fall_as_contacts(),
            _contacts(chess, [{"mover": "P", "onto": "P", "mover_to": "N"}]),
            _contacts(chess, [{"mover": "P", "onto": "P", "onto_to": "N"}], position="stay"),
            _contacts(
                chess,
                [{"mover": "P", "onto": "N", "mover_to": "B", "onto_to": "P"}],
                position="swap",
            ),
        ):
            engine = Engine(spec)
            rng = random.Random(11)
            state = engine.initial_state()
            for _ in range(24):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                for move in moves:
                    engine.apply(state, move)  # no panic, no illegal write
                state = engine.apply(state, rng.choice(moves))
                assert state.key == engine._compute_hash(state), "hash drifted"

    def test_the_record_names_what_the_capture_did(self, chess: dict[str, Any]) -> None:
        """A contact is generated as a capture and written as one, so the text has to say more.

        Keyed by the pair like everything else: the suffix a reader sees depends on who
        arrived, not only on who was standing there.
        """
        spec = _contacts(
            chess, [{"mover": "P", "onto": "P", "mover_to": "N", "onto_to": "B"}], position="stay"
        )
        engine = Engine(spec)
        state = engine.initial_state()
        for origin, target in [("e2", "e4"), ("d7", "d5")]:
            state = engine.apply(state, _move(engine, state, origin, target))
        text = engine.describe_move(state, _move(engine, state, "e4", "d5"))
        assert "->B" in text and "P->N" in text and "(stands fast)" in text

    def test_the_description_says_the_table_one_meeting_at_a_time(
        self, chess: dict[str, Any]
    ) -> None:
        from chess_variant_sandbox.spec.describe import _describe_contacts

        spec = _contacts(
            chess, [{"mover": "P", "onto": "R", "mover_to": "N", "onto_to": "B"}], position="swap"
        )
        lines = _describe_contacts(spec, spec["rules"])
        assert any("pawn takes rook" in line for line in lines), lines
        assert any(
            "the pawn becomes knight and the rook becomes bishop" in line for line in lines
        ), lines
        assert any("exchange squares" in line for line in lines)
        assert any("king cannot be reached by a contact" in line for line in lines)
        # A table with nothing in it is described as no rule, matching the compiler.
        assert _describe_contacts(chess, {"contacts": {"links": []}}) == []

    def test_it_has_distance_features_of_its_own(self, chess: dict[str, Any]) -> None:
        """A re-spelling is a neighbour of what it re-spells; the new axes measure apart."""
        assert {"contacts_by_pair", "contacts_change_the_mover"} <= set(_SCALES)
        assert features(_fall_as_contacts())["strikes"] == 1.0
        # Every behavioural axis agrees -- it is the same game -- and the only feature
        # that moves is what the rule costs to read, because a table asks for the pair
        # where a chain asks for the victim. So the two are neighbours rather than identical, which is the
        # honest answer: the loop should not treat a re-spelling as a rediscovery, and it
        # should not run a tournament to find out they play the same either.
        original, respelled = features(_fall()), features(_fall_as_contacts())
        assert {
            key for key in original if original[key] != respelled[key]
        } == {"rule_mass", "special_rules"}
        assert 0.0 < distance(_fall(), _fall_as_contacts()) < 0.01

        pair = _contacts(
            chess,
            [{"mover": "N", "onto": "P", "onto_to": "R"}, {"mover": "B", "onto": "P", "onto_to": "Q"}],
            position="stay",
        )
        assert features(pair)["contacts_by_pair"] == 1.0
        assert features(pair)["contacts_change_the_mover"] == 0.0
        mover = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])
        assert features(mover)["contacts_change_the_mover"] == 1.0
        assert features(mover)["strike_promotes"] == 1.0, "a pawn becoming a knight rises"
        assert features(_fall_as_contacts())["strike_promotes"] == 0.0
        assert distance(pair, mover) > 0.0

    @pytest.mark.parametrize("position", ["advance", "stay", "swap"])
    def test_a_table_of_only_immune_targets_measures_as_no_contact(
        self, chess: dict[str, Any], position: str
    ) -> None:
        """Q417: a table the compiler drops is chess, and measures zero from it.

        Every link below lands on the immune king, and a contact never reaches an immune
        target, so the table is plain capture written the long way although both links
        name a change. The compiler drops it and `complexity` charges nothing for it, but
        `distance` read any table with a link as a strike. The live neighbour changes only
        the mover, which is the table the survivor question alone would miss.
        """
        dead = _contacts(
            chess,
            [
                {"mover": "P", "onto": "K", "mover_to": "Q"},
                {"mover": "N", "onto": "K", "onto_to": "Q"},
            ],
            position=position,
        )
        # Refused since Q538 as stating nothing; `distance` still reads it as chess for a
        # spec it meets before the validator does.
        from chess_variant_sandbox.spec.schema import PLAYED_AS_CAPTURE

        (refusal,) = validate_spec(dead)
        assert PLAYED_AS_CAPTURE in refusal, refusal
        assert features(dead) == features(chess)
        assert distance(dead, chess) == 0.0

        live = _contacts(chess, [{"mover": "P", "onto": "P", "mover_to": "Q"}], position=position)
        assert validate_spec(live) == []
        assert compile_spec(live).strike is not None
        assert features(live)["strikes"] == 1.0
        assert features(live)["contacts_change_the_mover"] == 1.0
        assert distance(live, chess) > 0.0

    def test_a_link_the_compiler_drops_sets_no_feature_beside_a_live_one(
        self, chess: dict[str, Any]
    ) -> None:
        """Q417: the features are read off the links the compiler keeps.

        The second link lands on the immune king and names a rising mover, so it used to
        set `contacts_change_the_mover` and `strike_promotes` beside a live link that
        leaves the pawn a pawn and demotes the rook it takes. The two tables compile to
        the same rule, and measure as one game.
        """
        live = [{"mover": "P", "onto": "R", "onto_to": "B"}]
        mixed = [*live, {"mover": "N", "onto": "K", "mover_to": "Q"}]
        alone, both = _contacts(chess, live, "stay"), _contacts(chess, mixed, "stay")
        assert validate_spec(both) == []
        assert compile_spec(both).strike == compile_spec(alone).strike
        assert features(both)["contacts_change_the_mover"] == 0.0
        assert features(both)["strike_promotes"] == 0.0
        assert features(both) == features(alone)

    def test_it_is_charged_for_what_it_asks_a_player_to_learn(
        self, chess: dict[str, Any]
    ) -> None:
        spec = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])
        # `rule_mass` weights a special rule by six, so 2.5 in `_RULE_COSTS` is 15 here.
        assert measure_complexity(spec).special_rules == pytest.approx(
            measure_complexity(chess).special_rules + 2.5
        )
        assert measure_complexity(spec).rule_mass == pytest.approx(
            measure_complexity(chess).rule_mass + 15.0
        )
        # And half a point more than the chain it generalises, which is what the pair
        # key asks a player to learn.
        assert measure_complexity(_fall_as_contacts()).special_rules == pytest.approx(
            measure_complexity(_fall()).special_rules + 0.5
        )

    def test_fairy_refuses_it_by_name(self, chess: dict[str, Any]) -> None:
        """Q94's guard, working as designed: the key is listed *and* the branch decides.

        The refusal names the key the spec wrote. `compiled.strike` holds both spellings,
        so a message hard-coded to `rules.strike` would send a reader of a `contacts` file
        looking for a rule that is not in it.
        """
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(_contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}])))
        assert any("rules.contacts" in reason for reason in refusal.value.reasons)
        with pytest.raises(Untranslatable) as strike_refusal:
            translate(compile_spec(_fall()))
        assert any("rules.strike" in reason for reason in strike_refusal.value.reasons)

    def test_the_json_schema_offers_it_to_a_proposer(self) -> None:
        contacts = rule_spec_schema()["properties"]["rules"]["properties"]["contacts"]
        link = contacts["properties"]["links"]["items"]
        # `own` arrived with schema 11 (Q19b).
        assert {"mover", "onto", "mover_to", "onto_to", "own"} == set(link["properties"])
        assert set(link["required"]) == {"mover", "onto"}

    def test_the_bridge_refuses_a_core_that_cannot_answer(self) -> None:
        """The hold, and the record of when it came off."""
        from chess_variant_sandbox.engine import bridge

        assert bridge._CORE_KNOWS_CONTACTS is True
        assert rust_supports(compile_spec(_fall_as_contacts())) is None


class TestTheBumpIsInertOnEveryRulesetThatDoesNotUseIt:
    """The acceptance criterion, spelled where a reader of this file will look for it.

    The corpus-wide proof is `test_schema_nine.py::TestTheBumpMovedNothing` -- every
    committed and archived ruleset's mechanics hash *and* compiled tables, pinned. This
    is the one number that matters most, kept here so the bump's own file states it.
    """

    def test_chess_still_hashes_to_what_it_always_did(self, chess: dict[str, Any]) -> None:
        assert spec_hash(chess) == "4c1922ffae26"

    def test_every_committed_ruleset_still_validates(self) -> None:
        for path in sorted((ROOT / "rulesets").glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            assert validate_spec(spec) == [], path.name

    def test_no_calibration_ruleset_uses_anything_the_bump_added(self) -> None:
        """If one did, its hash would have to move, and the acceptance criterion says
        none does -- except where moving it was the point.

        Read over `rulesets/` alone since 2026-09-07. It covered `archive/` too while the
        bump had no showcase, which was the same statement said twice; Q101 then added
        seven archived games that use the bump on purpose (rows 23--30 of
        `reference/schema-showcase-coverage.md`), and the criterion this test exists for
        is about the games that were already there. The archive half is the test below,
        which names them.

        **`intransitive` is now a named user and not a drift** (Q120, 2026-09-07). Schema
        10 exists partly for it: its designer states there is no draw by repetition, the
        committed spec had no `repetition_draw` key and so read as the default 3, and
        `REPETITION_NEVER`'s own docstring names this ruleset as the spec that had to
        misstate itself. Correcting it moved the hash `09861015bb3a` -> `d93e28d09a33`
        deliberately, and `runs/09861015bb3a-0` is kept as the record of the spec that no
        longer exists. Asserted as a mapping rather than skipped by name, so a *second*
        committed ruleset reaching for the bump is still a named failure here.
        """
        found = {
            path.name: sorted(_schema_ten_uses(json.loads(path.read_text(encoding="utf-8"))))
            for path in sorted((ROOT / "rulesets").glob("*.json"))
            if _schema_ten_uses(json.loads(path.read_text(encoding="utf-8")))
        }
        assert found == {"intransitive.json": ['rules.repetition_draw "never"']}

    def test_exactly_the_showcases_use_the_bump(self) -> None:
        """The seven, by name, so an eighth is a deliberate act rather than a drift.

        A showcase demonstrates one feature (`schema-ten-showcases.md`), so this table is
        also the assertion that no archived spec quietly acquired a second one -- schema
        9's first showcase moved four things at once and could not say which moved the
        index.
        """
        found = {
            path.stem: sorted(_schema_ten_uses(json.loads(path.read_text(encoding="utf-8"))))
            for path in sorted((ROOT / "archive").glob("*.json"))
            if _schema_ten_uses(json.loads(path.read_text(encoding="utf-8")))
        }
        assert found == {
            # Schema 11's showcase rides on a contact table, so it uses this bump's key.
            "merger": ["rules.contacts"],
            # Q787's pawn-knockdown rides on a contact table too (schema 12's `choose`).
            "pawn-knockdown": ["rules.contacts"],
            # Q781's einstein is a contact table, the keeping link schema 18 lifted in it.
            "einstein": ["rules.contacts"],
            "perpetual": ['rules.repetition_draw "never"'],
            # Q784's placement showcases (schema 18) hold their armies in a finite stock.
            "placement-chess": ["rules.drops.stock"],
            "sittuyin": ["rules.drops.stock"],
            "queen-bomb": ["rules.explode_on_capture.by"],
            "queens-wake": ["rules.explode_on_capture.of"],
            "stones": ["a piece with no moves of its own", "rules.drops.stock"],
            "succession": ["rules.contacts"],
            "three-stones": ["a piece with no moves of its own", "rules.drops.stock"],
            # Q876: the game built around the word, for which `wolves-immune` was retired.
            "pitchfork": ["rules.npcs[].immune"],
            # Q772a's two showcases whose portals nobody moves declare the portal type with
            # no moves, schema 19's second exception to schema 10's rule.
            "carousel": ["a piece with no moves of its own"],
            "cylinder-wormhole": ["a piece with no moves of its own"],
        }


# ------------------------------------------------ Q922: what an explosion block is read as


@pytest.mark.parametrize(
    ("explosion", "features_of_it"),
    [
        (None, (0.0, 0.0, 0.0)),
        ({}, (0.0, 0.0, 0.0)),
        ("yes", (0.0, 0.0, 0.0)),
        ([1], (0.0, 0.0, 0.0)),
        (True, (0.0, 0.0, 0.0)),
        ({"by": []}, (0.0, 0.0, 0.0)),
        ({"of": []}, (0.0, 0.0, 0.0)),
        ({"by": ["N"], "of": []}, (0.0, 0.0, 0.0)),
        ({"radius": 1}, (1.0, 0.0, 0.0)),
        ({"by": ["N"]}, (1.0, 1.0, 0.0)),
        ({"of": ["N"], "radius": 0}, (1.0, 1.0, 1.0)),
        ({"radius": 0}, (1.0, 0.0, 1.0)),
    ],
    ids=str,
)
def test_an_explosion_block_is_read_as_the_compiler_reads_it(
    explosion: Any, features_of_it: tuple[float, float, float]
) -> None:
    from chess_variant_sandbox.spec.distance import _explosion_features

    got = _explosion_features(explosion)
    assert (got["explodes"], got["explodes_selectively"], got["explodes_alone"]) == features_of_it


# ------------------------------------------------ Q922: three refusals' edges


def _plain_chess() -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


class TestTheClocksAreCheckedPastANeverRepetition:
    """`repetition_draw: "never"` is one clock's own spelling; the others are still read."""

    @pytest.mark.parametrize("key", ["max_plies", "no_progress_draw"])
    @pytest.mark.parametrize("bad", [0, 1, "soon", True, 10_001])
    def test_a_bad_clock_is_refused_beside_never(self, key: str, bad: Any) -> None:
        spec = _never(_plain_chess())
        spec["rules"][key] = bad
        assert any(error.startswith(f"rules.{key} must") for error in validate_spec(spec)), key

    def test_never_beside_good_clocks_is_accepted(self) -> None:
        spec = _never(_plain_chess())
        spec["rules"].update({"max_plies": 200, "no_progress_draw": 50})
        assert validate_spec(spec) == []


class TestARoyalMetByAContactIsNamedOnItsSide:
    @staticmethod
    def _errors(links: list[dict[str, Any]]) -> list[str]:
        spec = _contacts(_plain_chess(), links)
        spec["rules"]["contacts"]["immune"] = []
        return [e for e in validate_spec(spec) if e.startswith("the royal piece 'K' is not in")]

    def test_a_royal_the_table_meets_is_named_as_met(self) -> None:
        (error,) = self._errors([{"mover": "Q", "onto": "K", "onto_to": "Q"}])
        assert error.endswith("-- and rules.contacts.links names 'K' on the 'onto' side")

    def test_a_royal_the_table_never_meets_is_not(self) -> None:
        (error,) = self._errors([{"mover": "Q", "onto": "R", "onto_to": "B"}])
        assert error.endswith("so every royal piece has to be named immune")


class TestAnImmuneWildPieceBesideADropsBlock:
    @staticmethod
    def _errors(drops: dict[str, Any]) -> list[str]:
        spec = _hunted(_wolves())
        spec["rules"]["drops"] = drops
        return [e for e in validate_spec(spec) if ".immune beside rules.drops" in e]

    @pytest.mark.parametrize(
        ("drops", "refused"),
        [
            ({"stock": {"P": 1}}, True),                         # captures fill a hand by default
            ({"from_captures": True}, True),
            ({"from_captures": False}, False),
            ({"from_captures": False, "stock": {"P": 1}}, False),  # a stock of another type
            ({"from_captures": False, "stock": {"W": 1}}, True),   # the wild type itself, stocked
        ],
        ids=str,
    )
    def test_when_the_wild_piece_can_reach_a_hand(self, drops: dict, refused: bool) -> None:
        spec = _hunted(_wolves())
        wild = spec["rules"]["npcs"][0]["piece"]
        drops = json.loads(json.dumps(drops).replace('"W"', json.dumps(wild)))
        assert bool(self._errors(drops)) is refused
