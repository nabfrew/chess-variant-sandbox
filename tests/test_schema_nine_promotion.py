"""Schema 9, the second key: a piece that enters play inside its promotion zone promotes.

`rules.promote_on_entry` is the user's answer to the first entry `docs/bugs.md` carried (now in
`docs/archive/bugs-resolved.md`) --
"pawns that spawn from promotion/demotion in a promotion zone do not promote; same for
drops" -- and it rides schema 9 beside `rules.npcs` rather than paying a bump of its own
(`docs/archive/plan-schema-6-9.md` C7). What is asserted here is the half that exists
before the core can play it: the flag at its inert default compiles to the tables that
were already there
(the corpus-wide pin is `tests/test_schema_nine.py`'s frozen table, which this key joins
without a row moving), the validator refuses what it says it refuses, the compiler
resolves the one table the core will read, `bridge.rust_supports` holds the intermediate
state shut behind the *same* flag as the wild pieces so the version opens once, and every
surface learned the rule. The core half is Q9b's, specified in
`docs/archive/plan-schema-6-9.md` C6.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.engine import Engine, bridge
from chess_variant_sandbox.engine.bridge import canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import promotion_squares, promotion_targets

ROOT = Path(__file__).resolve().parents[1]


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _with_flag(path: str) -> dict[str, Any]:
    spec = load_spec(ROOT / path)
    spec["schema_version"] = 9
    spec["rules"]["promote_on_entry"] = True
    return spec


@pytest.fixture
def fall() -> dict[str, Any]:
    """`hierarchy-fall` with the flag: the case the bug entry was written about.

    Its chain ends `N -> P` under `position: stay`, so a knight struck on the far rank
    becomes a pawn standing in its own promotion zone -- and, without the flag, a pawn
    that can never move again.
    """
    return _with_flag("archive/hierarchy-fall.json")


@pytest.fixture
def shogi_entry() -> dict[str, Any]:
    """Drops with no region and *optional* promotion over a three-rank zone.

    Stamped 9, so schema 15's `on_leaving`, on the committed file since Q454, is taken
    off: this is the shogi that promotes on arriving, which is what schema 9 could say.
    """
    spec = _with_flag("rulesets/shogi.json")
    for rule in spec["rules"]["promotion"]:
        rule.pop("on_leaving", None)
    return spec


@pytest.fixture
def dobutsu_entry() -> dict[str, Any]:
    """Drops with no region and a mandatory single-target rule on the far rank."""
    return _with_flag("rulesets/dobutsu.json")


def _without(spec: dict[str, Any]) -> dict[str, Any]:
    without = copy.deepcopy(spec)
    del without["rules"]["promote_on_entry"]
    return without


# ------------------------------------------------------------------ the bump moved nothing


class TestTheBumpMovedNothing:
    def test_false_is_the_game_that_was_already_there(self, chess_spec: dict[str, Any]) -> None:
        """The inert default: no rule compiled, nothing on the boundary.

        Every table but the Zobrist draw is byte-identical to plain chess's; the draw is
        seeded by the mechanics hash, which the version stamp and the key move, and that
        is the hash doing its job rather than the compiler doing anything.
        """
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        spec["rules"]["promote_on_entry"] = False
        assert validate_spec(spec) == []
        compiled = compile_spec(spec)
        assert compiled.entry_promotion is None
        tables = canonical_tables(compiled)
        assert "entry_promotion" not in tables
        plain = canonical_tables(compile_spec(chess_spec))
        strip = lambda t: {k: v for k, v in t.items() if not k.startswith("zobrist")}  # noqa: E731
        assert strip(tables) == strip(plain)

    def test_the_resolvers_the_validator_shares_with_the_compiler_agree(
        self, fall: dict[str, Any]
    ) -> None:
        """`schema.promotion_squares` and `schema.promotion_targets` are the compiler's reading."""
        compiled = compile_spec(fall)
        rule = fall["rules"]["promotion"]
        orientation = fall["setup"].get("orientation", "mirror")
        assert promotion_squares(rule, 8, 8, orientation) == compiled.promotions[0].squares
        first, second = promotion_targets(rule)
        assert [compiled.piece_ids[t] for t in compiled.promotions[0].targets[0]] == first
        assert [compiled.piece_ids[t] for t in compiled.promotions[0].targets[1]] == second


# ---------------------------------------------------------------- what the validator says


class TestWhatTheValidatorRefuses:
    def test_a_surviving_strike_is_a_way_in(self, fall: dict[str, Any]) -> None:
        assert validate_spec(fall) == []

    def test_a_drop_with_no_region_is_a_way_in(
        self, shogi_entry: dict[str, Any], dobutsu_entry: dict[str, Any]
    ) -> None:
        assert validate_spec(shogi_entry) == []
        assert validate_spec(dobutsu_entry) == []

    def test_it_needs_a_promotion_rule(self, chess_spec: dict[str, Any]) -> None:
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        del spec["rules"]["promotion"]
        del spec["rules"]["en_passant"]
        spec["rules"]["drops"] = {"from_captures": True}
        spec["rules"]["promote_on_entry"] = True
        errors = validate_spec(spec)
        assert any("needs rules.promotion" in error for error in errors), errors

    def test_it_needs_a_way_in(self, chess_spec: dict[str, Any]) -> None:
        """Chess has a promotion rule and no way for a piece to arrive without moving."""
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        spec["rules"]["promote_on_entry"] = True
        errors = validate_spec(spec)
        assert any("nothing enters play except by moving" in error for error in errors), errors

    def test_a_capture_strike_is_not_a_way_in(self, chess_spec: dict[str, Any]) -> None:
        """`{"effect": "capture"}` compiles to nothing, so it lets nothing in either."""
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        spec["rules"]["strike"] = {"effect": "capture"}
        spec["rules"]["promote_on_entry"] = True
        errors = validate_spec(spec)
        assert any("nothing enters play except by moving" in error for error in errors), errors

    def test_a_link_from_an_immune_piece_is_dead(self, chess_spec: dict[str, Any]) -> None:
        """The validator's idea of a survivor is the compiler's: an immune `from` never strikes."""
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        spec["rules"]["strike"] = {
            "effect": "demote", "position": "stay", "immune": ["K"],
            "chain": [{"from": "K", "to": "P"}],
        }
        spec["rules"]["promote_on_entry"] = True
        errors = validate_spec(spec)
        assert any("nothing enters play except by moving" in error for error in errors), errors
        # And the chain itself is refused as stating nothing (Q538): the compiler drops it.
        from chess_variant_sandbox.spec.schema import PLAYED_AS_CAPTURE, strike_changes_anything

        assert any(PLAYED_AS_CAPTURE in error for error in errors), errors
        ids = [piece["id"] for piece in spec["pieces"]]
        assert not strike_changes_anything(spec["rules"]["strike"], "strike", ids)

    def test_a_drop_region_that_stops_short_of_the_zone_is_refused(self) -> None:
        """Crazyhouse keeps the pawn off the far rank, so there the flag is a rule about nothing."""
        spec = _with_flag("rulesets/crazyhouse.json")
        errors = validate_spec(spec)
        assert any(
            "can never fire" in error and "P may not be dropped on rank 8" in error
            for error in errors
        ), errors
        # Let the pawn be dropped anywhere and the same spec is fine.
        del spec["rules"]["drops"]["region"]
        assert validate_spec(spec) == []

    def test_a_chain_that_never_produces_a_promoting_piece_is_refused(
        self, chess_spec: dict[str, Any]
    ) -> None:
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        spec["rules"]["strike"] = {
            "effect": "demote", "position": "stay", "immune": ["K"],
            "chain": [{"from": "Q", "to": "R"}],
        }
        spec["rules"]["promote_on_entry"] = True
        errors = validate_spec(spec)
        assert any(
            "can never fire" in error and "P is not on the strike chain's 'to' side" in error
            for error in errors
        ), errors
        spec["rules"]["strike"]["chain"].append({"from": "N", "to": "P"})
        assert validate_spec(spec) == []

    def test_it_must_be_a_boolean(self, fall: dict[str, Any]) -> None:
        fall["rules"]["promote_on_entry"] = "yes"
        assert any("promote_on_entry must be a boolean" in e for e in validate_spec(fall))

    def test_false_needs_nothing(self, chess_spec: dict[str, Any]) -> None:
        """`false` is the absence of the rule and is not held to the coherence checks."""
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        spec["rules"]["promote_on_entry"] = False
        assert validate_spec(spec) == []

    def test_it_is_a_schema_nine_word(self, fall: dict[str, Any]) -> None:
        fall["schema_version"] = 6  # what hierarchy-fall itself declares
        errors = validate_spec(fall)
        assert len(errors) == 1, errors
        assert "schema 9" in errors[0] and "rules.promote_on_entry" in errors[0]

    def test_both_schema_nine_keys_are_named_in_one_sentence(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """A spec reaching for the whole of schema 9 under an older stamp is told once."""
        spec = copy.deepcopy(chess_spec)
        spec["pieces"].append(
            {"id": "W", "name": "boar", "value": 3,
             "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]}
        )
        spec["rules"]["npcs"] = [
            {"piece": "W", "route": ["c3", "c4", "d4", "d3"], "bounty": 3, "mirror": True}
        ]
        spec["rules"]["drops"] = {"from_captures": True}  # the pawn's way in
        spec["rules"]["promote_on_entry"] = True
        spec["schema_version"] = 9
        assert validate_spec(spec) == []
        spec["schema_version"] = 8
        errors = validate_spec(spec)
        assert len(errors) == 1, errors
        assert "rules.npcs" in errors[0] and "rules.promote_on_entry" in errors[0]


# ------------------------------------------------------------------- what the compiler builds


class TestWhatTheCompilerBuilds:
    def test_a_demoted_knight_on_the_far_rank_becomes_a_queen(self, fall: dict[str, Any]) -> None:
        compiled = compile_spec(fall)
        entry = compiled.entry_promotion
        assert entry is not None
        pawn, knight, queen = (compiled.piece_ids.index(p) for p in ("P", "N", "Q"))
        # The first player's zone is the far rank, the second's the near one.
        assert entry.at(0, pawn, _square("e8")) == queen
        assert entry.at(1, pawn, _square("e1")) == queen
        assert entry.at(0, pawn, _square("e1")) == -1
        assert entry.at(1, pawn, _square("e8")) == -1
        assert entry.at(0, pawn, _square("e7")) == -1
        # A type with no promotion rule never promotes on entry, wherever it lands.
        assert entry.at(0, knight, _square("e8")) == -1
        assert all(entry.at(p, knight, s) == -1 for p in (0, 1) for s in range(64))

    def test_the_first_listed_target_wins(self, fall: dict[str, Any]) -> None:
        """A designer's ordering is the only choice available where nobody is choosing."""
        fall["rules"]["promotion"]["to"] = ["N", "Q", "R", "B"]
        compiled = compile_spec(fall)
        assert compiled.entry_promotion is not None
        pawn, knight = compiled.piece_ids.index("P"), compiled.piece_ids.index("N")
        assert compiled.entry_promotion.at(0, pawn, _square("a8")) == knight

    def test_each_player_takes_their_own_first_target(self, chess_spec: dict[str, Any]) -> None:
        spec = copy.deepcopy(chess_spec)
        spec["schema_version"] = 9
        spec["rules"]["promotion"] = {
            "piece": "P", "rank": "last", "to_by_player": {"0": ["Q", "N"], "1": ["N", "Q"]}
        }
        spec["rules"]["drops"] = {"from_captures": True}
        spec["rules"]["promote_on_entry"] = True
        assert validate_spec(spec) == []
        compiled = compile_spec(spec)
        assert compiled.entry_promotion is not None
        pawn, knight, queen = (compiled.piece_ids.index(p) for p in ("P", "N", "Q"))
        assert compiled.entry_promotion.at(0, pawn, _square("d8")) == queen
        assert compiled.entry_promotion.at(1, pawn, _square("d1")) == knight

    def test_an_optional_rule_over_a_deep_zone_fills_the_whole_zone(
        self, shogi_entry: dict[str, Any]
    ) -> None:
        """`mandatory: false` is about a move's option to decline, and there is no move here."""
        compiled = compile_spec(shogi_entry)
        entry = compiled.entry_promotion
        assert entry is not None
        pawn, tokin = compiled.piece_ids.index("P"), compiled.piece_ids.index("Q")
        assert not compiled.promotions[0].mandatory
        width = compiled.width
        for rank in (6, 7, 8):
            for file in range(width):
                assert entry.at(0, pawn, rank * width + file) == tokin
                assert entry.at(1, pawn, (8 - rank) * width + file) == tokin
        for rank in range(6):
            assert entry.at(0, pawn, rank * width) == -1

    def test_off_compiles_to_nothing(self, chess_spec: dict[str, Any]) -> None:
        assert compile_spec(chess_spec).entry_promotion is None


# ------------------------------------------------------------------------------ the boundary


class TestTheBoundary:
    def test_the_table_crosses_flat_in_player_type_square_order(
        self, fall: dict[str, Any]
    ) -> None:
        compiled = compile_spec(fall)
        tables = canonical_tables(compiled)
        n_types, n_squares = compiled.n_types, compiled.n_squares
        assert len(tables["entry_promotion"]) == 2 * n_types * n_squares
        pawn, queen = compiled.piece_ids.index("P"), compiled.piece_ids.index("Q")
        assert tables["entry_promotion"][(0 * n_types + pawn) * n_squares + _square("e8")] == queen
        assert tables["entry_promotion"][(1 * n_types + pawn) * n_squares + _square("e1")] == queen
        assert "entry_promotion" not in canonical_tables(compile_spec(_without(fall)))

    def test_the_flag_off_still_refuses_it_by_name(
        self, fall: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tenth hold: a core not taught the table would leave the pawn standing."""
        monkeypatch.setattr(bridge, "_CORE_KNOWS_NPCS", False)
        reason = rust_supports(compile_spec(fall))
        assert reason is not None and "rules.promote_on_entry" in reason
        with pytest.raises(NotImplementedError, match="promote_on_entry"):
            Engine(fall)

    def test_the_two_schema_nine_keys_are_lifted_by_one_flag(
        self, fall: dict[str, Any], chess_spec: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Q9b flipped `_CORE_KNOWS_NPCS` once and the whole version opened.

        Both keys, one flag, and the test says so in both directions: the version is a
        thing the core either knows or does not, and half of it opening would be a spec
        that validates, compiles and plays a different game than it asked for.
        """
        wild = copy.deepcopy(chess_spec)
        wild["schema_version"] = 9
        wild["pieces"].append({
            "id": "W", "name": "boar", "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
        })
        wild["rules"]["npcs"] = [{"piece": "W", "route": ["c3", "c4", "d4", "d3"]}]

        assert rust_supports(compile_spec(fall)) is None
        assert rust_supports(compile_spec(wild)) is None

        monkeypatch.setattr(bridge, "_CORE_KNOWS_NPCS", False)
        for spec, key in ((fall, "rules.promote_on_entry"), (wild, "rules.npcs")):
            reason = rust_supports(compile_spec(spec))
            assert reason is not None and key in reason

    def test_the_smoke_gate_plays_the_candidate(self, fall: dict[str, Any]) -> None:
        from chess_variant_sandbox.loop.gates import smoke_test

        result = smoke_test(fall, games=4)
        assert not any("core cannot play" in reason for reason in result.reasons)

    def test_the_fuzzer_walks_it(
        self, fall: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fuzz, "case",
            lambda seed, corpus=fuzz.CORPUS: fuzz.Case(seed, "fall", (), copy.deepcopy(fall)),
        )
        coverage = fuzz.run(range(3), plies=2)
        assert coverage.walked == 3
        assert not coverage.refused
        assert coverage.singles["promote_on_entry"] == 3


# --------------------------------------------------------------------------- the surfaces


class TestTheSurfaces:
    def test_the_model_can_propose_it(self) -> None:
        from chess_variant_sandbox.spec.jsonschema import rule_spec_schema

        rules = rule_spec_schema()["properties"]["rules"]["properties"]
        assert rules["promote_on_entry"]["type"] == "boolean"

    def test_the_prose_says_what_the_piece_becomes(self, fall: dict[str, Any]) -> None:
        from chess_variant_sandbox.spec.describe import describe_spec

        prose = describe_spec(fall)
        assert "promotes at once" in prose and "P to Q" in prose
        assert "left standing there by a strike" in prose
        without = describe_spec(_without(fall))
        assert without != prose and "promotes at once" not in without
        # Where the two players promote into different pieces, both are named.
        fall["rules"]["promotion"] = {
            "piece": "P", "rank": "last", "to_by_player": {"0": ["Q"], "1": ["N", "Q"]}
        }
        del fall["rules"]["en_passant"]
        assert "P to Q for the first player, or N for the second" in describe_spec(fall)

    def test_the_rule_is_charged_as_one_exception(self, fall: dict[str, Any]) -> None:
        from chess_variant_sandbox.spec.complexity import _RULE_COSTS, measure_complexity

        assert _RULE_COSTS["promote_on_entry"] == _RULE_COSTS["en_passant"] == 1.0
        charged = measure_complexity(fall).special_rules
        assert charged - measure_complexity(_without(fall)).special_rules == 1.0
        off = copy.deepcopy(fall)
        off["rules"]["promote_on_entry"] = False
        assert measure_complexity(off).special_rules == charged - 1.0

    def test_with_and_without_are_two_different_games(self, fall: dict[str, Any]) -> None:
        from chess_variant_sandbox.spec.distance import distance, features

        without = _without(fall)
        assert features(fall)["promotes_on_entry"] == 1.0
        assert features(without)["promotes_on_entry"] == 0.0
        assert distance(fall, without) > 0.0

    def test_the_operator_sets_it_and_takes_it_back(self) -> None:
        """Both halves, valid both ways, and the second half undoes the first."""
        from chess_variant_sandbox.loop.operators import OPERATORS, _promote_what_arrives

        assert "promote-what-arrives" in {operator.name for operator in OPERATORS}
        spec = load_spec(ROOT / "archive/hierarchy-fall.json")
        before, declared = spec_hash(spec), spec["schema_version"]
        summary = _promote_what_arrives(spec, random.Random(1))
        assert summary and "promotes at once" in summary
        assert spec["rules"]["promote_on_entry"] is True
        assert spec["schema_version"] == 9
        assert validate_spec(spec) == [], validate_spec(spec)
        assert _promote_what_arrives(spec, random.Random(1))
        assert "promote_on_entry" not in spec["rules"]
        assert spec["schema_version"] == declared
        assert spec_hash(spec) == before, "promote-what-arrives does not undo itself"

    def test_the_operator_refuses_where_nothing_could_enter(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """No way in at all, and a way in that stops short: both are refusals here.

        The second used to be emitted and left to the validator -- this test asserted
        `validate_spec(crazyhouse)` was non-empty, *"the validator refuses this child, and
        mutate draws again"*. Drawing again is the right answer for an operator that
        draws. **This one reads no random number**, so its answer on a given spec is the
        same answer every time it is picked: on `crazyhouse` and the three
        `hierarchy-rise` games it returned the identical rejected child at every seed,
        which is one twenty-eighth of every mutation budget spent forever on a candidate
        that can never be played, and it reads from outside exactly like an operator that
        had nothing to do (Q85).

        It asks the validator on a copy rather than paraphrasing
        `_check_entry_promotion_coherence`, so the coherence rule is still read in one
        place and the cost is the validation `mutate` was about to run anyway.
        """
        from chess_variant_sandbox.loop.operators import _promote_what_arrives

        assert _promote_what_arrives(copy.deepcopy(chess_spec), random.Random(1)) is None
        crazyhouse = load_spec(ROOT / "rulesets/crazyhouse.json")
        before = spec_hash(crazyhouse)
        assert _promote_what_arrives(crazyhouse, random.Random(1)) is None
        assert spec_hash(crazyhouse) == before, "a refusal must leave the spec as it was"

    def test_the_proposer_is_told_the_point_of_it(self) -> None:
        from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE

        assert "`rules.promote_on_entry`" in SCHEMA_PREAMBLE
        # And the paragraph says the point: what it does, and what it needs.
        paragraph = next(
            line for line in SCHEMA_PREAMBLE.splitlines()
            if line.startswith("`rules.promote_on_entry`")
        )
        assert "promotes at once" in paragraph
        assert "needs `promotion`" in paragraph and "`drops`" in paragraph

    def test_fairy_stockfish_refuses_it_by_name_and_false_refuses_nothing(
        self, dobutsu_entry: dict[str, Any]
    ) -> None:
        from chess_variant_sandbox.fairy.variant import Untranslatable, translate

        with pytest.raises(Untranslatable) as caught:
            translate(dobutsu_entry)
        assert any("rules.promote_on_entry" in reason for reason in caught.value.reasons)
        off = copy.deepcopy(dobutsu_entry)
        off["rules"]["promote_on_entry"] = False
        try:
            translate(off)
        except Untranslatable as refusal:
            assert not any("promote_on_entry" in reason for reason in refusal.reasons)


# ------------------------------------------------------------------------------- the oracle


class TestTheFuzzOracle:
    """Run against a compiled spec and a hand-built state, since no engine can play one."""

    def _engine(self, spec: dict[str, Any]) -> Any:
        return SimpleNamespace(spec=compile_spec(spec))

    def _state(self, engine: Any, **changes: int) -> Any:
        board = bytearray(engine.spec.initial_board)
        unmoved = engine.spec.initial_unmoved
        for name, value in changes.items():
            square = _square(name)
            board[square] = value
            unmoved &= ~(1 << square)  # arriving on a square clears its bit
        return SimpleNamespace(board=bytes(board), unmoved=unmoved)

    def test_the_opening_position_obeys_it(self, fall: dict[str, Any]) -> None:
        engine = self._engine(fall)
        fuzz.a_piece_that_entered_its_promotion_zone_has_promoted(
            engine, SimpleNamespace(board=engine.spec.initial_board, unmoved=engine.spec.initial_unmoved)
        )

    def test_a_pawn_that_arrived_on_the_far_rank_is_caught(self, fall: dict[str, Any]) -> None:
        engine = self._engine(fall)
        pawn = engine.spec.encode_piece(0, engine.spec.piece_ids.index("P"))
        state = self._state(engine, e8=pawn)
        with pytest.raises(fuzz.OracleFailure, match="unpromoted"):
            fuzz.a_piece_that_entered_its_promotion_zone_has_promoted(engine, state)
        # The same pawn one rank short of the zone is nobody's business.
        fuzz.a_piece_that_entered_its_promotion_zone_has_promoted(
            engine, self._state(engine, e7=pawn)
        )

    def test_a_pawn_the_setup_put_there_is_exempt(self, fall: dict[str, Any]) -> None:
        """Two readings of "it started here": the unmoved bit, and the opening board."""
        engine = self._engine(fall)
        pawn = engine.spec.encode_piece(0, engine.spec.piece_ids.index("P"))
        state = self._state(engine, e8=pawn)
        state.unmoved |= 1 << _square("e8")
        fuzz.a_piece_that_entered_its_promotion_zone_has_promoted(engine, state)
        # A pawn that has turned in place has lost its bit but not its alibi.
        board = bytearray(engine.spec.initial_board)
        board[_square("e8")] = pawn
        placed = SimpleNamespace(
            spec=dataclasses.replace(engine.spec, initial_board=bytes(board))
        )
        fuzz.a_piece_that_entered_its_promotion_zone_has_promoted(
            placed, self._state(placed, e8=pawn)
        )

    def test_an_optional_rule_is_not_asked(self, shogi_entry: dict[str, Any]) -> None:
        """A pawn that moved into a shogi zone and declined stands there by right."""
        engine = self._engine(shogi_entry)
        pawn = engine.spec.encode_piece(0, engine.spec.piece_ids.index("P"))
        board = bytearray(engine.spec.initial_board)
        square = 8 * engine.spec.width + 4  # the far rank, a file the array leaves empty
        board[square] = pawn
        fuzz.a_piece_that_entered_its_promotion_zone_has_promoted(
            engine, SimpleNamespace(board=bytes(board), unmoved=0)
        )

    def test_it_is_free_where_the_flag_is_off(self, chess_spec: dict[str, Any]) -> None:
        engine = SimpleNamespace(spec=compile_spec(chess_spec))
        fuzz.a_piece_that_entered_its_promotion_zone_has_promoted(
            engine, SimpleNamespace(board=engine.spec.initial_board, unmoved=0)
        )


# ------------------------------------------------ Q922: promote_on_entry, branch by branch


def _entry_errors(rules: dict[str, Any], board: Any = None, **spec: Any) -> list[str]:
    """`_check_entry_promotion_coherence` over a three-wide, four-high board of P, Q and R."""
    from chess_variant_sandbox.spec.schema import _check_entry_promotion_coherence

    whole = {
        "board": {"width": 3, "height": 4} if board is None else board,
        "pieces": [{"id": "P"}, {"id": "Q"}, {"id": "R"}],
        **spec,
    }
    errors: list[str] = []
    _check_entry_promotion_coherence(whole, {"promote_on_entry": True, **rules}, errors)
    return errors


PROMOTE_Q = {"piece": "Q", "rank": "last", "to": ["R"]}


class TestWhichPromotionsTheFlagReads:
    def test_a_promotion_without_a_piece_is_no_promotion(self) -> None:
        (error,) = _entry_errors({"promotion": [{"to": ["Q"]}, "junk", None, {"piece": 3}]})
        assert error.startswith("rules.promote_on_entry needs rules.promotion")

    def test_a_promotion_with_a_piece_is_one(self) -> None:
        (error,) = _entry_errors({"promotion": ["junk", PROMOTE_Q]})
        assert error.startswith("rules.promote_on_entry: nothing enters play except by moving")


class TestAContactLinkFeedsTheFlagUnlessItsTargetIsImmune:
    @staticmethod
    def _errors(links: list[Any], immune: list[str]) -> list[str]:
        rules = {"promotion": [PROMOTE_Q], "contacts": {"links": links, "immune": immune}}
        return _entry_errors(rules)

    def test_a_target_that_becomes_the_promoting_piece(self) -> None:
        link = {"mover": "R", "onto": "P", "onto_to": "Q"}
        assert self._errors(["junk", link], []) == []

    def test_an_immune_target_never_becomes_anything(self) -> None:
        link = {"mover": "R", "onto": "P", "onto_to": "Q"}
        (error,) = self._errors([link], ["P"])
        assert "nothing enters play except by moving" in error

    def test_a_mover_that_becomes_the_promoting_piece(self) -> None:
        link = {"mover": "R", "onto": "P", "mover_to": "Q"}
        assert self._errors([link], ["P"]) == []

    @pytest.mark.parametrize(
        "link", [{"mover": "R", "onto": "P"}, {"mover": "R", "onto": "P", "onto_to": 3}], ids=str
    )
    def test_a_link_that_turns_nothing_into_anything(self, link: dict[str, Any]) -> None:
        (error,) = self._errors([link], [])
        assert "nothing enters play except by moving" in error


class TestABoardTheFlagCannotMeasureIsLeftToItsOwnRefusal:
    HAND = {"promotion": [PROMOTE_Q], "drops": {"region": {"Q": ["a1"]}}}

    @pytest.mark.parametrize(
        "board",
        [{"width": None, "height": 4}, {"width": 3, "height": None},
         {"width": "3", "height": 4}, {"width": 3, "height": 4.0}, "board", None],
        ids=str,
    )
    def test_no_verdict(self, board: Any) -> None:
        spec = {"board": board}
        from chess_variant_sandbox.spec.schema import _check_entry_promotion_coherence

        errors: list[str] = []
        _check_entry_promotion_coherence(
            {**spec, "pieces": [{"id": "P"}, {"id": "Q"}]}, {"promote_on_entry": True, **self.HAND},
            errors,
        )
        assert errors == []

    def test_the_same_hand_on_a_real_board_is_refused(self) -> None:
        (error,) = _entry_errors(self.HAND)
        assert "Q may not be dropped on rank 4" in error


class TestWhereAPromotingPieceMayBeDropped:
    """Width three, height four; the zone is rank 4, squares 9, 10 and 11."""

    @pytest.mark.parametrize(
        ("squares", "refused"),
        [
            (["a4"], False), (["b4"], False), (["c4"], False),
            (["a3"], True), (["c3"], True), (["a1"], True), (["c1"], True),
            (["a1", "c4"], False),
            (["a1", 7, None, "z9"], True),
            ("c4", True),
            ([], True),
        ],
        ids=str,
    )
    def test_the_region_against_the_zone(self, squares: Any, refused: bool) -> None:
        rules = {"promotion": [PROMOTE_Q], "drops": {"region": {"Q": squares}}}
        assert bool(_entry_errors(rules)) is refused

    def test_no_region_entry_means_anywhere(self) -> None:
        rules = {"promotion": [PROMOTE_Q], "drops": {"region": {"R": ["a1"]}}}
        assert _entry_errors(rules) == []

    def test_the_orientation_is_read_when_there_is_a_setup_and_not_needed_when_there_is_none(
        self,
    ) -> None:
        rules = {"promotion": [PROMOTE_Q], "drops": {"region": {"Q": ["a1"]}}}
        for extra in ({}, {"setup": {}}, {"setup": {"orientation": "rotate"}}, {"setup": "x"}):
            (error,) = _entry_errors(rules, **extra)
            assert "Q may not be dropped on rank 4" in error, extra

    def test_every_promoting_piece_is_judged_not_only_the_first(self) -> None:
        """The first never reaches a hand; the second is kept out by its region."""
        rules = {
            "promotion": [PROMOTE_Q, {"piece": "R", "rank": "last", "to": ["P"]}],
            "drops": {"from_captures": False, "stock": {"R": 1}, "region": {"R": ["a1"]}},
        }
        (error,) = _entry_errors(rules)
        assert "Q never reaches a hand, since rules.drops.from_captures is false" in error
        assert "; R may not be dropped on rank 4" in error

    def test_captures_fill_a_hand_unless_it_says_they_do_not(self) -> None:
        # No `from_captures` key: a captured Q can be dropped, so the region is the reason.
        rules = {"promotion": [PROMOTE_Q], "drops": {"region": {"Q": ["a1"]}},
                 "win": "extinction"}
        spec_extra = {"setup": {"placements": [{"piece": "Q", "squares": ["a1"]}]}}
        (error,) = _entry_errors(rules, **spec_extra)
        assert "Q may not be dropped on rank 4" in error
        assert "never reaches a hand" not in error


def test_a_piece_with_no_id_is_passed_over_not_a_key_error() -> None:
    """F-spec-last (Q946): `_never_in_a_hand` reads a spec the validator is still collecting
    errors on, so a piece with no `id` is skipped, not looked up."""
    from chess_variant_sandbox.spec.schema import _never_in_a_hand

    spec = {"pieces": [{"id": "P"}, {"moves": []}, {"id": "Q", "promoted_from": "P"}]}
    reason = _never_in_a_hand(spec, {}, {"from_captures": False}, "P")
    assert reason is not None and "from_captures is false" in reason
    # and the walk past it, with captures on, answers rather than raising
    _never_in_a_hand(spec, {}, {}, "P")
