"""Schema 13: the `sweep` move kind (Q21).

A sweeper moves along its lines to any square on them, jumping what stands between, and
every piece it passes over is removed, its own included. A piece it may not capture
blocks. A sweep that could land beyond the enemy royal is check (the user's decision A,
2026-09-14). Specified in `docs/game-ideas.md`, *Q21 specified*.

The inertness proof is `test_schema_nine.py::TestTheBumpMovedNothing`, which pins the
mechanics hash and tables fingerprint of every committed and archived ruleset. This file
holds the feature itself: gated, refused where it means nothing, played by the core, and
read by every surface `new-move-primitive` names.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.minimax import MinimaxAgent
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_NORMAL,
    MOVE_SWEEP,
    MOVE_TYPE_NAMES,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.fairy import supports
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import _SCALES, distance, features
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION
from chess_variant_sandbox.spec.transform import (
    set_contacts,
    set_explosion,
    set_npc,
    set_strike,
    set_wrap,
)
from tests.feature_inventory import _schema_thirteen_uses

ROOT = Path(__file__).resolve().parents[1]

# Pre-registered in `docs/game-ideas.md` before the spec existed: chess's 20 plus five
# sweeps per bishop. The deeper rungs are this build's own counts, pinned as a regression
# and not verified elsewhere. Perft(2) is short of 30 x 30 partly because `Bf1xb5`
# (sweeping e2) is already mate: the bishop's line to e8 runs through a pawn it may take.
SHOWCASE_PERFT = {1: 30, 2: 867, 3: 27_166}


@pytest.fixture(scope="module")
def chess() -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sweeping() -> dict[str, Any]:
    return json.loads((ROOT / "archive" / "chess-sweeping-bishops.json").read_text(encoding="utf-8"))


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _name(square: int, width: int = 8) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


def _bishop(spec: dict[str, Any]) -> dict[str, Any]:
    return next(piece for piece in spec["pieces"] if piece["id"] == "B")


def _position(sweeping: dict[str, Any], placements: list[dict[str, Any]], **changes: Any) -> dict[str, Any]:
    """The showcase's rules on a hand-built position, white to move."""
    spec = copy.deepcopy(sweeping)
    spec["rules"].pop("castling")
    spec["rules"].pop("en_passant")
    if not any(placement["piece"] == "P" for placement in placements):
        spec["rules"].pop("promotion")
    spec["setup"] = {"mirror": False, "placements": placements}
    for key, value in changes.items():
        _bishop(spec)["moves"][0][key] = value
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _from(engine: Engine, state: Any, origin: str) -> dict[str, int]:
    return {
        _name(move_to(m)): move_type(m)
        for m in engine.legal_moves(state)
        if move_from(m) == _square(origin)
    }


class TestTheVersionIsBumped:
    def test_the_schema_version_is_thirteen(self) -> None:
        assert SCHEMA_VERSION >= 13

    def test_the_collector_names_the_one_feature(
        self, chess: dict[str, Any], sweeping: dict[str, Any]
    ) -> None:
        assert _schema_thirteen_uses(chess) == set()
        assert _schema_thirteen_uses(sweeping) == {"the 'sweep' move kind"}

class TestTheValidator:
    def test_the_showcase_validates(self, sweeping: dict[str, Any]) -> None:
        assert validate_spec(sweeping) == []

    def test_a_range_and_a_floor_are_allowed(self, sweeping: dict[str, Any]) -> None:
        spec = copy.deepcopy(sweeping)
        _bishop(spec)["moves"][0].update(range=3, min_range=2)
        assert validate_spec(spec) == []

    def test_a_screen_is_refused(self, sweeping: dict[str, Any]) -> None:
        spec = copy.deepcopy(sweeping)
        _bishop(spec)["moves"][0]["screen_range"] = 1
        assert any("only applies to a hop" in e for e in validate_spec(spec))

    @pytest.mark.parametrize(
        "key, add, rid",
        [
            ("strike", lambda spec: set_strike(spec, effect="demote", position="stay"), "W1"),
            ("contacts", lambda spec: set_contacts(spec, effect="promote"), "W2"),
            ("explode_on_capture", set_explosion, "W3"),
            ("drops", lambda spec: spec["rules"].__setitem__("drops", {"from_captures": True}),
             "W4"),
            ("npcs", lambda spec: set_npc(spec, "N", ["c3", "c4", "d4", "d3"]), "W5"),
            ("wrap", lambda spec: set_wrap(spec, "files"), "W6"),
        ],
    )
    def test_a_rule_that_cannot_share_a_game_with_a_sweep_is_refused_by_name(
        self, sweeping: dict[str, Any], key: str, add: Any, rid: str
    ) -> None:
        spec = copy.deepcopy(sweeping)
        spec["rules"].pop("en_passant")
        add(spec)
        assert spec["rules"].get(key), f"the helper did not write rules.{key}"
        errors = validate_spec(spec)
        assert rid in refused_ids(errors), errors

    def test_a_promoting_piece_may_not_sweep(self, sweeping: dict[str, Any]) -> None:
        """So no sweeper is ever the en passant piece, and no sweep opens a window."""
        spec = copy.deepcopy(sweeping)
        spec["pieces"][0]["moves"].append({"kind": "sweep", "vectors": [[0, 1]]})
        errors = validate_spec(spec)
        assert any("rules.promotion" in e and "sweep" in e for e in errors), errors

    def test_promoting_into_a_sweeper_is_fine(self, sweeping: dict[str, Any]) -> None:
        assert "B" in sweeping["rules"]["promotion"]["to"]
        assert validate_spec(sweeping) == []

    def test_a_move_only_sweeper_still_captures(self, sweeping: dict[str, Any]) -> None:
        """The path is a capture site the mode does not govern, so `must_capture` is not
        a rule about nothing."""
        spec = copy.deepcopy(sweeping)
        for piece in spec["pieces"]:
            for move in piece["moves"]:
                move["mode"] = "move"
        spec["rules"].pop("en_passant")
        spec["rules"]["must_capture"] = True
        assert validate_spec(spec) == []


class TestTheCompiledTables:
    def test_no_other_ruleset_sends_a_sweep_table(self, chess: dict[str, Any]) -> None:
        compiled = compile_spec(chess)
        tables = build_tables(compiled)
        keys = (
            "sweep_index", "sweep_refs", "sweep_squares", "sweep_modes",
            "attack_sweep", "attack_path_modes",
        )
        assert all(tables[key] == [] for key in keys)
        assert tables["has_sweep"] is False
        assert not set(keys) & set(canonical_tables(compiled))

    def test_the_bishop_reads_the_fourth_table(self, sweeping: dict[str, Any]) -> None:
        compiled = compile_spec(sweeping)
        bishop = compiled.piece_ids.index("B")
        c1 = _square("c1")
        assert compiled.slides[0][bishop][c1] == ()
        rays = compiled.sweeps[0][bishop][c1]
        assert sorted(tuple(_name(s) for s in ray) for ray, _, _ in rays) == [
            ("b2", "a3"), ("d2", "e3", "f4", "g5", "h6"),
        ]
        assert rust_supports(compiled) is None

    def test_the_attack_table_carries_the_sweep_flavour(self, sweeping: dict[str, Any]) -> None:
        tables = build_tables(compile_spec(sweeping))
        assert len(tables["attack_sweep"]) == len(tables["attack_srcs"]) // 8
        assert any(tables["attack_sweep"])
        assert len(tables["attack_path_modes"]) == len(tables["attack_paths"])


class TestTheCorePlaysIt:
    @pytest.mark.parametrize("depth", sorted(SHOWCASE_PERFT))
    def test_perft_matches_the_pre_registered_count(
        self, sweeping: dict[str, Any], depth: int
    ) -> None:
        engine = Engine(sweeping)
        assert engine.perft(engine.initial_state(), depth) == SHOWCASE_PERFT[depth]

    def test_each_bishop_sweeps_through_its_own_pawns(self, sweeping: dict[str, Any]) -> None:
        engine = Engine(sweeping)
        state = engine.initial_state()
        assert _from(engine, state, "c1") == {
            name: MOVE_SWEEP for name in ("a3", "e3", "f4", "g5", "h6")
        }
        assert _from(engine, state, "f1") == {
            name: MOVE_SWEEP for name in ("d3", "c4", "b5", "a6", "h3")
        }

    def test_a_sweep_removes_its_path_and_keeps_the_hash(self, sweeping: dict[str, Any]) -> None:
        engine = Engine(sweeping)
        state = engine.initial_state()
        (move,) = [
            m for m in engine.legal_moves(state)
            if (move_from(m), move_to(m)) == (_square("c1"), _square("f4"))
        ]
        assert engine.describe_move(state, move) == "Bc1xf4 (sweeps d2)"
        after = engine.apply(state, move)
        assert after.board[_square("d2")] == 0
        assert after.board[_square("e3")] == 0
        assert after.board[_square("c1")] == 0
        assert after.board[_square("f4")] == state.board[_square("c1")]
        assert after.key == engine._compute_hash(after)

    def test_an_empty_path_is_an_ordinary_move(self, sweeping: dict[str, Any]) -> None:
        engine = Engine(
            _position(
                sweeping,
                [
                    {"piece": "K", "player": 0, "squares": ["h1"]},
                    {"piece": "B", "player": 0, "squares": ["c1"]},
                    {"piece": "P", "player": 0, "squares": ["e3"]},
                    {"piece": "K", "player": 1, "squares": ["a8"]},
                ],
            )
        )
        moves = _from(engine, engine.initial_state(), "c1")
        assert moves == {
            "b2": MOVE_NORMAL, "a3": MOVE_NORMAL, "d2": MOVE_NORMAL,
            "f4": MOVE_SWEEP, "g5": MOVE_SWEEP, "h6": MOVE_SWEEP,
        }

    def test_a_piece_the_sweeper_may_not_take_blocks(self, sweeping: dict[str, Any]) -> None:
        placements = [
            {"piece": "K", "player": 0, "squares": ["h1"]},
            {"piece": "B", "player": 0, "squares": ["c1"]},
            {"piece": "N", "player": 1, "squares": ["e3"]},
            {"piece": "K", "player": 1, "squares": ["a8"]},
        ]
        engine = Engine(_position(sweeping, placements, mode="both"))
        assert "g5" in _from(engine, engine.initial_state(), "c1")
        spec = _position(sweeping, placements)
        _bishop(spec)["captures"] = ["P", "B", "R", "Q", "K"]
        blocked = Engine(spec)
        assert sorted(_from(blocked, blocked.initial_state(), "c1")) == ["a3", "b2", "d2"]

    def test_a_sweep_through_the_movers_own_royal_is_not_legal(
        self, sweeping: dict[str, Any]
    ) -> None:
        engine = Engine(
            _position(
                sweeping,
                [
                    {"piece": "K", "player": 0, "squares": ["c3"]},
                    {"piece": "B", "player": 0, "squares": ["a1"]},
                    {"piece": "K", "player": 1, "squares": ["a8"]},
                ],
            )
        )
        assert sorted(_from(engine, engine.initial_state(), "a1")) == ["b2"]

    def test_must_capture_counts_a_sweep_over_its_own_piece(
        self, sweeping: dict[str, Any]
    ) -> None:
        spec = copy.deepcopy(sweeping)
        spec["rules"]["must_capture"] = True
        engine = Engine(spec)
        moves = engine.legal_moves(engine.initial_state())
        assert len(moves) == 10 and all(move_type(m) == MOVE_SWEEP for m in moves)

    def test_make_and_unmake_agree_with_a_walk_that_never_unmakes(
        self, sweeping: dict[str, Any]
    ) -> None:
        """Perft plays every move with `make` and takes it back with `unmake`. `apply`
        builds each position afresh and never unmakes. A path square `unmake` failed to
        restore would show up as a different count one ply later."""
        engine = Engine(
            _position(
                sweeping,
                [
                    {"piece": "K", "player": 0, "squares": ["e1"]},
                    {"piece": "B", "player": 0, "squares": ["c1", "f1"]},
                    {"piece": "P", "player": 0, "squares": ["d2", "e2", "g2"]},
                    {"piece": "N", "player": 0, "squares": ["c3"]},
                    {"piece": "K", "player": 1, "squares": ["e8"]},
                    {"piece": "B", "player": 1, "squares": ["c8", "f8"]},
                    {"piece": "P", "player": 1, "squares": ["d7", "e7", "b7"]},
                    {"piece": "N", "player": 1, "squares": ["f6"]},
                ],
            )
        )
        state = engine.initial_state()
        walked = 0
        swept = 0
        for first in engine.legal_moves(state):
            swept += move_type(first) == MOVE_SWEEP
            one = engine.apply(state, first)
            for second in engine.legal_moves(one):
                walked += len(engine.legal_moves(engine.apply(one, second)))
        assert swept > 3
        assert engine.perft(state, 3) == walked

    def test_a_short_game_keeps_its_incremental_hash(self, sweeping: dict[str, Any]) -> None:
        """Several short games, since a sweep can end one within two plies."""
        engine = Engine(sweeping)
        swept = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(80):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                sweeps = [m for m in moves if move_type(m) == MOVE_SWEEP]
                move = rng.choice(sweeps) if sweeps and rng.random() < 0.5 else rng.choice(moves)
                swept += move_type(move) == MOVE_SWEEP
                state = engine.apply(state, move)
                assert state.key == engine._compute_hash(state), "hash drifted"
        assert swept > 5


class TestCheckUnderA:
    """Decision A: a sweep over the enemy royal is check."""

    @staticmethod
    def _defence(sweeping: dict[str, Any], beyond: list[str], **changes: Any) -> tuple[Engine, Any]:
        """White's king on e5 on the long diagonal of a black sweeper on h8, white to move."""
        placements = [
            {"piece": "K", "player": 0, "squares": ["e5"]},
            {"piece": "P", "player": 0, "squares": ["h2"]},
            {"piece": "K", "player": 1, "squares": ["a8"]},
            {"piece": "B", "player": 1, "squares": ["h8"]},
        ]
        if beyond:
            placements.append({"piece": "N", "player": 1, "squares": beyond})
        engine = Engine(_position(sweeping, placements, **changes))
        return engine, engine.initial_state()

    def test_a_landing_beyond_the_royal_is_check_and_must_be_answered(
        self, sweeping: dict[str, Any]
    ) -> None:
        """A move-only sweeper cannot land on the king, and can land past it."""
        engine, state = self._defence(sweeping, [], mode="move")
        assert engine.in_check(state, 0)
        assert not any(move_from(m) == _square("h2") for m in engine.legal_moves(state))

    def test_no_landing_beyond_is_no_check(self, sweeping: dict[str, Any]) -> None:
        engine, state = self._defence(sweeping, ["d4", "c3", "b2", "a1"], mode="move")
        assert not engine.in_check(state, 0)
        assert any(move_from(m) == _square("h2") for m in engine.legal_moves(state))

    def test_a_piece_between_does_not_shield_the_royal(self, sweeping: dict[str, Any]) -> None:
        engine, state = self._defence(sweeping, ["g7", "f6"])
        assert engine.in_check(state, 0)

    def test_a_piece_the_sweeper_may_not_take_does_shield_it(
        self, sweeping: dict[str, Any]
    ) -> None:
        engine, state = self._defence(sweeping, ["f6"])
        spec = json.loads(engine.spec.source)
        _bishop(spec)["captures"] = ["P", "B", "R", "Q", "K"]
        shielded = Engine(spec)
        assert not shielded.in_check(shielded.initial_state(), 0)


class TestTheOtherSurfaces:
    def test_the_move_kind_has_a_name(self) -> None:
        assert MOVE_TYPE_NAMES[MOVE_SWEEP] == "sweep"

    def test_the_description_says_what_is_removed(self, sweeping: dict[str, Any]) -> None:
        said = describe_spec(sweeping)
        assert "sweeping away every piece it passes over, its own included" in said

    def test_it_has_a_distance_feature_of_its_own(
        self, chess: dict[str, Any], sweeping: dict[str, Any]
    ) -> None:
        assert "sweep_share" in _SCALES
        assert features(chess)["sweep_share"] == 0.0
        assert features(sweeping)["sweep_share"] > 0.0
        assert distance(sweeping, chess) > 0.0

    def test_fairy_refuses_it_by_name(self, sweeping: dict[str, Any]) -> None:
        assert not supports(sweeping)
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(sweeping))
        assert any("Betza has no such piece" in reason for reason in refusal.value.reasons)

    def test_the_json_schema_and_the_preamble_offer_it(self) -> None:
        def enums(node: Any) -> list[list[Any]]:
            if isinstance(node, dict):
                found = [node["enum"]] if isinstance(node.get("enum"), list) else []
                return found + [e for value in node.values() for e in enums(value)]
            if isinstance(node, list):
                return [e for value in node for e in enums(value)]
            return []

        kinds = [enum for enum in enums(rule_spec_schema()) if "lame" in enum]
        assert kinds and all("sweep" in enum for enum in kinds)
        assert "**sweep**" in SCHEMA_PREAMBLE

    def test_the_metrics_read_the_fourth_table(
        self, chess: dict[str, Any], sweeping: dict[str, Any]
    ) -> None:
        """Q236. `_displacements` enumerated three move tables by name, so a sweeping
        piece read as one that cannot move at all: the showcase's bishop measured an
        empty reach, `threat_asymmetry` 0.300 against the true 0.200, and
        `traversal_time` 3.630 against 3.200 because the piece was skipped rather than
        counted. A sweep changes what happens on the path, not where the piece can go,
        so the showcase's two spec-only components are chess's.
        """
        from chess_variant_sandbox.engine.compile import MODE_CAPTURE, MODE_MOVE
        from chess_variant_sandbox.metrics.components import _displacements, spec_facts

        compiled = compile_spec(sweeping)
        bishop = compiled.piece_ids.index("B")
        reach = _displacements(compiled, MODE_MOVE | MODE_CAPTURE)[bishop]
        assert len(reach) == 28
        assert _displacements(compiled, MODE_CAPTURE)[bishop] == reach

        swept, plain = spec_facts(sweeping), spec_facts(chess)
        assert swept.threat_asymmetry == pytest.approx(plain.threat_asymmetry)
        assert swept.traversal_time == pytest.approx(plain.traversal_time)

    def test_the_search_oracle_sums_the_path(self, sweeping: dict[str, Any]) -> None:
        engine = Engine(sweeping)
        state = engine.initial_state()
        agent = MinimaxAgent(engine)
        (move,) = [
            m for m in engine.legal_moves(state)
            if (move_from(m), move_to(m)) == (_square("c1"), _square("e3"))
        ]
        pawn = engine.spec.piece_ids.index("P")
        assert agent._is_loud(state, move)
        assert agent._capture_gain(state.board, move) == agent.evaluator.values[pawn]
