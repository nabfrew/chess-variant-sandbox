"""SR3a (`docs/plan-schema-next.md` §1 rank 4): a goal for a list of pieces.

`rules.reach.piece` takes a list as well as one id, and any listed piece arriving (or
leaving) decides the game, so a side chooses its runner. The core's `ReachRule` reads a type
mask; one id is one bit, and crosses the boundary as it always did. The game here is king of
the hill, only because it is a committed `reach` game: nothing below reads chess.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.transform import drop_piece
from tests.feature_inventory import _schema_fifty_seven_uses

ROOT = Path(__file__).resolve().parent.parent
GOALS = ("rulesets/kingofthehill.json", "rulesets/racingkings.json",
         "rulesets/intransitive.json", "rulesets/dobutsu.json")


def _hill(piece: Any = None) -> dict[str, Any]:
    spec = load_spec(ROOT / "rulesets" / "kingofthehill.json")
    if piece is not None:
        spec["rules"]["reach"]["piece"] = piece
    return spec


def _go(engine: Engine, state: Any, text: str) -> Any:
    """Play the one legal move whose text starts with ``text``."""
    (move,) = [
        m for m in engine.legal_moves(state) if engine.describe_move(state, m).startswith(text)
    ]
    return engine.apply(state, move)


def _play(spec: dict[str, Any], moves: list[str]) -> Any:
    engine = Engine(spec)
    state = engine.initial_state()
    for text in moves:
        assert engine.terminal(state, engine.legal_moves(state)) is None
        state = _go(engine, state, text)
    return engine.terminal(state, engine.legal_moves(state))


# White's knight reaches d4 in two moves; its bishop reaches e4 in three; a pawn reaches e4 in
# one. Black shuffles a knight that never touches the centre.
KNIGHT_RUN = ["Ng1-f3", "Nb8-a6", "Nf3-d4"]
BISHOP_RUN = ["Pg2-g3", "Nb8-a6", "Bf1-g2", "Na6-b8", "Bg2-e4"]
PAWN_RUN = ["Pe2-e4"]


class TestAListOfOneIsTheOnePiece:
    @pytest.mark.parametrize("path", GOALS)
    def test_a_list_of_one_compiles_to_todays_tables_byte_for_byte(self, path: str) -> None:
        spec = load_spec(ROOT / path)
        piece = spec["rules"]["reach"].get("piece")
        if piece is None:
            pytest.skip("a goal for any piece names nobody to list")
        listed = copy.deepcopy(spec)
        listed["rules"]["reach"]["piece"] = [piece]
        assert validate_spec(listed) == []
        assert build_tables(compile_spec(listed)) == build_tables(compile_spec(spec))
        assert spec_hash(listed) == spec_hash(spec)
        assert normal_form(listed) == normal_form(spec)

    def test_the_runners_are_a_set(self) -> None:
        assert spec_hash(_hill(["N", "K"])) == spec_hash(_hill(["K", "N"]))
        assert build_tables(compile_spec(_hill(["N", "K"]))) == build_tables(
            compile_spec(_hill(["K", "N"]))
        )

    def test_two_runners_are_another_game(self) -> None:
        two = _hill(["K", "N"])
        assert spec_hash(two) != spec_hash(_hill())
        assert distance(_hill(), two) > 0.0
        assert _schema_fifty_seven_uses(two) == {"rules.reach.piece.list"}
        assert _schema_fifty_seven_uses(_hill(["K"])) == set()
        tables = build_tables(compile_spec(two))
        assert tables["reach_piece_type"] == -1
        ids = compile_spec(two).piece_ids
        assert tables["reach_piece_types"] == sorted([ids.index("K"), ids.index("N")])


class TestEitherRunnerWins:
    def test_the_knight_wins_a_two_runner_race(self) -> None:
        outcome = _play(_hill(["K", "N"]), KNIGHT_RUN)
        assert outcome is not None and outcome.reason == "reached" and outcome.winner == 0

    def test_the_bishop_wins_a_race_it_shares_with_the_knight(self) -> None:
        outcome = _play(_hill(["B", "N"]), BISHOP_RUN)
        assert outcome is not None and outcome.reason == "reached" and outcome.winner == 0
        outcome = _play(_hill(["B", "N"]), KNIGHT_RUN)
        assert outcome is not None and outcome.reason == "reached" and outcome.winner == 0

    def test_a_piece_off_the_list_decides_nothing(self) -> None:
        assert _play(_hill(["B", "N"]), PAWN_RUN) is None
        assert _play(_hill(), KNIGHT_RUN) is None, "the king alone: a knight arriving is play"

    def test_the_second_player_wins_with_either_runner_too(self) -> None:
        outcome = _play(_hill(["K", "N"]), ["Pa2-a3", "Ng8-f6", "Pa3-a4", "Nf6-e4"])
        assert outcome is not None and outcome.reason == "reached" and outcome.winner == 1

    def test_leaving_reads_the_list(self) -> None:
        spec = _hill(["N", "B"])
        spec["rules"]["reach"]["on"] = "leave"
        assert validate_spec(spec) == []
        arrived = _play(spec, KNIGHT_RUN)
        assert arrived is None, "standing on the goal decides nothing under leave"
        outcome = _play(spec, [*KNIGHT_RUN, "Na6-b8", "Nd4-f3"])
        assert outcome is not None and outcome.reason == "reached" and outcome.winner == 0


class TestTheSpelling:
    def test_an_empty_list_is_refused(self) -> None:
        assert any("empty list" in e for e in validate_spec(_hill([])))

    def test_an_undeclared_runner_is_refused_by_name(self) -> None:
        assert any("'Z'" in e for e in validate_spec(_hill(["K", "Z"])))

    def test_a_runner_listed_twice_is_refused(self) -> None:
        assert any("twice" in e for e in validate_spec(_hill(["N", "N"])))

    def test_the_description_names_every_runner(self) -> None:
        assert "any of K or N reaching" in describe_spec(_hill(["K", "N"]))

    def test_removing_a_runner_keeps_the_goal_for_the_rest(self) -> None:
        spec = _hill(["B", "N"])
        drop_piece(spec, "N")
        assert spec["rules"]["reach"]["piece"] == ["B"]
        assert validate_spec(spec) == []

    def test_removing_the_last_runner_takes_the_goal_with_it(self) -> None:
        spec = _hill(["N"])
        drop_piece(spec, "N")
        assert "reach" not in spec["rules"]
