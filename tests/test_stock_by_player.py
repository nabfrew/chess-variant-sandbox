"""SR3b (`docs/plan-schema-next.md` §1 rank 6): a stock per player.

`rules.drops.stock_by_player` (`{"0": {...}, "1": {...}}`) gives each player their own stock in
place of the one `stock` gives both, as `promotion.to_by_player` sits beside `to`. The core's
`DropRule.stock` is per player and `royal_stocked` is read per player. The games here are chess
with a hand, only because chess is a committed game: nothing below reads chess.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.normal import normal_form
from chess_variant_sandbox.spec.transform import drop_piece
from tests.feature_inventory import _schema_fifty_nine_uses

ROOT = Path(__file__).resolve().parent.parent
HANDS = ("rulesets/crazyhouse.json", "rulesets/seirawan.json", "rulesets/shogi.json")


def _chess() -> dict[str, Any]:
    return load_spec(ROOT / "rulesets" / "chess.json")


def _handicap(first: dict[str, Any] | None = None, second: dict[str, Any] | None = None,
              **drops: Any) -> dict[str, Any]:
    """Chess in which the second player starts with two knights in hand (by default)."""
    spec = _chess()
    spec["rules"]["drops"] = {
        "from_captures": False,
        "stock_by_player": {"0": first or {}, "1": {"N": 2} if second is None else second},
        **drops,
    }
    return spec


def _texts(engine: Engine, state: Any) -> list[str]:
    return [engine.describe_move(state, move) for move in engine.legal_moves(state)]


class TestTheSpelling:
    def test_a_handicap_is_valid(self) -> None:
        assert validate_spec(_handicap()) == []

    def test_beside_stock_is_refused(self) -> None:
        spec = _handicap()
        spec["rules"]["drops"]["stock"] = {"N": 1}
        assert any("both 'stock' and 'stock_by_player'" in e for e in validate_spec(spec))

    def test_a_missing_player_is_refused(self) -> None:
        spec = _handicap()
        del spec["rules"]["drops"]["stock_by_player"]["0"]
        assert any("missing player '0'" in e for e in validate_spec(spec))

    def test_an_unknown_player_is_refused(self) -> None:
        spec = _handicap()
        spec["rules"]["drops"]["stock_by_player"]["2"] = {}
        assert any("unknown players: 2" in e for e in validate_spec(spec))

    def test_both_empty_is_refused(self) -> None:
        assert any("stocks nothing" in e for e in validate_spec(_handicap({}, {})))

    def test_a_bad_count_is_refused_where_it_stands(self) -> None:
        errors = validate_spec(_handicap(second={"N": 0}))
        assert any("stock_by_player['1']['N'] must be between 1" in e for e in errors)

    def test_an_endless_slot_beside_a_count_is_refused(self) -> None:
        spec = _handicap({"N": "unlimited"}, {"N": 1}, from_captures=True)
        assert any("'unlimited' supply for one player" in e for e in validate_spec(spec))

    def test_an_endless_slot_for_one_player_alone_is_valid(self) -> None:
        assert validate_spec(_handicap({}, {"N": "unlimited"})) == []

    def test_the_two_equal_is_stock(self) -> None:
        """The normal form writes one stock for both as `stock`, so it is the same game
        under the same name, and it compiles to the same tables."""
        by_player = _handicap({"N": 2}, {"N": 2})
        shared = _chess()
        shared["rules"]["drops"] = {"from_captures": False, "stock": {"N": 2}}
        assert validate_spec(by_player) == []
        assert normal_form(by_player)["rules"]["drops"] == {"from_captures": False, "stock": {"N": 2}}
        assert spec_hash(by_player) == spec_hash(shared)
        assert build_tables(compile_spec(by_player)) == build_tables(compile_spec(shared))


class TestAbsentIsTodaysTables:
    @pytest.mark.parametrize("path", HANDS)
    def test_a_committed_hand_crosses_no_per_player_stock(self, path: str) -> None:
        compiled = compile_spec(load_spec(ROOT / path))
        assert compiled.drops is not None and compiled.drops.stock_by_player == ()
        assert build_tables(compiled)["drops_stock_by_player"] == []
        assert "drops_stock_by_player" not in canonical_tables(compiled)

    def test_an_uneven_stock_crosses_both_and_no_shared_one(self) -> None:
        tables = build_tables(compile_spec(_handicap()))
        n_types = len(_chess()["pieces"])
        knight = [p["id"] for p in _chess()["pieces"]].index("N")
        assert tables["drops_stock"] == []
        assert len(tables["drops_stock_by_player"]) == 2 * n_types
        assert tables["drops_stock_by_player"][n_types + knight] == 2
        assert sum(tables["drops_stock_by_player"][:n_types]) == 0


class TestAHandicapPlays:
    def test_only_the_second_player_holds_the_knights(self) -> None:
        engine = Engine(_handicap())
        state = engine.initial_state()
        knight = [p["id"] for p in _chess()["pieces"]].index("N")
        assert sum(state.hands[0]) == 0
        assert state.hands[1][knight] == 2
        assert not any("@" in text for text in _texts(engine, state))
        state = engine.apply(state, engine.legal_moves(state)[0])
        drops = [text for text in _texts(engine, state) if "@" in text]
        assert drops and all(text.startswith("N@") for text in drops)

    def test_a_game_plays_to_its_end_and_the_hand_counts_down(self) -> None:
        engine = Engine(_handicap())
        rng = random.Random(3)
        state = engine.initial_state()
        knight = [p["id"] for p in _chess()["pieces"]].index("N")
        dropped = 0
        for _ in range(400):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            texts = [engine.describe_move(state, move) for move in moves]
            # Drop while the hand lasts, so the walk reaches every count.
            pick = next((m for m, t in zip(moves, texts, strict=True) if "@" in t), None)
            if pick is None:
                pick = rng.choice(moves)
            else:
                dropped += 1
            state = engine.apply(state, pick)
        assert dropped == 2
        assert state.hands[1][knight] == 0 or engine.terminal(state) is not None

    def test_the_described_rules_say_whose_hand(self) -> None:
        text = describe_spec(_handicap())
        assert "The second player starts holding 2 N" in text
        assert "The first player starts with nothing in hand." in text

    def test_a_handicap_is_a_distance_from_the_same_hand_for_both(self) -> None:
        shared = _chess()
        shared["rules"]["drops"] = {"from_captures": False, "stock": {"N": 2}}
        assert distance(_handicap(), shared) > 0

    def test_dropping_the_stocked_piece_leaves_a_valid_game(self) -> None:
        dropped = drop_piece(copy.deepcopy(_handicap()), "N")
        assert validate_spec(dropped) == []

    def test_the_inventory_names_it_only_where_the_two_differ(self) -> None:
        assert _schema_fifty_nine_uses(_handicap()) == {"rules.drops.stock_by_player"}
        assert _schema_fifty_nine_uses(_handicap({"N": 2}, {"N": 2})) == set()
        assert _schema_fifty_nine_uses(_chess()) == set()


def _royal_in_the_second_hand(**overrides: Any) -> dict[str, Any]:
    """Chess under a placement phase where the second player starts with its king in hand
    and the first places its own: one side's royal in its stock, the other's on the board."""
    spec = _chess()
    placements = []
    for row in spec["setup"]["placements"]:
        placements.append({"piece": row["piece"], "squares": list(row["squares"])})
        if row["piece"] != "K":
            placements.append({
                "piece": row["piece"], "player": 1,
                "squares": [sq[0] + str(9 - int(sq[1:])) for sq in row["squares"]],
            })
    spec["setup"] = {"mirror": False, "placements": placements}
    spec["rules"].pop("castling", None)
    spec["rules"]["drops"] = {
        "from_captures": False, "placement": True,
        "stock_by_player": {"0": {}, "1": {"K": 1}},
        **overrides,
    }
    return spec


class TestARoyalInHandReadPerPlayer:
    """H14 to H16 (`docs/plan-schema-next.md` §2.2) read per player."""

    def test_one_side_placing_and_the_other_stocking_is_valid(self) -> None:
        assert validate_spec(_royal_in_the_second_hand()) == []

    def test_h14_outside_the_placement_phase_names_the_player(self) -> None:
        spec = _royal_in_the_second_hand()
        del spec["rules"]["drops"]["placement"]
        assert any(
            "stock_by_player['1'] holds the royal 'K' outside the placement phase" in e
            for e in validate_spec(spec)
        )

    def test_h15_a_royal_stocked_twice_names_the_player(self) -> None:
        spec = _royal_in_the_second_hand()
        spec["rules"]["drops"]["stock_by_player"]["1"] = {"K": 2}
        assert any("stock_by_player['1'] holds 2 of the royal 'K'" in e for e in validate_spec(spec))

    def test_h16_placed_and_stocked_for_one_player_is_refused(self) -> None:
        spec = _royal_in_the_second_hand()
        spec["rules"]["drops"]["stock_by_player"] = {"0": {"K": 1}, "1": {"K": 1}}
        errors = validate_spec(spec)
        assert any("so player 0 would hold two" in e for e in errors)
        assert not any("player 1 would hold two" in e for e in errors)

    def test_the_core_reads_the_royal_in_hand_per_player(self) -> None:
        """The second player's king is in hand, not lost: the game is not over, and once
        the first player has moved its only moves are to place the king."""
        engine = Engine(_royal_in_the_second_hand())
        state = engine.initial_state()
        assert engine.terminal(state, engine.legal_moves(state)) is None
        assert not any("@" in text for text in _texts(engine, state))
        state = engine.apply(state, engine.legal_moves(state)[0])
        assert engine.terminal(state, engine.legal_moves(state)) is None
        texts = _texts(engine, state)
        assert texts and all(text.startswith("K@") for text in texts)
        state = engine.apply(state, engine.legal_moves(state)[0])
        assert engine.terminal(state, engine.legal_moves(state)) is None
        assert not any("@" in text for text in _texts(engine, state))
