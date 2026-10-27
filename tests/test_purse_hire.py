"""TC6 (`docs/plan-turncoat.md` §5.5): hiring and bribing.

`pieces[].price` on a piece nobody owns makes it for hire: a player with a piece of theirs next
to it (the type's `control` geometry, or the king's step) may spend the price and make it
theirs, where it stands, as a whole move (`MOVE_HIRE`). `rules.purse.bribe: true` lets a player
buy an enemy piece of a priced type the same way, at twice its price. The piece changes sides
through `Tables::rehome_value`, so a facing piece keeps its heading (§6).

Specs, not chess: `tests/_purse_hire.json` (six by six, kings, a rook, knights, pawns and a
mercenary standing for nobody on b3 and b4; knights and mercenaries priced; bribes on) and
edits of it. The oracle is `tests/neutral_oracle.py`'s `purchases`, a declared test oracle
(rule 3) that reads *next to* off the geometry itself.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.evaluation import Evaluator, coin_value
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_HIRE, move_from, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests.neutral_oracle import NeutralOracle

HERE = Path(__file__).parent
ROOT = HERE.parent
WIDTH = 6
K, R, N, M, P = range(5)


def _hire() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((HERE / "_purse_hire.json").read_text(encoding="utf-8"))
    return loaded


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              purse: tuple[int, int] = (9, 9)) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement), purse=purse)


def _bought(engine: Engine, state: GameState) -> set[str]:
    return {
        engine.describe_move(state, move) for move in engine.legal_moves(state)
        if move_type(move) == MOVE_HIRE
    }


def _buy(engine: Engine, state: GameState, text: str) -> GameState:
    move = next(m for m in engine.legal_moves(state) if engine.describe_move(state, m) == text)
    after = engine.apply(state, move)
    assert after.key == engine._compute_hash(after)
    return after


# --------------------------------------------------------------------------- the spec


class TestTheKeys:
    def test_the_test_game_is_valid(self) -> None:
        assert validate_spec(_hire()) == []

    def test_bribe_false_is_the_game_without_it(self) -> None:
        spec = _hire()
        spec["rules"]["purse"]["bribe"] = False
        spec["rules"]["purse"]["market"] = ["N"]
        written = copy.deepcopy(spec)
        del spec["rules"]["purse"]["bribe"]
        assert validate_spec(written) == validate_spec(spec) == []
        assert spec_hash(written) == spec_hash(spec)
        assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(spec))

    @pytest.mark.parametrize(
        ("edit", "words"),
        [
            (lambda s: s["rules"]["purse"].update(bribe=False), "which nothing charges"),
            (lambda s: s["rules"]["purse"].update(bribe="yes"), "must be true or false"),
            (lambda s: s["pieces"][1].update(spy=True), "on a spy"),
            (lambda s: s["rules"]["purse"].update(region=["a1"]), "region with no market"),
            (lambda s: s["rules"]["purse"].update(market=[]), "market is empty"),
        ],
    )
    def test_each_refusal_is_named(self, edit: Any, words: str) -> None:
        spec = _hire()
        if words == "on a spy":
            spec["pieces"][2]["spy"] = True
        else:
            edit(spec)
        errors = validate_spec(spec)
        assert any(words in error for error in errors), errors

    def test_a_purse_with_nothing_to_spend_on_is_refused(self) -> None:
        spec = _hire()
        spec["rules"]["purse"] = {"start": 2}
        for piece in spec["pieces"]:
            piece.pop("price", None)
        errors = validate_spec(spec)
        assert any("nothing to spend on" in error for error in errors), errors

    def test_a_bribe_with_no_priced_piece_a_player_holds_is_refused(self) -> None:
        spec = _hire()
        for piece in spec["pieces"]:
            piece.pop("price", None)
        spec["pieces"].append({"id": "Z", "value": 1, "price": 3, "moves": []})
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p.get("player") != "nobody"
        ]
        errors = validate_spec(spec)
        assert any("bribe with no priced piece a player can hold" in e for e in errors), errors

    def test_hiring_alone_needs_no_market_and_no_bribe(self) -> None:
        spec = _hire()
        del spec["rules"]["purse"]["bribe"]
        spec["pieces"][N].pop("price")
        assert validate_spec(spec) == []

    def test_the_description_says_who_may_be_hired_and_bribed(self) -> None:
        text = describe_spec(_hire())
        assert "hired" in text and "next to" in text
        assert "bribed" in text and "twice" in text

    def test_fairy_stockfish_refuses_the_purse_by_name(self) -> None:
        with pytest.raises(Untranslatable, match="rules.purse"):
            translate(_hire())


# --------------------------------------------------------------------------- the tables


class TestTheTables:
    def test_a_game_without_a_hire_crosses_none(self) -> None:
        for path in (ROOT / "rulesets" / "chess.json", HERE / "_purse_market.json"):
            spec = json.loads(path.read_text(encoding="utf-8"))
            tables = build_tables(compile_spec(spec))
            assert tables["purse_hire"] == tables["purse_bribe"] == []
            assert tables["purse_neighbour_index"] == tables["purse_neighbour_squares"] == []

    def test_hire_and_bribe_cross_their_costs(self) -> None:
        tables = build_tables(compile_spec(_hire()))
        assert tables["purse_hire"] == [0, 0, 0, 2, 0]
        assert tables["purse_bribe"] == [0, 0, 8, 4, 0]
        index = tables["purse_neighbour_index"]
        near = tables["purse_neighbour_squares"]
        slot = M * WIDTH * WIDTH + _sq("a1")
        assert sorted(near[index[slot]:index[slot + 1]]) == sorted(
            [_sq("a2"), _sq("b1"), _sq("b2")]
        )


# --------------------------------------------------------------------------- hand positions


class TestAHire:
    _KINGS = {"a1": ("K", 0), "f6": ("K", 1)}

    def test_a_hire_next_to_and_not_next_to(self) -> None:
        engine = Engine(_hire())
        near = _position(engine, {**self._KINGS, "c3": ("M", 2), "d4": ("P", 0)})
        assert _bought(engine, near) == {"$Mc3"}
        far = _position(engine, {**self._KINGS, "c3": ("M", 2), "e5": ("P", 0)})
        assert _bought(engine, far) == set()

    def test_a_hire_spends_the_price_and_the_piece_is_the_hirers(self) -> None:
        engine = Engine(_hire())
        state = _position(engine, {**self._KINGS, "c3": ("M", 2), "d4": ("P", 0)}, purse=(5, 1))
        after = _buy(engine, state, "$Mc3")
        assert after.purse == (3, 1)
        assert after.board[_sq("c3")] == engine.spec.encode_piece(0, M)
        assert after.repetition == (after.key,) and after.halfmove == 0
        assert state.purse == (5, 1)  # `apply` leaves its argument untouched
        # From then it is an ordinary piece of the hirer's, and the opponent may bribe it back.
        assert after.side_to_move == 1
        black = _position(engine, {**self._KINGS, "c3": ("M", 0), "c4": ("P", 1)}, side=1)
        assert _bought(engine, black) == {"$Mc3"}
        back = _buy(engine, black, "$Mc3")
        assert back.purse == (9, 5)
        assert back.board[_sq("c3")] == engine.spec.encode_piece(1, M)

    def test_nothing_is_offered_that_the_purse_cannot_pay(self) -> None:
        engine = Engine(_hire())
        placement = {**self._KINGS, "c3": ("M", 2), "d4": ("P", 0), "b4": ("N", 1),
                     "c5": ("P", 0)}
        assert _bought(engine, _position(engine, placement, purse=(9, 0))) == {"$Mc3", "$Nb4"}
        assert _bought(engine, _position(engine, placement, purse=(7, 0))) == {"$Mc3"}
        assert _bought(engine, _position(engine, placement, purse=(1, 0))) == set()

    def test_a_royal_or_unpriced_piece_is_never_offered(self) -> None:
        engine = Engine(_hire())
        placement = {"a1": ("K", 0), "b2": ("K", 1), "a2": ("R", 1), "b1": ("P", 1)}
        texts = {engine.describe_move(_position(engine, placement, purse=(99, 99)), m)
                 for m in engine.legal_moves(_position(engine, placement, purse=(99, 99)))}
        assert not any(text.startswith("$") for text in texts)

    def test_a_bribe_of_the_checker_is_offered_in_check(self) -> None:
        engine = Engine(_hire())
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "b3": ("N", 1),
                                   "c3": ("P", 0)})
        assert engine.in_check(state)
        assert "$Nb3" in _bought(engine, state)
        after = _buy(engine, state, "$Nb3")
        assert not engine.in_check(after, 0)
        assert after.purse == (1, 9)

    def test_a_hire_that_leaves_check_standing_is_not_offered(self) -> None:
        engine = Engine(_hire())
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "a6": ("R", 1),
                                   "d3": ("M", 2), "d4": ("P", 0)})
        assert engine.in_check(state)
        assert _bought(engine, state) == set()

    def test_the_hire_is_written_and_is_no_capture(self) -> None:
        engine = Engine(_hire())
        record = play_game(engine, ("greedy", "greedy"), 3)
        state = engine.initial_state()
        hires = 0
        for index, move in enumerate(record.moves):
            if move_type(move) == MOVE_HIRE:
                hires += 1
                assert engine.describe_move(state, move).startswith("$")
                assert record.captures[index] == 0 if record.captures else True
            state = engine.apply(state, move)
        assert state.key == engine._compute_hash(state)


def _facing_spec() -> dict[str, Any]:
    """A five by five scratch spec: Black's lancer faces right, White's guard stands next to it."""
    return {
        "schema_version": 31,
        "name": "bribed lancers",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]},
            {"id": "L", "value": 2, "price": 2, "facing": {"axes": 4},
             "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 2}]},
            {"id": "G", "value": 1, "moves": [
                {"kind": "leap", "vectors": [[1, 0]], "symmetry": "dihedral8"}]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "G", "player": 0, "squares": ["c2"]},
            {"piece": "L", "player": 1, "squares": ["c3"], "facing": "right"},
        ]},
        "rules": {"win": "checkmate", "max_plies": 80, "purse": {"start": 4, "bribe": True}},
    }


class TestAChangeOfSide:
    def test_a_bribed_facing_piece_keeps_its_heading(self) -> None:
        spec = _facing_spec()
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = engine.initial_state()
        before = state.board[_sq5("c3")]
        after = _buy(engine, state, "$Lc3")
        bought = after.board[_sq5("c3")]
        compiled = engine.spec
        assert compiled.owner_of(bought) == 0
        headings = compiled.facing_headings()
        assert (headings[0][compiled.facing_of(bought)]
                == headings[1][compiled.facing_of(before)])
        assert after.purse == (0, 4)

    def test_a_rest_ends_when_the_piece_changes_sides(self) -> None:
        spec = _hire()
        spec["pieces"][N]["rest"] = 1
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c3": ("P", 0),
                                   "e6": ("N", 1), "a6": ("P", 1)}, side=1)
        rested = engine.apply(state, next(m for m in engine.legal_moves(state)
                                          if engine.describe_move(state, m) == "Ne6-d4"))
        assert rested.rest[1], rested.rest
        after = _buy(engine, rested, "$Nd4")
        assert not after.rest[1]
        assert not after.rest[0]


def _sq5(name: str) -> int:
    return (int(name[1:]) - 1) * 5 + ord(name[0]) - ord("a")


# --------------------------------------------------------------------------- the oracle


class TestTheOracle:
    def test_twenty_walks_offer_the_oracles_moves_and_key_every_ply(self) -> None:
        """Twenty random walks: the legal list is the oracle's (hires and bribes worked out a
        second way), each purchase spends its cost and changes the piece's side, and the
        incremental key is the full scan's at every ply."""
        spec = _hire()
        engine = Engine(spec)
        oracle = NeutralOracle(spec, engine)
        bought = 0
        for walk in range(20):
            rng = random.Random(walk)
            state = engine.initial_state().with_(purse=(rng.randrange(12), rng.randrange(12)))
            state = state.with_(key=engine._compute_hash(state))
            state = state.with_(repetition=(state.key,))
            for _ in range(60):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                assert sorted(moves) == oracle.legal_moves(state)
                hires = [m for m in moves if move_type(m) == MOVE_HIRE]
                move = rng.choice(hires) if hires and rng.random() < 0.5 else rng.choice(moves)
                after = engine.apply(state, move)
                if move_type(move) == MOVE_HIRE:
                    bought += 1
                    mover = state.side_to_move
                    square = move_from(move)
                    cost = oracle.cost(state, square)
                    assert after.purse[mover] == state.purse[mover] - cost
                    assert oracle.owner(after.board[square]) == mover
                    assert oracle.type_of(after.board[square]) == oracle.type_of(
                        state.board[square])
                assert after.key == engine._compute_hash(after)
                state = after
            assert dataclasses.replace(state) == state
        assert bought > 20

    def test_the_perft_is_frozen_and_unmake_puts_every_purchase_back(self) -> None:
        engine = Engine(_hire())
        state = engine.initial_state()
        assert [engine.perft(state, depth) for depth in range(1, 4)] == PERFT

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            if engine.terminal(position) is not None:
                return 0
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        rich = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c3": ("M", 2),
                                  "c4": ("N", 1), "b3": ("P", 0), "d5": ("P", 1)},
                         purse=(9, 9))
        assert engine.perft(rich, 3) == count(rich, 3)


# Frozen when TC6 landed; every hire and bribe at each node is the oracle walk's reading.
PERFT = [11, 121, 1486]


# --------------------------------------------------------------------------- the players


class TestTheCoinTerm:
    def test_a_coin_is_worth_the_best_rate_a_hire_or_a_bribe_pays(self) -> None:
        engine = Engine(_hire())
        evaluator = Evaluator.for_spec(engine.spec)
        values = evaluator.values
        assert coin_value(engine.spec, values) == max(
            values[M] / 2, values[N] / 8, values[M] / 4,
        ) > 0
