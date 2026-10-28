"""TC5 (`docs/plan-turncoat.md` §5.4 to §5.9): the purse and the market.

`rules.purse` gives each player coins (`start`, 0 to `PURSE_MAX`), filled by every bounty they
are paid (TC4), and a market (`market`, each type with a `pieces[].price`) to spend them in:
`MOVE_BUY` places a bought piece on an empty square of the buyer's region (`region`, by
default the first rank), as a whole move that cannot be taken back. The evaluator prices the
purse at the market's best rate (`eval.rs::coin_value`, `agents/evaluation.py::coin_value`).

Specs, not chess: `tests/_purse_market.json` (six by six, kings, a rook, knights and pawns,
pawns and knights for sale) and edits of it. The oracle for which buys a position offers is
this file's `_buys`: every affordable market type on every empty square of its compiled
region, read off the position and the compiled tables (a declared test oracle, rule 3); when
the side to move is not in check every one is legal, since placing a piece uncovers nothing.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents import build_agent
from chess_variant_sandbox.agents.evaluation import Evaluator, coin_value
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_BUY, decode_move, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import PURSE_MAX, validate_spec
from chess_variant_sandbox.tournament.play import choose_move, play_game, seat_rng

HERE = Path(__file__).parent
ROOT = HERE.parent
WIDTH = 6


def _market() -> dict[str, Any]:
    return json.loads((HERE / "_purse_market.json").read_text(encoding="utf-8"))


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              purse: tuple[int, int] | None = None) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement), purse=purse)


def _bought(engine: Engine, state: GameState) -> set[str]:
    return {
        engine.describe_move(state, move) for move in engine.legal_moves(state)
        if move_type(move) == MOVE_BUY
    }


def _buys(engine: Engine, state: GameState) -> set[int]:
    """The oracle: every affordable market type on every empty square of its region."""
    purse = engine.spec.purse
    assert purse is not None
    side = state.side_to_move
    out = set()
    for entry, (piece_type, price) in enumerate(zip(purse.market, purse.price, strict=True)):
        if price > state.purse[side]:
            continue
        for square in purse.squares[side][entry]:
            if state.board[square] == 0:
                out.add(square | (square << 8) | (MOVE_BUY << 16) | (piece_type << 20))
    return out


# --------------------------------------------------------------------------- the spec


class TestTheKeys:
    def test_the_test_game_is_valid_and_named_by_its_market(self) -> None:
        spec = _market()
        assert validate_spec(spec) == []
        plain = copy.deepcopy(spec)
        del plain["rules"]["purse"]
        for piece in plain["pieces"]:
            piece.pop("price", None)
            piece.pop("bounty", None)
        assert validate_spec(plain) == []
        assert spec_hash(plain) != spec_hash(spec)

    def test_a_price_of_zero_is_the_game_without_one(self) -> None:
        spec = _market()
        written = copy.deepcopy(spec)
        written["pieces"][1]["price"] = 0
        assert validate_spec(written) == []
        assert spec_hash(written) == spec_hash(spec)
        assert canonical_tables(compile_spec(written)) == canonical_tables(compile_spec(spec))

    @pytest.mark.parametrize("bad", [41, -1, 1.5, True, "3"])
    def test_a_price_is_a_whole_number_to_forty(self, bad: Any) -> None:
        spec = _market()
        spec["pieces"][2]["price"] = bad
        assert any("pieces[2].price" in error for error in validate_spec(spec))

    @pytest.mark.parametrize(
        ("edit", "words"),
        [
            (lambda s: s["pieces"][0].update(price=3), "on a royal"),
            (lambda s: s["pieces"][1].update(price=3), "which nothing charges"),
            (lambda s: s["rules"]["purse"].update(market=[]), "market is empty"),
            (lambda s: s["rules"]["purse"].pop("market"), "nothing to spend on"),
            (lambda s: s["rules"]["purse"].update(market=["K"]), "is royal"),
            (lambda s: s["rules"]["purse"].update(market=["P", "R"]), "'R' has no price"),
            (lambda s: s["rules"]["purse"].update(market=["Z"]), "not a declared piece"),
            (lambda s: s["rules"]["purse"].update(market=["P", "P"]), "lists a piece twice"),
            (lambda s: s["rules"]["purse"].update(start=PURSE_MAX + 1), "rules.purse.start"),
            (lambda s: s["rules"]["purse"].update(region=[]), "non-empty list"),
            (lambda s: s["rules"]["purse"].update(region=["g9"]), "not a square on this board"),
            (lambda s: s["rules"]["purse"].update(spend=1), "unknown keys: spend"),
            (lambda s: s["rules"]["purse"].pop("start"), "missing required key 'start'"),
        ],
    )
    def test_each_refusal_is_named(self, edit: Any, words: str) -> None:
        spec = _market()
        edit(spec)
        errors = validate_spec(spec)
        assert any(words in error for error in errors), errors

    def test_a_blast_beside_the_purse_is_refused(self) -> None:
        spec = _market()
        spec["rules"]["explode_on_capture"] = {}
        assert "O22" in refused_ids(validate_spec(spec))

    def test_a_market_type_with_no_square_it_may_stand_on_is_refused(self) -> None:
        spec = _market()
        spec["pieces"][2]["region"] = ["a3", "b3", "c3", "d3", "e3", "f3"]
        spec["setup"]["placements"][2]["squares"] = ["b3"]
        errors = validate_spec(spec)
        assert any("'N' has no square in rules.purse.region" in error for error in errors), errors
        spec["rules"]["purse"]["region"] = ["d1", "c3"]
        assert validate_spec(spec) == []

    def test_a_moveless_or_spy_or_wild_type_is_never_sold(self) -> None:
        spec = _market()
        spec["pieces"].append({"id": "W", "value": 1, "price": 1, "moves": []})
        spec["rules"]["purse"]["market"].append("W")
        assert any("has no moves" in error for error in validate_spec(spec))

    def test_a_bounty_needs_a_points_count_or_a_purse(self) -> None:
        spec = _market()
        assert "scoring" not in spec["rules"] and validate_spec(spec) == []
        del spec["rules"]["purse"]
        for piece in spec["pieces"]:
            piece.pop("price", None)
        assert any("or rules.purse" in error for error in validate_spec(spec))

    def test_the_description_says_what_is_sold_and_where_it_goes(self) -> None:
        text = describe_spec(_market())
        assert "P for 2, N for 4" in text
        assert "starting at 2" in text and f"at most {PURSE_MAX}" in text
        assert "the buyer's own first rank" in text
        assert "into the points and the purse" in text
        assert "costs 4 coins to buy, hire or bribe" in text

    def test_fairy_stockfish_refuses_the_purse_by_name(self) -> None:
        with pytest.raises(Untranslatable, match="rules.purse"):
            translate(_market())


# --------------------------------------------------------------------------- the tables


class TestTheTables:
    def test_a_game_without_a_purse_crosses_no_market(self) -> None:
        spec = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        tables = canonical_tables(compile_spec(spec))
        assert not {"purse_market", "purse_price", "purse_square_index", "purse_squares"} & set(
            tables
        )

    def test_the_market_crosses_its_types_prices_and_squares(self) -> None:
        compiled = compile_spec(_market())
        purse = compiled.purse
        assert purse is not None and purse.start == 2
        assert purse.market == (3, 2) and purse.price == (2, 4)
        first = tuple(range(WIDTH))
        last = tuple(range(WIDTH * (WIDTH - 1), WIDTH * WIDTH))
        assert purse.squares == ((first, first), (last, last))
        tables = build_tables(compiled)
        assert tables["purse_market"] == [3, 2] and tables["purse_price"] == [2, 4]
        assert tables["purse_square_index"] == [0, 6, 12, 18, 24]

    @pytest.mark.parametrize(("orientation", "image"), [("mirror", "b6"), ("rotate", "e6")])
    def test_a_named_region_is_reflected_for_the_second_player(
        self, orientation: str, image: str
    ) -> None:
        spec = _market()
        spec["setup"]["orientation"] = orientation
        spec["rules"]["purse"]["region"] = ["b1"]
        purse = compile_spec(spec).purse
        assert purse is not None
        assert purse.squares[0] == ((_sq("b1"),), (_sq("b1"),))
        assert purse.squares[1] == ((_sq(image),), (_sq(image),))

    def test_a_wall_in_the_region_is_never_a_square_to_buy_onto(self) -> None:
        spec = _market()
        spec["rules"]["obstacles"] = {"squares": ["f1"], "mirror": True}
        purse = compile_spec(spec).purse
        assert purse is not None
        assert _sq("f1") not in purse.squares[0][0] and _sq("f6") not in purse.squares[1][0]


# --------------------------------------------------------------------------- hand positions


class TestABuy:
    _KINGS = {"a1": ("K", 0), "a6": ("K", 1)}

    def test_a_buy_spends_the_price_and_places_the_piece(self) -> None:
        engine = Engine(_market())
        state = _position(engine, self._KINGS, purse=(5, 0))
        assert _bought(engine, state) == {
            f"{pid}${file}1" for pid in "PN" for file in "bcdef"
        }
        move = next(m for m in engine.legal_moves(state)
                    if engine.describe_move(state, m) == "N$d1")
        after = engine.apply(state, move)
        assert after.purse == (1, 0)
        assert after.board[_sq("d1")] == engine.spec.encode_piece(0, 2)
        assert after.key == engine._compute_hash(after)
        # Irreversible: the repetition window clears, and the no-progress clock resets.
        assert after.repetition == (after.key,) and after.halfmove == 0
        assert state.purse == (5, 0)  # `apply` leaves its argument untouched

    def test_nothing_is_offered_that_the_purse_cannot_pay(self) -> None:
        engine = Engine(_market())
        assert _bought(engine, _position(engine, self._KINGS, purse=(1, 9))) == set()
        assert {text[:2] for text in _bought(engine, _position(engine, self._KINGS,
                                                                purse=(3, 9)))} == {"P$"}

    def test_a_full_region_offers_nothing(self) -> None:
        engine = Engine(_market())
        full = _position(
            engine, {f"{file}1": ("P", 0) for file in "bcdef"} | self._KINGS, purse=(PURSE_MAX, 0)
        )
        assert _bought(engine, full) == set()
        # One square freed, one square offered, for each type.
        gap = _position(
            engine, {f"{file}1": ("P", 0) for file in "bcdf"} | self._KINGS, purse=(PURSE_MAX, 0)
        )
        assert _bought(engine, gap) == {"P$e1", "N$e1"}

    def test_the_second_player_buys_onto_its_own_first_rank(self) -> None:
        engine = Engine(_market())
        state = _position(engine, self._KINGS, side=1, purse=(0, 2))
        assert _bought(engine, state) == {f"P${file}6" for file in "bcdef"}

    def test_a_buy_that_blocks_a_check_is_the_only_kind_offered_in_check(self) -> None:
        engine = Engine(_market())
        state = _position(engine, {"a1": ("K", 0), "f1": ("R", 1), "a6": ("K", 1)},
                          purse=(4, 0))
        offered = _bought(engine, state)
        assert offered and all(text[-2:] in {"b1", "c1", "d1", "e1"} for text in offered)
        assert offered == {f"{pid}${file}1" for pid in "PN" for file in "bcde"}

    def test_the_placement_phase_places_before_anything_is_bought(self) -> None:
        spec = _market()
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] != "R"
        ]
        spec["rules"]["drops"] = {
            "from_captures": False, "placement": True, "stock": {"R": 1},
            "region": {"R": ["a1", "b1", "c1", "d1", "e1", "f1"]},
        }
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        state = engine.initial_state()
        assert state.purse == (2, 2)
        for _ in range(2):
            moves = engine.legal_moves(state)
            assert moves and all(move_type(m) != MOVE_BUY for m in moves)
            state = engine.apply(state, moves[0])
        assert any(move_type(m) == MOVE_BUY for m in engine.legal_moves(state))

    def test_a_bought_piece_promotes_on_arrival_in_its_zone(self) -> None:
        spec = _market()
        spec["rules"]["promote_on_entry"] = True
        spec["rules"]["purse"]["region"] = ["d1", "a6"]
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        state = _position(engine, {"c1": ("K", 0), "c6": ("K", 1)}, purse=(2, 0))
        move = next(m for m in engine.legal_moves(state)
                    if engine.describe_move(state, m) == "P$a6")
        after = engine.apply(state, move)
        assert after.board[_sq("a6")] == engine.spec.encode_piece(0, 1)  # the first target, R
        assert after.purse == (0, 0) and after.key == engine._compute_hash(after)

    def test_the_cap_holds_after_a_buy_and_a_bounty(self) -> None:
        engine = Engine(_market())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f2": ("R", 0),
                                   "f5": ("R", 1)}, purse=(PURSE_MAX, PURSE_MAX))
        after = engine.apply(state, next(
            m for m in engine.legal_moves(state) if decode_move(m)[:2] == (_sq("f2"), _sq("f5"))
        ))
        assert after.purse == (PURSE_MAX, PURSE_MAX)
        bought = engine.apply(after, next(
            m for m in engine.legal_moves(after) if move_type(m) == MOVE_BUY
        ))
        assert bought.purse in ((PURSE_MAX, PURSE_MAX - 2), (PURSE_MAX, PURSE_MAX - 4))
        assert bought.key == engine._compute_hash(bought)


# --------------------------------------------------------------------------- the oracle


class TestTheOracle:
    def test_twenty_walks_offer_the_oracles_buys_and_key_every_ply(self) -> None:
        """Twenty random walks: the buys offered are the oracle's (all of them out of check,
        a subset in check), each buy's price leaves the purse, and the incremental key is the
        full scan's at every ply."""
        engine = Engine(_market())
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
                offered = {m for m in moves if move_type(m) == MOVE_BUY}
                expected = _buys(engine, state)
                if engine.in_check(state):
                    assert offered <= expected
                else:
                    assert offered == expected
                move = rng.choice(moves)
                after = engine.apply(state, move)
                if move_type(move) == MOVE_BUY:
                    bought += 1
                    price = engine.spec.purse.price[engine.spec.purse.market.index(
                        decode_move(move)[3])]
                    mover = state.side_to_move
                    assert after.purse[mover] == state.purse[mover] - price
                assert after.key == engine._compute_hash(after)
                state = after
            assert dataclasses.replace(state) == state
        assert bought > 20

    def test_the_perft_is_frozen_and_unmake_puts_every_buy_back(self) -> None:
        engine = Engine(_market())
        state = engine.initial_state()
        assert [engine.perft(state, depth) for depth in range(1, 5)] == [12, 144, 1766, 21640]

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            if engine.terminal(position) is not None:
                return 0
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        rich = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "c3": ("N", 0),
                                  "d4": ("R", 1)}, purse=(6, 6))
        assert engine.perft(rich, 3) == count(rich, 3)


# --------------------------------------------------------------------------- the players


class TestTheCoinTerm:
    def test_a_coin_is_worth_the_markets_best_rate(self) -> None:
        engine = Engine(_market())
        evaluator = Evaluator.for_spec(engine.spec)
        rate = coin_value(engine.spec, evaluator.values)
        assert rate == max(evaluator.values[3] / 2, evaluator.values[2] / 4) > 0
        state = engine.initial_state()
        richer = state.with_(purse=(7, 2))
        delta = evaluator.evaluate(engine, richer, 0, 12) - evaluator.evaluate(engine, state, 0, 12)
        assert delta == pytest.approx(5 * rate)
        # The switch the term's duel holds a player without it by (`Evaluator.coins`).
        blind = dataclasses.replace(evaluator, coins=False)
        assert blind.evaluate(engine, richer, 0, 12) == blind.evaluate(engine, state, 0, 12)

    def test_no_game_without_a_purse_reads_it(self) -> None:
        spec = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        assert coin_value(compile_spec(spec), (1.0,) * 6) == 0.0

    @pytest.mark.parametrize("agent", ["greedy", "minimax:nodes=200"])
    def test_the_core_and_the_oracle_choose_alike_under_a_purse(self, agent: str) -> None:
        """The core's coin term is the declared evaluator's: one move at a time, the native
        player and the Python oracle agree over a walk that buys."""
        engine = Engine(_market())
        state = engine.initial_state().with_(purse=(6, 6))
        state = state.with_(key=engine._compute_hash(state))
        state = state.with_(repetition=(state.key,))
        for ply in range(16):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            native = choose_move(engine, agent, state, seed=ply)
            oracle = build_agent(agent, engine).select_move(state, moves, seat_rng(ply, 0))
            assert native == oracle, (ply, engine.describe_move(state, native))
            state = engine.apply(state, native)

    def test_a_searching_game_with_buys_replays(self) -> None:
        engine = Engine(_market())
        record = play_game(engine, ("minimax:nodes=200", "greedy"), 3)
        state = engine.initial_state()
        for move in record.moves:
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
        outcome = engine.terminal(state)
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
