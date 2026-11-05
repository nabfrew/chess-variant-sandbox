"""SR7g (R19, ruled 2026-10-27: *a collision like any other*): buying in rounds.

`docs/plan-schema-next.md` §2.3 row R19 and `docs/reference/rounds.md` §4 step 2. Under
`rules.purse` beside `rules.simultaneous` a buy is an arrival made with the round's moves, as
a drop is: two buys onto one square, or a buy and a piece's move onto one square, collide,
and `beats` and the tie decide, the bought piece by its own type. The price leaves the
buyer's purse at the departures, whether or not the bought piece survives. Every capture
pays its bounty as in `make`: to the taker (or under `bounty_to: "loser"` the piece's owner),
into the points and the purse. A purse that can hire (a priced piece placed for nobody) or
bribe stays refused beside rounds (the auction, R19's narrower row).

Every round here is checked against the declared oracle (`tests/simultaneous_oracle.py`),
and `tests/test_simultaneous_oracle.py` walks the two `_simultaneous_market*` specs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_BUY, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game

HERE = Path(__file__).parent
EVENT_COLLISION, EVENT_INITIATIVE = 0, 2


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


MARKET = load("market")
BLAST = load("market_blast")
KINGS = {"c1": ("K", 0), "c5": ("K", 1)}


def place(engine: Engine, pieces: dict[str, tuple[str, int]], purse: tuple[int, int] = (6, 6)) -> GameState:
    state = engine.make_state(pieces, purse=purse)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def move(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    o_want, t_want = engine._square_index(origin), engine._square_index(target)
    for m in engine.legal_moves_for(state, player):
        o, t, kind, _ = decode_move(m)
        if o == o_want and t == t_want and kind != MOVE_BUY:
            return m
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def buy(engine: Engine, state: GameState, player: int, pid: str, square: str) -> int:
    want = engine._square_index(square)
    piece_type = engine.spec.piece_ids.index(pid)
    for m in engine.legal_moves_for(state, player):
        _, t, kind, promotion = decode_move(m)
        if kind == MOVE_BUY and t == want and promotion == piece_type:
            return m
    raise AssertionError(f"player {player} cannot buy {pid} on {square}")


def round_of(engine: Engine, state: GameState, first: int, second: int):
    after, _, events = engine.apply_round_with_phase(state, first, second)
    assert after == resolve_round(engine, state, first, second)
    assert after.key == engine._compute_hash(after)
    return after, events


def at(engine: Engine, state: GameState, name: str) -> tuple[str, int] | None:
    value = state.board[engine._square_index(name)]
    if value == 0:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


class TestTheLift:
    def test_a_market_is_no_longer_refused_beside_rounds(self) -> None:
        assert validate_spec(MARKET) == []
        assert validate_spec(BLAST) == []

    def test_a_purse_that_bribes_stays_refused_beside_rounds(self) -> None:
        spec = copy.deepcopy(MARKET)
        spec["rules"]["purse"]["bribe"] = True
        assert "R19" in refused_ids(validate_spec(spec))

    def test_a_purse_that_hires_stays_refused_beside_rounds(self) -> None:
        spec = copy.deepcopy(MARKET)
        spec["pieces"].append({
            "id": "S", "name": "sellsword", "value": 2, "price": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "dihedral8", "mode": "both"}],
        })
        spec["setup"]["placements"].append({"piece": "S", "player": "nobody", "squares": ["e3"]})
        assert "R19" in refused_ids(validate_spec(spec))
        del spec["rules"]["simultaneous"]
        spec["rules"]["purse"].pop("region")
        assert "R19" not in refused_ids(validate_spec(spec))

    def test_a_buy_is_offered_to_each_player_on_the_round_board_while_it_is_affordable(self) -> None:
        engine = Engine(MARKET)
        state = place(engine, KINGS, purse=(1, 4))
        kinds = [{decode_move(m)[3] for m in engine.legal_moves_for(state, p)
                  if decode_move(m)[2] == MOVE_BUY} for p in (0, 1)]
        pawn = engine.spec.piece_ids.index("P")
        assert kinds == [set(), {pawn, engine.spec.piece_ids.index("N")}]


class TestTwoBuysOntoOneSquare:
    def test_a_tie_with_no_winner_leaves_the_square_empty_and_both_prices_paid(self) -> None:
        """A pawn beats only the king, so two pawns meet *no/no*: both go, nothing captured."""
        engine = Engine(MARKET)
        state = place(engine, KINGS)
        after, events = round_of(engine, state, buy(engine, state, 0, "P", "c3"),
                                 buy(engine, state, 1, "P", "c3"))
        assert at(engine, after, "c3") is None
        assert after.purse == (4, 4) and after.points == (0.0, 0.0)
        assert (EVENT_COLLISION, engine._square_index("c3"), -1) in events

    def test_the_winner_by_beats_stands_and_is_paid_the_loser_s_bounty(self) -> None:
        engine = Engine(MARKET)
        state = place(engine, KINGS)
        after, events = round_of(engine, state, buy(engine, state, 0, "N", "c3"),
                                 buy(engine, state, 1, "P", "c3"))
        assert at(engine, after, "c3") == ("N", 0)
        # Player 0 paid 4 and took a pawn's bounty of 1; player 1 paid 2 for nothing.
        assert after.purse == (3, 4) and after.points == (1.0, 0.0)
        assert (EVENT_COLLISION, engine._square_index("c3"), 0) in events

    def test_the_initiative_decides_a_tie_between_two_buys(self) -> None:
        spec = copy.deepcopy(MARKET)
        spec["rules"]["simultaneous"] = {"tie": "initiative"}
        engine = Engine(spec)
        state = place(engine, KINGS)
        holder = state.initiative
        after, events = round_of(engine, state, buy(engine, state, 0, "P", "c3"),
                                 buy(engine, state, 1, "P", "c3"))
        assert at(engine, after, "c3") == ("P", holder)
        purse = [4, 4]
        purse[holder] += 1
        assert after.purse == tuple(purse) and after.initiative == 1 - holder
        assert (EVENT_INITIATIVE, engine._square_index("c3"), holder) in events

    def test_buys_onto_two_squares_both_stand(self) -> None:
        engine = Engine(MARKET)
        state = place(engine, KINGS)
        after, _ = round_of(engine, state, buy(engine, state, 0, "P", "b3"),
                            buy(engine, state, 1, "N", "d3"))
        assert at(engine, after, "b3") == ("P", 0) and at(engine, after, "d3") == ("N", 1)
        assert after.purse == (4, 2)


class TestABuyAndAMoveOntoOneSquare:
    def test_the_moving_piece_wins_and_the_price_is_spent_all_the_same(self) -> None:
        """The rook beats everything (its `captures`), the pawn only the king."""
        engine = Engine(MARKET)
        state = place(engine, {**KINGS, "c4": ("R", 1)})
        after, _ = round_of(engine, state, buy(engine, state, 0, "P", "c3"),
                            move(engine, state, 1, "c4", "c3"))
        assert at(engine, after, "c3") == ("R", 1) and at(engine, after, "c4") is None
        assert after.purse == (4, 7) and after.points == (0.0, 1.0)

    def test_a_tie_both_beating_is_two_captures_each_paid(self) -> None:
        """The knight beats the rook and the rook the knight: both go under the default tie,
        and each player is paid the other's bounty."""
        engine = Engine(MARKET)
        state = place(engine, {**KINGS, "c4": ("R", 1)})
        after, _ = round_of(engine, state, buy(engine, state, 0, "N", "c3"),
                            move(engine, state, 1, "c4", "c3"))
        assert at(engine, after, "c3") is None and at(engine, after, "c4") is None
        assert after.purse == (6, 8) and after.points == (4.0, 2.0)

    def test_a_move_onto_a_square_a_buy_was_not_made_onto_takes_nothing(self) -> None:
        engine = Engine(MARKET)
        state = place(engine, {**KINGS, "c4": ("R", 1)})
        after, _ = round_of(engine, state, buy(engine, state, 0, "N", "a3"),
                            move(engine, state, 1, "c4", "c3"))
        assert at(engine, after, "a3") == ("N", 0) and at(engine, after, "c3") == ("R", 1)
        assert after.purse == (2, 6)


class TestThePayee:
    def test_under_bounty_to_loser_the_owner_of_the_taken_piece_is_paid(self) -> None:
        spec = copy.deepcopy(MARKET)
        spec["rules"]["purse"]["bounty_to"] = "loser"
        engine = Engine(spec)
        state = place(engine, KINGS)
        after, _ = round_of(engine, state, buy(engine, state, 0, "N", "c3"),
                            buy(engine, state, 1, "P", "c3"))
        assert at(engine, after, "c3") == ("N", 0)
        assert after.purse == (2, 5) and after.points == (0.0, 1.0)

    def test_the_cap_loses_what_passes_it(self) -> None:
        engine = Engine(MARKET)
        state = place(engine, {**KINGS, "c4": ("R", 1)}, purse=(98, 6))
        after, _ = round_of(engine, state, buy(engine, state, 0, "N", "c3"),
                            move(engine, state, 1, "c4", "c3"))
        assert after.purse == (98, 8)


class TestTheSearchPutsEveryRoundBack:
    def test_a_searched_game_replays_through_the_rounds_to_the_same_end(self) -> None:
        """`unmake_round` restores the purse: games the core plays with a round search (many
        `make_round`s and `unmake_round`s on one position a round) end where a replay of their
        rounds through `apply_round` ends, and some of their rounds buy."""
        for spec in (MARKET, BLAST):
            engine = Engine(spec)
            bought = 0
            for seed in range(2):
                record = play_game(engine, ("matrix:nodes=64", "matrix:nodes=64"), seed)
                state = engine.initial_state()
                pairs = list(zip(record.moves[0::2], record.moves[1::2], strict=True))
                for first, second in pairs:
                    bought += sum(decode_move(m)[2] == MOVE_BUY for m in (first, second))
                    state = engine.apply_round(state, first, second)
                outcome = engine.terminal(state)
                assert outcome is not None
                assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
            assert bought > 0


class TestTheAnalysisPassCountsThePurse:
    def test_each_round_is_two_side_turns_of_the_purse_series(self) -> None:
        """A market in rounds writes `PURSE_SERIES` as the turn path does: one entry a side-turn,
        each player's of a round read off the states before and after it (SR7g's leftover,
        fixed by SR7h)."""
        from chess_variant_sandbox.agents.evaluation import Evaluator
        from chess_variant_sandbox.tournament import analysis as analysis_pass
        from chess_variant_sandbox.tournament.analysis import PURSE_SERIES

        engine = Engine(MARKET)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        bought = 0
        for seed in (1, 2):
            record = play_game(engine, ("greedy", "random"), seed)
            found = analysis_pass._analyse_game(
                engine, evaluator, record, analysis_pass._epsilon_for(engine)
            )
            purse = found.purse
            assert purse is not None and list(purse) == list(PURSE_SERIES)
            assert all(len(series) == record.plies for series in purse.values())
            state = engine.initial_state()
            for index in range(0, len(record.moves), 2):
                pair = record.moves[index], record.moves[index + 1]
                after = engine.apply_round(state, *pair)
                for seat in (0, 1):
                    kind = decode_move(pair[seat])[2]
                    change = after.purse[seat] - state.purse[seat]
                    bought += int(kind == MOVE_BUY)
                    assert {key: purse[key][index + seat] for key in PURSE_SERIES} == {
                        "side": seat, "purse": state.purse[seat],
                        "spent": -change if kind == MOVE_BUY else 0,
                        "earned": 0 if kind == MOVE_BUY else max(change, 0),
                        "bought": int(kind == MOVE_BUY), "hired": 0, "bribed": 0,
                    }, (seed, index, seat)
                state = after
        assert bought > 0
