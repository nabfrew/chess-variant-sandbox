"""A round's reveal says batch E's rounds as the core played them (Q1099).

`web.play.round_reveal` names the core's events; three of batch E's lifts reach it:

- **A piece shot in a round** (SR7e, R11): the shot strikes before anything moves, so the
  struck piece's own move is `lost` in a round of equal speeds too. The sentence says the
  shooter *shot* it, and the page's story (`roundStory`) draws the lost move as lost and the
  shooter as staying, never as a piece arriving where it did not go.
- **A missed strike** (SR7f, R18): two strikes that stay at one piece nobody owns meet by
  `beats` and the tie; the loser *missed*. The core logs it as a void at the loser's origin,
  but it went for nothing that had moved, so the sentence and the result word are its own.
- **A buy in a round** (SR7g, R19): a buy enters play with no origin, as a drop does, and is
  named by the piece it buys, never by whatever stood on its square before.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from test_web import _in_page

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_BUY, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.web.play import round_reveal

HERE = Path(__file__).parent
NOBODY = 2
SIDES = ("white", "black")


def _load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


def _place(engine: Engine, pieces: dict[str, tuple[str, int]], **extra: Any) -> GameState:
    state = engine.make_state(pieces, **extra)
    if engine.spec.drops is not None:
        state = state.with_(hands=engine.initial_state().hands)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def _find(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    o_want, t_want = engine._square_index(origin), engine._square_index(target)
    for move in engine.legal_moves_for(state, player):
        o, t, kind, _ = decode_move(move)
        if o == o_want and t == t_want and kind != MOVE_BUY:
            return move
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def _buy(engine: Engine, state: GameState, player: int, pid: str, square: str) -> int:
    want, piece_type = engine._square_index(square), engine.spec.piece_ids.index(pid)
    for move in engine.legal_moves_for(state, player):
        _, t, kind, promotion = decode_move(move)
        if kind == MOVE_BUY and t == want and promotion == piece_type:
            return move
    raise AssertionError(f"player {player} cannot buy {pid} on {square}")


def _reveal(engine: Engine, state: GameState, first: int, second: int) -> dict[str, Any]:
    after, _, events = engine.apply_round_with_phase(state, first, second)
    return {"after": after,
            "reveal": round_reveal(engine, state, after, (first, second), events, SIDES)}


class TestAShotInARound:
    def test_the_struck_piece_was_shot_and_its_move_lost(self) -> None:
        engine = Engine(_load("rifle"))
        state = _place(engine, {"e1": ("K", 0), "e5": ("K", 1), "a1": ("R", 0),
                                "a4": ("N", 1)})
        rook, knight = (engine.spec.piece_names[engine.spec.piece_ids.index(p)] for p in "RN")
        got = _reveal(engine, state, _find(engine, state, 0, "a1", "a4"),
                      _find(engine, state, 1, "a4", "c3"))
        reveal = got["reveal"]
        assert reveal["results"] == [None, "lost"]
        assert reveal["words"] == [f"White's {rook} shot black's {knight} on a4 before it "
                                   "could move: its move was lost."]
        # The page's story: the shooter stays, the knight's move is lost and the knight is
        # taken where it stood -- never drawn arriving on c3.
        story = _in_page(
            f"const B = {json.dumps(list(state.board))}, A = {json.dumps(list(got['after'].board))},"
            f" R = {json.dumps(reveal)};",
            "roundStory(B, A, R)")
        kinds = {move["player"]: move["kind"] for move in story["moves"]}
        assert kinds == {0: "shot", 1: "lost"}
        assert story["moves"][1]["taken"] is True
        assert story["taken"] == [engine._square_index("a4")]


class TestAMissedStrike:
    def _game(self, tie: dict[str, Any] | None = None) -> tuple[Engine, dict[str, str]]:
        spec = _load("nobody_rifle")
        if tie is not None:
            spec = copy.deepcopy(spec)
            spec["rules"]["simultaneous"] = tie
        assert validate_spec(spec) == []
        engine = Engine(spec)
        names = {pid: engine.spec.piece_names[engine.spec.piece_ids.index(pid)]
                 for pid in ("N", "P", "R", "S")}
        return engine, names

    def test_the_loser_of_two_shots_missed(self) -> None:
        engine, names = self._game()
        state = _place(engine, {"c1": ("K", 0), "c5": ("K", 1), "b1": ("N", 0),
                                "d4": ("P", 1), "c3": ("S", NOBODY)})
        reveal = _reveal(engine, state, _find(engine, state, 0, "b1", "c3"),
                         _find(engine, state, 1, "d4", "c3"))["reveal"]
        assert reveal["results"] == [None, "missed"]
        assert reveal["words"] == [
            f"Black's {names['P']} struck at the {names['S']} on c3 too, but white's "
            f"{names['N']}'s strike won: it missed and stayed."]
        assert not any("which had moved" in line for line in reveal["words"])

    def test_two_tied_shots_with_no_winner_both_missed(self) -> None:
        engine, names = self._game({})
        state = _place(engine, {"c1": ("K", 0), "c5": ("K", 1), "a3": ("R", 0),
                                "e3": ("R", 1), "c3": ("S", NOBODY)})
        reveal = _reveal(engine, state, _find(engine, state, 0, "a3", "c3"),
                         _find(engine, state, 1, "e3", "c3"))["reveal"]
        assert reveal["results"] == ["missed", "missed"]
        assert reveal["words"] == [
            f"White's {names['R']} and black's {names['R']} both struck at the {names['S']} "
            "on c3, and neither beat the other: both missed and stayed."]

    def test_the_move_list_says_missed(self) -> None:
        said = _in_page("", "roundRowHtml(1, {player: 0, text: 'Ra3xc3', result: 'missed'},"
                            " {player: 1, text: 'Re3xc3', result: 'missed', reveal: {}})")
        assert said.count('data-result="missed">missed</span>') == 2


class TestABuyInARound:
    def test_a_buy_is_named_by_what_it_buys_and_has_no_origin(self) -> None:
        engine = Engine(_load("market"))
        state = _place(engine, {"c1": ("K", 0), "c5": ("K", 1)}, purse=(6, 6))
        pawn = engine.spec.piece_names[engine.spec.piece_ids.index("P")]
        reveal = _reveal(engine, state, _buy(engine, state, 0, "P", "c3"),
                         _buy(engine, state, 1, "P", "c3"))["reveal"]
        assert [move["from"] for move in reveal["moves"]] == [None, None]
        assert reveal["words"] == [f"White's {pawn} and black's {pawn} met on c3: both "
                                   "were lost."]
