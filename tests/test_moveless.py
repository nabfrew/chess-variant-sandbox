"""I1: pieces that never move (`docs/archive/plan-pieces-moves.md` §4).

`moves: []` is legal on any piece. These tests hold the rule's three halves: the shapes that
are mistakes are refused by name (§4.3, refusals 1 to 6 and the kept ones), the shapes that
are designs are not, and the core plays a moveless piece on the board as §4.2 says (it
blocks, it is taken, it never moves, a moveless royal is mated or taken). The library is
unmoved by the lift: `tests/test_schema_nine.py::FROZEN` and the perft tests hold that.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.describe import describe_pieces
from chess_variant_sandbox.spec.distance import distance
from chess_variant_sandbox.spec.plain import plain_problems
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament.play import play_game

ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
KNIGHT = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}


def _anchors() -> dict[str, Any]:
    """§4.1's `a1`: 6 x 6, a king and two rooks a side, and two anchors a side on c3/d3."""
    return {
        "name": "anchors",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "name": "rook", "value": 5, "moves": [ROOK]},
            {"id": "A", "name": "anchor", "value": 2, "moves": []},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "f1"]},
            {"piece": "A", "squares": ["c3", "d3"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 200},
    }


def _flag(win: str = "royal_capture") -> dict[str, Any]:
    """§4.1's `a6`: a royal flag that never moves on c1/c6, two rooks a side."""
    return {
        "name": "flag",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "R", "name": "rook", "value": 5, "moves": [ROOK]},
            {"id": "F", "name": "flag", "royal": True, "value": 0, "moves": []},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "F", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "f1"]},
        ]},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 200},
    }


def _with(spec: dict[str, Any], **rules: Any) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    spec["rules"].update(rules)
    return spec


def _piece(spec: dict[str, Any], pid: str) -> dict[str, Any]:
    return next(piece for piece in spec["pieces"] if piece["id"] == pid)


class TestWhatIsAdmitted:
    def test_an_anchor_on_the_board(self) -> None:
        assert validate_spec(_anchors()) == []

    @pytest.mark.parametrize("win", ["royal_capture", "checkmate"])
    def test_an_immobile_royal_under_either_royal_win(self, win: str) -> None:
        assert validate_spec(_flag(win)) == []

    def test_a_wild_type_written_with_no_moves(self) -> None:
        spec = _anchors()
        spec["pieces"].append({"id": "W", "name": "post", "value": 1, "moves": []})
        spec["rules"]["npcs"] = [{"piece": "W", "route": ["a3"]}]
        assert validate_spec(spec) == []

    def test_the_one_shot_bomb_is_a_composition(self) -> None:
        """Stratego's bomb as a one-shot: taking it destroys the taker too (§4.3)."""
        spec = _with(_anchors(), explode_on_capture={"radius": 0, "of": ["A"]})
        assert validate_spec(spec) == []

    def test_petrification_by_a_strike(self) -> None:
        spec = _with(_anchors(), strike={
            "effect": "demote", "chain": [{"from": "R", "to": "A"}], "position": "stay",
            "immune": ["K"],
        })
        assert validate_spec(spec) == []

    def test_a_promotion_into_one(self) -> None:
        spec = _anchors()
        spec["pieces"].append({"id": "P", "name": "pawn", "value": 1, "moves": [
            {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]})
        spec["setup"]["placements"].append({"piece": "P", "squares": ["b2"]})
        spec["rules"]["promotion"] = {"piece": "P", "rank": "last", "to": ["A"]}
        assert validate_spec(spec) == []

    def test_a_goal_for_a_moveless_piece_that_arrives_by_drop(self) -> None:
        spec = _with(_anchors(), reach={"piece": "A", "squares": ["a6"]},
                     drops={"from_captures": False, "stock": {"A": 1}})
        assert validate_spec(spec) == []

    def test_a_stocked_stone_keeps_a_round_s_keys(self) -> None:
        """A drop is the player's move, so a stone a hand holds may have a speed."""
        spec = _with(_anchors(), simultaneous={}, win="royal_capture",
                     drops={"from_captures": False, "stock": {"A": 1}})
        _piece(spec, "A")["speed"] = 1
        assert not [e for e in validate_spec(spec) if "never moves" in e]


def _one_sided_flag(**rules: Any) -> dict[str, Any]:
    spec = _flag()
    spec["setup"] = {"mirror": False, "placements": [
        {"piece": "F", "squares": ["c1"], "player": 0},
        {"piece": "R", "squares": ["a1"], "player": 0},
        {"piece": "F", "squares": ["c6"], "player": 1},
    ]}
    spec["rules"].update(rules)
    return spec


def _castling_on_anchors() -> dict[str, Any]:
    spec = _anchors()
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["c1"]}, {"piece": "A", "squares": ["a1", "f1"]},
        {"piece": "R", "squares": ["c3", "d3"]},
    ]
    return _with(spec, castling={"king": "K", "rook": "A"})


def _with_piece_key(key: str, value: Any, **rules: Any) -> dict[str, Any]:
    spec = _with(_anchors(), **rules)
    _piece(spec, "A")[key] = value
    return spec


# Each refusal of §4.3, on its scratch shape, with the words it must say.
REFUSALS = {
    "1 a player who can never move": (
        _one_sided_flag(), "player 1 can never move"),
    "1 a capture hand does not open it": (
        _one_sided_flag(drops={"from_captures": True}), "player 1 can never move"),
    "2 promotion": (
        _with(_anchors(), promotion={"piece": "A", "rank": "last", "to": ["R"]}),
        "never reaches its zone"),
    "3 castling's rook": (_castling_on_anchors(), "castling's rook 'A' has no moves"),
    "3 castling's king": (
        _with(_flag(), castling={"king": "F", "rook": "R"}), "castling's king 'F' has no moves"),
    "4 reach on arrival": (
        _with(_anchors(), reach={"piece": "A", "squares": ["a6"]}), "never arrives"),
    "4 reach on leaving": (
        _with(_anchors(), reach={"piece": "A", "squares": ["c3"], "on": "leave",
                                 "mirror": False}), "never leaves"),
    "5 a contact's mover": (
        _with(_anchors(), contacts={"links": [{"mover": "A", "onto": "R", "mover_to": "R"}],
                                    "immune": ["K"]}), "the link never fires"),
    "5 a side switch": (
        _with(_anchors(), strike={"switches_sides": ["A"], "immune": ["K"]}),
        "never captures"),
    "5 an explosion's capturers": (
        _with(_anchors(), explode_on_capture={"radius": 0, "by": ["A"]}),
        "lists only pieces with no moves"),
    "5 a portal end": (
        _with(_anchors(), portals=[{"orange": "b3", "blue": "e4", "carries": {"orange": ["A"]}}]),
        "carries only pieces with no moves"),
    "6 speed": (
        _with_piece_key("speed", 1, simultaneous={}, win="royal_capture"), "speed is on 'A'"),
    "6 beats": (
        _with_piece_key("beats", ["R"], simultaneous={}, win="royal_capture"), "beats is on 'A'"),
    "7 rest, as before": (_with_piece_key("rest", 1), "a piece that never moves never rests"),
    "7 facing": (_with_piece_key("facing", {"axes": 4}), "facing on a piece with no moves"),
}


class TestWhatIsRefused:
    @pytest.mark.parametrize("label", sorted(REFUSALS))
    def test_each_mistake_is_refused_in_words(self, label: str) -> None:
        spec, words = REFUSALS[label]
        errors = validate_spec(spec)
        assert any(words in error for error in errors), errors

    @pytest.mark.parametrize("label", sorted(REFUSALS))
    def test_each_reads_plainly_in_the_designer(self, label: str) -> None:
        spec, _ = REFUSALS[label]
        said = plain_problems(validate_spec(spec))
        assert said and all("rules." not in line and "pieces[" not in line for line in said), said

    def test_a_stock_opens_a_player_who_has_only_moveless_pieces(self) -> None:
        assert validate_spec(_one_sided_flag(drops={"stock": {"R": 1}})) == []

    def test_a_freeze_on_moveless_types_alone_is_refused_as_before(self) -> None:
        spec = _anchors()
        _piece(spec, "R")["sight"] = [{"vectors": [[1, 0]], "symmetry": "rotate4",
                                        "effect": "freeze", "whose": "enemy", "affects": ["A"]}]
        assert any("a freeze on types that have no move" in e for e in validate_spec(spec))


class TestTheCorePlaysIt:
    def test_an_anchor_makes_no_move_and_blocks(self) -> None:
        engine = Engine(_anchors())
        state = engine.initial_state()
        texts = engine.move_texts(state)
        assert len(texts) == 18, texts  # §4.1: the board's anchors add nothing
        assert not any(text.startswith("A") for text in texts)

    def test_random_games_never_move_an_anchor(self) -> None:
        """A move's text starts with the moving piece's letter (`Ra1-a2`)."""
        engine = Engine(_anchors())
        rng = random.Random(0)
        for _ in range(3):
            state = engine.initial_state()
            for _ply in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                assert not any(text.startswith("A") for text in engine.move_texts(state))
                state = engine.apply(state, rng.choice(moves))

    @pytest.mark.parametrize("label", ["anchors", "flag-mate", "flag-capture"])
    def test_the_short_game_walk_keeps_the_key(self, label: str) -> None:
        """`test_schema_two.py`'s walk, on these inline specs: twenty random plies, the
        incremental key against `_compute_hash` at each."""
        spec = {"anchors": _anchors, "flag-mate": lambda: _flag("checkmate"),
                "flag-capture": lambda: _flag("royal_capture")}[label]()
        engine = Engine(spec)
        rng = random.Random(7)
        state = engine.initial_state()
        for _ in range(20):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state)

    def test_an_anchor_is_taken(self) -> None:
        engine = Engine(_anchors())
        state = engine.make_state({"c1": ("K", 0), "c3": ("R", 0), "c6": ("K", 1),
                                   "c5": ("A", 1)})
        assert "Rc3xc5" in engine.move_texts(state)

    def test_a_checked_immobile_royal_with_no_answer_is_mated(self) -> None:
        """A rook checks the flag down the a-file, a second guards it, and nothing of the
        flag's side can take or block: mate, where a king would have stepped away."""
        engine = Engine(_flag("checkmate"))
        state = engine.make_state({"a1": ("F", 0), "f5": ("R", 0), "a3": ("R", 1),
                                   "e6": ("F", 1), "b2": ("R", 1)})
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 1 and outcome.reason == "checkmate"

    def test_the_same_check_under_royal_capture_is_taken_next_move(self) -> None:
        engine = Engine(_flag("royal_capture"))
        state = engine.make_state({"a1": ("F", 0), "f5": ("R", 0), "a3": ("R", 1),
                                   "e6": ("F", 1), "b2": ("R", 1)})
        assert engine.terminal(state) is None
        # Nothing obliges a defence, and none is possible: any move, then the flag falls.
        state = engine.apply(state, engine.legal_moves(state)[0])
        taking = [m for m in engine.legal_moves(state) if "a1" in engine.describe_move(state, m)]
        assert taking
        outcome = engine.terminal(engine.apply(state, taking[0]))
        assert outcome is not None and outcome.winner == 1

    def test_a_side_left_with_moveless_pieces_gets_the_stalemate_rule(self) -> None:
        spec = _with(_flag(), stalemate="loss")
        engine = Engine(spec)
        state = engine.make_state({"a1": ("F", 0), "e6": ("F", 1), "f5": ("R", 1)})
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 1


class TestThePlayersSearchIt:
    def test_a_game_whose_only_non_royals_never_move_is_searched(self) -> None:
        """§4.3's one thing to test rather than assume: kings and anchors alone, values
        as `eval_values` derives them, must not trip the evaluator's *every non-royal
        piece is worth nothing* guard (`eval.rs`, `Values::check`)."""
        spec = _anchors()
        spec["pieces"] = [p for p in spec["pieces"] if p["id"] != "R"]
        spec["setup"]["placements"] = [
            p for p in spec["setup"]["placements"] if p["piece"] != "R"]
        spec["rules"]["max_plies"] = 20
        assert validate_spec(spec) == []
        record = play_game(Engine(spec), ("minimax:nodes=200", "greedy"), seed=1)
        assert record.plies > 0


FAIRY_SHAPES = {
    "an anchor game": lambda: _anchors(),
    "an immobile royal under checkmate": lambda: _flag("checkmate"),
    "an immobile royal under royal capture (extinct)": lambda: _flag("royal_capture"),
}


class TestFairyStockfishAgrees:
    """§4.4: the three shapes cross with no new branch (`x:`, `king = f:`, the extinct
    model) and agree on legal moves. Depth 3 is the acceptance and takes seconds each, so
    it is `slow`; the fast suite asks depth 1 over one short game."""

    @staticmethod
    def _verify(spec: dict[str, Any], **kwargs: Any) -> Any:
        pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")
        from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

        return verify(spec, PyffishBackend(), "", **kwargs)

    @pytest.mark.parametrize("label", sorted(FAIRY_SHAPES))
    def test_a_shallow_walk_agrees(self, label: str) -> None:
        result = self._verify(FAIRY_SHAPES[label](), depth=1, games=1, plies=20)
        assert result.ok, result.reasons

    @pytest.mark.slow
    @pytest.mark.parametrize("label", sorted(FAIRY_SHAPES))
    def test_depth_three_agrees(self, label: str) -> None:
        result = self._verify(FAIRY_SHAPES[label](), depth=3)
        assert result.ok, result.reasons


class TestTheWords:
    def test_a_royal_that_never_moves_is_told_by_its_win(self) -> None:
        taken = describe_pieces(_flag("royal_capture"))
        mated = describe_pieces(_flag("checkmate"))
        assert "taking it is how the game is won" in " ".join(next(
            p for p in taken if p.id == "F").notes)
        assert "never by stepping away" in " ".join(next(
            p for p in mated if p.id == "F").notes)

    def test_an_anchor_and_its_stepping_twin_are_apart(self) -> None:
        twin = _anchors()
        _piece(twin, "A")["moves"] = [KING]
        assert distance(_anchors(), twin) > 0
