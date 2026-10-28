"""The duck (generation 42, N4b, `docs/archive/plan-npc.md` §4.6).

``rules.duck``: one piece nobody owns, which the mover moves as the second half of every turn,
to any empty square (``goes: "anywhere"``, Duck chess) or by its type's moves (``"by_moves"``),
passing only when it has nowhere to go. Nothing takes it and it takes nothing. The compiled turn
is two moves; ``remaining`` 1 owes the duck half (`Game::duck_owed`).

Here: the opening's perft frozen and held to Fairy-Stockfish's ``duck`` (pyffish); the duck's
moves (a placement, a step, the *may not stay*, a ``by_moves`` walk, the pass and its undo);
a king taken by the piece move ending the game with no duck half; the wild phase once a turn;
en passant closed by the duck; repetition and the clocks unasked mid-turn; the duck's sight;
the refusals; the words; a seeded random walk; and the agents searching both halves, native
and oracle alike, the same game twice.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.moves import (
    DUCK_PASS,
    DUCK_PLACE,
    DUCK_STEP,
    MOVE_DUCK,
    decode_move,
)
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.tournament.play import play_game, play_game_python

ROOT = Path(__file__).resolve().parents[1]
CLASSIC = ROOT / "tests" / "_duck_chess.json"
SHEPHERD = ROOT / "tests" / "_duck_shepherd.json"
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
ROOK = [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _classic() -> Engine:
    return Engine(_load(CLASSIC))


def _small(goes: str | None = None, duck_moves: Any = (), **rules: Any) -> dict[str, Any]:
    """Kings on a1 and f6, a white rook on a6, a black pawn on c5 and the duck ``D``, on a 6x6
    board won by taking the king."""
    duck: dict[str, Any] = {"piece": "D"}
    if goes is not None:
        duck["goes"] = goes
    return {
        "name": "duck hand positions",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING},
            {"id": "R", "name": "rook", "value": 5, "moves": ROOK},
            {"id": "P", "name": "stone", "value": 1, "moves": KING},
            {"id": "D", "name": "duck", "value": 0, "moves": list(duck_moves)},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 200,
                  "duck": duck, **rules},
    }


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _name(engine: Engine, square: int) -> str:
    return "abcdefghijklmnop"[square % engine.spec.width] + str(square // engine.spec.width + 1)


def _duck_moves(engine: Engine, state: GameState) -> list[tuple[str, str, int]]:
    moves = engine.legal_moves(state)
    assert all(decode_move(m)[2] == MOVE_DUCK for m in moves)
    return [(_name(engine, decode_move(m)[0]), _name(engine, decode_move(m)[1]), decode_move(m)[3])
            for m in moves]


def _play(engine: Engine, state: GameState, text: str) -> GameState:
    after = engine.apply(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after


class TestTheOpening:
    """Chess with the classic duck: 20 piece moves, then 32 duck squares each."""

    def test_perft_is_frozen(self) -> None:
        engine = _classic()
        state = engine.initial_state()
        assert [engine.perft(state, depth) for depth in (1, 2, 3)] == [20, 640, 12240]

    def test_perft_four_is_fairy_stockfishs_perft_two(self) -> None:
        """Our two plies are one of their moves (`e2e4,e4d5`): ours at 2k is theirs at k.
        379,440 was counted on pyffish 0.0.90's ``duck`` (2026-10-26, N4b)."""
        engine = _classic()
        assert engine.perft(engine.initial_state(), 4) == 379440

    def test_against_pyffish_live(self) -> None:
        sf = pytest.importorskip("pyffish")
        fen = sf.start_fen("duck")
        theirs = sf.legal_moves("duck", fen, [])
        engine = _classic()
        assert engine.perft(engine.initial_state(), 2) == len(theirs) == 640

    def test_the_duck_starts_off_the_board_and_is_placed(self) -> None:
        engine = _classic()
        state = engine.initial_state()
        assert not engine.duck_owed(state) and engine.duck_square(state) is None
        state = _play(engine, state, "Pe2-e4")
        assert engine.duck_owed(state) and state.side_to_move == 0
        moves = _duck_moves(engine, state)
        empty = {_name(engine, sq) for sq, held in enumerate(state.board) if not held}
        assert {target for _, target, _ in moves} == empty and len(moves) == 32
        assert all(origin == target and how == DUCK_PLACE for origin, target, how in moves)
        state = _play(engine, state, "the duck lands on d5")
        assert not engine.duck_owed(state) and state.side_to_move == 1
        assert engine.duck_square(state) == _sq(engine, "d5")


class TestTheDucksMoves:
    def test_the_duck_may_not_stay(self) -> None:
        engine = _engine(_small())
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("D", 2)}, duck_owed=True)
        moves = _duck_moves(engine, state)
        assert all(origin == "c3" and how == DUCK_STEP for origin, _, how in moves)
        assert {target for _, target, _ in moves} == {
            _name(engine, sq) for sq in range(36) if not state.board[sq]
        }
        assert "c3" not in {target for _, target, _ in moves} and len(moves) == 33

    def test_a_duck_by_its_moves(self) -> None:
        engine = _engine(_small("by_moves", KING))
        state = engine.make_state(
            {"a1": ("K", 0), "f6": ("K", 1), "c3": ("D", 2), "c4": ("R", 0)}, duck_owed=True,
        )
        targets = sorted(target for _, target, _ in _duck_moves(engine, state))
        assert targets == ["b2", "b3", "b4", "c2", "d2", "d3", "d4"]

    def test_a_boxed_duck_passes_and_the_pass_comes_back(self) -> None:
        engine = _engine(_small("by_moves", KING))
        state = engine.make_state(
            {"a1": ("D", 2), "a2": ("K", 0), "b1": ("R", 0), "b2": ("P", 1), "f6": ("K", 1)},
            duck_owed=True,
        )
        [move] = engine.legal_moves(state)
        assert decode_move(move)[2:] == (MOVE_DUCK, DUCK_PASS)
        assert engine.describe_move(state, move) == "the duck has nowhere to go"
        after = engine.apply(state, move)
        assert after.board == state.board and after.side_to_move == 1
        assert not engine.duck_owed(after) and after.key == engine._compute_hash(after)
        # The core's make and unmake of the pass: perft walks it and back.
        assert engine.perft(state, 1) == 1
        assert engine.perft(state, 2) == len(engine.legal_moves(after))
        assert engine.perft(state, 3) == sum(
            engine.perft(engine.apply(after, m), 1) for m in engine.legal_moves(after)
        )

    def test_nothing_takes_the_duck(self) -> None:
        engine = _engine(_small())
        state = engine.make_state({"a1": ("K", 0), "a4": ("R", 0), "f6": ("K", 1), "a5": ("D", 2)})
        targets = {_name(engine, decode_move(m)[1]) for m in engine.legal_moves(state)}
        assert "a5" not in targets and "a6" not in targets and "a4" not in targets
        assert {"a2", "a3", "b4"} <= targets

    def test_the_words(self) -> None:
        engine = _engine(_small())
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("D", 2)}, duck_owed=True)
        texts = {engine.describe_move(state, m) for m in engine.legal_moves(state)}
        assert "the duck to d5" in texts
        words = describe_spec(_small())
        assert "duck" in words


class TestTheTurn:
    def test_a_king_taken_by_the_piece_move_ends_the_game_at_once(self) -> None:
        engine = _engine(_small())
        state = engine.make_state({"a1": ("K", 0), "f1": ("R", 0), "f6": ("K", 1), "c3": ("D", 2)})
        state = _play(engine, state, "Rf1xf6")
        assert engine.duck_owed(state)
        outcome = engine.terminal(state)
        assert outcome is not None and (outcome.winner, outcome.reason) == (0, "royal_captured")

    def test_the_clocks_are_asked_only_at_a_turns_start(self) -> None:
        engine = _engine(_small(no_progress_draw=10))
        owed = engine.make_state(
            {"a1": ("K", 0), "f6": ("K", 1), "c3": ("D", 2)}, halfmove=10, duck_owed=True,
        )
        assert engine.terminal(owed) is None
        after = _play(engine, owed, "the duck to c4")
        assert after.halfmove == 10
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.reason == "no_progress"

    def test_repetition_is_asked_only_at_a_turns_start(self) -> None:
        engine = _engine(_small())
        for owed in (False, True):
            state = engine.make_state(
                {"a1": ("K", 0), "f6": ("K", 1), "c3": ("D", 2)}, duck_owed=owed,
            )
            state = state.with_(repetition=(state.key,) * 3)
            outcome = engine.terminal(state)
            assert (outcome is None) is owed

    def test_the_no_progress_clock_counts_piece_moves(self) -> None:
        engine = _classic()
        state = engine.initial_state()
        for text in ("Ng1-f3", "the duck lands on d4", "Ng8-f6", "the duck to d5"):
            state = _play(engine, state, text)
        assert state.halfmove == 2 and state.ply == 4

    def test_an_odd_ply_limit_is_refused(self) -> None:
        assert any("max_plies 201 is odd" in e for e in validate_spec(_small(max_plies=201)))


class TestEnPassant:
    def _after_double_step(self, duck: str) -> tuple[Engine, GameState]:
        engine = _classic()
        state = engine.make_state(
            {"e1": ("K", 0), "e8": ("K", 1), "e5": ("P", 0), "d7": ("P", 1), "h3": ("D", 2)},
            side_to_move=1,
        )
        state = _play(engine, state, "Pd7-d5")
        assert engine.duck_owed(state) and state.ep_target == _sq(engine, "d6")
        return engine, _play(engine, state, f"the duck to {duck}")

    def test_the_window_survives_the_duck_move(self) -> None:
        engine, state = self._after_double_step("h4")
        texts = {engine.describe_move(state, m) for m in engine.legal_moves(state)}
        assert "Pe5xd6 e.p." in texts

    def test_a_duck_on_the_target_closes_it(self) -> None:
        engine, state = self._after_double_step("d6")
        texts = {engine.describe_move(state, m) for m in engine.legal_moves(state)}
        assert not any(text.startswith("Pe5x") for text in texts)


class TestBesideWildPieces:
    def test_the_phase_runs_once_a_turn_after_the_duck_half(self) -> None:
        engine = Engine(_load(SHEPHERD))
        state = engine.initial_state()
        after, phase = engine.apply_with_phase(state, engine.find_move(state, "Pe2-e3"))
        assert phase == () or list(phase) == []
        after, phase = engine.apply_with_phase(after, engine.find_move(after, "the duck lands on c6"))
        assert len(phase) == 2 and {step.kind for step in phase} == {"step"}
        assert after.key == engine._compute_hash(after)

    def test_a_wolf_waits_at_the_duck_and_the_duck_never_walks_a_route(self) -> None:
        engine = Engine(_load(SHEPHERD))
        state = engine.make_state(
            {"e1": ("K", 0), "e8": ("K", 1), "c4": ("W", 2), "d4": ("D", 2), "h1": ("R", 0)},
        )
        state = _play(engine, state, "Rh1-h2")
        after, phase = engine.apply_with_phase(state, engine.find_move(state, "the duck to d3"))
        # The duck left d4 before the phase, so the wolf steps onto it.
        assert [(_name(engine, s.origin), _name(engine, s.target)) for s in phase] == [("c4", "d4")]
        state = engine.make_state(
            {"e1": ("K", 0), "e8": ("K", 1), "c4": ("W", 2), "d4": ("D", 2), "h1": ("R", 0)},
            duck_owed=True,
        )
        after, phase = engine.apply_with_phase(state, engine.find_move(state, "the duck to e4"))
        # Now the duck on e4 stands on the route, and the wolf on c4 steps to d4; the duck stays.
        assert engine.duck_square(after) == _sq(engine, "e4")
        assert [(_name(engine, s.origin), _name(engine, s.target)) for s in phase] == [("c4", "d4")]
        after, phase = engine.apply_with_phase(after, engine.find_move(after, "Ke8-f8"))
        assert phase == () or list(phase) == []
        after, phase = engine.apply_with_phase(after, engine.find_move(after, "the duck to a8"))
        assert engine.duck_square(after) == _sq(engine, "a8")

    def test_the_ducks_shield(self) -> None:
        """N2's duck half: the duck's sight shields the pieces round it from both players."""
        engine = Engine(_load(SHEPHERD))
        state = engine.make_state(
            {"e1": ("K", 0), "e8": ("K", 1), "a1": ("R", 0), "a7": ("P", 1), "b7": ("D", 2)},
        )
        texts = {engine.describe_move(state, m) for m in engine.legal_moves(state)}
        assert "Ra1xa7" not in texts and "Ra1-a6" in texts
        state = engine.make_state(
            {"e1": ("K", 0), "e8": ("K", 1), "a1": ("R", 0), "a7": ("P", 1), "d7": ("D", 2)},
        )
        texts = {engine.describe_move(state, m) for m in engine.legal_moves(state)}
        assert "Ra1xa7" in texts


def _refused(spec: dict[str, Any], words: str) -> None:
    errors = validate_spec(spec)
    assert any(words in error for error in errors), errors


def _refused_pair(spec: dict[str, Any], rid: str) -> None:
    """A refused pair, by its id in `spec/refusals.py`."""
    errors = validate_spec(spec)
    assert rid in refused_ids(errors), errors


class TestTheRefusals:
    """The thirteen of §4.6, each by name, and the shape."""

    def test_the_spec_these_edit_is_valid(self) -> None:
        assert validate_spec(_small()) == []
        assert validate_spec(_small("by_moves", KING)) == []

    def test_beside_a_check_rule(self) -> None:
        for spec in (_small(win="checkmate"), _small(forbid_check=True), _small(check_limit=3)):
            _refused_pair(spec, "K1")

    def test_in_rounds(self) -> None:
        _refused_pair(_small(simultaneous={}), "R15")

    def test_beside_a_turn(self) -> None:
        _refused_pair(_small(turn={"moves": 2}), "L7")

    def test_beside_portals(self) -> None:
        portal = [{"at": ["b3", "e4"]}]
        _refused_pair(_small(portals=portal), "K2")

    def test_beside_a_blast_a_strike_or_contacts(self) -> None:
        _refused_pair(_small(explode_on_capture={"radius": 1}), "K3")
        _refused_pair(_small(strike={"effect": "remove", "position": "stay"}), "K3")

    def test_a_duck_that_sweeps(self) -> None:
        """A player's sweep beside the duck is allowed since SR2 (W9: the duck blocks it,
        `tests/test_sweep_blockers.py`); a duck whose own type sweeps is not."""
        spec = _small(goes="by_moves", duck_moves=[
            {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}])
        _refused(spec, "the duck's type 'D' sweeps")
        spec = _small()
        spec["pieces"][1]["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}]
        assert validate_spec(spec) == []

    def test_a_duck_type_with_what_a_duck_cannot_mean(self) -> None:
        for key, value, words in (
            ("royal", True, "is royal"),
            ("captures", ["K"], "has captures"),
            ("rest", 1, "has a rest"),
            ("region", ["a1", "b1"], "has a region"),
        ):
            spec = _small()
            spec["pieces"][3][key] = value
            _refused(spec, words)
        spec = _small("by_moves", [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4",
                                    "mode": "capture"}])
        _refused(spec, "capture-only move")
        spec = _small("by_moves", [{"kind": "leap", "vectors": [[0, 1]], "symmetry": "rotate4",
                                    "condition": "first_move"}])
        _refused(spec, "move condition")

    def test_by_moves_that_read_differently(self) -> None:
        spec = _small("by_moves", [{"kind": "leap", "vectors": [[0, 1]]}])
        _refused(spec, "reads differently for the two players")

    def test_by_moves_on_a_type_with_no_moves(self) -> None:
        _refused(_small("by_moves"), 'goes "by_moves" on a type with no moves')

    def test_moves_on_a_duck_that_goes_anywhere(self) -> None:
        _refused(_small(None, KING), "never reads them; write moves: []")

    def test_the_ducks_type_anywhere_else(self) -> None:
        spec = _small()
        spec["setup"]["placements"].append({"piece": "D", "player": 0, "squares": ["c3"]})
        _refused(spec, "named in setup.placements[2]")
        spec = _small()
        spec["pieces"][1]["captures"] = ["K", "D"]
        _refused(spec, "named in pieces 'R' captures")
        spec = _small(reach={"piece": "D", "squares": ["f1"]})
        _refused(spec, "named in rules.reach")

    def test_a_start_the_opening_fills(self) -> None:
        spec = _small()
        spec["rules"]["duck"]["start"] = "a1"
        _refused(spec, "rules.duck.start a1 is a square the opening does not leave empty")
        spec["rules"]["duck"]["start"] = "c3"
        assert validate_spec(spec) == []

    def test_an_odd_max_plies(self) -> None:
        _refused(_small(max_plies=199), "is odd beside the duck")

    def test_the_shape(self) -> None:
        spec = _small()
        spec["rules"]["duck"] = {"piece": "X"}
        _refused(spec, "rules.duck.piece 'X' is not a declared piece")
        spec["rules"]["duck"] = {"piece": "D", "goes": "flies"}
        _refused(spec, "rules.duck.goes must be one of")
        spec["rules"]["duck"] = {"piece": "D", "start": "z9"}
        _refused(spec, "is not a square on this board")
        spec["rules"]["duck"] = "D"
        _refused(spec, "rules.duck must be an object")


class TestIdentity:
    def test_the_start_names_another_game(self) -> None:
        plain = _small()
        started = copy.deepcopy(plain)
        started["rules"]["duck"]["start"] = "c3"
        assert spec_hash(plain) != spec_hash(started)
        spelled = copy.deepcopy(plain)
        spelled["rules"]["duck"]["goes"] = "anywhere"
        assert spec_hash(spelled) == spec_hash(plain)


class TestTheWalk:
    @pytest.mark.parametrize("path", [CLASSIC, SHEPHERD], ids=["classic", "shepherd"])
    def test_a_seeded_random_walk(self, path: Path) -> None:
        """Twenty random games of 60 plies: every ply's key is the recomputed key, the core's
        perft(1) is the list (its make/unmake), and the halves alternate."""
        engine = Engine(_load(path))
        rng = random.Random(42)
        for _ in range(20):
            state = engine.initial_state()
            for ply in range(60):
                if engine.terminal(state) is not None:
                    break
                moves = engine.legal_moves(state)
                assert engine.duck_owed(state) is (ply % 2 == 1)
                assert all((decode_move(m)[2] == MOVE_DUCK) is (ply % 2 == 1) for m in moves)
                assert engine.perft(state, 1) == len(moves)
                state = engine.apply(state, rng.choice(moves))
                assert state.key == engine._compute_hash(state)


class TestThePlayers:
    """The searches play both halves, native and oracle alike, and a seed is a game."""

    @staticmethod
    def _gardner_duck() -> Engine:
        spec = _load(ROOT / "rulesets" / "gardner.json")
        spec["name"] = "gardner with a duck"
        spec["pieces"].append({"id": "D", "name": "duck", "value": 0, "moves": []})
        spec["rules"]["win"] = "royal_capture"
        spec["rules"]["duck"] = {"piece": "D"}
        spec["rules"]["max_plies"] = 60
        spec["rules"].pop("checkmate", None)
        return _engine(spec)

    @pytest.mark.parametrize(
        "agents", [("minimax:nodes=60", "greedy"), ("random", "minimax:nodes=60")],
        ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
    )
    def test_native_and_oracle_play_the_same_game(self, agents: tuple[str, str]) -> None:
        engine = self._gardner_duck()
        native = play_game(engine, agents, seed=5)
        in_python = play_game_python(engine, agents, seed=5)
        assert native.moves == in_python.moves
        assert native.nodes == in_python.nodes
        assert any(decode_move(m)[2] == MOVE_DUCK for m in native.moves)
        assert play_game(engine, agents, seed=5).moves == native.moves


class TestTheTally:
    def test_the_analysis_counts_what_the_duck_did(self) -> None:
        from chess_variant_sandbox.agents.evaluation import Evaluator
        from chess_variant_sandbox.tournament import analysis

        engine = Engine(_load(SHEPHERD))
        record = play_game(engine, ("random", "random"), 3)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        found = analysis._analyse_game(engine, evaluator, record, analysis._epsilon_for(engine))
        duck, wild = found.duck, found.wild
        assert duck is not None and wild is not None
        ducks = [m for m in record.moves if decode_move(m)[2] == MOVE_DUCK]
        assert duck["duck_moves"] == len(ducks) > 0 and duck["placements"] == 1
        assert len(duck["royal_distance"]) == len(ducks)
        assert all(-1 <= d <= 7 for d in duck["royal_distance"])
        # The duck is not a wild piece: two wolves at most.
        assert wild["most_standing"] == 2
        assert found.walked_plies == len(record.moves)
        # A duck move takes nothing, in the record's series too.
        assert all(
            not taken for m, taken in zip(record.moves, record.captures, strict=True)
            if decode_move(m)[2] == MOVE_DUCK
        )
        chess = Engine(_load(ROOT / "rulesets" / "chess.json"))
        plain = analysis._analyse_game(
            chess, Evaluator.for_engine(chess, mobility_weight=0.0),
            play_game(chess, ("random", "random"), 1), analysis._epsilon_for(chess),
        )
        assert "duck" not in plain.as_dict()
