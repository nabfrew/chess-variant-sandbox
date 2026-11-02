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

    def test_the_fuzz_route_oracles_do_not_read_the_duck_as_a_wild_piece(self) -> None:
        """Q1061 (fuzz seed 108): the duck stands in the third owner's block beside the wolves,
        on no route and worth no bounty; the oracles once took it for a wolf off its route."""
        from chess_variant_sandbox import fuzz

        engine = Engine(_load(SHEPHERD))
        state = engine.initial_state()
        state = _play(engine, state, "Pe2-e3")
        state = _play(engine, state, "the duck lands on c6")
        assert engine.duck_square(state) == _sq(engine, "c6")
        fuzz.every_npc_stands_on_its_own_route(engine, state)
        fuzz.the_points_are_the_bounties_of_the_npcs_that_are_gone(engine, state)

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

    def test_beside_portals_since_sr3d(self) -> None:
        """K2 is lifted (SR3d, `tests/test_duck_beside_portals.py`): a momentum pair and a fold
        pair both validate beside the duck. The duck may start in a fold (floor), never on a
        momentum end (a piece)."""
        for kind in (None, "fold"):
            group: dict[str, Any] = {"orange": "b3", "blue": "e4"}
            if kind:
                group["kind"] = kind
            spec = _small(portals=[group])
            assert validate_spec(spec) == [], kind
            assert "K2" not in refused_ids(validate_spec(spec))
            spec["rules"]["duck"]["start"] = "b3"
            if kind:
                assert validate_spec(spec) == []
            else:
                _refused(spec, "rules.duck.start b3 is a square the opening does not leave empty")

    def test_beside_a_blast_a_strike_or_contacts_since_sr3e(self) -> None:
        """K3 is lifted (SR3e, `tests/test_duck_untouched.py`): the duck is untouched by all
        three, so each validates beside it; their lists may not name it."""
        for rules in (
            {"explode_on_capture": {"radius": 1}},
            {"strike": {"effect": "capture", "position": "stay", "immune": ["K"]}},
            {"contacts": {"links": [{"mover": "R", "onto": "P", "onto_to": "R"}],
                          "position": "stay", "immune": ["K"]}},
        ):
            assert validate_spec(_small(**rules)) == [], rules
        _refused(_small(explode_on_capture={"radius": 1, "immune": ["D"]}),
                 "rules.duck.piece 'D' is named in rules.explode_on_capture.immune")

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

    # WK1 (`docs/reference/wakwak.md` §3 A, B): quiescence through the owed duck half and the
    # duck's own history, each arm alone and both, against the oracle move for move and node
    # for node; each moves the search on a duck game.
    WK1_ARMS = ("duck_quiesce=1", "duck_history=1", "duck_quiesce=1,duck_history=1")

    @pytest.mark.parametrize("arm", WK1_ARMS)
    def test_the_wk1_arms_native_and_oracle_alike(self, arm: str) -> None:
        engine = self._gardner_duck()
        agents = (f"minimax:nodes=200,{arm}", "minimax:nodes=200")
        native = play_game(engine, agents, seed=7)
        in_python = play_game_python(engine, agents, seed=7)
        assert native.moves == in_python.moves
        assert native.nodes == in_python.nodes

    @pytest.mark.slow
    @pytest.mark.parametrize("arm", WK1_ARMS)
    def test_the_wk1_arms_native_and_oracle_alike_deeper(self, arm: str) -> None:
        """At 800 nodes, where the duck's own history moves a game (at 200 it rarely does:
        within one search the duck stands on one square, so `from * n + to` already read
        nearly as `[side][to]` does)."""
        engine = self._gardner_duck()
        agents = (f"minimax:nodes=800,{arm}", "minimax:nodes=800")
        native = play_game(engine, agents, seed=2)
        in_python = play_game_python(engine, agents, seed=2)
        assert native.moves == in_python.moves
        assert native.nodes == in_python.nodes

    @pytest.mark.parametrize("arm", WK1_ARMS)
    def test_each_wk1_arm_moves_a_duck_game(self, arm: str) -> None:
        """Against the player WK1 was graded against (before WK2's orderings)."""
        engine = self._gardner_duck()
        control = "minimax:nodes=800,duck_refute=0,duck_front=0,duck_mobility=0"
        armed = play_game(engine, (f"{control},{arm}", control), seed=2)
        assert armed.moves != play_game(engine, (control,) * 2, seed=2).moves

    # WK1b (`wakwak.md` §6.6): where the duck's type sees, `duck_quiesce`'s candidates also
    # hold the squares from which its sight would bind an open attack's end.
    @pytest.mark.parametrize(
        ("nodes", "seeds"),
        [(60, (1,)), pytest.param(200, (1, 4), marks=pytest.mark.slow)],
        ids=["60", "200"],
    )
    def test_the_seeing_duck_arm_native_and_oracle_alike(
        self, nodes: int, seeds: tuple[int, ...]
    ) -> None:
        engine = Engine(_load(SHEPHERD))
        agents = (f"minimax:nodes={nodes},duck_quiesce=1", f"minimax:nodes={nodes}")
        for seed in seeds:
            native = play_game(engine, agents, seed=seed)
            in_python = play_game_python(engine, agents, seed=seed)
            assert native.moves == in_python.moves
            assert native.nodes == in_python.nodes

    def test_a_seeing_duck_widens_the_candidates(self) -> None:
        """Along a game, the seeing duck's candidates hold its blind twin's, and somewhere
        more: a square beside a threatened piece, on no attack path."""
        from chess_variant_sandbox.agents.minimax import DUCK_QUIESCE, MinimaxAgent

        spec = _load(SHEPHERD)
        blind_spec = copy.deepcopy(spec)
        for piece in blind_spec["pieces"]:
            piece.pop("sight", None)
        seeing, blind = Engine(spec), Engine(blind_spec)
        sees = MinimaxAgent(seeing, nodes=10, repairs=DUCK_QUIESCE)
        sees_not = MinimaxAgent(blind, nodes=10, repairs=DUCK_QUIESCE)
        record = play_game(seeing, ("random", "random"), 3)
        state, wider = seeing.initial_state(), 0
        for move in record.moves:
            moves = seeing.legal_moves(state)
            if decode_move(moves[0])[2] == MOVE_DUCK:
                mine = sees._duck_blocks(state, moves[0])
                theirs = sees_not._duck_blocks(state, moves[0])
                assert theirs <= mine
                wider += mine != theirs
            state = seeing.apply(state, move)
        assert wider > 0

    # WK2 (`wakwak.md` §7): the duck half's two orderings, each alone and both, against the
    # oracle on the plain duck game and on the seeing duck's; each moves a duck game.
    # Spelled whole: both are on by default since WK2 adopted them, so the control below is
    # the player before WK2.
    WK2_ARMS = (
        "duck_refute=1,duck_front=0", "duck_refute=0,duck_front=1", "duck_refute=1,duck_front=1",
    )
    BEFORE_WK2 = "duck_refute=0,duck_front=0,duck_mobility=0"

    @pytest.mark.parametrize(
        ("game", "arm"),
        [
            *(("gardner", arm) for arm in WK2_ARMS),
            ("shepherd", WK2_ARMS[2]),
            *(pytest.param("shepherd", arm, marks=pytest.mark.slow) for arm in WK2_ARMS[:2]),
        ],
    )
    def test_the_wk2_arms_native_and_oracle_alike(self, game: str, arm: str) -> None:
        engine = self._gardner_duck() if game == "gardner" else Engine(_load(SHEPHERD))
        nodes = 200 if game == "gardner" else 60
        agents = (f"minimax:nodes={nodes},{arm}", f"minimax:nodes={nodes},{self.BEFORE_WK2}")
        native = play_game(engine, agents, seed=7)
        in_python = play_game_python(engine, agents, seed=7)
        assert native.moves == in_python.moves
        assert native.nodes == in_python.nodes

    @pytest.mark.parametrize("arm", WK2_ARMS)
    def test_each_wk2_arm_reorders_a_duck_search(self, arm: str) -> None:
        """Each arm makes a depth-4 search (two whole turns) cheaper at a position of the
        gardner duck game, with the same value. A duck half cuts off only below a finite
        beta, so from ply 3, and a depth-2 search's duck halves never do; that is why
        ordering them moves a *game* rarely at the duel's budgets (`wakwak.md` §7)."""
        from chess_variant_sandbox.agents.minimax import MinimaxAgent, repairs_of

        engine = self._gardner_duck()
        spelled = dict(option.split("=") for option in arm.split(","))
        repairs = repairs_of({option: int(value) for option, value in spelled.items()})
        before = repairs_of({"duck_refute": 0, "duck_front": 0})
        record = play_game(engine, ("random", "random"), 3)
        state = engine.initial_state()
        for move in record.moves[:2]:
            state = engine.apply(state, move)
        costs, values = [], []
        for bits in (repairs, before):
            agent = MinimaxAgent(engine, nodes=10**7, max_depth=4, repairs=bits)
            agent._begin_search()
            values.append(agent._negamax(state, 4, 0, -float("inf"), float("inf")))
            costs.append(agent._nodes)
        assert values[0] == values[1]
        assert costs[0] < costs[1]

    def test_a_refutation_names_its_path(self) -> None:
        """`_reply_blocks` is a reply's target and the paths of the opponent's sources onto it
        from its origin by the piece there: a slide's crossed squares, a leap's nothing; and a
        drop's target alone."""
        from chess_variant_sandbox.agents.minimax import MinimaxAgent
        from chess_variant_sandbox.engine.moves import MOVE_DROP, encode_move

        engine = self._gardner_duck()
        agent = MinimaxAgent(engine, nodes=10)
        state = engine.initial_state()
        spec = engine.spec
        seen_path = seen_leap = False
        for target in range(spec.n_squares):
            for source in spec.attack_sources[1][target]:
                origin, value, prefix, window = source[:4]
                if state.board[origin] != value:
                    continue
                path = set(prefix) | set(window)
                assert agent._reply_blocks(state, encode_move(origin, target)) >= path | {target}
                seen_path |= bool(path)
                seen_leap |= not path
        assert seen_path and seen_leap
        assert agent._reply_blocks(state, encode_move(3, 3, MOVE_DROP)) == {3}

    # WK3 (`wakwak.md` §3 E, §8): a duck-half leaf hands the objective its turn's start's
    # move count, not the duck's squares.
    WK3_ARM = "duck_mobility=1"
    BEFORE_WK3 = "duck_mobility=0"

    @pytest.mark.parametrize("arm", (*WK1_ARMS, *WK2_ARMS, BEFORE_WK2, WK3_ARM, BEFORE_WK3))
    def test_the_wk1_arms_are_inert_without_a_duck(self, arm: str) -> None:
        engine = _engine(_load(ROOT / "rulesets" / "gardner.json"))
        plain = play_game(engine, ("minimax:nodes=200",) * 2, seed=7)
        armed = play_game(engine, (f"minimax:nodes=200,{arm}",) * 2, seed=7)
        assert armed.moves == plain.moves and armed.nodes == plain.nodes

    def test_turn_start_mobility_is_inert_in_a_long_turn(self) -> None:
        """A long turn (`rules.turn`) also moves the same side twice, but its mid-turn list is
        piece moves, so WK3 leaves it alone: the bit reads only a duck half's list."""
        engine = Engine(json.loads((ROOT / "tests" / "_command_turns.json").read_text("utf-8")))
        for seed in (1, 7):
            plain = play_game(engine, (f"minimax:nodes=300,{self.BEFORE_WK3}",) * 2, seed=seed)
            armed = play_game(engine, ("minimax:nodes=300",) * 2, seed=seed)
            assert armed.moves == plain.moves and armed.nodes == plain.nodes

    @pytest.mark.parametrize(
        "game", ["gardner", pytest.param("shepherd", marks=pytest.mark.slow)]
    )
    def test_the_wk3_arm_native_and_oracle_alike(self, game: str) -> None:
        engine = self._gardner_duck() if game == "gardner" else Engine(_load(SHEPHERD))
        nodes = 200 if game == "gardner" else 60
        agents = (f"minimax:nodes={nodes}", f"minimax:nodes={nodes},{self.BEFORE_WK3}")
        native = play_game(engine, agents, seed=7)
        in_python = play_game_python(engine, agents, seed=7)
        assert native.moves == in_python.moves
        assert native.nodes == in_python.nodes

    def test_the_wk3_arm_moves_a_duck_game(self) -> None:
        engine = self._gardner_duck()
        control = f"minimax:nodes=800,{self.BEFORE_WK3}"
        armed = play_game(engine, ("minimax:nodes=800", control), seed=2)
        assert armed.moves != play_game(engine, (control,) * 2, seed=2).moves

    def test_a_duck_half_leaf_reads_the_turns_start(self) -> None:
        """At depth 1 from a piece half every leaf is a duck half (no loud duck, no repair
        that chases one), so the search's value is the best child's evaluation: under the
        bit with the root's own count, every sibling alike; without it with the duck's."""
        from chess_variant_sandbox.agents.minimax import DUCK_MOBILITY, MinimaxAgent

        engine = self._gardner_duck()
        record = play_game(engine, ("random", "random"), 3)
        state = engine.initial_state()
        for move in record.moves[:4]:
            state = engine.apply(state, move)
        moves = engine.legal_moves(state)
        assert decode_move(moves[0])[2] != MOVE_DUCK
        side = state.side_to_move
        for bits in (DUCK_MOBILITY, 0):
            agent = MinimaxAgent(engine, nodes=10**7, max_depth=1, repairs=bits)
            agent._begin_search()
            value = agent._negamax(state, 1, 0, -float("inf"), float("inf"))
            leaves = []
            for move in moves:
                child = engine.apply(state, move)
                count = len(moves) if bits else len(engine.legal_moves(child))
                leaves.append(agent.evaluator.evaluate(engine, child, side, count))
            assert value == max(leaves)


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
