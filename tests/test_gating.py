"""Q1059a: gating (`drops.gating`), Seirawan's way of bringing a held piece into play.

A held piece enters only on the square a piece of its side leaves on its first move from
its setup square, as part of that move; a castle may gate on the king's square or the
rook's. The test spec is `tests/_gating_seirawan.json`, chess with a hawk (B+N) and an
elephant (R+N) in each hand, which is Seirawan chess's rules; its perft numbers are the ones
Fairy-Stockfish publishes for its built-in `seirawan` (`tests/perft.sh` in the
Fairy-Stockfish repository: startpos 4 = 782,599 and 5 = 27,639,803; the middle-game FEN
below, 4 = 890,467), which this core reaches independently.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    move_gate,
    move_gate_rook,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import validate_spec

HERE = Path(__file__).parent
SPEC_PATH = HERE / "_gating_seirawan.json"


def _spec() -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return spec


def _from_fen(engine: Engine, fen: str) -> GameState:
    """A test's reading of a Seirawan FEN: board, pocket, side, rights and gates.

    Test-only, and only as wide as the position below needs: the castling field's ``KQkq``
    and gate files become `unmoved` bits on the squares they name.
    """
    spec = engine.spec
    ids = spec.piece_ids
    board_field, side, rights = fen.split()[:3]
    placement, pocket = board_field.rstrip("]").split("[")
    board = bytearray(spec.n_squares)
    for row, text in enumerate(placement.split("/")):
        rank = spec.height - 1 - row
        file = 0
        for char in text:
            if char.isdigit():
                file += int(char)
                continue
            player = 0 if char.isupper() else 1
            board[rank * spec.width + file] = spec.encode_piece(player, ids.index(char.upper()))
            file += 1
    hands = [[0] * spec.n_types, [0] * spec.n_types]
    for char in pocket:
        hands[0 if char.isupper() else 1][ids.index(char.upper())] += 1
    unmoved = 0
    for char in rights:
        player = 0 if char.isupper() else 1
        rank = 0 if player == 0 else spec.height - 1
        files = {"K": [4, 7], "Q": [4, 0]}.get(char.upper(), [ord(char.upper()) - ord("A")])
        for file in files:
            unmoved |= 1 << (rank * spec.width + file)
    state = engine.initial_state()
    return replace(
        state,
        board=bytes(board),
        side_to_move=0 if side == "w" else 1,
        unmoved=unmoved,
        hands=(tuple(hands[0]), tuple(hands[1])),
        repetition=(),
    )


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(_spec())


class TestPerft:
    """Fairy-Stockfish's published `seirawan` numbers, reached by this core alone."""

    def test_the_opening_shallow(self, engine: Engine) -> None:
        state = engine.initial_state()
        assert [engine.perft(state, depth) for depth in (1, 2, 3)] == [28, 784, 24830]

    @pytest.mark.slow
    def test_the_opening_deep(self, engine: Engine) -> None:
        state = engine.initial_state()
        assert engine.perft(state, 4) == 782599
        assert engine.perft(state, 5) == 27639803

    @pytest.mark.slow
    def test_a_middle_game_with_one_gate_left(self, engine: Engine) -> None:
        state = _from_fen(
            engine, "reb1k2r/ppppqppp/2nb1n2/4p3/4P3/N1P2N2/PB1PQPPP/RE2KBHR[h] b KQkqc - 3 7"
        )
        assert engine.perft(state, 4) == 890467


class TestTheMoves:
    def test_every_back_rank_first_move_offers_each_held_piece(self, engine: Engine) -> None:
        state = engine.initial_state()
        moves = engine.legal_moves(state)
        gated = [move for move in moves if move_gate(move)]
        # Two knights, two squares each, two held pieces.
        assert len(gated) == 8
        texts = {engine.describe_move(state, move) for move in gated}
        assert "Ng1-f3/H" in texts and "Nb1-c3/E" in texts

    def test_a_gate_spends_the_hand_and_fills_the_square(self, engine: Engine) -> None:
        state = engine.initial_state()
        move = next(
            m for m in engine.legal_moves(state)
            if engine.describe_move(state, m) == "Ng1-f3/E"
        )
        after = engine.apply(state, move)
        spec = engine.spec
        g1 = 6
        assert spec.type_of(after.board[g1]) == spec.piece_ids.index("E")
        assert after.hands[0][spec.piece_ids.index("E")] == 0
        assert after.hands[0][spec.piece_ids.index("H")] == 1
        # The gated piece has moved, as far as rights go: it opens no gate of its own.
        assert not after.unmoved >> g1 & 1
        # Not progress: the clock reads the knight's move (Seirawan's fifty-move rule).
        assert after.halfmove == 1

    def test_a_castle_gates_on_either_vacated_square(self, engine: Engine) -> None:
        state = _from_fen(
            engine, "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R[HEhe] w KQkq - 0 1"
        )
        castles = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_CASTLE]
        texts = sorted(engine.describe_move(state, m) for m in castles)
        assert "Ke1-g1 (castle)/E@h1" in texts and "Ke1-g1 (castle)/H" in texts
        # Two castles, each plain or gating one of two pieces on one of two squares.
        assert len(castles) == 2 * (1 + 2 * 2)
        rook_gate = next(m for m in castles if move_gate_rook(m) and move_gate(m))
        assert engine.describe_move(state, rook_gate).endswith("@h1") or engine.describe_move(
            state, rook_gate
        ).endswith("@a1")

    def test_a_pinned_piece_may_leave_when_one_enters_behind_it(self, engine: Engine) -> None:
        """A gated piece fills the square its piece left, so a pin along that square holds."""
        # The bishop on d1 is pinned along the first rank by the rook on a1.
        state = _from_fen(engine, "4k3/8/8/8/8/8/8/r2BK3[H] w D - 0 1")
        texts = {engine.describe_move(state, m) for m in engine.legal_moves(state)}
        assert "Bd1-e2/H" in texts
        assert "Bd1-e2" not in texts

    def test_a_gate_on_an_emptied_hand_or_a_moved_piece_is_not_offered(
        self, engine: Engine
    ) -> None:
        state = _from_fen(engine, "4k3/8/8/8/8/8/8/4KN2[] w F - 0 1")
        assert not any(move_gate(m) for m in engine.legal_moves(state))
        state = _from_fen(engine, "4k3/8/8/8/8/8/8/4KN2[H] w - - 0 1")
        assert not any(move_gate(m) for m in engine.legal_moves(state))

    def test_the_hash_walk(self, engine: Engine) -> None:
        """A game played ply by ply: the incremental key equals the one computed afresh."""
        import random

        rng = random.Random(1059)
        for _ in range(4):
            state = engine.initial_state()
            for _ in range(80):
                if engine.terminal(state) is not None:
                    break
                moves = engine.legal_moves(state)
                gated = [m for m in moves if move_gate(m)]
                state = engine.apply(state, rng.choice(gated or moves))
                assert state.key == engine._compute_hash(state)


class TestTheRule:
    def test_it_is_inert_where_absent(self) -> None:
        chess = json.loads((HERE.parent / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        assert build_tables(compile_spec(chess))["drops_gates"] == []
        crazyhouse = json.loads(
            (HERE.parent / "rulesets" / "crazyhouse.json").read_text(encoding="utf-8")
        )
        assert build_tables(compile_spec(crazyhouse))["drops_gates"] == []

    def test_the_tables_name_each_sides_back_rank(self) -> None:
        compiled = compile_spec(_spec())
        assert compiled.drops is not None and compiled.drops.gates is not None
        assert compiled.drops.gates[0] == frozenset(range(8))
        assert compiled.drops.gates[1] == frozenset(range(56, 64))

    def test_squares_say_the_same_as_ranks(self) -> None:
        spec = _spec()
        spec["rules"]["drops"]["gating"] = {"squares": [f"{f}1" for f in "abcdefgh"]}
        assert not validate_spec(spec)
        assert compile_spec(spec).drops.gates == compile_spec(_spec()).drops.gates  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("edit", "words"),
        [
            (lambda d: d.pop("from_captures"), "fills from captures"),
            (lambda d: d.pop("stock"), "nothing in rules.drops.stock"),
            (lambda d: d.update(placement=True), "rules.drops.placement"),
            (lambda d: d.update(no_mate=["P"]), "rules.drops.no_mate"),
            (lambda d: d.update(gating={"ranks": [3]}), "no square a piece of the first"),
            (lambda d: d.update(gating={"ranks": [0], "squares": ["a1"]}), "exactly one"),
            (lambda d: d.update(gating={"ranks": [9]}), "not a rank of this board"),
            (lambda d: d.update(gating="back_rank"), "must be an object"),
        ],
    )
    def test_what_it_refuses_in_the_hand(self, edit: Any, words: str) -> None:
        spec = _spec()
        edit(spec["rules"]["drops"])
        errors = validate_spec(spec)
        assert any(words in error for error in errors), errors

    def test_what_it_refuses_beside_it(self) -> None:
        spec = _spec()
        spec["rules"]["explode_on_capture"] = {"radius": 1}
        assert any(
            "rules.drops.gating beside rules.explode_on_capture" in error
            for error in validate_spec(spec)
        )

    def test_the_spec_is_valid(self) -> None:
        assert validate_spec(_spec()) == []
        assert validate_spec(copy.deepcopy(_spec())) == []

    def test_the_library_game_is_this_spec(self) -> None:
        """Q1059b: `rulesets/seirawan.json` is gating's showcase and this file's game.

        The two differ in `name`, `description` and `provenance` alone, all outside the
        mechanics hash, so every number above is the library game's too.
        """
        library = json.loads(
            (HERE.parent / "rulesets" / "seirawan.json").read_text(encoding="utf-8")
        )
        assert spec_hash(library) == spec_hash(_spec())
