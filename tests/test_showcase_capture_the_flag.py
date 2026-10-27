"""V1 of I2's showcase *capture the flag* (`docs/archive/plan-pieces-moves.md` §4.7): the mechanism,
without players.

Chess with the king replaced by an immobile royal flag (`moves: []`), won by taking it, with
stalemate a loss and no castling. Built, measured and retired by the call registered before
its first measured game (the index missed V3's 0.800), so the file is
`tests/_retired_capture_the_flag.json` (`retired_specs.RETIRED_PREVIEWS`); the mechanism it
shows still holds and is held here.

Nothing here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, and
Fairy-Stockfish (`fairy.verify`, the extinct model) is the declared second reading.
"""

from __future__ import annotations

import copy
import random
from functools import cache
from typing import Any

import pytest
from retired_specs import retired_preview

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.spec import spec_hash

HASH, TWIN_HASH, CHESS_HASH = "407af145c5d6", "c4e50ed4ee63", "4c1922ffae26"
KING_STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}


@cache
def _spec() -> dict[str, Any]:
    return retired_preview("capture-the-flag")


def _engine(**rules: Any) -> Engine:
    spec = copy.deepcopy(_spec())
    spec["rules"].update(rules)
    return Engine(spec)


def _flag_squares(engine: Engine, board: Any) -> set[int]:
    flag = engine.spec.piece_ids.index("F")
    return {s for s, v in enumerate(board) if v and engine.spec.type_of(v) == flag}


def test_it_is_the_registered_game_and_its_twin_is_the_control() -> None:
    spec = _spec()
    assert spec_hash(spec) == HASH
    assert spec["provenance"]["parents"] == [CHESS_HASH]
    flag = next(p for p in spec["pieces"] if p["id"] == "F")
    assert flag["moves"] == [] and flag["royal"] is True
    assert spec["rules"]["win"] == "royal_capture" and spec["rules"]["stalemate"] == "loss"
    assert "castling" not in spec["rules"]
    assert spec["appearance"]["icons"]["F"] == "crown"
    twin = copy.deepcopy(spec)
    next(p for p in twin["pieces"] if p["id"] == "F")["moves"] = [KING_STEP]
    assert spec_hash(twin) == TWIN_HASH


def _random_games_never_move_a_flag(games: int) -> None:
    engine = _engine()
    for seed in range(games):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ply in range(200):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            flags = _flag_squares(engine, state.board)
            assert len(flags) == 2
            assert not [m for m in moves if decode_move(m)[0] in flags]
            state = engine.apply(state, rng.choice(moves))


def test_random_games_never_move_a_flag() -> None:
    _random_games_never_move_a_flag(3)


@pytest.mark.slow
def test_twenty_random_games_never_move_a_flag() -> None:
    _random_games_never_move_a_flag(20)


# An undefended knight fork: White's knight on c7 hits the flag on e8 and the rook on a8, and
# nothing of Black's can take the knight or stand between (a leap is never blocked).
FORK = {"e1": ("F", 0), "c7": ("N", 0), "h2": ("P", 0),
        "e8": ("F", 1), "a8": ("R", 1), "h7": ("P", 1)}


def test_an_undefended_knight_fork_of_the_flag_is_lost_next_move() -> None:
    engine = _engine()
    state = engine.make_state(FORK, side_to_move=1, moved=list(FORK))
    assert engine.terminal(state) is None  # no check: Black plays on
    replies = engine.legal_moves(state)
    assert replies
    for reply in replies:
        after = engine.apply(state, reply)
        if engine.terminal(after) is not None:
            continue
        takes = [m for m in engine.legal_moves(after)
                 if engine.describe_move(after, m).endswith("xe8")]
        assert takes, engine.describe_move(state, reply)
        outcome = engine.terminal(engine.apply(after, takes[0]))
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "royal_captured"


def test_the_same_fork_under_checkmate_is_mate() -> None:
    engine = _engine(win="checkmate")
    state = engine.make_state(FORK, side_to_move=1, moved=list(FORK))
    outcome = engine.terminal(state)
    assert outcome is not None and outcome.winner == 0 and outcome.reason == "checkmate"


def test_a_side_reduced_to_its_flag_loses_by_stalemate() -> None:
    engine = _engine()
    state = engine.make_state({"e1": ("F", 0), "e8": ("F", 1), "a8": ("R", 1)},
                              side_to_move=0, moved=["a8"])
    outcome = engine.terminal(state)
    assert outcome is not None and outcome.winner == 1 and outcome.reason == "stalemate_loss"


class TestFairyStockfishAgrees:
    """The extinct model (`docs/archive/plan-pieces-moves.md` §4.4): legal lists after every move."""

    @staticmethod
    def _verify(**kwargs: Any) -> Any:
        pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")
        from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

        return verify(copy.deepcopy(_spec()), PyffishBackend(), "", **kwargs)

    def test_three_random_games_agree(self) -> None:
        result = self._verify(depth=1, games=3, plies=50)
        assert result.ok, result.reasons

    @pytest.mark.slow
    def test_twenty_random_games_and_depth_three_agree(self) -> None:
        result = self._verify(depth=3, games=20, plies=300)
        assert result.ok, result.reasons
