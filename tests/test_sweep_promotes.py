"""SR8b (`docs/plan-schema-next.md` §2.6 row W7): a sweeper that promotes.

The reading, as ruled (2026-10-27): a promotion is not an on-capture rule, so the sweep switch
has nothing to choose; the sweeper promotes by its landing square, as any of its moves does
(under `on_leaving` by its origin too), and the move word carries both the promotion and the
sweep: a `MOVE_SWEEP` with the promoted type plus one in the promotion field
(`CompiledSpec.sweep_promotion`). One move clears the line and arrives promoted.

Settled here (the row left them unstated): over an empty line the landing is the slide's
`MOVE_PROMOTION`, as a sweep over nothing is the slide's `MOVE_NORMAL`; an optional promotion
offers the plain sweep too; two sweeps to one square that sweep the same pieces but promote to
different types are two moves; a commanded sweeper never promotes (a landing that would have to
is not offered); a sweep opens no en passant window. **Beside folds a promoting sweeper stays
refused** (SR6e's note): a sweep through a fold carries its fold step in the same field
(`moves.move_through`), so the two never share a game, and the field reads by the game.

`tests/sweep_oracle.py` (its promotion on the last rank, for any move) holds the core to it,
move by move and by perft. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
import sweep_oracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_EN_PASSANT,
    MOVE_NORMAL,
    MOVE_PROMOTION,
    MOVE_SWEEP,
    decode_move,
    move_from,
    move_promotion,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import NO_SQUARE, GameState
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_rules

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
SWEEP = {"kind": "sweep", "vectors": [[0, 1], [1, 0]], "symmetry": "lateral"}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(**promotion: Any) -> dict[str, Any]:
    """A sweeper X (forward and sideways) that promotes on the last rank to a queen-ish Q or a
    knight-ish N, kings K, and pawn-like blockers P."""
    return {
        "name": "the promoting sweep (SR8b test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "X", "name": "sweeper", "value": 3, "moves": [copy.deepcopy(SWEEP)]},
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [KING]},
            {"id": "P", "name": "pawn", "value": 1,
             "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]},
            {"id": "Q", "name": "queen", "value": 9,
             "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
            {"id": "N", "name": "knight", "value": 3,
             "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "X", "squares": ["b1", "e1"]},
            {"piece": "K", "squares": ["c1"]},
            {"piece": "P", "squares": ["a2", "b2", "d2", "e2", "f2"]},
        ]},
        "rules": {
            "win": "checkmate", "stalemate": "draw", "max_plies": 80,
            "promotion": {"piece": "X", "rank": "last", "to": ["Q", "N"], **promotion},
        },
    }


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


KINGS = {"a1": ("K", 0), "f6": ("K", 1)}


def _state(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _texts_from(engine: Engine, state: GameState, origin: str) -> list[str]:
    return sorted(
        engine.describe_move(state, m) for m in engine.legal_moves(state)
        if move_from(m) == _sq(origin)
    )


def _at(engine: Engine, state: GameState, square: str) -> str | None:
    value = state.board[_sq(square)]
    return None if not value else engine.spec.piece_ids[engine.spec.type_of(value)]


class TestTheReading:
    def test_the_refusal_went_but_beside_folds(self) -> None:
        assert validate_spec(_spec()) == []
        folded = _spec()
        folded["rules"]["portals"] = [{"orange": "c3", "blue": "d4", "kind": "fold"}]
        errors = validate_spec(folded)
        assert any("fold portal" in e and "'X'" in e for e in errors), errors
        momentum = _spec()
        momentum["rules"]["portals"] = [{"orange": "c3", "blue": "d4"}]
        assert validate_spec(momentum) == []

    def test_the_boundary_refuses_the_pair_the_validator_does(self) -> None:
        """Past the validator (a fold game's tables handed a promoting sweeper's promotion),
        `bridge.rust_supports` refuses it: the field cannot hold both."""
        folded = _spec()
        folded["rules"]["portals"] = [{"orange": "c3", "blue": "d4", "kind": "fold"}]
        del folded["rules"]["promotion"]
        compiled = compile_spec(folded)
        assert rust_supports(compiled) is None
        forced = replace(compiled, promotions=compile_spec(_spec()).promotions)
        why = rust_supports(forced)
        assert why is not None and "promoting piece beside folds" in why

    def test_a_sweep_onto_its_zone_clears_and_promotes_in_one_move(self) -> None:
        """X on c3 sweeps c4 and c5 and lands on c6: one word per target, each a sweep."""
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "c3": ("X", 0), "c4": ("P", 1), "c5": ("P", 0)})
        texts = _texts_from(engine, state, "c3")
        assert "Xc3xc6 (sweeps c4, c5)=Q" in texts and "Xc3xc6 (sweeps c4, c5)=N" in texts
        assert not any(t.startswith("Xc3xc6") and "=" not in t for t in texts)  # mandatory
        move = engine.find_move(state, "Xc3xc6 (sweeps c4, c5)=Q")
        assert move_type(move) == MOVE_SWEEP and move_to(move) == _sq("c6")
        assert engine.spec.sweep_promotion(move) == engine.spec.piece_ids.index("Q")
        after = engine.apply(state, move)
        assert _at(engine, after, "c6") == "Q"
        assert _at(engine, after, "c3") is None
        assert _at(engine, after, "c4") is None and _at(engine, after, "c5") is None

    def test_over_an_empty_line_it_is_the_slide_s_promotion(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "c3": ("X", 0)})
        moves = [m for m in engine.legal_moves(state) if move_to(m) == _sq("c6")]
        assert sorted(move_type(m) for m in moves) == [MOVE_PROMOTION, MOVE_PROMOTION]
        assert {engine.describe_move(state, m) for m in moves} == {"Xc3-c6=Q", "Xc3-c6=N"}

    def test_short_of_its_zone_it_sweeps_as_before(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "c3": ("X", 0), "c4": ("P", 1)})
        move = engine.find_move(state, "Xc3xc5 (sweeps c4)")
        assert move_type(move) == MOVE_SWEEP and move_promotion(move) == 0
        assert _at(engine, engine.apply(state, move), "c5") == "X"

    def test_an_optional_promotion_offers_the_plain_sweep_too(self) -> None:
        engine = _engine(_spec(mandatory=False))
        state = _state(engine, {**KINGS, "c3": ("X", 0), "c4": ("P", 1)})
        texts = [t for t in _texts_from(engine, state, "c3") if t.startswith("Xc3xc6")]
        assert texts == [
            "Xc3xc6 (sweeps c4)", "Xc3xc6 (sweeps c4)=N", "Xc3xc6 (sweeps c4)=Q",
        ]
        plain = engine.find_move(state, "Xc3xc6 (sweeps c4)")
        assert move_type(plain) == MOVE_SWEEP and engine.spec.sweep_promotion(plain) is None

    def test_a_capture_on_its_landing_promotes_too(self) -> None:
        engine = _engine(_spec())
        state = _state(engine, {**KINGS, "c3": ("X", 0), "c4": ("P", 0), "c6": ("P", 1)})
        texts = _texts_from(engine, state, "c3")
        assert "Xc3xc6 (sweeps c4)=N" in texts
        after = engine.apply(state, engine.find_move(state, "Xc3xc6 (sweeps c4)=N"))
        assert _at(engine, after, "c6") == "N" and _at(engine, after, "c4") is None

    def test_on_leaving_a_sweep_out_of_the_zone_promotes(self) -> None:
        """Under `on_leaving` a sweep that starts in the zone promotes wherever it lands."""
        engine = _engine(_spec(on_leaving=True))
        state = _state(engine, {**KINGS, "b6": ("X", 0), "c6": ("P", 1)})
        texts = _texts_from(engine, state, "b6")
        assert "Xb6xd6 (sweeps c6)=Q" in texts
        assert not any(t.startswith("Xb6xd6") and "=" not in t for t in texts)

    def test_the_text_and_the_word_round_trip(self) -> None:
        engine = _engine(_spec(mandatory=False))
        state = _state(engine, {**KINGS, "c3": ("X", 0), "c4": ("P", 1), "d5": ("X", 0),
                                "e5": ("P", 0)})
        moves = engine.legal_moves(state)
        texts = [engine.describe_move(state, m) for m in moves]
        assert len(set(texts)) == len(texts)
        for move, text in zip(moves, texts, strict=True):
            assert engine.find_move(state, text) == move

    def test_the_description_says_it(self) -> None:
        assert any("promotes by the square it lands on" in line for line in describe_rules(_spec()))


def _walk(engine: Engine, state: GameState, depth: int) -> int:
    """perft by `apply`, checking at every node that `apply` leaves its argument untouched and
    the incremental key is the one computed afresh."""
    moves = engine.legal_moves(state)
    if depth == 1:
        return len(moves)
    total = 0
    for move in moves:
        before = (bytes(state.board), state.key)
        child = engine.apply(state, move)
        assert (bytes(state.board), state.key) == before
        assert child.key == engine._compute_hash(child), engine.describe_move(state, move)
        total += _walk(engine, child, depth - 1)
    return total


class TestUndo:
    def test_the_core_s_make_and_unmake_agree_with_apply(self) -> None:
        """`perft` makes and unmakes in the core; the walk by `apply` copies. Equal counts
        from a position full of promoting sweeps, and a key that matches at every node."""
        engine = _engine(_spec(mandatory=False))
        state = _state(engine, {**KINGS, "c3": ("X", 0), "c4": ("P", 1), "c5": ("P", 0),
                                "d4": ("X", 1), "d3": ("P", 0), "e5": ("X", 0)})
        assert engine.perft(state, 3) == _walk(engine, state, 3)


# ------------------------------------------------------------------- the oracle


def _oracle_position(state: GameState, n_types: int) -> sweep_oracle.Position:
    ep = state.ep_target if state.ep_target != NO_SQUARE else None
    victim = state.ep_victim if state.ep_victim != NO_SQUARE else None
    return sweep_oracle.position_of(
        state.board, state.side_to_move, state.unmoved, n_types, ep, victim
    )


def _engine_moves(engine: Engine, state: GameState) -> Counter[tuple[int, int, int]]:
    out: Counter[tuple[int, int, int]] = Counter()
    for move in engine.legal_moves(state):
        origin, target, kind, field = decode_move(move)
        if kind == MOVE_PROMOTION:
            promoted = field
        elif kind == MOVE_SWEEP:
            swept_to = engine.spec.sweep_promotion(move)
            promoted = -1 if swept_to is None else swept_to
        else:
            assert kind in (MOVE_NORMAL, MOVE_EN_PASSANT), kind
            promoted = -1
        out[(origin, target, promoted)] += 1
    return out


def _oracle_moves(oracle: sweep_oracle.Oracle, pos: sweep_oracle.Position) -> Counter[tuple[int, int, int]]:
    return Counter(
        (origin, target, promoted) for (origin, target, promoted, _), _taken in oracle.legal_taken(pos)
    )


@pytest.mark.parametrize("seed", range(4))
def test_the_core_agrees_with_the_oracle_over_random_games(seed: int) -> None:
    spec = _spec()
    engine = _engine(spec)
    oracle = sweep_oracle.Oracle(spec)
    n_types = len(spec["pieces"])
    rng = random.Random(seed)
    state = engine.initial_state()
    promoted = 0
    for _ in range(60):
        moves = engine.legal_moves(state)
        pos = _oracle_position(state, n_types)
        assert _engine_moves(engine, state) == _oracle_moves(oracle, pos), (seed, state.ply)
        if not moves or engine.terminal(state, moves) is not None:
            break
        # Lean towards promoting sweeps, so the walk meets them.
        sweeps = [m for m in moves if engine.spec.sweep_promotion(m) is not None]
        move = rng.choice(sweeps) if sweeps and rng.random() < 0.7 else rng.choice(moves)
        promoted += engine.spec.sweep_promotion(move) is not None
        state = engine.apply(state, move)


def test_the_random_games_meet_a_promoting_sweep() -> None:
    """The walk above is not vacuous: some seed plays a promoting sweep."""
    spec = _spec()
    engine = _engine(spec)
    met = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(60):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            sweeps = [m for m in moves if engine.spec.sweep_promotion(m) is not None]
            met += bool(sweeps)
            move = rng.choice(sweeps) if sweeps and rng.random() < 0.7 else rng.choice(moves)
            state = engine.apply(state, move)
    assert met


def test_perft_against_the_oracle() -> None:
    spec = _spec()
    engine = _engine(spec)
    oracle = sweep_oracle.Oracle(spec)
    state = _state(engine, {**KINGS, "c3": ("X", 0), "c4": ("P", 1), "c5": ("P", 0),
                            "d4": ("X", 1), "d3": ("P", 0), "e5": ("X", 0)})
    pos = _oracle_position(state, len(spec["pieces"]))
    assert engine.perft(state, 2) == oracle.perft(pos, 2)


class TestACommandedSweeperNeverPromotes:
    def test_a_landing_that_would_have_to_promote_is_not_offered(self) -> None:
        """The second player commands the first player's X (a freezing-free command sight):
        its sweeps into the first player's zone would have to promote, so none is offered;
        short of the zone it moves as a commanded sweep does."""
        spec = _spec()
        spec["pieces"][1]["sight"] = [{
            "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
            "effect": "command", "whose": "enemy",
        }]
        engine = _engine(spec)
        state = _state(engine, {"a1": ("K", 0), "d3": ("K", 1), "c3": ("X", 0)}, side=1)
        commanded = [
            engine.describe_move(state, m) for m in engine.legal_moves(state)
            if move_from(m) == _sq("c3")
        ]
        assert commanded, "the command reaches X"
        assert any("c5" in text for text in commanded), commanded
        assert not any("c6" in text or "=" in text for text in commanded), commanded


def test_a_sweeping_en_passant_piece_opens_a_window_only_by_its_slide() -> None:
    """A promoting piece may now sweep, so it may be the en passant piece. Its double step
    over an empty line opens the window (the slide's word, `_ep_passes`); a sweep over a piece
    to the same square takes, and opens none."""
    spec = _spec()
    spec["pieces"][0]["moves"] = [
        {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move",
         "condition": {"from_rank": 1}},
        {"kind": "sweep", "vectors": [[0, 1]], "range": 2},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
    ]
    spec["rules"]["en_passant"] = True
    engine = _engine(spec)
    oracle = sweep_oracle.Oracle(spec)
    empty = _state(engine, {**KINGS, "c2": ("X", 0), "d4": ("X", 1)})
    opened = engine.apply(empty, engine.find_move(empty, "Xc2-c4"))
    assert opened.ep_target == _sq("c3")
    swept = _state(engine, {**KINGS, "c2": ("X", 0), "c3": ("P", 1), "d4": ("X", 1)})
    move = engine.find_move(swept, "Xc2xc4 (sweeps c3)")
    after = engine.apply(swept, move)
    assert after.ep_target == NO_SQUARE
    for state in (empty, swept, opened, after):
        pos = _oracle_position(state, len(spec["pieces"]))
        assert _engine_moves(engine, state) == _oracle_moves(oracle, pos)


def test_on_a_cylinder_two_ways_to_one_square_promote_apart() -> None:
    """SR6d's declared ray beside the promotion: round the seam X on c6 reaches e6 two ways,
    each sweeping its own pieces, and each way offers both targets (four moves)."""
    spec = _spec()
    spec["rules"]["wrap"] = {"files": "both"}
    engine = _engine(spec)
    assert engine.spec.sweep_rays
    state = _state(engine, {"a1": ("K", 0), "f4": ("K", 1), "c6": ("X", 0), "d6": ("P", 1),
                            "b6": ("P", 1)})
    texts = [t for t in _texts_from(engine, state, "c6") if "e6" in t]
    assert texts == [
        "Xc6xe6 (sweeps b6)=N", "Xc6xe6 (sweeps b6)=Q", "Xc6xe6 (sweeps d6)=N",
        "Xc6xe6 (sweeps d6)=Q",
    ]
    oracle = sweep_oracle.Oracle(spec)
    pos = _oracle_position(state, len(spec["pieces"]))
    assert engine.perft(state, 2) == oracle.perft(pos, 2)
    for text in texts:
        after = engine.apply(state, engine.find_move(state, text))
        assert _at(engine, after, "e6") == text[-1]
        assert _at(engine, after, text.split("sweeps ")[1][:2]) is None
