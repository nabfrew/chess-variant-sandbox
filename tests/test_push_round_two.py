"""The push rule's round-2 lifts (Q1074b8, `docs/plan-push.md` E2, N4, N5, G1, G2, §3.13's purse).

Six pairs round 1 refused beside a push, each probed first: fed to the core past the validator
and held to its reading move by move against `tests/push_oracle.py`, then lifted (in the
validator alone where the core already played it, after a core fix where it did not):

* **E2**, en passant beside the en passant piece's pushing capture entry, where that entry
  keeps its capture (`push.captures: true`, or a push of the mover's own pieces only): en
  passant stays a capture onto the empty window square, the push applies to occupied landings
  only. A pushing entry whose capture the push replaces stays refused (what it would push is
  unstated).
* **N4**, sights: a frozen piece may not push (it makes no move) and may be pushed; a disarmed
  piece may not push (a push attacks, R1); a shielded piece may not be pushed, so it blocks a
  line, under a row anywhere in it; a commanded piece never pushes. Attack under a sight reads
  each source by what it attacks with (R1: a push source under its pusher's list), as the
  plain attack test does. A command that ``captures_own`` stays refused.
* **N5**, rest: a pushed resting piece rests no more, under a row too (its square may then hold
  a piece of the same code pushed onto it, which does not inherit the rest); the pusher rests
  as after any move.
* **G1**, reach: under ``on: "arrive"`` a piece standing on a goal decides, so a piece pushed
  onto its goal decides for its side; ``on: "leave"`` is decided by a move of the piece, never
  by its being pushed off; ``safe`` reads a push threat as an attack.
* **G2**, territory: the pusher claims its landing; a pushed piece claims nothing.
* **M1**, the purse: a bought piece is pushable from the next move; a pushed-off enemy pays its
  bounty into the pusher's purse.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import push_oracle
import pytest
import rest_oracle
from sight_oracle import SightOracle

from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import (
    SIGHT_DISARM,
    SIGHT_FREEZE,
    SIGHT_SHIELD,
    compile_spec,
)
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_BUY,
    MOVE_EN_PASSANT,
    MOVE_FOREIGN,
    MOVE_PUSH,
    decode_move,
    move_push,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.refusals import refused_ids

ROOT = Path(__file__).resolve().parent.parent


def _load(path: str) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    """The spec's engine, the validator and the bridge both admitting it."""
    assert validate_spec(spec) == [], validate_spec(spec)
    assert rust_supports(compile_spec(spec)) is None
    return Engine(spec)


def _decoder(engine: Engine) -> Callable[[int], tuple[int, str]]:
    spec = engine.spec
    return lambda value: (spec.owner_of(value), spec.piece_ids[spec.type_of(value)])


def _as_oracle(engine: Engine, move: int) -> push_oracle.PushMove:
    push = engine.spec.push
    assert push is not None
    origin, target, _kind, promoted = decode_move(move)
    _player, dx, dy = push.directions[move_push(move)]
    return origin, target, (dx, dy), promoted


def _core_pushes(engine: Engine, state: GameState) -> set[push_oracle.PushMove]:
    return {
        _as_oracle(engine, move) for move in engine.pseudo_legal_moves(state)
        if move_type(move) == MOVE_PUSH
    }


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _pushes_to(engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
    o, t = _sq(engine, origin), _sq(engine, target)
    return [
        move for move in engine.legal_moves(state)
        if move_type(move) == MOVE_PUSH and decode_move(move)[:2] == (o, t)
    ]


def _at(engine: Engine, state: GameState, square: str) -> str | None:
    value = state.board[_sq(engine, square)]
    if not value:
        return None
    spec = engine.spec
    return f"{spec.owner_of(value)}{spec.piece_ids[spec.type_of(value)]}"


Expect = Callable[[Engine, GameState], set[push_oracle.PushMove]]
After = Callable[[Engine, GameState, int, GameState], None]


def _walk(
    spec: dict[str, Any], expect: Expect, after: After | None = None, seeds: int = 4,
    plies: int = 120, legal: Callable[[Engine, GameState], list[int]] | None = None,
) -> int:
    """Seeded games, both seats: at every ply the core's push words are ``expect``'s, every
    push leaves the board the oracle says, the key is the one computed from scratch, and
    ``after`` checks what the row's reading says of the move. How many pushes were played."""
    engine = _engine(spec)
    played = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(plies):
            assert _core_pushes(engine, state) == expect(engine, state), (seed, ply)
            if engine.terminal(state) is not None:
                break
            moves = legal(engine, state) if legal is not None else engine.legal_moves(state)
            pushes = [move for move in moves if move_type(move) == MOVE_PUSH]
            move = rng.choice(pushes if pushes and rng.random() < 0.7 else moves)
            nxt = engine.apply(state, move)
            if move_type(move) == MOVE_PUSH:
                played += 1
                if not decode_move(move)[3]:
                    oracle = push_oracle.after_push(spec, state.board, _as_oracle(engine, move))
                    assert bytes(nxt.board) == oracle, (seed, ply)
            if after is not None:
                after(engine, state, move, nxt)
            state = nxt
            assert engine.hash(state) == engine._compute_hash(state), (seed, ply)
    return played


def _plain(engine: Engine, state: GameState, spec: dict[str, Any]) -> set[push_oracle.PushMove]:
    return push_oracle.push_moves(spec, _decoder(engine), state.board, state.side_to_move)


# ============================================================================ E2: en passant


def _passing_chess(captures: bool = True) -> dict[str, Any]:
    """Chess whose every non-royal capture entry pushes; the pawn's diagonal keeps its capture
    beside the push (``captures``), so en passant has a capture to be."""
    spec = _load("rulesets/chess.json")
    spec["name"] = "push chess with en passant"
    for piece in spec["pieces"]:
        if piece.get("royal"):
            continue
        for move in piece["moves"]:
            if move.get("mode", "both") != "move":
                move["push"] = {"captures": True} if piece["id"] == "P" and captures else True
    return spec


def test_e2_en_passant_is_lifted_where_the_pushing_entry_keeps_its_capture() -> None:
    assert validate_spec(_passing_chess()) == []
    assert rust_supports(compile_spec(_passing_chess())) is None
    refused = _passing_chess(captures=False)
    assert "push.E2" in refused_ids(validate_spec(refused))
    # A pawn's pushing capture entry that pushes only its own pieces keeps its capture too.
    own = _passing_chess()
    pawn = next(p for p in own["pieces"] if p["id"] == "P")
    pawn["moves"][2]["push"] = {"whose": "own"}
    assert validate_spec(own) == [], validate_spec(own)


def test_e2_en_passant_stays_a_capture_beside_the_push() -> None:
    """The window square is empty, so the pawn's diagonal offers en passant there as a capture;
    on an occupied diagonal it offers both the capture and the push."""
    engine = _engine(_passing_chess())
    state = engine.make_state({
        "e1": ("K", 0), "e5": ("P", 0), "e8": ("K", 1), "d5": ("P", 1), "f6": ("N", 1),
    }, side_to_move=0, moved=("e5", "d5"), ep_target="d6", ep_victim="d5")
    texts = engine.move_texts(state)
    passing = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_EN_PASSANT]
    assert len(passing) == 1
    after = engine.apply(state, passing[0])
    assert [_at(engine, after, s) for s in ("d6", "d5", "e5")] == ["0P", None, None]
    assert len(_pushes_to(engine, state, "e5", "f6")) == 1
    assert any("xf6" in text for text in texts)
    pushed = engine.apply(state, _pushes_to(engine, state, "e5", "f6")[0])
    assert [_at(engine, pushed, s) for s in ("f6", "g7")] == ["0P", "1N"]


def test_e2_seeded_games_hold_the_pushes_and_the_window() -> None:
    """Both seats: the push words are the oracle's, and the en passant words are the ones the
    same game without any push offers on that position."""
    spec = _passing_chess()
    twin_spec = _load("rulesets/chess.json")
    twin = Engine(twin_spec)
    passes = 0

    def expect(engine: Engine, state: GameState) -> set[push_oracle.PushMove]:
        nonlocal passes
        mine = {m for m in engine.pseudo_legal_moves(state) if move_type(m) == MOVE_EN_PASSANT}
        theirs = {m for m in twin.pseudo_legal_moves(state) if move_type(m) == MOVE_EN_PASSANT}
        assert mine == theirs
        passes += len(mine)
        return _plain(engine, state, spec)

    assert _walk(spec, expect, seeds=8, plies=160) > 100
    assert passes > 0


# ================================================================================ N4: sights


def _sighted(win: str = "royal_capture", command: bool = False) -> dict[str, Any]:
    """A 7x7 board under `chain: "row"`: a rook that pushes any piece's row on its lines, a
    pawn that shoves straight ahead and never captures, and three seers that step as kings:
    a freezer (the enemy on its rook lines, two squares), a disarmer (the enemy on its
    diagonals, two squares) and a shielder (its own pieces a king's step round it). With
    ``command``, the freezer commands instead (empty landings only)."""
    step = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
    every = ["R", "P", "F", "D", "S"]
    first = "command" if command else "freeze"
    spec: dict[str, Any] = {
        "name": f"push sights {win} {first}",
        "board": {"width": 7, "height": 7},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [step]},
            {"id": "R", "value": 5, "pushes": every, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any"}},
            ]},
            {"id": "P", "value": 1, "pushes": every, "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move",
                 "push": True},
            ]},
            {"id": "F", "value": 2, "moves": [{**step, "mode": "move"}], "sight": [
                {"vectors": [[1, 0]], "symmetry": "rotate4", "range": 2, "effect": first,
                 "whose": "enemy"},
            ]},
            {"id": "D", "value": 2, "moves": [{**step, "mode": "move"}], "sight": [
                {"vectors": [[1, 1]], "symmetry": "rotate4", "range": 2, "effect": "disarm",
                 "whose": "enemy"},
            ]},
            {"id": "S", "value": 2, "moves": [{**step, "mode": "move"}], "sight": [
                {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
                 "effect": "shield", "whose": "own"},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "g1"]},
            {"piece": "F", "squares": ["b1"]},
            {"piece": "D", "squares": ["c1"]},
            {"piece": "S", "squares": ["e1"]},
            {"piece": "P", "squares": ["b2", "c2", "d2", "e2", "f2"]},
        ]},
        "rules": {"win": win, "push": {"chain": "row"}, "max_plies": 200},
    }
    return spec


def _sight_expect(spec: dict[str, Any]) -> Expect:
    seer = SightOracle(spec)

    def expect(engine: Engine, state: GameState) -> set[push_oracle.PushMove]:
        effects = seer.effects(state.board)
        return {
            push for push in _plain(engine, state, spec)
            if not effects[push[0]] & (SIGHT_FREEZE | SIGHT_DISARM)
            and not any(
                effects[square] & SIGHT_SHIELD
                for square in push_oracle.line_squares(spec, state.board, push)
            )
        }

    return expect


def test_n4_sights_are_lifted_and_a_turncoat_command_stays_refused() -> None:
    for spec in (_sighted(), _sighted("checkmate"), _sighted(command=True)):
        assert validate_spec(spec) == [], validate_spec(spec)
        assert rust_supports(compile_spec(spec)) is None
    turncoat = _sighted(command=True)
    turncoat["pieces"][3]["sight"][0]["captures_own"] = True
    assert "push.N4" in refused_ids(validate_spec(turncoat))
    assert rust_supports(compile_spec(_sighted(command=True))) is None


def test_n4_a_frozen_or_disarmed_piece_does_not_push_and_a_shielded_piece_is_not_pushed() -> None:
    engine = _engine(_sighted())
    # The black freezer on a6 freezes the white rook on a4 (two squares down its file).
    frozen = engine.make_state({
        "d1": ("K", 0), "a4": ("R", 0), "b4": ("P", 1), "a6": ("F", 1), "d7": ("K", 1),
    }, side_to_move=0)
    assert _pushes_to(engine, frozen, "a4", "b4") == []
    # A frozen piece may be pushed: the black pawn on b4, frozen by the white freezer on b6,
    # is still shoved by the white rook.
    pushed = engine.make_state({
        "d1": ("K", 0), "a4": ("R", 0), "b4": ("P", 1), "b6": ("F", 0), "d7": ("K", 1),
    }, side_to_move=0)
    assert len(_pushes_to(engine, pushed, "a4", "b4")) == 1
    # The black disarmer on c6 disarms the white rook on a4 (two squares down its diagonal).
    disarmed = engine.make_state({
        "d1": ("K", 0), "a4": ("R", 0), "b4": ("P", 1), "c6": ("D", 1), "d7": ("K", 1),
    }, side_to_move=0)
    assert _pushes_to(engine, disarmed, "a4", "b4") == []
    # A black shielder on d5 shields the black pawn on c4, the row's second piece (not b4,
    # the first): the row b4, c4 cannot move; take the shielder away and it can.
    shielded = {
        "d1": ("K", 0), "a4": ("R", 0), "b4": ("P", 1), "c4": ("P", 1), "d5": ("S", 1),
        "g7": ("K", 1),
    }
    assert _pushes_to(engine, engine.make_state(shielded, side_to_move=0), "a4", "b4") == []
    del shielded["d5"]
    open_row = engine.make_state(shielded, side_to_move=0)
    [push] = _pushes_to(engine, open_row, "a4", "b4")
    after = engine.apply(open_row, push)
    assert [_at(engine, after, s) for s in ("b4", "c4", "d4")] == ["0R", "1P", "1P"]


def test_n4_a_push_threat_under_a_sight_reads_the_pusher_s_list() -> None:
    """R1 in a sight game: a rook whose list does not name the king gives no check down its
    file (as in a game without a sight); a frozen pusher whose list names it gives none."""
    spec = _sighted("checkmate")
    rook = spec["pieces"][1]
    rook["pushes"] = ["R", "P", "F", "D", "S"]
    engine = _engine(spec)
    state = engine.make_state({
        "a1": ("K", 0), "d2": ("R", 0), "d6": ("K", 1), "g7": ("S", 1),
    }, side_to_move=1)
    assert not engine.in_check(state)
    rook["pushes"] = ["R", "P", "F", "D", "S", "K"]
    engine = _engine(spec)
    state = engine.make_state({
        "a1": ("K", 0), "d2": ("R", 0), "d6": ("K", 1), "g7": ("S", 1),
    }, side_to_move=1)
    assert engine.in_check(state)
    # The black freezer on b2 freezes the rook on d2 (two squares along the rank): no check,
    # though nothing stands between the rook and the king.
    state = engine.make_state({
        "a1": ("K", 0), "d2": ("R", 0), "d6": ("K", 1), "b2": ("F", 1),
    }, side_to_move=1)
    assert not engine.in_check(state)


def test_n4_a_commanded_piece_never_pushes() -> None:
    spec = _sighted(command=True)

    def no_foreign_push(engine: Engine, state: GameState, move: int, nxt: GameState) -> None:
        assert not (move_type(move) == MOVE_PUSH and move & MOVE_FOREIGN)

    def legal(engine: Engine, state: GameState) -> list[int]:
        moves = engine.legal_moves(state)
        assert not any(move_type(m) == MOVE_PUSH and m & MOVE_FOREIGN for m in moves)
        return moves

    expect = _sight_expect(spec)
    assert _walk(spec, expect, no_foreign_push, legal=legal) > 50


@pytest.mark.parametrize("win", ["royal_capture", "checkmate"])
def test_n4_seeded_games_hold_the_sights(win: str) -> None:
    spec = _sighted(win)
    assert _walk(spec, _sight_expect(spec)) > 50


# ================================================================================== N5: rest


def _resting() -> dict[str, Any]:
    """A 7x7 board under `chain: "row"`: a rook and pawns that rest a turn after they move, the
    rook pushing any piece's row on its lines, the pawn shoving straight ahead."""
    step = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
    return {
        "name": "push rest",
        "board": {"width": 7, "height": 7},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [step]},
            {"id": "R", "value": 5, "rest": 1, "pushes": ["R", "P"], "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any"}},
            ]},
            {"id": "P", "value": 1, "rest": 1, "pushes": ["P"], "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "symmetry": "rotate4",
                 "mode": "move", "push": True},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "g1"]},
            {"piece": "P", "squares": ["b2", "c2", "d2", "e2", "f2"]},
        ]},
        "rules": {"win": "royal_capture", "push": {"chain": "row"}, "max_plies": 200},
    }


def _rest_after(engine: Engine, before: GameState, move: int, after: GameState) -> Any:
    """N5's reading: the rest after ``move``, worked out by the rest oracle with every piece
    the push moved (the line, before the move) out of rest first."""
    spec_dict = json.loads(engine.spec.source)
    lists = rest_oracle.lists_of(before)
    player = before.side_to_move
    if move_type(move) == MOVE_PUSH:
        line = set(push_oracle.line_squares(
            spec_dict, before.board, _as_oracle(engine, move)
        ))
        lists = (
            [e for e in lists[0] if e[0] not in line],
            [e for e in lists[1] if e[0] not in line],
        )
        end: int | None = decode_move(move)[1]
    else:
        end = rest_oracle.end_of_move(engine, before, after, move)
    return rest_oracle.as_rest(rest_oracle.after_turn(engine, lists, after.board, player, end))


def test_n5_rest_is_lifted() -> None:
    assert validate_spec(_resting()) == [], validate_spec(_resting())
    assert rust_supports(compile_spec(_resting())) is None


def test_n5_a_pushed_resting_piece_rests_no_more_even_where_its_twin_is_pushed_on() -> None:
    """Black's pawn c5-c4 rests; white's rook shoves the row b4, c4 one on. The pawn that
    rested now stands on d4 and the pawn from b4 on c4, the square that rested: neither rests,
    and the rook does."""
    engine = _engine(_resting())
    state = engine.make_state({
        "a1": ("K", 0), "a4": ("R", 0), "b4": ("P", 1), "c5": ("P", 1), "g7": ("K", 1),
    }, side_to_move=1)
    step = next(
        m for m in engine.legal_moves(state)
        if decode_move(m)[:2] == (_sq(engine, "c5"), _sq(engine, "c4"))
    )
    state = engine.apply(state, step)
    assert state.rest == ((), ((_sq(engine, "c4"), 1),))
    [push] = _pushes_to(engine, state, "a4", "b4")
    after = engine.apply(state, push)
    assert [_at(engine, after, s) for s in ("b4", "c4", "d4")] == ["0R", "1P", "1P"]
    assert after.rest == (((_sq(engine, "b4"), 1),), ())
    assert after.rest == _rest_after(engine, state, push, after)
    # Both black pawns may move again.
    movers = {decode_move(m)[0] for m in engine.legal_moves(after)}
    assert {_sq(engine, "c4"), _sq(engine, "d4")} <= movers


def test_n5_seeded_games_hold_the_rest() -> None:
    spec = _resting()

    def after(engine: Engine, state: GameState, move: int, nxt: GameState) -> None:
        assert nxt.rest == _rest_after(engine, state, move, nxt)

    def legal(engine: Engine, state: GameState) -> list[int]:
        moves = engine.legal_moves(state)
        assert sorted(moves) == sorted(rest_oracle.legal(engine, state))
        return moves

    assert _walk(spec, lambda e, s: _plain(e, s, spec), after, legal=legal) > 50


# ================================================================================= G1: reach


def _reaching(on: str = "arrive", safe: bool = False) -> dict[str, Any]:
    """A 7x7 board with one goal for both players, d4: a rook that pushes any rook's or pawn's
    row on its lines, a pawn that shoves straight ahead."""
    step = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
    reach: dict[str, Any] = {"squares": ["d4"], "mirror": False, "result": "win"}
    if on != "arrive":
        reach["on"] = on
    if safe:
        reach["safe"] = True
    return {
        "name": f"push reach {on} {safe}",
        "board": {"width": 7, "height": 7},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [step]},
            {"id": "R", "value": 5, "pushes": ["R", "P"], "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any"}},
            ]},
            {"id": "P", "value": 1, "pushes": ["P"], "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move", "push": True},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "g1"]},
            {"piece": "P", "squares": ["b2", "c2", "e2", "f2"]},
        ]},
        "rules": {"win": "royal_capture", "reach": reach, "push": {"chain": "row"},
                  "max_plies": 200},
    }


def test_g1_reach_is_lifted() -> None:
    for spec in (_reaching(), _reaching("leave"), _reaching(safe=True)):
        assert validate_spec(spec) == [], validate_spec(spec)
        assert rust_supports(compile_spec(spec)) is None


def test_g1_a_piece_pushed_onto_its_goal_decides_for_its_side() -> None:
    engine = _engine(_reaching())
    for owner, winner in ((1, 1), (0, 0)):
        state = engine.make_state({
            "a1": ("K", 0), "b4": ("R", 0), "c4": ("P", owner), "g7": ("K", 1),
        }, side_to_move=0)
        [push] = _pushes_to(engine, state, "b4", "c4")
        after = engine.apply(state, push)
        assert _at(engine, after, "d4") == f"{owner}P"
        outcome = engine.terminal(after)
        assert outcome is not None and (outcome.winner, outcome.reason) == (winner, "reached")


def test_g1_being_pushed_off_a_goal_leaves_nothing_and_the_pusher_s_own_move_does() -> None:
    engine = _engine(_reaching("leave"))
    state = engine.make_state({
        "a1": ("K", 0), "c4": ("R", 0), "d4": ("P", 1), "g7": ("K", 1),
    }, side_to_move=0)
    [push] = _pushes_to(engine, state, "c4", "d4")
    after = engine.apply(state, push)
    assert [_at(engine, after, s) for s in ("d4", "e4")] == ["0R", "1P"]
    assert engine.terminal(after) is None
    # A rook standing on the goal: its own push off it (onto e4's pawn) leaves it.
    after = engine.make_state({
        "a1": ("K", 0), "d4": ("R", 0), "e4": ("P", 1), "g7": ("K", 1),
    }, side_to_move=0)
    [push] = _pushes_to(engine, after, "d4", "e4")
    left = engine.apply(after, push)
    outcome = engine.terminal(left)
    assert outcome is not None and (outcome.winner, outcome.reason) == (0, "reached")


def test_g1_safe_reads_a_push_threat_as_an_attack() -> None:
    """White's rook shoves its own pawn onto d4, under the black pawn on d5. A pawn whose list
    names pawns may push it on down the d-file: not safe, nothing decided. One whose list
    leaves pawns out threatens nothing there, and the arrival decides."""
    placement = {
        "a1": ("K", 0), "b4": ("R", 0), "c4": ("P", 0), "d5": ("P", 1), "g7": ("K", 1),
    }
    for listed, decided in ((["P"], False), (["R"], True)):
        spec = _reaching(safe=True)
        spec["pieces"][2]["pushes"] = listed
        engine = _engine(spec)
        state = engine.make_state(placement, side_to_move=0)
        [push] = _pushes_to(engine, state, "b4", "c4")
        after = engine.apply(state, push)
        assert _at(engine, after, "d4") == "0P"
        outcome = engine.terminal(after)
        if decided:
            assert outcome is not None and (outcome.winner, outcome.reason) == (0, "reached")
        else:
            assert outcome is None


@pytest.mark.parametrize("on", ["arrive", "leave"])
def test_g1_seeded_games_hold_the_goal(on: str) -> None:
    """Under ``arrive`` the game is decided, by reach, exactly where a piece stands on d4 (the
    first player's first); under ``leave`` exactly where the last move set out from d4."""
    spec = _reaching(on)
    goal = 3 * 7 + 3
    reached = 0

    def after(engine: Engine, state: GameState, move: int, nxt: GameState) -> None:
        nonlocal reached
        outcome = engine.terminal(nxt)
        spec_c = engine.spec
        kings = {spec_c.owner_of(v) for v in nxt.board if v and spec_c.type_of(v) == 0}
        if kings != {0, 1}:
            return
        if on == "arrive":
            value = nxt.board[goal]
            expected = spec_c.owner_of(value) if value else None
        else:
            expected = state.side_to_move if decode_move(move)[0] == goal else None
        got = outcome.winner if outcome is not None and outcome.reason == "reached" else None
        assert got == expected, (on, engine.describe_move(state, move))
        reached += got is not None

    assert _walk(spec, lambda e, s: _plain(e, s, spec), after, seeds=8) > 50
    assert reached > 0


# ============================================================================= G2: territory


def _claiming(mode: str = "repaint") -> dict[str, Any]:
    spec = _reaching()
    spec["name"] = f"push territory {mode}"
    del spec["rules"]["reach"]
    spec["rules"]["territory"] = {"mode": mode}
    return spec


def test_g2_territory_is_lifted() -> None:
    for mode in ("repaint", "permanent"):
        spec = _claiming(mode)
        assert validate_spec(spec) == [], validate_spec(spec)
        assert rust_supports(compile_spec(spec)) is None


@pytest.mark.parametrize("mode", ["repaint", "permanent"])
def test_g2_the_pusher_claims_its_landing_and_a_pushed_piece_nothing(mode: str) -> None:
    spec = _claiming(mode)
    seen = 0

    def after(engine: Engine, state: GameState, move: int, nxt: GameState) -> None:
        nonlocal seen
        if move_type(move) != MOVE_PUSH:
            return
        seen += 1
        target = decode_move(move)[1]
        before, now = state.colour, nxt.colour
        changed = {sq for sq in range(len(before)) if before[sq] != now[sq]}
        assert changed <= {target}
        mine = state.side_to_move + 1
        if before[target] == 0 or mode == "repaint":
            assert now[target] == mine

    assert _walk(spec, lambda e, s: _plain(e, s, spec), after) > 50
    assert seen > 50


# ============================================================================== M1: the purse


def _buying() -> dict[str, Any]:
    """The purse market (`tests/_purse_market.json`) with every capture entry pushing."""
    spec = _load("tests/_purse_market.json")
    spec["name"] = "push purse market"
    for piece in spec["pieces"]:
        if piece.get("royal"):
            continue
        for move in piece["moves"]:
            if move.get("mode", "both") != "move":
                move["push"] = True
    return spec


def test_m1_the_purse_is_lifted() -> None:
    spec = _buying()
    assert validate_spec(spec) == [], validate_spec(spec)
    assert rust_supports(compile_spec(spec)) is None


def test_m1_a_bought_piece_is_pushed_and_a_pushed_off_bounty_fills_the_purse() -> None:
    """Black buys a knight onto b6; white's rook on b3 slides up the b-file to it and shoves it
    off the board, which takes it: its bounty (2) fills white's purse."""
    engine = _engine(_buying())
    state = engine.make_state({
        "c1": ("K", 0), "b3": ("R", 0), "f6": ("K", 1),
    }, side_to_move=1, purse=(0, 6))
    knight = engine.spec.piece_ids.index("N")
    [buy] = [
        m for m in engine.legal_moves(state)
        if move_type(m) == MOVE_BUY and decode_move(m)[1] == _sq(engine, "b6")
        and decode_move(m)[3] == knight
    ]
    bought = engine.apply(state, buy)
    assert _at(engine, bought, "b6") == "1N"
    [push] = _pushes_to(engine, bought, "b3", "b6")
    assert engine.describe_move(bought, push) == "Rb3>>b6"
    after = engine.apply(bought, push)
    assert _at(engine, after, "b6") == "0R"
    assert after.purse == (2, bought.purse[1])
    assert engine.hash(after) == engine._compute_hash(after)


def test_m1_seeded_games_hold_the_pushes_beside_the_market() -> None:
    spec = _buying()
    bought = 0

    def after(engine: Engine, state: GameState, move: int, nxt: GameState) -> None:
        nonlocal bought
        bought += move_type(move) == MOVE_BUY

    assert _walk(spec, lambda e, s: _plain(e, s, spec), after, seeds=6) > 50
    assert bought > 0


# ========================================================== the bridge agrees with the validator


LIFTED: dict[str, Callable[[], dict[str, Any]]] = {
    "E2": _passing_chess, "N4": _sighted, "N5": _resting, "G1": _reaching, "G2": _claiming,
    "M1": _buying,
}


@pytest.mark.parametrize("rid", list(LIFTED))
def test_the_validator_and_the_bridge_agree_on_each_lift(rid: str) -> None:
    spec = LIFTED[rid]()
    assert validate_spec(spec) == []
    assert rust_supports(compile_spec(spec)) is None


def test_what_stays_refused_is_held_by_the_bridge_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A command that ``captures_own`` is refused by the validator and held by the bridge for a
    spec that reaches it past it. (E2's remainder, a pushing capture whose push replaces the
    capture, is the validator's alone: the compiled landing does not say a capture was
    replaced, and the core offers no en passant on a landing without its capture bit.)"""
    from chess_variant_sandbox.engine import compile as compile_module

    monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)
    turncoat = _sighted(command=True)
    turncoat["pieces"][3]["sight"][0]["captures_own"] = True
    held = rust_supports(compile_spec(turncoat))
    assert held is not None and "push.N4" in held

