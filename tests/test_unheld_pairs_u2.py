"""U2 (`docs/reference/unheld-pairs-plan.md` §2.2): nine refused pairs the core may already play.

Each pair was fed to the core past the validator (as SR2 did) and held to a declared oracle by
the meaning §2.2 decided. Seven play so and are lifted in the validator alone (no core change,
no table, no stored hash); one is narrowed; one is left refused. Each lift below is a perft
over make/unmake against fresh applies, with the core's moves and answers held to the oracle
move by move over random walks:

* ``income.I1`` -- income beside a points count fills the purse alone; the points count
  bounties (`tests/test_income.py::_earned` is the income oracle);
* ``gating.duck`` -- the gated piece enters in the piece half; the duck then lands only on an
  empty square (the duck game without a hand is the twin: every other move is its);
* ``gating.control`` -- a held or shared piece never gates, a held type with a first-move mark
  included (`tests/neutral_oracle.py` on the twin without a hand);
* ``goals.extinction``, ``goals.bare_royal`` -- a banked piece is off the board and both endings
  count the board; a bank that reaches the count wins first (a verdict oracle read off the spec);
* ``shared.price`` -- hiring a shared piece makes it the hirer's own (`NeutralOracle.purchases`);
* ``run.territory`` -- only the hopper claims (`tests/hop_run_oracle.py` beside a colour oracle);
* ``run.en_passant`` -- another piece's hop over a run opens no window; the en passant piece's
  own hop does (the core reads it as a double step), so that narrower case stays refused,
  held by U7, and is pinned here as a strict xfail;
* ``pass.P5`` -- the core offers a pass during a placement phase, which the reading forbids:
  left refused, held by U3, pinned as a strict xfail.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.engine.moves import (
    MOVE_DUCK,
    MOVE_EN_PASSANT,
    MOVE_HIRE,
    MOVE_PASS,
    decode_move,
    move_foreign,
    move_gate,
    move_type,
)
from chess_variant_sandbox.engine.state import NO_SQUARE
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import PURSE_MAX, validate_spec
from tests import hop_run_oracle
from tests.neutral_oracle import NeutralOracle
from tests.test_hop_runs import _decode, _word
from tests.test_income import _earned, _income
from tests.test_scoring_goals import _crossings
from tests.test_shared_control import _walk_spec

ROOT = Path(__file__).resolve().parents[1]
GATE_BITS = ~((0x1F << 25) | (1 << 31)) & 0xFFFFFFFF


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _fresh_perft(engine: Engine, state: Any) -> None:
    """Perft 2 over make/unmake against a count built from fresh applies."""
    moves = engine.legal_moves(state)
    assert engine.perft(state, 2) == sum(len(engine.legal_moves(engine.apply(state, m))) for m in moves)


# ------------------------------------------------------------------------------ income.I1


def _income_points() -> dict[str, Any]:
    spec = _income()
    spec["rules"]["scoring"] = {"count": "points"}
    spec["rules"]["purse"]["start"] = 40
    return spec


def test_income_beside_points_validates() -> None:
    assert validate_spec(_income_points()) == []


def test_income_fills_the_purse_and_the_points_count_bounties_alone() -> None:
    spec = _income_points()
    engine = Engine(spec)
    bounty = {i: p.get("bounty", 0) for i, p in enumerate(spec["pieces"])}
    paid = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = engine.initial_state()
        assert state.points == (0, 0)
        for ply in range(120):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            if ply % 10 == 0:
                _fresh_perft(engine, state)
            move = rng.choice(moves)
            mover, nxt = state.side_to_move, 1 - state.side_to_move
            after = engine.apply(state, move)

            def theirs(board: bytes, nxt: int = nxt) -> Counter[int]:
                return Counter(engine.spec.type_of(v) for v in board if v and engine.spec.owner_of(v) == nxt)

            gained = sum(bounty[t] * n for t, n in (theirs(state.board) - theirs(after.board)).items())
            assert after.points[mover] == state.points[mover] + gained
            assert after.points[nxt] == state.points[nxt]
            earned = _earned(engine, spec, after.board, nxt)
            assert after.purse[nxt] == min(PURSE_MAX, state.purse[nxt] + earned)
            cost = engine.move_cost(state, move)
            assert after.purse[mover] == min(PURSE_MAX, state.purse[mover] - cost + gained)
            assert after.key == engine._compute_hash(after)
            paid += gained > 0
            state = after
    assert paid > 0


# --------------------------------------------------------------------- goals beside a bank


def _bank_beside(ending: str) -> dict[str, Any]:
    spec = _crossings()
    if ending == "extinction":
        spec["rules"]["reach"]["count"] = 4
        spec["rules"]["extinction"] = {"pieces": ["R"], "min_count": 12}
        return spec
    spec["rules"]["reach"].update(count=8, piece="R")
    spec["rules"]["bare_royal"] = "loss"
    spec["pieces"].append({"id": "K", "royal": True, "value": 0, "moves": [
        {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}]})
    spec["setup"]["placements"] = [
        {"piece": "R", "squares": ["a1", "b1", "c1", "e1", "f1", "a2", "c2", "e2", "g2"]},
        {"piece": "K", "squares": ["d1"]},
    ]
    return spec


def _verdict(engine: Engine, spec: dict[str, Any], state: Any) -> tuple[int, str] | None:
    """The oracle: a bank reaching the count first, then the board's count of each ending."""
    rules = spec["rules"]
    for side in (0, 1):
        if state.banked[side] >= rules["reach"]["count"]:
            return side, "reached"
    kinds = [[spec["pieces"][engine.spec.type_of(v)] for v in state.board
              if v and engine.spec.owner_of(v) == side] for side in (0, 1)]
    if "extinction" in rules:
        listed = rules["extinction"]["pieces"]
        for side in (0, 1):
            if sum(p["id"] in listed for p in kinds[side]) < rules["extinction"].get("min_count", 1):
                return 1 - side, "extinction"
    if "bare_royal" in rules:
        for side in (0, 1):
            if kinds[side] and all(p.get("royal") for p in kinds[side]):
                return 1 - side, "bare_royal"
    return None


@pytest.mark.parametrize("ending", ["extinction", "bare_royal"])
def test_a_bank_beside_the_ending_validates_and_plays_as_the_board_counts(ending: str) -> None:
    spec = _bank_beside(ending)
    assert validate_spec(spec) == []
    engine = Engine(spec)
    seen: Counter[str] = Counter()
    for seed in range(30):
        rng = random.Random(seed)
        state = engine.initial_state()
        while True:
            outcome = engine.terminal(state)
            got = None if outcome is None else (outcome.winner, outcome.reason)
            if got is not None and got[1] not in ("reached", "extinction", "bare_royal"):
                break
            assert got == _verdict(engine, spec, state), (seed, state.ply, state.banked)
            if got is not None:
                seen[got[1]] += 1
                break
            if state.ply % 9 == 0:
                _fresh_perft(engine, state)
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
            assert state.key == engine._compute_hash(state)
    assert seen[ending] > 0, seen


@pytest.mark.parametrize(("banked", "winner", "reason"), [((2, 0), 0, "reached"), ((1, 0), 1, "extinction")])
def test_banking_the_last_runner_loses_unless_it_reaches_the_count(
    banked: tuple[int, int], winner: int, reason: str,
) -> None:
    spec = _crossings()
    spec["rules"]["reach"]["count"] = 3
    spec["rules"]["extinction"] = {"pieces": ["R"]}
    engine = Engine(spec)
    state = engine.make_state({"c6": ("R", 0), "e2": ("R", 1)}, banked=banked)
    after = engine.apply(state, engine.find_move(state, "Rc6-c7"))
    outcome = engine.terminal(after)
    assert outcome is not None and (outcome.winner, outcome.reason) == (winner, reason)


# ---------------------------------------------------------------------------- gating


def test_gating_beside_the_duck_gates_in_the_piece_half() -> None:
    duck = _load("tests/_duck_chess.json")
    gating = copy.deepcopy(duck)
    gating["rules"]["drops"] = {"from_captures": False, "stock": {"N": 1, "B": 1}, "gating": {"ranks": [0]}}
    assert validate_spec(gating) == []
    engine, twin = Engine(gating), Engine(duck)
    gated = 0
    for seed in range(12):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(50):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            if all(move_type(m) == MOVE_DUCK for m in moves):
                assert not any(move_gate(m) for m in moves)
                empty = {sq for sq, v in enumerate(state.board) if v == 0}
                assert {decode_move(m)[1] for m in moves} <= empty
            else:
                plain = dataclasses.replace(state, hands=twin.initial_state().hands)
                assert sorted(twin.legal_moves(plain)) == sorted(m for m in moves if not move_gate(m))
                held = sum(1 for n in state.hands[state.side_to_move] if n)
                home = 0 if state.side_to_move == 0 else 7
                for word, n in Counter(m & GATE_BITS for m in moves if move_gate(m)).items():
                    origin = decode_move(word)[0]
                    assert origin // 8 == home and (state.unmoved >> origin) & 1
                    assert n in (held, 2 * held)
            if ply % 7 == 0:
                _fresh_perft(engine, state)
            options = [m for m in moves if move_gate(m)]
            move = rng.choice(options) if options and rng.random() < 0.5 else rng.choice(moves)
            after = engine.apply(state, move)
            if move_gate(move):
                gated += 1
                origin = decode_move(move)[0]
                assert engine.spec.owner_of(after.board[origin]) == state.side_to_move
                assert origin not in {decode_move(m)[1] for m in engine.legal_moves(after)}
            assert after.key == engine._compute_hash(after)
            state = after
    assert gated > 0


@pytest.mark.parametrize("control", ["majority", "shared"])
@pytest.mark.parametrize("first_move", [False, True], ids=["plain", "first-move-mark"])
def test_a_held_piece_never_gates(control: str, first_move: bool) -> None:
    from tests.test_neutral_pieces import _spec

    plain = _spec()
    plain["pieces"][-1]["control"] = control
    plain["setup"]["placements"][-1]["squares"] = ["c1", "d3"]
    if first_move:
        plain["pieces"][-1]["moves"] = plain["pieces"][-1]["moves"] + [
            {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
             "condition": {"first_move": True}}]
    gating = copy.deepcopy(plain)
    gating["rules"]["drops"] = {"from_captures": False, "stock": {"G": 1}, "gating": {"ranks": [0]}}
    assert validate_spec(gating) == []
    engine, twin = Engine(gating), Engine(plain)
    oracle = NeutralOracle(plain, twin)
    if first_move:  # SR8i's mark: the held piece on its gate square is unmoved
        assert (engine.initial_state().unmoved >> 2) & 1
    gates = 0
    for seed in range(10):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(40):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            bare = dataclasses.replace(state, hands=twin.initial_state().hands)
            assert sorted(m for m in moves if not move_gate(m)) == oracle.legal_moves(bare)
            for move in moves:
                if move_gate(move):
                    gates += 1
                    assert not move_foreign(move)
                    assert engine.spec.owner_of(state.board[decode_move(move)[0]]) == state.side_to_move
            if ply % 5 == 0:
                _fresh_perft(engine, state)
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state)
    assert gates > 0


# ------------------------------------------------------------------------ shared.price


@pytest.mark.parametrize("bribe", [False, True])
def test_a_shared_piece_for_hire_becomes_the_hirer_s(bribe: bool) -> None:
    spec = _walk_spec()
    spec["pieces"][-1]["price"] = 3
    spec["rules"]["purse"] = {"start": 7, **({"bribe": True} if bribe else {})}
    assert validate_spec(spec) == []
    engine = Engine(spec)
    oracle = NeutralOracle(spec, engine)
    hires = 0
    for seed in range(12):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(60):
            if engine.terminal(state) is not None:
                break
            core = sorted(engine.legal_moves(state))
            assert core == oracle.legal_moves(state), (seed, ply)
            if ply % 8 == 0:
                _fresh_perft(engine, state)
            offered = [m for m in core if move_type(m) == MOVE_HIRE]
            move = rng.choice(offered) if offered and rng.random() < 0.6 else rng.choice(core)
            after = engine.apply(state, move)
            if move_type(move) == MOVE_HIRE:
                hires += 1
                square = decode_move(move)[1]
                assert engine.spec.owner_of(after.board[square]) == state.side_to_move
                assert after.purse[state.side_to_move] == (
                    state.purse[state.side_to_move] - oracle.cost(state, square)
                )
            assert after.key == engine._compute_hash(after)
            state = after
    assert hires > 0


# ------------------------------------------------------------------------------- runs


@pytest.mark.parametrize("mode", ["permanent", "repaint"])
def test_a_displacement_beside_territory_claims_only_the_landing(mode: str) -> None:
    spec = _load("rulesets/leapfrog.json")
    spec["rules"]["territory"] = {"mode": mode}
    assert validate_spec(spec) == []
    engine = Engine(spec)
    decode = _decode(engine)
    displaced = 0
    for seed in range(3):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(40):
            if engine.terminal(state) is not None:
                break
            legal = engine.legal_moves(state)
            assert {_word(m) for m in legal} == hop_run_oracle.moves(spec, decode, state.board, state.side_to_move)
            for move in legal:
                after = engine.apply(state, move)
                assert after.board == hop_run_oracle.after(spec, state.board, _word(move))
                target = decode_move(move)[1]
                colour = bytearray(state.colour)
                if mode == "repaint" or not colour[target]:
                    colour[target] = state.side_to_move + 1
                assert after.colour == bytes(colour)
                displaced += _word(move)[2] == "displace"
            if ply % 6 == 0:
                _fresh_perft(engine, state)
            state = engine.apply(state, rng.choice(legal))
            assert state.key == engine._compute_hash(state)
    assert displaced > 0


def _ep_runs(pawn_hops: bool, displace: bool = False) -> dict[str, Any]:
    """7x7: pawns with a double step and en passant, a royal king; a hop over a run on the
    stepping piece F or, with ``pawn_hops``, on the pawn itself."""
    pawn = {"id": "P", "value": 1, "moves": [
        {"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
        {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
         "condition": {"from_rank": 1}},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}]}
    stepper = {"id": "F", "value": 2, "moves": [
        {"kind": "leap", "vectors": [[1, 0], [0, 1]], "symmetry": "rotate4", "mode": "both"}]}
    hop: dict[str, Any] = {"kind": "hop", "vectors": [[0, 1], [1, 0]], "symmetry": "lateral",
                           "mode": "move", "run": True}
    if displace:
        hop["displace"] = "back"
    (pawn if pawn_hops else stepper)["moves"].append(hop)
    king = {"id": "K", "royal": not displace, "value": 0, "moves": [
        {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}]}
    return {
        "schema_version": 31, "name": "runs beside en passant", "board": {"width": 7, "height": 7},
        "pieces": [pawn, stepper, king],
        "setup": {"mirror": True, "placements": [
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2", "g2"]},
            {"piece": "F", "squares": ["a1", "b1", "c1", "e1", "f1", "c3"]},
            {"piece": "K", "squares": ["d1"]}]},
        "rules": {"win": "none" if displace else "royal_capture",
                  "stalemate": "loss" if displace else "draw", "max_plies": 200,
                  "promotion": {"piece": "P", "rank": "last", "to": ["F"]}, "en_passant": True},
    }


def _windows_over_empty_squares(spec: dict[str, Any], seeds: int) -> tuple[int, int]:
    """Random walks: every window opens over a square empty before and after the move, and the
    core's moves are its twin's without en passant plus captures in passing, offered only
    after a double step. Returns (hops seen, captures in passing offered)."""
    engine = Engine(spec)
    plain_spec = copy.deepcopy(spec)
    plain_spec["rules"].pop("en_passant")
    twin = Engine(plain_spec)
    hops = passing = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(60):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            bare = dataclasses.replace(state, ep_target=NO_SQUARE, ep_victim=NO_SQUARE)
            assert sorted(twin.legal_moves(bare)) == sorted(m for m in moves if move_type(m) != MOVE_EN_PASSANT)
            if any(move_type(m) == MOVE_EN_PASSANT for m in moves):
                passing += 1
                assert state.ep_target != NO_SQUARE
            for move in moves:
                after = engine.apply(state, move)
                origin, target, _, _ = decode_move(move)
                line = abs(target - origin)
                jumped = line in (2, 14) and state.board[(origin + target) // 2]
                hops += bool(jumped)
                if after.ep_target != NO_SQUARE:
                    assert not state.board[after.ep_target] and not after.board[after.ep_target], (seed, ply, move)
            if ply % 6 == 0:
                _fresh_perft(engine, state)
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state)
    return hops, passing


@pytest.mark.parametrize("displace", [False, True], ids=["run", "displace"])
def test_another_piece_s_hop_over_a_run_opens_no_window(displace: bool) -> None:
    spec = _ep_runs(pawn_hops=False, displace=displace)
    assert validate_spec(spec) == []
    hops, passing = _windows_over_empty_squares(spec, seeds=12)
    assert hops > 0 and passing > 0


def test_the_en_passant_piece_s_own_hop_stays_refused() -> None:
    """`refusals.toml`'s ``run.en_passant.self``, held by U7."""
    errors = validate_spec(_ep_runs(pawn_hops=True))
    assert errors == [
        "piece 'P' hops over a run or displaces and is the en passant piece: its hop would "
        "open an en passant window over the piece it jumped (first build)"
    ]


@pytest.mark.xfail(strict=True, reason="U7: the core reads the en passant piece's own hop over a "
                   "run as a double step and opens a window over the piece it jumped")
def test_the_en_passant_piece_s_own_hop_opens_no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)
    _windows_over_empty_squares(_ep_runs(pawn_hops=True), seeds=12)


# ---------------------------------------------------------------------------- pass.P5


def _placing_pass() -> dict[str, Any]:
    spec = _load("archive/placement-chess.json")
    spec["rules"]["pass"] = True
    return spec


def test_the_pass_beside_a_placement_phase_stays_refused() -> None:
    assert "pass.P5" in refused_ids(validate_spec(_placing_pass()))


@pytest.mark.xfail(strict=True, reason="U3: the core appends the pass after every filter, a "
                   "placement phase's included, so a player still placing may pass")
def test_no_pass_is_offered_while_a_player_is_placing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)
    engine = Engine(_placing_pass())
    state = engine.initial_state()
    assert any(state.hands[0])
    assert not [m for m in engine.legal_moves(state) if move_type(m) == MOVE_PASS]


# ------------------------------------------------------------------------ Fairy-Stockfish


@pytest.mark.parametrize(
    "build",
    [
        _income_points,
        lambda: _bank_beside("extinction"),
        lambda: _bank_beside("bare_royal"),
        lambda: (lambda s: (s["pieces"][-1].update(price=3), s["rules"].update(purse={"start": 7}), s)[-1])(_walk_spec()),
        lambda: (lambda s: (s["rules"].update(territory={"mode": "repaint"}), s)[-1])(_load("rulesets/leapfrog.json")),
        lambda: _ep_runs(pawn_hops=False),
    ],
    ids=["income.I1", "goals.extinction", "goals.bare_royal", "shared.price", "run.territory", "run.en_passant"],
)
def test_fairy_stockfish_refuses_the_lifted_pair_by_a_rule_of_it(build: Any) -> None:
    with pytest.raises(Untranslatable):
        translate(build())
