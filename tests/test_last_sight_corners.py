"""F10c (Q752c): the last corners of `rust/src/sight.rs` and of the boundary's player calls
(`rust/src/lib.rs`) that no test reached, held to the declared oracle or a hand count.

`docs/plan-tests.md` F10c lists them from Q752b's map (`runs/q752b/after-classified.txt`). The
sight's: the attack test's capture-matrix skip and its sweep arm (`attacks_in_sight_game`), the
placement phase that leaves no commanded move (`push_commanded`), `sight_of` on an empty square,
and the filter's reasons for a turn in place and for a sweep (`SightBarring::reason`). The
boundary's: a position handed over with its folds left out (`folds_in`), each player call asked
where nobody can move (`select_move`, `select_move_depth`, `select_round`).

Each sight case is chess with one more piece (`test_sight.chess_with`) and a sight that can bind,
played from a hand-built position, and held move list by move list to `tests/sight_oracle.py`
where the oracle reads the rule. The oracle reads a sweep (`MOVE_SWEEP`, schema 13), which takes
what it passes, since Q1021: a capture the disarm bars and `must_capture` counts, an attack on
every square it passes, and a move refused when it sweeps the mover's only royal. The sweep cases
are held to it list by list, with the hand counts kept as extra assertions (in check where a
sweep passes the royal, as the sight-free twin's own attack test, `attacks_square`, a separate
walk, also says; barred to a disarmed sweeper, as `docs/archive/plan-passive.md` §2's disarm says).

`effects_at`'s refusal of an empty square or another player's piece (`sight.rs` 57) is not
tested: every caller asks it about a piece it has just read off the board as that player's
(`attacks_in_sight_game` the occupant's owner or the attacking source's own piece,
`effects_report` and `SightBarring::shielded` the occupant's owner, `push_commanded` and
`SightBarring::new` a square of that player's occupancy set), so no position reaches it.
Nor is `lib.rs` 296, the `#[pymethods]` line: its only regions belong to the generated
`__pymethod_initial_state__` and `__pymethod_net_inputs__` wrappers, whose counters read 0 in an
instrumented run where `Core.initial_state` itself ran 129 times (`runs/q752c/new.json`), so
no call can mark it.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from fold_oracle import FoldOracle
from sight_oracle import twin_spec
from test_schema_twenty_four import _agrees, _gardner_folds
from test_sight import KING_STEP, chess_with, engine, legal, oracle, place, sq, warden

from chess_variant_sandbox.agents.base import native_spec
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import SIGHT_DISARM, SIGHT_FREEZE
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.moves import (
    MOVE_COMMANDED,
    MOVE_DROP,
    MOVE_FOLD,
    MOVE_ROTATE,
    MOVE_SWEEP,
    decode_move,
)
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.tournament.play import eval_values

DISARM_BESIDE = dict(KING_STEP, effect="disarm", whose="enemy")
FREEZE_BESIDE = dict(KING_STEP, effect="freeze", whose="enemy")

# A rook-like piece that may take only pawns: it attacks along its lines but never a king.
CATAPULT = {
    "id": "C", "name": "catapult", "value": 4,
    "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}],
    "captures": ["P"],
}
# A rook-like sweeper (schema 13): every square of its line is a landing, and what it passes
# is taken.
SWEEPER = {
    "id": "Z", "name": "sweeper", "value": 4,
    "moves": [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}],
}
# The same sweeper that lands only on empty squares (`mode: "move"`): it never lands on a piece,
# so it attacks one only by sweeping over it to a landing beyond.
LEAPING_SWEEPER = {
    "id": "Z", "name": "sweeper", "value": 4,
    "moves": [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"}],
}
# A piece that steps forward and turns to face any of four ways (schema 8).
LOOKOUT = {
    "id": "L", "name": "lookout", "value": 2,
    "moves": [{"kind": "leap", "vectors": [[0, 1]]}],
    "facing": {"axes": 4},
}


def _valid(spec: dict[str, Any]) -> dict[str, Any]:
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _with(sight: dict[str, Any], piece: dict[str, Any]) -> dict[str, Any]:
    """Chess with a warden carrying ``sight`` (on c3/c6) and ``piece`` (on f3/f6)."""
    spec = chess_with(warden(sight))
    spec["pieces"].append(copy.deepcopy(piece))
    spec["setup"]["placements"].append({"piece": piece["id"], "squares": ["f3"]})
    return _valid(spec)


MATRIX = _with(FREEZE_BESIDE, CATAPULT)
SWEEPS = _with(DISARM_BESIDE, SWEEPER)
TURNS = _with(DISARM_BESIDE, LOOKOUT)


def _kinds(moves: list[int], origin: int) -> list[int]:
    return [decode_move(m)[2] for m in moves if decode_move(m)[0] == origin]


# ------------------------------------------------------------------ the attack test's arms


def test_a_piece_that_may_not_take_the_king_gives_no_check_in_a_sight_game() -> None:
    """`attacks_in_sight_game` skips an attacker the capture matrix forbids the victim: Black's
    king on e8 stands on the open line of White's catapult on e2, which may take only pawns,
    so Black is not in check and the king may stay on the file; held to the oracle. A pawn on
    the same file one square further back is attacked, which is the line the skip spares."""
    state = place(MATRIX, {"e2": ("C", 0), "e8": ("K", 1), "a1": ("K", 0)}, side=1)
    assert not engine(MATRIX).in_check(state, 1)
    moves = legal(MATRIX, state)
    assert (sq(MATRIX, "e8"), sq(MATRIX, "e7")) in {decode_move(m)[:2] for m in moves}
    pawn = place(MATRIX, {"e2": ("C", 0), "e7": ("P", 1), "h8": ("K", 1), "a1": ("K", 0)}, side=1)
    assert oracle(MATRIX).attacked(pawn, 0, sq(MATRIX, "e7"))


def test_a_sweep_that_passes_the_king_checks_it_in_a_sight_game() -> None:
    """`attacks_in_sight_game`'s sweep arm (`sweep_attacks`): White's sweeper on e2 sweeps up
    the e-file over Black's pawn on e6, so it attacks the king on e8 though a piece stands
    between; a plain line test would call the king safe. Held to the oracle (`legal` compares
    the lists, `royal_attacked` the check); the sight-free twin's own attack walk
    (`attacks_square`) says the same, and the king's only escapes leave the file."""
    pieces = {"e2": ("Z", 0), "e6": ("P", 1), "e8": ("K", 1), "a1": ("K", 0)}
    state = place(SWEEPS, pieces, side=1)
    assert engine(SWEEPS).in_check(state, 1)
    assert oracle(SWEEPS).royal_attacked(state, 1)
    twin = Engine(twin_spec(SWEEPS))
    assert twin.in_check(state, 1)
    moves = legal(SWEEPS, state)
    king_moves = {decode_move(m)[1] for m in moves if decode_move(m)[0] == sq(SWEEPS, "e8")}
    assert king_moves == {sq(SWEEPS, name) for name in ("d8", "f8", "d7", "f7")}


def test_a_disarmed_or_frozen_sweeper_gives_no_check_in_a_sight_game() -> None:
    """The check above, with White's sweeper bound by a Black warden beside it (d3 for e2): a
    disarmed sweeper attacks nothing (its sweep over the pawn is no longer a move), and a frozen
    one neither. Black is not in check and may step along the file; held to the oracle."""
    for effect, other in (("disarm", DISARM_BESIDE), ("freeze", FREEZE_BESIDE)):
        spec = _with(other, SWEEPER)
        pieces = {"e2": ("Z", 0), "d3": ("W", 1), "e6": ("P", 1), "e8": ("K", 1), "a1": ("K", 0)}
        state = place(spec, pieces, side=1)
        assert engine(spec).effects(state).bits[sq(spec, "e2")] == (
            SIGHT_DISARM if effect == "disarm" else SIGHT_FREEZE)
        assert not engine(spec).in_check(state, 1)
        assert not oracle(spec).royal_attacked(state, 1)
        targets = {decode_move(m)[1] for m in legal(spec, state)
                   if decode_move(m)[0] == sq(spec, "e8")}
        assert sq(spec, "e7") in targets


def test_a_check_is_answered_by_a_sweep_that_takes_the_checker_over_a_piece() -> None:
    """Black's sweeper on e8 checks White's king on e1 over White's pawn on e4. White's sweeper
    on h8 sweeps the eighth rank over Black's pawn on g8 and lands on e8, taking the checker:
    a `MOVE_SWEEP` that answers the check. Core and oracle hold the same list, and every
    answer is one the board it makes allows (the king's steps, the sweep, the pawn's block is
    none: it stands on the file already)."""
    spec = SWEEPS
    pieces = {"e8": ("Z", 1), "e4": ("P", 0), "e1": ("K", 0), "h8": ("Z", 0), "g8": ("P", 1),
              "a8": ("K", 1)}
    state = place(spec, pieces)
    assert engine(spec).in_check(state, 0) and oracle(spec).royal_attacked(state, 0)
    moves = legal(spec, state)
    answer = (sq(spec, "h8"), sq(spec, "e8"), MOVE_SWEEP)
    assert answer in {decode_move(m)[:3] for m in moves}
    assert all(not oracle(spec).royal_attacked(engine(spec).apply(state, m), 0) for m in moves)


def test_a_sweep_over_its_own_king_is_held_to_the_oracle() -> None:
    """A sweep takes friendly pieces too, so White's sweeper on a1 landing on f1 with White's
    king on c1 between would leave White without a king. Whatever the core says of it, the
    oracle (which refuses a move that leaves the mover no royal) must say the same. The
    sight-free twin offers the sweeps (checkmate read as royal capture); neither list holds
    one, while the sweeper's plain slide to b1 and the king's own steps stand."""
    spec = SWEEPS
    origin = sq(spec, "a1")
    state = place(SWEEPS, {"a1": ("Z", 0), "c1": ("K", 0), "h8": ("K", 1)})
    twin = Engine(twin_spec(spec))
    over_king = {decode_move(m)[1] for m in twin.legal_moves(state)
                 if decode_move(m)[0] == origin and decode_move(m)[2] == MOVE_SWEEP}
    assert over_king == {sq(spec, name) for name in ("d1", "e1", "f1", "g1", "h1")}
    moves = legal(spec, state)
    assert not [m for m in moves if decode_move(m)[0] == origin and decode_move(m)[2] == MOVE_SWEEP]
    assert (origin, sq(spec, "b1")) in {decode_move(m)[:2] for m in moves}


def test_a_sweep_that_lands_beyond_the_king_checks_it_unless_the_sweeper_is_bound() -> None:
    """A sweeper that lands only on empty squares never takes the king where it stands, yet its
    sweep from e2 to e7 passes over a king on e6 and that is check (`sweep_attacks`' landing
    beyond), which the oracle reads off the sweep's path, not its landing. A Black warden on d3
    disarms the sweeper and the check is gone; a freezing one the same. Core and oracle agree."""
    for bound in (None, DISARM_BESIDE, FREEZE_BESIDE):
        spec = _with(bound or DISARM_BESIDE, LEAPING_SWEEPER)
        pieces = {"e2": ("Z", 0), "e6": ("K", 1), "a1": ("K", 0)}
        if bound:
            pieces["d3"] = ("W", 1)
        state = place(spec, pieces, side=1)
        assert engine(spec).in_check(state, 1) is (bound is None)
        assert oracle(spec).royal_attacked(state, 1) is (bound is None)
        moves = legal(spec, state)
        stays = (sq(spec, "e6"), sq(spec, "e5")) in {decode_move(m)[:2] for m in moves}
        assert stays is (bound is not None)


def _compulsory(spec: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    spec["rules"]["must_capture"] = True
    return _valid(spec)


def test_a_sweep_is_a_capture_under_must_capture_and_a_disarmed_one_is_not_offered() -> None:
    """`must_capture` counts a sweep over a piece as a capture even onto an empty square. White's
    sweeper on e2 can sweep Black's pawn on e5 (landing on e6, e7 or e8) or take it on e5; White's
    knight on b1 has quiet moves. Only the sweeper's captures are legal. With a Black warden
    beside the sweeper on d3 disarming it, no capture is on offer and the quiet moves return.
    Core and oracle hold the same lists."""
    spec = _compulsory(SWEEPS)
    pieces = {"e2": ("Z", 0), "e5": ("P", 1), "b1": ("N", 0), "a1": ("K", 0), "h8": ("K", 1)}
    moves = legal(spec, place(spec, pieces))
    assert {decode_move(m)[2] for m in moves} == {MOVE_SWEEP, 0}
    assert {decode_move(m)[0] for m in moves} == {sq(spec, "e2")}
    assert {decode_move(m)[1] for m in moves} == {sq(spec, name) for name in ("e5", "e6", "e7", "e8")}
    free = legal(spec, place(spec, pieces | {"d3": ("W", 1)}))
    assert not any(decode_move(m)[2] == MOVE_SWEEP for m in free)
    assert {decode_move(m)[0] for m in free} > {sq(spec, "b1"), sq(spec, "a1")}


def _walk(spec: dict[str, Any], games: int, plies: int) -> tuple[int, int]:
    """Seeded random games of ``spec`` through the core, each list and each check answer held to
    the oracle; the positions compared and the sweeps seen on the way."""
    judge = oracle(spec)
    eng = judge.engine
    compared = sweeps = 0
    for seed in range(games):
        rng = random.Random(seed)
        state = eng.initial_state()
        for _ in range(plies):
            if eng.terminal(state) is not None:
                break
            moves = eng.legal_moves(state)
            assert moves == judge.legal_moves(state), (seed, state.ply)
            assert eng.in_check(state) == judge.royal_attacked(state, state.side_to_move)
            compared += 1
            sweeps += sum(decode_move(m)[2] == MOVE_SWEEP for m in moves)
            if not moves:
                break
            state = eng.apply(state, rng.choice(moves))
    return compared, sweeps


@pytest.mark.parametrize("build", [
    lambda: SWEEPS, lambda: _with(FREEZE_BESIDE, SWEEPER), lambda: _compulsory(SWEEPS),
], ids=["disarm", "freeze", "must_capture"])
def test_a_sweep_beside_a_sight_agrees_with_the_oracle_over_random_games(build: Any) -> None:
    """The oracle's sweep arms (the capture, the attack over a passed square, the royal lost to
    a sweep over one's own king) held to the core over seeded random games: the walk compares
    every list and every check answer, and sees real sweeps and real sights."""
    compared, sweeps = _walk(build(), games=4, plies=80)
    assert compared > 100 and sweeps > 20


# ------------------------------------------------------------------ the filter's reasons


def test_a_disarmed_piece_may_still_turn_in_place() -> None:
    """`SightBarring::reason` lets a turn through: White's lookout on c4 stands beside Black's
    disarming warden on c5, so it may not take the warden, but each of its turns in place is
    still offered (a turn takes nothing; read as a landing on its own square it would be
    barred as a capture). Held to the oracle; `effects` counts the one capture removed."""
    state = place(TURNS, {"c4": ("L", 0), "c5": ("W", 1), "a1": ("K", 0), "h8": ("K", 1)})
    eng = engine(TURNS)
    assert eng.effects(state).bits[sq(TURNS, "c4")] == SIGHT_DISARM
    moves = legal(TURNS, state)
    kinds = _kinds(moves, sq(TURNS, "c4"))
    assert kinds.count(MOVE_ROTATE) == 3 and len(kinds) == 3
    assert eng.effects(state).disarmed == 1


def test_a_disarmed_sweeper_may_slide_but_not_sweep() -> None:
    """`SightBarring::reason` on a sweep: White's sweeper on e2 stands beside Black's disarming
    warden on d3, and Black's pawn on e5 stands on its file. Its plain slides to e3 and e4 and
    along the second rank stay; every sweep over the pawn (a capture, though it lands on an
    empty square) and the landing on the pawn go. Beside it, the same sweeper undisarmed
    keeps its sweeps while another White piece is disarmed (so the filter still runs). The
    oracle (`legal` holds the two lists equal) bars a sweep as a capture, and by hand: the
    twin's sweeps from e2 are gone."""
    eng = engine(SWEEPS)
    origin = sq(SWEEPS, "e2")
    pieces = {"e2": ("Z", 0), "d3": ("W", 1), "e5": ("P", 1), "a1": ("K", 0), "h8": ("K", 1)}
    state = place(SWEEPS, pieces)
    assert eng.effects(state).bits[origin] == SIGHT_DISARM
    core = legal(SWEEPS, state)
    twins = Engine(twin_spec(SWEEPS)).legal_moves(state)
    swept = [m for m in twins if decode_move(m)[0] == origin and decode_move(m)[2] == MOVE_SWEEP]
    assert swept, "the sight-free twin offers the sweeps the disarm must remove"
    assert not [m for m in core if m in swept]
    assert eng.effects(state).disarmed == len(swept) + 1  # every sweep and the landing on e5
    targets = {decode_move(m)[1] for m in core if decode_move(m)[0] == origin}
    assert targets == {sq(SWEEPS, name) for name in ("e3", "e4", "e1", "d2", "c2", "b2", "a2", "f2", "g2", "h2")}
    free = place(SWEEPS, {"e2": ("Z", 0), "b5": ("N", 0), "b6": ("W", 1), "e5": ("P", 1),
                          "a1": ("K", 0), "h8": ("K", 1)})
    assert eng.effects(free).bits[sq(SWEEPS, "b5")] == SIGHT_DISARM
    assert eng.effects(free).bits[origin] == 0
    moves = legal(SWEEPS, free)
    assert MOVE_SWEEP in _kinds(moves, origin)


# ------------------------------------------------------------------ the command and sight_of


def _commanding_placement() -> dict[str, Any]:
    """Chess with commanding knights (`tests/_command_k1.json`'s sight) and a queen a side
    placed from hand before anything moves (`drops.placement`)."""
    spec = chess_with(sights={"N": [{"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1,
                                     "effect": "command", "whose": "enemy"}]})
    spec["rules"].pop("castling", None)
    spec["rules"]["drops"] = {"from_captures": False, "placement": True, "stock": {"Q": 1}}
    return _valid(spec)


PLACEMENT = _commanding_placement()


def test_a_placement_phase_offers_no_commanded_move() -> None:
    """`push_commanded` adds nothing while the side to move still has a piece to place: White's
    knight on d4 commands Black's pawn on e6, but White holds a queen, so its only moves are
    the queen's drops (held to the oracle). With the hand empty the command is offered."""
    pieces = {"d4": ("N", 0), "e6": ("P", 1), "a1": ("K", 0), "h8": ("K", 1)}
    state = place(PLACEMENT, pieces)
    assert any(state.hands[0])
    moves = legal(PLACEMENT, state)
    assert moves and all(decode_move(m)[2] == MOVE_DROP for m in moves)
    assert not any(m & MOVE_COMMANDED for m in moves)
    emptied = state.with_(hands=tuple(tuple(0 for _ in hand) for hand in state.hands))
    emptied = emptied.with_(key=engine(PLACEMENT)._compute_hash(emptied))
    after = legal(PLACEMENT, emptied)
    assert any(m & MOVE_COMMANDED and decode_move(m)[0] == sq(PLACEMENT, "e6") for m in after)


def test_sight_of_an_empty_square_is_empty() -> None:
    """`Core.sight_of` on an empty square of a sight game answers nothing, and on the warden
    on a warden the eight squares around it, each with the freeze bit, empty or not."""
    state = place(MATRIX, {"d4": ("W", 0), "e5": ("P", 1), "a1": ("K", 0), "h8": ("K", 1)})
    eng = engine(MATRIX)
    assert eng.sight_of(state, sq(MATRIX, "c6")) == []
    seen = dict(eng.sight_of(state, sq(MATRIX, "d4")))
    around = ("c3", "d3", "e3", "c4", "e4", "c5", "d5", "e5")
    assert seen == {sq(MATRIX, name): SIGHT_FREEZE for name in around}


# ------------------------------------------------------------------ the boundary's calls


def test_a_position_handed_over_without_its_folds_has_them_where_they_start() -> None:
    """`folds_in`: a fold game's position whose folds are left out (`()`) is read with each
    fold where it starts, so it has the same moves and key as the position that names them,
    the fold ends' own moves among them; held to the fold oracle."""
    spec = _gardner_folds(moved_by={"orange": [0, 1], "blue": [0, 1]})
    eng = Engine(spec)
    named = eng.initial_state()
    assert named.folds
    bare = named.with_(folds=())
    moves = eng.legal_moves(bare)
    assert moves == eng.legal_moves(named)
    assert any(decode_move(m)[2] == MOVE_FOLD for m in moves)
    assert eng.core.compute_hash(_pack(bare)) == named.key
    _agrees(eng, FoldOracle(spec), named)


CHESS_ENGINE = Engine(copy.deepcopy(chess_with()))


@pytest.mark.parametrize("agent", ["random", "greedy", "minimax:nodes=200"])
def test_no_legal_move_is_no_choice(agent: str) -> None:
    """`select_move` and `select_move_depth` answer `None` (and depth 0) for a mated side: Black's
    king on h8 under White's queen on g7, guarded by the king on f6, has no legal move, so no
    player is asked."""
    eng = CHESS_ENGINE
    state = eng.make_state({"h8": ("K", 1), "g7": ("Q", 0), "f6": ("K", 0)}, side_to_move=1)
    assert eng.legal_moves(state) == [] and eng.terminal(state) is not None
    native = native_spec(agent, eng.spec.hash)
    assert native is not None
    values = eval_values(eng.spec)
    assert eng.select_move(state, native, values, 3) is None
    assert eng.select_move_depth(state, native, values, 3) == (None, 0)


ROUNDS = _valid(json.loads(
    (Path(__file__).parent / "_retired_gardner_in_rounds.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize("seat", [0, 1])
def test_a_round_where_one_side_cannot_move_is_no_choice(seat: int) -> None:
    """`select_round` answers `(None, [], 0, 0)` for either seat when one player has no move:
    Black's only piece, a pawn on c4, is blocked by White's on c3 with nothing to take."""
    eng = Engine(ROUNDS)
    state = eng.make_state({"a1": ("K", 0), "c3": ("P", 0), "c4": ("P", 1)})
    assert eng.legal_moves_for(state, 1) == [] and eng.legal_moves_for(state, 0)
    native = native_spec("matrix:nodes=50", eng.spec.hash)
    assert native is not None
    found = eng.select_round(state, seat, native, eval_values(eng.spec), 5)
    assert (found[0], list(found[1]), found[2], found[3]) == (None, [], 0, 0)

