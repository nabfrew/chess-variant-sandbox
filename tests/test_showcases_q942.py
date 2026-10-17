"""The chase remedies' showcases (Q942c), held to criterion K8 of `docs/plan-round-chase.md` §5.

K8, *the mechanism, without players*, written before this file and before any showcase was
measured: on hand-built positions and on seeded random play (20 games, `random.Random(seed)`
choosing from `legal_moves_for`), S1 (`archive/gardner-in-rounds-resting`): no player's piece
moves two rounds running except under the lift; S2 (`...-where-swaps-collide`): no round ends
with two pieces on each other's origins; S3 (`...-scored`): no game ends drawn with both
royals taken and unequal material; S4 (`...-against-the-chase`): all three. The §2 positions
play as §3 says under each rule.

Every answer is the core's, through `Engine`; the clauses only compare squares and owners on
the boards it returns. The material count in S3's clause is a declared test oracle: the sum
of the declared piece values, the comparison `rules.scoring` `material` names.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from simultaneous_showcase_tally import is_swap

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec

ROOT = Path(__file__).parent.parent
SEEDS = range(20)
MAX_PLIES = 200


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))


PARENT = load("gardner-in-rounds")
SHOWCASES = {
    "S1": load("gardner-in-rounds-resting"),
    "S2": load("gardner-in-rounds-where-swaps-collide"),
    "S3": load("gardner-in-rounds-scored"),
    "S4": load("gardner-in-rounds-against-the-chase"),
}
SWITCHES = {
    "S1": {"rest": True},
    "S2": {"swap": "collision"},
    "S3": {"both": "score"},
    "S4": {"rest": True, "swap": "collision", "both": "score"},
}
# How many of the 20 random games end with both kings taken in one round (counted, pinned).
BOTH_IN_RANDOM_PLAY = {"S3": 2, "S4": 1}
RESTING = ("S1", "S4")
COLLIDING = ("S2", "S4")
SCORED = ("S3", "S4")

# The lock of §2 (game 217 of runs/e043fce6968e-1000, round 37). White is player 0.
LOCK = {
    "e1": ("K", 0), "b5": ("Q", 0), "c1": ("B", 0), "d2": ("P", 0), "e3": ("P", 0),
    "b4": ("K", 1), "a1": ("Q", 1), "b3": ("R", 1), "d3": ("P", 1), "e4": ("P", 1),
}


@pytest.fixture(scope="module", params=sorted(SHOWCASES))
def showcase(request: pytest.FixtureRequest) -> tuple[str, Engine]:
    return request.param, Engine(SHOWCASES[request.param])


@pytest.fixture(scope="module")
def engines() -> dict[str, Engine]:
    return {name: Engine(spec) for name, spec in SHOWCASES.items()}


def square(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def find(engine: Engine, state: GameState, player: int, origin: str, target: str) -> int:
    for move in engine.legal_moves_for(state, player):
        o, t, _kind, _promotion = decode_move(move)
        if o == square(engine, origin) and t == square(engine, target):
            return move
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def place(engine: Engine, pieces: dict[str, tuple[str, int]]) -> GameState:
    state = engine.make_state(pieces)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def owner_at(engine: Engine, state: GameState, at: int) -> int | None:
    value = state.board[at]
    return None if value == 0 else engine.spec.owner_of(value)


VALUES = {piece["id"]: piece.get("value", 0) for piece in PARENT["pieces"]}


def material(engine: Engine, state: GameState, player: int) -> float:
    """The declared values of `player`'s pieces on the board (the test's oracle, see above)."""
    ids = engine.spec.piece_ids
    return sum(
        VALUES[ids[engine.spec.type_of(v)]]
        for v in state.board
        if v and engine.spec.owner_of(v) == player
    )


def royals_left(engine: Engine, state: GameState, player: int) -> int:
    return sum(
        1 for v in state.board
        if v and engine.spec.owner_of(v) == player and engine.spec.type_of(v) in engine.spec.royal_types
    )


Visit = Callable[[GameState, int, int, GameState], None]


def random_games(
    engine: Engine, visit: Visit, start: Callable[[], None] | None = None
) -> list[tuple[GameState, Any]]:
    """Seeds 0 to 19 of random play, at most 200 plies; `start()` opens each game and
    `visit(before, first, second, after)` sees every round. Returns each game's last
    position and outcome (None if cut)."""
    ends = []
    for seed in SEEDS:
        if start is not None:
            start()
        rng = random.Random(seed)
        state = engine.initial_state()
        plies = 0
        outcome = engine.terminal(state)
        while outcome is None and plies < MAX_PLIES:
            first = rng.choice(engine.legal_moves_for(state, 0))
            second = rng.choice(engine.legal_moves_for(state, 1))
            after = engine.apply_round(state, first, second)
            visit(state, first, second, after)
            state, plies = after, plies + 2
            outcome = engine.terminal(state)
        ends.append((state, outcome))
    return ends


# ------------------------------------------------------------------ the files


def test_each_showcase_is_its_parent_with_its_switches_and_nothing_else() -> None:
    for name, spec in SHOWCASES.items():
        assert validate_spec(spec) == [], name
        assert spec["schema_version"] == 27
        assert spec["rules"]["simultaneous"] == SWITCHES[name]
        rebuilt = json.loads(json.dumps(PARENT))
        rebuilt["schema_version"] = 27
        rebuilt["rules"]["simultaneous"] = dict(SWITCHES[name])
        if name in SCORED:
            rebuilt["rules"]["scoring"] = {"count": "material"}
        assert spec_hash(spec) == spec_hash(rebuilt), name
    assert len({spec_hash(spec) for spec in SHOWCASES.values()} | {spec_hash(PARENT)}) == 5


def test_s1_and_s4_are_the_test_specs_the_core_was_checked_on() -> None:
    """Q942a and Q942b proved the core and the oracle on these two test specs."""
    tests = ROOT / "tests"
    resting = json.loads((tests / "_simultaneous_resting.json").read_text(encoding="utf-8"))
    scored = json.loads((tests / "_simultaneous_scored.json").read_text(encoding="utf-8"))
    assert spec_hash(SHOWCASES["S1"]) == spec_hash(resting)
    assert spec_hash(SHOWCASES["S4"]) == spec_hash(scored)


# ------------------------------------------------------- S1, S4: the rest, in play


@pytest.mark.parametrize("name", RESTING)
def test_no_piece_moves_two_rounds_running_except_under_the_lift(
    engines: dict[str, Engine], name: str
) -> None:
    engine = engines[name]
    # Per player, the square the piece that made its last move stands on (None: gone).
    last: list[int | None] = [None, None]
    counts = {"barred": 0, "lifted": 0}

    def visit(before: GameState, first: int, second: int, after: GameState) -> None:
        for player, move in ((0, first), (1, second)):
            legal = engine.legal_moves_for(before, player)
            free = engine.legal_moves_for(before.with_(rest=(-1, -1)), player)
            mover = last[player]
            if mover is not None and move_from(move) == mover:
                # The same piece again: only when every legal move is that piece's.
                assert all(move_from(m) == mover for m in legal), (name, player)
                counts["lifted"] += 1
            if mover is not None and any(move_from(m) == mover for m in free) and any(
                move_from(m) != mover for m in free
            ):
                assert not any(move_from(m) == mover for m in legal)
                counts["barred"] += 1
            origin, target = move_from(move), move_to(move)
            if owner_at(engine, after, target) == player:
                last[player] = target
            elif owner_at(engine, after, origin) == player:
                last[player] = origin  # void: it stood
            else:
                last[player] = None

    def start() -> None:
        last[0] = last[1] = None

    random_games(engine, visit, start)
    # The rule bites in play, not only on paper: a piece that would have had a move was
    # refused it in many rounds.
    assert counts["barred"] >= 200, counts


def test_the_lift_offers_a_lone_king_its_moves(engines: dict[str, Engine]) -> None:
    for name in RESTING:
        engine = engines[name]
        state = place(engine, {"c1": ("K", 0), "c5": ("K", 1), "a5": ("R", 1)})
        after = engine.apply_round(
            state, find(engine, state, 0, "c1", "c2"), find(engine, state, 1, "a5", "a4")
        )
        assert after.rest[0] == square(engine, "c2")
        moves = engine.legal_moves_for(after, 0)
        assert moves and all(move_from(m) == square(engine, "c2") for m in moves), name
        # Black's rook rests and its king is offered every move.
        black = engine.legal_moves_for(after, 1)
        assert black and all(move_from(m) == square(engine, "c5") for m in black), name


@pytest.mark.parametrize("name", RESTING)
def test_the_lock_of_section_two_rests_both_pieces(engines: dict[str, Engine], name: str) -> None:
    engine = engines[name]
    state = place(engine, LOCK)
    # S1 plays round 37 as recorded, the swap (the two pass); under S4 a swap meets, so the
    # lock is played as the queen striking and the king running (a miss).
    queen_to, king_to = ("b4", "b5") if name == "S1" else ("b4", "c4")
    after = engine.apply_round(
        state, find(engine, state, 0, "b5", queen_to), find(engine, state, 1, "b4", king_to)
    )
    assert engine.terminal(after) is None
    assert after.rest == (square(engine, queen_to), square(engine, king_to))
    white, black = engine.legal_moves_for(after, 0), engine.legal_moves_for(after, 1)
    assert white and black
    assert all(move_from(m) != square(engine, queen_to) for m in white)
    assert all(move_from(m) != square(engine, king_to) for m in black)


# ------------------------------------------------------ S2, S4: a swap meets, in play


@pytest.mark.parametrize("name", COLLIDING)
def test_no_round_ends_with_two_pieces_on_each_others_origins(
    engines: dict[str, Engine], name: str
) -> None:
    engine = engines[name]
    swaps = [0]

    def visit(before: GameState, first: int, second: int, after: GameState) -> None:
        if not is_swap(first, second):
            return
        swaps[0] += 1
        passed = (
            owner_at(engine, after, move_to(first)) == 0
            and owner_at(engine, after, move_to(second)) == 1
        )
        assert not passed, (name, decode_move(first), decode_move(second))

    random_games(engine, visit)
    assert swaps[0] >= 5, swaps


def test_in_the_parent_the_same_swaps_pass() -> None:
    """The clause is not vacuous: without the switch two swapping pieces both stand."""
    engine = Engine(PARENT)
    passes = [0]

    def visit(before: GameState, first: int, second: int, after: GameState) -> None:
        if is_swap(first, second) and owner_at(engine, after, move_to(first)) == 0 and owner_at(
            engine, after, move_to(second)
        ) == 1:
            passes[0] += 1

    random_games(engine, visit)
    assert passes[0] >= 1, passes


@pytest.mark.parametrize("name", COLLIDING)
def test_the_lock_of_section_two_meets_and_white_wins(engines: dict[str, Engine], name: str) -> None:
    engine = engines[name]
    state = place(engine, LOCK)
    after = engine.apply_round(
        state, find(engine, state, 0, "b5", "b4"), find(engine, state, 1, "b4", "b5")
    )
    assert after.board[square(engine, "b4")] == 0 and after.board[square(engine, "b5")] == 0
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0 and outcome.reason == "royal_captured"


@pytest.mark.parametrize("name", COLLIDING)
def test_two_bare_kings_striking_each_other_end_drawn_at_once(
    engines: dict[str, Engine], name: str
) -> None:
    engine = engines[name]
    state = place(engine, {"e1": ("K", 0), "d2": ("K", 1)})
    after = engine.apply_round(
        state, find(engine, state, 0, "e1", "d2"), find(engine, state, 1, "d2", "e1")
    )
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner is None and outcome.reason == "royal_captured"


# --------------------------------------------- S3, S4: a double capture is scored


@pytest.mark.parametrize("name", SCORED)
def test_no_game_ends_drawn_with_both_royals_taken_and_unequal_material(
    engines: dict[str, Engine], name: str
) -> None:
    engine = engines[name]
    both = 0

    def ignore(before: GameState, first: int, second: int, after: GameState) -> None:
        return None

    for state, outcome in random_games(engine, ignore):
        if outcome is None or royals_left(engine, state, 0) or royals_left(engine, state, 1):
            continue
        both += 1
        mine, theirs = material(engine, state, 0), material(engine, state, 1)
        if mine == theirs:
            assert outcome.winner is None
        else:
            assert outcome.winner == (0 if mine > theirs else 1), (mine, theirs, outcome)
    # Random play takes both kings together rarely (BOTH_IN_RANDOM_PLAY counts it); the
    # hand-built clause below is the one that must fire.
    assert both == BOTH_IN_RANDOM_PLAY[name]


@pytest.mark.parametrize("name", SCORED)
@pytest.mark.parametrize(("extra", "winner"), [({"c3": ("N", 0)}, 0), ({"c3": ("N", 1)}, 1), ({}, None)])
def test_two_royals_taken_together_go_to_the_side_ahead(
    engines: dict[str, Engine], name: str, extra: dict[str, tuple[str, int]], winner: int | None
) -> None:
    engine = engines[name]
    state = place(engine, {"a1": ("K", 0), "e1": ("R", 0), "e5": ("K", 1), "a5": ("R", 1), **extra})
    after = engine.apply_round(
        state, find(engine, state, 0, "e1", "e5"), find(engine, state, 1, "a5", "a1")
    )
    assert not royals_left(engine, after, 0) and not royals_left(engine, after, 1)
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == winner and outcome.reason == "royal_captured"


def test_in_the_parent_the_same_double_capture_draws() -> None:
    engine = Engine(PARENT)
    state = place(engine, {"a1": ("K", 0), "e1": ("R", 0), "c3": ("N", 0), "e5": ("K", 1),
                           "a5": ("R", 1)})
    after = engine.apply_round(
        state, find(engine, state, 0, "e1", "e5"), find(engine, state, 1, "a5", "a1")
    )
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner is None
