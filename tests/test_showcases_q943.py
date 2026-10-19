"""W1 of `docs/plan-speed.md` §6.3: what speed does in its three showcases, without players.

The showcases (Q943c, §6.1): T1 `archive/gardner-in-rounds-with-a-swift-king` (the king at
speed 1), T2 `archive/gardner-in-rounds-with-swift-knights` (the knights at 1), T3
`archive/watering-hole-by-speed` (cat 3, rat 2, dog 1, elephant 0: strictly ordered). W1 is
pre-registered as: on seeded random play (20 games each, `random.Random(seed)` choosing from
`legal_moves_for`, at most 200 plies) no round between unequal speeds is a collision or a swap;
every ordered round makes the faster move as chosen; every slower move is made exactly when its
word is in its generator's list on the board after the faster move (checked against the
oracle); no ordered round that ended after the faster move made the slower one; in T3 no
collision, swap or void round ever. And §2.5's positions play as §2.5 says, in each showcase
where the pieces exist.

The board after the faster move alone is the **alternating twin's** `apply` (the same file
without rounds or speeds), so the slower move's test is read on a board the round code did
not make; the round itself is the core's, held to `tests/simultaneous_oracle.py`. Nothing here
decides a move's legality or a round's outcome (CLAUDE.md, rule 3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from simultaneous_oracle import offered, resolve_round
from simultaneous_showcase_tally import RoundTally
from test_speed import (
    EVENT_COLLISION,
    EVENT_LOST,
    EVENT_ORDER,
    EVENT_SWAP,
    EVENT_UNPLAYED,
    EVENT_VOID,
    alternating_twin,
    kinds,
    play,
    sq,
)

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec

ARCHIVE = Path(__file__).resolve().parent.parent / "archive"
SHOWCASES = {
    "T1": "gardner-in-rounds-with-a-swift-king",
    "T2": "gardner-in-rounds-with-swift-knights",
    "T3": "watering-hole-by-speed",
}
GAMES, MAX_ROUNDS = 20, 100


def spec(label: str) -> dict[str, Any]:
    return load_spec(ARCHIVE / f"{SHOWCASES[label]}.json")


def owner(engine: Engine, state: GameState, name: str) -> int | None:
    value = state.board[sq(engine, name)]
    return None if value == 0 else engine.spec.owner_of(value)


def kind(engine: Engine, state: GameState, name: str) -> str | None:
    value = state.board[sq(engine, name)]
    return None if value == 0 else engine.spec.piece_ids[engine.spec.type_of(value)]


# --------------------------------------------------------- random play, round by round


@dataclass
class Walked:
    rounds: int = 0
    ordered: int = 0
    slower_made: int = 0
    slower_void: int = 0
    slower_lost: int = 0
    unplayed: int = 0
    collisions: int = 0


def check_round(
    engine: Engine, twin: Engine, state: GameState, words: tuple[int, int], walked: Walked
) -> GameState:
    """One random round through the core, held to the oracle and, when ordered, to the
    faster move made alone by the alternating twin."""
    n = engine.spec.n_squares
    after, _phase, events = engine.apply_round_with_phase(state, *words)
    assert after == resolve_round(engine, state, *words)
    seen = kinds(list(events))
    walked.rounds += 1
    walked.collisions += seen.count(EVENT_COLLISION)
    if EVENT_ORDER not in seen:
        assert not {EVENT_LOST, EVENT_UNPLAYED} & set(seen)
        return after
    # An ordered round: never a collision or a swap.
    walked.ordered += 1
    assert seen.count(EVENT_ORDER) == 1
    assert EVENT_COLLISION not in seen and EVENT_SWAP not in seen
    f = next(player for event, _square, player in events if event == EVENT_ORDER)
    s = 1 - f
    # The faster move, made alone as the alternating game makes it.
    alone = state.with_(side_to_move=f, initiative=-1, void_run=(-1, -1), bars=(), rest=(-1, -1))
    middle = twin.apply(alone, words[f])
    middle = state.with_(board=middle.board, hands=middle.hands)
    origin, _target, move_kind, _ = decode_move(words[s])
    ended = engine.terminal_prefix(
        middle.with_(halfmove=0, ply=0, repetition=())
    ) is not None
    stands = move_kind == MOVE_DROP or middle.board[origin] == state.board[origin]
    if ended:
        # The game ended with the faster move: the slower was not made.
        walked.unplayed += 1
        assert EVENT_UNPLAYED in seen
        assert after.board[:n] == middle.board[:n]
    elif not stands:
        walked.slower_lost += 1
        assert EVENT_LOST in seen and after.board[:n] == middle.board[:n]
    elif offered(engine, middle, s, words[s]):
        walked.slower_made += 1
        assert not {EVENT_LOST, EVENT_UNPLAYED, EVENT_VOID} & set(seen)
        made = twin.apply(middle.with_(side_to_move=s, initiative=-1, void_run=(-1, -1)), words[s])
        assert after.board[:n] == made.board[:n]
    else:
        walked.slower_void += 1
        assert EVENT_VOID in seen and after.board[:n] == middle.board[:n]
    assert after.void_run == (-1, -1)
    return after


def walk(label: str) -> Walked:
    engine = Engine(spec(label))
    twin = Engine(alternating_twin(spec(label)))
    walked = Walked()
    for seed in range(GAMES):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(MAX_ROUNDS):
            if engine.terminal(state) is not None:
                break
            words = (
                rng.choice(engine.legal_moves_for(state, 0)),
                rng.choice(engine.legal_moves_for(state, 1)),
            )
            state = check_round(engine, twin, state, words, walked)
    return walked


@pytest.mark.parametrize("label", sorted(SHOWCASES))
def test_w1_every_ordered_round_is_the_faster_move_then_the_slower_if_it_can(label: str) -> None:
    walked = walk(label)
    assert walked.ordered > 0, "speed never ordered a round"
    if label == "T3":
        # Strictly ordered: every round is two moves in order, and nothing ever collides.
        assert walked.ordered == walked.rounds and walked.collisions == 0
    # The slower move is sometimes made and sometimes not, or the rule is decoration.
    assert walked.slower_made > 0
    assert walked.slower_void + walked.slower_lost + walked.unplayed > 0


@pytest.mark.parametrize("label", sorted(SHOWCASES))
def test_the_study_tally_counts_what_the_walk_counts(label: str) -> None:
    """§6.2's reader (`tests/simultaneous_showcase_tally.py`, extended by Q943c) over W1's
    random games counts the ordered rounds and the slower moves void, lost and unplayed as
    the walk does, and in T3 every round is ordered and none a swap or a collision."""
    engine = Engine(spec(label))
    tally = RoundTally()
    for seed in range(GAMES):
        rng = random.Random(seed)
        state = engine.initial_state()
        tally.start_game()
        for _ in range(MAX_ROUNDS):
            if engine.terminal(state) is not None:
                break
            words = (
                rng.choice(engine.legal_moves_for(state, 0)),
                rng.choice(engine.legal_moves_for(state, 1)),
            )
            state, _ = tally.round(engine, state, *words)
        outcome = engine.terminal(state)
        tally.end_game(state.ply, "", None if outcome is None else outcome.winner, False)
    walked = walk(label)
    assert (tally.rounds, tally.ordered) == (walked.rounds, walked.ordered)
    assert (tally.slower_void, tally.slower_lost, tally.ended_after_faster) == (
        walked.slower_void, walked.slower_lost, walked.unplayed
    )
    if label == "T3":
        assert tally.ordered == tally.rounds and tally.swaps == 0 and tally.collisions == []
    assert "ordered rounds" in tally.report(label)


# --------------------------------------------------------- §2.5's cases in each showcase


KINGS = {"e1": ("K", 0), "e5": ("K", 1)}


def test_t1_a_king_runs_before_the_strike_at_where_it_stood() -> None:
    """§2.5 A, turned round: the king is faster, leaves first, and the rook lands on empty a5."""
    pieces = {"a1": ("R", 0), "e1": ("K", 0), "a5": ("K", 1)}
    engine, after, events = play(spec("T1"), pieces, ("a1", "a5"), ("a5", "b4"))
    assert events == [(EVENT_ORDER, sq(engine, "b4"), 1)]
    assert kind(engine, after, "b4") == "K" and kind(engine, after, "a5") == "R"
    assert engine.terminal(after) is None


def test_t1_a_piece_the_king_goes_for_is_taken_before_it_moves() -> None:
    """§2.5 B, turned round: the queen and king strike at each other; the king is faster."""
    pieces = {"b5": ("Q", 0), "e1": ("K", 0), "b4": ("K", 1)}
    engine, after, events = play(spec("T1"), pieces, ("b5", "b4"), ("b4", "b5"))
    assert events == [(EVENT_ORDER, sq(engine, "b5"), 1), (EVENT_LOST, sq(engine, "b5"), 0)]
    assert kind(engine, after, "b5") == "K" and owner(engine, after, "b5") == 1
    assert after.board[sq(engine, "b4")] == 0 and engine.terminal(after) is None


def test_t1_a_piece_aimed_at_the_kings_flight_square_takes_it_there() -> None:
    """§2.5 C with the king faster: it runs to b4 and the rook, aimed at b4, takes it."""
    pieces = {"b1": ("R", 0), "e1": ("K", 0), "a5": ("K", 1)}
    engine, after, events = play(spec("T1"), pieces, ("b1", "b4"), ("a5", "b4"))
    assert events == [(EVENT_ORDER, sq(engine, "b4"), 1)]
    assert kind(engine, after, "b4") == "R" and owner(engine, after, "b4") == 0
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0


def test_t1_the_two_kings_are_one_speed_and_still_swap() -> None:
    """§3.2: the royal-royal lock is untouched (*chase* §2's bare kings, Ke1xd2 / Kd2xe1)."""
    pieces = {"e1": ("K", 0), "d2": ("K", 1)}
    engine, after, events = play(spec("T1"), pieces, ("e1", "d2"), ("d2", "e1"))
    assert events == []
    assert owner(engine, after, "d2") == 0 and owner(engine, after, "e1") == 1


def test_t2_a_knight_takes_the_king_before_it_can_run() -> None:
    """§2.5 A with the knight faster: the king's flight is never played and White wins."""
    pieces = {"c3": ("N", 0), "e1": ("K", 0), "d5": ("K", 1)}
    engine, after, events = play(spec("T2"), pieces, ("c3", "d5"), ("d5", "c4"))
    assert events == [(EVENT_ORDER, sq(engine, "d5"), 0), (EVENT_UNPLAYED, sq(engine, "d5"), 1)]
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0


def test_t2_the_other_knight_is_not_fast_enough_to_save_the_king() -> None:
    """Equal speeds: Black's knight strikes c3 as White's leaves it, and lands on empty c3."""
    pieces = {"c3": ("N", 0), "e1": ("K", 0), "d5": ("K", 1), "b5": ("N", 1)}
    engine, after, events = play(spec("T2"), pieces, ("c3", "d5"), ("b5", "c3"))
    assert EVENT_ORDER not in kinds(events)
    assert kind(engine, after, "c3") == "N" and owner(engine, after, "c3") == 1
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0


@pytest.mark.parametrize("slower", [("a5", "c3"), ("c4", "c3")], ids=["queen", "pawn"])
def test_t2_a_knight_lands_first_and_then_stands(slower: tuple[str, str]) -> None:
    """§2.5 C: the queen takes the knight that arrived first; the pawn's push is void."""
    pieces = {"b1": ("N", 0), "a5": ("Q", 1), "c4": ("P", 1), **KINGS}
    engine, after, events = play(spec("T2"), pieces, ("b1", "c3"), slower)
    if slower[0] == "a5":
        assert events == [(EVENT_ORDER, sq(engine, "c3"), 0)]
        assert kind(engine, after, "c3") == "Q" and owner(engine, after, "c3") == 1
    else:
        assert events == [(EVENT_ORDER, sq(engine, "c3"), 0), (EVENT_VOID, sq(engine, "c4"), 1)]
        assert kind(engine, after, "c3") == "N" and kind(engine, after, "c4") == "P"


def test_t2_a_knight_landing_on_a_rooks_path_voids_the_slide() -> None:
    """§2.5 D."""
    pieces = {"b1": ("N", 0), "a3": ("R", 1), **KINGS}
    engine, after, events = play(spec("T2"), pieces, ("b1", "c3"), ("a3", "e3"))
    assert events == [(EVENT_ORDER, sq(engine, "c3"), 0), (EVENT_VOID, sq(engine, "a3"), 1)]
    assert kind(engine, after, "a3") == "R" and after.board[sq(engine, "e3")] == 0


# Far-off pieces so that neither side is annihilated by the one capture a case makes.
REST = {"g1": ("E", 0), "a1": ("C", 0), "a7": ("R", 1), "g7": ("D", 1)}


def test_t3_a_cat_catches_a_rat_where_it_stands() -> None:
    """The faster hunter: the rat's flight is lost with it."""
    pieces = {"b3": ("C", 0), "c4": ("R", 1), **REST}
    engine, after, events = play(spec("T3"), pieces, ("b3", "c4"), ("c4", "c5"))
    assert events == [(EVENT_ORDER, sq(engine, "c4"), 0), (EVENT_LOST, sq(engine, "c4"), 1)]
    assert kind(engine, after, "c4") == "C" and after.board[sq(engine, "c5")] == 0


def test_t3_a_cat_steps_away_from_the_slower_dog() -> None:
    """The slower hunter: the cat has gone, and the dog lands on the empty square."""
    pieces = {"b3": ("C", 0), "c4": ("D", 1), **REST}
    engine, after, events = play(spec("T3"), pieces, ("b3", "b2"), ("c4", "b3"))
    assert events == [(EVENT_ORDER, sq(engine, "b2"), 0)]
    assert kind(engine, after, "b2") == "C" and kind(engine, after, "b3") == "D"


def test_t3_a_cat_that_steps_where_the_dog_was_going_is_eaten() -> None:
    """§2.5 C: the cat arrives first, and the dog's step onto it is a capture."""
    pieces = {"b2": ("C", 0), "c4": ("D", 1), **REST}
    engine, after, events = play(spec("T3"), pieces, ("b2", "b3"), ("c4", "b3"))
    assert events == [(EVENT_ORDER, sq(engine, "b3"), 0)]
    assert kind(engine, after, "b3") == "D" and owner(engine, after, "b3") == 1


def test_t3_at_the_hole_the_faster_animal_arrives_first() -> None:
    """Both step onto d4. A rat cannot eat the cat, so the cat wins on the spot and the rat's
    step is never played; a dog can, so the cat is not safe there and the dog eats it."""
    pieces = {"c3": ("C", 0), "e5": ("R", 1), **REST}
    engine, after, events = play(spec("T3"), pieces, ("c3", "d4"), ("e5", "d4"))
    assert events == [(EVENT_ORDER, sq(engine, "d4"), 0), (EVENT_UNPLAYED, sq(engine, "e5"), 1)]
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0
    pieces = {"c3": ("C", 0), "e5": ("D", 1), **REST}
    engine, after, events = play(spec("T3"), pieces, ("c3", "d4"), ("e5", "d4"))
    assert events == [(EVENT_ORDER, sq(engine, "d4"), 0)]
    assert kind(engine, after, "d4") == "D"
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 1
