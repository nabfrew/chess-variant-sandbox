"""F10b group 2 (Q752b): the round's corners no test reached, held to the declared oracle.

`docs/plan-tests.md` F10b: lines of `rust/src/rounds.rs` no test spec made a round reach. The
endings read between an ordered round's two moves (schema 28, `docs/plan-speed.md` §2.6) for
each goal rule a round may carry -- a goal left, the extinction of any piece or of named ones,
a bare royal, a territory count -- and stalemate as a win, each on Gardner in rounds with
speeds (`tests/_simultaneous_speed_gardner.json`) and that one rule added. Every round of
seeded walks is resolved by the core and by `tests/simultaneous_oracle.py` and the whole
position compared (`test_simultaneous_oracle.walk`); the oracle reads the middle of an ordered
round through the core's alternating-game verdict (`terminal_prefix`), so the round's own copy
of the endings is held to that one. Where a walk cannot be trusted to meet a case, a
hand-built round is played and its verdict counted by hand.
"""

from __future__ import annotations

import copy
import json
import random
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from simultaneous_oracle import legal_for, resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import MOVE_DROP, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec

HERE = Path(__file__).parent


def _speed_gardner(**rules: Any) -> dict[str, Any]:
    spec = json.loads((HERE / "_simultaneous_speed_gardner.json").read_text(encoding="utf-8"))
    spec = copy.deepcopy(spec)
    spec["name"] = "Gardner in rounds with speeds (F10b test spec)"
    spec["rules"].update(rules)
    return spec


# Each variant's rule, and the reason a game of it ends by when that rule ends it.
VARIANTS: dict[str, tuple[dict[str, Any], str]] = {
    "goal left": ({"reach": {"piece": "N", "squares": ["b1"], "mirror": True, "result": "win",
                             "on": "leave"}}, "reached"),
    "any piece": ({"extinction": {"pieces": "any", "min_count": 9, "result": "loss"}},
                  "extinction"),
    "named pieces": ({"extinction": {"pieces": ["P", "N"], "min_count": 6, "result": "loss"}},
                     "extinction"),
    "bare royal": ({"bare_royal": "loss"}, "bare_royal"),
    "territory": ({"territory": {"mode": "permanent", "win_at": 9}}, "coloured"),
}


@cache
def engine_for(name: str) -> Engine:
    return Engine(_speed_gardner(**VARIANTS[name][0]))


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_the_variants_validate(name: str) -> None:
    assert validate_spec(_speed_gardner(**VARIANTS[name][0])) == []


EVENT_UNPLAYED = 7


def walk_to_the_end(engine: Engine, seed: int, rounds: int = 60) -> tuple[Any, bool]:
    """`test_simultaneous_oracle.walk`, to the game's end: every round's two lists and the
    position it makes held to the oracle. The verdict, and whether the last round was cut
    short by an ending its faster move fulfilled (the slower move never played)."""
    rng = random.Random(seed)
    state: GameState = engine.initial_state()
    cut = False
    for _ in range(rounds):
        verdict = engine.terminal(state)
        if verdict is not None:
            return verdict, cut
        lists = [engine.legal_moves_for(state, player) for player in (0, 1)]
        assert lists == [legal_for(engine, state, player) for player in (0, 1)]
        first, second = rng.choice(lists[0]), rng.choice(lists[1])
        expected = resolve_round(engine, state, first, second)
        actual, _, events = engine.apply_round_with_phase(state, first, second)
        assert actual == expected, (seed, state.ply)
        assert actual.key == engine._compute_hash(actual)
        cut = any(kind == EVENT_UNPLAYED for kind, _, _ in events)
        state = actual
    return engine.terminal(state), cut


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_an_ordered_round_ends_at_its_faster_move(name: str) -> None:
    """Twelve seeded games of each variant, every round held to the oracle; at least one
    ends by the variant's own rule in a round whose faster move fulfilled it, so the slower
    move was never played (the oracle agreeing that it was not)."""
    engine = engine_for(name)
    reason = VARIANTS[name][1]
    endings = [walk_to_the_end(engine, seed) for seed in range(12)]
    assert any(verdict is not None and verdict.reason == reason and cut
               for verdict, cut in endings), endings


def test_a_round_with_a_royal_and_no_win_rule_agrees_with_the_oracle() -> None:
    """`win: "none"` beside a royal (and a bare royal, so the game can end): the round reads
    the royal ending's reason, but no royal's loss ends the game (only `royal_capture` would,
    since a round refuses `checkmate`), every round held to the oracle."""
    engine = Engine(_speed_gardner(win="none", bare_royal="loss"))
    for seed in range(2):
        verdict, _ = walk_to_the_end(engine, seed, rounds=30)
        assert verdict is None or verdict.reason != "royal_captured"


@pytest.mark.parametrize(("stalemate", "winner", "reason"), [
    ("win", 1, "stalemate_win"), ("loss", 0, "stalemate_loss"), ("draw", None, "stalemate"),
])
def test_a_player_with_no_move_in_a_round_takes_the_stalemate_result(
    stalemate: str, winner: int | None, reason: str,
) -> None:
    """Counted by hand: Black's one piece, a pawn on its own last rank, has no move, and White
    has a king and a queen, so Black is stuck at the start of the round and the game ends
    with `rules.stalemate`'s result for Black. (`win: "none"`, so no missing royal ends it
    first; the bare royal is there for the validator, and neither side is bare.)"""
    engine = Engine(_speed_gardner(win="none", bare_royal="loss", stalemate=stalemate))
    state = engine.make_state({"a1": ("K", 0), "c3": ("Q", 0), "e1": ("P", 1)})
    assert engine.legal_moves_for(state, 1) == []
    assert engine.legal_moves_for(state, 0) != []
    verdict = engine.terminal(state)
    assert verdict is not None and (verdict.winner, verdict.reason) == (winner, reason)


# -- meetings: a contact that stays, a strike that advances, a narrow blast -------------------


def _gardner_rounds(base: str = "_simultaneous_gardner", **rules: Any) -> dict[str, Any]:
    spec = json.loads((HERE / f"{base}.json").read_text(encoding="utf-8"))
    spec["name"] = "Gardner in rounds (F10b test spec)"
    spec["rules"].update(rules)
    return spec


MEETINGS: dict[str, dict[str, Any]] = {
    # A knight that touches an enemy pawn stays where it was and becomes a bishop, the pawn
    # a knight of its own side; a rook that touches its own pawn stays and the pawn becomes
    # a queen.
    "contacts stay": {"contacts": {"position": "stay", "immune": ["K"], "links": [
        {"mover": "N", "onto": "P", "onto_to": "N", "mover_to": "B"},
        {"mover": "R", "onto": "P", "onto_to": "Q", "own": True},
    ]}},
    # Andernach on an advancing capture: a knight or a bishop that captures changes sides
    # where it lands.
    "strike advances": {"strike": {"effect": "capture", "position": "advance",
                                   "switches_sides": ["N", "B"], "immune": ["K"]}},
    # Only a knight's capture explodes.
    "blast by knights": {"explode_on_capture": {"radius": 1, "by": ["N"]}},
}


@pytest.mark.parametrize("name", sorted(MEETINGS))
def test_meetings_in_rounds_agree_with_the_oracle(name: str) -> None:
    spec = _gardner_rounds(**MEETINGS[name])
    assert validate_spec(spec) == []
    engine = Engine(spec)
    for seed in range(6):
        walk_to_the_end(engine, seed, rounds=40)


# -- hand-built rounds ---------------------------------------------------------------------


def _word(engine: Engine, state: GameState, player: int, text: str) -> int:
    """``player``'s one move in ``state`` from the first square ``text`` names to the last
    (``"Nb1-a3"``). Read by squares, not by `describe_move`, which a circe capture in a round
    cannot describe (it asks `apply`, which refuses a round game)."""
    origin, target = (engine._square_index(name) for name in re.findall(r"[a-h][1-8]", text))
    found = [m for m in engine.legal_moves_for(state, player)
             if decode_move(m)[:2] == (origin, target)]
    assert len(found) == 1, (text, found)
    return found[0]


def _round(engine: Engine, state: GameState, first: str, second: str) -> GameState:
    """The round ``first`` | ``second``, as the core makes it, held to the oracle."""
    a, b = _word(engine, state, 0, first), _word(engine, state, 1, second)
    actual = engine.apply_round(state, a, b)
    assert actual == resolve_round(engine, state, a, b)
    assert actual.key == engine._compute_hash(actual)
    return actual


def _on(engine: Engine, state: GameState, square: str) -> tuple[str, int] | None:
    value = state.board[engine._square_index(square)]
    if not value:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def test_both_players_leaving_their_goal_in_one_round_is_a_draw() -> None:
    """`reach.on: "leave"`, the knight's start square the goal: both knights (speed 2 each,
    so the round is not ordered) leave in the first round, so both fulfil the goal at once
    and, with no scoring, the game is drawn by it."""
    engine = engine_for("goal left")
    after = _round(engine, engine.initial_state(), "Nb1-a3", "Nb5-c3")
    verdict = engine.terminal(after)
    assert verdict is not None and (verdict.winner, verdict.reason) == (None, "reached")


def _swap_game(**beats: list[str]) -> Engine:
    spec = _gardner_rounds(simultaneous={"swap": "collision"})
    for piece in spec["pieces"]:
        if piece["id"] in beats:
            piece["beats"] = beats[piece["id"]]
    return Engine(spec)


SWAP = {"a1": ("K", 0), "c3": ("R", 0), "c4": ("Q", 1), "e5": ("K", 1)}


def test_a_swap_won_by_the_second_player() -> None:
    """Counted by hand: White's rook on c3 and Black's queen on c4 swap squares, and only the
    queen beats a rook, so the queen wins the meeting: it stands on c3 and the rook is gone."""
    engine = _swap_game(R=["P"], Q=["R"])
    after = _round(engine, engine.make_state(SWAP), "Rc3xc4", "Qc4xc3")
    assert _on(engine, after, "c3") == ("Q", 1)
    assert _on(engine, after, "c4") is None
    assert "R" not in {engine.spec.piece_ids[engine.spec.type_of(v)] for v in after.board if v}


def test_a_swap_neither_piece_wins() -> None:
    """Counted by hand: neither the rook nor the queen beats the other, and the tie is not
    broken by an initiative, so neither wins the meeting: each left its square and neither
    arrives, so both are gone and only the kings stand."""
    engine = _swap_game(R=["P"], Q=["P"])
    after = _round(engine, engine.make_state(SWAP), "Rc3xc4", "Qc4xc3")
    pieces = sorted(engine.spec.piece_ids[engine.spec.type_of(v)] for v in after.board if v)
    assert pieces == ["K", "K"]


def test_a_collision_on_a_wild_piece_s_square_is_won_and_the_wild_piece_taken() -> None:
    """Counted by hand: White's rook and Black's bishop both go for the wild knight on c3;
    the rook beats a bishop and the bishop beats nothing it meets, so the rook wins the
    collision, takes the wild knight standing there and the bishop, and stands on c3."""
    spec = _gardner_rounds("_simultaneous_wild")
    for piece in spec["pieces"]:
        if piece["id"] == "R":
            piece["beats"] = ["B", "N"]
        if piece["id"] == "B":
            piece["beats"] = ["P"]
    engine = Engine(spec)
    state = engine.make_state({
        "a1": ("K", 0), "c1": ("R", 0), "c3": ("N", WILD), "e5": ("K", 1), "a5": ("B", 1),
    })
    after = _round(engine, state, "Rc1xc3", "Ba5xc3")
    assert _on(engine, after, "c3") == ("R", 0)
    assert _on(engine, after, "a5") is None


def test_a_rest_restrict_empties_is_lifted_after_all() -> None:
    """Counted by hand: White's king rests, so the rest leaves White only the pawn's step to
    c5, and `forbid_immobility` refuses that (no promotion here, so a pawn on the last rank
    could never move again): the rest emptied a list it had shortened, so it is made again
    without the rest, and White has the king's three steps. The lists are the rest oracle's."""
    spec = _gardner_rounds("_simultaneous_resting", forbid_immobility=True)
    del spec["rules"]["promotion"]
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.make_state({"a1": ("K", 0), "c4": ("P", 0), "e3": ("K", 1)})
    state = state.with_(rest=(((engine._square_index("a1"), 1),), ()))
    moves = engine.legal_moves_for(state, 0)
    assert moves == legal_for(engine, state, 0)
    assert sorted(engine.describe_move(state, m) for m in moves) == [
        "Ka1-a2", "Ka1-b1", "Ka1-b2",
    ]


def test_a_round_perft_counts_by_hand() -> None:
    """A round's perft: depth 0 is the position itself, depth 1 every pair of the two
    players' moves, and a decided position (White has no king) counts nothing under
    ``stop_at_terminal`` though it has moves to pair."""
    engine = Engine(_gardner_rounds())
    start = engine.initial_state()
    assert engine.perft(start, 0) == 1
    lists = [engine.legal_moves_for(start, player) for player in (0, 1)]
    assert engine.perft(start, 1) == len(lists[0]) * len(lists[1])
    decided = engine.make_state({"c3": ("Q", 0), "e5": ("K", 1)})
    assert engine.terminal(decided) is not None
    pairs = len(engine.legal_moves_for(decided, 0)) * len(engine.legal_moves_for(decided, 1))
    assert pairs > 0
    assert engine.perft(decided, 1) == pairs
    assert engine.perft(decided, 1, stop_at_terminal=True) == 0


def test_a_bar_never_removes_a_drop() -> None:
    """A barred pair (§2.7) takes away its capture-only move, never a drop: with a chick in
    White's hand and a bar standing, White's drops are the same as with no bar. (A drop's
    origin is its target, so no bar's two squares could name it; the early answer is a
    shortcut the list does not depend on.)"""
    engine = engine_for_file("_simultaneous_shogi")
    start = engine.initial_state()
    chick = engine.spec.piece_ids.index("C")
    hand = tuple(1 if piece_type == chick else 0 for piece_type in range(len(start.hands[0])))
    held = start.with_(hands=(hand, start.hands[1]))
    barred = held.with_(bars=((engine._square_index("b2"), engine._square_index("b3")),))

    def drops(state: GameState) -> list[int]:
        return [m for m in engine.legal_moves_for(state, 0) if decode_move(m)[2] == MOVE_DROP]

    assert drops(barred) == drops(held) != []
    assert engine.legal_moves_for(barred, 0) == legal_for(engine, barred, 0)


@cache
def engine_for_file(name: str) -> Engine:
    return Engine(json.loads((HERE / f"{name}.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize(("blocked", "reborn"), [(False, ("P", 1)), (True, ("B", 0))])
def test_a_circe_rebirth_needs_its_home_empty(blocked: bool, reborn: tuple[str, int]) -> None:
    """Counted by hand, in Gardner in rounds with circe: White's knight takes Black's pawn on
    c3 while Black's king steps aside; the pawn is reborn on its home square c4 when that is
    empty, and is simply gone when White's bishop stands there."""
    engine = engine_for_file("_simultaneous_circe")
    placement = {"a1": ("K", 0), "b1": ("N", 0), "c3": ("P", 1), "e5": ("K", 1)}
    if blocked:
        placement["c4"] = ("B", 0)
    after = _round(engine, engine.make_state(placement), "Nb1xc3", "Ke5-d5")
    assert _on(engine, after, "c3") == ("N", 0)
    assert _on(engine, after, "c4") == reborn

