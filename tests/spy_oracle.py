"""The declared Python oracle for a spy game (schema 27, Q935a).

`docs/plan-spy.md` §2.5: the phase is a function of the board and the side to move, and
each step is played by its phase -- an alternating or reply turn as the alternating game
plays it, a round as `tests/simultaneous_oracle.py` resolves one -- with the one write of
the spy's own, the initiative's deal at the first announced round (E10). This module works
all of that out from the compiled tables and the moves, and the tests hold the core to it
step by step. CLAUDE.md rule 3 allows it: it is a declared test oracle.

It asks the engine only questions answered elsewhere: an alternating turn is played by the
spy game's **alternating twin** (the same file without the spy flags, ``beats`` and
``rules.simultaneous``), whose moves the alternating game's own tests hold; a round is
resolved by the round oracle; and the key is the slow definition (``_compute_hash``). It
never asks the spy game's core what a step does.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from tests.simultaneous_oracle import legal_for, resolve_round

NO_PAIR = (-1, -1)


def has_spy(engine: Engine, state: GameState, player: int) -> bool:
    """Whether ``player`` has a spy: a piece of any of its spy types on the board, summed
    over the types (§1.2)."""
    spec = engine.spec
    assert spec.spy is not None
    types = set(spec.spy.types[player])
    return any(
        value and spec.owner_of(value) == player and spec.type_of(value) in types
        for value in state.board
    )


def phase(engine: Engine, state: GameState) -> tuple[str, int]:
    """§1.3's table: ``(name, announcer)``, the announcer -1 outside an announced round."""
    first, second = has_spy(engine, state, 0), has_spy(engine, state, 1)
    if first and second:
        return "alternating", -1
    if not first and not second:
        return "blind", -1
    holder = 0 if first else 1
    if state.side_to_move == holder:
        return "announced", 1 - holder
    return "reply", -1


def is_round(engine: Engine, state: GameState) -> bool:
    return phase(engine, state)[0] in ("announced", "blind")


def twin_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """The alternating twin (§1.1): the spy flags, ``beats`` and ``rules.simultaneous``
    taken out, everything else as written."""
    twin = copy.deepcopy(spec)
    for piece in twin["pieces"]:
        piece.pop("spy", None)
        piece.pop("beats", None)
    twin["rules"].pop("simultaneous", None)
    twin["name"] = f"{spec['name']} (alternating twin)"
    return twin


@cache
def twin(engine: Engine) -> Engine:
    return Engine(twin_spec(json.loads(engine.spec.source)))


def as_twin(engine: Engine, state: GameState) -> GameState:
    """The same position in the twin: the round state left out, its own key."""
    other = twin(engine)
    bare = state.with_(initiative=-1, void_run=NO_PAIR, bars=(), rest=NO_PAIR)
    key = other._compute_hash(bare)
    return bare.with_(key=key, repetition=(key,))


def legal(engine: Engine, state: GameState) -> list[int] | tuple[list[int], list[int]]:
    """A turn's moves (the twin's) or a round's two lists (the round oracle's)."""
    if is_round(engine, state):
        return legal_for(engine, state, 0), legal_for(engine, state, 1)
    return twin(engine).legal_moves(as_twin(engine, state))


def turn(engine: Engine, state: GameState, move: int) -> GameState:
    """An alternating or reply turn, as the twin plays it, then the deal (E10)."""
    assert not is_round(engine, state), "a turn in a round phase"
    played = twin(engine).apply(as_twin(engine, state), move)
    reset = len(played.repetition) == 1
    after = played.with_(
        initiative=state.initiative, void_run=state.void_run, bars=state.bars, rest=state.rest,
    )
    sim = engine.spec.simultaneous
    name, announcer = phase(engine, after)
    if sim is not None and sim.deal_initiative and name == "announced":
        after = after.with_(initiative=announcer)
    key = engine._compute_hash(after)
    repetition = (key,) if reset else (*state.repetition, key)
    return after.with_(key=key, repetition=repetition)


def round_(engine: Engine, state: GameState, first: int, second: int) -> GameState:
    """A round, announced or blind: the resolution never asks who announced (§1.4)."""
    assert is_round(engine, state), "a round in a turn phase"
    return resolve_round(engine, state, first, second)
