"""Where each wild piece goes next, and after that: the phases as the board now stands (Q660, Q764).

A ruleset with ``rules.npcs`` runs a phase after every move: each wild piece steps to the
next square of its route, eating what its type may take, or waits. The board draws where
each is going, and that is a question for the core, never for Python or the page
(CLAUDE.md rule 3), so this asks it: ``Core.coming_phases`` plays the phase on the position
as it is -- the same rule (``npc_step_of``), the same order, the same ``(from, to, eaten)``
steps ``Engine.apply_with_phase`` reports -- then plays it again on the board the first one
left, and puts the board back.

The board asks for two phases (Q764): the one after the next move and the one after the move
that follows it, which is the reply, or in a long turn (SR7b) the mover's own next move while
its turn lasts; a pouncer's prey is the player who makes that move, read off the turn's
schedule by the core. The real phases run after
the *coming* moves, which may step into a wild piece's way, clear it, or take the piece,
so this is not a prediction of those moves. Wild pieces walk fixed routes: nothing in it
is drawn at random.

**A module of its own, beside ``engine.py`` rather than inside it**, because
``engine/engine.py`` is in the ``analysis`` and ``metrics`` fingerprints
(``measurement.ANALYSIS_SOURCES``): a method there that no analysis or score ever calls
would report every stored measurement in ``runs/`` stale for an edit that changes none of
them. This file is in no set, as ``engine/compile.py`` is, and reaches no score.
"""

from __future__ import annotations

from chess_variant_sandbox.engine.engine import Engine, _pack
from chess_variant_sandbox.engine.state import GameState, PhaseStep

# How many phases the board draws: the one after the next move and the one after the move
# after that (the reply, or in a long turn the same player's next move).
COMING_PHASES = 2


def coming_phases(
    engine: Engine, state: GameState, count: int = COMING_PHASES
) -> tuple[tuple[PhaseStep, ...], ...]:
    """The steps the next ``count`` wild-piece phases would take on ``state``'s board.

    Each phase is played on the board the one before it left, without the moves that will
    come between them. Exactly ``count`` phases, each empty for a ruleset without
    ``rules.npcs``. A wild piece that would wait (its next square holds a piece its type
    may not take) has no step in that phase.
    """
    return tuple(
        tuple(PhaseStep.from_core(step) for step in phase)
        for phase in engine.core.coming_phases(_pack(state), count)
    )


def coming_phase(engine: Engine, state: GameState) -> tuple[PhaseStep, ...]:
    """The steps the wild-piece phase would take on ``state``'s board, from the core.

    Empty for every ruleset without ``rules.npcs``. A wild piece that would wait (its next
    square holds a piece its type may not take) has no step.
    """
    return coming_phases(engine, state, 1)[0]
