"""Where each wild piece goes next: the phase as the board now stands (Q660).

A ruleset with ``rules.npcs`` runs a phase after every move: each wild piece steps to the
next square of its route, eating what its type may take, or waits. The board draws where
each is going, and that is a question for the core, never for Python or the page
(CLAUDE.md rule 3), so this asks it: ``Core.coming_phase`` plays the phase on the position
as it is -- the same rule (``npc_step_of``), the same order, the same ``(from, to, eaten)``
steps ``Engine.apply_with_phase`` reports -- and puts the board back.

The real phase runs after the *coming* move, which may step into a wild piece's way, clear
it, or take the piece, so this is not a prediction of that move. Wild pieces walk fixed
routes: nothing in it is drawn at random.

**A module of its own, beside ``engine.py`` rather than inside it**, because
``engine/engine.py`` is in the ``analysis`` and ``metrics`` fingerprints
(``measurement.ANALYSIS_SOURCES``): a method there that no analysis or score ever calls
would report every stored measurement in ``runs/`` stale for an edit that changes none of
them. This file is in no set, as ``engine/compile.py`` is, and reaches no score.
"""

from __future__ import annotations

from chess_variant_sandbox.engine.engine import Engine, _pack
from chess_variant_sandbox.engine.state import GameState, PhaseStep


def coming_phase(engine: Engine, state: GameState) -> tuple[PhaseStep, ...]:
    """The steps the wild-piece phase would take on ``state``'s board, from the core.

    Empty for every ruleset without ``rules.npcs``. A wild piece that would wait (its next
    square holds a piece its type may not take) has no step.
    """
    return tuple(PhaseStep(*step) for step in engine.core.coming_phase(_pack(state)))
