"""``solve_endgame``: a material set, solved forward by the core (``Core.solve_endgame``).

The solver is ``rust/src/endgame.rs`` (N1, ``docs/reference/no-forced-mate.md``). This module
only spells a set in the spec's own piece ids, turns them into the type indices the core reads,
and reads the core's answer back into named fields. It decides nothing: every label and distance
is the core's (rule 3).

It sits beside ``engine.py`` rather than in it, and nothing a score reads imports it: ``engine.py``
is in the metrics fingerprint (``measurement.METRICS_SOURCES``), and a solver nothing measured
uses has no business moving every stored score's basis.

**How a set is spelled.** A tuple of piece-id lists, one per owner: player 0's pieces, then
player 1's, then (optionally) the pieces nobody owns. Every piece is named, royals included, and
a piece named twice stands twice: ``(["K", "N", "N"], ["K"])`` is two of player 0's ``N`` and a
``K`` each. The ids are the spec's ``pieces[].id``; nothing here knows any game's names.

**Wild pieces and the duck** are the game's, never named in a set: the solve stands the wild
pieces where the opening puts them and the duck on every square it may, and reports them among
nobody's pieces in every set it returns (N2, ``rust/src/endgame.rs``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess_variant_sandbox._core as _core
from chess_variant_sandbox.engine.engine import Engine, _pack, _unpack
from chess_variant_sandbox.engine.state import GameState

if TYPE_CHECKING:
    from chess_variant_sandbox.engine.compile import CompiledSpec

MaterialSet = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def material_set(spec: CompiledSpec, pieces: Sequence[Sequence[str]]) -> tuple[list[int], list[int], list[int]]:
    """The type indices of a set spelled in piece ids, by owner. Refuses an unknown id by name."""
    if not 1 <= len(pieces) <= 3:
        raise ValueError("a material set is one list of piece ids per owner: player 0, player 1, and nobody")
    if isinstance(pieces, str) or any(isinstance(side, str) for side in pieces):
        raise ValueError("a material set's sides are lists of piece ids, not strings")
    index = {piece_id: number for number, piece_id in enumerate(spec.piece_ids)}
    unknown = sorted({piece for side in pieces for piece in side if piece not in index})
    if unknown:
        raise ValueError(f"not a piece of this game: {', '.join(unknown)} (its ids: {', '.join(spec.piece_ids)})")
    sides = [[index[piece] for piece in side] for side in pieces]
    sides += [[] for _ in range(3 - len(sides))]
    return sides[0], sides[1], sides[2]


def material_name(material: MaterialSet) -> str:
    """``"KR v K"``-style text for a set, nobody's pieces after a second ``v`` when there are any."""
    text = "".join(material[0]) + " v " + "".join(material[1])
    return text + (" v " + "".join(material[2]) if material[2] else "")


@dataclass(frozen=True)
class MoverCounts:
    """One set's verdicts with one player to move: which player holds a win is read here."""

    won: int
    lost: int
    drawn: int
    longest_win: int
    longest_loss: int


@dataclass(frozen=True)
class EndgameSet:
    """One material set's verdicts for the side to move, every distance in plies.

    The asked set counts every placement of it; a set the closure reached (after a capture, a
    promotion, a rebirth, a side switch) counts only the positions reached. ``illegal`` is the
    asked set's placements no legal move leaves (the player who has just moved in check, where
    check binds), left unsolved. ``beyond_clock`` counts the wins whose distance to the end or to
    the next move that resets the no-progress clock is longer than that clock: wins the game's
    own rules would draw.
    """

    material: MaterialSet
    won: int
    lost: int
    drawn: int
    illegal: int
    longest_win: int
    longest_loss: int
    beyond_clock: int
    longest_to_reset: int
    requested: bool
    mover: tuple[MoverCounts, MoverCounts]

    @property
    def positions(self) -> int:
        return self.won + self.lost + self.drawn

    @property
    def name(self) -> str:
        return material_name(self.material)

    def forced_by(self, player: int) -> bool:
        """Whether ``player`` can force a win in this set: some position with the other player
        to move is lost for them with a move still to make (a win already on the board is not
        one forced)."""
        defender = self.mover[1 - player]
        return defender.lost > 0 and defender.longest_loss > 0


@dataclass(frozen=True)
class EndgameSample:
    """A position of the asked set the solve decided: ``value`` 1 won, -1 lost, for the side to move."""

    state: GameState
    value: int
    plies: int


PROBE_ABSENT = 2


@dataclass(frozen=True)
class EndgameProbe:
    """A given position's verdict: ``value`` 1 won, -1 lost, 0 drawn for its side to move, or
    None where the solve never reached it; ``plies`` to the end, ``to_reset`` to the end or the
    next move that resets the no-progress clock (0 for a draw)."""

    value: int | None
    plies: int
    to_reset: int


@dataclass(frozen=True)
class EndgameSolve:
    """A whole solve: the asked set first, then every set its closure reached, and its size."""

    sets: tuple[EndgameSet, ...]
    positions: int
    edges: int
    sweeps: int
    samples: tuple[EndgameSample, ...]
    probes: tuple[EndgameProbe, ...] = ()

    @property
    def requested(self) -> EndgameSet:
        return next(one for one in self.sets if one.requested)

    def of(self, *pieces: Sequence[str]) -> EndgameSet | None:
        """The set with these pieces (one id list per owner, as asked), if the solve reached it."""
        wanted = tuple(tuple(sorted(side)) for side in (*pieces, (), ())[:3])
        for one in self.sets:
            if tuple(tuple(sorted(side)) for side in one.material) == wanted:
                return one
        return None


def solve_endgame(
    engine: Engine,
    pieces: Sequence[Sequence[str]],
    max_positions: int | None = None,
    sample_plies: int = 0,
    probes: Sequence[GameState] = (),
) -> EndgameSolve:
    """Solve a material set forward in ``engine``'s game, and every set it leads to.

    ``pieces`` is one list of piece ids per owner, player 0's first, royals included (the
    module's docstring has the spelling). Every position of the set is labelled won, lost or
    drawn for the side to move, with its distance in plies, from the game's own moves and
    endings (``rust/src/endgame.rs``); the clocks are left out and a win longer than the
    no-progress clock is counted apart. ``max_positions`` caps the solve (at most
    ``_core.ENDGAME_MAX_POSITIONS``); a game the solver cannot read (a capture hand, the purse,
    rounds, spies) is refused by name, as ``ValueError``.
    ``sample_plies`` returns the asked set's positions decided in that many plies or fewer.
    ``probes`` are positions (from stored games, say) whose verdicts come back in order, each
    looked up after its clocks are zeroed and its castling rights dropped, as a start's are.
    """
    sides = material_set(engine.spec, pieces)
    cap = _core.ENDGAME_MAX_POSITIONS if max_positions is None else max_positions
    sets, (positions, edges, sweeps), samples, probed = engine.core.solve_endgame(
        sides, cap, sample_plies, [_pack(state) for state in probes]
    )
    ids = engine.spec.piece_ids

    def named(material: tuple[list[int], list[int], list[int]]) -> MaterialSet:
        return (
            tuple(ids[t] for t in material[0]),
            tuple(ids[t] for t in material[1]),
            tuple(ids[t] for t in material[2]),
        )

    solved = [
        EndgameSet(
            material=named(material),
            won=counts[0],
            lost=counts[1],
            drawn=counts[2],
            illegal=counts[3],
            longest_win=lengths[0],
            longest_loss=lengths[1],
            beyond_clock=lengths[2],
            longest_to_reset=lengths[3],
            requested=requested,
            mover=(MoverCounts(*by_mover[0]), MoverCounts(*by_mover[1])),
        )
        for material, counts, lengths, requested, by_mover in sets
    ]
    solved.sort(key=lambda one: not one.requested)
    return EndgameSolve(
        sets=tuple(solved),
        positions=positions,
        edges=edges,
        sweeps=sweeps,
        samples=tuple(EndgameSample(_unpack(state), value, plies) for state, value, plies in samples),
        probes=tuple(
            EndgameProbe(None if value == PROBE_ABSENT else value, plies, to_reset) for value, plies, to_reset in probed
        ),
    )
