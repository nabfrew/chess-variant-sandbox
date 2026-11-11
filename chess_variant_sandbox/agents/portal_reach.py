"""T7's P6 for the players (`docs/plan-player-weaknesses.md` §T7, Q1175b): capture reach through
the portals in the derived piece values, the ``portal_reach`` search bit's table.

The derived table prices a piece by its mean capture reach on an empty board
(``evaluation._measured_values``), read off the compiled ``leaps``, ``slides``, ``hops`` and
``sweeps`` tables, which hold no portal transit. In a portal game a piece beside a door takes on
squares those tables never name. This module is the declared oracle for
``rust/src/eval.rs::portal_reach_values`` (``Core.portal_reach_values`` answers the core's half;
``tests/test_portal_reach.py`` holds the two together), and reads only compiled tables: the
movement entries, the momentum masks parallel to their landings, the exit table, ``dest``,
``admits``, ``stand`` and where each end stands **at the start** (a moving door is read where it
starts: one static table, stated as such). Nothing here works out a square.

**The reading.** For each player ``p``, declared type ``t`` and origin square ``s`` not holding
a momentum end, on a board holding only the ends where they start, in frame ``p``'s tables
(facing 0):

- a **momentum end** is met as the core's ``Game::push_transits`` meets one: a leap's landing on
  it, or the first end on a slide's ray (an end blocks the ray); if it admits ``t`` for ``p`` and
  leads somewhere (``dest``), each direction in the landing's *capture-mode* momentum bits comes
  out one step past the end it leads to (the exit table, ``p``'s wrap): that exit is a capture
  square, unless it is off the board, the origin, outside ``t``'s region, or another end;
- a **fold** is floor to every ray, and a capture-mode landing on a fold's start square that
  admits ``t`` for ``p`` and leads somewhere goes through to the far fold's start square, which
  is then a capture square on the same terms.

A hop's transit needs a screen standing, and a board of one piece among the ends places none, so
hops add nothing; a sweep never transits (``portal_attacks`` skips sweep sources). The squares
added are those not already among the origin's empty-board capture landings in frame ``p``.

``portal_extra[t]`` is the added squares summed over both players and every origin, over
``2 x n_squares``: the transit's share of the mean reach, averaged over the two seats because a
passage may open to one player only and the derived table is one per type.

**The table.** ``portal_reach_values(spec, derived)`` is ``derived`` with every priced type
(not royal, not an unlimited drop) raised by ``REACH_SHARE x extra / scale``: the reach half of
``Evaluator.for_spec``'s blend, ``1 - DECLARED_WEIGHT``, on the derived table's own scale (the
cheapest priced type's empty-board reach), so the unit a piece is counted in does not move and a
type no transit reaches keeps its value bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.agents.evaluation import (
    DECLARED_WEIGHT,
    _measured_values,
    inexhaustible_types,
)
from chess_variant_sandbox.engine.compile import WILD, CompiledSpec, PortalRule

# The reach half of `Evaluator.for_spec`'s blend; `eval.rs::PORTAL_REACH_SHARE` is the core's.
REACH_SHARE = 1.0 - DECLARED_WEIGHT
# `MODE_CAPTURE`: a landing's capture bit.
_CAPTURE = 2


def _capture_landings(spec: CompiledSpec, frame: int, piece_type: int, square: int) -> set[int]:
    """The squares ``_measured_values`` counts for this slot: every capture-mode landing."""
    targets: set[int] = set()
    for table in (spec.leaps, spec.slides, spec.hops, spec.sweeps):
        for squares, modes, *_ in table[frame][piece_type][square]:
            targets.update(target for index, target in enumerate(squares) if modes[index] & _CAPTURE)
    return targets


@dataclass(frozen=True)
class _Doors:
    """Where the ends stand at the start, and what a transit through them reaches."""

    spec: CompiledSpec
    portals: PortalRule
    n_dirs: int
    end_at: dict[int, int]  # square -> the momentum end standing there
    square_of: dict[int, int]  # momentum end -> its square
    fold_at: dict[int, int]  # square -> the fold starting there
    fold_square: dict[int, int]  # fold -> its square

    @classmethod
    def of(cls, spec: CompiledSpec, portals: PortalRule) -> _Doors:
        end_at: dict[int, int] = {}
        square_of: dict[int, int] = {}
        for square in range(spec.n_squares):
            value = spec.initial_board[square]
            if value and spec.owner_of(value) == WILD and portals.is_end(spec.type_of(value)):
                end_at[square] = spec.type_of(value)
                square_of[spec.type_of(value)] = square
        fold_at: dict[int, int] = {}
        fold_square: dict[int, int] = {}
        if portals.folds is not None:
            for end, start in enumerate(portals.folds.starts):
                if start >= 0:
                    fold_at[start] = portals.declared_types + end
                    fold_square[portals.declared_types + end] = start
        return cls(
            spec, portals, len(portals.exits[0][0]), end_at, square_of, fold_at, fold_square
        )

    def _lands(self, player: int, piece_type: int, origin: int, square: int) -> bool:
        return (
            square >= 0
            and square != origin
            and square not in self.end_at
            and self.portals.may_stand(player, piece_type, square)
        )

    def _exits(self, end: int, bits: int, player: int, piece_type: int, origin: int) -> set[int]:
        """A momentum end entered with these momentum bits: its capture-mode exits."""
        taking = (bits >> self.n_dirs) & ((1 << self.n_dirs) - 1)
        to = self.portals.dest[end]
        if not taking or not self.portals.admitted(end, player, piece_type):
            return set()
        if to < 0 or to not in self.square_of:
            return set()
        far = self.square_of[to]
        return {
            square
            for direction in range(self.n_dirs)
            if taking >> direction & 1
            for square in (self.portals.exits[player][far][direction],)
            if self._lands(player, piece_type, origin, square)
        }

    def _through(self, end: int, player: int, piece_type: int, origin: int) -> set[int]:
        """A capture-mode landing on a fold: the far fold's square."""
        to = self.portals.dest[end]
        if not self.portals.admitted(end, player, piece_type):
            return set()
        if to < 0 or to not in self.fold_square:
            return set()
        far = self.fold_square[to]
        return {far} if self._lands(player, piece_type, origin, far) else set()

    def added(self, player: int, piece_type: int, origin: int) -> int:
        """How many capture squares the doors add to this piece's on ``origin``."""
        spec, portals = self.spec, self.portals
        frame = player * spec.n_facings
        found: set[int] = set()
        for number, (squares, modes, *_rest) in enumerate(spec.leaps[frame][piece_type][origin]):
            dirs = portals.leap_dirs[frame][piece_type][origin][number]
            for index, square in enumerate(squares):
                if square in self.end_at:
                    found |= self._exits(self.end_at[square], dirs[index], player, piece_type, origin)
                elif modes[index] & _CAPTURE and square in self.fold_at:
                    found |= self._through(self.fold_at[square], player, piece_type, origin)
        for number, (squares, modes, *_rest) in enumerate(spec.slides[frame][piece_type][origin]):
            dirs = portals.slide_dirs[frame][piece_type][origin][number]
            for index, square in enumerate(squares):
                if square in self.end_at:
                    found |= self._exits(self.end_at[square], dirs[index], player, piece_type, origin)
                    break
                if modes[index] & _CAPTURE and square in self.fold_at:
                    found |= self._through(self.fold_at[square], player, piece_type, origin)
        if not found:
            return 0
        return len(found - _capture_landings(spec, frame, piece_type, origin))


def portal_extra(spec: CompiledSpec) -> tuple[float, ...]:
    """Per type, the transit's share of the mean capture reach (zero for every type and every
    game without portals). See the module docstring for the reading."""
    portals = spec.portals
    if portals is None:
        return tuple(0.0 for _ in range(spec.n_types))
    doors = _Doors.of(spec, portals)
    extra = [0] * spec.n_types
    for player in (0, 1):
        for piece_type in range(portals.declared_types):
            for origin in range(spec.n_squares):
                if origin not in doors.end_at:
                    extra[piece_type] += doors.added(player, piece_type, origin)
    return tuple(count / (2 * spec.n_squares) for count in extra)


def portal_reach_values(spec: CompiledSpec, derived: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """``derived`` with the transit's reach added to every priced type, on its own scale.
    ``rust/src/eval.rs::portal_reach_values`` is the core's; see the module docstring."""
    extra = portal_extra(spec)
    endless = inexhaustible_types(spec)
    priced = [
        i for i in range(spec.n_types) if i not in spec.royal_types and i not in endless
    ]
    measured = _measured_values(spec)
    reaches = [measured[i] for i in priced]
    scale = min(reaches) if reaches and min(reaches) > 0 else 1.0
    out: list[Any] = []
    for piece_type, value in enumerate(derived):
        if piece_type in priced:
            out.append(value + REACH_SHARE * (extra[piece_type] / scale))
        else:
            out.append(value)
    return tuple(out)
