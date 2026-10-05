"""The compiler's hexagon geometry (Q281a2), against `tests/hex_geometry.py`'s oracle.

Unreachable until the gate (Q281a4): the validator still refuses `board.shape`, so this
drives `_compile_wrap` and `_compile_square` directly with a hand-built hexagon
`WrapRule` and `obstacles = absent_squares(board)`, exactly as
`docs/reference/hex-build-plan.md` *Q281a2* asks. `tests/hex_geometry.py` is the
independent geometry it is checked against -- `tests/test_hex_geometry.py` checks that
oracle against itself and today's engine, so this file need not repeat that.

A hex board validates `setup.orientation == "rotate"` (`spec/schema.py`), which is what
makes the second player's compiled directions the 180-degree turn of the first's -- and
every piece here (`hex_geometry.PIECES`) is already closed under that turn, so the
oracle's player-agnostic vectors are the right comparison for both players.
"""

from __future__ import annotations

import random
from typing import Any

import hex_geometry as oracle
import pytest

from chess_variant_sandbox.engine.compile import (
    WrapRule,
    _compile_square,
    _compile_wrap,
)
from chess_variant_sandbox.spec.schema import WRAP_PAIRS, absent_squares, parse_square

RADII = range(1, 8)
PIECE_ORDER = ("rook", "bishop", "queen", "knight", "king")
PAIRS = ("n_s", "ne_sw", "nw_se")
ORIENTATION = "rotate"  # what a hex lattice requires (schema.py:3030)

# The 64 settings every player shares: each of the three pairs independently "none",
# "both", or one of its two named sides (`docs/reference/hex-build-plan.md` *Q281a2*,
# *Published numbers*, point 2).
SHARED_SETTINGS: list[dict[str, str]] = [
    {"n_s": n_s, "ne_sw": ne_sw, "nw_se": nw_se}
    for n_s in ("none", "n", "s", "both")
    for ne_sw in ("none", "ne", "sw", "both")
    for nw_se in ("none", "nw", "se", "both")
]
assert len(SHARED_SETTINGS) == 64


def _board(radius: int) -> dict[str, Any]:
    width = 2 * radius + 1
    return {"width": width, "height": width, "lattice": "hex", "shape": "hexagon"}


def _square(name: str, width: int, height: int) -> int:
    coords = parse_square(name, width, height)
    assert coords is not None, name
    file, rank = coords
    return rank * width + file


def _piece(name: str) -> dict[str, Any]:
    """One of `hex_geometry.PIECES`, as a spec piece with one descriptor per vector."""
    return {
        "id": name,
        "moves": [
            {"kind": "slide" if slides else "leap", "vectors": [[v[0], v[1]]]}
            for v, slides in oracle.PIECES[name]
        ],
    }


def _compiled_targets(built: tuple[Any, ...]) -> set[int]:
    """A lone piece's distinct reachable squares, read off `_compile_square`'s tables.

    None of `PIECE_ORDER` hops or sweeps, so only the leap and slide lists hold anything;
    a slide's ray square counts only where some mode is set (a min-range or a confined
    region can leave a visited square with mode 0, though nothing here declares either).
    """
    leaps, slides, hops, sweeps = built
    assert not hops and not sweeps
    out: set[int] = set()
    for squares, _modes, _unmoved in leaps:
        out.update(squares)
    for squares, modes, _unmoved in slides:
        out.update(sq for sq, mode in zip(squares, modes, strict=True) if mode)
    return out


def _oracle_targets(
    radius: int, leave: frozenset[str], cell: tuple[int, int], piece: str
) -> set[int]:
    width = 2 * radius + 1
    step = oracle.hex_step(radius, leave)
    return {oracle.square(t, width) for t in oracle.targets(step, cell, oracle.PIECES[piece])}


def _wrap_json(setting: dict[str, Any]) -> dict[str, Any]:
    """A shared or per-player setting (``{pair: value}`` or ``{pair: (p0, p1)}``) as the
    JSON `rules.wrap` would spell it.
    """
    out: dict[str, Any] = {}
    for key, value in setting.items():
        if isinstance(value, tuple):
            v0, v1 = value
            if v0 == v1:
                out[key] = v0
            else:
                out[f"{key}_by_player"] = {"0": v0, "1": v1}
        else:
            out[key] = value
    return out


def _leave(setting: dict[str, Any], player: int) -> frozenset[str]:
    resolved = {
        key: (value[player] if isinstance(value, tuple) else value)
        for key, value in setting.items()
    }
    return oracle.may_leave(resolved)


def _per_player_sample(seed: int, count: int) -> list[dict[str, tuple[str, str]]]:
    """A seeded sample of per-player settings.

    The build plan leaves the size to the worker (`docs/game-ideas.md` *Q281 specified*,
    *Questions the spec leaves open* #5); a dozen, seeded, covers every pair independently
    disagreeing between the two players without the cost of the full per-player space
    (4**3 per player, 4096 pairs of settings).
    """
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        setting: dict[str, tuple[str, str]] = {}
        for key, pair in WRAP_PAIRS.items():
            values = ("none", pair.low, pair.high, "both")
            setting[key] = (rng.choice(values), rng.choice(values))
        out.append(setting)
    return out


class TestCompileWrapPermissions:
    """`_compile_wrap(wrap, "hexagon")` against `hex_geometry.may_leave`."""

    @pytest.mark.parametrize("setting", SHARED_SETTINGS)
    def test_shared_settings(self, setting: dict[str, str]) -> None:
        rule = _compile_wrap(_wrap_json(setting), "hexagon")
        for player in (0, 1):
            leave = _leave(setting, player)
            for side in ("n", "s", "ne", "sw", "nw", "se"):
                expected = side in leave
                got = rule.may_leave(side, player) if rule else False
                assert got == expected, (setting, player, side)

    @pytest.mark.parametrize("setting", _per_player_sample(seed=281, count=12))
    def test_per_player_settings(self, setting: dict[str, tuple[str, str]]) -> None:
        rule = _compile_wrap(_wrap_json(setting), "hexagon")
        for player in (0, 1):
            leave = _leave(setting, player)
            for side in ("n", "s", "ne", "sw", "nw", "se"):
                expected = side in leave
                got = rule.may_leave(side, player) if rule else False
                assert got == expected, (setting, player, side)

    def test_inert_when_nothing_is_joined(self) -> None:
        assert _compile_wrap({"n_s": "none"}, "hexagon") is None
        assert _compile_wrap(None, "hexagon") is None
        assert _compile_wrap({}, "hexagon") is None

    def test_seam_open_is_the_topology_question(self) -> None:
        """Anyone at all crossing a pair opens both its sides for `seam_open`."""
        rule = _compile_wrap({"n_s_by_player": {"0": "n", "1": "none"}}, "hexagon")
        assert isinstance(rule, WrapRule)
        assert rule.seam_open("n") and rule.seam_open("s")
        assert not rule.seam_open("ne") and not rule.seam_open("nw")
        # But the permission itself stays one-way and per player.
        assert rule.may_leave("n", 0) and not rule.may_leave("n", 1)
        assert not rule.may_leave("s", 0) and not rule.may_leave("s", 1)


class TestTheWorkedCrossings:
    """The two hand-worked examples in `docs/game-ideas.md` *Q281 specified*, *The
    spelling*, reproduced through the compiler rather than the oracle.
    """

    def test_c11_forward_right_leaves_by_n_and_lands_on_i1(self) -> None:
        radius = 5
        width = height = 2 * radius + 1
        board = _board(radius)
        absent = absent_squares(board)
        origin = _square("c11", width, height)
        piece = {"id": "stepper", "moves": [{"kind": "leap", "vectors": [[0, 1]]}]}

        joined = _compile_wrap({"n_s": "n"}, "hexagon")
        leaps, *_ = _compile_square(
            piece, 0, origin, width, height, ORIENTATION, joined, absent, radius=radius,
        )
        targets = {t for squares, _, _ in leaps for t in squares}
        assert _square("i1", width, height) in targets

        closed = _compile_wrap({"n_s": "s"}, "hexagon")
        leaps_closed, *_ = _compile_square(
            piece, 0, origin, width, height, ORIENTATION, closed, absent, radius=radius,
        )
        assert not {t for squares, _, _ in leaps_closed for t in squares}

    def test_k3_right_leaves_by_se_and_lands_on_a8(self) -> None:
        radius = 5
        width = height = 2 * radius + 1
        board = _board(radius)
        absent = absent_squares(board)
        origin = _square("k3", width, height)
        piece = {"id": "stepper", "moves": [{"kind": "leap", "vectors": [[1, 0]]}]}

        joined = _compile_wrap({"nw_se": "se"}, "hexagon")
        leaps, *_ = _compile_square(
            piece, 0, origin, width, height, ORIENTATION, joined, absent, radius=radius,
        )
        targets = {t for squares, _, _ in leaps for t in squares}
        assert _square("a8", width, height) in targets


class TestNoSeamEqualsTheRhombusWithWalls:
    """*What the compiler resolves it into*, *No seam, no change*: with nothing joined,
    a hexagon's landing is a rhombus landing that is not on an absent cell.
    """

    @pytest.mark.parametrize("radius", RADII)
    @pytest.mark.parametrize("piece_name", PIECE_ORDER)
    def test_unjoined_matches_the_plain_frame(self, radius: int, piece_name: str) -> None:
        width = height = 2 * radius + 1
        absent = absent_squares(_board(radius))
        piece = _piece(piece_name)
        for player in (0, 1):
            for square in range(width * height):
                if square in absent:
                    continue
                hex_built = _compile_square(
                    piece, player, square, width, height, ORIENTATION, None, absent,
                    radius=radius,
                )
                grid_built = _compile_square(
                    piece, player, square, width, height, ORIENTATION, None, absent,
                )
                assert hex_built == grid_built, (radius, piece_name, player, square)


def _max_leap_length(piece_name: str) -> int:
    """The longest non-sliding vector a piece declares, in hex length."""
    return max(
        (oracle.hexdist(*v) for v, slides in oracle.PIECES[piece_name] if not slides),
        default=0,
    )


def _check_setting(radius: int, setting: dict[str, Any]) -> None:
    width = height = 2 * radius + 1
    board = _board(radius)
    absent = absent_squares(board)
    wrap = _compile_wrap(_wrap_json(setting), "hexagon")
    for piece_name in PIECE_ORDER:
        # Refusal 7: on a hexagon with any side joined, a vector longer than R + 1 is
        # refused rather than answered (`docs/game-ideas.md` *Q281 specified*,
        # *Validator refusals* #7) -- the knight's leap of 3 at R = 1, where R + 1 = 2,
        # is the one case among these five pieces (`test_hex_geometry.py`'s own
        # `test_a_step_two_copies_out_is_refused_not_answered`).
        if _max_leap_length(piece_name) > radius + 1:
            continue
        piece = _piece(piece_name)
        for player in (0, 1):
            leave = _leave(setting, player)
            for cell in oracle.cells(radius):
                square = oracle.square(cell, width)
                built = _compile_square(
                    piece, player, square, width, height, ORIENTATION, wrap, absent,
                    radius=radius,
                )
                got = _compiled_targets(built)
                expected = _oracle_targets(radius, leave, cell, piece_name)
                assert got == expected, (radius, setting, piece_name, player, cell)


class TestLandingMatchesTheOracle:
    """`_compile_square`'s hexagon landing against `hex_geometry`'s independent one, over
    the 64 shared settings and a seeded sample of per-player ones
    (`docs/reference/hex-build-plan.md` *Q281a2*).
    """

    @pytest.mark.parametrize("radius", [1, 2, 3])
    @pytest.mark.parametrize("setting", SHARED_SETTINGS)
    def test_shared_settings(self, radius: int, setting: dict[str, str]) -> None:
        _check_setting(radius, setting)

    @pytest.mark.slow
    @pytest.mark.parametrize("radius", [4, 5, 6, 7])
    @pytest.mark.parametrize("setting", SHARED_SETTINGS)
    def test_shared_settings_larger_radii(self, radius: int, setting: dict[str, str]) -> None:
        _check_setting(radius, setting)

    @pytest.mark.parametrize("setting", _per_player_sample(seed=281, count=12))
    def test_per_player_sample_at_r3(self, setting: dict[str, tuple[str, str]]) -> None:
        _check_setting(3, setting)

    @pytest.mark.slow
    @pytest.mark.parametrize("setting", _per_player_sample(seed=281, count=12))
    def test_per_player_sample_at_r5(self, setting: dict[str, tuple[str, str]]) -> None:
        _check_setting(5, setting)
