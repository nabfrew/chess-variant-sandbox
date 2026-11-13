"""A hop's `range` and `from_rank` in the Fairy-Stockfish definition (Q1217).

R-S10b1 found `fairy/variant.py` writing every hop as `p` or `g` with no distance, so
cannon shogi's flying cannons, whose diagonal hop lands at most two squares off (the screen
next to them, the landing beyond), crossed as `pB`: a cannon that flies the whole diagonal.
The same code path also dropped a `from_rank` condition on a hop and on a lame, writing a
move only the first rank may make as a move from anywhere.

Fairy-Stockfish counts a hopper's digit from the origin with the screen among it, as our
compiler counts a hop's `range` (probed here on `pyffish`): `pB3` over a screen one square
off lands two or three off, over one two squares off lands three off. Range 1 leaves no
square past any screen, and `pB1` there behaves as `pB2`, so it is refused.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.fairy import betza, position
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
DIAGONALS = {(1, 1), (-1, 1), (1, -1), (-1, -1)}
ORTHOGONALS = {(1, 0), (-1, 0), (0, 1), (0, -1)}


def _pyffish() -> Any:
    return pytest.importorskip("pyffish", reason="pyffish is an optional dev dependency")


def _ranged_xiangqi(rng: int = 3) -> dict[str, Any]:
    """Xiangqi whose cannon takes by a hop at most ``rng`` squares long."""
    spec = load_spec(ROOT / "rulesets" / "xiangqi.json")
    for piece in spec["pieces"]:
        if piece["id"] == "C":
            for move in piece["moves"]:
                if move["kind"] == "hop":
                    move["range"] = rng
    return spec


def _cannon_line(spec: dict[str, Any]) -> str:
    variant = translate(spec)
    letter = variant.chars[[p["id"] for p in spec["pieces"]].index("C")]
    lines = [line for line in variant.config.splitlines() if line.endswith(tuple("0123456789R"))]
    return next(line for line in lines if f"{letter.lower()}:" in line)


class TestTheNotation:
    def test_a_ranged_hop_carries_its_digit(self) -> None:
        assert betza.term(DIAGONALS, distance=2, hop="p") == "pB2"
        assert betza.term(ORTHOGONALS, distance=2, hop="p") == "pR2"
        assert betza.term(ORTHOGONALS, distance=9, mode="capture", hop="p") == "cpR9"
        assert betza.term(DIAGONALS, distance=3, hop="g") == "gB3"

    def test_an_unlimited_hop_is_the_bare_rider(self) -> None:
        assert betza.term(DIAGONALS, distance=0, hop="p") == "pB"
        assert betza.term(ORTHOGONALS, distance=0, hop="g") == "gR"

    def test_a_hop_that_can_never_land_or_rides_past_a_digit_is_not_written(self) -> None:
        assert betza.term(DIAGONALS, distance=1, hop="p") is None
        assert betza.term(DIAGONALS, distance=10, hop="p") is None

    def test_a_first_rank_hop_or_lame_carries_the_initial_flag(self) -> None:
        assert betza.term(ORTHOGONALS, distance=0, mode="move", hop="p", initial=True) == "mipR"
        knight = {(1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (-2, 1), (2, -1), (-2, -1)}
        assert betza.term(knight, mode="move", lame=True, initial=True) == "minN"


class TestTheTranslation:
    def test_cannon_shoguis_flying_cannons_keep_their_range(self) -> None:
        config = translate(load_spec(ROOT / "labelled" / "cannon-shogi.json")).config
        pieces = [line.split(":", 1)[1] for line in config.splitlines()
                  if line.startswith("customPiece")]
        assert pieces.count("mRpRmFpB2") == 2
        assert pieces.count("mBpBmWpR2") == 2
        assert "mRpRmFpB" not in pieces and "mBpBmWpR" not in pieces

    def test_the_ranged_cannon_is_written_with_its_range(self) -> None:
        assert _cannon_line(_ranged_xiangqi(3)).endswith(":mRcpR3")

    def test_a_range_one_hop_is_refused_by_name(self) -> None:
        with pytest.raises(Untranslatable) as refused:
            translate(_ranged_xiangqi(1))
        assert any("no Betza notation for a hop" in r and "range 1" in r
                   for r in refused.value.reasons)

    def test_a_first_rank_hop_is_written_initial(self) -> None:
        spec = _ranged_xiangqi(0)
        for piece in spec["pieces"]:
            if piece["id"] == "C":
                piece["moves"] = [{
                    "kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4",
                    "mode": "move", "condition": {"from_rank": 2},
                }]
        assert _cannon_line(spec).endswith(":mipR")


class TestTheirParser:
    @staticmethod
    def _reach(sf: Any, notation: str, screens: dict[str, str]) -> set[tuple[int, int]]:
        """Where a lone piece on e5 goes, as offsets, with ``screens`` placed (W pieces)."""
        name = "hr" + hashlib.sha1((notation + repr(sorted(screens.items()))).encode()).hexdigest()[:10]
        board = {"e5": "P", "b1": "K", "h9": "k", **screens}
        rows = []
        for rank in range(9, 0, -1):
            row, empty = "", 0
            for file in range(9):
                square = f"{chr(ord('a') + file)}{rank}"
                if square in board:
                    row += (str(empty) if empty else "") + board[square]
                    empty = 0
                else:
                    empty += 1
            rows.append(row + (str(empty) if empty else ""))
        fen = "/".join(rows) + " w - - 0 1"
        sf.load_variant_config(
            f"[{name}]\nmaxRank = 9\nmaxFile = 9\ncustomPiece1 = p:{notation}\n"
            f"customPiece2 = x:W\nking = k\nstartFen = {fen}\ncastling = false\n"
        )
        return {
            (ord(move[2]) - ord("e"), int(move[3:]) - 5)
            for move in sf.legal_moves(name, fen, [])
            if move.startswith("e5")
        }

    def test_the_digit_counts_from_the_origin_screen_included(self) -> None:
        sf = _pyffish()
        assert self._reach(sf, "pB2", {"f6": "x"}) == {(2, 2)}
        assert self._reach(sf, "pB2", {"g7": "x"}) == set()
        assert self._reach(sf, "pB3", {"f6": "x"}) == {(2, 2), (3, 3)}
        assert self._reach(sf, "pB3", {"g7": "x"}) == {(3, 3)}
        assert self._reach(sf, "pB", {"f6": "x"}) == {(2, 2), (3, 3), (4, 4)}
        assert self._reach(sf, "gB2", {"g7": "x"}) == set()
        assert self._reach(sf, "gB", {"g7": "x"}) == {(3, 3)}


class TestAgainstPyffish:
    def test_the_ranged_cannon_plays_the_same_game(self) -> None:
        from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

        _pyffish()
        result = verify(_ranged_xiangqi(3), PyffishBackend(), "", depth=2, games=2, plies=40)
        assert result.ok, result.summary()
        assert result.positions > 44

    def test_the_cannon_without_its_digit_is_a_different_game(self) -> None:
        """The check has teeth: `cpR` where `cpR3` belongs moves the opening's count."""
        from chess_variant_sandbox.fairy.verify import PyffishBackend

        _pyffish()
        spec = _ranged_xiangqi(3)
        variant = translate(spec)
        broken = dataclasses.replace(
            variant,
            name=variant.name + "nodigit",
            config=variant.config.replace(":mRcpR3", ":mRcpR").replace(
                f"[{variant.name}]", f"[{variant.name}nodigit]"),
        )
        assert broken.config != variant.config
        engine = Engine(copy.deepcopy(spec))
        state = engine.initial_state()
        ours = len(engine.legal_moves(state))
        backend = PyffishBackend()
        backend.load(variant, "")
        fen = position.variant_fen(variant, engine.spec, state)
        assert backend.perft(fen, 1) == ours
        backend.load(broken, "")
        assert backend.perft(position.variant_fen(broken, engine.spec, state), 1) != ours

    def test_cannon_shogi_with_its_cannons_already_flying_plays_the_same_game(self) -> None:
        """The flying cannons from the first move, so the walk meets the ranged hops
        everywhere rather than only after a promotion a random game may never make."""
        from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

        _pyffish()
        spec = load_spec(ROOT / "labelled" / "cannon-shogi.json")
        flying = {"U": "W", "A": "V", "C": "F", "I": "J"}
        for placement in spec["setup"]["placements"]:
            placement["piece"] = flying.get(placement["piece"], placement["piece"])
        result = verify(spec, PyffishBackend(), "", depth=1, games=2, plies=40)
        assert result.ok, result.summary()
        assert result.positions > 60

        # The definition before Q1217 offers eight opening moves ours does not (63 and 71).
        variant = translate(spec)
        broken = dataclasses.replace(
            variant,
            name=variant.name + "nodigit",
            config=variant.config.replace("pB2", "pB").replace("pR2", "pR").replace(
                f"[{variant.name}]", f"[{variant.name}nodigit]"),
        )
        engine = Engine(copy.deepcopy(spec))
        state = engine.initial_state()
        backend = PyffishBackend()
        backend.load(broken, "")
        fen = position.variant_fen(broken, engine.spec, state)
        assert backend.perft(fen, 1) == len(engine.legal_moves(state)) + 8
