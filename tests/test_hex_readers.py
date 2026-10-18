"""Describe, distance, relate and transform on a hexagon (Q281a3).

The validator still refuses `board.shape` and the pair keys (that is Q281a4), so nothing
outside these tests reaches the hexagon branches. This file proves them at the function
level, on hand-built dicts, against `docs/game-ideas.md` *Q281 specified*, and proves that
every committed and archived spec reads exactly as it did before them.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from retired_specs import hex_glinski_promoting

from chess_variant_sandbox.spec.describe import (
    HEXAGON_WRAP_CHOICES,
    _board_words,
    _describe_wrap,
    describe_spec,
    rule_choices,
)
from chess_variant_sandbox.spec.distance import (
    _promotion_zone_share,
    distance,
    features,
)
from chess_variant_sandbox.spec.relate import identity_key, relate
from chess_variant_sandbox.spec.schema import _WRAP_KEYS, WRAP_PAIRS, absent_squares, wrap_axes
from chess_variant_sandbox.spec.transform import resize_board, set_wrap

ROOT = Path(__file__).resolve().parents[1]
# The library as it stood before the hexagon: every committed and archived spec that
# declares no `board.shape`. Schema 16's showcases (Q281a4) are the ones that do, and
# `tests/test_schema_sixteen.py` holds them to what they claim instead.
LIBRARY = [
    path
    for path in sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))
    if "shape" not in json.loads(path.read_text(encoding="utf-8"))["board"]
]
PAIRS = tuple(WRAP_PAIRS)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _parent() -> dict[str, Any]:
    """Glinski's hexagon as the library has it: an 11 by 11 rhombus cut by thirty walls."""
    return hex_glinski_promoting()


def _twin(wrap: dict[str, Any] | None = None) -> dict[str, Any]:
    """The same game on a native hexagon: `shape` in place of the thirty walls."""
    spec = _parent()
    del spec["rules"]["obstacles"]
    spec["board"]["shape"] = "hexagon"
    if wrap is not None:
        spec["rules"]["wrap"] = wrap
    return spec


def _hexagon(radius: int, wrap: dict[str, Any] | None = None) -> dict[str, Any]:
    size = 2 * radius + 1
    spec: dict[str, Any] = {
        "board": {"width": size, "height": size, "lattice": "hex", "shape": "hexagon"},
        "rules": {},
    }
    if wrap is not None:
        spec["rules"]["wrap"] = wrap
    return spec


def _wrap_lines(wrap: dict[str, Any], **rules: Any) -> list[str]:
    spec = _hexagon(5, wrap)
    spec["rules"].update(rules)
    return _describe_wrap(spec, spec["rules"])


def test_the_twin_is_what_the_spec_says() -> None:
    parent, twin = _parent(), _twin()
    assert parent["rules"]["obstacles"]["squares"]
    assert "obstacles" not in twin["rules"]
    # The twin's missing cells are the parent's walls, cell for cell. Worked out here from
    # the names (file letter, rank number, `width` per rank) and the 180-degree mirror the
    # parent states, without the package's own square reader.
    width = parent["board"]["width"]
    cells = width * parent["board"]["height"]
    walls = {
        (int(name[1:]) - 1) * width + (ord(name[0]) - ord("a"))
        for name in parent["rules"]["obstacles"]["squares"]
    }
    assert parent["rules"]["obstacles"]["mirror"] is True
    walls |= {cells - 1 - cell for cell in walls}
    assert len(walls) == 30
    assert set(absent_squares(twin["board"])) == walls


# ------------------------------------------------------------------------ describe


class TestTheBoardSentence:
    def test_glinskis_hexagon(self) -> None:
        assert _board_words(_twin()) == (
            "a hexagonal board of 91 cells, six on a side; every cell touches six others"
        )

    def test_the_walled_parent_still_counts_its_walls(self) -> None:
        assert _board_words(_parent()) == (
            "a hexagonal board of 91 cells, an 11 by 11 hex grid with 30 cut off; every "
            "cell touches six others"
        )

    @pytest.mark.parametrize("radius", range(1, 8))
    def test_every_radius(self, radius: int) -> None:
        words = ["two", "three", "four", "five", "six", "seven", "eight"][radius - 1]
        assert _board_words(_hexagon(radius)) == (
            f"a hexagonal board of {3 * radius * radius + 3 * radius + 1} cells, {words} "
            "on a side; every cell touches six others"
        )

    def test_a_wall_inside_the_hexagon_is_cut_off_and_the_corners_are_not(self) -> None:
        spec = _twin()
        spec["rules"]["obstacles"] = {"squares": ["f6"], "mirror": True}
        assert _board_words(spec) == (
            "a hexagonal board of 90 cells, six on a side with 1 cut off; every cell "
            "touches six others"
        )


class TestTheWrapSentences:
    def test_a_pair_joined_both_ways(self) -> None:
        assert _wrap_lines({"nw_se": "both"}) == [
            "The board has no upper-left or lower-right side: a piece that leaves by one "
            "comes back in by the other, straight across.",
            "Either player's pieces may cross that join in either direction.",
        ]

    def test_one_side_for_both_players(self) -> None:
        assert _wrap_lines({"ne_sw": "ne"}) == [
            "The board has no lower-left or upper-right side: a piece that leaves by one "
            "comes back in by the other, straight across.",
            "Either player's pieces may cross that join only by leaving by the upper-right "
            "side and coming back in by the lower-left one.",
        ]

    def test_each_player_crossing_forward(self) -> None:
        assert _wrap_lines({"n_s_by_player": {"0": "n", "1": "s"}}) == [
            "The board has no bottom or top side: a piece that leaves by one comes back "
            "in by the other, straight across.",
            "The first player's pieces may cross that join only by leaving by the top side "
            "and coming back in by the bottom one.",
            "The second player's pieces may cross that join only by leaving by the bottom "
            "side and coming back in by the top one.",
        ]

    def test_a_seam_only_one_player_may_use_is_still_a_seam(self) -> None:
        lines = _wrap_lines({"nw_se_by_player": {"0": "both", "1": "none"}})
        assert lines[0].startswith("The board has no upper-left or lower-right side")
        assert lines[1:] == [
            "The first player's pieces may cross that join in either direction.",
            "The second player's pieces may cross that join not at all.",
        ]

    def test_the_torus(self) -> None:
        lines = _wrap_lines({key: "both" for key in PAIRS})
        assert len(lines) == 7
        assert lines[-1] == (
            "Every line through a cell runs through every other cell before it comes back."
        )

    def test_two_pairs_are_not_the_torus(self) -> None:
        lines = _wrap_lines({"n_s": "both", "ne_sw": "both"})
        assert len(lines) == 4
        assert not any("every other cell" in line for line in lines)

    def test_a_pair_with_nothing_joined_says_nothing(self) -> None:
        assert _wrap_lines({"n_s": "none", "ne_sw_by_player": {"0": "none", "1": "none"}}) == []

    def test_the_rectangles_keys_say_nothing_on_a_hexagon(self) -> None:
        assert _wrap_lines({"files": "both", "ranks": "top"}) == []

    @pytest.mark.parametrize(("key", "said"), [("n_s", False), ("ne_sw", True), ("nw_se", True)])
    def test_castling_is_said_to_stay_on_its_rank_where_a_rank_step_crosses(
        self, key: str, said: bool
    ) -> None:
        lines = _wrap_lines({key: "both"}, castling={"king_steps": 2})
        assert (lines[-1].startswith("Castling does not use the join")) is said


class TestTheWrapCard:
    def test_one_group_per_pair_each_spelled_as_an_axis_is(self) -> None:
        assert [
            [option["keys"] for option in group["options"]] for group in HEXAGON_WRAP_CHOICES
        ] == [[(key,), (f"{key}_by_player",)] for key, _ in wrap_axes("hexagon")]

    def test_attached_to_the_wrap_card_at_the_gate(self) -> None:
        """Q281a4 folds the pairs into `RULE_CHOICES`, whose keys the JSON schema now has."""
        offered = {
            key
            for group in rule_choices()["rules.wrap"]
            for option in group["options"]
            for key in option["keys"]
        }
        assert offered == set(_WRAP_KEYS)


# `describe_spec` of every committed and archived spec, computed by the code before the
# hexagon branches (`624ece8`): sha256 of the text, first sixteen hex digits.
_DESCRIPTIONS_BEFORE_THE_HEXAGON: dict[str, str] = {
    "archive/andernach": "ffbf81cb033dc03c",  # Q540, schema 17; 2026-10-18: its description, written as the game's idea (was 62f0b7d191b8acf6)
    "archive/circe": "4cb05e7cd7912979",  # Q540, schema 17; 2026-10-18: its description, written as the game's idea (was dd25abf2bc7c6f21)
    "archive/kamikaze": "38d6273d3ddc7a2d",  # Q540, schema 17; 2026-10-18: its description, written as the game's idea (was 86810a81183707bc)
    "archive/rifle": "c59b317de29cd328",  # Q540, schema 17; 2026-10-18: its description, written as the game's idea (was 7ea1669e4e1aa9e2); 2026-10-18: its description's first sentence, for the new-games card (was e01240bc0637b02d)
    "archive/einstein": "b32d9486a7a3b562",  # Q781, schema 18; 2026-10-18: its description, written as the game's idea (was d3b6b02f8466f39b)
    "archive/omega": "e5069b0021e084e9",  # Q781, schema 18; 2026-10-18: its description, written as the game's idea (was 9c2b1ac6dbc35c11)
    "archive/extinction-chess": "595c52228ce929d1",  # Q783, schema 18; Q870: its name (was ed5f3e3428a44bcc); 2026-10-18: its description, written as the game's idea (was 4519aa68596fcce2); 2026-10-18: its description's first sentence, for the new-games card (was bf9ec657cbf572b5)
    "archive/grand": "58f1acd3d29274b5",  # Q783, schema 18; 2026-10-18: its description, written as the game's idea (was 8cd64b4fc05a23fe)
    "archive/placement-chess": "5e460e034efb1f73",  # Q784, schema 18; Q870: its name (was 86a5bb2444f119a3); 2026-10-18: its description, written as the game's idea (was ffe73cd78d4fcd2b)
    "archive/sittuyin": "7621d3cdfcdf85c0",  # Q784, schema 18; 2026-10-18: its description, written as the game's idea (was 51b0f5043e170580)
    "archive/kyoto": "c0da992841f63d86",  # Q785, schema 18; 2026-10-18: its description, written as the game's idea (was 042d5726eb291356)
    "archive/jeson-mor": "23ae0dec21f1f3e6",  # Q786, schema 18; Q870: its name (was 051d25a135b04a2c); 2026-10-18: its description, written as the game's idea (was a0e909de195b7cdd)
    # Q870, 2026-10-13: `archive/hex-glinski-turned` and `-turned-mirrored` retired for `pinwheel`,
    # which has a board shape and so is not read here (rows were 95f213b4e8e9fdbe and 93933865b7a20464).
    "archive/shogi-drop-promoted": "4c2fa92391c9501c",  # Q787; Q868: its description (was b9a76d36e4533433); Q870: its name (was 39a59b08b86fe4c4); 2026-10-18: its description, written as the game's idea (was 05d18683734a423a)
    "archive/shogi-backward-drops": "ae66fc9f669fa28d",  # Q787; Q870: its name (was 7146e06d789e999e); 2026-10-18: its description, written as the game's idea (was 20c148872b016efc)
    "archive/pawn-knockdown": "2bbaf9d86277517d",  # Q787; 2026-10-18: its description, written as the game's idea (was 16b16f92f478e283)
    "archive/divide": "899d104386d57d7c",  # Q772a, schema 19; Q838c, ends by square (was 9bb463b570c037d6); 2026-10-18: its description, written as the game's idea (was 15a420404366711e)
    "archive/ferry": "931121d3e0f76174",  # Q772a, schema 19; Q838c, ends by square (was 78f34d021bf69c60); Q868: its description (was 70ab6aac9fbc846d); Q870: its description names a renamed sibling (was 5e31c8eaabf21a3f); 2026-10-18: its description, written as the game's idea (was 4f48a98c36e1d7e0)
    "archive/ferry-no-straight-back": "4936891a9dc6dd52",  # Q807, schema 20; Q838c, ends by square (was 6496a58a9f77bd62); Q868: its description (was b699b6c35bdc886e); Q870: its name (was 278d1cb91060be73); 2026-10-18: its description, written as the game's idea (was 5623aae081bef87d)
    "archive/wolves-patrol": "d7b2fa01e6cb0796",  # Q850, schema 23; Q868: its description (was f66264efa6c41bae); Q870: its name (was 2c530701f62d18cb); 2026-10-18: its description, written as the game's idea (was d55a2b2a6e6d718a)
    "archive/carousel": "5c69152360142200",  # Q772a, schema 19; Q838c, ends by square (was adca05bb14f01944); 2026-10-18: its description, written as the game's idea (was 8f53c1f079a65be6)
    "archive/wormhole": "96e2749b5738e8b2",  # Q772a, schema 19; Q838c, ends by square (was f9e39fb07aaf10ef); Q868: its description (was c472a28083096dd3); 2026-10-18: its description, written as the game's idea (was 0fc9054ce1fcadd3)
    "archive/shut-wormhole": "dbf678f748e98955",  # Q772a, schema 19; Q838c, ends by square (was 899a44ea1a70e016); Q868: its description (was 88f92f6683674ae0); 2026-10-18: its description, written as the game's idea (was 5d7ca7fae57d4ab3); 2026-10-18: its description's first sentence, for the new-games card (was 80a43eccbc991b7a)
    "archive/cylinder-wormhole": "1078f5989a085fcc",  # Q772a, schema 19; Q838c, ends by square (was 956c4052a0efe3b2); Q868: its description (was 7754feef3dc87a15); 2026-10-18: its description, written as the game's idea (was 376857daebed08d9)
    "archive/cannonade": "fc242c00df9a0b1d",
    "archive/chess-sweeping-bishops": "b499391362e4a7dd",  # Q614 (W7): its description, for a player; Q870: its name (was e1fbb67c85b726f2); 2026-10-18: its description, written as the game's idea (was be9ce1bb3ec7ee78)
    "archive/hierarchy-extended": "9c383a159efb7b2f",  # Q614 (W7): its description, for a player; Q868: its description (was 087f7e3e22095b4c); Q870: its name (was bb3b5cc5749a560a); 2026-10-18: its description, written as the game's idea (was 0983c06122def826)
    "archive/hierarchy-fall": "a2829cb54aa455e6",  # Q870: its name (was f1d5f72ada58ae25); 2026-10-18: its description, written as the game's idea (was c037a03e23f19bb7)
    "archive/hierarchy-rise": "3ca2b6b7510e4f84",  # Q868: its description, what its difference does in play (was c721be4ecb12edac); Q870: its name (was 788cb832f824d4c5); Q962: its description, for a player (was b9bbbd571d6dc0cf); 2026-10-18: its description, written as the game's idea (was dd2e8a7b1aeb4c23)
    "archive/hierarchy-rise-stay": "fe6a625f7a464c41",  # Q868: its description, what its difference does in play (was d35f81c844ab404f); Q870: its name (was 3e40bfc19ce12d65); 2026-10-18: its description, written as the game's idea (was 57e401ef6336674b)
    "archive/hierarchy-rise-throw": "f5d554048e89d440",  # Q868: its description, what its difference does in play (was 88b88b82ae4f29b7); Q870: its name (was e2b782eb98c6fe7b); Q962: its description, for a player (was b33ef01792e81397); 2026-10-18: its description, written as the game's idea (was f01d96f758063c42)
    "archive/marchfield": "89eefc6a7516e09b",
    "archive/merger": "526768b09ed61d0a",  # 2026-10-18: its description, written as the game's idea (was 775d28c68e353d04)
    "archive/perpetual": "d97bd86bed5df13e",  # Q614 (W7): its description, for a player
    "archive/quarry": "b0f0905f796a54cc",  # 2026-10-18: its description, written as the game's idea (was 60c027a1e10f03e8)
    "archive/queen-bomb": "743c40c222e854ce",  # Q868: its description, what its difference does in play (was 09626cff98a6e6da); 2026-10-18: its description, written as the game's idea (was 603f5a2cedbda769); 2026-10-18: its description's first sentence, for the new-games card (was f5d9409f96ef3412)
    "archive/queens-wake": "74dfb2811c2d29e3",  # Q868: its description, what its difference does in play (was b3819b504d9d5fb7); 2026-10-18: its description, written as the game's idea (was 66793be5645ffbde); 2026-10-18: its description's first sentence, for the new-games card (was 39a1e7e7672f42ca)
    "archive/reckoning": "eae10ebfaeb450f3",
    "archive/stones": "18b3411e01eaa407",  # 2026-10-18: its description, written as the game's idea (was bd40338d79eed295)
    "archive/succession": "60a64497f07e456a",  # Q614 (W7): its description, for a player; 2026-10-18: its description, written as the game's idea (was 7e7db2cbe4838f68)
    "archive/three-stones": "78800ade84e9d8a5",  # Q868: its description, what its difference does in play (was e40fb77acc50ada9); 2026-10-18: its description, written as the game's idea (was 61cffe49585683c2)
    "archive/weathervane": "78b86183ad41acf0",  # Q614 (W7): its description, for a player; Q868: its description (was 4ff3b030a4c709e3); 2026-10-18: its description, written as the game's idea (was 2744ceba2d0228a6)
    "archive/weathervane-half-turn": "f0cab14e750f2a33",  # Q868: its description, what its difference does in play (was fa4e204bfcd6f14f); Q870: its name (was bde88ee9a56cd9e4); Q962: its description, for a player (was 6e700527b18ec2d0); 2026-10-18: its description, written as the game's idea (was cf8b50701c3f5d84)
    "archive/wolves": "ebf1fc481e9ba49d",  # 2026-10-18: its description, written as the game's idea (was 15b776a6a5da9ea3)
    "archive/wolves-bounty": "fd8aecb8c9d18e00",  # Q868: its description, what its difference does in play (was cc3b8b85d632ee54); Q870: its name (was 9e521a672dbe4e25); 2026-10-18: its description, written as the game's idea (was 22b8244382fe4c96)
    # Q876, 2026-10-13: `archive/wolves-immune` retired for `pitchfork` (the row was b871ae8b0daf46ac).
    "archive/pitchfork": "1bc1f4bcad35ab45",  # Q876, schema 25; 2026-10-18: its description, written as the game's idea (was a9f1a80e6430496a)
    "archive/sluice": "43a7459253aa92fb",  # Q844b, schema 21; 2026-10-18: its description, written as the game's idea (was 6468eb19d0ea6cf1)
    "archive/crease": "7075979eadfa0447",  # Q844b, schema 24; 2026-10-18: its description, written as the game's idea (was 39881186e149674a)
    "archive/pawnproof-wormhole": "7ca6672944785cd4",  # Q844b, schema 21; 2026-10-18: its description, written as the game's idea (was 4769876a3f4f76e3)
    # Q962b, 2026-10-18: `archive/gardner-in-rounds` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was 3bf914a9177e31c5.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-with-meek-pawns` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was b793c277f73875c2.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-with-the-initiative` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was d4228395719912a8.
    "archive/watering-hole": "7b9304551e1aa4e6",  # Q918e, schema 26; Q962: its description, for a player (was b06570f1fcf442e8); 2026-10-18: its description, written as the game's idea (was b2c0fe985587b544)
    "archive/gardner-in-rounds-resting": "d8b686bb1ae96bac",  # Q942c, schema 27; 2026-10-18: its description, written as the game's idea (was bdd5185ebbb0abf9)
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-where-swaps-collide` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was 5ecf690c98ed62f8.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-scored` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was 21bb31cf3d019bd6.
    "archive/breather": "e8fa3ceebdd3d5a3",  # Q942c, schema 27; Q962: its file, name and description (was archive/gardner-in-rounds-against-the-chase, e6b51ce8b7b2e11b); 2026-10-18: its description, written as the game's idea (was 77bebee4de9a7bd8)
    # Q962b, 2026-10-18: `archive/gardner-with-spies` retired from the previews (`docs/reference/previews-2026-10-18.md`); the row was d42324fc3f1bb1be.
    "archive/gardner-with-spies-resting": "816d0aff92f03381",  # Q935f, schema 27; 2026-10-18: its description, written as the game's idea (was 6ef889f11a8119cc)
    "archive/riders-and-leapers": "358b832b1209f1d3",  # Q936s, schema 27
    "rulesets/almost": "0ea224823977572d",
    "rulesets/antichess": "74c73dff0811c4a6",
    "rulesets/atomic": "e77e414d76c510af",
    "rulesets/bastion": "74c12b0fe52b876b",
    "rulesets/berolina": "733c98e8156d9fb6",  # Q614 (W7): its description, for a player; 2026-10-18: its description, written as the game's idea (was 32995edcf561f5e3)
    "rulesets/capablanca": "536dd74dd518d9da",  # 2026-10-18: its description, written as the game's idea (was d828fec94fd054ae)
    "rulesets/chess": "9d9595bc228f1a31",  # Q614 (W7): its description, for a player
    "rulesets/chess-oriented-pawns": "853eb17527f25798",  # Q868: its description, what its difference does in play (was e6142ab640c179da); Q870: its name (was e47286e8484e42c2); 2026-10-18: its description, written as the game's idea (was e5f76b8db6fdd732)
    "rulesets/courier": "e05e5ce959b9dc4a",
    "rulesets/crazyhouse": "16e10ba0206002db",
    "rulesets/crazyhouse-torus": "3e32273fcfbf13c3",  # Q869: the rear pawn respelled as a fixed facing, its pieces, setup and description (was 7ce10112bc1451ad); Q868: its description, what its difference does in play (was bade932232d89bc0); Q870: its name (was b729ce8de185148e); 2026-10-18: its description, written as the game's idea (was 2081116914528e2a)
    "rulesets/cylinder": "05ba8b9746256db8",  # Q614 (W7): its description, for a player
    "rulesets/dobutsu": "27c16c4bde91223c",
    "rulesets/dull": "69ec95c7a39e4b21",  # Q614 (W7): its description, for a player
    "rulesets/gardner": "bbf23b40daa0f1f6",
    "rulesets/horde": "2017410b2198a59f",  # 2026-10-18: its description, written as the game's idea (was abc825784f046f91)
    "rulesets/intransitive": "9a987297d2ca4101",  # Q696: its draw clock 100 -> 200 (was 37539e3b411a0217); 2026-10-18: its description, written as the game's idea (was 53f11ae7dc3846da)
    "rulesets/kingofthehill": "4fac9c48c5b8a3bb",
    "rulesets/knightmate": "85ebf632b6c8ee77",
    "rulesets/losalamos": "db5abe9fee8431d2",
    "rulesets/makruk": "c49065a985ace362",
    "rulesets/onewaycylinder": "4adbd11f9fcc7c1d",  # Q868: its description, what its difference does in play (was b75662db94554476); 2026-10-18: its description, written as the game's idea (was cfdba2286a0599e5)
    "rulesets/onewaytorus": "00911754603d811d",  # Q868: its description, what its difference does in play (was df8b34635ed44d36); 2026-10-18: its description, written as the game's idea (was ef3a79c884c820b5)
    "rulesets/racingkings": "320df46297d70604",
    "rulesets/rps-hierarchy-chess": "36fdaff7d499e344",  # Q962: its description, for a player (was b49fdbd4c30baca8)
    "rulesets/rps-v1": "2054c4c7aa052c77",  # Q614 (W7): its description, for a player; Q962: its description, for a player (was d482b0f390f351bc)
    "rulesets/rps-v2": "03b8a95dd5edad04",  # Q962: its description, for a player (was b36804702b937487); 2026-10-18: its description, written as the game's idea (was 408d270288469d87)
    "rulesets/rps-v3": "faf58cbf931bfa7d",  # Q614 (W7): its description, for a player; Q962: its description, for a player (was 743a34fb916aeea0); 2026-10-18: its description, written as the game's idea (was 0a856e0d2d334a31)
    "rulesets/rps-v4": "6e3de97a74e242a0",  # Q962: its description, for a player (was 9143427c049ab048); 2026-10-18: its description, written as the game's idea (was e6af1b103a4f3389)
    "rulesets/rps-v5": "192b8a8a18be50d4",  # Q614 (W7): its description, for a player; Q962: its description, for a player (was 79d5d8b4b48ee0c3); 2026-10-18: its description, written as the game's idea (was 3f56a66baeef47d7)
    "rulesets/sardines": "e3cdf99cf0097bd6",  # Q614 (W7): its description, for a player; 2026-10-18: its description, written as the game's idea (was 63de85e559ca5e19)
    "rulesets/shatranj": "828ec1dae768fb98",  # Q614 (W7): its description, for a player
    "rulesets/shogi": "1d9b14e68510b70a",
    "rulesets/threecheck": "b20a73916fc5ffa6",
    "rulesets/torus": "862c66721e158df7",  # Q868: its description, what its difference does in play (was b04e8dda82800fe1); Q870: its description names a renamed sibling (was d9c59025017f94c5); 2026-10-18: its description, written as the game's idea (was 4762590e8cff1d7e)
    "rulesets/torus-less-extended": "bb925d1cc7b11aba",  # Q869: the rear pawn respelled as a fixed facing, its pieces, setup and description (was da67440bc118259c); Q868: its description, what its difference does in play (was ca1ff7ab3635f952); Q870: its name (was 5d8ae197d82a2302); 2026-10-18: its description, written as the game's idea (was fe44d3b1896efe3d)
    "rulesets/xiangqi": "3099db784f4f8484",  # 2026-10-18: its description, written as the game's idea (was c661457848db5457)
}


_WORDED_SINCE: tuple[tuple[str, str], ...] = (
    (" turns without a capture or other lasting change,", " plies without progress,"),
    (" turns in all.", " plies total."),
    ("a royal piece taken or a whole army gone", "a royal captured or an army annihilated"),
)


class TestEveryDescriptionIsUnchanged:
    def test_the_freeze_covers_the_library(self) -> None:
        assert set(_DESCRIPTIONS_BEFORE_THE_HEXAGON) == {_key(path) for path in LIBRARY}

    @pytest.mark.parametrize("path", LIBRARY, ids=_key)
    def test_every_committed_spec(self, path: Path) -> None:
        text = describe_spec(_load(path))
        # Worded on purpose since the freeze, and nothing else: Q610 (G4 of
        # `docs/archive/plan-understanding.md` §3.3) counts the clocks in turns and says the
        # capture ending plainly. Put back, the text is the frozen one. Q614 (W7) rewrote
        # nineteen descriptions for a player and re-froze those rows, the describer unchanged.
        for now, then in _WORDED_SINCE:
            text = text.replace(now, then)
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        assert digest == _DESCRIPTIONS_BEFORE_THE_HEXAGON[_key(path)]


# ------------------------------------------------------------------------ distance


class TestDistance:
    def test_no_feature_reads_the_shape(self) -> None:
        """The twin and its parent differ by their walls alone.

        `has_obstacles` is the one feature that says so. `rule_mass` and `special_rules`
        come from `spec/complexity.py`, which charges the parent for its `obstacles` rule;
        `board.shape` is a board key and costs nothing (the spec's *other surfaces* row).
        """
        parent, twin = features(_parent()), features(_twin())
        apart = {name for name in parent if parent[name] != twin[name]}
        assert apart == {"has_obstacles", "rule_mass", "special_rules"}
        assert (parent["has_obstacles"], twin["has_obstacles"]) == (1.0, 0.0)

    def test_the_zone_is_counted_over_the_cells(self) -> None:
        parent, twin = _parent(), _twin()
        assert _promotion_zone_share(twin) == _promotion_zone_share(parent) > 0
        width = twin["board"]["width"]
        bare = copy.deepcopy(twin)
        del bare["board"]["shape"]
        assert _promotion_zone_share(bare) == pytest.approx(
            _promotion_zone_share(twin) * 91 / (width * width)
        )

    @pytest.mark.parametrize("key", PAIRS)
    def test_each_pair_is_its_own_feature(self, key: str) -> None:
        joined = features(_twin({key: "both"}))
        assert joined[f"wraps_{key}"] == 1.0
        assert all(joined[f"wraps_{other}"] == 0.0 for other in PAIRS if other != key)
        assert joined["wraps_files"] == joined["wraps_ranks"] == 0.0
        assert joined["wraps_asymmetrically"] == 0.0
        assert distance(_twin(), _twin({key: "both"})) > 0

    def test_one_way_is_a_joined_pair_and_an_asymmetry(self) -> None:
        joined = features(_twin({"n_s_by_player": {"0": "n", "1": "s"}}))
        assert joined["wraps_n_s"] == joined["wraps_asymmetrically"] == 1.0

    @pytest.mark.parametrize("path", LIBRARY, ids=_key)
    def test_no_committed_spec_joins_a_pair(self, path: Path) -> None:
        measured = features(_load(path))
        assert [measured[f"wraps_{key}"] for key in PAIRS] == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------------ relate


class TestRelate:
    def test_the_twin_and_its_parent_are_one_node(self) -> None:
        assert identity_key(_twin()) == identity_key(_parent())
        assert relate(_parent(), _twin()).differences == ()

    def test_a_seam_is_the_one_difference(self) -> None:
        units = [d.unit for d in relate(_twin(), _twin({"nw_se": "both"})).differences]
        assert units == ["board.wrap"]

    def test_a_wall_inside_the_hexagon_is_still_a_wall(self) -> None:
        walled = _twin()
        walled["rules"]["obstacles"] = {"squares": ["f6"], "mirror": True}
        units = [d.unit for d in relate(_twin(), walled).differences]
        assert units == ["board.walls"]


# ------------------------------------------------------------------------ transform


class TestTransform:
    def test_a_resized_hexagon_keeps_its_shape(self) -> None:
        resized = resize_board(_twin(), 13, 13)
        assert resized["board"] == {
            "width": 13, "height": 13, "lattice": "hex", "shape": "hexagon"
        }

    def test_a_resized_grid_gains_no_shape(self) -> None:
        resized = resize_board(_parent(), 13, 13)
        assert resized["board"] == {"width": 13, "height": 13, "lattice": "hex"}

    @pytest.mark.parametrize("key", PAIRS)
    def test_set_wrap_writes_a_pair(self, key: str) -> None:
        assert set_wrap(_twin(), key)["rules"]["wrap"] == {key: "both"}

    @pytest.mark.parametrize("axis", ["files", "ranks"])
    def test_set_wrap_refuses_an_axis_on_a_hexagon(self, axis: str) -> None:
        with pytest.raises(ValueError, match="n_s, ne_sw, nw_se"):
            set_wrap(_twin(), axis)

    @pytest.mark.parametrize("key", PAIRS)
    def test_set_wrap_refuses_a_pair_on_a_grid(self, key: str) -> None:
        with pytest.raises(ValueError, match="files, ranks"):
            set_wrap(_parent(), key)

    def test_set_wrap_on_a_grid_is_unchanged(self) -> None:
        assert set_wrap(_parent())["rules"]["wrap"] == {"files": "both"}
