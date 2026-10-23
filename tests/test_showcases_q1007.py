"""V1, the mechanism of Q1007f's command showcases (`docs/plan-schema-31.md` §6.3).

**K1**, *chess with commanding knights* (`adc1a91b0a83`): chess, and every enemy piece a knight
could jump to (kings excepted) may also be moved by the knight's owner, to empty squares only.
**K2**, *chess with a spellbinding king* (`6a8120390853`): chess, and an enemy piece next to a
king may be moved by the king's owner, to an empty square or onto its own side's non-royal
pieces (`captures_own`). Both were measured and retired by Q1007f (2026-10-24: each missed
V2, a commanded move offered too rarely), so their files are `tests/_retired_*.json`.

V1 as §6.3 writes it, held here for the two files:
- each file is chess with exactly its registered key, so its hash is that of Q1007c's test spec
  (`tests/_command_k1.json`, `_command_k2.json`), and every hand-built position of
  `tests/test_command.py` (the check answered by commanding the queen beside the king away in
  K2, the discovered check by command in K1, the commanded move exposing the commander's royal
  not offered, a frozen commanded piece not offered, no castling, promotion or en passant by
  command, the commanded rook losing its castling right) is a position of these games;
- over seeded random games (three of each here, twenty under `slow`), the core's legal list
  equals the sight oracle's after every move, and every commanded move played is sound: the
  piece is the opponent's and not royal, it takes none of the commander's pieces, and a
  turncoat capture happens only where `captures_own` is declared (never in K1).

Nothing here decides a move's legality (CLAUDE.md, rule 3): the lists are the core's, and the
oracle (`tests/sight_oracle.py`) is the declared second reading.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import test_command
from retired_specs import RETIRED_PREVIEWS, retired_preview
from test_command_walks import walk

from chess_variant_sandbox.spec import load_spec, spec_hash

HERE = Path(__file__).resolve().parent
CHESS = HERE.parent / "rulesets" / "chess.json"
KNIGHT_COMMAND = {"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1,
                  "effect": "command", "whose": "enemy"}
KING_COMMAND = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
                "effect": "command", "whose": "enemy", "captures_own": True}
SHOWCASES = {
    # name: (hash, the piece carrying the key, the key, Q1007c's test spec)
    "chess-with-commanding-knights": ("adc1a91b0a83", "N", KNIGHT_COMMAND, "_command_k1"),
    "chess-with-a-spellbinding-king": ("6a8120390853", "K", KING_COMMAND, "_command_k2"),
}


def chess_with(pid: str, sight: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(CHESS.read_text(encoding="utf-8"))
    spec = copy.deepcopy(spec)
    for piece in spec["pieces"]:
        if piece["id"] == pid:
            piece["sight"] = [copy.deepcopy(sight)]
    return spec


@pytest.mark.parametrize("name", sorted(SHOWCASES))
def test_each_showcase_is_chess_with_its_registered_key(name: str) -> None:
    digest, pid, key, test_spec = SHOWCASES[name]
    spec = retired_preview(name)
    assert RETIRED_PREVIEWS[name] == digest
    assert spec_hash(spec)[:12] == digest
    assert spec_hash(chess_with(pid, key))[:12] == digest
    assert spec_hash(load_spec(HERE / f"{test_spec}.json"))[:12] == digest
    assert spec_hash(test_command.k1() if pid == "N" else test_command.k2())[:12] == digest


def test_the_hand_built_positions_of_v1_are_held() -> None:
    """§6.3's hand-built positions, each a test of `tests/test_command.py` on these mechanics."""
    for held in (
        "test_a_check_is_answered_by_commanding_the_checker_away",
        "test_a_commanded_move_can_discover_check",
        "test_a_commanded_move_that_would_expose_the_commanders_royal_is_not_offered",
        "test_a_frozen_piece_is_not_commanded",
        "test_no_promotion_and_no_window_by_command",
        "test_a_commanded_move_never_castles",
        "test_a_commanded_rook_loses_its_castling_right",
        "test_a_command_without_affects_skips_the_royal_types",
    ):
        assert callable(getattr(test_command, held)), held


def _retired_name(name: str) -> str:
    return f"_retired_{name.replace('-', '_')}"


@pytest.mark.parametrize("name", sorted(SHOWCASES))
def test_the_core_agrees_with_the_oracle_over_three_games(name: str) -> None:
    compared, commanded = walk(_retired_name(name), games=3, plies=100)
    assert compared > 30
    assert commanded > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SHOWCASES))
def test_the_core_agrees_with_the_oracle_over_twenty_games(name: str) -> None:
    compared, commanded = walk(_retired_name(name), games=20)
    assert compared > 200
    assert commanded > 20
