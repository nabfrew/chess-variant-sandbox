"""Q999g: the still board draws the effects a line of sight puts on its position.

`docs/archive/plan-passive.md` Q999c's *as built* note left the still boards (the game page's start
position, `page/stillboard.js`) without the frozen / disarmed / shielded badges the play
board draws. The core answers which pieces are under an effect (`Engine.effects`); the
preview payload carries that answer as ``effects: [[square, bits], ...]`` (`card_rules.preview`),
and the still board draws the play board's own `effectBadges` from it, working nothing out.
Chess with a sheltering king has a shield on fourteen pieces at ply zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from page_text import STYLED
from test_web import _in_page

from chess_variant_sandbox.web import designing

ROOT = Path(__file__).resolve().parents[1]
SHIELD = 4


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))


def _drawn(preview: dict[str, Any], expression: str) -> Any:
    return _in_page(f"const preview = {json.dumps(preview)};", expression)


def _marks(preview: dict[str, Any], kind: str, wrapper: str = "preview") -> Any:
    return _drawn(preview, f"(boardPreview({wrapper}, null).match(/data-mark=\"{kind}\"/g) || []).length")


class TestThePayload:
    def test_the_core_answer_for_the_start_position_is_sent(self) -> None:
        preview = designing.opening_preview(_load("chess-with-a-sheltering-king"))
        effects = preview["effects"]
        assert len(effects) == 14 and all(bits == SHIELD for _, bits in effects)
        assert all(preview["board"][square] for square, _ in effects), "each on a piece"

    def test_a_game_with_no_sight_sends_none(self) -> None:
        spec = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        assert "effects" not in designing.opening_preview(spec)

    def test_a_sight_that_binds_on_nothing_at_the_start_sends_none(self) -> None:
        """Madrasi's twins start with their lines blocked: no piece stands frozen."""
        assert "effects" not in designing.opening_preview(_load("madrasi"))


class TestTheStillBoard:
    def test_each_affected_piece_wears_its_badge(self) -> None:
        preview = designing.opening_preview(_load("chess-with-a-sheltering-king"))
        assert _marks(preview, "shielded") == 14
        assert _marks(preview, "frozen") == 0 and _marks(preview, "disarmed") == 0

    def test_the_badge_is_the_play_boards_for_that_effect(self) -> None:
        preview = designing.opening_preview(_load("chess-with-a-sheltering-king"))
        # One piece frozen and disarmed wears the snowflake alone, a disarmed one the blade.
        preview["effects"] = [[0, 1 | 2 | 4], [1, 2], [6, 4]]
        owner = preview["board"][0]
        owners = _drawn(preview, "([...boardPreview(preview, null).matchAll("
                        "/data-mark=\"(frozen|disarmed|shielded)\"( data-player=\"(\\d)\")?/g)].map(m => [m[1], +m[3]]))")
        stride = preview.get("sideStride") or preview["nTypes"]
        assert owners == [["frozen", (owner - 1) // stride], ["shielded", (owner - 1) // stride],
                          ["disarmed", (preview["board"][1] - 1) // stride],
                          ["shielded", (preview["board"][6] - 1) // stride]]

    def test_a_thumbnail_and_a_lone_piece_draw_none(self) -> None:
        preview = designing.opening_preview(_load("chess-with-a-sheltering-king"))
        assert _marks(preview, "shielded", "Object.assign({}, preview, { plain: true })") == 0
        assert _marks(preview, "shielded", "loneBoard(preview, 4, preview.board[4])") == 0

    def test_a_square_with_no_piece_draws_none(self) -> None:
        preview = designing.opening_preview(_load("chess-with-a-sheltering-king"))
        empty = next(square for square, held in enumerate(preview["board"]) if not held)
        preview["effects"] = [[empty, SHIELD]]
        assert _marks(preview, "shielded") == 0

    def test_a_game_without_a_sight_is_drawn_as_it_was(self) -> None:
        spec = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        assert _drawn(designing.opening_preview(spec), "boardPreview(preview, null).includes('data-mark=\"shielded\"')") is False

    def test_the_game_pages_legend_names_the_badges(self) -> None:
        preview = designing.opening_preview(_load("chess-with-a-sheltering-king"))
        said = _drawn({"opening": preview, "hash": "x", "sides": ["White", "Black"]},
                      STYLED + "(openingHtml(preview))")
        assert "data-legend=\"shielded\"" in said and "data-legend=\"frozen\"" not in said
