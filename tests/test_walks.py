"""Q852: every wild piece's walk drawn in order (`viz/walks.py`, `page/stillboard.js`).

`docs/reference/designer-case-study-wolves.md` §3 fix 3. The walk's geometry is a pure
function: its squares in the order a piece on it steps, the squares pieces start on (several
pieces on one walk are one walk, schema 23's ``also_at``), and which steps jump between two
squares that do not touch, named in words. It reads the resolver the compiler reads, so a
walk is checked here against the compiled ``next`` table of every committed game with wild
pieces. The page's half lays those squares out on a still board, which runs in node
(`test_web._in_page`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_web import _in_page

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.schema import format_square
from chess_variant_sandbox.viz.walks import leap_words, step_apart, walk_payload, walks
from chess_variant_sandbox.web import designing

ROOT = Path(__file__).resolve().parents[1]
CIRCUIT = [f"{f}4" for f in "abcdefgh"] + [f"{f}5" for f in "hgfedcba"]
# The wolves walk's step 9: *a whole row* pressed twice, a wolf that jumps h4 to a5, h5 to a4.
TELEPORT = [f"{f}4" for f in "abcdefgh"] + [f"{f}5" for f in "abcdefgh"]


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + (ord(name[0]) - ord("a"))


def _names(squares: list[int], width: int = 8) -> list[str]:
    return [format_square(s % width, s // width) for s in squares]


def _with_npcs(npcs: list[dict[str, Any]]) -> dict[str, Any]:
    spec = copy.deepcopy(_load("wolves-patrol"))
    spec["rules"]["npcs"] = npcs
    return spec


def _wild_rulesets() -> list[Path]:
    return sorted(
        path for folder in ("rulesets", "archive") for path in (ROOT / folder).glob("*.json")
        if '"npcs"' in path.read_text(encoding="utf-8")
    )


class TestTheWalk:
    def test_a_walk_is_in_the_order_its_pieces_step(self) -> None:
        """Wolves: two five-square patrols, each mirrored, four walks, each in route order
        and named as the designer's card names it."""
        spec = _load("wolves")
        found = walk_payload(spec, Engine(spec).spec)
        assert [walk["name"] for walk in found] == [
            "wild pieces 1", "wild pieces 1's mirror", "wild pieces 2", "wild pieces 2's mirror"]
        assert _names(found[0]["squares"]) == ["a4", "b5", "c4", "b3", "a3"]
        # The mirror walks its reflected squares in the same order.
        assert _names(found[1]["squares"]) == ["a5", "b4", "c5", "b6", "a6"]
        assert [_names(walk["starts"]) for walk in found] == [["a4"], ["a5"], ["h4"], ["h5"]]
        assert all(walk["piece"] == "W" and walk["leaps"] == [] and walk["says"] is None
                   for walk in found)

    def test_several_pieces_on_one_walk_are_one_walk(self) -> None:
        """Wolves patrol (schema 23): one circuit, a wolf starting on a4 and one on h5."""
        spec = _load("wolves-patrol")
        compiled = Engine(spec).spec
        found = walk_payload(spec, compiled)
        assert len(found) == 1
        assert _names(found[0]["squares"]) == CIRCUIT
        assert _names(found[0]["starts"]) == ["a4", "h5"]
        assert found[0]["type"] == list(compiled.piece_ids).index("W")
        assert found[0]["leaps"] == []

    def test_a_walk_that_jumps_says_where(self) -> None:
        """The wolves walk's step 9: the validator accepts it without a word, so the board
        draws the two jumps and the line under it names them."""
        spec = _with_npcs([{"piece": "W", "route": TELEPORT}])
        (walk,) = walk_payload(spec, Engine(spec).spec)
        assert [(leap["from"], leap["to"]) for leap in walk["leaps"]] == [
            (_square("h4"), _square("a5")), (_square("h5"), _square("a4"))]
        assert walk["leaps"][0]["says"] == "from h4 to a5, seven files apart"
        assert walk["says"] == ("Wild pieces 1's walk steps from h4 to a5, seven files apart, "
                                "and from h5 to a4, seven files apart.")

    @pytest.mark.parametrize("path", _wild_rulesets(), ids=lambda path: path.stem)
    def test_every_walk_is_the_compiled_table(self, path: Path) -> None:
        """What a walk means here is what the core steps: each square's successor is the
        compiled ``next`` table's, and the starts are the compiled starts, in order."""
        spec = json.loads(path.read_text(encoding="utf-8"))
        compiled = Engine(spec).spec
        npcs = compiled.npcs
        found = walk_payload(spec, compiled)
        assert npcs is not None and found
        for walk in found:
            squares = walk["squares"]
            for at, square in enumerate(squares):
                assert npcs.next[square] == squares[(at + 1) % len(squares)], path.stem
                assert npcs.piece[square] == walk["type"], path.stem
        starts = [square for walk in found for square in walk["starts"]]
        assert sorted(starts) == sorted(square for square, _ in npcs.starts)

    def test_a_game_without_wild_pieces_has_no_walks(self) -> None:
        spec = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
        assert walk_payload(spec, Engine(spec).spec) == []
        assert "walks" not in designing.opening_preview(spec)


class TestTouching:
    """Which steps jump: the pure geometry, on each lattice and across each kind of seam."""

    def test_a_king_step_touches_and_anything_further_does_not(self) -> None:
        a4 = _square("a4")
        for near in ("a5", "b5", "b4", "b3", "a3"):
            assert step_apart(a4, _square(near), 8, 8)[2] == 1, near
        assert step_apart(a4, _square("c5"), 8, 8) == (2, 1, 2)
        assert step_apart(_square("h4"), _square("a5"), 8, 8) == (7, 1, 7)

    def test_a_seam_joins_its_edges_whoever_may_cross(self) -> None:
        h4, a4, a5 = _square("h4"), _square("a4"), _square("a5")
        assert step_apart(h4, a4, 8, 8, files_join=True) == (1, 0, 1)
        assert step_apart(h4, a5, 8, 8, files_join=True) == (1, 1, 1)
        assert step_apart(h4, a4, 8, 8, ranks_join=True)[2] == 7, "the other axis's seam"
        assert step_apart(_square("a1"), _square("a8"), 8, 8, ranks_join=True)[2] == 1

    def test_a_hex_cell_touches_its_six_neighbours(self) -> None:
        width, centre = 11, 5 * 11 + 5
        for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)):
            assert step_apart(centre, centre + dr * width + df, width, width, lattice="hex")[2] == 1
        # The square lattice's diagonal is two cells on a hexagon.
        assert step_apart(centre, centre + width + 1, width, width, lattice="hex")[2] == 2

    def test_a_hexagon_copy_joins_the_sides_it_names(self) -> None:
        """A copy's vector (a cell of the copy plus the vector is the board's same cell)."""
        width = 11
        a, b = 0 * width + 10, 0 * width + 0
        assert step_apart(a, b, width, width, lattice="hex")[2] == 10
        assert step_apart(a, b, width, width, lattice="hex", copies=[(11, 0)])[2] == 1

    def test_the_words_name_what_is_too_far(self) -> None:
        h4, a5, c6 = _square("h4"), _square("a5"), _square("c6")
        assert leap_words(h4, a5, 8, (7, 1, 7), "square") == "from h4 to a5, seven files apart"
        assert leap_words(h4, c6, 8, (5, 2, 5), "square") == (
            "from h4 to c6, five files and two ranks apart")
        assert leap_words(h4, c6, 8, (5, 2, 5), "hex") == "from h4 to c6, five cells apart"

    def test_a_walk_names_three_jumps_and_counts_the_rest(self) -> None:
        route = ["a1", "c1", "e1", "g1", "g3", "e3"]
        (walk,) = walks({"npcs": [{"piece": "W", "route": route}]}, 8, 8, piece_ids=["W"])
        assert len(walk["leaps"]) == 6
        assert walk["says"].endswith("and three more jumps.")


class TestNotAGameYet:
    def test_overlapping_walks_are_drawn_on_the_stand_in(self) -> None:
        """Two wolves typed on one circuit (step 13) are refused, and the board still draws
        both walks, each with the square its wolf would start on, so the refusal can be seen."""
        second = CIRCUIT[8:] + CIRCUIT[:8]
        spec = _with_npcs([{"piece": "W", "route": CIRCUIT}, {"piece": "W", "route": second}])
        answer = designing.check(spec)
        assert answer["valid"] is False
        preview = answer["preview"]
        assert preview["standIn"] is True
        assert [_names(walk["squares"]) for walk in preview["walks"]] == [CIRCUIT, second]
        assert [_names(walk["starts"]) for walk in preview["walks"]] == [["a4"], ["h5"]]

    def test_a_valid_draft_sends_its_walks(self) -> None:
        answer = designing.check(_load("wolves-patrol"))
        assert answer["valid"] is True
        assert len(answer["preview"]["walks"]) == 1


# ------------------------------------------------------------------ the still board

def _drawn(preview: dict[str, Any], expression: str) -> Any:
    return _in_page(f"const preview = {json.dumps(preview)};", expression)


_STEPS = ("[...boardPreview(preview, null).matchAll(/data-mark=\"walk\" data-walk=\"(\\d+)\""
          " data-step=\"(\\d+)>(\\d+)\"( data-leap=\"1\")?/g)]"
          ".map(m => [+m[1], +m[2], +m[3], !!m[4]])")


class TestTheStillBoard:
    def test_each_step_is_drawn_in_order_and_a_jump_dashed(self) -> None:
        spec = _with_npcs([{"piece": "W", "route": TELEPORT}])
        preview = designing.opening_preview(spec)
        steps = _drawn(preview, _STEPS)
        squares = [_square(name) for name in TELEPORT]
        assert [(a, b) for _, a, b, _ in steps] == [
            (a, squares[(at + 1) % 16]) for at, a in enumerate(squares)]
        assert [(a, b) for _, a, b, leap in steps if leap] == [
            (_square("h4"), _square("a5")), (_square("h5"), _square("a4"))]
        dashed = _drawn(preview, "(boardPreview(preview, null).match(/stroke-dasharray/g) || []).length")
        assert dashed == 2 * 2, "two jumps, each a shaft on its casing"
        said = _drawn(preview, "walkLeapsHtml(preview)")
        assert "steps from h4 to a5, seven files apart" in said and "var(--walk-1)" in said

    def test_two_wolves_on_one_walk_are_one_path_with_two_rings(self) -> None:
        preview = designing.opening_preview(_load("wolves-patrol"))
        seen = _drawn(preview, "(() => { const html = boardPreview(preview, null); return ["
                      "new Set([...html.matchAll(/data-walk=\"(\\d+)\"/g)].map(m => m[1])).size,"
                      "(html.match(/data-walkstart=/g) || []).length,"
                      "(html.match(/data-walkghost=/g) || []).length]; })()")
        assert seen == [1, 2, 0]
        assert _drawn(preview, "walksDrawn(preview)") is True

    def test_a_board_with_no_wild_piece_on_it_draws_them_faintly(self) -> None:
        """A draft that is not a game yet may be drawn on a stand-in without its wild pieces:
        the walk's starts then carry a faint wild piece."""
        preview = designing.opening_preview(_load("wolves-patrol"))
        preview["board"] = [0] * len(preview["board"])
        ghosts = _drawn(preview, "(boardPreview(preview, null).match(/data-walkghost=/g) || []).length")
        assert ghosts == 2

    def test_a_thumbnail_and_a_lone_piece_draw_no_walk(self) -> None:
        preview = designing.opening_preview(_load("wolves"))
        assert _drawn(preview, "boardPreview(Object.assign({}, preview, { plain: true }), null)"
                               ".includes('data-walk=')") is False
        assert _drawn(preview, f"boardPreview(loneBoard(preview, {_square('d4')}, 1), null)"
                               ".includes('data-walk=')") is False
        assert _drawn(preview, "boardPreview(preview, null).includes('data-walk=')") is True

    def test_the_designer_names_each_hue_as_its_refusals_do(self) -> None:
        """Two walks typed on one circuit are refused as *Wild pieces 1 and 2*; the line
        under the designer's board says which hue is which, and the game page's does not."""
        second = CIRCUIT[8:] + CIRCUIT[:8]
        preview = designing.check(_with_npcs([{"piece": "W", "route": CIRCUIT},
                                              {"piece": "W", "route": second}]))["preview"]
        named = _drawn(preview, "walkLeapsHtml(preview, true)")
        assert "data-walknames" in named
        assert named.index("var(--walk-1)") < named.index("wild pieces 1") < named.index(
            "var(--walk-2)") < named.index("wild pieces 2")
        assert _drawn(preview, "walkLeapsHtml(preview)") == ""

    def test_the_legend_names_the_walk_from_its_row(self) -> None:
        said = _in_page("", "legendEntries(new Set(['walk']), null, '', null).join('')")
        assert 'data-legend="walk"' in said and "coloured arrows" in said
        assert "var(--walk-1)" in said
