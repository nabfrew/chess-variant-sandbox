"""Q809: the designer's bench board names its squares, as the play board does.

*Teach a move* asks a person to click the squares a piece should reach and its sentences
name squares, but its board (`boardPreview`, the still board) drew no file and rank names.
The bench's three boards (*set up*, *teach a move*, *where it goes*) now set them outside the
board where its drawing ends at its own edge, through the same `edgeNamesHtml` the play
board's `drawCoords` uses, and on the edge squares where a seam's arrows stand round it. A
library thumbnail and the game page's start position name none. These run the page's own
script in node (`test_wood._run`) and read the markup.
"""

from __future__ import annotations

import json
import re

from test_wood import _preview, _run

from chess_variant_sandbox.web.page import style


def _names(drawn: str) -> list[str]:
    return re.findall(r'<span class="co"[^>]*>([^<]*)</span>', drawn)


class TestTheBenchNamesItsSquares:
    def test_the_bench_board_names_every_file_and_rank_outside_it(self) -> None:
        drawn = _run(f"boardPreview({json.dumps(_preview(8, 8))}, null, null, {{}})")
        assert 'class="still-named"' in drawn and 'class="board-box"' in drawn
        names = _names(drawn)
        assert sorted(names) == sorted([*"abcdefgh", *map(str, range(1, 9))])

    def test_a_tall_board_names_its_own_files_and_ranks(self) -> None:
        drawn = _run(f"boardPreview({json.dumps(_preview(10, 11))}, null, null, {{}})")
        assert sorted(_names(drawn)) == sorted([*"abcdefghij", *map(str, range(1, 12))])

    def test_a_board_with_a_seam_names_them_on_its_edge_squares(self) -> None:
        """The seam's arrowheads stand outside the board, as a wrapping play board's ghosts
        do, so the names go on the squares there (`coordsHtml`), as the play board puts them."""
        seam = _preview(8, 8, wrap={"files": [[True, False], [True, False]], "ranks": []})
        drawn = _run(f"boardPreview({json.dumps(seam)}, null, null, {{}})")
        assert not _names(drawn) and 'class="still-named"' not in drawn
        inside = re.findall(r'<text class="co"[^>]*>([^<]*)</text>', drawn)
        assert sorted(inside) == sorted([*"abcdefgh", *map(str, range(1, 9))])

    def test_a_thumbnail_and_a_start_position_name_none(self) -> None:
        plain = _run(f"[boardPreview({json.dumps(_preview(8, 8))}, null),"
                     f" boardPreview({json.dumps(_preview(8, 8, plain=True))}, null, null, {{}})]")
        for drawn in plain:
            assert 'class="co"' not in drawn and 'class="still-named"' not in drawn

    def test_the_play_board_and_the_bench_share_one_drawing_of_the_names(self) -> None:
        from page_text import PAGE

        assert PAGE.count("function edgeNamesHtml(") == 1
        assert "edgeNamesHtml(cell, FRAME, viewBox(FRAME, cell))" in PAGE
        assert PAGE.count("edgeLabels(cell, view).map(") == 1

    def test_the_bench_makes_room_for_them(self) -> None:
        css = style.STYLE
        start = css.index(".still-named {")
        rule = css[start:css.index("}", start)]
        assert "padding-left" in rule and "padding-bottom" in rule and "--co-out" in rule
        assert ".still-named .board-frame { overflow: visible; }" in css
