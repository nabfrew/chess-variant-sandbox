"""The board's wood as drawn (Q588, plan-visual.md §9's third answer).

Every square is its own piece of wood: a window of its tone's grain at an offset hashed from
the board's identity and the square's index (`web/static/page/wood.js`). These run the
page's own script in node, with a stand-in stylesheet that answers the real tokens, and read
the markup the still board draws. The tokens themselves, the subtlety and the contrast floors
over the worst streak are `tests/test_style.py`'s.
"""

from __future__ import annotations

import json
import re
from typing import Any

import node_session

from chess_variant_sandbox.web.page import tokens

# A document with nothing in it, and a stylesheet that answers the tokens the page reads.
_STUB = """
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                             set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el, documentElement: el };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  global.fetch = () => new Promise(() => {});
  const TOKENS = %s;
  global.getComputedStyle = () => ({ getPropertyValue: name => TOKENS[name] || '' });
"""


def _run(expression: str, styled: bool = True) -> Any:
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    answered = {f"--{name}": tokens.LIGHT[name] for name in ("grain-lt", "grain-dk")}
    stub = _STUB % json.dumps(answered if styled else {})
    body = stub + script.group(1) + "\nconsole.log(JSON.stringify(" + expression + "));"
    done = node_session.run(body)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _preview(width: int, height: int, **more: Any) -> dict[str, Any]:
    return {"width": width, "height": height, "board": [0] * (width * height), "nTypes": 1,
            "pieceIds": ["x"], "icons": [], **more}


class TestTheOffsets:
    def test_the_same_board_is_grained_the_same_every_time(self) -> None:
        """Never a live random draw: a pure function of the identity and the square."""
        once, again = _run("[[...Array(64).keys()].map(s => woodOffset(woodSeed('4c1922ffae26'), s)),"
                           " [...Array(64).keys()].map(s => woodOffset(woodSeed('4c1922ffae26'), s))]")
        assert once == again
        assert all(0 <= v < 1 for pair in once for v in pair)

    def test_every_square_is_its_own_piece_of_wood(self) -> None:
        """No two squares of a board share an offset, and no two squares that touch at a
        corner (a square board's same-tone neighbours) are cut within a hundredth of a tile
        of each other on both axes."""
        offsets = _run("[...Array(64).keys()].map(s => woodOffset(woodSeed('4c1922ffae26'), s))")
        assert len({tuple(o) for o in offsets}) == 64
        for square in range(64):
            file, rank = square % 8, square // 8
            for df in (-1, 1):
                if 0 <= file + df < 8 and rank + 1 < 8:
                    other = offsets[(rank + 1) * 8 + file + df]
                    apart = max(abs(offsets[square][0] - other[0]), abs(offsets[square][1] - other[1]))
                    assert apart > 0.01, (square, apart)

    def test_another_board_is_other_wood(self) -> None:
        one, two = _run("['4c1922ffae26', '975dde173f01'].map(id =>"
                        " [...Array(25).keys()].map(s => woodOffset(woodSeed(id), s)))")
        assert all(a != b for a, b in zip(one, two, strict=True))

    def test_the_frame_starts_where_the_board_says(self) -> None:
        one, again, other = _run("[woodFrameStyle('4c1922ffae26'), woodFrameStyle('4c1922ffae26'),"
                                 " woodFrameStyle('975dde173f01')]")
        assert one == again != other
        assert re.fullmatch(r"--wood-x: \d+px; --wood-y: \d+px", one)


class TestTheStillBoard:
    def test_every_square_carries_its_grain_under_everything_else(self) -> None:
        drawn = _run(f"boardPreview({json.dumps(_preview(5, 5))}, null)")
        assert drawn.count('class="grain lt"') + drawn.count('class="grain dk"') == 25
        assert drawn.count('class="grain lt"') == 12 and drawn.count('class="grain dk"') == 13
        # Each square's own pattern, cut from one of the two base tiles at its offset.
        assert len(re.findall(r'<pattern id="still\d+-wood\d+" href="#still\d+-wood-(?:lt|dk)"', drawn)) == 25
        assert drawn.count('patternUnits="userSpaceOnUse"') == 2
        # The square's colour first, then its grain, before any mark or piece.
        first = re.search(r'<g data-square="\d+"><rect [^>]*fill="var\(--(?:light|dark)\)"/><rect [^>]*'
                          r'class="grain', drawn)
        assert first
        assert re.search(r'class="board-frame" style="--wood-x: \d+px; --wood-y: \d+px"', drawn)

    def test_the_same_still_draws_the_same_wood(self) -> None:
        one, two = _run(f"[boardPreview({json.dumps(_preview(6, 6))}, null),"
                        f" boardPreview({json.dumps(_preview(6, 6))}, null)]")
        assert re.sub(r"still\d+", "still", one) == re.sub(r"still\d+", "still", two)

    def test_two_games_of_one_shape_are_two_boards_of_wood(self) -> None:
        """A card's thumbnail passes its game's hash, so two 8 x 8 games side by side are
        not cut alike; a draft, with no hash, keeps its wood while it is edited."""
        plain = _preview(8, 8)
        one, two, bare = _run(
            f"[boardPreview({json.dumps({**plain, 'hash': '4c1922ffae26'})}, null),"
            f" boardPreview({json.dumps({**plain, 'hash': '975dde173f01'})}, null),"
            f" boardPreview({json.dumps(plain)}, null)]")

        def cut(drawn: str) -> list[str]:
            return re.findall(r'patternTransform="([^"]+)"', drawn)

        assert cut(one) != cut(two) and cut(one) != cut(bare)

    def test_a_hexagon_grains_every_cell_it_draws(self) -> None:
        """A hex board's cells are hexagons and its corners are not cells; each cell that is
        drawn is grained in its own shape, and the third, in-between shade takes the dark
        squares' wood."""
        absent = [0, 1, 5]
        preview = _preview(3, 3, lattice="hex", absent=absent)
        drawn = _run(f"boardPreview({json.dumps(preview)}, null)")
        grains = re.findall(r'<path [^>]*class="grain (lt|dk)"', drawn)
        assert len(grains) == 9 - len(absent)

    def test_without_the_stylesheet_the_board_is_drawn_plain(self) -> None:
        drawn = _run(f"boardPreview({json.dumps(_preview(4, 4))}, null)", styled=False)
        assert "grain" not in drawn and "<rect" in drawn


class TestThePlayBoard:
    def test_the_play_board_grains_its_squares_by_the_game(self) -> None:
        from page_text import BOARD, PAGE

        assert PAGE.count("function woodGrain(") == 1
        assert "const wood = woodGrain('board', identity, cell);" in BOARD
        assert "+ wood.cell(x, y, FRAME, square, shade);" in BOARD
        assert "markDefs(SQUARE_MARKS, 'board') + wood.defs() + out" in BOARD
        assert "const identity = rules.hash ||" in BOARD


_SHEET = re.compile(r"--sheet-x: ([\d.]+)px; --sheet-y: ([\d.]+)px; --sheet-s: ([\d.]+); "
                    r"--sheet-a: (-?[\d.]+)deg; --sheet-t: (-?[\d.]+)")


class TestTheSheets:
    """A library card's own sheet of paper (Q684): `paperSheet` hashes the game into an
    offset, a size, a turn and a tone, within `tokens.SHEET`'s ranges. What the stylesheet
    does with them, and the floors at their ends, are `tests/test_style.py`'s."""

    HASHES = [f"{n:012x}" for n in range(0, 0xffffffffffff, 0x0a3d70a3d70a)][:100]

    def test_a_game_is_the_same_sheet_every_time(self) -> None:
        once, again = _run(f"[{json.dumps(self.HASHES)}.map(paperSheet),"
                           f" {json.dumps(self.HASHES)}.map(paperSheet)]")
        assert once == again

    def test_every_value_is_in_its_stated_range(self) -> None:
        sheets = _run(f"{json.dumps(self.HASHES)}.map(paperSheet)")
        limit = tokens.SHEET
        for sheet in sheets:
            assert len(sheet.encode()) < 200, sheet
            found = _SHEET.fullmatch(sheet)
            assert found, sheet
            x, y, scale, angle, tone = (float(v) for v in found.groups())
            assert abs(scale - 1) <= limit["scale"] + 1e-9
            assert 0 <= x < 180 * scale and 0 <= y < 180 * scale
            assert abs(angle) <= limit["angle"] and abs(tone) <= limit["tone"]

    def test_different_games_are_different_sheets(self) -> None:
        """No two of a hundred games share a sheet, and the hundred spread over each range
        rather than bunching at one value."""
        sheets = _run(f"{json.dumps(self.HASHES)}.map(paperSheet)")
        assert len(set(sheets)) == len(sheets)
        values = list(zip(*(map(float, _SHEET.fullmatch(s).groups()) for s in sheets), strict=True))  # type: ignore[union-attr]
        # Each range's outer sixth reached at both ends: x and y in px of a 180 px tile.
        reached = ((30, 150), (30, 150), (0.96, 1.04), (-2, 2), (-0.67, 0.67))
        for column, (low, high) in zip(values, reached, strict=True):
            assert min(column) < low and max(column) > high, column

    def test_a_card_is_not_its_board(self) -> None:
        """The sheet is hashed apart from the wood: the card's paper is not cut where its
        thumbnail's frame is."""
        sheet, frame = _run("[paperSheet('4c1922ffae26'), woodFrameStyle('4c1922ffae26')]")
        assert _SHEET.fullmatch(sheet) and sheet != frame

    def test_every_library_card_carries_its_sheet(self) -> None:
        from page_text import view

        assert ('<div class="panel card" data-hash="${esc(game.hash)}" style="${'
                'paperSheet(game.hash)}">') in view("browse")
