"""Each game's own link-preview picture (G6 of `docs/plan-growth.md`, §2.2).

The drawing (`web/page/game_picture.py`) and the pictures a site keeps
(`site/game_pictures.py`), against `rulesets/dull.json` and payloads shaped like the card's
still board, never chess. What a running site's pages and route say is in
`test_site_doors.py::TestTheGamePictures`.

- **The same game draws the same bytes**, twice in one process: the drawing is text, the
  faces are files passed in with the machine's own skipped, and resvg is deterministic.
- **A picture's name is the digest of what it draws**, so a renamed game is a new picture
  and an unchanged one keeps its address.
- **A square is where the page puts it**: `cell_place` restates `wrapped.js`'s `cellXY` for
  the flat unflipped view, and the test fails when the JS moves.
- **Every piece on the still board is drawn**, a drawing where it has one and a lettered
  disc where it has none.
- **The site keeps a picture once made**: asked for on a page read, made on a thread of its
  own, named only once on disk, and an older picture of the same game removed.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.site import game_pictures
from chess_variant_sandbox.viz.wrapped import WRAPPED
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.page import game as drawing
from chess_variant_sandbox.web.page import game_picture, picture

DULL = Path(__file__).resolve().parents[1] / "rulesets" / "dull.json"


def _game(name: str = "A dull test game", spec: dict[str, Any] | None = None) -> library.Game:
    found = spec if spec is not None else json.loads(DULL.read_text(encoding="utf-8"))
    return library.Game(hash="0" * 16, name=name, description="", path=DULL,
                        source="rulesets", spec=found)


def _shape(game: library.Game) -> dict[str, Any]:
    shape = library.opening(game)
    assert shape is not None
    return shape


def _drawn(name: str = "A dull test game") -> str:
    game = _game(name)
    return game_picture.svg(game.name, _shape(game), drawing.board_words(game.spec))


needs_resvg = pytest.mark.skipif(
    not game_picture.available(), reason="resvg-py is the site extra's: uv sync --extra site")


class TestTheDrawing:
    @needs_resvg
    def test_the_same_game_draws_the_same_bytes(self) -> None:
        once, twice = game_picture.png(_drawn()), game_picture.png(_drawn())
        assert once == twice
        assert picture.size(once) == (game_picture.WIDTH, game_picture.HEIGHT) == (1200, 630)
        assert len(once) < picture.BYTES_MOST

    @needs_resvg
    def test_the_words_are_set_in_the_sites_faces(self) -> None:
        """The faces are passed in by file and the machine's skipped: with them the name is
        ink on the paper; a family resvg could not find would leave the paper bare."""
        bare = game_picture.png(_drawn().replace(game_picture.HEADING_FAMILY, "'Nothing'"))
        assert game_picture.png(_drawn()) != bare

    def test_a_pictures_name_is_the_digest_of_what_it_draws(self) -> None:
        key = game_picture.key(_drawn())
        assert re.fullmatch(r"[0-9a-f]{16}", key)
        assert game_picture.key(_drawn()) == key
        assert game_picture.key(_drawn("Another name")) != key
        digest = "ab" * 8
        found = game_picture.ROUTE.match(game_picture.path(digest, key))
        assert found is not None and found.groups() == (digest, key)
        assert game_picture.ROUTE.match(f"/picture/game/{digest}-{key}.png/..") is None

    def test_every_piece_on_the_still_board_is_drawn(self) -> None:
        game = _game()
        shape = _shape(game)
        pieces = sum(1 for value in shape["board"] if value)
        assert pieces > 0
        drawn = game_picture.svg(game.name, shape)
        assert drawn.count("<use ") == pieces
        # One drawing per stem, set once and used by every piece that wears it.
        assert drawn.count("<image ") == len({icon["stem"] for pair in shape["icons"]
                                              for icon in pair if icon})
        # Without drawings, each piece is a disc with its letter, as the page draws it.
        lettered = {**shape, "icons": [[None, None] for _ in shape["icons"]]}
        plain = game_picture.svg(game.name, lettered)
        assert plain.count("<use ") == 0
        assert plain.count("<circle ") == pieces
        assert plain.count(f">{shape['pieceIds'][0]}</text>") == pieces

    def test_a_wild_piece_wears_its_drawing_on_the_third_disc(self) -> None:
        """PI3, as `stillPieceHtml` draws it (`test_web.TestAWildPieceWearsItsDrawing`): a
        wild piece (owner 2) wears its type's first seat's drawing on the third owner's
        disc; the duck bare; one with no drawing the lettered disc."""
        game = _game()
        shape = _shape(game)
        types = int(shape["nTypes"])
        stride = int(shape.get("sideStride") or types)
        wild = 2 * stride + 1  # type 0, owner 2
        drawn = game_picture._piece(shape, wild, 0, 0, 50, {})
        fill = game_picture.LIGHT["p2"]
        assert "<use " in drawn and f'fill="{fill}"' in drawn and "<text" not in drawn
        duck = game_picture._piece({**shape, "duck": 0}, wild, 0, 0, 50, {})
        assert "<use " in duck and "<circle" not in duck
        bare = {**shape, "icons": [[None, None] for _ in shape["icons"]]}
        lettered = game_picture._piece(bare, wild, 0, 0, 50, {})
        assert "<use " not in lettered and f">{shape['pieceIds'][0]}</text>" in lettered

    def test_the_name_is_set_to_fit(self) -> None:
        room = 470.0
        size, lines = game_picture.name_lines("Short", room)
        assert (size, lines) == (game_picture.NAME_SIZES[0], ["Short"])
        size, lines = game_picture.name_lines("Glinski's hexagonal chess on a torus", room)
        assert size < game_picture.NAME_SIZES[0] and 1 < len(lines) <= game_picture.NAME_LINES
        assert not lines[-1].endswith("…")
        size, lines = game_picture.name_lines(
            "A rather long name for a game that goes on and on and on and on " * 3, room)
        assert size == game_picture.NAME_SIZES[-1] and len(lines) == game_picture.NAME_LINES
        assert lines[-1].endswith("…")
        for line in lines:
            assert game_picture.text_width(line, game_picture.HEADING_FACE, size) <= room
        size, lines = game_picture.name_lines("Unbreakablewordthatisfartoowide" * 2, room)
        assert all(game_picture.text_width(line, game_picture.HEADING_FACE, size) <= room
                   for line in lines)

    def test_the_faces_are_measured_from_their_own_advances(self) -> None:
        for face in (game_picture.HEADING_FACE, game_picture.TEXT_FACE):
            assert face[1].is_file()
            wide = game_picture.text_width("W", face, 100)
            narrow = game_picture.text_width("i", face, 100)
            assert 0 < narrow < wide < 120
            assert game_picture.text_width("Wi", face, 100) == pytest.approx(wide + narrow)


class TestWhereASquareIsDrawn:
    def test_the_page_places_a_square_as_this_restates(self) -> None:
        """`cell_place` is `cellXY(squareCell(square))` on a flat unflipped view: a change to
        the page's arithmetic fails here, naming the copy to change with it."""
        body = WRAPPED[WRAPPED.index("function cellXY("):]
        body = body[: body.index("\n}\n")]
        assert ": [u * cell, (view.H - 1 - v) * cell];" in body
        assert "return [xy[0] + above * cell / 2, xy[1] * HEX_ROW];" in body
        assert "const above = ((view.H - 1) * cell - xy[1]) / cell;" in body
        assert "const HEX_ROW = Math.sqrt(3) / 2;" in WRAPPED

    def test_squares_and_hexagons(self) -> None:
        # Square lattice: the first player's first rank along the bottom.
        assert game_picture.cell_place(0, 5, 4, False) == (0.0, 3.0)
        assert game_picture.cell_place(19, 5, 4, False) == (4.0, 0.0)
        # Hex lattice: each rank up is half a cell right and `HEX_ROW` closer.
        assert game_picture.cell_place(0, 3, 3, True) == (0.0, 2 * game_picture.HEX_ROW)
        assert game_picture.cell_place(7, 3, 3, True) == pytest.approx((2.0, 0.0))

    def test_the_woods_are_the_pages(self) -> None:
        tokens = game_picture.LIGHT
        assert game_picture.shade(0, 0, False) == tokens["sq-dk"]
        assert game_picture.shade(1, 0, False) == tokens["sq-lt"]
        three = {game_picture.shade(file, 0, True) for file in range(3)}
        assert len(three) == 3 and tokens["sq-lt"] in three and tokens["sq-dk"] in three


class TestThePicturesASiteKeeps:
    @staticmethod
    def _made(drawings: list[str]) -> Any:
        def make(drawn: str) -> bytes:
            drawings.append(drawn)
            return b"\x89PNG made from " + str(len(drawings)).encode()
        return make

    def test_a_picture_is_named_only_once_it_is_on_disk(self, tmp_path: Path) -> None:
        drawings: list[str] = []
        held = game_pictures.Pictures(tmp_path / "pictures", can_draw=True,
                                      make=self._made(drawings))
        game = _game()
        assert held.address("ab" * 8, game) is None
        held.settle()
        named = held.address("ab" * 8, game)
        assert named is not None
        found = game_picture.ROUTE.match(named)
        assert found is not None and found.group(1) == "ab" * 8
        assert held.served("ab" * 8, found.group(2)) == b"\x89PNG made from 1"
        # Made once: more reads name it and draw nothing.
        for _ in range(3):
            assert held.address("ab" * 8, game) == named
        held.settle()
        assert len(drawings) == 1
        assert held.held() == 1

    def test_a_renamed_game_is_a_new_picture_and_the_old_one_goes(self, tmp_path: Path) -> None:
        drawings: list[str] = []
        held = game_pictures.Pictures(tmp_path, can_draw=True, make=self._made(drawings))
        held.address("cd" * 8, _game("First name"))
        held.settle()
        first = held.address("cd" * 8, _game("First name"))
        held.address("cd" * 8, _game("Second name"))
        held.settle()
        second = held.address("cd" * 8, _game("Second name"))
        assert first and second and first != second
        assert [path.name for path in tmp_path.glob("*.png")] == [second.rsplit("/", 1)[1]]

    def test_without_a_renderer_or_a_still_board_the_sites_picture_stands(
        self, tmp_path: Path
    ) -> None:
        assert game_pictures.Pictures(tmp_path, can_draw=False).address("ef" * 8, _game()) \
            is None
        held = game_pictures.Pictures(tmp_path, can_draw=True, make=self._made([]))
        broken = _game(spec={"name": "not a game", "board": {"width": 0}})
        assert held.address("ef" * 8, broken) is None
        held.settle()
        assert held.address("ef" * 8, broken) is None
        assert held.held() == 0

    def test_a_failed_drawing_is_not_tried_again(self, tmp_path: Path) -> None:
        tries: list[str] = []

        def fails(drawn: str) -> bytes:
            tries.append(drawn)
            raise ValueError("the renderer refused it")

        held = game_pictures.Pictures(tmp_path, can_draw=True, make=fails)
        for _ in range(3):
            assert held.address("01" * 8, _game()) is None
            held.settle()
        assert len(tries) == 1

    def test_many_readers_ask_for_one_drawing(self, tmp_path: Path) -> None:
        drawings: list[str] = []
        held = game_pictures.Pictures(tmp_path, can_draw=True, make=self._made(drawings))
        game = _game()
        readers = [threading.Thread(target=held.address, args=("23" * 8, game))
                   for _ in range(8)]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join()
        held.settle()
        assert len(drawings) == 1
