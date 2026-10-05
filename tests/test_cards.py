"""The library's cards carry the game's opening position (V6, `docs/plan-visual.md` §3.3).

`/api/library` sends each card a `still` (the opening position, shaped as the designer's
`preview`) with its pieces' drawings named by stem, and the drawings once for the whole
listing in `pieceArt`. The page puts them back and draws the card's thumbnail with the still
board, passing the game's own hash so each board's wood is its own.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from test_wood import _run

from chess_variant_sandbox.web.server import App

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"


def _listing(tmp_path: Path) -> dict:
    rulesets = tmp_path / "rulesets"
    rulesets.mkdir()
    for name in ("chess.json", "shogi.json"):
        shutil.copy(RULESETS / name, rulesets / name)
    app = App(rulesets, tmp_path / "runs", archive_dir=tmp_path / "archive")
    return app.get("/api/library")


class TestTheStillField:
    def test_every_card_carries_its_opening_position(self, tmp_path: Path) -> None:
        listing = _listing(tmp_path)
        assert len(listing["games"]) == 2
        for card in listing["games"]:
            still = card["still"]
            assert still["width"] * still["height"] == len(still["board"])
            assert any(still["board"]), card["name"]

    def test_each_drawing_is_sent_once_and_named_by_the_cards(self, tmp_path: Path) -> None:
        listing = _listing(tmp_path)
        art = listing["pieceArt"]
        named = set()
        for card in listing["games"]:
            for pair in card["still"]["icons"]:
                for icon in pair:
                    if icon is None:
                        continue
                    assert set(icon) == {"stem", "rotate"}, "a card carries no drawing"
                    named.add(icon["stem"])
        assert named and named <= set(art)
        assert all(href.startswith("data:image/svg+xml") for href in art.values())


_STILL = {"width": 4, "height": 4, "board": [1, 0, 0, 2] + [0] * 12, "nTypes": 1,
          "sideStride": 1, "pieceIds": ["x"], "pieceNames": ["x"],
          "icons": [[{"stem": "x-white", "rotate": False}, {"stem": "x-black", "rotate": False}]],
          "handsUsed": True, "hands": [[0], [0]],
          "zones": [[0, 1, 2, 3], [12, 13, 14, 15]]}


def _thumbs(*hashes: str) -> list[str]:
    games = [{"hash": digest, "still": _STILL} for digest in hashes]
    return _run("(() => { PIECE_ART = { 'x-white': 'data:image/svg+xml;base64,AAAA',"
                " 'x-black': 'data:image/svg+xml;base64,BBBB' };"
                f" return {json.dumps(games)}.map(thumbHtml); }})()")


class TestTheThumbnail:
    def test_each_game_has_its_own_wood(self) -> None:
        """The queue's note: a thumbnail passes its game's hash into `boardPreview`, so two
        games of one shape side by side are not cut alike."""
        one, two, again = _thumbs("4c1922ffae26", "975dde173f01", "4c1922ffae26")

        def cut(drawn: str) -> list[str]:
            return re.findall(r'patternTransform="([^"]+)"', drawn)

        assert cut(one) and cut(one) != cut(two)
        assert cut(one) == cut(again)

    def test_the_drawings_are_put_back(self) -> None:
        (drawn,) = _thumbs("4c1922ffae26")
        assert 'href="data:image/svg+xml;base64,AAAA"' in drawn
        assert 'href="data:image/svg+xml;base64,BBBB"' in drawn

    def test_a_thumbnail_is_a_glance(self) -> None:
        """No hand strip, and no mark: a card has no legend line to name one by (S3)."""
        (drawn,) = _thumbs("4c1922ffae26")
        assert "stillHand" not in drawn
        assert "hatch" not in drawn and "data-mark" not in drawn

    def test_a_card_without_a_still_draws_no_board(self) -> None:
        assert _run("thumbHtml({ hash: 'aaa', still: null })") == ""

    def test_the_card_leads_with_the_board_and_two_pills(self) -> None:
        """S5: the pills are the level and the score; what else the checks found is a
        quiet line under them."""
        from page_text import PAGE

        card = PAGE[PAGE.index("function cardHtml(game)"):]
        card = card[:card.index("\n}\n")]
        assert card.index("${thumbHtml(game)}") < card.index("<h2>")
        pills = card[card.index('<div class="row pills">'):card.index("</div>", card.index('<div class="row pills">'))]
        assert "levelTag(game)" in pills and "scoreTag(game)" in pills
        assert "unbeatenTag" not in pills and "fairyTag" not in pills
