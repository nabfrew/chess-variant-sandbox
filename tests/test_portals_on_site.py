"""Portal games on the game page and the site (P5 of `docs/archive/plan-portals.md`, Q772d).

What publishing and playing a portal game on a local site found (`docs/reference/
site-portals-look.md`): the game page's *watch two computer players first* called a portal
game far from chess because its ends stand in the third owner's block, the block a wild
piece stands in.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from chess_variant_sandbox.web.server import App

ARCHIVE = Path(__file__).resolve().parent.parent / "archive"


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> App:
    root = tmp_path_factory.mktemp("portals-on-site")
    (root / "rulesets").mkdir()
    archive = root / "archive"
    archive.mkdir()
    for name in ("divide.json", "wolves.json", "cylinder-wormhole.json"):
        shutil.copy(ARCHIVE / name, archive / name)
    return App(root / "rulesets", root / "runs", archive_dir=archive)


def _game(app: App, file: str) -> dict:
    name = json.loads((ARCHIVE / file).read_text(encoding="utf-8"))["name"]
    listing = app.get("/api/library")
    digest = next(card["hash"] for card in listing["games"] if card["name"] == name)
    return app.get("/api/game/" + digest)


class TestAPortalIsNotAWildPiece:
    @pytest.mark.parametrize("file", ["divide.json", "cylinder-wormhole.json"])
    def test_a_portal_game_has_no_wild_pieces(self, app: App, file: str) -> None:
        game = _game(app, file)
        assert game["opening"]["portals"]["ends"], "the opening position draws the ends"
        assert "it has wild pieces that belong to neither side" not in game["firstGame"]["far"]

    def test_a_wild_piece_still_is_one(self, app: App) -> None:
        far = _game(app, "wolves.json")["firstGame"]["far"]
        assert "it has wild pieces that belong to neither side" in far


# A game between two people, a 4 x 4 board: the second player's piece on a1, and a move
# held that goes into the end on b2 and comes out beside the end on c3, landing on d4.
_HELD = """
SESSION = { id: 'g1', ruleset: { sides: ['white', 'black'], width: 4, height: 4, nTypes: 1,
    pieceIds: ['R'], icons: [] }, seats: ['human', 'human'], sideToMove: 1, inCheck: false,
    remaining: 1, outcome: null, board: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    history: [{ player: 0, text: 'a4-a3' }],
    moves: [{ move: 7, from: 0, to: 15, portal: 5, portalOut: 10, text: 'Ra1-b2>d4' },
            { move: 8, from: 0, to: 15, text: 'Ra1-d4' }],
    correspondence: { you: 1, players: ['ada', 'bo'], drawOffer: null, mayCancel: true,
      times: [new Date(Date.now() - 3 * 3600e3).toISOString()],
      rules: ['The player to move loses after five days without a move.'] } };
FRAME = boardView(4, 4, wrapAxes(null), { files: 0, ranks: 0 }, false);
"""


class TestAHeldTransit:
    """Between two people a held move is drawn before it is sent (Q733). A move through a
    portal is drawn in its two legs, as the trail draws it once made, never as one arrow
    across the board from where it leaves to where it lands."""

    def test_a_transit_is_held_in_two_legs(self) -> None:
        from test_web import _in_page

        held = _in_page(_HELD + "CORR_HELD = 7;", "corrHeldDrawn()")
        assert (held["from"], held["portal"], held["portalOut"], held["dest"]) == (0, 5, 10, 15)
        html = _in_page(_HELD + "CORR_HELD = 7;", "heldHtml(corrHeldDrawn(), 25, 4, 4, null)")
        assert html.count('data-held="arrow"') == 1
        # Two arrows, each a halo and a dashed line with its head.
        assert html.count("<line") == 4 and html.count("stroke-dasharray") == 2

    def test_a_plain_move_keeps_its_one_arrow(self) -> None:
        from test_web import _in_page

        held = _in_page(_HELD + "CORR_HELD = 8;", "corrHeldDrawn()")
        assert "portal" not in held
        html = _in_page(_HELD + "CORR_HELD = 8;", "heldHtml(corrHeldDrawn(), 25, 4, 4, null)")
        assert html.count("<line") == 2 and html.count("stroke-dasharray") == 1
