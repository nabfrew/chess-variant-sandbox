"""Learning a game on its page (`docs/archive/plan-understanding.md` §4.2, K1-K3).

K1: the game page draws the start position before a game starts, from the server's
`opening` (the card's `still` with its drawings put back), with the marks a rule puts on a
square and the legend line naming them.

K2: each piece on its own small board, alone on an empty board, with the engine's own
moves for it (`move_map(alone=True)`), and *moves like a chess ‹piece›* where the server
found the reference game's piece giving the same answer.

K3: a person's first game on a game seats the gentlest computer player above random on
its ladder (the server's `firstGame`, read off the rating table as the front page's first
step is), as practice, with the rules tab open; a tapped piece's sentences under the
board; and *watch two computer players first* on a game far from chess.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from test_wood import _run

from chess_variant_sandbox.web.server import App

RULESETS = Path(__file__).resolve().parent.parent / "rulesets"


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> App:
    root = tmp_path_factory.mktemp("learning")
    rulesets = root / "rulesets"
    rulesets.mkdir()
    for name in ("chess.json", "shogi.json", "kingofthehill.json"):
        shutil.copy(RULESETS / name, rulesets / name)
    return App(rulesets, root / "runs", archive_dir=root / "archive")


def _game(app: App, name: str) -> dict:
    listing = app.get("/api/library")
    digest = next(card["hash"] for card in listing["games"] if card["name"] == name)
    return app.get("/api/game/" + digest)


def _name_of(file: str) -> str:
    return json.loads((RULESETS / file).read_text(encoding="utf-8"))["name"]


class TestTheStartPosition:
    def test_the_game_carries_its_opening_position_with_its_drawings(self, app: App) -> None:
        game = _game(app, _name_of("chess.json"))
        opening = game["opening"]
        assert opening["width"] * opening["height"] == len(opening["board"])
        assert any(opening["board"])
        assert opening["toMove"] == 0
        hrefs = [icon["href"] for pair in opening["icons"] for icon in pair if icon]
        assert hrefs and all(href.startswith("data:image/svg+xml") for href in hrefs)

    def test_it_is_the_cards_still(self, app: App) -> None:
        """One payload serves the card and the page: the same squares, the same marks."""
        listing = app.get("/api/library")
        card = next(c for c in listing["games"] if c["name"] == _name_of("kingofthehill.json"))
        opening = app.get("/api/game/" + card["hash"])["opening"]
        for key in ("width", "height", "board", "goals", "zones", "wrap"):
            assert opening[key] == card["still"][key], key

    def test_the_page_draws_it_with_its_marks_and_their_legend(self, app: App) -> None:
        game = _game(app, _name_of("kingofthehill.json"))
        drawn = _run(f"openingHtml({json.dumps(game)})")
        assert 'id="opening"' in drawn and "<svg" in drawn
        assert 'data-legend="goal"' in drawn, "the hill's goal squares are named"
        assert "moves first" in drawn
        assert 'aria-label="The start position.' in drawn

    def test_a_game_with_no_opening_draws_nothing(self) -> None:
        assert _run("openingHtml({ hash: 'aaa', opening: null })") == ""

    def test_it_heads_the_play_column_and_the_list_stays_folded(self) -> None:
        from page_text import DETAIL

        play = DETAIL[DETAIL.index('<div class="gamePlay">'):]
        assert play.index("${openingHtml(game)}") < play.index('<h3 class="u-mt-0">Play</h3>')
        assert "<summary class=\"muted small\">where the pieces start</summary>" in DETAIL


# ------------------------------------------------------------------------- K2

class TestEachPieceOnItsOwnBoard:
    """K2: each piece alone on an empty board, the engine's answer (`move_map(alone=True)`),
    sent with the game; `like` is the reference game's piece with the same answer."""

    def test_every_piece_has_its_map(self, app: App) -> None:
        game = _game(app, _name_of("shogi.json"))
        maps = game["pieceMaps"]
        assert len(maps) == len(game["pieces"])
        for found in maps:
            assert {"from", "value", "moves", "takes", "like"} <= set(found)
            assert found["moves"] or found["takes"]

    def test_the_maps_are_the_designers_diagram(self, app: App) -> None:
        """The same call the designer's *where it goes* makes, so the two never differ."""
        from chess_variant_sandbox.web import designing

        game = _game(app, _name_of("shogi.json"))
        for index, found in enumerate(game["pieceMaps"]):
            asked = designing.move_map(game["spec"], index, alone=True)
            assert (found["from"], found["moves"], found["takes"]) == (
                asked["from"], asked["moves"], asked["takes"])

    def test_the_reference_games_pieces_move_like_themselves(self, app: App) -> None:
        game = _game(app, _name_of("chess.json"))
        assert [m["like"] for m in game["pieceMaps"]] == [p["name"] for p in game["pieces"]]

    def test_a_piece_unlike_any_says_nothing(self, app: App) -> None:
        game = _game(app, _name_of("shogi.json"))
        named = {p["name"]: m["like"] for p, m in zip(game["pieces"], game["pieceMaps"], strict=True)}
        # Shogi's bishop and rook are chess's own; its lance and gold are nothing in chess.
        assert named["bishop"] == "bishop" and named["rook"] == "rook"
        assert named["lance"] is None and named["gold"] is None


def _map(width: int, height: int, steps: list[tuple[int, int]], *, sliding: bool) -> tuple:
    start = (height // 2) * width + width // 2
    x, y = start % width, start // width
    squares = []
    for dx, dy in steps:
        for k in range(1, (max(width, height) if sliding else 1) + 1):
            tx, ty = x + dx * k, y + dy * k
            if not (0 <= tx < width and 0 <= ty < height):
                break
            squares.append(ty * width + tx)
    found = {"from": start, "moves": squares, "takes": squares}
    return found, {"width": width, "height": height}


class TestTheComparison:
    ORTHOGONAL = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def test_a_rook_on_a_wider_board_is_a_rook(self) -> None:
        from chess_variant_sandbox.web.library import same_moves

        wide = _map(12, 8, self.ORTHOGONAL, sliding=True)
        chess = _map(8, 8, self.ORTHOGONAL, sliding=True)
        assert same_moves(*wide, *chess) and same_moves(*chess, *wide)

    def test_a_short_slider_is_not_a_rook(self) -> None:
        from chess_variant_sandbox.web.library import same_moves

        one, board = _map(8, 8, self.ORTHOGONAL, sliding=True)
        short = {**one, "moves": [s for s in one["moves"] if abs(s - one["from"]) in (1, 2, 8, 16)]}
        short["takes"] = short["moves"]
        assert not same_moves(short, board, one, board)

    def test_a_board_too_small_to_tell_says_nothing(self) -> None:
        from chess_variant_sandbox.web.library import same_moves

        small = _map(3, 4, [(1, 1), (-1, 1), (1, -1), (-1, -1)], sliding=False)
        bishop = _map(8, 8, [(1, 1), (-1, 1), (1, -1), (-1, -1)], sliding=True)
        assert not same_moves(*small, *bishop)


_PIECES_GAME = {
    "hash": "abc",
    "opening": {"width": 8, "height": 8, "board": [0] * 64, "nTypes": 2, "sideStride": 2,
                "pieceIds": ["a", "b"], "pieceNames": ["a", "b"], "icons": [], "toMove": 0},
    "pieces": [{"id": "a", "name": "plain", "moves": ["like a knight"]},
               {"id": "b", "name": "odd", "moves": ["one square any way"]}],
    "pieceMaps": [{"from": 27, "value": 1, "moves": [10, 12], "takes": [10, 12], "like": "knight"},
                  {"from": 27, "value": 2, "moves": [26, 28, 19], "takes": [], "like": None}],
}


class TestThePiecesOnThePage:
    def test_the_unusual_piece_comes_first_and_open(self) -> None:
        drawn = _run(f"pieceBoardsHtml({json.dumps(_PIECES_GAME)})")
        assert drawn.index('data-piece="1"') < drawn.index('data-piece="0"')
        odd = drawn[drawn.index('data-piece="1"'):drawn.index('data-piece="0"')]
        assert '<details class="pieceMap" open>' in odd

    def test_a_piece_like_chesss_says_so_and_stays_folded(self) -> None:
        drawn = _run(f"pieceBoardsHtml({json.dumps(_PIECES_GAME)})")
        plain = drawn[drawn.index('data-piece="0"'):]
        assert "moves like a chess knight" in plain
        assert '<details class="pieceMap">' in plain

    def test_the_dots_are_the_servers(self) -> None:
        drawn = _run(f"pieceBoardsHtml({json.dumps(_PIECES_GAME)})")
        odd = drawn[drawn.index('data-piece="1"'):drawn.index('data-piece="0"')]
        assert odd.count('fill="var(--dot)"') == 3

    def test_no_map_no_board(self) -> None:
        bare = {**_PIECES_GAME, "pieceMaps": None}
        drawn = _run(f"pieceBoardsHtml({json.dumps(bare)})")
        assert "pieceMap" not in drawn and "plain" in drawn


# ------------------------------------------------------------------------- K3

class TestTheFirstGamesOpponent:
    def test_the_gentlest_and_the_next_up_come_from_the_ladder(self, app: App) -> None:
        first = _game(app, _name_of("chess.json"))["firstGame"]
        assert first["gentle"]["agent"] not in ("random", "human")
        assert first["harder"]["agent"] != first["gentle"]["agent"]
        assert first["far"] == []

    def test_a_rating_table_orders_them(self, app: App, monkeypatch: pytest.MonkeyPatch) -> None:
        """The order differs by game, which is why it is read from the table: here the
        ladder's third player is rated lowest, so it is the gentlest."""
        game = _game(app, _name_of("chess.json"))
        ladder = [o["agent"] for o in game["opponents"]
                  if o["store"] == "ladder" and o["agent"] not in ("human", "random")]
        table = {"standings": [{"name": agent, "rating": 100.0 * at}
                               for at, agent in enumerate(ladder)]}
        table["standings"][2]["rating"] = -500.0
        monkeypatch.setattr(app.ratings, "stored", lambda digest: table)
        first = app.get("/api/game/" + game["hash"])["firstGame"]
        assert first["gentle"]["agent"] == ladder[2]
        assert first["harder"]["agent"] == ladder[0]
        assert first["basis"] == "ratings"

    def test_a_game_far_from_chess_says_why(self, app: App) -> None:
        far = _game(app, _name_of("kingofthehill.json"))["firstGame"]["far"]
        assert far == ["it is won by getting a piece to a particular place"]


_FIRST = {"hash": "abc", "firstGame": {
    "gentle": {"agent": "greedy", "name": "greedy"},
    "harder": {"agent": "minimax:nodes=100", "name": "easy"}, "basis": "ladder", "far": []}}
_ROSTER = "ROSTER = [{agent: 'human'}, {agent: 'greedy'}, {agent: 'minimax:nodes=100'}];"
_STORE = ("const box = {}; window.localStorage = { getItem: k => (k in box ? box[k] : null),"
          " setItem: (k, v) => { box[k] = v; } };")


class TestTheFirstGameOnThePage:
    def test_a_first_visit_seats_the_gentlest_as_practice(self) -> None:
        said = _run(f"(() => {{ {_ROSTER} const g = {json.dumps(_FIRST)};"
                    " return [firstSeat(g), secondSeat(g), firstGameHtml(g),"
                    " modePicker({...g, standing: {rated: true, signedIn: true}})]; })()")
        seat, second, line, mode = said
        assert seat == second == "greedy"
        assert "gentlest computer player on this game, as practice" in " ".join(line.split())
        assert "<b>greedy</b>" in line and "a harder one: easy" in line
        assert '<option value="sandbox" selected>' in mode

    def test_after_a_game_here_the_usual_opponent_comes_back(self) -> None:
        said = _run(f"(() => {{ {_ROSTER} {_STORE} const g = {json.dumps(_FIRST)};"
                    " notePlayed('abc'); return [firstSeat(g), secondSeat(g), firstGameHtml(g)];"
                    " })()")
        assert said[0] is None and said[1] != "greedy" and said[2] == ""

    def test_the_gentlest_is_labelled_in_the_picker(self) -> None:
        drawn = _run(f"(() => {{ {_ROSTER} const g = {json.dumps(_FIRST)};"
                     " const options = [{agent: 'human', name: 'you'},"
                     " {agent: 'greedy', name: 'greedy', store: 'ladder'},"
                     " {agent: 'minimax:nodes=100', name: 'easy', store: 'ladder'}];"
                     " return seatCardsHtml('seat1', options, 'greedy', null, g); })()")
        assert drawn.count('class="gentlest">gentlest</span>') == 1
        assert "greedy \u00b7 gentlest" in drawn or "greedy · gentlest" in drawn

    def test_a_far_game_offers_a_watch_first(self) -> None:
        far = {**_FIRST, "firstGame": {**_FIRST["firstGame"], "far": ["it has no royal piece to trap"]}}
        line = _run(f"(() => {{ {_ROSTER} return firstGameHtml({json.dumps(far)}); }})()")
        assert "watch two computer players first" in line
        assert "it has no royal piece to trap" in line

    def test_a_first_game_opens_on_the_rules(self) -> None:
        from page_text import BOARD, DETAIL

        assert "if (firstSeat(game) && seats.includes('human')) OPEN_TAB = 'rules';" in DETAIL
        assert "TAB = OPEN_TAB || 'moves';" in BOARD


class TestATappedPiecesSentence:
    def test_it_is_the_pieces_tabs_own_words_under_the_board(self) -> None:
        from page_text import BOARD

        frame = BOARD[BOARD.index("function frameHtml("):]
        assert frame.index('id="hand"') < frame.index('id="pieceSays"') < frame.index(
            'data-frame="controls"')
        says = BOARD[BOARD.index("function drawPieceSays("):]
        says = says[:says.index("\n}\n")]
        assert "(game.pieces || [])[piece.type]" in says and "about.moves" in says
