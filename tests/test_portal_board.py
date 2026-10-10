"""Portals on the board, as P2 of `docs/plan-portals.md` draws them (Q772b).

What can be held without a browser: the glyph's markup for each end (§4, D3, D9), its words
(the label, the tooltip and the line under the board), the size classes it keeps, the
transit's two ends and its way out as the server reads them off the position, the chooser
of where a piece comes out (D6), the copy it lights on a view slid across the seam (N6), the
legend's two new rows and the tokens the glyph draws in. The look itself -- that the rim reads
on both woods, that the chevrons and the double rim are legible -- is the look check's
(`docs/reference/portal-board-look.md`); the move's journey through a portal is
`test_portal_travel.py`'s (Q836).

The chooser's position is hand-built on the archived `cylinder wormhole` (cylinder chess, one
fixed two-way pair on d4 and d5): a white rook on a4 with rank 4 otherwise empty reaches d4
going right and, round the seam, going left, and comes out on e5 or on c5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_style import _grained, _ratio, _rgb, _theme
from test_viz import _wrapped
from test_web import _in_page
from test_web_landing import _HARNESS

from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web import designing
from chess_variant_sandbox.web.library import Game, engine_for
from chess_variant_sandbox.web.page import tokens
from chess_variant_sandbox.web.play import Session

ROOT = Path(__file__).resolve().parent.parent


def _archived(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "archive" / f"{name}.json").read_text(encoding="utf-8"))


def _game(spec: dict[str, Any]) -> Game:
    return Game(hash=spec_hash(spec), name=str(spec["name"]), description="", path=None,
                source="scratch", spec=spec)  # type: ignore[arg-type]


def _cylinder_session() -> Session:
    spec = _archived("cylinder-wormhole")
    engine = engine_for(spec)
    state = engine.make_state({
        "d4": ("1o", WILD), "d5": ("1b", WILD), "e1": ("K", 0), "e8": ("K", 1),
        "a4": ("R", 0),
    })
    return Session(id="c1", game=_game(spec), seats=("human", "medium"), seed=0, state=state)


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


_SNAPSHOT: dict[str, Any] = _cylinder_session().snapshot()


# ------------------------------------------------------------------ the server's half

class TestTheTransitAsTheServerSaysIt:
    def test_each_way_round_names_its_ends_and_its_way_out(self) -> None:
        ways = {m["text"]: m for m in _SNAPSHOT["moves"] if m.get("portal") is not None}
        assert set(ways) == {"Ra4-d4>c5", "Ra4-d4>e5"}
        for way in ways.values():
            assert (way["portal"], way["portalName"]) == (_square("d4"), "d4")
            assert (way["portalOut"], way["portalOutName"]) == (_square("d5"), "d5")
        # `PORTAL_DIRECTIONS` order: right is 0, left is 4.
        assert ways["Ra4-d4>e5"]["exitDir"] == 0
        assert ways["Ra4-d4>c5"]["exitDir"] == 4

    def test_a_move_played_through_a_portal_keeps_its_two_ends(self) -> None:
        session = _cylinder_session()
        (move,) = [m["move"] for m in session.snapshot()["moves"] if m["text"] == "Ra4-d4>c5"]
        session.play(move)
        entry = session.history[-1]
        assert (entry["from"], entry["to"]) == (_square("a4"), _square("c5"))
        assert (entry["portal"], entry["portalOut"], entry["exitDir"]) == (
            _square("d4"), _square("d5"), 4)

    def test_an_ordinary_move_carries_none_of_it(self) -> None:
        plain = [m for m in _SNAPSHOT["moves"] if m["text"] == "Ra4-a5"]
        assert plain and all("portal" not in m and "exitDir" not in m for m in plain)

    def test_a_still_carries_the_ends_only_where_there_are_portals(self) -> None:
        ferry = designing.opening_preview(_archived("ferry"))
        assert [(end["pair"], end["end"]) for end in ferry["portals"]["ends"]] == [
            (1, "orange"), (1, "blue"), (2, "orange"), (2, "blue")]
        chess = designing.opening_preview(json.loads(
            (ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8")))
        assert "portals" not in chess

    def test_the_offline_replay_carries_the_ends_and_the_transit(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.viz.html import _write_page

        session = _cylinder_session()
        engine = session.engine
        (move,) = [m["move"] for m in session.snapshot()["moves"] if m["text"] == "Ra4-d4>e5"]
        before = session.state
        session.play(move)
        from chess_variant_sandbox.viz.wrapped import portal_transit

        frame = {"text": "Ra4-d4>e5", "from": _square("a4"), "to": _square("e5"),
                 "phase": [], "player": 0, **(portal_transit(engine, before, move) or {}),
                 "material": None, "branching": None}
        path = _write_page(engine, _archived("cylinder-wormhole"),
                           boards=[list(before.board), list(session.state.board)], frames=[frame],
                           meta={"agents": ["-", "-"], "seed": 0, "winner": None,
                                 "reason": "", "plies": 1}, out_path=tmp_path / "r.html")
        page = path.read_text(encoding="utf-8")
        assert '"portals": {"ends": [' in page and '"portalOut": 35' in page
        assert "--portal-a:" in page and "--portal-edge:" in page


# ------------------------------------------------------------------- the glyph itself

_END = "({pair: %d, end: '%s', movedBy: %s, open: %s})"


def _glyph(end: str, partner: str | None, viewer: str, numbered: bool = False) -> str:
    return _wrapped(f"portalGlyph({end}, {partner or 'null'}, {viewer}, 0, 0, 10,"
                    f" {{ numbered: {'true' if numbered else 'false'}, label: 'x' }})")  # type: ignore[return-value]


class TestTheGlyph:
    def test_the_blue_end_is_drawn_double_and_the_orange_end_is_not(self) -> None:
        orange = _glyph(_END % (1, "orange", "[]", "[true, true]"), None, "0")
        blue = _glyph(_END % (1, "blue", "[]", "[true, true]"), None, "0")
        assert "var(--portal-a)" in orange and "pg-double" not in orange
        assert "var(--portal-b)" in blue and blue.count('class="pg-double"') == 1
        assert 'data-portal="1o"' in orange and 'data-portal="1b"' in blue

    @pytest.mark.parametrize(("movers", "fill"), [
        ("[0]", 'fill="var(--p0)"'), ("[1]", 'fill="var(--p1)"'),
        ("[]", 'fill="var(--portal-stone)"'), ("[0, 1]", 'fill-opacity="0.3"')])
    def test_the_interior_says_who_may_move_it(self, movers: str, fill: str) -> None:
        drawn = _glyph(_END % (1, "orange", movers, "[true, true]"), None, "0")
        assert fill in drawn
        assert ('class="pg-hatch"' in drawn) == (movers == "[0, 1]"), "see-through is hatched"

    def test_the_number_only_where_there_are_several_pairs(self) -> None:
        end = _END % (2, "blue", "[1]", "[true, true]")
        assert ">2</text>" in _glyph(end, None, "0", numbered=True)
        assert "<text" not in _glyph(end, None, "0", numbered=False)
        assert 'fill="var(--p1-ink)"' in _glyph(end, None, "0", numbered=True)

    def test_passage_is_drawn_for_the_player_looking(self) -> None:
        # Divide's pair: White goes in at orange and out at blue; Black the other way.
        orange = _END % (1, "orange", "[0]", "[true, false]")
        blue = _END % (1, "blue", "[1]", "[false, true]")
        white_in = _glyph(orange, blue, "0")
        black_out = _glyph(orange, blue, "1")
        assert 'data-passage="in"' in white_in and white_in.count('class="pg-arrow"') == 4
        assert 'data-passage="out"' in black_out and black_out.count('class="pg-arrow"') == 4
        # In points at the middle, out away from it: the chevron's tip is nearer the
        # centre (5, 5) than its arms going in, and further coming out.
        assert white_in != black_out
        two_way = _glyph(_END % (1, "orange", "[]", "[true, true]"),
                         _END % (1, "blue", "[]", "[true, true]"), "0")
        assert 'data-passage="both"' in two_way and "pg-arrow" not in two_way
        nobody = _glyph(orange, blue, "null")
        assert 'data-passage=""' in nobody and "pg-arrow" not in nobody

    def test_an_end_closed_to_the_player_looking_is_dashed_on_stone(self) -> None:
        # Carousel's second pair, to the first player: both ends closed.
        closed = _glyph(_END % (2, "orange", "[]", "[false, true]"),
                        _END % (2, "blue", "[]", "[false, false]"), "0")
        assert 'data-passage="closed"' in closed
        assert 'stroke="var(--portal-stone)"' in closed and "stroke-dasharray" in closed
        assert "pg-arrow" not in closed

    def test_the_label_is_on_the_glyph_for_a_screen_reader(self) -> None:
        drawn = _wrapped("portalGlyph({pair: 1, end: 'orange', movedBy: [], open: [true, true]},"
                         " null, 0, 0, 0, 10, { label: 'Portal \"1\"' })")
        assert 'role="img" aria-label="Portal &quot;1&quot;"' in drawn  # type: ignore[operator]

    def test_the_small_sizes_drop_the_chevrons_then_the_detail(self) -> None:
        tiers = _wrapped("[40, 28, 27.9, 20, 19.9, 0].map(portalTier)")
        assert tiers == ["full", "full", "mid", "mid", "small", "full"]
        from chess_variant_sandbox.web.page import style

        assert 'svg[data-portals="mid"] .pg-arrow' in style.RULES
        for part in ("pg-arrow", "pg-double", "pg-hatch"):
            assert f'svg[data-portals="small"] .{part}' in style.RULES
        assert _wrapped("PORTAL_NUMBER_LEAST") == 10


class TestTheWords:
    def _words(self, end: str, partner: str, square: str = "e3", pairs: int = 1) -> Any:
        return _wrapped(f"portalWords({end}, {partner}, '{square}', {pairs},"
                        " p => ['white', 'black'][p])")

    def test_a_one_way_door_says_who_goes_in_and_who_comes_out(self) -> None:
        said = self._words(_END % (1, "orange", "[0]", "[true, false]"),
                           _END % (1, "blue", "[1]", "[false, true]"))
        assert said["head"] == "The portal, orange end, on e3"
        assert said["label"] == (
            "The portal, orange end, on e3. White's pieces go in here and come out beside"
            " the blue end. Black's pieces come out beside this end, going in at the blue"
            " end. Only white may move it, as a turn.")
        assert said["says"].endswith("one step beyond the other end, in the direction it was going.")

    def test_one_rule_for_both_sides_is_said_once(self) -> None:
        said = self._words(_END % (2, "blue", "[0, 1]", "[true, true]"),
                           _END % (2, "orange", "[0, 1]", "[true, true]"), "b4", pairs=2)
        assert said["label"] == (
            "Portal 2, blue end, on b4. Both sides' pieces go in at either end and come out"
            " beside the other. Either side may move it, as its turn.")

    def test_a_door_nobody_may_use_says_so_and_nothing_of_coming_out(self) -> None:
        said = self._words(_END % (1, "orange", "[]", "[false, false]"),
                           _END % (1, "blue", "[]", "[false, false]"))
        assert "Both sides' pieces cannot go through: to them it is a solid block." in said["label"]
        assert "Nobody moves it." in said["label"] and "comes out" not in said["says"]

    def test_the_way_out_is_named_as_the_board_is_drawn(self) -> None:
        assert _wrapped("[portalStep(4, false), portalStep(4, true), portalStep(1, false),"
                        " portalStep(null, false)]") == [
            {"words": "left", "arrow": "←"}, {"words": "right", "arrow": "→"},
            {"words": "up and right", "arrow": "↗"}, None]


class TestTheLegend:
    def test_a_wall_and_a_portal_have_rows_and_keys(self) -> None:
        rows = _wrapped("MARK_KINDS")
        assert rows["wall"]["shape"] == "wall" and rows["portal"]["shape"] == "portal"  # type: ignore[index]
        keys = _wrapped("[markKey('wall'), markKey('portal')]")
        assert 'data-mark="wall"' in keys[0] and 'data-mark="portal"' in keys[1]  # type: ignore[index]
        assert "var(--portal-a)" in keys[1]  # type: ignore[index]

    def test_the_board_names_them_where_it_draws_them(self) -> None:
        assert _wrapped("[boardKinds([60], {ends: []}), boardKinds([], null), boardKinds(null, null)]"
                        ) == [["wall", "portal"], [], []]
        # And no square mark is ever made of either.
        assert _wrapped("Object.values(squareMarks({goals: [[4], [4]], zones: [[0], [56]]}))"
                        ".flat().map(m => m.kind).sort()") == ["goal", "zone", "zone"]


class TestTheTokens:
    def test_the_rim_holds_against_its_edge_and_the_edge_against_both_woods(self) -> None:
        for dark in (False, True):
            theme = _theme(dark)
            edge = _rgb(theme["portal-edge"])
            for hue in ("portal-a", "portal-b"):
                assert _ratio(_rgb(theme[hue]), edge) >= 3.0, (dark, hue)
            for square in ("sq-lt", "sq-dk"):
                assert _ratio(edge, _grained(dark, square)) >= 3.0, (dark, square)

    def test_orange_and_blue_are_set_once_for_both_themes(self) -> None:
        for name in ("portal-a", "portal-b", "portal-edge", "portal-stone"):
            assert name in tokens.LIGHT
        assert tokens.DARK["portal-stone"] != tokens.LIGHT["portal-stone"]

    def test_the_scripts_draw_a_portal_only_through_the_tokens(self) -> None:
        from chess_variant_sandbox.web.page import SCRIPTS

        for path in SCRIPTS.values():
            text = path.read_text(encoding="utf-8")
            for literal in ("#e8590c", "#1c7ed6", "#2a86e0", "#1e140c"):
                assert literal not in text, (path.name, literal)


# ------------------------------------------------------------- the copy that is lit (N6)

class TestTheLitCopy:
    def test_the_exit_beside_the_end_at_the_seam_is_its_ghost(self) -> None:
        """Cylinder chess slid three files on: d5 stands at the outline's left edge, and c5,
        beside it round the seam, is drawn inside the outline at the far right. The copy to
        light is the ghost of c5 drawn beside d5."""
        found = _wrapped("""(() => {
          const view = boardView(8, 8, { files: true, ranks: false }, { files: 3, ranks: 0 }, false);
          const d5 = squareCell(35, view);
          return { d5, inside: squareCell(34, view), lit: nearestCopy(34, d5, view),
                   right: nearestCopy(36, d5, view) };
        })()""")
        assert found["d5"] == [0, 4] and found["inside"] == [7, 4]  # type: ignore[index]
        assert (found["lit"]["u"], found["lit"]["v"], found["lit"]["depth"]) == (-1, 4, 1)  # type: ignore[index]
        assert (found["right"]["u"], found["right"]["depth"]) == (1, 0)  # type: ignore[index]

    def test_a_board_with_edges_lights_the_square_itself(self) -> None:
        found = _wrapped("nearestCopy(34, [3, 4], flatView(8, 8, false))")
        assert (found["u"], found["v"], found["depth"]) == (2, 4, 0)  # type: ignore[index]


# ------------------------------------------------------------------- the chooser (D6)

_EXITS = """
// The bar's buttons, read back with the way out each names, and their own handlers.
const exits = () => BOX.buttons.filter(b => b.dataset.move !== undefined);
BOX.querySelectorAll = (sel) => sel === '[data-exit]' ? exits().map(b => {
    const at = BOX.html.indexOf('data-move="' + b.dataset.move + '"');
    b.dataset.exit = /data-exit="(\\d+)"/.exec(BOX.html.slice(at))[1]; return b; })
  : sel === '[data-move]' ? exits() : sel === 'button' ? BOX.buttons : [];
const open = () => { clickSquare(%(a4)d); clickSquare(%(d4)d); };
"""


def _run(steps: str, expression: str) -> Any:
    setup = (f"SESSION = {json.dumps(_SNAPSHOT)};\n" + _HARNESS
             + _EXITS % {"a4": _square("a4"), "d4": _square("d4")})
    return _in_page(setup + steps, expression)


class TestTheChooser:
    def test_a_tap_on_the_end_asks_where_to_come_out(self) -> None:
        stage, looking, modal, text = _run(
            "open();", "[promotionStage(), BOX.looking, BOX.modal, said()]")
        assert stage == "exit"
        # The bar under the board, the board left undimmed, so the exit can be lit on it.
        assert looking is True and modal == "false"
        assert text.startswith("Come out where?")
        assert "← Left of the portal, c5" in text and "→ Right of the portal, e5" in text
        assert "Esc: choose another move" in text and "Cancel" in text

    def test_a_board_turned_round_names_the_ways_as_drawn(self) -> None:
        text = _run("FLIPPED = true; open();", "said()")
        assert "→ Right of the portal, c5" in text and "← Left of the portal, e5" in text

    def test_a_number_key_takes_that_way(self) -> None:
        sent = _run("open(); key('2');", "CALLS.map(c => textOf(c[1].move))")
        assert sent == ["Ra4-d4>e5"]

    def test_escape_puts_it_away_with_nothing_moved(self) -> None:
        assert _run("open(); key('Escape');", "[BOX.hidden, CALLS.length, PROMO]") == [
            True, 0, None]

    def test_a_tap_on_an_exit_square_is_that_move_at_once(self) -> None:
        sent = _run(f"clickSquare({_square('a4')}); clickSquare({_square('c5')});",
                    "[BOX.hidden, CALLS.map(c => textOf(c[1].move))]")
        assert sent == [True, ["Ra4-d4>c5"]]


class TestTheLineUnderTheBoard:
    def test_a_tapped_end_says_what_it_does_whoever_is_to_move(self) -> None:
        said = _run(f"""
          const line = {{ hidden: true, innerHTML: '' }};
          const byId = document.getElementById;
          document.getElementById = id => id === 'pieceSays' ? line : byId(id);
          SESSION.sideToMove = 1;   // the machine's turn: the tap still asks
          clickSquare({_square("d5")});
        """, "[line.hidden, line.innerHTML, INSPECT]")
        hidden, html, inspected = said
        assert hidden is False and inspected == _square("d5")
        assert html.startswith("<b>The portal, blue end, on d5</b>. Both sides&#39; pieces go in")
        assert "Nobody moves it." in html
