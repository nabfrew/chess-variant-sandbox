"""Portals on the board, as P2 of `docs/archive/plan-portals.md` draws them (Q772b).

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
import re
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
        assert "--pair-1:" in page and "--portal-edge:" in page and "--portal-a" not in page
        assert ".pg-idle { animation: pg-turn 14s linear infinite; }" in page
        assert "{ .pg-idle { animation: none; } }" in page


# ------------------------------------------------------------------- the glyph itself

_END = "({pair: %d, end: '%s', movedBy: %s, open: %s})"


def _glyph(end: str, numbered: bool = False, **more: str) -> str:
    extra = "".join(f", {k}: {v}" for k, v in more.items())
    return _wrapped(f"portalGlyph({end}, 0, 0, 10,"  # type: ignore[return-value]
                    f" {{ numbered: {'true' if numbered else 'false'}, label: 'x'{extra} }})")


def _radius(drawn: str, part: str) -> float:
    """The radius of the glyph's circle of this class (a cell is 10 in `_glyph`)."""
    found = re.search(rf'r="([\d.]+)" class="{part}"', drawn)
    assert found, part
    return float(found.group(1))


def _arms(drawn: str) -> list[str]:
    """The players each arm of the glyph is drawn for, in order; none where nobody goes in."""
    return re.findall(r'class="pg-arm" data-arm="(\w+)"', drawn)


class TestTheGlyph:
    """The vortex (Q838b, `docs/archive/plan-portal-look.md` §2.3, as Q838f corrected it after the cold
    read): the well is the pair's colour, the arms are whose pieces may go in, and what it
    stands on is who may move it."""

    def test_both_ends_of_a_pair_are_its_colour_and_nothing_says_which_end(self) -> None:
        for pair in (1, 2, 3, 4):
            first = _glyph(_END % (pair, "orange", "[]", "[true, true]"))
            second = _glyph(_END % (pair, "blue", "[]", "[true, true]"))
            for drawn in (first, second):
                assert f'stop-color="var(--pair-{pair})"' in drawn
                assert "--portal-a" not in drawn and "--portal-b" not in drawn
                assert "pg-double" not in drawn and f'data-pair="{pair}"' in drawn
            # The two ends differ only by their name (and the well's own id).
            assert (re.sub(r"pgwell\d+", "", first).replace(f'data-portal="{pair}o"', "")
                    == re.sub(r"pgwell\d+", "", second).replace(f'data-portal="{pair}b"', ""))

    @pytest.mark.parametrize(("open_", "arms"), [
        ("[true, true]", ["0", "1"]), ("[true, false]", ["0", "0"]),
        ("[false, true]", ["1", "1"]), ("[false, false]", [])])
    def test_the_arms_are_whose_pieces_may_go_in(self, open_: str, arms: list[str]) -> None:
        drawn = _glyph(_END % (1, "orange", "[]", open_))
        assert _arms(drawn) == arms
        fills = re.findall(r'class="pg-arm" data-arm="\w+" d="[^"]+" fill="([^"]+)"', drawn)
        want = {"0": "var(--p0)", "1": "var(--p1)"}
        assert fills == [want[a] for a in arms]

    def test_an_end_nobody_goes_in_by_has_no_arms_at_all(self) -> None:
        """Q838f (F3): the way out's faint dark arms were read as Black's arms on yellow; a way
        out is now a still well, with no arm, no wedge and nothing to turn."""
        for movers in ("[]", "[0]", "[0, 1]"):
            drawn = _glyph(_END % (2, "orange", movers, "[false, false]"), idle="true")
            assert "pg-arm" not in drawn and "pg-wedge" not in drawn and "data-arm" not in drawn
            assert "pg-idle" not in drawn and "animation-delay" not in drawn
            assert 'class="pg-well"' in drawn

    def test_with_both_the_arm_from_the_bottom_is_the_bottom_players(self) -> None:
        assert _arms(_glyph(_END % (1, "orange", "[]", "[true, true]"))) == ["0", "1"]
        assert _arms(_glyph(_END % (1, "orange", "[]", "[true, true]"), flipped="true")) == [
            "1", "0"]

    def test_a_white_arm_is_edged_dark_and_a_black_one_in_the_pairs_colour(self) -> None:
        """Q838f (F3): a black arm's light edge read as a white arm at the mid size, so a black
        arm is edged in the pair's own colour; nothing light edges it."""
        for pair in (1, 2):
            drawn = _glyph(_END % (pair, "orange", "[]", "[true, true]"))
            assert 'fill="var(--p0)" stroke="var(--portal-edge)"' in drawn
            assert f'fill="var(--p1)" stroke="var(--pair-{pair})"' in drawn
            assert "--p1-ink" not in drawn

    def test_the_arms_stop_short_of_the_rim_so_the_outer_ring_is_the_pairs(self) -> None:
        """Q838f (F2): white arms reaching a sunk well's rim read as a white token's collar. The
        arms (and the small tier's wedges) end inside the well, so its outermost ring is
        always the pair's colour."""
        for movers in ("[]", "[0]"):
            drawn = _glyph(_END % (1, "orange", movers, "[true, true]"))
            well = _radius(drawn, "pg-well")
            paths = re.findall(r'class="pg-(?:arm|wedge)" data-arm="\w" d="([^"]+)"', drawn)
            assert len(paths) == 4
            for path in paths:
                points = [tuple(map(float, xy.split())) for xy in re.split(r"[MLZ]", path) if xy.strip()]
                reach = max(((x - 5) ** 2 + (y - 5) ** 2) ** 0.5 for x, y in points)
                assert reach < well * 0.9

    @pytest.mark.parametrize(("movers", "fills"), [
        ("[0]", ['fill="var(--p0)"']), ("[1]", ['fill="var(--p1)"']),
        ("[0, 1]", ['fill="var(--p0)"', 'fill="var(--p1)"'])])
    def test_a_portal_somebody_moves_stands_on_a_token_of_their_colour(
            self, movers: str, fills: list[str]) -> None:
        drawn = _glyph(_END % (1, "orange", movers, "[false, false]"))
        token = re.search(r'<g class="pg-token" data-moved="([^"]*)">(.*?)</g>', drawn, re.S)
        assert token and token.group(1) == movers.strip("[]").replace(",", "")
        for fill in fills:
            assert fill in token.group(2)
        assert "pg-hole" not in drawn

    def test_a_tokens_collar_is_wide_round_a_smaller_well(self) -> None:
        """Q838f (F2): the collar is a sixth of a cell or more, so a token reads as one at every
        size, and its well is smaller than a sunk one."""
        raised = _glyph(_END % (1, "orange", "[0]", "[true, true]"))
        sunk = _glyph(_END % (1, "orange", "[]", "[true, true]"))
        token = re.search(r'<g class="pg-token"[^>]*>.*?r="([\d.]+)"', raised, re.S)
        assert token and float(token.group(1)) - _radius(raised, "pg-well") >= 1.6  # cell: 10
        assert _radius(raised, "pg-well") < _radius(sunk, "pg-well")

    def test_with_both_the_tokens_bottom_half_is_the_bottom_players(self) -> None:
        upright = _glyph(_END % (1, "orange", "[0, 1]", "[true, true]"))
        turned = _glyph(_END % (1, "orange", "[0, 1]", "[true, true]"), flipped="true")
        # The first half drawn is the lower one (`0 0 0`: the arc through the bottom).
        assert re.search(r'0 0 0 [\d.]+ [\d.]+Z"\s+fill="var\(--p0\)"', upright)
        assert re.search(r'0 0 0 [\d.]+ [\d.]+Z"\s+fill="var\(--p1\)"', turned)

    def test_a_portal_nobody_moves_is_sunk_into_the_board(self) -> None:
        """A hole: the shadow of its top edge inside it, over the arms, and no light rim (Q838f,
        F2: the old lip's light lower edge read as a token's collar)."""
        for open_ in ("[true, true]", "[false, false]"):
            drawn = _glyph(_END % (1, "orange", "[]", open_))
            assert 'class="pg-hole"' in drawn and "pg-token" not in drawn
            assert "pg-lip" not in drawn and "var(--p0)" not in drawn.split("pg-well")[0]
            hole = re.search(r'<path class="pg-hole" d="[^"]+" fill="([^"]+)"', drawn)
            assert hole and hole.group(1) == "var(--portal-edge)"
        # Drawn over the arms, so they go under the edge's shadow.
        drawn = _glyph(_END % (1, "orange", "[]", "[true, true]"))
        assert drawn.index("pg-arms") < drawn.index("pg-hole")

    def test_the_arms_turn_in_only_where_somebody_may_go_in_and_only_when_idle(self) -> None:
        inward = _glyph(_END % (1, "orange", "[]", "[true, false]"), idle="true")
        way_out = _glyph(_END % (1, "orange", "[]", "[false, false]"), idle="true")
        still = _glyph(_END % (1, "orange", "[]", "[true, false]"))
        assert 'class="pg-arms pg-idle sc-' in inward and 'data-turn="in"' in inward
        assert "pg-arms" not in way_out and "pg-idle" not in way_out  # no arms, nothing turns
        assert "pg-idle" not in still and "animation-delay" not in still
        # Its turning point is the end's own centre, in the rule its class names (G12).
        from test_fold_look import STYLES_OF

        styles = _wrapped(STYLES_OF + "(portalGlyph(" + _END % (1, "orange", "[]", "[true, false]")
                          + ", 0, 0, 10, { numbered: false, label: 'x', idle: true }))")
        assert "transform-origin:5.000px 5.000px" in styles  # type: ignore[operator]

    def test_the_two_ends_of_a_pair_turn_in_step_and_pairs_out_of_step(self) -> None:
        phases = _wrapped("[portalPhase(1), portalPhase(1), portalPhase(2), portalPhase(3)]")
        assert phases[0] == phases[1]  # type: ignore[index]
        assert len(set(phases[1:])) == 3  # type: ignore[index]
        assert all(-14 < p <= 0 for p in phases)  # type: ignore[union-attr]

    def test_the_number_sits_in_the_eye_in_the_pairs_colour(self) -> None:
        end = _END % (2, "blue", "[1]", "[true, true]")
        numbered = _glyph(end, numbered=True)
        assert ">2</text>" in numbered and 'class="pg-eye"' in numbered
        assert re.search(r'<text class="pg-num"[^>]*fill="var\(--pair-2\)"', numbered)
        plain = _glyph(end, numbered=False)
        assert "<text" not in plain and "pg-eye" not in plain

    def test_the_label_is_on_the_glyph_for_a_screen_reader(self) -> None:
        drawn = _wrapped("portalGlyph({pair: 1, end: 'orange', movedBy: [], open: [true, true]},"
                         " 0, 0, 10, { label: 'Portal \"1\"' })")
        assert 'role="img" aria-label="Portal &quot;1&quot;"' in drawn  # type: ignore[operator]

    def test_the_small_sizes_thin_the_edges_then_draw_wedges(self) -> None:
        tiers = _wrapped("[40, 28, 27.9, 20, 19.9, 0].map(portalTier)")
        assert tiers == ["full", "full", "mid", "mid", "small", "full"]
        from chess_variant_sandbox.web.page import style

        drawn = _glyph(_END % (1, "orange", "[]", "[true, true]"))
        assert drawn.count('class="pg-arm"') == 2 and drawn.count('class="pg-wedge"') == 2
        assert drawn.count('class="pg-mass"') == 2
        assert ".pg-wedge, .pg-mass, .pg-beak-cut { display: none; }" in style.RULES
        # The token and the hole are drawn at every size (Q838f, F2): no tier hides either.
        for part in ("pg-hole", "pg-token", "pg-lip"):
            assert f".{part}" not in style.RULES
        # Q838h (the cold read's §7.5): at the mid size the arms are drawn by mass, not by
        # their edges -- the wider `pg-mass` in place of `pg-arm`, its edge a screen pixel.
        assert 'svg[data-portals="mid"] .pg-arm, svg[data-phone] .pg-arm { display: none; }' \
            in style.RULES
        assert ('svg[data-portals="mid"] .pg-mass, svg[data-phone] .pg-mass {\n'
                '    display: inline; vector-effect: non-scaling-stroke; stroke-width: 1px; }'
                in style.RULES)
        assert 'svg[data-portals="small"] .pg-arm' in style.RULES
        assert ".pg-small .pg-arm" in style.RULES and ".pg-small .pg-mass" in style.RULES
        assert ('svg[data-portals="small"] .pg-wedge, .pg-small .pg-wedge {\n'
                '    display: inline; vector-effect: non-scaling-stroke; stroke-width: 1px; }'
                in style.RULES)
        assert _wrapped("PORTAL_NUMBER_LEAST") == 10

    def test_reduced_motion_stops_the_turn(self) -> None:
        from chess_variant_sandbox.web.page import style

        assert ".pg-idle { animation: pg-turn 14s linear infinite; }" in style.RULES
        assert ("@media (prefers-reduced-motion: reduce) { .pg-idle { animation: none; } }"
                in style.RULES)
        assert "pg-out" not in style.RULES  # nothing turns outwards: a way out has no arms
        assert _wrapped("PORTAL_TURN") == 14


class TestWhereItTurns:
    """Only a live board turns: the play board's own squares, the designer's bench and the
    offline replay; never a ghost copy, a still, a thumbnail or a key."""

    def test_the_play_board_turns_each_end_on_its_square_and_not_its_ghosts(self) -> None:
        from page_text import PAGE

        assert "idle: square !== undefined, flipped: FLIPPED," in PAGE
        assert "drawPiece(piece, x, y, cell, real ? square : undefined)" in PAGE

    def test_a_still_turns_only_on_the_designers_bench(self) -> None:
        from page_text import PAGE

        assert "idle: !!bench, label:" in PAGE
        assert "{ numbered, flipped: view.flipped })}</g>`" in PAGE

    def test_the_legend_key_stands_still(self) -> None:
        assert "pg-idle" not in _wrapped("markKey('portal')")  # type: ignore[operator]


class TestTheWords:
    """Q838c (`docs/archive/plan-portal-look.md` §5): an end is named by its pair's number and its
    square, its partner by its square, and the pair's colour once, as a friendly name."""

    def _words(self, end: str, partner: str, square: str = "e3", pairs: int = 1,
               there: str | None = "e4") -> Any:
        return _wrapped(f"portalWords({end}, {partner}, '{square}', {pairs},"
                        f" p => ['white', 'black'][p], {json.dumps(there)})")

    def test_a_one_way_door_says_who_goes_in_and_who_comes_out(self) -> None:
        said = self._words(_END % (1, "orange", "[0]", "[true, false]"),
                           _END % (1, "blue", "[1]", "[false, true]"))
        assert said["head"] == "The portal on e3"
        assert said["label"] == (
            "The portal on e3. White's pieces go in here and come out beside its partner on"
            " e4. Black's pieces come out beside this end, going in at its partner on e4."
            " Only white may move it, as a turn.")
        assert said["says"].endswith("one step beyond the other end, in the direction it was going.")

    def test_one_rule_for_both_sides_is_said_once_and_the_colour_once(self) -> None:
        said = self._words(_END % (2, "blue", "[0, 1]", "[true, true]"),
                           _END % (2, "orange", "[0, 1]", "[true, true]"), "b4", pairs=2,
                           there="e3")
        assert said["head"] == "Portal 2 (yellow) on b4"
        assert said["label"] == (
            "Portal 2 (yellow) on b4. Both sides' pieces go in here or at its partner on e3,"
            " and come out beside the other. Either side may move it, as its turn.")

    def test_each_pair_has_its_own_colour_word_and_none_is_an_old_ends(self) -> None:
        names = _wrapped("PORTAL_PAIR_NAMES")
        assert names == ["violet", "yellow", "cyan", "vermilion"]
        heads = [self._words(_END % (n, "orange", "[]", "[true, true]"),
                             _END % (n, "blue", "[]", "[true, true]"), pairs=4)["head"]
                 for n in (1, 2, 3, 4)]
        assert heads == [f"Portal {n} ({names[n - 1]}) on e3" for n in (1, 2, 3, 4)]  # type: ignore[index]

    def test_a_partner_off_the_board_is_still_its_partner(self) -> None:
        said = self._words(_END % (1, "orange", "[]", "[true, false]"),
                           _END % (1, "blue", "[]", "[false, false]"), there=None)
        assert "White's pieces go in here and come out beside its partner." in said["label"]

    def test_a_door_nobody_may_use_says_so_and_nothing_of_coming_out(self) -> None:
        said = self._words(_END % (1, "orange", "[]", "[false, false]"),
                           _END % (1, "blue", "[]", "[false, false]"))
        assert "Both sides' pieces cannot go through: to them it is a solid block." in said["label"]
        assert "Nobody moves it." in said["label"] and "comes out" not in said["says"]

    def test_the_partners_square_is_read_off_the_board(self) -> None:
        # Values as the core sends them: type + 1 for the first army, here three types.
        assert _wrapped("[portalSquareIn([0, 0, 3, 0, 2], 3, 2), portalSquareIn([0, 1], 3, 2),"
                        " portalSquareIn(null, 3, 2)]") == [2, -1, -1]

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
        assert "var(--pair-1)" in keys[1]  # type: ignore[index]

    def test_the_portals_row_says_how_to_read_it(self) -> None:
        row = _wrapped("MARK_KINDS.portal")
        assert row["name"] == "portal" and row["gloss"] == "portalKey"  # type: ignore[index]
        # Q843b: where a piece comes out is the end the beak points at (schema 21's maps).
        assert row["label"] == (  # type: ignore[index]
            "pieces of the arms' colours go in, and come out of the same-coloured portal its"
            " beak points at")

    def test_the_key_is_the_games_own_first_end(self) -> None:
        drawn = _wrapped("markKey('portal', {pair: 1, end: 'orange', movedBy: [0], open: [true, false]})")
        assert 'class="pg-token" data-moved="0"' in drawn  # type: ignore[operator]
        assert re.findall(r'class="pg-arm" data-arm="(\w+)"', drawn) == ["0", "0"]  # type: ignore[arg-type]

    def test_the_i_teaches_the_stand_and_the_arms_as_two_rows(self) -> None:
        """Q838f (F1): the cold read took the old key's one picture, white arms on a white token,
        for *white arms mean White's token*, and guessed at the stands it never drew. The key
        now draws what an end stands on as one row, every stand with the same arms, and its
        arms as another, every one on the same sunk well; then (Q843b, schema 21) the beak on a
        pair and on a funnel, and an end whose dashed arms let only some pieces in."""
        figure = _wrapped("portalKeyFigure(['white', 'black'])")
        heads = re.findall(r'<span class="portal-head">(.*?)</span>', figure)  # type: ignore[arg-type]
        assert heads == ["What it stands on: who may move it", "Its arms: whose pieces go in",
                         "Its colour, number and beak: where it leads",
                         "Dashed arms: only some pieces"]
        rows = figure.split('class="portal-head"')[1:]  # type: ignore[union-attr]
        examples = [row.split('class="portal-example"')[1:] for row in rows]
        said = [[re.search(r"<span>(.*?)</span>", one).group(1)  # type: ignore[union-attr]
                 for one in row] for row in examples]
        assert said == [
            ["On a white token: White moves it.", "On a black token: Black moves it.",
             "Half and half: either side moves it.", "Sunk into the board: nobody moves it."],
            ["White arms: White's pieces go in.", "Black arms: Black's pieces go in.",
             "One of each: both sides' pieces go in.", "No arms: nobody goes in. It is a way out."],
            ["The same colour and number are one group. The beak points where you come out:"
             " in at one, out beside the end it points at.",
             "Several beaks may point at one end: pieces come out beside it, and it has no arms,"
             " since nothing goes in there."],
            ["Dashed arms: only some pieces may go in. The words under the board say which, and"
             " when you pick a piece the doors it may not use go dim."],
        ]
        stands, arms, (pair, funnel), (filtered,) = examples
        # The stands: a white, a black and a half-and-half token, then a sunk well; the
        # same arms (one of each) on all four, so only the stand changes along the row.
        moved = [re.findall(r'data-moved="([^"]*)"', one) for one in stands]
        assert moved == [["0"], ["1"], ["0 1"], []]
        assert 'class="pg-hole"' in stands[3]
        assert all(_arms(one) == ["0", "1"] for one in stands)
        # The arms: White's, Black's, both and none, every one on a sunk well.
        assert [_arms(one) for one in arms] == [["0", "0"], ["1", "1"], ["0", "1"], []]
        assert all('class="pg-hole"' in one and "pg-token" not in one for one in arms)
        # One pair: two ends of one colour, numbered, their beaks at each other.
        assert pair.count('data-pair="2"') == 2 and pair.count(">2</text>") == 2
        assert re.findall(r'data-aim="([^"]+)"', pair) == ["0.0", "180.0"]
        # A funnel: two beaks at the third end, a way out with no arms and no beak.
        assert re.findall(r'data-aim="([^"]+)"', funnel) == ["0.0", "0.0"]
        assert funnel.count('class="pg-beak"') == 2 and _arms(funnel.split('data-portal="3c"')[1]) == []
        # A filtered end: its arms dashed.
        assert "data-filtered" in filtered and "data-filtered" not in "".join(stands + arms)
        assert "pg-idle" not in figure  # type: ignore[operator]

    def test_the_key_names_the_sides_as_the_ruleset_does(self) -> None:
        figure = _wrapped("portalKeyFigure(['first', 'second'])")
        for words in ("On a light token: First moves it.", "On a dark token: Second moves it.",
                      "Light arms: First's pieces go in.", "Dark arms: Second's pieces go in."):
            assert f"<span>{words}</span>" in figure  # type: ignore[operator]

    def test_every_end_on_the_librarys_portal_boards_is_in_the_key(self) -> None:
        """F1's bar: every combination of stand and arms on a portal board in the library can be
        read off the key, the stand from the first row and the arms from the second."""
        figure = _wrapped("portalKeyFigure(['white', 'black'])")
        stands, arms = figure.split('class="portal-head"')[1:3]  # type: ignore[union-attr]
        keyed_stands = {tuple(m.split()) for m in re.findall(r'data-moved="([^"]*)"', stands)}
        keyed_arms = {tuple(sorted(set(_arms(one)))) for one in arms.split('class="portal-example"')[1:]}
        from chess_variant_sandbox.viz.wrapped import portal_payload

        seen = 0
        for path in sorted((ROOT / "archive").glob("*.json")) + sorted((ROOT / "rulesets").glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            if not (spec.get("rules") or {}).get("portals"):
                continue
            payload = portal_payload(engine_for(spec).spec)
            assert payload, path.name
            seen += 1
            for end in payload["ends"]:
                movers = tuple(str(p) for p in end["movedBy"])
                assert not movers or movers in keyed_stands, (path.name, end)
                opened = tuple(sorted(str(p) for p in (0, 1) if end["open"][p]))
                assert opened in keyed_arms, (path.name, end)
        assert seen >= 7  # the seven archived portal games, at least (Q868 retired divide two way,
        # divide fixed and wormhole adrift)

    def test_the_legend_entry_carries_the_i(self) -> None:
        from page_text import PAGE

        assert "portalKey: ['How to read a portal'," in PAGE
        assert "{ figure: () => portalKeyFigure(" in PAGE
        assert "if (figure) return `<b>${esc(entry[0])}</b>${figure()}`;" in PAGE
        entries = _in_page(f"SESSION = {json.dumps(_SNAPSHOT)};",
                           "legendEntries(new Set(['portal']), null, '', SESSION.ruleset.portals)")
        (entry,) = entries
        assert 'data-gloss="portalKey"' in entry and 'data-legend="portal"' in entry
        assert "same-coloured portal" in entry

    def test_the_board_names_them_where_it_draws_them(self) -> None:
        assert _wrapped("[boardKinds([60], {ends: []}), boardKinds([], null), boardKinds(null, null)]"
                        ) == [["wall", "portal"], [], []]
        # And no square mark is ever made of either.
        assert _wrapped("Object.values(squareMarks({goals: [[4], [4]], zones: [[0], [56]]}))"
                        ".flat().map(m => m.kind).sort()") == ["goal", "zone", "zone"]


class TestTheTokens:
    def test_each_pair_holds_against_its_edge_and_the_edge_against_both_woods(self) -> None:
        for dark in (False, True):
            theme = _theme(dark)
            edge = _rgb(theme["portal-edge"])
            for n in range(1, 5):
                assert _ratio(_rgb(theme[f"pair-{n}"]), edge) >= 3.0, (dark, n)
            for square in ("sq-lt", "sq-dk"):
                assert _ratio(edge, _grained(dark, square)) >= 3.0, (dark, square)

    def test_the_pairs_are_set_once_for_both_themes(self) -> None:
        for name in ("pair-1", "pair-2", "pair-3", "pair-4", "portal-edge", "portal-stone"):
            assert name in tokens.LIGHT
        assert tokens.DARK["portal-stone"] != tokens.LIGHT["portal-stone"]
        for n in range(1, 5):
            assert (f"pair-{n}", "portal-edge", 3.0) in tokens.FLOORS

    def test_the_scripts_draw_a_portal_only_through_the_tokens(self) -> None:
        from chess_variant_sandbox.web.page import SCRIPTS

        for path in SCRIPTS.values():
            text = path.read_text(encoding="utf-8")
            for literal in ("#e8590c", "#1c7ed6", "#2a86e0", "#1e140c", "#7b4fd6", "#efc12b",
                            "#22b8cf", "#d55e00", "#f4f2ee"):
                assert literal not in text, (path.name, literal)

    def test_the_old_ends_colours_are_gone_from_the_board(self) -> None:
        """`--portal-a` and `--portal-b` are gone from the whole package since the designer's
        way buttons drew the pair's colour (Q838d), and from the tokens with them."""
        from chess_variant_sandbox.web.page import tokens

        assert "portal-a" not in tokens.LIGHT and "portal-b" not in tokens.LIGHT
        package = ROOT / "chess_variant_sandbox"
        for path in package.rglob("*"):
            if path.suffix not in (".py", ".js", ".css", ".html"):
                continue
            text = path.read_text(encoding="utf-8")
            for word in ("portal-a", "portal-b", "pg-double"):
                assert word not in text, (path.relative_to(ROOT), word)


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
        assert html.startswith("<b>The portal on d5</b>. Both sides&#39; pieces go in here or at"
                               " its partner on d4, and come out beside the other.")
        assert "Nobody moves it." in html


# ------------------------------------------------- the words by number and square (Q838c)

def _ferry_session(name: str = "ferry") -> Session:
    """Ferry (two pairs: 1 from b3 to e4, 2 from e3 to b4, both moved by either side) at the
    start, both seats a person."""
    spec = _archived(name)
    return Session(id="f1", game=_game(spec), seats=("human", "human"), seed=0,
                   state=engine_for(spec).initial_state())


class TestTheRoutesWords:
    """The route chooser names a portal by its number and square, never by its colour."""

    def test_with_two_pairs_it_says_the_number(self) -> None:
        snapshot = _ferry_session().snapshot()
        said = _in_page(f"SESSION = {json.dumps(snapshot)};\n" + _HARNESS,
                        f"[routeWords({{portal: {_square('e3', 6)}, portalName: 'e3'}}),"
                        f" routeWords({{portal: {_square('b3', 6)}, portalName: 'b3'}}),"
                        " routeWords({portalName: 'b3'})]")
        assert said == ["through portal 2 on e3", "through portal 1 on b3", "directly"]

    def test_with_one_pair_it_says_the_portal(self) -> None:
        said = _in_page(f"SESSION = {json.dumps(_SNAPSHOT)};\n" + _HARNESS,
                        f"routeWords({{portal: {_square('d4')}, portalName: 'd4'}})")
        assert said == "through the portal on d4"

    def test_an_end_names_its_partner_by_the_square_it_stands_on(self) -> None:
        snapshot = _ferry_session().snapshot()
        said = _in_page(f"SESSION = {json.dumps(snapshot)};\n" + _HARNESS,
                        f"portalOn({_square('e3', 6)}).words.label")
        assert said == (
            "Portal 2 (yellow) on e3. Both sides' pieces go in here or at its partner on b4,"
            " and come out beside the other. Either side may move it, as its turn.")


class TestNoStraightBack:
    """Schema 20: a portal that has just moved says why it may not go straight back (§5)."""

    def _moved(self) -> tuple[Session, int, int]:
        session = _ferry_session("ferry-no-straight-back")
        b3 = _square("b3", 6)
        move = next(m for m in session.snapshot()["moves"] if m["from"] == b3)
        session.play(move["move"])
        return session, b3, move["to"]

    def test_the_snapshot_carries_the_cores_bar(self) -> None:
        session, b3, standing = self._moved()
        assert session.snapshot()["portalBar"] == {"standing": standing, "left": b3}
        assert _ferry_session().snapshot()["portalBar"] is None
        assert _SNAPSHOT["portalBar"] is None

    def test_a_tap_on_the_moved_end_says_so_in_the_status_line(self) -> None:
        session, b3, standing = self._moved()
        snapshot = session.snapshot()
        # The engine left the move straight back out; the words only say so.
        assert not [m for m in snapshot["moves"] if m["from"] == standing and m["to"] == b3]
        said = _in_page(f"SESSION = {json.dumps(snapshot)};\n" + _HARNESS,
                        f"(() => {{ clickSquare({standing}); const moved = NOTE;"
                        f" clickSquare({_square('e3', 6)}); return [moved, NOTE]; }})()")
        assert said == ["Portal 1 has just moved and may not go straight back to b3 this turn", ""]

    def test_the_words_need_the_bar_on_that_end(self) -> None:
        end = "{pair: 1, end: 'orange', movedBy: [0, 1], open: [true, true]}"
        said = _wrapped(f"[portalBarWords({{standing: 9, left: 3}}, 9, {end}, 1, s => 'sq' + s),"
                        f" portalBarWords({{standing: 9, left: 3}}, 8, {end}, 1, s => 'sq' + s),"
                        f" portalBarWords(null, 9, {end}, 1, s => 'sq' + s)]")
        assert said == [
            "The portal has just moved and may not go straight back to sq3 this turn", "", ""]


class TestNoColourWords:
    """Q838c's acceptance: no user-visible string in the page's scripts, `viz/wrapped.js`,
    `spec/describe.py` or `spec/plain.py` names a portal's end by *orange* or *blue*. The
    payload's and the file's names for the two ends (`end: 'orange'`, `rules.portals.blue`)
    are data, not words. The designer's portal editor (`page/portals.js`) reads those names
    as data on nearly every line, so it is held by what it renders and says instead
    (`tests/test_designer_portals.py::TestNoColourWordInTheDesigner`, Q838d)."""

    _SEAM = "name: 'blue line'"  # the one-way seam's legend name: a seam, not a portal

    def test_the_scripts_say_no_colour_for_an_end(self) -> None:
        package = ROOT / "chess_variant_sandbox"
        paths = sorted((package / "web" / "static" / "page").glob("*.js"))
        paths.append(package / "viz" / "wrapped.js")
        for path in paths:
            if path.name == "portals.js":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                code = "" if line.lstrip().startswith(("//", "*", "/*")) else line
                code = code.replace(self._SEAM, "")
                for found in re.finditer(r"(?i)orange|blue", code):
                    quoted = (code[found.start() - 1:found.start()] == "'"
                              and code[found.end():found.end() + 1] == "'")
                    assert quoted, (path.name, number, line.strip())

    def test_the_rules_text_names_ends_by_square(self) -> None:
        from chess_variant_sandbox.spec.describe import describe_spec

        for path in sorted((ROOT / "archive").glob("*.json")):
            spec = json.loads(path.read_text(encoding="utf-8"))
            if not (spec.get("rules") or {}).get("portals"):
                continue
            assert not re.search(r"(?i)orange|blue", describe_spec(spec)), path.name
            assert not re.search(r"(?i)orange|blue", spec.get("description", "")), path.name
        divide = describe_spec(_archived("divide"))
        assert ("Portal pair 1 joins e3 and e9: the end on e3 may be moved by the first player;"
                " the end on e9 may be moved by the second player; the first player's pieces may"
                " go in only at e3, and the second player's pieces may go in only at e9.") in divide

    def test_the_fields_are_first_and_second_ends(self) -> None:
        from chess_variant_sandbox.spec.plain import FIELD_WORDS

        portal = {k: v for k, v in FIELD_WORDS.items() if k.startswith("rules.portals")}
        assert portal["rules.portals.orange"][0] == "first end on"
        assert portal["rules.portals.blue"][0] == "second end on"
        for label, sentence in portal.values():
            assert not re.search(r"(?i)orange|blue", label + (sentence or "")), label
