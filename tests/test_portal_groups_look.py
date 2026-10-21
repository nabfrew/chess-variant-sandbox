"""Schema 21's look and designer (Q843b, `docs/archive/plan-portals-2.md` §5.1, §5.2, §5.4).

What can be held without a browser. On the board: where a beak points, a pure function of
two drawn positions (as `portalJourney` is a pure function of a distance), and the nearest
drawn copy round a seam; the beak's three sizes; a filtered end's dashed arms, and the end a
picked piece may not go in by dimmed; a group turning in step; the group map's arrows; the
arms by mass on a phone and the half-and-half token's arms set against its collar (Q838h,
`docs/reference/portal-look-cold-read.md` §7.5). In the designer: a group put down with
*+ portals* and *done* (or Enter), the three named shapes and *your own map*, each end's
*leads to* row, the filter chooser, a pair always written in the pair spelling, and D7's new
refusals in words. The look itself is the look check's `beak`, `group map` and `filter` rows
(`docs/reference/look-check.md` *After Q843b*).
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

import pytest
from test_designer_portals import _GAME, _clicks, _page, _with
from test_designer_portals import _square as _square6
from test_fold_look import STYLES_OF
from test_portal_board import _game
from test_schema_twenty_one import filtered, funnel, ring
from test_viz import _wrapped
from test_web import _in_page

from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.web import designing
from chess_variant_sandbox.web.library import engine_for
from chess_variant_sandbox.web.play import Session


def _snapshot(spec: dict[str, Any]) -> dict[str, Any]:
    engine = engine_for(spec)
    session = Session(id="g1", game=_game(spec), seats=("human", "human"), seed=0,
                      state=engine.initial_state())
    return session.snapshot()


def _square(name: str, width: int) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


_END = "({pair: 1, end: 'orange', movedBy: %s, open: %s%s})"


def _glyph(movers: str = "[]", open_: str = "[true, true]", more: str = "", **options: str) -> str:
    extra = "".join(f", {k}: {v}" for k, v in options.items())
    return _wrapped(f"portalGlyph({_END % (movers, open_, more)}, 0, 0, 10, {{ numbered: false{extra} }})")  # type: ignore[return-value]


# ------------------------------------------------------------------ the beak (§5.1)

class TestWhereTheBeakPoints:
    """The aim is the angle on screen from the centre of the end as drawn to the centre of
    the end it leads to as drawn: nothing else goes into it."""

    @pytest.mark.parametrize(("to", "degrees"), [
        ((10, 0), 0), ((0, 10), 90), ((-10, 0), 180), ((0, -10), -90), ((7, 7), 45)])
    def test_it_is_the_angle_from_one_drawn_centre_to_the_other(self, to: tuple[int, int],
                                                                degrees: float) -> None:
        aim = _wrapped(f"portalAim(0, 0, {to[0]}, {to[1]})")
        assert math.isclose(aim, math.radians(degrees), abs_tol=1e-9)  # type: ignore[arg-type]

    def test_it_points_nowhere_from_one_spot_to_the_same(self) -> None:
        assert _wrapped("portalAim(3, 4, 3, 4)") is None

    def test_round_a_seam_it_points_at_the_nearest_copy(self) -> None:
        """Cylinder chess: an end on a4 that leads to h4 points left, across the seam, at the
        ghost of h4 drawn beside it, not right across the whole board."""
        found = _wrapped("""(() => {
          const view = boardView(8, 8, { files: true, ranks: false }, null, false);
          const end = { pair: 1, end: 'orange', movedBy: [], open: [true, true] };
          const [x, y] = squareXY(24, 10, view);
          const flat = flatView(8, 8, false);
          const [fx, fy] = squareXY(24, 10, flat);
          return [portalAimFor(end, 31, x, y, 10, view), portalAimFor(end, 31, fx, fy, 10, flat)];
        })()""")
        assert math.isclose(found[0], math.pi)  # type: ignore[index]
        assert math.isclose(found[1], 0, abs_tol=1e-9)  # type: ignore[index]

    def test_a_way_out_has_no_beak(self) -> None:
        assert _wrapped("portalAimFor({pair: 1, end: 'a', movedBy: [], open: [false, false]},"
                        " 3, 0, 0, 10, flatView(8, 8, false))") is None
        drawn = _glyph(open_="[false, false]", aim="0")
        assert "pg-beak" not in drawn and "data-aim" not in drawn


class TestTheBeaksSizes:
    def test_it_is_drawn_outside_the_well_and_as_a_wedge_cut_into_the_rim(self) -> None:
        drawn = _glyph(aim="Math.PI / 2")
        assert drawn.count('class="pg-beak"') == 1 and drawn.count('class="pg-beak-cut"') == 1
        assert 'data-aim="90.0"' in drawn
        beak = re.search(r'class="pg-beak" d="([^"]+)" fill="var\(--pair-1\)" stroke="var\(--portal-edge\)"',
                         drawn)
        assert beak
        points = [tuple(map(float, xy.split())) for xy in re.split(r"[MLZ]", beak.group(1)) if xy.strip()]
        tip = points[0]
        # Pointing down (90 degrees): its tip below the centre, beyond the well's rim.
        assert math.isclose(tip[0], 5, abs_tol=1e-3) and tip[1] > 5 + 3.8
        # The outside beak comes before the well (its base under the rim), the cut one after.
        assert drawn.index("pg-beak") < drawn.index('class="pg-well"') < drawn.index("pg-beak-cut")

    def test_no_aim_no_beak_so_a_still_without_one_draws_as_before(self) -> None:
        assert "pg-beak" not in _glyph() and "data-aim" not in _glyph()

    def test_the_stylesheet_shows_one_size_and_none_under_twenty_px(self) -> None:
        from chess_variant_sandbox.web.page import style

        assert ".pg-wedge, .pg-mass, .pg-beak-cut { display: none; }" in style.RULES
        assert 'svg[data-portals="mid"] .pg-beak { display: none; }' in style.RULES
        assert 'svg[data-portals="mid"] .pg-beak-cut {' in style.RULES
        assert 'svg[data-portals="small"] .pg-beak, .pg-small .pg-beak { display: none; }' in style.RULES


# ------------------------------------------------------------------ filters (§5.2)

class TestFilteredArms:
    def test_a_filtered_ends_arms_are_three_strokes_each(self) -> None:
        dashed = _glyph(more=", filtered: true")
        plain = _glyph()
        assert "data-filtered" in dashed and "data-filtered" not in plain
        arms = re.findall(r'class="pg-arm" data-arm="\w" d="([^"]+)"', dashed)
        assert len(arms) == 2 and all(arm.count("M") == 3 for arm in arms)
        assert all(arm.count("M") == 1 for arm in re.findall(r'class="pg-arm" data-arm="\w" d="([^"]+)"', plain))

    def test_a_dimmed_end_dims_its_arms_and_its_beak(self) -> None:
        drawn = _glyph(aim="0", dim="true")
        assert "data-dim" in drawn
        assert re.search(r'class="pg-arms"[^>]*opacity="0.35"', drawn)
        assert re.search(r'class="pg-beak"[^>]*opacity="0.35"', drawn)
        assert "data-dim" not in _glyph(aim="0")


# ------------------------------------------------------------------ Q838h

class TestArmsAPhoneCanRead:
    def test_the_mass_arms_are_wider_and_a_white_one_is_edged_in_the_groups_colour(self) -> None:
        drawn = _glyph()
        mass = re.findall(r'class="pg-mass" data-arm="(\w)" d="[^"]+" fill="([^"]+)" stroke="([^"]+)"', drawn)
        assert mass == [("0", "var(--p0)", "var(--pair-1)"), ("1", "var(--p1)", "var(--pair-1)")]
        assert _wrapped("PORTAL_MASS_WIDTH") > _wrapped("PORTAL_ARM_WIDTH")  # type: ignore[operator]

    def test_on_a_half_and_half_token_each_arm_starts_by_the_other_colours_half(self) -> None:
        """The arm's outer end is where it meets the collar. Sunk, the bottom player's arm
        starts at the bottom; on a half-and-half token (whose bottom half is the bottom
        player's colour) it starts at the top, beside the other colour."""
        def start_y(drawn: str) -> float:
            path = re.search(r'class="pg-arm" data-arm="0" d="M[\d.]+ ([\d.]+)', drawn)
            assert path
            return float(path.group(1))
        assert start_y(_glyph()) > 5
        assert start_y(_glyph(movers="[0, 1]")) < 5
        assert start_y(_glyph(movers="[0]")) > 5  # one colour's token: as before

    def test_a_phone_draws_the_mass_arms(self) -> None:
        from page_text import PAGE

        assert "if (portalPhone()) svg.setAttribute('data-phone', '');" in PAGE
        assert _wrapped("PORTAL_PHONE") == 480


# ------------------------------------------------------------------ on the play board

_BOARD = """
FRAME = boardView(SESSION.ruleset.width, SESSION.ruleset.height, wrapAxes(SESSION.ruleset.wrap),
                  SHIFT, FLIPPED, SESSION.ruleset.lattice, boardCut(SESSION.absent, SESSION.ruleset.wrap));
const CELL = 100 / Math.max(SESSION.ruleset.width, SESSION.ruleset.height);
const drawnAt = sq => { const [x, y] = squareXY(sq, CELL, FRAME); return drawPiece(pieceAt(sq), x, y, CELL, sq); };
const aimAt = sq => { const m = /data-aim="([^"]+)"/.exec(drawnAt(sq)); return m ? Number(m[1]) : null; };
let PAINTED = '';
document.createElementNS = () => ({ setAttribute() {}, set innerHTML(v) { PAINTED = v; } });
const BOARD_SVG = { querySelector: () => null, appendChild() {} };
const byId0 = document.getElementById;
document.getElementById = id => id === 'board' ? BOARD_SVG : byId0(id);
const ways = () => { PAINTED = ''; paintLinks(); return [...PAINTED.matchAll(/data-way="([^"]+)"/g)].map(m => m[1]); };
"""


def _board(spec: dict[str, Any], steps: str, expression: str) -> Any:
    return _in_page(f"SESSION = {json.dumps(_snapshot(spec))};\n" + _BOARD + steps, expression)


class TestOnThePlayBoard:
    def test_a_rings_beaks_chase_each_other_round(self) -> None:
        a3, c3, e3 = (_square(name, 5) for name in ("a3", "c3", "e3"))
        assert _board(ring(), "", f"[aimAt({a3}), aimAt({c3}), aimAt({e3})]") == [0, 0, 180]

    def test_a_funnels_beaks_all_point_at_its_way_out_which_has_none(self) -> None:
        b3, e4, c4 = (_square(name, 6) for name in ("b3", "e4", "c4"))
        aims = _board(funnel(), "", f"[aimAt({b3}), aimAt({e4}), aimAt({c4})]")
        assert math.isclose(aims[0], math.degrees(math.atan2(-1, 1)), abs_tol=0.1)  # up and right
        assert aims[1] == 180 and aims[2] is None

    def test_a_pairs_two_beaks_point_at_each_other(self) -> None:
        b3, e4 = (_square(name, 6) for name in ("b3", "e4"))
        aims = _board(filtered(), "", f"[aimAt({b3}), aimAt({e4})]")
        assert math.isclose(aims[0], math.degrees(math.atan2(-1, 3)), abs_tol=0.1)
        assert math.isclose(aims[1], math.degrees(math.atan2(1, -3)), abs_tol=0.1)

    def test_a_picked_piece_dims_the_doors_it_may_not_use(self) -> None:
        """Knights only at b3, rooks and pawns at e4: a pawn picked dims b3 and not e4; a
        knight the other way round."""
        b3, e4 = (_square(name, 6) for name in ("b3", "e4"))
        pawn, knight = _square("c2", 6), _square("b1", 6)
        dims = _board(filtered(), "", f"""[{pawn}, {knight}, null].map(sq => {{ SELECTED = sq;
          return [portalRefuses(portalOn({b3}).end), portalRefuses(portalOn({e4}).end),
                  /data-dim/.test(drawnAt({b3}))]; }})""")
        assert dims == [[True, False, True], [False, True, False], [False, False, False]]

    def test_a_game_with_no_filter_dims_nothing(self) -> None:
        a3 = _square("a3", 5)
        assert _board(ring(), f"SELECTED = {_square('b2', 5)};",
                      f"/data-dim/.test(drawnAt({a3}))") is False

    def test_the_map_draws_an_arrow_from_every_end_somebody_goes_in_by(self) -> None:
        found = _board(funnel(), "", "[ways(), (LINKS = true, ways())]")
        assert found[0] == [] and len(found[1]) == 2

    def test_a_tapped_end_shows_its_way_out_and_every_way_in(self) -> None:
        c3 = _square("c3", 5)
        found = _board(ring(), f"INSPECT = {c3};", "ways()")
        assert len(found) == 2  # c3 to e3, and a3 to c3

    def test_a_group_turns_in_step(self) -> None:
        delays = _board(ring(), "", f"""[{_square('a3', 5)}, {_square('c3', 5)}, {_square('e3', 5)}]
          .map(sq => /animation-delay:([-\\d.]+)s/.exec({STYLES_OF}(drawnAt(sq)))[1])""")
        assert len(set(delays)) == 1

    def test_the_words_name_the_filters_pieces(self) -> None:
        said = _board(filtered(), "", f"portalOn({_square('e4', 6)}).words.label")
        assert said.endswith("Only pawns and rooks may go in here.")


# ------------------------------------------------------------------ the designer (§5.4)

def _drafted(steps: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """The draft after `steps`, as the page wrote it."""
    return _page(spec or copy.deepcopy(_GAME), steps, "DRAFT")  # type: ignore[no-any-return]


def _valid(draft: dict[str, Any]) -> None:
    answer = designing.check(draft)
    assert answer["valid"], answer["errors"]
    assert validate_spec(designing.settled(draft)) == []


class TestTheDesignerWritesGroups:
    def test_three_clicks_and_done_make_a_ring_that_validates(self) -> None:
        draft = _drafted("startPortalTool();" + _clicks("b2", "e2", "c6") + "finishPortalTool();")
        (group,) = draft["rules"]["portals"]
        assert group == {"piece": "O", "ends": [{"at": "b2", "to": 1}, {"at": "e2", "to": 2},
                                                 {"at": "c6", "to": 0}]}
        _valid(draft)

    def test_a_funnel_validates_and_its_way_out_leads_nowhere(self) -> None:
        draft = _drafted("startPortalTool();" + _clicks("b2", "e2", "c6")
                         + "finishPortalTool(); setShape(0, 'funnel');")
        (group,) = draft["rules"]["portals"]
        assert group["ends"] == [{"at": "b2", "to": 2}, {"at": "e2", "to": 2}, {"at": "c6"}]
        _valid(draft)

    def test_two_ends_are_always_written_as_a_pair(self) -> None:
        """A pair is the one spelling of a group of two (Q843a's *one game, one hash*): made a
        funnel, it is a pair whose way is one way for both sides."""
        pair = _drafted("startPortalTool();" + _clicks("b2", "e6") + "finishPortalTool();")
        assert pair["rules"]["portals"] == [{"piece": "O", "orange": "b2", "blue": "e6"}]
        _valid(pair)
        one_way = _drafted("setShape(0, 'funnel');", _with({"orange": "b2", "blue": "e6"}))
        assert one_way["rules"]["portals"] == [{"piece": "O", "orange": "b2", "blue": "e6", "passage": {
            "0": "orange_to_blue", "1": "orange_to_blue"}}]
        _valid(one_way)
        back = _drafted("setShape(0, 'funnel'); setShape(0, 'pair');", _with({"orange": "b2", "blue": "e6"}))
        assert back["rules"]["portals"] == [{"piece": "O", "orange": "b2", "blue": "e6"}]

    def test_a_row_changed_makes_your_own_map(self) -> None:
        spec = _with({"ends": [{"at": "b2", "to": 1}, {"at": "e2", "to": 2}, {"at": "c6", "to": 0}]})
        shape, html, draft = _page(spec, "setLeadsTo(0, 2, 1);",
                                   "[groupShape(groupEnds(DRAFT.rules.portals[0])),"
                                   " pairCardHtml(DRAFT.rules.portals[0], 0, 1), DRAFT]")
        assert shape == "own" and "own map" in html
        assert draft["rules"]["portals"][0]["ends"][2] == {"at": "c6", "to": 1}
        _valid(draft)

    def test_the_card_offers_the_shapes_that_fit_and_a_row_per_end(self) -> None:
        ring_spec = _with({"ends": [{"at": "b2", "to": 1}, {"at": "e2", "to": 2}, {"at": "c6", "to": 0}]})
        html = _page(ring_spec, "", "portalCardsHtml([])", designing.check(ring_spec))
        assert 'data-pshape="0.ring"' in html and 'data-pshape="0.funnel"' in html
        assert 'data-pshape="0.pair"' not in html
        assert 'data-pshape="0.ring" class="wayChoice shapeChoice on"' in html
        for name in ("b2", "e2", "c6"):
            assert f"The end on {name} leads to" in html
        assert "nowhere: it is a way out" in html
        pair = _with({"orange": "b2", "blue": "e6"})
        html = _page(pair, "", "portalCardsHtml([])", designing.check(pair))
        assert 'data-pshape="0.pair"' in html and 'data-pshape="0.ring"' not in html
        assert "The end on b2 leads to" in html and "The end on e6 leads to" in html

    def test_whose_pieces_go_in_and_who_moves_a_maps_end(self) -> None:
        spec = _with({"ends": [{"at": "b2", "to": 2}, {"at": "e2", "to": 2}, {"at": "c6"}]},
                     moves=[designing.PORTAL_STEP])
        draft = _drafted("setEnter(0, 1, [0]); setMovers(0, 2, [1]);", spec)
        assert draft["rules"]["portals"][0]["ends"] == [
            {"at": "b2", "to": 2}, {"at": "e2", "to": 2, "enter": [0]}, {"at": "c6", "moved_by": [1]}]
        _valid(draft)

    def test_the_filter_chooser_starts_full_and_writes_what_is_ticked(self) -> None:
        pair = _with({"orange": "b2", "blue": "e6"})
        full = _drafted("setFilter(0, 0, 'choose');", pair)
        assert full["rules"]["portals"][0]["carries"] == {"orange": ["K", "R"]}
        refused = designing.check(full)["errors"]
        assert "The end on b2 lets every piece in, so its filter keeps nothing out. Untick the " \
               "pieces it keeps out, or choose all." in refused
        rooks = _drafted("setFilter(0, 0, 'choose'); toggleCarried(0, 0, 'K');", pair)
        assert rooks["rules"]["portals"][0]["carries"] == {"orange": ["R"]}
        _valid(rooks)
        html = _page(pair, "setFilter(0, 0, 'choose'); toggleCarried(0, 0, 'K');",
                     "pairCardHtml(DRAFT.rules.portals[0], 0, 1)")
        assert "Only rooks may go in here." in html
        assert 'data-pcarry="0.0.R" class="filterPiece on"' in html
        assert 'data-pcarry="0.0.K" class="filterPiece"' in html
        cleared = _drafted("setFilter(0, 0, 'choose'); setFilter(0, 0, 'all');", pair)
        assert "carries" not in cleared["rules"]["portals"][0]

    def test_a_way_out_keeps_no_filter(self) -> None:
        spec = _with({"ends": [{"at": "b2", "to": 1, "carries": ["R"]}, {"at": "e2", "to": 2},
                               {"at": "c6", "to": 0}]})
        draft = _drafted("setLeadsTo(0, 0, null);", spec)
        assert draft["rules"]["portals"][0]["ends"][0] == {"at": "b2"}

    def test_a_kind_and_a_bar_are_kept_through_a_change(self) -> None:
        spec = _with({"ends": [{"at": "b2", "to": 1, "moved_by": [0]}, {"at": "e2", "to": 2},
                               {"at": "c6", "to": 0}], "no_straight_back": True},
                     moves=[designing.PORTAL_STEP])
        draft = _drafted("setShape(0, 'funnel');", spec)
        assert draft["rules"]["portals"][0]["no_straight_back"] is True


class TestTheKeyboardPutsAGroupDown:
    def test_arrows_space_and_enter_make_a_ring_and_its_shape_is_a_button(self) -> None:
        """D9: the arrows walk the board and space puts an end down, three times; Enter is
        *done*. The card's shapes are buttons, reached and pressed by the keyboard as any
        button is."""
        walk = (f"startPortalTool(); BENCH_FOCUS = {_square6('b2')}; benchKey(' ');"
                f" BENCH_FOCUS = {_square6('b6')}; benchKey(' '); benchKey('ArrowRight');"
                " benchKey('ArrowRight'); benchKey(' '); benchKey('Enter');")
        draft, html = _page(copy.deepcopy(_GAME), walk + " const made = JSON.parse(JSON.stringify(DRAFT));"
                            " setShape(0, 'funnel');",
                            "[made, portalCardsHtml([])]")
        assert draft["rules"]["portals"] == [{"piece": "O", "ends": [
            {"at": "b2", "to": 1}, {"at": "b6", "to": 2}, {"at": "d6", "to": 0}]}]
        assert re.search(r'<button data-pshape="0.funnel" class="wayChoice shapeChoice on"', html)

    def test_enter_with_one_end_down_puts_an_end_down_not_the_group(self) -> None:
        tool, groups = _page(copy.deepcopy(_GAME), f"startPortalTool(); BENCH_FOCUS = {_square6('b2')};"
                             " benchKey('Enter');", "[PORTAL_TOOL, (DRAFT.rules.portals || []).length]")
        assert tool == {"squares": [_square6("b2")]} and groups == 0

    def test_an_end_clicked_again_is_taken_back_up(self) -> None:
        tool, said = _page(copy.deepcopy(_GAME), "startPortalTool();" + _clicks("b2", "e6", "b2"),
                           "[PORTAL_TOOL, BENCH_SAID]")
        assert tool == {"squares": [_square6("e6")]}
        assert said.startswith("Took the end on b2 back up.")


# ------------------------------------------------------------------ D7's new rows

_GROUP_REFUSALS = [
    ("an end nobody can use",
     {"ends": [{"at": "b2", "to": 1}, {"at": "e2", "to": 0}, {"at": "c6"}]},
     "Group 1's end on c6 leads nowhere and no end leads to it, so no piece can use it. Lead it "
     "to another end, or lead another end to it."),
    ("a destination that is itself",
     {"ends": [{"at": "b2", "to": 0}, {"at": "e2", "to": 0}, {"at": "c6", "to": 1}]},
     "Group 1's end on b2 leads to itself. Choose another end for it to lead to."),
    ("a filter that names every piece",
     {"orange": "b2", "blue": "e6", "carries": {"blue": ["K", "R"]}},
     "The end on e6 lets every piece in, so its filter keeps nothing out. Untick the pieces it "
     "keeps out, or choose all."),
    ("an empty filter",
     {"ends": [{"at": "b2", "to": 1, "carries": []}, {"at": "e2", "to": 2}, {"at": "c6", "to": 0}]},
     "The end on b2 has no piece ticked, so nothing could go in by it. Tick the pieces it lets "
     "in, or choose all."),
    ("two ends of a group on one square",
     {"ends": [{"at": "b2", "to": 1}, {"at": "b2", "to": 2}, {"at": "c6", "to": 0}]},
     "Group 1 has two ends on b2. Each end goes on a square of its own."),
]


class TestTheNewRefusalsInWords:
    @pytest.mark.parametrize(("why", "group", "says"), _GROUP_REFUSALS,
                             ids=[row[0] for row in _GROUP_REFUSALS])
    def test_each_reads_as_the_designer_says_it(self, why: str, group: dict[str, Any],
                                                says: str) -> None:
        answer = designing.check(_with(group))
        assert not answer["valid"]
        assert says in answer["errors"], answer["errors"]
        for line in answer["errors"]:
            assert "rules.portals" not in line and "carries" not in line, line

    def test_more_than_eight_ends(self) -> None:
        spec = _with({"ends": [{"at": at, "to": (i + 1) % 5} for i, at in
                               enumerate(("a2", "b2", "c2", "d2", "e2"))]},
                     {"ends": [{"at": at, "to": (i + 1) % 4} for i, at in
                               enumerate(("a6", "b6", "c6", "d6"))]})
        assert "A game can have at most eight portals." in designing.check(spec)["errors"]

    def test_the_end_at_fault_is_named_for_show_me(self) -> None:
        errors = designing.check(_with(_GROUP_REFUSALS[0][1]))["portals"]["errors"]
        assert {"says": _GROUP_REFUSALS[0][2], "pair": 1, "end": "ends[2]", "square": "c6",
                "place": 2} in errors


class TestTheMapsCard:
    def test_the_sentences_say_where_each_end_leads_and_where_pieces_come_out(self) -> None:
        spec = _with({"ends": [{"at": "b2", "to": 2}, {"at": "e2", "to": 2, "enter": [0],
                                "carries": ["R"]}, {"at": "c6"}]})
        (card,) = designing.check(spec)["portals"]["cards"]
        assert card["shape"] == "funnel" and card["spelling"] == "ends"
        assert card["ways"] == [
            "White's and Black's pieces can go in at the end on b2 and come out by the end on c6.",
            "White's pieces can go in at the end on e2 and come out by the end on c6.",
            "Nothing goes in at the end on c6: pieces come out by it from b2 and e2."]
        assert card["filters"] == ["Only rooks may go in at the end on e2."]
        assert card["moves"] == "Nobody moves these portals."
        assert "The end on e2 is solid to Black's pieces" in " ".join(card["solid"])
