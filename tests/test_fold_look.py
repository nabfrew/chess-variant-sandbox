"""Schema 22's look and designer (Q843d, `docs/plan-portals-2.md` §5.3, §5.4), revised for
schema 24's rules (Q867b, §3R) and its look (Q867c, §5.3R).

What can be held without a browser. The pool (`foldGlyph`): bright in the middle where the
vortex is dark, so the two kinds are told apart in a grey picture; its bands, whose edges are
whose pieces may go through (plain and turning outward where it leads nowhere, dashed where
it is filtered); its collar, who may move it; its beak; its number, moved to a badge when a
piece stands in it; its rim's turbulence; its idle turn in step across a group. On the play
board: the pool drawn under the piece standing in it, nothing drawn on the far end (the echo
went, §5.3R), the glow over the squares, the ordinary dots for staying and for going through
and a tap on each playing it, the piece-or-portal chooser in the user's words, the misplay
it now catches, the words, the legend's row and its (i). The server: the fold's transit
(`viz.wrapped.fold_transit`), the still's folds and the move map's fold transits, the group's
card in a fold's words. The designer's *how it works* choice. The ripple trip is
`test_portal_travel.py`'s; the look itself is the look check's `pool` rows
(`docs/reference/look-check.md` *After Q843d*).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from test_designer_portals import _GAME, _page
from test_portal_board import _game
from test_schema_twenty_four import _gardner_folds
from test_viz import _wrapped
from test_web import _in_page

from chess_variant_sandbox.web import designing
from chess_variant_sandbox.web.library import engine_for
from chess_variant_sandbox.web.play import Session


def _square(name: str, width: int = 5) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


_FOLD = "({pair: %d, end: 'orange', kind: 'fold', dest: 9, type: 8, movedBy: %s, open: %s%s})"


def _pool(movers: str = "[]", open_: str = "[true, true]", more: str = "", pair: int = 1,
          **options: str) -> str:
    extra = ", ".join(f"{k}: {v}" for k, v in options.items())
    return _wrapped(f"foldGlyph({_FOLD % (pair, movers, open_, more)}, 0, 0, 10, {{ {extra} }})")  # type: ignore[return-value]


# ------------------------------------------------------------------ the pool (§5.3)

class TestThePool:
    def test_it_is_bright_in_the_middle_where_the_vortex_is_dark(self) -> None:
        """The two kinds told apart in grey: the pool's centre is `--fold-light`, the
        vortex's eye the keyline's near-black."""
        pool = _pool()
        vortex = _wrapped("portalGlyph({pair: 1, end: 'orange', movedBy: [], open: [true, true]},"
                          " 0, 0, 10, {})")
        centre = re.search(r'<stop offset="0" style="stop-color:([^;"]+)', pool)
        eye = re.search(r'<stop offset="0" style="stop-color:([^;"]+)', vortex)  # type: ignore[arg-type]
        assert centre and centre.group(1) == "var(--fold-light)"
        assert eye and eye.group(1) == "var(--portal-edge)"
        # 0.9 of a cell across.
        assert 'class="pf-pool" fill="var(--pair-1)"' in pool and 'r="4.500"' in pool

    def test_its_bands_edges_are_whose_pieces_may_use_it(self) -> None:
        both, white, black, nobody = (_pool(open_=o) for o in (
            "[true, true]", "[true, false]", "[false, true]", "[false, false]"))
        assert re.findall(r'data-band="(\d)"', both) == ["0", "1"]
        assert re.findall(r'data-band="(\d)"', white) == ["0", "0"]
        assert re.findall(r'data-band="(\d)"', black) == ["1", "1"]
        assert 'data-band' not in nobody and nobody.count('class="pf-band"') == 2
        assert 'data-turn="in"' in both and 'data-turn="out"' in nobody

    def test_a_filtered_ends_bands_are_dashed(self) -> None:
        plain, dashed = _pool(), _pool(more=", filtered: true")
        bands = lambda drawn: re.findall(r'class="pf-band"[^>]* d="([^"]+)"', drawn)  # noqa: E731
        assert all(d.count("M") == 1 for d in bands(plain))
        assert all(d.count("M") == 3 for d in bands(dashed))
        assert "data-filtered" in dashed

    def test_its_collar_is_who_may_move_it(self) -> None:
        assert "pf-collar" not in _pool()
        white = _pool(movers="[0]")
        assert 'data-moved="0"' in white and 'stroke="var(--p0)"' in white
        both = _pool(movers="[0, 1]")
        collar = both[both.index("pf-collar"):]
        # Half and half, the bottom half the bottom player's.
        assert collar.count("<path") == 2 and "A" in collar

    def test_its_beak_points_where_it_leads_and_a_dead_end_has_none(self) -> None:
        aimed = _pool(aim="Math.PI / 2")
        assert 'data-aim="90.0"' in aimed and 'class="pg-beak"' in aimed and 'class="pg-beak-cut"' in aimed
        assert "data-aim" not in _pool(open_="[false, false]", aim="0")

    def test_its_number_moves_to_a_badge_when_a_piece_stands_in_it(self) -> None:
        assert 'class="pg-num"' in _pool(numbered="true")
        assert 'class="pg-num"' not in _pool(numbered="true", occupied="true")
        badge = _wrapped("foldBadgeHtml({pair: 3}, 0, 0, 10)")
        assert 'data-badge="3"' in badge and ">3</text>" in badge and 'cx="8.400"' in badge

    def test_its_rim_is_liquid_and_a_thumbnail_keeps_it_plain(self) -> None:
        assert "<feTurbulence" in _pool() and 'filter="url(#pfwob' in _pool()
        assert "<feTurbulence" not in _pool(plain="true")

    def test_idle_it_turns_and_a_groups_ends_turn_in_step(self) -> None:
        idle = _pool(idle="true")
        assert "pf-idle" in idle and "pf-wobble" in idle
        assert "pf-idle" not in _pool() and "pf-wobble" not in _pool()
        delays = _wrapped("[1, 1, 2].map(p => /animation-delay:([-\\d.]+)s/.exec(foldGlyph("
                          + _FOLD % (1, "[]", "[true, true]", "") + ".pair === p ? "
                          "Object.assign(" + _FOLD % (1, "[]", "[true, true]", "") + ", {pair: p}) : "
                          "Object.assign(" + _FOLD % (1, "[]", "[true, true]", "") + ", {pair: p}),"
                          " 0, 0, 10, {idle: true}))[1])")
        assert delays[0] == delays[1] != delays[2]  # type: ignore[index]

    def test_the_glow_is_clear_inside_the_rim(self) -> None:
        glow = _wrapped("foldGlowHtml(0, 0, 10, 2)")
        stops = re.findall(r'<stop offset="([\d.]+)" style="stop-color:var\(--pair-2\);stop-opacity:([\d.]+)"', glow)
        assert stops[0][1] == "0" and float(stops[0][0]) * 6.4 >= 3.9  # nothing inside 0.39 of a cell
        assert stops[-1] == ("1", "0") and float(stops[1][1]) > 0

    def test_the_stylesheet_turns_it_and_holds_it_still_under_reduced_motion(self) -> None:
        from page_text import PAGE

        assert ".pf-idle { animation: pf-turn 12s linear infinite; }" in PAGE
        assert ".pf-wobble { animation: pf-wobble 8s linear infinite; }" in PAGE
        assert "@media (prefers-reduced-motion: reduce) { .pf-idle, .pf-wobble { animation: none; } }" in PAGE


# ------------------------------------------------------------------ on the play board

def _snapshot(placement: dict[str, tuple[str, int]], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or _gardner_folds()
    engine = engine_for(spec)
    session = Session(id="f1", game=_game(spec), seats=("human", "human"), seed=0,
                      state=engine.make_state(placement))
    return session.snapshot()


_BOARD = """
let DRAWN = '';
const BOARD_SVG = { setAttribute() {}, getAttribute: () => null, querySelector: () => null,
  querySelectorAll: () => [], appendChild() {}, closest: () => null, parentNode: null,
  set innerHTML(v) { DRAWN = v; }, get innerHTML() { return DRAWN; } };
// Every other element a draw touches: anything asked of it is another such element, called
// or not, and it reads as nothing in a sentence or a sum.
const deep = () => new Proxy(function () {}, {
  get: (t, k) => k === Symbol.toPrimitive ? (() => '') : k === 'style' || k === 'dataset' ? {} : deep(),
  set: () => true, apply: () => deep() });
document.getElementById = id => id === 'board' ? BOARD_SVG : deep();
document.createElementNS = () => deep();
const cellOf = sq => { const m = DRAWN.split(`data-square="${sq}"`); return m.length > 1 ? m[1].split('data-square=')[0] : ''; };
"""

_ROOK_IN = {"e1": ("K", 0), "e5": ("K", 1), "b3": ("R", 0)}
# Schema 24: a rook below the fold on b3, whose landing there may stay or come out on d3.
_ROOK_BELOW = {"e1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0)}


def _board(placement: dict[str, tuple[str, int]], steps: str, expression: str,
           spec: dict[str, Any] | None = None) -> Any:
    return _in_page(f"SESSION = {json.dumps(_snapshot(placement, spec))};\n" + _BOARD + steps
                    + "drawBoard();", expression)


class TestOnThePlayBoard:
    def test_the_pool_lies_under_the_piece_standing_in_it(self) -> None:
        cell = _board(_ROOK_IN, "", f"cellOf({_square('b3')})")
        assert 'data-fold-at="11"' in cell and 'data-piece="11"' in cell
        assert cell.index("data-fold-at") < cell.index("data-piece")

    def test_nothing_is_drawn_on_the_far_end(self) -> None:
        """§5.3R, the user: *"let's remove the ghost image at the exit portal"*. The far end
        of a fold a piece stands in draws its pool and nothing else, empty or held; the
        beak says where the fold leads."""
        far = _board(_ROOK_IN, "", f"cellOf({_square('d3')})")
        assert 'data-fold-at="13"' in far and "data-aim" in far
        assert "pf-echo" not in far and "<image" not in far and "data-piece" not in far
        held = _board({**_ROOK_IN, "d3": ("N", 1)}, "", f"cellOf({_square('d3')})")
        assert held.count("<image") == 1 and "pf-echo" not in held
        drawn = _board(_ROOK_IN, "", "DRAWN")
        assert drawn.count("<image") == 3  # the two kings and the rook, once each
        from page_text import PAGE

        assert "foldEcho" not in PAGE and "pf-echo" not in PAGE and "FOLD_ECHO" not in PAGE

    def test_the_glow_is_laid_over_the_squares(self) -> None:
        drawn = _board(_ROOK_IN, "", "DRAWN")
        assert drawn.count('class="pf-glow"') == 2
        assert drawn.index("pf-glow") > drawn.rindex('data-square="24"')

    def test_staying_and_going_through_each_take_an_ordinary_dot(self) -> None:
        """§5.3R, the user: *"as simple as selecting which square to land on, just as any
        other landing square is selected, with legal ones marked with a dot"*. The fold (the
        stay) and its far end (the go-through) each take the plain dot, no hue of their own."""
        drawn = _board(_ROOK_BELOW, f"SELECTED = {_square('b1')};", "DRAWN")
        assert "data-through" not in drawn
        for name in ("b3", "d3", "b2"):
            dots = re.findall(rf'<circle[^>]*fill="([^"]+)"[^>]*data-square="{_square(name)}"', drawn)
            assert dots == ["var(--dot)"], name

    def test_a_go_through_onto_an_enemy_piece_takes_the_ordinary_capture_ring(self) -> None:
        drawn = _board({**_ROOK_BELOW, "d3": ("B", 1)}, f"SELECTED = {_square('b1')};", "DRAWN")
        rings = re.findall(rf'<circle[^>]*fill="([^"]+)"\s+stroke="([^"]+)"[^>]*data-square="{_square("d3")}"',
                           drawn)
        assert rings == [("none", "var(--dot)")] and "data-through" not in drawn

    def test_its_words_say_what_a_fold_does(self) -> None:
        said = _board(_ROOK_IN, "", f"foldOn({_square('d3')}).words")
        assert said["head"] == "The portal on d3"
        assert said["says"].startswith("A fold: a piece of either side that ends its move on d3"
                                       " may stay there, or come out on b3, taking a piece there"
                                       " as any move does. A move that passes over it is not"
                                       " changed.")
        assert said["says"].endswith("Nobody moves it.")

    def test_a_tap_on_an_empty_fold_says_it_under_the_board(self) -> None:
        assert _board(_ROOK_IN, f"clickSquare({_square('d3')});", "INSPECT") == _square("d3")
        assert _board(_ROOK_IN, f"clickSquare({_square('b3')});", "[INSPECT, SELECTED]") == [
            None, _square("b3")]

    def test_the_legend_names_a_fold_and_a_portal_apart(self) -> None:
        kinds = _wrapped("[boardKinds([], {ends: [{kind: 'fold'}]}), boardKinds([], {ends: [{kind:"
                         " 'momentum'}, {kind: 'fold'}]}), boardKinds([], {ends: [{}]})]")
        assert kinds == [["fold"], ["portal", "fold"], ["portal"]]
        row = _wrapped("MARK_KINDS.fold")
        assert row["name"] == "fold portal" and row["gloss"] == "foldKey"  # type: ignore[index]
        assert row["label"] == ("a piece that stops here may stay, or come out of the end its"  # type: ignore[index]
                                " beak points at")

    def test_the_legend_key_is_a_pool(self) -> None:
        key = _wrapped("markKey('fold')")
        assert 'class="fold"' in key and 'data-mark="fold"' in key

    def test_its_gloss_draws_a_pool_with_a_piece_in_it_and_no_echo(self) -> None:
        """§5.3R: the pool with its beak and a piece standing in it with its badge, no echo;
        a row for the beak and the portal map; the bands' and the collar's rows."""
        figure: str = _wrapped("foldKeyFigure(['white', 'black'])")  # type: ignore[assignment]
        assert "pf-echo" not in figure and figure.count("pf-badge") == 1
        assert "Its beak and the portal map: where a piece comes out" in figure
        beak_row = figure[figure.index("Its beak and the portal map"):figure.index("Its bands")]
        assert beak_row.count('class="fold"') == 2 and beak_row.count("data-aim") == 2
        assert beak_row.count('data-way="') == 2  # the map's two arrows, one each way
        assert "Its bands: whose pieces may go through" in figure
        assert "Its collar: who may move it" in figure
        from page_text import PAGE

        assert "foldKey: ['How to read a fold portal'," in PAGE


# ------------------------------------------------------------------ tapping (§5.3R)
#
# The page as the reader taps it: `test_web_landing`'s harness, whose `api` records what is
# sent and whose chooser box reads its buttons back.

def _tap(placement: dict[str, tuple[str, int]], steps: str, expression: str,
         spec: dict[str, Any] | None = None) -> Any:
    from test_web_landing import _HARNESS

    return _in_page(f"SESSION = {json.dumps(_snapshot(placement, spec))};\n" + _HARNESS
                    + "const sentText = () => CALLS.map(c => textOf(c[1].move));\n" + steps,
                    expression)


def _taps(*names: str) -> str:
    return "".join(f"clickSquare({_square(name)});" for name in names)


_MOVABLE = {"orange": [0, 1], "blue": [0, 1]}


class TestStayOrGoThrough:
    def test_a_tap_on_the_far_end_goes_through_and_a_tap_on_the_fold_stays(self) -> None:
        through = _tap(_ROOK_BELOW, _taps("b1", "d3"), "[BOX.hidden, sentText()]")
        assert through == [True, ["Rb1-b3~d3"]]
        stay = _tap(_ROOK_BELOW, _taps("b1", "b3"), "[BOX.hidden, sentText()]")
        assert stay == [True, ["Rb1-b3"]]

    def test_a_go_through_onto_an_enemy_piece_is_a_tap_too(self) -> None:
        taken = _tap({**_ROOK_BELOW, "d3": ("B", 1)}, _taps("b1", "d3"), "sentText()")
        assert taken == ["Rb1-b3~xd3"]


class TestPieceOrPortal:
    """A piece standing in a fold its side may move, and a square both can reach: the user's
    two choices, *Move the piece* and *Move the portal*, and never both."""

    _IN = {"e1": ("K", 0), "e5": ("K", 1), "b3": ("R", 0)}

    def _open(self, expression: str, more: str = "") -> Any:
        return _tap(self._IN, _taps("b3", "b2") + more, expression,
                    _gardner_folds(moved_by=_MOVABLE))

    def test_it_offers_exactly_the_piece_and_the_portal(self) -> None:
        hidden, ways, said = self._open(
            "[BOX.hidden, BOX.picks.map(b => b.dataset.which), said()]")
        assert hidden is False and ways == ["piece", "portal"]
        assert said.startswith("Move the piece or the portal?")
        assert "Move the piece" in said and "Move the portal" in said
        assert "both" not in said.lower().split("tap the one")[1]
        assert "not the piece" not in said and "fold portal" not in said

    def test_each_choice_is_its_own_move(self) -> None:
        piece = self._open("sentText()", "BOX.picks[0].click();")
        portal = self._open("sentText()", "BOX.picks[1].click();")
        assert piece == ["Rb3-b2"] and len(portal) == 1 and portal[0].endswith(":b3-b2")
        assert piece != portal

    def test_no_move_carries_the_piece_and_its_fold_together(self) -> None:
        engine = engine_for(_gardner_folds(moved_by=_MOVABLE))
        texts = engine.move_texts(engine.make_state(self._IN))
        to_b2 = sorted(t for t in texts if t.endswith("b2"))
        assert len(to_b2) == 2 and to_b2[1] == "Rb3-b2" and ":b3-b2" in to_b2[0]


class TestTheMisplay:
    """The user (2026-10-12): *"selecting a piece to see its legal moves, then intending to
    cancel the move and move the portal instead, but if the portal is a legal destination,
    so I make an unintended move."* That route now asks; it never plays."""

    def _route(self, expression: str, more: str = "", placement: dict[str, Any] | None = None) -> Any:
        return _tap(placement or _ROOK_BELOW, _taps("b1", "b3") + more, expression,
                    _gardner_folds(moved_by=_MOVABLE))

    def test_a_tap_on_a_movable_fold_the_piece_can_land_on_plays_nothing(self) -> None:
        hidden, sent, ways = self._route("[BOX.hidden, CALLS.length, BOX.picks.map(b => b.dataset.which)]")
        assert hidden is False and sent == 0 and ways == ["piece", "portal"]

    def test_move_the_portal_picks_it_and_plays_nothing(self) -> None:
        picked = self._route("[SELECTED, CALLS.length, BOX.hidden]", "BOX.picks[1].click();")
        assert picked == [_square("b3"), 0, True]

    def test_move_the_piece_lands_there(self) -> None:
        assert self._route("sentText()", "BOX.picks[0].click();") == ["Rb1-b3"]

    def test_a_fold_the_side_may_not_move_is_still_one_tap(self) -> None:
        sent = _tap(_ROOK_BELOW, _taps("b1", "b3"), "[BOX.hidden, sentText()]")
        assert sent == [True, ["Rb1-b3"]]

    def test_every_other_landing_is_still_one_tap(self) -> None:
        sent = _tap(_ROOK_BELOW, _taps("b1", "b2"), "sentText()", _gardner_folds(moved_by=_MOVABLE))
        assert sent == ["Rb1-b2"]

    def test_the_far_end_asks_too_and_move_the_piece_goes_through(self) -> None:
        """A go-through ends on the far end, a fold this side may move too: the same
        question, and *Move the piece* is the go-through."""
        ways, sent = _tap(_ROOK_BELOW, _taps("b1", "d3") + "const W = BOX.picks.map(b => b.dataset.which);"
                          " BOX.picks[0].click();", "[W, sentText()]", _gardner_folds(moved_by=_MOVABLE))
        assert ways == ["piece", "portal"] and sent == ["Rb1-b3~d3"]


# ------------------------------------------------------------------ the server's half

class TestTheServer:
    def test_a_fold_transit_names_its_way_in_and_out_and_a_direct_move_none(self) -> None:
        moves = {m["text"]: m for m in _snapshot({**_ROOK_BELOW, "d3": ("B", 1)})["moves"]}
        through = moves["Rb1-b3~xd3"]
        assert (through["foldIn"], through["foldOut"], through["foldPair"]) == (
            _square("b3"), _square("d3"), 1)
        assert (through["foldInName"], through["foldOutName"]) == ("b3", "d3")
        assert "foldIn" not in moves["Rb1-b3"] and "foldIn" not in moves["Rb1-b2"]

    def test_a_still_draws_no_echo(self) -> None:
        """The designer's bench, the game page's start and a card's thumbnail draw a piece
        standing in a fold once, and nothing on the far end (§5.3R)."""
        preview = designing.check(_gardner_folds())["preview"]
        rook = _snapshot(_ROOK_IN)["board"][_square("b3")]
        preview["board"][_square("b3")] = rook
        for how in ("null, null, {}", "null, null, null"):
            drawn = _in_page(f"const P = {json.dumps(preview)};", f"boardPreview(P, {how})")
            assert "pf-echo" not in drawn
            far = drawn.split(f'<g data-square="{_square("d3")}">')[1].split("<g data-square=")[0]
            assert 'class="fold"' in far and "<image" not in far
        thumb = _in_page(f"const P = {json.dumps({**preview, 'plain': True})};", "boardPreview(P)")
        assert "pf-echo" not in thumb and thumb.count("<image") == sum(1 for value in preview["board"] if value)

    def test_a_still_carries_the_folds(self) -> None:
        preview = designing.check(_gardner_folds())["preview"]
        assert [(f["square"], f["pair"]) for f in preview["folds"]] == [(_square("b3"), 1), (_square("d3"), 1)]
        assert "folds" not in designing.check(copy.deepcopy(_GAME))["preview"]

    def test_the_move_map_carries_fold_transits(self) -> None:
        spec = _gardner_folds()
        rook = [p["id"] for p in spec["pieces"]].index("R")
        found = designing.move_map(spec, rook, _square("b1"))
        transits = [t for t in found.get("transits", []) if "foldIn" in t]
        assert transits and all(t["foldIn"] == _square("b3") and t["foldOut"] == _square("d3")
                                == t["to"] for t in transits)

    def test_a_fold_groups_card_says_a_fold(self) -> None:
        (card,) = designing.check(_gardner_folds(moved_by={"orange": [0, 1]}))["portals"]["cards"]
        assert card["kind"] == "fold"
        assert card["ways"] == [
            "White's and Black's pieces that stop on the end on b3 may stay there, or come out"
            " on the end on d3.",
            "White's and Black's pieces that stop on the end on d3 may stay there, or come out"
            " on the end on b3."]
        assert card["solid"] == [] and card["warnings"] == [] and card["exits"] is None

    def test_a_momentum_card_says_its_kind_and_is_otherwise_as_it_was(self) -> None:
        from test_designer_portals import _with

        (card,) = designing.check(_with({"orange": "b2", "blue": "e6"}))["portals"]["cards"]
        assert card["kind"] == "momentum"


# ------------------------------------------------------------------ the designer (§5.4)

class TestHowItWorks:
    def test_the_card_offers_momentum_and_fold_each_drawn(self) -> None:
        from test_designer_portals import _with

        html = _page(_with({"orange": "b2", "blue": "e6"}), "", "pairCardHtml(draftPortals()[0], 0, 1)")
        assert "How it works" in html
        assert 'data-pkind="0.momentum"' in html and 'data-pkind="0.fold"' in html
        momentum = html[html.index('data-pkind="0.momentum"'):html.index('data-pkind="0.fold"')]
        assert 'aria-pressed="true"' in momentum and "data-portal" in momentum
        fold = html[html.index('data-pkind="0.fold"'):]
        assert 'class="fold"' in fold and "may stay, or come out of the other end" in fold

    def test_choosing_fold_writes_it_and_momentum_takes_it_out(self) -> None:
        from test_designer_portals import _with

        spec = _with({"orange": "b2", "blue": "e6"})
        folded = _page(spec, "setKind(0, 'fold');", "DRAFT.rules.portals")
        assert folded == [{"piece": "O", "orange": "b2", "blue": "e6", "kind": "fold"}]
        back = _page(spec, "setKind(0, 'fold'); setKind(0, 'momentum');", "DRAFT.rules.portals")
        assert back == [{"piece": "O", "orange": "b2", "blue": "e6"}]
        draft = _page(spec, "setKind(0, 'fold');", "DRAFT")
        answer = designing.check(draft)
        assert answer["valid"], answer["errors"]

    def test_a_ring_keeps_its_kind_through_a_shape_change(self) -> None:
        from test_designer_portals import _with

        spec = _with({"ends": [{"at": "b2", "to": 1}, {"at": "e2", "to": 2}, {"at": "c6", "to": 0}]})
        group = _page(spec, "setKind(0, 'fold'); setShape(0, 'funnel');", "DRAFT.rules.portals[0]")
        assert group["kind"] == "fold" and group["ends"][2] == {"at": "c6"}


class TestTheReplay:
    def test_the_offline_replay_carries_the_folds_and_draws_the_pool(self, tmp_path: Any) -> None:
        from chess_variant_sandbox.engine import Engine
        from chess_variant_sandbox.viz.html import render_position

        spec = _gardner_folds()
        state = Engine(spec).make_state(_ROOK_IN)
        page = render_position(spec, state, tmp_path / "fold.html").read_text(encoding="utf-8")
        found = re.search(r"const DATA = (\{.*?\});\s", page, re.S)
        assert found
        data = json.loads(found.group(1))
        assert [f["square"] for f in data["folds"][0]] == [_square("b3"), _square("d3")]
        assert "--fold-light: #fffbe8;" in page and "foldGlyph(foldEnd, x, y, CELL" in page


class TestTheBench:
    def test_a_fold_the_still_draws_is_no_pending_end(self) -> None:
        """The bench drew a vortex over each pool: a fold is in the still's `folds`, not its
        board, and was taken for an end the check had not drawn yet."""
        spec = _gardner_folds()
        pending = _page(spec, "", "benchMarks('setup').pending", designing.check(spec))
        assert pending == []
