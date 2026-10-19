"""Q638: the board alone, full screen, on a phone.

The user, 2026-10-02, on their own iPhone (`docs/reference/first-visit-study.md` §1.2a): *"it
would maybe be nice if it would put the board and only the board in full screen mode."* The
play frame's bar carries *full screen* (`#fullBoard`), which turns the frame into a view of
its own over the page (`setBoardFull`, `html.board-full`): the strip, the board, the slot a
round sends its move from, the hand, and the way out. The browser's Fullscreen API hides
its bars where it has one; where it has none (an iPhone's Safari) or refuses, the view is
the same, filling the window. A history entry makes a phone's back gesture leave the view
rather than the game. `fitFull` sizes the board to the room the view leaves, upright or on
its side, and turning the phone round refits it without leaving the view.

The checks run the page's own functions in node: the toggle against a stubbed document
whose Fullscreen API and history record what was asked of them, and the fit against a
small model of the view's layout (`_MODEL`), whose boxes follow the styles `fitFull`
writes, as `test_web_board_fit.py`'s model does for the page's own frame. The browser's
reading is the look in the in-app browser at 375 x 812 and 812 x 375 (the branch's record,
`docs/reference/site-as-built.md` §4).
"""

from __future__ import annotations

import json
import re
from typing import Any

import node_session
from test_web import _in_page

from chess_variant_sandbox.web.page import style

# A document with a Fullscreen API, a history and listeners that record what is asked of
# them. `api` false is an iPhone's Safari: no `requestFullscreen` on an element at all.
_STUB = """
const CALLS = [];
const classes = (...names) => { const s = new Set(names); return {
  contains: n => s.has(n), add: n => s.add(n), remove: n => s.delete(n),
  toggle: (n, on) => { if (on === undefined ? !s.has(n) : on) s.add(n); else s.delete(n); return s.has(n); },
  names: () => [...s] }; };
const HEARD = { document: {}, window: {} };
const hear = where => (type, fn) => (HEARD[where][type] = HEARD[where][type] || []).push(fn);
const fire = (where, type, event) => (HEARD[where][type] || []).forEach(fn => fn(event || {}));
document.addEventListener = hear('document');
window.addEventListener = hear('window');
global.history = { pushState: state => CALLS.push(['push', state]), back: () => CALLS.push(['back']),
                   replaceState: () => {} };
const BUTTON = { id: 'fullBoard', innerHTML: '', title: '', attrs: {},
                 setAttribute(k, v) { this.attrs[k] = v; }, querySelector: () => null };
document.getElementById = id => (id === 'fullBoard' ? BUTTON : null);
function browser(api, refuses) {
  document.documentElement = { classList: classes() };
  document.fullscreenElement = null;
  if (api) {
    document.fullscreenEnabled = true;
    document.documentElement.requestFullscreen = opts => {
      CALLS.push(['request', opts && opts.navigationUI]);
      if (refuses) return Promise.reject(new Error('refused'));
      document.fullscreenElement = document.documentElement;
      return Promise.resolve();
    };
    document.exitFullscreen = () => {
      CALLS.push(['exit']); document.fullscreenElement = null; return Promise.resolve(); };
  }
}
const state = () => ({ full: BOARD_FULL, cls: document.documentElement.classList.contains('board-full'),
                       calls: CALLS.splice(0), words: BUTTON.innerHTML.includes('leave full screen') });
"""


def _toggle(setup: str, expression: str) -> Any:
    """`_in_page`, awaiting the expression: a browser's answer to a request comes later.
    `_in_page` prints its expression at once, so the answer is printed after it once the
    promise has settled, and the last line is the one read."""
    done = node_session.run(_PAGE_STUB + _script() + "\n" + _STUB + setup
                            + "\n(async () => { const answer = await (" + expression
                            + "); console.log(JSON.stringify(answer)); })();\n")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# `_in_page`'s document, good enough for the page to load.
_PAGE_STUB = """
  const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null), set: () => true });
  global.document = { getElementById: () => el, querySelector: () => null,
                      querySelectorAll: () => [], addEventListener: () => {},
                      createElement: () => el, body: el };
  global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                    addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
  global.location = { pathname: '/game', hash: '' };
  global.fetch = () => new Promise(() => {});
"""


def _script() -> str:
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    return script.group(1)


class TestTheControl:
    def test_the_page_board_offers_it_and_the_designer_board_does_not(self) -> None:
        page = _in_page("", "frameHtml(false, 'sandbox', false)")
        assert 'id="fullBoard"' in page and "full screen" in page
        assert 'id="fullBoard"' not in _in_page("", "frameHtml(true, 'sandbox', false)")
        # A round's frame (its slot) and a rated game's carry it too.
        assert 'id="fullBoard"' in _in_page("", "frameHtml(false, 'leaderboard', false, null, true)")

    def test_the_control_turns_the_view_on_and_off(self) -> None:
        said = _toggle("browser(false); wireFullButton();",
                       "(() => { BUTTON.onclick(); const on = state();"
                       " BUTTON.onclick(); return [on, state()]; })()")
        on, off = said
        assert on["full"] and on["cls"] and on["words"]
        assert not off["full"] and not off["cls"] and not off["words"]

    def test_at_a_desk_it_is_not_offered_unless_a_view_is_open(self) -> None:
        css = style.STYLE
        start = css.index("@media (min-width: 861px) and (hover: hover) and (pointer: fine)")
        rule = css[start:css.index("}", start) + 1]
        assert rule.endswith("html:not(.board-full) .frame-controls #fullBoard { display: none; }")


class TestTheFullscreenApi:
    def test_where_there_is_one_it_is_asked_and_the_way_back_pushed(self) -> None:
        on = _toggle("browser(true);", "(() => { setBoardFull(true, 'button'); return state(); })()")
        assert on["full"] and on["cls"]
        assert on["calls"] == [["push", {"boardFull": True}], ["request", "hide"]]

    def test_leaving_by_the_control_leaves_the_browsers_full_screen_and_the_entry(self) -> None:
        said = _toggle("browser(true);", """(async () => {
            setBoardFull(true, 'button'); await null; fire('document', 'fullscreenchange');
            state(); setBoardFull(false, 'button'); const off = state();
            fire('document', 'fullscreenchange'); return [off, state()]; })()""")
        off, after = said
        assert not off["full"] and not off["cls"]
        assert off["calls"] == [["exit"], ["back"]]
        # The browser's own change that follows is not a second way out.
        assert after["calls"] == []

    def test_the_browsers_own_way_out_leaves_the_view(self) -> None:
        """Escape at a desk, the back gesture on an Android phone: the browser leaves its
        full screen, and the view goes with it, its history entry taken back off."""
        said = _toggle("browser(true);", """(async () => {
            setBoardFull(true, 'button'); await null; fire('document', 'fullscreenchange');
            state(); document.fullscreenElement = null; fire('document', 'fullscreenchange');
            return state(); })()""")
        assert not said["full"] and not said["cls"]
        assert said["calls"] == [["back"]]

    def test_a_refusal_keeps_the_view(self) -> None:
        said = _toggle("browser(true, true);", """(async () => {
            setBoardFull(true, 'button'); await new Promise(r => setTimeout(r, 0));
            return state(); })()""")
        assert said["full"] and said["cls"]
        assert said["calls"] == [["push", {"boardFull": True}], ["request", "hide"]]

    def test_without_one_the_view_fills_the_window(self) -> None:
        """An iPhone's Safari has no `requestFullscreen` on an element: the view is the
        page's alone, and nothing is asked of a browser that cannot answer."""
        said = _toggle("browser(false);", """(() => {
            setBoardFull(true, 'button'); const on = state();
            setBoardFull(false, 'button'); return [on, state()]; })()""")
        on, off = said
        assert on["full"] and on["cls"] and on["calls"] == [["push", {"boardFull": True}]]
        assert not off["full"] and not off["cls"] and off["calls"] == [["back"]]

    def test_the_back_gesture_leaves_the_view_not_the_game(self) -> None:
        said = _toggle("browser(false);", """(() => {
            setBoardFull(true, 'button'); state();
            fire('window', 'popstate', { state: null }); return state(); })()""")
        assert not said["full"] and not said["cls"]
        # The entry is already gone: nothing more is taken off the history.
        assert said["calls"] == []

    def test_another_page_leaves_the_view(self) -> None:
        said = _toggle("browser(false);", """(() => {
            setBoardFull(true, 'button'); state();
            const plain = { textContent: '', style: {}, classList: classes() };
            document.getElementById = id => (id === 'fullBoard' ? BUTTON : plain);
            location.hash = '#/whats-new'; render(); return state(); })()""")
        assert not said["full"] and not said["cls"]


# The view laid out the way the stylesheet lays it out, close enough for `fitFull`'s
# arithmetic: 8 px of padding top and bottom and 10 px at the sides; upright, the strip and
# the way out on the top row and the board under them (8 px over it), then the slot and the
# hand, all centred in the height; on its side, the board centred in the column the side
# column (at least 200 px, 36% of the width) and a 12 px gap leave, the strip, the slot and
# the hand down the side column and the way out at its foot. The panel's rim is the play
# panel's: 38 px across and 40 down round the board's wood, the wood a share of the width.
_MODEL = """
const classes = (...names) => { const s = new Set(names); return {
  contains: n => s.has(n), add: n => s.add(n), remove: n => s.delete(n),
  toggle: (n, on) => { if (on === undefined ? !s.has(n) : on) s.add(n); else s.delete(n); return s.has(n); },
  names: () => [...s] }; };
const px = v => (v ? parseFloat(v) : NaN);
const M = {};
const HEARD = {};
window.addEventListener = (type, fn) => (HEARD[type] = HEARD[type] || []).push(fn);
document.addEventListener = () => {};
global.history = { pushState: () => {}, back: () => {}, replaceState: () => {} };
const rect = (left, top, width, height) => ({ left, top, width, height, right: left + width,
                                              bottom: top + height });
function screenIs(W, H) { M.W = W; M.H = H; window.innerWidth = W; window.innerHeight = H; }
function lay() {
  const wide = M.frame.classList.contains('wide');
  const full = document.documentElement.classList.contains('board-full');
  const iw = M.W - 20, ih = M.H - 16;
  const statusH = (M.status.classList.contains('narrow') ? 3 : 2) * 22 + 14;
  const barW = 56, barH = 50, sendH = M.sending ? (M.W > 480 ? 140 : 68) : 0;
  const side = Math.max(200, 0.36 * iw), boardCol = wide ? iw - side - 12 : iw;
  const P = px(M.panel.style.width) || boardCol;
  const wood = Math.min(8, Math.max(2, 0.015 * (P - 38)));
  const sw = P - 38 - 2 * wood, sh = sw * M.ratio, ph = sh + 40 + 2 * wood;
  if (!full) {
    const panel = rect(0, 200, P, ph);
    return { panel, svg: rect(20 + wood, 216 + wood, sw, sh), status: rect(0, 100, iw, statusH),
             bar: rect(0, panel.bottom + 10, iw, barH), send: null, hand: rect(0, panel.bottom, iw, 0) };
  }
  if (!wide) {
    const top = Math.max(statusH, barH);
    const total = top + 8 + ph + (sendH ? sendH + 8 : 0) + (M.handH ? M.handH + 8 : 0)
      + (M.reviewH ? M.reviewH + 8 : 0);
    const y = 8 + Math.max(0, (ih - total) / 2);
    const panel = rect(10 + (iw - P) / 2, y + top + 8, P, ph);
    const send = sendH ? rect(10, panel.bottom + 8, iw, sendH) : null;
    const hand = rect(10, (send || panel).bottom + (M.handH ? 8 : 0), iw, M.handH);
    return { panel, svg: rect(panel.left + 20 + wood, panel.top + 16 + wood, sw, sh),
             status: rect(10, y, iw - barW - 12, statusH), bar: rect(10 + iw - barW, y, barW, barH),
             send, hand, review: rect(10, hand.bottom + (M.reviewH ? 8 : 0), iw, M.reviewH) };
  }
  const x = 10 + boardCol + 12;
  const panel = rect(10 + (boardCol - P) / 2, 8 + (ih - ph) / 2, P, ph);
  const status = rect(x, 8, side, statusH);
  const send = sendH ? rect(x, status.bottom + 8, side, sendH) : null;
  const hand = rect(x, (send || status).bottom + (M.handH ? 8 : 0), side, M.handH);
  return { panel, svg: rect(panel.left + 20 + wood, panel.top + 16 + wood, sw, sh), status,
           send, hand, review: rect(x, hand.bottom + (M.reviewH ? 8 : 0), side, M.reviewH),
           bar: rect(x + side - barW, 8 + ih - barH, barW, barH) };
}
function viewModel(W, H, ratio, sending, handH, reviewH) {
  screenIs(W, H);
  M.ratio = ratio; M.sending = sending; M.handH = handH || 0; M.reviewH = reviewH || 0;
  document.documentElement = { classList: classes() };
  const node = (id, box, extra = {}) => Object.assign({ id, style: {}, hidden: false,
    classList: classes(), getBoundingClientRect: () => box() }, extra);
  M.frame = node('frame', () => rect(0, 0, M.W, M.H), { classList: classes('play-frame') });
  M.panel = node('panel', () => lay().panel, { closest: () => M.frame });
  M.svg = node('board', () => lay().svg, {
    getAttribute: () => `0 0 100 ${100 * ratio}`, closest: () => M.panel });
  M.status = node('status', () => lay().status, { classList: classes('row', 'frame-status') });
  M.controls = node('controls', () => lay().bar);
  M.send = sending ? node('corrSend', () => lay().send) : null;
  M.hand = node('hand', () => lay().hand);
  M.review = node('review', () => lay().review);
  M.row = node('boardRow', () => rect(0, 0, 0, 0));
  const ids = { board: M.svg, hand: M.hand, boardRow: M.row, corrSend: M.send, review: M.review };
  document.getElementById = id => ids[id] || null;
  const slots = { '[data-frame="controls"]': M.controls, '[data-frame="status"]': M.status };
  document.querySelector = q => slots[q] || null;
  global.getComputedStyle = el => {
    if (el === M.frame) return { paddingLeft: '10px', paddingRight: '10px', paddingTop: '8px',
                                 paddingBottom: '8px', columnGap: '12px', marginTop: '0px' };
    if (el === M.panel) return { marginTop: M.frame.classList.contains('wide') ? '0px' : '8px' };
    if (el === M.send || el === M.hand || el === M.review) return { marginTop: '8px' };
    return { marginTop: '0px' };
  };
}
function reading() {
  const l = lay();
  const boxes = [l.panel, l.status, l.bar, l.hand, l.review].concat(l.send ? [l.send] : []);
  const r = n => Math.round(n * 10) / 10;
  return { wide: M.frame.classList.contains('wide'), full: BOARD_FULL,
    panel: [r(l.panel.left), r(l.panel.top), r(l.panel.width), r(l.panel.height)],
    svg: [r(l.svg.width), r(l.svg.height)], width: M.panel.style.width,
    inside: boxes.every(b => b.left >= -0.01 && b.top >= -0.01 && b.right <= M.W + 0.01
                             && b.bottom <= M.H + 0.01),
    room: { across: M.W - 20, down: M.H - 16 } };
}
function fullView(W, H, ratio, sending, handH, reviewH) {
  viewModel(W, H, ratio, sending, handH, reviewH);
  setBoardFull(true, 'button');
  return reading();
}
"""


def _view(width: int, height: int, ratio: float, sending: bool = False, hand: int = 0,
          review: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = _in_page(
        _MODEL + "SESSION = { id: 's1', ruleset: {}, history: [] };",
        f"fullView({width}, {height}, {ratio}, {str(sending).lower()}, {hand}, {review})")
    return result


class TestTheFit:
    def test_upright_a_square_board_takes_the_screens_width(self) -> None:
        view = _view(375, 812, 1.0)
        assert view["full"] and not view["wide"] and view["inside"]
        left, _top, width, _height = view["panel"]
        assert 355 - 1 <= width <= 355 and left >= 10

    def test_on_its_side_a_square_board_takes_the_screens_height(self) -> None:
        view = _view(812, 375, 1.0)
        assert view["wide"] and view["inside"]
        _left, top, _width, height = view["panel"]
        assert 359 - 1.5 <= height <= 359 and top >= 8
        # The side column keeps its room: the board stands left of it.
        assert view["panel"][0] + view["panel"][2] <= 812 - 10 - 0.36 * 792 - 12 + 0.5

    def test_a_tall_board_upright_takes_the_height_left(self) -> None:
        view = _view(375, 812, 2.4)
        assert view["inside"]
        # 796 of height, less the strip's row (80), the board's margin (8) and the panel's
        # rim (40 and the wood): about 660, where the width would have given it 737.
        assert 655 < view["svg"][1] < 670 and view["panel"][2] < 355

    def test_a_round_keeps_its_slot_on_the_screen(self) -> None:
        """A game in rounds (and a game between two people) is sent from the slot under the
        board, which says the round's order once it is revealed: the board leaves it room,
        upright and on its side."""
        for width, height in ((375, 812), (812, 375)):
            for ratio in (1.0, 1.25, 0.8):
                view = _view(width, height, ratio, sending=True)
                assert view["inside"], (width, height, ratio, view)

    def test_a_hand_keeps_its_row(self) -> None:
        for width, height in ((375, 812), (812, 375)):
            view = _view(width, height, 1.0, hand=44)
            assert view["inside"], (width, height, view)

    def test_a_finished_game_keeps_its_review_row(self) -> None:
        """The end banner's *review* steps back through the game, so once it is over the
        review row is in the view too, and the board leaves it room."""
        for width, height in ((375, 812), (812, 375)):
            for sending in (False, True):
                view = _view(width, height, 1.0, sending=sending, hand=44, review=44)
                assert view["inside"], (width, height, sending, view)

    def test_a_second_fit_moves_nothing(self) -> None:
        for width, height, ratio in ((375, 812, 1.0), (812, 375, 1.0), (375, 812, 1.6), (812, 375, 0.7)):
            twice = _in_page(_MODEL + "SESSION = { id: 's1', ruleset: {}, history: [] };",
                             f"(() => {{ const a = fullView({width}, {height}, {ratio}, true, 0);"
                             " fitBoard(); return [a, reading()]; })()")
            assert twice[0] == twice[1], (width, height, ratio)


class TestTurningThePhone:
    def test_rotating_keeps_the_view_and_the_position_and_refits(self) -> None:
        said = _in_page(_MODEL + """
            const GAME = { id: 's1', ruleset: {}, history: [{ move: 'e2e4' }], board: { e4: 'P' } };
            SESSION = GAME;
            wireFit();
            const turn = (W, H) => { screenIs(W, H); (HEARD.resize || []).forEach(fn => fn({})); };
        """, """(() => {
            const upright = fullView(375, 812, 1.0, true, 0);
            turn(812, 375); const side = reading();
            turn(375, 812); const back = reading();
            return { upright, side, back, same: SESSION === GAME && SESSION.history.length === 1,
                     cls: document.documentElement.classList.contains('board-full') };
        })()""")
        assert said["same"] and said["cls"]
        assert said["side"]["full"] and said["side"]["wide"] and said["side"]["inside"]
        assert not said["upright"]["wide"]
        assert said["back"] == said["upright"]

    def test_leaving_gives_the_page_its_own_fit_back(self) -> None:
        said = _in_page(_MODEL + "SESSION = { id: 's1', ruleset: {}, history: [] };", """(() => {
            fullView(375, 812, 1.0, false, 0); const width = M.panel.style.width;
            setBoardFull(false, 'button');
            return [width, M.panel.style.width, document.documentElement.classList.contains('board-full')];
        })()""")
        assert said[0].endswith("px") and said[1] == "" and said[2] is False


class TestTheView:
    """What the view holds is the stylesheet's: the strip (whose move, the result), the
    board (its banner and cards are inside it), the slot, the hand and the way out."""

    def test_only_what_a_move_needs_is_drawn(self) -> None:
        css = style.STYLE
        # The review row, the next-game row, the ground line, a picked piece's sentences and
        # the legend; the strip, the board's row, the hand and the bar are never named.
        start = css.index("html.board-full .play-frame > .row:not([data-frame])")
        hidden = css[start:css.index("}", start)]
        assert hidden.rstrip().endswith("display: none;")
        for kept in (".frame-status", ".board-row", ".hand", ".frame-controls", ".corr-send"):
            assert kept not in hidden, kept
        assert "html.board-full .frame-controls > :not(#fullBoard) { display: none; }" in css
        assert "html.board-full .play-frame > .board-row { display: contents; }" in css
        # The slot that sends a move (and says a round's order) is placed, never hidden.
        assert "html.board-full .play-frame .corr-send { grid-area: send;" in css
        # An error raised in the view is shown over it.
        assert "html.board-full .toast.error { z-index: 70; }" in css
