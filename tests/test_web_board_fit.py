"""Q761: the play frame of a game between two people fills the room it has.

The user, on torus chess crazyhouse between two people: the board used about half the
room it could, *Send your move* and *Change my move* took only the board's width, and
*Change my move* was squeezed onto three lines and still clipped. The board is held short
by the screen's height there (a tall board, and a torus draws copies round it), and the
slot under it took the board's width and 150 px of that height.

`fitBoard` now puts the slot beside a board the height holds short when that gives the
bigger board, gives the strip and the slot at least the width the two buttons need, and
works each fit out for the layout it ends in, so the first move picked never moves the
board. The checks here run the page's own `fitBoard` against a small model of the frame's
layout (`_MODEL`): its boxes follow the styles `fitBoard` writes, its wood is a share of
the panel's width as the style's is, and it lays the slot under or beside the board as
`.board-row.beside` does. The browser's own reading is the look check's `text fits`,
`held steady` and `change move` rows on the `play-between-*` pages.
"""

from __future__ import annotations

from typing import Any

from test_web import _in_page

from chess_variant_sandbox.web.page import style

# A frame laid out the way the page lays it out, close enough for `fitBoard`'s arithmetic:
# the strip at the top of the column, the board panel under it (16 px of padding, and the
# wood a share of its width), the slot under the board or beside it, then the hand, the
# bar and the legend. Every box is worked out afresh from what `fitBoard` last wrote.
_MODEL = """
const classes = (...names) => { const s = new Set(names); return {
  contains: n => s.has(n), add: n => s.add(n), remove: n => s.delete(n),
  toggle: (n, on) => { if (on === undefined ? !s.has(n) : on) s.add(n); else s.delete(n); return s.has(n); },
  names: () => [...s] }; };
const px = v => (v ? parseFloat(v) : NaN);
const FIT = {};
function frameModel(H, C, ratio, between) {
  window.innerHeight = H; window.scrollY = 0;
  const rect = (left, top, width, height) => ({ left, top, width, height, right: left + width,
                                                bottom: top + height });
  const node = (id, box, extra = {}) => Object.assign({ id, style: {}, hidden: false,
    classList: classes(), getBoundingClientRect: () => box() }, extra);
  const lay = () => {
    const narrow = FIT.status.classList.contains('narrow');
    const lines = 2 + (between ? 1 : 0) + (narrow ? 1 : 0);
    const strip = rect(0, 100, Math.min(C, px(FIT.status.style.maxWidth) || C), lines * 22 + 14);
    const beside = FIT.row.classList.contains('beside');
    const P = beside ? px(FIT.panel.style.width) : Math.min(C, px(FIT.panel.style.maxWidth) || C);
    const pad = Math.min(8, Math.max(2, 0.015 * (P - 32)));
    const sw = P - 32 - 2 * pad, sh = sw * ratio;
    const panel = rect(0, strip.bottom + 6, P, sh + 32 + 2 * pad);
    const svg = rect(16 + pad, panel.top + 16 + pad, sw, sh);
    let send = null, rowBottom = panel.bottom;
    if (between) {
      send = beside ? rect(P + 16, panel.bottom - 186, C - P - 16, 186)
        : rect(0, panel.bottom + 8, Math.min(C, px(FIT.send.style.maxWidth) || C), 140);
      if (!beside) rowBottom = send.bottom;
    }
    const hand = rect(0, rowBottom + 10, C, 0);
    // A tapped piece's sentences, under the hand: `SAYS` lines of 18 px, 6 px over them.
    const says = FIT.says.hidden ? rect(0, hand.bottom, C, 0) : rect(0, hand.bottom + 6, C, 18 * FIT.SAYS);
    const controls = rect(0, says.bottom + 8, C, 50);
    const legend = rect(0, controls.bottom + 8, C, 30);
    return { strip, panel, svg, send, hand, says, controls, legend, row: rect(0, panel.top, C, rowBottom - panel.top) };
  };
  FIT.status = node('status', () => lay().strip, { classList: classes('row', 'frame-status') });
  FIT.row = node('boardRow', () => lay().row);
  FIT.panel = node('panel', () => lay().panel);
  FIT.svg = node('board', () => lay().svg, {
    getAttribute: () => `0 0 100 ${100 * ratio}`, closest: () => FIT.panel });
  FIT.send = between ? node('corrSend', () => lay().send) : null;
  FIT.hand = node('hand', () => lay().hand);
  FIT.controls = node('controls', () => lay().controls);
  FIT.legend = node('legend', () => lay().legend);
  FIT.says = node('pieceSays', () => lay().says, { hidden: true });
  FIT.SAYS = 1;
  FIT.lay = lay;
  const ids = { board: FIT.svg, hand: FIT.hand, boardRow: FIT.row, corrSend: FIT.send,
                pieceSays: FIT.says };
  document.getElementById = id => ids[id] || null;
  const slots = { '[data-frame="controls"]': FIT.controls, '[data-frame="status"]': FIT.status,
                  '[data-frame="legend"]': FIT.legend };
  document.querySelector = q => slots[q] || null;
  global.getComputedStyle = el => ({ marginTop: (el === FIT.send
    && !FIT.row.classList.contains('beside')) ? '8px' : el === FIT.says ? '6px' : '0px' });
}
function reading() {
  const l = FIT.lay();
  return { beside: FIT.row.classList.contains('beside'), narrow: FIT.status.classList.contains('narrow'),
    board: [Math.round(l.svg.top * 10) / 10, Math.round(l.svg.width * 10) / 10,
            Math.round(l.svg.height * 10) / 10],
    width: FIT.panel.style.maxWidth, strip: Math.round(l.strip.width),
    send: l.send && Math.round(l.send.width), legend: Math.round(l.legend.bottom) };
}
function fitted(H, C, ratio, between) {
  frameModel(H, C, ratio, between);
  fitBoard();
  const first = reading();
  fitBoard();
  const again = reading();
  return { first, again };
}
"""


def _fit(height: int, column: int, ratio: float, between: bool = True) -> dict[str, Any]:
    flag = "true" if between else "false"
    result: dict[str, Any] = _in_page(_MODEL, f"fitted({height}, {column}, {ratio}, {flag})")
    return result


class TestTheFrameMarkup:
    def test_the_board_and_the_slot_share_one_row(self) -> None:
        """The slot can stand beside the board only if the two are one row's children."""
        mine = _in_page("", "frameHtml(false, 'leaderboard', false, { you: 0, mayCancel: false })")
        row = mine[mine.index('<div class="board-row" id="boardRow">'):]
        assert row.index('class="panel board-panel"') < row.index('id="corrSend"')
        assert row.index('id="corrSend"') < row.index('id="hand"')
        watched = _in_page("", "frameHtml(false, 'leaderboard', false, { you: null, mayCancel: false })")
        assert 'id="boardRow"' in watched and 'id="corrSend"' not in watched
        plain = _in_page("", "frameHtml(false, 'sandbox', false)")
        assert 'id="boardRow"' in plain


class TestTheButtonsWrapTheirWords:
    """*Change my move* was given a width of nothing on a phone (`min-width: 0`) and could
    not wrap at a desk (`white-space: nowrap`); now a button too narrow wraps its words and
    is never narrower than its longest one."""

    def _rule(self, selector: str, phone: bool = False) -> str:
        css = style.STYLE
        if phone:
            css = css[css.index("@media (max-width: 480px)"):]
        start = css.index(selector + " {")
        return css[start:css.index("}", start)]

    def test_no_send_button_clips_or_refuses_to_wrap(self) -> None:
        for phone in (False, True):
            for selector in (".corr-send button.send-big", ".corr-send button.send-change"):
                rule = self._rule(selector, phone)
                assert "nowrap" not in rule, (selector, phone)
                assert "min-width: 0" not in rule, (selector, phone)
                assert "flex:" in rule, (selector, phone)

    def test_the_slot_beside_the_board_grows_to_its_words(self) -> None:
        beside = self._rule(".board-row.beside > .corr-send")
        assert "height: auto" in beside and "min-width: 0" in beside
        assert "flex-wrap: wrap" in self._rule(".board-row.beside .send-row")
        assert "display: flex" in self._rule(".board-row.beside")


class TestFitBoard:
    def test_a_tall_board_at_a_desk_takes_the_slot_beside_it_and_grows(self) -> None:
        """Torus chess crazyhouse's shape, 1400 x 900: the board was 200 px wide of a
        720 px column with the slot under it."""
        tall = _fit(900, 720, 1.6)
        first = tall["first"]
        assert first["beside"]
        # The slot under it would have left it about 380 px tall; beside, it has the
        # 140 px and the slot's margin back.
        assert first["board"][2] > 500
        assert first["send"] >= 260
        # The strip spans the board and the slot, so it is not narrow.
        assert not first["narrow"] and first["strip"] == 720
        assert first["legend"] <= 900

    def test_a_wide_board_keeps_the_slot_under_it(self) -> None:
        wide = _fit(900, 720, 0.6)["first"]
        assert not wide["beside"]
        assert wide["board"][1] <= 720

    def test_the_slot_is_never_narrower_than_its_buttons_need(self) -> None:
        """A phone's column: the board is narrower than the column, and the slot and the
        strip take the column (they took the board's width, 215 px of 343)."""
        phone = _fit(812, 343, 1.6)["first"]
        assert not phone["beside"]
        assert phone["send"] == 343 and phone["strip"] == 343
        assert phone["narrow"]

    def test_a_square_board_takes_whichever_layout_gives_it_more(self) -> None:
        """In the model, a square board with the slot under it is about 383 px (the
        height holds it); beside, the column's width holds it at about 400."""
        chess = _fit(900, 720, 1.0)["first"]
        assert chess["beside"] and chess["board"][1] > 390
        # A board the column's width holds at its full size keeps the slot under it.
        roomy = _fit(1400, 720, 1.0)["first"]
        assert not roomy["beside"] and roomy["board"][1] > 600

    def test_a_second_fit_moves_nothing(self) -> None:
        """The first fit of a game is read off a panel as wide as the column, whose wood
        is thicker than the fitted board's: it settles before it returns, so the first
        move picked does not shrink the board (Q761's few pixels)."""
        for height, column, ratio, between in (
            (900, 720, 1.6, True), (900, 720, 1.0, True), (900, 720, 0.6, True),
            (812, 343, 1.6, True), (812, 343, 1.0, True), (900, 720, 1.6, False),
        ):
            fit = _fit(height, column, ratio, between)
            assert fit["first"] == fit["again"], (height, column, ratio, between)

    def test_the_strip_is_sized_for_the_layout_it_ends_in(self) -> None:
        """A strip narrow from the last layout (a line taller) no longer decides this one."""
        start_narrow = _in_page(
            _MODEL, "(() => { frameModel(900, 720, 1.0, true);"
            " FIT.status.classList.add('narrow'); fitBoard(); return reading(); })()")
        start_wide = _in_page(_MODEL, "(() => { frameModel(900, 720, 1.0, true);"
                                      " fitBoard(); return reading(); })()")
        assert start_narrow == start_wide

    def test_the_numbers_it_reads_are_the_styles(self) -> None:
        from page_text import PAGE

        assert "const SEND_WIDE = 440;" in PAGE and "const SEND_BESIDE = 260;" in PAGE
        assert "const STRIP_LINE = 22;" in PAGE
        assert "--strip-lines: 2; height: calc(var(--strip-lines) * 22px + 14px);" in style.STYLE


_PICKED = """
function picked(H, C, ratio, between, lines) {
  frameModel(H, C, ratio, between);
  fitBoard();
  const out = [reading()];
  for (const n of lines) {
    FIT.says.hidden = false; FIT.SAYS = n; fitBoard(); out.push(reading());
  }
  FIT.says.hidden = true; fitBoard(); out.push(reading());
  return out;
}
"""


class TestPickingAPiece:
    """Q809: in *play it here* the board changed size when a piece was picked, because the
    piece's sentences under it took two or three lines where one was kept, so a second
    click could land on another square than the one aimed at."""

    def test_the_board_is_the_same_whatever_is_picked(self) -> None:
        for height, column, ratio, between in (
            (640, 400, 1.1, False), (900, 720, 1.6, False), (700, 500, 1.0, False),
            (812, 343, 1.1, True), (900, 720, 1.0, True),
        ):
            readings = _in_page(_MODEL + _PICKED,
                                f"picked({height}, {column}, {ratio}, {str(between).lower()},"
                                " [1, 2, 3, 1])")
            boards = {(tuple(r["board"]), r["width"], r["beside"], r["narrow"]) for r in readings}
            assert len(boards) == 1, (height, column, ratio, between, readings)
            # The sentences push what is under them down; they never pull the board up.
            assert readings[3]["legend"] > readings[0]["legend"]

    def test_the_sentences_own_height_is_never_read(self) -> None:
        from page_text import PAGE

        start = PAGE.index("function fitBoard(")
        body = PAGE[start:PAGE.index("\nfunction ", start + 1)]
        assert "says.hidden ? PIECE_SAYS_LINE : 0" not in body
        assert "+ 12 + PIECE_SAYS_LINE" in body

    def test_the_sentences_are_held_to_the_line_kept_for_them(self) -> None:
        """The line kept is one line, so the sentences are one line until the reader taps
        them open (`sayMore`): a three-line sentence no longer pushes the legend off a desk's
        first screen when a pawn is picked."""
        css = style.STYLE
        start = css.index(".piece-says {")
        rule = css[start:css.index("}", start)]
        assert "-webkit-line-clamp: 1" in rule and "overflow: hidden" in rule
        assert ".piece-says.open { display: block; -webkit-line-clamp: none; }" in css
        # `[hidden]` still wins over the open line's `display`.
        assert css.index(".piece-says.open {") < css.index(".piece-says[hidden] { display: none; }")

    def test_a_cut_sentence_opens_and_shuts_on_a_tap(self) -> None:
        said = _in_page("""
          const attrs = {};
          const names = new Set();
          const line = { hidden: false, textContent: 'pawn: one square forward; two from rank 2',
            scrollHeight: 54, clientHeight: 18,
            classList: { add: n => names.add(n), remove: n => names.delete(n), contains: n => names.has(n),
              toggle: (n, on) => { if (on) names.add(n); else names.delete(n); return on; } },
            setAttribute: (k, v) => { attrs[k] = v; }, removeAttribute: k => { delete attrs[k]; } };
          const read = () => ({ more: names.has('more'), open: names.has('open'),
                                expanded: attrs['aria-expanded'] || null, role: attrs.role || null });
          sayMore(line);
          const cut = read();
          line.onclick();
          const opened = read();
          line.onclick();
          const shut = read();
          line.scrollHeight = 18; sayMore(line);
          const whole = read();
        """, "[cut, opened, shut, whole]")
        cut, opened, shut, whole = said
        assert cut == {"more": True, "open": False, "expanded": "false", "role": "button"}
        assert opened == {"more": True, "open": True, "expanded": "true", "role": "button"}
        assert shut == cut
        assert whole == {"more": False, "open": False, "expanded": None, "role": None}


class TestTheFloorGivesALine:
    """Q809: a board ten wide and eleven tall between two people on a 375 x 812 phone was
    held at `BOARD_FLOOR` with its legend's last row 4 px under the first screen. Where the
    frame needs no more than the line kept for a tapped piece below the floor, the board
    gives that much; further off, it keeps the floor and the page scrolls."""

    def _board(self, height: int, column: int, ratio: float, between: bool) -> dict[str, Any]:
        return _fit(height, column, ratio, between)["first"]

    def test_a_tall_board_just_off_the_screen_gives_way(self) -> None:
        # In the model, a 1.1 board between two people in a phone's column stands at the
        # floor (280 px) with its legend ending at 783: a screen of 780 is a few pixels
        # short of that, as the 10 x 11 board's 812 was.
        gave = self._board(780, 343, 1.1, True)
        assert 258 <= gave["board"][2] < 280
        assert gave["legend"] <= 780
        # ... and a frame further off than one line keeps the floor and scrolls.
        for height in (740, 700):
            far = self._board(height, 343, 1.1, True)
            assert 280 <= far["board"][2] < 281 and far["legend"] > height

    def test_a_board_that_fits_is_not_touched(self) -> None:
        for height, column, ratio, between in ((812, 343, 1.0, True), (812, 343, 1.0, False),
                                                (900, 720, 1.0, False), (812, 343, 1.6, True)):
            board = self._board(height, column, ratio, between)
            assert board["board"][2] >= 280 or board["legend"] <= height, (height, column, ratio)
