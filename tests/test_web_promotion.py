"""The choice of what a promoting piece becomes, drawn over the board (Q765).

Two members each took the game for stalled when a pawn reached the far side: the choice was
a row of small buttons under the board, out of sight on a phone. It is a card over the board
now, the board dimmed under it, in plain words, with the pieces drawn large; it takes the
keyboard, and a pick is the move -- played against the machine, held for *Send your move*
between two people (Q733, Q740). *Look at the board* puts it aside to see the whole
position with the move still not made, and a bar under the board brings it back.

Run in node over a hand-made position on a one-file board: a pawn of the first player on
square 1 whose only moves are to square 0, as a queen or as a knight. The moves are the
snapshot's, as the engine would send them; nothing here works out a legal move.
"""

from __future__ import annotations

from typing import Any

from test_web import _in_page

from chess_variant_sandbox.web.page import style, tokens

_BOARD = """
SESSION = { id: 'g1', ruleset: { width: 1, height: 2, nTypes: 3, pieceIds: ['P', 'Q', 'N'],
    pieceNames: ['pawn', 'queen', 'knight'], icons: [] },
  seats: ['human', 'medium'], sideToMove: 0, inCheck: false, outcome: null, awaitingAgent: false,
  board: [0, 1], history: [{ player: 1, text: 'b2-b1' }],
  moves: [{ move: 5, from: 1, to: 0, promotion: 1, text: 'a1-a2=Q' },
          { move: 6, from: 1, to: 0, promotion: 2, text: 'a1-a2=N' }] };
const CALLS = [];
api = (path, body) => { CALLS.push([path, body]); return new Promise(() => {}); };
drawBoard = () => {};
drawStatus = () => {};
const button = (move, id) => ({ id, dataset: move === null ? {} : { move: String(move) },
  focus() { document.activeElement = this; }, click() { this.onclick(); } });
// The choice's markup read back as its buttons, in order: a piece by its move, the rest by id.
const BOX = { hidden: true, html: '', buttons: [], looking: false, modal: null,
  classList: { toggle: (name, on) => { if (name === 'looking') BOX.looking = !!on; } },
  setAttribute(name, value) { if (name === 'aria-modal') this.modal = value; },
  set innerHTML(v) { this.html = v;
    this.buttons = [...v.matchAll(/<button([^>]*)>/g)].map(m => {
      const move = /data-move="(\\d+)"/.exec(m[1]), id = /id="(\\w+)"/.exec(m[1]);
      return button(move ? Number(move[1]) : null, id ? id[1] : null); }); },
  get innerHTML() { return this.html; },
  get picks() { return this.buttons.filter(b => b.dataset.move !== undefined); },
  get cancel() { return this.byId('promoCancel'); },
  byId(id) { return this.buttons.find(b => b.id === id) || null; },
  querySelectorAll(sel) { return sel === '[data-move]' ? this.picks : sel === 'button' ? this.buttons : []; },
  querySelector(sel) { return sel === '[data-move]' ? this.picks[0] || null
    : sel === '.promo-card' || sel === '.promo-bar' ? {} : null; },
  contains(node) { return this.buttons.includes(node); } };
const HAND = { innerHTML: 'untouched' };
document.getElementById = id => (id === 'promo' ? BOX : id === 'hand' ? HAND : BOX.byId(id));
document.activeElement = null;
const key = (k) => { const e = { key: k, shiftKey: false, defaultPrevented: false, stopped: false,
  preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; } };
  promotionKeys(e); return e; };
const open = () => { clickSquare(1); clickSquare(0); };
"""

# Between two people: the reader is the first player, to move, and nothing is held yet.
_BETWEEN = """
SESSION.seats = ['human', 'human'];
SESSION.correspondence = { you: 0, players: ['ada', 'bo'], drawOffer: null, mayCancel: false,
  times: [new Date().toISOString()], rules: [] };
"""


def _run(steps: str, expression: str, between: bool = False) -> Any:
    return _in_page(_BOARD + (_BETWEEN if between else "") + steps, expression)


class TestTheChoiceIsOverTheBoard:
    def test_the_frame_carries_it_over_the_board_hidden(self) -> None:
        frame = _in_page("", "frameHtml(false, 'sandbox', false, null)")
        assert 'id="promo"' in frame and 'role="dialog"' in frame and 'aria-modal="true"' in frame
        # Inside the board's frame (over the board), not under it with the hand.
        promo = frame.index('id="promo"')
        assert frame.index('id="board"') < promo < frame.index('id="hand"')
        assert frame[promo:frame.index(">", promo)].rstrip().endswith("hidden")

    def test_a_pawn_on_its_square_opens_the_choice_in_plain_words(self) -> None:
        box, hand, promo = _run("open();", "[BOX, HAND.innerHTML, PROMO.map(m => m.move)]")
        assert not box["hidden"] and promo == [5, 6]
        html = " ".join(box["html"].split())
        assert "Choose what your pawn becomes" in html
        assert "Tap the piece you want." in html
        # One big target per piece the engine offers, each drawn and named.
        assert html.count('class="promo-pick"') == 2
        assert 'aria-label="queen"' in html and 'aria-label="knight"' in html
        assert html.count('class="promo-piece"') == 2 and ">Q</text>" in html
        assert 'id="promoCancel"' in html and ">Cancel</button>" in html
        # Nothing is written under the board any more.
        assert hand == "untouched"
        assert "promote to:" not in _in_page("", "choosePromotion.toString()")

    def test_the_first_piece_takes_the_focus(self) -> None:
        assert _run("open();", "document.activeElement === BOX.picks[0]") is True

    def test_the_board_takes_no_tap_while_it_is_open(self) -> None:
        assert _run("open(); clickSquare(1); clickSquare(0);", "[CALLS.length, SELECTED]") == [0, 1]


class TestAPickIsTheMove:
    def test_against_the_machine_the_piece_picked_is_played(self) -> None:
        calls, box, promo = _run("open(); BOX.picks[1].click();", "[CALLS, BOX.hidden, PROMO]")
        assert calls == [["/api/session/g1/move", {"move": 6}]]
        assert box is True and promo is None

    def test_between_two_people_it_is_held_for_send(self) -> None:
        said = _run("open();", "BOX.html", between=True)
        assert "You can still change your move until you send it." in said
        held, calls, promo = _run("open(); BOX.picks[0].click();", "[CORR_HELD, CALLS, PROMO]",
                                  between=True)
        assert held == 5 and calls == [] and promo is None


class TestItsKeys:
    def test_escape_puts_it_away_with_nothing_moved(self) -> None:
        stopped, prevented, box, promo, selected, calls = _run(
            "open(); const e = key('Escape');",
            "[e.stopped, e.defaultPrevented, BOX.hidden, PROMO, SELECTED, CALLS]")
        assert stopped and prevented and box is True and promo is None
        assert selected is None and calls == []

    def test_cancel_does_the_same(self) -> None:
        assert _run("open(); BOX.cancel.click();", "[BOX.hidden, PROMO, CALLS]") == [True, None, []]

    def test_arrows_walk_the_pieces_and_never_step_the_game(self) -> None:
        walked = _run(
            "open(); const e = key('ArrowRight'); const second = document.activeElement;"
            "key('ArrowRight');",
            "[e.stopped, second === BOX.picks[1], document.activeElement === BOX.picks[0], AT]")
        assert walked == [True, True, True, None]

    def test_tab_stays_inside_the_card(self) -> None:
        cycled = _run(
            "open(); const seen = [];"
            "for (let n = 0; n < 4; n++) { key('Tab'); seen.push(document.activeElement.id"
            " || document.activeElement.dataset.move); }",
            "seen")
        assert cycled == ["6", "promoLook", "promoCancel", "5"]

    def test_escape_on_the_choice_is_not_also_a_take_back(self) -> None:
        """A move already held between two people stays held when Escape puts the choice
        away; the page's Escape for a held move reads the event the choice took."""
        steps = """
        const HEARD = {};
        window.addEventListener = (name, listen) => { HEARD[name] = listen; };
        document.getElementById = id => (id === 'promo' ? BOX : id === 'corrSend' ? {}
          : id === 'hand' ? HAND : BOX.byId(id));
        wireHeldKeys();
        CORR_HELD = 5;
        PROMO = SESSION.moves;
        const e = key('Escape');
        HEARD.keydown(Object.assign(e, { target: { tagName: 'BUTTON' } }));
        """
        assert _run(steps, "CORR_HELD", between=True) == 5


class TestLookingAtTheBoardWhileChoosing:
    """The user, after the first cut: people need the whole board to think about their
    choice. *Look at the board* puts the card and the dimming away with the move not made,
    and a bar under the board brings the choice back or cancels it."""

    def test_the_card_offers_a_look_at_the_board(self) -> None:
        html = " ".join(_run("open();", "BOX.html").split())
        assert 'id="promoLook"' in html and ">Look at the board</button>" in html

    def test_looking_shows_the_board_with_nothing_moved(self) -> None:
        looked = _run(
            "open(); BOX.byId('promoLook').click();",
            "[BOX.looking, BOX.modal, BOX.hidden, PROMO_LOOKING, PROMO.length, CALLS, SELECTED,"
            " document.activeElement.id, BOX.html]")
        looking, modal, hidden, flag, offered, calls, selected, focus, html = looked
        # The dimming goes (`.promo.looking`), the choice stays open, the move is not made.
        assert looking and modal == "false" and not hidden and flag and offered == 2
        assert calls == [] and selected == 1
        said = " ".join(html.split())
        assert "Your pawn is waiting to become a new piece." in said
        assert ">Choose the new piece</button>" in said and ">Cancel</button>" in said
        assert 'class="promo-card"' not in said and "data-move" not in said
        assert focus == "promoBack"
        # The board still takes no tap while the move waits for its piece.
        assert _run("open(); BOX.byId('promoLook').click(); clickSquare(1); clickSquare(0);",
                    "[CALLS.length, PROMO_LOOKING]") == [0, True]
        css = style.RULES[style.RULES.index(".promo.looking {"):]
        assert "background: transparent" in css[:css.index("}")]

    def test_the_bar_brings_the_choice_back(self) -> None:
        back = _run("open(); BOX.byId('promoLook').click(); BOX.byId('promoBack').click();",
                    "[BOX.looking, BOX.modal, PROMO_LOOKING, document.activeElement === BOX.picks[0]]")
        assert back == [False, "true", False, True]
        # ... and a pick from it is still the move.
        assert _run("open(); BOX.byId('promoLook').click(); BOX.byId('promoBack').click();"
                    "BOX.picks[0].click();", "CALLS") == [["/api/session/g1/move", {"move": 5}]]

    def test_escape_or_cancel_while_looking_leaves_the_move_unmade(self) -> None:
        for how in ("key('Escape');", "BOX.byId('promoCancel').click();"):
            gone = _run("open(); BOX.byId('promoLook').click();" + how,
                        "[BOX.hidden, BOX.looking, PROMO, PROMO_LOOKING, SELECTED, CALLS]")
            assert gone == [True, False, None, False, None, []], how
        # Between two people nothing is held either.
        assert _run("open(); BOX.byId('promoLook').click(); key('Escape');",
                    "[CORR_HELD, CALLS]", between=True) == [None, []]

    def test_the_arrows_never_step_the_game_while_looking(self) -> None:
        stepped = _run("open(); BOX.byId('promoLook').click(); const e = key('ArrowLeft');",
                       "[e.stopped, AT, PROMO_LOOKING]")
        assert stepped == [True, None, True]


class TestItBelongsToThePosition:
    def test_a_position_the_game_left_puts_it_away(self) -> None:
        for change in ("AT = { board: [0, 1] };", "SESSION.moves = [];",
                       "SESSION.outcome = { winner: 1, reason: 'resigned' };"):
            assert _run("open();" + change, "promotionStands()") is False, change
        assert _run("open();", "promotionStands()") is True

    def test_leaving_the_board_forgets_it(self) -> None:
        assert _run("open(); leaveBoard();", "PROMO") is None


class TestItsLook:
    def test_the_board_is_dimmed_under_a_card_in_both_themes(self) -> None:
        assert "scrim" in tokens.LIGHT and "scrim" in tokens.DARK
        rules = style.RULES
        start = rules.index(".promo {")
        assert "background: var(--scrim)" in rules[start:rules.index("}", start)]
        assert "inset: 0" in rules[start:rules.index("}", start)]
        assert ".promo[hidden] { display: none; }" in rules
