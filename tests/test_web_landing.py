"""Where a strike's survivor goes, chosen on a card over the board (Q769).

A strike under `position: "choose"` takes the square and places the piece it struck on an
empty square of the mover's choosing: one move per landing square, and, when the striker
promotes as it takes, one per landing square *and* piece. That choice was a row of small
buttons under the board (*place it on:*, `b7 =Q`), the new piece named by its letter. It
is the promotion's card now (Q765): the piece first, drawn large and named, then the
square, each a large target; *Look at the board*, Cancel, Escape and the keys as there.

The positions are a real game's: `archive/hierarchy-rise-throw.json` walked by the
engine's own move texts to a pawn on d7 that can take the bishop on c8, promoting, and
throw it (a rook now) onto b7 or d7. The snapshot is the server's, so the moves the card
offers are the engine's; nothing here works out a legal move.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_web import _in_page

from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web.library import Game, engine_for
from chess_variant_sandbox.web.play import Session

ROOT = Path(__file__).resolve().parent.parent
THROW = ROOT / "archive" / "hierarchy-rise-throw.json"
# Found by a seeded random walk of the engine's own moves (Q769); each is a move text the
# engine writes, so a change to the game or the generator fails here, not silently.
WALK = ("Pc2-c4", "Pe7-e5", "Pc4-c5", "Ke8-e7", "Pc5-c6", "Ng8-f6",
        "Pc6xd7->N (placed on d6)", "Pb7-b6")


def _session(plies: int) -> Session:
    spec = json.loads(THROW.read_text(encoding="utf-8"))
    game = Game(hash=spec_hash(spec), name=str(spec["name"]),
                description=str(spec.get("description", "")), path=THROW,
                source="archived", spec=spec)
    session = Session(id="g1", game=game, seats=("human", "medium"), seed=0,
                      state=engine_for(spec).initial_state())
    for text in WALK[:plies]:
        (move,) = [m["move"] for m in session.snapshot()["moves"] if m["text"] == text]
        session.play(move)
    return session


_STRIKE: dict[str, Any] = _session(len(WALK)).snapshot()
_THROW_ONLY: dict[str, Any] = _session(6).snapshot()


def _pair(snapshot: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    return [m for m in snapshot["moves"] if m["text"].startswith(prefix)]


class TestAGameReachesIt:
    def test_a_pawn_taking_on_the_last_rank_chooses_a_piece_and_a_square(self) -> None:
        moves = _pair(_STRIKE, "Pd7xc8")
        assert len(moves) == 8
        assert all("landing" in m and "promotion" in m for m in moves)
        assert sorted({m["landingName"] for m in moves}) == ["b7", "d7"]
        assert len({m["promotion"] for m in moves}) == 4
        assert len({(m["from"], m["to"]) for m in moves}) == 1

    def test_a_strike_with_no_promotion_chooses_the_square_alone(self) -> None:
        moves = _pair(_THROW_ONLY, "Pc6xd7")
        assert len(moves) > 1 and all("promotion" not in m for m in moves)
        assert all("landing" in m for m in moves)


_HARNESS = """
const CALLS = [];
api = (path, body) => { CALLS.push([path, body]); return new Promise(() => {}); };
drawBoard = () => {};
drawStatus = () => {};
const button = (attrs) => ({ id: attrs.id || null, dataset: attrs.dataset,
  focus() { document.activeElement = this; }, click() { this.onclick(); } });
// The choice's markup read back as its buttons, in order.
const BOX = { hidden: true, html: '', buttons: [], looking: false, modal: null,
  classList: { toggle: (name, on) => { if (name === 'looking') BOX.looking = !!on; } },
  setAttribute(name, value) { if (name === 'aria-modal') this.modal = value; },
  set innerHTML(v) { this.html = v;
    this.buttons = [...v.matchAll(/<button([^>]*)>/g)].map(m => {
      const move = /data-move="(\\d+)"/.exec(m[1]), piece = /data-piece="(\\d+)"/.exec(m[1]);
      const id = /id="(\\w+)"/.exec(m[1]), which = /data-which="(\\w+)"/.exec(m[1]);
      const dataset = move ? { move: move[1] } : piece ? { piece: piece[1] }
        : which ? { which: which[1] } : {};
      return button({ id: id && id[1], dataset }); }); },
  get innerHTML() { return this.html; },
  get picks() { return this.buttons.filter(b => b.dataset.move !== undefined
    || b.dataset.piece !== undefined || b.dataset.which !== undefined); },
  byId(id) { return this.buttons.find(b => b.id === id) || null; },
  querySelectorAll(sel) {
    return sel === '[data-move]' ? this.buttons.filter(b => b.dataset.move !== undefined)
      : sel === '[data-piece]' ? this.buttons.filter(b => b.dataset.piece !== undefined)
      : sel === '[data-which]' ? this.buttons.filter(b => b.dataset.which !== undefined)
      : sel === 'button' ? this.buttons : []; },
  querySelector(sel) { return sel === '[data-move]' || sel === '[data-piece]' || sel === '[data-which]'
    ? this.querySelectorAll(sel)[0] || null
    : sel === '.promo-card' || sel === '.promo-bar' ? {} : null; },
  contains(node) { return this.buttons.includes(node); } };
const HAND = { innerHTML: 'untouched' };
document.getElementById = id => (id === 'promo' ? BOX : id === 'hand' ? HAND : BOX.byId(id));
document.activeElement = null;
const key = (k) => { const e = { key: k, shiftKey: false, defaultPrevented: false, stopped: false,
  preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; } };
  promotionKeys(e); return e; };
const said = () => BOX.html.replace(/<[^>]+>/g, ' ').replace(/\\s+/g, ' ').trim();
const pieceNamed = (name) => BOX.buttons.find(b => b.dataset.piece !== undefined
  && SESSION.ruleset.pieceNames[Number(b.dataset.piece)] === name);
const textOf = (move) => SESSION.moves.find(m => m.move === move).text;
"""

_BETWEEN = """
SESSION.seats = ['human', 'human'];
SESSION.correspondence = { you: 0, players: ['ada', 'bo'], drawOffer: null, mayCancel: false,
  times: [new Date().toISOString()], rules: [] };
"""


def _run(steps: str, expression: str, snapshot: dict[str, Any] = _STRIKE,
         prefix: str = "Pd7xc8", between: bool = False) -> Any:
    (move,) = {(m["from"], m["to"]) for m in _pair(snapshot, prefix)}
    setup = (f"SESSION = {json.dumps(snapshot)};\n" + _HARNESS + (_BETWEEN if between else "")
             + f"const open = () => {{ clickSquare({move[0]}); clickSquare({move[1]}); }};\n")
    return _in_page(setup + steps, expression)


class TestThePieceComesFirst:
    def test_it_is_a_card_over_the_board_in_plain_words(self) -> None:
        hidden, hand, text, html = _run("open();", "[BOX.hidden, HAND.innerHTML, said(), BOX.html]")
        assert hidden is False and hand == "untouched"
        assert text.startswith("Choose what your pawn becomes")
        assert "Tap the piece you want. Then place the bishop you took." in text
        # Each piece once, drawn and named; no letter standing for a piece.
        flat = " ".join(html.split())
        assert flat.count('class="promo-pick"') == 4 and flat.count('class="promo-piece"') == 4
        for name in ("queen", "rook", "bishop", "knight"):
            assert f'aria-label="{name}"' in flat
        assert "=Q" not in text and "place it on" not in text
        assert "place it on:" not in _in_page("", "chooseLanding.toString()")

    def test_the_first_piece_takes_the_focus_and_the_board_takes_no_tap(self) -> None:
        assert _run("open();", "document.activeElement === BOX.picks[0]") is True
        assert _run("open(); open();", "[CALLS.length, PROMO.length]") == [0, 8]


class TestThenTheSquare:
    def test_a_piece_picked_asks_for_the_square(self) -> None:
        stage, text, picks, focus, calls = _run(
            "open(); pieceNamed('queen').click();",
            "[promotionStage(), said(), BOX.picks.map(b => textOf(Number(b.dataset.move))),"
            " document.activeElement === BOX.picks[0], CALLS]")
        assert stage == "square" and calls == []
        assert text.startswith("Place the bishop you took")
        assert "Your pawn becomes a queen. Tap a square." in text
        assert "Change the piece" in text and "Look at the board" in text and "Cancel" in text
        assert [t[-2:] for t in picks] == ["=Q", "=Q"]
        assert sorted(t.split("placed on ")[1][:2] for t in picks) == ["b7", "d7"]
        assert focus is True

    def test_each_square_is_a_large_named_target(self) -> None:
        html = " ".join(_run("open(); pieceNamed('rook').click();", "BOX.html").split())
        assert html.count('class="promo-pick promo-square"') == 2
        assert 'aria-label="square b7"' in html and '<span class="promo-sq">d7</span>' in html
        from chess_variant_sandbox.web.page import style
        rule = style.RULES[style.RULES.index(".promo-sq {"):]
        rule = rule[:rule.index("}")]
        assert "width: 72px" in rule and "height: 72px" in rule

    def test_the_square_picked_is_the_move(self) -> None:
        calls, box, promo, piece = _run(
            "open(); pieceNamed('knight').click(); BOX.picks[1].click();",
            "[CALLS, BOX.hidden, PROMO, PROMO_PIECE]")
        assert box is True and promo is None and piece is None
        ((path, body),) = calls
        assert path == "/api/session/g1/move"
        (text,) = [m["text"] for m in _STRIKE["moves"] if m["move"] == body["move"]]
        assert text.startswith("Pd7xc8") and text.endswith("=N")

    def test_change_the_piece_goes_back_a_step(self) -> None:
        stage, piece, text = _run(
            "open(); pieceNamed('queen').click(); BOX.byId('promoPiece').click();",
            "[promotionStage(), PROMO_PIECE, said()]")
        assert stage == "piece" and piece is None
        assert text.startswith("Choose what your pawn becomes")

    def test_one_square_left_makes_the_piece_the_move(self) -> None:
        calls = _run("open(); PROMO = PROMO.filter(m => m.landingName === 'b7'); drawPromotion();"
                     "pieceNamed('bishop').click();", "CALLS.map(c => textOf(c[1].move))")
        assert calls == ["Pd7xc8->R (placed on b7)=B"]

    def test_between_two_people_it_is_held_for_send(self) -> None:
        text, held, calls = _run(
            "open(); const first = said(); pieceNamed('queen').click(); const second = said();"
            "BOX.picks[0].click();", "[[first, second], CORR_HELD, CALLS]", between=True)
        assert all("You can still change your move until you send it." in t for t in text)
        assert held is not None and calls == []


class TestAStrikeThatDoesNotPromote:
    def test_it_asks_for_the_square_alone(self) -> None:
        stage, text, picks = _run(
            "open();", "[promotionStage(), said(), BOX.picks.length]",
            snapshot=_THROW_ONLY, prefix="Pc6xd7")
        assert stage == "square"
        assert text.startswith("Place the pawn you took")
        assert "Tap a square." in text and "becomes" not in text and "Change the piece" not in text
        assert picks == len(_pair(_THROW_ONLY, "Pc6xd7"))


class TestItsKeysAndTheBoard:
    def test_arrows_walk_the_pieces_then_the_squares(self) -> None:
        walked = _run(
            "open(); const e = key('ArrowRight'); const a = document.activeElement === BOX.picks[1];"
            "key('ArrowLeft'); key('ArrowLeft'); const b = document.activeElement === BOX.picks[3];"
            "pieceNamed('queen').click(); key('ArrowRight');"
            "const c = document.activeElement === BOX.picks[1];",
            "[e.stopped, a, b, c, AT]")
        assert walked == [True, True, True, True, None]

    def test_tab_stays_inside_the_card(self) -> None:
        cycled = _run(
            "open(); pieceNamed('queen').click(); const seen = [];"
            "for (let n = 0; n < 5; n++) { key('Tab'); seen.push(document.activeElement.id"
            " || 'square'); }", "seen")
        assert cycled == ["square", "promoPiece", "promoLook", "promoCancel", "square"]

    def test_escape_or_cancel_on_the_square_leaves_nothing_moved(self) -> None:
        for how in ("key('Escape');", "BOX.byId('promoCancel').click();"):
            gone = _run("open(); pieceNamed('queen').click();" + how,
                        "[BOX.hidden, PROMO, PROMO_PIECE, SELECTED, CALLS]")
            assert gone == [True, None, None, None, []], how

    def test_looking_at_the_board_keeps_the_piece_picked(self) -> None:
        looked = _run(
            "open(); pieceNamed('queen').click(); BOX.byId('promoLook').click();"
            "const bar = [BOX.looking, BOX.modal, said(), document.activeElement.id];"
            "BOX.byId('promoBack').click();",
            "[bar, promotionStage(), PROMO_PIECE !== null, BOX.looking, CALLS]")
        bar, stage, kept, looking, calls = looked
        assert bar[0] is True and bar[1] == "false" and bar[3] == "promoBack"
        assert "The bishop you took is waiting to be placed." in bar[2]
        assert "Choose the square" in bar[2] and "Cancel" in bar[2]
        assert stage == "square" and kept and looking is False and calls == []

    def test_looking_before_the_piece_says_the_promotion_s_words(self) -> None:
        text = _run("open(); BOX.byId('promoLook').click();", "said()")
        assert "Your pawn is waiting to become a new piece." in text
        assert "Choose the new piece" in text

    def test_a_position_the_game_left_puts_it_away(self) -> None:
        assert _run("open(); pieceNamed('queen').click();", "promotionStands()") is True
        assert _run("open(); pieceNamed('queen').click(); SESSION.moves = [];",
                    "promotionStands()") is False
        assert _run("open(); pieceNamed('queen').click(); leaveBoard();",
                    "[PROMO, PROMO_PIECE]") == [None, None]
