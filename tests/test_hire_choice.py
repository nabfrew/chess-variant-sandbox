"""A hire or a bribe is never mis-clicked as a capture (Q1071).

The user, 2026-10-30: with a piece picked that attacks a mercenary, a tap meant to hire it
captured it, and the mercenary's disc looked like the ring a capturable piece wears. What is
held here, by behaviour, on the page's own functions over the core's snapshots:

- **A square offering two kinds of move asks.** A tap there plays nothing; it opens a small
  choice at the square, one button a kind (the capture, the hire or bribe), each holding the
  engine's own moves, and the button pressed sends that move. Cancel sends nothing. A tap on
  a piece whose only move there is a hire asks too, with the hire alone.
- **A hire has its own mark.** Every square the engine offers a hire or bribe on, for a person
  to move, wears a coin with the cost in it, named in the legend; a game without a purse and
  a position the person is not moving in draw none.
- **The mercenary is not a ring.** A hireable piece nobody owns stands on a shield; every
  other wild piece keeps the disc.
- **The purses are under the board.** Both players' coins are on a line in the strip under
  the board, which the style shows where the seats' panel is not beside the board.
"""

from __future__ import annotations

import re
from typing import Any

from test_sight_page import _drawn
from test_turncoat_page import _HAND, BRIBE, _session, _spec, _square

from chess_variant_sandbox.web.page.style import RULES

# The choice's box, as far as `drawKinds` uses it: its markup, and a node per `data-kind`
# button whose `onclick` the wiring sets (`KIND_NODES`), so a test can press one.
_PROMO = r"""
let PROMO_HTML = ''; const KIND_NODES = [];
const PROMO_BOX = { hidden: true, classList: { toggle() {} }, setAttribute() {},
  set innerHTML(v) { PROMO_HTML = v; }, get innerHTML() { return PROMO_HTML; },
  querySelector: () => null,
  querySelectorAll(selector) {
    if (selector !== '[data-kind]') return [];
    const found = [], pattern = /data-kind="(\d+)"/g;
    let hit;
    while ((hit = pattern.exec(PROMO_HTML))) {
      const node = { dataset: { kind: hit[1] } };
      KIND_NODES.push(node);
      found.push(node);
    }
    return found;
  } };
const CANCEL = { onclick: null };
const PROMO_GET = document.getElementById;
document.getElementById = id => id === 'promo' ? PROMO_BOX : id === 'promoCancel' ? CANCEL
  : PROMO_GET(id);
let SENT = null; send = m => { SENT = m; };
"""


def _page(snapshot: dict[str, Any], steps: str, expression: str) -> Any:
    return _drawn(snapshot, _HAND + _PROMO + steps, expression)


def _bribe_snapshot() -> dict[str, Any]:
    """White to move with 9 coins: the pawn on d2 may take the black knight on c3, and the
    pawns beside it may bribe it for 8."""
    return _session(_spec(), BRIBE, purse=(9, 0)).snapshot()


class TestTheChoice:
    def test_a_square_with_a_capture_and_a_bribe_asks_and_plays_nothing(self) -> None:
        snap = _bribe_snapshot()
        pawn, knight = _square("d2"), _square("c3")
        capture = [m for m in snap["moves"] if m["from"] == pawn and m["to"] == knight]
        bribe = next(m for m in snap["moves"] if m["text"] == "$Nc3")
        assert capture and "hire" not in capture[0], "the position offers both kinds"
        steps = f"clickSquare({pawn}); clickSquare({knight});"
        sent, kinds, moves, html = _page(snap, steps, (
            "[SENT, PROMO_KIND.options.map(o => o.kind),"
            " PROMO_KIND.options.map(o => o.moves.map(m => m.move)), PROMO_HTML]"))
        assert sent is None, "a tap on a square with two kinds of move plays neither"
        assert kinds == ["capture", "bribe"]
        assert moves == [[m["move"] for m in capture], [bribe["move"]]]
        assert 'data-kind-is="capture"' in html and 'data-kind-is="bribe"' in html
        assert "Bribe for 8 coins" in html and bribe["text"] in html
        assert "Take the knight, or bribe it?" in html
        # The capture's button shows the board's ring, the bribe's the coin with its cost.
        assert 'data-mark="hire" data-cost="8" data-bribe="1"' in html

    def test_each_button_sends_its_own_move_and_cancel_sends_none(self) -> None:
        snap = _bribe_snapshot()
        pawn, knight = _square("d2"), _square("c3")
        bribe = next(m for m in snap["moves"] if m["text"] == "$Nc3")
        capture = next(m for m in snap["moves"] if m["from"] == pawn and m["to"] == knight)
        tap = f"clickSquare({pawn}); clickSquare({knight});"
        assert _page(snap, tap + "KIND_NODES[1].onclick();", "SENT") == bribe["move"]
        assert _page(snap, tap + "KIND_NODES[0].onclick();", "SENT") == capture["move"]
        said = _page(snap, tap + "CANCEL.onclick();", "[SENT, PROMO, PROMO_KIND, SELECTED]")
        assert said == [None, None, None, None]

    def test_a_tap_on_a_piece_whose_only_move_is_a_hire_asks_with_the_hire(self) -> None:
        snap = _bribe_snapshot()
        condottiere = _square("e3")
        hire = next(m for m in snap["moves"] if m["text"] == "$Ce3")
        kinds, sent = _page(snap, f"clickSquare({condottiere});",
                            "[PROMO_KIND.options.map(o => [o.kind, o.words]), SENT]")
        assert kinds == [["hire", "Hire for 3 coins"]] and sent is None
        assert _page(snap, f"clickSquare({condottiere}); KIND_NODES[0].onclick();",
                     "SENT") == hire["move"]

    def test_a_plain_capture_is_still_one_tap(self) -> None:
        snap = _session(_spec()).snapshot()
        hired = {m["from"] for m in snap["moves"] if "hire" in m}
        move = next(m for m in snap["moves"] if m["to"] not in hired and "hire" not in m
                    and "buy" not in m and m["from"] is not None)
        steps = f"clickSquare({move['from']}); clickSquare({move['to']});"
        sent, open_ = _page(snap, steps, "[SENT, PROMO_KIND]")
        assert open_ is None
        assert sent in {m["move"] for m in snap["moves"]
                        if m["from"] == move["from"] and m["to"] == move["to"]}


class TestTheMarks:
    def test_every_hire_square_wears_a_coin_with_its_cost(self) -> None:
        snap = _bribe_snapshot()
        offered: dict[int, int] = {}
        for m in snap["moves"]:
            if "hire" in m:
                offered.setdefault(m["from"], m["cost"])
        assert len(offered) >= 2, "a hire and a bribe on offer"
        drawn = _drawn(snap, "", "DRAWN")
        coins = re.findall(r'data-mark="hire" data-cost="(\d+)"', drawn)
        assert sorted(int(c) for c in coins) == sorted(offered.values())
        legend = " ".join(_drawn(snap, "", "legendEntries(new Set(['hire']), null, '', null)"))
        assert 'data-legend="hire"' in legend and "gold coin" in legend

    def test_no_coin_where_nothing_may_be_hired_by_the_person_to_move(self) -> None:
        snap = _bribe_snapshot()
        machine = dict(snap, seats=["greedy", "human"])
        assert "data-mark=\"hire\"" not in _drawn(machine, "", "DRAWN")
        other = _session(_spec("command_k1")).snapshot()
        assert "data-mark=\"hire\"" not in _drawn(other, "", "DRAWN")

    def test_a_hireable_piece_nobody_owns_stands_on_a_shield_not_a_disc(self) -> None:
        shield, disc = _drawn(_bribe_snapshot(), "",
                              "[wildBase(5, 5, 10, true), wildBase(5, 5, 10, false)]")
        assert "<path" in shield and "data-hired" in shield and "<circle" not in shield
        assert "<circle" in disc and "data-hired" not in disc
        drawn = _drawn(_bribe_snapshot(), "", "DRAWN")
        assert 'data-hired="1"' in drawn, "the condottiere nobody owns is on the shield"
        flags = _drawn(_bribe_snapshot(), "",
                       "SESSION.ruleset.pieceIds.map((_, t) => hireable(SESSION.ruleset, t))")
        rules = _bribe_snapshot()["ruleset"]
        assert flags == [cost > 0 for cost in rules["purse"]["hire"]]
        # The compiled hire table, not a price: a pawn sold in the market is no mercenary.
        ids = rules["pieceIds"]
        assert flags[ids.index("C")] and not flags[ids.index("P")]


class TestThePurses:
    def test_both_purses_are_on_a_line_in_the_strip_under_the_board(self) -> None:
        snap = _bribe_snapshot()
        hand = _drawn(snap, _HAND, "HAND_HTML")
        line = re.search(r'<span class="purse-line" data-purse-line.*?</span></span>', hand, re.S)
        assert line and hand.startswith('<span class="purse-line"'), "first in the strip"
        text = re.sub(r"<[^>]+>|\s+", " ", line.group(0))
        assert re.search(r"white\s+9 coins", text) and re.search(r"black\s+0 coins", text)
        other = _session(_spec("command_k1")).snapshot()
        assert _drawn(other, "", "purseLineHtml()") == ""

    def test_the_line_shows_where_the_seats_panel_is_not_beside_the_board(self) -> None:
        assert ".hand .purse-line { display: none;" in RULES
        assert "@media (max-width: 860px) { .hand .purse-line { display: flex; } }" in RULES
        assert "html.board-full .hand .purse-line { display: flex; }" in RULES
