"""Turncoat pieces on the page, in the designer and in the analysis tallies (TC7,
`docs/plan-turncoat.md` §5.8).

What is held here, by behaviour:

- **The play page asks the core.** A position of a purse game carries each player's coins
  (`GameState.purse`) and the compiled market and costs; a game with majority control carries
  the core's answer for who holds each piece nobody owns (`Engine.controller_of`). A buy is
  offered with no origin, its type and its price; a hire or a bribe with its piece's type, its
  cost and which it is. A history entry and a replayed position carry the purses of the
  position the move left. Every other game's snapshot is what it was: none of these keys.
- **The board.** The market tray draws each piece for sale with its price, greyed where the
  engine offers no buy of it; picking one lights the engine's squares, and a tap there sends
  the engine's buy. A picked piece with a hire or bribe shows it as a button, never as a dot on
  its own square, and the button sends the engine's move. A held neutral wears the holder's
  ring with the holder's bead, named in the legend. The move list says what a purchase cost.
- **The designer.** A placement row may be nobody's, under a mirror too, and a mirror keeps it
  nobody's; the piece card has *price*, *bounty* and *when nobody owns it*; the purse card has
  *coins at the start*, *for sale*, *bribes* and *where bought pieces go*; the refusals are the
  validator's words.
- **The tallies.** `PURSE_SERIES` in a purse game and `NEUTRAL_SERIES` in a game with a piece
  nobody owns, each the core's state or answer and the record's move; absent byte for byte
  from every other game.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from test_designer import _kit_setup, _run_page
from test_portal_board import _game
from test_sight_page import _drawn

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_BUY,
    MOVE_FOREIGN,
    MOVE_HIRE,
    move_from,
    move_type,
)
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament.analysis import NEUTRAL_SERIES, PURSE_SERIES, GameAnalysis
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.web import cards, designing
from chess_variant_sandbox.web.library import engine_for
from chess_variant_sandbox.web.play import Session

HERE = Path(__file__).parent


def _spec(name: str = "turncoat_page") -> dict[str, Any]:
    return json.loads((HERE / f"_{name}.json").read_text(encoding="utf-8"))


def _square(name: str, width: int = 6) -> int:
    return (int(name[1:]) - 1) * width + "abcdefgh".index(name[0])


def _session(spec: dict[str, Any], placement: dict[str, tuple[str, int]] | None = None,
             purse: tuple[int, int] | None = None) -> Session:
    engine = engine_for(spec)
    state = engine.make_state(placement, purse=purse) if placement else engine.initial_state()
    return Session(id="s1", game=_game(spec), seats=("human", "human"), seed=0, state=state)


# A white pawn on c2 beside a black knight on c3 (a bribe at twice the knight's price, 8), a
# condottiere nobody owns on e3 beside white's pawn on d2 (a hire at 3, and held by white).
BRIBE = {"c1": ("K", 0), "c2": ("P", 0), "d2": ("P", 0), "c3": ("N", 1), "e3": ("C", 2),
         "f6": ("K", 1)}


# ------------------------------------------------------------------ the server's half


class TestTheServer:
    def test_the_position_carries_the_purses_the_market_and_the_held_neutrals(self) -> None:
        session = _session(_spec())
        engine, state = session.engine, session.state
        snap = session.snapshot()
        assert snap["purse"] == [4, 4]
        rule = snap["ruleset"]["purse"]
        ids = snap["ruleset"]["pieceIds"]
        assert [(ids[item["type"]], item["price"]) for item in rule["market"]] == [
            ("P", 2), ("N", 4)]
        assert rule["cap"] == 99 and rule["start"] == 4
        # The held neutrals are the core's answer, square by square.
        expected = [[square, engine.controller_of(state, square)]
                    for square, value in enumerate(state.board)
                    if value and engine.spec.owner_of(value) == 2
                    and engine.controller_of(state, square) in (0, 1)]
        assert snap["held"] == expected and expected

    def test_a_buy_and_a_hire_are_offered_with_what_they_buy_and_cost(self) -> None:
        snap = _session(_spec()).snapshot()
        buys = [m for m in snap["moves"] if m["kind"] == "buy"]
        assert buys and all(m["from"] is None and move_type(m["move"]) == MOVE_BUY for m in buys)
        ids = snap["ruleset"]["pieceIds"]
        assert {(ids[m["buy"]], m["cost"]) for m in buys} == {("P", 2), ("N", 4)}
        assert all("$" in m["text"] for m in buys)
        hires = [m for m in snap["moves"] if m["kind"] == "hire"]
        assert [(m["text"], m["cost"], m["bribe"]) for m in hires] == [("$Cb3", 3, False)]
        assert hires[0]["from"] == hires[0]["to"] == _square("b3")
        # A held neutral's moves are foreign moves from its square.
        held = [m for m in snap["moves"] if m.get("foreign")]
        assert held and all(m["text"].startswith("~C") and m["move"] & MOVE_FOREIGN
                            for m in held)

    def test_a_bribe_costs_twice_the_price(self) -> None:
        snap = _session(_spec(), BRIBE, purse=(9, 0)).snapshot()
        hires = {m["text"]: m for m in snap["moves"] if m["kind"] == "hire"}
        assert hires["$Nc3"]["bribe"] is True and hires["$Nc3"]["cost"] == 8
        assert hires["$Ce3"]["bribe"] is False and hires["$Ce3"]["cost"] == 3

    def test_a_purchase_is_said_in_history_and_the_timeline_with_its_purses(self) -> None:
        session = _session(_spec())
        buy = next(m for m in session.snapshot()["moves"] if m["text"] == "N$e1")
        snap = session.play(buy["move"])
        last = snap["history"][-1]
        assert last["text"] == "N$e1" and last["from"] is None and last["cost"] == 4
        assert last["purse"] == [0, 4] == snap["purse"]
        hire = next(m for m in snap["moves"] if m["kind"] == "hire")
        snap = session.play(hire["move"])
        assert snap["history"][-1]["cost"] == 3 and snap["purse"] == [0, 1]
        timeline = designing.timeline(session)
        assert [position["purse"] for position in timeline] == [[4, 4], [0, 4], [0, 1]]
        assert all("held" in position for position in timeline)

    def test_the_far_from_chess_words_call_a_neutral_no_wild_piece(self) -> None:
        from chess_variant_sandbox.web.library import far_from_chess

        far = far_from_chess(_game(_spec()))
        assert "it has pieces that belong to neither side" in far
        assert not any("wild" in reason for reason in far)

    def test_another_game_carries_none_of_it(self) -> None:
        spec = _spec("command_k1")
        session = _session(spec)
        move = session.snapshot()["moves"][0]
        snap = session.play(move["move"])
        for block in (snap, snap["history"][-1], *designing.timeline(session)):
            assert "purse" not in block and "held" not in block
        assert "purse" not in snap["ruleset"]
        assert not any("cost" in m or "foreign" in m for m in snap["moves"])


# ------------------------------------------------------------------ the page's half


# The hand's strip, as far as `drawHand` and `wireMarket` use it: its markup, and a node per
# `data-*` attribute a selector asks for, whose `onclick` the wiring sets (`HAND_NODES`).
_HAND = r"""
let HAND_HTML = ''; const HAND_NODES = [];
const HAND = { set innerHTML(v) { HAND_HTML = v; }, get innerHTML() { return HAND_HTML; },
  querySelectorAll(selector) {
    const name = (selector.match(/\[data-([a-z-]+)\]/) || [])[1];
    if (!name) return [];
    const key = name.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const found = [], pattern = new RegExp(`data-${name}="([^"]*)"`, 'g');
    let hit;
    while ((hit = pattern.exec(HAND_HTML))) {
      const node = { dataset: { [key]: hit[1] } };
      HAND_NODES.push([name, node]);
      found.push(node);
    }
    return found;
  } };
const BOARD_GET = document.getElementById;
document.getElementById = id => id === 'hand' ? HAND : BOARD_GET(id);
"""


def _with_hand(snapshot: dict[str, Any], steps: str, expression: str) -> Any:
    return _drawn(snapshot, _HAND + steps, expression)


def _marks(html: str) -> list[tuple[str, str]]:
    return re.findall(r'data-mark="holder"[^>]*/>\s*<circle[^>]*fill="var\(--(p[01])\)"', html)


class TestThePage:
    def test_the_market_greys_what_the_engine_does_not_offer(self) -> None:
        snap = _session(_spec(), BRIBE, purse=(3, 0)).snapshot()
        html = _with_hand(snap, "", "HAND_HTML")
        buttons = dict(re.findall(r'data-buy="(\d)"[^>]*?(disabled)?>', html))
        ids = snap["ruleset"]["pieceIds"]
        assert buttons == {str(ids.index("P")): "", str(ids.index("N")): "disabled"}, html
        assert "market (3 coins):" in html
        assert re.search(r'class="buy-face"', html)

    def test_a_market_piece_picked_lights_its_squares_and_a_tap_buys(self) -> None:
        snap = _session(_spec()).snapshot()
        pawn = snap["ruleset"]["pieceIds"].index("P")
        move = next(m for m in snap["moves"] if m["text"] == "P$e1")
        steps = (f"let SENT = null; send = m => {{ SENT = m; }}; BUY = {pawn}; drawBoard();"
                 f" const LIT_SQUARES = [...buyTargets()].sort((a, b) => a - b);"
                 f" clickSquare({_square('e1')});")
        lit, sent = _drawn(snap, steps, "[LIT_SQUARES, SENT]")
        assert lit == sorted(m["to"] for m in snap["moves"] if m.get("buy") == pawn)
        assert sent == move["move"]

    def test_a_picked_piece_shows_its_hire_as_a_button_and_never_a_dot(self) -> None:
        snap = _session(_spec(), BRIBE, purse=(9, 0)).snapshot()
        knight, condottiere = _square("c3"), _square("e3")
        bribe = next(m for m in snap["moves"] if m["text"] == "$Nc3")
        steps = (f"let SENT = null; send = m => {{ SENT = m; }}; clickSquare({knight});"
                 " const SHOWN_HAND = HAND_HTML, SAID = NOTE;"
                 f" clickSquare({knight}); const AFTER = SENT;")
        hand, said, after, board = _with_hand(snap, steps, "[SHOWN_HAND, SAID, AFTER, DRAWN]")
        assert f'data-hire="{bribe["move"]}"' in hand and "bribe for 8" in hand
        assert "bribe it for 8 coins" in said
        assert after is None, "a second tap on the piece is not the bribe"
        assert not re.search(rf'<circle[^>]*data-square="{knight}"', board), "no dot on its square"
        steps = (f"let SENT = null; send = m => {{ SENT = m; }}; clickSquare({condottiere});"
                 " HAND_NODES.filter(([name]) => name === 'hire').pop()[1].onclick();")
        hire = next(m for m in snap["moves"] if m["text"] == "$Ce3")
        assert _with_hand(snap, steps, "SENT") == hire["move"]

    def test_a_held_neutral_wears_the_holders_ring_and_the_legend_names_it(self) -> None:
        snap = _session(_spec()).snapshot()
        drawn = _drawn(snap, "", "DRAWN")
        rings = _marks(drawn)
        assert sorted(rings) == sorted("p0" if player == 0 else "p1"
                                       for _, player in snap["held"])
        legend = " ".join(_drawn(snap, "", "legendEntries(new Set(['holder']), null, '', null)"))
        assert "nobody owns, held by the side of its bead" in legend

    def test_the_purses_and_what_a_purchase_cost_are_said(self) -> None:
        session = _session(_spec())
        buy = next(m for m in session.snapshot()["moves"] if m["text"] == "N$e1")
        snap = session.play(buy["move"])
        said = _drawn(snap, "", "[purseShown(), spentHtml(SESSION.history[0]),"
                                " spentHtml({cost: 8, hire: 2, bribe: true}), purseSlotHtml(1)]")
        assert said[0] == [0, 4]
        assert "bought for 4" in said[1] and "bribed for 8" in said[2]
        assert 'data-purse="1"' in said[3]
        other = _session(_spec("command_k1")).snapshot()
        assert _drawn(other, "", "[purseShown(), purseSlotHtml(0), marketHtml(0)]") == [
            None, "", ""]


# ------------------------------------------------------------------ the designer's half


def _designer(expression: str, setup: str = "") -> Any:
    return _run_page(_kit_setup() + setup, expression)


class TestTheDesigner:
    def test_a_placement_may_be_nobodys_under_a_mirror_and_without_one(self) -> None:
        spec = _spec()
        nobody = next(i for i, p in enumerate(spec["setup"]["placements"])
                      if p.get("player") == "nobody")
        html = _designer(f"(DRAFT = {json.dumps(spec)}, placementEditor("
                         f"DRAFT.setup.placements[{nobody}], {nobody}))")
        assert re.search(r'<option value="nobody" selected>nobody</option>', html)
        assert '<option value="1"' not in html, "under a mirror the second army is reflected"
        unmirrored = copy.deepcopy(spec)
        unmirrored["setup"]["mirror"] = False
        html = _designer(f"(DRAFT = {json.dumps(unmirrored)}, placementEditor("
                         f"DRAFT.setup.placements[0], 0))")
        assert '<option value="1"' in html and '<option value="nobody"' in html

    def test_the_cards_offer_the_keys(self) -> None:
        model = cards.card_model()
        piece = {field["key"] for field in model["piece"]}
        assert {"price", "bounty", "control"} <= piece
        purse = next(field for field in model["rules"] if field["key"] == "purse")
        assert [field["key"] for field in purse["fields"]] == [
            "start", "market", "bribe", "region"]

    def test_a_turncoat_draft_is_a_game_and_its_refusals_are_the_validators(self) -> None:
        spec = _spec()
        assert validate_spec(spec) == []
        answer = designing.check(spec)
        assert answer["valid"]
        assert any("nobody's" in line for line in answer["setupLines"])
        broken = copy.deepcopy(spec)
        broken["setup"]["placements"][0]["player"] = "nobody"
        answer = designing.check(broken)
        assert not answer["valid"]
        assert any("royal always belongs to a player" in line for line in answer["errors"])


# ------------------------------------------------------------------ the analysis tallies


def _analysed(spec: dict[str, Any], seed: int, agents: tuple[str, str] = ("greedy", "random")
              ) -> tuple[Engine, Any, GameAnalysis]:
    engine = Engine(spec)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    record = play_game(engine, agents, seed)
    found = analysis_pass._analyse_game(engine, evaluator, record,
                                        analysis_pass._epsilon_for(engine))
    return engine, record, found


class TestTheTallies:
    def test_each_turn_is_the_cores_state_and_the_records_move(self) -> None:
        spent = 0
        for seed in (1, 2, 3):
            engine, record, found = _analysed(_spec(), seed)
            purse, neutral = found.purse, found.neutral
            assert purse is not None and list(purse) == list(PURSE_SERIES)
            assert neutral is not None and list(neutral) == list(NEUTRAL_SERIES)
            assert all(len(series) == record.plies for series in [*purse.values(),
                                                                  *neutral.values()])
            state = engine.initial_state()
            for ply, move in enumerate(record.moves):
                seat, kind = state.side_to_move, move_type(move)
                after = engine.apply(state, move)
                nobodys = [sq for sq, v in enumerate(state.board)
                           if v and engine.spec.owner_of(v) == 2]
                bribe = kind == MOVE_HIRE and move_from(move) not in nobodys
                purchase = kind in (MOVE_BUY, MOVE_HIRE)
                change = after.purse[seat] - state.purse[seat]
                assert {key: purse[key][ply] for key in PURSE_SERIES} == {
                    "side": seat, "purse": state.purse[seat],
                    "spent": -change if purchase else 0,
                    "earned": 0 if purchase else max(change, 0),
                    "bought": int(kind == MOVE_BUY),
                    "hired": int(kind == MOVE_HIRE and not bribe), "bribed": int(bribe),
                }, (seed, ply)
                assert {key: neutral[key][ply] for key in NEUTRAL_SERIES} == {
                    "side": seat,
                    "held": sum(1 for sq in nobodys if engine.controller_of(state, sq) == seat),
                    "neutral_offered": sum(1 for m in engine.legal_moves(state)
                                           if m & MOVE_FOREIGN and move_from(m) in nobodys),
                    "neutral_made": int(bool(move & MOVE_FOREIGN)
                                        and move_from(move) in nobodys),
                }, (seed, ply)
                state = after
            spent += sum(purse["spent"])
            assert GameAnalysis.from_dict(json.loads(json.dumps(found.as_dict()))) == found
        assert spent > 0, "some game spends coins"

    def test_other_games_write_the_keys_they_did(self) -> None:
        for name in ("command_k1", "rest_drops"):
            _, _, found = _analysed(_spec(name), 1)
            data = found.as_dict()
            assert "purse" not in data and "neutral" not in data
        # A purse without a piece nobody owns writes the purse alone.
        _, _, found = _analysed(_spec("purse_market"), 1)
        assert found.purse is not None and "neutral" not in found.as_dict()
