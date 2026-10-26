"""The duck on the page (generation 42, N4c, `docs/plan-npc.md` §4.6).

The play session's side of the two-part move: the snapshot's `duckOwed` and `duck`, `_offer`'s
duck entry, the history's, and the take-back that never leaves half a turn. The board's duck
mode, the pass button and the one-line turn are the page's (`board.js`); they are held here
by their names in the page's text, and in a browser by the look check.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from page_text import PAGE

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import (
    DUCK_PASS,
    DUCK_PLACE,
    MOVE_DUCK,
    move_promotion,
    move_type,
)
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.play import Sessions

ROOT = Path(__file__).resolve().parents[1]
DUCK_CHESS = json.loads((ROOT / "tests" / "_duck_chess.json").read_text(encoding="utf-8"))

# A duck boxed in on a5 by its own moves (one step each way), so its only move is the pass:
# a5's neighbours are walls or pieces nobody moves into before the duck's half.
BOXED: dict[str, Any] = {
    "name": "boxed duck (test)",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "name": "king", "royal": True, "value": 0,
         "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
        {"id": "D", "name": "duck", "value": 0,
         "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"}]},
    ],
    "setup": {"mirror": False, "placements": [
        {"piece": "K", "player": 0, "squares": ["e1"]},
        {"piece": "K", "player": 1, "squares": ["e5"]},
    ]},
    "rules": {"win": "royal_capture", "max_plies": 40,
              "obstacles": {"squares": ["a4", "b5"]},
              "duck": {"piece": "D", "start": "a5", "goes": "by_moves"}},
}


def _game(spec: dict[str, Any], name: str) -> library.Game:
    return library.Game(hash=spec_hash(spec), name=name, description="",
                        path=Path(f"rulesets/{name}.json"), source="designed",
                        spec=copy.deepcopy(spec))


@pytest.fixture(scope="module")
def duck_game() -> library.Game:
    return _game(DUCK_CHESS, "duck-chess")


def _named(snapshot: dict[str, Any], text: str) -> int:
    return next(m["move"] for m in snapshot["moves"] if m["text"] == text)


class TestTheSnapshot:
    def test_the_opening_owes_a_piece_move_and_the_duck_is_off_the_board(
        self, duck_game: library.Game
    ) -> None:
        snap = Sessions().start(duck_game, ("human", "human")).snapshot()
        assert snap["duckOwed"] is False and snap["duck"] is None
        assert snap["ruleset"]["duck"] is not None
        assert all(m["kind"] != "duck" for m in snap["moves"])

    def test_after_the_piece_move_every_empty_square_is_offered_as_a_placement(
        self, duck_game: library.Game
    ) -> None:
        session = Sessions().start(duck_game, ("human", "human"))
        snap = session.play(_named(session.snapshot(), "Pe2-e4"))
        assert snap["duckOwed"] is True and snap["sideToMove"] == 0
        ducks = snap["moves"]
        assert len(ducks) == 32 and all(m["kind"] == "duck" for m in ducks)
        assert all(m["from"] is None and m["pass"] is False for m in ducks)
        assert {m["to"] for m in ducks} == {
            sq for sq, held in enumerate(snap["board"]) if held == 0}

    def test_a_standing_duck_moves_from_its_square_and_the_history_says_so(
        self, duck_game: library.Game
    ) -> None:
        session = Sessions().start(duck_game, ("human", "human"))
        session.play(_named(session.snapshot(), "Pe2-e4"))
        snap = session.play(_named(session.snapshot(), "the duck lands on d5"))
        placed = snap["history"][-1]
        assert placed["from"] is None and placed["to"] == 35 and placed["pass"] is False
        assert snap["duck"] == 35 and snap["duckOwed"] is False and snap["sideToMove"] == 1
        session.play(_named(snap, "Pe7-e5"))
        snap = session.snapshot()
        assert {m["from"] for m in snap["moves"]} == {35}
        assert 35 not in {m["to"] for m in snap["moves"]}
        snap = session.play(_named(snap, "the duck to c3"))
        assert snap["history"][-1]["from"] == 35 and snap["history"][-1]["to"] == 18

    def test_the_pass_is_offered_alone_and_goes_nowhere(self) -> None:
        session = Sessions().start(_game(BOXED, "boxed"), ("human", "human"))
        snap = session.play(_named(session.snapshot(), "Ke1-d1"))
        assert snap["duckOwed"] is True and snap["duck"] == 20
        (only,) = snap["moves"]
        assert only["pass"] is True and only["from"] is None and only["to"] is None
        assert move_type(only["move"]) == MOVE_DUCK and move_promotion(only["move"]) == DUCK_PASS
        snap = session.play(only["move"])
        assert snap["history"][-1]["pass"] is True and snap["sideToMove"] == 1
        assert snap["duck"] == 20


class TestTheTakeBack:
    def test_begun_with_the_duck_owed_it_returns_the_piece_move_alone(
        self, duck_game: library.Game
    ) -> None:
        session = Sessions().start(duck_game, ("human", "random"), seed=3)
        session.play(_named(session.snapshot(), "Pe2-e4"))
        snap = session.undo()
        assert snap["history"] == [] and snap["duckOwed"] is False

    def test_after_the_reply_it_returns_the_whole_turn(self, duck_game: library.Game) -> None:
        session = Sessions().start(duck_game, ("human", "random"), seed=3)
        session.play(_named(session.snapshot(), "Pe2-e4"))
        snap = session.play(_named(session.snapshot(), "the duck lands on d5"))
        while snap["awaitingAgent"]:
            snap = session.play_agent()
        assert len(snap["history"]) == 4 and snap["duckOwed"] is False
        snap = session.undo()
        assert snap["history"] == [] and snap["duckOwed"] is False

    def test_the_duck_half_alone_comes_back_between_two_people_never_half_a_turn(
        self, duck_game: library.Game
    ) -> None:
        session = Sessions().start(duck_game, ("human", "human"))
        session.play(_named(session.snapshot(), "Pe2-e4"))
        session.play(_named(session.snapshot(), "the duck lands on d5"))
        snap = session.undo()
        # The second player owes a piece move at the turn's start: back to the first's.
        assert snap["history"] == [] and not snap["duckOwed"]

    def test_a_placement_alone_comes_back_once_the_piece_move_is_made(self) -> None:
        engine = Engine(DUCK_CHESS)
        state = engine.initial_state()
        assert not engine.duck_owed(state)
        moves = engine.legal_moves(engine.apply(state, engine.legal_moves(state)[0]))
        assert all(move_promotion(m) == DUCK_PLACE for m in moves)


class TestThePage:
    def test_the_board_has_a_duck_mode_a_pass_button_and_a_one_line_turn(self) -> None:
        for needle in ("function duckMode", "function duckTargets", "now move the duck",
                       "the duck has nowhere to go: pass", "function duckRowHtml",
                       "function duckFlightHtml", "data-duck-pass"):
            assert needle in PAGE, needle

    def test_the_duck_flight_is_a_mark_kind_with_words(self) -> None:
        assert "duckFlight:" in PAGE and "var(--duck)" in PAGE


class TestTheDesignersDuckCard:
    def _card(self, draft: dict[str, Any], check: dict[str, Any] | None = None) -> str:
        from test_designer import _kit_setup, _run_page

        return str(_run_page(
            _kit_setup() + f"DRAFT = {json.dumps(draft)}; DRAFT_CHECK = {json.dumps(check)};",
            "rulesCardsHtml([])",
        ))

    def test_the_card_says_where_it_starts_how_it_goes_and_the_rule_in_words(self) -> None:
        from chess_variant_sandbox.web import designing

        answer = designing.check(copy.deepcopy(DUCK_CHESS))
        assert answer["valid"], answer.get("errors")
        assert answer["duck"]["says"] and "Every turn is two halves" in answer["duck"]["says"][0]
        html = self._card(DUCK_CHESS, answer)
        assert 'class="ruleCard duckCard" data-rule="duck"' in html
        assert "off the board: the first player" in html
        assert "pick a square on the board" in html and 'data-csingle="1"' in html
        assert "Every turn is two halves" in html
        placed = copy.deepcopy(DUCK_CHESS)
        placed["rules"]["duck"]["start"] = "d4"
        assert "on <b>d4</b>" in self._card(placed) and "data-cduckoff" in self._card(placed)

    def test_its_refusals_are_on_the_card_in_the_designers_words(self) -> None:
        from chess_variant_sandbox.web import designing

        mated = copy.deepcopy(DUCK_CHESS)
        mated["rules"]["win"] = "checkmate"
        answer = designing.check(mated)
        assert not answer["valid"] and answer["duck"]["errors"]
        assert "legality would depend on the duck move after it" in answer["duck"]["errors"][0]
        card = self._card(mated, answer)
        assert card.count("legality would depend on the duck move after it") == 1

    def test_adding_the_rule_finds_or_makes_the_duck_and_drops_mate(self) -> None:
        from test_designer import _kit_setup, _run_page

        draft = {"pieces": [{"id": "K", "name": "king", "royal": True, "value": 0,
                             "moves": []}],
                 "setup": {"placements": [{"piece": "K", "player": 0, "squares": ["e1"]}]},
                 "rules": {"win": "checkmate", "max_plies": 201}}
        made = _run_page(
            _kit_setup() + f"DRAFT = {json.dumps(draft)};",
            "(RULE_COMPANIONS.duck(DRAFT.rules), DRAFT)",
        )
        assert made["rules"]["win"] == "royal_capture" and made["rules"]["max_plies"] == 202
        duck = made["rules"]["duck"]["piece"]
        assert duck == "D"
        assert {"id": "D", "name": "duck", "value": 0, "moves": []} in made["pieces"]

    def test_the_menu_offers_the_duck_on_every_template_as_the_card_arrives(self) -> None:
        """The server's trial (`designing._duck_companion`) is the page's companion: the same
        draft from both, and the menu refuses the duck on no template."""
        from test_designer import _kit_setup, _run_page

        from chess_variant_sandbox.web import designing

        for template in designing.templates():
            draft = copy.deepcopy(template["spec"])
            page = _run_page(
                _kit_setup() + f"DRAFT = {json.dumps(draft)};",
                "(() => { const c = ruleCandidates(); DRAFT.rules.duck = c.duck;"
                " RULE_COMPANIONS.duck(DRAFT.rules); return [c.duck, DRAFT]; })()",
            )
            candidate, made = page
            trial = copy.deepcopy(draft)
            trial["rules"]["duck"] = candidate
            designing._duck_companion(trial)
            assert trial == made, template["key"]
            assert designing.refusals(draft, {"duck": candidate})["refused"] == {}, template["key"]
