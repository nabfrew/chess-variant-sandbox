"""A draw offered to the computer, judged by the core (Q790, §e.1 of
`docs/reference/intransitive-quiet-endgame.md`).

What is held here, with no store: the judge is the core's `search_values` on *hardest* with
mobility off, whatever the seat; a held fortress of §b reads level and is accepted, a race
somebody wins is declined; the band is half the game's cheapest piece; the guard (ply 40,
the person's own turn) and the one-offer-in-ten-plies rule say why in a sentence; and a
session's snapshot carries what the page needs and nothing it would have to work out.

The positions are §e.1's F1 (the fortress, the first player to move) and L1 (the second
player's blocker four steps out, the first player's attacker two from its goal). Both are
*Intransitive*'s, because §b is about that game; the rule itself names no piece or game,
and the band is read off a second spec (`dull`) to show it is the table's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.web import draw_offer, library
from chess_variant_sandbox.web.play import HUMAN, Session

ROOT = Path(__file__).resolve().parents[1]
INTRANSITIVE = json.loads((ROOT / "rulesets" / "intransitive.json").read_text(encoding="utf-8"))
DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))

FORTRESS = {"a1": ("R", 0), "h8": ("R", 0), "i9": ("R", 1), "b2": ("R", 1)}
LOST_RACE = {"a1": ("R", 0), "g9": ("R", 0), "e5": ("R", 1), "b2": ("R", 1)}


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(INTRANSITIVE)


def _state(engine: Engine, placement: dict[str, Any], mover: int) -> Any:
    return engine.make_state(placement, side_to_move=mover, moved=list(placement), ply=100)


class TestTheJudge:
    def test_the_judge_is_hardest_with_mobility_off(self) -> None:
        assert draw_offer.JUDGE == "minimax:nodes=200000,mobility=0"

    def test_the_held_fortress_is_level_and_accepted(self, engine: Engine) -> None:
        verdict = draw_offer.judge(engine, _state(engine, FORTRESS, 0), 100)
        assert verdict.accepted
        assert verdict.value == 0.0
        assert verdict.depth is not None and verdict.depth >= 10

    def test_a_lost_race_is_declined_from_either_side(self, engine: Engine) -> None:
        losing = draw_offer.judge(engine, _state(engine, LOST_RACE, 1), 100)
        winning = draw_offer.judge(engine, _state(engine, LOST_RACE, 0), 100)
        assert not losing.accepted and not winning.accepted
        assert losing.value is not None and losing.value < -1000
        assert winning.value is not None and winning.value > 1000

    def test_the_answer_is_the_cores_own_and_repeats(self, engine: Engine) -> None:
        """The same offer in the same position, at the same seed, gets the same answer."""
        state = _state(engine, FORTRESS, 1)
        assert draw_offer.judge(engine, state, 7) == draw_offer.judge(engine, state, 7)

    def test_a_finished_position_has_no_value_and_is_declined(self, engine: Engine) -> None:
        reached = {"a1": ("R", 0), "i9": ("R", 0), "e5": ("R", 1), "b2": ("R", 1)}
        state = _state(engine, reached, 1)
        assert engine.terminal(state) is not None
        verdict = draw_offer.judge(engine, state, 0)
        assert not verdict.accepted and verdict.value is None

    def test_the_band_is_half_the_games_cheapest_piece(self, engine: Engine) -> None:
        assert draw_offer.band(engine.spec) == 0.5
        dull = Engine(DULL)
        assert draw_offer.band(dull.spec) == 0.5 * min(
            value for value in draw_offer.eval_values(dull.spec) if value > 0
        )

    def test_a_logged_offer_keeps_what_the_core_said(self, engine: Engine) -> None:
        verdict = draw_offer.judge(engine, _state(engine, FORTRESS, 0), 100)
        entry = verdict.as_entry(seat=0, ply=100)
        assert entry == {
            "seat": 0, "ply": 100, "accepted": True, "value": verdict.value,
            "depth": verdict.depth, "band": verdict.band,
        }


class TestWhenAnOfferMayBeMade:
    def test_not_before_ply_forty(self) -> None:
        said = draw_offer.against_computer_refusal([], 0, draw_offer.EARLIEST - 2, 0)
        assert said == draw_offer.TOO_EARLY_SAID
        assert draw_offer.against_computer_refusal([], 0, draw_offer.EARLIEST, 0) is None

    def test_only_on_the_persons_own_turn(self) -> None:
        assert draw_offer.against_computer_refusal([], 0, 60, 1) == draw_offer.OWN_TURN_SAID

    def test_one_offer_in_ten_plies(self) -> None:
        offers = [{"seat": 0, "ply": 50}]
        said = draw_offer.gap_refusal(offers, 0, 52)
        assert said is not None and "4 more moves" in said
        assert draw_offer.gap_refusal(offers, 0, 58) is not None
        assert draw_offer.gap_refusal(offers, 0, 60) is None
        # The other side's offers are its own count.
        assert draw_offer.gap_refusal(offers, 1, 51) is None

    def test_a_take_back_before_an_offer_does_not_open_a_second(self) -> None:
        offers = [{"seat": 0, "ply": 50}]
        assert draw_offer.gap_refusal(offers, 0, 44) is not None
        assert draw_offer.gap_refusal(offers, 0, 40) is None

    def test_the_answer_said_is_the_entrys(self) -> None:
        assert draw_offer.answer_said({"accepted": True}) == draw_offer.ACCEPTED_SAID
        assert draw_offer.answer_said({"accepted": False}) == draw_offer.DECLINED_SAID
        assert draw_offer.answer_said(None) is None


def _session(engine: Engine, state: Any, mode: str, offers: list[dict[str, Any]]) -> Session:
    game = library.Game(
        hash=engine.spec.hash, name="intransitive", description="", path=ROOT,
        source="test", spec=INTRANSITIVE,
    )
    return Session(
        id="x", game=game, seats=(HUMAN, "minimax:nodes=400"), seed=0, state=state,
        mode=mode, draw_offers=offers,
    )


class TestTheSnapshot:
    def test_a_game_against_the_computer_says_whether_an_offer_may_be_made(
        self, engine: Engine
    ) -> None:
        offer = _session(engine, _state(engine, FORTRESS, 0), "sandbox", []).snapshot()["drawOffer"]
        assert offer == {"may": True, "why": None, "said": None}

    def test_a_decline_in_this_position_is_said_and_the_next_offer_waits(
        self, engine: Engine
    ) -> None:
        entry = {"seat": 0, "ply": 100, "accepted": False}
        offer = _session(engine, _state(engine, FORTRESS, 0), "sandbox", [entry]).snapshot()[
            "drawOffer"
        ]
        assert offer["may"] is False
        assert offer["said"] == draw_offer.DECLINED_SAID
        assert offer["why"] == draw_offer.gap_refusal([entry], 0, 100)

    def test_the_local_application_and_the_computers_turn(self, engine: Engine) -> None:
        local = _session(engine, _state(engine, FORTRESS, 0), "", []).snapshot()
        assert local["drawOffer"] is None, "the local application keeps no result"
        waiting = _session(engine, _state(engine, FORTRESS, 1), "sandbox", []).snapshot()
        assert waiting["drawOffer"]["may"] is False
        assert waiting["drawOffer"]["why"] == draw_offer.OWN_TURN_SAID
