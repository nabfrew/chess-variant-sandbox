"""The designer says why a combination is refused and offers the nearest allowed game (R-L3 B3).

`docs/reference/refusal-offers.md` §6.2, §6.3 and §8's test 7. The rule menu's refusals carry
their reasons (a lookup), `POST /api/design/instead` answers the reasons and the offers of
`spec/instead.nearest`, and a check of a draft refused as it stands carries the reasons the
page asks the same route with. The three reason sentences are one constant,
`refusals.REASON_SAYS`, that `RULES.md` and the designer both say and the page never copies.

The draft here is the chess ruleset, used because rounds beside it is the design's own pinned
witness (§8 test 2: `R1` and `R7`), not because anything below is about chess.
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest
from page_text import DESIGN, PAGE

from chess_variant_sandbox.site import routes
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.instead import apply_edits, parse_path
from chess_variant_sandbox.spec.reference import render_reference
from chess_variant_sandbox.spec.refusals import PLANNED, REASON_SAYS, Entry
from chess_variant_sandbox.web import designing
from chess_variant_sandbox.web.server import REQUEST_HEADER, REQUEST_VALUE, App, Server, handler_for

ROOT = Path(__file__).resolve().parent.parent
CHESS = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
ROUNDS_EDITS = ['set rules.win = "royal_capture"', "drop rules.en_passant"]


def _in_rounds() -> dict[str, Any]:
    draft = copy.deepcopy(CHESS)
    draft["rules"]["simultaneous"] = {}
    return draft


# --------------------------------------------------------------- one constant, said twice


class TestTheReasonSentencesAreOneConstant:
    def test_rules_md_says_each_sentence_of_the_constant(self) -> None:
        text = render_reference()
        for said in REASON_SAYS.values():
            assert f"*{said}*" in text

    def test_the_designer_says_the_constant(self) -> None:
        for reason, said in REASON_SAYS.items():
            entry = Entry("x", reason, "why", ("a", "b"), held_by="none")
            assert designing.reason_says(entry) == said
        planned = Entry("x", "not_built", "why", ("a", "b"), held_by="SR9a")
        assert designing.reason_says(planned) == f"{REASON_SAYS['not_built']} {PLANNED}"
        assert designing.reason_says(Entry("x", "inert", "why", ("a", "b"))) is None

    def test_the_page_holds_no_copy(self) -> None:
        # The sentences come in the server's answer; a copy in a script could drift.
        for said in REASON_SAYS.values():
            assert said not in PAGE
        assert PLANNED not in DESIGN


# --------------------------------------------------------------- the menu (§6.2)


class TestTheMenu:
    def test_a_refused_rule_carries_its_reasons(self) -> None:
        answer = designing.refusals(CHESS, {"simultaneous": {}})
        assert answer["refused"]["simultaneous"]
        assert answer["reasons"]["simultaneous"] == [
            {"reason": "contradictory", "says": REASON_SAYS["contradictory"]}]

    def test_a_draft_that_is_not_a_game_is_asked_nothing(self) -> None:
        broken = copy.deepcopy(CHESS)
        broken["pieces"][0]["stamina"] = 2
        assert designing.refusals(broken, {"simultaneous": {}}) == {
            "refused": {}, "reasons": {}, "asked": False}

    def test_opening_rounds_offers_the_pinned_game(self) -> None:
        answer = designing.instead(CHESS, "simultaneous", {})
        assert answer["searched"] is True
        assert answer["reasons"] == [
            {"reason": "contradictory", "says": REASON_SAYS["contradictory"]}]
        first = answer["offers"][0]
        assert first["edits"] == ROUNDS_EDITS
        assert first["changes"] and first["units"] >= 1
        for offer in answer["offers"]:
            draft = offer["draft"]
            assert validate_spec(draft) == []
            assert isinstance(draft["rules"].get("simultaneous"), dict)
            assert spec_hash(draft) != spec_hash(designing.settled(CHESS))

    def test_the_offer_is_the_trial_with_its_edits(self) -> None:
        offer = designing.instead(CHESS, "simultaneous", {})["offers"][0]
        trial = designing.settled(_in_rounds())
        edits = tuple(
            ("drop", text.removeprefix("drop ")) if text.startswith("drop ")
            else ("set", text.removeprefix("set ").split(" = ")[0],
                  json.loads(text.split(" = ", 1)[1]))
            for text in offer["edits"])
        for edit in edits:
            parse_path(edit[1])
        assert apply_edits(trial, edits) == offer["draft"]

    def test_a_refusal_no_small_change_mends_is_searched_and_offers_nothing(self) -> None:
        # B2's pinned no-offer witness (`tests/test_instead.py`, a pass beside a push): the
        # reason is said and the page says no small change makes it allowed.
        dohyo = json.loads((ROOT / "rulesets" / "dohyo.json").read_text(encoding="utf-8"))
        answer = designing.instead(dohyo, "pass", True)
        assert answer["searched"] is True and answer["offers"] == []
        assert [row["reason"] for row in answer["reasons"]]

    def test_an_answer_is_the_callers_own(self) -> None:
        one = designing.instead(CHESS, "simultaneous", {})
        one["offers"][0]["draft"]["rules"]["win"] = "mangled"
        two = designing.instead(CHESS, "simultaneous", {})
        assert two["offers"][0]["draft"]["rules"]["win"] == "royal_capture"


# --------------------------------------------------------------- the check (§6.3)


class TestTheCheck:
    def test_a_refused_draft_carries_its_reasons(self) -> None:
        answer = designing.check(_in_rounds())
        assert answer["valid"] is False
        assert answer["reasons"] == [
            {"reason": "contradictory", "says": REASON_SAYS["contradictory"]}]

    def test_the_nearest_allowed_game_is_offered(self) -> None:
        answer = designing.instead(_in_rounds())
        assert answer["searched"] is True and answer["offers"]
        for offer in answer["offers"]:
            assert validate_spec(offer["draft"]) == []

    def test_a_malformed_draft_gets_no_reason_and_no_offer(self) -> None:
        broken = copy.deepcopy(CHESS)
        broken["pieces"][0]["stamina"] = 2
        assert designing.check(broken)["reasons"] == []
        assert designing.instead(broken) == {"reasons": [], "offers": [], "searched": False}

    def test_an_allowed_draft_is_offered_nothing(self) -> None:
        assert designing.instead(CHESS) == {"reasons": [], "offers": [], "searched": False}

    def test_a_draft_that_is_not_a_draft_is_refused(self) -> None:
        with pytest.raises(designing.DesignError):
            designing.instead("not a draft")


# --------------------------------------------------------------- the route and the page


class TestTheRoute:
    def test_it_is_listed(self) -> None:
        assert routes.find("POST", "/api/design/instead") is not None

    def test_it_answers_over_the_socket(self, tmp_path: Path) -> None:
        rules = tmp_path / "rulesets"
        rules.mkdir()
        app = App(rules, tmp_path / "runs")
        server = Server(("127.0.0.1", 0), handler_for(app))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()
        try:
            body = json.dumps({"draft": CHESS, "rule": "simultaneous", "value": {}}).encode()
            request = Request(f"http://127.0.0.1:{server.server_port}/api/design/instead",
                              data=body, method="POST",
                              headers={"Content-Type": "application/json",
                                       REQUEST_HEADER: REQUEST_VALUE})
            with urlopen(request, timeout=60) as response:
                answer = json.loads(response.read().decode())
        finally:
            server.shutdown()
            server.server_close()
        assert answer["offers"][0]["edits"] == ROUNDS_EDITS

    def test_the_page_asks_it_per_row_and_for_the_draft(self) -> None:
        assert "api('/api/design/instead', { draft: DRAFT, rule: key, value: RULE_CARDS[key] })" \
            in DESIGN
        assert "api('/api/design/instead', { draft: JSON.parse(asked) })" in DESIGN
        # Asked when a row is opened, never for the whole menu at once (§6.2).
        assert "row.ontoggle" in DESIGN
        # Taking an offer is one step undo takes back whole, as a turn choice is.
        assert "snapshotDraft();\n  DRAFT = offer.draft;\n  snapshotDraft();" in DESIGN
        assert "No small change to this game makes it allowed." in DESIGN
