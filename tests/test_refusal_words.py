"""R-S17a: no refusal is shown without its test, its run count and the players' part.

`docs/reference/wrong-no.md` §2 items 2 and 4. A refusal used to read *It is ruled out, not
marked down* and nothing of which test, how many runs, or that the project's own computer
players produced it. Each now says all three, and splits by which test failed into *by its
rules* (true whoever plays) and *as these computer players play it* (a reading that can
change). Words only: nothing here moves a verdict, a score or a stamp, which the last class
holds.
"""

from __future__ import annotations

import copy

from test_sitescore import SHAPES

from chess_variant_sandbox.web import library, sitescore

# One test of each kind, in the sentences `metrics/` stores.
BRANCHING = ("mean_branching=1.200 below minimum 1.5 (with no choices there is no play)")
QUICK_KILL = SHAPES["gate above"]


def _score(violations: list[str], seeds: list[int], **extra: object) -> dict[str, object]:
    return {"viable": False, "measured": True, "violations": violations, "seeds": seeds,
            "index": 0.7, **extra}


class TestTheKindOfACheck:
    """The split, by which test failed: one of each kind, then every stored shape."""

    def test_few_moves_a_turn_is_the_rules(self) -> None:
        assert sitescore.check_of(BRANCHING) == ("choices per turn", sitescore.RULES)

    def test_quick_kills_are_the_players(self) -> None:
        assert sitescore.check_of(QUICK_KILL) == ("quick kills, worse half", sitescore.PLAYERS)

    def test_every_stored_shape_has_a_kind_or_is_no_test(self) -> None:
        for name, text in SHAPES.items():
            found = sitescore.check_of(text)
            if name.startswith("unreadable"):
                assert found is None, name
            else:
                assert found is not None and found[1] == sitescore.PLAYERS, name

    def test_a_run_s_place_and_a_split_are_read_through(self) -> None:
        assert sitescore.check_of("run 2000: " + BRANCHING)[1] == sitescore.RULES
        assert sitescore.check_of(
            "Three of its four runs refuse it and one accepts it, so it is ruled out.") is None

    def test_an_unknown_sentence_reads_as_the_players(self) -> None:
        """The kind that admits it may be wrong: the costly error is the wrong "no"."""
        assert sitescore.check_of("something new")[1] == sitescore.PLAYERS


class TestARefusalSaysAllThree:
    """Its test, its run count, and the players' part, on one run and on a library vote."""

    def test_one_run_by_the_players(self) -> None:
        said = sitescore.refusal(_score([QUICK_KILL], [0]))
        assert said["kind"] == sitescore.PLAYERS
        assert "on its one run" in said["head"] and "quick kills, worse half" in said["head"]
        assert "as these computer players play it" in said["head"]
        assert said["note"] == sitescore.PLAYERS_NOTE
        assert "can change" in said["note"] and "measured again" in said["note"]

    def test_one_run_by_the_rules(self) -> None:
        said = sitescore.refusal(_score([BRANCHING], [0]))
        assert said["kind"] == sitescore.RULES and "by its rules" in said["head"]
        assert said["note"] == sitescore.RULES_NOTE and "whoever plays" in said["note"]

    def test_both_kinds(self) -> None:
        said = sitescore.refusal(_score([BRANCHING, QUICK_KILL], [0]))
        assert said["kind"] == "both" and said["note"] == sitescore.BOTH_NOTE
        assert "tests" in said["head"]

    def test_a_library_vote_names_its_runs(self) -> None:
        verdict = {"accept": 1, "refuse": 3, "split": "One of its four runs accepts it"}
        said = sitescore.refusal(_score(
            ["run 0: " + QUICK_KILL, "run 1000: " + QUICK_KILL],
            [0, 1000, 2000, 3000], verdict=verdict))
        assert "by three of its four runs" in said["head"]
        assert [check["title"] for check in said["checks"]] == ["quick kills, worse half"]
        unanimous = sitescore.refusal(_score([QUICK_KILL], [0, 1000, 2000, 3000]))
        assert "on all four of its runs" in unanimous["head"]

    def test_a_pooled_seat_alone_is_the_runs_together(self) -> None:
        verdict = {"accept": 4, "refuse": 0, "split": ""}
        said = sitescore.refusal(_score([SHAPES["seat, pooled"]], [0, 1000, 2000, 3000],
                                        verdict=verdict))
        assert "over its four runs taken together" in said["head"]
        assert said["kind"] == sitescore.PLAYERS

    def test_no_refusal_where_there_is_none(self) -> None:
        assert sitescore.refusal(None) is None
        assert sitescore.refusal({"viable": True, "violations": []}) is None
        assert sitescore.refusal(_score(["Not measured: x."], [0], measured=False)) is None


class TestEveryPayloadCarriesIt:
    """The page draws `refusal`; every payload that can say *ruled out* sends one."""

    def test_the_worded_score(self) -> None:
        worded = sitescore.worded(_score([QUICK_KILL], [0]))
        assert worded["refusal"]["kind"] == sitescore.PLAYERS
        assert sitescore.worded({"viable": True, "violations": []})["refusal"] is None

    def test_the_cheap_check(self) -> None:
        rules = sitescore.smoke_refusal(["the first player has no legal move from the "
                                         "starting position"])
        assert rules["kind"] == sitescore.RULES and rules["note"] == sitescore.RULES_NOTE
        thin = sitescore.smoke_refusal(["a player has 1.20 moves to choose from on average; "
                                        "there is nothing to decide"])
        assert thin["kind"] == sitescore.RULES
        random_play = sitescore.smoke_refusal(["none of 150 random games had a winner; from "
                                               "this start, the way to win may be out of reach"])
        assert random_play["kind"] == sitescore.PLAYERS
        assert random_play["note"] == sitescore.RANDOM_NOTE
        assert sitescore.smoke_refusal([]) is None

    def test_the_designer_s_check_and_the_card(self) -> None:
        """A draft whose random games stall reads as random play, and a playable one
        carries no refusal; the card's playability carries the same."""
        import json

        from chess_variant_sandbox.web import designing

        template = next(t for t in designing.templates() if t["key"] == "skirmish")
        draft = json.loads(json.dumps(template["spec"]))
        assert designing.check(draft)["playableRefusal"] is None
        draft["rules"].update({"max_plies": 4, "no_progress_draw": 1000})
        checked = designing.check(draft)
        assert checked["playable"] is False
        assert checked["playableRefusal"] == sitescore.smoke_refusal(checked["playableReasons"])
        assert checked["playableRefusal"]["kind"] == sitescore.PLAYERS
        card = library._playability(draft)
        assert card["playableRefusal"]["kind"] == sitescore.PLAYERS

    def test_the_browser_s_words(self) -> None:
        words = sitescore.refusal_words()
        assert words["rulesGates"] == ["mean_branching"]
        assert {words["rulesNote"], words["playersNote"], words["bothNote"]} == {
            sitescore.RULES_NOTE, sitescore.PLAYERS_NOTE, sitescore.BOTH_NOTE}


class TestNoVerdictMoves:
    """Words only: the score a refusal is read from is left as it was."""

    def test_the_score_is_untouched(self) -> None:
        score = _score([QUICK_KILL, BRANCHING], [0, 1000, 2000, 3000],
                       verdict={"accept": 0, "refuse": 4, "split": ""})
        before = copy.deepcopy(score)
        sitescore.refusal(score)
        sitescore.worded(score)
        assert score == before


class TestThePageDrawsIt:
    """No refusal on a page is the bare *ruled out*: each reader goes through the words."""

    def test_the_old_sentence_is_gone(self) -> None:
        from page_text import PAGE

        assert "It is ruled out${info('gated')}, not marked down." not in PAGE
        assert "does not work as a game" not in PAGE
        assert "function refusalHtml(" in PAGE and "function refusalText(" in PAGE
        assert "function hereRefusal(" in PAGE
        assert "SCREEN_REFUSAL_NOTE" in PAGE and "playableRefusal" in PAGE

    def test_each_reader_draws_the_refusal(self) -> None:
        from page_text import PAGE

        # The game's page, the browser's measurement, the designer's rung, the card's
        # hover and the map's panel.
        assert PAGE.count("refusalHtml(score.refusal") == 1
        # PF1 (PW1-F6): through `hereRefusalHtml`, which words the measurement's lines.
        assert PAGE.count("hereRefusalHtml(hereRefusal(r)") == 1
        assert PAGE.count("refusalText(m.refusal)") == 2
        assert "refusalText(game.refusal)" in PAGE
        assert "refusalText(score.refusal)" in PAGE


class TestAReadingThatStandsByIdentity:
    """`strength-check-all-games.md` §18's last paragraph: the page says whose players."""

    @staticmethod
    def _reading(stands_by: str | None, belongs: bool) -> dict[str, object]:
        from chess_variant_sandbox import measurement

        return {"word": measurement.HELD_OWN, "recorded": measurement.HELD_OWN,
                "belongs": belongs, "standsBy": stands_by, "readAt": "2026-11-13", "c": 0.55}

    def test_identity_names_the_earlier_players(self) -> None:
        said = library.strength_payload(self._reading("identity", True))
        assert said["details"][0] == f"Checked on 13 November 2026, on {library.STRENGTH_BY_IDENTITY}."
        assert "move for move" in said["details"][0]
        assert "that measured this score" not in " ".join(said["details"])

    def test_its_own_reading_keeps_its_words(self) -> None:
        said = library.strength_payload(self._reading("reading", True))
        assert "on the computer players that measured this score" in said["details"][0]


class TestTheWordsAroundIt:
    """The tier and the glossary say the same: a reading on these players."""

    def test_below_chess_is_a_reading(self) -> None:
        note = library.TIER_WORDS[library.BELOW]["note"]
        assert "these computer players" in note and "not a verdict" in note
        assert "not a ranking of worth" in library.TIER_SENTENCE

    def test_the_site_s_account_drops_means_nothing(self) -> None:
        from chess_variant_sandbox.metrics.glossary import SITE_HOW_IT_WORKS

        body = SITE_HOW_IT_WORKS[0]["body"]
        assert "means nothing" not in body
        assert "whoever plays" in body and "can change" in body
