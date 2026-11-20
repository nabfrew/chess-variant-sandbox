"""PF1: the faults a first designer met on the way to Publish (`docs/reference/publish-walk.md`
§3, PW1-F1 to PW1-F10), held by running the page's own functions in node.

The site's half of F1 (the standing carrying the limit) is `test_site_admin.py`'s
`TestPublishing`; F2 (Enter on `/join`) is `test_site_requests.py`'s. These are the
designer's and the game page's drawings: each test names the fault it holds.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from test_web import _in_page, _kit, _said

from chess_variant_sandbox.site import drafts, terms

LIMIT = {
    "says": "You have published 1 game in the last day, which is the limit of 1, so this one "
            "cannot be published yet. You can still measure it and save it.",
    "opens": "2031-05-04T09:30:00+00:00",
}

DRAFT_PANEL = {
    "level": "Draft", "mine": True, "open": False, "publishable": True, "says": "",
    "waiting": "", "waitsForCheck": False, "stopped": "",
    "ticks": [{"met": True, "says": "it passes the playability check"},
              {"met": False, "says": drafts.READING_TICK}],
    "terms": {"version": terms.VERSION, "says": terms.DEDICATION, "href": terms.PATH},
}


def _visible(markup: str) -> str:
    """The words a reader sees: tags and their attributes gone, entities read."""
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def _drawn(standing: dict[str, Any], setup: str = "") -> str:
    return _said(_in_page(_kit() + "ACCOUNTS = true; VIEWER = 'ada';" + setup,
                          "publishHtml(" + json.dumps(standing) + ")"))


class TestTheLimitBeforeTheWork:
    """PW1-F1: the one-a-day limit is said beside Publish when the draft opens, with when the
    next publish opens, before the measuring control; the button is off with the reason."""

    def test_a_saved_draft_reads_the_limit_and_the_hour_beside_publish(self) -> None:
        drawn = _drawn({"hash": "h", "publish": DRAFT_PANEL, "publishLimit": LIMIT})
        seen = _visible(drawn)

        assert 'id="dPublishLimit"' in drawn
        assert LIMIT["says"] in seen
        assert "The next one can be published from" in seen
        # Before the ticks, the measuring control and the button: read before any work.
        assert drawn.index("dPublishLimit") < drawn.index('id="dPublishGo"')
        assert "Off until the limit above opens." in seen

    def test_rules_nobody_has_saved_yet_read_it_too(self) -> None:
        drawn = _drawn({"hash": "h", "track": [], "publishLimit": LIMIT})
        assert LIMIT["says"] in _visible(drawn)

    def test_the_on_trial_limit_names_no_hour(self) -> None:
        held = {"says": "You have 5 games on trial, which is the limit of 5.", "opens": None}
        seen = _visible(_drawn({"hash": "h", "publish": DRAFT_PANEL, "publishLimit": held}))
        assert held["says"] in seen and "can be published from" not in seen

    def test_no_limit_draws_nothing(self) -> None:
        assert "dPublishLimit" not in _drawn({"hash": "h", "publish": DRAFT_PANEL})

    def test_a_press_that_waits_is_not_told_a_limit(self) -> None:
        waiting = {**DRAFT_PANEL, "waitsForCheck": True, "waiting": drafts.WAITING}
        drawn = _drawn({"hash": "h", "publish": waiting, "publishLimit": LIMIT})
        assert "dPublishLimit" not in drawn

    def test_the_hour_is_written_in_the_readers_clock(self) -> None:
        words = _in_page("", "publishOpensWords(" + json.dumps(LIMIT["opens"]) + ")")
        assert words.startswith("The next one can be published from ")
        assert _in_page("", "publishOpensWords('not a time')") == ""


class TestTheConfirmationInSight:
    """PW1-F3: the confirmation carries the way to the game's page, survives a redraw of the
    panel, and the press brings it into view."""

    def test_it_links_the_games_page(self) -> None:
        drawn = _in_page("", "publishedHtml({ said: 'Published. It is on trial.' }, 'abc')")
        assert "Published. It is on trial." in drawn
        assert 'href="/game/abc"' in drawn and "open its page" in drawn

    def test_a_redraw_keeps_it(self) -> None:
        on_trial = {**DRAFT_PANEL, "level": "On trial", "publishable": False}
        setup = "PUBLISH_SAID = { hash: 'h', html: publishedHtml({ said: 'Published.' }, 'h') };"
        assert 'id="dPublished"' in _drawn({"hash": "h", "publish": on_trial}, setup)
        assert 'id="dPublished"' not in _drawn({"hash": "other", "publish": on_trial}, setup)

    def test_the_press_scrolls_it_into_view(self) -> None:
        from page_text import PAGE

        start = PAGE.index("if (publish) publish.onclick")
        press = PAGE[start:PAGE.index("// Q518. The track's", start)]
        assert "scrollIntoView({ block: 'center' })" in press


class TestOneTapToPublishOnAPhone:
    """PW1-F4: the foot bar, which is what a phone has on screen, carries a way down to the
    score-and-publish step; the stylesheet shows it only where the bar is fixed."""

    def test_the_bar_carries_it_and_it_goes_to_publish(self) -> None:
        from page_text import PAGE

        drawn = _in_page("ACCOUNTS = true;", "jumpPublishHtml({ valid: true })")
        assert 'id="dJumpPublish"' in drawn and "measure and publish it" in drawn
        assert _in_page("", "jumpPublishHtml({ valid: false })") == ""
        jump = PAGE[PAGE.index("function jumpToPublish"):]
        jump = jump[:jump.index("\n}\n")]
        assert "getElementById('dPublishLimit')" in jump and "getElementById('dHere')" in jump
        assert ".designer .footbar .jumpPublish { display: none; }" in PAGE
        assert "${jumpPublishHtml(check)}" in PAGE


class TestNothingStaleAfterSaveAndPublish:
    """PW1-F5: after a publish the save's sentence follows the standing; the family names
    the game on screen by its name field and shows a checked score from the designer's
    computer rather than *not measured*."""

    KEPT = {"created": True, "name": "my game", "hash": "h",
            "said": "Saved as a draft: only people with its link can open it until you publish it."}

    def test_the_save_box_follows_the_standing_once_published(self) -> None:
        on_trial = {"mine": True, "publishable": False,
                    "says": "On trial: listed with the other games."}
        setup = "DRAFT_STANDING = " + json.dumps({"hash": "h", "publish": on_trial}) + ";"
        seen = _visible(_in_page(_kit() + setup, "savedHtml(" + json.dumps(self.KEPT) + ")"))
        assert "On trial: listed with the other games." in seen
        assert "only people with its link" not in seen

    def test_a_draft_keeps_the_saves_own_sentence(self) -> None:
        draft = {"mine": True, "publishable": True, "says": "Saved as a draft."}
        setup = "DRAFT_STANDING = " + json.dumps({"hash": "h", "publish": draft}) + ";"
        seen = _visible(_in_page(_kit() + setup, "savedHtml(" + json.dumps(self.KEPT) + ")"))
        assert "only people with its link can open it until you publish it" in seen

    def test_the_save_sentence_names_no_measurement_it_cannot_know_of(self) -> None:
        row = drafts.Kept("abc", drafts.UNLISTED, created=True, revoked=False,
                          screen_passed=True)
        said = drafts.why(row, {"passed": True, "reasons": []})
        assert "once you have measured" not in said
        assert "Where this game stands" in said

    def test_the_family_names_the_game_by_its_name_field(self) -> None:
        setup = "DRAFT_HASH = 'h'; DRAFT = { name: 'the new name' };"
        tree = {"hash": "h", "draft": True, "here": True, "name": "los alamos variant"}
        seen = _visible(_in_page(setup, "lineageHtml(" + json.dumps(tree) + ")"))
        assert "the new name" in seen and "los alamos variant" not in seen

    def test_a_checked_score_from_the_designers_computer_is_not_not_measured(self) -> None:
        node = {"hash": "g", "known": True, "measured": {"state": "unscored"},
                "provisional": {"index": 0.745, "viable": True, "measured": True,
                                "state": "spot-checked", "sample": 32, "yours": True,
                                "seeds": [7], "weights": "v15"}}
        drawn = _in_page("ACCOUNTS = true;", "lineageStateHtml(" + json.dumps(node) + ")")
        assert "not measured" not in _visible(drawn)
        assert "spot-checked" in _visible(drawn) and "0.74" in drawn


class TestARuledOutReadingInWords:
    """PW1-F6: a failed test leads with its plain sentence and the two numbers in words; the
    code name is behind the (i), never in the text."""

    LINES = [
        "decisive_rate_worst_half=0.188 below minimum 0.2 (each side's computer players must "
        "finish enough of their games with a winner)",
        "search_advantage=0.41 below minimum 0.55",
        "top_two_seat_bias=1.2e-03 below minimum 0.01: the first seat went 30-2 with 0 draws",
    ]

    def test_no_code_name_or_equals_sign_shows(self) -> None:
        for line in self.LINES:
            drawn = _in_page("", "hereViolationHtml(" + json.dumps(line) + ")")
            seen = _visible(drawn)
            assert "=" not in seen, seen
            assert not re.search(r"\b[a-z]+_[a-z_]+\b", seen), seen
            name = line.split("=")[0]
            assert name in drawn, "the code name is still there, behind the (i)"

    def test_the_plain_part_leads_and_the_numbers_follow(self) -> None:
        seen = _visible(_in_page("", "hereViolationHtml(" + json.dumps(self.LINES[0]) + ")"))
        assert seen.startswith("Each side's computer players must finish")
        assert "it measured 0.188, where at least 0.2 is needed" in seen

    def test_a_line_in_another_shape_is_drawn_as_it_came(self) -> None:
        assert _in_page("", "hereViolationHtml('no move at the start')") == "no move at the start"

    def test_the_refusal_block_draws_its_lines_in_words(self) -> None:
        refusal = {"kind": "players", "head": "Ruled out on its one run.", "note": "A reading."}
        drawn = _in_page("", "hereRefusalHtml(" + json.dumps(refusal) + ", "
                         + json.dumps(self.LINES[:1]) + ")")
        seen = _visible(drawn)
        assert "Ruled out on its one run." in seen and "it measured 0.188" in seen
        assert "=" not in seen and drawn.count("<div") == drawn.count("</div>")


class TestWhyPublishIsOff:
    """PW1-F7: the reason is a visible line under the button, and the button's spoken name
    is its own word, not a title."""

    def test_the_reason_shows_and_the_name_is_publish(self) -> None:
        drawn = _drawn({"hash": "h", "publish": {**DRAFT_PANEL, "open": True}})
        button = re.search(r"<button id=\"dPublishGo\"[^>]*>([^<]*)</button>", drawn)
        assert button is not None and button.group(1).strip() == "publish"
        assert "title=" not in button.group(0)
        assert "Tick the box above to agree to the terms first." in _visible(drawn)

    def test_a_reason_from_the_server_shows_too(self) -> None:
        panel = {**DRAFT_PANEL, "waiting": drafts.NEEDS_A_READING}
        assert drafts.NEEDS_A_READING in _visible(_drawn({"hash": "h", "publish": panel}))

    def test_ticked_and_open_says_nothing(self) -> None:
        assert _in_page("", "publishWhy({ open: true }, null, true)") == ""


class TestTheTwoRatings:
    """PW1-F8: the checklist's official-measurement rung says it is not the rating of the
    computer players a person plays, which the site rates once the game is published."""

    def test_the_rung_names_the_difference_on_the_site(self) -> None:
        states = _in_page("ACCOUNTS = true; VIEWER = 'ada';",
                          "rungStates({ valid: true, hash: 'h', playable: true }, "
                          "{ hash: 'h', saved: true }, { hash: 'h', passed: true })")
        says = states["canonical"]["says"]
        assert says.startswith("not yet")
        assert "not the rating of the computer players you can play against" in " ".join(
            says.split())

    def test_the_local_application_is_not_told_it(self) -> None:
        states = _in_page("", "rungStates({ valid: true, hash: 'h', playable: true }, "
                              "{ hash: 'h', saved: true }, { hash: 'h', passed: true })")
        assert states["canonical"]["says"] == "not yet"


class TestTheDesignerBesideTheTitle:
    """PW1-F9: a member's game names its designer under its title, unfolded."""

    def test_a_designed_game_names_its_designer(self) -> None:
        drawn = _in_page("ACCOUNTS = true; VIEWER = 'bob';",
                         "designerByHtml({ source: 'designed', designer: { by: 'ada' } })")
        assert _visible(drawn) == "by ada" and 'href="/players/ada"' in drawn

    def test_any_other_game_names_nobody(self) -> None:
        assert _in_page("", "designerByHtml({ source: 'library' })") == ""
        assert _in_page("", "designerByHtml({ source: 'designed' })") == ""

    def test_it_is_drawn_under_the_title(self) -> None:
        from page_text import PAGE

        title = PAGE.index('<h2 class="u-m-10-0-0">${esc(game.name)}')
        assert PAGE.index("${designerByHtml(game)}", title) - title < 400


class TestProgressNeverGoesBack:
    """PW1-F10: one bar over both long steps, each step named as *step n of 2*."""

    def test_the_share_only_grows_across_both_steps(self) -> None:
        steps = """
          const run = { step: 'loading', done: 0, total: 0 };
          const seen = [];
          const at = (step, done, total) => { run.step = step; run.done = done;
            run.total = total; seen.push(hereShare(run)); };
          at('loading', 0, 0); at('gathering', 0, 0);
          for (let i = 0; i <= 16; i++) at('tournament', i, 16);
          at('gathering', 0, 0);
          for (let i = 0; i <= 32; i++) at('reading', i, 32);
          at('gathering', 0, 0);
        """
        seen = _in_page(steps, "seen")
        assert all(b >= a for a, b in zip(seen, seen[1:], strict=False)), seen
        assert seen[0] == 0 and seen[-1] == 100
        assert 45 <= seen[18] <= 55, "the reading-back step starts at half way"

    def test_each_step_is_named_with_its_place(self) -> None:
        line = _in_page(
            "", "hereRunHtml({ state: 'running', step: 'reading', done: 3, total: 32, "
                "started: 0, workers: 4, games: 256, name: 'x' })")
        seen = _visible(line)
        assert "step 2 of 2, reading every game back — 3 of 32" in seen
        assert "% of the way" in seen
