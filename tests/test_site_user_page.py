"""The user page in the chosen direction (UP2, `docs/archive/plan-user-page.md` §5, direction B).

What waits on you leads, a row each with the position as the engine left it and one action;
your games stand on the path every game takes (`workshop.columns`), each with its board, the
change from its parent, the designer's rungs and its next step in the server's words; the
record, what you have done grouped by kind and day, proven positions in one line; settings
folded at the foot; and a new member's page opens on *Start building a game*.

The page works nothing out (rule 3): the columns, the next steps and the positions are the
server's, and the boards are the application's own still board (`stillboard.js`).
"""

from __future__ import annotations

import json
from typing import Any

import page_text
from test_site import _done, _rendered

from chess_variant_sandbox.site import drafts, stages, workshop
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.web.page import SCRIPT_DIR as PAGE_DIR


def _kept(level: str, reading: str = "none", screened: bool = True) -> workshop.Kept:
    return workshop.Kept(hash="h" + level[:3], name=level, spec={}, parent="", screened=screened,
                         level=level, waking=False, reading=reading)


class TestTheColumns:
    """Every column is a level `stages` knows, or *Measured*, read from the attached score."""

    def test_a_draft_stands_in_draft_until_its_score_is_checked(self) -> None:
        assert workshop.column(_kept(stages.DRAFT)) == stages.DRAFT
        assert workshop.column(_kept(stages.DRAFT, "checking")) == stages.DRAFT
        assert workshop.column(_kept(stages.DRAFT, "checked")) == workshop.MEASURED

    def test_a_baseline_game_is_in_the_librarys_column(self) -> None:
        assert workshop.column(_kept(stages.BASELINE)) == stages.LIBRARY
        for level in (stages.ON_TRIAL, stages.SHORTLISTED, stages.LIBRARY, stages.RESTING):
            assert workshop.column(_kept(level)) == level

    def test_the_path_is_drawn_whole_and_an_off_path_level_only_when_it_holds_a_game(
        self,
    ) -> None:
        groups = [{"level": "x", "games": [
            {"hash": "a", "column": stages.DRAFT}, {"hash": "b", "column": workshop.MEASURED},
            {"hash": "c", "column": stages.RESTING}, {"hash": "d", "column": stages.DRAFT},
        ]}]
        found = workshop.columns(groups)
        assert [column["word"] for column in found] == [*workshop.COLUMNS, stages.RESTING]
        held = {column["word"]: column["games"] for column in found}
        assert held[stages.DRAFT] == ["a", "d"] and held[workshop.MEASURED] == ["b"]
        assert held[stages.ON_TRIAL] == []
        known = {*stages.LEVELS, workshop.MEASURED}
        assert all(column["word"] in known and column["says"] for column in found)

    def test_only_a_published_column_names_a_game_to_show_large(self) -> None:
        found = workshop.columns([{"games": [{"hash": "a", "column": stages.DRAFT}]}])
        featured = [column["word"] for column in found if column["feature"]]
        assert featured == [stages.ON_TRIAL, stages.SHORTLISTED, stages.LIBRARY]

    def test_the_page_shows_the_game_furthest_along_with_the_servers_kicker(self) -> None:
        trial = {**GAME, "hash": "tttt", "name": "five stones", "step": "It can be nominated.",
                 "track": [{"word": "Draft", "lit": False}, {"word": "On trial", "lit": True}]}
        columns = [{**column, "feature": f"kicker {index}" if index > 1 else "",
                    "games": ["tttt"] if index == 2 else column["games"]}
                   for index, column in enumerate(COLUMNS)]
        said = _rendered(
            f"const shown = drawFeatured({json.dumps([{'games': [GAME, trial]}])},"
            f" {json.dumps(columns)});"
            "console.log(JSON.stringify([shown.name,"
            " document.getElementById('featured').textContent]));"
        )
        assert said == ["five stones", "kicker 2five stonesDraftOn trialIt can be nominated."]

    def test_nobody_who_made_nothing_gets_an_empty_path(self) -> None:
        assert workshop.columns([]) == []

    def test_measured_says_what_the_designer_reads_of_a_checked_score(self) -> None:
        assert workshop.COLUMN_SAYS[workshop.MEASURED] == drafts.READING_CHECKED


class TestTheNextStep:
    def test_a_drafts_next_step_is_the_designers_next_rung(self) -> None:
        said = stages.NEXT[stages.DRAFT]
        assert drafts.READING_TICK in workshop.step(_kept(stages.DRAFT), said)
        assert "playability check" in workshop.step(_kept(stages.DRAFT, screened=False), said)
        assert drafts.READING_CHECKING in workshop.step(_kept(stages.DRAFT, "checking"), said)
        assert "publish it" in workshop.step(_kept(stages.DRAFT, "checked"), said)

    def test_a_publish_waiting_on_the_check_and_every_later_level_keep_the_servers_words(
        self,
    ) -> None:
        assert workshop.step(_kept(stages.DRAFT, "checked"), stages.WAITING) == stages.WAITING
        said = stages.NEXT[stages.ON_TRIAL]
        assert workshop.step(_kept(stages.ON_TRIAL), said) == said


PROFILE = (site_pages.SCRIPT_DIR / "profile.js").read_text(encoding="utf-8")


class TestThePageDrawsBoardsWithTheApplicationsStillBoard:
    def test_the_page_loads_the_still_board_before_its_own_script(self) -> None:
        page = site_pages.profile("ada")
        routes = [f"/static/page/{name}" for name in site_pages.BOARD_SCRIPTS]
        at = [page.index(route) for route in routes]
        assert at == sorted(at) and at[-1] < page.index("/static/site/still")
        assert page.index("/static/site/still") < page.index("/static/site/profile")
        assert site_pages.BOARD_SCRIPTS == ("wrapped", "wood", "stillboard")

    def test_the_board_is_the_only_markup_and_comes_from_board_preview(self) -> None:
        """Every name on the page is a text node; the one string put in as markup is the
        board `boardPreview` drew, every name in which went through `esc`."""
        assert "innerHTML" not in page_text.site_own(site_pages.profile("ada"))
        assert "insertAdjacentHTML" not in PROFILE
        # UP4: the adapter is `still.js`, shared with a member's page.
        still = (site_pages.SCRIPT_DIR / "still.js").read_text(encoding="utf-8")
        assert "createContextualFragment(" not in PROFILE
        assert still.count("createContextualFragment(") == 1
        body = still[still.index("function stillNode"):]
        body = body[: body.index("\n}\n")]
        assert "createContextualFragment(drawn)" in body and "= boardPreview(" in body

    def test_no_square_is_worked_out_here(self) -> None:
        for word in ("squareXY", "squareName", "% preview.width", "lastMove.from"):
            assert word not in PROFILE, word

    def test_the_last_move_is_tinted_by_one_answer_shared_with_the_play_board(self) -> None:
        board = (PAGE_DIR / "board.js").read_text(encoding="utf-8")
        still = (PAGE_DIR / "stillboard.js").read_text(encoding="utf-8")
        assert "lastMoveHas(last, square)" in board and "function lastMoveHas" in still
        assert "square === last.from" not in board
        assert page_text.PAGE.count("square === last.from") == 1


def _pipe(groups: list[dict[str, Any]], columns: list[dict[str, Any]], lead: bool) -> Any:
    return _rendered(
        f"drawYourGames({json.dumps(groups)}, {json.dumps(columns)}, {json.dumps(lead)});"
        "const mine = document.getElementById('mine');"
        "const links = [];"
        "(function walk(node) {"
        "  if (node.href !== undefined) links.push([node.href, node.textContent,"
        "    node.className]);"
        "  (node.children || []).forEach(walk);"
        "})(mine);"
        "console.log(JSON.stringify({said: mine.textContent, links,"
        " cols: (mine.children[2] || {children: []}).children.map(col => col.className)}));"
    )


GAME = {"hash": "aaaa", "name": "one stone", "inDesigner": True, "tagCounts": [],
        "still": None, "column": "Draft", "step": "Next: a score measured on your computer.",
        "change": ["Rule changed: three is now one."],
        "rungs": [{"key": "validate", "name": "complete", "state": "done", "says": "yes",
                   "next": False}],
        "parent": {"hash": "pppp", "name": "three stones", "level": "Library",
                   "score": {"index": 0.8738, "status": "current", "seeds": [0, 1, 2, 3],
                             "certified": {"sample": 128}}}}
COLUMNS = [{"word": word, "says": "its line", "games": ["aaaa"] if word == "Draft" else []}
           for word in workshop.COLUMNS]


class TestTheWorkshop:
    def test_each_game_stands_in_the_servers_column_with_its_change_and_step(self) -> None:
        drawn = _pipe([{"level": "Draft", "games": [GAME]}], COLUMNS, lead=True)
        said = drawn["said"]
        assert said.startswith("Your workshopStart a new game")
        assert "Made from three stones (Library, 0.874 · 4 runs, re-certified from a sample)" \
            ": one stone." in said
        assert "Rule changed: three is now one." in said
        assert "Next: a score measured on your computer." in said
        assert drawn["cols"] == ["pipeCol first", "pipeCol empty", "pipeCol empty",
                                 "pipeCol empty", "pipeCol empty"]

    def test_start_and_try_another_change_open_the_designer(self) -> None:
        drawn = _pipe([{"level": "Draft", "games": [GAME]}], COLUMNS, lead=True)
        assert ["/#/design", "Start a new game", "button primary"] in drawn["links"]
        # *Try another change* opens the designer on a copy of the game it was made from.
        assert ["/#/design/pppp", "three stones", ""] in drawn["links"]
        # UP7: your own draft opens as itself, never as a copy of it.
        assert ["/#/design/aaaa?as=draft", "open in the designer", "go"] in drawn["links"]

    def test_with_something_waiting_start_a_new_game_is_not_a_primary(self) -> None:
        drawn = _pipe([{"level": "Draft", "games": [GAME]}], COLUMNS, lead=False)
        assert ["/#/design", "Start a new game", "button"] in drawn["links"]

    def test_a_provisional_parent_score_is_marked_with_its_basis(self) -> None:
        game = {**GAME, "parent": {**GAME["parent"], "score": {
            "index": 0.81, "status": "provisional", "seeds": [0], "revision": "v11",
            "measuredAt": "2026-10-03"}}}
        said = _pipe([{"level": "Draft", "games": [game]}], COLUMNS, lead=True)["said"]
        assert "provisional: 0.810 · 1 run (v11, 3 Oct 2026)" in said

    def test_nothing_made_is_the_door_and_a_line_never_an_empty_path(self) -> None:
        drawn = _pipe([], [], lead=True)
        assert drawn["said"].startswith("Your workshopStart a new gameYou have not made a game")
        assert drawn["cols"] == []


class TestWhatWaits:
    ENTRY = {"id": "m1", "game": "gggg", "name": "chess", "opponent": "grace", "seat": 1,
             "yourMove": True, "rated": True, "plies": 1, "moved": "2026-10-26T10:00:00+00:00",
             "toMove": "You have 4 days 23 hours to move.", "position": None,
             "lastMove": None}

    def test_your_move_is_a_row_and_their_move_a_quiet_line_with_the_rule_once(self) -> None:
        theirs = {**self.ENTRY, "id": "m2", "name": "cylinder chess", "yourMove": False,
                  "toMove": "They have 2 days to move."}
        between = {"running": [self.ENTRY, theirs], "finished": None,
                   "rule": "The player to move loses after five days without a move."}
        said = _rendered(
            f"drawBetween({json.dumps(between)});"
            "console.log(JSON.stringify({rows: document.getElementById('between').textContent,"
            " quiet: document.getElementById('between-quiet').textContent}));"
        )
        assert "Your move · by correspondence" in said["rows"]
        assert "Make your move" in said["rows"] and "cylinder chess" not in said["rows"]
        assert said["quiet"].startswith("Their movecylinder chess against grace")
        assert said["quiet"].count("five days") == 1

    def test_a_challenge_for_you_is_a_row_with_accept_and_decline(self) -> None:
        asked = {"forYou": [{"id": 3, "from": "lin", "game": "tttt", "name": "three stones",
                             "seat": "either", "rated": True, "rematch": False,
                             "made": "2026-10-26T10:00:00+00:00", "position": None,
                             "lastMove": None}],
                 "sent": [], "open": [], "pausedSays": None}
        said = _rendered(
            f"drawChallenges({json.dumps(asked)});"
            "console.log(JSON.stringify(document.getElementById('challenges').textContent));"
        )
        assert said.startswith("A challenge for youthree stoneslin challenged you")
        assert said.endswith("Acceptdecline")


class TestTheRecordAndWhatYouHaveDone:
    FIRSTS = [
        {"kind": "first-here", "rule": "The first rated game anybody finished on this game.",
         "when": f"2026-10-24T1{hour}:00:00+00:00", "game": game * 3, "name": game,
         "seat": hour % 2}
        for hour, game in enumerate(("atomic", "chess", "courier"))
    ]

    def test_one_kind_on_one_day_is_one_line_naming_the_games_and_its_rule_once(self) -> None:
        said = _done(self.FIRSTS)
        rule = self.FIRSTS[0]["rule"]
        assert "The first game played here, on courier, chess and atomic, 24 Oct 2026." in said
        assert said.count(rule) == 1
        assert "playing" not in said

    def test_a_kind_on_two_days_is_two_lines(self) -> None:
        other = {**self.FIRSTS[0], "when": "2026-10-25T09:00:00+00:00"}
        assert _done([other, *self.FIRSTS[1:]]).count(self.FIRSTS[0]["rule"]) == 2

    def test_proven_positions_are_one_line_with_their_rule(self) -> None:
        lines = [{"hash": "pppp", "name": "almost chess", "of": 10, "solved": 2, "share": "20%",
                  "converted": 2, "found": 2, "singles": 10}]
        said = _rendered(
            f"drawProven({json.dumps(lines)}, {{said: 'You have solved 2 of 177 (1%).'}});"
            "const where = document.getElementById('proven');"
            "console.log(JSON.stringify([where.children.length, where.textContent]));"
        )
        assert said[0] == 2
        assert said[1].startswith("Proven positions. You have solved 2 of 177 (1%). almost "
                                  "chess: solved 2 of 10 (20%), converted 2 of 10, found the "
                                  "move in 2 of 10.")
        assert "Only you see this." in said[1]


class TestTheOrder:
    def test_what_waits_leads_then_the_workshop_record_done_and_settings_folded(self) -> None:
        page = site_pages.profile("ada")
        order = ['id="start"', 'id="waiting"', 'id="mine"', 'id="record-part"', 'id="done"',
                 '<details class="settings"', 'id="move-email"', 'id="invite"', 'id="out"']
        at = [page.index(mark) for mark in order]
        assert at == sorted(at)

    def test_every_settings_box_keeps_its_sentence(self) -> None:
        for name in ("drawMoveEmail", "drawTakesChallengers", "drawYourPage", "drawInvite"):
            body = PROFILE[PROFILE.index(f"function {name}"):]
            body = body[: body.index("\n}\n")]
            assert "under" in body, name
