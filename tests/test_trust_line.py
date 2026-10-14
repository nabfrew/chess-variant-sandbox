"""The trust line on the pages (Q647, `docs/plan-confidence.md` T2, §2.1 and §2.2).

What is held down here, on synthetic records for a five-square game (never chess):

- **The words.** Each level's lead sentence is §2.2's, opened by the evidence that made it;
  the line carries the level word and no count (§5 Q2, answered A); *more* carries one line
  per piece of evidence there is, each with at most one plain count and a month, and each
  earlier score dated and labelled with its players (§5 Q1, answered A). The maintainer's
  numbers ride in `numbers` alone, and none of a visitor's words holds a phrase of the
  words guard (`site_words.PHRASES`, *ceiling*, *floor* and *verdict* among them).
- **People move the level, never the score** (§5 Q3, answered A): `trust.with_people`
  re-reads the level from the counts with no file read again.
- **The payloads.** `/api/library` cards carry `trust` (the chip) and `/api/game` the line;
  the strength check is no longer sent or drawn on its own.
- **The page.** The card draws the chip beside the score, the game page the line under it;
  no level, nothing drawn. `premiseHtml` is gone.
- **The site** (needs a database): a visitor is sent no numbers, an admin the numbers,
  and a person's clears question the game on its card and its page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_corpus import SPEC

from chess_variant_sandbox import corpus, history, premise, trust
from chess_variant_sandbox.spec import spec_hash

DIGEST = spec_hash(SPEC)


def _item(kind: str, reading: str, sentence: str, state: str = trust.CURRENT,
          count: trust.Count | None = None, date: str | None = None,
          **detail: Any) -> trust.Item:
    return trust.Item(kind, state, reading, count=count, date=date, sentence=sentence,
                      detail=detail)


def _trust(*items: trust.Item, score: str = trust.CURRENT) -> trust.Trust:
    every = list(items)
    for kind in trust.KINDS:
        if not any(item.kind == kind for item in every):
            every.append(trust.Item(kind, trust.ABSENT, trust.SILENT,
                                    sentence=f"{kind}.absent"))
    return trust.Trust(DIGEST, score,
                       trust.level_of(every) if score == trust.CURRENT else None,
                       tuple(every), trust.RULE.spelled())


ORDERED = _item(trust.LADDER, trust.SILENT, "ladder.ordered", count=trust.Count(0, 4, "runs"))
WITHHELD = _item(trust.LADDER, trust.FINDS_MORE, "ladder.withheld",
                 count=trust.Count(4, 4, "runs"), withheld=["skill_gradient"])
PARTLY = _item(trust.LADDER, trust.SILENT, "ladder.partly", count=trust.Count(2, 4, "runs"))
PARTLY_PARTS = _item(trust.LADDER, trust.SILENT, "ladder.partly-parts",
                     count=trust.Count(2, 4, "runs"))
FAIRY_HELD = _item(trust.FAIRY, trust.AGREES, "fairy.held-its-own",
                   count=trust.Count(12, 32, "games"), date="2026-10-02", score=0.37)
FAIRY_BEATS = _item(trust.FAIRY, trust.FINDS_MORE, "fairy.beats-ours",
                    count=trust.Count(6, 32, "games"), date="2026-10-02", score=0.19)
NET = _item(trust.NET, trust.FINDS_MORE, "net.beats-ours", count=trust.Count(30, 48, "games"),
            date="2026-09-30", procedure="1d160257b8e5", score=0.615)


def _visitor_words(shown: dict[str, Any]) -> list[str]:
    return [shown["level"], shown["said"], *shown["more"]]


# ------------------------------------------------------------------ the words


class TestTheWords:
    def test_an_early_estimate_says_so_and_invites_the_visitor(self) -> None:
        shown = trust.shown(_trust(ORDERED))
        assert shown is not None
        assert shown["level"] == trust.EARLY
        assert shown["said"] == trust.EARLY_SAID
        assert "If you beat the strongest computer player here" in shown["said"]
        assert shown["more"][0].startswith("Its computer players get steadily stronger")

    def test_a_partly_blind_ladder_names_its_runs_and_keeps_the_level(self) -> None:
        # Q793: *dull*, blind on 2 of 4 runs, no longer reads *steadily stronger*.
        assert _trust(PARTLY).level == _trust(ORDERED).level == trust.EARLY
        shown = trust.shown(_trust(PARTLY))
        assert shown is not None and shown["said"] == trust.EARLY_SAID
        assert shown["more"][0] == (
            "Its computer players got clearly stronger with more thinking time in only 2 of "
            "the 4 runs that measured this game, so the score rests on 2 runs alone.")
        parts = trust.shown(_trust(PARTLY_PARTS))
        assert parts is not None and parts["more"][0] == (
            "Its computer players got clearly stronger with more thinking time in only 2 of "
            "the 4 runs that measured this game, so parts of the score rest on those runs "
            "alone.")

    def test_questioned_opens_with_the_finding(self) -> None:
        shown = trust.shown(_trust(ORDERED, FAIRY_BEATS))
        assert shown is not None and shown["level"] == trust.QUESTIONED
        assert shown["said"].startswith("An engine written elsewhere (Fairy-Stockfish) beats")
        assert shown["said"].endswith(trust.QUESTIONED_TAIL)

    def test_a_current_finding_opens_before_a_stale_one(self) -> None:
        stale = _item(trust.NET, trust.FINDS_MORE, "net.beats-ours", state=trust.STALE)
        shown = trust.shown(_trust(stale, FAIRY_BEATS))
        assert shown is not None and shown["said"].startswith("An engine written elsewhere")

    def test_checked_opens_with_the_agreement(self) -> None:
        shown = trust.shown(_trust(ORDERED, FAIRY_HELD))
        assert shown is not None and shown["level"] == trust.CHECKED
        assert shown["said"] == (
            "An engine written elsewhere (Fairy-Stockfish) played ours on this game, and ours "
            "held its own. " + trust.CHECKED_TAIL)

    def test_the_line_is_a_word_and_a_sentence_and_carries_no_count(self) -> None:
        """§5 Q2, answered A: the level is a word only; counts live under *more*."""
        for found in (_trust(ORDERED), _trust(FAIRY_HELD), _trust(FAIRY_BEATS, NET),
                      _trust(WITHHELD),
                      trust.with_people(_trust(ORDERED), trust.People(9, 4, 2)),
                      trust.with_people(_trust(ORDERED), trust.People(25, 6, 0))):
            shown = trust.shown(found)
            assert shown is not None
            assert not any(ch.isdigit() for ch in shown["said"]), shown["said"]
            card = trust.card(found)
            assert card == {"level": shown["level"], "said": shown["said"]}

    def test_more_carries_one_plain_count_and_a_month(self) -> None:
        shown = trust.shown(_trust(ORDERED, FAIRY_BEATS, NET))
        assert shown is not None
        [fairy] = [line for line in shown["more"] if "Fairy-Stockfish" in line]
        assert fairy == ("An engine written elsewhere (Fairy-Stockfish) beats ours on this "
                         "game, so ours are missing something. Ours scored 6 of 32 games "
                         "against it (October 2026).")
        [net] = [line for line in shown["more"] if "learned" in line]
        assert "30 of 48 games" in net and net.endswith("(September 2026).")

    def test_the_self_duel_is_said_under_more_only(self) -> None:
        duel = _item(trust.SELF_DUEL, trust.REPORTED, "self-duel.reported",
                     count=trust.Count(5, 32, "games"))
        shown = trust.shown(_trust(ORDERED, duel))
        assert shown is not None and shown["level"] == trust.EARLY
        assert any("so it is not counted" in line for line in shown["more"])
        assert "copy" not in shown["said"]

    def test_a_stale_piece_says_it_was_found_with_earlier_players(self) -> None:
        stale = _item(trust.FAIRY, trust.FINDS_MORE, "fairy.beats-ours", state=trust.STALE)
        shown = trust.shown(_trust(stale))
        assert shown is not None and shown["level"] == trust.QUESTIONED
        assert trust.EARLIER in shown["said"]

    def test_an_absent_fairy_reading_says_whether_it_could_play_at_all(self) -> None:
        found = _trust(ORDERED)
        assert trust.FAIRY_NOT_TRIED in trust.shown(found, fairy_translates=True)["more"]
        assert any("cannot play its rules" in line
                   for line in trust.shown(found, fairy_translates=False)["more"])
        assert not any("Fairy-Stockfish" in line for line in trust.shown(found)["more"])

    def test_people_are_said_with_one_count(self) -> None:
        nobody = trust.with_people(_trust(ORDERED), trust.People(0, 0, 0))
        assert trust.NOBODY_YET in trust.shown(nobody)["more"]
        few = trust.with_people(_trust(ORDERED), trust.People(4, 3, 0, "2026-10-01"))
        assert ("3 people have played the strongest computer player here; none has beaten it "
                "yet (October 2026).") in trust.shown(few)["more"]
        cleared = trust.with_people(_trust(ORDERED), trust.People(9, 4, 2))
        assert trust.shown(cleared)["more"][-1].startswith(
            "2 people have beaten the strongest computer player here.")
        # The line itself carries no count: the level is a word only.
        assert trust.shown(cleared)["said"].startswith(trust.PEOPLE_CLEARED_BARE)
        unbeaten = trust.with_people(_trust(ORDERED), trust.People(25, 6, 0))
        assert trust.shown(unbeaten)["said"].startswith(trust.PEOPLE_UNBEATEN_BARE)

    def test_an_earlier_score_is_dated_and_labelled_with_its_players(self) -> None:
        """§5 Q1, answered A: under *more*, never the headline."""
        was = [{"index": 0.608, "runs": 4, "measured": "2026-09-28"}]
        shown = trust.shown(_trust(ORDERED), was=was)
        assert shown is not None
        assert shown["more"][-1] == (
            "Was 0.608 (4 runs) when measured by the earlier computer players (September 2026).")
        assert "0.608" not in shown["said"]

    def test_no_level_no_line(self) -> None:
        stale = _trust(FAIRY_BEATS, score=trust.STALE)
        assert trust.shown(stale) is None and trust.card(stale) is None
        assert trust.shown(None) is None and trust.card(None) is None

    def test_the_numbers_are_the_maintainers_alone(self) -> None:
        found = _trust(ORDERED, FAIRY_BEATS, NET)
        mine = trust.shown(found, premise_words="Scored by players not yet strong enough.")
        assert mine is not None
        assert any("score 0.19" in line for line in mine["numbers"])
        assert any(line.startswith("strength check: Scored by") for line in mine["numbers"])
        assert mine["numbers"][-1] == f"rule: {trust.RULE.spelled()}"
        visitor = trust.shown(found, premise_words="x", maintainer=False)
        assert visitor is not None and "numbers" not in visitor
        assert {key: value for key, value in mine.items() if key != "numbers"} == visitor

    def test_no_visitors_word_holds_a_maintainers_phrase(self) -> None:
        """Every sentence the words can say, against every phrase the guard knows."""
        stale = {kind: _item(kind, reading, sentence, state=trust.STALE,
                             count=trust.Count(1, 2, "games"), date="2026-10-01")
                 for kind, reading, sentence in (
                     (trust.FAIRY, trust.FINDS_MORE, "fairy.beats-ours"),
                     (trust.NET, trust.SILENT, "net.refused"),
                     (trust.CONVERGENCE, trust.FINDS_MORE, "convergence.moved"))}
        readings = [
            _trust(ORDERED), _trust(WITHHELD), _trust(PARTLY), _trust(PARTLY_PARTS), _trust(FAIRY_HELD), _trust(FAIRY_BEATS, NET),
            _trust(*stale.values()),
            _trust(_item(trust.CONVERGENCE, trust.AGREES, "convergence.held")),
            _trust(_item(trust.NET, trust.SILENT, "net.in-the-score")),
            _trust(_item(trust.FAIRY, trust.SILENT, "fairy.not-played")),
            _trust(_item(trust.SELF_DUEL, trust.REPORTED, "self-duel.reported",
                         count=trust.Count(5, 32, "games"))),
        ]
        said = [word for found in readings
                for people in (None, trust.People(0, 0, 0), trust.People(3, 1, 1),
                               trust.People(25, 6, 0), trust.People(9, 4, 2))
                for fairy in (None, True, False)
                for word in _visitor_words(trust.shown(
                    trust.with_people(found, people), was=[{"index": 0.5, "measured": "x"}],
                    fairy_translates=fairy, maintainer=False))]
        assert len(said) > 100
        trouble = [(text, site_words.hits(text, tuple(site_words.PHRASES))) for text in said
                   if site_words.hits(text, tuple(site_words.PHRASES))]
        assert not trouble, trouble[:5]


class TestPeopleMoveTheLevel:
    """§5 Q3, answered A on 2026-10-03: the switch is on and the rule is §2.3's."""

    def test_the_switch_is_on(self) -> None:
        assert trust.RULE.people_move_level is True
        assert "reported only for now" not in trust.RULE.spelled()

    @pytest.mark.parametrize(("people", "level"), [
        (trust.People(games=3, people=2, cleared=2), trust.QUESTIONED),
        (trust.People(games=20, people=5, cleared=0), trust.CHECKED),
        (trust.People(games=19, people=5, cleared=0), trust.EARLY),
        (trust.People(games=20, people=5, cleared=1), trust.EARLY),
    ])
    def test_with_people_rereads_the_level_and_never_the_score(
        self, people: trust.People, level: str
    ) -> None:
        found = _trust(ORDERED)
        moved = trust.with_people(found, people)
        assert moved.level == level
        assert moved.score == found.score and moved.hash == found.hash
        assert [item for item in moved.items if item.kind != trust.PEOPLE] == [
            item for item in found.items if item.kind != trust.PEOPLE]

    def test_a_score_out_of_date_stays_without_a_level_whatever_people_did(self) -> None:
        stale = _trust(ORDERED, score=trust.STALE)
        assert trust.with_people(stale, trust.People(30, 9, 3)).level is None

    def test_the_rule_can_still_be_turned_off_by_a_dated_edit(self) -> None:
        off = trust.Rule(people_move_level=False)
        assert trust.with_people(_trust(ORDERED), trust.People(3, 2, 2), off).level == trust.EARLY


# ------------------------------------------------------------------ the payloads


def _library(tmp_path: Path, fairy: float | None = None) -> tuple[Any, Path]:
    """One scored game, and a current strength check where ``fairy`` is given."""
    from test_premise import _synthetic
    from test_web import TestTheIndexBand

    from chess_variant_sandbox.web.server import App

    rulesets, runs = tmp_path / "rulesets", tmp_path / "runs"
    rulesets.mkdir()
    runs.mkdir()
    (rulesets / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
    TestTheIndexBand._score(SPEC, DIGEST, runs, 0.61, 0.58, 0.64)
    if fairy is not None:
        check = _synthetic(premise.FAILS if fairy < 0.30 else premise.PASSES)
        check["fairy"] = {"played": True, "score": fairy, "error": 0.03, "games": 32,
                          "floor": 0.30}
        premise.write(check, runs)
    return App(rulesets, runs, archive_dir=tmp_path / "archive"), runs


class TestThePayloads:
    def test_a_card_carries_the_chip_and_the_game_the_line(self, tmp_path: Path) -> None:
        app, _runs = _library(tmp_path, fairy=0.19)
        [card] = app.get("/api/library")["games"]
        assert card["index"] == 0.61, "the line hides no number"
        assert card["trust"]["level"] == trust.QUESTIONED
        assert "premise" not in card
        line = app.get(f"/api/game/{DIGEST}")["trust"]
        assert (line["level"], line["said"]) == (card["trust"]["level"], card["trust"]["said"])
        assert any("Fairy-Stockfish" in more and "6 of 32" in more for more in line["more"])
        assert any("not yet strong enough" in number for number in line["numbers"])

    def test_an_agreeing_check_reads_checked(self, tmp_path: Path) -> None:
        app, _runs = _library(tmp_path, fairy=0.40)
        [card] = app.get("/api/library")["games"]
        assert card["trust"]["level"] == trust.CHECKED

    def test_nothing_but_the_score_is_an_early_estimate(self, tmp_path: Path) -> None:
        app, _runs = _library(tmp_path)
        [card] = app.get("/api/library")["games"]
        assert card["trust"] == {"level": trust.EARLY, "said": trust.EARLY_SAID}

    def test_an_unmeasured_game_carries_none(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.web.server import App

        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        (tmp_path / "runs").mkdir()
        (rulesets / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
        app = App(rulesets, tmp_path / "runs", archive_dir=tmp_path / "archive")
        [card] = app.get("/api/library")["games"]
        assert card["trust"] is None and app.get(f"/api/game/{DIGEST}")["trust"] is None

    def test_an_earlier_score_rides_under_more(self, tmp_path: Path) -> None:
        app, runs = _library(tmp_path)
        entry = history.entry_for(1, {"index": 0.5, "basis": {"net": ""}}, {"net": "x"},
                                  "2026-09-28", "replay", ("its nets were none",))
        history.append(DIGEST, runs, entry)
        line = app.get(f"/api/game/{DIGEST}")["trust"]
        assert ("Was 0.500 when measured by the earlier computer players (September 2026)."
                in line["more"]), "an entry with no runs recorded says none"
        assert line["said"] == trust.EARLY_SAID, "never the headline"

    def test_a_new_check_reaches_a_listing_held_in_the_cache(self, tmp_path: Path) -> None:
        app, runs = _library(tmp_path)
        assert app.get("/api/library")["games"][0]["trust"]["level"] == trust.EARLY
        convergence = corpus.measurement_dir(DIGEST, runs) / trust.CONVERGENCE_FILE
        basis = corpus.read_measurement(DIGEST, runs, spec=SPEC).basis
        assert basis is not None
        convergence.write_text(json.dumps({
            "move": 0.0, "sd": 0.01, "to": trust.measurement_digest(basis.as_dict())}),
            encoding="utf-8")
        assert app.get("/api/library")["games"][0]["trust"]["level"] == trust.CHECKED


# ------------------------------------------------------------------ the page


LINE = {"level": "questioned", "said": "A test finding. Read the score as a first guess.",
        "more": ["A piece of evidence (October 2026).", "Was 0.608 when measured earlier."],
        "numbers": ["fairy-stockfish: finds more, current; score 0.19"]}


class TestThePage:
    def test_the_premise_mark_is_gone(self) -> None:
        from page_text import PAGE

        assert "premiseHtml" not in PAGE and 'class="premise"' not in PAGE

    @pytest.mark.parametrize("accounts", [False, True], ids=["local", "site"])
    def test_the_card_draws_the_chip_beside_the_score(self, accounts: bool) -> None:
        from test_web import _card, _game_payload, _page_on

        marked = _card("bbbbbbbbbbbb", "a marked game", 0.93, None)
        marked["trust"] = {"level": "questioned", "said": LINE["said"]}
        plain = _card("aaaaaaaaaaaa", "an unmeasured game", 0.8, None)
        plain["trust"] = None
        listing = {
            "games": [plain, marked],
            "opponents": [], "coverage": {"axes": []}, "metrics": None,
            **({"site": {"signedIn": True, "admin": False, "sorts": [], "filters": []}}
               if accounts else {}),
        }
        drawn = _page_on(accounts, "#/", _game_payload("cccccccccccc", site=accounts),
                         answers={"/api/library": listing}, landed='class="panel card"')
        html = drawn["every"]
        assert html.count('class="tag trustLevel"') == 1
        kept = html[html.index('data-hash="bbbbbbbbbbbb"'):]
        assert '<span class="tag trustLevel">questioned</span>' in kept
        assert "0.93" in kept.split('class="tag trustLevel"')[0], "the chip follows the score"
        assert "A test finding." in kept.split('class="tag trustLevel"')[1].split("</button>")[0]

    @pytest.mark.parametrize("accounts", [False, True], ids=["local", "site"])
    def test_the_game_page_draws_the_line_under_the_number(
        self, accounts: bool, tmp_path: Path
    ) -> None:
        """Under the number and its band, before the parts: read with what it is about."""
        from test_web import _game_payload, _page_on

        app, _runs = _library(tmp_path)
        game = {**_game_payload(DIGEST, site=accounts),
                **{key: app.get(f"/api/game/{DIGEST}")[key]
                   for key in ("score", "scoreGuide", "staleness")}}
        game["trust"] = LINE
        html = _page_on(accounts, f"#/game/{DIGEST}", game)["html"]
        line = html.index("<strong>How far to trust it: questioned.</strong>")
        assert html.index("How it scored") < html.index("0.61") < line
        assert line < html.index("What it is good at")
        assert "<summary>more</summary>" in html
        for said in (*LINE["more"], *LINE["numbers"]):
            assert said in html
        game["trust"] = {key: value for key, value in LINE.items() if key != "numbers"}
        html = _page_on(accounts, f"#/game/{DIGEST}", game)["html"]
        assert "the numbers" not in html and LINE["numbers"][0] not in html
        game["trust"] = None
        assert "How far to trust it" not in _page_on(accounts, f"#/game/{DIGEST}", game)["html"]


# ------------------------------------------------------------------ the site


@pytest.fixture
def scored_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A running site over one scored game with a current, agreeing strength check."""
    from test_premise import _synthetic
    from test_site_admin import SITE_URL, _running_site
    from test_web import TestTheIndexBand

    from chess_variant_sandbox.site import mail

    monkeypatch.setenv(mail.URL_VAR, SITE_URL)
    monkeypatch.delenv(mail.KEY_VAR, raising=False)
    for name in ("rulesets", "runs", "archive"):
        (tmp_path / name).mkdir()
    (tmp_path / "rulesets" / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
    TestTheIndexBand._score(SPEC, DIGEST, tmp_path / "runs", 0.61, 0.58, 0.64)
    check = _synthetic(premise.PASSES)
    check["fairy"] = {"played": True, "score": 0.40, "error": 0.03, "games": 32, "floor": 0.30}
    premise.write(check, tmp_path / "runs")
    with _running_site(tmp_path) as running:
        running.game(DIGEST, SPEC)
        yield running


class TestTheSite:
    def test_a_visitor_is_sent_the_words_and_no_numbers(self, scored_site: Any) -> None:
        line = scored_site.browser().get(f"/api/game/{DIGEST}").payload["trust"]
        assert line["level"] == trust.CHECKED and "numbers" not in line
        assert trust.NOBODY_YET in line["more"]
        for text in _visitor_words(line):
            assert not site_words.hits(text), text
        [card] = [game for game in scored_site.browser().get("/api/library").payload["games"]
                  if game["hash"] == DIGEST]
        assert card["trust"]["level"] == trust.CHECKED and "numbers" not in card["trust"]

    def test_an_admin_is_sent_the_numbers(self, scored_site: Any) -> None:
        from chess_variant_sandbox.site import admin

        browser = scored_site.signed_in("ada", admin.ADMIN)
        line = browser.get(f"/api/game/{DIGEST}").payload["trust"]
        assert any(number.startswith("rule: ") for number in line["numbers"])

    def test_people_clearing_the_top_question_the_game(
        self, scored_site: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two distinct people clearing the top move the level, on the card and the page;
        the score does not move."""
        from chess_variant_sandbox.site import people

        cleared = trust.People(games=4, people=2, cleared=2, latest="2026-10-02")
        monkeypatch.setattr(people, "counts", lambda connection, game: cleared)
        monkeypatch.setattr(people, "counts_of",
                            lambda connection, games: {game: cleared for game in games})
        browser = scored_site.browser()
        game = browser.get(f"/api/game/{DIGEST}").payload
        assert game["trust"]["level"] == trust.QUESTIONED
        assert game["trust"]["said"].startswith(trust.PEOPLE_CLEARED_BARE)
        assert "2 people have beaten" in " ".join(game["trust"]["more"])
        [card] = [one for one in browser.get("/api/library").payload["games"]
                  if one["hash"] == DIGEST]
        assert card["trust"]["level"] == trust.QUESTIONED and card["index"] == 0.61
