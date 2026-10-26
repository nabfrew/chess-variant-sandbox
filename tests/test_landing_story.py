"""The landing's hero and its story, on direction B's template (`docs/plan-landing.md` box LP4).

What is held here without a browser: the hero's question leads with the first holding story's
idea and turns through the other families' games, each a listed library game; the first story's
six steps are drawn from `stories.render`'s record (its sentences, its numbers with their basis,
the card's struck and new words, each verdict hidden until its game has ended); with no story
holding, the journey goes and the hero stays; the games' frames are `/api/stories`'s, never the
page's; the play board and the landing share one copy of the motion (`motion.js`); and stones'
quoted quick answer is the designer's own on `archive/stones.json`. The motion itself, the
first screen and reduced motion are the browser's and the look check's.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
from page_text import PAGE
from test_stories import LIBRARY_STORIES, Roots, made_score

from chess_variant_sandbox.spec.canonical import load_spec
from chess_variant_sandbox.web import candidacy, designing, landing, stories
from chess_variant_sandbox.web.page import front

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "chess_variant_sandbox" / "web" / "static" / "page"


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """The four stories rendered with each library pair reading as told."""
    made = {}
    for story in LIBRARY_STORIES:
        made[story.wrong.game.hash] = made_score(False)
        made[story.right.game.hash] = made_score(True)
    monkeypatch.setattr(stories, "read_score",
                        lambda app, digest, *_: copy.deepcopy(made.get(digest)))
    return stories.render(Roots(tmp_path))


@pytest.fixture
def drawn(rendered: dict[str, Any]) -> str:
    return landing.html_page(stories=rendered["stories"])


# ------------------------------------------------------------------ the hero


def test_the_hero_games_are_listed_library_games() -> None:
    names = {str(load_spec(path).get("name")) for path, _ in
             stories.library_paths(Roots(ROOT / "runs")).values()}
    families = {block.family for block in landing.BLOCKS}
    for idea in landing.HERO_IDEAS:
        assert idea.game in names, idea.game
        assert idea.family in families, idea.family


def test_the_question_leads_with_the_storys_idea_and_leaves_out_its_family(
        drawn: str, rendered: dict[str, Any]) -> None:
    story = rendered["stories"][0]
    clauses = re.findall(r'<span class="clause ([\w-]+)(?: on)?">([^<]*)</span>', drawn)
    assert clauses[0] == (story["hue"], landing._e(story["hero"]["clause"]))
    others = [idea for idea in landing.HERO_IDEAS if idea.family != story["family"]]
    assert [c[1] for c in clauses[1:]] == [landing._e(idea.clause) for idea in others]
    assert drawn.count('class="heroState') == len(clauses) == drawn.count('class="heroDot')


def test_both_ways_in_and_the_primary_is_building(drawn: str) -> None:
    hero = drawn[:drawn.index('id="journey"')]
    assert '<a class="button primary" href="#/design">Start building a game</a>' in hero
    assert "Play today&#x27;s game</a>" in hero
    assert hero.count("button primary") == 1


def test_the_drawn_intro_is_the_heros_start(rendered: dict[str, Any]) -> None:
    intro = landing.drawn_intro(rendered["stories"][0])
    assert landing._e(rendered["stories"][0]["hero"]["clause"]) in intro
    assert landing._e(landing.LEAD) in intro and "Start building a game" in intro


def test_the_front_pages_description_is_the_question(rendered: dict[str, Any]) -> None:
    clause = stories.STORIES[0].clause
    assert front.HOME_DESCRIPTION.startswith(f"Wouldn't it be cool if {clause}")


# ------------------------------------------------------------------ the journey


def test_six_steps_from_the_record(drawn: str, rendered: dict[str, Any]) -> None:
    story = rendered["stories"][0]
    journey = drawn[drawn.index('id="journey"'):]
    assert re.findall(r'data-beat="(\d)"', journey) == ["1", "2", "3", "4", "5", "6"]
    for said in (story["idea"], story["wrong"]["say"], story["change"]["say"],
                 story["right"]["say"], story["wrong"]["end"], story["right"]["end"]):
        assert landing._e(said) in journey
    card = story["card"]
    assert f"<del>{landing._e(card['was'])}</del> <ins>{landing._e(card['now'])}</ins>" in journey
    assert '<h3 class="say bad">' in journey and '<h3 class="say good">' in journey


def test_each_verdict_waits_for_its_game_and_carries_its_basis(
        drawn: str, rendered: dict[str, Any]) -> None:
    story = rendered["stories"][0]
    assert drawn.count('class="verdictBlock pending"') == 2
    for side in (story["failed"], story["fixed"]):
        assert landing._e(side["line"]) in drawn
        assert "4 runs" in side["line"] and "scoring rules version" in side["line"]


def test_the_qualities_are_paired_with_their_numbers(drawn: str, rendered: dict[str, Any]) -> None:
    story = rendered["stories"][0]
    bars = re.findall(r'<b data-v="([\d.]+)"></b><i data-v="([\d.]+)"></i>', drawn)
    assert len(bars) == 6
    for (a, b), name in zip(bars, landing._BAR_ORDER, strict=True):
        assert float(a) == pytest.approx(story["failed"]["qualities"][name], abs=1e-4)
        assert float(b) == pytest.approx(story["fixed"]["qualities"][name], abs=1e-4)


def test_no_story_holding_leaves_the_hero_and_drops_the_journey() -> None:
    drawn = landing.html_page(stories=[])
    assert 'id="journey"' not in drawn
    assert drawn.count('class="clause ') == len(landing.HERO_IDEAS)
    assert 'id="landing-palette"' in drawn


def test_the_page_carries_the_landing_the_server_drew(rendered: dict[str, Any]) -> None:
    from chess_variant_sandbox.web import page

    try:
        landing.use(rendered["stories"])
        assert 'data-story="stones"' in page.page()
    finally:
        landing.use([])
    assert 'id="journey"' not in page.page()


def test_no_inline_style_and_no_number_typed_in_the_markup(drawn: str) -> None:
    assert " style=" not in drawn and "<style" not in drawn and "<script" not in drawn


# ------------------------------------------------------------------ the quick answers


def test_stones_quoted_quick_answer_is_the_designers(rendered: dict[str, Any]) -> None:
    stones = next(s for s in rendered["stories"] if s["key"] == "stones")
    raw = load_spec(ROOT / "archive" / "stones.json")
    checked = designing.check(raw)
    screened = candidacy.screen_draft(raw)
    quick = stones["failed"]["quick"]
    assert quick["playable"] == checked["playable"] is True
    assert quick["opening"] == checked["openingWords"]
    assert screened["passed"] is False
    assert quick["said"][1] == "✗ a searching player can win it: " + " ".join(screened["reasons"])
    assert quick["when"] == stories.LIBRARY_QUICK_WHEN


# ------------------------------------------------------------------ the games and the motion


def test_the_frames_are_the_servers(rendered: dict[str, Any], tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_variant_sandbox.web.server import App

    app = App(ROOT / "rulesets", tmp_path, archive_dir=ROOT / "archive")
    monkeypatch.setattr(app, "stories", lambda: rendered)
    payload = app.get("/api/stories")
    assert payload["pool"] == rendered["pool"]
    stones = payload["games"]["stones"]
    drops = [f for f in stones["wrong"]["frames"] if "drop" in f]
    assert drops and all(f["drop"] == f["to"] and "from" not in f for f in drops)
    assert not any(f.get("promotion") for f in stones["right"]["frames"] if "drop" in f)
    assert payload["pieceArt"]


def test_one_copy_of_the_motion() -> None:
    motion = (STATIC / "motion.js").read_text(encoding="utf-8")
    board = (STATIC / "board.js").read_text(encoding="utf-8")
    for name in ("travel", "capture", "ghostAt", "strike", "blastFire", "burn", "resist",
                 "stoneDrop", "portalJourney", "portalSwirl", "portalPulse", "portalFlight"):
        assert f"function {name}(" in motion, name
        assert PAGE.count(f"function {name}(") == 1, name
    assert "function explode(" not in PAGE and "explode(svg" not in PAGE
    assert "const PACE = {" in motion and "const PACE" not in board
    assert "portalFlight(" in board and "blastFire(" in board
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    for used in ("portalFlight(", "blastFire(", "burn(", "stoneDrop(", "travel(", "capture("):
        assert used in landing_js, used


def test_the_landing_plays_no_move_of_its_own() -> None:
    """Rule 3: the page lays out positions and draws the way between them; it asks no legal
    move and decides no ending."""
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    for asked in ("legal", "/api/play", "/api/move", "wrapStep(", "explosion"):
        assert asked not in landing_js, asked
