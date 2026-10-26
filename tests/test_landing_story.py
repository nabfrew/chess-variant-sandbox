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
import dataclasses
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


@pytest.fixture
def first(drawn: str, rendered: dict[str, Any]) -> str:
    """The first story's part: its hero and its journey, as a page keeps it."""
    return _part(drawn, rendered["stories"][0]["key"])


# ------------------------------------------------------------------ the hero


def test_the_hero_games_are_listed_library_games() -> None:
    names = {str(load_spec(path).get("name")) for path, _ in
             stories.library_paths(Roots(ROOT / "runs")).values()}
    families = {block.family for block in landing.BLOCKS}
    for idea in landing.HERO_IDEAS:
        assert idea.game in names, idea.game
        assert idea.family in families, idea.family


def test_the_question_leads_with_the_storys_idea_and_leaves_out_its_family(
        first: str, rendered: dict[str, Any]) -> None:
    drawn = first
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


def test_six_steps_from_the_record(first: str, rendered: dict[str, Any]) -> None:
    drawn = first
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
        first: str, rendered: dict[str, Any]) -> None:
    drawn = first
    story = rendered["stories"][0]
    assert drawn.count('class="verdictBlock pending"') == 2
    for side in (story["failed"], story["fixed"]):
        assert landing._e(side["line"]) in drawn
        assert "4 runs" in side["line"] and "scoring rules version" in side["line"]


def test_the_qualities_are_paired_with_their_numbers(first: str, rendered: dict[str, Any]) -> None:
    drawn = first
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


# ------------------------------------------------------------------ every story (LP5)


def _part(drawn: str, key: str) -> str:
    """One story's part of the page: its hero and its journey (`storyPart`)."""
    start = drawn.index(f'<div class="storyPart" data-story="{key}">')
    rest = [drawn.index(mark, start + 1) for mark in ('<div class="storyPart"', 'id="landing-palette"')
            if mark in drawn[start + 1:]]
    return drawn[start:min(rest)]


def _text(markup: str) -> str:
    return re.sub(r"<[^>]+>", " ", markup)


def test_every_holding_story_is_one_part_in_the_pools_order(
        drawn: str, rendered: dict[str, Any]) -> None:
    keys = [s["key"] for s in rendered["stories"]]
    assert keys == ["stones", "door", "blast", "sweep"]
    assert re.findall(r'<div class="storyPart" data-story="(\w+)">', drawn) == keys
    assert f'data-stories="{" ".join(keys)}"' in drawn
    for key in keys:
        part = _part(drawn, key)
        assert part.count('id="journey"') == part.count('id="heroShow"') == 1
        assert '<section class="journey' in part and f'data-story="{key}"' in part


@pytest.mark.parametrize("key", ["stones", "door", "blast", "sweep"])
def test_each_storys_six_steps_are_its_record(drawn: str, rendered: dict[str, Any],
                                               key: str) -> None:
    story = next(s for s in rendered["stories"] if s["key"] == key)
    part = _part(drawn, key)
    hero = part[:part.index('id="journey"')]
    clauses = re.findall(r'<span class="clause ([\w-]+)(?: on)?">([^<]*)</span>', hero)
    assert clauses[0] == (story["hue"], landing._e(story["hero"]["clause"]))
    journey = part[part.index('id="journey"'):]
    assert re.findall(r'data-beat="(\d)"', journey) == ["1", "2", "3", "4", "5", "6"]
    for said in (story["idea"], story["wrong"]["say"], story["change"]["say"],
                 story["right"]["say"], story["wrong"]["end"], story["right"]["end"],
                 story["failed"]["line"], story["fixed"]["line"]):
        assert landing._e(said) in journey, said
    card = story["card"]
    assert f"<del>{landing._e(card['was'])}</del> <ins>{landing._e(card['now'])}</ins>" in journey
    assert journey.count('class="verdictBlock pending"') == 2
    # The foot opens the page on each other story.
    others = [s["key"] for s in rendered["stories"] if s["key"] != key]
    assert re.findall(r'href="/\?story=(\w+)"', journey) == others


def test_a_storys_hero_board_at_a_turn_is_its_own_game(drawn: str) -> None:
    assert stories.BY_KEY["blast"].hero_at == 13
    assert 'data-at="13" data-story="blast"' in _part(drawn, "blast")
    assert "data-at=" not in _part(drawn, "sweep")[:_part(drawn, "sweep").index('id="journey"')]


@pytest.fixture
def with_ground(monkeypatch: pytest.MonkeyPatch, rendered: dict[str, Any],
                tmp_path: Path) -> tuple[str, dict[str, Any]]:
    """The page with ground, the constructed example out of the pool since LPD, put back in:
    the constructed kind's drawing is held on it."""
    monkeypatch.setattr(stories, "STORIES", (*stories.STORIES, stories.GROUND))
    again = stories.render(Roots(tmp_path))
    assert again["pool"][-1] == "ground"
    return landing.html_page(stories=again["stories"]), again


def test_ground_says_example_and_carries_no_score(
        with_ground: tuple[str, dict[str, Any]]) -> None:
    """§2.1's honesty rules for a constructed story: the hero's caption, step 1's note, the
    other stories' chip and both verdict lines say *example*; no index, no runs, no ±, no
    quality bars, no ladder read from a game page it has not got."""
    drawn, rendered = with_ground
    assert 'data-at="44" data-story="ground"' in _part(drawn, "ground")
    part = _part(drawn, "ground")
    hero = part[:part.index('id="journey"')]
    assert landing.EXAMPLE_CAPTION in hero and 'href="/game" data-name>ground for good' not in hero
    ground = next(s for s in rendered["stories"] if s["key"] == "ground")
    assert "example" in ground["note"]
    assert part.count(f": {stories.EXAMPLE_LINE}</p>") == 2
    text = _text(part)
    assert not re.search(r"\bruns?\b", text.replace("run out", "")) and "±" not in text
    assert "pairBars" not in part and "data-ladder" not in part
    for key in ("stones", "door", "blast", "sweep"):
        chips = re.findall(r'href="/\?story=ground"[^>]*>(.*?)</a>', _part(drawn, key))
        assert len(chips) == 1 and "an example" in chips[0]


def test_the_door_carries_the_sites_lean_sentence(rendered: dict[str, Any]) -> None:
    door = copy.deepcopy(next(s for s in rendered["stories"] if s["key"] == "door"))
    door["fixed"]["lean"] = "the second side won more, though not by enough to rule it out"
    assert ("The site adds: the second side won more, though not by enough to rule it out."
            in landing.story_html(door))


@pytest.mark.parametrize("key", ["door", "blast"])
def test_a_library_storys_quick_answers_are_the_designers(rendered: dict[str, Any],
                                                          key: str) -> None:
    """Wormhole's and queens wake's quick answers on step 2 are the designer's two rungs on
    the failed game's own file: `designing.check` and the eight-game screen."""
    story = next(s for s in rendered["stories"] if s["key"] == key)
    path, _ = stories.library_paths(Roots(ROOT / "runs"))[stories.BY_KEY[key].wrong.game.hash]
    raw = load_spec(path)
    checked = designing.check(raw)
    screened = candidacy.screen_draft(raw)
    quick = story["failed"]["quick"]
    assert quick["playable"] == checked["playable"] is True
    assert quick["opening"] == checked["openingWords"]
    assert quick["said"][0] == f"It plays and reaches an ending, with {checked['openingWords']}."
    stats = screened["stats"]
    assert screened["passed"] is True
    assert quick["said"][1] == ("The eight-game check finds a winner in "
                                f"{int(stats['decisive'])} of its {int(stats['games'])} games.")
    assert quick["when"] == stories.LIBRARY_QUICK_WHEN


def test_the_landing_draws_claims_from_the_frames() -> None:
    """Ground's paint and its count are the server's (`claims`, `counts`), and its motion is
    `motion.js`'s one copy."""
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    motion = (STATIC / "motion.js").read_text(encoding="utf-8")
    assert "function claimPaint(" in motion and PAGE.count("function claimPaint(") == 1
    assert "claimPaint(" in landing_js and "item.claims" in landing_js
    assert "game.start.counts" in landing_js and ".filter(o =>" not in landing_js


def test_the_sweep_is_drawn_from_its_frames(rendered: dict[str, Any]) -> None:
    """The squares a sweep cleared are the server's (`sweep` on a frame); the page strikes
    each as the sweeper passes and tints it after, and counts the sweep as the rule acting."""
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    assert "item.sweep" in landing_js and "frame.item.sweep" in landing_js
    sweep = next(s for s in rendered["stories"] if s["key"] == "sweep")
    first = sweep["wrong"]["game"]["frames"][0]
    assert first["n"] == 1 and first["sweep"] and first["key"]


def test_the_failings_say_each_test_once_where_every_run_failed_it() -> None:
    runs = [f"run {k} of 4: game length, worse half came to 1.000 turns" for k in range(1, 5)]
    runs += [f"run {k} of 4: quick kills, worse half came to 1.000" for k in range(1, 5)]
    runs.append("run 3 of 4: recovery from a mistake could not be measured on this game")
    assert landing._failings(runs) == (
        "run 1 of 4: game length, worse half came to 1.000 turns; the same in all 4 runs; "
        "run 1 of 4: quick kills, worse half came to 1.000; the same in all 4 runs; "
        "run 3 of 4: recovery from a mistake could not be measured on this game")
    assert landing._failings(["a, b", "c, d"]) == "a, b; c, d"


def test_each_storys_card_is_a_door_into_the_designer(drawn: str,
                                                      rendered: dict[str, Any]) -> None:
    # LP7: the card of the idea, as a rule, opens the designer at the story's rule.
    assert rendered["stories"]
    for story in rendered["stories"]:
        part = _part(drawn, story["key"])
        door = landing.door_link(story["card"]["door"])
        assert (f'<p class="storyDoor small"><a href="{door}">{landing.DOOR_SAID}</a></p>'
                in part), story["key"]
        assert part.count('class="storyDoor') == 1


# ------------------------------------------------------------------ the play doors (LP9)


def test_each_storys_play_doors_start_its_fixed_game_in_its_seat(
        drawn: str, rendered: dict[str, Any]) -> None:
    """LP9 (`landing-cold-read.md` §3): the hero and step 5 each carry one door that starts the
    fixed game, the visitor in the story's seat, a link to the game's page with that seat
    chosen; the door story's is Black."""
    seats = {story["key"]: story["seat"] for story in rendered["stories"]}
    assert seats["door"] == {"index": 1, "side": "black"}
    assert seats["sweep"]["index"] == 0
    for story in rendered["stories"]:
        part = _part(drawn, story["key"])
        fixed = story["fixed"]
        seat = story["seat"]
        agent = f' data-agent="{story["opponent"]["agent"]}"' if story["opponent"] else ""
        door = (f'href="/game/{fixed["hash"]}?seat={seat["side"]}" data-play="{fixed["hash"]}" '
                f'data-seat="{seat["index"]}"{agent}')
        assert part.count(door) == 2, story["key"]
        hero = part[:part.index('id="journey"')]
        assert f'class="button storyWay" {door}' in hero
        assert 'class="button todayWay" href="/game" data-today' in hero
        five = part[part.index('data-beat="5"'):part.index('data-beat="6"')]
        assert f'class="button" {door}>{landing.PLAY_THIS}</a>' in five
        assert landing._e(landing.play_this_said(story)) in five
        assert hero.count("button primary") == 1 and "button primary" not in five


def test_without_a_story_the_hero_keeps_todays_door_alone() -> None:
    hero = landing.html_page(stories=[])
    assert "data-play" not in hero
    assert '<a class="button" href="/game" data-today>' in hero


def test_the_doors_ask_the_server_for_the_session_and_its_opponent() -> None:
    """The landing seats the gentlest opponent the game's page offers and starts the game
    through the games page's own `startPractice`: it picks no move and no game (rule 3)."""
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    browse_js = (STATIC / "browse.js").read_text(encoding="utf-8")
    assert "await startPractice(hash, against, seat);" in landing_js
    assert "seats: seat === 1 ? [agent, 'human'] : ['human', agent]" in browse_js
    assert "detailFor(hash)" in landing_js and "game.firstGame && game.firstGame.gentle" in landing_js


def test_the_game_page_honours_the_seat() -> None:
    detail_js = (STATIC / "detail.js").read_text(encoding="utf-8")
    assert "const asked = askedSeat(game);" in detail_js
    assert "new URLSearchParams(location.search).get('seat')" in detail_js
    assert "start.scrollIntoView({ block: 'center' })" in detail_js


def test_step_two_holds_on_its_moment_and_the_hero_on_its_story() -> None:
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    assert "new StoryPlayer('wrong', story.wrong, art, story.failedLine, false, true)" in landing_js
    assert "new StoryPlayer('right', story.right, art, story.fixedLine, true)" in landing_js
    assert "if (this.hold) { this.holdOnMoment(); return; }" in landing_js
    assert "if (show.closest('.storyPart')) { stop(); return; }" in landing_js
    assert "window.addEventListener('scroll', settle, { passive: true });" in landing_js


# ------------------------------------------------------------------ the way back (LP11)


def test_a_story_names_its_doors_opponent_as_it_names_its_seat(
        drawn: str, rendered: dict[str, Any]) -> None:
    """LP11 (`landing-cold-read.md` §6.2 item 4): blast's gentlest fell to a three-move queen
    trap, so its doors seat *easy*; every other story keeps LP9's gentlest, asked of the game's
    page, and its step 5 says so."""
    against = {story["key"]: story["opponent"] for story in rendered["stories"]}
    assert against["blast"] == {"agent": "minimax:nodes=400", "name": "easy"}
    assert all(against[key] is None for key in ("stones", "door", "sweep"))
    for story in rendered["stories"]:
        part = _part(drawn, story["key"])
        five = part[part.index('data-beat="5"'):part.index('data-beat="6"')]
        if story["opponent"]:
            assert part.count(f'data-agent="{story["opponent"]["agent"]}"') == 2
            assert "against the computer player easy," in five
        else:
            assert "data-agent" not in part
            assert "against the gentlest," in five
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    assert ("playNow(door, door.dataset.play, Number(door.dataset.seat) || 0, door.dataset.agent);"
            in landing_js)
    assert "const against = agent || await gentlestAgent(hash);" in landing_js


def test_a_story_naming_no_rung_is_refused() -> None:
    story = next(one for one in stories.STORIES if one.key == "blast")
    with pytest.raises(stories.StoryError):
        stories.door_opponent(dataclasses.replace(story, opponent="minimax:nodes=7"))
    with pytest.raises(stories.StoryError):
        stories.door_opponent(dataclasses.replace(story, opponent="human"))


def test_the_heros_own_board_starts_the_storys_game(drawn: str) -> None:
    """LP11 (§6.2 item 2): a tap on the story's own board in the hero takes the story's door
    (the same game, started); the other endings' boards stay pictures."""
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    assert "wireHeroBoard(root);" in landing_js
    # Every story's own ending names it, at a turn (blast) or at the opening (stones).
    for key in ("stones", "blast"):
        part = _part(drawn, key)
        stage = part[part.index('id="heroStage"'):part.index('id="journey"')]
        assert stage.count(f'data-story="{key}"') == 1 and stage.count("data-story=") == 1
    assert ".heroState[data-story=\"${part.dataset.story}\"]" in landing_js
    assert "board.addEventListener('click', () => door.click());" in landing_js
    assert "board.setAttribute('role', 'button');" in landing_js
    assert ".heroBoard.heroPlays { cursor: pointer; }" in PAGE


def test_a_game_a_story_started_links_back_to_the_story() -> None:
    """LP11 (§6.2 item 1): the door keeps the story and its game in the tab; the play screen
    carries *← the story* and its ending *how this game was made*, both to the story's step 2,
    and only for a game of the story's own fixed game."""
    landing_js = (STATIC / "landing.js").read_text(encoding="utf-8")
    board_js = (STATIC / "board.js").read_text(encoding="utf-8")
    assert "if (part) keepStoryGame(part.dataset.story, door.dataset.play);" in landing_js
    assert "return kept && kept.hash === hash ? kept.key : null;" in landing_js
    assert "const STORY_BACK_BEAT = 2;" in landing_js
    assert "if (beat) beat.scrollIntoView({ block: 'start' });" in landing_js
    assert "storyGameFor(rules.hash)" in board_js
    assert ">&larr; the story</a>" in board_js
    assert "const STORY_ENDED_SAID = 'how this game was made';" in board_js
    assert "STORY_SHOWN ? `<p class=\"ended-story small\">" in board_js
    # A draft, a try from a proven position and an inline board carry none.
    assert ("const story = !inline && !proven && !rules.scratch && !snapshot.puzzle" in board_js)


def test_the_opponents_name_opens_the_picker_in_the_readers_seat() -> None:
    """LP11 (§6.2 item 3): the computer's name on a practice board links to the game page's
    picker with the reader's seat chosen (`?seat=`, which scrolls to it); chosen over *one
    harder* on *new game*, because the reader clicked the name first and the picker holds the
    whole ladder with its strengths."""
    board_js = (STATIC / "board.js").read_text(encoding="utf-8")
    assert "const picker = pickerPath(player);" in board_js
    assert "data-picker title=\"${esc(PICKER_TITLE)}\"" in board_js
    assert "return gamePath(rules.hash) + '?seat=' + encodeURIComponent(said);" in board_js
    assert ("if (!SESSION || BOARD_INLINE || BOARD_PROVEN || SESSION.correspondence "
            "|| SESSION.puzzle) {") in board_js
    assert "wireOpponentPicker(root);" in board_js


_PRACTICE = """
SESSION = { id: 's1', seats: ['human', 'minimax:nodes=400'],
            ruleset: { hash: 'abcdef012345', name: 'queen bomb', sides: ['white', 'black'] } };
"""


def test_the_picker_link_runs_for_a_practice_board_alone() -> None:
    from test_web import _in_page

    others = ("SESSION.seats = ['human', 'human']", "SESSION.seats = ['greedy', 'greedy']",
              "SESSION.ruleset.scratch = true", "SESSION.puzzle = { id: 'p' }",
              "BOARD_INLINE = true")
    probe = ("(() => { const base = JSON.stringify(SESSION); const out = {"
             " paths: [pickerPath(0), pickerPath(1)],"
             " seats: [seatHtml(0).includes('data-picker'), seatHtml(1).includes('data-picker')] };"
             " SESSION.seats = ['greedy', 'human']; out.black = pickerPath(0); out.others = [];"
             + "".join(f" SESSION = JSON.parse(base); BOARD_INLINE = false; {one};"
                       " out.others.push(pickerPath(1));" for one in others)
             + " return out; })()")
    got = _in_page(_PRACTICE, probe)
    assert got["paths"] == [None, "/game/abcdef012345?seat=white"]
    assert got["seats"] == [False, True]
    assert got["black"] == "/game/abcdef012345?seat=black"
    assert got["others"] == [None] * len(others)


def test_the_story_a_game_came_from_is_its_fixed_games_alone() -> None:
    from test_web import _in_page

    probe = ("[storyGameFor('abcdef012345'), (keepStoryGame('blast', 'abcdef012345'),"
             " storyGameFor('abcdef012345')), storyGameFor('012345abcdef'), storyPath('blast')]")
    assert _in_page("", probe) == [None, "blast", None, "/?story=blast"]


def test_back_from_the_picker_draws_the_game_it_left() -> None:
    """LP11: back from `/game/<hash>?seat=` to `/#/play/<id>` changes the path and the fragment,
    and Chromium sends no `hashchange` for it, so the `popstate` draws; a fragment-only step is
    still the `hashchange`'s, and one that follows a drawn `popstate` draws nothing."""
    from test_web import _in_page

    probe = ("(() => { let drawn = 0; render = () => { drawn += 1; RENDERED = location.href; };"
             " const at = href => { const u = new URL(href); Object.assign(location, { href: u.href,"
             " pathname: u.pathname, search: u.search, hash: u.hash }); };"
             " const out = [];"
             " RENDERED = 'http://x/game/ab?seat=white'; RENDERED_HASH = '';"
             " at('http://x/?story=stones#/play/s1'); onHistory(); out.push(drawn);"
             " onHashChange(); out.push(drawn);"
             " RENDERED = 'http://x/#/play/s1'; RENDERED_HASH = '#/play/s1';"
             " at('http://x/#/game/ab'); onHistory(); out.push(drawn);"
             " onHashChange(); out.push(drawn); return out; })()")
    assert _in_page("", probe) == [1, 1, 1, 2]
