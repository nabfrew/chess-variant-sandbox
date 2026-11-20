"""The first minute's faults, held (PF2, `docs/reference/first-visit-study.md` §6.5).

Sixteen first-minute runs at 375 px found seven faults between the front page and a first
move. Item 7 (the invitation) is held in `tests/test_site_mail.py`; the six on the page are
held here, as the page's text, the idiom of the page's tests (`tests/page_text.py`). The
walk that read them fixed is the record's §6.6.
"""

from __future__ import annotations

from pathlib import Path

from page_text import PAGE

from chess_variant_sandbox.web import landing

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "chess_variant_sandbox" / "web" / "static" / "page"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _body(source: str, head: str) -> str:
    """One top-level function's text, from its head to the next top-level line."""
    start = source.index(head)
    end = source.find("\n}\n", start)
    return source[start:end]


def test_a_press_on_a_play_door_puts_the_game_starting_at_the_top_at_once() -> None:
    """Item 1: the press clears the front page at once (so its story board cannot stand in for
    the game), puts the view at its top, and shows the status line over an empty board's
    place; a failure puts the front page back with the reason."""
    js = _read("landing.js")
    play_now = _body(js, "async function playNow(")
    assert "showStarting();" in play_now and "leaveStarting(error);" in play_now
    assert play_now.index("showStarting();") < play_now.index("await gentlestAgent(hash)")
    starting = _body(js, "function showStarting(")
    assert "view.textContent = '';" in starting
    assert "window.scrollTo(0, 0);" in starting
    assert "board.className = 'startingBoard';" in starting
    assert "pending('Starting the game…');" in starting
    assert "player.stop()" in starting
    leaving = _body(js, "function leaveStarting(")
    assert "showLanding();" in leaving and "fail(error);" in leaving
    # *Play today's game* clears the page before it waits on the listing.
    wire = _body(js, "function wirePlayDoors(")
    today = wire[wire.index("today.addEventListener"):]
    assert today.index("showStarting();") < today.index("await listing")
    assert ".startingBoard {" in PAGE and "aspect-ratio: 1" in PAGE


def test_the_heros_door_says_it_starts_a_game_now_and_names_the_game_in_its_title() -> None:
    """Item 2: *Play a game now*, the game named in the door's title and label; *Start
    building a game* stays the one primary."""
    assert landing.PLAY_STORY == "Play a game now"
    assert "{name}" not in landing.PLAY_STORY and "{name}" in landing.PLAY_STORY_SAID
    story = {"key": "k", "fixed": {"hash": "abc", "name": "three stones"},
             "seat": {"index": 0, "side": "white"}, "opponent": None}
    ways = landing._ways(story)
    said = "Start a game of three stones against the computer, now"
    assert f'title="{said}" aria-label="{said}">Play a game now</a>' in ways
    assert ways.count("button primary") == 1 and landing.START in ways
    # The hero's board, which takes the door, carries the same label.
    assert "door.title || door.textContent" in _read("landing.js")


def test_the_hand_names_its_pieces_and_says_how_to_place_one() -> None:
    """Item 3: a held piece's button names it, and a line under the hand says placing one is
    the move and what to tap, while a drop is on offer."""
    js = _read("board.js")
    hand = _body(js, "function drawHand(")
    assert "const called = face => names[face] || ids[face];" in hand
    assert "${esc(called(face))}" in hand and "${esc(ids[face])}" not in hand
    assert "handHowWords(held.length && !AT && droppable.size > 0)" in hand
    how = _body(js, "function handHowWords(")
    assert "if (!offered) return '';" in how and "DROP !== null ? HAND_PICKED : HAND_HOW" in how
    assert "Placing one is your move: tap it here, then a square on the board." in js
    assert ".handHow { flex-basis: 100%;" in PAGE


def test_a_cut_piece_line_shows_its_way_on() -> None:
    """Item 4: a cut line says *more* over its end, and holds two lines on a phone."""
    assert '.piece-says.more:not(.open)::after { content: "more";' in PAGE
    assert ("@media (max-width: 600px) { html:not(.board-full) .piece-says:not(.open) "
            "{ -webkit-line-clamp: 2; } }") in PAGE


def test_the_readers_own_move_lands_at_once() -> None:
    """Item 5: the move the reader tapped travels in `PACE.own`, the reply at the reading pace;
    the cap is set only for the synchronous draw of `send`'s answer, and always unset."""
    motion = _read("motion.js")
    assert "  own: 120," in motion
    assert "Math.min(most, TRAVEL_CAP, PACE.travelMax," in motion
    js = _read("board.js")
    assert ".then(s => ownMove(() => applySnapshot(s)))" in js
    own = _body(js, "function ownMove(")
    assert "try { return draw(); } finally { OWN_MOVE = false; }" in own
    change = _body(js, "function animateChange(")
    assert "TRAVEL_CAP = OWN_MOVE && plainStep(last, before) ? PACE.own : Infinity;" in change
    # A move where the rule acts keeps the time its motion needs (the look check's push and
    # sweep rows read it moving).
    plain = _body(js, "function plainStep(")
    for field in ("last.reveal", "last.pushed", "last.swept", "last.portal", "last.foldIn",
                  "blastFires("):
        assert field in plain, field
    assert "} finally {\n    TRAVEL_CAP = Infinity;" in change
    # The phase after the move is drawn after the cap is lifted.
    assert change.index("TRAVEL_CAP = Infinity;") < change.index("animatePhase(")


def test_the_storys_board_is_drawn_without_waiting_for_the_listing() -> None:
    """Item 6: the story's own hero board is drawn from `/api/stories` as the landing opens,
    before the listing is awaited, and not drawn again when the listing lands."""
    js = _read("landing.js")
    show = _body(js, "async function showLanding(")
    early = "view.querySelectorAll('#heroShow .heroState[data-at][data-story]').forEach(heroFrame);"
    assert early in show and show.index(early) < show.index("await listing")
    hero = _body(js, "function startHero(")
    assert "if (!state.querySelector('[data-board] svg')) heroFrame(state);" in hero
