"""The user page's one motion (UP5, `docs/archive/plan-user-page.md` §5).

A waiting game's last move, the engine's `lastMove`, plays once the first time its board comes
into view, with the board motion every page shares (`motion.js`); nothing else moves, and under
`prefers-reduced-motion` nothing moves at all. The checks run `still.js` and `profile.js`'s trigger in node against a
stubbed observer and drawing, over the real `motion.js`, so what they hold is when the motion is asked for and
what it is asked to do; how it looks is `motion.js`'s, and the browser's acceptance is in the
box's note.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.web.page import SCRIPT_DIR as PAGE_DIR

PROFILE = (site_pages.SCRIPT_DIR / "profile.js").read_text(encoding="utf-8")
STILL = (site_pages.SCRIPT_DIR / "still.js").read_text(encoding="utf-8")
MOTION = (PAGE_DIR / "motion.js").read_text(encoding="utf-8")
# The user page's part: from the moves already played to the row's key, which the stubs below
# run as the page does, after the still board and the motion.
TRIGGER = PROFILE[PROFILE.index("const SEEN = new Set()"):PROFILE.index("function moveKey(")]

# A stand-in DOM: the observer is held so a check can say when the board is seen; the drawing
# is a 3 by 3 board whose squares sit at (square, 10 * square); each piece says how it was
# asked to move (`motion.js`'s own `travel` and `stoneDrop`, run for real).
_STUB = """
const asked = [];
let observers = [];
global.IntersectionObserver = class {
  constructor(back, how) { this.back = back; this.how = how; this.on = false; observers.push(this); }
  observe() { this.on = true; }
  disconnect() { this.on = false; }
};
let reduced = %(reduced)s;
global.window = { matchMedia: () => ({ matches: reduced }) };
function flatView() { return {}; }
function boardCut() { return null; }
function squareXY(square) { return [square, 10 * square]; }
function boardFor(squares) {
  const svg = { querySelector: q => {
    const m = /data-piece="(\\d+)"/.exec(q);
    if (!m || !squares.includes(+m[1])) return null;
    return { style: {}, animate: (frames, how) => {
      asked.push([+m[1], frames[0].transform, frames[0].opacity === 0]); return {}; } }; } };
  return { querySelector: q => (q === 'svg' ? svg : null) };
}
function see(visible) {
  for (const o of observers.filter(o => o.on)) o.back([{ isIntersecting: visible }]);
}
const still = { width: 3, height: 3 };
"""


def _run(body: str, reduced: bool = False) -> Any:
    from node_session import run

    source = _STUB % {"reduced": "true" if reduced else "false"} + MOTION + STILL + TRIGGER + body
    source += "\nconsole.log(JSON.stringify({ asked, open: observers.filter(o => o.on).length }));"
    done = run(source)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestTheMovePlaysOnceWhenSeen:
    def test_nothing_moves_until_the_board_is_seen_then_the_move_slides_once(self) -> None:
        said = _run("""
          const box = boardFor([5]);
          slideOnSight(box, still, { from: 2, to: 5 }, 'g1');
          const before = asked.length;
          see(false);
          const unseen = asked.length;
          see(true); see(true);
          slideOnSight(box, still, { from: 2, to: 5 }, 'g1');  // a redraw of the same row
          see(true);
          asked.push(['counts', before, unseen]);
        """)
        assert said["asked"] == [[5, "translate(-3px, -30px)", False], ["counts", 0, 0]]
        assert said["open"] == 0

    def test_a_piece_from_a_hand_drops_in(self) -> None:
        said = _run("slideOnSight(boardFor([4]), still, { from: null, to: 4 }, 'g2'); see(true);")
        assert said["asked"] == [[4, "translate(0px, -18.333333333333336px) scale(1.5)", True]]

    def test_two_games_each_play_their_own_move(self) -> None:
        said = _run("""
          slideOnSight(boardFor([5]), still, { from: 2, to: 5 }, 'a');
          slideOnSight(boardFor([1]), still, { from: 0, to: 1 }, 'b');
          see(true);
        """)
        assert [one[0] for one in said["asked"]] == [5, 1]


class TestUnderReducedMotionNothingMoves:
    def test_no_observer_is_set_and_nothing_is_asked(self) -> None:
        said = _run(
            "slideOnSight(boardFor([5]), still, { from: 2, to: 5 }, 'g'); see(true);",
            reduced=True,
        )
        assert said == {"asked": [], "open": 0}

    def test_reduced_motion_asked_for_before_the_board_is_seen_is_kept(self) -> None:
        said = _run("""
          slideOnSight(boardFor([5]), still, { from: 2, to: 5 }, 'g');
          reduced = true;
          see(true);
        """)
        assert said["asked"] == []


class TestOnlyAPlainMoveOrADropMoves:
    def test_a_portal_a_fold_a_reveal_a_pass_and_a_challenge_draw_still(self) -> None:
        said = _run("""
          global.document = { createElement: () => ({}) };
          const lasts = [null, { from: 1, to: null }, { from: 1, to: 2, portal: 4, portalOut: 6 },
            { from: 1, to: 2, foldIn: 3, foldOut: 7 }, { from: 1, to: 2, reveal: { moves: [] } },
            { from: null, to: null, pass: true }];
          for (const [i, last] of lasts.entries()) waitingBoard(null, 'h', last, 'k' + i);
          see(true);
        """)
        assert said == {"asked": [], "open": 0}


class TestTheMotionIsTheSharedFile:
    def test_the_profile_loads_motion_after_the_still_board_and_the_member_page_does_not(
        self,
    ) -> None:
        page = site_pages.profile("ada")
        at = page.index("/static/page/motion")
        assert page.index("/static/page/stillboard") < at < page.index("/static/site/still")
        assert site_pages.MOTION_SCRIPTS == ("motion",)
        assert "motion=True" not in inspect.getsource(site_pages.member)

    def test_the_page_reaches_the_motion_only_through_still_move_which_writes_no_markup(
        self,
    ) -> None:
        declared = set(re.findall(r"^function ([A-Za-z]+)\(", MOTION, re.M))
        called = set(re.findall(r"\b([A-Za-z]+)\(", PROFILE + STILL)) & declared
        assert called == {"stillMove"}
        body = MOTION[MOTION.index("function stillMove("):]
        body = body[: body.index("\n}\n")]
        assert "innerHTML" not in body and "ghostAt" not in body and "capture(" not in body

    def test_nothing_on_the_site_animates_by_itself(self) -> None:
        for text in (STILL, PROFILE):
            assert ".animate(" not in text and "@keyframes" not in text
        # The waiting rows draw through the one helper; the cards and the showcase do not.
        assert PROFILE.count("waitingBoard(entry.") == 3
