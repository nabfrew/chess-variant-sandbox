"""A piece's trip through a portal, animated (Q836, `docs/plan-portals.md` *Travel through a
portal, animated*).

The board's own `animateChange` is run in node over a stand-in for the SVG that records every
`animate` call: which element, its keyframes and its timing. What can be held without a
browser is the order of the journey's movements (the slide to the end it goes in by, the dive
into it overlapping the slide's end, the instant inside, the emergence from the other end and
the settle), that they spring rather than step, that the piece at rest is what the board drew,
that a capture on arrival is struck as the piece arrives, the spin's sense both ways round, a
review step forward playing the trip and a step back not, and the reduced-motion branch. The
motion itself is the look check's (`docs/reference/look-check.md`, *After Q836*).

The position is `test_portal_board`'s: cylinder chess with one two-way pair on d4 and d5, a
white rook on a4 that reaches d4 going right (out at e5) and, round the seam, going left (out
at c5).
"""

from __future__ import annotations

import json
from typing import Any

from test_portal_board import _cylinder_session, _square
from test_web import _in_page

from chess_variant_sandbox.engine.compile import WILD

# The stand-in: an element records its `animate` calls under its label (`data-journey` when
# the board names one, `piece <square>` for what the board drew there, `g` otherwise).
_STAGE = """
const JLOG = [], JORDER = [];
const jfake = (label) => {
  const el = { label: label, attrs: {}, style: {}, parentNode: null, firstElementChild: null,
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'data-journey') this.label = v; },
    getAttribute(k) { return this.attrs[k]; },
    set innerHTML(v) { this._html = v; this.firstElementChild = jfake(this.label + '>');
                       this.firstElementChild.parentNode = this; },
    get innerHTML() { return this._html; },
    appendChild(c) { c.parentNode = this; JORDER.push(c.label); return c; },
    insertBefore(c) { c.parentNode = this; return c; },
    remove() {},
    animate(frames, opts) { JLOG.push({ who: this.label, frames: frames, opts: opts || {} });
                            return {}; } };
  return el;
};
document.createElementNS = () => jfake('g');
const JDRAWN = {};
const JSVG = jfake('svg');
JSVG.querySelector = sel => {
  const m = /data-piece="(\\d+)"/.exec(sel);
  if (!m) return null;
  return JDRAWN[m[1]] || (JDRAWN[m[1]] = Object.assign(jfake('piece ' + m[1]), { parentNode: JSVG }));
};
const JR = SESSION.ruleset, JW = JR.width, JH = JR.height, JCELL = 100 / Math.max(JW, JH);
const JVIEW = boardView(JW, JH, wrapAxes(JR.wrap), SHIFT, FLIPPED, JR.lattice,
                       boardCut(SESSION.absent, JR.wrap));
const jplay = () => animateChange(JSVG, { board: BEFORE }, JCELL, JW, JH, JVIEW,
                                 [{ kind: 'move', board: SESSION.board, steps: null }],
                                 SESSION.lastMove);
const jone = who => JLOG.find(e => e.who === who) || null;
"""


def _played(text: str, *, capture: bool = False) -> tuple[dict[str, Any], list[int]]:
    """The snapshot after `text` is played from the cylinder position, and the board before."""
    session = _cylinder_session()
    if capture:
        engine = session.engine
        session.state = engine.make_state({
            "d4": ("1o", WILD), "d5": ("1b", WILD), "e1": ("K", 0), "e8": ("K", 1),
            "a4": ("R", 0), "e5": ("N", 1), "c5": ("N", 1),
        })
    before = session.snapshot()
    (move,) = [m["move"] for m in before["moves"] if m["text"].startswith(text)]
    session.play(move)
    return session.snapshot(), before["board"]


def _run(text: str, expression: str, steps: str = "", *, capture: bool = False) -> Any:
    after, before = _played(text, capture=capture)
    setup = f"SESSION = {json.dumps(after)};\nconst BEFORE = {json.dumps(before)};\n" + _STAGE
    return _in_page(setup + steps, expression)


_ORDER = """({
  total: jplay(),
  plan: portalJourney(Math.hypot(...stepXY(portalWayIn(wrapStep(SESSION.lastMove.from,
          SESSION.lastMove.portal, JW, JH, wrapAllow(JR.wrap, 0)), SESSION.lastMove.exitDir,
          JW, JH), JCELL, JVIEW))),
  approach: jone('approach'), dive: jone('dive'), swirlIn: jone('swirl-in'),
  swirlOut: jone('swirl-out'), emerge: jone('piece ' + SESSION.lastMove.to),
  gulp: jone('piece ' + SESSION.lastMove.portal), kick: jone('piece ' + SESSION.lastMove.portalOut),
  log: JLOG, order: JORDER,
})"""


class TestTheJourneyIsOneMotionInOrder:
    def test_slide_then_dive_then_emerge_then_settle(self) -> None:
        got = _run("Ra4-d4>e5", _ORDER)
        plan, approach, dive, emerge = got["plan"], got["approach"], got["dive"], got["emerge"]
        assert approach and dive and emerge
        # The slide starts the move and is the plan's length.
        assert approach["opts"].get("delay", 0) == 0
        assert approach["opts"]["duration"] == plan["approach"]["for"]
        # The dive starts before the slide is over: one movement, not two beats.
        assert 0 < dive["opts"]["delay"] < approach["opts"]["duration"]
        # The piece comes out only once it has gone in, after the instant inside.
        assert dive["opts"]["delay"] + dive["opts"]["duration"] < emerge["opts"]["delay"]
        assert emerge["opts"]["delay"] == plan["emerge"]["at"]
        # The ends react as the piece passes: the entry gulps with the dive, the exit kicks
        # with the emergence, each with its swirl.
        assert got["gulp"]["opts"]["delay"] == dive["opts"]["delay"]
        assert got["swirlIn"]["opts"]["delay"] == dive["opts"]["delay"]
        assert abs(got["kick"]["opts"]["delay"] - emerge["opts"]["delay"]) <= 60
        assert abs(got["swirlOut"]["opts"]["delay"] - emerge["opts"]["delay"]) <= 60
        # About 0.6 to 1.0 s in all, and that is how long the board says it is busy.
        assert got["total"] == plan["landed"] == emerge["opts"]["delay"] + emerge["opts"]["duration"]
        assert 600 <= got["total"] <= 1000

    def test_it_shrinks_into_one_end_and_grows_out_of_the_other(self) -> None:
        got = _run("Ra4-d4>e5", _ORDER)
        dive, emerge = got["dive"]["frames"], got["emerge"]["frames"]
        assert dive[0]["transform"].endswith("scale(1, 1)") and dive[-1]["opacity"] == 0
        assert "scale(0.02)" in dive[-1]["transform"]
        assert emerge[0]["opacity"] == 0 and "scale(0.04)" in emerge[0]["transform"]
        # Held unseen until its moment, and nothing held after: at rest the piece is what
        # the board drew, and so are the two ends.
        assert got["emerge"]["opts"]["fill"] == "backwards"
        assert emerge[-1]["transform"] == "translate(0px, 0px) rotate(0deg) scale(1, 1)"
        assert "fill" not in got["gulp"]["opts"] and "fill" not in got["kick"]["opts"]

    def test_it_springs_rather_than_steps(self) -> None:
        got = _run("Ra4-d4>e5", _ORDER)
        emerge = got["emerge"]["frames"]
        # Past full size coming up, past its square and squashed across the way it moves
        # (out of d5 to e5, rightwards: narrower than tall), then settled.
        assert any("scale(1.2)" in f["transform"] for f in emerge)
        assert any(f["transform"].startswith("translate(1.25px, 0px)") for f in emerge)
        assert any("scale(0.9, 1.08)" in f["transform"] for f in emerge)
        assert any("scale(1.14, 0.86)" in f["transform"] for f in got["dive"]["frames"])
        eased = [f.get("easing", "") for f in emerge + got["dive"]["frames"]]
        assert sum("cubic-bezier" in e for e in eased) >= 5
        assert all("linear" not in e for e in eased)
        assert all("linear" not in str(e["opts"].get("easing", "")) for e in got["log"])

    def test_it_turns_the_way_it_goes_both_ways_round(self) -> None:
        right = _run("Ra4-d4>e5", _ORDER)
        left = _run("Ra4-d4>c5", _ORDER)
        assert "rotate(400deg)" in right["dive"]["frames"][-1]["transform"]
        assert "rotate(-320deg)" in right["emerge"]["frames"][0]["transform"]
        assert "rotate(-400deg)" in left["dive"]["frames"][-1]["transform"]
        assert "rotate(320deg)" in left["emerge"]["frames"][0]["transform"]
        # Going left is round the seam: a copy leaves by the near edge as the piece comes
        # in by the far one, the same slide one repeat apart.
        assert sum(e["who"] == "approach" for e in left["log"]) == 2
        assert 600 <= left["total"] <= 1000

    def test_both_colours_of_end_swirl_in_their_own_colour(self) -> None:
        entry, out = _run("Ra4-d4>e5", "[portalLook(BEFORE, SESSION.lastMove.portal),"
                          " portalLook(BEFORE, SESSION.lastMove.portalOut)]")
        assert (entry["ink"], entry["mark"]) == ("var(--portal-a)", "1o")
        assert (out["ink"], out["mark"]) == ("var(--portal-b)", "1b")

    def test_the_swirl_is_styled_by_a_class_and_custom_properties(self) -> None:
        """Q838 redraws the glyph: the ring's look is a hook it can restyle, not a colour."""
        rings = _run("Ra4-d4>e5", "(jplay(), JMADE.map(g => ({ attrs: g.attrs, html: g._html || '' })))",
                     "const JMADE = []; const jmake = document.createElementNS;"
                     " document.createElementNS = () => { const g = jmake(); JMADE.push(g);"
                     " return g; };")
        swirls = [r for r in rings if r["attrs"].get("class") == "portal-swirl"]
        assert [(r["attrs"]["data-swirl"], r["attrs"]["data-portal"]) for r in swirls] == [
            ("in", "1o"), ("out", "1b")]
        for ring in swirls:
            assert "var(--swirl-ink, var(--portal-" in ring["html"]
            assert "var(--swirl-edge, var(--portal-edge))" in ring["html"]
            assert "#" not in ring["html"]


class TestACaptureOnArrival:
    def test_the_piece_there_is_struck_as_the_traveller_arrives(self) -> None:
        got = _run("Ra4-d4>xe5", _ORDER, capture=True)
        plan, emerge = got["plan"], got["emerge"]
        struck = [e for e in got["log"] if e["who"] == "g" and delay(e) == plan["strike"]]
        # The victim knocked off and the ring struck where it stood (`strike`).
        assert len(struck) == 2
        start = emerge["opts"]["delay"]
        assert start < plan["strike"] < start + emerge["opts"]["duration"]
        assert got["total"] > plan["landed"]          # the knock-off finishes after
        # The traveller is laid over its victim, drawn after it, so the arrival is seen.
        assert got["order"][-1] == "piece 36"


def delay(entry: dict[str, Any]) -> float:
    return float(entry["opts"].get("delay", 0))


class TestReducedMotion:
    def test_a_portal_move_is_a_short_crossfade_with_nothing_moving(self) -> None:
        got = _run("Ra4-d4>e5", "({ total: jplay(), log: JLOG })", "STILL.matches = true;")
        assert 0 < got["total"] <= 400
        assert sorted(e["who"] for e in got["log"]) == ["piece 36", "still-from"]
        for entry in got["log"]:
            assert all(set(frame) == {"opacity"} for frame in entry["frames"])

    def test_a_capture_fades_too(self) -> None:
        got = _run("Ra4-d4>xe5", "({ total: jplay(), log: JLOG })", "STILL.matches = true;",
                   capture=True)
        assert sorted(e["who"] for e in got["log"]) == ["piece 36", "still-from", "still-taken"]

    def test_any_other_move_still_draws_at_once(self) -> None:
        got = _run("Ra4-a5", "({ total: jplay(), log: JLOG })", "STILL.matches = true;")
        assert got == {"total": 0, "log": []}


class TestReplayAndTakeBack:
    _TIMELINE = """
TIMELINE = [{ kind: 'start', board: BEFORE, lastMove: null },
            { kind: 'move', board: SESSION.board, lastMove: SESSION.lastMove }];
"""

    def test_a_step_forward_over_a_portal_move_plays_it(self) -> None:
        got = _run("Ra4-d4>e5", "[reviewTrip(0, 1), reviewTrip(1, 0), reviewTrip(0, 0),"
                   " reviewTrip(0, 1).board === BEFORE]", self._TIMELINE)
        assert got[0]["review"] is True and got[3] is True
        assert got[1:3] == [None, None]

    def test_a_step_forward_over_any_other_move_hands_the_position_over(self) -> None:
        assert _run("Ra4-a5", "reviewTrip(0, 1)", self._TIMELINE) is None

    def test_a_review_draw_plays_the_beats_move_and_a_take_back_plays_nothing(self) -> None:
        from page_text import PAGE

        assert "const review = !!(previous && previous.review && AT);" in PAGE
        assert "review ? AT.lastMove : SESSION.lastMove" in PAGE
        # A take-back draws the restored position whole (`applySnapshot(s, false)`).
        assert ".then(s => applySnapshot(s, false))" in PAGE
        assert "drawBoard(animate ? previous : null)" in PAGE


def test_the_rook_lands_where_the_engine_says() -> None:
    after, _ = _played("Ra4-d4>e5")
    assert (after["lastMove"]["from"], after["lastMove"]["to"]) == (_square("a4"), _square("e5"))
