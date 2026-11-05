"""Q1100: what the page draws a sweep's blasts and a sweep through a fold with.

SR6c's path blasts reach the page through the compiled rule (`explosion_payload`'s
``sweepPath``, beside ``by`` and ``of``), so `board.js` knows a path blast from a landing one;
SR6e's sweep through a fold reaches it through `fold_transit`, which reads the route off
`Engine.sweep_fold_route` and gives the sweep the ripple a go-through gets. Hand-built 6x6
specs, the shapes of `tests/test_sweep_blast.py` and `tests/test_sweep_folds.py`, not chess.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import node_session
import pytest
from page_text import PAGE

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_SWEEP, move_type
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.viz.rules_payload import explosion_payload, rules_payload
from chess_variant_sandbox.viz.wrapped import fold_transit, sweep_transit

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _sweeper(effects: str | None = None) -> dict[str, Any]:
    sweep: dict[str, Any] = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
    if effects is not None:
        sweep["effects"] = effects
    return {"id": "S", "value": 5, "moves": [sweep]}


def _spec(rules: dict[str, Any], effects: str | None = None) -> dict[str, Any]:
    return copy.deepcopy({
        "name": "a sweep on the page (Q1100 test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            _sweeper(effects),
            {"id": "G", "value": 2, "moves": [KING]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "S", "squares": ["b1"]},
            {"piece": "G", "squares": ["c1", "d1"]},
        ]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 60, **rules},
    })


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


BLAST = {"explode_on_capture": {"radius": 1, "immune": ["K"]}}


class TestPathBlastsReachThePage:
    def test_path_flags_the_sweeper(self) -> None:
        payload = explosion_payload(_engine(_spec(BLAST, effects="path")).spec)
        assert payload is not None
        assert payload["sweepPath"] == [False, True, False]

    def test_landing_sends_none(self) -> None:
        payload = explosion_payload(_engine(_spec(BLAST)).spec)
        assert payload is not None and payload["sweepPath"] is None

    def test_the_board_reads_it(self) -> None:
        """`board.js` centres a sweep's blasts on the engine's swept squares under the flag,
        through `sweepBlastCentres`, the one copy the offline replay shares (Q1101)."""
        assert "ex.sweepPath[sweeper.type]" in PAGE
        assert PAGE.count("function sweepBlastCentres(") == 1
        assert "const centres = swept.filter(fires);" in PAGE
        assert "sweepBlastCentres(ex, before, pieceOf, last.from, last.to, last.swept)" in PAGE


FOLDS = {"portals": [{"orange": "c3", "blue": "e5", "kind": "fold"}],
         "win": "checkmate"}


class TestASweepThroughAFold:
    def _played(self) -> tuple[Engine, Any, int]:
        spec = _spec(FOLDS)
        engine = _engine(spec)
        placement = {"a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0), "b3": ("G", 1)}
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        move = engine.find_move(state, "Sa3-c3~e5 (sweeps b3)")
        assert move_type(move) == MOVE_SWEEP
        return engine, state, move

    def test_the_ripple_gets_the_engines_route(self) -> None:
        engine, state, move = self._played()
        assert fold_transit(engine, state, move) == {
            "foldIn": _sq("c3"), "foldInName": "c3",
            "foldOut": _sq("e5"), "foldOutName": "e5", "foldPair": 1,
        }
        assert sweep_transit(engine, state, move) == {"swept": [_sq("b3")]}

    def test_a_sweep_that_stops_in_the_fold_goes_through_none(self) -> None:
        engine, state, _ = self._played()
        stay = engine.find_move(state, "Sa3xc3 (sweeps b3)")
        assert move_type(stay) == MOVE_SWEEP
        assert fold_transit(engine, state, stay) is None

    def test_the_board_cuts_the_line_before_the_fold(self) -> None:
        assert "sweepCut(svg, node, at(from), at(last.foldIn)" in PAGE


class TestASweepThroughAFoldThatBlasts:
    """Q1101: the validator takes a sweep through a fold beside a blast, and the move carries
    both the fold's route and the swept squares; the sweeper dies in its blast, so it never
    pairs, and `board.js` sends its ghost the fold's way (`foldTrip`), not straight across."""

    def test_the_pairing_validates_and_carries_both(self) -> None:
        spec = _spec({**FOLDS, **BLAST}, effects="path")
        engine = _engine(spec)
        placement = {"a1": ("K", 0), "a6": ("K", 1), "a3": ("S", 0), "b3": ("G", 1)}
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        move = engine.find_move(state, "Sa3-c3~e5 (sweeps b3)")
        assert fold_transit(engine, state, move) is not None
        assert sweep_transit(engine, state, move) == {"swept": [_sq("b3")]}
        after = engine.apply(state, move)
        assert after.board[_sq("e5")] == 0 and after.board[_sq("a3")] == 0

    def test_the_ghost_goes_through_the_fold(self) -> None:
        assert "foldTrip(svg, ghost, sweeper, before, last, cell, W, H, view)" in PAGE
        assert "sweepCut(svg, ghost, [fx, fy], at(last.foldIn)" in PAGE


# The page's script run in node with a DOM good enough to load it (`test_web._in_page`'s), and
# the drawing helpers `animateMove` calls replaced by recorders: what it asks to be drawn is
# read, never a pixel. Each recorder names the square it was handed by its corner.
_ANIMATE = """
const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null), set: () => true });
global.document = { getElementById: () => el, querySelector: () => null,
                    querySelectorAll: () => [], addEventListener: () => {},
                    createElement: () => el, createElementNS: () => el, body: el };
global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                  addEventListener: () => {}, innerWidth: 1200, scrollTo: () => {} };
global.location = { pathname: '/game', hash: '' };
global.fetch = () => new Promise(() => {});
__PAGE__
const DRAWN = { fired: [], burned: [], knocked: [], cut: [] };
const node = { animate: () => ({}), setAttribute: () => {}, remove: () => {} };
const svg = { querySelector: () => node, appendChild: () => {} };
SESSION = { ruleset: __RULESET__ };
const VIEW = boardView(6, 6, {}, null, false);
const named = (x, y) => [...Array(36).keys()].find(s => {
  const [a, b] = squareXY(s, 10, VIEW); return a === x && b === y; });
blastFire = (svg, area) => { DRAWN.fired.push(area[0]); return 0; };
burn = (svg, piece, x, y) => { DRAWN.burned.push(named(x, y)); return 0; };
capture = (svg, piece, x, y) => { DRAWN.knocked.push(named(x, y)); return 0; };
sweepCut = (svg, mover, from, to, victims) => { DRAWN.cut.push(victims.length); return 0; };
travel = () => 100;
ghostAt = () => node;
resist = () => 0;
animateMove(svg, __BEFORE__, __AFTER__, new Set(), 10, 6, 6, VIEW, __LAST__);
console.log(JSON.stringify(DRAWN));
"""


class TestAnImmuneSweepersBlasts:
    """Q1108: a sweeper named under `immune` does not survive its own blast. The core destroys
    the sweeper once, on its landing, whenever a blast goes off (RULES.md's `effects`: *the
    sweeper destroyed once*; `immune` spares bystanders, never the capturer), and a strike or
    contacts beside a sweep's blast is refused (W3), so nothing ever stands on the landing of a
    sweep that blasted. `board.js` therefore takes the sweeper's ghost branch for it, as for any
    sweeper: its blasts go off where `sweepBlastCentres` (`viz/wrapped.js`, the replay's rule)
    puts them -- under `effects: "path"` on each swept guard, else on the landing -- the
    sweeper burns on its landing, and a guard beside a path blast burns rather than being
    knocked off."""

    @staticmethod
    def _spec(effects: str | None) -> dict[str, Any]:
        spec = _spec({"explode_on_capture": {"radius": 1, "immune": ["K", "S"]}}, effects)
        spec["setup"] = {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "S", "player": 0, "squares": ["a3"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "G", "player": 1, "squares": ["b3", "d3", "c4"]},
        ]}
        return spec

    @pytest.mark.parametrize(("effects", "centres", "burned", "knocked"), [
        ("path", ["b3", "d3"], ["e3", "c4"], []),
        (None, ["e3"], ["e3"], []),
    ])
    def test_the_board_draws_them(self, effects: str | None, centres: list[str],
                                  burned: list[str], knocked: list[str]) -> None:
        engine = _engine(self._spec(effects))
        state = engine.initial_state()
        move = engine.find_move(state, "Sa3xe3 (sweeps b3, d3)")
        after = engine.apply(state, move)
        assert after.board[_sq("e3")] == 0 and after.board[_sq("a3")] == 0  # it went up
        last = {"from": _sq("a3"), "to": _sq("e3"), **sweep_transit(engine, state, move)}
        script = re.search(r"<script>(.*)</script>", PAGE, re.S)
        assert script is not None
        body = (_ANIMATE.replace("__PAGE__", script.group(1))
                .replace("__RULESET__", json.dumps(rules_payload(engine)))
                .replace("__BEFORE__", json.dumps(list(state.board)))
                .replace("__AFTER__", json.dumps(list(after.board)))
                .replace("__LAST__", json.dumps(last)))
        done = node_session.run(body)
        assert done.returncode == 0, done.stderr
        drawn = json.loads(done.stdout)
        assert drawn["fired"] == [_sq(name) for name in centres]
        assert drawn["cut"] == [2]                       # b3 and d3, cut down as it passes
        assert drawn["burned"] == [_sq(name) for name in burned]
        assert drawn["knocked"] == [_sq(name) for name in knocked]
