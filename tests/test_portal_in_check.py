"""In check with no piece to move, and a portal end that can (Q833).

A player on the live site, White on `archive/ferry-no-straight-back.json` against the
computer, reached a position in check where none of White's pieces had a legal move, read it
as a stalemate the page did not end, and resigned. The game is stored on the site
(`finishes`, reason `resigned`); its 28 moves are `STORED` below, replayed through the core.

The core is right (`docs/reference/q833-ferry-no-moves.md`): every portal end in that game
may be moved by either player, and two of them can step onto a3, between the queen on a2
and the king on a4, which answers the check. So the game was not over, and *no legal move
while in check* in this game is checkmate (`rules.win`), not stalemate. What was missing was
the page saying so: a tapped piece read *no legal move - your side is in check*, and nothing
pointed at the portals. The note now names them when they are all that can move.

Asserted against the spec and the core; the position is the one the player's screenshot
showed, built independently with `make_state` and compared with the replay's.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from test_web import _in_page
from test_web_landing import _HARNESS

from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.site.finishes import replay
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web.library import Game, engine_for
from chess_variant_sandbox.web.play import Session

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "archive" / "ferry-no-straight-back.json"

# Match `TIQJSEtVAcQQ`, finished 2026-10-10 19:22 UTC, reason `resigned`: White (a person)
# against `greedy`, read from the site's store with a read-only query.
STORED = [201792774, 235341851, 2050, 235341856, 2051, 5410, 776, 302450970, 2307, 5151, 521,
          2581, 770, 1812, 2563, 7459, 268900618, 7966, 7961, 5127, 6431, 268897812, 3585,
          235341596, 4633, 3342343, 1550, 1536]

# The screenshot: White to move and in check, the last move a1 to a2.
SCREENSHOT = {
    "a4": ("K", 0), "e1": ("N", 0), "f1": ("R", 0), "f2": ("P", 0),
    "d6": ("K", 1), "a2": ("Q", 1), "f5": ("R", 1), "a5": ("P", 1),
    "b4": ("2b", WILD), "e4": ("1b", WILD), "b3": ("1o", WILD), "e3": ("2o", WILD),
}


def _spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _final() -> Any:
    return replay(_spec(), STORED).state


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * 6 + ord(name[0]) - ord("a")


def _pairs(engine: Any, state: Any) -> set[tuple[int, int]]:
    return {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}


class TestTheStoredGame:
    def test_it_replays_to_the_screenshot(self) -> None:
        engine = engine_for(_spec())
        state = _final()
        assert state.board == engine.make_state(SCREENSHOT).board
        assert state.side_to_move == 0 and engine.in_check(state)
        # No portal had just moved, so `no_straight_back` bars nothing here.
        assert state.portal_bar == (-1, -1)

    def test_the_only_moves_put_a_portal_between_queen_and_king(self) -> None:
        engine = engine_for(_spec())
        state = _final()
        assert _pairs(engine, state) == {(_sq("b3"), _sq("a3")), (_sq("b4"), _sq("a3"))}
        for move in engine.legal_moves(state):
            after = engine.apply(state, move)
            assert not engine.in_check(after, 0)
        assert engine.terminal(state) is None

    def test_with_the_portals_out_of_whites_hands_it_is_checkmate(self) -> None:
        # The same position in a spec where only Black may move a portal end: White has no
        # move, and the core calls it what `rules.win` says, a loss, not a stalemate. Without
        # `no_straight_back`, which no move could meet with one player moving every end
        # (refused, Q879); it bars nothing in this position either way.
        spec = copy.deepcopy(_spec())
        for pair in spec["rules"]["portals"]:
            pair["moved_by"] = {"orange": [1], "blue": [1]}
            pair.pop("no_straight_back", None)
        engine = engine_for(spec)
        state = engine.make_state(SCREENSHOT)
        assert engine.legal_moves(state) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 1 and outcome.reason == "checkmate"


def _snapshot() -> dict[str, Any]:
    spec = _spec()
    game = Game(hash=spec_hash(spec), name=str(spec["name"]), description="",
                path=None, source="scratch", spec=spec)  # type: ignore[arg-type]
    return Session(id="q833", game=game, seats=("human", "greedy"), seed=0,
                   state=_final()).snapshot()


class TestThePage:
    def test_the_snapshot_offers_both_portal_moves(self) -> None:
        moves = _snapshot()["moves"]
        assert {(m["from"], m["to"]) for m in moves} == {
            (_sq("b3"), _sq("a3")), (_sq("b4"), _sq("a3"))}

    def _run(self, steps: str, expression: str) -> Any:
        setup = f"SESSION = {json.dumps(_snapshot())};\n" + _HARNESS
        return _in_page(setup + steps, expression)

    def test_a_tapped_piece_points_at_the_portals(self) -> None:
        note = self._run(f"clickSquare({_sq('a4')});", "NOTE")
        assert note == ("that king has no legal move - your side is in check; "
                        "only a portal can move: tap the one on b3 or b4")

    def test_a_tapped_portal_end_is_picked(self) -> None:
        picked = self._run(f"clickSquare({_sq('b3')});", "SELECTED")
        assert picked == _sq("b3")
