"""Portals on the play page, as P1 of `docs/plan-portals.md` draws them (Q771).

The minimal page: the compiled payload sent once per ruleset (`portal_payload`), each
transit offered as the engine's own move with the end it goes in by named, and a choice of
route where one square is reached more than one way -- the promotion's card, one button per
move. P2 draws the real glyph and the two-leg trail.

The position is hand-built on §6's Wormhole (Gardner, one pair on b3 and d3): a knight on c1
reaches both ends, and going in by either it comes out on c4, so c4 is two moves.
"""

from __future__ import annotations

import json
from typing import Any

from test_web import _in_page
from test_web_landing import _HARNESS

from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web.library import Game, engine_for
from chess_variant_sandbox.web.play import Session


def _wormhole() -> dict[str, Any]:
    from test_schema_nineteen import wormhole

    return wormhole()


def _session() -> Session:
    spec = _wormhole()
    engine = engine_for(spec)
    state = engine.make_state({
        "b3": ("1o", WILD), "d3": ("1b", WILD), "e1": ("K", 0), "a5": ("K", 1),
        "c1": ("N", 0),
    })
    game = Game(hash=spec_hash(spec), name=str(spec["name"]), description="",
                path=None, source="scratch", spec=spec)  # type: ignore[arg-type]
    return Session(id="g1", game=game, seats=("human", "medium"), seed=0, state=state)


_SNAPSHOT: dict[str, Any] = _session().snapshot()


def _to(square: str) -> list[dict[str, Any]]:
    index = (int(square[1:]) - 1) * 5 + ord(square[0]) - ord("a")
    return [m for m in _SNAPSHOT["moves"] if m["from"] == 2 and m["to"] == index]


class TestTheSnapshot:
    def test_the_payload_names_each_end(self) -> None:
        ends = _SNAPSHOT["ruleset"]["portals"]["ends"]
        assert [(end["pair"], end["end"]) for end in ends] == [(1, "orange"), (1, "blue")]
        assert all(end["open"] == [True, True] and end["movedBy"] == [] for end in ends)
        assert [_SNAPSHOT["ruleset"]["pieceIds"][end["type"]] for end in ends] == ["1o", "1b"]

    def test_a_transit_names_the_end_it_goes_in_by(self) -> None:
        moves = _to("c4")
        assert sorted(m["text"] for m in moves) == ["Nc1-b3>c4", "Nc1-d3>c4"]
        assert sorted(m["portalName"] for m in moves) == ["b3", "d3"]
        assert all(m["kind"] == "portal" for m in moves)

    def test_a_game_without_portals_sends_none(self) -> None:
        assert '"portals": null' in json.dumps(
            Session(id="g2", game=_chess(), seats=("human", "medium"), seed=0,
                    state=engine_for(_chess().spec).initial_state()).snapshot()["ruleset"]
        )


def _chess() -> Game:
    from test_schema_nineteen import _ruleset

    spec = _ruleset("chess")
    return Game(hash=spec_hash(spec), name="chess", description="", path=None,
                source="scratch", spec=spec)  # type: ignore[arg-type]


def _run(steps: str, expression: str) -> Any:
    setup = (f"SESSION = {json.dumps(_SNAPSHOT)};\n" + _HARNESS
             + "const open = () => { clickSquare(2); clickSquare(17); };\n")
    return _in_page(setup + steps, expression)


class TestTheRouteChooser:
    def test_two_ways_to_one_square_open_the_card(self) -> None:
        hidden, text, picks = _run("open();", "[BOX.hidden, said(), BOX.picks.length]")
        assert hidden is False and picks == 2
        assert text.startswith("Choose how your knight gets there")
        assert "through portal 1, the orange one on b3" in text
        assert "through portal 1, the blue one on d3" in text

    def test_a_pick_is_the_move(self) -> None:
        sent = _run("open(); BOX.picks[1].click();", "CALLS.map(c => c[1].move)")
        moves = sorted(m["move"] for m in _to("c4"))
        assert sent and sent[0] in moves

    def test_one_way_to_a_square_is_sent_at_once(self) -> None:
        index = (2 - 1) * 5 + 0  # a2
        (move,) = [m for m in _SNAPSHOT["moves"] if m["from"] == 2 and m["to"] == index]
        sent = _run(f"clickSquare(2); clickSquare({index});", "[BOX.hidden, CALLS.length]")
        assert sent == [True, 1], move
