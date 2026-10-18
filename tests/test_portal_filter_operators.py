"""Q849: an operator that changes the piece set never leaves a portal filter refused.

Schema 21's ``carries`` names the pieces that may go in by an end, and the validator refuses
one that names every piece that can move (it is no filter, written the long way: one game,
one hash) and one that names nobody (`docs/archive/plan-portals-2.md` §2.3: say that with who may
``enter``). Removing the one type a filter kept out leaves the first, which
``transform.settle_carries`` drops; removing the one type a filter let in leaves the second,
which the design loop's ``remove-a-piece`` refuses as a draw (``empties_a_filter``) rather
than widening the door to everyone. Tested on Los Alamos (P, N, R, Q, K) with one filtered
pair or ring, since no committed spec carries a filter yet.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.loop.operators import OPERATORS
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.schema import STOCK_UNLIMITED
from chess_variant_sandbox.spec.transform import (
    drop_piece,
    empties_a_filter,
    settle_carries,
    settle_inert_rules,
)

ROOT = Path(__file__).resolve().parents[1]
REMOVE = next(op for op in OPERATORS if op.name == "remove-a-piece")
MOVERS = ["P", "N", "R", "Q", "K"]


def _losalamos(portals: list[dict[str, Any]]) -> dict[str, Any]:
    spec = json.loads((ROOT / "rulesets" / "losalamos.json").read_text(encoding="utf-8"))
    spec["schema_version"] = 21
    spec["pieces"].append({"id": "O", "name": "portal", "value": 0, "moves": []})
    spec["rules"]["portals"] = portals
    return spec


def pair(orange: list[str] | None = None, blue: list[str] | None = None) -> dict[str, Any]:
    """A fixed pair on b3 and e4, each end filtered as given (``None``: no filter)."""
    carries = {end: ids for end, ids in (("orange", orange), ("blue", blue)) if ids}
    group: dict[str, Any] = {"piece": "O", "orange": "b3", "blue": "e4"}
    if carries:
        group["carries"] = carries
    return _losalamos([group])


def ring(*filters: list[str] | None) -> dict[str, Any]:
    """A ring b3 to e3 to c4 to b3, each end filtered as given (``None``: no filter)."""
    ends: list[dict[str, Any]] = [
        {"at": "b3", "to": 1}, {"at": "e3", "to": 2}, {"at": "c4", "to": 0},
    ]
    for end, ids in zip(ends, filters, strict=False):
        if ids:
            end["carries"] = ids
    return _losalamos([{"piece": "O", "ends": ends}])


class _Pick(random.Random):
    """An rng whose ``choice`` takes the named piece, and remembers what it was offered."""

    def __init__(self, piece_id: str) -> None:
        super().__init__(0)
        self.piece_id = piece_id
        self.offered: list[str] = []

    def choice(self, seq: Any) -> Any:
        self.offered = [str(item["id"]) for item in seq]
        return next(item for item in seq if item["id"] == self.piece_id)


def _all_but(piece_id: str) -> list[str]:
    return [pid for pid in MOVERS if pid != piece_id]


class TestAFilterLeftNamingEveryMoverIsDropped:
    def test_the_parents_validate(self) -> None:
        assert validate_spec(pair(orange=_all_but("R"))) == []
        assert validate_spec(ring(_all_but("R"))) == []

    def test_on_a_pair(self) -> None:
        spec = pair(orange=_all_but("R"), blue=["N"])
        summary = REMOVE.apply(spec, _Pick("R"))
        assert summary is not None and summary.startswith("removed the")
        assert spec["rules"]["portals"][0]["carries"] == {"blue": ["N"]}
        assert validate_spec(spec) == []

    def test_on_a_pair_whose_last_filter_it_was(self) -> None:
        spec = pair(orange=_all_but("R"))
        assert REMOVE.apply(spec, _Pick("R")) is not None
        assert "carries" not in spec["rules"]["portals"][0]
        assert validate_spec(spec) == []

    def test_on_an_end_of_a_map(self) -> None:
        spec = ring(_all_but("R"), ["Q", "R"])
        assert REMOVE.apply(spec, _Pick("R")) is not None
        ends = spec["rules"]["portals"][0]["ends"]
        assert "carries" not in ends[0]
        assert ends[1]["carries"] == ["Q"]
        assert validate_spec(spec) == []

    def test_drop_piece_alone_drops_it_too(self) -> None:
        spec = drop_piece(ring(_all_but("N")), "N")
        assert "carries" not in spec["rules"]["portals"][0]["ends"][0]
        assert validate_spec(spec) == []

    def test_the_settle_after_every_operator_names_the_end(self) -> None:
        """A filter the piece set grew around (the one mover it kept out made a stone,
        here by hand) is what ``settle_inert_rules`` drops and says, and nothing on a
        valid spec."""
        spec = ring(_all_but("R"))
        assert settle_inert_rules(copy.deepcopy(spec)) == []
        rook = next(p for p in spec["pieces"] if p["id"] == "R")
        rook["moves"] = []
        spec["rules"]["drops"] = {"from_captures": False, "stock": {"R": STOCK_UNLIMITED}}
        assert any("names every piece that can move" in e for e in validate_spec(spec))
        assert settle_inert_rules(spec) == ["carries:b3"]
        assert validate_spec(spec) == []

    def test_settle_carries_is_a_no_op_on_a_valid_spec(self) -> None:
        spec = pair(orange=_all_but("R"), blue=["N", "P"])
        before = copy.deepcopy(spec)
        assert settle_carries(spec) == []
        assert spec == before


class TestAFilterLeftNamingNobodyIsRefused:
    """An end only the removed piece went in by lets nobody in; no filter says that, and
    dropping it would open the end to every piece. The operator does not draw that piece."""

    def test_the_draw_never_offers_the_one_piece_a_filter_names(self) -> None:
        spec = pair(orange=["N"], blue=["R", "Q"])
        assert empties_a_filter(spec, "N")
        assert not empties_a_filter(spec, "R")
        pick = _Pick("P")
        assert REMOVE.apply(spec, pick) is not None
        assert "N" not in pick.offered and "R" in pick.offered
        assert spec["rules"]["portals"][0]["carries"] == {"orange": ["N"], "blue": ["R", "Q"]}
        assert validate_spec(spec) == []

    def test_on_a_map(self) -> None:
        spec = ring(None, ["Q"])
        pick = _Pick("N")
        assert REMOVE.apply(spec, pick) is not None
        assert "Q" not in pick.offered
        assert validate_spec(spec) == []

    def test_a_refusal_when_nothing_else_can_go(self) -> None:
        """Every non-royal type is some end's only piece, and the portal's own type (whose
        removal takes the filters with it) is not on offer: the operator does nothing."""
        spec = _losalamos([
            {"piece": "O", "orange": "b3", "blue": "e4",
             "carries": {"orange": ["N"], "blue": ["R"]}},
            {"piece": "O", "orange": "e3", "blue": "b4",
             "carries": {"orange": ["Q"], "blue": ["P"]}},
        ])
        assert validate_spec(spec) == []
        offered = [pid for pid in "PNRQ" if not empties_a_filter(spec, pid)]
        assert offered == []
        before = copy.deepcopy(spec)
        trial = copy.deepcopy(spec)
        trial["pieces"][-1]["royal"] = True  # hide the portal type from the draw only
        assert REMOVE.apply(trial, random.Random(1)) is None
        trial["pieces"][-1].pop("royal")
        assert trial == before

    def test_drop_piece_still_hands_back_a_valid_spec(self) -> None:
        """Outside the loop (the metrics' ablations) the emptied filter goes, as castling
        goes with its rook: the end then lets every piece in."""
        spec = drop_piece(pair(orange=["N"], blue=["R"]), "N")
        assert spec["rules"]["portals"][0]["carries"] == {"blue": ["R"]}
        assert validate_spec(spec) == []


def _random_filtered(rng: random.Random) -> dict[str, Any]:
    """A pair or a ring on Los Alamos with a random valid filter on some of its ends."""
    def draw() -> list[str] | None:
        if rng.random() < 0.3:
            return None
        return sorted(rng.sample(MOVERS, rng.randint(1, len(MOVERS) - 1)))

    if rng.random() < 0.5:
        return pair(orange=draw(), blue=draw())
    return ring(draw(), draw(), draw())


@pytest.mark.parametrize("seed", range(40))
def test_remove_a_piece_on_a_filtered_spec_is_valid_or_refused(seed: int) -> None:
    """The property: whatever the filters, ``remove-a-piece`` hands back a spec the
    validator takes, or ``None``; and taking any one piece out by hand, a valid spec."""
    rng = random.Random(seed)
    parent = _random_filtered(rng)
    assert validate_spec(parent) == []
    for draw in range(6):
        child = copy.deepcopy(parent)
        summary = REMOVE.apply(child, random.Random(seed * 100 + draw))
        if summary is None:
            assert child == parent
        else:
            assert validate_spec(child) == [], summary
    for piece in parent["pieces"]:
        if piece.get("royal"):
            continue
        child = drop_piece(copy.deepcopy(parent), piece["id"])
        settle_inert_rules(child)
        assert validate_spec(child) == [], piece["id"]
