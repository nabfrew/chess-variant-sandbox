"""H1 (Q1125, `docs/plan-hunger.md` §2): a move's cost, ``pieces[].upkeep`` and
``rules.purse.upkeep``.

The core's filtered move list, its `Engine.move_cost` and the purse after every move are held to
`tests/upkeep_oracle.py`, which reads the spec by file and rank against the twin without the
rule, at every position of playouts on four arms (Hunger Chess's charges on chess, a market
whose buys pay their square too under each promotion reading, and a rifle whose captures pay
for the square the shooter stays on). Perft against a walk of fresh applies holds the undo, and
the incremental key is checked at every ply. The rule is inert at its off values, refused by
name where its first build has no reading (§2.4), and an empty purse with nothing free to move
is stalemate.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_BUY,
    MOVE_PROMOTION,
    move_type,
)
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec as describe
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests import upkeep_oracle as oracle
from tests.feature_inventory import _schema_one_hundred_fifteen_uses as uses

ROOT = Path(__file__).resolve().parents[1]
_HUNGER = {"P": 1, "N": 2, "B": 2, "R": 2, "Q": 2}


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _hunger(start: int = 10, upkeep: Any = True) -> dict[str, Any]:
    """Chess with Hunger Chess's charges (the king free) and booty, no income or pass (H1)."""
    spec = _load("rulesets/chess.json")
    for piece in spec["pieces"]:
        if piece["id"] in _HUNGER:
            piece["upkeep"] = _HUNGER[piece["id"]]
            piece["bounty"] = 1 if piece["id"] == "P" else 2
    spec["rules"]["purse"] = {"start": start, "upkeep": upkeep}
    return spec


def _market(upkeep: Any) -> dict[str, Any]:
    """The purse market (6x6, buys on the back rank, promotion on the last) with upkeep."""
    spec = _load("tests/_purse_market.json")
    for piece in spec["pieces"]:
        piece["upkeep"] = {"K": 1, "R": 2, "N": 2, "P": 1}[piece["id"]]
    spec["rules"]["purse"].update(start=99, upkeep=upkeep)
    return spec


def _rifle() -> dict[str, Any]:
    """The rifle (captures leave the shooter where it stood) with upkeep and booty."""
    spec = _load("archive/rifle.json")
    for piece in spec["pieces"]:
        if piece["id"] in _HUNGER:
            piece["upkeep"] = _HUNGER[piece["id"]]
            piece["bounty"] = 2
    spec["rules"]["purse"] = {"start": 20, "upkeep": True}
    return spec


ARMS = {
    "hunger without income": _hunger(),
    "market, both arrivals": _market(True),
    "market, the new type alone": _market({"promotion": "new"}),
    "rifle": _rifle(),
}


@pytest.mark.parametrize("name", sorted(ARMS))
def test_the_arms_are_valid(name: str) -> None:
    assert validate_spec(ARMS[name]) == []


@pytest.mark.parametrize("name", sorted(ARMS))
def test_the_core_charges_what_the_oracle_charges(name: str) -> None:
    """Every position of six playouts: the list, each move's cost, the purse after it, the
    incremental key."""
    spec = ARMS[name]
    engine, twin = Engine(spec), Engine(oracle.twin(spec))
    rng = random.Random(name)
    filtered = promotions = bought = 0
    for _ in range(6):
        state = engine.initial_state()
        for _ply in range(80):
            legal = engine.legal_moves(state)
            expected = oracle.moves(spec, engine, twin, state)
            assert legal == expected, (state.purse, state.side_to_move)
            filtered += len(twin.legal_moves(state)) - len(legal)
            mover = state.side_to_move
            for move in legal:
                charged = oracle.cost(spec, engine, state, move)
                assert engine.move_cost(state, move) == charged, move
                after = engine.apply(state, move)
                assert after.purse[mover] == oracle.purse_after(spec, engine, state, move), move
                assert after.purse[1 - mover] == state.purse[1 - mover]
                promotions += move_type(move) == MOVE_PROMOTION
                bought += move_type(move) == MOVE_BUY
            if engine.terminal(state, legal) is not None:
                break
            state = engine.apply(state, rng.choice(legal))
            assert engine.hash(state) == engine._compute_hash(state)
    assert filtered > 0, "the purse never refused a move: the arm shows nothing"
    if name.startswith("market"):
        assert promotions > 0 and bought > 0


def _walk(engine: Engine, state: Any, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(_walk(engine, engine.apply(state, m), depth - 1) for m in engine.legal_moves(state))


@pytest.mark.parametrize("name", sorted(ARMS))
def test_perft_unmakes_the_payment(name: str) -> None:
    """Perft makes and unmakes in place, the walk applies afresh: equal counts from a middle
    position say `unmake` gave every coin back (the filter reads the purse at every node)."""
    engine = Engine(ARMS[name])
    state = engine.initial_state()
    rng = random.Random(2)
    for _ in range(8):
        legal = engine.legal_moves(state)
        if not legal:
            break
        state = engine.apply(state, rng.choice(legal))
    assert engine.perft(state, 3) == _walk(engine, state, 3)


def test_hungers_numbers() -> None:
    """The designer's own: a pawn's double step pays 3, a knight's first move 4; a promotion
    to a queen on the eighth pays 21 under both arrivals and 14 for the new type alone."""
    engine = Engine(_hunger())
    state = engine.initial_state()
    costs = {engine.describe_move(state, m): engine.move_cost(state, m) for m in engine.legal_moves(state)}
    assert costs["Pe2-e4"] == 3 and costs["Pe2-e3"] == 2 and costs["Ng1-f3"] == 4
    for reading, queen in ((True, 21), ({"promotion": "new"}, 14)):
        spec = _hunger(start=99, upkeep=reading)
        spec["setup"] = {"mirror": False, "placements": [
            {"piece": "K", "squares": ["e1"]}, {"piece": "P", "squares": ["a7"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
        ]}
        spec["rules"].pop("castling", None)
        engine = Engine(spec)
        state = engine.initial_state()
        paid = {
            engine.describe_move(state, m): engine.move_cost(state, m)
            for m in engine.legal_moves(state) if move_type(m) == MOVE_PROMOTION
        }
        assert paid and max(paid.values()) == queen, paid


def test_an_empty_purse_with_nothing_free_is_stalemate() -> None:
    """No pass until H3: a player who can pay for nothing has no legal move."""
    engine = Engine(_hunger(start=0))
    state = engine.initial_state()
    assert engine.legal_moves(state) == []
    outcome = engine.terminal(state)
    assert outcome is not None and outcome.reason == "stalemate"


def test_the_off_values_and_spellings_name_one_game() -> None:
    """``true``, ``{}`` and ``{"promotion": "both"}`` are one game; ``false`` is the purse
    without the rule; a zero upkeep is no upkeep."""
    true = _hunger()
    assert spec_hash(_hunger(upkeep={})) == spec_hash(true)
    assert spec_hash(_hunger(upkeep={"promotion": "both"})) == spec_hash(true)
    assert spec_hash(_hunger(upkeep={"promotion": "new"})) != spec_hash(true)
    market = _load("tests/_purse_market.json")
    off = copy.deepcopy(market)
    off["rules"]["purse"]["upkeep"] = False
    off["pieces"][0]["upkeep"] = 0
    assert validate_spec(off) == []
    assert spec_hash(off) == spec_hash(market)
    tables = build_tables(compile_spec(market))
    assert tables["purse_upkeep"] == [] and tables["purse_upkeep_new"] is False
    assert build_tables(compile_spec(true))["purse_upkeep"] != []


def test_the_upkeep_alone_is_something_to_spend_on() -> None:
    """A purse whose only use is paying for moves is not refused as having nothing to spend on."""
    assert validate_spec(_hunger()) == []


def _refusal(edit: Any) -> list[str]:
    spec = _hunger()
    edit(spec)
    return validate_spec(spec)


def _set(path: str, value: Any) -> Any:
    def edit(spec: dict[str, Any]) -> None:
        spec["rules"][path] = value
    return edit


def _command(spec: dict[str, Any]) -> None:
    spec["pieces"][1]["sight"] = [
        {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 1,
         "effect": "command", "whose": "enemy"},
    ]


def _shared(spec: dict[str, Any]) -> None:
    spec["pieces"].append({"id": "S", "name": "stone", "value": 1, "control": "shared",
                           "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]})
    spec["setup"]["placements"].append({"piece": "S", "player": "nobody", "squares": ["d4"]})


def _hex(spec: dict[str, Any]) -> None:
    spec["board"]["lattice"] = "hex"


REFUSED = {
    "upkeep.U1": _set("simultaneous", {}),
    "upkeep.U2": _set("turn", {"moves": 2}),
    "upkeep.U5": _command,
    "upkeep.U6": _shared,
    "upkeep.U7": _set("drops", {"from_captures": True}),
    "upkeep.U8": _hex,
    "upkeep.U9": _set("contacts", {"links": [{"mover": "N", "onto": "P", "mover_to": "B"}]}),
    "upkeep.U10": _set("strike", {"effect": "capture", "switches_sides": ["N"]}),
}


@pytest.mark.parametrize("rid", sorted(REFUSED))
def test_each_refusal_by_name(rid: str) -> None:
    assert rid in refused_ids(_refusal(REFUSED[rid]))


def test_the_rule_and_its_rates_need_each_other() -> None:
    lone_rule = _hunger()
    for piece in lone_rule["pieces"]:
        piece.pop("upkeep", None)
    assert any("no pieces[].upkeep" in e for e in validate_spec(lone_rule))
    lone_rates = _hunger()
    lone_rates["rules"]["purse"].pop("upkeep")
    assert any("without rules.purse.upkeep" in e for e in validate_spec(lone_rates))
    no_promotion = _hunger(upkeep={"promotion": "new"})
    no_promotion["rules"].pop("promotion")
    assert any('promotion "new" without' in e for e in validate_spec(no_promotion))
    bad = _hunger(upkeep={"promotion": "old"})
    assert any("must be \"both\" or \"new\"" in e for e in validate_spec(bad))
    too_dear = _hunger()
    too_dear["pieces"][0]["upkeep"] = 11
    assert any("from 0 to 10" in e for e in validate_spec(too_dear))


def test_the_names_the_words_and_fairy() -> None:
    assert uses(_hunger()) == {"pieces.upkeep", "rules.purse.upkeep"}
    assert uses(_hunger(upkeep={"promotion": "new"})) == {
        "pieces.upkeep", "rules.purse.upkeep", "rules.purse.upkeep.promotion",
    }
    assert uses(_load("tests/_purse_market.json")) == set()
    words = describe(_hunger())
    assert "Moves cost coins" in words and "both arrivals" in words
    assert "promoted piece's arrival only" in describe(_hunger(upkeep={"promotion": "new"}))
    with pytest.raises(Untranslatable, match="a move that costs coins"):
        translate(_hunger())
