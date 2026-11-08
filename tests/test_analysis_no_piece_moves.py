"""Q1167: the analysis pass, and the site's walk, read past a move that moves no piece.

A keep (TS3, `MOVE_KEEP`) and a pass (H3, `MOVE_PASS`) are legal moves that move nothing, so
`analysis.mover_kind` answers `None` for them, as it does for a record drifted from its spec.
`_analyse_game` and `site/legality.py::walk` both stopped at that answer, and every later ply
of the game went unread: Hunger's players pass on ply 1, so no Hunger run had an index, and
Tessabi's run 0 was read for 2,027 of its 15,802 plies (`docs/plan-hunger.md` §6). Held here on
two small specs of this file's own, the dull control's grunts with a pass, and with a turn of
two parts whose second is optional.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_KEEP, MOVE_PASS, move_type
from chess_variant_sandbox.site import legality
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.play import play_game

_GRUNTS: dict[str, Any] = {
    "schema_version": 31,
    "name": "grunts",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {
            "id": "G",
            "name": "grunt",
            "value": 1,
            "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
            ],
        }
    ],
    "setup": {"mirror": True, "placements": [{"piece": "G", "squares": ["a1", "c1", "e1"]}]},
    "rules": {"win": "annihilation", "stalemate": "draw", "max_plies": 60},
}


def _passing() -> dict[str, Any]:
    spec = copy.deepcopy(_GRUNTS)
    spec["name"] = "grunts that pass"
    spec["rules"]["pass"] = True
    return spec


def _keeping() -> dict[str, Any]:
    spec = copy.deepcopy(_GRUNTS)
    spec["name"] = "grunts that keep"
    spec["rules"]["turn"] = {"parts": [{"pieces": "own"}, {"pieces": "own", "optional": True}]}
    return spec


SPECS = {"pass": (_passing, MOVE_PASS), "keep": (_keeping, MOVE_KEEP)}


def _games(name: str) -> tuple[Engine, list[Any]]:
    build, kind = SPECS[name]
    spec = build()
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    records = [play_game(engine, ("random", "random"), seed) for seed in range(6)]
    # Each game is here for a move that moves nothing with plies still to read after it.
    early = [
        r for r in records
        if any(move_type(m) == kind for m in r.moves[: max(r.plies - 1, 0)])
    ]
    assert early, f"no game plays a {name} before its last ply"
    return engine, early


@pytest.mark.parametrize("name", list(SPECS))
def test_the_pass_reads_every_ply(name: str) -> None:
    engine, records = _games(name)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    epsilon = analysis_pass._epsilon_for(engine)
    for record in records:
        read = analysis_pass._analyse_game(engine, evaluator, record, epsilon)
        nothing = sum(1 for m in record.moves if move_type(m) in (MOVE_KEEP, MOVE_PASS))
        assert read.walked_plies == record.plies
        assert len(read.position_keys) == record.plies
        # A move that moves nothing counts under no type; every other move under one.
        assert sum(read.moves_by_type) == record.plies - nothing


@pytest.mark.parametrize("name", list(SPECS))
def test_the_site_walk_reads_every_ply_as_the_pass_does(name: str) -> None:
    engine, records = _games(name)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    epsilon = analysis_pass._epsilon_for(engine)
    for index, record in enumerate(records):
        walked = legality.walk(
            engine, evaluator, record, phase=slices.TOURNAMENT, game=index, series=True
        )
        nothing = sum(1 for m in record.moves if move_type(m) in (MOVE_KEEP, MOVE_PASS))
        assert len(walked.positions.position_keys) == record.plies
        assert sum(walked.positions.moves_by_type) == record.plies - nothing
        own = analysis_pass._analyse_game(engine, evaluator, record, epsilon)
        assert walked.positions.legal_counts == own.legal_counts
        assert walked.positions.position_keys == own.position_keys
