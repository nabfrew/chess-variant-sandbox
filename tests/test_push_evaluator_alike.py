"""Q1221: in a push game whose spec restricts no capture, the core's evaluator reads what the
oracle's reads.

The bridge sets `has_capture_matrix` in every push game (Q1074b2) so that the attack tests
read a push source by the victim's type; the matrix it switches on there is all true. The
core's evaluator read that flag as "the spec declares a capture matrix" and so turned on the
relational material and the immunity term (Q466), which `agents/evaluation.py` leaves off
(`CompiledSpec.has_capture_matrix` is false). On dohyo nothing captures (a shove is no
capture), so every piece read immune and a piece thrown out of the ring was priced twice:
the two players agreed until the material came apart, then parted (Q1221; the finder saw it
at ply 8 of seed 3, `minimax:nodes=3000` against `minimax:nodes=2000`, and read it as the
`threats` chase's, but `threats=0` parts too at other seeds).
"""

from __future__ import annotations

import json
from pathlib import Path

from chess_variant_sandbox.agents.base import build_agent
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.analysis import _root_scores, _score_roots
from chess_variant_sandbox.tournament.play import choose_move, seat_rng

ROOT = Path(__file__).resolve().parents[1]

# Seed 3's opening on dohyo (both players `minimax`, 3,000 and 2,000 nodes), then a line of
# shoves out of the 3,000-node search: player 0 has thrown one of player 1's pieces out, so
# the material is unequal and the immunity term, where it is wrongly on, is not zero.
OPENING = [4619, 369894700, 6418, 369892901, 4357, 8488, 7953, 9517]
SHOVES = [202121503, 369891102, 135016225]


def _dohyo(plies: int = 16) -> Engine:
    spec = json.loads((ROOT / "rulesets/dohyo.json").read_text(encoding="utf-8"))
    spec["rules"]["max_plies"] = plies
    return Engine(spec)


def _play(engine: Engine, moves: list[int]):
    state = engine.initial_state()
    for move in moves:
        state = engine.apply(state, move)
    return state


def test_the_spec_restricts_no_capture_but_the_core_reads_the_victims_type() -> None:
    """The premise: the oracle's flag is off, the boundary's on."""
    engine = _dohyo()
    assert engine.spec.push is not None
    assert not engine.spec.has_capture_matrix
    assert all(all(row) for row in engine.spec.capture_allowed)


def test_the_probe_scores_alike_where_the_material_is_unequal() -> None:
    """The evaluator itself, through the analysis probe (which reads the immunity term at the
    rated players' weight): every root move's score, the core's against the oracle's."""
    engine = _dohyo()
    state = _play(engine, OPENING + SHOVES)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    legal = engine.legal_moves(state)
    assert _score_roots(engine, evaluator, state, legal) == _root_scores(
        engine, evaluator, state, legal
    )


def test_the_finders_position_plays_alike() -> None:
    """The reproduction itself: after seed 3's opening, at 3,000 nodes, seed 18 (and with
    `threats=0`, seed 3): before the fix the core finished a third ply the oracle could not
    afford, its quiescence standing pat on 1.25 units the oracle never counted."""
    engine = _dohyo()
    state = _play(engine, OPENING)
    legal = engine.legal_moves(state)
    for agent, seed in (("minimax:nodes=3000", 18), ("minimax:nodes=3000,threats=0", 3)):
        native = choose_move(engine, agent, state, seed=seed)
        oracle = build_agent(agent, engine).select_move(state, legal, seat_rng(seed, 0))
        assert native == oracle, (agent, seed)
