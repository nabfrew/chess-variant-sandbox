"""Schema 30 (Q999a): lines of sight with passive effects, one test per decision.

`docs/archive/plan-passive.md` §2 is the rule and §3 its build; this file is the Q999a box's acceptance:
the key and each refusal of §2.1 and §2.8 in words; a ray stopped and an aura not; `min_range`
blocking; a facing sight turning; mutual freezing; the union of sources; a frozen checker no
check and the royal stepping into its line; a check answered by freezing, by a drop and by a
promotion; an unfreezing move illegal; a shielded royal not in check; no lift; castling with a
frozen king, a frozen partner and a frozen attacker of its transit; en passant against a
shielded pawn; disarm against a strike and a contact; `must_capture` after the effects; the
rest's lift not thawing a frozen piece; a round where a frozen piece is offered nothing and a
disarmed piece wins a collision; speed's slower move voided by a freeze; the core's `effects`
and `sight_of` queries; and the bump inert. Every hand-built position is also held to the
declared oracle (`tests/sight_oracle.py`); its random walks are `test_sight_walks.py`'s.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from sight_oracle import SightOracle

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import (
    SIGHT_DISARM,
    SIGHT_FREEZE,
    SIGHT_SHIELD,
    compile_spec,
)
from chess_variant_sandbox.engine.moves import MOVE_CASTLE, MOVE_DROP, MOVE_EN_PASSANT, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import SIGHT_MAX, piece_sights
from tests.feature_inventory import _schema_thirty_uses

HERE = Path(__file__).parent
ROOT = HERE.parent


def load(path: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return loaded


CHESS = load("rulesets/chess.json")
S1 = load("tests/_sight_madrasi.json")
S2 = load("tests/_sight_sheltering_king.json")
S3 = load("tests/_sight_daunting_knights.json")

KING_STEP = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1}
KNIGHT = {"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1}


def warden(*sights: dict[str, Any], captures: bool = True) -> dict[str, Any]:
    """A piece that steps like a king (capturing or not) and carries these sights."""
    step: dict[str, Any] = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
    if not captures:
        step["mode"] = "move"
    return {"id": "W", "name": "warden", "value": 3, "moves": [step], "sight": list(sights)}


FREEZE_BESIDE = dict(KING_STEP, effect="freeze", whose="enemy")


def chess_with(
    *extra: dict[str, Any], sights: dict[str, list[dict[str, Any]]] | None = None,
    base: dict[str, Any] = CHESS, **rules: Any,
) -> dict[str, Any]:
    """``base`` at schema 30 with these extra types (each placed on c3/c6 so both sides hold
    it), these sights, and these rules set (``None`` removes one)."""
    spec = copy.deepcopy(base)
    spec["schema_version"] = 30
    for piece in extra:
        spec["pieces"].append(copy.deepcopy(piece))
        spec["setup"]["placements"].append({"piece": piece["id"], "squares": ["c3"]})
    for pid, written in (sights or {}).items():
        next(p for p in spec["pieces"] if p["id"] == pid)["sight"] = written
    for key, value in rules.items():
        if value is None:
            spec["rules"].pop(key, None)
        else:
            spec["rules"][key] = value
    return spec


@cache
def _engine(text: str) -> Engine:
    return Engine(json.loads(text))


def engine(spec: dict[str, Any]) -> Engine:
    return _engine(json.dumps(spec, sort_keys=True))


@cache
def _oracle(text: str) -> SightOracle:
    return SightOracle(json.loads(text))


def oracle(spec: dict[str, Any]) -> SightOracle:
    return _oracle(json.dumps(spec, sort_keys=True))


def place(
    spec: dict[str, Any], pieces: dict[str, tuple[str, int]], side: int = 0,
    unmoved: tuple[str, ...] = (), **extra: Any,
) -> GameState:
    """A position with these pieces (all moved but ``unmoved``), keyed by the slow hash."""
    eng = engine(spec)
    state = eng.make_state(pieces, side_to_move=side, moved=[s for s in pieces if s not in unmoved],
                           **extra)
    state = state.with_(hands=eng.initial_state().hands)
    key = eng._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def sq(spec: dict[str, Any], name: str) -> int:
    return engine(spec)._square_index(name)


def legal(spec: dict[str, Any], state: GameState) -> list[int]:
    """The core's list, held to the oracle's."""
    moves = engine(spec).legal_moves(state)
    assert moves == oracle(spec).legal_moves(state)
    return moves


def pairs(spec: dict[str, Any], moves: list[int]) -> set[tuple[int, int]]:
    return {decode_move(word)[:2] for word in moves}


def step(spec: dict[str, Any], origin: str, target: str) -> tuple[int, int]:
    return sq(spec, origin), sq(spec, target)


def bits(spec: dict[str, Any], state: GameState, name: str) -> int:
    found = engine(spec).effects(state).bits[sq(spec, name)]
    assert found == oracle(spec).effects(state.board)[sq(spec, name)]
    return found


# ------------------------------------------------------------------ the key and its refusals


def test_the_collector_names_each_effect_and_the_aura() -> None:
    assert _schema_thirty_uses(S1) == {"pieces.sight.freeze"}
    assert _schema_thirty_uses(S2) == {"pieces.sight.shield", "pieces.sight.through"}
    assert _schema_thirty_uses(S3) == {"pieces.sight.disarm"}
    assert _schema_thirty_uses(CHESS) == set()


def test_the_three_showcases_are_valid_and_read() -> None:
    for spec in (S1, S2, S3):
        assert validate_spec(spec) == []
    queen = piece_sights(S1)["Q"][0]
    assert (queen.effect, queen.whose, queen.affects, queen.through, queen.range) == (
        "freeze", "enemy", ("Q",), False, 0,
    )
    assert piece_sights(S2)["K"][0].through


def refused(spec: dict[str, Any], words: str) -> None:
    errors = validate_spec(spec)
    assert any(words in error for error in errors), errors


def refused_pair(spec: dict[str, Any], rid: str) -> None:
    """A refused pair, by its id in `spec/refusals.py`."""
    errors = validate_spec(spec)
    assert rid in refused_ids(errors), errors


def with_sight(*written: Any, pid: str = "Q", **rules: Any) -> dict[str, Any]:
    spec = chess_with(**rules)
    next(p for p in spec["pieces"] if p["id"] == pid)["sight"] = list(written)
    return spec


QUEEN_FREEZE = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "freeze", "whose": "enemy"}


def test_each_spelling_refusal_is_said_in_words() -> None:
    refused(with_sight(*[QUEEN_FREEZE] * (SIGHT_MAX + 1)), f"at most {SIGHT_MAX}")
    refused(with_sight(dict(QUEEN_FREEZE, colour="dark")), "colour")
    refused(with_sight({k: v for k, v in QUEEN_FREEZE.items() if k != "whose"}), "whose")
    refused(with_sight(dict(QUEEN_FREEZE, effect="petrify")), "effect must be one of")
    refused(with_sight(dict(QUEEN_FREEZE, whose="both")), "whose must be one of")
    refused(with_sight(dict(QUEEN_FREEZE, through="yes")), "through must be true or false")
    refused(with_sight(dict(QUEEN_FREEZE, affects=[])), "affects nothing; leave it out")
    refused(with_sight(dict(QUEEN_FREEZE, affects=["Z"])), "is not a declared piece")
    refused(with_sight(dict(QUEEN_FREEZE, range=-1)), "range must be a non-negative integer")
    refused(with_sight(dict(QUEEN_FREEZE, min_range=3, range=2)), "may not exceed range")
    refused(with_sight(dict(QUEEN_FREEZE, vectors=[[0, 0]])), "null vector")
    refused(with_sight(dict(QUEEN_FREEZE, symmetry="rotate6")), "60-degree")
    refused(with_sight([1, 0]), "must be an object")


def test_an_off_value_is_the_sight_left_out() -> None:
    """Q1008c: ``"sight": []`` and ``"through": false`` state nothing, so they are accepted and
    name the game without them."""
    plain = chess_with()
    assert validate_spec(with_sight()) == []
    assert spec_hash(with_sight()) == spec_hash(plain)
    ray = with_sight(QUEEN_FREEZE)
    assert spec_hash(with_sight(dict(QUEEN_FREEZE, through=False))) == spec_hash(ray)
    assert validate_spec(with_sight(dict(QUEEN_FREEZE, through=False))) == validate_spec(ray)


def test_a_sight_is_refused_beside_portals_and_explosions() -> None:
    atomic = load("rulesets/atomic.json")
    refused_pair(chess_with(base=atomic, sights={"Q": [QUEEN_FREEZE]}), "V2")
    wormhole = load("archive/wormhole.json")
    pid = next(p["id"] for p in wormhole["pieces"] if p.get("moves"))
    refused_pair(chess_with(base=wormhole, sights={pid: [QUEEN_FREEZE]}), "V1")


def test_a_shield_beside_wild_pieces_and_a_wild_piece_seeing_both_sides() -> None:
    """Generation 37 (N2, `docs/archive/plan-npc.md` §4.2) lifted the shield's refusal beside wild
    pieces and let a wild piece see, with ``whose`` ``"any"`` only (`tests/test_wild_sight.py`
    holds the rest)."""
    wolves = load("archive/wolves.json")
    wild = {npc["piece"] for npc in wolves["rules"]["npcs"]}
    tame = next(p["id"] for p in wolves["pieces"] if p["id"] not in wild and p.get("moves"))
    shield = dict(KING_STEP, effect="shield", whose="own")
    assert validate_spec(chess_with(base=wolves, sights={tame: [shield]})) == []
    refused(chess_with(base=wolves, sights={next(iter(wild)): [FREEZE_BESIDE]}),
            "which has no side; use 'any'")
    assert validate_spec(chess_with(
        base=wolves, sights={next(iter(wild)): [dict(FREEZE_BESIDE, whose="any")]}
    )) == []
    freeze_tame = chess_with(base=wolves, sights={tame: [FREEZE_BESIDE]})
    assert not any("sight" in error for error in validate_spec(freeze_tame))
    refused(chess_with(base=wolves, sights={tame: [dict(FREEZE_BESIDE, affects=sorted(wild))]}),
            "a wild piece (rules.npcs), which nothing affects")


def test_a_shield_is_refused_beside_a_sweep() -> None:
    spec = chess_with(sights={"K": [dict(KING_STEP, effect="shield", whose="own")]})
    rook = next(p for p in spec["pieces"] if p["id"] == "R")
    rook["moves"].append({"kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4"})
    refused_pair(spec, "W10")


def _small(**sight: Any) -> dict[str, Any]:
    """A 4x4 game whose sentry ``S``, carrying the sight, only the first player holds, and whose
    guard ``G`` only the second player holds and nothing can capture."""
    king_step = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
    return {
        "schema_version": 30, "name": "small", "board": {"width": 4, "height": 4},
        "pieces": [
            {"id": "K", "royal": True, "moves": [king_step], "captures": ["P"]},
            {"id": "S", "moves": [king_step], "captures": ["P"], "sight": [sight]},
            {"id": "P", "moves": [king_step], "captures": ["K", "P"]},
            {"id": "C", "moves": [king_step], "captures": []},
            {"id": "G", "moves": [dict(king_step, mode="move")]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "P", "player": 0, "squares": ["b1"]},
            {"piece": "C", "player": 0, "squares": ["c1"]},
            {"piece": "S", "player": 0, "squares": ["d1"]},
            {"piece": "K", "player": 1, "squares": ["d4"]},
            {"piece": "P", "player": 1, "squares": ["c4"]},
            {"piece": "C", "player": 1, "squares": ["b4"]},
            {"piece": "G", "player": 1, "squares": ["a4"]},
        ]},
        "rules": {"win": "royal_capture"},
    }


def test_a_sight_that_can_never_bind_is_refused() -> None:
    reach = dict(KING_STEP)
    assert validate_spec(_small(**reach, effect="freeze", whose="enemy")) == []
    refused(_small(**reach, effect="freeze", whose="own", affects=["G"]),
            "no type it affects is ever held by the side it affects")
    refused(_small(**reach, effect="disarm", whose="enemy", affects=["C", "G"]),
            "a disarm on types that can capture nothing")
    refused(_small(**reach, effect="shield", whose="any", affects=["G"]),
            "a shield on types that nothing can capture")
    assert validate_spec(_small(**reach, effect="shield", whose="own", affects=["P"])) == []


# --------------------------------------------------------------------- what a sight sees


ROOK_SEES = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "freeze", "whose": "enemy"}


def test_a_ray_stops_at_the_first_piece_and_an_aura_sees_past_it() -> None:
    ray = chess_with(sights={"R": [ROOK_SEES]})
    aura = chess_with(sights={"R": [dict(ROOK_SEES, through=True)]})
    pieces = {"a1": ("R", 0), "a3": ("N", 1), "a5": ("B", 1), "h1": ("K", 0), "h8": ("K", 1)}
    for spec, far in ((ray, 0), (aura, SIGHT_FREEZE)):
        state = place(spec, pieces)
        assert bits(spec, state, "a3") == SIGHT_FREEZE
        assert bits(spec, state, "a5") == far
        legal(spec, place(spec, pieces, side=1))
    seen = dict(engine(ray).sight_of(place(ray, pieces), sq(ray, "a1")))
    assert sq(ray, "a3") in seen and sq(ray, "a4") not in seen and sq(ray, "b1") in seen


def test_a_square_nearer_than_min_range_is_unseen_but_stops_a_ray() -> None:
    spec = chess_with(sights={"R": [dict(ROOK_SEES, min_range=2, range=3)]})
    blocked = {"a1": ("R", 0), "a2": ("N", 1), "a3": ("B", 1), "h1": ("K", 0), "h8": ("K", 1)}
    state = place(spec, blocked)
    assert bits(spec, state, "a2") == 0 and bits(spec, state, "a3") == 0
    open_line = {k: v for k, v in blocked.items() if k != "a2"}
    assert bits(spec, place(spec, open_line), "a3") == SIGHT_FREEZE


def test_a_facing_piece_looks_the_way_it_faces() -> None:
    lookout = {
        "id": "L", "name": "lookout", "value": 2,
        "moves": [{"kind": "leap", "vectors": [[0, 1]]}],
        "facing": {"axes": 4},
        "sight": [{"vectors": [[0, 1]], "range": 1, "effect": "freeze", "whose": "enemy"}],
    }
    spec = chess_with(lookout)
    state = place(spec, {"c4": ("L", 0), "c5": ("N", 1), "d4": ("N", 1),
                         "h1": ("K", 0), "h8": ("K", 1)})
    assert bits(spec, state, "c5") == SIGHT_FREEZE and bits(spec, state, "d4") == 0
    eng = engine(spec)
    turned = None
    for word in legal(spec, state):
        origin, _, kind, facing = decode_move(word)
        if origin == sq(spec, "c4") and kind != 0 and facing == 1:
            turned = eng.apply(state, word)
    assert turned is not None
    assert bits(spec, turned, "c5") == 0 and bits(spec, turned, "d4") == SIGHT_FREEZE


def test_two_queens_on_one_open_line_freeze_each_other() -> None:
    state = place(S1, {"d4": ("Q", 0), "d7": ("Q", 1), "a1": ("K", 0), "h8": ("K", 1)})
    assert bits(S1, state, "d4") == SIGHT_FREEZE == bits(S1, state, "d7")
    assert sq(S1, "d4") not in {decode_move(w)[0] for w in legal(S1, state)}


def test_two_sources_of_one_effect_are_one_and_effects_combine() -> None:
    spec = chess_with(sights={"R": [ROOK_SEES], "K": [dict(KING_STEP, effect="shield", whose="own")]})
    both = {"a1": ("R", 0), "h4": ("R", 0), "a4": ("N", 1), "b5": ("K", 1), "h1": ("K", 0)}
    assert bits(spec, place(spec, both), "a4") == SIGHT_FREEZE | SIGHT_SHIELD
    one = {k: v for k, v in both.items() if k != "h4"}
    assert bits(spec, place(spec, one), "a4") == SIGHT_FREEZE | SIGHT_SHIELD
    alone = {k: v for k, v in one.items() if k != "b5"} | {"h8": ("K", 1)}
    assert bits(spec, place(spec, alone), "a4") == SIGHT_FREEZE


# ------------------------------------------------------------------- attack and check

WARD = chess_with(warden(FREEZE_BESIDE, captures=False))


def test_a_frozen_checker_gives_no_check_and_the_royal_may_step_into_its_line() -> None:
    state = place(WARD, {"e1": ("K", 0), "h3": ("W", 0), "h4": ("Q", 1), "a8": ("K", 1)})
    eng = engine(WARD)
    assert not eng.in_check(state)
    assert eng.effects(state).suppressed
    moves = pairs(WARD, legal(WARD, state))
    assert step(WARD, "e1", "f2") in moves
    # The warden may stay beside the queen; stepping away would thaw it onto the king.
    assert step(WARD, "h3", "g4") in moves
    assert step(WARD, "h3", "g3") in moves
    assert step(WARD, "h3", "h2") not in moves


def test_a_check_is_answered_by_freezing_the_checker() -> None:
    state = place(WARD, {"e1": ("K", 0), "c6": ("W", 0), "e8": ("R", 1), "h8": ("K", 1)})
    assert engine(WARD).in_check(state)
    moves = pairs(WARD, legal(WARD, state))
    assert step(WARD, "c6", "d7") in moves
    assert step(WARD, "c6", "c7") not in moves


def test_a_check_is_answered_by_a_drop_that_freezes() -> None:
    spec = chess_with(warden(FREEZE_BESIDE, captures=False), drops={"from_captures": True})
    state = place(spec, {"e1": ("K", 0), "e8": ("R", 1), "h8": ("K", 1), "a1": ("R", 0)})
    hand = list(state.hands[0])
    hand[engine(spec).spec.piece_ids.index("W")] = 1
    state = state.with_(hands=(tuple(hand), state.hands[1]))
    drops = {decode_move(w)[1] for w in legal(spec, state) if decode_move(w)[2] == MOVE_DROP}
    assert sq(spec, "d7") in drops and sq(spec, "f7") in drops  # freezes, does not block
    assert sq(spec, "e4") in drops  # blocks
    assert sq(spec, "c7") not in drops  # neither


def test_a_check_is_answered_by_a_promotion_that_freezes() -> None:
    state = place(S1, {"h1": ("K", 0), "a7": ("P", 0), "a1": ("R", 1), "h6": ("K", 1)})
    eng = engine(S1)
    assert eng.in_check(state)
    promoted = {decode_move(w)[3] for w in legal(S1, state) if decode_move(w)[0] == sq(S1, "a7")}
    assert promoted == {eng.spec.piece_ids.index("R")}


def test_a_shielded_royal_is_not_in_check_and_may_not_be_bared() -> None:
    spec = chess_with(warden(dict(KING_STEP, range=2, through=True, effect="shield",
                                  whose="own", affects=["K"]), captures=False))
    state = place(spec, {"e1": ("K", 0), "c1": ("W", 0), "e8": ("R", 1), "h8": ("K", 1)})
    eng = engine(spec)
    assert not eng.in_check(state)
    assert eng.effects(state).suppressed
    moves = pairs(spec, legal(spec, state))
    assert step(spec, "c1", "d1") in moves
    assert step(spec, "c1", "b1") not in moves
    assert step(spec, "c1", "c2") not in moves


def test_a_total_freeze_is_stalemate_with_no_lift() -> None:
    for result, winner in (("draw", None), ("loss", 1)):
        spec = chess_with(warden(FREEZE_BESIDE, captures=False), stalemate=result)
        state = place(spec, {"a1": ("K", 0), "b2": ("W", 1), "h8": ("K", 1)})
        assert legal(spec, state) == []
        outcome = engine(spec).terminal(state)
        assert outcome is not None and outcome.winner == winner
        assert "stalemate" in outcome.reason


# -------------------------------------------------------------- castling and en passant


def castlings(spec: dict[str, Any], state: GameState) -> set[int]:
    return {decode_move(w)[1] for w in legal(spec, state) if decode_move(w)[2] == MOVE_CASTLE}


def test_castling_reads_the_effects() -> None:
    base = {"e1": ("K", 0), "a1": ("R", 0), "h1": ("R", 0), "e8": ("K", 1)}
    unmoved = ("e1", "a1", "h1")
    assert castlings(WARD, place(WARD, base, unmoved=unmoved)) == {sq(WARD, "c1"), sq(WARD, "g1")}
    frozen_king = place(WARD, base | {"e2": ("W", 1)}, unmoved=unmoved)
    assert castlings(WARD, frozen_king) == set()
    frozen_rook = place(WARD, base | {"h2": ("W", 1)}, unmoved=unmoved)
    assert castlings(WARD, frozen_rook) == {sq(WARD, "c1")}
    attacked = place(WARD, base | {"f8": ("R", 1), "e8": ("K", 1), "a8": ("K", 1)} | {}, unmoved=unmoved)
    del attacked
    transit = base | {"f8": ("R", 1)}
    transit.pop("e8")
    transit["a8"] = ("K", 1)
    assert castlings(WARD, place(WARD, transit, unmoved=unmoved)) == {sq(WARD, "c1")}
    thawed = place(WARD, transit | {"f7": ("W", 0)}, unmoved=unmoved)
    assert castlings(WARD, thawed) == {sq(WARD, "c1"), sq(WARD, "g1")}


def test_a_shielded_pawn_cannot_be_taken_in_passing() -> None:
    spec = chess_with(warden(dict(KING_STEP, effect="shield", whose="own", affects=["P"]),
                             captures=False))
    pieces = {"e5": ("P", 0), "d5": ("P", 1), "e1": ("K", 0), "e8": ("K", 1)}
    window = {"ep_target": "d6", "ep_victim": "d5"}

    def passing(state: GameState) -> bool:
        return any(decode_move(w)[2] == MOVE_EN_PASSANT for w in legal(spec, state))

    assert passing(place(spec, pieces, **window))
    assert not passing(place(spec, pieces | {"c6": ("W", 1)}, **window))


# ------------------------------------------------------------- disarm and the capture forms


def test_a_disarmed_piece_makes_no_strike() -> None:
    circe = chess_with(sights={"N": [dict(KNIGHT, effect="disarm", whose="enemy")]},
                       strike={"effect": "capture", "position": "home", "immune": ["K"]})
    pieces = {"a1": ("R", 0), "a5": ("P", 1), "e1": ("K", 0), "e8": ("K", 1)}
    assert step(circe, "a1", "a5") in pairs(circe, legal(circe, place(circe, pieces)))
    disarmed = place(circe, pieces | {"b3": ("N", 1)})
    assert step(circe, "a1", "a5") not in pairs(circe, legal(circe, disarmed))
    assert step(circe, "a1", "a4") in pairs(circe, legal(circe, disarmed))


def test_a_disarmed_piece_makes_no_contact() -> None:
    knockdown = load("archive/pawn-knockdown.json")
    spec = chess_with(base=knockdown, sights={"N": [dict(KNIGHT, effect="disarm", whose="enemy")]})
    pieces = {"d4": ("P", 0), "c5": ("Q", 1), "e1": ("K", 0), "e8": ("K", 1)}
    contact = step(spec, "d4", "c5")
    assert contact in pairs(spec, engine(spec).legal_moves(place(spec, pieces)))
    disarmed = place(spec, pieces | {"e6": ("N", 1)})
    moves = pairs(spec, engine(spec).legal_moves(disarmed))
    assert contact not in moves and step(spec, "d4", "d5") in moves


def test_must_capture_applies_after_the_effects() -> None:
    spec = chess_with(sights={"N": [dict(KNIGHT, effect="disarm", whose="enemy")]},
                      must_capture=True)
    pieces = {"a1": ("R", 0), "a5": ("P", 1), "h1": ("K", 0), "h8": ("K", 1)}
    assert pairs(spec, legal(spec, place(spec, pieces))) == {step(spec, "a1", "a5")}
    disarmed = pairs(spec, legal(spec, place(spec, pieces | {"b3": ("N", 1)})))
    assert step(spec, "a1", "a2") in disarmed and step(spec, "a1", "a5") not in disarmed


# ----------------------------------------------------------- the rest, rounds and speed


def test_the_rests_lift_never_thaws_a_frozen_piece() -> None:
    rest = load("tests/_rest_checkmate.json")
    spec = chess_with(warden(FREEZE_BESIDE, captures=False), base=rest)
    state = place(spec, {"a1": ("K", 0), "d3": ("R", 0), "b2": ("W", 1), "e5": ("K", 1)})
    resting = state.with_(rest=(((sq(spec, "d3"), 1),), ()))
    moves = legal(spec, resting)
    assert moves and {decode_move(w)[0] for w in moves} == {sq(spec, "d3")}


ROUNDS = load("tests/_sight_rounds.json")


def test_a_frozen_piece_is_offered_nothing_in_a_round() -> None:
    state = place(ROUNDS, {"b1": ("N", 0), "d3": ("B", 1), "e1": ("K", 0), "e5": ("K", 1)})
    offered = engine(ROUNDS).legal_moves_for(state, 0)
    assert offered == oracle(ROUNDS).legal_moves_for(state, 0)
    assert sq(ROUNDS, "b1") not in {decode_move(w)[0] for w in offered}
    assert offered


def test_a_disarmed_piece_still_wins_a_collision() -> None:
    spec = copy.deepcopy(ROUNDS)
    next(p for p in spec["pieces"] if p["id"] == "P")["beats"] = []
    state = place(spec, {"a3": ("R", 0), "b5": ("N", 1), "a5": ("Q", 1), "c4": ("P", 1),
                         "e1": ("K", 0), "e4": ("K", 1)})
    eng = engine(spec)
    mine = pairs(spec, eng.legal_moves_for(state, 0))
    assert step(spec, "a3", "c3") in mine and step(spec, "a3", "a5") not in mine
    rook = next(w for w in eng.legal_moves_for(state, 0) if decode_move(w)[:2] == step(spec, "a3", "c3"))
    pawn = next(w for w in eng.legal_moves_for(state, 1) if decode_move(w)[:2] == step(spec, "c4", "c3"))
    after = eng.apply_round(state, rook, pawn)
    assert after.board[sq(spec, "c3")] == eng.spec.encode_piece(0, eng.spec.piece_ids.index("R"))


def test_a_slower_move_the_faster_one_froze_is_void() -> None:
    spec = load("tests/_sight_speed.json")
    state = place(spec, {"c1": ("K", 0), "b3": ("R", 1), "e5": ("K", 1)})
    eng = engine(spec)
    king = next(w for w in eng.legal_moves_for(state, 0) if decode_move(w)[:2] == step(spec, "c1", "c2"))
    rook = next(w for w in eng.legal_moves_for(state, 1) if decode_move(w)[:2] == step(spec, "b3", "b5"))
    after, _, events = eng.apply_round_with_phase(state, king, rook)
    assert after.board[sq(spec, "b3")] == state.board[sq(spec, "b3")]
    assert after.board[sq(spec, "b5")] == 0
    assert any(event == 1 and player == 1 for event, _, player in events)


# ------------------------------------------------------------- the core's queries, inertness


def test_effects_counts_the_moves_removed_by_reason() -> None:
    state = place(S3, {"a1": ("R", 0), "a5": ("P", 1), "b3": ("N", 1), "h1": ("K", 0), "h8": ("K", 1)})
    report = engine(S3).effects(state)
    assert report.bits[sq(S3, "a1")] == SIGHT_DISARM
    assert (report.frozen, report.disarmed, report.shielded) == (0, 1, 0)
    assert report.binds and not report.suppressed
    assert engine(CHESS).effects(engine(CHESS).initial_state()).bits == (0,) * 64
    assert engine(CHESS).sight_of(engine(CHESS).initial_state(), 3) == []


def test_a_game_without_a_sight_compiles_and_crosses_as_before() -> None:
    compiled = compile_spec(CHESS)
    assert compiled.sight is None and compiled.hash == "4c1922ffae26"
    tables = build_tables(compiled)
    assert all(tables[key] == [] for key in ("sight_index", "sight_srcs", "sight_paths", "sight_flags"))
    assert not any(key.startswith("sight") for key in canonical_tables(compiled))


@pytest.mark.parametrize("spec", [S1, S2, S3], ids=["S1", "S2", "S3"])
def test_the_flags_say_where_a_sight_bears_on_check(spec: dict[str, Any]) -> None:
    """A freeze or a disarm always bears on check; S2's king shields its own pieces, which a
    king never is (a piece never sees itself, and a side holds one king), so its pin shortcut
    stays on."""
    sight = compile_spec(spec).sight
    assert sight is not None
    assert sight.bears_on_check == (spec is not S2)
    assert sight.has_shield == (spec is S2) != sight.has_restraint


def test_a_shield_on_another_royal_bears_on_check() -> None:
    shield_king = dict(KING_STEP, range=2, through=True, effect="shield", whose="own",
                       affects=["K"])
    sight = compile_spec(chess_with(warden(shield_king, captures=False))).sight
    assert sight is not None and sight.bears_on_check and not sight.has_restraint


def test_fairy_stockfish_refuses_a_sight_and_verify_reads_the_twin() -> None:
    """§4.5: refused with its rule, read off the compiled rule; the sight-stripped twin is
    checked as its own game (chess, here), which plays the same game in Fairy-Stockfish."""
    pytest.importorskip("pyffish")
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate
    from chess_variant_sandbox.fairy.verify import PyffishBackend, verify

    with pytest.raises(Untranslatable) as refused:
        translate(S1)
    assert any(reason.startswith("pieces[].sight") for reason in refused.value.reasons)
    twin = copy.deepcopy(S1)
    for piece in twin["pieces"]:
        piece.pop("sight", None)
    translate(twin)
    result = verify(twin, PyffishBackend(), "", depth=2, games=1, plies=20)
    assert result.ok, result.summary()


def test_the_play_page_is_sent_the_effects_only_in_a_game_with_a_sight() -> None:
    """§4.6's data path: the position block carries the core's effect bits for the page's
    marks, and a game without a sight is sent the block it always was."""
    from chess_variant_sandbox.web.library import Game
    from chess_variant_sandbox.web.play import Sessions

    def session(spec: dict[str, Any]) -> Any:
        game = Game(hash="sight", name=spec["name"], description="", path=Path(), source="test",
                    spec=spec)
        return Sessions().start(game, ("human", "human"), seed=0, scratch=True)

    assert "effects" not in session(CHESS).snapshot()
    played = session(S1)
    eng = played.engine
    for text in ("Pe2-e4", "Pd7-d5"):
        played.state = eng.apply(played.state, eng.find_move(played.state, text))
    sent = played.snapshot()["effects"]
    assert sent == list(eng.effects(played.state).bits)
    assert sent[sq(S1, "e4")] == SIGHT_FREEZE == sent[sq(S1, "d5")]
