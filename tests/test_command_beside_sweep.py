"""SR6f (`docs/plan-schema-next.md` §2.7 row V8, ruled 2026-10-27): a commanded sweep.

**The ruling** (the user's reading, replacing the recommendation): *a commanded sweep moves as
the piece's sweep always does (its lines, modes, and its capture list, by which a piece it may
not capture blocks); which sides it removes follow the command.* The commander's pieces are
never removed by it: one on the line ends the ray and is never a landing. The commanded piece's
own side is removed (swept, or taken on the landing) only under `captures_own`, and then only
types its captures allow, never a royal or a shielded piece; without `captures_own` any piece on
the line ends the ray, so the commanded sweep is a move over an empty line onto an empty square.

**Readings settled here** (generation 95 in `docs/reference/engine-and-spec-record.md`): a
nobody's or wild piece ends the ray as the commander's does (a commanded piece takes only its
owner's side); under `effects: "path"` a commanded sweep strikes and meets nothing (SR4h's
ordinary removal: no link read, no mover change); a blast reads its removals as any sweep's; a
fold admits it as its owner's (SR5c2); its swept pieces go to the commander's hand.

The oracles: `tests/sight_oracle.py`'s commanded list, widened to sweeps (the owner's twin list
with the owner to move, each sweep kept where it removes only the owner's non-royal, unshielded
pieces under `captures_own`), the distinguishing positions below for both seats, and `perft`
against an `apply`-built count. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_NORMAL,
    MOVE_SWEEP,
    move_foreign,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_ninety_five_uses
from tests.sight_oracle import SightOracle, normalise

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
STEP = {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}
# The preacher commands the first enemy piece on each of its rook lines.
PREACH = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "command", "whose": "enemy"}
IDS = ("K", "M", "S", "G", "R")


def _flip(name: str, side: int) -> str:
    return name if side == 0 else f"{name[0]}{WIDTH + 1 - int(name[1:])}"


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return f"{'abcdef'[square % WIDTH]}{square // WIDTH + 1}"


def _spec(
    *, own: bool = True, captures: list[str] | None = None, sweep_range: int | None = None,
    effects: str | None = None, rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Kings, a preacher `M`, a rook-line sweeper `S`, guards `G` and rooks `R` (king steppers);
    ``own`` is the command's `captures_own`."""
    sweep: dict[str, Any] = {"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}
    if sweep_range is not None:
        sweep["sweep_range"] = sweep_range
    if effects is not None:
        sweep["effects"] = effects
    sweeper: dict[str, Any] = {"id": "S", "value": 5, "moves": [sweep]}
    if captures is not None:
        sweeper["captures"] = captures
    spec: dict[str, Any] = {
        "name": "the preacher and the broom (SR6f test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "M", "value": 3, "moves": [KING],
             "sight": [dict(PREACH, captures_own=True) if own else dict(PREACH)]},
            sweeper,
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "R", "value": 4, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "M", "squares": ["c2"]},
            {"piece": "S", "squares": ["f1"]}, {"piece": "G", "squares": ["b2", "d2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80, **(rules or {})},
    }
    return copy.deepcopy(spec)


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _position(
    engine: Engine, side: int, extra: dict[str, tuple[str, int]] | None = None,
    folds: list[str] | None = None,
) -> GameState:
    """``side`` to move, its preacher on c3 commanding the opponent's sweeper on c5 (owners as
    the first player sees it: 0 the commander, 1 the owner), mirrored by rank for the second."""
    placement = {"a1": ("K", 0), "c3": ("M", 0), "a6": ("K", 1), "c5": ("S", 1), **(extra or {})}
    mirrored = {_flip(sq, side): (pid, owner ^ side) for sq, (pid, owner) in placement.items()}
    kwargs: dict[str, Any] = {"folds": [_flip(end, side) for end in folds]} if folds else {}
    state = engine.make_state(mirrored, side_to_move=side, moved=list(mirrored), **kwargs)
    # `make_state` builds no hand; a hand game's position holds its empty ones.
    state = dataclasses.replace(state, hands=engine.initial_state().hands)
    return dataclasses.replace(state, key=engine._compute_hash(state))


def _commanded(engine: Engine, state: GameState, side: int) -> dict[str, tuple[int, tuple[str, ...]]]:
    """The commanded sweeper's moves, by target (as the first player names it): its kind and
    what it sweeps."""
    origin = _sq(_flip("c5", side))
    found: dict[str, tuple[int, tuple[str, ...]]] = {}
    for move in engine.legal_moves(state):
        if move_foreign(move) and move_from(move) == origin and move_type(move) != 9:
            swept = tuple(
                _flip(_name(square), side)
                for square, _ in engine.spec.sweep_victims(state.board, move)
            )
            key = _flip(_name(move_to(move)), side)
            assert key not in found or found[key][1] != swept, "one move per target and set"
            found[key if key not in found else f"{key}'"] = (move_type(move), swept)
    return found


class TestTheValidator:
    def test_a_command_beside_a_sweep_validates(self) -> None:
        for own in (False, True):
            spec = _spec(own=own)
            assert validate_spec(spec) == [], validate_spec(spec)
            assert "V8" not in refused_ids(validate_spec(spec))

    def test_the_inventory_names_it(self) -> None:
        assert _schema_ninety_five_uses(_spec()) == {"pieces.sight.command.sweep"}
        plain = _spec()
        plain["pieces"][2]["moves"][0]["kind"] = "slide"
        assert _schema_ninety_five_uses(plain) == set()

    def test_describe_says_what_a_commanded_sweep_removes(self) -> None:
        text = describe_spec(_spec())
        assert "commanded sweeper sweeps as its owner's would" in text


class TestTheReading:
    @pytest.mark.parametrize("side", [0, 1])
    def test_without_captures_own_it_moves_over_an_empty_line(self, side: int) -> None:
        """Any piece on the line ends the ray: its own (e5), the commander's (b5, and the
        preacher on c3)."""
        engine = _engine(_spec(own=False))
        state = _position(engine, side, {"e5": ("G", 1), "b5": ("G", 0)})
        moves = _commanded(engine, state, side)
        assert set(moves) == {"d5", "c6", "c4"}
        assert all(kind == MOVE_NORMAL and not swept for kind, swept in moves.values())

    @pytest.mark.parametrize("side", [0, 1])
    def test_with_captures_own_it_sweeps_its_own_side(self, side: int) -> None:
        engine = _engine(_spec())
        state = _position(engine, side, {"e5": ("G", 1), "f5": ("R", 1), "b5": ("G", 0)})
        moves = _commanded(engine, state, side)
        assert moves == {
            "d5": (MOVE_NORMAL, ()), "e5": (MOVE_NORMAL, ()), "f5": (MOVE_SWEEP, ("e5",)),
            "c6": (MOVE_NORMAL, ()), "c4": (MOVE_NORMAL, ()),
        }

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_commanders_piece_ends_the_ray(self, side: int) -> None:
        """Its own guard on d5, the commander's on e5: a sweep of d5 stops before e5, and e5 is
        never a landing."""
        engine = _engine(_spec())
        state = _position(engine, side, {"d5": ("G", 1), "e5": ("G", 0)})
        moves = _commanded(engine, state, side)
        assert "d5" in moves and "e5" not in moves and "f5" not in moves

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_royal_ends_the_ray(self, side: int) -> None:
        engine = _engine(_spec())
        state = _position(engine, side, {"e5": ("G", 1)})
        state = engine.make_state(
            {**{_name(sq): (IDS[engine.spec.type_of(v)], engine.spec.owner_of(v))
                for sq, v in enumerate(state.board) if v and _name(sq) != _flip("a6", side)},
             _flip("f5", side): ("K", 1 ^ side)},
            side_to_move=side, moved=[],
        )
        moves = _commanded(engine, state, side)
        assert "e5" in moves and "f5" not in moves

    @pytest.mark.parametrize("side", [0, 1])
    def test_its_capture_list_ends_the_ray(self, side: int) -> None:
        """A sweeper that may not capture a guard: the guard blocks, as for its owner."""
        engine = _engine(_spec(captures=["K", "M", "S", "R"]))
        state = _position(engine, side, {"e5": ("G", 1), "c6": ("R", 1)})
        moves = _commanded(engine, state, side)
        # Not a5: there it would check the commander's royal on a1, so it is no move.
        assert set(moves) == {"d5", "c6", "c4", "b5"}

    @pytest.mark.parametrize("side", [0, 1])
    def test_its_owner_still_sweeps_both_sides(self, side: int) -> None:
        """The owner's own sweep is unchanged: on the owner's turn it sweeps the commander's
        guard on d5 and lands past it, on e5 and on f5."""
        engine = _engine(_spec())
        state = _position(engine, side, {"d5": ("G", 0)})
        state = engine.make_state(
            {_name(sq): (IDS[engine.spec.type_of(v)], engine.spec.owner_of(v))
             for sq, v in enumerate(state.board) if v},
            side_to_move=1 ^ side, moved=[],
        )
        origin = _sq(_flip("c5", side))
        sweeps = {
            _flip(_name(move_to(m)), side) for m in engine.legal_moves(state)
            if move_from(m) == origin and move_type(m) == MOVE_SWEEP
        }
        assert {"e5", "f5"} <= sweeps
        assert not {"e5", "f5"} & set(_commanded(engine, _position(engine, side, {"d5": ("G", 0)}), side))

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_sweep_removes_and_reads(self, side: int) -> None:
        """The commanded sweep to f5 removes its own guard on e5 and writes `~` before its
        text; `find_move` round-trips it."""
        engine = _engine(_spec())
        state = _position(engine, side, {"e5": ("G", 1)})
        move = next(
            m for m in engine.legal_moves(state)
            if move_foreign(m) and move_type(m) == MOVE_SWEEP
            and move_to(m) == _sq(_flip("f5", side))
        )
        text = engine.describe_move(state, move)
        assert text == f"~Sc{_flip('c5', side)[1]}xf{_flip('f5', side)[1]} (sweeps e{_flip('e5', side)[1]})"
        assert engine.find_move(state, text) == move
        after = engine.apply(state, move)
        assert not after.board[_sq(_flip("e5", side))]
        assert engine.spec.owner_of(after.board[_sq(_flip("f5", side))]) == 1 ^ side
        assert after.key == engine._compute_hash(after)


@pytest.mark.parametrize("side", [0, 1])
def test_a_shielded_piece_ends_the_commanded_ray(side: int) -> None:
    """A shield beside a sweep is valid since SR6g (W10); a commanded sweep reads the shield
    off the board: the warden on e6 shields the guard on e5, which ends the commanded ray and
    is never a landing."""
    spec = _spec()
    spec["pieces"].append({"id": "W", "value": 2, "moves": [KING], "sight": [
        {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1,
         "effect": "shield", "whose": "own"}]})
    spec["setup"]["placements"].append({"piece": "W", "squares": ["e1"]})
    engine = _engine(spec)
    unshielded = _position(engine, side, {"e5": ("G", 1), "e3": ("W", 1)})
    assert {"e5", "f5"} <= set(_commanded(engine, unshielded, side))
    shielded = _position(engine, side, {"e5": ("G", 1), "e6": ("W", 1)})
    moves = _commanded(engine, shielded, side)
    assert "d5" in moves and "e5" not in moves and "f5" not in moves


class TestBesideTheOtherSweepRules:
    @pytest.mark.parametrize("side", [0, 1])
    def test_a_path_contact_meets_nothing_it_sweeps(self, side: int) -> None:
        """Under contacts and `effects: "path"` the owner's own sweep over its guard turns it
        into a rook by the own link; a commanded sweep removes it (SR4h's ordinary removal)."""
        contacts = {"position": "stay", "immune": ["K"], "links": [
            {"mover": "S", "onto": "G", "onto_to": "R"},
            {"mover": "S", "onto": "G", "own": True, "onto_to": "R"},
        ]}
        engine = _engine(_spec(effects="path", rules={"contacts": contacts}))
        state = _position(engine, side, {"e5": ("G", 1)})
        move = next(
            m for m in engine.legal_moves(state)
            if move_foreign(m) and move_type(m) == MOVE_SWEEP
            and move_to(m) == _sq(_flip("f5", side))
        )
        assert not engine.apply(state, move).board[_sq(_flip("e5", side))]
        assert "(met" not in engine.describe_move(state, move)
        owners = engine.make_state(
            {_name(sq): (IDS[engine.spec.type_of(v)], engine.spec.owner_of(v))
             for sq, v in enumerate(state.board) if v},
            side_to_move=1 ^ side, moved=[],
        )
        own = next(
            m for m in engine.legal_moves(owners)
            if move_type(m) == MOVE_SWEEP and move_to(m) == _sq(_flip("f5", side))
        )
        after = engine.apply(owners, own).board[_sq(_flip("e5", side))]
        assert IDS[engine.spec.type_of(after)] == "R"

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_fold_admits_it_as_its_owners(self, side: int) -> None:
        """A fold on e5 leading to b2: the commanded sweep over its guard on d5 goes through
        where the fold opens to its owner, and not where it opens to the commander alone."""
        for open_to, expect in (("owner", True), ("commander", False)):
            passage = {str(side): "both_ways" if open_to == "commander" else "closed",
                       str(1 - side): "both_ways" if open_to == "owner" else "closed"}
            group = {"orange": _flip("e5", side), "blue": _flip("b2", side), "kind": "fold",
                     "moves": [STEP], "passage": passage}
            engine = _engine(_spec(rules={"portals": [group]}))
            state = _position(engine, side, {"d5": ("G", 1)}, folds=["e5", "b2"])
            moves = _commanded(engine, state, side)
            assert ("b2" in moves) == expect, (open_to, moves)
            if expect:
                assert moves["b2"] == (MOVE_SWEEP, ("d5",))

    @pytest.mark.parametrize("side", [0, 1])
    def test_two_rays_on_a_cylinder_are_two_moves(self, side: int) -> None:
        """On a cylinder the sweeper reaches f5 both ways round: through its guard on e5, and
        through its guard on b5 (and a5); with the commander's guard on b5 only the first."""
        wrap = {"wrap": {"files": "both"}}
        engine = _engine(_spec(rules=wrap))
        both = _position(engine, side, {"e5": ("G", 1), "b5": ("G", 1)})
        sets = sorted(
            swept for target, (kind, swept) in _commanded(engine, both, side).items()
            if target.rstrip("'") == "f5"
        )
        assert sets == [("b5",), ("e5",)]
        one = _position(engine, side, {"e5": ("G", 1), "b5": ("G", 0)})
        sets = [swept for target, (_, swept) in _commanded(engine, one, side).items()
                if target.rstrip("'") == "f5"]
        assert sets == [("e5",)]

    @pytest.mark.parametrize("side", [0, 1])
    def test_its_swept_pieces_go_to_the_commanders_hand(self, side: int) -> None:
        engine = _engine(_spec(rules={"drops": {"from_captures": True}}))
        state = _position(engine, side, {"e5": ("G", 1)})
        move = next(
            m for m in engine.legal_moves(state)
            if move_foreign(m) and move_type(m) == MOVE_SWEEP
            and move_to(m) == _sq(_flip("f5", side))
        )
        after = engine.apply(state, move)
        assert sum(after.hands[side]) == sum(state.hands[side]) + 1
        assert after.hands[1 - side] == state.hands[1 - side]


def _walking(**kwargs: Any) -> dict[str, Any]:
    """A busier game for the walks: the sweeper on c1 behind the preacher on c2, whose command
    is an aura along its rook lines, so the opponent's sweeper (on c6) is commanded early."""
    spec = _spec(**kwargs)
    spec["pieces"][1]["sight"][0]["through"] = True
    spec["setup"]["placements"] = [
        {"piece": "K", "squares": ["a1"]}, {"piece": "S", "squares": ["c1"]},
        {"piece": "M", "squares": ["c2"]}, {"piece": "G", "squares": ["b2", "d2", "e1"]},
        {"piece": "R", "squares": ["f2"]},
    ]
    return spec


def _games() -> dict[str, dict[str, Any]]:
    return {
        "plain": _walking(own=False),
        "own": _walking(),
        "short": _walking(sweep_range=1),
        "cylinder": _walking(rules={"wrap": {"files": "both"}}),
        "fold": _walking(rules={"portals": [
            {"orange": "c4", "blue": "e3", "kind": "fold", "moves": [STEP]}]}),
        "blast": _walking(rules={"explode_on_capture": {"radius": 1}}),
        "hand": _walking(rules={"drops": {"from_captures": True}}),
    }


def _walk(engine: Engine, seed: int, step: Any = None) -> int:
    """A seeded walk that plays a commanded move half the time one is offered (a commanded
    sweep first where there is one); ``step`` is asked at every ply. The commanded sweeps
    played."""
    rng = random.Random(seed)
    state = engine.initial_state()
    played = 0
    for _ in range(60):
        moves = engine.legal_moves(state)
        if step is not None:
            step(state, moves)
        if engine.terminal(state, moves) is not None:
            break
        sweeps = [m for m in moves if move_foreign(m) and move_type(m) == MOVE_SWEEP]
        commanded = sweeps or [m for m in moves if move_foreign(m)]
        move = rng.choice(commanded) if commanded and rng.random() < 0.5 else rng.choice(moves)
        played += move in sweeps
        state = engine.apply(state, move)
    return played


class TestAgainstTheOracle:
    @pytest.mark.parametrize("game", sorted(_games()))
    @pytest.mark.parametrize("seed", range(3))
    def test_walks_agree_with_the_oracle(self, game: str, seed: int) -> None:
        """Every ply of seeded walks: the core's legal list is the oracle's, and the
        incremental key the full scan's."""
        oracle = SightOracle(_games()[game])
        engine = oracle.engine

        def step(state: GameState, moves: list[int]) -> None:
            assert normalise(moves) == oracle.legal_moves(state)
            assert state.key == engine._compute_hash(state)

        _walk(engine, seed, step)

    def test_the_walks_meet_the_case(self) -> None:
        """The walks above play commanded sweeps in every game where the command takes."""
        played = {game: sum(_walk(Engine(spec), seed) for seed in range(3))
                  for game, spec in _games().items()}
        assert all(count > 0 for game, count in played.items() if game != "plain"), played

    @pytest.mark.parametrize("game", sorted(_games()))
    @pytest.mark.parametrize("side", [0, 1])
    def test_the_probe_positions_agree_with_the_oracle(self, game: str, side: int) -> None:
        oracle = SightOracle(_games()[game])
        folds = ["c4", "d5"] if game == "fold" else None
        extra = {"e5": ("G", 1), "f5": ("R", 1), "b5": ("G", 0), "c6": ("G", 1)}
        if folds:
            extra.pop("e5")
        state = _position(oracle.engine, side, extra, folds=folds)
        assert normalise(oracle.engine.legal_moves(state)) == oracle.legal_moves(state)


class TestUndo:
    @pytest.mark.parametrize("game", ["own", "cylinder", "hand"])
    @pytest.mark.parametrize("side", [0, 1])
    def test_perft_matches_an_apply_count(self, game: str, side: int) -> None:
        engine = _engine(_games()[game])
        state = _position(engine, side, {"e5": ("G", 1), "f4": ("R", 1)})

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)
