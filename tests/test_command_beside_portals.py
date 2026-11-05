"""SR5c2 (`docs/plan-schema-next.md` §2.7 row V1, its command arm): a command beside portals.

SR5c lifted V1 for freeze, disarm and shield and kept a command beside portals refused: no row
said whether a commanded piece passing a portal end or a fold answers to its commander's `enter`
and `carries` or to its owner's. **The reading (SR5c2, settled on the plan's principles): a
passage is a fact about the piece, as an arrival is (V3, V4).** A commanded piece goes through
a momentum end or a fold exactly where its owner's piece would: the owner's `enter` (a pair's
`passage`) and `carries` decide whether it may go in, its exit is read under the owner's wrap,
and it lands at the exit only where a commanded piece may land (an empty square, or under
`captures_own` a non-royal piece of its own side). A held piece nobody owns passes as its
holder's (P10), so the one rule is `Game::passage_player`: the piece's owner, the mover where
nobody owns it -- `Game::arrival_owner`'s answer.

**The probe** (fed past the validator with V1's command arm gone and nothing else, on SR5d's
build `7e91fbcb2f935e3a`): a commanded piece transited a momentum end by the *commander's*
passage (`push_transits` read the side to move), and the fold copy of movegen
(`generate_folded`) listed no commanded move at all (it walked only held neutrals).

The oracles: `tests/sight_oracle.py`'s commanded list, widened to portals (the owner's twin
list with the owner to move, which is the reading itself), and the probe's distinguishing
positions below, each for both seats. Undo through `perft` against an `apply`-built count.
Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD
from chess_variant_sandbox.engine.moves import (
    MOVE_FOREIGN,
    MOVE_PORTAL,
    move_fold,
    move_foreign,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_eighty_two_uses
from tests.sight_oracle import SightOracle, normalise

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
STEP = {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}
# The preacher commands the first enemy piece on each of its rook lines.
PREACH = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "command", "whose": "enemy"}

# As the first player commands (the second player's is mirrored by rank): the commanded rook on
# c5 slides east into the end on e5 and comes out of a3 at b3 (momentum), or goes through the
# fold on e5 to b4. The guard's square in each game is that exit.
FAR = {"momentum": "a3", "fold": "b4"}
EXIT = {"momentum": "b3", "fold": "b4"}


def _flip(name: str, side: int) -> str:
    return name if side == 0 else f"{name[0]}{WIDTH + 1 - int(name[1:])}"


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(
    kind: str, side: int, *, open_to: str = "owner", carries: list[str] | None = None,
    own: bool = False, wrap: str | None = None, group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Kings, a preacher `M`, rooks and guards `G`; one portal group of ``kind``, seen from
    ``side`` commanding. ``open_to`` is whose pieces may go in (``owner``, ``commander`` or
    ``both``), ``carries`` the entered end's list, ``own`` the command's `captures_own`, ``wrap``
    whose files wrap (``owner`` or ``commander``)."""
    owner = 1 - side
    passage = {
        str(side): "both_ways" if open_to in ("commander", "both") else "closed",
        str(owner): "both_ways" if open_to in ("owner", "both") else "closed",
    }
    if group is None:
        group = {"orange": _flip("e5", side), "blue": _flip(FAR[kind], side)}
    group = {**group, "moves": [STEP], "passage": passage}
    if kind == "fold":
        group["kind"] = "fold"
    if carries is not None:
        group["carries"] = {"orange": carries}
    spec: dict[str, Any] = {
        "name": "the preachers at the portals (SR5c2 test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "M", "value": 3, "moves": [KING],
             "sight": [dict(PREACH, captures_own=True) if own else dict(PREACH)]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "G", "value": 2, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "M", "squares": ["c2"]},
            {"piece": "R", "squares": ["f2"]}, {"piece": "G", "squares": ["b2", "d2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80, "portals": [group]},
    }
    if wrap is not None:
        wrapper = side if wrap == "commander" else owner
        spec["rules"]["wrap"] = {"files_by_player": {
            str(wrapper): "both", str(1 - wrapper): "none"}}
    return copy.deepcopy(spec)


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _position(
    engine: Engine, kind: str, side: int, extra: dict[str, tuple[str, int]] | None = None,
    ends: tuple[str, str] | None = None,
) -> GameState:
    """``side`` to move, its preacher on c3 commanding the opponent's rook on c5 (owners as the
    first player sees it: 0 the commander, 1 the owner), mirrored by rank for the second."""
    placement = {"a1": ("K", 0), "c3": ("M", 0), "e6": ("K", 1), "c5": ("R", 1), **(extra or {})}
    ends = ends or ("e5", FAR[kind])
    if kind == "momentum":
        placement |= {ends[0]: ("1o", WILD), ends[1]: ("1b", WILD)}
    mirrored = {
        _flip(sq, side): (pid, owner if owner == WILD else owner ^ side)
        for sq, (pid, owner) in placement.items()
    }
    folds = [_flip(end, side) for end in ends] if kind == "fold" else None
    return engine.make_state(
        mirrored, side_to_move=side, moved=list(mirrored), **({"folds": folds} if folds else {}),
    )


def _passes(engine: Engine, state: GameState) -> set[int]:
    """The commanded moves a portal makes: transits and go-throughs."""
    return {
        m for m in engine.legal_moves(state)
        if move_foreign(m) and (move_type(m) == MOVE_PORTAL or move_fold(m))
    }


def _targets(moves: set[int]) -> set[int]:
    return {move_to(m) for m in moves}


KINDS = ["momentum", "fold"]


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("kind", KINDS)
    def test_a_command_beside_portals_validates(self, kind: str) -> None:
        spec = _spec(kind, 0)
        assert validate_spec(spec) == []
        assert "V1" not in refused_ids(validate_spec(spec))

    def test_the_inventory_names_it(self) -> None:
        assert _schema_eighty_two_uses(_spec("fold", 0)) == {"pieces.sight.command.portals"}
        frozen = _spec("fold", 0)
        frozen["pieces"][1]["sight"][0]["effect"] = "freeze"
        assert _schema_eighty_two_uses(frozen) == set()
        bare = _spec("fold", 0)
        del bare["rules"]["portals"]
        assert _schema_eighty_two_uses(bare) == set()

    def test_describe_says_whose_passage_it_is(self) -> None:
        assert "A commanded piece goes through a portal as its owner's piece would" in (
            describe_spec(_spec("momentum", 0)))
        bare = _spec("momentum", 0)
        del bare["rules"]["portals"]
        assert "goes through a portal as its owner's" not in describe_spec(bare)

    def test_fairy_stockfish_refuses_the_command(self) -> None:
        with pytest.raises(Untranslatable, match="sight"):
            translate(_spec("momentum", 0))


# --------------------------------------------------------------------------- the reading


class TestThePassageIsTheOwners:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("open_to", ["owner", "commander", "both"])
    def test_the_owners_passage_decides(self, kind: str, side: int, open_to: str) -> None:
        """The probe's finding (momentum): the commander's passage was read. A portal open only
        to the commander's pieces carries no commanded piece; one open only to the owner's does."""
        engine = _engine(_spec(kind, side, open_to=open_to))
        state = _position(engine, kind, side)
        expected = {_sq(_flip(EXIT[kind], side))} if open_to != "commander" else set()
        passes = _passes(engine, state)
        assert _targets(passes) == expected, engine.move_texts(state)
        assert {move_from(m) for m in passes} <= {_sq(_flip("c5", side))}

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize(
        ("open_to", "carries", "passes"),
        [("owner", ["R"], True), ("owner", ["G"], False), ("commander", ["R"], False),
         ("both", ["G", "M"], False), ("both", ["R", "G"], True)],
    )
    def test_carries_reads_the_piece_beside_the_owners_passage(
        self, kind: str, side: int, open_to: str, carries: list[str], passes: bool
    ) -> None:
        engine = _engine(_spec(kind, side, open_to=open_to, carries=carries))
        state = _position(engine, kind, side)
        assert bool(_passes(engine, state)) is passes, engine.move_texts(state)

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("wrap", ["owner", "commander"])
    def test_the_exit_is_read_under_the_owners_wrap(self, side: int, wrap: str) -> None:
        """The rook on c5 slides west into the end on a5 and comes out of a3 westward: across
        the seam onto f3 where its owner's files wrap, nowhere where only the commander's do.
        Under its owner's wrap it also slides east round the seam into a5 and out at b3."""
        group = {"orange": _flip("a5", side), "blue": _flip("a3", side)}
        engine = _engine(_spec("momentum", side, open_to="both", wrap=wrap, group=group))
        state = _position(engine, "momentum", side, ends=("a5", "a3"))
        expected = {_sq(_flip("f3", side)), _sq(_flip("b3", side))} if wrap == "owner" else set()
        assert _targets(_passes(engine, state)) == expected, engine.move_texts(state)


class TestACommandedCaptureThroughAPortal:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_it_takes_its_own_side_under_captures_own(self, kind: str, side: int) -> None:
        exit_ = _sq(_flip(EXIT[kind], side))
        guard = {EXIT[kind]: ("G", 1)}
        engine = _engine(_spec(kind, side, own=True))
        state = _position(engine, kind, side, guard)
        (move,) = [m for m in _passes(engine, state) if move_to(m) == exit_]
        assert move & MOVE_FOREIGN
        after = engine.apply(state, move)
        spec = engine.spec
        assert spec.piece_ids[spec.type_of(after.board[exit_])] == "R"
        assert spec.owner_of(after.board[exit_]) == 1 - side  # still the owner's
        assert after.board[_sq(_flip("c5", side))] == 0
        assert engine._compute_hash(after) == after.key
        # Without captures_own the guard is no landing.
        plain = _engine(_spec(kind, side))
        assert exit_ not in _targets(_passes(plain, _position(plain, kind, side, guard)))

    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("kind", KINDS)
    def test_it_never_takes_the_commanders_piece(self, kind: str, side: int) -> None:
        engine = _engine(_spec(kind, side, own=True))
        state = _position(engine, kind, side, {EXIT[kind]: ("G", 0)})
        assert _sq(_flip(EXIT[kind], side)) not in _targets(_passes(engine, state)), (
            engine.move_texts(state))


# --------------------------------------------------------------------------- oracle and undo


def _games() -> dict[str, dict[str, Any]]:
    both = _spec("momentum", 0, open_to="both", own=True)
    both["rules"]["portals"].append({
        "orange": "a4", "blue": "f3", "kind": "fold", "moves": [STEP],
        "passage": {"0": "closed", "1": "both_ways"},
    })
    return {
        "momentum": _spec("momentum", 0, open_to="owner", own=True),
        "fold": _spec("fold", 0, open_to="commander", own=True),
        "both": both,
    }


class TestTheOracle:
    @pytest.mark.parametrize("game", sorted(_games()))
    @pytest.mark.parametrize("seed", range(4))
    def test_walks_agree_with_the_oracle(self, game: str, seed: int) -> None:
        """Every ply of seeded walks (a commanded pass played where one is offered, half the
        time): the core's legal list is the oracle's, and the incremental key the full scan's."""
        spec = _games()[game]
        oracle = SightOracle(spec)
        engine = oracle.engine
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(70):
            moves = engine.legal_moves(state)
            assert normalise(moves) == oracle.legal_moves(state)
            assert state.key == engine._compute_hash(state)
            if engine.terminal(state, moves) is not None:
                break
            passes = [m for m in moves if move_foreign(m) and (
                move_type(m) == MOVE_PORTAL or move_fold(m))]
            state = engine.apply(
                state, rng.choice(passes) if passes and rng.random() < 0.5 else rng.choice(moves))

    def test_the_walks_meet_the_case(self) -> None:
        """The walks above play a commanded pass through each kind."""
        played: dict[str, int] = {}
        for game, spec in _games().items():
            engine = Engine(spec)
            for seed in range(4):
                rng = random.Random(seed)
                state = engine.initial_state()
                for _ in range(70):
                    moves = engine.legal_moves(state)
                    if engine.terminal(state, moves) is not None:
                        break
                    passes = [m for m in moves if move_foreign(m) and (
                        move_type(m) == MOVE_PORTAL or move_fold(m))]
                    move = (rng.choice(passes) if passes and rng.random() < 0.5
                            else rng.choice(moves))
                    played[game] = played.get(game, 0) + int(move in passes)
                    state = engine.apply(state, move)
        assert all(played.get(game, 0) > 0 for game in _games()), played

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("side", [0, 1])
    def test_the_probe_positions_agree_with_the_oracle(self, kind: str, side: int) -> None:
        for open_to in ("owner", "commander", "both"):
            spec = _spec(kind, side, open_to=open_to, own=True)
            oracle = SightOracle(spec)
            state = _position(oracle.engine, kind, side, {EXIT[kind]: ("G", 1)})
            assert normalise(oracle.engine.legal_moves(state)) == oracle.legal_moves(state)


class TestUndo:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("side", [0, 1])
    def test_perft_matches_an_apply_count(self, kind: str, side: int) -> None:
        engine = _engine(_spec(kind, side, open_to="both", own=True))
        state = _position(engine, kind, side, {"b6": ("G", 1)})
        assert _passes(engine, state)

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)
