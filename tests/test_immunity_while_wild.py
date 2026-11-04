"""SR4g (`docs/plan-schema-next.md` §2.8 row N17, box SR4g): a wild immunity holds only while
the piece is wild.

The ruling (2026-10-27): `npcs[].immune` -- *only these types may take this wild piece* -- is a
fact about the wild piece, not about its type. A wolf taken into a hand and dropped is a
player's piece, and is captured as any piece of its type.

The probe (fed past the validator on SR4f's build `1795ac6ba1c68079`; `TestTheProbe` keeps it
as the twin that crosses no `capture_owned`): the immunity followed the piece. A guard took the
wolf into its hand, the wolf was dropped, and the enemy rook on its file was offered no capture
of it; a wolf placed for a player was as immune. The matrix is keyed by type, and the core read
it for every byte alike. So the core now reads the matrix by *victim column*
(`Tables::victim_of`): a wild byte's column is its type, any other byte's its type plus
`n_types`, into a matrix whose second half is `capture_owned`, the compiler's matrix without
the immunity. It crosses only where a piece of an immune type can stand on the board without
being wild (`schema.npc_immunity_unbound`), so no stored game's tables move.

The oracle is the game without the immunity (rule 3's declared test oracle is this file's
`_expected`): from any position, the moves are the twin's less exactly the captures of a *wild*
immune piece by a type its list does not name, and each such move leaves what the twin's
leaves. `make` and `unmake` are held by seeded walks (the key the full scan's at every ply) and
by `perft` against an `apply`-built count. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

import chess_variant_sandbox.engine.compile as compile_module
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, move_from, move_to, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.loop.operators import OPERATORS
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import npc_immunity_unbound, validate_spec
from tests.feature_inventory import _schema_seventy_one_uses

WIDTH = 6
WILD = 2
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
LOOP = ["b3", "c3", "d3", "e3", "e4", "d4", "c4", "b4"]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(**rules: Any) -> dict[str, Any]:
    """6x6 won by taking the king: a king, a rook and two guards each, a wolf on a loop through
    the middle that only a guard may take, and a hand that fills from captures."""
    return copy.deepcopy({
        "schema_version": 31, "name": "the tame wolf",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
            {"id": "W", "name": "wolf", "value": 3, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["f1"]},
            {"piece": "N", "squares": ["c2", "d2"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 80,
            "npcs": [{"piece": "W", "route": list(LOOP), "immune": ["N"]}],
            "drops": {"from_captures": True},
            **rules,
        },
    })


def _twin(spec: dict[str, Any]) -> dict[str, Any]:
    """The same game without the immunity: what a capture of anything is."""
    twin = copy.deepcopy(spec)
    for npc in twin["rules"]["npcs"]:
        npc.pop("immune", None)
    return twin


def _owner_blind(spec: dict[str, Any]) -> Engine:
    """Today's reading before SR4g: the immunity on every piece of the type, whoever owns it
    (the compiled game with no `capture_owned`, past the validator's refusal of the time)."""
    return Engine(dataclasses.replace(compile_module.compile_spec(spec), capture_owned=()))


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              hand: tuple[tuple[int, ...], tuple[int, ...]] | None = None) -> GameState:
    moved = [square for square, (_, player) in placement.items() if player != WILD]
    state = engine.make_state(placement, side_to_move=side, moved=moved)
    # `make_state` leaves the hands empty; a hand game's position holds one slot per type.
    state = dataclasses.replace(state, hands=hand or engine.initial_state().hands)
    return dataclasses.replace(state, key=engine._compute_hash(state))


def _moves(engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
    return [m for m in engine.legal_moves(state)
            if (move_from(m), move_to(m)) == (_sq(origin), _sq(target))]


def _byte(engine: Engine, pid: str, player: int) -> int:
    return engine.spec.encode_piece(player, engine.spec.piece_ids.index(pid))


def _one_wolf_in_hand(engine: Engine) -> tuple[tuple[int, ...], tuple[int, ...]]:
    hand = [0] * len(engine.spec.piece_ids)
    hand[engine.spec.piece_ids.index("W")] = 1
    return tuple(hand), tuple([0] * len(hand))


def _walk_start(engine: Engine) -> GameState:
    """The opening with a wolf in each player's hand, so a walk meets tame wolves early."""
    state = engine.initial_state()
    hand = _one_wolf_in_hand(engine)[0]
    state = dataclasses.replace(state, hands=(hand, hand))
    return dataclasses.replace(state, key=engine._compute_hash(state))


def _expected(engine: Engine, twin: Engine, state: GameState) -> set[int]:
    """The oracle: the twin's moves less a capture of a *wild* wolf by anything but a guard or
    a wolf (an immunity never binds a wild type, `schema.npc_immune_attackers`, so a player's
    wolf may take a wild one)."""
    spec = engine.spec
    wolf = spec.piece_ids.index("W")
    takers = {spec.piece_ids.index("N"), wolf}
    kept = set()
    for move in twin.legal_moves(state):
        taken = state.board[move_to(move)]
        mover = state.board[move_from(move)]
        if (taken and spec.owner_of(taken) == WILD and spec.type_of(taken) == wolf
                and move_type(move) != MOVE_DROP and spec.type_of(mover) not in takers):
            continue
        kept.add(move)
    return kept


# --------------------------------------------------------------------------- the probe


class TestTheProbe:
    """What the core did with the pair before SR4g, kept as the owner-blind twin."""

    PLACEMENT = {"a1": ("K", 0), "a6": ("K", 1), "f6": ("R", 1), "f2": ("W", 0)}

    def test_owner_blind_a_players_wolf_was_immune(self) -> None:
        engine = _owner_blind(_spec())
        state = _position(engine, self.PLACEMENT, side=1)
        assert _moves(engine, state, "f6", "f2") == []

    def test_owner_blind_a_dropped_wolf_was_immune(self) -> None:
        engine = _owner_blind(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f6": ("R", 1)},
                          hand=_one_wolf_in_hand(engine))
        (drop,) = [m for m in engine.legal_moves(state)
                   if move_type(m) == MOVE_DROP and move_to(m) == _sq("f2")]
        after = engine.apply(state, drop)
        assert _moves(engine, after, "f6", "f2") == []


# --------------------------------------------------------------------------- the rule


class TestImmuneOnlyWhileWild:
    def test_a_wild_wolf_is_still_immune(self) -> None:
        """Only the guard may take the wild wolf; the rook beside it on the rank may not."""
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "c2": ("N", 0), "f3": ("R", 0), "c3": ("W", WILD),
        })
        assert _moves(engine, state, "f3", "c3") == []
        (take,) = _moves(engine, state, "c2", "c3")
        after = engine.apply(state, take)
        assert after.hands[0][engine.spec.piece_ids.index("W")] == 1
        assert after.key == engine._compute_hash(after)

    def test_a_dropped_wolf_is_captured_as_any_piece(self) -> None:
        """The wolf the guard took is dropped on f2; the enemy rook on f6 takes it, and it
        goes to that player's hand as any piece would."""
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f6": ("R", 1)},
                          hand=_one_wolf_in_hand(engine))
        (drop,) = [m for m in engine.legal_moves(state)
                   if move_type(m) == MOVE_DROP and move_to(m) == _sq("f2")]
        dropped = engine.apply(state, drop)
        assert dropped.board[_sq("f2")] == _byte(engine, "W", 0)
        (take,) = _moves(engine, dropped, "f6", "f2")
        after = engine.apply(dropped, take)
        assert after.board[_sq("f2")] == _byte(engine, "R", 1)
        assert after.hands[1][engine.spec.piece_ids.index("W")] == 1
        assert after.key == engine._compute_hash(after)

    def test_a_wolf_placed_for_a_player_is_captured_as_any_piece(self) -> None:
        """A wolf in the setup for the first player is a player's piece from the start."""
        spec = _spec()
        spec["setup"]["placements"].append({"piece": "W", "squares": ["b1"]})
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = engine.initial_state()
        assert state.board[_sq("b6")] == _byte(engine, "W", 1)
        position = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "b1": ("W", 0), "b6": ("R", 1),
        }, side=1)
        assert _moves(engine, position, "b6", "b1")

    def test_a_wild_wolf_eats_a_players_wolf(self) -> None:
        """The wild wolf's own row is untouched: it steps onto a dropped wolf and eats it."""
        engine = Engine(_spec())
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "c3": ("W", 1), "b3": ("W", WILD),
        })
        after, steps = engine.apply_with_phase(state, _moves(engine, state, "a1", "a2")[0])
        assert [(s.origin, s.target, s.eaten) for s in steps] == [
            (_sq("b3"), _sq("c3"), _byte(engine, "W", 1))
        ]
        assert after.key == engine._compute_hash(after)


# --------------------------------------------------------------------------- the oracle


class TestAgainstTheTwin:
    @pytest.mark.parametrize("seed", range(6))
    def test_walks_offer_the_oracles_moves(self, seed: int) -> None:
        """Every ply of a walk, captures preferred: the moves are `_expected`'s, each leaves
        the twin's board and hands, and the key is the full scan's."""
        engine, twin = Engine(_spec()), Engine(_twin(_spec()))
        rng = random.Random(seed)
        state = _walk_start(engine)
        for _ in range(70):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state, moves) is not None:
                break
            assert set(moves) == _expected(engine, twin, state)
            captures = [m for m in moves if state.board[move_to(m)]]
            move = rng.choice(captures if captures and rng.random() < 0.6 else moves)
            after = engine.apply(state, move)
            mirror = twin.apply(state, move)
            assert (after.board, after.hands) == (mirror.board, mirror.hands)
            assert after.key == engine._compute_hash(after)
            state = after

    def test_the_walks_capture_a_tame_wolf(self) -> None:
        """The walks above reach what they are for: a capture of a player's wolf."""
        engine = Engine(_spec())
        wolf = engine.spec.piece_ids.index("W")
        reached = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = _walk_start(engine)
            for _ in range(70):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                captures = [m for m in moves if state.board[move_to(m)]]
                move = rng.choice(captures if captures and rng.random() < 0.6 else moves)
                taken = state.board[move_to(move)]
                reached += bool(
                    taken and engine.spec.owner_of(taken) < 2
                    and engine.spec.type_of(taken) == wolf and move_type(move) != MOVE_DROP
                )
                state = engine.apply(state, move)
        assert reached > 0

    def test_the_oracle_refuses_the_owner_blind_reading(self) -> None:
        """The walks would catch today's reading: somewhere a tame wolf is not offered to a
        piece the oracle lets take it."""
        engine, twin = _owner_blind(_spec()), Engine(_twin(_spec()))
        caught = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = _walk_start(engine)
            for _ in range(70):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                caught += set(moves) != _expected(engine, twin, state)
                captures = [m for m in moves if state.board[move_to(m)]]
                move = rng.choice(captures if captures and rng.random() < 0.6 else moves)
                state = engine.apply(state, move)
        assert caught > 0

    def test_perft_counts_what_apply_counts(self) -> None:
        """`perft` makes and unmakes on one position, with a wolf in hand, one wild and one
        a player's: a capture an unmake left wrong changes a sibling's moves."""
        engine = Engine(_spec())
        hand = _one_wolf_in_hand(engine)
        state = _position(engine, {
            "a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f6": ("R", 1), "c2": ("N", 0),
            "e5": ("W", 1), "c3": ("W", WILD),
        }, hand=(hand[0], hand[0]))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)


# --------------------------------------------------------------------------- the tables


class TestTheTables:
    def test_the_pair_crosses_the_second_matrix(self) -> None:
        compiled = compile_module.compile_spec(_spec())
        wolf, rook = compiled.piece_ids.index("W"), compiled.piece_ids.index("R")
        assert not compiled.capture_allowed[rook][wolf]
        assert compiled.capture_owned[rook][wolf]

    def test_without_a_way_off_the_route_nothing_crosses(self) -> None:
        """No hand, no placement, no promotion: the tables are the ones they were."""
        spec = _spec()
        spec["rules"].pop("drops")
        assert npc_immunity_unbound(spec) == set()
        assert compile_module.compile_spec(spec).capture_owned == ()

    def test_a_hand_that_never_holds_the_wolf_crosses_nothing(self) -> None:
        spec = _spec(drops={"from_captures": False, "stock": {"R": 1}})
        assert npc_immunity_unbound(spec) == set()


# --------------------------------------------------------------------------- the lift


class TestTheLift:
    def test_the_pair_is_valid(self) -> None:
        assert validate_spec(_spec()) == []

    def test_describe_says_it(self) -> None:
        text = describe_spec(_spec())
        assert "immune only while it is wild" in text
        spec = _spec()
        spec["rules"].pop("drops")
        assert "immune only while it is wild" not in describe_spec(spec)

    def test_fairy_stockfish_refuses_the_pair(self) -> None:
        """Fairy-Stockfish refuses `npcs` itself, so the pair is refused there by that."""
        with pytest.raises(Untranslatable, match="rules.npcs"):
            translate(compile_module.compile_spec(_spec()))

    def test_the_inventory_names_it(self) -> None:
        assert _schema_seventy_one_uses(_spec()) == {"rules.npcs.immune.drops"}
        assert _schema_seventy_one_uses(_twin(_spec())) == set()

    @pytest.mark.parametrize("name", ["hand-the-pieces-back", "set-a-patrol"])
    def test_the_operators_enter_the_pair(self, name: str) -> None:
        """`hand-the-pieces-back` adds a hand to a game with an immune wild piece, `set-a-patrol`
        an immune wild piece to a hand game; neither declines on the other rule now."""
        operator = next(op for op in OPERATORS if op.name == name)
        parent = _spec()
        if name == "hand-the-pieces-back":
            parent["rules"].pop("drops")
        else:
            parent["rules"].pop("npcs")
        entered = 0
        for seed in range(60):
            child = copy.deepcopy(parent)
            if operator.apply(child, random.Random(seed)) is None:
                continue
            if validate_spec(child) == [] and _schema_seventy_one_uses(child):
                entered += 1
        assert entered > 0
