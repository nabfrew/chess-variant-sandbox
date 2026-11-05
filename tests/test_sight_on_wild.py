"""Sight on wild pieces, the duck and nobody's pieces (generation 84, SR5f,
`docs/plan-schema-next.md` §2.7 rows V11, V15, ruled 2026-10-27).

**V11.** A sight whose ``affects`` names a wild type, under ``whose: "any"`` (a wild piece has no
side), binds those wild pieces: a frozen wild piece neither walks nor pounces, a disarmed one walks
onto empty squares only and never pounces, a shielded one is taken by nobody, and a frozen or
disarmed one threatens nothing (a royal on its next square is not in check from it). The phase
reads the effects once, at its start (§3.1's snapshot), as it reads the shields. The duck may be
named by a freeze only (nothing takes it, it takes nothing): a frozen duck passes. A command on
either stays refused (V13). A sight without ``affects`` touches no wild piece, as before.

**V15.** A piece nobody owns sees as a wild piece does: ``whose: "any"``, no command, lines that
read alike for both players; its effects bind both players' pieces alike.

Hand positions on a 6x6 board (`tests/test_wild_sight.py`'s spec, with a warden ``G`` whose ring
names the wolf), the refusals, the words, seeded walks against `tests/sight_oracle.py` (widened to
the wild arm), and `perft` against an `apply`-built count for the undo.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import (
    SIGHT_DISARM,
    SIGHT_FREEZE,
    SIGHT_SHIELD,
    compile_spec,
)
from chess_variant_sandbox.engine.moves import MOVE_DUCK, decode_move
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from tests.feature_inventory import _schema_eighty_four_uses
from tests.sight_oracle import SightOracle
from tests.test_duck import _small as _duck_spec
from tests.test_wild_sight import _spec as _wild_spec

RING = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1, "through": True,
        "whose": "any"}


def _tamed(effect: str, npcs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """`test_wild_sight._spec`'s game, the warden's ring naming the wolf under ``effect``."""
    spec = _wild_spec(npcs if npcs is not None else [{"piece": "W", "route": ["c3", "c4"]}])
    warden = next(piece for piece in spec["pieces"] if piece["id"] == "G")
    warden["sight"] = [dict(RING, effect=effect, affects=["W"])]
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def _phase(engine: Engine, state: GameState, text: str) -> tuple[GameState, tuple]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, phase


def _statue(sight: dict[str, Any] | None = None) -> dict[str, Any]:
    """Kings, rooks and knights on 6x6, and a statue ``S`` placed for nobody on c3 whose ring
    freezes both players' pieces beside it."""
    spec = _wild_spec([], warden=False)
    del spec["rules"]["npcs"]
    spec["pieces"] = [piece for piece in spec["pieces"] if piece["id"] not in ("W", "H")]
    spec["pieces"].append({"id": "S", "name": "statue", "value": 1, "moves": [],
                           "sight": [sight or dict(RING, effect="freeze")]})
    spec["setup"]["placements"].append({"piece": "S", "player": "nobody", "squares": ["c3"]})
    return spec


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    @pytest.mark.parametrize("effect", ["freeze", "disarm", "shield"])
    def test_a_sight_may_name_a_wild_type(self, effect: str) -> None:
        assert validate_spec(_tamed(effect)) == []

    def test_a_command_never_names_a_wild_type(self) -> None:
        spec = _tamed("freeze")
        spec["pieces"][3]["sight"] = [dict(RING, effect="command", whose="enemy", affects=["W"])]
        assert any("nobody commands for it, so a command never touches it" in e
                   for e in validate_spec(spec))

    @pytest.mark.parametrize("whose", ["enemy", "own"])
    def test_only_whose_any_reaches_a_wild_piece(self, whose: str) -> None:
        spec = _tamed("freeze")
        spec["pieces"][3]["sight"][0]["whose"] = whose
        assert any(f"which has no side, under whose {whose!r}" in e for e in validate_spec(spec))

    def test_the_duck_takes_a_freeze_only(self) -> None:
        spec = _duck_spec()
        spec["pieces"][1]["sight"] = [dict(RING, effect="freeze", affects=["D"])]
        assert validate_spec(spec) == []
        for effect in ("shield", "disarm"):
            spec["pieces"][1]["sight"] = [dict(RING, effect=effect, affects=["D"])]
            assert any("nothing takes the duck and it takes nothing" in e
                       for e in validate_spec(spec)), effect

    def test_a_nobodys_sight_reads_as_a_wild_pieces(self) -> None:
        assert validate_spec(_statue()) == []
        refusals = {
            "which has no side; use 'any'": dict(RING, effect="freeze", whose="enemy"),
            "nobody commands for it": dict(RING, effect="command", whose="any"),
            "reads differently for the two players": {
                "vectors": [[0, 1]], "range": 1, "effect": "freeze", "whose": "any",
            },
        }
        for words, sight in refusals.items():
            assert any(words in e for e in validate_spec(_statue(sight))), words

    def test_the_inventory_names_both(self) -> None:
        assert _schema_eighty_four_uses(_tamed("freeze")) == {"pieces.sight.wild"}
        assert _schema_eighty_four_uses(_statue()) == {"pieces.sight.nobodys"}
        assert _schema_eighty_four_uses(_wild_spec([{"piece": "W", "route": ["c3"]}])) == set()

    def test_describe_says_it(self) -> None:
        said = describe_spec(_tamed("freeze"))
        assert "a frozen wild piece neither walks nor pounces" in said
        assert "a disarmed wild piece walks only onto empty squares" in describe_spec(
            _tamed("disarm"))
        assert "a shielded wild piece cannot be taken" in describe_spec(_tamed("shield"))
        assert "A piece nobody owns sees as a wild piece does" in describe_spec(_statue())
        assert "binds it too" not in describe_spec(_wild_spec([{"piece": "W", "route": ["c3"]}]))

    def test_fairy_stockfish_refuses_every_sight(self) -> None:
        for spec in (_tamed("freeze"), _statue()):
            with pytest.raises(Untranslatable):
                translate(spec)

    def test_a_sight_naming_no_wild_type_compiles_as_before(self) -> None:
        """No third block of sources unless a sight names a wild type: every stored game's
        tables are the ones they were."""
        plain = compile_spec(_wild_spec([{"piece": "W", "route": ["c3", "c4"]}]))
        assert plain.sight is not None and not plain.sight.tamed
        tamed = compile_spec(_tamed("freeze"))
        assert tamed.sight is not None and tamed.sight.tamed


# --------------------------------------------------------------------------- the acceptance


class TestAFrozenWildPiece:
    PLACE = {"a1": ("K", 0), "f6": ("K", 1), "d3": ("G", 0), "c3": ("W", 2), "c4": ("N", 1)}

    def test_it_neither_walks_nor_eats(self) -> None:
        engine = _engine(_tamed("freeze"))
        state = engine.make_state(self.PLACE)
        assert engine.effects(state).bits[_sq(engine, "c3")] == SIGHT_FREEZE
        _, phase = _phase(engine, state, "Ka1-a2")
        assert phase == ()
        # Without the warden beside it, the wolf eats the knight.
        _, phase = _phase(engine, state, "Gd3-e3")
        assert [(step.origin, step.target) for step in phase] == [
            (_sq(engine, "c3"), _sq(engine, "c4"))
        ]

    def test_it_threatens_nothing(self) -> None:
        """A royal on the frozen wolf's next square is not in check from it, and may step
        there; with the warden gone it may not."""
        engine = _engine(_tamed("freeze"))
        place = {"a1": ("K", 0), "f6": ("K", 1), "d3": ("G", 0), "c3": ("W", 2), "b4": ("K", 0)}
        del place["a1"]
        state = engine.make_state(place)
        assert "Kb4-c4" in engine.move_texts(state)
        plain = _engine(_wild_spec([{"piece": "W", "route": ["c3", "c4"]}]))
        del place["d3"]  # the plain warden shields its king: no warden at all
        state = plain.make_state(place)
        assert "Kb4-c4" not in plain.move_texts(state)

    def test_a_frozen_pouncer_does_not_pounce(self) -> None:
        npcs = [{"piece": "W", "route": ["c3"], "pounce": "prey"}]
        for effect, pounces in (("freeze", False), ("disarm", False), ("shield", True)):
            engine = _engine(_tamed(effect, npcs))
            state = engine.make_state(
                {"a1": ("K", 0), "f6": ("K", 1), "d3": ("G", 0), "c3": ("W", 2), "a4": ("R", 0)}
            )
            _, phase = _phase(engine, state, "Ra4-b4")
            assert bool(phase) is pounces, effect
            if pounces:
                assert phase[0].target == _sq(engine, "b4")


class TestADisarmedWildPiece:
    def test_it_walks_onto_an_empty_square(self) -> None:
        engine = _engine(_tamed("disarm"))
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "d3": ("G", 0), "c3": ("W", 2)})
        assert engine.effects(state).bits[_sq(engine, "c3")] == SIGHT_DISARM
        _, phase = _phase(engine, state, "Ka1-a2")
        assert [(step.origin, step.target) for step in phase] == [
            (_sq(engine, "c3"), _sq(engine, "c4"))
        ]

    def test_it_does_not_eat(self) -> None:
        engine = _engine(_tamed("disarm"))
        state = engine.make_state(TestAFrozenWildPiece.PLACE)
        _, phase = _phase(engine, state, "Ka1-a2")
        assert phase == ()


class TestAShieldedWildPiece:
    def test_nobody_takes_it(self) -> None:
        engine = _engine(_tamed("shield"))
        place = {"a1": ("K", 0), "f6": ("K", 1), "d3": ("G", 0), "c3": ("W", 2)}
        for side in (0, 1):
            state = engine.make_state({**place, "a3": ("R", side)}, side_to_move=side)
            assert engine.effects(state).bits[_sq(engine, "c3")] == SIGHT_SHIELD
            assert "Ra3xc3" not in engine.move_texts(state), side
        plain = _engine(_wild_spec([{"piece": "W", "route": ["c3", "c4"]}]))
        state = plain.make_state({**place, "a3": ("R", 0)})
        assert "Ra3xc3" in plain.move_texts(state)


class TestAFrozenDuck:
    def test_it_passes(self) -> None:
        spec = _duck_spec()
        spec["pieces"][1]["sight"] = [dict(RING, effect="freeze", affects=["D"])]
        engine = _engine(spec)
        place = {"a1": ("K", 0), "f6": ("K", 1), "c3": ("D", 2)}
        for side in (0, 1):
            frozen = engine.make_state({**place, "d3": ("R", side)}, side_to_move=side,
                                       duck_owed=True)
            moves = engine.legal_moves(frozen)
            assert len(moves) == 1 and decode_move(moves[0])[2] == MOVE_DUCK
            assert decode_move(moves[0])[0] == decode_move(moves[0])[1]  # the pass
            after = engine.apply(frozen, moves[0])
            assert after.key == engine._compute_hash(after)
            free = engine.make_state({**place, "e3": ("R", side)}, side_to_move=side,
                                     duck_owed=True)
            assert len(engine.legal_moves(free)) > 1


class TestANobodysSight:
    def test_it_reads_alike_for_both_players(self) -> None:
        """A statue on c3 freezes a rook of each side beside it; each side's rook beside it
        has no move, and the mirrored position mirrors the list and the bits."""
        engine = _engine(_statue())
        place = {"a1": ("K", 0), "f6": ("K", 1), "d3": ("R", 0), "b4": ("R", 1), "e5": ("N", 0),
                 "c3": ("S", 2)}
        for side in (0, 1):
            state = engine.make_state(place, side_to_move=side)
            bits = engine.effects(state).bits
            assert bits[_sq(engine, "d3")] == SIGHT_FREEZE
            assert bits[_sq(engine, "b4")] == SIGHT_FREEZE
            assert not [t for t in engine.move_texts(state) if t.startswith("R")], side
        mirror = {
            f"{name[0]}{7 - int(name[1])}": (piece, owner if owner == 2 else 1 - owner)
            for name, (piece, owner) in place.items()
        }
        spec = _statue()
        spec["setup"]["placements"][-1]["squares"] = ["c4"]
        image_engine = _engine(spec)
        image = image_engine.make_state(mirror, side_to_move=1)
        state = engine.make_state(place)

        def reflect(text: str) -> str:
            return "".join(str(7 - int(c)) if c.isdigit() else c for c in text)

        assert sorted(map(reflect, engine.move_texts(state))) == sorted(
            image_engine.move_texts(image))


# --------------------------------------------------------------------------- the oracle


def _walk_spec(effect: str) -> dict[str, Any]:
    """A wolf on a loop through the middle that the wardens' rings name, both armies whole."""
    spec = _tamed(effect, [{"piece": "W", "route": ["b3", "c3", "d3", "d4", "c4", "b4"]}])
    spec["pieces"][3]["sight"].append(dict(RING, effect="shield", whose="own"))
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "G":
            placement["squares"] = ["c2"] if placement["player"] == 0 else ["d5"]
    return spec


def _apply_count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _apply_count(engine, engine.apply(state, move), depth - 1)
        for move in engine.legal_moves(state)
    )


class TestTheWalks:
    @pytest.mark.parametrize("effect", ["freeze", "disarm", "shield"])
    def test_every_list_is_the_oracles(self, effect: str) -> None:
        spec = _walk_spec(effect)
        assert validate_spec(spec) == [], validate_spec(spec)
        judge = SightOracle(spec)
        engine = judge.engine
        bound = 0
        for seed in range(6):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(60):
                if engine.terminal(state) is not None:
                    break
                moves = engine.legal_moves(state)
                assert moves == judge.legal_moves(state), (effect, seed, state.ply)
                assert engine.in_check(state) == judge.royal_attacked(state, state.side_to_move)
                bits = judge.effects(state.board)
                wolf = next((sq for sq, v in enumerate(state.board)
                             if v and engine.spec.owner_of(v) == 2), None)
                bound += wolf is not None and bool(bits[wolf])
                if not moves:
                    break
                state = engine.apply(state, rng.choice(moves))
                assert state.key == engine._compute_hash(state)
        assert bound, "no walk ever bound the wolf"

    @pytest.mark.parametrize("effect", ["freeze", "disarm"])
    def test_perft_counts_what_apply_counts(self, effect: str) -> None:
        engine = _engine(_walk_spec(effect))
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c2": ("G", 0), "d5": ("G", 1),
                                   "c3": ("W", 2), "b1": ("R", 0), "e6": ("R", 1)})
        for side in (0, 1):
            here = state.with_(side_to_move=side)
            assert engine.perft(here, 3) == _apply_count(engine, here, 3)

    def test_the_duck_perft_counts_what_apply_counts(self) -> None:
        spec = _duck_spec()
        spec["pieces"][1]["sight"] = [dict(RING, effect="freeze", affects=["D"])]
        engine = _engine(spec)
        state = engine.make_state({"a1": ("K", 0), "f6": ("K", 1), "c3": ("D", 2),
                                   "d3": ("R", 0), "b5": ("R", 1)})
        assert engine.perft(state, 3) == _apply_count(engine, state, 3)

    def test_a_frozen_line_piece_waits_unturned(self) -> None:
        """A turning-back wolf frozen on its line waits and does not turn round."""
        spec = _tamed("freeze", [{"piece": "W", "route": ["c3", "c4"], "turns_back": True}])
        spec["rules"]["win"] = "royal_capture"
        engine = _engine(spec)
        state = engine.make_state(TestAFrozenWildPiece.PLACE)
        after, phase = _phase(engine, state, "Ka1-a2")
        assert phase == () and after.heading_back == state.heading_back
        free = copy.deepcopy(TestAFrozenWildPiece.PLACE)
        del free["d3"]
        state = engine.make_state(free)
        _, phase = _phase(engine, state, "Ka1-a2")
        assert phase != ()
