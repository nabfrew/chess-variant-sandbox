"""SR5a (`docs/plan-schema-next.md` §2.4 rows T1, T4): a strike or contacts beside a blast, and
the switch `explode_on_capture.strikes`.

The rulings (2026-10-27): T1 is a switch, `strikes`, one of `"none"` (the default: a capture
that explodes strikes nothing; its target and the blast's victims are removed and the capturer
destroyed), `"target"` (the piece taken is struck: it follows its chain link and stands on the
capture square, or goes home under `"home"`), `"blast"` (each victim not immune to the blast or
the strike is struck where it stands, or sent home) or `"both"`. A value but `"none"` needs
position `"stay"` or `"home"`, is refused beside the other three and as inert under effect
`"capture"` placed `"stay"`. T4 reads as T1's default: a capture that explodes makes no contact,
and a value but `"none"` is refused beside contacts. The blast clears first, then the struck
pieces are placed; nothing a strike makes explodes or is struck again.

The probe (fed past the validator on SR4m's build `b42c85254ca8c92c`, a rook on f3 taking a
guard on d3 beside a blast of a king's step): the core played the strike before the blast and
read neither ruling. Under rifle with a chain the rook stayed on f3, the guard was demoted on
d3, and the blast then cleared d3 and its neighbours round a capturer that was not there; under
circe the taken guard was reborn on its home before the blast and survived it whether or not it
should, and a demoted one with no home was removed. So `make` now asks `blast_sets_off` first: a
capture that explodes is the blast's (no strike, no contact), and `blast_strikes` places what
`strikes` hands the strike after the clear, the one statement the legality filter plays too.

Readings the rows left open, settled on the plan's principles and written in the record: a blast
victim of either side is struck (the blast reaches both); the placements go in order, the piece
taken first, then the victims in the blast's order, each onto an empty square only; a player's
own contact link is a meeting, not a capture (SR4i's T9), so it sets off no blast; a strike or
contact placed `"choose"`, and either in rounds, stays refused beside a blast; and under
`"none"` a blast on every capture beside a strike (or a contact table with no own link) leaves
it nothing to act on, refused as a rule that never fires. Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refusal, refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_seventy_eight_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
IDS = ("K", "R", "G", "P", "W")
CHAIN = [{"from": "G", "to": "P"}, {"from": "R", "to": "G"}]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(position: str = "stay", effect: str = "demote", strikes: str | None = "target",
          **explosion: Any) -> dict[str, Any]:
    """Kings a1/a6, a rook f1, guards c1/d1 and a ward (W, immune to the blast) e1, mirrored;
    a blast of a king's step sparing kings and wards; a strike demoting a guard to a pawn and
    a rook to a guard, kings and rooks immune to it."""
    spec: dict[str, Any] = {
        "name": "the blast that strikes (SR5a test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "P", "value": 1, "moves": [KING]},
            {"id": "W", "value": 1, "moves": [KING]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["c1", "d1"]},
            {"piece": "W", "squares": ["e1"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 60,
            "explode_on_capture": {"radius": 1, "immune": ["K", "W"], **explosion},
            "strike": {"effect": effect, "position": position, "immune": ["K", "R"]},
        },
    }
    if effect != "capture":
        spec["rules"]["strike"]["chain"] = CHAIN
    if strikes is not None:
        spec["rules"]["explode_on_capture"]["strikes"] = strikes
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move_from(move), move_to(move)) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _board(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    spec = engine.spec
    return {
        f"{'abcdef'[sq % WIDTH]}{sq // WIDTH + 1}": (IDS[spec.type_of(v)], spec.owner_of(v))
        for sq, v in enumerate(state.board) if v
    }


# The rook on f3 takes the guard on d3. In its blast: an enemy rook on d4 (immune to the strike),
# an enemy guard on c3, the mover's own guard on c2, and an enemy ward on e4 (immune to the blast).
BLAST = {
    "a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("G", 1), "d4": ("R", 1),
    "c3": ("G", 1), "c2": ("G", 0), "e4": ("W", 1),
}
UNTOUCHED = {"a1": ("K", 0), "a6": ("K", 1), "e4": ("W", 1)}


def _play(spec: dict[str, Any], placement: dict[str, tuple[str, int]] = BLAST,
          origin: str = "f3", target: str = "d3") -> tuple[Engine, GameState, GameState, int]:
    engine = _engine(spec)
    state = engine.make_state(placement, side_to_move=0, moved=list(placement))
    move = _move(engine, state, origin, target)
    after = engine.apply(state, move)
    assert after.key == engine._compute_hash(after)
    return engine, state, after, move


class TestTheFourValues:
    def test_none_strikes_nothing_and_a_capture_that_does_not_explode_is_struck(self) -> None:
        """`"none"` beside a blast of the guard's being taken: the capturer, the guard and the
        victims go, nothing is struck; a capture of a rook, which sets off no blast, is struck
        as alone (rifle with a chain: the rook is demoted where it stood, the taker stays)."""
        spec = _spec(strikes=None, of=["G"])
        engine, _, after, _ = _play(spec)
        assert _board(engine, after) == UNTOUCHED
        placement = {**BLAST, "d3": ("R", 1), "d4": ("G", 1)}
        spec["rules"]["strike"]["immune"] = ["K"]
        engine, _, after, _ = _play(spec, placement)
        assert _board(engine, after) == {**placement, "d3": ("G", 1)}

    def test_target_under_stay_stands_on_the_capture_square(self) -> None:
        engine, _, after, move = _play(_spec("stay", strikes="target"))
        assert _board(engine, after) == {**UNTOUCHED, "d3": ("P", 1)}

    def test_target_under_home_goes_home(self) -> None:
        """Circe (effect capture): the guard is reborn on its home, outside the blast."""
        engine, _, after, _ = _play(_spec("home", "capture", strikes="target"))
        board = _board(engine, after)
        reborn = {sq: p for sq, p in board.items() if sq not in UNTOUCHED}
        assert list(reborn.values()) == [("G", 1)]
        assert next(iter(reborn)) in ("c6", "d6")

    def test_target_under_home_with_its_home_taken_is_removed(self) -> None:
        placement = {**BLAST, "c6": ("W", 0), "d6": ("W", 0)}
        engine, _, after, _ = _play(_spec("home", "capture", strikes="target"), placement)
        assert _board(engine, after) == {**UNTOUCHED, "c6": ("W", 0), "d6": ("W", 0)}

    def test_blast_strikes_its_victims_but_the_immune_ones(self) -> None:
        """`"blast"` under rifle with a chain: both guards in the blast, either side's, are
        demoted where they stood; the rook (immune to the strike) is removed; the ward (immune
        to the blast) is untouched; the guard taken is removed."""
        engine, state, after, move = _play(_spec("stay", strikes="blast"))
        assert _board(engine, after) == {**UNTOUCHED, "c3": ("P", 1), "c2": ("P", 0)}
        assert engine.describe_move(state, move) == "Rf3xd3 (struck: P@c2, P@c3)"

    def test_both_under_home_places_the_target_first(self) -> None:
        """`"both"` under circe: the guard taken is reborn first, then each struck victim in
        the blast's order, each onto an empty home; the rook is immune to the strike."""
        engine, _, after, _ = _play(_spec("home", "capture", strikes="both"))
        board = _board(engine, after)
        assert board["d1"] == ("G", 0) or board["c1"] == ("G", 0)
        black_guards = sorted(sq for sq, p in board.items() if p == ("G", 1))
        assert black_guards == ["c6", "d6"]
        assert sorted(board) == sorted([*UNTOUCHED, *black_guards,
                                        next(sq for sq in ("c1", "d1") if sq in board)])

    def test_both_with_one_home_free_places_only_the_first(self) -> None:
        placement = {**BLAST, "c6": ("W", 0)}
        placement.pop("c2")
        engine, _, after, _ = _play(_spec("home", "capture", strikes="both"), placement)
        board = _board(engine, after)
        assert board == {**UNTOUCHED, "c6": ("W", 0), "d6": ("G", 1)}

    def test_the_capturer_never_survives(self) -> None:
        for position, effect in (("stay", "demote"), ("home", "capture")):
            for strikes in ("target", "blast", "both"):
                engine, _, after, _ = _play(_spec(position, effect, strikes=strikes))
                assert ("R", 0) not in _board(engine, after).values()

    def test_the_move_text_names_what_stands(self) -> None:
        engine, state, _, move = _play(_spec("stay", strikes="target"))
        assert engine.describe_move(state, move) == "Rf3xd3 (struck: P@d3)"
        engine, state, _, move = _play(_spec(strikes=None, of=["G"]))
        assert engine.describe_move(state, move) == "Rf3xd3"


class TestTheFilter:
    def test_a_struck_piece_that_gives_check_makes_the_capture_illegal(self) -> None:
        """Under checkmate a guard promoted to a rook on a3 checks the king on a1: the filter
        places it as `make` does, so the capture is not offered; under `"none"` it is."""
        def spec(strikes: str | None) -> dict[str, Any]:
            out = _spec("stay", "promote", strikes=strikes, of=["G"])
            out["rules"]["win"] = "checkmate"
            out["rules"]["strike"]["chain"] = [{"from": "G", "to": "R"}]
            out["rules"]["strike"]["immune"] = ["K"]
            return out

        placement = {"a1": ("K", 0), "f6": ("K", 1), "f3": ("R", 0), "a3": ("G", 1)}
        for strikes, offered in (("target", False), (None, True)):
            engine = _engine(spec(strikes))
            state = engine.make_state(placement, side_to_move=0, moved=list(placement))
            pairs = {(move_from(m), move_to(m)) for m in engine.legal_moves(state)}
            assert ((_sq("f3"), _sq("a3")) in pairs) is offered, strikes


class TestUndo:
    @pytest.mark.parametrize(("position", "effect", "strikes"), [
        ("stay", "demote", "target"), ("stay", "demote", "blast"),
        ("home", "capture", "target"), ("home", "capture", "both"),
    ])
    def test_perft_counts_what_apply_counts(self, position: str, effect: str, strikes: str) -> None:
        """A tree whose every capture explodes and strikes: an unmake that left a struck piece
        or a cleared one wrong would move the next sibling's count."""
        engine = _engine(_spec(position, effect, strikes=strikes))
        state = engine.make_state(BLAST, side_to_move=0, moved=list(BLAST))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_walks_keep_the_key(self) -> None:
        """Seeded walks from the opening, captures preferred: the key is the full scan's every
        ply, and `apply` leaves its argument as it was."""
        for strikes in ("target", "blast", "both"):
            engine = _engine(_spec("home", "capture", strikes=strikes))
            for seed in range(3):
                rng = random.Random(seed)
                state = engine.initial_state()
                for _ in range(40):
                    moves = engine.legal_moves(state)
                    if not moves or engine.terminal(state, moves) is not None:
                        break
                    loud = [m for m in moves if state.board[move_to(m)]]
                    move = rng.choice(loud if loud and rng.random() < 0.7 else moves)
                    board = bytes(state.board)
                    after = engine.apply(state, move)
                    assert bytes(state.board) == board
                    assert after.key == engine._compute_hash(after)
                    state = after


class TestContacts:
    """T4: a capture that explodes makes no contact; one that does not is met as alone."""

    @staticmethod
    def _spec(**links: Any) -> dict[str, Any]:
        spec = _spec(strikes=None, of=["G"])
        spec["rules"].pop("strike")
        spec["rules"]["contacts"] = {
            "position": "stay", "immune": ["K"],
            "links": [{"mover": "R", "onto": "G", "onto_to": "P"},
                      {"mover": "R", "onto": "W", "onto_to": "P"}, *links.get("more", [])],
        }
        spec["rules"]["explode_on_capture"]["immune"] = ["K"]
        return spec

    def test_an_exploding_capture_makes_no_contact(self) -> None:
        engine, _, after, _ = _play(self._spec())
        assert _board(engine, after) == {"a1": ("K", 0), "a6": ("K", 1)}

    def test_a_capture_that_does_not_explode_is_met(self) -> None:
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("W", 1)}
        engine, _, after, _ = _play(self._spec(), placement)
        assert _board(engine, after) == {**placement, "d3": ("P", 1)}

    def test_an_own_link_is_a_meeting_and_sets_off_nothing(self) -> None:
        own = {"mover": "R", "onto": "G", "onto_to": "P", "own": True}
        placement = {"a1": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("G", 0),
                     "d4": ("G", 1)}
        engine, _, after, _ = _play(self._spec(more=[own]), placement)
        assert _board(engine, after) == {**placement, "d3": ("P", 0)}


class TestTheRefusals:
    def test_each_value_validates_where_it_reads(self) -> None:
        for position, effect in (("stay", "demote"), ("home", "capture"), ("home", "demote")):
            for strikes in ("target", "blast", "both"):
                assert validate_spec(_spec(position, effect, strikes=strikes)) == []

    @pytest.mark.parametrize("position", ["advance", "swap"])
    def test_another_position_is_refused(self, position: str) -> None:
        spec = _spec(position, strikes="target")
        if position == "advance":
            spec["rules"]["strike"]["switches_sides"] = ["G"]
        errors = validate_spec(spec)
        assert refusal("T1", 'rules.strike.position (not "stay" or "home")') in errors, errors

    def test_choose_is_refused_beside_any_blast(self) -> None:
        errors = validate_spec(_spec("choose", strikes=None, of=["G"]))
        assert refusal("T1", "rules.explode_on_capture") in errors, errors

    def test_a_shot_is_refused_as_inert(self) -> None:
        errors = validate_spec(_spec("stay", "capture", strikes="blast"))
        assert refusal("T1", 'rules.strike (effect "capture", position "stay")') in errors, errors

    def test_beside_contacts_only_none_stands(self) -> None:
        spec = TestContacts._spec()
        assert validate_spec(spec) == []
        spec["rules"]["explode_on_capture"]["strikes"] = "target"
        errors = validate_spec(spec)
        assert refusal("T4", "rules.explode_on_capture.strikes") in errors, errors

    def test_a_choosing_contact_is_refused(self) -> None:
        spec = TestContacts._spec()
        spec["rules"]["contacts"]["position"] = "choose"
        assert "T4" in refused_ids(validate_spec(spec))

    def test_rounds_are_refused(self) -> None:
        spec = _spec("home", "capture", strikes=None, of=["G"])
        spec["rules"]["simultaneous"] = {}
        assert refusal("T1", "rules.strike") in validate_spec(spec)
        spec = TestContacts._spec()
        spec["rules"]["simultaneous"] = {}
        assert refusal("T4", "rules.contacts") in validate_spec(spec)

    def test_a_switch_with_no_strike_is_refused(self) -> None:
        spec = _spec(strikes="target")
        spec["rules"].pop("strike")
        assert any("with no rules.strike strikes nothing" in e for e in validate_spec(spec))

    def test_a_value_out_of_the_four_is_refused(self) -> None:
        errors = validate_spec(_spec(strikes="all"))
        assert any("strikes must be one of" in e for e in errors), errors

    def test_a_blast_on_every_capture_leaves_none_nothing_to_strike(self) -> None:
        errors = validate_spec(_spec(strikes=None))
        assert any("the strike never acts" in e for e in errors), errors
        errors = validate_spec(_spec(strikes=None, by=["R", "G", "P", "W", "K"]))
        assert any("the strike never acts" in e for e in errors), errors
        spec = TestContacts._spec()
        spec["rules"]["explode_on_capture"].pop("of")
        assert any("no link is ever met" in e for e in validate_spec(spec))

    def test_none_written_is_the_game_without_it(self) -> None:
        spec = _spec(strikes=None, of=["G"])
        written = _spec(strikes="none", of=["G"])
        assert spec_hash(written) == spec_hash(spec)


class TestTheSurfaces:
    def test_the_inventory_names_it(self) -> None:
        assert _schema_seventy_eight_uses(_spec(strikes="target")) == {
            "rules.explode_on_capture.strike", "rules.explode_on_capture.strikes",
        }
        assert _schema_seventy_eight_uses(_spec(strikes=None, of=["G"])) == {
            "rules.explode_on_capture.strike",
        }
        assert _schema_seventy_eight_uses(TestContacts._spec()) == {
            "rules.explode_on_capture.contacts",
        }

    def test_describe_says_it(self) -> None:
        assert "A capture that explodes still strikes the piece taken" in describe_spec(
            _spec(strikes="target")
        )
        assert "A capture that explodes is the blast's alone" in describe_spec(
            _spec(strikes=None, of=["G"])
        )

    def test_the_compiled_rule_carries_the_bits(self) -> None:
        for strikes, bits in (("target", 1), ("blast", 2), ("both", 3)):
            explosion = compile_spec(_spec(strikes=strikes)).explosion
            assert explosion is not None and explosion.strikes == bits
        explosion = compile_spec(_spec(strikes=None, of=["G"])).explosion
        assert explosion is not None and explosion.strikes == 0

    def test_fairy_stockfish_refuses_the_strike(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec(strikes="target"))
