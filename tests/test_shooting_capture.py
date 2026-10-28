"""A1: the shooting capture, `rules.strike.by` (`docs/archive/plan-pieces-moves.md` §6).

A list on the existing strike, copying the explosion's `by`: only a capture made by a listed
type strikes, and every other capture is an ordinary one. Rifle given to the bishops is the
arrow: a bishop's capture removes its target and the bishop stays where it stood. It is a row
mask in `compile._compile_strike`, so the core and the bridge are unchanged; this file holds
the compiled rows, the refusals in words, the hand positions H1 to H4 of §6.1 and the rest
probe on the real compiler, the readings of §6.3 (every position, en passant, promotion, a
shooting cannon), the twin-type contact spelling as an independent second spelling, and the
surfaces (`describe`, distance, Fairy-Stockfish, the collector, `_forget_piece`).

The inertness proof (every committed and archived hash and table unmoved) is
`test_schema_nine.py`'s FROZEN pins and the perft-bearing files, which this rule leaves alone.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import REMOVED_IN_PLACE, compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import distance, features
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.transform import drop_piece
from tests.feature_inventory import _schema_thirty_five_uses

ROOT = Path(__file__).resolve().parents[1]
RIFLE = {"effect": "capture", "position": "stay", "immune": ["K"]}
ARCHERS = {**RIFLE, "by": ["B"]}


def _chess() -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


def _with(strike: dict[str, Any] | None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = copy.deepcopy(base or _chess())
    if strike is None:
        spec["rules"].pop("strike", None)
    else:
        spec["rules"]["strike"] = copy.deepcopy(strike)
    return spec


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _position(spec: dict[str, Any], white: dict[str, str], black: dict[str, str]) -> dict[str, Any]:
    """The spec's rules on a hand-built position, white to move: ``{square: piece id}``."""
    built = copy.deepcopy(spec)
    built["rules"].pop("castling", None)
    if "P" not in {*white.values(), *black.values()}:
        built["rules"].pop("promotion", None)  # a promotion about a piece never placed
        built["rules"].pop("en_passant", None)
    placements = [
        {"piece": piece, "player": player, "squares": [square]}
        for player, side in ((0, white), (1, black))
        for square, piece in side.items()
    ]
    built["setup"] = {"mirror": False, "placements": placements}
    assert validate_spec(built) == [], validate_spec(built)
    return built


def _between(engine: Engine, state: Any, origin: str, target: str) -> list[int]:
    wanted = (_square(origin), _square(target))
    return [m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == wanted]


def _one(engine: Engine, state: Any, origin: str, target: str) -> Any:
    moves = _between(engine, state, origin, target)
    assert len(moves) == 1, (origin, target, moves)
    after = engine.apply(state, moves[0])
    assert after.key == engine._compute_hash(after), "hash drifted"
    return after


def _at(engine: Engine, state: Any, square: str) -> tuple[int, str] | None:
    value = state.board[_square(square)]
    if value == 0:
        return None
    return engine.spec.owner_of(value), engine.spec.piece_ids[engine.spec.type_of(value)]


class TestTheCompiledRows:
    def test_only_the_listed_row_strikes(self) -> None:
        compiled = compile_spec(_with(ARCHERS))
        rule = compiled.strike
        assert rule is not None and rule.compact_to is None
        n, ids = rule.n_types, compiled.piece_ids
        bishop, king = ids.index("B"), ids.index("K")
        for mover in range(n):
            row = rule.onto_to[mover * n : (mover + 1) * n]
            if mover == bishop:
                assert [t for v, t in enumerate(row) if v != king] == [REMOVED_IN_PLACE] * (n - 1)
                assert row[king] == -1
            else:
                assert row == (-1,) * n

    def test_rifle_without_by_compiles_as_it_did(self) -> None:
        """No `by` is every type: the compact array, the tables `archive/rifle.json` has."""
        rifle = json.loads((ROOT / "archive" / "rifle.json").read_text(encoding="utf-8"))
        assert "by" not in rifle["rules"]["strike"]
        assert compile_spec(rifle).strike.compact_to is not None

    def test_the_archers_are_a_game_of_their_own(self) -> None:
        archers, rifle = _with(ARCHERS), _with(RIFLE)
        assert spec_hash(archers) != spec_hash(rifle) != spec_hash(_chess())
        assert canonical_tables(compile_spec(archers)) != canonical_tables(compile_spec(rifle))


class TestTheRefusals:
    """§6.3's refusals 1 to 4, each in its words (`test_plain_words.py` holds the designer's)."""

    @pytest.mark.parametrize(
        ("strike", "words"),
        [
            ({**RIFLE, "by": []}, "a strike nobody makes is no rule"),
            ({**RIFLE, "by": ["Z"]}, "rules.strike.by entry 'Z' is not a declared piece"),
            ({**RIFLE, "by": ["B", "B"]}, "rules.strike.by names a piece twice"),
            ({**RIFLE, "by": ["P", "N", "B", "R", "Q", "K"]},
             "rules.strike.by names every piece that captures"),
            ({"effect": "capture", "by": ["B"]}, "which is no strike at all"),
        ],
        ids=["empty", "undeclared", "twice", "every capturer", "inert"],
    )
    def test_each_is_refused_in_words(self, strike: dict[str, Any], words: str) -> None:
        errors = validate_spec(_with(strike))
        assert any(words in error for error in errors), errors

    def test_beside_switches_sides_is_refused(self) -> None:
        strike = {"effect": "capture", "immune": ["K"], "by": ["B"], "switches_sides": ["B"]}
        errors = validate_spec(_with(strike))
        assert "T10" in refused_ids(errors), errors

    def test_a_listed_piece_that_never_captures_is_refused(self) -> None:
        spec = _with(ARCHERS)
        next(p for p in spec["pieces"] if p["id"] == "B")["captures"] = []
        errors = validate_spec(spec)
        assert any("the bishop never captures anything the strike reaches" in e for e in errors)

    def test_a_listed_piece_whose_captures_reach_only_immune_types_is_refused(self) -> None:
        spec = _with(ARCHERS)
        next(p for p in spec["pieces"] if p["id"] == "B")["captures"] = ["K"]
        assert any("never fires" in e for e in validate_spec(spec))

    def test_a_piece_that_only_walks_is_refused(self) -> None:
        spec = _with(ARCHERS)
        for move in next(p for p in spec["pieces"] if p["id"] == "B")["moves"]:
            move["mode"] = "move"
        assert any("never fires" in e for e in validate_spec(spec))

    def test_every_capturer_but_one_is_a_game(self) -> None:
        assert validate_spec(_with({**RIFLE, "by": ["P", "N", "B", "R", "Q"]})) == []

    def test_the_existing_refusals_still_reach_a_strike_with_by(self) -> None:
        royal = _with({"effect": "capture", "position": "stay", "by": ["B"]})
        assert any("not in rules.strike.immune" in e for e in validate_spec(royal))
        # Beside a hand an archer is allowed since SR2 (T2): its victim goes to the hand
        # (`tests/test_removing_strike_hand.py`); a demoting chain beside one is not.
        drops = _with(ARCHERS)
        drops["rules"]["drops"] = {"from_captures": True}
        assert not any("rules.strike and rules.drops" in e for e in validate_spec(drops))


class TestTheHandPositions:
    """H1 to H4 and the rest probe of §6.1, on the real compiler."""

    def test_h1_the_bishop_shoots_and_the_knight_takes(self) -> None:
        spec = _position(_with(ARCHERS), {"c1": "B", "g4": "N", "h1": "K"},
                         {"e3": "N", "f4": "P", "h8": "K"})
        engine = Engine(spec)
        start = engine.initial_state()
        shot = _one(engine, start, "c1", "e3")
        assert _at(engine, shot, "c1") == (0, "B") and _at(engine, shot, "e3") is None
        assert "(stays)" in engine.describe_move(start, _between(engine, start, "c1", "e3")[0])
        taken = _one(engine, start, "g4", "e3")
        assert _at(engine, taken, "e3") == (0, "N") and _at(engine, taken, "g4") is None

    def test_h2_a_pinned_shooter_shoots_off_its_pin_line(self) -> None:
        spec = _position(_with(ARCHERS), {"e2": "B", "e1": "K"},
                         {"e6": "R", "c4": "N", "g4": "N", "h8": "K"})
        engine = Engine(spec)
        start = engine.initial_state()
        for target in ("c4", "g4"):
            after = _one(engine, start, "e2", target)
            assert _at(engine, after, "e2") == (0, "B") and _at(engine, after, target) is None
        assert _between(engine, start, "e2", "d3") == []  # a move off the line is still pinned

    def test_h3_a_shot_answers_a_check_only_by_removing_the_checker(self) -> None:
        spec = _position(_with(ARCHERS), {"b5": "B", "e1": "K"}, {"e8": "R", "h8": "K"})
        engine = Engine(spec)
        start = engine.initial_state()
        shot = _one(engine, start, "b5", "e8")
        assert _at(engine, shot, "b5") == (0, "B") and _at(engine, shot, "e8") is None
        block = _one(engine, start, "b5", "e2")
        assert _at(engine, block, "e2") == (0, "B")

    def test_h4_a_shot_may_not_open_a_line_onto_its_own_royal(self) -> None:
        placed = ({"c3": "B", "e1": "K"}, {"e5": "N", "e8": "R", "a8": "K"})
        archers = Engine(_position(_with(ARCHERS), *placed))
        assert _between(archers, archers.initial_state(), "c3", "e5") == []
        chess = Engine(_position(_with(None), *placed))
        assert len(_between(chess, chess.initial_state(), "c3", "e5")) == 1

    def test_a_shot_is_the_shooters_move_and_starts_its_rest(self) -> None:
        base = _with(ARCHERS)
        next(p for p in base["pieces"] if p["id"] == "B")["rest"] = 1
        spec = _position(base, {"c1": "B", "h1": "K", "a2": "P"}, {"e3": "N", "h8": "K"})
        engine = Engine(spec)
        shot = _one(engine, engine.initial_state(), "c1", "e3")
        reply = _one(engine, shot, "h8", "g8")
        assert all(move_from(m) != _square("c1") for m in engine.legal_moves(reply))


# Every strike form under `by`, so a listed capture plays the rule and an unlisted one chess.
_FORMS = {
    "rifle": RIFLE,
    "home": {"effect": "capture", "position": "home", "immune": ["K"]},
    "demote stay": {"effect": "demote", "position": "stay", "immune": ["K"],
                    "chain": [{"from": "R", "to": "N"}]},
    "demote swap": {"effect": "demote", "position": "swap", "immune": ["K"],
                    "chain": [{"from": "R", "to": "N"}]},
    "demote choose": {"effect": "demote", "position": "choose", "immune": ["K"],
                      "chain": [{"from": "R", "to": "N"}],
                      "landing": {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}},
}


def _outcomes(
    engine: Engine, state: Any, origin: str, target: str, prelude: tuple[tuple[str, str], ...] = ()
) -> list[tuple[int, ...]]:
    for before, after in prelude:
        state = _one(engine, state, before, after)
    return sorted(tuple(engine.apply(state, m).board) for m in _between(engine, state, origin, target))


# A quiet move each, so the victim stands off its home square and circe has somewhere to send it.
_PRELUDE = (("h1", "g1"), ("e4", "e3"))


class TestEveryPositionReadsTheMask:
    """§6.2: the row mask does not care what the cell holds."""

    @pytest.mark.parametrize("form", sorted(_FORMS))
    def test_a_listed_capture_strikes_and_an_unlisted_one_advances(self, form: str) -> None:
        placed = ({"c1": "B", "g4": "N", "h1": "K"}, {"e4": "R", "f4": "P", "h7": "K"})
        narrowed = Engine(_position(_with({**_FORMS[form], "by": ["B"]}), *placed))
        everyone = Engine(_position(_with(_FORMS[form]), *placed))
        plain = Engine(_position(_with(None), *placed))
        start = narrowed.initial_state()
        listed = _outcomes(narrowed, start, "c1", "e3", _PRELUDE)
        assert listed and listed == _outcomes(everyone, everyone.initial_state(), "c1", "e3", _PRELUDE)
        assert listed != _outcomes(plain, plain.initial_state(), "c1", "e3", _PRELUDE)
        unlisted = _outcomes(narrowed, start, "g4", "e3", _PRELUDE)
        assert unlisted == _outcomes(plain, plain.initial_state(), "g4", "e3", _PRELUDE)
        assert len(unlisted) == 1


class TestTheOtherReadings:
    def test_a_listed_pawn_shoots_in_passing_and_stays(self) -> None:
        def after(by: list[str]) -> tuple[Engine, Any]:
            spec = _position(_with({**RIFLE, "by": by}),
                             {"e5": "P", "e1": "K", "c1": "B"}, {"d7": "P", "e8": "K", "a6": "N"})
            engine = Engine(spec)
            state = _one(engine, engine.initial_state(), "c1", "d2")
            state = _one(engine, state, "d7", "d5")
            return engine, _one(engine, state, "e5", "d6")

        engine, shot = after(["P"])
        assert _at(engine, shot, "e5") == (0, "P")
        assert _at(engine, shot, "d5") is None and _at(engine, shot, "d6") is None
        engine, taken = after(["B"])
        assert _at(engine, taken, "d6") == (0, "P") and _at(engine, taken, "d5") is None

    def test_a_listed_pawn_promotes_where_it_stands_as_rifle_does(self) -> None:
        placed = ({"b7": "P", "e1": "K", "h2": "B"}, {"a8": "R", "e8": "K"})
        narrowed = Engine(_position(_with({**RIFLE, "by": ["P"]}), *placed))
        rifle = Engine(_position(_with(RIFLE), *placed))
        listed = _outcomes(narrowed, narrowed.initial_state(), "b7", "a8")
        assert listed and listed == _outcomes(rifle, rifle.initial_state(), "b7", "a8")
        for board in listed:
            assert board[_square("a8")] == 0 and board[_square("b7")] != 0

    def test_a_pawn_that_promotes_to_a_shooter_shoots_from_its_next_capture(self) -> None:
        spec = _position(_with(ARCHERS), {"b7": "P", "e1": "K"},
                         {"a8": "N", "d5": "P", "h8": "K"})
        engine = Engine(spec)
        start = engine.initial_state()
        promote = [
            m for m in _between(engine, start, "b7", "a8")
            if engine.spec.piece_ids[engine.spec.type_of(engine.apply(start, m).board[_square("a8")])] == "B"
        ]
        assert len(promote) == 1
        state = engine.apply(start, promote[0])
        assert _at(engine, state, "a8") == (0, "B") and _at(engine, state, "b7") is None
        state = _one(engine, state, "h8", "g8")
        shot = _one(engine, state, "a8", "d5")
        assert _at(engine, shot, "a8") == (0, "B") and _at(engine, shot, "d5") is None

    def test_a_shooting_cannon_stays_behind_its_screen(self) -> None:
        base = _with({**RIFLE, "by": ["R"]})
        rook = next(p for p in base["pieces"] if p["id"] == "R")
        rook["moves"] = [
            {"kind": "slide", "vectors": [[0, 1]], "symmetry": "rotate4", "mode": "move"},
            {"kind": "hop", "vectors": [[0, 1]], "symmetry": "rotate4", "mode": "capture"},
        ]
        spec = _position(base, {"a1": "R", "a4": "P", "e1": "K"}, {"a6": "N", "e8": "K"})
        engine = Engine(spec)
        shot = _one(engine, engine.initial_state(), "a1", "a6")
        assert _at(engine, shot, "a1") == (0, "R") and _at(engine, shot, "a6") is None
        assert _at(engine, shot, "a4") == (0, "P")


def _twin() -> dict[str, Any]:
    """The archers in the language they had before `by` (§6.1's `c4`): two bishop types, `B`
    and `B2`, under contacts at `stay`, each turning into the other when it captures. A
    capture by either removes its target and leaves the bishop where it stood. `B2` is
    appended last, so every other type keeps its index and a move word means the same."""
    spec = _chess()
    bishop = next(p for p in spec["pieces"] if p["id"] == "B")
    spec["pieces"].append({**copy.deepcopy(bishop), "id": "B2", "name": "bishop two"})
    victims = ("P", "N", "B", "B2", "R", "Q")
    spec["rules"]["contacts"] = {
        "position": "stay",
        "immune": ["K"],
        "links": [
            {"mover": mover, "onto": onto, "mover_to": other}
            for mover, other in (("B", "B2"), ("B2", "B"))
            for onto in victims
        ],
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _agree(seed: int, plies: int = 160) -> tuple[int, int]:
    """Play one seeded random game in the archers and the twin side by side, comparing the
    legal move words at every node and the boards with `B2` read as `B`."""
    archers, twin = Engine(_with(ARCHERS)), Engine(_twin())
    bishop, second = twin.spec.piece_ids.index("B"), twin.spec.piece_ids.index("B2")
    stride = len(twin.spec.piece_ids)

    def plain(board: Any, n_types: int, rename: bool) -> tuple[int, ...]:
        out = []
        for value in board:
            if value == 0:
                out.append(0)
                continue
            owner, kind = (value - 1) // n_types, (value - 1) % n_types
            if rename and kind == second:
                kind = bishop
            out.append((owner, kind))
        return tuple(out)

    rng = random.Random(seed)
    a, b = archers.initial_state(), twin.initial_state()
    nodes = shots = 0
    for _ in range(plies):
        moves = archers.legal_moves(a)
        assert sorted(moves) == sorted(twin.legal_moves(b)), (seed, nodes)
        assert plain(a.board, len(archers.spec.piece_ids), False) == plain(b.board, stride, True)
        nodes += 1
        if not moves or archers.terminal(a, moves) is not None or twin.terminal(b) is not None:
            break
        taking = [m for m in moves if a.board[move_to(m)]]
        move = rng.choice(taking) if taking and rng.random() < 0.5 else rng.choice(moves)
        mover = archers.spec.piece_ids[archers.spec.type_of(a.board[move_from(m := move)])]
        shots += mover == "B" and a.board[move_to(m)] != 0
        a, b = archers.apply(a, move), twin.apply(b, move)
    return nodes, shots


class TestTheTwinTypeSpellingAgrees:
    """§6.7 V1: the core's legal list under `by` equals the contact spelling's, every node."""

    @pytest.mark.parametrize("seed", range(3))
    def test_three_games_agree(self, seed: int) -> None:
        nodes, _shots = _agree(seed)
        assert nodes > 20

    @pytest.mark.slow
    def test_twenty_games_agree_and_the_bishops_shoot(self) -> None:
        total_nodes = total_shots = 0
        for seed in range(20):
            nodes, shots = _agree(seed)
            total_nodes += nodes
            total_shots += shots
        assert total_nodes > 1000 and total_shots > 20


class TestTheOtherSurfaces:
    def test_the_description_says_who_shoots(self) -> None:
        text = describe_spec(_with(ARCHERS))
        assert "A bishop shoots" in text and "stays where it stood" in text
        assert "Every other piece captures as usual" in text
        demote = describe_spec(_with({**_FORMS["demote stay"], "by": ["B"]}))
        assert "Only a capture made by a bishop strikes" in demote

    def test_it_is_apart_from_rifle_and_from_chess(self) -> None:
        archers, rifle = _with(ARCHERS), _with(RIFLE)
        assert features(rifle)["strikers_share"] == 1.0
        assert features(archers)["strikers_share"] == pytest.approx(1 / 6)  # the king captures too
        assert features(_chess())["strikers_share"] == 0.0
        assert distance(archers, rifle) > 0 and distance(archers, _chess()) > 0

    def test_fairy_refuses_it_by_name(self) -> None:
        with pytest.raises(Untranslatable, match="strike"):
            translate(_with(ARCHERS))

    def test_the_collector_names_it(self) -> None:
        assert _schema_thirty_five_uses(_with(ARCHERS)) == {"rules.strike.by"}
        assert _schema_thirty_five_uses(_with(RIFLE)) == set()
        assert _schema_thirty_five_uses(_chess()) == set()

    def test_removing_the_only_shooter_takes_the_strike(self) -> None:
        spec = drop_piece(_with(ARCHERS), "B")
        assert "strike" not in spec["rules"]
        two = drop_piece(_with({**RIFLE, "by": ["B", "N"]}), "B")
        assert two["rules"]["strike"]["by"] == ["N"]
