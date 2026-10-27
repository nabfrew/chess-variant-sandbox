"""Wild pieces that pounce on prey (generation 39, N3b, `docs/archive/plan-npc.md` §3, §4.3).

An entry with ``pounce: "prey"``: after a move, before any wild piece steps, the pouncer takes a
piece of the player who has just moved that it attacks by its own type's moves, moving onto it;
the prey order is a royal, then the higher value, then the nearer, then the mover's reading
order. The pouncers go in the mover's reading order. A pouncer walks only a route of its own
type. Check for the side to move is the pass test.

Hand positions on an 8x8 board (one per prey-order rule, mirrored pairs under both orientations,
only the mover's pieces, the pass test and the two cases the tables get wrong, the walk off and
back onto a loop, a pouncer on another type's route, the bounty by type, perft against `apply`),
the inert spellings, the refusals and the words; then the oracle (`tests/pounce_oracle.py`)
walked over seeded games of `tests/_pounce_hounds.json`, legal lists, check answers, boards and
phase logs.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from pounce_oracle import PounceOracle

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.tournament import analysis
from chess_variant_sandbox.tournament.play import play_game

ROOT = Path(__file__).resolve().parents[1]
HOUNDS = ROOT / "tests" / "_pounce_hounds.json"
KING = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]
FILES = "abcdefgh"


def _spec(win: str = "royal_capture", orientation: str = "mirror", pounce: Any = "prey") -> dict:
    """Kings on e1 and e8. A hound ``H`` (king's moves) on two routes, a5-a3 and d3-d4-d5; a
    rook-moving lurker ``Z`` that never walks (h1); a wolf ``W`` that does not pounce (f3-f4-f5);
    a stone ``T`` that never moves. Hand positions place what each test needs."""
    def hound(route: list[str]) -> dict[str, Any]:
        npc: dict[str, Any] = {"piece": "H", "route": route}
        if pounce is not None:
            npc["pounce"] = pounce
        return npc

    lurker: dict[str, Any] = {"piece": "Z", "route": ["h1"]}
    if pounce is not None:
        lurker["pounce"] = pounce
    return copy.deepcopy({
        "schema_version": 31,
        "name": "pounce hand positions",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": KING},
            {"id": "Q", "name": "queen", "value": 9,
             "moves": [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}]},
            {"id": "R", "name": "rook", "value": 5,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "dihedral8"}]},
            {"id": "B", "name": "bishop", "value": 3,
             "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "dihedral8"}]},
            {"id": "N", "name": "knight", "value": 3,
             "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
            {"id": "S", "name": "stone", "value": 1, "moves": KING},
            {"id": "T", "name": "totem", "value": 1, "moves": []},
            {"id": "H", "name": "hound", "value": 2, "moves": KING},
            {"id": "Z", "name": "lurker", "value": 4,
             "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "dihedral8"}]},
            {"id": "W", "name": "wolf", "value": 3, "moves": KING},
        ],
        "setup": {"mirror": True, "orientation": orientation, "placements": [
            {"piece": "K", "player": 0, "squares": ["e1"]},
            {"piece": "T", "player": 0, "squares": ["b1"]},
        ]},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 200, "npcs": [
            hound(["a5", "a3"]), hound(["d3", "d4", "d5"]), lurker,
            {"piece": "W", "route": ["f3", "f4", "f5"]},
        ]},
    })


def _engine(**kwargs: Any) -> Engine:
    spec = _spec(**kwargs)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _name(square: int) -> str:
    return FILES[square % 8] + str(square // 8 + 1)


def _at(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> GameState:
    return engine.make_state(placement, side_to_move=side)


def _log(phase: Any) -> list[tuple[str, str, str]]:
    return [(_name(s.origin), _name(s.target), s.kind) for s in phase]


def _play(engine: Engine, state: GameState, text: str) -> tuple[GameState, list[tuple[str, str, str]]]:
    after, phase = engine.apply_with_phase(state, engine.find_move(state, text))
    assert after.key == engine._compute_hash(after), text
    return after, _log(phase)


def _coming(engine: Engine, state: GameState) -> list[tuple[str, str, str]]:
    return _log(coming_phases(engine, state, 1)[0])


# Both kings out of every test's way; white shuffles its king, black its own.
KINGS = {"a1": ("K", 0), "h8": ("K", 1)}


class TestThePreyOrder:
    def test_a_royal_first(self) -> None:
        engine = _engine()
        state = _at(engine, {"h8": ("K", 1), "a1": ("S", 0), "c4": ("H", 2),
                             "d3": ("K", 0), "b3": ("Q", 0)})
        _, log = _play(engine, state, "Sa1-a2")
        assert log == [("c4", "d3", "pounce")]

    def test_then_the_higher_value(self) -> None:
        engine = _engine()
        state = _at(engine, {**KINGS, "c4": ("H", 2), "d3": ("R", 0), "b3": ("Q", 0)})
        _, log = _play(engine, state, "Ka1-b1")
        assert log == [("c4", "b3", "pounce")]

    def test_then_the_nearer(self) -> None:
        engine = _engine()
        state = _at(engine, {"a8": ("K", 1), "c6": ("K", 0), "h1": ("Z", 2),
                             "h3": ("N", 0), "e1": ("B", 0)})
        _, log = _play(engine, state, "Kc6-d6")
        assert log == [("h1", "h3", "pounce")]

    @pytest.mark.parametrize(
        ("pieces", "taken"),
        [(["b3", "d3"], "b3"), (["b5", "d3"], "d3"), (["d5", "b5"], "b5")],
        ids=["the-file", "the-rank-first", "left-on-a-far-rank"],
    )
    def test_then_the_movers_reading_order(self, pieces: list[str], taken: str) -> None:
        engine = _engine()
        state = _at(engine, {**KINGS, "c4": ("H", 2), **{sq: ("N", 0) for sq in pieces}})
        _, log = _play(engine, state, "Ka1-b1")
        assert log == [("c4", taken, "pounce")]

    @pytest.mark.parametrize(
        ("orientation", "pieces", "taken"),
        [
            ("mirror", ["b6", "d6"], "b6"),
            ("rotate", ["b6", "d6"], "d6"),
            ("mirror", ["b4", "d6"], "d6"),
        ],
        ids=["mirror-reads-a-first", "rotate-reads-h-first", "the-second-player-reads-from-rank-8"],
    )
    def test_the_second_player_reads_from_its_own_seat(
        self, orientation: str, pieces: list[str], taken: str
    ) -> None:
        engine = _engine(orientation=orientation)
        state = _at(engine, {**KINGS, "c5": ("H", 2), **{sq: ("N", 1) for sq in pieces}}, side=1)
        _, log = _play(engine, state, "Kh8-g8")
        assert log == [("c5", taken, "pounce")]

    def test_only_the_movers_pieces(self) -> None:
        engine = _engine()
        state = _at(engine, {**KINGS, "c4": ("H", 2), "b3": ("Q", 1), "d3": ("N", 0)})
        _, log = _play(engine, state, "Ka1-b1")
        assert log == [("c4", "d3", "pounce")]
        _, log = _play(engine, _at(engine, {**KINGS, "c4": ("H", 2), "b3": ("Q", 1)}), "Ka1-b1")
        assert log == []

    def test_the_pouncers_go_in_the_movers_reading_order(self) -> None:
        """Two hounds want one knight: the one nearer the mover's first rank goes first, and
        the other finds nothing left."""
        engine = _engine()
        state = _at(engine, {**KINGS, "c5": ("H", 2), "c3": ("H", 2), "c4": ("N", 0)})
        _, log = _play(engine, state, "Ka1-b1")
        assert log == [("c3", "c4", "pounce")]
        state = _at(engine, {**KINGS, "c5": ("H", 2), "c3": ("H", 2), "c4": ("N", 1)}, side=1)
        _, log = _play(engine, state, "Kh8-g8")
        assert log == [("c5", "c4", "pounce")]


def _mirror(square: str, orientation: str) -> str:
    file, rank = FILES.index(square[0]), int(square[1]) - 1
    if orientation == "rotate":
        file = 7 - file
    return FILES[file] + str(8 - rank)


class TestMirroredPositions:
    @pytest.mark.parametrize("orientation", ["mirror", "rotate"])
    def test_mirrored_positions_choose_mirrored_victims(self, orientation: str) -> None:
        engine = _engine(orientation=orientation)
        rng = random.Random(7)
        squares = [f + str(r) for f in FILES for r in range(1, 9)]
        seen = 0
        for _ in range(60):
            chosen = rng.sample(squares, 9)
            placement = {chosen[0]: ("K", 0), chosen[1]: ("K", 1),
                         chosen[2]: ("H", 2), chosen[3]: ("Z", 2)}
            for square in chosen[4:]:
                placement[square] = (rng.choice("QRBNS"), rng.choice((0, 1)))
            image = {_mirror(sq, orientation): (pid, owner if owner == 2 else 1 - owner)
                     for sq, (pid, owner) in placement.items()}
            # The pounces, in order (the steps walk routes the mirror does not carry).
            here = [step for step in _coming(engine, _at(engine, placement, side=0))
                    if step[2] == "pounce"]
            there = [step for step in _coming(engine, _at(engine, image, side=1))
                     if step[2] == "pounce"]
            assert [(_mirror(a, orientation), _mirror(b, orientation), k) for a, b, k in here] == (
                there
            ), (placement, here, there)
            seen += sum(1 for step in here if step[2] == "pounce")
        assert seen > 20


class TestCheck:
    def test_a_royal_in_reach_is_check(self) -> None:
        engine = _engine(win="checkmate")
        state = _at(engine, {"e1": ("K", 0), "e8": ("K", 1), "d7": ("H", 2)}, side=1)
        assert engine.in_check(state)
        # The other player is asked by the tables: no pounce takes a piece of theirs next.
        state = _at(engine, {"e1": ("K", 0), "e8": ("K", 1), "d2": ("H", 2)}, side=1)
        assert not engine.in_check(state) and not engine.in_check(state, 0)

    def test_a_move_leaving_the_royal_in_reach_is_refused(self) -> None:
        engine = _engine(win="checkmate")
        state = _at(engine, {"e1": ("K", 0), "e8": ("K", 1), "c3": ("H", 2)})
        moves = engine.move_texts(state)
        assert "Ke1-d2" not in moves and "Ke1-e2" in moves

    def test_a_rider_unblocked_by_an_earlier_pouncer_is_mate(self) -> None:
        """The hound on h4 blocks the lurker's file to the king on h8; it pounces first (black
        reads from rank 8) onto the totem on g4, the lurker then takes the king. The tables see
        no check, and would call this stalemate."""
        engine = _engine(win="checkmate")
        state = _at(engine, {"h8": ("K", 1), "g4": ("T", 1), "h4": ("H", 2), "h1": ("Z", 2),
                             "f6": ("N", 0), "a7": ("R", 0), "c1": ("K", 0)}, side=1)
        assert engine.legal_moves(state) == []
        assert engine.in_check(state)
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.reason == "checkmate" and outcome.winner == 0

    def test_a_route_step_onto_the_royal_skipped_for_a_pounce_is_not_check(self) -> None:
        """The hound on a5 steps to a3, where the king stands, but pounces first on the stone
        it attacks, and a piece that pounced does not step."""
        engine = _engine(win="checkmate")
        placement = {"e1": ("K", 0), "a3": ("K", 1), "a5": ("H", 2)}
        assert engine.in_check(_at(engine, placement, side=1))
        assert not engine.in_check(_at(engine, {**placement, "b6": ("S", 1)}, side=1))


class TestTheWalk:
    def test_off_its_route_it_stands(self) -> None:
        engine = _engine()
        state = _at(engine, {**KINGS, "e6": ("H", 2)})
        state, log = _play(engine, state, "Ka1-b1")
        assert log == []

    def test_a_pouncer_resumes_its_route(self) -> None:
        engine = _engine()
        state = _at(engine, {**KINGS, "e6": ("H", 2), "d5": ("S", 0)})
        state, log = _play(engine, state, "Ka1-b1")
        assert log == [("e6", "d5", "pounce")]
        state, log = _play(engine, state, "Kh8-g8")
        assert log == [("d5", "d3", "step")]
        state, log = _play(engine, state, "Kb1-a1")
        assert log == [("d3", "d4", "step")]

    def test_a_pouncer_on_another_types_route_stands_in_its_way(self) -> None:
        engine = _engine()
        state = _at(engine, {**KINGS, "f4": ("H", 2), "f3": ("W", 2)})
        _, log = _play(engine, state, "Ka1-b1")
        assert log == []

    def test_a_wolf_behind_a_hound_that_pounced_follows(self) -> None:
        """The hound on f4 leaves the wolf's route by pouncing, so the wolf behind it steps."""
        engine = _engine()
        state = _at(engine, {**KINGS, "f4": ("H", 2), "f3": ("W", 2), "g5": ("S", 0)})
        _, log = _play(engine, state, "Ka1-b1")
        assert log == [("f4", "g5", "pounce"), ("f3", "f4", "step")]

    def test_the_bounty_is_the_types(self) -> None:
        spec = _spec()
        spec["rules"]["scoring"] = {"count": "points"}
        engine = Engine(spec)
        state = _at(engine, {**KINGS, "e6": ("H", 2), "e5": ("R", 0)})
        state, _ = _play(engine, state, "Re5xe6")
        assert state.points == (2.0, 0.0)

    def test_unmake_puts_the_pounce_back(self) -> None:
        engine = _engine(win="checkmate")
        state = _at(engine, {"e1": ("K", 0), "e8": ("K", 1), "c4": ("H", 2), "h1": ("Z", 2),
                             "b3": ("N", 0), "e4": ("B", 1), "g6": ("R", 0)})

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            if engine.terminal(position) is not None:
                return 0
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_coming_shows_the_pounce_and_alternates_the_mover(self) -> None:
        engine = _engine()
        state = _at(engine, {**KINGS, "c4": ("H", 2), "d3": ("N", 0), "e4": ("N", 1)})
        first, then = coming_phases(engine, state, 2)
        assert _log(first) == [("c4", "d3", "pounce")]
        assert _log(then) == [("d3", "e4", "pounce")]


class TestInert:
    def test_a_game_without_it_crosses_no_pounce(self) -> None:
        tables = canonical_tables(compile_spec(_spec(pounce=None)))
        assert not any(key.startswith("pounce_") or key == "npc_piece" for key in tables)
        assert compile_spec(_spec(pounce=None)).pounce is None


class TestRefusals:
    def _errors(self, spec: dict) -> list[str]:
        return [e for e in validate_spec(spec) if "pounce" in e]

    def test_not_a_kind(self) -> None:
        assert any("must be one of" in e for e in self._errors(_spec(pounce="all")))

    @pytest.mark.parametrize(
        ("change", "said"),
        [
            (lambda s: s["rules"].update(forbid_check=True), "beside forbid_check"),
            (lambda s: s["rules"]["npcs"][3].update(respawn=4), "respawn beside a pounce"),
            (lambda s: s["rules"]["npcs"][3].update(turns_back=True), "turns_back beside a pounce"),
            (lambda s: s["rules"]["npcs"][1].update(bounty=7), "disagree on bounty"),
            (lambda s: s["rules"]["npcs"][1].pop("pounce"), "disagree on pounce"),
            (lambda s: s["pieces"][7].update(captures=[]), "takes nothing"),
            (lambda s: s["pieces"][7].update(moves=[{"kind": "leap", "vectors": [[0, 1]]}]),
             "reads differently for the two players"),
            (lambda s: s["pieces"][7]["moves"][0].update(condition={"first_move": True}),
             "has a condition"),
            (lambda s: s["pieces"][8]["moves"].append(
                {"kind": "hop", "vectors": [[1, 0]], "symmetry": "dihedral8", "screen": "enemy"}),
             "screen names an owner"),
        ],
        ids=["forbid-check", "respawn", "turns-back", "bounty", "pounce", "attacks-nothing",
             "forward", "condition", "screen"],
    )
    def test_refused(self, change: Any, said: str) -> None:
        spec = _spec()
        change(spec)
        errors = self._errors(spec)
        assert any(said in e for e in errors), errors

    def test_in_rounds(self) -> None:
        spec = _spec()
        spec["rules"]["simultaneous"] = {}
        assert any("in a game in rounds" in e for e in self._errors(spec))

    def test_fairy_stockfish_refuses_it_with_every_wild_piece(self) -> None:
        with pytest.raises(Untranslatable) as caught:
            translate(_spec())
        assert any("rules.npcs" in reason for reason in caught.value.reasons)


class TestTheWords:
    def test_describe_says_the_pounce_and_the_order(self) -> None:
        text = describe_spec(_spec())
        assert "It pounces: after each move" in text
        assert "a royal piece first, then the most valuable, then the nearest" in text
        assert "When several wild pieces could pounce after one move" in text

    def test_a_game_without_it_says_nothing_of_it(self) -> None:
        assert "pounces" not in describe_spec(_spec(pounce=None))


def _walk(spec: dict, seed: int, plies: int) -> tuple[int, int]:
    """One seeded random game checked by the oracle at every ply: legal list, check, board and
    phase log; the pounces and the checks it saw."""
    oracle = PounceOracle(spec)
    engine = oracle.engine
    rng = random.Random(seed)
    state = engine.initial_state()
    pounces = checks = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        assert moves == oracle.legal_moves(state)
        check = engine.in_check(state)
        assert check == oracle.in_check(state)
        checks += check
        move = rng.choice(moves)
        after = oracle.check(state, move)
        assert after.key == engine._compute_hash(after)
        fuzz.every_npc_stands_on_its_own_route(engine, after)
        pounces += sum(1 for s in engine.apply_with_phase(state, move)[1] if s.kind == "pounce")
        state = after
    return pounces, checks


class TestTheOracle:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_a_seeded_walk(self, seed: int) -> None:
        _walk(json.loads(HOUNDS.read_text(encoding="utf-8")), seed, 80)

    @pytest.mark.slow
    def test_twenty_walks_see_pounces_and_checks(self) -> None:
        spec = json.loads(HOUNDS.read_text(encoding="utf-8"))
        totals = [_walk(spec, seed, 200) for seed in range(100, 120)]
        assert sum(p for p, _ in totals) > 100 and sum(c for _, c in totals) > 10

    def test_a_rotated_walk(self) -> None:
        spec = copy.deepcopy(json.loads(HOUNDS.read_text(encoding="utf-8")))
        spec["setup"]["orientation"] = "rotate"
        assert validate_spec(spec) == []
        _walk(spec, 4, 80)


class TestTheTally:
    def test_the_analysis_counts_what_the_wild_pieces_did(self) -> None:
        engine = Engine(json.loads(HOUNDS.read_text(encoding="utf-8")))
        record = play_game(engine, ("random", "random"), 3)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        wild = analysis._analyse_game(
            engine, evaluator, record, analysis._epsilon_for(engine)
        ).wild
        assert wild is not None
        state, pounces, pounced = engine.initial_state(), 0, [0] * engine.spec.n_types
        for move in record.moves:
            state, phase = engine.apply_with_phase(state, move)
            for step in phase:
                if step.kind == "pounce":
                    pounces += 1
                    pounced[engine.spec.type_of(step.eaten)] += 1
        assert wild["pounces"] == pounces > 0 and wild["pounced"] == pounced
        assert wild["most_standing"] == 6 and wild["returns"] == wild["turns"] == 0

    def test_a_game_without_wild_pieces_has_no_tally(self) -> None:
        engine = Engine(json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8")))
        record = play_game(engine, ("random", "random"), 3)
        found = analysis._analyse_game(
            engine, Evaluator.for_engine(engine, mobility_weight=0.0), record,
            analysis._epsilon_for(engine),
        )
        assert found.wild is None and "wild" not in found.as_dict()
