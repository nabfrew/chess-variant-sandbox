"""Wild pieces that pounce on their attackers (generation 40, N3c, `docs/archive/plan-npc.md` §4.4).

An entry with ``pounce: "attackers"``: after a move, before any wild piece steps, the pouncer takes
a piece of the player who has just moved that attacks it (its own capture moves, the capture
matrix both ways, a frozen or disarmed piece attacking nothing), by the prey order of §4.3. So no
piece it may take can end a move attacking it, a royal that attacks it is in check, and it falls
to a piece that already attacks it when its owner comes to move.

Hand positions on `tests/test_pounce.py`'s board with every pouncer turned to ``"attackers"``: the
take and the prey order, only the mover's pieces, a walk-up eaten, a discovery letting the other
side take it, an immune attacker and one it may not take ignored, a frozen attacker ignored, a
commanded attacker spared; a royal barred from ending a move attacking it, a royal attacked at its
turn as check (the pouncer stepped up), and mate where nothing answers it; perft against `apply`;
the refusals and the words. Then the oracle (`tests/pounce_oracle.py`, extended) walked over seeded
games of `tests/_pounce_guards.json` (hounds on prey, a lurker and a walking sentinel on their
attackers), legal lists, check answers, boards and phase logs.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from test_pounce import KINGS, _at, _play, _spec, _walk

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.tournament import analysis
from chess_variant_sandbox.tournament.play import play_game

ROOT = Path(__file__).resolve().parents[1]
GUARDS = ROOT / "tests" / "_pounce_guards.json"


def _guards(win: str = "royal_capture", **change: Any) -> dict:
    return _spec(win=win, pounce="attackers", **change)


def _engine(spec: dict) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _with_sight(spec: dict, pid: str, sight: dict) -> dict:
    """``spec`` with ``sight`` on ``pid``, and a rook, knight and bishop in each army's setup, so
    a sight has something it could ever bind (the hand positions place their own pieces)."""
    spec = copy.deepcopy(spec)
    spec["setup"]["placements"].append({"piece": "R", "player": 0, "squares": ["c1"]})
    spec["setup"]["placements"].append({"piece": "N", "player": 0, "squares": ["d1"]})
    spec["setup"]["placements"].append({"piece": "B", "player": 0, "squares": ["f1"]})
    for piece in spec["pieces"]:
        if piece["id"] == pid:
            piece.setdefault("sight", []).append(sight)
    return spec


class TestTheTake:
    def test_it_takes_the_piece_that_attacks_it(self) -> None:
        engine = _engine(_guards())
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0)})
        _, log = _play(engine, state, "Ra5-c5")
        assert log == [("e5", "c5", "pounce")]

    def test_a_piece_that_does_not_attack_it_is_safe(self) -> None:
        engine = _engine(_guards())
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0)})
        _, log = _play(engine, state, "Ra5-a6")
        assert log == []

    def test_the_higher_value_first(self) -> None:
        """A rook's move opens the queen's diagonal: both attack it, the queen goes."""
        engine = _engine(_guards())
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0), "b2": ("Q", 0), "c3": ("N", 0)})
        _, log = _play(engine, state, "Nc3-d1")
        assert log == [("e5", "b2", "pounce")]

    def test_then_the_nearer(self) -> None:
        engine = _engine(_guards())
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "e1": ("R", 0), "b4": ("R", 0)})
        _, log = _play(engine, state, "Rb4-b5")
        assert log == [("e5", "b5", "pounce")]
        # Four away each (a5 and e1): the reading order, nearest the mover's first rank, settles it.
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "e1": ("R", 0), "a4": ("R", 0)})
        _, log = _play(engine, state, "Ra4-a5")
        assert log == [("e5", "e1", "pounce")]

    def test_only_the_movers_pieces(self) -> None:
        """Black's rook already attacks it; white's move leaves it standing."""
        engine = _engine(_guards())
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "e8": ("R", 1)})
        _, log = _play(engine, state, "Ka1-b1")
        assert log == []

    def test_a_discovery_lets_the_other_side_take_it(self) -> None:
        """Black's stone steps off the e-file: the white rook now attacks the lurker, untaken
        (it pounces only on the mover's), and white takes it."""
        engine = _engine(_guards())
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "e1": ("R", 0), "e2": ("S", 1)}, side=1)
        state, log = _play(engine, state, "Se2-f1")
        assert log == []
        assert "Re1xe5" in engine.move_texts(state)
        state, log = _play(engine, state, "Re1xe5")
        assert log == [] and engine.spec.type_of(state.board[engine._square_index("e5")]) == 2

    def test_a_pouncer_steps_up_and_is_taken(self) -> None:
        """The hound walks d4 to d5, next to the white stone on e6: it is white's to take."""
        engine = _engine(_guards())
        state = _at(engine, {**KINGS, "d4": ("H", 2), "e6": ("S", 0)}, side=1)
        state, log = _play(engine, state, "Kh8-g8")
        assert log == [("d4", "d5", "step")]
        assert "Se6xd5" in engine.move_texts(state)


class TestWhatAttacks:
    def test_an_immune_attacker_is_ignored(self) -> None:
        """Only a knight may take the lurker, so a rook attacks nothing there."""
        spec = _guards()
        spec["rules"]["npcs"][2]["immune"] = ["N"]
        engine = _engine(spec)
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0), "b4": ("N", 0)})
        _, log = _play(engine, state, "Ra5-c5")
        assert log == []
        _, log = _play(engine, state, "Nb4-c6")
        assert log == [("e5", "c6", "pounce")]

    def test_an_attacker_it_may_not_take_is_ignored(self) -> None:
        spec = _guards()
        spec["pieces"][8]["captures"] = ["Q", "B", "N", "K"]
        engine = _engine(spec)
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0)})
        _, log = _play(engine, state, "Ra5-c5")
        assert log == []

    def test_a_frozen_attacker_is_ignored(self) -> None:
        """Black's bishop freezes along files and ranks: the rook on c5 attacks nothing."""
        freeze = {"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "freeze", "whose": "enemy"}
        engine = _engine(_with_sight(_guards(), "B", freeze))
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0), "c8": ("B", 1)})
        _, log = _play(engine, state, "Ra5-c5")
        assert log == []
        engine = _engine(_guards())
        _, log = _play(engine, _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0),
                                            "c8": ("B", 1)}), "Ra5-c5")
        assert log == [("e5", "c5", "pounce")]

    def test_a_commanded_attacker_is_spared(self) -> None:
        """White's knight commands black's rook into an attack on the lurker: the rook is black's,
        so this phase spares it, and black may take the lurker with it."""
        command = {"vectors": [[1, 2]], "symmetry": "dihedral8", "range": 1, "effect": "command",
                   "whose": "enemy"}
        engine = _engine(_with_sight(_guards(), "N", command))
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a2": ("N", 0), "c3": ("R", 1)})
        commanded = [text for text in engine.move_texts(state) if text.startswith("~Rc3-c5")]
        assert commanded
        state, log = _play(engine, state, commanded[0])
        assert log == []
        assert "Rc5xe5" in engine.move_texts(state)
        # Black's own move there is eaten.
        black = _at(engine, {**KINGS, "e5": ("Z", 2), "a2": ("N", 0), "c3": ("R", 1)}, side=1)
        _, log = _play(engine, black, "Rc3-c5")
        assert log == [("e5", "c5", "pounce")]


class TestCheck:
    def test_a_royal_may_not_end_a_move_attacking_it(self) -> None:
        """The lurker moves like a rook, so the king on e3 does not stand in its reach; it
        attacks the lurker from there, which a prey pouncer would not mind."""
        placement = {"f2": ("K", 0), "h8": ("K", 1), "d4": ("Z", 2)}
        engine = _engine(_guards(win="checkmate"))
        moves = engine.move_texts(_at(engine, placement))
        assert "Kf2-e3" not in moves and "Kf2-f3" in moves
        prey = _engine(_spec(win="checkmate"))
        assert "Kf2-e3" in prey.move_texts(_at(prey, placement))

    def test_a_royal_attacked_at_its_turn_is_check(self) -> None:
        """The hound steps d4 to d5 after black's move, next to the white king: white is in
        check from it, and the tables (no step source on e6) would not say so."""
        engine = _engine(_guards(win="checkmate"))
        state = _at(engine, {"e6": ("K", 0), "h8": ("K", 1), "d4": ("H", 2)}, side=1)
        assert not engine.in_check(state, 0)
        state, log = _play(engine, state, "Kh8-g8")
        assert log == [("d4", "d5", "step")]
        assert engine.in_check(state)
        assert not engine.in_check(state, 1)
        # Every answer takes it or walks out of its reach.
        moves = engine.move_texts(state)
        assert "Ke6xd5" in moves and "Ke6-f6" in moves
        assert not {"Ke6-d6", "Ke6-e5", "Ke6-e6"} & set(moves)

    def test_mate_where_nothing_answers_it(self) -> None:
        """The hound on b2 next to the king in the corner, the black rook on the b-file: every
        king move ends attacking the hound, and taking it walks into the rook."""
        engine = _engine(_guards(win="checkmate"))
        state = _at(engine, {"a1": ("K", 0), "h8": ("K", 1), "b2": ("H", 2), "b8": ("R", 1)})
        assert engine.in_check(state)
        assert engine.legal_moves(state) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.reason == "checkmate" and outcome.winner == 1

    def test_the_other_player_is_asked_by_the_tables(self) -> None:
        engine = _engine(_guards(win="checkmate"))
        state = _at(engine, {"a1": ("K", 0), "h8": ("K", 1), "b2": ("H", 2)}, side=1)
        assert not engine.in_check(state) and not engine.in_check(state, 0)


class TestThePlay:
    def test_perft_matches_apply(self) -> None:
        engine = _engine(_guards(win="checkmate"))
        state = _at(engine, {"e1": ("K", 0), "e8": ("K", 1), "a3": ("H", 2), "d4": ("H", 2),
                             "h1": ("Z", 2), "f3": ("W", 2), "b1": ("R", 0), "g8": ("R", 1),
                             "c6": ("N", 1), "f2": ("B", 0)})

        def count(position: Any, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_an_attackers_pouncer_reads_no_reach(self) -> None:
        compiled = compile_spec(_guards())
        assert compiled.pounce is not None
        assert set(compiled.pounce.kind) == {0, 2}
        assert all(per_type == () for per_type in compiled.pounce.reach)


class TestTheRefusals:
    def test_a_pouncer_that_takes_nothing(self) -> None:
        spec = _guards()
        spec["pieces"][8]["captures"] = []
        assert any("takes nothing" in error for error in validate_spec(spec))

    def test_no_capture_move_of_its_own_is_allowed(self) -> None:
        spec = _guards()
        spec["pieces"][8]["moves"] = [{"kind": "slide", "mode": "move", "vectors": [[1, 0]],
                                       "symmetry": "dihedral8"}]
        assert validate_spec(spec) == []
        spec["rules"]["npcs"][2]["pounce"] = "prey"
        spec["rules"]["npcs"][0]["pounce"] = spec["rules"]["npcs"][1]["pounce"] = "prey"
        assert any("attacks nothing" in error for error in validate_spec(spec))

    def test_an_unknown_kind(self) -> None:
        spec = _guards()
        spec["rules"]["npcs"][2]["pounce"] = "victims"
        assert any("must be one of ('prey', 'attackers')" in error for error in validate_spec(spec))

    def test_two_entries_of_one_type_disagreeing(self) -> None:
        spec = _guards()
        spec["rules"]["npcs"][0]["pounce"] = "prey"
        assert any("disagree on pounce" in error for error in validate_spec(spec))


class TestTheWords:
    def test_describe_says_it(self) -> None:
        text = describe_spec(_guards())
        assert "It pounces on its attackers" in text
        assert "that attacks it" in text

    def test_a_prey_game_does_not(self) -> None:
        assert "pounces on its attackers" not in describe_spec(_spec())


class TestTheTally:
    def test_the_analysis_counts_the_pounces_on_attackers(self) -> None:
        spec = json.loads(GUARDS.read_text(encoding="utf-8"))
        engine = Engine(spec)
        record = play_game(engine, ("random", "random"), 3)
        wild = analysis._analyse_game(
            engine, Evaluator.for_engine(engine, mobility_weight=0.0), record,
            analysis._epsilon_for(engine),
        ).wild
        assert wild is not None
        state, by_guards = engine.initial_state(), 0
        for move in record.moves:
            after, phase = engine.apply_with_phase(state, move)
            by_guards += sum(1 for step in phase if step.kind == "pounce"
                             and engine.spec.piece_ids[engine.spec.type_of(state.board[step.origin])]
                             in {"L", "S"})
            state = after
        assert wild["pounces_on_attackers"] == by_guards > 0
        assert wild["pounces"] > by_guards


class TestTheOracle:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_a_seeded_walk(self, seed: int) -> None:
        _walk(json.loads(GUARDS.read_text(encoding="utf-8")), seed, 80)

    @pytest.mark.slow
    def test_twenty_walks_see_pounces_and_checks(self) -> None:
        spec = json.loads(GUARDS.read_text(encoding="utf-8"))
        totals = [_walk(spec, seed, 200) for seed in range(100, 120)]
        assert sum(p for p, _ in totals) > 100 and sum(c for _, c in totals) > 10
        # The same games' pounces on attackers alone (about six a game at four seeds).
        assert _guard_pounces(spec, range(100, 120), 200) > 60


def _guard_pounces(spec: dict, seeds: Any, plies: int) -> int:
    """The pounces by ``"attackers"`` pouncers in ``_walk``'s games (its seeds, its moves)."""
    engine = Engine(spec)
    guards = {spec["rules"]["npcs"][i]["piece"] for i in range(len(spec["rules"]["npcs"]))
              if spec["rules"]["npcs"][i].get("pounce") == "attackers"}
    ids = list(engine.spec.piece_ids)
    found = 0
    for seed in seeds:
        rng, state = random.Random(seed), engine.initial_state()
        for _ in range(plies):
            if engine.terminal(state) is not None:
                break
            after, phase = engine.apply_with_phase(state, rng.choice(engine.legal_moves(state)))
            found += sum(
                1 for step in phase if step.kind == "pounce"
                and ids[engine.spec.type_of(state.board[step.origin])] in guards
            )
            state = after
    return found

    def test_a_rotated_walk(self) -> None:
        spec = copy.deepcopy(json.loads(GUARDS.read_text(encoding="utf-8")))
        spec["setup"]["orientation"] = "rotate"
        assert validate_spec(spec) == []
        _walk(spec, 4, 80)
