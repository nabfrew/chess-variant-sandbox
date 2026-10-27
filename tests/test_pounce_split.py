"""Wild pieces that split (generation 41, N3d, `docs/archive/plan-npc.md` §4.5), the user's mitosis.

An ``"attackers"`` pouncer with ``split: n``: when it has attackers to take, a copy of itself grows
on each attacker's square in prey order, eating it, while *room* lasts (the cap less the wild
pieces of its type standing, counted live through the phase); past the room the original moves
onto the next attacker as plain ``"attackers"`` does, and the rest are spared.

Hand positions on `tests/test_pounce.py`'s board with every pouncer on ``"attackers"`` and the
rook-moving lurker ``Z`` splitting: §4.5's three room cases (all copied, part copied and the
original moving, none), two lurkers of one type sharing the room, a copy acting from the next
phase, a royal attacker eaten by a copy as check and the move that leaves one refused, the coming
phase, perft against `apply`, the refusals (respawn beside split among them), the words and the
tally. Then `tests/_pounce_hydra.json` walked: the cap held over 300-ply games, and the oracle
(`tests/pounce_oracle.py`, extended) agreeing over 20 walks, legal lists, check answers, boards
and phase logs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from pounce_oracle import PounceOracle
from test_pounce import KINGS, _at, _coming, _play, _spec

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.tournament import analysis
from chess_variant_sandbox.tournament.play import play_game

ROOT = Path(__file__).resolve().parents[1]
HYDRA = ROOT / "tests" / "_pounce_hydra.json"
LURKER = 2  # `_spec`'s npcs[2], the lurker ``Z`` on h1


def _hydras(cap: int = 4, win: str = "royal_capture") -> dict:
    spec = _spec(win=win, pounce="attackers")
    spec["rules"]["npcs"][LURKER]["split"] = cap
    return spec


def _engine(spec: dict) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


# The rook's move from c3 opens the queen's diagonal b2-e5; the rook on a5 attacks e5 already.
# The lurker on e5 has two attackers of the mover's, the queen first (the higher value).
TWO_ATTACKERS = {**KINGS, "e5": ("Z", 2), "a5": ("R", 0), "b2": ("Q", 0), "c3": ("N", 0)}


def _lurkers(engine: Engine, state: Any) -> list[str]:
    z = engine.spec.piece_ids.index("Z")
    return sorted(
        "abcdefgh"[sq % 8] + str(sq // 8 + 1) for sq, value in enumerate(state.board)
        if value and engine.spec.owner_of(value) == 2 and engine.spec.type_of(value) == z
    )


class TestTheRoom:
    def test_all_copied(self) -> None:
        """Room for three: a copy on each attacker, the original stays."""
        engine = _engine(_hydras(4))
        state, log = _play(engine, _at(engine, TWO_ATTACKERS), "Nc3-d1")
        assert log == [("e5", "b2", "copy"), ("e5", "a5", "copy")]
        assert _lurkers(engine, state) == ["a5", "b2", "e5"]

    def test_part_copied_and_the_original_moves(self) -> None:
        """Room for one: a copy on the queen, then the original onto the rook."""
        engine = _engine(_hydras(2))
        state, log = _play(engine, _at(engine, TWO_ATTACKERS), "Nc3-d1")
        assert log == [("e5", "b2", "copy"), ("e5", "a5", "pounce")]
        assert _lurkers(engine, state) == ["a5", "b2"]

    def test_none_and_it_lunges_as_plain_attackers(self) -> None:
        """A second lurker fills the cap: the original takes the queen, the rook is spared."""
        engine = _engine(_hydras(2))
        state, log = _play(engine, _at(engine, {**TWO_ATTACKERS, "h4": ("Z", 2)}), "Nc3-d1")
        assert log == [("e5", "b2", "pounce")]
        assert _lurkers(engine, state) == ["b2", "h4"]
        plain = _engine(_spec(pounce="attackers"))
        _, log = _play(plain, _at(plain, {**TWO_ATTACKERS, "h4": ("Z", 2)}), "Nc3-d1")
        assert log == [("e5", "b2", "pounce")]

    def test_over_the_cap_it_splits_nothing(self) -> None:
        """A hand-built position already over the cap: room is never below 0."""
        engine = _engine(_hydras(2))
        extra = {"h4": ("Z", 2), "h6": ("Z", 2)}
        _, log = _play(engine, _at(engine, {**TWO_ATTACKERS, **extra}), "Nc3-d1")
        assert log == [("e5", "b2", "pounce")]

    def test_no_attacker_no_split(self) -> None:
        engine = _engine(_hydras(4))
        _, log = _play(engine, _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0)}), "Ra5-a6")
        assert log == []


class TestTheLiveCount:
    def test_two_lurkers_share_the_room(self) -> None:
        """Lurkers on e5 and c7, room for two. The rook's move to c5 attacks both, the bishop on
        b2 attacks e5, the knight on a6 attacks c7. The e5 lurker reads first (nearest white's
        first rank) and fills the room with copies on c5 and b2; the c7 lurker, its rook gone,
        finds no room and lunges onto the knight."""
        engine = _engine(_hydras(4))
        placement = {**KINGS, "e5": ("Z", 2), "c7": ("Z", 2), "c2": ("R", 0), "b2": ("B", 0),
                     "a6": ("N", 0)}
        state, log = _play(engine, _at(engine, placement), "Rc2-c5")
        assert log == [("e5", "c5", "copy"), ("e5", "b2", "copy"), ("c7", "a6", "pounce")]
        assert _lurkers(engine, state) == ["a6", "b2", "c5", "e5"]
        # With more room the second lurker copies too.
        engine = _engine(_hydras(5))
        _, log = _play(engine, _at(engine, placement), "Rc2-c5")
        assert log == [("e5", "c5", "copy"), ("e5", "b2", "copy"), ("c7", "a6", "copy")]

    def test_a_copy_acts_from_the_next_phase(self) -> None:
        """A copy grown on b2 stands beside the white king on a1 but is not a pouncer this phase;
        the next white move that leaves an attacker on it is eaten as any other."""
        engine = _engine(_hydras(4))
        state, log = _play(engine, _at(engine, TWO_ATTACKERS), "Nc3-d1")
        assert ("e5", "b2", "copy") in log
        state, log = _play(engine, state, "Kh8-g8")
        assert log == []
        # The king on a1 attacks the copy on b2: a white king move that keeps it there is eaten.
        state, log = _play(engine, state, "Nd1-c3")
        assert ("b2", "a1", "pounce") in log or ("b2", "a1", "copy") in log


class TestCheck:
    def test_a_royal_attacker_eaten_by_a_copy_is_check(self) -> None:
        """The white king beside the lurker, with room: the pass test grows a copy on the king,
        so white is in check, and every legal move takes it off or walks away."""
        engine = _engine(_hydras(4, win="checkmate"))
        state = _at(engine, {"e4": ("K", 0), "h8": ("K", 1), "e5": ("Z", 2), "a5": ("R", 0)})
        assert engine.in_check(state)
        assert not engine.in_check(state, 1)
        moves = engine.move_texts(state)
        assert moves and not {"Ke4-d4", "Ke4-f4", "Ke4-d5", "Ke4-f5", "Ra5-a6"} & set(moves)
        assert "Ke4-e3" in moves and "Ke4xe5" in moves

    def test_a_copy_on_a_royal_ends_a_royal_capture_game(self) -> None:
        engine = _engine(_hydras(4))
        state = _at(engine, {"e4": ("K", 0), "h8": ("K", 1), "e5": ("Z", 2), "a5": ("R", 0)})
        after, log = _play(engine, state, "Ra5-a6")
        assert log[0] == ("e5", "e4", "copy")
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == 1

    def test_mate_where_nothing_answers_it(self) -> None:
        """The king in the corner beside the lurker, the black rook on the b-file: every king move
        ends attacking it, and taking it walks into the rook."""
        engine = _engine(_hydras(4, win="checkmate"))
        state = _at(engine, {"a1": ("K", 0), "h8": ("K", 1), "b2": ("Z", 2), "b8": ("R", 1),
                             "h2": ("R", 1)})
        assert engine.in_check(state)
        assert engine.legal_moves(state) == []


class TestThePlay:
    def test_the_coming_phase_shows_the_copies(self) -> None:
        engine = _engine(_hydras(2))
        state = _at(engine, {**TWO_ATTACKERS, "c3": ("N", 1)}, side=1)
        # Black moves; then white's phase. Here: the knight is black's, so the diagonal is shut;
        # the coming phase after black's move is black's, and nothing of black's attacks it.
        assert _coming(engine, state) == []
        state = _at(engine, {**KINGS, "e5": ("Z", 2), "a5": ("R", 0), "b2": ("Q", 0)})
        assert _coming(engine, state) == [("e5", "b2", "copy"), ("e5", "a5", "pounce")]

    def test_perft_matches_apply(self) -> None:
        engine = _engine(_hydras(4, win="checkmate"))
        state = _at(engine, {"e1": ("K", 0), "e8": ("K", 1), "a3": ("H", 2), "d4": ("Z", 2),
                             "f3": ("W", 2), "b1": ("R", 0), "g8": ("R", 1), "c6": ("N", 1),
                             "f2": ("B", 0), "a7": ("Q", 0)})

        def count(position: Any, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_the_undo_puts_back_only_the_eaten(self) -> None:
        engine = _engine(_hydras(4))
        state = _at(engine, TWO_ATTACKERS)
        oracle = PounceOracle(_hydras(4))
        after = oracle.check(state, engine.find_move(state, "Nc3-d1"))
        assert after.key == engine._compute_hash(after)


class TestTheTables:
    def test_a_game_without_it_crosses_no_split(self) -> None:
        assert not canonical_tables(compile_spec(_spec(pounce="attackers"))).get("pounce_split")
        assert canonical_tables(compile_spec(_hydras(4)))["pounce_split"]
        assert compile_spec(_spec(pounce="attackers")).pounce.split == ()

    def test_the_cap_is_the_types(self) -> None:
        compiled = compile_spec(_hydras(5))
        z = compiled.piece_ids.index("Z")
        assert compiled.pounce is not None
        assert compiled.pounce.split[z] == 5 and sum(compiled.pounce.split) == 5


class TestTheRefusals:
    def _errors(self, spec: dict) -> list[str]:
        return validate_spec(spec)

    @pytest.mark.parametrize("cap", [0, 1, 9, True, "6"])
    def test_out_of_range(self, cap: Any) -> None:
        spec = _hydras()
        spec["rules"]["npcs"][LURKER]["split"] = cap
        assert any("split must be a whole number from 2 to 8" in e for e in self._errors(spec))

    def test_without_pounce_attackers(self) -> None:
        spec = _spec(pounce="prey")
        spec["rules"]["npcs"][LURKER]["split"] = 3
        assert any('without pounce "attackers"' in e for e in self._errors(spec))
        spec = _spec(pounce=None)
        spec["rules"]["npcs"][LURKER]["split"] = 3
        assert any('without pounce "attackers"' in e for e in self._errors(spec))

    def test_below_the_types_starts(self) -> None:
        spec = _spec(pounce="attackers")
        for npc in spec["rules"]["npcs"][:2]:
            npc["split"] = 2
        assert self._errors(spec) == []
        spec["rules"]["npcs"][0]["mirror"] = True
        assert any("below the 3 wild 'H'" in e for e in self._errors(spec))

    def test_beside_drops_that_fill_a_hand(self) -> None:
        spec = _hydras()
        spec["rules"]["drops"] = {"from_captures": True}
        assert any("beside drops that fill a hand from captures" in e for e in self._errors(spec))

    def test_respawn_beside_split(self) -> None:
        spec = _hydras()
        spec["rules"]["npcs"][LURKER]["respawn"] = 4
        errors = self._errors(spec)
        assert any("respawn beside split" in e for e in errors), errors
        assert any("respawn beside a pounce" in e for e in errors)

    def test_two_entries_of_one_type_disagreeing(self) -> None:
        spec = _spec(pounce="attackers")
        spec["rules"]["npcs"][0]["split"] = 3
        assert any("disagree on split" in e for e in self._errors(spec))


class TestTheWords:
    def test_describe_says_it(self) -> None:
        text = describe_spec(_hydras(5))
        assert "It splits" in text and "at most 5" in text

    def test_a_guard_game_does_not(self) -> None:
        assert "It splits" not in describe_spec(_spec(pounce="attackers"))


class TestTheTally:
    def test_the_analysis_counts_the_copies(self) -> None:
        spec = json.loads(HYDRA.read_text(encoding="utf-8"))
        engine = Engine(spec)
        for seed in range(1, 6):
            record = play_game(engine, ("random", "random"), seed)
            wild = analysis._analyse_game(
                engine, Evaluator.for_engine(engine, mobility_weight=0.0), record,
                analysis._epsilon_for(engine),
            ).wild
            assert wild is not None
            state, copies = engine.initial_state(), 0
            for move in record.moves:
                state, phase = engine.apply_with_phase(state, move)
                copies += sum(1 for step in phase if step.kind == "copy")
            assert wild["copies"] == copies
            if copies:
                return
        pytest.fail("no copies in five random games")


def _split_walk(spec: dict, seed: int, plies: int, oracle: bool) -> tuple[int, int]:
    """One seeded random game; every ply the hydras standing held to the cap, and with
    ``oracle`` the legal list, check, board and phase log held to `PounceOracle`'s. The copies
    grown and the most hydras standing."""
    checker = PounceOracle(spec)
    engine = checker.engine
    cap = max(checker.split.values())
    hydra = engine.spec.piece_ids.index("Y")
    rng, state = random.Random(seed), engine.initial_state()
    copies = most = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        if oracle:
            assert moves == checker.legal_moves(state)
            assert engine.in_check(state) == checker.in_check(state)
        move = rng.choice(moves)
        after, phase = engine.apply_with_phase(state, move)
        if oracle:
            assert checker.check(state, move) == after
        assert after.key == engine._compute_hash(after)
        fuzz.every_npc_stands_on_its_own_route(engine, after)
        heads = sum(1 for v in after.board if v and engine.spec.owner_of(v) == 2
                    and engine.spec.type_of(v) == hydra)
        assert heads <= cap
        copies += sum(1 for step in phase if step.kind == "copy")
        most = max(most, heads)
        state = after
    return copies, most


class TestTheWalk:
    @pytest.mark.parametrize("seed", [1, 2])
    def test_a_seeded_walk(self, seed: int) -> None:
        _split_walk(json.loads(HYDRA.read_text(encoding="utf-8")), seed, 80, oracle=True)

    @pytest.mark.slow
    def test_the_cap_holds_over_300_plies(self) -> None:
        """Games of up to 300 plies (each ends where the rules end it); the cap is reached and
        never passed."""
        spec = json.loads(HYDRA.read_text(encoding="utf-8"))
        spec["rules"]["max_plies"] = 300
        spec["rules"]["no_progress_draw"] = 300
        spec["rules"]["repetition_draw"] = "never"
        assert validate_spec(spec) == []
        results = [_split_walk(spec, seed, 300, oracle=False) for seed in range(20)]
        assert max(most for _, most in results) == 6
        assert sum(copies for copies, _ in results) > 40

    @pytest.mark.slow
    def test_the_oracle_agrees_over_twenty_walks(self) -> None:
        spec = json.loads(HYDRA.read_text(encoding="utf-8"))
        results = [_split_walk(spec, seed, 200, oracle=True) for seed in range(100, 120)]
        assert sum(copies for copies, _ in results) > 40
        assert max(most for _, most in results) == 6
