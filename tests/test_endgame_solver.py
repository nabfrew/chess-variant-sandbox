"""The forward endgame solver (N1, `rust/src/endgame.rs`, `docs/reference/no-forced-mate.md`).

`engine/endgame.py::solve_endgame`, over `Core.solve_endgame`, labels every position of a material
set won, lost or drawn for the side
to move, with its distance in plies, from the game's own moves and endings, and follows every move
into the sets it leads to. The fast tests run on a 4x4 board built here, where every won or lost
position at seven plies or fewer is checked against the analysis probe (`Core.root_scores`, the
declared test oracle). The tests named for chess check published chess numbers (CLAUDE.md, rule
5) and are slow: king and rook mate in 16 moves at most, king and queen in 10, a pawn in 28, two
knights never force mate, and FIDE's dead sets are drawn in every game where the rule is live.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.evaluation import WIN_SCORE
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import INSUFFICIENT_LEAPER, INSUFFICIENT_RIDER
from chess_variant_sandbox.engine.endgame import solve_endgame
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import eval_values

ROOT = Path(__file__).resolve().parents[1]


def _plies(moves: int) -> int:
    """A mate in `moves` moves of one ply each is `2 * moves - 1` plies."""
    return 2 * moves - 1


def _small() -> dict[str, Any]:
    """A 4x4 game of a royal stepper, a rider and a leaper (renamed, so nothing reads as chess)."""
    return {
        "schema_version": 31,
        "name": "small endgame board",
        "board": {"width": 4, "height": 4},
        "pieces": [
            {"id": "A", "value": 5, "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}]},
            {"id": "J", "value": 3, "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}]},
            {
                "id": "W",
                "value": 1,
                "moves": [
                    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
                ],
            },
            {
                "id": "G",
                "value": 0,
                "royal": True,
                "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}],
            },
        ],
        "setup": {"mirror": True, "placements": [{"piece": "G", "squares": ["a1"]}, {"piece": "A", "squares": ["b1"]}, {"piece": "W", "squares": ["c2"]}]},
        "rules": {"promotion": {"piece": "W", "rank": "last", "to": ["A"]}, "win": "checkmate", "stalemate": "draw", "max_plies": 200, "repetition_draw": 3, "no_progress_draw": 100},
    }


@cache
def _small_engine(no_progress: int = 100) -> Engine:
    spec = _small()
    spec["rules"]["no_progress_draw"] = no_progress
    return Engine(spec)


@cache
def _chess() -> Engine:
    return Engine(load_spec(ROOT / "rulesets" / "chess.json"))


def _probe(engine: Engine, state: GameState, depth: int) -> float:
    """The probe's value of `state` for its side to move, searched `depth` plies (a forced
    move, which the probe does not score, is played and its child asked)."""
    legal = engine.legal_moves(state)
    if len(legal) >= 2:
        return max(engine.root_scores(state, eval_values(engine.spec), 50_000_000, depth))
    child = engine.apply(state, legal[0])
    ended = engine.terminal(child)
    if ended is not None:
        if ended.winner is None:
            return 0.0
        signed = WIN_SCORE - child.ply
        return signed if ended.winner == state.side_to_move else -signed
    value = _probe(engine, child, depth - 1)
    return value if child.side_to_move == state.side_to_move else -value


class TestTheSmallBoard:
    @pytest.mark.parametrize(
        "plies", [5, pytest.param(7, marks=pytest.mark.slow)], ids=["five plies", "seven plies"]
    )
    def test_every_short_decision_agrees_with_the_probe(self, plies: int) -> None:
        engine = _small_engine()
        solve = solve_endgame(engine, (["G", "A"], ["G"]), sample_plies=plies)
        assert solve.samples, "a rook-like rider against a bare royal decides positions quickly"
        for sample in solve.samples:
            score = _probe(engine, sample.state, sample.plies)
            assert abs(score) > WIN_SCORE / 2, (sample.plies, sample.value)
            assert (score > 0) == (sample.value > 0)
            assert WIN_SCORE - abs(score) == sample.plies

    def test_a_short_win_is_not_one_the_probe_finds_sooner(self) -> None:
        engine = _small_engine()
        solve = solve_endgame(engine, (["G", "A"], ["G"]), sample_plies=5)
        for sample in solve.samples:
            if sample.plies >= 3:
                shallower = _probe(engine, sample.state, sample.plies - 2)
                assert abs(shallower) < WIN_SCORE / 2

    def test_the_counts_cover_every_placement(self) -> None:
        solve = solve_endgame(_small_engine(), (["G", "A"], ["G"]))
        asked = solve.requested
        assert asked.name == "AG v G"
        # 16 x 15 x 14 placements, two sides to move: each solved or set aside as impossible.
        assert asked.positions + asked.illegal == 2 * 16 * 15 * 14
        assert asked.won > 0 and asked.lost > 0 and asked.drawn > 0
        assert asked.longest_loss == asked.longest_win + 1
        bare = solve.of(["G"], ["G"])
        assert bare is not None and bare.won == bare.lost == 0

    def test_a_lone_leaper_never_wins_here(self) -> None:
        asked = solve_endgame(_small_engine(), (["G", "J"], ["G"])).requested
        assert asked.won == 0 and asked.lost == 0

    def test_a_short_clock_makes_long_wins_cursed(self) -> None:
        long = solve_endgame(_small_engine(), (["G", "A"], ["G"])).requested
        assert long.beyond_clock == 0 and long.longest_to_reset == long.longest_win
        short = solve_endgame(_small_engine(no_progress=3), (["G", "A"], ["G"])).requested
        assert short.won == long.won
        assert short.beyond_clock > 0

    def test_the_answer_does_not_depend_on_how_the_set_is_ordered(self) -> None:
        engine = _small_engine()
        one = solve_endgame(engine, (["G", "A"], ["G"]))
        two = solve_endgame(engine, (["A", "G"], ["G"]))
        assert one.sets == two.sets and one.positions == two.positions


class TestRefusals:
    def test_an_unknown_piece_is_named(self) -> None:
        with pytest.raises(ValueError, match="not a piece of this game: Z"):
            solve_endgame(_small_engine(), (["G", "Z"], ["G"]))

    def test_a_set_over_the_cap_is_refused_before_it_starts(self) -> None:
        with pytest.raises(ValueError, match="over the cap of 100"):
            solve_endgame(_small_engine(), (["G", "A"], ["G"]), max_positions=100)

    def test_a_closure_over_the_cap_is_refused(self) -> None:
        # The forward stepper's set starts on two ranks (4,096 placements at most); its
        # promotion leads into the rider's set, which the cap does not leave room for.
        with pytest.raises(ValueError, match="passed its cap of 4200 positions"):
            solve_endgame(_small_engine(), (["G", "W"], ["G"]), max_positions=4200)

    def test_a_promotion_is_followed_into_the_set_it_makes(self) -> None:
        solve = solve_endgame(_small_engine(), (["G", "W"], ["G"]))
        promoted = solve.of(["G", "A"], ["G"])
        assert promoted is not None and promoted.won > 0
        assert solve.requested.longest_win > promoted.longest_win

    @pytest.mark.parametrize(
        ("path", "rule"),
        [
            ("rulesets/crazyhouse.json", "a capture hand"),
            ("tests/_purse_market.json", "the purse"),
            ("tests/_retired_gardner_in_rounds.json", "rounds"),
            ("tests/_retired_gardner_with_spies.json", "spies"),
            ("rulesets/duck-chess.json", "the duck"),
            ("tests/_retired_the_menagerie.json", "wild pieces"),
        ],
    )
    def test_a_game_it_cannot_read_is_refused_by_name(self, path: str, rule: str) -> None:
        engine = Engine(json.loads((ROOT / path).read_text(encoding="utf-8")))
        royal = [engine.spec.piece_ids[engine.spec.royal_types[0]]] if engine.spec.royal_types else []
        any_piece = royal or [engine.spec.piece_ids[0]]
        with pytest.raises(ValueError, match=f"cannot be solved here: {rule}"):
            solve_endgame(engine, (any_piece, any_piece))


@pytest.mark.slow
class TestChessPublishedMaxima:
    """Published chess endgame numbers: chess here only because they exist (rule 5)."""

    def test_chess_king_and_rook_mate_in_sixteen(self) -> None:
        asked = solve_endgame(_chess(), (["K", "R"], ["K"])).requested
        assert asked.longest_win == _plies(16)
        assert asked.beyond_clock == 0

    def test_chess_king_and_queen_mate_in_ten(self) -> None:
        asked = solve_endgame(_chess(), (["K", "Q"], ["K"])).requested
        assert asked.longest_win == _plies(10)

    def test_chess_king_and_pawn_mate_in_twenty_eight_through_its_promotions(self) -> None:
        solve = solve_endgame(_chess(), (["K", "P"], ["K"]))
        assert solve.requested.longest_win == _plies(28)
        queen = solve.of(["K", "Q"], ["K"])
        assert queen is not None and queen.longest_win == _plies(10)

    def test_chess_two_knights_never_force_mate(self) -> None:
        # The only wins are mates in one already on the board; no defender is ever forced into one.
        asked = solve_endgame(_chess(), (["K", "N", "N"], ["K"])).requested
        assert asked.longest_win == 1
        assert asked.longest_loss == 0


def _live_games() -> list[Path]:
    paths = sorted((ROOT / "rulesets").glob("*.json")) + sorted((ROOT / "archive").glob("*.json"))
    return [path for path in paths if json.loads(path.read_text(encoding="utf-8")).get("rules", {}).get("insufficient_material")]


@pytest.mark.slow
@pytest.mark.parametrize("path", _live_games(), ids=lambda path: path.stem)
def test_the_dead_sets_are_drawn_wherever_the_rule_is_live(path: Path) -> None:
    engine = Engine(load_spec(path))
    rule = engine.spec.insufficient_material
    if rule is None:
        pytest.skip("the rule is inert in this game (not chess underneath)")
    royal = engine.spec.piece_ids[engine.spec.royal_types[0]]
    sets = [([royal], [royal])]
    for piece_type, role in enumerate(rule.roles):
        if role in (INSUFFICIENT_LEAPER, INSUFFICIENT_RIDER):
            sets.append(([royal, engine.spec.piece_ids[piece_type]], [royal]))
    for pieces in sets:
        asked = solve_endgame(engine, copy.deepcopy(pieces)).requested
        assert asked.won == asked.lost == 0, (asked.name, asked.won, asked.lost)
        assert asked.drawn > 0
