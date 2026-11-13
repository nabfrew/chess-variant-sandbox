"""The pure readings of `docs/reference/score-dials.py`'s R-S11 and R-S12 parts (§6.1, §7.1),
on hand-built ladders and games: no stored run is read."""

from __future__ import annotations

import math

import private_paths
import pytest

from chess_variant_sandbox.metrics.scoring import Weights, parse_weights

SCRIPT = "docs/reference/score-dials.py"
dials = private_paths.script(SCRIPT, "score_dials")
pytestmark = private_paths.needs(SCRIPT)


def _standings(ratings: dict[str, float]) -> list[dict[str, object]]:
    return [{"name": name, "rating": value} for name, value in ratings.items()]


def test_ladder_readings_on_a_linear_ladder() -> None:
    # 100 Elo and half a ply a doubling: 200 Elo a ply; bottom rung at 300 of a 600 range.
    ratings = {"random": -300.0, "greedy": -100.0, "minimax:nodes=400": 0.0,
               "minimax:nodes=800": 100.0, "minimax:nodes=1600": 200.0, "minimax:nodes=3200": 300.0}
    depth = {f"minimax:nodes={n}": {"mean_depth": 1.0 + 0.5 * i}
             for i, n in enumerate((400, 800, 1600, 3200))}
    got = dials.ladder_readings(_standings(ratings), depth)
    assert got["G"] == pytest.approx(100.0)
    assert got["P"] == pytest.approx(0.5)
    assert got["E"] == pytest.approx(200.0)
    assert got["B"] == pytest.approx(0.5)


def test_ladder_readings_withholds_e_on_flat_depth_and_b_without_random() -> None:
    ratings = {"minimax:nodes=400": 0.0, "minimax:nodes=800": 50.0}
    depth = {"minimax:nodes=400": {"mean_depth": 2.0}, "minimax:nodes=800": {"mean_depth": 2.01}}
    got = dials.ladder_readings(_standings(ratings), depth)
    assert got["G"] == pytest.approx(50.0)
    assert got["E"] is None
    assert got["B"] is None


def test_strength_order_puts_greedy_below_the_rungs() -> None:
    order = sorted(["minimax:nodes=1600", "greedy", "minimax:nodes=400"], key=dials.strength_key)
    assert order == ["greedy", "minimax:nodes=400", "minimax:nodes=1600"]
    with pytest.raises(ValueError):
        dials.strength_key("mystery")


def test_pairing_spreads_separates_mismatches_from_even_games() -> None:
    a, b, c = "minimax:nodes=400", "minimax:nodes=800", "minimax:nodes=1600"
    games = (
        [(a, a, 100)] * 2 + [(a, a, 100)] * 2        # equal, no spread
        + [(a, b, 90), (b, a, 110)] * 2              # adjacent, spread 0.1
        + [(a, c, 20), (c, a, 180)] * 2              # a mismatch two rungs apart, spread 0.8
        + [("random", a, 5)] * 4                     # never graded
    )
    got = dials.pairing_spreads(games)
    lengths = [100.0] * 4 + [90.0, 110.0] * 2 + [20.0, 180.0] * 2
    assert got["S"] == pytest.approx(dials._spread(lengths))
    assert got["W"] == pytest.approx((0.0 + 0.1 + 0.8) / 3)
    assert got["W_even"] == pytest.approx(0.05)


def test_pairing_spreads_skips_thin_pairings() -> None:
    a, b = "minimax:nodes=400", "minimax:nodes=800"
    got = dials.pairing_spreads([(a, b, 10), (b, a, 30), (a, a, 50), (a, a, 50), (a, a, 50), (a, a, 50)])
    assert got["W"] == pytest.approx(0.0)
    assert not math.isnan(got["S"])


# --- R-S14 (§8.1): published figures through a weights file's own curves ---------------------

def _weights() -> Weights:
    def component(family: str, kind: str, points: dict[str, float]) -> dict[str, object]:
        return {"family": family, "weight": 1.0, "curve": kind, "points": points}

    return parse_weights({
        "version": "t", "llm_weight": 0.0,
        "components": {
            "mean_branching": component("pacing", "band", {
                "zero_low": 1.0, "good_low": 10.0, "good_high": 50.0, "zero_high": 100.0}),
            "ply_spread": component("pacing", "higher_better", {"zero_at": 0.0, "good_at": 0.8}),
            "first_player_advantage": component("balance", "lower_better",
                                                {"good_at": 0.0, "zero_at": 0.5}),
            "piece_utilisation": component("decisions", "higher_better",
                                           {"zero_at": 0.5, "good_at": 1.0}),
            "instance_participation": component("decisions", "band", {
                "zero_low": 0.0, "good_low": 0.5, "good_high": 0.9, "zero_high": 1.0}),
            "skill_gradient": component("depth", "higher_better", {"zero_at": 8.0, "good_at": 100.0}),
        },
        "gates": [{"component": "mean_branching", "minimum": 1.5, "reason": "r"},
                  {"component": "mean_plies_worst_half", "minimum": 8.0, "reason": "r"},
                  {"component": "decisive_rate_worst_half", "minimum": 0.2, "reason": "r"}],
    })


def test_rs14_reading_scores_only_what_has_a_figure_and_grades_the_pooled_spread() -> None:
    got = dials.rs14_reading(_weights(), {"mean_branching": 75.0, "ply_spread": 0.25,
                                          "mean_plies": 60.0, "decisive_rate": 0.1})
    assert got["scores"] == {"mean_branching": pytest.approx(0.5), "ply_spread": pytest.approx(0.3125)}
    assert got["graded"]["ply_spread"] == pytest.approx(0.5)      # 1.6 x 0.25 on a ramp to 0.8
    assert got["gates"]["mean_branching"] is None
    assert got["gates"]["mean_plies_worst_half"] is None
    assert got["gates"]["decisive_rate_worst_half"] is not None
    # Pacing is the mean of two halves and every other family is perfect; it holds two of the
    # six unit weights, a third of the index.
    assert got["ceiling"] == pytest.approx(0.5 ** (1 / 3))
    assert got["cost"]["mean_branching"] == pytest.approx(1.0 - 0.75 ** (1 / 3))


def test_rs14_reading_of_no_figures_is_a_perfect_ceiling() -> None:
    got = dials.rs14_reading(_weights(), {})
    assert got["scores"] == {}
    assert got["ceiling"] == pytest.approx(1.0)
    assert all(violation is None for violation in got["gates"].values())


def test_rs14_verdict_is_the_registered_line() -> None:
    assert dials.rs14_verdict([0.0, 0.1]) == "not gradeable"
    assert dials.rs14_verdict([0.49, 0.0, 0.2, 1.0]) == "named"
    assert dials.rs14_verdict([0.5, 0.0, 0.2, 1.0]) == "not named"


def test_rs14_raw_turns_a_first_seat_score_into_the_dial() -> None:
    raw = dials.rs14_raw({"name": "g", "first_seat": (0.4, "s", "b"), "mean_plies": (30.0, "s", "b")})
    assert raw == {"mean_plies": 30.0, "first_player_advantage": pytest.approx(0.2)}


def test_rs14_figures_are_ten_games_each_with_a_source() -> None:
    assert len(dials.RS14_GAMES) == 10
    for game in dials.RS14_GAMES:
        figures = [value for key, value in game.items() if key not in ("name", "alt")]
        figures += [alt for alts in (game.get("alt") or {}).values() for alt in alts]
        assert all(len(figure) == 3 and figure[1] and figure[2] for figure in figures), game["name"]
        assert "mean_branching" in game and "mean_plies" in game


# --- R-S14a (§12.1): game lengths from archive formats, on hand-written records ---------------

_SGF = (
    "(;GM[1]FF[4]SZ[19]PB[a]PW[b]DT[2015-01-01]RE[B+R]C[a comment with ;B[aa\\] inside]"
    ";B[pd];W[dp];B[](;W[qq];B[cc])(;W[aa];B[bb];W[cc]))\n"
    "(;SZ[19]HA[2]AB[dd][pp]RE[W+3.5];W[pd])\n"
    "(;SZ[19]RE[W+T];B[pd])(;SZ[13]RE[B+2];B[pd])(;SZ[19]RE[W+12.5];B[pd];W[dd])"
)


def test_sgf_games_reads_the_root_and_the_main_line() -> None:
    games = dials.sgf_games(_SGF)
    assert len(games) == 5
    first = games[0]
    assert first["root"]["RE"] == "B+R" and first["root"]["PB"] == "a"
    # three moves, a pass included, then the first child only: W[qq] B[cc]
    assert first["moves"] == [("B", "pd"), ("W", "dp"), ("B", ""), ("W", "qq"), ("B", "cc")]
    assert [dials.go_class(g) for g in games] == ["resigned", "handicap", "time", "not 19x19", "end"]


def test_go_class_reads_results_by_the_registered_rule() -> None:
    def cls(result: str) -> str:
        return dials.go_class({"root": {"RE": result}, "moves": []})

    assert cls("B+Resign") == "resigned" and cls("w+r") == "resigned"
    assert cls("W+0.5") == "end" and cls("0") == "end" and cls("Jigo") == "end"
    assert cls("B+Time") == "time" and cls("W+F") == "forfeit"
    assert cls("Void") == "void or unknown" and cls("?") == "void or unknown"
    assert dials.go_class({"root": {}, "moves": []}) == "no result"


def _wtb(records: list[list[int]], board: int = 8) -> bytes:
    header = bytes([20, 26, 11, 13]) + len(records).to_bytes(4, "little") + (0).to_bytes(2, "little")
    header += (2026).to_bytes(2, "little") + bytes([board, 0, 22, 0])
    body = b""
    for moves in records:
        body += (0).to_bytes(2, "little") + (1).to_bytes(2, "little") + (2).to_bytes(2, "little")
        body += bytes([32, 32]) + bytes(moves + [0] * (60 - len(moves)))
    return header + body


def test_wthor_games_counts_moves_to_the_first_empty_slot() -> None:
    full = [56, 64] + [11] * 58
    games = dials.wthor_games(_wtb([full, [56, 64, 53], []]))
    assert [len(g) - 4 for g in games] == [60, 3, 0]
    assert games[0][:4] == (2026, 0, 1, 2)
    assert dials.wthor_games(_wtb([full], board=10)) == []
    assert dials.wthor_games(_wtb([full])[:-1]) == []           # size does not match the header
    assert dials.wthor_valid((56, 64)) and not dials.wthor_valid((56, 90)) and not dials.wthor_valid((19,))


_PDN = """[Event "a"]
[White "x"]
[Black "y"]
[Result "1-0"]
1. 11-15 {a comment 9-14} 23-19 2. 8-11 (2. 9-14 22-17) 22-17 3. 15x24x31 1-0

[Event "b"]
[FEN "B:W21-32:B1-12"]
[Result "1/2-1/2"]
1. 9-13 21-17 1/2-1/2

[Event "c"]
[FEN "W:W21,22:BK1"]
[Result "0-1"]
1. 21-17 0-1

[Event "d"]
[GameType "20"]
1. 32-28 *

[Event "e"]
1. 11-15 *
"""


def test_pdn_games_and_plies() -> None:
    games = dials.pdn_games(_PDN)
    assert [g["tags"].get("Event") for g in games] == ["a", "b", "c", "d", "e"]
    assert dials.pdn_plies(games[0]["movetext"]) == ["11-15", "23-19", "8-11", "22-17", "15x24x31"]
    assert dials.pdn_plies("1-0 0-1 2-0 1-1 1/2-1/2 33-29 5-5") == []
    assert [dials.draughts_class(g) for g in games] == [
        "counted", "counted", "set-up position", "another game type", "no result"]


def test_english_start_reads_ranges_and_lists() -> None:
    assert dials.english_start("B:W21-32:B1-12")
    assert dials.english_start('B:B1,2,3,4,5,6,7,8,9,10,11,12:W21-32.')
    assert not dials.english_start("W:W21-32:B1-12")
    assert not dials.english_start("B:W21-32:B1-11,K12")
    assert not dials.english_start("nonsense")


def test_rs14a_archives_count_games_once_and_by_the_rule(tmp_path) -> None:
    go, oth, dra = tmp_path / "go", tmp_path / "oth", tmp_path / "dra"
    for d in (go, oth, dra):
        d.mkdir()
    (go / "one.sgf").write_text(_SGF, encoding="latin-1")
    (go / "copy.SGF").write_text(_SGF, encoding="latin-1")          # every game a duplicate
    lengths, left = dials.rs14a_go(go)
    assert lengths == {"end": [2.0], "resigned": [5.0]}
    assert left["duplicate"] == 2 and left["handicap"] == 2
    (oth / "WTH_2026.wtb").write_bytes(_wtb([[56] * 60, [56, 64], [56, 64], []]))
    lengths_o, left_o = dials.rs14a_othello(oth)
    assert lengths_o == [60.0, 2.0] and left_o == {"duplicate": 1, "no move": 1}
    (dra / "games.pdn").write_text(_PDN, encoding="latin-1")
    lengths_d, left_d = dials.rs14a_draughts(dra)
    assert lengths_d == [5.0, 2.0]
    assert left_d == {"set-up position": 1, "another game type": 1, "no result": 1}


def test_rs14a_reading_is_the_dials_statistic_through_the_curve() -> None:
    got = dials.rs14a_reading(_weights(), [100.0, 300.0])
    assert got["spread"] == pytest.approx(0.5) and got["mean"] == pytest.approx(200.0)
    assert got["score"] == pytest.approx(0.625)                    # 0.5 on a ramp to 0.8
    assert got["graded"] == pytest.approx(1.0)                     # 1.6 x 0.5 = 0.8


# --- R-S11b (§9.1): the v12 ladder below 400 nodes -------------------------------------------

def _v12(scale: float, front: float = 0.0) -> dict[str, float]:
    """A v12 ladder gaining `scale` Elo a doubling from 100 to 6400, plus `front` Elo by 400."""
    out = {"random": -1000.0, "greedy": -800.0}
    for i, n in enumerate((100, 200, 400, 800, 1600, 3200, 6400)):
        if n != 3200:
            out[f"minimax:nodes={n}"] = scale * i + (front if n >= 400 else front * i / 2)
    return out


def test_low_ladder_readings_on_a_linear_ladder() -> None:
    got = dials.low_ladder_readings(_standings(_v12(100.0)))
    assert got["S"] == pytest.approx(1 / 3)
    assert got["Span"] == pytest.approx(600.0)
    assert got["L"] == pytest.approx(100.0)
    assert got["H"] == pytest.approx(100.0)
    assert got["r100_random"] == pytest.approx(1000.0)
    assert got["r400_random"] == pytest.approx(1200.0)


def test_low_ladder_share_ignores_scale_and_sees_a_front_loaded_ladder() -> None:
    # A smaller copy keeps the share (reading ii); range gained early raises it (reading i).
    small = dials.low_ladder_readings(_standings(_v12(40.0)))
    assert small["S"] == pytest.approx(1 / 3)
    assert small["Span"] == pytest.approx(240.0)
    early = dials.low_ladder_readings(_standings(_v12(100.0, front=300.0)))
    assert early["S"] == pytest.approx(500.0 / 900.0)
    assert early["L"] > early["H"]


def test_low_ladder_readings_refuses_another_roster_and_withholds_a_flat_share() -> None:
    with pytest.raises(AssertionError):
        dials.low_ladder_readings(_standings({"minimax:nodes=400": 0.0, "minimax:nodes=800": 1.0}))
    flat = {name: 0.0 for name in _v12(1.0)}
    assert dials.low_ladder_readings(_standings(flat))["S"] is None


def test_shared_slope_reads_the_four_common_rungs() -> None:
    today = {"random": -900.0, "minimax:nodes=400": 0.0, "minimax:nodes=800": 50.0,
             "minimax:nodes=1600": 100.0, "minimax:nodes=3200": 400.0, "minimax:nodes=6400": 200.0,
             "minimax:nodes=25600": 999.0}
    assert dials.shared_slope(_standings(today)) == pytest.approx(50.0)
    assert dials.shared_slope(_standings({"minimax:nodes=400": 0.0})) is None


def test_rs11b_grade_is_the_registered_line() -> None:
    assert dials.rs11b_grade(0.60, 0.50, 0.60) == ("(i) the window", "comparable")
    assert dials.rs11b_grade(0.70, 0.40, 0.59) == ("(i), and smaller too", "not comparable")
    assert dials.rs11b_grade(0.55, 0.40, 0.9)[0] == "(ii) small games"
    assert dials.rs11b_grade(0.57, 0.30, 0.9)[0] == "neither"
    assert dials.rs11b_grade(0.50, 0.41, 0.9)[0] == "neither"


def test_qualifying_revision_takes_the_latest_full_v12_roster(tmp_path) -> None:
    import json

    game = tmp_path / "abc"
    v12 = ["random", "greedy"] + [f"minimax:nodes={n}" for n in dials.V12_RUNGS]
    v13 = ["random", "greedy"] + [f"minimax:nodes={n}" for n in (400, 800, 1600, 3200, 6400, 25600)]
    for k, agents, with_runs in ((1, v12, True), (2, v12, False), (3, v13, True)):
        revision = game / "previous" / str(k)
        revision.mkdir(parents=True)
        (revision / "canonical.json").write_text(json.dumps({"agents": agents, "seeds": [0, 1000]}))
        if with_runs:
            for seed in (0, 1000):
                (revision / f"abc-{seed}").mkdir()
                (revision / f"abc-{seed}" / "tournament.json").write_text("{}")
    found = dials.qualifying_revision(game)
    assert found is not None and found[0].name == "1" and len(found[1]) == 2
    assert dials.qualifying_revision(tmp_path / "none") is None


def test_spread_bounds_bracket_a_two_point_sample() -> None:
    # Half the games at 100 plies and half at 300: sd 100 on a mean of 200, and the range is tight.
    least, most = dials.spread_bounds(200.0, 100.0, 300.0, 100.0, 300.0)
    assert least == pytest.approx(0.5)
    assert most == pytest.approx(0.5)


# --- R-S15 (§10.1): the material dials re-read with each game's own players as the ruler ------

def test_mover_value_reads_the_child_from_the_side_that_moved() -> None:
    assert dials.mover_value(0, 1, None, 2.5) == pytest.approx(-2.5)   # the side changed
    assert dials.mover_value(0, 0, None, 2.5) == pytest.approx(2.5)    # a long turn: it did not
    assert dials.mover_value(1, 0, (1, False), None) == dials.WIN       # moved into its own win
    assert dials.mover_value(1, 0, (0, False), None) == -dials.WIN
    assert dials.mover_value(1, 0, (None, True), 7.0) == 0.0            # a draw, whatever was asked
    assert dials.mover_value(0, 1, None, None) is None


def test_near_best_and_criticality_follow_the_analysis_pass() -> None:
    assert dials.near_best([3.0, 2.5, 2.0, 1.9, -5.0], 1.0) == 3       # within one cheapest piece
    assert dials.criticality([3, 1], [5, 4]) == pytest.approx(1 - (0.6 + 0.25) / 2)
    assert dials.criticality([], []) is None


def test_conversion_counts_only_edges_of_a_unit_and_needs_eight() -> None:
    leads = [(1.5, 0)] * 6 + [(-2.0, 0)] * 2 + [(0.5, 1)] * 5           # the last five are no lead
    assert dials.conversion(leads) == (pytest.approx(0.75), 8)
    assert dials.conversion(leads[:7]) == (None, 7)
    assert dials.lead_index(100) == 34


def test_ask_ladder_asks_again_only_what_got_no_answer() -> None:
    calls: list[tuple[list[int], int]] = []

    def ask(todo: list[int], budget: int) -> list[tuple[float, int, int] | None]:
        calls.append((list(todo), budget))
        return [(float(k), 0, 1) if (k % 2 == 0 or budget >= 2) and k != 5 else None for k in todo]

    values, retried = dials.ask_ladder(ask, 6, [1, 2, 3])
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0, None]
    assert calls == [([0, 1, 2, 3, 4, 5], 1), ([1, 3, 5], 2), ([5], 3)]
    assert retried == [3, 1]


def test_rs15_bands_are_the_registered_ones() -> None:
    assert dials.moves_with_ruler([0.10, 0.12, 0.11, 0.09], 0.06)
    assert not dials.moves_with_ruler([0.10, 0.12, 0.11, 0.09], 0.04)  # the score barely moves
    assert not dials.moves_with_ruler([0.3, -0.3, 0.3, -0.2], 0.2)    # inside the noise
    assert not dials.moves_with_ruler([0.1], 0.2)                     # one run reads nothing
    assert dials.rs15_depends(4, 12) == "depends on the ruler"
    assert dials.rs15_depends(1, 12) == "does not"
    assert dials.rs15_depends(2, 12) == "partly"
    assert dials.rs15_depends(0, 0) == "no reading"
    assert dials.rs15_family(0.70, 7, 5) == "larger where unlike"
    assert dials.rs15_family(0.50, 7, 5) == "not larger"
    assert dials.rs15_family(0.60, 7, 5) == "unclear"
    assert dials.rs15_family(0.90, 2, 5) == "not graded"
    assert dials.rs15_verdict("depends on the ruler", "larger where unlike") == (
        "depends on material being the game's currency")
    assert dials.rs15_verdict("depends on the ruler", "unclear") == "depends on the ruler, not by family"
    assert dials.rs15_verdict("does not", "larger where unlike") == "does not depend on the ruler"
    assert dials.rs15_verdict("partly", "larger where unlike") == "not shown"


def test_rs15_seed_is_fixed_by_the_record() -> None:
    assert dials.rs15_seed(1000, 8, 3) == dials.rs15_seed(1000, 8, 3)
    assert len({dials.rs15_seed(1000, 8, k) for k in range(50)}) == 50


# --- R-S15b (§11.1): the move values on each arm's own expected-score scale ------------------

def test_expected_is_a_safe_sigmoid() -> None:
    assert dials.expected(0.0, 1.0) == pytest.approx(0.5)
    assert dials.expected(2.0, 2.0) == pytest.approx(1 / (1 + math.exp(-1)))
    assert dials.expected(dials.WIN, 0.01) == 1.0
    assert dials.expected(-dials.WIN, 1000.0) == 0.0


def test_fit_scale_recovers_the_scale_that_made_the_results() -> None:
    points = [(v / 4, dials.expected(v / 4, 2.5)) for v in range(-20, 21)]
    assert dials.fit_scale(points) == pytest.approx(2.5, rel=1e-3)
    # all draws: no value predicts anything, so the widest scale
    assert dials.fit_scale([(1.0, 0.5), (-3.0, 0.5)]) == pytest.approx(1000.0, rel=1e-2)


def test_es_near_and_loss_read_the_expected_score_gap() -> None:
    # at k = 1: 0 -> 0.5, -0.3 -> 0.426, -1 -> 0.269
    assert dials.es_near([0.0, -0.3, -1.0], 1.0, 0.10) == 2
    assert dials.es_near([0.0, -0.3, -1.0], 10.0, 0.10) == 3   # a wider scale squeezes the gaps
    assert dials.es_loss([0.0, 0.0], 1.0) == 0.0


def test_result_for_reads_the_side_to_move() -> None:
    assert dials.result_for(0, 0) == 1.0
    assert dials.result_for(1, 0) == 0.0
    assert dials.result_for(1, None) == 0.5


def test_rs15b_bands_are_the_registered_ones() -> None:
    assert dials.still_fewer([-0.10, -0.12, -0.11, -0.09])
    assert not dials.still_fewer([-0.3, 0.3, -0.3, 0.2])
    assert not dials.still_fewer([0.10, 0.12, 0.11, 0.09])
    assert not dials.still_fewer([-0.1])
    assert dials.rs15b_verdict(4, 12, 0.5) == "compressed scale"
    assert dials.rs15b_verdict(4, 12, 0.6) == "neither"
    assert dials.rs15b_verdict(5, 12, 0.2) == "neither"
    assert dials.rs15b_verdict(8, 12, 0.5) == "genuinely fewer costly moves"
    assert dials.rs15b_verdict(8, 12, 0.4) == "neither"
    assert dials.rs15b_verdict(0, 0, float("nan")) == "neither"


def test_rs15b_game_undoes_a_pure_squeeze() -> None:
    # The net's values are the built-in's halved, on the same results: in units the net sees
    # fewer gaps of a piece; with each arm's own fitted scale the gaps agree.
    def run(shift: float) -> dict[str, object]:
        builtin, net = [], []
        for i in range(30):
            base = (i - 15) / 5 + shift
            vals = [base, base - 0.6, base - 1.5, base - 3.0]
            y = (0.0 if i % 4 else 1.0) if i < 12 else (1.0 if i % 3 else 0.0)
            builtin.append((0, vals, y))
            net.append((0, [v / 2 for v in vals], y))
        return {"epsilon": 1.0, "builtin": builtin, "net": net}

    res = dials.rs15b_game([run(0.0), run(0.1), run(-0.1)], 0.10)
    assert res["mean_raw"] < 0
    assert abs(res["mean_es"]) < 1e-9
    assert not res["still"]
    for ks in res["k"]:
        assert ks["builtin"] / ks["net"] == pytest.approx(2.0, rel=1e-3)
