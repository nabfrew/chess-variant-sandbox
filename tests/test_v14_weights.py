"""`weights/v14.json` and its two new readings (R-S6b, `docs/reference/v14-registration.md`).

Held here: that `v14` is `v13` with exactly the six changes of §2.1 and nothing else (§3.6's
first check), that `v13.json` is untouched, the expected-score map and its fit
(`metrics/decision_cost.py`), the pieces followed by the analysis pass (`PieceFollower`), and
that a pass run without `readings` writes the file it always wrote. Assertions are against
specs built here, never against a stored run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import private_paths
import pytest

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_CASTLE, move_type
from chess_variant_sandbox.metrics import decision_cost
from chess_variant_sandbox.metrics.components import _analysis_components
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.analysis import (
    CorpusAnalysis,
    GameAnalysis,
    PieceFollower,
    _analyse_game,
    _epsilon_for,
    movable_types,
)
from chess_variant_sandbox.tournament.record import GameRecord

ROOT = Path(__file__).resolve().parents[1]
WRITE_SCRIPT = "docs/reference/v14-write.py"
EMPTY_BOARD_SCRIPT = "docs/reference/empty-board-participation.py"
WEIGHTS = ROOT / "chess_variant_sandbox" / "metrics" / "weights"
# `v13.json` byte for byte as Q1104 wrote it (§3.6: a re-anchor is never an edit).
V13_SHA256 = "1527c271a2fb0f7d86bafcacca998f769057825c6d87df6ab49fa46750e604c8"
CHANGED = {"skill_gradient", "mean_branching", "instance_participation", "piece_participation",
           "decision_criticality", "decision_cost", "lead_conversion", "blunder_recoverability"}


def _json(version: str) -> dict:
    return json.loads((WEIGHTS / f"{version}.json").read_text(encoding="utf-8"))


class TestTheVersion:
    def test_v13_is_unedited(self) -> None:
        # Pinned from git: the file as committed by Q1104.
        assert hashlib.sha256((WEIGHTS / "v13.json").read_bytes()).hexdigest() == V13_SHA256
        assert _json("v13")["version"] == "v13"

    def test_v14_loads_and_was_current(self) -> None:
        # R-S6d (§3.7, §11): every line held, so `v14` was current and `v13` stays a version.
        # R-S17i made `v15` current (`tests/test_v15_weights.py`); `v14` stays a version.
        from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS

        weights = load_weights("v14")
        assert weights.version == "v14"
        assert DEFAULT_WEIGHTS == "v15"
        assert load_weights("v13").version == "v13"

    def test_everything_section_2_1_does_not_name_is_v13s_to_the_digit(self) -> None:
        v13, v14 = _json("v13"), _json("v14")
        assert v14["gates"] == v13["gates"]
        assert v14["llm_weight"] == v13["llm_weight"]
        for name, config in v13["components"].items():
            if name in CHANGED:
                continue
            assert v14["components"][name] == config, name

    def test_family_totals_do_not_move(self) -> None:
        a, b = load_weights("v13").family_weights(), load_weights("v14").family_weights()
        assert set(a) == set(b)
        for family in a:
            assert math.isclose(a[family], b[family], abs_tol=1e-12), family

    def test_the_six_changes(self) -> None:
        v13, v14 = _json("v13")["components"], _json("v14")["components"]
        sg = v14["skill_gradient"]
        assert sg["points"] == {"zero_at": 8.0, "good_at": 190.85}
        assert sg["sources"]["good_at"]["source"] == "meaning"
        assert math.isclose(400 * math.log10(3), 190.85, abs_tol=0.005)
        mb = v14["mean_branching"]
        assert mb["curve"] == "higher_better"
        assert mb["points"] == {"zero_at": 1.5, "good_at": v13["mean_branching"]["points"]["good_low"]}
        assert mb["sources"]["zero_at"] == v13["mean_branching"]["sources"]["zero_low"]
        pp = v14["piece_participation"]
        assert pp["curve"] == "higher_better" and pp["optional"] is True
        assert pp["points"]["zero_at"] == 0.0
        assert pp["weight"] == v13["instance_participation"]["weight"]
        dc = v14["decision_cost"]
        assert dc["points"] == {"zero_low": 0.0, "good_low": 0.25, "good_high": 0.75,
                                "zero_high": 1.0}
        assert dc["optional"] is True and dc["weight"] == v13["decision_criticality"]["weight"]
        assert "lead_conversion" not in v14 and "instance_participation" not in v14
        assert "decision_criticality" not in v14
        assert math.isclose(
            v14["blunder_recoverability"]["weight"],
            v13["blunder_recoverability"]["weight"] + v13["lead_conversion"]["weight"],
        )

    def test_many_moves_take_no_marks_off(self) -> None:
        curve = load_weights("v14").curves["mean_branching"]
        assert curve.score(250.0) == 1.0 and curve.score(8.9219) == 1.0
        assert curve.score(1.5) == 0.0
        assert load_weights("v13").curves["mean_branching"].score(250.0) == 0.0

    def test_every_piece_moving_takes_no_marks_off(self) -> None:
        curve = load_weights("v14").curves["piece_participation"]
        assert curve.score(1.0) == 1.0 and curve.score(0.0) == 0.0

    @private_paths.needs(WRITE_SCRIPT)
    def test_the_file_is_its_build(self) -> None:
        """`docs/reference/v14-write.py --check`: the file is built from `v13.json`, not typed."""
        module = private_paths.script(WRITE_SCRIPT, "v14_write")
        on_disk = _json("v14")
        end = on_disk["components"]["piece_participation"]["points"]["good_at"]
        why = on_disk["components"]["piece_participation"]["sources"]["good_at"]["why"]
        if why.startswith("UNPINNED"):
            built = module.build(None, None)
        else:
            built = module.build(end, why.rsplit(": ", 1)[-1])
        assert built == on_disk


# ------------------------------------------------------------------ the expected-score map


def _synthetic(k: float, n: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    points = []
    for _ in range(n):
        v = rng.uniform(-3 * k, 3 * k)
        points.append((v, 1.0 if rng.random() < decision_cost.expected(v, k) else 0.0))
    return points


class TestTheMap:
    def test_the_fit_recovers_a_scale(self) -> None:
        k = decision_cost.fit_scale(_synthetic(4.0, 4000, 1))
        assert 3.2 < k < 5.0

    def test_win_scores_map_to_the_ends(self) -> None:
        assert decision_cost.expected(1e6, 3.0) == 1.0
        assert decision_cost.expected(-1e6, 3.0) == 0.0
        assert decision_cost.expected(0.0, 3.0) == 0.5

    def test_a_fit_at_an_edge_is_flagged(self) -> None:
        assert decision_cost.at_edge(decision_cost.K_RANGE[1])
        assert decision_cost.at_edge(decision_cost.K_RANGE[0])
        assert not decision_cost.at_edge(5.0)
        # Results that never follow the values push `k` to the top of the range.
        flat = [(float(v), 0.5) for v in range(-50, 50)] * 2
        k, why = decision_cost._fit(flat)
        assert k is None and "edge" in why


def _game(seed: int, values: list[list[float]], movers: list[int]) -> GameAnalysis:
    return GameAnalysis(seed=seed, agents=("a", "b"), sample_values=values,
                        sample_movers=movers, fielded_instances=0, fielded_moved=0)


def _record(seed: int, winner: int | None) -> GameRecord:
    return GameRecord(ruleset_hash="x", ruleset_name="x", seed=seed, agents=("a", "b"),
                      moves=[], winner=winner, reason="", plies=0)


def _run(n_games: int, per_game: int, seed: int) -> tuple[CorpusAnalysis, list[GameRecord]]:
    rng = random.Random(seed)
    games, records = [], []
    for g in range(n_games):
        values = [[rng.uniform(-5, 5) for _ in range(4)] for _ in range(per_game)]
        winner = rng.choice([0, 1, None])
        games.append(_game(g, values, [i % 2 for i in range(per_game)]))
        records.append(_record(g, winner))
    return CorpusAnalysis(games=games), records


class TestTheReading:
    def test_too_few_positions_withholds(self) -> None:
        analysis, records = _run(4, 5, 0)  # 20 positions
        value, detail = decision_cost.reading(analysis, records)
        assert value is None and "fewer than" in detail["withheld"]

    def test_one_run_reads_odd_games_with_the_even_fit(self) -> None:
        rng = random.Random(2)
        games, records = [], []
        for g in range(40):
            # One value a game, seat 0 to move at every sample, and the result drawn from it
            # on k = 2, so the fit has a scale to find.
            v = rng.uniform(-8, 8)
            values = [[v, v - 1.0, v - 3.0] for _ in range(10)]
            movers = [0] * 10
            winner = 0 if rng.random() < decision_cost.expected(v, 2.0) else 1
            games.append(_game(g, values, movers))
            records.append(_record(g, winner))
        analysis = CorpusAnalysis(games=games)
        fit = decision_cost.one_run_fit(analysis, records)
        assert not fit.withheld, fit.why
        keys = [(g.seed, tuple(g.agents)) for g in analysis.games]
        even = decision_cost._fit([p for i, (_k, pts) in enumerate(
            decision_cost.game_points(analysis, records)) if i % 2 == 0 for p in pts])[0]
        assert {fit.scales[k] for k in keys[1::2]} == {even}
        assert len({fit.scales[k] for k in keys[0::2]}) == 1

    def test_a_run_never_fits_on_itself(self) -> None:
        runs = [_run(30, 10, s) for s in range(4)]
        fits = decision_cost.cross_run_fits(runs)
        assert len(fits) == 4
        for index, fit in enumerate(fits):
            others = [p for j, (a, r) in enumerate(runs) if j != index
                      for _k, pts in decision_cost.game_points(a, r) for p in pts]
            expected = decision_cost._fit(others)[0]
            got = set(fit.scales.values())
            assert got == ({expected} if expected is not None else set())

    def test_one_run_from_its_siblings_is_the_cross_run_fit(self) -> None:
        runs = [_run(30, 10, s) for s in range(4)]
        whole = decision_cost.cross_run_fits(runs)
        for index, (analysis, _records) in enumerate(runs):
            alone = decision_cost.others_fit(analysis, [r for j, r in enumerate(runs) if j != index])
            assert alone.scales == whole[index].scales and alone.why == whole[index].why

    def test_every_v14_component_is_explained(self) -> None:
        from chess_variant_sandbox.metrics.glossary import MEASUREMENTS, explain

        assert {"decision_cost", "piece_participation"} <= set(MEASUREMENTS)
        assert set(explain("v14")["components"]) == set(load_weights("v14").weights)

    def test_the_share_of_costly_moves(self) -> None:
        # One sample: values 0, 0, -100 at k=1 -> the third move gives up nearly a whole game.
        game = _game(0, [[0.0, 0.0, -100.0]], [0])
        analysis = CorpusAnalysis(games=[game], cost_fit=decision_cost.CostFit(
            scales={(0, ("a", "b")): 1.0}, positions=999, rule="test"))
        value, _ = decision_cost.reading(analysis, [_record(0, 0)])
        assert value == pytest.approx(1.0 - 2.0 / 3.0)

    def test_absent_without_the_values(self) -> None:
        game = GameAnalysis(seed=0, agents=("a", "b"), near_best=[1], legal_counts=[2])
        raw = _analysis_components(CorpusAnalysis(games=[game]), {}, {"pieces": []},
                                   [_record(0, 0)])
        assert "decision_cost" not in raw and "piece_participation" not in raw
        assert "decision_criticality" in raw


# ------------------------------------------------------------------ a library run's scale

READ_SCRIPT = "docs/reference/v14-read.py"
FOUR = (0, 1000, 2000, 3000)


def _fitted_run(seed: int, n_games: int = 30) -> tuple[CorpusAnalysis, list[GameRecord]]:
    """A run whose results follow its values on k = 2, so its siblings have a scale to fit."""
    rng = random.Random(seed + 7)
    agents = ("random", "greedy")
    games, records = [], []
    for g in range(n_games):
        v = rng.uniform(-8, 8)
        games.append(GameAnalysis(
            seed=seed + g, agents=agents, sample_values=[[v, v - 0.5, v - 3.0]] * 10,
            sample_movers=[g % 2] * 10, fielded_instances=0, fielded_moved=0))
        mover_wins = rng.random() < decision_cost.expected(v, 2.0)
        records.append(GameRecord(
            ruleset_hash="x", ruleset_name="x", seed=seed + g, agents=agents, moves=[],
            winner=(g % 2) if mover_wins else 1 - (g % 2), reason="", plies=0))
    return CorpusAnalysis(games=games), records


def _lay_out(root: Path, digest: str, seeds: tuple[int, ...]) -> dict[int, Path]:
    """Each run as a library run sits on disk, and a copy of its analysis where
    `v14-read.py` reads the re-analysed files (`<out>/analysis/<hash>-<seed>/`)."""
    from chess_variant_sandbox.tournament.record import write_records

    outs = {}
    for seed in seeds:
        analysis, records = _fitted_run(seed)
        out = root / "runs" / f"{digest}-{seed}"
        out.mkdir(parents=True)
        write_records(records, out / "games.jsonl")
        analysis.save(out / "analysis.json")
        read = root / "read" / "analysis" / f"{digest}-{seed}"
        read.mkdir(parents=True)
        analysis.save(read / "analysis.json")
        outs[seed] = out
    return outs


def _through_score_run(monkeypatch: pytest.MonkeyPatch, stage: str, seed: int, out: Path,
                       spec: dict) -> CorpusAnalysis:
    """The analysis `corpus._score_run` hands the scorer for this run, read off disk."""
    from chess_variant_sandbox import corpus
    from chess_variant_sandbox.metrics.scoring import WeightsError
    from chess_variant_sandbox.tournament import rate, read_records

    handed: list[CorpusAnalysis] = []

    def scorer(*_args, analysis=None, **_kwargs):
        handed.append(analysis)
        raise WeightsError("held here: only the analysis handed over is read")

    monkeypatch.setattr(corpus, "score_ruleset", scorer)
    digest = out.name.rsplit("-", 1)[0]
    candidate = corpus.Candidate(hash=digest, name="fixture", spec=spec, path=out, stage=stage)
    records = list(read_records(out / "games.jsonl"))
    reading = corpus._score_run(candidate, seed, out, records, rate(records), "v14",
                                CorpusAnalysis.load(out / "analysis.json"), lambda *_: None)
    assert not reading.scored and len(handed) == 1
    return handed[0]


class TestALibraryRunsScale:
    """R-S6d, `v14-registration.md` §11 step 1: `corpus._score_run` reads a library run's
    `decision_cost` on the scale fitted on the game's other runs beside it."""

    @private_paths.needs(READ_SCRIPT, "docs/reference/v13-reference-read.py")
    def test_a_four_run_game_reads_as_the_read_read_it(self, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
        from chess_variant_sandbox.metrics.components import compute_components
        from chess_variant_sandbox.tournament import rate, read_records

        read = private_paths.script(READ_SCRIPT, "v14_read")
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        digest = "f1x7ure00000"
        outs = _lay_out(tmp_path, digest, FOUR)
        game = {"hash": digest, "name": "fixture", "set": "library", "spec": spec,
                "runs": str(tmp_path / "runs")}
        # A label outside the read's `INTERVAL`, so no bootstrap is drawn.
        scored = read.score_game((game, str(tmp_path / "read"), {"current": _json("v14")}))
        expected = {run["seed"]: run["raw"]["decision_cost"] for run in scored["runs"]}
        assert len(expected) == 4

        ks = set()
        for seed, out in outs.items():
            handed = _through_score_run(monkeypatch, "committed", seed, out, spec)
            assert handed.cost_fit is not None and not handed.cost_fit.withheld
            assert handed.cost_fit.rule == "the game's other 3 runs"
            ks |= set(handed.cost_fit.scales.values())
            records = list(read_records(out / "games.jsonl"))
            raw = compute_components(spec, records, rate(records), handed).raw
            assert round(raw["decision_cost"], 6) == expected[seed], seed
        # Four runs, four fits, each on another three: never one scale for the game.
        assert len(ks) == 4

    def test_a_missing_sibling_is_skipped(self, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        outs = _lay_out(tmp_path, "f1x7ure00001", (0, 1000, 2000))
        handed = _through_score_run(monkeypatch, "committed", 0, outs[0], spec)
        assert handed.cost_fit.rule == "the game's other 2 runs"
        siblings = [_fitted_run(s) for s in (1000, 2000)]
        assert handed.cost_fit.scales == decision_cost.others_fit(handed, siblings).scales

    def test_a_one_run_game_leaves_the_fit_unset(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        # A candidate owes one run; its siblings on disk are not its measurement.
        outs = _lay_out(tmp_path, "f1x7ure00002", FOUR)
        handed = _through_score_run(monkeypatch, "candidate", 0, outs[0], spec)
        assert handed.cost_fit is None


# ------------------------------------------------------------------ pieces followed

def _muster(stones: bool) -> dict:
    """R-S14b's `muster`, kings set or stones set (`empty-board-participation.md` §1.2), from
    the record's own script."""
    record = private_paths.script(EMPTY_BOARD_SCRIPT, "empty_board_participation")
    return record.SPECS["muster with stones set" if stones else "muster with kings set"]


def _walk(spec: dict, plies: int, seed: int) -> tuple[PieceFollower, Engine]:
    engine = Engine(spec)
    state = engine.initial_state()
    follower = PieceFollower(engine, state.board, movable_types(spec))
    rng = random.Random(seed)
    for _ in range(plies):
        legal = engine.legal_moves(state)
        if not legal or engine.terminal(state, legal) is not None:
            break
        move = rng.choice(legal)
        after = engine.apply(state, move)
        follower.step(state.board, (move,), after.board)
        state = after
    return follower, engine


class TestPiecesFollowed:
    @private_paths.needs(EMPTY_BOARD_SCRIPT)
    def test_a_piece_that_cannot_move_is_never_counted(self) -> None:
        follower, _ = _walk(_muster(True), 60, 3)
        # Two stones set, then each side places a king, a rook and two knights: eight
        # pieces fielded that can move, and the stones in no count.
        assert follower.fielded == 8
        assert follower.named == 10
        assert 0 < follower.fielded_moved <= follower.fielded

    @private_paths.needs(EMPTY_BOARD_SCRIPT)
    def test_placed_pieces_are_counted(self) -> None:
        follower, _ = _walk(_muster(False), 60, 5)
        # Two kings set, then each side places a rook and two knights: eight fielded.
        assert follower.fielded == 8

    def test_a_castling_partner_is_credited(self) -> None:
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        engine = Engine(spec)
        state = engine.initial_state()
        follower = PieceFollower(engine, state.board, movable_types(spec))
        rng = random.Random(0)
        for _ in range(200):
            legal = engine.legal_moves(state)
            if not legal or engine.terminal(state, legal) is not None:
                break
            castles = [m for m in legal if move_type(m) == MOVE_CASTLE]
            before_moved = set(follower.moved)
            move = castles[0] if castles else rng.choice(legal)
            after = engine.apply(state, move)
            follower.step(state.board, (move,), after.board)
            if castles:
                # The king and its rook, both on the opening board and both unmoved until
                # now (castling needs it): two pieces credited on one move.
                assert len(follower.moved - before_moved) == 2
                return
            state = after
        pytest.skip("no castling reached in this walk")


class TestTheFileWithoutReadings:
    def test_no_new_field_is_written(self) -> None:
        game = GameAnalysis(seed=0, agents=("a", "b"))
        for key in ("sample_values", "sample_movers", "fielded_instances", "fielded_moved"):
            assert key not in game.as_dict()

    def test_with_readings_the_pass_writes_them(self) -> None:
        spec = load_spec(ROOT / "rulesets" / "chess.json")
        engine = Engine(spec)
        evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
        rng = random.Random(1)
        state = engine.initial_state()
        moves = []
        for _ in range(17):
            move = rng.choice(engine.legal_moves(state))
            moves.append(move)
            state = engine.apply(state, move)
        record = GameRecord(ruleset_hash="x", ruleset_name="chess", seed=0,
                            agents=("minimax:nodes=400", "minimax:nodes=400"), moves=moves,
                            winner=None, reason="", plies=len(moves))
        eps = _epsilon_for(engine)
        plain = _analyse_game(engine, evaluator, record, eps)
        read = _analyse_game(engine, evaluator, record, eps, movable_types(spec))
        assert plain.sample_values is None and read.sample_values is not None
        assert len(read.sample_values) == len(read.near_best) == len(read.sample_movers)
        assert [len(v) for v in read.sample_values] == read.legal_counts
        for values, near in zip(read.sample_values, read.near_best, strict=True):
            best = max(values)
            # The persisted values (four places) give the near-best count v13 counted.
            assert abs(sum(1 for v in values if best - v <= eps) - near) <= 1
        assert read.fielded_instances == 32
        stripped = read.as_dict()
        for key in ("sample_values", "sample_movers", "fielded_instances", "fielded_moved"):
            stripped.pop(key)
        assert stripped == plain.as_dict()
