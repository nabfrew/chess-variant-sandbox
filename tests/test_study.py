"""The study record (R-W4b1, `docs/plan-website.md` *The study page*): the counting half of
the behaviour test, on a spec and games the core plays here (no library game, no chess);
that every delivered library record replays and matches its basis; and that the module
names no piece and no game."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import stored_games, study
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0,)
AGENTS = ("random", "minimax:nodes=40", "minimax:nodes=80", "minimax:nodes=160")
BANNED = ("theory", "book", "best", "recommended", "win rate", "accuracy")


def _spec() -> dict[str, Any]:
    """A small game written here: a royal stepper, a slider and a leaper on four by four."""
    return {
        "schema_version": 31,
        "name": "study fixture",
        "board": {"width": 4, "height": 4},
        "pieces": [
            {"id": "L", "name": "leader", "value": 0, "royal": True, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
                 "mode": "both"}]},
            {"id": "A", "name": "arrow", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}]},
            {"id": "F", "name": "frog", "value": 3, "moves": [
                {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "A", "squares": ["a1"]},
            {"piece": "F", "squares": ["b1"]},
            {"piece": "L", "squares": ["c1"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 60,
                  "repetition_draw": 3},
    }


@pytest.fixture(scope="module")
def played(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    """The core plays one run of the fixture game; its records are saved as a measurement's."""
    spec = _spec()
    runs = tmp_path_factory.mktemp("runs")
    config = TournamentConfig(agents=AGENTS, games_per_pairing=4, seed=0, workers=1)
    run_tournament(spec, config).save(runs / f"{spec_hash(spec)}-0")
    return spec, runs


@pytest.fixture(scope="module")
def record(played: tuple[dict[str, Any], Path]) -> dict[str, Any]:
    spec, runs = played
    return study.build(spec, runs, SEEDS)


def _pool(played: tuple[dict[str, Any], Path]) -> list[stored_games.Stored]:
    spec, runs = played
    return [game for game in stored_games.read(runs, spec_hash(spec), SEEDS)
            if study.searches(game.agents[0]) and study.searches(game.agents[1])]


def _steps(engine: Engine, game: stored_games.Stored) -> list[tuple[int, ...]]:
    return [step for _state, step in engine.steps(engine.initial_state(), game.moves)]


def _rows(row: dict[str, Any], path: tuple[tuple[int, ...], ...] = (), depth: int = 0) -> list[Any]:
    found = []
    for child in row.get("next") or []:
        here = (*path, tuple(child["moves"]))
        found.append((here, child, depth + 1))
        found.extend(_rows(child, here, depth + 1))
    return found


def test_the_roots_count_is_the_games_with_a_searching_player_in_both_seats(
    played: tuple[dict[str, Any], Path], record: dict[str, Any],
) -> None:
    pool = _pool(played)
    assert record["pool"] == len(pool) == 3 * 3 * 4
    assert record["games"] == len(AGENTS) ** 2 * 4
    assert record["left_out"] == record["games"] - record["pool"]
    tree = record["tree"]
    assert tree["games"] == len(pool)
    assert tree["results"] == [sum(1 for g in pool if g.winner == 0),
                               sum(1 for g in pool if g.winner is None),
                               sum(1 for g in pool if g.winner == 1)]


def test_every_step_counts_the_games_that_begin_with_its_moves(
    played: tuple[dict[str, Any], Path], record: dict[str, Any],
) -> None:
    spec, _runs = played
    engine = Engine(spec)
    lines = [(game, _steps(engine, game)) for game in _pool(played)]
    rows = _rows(record["tree"])
    assert rows, "the fixture's tree keeps no step"
    for path, row, depth in rows:
        through = [(game, steps) for game, steps in lines if tuple(steps[:len(path)]) == path]
        assert row["games"] == len(through) >= study.STUDY_FLOOR
        assert sum(row["results"]) == row["games"]
        assert row["results"] == [sum(1 for g, _s in through if g.winner == 0),
                                  sum(1 for g, _s in through if g.winner is None),
                                  sum(1 for g, _s in through if g.winner == 1)]
        assert row["ended"] == sum(1 for _g, steps in through if len(steps) == len(path))
        assert sum(count for _seat, _player, count in row["players"]) == row["games"]
        if depth < study.STUDY_DEPTH:
            below = sum(child["games"] for child in row["next"])
            below += (row.get("other") or {}).get("games", 0)
            assert below == row["games"] - row["ended"]


def test_the_roots_next_steps_and_other_moves_sum_to_its_count(record: dict[str, Any]) -> None:
    tree = record["tree"]
    below = sum(child["games"] for child in tree["next"]) + (tree.get("other") or {}).get("games", 0)
    assert below == tree["games"] - tree["ended"]
    assert [child["games"] for child in tree["next"]] == sorted(
        (child["games"] for child in tree["next"]), reverse=True)


def test_each_example_is_a_pool_game_with_the_records_moves_taken_once(
    played: tuple[dict[str, Any], Path], record: dict[str, Any],
) -> None:
    by_place = {game.order: game for game in _pool(played)}
    picked = record["examples"]
    assert picked and len(picked) + len(record["missing_examples"]) == len(study.RULES)
    assert [e["rule"] for e in picked] == [r for r in study.RULES if r in {e["rule"] for e in picked}]
    assert len({(e["seed"], e["place"]) for e in picked}) == len(picked)
    assert len({tuple(e["moves"]) for e in picked}) == len(picked)
    for example in picked:
        game = by_place[(example["seed"], example["place"])]
        assert example["moves"] == game.moves
        assert example["winner"] == game.winner
        assert example["players"] == list(game.agents)


def test_the_record_replays_and_a_changed_step_does_not(
    played: tuple[dict[str, Any], Path], record: dict[str, Any],
) -> None:
    spec, _runs = played
    engine = Engine(spec)
    assert study.check(engine, record) is None
    broken = json.loads(json.dumps(record))
    first = broken["tree"]["next"][0]
    first["next"] = [dict(first["next"][0], moves=list(first["moves"]))] if first["next"] else []
    if not first["next"]:
        broken["examples"][0]["moves"] = [first["moves"][0]] * 2
    assert study.check(engine, broken) is not None


def test_a_record_is_owed_when_its_games_or_its_version_move(
    played: tuple[dict[str, Any], Path], record: dict[str, Any], tmp_path: Path,
) -> None:
    spec, runs = played
    assert study.owes(record, spec, runs, SEEDS) is None
    older = dict(record, basis=dict(record["basis"], study_version=study.STUDY_VERSION - 1))
    assert study.owes(older, spec, runs, SEEDS) is not None
    assert study.owes(record, spec, runs, (0, 1000)) is not None
    copy = tmp_path / f"{spec_hash(spec)}-0"
    copy.mkdir()
    body = (runs / copy.name / "games.jsonl").read_text(encoding="utf-8")
    (copy / "games.jsonl").write_text(body + body.splitlines()[0] + "\n", encoding="utf-8")
    assert study.owes(record, spec, tmp_path, SEEDS) is not None


def test_a_game_with_no_records_answers_its_reason(tmp_path: Path) -> None:
    made = study.build(_spec(), tmp_path, SEEDS)
    assert made["none"] == study.NO_RECORDS
    assert "tree" not in made
    assert study.owes(made, _spec(), tmp_path, SEEDS) is None
    assert study.check(Engine(_spec()), made) is None


def test_the_players_and_their_budgets_are_read_from_the_records(record: dict[str, Any]) -> None:
    found = {player["agent"]: player for player in record["players"]}
    assert set(found) == set(AGENTS)
    assert not found["random"]["searches"] and found["random"]["budget"] is None
    assert [found[a]["budget"] for a in AGENTS[1:]] == [40, 80, 160]


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def test_no_sentence_the_record_holds_says_a_banned_word(record: dict[str, Any]) -> None:
    sentences = [s for s in _strings(record) if " " in s]
    sentences += [study.NO_RECORDS]
    for sentence in sentences:
        for word in BANNED:
            assert not re.search(rf"\b{word}\b", sentence.lower()), (word, sentence)


def test_the_module_names_no_piece_and_no_game() -> None:
    named: set[str] = set()
    for folder in ("rulesets", "archive"):
        for path in sorted((ROOT / folder).glob("*.json")):
            held = json.loads(path.read_text(encoding="utf-8"))
            named.add(path.stem.lower())
            named.add(str(held.get("name", "")).lower())
            named.update(str(piece["name"]).lower()
                         for piece in held.get("pieces", []) if piece.get("name"))
    named -= {""}
    for module in ("study", "stored_games"):
        tree = ast.parse((ROOT / f"chess_variant_sandbox/{module}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body[0] = ast.Pass()
        code = ast.unparse(tree).lower()
        found = sorted(word for word in named if re.search(rf"\b{re.escape(word)}\b", code))
        assert not found, (module, found)


@pytest.mark.slow
def test_every_library_record_replays_and_matches_its_basis() -> None:
    """Over the main checkout's `runs/study/` (skipped where it is absent): every Library
    and Baseline game has a record, and each replays and is owed nothing."""
    from chess_variant_sandbox.corpus import LIBRARY_SEEDS
    from chess_variant_sandbox.measurement import main_runs_dir

    runs = main_runs_dir() or ROOT / "runs"
    if not (runs / study.DIRECTORY).is_dir():
        pytest.skip(f"no study records in {runs}")
    for digest, name, spec in study.library(ROOT):
        data = study.load(study.record_path(runs, digest))
        assert data is not None, f"{name} has no study record"
        assert study.owes(data, spec, runs, LIBRARY_SEEDS) is None, name
        assert study.check(Engine(spec), data) is None, name
