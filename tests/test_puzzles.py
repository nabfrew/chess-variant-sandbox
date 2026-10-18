"""Proven positions (R9a): the candidates, the core's verdict read back, and the record.

Deliberately not chess: a 5x5 game of a royal, a stepper and a walker a side, won by
taking the royal, so every position is a small endgame from the first ply. The verdicts
here are the core's (`Engine.root_scores`); the tests read them, as `puzzles` does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import puzzles
from chess_variant_sandbox.corpus import RUNS_DIR, find
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.play import choose_move
from chess_variant_sandbox.tournament.record import GameRecord

PIECES: list[dict[str, Any]] = [
    {"id": "K", "name": "king", "value": 0, "royal": True,
     "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
    {"id": "S", "name": "stepper", "value": 1,
     "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
    {"id": "T", "name": "walker", "value": 1,
     "moves": [{"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"}]},
]

SPEC: dict[str, Any] = {
    "schema_version": 4,
    "name": "puzzle probe",
    "board": {"width": 5, "height": 5},
    "pieces": PIECES,
    "setup": {"mirror": False, "placements": [
        {"piece": "K", "player": 0, "squares": ["a1"]},
        {"piece": "K", "player": 1, "squares": ["e5"]},
        {"piece": "S", "player": 0, "squares": ["b1"]},
        {"piece": "S", "player": 1, "squares": ["d5"]},
        {"piece": "T", "player": 0, "squares": ["c1"]},
        {"piece": "T", "player": 1, "squares": ["c5"]},
    ]},
    "rules": {"win": "royal_capture"},
}

SMALL = 20_000  # nodes: enough for a capture one ply ahead, a test's worth of time


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(SPEC)


def _game(engine: Engine, seed: int, agents: tuple[str, str], limit: int = 120) -> GameRecord:
    """A game the core's own players play, recorded as a library run records one."""
    state = engine.initial_state()
    moves: list[int] = []
    while engine.terminal(state) is None and len(moves) < limit:
        move = choose_move(engine, agents[state.side_to_move], state, seed=seed * 1000 + len(moves))
        assert move is not None
        moves.append(move)
        state = engine.apply(state, move)
    outcome = engine.terminal(state)
    return GameRecord(
        ruleset_hash=engine.spec.hash, ruleset_name="puzzle probe", seed=seed, agents=agents,
        moves=moves, winner=None if outcome is None else outcome.winner,
        reason="" if outcome is None else outcome.reason, plies=len(moves),
    )


@pytest.fixture(scope="module")
def records(engine: Engine) -> list[GameRecord]:
    games = [_game(engine, seed, ("minimax:nodes=400", "random")) for seed in range(3)]
    assert all(game.winner is not None for game in games)
    return games


def _drawn(record: GameRecord) -> GameRecord:
    return GameRecord(**{**record.__dict__, "winner": None, "reason": "ply_limit"})


# ------------------------------------------------------------------ candidates


def test_every_candidate_is_a_distinct_small_unfinished_position(
    engine: Engine, records: list[GameRecord]
) -> None:
    found = list(puzzles.candidates(engine, [(0, records)]))
    assert found
    seen = set()
    for candidate in found:
        state = engine.play(engine.initial_state(), candidate.moves)
        assert engine.terminal(state) is None
        assert puzzles.pieces_in_play(state) == candidate.pieces <= puzzles.MAX_PIECES
        assert candidate.moves == tuple(records[candidate.game].moves[: candidate.ply])
        same = (state.key, state.hands, state.checks)
        assert same not in seen
        seen.add(same)


def test_a_position_with_more_pieces_than_the_bound_is_not_a_candidate(
    engine: Engine, records: list[GameRecord]
) -> None:
    every = list(puzzles.candidates(engine, [(0, records)]))
    fewer = list(puzzles.candidates(engine, [(0, records)], max_pieces=5))
    assert fewer == [c for c in every if c.pieces <= 5]
    assert len(fewer) < len(every)  # the six-piece start is one of them
    assert list(puzzles.candidates(engine, [(0, records)], max_pieces=1)) == []


def test_near_keeps_only_the_recorded_winner_to_move_close_to_the_end(
    engine: Engine, records: list[GameRecord]
) -> None:
    near = list(puzzles.candidates(engine, [(0, records)], near=6))
    assert near
    for candidate in near:
        record = records[candidate.game]
        state = engine.play(engine.initial_state(), candidate.moves)
        assert state.side_to_move == record.winner
        assert len(record.moves) - candidate.ply <= 6


def test_near_reads_nothing_from_a_game_without_a_winner(
    engine: Engine, records: list[GameRecord]
) -> None:
    drawn = [_drawn(record) for record in records]
    assert list(puzzles.candidates(engine, [(0, drawn)], near=6)) == []
    assert list(puzzles.candidates(engine, [(0, drawn)]))  # the plain count still sees them


def test_the_list_is_a_function_of_the_runs_alone(
    engine: Engine, records: list[GameRecord]
) -> None:
    once = list(puzzles.candidates(engine, [(0, records), (1000, records)]))
    again = list(puzzles.candidates(engine, [(0, records), (1000, records)]))
    assert once == again
    # The second run repeats the first, so it adds nothing: first reached names it.
    assert all(candidate.seed == 0 for candidate in once)


def test_a_game_of_rounds_is_refused() -> None:
    rounds = Engine({**SPEC, "schema_version": 26,
                     "rules": {"win": "royal_capture", "simultaneous": {}}})
    assert rounds.simultaneous
    with pytest.raises(ValueError):
        list(puzzles.candidates(rounds, []))


# ------------------------------------------------------------------ sample


def _candidates(games: dict[int, int]) -> list[puzzles.Candidate]:
    return [puzzles.Candidate(0, game, ply, 4, ()) for game, n in games.items() for ply in range(n)]


def test_the_sample_takes_whole_games_within_the_cap() -> None:
    found = _candidates({game: 5 for game in range(20)})  # 100 positions in 20 games
    taken = puzzles.sample(found, cap=30)
    assert len(taken) <= 30
    by_game: dict[int, int] = {}
    for candidate in taken:
        by_game[candidate.game] = by_game.get(candidate.game, 0) + 1
    assert set(by_game.values()) == {5}  # every chosen game whole
    assert sorted(by_game) == list(range(0, 20, 4))  # every fourth game, from the first
    assert puzzles.sample(found, cap=30) == taken


def test_a_sample_under_the_cap_is_everything() -> None:
    found = _candidates({0: 3, 1: 2})
    assert puzzles.sample(found, cap=240) == found


# ------------------------------------------------------------------ the verdict


def test_a_capture_of_the_royal_is_proved_won_in_one(engine: Engine) -> None:
    # The stepper on c3 takes the royal on c4; nothing else wins at once, and the royal
    # on c4 takes the stepper back if it is left there.
    state = engine.make_state(
        {"a1": ("K", 0), "c3": ("S", 0), "c4": ("K", 1), "e5": ("T", 1)}, side_to_move=0
    )
    row = puzzles.verdict(engine, state, nodes=SMALL)
    assert row["result"] == puzzles.WON
    assert row["ends_in"] == 1
    winners = [entry for entry in row["scores"] if entry["proof"] == puzzles.WON]
    assert [entry["text"] for entry in winners] == ["Sc3xc4"]
    assert winners[0]["move"] == engine.find_move(state, "Sc3xc4")
    assert row["winning"] == 1 and row["single"]
    assert [entry["move"] for entry in row["scores"]] == engine.legal_moves(state)


def test_the_scores_are_the_cores_in_its_own_order(engine: Engine) -> None:
    from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES
    from chess_variant_sandbox.tournament.play import eval_values

    state = engine.make_state(
        {"a1": ("K", 0), "c3": ("S", 0), "c4": ("K", 1), "e5": ("T", 1)}, side_to_move=0
    )
    row = puzzles.verdict(engine, state, nodes=SMALL)
    asked = engine.root_scores(state, eval_values(engine.spec), SMALL, puzzles.MAX_DEPTH,
                               GOAL_PIECES, IMMUNE_PIECES)
    assert [entry["score"] for entry in row["scores"]] == asked


def test_a_forced_position_is_not_asked() -> None:
    # Two by two: the royal is boxed by its own stepper and walker, the walker's one
    # diagonal is its own stepper, and the stepper's one way out takes the enemy royal.
    boxed = Engine({**SPEC, "board": {"width": 2, "height": 2}, "setup": {
        "mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "S", "player": 0, "squares": ["a2"]},
            {"piece": "T", "player": 0, "squares": ["b1"]},
            {"piece": "K", "player": 1, "squares": ["b2"]},
        ]}})
    state = boxed.initial_state()
    assert len(boxed.legal_moves(state)) == 1
    assert puzzles.verdict(boxed, state, nodes=SMALL) == {"side": 0, "result": "forced"}


@pytest.mark.parametrize(
    ("score", "proof"),
    [(puzzles.PROVEN, "won"), (puzzles.PROVEN - 1, "open"), (0.0, "open"),
     (-puzzles.PROVEN, "lost"), (puzzles.WIN_SCORE - 7, "won")],
)
def test_a_score_proves_only_past_the_bar(score: float, proof: str) -> None:
    assert puzzles.proof_of(score) == proof


def test_ends_in_counts_from_the_position() -> None:
    assert puzzles.ends_in(puzzles.WIN_SCORE - 47, 40) == 7
    assert puzzles.ends_in(-(puzzles.WIN_SCORE - 47), 40) == 7


# ------------------------------------------------------------------ the record


def test_the_record_keeps_the_proofs_and_counts_the_rest(
    engine: Engine, records: list[GameRecord]
) -> None:
    found = list(puzzles.candidates(engine, [(0, records)], near=4))
    rows = [puzzles.prove(engine, candidate, nodes=SMALL) for candidate in found]
    data = puzzles.record(engine, "puzzle probe", (0,), rows, built="test", distinct=99)
    assert data["format"] == puzzles.FORMAT
    assert data["ruleset_hash"] == engine.spec.hash
    assert data["basis"]["nodes"] == puzzles.NODES  # the default the campaign proves at
    assert data["basis"]["built"] == "test"
    assert data["basis"]["distinct"] == 99
    assert data["counts"]["candidates"] == len(rows)
    tally = data["counts"]
    assert tally["won"] + tally["lost"] + tally["open"] + tally["forced"] == len(rows)
    assert len(data["positions"]) == tally["won"] + tally["lost"]
    assert tally["won"] > 0
    for position in data["positions"]:
        assert position["result"] in ("won", "lost")
        assert position["moves"] == list(records[position["game"]].moves[: position["ply"]])
    order = [(p["seed"], p["game"], p["ply"]) for p in data["positions"]]
    assert order == sorted(order)


def test_a_record_is_written_whole(tmp_path: Path) -> None:
    path = tmp_path / "puzzles" / "abc.json"
    puzzles.write(path, {"format": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"format": 1}
    assert not path.with_suffix(".tmp").exists()


# ------------------------------------------------------------------ what was delivered


def _delivered() -> list[Path]:
    return sorted((Path(RUNS_DIR) / "puzzles").glob("*.json"))


def test_every_delivered_record_replays_to_the_positions_it_names() -> None:
    """Each position's prefix is legal in its game, unfinished, and its stored moves are
    the engine's legal moves there, in order: what the site will look up is what the
    engine will offer. Skips where no record is on disk (the gate re-runs it there)."""
    paths = _delivered()
    if not paths:
        pytest.skip("no runs/puzzles/ record here")
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["format"] == puzzles.FORMAT
        engine = Engine(find(data["ruleset_hash"]).spec)
        assert engine.spec.hash == data["ruleset_hash"] == path.stem
        for position in data["positions"]:
            state = engine.play(engine.initial_state(), position["moves"])
            assert engine.terminal(state) is None, position["id"]
            assert state.side_to_move == position["side"]
            assert [m["move"] for m in position["scores"]] == engine.legal_moves(state)
            proofs = {m["proof"] for m in position["scores"]}
            if position["result"] == "won":
                assert "won" in proofs
            else:
                assert proofs == {"lost"}
