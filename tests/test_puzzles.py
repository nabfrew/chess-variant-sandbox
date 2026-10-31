"""Proven positions (R9a): the candidates, the core's verdict read back, and the record.

Deliberately not chess: a 5x5 game of a royal, a stepper and a walker a side, won by
taking the royal, so every position is a small endgame from the first ply. The verdicts
here are the core's (`Engine.root_scores`); the tests read them, as `puzzles` does.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import puzzles
from chess_variant_sandbox.corpus import RUNS_DIR
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
        same = puzzles.content(state)
        assert same[:3] == (state.key, state.hands, state.checks)
        assert same not in seen and puzzles.name(state) not in seen
        seen |= {same, puzzles.name(state)}


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


# ------------------------------------------------------------------ a position's name (Q993c)


def _reach(engine: Engine, *texts: str) -> Any:
    state = engine.initial_state()
    for text in texts:
        state = engine.apply(state, engine.find_move(state, text))
    return state


# Frozen across rebuilds (`docs/archive/plan-proven.md` §5.3): the site keys people's marks and tries by
# these names, so a core or compiler change that moves one moves every stored mark off its
# position. If this fails, the build changed the core's key (`compile.py` seeds it from the
# rules' hash) or `puzzles.content`: that change owes a store migration like
# `0040_proven_names.sql`, never an edit of these numbers alone.
#
# Q1008a re-pinned them, and owes no migration: `SPEC` is this file's probe and names no
# stored game, so its identity became its normal-form hash and its keys moved, while every
# game that had a name keeps it and its keys (`tests/test_schema_nine.py::FROZEN` pins each
# one's tables, position keys included). Were 10f4889a84a0fb52, 847c52d5ef54fab8,
# c501fb2704dd67af and 85a192d09e9e57d3.
FROZEN_NAMES = {
    (): "b4c9547e94ae83c2",
    ("Sb1-b2",): "a0f740c3fc140de2",
    ("Sb1-b2", "Sd5-d4"): "fd279e98d66437e9",
    ("Sb1-b2", "Sd5-d4", "Tc1-d2"): "43bf4bf284a172c4",
}


def test_a_positions_name_is_pinned_across_a_rebuild(engine: Engine) -> None:
    for texts, frozen in FROZEN_NAMES.items():
        state = _reach(engine, *texts)
        assert puzzles.name(state) == frozen, texts
        assert len(frozen) == puzzles.NAME_LENGTH
    # A fresh engine of the same rules, built again, names them the same.
    assert puzzles.name(_reach(Engine(json.loads(json.dumps(SPEC))), "Sb1-b2")) == (
        FROZEN_NAMES[("Sb1-b2",)])


def test_two_prefixes_reaching_one_position_give_one_name(engine: Engine) -> None:
    one = _reach(engine, "Sb1-b2", "Sd5-d4", "Tc1-d2")
    other = _reach(engine, "Tc1-d2", "Sd5-d4", "Sb1-b2")
    assert one.ply == other.ply and one.board == other.board
    assert puzzles.name(one) == puzzles.name(other) == FROZEN_NAMES[
        ("Sb1-b2", "Sd5-d4", "Tc1-d2")]
    # The way there is not the position: a king that steps out and back reaches the
    # start's board two turns later, and it is the start.
    back = _reach(engine, "Ka1-a2", "Ke5-e4", "Ka2-a1", "Ke4-e5")
    assert back.ply == 4 and back.board == engine.initial_state().board
    assert puzzles.name(back) == FROZEN_NAMES[()]


def test_what_the_key_leaves_out_tells_two_positions_apart(engine: Engine) -> None:
    """The hands, the check counts, a colouring and the points are not in the core's key
    (repetition never compares across them) and are in the name: a position with a piece
    in hand is not the one without it. The states are the engine's with one field set, to
    ask `name` alone; nothing here is a position the engine is asked to play."""
    state = _reach(engine, "Sb1-b2")
    called = puzzles.name(state)
    for changed in (
        state.with_(hands=((1, 0, 0), (0, 0, 0))),
        state.with_(checks=(1, 0)),
        state.with_(colour=bytes([1] + [0] * 24)),
        state.with_(points=(1.0, 0.0)),
        state.with_(side_to_move=1, key=state.key ^ 1),
    ):
        assert puzzles.name(changed) != called
    for same in (state.with_(ply=99, halfmove=7, turns_taken=50), state.with_(unmoved=0),
                 state.with_(repetition=())):
        assert puzzles.name(same) == called


def test_renames_name_every_position_a_record_replays(engine: Engine) -> None:
    moves = [engine.find_move(engine.initial_state(), "Sb1-b2")]
    data = {"positions": [
        {"id": "0-0-1", "moves": moves},
        {"id": "0-1-0", "moves": []},
        {"id": "0-2-1", "moves": [123_456]},
        {"id": "0-3-2", "moves": [moves[0], moves[0]]},
    ]}
    assert puzzles.renames(engine, data) == [
        ("0-0-1", FROZEN_NAMES[("Sb1-b2",)]), ("0-1-0", FROZEN_NAMES[()])]
    assert puzzles.reached(engine, ["x"]) is None


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


LIBRARY = {digest: spec for digest, _name, spec in puzzles.library(".")}


@pytest.mark.parametrize("digest", sorted(LIBRARY))
def test_every_delivered_record_replays_to_the_positions_it_names(digest: str) -> None:
    """Each position's prefix is legal in its game, unfinished, and its stored moves are
    the engine's legal moves there, in order: what the site will look up is what the
    engine will offer. One case a game of the library, which the tree fixes, so every xdist
    worker collects the same cases while the refresher writes records (Q997: one test over
    the growing set ran past the duration guard). Skips where the game has no record here
    (the gate re-runs it there)."""
    path = puzzles.record_path(RUNS_DIR, digest)
    if not path.is_file():
        pytest.skip("no runs/puzzles/ record of this game here")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] in puzzles.FORMATS
    engine = Engine(LIBRARY[digest])
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


def test_every_delivered_record_is_a_game_of_the_library() -> None:
    """The per-game cases above only visit library games; a record of any other game is
    one the cases would miss. Skips where no record is on disk."""
    paths = _delivered()
    if not paths:
        pytest.skip("no runs/puzzles/ record here")
    assert {path.stem for path in paths} <= set(LIBRARY)


def test_a_record_of_a_game_that_left_the_library_is_retired_and_kept(
    tmp_path: Path,
) -> None:
    """An in-place edit re-hashes a game (Q1074bdf): its old record leaves what is offered
    for its work directory, and is kept there, never over a file already there."""
    (tmp_path / "rulesets").mkdir()
    spec = json.loads(Path("rulesets/dull.json").read_text(encoding="utf-8"))
    (tmp_path / "rulesets" / "dull.json").write_text(json.dumps(spec), encoding="utf-8")
    [(kept, _name, _spec)] = puzzles.library(tmp_path)
    runs = tmp_path / "runs"
    gone = "aaaaaaaaaaaa"
    puzzles.write(puzzles.record_path(runs, kept), {"ruleset_hash": kept})
    day = datetime.date(2026, 10, 31)
    for n in (1, 2):
        puzzles.write(puzzles.record_path(runs, gone), {"ruleset_hash": gone, "n": n})
        [moved] = puzzles.retire(runs, tmp_path, today=day)
        assert moved.parent == puzzles.work_dir(runs, gone)
        assert json.loads(moved.read_text(encoding="utf-8"))["n"] == n
    assert sorted(path.name for path in puzzles.work_dir(runs, gone).iterdir()) == [
        "record-2026-10-31-retired.before-2026-10-31.json", "record-2026-10-31-retired.json"]
    assert [path.stem for path in (runs / "puzzles").glob("*.json")] == [kept]
    assert puzzles.retire(runs, tmp_path, today=day) == []
    assert puzzles.retire(tmp_path / "nowhere", tmp_path) == []


# ------------------------------------------------------------------ too easy (Q993a)
#
# `docs/archive/plan-proven.md` §3.1, pre-registered: T1 and T2 read the record's proofs, T3 asks
# the core what every reply to the shortest winning move leaves. The rows below are
# hand-made: what is tested is how the test reads them.


def _row(n: int, proofs: list[str], **extra: Any) -> dict[str, Any]:
    scores = [{"move": index, "text": f"m{index}", "proof": proof,
               **({"ends_in": n + 2 * index} if proof != "open" else {})}
              for index, proof in enumerate(proofs)]
    return {"id": "0-0-0", "result": "won", "ends_in": n, "scores": scores, **extra}


def test_t1_is_a_win_in_one_with_a_choice() -> None:
    assert puzzles.trivial(_row(1, ["won", "open", "open"])) == ()
    assert puzzles.trivial(_row(1, ["won", "won", "open", "open", "open"])) == (puzzles.T1,)
    assert puzzles.trivial(_row(3, ["won", "won", "open", "open", "open"])) == ()


def test_t2_is_half_the_moves_or_more_won() -> None:
    assert puzzles.trivial(_row(3, ["won", "won", "open", "lost"])) == (puzzles.T2,)
    assert puzzles.trivial(_row(3, ["won", "won", "open", "lost", "open"])) == ()
    # Two moves, one winning: exactly half, so the winning move is no find.
    assert puzzles.trivial(_row(3, ["won", "open"])) == (puzzles.T2,)
    assert puzzles.trivial(_row(1, ["won", "won", "won"])) == (puzzles.T1, puzzles.T2)


def test_t3_reads_the_stored_answer_and_unsettled_is_a_say() -> None:
    proofs = ["won", "won", "open", "open", "open"]
    assert puzzles.trivial(_row(3, proofs, defence=puzzles.NONE)) == (puzzles.T3,)
    for kept in (puzzles.SOME, puzzles.UNSETTLED, None):
        assert puzzles.trivial(_row(3, proofs, defence=kept)) == ()


def test_t4_reads_the_stored_answer_and_unsettled_is_a_say() -> None:
    proofs = ["won", "won", "open", "open", "open"]
    assert puzzles.trivial(_row(3, proofs, hasten=puzzles.NONE)) == (puzzles.T4,)
    assert puzzles.trivial(_row(3, proofs, defence=puzzles.NONE, hasten=puzzles.NONE)) == (
        puzzles.T3, puzzles.T4)
    assert puzzles.trivial(_row(3, proofs, defence=puzzles.SOME, hasten=puzzles.NONE)) == (
        puzzles.T4,)
    for kept in (puzzles.SOME, puzzles.UNSETTLED, None):
        assert puzzles.trivial(_row(3, proofs, defence=puzzles.SOME, hasten=kept)) == ()


def test_a_position_not_won_is_never_trivial() -> None:
    assert puzzles.trivial({**_row(1, ["won", "won"]), "result": "lost"}) == ()


def test_the_key_move_is_the_shortest_win_first_in_the_engines_order() -> None:
    row = _row(3, ["open", "won", "won", "won"])
    row["scores"][1]["ends_in"] = 5
    row["scores"][2]["ends_in"] = 3
    row["scores"][3]["ends_in"] = 3
    key = puzzles.key_move(row)
    assert key is not None and key["move"] == 2
    assert puzzles.key_move(_row(3, ["open", "lost"])) is None


def test_t3_is_asked_only_of_what_t1_and_t2_keep_with_two_winning_moves() -> None:
    assert puzzles.asks_defence(_row(3, ["won", "won", "open", "open", "open"]))
    assert not puzzles.asks_defence(_row(3, ["won", "open", "open"]))  # one key
    assert not puzzles.asks_defence(_row(1, ["won", "won", "open", "open", "open"]))  # T1
    assert not puzzles.asks_defence(_row(3, ["won", "won", "open"]))  # T2
    # An answer already stored does not stop it being asked again, T3's or T4's.
    assert puzzles.asks_defence(
        _row(3, ["won", "won", "open", "open", "open"], defence=puzzles.NONE))
    assert puzzles.asks_defence(
        _row(3, ["won", "won", "open", "open", "open"], hasten=puzzles.NONE))


# Found by a search over this spec's four-piece endings at 20,000 nodes, then fixed here:
# each is a proved win with two winning moves of seven, so T1 and T2 keep it.
NO_SAY = {"d2": ("K", 0), "b1": ("S", 0), "e1": ("K", 1), "a1": ("T", 1)}  # n = 3
A_SAY = {"b2": ("K", 0), "c5": ("S", 0), "a1": ("K", 1), "e5": ("T", 1)}  # n = 5


def _asked(engine: Engine, board: dict[str, tuple[str, int]]) -> tuple[Any, dict[str, Any]]:
    state = engine.make_state(board, side_to_move=0)
    row = puzzles.verdict(engine, state, nodes=SMALL)
    assert row["result"] == puzzles.WON and row["winning"] == 2 and len(row["scores"]) == 7
    assert puzzles.asks_defence(row)
    return state, row


def test_t3_finds_no_say_where_every_reply_loses_as_fast(engine: Engine) -> None:
    state, row = _asked(engine, NO_SAY)
    assert row["ends_in"] == 3
    assert puzzles.defence(engine, row, nodes=SMALL, state=state) == puzzles.NONE
    assert puzzles.trivial({**row, "defence": puzzles.NONE}) == (puzzles.T3,)


def test_t3_finds_a_say_where_a_reply_changes_the_count(engine: Engine) -> None:
    state, row = _asked(engine, A_SAY)
    assert row["ends_in"] == 5
    assert puzzles.defence(engine, row, nodes=SMALL, state=state) == puzzles.SOME


class _Unsure(Engine):
    """The core, except that its probe settles nothing: every root score is 0."""

    def root_scores(self, state: Any, values: Any, nodes: int, max_depth: int,
                    goal: float = 0.0, immune: float = 0.0) -> list[float]:
        return [0.0] * len(self.legal_moves(state))


def test_t3_counts_a_reply_it_cannot_settle_as_a_say(engine: Engine) -> None:
    state, row = _asked(engine, NO_SAY)
    assert puzzles.defence(_Unsure(SPEC), row, nodes=SMALL, state=state) == puzzles.UNSETTLED
    assert puzzles.trivial({**row, "defence": puzzles.UNSETTLED}) == ()


def test_t4_finds_no_say_where_a_reply_only_loses_faster(engine: Engine) -> None:
    """A_SAY's say under T3 is a reply that walks into a quicker loss: T4 calls it none."""
    state, row = _asked(engine, A_SAY)
    assert puzzles.defence(engine, row, nodes=SMALL, state=state) == puzzles.SOME
    assert puzzles.hasten(engine, row, nodes=SMALL, state=state) == puzzles.NONE
    assert puzzles.trivial({**row, "defence": puzzles.SOME, "hasten": puzzles.NONE}) == (
        puzzles.T4,)
    # Where T3 finds no say, T4 finds none either: exactly n - 2 is at most n - 2.
    state, row = _asked(engine, NO_SAY)
    assert puzzles.hasten(engine, row, nodes=SMALL, state=state) == puzzles.NONE


class _Slower(Engine):
    """The core, except that every proved win it reports comes two plies later: a reply
    the real core proves lost in n - 2 now reads as putting the win off to n. An exact
    proof cannot report that (the key move's n is its slowest reply's win plus two), so a
    stand-in is how T4's *put it off* branch is reached."""

    def root_scores(self, state: Any, values: Any, nodes: int, max_depth: int,
                    goal: float = 0.0, immune: float = 0.0) -> list[float]:
        scores = super().root_scores(state, values, nodes, max_depth, goal, immune)
        return [score - 2 if score >= puzzles.PROVEN else score for score in scores]


def test_t4_finds_a_say_where_a_reply_puts_the_win_off(engine: Engine) -> None:
    for board in (A_SAY, NO_SAY):
        state, row = _asked(engine, board)
        assert puzzles.hasten(_Slower(SPEC), row, nodes=SMALL, state=state) == puzzles.SOME


def test_t4_counts_a_reply_it_cannot_settle_as_a_say(engine: Engine) -> None:
    state, row = _asked(engine, A_SAY)
    assert puzzles.hasten(_Unsure(SPEC), row, nodes=SMALL, state=state) == puzzles.UNSETTLED
    assert puzzles.trivial({**row, "hasten": puzzles.UNSETTLED}) == ()


def test_t3_does_not_apply_to_one_winning_move(engine: Engine) -> None:
    state = engine.make_state(
        {"a1": ("K", 0), "c3": ("S", 0), "c4": ("K", 1), "e5": ("T", 1)}, side_to_move=0
    )
    row = puzzles.verdict(engine, state, nodes=SMALL)
    assert row["winning"] == 1
    assert puzzles.defence(engine, row, nodes=SMALL, state=state) is None
    assert puzzles.hasten(engine, row, nodes=SMALL, state=state) is None


# A turn of two plies (Q1039): the probe with two moves a turn, and with a duck. Found by a
# search over the four-piece endings, then fixed here. `make_state` builds the last ply of a
# turn; a turn's first ply is reached by a move of the other side's last.
TWO_MOVES: dict[str, Any] = {**SPEC, "name": "puzzle probe, two moves a turn",
                             "rules": {**SPEC["rules"], "turn": {"moves": 2}}}
DUCK: dict[str, Any] = {
    **SPEC, "name": "puzzle probe with a duck",
    "pieces": [*PIECES, {"id": "D", "name": "duck", "value": 0, "moves": []}],
    "rules": {**SPEC["rules"], "duck": {"piece": "D"}},
}
MORE = 60_000  # nodes: the first ply of a two-move turn proves its second winning move here
LAST_PLY = {"c3": ("K", 0), "e5": ("K", 1), "d4": ("S", 0), "a3": ("T", 1)}  # n = 5
# Each the board before the other side's last ply, and that ply: the attacker then begins a
# turn of two (n = 6).
FIRST_PLY_A_SAY = ({"d2": ("K", 0), "a5": ("K", 1), "d4": ("T", 0), "e2": ("T", 1)}, "Te2-d1")
FIRST_PLY_NO_SAY = ({"c4": ("K", 0), "b1": ("K", 1), "c2": ("S", 0), "c1": ("T", 1)}, "Kb1-a1")
DUCK_OWED = {"e3": ("K", 0), "d5": ("S", 0), "c4": ("T", 0), "e5": ("K", 1), "a5": ("T", 1),
             "a1": ("D", 2)}  # n = 4: the attacker's duck, the defender's move and duck, a capture


@pytest.fixture(scope="module")
def two_moves() -> Engine:
    return Engine(TWO_MOVES)


def _first_ply(engine: Engine, board: dict[str, tuple[str, int]], before: str) -> Any:
    state = engine.make_state(board, side_to_move=1)
    [move] = [move for move in engine.legal_moves(state)
              if engine.describe_move(state, move) == before]
    state = engine.apply(state, move)
    assert state.side_to_move == 0
    return state


def test_t3_and_t4_read_the_defenders_two_moves_as_one_reply(two_moves: Engine) -> None:
    """The key move ends the attacker's turn and the defender moves twice: each way it can
    play out the turn is one reply, asked where the attacker is next to move (n - 4 here,
    since the move and the reply took one ply and two)."""
    state = two_moves.make_state(LAST_PLY, side_to_move=0)
    row = puzzles.verdict(two_moves, state, nodes=SMALL)
    assert row["result"] == puzzles.WON and row["ends_in"] == 5 and row["winning"] == 2
    assert puzzles.asks_defence(row)
    after = two_moves.apply(state, int(puzzles.key_move(row)["move"]))
    assert after.side_to_move == 1
    turns = list(puzzles._turns(two_moves, after, 0))
    assert all(outcome is not None or there.side_to_move == 0 for there, outcome in turns)
    assert any(there.ply == after.ply + 2 for there, _outcome in turns)
    assert puzzles.defence(two_moves, row, nodes=SMALL, state=state) == puzzles.SOME
    assert puzzles.hasten(two_moves, row, nodes=SMALL, state=state) == puzzles.NONE


@pytest.mark.parametrize(("case", "defence"), [(FIRST_PLY_A_SAY, puzzles.SOME),
                                               (FIRST_PLY_NO_SAY, puzzles.NONE)])
def test_the_attackers_turn_goes_on_by_its_shortest_win(
    two_moves: Engine, case: tuple[dict[str, tuple[str, int]], str], defence: str
) -> None:
    """The key move leaves the attacker to move again: its turn goes on by the core's
    shortest proved win, and only then does the defender reply, with its whole turn."""
    state = _first_ply(two_moves, *case)
    row = puzzles.verdict(two_moves, state, nodes=MORE)
    assert row["result"] == puzzles.WON and row["ends_in"] == 6 and row["winning"] == 2
    assert puzzles.asks_defence(row)
    assert two_moves.apply(state, int(puzzles.key_move(row)["move"])).side_to_move == 0
    assert puzzles.defence(two_moves, row, nodes=MORE, state=state) == defence
    assert puzzles.hasten(two_moves, row, nodes=MORE, state=state) == puzzles.NONE


def test_a_continuation_the_probe_cannot_settle_is_a_say() -> None:
    engine = Engine(TWO_MOVES)
    state = _first_ply(engine, *FIRST_PLY_NO_SAY)
    row = puzzles.verdict(engine, state, nodes=MORE)
    unsure = _Unsure(TWO_MOVES)
    assert puzzles.defence(unsure, row, nodes=MORE, state=state) == puzzles.UNSETTLED
    assert puzzles.hasten(unsure, row, nodes=MORE, state=state) == puzzles.UNSETTLED


def test_the_duck_half_of_a_reply_is_part_of_it() -> None:
    """The shepherd's duck's shape (Q1039): the attacker places the duck, and the defender's
    reply is its piece move and its duck together."""
    engine = Engine(DUCK)
    state = engine.make_state(DUCK_OWED, side_to_move=0, duck_owed=True)
    row = puzzles.verdict(engine, state, nodes=SMALL)
    assert row["result"] == puzzles.WON and row["ends_in"] == 4 and row["winning"] >= 2
    after = engine.apply(state, int(puzzles.key_move(row)["move"]))
    assert after.side_to_move == 1 and not engine.duck_owed(after)
    assert puzzles.defence(engine, row, nodes=SMALL, state=state) == puzzles.NONE
    assert puzzles.hasten(engine, row, nodes=SMALL, state=state) == puzzles.NONE


def test_t3_replays_the_records_prefix(engine: Engine, records: list[GameRecord]) -> None:
    """Asked without a state, T3 reaches the position by the record's prefix: the same
    answer as asked of the position held."""
    found = list(puzzles.candidates(engine, [(0, records)], near=6))
    rows = [puzzles.prove(engine, candidate, nodes=SMALL) for candidate in found]
    asked = [row for row in rows if row["result"] == puzzles.WON and row["winning"] >= 2]
    assert asked
    for row in asked[:3]:
        state = engine.play(engine.initial_state(), row["moves"])
        assert puzzles.defence(engine, row, nodes=SMALL) == puzzles.defence(
            engine, row, nodes=SMALL, state=state)
        assert puzzles.hasten(engine, row, nodes=SMALL) == puzzles.hasten(
            engine, row, nodes=SMALL, state=state)
    # T5 too, asked of every proved win in the runs.
    won = [row for row in rows if row["result"] == puzzles.WON]
    assert won
    for row in won[:3]:
        state = engine.play(engine.initial_state(), row["moves"])
        assert puzzles.shallow(engine, row, nodes=SMALL) == puzzles.shallow(
            engine, row, nodes=SMALL, state=state)


def test_a_format_three_record_keeps_t3_and_t4_and_names_their_basis(
    engine: Engine, records: list[GameRecord]
) -> None:
    found = list(puzzles.candidates(engine, [(0, records)], near=4))
    rows = [puzzles.prove(engine, candidate, nodes=SMALL) for candidate in found]
    rows = [{**row, "defence": puzzles.defence(engine, row, nodes=SMALL),
             "hasten": puzzles.hasten(engine, row, nodes=SMALL)}
            if puzzles.asks_defence(row) else row for row in rows]
    # The runs' short games set none that T3 asks, so the two fixed endings join them.
    for game, board in enumerate((NO_SAY, A_SAY)):
        state, row = _asked(engine, board)
        rows.append({**row, "id": f"9-{game}-0", "seed": 9, "game": game, "ply": 0,
                     "defence": puzzles.defence(engine, row, nodes=SMALL, state=state),
                     "hasten": puzzles.hasten(engine, row, nodes=SMALL, state=state)})
    data = puzzles.record(engine, "puzzle probe", (0,), rows, built="test",
                          defence_nodes=SMALL, defence_built="test",
                          defence_made=datetime.date(2026, 10, 20),
                          hasten_nodes=SMALL, hasten_built="test",
                          hasten_made=datetime.date(2026, 10, 21))
    assert data["format"] == 4 == puzzles.FORMAT and puzzles.FORMATS == (1, 2, 3, 4)
    assert data["basis"]["defence_nodes"] == SMALL == data["basis"]["hasten_nodes"]
    assert data["basis"]["defence_made"] == "2026-10-20"
    assert data["basis"]["hasten_made"] == "2026-10-21"
    tally = data["counts"]
    asked = [row for row in rows if "defence" in row]
    assert tally["defence_none"] >= 1 and tally["defence_some"] >= 1
    assert tally["hasten_none"] >= 2
    assert sum(tally[f"defence_{answer}"] for answer in puzzles.DEFENCES) == len(asked)
    assert sum(tally[f"hasten_{answer}"] for answer in puzzles.DEFENCES) == len(asked)
    assert tally["trivial"] == sum(bool(puzzles.trivial(row)) for row in rows)
    # T3's none is T4's none: at most n - 2 holds wherever exactly n - 2 does.
    assert all(row["hasten"] == puzzles.NONE for row in asked if row["defence"] == puzzles.NONE)
    # Without T4 the basis and the tally say nothing of it; without either, of both.
    t3 = puzzles.record(engine, "puzzle probe", (0,),
                        [{k: v for k, v in row.items() if k != "hasten"} for row in rows],
                        built="test", defence_nodes=SMALL, defence_built="test")
    assert "hasten_nodes" not in t3["basis"] and "hasten_none" not in t3["counts"]
    assert "defence_none" in t3["counts"] and "trivial" in t3["counts"]
    plain = puzzles.record(
        engine, "puzzle probe", (0,),
        [{k: v for k, v in row.items() if k not in ("defence", "hasten")} for row in rows],
        built="test")
    assert "defence_nodes" not in plain["basis"] and "trivial" not in plain["counts"]


# ------------------------------------------------------- seen at a glance (Q993g)
#
# `docs/archive/plan-proven.md` §3.7, pre-registered: T5 asks the core's two quick players, from
# eight fixed seeds, what they play, and reads each choice against the record's proofs,
# asking the core again of the position an `open` choice leaves.


def test_t5_reads_the_stored_answer() -> None:
    proofs = ["won", "open", "open", "open", "open"]
    assert puzzles.trivial(_row(3, proofs, shallow=puzzles.FOUND)) == (puzzles.T5,)
    for kept in (puzzles.MISSED, None):
        assert puzzles.trivial(_row(3, proofs, shallow=kept)) == ()
    assert puzzles.trivial(_row(3, proofs, hasten=puzzles.NONE, shallow=puzzles.FOUND)) == (
        puzzles.T4, puzzles.T5)


def test_t5_is_asked_of_every_proved_win_t1_to_t4_keep() -> None:
    assert puzzles.asks_shallow(_row(3, ["won", "open", "open"]))  # one key
    assert puzzles.asks_shallow(_row(3, ["won", "won", "open", "open", "open"]))
    assert not puzzles.asks_shallow(_row(1, ["won", "won", "open", "open", "open"]))  # T1
    assert not puzzles.asks_shallow(_row(3, ["won", "won", "open"]))  # T2
    assert not puzzles.asks_shallow(
        _row(3, ["won", "won", "open", "open", "open"], hasten=puzzles.NONE))  # T4
    assert not puzzles.asks_shallow({**_row(3, ["won", "open", "open"]), "result": "lost"})
    # T5's own stored answer does not stop it being asked again, nor T3 or T4 asking.
    found = _row(3, ["won", "open", "open"], shallow=puzzles.FOUND, shallow_moves={})
    assert puzzles.asks_shallow(found)
    assert puzzles.asks_defence(
        _row(3, ["won", "won", "open", "open", "open"], shallow=puzzles.FOUND))


TAKES = {"a1": ("K", 0), "c3": ("S", 0), "c4": ("K", 1), "e5": ("T", 1)}  # Sc3xc4 wins at once


def test_t5_finds_a_win_in_one_from_every_seed(engine: Engine) -> None:
    state = engine.make_state(TAKES, side_to_move=0)
    row = puzzles.verdict(engine, state, nodes=SMALL)
    assert row["single"] and row["ends_in"] == 1
    answer = puzzles.shallow(engine, row, nodes=SMALL, state=state)
    assert answer is not None
    found, played = answer
    assert found == puzzles.FOUND
    key = engine.find_move(state, "Sc3xc4")
    assert set(played) == set(puzzles.SHALLOW_AGENTS)
    assert played["greedy"] == [key] * len(puzzles.SHALLOW_SEEDS)
    # The same position asked again is the same answer: the seeds fix every choice.
    assert puzzles.shallow(engine, row, nodes=SMALL, state=state) == answer


class _Plays(Engine):
    """The core, except that every player plays one fixed move: what is tested is how T5
    reads a choice, not which choice the core makes."""

    def __init__(self, spec: dict[str, Any], move: int, settles: bool = True) -> None:
        super().__init__(spec)
        self.fixed = move
        self.settles = settles

    def select_move(self, state: Any, agent: Any, values: Any, seed: int = 0) -> int | None:
        return self.fixed

    def root_scores(self, state: Any, values: Any, nodes: int, max_depth: int,
                    goal: float = 0.0, immune: float = 0.0) -> list[float]:
        if not self.settles:
            return [0.0] * len(self.legal_moves(state))
        return super().root_scores(state, values, nodes, max_depth, goal, immune)


def test_t5_reads_each_choice_against_the_record_and_asks_again_where_it_is_open(
    engine: Engine,
) -> None:
    state, row = _asked(engine, NO_SAY)
    won = [entry for entry in row["scores"] if entry["proof"] == puzzles.WON]
    rest = [entry for entry in row["scores"] if entry["proof"] != puzzles.WON]
    key = int(won[0]["move"])

    def ask(move: int, record: dict[str, Any], settles: bool = True) -> str:
        answer = puzzles.shallow(_Plays(SPEC, move, settles), record, nodes=SMALL, state=state)
        assert answer is not None
        assert answer[1] == {agent: [move] * 8 for agent in puzzles.SHALLOW_AGENTS}
        return answer[0]

    assert ask(key, row) == puzzles.FOUND  # proved won in the record
    other = int(rest[0]["move"])
    lost = {**row, "scores": [{**entry, "proof": "lost"} if entry["move"] == other else entry
                              for entry in row["scores"]]}
    assert ask(other, lost) == puzzles.MISSED  # proved lost: not asked again
    # The key left `open` in a record: the core, asked of the position it leaves, proves
    # it won for the mover, so the choice wins anyway.
    hidden = {**row, "scores": [{**entry, "proof": "open"} if entry["move"] == key else entry
                                for entry in row["scores"]]}
    assert ask(key, hidden) == puzzles.FOUND
    # The same, where the second ask cannot settle it: a miss, never a find.
    assert ask(key, hidden, settles=False) == puzzles.MISSED


def test_t5_does_not_apply_to_a_position_not_won(engine: Engine) -> None:
    state = engine.make_state(TAKES, side_to_move=0)
    row = puzzles.verdict(engine, state, nodes=SMALL)
    assert puzzles.shallow(engine, {**row, "result": "lost"}, nodes=SMALL, state=state) is None


def test_a_format_four_record_keeps_t5_and_names_its_basis(engine: Engine) -> None:
    state = engine.make_state(TAKES, side_to_move=0)
    row = {**puzzles.verdict(engine, state, nodes=SMALL), "id": "9-0-0", "seed": 9,
           "game": 0, "ply": 0, "moves": []}
    asked = puzzles.shallow(engine, row, nodes=SMALL, state=state)
    assert asked is not None
    rows = [{**row, "shallow": asked[0], "shallow_moves": asked[1]}]
    data = puzzles.record(engine, "puzzle probe", (0,), rows, built="test",
                          shallow_nodes=SMALL, shallow_built="test",
                          shallow_made=datetime.date(2026, 10, 20))
    basis = data["basis"]
    assert basis["shallow_agents"] == list(puzzles.SHALLOW_AGENTS)
    assert basis["shallow_seeds"] == list(range(8))
    assert basis["shallow_nodes"] == SMALL and basis["shallow_made"] == "2026-10-20"
    assert data["counts"]["shallow_found"] == 1 and data["counts"]["shallow_missed"] == 0
    assert data["counts"]["trivial"] == 1
    assert data["positions"][0]["shallow_moves"] == asked[1]
    plain = puzzles.record(engine, "puzzle probe", (0,), rows, built="test")
    assert "shallow_nodes" not in plain["basis"] and "shallow_found" not in plain["counts"]


# ------------------------------------------------- the count, over the seven (§3.2)

SHAPES = Path(__file__).with_name("_proven_shapes.json")
"""Each delivered record's won positions as `[n, L, W, defence, hasten, shallow]`: what T1
to T5 read (`defence`, `hasten` and `shallow` null where T3, T4 and T5 were not asked), so
the count holds without the records."""

# `docs/archive/plan-proven.md` §3.2, by game: positions, T1, T2, T1 or T2, left, left with one key.
COUNT_32 = {
    "intransitive": (112, 35, 58, 59, 53, 16),
    "los alamos": (148, 38, 91, 93, 55, 12),
    "cylinder chess": (128, 30, 29, 37, 91, 19),
    "quarry": (112, 32, 53, 60, 52, 15),
    "reckoning": (157, 33, 84, 88, 69, 21),
    "wolves": (59, 17, 21, 29, 30, 14),
    "glinski hexagonal chess": (94, 28, 21, 35, 59, 22),
}


def _shaped(
    n: int, legal: int, won: int, answer: str | None, hastens: str | None = None,
    seen: str | None = None,
) -> dict[str, Any]:
    proofs = ["won"] * won + ["open"] * (legal - won)
    return _row(n, proofs, **({"defence": answer} if answer else {}),
                **({"hasten": hastens} if hastens else {}),
                **({"shallow": seen} if seen else {}))


def _shapes() -> dict[str, Any]:
    return dict(json.loads(SHAPES.read_text(encoding="utf-8")))


def test_the_count_of_plan_proven_32_holds_over_the_fixture() -> None:
    shapes = _shapes()
    assert set(shapes) == set(COUNT_32)
    for name, game in shapes.items():
        rows = [_shaped(*shape) for shape in game["positions"]]
        parts = [set(puzzles.trivial(row)) for row in rows]
        assert all(len(shape) == 6 for shape in game["positions"]), name
        t1 = sum(puzzles.T1 in found for found in parts)
        t2 = sum(puzzles.T2 in found for found in parts)
        either = sum(bool(found & {puzzles.T1, puzzles.T2}) for found in parts)
        left = [row for row, found in zip(rows, parts, strict=True)
                if not found & {puzzles.T1, puzzles.T2}]
        one = sum(puzzles.winning_moves(row) == 1 for row in left)
        assert (len(rows), t1, t2, either, len(left), one) == COUNT_32[name], name
        # T3 was asked of exactly the positions §3.3 costs: kept, with two winning moves.
        assert [puzzles.asks_defence(row) for row in rows] == [
            shape[3] is not None for shape in game["positions"]], name


# `proven-positions.md` §6, by game: T3 asked, `none`, `some`, `unsettled`, and offered.
COUNT_T3 = {
    "intransitive": (37, 16, 21, 0, 37),
    "los alamos": (43, 34, 9, 0, 21),
    "cylinder chess": (72, 67, 5, 0, 24),
    "quarry": (37, 29, 8, 0, 23),
    "reckoning": (48, 35, 13, 0, 34),
    "wolves": (16, 15, 1, 0, 15),
    "glinski hexagonal chess": (37, 30, 7, 0, 29),
}


def test_the_t3_count_of_proven_positions_6_holds_over_the_fixture() -> None:
    for name, game in _shapes().items():
        answers = [shape[3] for shape in game["positions"] if shape[3] is not None]
        # The offer as it stood with T1 to T3 alone (Q993a), before T4's answers.
        offered = sum(not puzzles.trivial(_shaped(*shape[:4])) for shape in game["positions"])
        assert (len(answers), answers.count(puzzles.NONE), answers.count(puzzles.SOME),
                answers.count(puzzles.UNSETTLED), offered) == COUNT_T3[name], name


# `proven-positions.md` §6.2, by game: T4 asked, `none`, `some`, `unsettled`, offered, and
# of those offered how many have one winning move. A fact, not a prediction (§3.6).
COUNT_T4 = {
    "intransitive": (37, 37, 0, 0, 16, 16),
    "los alamos": (43, 43, 0, 0, 12, 12),
    "cylinder chess": (72, 72, 0, 0, 19, 19),
    "quarry": (37, 37, 0, 0, 15, 15),
    "reckoning": (48, 48, 0, 0, 21, 21),
    "wolves": (16, 16, 0, 0, 14, 14),
    "glinski hexagonal chess": (37, 37, 0, 0, 22, 22),
}


def test_the_t4_count_of_proven_positions_62_holds_over_the_fixture() -> None:
    for name, game in _shapes().items():
        shapes = game["positions"]
        # T4 was asked of exactly the positions T3 was (`asks_defence`).
        assert [shape[3] is None for shape in shapes] == [shape[4] is None for shape in shapes]
        answers = [shape[4] for shape in shapes if shape[4] is not None]
        # The offer as it stood with T1 to T4 alone (Q993f), before T5's answers.
        kept = [row for row in map(lambda shape: _shaped(*shape[:5]), shapes)
                if not puzzles.trivial(row)]
        one = sum(puzzles.winning_moves(row) == 1 for row in kept)
        assert (len(answers), answers.count(puzzles.NONE), answers.count(puzzles.SOME),
                answers.count(puzzles.UNSETTLED), len(kept), one) == COUNT_T4[name], name


# `proven-positions.md` §6.3, by game: T5 asked, `found` (left out), `missed`, offered, and
# of those offered how many have one winning move. `docs/archive/plan-proven.md` §3.7 predicted the
# offered column before it was counted; the grade is in the record.
COUNT_T5 = {
    "intransitive": (16, 13, 3, 3, 3),
    "los alamos": (12, 11, 1, 1, 1),
    "cylinder chess": (19, 15, 4, 4, 4),
    "quarry": (15, 14, 1, 1, 1),
    "reckoning": (21, 13, 8, 8, 8),
    "wolves": (14, 13, 1, 1, 1),
    "glinski hexagonal chess": (22, 19, 3, 3, 3),
}


def test_the_t5_count_of_proven_positions_63_holds_over_the_fixture() -> None:
    for name, game in _shapes().items():
        shapes = game["positions"]
        # T5 was asked of exactly what T1 to T4 keep (`asks_shallow`).
        assert [puzzles.asks_shallow(_shaped(*shape[:5])) for shape in shapes] == [
            shape[5] is not None for shape in shapes], name
        answers = [shape[5] for shape in shapes if shape[5] is not None]
        kept = [row for row in map(lambda shape: _shaped(*shape), shapes)
                if not puzzles.trivial(row)]
        one = sum(puzzles.winning_moves(row) == 1 for row in kept)
        assert (len(answers), answers.count(puzzles.FOUND), answers.count(puzzles.MISSED),
                len(kept), one) == COUNT_T5[name], name


def test_the_fixture_is_the_delivered_records() -> None:
    """Skips where the records are not on disk (the gate re-runs it there).

    The fixture is the seven records R9a proved and Q993a to Q993g answered, by hand. A
    record the upkeep writes (`puzzles.make`, Q993d: a new game's, or one of the seven
    proved again) names its `near_min`, and is no record of that snapshot, so it is not
    read here; `test_refresher_proofs.py` holds what `make` writes."""
    paths = _delivered()
    if not paths:
        pytest.skip("no runs/puzzles/ record here")
    shapes = _shapes()
    by_hash = {game["hash"]: name for name, game in shapes.items()}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.stem not in by_hash or "near_min" in (data.get("basis") or {}):
            continue
        game = shapes[by_hash[path.stem]]
        assert game["hash"] == path.stem
        assert game["positions"] == [
            [row["ends_in"], len(row["scores"]), puzzles.winning_moves(row),
             row.get("defence"), row.get("hasten"), row.get("shallow")]
            # The sift (Q1084) moves rows to `set_aside` and deletes none: the record's
            # proofs are its positions and what is set aside, in the record's order.
            for row in sorted([*data["positions"], *(data.get("set_aside") or [])],
                              key=lambda row: (row["seed"], row["game"], row["ply"]))
            if row["result"] == puzzles.WON
        ]
