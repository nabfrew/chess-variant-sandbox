"""Schema 28 (Q943b): the round players' term for a royal under a faster attacker, and the
pass's speed tallies.

`docs/plan-speed.md` §4.4 and §5's Q943b row. Between unequal speeds the faster move is made
first, so a royal that a strictly faster enemy piece reaches can neither run nor strike back
first (§2.5 A, §3.2): it falls next round unless its side takes the attacker with a piece
faster still. A depth-one round player saw nothing of that past its own matrix. Both sides of
the boundary now price it (`eval.rs::caught_royal_falls`, its oracle
`agents/evaluation.py::caught_royal_falls`) at `RESTED_ROYAL_PIECES` mean pieces, once beside the rested royal:
held to each other on the speed test specs, the test itself on hand-built positions (the
answer, a second attacker, an equal-speed attacker, a second royal), a `greedy` player that
keeps its royal's shield where taking a piece would open the line, and every round player on
the round games without a speed playing what it played before the term.

Then the tallies (`SPEED_COUNTS`): an ordered round, a slower move void, lost, unplayed, read
off the core's events, and absent from a game without speed, whose `analysis.json` is the
bytes it was.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_matrix_players import core_round, oracle_round, walk
from test_speed import gardner as speed_gardner

from chess_variant_sandbox.agents import evaluation
from chess_variant_sandbox.agents.evaluation import caught_royal_falls
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.engine import _square_name
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
from chess_variant_sandbox.tournament.analysis import (
    ROUND_COUNTS,
    SPEED_COUNTS,
    _count_round,
    analyse_corpus,
)

HERE = Path(__file__).parent
ROOT = HERE.parent

# Q943a's nine speed test specs (`tests/_simultaneous_speed_*.json`).
SPEED_SPECS = (
    "atomic", "circe", "collide", "gardner", "merge", "portals", "resting", "shogi", "wild",
)


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_speed_{name}.json").read_text(encoding="utf-8"))


def place(engine: Engine, pieces: dict[str, tuple[str, int]]) -> GameState:
    state = engine.make_state(pieces)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def lists(engine: Engine, state: GameState) -> tuple[list[int], list[int]]:
    return (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))


def falls(engine: Engine, state: GameState, side: int) -> bool:
    sim = engine.spec.simultaneous
    assert sim is not None and sim.speed
    return caught_royal_falls(engine.spec, state, side, lists(engine, state), sim.speed)


def ranked(speeds: dict[str, int]) -> Engine:
    """Gardner in rounds with these speeds, every other type at 0."""
    spec = speed_gardner(speeds)
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


# --------------------------------------------------------------- the test itself

# Player 1's king (speed 0) on a5, the a-file open to player 0's rook (speed 1) on a1.
OPEN = {"e1": ("K", 0), "a1": ("R", 0), "a5": ("K", 1), "e4": ("P", 1)}


def test_a_royal_a_faster_piece_reaches_falls() -> None:
    engine = ranked({"R": 1, "N": 2})
    assert falls(engine, place(engine, OPEN), 1)
    assert not falls(engine, place(engine, OPEN), 0)
    # Shielded: the rook does not reach it.
    assert not falls(engine, place(engine, {**OPEN, "a3": ("B", 1)}), 1)


def test_an_attacker_of_the_royals_own_speed_or_slower_is_a_guess_not_a_catch() -> None:
    """At equal speeds the king may still run (the chase of `docs/reference/rounds.md`), so the
    term says nothing; the rook at the king's speed, then a slower rook under a faster king."""
    assert not falls(ranked({"N": 1}), place(ranked({"N": 1}), OPEN), 1)
    engine = ranked({"K": 1})
    assert not falls(engine, place(engine, OPEN), 1)


def test_a_piece_faster_than_the_attacker_that_takes_it_answers() -> None:
    engine = ranked({"R": 1, "N": 2})
    # Player 1's knight (speed 2) on b3 takes the rook on a1 before the rook moves.
    assert not falls(engine, place(engine, {**OPEN, "b3": ("N", 1)}), 1)
    # A knight only as fast as the rook does not: the rook's strike is over first or at once.
    level = ranked({"R": 1, "N": 1})
    assert falls(level, place(level, {**OPEN, "b3": ("N", 1)}), 1)
    # Two faster attackers are two threats, and one move answers one: player 0's knight on
    # c4 also reaches a5, and player 1's knight on b3 can take only the rook.
    two = place(engine, {**OPEN, "b3": ("N", 1), "c4": ("N", 0)})
    assert falls(engine, two, 1)


def test_a_side_with_a_second_royal_has_one_to_spare() -> None:
    spec = speed_gardner({"R": 1, "N": 2})
    for piece in spec["pieces"]:
        if piece["id"] == "N":
            piece["royal"] = True
    engine = Engine(spec)
    assert not falls(engine, place(engine, {**OPEN, "c5": ("N", 1)}), 1)
    assert falls(engine, place(engine, OPEN), 1)


def test_the_price_is_the_rested_royals_and_counted_once() -> None:
    engine = ranked({"R": 1, "N": 2})
    twin = Engine(speed_gardner(None))
    state = place(engine, OPEN)
    both = lists(engine, state)
    counts = (len(both[0]), len(both[1]))
    priced = evaluation.Evaluator.for_engine(engine).evaluate_round(engine, state, 0, counts, both)
    blind = evaluation.Evaluator.for_engine(twin).evaluate_round(twin, state, 0, counts, both)
    assert priced == blind + evaluation.Evaluator.for_engine(engine).rested_royal
    # The probe's ruler passes no lists: the engine is asked for them, the same number.
    assert evaluation.Evaluator.for_engine(engine).evaluate_round(engine, state, 0, counts) == priced


def test_a_royal_both_rested_and_caught_is_one_near_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a rest (schema 29's `"rest": 1` on every piece) and speeds at once, player 1's
    king resting on a5 under the faster rook falls by both tests, and is priced once: the
    number either test alone gives."""
    spec = speed_gardner({"R": 1, "N": 2})
    spec["schema_version"] = 29
    for piece in spec["pieces"]:
        piece["rest"] = 1
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    rested = place(engine, OPEN).with_(rest=((), ((engine._square_index("a5"), 1),)))
    both = lists(engine, rested)
    sim = engine.spec.simultaneous
    assert sim is not None
    assert evaluation.rested_royal_falls(engine.spec, rested, 1, both)
    assert caught_royal_falls(engine.spec, rested, 1, both, sim.speed)
    counts = (len(both[0]), len(both[1]))
    evaluator = evaluation.Evaluator.for_engine(engine)
    once = evaluator.evaluate_round(engine, rested, 0, counts, both)
    # The caught test alone (no piece rests), on the same lists: the same number.
    unrested = rested.with_(rest=((), ()))
    assert evaluator.evaluate_round(engine, unrested, 0, counts, both) == once
    # The rested test alone: the same number again.
    monkeypatch.setattr(evaluation, "caught_royal_falls", lambda *_: False)
    assert evaluator.evaluate_round(engine, rested, 0, counts, both) == once


# --------------------------------------------- the core and the oracle, held together


def _assert_term_rounds_agree(name: str, agents: tuple[str, ...], rounds: int, seed: int) -> None:
    engine = Engine(load(name))
    states = walk(engine, rounds, seed=seed)
    for index, state in enumerate(states):
        for agent in agents:
            for seat in (0, 1):
                core = core_round(engine, agent, state, seat, seed=index)
                assert core == oracle_round(engine, agent, state, seat, seed=index), (
                    name, agent, index, seat,
                )


def test_the_core_and_the_oracle_agree_on_a_speed_spec() -> None:
    _assert_term_rounds_agree("gardner", ("greedy", "matrix:nodes=250"), rounds=3, seed=11)


@pytest.mark.slow
@pytest.mark.parametrize("name", SPEED_SPECS)
def test_the_core_and_the_oracle_agree_on_every_speed_spec(name: str) -> None:
    _assert_term_rounds_agree(name, ("greedy", "matrix:nodes=600"), rounds=8, seed=11)


# Player 1's bishop on a4 shields its king (speed 0) on a5 from player 0's rook (speed 1) on
# a1, and may take player 0's knight on d1. Every bishop move opens the a-file, where the
# rook's strike next round is over before the king can move, and the king's own pawns on b5
# and b4 keep it on the file. Before Q943b greedy took the knight; the pawn's push keeps the
# shield.
#
# Q986 moved the knight from b5 (where the king could take it) to d1 and walled the king in:
# once a finished round was worth a bounded number of units and not 10⁶ (Q986b: 100 mean
# pieces), greedy heard the evaluator and took the knight with the king, term or no term, so
# the old position no longer showed the term's doing.
SHIELD = {
    "e1": ("K", 0), "a1": ("R", 0), "d1": ("N", 0),
    "a5": ("K", 1), "a4": ("B", 1), "b5": ("P", 1), "b4": ("P", 1),
}


def _shield_weights(engine: Engine, state: GameState, strategy: list[float]) -> float:
    origin = engine._square_index("a4")
    return sum(
        weight
        for move, weight in zip(engine.legal_moves_for(state, 1), strategy, strict=True)
        if decode_move(move)[0] == origin
    )


def test_greedy_keeps_the_shield_a_faster_attacker_would_strike_through() -> None:
    engine = Engine(load("collide"))
    state = place(engine, SHIELD)
    for seed in range(4):
        core = core_round(engine, "greedy", state, 1, seed=seed)
        assert core == oracle_round(engine, "greedy", state, 1, seed=seed)
        move, strategy, _, _ = core
        assert move is not None
        start = _square_name(decode_move(move)[0], engine.spec.width)
        assert start != "a4", engine.describe_move(state, move)
        # Q986b: half a round win is 232.5 units here, beside a rook's strike the column
        # player may make anyway, so the term moves nearly all the weight off the bishop, not
        # all of it (under 10⁶ it was all; the control below is over 0.9).
        assert _shield_weights(engine, state, strategy) < 0.1, strategy


def test_without_the_term_the_same_player_would_open_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the oracle with the test switched off moves the bishop nearly always, so
    the test above is the term's doing and not the position's."""
    engine = Engine(load("collide"))
    state = place(engine, SHIELD)
    monkeypatch.setattr(evaluation, "caught_royal_falls", lambda *_: False)
    _, strategy, _, _ = oracle_round(engine, "greedy", state, 1, seed=0)
    assert _shield_weights(engine, state, strategy) > 0.9, strategy


# ------------------------------------------- inert on every game without a speed

# The round players' short tournaments (`tests/inert_players.py::ROUND_TOURNAMENTS`, one game
# a pairing), taken on main's core 9c09c4929d2fdc4a before Q943b and on this one, and equal:
# the term reads an empty speed table on each of these games and runs nothing.
#
# Q983a (2026-10-20) re-pinned the three rewritten for schema 29 (`docs/plan-rest.md` §4.5):
# breather and the scored test spec are new hashes whose records, with the hash and name left
# out, are the old files' on main's core (4271f9ff) record for record; the spy game is the
# nearest new game (its alternating phase rests). They were 55fa6c07..., eacc125d...,
# fe92fc7b... .
#
# Q986b (2026-10-20) re-pinned three on purpose: a finished round is worth `ROUND_WIN_PIECES`
# (100) mean pieces in the round search, not 10^6, a new player under every round name
# (`docs/reference/round-hardest-weakness.md` §10). The watering hole's is Q986's pin (the
# same player there: every piece is worth one). The spy game's one-game tournament never
# reaches a round search's ending and is unmoved. They were a98e81e0..., 1bac2ed5...,
# b73d89a9... .
_INERT = {
    "archive/breather.json": "23a6c532fd0bb32f7512deed4207e43f6ceca0dea62df516b2e0a9184d3e79f5",
    "archive/watering-hole.json":
        "86e042840f5cc9ac30322be448fb93f4dfde601fa8ba638c2cc0f5f355356e75",
    "archive/gardner-with-spies-resting.json":
        "0d8f72477bc1214062de25855bdfc8a6b6998338f27467c7183cb49fe7fb1447",
    "tests/_simultaneous_scored.json":
        "1c848d867d424531619d50ac937f56f0e5851ab46fbca6345e8fe00a0c7a04d3",
}


@pytest.mark.parametrize("path", sorted(_INERT))
def test_the_round_players_play_a_game_without_a_speed_as_before(path: str) -> None:
    from tests.inert_players import digest

    assert digest(path, games=1) == _INERT[path]


# --------------------------------------------------------------- the tallies

EVENTS = ("collision", "void", "initiative", "barred", "swap", "order", "lost", "unplayed")


def _tally(engine: Engine, state: GameState, first: tuple[str, str],
           second: tuple[str, str]) -> dict[str, int]:
    def word(player: int, origin: str, target: str) -> int:
        a, b = engine._square_index(origin), engine._square_index(target)
        for m in engine.legal_moves_for(state, player):
            if decode_move(m)[:2] == (a, b):
                return m
        raise AssertionError((player, origin, target))

    pair = (word(0, *first), word(1, *second))
    _, _, events = engine.apply_round_with_phase(state, *pair)
    counts = dict.fromkeys(ROUND_COUNTS + SPEED_COUNTS, 0)
    _count_round(counts, pair, events)
    return {key: value for key, value in counts.items() if value}


def test_each_speed_tally_is_read_off_its_event() -> None:
    engine = ranked({"R": 1, "N": 2})
    # §2.5 A: the rook strikes first and the king's flight is never played.
    assert _tally(engine, place(engine, OPEN), ("a1", "a5"), ("a5", "b4")) == {
        "rounds": 1, "ordered_rounds": 1, "ended_after_faster": 1,
    }
    # §2.5 B between unequal speeds: the faster takes the slower where it stands, no swap.
    pieces = {"e1": ("K", 0), "b5": ("R", 0), "b4": ("Q", 1), "a5": ("K", 1)}
    assert _tally(engine, place(engine, pieces), ("b5", "b4"), ("b4", "b5")) == {
        "rounds": 1, "ordered_rounds": 1, "slower_lost": 1,
    }
    # §2.5 C, the push: the knight lands first, and the pawn's push onto it is void.
    pieces = {"a1": ("K", 0), "b1": ("N", 0), "c4": ("P", 1), "e5": ("K", 1)}
    assert _tally(engine, place(engine, pieces), ("b1", "c3"), ("c4", "c3")) == {
        "rounds": 1, "ordered_rounds": 1, "void_moves": 1, "slower_void": 1,
    }
    # An equal-speed round counts none of them.
    assert _tally(engine, place(engine, pieces), ("a1", "a2"), ("c4", "c3")) == {"rounds": 1}


def _played(spec: dict[str, Any], agents: tuple[str, ...]) -> tuple[Any, ...]:
    result = run_tournament(
        spec, TournamentConfig(agents=agents, games_per_pairing=1, seed=7, workers=1),
    )
    return tuple(result.records)


def test_a_speed_games_analysis_holds_the_four_tallies() -> None:
    spec = load("collide")
    spec["rules"]["max_plies"] = 40
    records = _played(spec, ("greedy", "matrix:nodes=200"))
    analysed = analyse_corpus(spec, list(records), blunder_sample=0, seed=0, workers=1)
    total = dict.fromkeys(SPEED_COUNTS, 0)
    for game in analysed.games:
        assert game.rounds is not None
        assert set(game.rounds) == set(ROUND_COUNTS) | set(SPEED_COUNTS)
        assert game.rounds["ordered_rounds"] <= game.rounds["rounds"]
        assert game.rounds["slower_void"] <= game.rounds["void_moves"]
        assert game.rounds["ended_after_faster"] <= 1
        for key in SPEED_COUNTS:
            total[key] += game.rounds[key]
    assert total["ordered_rounds"] > 0, total


# A speedless round game and an alternating one, analysed: the files' digests, taken with
# main's `tournament/analysis.py` before Q943b and with this one, and equal. Q986b re-pinned
# the round game's on purpose: its records are a new round player's (`ROUND_WIN_PIECES`); the
# analysis pass is unchanged. It was 099aacbe... .
ANALYSED = {
    "tests/_simultaneous_gardner.json": ("greedy", "matrix:nodes=200"),
    "rulesets/gardner.json": ("greedy", "minimax:nodes=200"),
}
_ANALYSIS_DIGESTS = {
    "tests/_simultaneous_gardner.json":
        "9bccaa3093a2b7791fe8146700ab8e4ba79acbf582a63dc279c90d46d1f5c598",
    "rulesets/gardner.json": "4b5c1c106b4487cff31c41e4ee8f912e33ba68f1cde785e9bc93e7aefdb07603",
}


def analysis_digest(name: str, analyse: Callable[..., Any] = analyse_corpus) -> str:
    spec = json.loads((ROOT / name).read_text(encoding="utf-8"))
    spec["rules"]["max_plies"] = 40
    records = _played(spec, ANALYSED[name])
    analysed = analyse(spec, list(records), blunder_sample=2, seed=0, workers=1)
    text = json.dumps(analysed.as_dict(), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(ANALYSED))
def test_a_game_without_a_speed_is_analysed_byte_for_byte_as_before(name: str) -> None:
    assert analysis_digest(name) == _ANALYSIS_DIGESTS[name]
