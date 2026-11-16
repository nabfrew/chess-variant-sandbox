"""Random openings drawn from a game's seed: what every opened duel and diverse self-play play.

**Why a duel is opened** (R-P1, ``docs/plan-player-weaknesses.md``, *The probe's openings*).
The searching players are deterministic up to their seat's stream, so a duel played from
the start position replays a handful of lines: T1c's 400-game probe held 24 distinct
10-ply openings, one of them 122 times. Its error treated 200 seed-pairs as independent
when the real sample was a few lines. An opened duel starts each pair from its own
position, up to ``most`` plies of the core's ``random`` agent, so the pairs sample
positions rather than repeat one line.

**Deterministic, and the same at any thread count.** How many plies a game opens with and
every opening move's stream are named by the game's schedule seed alone
(``opening_plies_for``, ``opening_seed``). Both games of a paired seed share the seed, so
they share the opening with the seats swapped. The schedule fixes every seed before a thread
starts and the records come back in schedule order. No module-level ``random``: the streams
are SHA-256 of the seed and the ply.

**Who chooses what** (CLAUDE.md rule 3): the core's ``random`` agent picks every opening
move (``play.choose_move``) and the core finishes the game from that position
(``play.finish_game``); Python holds the position between the two. A seat with no native
form (Fairy-Stockfish) is finished by ``play_game_mixed``'s loop instead (``mixed_game``):
our seat asked through ``Engine.select_move_depth``, theirs through its own process, because
``finish_game`` would hand such a game to the Python oracle.

Moved here from ``agents/training/data.py`` (self-play's openings, Q637, Q918g) and
``docs/reference/strength-check-all-games.py`` (the mixed loop) by R-P1, unchanged, so the
games every stored opened duel and dataset played are the games these functions play.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from typing import Any

DUEL_OPENING_PLIES = 8
"""The most random plies a duel opens with unless it names another number: the acceptance
duel's ``ACCEPT_OPENING_PLIES`` (Q643) and the T-series' and strength check's 8."""


def _hashed(*parts: object) -> int:
    """A 64-bit number named by ``parts``: how a game's seed becomes its opening's streams."""
    blob = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def opening_plies_for(seed: int, most: int) -> int:
    """How many random plies the game at ``seed`` opens with: uniform in ``0..most``."""
    return _hashed("opening-plies", seed) % (most + 1) if most > 0 else 0


def opening_seed(seed: int, ply: int) -> int:
    """The stream the core's ``random`` agent draws opening ply ``ply`` of game ``seed`` from."""
    return _hashed("opening-move", seed, ply)


def opened_plies(seed: int, most: int, rounds: bool = False) -> int:
    """How long the opening of the game at ``seed`` is, in the record's moves: a round game
    opens with whole rounds, two moves each (Q918g)."""
    plies = opening_plies_for(seed, most)
    return 2 * (plies // 2) if rounds else plies


def count_openings(records: Iterable[Any], most: int, rounds: bool = False) -> int:
    """The distinct openings ``records`` played: each game's first ``opened_plies`` moves.

    An unopened duel (``most`` 0) plays one opening, the start position."""
    if most <= 0:
        return 1
    return len({tuple(r.moves[: opened_plies(r.seed, most, rounds)]) for r in records})


def count_games(records: Iterable[Any]) -> int:
    """The distinct games ``records`` hold, move for move (two games of a pair whose players
    never disagreed count once)."""
    return len({tuple(r.moves) for r in records})


def _alternating_opening(engine: Any, seed: int, most: int) -> tuple[Any, list[int]]:
    from chess_variant_sandbox.tournament.play import choose_move

    state = engine.initial_state()
    opening: list[int] = []
    for ply in range(opening_plies_for(seed, most)):
        legal = engine.legal_moves(state)
        if engine.terminal(state, legal) is not None:
            break
        move = choose_move(engine, "random", state, opening_seed(seed, ply))
        if move is None:
            break
        opening.append(move)
        state = engine.apply(state, move)
    return state, opening


def _round_opening(engine: Any, seed: int, most: int) -> tuple[Any, list[int]]:
    """Q918g: a round game's random opening, ``opening_plies_for(seed, most) // 2`` rounds
    (a round is two plies, ``docs/reference/rounds.md`` §2.6), each seat's move the core's
    ``random`` agent's from its own stream (``opening_seed(seed, 2r + seat)``). The position
    and the moves, each round's two in seat order; it stops where a round is over before
    anyone chooses."""
    from chess_variant_sandbox.agents.base import native_spec
    from chess_variant_sandbox.tournament.play import eval_values

    agent = native_spec("random")
    values = eval_values(engine.spec)
    state = engine.initial_state()
    moves: list[int] = []
    for round_index in range(opening_plies_for(seed, most) // 2):
        if engine.terminal(state) is not None:
            break
        pair = [
            engine.select_round(state, seat, agent, values, opening_seed(seed, 2 * round_index + seat))[0]
            for seat in (0, 1)
        ]
        if pair[0] is None or pair[1] is None:
            break
        moves.extend((pair[0], pair[1]))
        state = engine.apply_round(state, pair[0], pair[1])
    return state, moves


def diverse_game(engine: Any, task: tuple[int, str, str], most: int) -> Any:
    """One game with a random opening: the core picks every move.

    The opening's moves are the core's ``random`` agent's (``choose_move``), each from its
    own stream named by the schedule seed and the ply; the players then finish the game
    from that position at the schedule seed (``finish_game``, the core's ``play_from``).
    Python only holds the position between the two and asks whether it is over.
    """
    from dataclasses import replace

    from chess_variant_sandbox.tournament.play import finish_game

    seed, first, second = task
    if engine.simultaneous:
        # Q918g: a round game opens with random rounds, both seats from their own streams.
        state, opening = _round_opening(engine, seed, most)
    else:
        state, opening = _alternating_opening(engine, seed, most)
    record = finish_game(engine, (first, second), state, seed)
    moves = opening + list(record.moves)
    return replace(record, moves=moves, plies=len(moves))


def mixed_game(engine: Any, task: tuple[int, str, str], most: int) -> Any:
    """One game with a seat that has no native form (Fairy-Stockfish): ``diverse_game``'s
    alternating opening, then ``play_game_mixed``'s loop from that position."""
    from chess_variant_sandbox.agents import build_agent
    from chess_variant_sandbox.agents.base import native_spec
    from chess_variant_sandbox.tournament.play import eval_values, seat_rng
    from chess_variant_sandbox.tournament.record import GameRecord

    seed, first, second = task
    agents = (first, second)
    state, moves = _alternating_opening(engine, seed, most)

    native = [native_spec(agent, engine.spec.hash) for agent in agents]
    built = [None if native[s] is not None else build_agent(agents[s], engine) for s in (0, 1)]
    rngs = [seat_rng(seed, 0), seat_rng(seed, 1)]
    values = eval_values(engine.spec)
    depth_sum = [0, 0]
    searches = [0, 0]
    started = time.perf_counter()
    try:
        while True:
            legal = engine.legal_moves(state)
            outcome = engine.terminal(state, legal)
            if outcome is not None:
                break
            mover = state.side_to_move
            player = built[mover]
            if player is None:
                chosen, depth = engine.select_move_depth(
                    state, native[mover], values, rngs[mover].getrandbits(31)
                )
                if chosen is None:  # pragma: no cover
                    raise RuntimeError("the core found no move in a position that has some")
                move = chosen
                if len(legal) > 1:
                    depth_sum[mover] += depth
                    searches[mover] += 1
            else:
                move = player.select_move(state, legal, rngs[mover])
            moves.append(move)
            state = engine.apply(state, move)
    finally:
        for player in built:
            if player is not None and hasattr(player, "close"):
                player.close()
    nodes = [0, 0]
    for seat in (0, 1):
        if built[seat] is not None:
            nodes[seat] = built[seat].stats.nodes  # type: ignore[union-attr]
    return GameRecord(
        ruleset_hash=engine.spec.hash,
        ruleset_name=engine.spec.name,
        seed=seed,
        agents=(first, second),
        moves=moves,
        winner=outcome.winner,
        reason=outcome.reason,
        plies=len(moves),
        branching=[],
        material=[],
        captures=[],
        nodes=(nodes[0], nodes[1]),
        depth_sum=(depth_sum[0], depth_sum[1]),
        searches=(searches[0], searches[1]),
        duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
    )


def play_opened(spec: dict[str, Any], config: Any, tasks: Sequence[Any], most: int) -> list[Any]:
    """These schedule tasks, each opened by up to ``most`` random plies, in task order.

    Inside a campaign directory (Q732) in sealed pieces a restart skips: the same games."""
    from concurrent.futures import ThreadPoolExecutor

    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament import checkpoint

    engine = Engine(spec)
    resumable = checkpoint.play_games(
        {"ruleset_hash": engine.spec.hash, "opening_plies": most},
        config.agents,
        list(tasks),
        lambda piece: [diverse_game(engine, task, most) for task in piece],
        config.resolved_workers(),
    )
    if resumable is not None:
        return resumable
    with ThreadPoolExecutor(max_workers=config.resolved_workers()) as pool:
        return list(pool.map(lambda task: diverse_game(engine, task, most), tasks))


def play_opened_mixed(
    spec: dict[str, Any], config: Any, tasks: Sequence[Any], most: int
) -> list[Any]:
    """``play_opened`` for a duel with a Fairy-Stockfish seat (``mixed_game``), in task order;
    its campaign pieces are named apart from the native ones by the finish and the engine."""
    from concurrent.futures import ThreadPoolExecutor

    from chess_variant_sandbox import premise
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament import checkpoint

    engine = Engine(spec)
    identity = {
        "ruleset_hash": engine.spec.hash,
        "opening_plies": most,
        "finish": "mixed",
        "fairy_engine": premise.fairy_engine(),
    }
    resumable = checkpoint.play_games(
        identity, config.agents, list(tasks),
        lambda piece: [mixed_game(engine, task, most) for task in piece],
        config.resolved_workers(),
    )
    if resumable is not None:
        return resumable
    with ThreadPoolExecutor(max_workers=config.resolved_workers()) as pool:
        return list(pool.map(lambda task: mixed_game(engine, task, most), tasks))


def native_seats(agents: Sequence[str], ruleset_hash: str) -> bool:
    """Whether every seat has a native form, so the core finishes the game."""
    from chess_variant_sandbox.agents.base import native_spec

    return all(native_spec(agent, ruleset_hash) is not None for agent in agents)


def play_opened_schedule(spec: dict[str, Any], config: Any, most: int) -> list[Any]:
    """A config's whole schedule, every game opened: the core's finish where every seat is
    native, the mixed loop where one is not."""
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament.runner import schedule

    tasks = schedule(config)
    if native_seats(config.agents, Engine(spec).spec.hash):
        return play_opened(spec, config, tasks, most)
    return play_opened_mixed(spec, config, tasks, most)
