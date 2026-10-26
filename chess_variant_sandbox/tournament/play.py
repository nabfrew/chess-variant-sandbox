"""Playing a single game, reproducibly.

The reproducibility contract lives here: a game is a pure function of
``(ruleset, seed, agent configs)``. Each seat gets its own RNG stream derived from the
seed and the seat, so that how much randomness one agent happens to consume cannot
shift what the other draws. Without that, swapping in a differently-implemented agent
silently changes the *opponent's* choices too, and a strength comparison stops meaning
anything.

The whole loop runs in the Rust core when both agents have a native form, which is the
case for every built-in. That includes the agents' own randomness: the core reproduces
CPython's Mersenne Twister draw for draw, so moving the loop across the boundary did
not move a single recorded game. ``play_game_python`` is the same loop in Python; it
runs whenever a custom agent is in the pairing, and it is what the port is tested
against.
"""

from __future__ import annotations

import random
import time
from typing import Any

from chess_variant_sandbox.agents import build_agent, native_spec
from chess_variant_sandbox.agents.base import AgentConfig
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import CompiledSpec
from chess_variant_sandbox.engine.moves import MOVE_DUCK, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.record import GameRecord

_EVAL_VALUES: dict[str, list[float]] = {}

# What a `NativeSpec` that the core will not accept almost always means.
#
# The tuple's shape is the one thing about this boundary that both sides have to agree on
# from memory rather than by construction, so a `_core` built before a slot was added
# rejects every agent with a conversion error -- and it does it deep inside a request,
# where the message says nothing about builds. `uv sync` does not notice that `rust/`
# changed, which is exactly how a tree gets into this state.
_STALE_CORE = (
    "{error}\n\n"
    "That is almost certainly a stale build of the extension: the agent boundary grew "
    "a slot and `chess_variant_sandbox._core` was compiled before it. A plain `uv sync` does not "
    "notice that `rust/` changed. Rebuild with:\n"
    "    uv sync --extra dev --reinstall-package chess-variant-sandbox\n"
    "or, in the edit loop:\n"
    "    uv run maturin develop --release --manifest-path rust/Cargo.toml"
)


def eval_values(spec: CompiledSpec) -> list[float]:
    """The blended piece values the core's agents need, memoised per ruleset.

    Deriving them walks every piece's table on every square: about two milliseconds,
    which is the same order as a whole short game now, so paying it per game would show
    up in a sweep. The spec hash is the right key because it covers exactly what the
    values depend on -- the mechanics, excluding the name.
    """
    values = _EVAL_VALUES.get(spec.hash)
    if values is None:
        values = list(Evaluator.for_spec(spec).values)
        _EVAL_VALUES[spec.hash] = values
    return values


def seat_rng(seed: int, seat: int) -> random.Random:
    """An independent, reproducible stream per seat."""
    return random.Random((seed << 1) ^ (seat + 0x9E3779B9))


def play_game(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    seed: int,
    collect_series: bool = True,
) -> GameRecord:
    """Play one game to a verdict and return its record.

    ``collect_series`` controls whether per-ply branching and material are recorded.
    The smoke-test gate in the design loop turns it off: it only needs to know whether
    a ruleset produces finite, legal games, and the series would be pure overhead
    across the hundreds of throwaway games it plays.
    """
    native = [native_spec(agent, engine.spec.hash) for agent in agents]
    if engine.core is not None and native[0] is not None and native[1] is not None:
        result = engine.core.play_game(
            (native[0], native[1]), eval_values(engine.spec), seed, collect_series
        )
        return build_record(engine, agents, seed, result)
    if engine.core is not None and (native[0] is not None or native[1] is not None):
        if engine.simultaneous:
            # Schema 26: a round asks both seats on one board, which the mixed loop (one
            # move at a time) does not; a custom player in a round game plays the round
            # loop with both seats built here.
            return play_game_python(engine, agents, seed, collect_series=collect_series)
        return play_game_mixed(engine, agents, native, seed, collect_series=collect_series)
    return play_game_python(engine, agents, seed, collect_series=collect_series)


def play_game_mixed(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    native: list[Any],
    seed: int,
    collect_series: bool = True,
) -> GameRecord:
    """One seat in the core and one outside it.

    The loop is Python's, because one of the players is; but the *other* player is not
    built in Python. It is asked for its move through ``select_move``, which is the
    same search the core runs when it owns the whole game.

    **This is worth a function of its own rather than falling through to
    `play_game_python`.** That one builds both agents from `agents/`, and those are
    declared oracles: `agents/minimax.py` exists to be compared against `agents.rs` and
    is a hundred times slower than it. So the moment a custom agent joined the roster,
    every game it played dragged the *whole ladder* into Python with it -- the picker's
    rating run on the smallest board in the corpus took 283 seconds, and almost all of
    it was our own top rung searching 200,000 nodes an interpreter at a time. Running
    the native side natively is also the more honest comparison: the opponent whose
    rating this produces is then the player the index rates, rather than a slower
    reimplementation of it.

    Each seat keeps its own stream, exactly as everywhere else, and the native seat
    draws a fresh seed from its stream every ply. A single seed reused would make a
    random mover pick the same index in every position -- reproducible, and not random.
    """
    spec = engine.spec
    values = eval_values(spec)
    players = [
        None if native[seat] is not None else build_agent(agents[seat], engine)
        for seat in (0, 1)
    ]
    rngs = [seat_rng(seed, 0), seat_rng(seed, 1)]
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0) if collect_series else None

    state = engine.initial_state()
    moves: list[int] = []
    branching: list[int] = []
    material: list[float] = []
    captures: list[int] = []
    nodes = [0, 0]
    depth_sum = [0, 0]
    searches = [0, 0]

    started = time.perf_counter()
    while True:
        legal = engine.legal_moves(state)
        outcome = engine.terminal(state, legal)
        if outcome is not None:
            break
        if collect_series:
            branching.append(len(legal))
            assert evaluator is not None
            material.append(round(evaluator.material(state, 0), 4))

        mover = state.side_to_move
        player = players[mover]
        if player is None:
            chosen, depth = engine.select_move_depth(
                state, native[mover], values, rngs[mover].getrandbits(31)
            )
            if chosen is None:  # pragma: no cover - a move list that is not empty
                raise RuntimeError("the core found no move in a position that has some")
            move = chosen
            if len(legal) > 1:
                depth_sum[mover] += depth
                searches[mover] += 1
        else:
            move = player.select_move(state, legal, rngs[mover])
        successor = engine.apply(state, move)
        if collect_series:
            # Generation 42 (N4b): a duck move takes nothing (`play.rs`'s reading).
            captures.append(1 if successor.halfmove == 0 and move_type(move) != MOVE_DUCK else 0)
        moves.append(move)
        state = successor

    for seat in (0, 1):
        if players[seat] is not None:
            stats = players[seat].stats  # type: ignore[union-attr]
            nodes[seat], depth_sum[seat], searches[seat] = (
                stats.nodes, stats.depth_sum, stats.searches
            )

    return GameRecord(
        ruleset_hash=spec.hash,
        ruleset_name=spec.name,
        seed=seed,
        agents=(str(agents[0]), str(agents[1])),
        moves=moves,
        winner=outcome.winner,
        reason=outcome.reason,
        plies=len(moves),
        branching=branching,
        material=material,
        captures=captures,
        # The core reports no node count for a single decision, so a seat played
        # through it contributes none. Left at zero rather than filled in with the
        # budget, because a budget is what a player was *allowed* and this field is
        # what it spent -- and no game with a foreign player in it is a corpus
        # measurement, so nothing under `metrics/` ever reads these.
        nodes=(nodes[0], nodes[1]),
        depth_sum=(depth_sum[0], depth_sum[1]),
        searches=(searches[0], searches[1]),
        duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
    )


def build_record(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    seed: int,
    result: tuple,
) -> GameRecord:
    """Assemble a record from what the core reports.

    Shared by the single-game path here and the tournament runner, so a game played on
    its own and the same game played inside a schedule cannot end up recorded
    differently.

    Rounding happens here rather than in Rust: ``round`` on a float is a decimal
    operation with its own tie-breaking, and the records are compared against ones
    written before the port. ``captures`` arrives as bytes, which is what a Rust byte
    vector converts to, and a record has to stay JSON-serialisable.
    """
    (
        moves,
        winner,
        reason,
        branching,
        material,
        captures,
        nodes,
        depth_sum,
        searches,
        duration_ms,
    ) = result
    return GameRecord(
        ruleset_hash=engine.spec.hash,
        ruleset_name=engine.spec.name,
        seed=seed,
        agents=(str(agents[0]), str(agents[1])),
        moves=moves,
        winner=winner,
        reason=reason,
        plies=len(moves),
        branching=branching,
        material=[round(value, 4) for value in material],
        captures=list(captures),
        nodes=tuple(nodes),
        depth_sum=tuple(depth_sum),
        searches=tuple(searches),
        duration_ms=round(duration_ms, 2),
    )


def choose_move(
    engine: Engine,
    agent: AgentConfig | str,
    state: GameState,
    seed: int = 0,
) -> int | None:
    """What one agent would play in one position.

    The counterpart to ``play_game`` for callers that hold a position rather than a
    schedule: an interactive opponent, or a probe asking what a rated player would do
    here. It runs in the core wherever the core implements the agent, which is the whole
    point -- an opponent built from ``agents/minimax.py`` would be a *different*
    implementation from the one whose rating the index reads, so a game played against
    it and a game scored by the index would not be the same game.

    Falls back to the Python agent for a custom kind the core knows nothing about,
    exactly as ``play_game`` does. That fallback is the extension point, not a
    workaround: a human player or an LLM player is an ``Agent`` and nothing else.

    The seat stream is fresh per call. Two consequences worth stating: asking twice with
    the same seed gives the same move, and a sequence of these calls is not the game
    ``play_game`` plays from that seed, because that game's streams advance. Anything
    that needs a whole game should ask for a whole game.
    """
    legal = engine.legal_moves(state)
    if not legal:
        return None

    native = native_spec(agent, engine.spec.hash)
    if engine.core is None or native is None:
        return build_agent(agent, engine).select_move(state, legal, seat_rng(seed, 0))
    try:
        return engine.select_move(state, native, eval_values(engine.spec), seed)
    except TypeError as error:  # pragma: no cover - a build problem, not a code path
        raise TypeError(_STALE_CORE.format(error=error)) from error


def finish_game(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    state: GameState,
    seed: int,
) -> GameRecord:
    """Play a position out to a verdict with the given agents.

    The blunder trial in ``tournament/analysis.py`` is what wants this: force one
    arbitrary move into a recorded game, then ask whether the game is still a game.
    Fresh per-seat streams, matching the Python pass it replaces -- the resumed game is
    a new game with its own randomness rather than a continuation of the original's.
    """
    native = [native_spec(agent, engine.spec.hash) for agent in agents]
    if engine.core is None or native[0] is None or native[1] is None:
        return _finish_game_python(engine, agents, state, seed)

    result = engine.play_from(
        state, (native[0], native[1]), eval_values(engine.spec), seed
    )
    return build_record(engine, agents, seed, result)


def _finish_game_python(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    state: GameState,
    seed: int,
) -> GameRecord:
    """``finish_game`` in Python, for agents the core does not implement."""
    players = [build_agent(agents[0], engine), build_agent(agents[1], engine)]
    rngs = [seat_rng(seed, 0), seat_rng(seed, 1)]
    moves: list[int] = []

    started = time.perf_counter()
    while True:
        legal = engine.legal_moves(state)
        outcome = engine.terminal(state, legal)
        if outcome is not None:
            break
        mover = state.side_to_move
        move = players[mover].select_move(state, legal, rngs[mover])
        moves.append(move)
        state = engine.apply(state, move)

    return GameRecord(
        ruleset_hash=engine.spec.hash,
        ruleset_name=engine.spec.name,
        seed=seed,
        agents=(str(agents[0]), str(agents[1])),
        moves=moves,
        winner=outcome.winner,
        reason=outcome.reason,
        plies=len(moves),
        branching=[],
        material=[],
        captures=[],
        nodes=(players[0].stats.nodes, players[1].stats.nodes),
        depth_sum=(players[0].stats.depth_sum, players[1].stats.depth_sum),
        searches=(players[0].stats.searches, players[1].stats.searches),
        duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
    )


def play_game_python(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    seed: int,
    collect_series: bool = True,
) -> GameRecord:
    """The game loop in Python, for agents the core does not implement."""
    spec = engine.spec
    players = [build_agent(agents[0], engine), build_agent(agents[1], engine)]
    rngs = [seat_rng(seed, 0), seat_rng(seed, 1)]
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0) if collect_series else None
    if engine.spy:
        return _play_spy_python(engine, agents, players, rngs, seed, evaluator)
    if engine.simultaneous:
        return _play_rounds_python(engine, agents, players, rngs, seed, evaluator)

    state = engine.initial_state()
    moves: list[int] = []
    branching: list[int] = []
    material: list[float] = []
    captures: list[int] = []

    started = time.perf_counter()
    while True:
        legal = engine.legal_moves(state)
        outcome = engine.terminal(state, legal)
        if outcome is not None:
            break
        if collect_series:
            branching.append(len(legal))
            assert evaluator is not None
            material.append(round(evaluator.material(state, 0), 4))

        mover = state.side_to_move
        move = players[mover].select_move(state, legal, rngs[mover])
        successor = engine.apply(state, move)
        if collect_series:
            # Generation 42 (N4b): a duck move takes nothing (`play.rs`'s reading).
            captures.append(1 if successor.halfmove == 0 and move_type(move) != MOVE_DUCK else 0)
        moves.append(move)
        state = successor

    duration = (time.perf_counter() - started) * 1000.0
    return GameRecord(
        ruleset_hash=spec.hash,
        ruleset_name=spec.name,
        seed=seed,
        agents=(str(agents[0]), str(agents[1])),
        moves=moves,
        winner=outcome.winner,
        reason=outcome.reason,
        plies=len(moves),
        branching=branching,
        material=material,
        captures=captures,
        nodes=(players[0].stats.nodes, players[1].stats.nodes),
        depth_sum=(players[0].stats.depth_sum, players[1].stats.depth_sum),
        searches=(players[0].stats.searches, players[1].stats.searches),
        duration_ms=round(duration, 2),
    )


def _play_rounds_python(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    players: list[Any],
    rngs: list[random.Random],
    seed: int,
    evaluator: Evaluator | None,
) -> GameRecord:
    """Schema 26 (Q918b): ``play_game_python``'s loop for a game played in rounds.

    The oracle for `rust/src/play.rs`'s round loop: both seats asked on the same board,
    each with its own stream and both lists, neither seeing the other's choice; the
    record keeps both moves of a round in seat order and every per-ply series twice.
    """
    for seat, player in enumerate(players):
        # `plays_rounds` False: a player with a round arm for a spy game's blind rounds only
        # (minimax, Q935b), which a round game refuses as the core does.
        if not getattr(player, "plays_rounds", hasattr(player, "select_round_move")):
            raise ValueError(
                f"{agents[seat]} does not play a simultaneous game (rules.simultaneous); "
                "random, greedy and matrix:nodes=N do"
            )
    state = engine.initial_state()
    moves: list[int] = []
    branching: list[int] = []
    material: list[float] = []
    captures: list[int] = []

    started = time.perf_counter()
    while True:
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        outcome = engine.terminal(state)
        if outcome is not None:
            break
        if evaluator is not None:
            branching.extend((len(lists[0]), len(lists[1])))
            value = round(evaluator.material(state, 0), 4)
            material.extend((value, value))
        first = players[0].select_round_move(state, 0, lists[0], lists[1], rngs[0])
        second = players[1].select_round_move(state, 1, lists[1], lists[0], rngs[1])
        state = engine.apply_round(state, first, second)
        if evaluator is not None:
            quiet = 1 if state.halfmove == 0 else 0
            captures.extend((quiet, quiet))
        moves.extend((first, second))

    duration = (time.perf_counter() - started) * 1000.0
    return GameRecord(
        ruleset_hash=engine.spec.hash,
        ruleset_name=engine.spec.name,
        seed=seed,
        agents=(str(agents[0]), str(agents[1])),
        moves=moves,
        winner=outcome.winner,
        reason=outcome.reason,
        plies=len(moves),
        branching=branching,
        material=material,
        captures=captures,
        nodes=(players[0].stats.nodes, players[1].stats.nodes),
        depth_sum=(players[0].stats.depth_sum, players[1].stats.depth_sum),
        searches=(players[0].stats.searches, players[1].stats.searches),
        duration_ms=round(duration, 2),
    )


def _play_spy_python(
    engine: Engine,
    agents: tuple[AgentConfig | str, AgentConfig | str],
    players: list[Any],
    rngs: list[random.Random],
    seed: int,
    evaluator: Evaluator | None,
) -> GameRecord:
    """Schema 27 (Q935a, players Q935b): ``play_game_python``'s loop for a spy game.

    The oracle for `rust/src/play.rs`'s spy loop (`docs/plan-spy.md` §2.5): each step asks
    the engine its phase. An alternating or reply turn is one move by the side to move; an
    announced round asks the announcer first and shows its move to the answerer; a blind
    round asks both on the same board. The record keeps a turn's one move and a round's two
    in seat order, and every per-ply series once a turn and twice a round.
    """
    for seat, player in enumerate(players):
        if not hasattr(player, "select_answer"):
            raise ValueError(
                f"{agents[seat]} does not play a spy game (pieces[].spy); random, greedy and "
                "minimax:nodes=N do"
            )
    state = engine.initial_state()
    moves: list[int] = []
    branching: list[int] = []
    material: list[float] = []
    captures: list[int] = []

    started = time.perf_counter()
    while True:
        phase = engine.phase(state)
        if not phase.is_round:
            legal = engine.legal_moves(state)
            outcome = engine.terminal(state, legal)
            if outcome is not None:
                break
            if evaluator is not None:
                branching.append(len(legal))
                material.append(round(evaluator.material(state, 0), 4))
            mover = state.side_to_move
            move = players[mover].select_move(state, legal, rngs[mover])
            state = engine.apply(state, move)
            if evaluator is not None:
                captures.append(1 if state.halfmove == 0 else 0)
            moves.append(move)
            continue
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        outcome = engine.terminal(state)
        if outcome is not None:
            break
        if evaluator is not None:
            branching.extend((len(lists[0]), len(lists[1])))
            value = round(evaluator.material(state, 0), 4)
            material.extend((value, value))
        if phase.name == "announced":
            announcer = phase.announcer
            announced = players[announcer].select_announcement(
                state, announcer, lists, rngs[announcer]
            )
            answer = players[1 - announcer].select_answer(
                state, 1 - announcer, lists, announced, rngs[1 - announcer]
            )
            pair = (announced, answer) if announcer == 0 else (answer, announced)
        else:
            pair = (
                players[0].select_round_move(state, 0, lists[0], lists[1], rngs[0]),
                players[1].select_round_move(state, 1, lists[1], lists[0], rngs[1]),
            )
        state = engine.apply_round(state, *pair)
        if evaluator is not None:
            quiet = 1 if state.halfmove == 0 else 0
            captures.extend((quiet, quiet))
        moves.extend(pair)

    duration = (time.perf_counter() - started) * 1000.0
    return GameRecord(
        ruleset_hash=engine.spec.hash,
        ruleset_name=engine.spec.name,
        seed=seed,
        agents=(str(agents[0]), str(agents[1])),
        moves=moves,
        winner=outcome.winner,
        reason=outcome.reason,
        plies=len(moves),
        branching=branching,
        material=material,
        captures=captures,
        nodes=(players[0].stats.nodes, players[1].stats.nodes),
        depth_sum=(players[0].stats.depth_sum, players[1].stats.depth_sum),
        searches=(players[0].stats.searches, players[1].stats.searches),
        duration_ms=round(duration, 2),
    )


def replay(engine: Engine, record: GameRecord) -> list:
    """Re-derive the position after every ply. Used by the visualiser and by tests.

    A record stores moves rather than positions, so this is also the check that a
    record is self-consistent: an illegal move here means the record and the ruleset
    it names have drifted apart.
    """
    if engine.spy:
        return _replay_spy(engine, record)
    state = engine.initial_state()
    states = [state]
    for index, move in enumerate(record.moves):
        if move not in engine.legal_moves(state):
            raise ValueError(
                f"ply {index} of game {record.seed} is not legal under ruleset "
                f"{record.ruleset_hash}; the record and the spec disagree"
            )
        state = engine.apply(state, move)
        states.append(state)
    return states


def _replay_spy(engine: Engine, record: GameRecord) -> list:
    """``replay`` for a spy game (schema 27): the position after every step, a turn or a
    round by the phase the engine reports, each move checked legal on the board it was
    chosen on."""
    state = engine.initial_state()
    states = [state]
    moves = record.moves
    index = 0
    while index < len(moves):
        if engine.is_round(state):
            step = moves[index:index + 2]
            legal = [engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)]
        else:
            step = moves[index:index + 1]
            legal = [engine.legal_moves(state)]
        if len(step) != len(legal) or any(m not in ok for m, ok in zip(step, legal, strict=True)):
            raise ValueError(
                f"ply {index} of game {record.seed} is not legal under ruleset "
                f"{record.ruleset_hash}; the record and the spec disagree"
            )
        state = engine.apply_round(state, *step) if len(step) == 2 else engine.apply(state, step[0])
        states.append(state)
        index += len(step)
    return states
