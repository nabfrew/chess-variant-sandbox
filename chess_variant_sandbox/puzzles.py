"""Proven positions: small endgames from a game's recorded runs that the core has proved.

`docs/plan-website.md`, *Proven positions* (R9a); the campaign and its numbers are
`docs/reference/proven-positions.md`, driven by the script beside it.

A position is carried as the move prefix that reaches it, so there is no new position
format: the site replays the prefix through the engine, as it replays any match. The record
calls it by where it was found (`id`, `seed-game-ply`, a game of a library run and a ply);
the site names it by what it is (`name`, Q993c), a digest of the replayed state, so a
re-proof from other runs can neither move a person's mark onto another position nor lose it. Its verdict is the core's
analysis probe (`Engine.root_scores`, a full-window search under a node budget), never
Python's. This module holds positions, asks the core, and reads the numbers back: a root
move whose score is at least `PROVEN` was searched to a forced win, and one at most
`-PROVEN` to a forced loss, because the probe is full width and scores an ending as
`WIN_SCORE - ply` while no evaluation comes near that. Anything between is *open*: the
budget ran out first, and the move is neither won nor lost as far as anyone knows.

The record, `runs/puzzles/<hash>.json` (`FORMAT` 4), is what the site loads (R9b). Each
position carries its prefix, the side to move, its result (`won` or `lost` for the side to
move), the plies to the ending under best play as far as the proof saw it, and every root
move's score and proof, so *find the move* is a lookup.

**Too easy** (`docs/plan-proven.md` §3.1, pre-registered before anything was counted). A
position is trivial under T1 (a win in one with two or more winning moves), T2 (half the
legal moves or more proved won), both read off the record's own proofs (`trivial`), or T3
(the defence has no say: after the shortest winning move, every reply leaves a win exactly
two plies shorter), which asks the core again (`defence`). T4 (`plan-proven.md` §3.6,
adopted after the look that motivated it): the defence can only hasten it, every reply
leaves a win at most two plies shorter, asked the same way (`hasten`). T5 (`plan-proven.md`
§3.7, pre-registered): seen at a glance, the core's two quickest players, asked from eight
fixed seeds, play a move that wins (`shallow`). `FORMAT` 2 adds each asked position's T3
answer and the basis it was asked on, `FORMAT` 3 T4's beside it, `FORMAT` 4 T5's; `FORMAT`
1 records carry none of them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, WIN_SCORE
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import choose_move, eval_values
from chess_variant_sandbox.tournament.record import GameRecord

FORMAT = 4
FORMATS = (1, 2, 3, 4)
"""Every record format a reader accepts: 1 has no T3 answers, 2 has them, 3 adds T4's, 4
adds T5's."""
NODES = 20_000_000
# T3's budget a reply (`plan-proven.md` §3.1): a tenth of the proof's, since each of 20 to
# 30 replies is asked and the position after a reply is two plies nearer its end.
DEFENCE_NODES = 2_000_000
MAX_DEPTH = 400
MAX_PIECES = 6
# Positions proved per game, at most. E2 found 241 on *Intransitive*'s predecessor and the
# plan's cost (3-4 core-hours a game at 20 M nodes) is that many; today's runs hold
# thousands per game (`docs/reference/proven-positions.md` §1), so they are sampled.
CAP = 240
# How far before a won game's recorded end a position is asked. A 20 M-node full-width
# probe proves about eight plies ahead on *Intransitive*, and nothing proved further back
# in the first, evenly spread sample (`proven-positions.md` §2), so twelve leaves room.
NEAR = 12
# A score this close to a win is the probe having reached an ending, not an evaluation:
# no evaluator reads within a thousand of `WIN_SCORE`, and no game's clock runs that long.
PROVEN = WIN_SCORE - 1000

WON, LOST, OPEN = "won", "lost", "open"

# T3's answers, a position's `defence` in a `FORMAT` 2 or 3 record (absent where T3 does not
# apply): no reply changed the shortest win; some reply was proved to change it; or none
# was proved to, but at least one the probe could not settle, which counts as a say.
NONE, SOME, UNSETTLED = "none", "some", "unsettled"
DEFENCES = (NONE, SOME, UNSETTLED)
# T4's answers, a position's `hasten` in a `FORMAT` 3 record, are the same three words
# read against *at most* n - 2: `none` is that no reply did anything but keep pace or lose
# faster; `some`, that a reply was proved to put the win off or escape it.
T1, T2, T3, T4, T5 = "T1", "T2", "T3", "T4", "T5"

# T5 (`plan-proven.md` §3.7): the core's two quickest players a person meets, at their own
# budgets on the built-in objective (*greedy* and *easy* on the site), each asked from these
# fixed seeds. A position's `shallow` is `found` where any choice wins and `missed` where none
# does; there is no third answer, since a choice the core cannot prove won counts as a miss.
SHALLOW_AGENTS = ("greedy", "minimax:nodes=400")
SHALLOW_SEEDS = tuple(range(8))
FOUND, MISSED = "found", "missed"
SHALLOWS = (FOUND, MISSED)
# The record fields a later test adds to a row; what was asked before it never reads them.
ANSWERS = ("defence", "hasten", "shallow", "shallow_moves")


@dataclass(frozen=True)
class Candidate:
    """A position reached in a recorded game: run seed, game number, ply and its prefix."""

    seed: int
    game: int
    ply: int
    pieces: int
    moves: tuple[int, ...]

    @property
    def id(self) -> str:
        return f"{self.seed}-{self.game}-{self.ply}"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "seed": self.seed, "game": self.game, "ply": self.ply,
                "pieces": self.pieces, "moves": list(self.moves)}


def pieces_in_play(state: GameState) -> int:
    """Every piece on the board and in either hand: what a small endgame is small in.

    A hand counts because a piece in hand is a piece in play; a drop game with six on the
    board and four in hand is not a six-piece ending.
    """
    return sum(1 for cell in state.board if cell) + sum(sum(hand) for hand in state.hands)


# ------------------------------------------------------------------ a position's name

NAME_LENGTH = 16
"""Hex characters of a position's name: 64 bits, against at most `CAP` positions a game."""


def content(state: GameState) -> tuple[Any, ...]:
    """What a position *is*, as the engine returned it (Q993c, `docs/plan-proven.md` §5.3).

    The core's position key (`state.key`, the Zobrist key the core maintains and seeds from
    the rules' hash) covers the board, the side to move, the en passant file and every part
    of the state a repetition can return to. It leaves out, by design, what only an
    irreversible move changes, since repetition never compares across it: the hands, the
    check counts, a permanent colouring and the points. Those are here beside it, since two
    positions apart in any of them can end differently.

    Left out, as the key leaves them out: the counters (`ply`, `halfmove`, `turns_taken`,
    the repetition window), which say how the position was reached, and the unmoved mask,
    which says which pieces ever moved: a royal that stepped out and back makes it differ
    on one board, and two prefixes reaching one position must name it once. Python
    computes none of it; every field is the engine's.
    """
    return (
        state.key,
        tuple(tuple(hand) for hand in state.hands),
        tuple(state.checks),
        state.colour.hex(),
        tuple(float(points) for points in state.points),
    )


def name(state: GameState) -> str:
    """A position's name: a digest of its `content`, the same wherever and however it is
    reached, in any run, under any build that keeps the core's key. The site keys a
    person's marks and tries by it (`proven_marks.position`, `matches.puzzle`), so a
    re-proof that finds the same position again finds their marks with it, and one that
    finds another position can never inherit them."""
    held = content(state)
    text = json.dumps(
        [held[0], [list(hand) for hand in held[1]], list(held[2]), held[3], list(held[4])],
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:NAME_LENGTH]


def reached(engine: Engine, moves: Iterable[Any]) -> GameState | None:
    """The position a prefix reaches, every move put to the engine's `legal_moves` before
    it is applied, or None where one is not legal (or not a move at all)."""
    state = engine.initial_state()
    try:
        prefix = [int(move) for move in moves]
    except (TypeError, ValueError):
        return None
    for move in prefix:
        if move not in engine.legal_moves(state):
            return None
        state = engine.apply(state, move)
    return state


def renames(engine: Engine, data: dict[str, Any]) -> list[tuple[str, str]]:
    """`(old, new)` for every position of a record: its `seed-game-ply` id and the name
    its replayed prefix reaches (`name`), in the record's order. A prefix that no longer
    replays has no new name and is left out. What `0040_proven_names.sql` was written from,
    and what its test reads it against."""
    found = []
    for row in data.get("positions") or []:
        state = reached(engine, row.get("moves") or [])
        if state is not None:
            found.append((str(row["id"]), name(state)))
    return found


def candidates(
    engine: Engine,
    runs: Iterable[tuple[int, Iterable[GameRecord]]],
    max_pieces: int = MAX_PIECES,
    near: int | None = None,
) -> Iterator[Candidate]:
    """Every distinct non-terminal position with at most `max_pieces` in play.

    `runs` is `(seed, records)` in a fixed order; the first time a position is reached
    is divided. Distinct is by the position's `content`, its key with what the key
    leaves out because repetition never compares across it (the hands, the check counts,
    a permanent colouring, the points), which is what the site names it by (`name`).

    With `near`, only the positions a proof can reach: in a game the record says was won,
    with its winner to move, at most `near` plies before the recorded end. The record's
    result is read, not judged; whether the position is won is still the probe's answer.
    """
    if engine.simultaneous or engine.spy:
        raise ValueError("a game of rounds has no position to prove move by move")
    seen: set[tuple[Any, ...]] = set()
    for seed, records in runs:
        for number, record in enumerate(records):
            if near is not None and record.winner is None:
                continue
            length = len(record.moves)
            state = engine.initial_state()
            for ply, move in enumerate([*record.moves, None]):
                if near is None or (
                    length - ply <= near and state.side_to_move == record.winner
                ):
                    pieces = pieces_in_play(state)
                    same = content(state)
                    if (pieces <= max_pieces and same not in seen
                            and engine.terminal(state) is None):
                        seen.add(same)
                        yield Candidate(seed, number, ply, pieces, tuple(record.moves[:ply]))
                if move is None:
                    break
                state = engine.apply(state, move)


def sample(found: list[Candidate], cap: int = CAP) -> list[Candidate]:
    """At most `cap` of the candidates, whole games at a time, evenly by their order.

    Every `k`-th game that has any, for the smallest `k` that keeps the total within
    `cap`: the positions spread over many games, each game keeps every depth it had, and
    the choice is fixed by the list alone.
    """
    games: dict[tuple[int, int], list[Candidate]] = {}
    for candidate in found:
        games.setdefault((candidate.seed, candidate.game), []).append(candidate)
    groups = list(games.values())
    stride = 1
    while sum(len(group) for group in groups[::stride]) > cap:
        stride += 1
    return [candidate for group in groups[::stride] for candidate in group][:cap]


def proof_of(score: float) -> str:
    """What one root score proves for the side to move."""
    if score >= PROVEN:
        return WON
    if score <= -PROVEN:
        return LOST
    return OPEN


def ends_in(score: float, ply: int) -> int:
    """Plies from the position to the ending the proof reached, for a proven score."""
    return int(round(WIN_SCORE - abs(score))) - ply


def prove(
    engine: Engine, candidate: Candidate, nodes: int = NODES, max_depth: int = MAX_DEPTH
) -> dict[str, Any]:
    """The candidate's prefix replayed, and `verdict` of the position it reaches."""
    state = engine.play(engine.initial_state(), candidate.moves)
    return {**candidate.as_dict(), **verdict(engine, state, nodes, max_depth)}


def verdict(
    engine: Engine, state: GameState, nodes: int = NODES, max_depth: int = MAX_DEPTH
) -> dict[str, Any]:
    """Ask the core for every root move's score in this position and read them back.

    A position with one legal move is not asked: the probe does not score a forced move,
    and a position whose only move is forced is no question to set. It is returned as
    `forced` and counted, never offered.
    """
    legal = engine.legal_moves(state)
    row: dict[str, Any] = {"side": state.side_to_move}
    if len(legal) < 2:
        row["result"] = "forced"
        return row
    scores = engine.root_scores(
        state, eval_values(engine.spec), nodes, max_depth, GOAL_PIECES, IMMUNE_PIECES
    )
    moves = []
    for move, score in zip(legal, scores, strict=True):
        proof = proof_of(score)
        entry: dict[str, Any] = {
            "move": move, "text": engine.describe_move(state, move),
            "score": score, "proof": proof,
        }
        if proof != OPEN:
            entry["ends_in"] = ends_in(score, state.ply)
        moves.append(entry)
    best = max(scores)
    winning = sum(1 for entry in moves if entry["proof"] == WON)
    if proof_of(best) == WON:
        row["result"] = WON
    elif all(entry["proof"] == LOST for entry in moves):
        row["result"] = LOST
    else:
        row["result"] = OPEN
    if row["result"] != OPEN:
        row["ends_in"] = ends_in(best, state.ply)
    row["winning"] = winning
    row["single"] = row["result"] == WON and winning == 1
    row["others_lost"] = row["single"] and all(
        entry["proof"] == LOST for entry in moves if entry["proof"] != WON
    )
    row["scores"] = moves
    return row


def counts(
    rows: Iterable[dict[str, Any]], defence: bool = False, hasten: bool = False,
    shallow: bool = False,
) -> dict[str, int]:
    """The record's tally: what was asked, what was proved, how many set one move, and,
    where T3 was asked (`defence`), its answers and how many positions are too easy to
    offer under T1, T2, T3 and, where T4 was asked (`hasten`) or T5 (`shallow`), their
    answers too."""
    tally = {"candidates": 0, "forced": 0, "won": 0, "lost": 0, "open": 0,
             "single": 0, "others_lost": 0}
    defences = dict.fromkeys(DEFENCES, 0)
    hastens = dict.fromkeys(DEFENCES, 0)
    shallows = dict.fromkeys(SHALLOWS, 0)
    easy = 0
    for row in rows:
        tally["candidates"] += 1
        tally[row["result"]] += 1
        tally["single"] += bool(row.get("single"))
        tally["others_lost"] += bool(row.get("others_lost"))
        if row.get("defence") in defences:
            defences[row["defence"]] += 1
        if row.get("hasten") in hastens:
            hastens[row["hasten"]] += 1
        if row.get("shallow") in shallows:
            shallows[row["shallow"]] += 1
        easy += bool(trivial(row))
    if defence:
        tally.update({f"defence_{answer}": count for answer, count in defences.items()})
    if hasten:
        tally.update({f"hasten_{answer}": count for answer, count in hastens.items()})
    if shallow:
        tally.update({f"shallow_{answer}": count for answer, count in shallows.items()})
    if defence or hasten or shallow:
        tally["trivial"] = easy
    return tally


# ------------------------------------------------------------------- too easy (§3.1)


def winning_moves(row: dict[str, Any]) -> int:
    """W: the root moves the record proves won. An *open* move is not won."""
    return sum(1 for entry in row.get("scores") or [] if entry.get("proof") == WON)


def trivial(row: dict[str, Any]) -> tuple[str, ...]:
    """The parts of the pre-registered test (`plan-proven.md` §3.1) a won position fails.

    Empty where the position is kept. Read off the record alone, deciding nothing: `n` is
    the record's `ends_in`, `L` its number of root moves, `W` those proved won.

    - **T1**, over at once with a choice: n = 1 and W >= 2;
    - **T2**, a wasted move still wins: W >= L / 2;
    - **T3**, the defence has no say: the record's stored `defence` answer is `none`
      (`defence` asks the core for it; a `FORMAT` 1 record has none to read);
    - **T4**, the defence can only hasten it (§3.6): the stored `hasten` answer is `none`
      (`hasten` asks the core; a `FORMAT` 3 or 4 record has it);
    - **T5**, seen at a glance (§3.7): the stored `shallow` answer is `found` (`shallow`
      asks the core; only a `FORMAT` 4 record has it).

    A position that is not a proved win is not offered at all, so it fails nothing here.
    """
    if row.get("result") != WON:
        return ()
    legal = len(row.get("scores") or [])
    won = winning_moves(row)
    parts = []
    if int(row.get("ends_in") or 0) == 1 and won >= 2:
        parts.append(T1)
    if legal and 2 * won >= legal:
        parts.append(T2)
    if row.get("defence") == NONE:
        parts.append(T3)
    if row.get("hasten") == NONE:
        parts.append(T4)
    if row.get("shallow") == FOUND:
        parts.append(T5)
    return tuple(parts)


def key_move(row: dict[str, Any]) -> dict[str, Any] | None:
    """The attacker's shortest winning move: among the root moves proved won, those with
    the least `ends_in`, and of those the first in the engine's order (the record's)."""
    won = [entry for entry in row.get("scores") or [] if entry.get("proof") == WON]
    if not won:
        return None
    shortest = min(int(entry["ends_in"]) for entry in won)
    return next(entry for entry in won if int(entry["ends_in"]) == shortest)


def asks_defence(row: dict[str, Any]) -> bool:
    """Whether T3 and T4 are put to this position: a proved win with two or more winning
    moves that T1 and T2 keep (`plan-proven.md` §3.3: those already left out are not
    asked). T4 is asked of the same positions as T3 (§3.6), whatever T3 answered."""
    return row.get("result") == WON and winning_moves(row) >= 2 and not trivial(_unasked(row))


def _unasked(row: dict[str, Any]) -> dict[str, Any]:
    """The row as the proofs left it, before any engine-asked test's answer."""
    return {key: value for key, value in row.items() if key not in ANSWERS}


def asks_shallow(row: dict[str, Any]) -> bool:
    """Whether T5 is put to this position (`plan-proven.md` §3.7): a proved win that T1 to
    T4 keep, read with T3's and T4's stored answers and without T5's own."""
    return row.get("result") == WON and not trivial(
        {key: value for key, value in row.items() if key not in ("shallow", "shallow_moves")}
    )


def defence(
    engine: Engine,
    row: dict[str, Any],
    nodes: int = DEFENCE_NODES,
    max_depth: int = MAX_DEPTH,
    goal: float = GOAL_PIECES,
    immune: float = IMMUNE_PIECES,
    state: GameState | None = None,
) -> str | None:
    """T3 (`plan-proven.md` §3.1): does anything the defender does change the count?

    None where T3 does not apply (not a proved win, or fewer than two winning moves).
    Otherwise the position is reached (see `state`), the attacker's shortest winning move
    (`key_move`) is applied, and for **every** legal reply the core is asked
    (`Engine.root_scores` at `nodes`, the analysis evaluator and the given weights) about
    the position it leaves. A reply has no say only when the core proves that position won
    for the attacker with a shortest win of exactly n - 2 plies (or the game has ended
    there, in the attacker's win, with n = 2). Answers: `none` (no reply had a say),
    `some` (a reply was proved to change it: a longer or shorter win, or no win), or
    `unsettled` (none was proved to, but the probe could not settle at least one, which
    counts as a say). The first reply proved to have a say decides the answer, so the
    rest are not asked; the answer is the same either way.

    `state` is the position itself where the caller holds it; otherwise the record's
    prefix is replayed. Rule 3: every legal move, outcome and score is the core's; this
    compares the numbers.
    """
    return _says(engine, row, nodes, max_depth, goal, immune, state, faster=False)


def hasten(
    engine: Engine,
    row: dict[str, Any],
    nodes: int = DEFENCE_NODES,
    max_depth: int = MAX_DEPTH,
    goal: float = GOAL_PIECES,
    immune: float = IMMUNE_PIECES,
    state: GameState | None = None,
) -> str | None:
    """T4 (`plan-proven.md` §3.6): can the defender do anything but hasten the loss?

    T3 exactly as `defence` asks it, with one difference: a reply has no say when the core
    proves the position it leaves won for the attacker with a shortest win of **at most**
    n - 2 plies, not exactly n - 2 (or the game has ended there in the attacker's win). So
    a reply that walks into a quicker loss has no say; one proved to put the win off, to
    escape it, or to lose it is a say, and one the probe cannot settle counts as one.
    The answers are T3's three words: `none` is that every reply keeps pace or loses
    faster. Asked of the same positions as T3, and on its own pass, because T3 stops at
    the first reply that changes the count and so cannot answer this.
    """
    return _says(engine, row, nodes, max_depth, goal, immune, state, faster=True)


def _says(
    engine: Engine,
    row: dict[str, Any],
    nodes: int,
    max_depth: int,
    goal: float,
    immune: float,
    state: GameState | None,
    faster: bool,
) -> str | None:
    """T3 (`faster` false: exactly n - 2) or T4 (`faster`: at most n - 2), for `defence`
    and `hasten`. The core answers every move, outcome and score; this compares."""
    if row.get("result") != WON or winning_moves(row) < 2:
        return None
    key = key_move(row)
    assert key is not None
    want = int(row["ends_in"]) - 2
    attacker = int(row["side"])
    if state is None:
        state = engine.play(engine.initial_state(), [int(move) for move in row["moves"]])
    if state.side_to_move != attacker:
        raise ValueError(f"{row.get('id')}: the record's side is not the side to move")
    after = engine.apply(state, int(key["move"]))
    replies = engine.legal_moves(after)
    ended = engine.terminal(after, replies)
    if ended is not None:
        # The winning move ends the game (n = 1): there is no reply to make.
        return NONE if ended.winner == attacker else SOME
    if after.side_to_move == attacker:
        raise ValueError(f"{row.get('id')}: the attacker moves again; T3 and T4 ask of a reply")
    values = eval_values(engine.spec)

    def no_say(plies: int) -> bool:
        return plies <= want if faster else plies == want

    said = NONE
    for reply in replies:
        there = engine.apply(after, reply)
        legal = engine.legal_moves(there)
        outcome = engine.terminal(there, legal)
        if outcome is not None:
            answer = NONE if outcome.winner == attacker and no_say(0) else SOME
        elif there.side_to_move != attacker:
            raise ValueError(
                f"{row.get('id')}: the defender moves again; T3 and T4 ask of a reply")
        else:
            best = max(engine.root_scores(there, values, nodes, max_depth, goal, immune))
            proof = proof_of(best)
            if proof == WON:
                answer = NONE if no_say(ends_in(best, there.ply)) else SOME
            elif proof == LOST:
                answer = SOME
            else:
                answer = UNSETTLED
        if answer == SOME:
            return SOME
        if answer == UNSETTLED:
            said = UNSETTLED
    return said


def shallow(
    engine: Engine,
    row: dict[str, Any],
    agents: tuple[str, ...] = SHALLOW_AGENTS,
    seeds: tuple[int, ...] = SHALLOW_SEEDS,
    nodes: int = DEFENCE_NODES,
    max_depth: int = MAX_DEPTH,
    goal: float = GOAL_PIECES,
    immune: float = IMMUNE_PIECES,
    state: GameState | None = None,
) -> tuple[str, dict[str, list[int]]] | None:
    """T5 (`plan-proven.md` §3.7): do the core's quick players play a move that wins?

    None where the position is not a proved win. Otherwise each of `agents` is asked from
    each of `seeds` what it plays here (`choose_move`: the core's own player, its seat
    stream fixed by the seed), and each move chosen is read against the record: proved
    `won` wins, proved `lost` does not, and an `open` one is asked again of the position
    it leaves (`Engine.terminal`, else `Engine.root_scores` at `nodes` with the given
    weights), winning only where the core proves that position won for the mover. The
    answer is `found` when any choice wins and `missed` when none does, with the moves
    each agent played in seed order. The first choice proved to win decides it, so the
    rest are not asked again; the answer is the same either way.

    `state` is the position itself where the caller holds it; otherwise the record's
    prefix is replayed. Rule 3: every move, outcome and score is the core's; this compares.
    """
    if row.get("result") != WON:
        return None
    mover = int(row["side"])
    if state is None:
        state = engine.play(engine.initial_state(), [int(move) for move in row["moves"]])
    if state.side_to_move != mover:
        raise ValueError(f"{row.get('id')}: the record's side is not the side to move")
    proofs = {int(entry["move"]): entry["proof"] for entry in row.get("scores") or []}
    played: dict[str, list[int]] = {}
    for agent in agents:
        moves = []
        for seed in seeds:
            move = choose_move(engine, agent, state, seed=seed)
            if move is None or move not in proofs:
                raise ValueError(f"{row.get('id')}: {agent} played no root move of the record")
            moves.append(move)
        played[agent] = moves
    values = eval_values(engine.spec)

    def wins(move: int) -> bool:
        if proofs[move] != OPEN:
            return proofs[move] == WON
        after = engine.apply(state, move)
        legal = engine.legal_moves(after)
        outcome = engine.terminal(after, legal)
        if outcome is not None:
            return outcome.winner == mover
        best = proof_of(max(engine.root_scores(after, values, nodes, max_depth, goal, immune)))
        return best == (WON if after.side_to_move == mover else LOST)

    chosen = list(dict.fromkeys(move for moves in played.values() for move in moves))
    answer = FOUND if any(wins(move) for move in chosen) else MISSED
    return answer, played


def record(
    engine: Engine,
    name: str,
    seeds: Iterable[int],
    rows: list[dict[str, Any]],
    built: str,
    nodes: int = NODES,
    max_pieces: int = MAX_PIECES,
    made: datetime.date | None = None,
    distinct: int | None = None,
    near: int | None = NEAR,
    cap: int = CAP,
    defence_nodes: int | None = None,
    defence_built: str | None = None,
    defence_made: datetime.date | None = None,
    hasten_nodes: int | None = None,
    hasten_built: str | None = None,
    hasten_made: datetime.date | None = None,
    shallow_nodes: int | None = None,
    shallow_built: str | None = None,
    shallow_made: datetime.date | None = None,
    shallow_agents: tuple[str, ...] = SHALLOW_AGENTS,
    shallow_seeds: tuple[int, ...] = SHALLOW_SEEDS,
) -> dict[str, Any]:
    """The record the site loads: the basis, the tally, and the proved positions only.

    Open and forced positions are counted and left out, since nothing can be offered from
    them; lost ones stay, as a proof the side to move cannot escape is still a proof.
    `distinct` is how many positions the runs held before `sample` took `cap` of them.

    Rows that carry T3's answer (`defence`) keep it, and `defence_nodes` names the budget
    it was asked at, with the core and the day (`defence_built`, `defence_made`): the
    basis of every `none` the site leaves out; T4's answers (`hasten`) likewise with
    `hasten_nodes`, `hasten_built` and `hasten_made`; T5's (`shallow`, `shallow_moves`) with
    `shallow_nodes` (the second ask's budget), the players and seeds asked, `shallow_built`
    and `shallow_made`. Positions too easy to offer stay in
    the record; the reader leaves them out (`trivial`), so the record holds every proof.
    """
    ordered = sorted(rows, key=lambda row: (row["seed"], row["game"], row["ply"]))
    basis: dict[str, Any] = {
        "probe": "Engine.root_scores",
        "nodes": nodes,
        "max_depth": MAX_DEPTH,
        "proven_at": PROVEN,
        "goal": GOAL_PIECES,
        "immune": IMMUNE_PIECES,
        "built": built,
        "runs": [f"{engine.spec.hash}-{seed}" for seed in seeds],
        "max_pieces": max_pieces,
        "distinct": len(ordered) if distinct is None else distinct,
        "near": near,
        "cap": cap,
    }
    if defence_nodes is not None:
        basis["defence_nodes"] = defence_nodes
        basis["defence_built"] = defence_built
        basis["defence_made"] = (defence_made or datetime.date.today()).isoformat()
    if hasten_nodes is not None:
        basis["hasten_nodes"] = hasten_nodes
        basis["hasten_built"] = hasten_built
        basis["hasten_made"] = (hasten_made or datetime.date.today()).isoformat()
    if shallow_nodes is not None:
        basis["shallow_agents"] = list(shallow_agents)
        basis["shallow_seeds"] = list(shallow_seeds)
        basis["shallow_nodes"] = shallow_nodes
        basis["shallow_built"] = shallow_built
        basis["shallow_made"] = (shallow_made or datetime.date.today()).isoformat()
    return {
        "format": FORMAT,
        "ruleset_hash": engine.spec.hash,
        "name": name,
        "made": (made or datetime.date.today()).isoformat(),
        "basis": basis,
        "counts": counts(ordered, defence=defence_nodes is not None,
                         hasten=hasten_nodes is not None,
                         shallow=shallow_nodes is not None),
        "positions": [row for row in ordered if row["result"] in (WON, LOST)],
    }


def write(path: Path, data: dict[str, Any]) -> None:
    """Whole or not at all: a reader never sees half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)
