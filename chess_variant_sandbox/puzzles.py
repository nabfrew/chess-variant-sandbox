"""Proven positions: small endgames from a game's recorded runs that the core has proved.

`docs/plan-website.md`, *Proven positions* (R9a); the campaign and its numbers are
`docs/reference/proven-positions.md`, driven by the script beside it.

A position is named by where it was reached, a game of a library run and a ply, and is
carried as the move prefix that reaches it, so there is no new position format: the site
replays the prefix through the engine, as it replays any match. Its verdict is the core's
analysis probe (`Engine.root_scores`, a full-window search under a node budget), never
Python's. This module holds positions, asks the core, and reads the numbers back: a root
move whose score is at least `PROVEN` was searched to a forced win, and one at most
`-PROVEN` to a forced loss, because the probe is full width and scores an ending as
`WIN_SCORE - ply` while no evaluation comes near that. Anything between is *open*: the
budget ran out first, and the move is neither won nor lost as far as anyone knows.

The record, `runs/puzzles/<hash>.json` (`FORMAT` 1), is what the site loads (R9b). Each
position carries its prefix, the side to move, its result (`won` or `lost` for the side to
move), the plies to the ending under best play as far as the proof saw it, and every root
move's score and proof, so *find the move* is a lookup.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, WIN_SCORE
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import eval_values
from chess_variant_sandbox.tournament.record import GameRecord

FORMAT = 1
NODES = 20_000_000
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


def candidates(
    engine: Engine,
    runs: Iterable[tuple[int, Iterable[GameRecord]]],
    max_pieces: int = MAX_PIECES,
    near: int | None = None,
) -> Iterator[Candidate]:
    """Every distinct non-terminal position with at most `max_pieces` in play.

    `runs` is `(seed, records)` in a fixed order; the first time a position is reached
    names it, so the answer depends on the runs and their order, never on how the work
    is divided. Distinct is by the position's key with the hands and the check counts
    beside it, which the key leaves out because repetition never compares across them.

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
                    same = (state.key, state.hands, state.checks)
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


def counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """The record's tally: what was asked, what was proved, and how many set one move."""
    tally = {"candidates": 0, "forced": 0, "won": 0, "lost": 0, "open": 0,
             "single": 0, "others_lost": 0}
    for row in rows:
        tally["candidates"] += 1
        tally[row["result"]] += 1
        tally["single"] += bool(row.get("single"))
        tally["others_lost"] += bool(row.get("others_lost"))
    return tally


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
) -> dict[str, Any]:
    """The record the site loads: the basis, the tally, and the proved positions only.

    Open and forced positions are counted and left out, since nothing can be offered from
    them; lost ones stay, as a proof the side to move cannot escape is still a proof.
    `distinct` is how many positions the runs held before `sample` took `cap` of them.
    """
    ordered = sorted(rows, key=lambda row: (row["seed"], row["game"], row["ply"]))
    return {
        "format": FORMAT,
        "ruleset_hash": engine.spec.hash,
        "name": name,
        "made": (made or datetime.date.today()).isoformat(),
        "basis": {
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
        },
        "counts": counts(ordered),
        "positions": [row for row in ordered if row["result"] in (WON, LOST)],
    }


def write(path: Path, data: dict[str, Any]) -> None:
    """Whole or not at all: a reader never sees half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)
