"""`verify` from late positions, not only the opening (Q1219).

`verify` compares the two engines over a perft from the opening array and a few random
walks. Both stay near the opening: a perft never leaves it, and a random mover rarely
brings about what a searched game does late (two royals beside each other, a bare board,
a pawn about to promote under a blast). Q1219's difference, atomic's touching kings, was
found by a game, at ply 82 of seed 22, and nothing shallower had met it.

This asks the same question past the opening: for each library ruleset that translates,
positions are drawn from stored game records where there are some (each record replayed
through our engine, keeping the plies from `MIN_PLY` on; at most half the sample) and from
many long random games (always: records hold only the games that finished, and a game a
`MoveMismatch` stopped is the one never stored; atomic's 96 stored games never brought the
kings within two squares, and its random games do so often). At each position the two
engines' legal moves are compared (`verify._agree_here`), and from the first `PERFTS` of
them the tree to `DEPTH` is compared move list by move list (a perft *count* differs at
every position one move from a win, since ours counts past an ending and theirs stops).
Every difference is kept, not only the first, and the table says which games diverge.

It runs against `pyffish` (the library build) only: it starts no engine process and plays
no game, so it is light. The installed binary is a different build; a difference here
that the binary does not share is still a difference in the translation's premise.

    uv run python -m chess_variant_sandbox.fairy.late [--records DIR] [--names a b ...]
        [--positions N] [--json out.json]

`--records` is a directory holding `<hash>/*.jsonl` game records (R-P3's split run keeps
them under `runs/R-P3/split/games`). The last table is in
`docs/reference/fairy-stockfish.md` (Q1219).
"""

# compute-lease: light

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# A position is "late" from this ply on: past where the opening perft reaches. Ten, not
# more, because atomic's games are short (a king is blown up early) and the positions
# that found Q1219 come between ply 10 and 30 of a random game.
MIN_PLY = 10
# Positions compared per game, and from how many of them the tree to `DEPTH` is compared.
POSITIONS = 300
PERFTS = 20
DEPTH = 2
# Random games this long, from this many seeds, beside any records.
RANDOM_PLIES = 200
RANDOM_GAMES = 60


@dataclass
class Divergence:
    """One position where the two engines' moves differ."""

    fen: str
    only_ours: list[str] = field(default_factory=list)
    only_theirs: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class LateCheck:
    """What the late check found for one game."""

    name: str
    ruleset_hash: str
    source: str  # "records+random", "random" or "refused"
    positions: int = 0
    perfts: int = 0
    divergences: list[Divergence] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.source == "refused":
            return "refused"
        return "diverges" if self.divergences else "agrees"


def library_specs(root: Path) -> list[dict[str, Any]]:
    """Every ruleset in ``root/rulesets``, in name order."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "rulesets").glob("*.json"))
    ]


def _records(records: Path | None, ruleset_hash: str) -> list[list[int]]:
    """The move lists of every stored game of this ruleset under ``records``."""
    if records is None:
        return []
    folder = records / ruleset_hash
    if not folder.is_dir():
        return []
    games: list[list[int]] = []
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                games.append(list(json.loads(line)["moves"]))
    return games


def late_states(engine: Any, games: list[list[int]], limit: int, seed: int) -> list[Any]:
    """Positions from ``MIN_PLY`` on, replayed from records or from random games.

    A position our engine calls over is never kept: `verify` does not compare those.
    """
    replayed: list[Any] = []
    for moves in games:
        state = engine.initial_state()
        for ply, move in enumerate(moves):
            legal = engine.legal_moves(state)
            if not legal or engine.terminal(state, legal) is not None:
                break
            if ply >= MIN_PLY and not engine.duck_owed(state):
                replayed.append(state)
            state = engine.apply(state, move)
    # Random games as well, always: stored records hold only the games that finished, and
    # a game stopped by a `MoveMismatch` is the one game never stored. Atomic's 96 stored
    # games never put the two kings within two squares; random games do it often.
    walked: list[Any] = []
    for game in range(RANDOM_GAMES):
        rng = random.Random((seed << 8) ^ game)
        state = engine.initial_state()
        for ply in range(RANDOM_PLIES):
            legal = engine.legal_moves(state)
            if not legal or engine.terminal(state, legal) is not None:
                break
            if ply >= MIN_PLY and not engine.duck_owed(state):
                walked.append(state)
            state = engine.apply(state, rng.choice(legal))
    pick = random.Random(seed)
    half = limit // 2 if replayed else 0
    if len(replayed) > half:
        replayed = pick.sample(replayed, half)
    if len(walked) > limit - len(replayed):
        walked = pick.sample(walked, limit - len(replayed))
    return replayed + walked


def check_states(spec: dict[str, Any], states: list[Any], backend: Any, *, perfts: int,
                 depth: int) -> tuple[int, int, list[Divergence], list[str]]:
    """Compare the engines at each state; a perft at the first ``perfts`` of them."""
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.fairy import config_path
    from chess_variant_sandbox.fairy.variant import translate
    from chess_variant_sandbox.fairy.verify import Verification, _agree_here, _fen

    variant = translate(spec)
    engine = Engine(spec)
    backend.load(variant, config_path(variant))
    divergences: list[Divergence] = []
    seen: set[str] = set()
    compared = 0
    counted = 0

    def agree(state: Any) -> bool:
        nonlocal compared
        text = _fen(engine, variant, state)
        if text in seen:
            return True
        seen.add(text)
        probe = Verification(ruleset_hash=engine.spec.hash, name=engine.spec.name, ok=True)
        compared += 1
        if _agree_here(engine, variant, backend, state, probe):
            return True
        divergences.append(
            Divergence(
                fen=probe.fen or text,
                only_ours=sorted(set(probe.ours) - set(probe.theirs)),
                only_theirs=sorted(set(probe.theirs) - set(probe.ours)),
                reason="; ".join(probe.reasons),
            )
        )
        return False

    def tree(state: Any, left: int) -> None:
        # A perft's tree, compared move list by move list rather than by count: a count
        # runs on past an ending (ours has no terminal test in perft, theirs stops), so
        # it differs at every position one move from a win; `verify`'s walk asks only
        # positions our engine calls live, and so does this.
        if left <= 0 or not agree(state):
            return
        for move in engine.legal_moves(state):
            child = engine.apply(state, move)
            legal = engine.legal_moves(child)
            if not legal or engine.terminal(child, legal) is not None:
                continue
            if engine.duck_owed(child):
                continue
            tree(child, left - 1)

    for index, state in enumerate(states):
        if index < perfts:
            counted += 1
            tree(state, depth)
        else:
            agree(state)
    return compared, counted, divergences, []


def late_check(spec: dict[str, Any], *, records: Path | None, positions: int = POSITIONS,
               perfts: int = PERFTS, depth: int = DEPTH, seed: int = 0) -> LateCheck:
    """The late check for one ruleset, against `pyffish`."""
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.fairy import why_not
    from chess_variant_sandbox.fairy.verify import PyffishBackend

    engine = Engine(spec)
    refused = why_not(spec)
    if refused:
        return LateCheck(engine.spec.name, engine.spec.hash, "refused", reasons=refused)
    games = _records(records, engine.spec.hash)
    states = late_states(engine, games, positions, seed)
    result = LateCheck(engine.spec.name, engine.spec.hash, "records+random" if games else "random")
    try:
        compared, counted, found, _ = check_states(
            spec, states, PyffishBackend(), perfts=perfts, depth=depth
        )
    except Exception as error:  # noqa: BLE001 - a backend that fails is a finding
        result.reasons.append(f"{type(error).__name__}: {error}")
        return result
    result.positions, result.perfts, result.divergences = compared, counted, found
    return result


def table(results: list[LateCheck]) -> str:
    """The Markdown table the record keeps."""
    lines = [
        "| game | hash | source | positions | trees | verdict | first difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in results:
        first = ""
        if item.divergences:
            d = item.divergences[0]
            first = (
                f"{len(item.divergences)} positions; `{d.fen}`: ours only {d.only_ours}, "
                f"theirs only {d.only_theirs}" if not d.reason.startswith("perft")
                else f"`{d.fen}`: {d.reason}"
            )
        elif item.reasons and item.source != "refused":
            first = item.reasons[0]
        lines.append(
            f"| {item.name} | `{item.ruleset_hash}` | {item.source} | {item.positions} | "
            f"{item.perfts} | {item.verdict} | {first} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--positions", type=int, default=POSITIONS)
    parser.add_argument("--perfts", type=int, default=PERFTS)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    results: list[LateCheck] = []
    for spec in library_specs(args.root):
        if args.names and spec.get("name") not in args.names:
            continue
        item = late_check(spec, records=args.records, positions=args.positions,
                          perfts=args.perfts)
        if item.source == "refused" and not args.names:
            continue
        results.append(item)
        print(f"{item.name}: {item.verdict} ({item.source}, {item.positions} positions, "
              f"{len(item.divergences)} differ)", flush=True)
    print(table(results))
    if args.json is not None:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=1), "utf-8")
    return 1 if any(r.verdict == "diverges" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
