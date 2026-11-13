"""Searches behind the bad games' proofs (labelled-set.md §1.2), over the engine's own rules.

Every function here asks `Engine` for legal moves, the move's text, the next position and
the ending, and nothing else: no evaluator, no player, no score (CLAUDE.md rule 3 allows
Python to hold a position and ask; these searches are the proofs' declared oracles, as
`labelled/verify.py` is the certificates'). The proof scripts in `labelled/proofs/` call them.

- `forced_win(engine, state, winner, n)`: a certificate tree in `verify.py`'s shape that the
  winner wins within `n` of its own moves against every defence, or None. Depth-first AND-OR
  search, memoised by position key and moves left (the walk in `verify.py` is the check).
- `shortest_win(engine, state, winner, most)`: the least such `n`, with its tree.
- `explore(engine, state, allowed=...)`: every position reachable from `state`, a side's
  moves optionally cut to those a recipe allows (`allowed(side, text) -> bool`), with every
  decided ending found and every position where a restricted side has a legal move but none
  the recipe allows. Clock endings (repetition, stall, ply limit) are walked past, so the set
  over-approximates what any history reaches: a claim that no loss is reachable is the
  stronger for it.
"""

# compute-lease: light

from __future__ import annotations

import json
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLOCKS = {"repetition", "no_progress", "ply_limit"}


def load(name: str) -> tuple[Any, Any]:
    """The engine of `labelled/<name>.json` and its opening position."""
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.spec import load_spec

    engine = Engine(load_spec(HERE / f"{name}.json"))
    return engine, engine.initial_state()


def key(engine: Any, state: Any) -> tuple[int, int, int]:
    return engine.hash(state), state.side_to_move, state.remaining


def play(engine: Any, state: Any, texts: list[str]) -> Any:
    for text in texts:
        if engine.terminal(state) is not None:
            raise ValueError(f"the game is over before {text}")
        state = engine.apply(state, engine.find_move(state, text))
    return state


def moves(engine: Any, state: Any) -> list[tuple[str, Any]]:
    """Every legal move's text and the position after it, sorted by text."""
    out = []
    for move in engine.legal_moves(state):
        out.append((engine.describe_move(state, move), engine.apply(state, move)))
    out.sort(key=lambda pair: pair[0])
    return out


class _Search:
    def __init__(self, engine: Any, winner: int) -> None:
        self.engine = engine
        self.winner = winner
        self.lost: dict[tuple[Any, ...], int] = {}  # key -> the most moves known not to win
        self.nodes = 0

    def win(self, state: Any, n: int) -> dict[str, Any] | None:
        """The winner to move: a node that wins within n of its moves, or None."""
        if n <= 0:
            return None
        k = key(self.engine, state)
        if self.lost.get(k, 0) >= n:
            return None
        self.nodes += 1
        children = moves(self.engine, state)
        # Moves that end the game at once first, then by fewest replies.
        for text, after in children:
            outcome = self.engine.terminal(after)
            if outcome is not None and outcome.winner == self.winner:
                return {"move": text}
        # With one move left a win may still come from the loser's forced reply (antichess: a
        # capture it must make that ends the game as the winner's win), so no early return here:
        # `defend(after, 0)` accepts exactly the replies that end the game for the winner.
        ranked = []
        for text, after in children:
            if self.engine.terminal(after) is not None:
                continue
            if after.side_to_move == self.winner:
                ranked.append((0, text, after))  # a turn of several moves: rare here
                continue
            ranked.append((len(self.engine.legal_moves(after)), text, after))
        ranked.sort(key=lambda r: (r[0], r[1]))
        for _, text, after in ranked:
            if after.side_to_move == self.winner:
                sub = self.win(after, n)
                if sub is not None:
                    return {"move": text, "then": sub}
                continue
            replies = self.defend(after, n - 1)
            if replies is not None:
                return {"move": text, "replies": replies}
        self.lost[k] = max(self.lost.get(k, 0), n)
        return None

    def defend(self, state: Any, n: int) -> dict[str, Any] | None:
        """The loser to move: every reply loses within n more winner's moves, or None."""
        out: dict[str, Any] = {}
        for text, after in moves(self.engine, state):
            outcome = self.engine.terminal(after)
            if outcome is not None:
                if outcome.winner == self.winner:
                    out[text] = None  # the loser's own move ends it as the winner's win
                    continue
                return None  # a reply that ends the game otherwise refutes the line
            if after.side_to_move == state.side_to_move:
                return None  # a turn of several moves: not searched here
            sub = self.win(after, n)
            if sub is None:
                return None
            out[text] = sub
        return out


def forced_win(engine: Any, state: Any, winner: int, n: int) -> dict[str, Any] | None:
    return _Search(engine, winner).win(state, n)


def shortest_win(engine: Any, state: Any, winner: int, most: int) -> tuple[int, dict[str, Any]] | None:
    """The least n <= most within which the winner forces a win, with its tree."""
    search = _Search(engine, winner)
    for n in range(1, most + 1):
        tree = search.win(state, n)
        if tree is not None:
            return n, tree
    return None


def size(tree: dict[str, Any] | None) -> int:
    if tree is None:
        return 1
    return 1 + sum(size(t) for t in (tree.get("replies") or {}).values()) + (
        size(tree["then"]) if "then" in tree else 0
    )


def write_certificate(name: str, winner: int, tree: dict[str, Any], cert: str | None = None) -> Path:
    from chess_variant_sandbox.spec import load_spec, spec_hash

    spec = load_spec(HERE / f"{name}.json")
    path = HERE / "certificates" / f"{cert or name}.json"
    path.parent.mkdir(exist_ok=True)
    body = {"game": f"labelled/{name}.json", "spec_hash": spec_hash(spec), "winner": winner, "tree": tree}
    path.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def explore(
    engine: Any,
    state: Any,
    allowed: Callable[[int, str], bool] | None = None,
    restricted: tuple[int, ...] = (),
    limit: int = 5_000_000,
) -> dict[str, Any]:
    """Every position reachable from `state`, the `restricted` sides held to `allowed`.

    Returns the count of positions, every decided ending reached (winner, reason, and one
    line to it), and every position where a restricted side had legal moves but none the
    recipe allows (`stuck`), each with a line.
    """
    seen: dict[tuple[Any, ...], tuple[Any, str] | None] = {key(engine, state): None}
    queue = deque([state])
    wins: dict[int, list[list[str]]] = {0: [], 1: []}
    reasons: dict[str, int] = {}
    stuck: list[list[str]] = []

    def line(k: tuple[Any, ...]) -> list[str]:
        out = []
        while seen[k] is not None:
            parent, text = seen[k]  # type: ignore[misc]
            out.append(text)
            k = parent
        return out[::-1]

    while queue:
        if len(seen) > limit:
            raise RuntimeError(f"more than {limit} positions")
        here = queue.popleft()
        k = key(engine, here)
        outcome = engine.terminal(here)
        if outcome is not None:
            reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
            if outcome.winner is not None:
                if len(wins[outcome.winner]) < 3:
                    wins[outcome.winner].append(line(k))
                else:
                    wins[outcome.winner].append([])
                continue
            if outcome.reason not in CLOCKS:
                continue
        side = here.side_to_move
        children = moves(engine, here)
        if allowed is not None and side in restricted:
            kept = [(t, a) for t, a in children if allowed(side, t)]
            if children and not kept:
                stuck.append(line(k))
            children = kept
        for text, after in children:
            ka = key(engine, after)
            if ka not in seen:
                seen[ka] = (k, text)
                queue.append(after)
    return {
        "positions": len(seen),
        "wins": {w: len(v) for w, v in wins.items()},
        "win_lines": {w: [x for x in v if x][:3] for w, v in wins.items()},
        "endings": reasons,
        "stuck": len(stuck),
        "stuck_lines": stuck[:3],
    }


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
