"""The equivalence fixture for schema 29's rewrite of the old rest (`docs/plan-rest.md` §4.5).

Schema 29 (Q983a) retired `rules.simultaneous.rest` for `pieces[].rest` and rewrote every
game that used it. Three of those games are played in rounds only, and for them the rewrite
(`"rest": 1` on every piece type) claims to be the same game move for move. This module is
the claim's evidence: run as a script **on main's core before the bump** (Q983a's first
commit), it wrote `tests/rest_equivalence.json` -- each game's joint perft at depths 1 to 3
and twenty seeded random walks, every round's two moves, the key after it, each player's
rested square and the result. `tests/test_rest_equivalence.py` replays the walks on the
rewritten files and asserts every number, move, key, rest and result.

    uv run python tests/rest_equivalence.py     # rewrites the fixture: only ever on the old core

The walk reads a position's rest through `rested_squares`, which understands both the old
boundary shape (`(square or -1, square or -1)`) and schema 29's (`((square, left), ...)` per
player), so the same walk runs on both cores.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.canonical import spec_hash

HERE = Path(__file__).parent
ROOT = HERE.parent
FIXTURE = HERE / "rest_equivalence.json"

# The three rounds-only games, by the file that carries each after the bump. Two other test
# files share a game with these (`_simultaneous_scored.json` is breather's game,
# `_retired_gardner_in_rounds_resting.json` is `_simultaneous_resting.json`'s).
GAMES = {
    "632c199e4589": "archive/breather.json",
    "4da669bb3d50": "tests/_simultaneous_resting.json",
    "aeda9633fed3": "tests/_simultaneous_speed_resting.json",
}
SEEDS = range(20)
MAX_PLIES = 200
PERFT_DEPTHS = (1, 2, 3)


REWRITE_HISTORY = (
    "Rewritten for schema 29 by Q983a (2026-10-20): the old switch rules.simultaneous.rest as "
    '"rest": 1 on every piece (docs/plan-rest.md section 4.5).'
)


def rewrite(spec: dict[str, Any]) -> dict[str, Any]:
    """Schema 29's rewrite of a spec carrying schema 27's ``rules.simultaneous.rest``
    (`docs/plan-rest.md` §4.5): the switch removed, and where it was on, ``"rest": 1`` on
    every piece type a player moves (one with a move, and no wild piece, which never made a
    player's move); ``schema_version`` 29, and a sentence on ``provenance.history`` where the
    spec has one. The store's migration (``site/migrations/0039_rest_on_pieces.sql``) is the
    same rewrite in SQL, and its test holds the two together."""
    out = json.loads(json.dumps(spec))
    out["schema_version"] = 29
    was_on = out["rules"]["simultaneous"].pop("rest") is True
    npcs = out["rules"].get("npcs")
    wild = {npc.get("piece") for npc in npcs if isinstance(npc, dict)} if isinstance(npcs, list) else set()
    for piece in out["pieces"]:
        if was_on and piece.get("moves") and piece.get("id") not in wild:
            piece["rest"] = 1
    provenance = out.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("history"), str):
        provenance["history"] = f"{provenance['history']} {REWRITE_HISTORY}"
    return out


def rested_squares(state: GameState) -> list[int]:
    """Each player's rested square, or -1: the old shape read as is, schema 29's read off
    its list, which for these games holds at most one entry a player, one turn left."""
    rest: Any = state.rest
    if rest and isinstance(rest[0], int):
        return [int(rest[0]), int(rest[1])]
    squares = []
    for entries in rest:
        assert len(entries) <= 1, f"more than one piece rests: {rest}"
        assert all(left == 1 for _, left in entries), f"a rest longer than one turn: {rest}"
        squares.append(entries[0][0] if entries else -1)
    return squares


def walk(engine: Engine, seed: int) -> dict[str, Any]:
    """One seeded random game in rounds, at most `MAX_PLIES` plies, every round recorded."""
    rng = random.Random(seed)
    state = engine.initial_state()
    rounds: list[list[Any]] = []
    while state.ply < MAX_PLIES:
        if engine.terminal(state) is not None:
            break
        first = rng.choice(engine.legal_moves_for(state, 0))
        second = rng.choice(engine.legal_moves_for(state, 1))
        state = engine.apply_round(state, first, second)
        rounds.append([first, second, state.key, rested_squares(state)])
    outcome = engine.terminal(state)
    result = None if outcome is None else [outcome.winner, outcome.reason]
    return {"seed": seed, "start": engine.initial_state().key, "rounds": rounds, "result": result}


def capture(path: str) -> dict[str, Any]:
    spec = json.loads((ROOT / path).read_text(encoding="utf-8"))
    engine = Engine(spec)
    state = engine.initial_state()
    return {
        "file": path,
        "hash": spec_hash(spec),
        "perft": [engine.perft(state, depth) for depth in PERFT_DEPTHS],
        "walks": [walk(engine, seed) for seed in SEEDS],
    }


def main() -> None:
    games = {}
    for old_hash, path in GAMES.items():
        captured = capture(path)
        assert captured["hash"] == old_hash, f"{path} is not the old game: {captured['hash']}"
        games[old_hash] = captured
        print(old_hash, path, captured["perft"])
    lines = ["{"]
    for index, (old_hash, game) in enumerate(games.items()):
        lines.append(f'  "{old_hash}": {{')
        lines.append(f'    "file": {json.dumps(game["file"])},')
        lines.append(f'    "perft": {json.dumps(game["perft"])},')
        lines.append('    "walks": [')
        walks = [f"      {json.dumps(one, separators=(',', ':'))}" for one in game["walks"]]
        lines.append(",\n".join(walks))
        lines.append("    ]")
        lines.append("  }" + ("," if index < len(games) - 1 else ""))
    lines.append("}")
    FIXTURE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
