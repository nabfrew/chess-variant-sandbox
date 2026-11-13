"""Re-run the mechanical part of every argued proof in labelled/labels/ (labelled-set.md §1.2).

    uv run python labelled/check.py            # every label with proof checks
    uv run python labelled/check.py <name>     # labelled/labels/<name>.json only

An argued proof (`proof.form` "argument") may carry `proof.checks`, a list of claims that the
engine can settle by walking the game's own rules. Each is data in the label; this script
reads it, asks `Engine` for legal moves, positions and endings (through `prove.py`), and
prints whether it holds. It chooses no move and plays no game (CLAUDE.md rule 3): a line is
walked as written, and a search visits every legal move.

The kinds of check:

- `line`: `moves` walked from the opening; the game must end there, won by `winner`.
- `recipe`: `side` moves only its piece `piece` and only onto `squares` (a step onto `take`
  is also allowed). Every position reachable with the other side free is visited: no win for
  the other side may be reached, and the recipe may never leave the side with legal moves but
  none it allows.
- `no_win`: every position reachable is visited; no ending with a winner is reached.
- `no_win_for`: as `no_win`, for one `loser` side only: no ending won by the other side.
- `no_capture`: `depth` plies of every line from the opening hold no capture (`x` in a move).
- `same_moves`: the game and `parent` have the same legal moves, spelled alike, in every
  position of every line to `depth` plies (perft and move lists compared).
- `shortest`: `winner` has no forced win in fewer than `moves` of its own moves.

Every reachable-position search walks past clock endings (repetition, no progress, the ply
limit), so it covers more than any one history reaches, and a claim that no loss is reached
is the stronger for it.
"""

# compute-lease: light

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prove  # noqa: E402


def _engine(path: Path) -> Any:
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.spec import load_spec

    return Engine(load_spec(path))


def _dest(text: str) -> str:
    tail = text.split("-")[-1].split("x")[-1]
    return tail.split("=")[0].rstrip("+#")


def _recipe(side: int, piece: str, squares: set[str], take: str | None) -> Any:
    def allowed(mover: int, text: str) -> bool:
        if mover != side:
            return True
        return text.startswith(piece) and (_dest(text) in squares or _dest(text) == take)
    return allowed


def run(check: dict[str, Any], engine: Any, root: Path) -> tuple[bool, str]:
    state = engine.initial_state()
    kind = check["type"]
    if kind == "line":
        end = prove.play(engine, state, check["moves"])
        outcome = engine.terminal(end)
        ok = outcome is not None and outcome.winner == check["winner"]
        return ok, f"ends {outcome}"
    if kind == "recipe":
        side = check["side"]
        found = prove.explore(
            engine, state,
            allowed=_recipe(side, check["piece"], set(check["squares"]), check.get("take")),
            restricted=(side,),
        )
        lost = found["wins"][1 - side]
        ok = lost == 0 and found["stuck"] == 0
        return ok, f"{found['positions']} positions, {lost} lost, {found['stuck']} stuck"
    if kind in ("no_win", "no_win_for"):
        found = prove.explore(engine, state)
        if kind == "no_win":
            won = found["wins"][0] + found["wins"][1]
        else:
            won = found["wins"][1 - check["loser"]]
        return won == 0, f"{found['positions']} positions, {won} won, endings {found['endings']}"
    if kind == "no_capture":
        seen = 0

        def walk(st: Any, depth: int) -> bool:
            nonlocal seen
            if depth == 0 or engine.terminal(st) is not None:
                return True
            for text, after in prove.moves(engine, st):
                seen += 1
                if "x" in text or not walk(after, depth - 1):
                    return False
            return True
        ok = walk(state, check["depth"])
        return ok, f"{seen} moves read"
    if kind == "same_moves":
        other = _engine(root / check["parent"])
        theirs = other.initial_state()
        count = 0

        def same(a: Any, b: Any, depth: int) -> bool:
            nonlocal count
            mine, its = prove.moves(engine, a), prove.moves(other, b)
            if [t for t, _ in mine] != [t for t, _ in its]:
                return False
            if (engine.terminal(a) is None) != (other.terminal(b) is None):
                return False
            count += 1
            if depth == 0:
                return True
            return all(same(x, y, depth - 1) for (_, x), (_, y) in zip(mine, its, strict=True))
        ok = same(state, theirs, check["depth"])
        return ok, f"{count} positions compared"
    if kind == "shortest":
        winner, n = check["winner"], check["moves"]
        search = prove._Search(engine, winner)
        sooner = (
            search.win(state, n - 1) if state.side_to_move == winner
            else search.defend(state, n - 1)
        )
        return sooner is None, f"no forced win in {n - 1}" if sooner is None else "a sooner win exists"
    return False, f"unknown check {kind}"


def main(argv: list[str]) -> int:
    names = [a for a in argv if not a.startswith("-")]
    labels = sorted((HERE / "labels").glob("*.json"))
    if names:
        # A name, or a path to a label file (a label not yet written into labels/).
        labels = []
        for n in names:
            if Path(n).is_dir():
                labels += sorted(Path(n).glob("*.json"))
            else:
                labels.append(Path(n) if n.endswith(".json") else HERE / "labels" / f"{n}.json")
    failed = 0
    for path in labels:
        label = json.loads(path.read_text(encoding="utf-8"))
        checks = (label.get("proof") or {}).get("checks") or []
        if not checks:
            continue
        engine = _engine(HERE.parent / label["game"])
        for check in checks:
            ok, says = run(check, engine, HERE.parent)
            failed += not ok
            print(f"{'holds  ' if ok else 'FAILS  '} {path.stem}: {check['type']} {says}", flush=True)
    print("every check holds" if not failed else f"{failed} checks fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
