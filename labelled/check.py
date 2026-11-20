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
- `mirror`: `side` answers every move of the other side with its image, the same move with
  every square turned half round the `board` ([width, height]). Every position
  reachable so is visited: the image must always be a legal move, the other side must never
  win, and, where `decisive` is true, every ending must be `side`'s win (no clock ending).

Every reachable-position search walks past clock endings (repetition, no progress, the ply
limit), so it covers more than any one history reaches, and a claim that no loss is reached
is the stronger for it.
"""

# compute-lease: light

from __future__ import annotations

import json
import re
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


_SQUARE = re.compile(r"([a-p])(1[0-6]|[1-9])")


def _turned(text: str, width: int, height: int) -> str:
    """The move text with every square turned half round a width x height board."""
    def turn(m: re.Match[str]) -> str:
        return f"{chr(ord('a') + width - 1 - (ord(m.group(1)) - ord('a')))}{height + 1 - int(m.group(2))}"
    return _SQUARE.sub(turn, text)


def _mirror(check: dict[str, Any], engine: Any, state: Any) -> tuple[bool, str]:
    from collections import deque

    width, height = check["board"]
    side = check["side"]
    seen = {prove.key(engine, state)}
    queue = deque([state])
    endings: dict[str, int] = {}
    lost = stuck = 0

    def ended(st: Any) -> bool:
        nonlocal lost
        outcome = engine.terminal(st)
        if outcome is None:
            return False
        name = f"{outcome.winner}:{outcome.reason}"
        endings[name] = endings.get(name, 0) + 1
        if outcome.winner is not None and outcome.winner != side:
            lost += 1
        if check.get("decisive") and outcome.winner != side:
            lost += 1
        return True

    while queue:
        here = queue.popleft()
        if ended(here):
            continue
        for text, after in prove.moves(engine, here):
            if ended(after):
                continue
            image = _turned(text, width, height)
            answer = dict(prove.moves(engine, after)).get(image)
            if answer is None:
                stuck += 1
                continue
            k = prove.key(engine, answer)
            if k not in seen:
                seen.add(k)
                queue.append(answer)
    ok = lost == 0 and stuck == 0
    return ok, f"{len(seen)} positions, endings {endings}, {lost} not {side}'s, {stuck} without the image"


def run(check: dict[str, Any], engine: Any, root: Path) -> tuple[bool, str]:
    state = engine.initial_state()
    kind = check["type"]
    if kind == "mirror":
        return _mirror(check, engine, state)
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
