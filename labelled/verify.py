"""Walk every certificate in labelled/certificates/ through the engine (labelled-set.md §1.2).

    uv run python labelled/verify.py            # every certificate
    uv run python labelled/verify.py <name>     # labelled/certificates/<name>.json only

A certificate proves a `forced-win` claim: the winner's move at each of its turns and *every*
legal reply at each of the loser's, down to positions the engine calls won for the winner.
The moves are data, spelled as `Engine.describe_move` spells them. This script chooses none
(CLAUDE.md rule 3): it asks the engine for the legal moves, applies the ones the certificate
names, and asks the engine for the ending. It plays no game and runs no player.

The file is

    {"game": "labelled/<name>.json", "spec_hash": "...", "winner": 0 | 1,
     "tree": {"move": "<winner's move>",
              "replies": {"<every legal reply>": {"move": ..., "replies": {...}}, ...}}}

where a node is a winner's move. A move after which the game is over has no "replies" (or an
empty object), and the ending must be the winner's. At a loser's turn the replies' keys must
be exactly the loser's legal moves; a reply that itself ends the game (a capture the loser is
made to take, say) must end it as the winner's win, and its value is null. When the winner
moves second the tree is `{"replies": {...}}`, the loser's every first move. A certificate that passes has its SHA-256 written into
`labelled/verified.json`, which `tests/test_labelled_set.py` checks against the file.
"""

# compute-lease: light

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CERTIFICATES = HERE / "certificates"
VERIFIED = HERE / "verified.json"


class Refuted(Exception):
    """The certificate does not prove its claim; the message says where."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(engine: Any, state: Any, node: dict[str, Any], winner: int, line: list[str]) -> int:
    """Check one winner's move and everything under it; the count of positions walked."""
    if engine.terminal(state) is not None:
        raise Refuted(f"after {' '.join(line) or 'the opening'} the game is already over")
    if state.side_to_move != winner:
        raise Refuted(f"after {' '.join(line) or 'the opening'} it is not the winner's turn")
    text = node.get("move")
    if not isinstance(text, str):
        raise Refuted(f"after {' '.join(line) or 'the opening'} no winner's move is named")
    try:
        move = engine.find_move(state, text)
    except LookupError as error:
        raise Refuted(f"after {' '.join(line) or 'the opening'}: {error}") from None
    state = engine.apply(state, move)
    line = [*line, text]
    outcome = engine.terminal(state)
    replies = node.get("replies") or {}
    if outcome is not None:
        if replies:
            raise Refuted(f"after {' '.join(line)} the game is over but replies are listed")
        if outcome.winner != winner:
            raise Refuted(f"after {' '.join(line)} the game ends {outcome}, not a win")
        return 1
    if state.side_to_move == winner:
        # A turn of several moves: the winner moves again, and the node continues as one.
        if replies:
            raise Refuted(f"after {' '.join(line)} the winner moves again; list no replies")
        nxt = node.get("then")
        if not isinstance(nxt, dict):
            raise Refuted(f"after {' '.join(line)} the winner moves again and no 'then' is given")
        return 1 + _walk(engine, state, nxt, winner, line)
    return 1 + _answer(engine, state, replies, winner, line)


def _answer(engine: Any, state: Any, replies: dict[str, Any], winner: int, line: list[str]) -> int:
    """The loser to move: every legal reply is listed and each one loses."""
    legal = sorted(engine.move_texts(state))
    if sorted(replies) != legal:
        missing = sorted(set(legal) - set(replies))
        extra = sorted(set(replies) - set(legal))
        raise Refuted(
            f"after {' '.join(line) or 'the opening'} the replies are not every legal move: "
            f"missing {missing}, not legal {extra}"
        )
    walked = 0
    for reply in legal:
        after = engine.apply(state, engine.find_move(state, reply))
        outcome = engine.terminal(after)
        if outcome is not None:
            # A reply that ends the game must end it as the winner's win (a capture the loser
            # is forced to make, say), and is then a leaf, written as null.
            if outcome.winner != winner or replies[reply] is not None:
                raise Refuted(f"after {' '.join([*line, reply])} the loser's reply ends the game {outcome}")
            walked += 1
            continue
        if replies[reply] is None:
            raise Refuted(f"after {' '.join([*line, reply])} the game goes on; a node is needed")
        walked += _walk(engine, after, replies[reply], winner, [*line, reply])
    return walked


def check(path: Path) -> int:
    """Raise `Refuted` unless the certificate at ``path`` proves its claim."""
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.spec import load_spec, spec_hash

    data = json.loads(path.read_text(encoding="utf-8"))
    spec = load_spec(HERE.parent / data["game"])
    if spec_hash(spec) != data.get("spec_hash"):
        raise Refuted(f"{data['game']} is not the game the certificate names")
    engine = Engine(spec)
    if engine.simultaneous:
        raise Refuted("a game played in rounds has no certificate of this shape")
    winner = data.get("winner")
    if winner not in (0, 1):
        raise Refuted("winner must be 0 or 1")
    state = engine.initial_state()
    tree = data["tree"]
    if state.side_to_move != winner:
        # The winner moves second: the tree opens with the loser's every first move.
        if engine.terminal(state) is not None or set(tree) != {"replies"}:
            raise Refuted("the loser moves first: the tree must be {'replies': {...}} alone")
        return _answer(engine, state, tree["replies"], winner, [])
    return _walk(engine, state, tree, winner, [])


def main(argv: list[str]) -> int:
    names = [a for a in argv if not a.startswith("-")]
    paths = (
        [CERTIFICATES / f"{n}.json" for n in names]
        if names else sorted(CERTIFICATES.glob("*.json"))
    )
    verified = json.loads(VERIFIED.read_text(encoding="utf-8")) if VERIFIED.exists() else {}
    failed = 0
    for path in paths:
        try:
            walked = check(path)
        except Refuted as error:
            failed += 1
            verified.pop(path.stem, None)
            print(f"REFUTED  {path.stem}: {error}")
            continue
        verified[path.stem] = digest(path)
        print(f"holds    {path.stem}: {walked} positions walked")
    VERIFIED.write_text(json.dumps(dict(sorted(verified.items())), indent=2) + "\n", encoding="utf-8")
    print(f"{len(paths) - failed} of {len(paths)} certificates hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
