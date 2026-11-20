"""Random legal combinations of rules, built so that a lifted pair occurs.

A case is a pure function of its seed. The seed draws a `[[lifted]]` row, a small parent, and
then puts each of the row's two key paths into the parent with a *maker*: one of the design
loop's own operators (`loop.operators`, each a valid-spec edit with its settle) or a fragment
written here for a key no operator reaches on a small board (en passant, castling, a capture
duty, a wild piece's route, an immunity, a region, an extinction by kind, a sight on wild
pieces). A maker's child counts only if the validator accepts it, the key it was asked for is
now held, and every key already put in still is. Then up to two more operators land at
random, keeping the pair, so the pair is seen beside a third rule too.

**Nothing the validator refuses is played.** Every child is validated where it is made, and
the finished case once more; a seed whose draws never reach a valid spec holding its pair is
returned with `spec = None` and the reason, which the run counts (`Coverage.unbuilt`), so a
generator that silently stops building a pair shows up in the report as that pair unbuilt.

Small on purpose: six by six or smaller (a five-cell-a-side hex rhombus for the hex rows), at
most eight types, so a case compiles in milliseconds and a short game is a few hundred
microseconds of core time a ply.
"""

from __future__ import annotations

import copy
import functools
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.fuzz_lifted.keys import Lifted, holds, lifted, pairs_present
from chess_variant_sandbox.loop.operators import OPERATORS
from chess_variant_sandbox.spec.canonical import load_spec
from chess_variant_sandbox.spec.schema import (
    format_square,
    parse_square,
    reflect_square,
    validate_spec,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "rulesets"

# How many draws a maker gets to put one key in before the case gives up on it.
ATTEMPTS = 24
# How many further random operators may land once the pair is in, drawn per case.
EXTRAS = (0, 0, 1, 2)
# The operators those draws come from: every one but an operator that edits the capture list
# itself (R-L4b4's `order-the-capture`). The makers already write each row's list in a random
# order, and leaving it out keeps every pinned seed's case the one it was (a draw over a longer
# tuple maps every seed to another case).
_EXTRA_OPERATORS = tuple(op for op in OPERATORS if not op.reads_list)

_KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}

# The grid parent: room on ranks 3 and 4 for a route, a castling lane (b1, c1), a pawn with a
# double step and one promotion rule, so en passant can fire.
SIX: dict[str, Any] = {
    "name": "fuzz six", "board": {"width": 6, "height": 6},
    "pieces": [
        {"id": "K", "royal": True, "value": 0, "moves": [_KING]},
        {"id": "R", "value": 5, "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        {"id": "B", "value": 3, "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}]},
        {"id": "N", "value": 3, "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
        {"id": "P", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
             "condition": {"first_move": True}},
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}]},
    ],
    "setup": {"placements": [
        {"piece": "R", "squares": ["a1"]}, {"piece": "K", "squares": ["d1"]},
        {"piece": "B", "squares": ["e1"]}, {"piece": "N", "squares": ["f1"]},
        {"piece": "P", "squares": ["b2", "c2", "e2"]}]},
    "rules": {"win": "royal_capture", "promotion": {"piece": "P", "to": ["R", "B", "N"]},
              "max_plies": 120},
}

# The same army on five by five, with a single-step pawn: a tighter board, other draws.
FIVE: dict[str, Any] = {
    "name": "fuzz five", "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "royal": True, "value": 0, "moves": [_KING]},
        {"id": "R", "value": 5, "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        {"id": "N", "value": 3, "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}]},
        {"id": "P", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}]},
    ],
    "setup": {"placements": [
        {"piece": "R", "squares": ["a1"]}, {"piece": "K", "squares": ["c1"]},
        {"piece": "N", "squares": ["e1"]}, {"piece": "P", "squares": ["b2", "d2"]}]},
    "rules": {"win": "royal_capture", "promotion": {"piece": "P", "to": ["R", "N"]},
              "max_plies": 120},
}

# A five-by-five hex rhombus, for the rows that name `board.lattice: "hex"`.
HEX: dict[str, Any] = {
    "name": "fuzz hex", "board": {"width": 5, "height": 5, "lattice": "hex"},
    "pieces": [
        {"id": "K", "royal": True, "value": 0,
         "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate6"}]},
        {"id": "R", "value": 5, "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate6"}]},
        {"id": "N", "value": 3, "moves": [{"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate6"}]},
    ],
    "setup": {"mirror": False, "orientation": "rotate", "placements": [
        {"piece": "K", "player": 0, "squares": ["a1"]}, {"piece": "R", "player": 0, "squares": ["b1"]},
        {"piece": "N", "player": 0, "squares": ["a2"]},
        {"piece": "K", "player": 1, "squares": ["e5"]}, {"piece": "R", "player": 1, "squares": ["d5"]},
        {"piece": "N", "player": 1, "squares": ["e4"]}]},
    "rules": {"win": "royal_capture", "max_plies": 120},
}

# Corpus parents small enough to keep a case cheap.
CORPUS_PARENTS = ("gardner", "losalamos", "dobutsu")


@functools.cache
def parents() -> tuple[tuple[str, dict[str, Any]], ...]:
    """The grid parents, by name: the two built here and the small corpus games."""
    found: list[tuple[str, dict[str, Any]]] = [("six", SIX), ("five", FIVE)]
    for name in CORPUS_PARENTS:
        path = CORPUS / f"{name}.json"
        if path.exists():
            found.append((name, load_spec(path)))
    return tuple(found)


# ------------------------------------------------------------------- spelling helpers

def _rules(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.setdefault("rules", {})


def _ids(spec: dict[str, Any], royal: bool | None = None) -> list[str]:
    return [
        str(p["id"]) for p in spec.get("pieces", [])
        if royal is None or bool(p.get("royal")) == royal
    ]


def _piece(spec: dict[str, Any], piece_id: str) -> dict[str, Any]:
    return next(p for p in spec["pieces"] if p["id"] == piece_id)


def _placed(spec: dict[str, Any]) -> set[str]:
    """The types some placement puts on the board."""
    return {str(p.get("piece")) for p in spec.get("setup", {}).get("placements", [])}


def _movers(spec: dict[str, Any]) -> list[str]:
    return [str(p["id"]) for p in spec.get("pieces", []) if p.get("moves") and not p.get("royal")]


def _occupied(spec: dict[str, Any]) -> set[tuple[int, int]]:
    """Every (file, rank) the opening array or a wall uses, reflections included."""
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    setup = spec.get("setup", {})
    orientation = str(setup.get("orientation", "mirror"))
    mirrored = setup.get("mirror", True) and board.get("lattice", "square") != "hex"
    taken: set[tuple[int, int]] = set()
    names: list[str] = [n for p in setup.get("placements", []) for n in p.get("squares", [])]
    obstacles = _rules(spec).get("obstacles")
    walls = list(obstacles.get("squares", [])) if isinstance(obstacles, dict) else []
    for name in names + walls:
        coords = parse_square(name, width, height)
        if coords is None:
            continue
        taken.add(coords)
        if mirrored:
            taken.add(reflect_square(coords[0], coords[1], width, height, orientation))
    for group in _rules(spec).get("portals", []) or []:
        if isinstance(group, dict):
            for name in [group.get("orange"), group.get("blue")] + [
                e.get("at") for e in group.get("ends", []) or [] if isinstance(e, dict)
            ]:
                coords = parse_square(name, width, height) if isinstance(name, str) else None
                if coords:
                    taken.add(coords)
    for entry in _rules(spec).get("npcs", []) or []:
        for name in entry.get("route", []) + entry.get("also_at", []):
            coords = parse_square(name, width, height)
            if coords:
                taken.add(coords)
    return taken


def _empty_blocks(spec: dict[str, Any]) -> list[list[str]]:
    """Every 2x2 block of squares nothing uses, as a four-square loop."""
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    taken = _occupied(spec)
    blocks = []
    for rank in range(height - 1):
        for file in range(width - 1):
            loop = [(file, rank), (file + 1, rank), (file + 1, rank + 1), (file, rank + 1)]
            if not set(loop) & taken:
                blocks.append([format_square(f, r) for f, r in loop])
    return blocks


def _empty_squares(spec: dict[str, Any]) -> list[str]:
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    taken = _occupied(spec)
    return [
        format_square(f, r) for r in range(height) for f in range(width) if (f, r) not in taken
    ]


def _unmirror(spec: dict[str, Any]) -> None:
    """Write the second player's army out, so the two can differ."""
    setup = spec.setdefault("setup", {})
    if not setup.get("mirror", True):
        return
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    orientation = str(setup.get("orientation", "mirror"))
    written = []
    for placement in setup.get("placements", []):
        player = placement.get("player", 0)
        own = dict(placement, player=player)
        written.append(own)
        if player == "nobody" or player == 1:
            continue
        mirrored = []
        for name in placement["squares"]:
            coords = parse_square(name, width, height)
            if coords is None:
                continue
            f, r = reflect_square(coords[0], coords[1], width, height, orientation)
            mirrored.append(format_square(f, r))
        written.append(dict(placement, player=1, squares=mirrored))
    setup["placements"] = written
    setup["mirror"] = False


# ---------------------------------------------------------------- written fragments

Fragment = Callable[[dict[str, Any], random.Random], bool]


def _en_passant(spec: dict[str, Any], rng: random.Random) -> bool:
    """En passant, with a double step given to the promoting piece if it has none (the
    validator refuses the rule where it can never fire)."""
    promoters = _promotion_ids(spec)
    if len(promoters) != 1:
        return False
    _rules(spec)["en_passant"] = True
    if validate_spec(spec):
        piece = next(p for p in spec["pieces"] if p["id"] in promoters)
        piece["moves"] = list(piece.get("moves", [])) + [
            {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
             "condition": {"first_move": True}}
        ]
    return True


def _entry(spec: dict[str, Any], rng: random.Random) -> bool:
    """Promotion on entry, with a hand to enter from if the game has none."""
    if not _promotion_ids(spec):
        return False
    if not isinstance(_rules(spec).get("drops"), dict):
        _rules(spec)["drops"] = {"from_captures": True}
    _rules(spec)["promote_on_entry"] = True
    return True


def _castling_pairs(spec: dict[str, Any], kings: list[str]) -> list[tuple[str, str]]:
    """(castling piece, partner) pairs of the first player's that stand on one rank, three or
    more files apart with nothing between: the ones a castle can actually be played with."""
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    taken = _occupied(spec)
    at: list[tuple[str, int, int]] = []
    for placement in spec.get("setup", {}).get("placements", []):
        if placement.get("player", 0) != 0:
            continue
        for name in placement.get("squares", []):
            coords = parse_square(name, width, height)
            if coords:
                at.append((str(placement["piece"]), coords[0], coords[1]))
    movers = set(_movers(spec))
    found = []
    for king, kf, kr in at:
        if king not in kings:
            continue
        for rook, rf, rr in at:
            if rr != kr or rook == king or rook not in movers or abs(rf - kf) < 3:
                continue
            low, high = sorted((kf, rf))
            if not any((f, kr) in taken for f in range(low + 1, high)):
                found.append((king, rook))
    return found


def _castling(spec: dict[str, Any], rng: random.Random) -> bool:
    royals, others = _ids(spec, royal=True), _movers(spec)
    if not royals or not others:
        return False
    playable = _castling_pairs(spec, royals)
    king, rook = rng.choice(playable) if playable else (rng.choice(royals), rng.choice(others))
    _rules(spec)["castling"] = {"king": king, "rook": rook}
    return True


def _castling_not_royal(spec: dict[str, Any], rng: random.Random) -> bool:
    others = _movers(spec)
    if len(others) < 2:
        return False
    playable = _castling_pairs(spec, others)
    king, rook = rng.choice(playable) if playable else tuple(rng.sample(others, 2))
    _rules(spec)["castling"] = {"king": king, "rook": rook}
    return True


def _must_capture(spec: dict[str, Any], rng: random.Random) -> bool:
    _rules(spec)["must_capture"] = True
    return True


def _route(spec: dict[str, Any], rng: random.Random, **extra: Any) -> bool:
    """A wild piece's loop: a free 2x2 block, or where a parent's ranks leave none (the five
    by five, Gardner, dobutsu: one free rank), two free squares side by side walked back
    and forth (R-L6d: N17 and the other wild rows went unbuilt there)."""
    blocks = _empty_blocks(spec) or _empty_lines(spec, 2)
    types = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    if not blocks or not types:
        return False
    loop = rng.choice(blocks)
    if rng.random() < 0.5:
        loop = loop[:1] + loop[1:][::-1]
    entry: dict[str, Any] = {"piece": rng.choice(types), "route": loop}
    entry.update(extra)
    _rules(spec).setdefault("npcs", []).append(entry)
    return True


def _promotion_ids(spec: dict[str, Any]) -> set[str]:
    """The promoting pieces (not their targets): the type a fragment should leave alone."""
    promotion = _rules(spec).get("promotion")
    rules = promotion if isinstance(promotion, list) else [promotion] if promotion else []
    return {str(rule.get("piece")) for rule in rules if isinstance(rule, dict)}


def _empty_lines(spec: dict[str, Any], length: int) -> list[list[str]]:
    """Every run of ``length`` squares along a rank that nothing uses."""
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    taken = _occupied(spec)
    lines = []
    for rank in range(height):
        for file in range(width - length + 1):
            run = [(file + i, rank) for i in range(length)]
            if not set(run) & taken:
                lines.append([format_square(f, r) for f, r in run])
    return lines


def _line(spec: dict[str, Any], rng: random.Random, queue: bool) -> bool:
    """A wild piece walking a line back and forth (`turns_back`), with a second one on it
    (`also_at`) when ``queue`` -- SR8f's head-on routes (N4). Beside check since SR8h (N5):
    the game keeps its ending, and a line's threat is the square it steps to after its turn."""
    lines = _empty_lines(spec, rng.choice((3, 4)))
    types = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    if not lines or not types:
        return False
    route = rng.choice(lines)
    entry: dict[str, Any] = {"piece": rng.choice(types), "route": route, "turns_back": True}
    if queue:
        entry["also_at"] = [route[-1]]
    _rules(spec).setdefault("npcs", []).append(entry)
    return True


def _checkmate(spec: dict[str, Any], rng: random.Random) -> bool:
    """The game won by checkmate (SR8h's N5 beside a line): only where a side has a royal."""
    if not _ids(spec, royal=True):
        return False
    _rules(spec)["win"] = "checkmate"
    return True


def _forbid_check(spec: dict[str, Any], rng: random.Random) -> bool:
    """Giving check forbidden (SR8h's N5 beside a line): only where a side has a royal."""
    if not _ids(spec, royal=True):
        return False
    _rules(spec)["forbid_check"] = True
    return True


def _reach_safe(spec: dict[str, Any], rng: random.Random) -> bool:
    """A goal that counts only where the arrival stands unattacked (SR8h's N5 beside a line):
    on a goal already there, or one an operator sets first."""
    reach = _rules(spec).get("reach")
    if not isinstance(reach, dict):
        if not _ops("win-by-arrival")[0](spec, rng):
            return False
        reach = _rules(spec).get("reach")
        if not isinstance(reach, dict):
            return False
    reach["safe"] = True
    return True


def _npc(spec: dict[str, Any], rng: random.Random) -> bool:
    return _route(spec, rng)


def _npc_immune(spec: dict[str, Any], rng: random.Random) -> bool:
    takers = _movers(spec) + _ids(spec, royal=True)
    if not takers:
        return False
    immune = sorted(rng.sample(takers, k=min(len(takers), rng.choice((1, 2)))))
    npcs = _rules(spec).get("npcs")
    if npcs:
        rng.choice(npcs)["immune"] = immune
        return True
    return _route(spec, rng, immune=immune)


def _npc_pounce(spec: dict[str, Any], rng: random.Random) -> bool:
    npcs = _rules(spec).get("npcs")
    if npcs:
        rng.choice(npcs)["pounce"] = "prey"
        return True
    return _route(spec, rng, pounce="prey")


def _npc_respawn(spec: dict[str, Any], rng: random.Random) -> bool:
    """A wild entry that comes back (SR8g's N7 beside a pounce): on an entry already there,
    or a new one. Beside split the validator refuses it, so a splitting entry is left."""
    npcs = [e for e in _rules(spec).get("npcs") or [] if not e.get("split")]
    every = rng.choice((1, 2, 3, 4))
    if npcs:
        rng.choice(npcs)["respawn"] = every
        return True
    return _route(spec, rng, respawn=every)


def _npc_pounce_any(spec: dict[str, Any], rng: random.Random) -> bool:
    """A pounce of either kind, on an entry already there or a new one."""
    kind = rng.choice(("prey", "attackers"))
    npcs = _rules(spec).get("npcs")
    if npcs:
        piece = rng.choice(npcs)["piece"]
        for entry in npcs:  # every entry of a type agrees on its pounce
            if entry.get("piece") == piece:
                entry["pounce"] = kind
        return True
    return _route(spec, rng, pounce=kind)


def _extinction_own(spec: dict[str, Any], rng: random.Random) -> bool:
    """Two armies of different kinds: the second player's copy of one type gets its own id."""
    movers = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    if not movers:
        return False
    original = rng.choice(movers)
    twin = next((c for c in "XYZWQ" if c not in _ids(spec)), None)
    if twin is None:
        return False
    _unmirror(spec)
    piece = next(p for p in spec["pieces"] if p["id"] == original)
    spec["pieces"].append(dict(copy.deepcopy(piece), id=twin, name=twin))
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == original and placement.get("player") == 1:
            placement["piece"] = twin
    kinds = sorted(set(_movers(spec)) & _placed(spec))
    _rules(spec)["extinction"] = {"pieces": kinds, "per_type": "own"}
    return True


def _region(spec: dict[str, Any], rng: random.Random) -> bool:
    """A piece kept to its own half and the rank past it."""
    movers = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    if not movers:
        return False
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    piece = _piece(spec, rng.choice(movers))
    depth = height // 2 + rng.choice((0, 1))
    piece["region"] = [format_square(f, r) for r in range(depth) for f in range(width)]
    return True


def _nobody(spec: dict[str, Any], rng: random.Random) -> bool:
    empty = _empty_squares(spec)
    types = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    if not empty or not types:
        return False
    spec["setup"]["placements"].append(
        {"piece": rng.choice(types), "player": "nobody", "squares": [rng.choice(empty)]}
    )
    return True


# ------------------------------------------------- SR8i: a held piece in its holder's frame

_FORWARD_STEP = {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}


def _held_ids(spec: dict[str, Any]) -> list[str]:
    return [
        str(p["id"]) for p in spec.get("pieces", [])
        if p.get("control") == "shared" or p.get("control") == "majority"
        or (isinstance(p.get("control"), dict) and "majority" in p["control"])
    ]


def _held_target(spec: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """A type a player moves but nobody owns where there is one, else a non-royal mover that
    never promotes (the next maker may make it held): the piece SR8i's second key goes on."""
    held = _held_ids(spec)
    if held:
        return _piece(spec, rng.choice(held))
    movers = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    return _piece(spec, rng.choice(movers)) if movers else None


def _directional(piece: dict[str, Any]) -> bool:
    return bool(piece.get("facing") or piece.get("region")) or any(
        isinstance(m, dict) and (m.get("condition") or m.get("screen") in ("own", "enemy")
                                 or m.get("vectors") == [[0, 1]])
        for m in piece.get("moves", [])
    )


def _held(spec: dict[str, Any], rng: random.Random, shared: float = 0.25) -> bool:
    """A type made held (majority, a quarter of the time shared) and placed for nobody on an
    empty square, preferring one that reads differently for the two players."""
    movers = [t for t in _movers(spec) if t not in _promotion_ids(spec) and t not in _held_ids(spec)]
    empty = _empty_squares(spec)
    if not movers or not empty:
        return False
    leaning = [t for t in movers if _directional(_piece(spec, t))]
    piece = _piece(spec, rng.choice(leaning or movers))
    piece["control"] = "shared" if rng.random() < shared else "majority"
    placement: dict[str, Any] = {"piece": piece["id"], "player": "nobody", "squares": [rng.choice(empty)]}
    facing = piece.get("facing")
    if isinstance(facing, dict) and rng.random() < 0.5:
        axes = int(facing.get("axes", 4))
        placement["facing"] = rng.choice(("forward", "right", "backward", "left")[:: 4 // axes])
    spec["setup"]["placements"].append(placement)
    return True


def _held_promoting(spec: dict[str, Any], rng: random.Random) -> bool:
    """A promoting type made held (majority, a quarter of the time shared) and placed for
    nobody on an empty square, so a held piece can promote in its holder's zone (SR8j, O15)."""
    royal = {str(p["id"]) for p in spec.get("pieces", []) if p.get("royal")}
    promoting = [
        t for t in _movers(spec)
        if t in _promotion_ids(spec) and t not in royal and t not in _held_ids(spec)
    ]
    empty = _empty_squares(spec)
    if not promoting or not empty:
        return False
    piece = _piece(spec, rng.choice(promoting))
    piece["control"] = "shared" if rng.random() < 0.25 else "majority"
    spec["setup"]["placements"].append(
        {"piece": piece["id"], "player": "nobody", "squares": [rng.choice(empty)]}
    )
    return True


def _sweeps(piece: dict[str, Any]) -> bool:
    return any(isinstance(m, dict) and m.get("kind") == "sweep" for m in piece.get("moves", []))


def _held_sweeper(spec: dict[str, Any], rng: random.Random) -> bool:
    """A sweeping type made held (majority, a quarter of the time shared) and placed for nobody
    on an empty square, so a held piece sweeps (SR8k, O16)."""
    sweepers = [
        t for t in _movers(spec)
        if _sweeps(_piece(spec, t)) and t not in _promotion_ids(spec) and t not in _held_ids(spec)
    ]
    empty = _empty_squares(spec)
    if not sweepers or not empty:
        return False
    piece = _piece(spec, rng.choice(sweepers))
    piece["control"] = "shared" if rng.random() < 0.25 else "majority"
    spec["setup"]["placements"].append(
        {"piece": piece["id"], "player": "nobody", "squares": [rng.choice(empty)]}
    )
    return True


def _held_sweep(spec: dict[str, Any], rng: random.Random) -> bool:
    """A held type's slide turned into a sweep, or, where none slides, a rook's sweep given to
    it in place of its moves (SR8k, O16)."""
    held = [i for i in _held_ids(spec) if _piece(spec, i).get("moves")]
    if not held:
        return False
    piece = _piece(spec, rng.choice(held))
    slides = [m for m in piece["moves"] if isinstance(m, dict) and m.get("kind") == "slide"
              and not m.get("path") and not m.get("push")]
    if slides:
        rng.choice(slides)["kind"] = "sweep"
    else:
        piece["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4"}]
    return True


def _shared(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (shared.price): a type either player moves, placed for nobody."""
    return _held(spec, rng, shared=1.0)


def _shared_for_hire(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (shared.price): a price on a shared type placed for nobody, so it is for hire, and a
    purse to pay it from where the game has none."""
    shared = [i for i in _held_ids(spec) if _piece(spec, i).get("control") == "shared"]
    if not shared:
        if not _shared(spec, rng):
            return False
        shared = [i for i in _held_ids(spec) if _piece(spec, i).get("control") == "shared"]
    piece = _piece(spec, rng.choice(shared))
    piece["price"] = rng.randint(1, 4)
    purse = _rules(spec).setdefault("purse", {"start": rng.randint(2, 8)})
    return isinstance(purse, dict)


def _bounty_somewhere(spec: dict[str, Any], rng: random.Random) -> None:
    if not any(p.get("bounty") for p in spec.get("pieces", [])):
        movers = [p for p in spec.get("pieces", []) if not p.get("royal") and p.get("moves")]
        if movers:
            rng.choice(movers)["bounty"] = rng.randint(1, 3)


def _points(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (income.I1): the clock decided on points, which a bounty fills."""
    _rules(spec)["scoring"] = {"count": "points"}
    _bounty_somewhere(spec, rng)
    return True


def _income(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (income.I1): income on the first rank, with a purse (and a bounty for it to spend
    on) where the game has none."""
    rules = _rules(spec)
    if not isinstance(rules.get("purse"), dict):
        movers = [t for t in _movers(spec) if t not in _ids(spec, royal=True)]
        if not movers:
            return False
        sold = rng.choice(movers)
        _piece(spec, sold)["price"] = rng.randint(2, 5)
        rules["purse"] = {"start": rng.randint(0, 6), "market": [sold]}
    rules["income"] = True if rng.random() < 0.5 else {"per_piece": rng.randint(1, 3)}
    return True


def _bank(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (goals.extinction, goals.bare_royal): a goal on the far rank for the non-royal pieces
    that do not promote, whose arrivals are banked, two to win."""
    runners = [t for t in _movers(spec)
               if t not in _ids(spec, royal=True) and t not in _promotion_ids(spec)]
    if not runners:
        return False
    width, height = spec["board"]["width"], spec["board"]["height"]
    rules = _rules(spec)
    rules["reach"] = {
        "piece": sorted(rng.sample(runners, k=min(len(runners), 2))),
        "squares": [format_square(f, height - 1) for f in range(width)],
        "removes": True, "count": 2,
    }
    if rules.get("win") == "checkmate":
        rules["win"] = "royal_capture"
    return True


def _displace(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (run.territory): a hop over a run that displaces, on a game with no royal (a
    displacement beside a royal stays refused): the royals step down and the game is won by
    annihilation."""
    rules = _rules(spec)
    for piece in spec.get("pieces", []):
        piece.pop("royal", None)
    rules["win"] = "annihilation"
    for key in ("castling", "forbid_check", "forbid_immobility", "check_limit", "bare_royal"):
        rules.pop(key, None)
    movers = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    if not movers:
        return False
    _piece(spec, rng.choice(movers))["moves"].append({
        "kind": "hop", "vectors": [[0, 1], [1, 0]], "symmetry": "rotate4", "mode": "move",
        "run": True, "displace": "back",
    })
    return True


def _extinction(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (goals.extinction): a side that runs out of one placed, non-royal type loses."""
    kinds = sorted(_placed(spec) - set(_ids(spec, royal=True)))
    if not kinds:
        return False
    chosen = rng.sample(kinds, k=min(len(kinds), rng.choice((1, 1, 2))))
    _rules(spec)["extinction"] = {"pieces": chosen}
    return True


def _bare_royal(spec: dict[str, Any], rng: random.Random) -> bool:
    """U2 (goals.bare_royal): a side left with its royal alone loses."""
    if not set(_ids(spec, royal=True)) & _placed(spec):
        return False
    _rules(spec)["bare_royal"] = "loss"
    return True


def _face(spec: dict[str, Any], rng: random.Random) -> bool:
    """A facing on the held type (or a mover), its moves made a pawn's (a forward step, the
    forward diagonals to take) so that a turn changes them (the compiler refuses a facing its
    moves are symmetric under)."""
    piece = _held_target(spec, rng)
    if piece is None or piece.get("facing"):
        return False
    piece["facing"] = {"axes": rng.choice((2, 4)), "initial": "forward"}
    piece["moves"] = [
        copy.deepcopy(_FORWARD_STEP),
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
    ]
    return True


def _nobody_facing(spec: dict[str, Any], rng: random.Random) -> bool:
    """A placement for nobody that states a facing (O9), on a type that faces."""
    facing = [p for p in spec.get("pieces", []) if isinstance(p.get("facing"), dict) and not p.get("royal")]
    if not facing and not _face(spec, rng):
        return False
    facing = [p for p in spec.get("pieces", []) if isinstance(p.get("facing"), dict) and not p.get("royal")]
    piece = rng.choice(facing)
    axes = int(piece["facing"].get("axes", 4))
    name = rng.choice(("forward", "right", "backward", "left")[:: 4 // axes])
    for placement in spec["setup"]["placements"]:
        if placement.get("player") == "nobody" and placement.get("piece") == piece["id"]:
            placement["facing"] = name
            return True
    empty = _empty_squares(spec)
    if not empty:
        return False
    spec["setup"]["placements"].append(
        {"piece": piece["id"], "player": "nobody", "squares": [rng.choice(empty)], "facing": name}
    )
    return True


def _held_move(move: dict[str, Any]) -> Fragment:
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        piece = _held_target(spec, rng)
        if piece is None:
            return False
        piece["moves"] = list(piece.get("moves", [])) + [copy.deepcopy(move)]
        return True
    add.__name__ = f"held_{move['kind']}"
    return add


def _held_condition(spec: dict[str, Any], rng: random.Random) -> bool:
    condition = rng.choice(({"from_rank": 1}, {"first_move": True}))
    return _held_move({"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2,
                       "mode": "move", "condition": condition})(spec, rng)


def _held_screen(spec: dict[str, Any], rng: random.Random) -> bool:
    return _held_move({"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4",
                       "screen": rng.choice(("own", "enemy"))})(spec, rng)


def _held_region(spec: dict[str, Any], rng: random.Random) -> bool:
    """A region on the held type (or a mover): its own half and the rank past it."""
    piece = _held_target(spec, rng)
    if piece is None:
        return False
    board = spec["board"]
    width, height = int(board["width"]), int(board["height"])
    depth = height // 2 + rng.choice((0, 1))
    piece["region"] = [format_square(f, r) for r in range(depth) for f in range(width)]
    return True


def _sight(effect: str, **extra: Any) -> Fragment:
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        movers = _movers(spec)
        if not movers:
            return False
        piece = _piece(spec, rng.choice(movers))
        whose = {"command": "enemy", "shield": rng.choice(("own", "any"))}.get(
            effect, rng.choice(("enemy", "any"))
        )
        sight = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": rng.choice((1, 2)),
                 "effect": effect, "whose": whose}
        sight.update(extra)
        piece["sight"] = [sight]
        return True
    return add


def _sight_on_wild(spec: dict[str, Any], rng: random.Random) -> bool:
    rules = _rules(spec)
    wild = [str(e["piece"]) for e in rules.get("npcs", []) or [] if isinstance(e, dict)]
    duck = rules.get("duck", {}).get("piece") if isinstance(rules.get("duck"), dict) else None
    if not wild and not duck:
        if not _route(spec, rng):
            return False
        wild = [str(rules["npcs"][-1]["piece"])]
    movers = [t for t in _movers(spec) if t not in wild and t != duck]
    if not movers:
        return False
    target = rng.choice(wild + ([duck] if duck else []))
    effect = "freeze" if target == duck else rng.choice(("freeze", "disarm", "shield"))
    piece = _piece(spec, rng.choice(movers))
    piece["sight"] = [{"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 2,
                       "effect": effect, "whose": "any", "affects": [target]}]
    return True


def _portal_moved(spec: dict[str, Any], rng: random.Random) -> bool:
    groups = _rules(spec).get("portals")
    if not groups:
        if not _operator("open-a-portal")(spec, rng):
            return False
        groups = _rules(spec).get("portals")
    group = next((g for g in groups or [] if isinstance(g, dict) and "orange" in g), None)
    if group is None:
        return False
    hex_board = spec["board"].get("lattice") == "hex"
    group["moves"] = [{"kind": "leap", "vectors": [[1, 0]] if hex_board else [[1, 0], [1, 1]],
                       "symmetry": "rotate6" if hex_board else "rotate4"}]
    players = rng.choice(([0, 1], [0], [1]))
    group["moved_by"] = {"orange": players, "blue": rng.choice(([0, 1], players, []))}
    return True


def _strike(**strike: Any) -> Fragment:
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        rules = _rules(spec)
        royals = _ids(spec, royal=True)
        wild = {str(e.get("piece")) for e in rules.get("npcs", []) or [] if isinstance(e, dict)}
        written = dict(strike)
        if "by" in written:
            movers = [t for t in _movers(spec) if t not in wild]
            if len(movers) < 2:
                return False
            written["by"] = [rng.choice(movers)]
        if "switches_sides" in written:
            movers = [t for t in _movers(spec) if t not in wild]
            if not movers:
                return False
            written["switches_sides"] = sorted(rng.sample(movers, k=min(2, len(movers))))
        if royals:
            written["immune"] = sorted(royals)
        rules["strike"] = written
        return True
    return add


def _switch_and_by(spec: dict[str, Any], rng: random.Random) -> bool:
    """T10: a shooter list beside a switch list, read apart."""
    if not _strike(effect="capture", position="stay", by=True)(spec, rng):
        return False
    movers = [t for t in _movers(spec) if t not in _rules(spec)["strike"]["by"]]
    if not movers:
        return False
    _rules(spec)["strike"]["switches_sides"] = [rng.choice(movers)]
    return True


# ------------------------------------- R-L6d: lame jumps, turning rides, legs on hex

# The xiangqi horse, and on a hex lattice the hex knight a neighbour blocks
# (`tests/test_lame_legs.py`): a lame jump must state its legs there.
_SQUARE_LAME: dict[str, Any] = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"}
_HEX_LAMES: tuple[dict[str, Any], ...] = (
    {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral12", "legs": [[[0, 1]]]},
    {"kind": "lame", "vectors": [[1, 1]], "symmetry": "rotate6", "legs": [[[1, 0], [0, 1]]]},
)
# Stated legs on a square board: the computed leg, or the diagonal the other walk passes.
_SQUARE_LEGS = ([[[0, 1]]], [[[1, 1]]])


def _hex_board(spec: dict[str, Any]) -> bool:
    return spec["board"].get("lattice") == "hex"


def _lame_move(spec: dict[str, Any], rng: random.Random, rider: bool, legs: bool) -> dict[str, Any]:
    move: dict[str, Any]
    if _hex_board(spec):
        move = copy.deepcopy(rng.choice(_HEX_LAMES))
    else:
        move = copy.deepcopy(_SQUARE_LAME)
        if legs:
            move["legs"] = copy.deepcopy(rng.choice(_SQUARE_LEGS))
    if rider:
        move["range"] = rng.choice((0, 2, 3))
        if rng.random() < 0.25:
            move["min_range"] = 2
    return move


def _lame(rider: bool, legs: bool) -> Fragment:
    """A lame jump given to a mover that does not promote (beside its moves, or in their
    place): the parents have none, so `ride-a-leap` and `state-the-legs`, which need one,
    never fired on them (the reason PA4 and X1 had no case before R-L6d)."""
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        movers = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
        if not movers:
            return False
        piece = _piece(spec, rng.choice(movers))
        move = _lame_move(spec, rng, rider, legs)
        piece["moves"] = ([] if rng.random() < 0.5 else list(piece.get("moves", []))) + [move]
        return True
    add.__name__ = f"lame{'_rider' if rider else ''}{'_legs' if legs else ''}"
    return add


def _bend_a_ride(spec: dict[str, Any], rng: random.Random) -> bool:
    """A turning path on a lame rider (SR11a, PA4), one made first where the game has none."""
    riders = [m for p in spec.get("pieces", []) for m in p.get("moves", [])
              if isinstance(m, dict) and m.get("kind") == "lame" and m.get("range", 1) != 1]
    if not riders:
        if not _lame(rider=True, legs=rng.random() < 0.5)(spec, rng):
            return False
        return _bend_a_ride(spec, rng)
    move = rng.choice(riders)
    move["path"] = rng.choice(("circle", "zigzag"))
    if move["path"] == "zigzag" and move.get("range") == 0:
        move["range"] = rng.choice((0, 3))
    return True


def _blocks_leaps(spec: dict[str, Any], rng: random.Random) -> bool:
    """`obstacles.blocks_leaps` with a wall or two on empty squares, every leap longer than a
    step stating its legs (on a hex lattice it must, SR1a, X8): the hex knight's diagonal
    jump blocked on either neighbour it passes."""
    empty = _empty_squares(spec)
    if len(empty) < 2:
        return False
    for piece in spec.get("pieces", []):
        for move in piece.get("moves", []):
            if isinstance(move, dict) and move.get("kind") == "leap" and "legs" not in move:
                if move.get("vectors") == [[1, 1]] and _hex_board(spec):
                    move["legs"] = [[[1, 0], [0, 1]]]
                elif move.get("vectors") == [[1, 2]]:
                    move["legs"] = copy.deepcopy(rng.choice(_SQUARE_LEGS))
    walls = sorted(rng.sample(empty, k=rng.choice((1, 2))))
    obstacles = _rules(spec).setdefault("obstacles", {})
    if not isinstance(obstacles, dict):
        return False
    obstacles["squares"] = sorted(set(obstacles.get("squares", [])) | set(walls))
    obstacles["blocks_leaps"] = True
    return True


def _pouncing_path(spec: dict[str, Any], rng: random.Random) -> bool:
    """A turning path on a prey pouncer's own slide (SR2, N20), so the pounce rides it: a
    pouncer's slides bent, or a queen's ride given to it, bent."""
    npcs = [e for e in _rules(spec).get("npcs", []) or [] if e.get("pounce") == "prey"]
    if not npcs:
        if not _npc_pounce(spec, rng):
            return False
        npcs = [e for e in _rules(spec)["npcs"] if e.get("pounce") == "prey"]
    piece = _piece(spec, str(rng.choice(npcs)["piece"]))
    slides = [m for m in piece.get("moves", []) if isinstance(m, dict) and m.get("kind") == "slide"
              and m.get("range", 0) != 1 and not m.get("push")]
    if not slides:
        slides = [{"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
        piece["moves"] = list(piece.get("moves", [])) + slides
    rng.choice(slides)["path"] = rng.choice(("circle", "zigzag"))
    return True


def _two_faces(spec: dict[str, Any], rng: random.Random) -> bool:
    """A promoting piece's promotion made a second face of it (`promoted_from`), with a
    capture hand that returns either face (`drops.either_face`, H6): `either-face` needs a
    face pair, which only dobutsu among the parents has (R-L6d: H6 fell under eight)."""
    promotion = _rules(spec).get("promotion")
    if not isinstance(promotion, dict) or not promotion.get("piece"):
        return False
    base = str(promotion["piece"])
    face = next((c for c in "WXYZQ" if c not in _ids(spec)), None)
    sources = [t for t in _movers(spec) if t != base]
    if face is None or not sources:
        return False
    source = _piece(spec, rng.choice(sources))
    spec["pieces"].append({"id": face, "value": 4, "promoted_from": base,
                           "moves": copy.deepcopy(source["moves"])})
    promotion["to"] = [face]
    drops = _rules(spec).get("drops")
    if not isinstance(drops, dict):
        drops = _rules(spec)["drops"] = {"from_captures": True}
    drops["either_face"] = True
    return True


def _removing_link(spec: dict[str, Any], rng: random.Random) -> bool:
    """A contact link that removes its target and changes its mover (SR2, T5): the meeting's
    target leaves the board as a capture's does. Written here beside the two operators,
    which decline on most parents (R-L6d: T5 fell under eight)."""
    movers = [t for t in _movers(spec) if t not in _promotion_ids(spec)]
    if len(movers) < 2:
        return False
    mover, becomes = rng.sample(movers, 2)
    onto = rng.choice(movers)
    contacts: dict[str, Any] = {"links": [{"mover": mover, "onto": onto, "mover_to": becomes}]}
    royals = _ids(spec, royal=True)
    if royals:
        contacts["immune"] = sorted(royals)
    _rules(spec)["contacts"] = contacts
    return True


# ------------------------------------------------------- R-L4b stage 3: the capture list

#: A ``choose`` table's landings: the four orthogonal neighbours of the struck piece.
_LANDINGS = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def _ladder(spec: dict[str, Any]) -> list[str]:
    """The non-royal movers, most valuable first: a demotion chain walks down it."""
    wild = {str(e.get("piece")) for e in _rules(spec).get("npcs", []) or [] if isinstance(e, dict)}
    movers = [t for t in _movers(spec) if t not in wild and t not in _ids(spec, royal=True)]
    return sorted(movers, key=lambda t: -float(_piece(spec, t).get("value", 1) or 0))


def _demote(position: str) -> Fragment:
    """A strike that demotes down the value ladder, placed ``position`` (it keeps its victim)."""
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        ladder = _ladder(spec)
        if len(ladder) < 2:
            return False
        strike: dict[str, Any] = {
            "effect": "demote", "position": position,
            "chain": [{"from": a, "to": b} for a, b in zip(ladder, ladder[1:], strict=False)],
        }
        if position == "choose":
            strike["landing"] = {"vectors": _LANDINGS}
        royals = _ids(spec, royal=True)
        if royals:
            strike["immune"] = sorted(royals)
        _rules(spec)["strike"] = strike
        return True
    add.__name__ = f"demote_{position}"
    return add


def _keeping_link(position: str = "advance") -> Fragment:
    """A contact table whose link keeps its target (``onto_to``), placed ``position``."""
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        ladder = _ladder(spec)
        if len(ladder) < 2:
            return False
        mover, onto = rng.choice(ladder), rng.choice(ladder)
        contacts: dict[str, Any] = {
            "links": [{"mover": mover, "onto": onto, "onto_to": rng.choice(ladder)}],
            "position": position,
        }
        if position == "choose":
            contacts["landing"] = {"vectors": _LANDINGS}
        royals = _ids(spec, royal=True)
        if royals:
            contacts["immune"] = sorted(royals)
        _rules(spec)["contacts"] = contacts
        return True
    add.__name__ = f"keeping_link_{position}"
    return add


def _capture_hand(spec: dict[str, Any], rng: random.Random) -> bool:
    """A hand that fills from captures: the game's own hand switched on, or a new one."""
    drops = _rules(spec).get("drops")
    if isinstance(drops, dict) and drops:
        drops["from_captures"] = True
        return True
    return _operator("hand-the-pieces-back")(spec, rng)


def _as_list(spec: dict[str, Any], rng: random.Random, shuffle: bool = True) -> bool:
    """The game's capture rules written as ``rules.capture``, in an order drawn at random."""
    from chess_variant_sandbox.spec.capture import as_list_spelling

    listed = as_list_spelling(spec)
    if listed is None:
        return False
    if shuffle:
        rng.shuffle(listed["rules"]["capture"])
    spec.clear()
    spec.update(listed)
    return True


def _listed(fragment: Fragment) -> Fragment:
    """``fragment``, then the capture rules written as a list in a random order: the list the
    old keys cannot say, beside the same rules."""
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        from chess_variant_sandbox.spec.capture import capture_written

        if capture_written(spec):
            return False  # a fragment writes the old keys; a list already written stays
        return fragment(spec, rng) and _as_list(spec, rng)
    add.__name__ = f"listed_{getattr(fragment, '__name__', 'maker').lstrip('_')}"
    return add


def _entry_of(spec: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for entry in _rules(spec).get("capture") or []:
        if isinstance(entry, dict) and entry.get("do") == kind:
            return entry
    return None


def _spared_blast(spec: dict[str, Any], rng: random.Random) -> bool:
    """A blast that spares the piece whose capture set it off, ahead of a strike that then
    reaches it (one placed ``swap`` put in where the game has no strike or contacts), written
    as a list."""
    rules = _rules(spec)
    if "capture" not in rules:
        if not rules.get("explode_on_capture") and not _operator("set-the-blast")(spec, rng):
            return False
        if (not rules.get("strike") and not rules.get("contacts")
                and not _demote(rng.choice(("swap", "stay", "home")))(spec, rng)):
            return False
        if not _as_list(spec, rng, shuffle=False):
            return False
    blast = _entry_of(spec, "blast")
    if blast is None:
        return False
    blast["spares_taken"] = True
    if rng.random() < 0.5:
        rng.shuffle(_rules(spec)["capture"])
    return True


def _blast_read(spec: dict[str, Any], rng: random.Random) -> bool:
    """A strike or contact entry that reads a blast's victims, written as a list."""
    if "capture" not in _rules(spec) and not _as_list(spec, rng):
        return False
    meets = [e for e in (_entry_of(spec, "strike"), _entry_of(spec, "contacts")) if e is not None]
    if not meets or _entry_of(spec, "blast") is None:
        return False
    entry = rng.choice(meets)
    from chess_variant_sandbox.spec.capture import CAPTURE_DEFAULT_ON

    entry["on"] = sorted(set(entry.get("on", CAPTURE_DEFAULT_ON[entry["do"]])) | {"blast"})
    return True


def _path_by(spec: dict[str, Any], rng: random.Random) -> bool:
    """A strike entry reading the path of one sweeper of two, written as a list."""
    from chess_variant_sandbox.spec.capture import CAPTURE_DEFAULT_ON, sweeper_ids

    sliders = [
        p for p in spec["pieces"] if not p.get("royal") and p["id"] not in sweeper_ids(spec)
        and any(m.get("kind") == "slide" and m.get("mode", "both") == "both" for m in p.get("moves", []))
    ]
    rng.shuffle(sliders)
    for piece in sliders[: max(0, 2 - len(sweeper_ids(spec)))]:
        for move in piece["moves"]:
            if move.get("kind") == "slide" and move.get("mode", "both") == "both":
                move["kind"] = "sweep"
                move.pop("path", None)
    sweepers = sweeper_ids(spec)
    if len(sweepers) < 2:
        return False
    if "capture" not in _rules(spec):
        if not _rules(spec).get("strike") and not _demote(rng.choice(("stay", "home")))(spec, rng):
            return False
        if not _as_list(spec, rng):
            return False
    entry = _entry_of(spec, "strike") or _entry_of(spec, "contacts")
    if entry is None:
        return False
    entry["on"] = sorted(set(entry.get("on", CAPTURE_DEFAULT_ON[entry["do"]])) | {"path"})
    entry["path_by"] = [rng.choice(sweepers)]
    return True


def _blast_into_list(spec: dict[str, Any], rng: random.Random) -> bool:
    """A blast of radius 1 sparing the royals, put at a random place in a written list."""
    listed = _rules(spec).get("capture")
    if not isinstance(listed, list) or _entry_of(spec, "blast") is not None:
        return False
    blast: dict[str, Any] = {"do": "blast", "radius": 1}
    royals = _ids(spec, royal=True)
    if royals:
        blast["immune"] = sorted(royals)
    listed.insert(rng.randrange(len(listed) + 1), blast)
    return True


def _blast_beside_a_strike(spec: dict[str, Any], rng: random.Random) -> bool:
    """A blast and a strike, each put in where the game has none."""
    if not _rules(spec).get("explode_on_capture") and not _operator("set-the-blast")(spec, rng):
        return False
    if not _rules(spec).get("strike") and not _rules(spec).get("contacts"):
        return _demote(rng.choice(("stay", "home")))(spec, rng)
    return True


def _blast_beside_a_mover(kind: str) -> Fragment:
    """R-L4b5 (T1.rounds, T4.rounds): a blast and a strike (``kind`` "strike") or a contact
    table that moves its taker (placed "advance", "swap" or "home", never "stay"), each put in
    where the game has none: the pair a round walks as its capture list."""
    def add(spec: dict[str, Any], rng: random.Random) -> bool:
        rules = _rules(spec)
        if not rules.get("explode_on_capture") and not _operator("set-the-blast")(spec, rng):
            return False
        if kind == "strike":
            if rules.get("strike"):
                return rules["strike"].get("position", "advance") != "stay"
            return _demote(rng.choice(("swap", "home")))(spec, rng)
        if rules.get("contacts"):
            return rules["contacts"].get("position", "advance") != "stay"
        return _keeping_link(rng.choice(("advance", "swap", "home")))(spec, rng)
    add.__name__ = f"blast_beside_a_moving_{kind}"
    return add


@functools.cache
def _operators() -> dict[str, Any]:
    return {op.name: op for op in OPERATORS}


def _operator(name: str) -> Fragment:
    def apply(spec: dict[str, Any], rng: random.Random) -> bool:
        op = _operators().get(name)
        return op is not None and op.apply(spec, rng) is not None
    apply.__name__ = name
    return apply


def _ops(*names: str) -> list[Fragment]:
    return [_operator(name) for name in names]


_STAY = _strike(effect="capture", position="stay")
_HOME = _strike(effect="capture", position="home")
_BY = _strike(effect="capture", position="stay", by=True)
_SWITCH = _strike(switches_sides=True)

# Each key path to the makers that can put it in. The first entries are tried as often as the
# rest: the draw is uniform over the list.
MAKERS: dict[str, list[Fragment]] = {
    "rules.en_passant": [_en_passant, *_ops("toggle-a-rule")],
    "rules.turn": _ops("lengthen-the-turn", "split-the-turn"),
    "pieces[].rest": _ops("tire-a-piece"),
    "rules.npcs": [_npc, _npc, *_ops("set-a-patrol", "pace-a-line", "set-the-hounds")],
    'pieces[].control: "majority"': _ops("gather-a-crowd"),
    "rules.purse": _ops("open-a-market", "hire-a-sword", "post-a-bounty"),
    "rules.duck": _ops("loose-a-duck"),
    "rules.simultaneous": _ops("play-at-once"),
    "rules.castling": [_castling],
    "rules.portals[].moved_by": [_portal_moved, *_ops("open-a-portal")],
    'rules.strike.position: "stay"': [_STAY, *_ops("soften-the-capture", "leave-something-behind")],
    'rules.strike.position: "home"': [_HOME, *_ops("leave-something-behind")],
    'setup.placements[].player: "nobody"': [_nobody, *_ops("gather-a-crowd", "hire-a-sword")],
    "rules.purse.market": _ops("open-a-market"),
    "rules.strike": [_STAY, _HOME, _SWITCH, *_ops("soften-the-capture", "leave-something-behind")],
    "rules.strike (removing its victim)": [_STAY, _BY],
    "rules.strike.by": [_BY, _switch_and_by],
    "rules.strike.switches_sides": [_SWITCH, _switch_and_by],
    "rules.strike.chain": _ops("soften-the-capture"),
    "rules.explode_on_capture": _ops("set-the-blast", "shape-the-blast"),
    "rules.drops": _ops("hand-the-pieces-back", "place-a-stone", "give-a-handicap"),
    "rules.drops.from_captures": _ops("hand-the-pieces-back"),
    "rules.drops.stock": _ops("place-a-stone", "give-a-handicap", "place-the-army"),
    "rules.drops.either_face": [_two_faces, *_ops("either-face")],
    "rules.contacts": _ops("become-what-you-take", "merge-your-own"),
    "rules.contacts.links": _ops("become-what-you-take", "merge-your-own"),
    "rules.contacts (links removing their target)": [
        _removing_link, *_ops("become-what-you-take", "merge-your-own"),
    ],
    "rules.contacts (an own link)": _ops("merge-your-own"),
    "rules.must_capture": [_must_capture],
    "rules.promotion": _ops("shift-promotion"),
    "rules.promotion.only_lost": _ops("promote-only-lost"),
    "rules.promote_on_entry": [_entry, *_ops("promote-what-arrives")],
    'rules.extinction.per_type: "own"': [_extinction_own, *_ops("count-each-kind")],
    # SR8k (O16): a held type's slide made a sweep, so the sweep is a held piece's.
    'pieces[].moves[].kind: "sweep"': [*_ops("sweep-a-slider"), _held_sweep],
    "rules.wrap": _ops("join-the-edges"),
    "rules.portals": _ops("open-a-portal", "fold-a-portal"),
    'pieces[].sight effect "shield"': [_sight("shield"), *_ops("cast-a-gaze")],
    'pieces[].sight effect "command"': [_sight("command"), *_ops("cast-a-gaze")],
    "pieces[].sight": [_sight("freeze"), _sight("disarm"), _sight("shield"), *_ops("cast-a-gaze")],
    "pieces[].sight captures_own": [_sight("command", captures_own=True)],
    "pieces[].sight affects": [_sight_on_wild],
    "a wild type or the duck": [_npc, *_ops("loose-a-duck")],
    "rules.reach": _ops("win-by-arrival", "win-by-leaving"),
    "rules.territory": _ops("claim-the-ground"),
    "rules.npcs[].immune": [_npc_immune],
    "rules.npcs[].turns_back": [lambda s, r: _line(s, r, queue=True)],
    "rules.npcs[].also_at": [lambda s, r: _line(s, r, queue=True)],
    'rules.npcs[].pounce: "prey"': [_npc_pounce, _pouncing_path],
    "rules.npcs[].respawn": [_npc_respawn],
    'rules.win: "checkmate"': [_checkmate],
    "rules.forbid_check": [_forbid_check],
    "rules.reach.safe": [_reach_safe],
    "rules.npcs[].pounce": [_npc_pounce_any],
    # R-L6d: a ride bent on a lame rider (PA4) or on a prey pouncer's own slide (N20).
    "pieces[].moves[].path": [*_ops("bend-a-line"), _bend_a_ride, _pouncing_path],
    "rules.explode_on_capture, rules.strike or rules.contacts": [
        _STAY, *_ops("set-the-blast", "become-what-you-take")
    ],
    'board.lattice: "hex"': [],  # the hex parent carries it
    "pieces[].region": [_region, _held_region],
    "pieces[].spy": _ops("plant-a-spy"),
    "pieces[].price": _ops("open-a-market", "hire-a-sword"),
    'moves[].kind: "lame" (a rider)': [_lame(rider=True, legs=False), _bend_a_ride,
                                       *_ops("ride-a-leap")],
    'pieces[].moves[].kind: "lame" (stating its legs)': [
        _lame(rider=False, legs=True), _lame(rider=True, legs=True), *_ops("state-the-legs"),
    ],
    "rules.obstacles.blocks_leaps (leaps stating their legs)": [_blocks_leaps],
    "rules.castling.king": [_castling_not_royal],
    "a piece that is not royal": [_castling_not_royal],
    "pieces[].moves[].push": _ops("push-instead"),
    "pieces[].sight (but a command that captures_own: push.N4)": [_sight("freeze"), _sight("shield")],
    # SR8i (O9, O13, O14, O17, O18, O19): the second key goes on the held type where there is one.
    # SR8j (O15): a promoting type made held, so the promotion is a held piece's.
    # SR8k (O16): a sweeping type made held, so the sweep is a held piece's.
    'pieces[].control: "majority" or "shared"': [
        _held, _held, _held_promoting, _held_sweeper, *_ops("gather-a-crowd"),
    ],
    "pieces[].facing": [_face],
    "setup.placements[].facing": [_nobody_facing],
    "pieces[].moves (reading differently for the two players)": [_held_move(_FORWARD_STEP)],
    "pieces[].moves[].condition": [_held_condition],
    "pieces[].moves[].screen own or enemy": [_held_screen],
    # U2 (unheld-pairs-plan.md section 2.2): nine pairs the core already played.
    "rules.income": [_income, *_ops("pay-an-income")],
    'rules.scoring.count "points"': [_points],
    "rules.drops.gating": _ops("gate-the-hand"),
    "rules.reach.removes": [_bank, *_ops("bank-the-arrivals")],
    "rules.extinction": [_extinction],
    "rules.bare_royal": [_bare_royal],
    'pieces[].control: "shared"': [_shared],
    "pieces[].price (on a shared type)": [_shared_for_hire],
    "pieces[].moves[].run or displace": _ops("leap-a-row"),
    'pieces[].moves[].displace: "back"': [_displace, *_ops("leap-a-row")],
    # R-L4b stage 3: the capture list's entries, each written as the old keys (the list's
    # default order) or, by `_listed`, as rules.capture in a random order.
    "rules.capture: a blast": [*_ops("set-the-blast", "shape-the-blast"), _listed(_operator("set-the-blast")),
                               _blast_into_list],
    "rules.capture: a hand": [_capture_hand, _listed(_capture_hand)],
    "rules.capture: a strike": [_STAY, _HOME, _SWITCH, _demote("advance"), _listed(_HOME),
                                _listed(_demote("swap"))],
    "rules.capture: contacts": [_removing_link, _keeping_link(), *_ops("become-what-you-take"),
                                _listed(_keeping_link())],
    "rules.capture: a blast that spares the taken piece": [_spared_blast],
    "rules.capture: a blast that spares the taken piece or is read": [_spared_blast, _blast_read],
    'rules.capture: a strike placed "advance", "swap" or "choose"': [
        _demote("advance"), _demote("swap"), _demote("choose"), _listed(_demote("swap")),
    ],
    'rules.capture: a strike placed "choose"': [_demote("choose"), _listed(_demote("choose"))],
    'rules.capture: contacts placed "choose"': [_keeping_link("choose"), _listed(_keeping_link("choose"))],
    'rules.capture: a strike placed "stay", "swap", "home" or "choose"': [
        _STAY, _HOME, _demote("stay"), _demote("swap"), _listed(_HOME),
    ],
    "rules.capture: a strike that keeps its victim": [_HOME, _SWITCH, _demote("stay"), _demote("advance"),
                                                      _listed(_demote("home"))],
    "rules.capture: contacts that keep their target": [_keeping_link(), _keeping_link("stay"),
                                                       _listed(_keeping_link())],
    "rules.capture: a strike that switches sides": [_SWITCH, _listed(_SWITCH)],
    "rules.capture: an entry reading some sweepers' path (path_by)": [_path_by],
    "rules.capture: a blast beside a strike or contacts": [_blast_beside_a_strike,
                                                          _listed(_blast_beside_a_strike)],
    "rules.purse.upkeep": _ops("charge-for-moves"),
    # R-L4b5: the two round rows, as the old keys (the default order) or as a list.
    "rules.capture: a blast beside a strike that moves its taker": [
        _blast_beside_a_mover("strike"), _listed(_blast_beside_a_mover("strike")),
    ],
    "rules.capture: a blast beside contacts that move their taker": [
        _blast_beside_a_mover("contacts"), _listed(_blast_beside_a_mover("contacts")),
    ],
}


@dataclass
class LiftedCase:
    """One generated case: the row it was built for, and the spec, or why there is none."""

    seed: int
    row: str
    parent: str
    makers: tuple[str, ...]
    spec: dict[str, Any] | None
    present: tuple[str, ...] = ()
    why: str = ""
    refused: int = 0
    extras: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        made = " + ".join(self.makers + self.extras) or "nothing"
        return f"seed {self.seed}: row {self.row}, {self.parent} + {made}"


def _put(
    spec: dict[str, Any], key: str, kept: list[str], rng: random.Random
) -> tuple[dict[str, Any] | None, str, int]:
    """``spec`` with ``key`` put in and every key of ``kept`` still held, or ``None``; the
    maker's name, and how many valid-looking children the validator refused."""
    if holds(spec, key):
        return spec, "(held)", 0
    makers = MAKERS.get(key, [])
    refused = 0
    if not makers:
        return None, "", 0
    for _ in range(ATTEMPTS):
        maker = rng.choice(makers)
        child = copy.deepcopy(spec)
        try:
            made = maker(child, rng)
        except Exception:  # noqa: BLE001 - an edit that cannot apply here is a no
            made = False
        if not made:
            continue
        if validate_spec(child):
            refused += 1
            continue
        if holds(child, key) and all(holds(child, k) for k in kept):
            return child, getattr(maker, "__name__", "maker").lstrip("_"), refused
    return None, "", refused


def buildable(rows: tuple[Lifted, ...] | None = None) -> tuple[Lifted, ...]:
    """The rows a maker exists for on both sides (the hex key is the parent's)."""
    rows = lifted() if rows is None else rows
    return tuple(
        row for row in rows
        if all(MAKERS.get(k) or k == 'board.lattice: "hex"' for k in row.pair)
    )


def case(seed: int, rows: tuple[Lifted, ...] | None = None) -> LiftedCase:
    """The case ``seed`` builds. A pure function of the seed (and the row list)."""
    rng = random.Random(seed)
    rows = buildable(rows)
    row = rng.choice(rows)
    if 'board.lattice: "hex"' in row.pair:
        parent_name, parent = "hex", HEX
    else:
        parent_name, parent = rng.choice(parents())
    spec = copy.deepcopy(parent)
    spec["name"] = f"fuzz {row.id} seed {seed}"
    keys = list(row.pair)
    rng.shuffle(keys)
    made: list[str] = []
    refused = 0
    kept: list[str] = []
    for key in keys:
        child, maker, failed = _put(spec, key, kept, rng)
        refused += failed
        if child is None:
            return LiftedCase(seed, row.id, parent_name, tuple(made), None,
                              why=f"no maker reached {key}", refused=refused)
        spec = child
        made.append(maker)
        kept.append(key)
    extras: list[str] = []
    for _ in range(rng.choice(EXTRAS)):
        op = rng.choice(_EXTRA_OPERATORS)
        child = copy.deepcopy(spec)
        try:
            if op.apply(child, rng) is None:
                continue
        except Exception:  # noqa: BLE001
            continue
        if validate_spec(child):
            refused += 1
            continue
        if all(holds(child, k) for k in kept):
            spec = child
            extras.append(op.name)
    if validate_spec(spec):  # never played: the last word is the validator's
        return LiftedCase(seed, row.id, parent_name, tuple(made), None,
                          why="the finished case is refused", refused=refused + 1)
    return LiftedCase(
        seed, row.id, parent_name, tuple(made), spec,
        present=tuple(pairs_present(spec)), refused=refused, extras=tuple(extras),
    )
