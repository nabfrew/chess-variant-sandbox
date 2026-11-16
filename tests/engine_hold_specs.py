"""The smallest specs that reach each of the eleven engine holds the validator names (U1's H1 to
H6 and C1 to C4; U1b's H7, from the fuzz).

`docs/reference/unheld-pairs-plan.md` §1.2 and §1.3 give each one as a library game plus one
change; these build exactly that, so a test can ask the validator, the compiler and the boundary
about the same file. Each builder returns a fresh dict.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


@cache
def _library(name: str) -> str:
    return (RULESETS / f"{name}.json").read_text(encoding="utf-8")


def library(name: str) -> dict[str, Any]:
    """A fresh copy of ``rulesets/<name>.json``."""
    return json.loads(_library(name))


def _piece(spec: dict[str, Any], pid: str) -> dict[str, Any]:
    return next(piece for piece in spec["pieces"] if piece["id"] == pid)


def push_beside_gating() -> dict[str, Any]:
    """H1: seirawan, the knight given a king's-step leap that pushes."""
    spec = library("seirawan")
    _piece(spec, "N")["moves"].append({
        "kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
        "mode": "move", "push": True,
    })
    return spec


def push_beside_held_piece(control: Any = "majority") -> dict[str, Any]:
    """H2: dohyo plus a type with ``control`` placed for nobody on d4."""
    spec = library("dohyo")
    spec["pieces"].append({
        "id": "B", "name": "boulder", "value": 3, "weight": 1, "control": control,
        "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
    })
    spec["setup"]["placements"].append({"piece": "B", "player": "nobody", "squares": ["d4"]})
    return spec


def push_beside_attackers_pouncer() -> dict[str, Any]:
    """H3: dohyo plus a wild type on a three-square route that pounces on its attackers."""
    spec = library("dohyo")
    spec["pieces"].append({
        "id": "Y", "name": "hydra", "value": 0,
        "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
    })
    spec["rules"]["npcs"] = [{"piece": "Y", "route": ["c4", "d4", "e4"], "pounce": "attackers"}]
    return spec


def upkeep_beside_push() -> dict[str, Any]:
    """H4: hunger without the pass, the knight given a pushing move."""
    spec = library("hunger")
    del spec["rules"]["pass"]
    _piece(spec, "N")["moves"].append({
        "kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move", "push": True,
    })
    return spec


def upkeep_beside_the_duck() -> dict[str, Any]:
    """H7 (U1b, fuzz find F5): hunger without the pass, won by taking the royal, plus a
    moveless duck."""
    spec = library("hunger")
    del spec["rules"]["pass"]
    spec["rules"]["win"] = "royal_capture"
    spec["pieces"].append({"id": "D", "name": "duck", "value": 0, "moves": []})
    spec["rules"]["duck"] = {"piece": "D"}
    return spec


def two_displacing_hops_meet() -> dict[str, Any]:
    """H5: leapfrog, the frog given a second displacing hop with ``run: 3``."""
    spec = library("leapfrog")
    frog = _piece(spec, "F")
    second = copy.deepcopy(frog["moves"][1])
    second["run"] = 3
    frog["moves"].append(second)
    return spec


def two_pushes_one_landing() -> dict[str, Any]:
    """H6: dohyo, the bouncer given two diagonal leaps that push, one along the rank, one
    along the file."""
    spec = library("dohyo")
    bouncer = _piece(spec, "O")
    for along in ("rank", "file"):
        bouncer["moves"].append({
            "kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "move",
            "push": {"along": along},
        })
    return spec


def too_many_push_directions() -> dict[str, Any]:
    """C1: dohyo, the bouncer given a pushing leap over five vectors under dihedral8 (40
    directions with both players')."""
    spec = library("dohyo")
    _piece(spec, "O")["moves"].append({
        "kind": "leap", "vectors": [[1, 2], [1, 3], [2, 3], [1, 4], [3, 4]],
        "symmetry": "dihedral8", "mode": "move", "push": True,
    })
    return spec


def too_many_sweep_rays() -> dict[str, Any]:
    """C2: torus, the rook given a sweep over six vectors under dihedral8 (40 rays)."""
    spec = library("torus")
    _piece(spec, "R")["moves"] = [{
        "kind": "sweep", "vectors": [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [1, 4]],
        "symmetry": "dihedral8", "mode": "both",
    }]
    return spec


def sweep_ray_too_long_beside_folds() -> dict[str, Any]:
    """C3: chess on a 7 by 8 board joined on both pairs of edges, one fold pair, the bishop's
    move a sweep (its diagonal runs 56 squares)."""
    spec = library("chess")
    spec["board"]["width"] = 7
    for placement in spec["setup"]["placements"]:
        placement["squares"] = [square for square in placement["squares"] if square[0] != "h"]
    rules = spec["rules"]
    rules.pop("castling", None)
    rules["wrap"] = {"files": "both", "ranks": "both"}
    rules["portals"] = [{"kind": "fold", "orange": "c4", "blue": "e5"}]
    _piece(spec, "B")["moves"] = [{
        "kind": "sweep", "vectors": [[1, 1]], "symmetry": "rotate4", "mode": "both",
    }]
    return spec


def too_many_landing_candidates() -> dict[str, Any]:
    """C4: chess plus a ``choose`` strike whose ``landing`` is seven vectors under dihedral8
    (40 squares)."""
    spec = library("chess")
    spec["rules"]["strike"] = {
        "effect": "promote", "position": "choose",
        "landing": {
            "vectors": [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [1, 4], [2, 2]],
            "symmetry": "dihedral8",
        },
        "chain": [{"from": "P", "to": "N"}],
        "immune": ["K"],
    }
    return spec


# id -> (builder, the words of the boundary's or compiler's refusal it reaches)
HOLDS: dict[str, tuple[Callable[[], dict[str, Any]], str]] = {
    "H1": (push_beside_gating, "gating (E5)"),
    "H2": (push_beside_held_piece, "majority control: a held piece's push"),
    "H3": (push_beside_attackers_pouncer, "a push source as an attacker of a wild piece"),
    "H4": (upkeep_beside_push, "rules.purse.upkeep beside rules.push"),
    "H5": (two_displacing_hops_meet, "two displacing hops of one piece reach one square"),
    "H6": (two_pushes_one_landing, "push the piece on one landing two ways"),
    "H7": (upkeep_beside_the_duck, "rules.purse.upkeep beside a long turn"),
    "C1": (too_many_push_directions, "push directions, which the move word cannot name"),
    "C2": (too_many_sweep_rays, "sweep rays from one square on a board where two of them meet"),
    "C3": (sweep_ray_too_long_beside_folds, "sweep beside folds: more than"),
    "C4": (too_many_landing_candidates, "landing squares, and the move word holds at most"),
}
