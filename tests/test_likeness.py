"""How alike the games of a set are, weighed by the set (`spec/likeness.py`).

Every spec here is made in this file: what is under test is what the DSL's words mean and
how the library's own rarity weighs them, not where any game happens to sit.
`docs/plan-map.md` (M1) holds the readings on the real library.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.likeness import (
    IDENTITY,
    PIECE,
    likeness,
    words_of,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "chess_variant_sandbox" / "spec" / "likeness.py"

# A small game: a royal stepper, a promoting walker, a tower. Everything below is a copy of
# this with something changed.
BASE: dict = {
    "schema_version": 3,
    "name": "base",
    "board": {"width": 6, "height": 6},
    "pieces": [
        {
            "id": "K",
            "name": "chief",
            "royal": True,
            "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        },
        {
            "id": "W",
            "name": "walker",
            "value": 1,
            "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1], [-1, 1]], "mode": "capture"},
            ],
        },
        {
            "id": "R",
            "name": "tower",
            "value": 5,
            "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}],
        },
    ],
    "setup": {
        "mirror": True,
        "placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "f1"]},
            {"piece": "W", "squares": ["a2", "b2", "c2", "d2", "e2", "f2"]},
        ],
    },
    "rules": {"promotion": {"piece": "W", "rank": 5, "to": ["R"]}},
}


def made(*, width: int = 6, rules: dict | None = None, extra: dict | None = None) -> dict:
    """`BASE` on a board this wide, with these rules added and this piece added."""
    spec = copy.deepcopy(BASE)
    spec["board"]["width"] = width
    spec["rules"].update(rules or {})
    if extra is not None:
        spec["pieces"].append(extra)
        spec["setup"]["placements"].append({"piece": extra["id"], "squares": ["b1"]})
    return spec


def renamed(spec: dict, ids: dict[str, str]) -> dict:
    """The same game with its pieces' ids changed everywhere these specs name one."""
    out = copy.deepcopy(spec)
    for piece in out["pieces"]:
        piece["id"] = ids.get(piece["id"], piece["id"])
        if "captures" in piece:
            piece["captures"] = [ids.get(other, other) for other in piece["captures"]]
    for placement in out["setup"]["placements"]:
        placement["piece"] = ids.get(placement["piece"], placement["piece"])
    promotion = out["rules"]["promotion"]
    promotion["piece"] = ids.get(promotion["piece"], promotion["piece"])
    promotion["to"] = [ids.get(other, other) for other in promotion["to"]]
    return out


# Pieces that move in ways nothing else here does, so each adds its own words.
CAMEL = {
    "id": "C",
    "name": "camel",
    "moves": [{"kind": "leap", "vectors": [[1, 3]], "symmetry": "dihedral8"}],
}
ZEBRA = {
    "id": "Z",
    "name": "zebra",
    "moves": [{"kind": "leap", "vectors": [[2, 3]], "symmetry": "dihedral8"}],
}
GIRAFFE = {
    "id": "G",
    "name": "giraffe",
    "moves": [{"kind": "leap", "vectors": [[1, 4]], "symmetry": "dihedral8"}],
}


# A game on another lattice with one piece and one rule, none of them anything `BASE` has.
STRANGER: dict = {
    "schema_version": 18,
    "name": "stranger",
    "board": {"width": 5, "height": 5, "lattice": "hex"},
    "pieces": [
        {
            "id": "H",
            "name": "hopper",
            "royal": True,
            "moves": [{"kind": "leap", "vectors": [[2, 1]], "symmetry": "rotate6"}],
        },
    ],
    "setup": {
        "mirror": False,
        "orientation": "rotate",
        "placements": [
            {"piece": "H", "player": 0, "squares": ["a1"]},
            {"piece": "H", "player": 1, "squares": ["e5"]},
        ],
    },
    "rules": {"win": "royal_capture"},
}


def test_the_made_specs_are_valid() -> None:
    for spec in (
        BASE,
        made(extra=CAMEL),
        made(extra={**CAMEL, "captures": ["W", "C"]}),
        made(rules={"stalemate": "loss"}),
        STRANGER,
    ):
        assert validate_spec(spec) == []


# --------------------------------------------------------------- what the numbers are


def test_a_game_is_its_own_nearest_and_the_relation_is_symmetric() -> None:
    specs = [BASE, made(extra=CAMEL), made(width=7), made(rules={"stalemate": "loss"})]
    relation = likeness(specs)
    for i in range(len(specs)):
        assert relation.distance(i, i) == 0.0
        assert relation.similarity(i, i) == 1.0
        for j in range(len(specs)):
            assert relation.distance(i, j) == relation.distance(j, i)
            assert 0.0 <= relation.distance(i, j) <= 1.0
    matrix = relation.matrix()
    assert all(matrix[i][i] == 0.0 for i in range(len(specs)))
    assert all(matrix[i][j] == matrix[j][i] for i in range(len(specs)) for j in range(i))


def test_the_weights_are_the_inverse_document_frequency_over_the_set_given() -> None:
    specs = [BASE, made(extra=CAMEL), made(extra=CAMEL, width=7), made(width=7)]
    relation = likeness(specs)
    camel = {word for word in relation.words[1] if word not in relation.words[0]}
    assert camel, "the camel adds words"
    for word in camel & relation.words[2]:
        assert relation.weights[word] == math.log(4 / 2)
    # What every game has weighs nothing.
    everywhere = frozenset.intersection(*relation.words)
    assert everywhere and all(relation.weights[word] == 0.0 for word in everywhere)


def test_a_rule_every_game_has_moves_no_distance() -> None:
    specs = [BASE, made(extra=CAMEL), made(width=7), made(extra=ZEBRA, width=8)]
    everywhere = [made_with(spec, {"stalemate": "loss", "max_plies": 300}) for spec in specs]
    assert likeness(specs).matrix() == likeness(everywhere).matrix()


def made_with(spec: dict, rules: dict) -> dict:
    out = copy.deepcopy(spec)
    out["rules"].update(rules)
    return out


def test_two_games_differing_in_a_rare_mechanic_are_further_apart_than_in_a_common_one() -> None:
    # The camel is in one game; the zebra in four. Each pair differs only in that piece.
    specs = [
        BASE,
        made(extra=CAMEL),
        made(extra=ZEBRA),
        made(extra=ZEBRA, width=7),
        made(extra=ZEBRA, width=8),
        made(extra=ZEBRA, rules={"stalemate": "loss"}),
    ]
    relation = likeness(specs)
    assert relation.distance(0, 1) > relation.distance(0, 2)


def test_two_games_sharing_a_rare_rule_are_nearer_than_two_sharing_a_common_one() -> None:
    rare = {"bare_royal": "loss"}
    common = {"stalemate": "loss"}
    specs = [
        made(width=7, rules=rare),
        made(width=8, rules=rare),
        made(width=7, rules=common),
        made(width=8, rules=common),
        made(width=5, rules=common),
        made(width=9, rules=common),
    ]
    relation = likeness(specs)
    assert relation.distance(0, 1) < relation.distance(2, 3)


# --------------------------------------------------------------- what is not a difference


def test_a_renamed_piece_id_moves_nothing() -> None:
    spec = made(extra={**CAMEL, "captures": ["W", "C"]})
    other = renamed(spec, {"K": "Q", "W": "P", "R": "T", "C": "X"})
    assert validate_spec(other) == []
    assert words_of(spec) == words_of(other)
    # And a setting names a piece by its position in the schema, not its letter.
    assert f"set:promotion[].piece={PIECE}" in words_of(spec)


def test_a_piece_name_or_price_is_not_part_of_what_it_is() -> None:
    spec = made(extra=CAMEL)
    other = copy.deepcopy(spec)
    for piece in other["pieces"]:
        piece["name"] = piece["name"] + " the second"
        piece["value"] = 9
    assert words_of(spec) == words_of(other)


def test_whom_a_piece_may_take_is_part_of_what_it_is() -> None:
    free = made(extra=CAMEL)
    choosy = made(extra={**CAMEL, "captures": ["W"]})
    other_target = made(extra={**CAMEL, "captures": ["R"]})
    identities = [
        {word for word in words_of(spec) if word.startswith(IDENTITY)}
        for spec in (free, choosy, other_target)
    ]
    assert identities[0] != identities[1] != identities[2] != identities[0]
    # Read by what the target is, not its id: the same target under another letter is one.
    assert words_of(choosy) == words_of(renamed(choosy, {"W": "P"}))


def test_a_spelling_the_normal_form_folds_moves_nothing() -> None:
    spec = made(extra=CAMEL)
    other = copy.deepcopy(spec)
    other["pieces"].reverse()
    other["setup"]["placements"].reverse()
    tower = next(piece for piece in other["pieces"] if piece["id"] == "R")
    tower["moves"] = [{"kind": "slide", "vectors": [[1, 0], [0, 1], [-1, 0], [0, -1]]}]
    assert words_of(spec) == words_of(other)


# ------------------------------------------------------------- what the set does to it


def test_the_order_of_the_library_changes_no_number() -> None:
    specs = [
        BASE,
        made(extra=CAMEL),
        made(extra=ZEBRA, width=7),
        made(extra=GIRAFFE, rules={"stalemate": "loss"}),
        made(width=8, rules={"bare_royal": "loss"}),
        made(extra=ZEBRA),
    ]
    order = list(range(len(specs)))
    random.Random(7).shuffle(order)
    first, second = likeness(specs), likeness([specs[i] for i in order])
    at = {i: position for position, i in enumerate(order)}
    for i in range(len(specs)):
        for j in range(len(specs)):
            assert first.distance(i, j) == second.distance(at[i], at[j])
        assert [order[k] for k in second.nearest(at[i], 3)] == first.nearest(i, 3)
    assert first.weights == second.weights


def test_an_unrelated_game_moves_the_others_only_through_the_weights() -> None:
    """A game sharing no word with two others moves their likeness by re-weighing alone,
    and by a bounded amount: every word of theirs gains the same `log((N+1)/N)`, so their
    weighted likeness moves toward their unweighted Jaccard and never past it."""
    specs = [
        BASE,
        made(extra=CAMEL),
        made(extra=ZEBRA, width=7),
        made(rules={"stalemate": "loss"}),
    ]
    before, after = likeness(specs), likeness([*specs, STRANGER])
    for i in range(len(specs)):
        assert before.words[i] == after.words[i]
        assert not before.words[i] & after.words[len(specs)], "the stranger shares a word"
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            one, other = before.words[i], before.words[j]
            plain = len(one & other) / len(one | other)
            old, new = before.similarity(i, j), after.similarity(i, j)
            assert min(old, plain) - 1e-12 <= new <= max(old, plain) + 1e-12
            # Exactly the weighted Jaccard of the same words under the new weights.
            shared = math.fsum(after.weights[word] for word in one & other)
            union = math.fsum(after.weights[word] for word in one | other)
            assert math.isclose(new, shared / union, rel_tol=1e-12)


def test_a_set_of_one_or_none_is_a_relation_too() -> None:
    assert len(likeness([])) == 0
    alone = likeness([BASE])
    assert alone.distance(0, 0) == 0.0 and alone.nearest(0, 5) == []
    assert all(weight == 0.0 for weight in alone.weights.values())


def test_shared_lists_the_rarest_word_first() -> None:
    specs = [made(extra=CAMEL), made(extra=CAMEL, width=7), made(width=8), made(width=9)]
    relation = likeness(specs)
    shared = relation.shared([0, 1])
    weights = [relation.weights[word] for word in shared]
    assert weights == sorted(weights, reverse=True)
    camel = {word for word in relation.words[0] - relation.words[2] if word.startswith(IDENTITY)}
    assert camel and camel <= set(shared)
    assert all(relation.weights[word] == weights[0] for word in camel)


def test_the_words_do_not_depend_on_the_hash_seed() -> None:
    spec = json.dumps(made(extra={**CAMEL, "captures": ["W", "R"]}))
    script = (
        "import json, sys; from chess_variant_sandbox.spec.likeness import words_of; "
        "print(json.dumps(sorted(words_of(json.loads(sys.stdin.read())))))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            input=spec,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outputs) == 1


# ------------------------------------------------------------------ what it may read


def test_the_module_reads_only_the_spec() -> None:
    """The relation is the DSL's meaning of alike: it imports the standard library and
    `spec/` alone -- no runs, no engine, no web -- so nothing downstream recomputes it."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            if top == "chess_variant_sandbox":
                assert node.module.startswith("chess_variant_sandbox.spec."), node.module
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("chess_variant_sandbox") for alias in node.names)


def test_no_game_name_and_no_number_read_off_a_library() -> None:
    """N6: no constant names a game or carries a value read off one library. The only
    numbers in the module are 0 and 1, and no ruleset's name is a word of its text."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }
    assert numbers <= {0, 1}, numbers
    text = " ".join(
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    names = set()
    for folder in ("rulesets", "archive"):
        for path in (ROOT / folder).glob("*.json"):
            names.add(path.stem.lower())
            names.add(str(json.loads(path.read_text(encoding="utf-8")).get("name", "")).lower())
    named = sorted(name for name in names if name and re.search(rf"\b{re.escape(name)}\b", text))
    assert named == []
