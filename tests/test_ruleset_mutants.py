"""`metrics/mutants.py`: the corpus of deliberately perturbed rulesets.

What each transform builds, pinned on the committed bases it is built from (read-only):
the board a resize leaves, the range a widened royal reaches, the rank a moved promotion
lands on, the declarations a flattened army keeps, the wanderer's own piece, and what
`idle_wanderer` refuses. `tests/test_index_quality.py::TestMutantCorpus` holds the corpus
properties (every mutant validates, differs from its base, carries a label).
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.metrics import mutants
from chess_variant_sandbox.spec import load_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _bases() -> dict[str, dict[str, Any]]:
    loaded = [load_spec(RULESETS / f"{name}.json") for name in ("chess", "gardner", "shogi")]
    return {spec["name"]: spec for spec in loaded}


def _built() -> dict[str, dict[str, Any]]:
    return {mutant.name: spec for mutant, spec in mutants.build_mutants(_bases())}


def _royal_moves(spec: dict[str, Any]) -> list[dict[str, Any]]:
    (royal,) = [piece for piece in spec["pieces"] if piece.get("royal")]
    return royal["moves"]


class TestTheCorpusParameters:
    def test_each_parametrised_transform_builds_what_its_rationale_says(self) -> None:
        """The numbers in the corpus are the claims its rationales make: a 16x16 board, a
        board four ranks deep, two files wider, promotion on the sixth rank, a royal
        sliding six (four on the five-square base), and only royal and most numerous kept."""
        built = _built()
        assert built["mutant-vast-board"]["board"] == {"width": 16, "height": 16}
        assert built["mutant-cramped-board"]["board"] == {"width": 8, "height": 4}
        assert built["probe-wide-board"]["board"] == {"width": 10, "height": 8}
        assert built["probe-midboard-promotion"]["rules"]["promotion"]["rank"] == 5
        assert [move["range"] for move in _royal_moves(built["probe-running-king"])] == [6]
        assert [
            move["range"] for move in _royal_moves(built["probe-gardner-running-king"])
        ] == [4]
        assert [piece["id"] for piece in built["mutant-royal-and-pawns"]["pieces"]] == [
            "P", "K",
        ]

    def test_flattening_drops_the_declarations_nothing_places(self) -> None:
        """Shogi's promoted types are declared and never placed; once promotion goes with
        the flattening they are unreachable, and they go too (the royal stays)."""
        built = _built()
        assert [piece["id"] for piece in built["mutant-shogi-uniform"]["pieces"]] == [
            "P", "L", "N", "S", "G", "B", "R", "K",
        ]

    def test_a_mutant_is_immutable(self) -> None:
        """`MUTANTS` is one module-level tuple every caller shares; a caller that could
        rebind a mutant's transform or label would change it for all the others."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            mutants.MUTANTS[0].expectation = "same"  # type: ignore[misc]


class TestBuildMutants:
    def test_only_keeps_the_labels_asked_for_in_corpus_order(self) -> None:
        """The first mutant is a `worse`, so the filter must skip it and go on."""
        names = [mutant.name for mutant, _ in mutants.build_mutants(_bases(), only=("same",))]
        assert names == [
            "null-renamed", "null-reordered-pieces", "null-reordered-gardner",
            "null-renamed-gardner", "null-idle-wanderer",
        ]


class TestIdleWanderer:
    def test_the_wanderer_steps_one_square_up_or_down_never_captures_and_is_worth_nothing(
        self,
    ) -> None:
        spec = _built()["null-idle-wanderer"]
        assert spec["pieces"][-1] == {
            "id": "W",
            "name": "wanderer",
            "value": 0,
            "moves": [{"kind": "leap", "vectors": [[0, 1], [0, -1]], "mode": "move"}],
        }
        assert spec["setup"]["placements"][-1] == {"piece": "W", "squares": ["a1"]}

    @pytest.mark.parametrize(
        "key",
        ["reach", "obstacles", "scoring", "territory", "npcs", "drops", "wrap",
         "promote_on_entry"],
    )
    def test_a_rule_that_names_squares_is_refused(self, key: str) -> None:
        spec = copy.deepcopy(_bases()["chess"])
        spec.setdefault("rules", {})[key] = {}
        with pytest.raises(ValueError, match=rf"^idle_wanderer cannot shift rules\.{key}$"):
            mutants.idle_wanderer(spec)

    def test_a_move_condition_region_is_refused(self) -> None:
        spec = copy.deepcopy(_bases()["chess"])
        spec["pieces"][0]["moves"].append(
            {"kind": "leap", "vectors": [[0, 1]], "mode": "move",
             "condition": {"region": ["a1"]}}
        )
        with pytest.raises(ValueError, match="cannot shift a move condition's region"):
            mutants.idle_wanderer(spec)
