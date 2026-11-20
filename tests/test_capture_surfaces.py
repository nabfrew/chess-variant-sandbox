"""R-L4b stage 4 (`docs/reference/capture-effects-design.md` §6 stage 4): the surfaces read the list.

Every surface that states, prices, compares, edits or translates a capture rule reads it from
`spec/capture.py` (`capture_list`, through `rules_as_old_keys` for a kind's rule), and says what
only ``rules.capture`` can say from the list itself. The proof that no stored game moved: each
stored file and the same file written as the list read alike on every surface, and only a list
the old keys cannot say gains its own sentence, its own charge and its own distance.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.loop.grid import descriptor
from chess_variant_sandbox.loop.operators import OPERATORS
from chess_variant_sandbox.spec import capture, complexity
from chess_variant_sandbox.spec.capture import (
    as_list_spelling,
    as_played_old_keys,
    list_features,
    rules_as_old_keys,
    says_more_than_old_keys,
)
from chess_variant_sandbox.spec.complexity import measure_complexity
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.distance import features
from chess_variant_sandbox.spec.reachability import _type_changes
from chess_variant_sandbox.spec.relate import identity_key, relate
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.viz.rules_payload import explosion_payload
from chess_variant_sandbox.web.cards import card_model

ROOT = Path(__file__).resolve().parent.parent
STORED = sorted(
    str(path.relative_to(ROOT)).replace("\\", "/")
    for folder in ("rulesets", "archive", "unlike") for path in (ROOT / folder).glob("*.json")
)


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _said(spec: dict[str, Any]) -> list[str]:
    try:
        translate(spec)
    except Untranslatable as refused:
        return list(refused.reasons)
    return []


def _ordered() -> dict[str, Any]:
    """Hierarchy-fall's demoting strike ahead of a capture hand: a list only the key says."""
    spec = _load("archive/hierarchy-fall.json")
    strike = spec["rules"].pop("strike")
    spec["rules"]["drops"] = {}
    spec["rules"]["capture"] = [{"do": "strike", **strike}, {"do": "hand"}]
    return spec


def _default(spec: dict[str, Any]) -> dict[str, Any]:
    """The same rules in the usual order, as the old keys."""
    return capture.default_order(spec)


def _chain() -> dict[str, Any]:
    """Atomic's blast reading a blast's victims too: a chain of blasts."""
    spec = _load("rulesets/atomic.json")
    blast = spec["rules"].pop("explode_on_capture")
    spec["rules"]["capture"] = [{"do": "blast", **blast, "on": ["landing", "blast"]}]
    return spec


class TestEveryStoredGameReadsAlikeInEitherSpelling:
    """Each stored file with a capture rule, and the file written as ``rules.capture``, are
    one reading on every surface: the rules text, the complexity counts, the distance's
    features, the relation, the grid cell, the reachability walk and Fairy-Stockfish's
    verdict. So no stored game's text or counts moved, and the list is read, not skipped."""

    @pytest.mark.parametrize("path", STORED)
    def test_every_surface_reads_the_list_spelling_as_the_file(self, path: str) -> None:
        spec = _load(path)
        listed = as_list_spelling(spec)
        if listed is None:
            return
        assert describe_spec(listed) == describe_spec(spec), path
        # `spec_bytes` is the spelling's own length, read by no metric; every count is the game's.
        assert replace(measure_complexity(listed), spec_bytes=0) == replace(
            measure_complexity(spec), spec_bytes=0
        ), path
        assert features(listed) == features(spec), path
        assert identity_key(listed) == identity_key(spec), path
        assert descriptor(listed) == descriptor(spec), path
        rules = spec.get("rules") or {}
        assert _type_changes(spec.get("pieces", []), (listed.get("rules") or {}), set())[3] == (
            _type_changes(spec.get("pieces", []), rules, set())[3]
        ), path
        assert bool(_said(listed)) == bool(_said(spec)), path

    def test_no_stored_game_writes_a_list_only_the_key_says(self) -> None:
        assert not [path for path in STORED if says_more_than_old_keys(_load(path))]


class TestAListOnlyTheKeySays:
    def test_it_is_valid_and_says_more(self) -> None:
        assert validate_spec(_ordered()) == [] and validate_spec(_chain()) == []
        assert list_features(_ordered()) == {"order"}
        assert list_features(_chain()) == {"on"}

    def test_the_rules_text_states_the_list_in_order(self) -> None:
        text = describe_spec(_ordered())
        line = next(line for line in text.splitlines() if "What a capture does, in this order" in line)
        assert line.index("(1) the strike") < line.index("(2) the taken piece goes to the taker's hand")
        assert "What a capture does" not in describe_spec(_default(_ordered()))
        # The strike's own sentences are read from the list, as the old keys' were.
        assert "In the chain" in text

    def test_complexity_charges_each_kind_what_its_old_key_cost_and_the_list_one_sentence(self) -> None:
        listed, old = measure_complexity(_ordered()), measure_complexity(_default(_ordered()))
        assert listed.special_rules == pytest.approx(
            old.special_rules + complexity._CAPTURE_LIST_SENTENCE
        )
        assert listed.move_descriptors == old.move_descriptors

    def test_the_distance_and_the_relation_see_the_order(self) -> None:
        ordered, default = _ordered(), _default(_ordered())
        assert features(ordered)["capture_list"] == 1.0 and features(default)["capture_list"] == 0.0
        assert features(ordered)["has_drops"] == features(default)["has_drops"]
        assert identity_key(ordered) != identity_key(default)
        assert "capture" in {d.key for d in relate(ordered, default).differences if hasattr(d, "key")} or (
            relate(ordered, default).differences
        )

    def test_reachability_reads_the_hand_entry(self) -> None:
        spec = _ordered()
        assert _type_changes(spec["pieces"], spec["rules"], set())[3] is True
        spec["rules"]["capture"] = [spec["rules"]["capture"][0]]
        assert _type_changes(spec["pieces"], spec["rules"], set())[3] is False

    def test_the_core_plays_it(self) -> None:
        for spec in (_ordered(), _chain()):
            engine = Engine(spec)
            state = engine.initial_state()
            rng = random.Random(1)
            for _ in range(30):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state) is not None:
                    break
                state = engine.apply(state, rng.choice(moves))


class TestTheTablesAgree:
    """`complexity.py` reads the spelling itself (it is a fingerprinted metrics source and
    `spec/capture.py` is not), so its copies of the list's tables are held to the reader's."""

    def test_complexity_s_tables_are_capture_s(self) -> None:
        assert complexity._CAPTURE_ORDER == capture.CAPTURE_KINDS
        assert complexity._CAPTURE_DEFAULT_ON == capture.CAPTURE_DEFAULT_ON
        assert complexity._CAPTURE_LIST_KEYS == capture.CAPTURE_LIST_KEYS | capture.CAPTURE_BLAST_KEYS

    @pytest.mark.parametrize("make", [_ordered, _chain])
    def test_complexity_s_reading_of_says_more_is_capture_s(self, make: Any) -> None:
        spec = make()
        assert complexity._capture_list_says_more(spec["rules"]["capture"]) is says_more_than_old_keys(spec)


class TestFairyStockfish:
    def test_a_list_apart_only_in_a_cause_the_game_never_makes_is_re_expressed(self) -> None:
        spec = _load("rulesets/crazyhouse.json")
        drops = spec["rules"]["drops"]
        drops.pop("from_captures", None)
        spec["rules"]["capture"] = [{"do": "hand", "on": ["landing", "collision"]}]
        assert validate_spec(spec) == []
        assert says_more_than_old_keys(spec) and as_played_old_keys(spec) is not None
        assert list_features(spec) == set()
        assert _said(spec) == _said(_load("rulesets/crazyhouse.json"))

    def test_a_list_only_the_key_says_is_refused_by_name(self) -> None:
        assert any(reason.startswith("rules.capture:") for reason in _said(_chain()))
        assert any("ordered list" in reason for reason in _said(_ordered()))


class TestThePage:
    def test_the_designer_has_the_capture_list_s_card(self) -> None:
        field = next(f for f in card_model()["rules"] if f["key"] == "capture")
        assert field["control"] == "list" and field["ordered"] is True
        keys = field["item"]["byKind"]["keys"]
        assert set(keys) == set(capture.CAPTURE_KINDS)
        assert "chain" in keys["strike"] and "radius" in keys["blast"] and "links" in keys["contacts"]
        on = next(f for f in field["item"]["fields"] if f["key"] == "on")
        assert on["control"] == "list" and on["item"]["control"] == "menu"

    def test_the_board_reads_a_blast_that_spares_the_piece_taken_from_the_list(self) -> None:
        spec = _load("archive/circe.json")
        strike = spec["rules"].pop("strike")
        spec["rules"]["capture"] = [
            {"do": "blast", "radius": 1, "immune": ["K"], "spares_taken": True, "on": ["landing"]},
            {"do": "strike", **strike},
        ]
        if validate_spec(spec):
            pytest.skip(f"the arm is refused: {validate_spec(spec)}")
        payload = explosion_payload(Engine(spec).spec)
        assert payload is not None and payload["strikes"]
        assert payload["order"] == ["blast", "strike"]


class TestTheLoopReachesIt:
    """The second edit `_CANNOT_ENTER`'s two rows name (`tests/test_operators.py`): from a game
    with two capture rules `order-the-capture` puts them in another order, and from a game with
    an entry reading two causes it makes a kind twice."""

    _order = next(op for op in OPERATORS if op.name == "order-the-capture")

    def _reaches(self, parent: dict[str, Any], name: str) -> bool:
        for seed in range(40):
            child = copy.deepcopy(parent)
            if self._order.apply(child, random.Random(seed)) is None:
                continue
            if validate_spec(child) == [] and name in list_features(child):
                return True
        return False

    def test_an_order(self) -> None:
        assert self._reaches(_default(_ordered()), "order")

    def test_a_kind_twice(self) -> None:
        assert self._reaches(_chain(), "twice")

    def test_every_operator_on_a_list_game_keeps_the_list_or_declines(self) -> None:
        parent = _ordered()
        for operator in OPERATORS:
            child = copy.deepcopy(parent)
            if operator.apply(child, random.Random(3)) is not None:
                assert validate_spec(child) == [], operator.name

    def test_the_projection_is_the_list_s_rules(self) -> None:
        view = rules_as_old_keys(_ordered())
        assert "capture" not in view["rules"] and view["rules"]["strike"]["effect"] == "demote"
        assert view["rules"]["drops"].get("from_captures") is True


def test_the_converter_keeps_the_hand_a_hand_entry_files_into() -> None:
    """Beside a written list, ``drops: {}`` is the hand a hand entry files into, not a hand
    stated off; dropped, the designer's load refused the draft it had just been given."""
    from chess_variant_sandbox.spec.spelling import current_spelling

    spec = _ordered()
    assert current_spelling(spec)["rules"]["drops"] == {}
    assert validate_spec(current_spelling(spec)) == []
